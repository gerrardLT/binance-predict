"""报价 edge 影子检测器单元测试。

覆盖：首个命中点查找（时点/报价区间/乱序/首中优先）、EV 公式、
UP 对照价不含未来、outcome 判定、_process_window 落表（fake session）、
v2 价格门禁（触发时点 BTC vs 窗口开盘；数据缺失/门禁未过只落 v1）。

不触网络/真实 DB：async_session_factory 用 fake session 替身。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from binance_predict.db.models import MisalignmentSignal, SentimentWindow
from binance_predict.services import quote_edge_detector as qed
from binance_predict.services.quote_edge_detector import QuoteEdgeDetector

# 纯函数测试专用传参（非生产口径；生产表为 QUOTE_EDGE_RULES，A 格 t∈[90,120)）
A_RULE = (90.0, 210.0, 0.69, 0.75)
B_RULE = (45.0, 60.0, 0.15, 0.25)    # quote_contrarian_v1


# ============================================================
# 纯函数：首个命中点
# ============================================================

def test_find_first_hit_basic() -> None:
    curve = [{"t": 30_000, "v": 0.30}, {"t": 120_000, "v": 0.71},
             {"t": 150_000, "v": 0.80}]
    hit = qed._find_first_hit(curve, 0, *A_RULE)
    assert hit == (0.71, 120_000)


def test_find_first_hit_time_window_excludes() -> None:
    """报价对但时点不对（<90s / ≥210s）→ 不命中。"""
    curve = [{"t": 60_000, "v": 0.71}, {"t": 210_000, "v": 0.72}]
    assert qed._find_first_hit(curve, 0, *A_RULE) is None


def test_find_first_hit_quote_band_excludes() -> None:
    """时点对但报价不在 [0.69,0.75) → 不命中（上界开区间）。"""
    curve = [{"t": 120_000, "v": 0.68}, {"t": 150_000, "v": 0.75}]
    assert qed._find_first_hit(curve, 0, *A_RULE) is None


def test_find_first_hit_first_in_band_wins() -> None:
    """区间外出现在前、区间内出现在后 → 取首个进入区间的点。"""
    curve = [{"t": 100_000, "v": 0.80}, {"t": 130_000, "v": 0.72}]
    assert qed._find_first_hit(curve, 0, *A_RULE) == (0.72, 130_000)


def test_find_first_hit_unordered_curve() -> None:
    """乱序曲线按 t 排序后扫描。"""
    curve = [{"t": 150_000, "v": 0.80}, {"t": 120_000, "v": 0.70}]
    assert qed._find_first_hit(curve, 0, *A_RULE) == (0.70, 120_000)


def test_find_first_hit_contra_rule() -> None:
    """B 格：t∈[45,60)s × q∈[0.15,0.25)。"""
    curve = [{"t": 30_000, "v": 0.10}, {"t": 50_000, "v": 0.20}]
    assert qed._find_first_hit(curve, 0, *B_RULE) == (0.20, 50_000)
    # t=45s 前的低报价不命中
    assert qed._find_first_hit([{"t": 44_000, "v": 0.20}], 0, *B_RULE) is None


def test_find_first_hit_relative_to_start() -> None:
    """t 相对 window_start 计（非绝对时刻）。"""
    start = 1_000_000_000
    curve = [{"t": start + 100_000, "v": 0.70}]
    assert qed._find_first_hit(curve, start, *A_RULE) == (0.70, start + 100_000)


# ============================================================
# 纯函数：EV / UP 对照价 / outcome
# ============================================================

def test_ev_at_entry() -> None:
    assert qed._ev_at_entry(True, 0.70) == pytest.approx(0.98 / 0.70 - 1.0)
    assert qed._ev_at_entry(False, 0.70) == -1.0


def test_up_price_at_or_before_excludes_future() -> None:
    curve = [{"t": 100_000, "v": 0.32}, {"t": 125_000, "v": 0.28}]
    assert qed._up_price_at_or_before(curve, 120_000) == 0.32  # 未来点 125s 不用
    assert qed._up_price_at_or_before(curve, 200_000) == 0.28
    assert qed._up_price_at_or_before(None, 100_000) is None


def test_outcome_of() -> None:
    def _w(ret, outcome="DOWN"):
        return SentimentWindow(start_time=0, end_time=1, actual_return=ret,
                               outcome=outcome)

    assert qed._outcome_of(_w(-0.001)) == "DOWN"
    assert qed._outcome_of(_w(0.001, "UP")) == "UP"
    assert qed._outcome_of(_w(None)) is None       # 归档缺结算
    assert qed._outcome_of(_w(0.0)) is None        # NOISE 平盘
    assert qed._outcome_of(_w(-0.001, "NOISE")) is None  # 非法标注


# ============================================================
# v2 价格门禁：窗口开盘基准 / 触发时点涨跌幅
# ============================================================

def test_window_open_btc_price_priority_and_fallback() -> None:
    """entry_price 优先；缺失回退 curve_btc_price 首个有效点（按 t 排序）。"""
    w = SentimentWindow(start_time=0, end_time=1, entry_price=100.0,
                        curve_btc_price=[{"t": 0, "v": 99.9}])
    assert qed._window_open_btc_price(w) == 100.0
    w2 = SentimentWindow(start_time=0, end_time=1,
                         curve_btc_price=[{"t": 5_000, "v": 99.9}, {"t": 0, "v": 100.1}])
    assert qed._window_open_btc_price(w2) == 100.1  # 乱序取最早点
    w3 = SentimentWindow(start_time=0, end_time=1)
    assert qed._window_open_btc_price(w3) is None


def test_pass_v2_price_guard_modes_and_missing() -> None:
    """min_drop（momentum：已跌 ≥0.10%）/ max_rise（contrarian：未涨 ≥0.10%）/ 缺数据 None。"""
    # momentum：跌 0.2% 过，跌 0.05%（假恐慌）拒
    assert qed._pass_v2_price_guard(
        "quote_momentum_v2",
        SimpleNamespace(entry_price=100.0, curve_btc_price=[{"t": 100_000, "v": 99.8}]),
        100_000) is True
    assert qed._pass_v2_price_guard(
        "quote_momentum_v2",
        SimpleNamespace(entry_price=100.0, curve_btc_price=[{"t": 100_000, "v": 99.95}]),
        100_000) is False
    # contrarian：涨 0.05% 过，涨 0.4%（真冲高）拒
    assert qed._pass_v2_price_guard(
        "quote_contrarian_v2",
        SimpleNamespace(entry_price=100.0, curve_btc_price=[{"t": 100_000, "v": 100.05}]),
        100_000) is True
    assert qed._pass_v2_price_guard(
        "quote_contrarian_v2",
        SimpleNamespace(entry_price=100.0, curve_btc_price=[{"t": 100_000, "v": 100.4}]),
        100_000) is False
    # 门禁数据缺失 → None（保守不落 v2）
    assert qed._pass_v2_price_guard(
        "quote_momentum_v2",
        SimpleNamespace(entry_price=None, curve_btc_price=None), 100_000) is None


def test_pass_v2_price_guard_ex_ante() -> None:
    """门禁取价不含未来：触发点之后的采样点不参与判定。"""
    w = SimpleNamespace(
        entry_price=100.0,
        curve_btc_price=[{"t": 50_000, "v": 100.0}, {"t": 200_000, "v": 99.5}],  # 200s 在触发点后
    )
    # 触发点 100s：≤100s 最晚点是 50s 的 100.0 → chg=0% → momentum（需 ≤−0.10%）拒
    assert qed._pass_v2_price_guard("quote_momentum_v2", w, 100_000) is False


# ============================================================
# _process_window 落表（fake session）
# ============================================================

class _FakeResult:
    def __init__(self, first_val):
        self._first = first_val

    def first(self):
        return self._first


class _FakeSession:
    """最小替身：dup 查重返回固定值，add/commit 全记录。"""

    def __init__(self, dup_first=None):
        self.added: list = []
        self._dup_first = dup_first

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        return _FakeResult(self._dup_first)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass


def _make_detector(monkeypatch, dup_first=None) -> tuple[QuoteEdgeDetector, _FakeSession]:
    session = _FakeSession(dup_first=dup_first)

    def _factory():
        return session

    monkeypatch.setattr(qed, "async_session_factory", _factory)
    return QuoteEdgeDetector(), session


@pytest.mark.asyncio
async def test_process_window_momentum_hit(monkeypatch) -> None:
    """A 格命中：t=100s q=0.71（[90,120) 内），窗口收 DOWN → SETTLED win=True。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=1_000_000_000, end_time=1_000_300_000,
        actual_return=-0.001, outcome="DOWN",
        curve_down_price=[{"t": 1_000_030_000, "v": 0.30},
                          {"t": 1_000_100_000, "v": 0.71},
                          {"t": 1_000_150_000, "v": 0.80}],
        curve_up_price=[{"t": 1_000_100_000, "v": 0.32},
                        {"t": 1_000_125_000, "v": 0.28}],
    )
    await d._process_window(w)
    assert len(session.added) == 1
    sig: MisalignmentSignal = session.added[0]
    assert sig.version == "quote_momentum_v1"
    assert sig.window_start == 1_000_000_000
    assert sig.target_window_start == 1_000_000_000  # 本窗即目标窗
    assert sig.end_pct == 0.71                        # 语义扩展：触发时刻报价
    assert sig.entry_down_price == 0.71
    assert sig.entry_quote_ts == 1_000_100_000
    assert sig.entry_quote_kind == "real"
    assert sig.entry_up_price == 0.32                 # ≤触发点的 UP 对照价
    assert sig.settle_outcome == "DOWN"
    assert sig.win is True
    assert sig.ev_at_entry == pytest.approx(0.98 / 0.71 - 1.0)
    assert sig.status == "SETTLED"
    assert d._trigger_count == 1


@pytest.mark.asyncio
async def test_process_window_contra_both_rules_one_window(monkeypatch) -> None:
    """同窗双命中：B 格（t=50s q=0.20）+ A 格（t=100s q=0.70），各落一条。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=0.001, outcome="UP",
        curve_down_price=[{"t": 50_000, "v": 0.20},
                          {"t": 100_000, "v": 0.70}],
    )
    await d._process_window(w)
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_contrarian_v1", "quote_momentum_v1"]
    contra = next(s for s in session.added if s.version == "quote_contrarian_v1")
    assert contra.win is False and contra.ev_at_entry == -1.0  # 收 UP，押 DOWN 输
    assert contra.end_pct == 0.20


@pytest.mark.asyncio
async def test_process_window_dup_skipped(monkeypatch) -> None:
    """查重命中（dup_first 非 None）→ 不落表。"""
    d, session = _make_detector(monkeypatch, dup_first=123)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        curve_down_price=[{"t": 100_000, "v": 0.71}],
    )
    await d._process_window(w)
    assert session.added == []
    assert d._trigger_count == 0


@pytest.mark.asyncio
async def test_process_window_noise_no_signal(monkeypatch) -> None:
    """NOISE（actual_return=0）→ 胜负不可判，不产生信号（不开 session）。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=0.0, outcome="NOISE",
        curve_down_price=[{"t": 100_000, "v": 0.71}],
    )
    await d._process_window(w)
    assert session.added == []


@pytest.mark.asyncio
async def test_process_window_no_hit_no_signal(monkeypatch) -> None:
    """报价曲线无命中 → 不落表。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        curve_down_price=[{"t": 100_000, "v": 0.50}],
    )
    await d._process_window(w)
    assert session.added == []


# ============================================================
# v2 价格门禁落表：门禁过 → v1+v2 双落；未过/缺数据 → 只 v1
# ============================================================

@pytest.mark.asyncio
async def test_v2_momentum_gate_pass_dual_insert(monkeypatch) -> None:
    """momentum 触发时点已跌 0.2%（≥0.10%）→ v1+v2 双落，字段同源。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        entry_price=100.0,
        curve_down_price=[{"t": 100_000, "v": 0.71}],
        curve_btc_price=[{"t": 0, "v": 100.0}, {"t": 100_000, "v": 99.8}],  # 触发时点已跌 0.2%
    )
    await d._process_window(w)
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_momentum_v1", "quote_momentum_v2"]
    v2 = next(s for s in session.added if s.version == "quote_momentum_v2")
    assert v2.entry_down_price == 0.71 and v2.win is True
    assert v2.ev_at_entry == pytest.approx(0.98 / 0.71 - 1.0)  # v2 同 v1 口径（无溢价）
    assert v2.status == "SETTLED"


@pytest.mark.asyncio
async def test_v2_momentum_gate_fail_v1_only(monkeypatch) -> None:
    """momentum 触发时点只跌 0.05%（假恐慌）→ v2 被拒，只落 v1。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        entry_price=100.0,
        curve_down_price=[{"t": 100_000, "v": 0.71}],
        curve_btc_price=[{"t": 0, "v": 100.0}, {"t": 100_000, "v": 99.95}],
    )
    await d._process_window(w)
    assert [s.version for s in session.added] == ["quote_momentum_v1"]


@pytest.mark.asyncio
async def test_v2_contrarian_gate_pass_and_fail(monkeypatch) -> None:
    """contrarian：触发时点未涨（平盘）过门禁；已涨 0.4% 被拒。"""
    d, session = _make_detector(monkeypatch)
    w_calm = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=0.001, outcome="UP",
        entry_price=100.0,
        curve_down_price=[{"t": 50_000, "v": 0.20}],
        curve_btc_price=[{"t": 0, "v": 100.0}, {"t": 50_000, "v": 100.05}],  # +0.05% 未真涨
    )
    await d._process_window(w_calm)
    assert [s.version for s in session.added] == ["quote_contrarian_v1", "quote_contrarian_v2"]

    d2, session2 = _make_detector(monkeypatch)
    w_melt = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=0.001, outcome="UP",
        entry_price=100.0,
        curve_down_price=[{"t": 50_000, "v": 0.20}],
        curve_btc_price=[{"t": 0, "v": 100.0}, {"t": 50_000, "v": 100.4}],  # +0.4% 真冲高
    )
    await d2._process_window(w_melt)
    assert [s.version for s in session2.added] == ["quote_contrarian_v1"]


@pytest.mark.asyncio
async def test_v2_gate_missing_data_v1_only(monkeypatch) -> None:
    """门禁数据缺失（无 entry_price/curve_btc_price）→ v2 不落，v1 照常。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        curve_down_price=[{"t": 100_000, "v": 0.71}],
    )
    await d._process_window(w)
    assert [s.version for s in session.added] == ["quote_momentum_v1"]
