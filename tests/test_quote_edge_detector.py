"""报价 edge 影子检测器单元测试。

覆盖：首个命中点查找（时点/报价区间/乱序/首中优先）、EV 公式、
UP 对照价不含未来、outcome 判定、_process_window 落表（fake session）、
v2 价格门禁（触发时点 BTC vs 窗口开盘；数据缺失/门禁未过只落 v1）、
v3 环境门禁（前窗 DOWN + 可选距日高回落；缺失/未过只落 v1/v2）。

不触网络/真实 DB：async_session_factory 用 fake session 替身。
"""
from __future__ import annotations

import operator as _op
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

    def scalar_one_or_none(self):
        return None  # v3 环境查询：前窗未归档 → 环境门禁拒（v3 不落）

    def scalars(self):
        # v3b 日高查询：无曲线 → 日高缺失（环境门禁拒）
        return SimpleNamespace(all=lambda: [])


class _FakeSession:
    """最小替身：dup 查重返回固定值，add/commit/rollback 全记录。"""

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

    async def rollback(self):
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


# ============================================================
# v3 环境门禁：前窗 DOWN（交替环境）+ 可选距日高回落≥0.30%（含边界）
# ============================================================

def _stmt_range_bounds(stmt):
    """解析 start_time 范围谓词 → (lo, hi)，钉死日高查询的无未来函数边界。"""
    lo = hi = None
    for cl in getattr(stmt.whereclause, "clauses", None) or []:
        right = getattr(cl, "right", None)
        val = getattr(right, "effective_value", None) if right is not None else None
        if val is None:
            continue
        if getattr(cl, "operator", None) is _op.ge:
            lo = val
        elif getattr(cl, "operator", None) is _op.le:
            hi = val
    return lo, hi


class _EnvSession:
    """v3 环境查询替身：按 select 列名路由，并校验日高查询的范围谓词。

    _prev_window_outcome 查 outcome 列（.scalar_one_or_none()），
    _day_high_btc 查 curve_btc_price 列（.scalars().all()）；
    expect_day_start/expect_max_start 非 None 时断言 SQL 的
    start_time ∈ [lo, hi] 与期望一致（无未来函数边界被改即测试失败）。
    """

    def __init__(self, prev_outcome=None, day_high_curves=(),
                 expect_day_start=None, expect_max_start=None):
        self._prev = prev_outcome
        self._curves = day_high_curves
        self._expect_lo = expect_day_start
        self._expect_hi = expect_max_start

    async def execute(self, stmt):
        keys = {getattr(c, "key", None) for c in stmt.selected_columns}
        if "outcome" in keys:
            return _EnvResult(scalar=self._prev)
        lo, hi = _stmt_range_bounds(stmt)
        if self._expect_lo is not None:
            assert lo == self._expect_lo, f"日高查询下界 {lo} != 期望 {self._expect_lo}"
        if self._expect_hi is not None:
            assert hi == self._expect_hi, f"日高查询上界 {hi} != 期望 {self._expect_hi}"
        return _EnvResult(scalars=list(self._curves))


class _EnvResult:
    def __init__(self, scalar=None, scalars=None):
        self._scalar = scalar
        self._scalars = scalars

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return SimpleNamespace(all=lambda: self._scalars)


def _contra_v3_window(start_ms: int = 86_400_000) -> SentimentWindow:
    """contrarian 触发窗（t=+50s q=0.20，触发时点 BTC 平盘过 v2 门禁）。"""
    return SentimentWindow(
        start_time=start_ms, end_time=start_ms + 300_000,
        actual_return=-0.001, outcome="DOWN",
        entry_price=100.0,
        curve_down_price=[{"t": start_ms + 50_000, "v": 0.20}],
        curve_btc_price=[{"t": start_ms, "v": 100.0},
                         {"t": start_ms + 50_000, "v": 100.0}],
    )


def test_pass_v3_env_guard_prev_down_only() -> None:
    """v3a：前窗 DOWN 过；前窗 UP / NOISE / 未归档 → 拒（保守不落表）。"""
    w = _contra_v3_window()
    ts = int(w.start_time) + 50_000
    assert qed._pass_v3_env_guard("quote_contrarian_v3a", w, ts, "DOWN", None) is True
    for prev in ("UP", "NOISE", None):
        assert qed._pass_v3_env_guard("quote_contrarian_v3a", w, ts, prev, None) is False


def test_pass_v3_env_guard_day_high_drawdown() -> None:
    """v3b：v3a 基础上要求距日高回落≥0.30%（含边界）；缺失 → 拒。"""
    w = _contra_v3_window()
    ts = int(w.start_time) + 50_000
    # 日高 100.50 → 触发时点 100.0 回落 −0.498% ≤ −0.30% → 过
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", 100.50) is True
    # 回落 −0.31%（过门槛）→ 过
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", 100.31) is True
    # 只回落 −0.10% → 拒；日高缺失 / 前窗非 DOWN → 拒
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", 100.10) is False
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", None) is False
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "UP", 100.50) is False


def test_pass_v3_env_guard_boundary_inclusive(monkeypatch) -> None:
    """含边界语义钉死：dd 恰等于阈值 → 过；阈值收紧一线 → 拒。"""
    w = _contra_v3_window()
    ts = int(w.start_time) + 50_000
    high = 100.50
    dd = (100.0 - high) / high * 100.0       # 实际回落 −0.4975…%
    monkeypatch.setattr(qed, "V3_DD_THRESHOLD", dd)
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", high) is True
    monkeypatch.setattr(qed, "V3_DD_THRESHOLD", dd - 1e-9)
    assert qed._pass_v3_env_guard("quote_contrarian_v3b", w, ts, "DOWN", high) is False


@pytest.mark.asyncio
async def test_day_high_btc_ex_ante_and_predicate_bounds() -> None:
    """日高只取 ≤quote_ts 的点；SQL 范围谓词钉死为 [UTC 日界, 本窗 start]。"""
    w = _contra_v3_window()
    ts = int(w.start_time) + 50_000
    day_start = (ts // 86_400_000) * 86_400_000
    curves = [[{"t": ts - 300_000, "v": 101.0}, {"t": ts + 100_000, "v": 200.0}]]
    session = _EnvSession(day_high_curves=curves,
                          expect_day_start=day_start,
                          expect_max_start=int(w.start_time))
    d = QuoteEdgeDetector()
    assert await d._day_high_btc(session, w, ts) == 101.0  # 未来点 200.0 不算


@pytest.mark.asyncio
async def test_day_high_btc_cache_incremental_and_reset() -> None:
    """增量缓存：第二次查询下界推进到水位+1、高点累计；乱序窗口触发重置。"""
    w1 = _contra_v3_window(start_ms=86_400_000)
    w2 = _contra_v3_window(start_ms=86_700_000)
    ts1 = int(w1.start_time) + 50_000
    ts2 = int(w2.start_time) + 50_000
    d = QuoteEdgeDetector()
    s1 = _EnvSession(day_high_curves=[[{"t": ts1 - 50_000, "v": 102.0}]],
                     expect_day_start=86_400_000, expect_max_start=int(w1.start_time))
    assert await d._day_high_btc(s1, w1, ts1) == 102.0
    # 第二次：下界 = 首窗 start+1（只拉增量）；新曲线 101 < 缓存 102 → 高点不变
    s2 = _EnvSession(day_high_curves=[[{"t": ts2 - 50_000, "v": 101.0}]],
                     expect_day_start=int(w1.start_time) + 1,
                     expect_max_start=int(w2.start_time))
    assert await d._day_high_btc(s2, w2, ts2) == 102.0
    # 乱序窗口（start < 水位）→ 缓存重置，下界回到 UTC 日界
    s3 = _EnvSession(day_high_curves=[[{"t": ts1 - 50_000, "v": 103.0}]],
                     expect_day_start=86_400_000, expect_max_start=int(w1.start_time))
    assert await d._day_high_btc(s3, w1, ts1) == 103.0


def _v3_gate_stub(monkeypatch, v3a, v3b) -> None:
    """把 v2 门禁固定为通过、v3 环境门禁按版本返回指定值。"""
    monkeypatch.setattr(qed, "_pass_v2_price_guard", lambda *a: True)

    def _guard(version, w, quote_ts, prev_outcome, day_high):
        return {"quote_contrarian_v3a": v3a, "quote_contrarian_v3b": v3b}[version]

    monkeypatch.setattr(qed, "_pass_v3_env_guard", _guard)


@pytest.mark.asyncio
async def test_v3_gate_pass_inserts_v1_v2_v3a_v3b(monkeypatch) -> None:
    """v2+v3a+v3b 全过 → v1/v2/v3a/v3b 四条同窗落表，v3 字段与 v2 同源。"""
    d, session = _make_detector(monkeypatch)
    _v3_gate_stub(monkeypatch, v3a=True, v3b=True)
    await d._process_window(_contra_v3_window())
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_contrarian_v1", "quote_contrarian_v2",
                        "quote_contrarian_v3a", "quote_contrarian_v3b"]
    v3b = next(s for s in session.added if s.version == "quote_contrarian_v3b")
    assert v3b.entry_down_price == 0.20 and v3b.win is True
    assert v3b.ev_at_entry == pytest.approx(0.98 / 0.20 - 1.0)  # 与 v1/v2 同口径
    assert v3b.status == "SETTLED"


@pytest.mark.asyncio
async def test_v3a_fail_v3b_fail_only_v1_v2(monkeypatch) -> None:
    """v3a 环境门禁未过 → v3a/v3b 都不落，v1/v2 不受影响（v3b⊂v3a）。"""
    d, session = _make_detector(monkeypatch)
    _v3_gate_stub(monkeypatch, v3a=False, v3b=False)
    await d._process_window(_contra_v3_window())
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_contrarian_v1", "quote_contrarian_v2"]


@pytest.mark.asyncio
async def test_v3a_pass_v3b_fail_only_v3a(monkeypatch) -> None:
    """v3a 过、v3b 距日高不足 → 只多落 v3a，v3b 拒。"""
    d, session = _make_detector(monkeypatch)
    _v3_gate_stub(monkeypatch, v3a=True, v3b=False)
    await d._process_window(_contra_v3_window())
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_contrarian_v1", "quote_contrarian_v2",
                        "quote_contrarian_v3a"]


@pytest.mark.asyncio
async def test_v3_requires_v2_price_gate(monkeypatch) -> None:
    """v2 价格门禁未过（真冲高）→ v3 也不落：v3 = v2 ∩ 环境门禁。"""
    d, session = _make_detector(monkeypatch)

    def _guard(version, w, quote_ts, prev_outcome, day_high):
        return True  # 环境门禁永远过

    monkeypatch.setattr(qed, "_pass_v3_env_guard", _guard)
    monkeypatch.setattr(qed, "_pass_v2_price_guard",
                        lambda version, w, ts: version != "quote_contrarian_v2")
    await d._process_window(_contra_v3_window())
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_contrarian_v1"]  # v2/v3a/v3b 全被价格门禁拦住


@pytest.mark.asyncio
async def test_momentum_unaffected_by_v3(monkeypatch) -> None:
    """v3 只基于 contrarian 区间：momentum 窗不产生任何 v3 记录。"""
    d, session = _make_detector(monkeypatch)
    w = SentimentWindow(
        start_time=0, end_time=300_000, actual_return=-0.001, outcome="DOWN",
        entry_price=100.0,
        curve_down_price=[{"t": 100_000, "v": 0.71}],
        curve_btc_price=[{"t": 0, "v": 100.0}, {"t": 100_000, "v": 99.8}],
    )
    await d._process_window(w)
    versions = sorted(s.version for s in session.added)
    assert versions == ["quote_momentum_v1", "quote_momentum_v2"]
