"""吸收/欠反应跟随影子检测器测试（absorption_follow_v1 族）。

不触网络/真实 DB：session/gate 全用替身。本文件锁定「影子阶段的生命线」——
冻结口径逐位保真（.pytest_tmp/absorption_realprice.py 真实价复核同源，规则冻结勿动）：

  1. _pct_sorted   nearest-rank 分位（p80[1..10]=8.0，非 np.percentile 线性插值 8.2）
  2. _extract      real 基特征（up_move 用真实 up_price 残差）+ 有效性过滤（复刻 realprice.features）
  3. _calibrate    滚动标定 k/b（np.polyfit）+ 位移门 p50 + 欠反应门 p80；样本不足/零方差 → None
  4. _process_window  严格 ex-ante（calib 只取 cutoff≤s<本窗 start）+ 双 variant 独立缓冲 +
                      方向标签（btc涨押UP/跌押DOWN）+ ev 口径 + 幂等 + shadow_gate 拦截 +
                      NOISE 早退 + 归档后处理直接落 SETTLED（无 PENDING）
  5. 预热/轮询     _warmup 只填缓冲不落表；_poll_once 推进水位
  6. 物理隔离      version 不进 X4_VERSIONS/LIVE_CHANNELS/RETIRED_VERSIONS；settings 默认开启
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.absorption_shadow_detector as asd
from binance_predict.services.absorption_shadow_detector import (
    ABSORPTION_SPECS,
    CALIB_BUFFER_MS,
    DISP_PCT,
    FEE_RET,
    MIN_CALIB,
    UNDER_PCT,
    AbsorptionShadowDetector,
    _at,
    _at_pt,
    _calibrate,
    _ev_at_entry,
    _extract,
    _first,
    _outcome_of,
    _pct_sorted,
    _ser,
)

TD120 = "absorption_follow_td120_v1"
TD150 = "absorption_follow_td150_v1"
START = 1_700_000_000_000 // 300_000 * 300_000   # 对齐 5m 整点


# ============================================================
# 替身（不触真实 DB）：_FakeResult 比 combo 多一个 first()（本检测器 dup 查询用）
# ============================================================

class _FakeResult:
    def __init__(self, rows: list | None = None, scalar=None, first=None) -> None:
        self._rows, self._scalar, self._first = rows or [], scalar, first

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar

    def first(self):
        return self._first


class _FakeSession:
    def __init__(self, rows: list | None = None, scalar=None, first=None) -> None:
        self.rows, self.scalar, self.first_val = rows or [], scalar, first
        self.added: list = []
        self.committed = 0
        self.executed = 0

    async def execute(self, _stmt) -> _FakeResult:
        self.executed += 1
        return _FakeResult(self.rows, self.scalar, self.first_val)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed += 1

    async def rollback(self) -> None:
        pass


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


# ============================================================
# 构造器：标定缓冲点 + 合成 SentimentWindow（real 基曲线）
# ============================================================

def _bm_series(n: int = 600) -> list[float]:
    """确定性 btc_move 序列：|bm| 均匀铺在 0.16~8.0bp，正负交替（零均值、有方差）。"""
    out: list[float] = []
    for i in range(n):
        bm = 8.0 * (0.02 + (i % 50) / 50.0)
        out.append(-bm if i % 2 else bm)
    return out


def _seed(d: AbsorptionShadowDetector, version: str, start_ms: int,
          k: float = 2.0, n: int = 600) -> list[float]:
    """填充标定缓冲：n 个严格在 start_ms 之前、14d 内的窗，up_move=k·btc_move（完美共线）。

    → _calibrate 应恢复 k/b≈(k,0)、位移门=p50(|bm|)、欠反应门≈0（残差恒 0）。
    """
    bms = _bm_series(n)
    buf = d._buffers[version]
    buf.clear()
    step_ms = 60_000                                    # 1min 间隔 → n=600 跨 10h ≪ 14d
    for i, bm in enumerate(bms):
        s = start_ms - (n - i) * step_ms                # 升序、全 < start_ms、全 ≥ cutoff
        buf.append((s, bm, k * bm))
    return bms


def _mk_window(start_ms: int, *, btc_move_bp: float, up_move_pp: float, outcome: str,
               bo: float = 100000.0, up_pr_o: float = 0.50, dn_pr_o: float = 0.48,
               dn_pr_d: float = 0.52, up_pct_o: float = 50.0, dn_pct_o: float = 48.0,
               dn_pct_d: float = 52.0, actual_return: float | None = None,
               offsets: tuple[int, ...] = (120_000, 150_000)) -> SimpleNamespace:
    """合成已归档 SentimentWindow：决策点铺在 offsets（默认 120s/150s，两 variant 均可读）。

    bd/up_pr_d 由 btc_move_bp/up_move_pp 反解（与 _extract 正向公式互逆）；
    up_pct_d 随 up_move_pp 联动（q_pct>0 有效性）。actual_return 默认非零（可判定）。
    """
    bd = bo * (1.0 + btc_move_bp / 1e4)
    up_pr_d = up_pr_o + up_move_pp / 100.0
    up_pct_d = up_pct_o + up_move_pp
    if actual_return is None:
        actual_return = btc_move_bp / 1e4 or 1e-4       # 非零、符号随 btc_move

    def curve(o: float, dv: float) -> list[dict]:
        pts = [{"t": start_ms, "v": o}]
        for off in offsets:
            pts.append({"t": start_ms + off, "v": dv})
        return pts

    return SimpleNamespace(
        start_time=start_ms, end_time=start_ms + 300_000,
        entry_price=bo, actual_return=actual_return, outcome=outcome,
        curve_btc_price=curve(bo, bd),
        curve_up_price=curve(up_pr_o, up_pr_d),
        curve_down_price=curve(dn_pr_o, dn_pr_d),
        curve_up_pct=curve(up_pct_o, up_pct_d),
        curve_down_pct=curve(dn_pct_o, dn_pct_d),
        curve_trade_volume=curve(1000.0, 2000.0),
        curve_participants=curve(10.0, 20.0),
    )


# ============================================================
# 1. 冻结统计算子口径保真
# ============================================================

def test_pct_sorted_nearest_rank_not_percentile() -> None:
    """口径红线：p80[1..10]=8.0（nearest-rank round(p/100·(n−1)) 索引），
    区别于 np.percentile 线性插值（会给 8.2）。位移门/欠反应门全靠它，漂移即口径失真。"""
    vals = [float(i) for i in range(1, 11)]             # 1..10
    assert _pct_sorted(vals, 80) == 8.0
    assert np.percentile(vals, 80) == pytest.approx(8.2)  # 反例：证明二者不同
    assert _pct_sorted(vals, 50) == 5.0                 # round(0.5·9)=round(4.5)=4 → v[4]=5
    assert _pct_sorted(vals, 90) == 9.0                 # round(0.9·9)=round(8.1)=8 → v[8]=9
    assert _pct_sorted(vals, 0) == 1.0
    assert _pct_sorted(vals, 100) == 10.0


def test_pct_sorted_edge_cases() -> None:
    """空集 → nan；单元素 → 该值；乱序输入内部先排序。"""
    assert math.isnan(_pct_sorted([], 50))
    assert _pct_sorted([3.5], 80) == 3.5
    assert _pct_sorted([10.0, 1.0, 5.0], 50) == 5.0     # 排序后 [1,5,10]，round(1)=1 → 5


def test_ev_at_entry_fee_two_pct_no_premium() -> None:
    """EV 口径（与 quote_edge._ev_at_entry 同源）：赢 0.98/q−1 / 输 −1。"""
    assert FEE_RET == 0.98
    assert _ev_at_entry(True, 0.48) == pytest.approx(0.98 / 0.48 - 1.0)   # ≈ +1.0417
    assert _ev_at_entry(True, 0.52) == pytest.approx(0.98 / 0.52 - 1.0)   # ≈ +0.8846
    assert _ev_at_entry(False, 0.48) == -1.0
    assert _ev_at_entry(False, 0.99) == -1.0


def test_outcome_of_judgeable_only() -> None:
    """可判定结算：actual_return None/0 → None；outcome 非 UP/DOWN → None。"""
    assert _outcome_of(SimpleNamespace(actual_return=None, outcome="UP")) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.0, outcome="UP")) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.001, outcome="NOISE")) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.001, outcome=None)) is None
    assert _outcome_of(SimpleNamespace(actual_return=0.001, outcome="UP")) == "UP"
    assert _outcome_of(SimpleNamespace(actual_return=-0.001, outcome="DOWN")) == "DOWN"


# ============================================================
# 2. 曲线算子：清洗排序 + 严格 ex-ante TD 采样
# ============================================================

def test_ser_cleans_and_sorts_ascending() -> None:
    """_ser 剔除 t/v 为空的点并按 int(t) 升序（防御 JSONB 乱序，同 quote_edge 口径）。"""
    raw = [{"t": 300, "v": 3.0}, {"t": 100, "v": 1.0}, {"t": None, "v": 9.0},
           {"t": 200, "v": None}, {"t": 200, "v": 2.0}]
    out = _ser(raw)
    assert [p["t"] for p in out] == [100, 200, 300]
    assert [p["v"] for p in out] == [1.0, 2.0, 3.0]
    assert _ser(None) == []


def test_first_is_earliest_sample() -> None:
    assert _first([{"t": 100, "v": 7.0}, {"t": 200, "v": 8.0}]) == 7.0
    assert _first([]) is None


def test_at_strictly_ex_ante_last_within_td() -> None:
    """_at 取 ≤TD 的最后一点，越界即 break（复刻 _at_or_before，严格 ex-ante 不偷看未来）。"""
    pts = [{"t": START, "v": 100.0}, {"t": START + 120_000, "v": 110.0},
           {"t": START + 150_000, "v": 120.0}, {"t": START + 200_000, "v": 130.0}]
    assert _at(pts, START, 120.0) == 110.0              # 含 120s 点，排除 150/200
    assert _at(pts, START, 150.0) == 120.0              # 含 150s 点
    assert _at(pts, START, 130.0) == 110.0              # 130s：最后 ≤130 的是 120s 点
    assert _at(pts, START, 0.0) == 100.0                # 仅开盘点
    assert _at([], START, 120.0) is None


def test_at_pt_returns_value_and_ts() -> None:
    """_at_pt 同 _at，但一并返回采样时刻（入场报价快照审计用）。"""
    pts = [{"t": START, "v": 100.0}, {"t": START + 120_000, "v": 110.0},
           {"t": START + 150_000, "v": 120.0}]
    assert _at_pt(pts, START, 120.0) == (110.0, START + 120_000)
    assert _at_pt(pts, START, 150.0) == (120.0, START + 150_000)
    assert _at_pt([], START, 120.0) is None


# ============================================================
# 3. 特征抽取：real 基 + 有效性过滤（复刻 realprice.features）
# ============================================================

def test_extract_real_base_up_direction() -> None:
    """btc涨 → bet=UP；btc_move(bp)/up_move(pp real 基)/q_real(真实 up_price)/entry_ts 正确。"""
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    ext = _extract(w, 120.0)
    assert ext is not None
    assert ext["btc_move"] == pytest.approx(9.0)                 # (100090−100000)/100000×1e4
    assert ext["up_move"] == pytest.approx(-2.0)                 # (0.48−0.50)×100（real 基）
    assert ext["bet"] == "UP"
    assert ext["q_real"] == pytest.approx(0.48)                  # 押 UP → 真实 up_price@TD
    assert ext["up_pr_d"] == pytest.approx(0.48)
    assert ext["dn_pr_d"] == pytest.approx(0.52)
    assert ext["entry_ts"] == START + 120_000                    # ≤TD 最晚采样点
    assert ext["start"] == START


def test_extract_down_direction_uses_down_price() -> None:
    """btc跌 → bet=DOWN；q_real 取真实 down_price@TD、entry_ts 取 down 曲线时刻。"""
    w = _mk_window(START, btc_move_bp=-9.0, up_move_pp=-2.0, outcome="DOWN")
    ext = _extract(w, 150.0)
    assert ext is not None and ext["bet"] == "DOWN"
    assert ext["q_real"] == pytest.approx(0.52)                  # 押 DOWN → down_price@TD
    assert ext["entry_ts"] == START + 150_000


def test_extract_soft_volume_dimensions() -> None:
    """Δvol/Δpar 为 soft 记录维度（@TD−open），随特征一并抽取但不参与门判定。"""
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    ext = _extract(w, 120.0)
    assert ext["dvol"] == pytest.approx(1000.0)                  # 2000−1000
    assert ext["dpar"] == pytest.approx(10.0)                    # 20−10


def test_extract_zero_btc_move_returns_none() -> None:
    """btc_move==0（bd==bo）→ None（复刻 realprice 有效性过滤，退化窗不入样本集）。"""
    w = _mk_window(START, btc_move_bp=0.0, up_move_pp=-2.0, outcome="UP")
    assert _extract(w, 120.0) is None


def test_extract_missing_pct_returns_none() -> None:
    """缺 up_pct 曲线（空）→ up_pct_o/up_pct_d 均 None → 有效性过滤返回 None
    （pct/price 齐全才入样本集；注意单开盘点会被 _at 当作 ≤TD 的最后一点读到，故用空曲线）。"""
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    w.curve_up_pct = []
    assert _extract(w, 120.0) is None


def test_extract_missing_price_returns_none() -> None:
    """缺 up_price 曲线（空）→ up_pr_o/_at_pt 均 None → real 基入场价不可算 → None。"""
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    w.curve_up_price = []
    assert _extract(w, 120.0) is None


def test_extract_quote_out_of_range_returns_none() -> None:
    """q_real 越界（≥1 或 ≤0）或 q_pct≤0 → None（无效报价不入样本集）。"""
    # up_price@TD ≥ 1（bet=UP → q_real≥1 非法）
    w_hi = _mk_window(START, btc_move_bp=9.0, up_move_pp=60.0, outcome="UP")  # up_pr_d=1.10
    assert _extract(w_hi, 120.0) is None
    # down_pct@TD = 0（bet=DOWN → q_pct≤0 非法）
    w_zero = _mk_window(START, btc_move_bp=-9.0, up_move_pp=-2.0, outcome="DOWN", dn_pct_d=0.0)
    assert _extract(w_zero, 120.0) is None


def test_extract_no_btc_open_returns_none() -> None:
    """开盘 BTC 基准缺失（entry_price≤0 且 curve_btc_price 空）→ None。"""
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    w.entry_price = None
    w.curve_btc_price = []
    assert _extract(w, 120.0) is None


# ============================================================
# 4. 滚动标定：np.polyfit k/b + 位移门 p50 + 欠反应门 p80
# ============================================================

def test_calibrate_recovers_k_b_and_gates() -> None:
    """完美共线标定集（up=k·btc）→ 恢复 k/b、位移门=p50(|bm|)、欠反应门≈0（残差恒 0）。"""
    bms = _bm_series(600)
    calib = [(bm, 2.0 * bm) for bm in bms]
    fitted = _calibrate(calib)
    assert fitted is not None
    k, b, disp_gate, under_gate = fitted
    assert k == pytest.approx(2.0, abs=1e-9)
    assert b == pytest.approx(0.0, abs=1e-6)
    assert disp_gate == pytest.approx(_pct_sorted([abs(x) for x in bms], DISP_PCT))
    assert abs(under_gate) < 1e-6                       # 全在线上 → under 恒≈0 → p80≈0


def test_calibrate_insufficient_samples_none() -> None:
    """样本 < MIN_CALIB → None（冷启动保护，保守不落表）。"""
    assert MIN_CALIB == 500
    calib = [(bm, 2.0 * bm) for bm in _bm_series(MIN_CALIB - 1)]
    assert _calibrate(calib) is None


def test_calibrate_zero_variance_none() -> None:
    """btc_move 零方差（全同值）→ polyfit 退化 → None。"""
    calib = [(5.0, 10.0)] * 600
    assert _calibrate(calib) is None


def test_calibrate_gates_match_pct_sorted_of_unders() -> None:
    """口径保真：位移门=p50(|bm|) 全集、欠反应门=p80(under) 仅「过位移门子集」，
    且 under 用拟合出的 k/b 现算——逐项与同口径重算值逐位一致（含粘滞点使分布右偏）。"""
    bms = _bm_series(600)
    calib: list[tuple[float, float]] = []
    for i, bm in enumerate(bms):
        um = 2.0 * bm - (math.copysign(6.0, bm) if i % 5 == 0 else 0.0)  # 20% 点粘滞
        calib.append((bm, um))
    fitted = _calibrate(calib)
    assert fitted is not None
    k, b, disp_gate, under_gate = fitted
    assert disp_gate == pytest.approx(_pct_sorted([abs(bm) for bm, _ in calib], DISP_PCT))
    unders = [-math.copysign(1.0, bm) * (um - (k * bm + b))
              for bm, um in calib if abs(bm) >= disp_gate]
    assert under_gate == pytest.approx(_pct_sorted(unders, UNDER_PCT))
    assert len(unders) < len(calib)                     # 位移门确实筛掉了小位移子集


# ============================================================
# 5. DB 编排：触发落 SETTLED / 双 variant / 门 / 幂等 / gate / ex-ante
# ============================================================

@pytest.mark.asyncio
async def test_process_window_triggers_settled_up(monkeypatch) -> None:
    """强欠反应窗（btc涨9bp、up 反跌2pp）→ 落 SETTLED：方向 UP、win、ev、标定快照、real 入场价。"""
    session = _FakeSession(first=None)                  # 无 dup → 落行
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)                              # 600 点标定缓冲（k=2）
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    await d._process_window(w)
    assert len(session.added) == 1 and session.committed == 1
    row = session.added[0]
    assert row.version == TD120 and row.status == "SETTLED"
    assert row.td == 120 and row.direction == "UP"
    assert row.window_start == START and row.window_end == START + 300_000
    assert row.btc_move_bp == pytest.approx(9.0) and row.up_move_pp == pytest.approx(-2.0)
    assert row.k == pytest.approx(2.0, abs=1e-9) and row.calib_n == 600
    assert row.under_pp >= row.under_gate               # 触发即越过欠反应门
    assert abs(row.btc_move_bp) >= row.disp_gate        # 且越过的位移门
    assert row.win is True and row.settle_outcome == "UP"
    assert row.ev_at_entry == pytest.approx(0.98 / 0.48 - 1.0)
    assert row.entry_up_price == pytest.approx(0.48) and row.entry_quote_kind == "real"
    assert row.entry_quote_ts == START + 120_000
    assert d._trigger_count == 1


@pytest.mark.asyncio
async def test_process_window_both_variants_distinct_td(monkeypatch) -> None:
    """双 variant 各维护独立缓冲 → 同窗落两行，td/entry_quote_ts 分别对应 120/150。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    _seed(d, TD150, START)
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    await d._process_window(w)
    assert len(session.added) == 2 and session.committed == 2
    by_ver = {r.version: r for r in session.added}
    assert by_ver[TD120].td == 120 and by_ver[TD120].entry_quote_ts == START + 120_000
    assert by_ver[TD150].td == 150 and by_ver[TD150].entry_quote_ts == START + 150_000
    assert d._trigger_count == 2


@pytest.mark.asyncio
async def test_process_window_direction_down_win(monkeypatch) -> None:
    """btc跌 → 押 DOWN；窗结算 DOWN → win=True、ev 按 down_price 计。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    # btc跌9bp，up 仅跌2pp（正常应跌~18pp）→ 强粘滞 → under 大 → 触发 DOWN
    w = _mk_window(START, btc_move_bp=-9.0, up_move_pp=-2.0, outcome="DOWN")
    await d._process_window(w)
    assert len(session.added) == 1
    row = session.added[0]
    assert row.direction == "DOWN" and row.win is True and row.settle_outcome == "DOWN"
    assert row.ev_at_entry == pytest.approx(0.98 / 0.52 - 1.0)


@pytest.mark.asyncio
async def test_process_window_lose_ev_minus_one(monkeypatch) -> None:
    """follow 押错（btc涨押UP但窗结算DOWN）→ win=False、ev=−1。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="DOWN")  # bet=UP，outcome=DOWN
    await d._process_window(w)
    assert len(session.added) == 1
    assert session.added[0].win is False and session.added[0].ev_at_entry == -1.0


@pytest.mark.asyncio
async def test_process_window_skips_displacement_gate(monkeypatch) -> None:
    """位移门未过（|btc_move|<p50）→ 不落表（无位移谈不上吸收）。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    bms = _seed(d, TD120, START)
    disp = _pct_sorted([abs(x) for x in bms], DISP_PCT)
    w = _mk_window(START, btc_move_bp=min(1.0, disp / 2), up_move_pp=0.2, outcome="UP")
    await d._process_window(w)
    assert session.added == []


@pytest.mark.asyncio
async def test_process_window_skips_underreaction_gate(monkeypatch) -> None:
    """过位移门但「过度反应」（up 涨幅远超 k·btc → under<0<门）→ 非吸收，不落表。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    # btc涨9bp（过位移门），up 暴涨25pp（远超 k·9=18）→ resid=+7 → under=−7 <门(≈0) → skip
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=25.0, outcome="UP")
    await d._process_window(w)
    assert session.added == []


@pytest.mark.asyncio
async def test_process_window_skips_noise_outcome(monkeypatch) -> None:
    """结算不可判（NOISE / actual_return=0）→ 早退，连 session 都不开、不落表。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="NOISE", actual_return=0.0)
    await d._process_window(w)
    assert session.added == [] and session.executed == 0


@pytest.mark.asyncio
async def test_process_window_idempotent(monkeypatch) -> None:
    """(version, window_start) 已存在（dup 查询命中）→ 不重复落行。"""
    session = _FakeSession(first=123)                   # dup.first() 命中 → 幂等跳过
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    await d._process_window(w)
    assert session.added == [] and d._trigger_count == 0


@pytest.mark.asyncio
async def test_process_window_blocked_when_gate_offline(monkeypatch) -> None:
    """shadow_gate 下线该 version → 落表前拦截（不查 dup、不落库），另一 variant 不受影响。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(asd.shadow_gate, "_overrides", {TD120: False})  # 只下线 TD120
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START)
    _seed(d, TD150, START)
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    await d._process_window(w)
    versions = [r.version for r in session.added]
    assert TD120 not in versions, "下线版本不得落库"
    assert TD150 in versions, "version 级隔离：另一 variant 照常采集"


@pytest.mark.asyncio
async def test_process_window_calib_strictly_ex_ante(monkeypatch) -> None:
    """标定严格 ex-ante：calib 只含 cutoff≤s<本窗 start 的缓冲窗——
    太老（<cutoff）被逐出、s≥start（含本窗自身）不计入 calib_n。"""
    session = _FakeSession(first=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    _seed(d, TD120, START, n=600)                       # 600 个 in-range 前置窗
    buf = d._buffers[TD120]
    buf.appendleft((START - CALIB_BUFFER_MS - 1, 5.0, 10.0))   # 太老 → 应被 popleft 逐出
    buf.append((START, 5.0, 10.0))                      # s==start → 排除（非严格过去）
    buf.append((START + 1, 5.0, 10.0))                  # s>start → 排除（未来窗）
    w = _mk_window(START, btc_move_bp=9.0, up_move_pp=-2.0, outcome="UP")
    await d._process_window(w)
    assert len(session.added) == 1
    assert session.added[0].calib_n == 600, "calib_n 应只计 600 个 in-range 前置窗"


# ============================================================
# 6. 缓冲吸收 / 预热 / 轮询
# ============================================================

def test_absorb_into_buffers_both_variants() -> None:
    """_absorb_into_buffers：有效窗 → 两 variant 缓冲各 +1；NOISE → 跳过。"""
    d = AbsorptionShadowDetector()
    d._absorb_into_buffers(_mk_window(START, btc_move_bp=5.0, up_move_pp=1.0, outcome="UP"))
    assert len(d._buffers[TD120]) == 1 and len(d._buffers[TD150]) == 1
    d._absorb_into_buffers(
        _mk_window(START + 300_000, btc_move_bp=5.0, up_move_pp=1.0,
                   outcome="NOISE", actual_return=0.0))
    assert len(d._buffers[TD120]) == 1, "NOISE 窗不入缓冲"


@pytest.mark.asyncio
async def test_warmup_fills_buffers_without_writing(monkeypatch) -> None:
    """预热只填标定缓冲、绝不落表（预热窗自身无完整 14d 前置，落表违背冻结口径）；水位推进到最新。"""
    wins = [
        _mk_window(START + i * 300_000, btc_move_bp=5.0, up_move_pp=1.0, outcome="UP")
        for i in range(3)
    ]
    max_end = max(int(w.end_time) for w in wins)
    session = _FakeSession(rows=wins, scalar=max_end)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    await d._warmup()
    assert session.added == [], "预热不得落表"
    assert len(d._buffers[TD120]) == 3 and len(d._buffers[TD150]) == 3
    assert d._last_window_end == max_end                # 水位=最新 → 主循环只处理新归档窗


@pytest.mark.asyncio
async def test_warmup_no_archived_windows(monkeypatch) -> None:
    """无归档窗（max(end_time) is None）→ 跳过预热、缓冲空（MIN_CALIB 保护，循环内自愈）。"""
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    await d._warmup()                                   # 不应抛异常
    assert len(d._buffers[TD120]) == 0 and d._last_window_end is None


@pytest.mark.asyncio
async def test_poll_once_advances_watermark(monkeypatch) -> None:
    """轮询新归档窗 → 逐窗处理后水位推进到最大 end_time（缓冲不足则不落表但水位仍进）。"""
    wins = [
        _mk_window(START + i * 300_000, btc_move_bp=5.0, up_move_pp=1.0, outcome="UP")
        for i in range(2)
    ]
    session = _FakeSession(rows=wins)
    monkeypatch.setattr(asd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = AbsorptionShadowDetector()
    await d._poll_once()
    assert d._last_window_end == max(int(w.end_time) for w in wins)
    assert session.added == []                          # 缓冲空 → calib<MIN_CALIB → 不落表


def test_status_shape() -> None:
    """status() 供状态 API：running/last_window_end/trigger_count/双 variant 缓冲计数。"""
    d = AbsorptionShadowDetector()
    st = d.status()
    assert st["running"] is False and st["trigger_count"] == 0
    assert set(st["calib_buffer"]) == {TD120, TD150}


# ============================================================
# 7. 物理隔离 + 开关 + spec 自洽 + 注册
# ============================================================

def test_versions_isolated_from_trading_path() -> None:
    """影子纪律：version 不进 X4 下单白名单 / 不注册实盘通道 / 非退役（默认在线采集）。"""
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    from binance_predict.services.shadow_version_gate import RETIRED_VERSIONS
    for v in ABSORPTION_SPECS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"
        assert v not in RETIRED_VERSIONS, f"{v} 非退役版本（应默认在线采集）"


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号一致（record-only 零资金风险，口径保真测试全绿）。"""
    from binance_predict.config.settings import settings
    assert settings.absorption_shadow_enabled is True


def test_specs_self_consistent() -> None:
    """spec 自洽：双 variant 版本名/TD 合法、不超 DB 列宽（String(32)），冻结常数正确。"""
    assert set(ABSORPTION_SPECS) == {TD120, TD150}
    for v, td in ABSORPTION_SPECS.items():
        assert td in (120, 150), f"{v} TD 只能是冻结的 120/150"
        assert len(v) <= 32, f"{v} 超出 DB 列宽 String(32)"
    assert FEE_RET == 0.98 and DISP_PCT == 50 and UNDER_PCT == 80
    assert asd.CALIB_BUFFER_DAYS == 14                  # 用户冻结：trailing ~14 天


def test_shadow_bench_registered() -> None:
    """两 version 已登记进 main.SHADOW_BENCH（analytics 面板 + toggle 白名单的唯一事实源）。

    基准取 real 基（与生产信号基一致）：TD120 RECENT real 命中0.798/EV+0.052（CI 跨 0，
    单看不显著、FULL 显著）；TD150 RECENT real 命中0.885/EV+0.105（CI✓，双 variant 最稳健）。
    """
    import binance_predict.main as m
    assert TD120 in m.SHADOW_BENCH and TD150 in m.SHADOW_BENCH
    assert m.SHADOW_BENCH[TD120][0] == 0.798 and m.SHADOW_BENCH[TD120][1] == 0.052
    assert m.SHADOW_BENCH[TD150][0] == 0.885 and m.SHADOW_BENCH[TD150][1] == 0.105
