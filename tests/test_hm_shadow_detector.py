"""HM 上吊线反弹入场影子检测器测试：口径保真（硬闸门）+ 状态机 + 编排 + 物理隔离。

不触网络/真实 DB：collector/session 全用替身；口径保真测试依赖
output/klines_15m_720d.csv（缺失时 skip，CI 无产物不阻塞）。
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.hm_shadow_detector as hsd
from binance_predict.discovery.data import load_klines_csv
from binance_predict.discovery.features import atr_series
from binance_predict.services.hm_shadow_detector import (
    BAR_MS_1M,
    BAR_MS_15M,
    CLV_MAX,
    ENTRY_X,
    HmShadowDetector,
    RULE_TEXT,
    RULE_TEXT_V2,
    VERSION,
    VERSION2,
    _to_klines,
    atr_for_target,
    clv_series,
    detect_weak_hm,
    entry_decision,
    v2_gate_mask,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_15M = ROOT / "output" / "klines_15m_720d.csv"

T0 = 1_700_000_000_000 // BAR_MS_15M * BAR_MS_15M


# ============================================================
# 口径保真（硬闸门）：720d 重放触发计数 == 研究脚本冻结数
# ============================================================

@pytest.fixture(scope="module")
def kl720():
    if not CSV_15M.exists():
        pytest.skip("720d 15m K 线产物不存在（离线口径测试跳过）")
    return load_klines_csv(str(CSV_15M), BAR_MS_15M)


def test_720d_replay_trigger_count(kl720) -> None:
    """硬闸门：检测器 detect_weak_hm 全量重放计数 == 研究脚本冻结数 127。

    研究脚本（.pytest_tmp/optimal_entry.py → scripts/hm_touch_entry_research_720d.py）
    冻结口径：hm & cont & roll(cont,1)，A∈(20, n-8)，弱收盘 CLV≤0.75 → n=127。
    检测器口径只含 cont 守卫（无 roll(cont,1)/边界裁剪），720d 上计数同为 127：
    无前 20 根/末 8 根命中，且无「次根不连续」的信号根。任何阈值漂移都会对不上。
    """
    atr = atr_series(kl720)
    mask = detect_weak_hm(kl720, atr)
    idx = np.flatnonzero(mask)
    assert len(idx) == 127, f"720d 重放计数 {len(idx)} != 冻结数 127（口径漂移）"
    # 与研究口径交叉验证：加次根连续守卫与边界裁剪后计数不变
    m2 = mask & np.roll(kl720.cont, 1)
    i2 = np.flatnonzero(m2)
    i2 = i2[(i2 > 20) & (i2 < len(kl720) - 8)]
    assert len(i2) == 127


def test_720d_replay_v2_gate_count(kl720) -> None:
    """硬闸门：v1 触发 ∩ v2 门禁（非下跌段∧非低波）计数 == 冻结数 78。

    冻结口径（.pytest_tmp/hm_v2_freeze_counts.py，2026-09-01）：ret24>−1% ∧
    ATR/前 24h ATR 中位数≥0.8 → 127 触发中 78 个通过（触价 29，收跌 69.0%）。
    同时验证向量化门禁与切片脚本循环口径在真实数据上逐位一致。
    """
    atr = atr_series(kl720)
    idx = np.flatnonzero(detect_weak_hm(kl720, atr) & v2_gate_mask(kl720, atr))
    assert len(idx) == 78, f"v2 门禁后计数 {len(idx)} != 冻结数 78（口径漂移）"
    # 交叉验证：切片脚本的循环口径（中位数窗口有限值 >20）与向量化逐位一致
    n = len(kl720)
    c, W = kl720.c, hsd.V2_REGIME_WINDOW
    med_loop = np.full(n, np.nan)
    for i in range(W, n):
        w = atr[i - W:i]
        w = w[np.isfinite(w)]
        if len(w) > 20:
            med_loop[i] = np.median(w)
    ret24 = np.full(n, np.nan)
    ret24[W:] = c[W:] / c[:-W] - 1
    gate_loop = np.zeros(n, dtype=bool)
    fin = np.isfinite(ret24) & np.isfinite(med_loop) & (med_loop > 0) & np.isfinite(atr)
    gate_loop[fin] = ((ret24[fin] > hsd.V2_RET24_MIN)
                      & (atr[fin] / med_loop[fin] >= hsd.V2_ATR_RATIO_MIN))
    assert bool((gate_loop != v2_gate_mask(kl720, atr)).any()) is False


def test_short_window_fidelity(kl720) -> None:
    """硬闸门：短窗（120 根）触发与 v2 门禁判定与全量矩阵逐根一致。"""
    atr = atr_series(kl720)
    mask = detect_weak_hm(kl720, atr)
    gate = v2_gate_mask(kl720, atr)
    W = hsd.WARMUP_BARS
    rows = [{"open_time": int(kl720.t[i]), "open": float(kl720.o[i]),
             "high": float(kl720.h[i]), "low": float(kl720.l[i]),
             "close": float(kl720.c[i]), "volume": float(kl720.v[i])}
            for i in range(len(kl720) - W, len(kl720))]
    kl_short = _to_klines(rows, BAR_MS_15M)
    m_short = detect_weak_hm(kl_short, atr_series(kl_short))
    assert int((m_short != mask[-W:]).sum()) == 0, "短窗判定与全量不一致"
    # v2 门禁只对末 BACKSCAN_BARS 根保真：短窗内更早索引的 24h 中位数窗口被短窗
    # 边界截断，数学上不可比；实时/回补只评估末 12 根，末 12 根的中位数窗口恰在窗内。
    B = hsd.BACKSCAN_BARS
    g_short = v2_gate_mask(kl_short, atr_series(kl_short))
    assert int((g_short[-B:] != gate[-B:]).sum()) == 0, "末 12 根 v2 门禁与全量不一致"


def test_atr_for_target_matches_atr_series(kl720) -> None:
    """口径保真：入场障碍 ATR == atr_series 在目标根下标的值（回测 lev 同源）。"""
    atr = atr_series(kl720)
    i = len(kl720) - 1  # 以末根为「目标根」，前 20 根已收盘行构造回读
    rows = [{"open_time": int(kl720.t[j]), "open": float(kl720.o[j]),
             "high": float(kl720.h[j]), "low": float(kl720.l[j]),
             "close": float(kl720.c[j]), "volume": float(kl720.v[j])}
            for j in range(i - 20, i)]
    got = atr_for_target(rows, float(kl720.o[i]))
    assert got == pytest.approx(float(atr[i]), rel=1e-12)


# ============================================================
# 触发判定：合成 K 线边界
# ============================================================

def _hm_rows(n_base=21, o=100.0) -> list[dict]:
    """21 根平盘基底（前 20 根定 ATR）+ 1 根可调形态根。

    基底：o=c=100、range=0.04（range%=0.0004）→ 末位 ATR=0.0004×100=0.04；
    阈值：实体≤0.012 / 下影≥0.012 / 上影≤0.006 / 贴顶距≤0.03（0.75×ATR）。
    """
    rows = []
    for i in range(n_base):
        rows.append({"open_time": T0 + i * BAR_MS_15M, "open": o, "high": o + 0.02,
                     "low": o - 0.02, "close": o, "volume": 1.0})
    return rows


def _append_hm_bar(rows: list[dict], o: float, c: float, h: float, l: float) -> list[dict]:
    rows.append({"open_time": T0 + len(rows) * BAR_MS_15M, "open": o, "high": h,
                 "low": l, "close": c, "volume": 1.0})
    return rows


def _kl(rows: list[dict]):
    kl = _to_klines(rows, BAR_MS_15M)
    return kl, atr_series(kl)


def test_synthetic_hm_triggers() -> None:
    """构造满足全部冻结条件的弱收盘上吊线（各条件留浮点余量，不卡等号边界）。

    注：基底 range%=0.0004 的均值经浮点运算得 ATR=0.0399…92 < 0.04，
    故所有贴阈值条件均留 ≥1e-3 余量，避免浮点尾差误伤；
    纯等号边界的判定方向由 entry_decision 专项用例覆盖。
    """
    rows = _hm_rows()  # ATR(末根)≈0.04：实体≤0.012/下影≥0.012/上影≤0.006/贴顶距≤0.03
    # 形态根：实体=0.009（≤0.012），下影=0.021（≥2×实体=0.018 且 ≥0.012），
    # 上影=0.005（≤0.006），收盘距 20 根最高（含本根，=100.02）为 0.029 < 0.75×ATR，
    # CLV=0.021/0.035=0.6 ≤ 0.75
    _append_hm_bar(rows, o=100.0, c=99.991, h=100.005, l=99.97)
    kl, atr = _kl(rows)
    assert float(atr[-1]) == pytest.approx(0.04)
    mask = detect_weak_hm(kl, atr)
    assert bool(mask[-1]) is True, f"应触发 | atr={atr[-1]}"
    assert 0 < float(clv_series(kl)[-1]) <= CLV_MAX


def test_range_zero_no_trigger() -> None:
    """range≤0（h==l==c）：CLV 不可定义 → 冻结规则不触发。"""
    rows = _hm_rows()
    _append_hm_bar(rows, o=100.0, c=100.0, h=100.0, l=100.0)
    kl, atr = _kl(rows)
    assert bool(detect_weak_hm(kl, atr)[-1]) is False


def test_clv_above_threshold_no_trigger() -> None:
    """CLV>0.75（收盘偏高，非弱收盘）→ 不触发。"""
    rows = _hm_rows()
    # 下影极短使收盘贴近 high：CLV=(c−l)/(h−l)≈0.9
    _append_hm_bar(rows, o=100.0, c=100.009, h=100.01, l=99.9)
    kl, atr = _kl(rows)
    assert float(clv_series(kl)[-1]) > CLV_MAX
    assert bool(detect_weak_hm(kl, atr)[-1]) is False


def test_short_history_no_trigger() -> None:
    """历史不足 20 根：top20/ATR 不可得 → 保守不触发。"""
    rows = _hm_rows(n_base=10)
    _append_hm_bar(rows, o=100.0, c=99.99, h=100.005, l=99.97)
    kl, atr = _kl(rows)
    assert bool(detect_weak_hm(kl, atr)[-1]) is False


def test_gap_before_signal_no_trigger() -> None:
    """信号根与前根不连续（cont 守卫）→ 不触发（其余几何条件均满足）。"""
    rows = _hm_rows()
    bar = {"open_time": T0 + (len(rows) + 5) * BAR_MS_15M,  # 跳过 5 根 → 断档
           "open": 100.0, "high": 100.005, "low": 99.97, "close": 99.99, "volume": 1.0}
    rows.append(bar)
    kl, atr = _kl(rows)
    assert bool(kl.cont[-1]) is False
    assert bool(detect_weak_hm(kl, atr)[-1]) is False


# ============================================================
# v2 门禁：合成边界（非下跌段 ∧ 非低波）
# ============================================================

def test_v2_gate_flat_history_passes() -> None:
    """121 根平盘基底：ret24=0 > −1% ∧ ATR比=1.0 ≥ 0.8 → 通过。"""
    rows = _hm_rows(n_base=121)
    kl, atr = _kl(rows)
    assert bool(v2_gate_mask(kl, atr)[-1]) is True


def test_v2_gate_downtrend_blocked() -> None:
    """过去 24h 跌 >1%（末根对 96 根前收盘 −1.5%）→ 下跌段拦截。"""
    rows = _hm_rows(n_base=121)
    rows[24]["close"] = 101.5  # 末根下标 120；120−96=24 → ret24 ≈ −1.48%
    kl, atr = _kl(rows)
    assert bool(v2_gate_mask(kl, atr)[-1]) is False


def test_v2_gate_low_vol_blocked() -> None:
    """近段波动压缩（末 30 根 range 降为 1/10）→ ATR比 <0.8 拦截。"""
    rows = _hm_rows(n_base=121)
    for r in rows[-30:]:
        r["high"], r["low"] = r["open"] + 0.002, r["open"] - 0.002
    kl, atr = _kl(rows)
    assert bool(v2_gate_mask(kl, atr)[-1]) is False


def test_v2_gate_insufficient_history_blocked() -> None:
    """历史 ≤96 根：24h 回看不足 → 一律不通过（保守）。"""
    rows = _hm_rows(n_base=96)
    kl, atr = _kl(rows)
    assert bool(v2_gate_mask(kl, atr).any()) is False


@pytest.mark.asyncio
async def test_evaluate_new_bars_dual_rows_when_gate_passes(monkeypatch) -> None:
    """形态触发 ∩ 门禁通过 → 同信号根双行落库（v1 基准 + v2 过滤版）。"""
    rows = _hm_rows(n_base=120)
    _append_hm_bar(rows, o=100.0, c=99.991, h=100.005, l=99.97)
    session = _FakeSession(scalar=None)  # 两个版本存在性查询均未命中
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    d._spawn_watcher = lambda *a: None  # type: ignore[method-assign]
    await d._evaluate_new_bars(rows)
    assert sorted(s.version for s in session.added) == [VERSION, VERSION2]
    v2 = next(s for s in session.added if s.version == VERSION2)
    assert v2.rule_text == RULE_TEXT_V2 and v2.entry_state == "WAITING"


@pytest.mark.asyncio
async def test_evaluate_new_bars_single_row_when_gate_blocked(monkeypatch) -> None:
    """形态触发但门禁拦截（下跌段）→ 仅落 v1 行，v2 不记录。

    用下跌段而非低波做拦截源：低波压缩会连带改变 ATR，几何触发条件全按
    ATR 缩放，会破坏形态构造；改 96 根前收盘只动 ret24，几何条件不受影响。
    """
    rows = _hm_rows(n_base=120)
    rows[24]["close"] = 101.5  # 末根下标 120；120−96=24 → ret24 ≈ −1.49% ≤ −1%
    _append_hm_bar(rows, o=100.0, c=99.991, h=100.005, l=99.97)
    kl_chk, atr_chk = _kl(rows)
    assert bool(detect_weak_hm(kl_chk, atr_chk)[-1]) is True   # 前置：形态仍触发
    assert bool(v2_gate_mask(kl_chk, atr_chk)[-1]) is False    # 前置：门禁拦截
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    d._spawn_watcher = lambda *a: None  # type: ignore[method-assign]
    await d._evaluate_new_bars(rows)
    assert [s.version for s in session.added] == [VERSION]


# ============================================================
# 入场状态机：entry_decision 纯函数
# ============================================================

def test_entry_decision_touch_up() -> None:
    assert entry_decision(100.30, 100.25, 99.75, 300.0) == "TOUCHED"


def test_entry_decision_touch_at_deadline() -> None:
    """600s 死线含边界。"""
    assert entry_decision(100.30, 100.25, 99.75, 600.0) == "TOUCHED"


def test_entry_decision_late_touch() -> None:
    assert entry_decision(100.30, 100.25, 99.75, 620.0) == "ABANDON_LATE"


def test_entry_decision_lower_first() -> None:
    assert entry_decision(99.70, 100.25, 99.75, 100.0) == "ABANDON_LOWER"


def test_entry_decision_lower_at_boundary() -> None:
    """贴下障碍含边界即弃。"""
    assert entry_decision(99.75, 100.25, 99.75, 100.0) == "ABANDON_LOWER"


def test_entry_decision_wait() -> None:
    assert entry_decision(100.0, 100.25, 99.75, 100.0) == "WAIT"


def test_entry_decision_degenerate_dual_touch() -> None:
    """退化区间（up≤dn，同采样「双触」的唯一可能形态）→ 保守判下弃。"""
    assert entry_decision(100.0, 100.0, 100.0, 100.0) == "ABANDON_LOWER"
    assert entry_decision(100.0, 99.9, 100.1, 100.0) == "ABANDON_LOWER"


# ============================================================
# DB 编排：结算方向 / 幂等 / 超时（伪 session，不触真实库）
# ============================================================

class _FakeResult:
    def __init__(self, rows: list, scalar=None) -> None:
        self._rows, self._scalar = rows, scalar

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list:
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    def __init__(self, rows: list | None = None, scalar=None) -> None:
        self.rows, self.scalar = rows or [], scalar
        self.added: list = []
        self.executed: list = []
        self.committed = False

    async def execute(self, stmt) -> _FakeResult:
        self.executed.append(stmt)
        return _FakeResult(self.rows, self.scalar)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


def _pending(target_start: int, entry_state: str = "TOUCHED") -> SimpleNamespace:
    return SimpleNamespace(
        version=VERSION, signal_bar_start=target_start - BAR_MS_15M,
        target_bar_start=target_start, entry_state=entry_state, status="PENDING",
        win=None, settle_outcome=None, settle_open=None, settle_close=None,
        settled_at=None,
    )


def _detector() -> HmShadowDetector:
    return HmShadowDetector(collector=None, pm_15m_latest={})


@pytest.mark.asyncio
async def test_settle_win_on_red_target_bar(monkeypatch) -> None:
    """TOUCHED + 目标根收阴（close<open）→ SETTLED win=True（押 DOWN 命中）。"""
    target = T0 + 40 * BAR_MS_15M
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    closed = [{"open_time": target, "open": 100.0, "high": 100.8,
               "low": 99.2, "close": 99.4, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True and sig.settle_outcome == "DOWN"
    assert sig.settle_open == 100.0 and session.committed


@pytest.mark.asyncio
async def test_settle_lose_on_green_target_bar(monkeypatch) -> None:
    target = T0 + 40 * BAR_MS_15M
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    closed = [{"open_time": target, "open": 100.0, "high": 101.2,
               "low": 99.8, "close": 101.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "UP"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    """目标根平盘 → NOISE → EXPIRED（方向无法判定，不进胜率统计）。"""
    target = T0 + 40 * BAR_MS_15M
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    closed = [{"open_time": target, "open": 100.0, "high": 100.4,
               "low": 99.7, "close": 100.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_non_touched_expires_without_settlement(monkeypatch) -> None:
    """非 TOUCHED 行（如 ABANDON_LOWER）：目标根到期后仅转 EXPIRED，win 保持 NULL。"""
    target = T0 + 40 * BAR_MS_15M
    session = _FakeSession(rows=[_pending(target, entry_state="ABANDON_LOWER")])
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 98.0, "close": 98.5, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome is None


@pytest.mark.asyncio
async def test_record_signal_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)  # 存在性查询命中
    d = _detector()
    bar = {"open_time": T0, "open": 100.0, "close": 99.97}
    added = await d._record_signal(session, bar, atr_val=0.1, clv_val=0.3)
    assert added is False and session.added == []


@pytest.mark.asyncio
async def test_record_signal_inserts_frozen_fields(monkeypatch) -> None:
    """新信号落库：版本/规则原文/唯一键字段齐全，入场态 WAITING。"""
    session = _FakeSession(scalar=None)
    d = _detector()
    bar = {"open_time": T0, "open": 100.0, "close": 99.97}
    added = await d._record_signal(session, bar, atr_val=0.1, clv_val=0.3)
    assert added is True
    sig = session.added[0]
    assert sig.version == VERSION and sig.rule_text == RULE_TEXT
    assert sig.signal_bar_start == T0 and sig.target_bar_start == T0 + BAR_MS_15M
    assert sig.entry_state == "WAITING" and sig.status == "PENDING"


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    """目标根起点后 4h 仍未结算 → EXPIRED。"""
    old = int(time.time() * 1000) - 10 * 3_600_000
    session = _FakeSession(rows=[_pending(old)])
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = _detector()
    await d._expire_stale_pending()
    assert session.rows[0].status == "EXPIRED" and session.committed


# ============================================================
# 入场监控：1m 重建回放 / 报价快照（fake collector，不触网络）
# ============================================================

class _FakeStore:
    def __init__(self, mid=None, last_upd=None) -> None:
        self.mid_price = mid
        self.last_ws_spot_update = last_upd


class _FakeCollector:
    def __init__(self, k1m: list[dict] | None = None, k15: list[dict] | None = None,
                 open_price: float = 0.0) -> None:
        self.store = _FakeStore()
        self._k1m = k1m or []
        self._k15 = k15 or []
        self._open = open_price

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        return {"1m": self._k1m, "15m": self._k15}.get(interval, [])[-limit:]

    async def fetch_kline_open(self, interval: str, start_ms: int) -> float:
        return self._open


def _m1_rows(target_start: int, seq: list[tuple[float, float]], start_idx: int = 0):
    """构造 1m 棒：[(high, low), ...]，自 target_start 起连续。"""
    return [{"open_time": target_start + (start_idx + i) * BAR_MS_1M,
             "open": 100.0, "high": h, "low": l, "close": 100.0, "volume": 1.0}
            for i, (h, l) in enumerate(seq)]


@pytest.mark.asyncio
async def test_replay_1m_touch_early(monkeypatch) -> None:
    """重建：第 3 根 1m 触上障碍（未破下）→ TOUCHED（触及时刻=棒起点近似）。"""
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + 5 * BAR_MS_1M)
    d = HmShadowDetector(
        collector=_FakeCollector(k1m=_m1_rows(target, [(100.1, 99.9)] * 2
                                              + [(100.3, 100.0)] + [(100.1, 99.9)] * 2)),
        pm_15m_latest={},
    )
    state, ts, price = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "TOUCHED" and ts == target + 2 * BAR_MS_1M and price == 100.25


@pytest.mark.asyncio
async def test_replay_1m_lower_first(monkeypatch) -> None:
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + 4 * BAR_MS_1M)
    d = HmShadowDetector(
        collector=_FakeCollector(k1m=_m1_rows(target, [(100.1, 99.9)]
                                              + [(100.1, 99.7)] + [(100.4, 99.8)]
                                              + [(100.1, 99.9)])),
        pm_15m_latest={},
    )
    state, _, _ = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "ABANDON_LOWER"


@pytest.mark.asyncio
async def test_replay_1m_dual_touch_in_bar_gap(monkeypatch) -> None:
    """棒内双触（同根 1m 既破上又破下）：先后顺序不可知 → RESTART_GAP。"""
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + 3 * BAR_MS_1M)
    d = HmShadowDetector(
        collector=_FakeCollector(k1m=_m1_rows(target, [(100.1, 99.9), (100.4, 99.6),
                                                       (100.1, 99.9)])),
        pm_15m_latest={},
    )
    state, _, _ = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "RESTART_GAP"


@pytest.mark.asyncio
async def test_replay_1m_missing_bar_gap(monkeypatch) -> None:
    """1m 缺棒（重建不完整）→ RESTART_GAP 保守弃。"""
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + 5 * BAR_MS_1M)
    bars = _m1_rows(target, [(100.1, 99.9)] * 3) + _m1_rows(target, [(100.1, 99.9)], start_idx=4)
    d = HmShadowDetector(collector=_FakeCollector(k1m=bars), pm_15m_latest={})
    state, _, _ = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "RESTART_GAP"


@pytest.mark.asyncio
async def test_replay_1m_late_touch(monkeypatch) -> None:
    """重建：第 11 根 1m（≥600s）才触上 → ABANDON_LATE。"""
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + 12 * BAR_MS_1M)
    seq = [(100.1, 99.9)] * 10 + [(100.3, 100.0), (100.1, 99.9)]
    d = HmShadowDetector(collector=_FakeCollector(k1m=_m1_rows(target, seq)),
                         pm_15m_latest={})
    state, ts, _ = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "ABANDON_LATE" and ts == target + 10 * BAR_MS_1M


@pytest.mark.asyncio
async def test_replay_1m_not_touched_full_window(monkeypatch) -> None:
    """重建：整窗 15 根 1m 齐全且无触 → NOT_TOUCHED。"""
    target = T0
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: target + BAR_MS_15M + 1000)
    d = HmShadowDetector(
        collector=_FakeCollector(k1m=_m1_rows(target, [(100.1, 99.9)] * 15)),
        pm_15m_latest={},
    )
    state, _, _ = await d._replay_1m(target, target + BAR_MS_15M, 100.25, 99.75)
    assert state == "NOT_TOUCHED"


@pytest.mark.asyncio
async def test_finalize_entry_snapshots_fresh_quote(monkeypatch) -> None:
    """TOUCHED 落库时快照同周期新鲜 DOWN 报价（护栏定标数据）。"""
    target = T0 + BAR_MS_15M
    now = target + 30_000
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: now)
    session = _FakeSession()
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    pm = {"start_date": target, "down_price": 0.42, "updated_ts": now - 5_000}
    d = HmShadowDetector(collector=None, pm_15m_latest=pm)
    await d._finalize_entry(T0, "TOUCHED", touch_ts=now, touch_price=100.3)
    stmt = session.executed[-1]
    vals = stmt.compile().params
    assert vals["entry_state"] == "TOUCHED" and vals["entry_down_quote"] == 0.42
    assert session.committed


@pytest.mark.asyncio
async def test_finalize_entry_skips_stale_or_wrong_market_quote(monkeypatch) -> None:
    """报价属于旧市场（start_date 不匹配）或超龄 → 不回填（保持 NULL，不回灌）。"""
    target = T0 + BAR_MS_15M
    now = target + 30_000
    monkeypatch.setattr(hsd.clock_sync, "now_ms", lambda: now)
    session = _FakeSession()
    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    # 场景 1：市场未切换（start_date 是上一周期）
    pm = {"start_date": T0, "down_price": 0.42, "updated_ts": now - 1_000}
    d = HmShadowDetector(collector=None, pm_15m_latest=pm)
    await d._finalize_entry(T0, "TOUCHED")
    assert session.executed[-1].compile().params["entry_down_quote"] is None
    # 场景 2：报价超龄（>20s）
    pm2 = {"start_date": target, "down_price": 0.42, "updated_ts": now - 30_000}
    d2 = HmShadowDetector(collector=None, pm_15m_latest=pm2)
    await d2._finalize_entry(T0, "TOUCHED")
    assert session.executed[-1].compile().params["entry_down_quote"] is None


@pytest.mark.asyncio
async def test_resume_pending_respawns_watchers(monkeypatch) -> None:
    """重启恢复：WAITING 行重派监控任务（去重防重派）。"""
    rows = [SimpleNamespace(signal_bar_start=T0, target_bar_start=T0 + BAR_MS_15M)]

    class _ResumeSession(_FakeSession):
        async def execute(self, stmt) -> _FakeResult:
            return _FakeResult(rows)

    monkeypatch.setattr(hsd, "async_session_factory", lambda: _FakeSessionCtx(_ResumeSession()))
    d = _detector()
    spawned: list[int] = []
    d._spawn_watcher = lambda sig, tgt: spawned.append(tgt)  # type: ignore[method-assign]
    await d._resume_pending()
    assert spawned == [T0 + BAR_MS_15M]


# ============================================================
# 物理隔离：影子版本绝不进入下单路径
# ============================================================

def test_version_isolated_from_trading_path() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    for v in (VERSION, VERSION2):
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_rule_text_frozen_constants() -> None:
    """冻结参数与预注册口径一致（防手滑改常量）。"""
    assert ENTRY_X == 0.25
    assert hsd.TOUCH_DEADLINE_S == 600
    assert CLV_MAX == 0.75
    assert "0.25" in RULE_TEXT and "600s" in RULE_TEXT
    # v2 门禁冻结参数（2026-09-01 切片分析）
    assert hsd.V2_RET24_MIN == -0.01
    assert hsd.V2_ATR_RATIO_MIN == 0.8
    assert hsd.V2_REGIME_WINDOW == 96
    assert "−1.0%" in RULE_TEXT_V2 and "≥ 0.8" in RULE_TEXT_V2


def test_settings_default_on() -> None:
    """默认开启：零资金风险（只记录不下注、新表物理隔离），开关仅作紧急制动力。"""
    from binance_predict.config.settings import settings
    assert settings.hm_shadow_enabled is True
