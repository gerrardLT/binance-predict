"""BRK 假突破回归影子检测器测试：研究口径逐位保真（numpy 参照）+ 判定正反例 +
direction 结算 + 幂等 + 新鲜度派单 + 实盘注册默认 OFF + 与 CRV 同窗互斥。

不触网络/真实 DB：collector/session 全用替身。numpy 参照独立转写研究
kline_features.py 的 BREAK_UP 公式（roll_max(high,w).shift(1)，min_periods=window），
对尾部掩码逐位断言——研究口径的守门测试（实现结构与参照相互独立）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.brk_reversion_detector as brd
from binance_predict.services.brk_reversion_detector import (
    BAR_MS,
    BRK_SHADOW_SPECS,
    BRK_VERSIONS,
    BRK_WINDOWS,
    BrkReversionDetector,
    compute_brk_features,
    evaluate_brk,
    spec_mask,
    _to_klines,
)

T0 = 1_700_000_000_000 // 900_000 * 900_000


# ============================================================
# 合成 K 线构造
# ============================================================

def _mk(tf_ms: int, i: int, o: float, h: float, l: float, c: float) -> dict:
    return {"open_time": T0 + i * tf_ms, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def _flat_rows(n: int, tf_ms: int) -> list[dict]:
    """确定性平盘带：全部 high=100.1（prior 窗最大恒 100.1）。"""
    return [_mk(tf_ms, i, 99.9, 100.1, 99.8, 100.0) for i in range(n)]


def _breakout_rows(tf_ms: int, break_close: float = 103.0) -> list[dict]:
    """末根为（可选）突破阳线：close 突破前窗全部高点 100.1。"""
    rows = _flat_rows(109, tf_ms)
    o = rows[-1]["close"]
    rows.append(_mk(tf_ms, 109, o, break_close + 0.2, o - 0.2, break_close))
    return rows


def _spec(version: str) -> dict:
    return next(s for s in BRK_SHADOW_SPECS if s["version"] == version)


def _tail_hits(feat: dict, version: str, n_tail: int = 3) -> list[int]:
    return [h["idx"] for h in evaluate_brk(feat, [_spec(version)], n_tail)]


# ============================================================
# 研究口径逐位保真：numpy 参照（独立转写研究 kline_features.py 同公式）
# ============================================================

def _research_reference(rows: list[dict], window: int) -> dict:
    """研究 BREAK_UP 公式的独立 numpy 转写（滑窗实现，与检测器的循环结构不同）。"""
    h = np.array([r["high"] for r in rows], dtype=np.float64)
    c = np.array([r["close"] for r in rows], dtype=np.float64)

    def roll_max(x, w):
        out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            win = x[i - w + 1:i + 1]
            out[i] = np.nan if np.isnan(win).any() else float(win.max())
        return out

    def shift1(x):
        out = np.full(len(x), np.nan)
        out[1:] = x[:-1]
        return out

    prior_hi = shift1(roll_max(h, window))
    mask = c > prior_hi
    mask[np.isnan(prior_hi)] = False
    return {"prior_hi": prior_hi, "mask": mask}


@pytest.mark.parametrize("tf", ["15m", "5m"])
@pytest.mark.parametrize("seed", [3, 11, 42])
def test_feature_bitwise_parity_vs_research_reference(tf: str, seed: int) -> None:
    """随机游走数据上，检测器掩码与独立参照逐位一致（两个窗口均验）。"""
    tf_ms = BAR_MS[tf]
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for i in range(110):
        o = price
        c = price + rng.uniform(-0.3, 0.3)
        rows.append(_mk(tf_ms, i, o, max(o, c) + rng.uniform(0.0, 0.5),
                        min(o, c) - rng.uniform(0.0, 0.5), c))
        price = c
    kl = _to_klines(rows, tf_ms)
    feat = compute_brk_features(kl, BRK_WINDOWS[tf])
    ref = _research_reference(rows, BRK_WINDOWS[tf])
    tail = slice(BRK_WINDOWS[tf] + 1, 110)  # 窗口 warmup 之后的全部根
    assert np.allclose(feat["prior_hi"][tail], ref["prior_hi"][tail], equal_nan=True)
    assert np.array_equal(spec_mask(_spec(f"brkrv_brk8h_{tf}_v1"), feat)[tail],
                          ref["mask"][tail]), "BREAK_UP 掩码与研究参照不一致"


# ============================================================
# 判定正反例（手造 K 线，末根命中/不命中）
# ============================================================

@pytest.mark.parametrize("tf", ["15m", "5m"])
def test_breakout_positive_and_boundary_negatives(tf: str) -> None:
    tf_ms = BAR_MS[tf]
    version = f"brkrv_brk8h_{tf}_v1"
    # 正例：收盘 103 > 前窗全部高点 100.1 → 命中
    feat = compute_brk_features(_to_klines(_breakout_rows(tf_ms), tf_ms), BRK_WINDOWS[tf])
    assert 109 in _tail_hits(feat, version, n_tail=1)
    # 反例 1：收盘恰好等于前窗高点 100.1（严格 >，贴线不触发）
    feat_eq = compute_brk_features(
        _to_klines(_breakout_rows(tf_ms, break_close=100.1), tf_ms), BRK_WINDOWS[tf])
    assert _tail_hits(feat_eq, version, n_tail=1) == []
    # 反例 2：收盘低于前窗高点 → 不命中
    feat_below = compute_brk_features(
        _to_klines(_breakout_rows(tf_ms, break_close=100.05), tf_ms), BRK_WINDOWS[tf])
    assert _tail_hits(feat_below, version, n_tail=1) == []


def test_warmup_nan_never_triggers() -> None:
    """数据不足窗口（5m 需 96 根）→ prior_hi 全 NaN → 保守不触发。"""
    tf_ms = BAR_MS["5m"]
    rows = _flat_rows(50, tf_ms)
    rows.append(_mk(tf_ms, 50, 100.0, 103.2, 99.8, 103.0))
    feat = compute_brk_features(_to_klines(rows, tf_ms), BRK_WINDOWS["5m"])
    assert _tail_hits(feat, "brkrv_brk8h_5m_v1", n_tail=1) == []


# ============================================================
# DB 编排：direction 结算 / 幂等 / 新鲜度派单（伪 session，不触真实库）
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
        self.committed = False

    async def execute(self, _stmt) -> _FakeResult:
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


def _pending(version: str, direction: str, target_start: int, bar_ms: int) -> SimpleNamespace:
    return SimpleNamespace(
        version=version, direction=direction,
        signal_bar_start=target_start - bar_ms,
        target_bar_start=target_start, status="PENDING", win=None,
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("version", "bar_ms", "green", "win"), [
    ("brkrv_brk8h_15m_v1", 900_000, False, True),   # 押DOWN：次根收阴 → 赢
    ("brkrv_brk8h_15m_v1", 900_000, True, False),   # 次根收阳 → 输
    ("brkrv_brk8h_5m_v1", 300_000, False, True),
    ("brkrv_brk8h_5m_v1", 300_000, True, False),
])
async def test_settle_by_direction(monkeypatch, version, bar_ms, green, win) -> None:
    target = T0 + 60 * bar_ms
    session = _FakeSession(rows=[_pending(version, "DOWN", target, bar_ms)])
    monkeypatch.setattr(brd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    c = 101.5 if green else 99.0
    closed = [{"open_time": target, "open": 100.0, "high": 102.0, "low": 98.5,
               "close": c, "volume": 1.0}]
    tf = "15m" if bar_ms == 900_000 else "5m"
    await d._settle_pending(tf, closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is win
    assert sig.settle_outcome == ("UP" if green else "DOWN")


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    target = T0 + 60 * 300_000
    session = _FakeSession(rows=[_pending("brkrv_brk8h_5m_v1", "DOWN", target, 300_000)])
    monkeypatch.setattr(brd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending("5m", closed)
    assert session.rows[0].status == "EXPIRED" and session.rows[0].win is None


@pytest.mark.asyncio
async def test_record_signal_fields_and_idempotent(monkeypatch) -> None:
    tf_ms = BAR_MS["15m"]
    rows = _breakout_rows(tf_ms)
    kl = _to_klines(rows, tf_ms)
    feat = compute_brk_features(kl, BRK_WINDOWS["15m"])
    spec = _spec("brkrv_brk8h_15m_v1")
    idx = len(rows) - 1
    fresh = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(brd, "async_session_factory", lambda: _FakeSessionCtx(fresh))
    monkeypatch.setattr(brd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(brd.time, "time", lambda: (int(rows[idx]["open_time"]) + 900_000 + 30_000) / 1000)
    d = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    assert await d._record_signal("15m", fresh, spec, rows[idx], feat, idx, None)
    sig = fresh.added[0]
    assert sig.version == "brkrv_brk8h_15m_v1" and sig.direction == "DOWN"
    assert sig.timeframe == "15m" and sig.status == "PENDING"
    assert sig.target_bar_start == int(rows[idx]["open_time"]) + 900_000
    # 审计快照含突破幅度
    assert sig.feature_snapshot["dist_prior_hi"] == round(103.0 - 100.1, 6)
    # 幂等：已有同 (version, signal_bar_start) 行 → 不再落
    dup = _FakeSession(rows=[], scalar=123)
    assert not await d._record_signal("15m", dup, spec, rows[idx], feat, idx, None)
    assert dup.added == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tf", ["15m", "5m"])
async def test_fresh_hit_dispatches_frozen_direction(monkeypatch, tf: str) -> None:
    tf_ms = BAR_MS[tf]
    rows = _breakout_rows(tf_ms)
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(brd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(brd.shadow_gate, "is_enabled", lambda _v: True)
    d = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    d._last_evaluated_bar[tf] = int(rows[-2]["open_time"])
    fired: list[dict] = []
    d._on_live_fire = fired.append
    target = int(rows[-1]["open_time"]) + tf_ms
    monkeypatch.setattr(brd.time, "time", lambda: (target + 30_000) / 1000)
    await d._evaluate_new_bars(tf, rows)
    payload = next(p for p in fired if p["version"] == f"brkrv_brk8h_{tf}_v1")
    assert payload["market_start"] == target
    assert payload["direction"] == "DOWN"
    assert payload["market_end"] == target + tf_ms
    assert payload["signal_bar_start"] == int(rows[-1]["open_time"])


@pytest.mark.asyncio
async def test_stale_and_backscan_hits_never_dispatch(monkeypatch) -> None:
    tf_ms = BAR_MS["15m"]
    rows = _breakout_rows(tf_ms)
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(brd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(brd.shadow_gate, "is_enabled", lambda _v: True)
    target = int(rows[-1]["open_time"]) + 900_000
    d = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    fired: list[dict] = []
    d._on_live_fire = fired.append
    # 陈旧命中（目标根开盘后 10 分钟）→ 只落影子不派单
    d._last_evaluated_bar["15m"] = int(rows[-2]["open_time"])
    monkeypatch.setattr(brd.time, "time", lambda: (target + 600_000) / 1000)
    await d._evaluate_new_bars("15m", rows)
    assert fired == []
    # 冷启动回补 → 永不派单
    d2 = BrkReversionDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    d2._on_live_fire = fired.append
    monkeypatch.setattr(brd.time, "time", lambda: (target + 30_000) / 1000)
    await d2._evaluate_new_bars("15m", rows)
    assert fired == []
    assert session.added  # 影子照常落库


# ============================================================
# 实盘注册：默认 OFF / 族与护栏 / 与 CRV 突破族同窗互斥 / 影子投影
# ============================================================

def test_brkrv_live_channels_registered_default_off() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS, exclusive_group, parse_channel_config

    expected = {
        "brkrv_brk8h_15m_v1": ("15m", "DOWN", 0.54),
        "brkrv_brk8h_5m_v1": ("5m", "DOWN", 0.55),
    }
    for version, (period, direction, guard) in expected.items():
        spec = LIVE_CHANNELS[version]
        assert spec.family == "kline_reversal"
        assert spec.market_period == period and spec.direction == direction
        assert spec.auto_max_exec == guard
    # 默认全部 OFF（parse_channel_config 无覆盖时）
    configs = parse_channel_config()
    for version in expected:
        assert configs[version].enabled is False
    # 15m：并入 CRV 15m 突破族互斥组（同一根突破阳线事件子集、同押 DOWN）
    grp15 = exclusive_group("brkrv_brk8h_15m_v1")
    assert grp15 == {"crv_hi_brk20_15m_v1", "crv_brk20_15m_v1",
                     "crv_brk50_hi_15m_v1", "brkrv_brk8h_15m_v1"}
    # 5m：与 crv_brk20_5m 同窗互斥（brk8h ⊂ brk20 事件集）
    grp5 = exclusive_group("brkrv_brk8h_5m_v1")
    assert grp5 == {"crv_brk20_5m_v1", "brkrv_brk8h_5m_v1"}
    assert exclusive_group("crv_brk20_5m_v1") == grp5


def test_brkrv_versions_in_shadow_projection() -> None:
    from binance_predict.services.shadow_execution_registry import (
        SHADOW_VERSIONS, SHADOW_VERSION_SPECS)

    assert set(BRK_VERSIONS) <= set(SHADOW_VERSIONS)
    for version in BRK_VERSIONS:
        spec = SHADOW_VERSION_SPECS[version]
        assert spec.family == "kline_reversal"
        assert spec.direction_mode == "row"
        assert spec.live_channel == version


def test_settings_default_on() -> None:
    from binance_predict.config.settings import settings
    assert settings.brk_shadow_enabled is True


def test_brkrv_versions_in_shadow_bench() -> None:
    """防回归：两版本已登记进 main.SHADOW_BENCH——analytics 面板基准与
    /api/shadow/toggle 白名单的唯一事实源（CodeReview H-1：镜像 CRV 时
    曾漏注册，面板下线按钮 422、基准列空白）。"""
    from binance_predict.main import SHADOW_BENCH

    assert "brkrv_brk8h_15m_v1" in SHADOW_BENCH
    assert "brkrv_brk8h_5m_v1" in SHADOW_BENCH
    # 冻结基准胜率：15m 盲测 55.7%；5m 盲测市场结算口径 56.94%
    assert SHADOW_BENCH["brkrv_brk8h_15m_v1"][0] == 0.557
    assert SHADOW_BENCH["brkrv_brk8h_5m_v1"][0] == 0.5694
    # EV 基准留 None（研究代理口径，面板 EV 由目标窗真实报价前向现算）
    assert SHADOW_BENCH["brkrv_brk8h_15m_v1"][1] is None
    assert SHADOW_BENCH["brkrv_brk8h_5m_v1"][1] is None
