"""CRV 情绪反转影子检测器测试：研究口径逐位保真（numpy 参照）+ 判定正反例 +
双周期 direction 结算 + 幂等 + 新鲜度派单 + 实盘注册默认 OFF。

不触网络/真实 DB：collector/session 全用替身。numpy 参照独立转写研究
features.py 的公式（trailing 滚动、prior shift、Wilder RSI、实体 streak），
对尾部掩码逐位断言——研究口径的守门测试（运行时依赖无 pandas，参照用
sliding_window_view 实现，与检测器的循环滚动结构独立）。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.crv_shadow_detector as csd
from binance_predict.services.crv_shadow_detector import (
    BAR_MS,
    CRV_SHADOW_SPECS,
    CRV_VERSIONS,
    CrvShadowDetector,
    compute_crv_features,
    evaluate_crv,
    spec_mask,
    _to_klines,
)

T0 = 1_700_000_000_000 // 900_000 * 900_000


# ============================================================
# 合成 K 线构造
# ============================================================

def _mk(tf_ms: int, i: int, o: float, h: float, l: float, c: float) -> dict:
    return {"open_time": T0 + i * tf_ms, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def _quiet_rows(n: int, seed: int = 3) -> list[dict]:
    """平静随机游走（100 附近小幅摆动，无突破）。"""
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for i in range(n):
        o = price
        c = price + rng.uniform(-0.3, 0.3)
        rows.append(_mk(900_000, i, o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        price = c
    return rows


def _breakout_rows(spike_high: float | None = None, break_close: float | None = 103.0) -> list[dict]:
    """末根为（可选）突破阳线；spike_high 在 i=60 插入高影线压低 pos100。"""
    rows = _quiet_rows(129)
    if spike_high is not None:
        r = rows[60]
        rows[60] = {**r, "high": spike_high}
    o = rows[-1]["close"]
    if break_close is not None:
        rows.append(_mk(900_000, 129, o, break_close + 0.2, o - 0.2, break_close))
    else:
        # 无突破版：收盘仍在平静带内
        rows.append(_mk(900_000, 129, o, max(o, 100.5) + 0.1, o - 0.3, 100.4))
    return rows


def _spec(version: str) -> dict:
    return next(s for s in CRV_SHADOW_SPECS if s["version"] == version)


def _tail_hits(feat: dict, version: str, n_tail: int = 3) -> list[int]:
    return [h["idx"] for h in evaluate_crv(feat, [_spec(version)], n_tail)]


# ============================================================
# 研究口径逐位保真：numpy 参照（独立转写研究 features.py 同公式）
# ============================================================

def _research_reference(rows: list[dict], tf_ms: int) -> dict:
    """研究 features.py 公式的独立 numpy 转写（滑窗视图实现，与检测器结构不同）。"""
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64)
           for k in ("open", "high", "low", "close")}
    o, h, l, c = arr["open"], arr["high"], arr["low"], arr["close"]

    def roll(fn, x, w):
        out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            win = x[i - w + 1:i + 1]
            out[i] = np.nan if np.isnan(win).any() else float(fn(win))
        return out

    def shift1(x):
        out = np.full(len(x), np.nan)
        out[1:] = x[:-1]
        return out

    feat: dict[str, np.ndarray] = {}
    feat["up"] = c > o
    feat["dn"] = c < o
    feat["body"] = np.abs(c - o)
    feat["prior_hi20"] = shift1(roll(np.max, h, 20))
    feat["prior_lo20"] = shift1(roll(np.min, l, 20))
    feat["prior_hi50"] = shift1(roll(np.max, h, 50))
    lo100, hi100 = roll(np.min, l, 100), roll(np.max, h, 100)
    feat["pos100"] = (c - lo100) / (hi100 - lo100 + 1e-12)
    # z20：(close − roll_mean(c,20)) / roll_std(c,20,ddof=1).clip(1e-12)
    def roll_std(x, w):
        out = np.full(len(x), np.nan)
        for i in range(w - 1, len(x)):
            win = x[i - w + 1:i + 1]
            if not np.isnan(win).any():
                d = win - win.mean()
                out[i] = float(np.sqrt((d * d).sum() / (w - 1)))
        return out
    feat["z20"] = (c - roll(np.mean, c, 20)) / np.maximum(roll_std(c, 20), 1e-12)
    pc = shift1(c)
    tr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - pc[i]), abs(l[i] - pc[i]))
    feat["atr"] = roll(np.mean, tr, 14)
    # RSI14 Wilder（ewm alpha=1/14 adjust=False 的等价递推；首根 delta 缺失跳过）
    ag = al = np.nan
    rsi = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        gain, loss = max(d, 0.0), max(-d, 0.0)
        if np.isnan(ag):
            ag, al = gain, loss
        else:
            ag = ag + (gain - ag) / 14.0
            al = al + (loss - al) / 14.0
        rsi[i] = 100.0 - 100.0 / (1.0 + ag / max(al, 1e-12))
    feat["rsi"] = rsi
    # 实体 streak（研究循环：首根恒 0，同向 +1 / 换向重起 1 / 零实体 0）
    s = np.sign(c - o)
    streak = np.zeros(len(s), dtype=np.int64)
    for i in range(1, len(s)):
        if s[i] != 0 and s[i] == s[i - 1]:
            streak[i] = streak[i - 1] + 1
        elif s[i] != 0:
            streak[i] = 1
    feat["streak"] = streak
    feat["close"] = c
    with np.errstate(invalid="ignore"):
        feat["near_lo20"] = np.abs(c - feat["prior_lo20"]) < 0.25 * feat["atr"]
    return feat


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_feature_bitwise_parity_vs_research_reference(seed: int) -> None:
    rows = _quiet_rows(130, seed=seed)
    kl = _to_klines(rows, 900_000)
    feat = compute_crv_features(kl)
    ref = _research_reference(rows, 900_000)
    tail = slice(105, 130)  # 全部 warmup（pos100=100/atr=14/滚动20/50）之后
    for key in ("up", "dn", "prior_hi20", "prior_lo20", "prior_hi50", "pos100",
                "z20", "atr", "streak"):
        mine, ref_v = feat[key][tail], ref[key][tail]
        assert np.array_equal(mine, ref_v), f"{key} 与研究参照不一致"
    assert np.allclose(feat["rsi"][tail], np.nan_to_num(ref["rsi"][tail], nan=-1), atol=1e-8)
    # 六个 spec 的掩码逐位一致
    for spec in CRV_SHADOW_SPECS:
        assert np.array_equal(spec_mask(spec, feat)[tail], spec_mask(spec, ref)[tail]), \
            f"{spec['version']} 掩码与研究参照不一致"


# ============================================================
# 判定正反例（手造 K 线，末根命中/不命中）
# ============================================================

def test_brk20_positive_and_no_break_negative() -> None:
    rows = _breakout_rows()
    feat = compute_crv_features(_to_klines(rows, 900_000))
    assert len(rows) - 1 in _tail_hits(feat, "crv_brk20_15m_v1")
    feat_nb = compute_crv_features(_to_klines(_breakout_rows(break_close=None), 900_000))
    assert _tail_hits(feat_nb, "crv_brk20_15m_v1") == []


def test_hi_brk20_requires_pos100() -> None:
    # 突破 + 高位 → 命中
    feat = compute_crv_features(_to_klines(_breakout_rows(), 900_000))
    assert len(_breakout_rows()) - 1 in _tail_hits(feat, "crv_hi_brk20_15m_v1")
    # 突破但 100 根区间内有旧高影（i=60 spike 110）→ pos100 压低 → 不命中
    feat_low = compute_crv_features(_to_klines(_breakout_rows(spike_high=110.0), 900_000))
    assert _tail_hits(feat_low, "crv_hi_brk20_15m_v1") == []
    # 同一根上 brk20（无 pos 门）仍命中：门控差异生效
    assert len(_breakout_rows(spike_high=110.0)) - 1 in _tail_hits(
        compute_crv_features(_to_klines(_breakout_rows(spike_high=110.0), 900_000)),
        "crv_brk20_15m_v1")


def test_brk50_hi_positive() -> None:
    feat = compute_crv_features(_to_klines(_breakout_rows(), 900_000))
    assert len(_breakout_rows()) - 1 in _tail_hits(feat, "crv_brk50_hi_15m_v1")


def test_dnex3_positive_and_negatives() -> None:
    def _band(n: int, last_open: float, last_close: float) -> list[dict]:
        """确定性 ±0.3 交替摆动带（TR≈0.61 稳定，prior_lo20 可控）。"""
        rows = []
        for i in range(n):
            if i % 2 == 0:
                o, c = 99.7, 100.3
            else:
                o, c = 100.3, 99.7
            rows.append(_mk(900_000, i, o, max(o, c) + 0.01, min(o, c) - 0.01, c))
        rows.append(_mk(900_000, n, last_open, last_open + 0.01, last_close - 0.01, last_close))
        return rows

    # 正例：跌破前低(99.69) 0.04、距前低 < 0.25×ATR(≈0.15)、实体 0.65 > ATR(≈0.61)
    rows = _band(129, last_open=100.3, last_close=99.65)
    feat = compute_crv_features(_to_klines(rows, 900_000))
    idx = len(rows) - 1
    assert idx in _tail_hits(feat, "crv_dnex_15m_v1", n_tail=1)
    # 反例 1：深跌破（距前低远超 0.25×ATR）→ 不命中
    feat_deep = compute_crv_features(_to_klines(_band(129, 100.3, 99.0), 900_000))
    assert _tail_hits(feat_deep, "crv_dnex_15m_v1", n_tail=1) == []
    # 反例 2：跌破但小实体（body<ATR）→ 不命中
    feat_small = compute_crv_features(_to_klines(_band(129, 99.9, 99.65), 900_000))
    assert _tail_hits(feat_small, "crv_dnex_15m_v1", n_tail=1) == []


def test_hi_z3_positive_and_negatives() -> None:
    """高位∧z20>+2∧恰连3阳：正例 + streak=4 反例 + z20 不足反例。"""
    def _hiz3_rows(n_up: int = 3, down_dips: bool = False) -> list[dict]:
        rows = []
        price = 100.0
        for i in range(126):
            o = price
            c = price + (0.05 if i % 2 == 0 else -0.05)
            if down_dips and i in (110, 112):
                # 深回撤：收盘 −4.0 但不移动基准价位（下一根收回中枢）
                rows.append(_mk(900_000, i, o, o + 0.05, o - 4.1, o - 4.0))
                continue
            if down_dips and i in (111, 113):
                # 修复性反弹：开盘接前根深收、收盘回中枢基准
                rows.append(_mk(900_000, i, o - 4.0, price + 0.05, o - 4.1, price))
                continue
            rows.append(_mk(900_000, i, o, max(o, c) + 0.02, min(o, c) - 0.02, c))
            price = c
        for k in range(n_up):
            o = price
            c = o + 0.8
            rows.append(_mk(900_000, 126 + k, o, c + 0.05, o - 0.05, c))
            price = c
        return rows

    rows = _hiz3_rows()
    feat = compute_crv_features(_to_klines(rows, 900_000))
    idx = len(rows) - 1
    assert feat["streak"][idx] == 3 and feat["pos100"][idx] > 0.90 and feat["z20"][idx] > 2.0
    assert idx in _tail_hits(feat, "crv_hi_z3_15m_v1", n_tail=1)
    # 反例 1：连 4 阳（streak==4）→ 不命中（恰 3 的精确性）
    feat4 = compute_crv_features(_to_klines(_hiz3_rows(n_up=4), 900_000))
    assert _tail_hits(feat4, "crv_hi_z3_15m_v1", n_tail=1) == []
    # 反例 2：近 20 根内两根深回撤拉大 std20 → z20<2 → 不命中（pos100/streak 仍满足）
    featz = compute_crv_features(_to_klines(_hiz3_rows(down_dips=True), 900_000))
    idxz = len(rows) - 1
    assert featz["streak"][idxz] == 3 and featz["pos100"][idxz] > 0.90
    assert featz["z20"][idxz] < 2.0
    assert _tail_hits(featz, "crv_hi_z3_15m_v1", n_tail=1) == []


def test_os_st4_positive_and_streak_negatives() -> None:
    def _os_rows(n_down: int) -> list[dict]:
        rows = []
        price = 120.0
        for i in range(126):
            rows.append(_mk(300_000, i, price, price + 0.01, price - 0.01, price))
            price -= 0.05
        for k in range(n_down):
            o = price
            c = o - 0.5
            rows.append(_mk(300_000, 126 + k, o, o + 0.02, c - 0.02, c))
            price = c
        return rows

    rows = _os_rows(4)
    feat = compute_crv_features(_to_klines(rows, 300_000))
    assert len(rows) - 1 in _tail_hits(feat, "crv_os_st4_5m_v1", n_tail=1)
    # streak==5 → 不命中（恰 4 的精确性）
    feat5 = compute_crv_features(_to_klines(_os_rows(5), 300_000))
    assert _tail_hits(feat5, "crv_os_st4_5m_v1", n_tail=1) == []
    # streak==3 → 不命中
    feat3 = compute_crv_features(_to_klines(_os_rows(3), 300_000))
    assert _tail_hits(feat3, "crv_os_st4_5m_v1", n_tail=1) == []


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


def _pending(version: str, direction: str, target_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        version=version, direction=direction,
        signal_bar_start=target_start - 900_000,
        target_bar_start=target_start, status="PENDING", win=None,
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


_TARGET = T0 + 60 * 900_000


@pytest.mark.asyncio
@pytest.mark.parametrize(("version", "direction", "green", "win"), [
    ("crv_brk20_15m_v1", "DOWN", False, True),    # 押DOWN：次根收阴 → 赢
    ("crv_brk20_15m_v1", "DOWN", True, False),    # 次根收阳 → 输
    ("crv_dnex_15m_v1", "UP", True, True),        # 探索级押UP：次根收阳 → 赢
    ("crv_dnex_15m_v1", "UP", False, False),
])
async def test_settle_by_direction(monkeypatch, version, direction, green, win) -> None:
    session = _FakeSession(rows=[_pending(version, direction, _TARGET)])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    c = 101.5 if green else 99.0
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 102.0, "low": 98.5,
               "close": c, "volume": 1.0}]
    await d._settle_pending("15m", closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is win
    assert sig.settle_outcome == ("UP" if green else "DOWN")


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    session = _FakeSession(rows=[_pending("crv_brk20_5m_v1", "DOWN", _TARGET)])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 100.5,
               "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending("5m", closed)
    assert session.rows[0].status == "EXPIRED" and session.rows[0].win is None


@pytest.mark.asyncio
async def test_record_signal_fields_and_idempotent(monkeypatch) -> None:
    rows = _breakout_rows()
    kl = _to_klines(rows, 900_000)
    feat = compute_crv_features(kl)
    spec = _spec("crv_brk20_15m_v1")
    idx = len(rows) - 1
    fresh = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(fresh))
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(csd.time, "time", lambda: (int(rows[idx]["open_time"]) + 900_000 + 30_000) / 1000)
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    assert await d._record_signal("15m", fresh, spec, rows[idx], feat, idx, None)
    sig = fresh.added[0]
    assert sig.version == "crv_brk20_15m_v1" and sig.direction == "DOWN"
    assert sig.timeframe == "15m" and sig.status == "PENDING"
    assert sig.target_bar_start == int(rows[idx]["open_time"]) + 900_000
    # 幂等：已有同 (version, signal_bar_start) 行 → 不再落
    dup = _FakeSession(rows=[], scalar=123)
    assert not await d._record_signal("15m", dup, spec, rows[idx], feat, idx, None)
    assert dup.added == []


@pytest.mark.asyncio
async def test_fresh_hit_dispatches_frozen_direction(monkeypatch) -> None:
    rows = _breakout_rows()
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    d._last_evaluated_bar["15m"] = int(rows[-2]["open_time"])
    fired: list[dict] = []
    d._on_live_fire = fired.append
    target = int(rows[-1]["open_time"]) + 900_000
    monkeypatch.setattr(csd.time, "time", lambda: (target + 30_000) / 1000)
    await d._evaluate_new_bars("15m", rows)
    payload = next(p for p in fired if p["version"] == "crv_brk20_15m_v1")
    assert payload["market_start"] == target
    assert payload["direction"] == "DOWN"
    assert payload["market_end"] == target + 900_000


@pytest.mark.asyncio
async def test_stale_and_backscan_hits_never_dispatch(monkeypatch) -> None:
    rows = _breakout_rows()
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    target = int(rows[-1]["open_time"]) + 900_000
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    fired: list[dict] = []
    d._on_live_fire = fired.append
    # 陈旧命中（目标根开盘后 10 分钟）→ 只落影子不派单
    d._last_evaluated_bar["15m"] = int(rows[-2]["open_time"])
    monkeypatch.setattr(csd.time, "time", lambda: (target + 600_000) / 1000)
    await d._evaluate_new_bars("15m", rows)
    assert fired == []
    # 冷启动回补 → 永不派单
    d2 = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    d2._on_live_fire = fired.append
    monkeypatch.setattr(csd.time, "time", lambda: (target + 30_000) / 1000)
    await d2._evaluate_new_bars("15m", rows)
    assert fired == []
    assert session.added  # 影子照常落库


# ============================================================
# 实盘注册：默认 OFF / 族与护栏 / 互斥组 / 影子投影
# ============================================================

def test_crv_live_channels_registered_default_off() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS, exclusive_group, parse_channel_config

    expected = {
        "crv_hi_brk20_15m_v1": ("15m", "DOWN", 0.62),
        "crv_brk20_15m_v1": ("15m", "DOWN", 0.57),
        "crv_brk50_hi_15m_v1": ("15m", "DOWN", 0.59),
        "crv_brk20_5m_v1": ("5m", "DOWN", 0.53),
    }
    for version, (period, direction, guard) in expected.items():
        spec = LIVE_CHANNELS[version]
        assert spec.family == "kline_reversal"
        assert spec.market_period == period and spec.direction == direction
        assert spec.auto_max_exec == guard
    # 探索级/观察级版本不注册实盘
    for shadow_only in ("crv_dnex_15m_v1", "crv_os_st4_5m_v1", "crv_hi_z3_15m_v1"):
        assert shadow_only not in LIVE_CHANNELS
    # 默认全部 OFF（parse_channel_config 无覆盖时）
    configs = parse_channel_config()
    for version in expected:
        assert configs[version].enabled is False
    # 15m 三通道同窗互斥（2026-10-05 起并入 BRK 同源突破回归通道：同一根突破
    # 阳线事件子集、同押 DOWN 同目标窗 → 同窗至多一单成交）
    grp = exclusive_group("crv_hi_brk20_15m_v1")
    assert grp == {"crv_hi_brk20_15m_v1", "crv_brk20_15m_v1", "crv_brk50_hi_15m_v1",
                   "brkrv_brk8h_15m_v1"}
    assert exclusive_group("crv_brk20_5m_v1") == {"crv_brk20_5m_v1", "brkrv_brk8h_5m_v1"}


def test_crv_versions_in_shadow_projection() -> None:
    from binance_predict.services.shadow_execution_registry import (
        SHADOW_VERSIONS, SHADOW_VERSION_SPECS)

    assert set(CRV_VERSIONS) <= set(SHADOW_VERSIONS)
    for version in CRV_VERSIONS:
        spec = SHADOW_VERSION_SPECS[version]
        assert spec.family == "kline_reversal"
        assert spec.direction_mode == "row"
    # 四个确认级映射同名实盘通道；两个探索级无实盘映射
    for version in ("crv_hi_brk20_15m_v1", "crv_brk20_15m_v1",
                    "crv_brk50_hi_15m_v1", "crv_brk20_5m_v1"):
        assert SHADOW_VERSION_SPECS[version].live_channel == version
    for version in ("crv_dnex_15m_v1", "crv_os_st4_5m_v1", "crv_hi_z3_15m_v1"):
        assert SHADOW_VERSION_SPECS[version].live_channel is None


def test_settings_default_on() -> None:
    from binance_predict.config.settings import settings
    assert settings.crv_shadow_enabled is True
