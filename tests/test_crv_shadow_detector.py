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
    atr_pct_tail,
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


def _band_rows(n: int, tf_ms: int, last: tuple[float, float, float, float] | None = None) -> list[dict]:
    """n 根确定性交替带（低点恒 99.69，prior_lo24 可控）+ 可选末根 (o,h,l,c)。"""
    rows = []
    for i in range(n):
        o, c = (99.7, 100.3) if i % 2 == 0 else (100.3, 99.7)
        rows.append(_mk(tf_ms, i, o, max(o, c) + 0.01, min(o, c) - 0.01, c))
    if last is not None:
        lo, hi, o, c = last
        rows.append(_mk(tf_ms, n, o, hi, lo, c))
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
           for k in ("open", "high", "low", "close", "volume")}
    o, h, l, c, v = arr["open"], arr["high"], arr["low"], arr["close"], arr["volume"]
    t = np.array([r["open_time"] for r in rows], dtype=np.int64)

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
    feat["prior_lo24"] = shift1(roll(np.min, l, 24))
    # engulf：阴线吞没（当前实体 > 1.5×前一实体 ∧ 方向反转；前一根缺失不触发）
    prev_body = shift1(feat["body"])
    prev_sign = shift1(np.sign(c - o))
    with np.errstate(invalid="ignore"):
        engulf = (np.sign(c - o) != prev_sign) & (np.sign(c - o) != 0) & (feat["body"] > 1.5 * prev_body)
    engulf[np.isnan(prev_body)] = False
    feat["engulf"] = engulf
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
    # 2026-10-15 新特征（独立转写）：前 20 根（不含当前）均量比、实体/振幅、15m 边界收盘
    vr = np.full(len(v), np.nan)
    for i in range(20, len(v)):
        vr[i] = v[i] / v[i - 20:i].mean()
    feat["vol_ratio"] = vr
    span = h - l
    feat["body_ratio"] = np.where(span > 0, np.abs(c - o) / np.where(span > 0, span, 1.0), np.nan)
    feat["b15_end"] = (t % 900_000) == 600_000
    # 2026-10-16 KDC 新特征（独立转写：prior24/96高、ATR20、6根动量、收盘位置）
    feat["prior_hi24"] = shift1(roll(max, h, 24))
    feat["prior_hi96"] = shift1(roll(max, h, 96))
    tr_full = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        tr_full[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    feat["atr20"] = roll(lambda w: float(np.mean(w)), tr_full, 20)
    dc = np.full(len(c), np.nan)
    dc[1:] = c[1:] - c[:-1]
    feat["mom6"] = roll(lambda w: float(np.sum(w)), dc, 6)
    feat["close_pos"] = np.where(span > 0, (c - l) / np.where(span > 0, span, 1.0), np.nan)
    return feat


@pytest.mark.parametrize("seed", [3, 11, 42])
def test_feature_bitwise_parity_vs_research_reference(seed: int) -> None:
    rows = _quiet_rows(130, seed=seed)
    kl = _to_klines(rows, 900_000)
    feat = compute_crv_features(kl)
    ref = _research_reference(rows, 900_000)
    tail = slice(105, 130)  # 全部 warmup（pos100=100/atr=14/滚动20/50）之后
    for key in ("up", "dn", "prior_hi20", "prior_lo20", "prior_hi50", "prior_lo24",
                "pos100", "z20", "atr", "streak", "vol_ratio", "body_ratio"):
        mine, ref_v = feat[key][tail], ref[key][tail]
        assert np.array_equal(mine, ref_v), f"{key} 与研究参照不一致"
    assert np.array_equal(feat["engulf"][tail], ref["engulf"][tail]), "engulf 与研究参照不一致"
    assert np.array_equal(feat["b15_end"][tail], ref["b15_end"][tail])
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


def test_dnbrk24_positive_and_negatives() -> None:
    """阴线吞没∧跌破前24低（2026-10-12 复核「A 信号下破半」）正例 + 三类反例。"""
    # 正例：末根阴线实体 1.0 > 1.5×前一实体(0.6)=0.9 且跌破前低(99.69) → 命中
    rows_p = _band_rows(129, 900_000, last=(99.3, 100.35, 100.3, 99.3))
    feat_p = compute_crv_features(_to_klines(rows_p, 900_000))
    assert len(rows_p) - 1 in _tail_hits(feat_p, "crv_dnbrk24_15m_v1", n_tail=1)
    # 反例 1：阴线破低但小实体（未吞没）→ 不命中
    rows_small = _band_rows(129, 900_000, last=(99.6, 99.75, 99.7, 99.6))
    feat_small = compute_crv_features(_to_klines(rows_small, 900_000))
    assert _tail_hits(feat_small, "crv_dnbrk24_15m_v1", n_tail=1) == []
    # 反例 2：吞没但未跌破前 24 低（收盘 99.7 > 前低 99.69）→ 不命中
    rows_nb = _band_rows(129, 900_000, last=(99.7, 100.85, 100.7, 99.7))
    feat_nb = compute_crv_features(_to_klines(rows_nb, 900_000))
    assert feat_nb["engulf"][len(rows_nb) - 1] and not bool(
        spec_mask(_spec("crv_dnbrk24_15m_v1"), feat_nb)[len(rows_nb) - 1])
    assert _tail_hits(feat_nb, "crv_dnbrk24_15m_v1", n_tail=1) == []
    # 反例 3：阳线吞没（方向非阴）→ 不命中
    rows_up = _band_rows(129, 900_000, last=(99.0, 100.35, 99.0, 100.0))
    feat_up = compute_crv_features(_to_klines(rows_up, 900_000))
    assert _tail_hits(feat_up, "crv_dnbrk24_15m_v1", n_tail=1) == []


def test_dnbrk24_warmup_nan_not_trigger() -> None:
    """prior_lo24 需 24 根 trailing；不足 → NaN → 不触发（NaN 保守）。"""
    rows = _band_rows(20, 900_000)
    feat = compute_crv_features(_to_klines(rows, 900_000))
    assert np.all(np.isnan(feat["prior_lo24"]))
    assert _tail_hits(feat, "crv_dnbrk24_15m_v1", n_tail=20) == []


def test_dnbrk24_5m_spec_and_mask() -> None:
    """5m 版本同样注册且掩码口径一致。"""
    rows = _band_rows(129, 300_000, last=(99.3, 100.35, 100.3, 99.3))
    feat = compute_crv_features(_to_klines(rows, 300_000))
    assert len(rows) - 1 in _tail_hits(feat, "crv_dnbrk24_5m_v1", n_tail=1)


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

    async def flush(self) -> None:  # KDC 路径取 id 用；假会话无自增
        for i, obj in enumerate(self.added, start=1):
            if getattr(obj, "id", None) is None:
                try:
                    obj.id = i
                except AttributeError:
                    pass

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
        # 2026-10-12 复核「A 信号下破半」：方向与突破族相反（押 UP）
        "crv_dnbrk24_15m_v1": ("15m", "UP", 0.606),
        "crv_dnbrk24_5m_v1": ("5m", "UP", 0.546),
        # 2026-10-15 反转研究回顾性验证：放量下破（5m 求值→15m 市场押 UP）/ 大实体突破
        "crv_dnvol20_15m_v1": ("15m", "UP", 0.57),
        "crv_bodybrk20_5m_v1": ("5m", "DOWN", 0.53),
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
    # 2026-10-22 治理：brkrv_brk8h_15m_v1 已归档 → 运行时求交后不再出现在组内
    assert grp == {"crv_hi_brk20_15m_v1", "crv_brk20_15m_v1", "crv_brk50_hi_15m_v1",
                   "kdc_s1_96hi_15m_v1", "kdc_s2_ext_15m_v1",
                   "kdc_s3_3up_15m_v1"}
    assert exclusive_group("crv_brk20_5m_v1") == {"crv_brk20_5m_v1", "brkrv_brk8h_5m_v1",
                                                  "crv_bodybrk20_5m_v1"}
    # 下破吞没族事件集与突破族完全不相交（阴线破低 vs 阳线破高），不并入其互斥组；
    # 2026-10-16 起 KDC 破20低3连阴（押 UP 同向同窗）并入本组
    assert exclusive_group("crv_dnbrk24_15m_v1") == {"crv_dnbrk24_15m_v1", "crv_dnvol20_15m_v1",
                                                     "kdc_s4_3dn_15m_v1"}
    assert exclusive_group("crv_dnbrk24_5m_v1") is None


def test_crv_versions_in_shadow_projection() -> None:
    from binance_predict.services.shadow_execution_registry import (
        SHADOW_VERSIONS, SHADOW_VERSION_SPECS)

    assert set(CRV_VERSIONS) <= set(SHADOW_VERSIONS)
    for version in CRV_VERSIONS:
        spec = SHADOW_VERSION_SPECS[version]
        assert spec.family == "kline_reversal"
        assert spec.direction_mode == "row"
    # 确认级映射同名实盘通道；探索/观察级无实盘映射
    for version in ("crv_hi_brk20_15m_v1", "crv_brk20_15m_v1",
                    "crv_brk50_hi_15m_v1", "crv_brk20_5m_v1",
                    "crv_dnbrk24_15m_v1", "crv_dnbrk24_5m_v1",
                    "crv_dnvol20_15m_v1", "crv_bodybrk20_5m_v1"):
        assert SHADOW_VERSION_SPECS[version].live_channel == version
    for version in ("crv_dnex_15m_v1", "crv_os_st4_5m_v1", "crv_hi_z3_15m_v1"):
        assert SHADOW_VERSION_SPECS[version].live_channel is None


def test_settings_default_on() -> None:
    from binance_predict.config.settings import settings
    assert settings.crv_shadow_enabled is True


def test_dnbrk24_versions_in_shadow_bench() -> None:
    """防回归：dnbrk24 两版本已登记 main.SHADOW_BENCH——面板基准与
    /api/shadow/toggle 白名单的唯一事实源（漏注册会导致面板下线按钮 422）。"""
    from binance_predict.main import SHADOW_BENCH

    assert "crv_dnbrk24_15m_v1" in SHADOW_BENCH
    assert "crv_dnbrk24_5m_v1" in SHADOW_BENCH
    # 冻结基准胜率：样本外(验证+盲测+前向)口径 15m 61.85% / 5m 55.74%
    assert SHADOW_BENCH["crv_dnbrk24_15m_v1"][0] == 0.6185
    assert SHADOW_BENCH["crv_dnbrk24_5m_v1"][0] == 0.5574
    assert SHADOW_BENCH["crv_dnbrk24_15m_v1"][1] is None
    assert SHADOW_BENCH["crv_dnbrk24_5m_v1"][1] is None


# ============================================================
# 2026-10-15 反转研究回顾性验证：crv_dnvol20_15m_v1 / crv_bodybrk20_5m_v1
# ============================================================

def _rows5(n: int, last: tuple[float, float, float, float] | None = None,
           last_vol: float = 1.0) -> list[dict]:
    """5m 确定性交替带（成交量恒 1.0，T0 为 15m 边界）+ 可选末根 (lo,hi,o,c) 与其成交量。"""
    rows = _band_rows(n, 300_000)
    if last is not None:
        lo, hi, o, c = last
        rows.append({**_mk(300_000, n, o, hi, lo, c), "volume": last_vol})
    return rows


def test_dnvol20_positive_and_negatives() -> None:
    """放量下破20低∧15m边界收盘：正例 + 缩量/非边界/阳线/未破低 四类反例。"""
    spec = "crv_dnvol20_15m_v1"
    # 末根下标 131 → open_time = T0+131*5m，131%3==2 即 15m 窗口最后一根 5m
    pos = _rows5(131, last=(99.3, 100.35, 100.3, 99.3), last_vol=2.0)
    feat = compute_crv_features(_to_klines(pos, 300_000))
    assert feat["b15_end"][len(pos) - 1]
    assert len(pos) - 1 in _tail_hits(feat, spec, n_tail=1)
    # 反例 1：量比 1.4 < 1.5 → 不命中；恰 1.5 → 命中（含贴线）
    low = compute_crv_features(_to_klines(_rows5(131, (99.3, 100.35, 100.3, 99.3), 1.4), 300_000))
    assert _tail_hits(low, spec, n_tail=1) == []
    edge = compute_crv_features(_to_klines(_rows5(131, (99.3, 100.35, 100.3, 99.3), 1.5), 300_000))
    assert len(pos) - 1 in _tail_hits(edge, spec, n_tail=1)
    # 反例 2：非 15m 边界（末根下标 130，130%3==1）→ 不命中
    nb = compute_crv_features(_to_klines(_rows5(130, (99.3, 100.35, 100.3, 99.3), 2.0), 300_000))
    assert not nb["b15_end"][130] and _tail_hits(nb, spec, n_tail=1) == []
    # 反例 3：阳线（即便放量、边界）→ 不命中
    up = compute_crv_features(_to_klines(_rows5(131, (99.3, 100.35, 99.3, 100.3), 2.0), 300_000))
    assert _tail_hits(up, spec, n_tail=1) == []
    # 反例 4：阴线放量但未跌破前 20 低（收盘 99.7 > 前低 99.69）→ 不命中
    nobrk = compute_crv_features(_to_klines(_rows5(131, (99.69, 100.35, 100.3, 99.7), 2.0), 300_000))
    assert _tail_hits(nobrk, spec, n_tail=1) == []


def test_dnvol20_never_fires_on_15m_bars() -> None:
    """15m K 线上 b15_end 恒 False（open_time%15m==0）→ 该 spec 永不命中。"""
    rows = _band_rows(131, 900_000, last=(99.3, 100.35, 100.3, 99.3))
    rows[-1] = {**rows[-1], "volume": 5.0}
    feat = compute_crv_features(_to_klines(rows, 900_000))
    assert not feat["b15_end"].any()
    assert _tail_hits(feat, "crv_dnvol20_15m_v1", n_tail=5) == []


def test_bodybrk20_positive_and_negatives() -> None:
    """大实体突破20高：正例 + 长上影/未突破/阴线 反例；实体比 0.75 含贴线。"""
    spec = "crv_bodybrk20_5m_v1"
    pos = _rows5(129, last=(99.98, 101.05, 100.0, 101.0))
    feat = compute_crv_features(_to_klines(pos, 300_000))
    assert len(pos) - 1 in _tail_hits(feat, spec, n_tail=1)
    # 长上影（实体 1.0 / 振幅 2.02 < 0.75）→ 不命中
    wick = compute_crv_features(_to_klines(_rows5(129, (99.98, 102.0, 100.0, 101.0)), 300_000))
    assert _tail_hits(wick, spec, n_tail=1) == []
    # 实体占比高但未突破前 20 高 → 不命中
    nb = compute_crv_features(_to_klines(_rows5(129, (99.99, 100.21, 100.0, 100.2)), 300_000))
    assert nb["body_ratio"][len(pos) - 1] > 0.75 and _tail_hits(nb, spec, n_tail=1) == []
    # 阴线 → 不命中
    dn = compute_crv_features(_to_klines(_rows5(129, (99.0, 101.05, 101.0, 99.98)), 300_000))
    assert _tail_hits(dn, spec, n_tail=1) == []
    # 贴线：实体/振幅恰 0.75 → 命中（>=）；略小 → 不命中
    exact = compute_crv_features(_to_klines(_rows5(129, (100.0, 101.0, 100.0, 100.75)), 300_000))
    assert exact["body_ratio"][len(pos) - 1] == pytest.approx(0.75)
    assert len(pos) - 1 in _tail_hits(exact, spec, n_tail=1)
    under = compute_crv_features(_to_klines(_rows5(129, (100.0, 101.0, 100.0, 100.74)), 300_000))
    assert _tail_hits(under, spec, n_tail=1) == []


def test_new_features_parity_on_5m_random_walk() -> None:
    """vol_ratio/body_ratio/b15_end 与独立参照逐位一致（5m 随机游走+随机成交量）。"""
    rng = np.random.default_rng(7)
    rows, price = [], 100.0
    for i in range(2400):
        o = price
        c = price + rng.normal(0, 0.4)
        rows.append({**_mk(300_000, i, o, max(o, c) + abs(rng.normal(0, 0.15)),
                           min(o, c) - abs(rng.normal(0, 0.15)), c),
                     "volume": float(rng.uniform(0.5, 3.0))})
        price = c
    feat = compute_crv_features(_to_klines(rows, 300_000))
    ref = _research_reference(rows, 300_000)
    tail = slice(130, 2400)
    for key in ("vol_ratio", "body_ratio"):
        assert np.array_equal(feat[key][tail], ref[key][tail]), key
    assert np.array_equal(feat["b15_end"][tail], ref["b15_end"][tail])
    assert ref["b15_end"][tail].sum() > 0
    for version in ("crv_dnvol20_15m_v1", "crv_bodybrk20_5m_v1"):
        spec = _spec(version)
        mine, theirs = spec_mask(spec, feat)[tail], spec_mask(spec, ref)[tail]
        assert mine.sum() > 0, f"{version} 随机序列应至少命中一次，否则本对照无意义"
        assert np.array_equal(mine, theirs), version


def test_dnvol20_groups_by_target_tf() -> None:
    """5m 求值、押下一根 15m：结算/过期按 15m 分组；bodybrk20 仍归 5m。"""
    assert "crv_dnvol20_15m_v1" in csd.CRV_VERSIONS_BY_TF["15m"]
    assert "crv_dnvol20_15m_v1" not in csd.CRV_VERSIONS_BY_TF["5m"]
    assert "crv_bodybrk20_5m_v1" in csd.CRV_VERSIONS_BY_TF["5m"]


@pytest.mark.asyncio
async def test_dnvol20_record_and_dispatch_use_15m_target(monkeypatch) -> None:
    rows = _rows5(131, last=(99.3, 100.35, 100.3, 99.3), last_vol=2.0)
    feat = compute_crv_features(_to_klines(rows, 300_000))
    idx = len(rows) - 1
    bar = rows[idx]
    target = int(bar["open_time"]) + 300_000
    assert target % 900_000 == 0  # 目标窗恰为 15m 边界
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(csd.time, "time", lambda: (target + 3_000) / 1000)
    pm15 = {"start_date": target, "up_price": 0.55, "down_price": 0.45, "updated_ts": target + 2_000}
    pm5 = {"start_date": target, "up_price": 0.99, "down_price": 0.01, "updated_ts": target + 2_000}
    d = CrvShadowDetector(collector=None, pm_15m_latest=pm15, pm_5m_info=pm5)
    payloads: list[dict] = []
    assert await d._record_signal("5m", session, _spec("crv_dnvol20_15m_v1"), bar, feat, idx, payloads)
    sig = session.added[0]
    assert sig.timeframe == "15m" and sig.direction == "UP" and sig.status == "PENDING"
    assert sig.signal_bar_start == int(bar["open_time"]) and sig.target_bar_start == target
    assert sig.entry_up_price == 0.55  # 取 15m 报价缓存，而非 5m 缓存的 0.99
    assert payloads == [{
        "version": "crv_dnvol20_15m_v1", "market_start": target, "market_end": target + 900_000,
        "direction": "UP", "signal_bar_start": int(bar["open_time"]),
    }]
    assert sig.feature_snapshot["vol_ratio"] == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_bodybrk20_payload_stays_5m(monkeypatch) -> None:
    rows = _rows5(129, last=(99.98, 101.05, 100.0, 101.0))
    feat = compute_crv_features(_to_klines(rows, 300_000))
    idx = len(rows) - 1
    target = int(rows[idx]["open_time"]) + 300_000
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(csd.time, "time", lambda: (target + 3_000) / 1000)
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    payloads: list[dict] = []
    assert await d._record_signal("5m", session, _spec("crv_bodybrk20_5m_v1"), rows[idx], feat, idx, payloads)
    assert session.added[0].timeframe == "5m" and session.added[0].direction == "DOWN"
    assert payloads[0]["market_end"] - payloads[0]["market_start"] == 300_000


@pytest.mark.asyncio
@pytest.mark.parametrize(("green", "win"), [(True, True), (False, False)])
async def test_dnvol20_settles_on_15m_bar(monkeypatch, green, win) -> None:
    """押 UP：下一根 15m 收阳赢，收阴输。"""
    session = _FakeSession(rows=[_pending("crv_dnvol20_15m_v1", "UP", _TARGET)])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    c = 101.5 if green else 99.0
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 102.0, "low": 98.5, "close": c, "volume": 1.0}]
    await d._settle_pending("15m", closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is win


def test_new_channels_default_off_exclusive_and_bench() -> None:
    from binance_predict.main import SHADOW_BENCH
    from binance_predict.services.live_channels import LIVE_CHANNELS, exclusive_group, parse_channel_config

    cfg = parse_channel_config()
    for v in ("crv_dnvol20_15m_v1", "crv_bodybrk20_5m_v1"):
        assert LIVE_CHANNELS[v].family == "kline_reversal"
        assert cfg[v].enabled is False            # 默认不开实盘
        assert v in SHADOW_BENCH and SHADOW_BENCH[v][1] is None
    assert LIVE_CHANNELS["crv_dnvol20_15m_v1"].market_period == "15m"
    assert exclusive_group("crv_dnvol20_15m_v1") == {
        "crv_dnbrk24_15m_v1", "crv_dnvol20_15m_v1", "kdc_s4_3dn_15m_v1"}
    assert "crv_bodybrk20_5m_v1" in exclusive_group("crv_brk20_5m_v1")
    assert SHADOW_BENCH["crv_dnvol20_15m_v1"][0] == 0.5906
    assert SHADOW_BENCH["crv_bodybrk20_5m_v1"][0] == 0.5517


# ============================================================
# KDC 方向确定性四版本（2026-10-16，研究 run 20261015T153500Z 冻结口径）
# ============================================================

def test_atr_pct_tail_rank_and_insufficient() -> None:
    """ATR20 近500根分位：最大→1.0、最小→1/500、不足/NaN→None。"""
    arr = np.concatenate([np.full(499, 1.0), [2.0]])
    assert atr_pct_tail(arr, 499) == 1.0
    arr2 = np.concatenate([np.full(499, 1.0), [0.5]])
    assert atr_pct_tail(arr2, 499) == pytest.approx(1.0 / 500)
    assert atr_pct_tail(np.full(300, 1.0), 299) is None
    assert atr_pct_tail(np.full(500, np.nan), 499) is None


def test_kdc_s1_96hi_positive_and_negatives() -> None:
    """破96高收顶：正例 + 长上影/未突破 反例；close_pos 0.8 含贴线。"""
    spec = "kdc_s1_96hi_15m_v1"
    pos = _band_rows(97, 900_000, last=(100.0, 102.0, 99.9, 101.9))
    feat = compute_crv_features(_to_klines(pos, 900_000))
    assert feat["close_pos"][-1] > 0.8 and feat["close"][-1] > feat["prior_hi96"][-1]
    assert len(pos) - 1 in _tail_hits(feat, spec, n_tail=1)
    # 反例1：破96高但收在区间下沿（长上影 close_pos<0.8）→ 不命中
    wick = compute_crv_features(
        _to_klines(_band_rows(97, 900_000, (100.0, 104.0, 99.9, 101.0)), 900_000))
    assert wick["close"][-1] > wick["prior_hi96"][-1] and wick["close_pos"][-1] < 0.8
    assert _tail_hits(wick, spec, n_tail=1) == []
    # 反例2：收顶但未破 96 高 → 不命中
    nb = compute_crv_features(
        _to_klines(_band_rows(97, 900_000, (99.7, 100.4, 99.68, 100.3)), 900_000))
    assert nb["close_pos"][-1] > 0.8 and nb["close"][-1] < nb["prior_hi96"][-1]
    assert _tail_hits(nb, spec, n_tail=1) == []
    # 贴线：close_pos 恰 0.8 → 命中（>=）；略小 → 不命中
    # 贴线：close_pos 恰 0.8（l=100,c=104,h=105 → 4/5 二进制精确）→ 命中（>=）
    edge = compute_crv_features(
        _to_klines(_band_rows(97, 900_000, (100.0, 105.0, 100.0, 104.0)), 900_000))
    assert edge["close_pos"][-1] == pytest.approx(0.8)
    assert len(pos) - 1 in _tail_hits(edge, spec, n_tail=1)
    under = compute_crv_features(
        _to_klines(_band_rows(97, 900_000, (100.0, 105.0, 100.0, 103.99)), 900_000))
    assert _tail_hits(under, spec, n_tail=1) == []


def test_kdc_s2_ext_positive_and_negatives() -> None:
    """超24高≥1ATR∧6根冲顶：正例（6根连涨冲出）+ 冲出不足反例；阈值含贴线。"""
    spec = "kdc_s2_ext_15m_v1"
    rows = _band_rows(130, 900_000)
    price = rows[-1]["close"]
    for k in range(5):  # 前 5 根小步连涨（+0.10），末根大跳 +1.5 冲出 24 高
        o = price
        price += 0.10
        rows.append(_mk(900_000, 130 + k, o, price + 0.02, o - 0.02, price))
    o = price
    price += 1.5
    rows.append(_mk(900_000, 135, o, price + 0.05, o - 0.05, price))
    feat = compute_crv_features(_to_klines(rows, 900_000))
    i = len(rows) - 1
    assert (feat["close"][i] - feat["prior_hi24"][i]) / feat["atr20"][i] >= 1.0
    assert feat["mom6"][i] / feat["atr20"][i] >= 2.0
    assert i in _tail_hits(feat, spec, n_tail=1)
    # 反例：6 根连涨但冲出幅度 <1×ATR → 不命中
    rows2 = _band_rows(130, 900_000)
    price = rows2[-1]["close"]
    for k in range(6):
        o = price
        price += 0.06
        rows2.append(_mk(900_000, 130 + k, o, price + 0.02, o - 0.02, price))
    f2 = compute_crv_features(_to_klines(rows2, 900_000))
    j = len(rows2) - 1
    assert (f2["close"][j] - f2["prior_hi24"][j]) / f2["atr20"][j] < 1.0
    assert _tail_hits(f2, spec, n_tail=1) == []


def test_kdc_s3_3up_positive_and_negatives() -> None:
    """收顶3连阳破20高：正例 + 长上影反例 + 第3根阴线反例。"""
    spec = "kdc_s3_3up_15m_v1"

    def _rows3(third: tuple[float, float, float, float]) -> list[dict]:
        rows = _band_rows(100, 900_000)
        rows.append(_mk(900_000, 100, 99.7, 100.35, 99.69, 100.3))   # 阳线1
        rows.append(_mk(900_000, 101, 100.3, 100.95, 100.29, 100.9))  # 阳线2
        lo, hi, o, c = third
        rows.append(_mk(900_000, 102, o, hi, lo, c))
        return rows

    pos = _rows3((100.85, 103.0, 100.9, 102.9))
    feat = compute_crv_features(_to_klines(pos, 900_000))
    assert feat["streak"][-1] == 3 and feat["close_pos"][-1] > 0.8
    assert len(pos) - 1 in _tail_hits(feat, spec, n_tail=1)
    wick = compute_crv_features(_to_klines(_rows3((100.85, 104.5, 100.9, 101.6)), 900_000))
    assert wick["streak"][-1] == 3 and wick["close_pos"][-1] < 0.8
    assert _tail_hits(wick, spec, n_tail=1) == []
    rows = _band_rows(100, 900_000)
    rows.append(_mk(900_000, 100, 99.7, 100.35, 99.69, 100.3))
    rows.append(_mk(900_000, 101, 100.3, 100.95, 100.29, 100.9))
    # 第3根阴线（o=103.5>c=102.9）但收顶(close_pos=0.806)且破20高 → 仅连阳断裂
    rows.append(_mk(900_000, 102, 103.5, 103.6, 100.0, 102.9))
    two = compute_crv_features(_to_klines(rows, 900_000))
    assert two["close_pos"][-1] > 0.8 and two["close"][-1] > two["prior_hi20"][-1]
    assert _tail_hits(two, spec, n_tail=1) == []


def test_kdc_s4_3dn_positive_and_negatives() -> None:
    """破20低3连阴大阴：正例 + 小实体反例 + 第3根阳线反例；0.75 含贴线。"""
    spec = "kdc_s4_3dn_15m_v1"

    def _rows4(third: tuple[float, float, float, float]) -> list[dict]:
        rows = _band_rows(100, 900_000)
        rows.append(_mk(900_000, 100, 100.3, 100.35, 99.69, 99.75))  # 阴线1
        rows.append(_mk(900_000, 101, 99.75, 99.8, 99.1, 99.2))      # 阴线2
        lo, hi, o, c = third
        rows.append(_mk(900_000, 102, o, hi, lo, c))
        return rows

    pos = _rows4((97.0, 99.25, 99.2, 97.1))
    feat = compute_crv_features(_to_klines(pos, 900_000))
    assert feat["body_ratio"][-1] > 0.75 and feat["close"][-1] < feat["prior_lo20"][-1]
    assert len(pos) - 1 in _tail_hits(feat, spec, n_tail=1)
    small = compute_crv_features(_to_klines(_rows4((97.0, 99.25, 99.2, 98.6)), 900_000))
    assert small["close"][-1] < small["prior_lo20"][-1]
    assert small["body_ratio"][-1] < 0.75
    assert _tail_hits(small, spec, n_tail=1) == []
    rows = _band_rows(100, 900_000)
    rows.append(_mk(900_000, 100, 100.3, 100.35, 99.69, 99.75))
    rows.append(_mk(900_000, 101, 99.75, 99.8, 99.1, 99.2))
    # 第3根阳线（o=97.0<c=98.0）破低且大实体 → 仅连阴断裂
    rows.append(_mk(900_000, 102, 97.0, 98.1, 96.9, 98.0))
    two = compute_crv_features(_to_klines(rows, 900_000))
    assert two["close"][-1] < two["prior_lo20"][-1] and two["body_ratio"][-1] > 0.75
    assert _tail_hits(two, spec, n_tail=1) == []


def test_kdc_features_parity_random_walk() -> None:
    """KDC 新特征与独立 numpy 参照逐位一致；四版本掩码在随机序列上与参照全等。"""
    rng = np.random.default_rng(20261015)
    rows, price = [], 100.0
    for i in range(700):
        o = price
        c = price + rng.normal(0, 0.5)
        rows.append({**_mk(900_000, i, o, max(o, c) + abs(rng.normal(0, 0.2)),
                           min(o, c) - abs(rng.normal(0, 0.2)), c),
                     "volume": float(rng.uniform(0.5, 3.0))})
        price = c
    kl = _to_klines(rows, 900_000)
    feat = compute_crv_features(kl)
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    from numpy.lib.stride_tricks import sliding_window_view as swv

    def _roll(fn, x, w):
        out = np.full(len(x), np.nan)
        out[w - 1:] = fn(swv(x, w), axis=1)
        return out

    def _shift1(x):
        out = np.full(len(x), np.nan)
        out[1:] = x[:-1]
        return out

    pc = _shift1(c)
    tr = np.full(len(c), np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])])
    ref = {
        "prior_hi24": _shift1(_roll(np.max, h, 24)),
        "prior_hi96": _shift1(_roll(np.max, h, 96)),
        "atr20": _roll(np.mean, tr, 20),
        "mom6": _roll(np.sum, c - pc, 6),
        "close_pos": np.where((h - l) > 0, (c - l) / np.where((h - l) > 0, h - l, 1.0), np.nan),
    }
    tail = slice(130, 700)
    for key in ref:
        a, b = feat[key][tail], ref[key][tail]
        assert np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)]), key
    ph20 = _shift1(_roll(np.max, h, 20))
    pl20 = _shift1(_roll(np.min, l, 20))
    br = np.where((h - l) > 0, np.abs(c - o) / np.where((h - l) > 0, h - l, 1.0), np.nan)

    def _run3(mask):
        out = np.ones(len(c), bool)
        for k in (1, 2):
            sh = np.roll(mask, k)
            sh[:k] = False
            out &= sh
        return out

    with np.errstate(invalid="ignore", divide="ignore"):
        expected = {
            "kdc_s1_96hi_15m_v1": (ref["close_pos"] >= 0.8) & (c > ref["prior_hi96"]),
            "kdc_s2_ext_15m_v1": ((c - ref["prior_hi24"]) / ref["atr20"] >= 1.0)
                                  & (ref["mom6"] / ref["atr20"] >= 2.0),
            "kdc_s3_3up_15m_v1": ((ref["close_pos"] >= 0.8) & (c > o) & _run3(c > o)
                                  & (c > ph20)),
            "kdc_s4_3dn_15m_v1": ((c < pl20) & (c < o) & _run3(c < o) & (br >= 0.75)),
        }
    for version, theirs in expected.items():
        mine = spec_mask(_spec(version), feat)[tail]
        assert mine.sum() > 0, f"{version} 随机序列应至少命中一次，否则本对照无意义"
        assert np.array_equal(mine, theirs[tail]), version


@pytest.mark.asyncio
async def test_kdc_record_snapshot_and_payload_consistency(monkeypatch) -> None:
    """KDC 落表：cross_cost/vol_tier/entry_band 与 payload 同源；unknown 层不留 atr_pct。"""
    rows = _band_rows(97, 900_000, last=(100.0, 102.0, 99.9, 101.9))
    feat = compute_crv_features(_to_klines(rows, 900_000))
    idx = len(rows) - 1
    bar = rows[idx]
    target = int(bar["open_time"]) + 900_000
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(csd.time, "time", lambda: (target + 3_000) / 1000)
    pm15 = {"start_date": target, "up_price": 0.56, "down_price": 0.53,
            "updated_ts": target + 2_000}
    d = CrvShadowDetector(collector=None, pm_15m_latest=pm15, pm_5m_info={})
    payloads: list[dict] = []
    row = await d._record_signal(
        "15m", session, _spec("kdc_s1_96hi_15m_v1"), bar, feat, idx, payloads)
    assert row is not None
    snap = session.added[0].feature_snapshot
    assert snap["quote_cross_cost"] == pytest.approx(0.56 + 0.53 - 1.0)
    assert snap["entry_band"] == "[0.52,0.55)"      # DOWN 押注取 down 价 0.53
    assert snap["vol_tier"] == "unknown"            # idx<499 → 无 atr_pct
    assert "atr_pct" not in snap
    assert payloads[0]["cross_cost"] == pytest.approx(0.09)
    assert payloads[0]["vol_tier"] == "unknown"
    assert payloads[0]["direction"] == "DOWN"
    assert "q10_up" not in snap                      # 双时点第二档由 +10s 任务异步补


@pytest.mark.asyncio
async def test_kdc_record_high_vol_tier(monkeypatch) -> None:
    """末根巨幅 → ATR20 近500根分位 1.0 → vol_tier=high（实盘半仓依据）。"""
    rows = _band_rows(560, 900_000, last=(100.0, 104.0, 96.0, 101.9))
    feat = compute_crv_features(_to_klines(rows, 900_000))
    idx = len(rows) - 1
    assert csd.atr_pct_tail(feat["atr20"], idx) == 1.0
    target = int(rows[idx]["open_time"]) + 900_000
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _v: True)
    monkeypatch.setattr(csd.time, "time", lambda: (target + 3_000) / 1000)
    d = CrvShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    payloads: list[dict] = []
    await d._record_signal(
        "15m", session, _spec("kdc_s2_ext_15m_v1"), rows[idx], feat, idx, payloads)
    snap = session.added[0].feature_snapshot
    assert snap["vol_tier"] == "high"
    assert snap["atr_pct"] == 1.0
    assert payloads[0]["vol_tier"] == "high"         # 无报价缓存 → cross_cost 缺省 None
    assert payloads[0]["cross_cost"] is None


@pytest.mark.asyncio
async def test_kdc_q10_snapshot_writes_fields(monkeypatch) -> None:
    """+10s 二次报价：窗口对齐守卫通过 → feature_snapshot 增补 q10_* 且保留原字段。"""
    row = SimpleNamespace(id=7, feature_snapshot={"atr_pct": 0.9})
    session = _FakeSession(rows=[], scalar=row)
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    pm15 = {"start_date": T0, "up_price": 0.55, "down_price": 0.50,
            "updated_ts": T0 + 9_000}
    d = CrvShadowDetector(collector=None, pm_15m_latest=pm15, pm_5m_info={})
    await d._q10_snapshot(7, 0.0, T0, "15m")
    assert row.feature_snapshot == {"atr_pct": 0.9, "q10_up": 0.55,
                                    "q10_down": 0.50, "q10_ts": T0 + 9_000}


def test_kdc_channels_registry_and_bench() -> None:
    """四通道默认 OFF、护栏/优化字段/互斥组/基准表/影子账本注册齐备。"""
    from binance_predict.main import SHADOW_BENCH
    from binance_predict.services.live_channels import (
        LIVE_CHANNELS, exclusive_group, parse_channel_config)
    from binance_predict.services.shadow_execution_registry import SHADOW_VERSION_SPECS

    guards = {"kdc_s1_96hi_15m_v1": 0.64, "kdc_s2_ext_15m_v1": 0.61,
              "kdc_s3_3up_15m_v1": 0.61, "kdc_s4_3dn_15m_v1": 0.59}
    cfg = parse_channel_config()
    for v, g in guards.items():
        spec = LIVE_CHANNELS[v]
        assert spec.family == "kline_reversal" and spec.market_period == "15m"
        assert spec.auto_max_exec == g
        assert spec.spread_guard == 0.04 and spec.highvol_amount_factor == 0.5
        assert cfg[v].enabled is False            # 默认不开实盘
        assert v in SHADOW_BENCH and SHADOW_BENCH[v][1] is None
        assert v in SHADOW_VERSION_SPECS
    assert LIVE_CHANNELS["kdc_s4_3dn_15m_v1"].direction == "UP"
    assert exclusive_group("kdc_s1_96hi_15m_v1") == exclusive_group("crv_brk20_15m_v1")
    assert "kdc_s2_ext_15m_v1" in exclusive_group("crv_brk50_hi_15m_v1")
    assert "kdc_s4_3dn_15m_v1" in exclusive_group("crv_dnbrk24_15m_v1")
