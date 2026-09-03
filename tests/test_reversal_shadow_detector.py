"""反转形态影子检测器（P1/P2）测试：几何口径保真 + 判定正反例 + direction 结算 + 幂等 + 回补 + 隔离。

不触网络/真实 DB：collector/session 全用替身。rev_common 逐位对照测试依赖研究库
.pytest_tmp/rev_common.py（CI 无此产物 → skip，不阻塞）；核心判定/结算用例全部
自包含（固定合成 K 线序列 + 手算期望），保证 CI 无产物也能守住口径。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.reversal_shadow_detector as rsd
from binance_predict.services.reversal_shadow_detector import (
    BACKSCAN_BARS,
    BAR_MS_15M,
    REVERSAL_SHADOW_SPECS,
    REVERSAL_VERSIONS,
    ReversalShadowDetector,
    _to_klines,
    compute_geometry,
    compute_streak,
    evaluate_reversals,
    roll_mean,
)

ROOT = Path(__file__).resolve().parents[1]
REV_COMMON = ROOT / ".pytest_tmp" / "rev_common.py"
T0 = 1_700_000_000_000 // BAR_MS_15M * BAR_MS_15M

P1 = next(s for s in REVERSAL_SHADOW_SPECS if s["version"] == "rev_p1_v1")
P2 = next(s for s in REVERSAL_SHADOW_SPECS if s["version"] == "rev_p2_v1")


# ============================================================
# 移植特征微单元：compute_streak / roll_mean（手算精确断言）
# ============================================================

def test_compute_streak_rise_then_fall() -> None:
    """收盘环比连涨/连跌计数：涨 2 转跌 4 → [0,1,2,-1,-2,-3,-4]。"""
    c = np.array([10, 11, 12, 11, 10, 9, 8], dtype=np.float64)
    cont = np.array([False, True, True, True, True, True, True])
    assert list(compute_streak(c, cont)) == [0, 1, 2, -1, -2, -3, -4]


def test_compute_streak_resets_on_break() -> None:
    """cont=False（断点）→ streak 归零重计，与 rev_common 口径一致。"""
    c = np.array([10, 11, 12, 13, 12], dtype=np.float64)
    cont = np.array([False, True, True, False, True])
    # i=3 断点先归零再计 +1；若无断点应为 3
    assert list(compute_streak(c, cont)) == [0, 1, 2, 1, -1]


def test_roll_mean_trailing_w7() -> None:
    """trailing 均值（含当前根）：前 w-1 根 NaN，out[6]=4.0、out[7]=5.0。"""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    out = roll_mean(x, 7)
    assert np.isnan(out[:6]).all()
    assert out[6] == pytest.approx(4.0)
    assert out[7] == pytest.approx(5.0)


def test_w20_is_7_for_15m() -> None:
    """15m 的 w20 折算为 7（rev_common 同源窗口）。"""
    rows = _p1_rows()
    geo = compute_geometry(_to_klines(rows, BAR_MS_15M), BAR_MS_15M)
    assert geo["w20"] == 7


# ============================================================
# 合成序列构造（doji 累积 streak + 末根弱收盘）
# ============================================================

def _bar(i: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> dict:
    return {"open_time": T0 + i * BAR_MS_15M, "open": o, "high": h,
            "low": l, "close": c, "volume": v}


def _p1_rows(n: int = 12, last_vol: float = 1.2) -> list[dict]:
    """n-1 根 doji（o==c）收盘逐根下行（累积负 streak，dir_=0 非弱阴）+ 末根弱阴收（贴最低）。

    末根：o=前收、c=o−2、h=o+0.1、l=c−0.05 → close_pos≈0.023≤0.15、dir<0 → weak_close_dn。
    量：前 n-1 根 v=1.0、末根 v=last_vol（默认 1.2 → vol_ratio≈1.167∈[1.0,1.5)）。
    """
    rows, price = [], 100.0
    for i in range(n - 1):
        rows.append(_bar(i, price, price + 0.05, price - 0.05, price))
        price -= 0.5
    o = rows[-1]["close"]
    c = o - 2.0
    rows.append(_bar(n - 1, o, o + 0.1, c - 0.05, c, last_vol))
    return rows


def _p2_rows(n: int = 12) -> list[dict]:
    """n-1 根 doji 收盘逐根上行（累积正 streak）+ 末根弱阳收（贴最高）。

    末根：o=前收、c=o+2、h=c+0.05、l=o−0.1 → close_pos≈0.977≥0.85、dir>0 → weak_close_up。
    """
    rows, price = [], 100.0
    for i in range(n - 1):
        rows.append(_bar(i, price, price + 0.05, price - 0.05, price))
        price += 0.5
    o = rows[-1]["close"]
    c = o + 2.0
    rows.append(_bar(n - 1, o, c + 0.05, o - 0.1, c))
    return rows


def _geo(rows: list[dict]) -> dict:
    return compute_geometry(_to_klines(rows, BAR_MS_15M), BAR_MS_15M)


# ============================================================
# P1/P2 判定正反例（末根 n_tail=1）
# ============================================================

def test_p1_triggers_on_deep_decline_weak_down_close() -> None:
    rows = _p1_rows()
    geo = _geo(rows)
    # 末根几何自检（防合成数据失效）
    assert geo["streak"][-1] <= -4
    assert bool(geo["weak_close_dn"][-1]) is True
    assert bool(geo["vol_norm"][-1]) is True
    versions = {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}
    assert versions == {"rev_p1_v1"}


def test_p1_no_trigger_on_up_close() -> None:
    """末根改收阳（dir>0）但收盘仍低于前收（streak 续负）→ 非弱阴 → P1 不触发。"""
    rows = _p1_rows()
    prev_c = rows[-2]["close"]
    o = prev_c - 1.0
    c = o + 0.8                       # c<prev_c（streak 续负）、c>o（收阳）
    rows[-1] = _bar(11, o, c + 0.05, o - 0.05, c, 1.2)
    geo = _geo(rows)
    assert geo["streak"][-1] <= -4
    assert bool(geo["weak_close_dn"][-1]) is False
    assert "rev_p1_v1" not in {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}


def test_p1_no_trigger_shallow_streak() -> None:
    """仅末 2 根下行（streak=-2 > -4）→ P1 不触发（弱阴/量能均满足也不行）。"""
    rows, price = [], 100.0
    for i in range(10):
        rows.append(_bar(i, price, price + 0.05, price - 0.05, price))
    rows.append(_bar(10, 100.0, 100.05, 99.45, 99.5))       # c<prev → streak -1
    rows.append(_bar(11, 99.5, 99.55, 98.95, 99.0, 1.2))    # 弱阴但 streak -2
    geo = _geo(rows)
    assert geo["streak"][-1] == -2
    assert bool(geo["weak_close_dn"][-1]) is True
    assert "rev_p1_v1" not in {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}


def test_p1_no_trigger_vol_out_of_range() -> None:
    """末根量比≥1.5（vol_norm=False）→ 即便 streak/弱阴满足，P1 仍不触发。"""
    rows = _p1_rows(last_vol=2.0)     # vol_ratio≈1.75
    geo = _geo(rows)
    assert geo["streak"][-1] <= -4 and bool(geo["weak_close_dn"][-1]) is True
    assert bool(geo["vol_norm"][-1]) is False
    assert "rev_p1_v1" not in {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}


def test_p2_triggers_on_deep_rally_weak_up_close() -> None:
    rows = _p2_rows()
    geo = _geo(rows)
    assert geo["streak"][-1] >= 5
    assert bool(geo["weak_close_up"][-1]) is True
    versions = {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}
    assert versions == {"rev_p2_v1"}


def test_p2_no_trigger_on_down_close() -> None:
    """末根改收阴（dir<0）但收盘仍高于前收（streak 续正）→ 非弱阳 → P2 不触发。"""
    rows = _p2_rows()
    prev_c = rows[-2]["close"]
    o = prev_c + 1.0
    c = o - 0.8                       # c>prev_c（streak 续正）、c<o（收阴）
    rows[-1] = _bar(11, o, o + 0.05, c - 0.05, c)
    geo = _geo(rows)
    assert geo["streak"][-1] >= 5
    assert bool(geo["weak_close_up"][-1]) is False
    assert "rev_p2_v1" not in {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}


def test_p2_no_trigger_shallow_streak() -> None:
    """仅末 3 根上行（streak=+3 < 5）→ P2 不触发。"""
    rows, price = [], 100.0
    for i in range(9):
        rows.append(_bar(i, price, price + 0.05, price - 0.05, price))
    rows.append(_bar(9, 100.0, 100.55, 99.95, 100.5))       # streak +1
    rows.append(_bar(10, 100.5, 101.05, 100.45, 101.0))     # +2
    rows.append(_bar(11, 101.0, 101.55, 100.95, 101.5))     # +3，弱阳但 streak 不足
    geo = _geo(rows)
    assert geo["streak"][-1] == 3
    assert "rev_p2_v1" not in {h["spec"]["version"] for h in evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, 1)}


def test_backscan_tail_evaluates_multiple_bars() -> None:
    """n_tail=BACKSCAN_BARS 时，回补窗口内多根命中均被枚举（bar_offset 递增）。"""
    rows = _p1_rows(n=40)
    geo = _geo(rows)
    hits = evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, BACKSCAN_BARS)
    assert len(hits) >= 1
    assert all(h["spec"]["version"] == "rev_p1_v1" for h in hits)
    assert hits[-1]["idx"] == 39


# ============================================================
# rev_common 逐位对照（研究库在则硬闸门；CI 无产物 skip）
# ============================================================

def _load_rev_common():
    if not REV_COMMON.exists():
        pytest.skip("rev_common 研究库不在（CI 无 .pytest_tmp，跳过逐位对照）")
    spec = importlib.util.spec_from_file_location("rev_common_parity", str(REV_COMMON))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_walk_rows(n: int = 120, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for i in range(n):
        o = price
        c = o * (1 + rng.normal(0, 0.01))
        hi = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
        rows.append(_bar(i, o, hi, lo, c, float(rng.uniform(0.5, 3.0))))
        price = c
    return rows


def test_geometry_matches_rev_common_bitwise() -> None:
    """硬闸门：compute_geometry 与研究库 rev_common.compute_features 逐位一致。

    任何 w20/close_pos/vol_ratio/streak/weak_close/vol_norm 口径漂移都会在此暴露
    → 影子阶段失去意义（实时值对不上研究回测）。
    """
    rc = _load_rev_common()
    kl = _to_klines(_random_walk_rows(), BAR_MS_15M)
    geo = compute_geometry(kl, BAR_MS_15M)
    feat = rc.compute_features(kl, BAR_MS_15M)
    assert geo["w20"] == feat["meta"]["w20"] == 7
    for key in ("streak", "close_pos", "vol_ratio"):
        a = np.asarray(geo[key], dtype=np.float64)
        b = np.asarray(feat["num"][key], dtype=np.float64)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{key} NaN 位不一致"
        fin = ~np.isnan(a)
        assert np.array_equal(a[fin], b[fin]), f"{key} 有限值非逐位相等"
    for key in ("weak_close_dn", "weak_close_up", "vol_norm"):
        assert np.array_equal(geo[key].astype(bool), feat["bool"][key].astype(bool)), f"{key} 布尔不一致"


# ============================================================
# DB 编排：direction 结算 / 幂等 / 超时 / 回补（伪 session，不触真实库）
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


class _FakeCollector:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    async def fetch_recent_klines(self, timeframe: str, limit: int) -> list[dict]:
        return self._rows[-limit:]


def _pending(version: str, direction: str, target_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        version=version, direction=direction,
        signal_bar_start=target_start - BAR_MS_15M,
        target_bar_start=target_start, status="PENDING", win=None,
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


_TARGET = T0 + 40 * BAR_MS_15M


@pytest.mark.asyncio
async def test_settle_p1_win_on_green(monkeypatch) -> None:
    """P1（direction=UP）：次根收阳 → SETTLED win=True。"""
    session = _FakeSession(rows=[_pending("rev_p1_v1", "UP", _TARGET)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 102.0, "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True and sig.settle_outcome == "UP"
    assert session.committed


@pytest.mark.asyncio
async def test_settle_p1_lose_on_red(monkeypatch) -> None:
    """P1（UP）：次根收阴 → win=False。"""
    session = _FakeSession(rows=[_pending("rev_p1_v1", "UP", _TARGET)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 100.5, "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending(closed)
    assert session.rows[0].win is False and session.rows[0].settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_p2_win_on_red(monkeypatch) -> None:
    """P2（direction=DOWN）：次根收阴 → win=True（KREV 硬编码 UP 会误判，此处必须按 direction）。"""
    session = _FakeSession(rows=[_pending("rev_p2_v1", "DOWN", _TARGET)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 100.5, "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True and sig.settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_p2_lose_on_green(monkeypatch) -> None:
    """P2（DOWN）：次根收阳 → win=False（防跨 version 污染的关键反例）。"""
    session = _FakeSession(rows=[_pending("rev_p2_v1", "DOWN", _TARGET)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 102.0, "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending(closed)
    assert session.rows[0].win is False and session.rows[0].settle_outcome == "UP"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    """次根平盘（c==o）→ NOISE → EXPIRED，win=None（不进胜率统计）。"""
    session = _FakeSession(rows=[_pending("rev_p1_v1", "UP", _TARGET)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": _TARGET, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_record_signal_fields(monkeypatch) -> None:
    """不存在 → 落 PENDING，字段/快照正确（discovery_id 占位、direction、target=次根）。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    rows = _p1_rows()
    geo = _geo(rows)
    added = await d._record_signal(session, P1, rows[11], geo, 11)
    assert added is True and len(session.added) == 1
    obj = session.added[0]
    assert obj.version == "rev_p1_v1" and obj.discovery_id == "rev_p1"
    assert obj.direction == "UP" and obj.timeframe == "15m" and obj.status == "PENDING"
    assert obj.signal_bar_start == rows[11]["open_time"]
    assert obj.target_bar_start == rows[11]["open_time"] + BAR_MS_15M
    assert set(obj.feature_snapshot) == {"streak", "close_pos", "vol_ratio"}
    assert obj.feature_snapshot["streak"] <= -4


@pytest.mark.asyncio
async def test_record_signal_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    rows = _p1_rows()
    geo = _geo(rows)
    added = await d._record_signal(session, P1, rows[11], geo, 11)
    assert added is False and session.added == []


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    """目标根起点后 4h 仍未结算 → EXPIRED（数据缺失兜底）。"""
    import time as _time
    old = int(_time.time() * 1000) - 10 * 3_600_000
    session = _FakeSession(rows=[_pending("rev_p1_v1", "UP", old)])
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=None, pm_15m_latest={})
    await d._expire_stale_pending()
    assert session.rows[0].status == "EXPIRED" and session.committed


@pytest.mark.asyncio
async def test_backscan_records_and_advances_cursor(monkeypatch) -> None:
    """冷启动回补：拉满 WARMUP 根 → 末根命中 P1 落表 + 游标推进 + 触发计数 +1。"""
    rows = _p1_rows(n=rsd.WARMUP_BARS)
    session = _FakeSession(rows=[], scalar=None)
    monkeypatch.setattr(rsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ReversalShadowDetector(collector=_FakeCollector(rows), pm_15m_latest={})
    await d._backscan()
    assert len(session.added) == 1
    assert session.added[0].version == "rev_p1_v1"
    assert d._last_evaluated_bar == rows[-1]["open_time"]
    assert d._trigger_count == 1


# ============================================================
# 物理隔离 + 开关默认
# ============================================================

def test_versions_isolated_from_trading_path() -> None:
    """影子 version 绝不进入下单白名单/实盘通道（物理隔离红线）。"""
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    for v in REVERSAL_VERSIONS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号一致（零资金风险，开关仅作紧急停用制动力）。"""
    from binance_predict.config.settings import settings
    assert settings.reversal_shadow_enabled is True
