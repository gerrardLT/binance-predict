"""combo 组合条件影子检测器测试：口径保真（硬闸门）+ DB 编排 + 物理隔离。

不触网络/真实 DB：collector/session 全用替身；口径保真测试依赖
output/klines_15m_720d.csv（缺失时 skip，CI 无产物不阻塞）。

硬闸门口径来源（.pytest_tmp/mr_freeze_bench.py 冻结口径验证，720d 全样本）：
    combo_p1_v1  连阳3+∧周末∧乖离≥+0.3%          触发 490 / 赢 313（DOWN）
    combo_p2_v1  大实体(body_bp≥23.4633)∧连阳3+∧周末 触发 206 / 赢 137（DOWN）
    combo_p3_v1  贴1天高∧美盘∧7d涨≥4%             触发 176 / 赢 120（DOWN）
    combo_p4_v1  收低位∧周末∧RSI14≤25              触发 322 / 赢 202（UP）
    combo_p5_v1  近光脚∧周末∧RSI14≤25              触发 131 / 赢 90（UP）
检测器实时路径用自包含 compute_feature_tables（逐字照抄 grand_search_v2 口径，
大实体为 bp 冻结常数），本测试证明其重放上述计数（口径与离线冻结逐位一致），
且 700 根短窗末根特征 == 全量计算同位值。
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.combo_shadow_detector as csd
from binance_predict.services.combo_shadow_detector import (
    BIG_BODY_BP,
    COMBO_SHADOW_SPECS,
    COMBO_VERSIONS,
    ComboShadowDetector,
    compute_feature_tables,
    evaluate_bars,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_15M = ROOT / "output" / "klines_15m_720d.csv"

# 冻结全样本触发/win 计数（mr_freeze_bench.py，720d + WARM=700 掩码后）
FULL_COUNTS = {
    "combo_p1_v1": (490, 313),   # (触发, 赢)  direction=DOWN：赢=次根收阴
    "combo_p2_v1": (206, 137),
    "combo_p3_v1": (176, 120),
    "combo_p4_v1": (322, 202),   # direction=UP：赢=次根收阳
    "combo_p5_v1": (131, 90),
}


def _rows_from_csv() -> list[dict]:
    """720d CSV → collector 风格 dict 行（timestamp ISO → open_time ms，UTC）。"""
    rows: list[dict] = []
    with open(CSV_15M, newline="") as f:
        for r in csv.DictReader(f):
            dt = datetime.fromisoformat(r["timestamp"])
            rows.append({
                "open_time": int(dt.timestamp() * 1000),
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("volume") or 0.0),
            })
    return rows


@pytest.fixture(scope="module")
def full_rows() -> list[dict]:
    if not CSV_15M.exists():
        pytest.skip("720d 15m K 线产物不存在（离线口径测试跳过）")
    return _rows_from_csv()


# ============================================================
# 口径保真（硬闸门）：实时求值路径与 720d 离线冻结计数逐位一致
# ============================================================

def test_full_replay_counts(full_rows) -> None:
    """硬闸门：检测器 compute_feature_tables 重放 5 组合，全样本触发/win 数 == 冻结值。

    任何特征/阈值/口径漂移都会让计数对不上 → 影子阶段失去意义。
    """
    rows = full_rows
    n = len(rows)
    masks, _values = compute_feature_tables(rows)
    # 次根方向（j+1 收阳/收阴）；末根无次根不结算
    o = np.asarray([r["open"] for r in rows])
    c = np.asarray([r["close"] for r in rows])
    red = np.zeros(n, dtype=bool)      # red[j] = 次根收阴
    red[: n - 1] = c[1:] < o[1:]
    green = np.zeros(n, dtype=bool)    # green[j] = 次根收阳
    green[: n - 1] = c[1:] > o[1:]
    # 热身掩码：前 WARM 根不评估（与 grand_search_v2/mr_freeze_bench 的 ok[700:] 一致）
    ok = np.zeros(n, dtype=bool)
    ok[csd.WARMUP_BARS:] = True
    for spec in COMBO_SHADOW_SPECS:
        m = np.ones(n, dtype=bool)
        for feat in spec["features"]:
            m &= masks[feat]
        m &= ok
        trig = int(m.sum())
        assert trig == FULL_COUNTS[spec["version"]][0], (
            f"{spec['version']} 回放触发数 {trig} != 冻结 {FULL_COUNTS[spec['version']][0]}"
        )
        winmask = red if spec["direction"] == "DOWN" else green
        wins = int((m[:-1] & winmask[:-1]).sum())
        assert wins == FULL_COUNTS[spec["version"]][1], (
            f"{spec['version']} 回放 win 数 {wins} != 冻结 {FULL_COUNTS[spec['version']][1]}"
        )


def test_short_window_feature_alignment(full_rows) -> None:
    """硬闸门：短窗（700 根）末根特征 == 全量计算同位值（掩码逐位 + 数值 <1e-9）。

    实时侧每个新收盘根只拉 700 根历史，必须证明短窗不改变判定特征值
    （EMA20 从窗口头递推仅余 <1e-16 相对尾差；run/RSI/滚动高/7d 收益窗口内精确）。
    """
    rows = full_rows
    masks_full, values_full = compute_feature_tables(rows)
    w = csd.WARMUP_BARS
    masks_short, values_short = compute_feature_tables(rows[-w:])
    for name, col in masks_full.items():
        assert bool(col[-1]) == bool(masks_short[name][-1]), f"掩码 {name} 短窗末根漂移"
    for name, col in values_full.items():
        fv, sv = float(col[-1]), float(values_short[name][-1])
        assert np.isnan(fv) == np.isnan(sv), f"数值 {name} NaN 不一致"
        if not np.isnan(fv):
            assert abs(fv - sv) < 1e-9, f"数值 {name} 短窗末根漂移: {fv} vs {sv}"


def test_real_trigger_replay(full_rows) -> None:
    """端到端：每个组合取真实命中根 → 构造 700 根短窗 → evaluate_bars 捕获。"""
    rows = full_rows
    n = len(rows)
    masks, _values = compute_feature_tables(rows)
    ok = np.zeros(n, dtype=bool)
    ok[csd.WARMUP_BARS:] = True
    for spec in COMBO_SHADOW_SPECS:
        m = np.ones(n, dtype=bool)
        for feat in spec["features"]:
            m &= masks[feat]
        m &= ok
        fire = np.flatnonzero(m)
        assert len(fire) > 0, f"{spec['version']} 冻结条件在 720d 全样本零命中（口径异常）"
        j = int(fire[-1])  # 最近命中根（必有 >=WARMUP 的前置历史）
        short = rows[j - csd.WARMUP_BARS + 1: j + 1]
        masks_short, _ = compute_feature_tables(short)
        hits = evaluate_bars(masks_short, [spec], n_tail=1)
        assert any(h["spec"]["version"] == spec["version"] for h in hits), (
            f"{spec['version']} 真实命中根在短窗末根未被 evaluate_bars 捕获"
        )


# ============================================================
# 保守不触发（CSV 无关）：平盘市零命中
# ============================================================

def _flat_rows(n: int = csd.WARMUP_BARS) -> list[dict]:
    bar_ms = csd.BAR_MS
    t0 = 1_700_000_000_000 // bar_ms * bar_ms  # 2023-11-14 周二 UTC → 非周末
    return [
        {"open_time": t0 + i * bar_ms, "open": 100.0, "high": 100.05,
         "low": 99.95, "close": 100.0, "volume": 1.0}
        for i in range(n)
    ]


def test_flat_market_no_trigger() -> None:
    """纯平盘市：doji 重置连阳、乖离 0、7d 涨 0 → 五组合零命中（保守不触发）。"""
    masks, _ = compute_feature_tables(_flat_rows())
    assert evaluate_bars(masks, csd.COMBO_SHADOW_SPECS, n_tail=1) == [], "平盘市误触发"


def test_missing_feature_conservative_skip() -> None:
    """特征掩码缺失 → 该组合保守跳过（不异常抛出中断循环）。"""
    empty: dict[str, np.ndarray] = {}
    assert evaluate_bars(empty, csd.COMBO_SHADOW_SPECS, n_tail=1) == []


# ============================================================
# DB 编排：结算方向 / 幂等 / 超时 / 入场报价（伪 session，不触真实库）
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


def _pending(version: str, target_start: int, direction: str = "DOWN") -> SimpleNamespace:
    return SimpleNamespace(
        version=version, signal_bar_start=target_start - csd.BAR_MS,
        target_bar_start=target_start, status="PENDING", win=None, direction=direction,
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


@pytest.mark.asyncio
async def test_settle_down_wins_on_red_next_bar(monkeypatch) -> None:
    """direction=DOWN：次根收阴 → SETTLED win=True（P1~P3 口径）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("combo_p1_v1", target, "DOWN")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True
    assert sig.settle_outcome == "DOWN" and sig.settle_open == 100.0
    assert session.committed


@pytest.mark.asyncio
async def test_settle_down_loses_on_green_next_bar(monkeypatch) -> None:
    """direction=DOWN：次根收阳 → SETTLED win=False。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("combo_p3_v1", target, "DOWN")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": target, "open": 100.0, "high": 102.0,
               "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "UP"


@pytest.mark.asyncio
async def test_settle_up_wins_on_green_next_bar(monkeypatch) -> None:
    """direction=UP：次根收阳 → SETTLED win=True（P4/P5 口径）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("combo_p4_v1", target, "UP")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": target, "open": 100.0, "high": 102.0,
               "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True and sig.settle_outcome == "UP"


@pytest.mark.asyncio
async def test_settle_up_loses_on_red_next_bar(monkeypatch) -> None:
    """direction=UP：次根收阴 → SETTLED win=False。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("combo_p5_v1", target, "UP")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    """次根平盘 → NOISE → EXPIRED（方向无法判定，不进胜率统计）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("combo_p1_v1", target, "DOWN")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_record_signal_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)  # 存在性查询命中
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    spec = d._specs[0]
    rows = _flat_rows()
    _masks, values = compute_feature_tables(rows)
    added = await d._record_signal(session, spec, rows[-1], values, len(rows) - 1)
    assert added is False and session.added == []


@pytest.mark.asyncio
async def test_record_signal_captures_aligned_entry_quote(monkeypatch) -> None:
    """目标窗对齐的实时报价缓存 → _record_signal 落库 entry_up/down_price + entry_quote_ts。"""
    session = _FakeSession(scalar=None)  # 不存在 → 落行
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    rows = _flat_rows()
    sig_bar = rows[-1]
    target = int(sig_bar["open_time"]) + csd.BAR_MS
    cache = {"start_date": target, "up_price": 0.62, "down_price": 0.41,
             "updated_ts": target + 38_000}  # 开盘后 38s，近开盘守卫内
    d = ComboShadowDetector(collector=None, pm_15m_latest=cache)
    _masks, values = compute_feature_tables(rows)
    added = await d._record_signal(session, d._specs[0], sig_bar, values, len(rows) - 1)
    assert added is True and len(session.added) == 1
    row = session.added[0]
    assert row.entry_up_price == 0.62 and row.entry_down_price == 0.41
    assert row.entry_quote_ts == target + 38_000
    assert row.direction == "DOWN" and row.timeframe == "15m"


@pytest.mark.asyncio
async def test_record_signal_skips_misaligned_quote(monkeypatch) -> None:
    """缓存窗口与目标窗不对齐（冷启动回补/停在下一窗）→ entry 报价留空，EV 不计（保守）。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    rows = _flat_rows()
    sig_bar = rows[-1]
    target = int(sig_bar["open_time"]) + csd.BAR_MS
    cache = {"start_date": target + csd.BAR_MS, "up_price": 0.62, "down_price": 0.41,
             "updated_ts": target + 38_000}  # 缓存停在下一窗 → 不对齐
    d = ComboShadowDetector(collector=None, pm_15m_latest=cache)
    _masks, values = compute_feature_tables(rows)
    added = await d._record_signal(session, d._specs[0], sig_bar, values, len(rows) - 1)
    assert added is True
    row = session.added[0]
    assert row.entry_up_price is None and row.entry_down_price is None
    assert row.entry_quote_ts is None


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    """目标根起点后超时仍未结算 → EXPIRED。"""
    import time as _time
    old = int(_time.time() * 1000) - 10 * 3_600_000
    session = _FakeSession(rows=[_pending("combo_p1_v1", old, "DOWN")])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=None, pm_15m_latest={})
    await d._expire_stale_pending()
    assert session.rows[0].status == "EXPIRED" and session.committed


# ============================================================
# 轻探测轮询：无新收盘根不拉重窗（fetch 次数可观测）
# ============================================================

class _FakeCollector:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, int]] = []

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        self.calls.append((interval, limit))
        out = self._rows[-limit:]
        return out


@pytest.mark.asyncio
async def test_poll_probe_skips_heavy_fetch_without_new_bar(monkeypatch) -> None:
    """水位已是最新收盘根 → 本轮仅 limit=2 探测，不拉 700 根重窗（减重路径）。"""
    rows = _flat_rows()
    collector = _FakeCollector(rows)
    monkeypatch.setattr(csd, "async_session_factory",
                        lambda: _FakeSessionCtx(_FakeSession()))
    d = ComboShadowDetector(collector=collector, pm_15m_latest={})
    d._last_evaluated_bar = int(rows[-1]["open_time"])  # 水位=最新
    await d._poll_once()
    assert collector.calls == [("15m", csd.PROBE_BARS)], (
        f"无新根时不得拉重窗，实际调用 {collector.calls}"
    )


@pytest.mark.asyncio
async def test_poll_new_bar_triggers_full_evaluation(monkeypatch) -> None:
    """新收盘根出现 → 探测后拉 700 根 → 评估 + 推进水位（平盘市零命中也推进）。"""
    rows = _flat_rows()
    collector = _FakeCollector(rows)
    session = _FakeSession()
    monkeypatch.setattr(csd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = ComboShadowDetector(collector=collector, pm_15m_latest={})
    d._last_evaluated_bar = int(rows[-2]["open_time"])  # 水位落后一根
    await d._poll_once()
    assert ("15m", csd.WARMUP_BARS) in collector.calls, "新根出现必须拉重窗求值"
    assert d._last_evaluated_bar == int(rows[-1]["open_time"]), "水位未推进到最新收盘根"


# ============================================================
# 物理隔离 + 开关 + spec 自洽
# ============================================================

def test_versions_isolated_from_trading_path() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    for v in COMBO_VERSIONS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号一致（record-only 零资金风险，口径保真测试已全绿）。"""
    from binance_predict.config.settings import settings
    assert settings.combo_shadow_enabled is True


def test_specs_self_consistent() -> None:
    """spec 自洽：版本名/方向/特征键合法，快照键在 values 表内，DB 列宽不超。"""
    val_keys = {"run_len", "bias", "body_bp", "dist_1dh_bp", "r7d_pct", "pos", "lower", "rsi"}
    mask_keys = {"up3", "weekend", "bias_pos", "big_body", "near1dh",
                 "us_sess", "r7d_up", "pos_low", "no_lower", "rsi_lt25"}
    for s in COMBO_SHADOW_SPECS:
        assert s["direction"] in ("UP", "DOWN")
        assert len(s["version"]) <= 24 and len(s["discovery_id"]) <= 16, "超出 DB 列宽"
        assert set(s["features"]) <= mask_keys, f"{s['version']} 特征键不在掩码表"
        assert set(s["snapshot_features"]) <= val_keys, f"{s['version']} 快照键不在数值表"
        assert s["condition_text"], "condition_text 不得为空（审计口径）"
    # 冻结常数全精度（勿手抄渲染值）
    assert BIG_BODY_BP == 23.46330248119974
