"""KREV K 线影子检测器测试：口径保真（硬闸门）+ 编排 + 物理隔离。

不触网络/真实 DB：collector/session 全用替身；口径保真测试依赖
output/klines_{15m,5m}_720d.csv（缺失时 skip，CI 无产物不阻塞）。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.kline_shadow_detector as ksd
from binance_predict.discovery.data import load_klines_csv
from binance_predict.discovery.features import atr_series, build_feature_matrix
from binance_predict.discovery.hypotheses import condition_mask, parse_condition
from binance_predict.discovery.targets import build_targets, seg_bounds
from binance_predict.services.kline_shadow_detector import (
    SHADOW_CONDITIONS,
    KlineShadowDetector,
    _to_klines,
    evaluate_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_15M = ROOT / "output" / "klines_15m_720d.csv"
CSV_5M = ROOT / "output" / "klines_5m_720d.csv"

# 冻结注册表逐字条件（与 SHADOW_CONDITIONS 同源，测试独立复核）
COND_A = ("dist_prior_low_atr_5 <= -0.0935059731 AND "
          "efficiency_5 >= 0.861468132 AND path3_all_down == True")
COND_B = ("range_pos_prior_5 <= -0.0467509994 AND "
          "efficiency_5 >= 0.861468132 AND path3_all_down == True")

# 冻结注册表命中计数（output/kline_discovery_15m_720d_v2/discovery_registry.csv）
# 2026-09-01 重冻结：720d K 线产物窗口滑动重生成，三段边界随 n 平移，计数按
# 当前窗口重放复核（重放脚本 .pytest_tmp/krev_refreeze_counts.py）；性能基准
# （胜率 64.2%/63.4%）仍锚定原发现注册表，不随窗口滑动重算。
REGISTRY_COUNTS = {
    "fd191c44fb5c36": (384, 130, 133),  # KREV-A：发现/验证/holdout
    "5c5e4c78ab4c3f": (382, 128, 130),  # KREV-B
}


# ============================================================
# 口径保真（硬闸门）：实时求值路径与 720d 离线产物逐位一致
# ============================================================

@pytest.fixture(scope="module")
def full_env():
    """720d 全量 K 线 + 特征矩阵 + reversal_1 目标（缺失产物则整组 skip）。"""
    if not (CSV_15M.exists() and CSV_5M.exists()):
        pytest.skip("720d K 线产物不存在（离线口径测试跳过）")
    kl15 = load_klines_csv(str(CSV_15M), 900_000)
    kl5 = load_klines_csv(str(CSV_5M), 300_000)
    fm = build_feature_matrix(kl15, 900_000, k5=kl5)
    tg = build_targets(kl15.t, kl15.o, kl15.h, kl15.l, kl15.c, kl15.cont,
                       [1], atr_series(kl15, 20))
    return kl15, kl5, fm, tg.items["reversal_1"]


def test_registry_replay_counts(full_env) -> None:
    """硬闸门：检测器同款求值函数重放冻结条件，三段命中计数 == 注册表。

    任何特征/条件/切分口径漂移都会让计数对不上 → 影子阶段失去意义。
    """
    kl15, _kl5, fm, ts = full_env
    n = len(kl15.t)
    i1, i2 = seg_bounds(n)
    for spec, cond in zip(SHADOW_CONDITIONS, (COND_A, COND_B)):
        assert spec["condition"] == cond, "SHADOW_CONDITIONS 与注册表原文不一致"
        mask = condition_mask(fm, parse_condition(cond)) & ts.valid
        counts = (int(mask[:i1].sum()), int(mask[i1:i2].sum()), int(mask[i2:].sum()))
        assert counts == REGISTRY_COUNTS[spec["discovery_id"]], (
            f"{spec['version']} 回放计数 {counts} != 注册表 "
            f"{REGISTRY_COUNTS[spec['discovery_id']]}"
        )


def test_short_window_feature_alignment(full_env) -> None:
    """硬闸门：短窗（40 根 15m）构建的末根特征 == 全量矩阵同位值逐位相等。

    实时侧每周期只拉 40 根历史，必须证明短窗不改变判定特征值。
    """
    kl15, kl5, fm, _ts = full_env
    W = ksd.WARMUP_BARS
    t_min = int(kl15.t[-W])
    kl15_short = _to_klines([
        {"open_time": int(kl15.t[i]), "open": float(kl15.o[i]), "high": float(kl15.h[i]),
         "low": float(kl15.l[i]), "close": float(kl15.c[i]), "volume": float(kl15.v[i])}
        for i in range(len(kl15) - W, len(kl15))
    ], 900_000)
    keep = kl5.t >= t_min
    kl5_short = _to_klines([
        {"open_time": int(kl5.t[i]), "open": float(kl5.o[i]), "high": float(kl5.h[i]),
         "low": float(kl5.l[i]), "close": float(kl5.c[i]), "volume": float(kl5.v[i])}
        for i in np.flatnonzero(keep)
    ], 300_000)
    fm_short = build_feature_matrix(kl15_short, 900_000, k5=kl5_short)
    for feat in ksd.SNAPSHOT_FEATURES:
        full_v, short_v = fm.cols[feat][-1], fm_short.cols[feat][-1]
        if isinstance(full_v, (bool, np.bool_)):
            assert bool(full_v) == bool(short_v), f"{feat} 短窗布尔不一致"
        else:
            assert np.isnan(full_v) == np.isnan(short_v), f"{feat} NaN 不一致"
            if not np.isnan(full_v):
                assert float(full_v) == float(short_v), f"{feat} 短窗值漂移"


# ============================================================
# 条件求值：合成数据触发 / 保守不触发
# ============================================================

def _flat_then_drop_rows(n_flat=34, n_drop=6) -> list[dict]:
    """34 根平盘 + 6 根匀速直跌（效率≈1、深破前低），满足 KREV-A 三条件。"""
    t0 = 1_700_000_000_000 // 900_000 * 900_000
    rows, price = [], 100.0
    for i in range(n_flat):
        rows.append({"open_time": t0 + i * 900_000, "open": price, "high": price + 0.05,
                     "low": price - 0.05, "close": price, "volume": 1.0})
    for i in range(n_drop):
        o = price
        c = price * 0.995
        rows.append({"open_time": t0 + (n_flat + i) * 900_000, "open": o,
                     "high": max(o, c) + 0.02, "low": min(o, c) - 0.02,
                     "close": c, "volume": 1.0})
        price = c
    return rows


def _sub_rows_for_last_cycle(rows: list[dict]) -> list[dict]:
    """末周期 3 根 5m 子根全收阴（path3_all_down=True）。"""
    last = rows[-1]
    t_c, o, c = last["open_time"], last["open"], last["close"]
    m1, m2 = o + (c - o) / 3, o + 2 * (c - o) / 3
    return [
        {"open_time": t_c, "open": o, "high": o + 0.01, "low": m1 - 0.01,
         "close": m1, "volume": 1.0},
        {"open_time": t_c + 300_000, "open": m1, "high": m1 + 0.01, "low": m2 - 0.01,
         "close": m2, "volume": 1.0},
        {"open_time": t_c + 600_000, "open": m2, "high": m2 + 0.01, "low": c - 0.01,
         "close": c, "volume": 1.0},
    ]


def _specs() -> list[dict]:
    return [{**s, "parts": parse_condition(s["condition"])} for s in SHADOW_CONDITIONS]


def test_synthetic_krev_a_triggers() -> None:
    rows = _flat_then_drop_rows()
    kl15 = _to_klines(rows, 900_000)
    kl5 = _to_klines(_sub_rows_for_last_cycle(rows), 300_000)
    fm = build_feature_matrix(kl15, 900_000, k5=kl5)
    # 先验证三特征方向正确（防合成数据本身失效）
    assert bool(fm.cols["path3_all_down"][-1]) is True
    assert float(fm.cols["dist_prior_low_atr_5"][-1]) <= -0.0935059731
    assert float(fm.cols["efficiency_5"][-1]) >= 0.861468132
    hits = evaluate_conditions(fm, _specs(), n_tail=1)
    versions = {h["spec"]["version"] for h in hits}
    assert "krev_a_v1" in versions


def test_no_5m_subbars_conservative_skip() -> None:
    """5m 缺失 → path3 特征不存在 → KeyError 保守不触发（而非异常抛出）。"""
    rows = _flat_then_drop_rows()
    kl15 = _to_klines(rows, 900_000)
    fm = build_feature_matrix(kl15, 900_000, k5=None)
    assert evaluate_conditions(fm, _specs(), n_tail=1) == []


def test_flat_market_no_trigger() -> None:
    """纯平盘市：无下跌无前低破位 → 零命中。"""
    rows = _flat_then_drop_rows(n_flat=40, n_drop=0)
    kl15 = _to_klines(rows, 900_000)
    kl5 = _to_klines(_sub_rows_for_last_cycle(rows[:1]), 300_000)
    fm = build_feature_matrix(kl15, 900_000, k5=kl5)
    assert evaluate_conditions(fm, _specs(), n_tail=1) == []


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


def _pending(target_start: int) -> SimpleNamespace:
    return SimpleNamespace(
        version="krev_a_v1", signal_bar_start=target_start - 900_000,
        target_bar_start=target_start, status="PENDING", win=None,
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


@pytest.mark.asyncio
async def test_settle_win_on_green_next_bar(monkeypatch) -> None:
    """次根收阳 → SETTLED win=True（回测 reversal_1 口径）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(ksd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = KlineShadowDetector(collector=None)
    closed = [{"open_time": target, "open": 100.0, "high": 102.0,
               "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True
    assert sig.settle_outcome == "UP" and sig.settle_open == 100.0
    assert session.committed


@pytest.mark.asyncio
async def test_settle_lose_on_red_next_bar(monkeypatch) -> None:
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(ksd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = KlineShadowDetector(collector=None)
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    """次根平盘 → NOISE → EXPIRED（方向无法判定，不进胜率统计）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending(target)])
    monkeypatch.setattr(ksd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = KlineShadowDetector(collector=None)
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending(closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_record_signal_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)  # 存在性查询命中
    monkeypatch.setattr(ksd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = KlineShadowDetector(collector=None)
    spec = d._specs[0]
    rows = _flat_then_drop_rows()
    fm = build_feature_matrix(_to_klines(rows, 900_000), 900_000)
    added = await d._record_signal(session, spec, rows[-1], fm, len(rows) - 1)
    assert added is False and session.added == []


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    """目标根起点后 4h 仍未结算 → EXPIRED。"""
    import time as _time
    old = int(_time.time() * 1000) - 10 * 3_600_000
    session = _FakeSession(rows=[_pending(old)])
    monkeypatch.setattr(ksd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = KlineShadowDetector(collector=None)
    await d._expire_stale_pending()
    assert session.rows[0].status == "EXPIRED" and session.committed


# ============================================================
# 物理隔离：影子版本绝不进入下单路径
# ============================================================

def test_versions_isolated_from_trading_path() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    versions = [s["version"] for s in SHADOW_CONDITIONS]
    for v in versions:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号（fake_breakout/misalignment/quote_edge）一致。

    影子模式零资金风险（只记录不下注、新表物理隔离），口径保真测试
    已全绿，无需人为显式开启；开关仅作紧急停用制动力。
    """
    from binance_predict.config.settings import settings
    assert settings.kline_shadow_enabled is True
    assert settings.kline_shadow_email_enabled is False
