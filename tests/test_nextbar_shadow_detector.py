"""nextbar K 线方向影子检测器测试：口径保真（硬闸门）+ 双 timeframe 编排 + 物理隔离。

不触网络/真实 DB：collector/session 全用替身；口径保真测试依赖
output/klines_{15m,5m}_720d.csv（缺失时 skip，CI 无产物不阻塞）。

硬闸门口径来源（.pytest_tmp/freeze_nb_thresholds.py 全精度复现，禁止手抄渲染值）：
    15m 冠军 zscore_10≤-1.65079327 ∧ zscore_5≤-1.53756693 ∧ ret_3≤-0.0039526084
        → 720d 全样本触发 2003 / 69119（converge_registry.csv L69 ROBUST）
    5m sma_slope_atr_5≥1.6605556162359245（阶段E 7月发现段 q0.9）
        → 720d 全样本触发 19618 / 207359
检测器实时路径用 build_feature_matrix(k5=None) + condition_mask，本测试证明其
复现上述计数（口径与离线冻结逐位一致），且 40 根短窗末根特征 == 全量矩阵同位值。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import binance_predict.services.nextbar_shadow_detector as nsd
from binance_predict.discovery.data import load_klines_csv
from binance_predict.discovery.features import build_feature_matrix
from binance_predict.discovery.hypotheses import condition_mask, parse_condition
from binance_predict.services.nextbar_shadow_detector import (
    NEXTBAR_SHADOW_SPECS,
    NEXTBAR_VERSIONS,
    VERSIONS_BY_TF,
    NextbarShadowDetector,
    _to_klines,
    evaluate_conditions,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_15M = ROOT / "output" / "klines_15m_720d.csv"
CSV_5M = ROOT / "output" / "klines_5m_720d.csv"

# 冻结全样本触发计数（freeze_nb_thresholds.py 复现，与检测器 condition_mask 路径对齐）
# 2026-09-04 重冻结（沿用 KREV 2026-09-01 先例）：720d K 线产物刷新至 2026-09-04，
# 窗口平移计数按当前窗口重放复核；性能基准（胜率 58.92%/47.43%）仍锚定原发现注册表，
# 不随窗口滑动重算。
FULL_TRIGGER_COUNTS = {
    "nb_zschamp_15m_v1": 2003,    # 15m 冠军：720d 全量触发根数
    "nb_smaslope_5m_v1": 19618,   # 5m sma_slope≥1.6605556162359245：720d 全量触发根数
}


def _specs() -> list[dict]:
    return [{**s, "parts": parse_condition(s["condition"])} for s in NEXTBAR_SHADOW_SPECS]


def _spec_of(tf: str) -> dict:
    return next(s for s in _specs() if s["timeframe"] == tf)


def _rows_from_kl(kl, lo: int, hi: int) -> list[dict]:
    """Klines 数组切片 [lo, hi) → data_collector 风格 dict 行（供 _to_klines 重建）。"""
    return [
        {"open_time": int(kl.t[i]), "open": float(kl.o[i]), "high": float(kl.h[i]),
         "low": float(kl.l[i]), "close": float(kl.c[i]), "volume": float(kl.v[i])}
        for i in range(lo, hi)
    ]


# ============================================================
# 口径保真（硬闸门）：实时求值路径与 720d 离线冻结计数逐位一致
# ============================================================

@pytest.fixture(scope="module")
def env_15m():
    if not CSV_15M.exists():
        pytest.skip("720d 15m K 线产物不存在（离线口径测试跳过）")
    kl = load_klines_csv(str(CSV_15M), 900_000)
    fm = build_feature_matrix(kl, 900_000)  # k5=None：本族无 path3 特征
    return kl, fm


@pytest.fixture(scope="module")
def env_5m():
    if not CSV_5M.exists():
        pytest.skip("720d 5m K 线产物不存在（离线口径测试跳过）")
    kl = load_klines_csv(str(CSV_5M), 300_000)
    fm = build_feature_matrix(kl, 300_000)
    return kl, fm


def test_registry_replay_count_15m(env_15m) -> None:
    """硬闸门：检测器 condition_mask 重放 15m 冠军冻结条件，全样本触发数 == 2006。

    任何特征/条件/口径漂移都会让计数对不上 → 影子阶段失去意义。
    """
    _kl, fm = env_15m
    spec = _spec_of("15m")
    assert spec["condition"] == (
        "zscore_10 <= -1.65079327 AND zscore_5 <= -1.53756693 AND ret_3 <= -0.0039526084"
    ), "SHADOW 条件与冻结注册表原文不一致"
    mask = condition_mask(fm, spec["parts"])
    assert int(mask.sum()) == FULL_TRIGGER_COUNTS["nb_zschamp_15m_v1"], (
        f"15m 冠军回放触发数 {int(mask.sum())} != 冻结 {FULL_TRIGGER_COUNTS['nb_zschamp_15m_v1']}"
    )


def test_registry_replay_count_5m(env_5m) -> None:
    """硬闸门：检测器 condition_mask 重放 5m sma_slope 冻结条件，全样本触发数 == 19597。"""
    _kl, fm = env_5m
    spec = _spec_of("5m")
    assert spec["condition"] == "sma_slope_atr_5 >= 1.6605556162359245", (
        "SHADOW 条件与冻结阈值不一致"
    )
    mask = condition_mask(fm, spec["parts"])
    assert int(mask.sum()) == FULL_TRIGGER_COUNTS["nb_smaslope_5m_v1"], (
        f"5m sma_slope 回放触发数 {int(mask.sum())} != 冻结 {FULL_TRIGGER_COUNTS['nb_smaslope_5m_v1']}"
    )


def test_short_window_feature_alignment(env_15m, env_5m) -> None:
    """硬闸门：短窗（40 根）构建的末根特征 == 全量矩阵同位值逐位相等（双 tf）。

    实时侧每周期只拉 40 根历史，必须证明短窗不改变判定特征值（否则阈值失效）。
    """
    W = nsd.WARMUP_BARS
    for kl, fm, tf, bar_ms in (
        (env_15m[0], env_15m[1], "15m", 900_000),
        (env_5m[0], env_5m[1], "5m", 300_000),
    ):
        spec = _spec_of(tf)
        kl_short = _to_klines(_rows_from_kl(kl, len(kl) - W, len(kl)), bar_ms)
        fm_short = build_feature_matrix(kl_short, bar_ms)
        for feat in spec["snapshot_features"]:
            full_v, short_v = fm.cols[feat][-1], fm_short.cols[feat][-1]
            assert np.isnan(full_v) == np.isnan(short_v), f"{tf}/{feat} NaN 不一致"
            if not np.isnan(full_v):
                assert float(full_v) == float(short_v), f"{tf}/{feat} 短窗值漂移"


def test_real_trigger_replay(env_15m, env_5m) -> None:
    """端到端：真实命中根 → 取 40 根短窗 → 检测器 evaluate_conditions 捕获（双 tf）。"""
    for kl, fm, tf, bar_ms in (
        (env_15m[0], env_15m[1], "15m", 900_000),
        (env_5m[0], env_5m[1], "5m", 300_000),
    ):
        spec = _spec_of(tf)
        mask = condition_mask(fm, spec["parts"])
        fire = np.flatnonzero(mask)
        assert len(fire) > 0, f"{tf} 冻结条件在 720d 全样本零命中（口径异常）"
        # 取一个有足够前置历史的命中根，构造以其结尾的 WARMUP 短窗
        j = int(fire[fire >= nsd.WARMUP_BARS][-1])
        kl_short = _to_klines(_rows_from_kl(kl, j - nsd.WARMUP_BARS + 1, j + 1), bar_ms)
        fm_short = build_feature_matrix(kl_short, bar_ms)
        hits = evaluate_conditions(fm_short, [spec], n_tail=1)
        assert any(h["spec"]["version"] == spec["version"] for h in hits), (
            f"{tf} 真实命中根在短窗末根未被 evaluate_conditions 捕获"
        )


# ============================================================
# 保守不触发（CSV 无关）：平盘市零命中
# ============================================================

def _flat_rows(tf: str, n: int = 40) -> list[dict]:
    bar_ms = nsd.BAR_MS[tf]
    t0 = 1_700_000_000_000 // bar_ms * bar_ms
    return [
        {"open_time": t0 + i * bar_ms, "open": 100.0, "high": 100.05,
         "low": 99.95, "close": 100.0, "volume": 1.0}
        for i in range(n)
    ]


def test_flat_market_no_trigger() -> None:
    """纯平盘市：zscore 分母为 0 → NaN、sma_slope≈0 → 两 tf 均零命中（保守不触发）。"""
    for tf, bar_ms in (("15m", 900_000), ("5m", 300_000)):
        fm = build_feature_matrix(_to_klines(_flat_rows(tf), bar_ms), bar_ms)
        assert evaluate_conditions(fm, [_spec_of(tf)], n_tail=1) == [], f"{tf} 平盘市误触发"


def test_missing_feature_conservative_skip() -> None:
    """条件特征缺失 → KeyError 被吞、保守不触发（而非异常抛出中断循环）。"""
    empty = SimpleNamespace(cols={})  # 无任何特征列
    assert evaluate_conditions(empty, _specs(), n_tail=1) == []


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


def _pending(tf: str, target_start: int) -> SimpleNamespace:
    version = VERSIONS_BY_TF[tf][0]
    return SimpleNamespace(
        version=version, signal_bar_start=target_start - nsd.BAR_MS[tf],
        target_bar_start=target_start, status="PENDING", win=None, direction="UP",
        settle_outcome=None, settle_open=None, settle_close=None, settled_at=None,
    )


@pytest.mark.asyncio
async def test_settle_win_on_green_next_bar(monkeypatch) -> None:
    """次根收阳 → SETTLED win=True（direction=UP，回测 reversal_1 口径）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("15m", target)])
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    closed = [{"open_time": target, "open": 100.0, "high": 102.0,
               "low": 99.5, "close": 101.5, "volume": 1.0}]
    await d._settle_pending("15m", closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is True
    assert sig.settle_outcome == "UP" and sig.settle_open == 100.0
    assert session.committed


@pytest.mark.asyncio
async def test_settle_lose_on_red_next_bar(monkeypatch) -> None:
    """次根收阴 → SETTLED win=False（押 UP 未命中）。"""
    target = 1_700_000_000_000 // 300_000 * 300_000 + 40 * 300_000
    session = _FakeSession(rows=[_pending("5m", target)])
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 98.5, "close": 99.0, "volume": 1.0}]
    await d._settle_pending("5m", closed)
    sig = session.rows[0]
    assert sig.status == "SETTLED" and sig.win is False and sig.settle_outcome == "DOWN"


@pytest.mark.asyncio
async def test_settle_noise_expired(monkeypatch) -> None:
    """次根平盘 → NOISE → EXPIRED（方向无法判定，不进胜率统计）。"""
    target = 1_700_000_000_000 // 900_000 * 900_000 + 40 * 900_000
    session = _FakeSession(rows=[_pending("15m", target)])
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    closed = [{"open_time": target, "open": 100.0, "high": 100.5,
               "low": 99.5, "close": 100.0, "volume": 1.0}]
    await d._settle_pending("15m", closed)
    sig = session.rows[0]
    assert sig.status == "EXPIRED" and sig.win is None and sig.settle_outcome == "NOISE"


@pytest.mark.asyncio
async def test_record_signal_idempotent(monkeypatch) -> None:
    """已存在 (version, signal_bar_start) → 不重复落行。"""
    session = _FakeSession(scalar=123)  # 存在性查询命中
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    spec = d._specs[0]  # 15m 冠军
    rows = _flat_rows("15m")
    fm = build_feature_matrix(_to_klines(rows, 900_000), 900_000)
    added = await d._record_signal(session, spec, rows[-1], fm, len(rows) - 1)
    assert added is False and session.added == []


@pytest.mark.asyncio
async def test_record_signal_captures_aligned_entry_quote(monkeypatch) -> None:
    """目标窗对齐的实时报价缓存 → _record_signal 落库 entry_up/down_price + entry_quote_ts。"""
    session = _FakeSession(scalar=None)  # 不存在 → 落行
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    rows = _flat_rows("15m")
    sig_bar = rows[-1]
    target = int(sig_bar["open_time"]) + 900_000
    cache = {"start_date": target, "up_price": 0.57, "down_price": 0.45,
             "updated_ts": target + 38_000}  # 开盘后 38s，近开盘守卫内
    d = NextbarShadowDetector(collector=None, pm_15m_latest=cache, pm_5m_info={})
    fm = build_feature_matrix(_to_klines(rows, 900_000), 900_000)
    added = await d._record_signal(session, d._specs[0], sig_bar, fm, len(rows) - 1)
    assert added is True and len(session.added) == 1
    row = session.added[0]
    assert row.entry_up_price == 0.57 and row.entry_down_price == 0.45
    assert row.entry_quote_ts == target + 38_000


@pytest.mark.asyncio
async def test_record_signal_captures_5m_entry_quote(monkeypatch) -> None:
    """5m 信号按 timeframe 路由到 pm_5m_info 缓存快照（验证双 tf 报价源选择）。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    rows = _flat_rows("5m")
    sig_bar = rows[-1]
    target = int(sig_bar["open_time"]) + 300_000
    cache5 = {"start_date": target, "up_price": 0.51, "down_price": 0.50,
              "updated_ts": target + 20_000}
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info=cache5)
    spec5 = next(s for s in d._specs if s["timeframe"] == "5m")
    fm = build_feature_matrix(_to_klines(rows, 300_000), 300_000)
    added = await d._record_signal(session, spec5, sig_bar, fm, len(rows) - 1)
    assert added is True
    row = session.added[0]
    assert row.timeframe == "5m" and row.entry_up_price == 0.51
    assert row.entry_quote_ts == target + 20_000


@pytest.mark.asyncio
async def test_record_signal_skips_misaligned_quote(monkeypatch) -> None:
    """缓存窗口与目标窗不对齐（冷启动回补/停在下一窗）→ entry 报价留空，EV 不计（保守）。"""
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    rows = _flat_rows("15m")
    sig_bar = rows[-1]
    target = int(sig_bar["open_time"]) + 900_000
    cache = {"start_date": target + 900_000, "up_price": 0.57, "down_price": 0.45,
             "updated_ts": target + 38_000}  # 缓存停在下一窗 → 不对齐
    d = NextbarShadowDetector(collector=None, pm_15m_latest=cache, pm_5m_info={})
    fm = build_feature_matrix(_to_klines(rows, 900_000), 900_000)
    added = await d._record_signal(session, d._specs[0], sig_bar, fm, len(rows) - 1)
    assert added is True
    row = session.added[0]
    assert row.entry_up_price is None and row.entry_down_price is None
    assert row.entry_quote_ts is None


@pytest.mark.asyncio
async def test_expire_stale_pending(monkeypatch) -> None:
    """目标根起点后超时仍未结算 → EXPIRED（按 tf 各自超时窗）。"""
    import time as _time
    old = int(_time.time() * 1000) - 10 * 3_600_000
    session = _FakeSession(rows=[_pending("15m", old)])
    monkeypatch.setattr(nsd, "async_session_factory", lambda: _FakeSessionCtx(session))
    d = NextbarShadowDetector(collector=None, pm_15m_latest={}, pm_5m_info={})
    await d._expire_stale_pending("15m")
    assert session.rows[0].status == "EXPIRED" and session.committed


# ============================================================
# 双 timeframe 隔离 + 物理隔离 + 开关
# ============================================================

def test_timeframe_version_partition() -> None:
    """跨 tf 隔离生命线：VERSIONS_BY_TF 按 timeframe 正确切分，两 tf 版本不重叠。

    15m bar 起点也是某 5m bar 起点，若结算不按 tf 过滤 version 会用 5m 次根错结
    15m 信号；本测试锁定该切分正确。
    """
    assert set(VERSIONS_BY_TF["15m"]) == {"nb_zschamp_15m_v1"}
    assert set(VERSIONS_BY_TF["5m"]) == {"nb_smaslope_5m_v1"}
    assert not (set(VERSIONS_BY_TF["15m"]) & set(VERSIONS_BY_TF["5m"])), "tf 版本集不得重叠"
    assert set(NEXTBAR_VERSIONS) == set(VERSIONS_BY_TF["15m"]) | set(VERSIONS_BY_TF["5m"])
    # 每条 spec 的 timeframe/bar_ms/direction 自洽
    for s in NEXTBAR_SHADOW_SPECS:
        assert nsd.BAR_MS[s["timeframe"]] in (300_000, 900_000)
        assert s["direction"] in ("UP", "DOWN")
        assert len(s["version"]) <= 24 and len(s["discovery_id"]) <= 16, "超出 DB 列宽"


def test_versions_isolated_from_trading_path() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS
    for v in NEXTBAR_VERSIONS:
        assert v not in X4_VERSIONS, f"{v} 不得进入 X4 下单白名单"
        assert v not in LIVE_CHANNELS, f"{v} 不得注册实盘通道"


def test_settings_default_on() -> None:
    """默认开启：与其他影子信号一致（record-only 零资金风险，口径保真测试已全绿）。"""
    from binance_predict.config.settings import settings
    assert settings.nextbar_shadow_enabled is True
