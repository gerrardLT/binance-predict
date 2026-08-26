"""信号分析面板端点单元测试（2026-08-21）：
GET /api/chart/btc-klines（main.get_btc_klines）
GET /api/signals/analytics（main.get_signals_analytics）

不触网络/真实 DB：db 为 AsyncMock 替身，信号行用 SimpleNamespace，
collector.fetch_recent_klines 用 AsyncMock patch。

核心口径（与 scripts/local_shadow_full_analysis.py 审计版一致）：
- 盈亏平衡：x4_v1 = (q+0.01)/0.98（含溢价），其余 = q/0.98（无溢价）
- 场景 EV：审计口径逐笔现算（赢 0.98/(q+0.01)−1 截断[0.01,0.99] / 输 −1，
  q 按 side 取 entry_up/down_15m，缺失不计入）——不读落库期望 EV
- 周期切分：window_start/signal_time >= PUMP_TS_MS → pump，否则 pre
- 影子累计曲线按 window_start 升序，win 非空才计入
- 影子版本 = 冻结基准 ∪ 数据中出现版本（新版本 bench=None 容错）
- 场景信号过滤 = 排除 SceneParamVersion 中 SHADOW 版本名（ACTIVE 演进名视为正式）；
  端点共 3 次 db.execute（影子行 → SHADOW 版本名 → 场景行）
- K 线缓存键 = interval:档位（limit 归档到固定档），上游失败 10s 负缓存
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


def _shadow_row(**over) -> SimpleNamespace:
    """影子信号行替身：默认 x4_v1 已结算押 DOWN 赢。"""
    base = dict(
        version="x4_v1", status="SETTLED", window_start=1_000_000_000_000,
        win=True, ev_at_entry=0.3, entry_down_price=0.5, entry_up_price=0.49,
        direction="DOWN",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _scene_row(**over) -> SimpleNamespace:
    """场景信号行替身：默认 bull_exhaust side=high 结算 DOWN 赢，q 缺失。"""
    base = dict(
        pattern_type="bull_exhaust", pattern=None, version=None, status="SETTLED",
        signal_time=1_000_000_000_000, side="high", settle_outcome="DOWN",
        ev_at_entry=0.2, cumulative_winrate=1.0, cumulative_ev=0.2,
        entry_down_price_15m=None, entry_up_price_15m=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _make_db(shadow_rows, scene_rows) -> AsyncMock:
    db = AsyncMock()
    r1, r2, r3 = MagicMock(), MagicMock(), MagicMock()
    # 端点用指定列 SELECT，结果直接 .all()（不再 .scalars()）；
    # 3 次查询顺序：影子行 → SceneParamVersion SHADOW 版本名（默认空）→ 场景行
    r1.all.return_value = shadow_rows
    r2.all.return_value = []
    r3.all.return_value = scene_rows
    db.execute = AsyncMock(side_effect=[r1, r2, r3])
    return db


# ============================================================
# /api/chart/btc-klines
# ============================================================


@pytest.mark.asyncio
async def test_btc_klines_invalid_interval() -> None:
    """interval 不在 5m/15m/1h/4h/1d → 422（5m/15m 已开放给实盘对照图）。"""
    import binance_predict.main as m

    with pytest.raises(HTTPException) as exc:
        await m.get_btc_klines(interval="3m", limit=30)
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_btc_klines_cache_hit(monkeypatch) -> None:
    """TTL 内第二次调用命中缓存，不再请求 collector。"""
    import binance_predict.main as m

    m._btc_kline_cache.clear()
    m._btc_kline_fail.clear()
    fake = [{"open_time": i, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for i in range(12)]
    fetch = AsyncMock(return_value=fake)
    monkeypatch.setattr(m.collector, "fetch_recent_klines", fetch)

    out1 = await m.get_btc_klines(interval="1d", limit=30)
    out2 = await m.get_btc_klines(interval="1d", limit=30)

    assert out1["klines"] == fake and out2["klines"] == fake
    assert fetch.await_count == 1  # 第二次走缓存
    m._btc_kline_cache.clear()


# ============================================================
# /api/signals/analytics
# ============================================================


@pytest.mark.asyncio
async def test_analytics_shadow_curve_and_breakeven() -> None:
    """影子曲线累计正确；x4 盈亏平衡含溢价、momentum 无溢价（q=0.5 时 0.5204 vs 0.5102）。"""
    import binance_predict.main as m

    rows = [
        _shadow_row(window_start=1_000, win=True, ev_at_entry=0.3, entry_down_price=0.5),
        _shadow_row(window_start=2_000, win=False, ev_at_entry=-1.0, entry_down_price=0.5),
        _shadow_row(version="quote_momentum_v1", window_start=1_500, win=True,
                    ev_at_entry=0.4, entry_down_price=0.5),
    ]
    db = _make_db(rows, [])
    out = await m.get_signals_analytics(db)

    x4 = out["shadow"]["x4_v1"]
    assert x4["summary"]["n"] == 2
    assert x4["summary"]["win_rate"] == 0.5
    # 累计曲线：第 1 笔赢 → 1.0；第 2 笔输 → 0.5
    assert [p["cum_wr"] for p in x4["curve"]] == [1.0, 0.5]
    assert [p["cum_ev"] for p in x4["curve"]] == [0.3, -0.7]
    # x4 含溢价：(0.5+0.01)/0.98 ≈ 0.5204
    assert x4["summary"]["avg_breakeven"] == pytest.approx(0.51 / 0.98, abs=1e-9)

    mom = out["shadow"]["quote_momentum_v1"]
    # momentum 无溢价：0.5/0.98 ≈ 0.5102，与 x4 口径不同
    assert mom["summary"]["avg_breakeven"] == pytest.approx(0.5 / 0.98, abs=1e-9)
    assert mom["summary"]["bench_winrate"] == 0.799
    assert out["shadow"]["quote_contrarian_v1"]["summary"]["n"] == 0


@pytest.mark.asyncio
async def test_analytics_scene_curve_prefers_db_fields() -> None:
    """场景曲线优先 DB 落库 cumulative_winrate；胜负按 side→方向映射；EV 审计口径现算。"""
    import binance_predict.main as m

    rows = [
        # 赢（side=high 押 DOWN）：q=0.5 → EV = 0.98/0.51 − 1 ≈ 0.9216
        _scene_row(signal_time=1_000, settle_outcome="DOWN",
                   cumulative_winrate=1.0, entry_down_price_15m=0.5),
        # side=low → 赢的条件是 settle_outcome == "UP"；q=0.3 → EV = 0.98/0.31 − 1 ≈ 2.1613
        _scene_row(signal_time=2_000, side="low", settle_outcome="UP",
                   cumulative_winrate=1.0, entry_up_price_15m=0.3),
        # 第 3 笔输（side=high 但结算 UP）→ EV = −1
        _scene_row(signal_time=3_000, settle_outcome="UP",
                   cumulative_winrate=2 / 3, entry_down_price_15m=0.4),
    ]
    db = _make_db([], rows)
    out = await m.get_signals_analytics(db)

    blk = out["scene"]["bull_exhaust"]
    assert blk["summary"]["n"] == 3
    assert blk["summary"]["winrate"] == pytest.approx(2 / 3)
    assert blk["summary"]["bench_winrate"] == 0.644  # RESEARCH_WIN_RATES
    # 累计胜率优先 DB 落库字段
    assert [p["cum_wr"] for p in blk["curve"]] == [1.0, 1.0, round(2 / 3, 4)]
    # EV 为审计口径逐笔现算（落库 ev_at_entry/cumulative_ev 不参与）：
    # 0.9216 → +2.1613=3.0829 → −1=2.0829
    assert [p["cum_ev"] for p in blk["curve"]] == [0.9216, 3.0829, 2.0829]
    assert blk["summary"]["avg_ev"] == pytest.approx(
        ((0.98 / 0.51 - 1) + (0.98 / 0.31 - 1) - 1) / 3, rel=1e-6)
    assert blk["summary"]["cum_ev"] == pytest.approx(
        (0.98 / 0.51 - 1) + (0.98 / 0.31 - 1) - 1, abs=2e-4)


@pytest.mark.asyncio
async def test_analytics_scene_ev_q_missing_and_clamp() -> None:
    """场景 EV：q 缺失不计入；q+0.01 超 0.99 时截断（0.98/0.99−1）。"""
    import binance_predict.main as m

    rows = [
        # 赢但 q 缺失 → 不计入 EV（wins 仍计入）
        _scene_row(signal_time=1_000, settle_outcome="DOWN"),
        # 赢 + q=0.99 → 分母截断为 0.99：EV = 0.98/0.99 − 1 ≈ −0.0101
        _scene_row(signal_time=2_000, settle_outcome="DOWN", entry_down_price_15m=0.99),
    ]
    db = _make_db([], rows)
    out = await m.get_signals_analytics(db)

    blk = out["scene"]["bull_exhaust"]
    assert blk["summary"]["n"] == 2
    assert blk["summary"]["winrate"] == 1.0  # q 缺失不影响胜负
    assert blk["summary"]["avg_ev"] == pytest.approx(0.98 / 0.99 - 1, rel=1e-6)
    assert [p["cum_ev"] for p in blk["curve"]] == [0.0, -0.0101]


@pytest.mark.asyncio
async def test_analytics_regime_split_and_daily() -> None:
    """regime：>= PUMP_TS_MS 进 pump，按 UTC 日聚合。"""
    import binance_predict.main as m

    pump_ts = m.PUMP_TS_MS
    rows = [
        _shadow_row(window_start=pump_ts - 1_000, win=True),   # pre
        _shadow_row(window_start=pump_ts, win=False),          # pump 边界含入
    ]
    scene = [
        _scene_row(signal_time=pump_ts + 3_600_000, settle_outcome="DOWN"),  # pump 赢
    ]
    db = _make_db(rows, scene)
    out = await m.get_signals_analytics(db)

    phases = out["regime"]["phases"]
    assert phases["pre"] == {"n": 1, "wins": 1, "winrate": 1.0}
    assert phases["pump"] == {"n": 2, "wins": 1, "winrate": 0.5}
    assert out["pump_ts"] == pump_ts
    # 逐影子版本拆分：两笔 x4_v1 分落 pre（赢）/ pump（输）；场景信号不参与 by_version
    bv = out["regime"]["by_version"]
    assert bv == {"x4_v1": {
        "pre": {"n": 1, "wins": 1, "winrate": 1.0},
        "pump": {"n": 1, "wins": 0, "winrate": 0.0},
    }}
    # 按天：3 笔信号分布在两天（pump_ts 当天与次日取决于 +1h 是否跨日——均为同 1h 内）
    daily = out["regime"]["daily"]
    assert sum(d["n"] for d in daily) == 3
    assert all(0.0 <= (d["winrate"] or 0) <= 1.0 for d in daily)


@pytest.mark.asyncio
async def test_analytics_empty_db() -> None:
    """空数据：结构完整不崩溃（v2/v3 门禁版部署即入面板，bench=真实回测基准）。"""
    import binance_predict.main as m

    db = _make_db([], [])
    out = await m.get_signals_analytics(db)

    assert set(out["shadow"].keys()) == {
        "x4_v1", "quote_momentum_v1", "quote_contrarian_v1",
        "x4_v2", "quote_momentum_v2", "quote_contrarian_v2",
        "quote_contrarian_v3a", "quote_contrarian_v3b",
        "late_night_contrarian_v1"}
    for v, blk in out["shadow"].items():
        assert blk["summary"]["n"] == 0
        assert blk["summary"]["win_rate"] is None
        assert blk["curve"] == []
    # v2/v3 门禁版基准 = 2026-08-26 真实数据回测（local_shadow_v2v3_real_backtest.py）
    assert out["shadow"]["x4_v2"]["summary"]["bench_winrate"] == 0.553
    assert out["shadow"]["quote_contrarian_v3b"]["summary"]["bench_ev"] == 0.646
    assert out["shadow"]["quote_momentum_v2"]["summary"]["desc"].startswith("顺势v2")
    # 深夜变体：K 线代理回测只钉胜率基准，EV 基准因溢价口径差异留 None
    ln = out["shadow"]["late_night_contrarian_v1"]["summary"]
    assert ln["bench_winrate"] == 0.347 and ln["bench_ev"] is None
    assert ln["desc"].startswith("深夜逆势v1")
    assert out["scene"] == {}
    assert out["regime"]["phases"] == {}
    assert out["regime"]["by_version"] == {}
    assert out["regime"]["daily"] == []


def test_shadow_breakeven_x4_family_includes_premium() -> None:
    """盈亏平衡口径：x4 系（v1/v2）含溢 0.01，quote 系无溢价。"""
    import binance_predict.main as m

    assert m._shadow_breakeven("x4_v1", 0.5) == pytest.approx(0.51 / 0.98, abs=1e-9)
    assert m._shadow_breakeven("x4_v2", 0.5) == pytest.approx(0.51 / 0.98, abs=1e-9)
    assert m._shadow_breakeven("quote_momentum_v2", 0.5) == pytest.approx(0.5 / 0.98, abs=1e-9)
    assert m._shadow_breakeven("quote_contrarian_v2", 0.5) == pytest.approx(0.5 / 0.98, abs=1e-9)


@pytest.mark.asyncio
async def test_analytics_unknown_version_no_bench() -> None:
    """数据中出现冻结基准之外的版本：bench 为 None 容错，曲线正常。"""
    import binance_predict.main as m

    rows = [_shadow_row(version="new_v9", window_start=1_000, win=True)]
    db = _make_db(rows, [])
    out = await m.get_signals_analytics(db)

    blk = out["shadow"]["new_v9"]
    assert blk["summary"]["n"] == 1
    assert blk["summary"]["win_rate"] == 1.0
    assert blk["summary"]["bench_winrate"] is None
    assert blk["summary"]["bench_ev"] is None
    assert blk["summary"]["desc"] == ""
    assert [p["cum_wr"] for p in blk["curve"]] == [1.0]
    # 已知三版本仍在（含空 contrarian）
    assert "x4_v1" in out["shadow"] and "quote_contrarian_v1" in out["shadow"]


def _make_filter_db(shadow_versions=()) -> AsyncMock:
    """_official_scene_version_filter 专用替身：单次查询返回 SHADOW 版本名行。"""
    db = AsyncMock()
    r = MagicMock()
    r.all.return_value = [(v,) for v in shadow_versions]
    db.execute = AsyncMock(return_value=r)
    return db


@pytest.mark.asyncio
async def test_official_version_filter_no_shadow() -> None:
    """无 SHADOW 版本 → 恒真条件（全部信号视为正式，含 ACTIVE 演进名）。"""
    import binance_predict.main as m
    from sqlalchemy import true as sa_true

    clause = await m._official_scene_version_filter(_make_filter_db())
    assert isinstance(clause, type(sa_true()))


@pytest.mark.asyncio
async def test_official_version_filter_excludes_shadow_keeps_null() -> None:
    """有 SHADOW 版本 → 排除其版本名；NULL 显式保留（NOT IN 对 NULL 不为 TRUE）。"""
    import binance_predict.main as m

    clause = await m._official_scene_version_filter(
        _make_filter_db(["v2-shadow-20260821"]))
    sql = str(clause.compile(compile_kwargs={"literal_binds": True}))
    # 排除 SHADOW 名 + NULL OR 保留；ACTIVE 演进名（如 v1-20260816）不落入排除集
    assert "v2-shadow-20260821" in sql
    assert "IS NULL" in sql.upper()
    assert "NOT" in sql.upper()


@pytest.mark.asyncio
async def test_btc_klines_limit_tier_quantization(monkeypatch) -> None:
    """limit 归档固定档位：100/120 同档只打一次上游且按档位取数；limit 下限 clamp 10。"""
    import binance_predict.main as m

    m._btc_kline_cache.clear()
    m._btc_kline_fail.clear()
    fake = [{"open_time": i, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
            for i in range(20)]
    fetch = AsyncMock(return_value=fake)
    monkeypatch.setattr(m.collector, "fetch_recent_klines", fetch)

    out1 = await m.get_btc_klines(interval="1d", limit=100)  # 档位 120
    out2 = await m.get_btc_klines(interval="1d", limit=120)  # 同档 → 缓存命中

    assert len(out1["klines"]) == 20 and len(out2["klines"]) == 20
    assert fetch.await_count == 1
    assert fetch.await_args.args == ("1d", 120)  # 上游按档位取

    # limit=5 → clamp 到 10 → 档位 30（独立键，再打一次上游；返回尾部 10 根）
    out3 = await m.get_btc_klines(interval="4h", limit=5)
    assert len(out3["klines"]) == 10
    assert fetch.await_count == 2
    assert fetch.await_args.args == ("4h", 30)
    m._btc_kline_cache.clear()
    m._btc_kline_fail.clear()


@pytest.mark.asyncio
async def test_btc_klines_failure_negative_cache(monkeypatch) -> None:
    """上游返回空 → 记 10s 负缓存：窗口内再请求直接返回空且不打上游。"""
    import binance_predict.main as m

    m._btc_kline_cache.clear()
    m._btc_kline_fail.clear()
    fetch = AsyncMock(return_value=[])
    monkeypatch.setattr(m.collector, "fetch_recent_klines", fetch)

    out1 = await m.get_btc_klines(interval="1d", limit=30)
    out2 = await m.get_btc_klines(interval="1d", limit=30)

    assert out1["klines"] == [] and out2["klines"] == []
    assert fetch.await_count == 1  # 第二次命中负缓存，不打上游
    # 其它 interval 不受负缓存影响
    await m.get_btc_klines(interval="1h", limit=60)
    assert fetch.await_count == 2
    m._btc_kline_fail.clear()
