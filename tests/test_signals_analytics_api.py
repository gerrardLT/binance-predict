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
  端点共 5 次 db.execute（影子行 → KREV 行 → pattern 行 → SHADOW 版本名 → 场景行）
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


def _krev_row(**over) -> SimpleNamespace:
    """KREV 影子行替身（kline_shadow_signals）：无报价/无 EV，direction 恒 UP。"""
    base = dict(
        version="krev_a_v1", window_start=1_000_000_000_000,
        win=True, ev_at_entry=None, entry_down_price=None,
        entry_up_price=None, direction="UP",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _pattern_row(**over) -> SimpleNamespace:
    """HM 触价影子行替身（pattern_shadow_signals）：仅 TOUCHED 进 SETTLED，
    direction 恒 DOWN，入场报价=触及时刻 DOWN 真实报价，无逐笔 EV。"""
    base = dict(
        version="hm_touch_down_v1", window_start=1_000_000_000_000,
        win=True, ev_at_entry=None, entry_down_price=0.52,
        entry_up_price=None, direction="DOWN",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _make_db(shadow_rows, scene_rows, krev_rows=(), pattern_rows=()) -> AsyncMock:
    db = AsyncMock()
    r1, r2, r3, r4, r5 = (MagicMock() for _ in range(5))
    # 端点用指定列 SELECT，结果直接 .all()（不再 .scalars()）；
    # 5 次查询顺序：影子行 → KREV 行 → pattern 行 → SceneParamVersion SHADOW 版本名（默认空）→ 场景行
    r1.all.return_value = shadow_rows
    r2.all.return_value = krev_rows
    r3.all.return_value = pattern_rows
    r4.all.return_value = []
    r5.all.return_value = scene_rows
    db.execute = AsyncMock(side_effect=[r1, r2, r3, r4, r5])
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
        "quote_contrarian_v3a", "quote_contrarian_v3b", "quote_contrarian_v4",
        "late_night_contrarian_v1", "late_night_contrarian_v2",
        "krev_a_v1", "krev_b_v1", "hm_touch_down_v1", "hm_touch_down_v2",
        "rev_p1_v1", "rev_p2_v1", "s5_deep_z20_v1", "quote_momentum_v3"}
    for v, blk in out["shadow"].items():
        assert blk["summary"]["n"] == 0
        assert blk["summary"]["win_rate"] is None
        assert blk["curve"] == []
    # v2/v3 门禁版基准 = 2026-08-26 真实数据回测（local_shadow_v2v3_real_backtest.py）
    assert out["shadow"]["x4_v2"]["summary"]["bench_winrate"] == 0.553
    assert out["shadow"]["quote_contrarian_v3b"]["summary"]["bench_ev"] == 0.646
    assert out["shadow"]["quote_momentum_v2"]["summary"]["desc"].startswith("顺势v2")
    # v4 regime 门禁版：62 天真实订单簿回测 down 段基准（胜率+EV 双钉）
    v4 = out["shadow"]["quote_contrarian_v4"]["summary"]
    assert v4["bench_winrate"] == 0.303 and v4["bench_ev"] == 0.372
    assert v4["desc"].startswith("逆势v4")
    # 深夜变体：K 线代理回测只钉胜率基准，EV 基准因溢价口径差异留 None
    ln = out["shadow"]["late_night_contrarian_v1"]["summary"]
    assert ln["bench_winrate"] == 0.347 and ln["bench_ev"] is None
    assert ln["desc"].startswith("深夜逆势v1")
    # 深夜门禁 v2：线上情绪窗重放 OOS 只钉胜率基准，EV 基准留 None（口径不直比）
    ln2 = out["shadow"]["late_night_contrarian_v2"]["summary"]
    assert ln2["bench_winrate"] == 0.440 and ln2["bench_ev"] is None
    assert ln2["desc"].startswith("深夜逆势v2")
    # KREV K 线反转族：720d 冻结 holdout 只钉胜率基准，EV 基准留 None（口径不直比）
    kr = out["shadow"]["krev_a_v1"]["summary"]
    assert kr["bench_winrate"] == 0.642 and kr["bench_ev"] is None
    assert kr["desc"].startswith("K线反转A")
    assert out["shadow"]["krev_b_v1"]["summary"]["bench_winrate"] == 0.634
    # HM 触价族：720d 探索性回测只钉胜率基准 0.587，EV 基准留 None（口径不直比）
    hm = out["shadow"]["hm_touch_down_v1"]["summary"]
    assert hm["bench_winrate"] == 0.587 and hm["bench_ev"] is None
    assert hm["desc"].startswith("HM上吊线反弹入场")
    # HM v2：后验切片基准 0.690（非预注册，影子期前向验证）
    hm2 = out["shadow"]["hm_touch_down_v2"]["summary"]
    assert hm2["bench_winrate"] == 0.690 and hm2["bench_ev"] is None
    assert hm2["desc"].startswith("HM上吊线反弹v2")
    # 反转形态 P1/P2（共表 kline_shadow_signals）：720d/oos 几何口径回测点估计只钉胜率，
    # EV 基准留 None（纯 K 线收盘结算无报价，与 KREV 同构）
    rp1 = out["shadow"]["rev_p1_v1"]["summary"]
    assert rp1["bench_winrate"] == 0.620 and rp1["bench_ev"] is None
    assert rp1["desc"].startswith("反转P1")
    rp2 = out["shadow"]["rev_p2_v1"]["summary"]
    assert rp2["bench_winrate"] == 0.624 and rp2["bench_ev"] is None
    assert rp2["desc"].startswith("反转P2")
    # S5 深档（共表 pattern_shadow_signals）：深档回测点估计只钉胜率，EV 基准留 None（含机械成分）
    s5 = out["shadow"]["s5_deep_z20_v1"]["summary"]
    assert s5["bench_winrate"] == 0.913 and s5["bench_ev"] is None
    assert s5["desc"].startswith("S5深档")
    # 报价动量 v3（misalignment_signals）：修正未来函数后回测只钉胜率，EV 基准留 None（门禁待前向验证）
    qm3 = out["shadow"]["quote_momentum_v3"]["summary"]
    assert qm3["bench_winrate"] == 0.802 and qm3["bench_ev"] is None
    assert qm3["desc"].startswith("报价动量v3")
    assert out["scene"] == {}
    assert out["regime"]["phases"] == {}
    assert out["regime"]["by_version"] == {}
    assert out["regime"]["daily"] == []


@pytest.mark.asyncio
async def test_analytics_krev_merged_from_kline_shadow_table() -> None:
    """KREV 行（kline_shadow_signals 表）并入影子统计：胜率曲线正常，
    EV/盈亏平衡恒空（无报价结算），并参与 regime 归因。"""
    import binance_predict.main as m

    krev = [
        _krev_row(version="krev_a_v1", window_start=1_000, win=True),
        _krev_row(version="krev_b_v1", window_start=2_000, win=False),
    ]
    db = _make_db([], [], krev_rows=krev)
    out = await m.get_signals_analytics(db)

    a = out["shadow"]["krev_a_v1"]
    assert a["summary"]["n"] == 1
    assert a["summary"]["win_rate"] == 1.0
    # 无报价/无 EV：EV 与盈亏平衡列恒空（前端显示 '—'），胜率曲线正常
    assert a["summary"]["avg_ev"] is None and a["summary"]["cum_ev"] is None
    assert a["summary"]["avg_breakeven"] is None
    assert a["summary"]["bench_winrate"] == 0.642
    assert [p["cum_wr"] for p in a["curve"]] == [1.0]
    b = out["shadow"]["krev_b_v1"]
    assert b["summary"]["win_rate"] == 0.0
    assert b["summary"]["bench_winrate"] == 0.634
    # regime 归因：KREV 行同样参与 pre/pump 拆分（早于 PUMP_TS_MS → pre）
    bv = out["regime"]["by_version"]
    assert bv["krev_a_v1"]["pre"] == {"n": 1, "wins": 1, "winrate": 1.0}
    assert bv["krev_b_v1"]["pre"] == {"n": 1, "wins": 0, "winrate": 0.0}


@pytest.mark.asyncio
async def test_analytics_pattern_merged_from_pattern_shadow_table() -> None:
    """HM 触价行（pattern_shadow_signals 表）并入影子统计：胜率曲线正常，
    盈亏平衡按 DOWN 入场报价算（q/0.98 无溢价），EV 恒空（无逐笔落库口径）。"""
    import binance_predict.main as m

    patt = [
        _pattern_row(window_start=1_000, win=True, entry_down_price=0.52),
        _pattern_row(window_start=2_000, win=False, entry_down_price=0.49),
    ]
    db = _make_db([], [], pattern_rows=patt)
    out = await m.get_signals_analytics(db)

    hm = out["shadow"]["hm_touch_down_v1"]
    assert hm["summary"]["n"] == 2
    assert hm["summary"]["win_rate"] == 0.5
    assert hm["summary"]["avg_ev"] is None and hm["summary"]["cum_ev"] is None
    # 盈亏平衡 = 逐笔 DOWN 报价均值 / 0.98（非 x4 系，无溢价）
    assert hm["summary"]["avg_breakeven"] == pytest.approx(
        ((0.52 / 0.98) + (0.49 / 0.98)) / 2, abs=1e-9)
    assert hm["summary"]["bench_winrate"] == 0.587
    assert [p["cum_wr"] for p in hm["curve"]] == [1.0, 0.5]
    # regime 归因：早于 PUMP_TS_MS → pre
    bv = out["regime"]["by_version"]
    assert bv["hm_touch_down_v1"]["pre"] == {"n": 2, "wins": 1, "winrate": 0.5}


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
