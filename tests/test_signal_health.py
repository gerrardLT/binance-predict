"""信号体检纯函数自检（无 DB/网络）：红黄绿灯判据、剔除 Top-K、分桶、告警差分。"""
from __future__ import annotations

from binance_predict.services import signal_health as sh


def _ev(win: bool, q: float, ts: int = 0) -> dict:
    """单笔事件：赢 0.98/q−1，输 −1；保本 q/0.98。"""
    return {"ts": ts, "win": win, "ev": (0.98 / q - 1) if win else -1.0, "be": q / 0.98, "q": q}


def _seq(wins: int, losses: int, q: float = 0.5) -> list[dict]:
    rows = [_ev(True, q, i * 300_000) for i in range(wins)] + [_ev(False, q, (wins + i) * 300_000) for i in range(losses)]
    return sorted(rows, key=lambda e: e["ts"])


def test_gray_below_min_n() -> None:
    m = sh.compute_metrics(_seq(5, 10))
    assert sh.classify(m)[0] == "GRAY"


def test_red_when_edge_clearly_negative() -> None:
    # 100 单胜率 30%，保本 0.51 → 后验几乎确定低于保本
    m = sh.compute_metrics(_seq(30, 70))
    light, reasons = sh.classify(m)
    assert m["p_edge_negative"] > 0.99
    assert light == "RED" and "优势已变负" in reasons[0]


def test_green_when_edge_clearly_positive() -> None:
    m = sh.compute_metrics(_seq(70, 30), bench_win=0.68)
    assert m["p_edge_negative"] < 0.01
    assert sh.classify(m) == ("GREEN", [])


def test_yellow_when_below_bench_but_not_negative_edge() -> None:
    # 胜率 50% 对保本 0.4（q=0.392），基准 0.65 → 显著低于基准但仍在保本之上
    m = sh.compute_metrics(_seq(50, 50, q=0.392), bench_win=0.65)
    light, reasons = sh.classify(m)
    assert light == "YELLOW" and any("冻结基准" in r for r in reasons)


def test_yellow_when_profit_is_right_tail() -> None:
    # 3 个大赢单(q=0.05, EV≈+18.6) + 47 个小亏：总体为正，剔除 Top5 后为负
    rows = [_ev(True, 0.05, i) for i in range(3)] + [_ev(False, 0.05, 10 + i) for i in range(47)]
    m = sh.compute_metrics(rows)
    assert m["avg_ev"] > 0 and m["ev_ex_top5"] < 0 and m["top5_share"] == 1.0
    light, reasons = sh.classify(m)
    assert any("右尾" in r for r in reasons) and light in ("YELLOW", "RED")


def test_ex_top_handles_short_series() -> None:
    m = sh.compute_metrics(_seq(2, 1))
    assert m["ev_ex_top5"] is None and m["ev_ex_top1"] is not None


def test_price_and_hour_buckets_flag_small_samples() -> None:
    rows = _seq(6, 4, q=0.45) + [_ev(True, 0.85, 0)]
    pb = {r["segment"]: r for r in sh.price_buckets(rows)}
    assert pb["0.4-0.5"]["n"] == 10 and not pb["0.4-0.5"]["small_sample"]
    assert pb["0.8-0.9"]["small_sample"]
    hb = sh.hour_buckets([_ev(True, 0.5, 13 * 3_600_000)])
    assert hb[0]["segment"] == "12-16"


def test_series_is_bounded_and_ends_at_last_event() -> None:
    rows = _seq(300, 300)
    s = sh.series(rows, max_points=50)
    assert len(s) <= 60 and s[-1]["i"] == 600


def test_new_red_only_on_transition_not_cold_start_or_gray() -> None:
    row = {"scope": "shadow", "key": "a", "light": "RED"}
    assert sh.new_red_transitions({}, [row]) == []                       # 冷启动不报
    assert sh.new_red_transitions({("shadow", "a"): "GRAY"}, [row]) == []  # 灰→红：样本刚够，不算转红
    assert sh.new_red_transitions({("shadow", "a"): "GREEN"}, [row]) == [row]
    assert sh.new_red_transitions({("shadow", "a"): "RED"}, [row]) == []   # 持续红不重复报


def test_diagnosis_payload_contract_is_stable() -> None:
    m = sh.compute_metrics(_seq(30, 70))
    p = sh.build_diagnosis_payload("c", "live", "RED", ["r"], m, [], [])
    assert p["schema"] == "signal_health_diagnosis_v1" and "不得建议直接改实盘护栏" in p["guardrails"]


# ---------------- 数据层（signal_health_service） ----------------
from types import SimpleNamespace  # noqa: E402

from binance_predict.services import signal_health_service as shs  # noqa: E402
from binance_predict.services.live_performance import LiveOrder  # noqa: E402


def _row(version="v", ts=1, win=True, up=None, down=0.4, ev=None, direction="DOWN"):
    return SimpleNamespace(version=version, window_start=ts, win=win, entry_up_price=up,
                           entry_down_price=down, ev_at_entry=ev, direction=direction)


def test_shadow_events_prefers_stored_ev_and_falls_back_to_quote() -> None:
    be = lambda v, q: q / 0.98  # noqa: E731
    ev = lambda v, w, q: (0.98 / q - 1) if w else -1.0  # noqa: E731
    out = shs.shadow_events([
        _row(ts=2, win=True, ev=0.5), _row(ts=1, win=False),            # 乱序 + 落库值优先 + 兜底现算
        _row(ts=3, win=True, down=None), _row(ts=4, win=None),          # 无报价: ev/be 为 None；win 空被丢弃
    ], be, ev)["v"]
    assert [e["ts"] for e in out] == [1, 2, 3]
    assert out[0]["ev"] == -1.0 and out[1]["ev"] == 0.5
    assert out[2]["ev"] is None and out[2]["be"] is None


def test_live_events_use_unit_return_and_skip_unsettled() -> None:
    wei = 10**18
    ok = LiveOrder(channel="c", window_start=100, win=True, pnl=1.5, amount_in=str(3 * wei),
                   quote_json={"averagePrice": 0.4, "filledShareQty": 7.5, "fillSource": "binance_history_confirm"})
    none_win = LiveOrder(channel="c", window_start=200, win=None, pnl=None, amount_in=str(wei))
    out = shs.live_events([none_win, ok])
    assert list(out) == ["c"] and len(out["c"]) == 1
    e = out["c"][0]
    assert e["ev"] == 0.5 and e["q"] == 0.4 and abs(e["be"] - 3 / 7.5) < 1e-9


def test_assemble_report_tally_counts_only_active_and_hides_nothing() -> None:
    bad = _seq(30, 70)
    rep = shs.assemble_report(
        {"s_on": bad, "s_off": bad}, {}, 
        {"s_on": {"active": True, "enabled": True}, "s_off": {"active": False, "retired": True}},
        {}, lambda scope, key: None, now_ms=1,
    )
    assert rep["tally_active"]["RED"] == 1                     # 退役/下线版本不计入告警汇总
    assert {r["key"] for r in rep["rows"]} == {"s_on", "s_off"}  # 但仍在矩阵里可见
    assert ("shadow", "s_on") in rep["_events"]


def test_sentiment_summary_handles_empty_and_shares() -> None:
    assert shs.sentiment_summary([])["n"] == 0
    s = shs.sentiment_summary([("UP", 10, 1.0, 0.01), ("DOWN", 20, 3.0, -0.03), ("NOISE", None, None, None)])
    assert s["n"] == 3 and abs(s["up_share"] - 1 / 3) < 1e-9
    assert s["avg_participants"] == 15 and abs(s["mean_abs_return"] - 0.02) < 1e-9


import asyncio  # noqa: E402


def test_llm_hook_default_is_not_implemented_and_pluggable(monkeypatch) -> None:
    out = asyncio.run(shs.run_llm_diagnosis({"x": 1}))
    assert out["status"] == "not_implemented" and out["payload"] == {"x": 1}

    async def fake(payload):
        return {"summary": "ok"}
    monkeypatch.setattr(shs, "llm_diagnose_hook", fake)
    assert asyncio.run(shs.run_llm_diagnosis({}))["status"] == "ok"


def test_diagnose_death_status_expired_and_spurious() -> None:
    # 1. 样本少于 30 -> ALIVE
    assert sh.diagnose_death_status(_seq(10, 10))[0] == "ALIVE"

    # 2. 曾显著盈利（前期胜率 80%），后期衰退（后 20 笔胜率 25%） -> EXPIRED
    glory = [_ev(True, 0.5, i) for i in range(25)] + [_ev(False, 0.5, 25 + i) for i in range(5)]
    decay = [_ev(True, 0.5, 30 + i) for i in range(5)] + [_ev(False, 0.5, 35 + i) for i in range(15)]
    expired_seq = glory + decay
    status, reason = sh.diagnose_death_status(expired_seq, bench_win=0.6)
    assert status == "EXPIRED" and "衰减" in reason

    # 3. 从未盈利过（100 笔交错分布，胜率 30% 徘徊） -> SPURIOUS
    never_worked = [_ev(i % 10 < 3, 0.5, i * 1000) for i in range(100)]
    status2, reason2 = sh.diagnose_death_status(never_worked, bench_win=0.55)
    assert status2 == "SPURIOUS" and "从未建立超额优势" in reason2


def test_detect_change_point_identifies_peak_and_drawdown() -> None:
    # 前 20 笔全赢（累计 EV 上升），后 20 笔全输（累计 EV 回撤）
    wins = [_ev(True, 0.5, i * 1000) for i in range(20)]
    losses = [_ev(False, 0.5, (20 + i) * 1000) for i in range(20)]
    cp = sh.detect_change_point(wins + losses)
    assert cp is not None and cp["is_declining"] is True
    assert cp["change_ts"] == 19000
    assert cp["drawdown_ev"] >= 15.0


def test_compute_channel_correlations() -> None:
    # 通道 A 与通道 B 在 10 个窗口完全重叠且同向
    evA = [_ev(True, 0.5, i * 100) for i in range(10)]
    evB = [_ev(True, 0.5, i * 100) for i in range(10)] + [_ev(False, 0.5, 2000)]
    corrs = sh.compute_channel_correlations({"A": evA, "B": evB}, min_shared=5)
    assert len(corrs) == 1
    assert corrs[0]["channel_a"] == "A" and corrs[0]["channel_b"] == "B"
    assert corrs[0]["shared_windows"] == 10
    assert corrs[0]["agreement_rate"] == 1.0


def test_trend_and_volatility_buckets() -> None:
    events = [
        {"ts": 1, "win": True, "ev": 0.5, "ret_4h": -0.015, "ret_24h": -0.025, "vol_4h": 0.001},
        {"ts": 2, "win": False, "ev": -1.0, "ret_4h": 0.015, "ret_24h": 0.025, "vol_4h": 0.004},
    ]
    t4 = {r["segment"]: r for r in sh.trend_4h_buckets(events)}
    assert t4["急跌(<-1%)"]["n"] == 1 and t4["大涨(>+1%)"]["n"] == 1

    t24 = {r["segment"]: r for r in sh.trend_24h_buckets(events)}
    assert t24["暴跌(<-2%)"]["n"] == 1 and t24["过热(>+2%)"]["n"] == 1

    vb = {r["segment"]: r for r in sh.volatility_buckets(events)}
    assert vb["低波动"]["n"] == 1 and vb["高波动"]["n"] == 1


# ---------------- 编排层/端点冒烟（main.py，取数函数替身，不触 DB） ----------------
import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402


@pytest.fixture
def health_env(monkeypatch):
    import binance_predict.main as m
    from binance_predict.services import signal_health_service as shs

    rows = [_row(version="x4_v2", ts=i, win=(i % 10 < 3), down=0.5) for i in range(100)]
    monkeypatch.setattr(m, "_load_shadow_settled_rows", AsyncMock(return_value=rows))
    monkeypatch.setattr(m, "_load_live_performance_orders", AsyncMock(return_value=[]))
    monkeypatch.setattr(shs, "load_market_regime_lookup", AsyncMock(return_value=lambda ts: {"ret_4h": None, "ret_24h": None, "vol_4h": None}))
    monkeypatch.setattr(m, "_current_live_channels", lambda: {})
    monkeypatch.setattr(m, "_signal_health_cache", None)
    return m


@pytest.mark.asyncio
async def test_build_signal_health_covers_registry_and_marks_bad_shadow_red(health_env) -> None:
    m = health_env
    rep = await m._build_signal_health(AsyncMock())
    by = {(r["scope"], r["key"]): r for r in rep["rows"]}
    assert by[("shadow", "x4_v2")]["light"] == "RED"          # 30% 胜率 vs 保本 ~0.51
    assert any(k[0] == "live" for k in by)                    # 实盘通道即使无订单也在矩阵（灰灯）
    assert all(r["light"] == "GRAY" for k, r in by.items() if k[0] == "live")
    assert "_events" in rep


@pytest.mark.asyncio
async def test_health_endpoint_hides_internal_events_and_detail_validates(health_env) -> None:
    m = health_env
    out = await m.get_signals_health(refresh=True, _=None, db=AsyncMock())
    assert "_events" not in out and out["thresholds"]["min_n"] == 30

    with pytest.raises(HTTPException) as e1:
        await m.get_signals_health_detail(scope="bogus", key="x", _=None, db=AsyncMock())
    assert e1.value.status_code == 422
    with pytest.raises(HTTPException) as e2:
        await m.get_signals_health_detail(scope="shadow", key="nope", _=None, db=AsyncMock())
    assert e2.value.status_code == 404


@pytest.mark.asyncio
async def test_detail_and_diagnose_stub_contract(health_env, monkeypatch) -> None:
    m = health_env
    from binance_predict.services import signal_health_service as svc

    monkeypatch.setattr(svc, "load_history", AsyncMock(return_value=[]))
    d = await m.get_signals_health_detail(scope="shadow", key="x4_v2", _=None, db=AsyncMock())
    assert d["light"] == "RED" and d["series"] and d["price_buckets"] and d["hour_buckets"]
    assert d["diagnosis_payload"]["schema"] == "signal_health_diagnosis_v1"

    diag = await m.diagnose_signal_health(scope="shadow", key="x4_v2", _=None, db=AsyncMock())
    assert diag["status"] == "not_implemented" and diag["payload"]["channel"] == "x4_v2"
