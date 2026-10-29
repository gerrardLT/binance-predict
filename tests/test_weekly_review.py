"""LLM 每周复盘单测：事实卡纯函数 + 10 条注意点规则 + 服务端硬闸。"""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest

from binance_predict.services.llm_panel import verify_weekly_review
from binance_predict.services.weekly_review import (
    _channel_numbers,
    _concentration,
    _numeric_paths,
    _portfolio_metrics,
    _unsettled_stuck,
    collect_facts,
    derive_watchouts,
    run_weekly_review,
    week_window,
)

_CH_DEFAULTS = {
    "channel": "c1", "display_name": "C1", "family": "quote_edge",
    "market_period": "5m", "enabled": True,
    "params": {}, "n": 60, "settled_n": 60, "cost": 120.0, "pnl": 12.0,
    "roi": 0.1, "win_rate": 0.55, "win_rate_low": 0.42, "mean_breakeven": 0.52,
    "edge": 0.03, "benchmark": 0.52, "benchmark_gap": 0.03,
    "p_edge_negative": 0.2, "trend_level": "L0", "trend_rank": 0,
    "trend_reasons": [], "avg_stake": 2.0, "rule_stake": None,
    "amount_usdt": 2.0, "amount_pct": None, "amount_cap_usdt": 20.0,
    "max_exec_price": 0.55, "max_daily_orders": 100,
    "rejected": {"total": 0, "guard": 0, "other": 0}, "reject_rate": 0.0,
    "first_order_age_days": 90.0, "facts": {},
}


def _ch(**kw) -> dict:
    out = dict(_CH_DEFAULTS)
    out.update(kw)
    return out


def _facts(channels=None, *, top1=0.2, top3=0.5, drift=None,
           unsettled=0, orders=0, settled=0) -> dict:
    channels = channels or []
    facts = {
        "schema": "weekly_review_facts_v1",
        "week": {"key": "2026-W44"},
        "portfolio": {
            "n": 10, "cost": 20.0, "pnl": 1.0, "roi": 0.05,
            "win_rate": 0.5, "edge": 0.01, "mean_breakeven": 0.5,
            "delta": {}, "daily": [],
            "concentration": {"top1_channel": "c1", "top1_share": top1,
                              "top3_share": top3, "by_channel": []},
            "equity": {"base_equity_usdt": 200.0, "available_usdt": None,
                       "open_cost": 0.0, "drift_ratio": drift},
            "ev_bootstrap": {"n": 10, "p_le_zero": 0.3, "significant_positive": False},
            "rejected": {"total": 0, "guard": 0, "other": 0},
        },
        "channels": channels,
        "ch": {c["channel"]: _channel_numbers(c) for c in channels},
        "coverage": {"channels": len(channels), "orders": orders,
                     "orders_with_settlement": settled,
                     "unsettled_stuck": unsettled},
    }
    facts["watchouts"] = derive_watchouts(facts)
    facts["evidence_ids"] = sorted(_numeric_paths(facts))
    return facts


# ======================================================================
# 时间窗口
# ======================================================================

def test_week_window_is_iso_monday():
    w = week_window(int(dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc)
                        .timestamp() * 1000), offset_weeks=-1)
    start = dt.datetime.fromtimestamp(w["start_ms"] / 1000, dt.timezone.utc)
    assert start.weekday() == 0 and (start.hour, start.minute, start.second) == (0, 0, 0)
    assert w["end_ms"] - w["start_ms"] == 7 * 86_400_000
    assert w["prev_end_ms"] == w["start_ms"]
    assert w["prev_start_ms"] == w["start_ms"] - 7 * 86_400_000
    assert w["key"] == f"{start.isocalendar().year}-W{start.isocalendar().week:02d}"


def test_week_window_offset_zero_is_current_week():
    now_ms = int(dt.datetime(2026, 10, 28, 12, tzinfo=dt.timezone.utc).timestamp() * 1000)
    cur, prev = week_window(now_ms, 0), week_window(now_ms, -1)
    assert cur["start_ms"] == prev["start_ms"] + 7 * 86_400_000


# ======================================================================
# 事实卡纯函数
# ======================================================================

def test_portfolio_metrics_basic():
    rows = [
        {"status": "FILLED", "win": True, "pnl": 2.0, "amount_usdt": 2.0,
         "created_ts": 1000, "avg_price": 0.5, "quoted_avg_price": 0.5},
        {"status": "FILLED", "win": False, "pnl": -2.0, "amount_usdt": 2.0,
         "created_ts": 2000, "avg_price": 0.5, "quoted_avg_price": 0.5},
        {"status": "FAILED", "error_message": "执行价护栏弃单: ...", "amount_usdt": 0.0},
        {"status": "FAILED", "error_message": "余额不足", "amount_usdt": 0.0},
    ]
    m = _portfolio_metrics(rows)
    assert m["n"] == 2 and m["cost"] == 4.0 and m["pnl"] == 0.0
    assert m["win_rate"] == 0.5
    assert m["rejected"] == {"total": 2, "guard": 1, "other": 1}


def test_empty_portfolio_metrics():
    m = _portfolio_metrics([])
    assert m["n"] == 0 and m["roi"] is None and m["edge"] is None


def test_concentration():
    c = _concentration([_ch(channel="a", cost=30.0), _ch(channel="b", cost=10.0)])
    assert c["top1_channel"] == "a"
    assert c["top1_share"] == 0.75
    assert c["total_cost"] == 40.0


def test_unsettled_stuck_counts_only_expired():
    now = 10_000_000
    rows = [
        {"status": "FILLED", "win": None, "window_start": now - 3_600_000,
         "market_period": "5m"},                                  # 超时 → 计
        {"status": "FILLED", "win": None, "window_start": now - 60_000,
         "market_period": "5m"},                                  # 未超时
        {"status": "FILLED", "win": True, "window_start": now - 3_600_000,
         "market_period": "5m"},                                  # 已结算
        {"status": "FAILED", "win": None, "window_start": now - 3_600_000,
         "market_period": "5m"},                                  # 未成交
    ]
    assert _unsettled_stuck(rows, now) == 1


def test_numeric_paths_skip_lists():
    """list 内数字不可寻址（与 _flatten_facts 一致），只有 dict 路径。"""
    paths = _numeric_paths({"a": {"b": 1}, "items": [{"x": 2}]})
    assert "a.b" in paths
    assert "items" in paths
    assert not any(p.startswith("items.") for p in paths)


# ======================================================================
# 10 条注意点规则
# ======================================================================

def test_no_watchouts_on_clean_facts():
    assert derive_watchouts(_facts([_ch(benchmark=0.60, max_exec_price=0.55)])) == []


def test_concentration_watchout():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch()], top1=0.40, top3=0.70))}
    assert "CONCENTRATION" in codes


def test_small_sample_positive_pnl():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(n=5, pnl=3.0, settled_n=5)]))}
    assert "SMALL_SAMPLE_RANK" in codes


def test_benchmark_divergence():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(win_rate=0.40, benchmark_gap=-0.12, settled_n=40)]))}
    assert "BENCHMARK_DIVERGENCE" in codes


def test_guard_at_breakeven():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(max_exec_price=0.60, benchmark=0.58)]))}
    assert "GUARD_AT_BREAKEVEN" in codes


def test_guard_reject_rate():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(reject_rate=0.6)]))}
    assert "GUARD_REJECT_RATE" in codes


def test_edge_negative():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(edge=-0.05, p_edge_negative=0.95, settled_n=40)]))}
    assert "EDGE_NEGATIVE" in codes


def test_new_channel_no_gate():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(enabled=True, first_order_age_days=10.0)]))}
    assert "NEW_CHANNEL_NO_GATE" in codes


def test_trend_unresolved():
    codes = {w["code"] for w in derive_watchouts(
        _facts([_ch(trend_rank=2, trend_level="L2")]))}
    assert "TREND_UNRESOLVED" in codes


def test_unsettled_backlog_watchout_is_p0():
    ws = [w for w in derive_watchouts(_facts([_ch()], unsettled=2))
          if w["code"] == "UNSETTLED_BACKLOG"]
    assert ws and ws[0]["severity"] == "P0"


def test_equity_drift_watchout():
    codes = {w["code"] for w in derive_watchouts(_facts([_ch()], drift=0.5))}
    assert "EQUITY_DRIFT" in codes


def test_watchouts_sorted_by_severity():
    ws = derive_watchouts(_facts(
        [_ch(n=5, pnl=3.0, edge=-0.05, p_edge_negative=0.95, settled_n=40)],
        top1=0.4, top3=0.7, unsettled=1))
    ranks = [{"P0": 3, "P1": 2, "P2": 1}[w["severity"]] for w in ws]
    assert ranks == sorted(ranks, reverse=True)


# ======================================================================
# 服务端硬闸
# ======================================================================

def _facts_for_gate() -> dict:
    return _facts([_ch(channel="c1", amount_usdt=5.0, amount_pct=None)])


def _verify(parsed: dict, facts: dict | None = None):
    facts = facts or _facts_for_gate()
    return verify_weekly_review(parsed, facts, set(facts["evidence_ids"]),
                               channel_meta={"c1": {"display_name": "C1"}},
                               base_equity=200.0)


def test_gate_rejects_bad_schema():
    report, rejected = _verify({"schema": "nope"})
    assert report is None and rejected[0]["reason"] == "schema_mismatch"


def test_gate_keeps_valid_claim_and_drops_bad_number():
    parsed = {
        "schema": "weekly_review_v1",
        "headline": "h",
        "overall": {"narrative": "n", "claims": [
            {"text": "ok", "evidence_ids": ["portfolio.pnl"],
             "numbers": [{"name": "portfolio.pnl", "value": 1.0}]},
            {"text": "bad", "evidence_ids": ["portfolio.pnl"],
             "numbers": [{"name": "portfolio.pnl", "value": 9.0}]},
            {"text": "fake id", "evidence_ids": ["portfolio.nope"],
             "numbers": [{"name": "portfolio.pnl", "value": 1.0}]},
        ]},
    }
    report, rejected = _verify(parsed)
    assert report is not None
    assert len(report["overall"]["claims"]) == 1
    assert {r["reason"] for r in rejected} >= {"number_mismatch"}


def test_gate_unknown_channel_dropped():
    report, rejected = _verify({"schema": "weekly_review_v1", "per_signal": [
        {"channel": "ghost", "verdict": "KEEP"}]})
    assert report is not None and report["per_signal"] == []
    assert any(r["reason"].startswith("unknown_channel") for r in rejected)


def test_gate_invalid_verdict_downgrades_to_watch():
    report, _ = _verify({"schema": "weekly_review_v1", "per_signal": [
        {"channel": "c1", "verdict": "YOLO"}]})
    assert report["per_signal"][0]["verdict"] == "WATCH"


def test_gate_improvement_requires_test_and_kill_criteria():
    report, rejected = _verify({"schema": "weekly_review_v1", "per_signal": [
        {"channel": "c1", "verdict": "TIGHTEN", "improvements": [
            {"change": "收紧护栏", "test": {"name": "not_a_test"},
             "kill_criteria": "x"},
            {"change": "收紧护栏", "test": {"name": "streak_binomial"},
             "kill_criteria": ""},
            {"change": "收紧护栏", "test": {"name": "streak_binomial"},
             "kill_criteria": "若 n≥30 仍无显著差异则撤回", "priority": "P1"},
        ]}]})
    imps = report["per_signal"][0]["improvements"]
    assert len(imps) == 1 and imps[0]["test"]["name"] == "streak_binomial"
    assert len([r for r in rejected if r["reason"].startswith("improvement_invalid")]) == 2


def test_gate_watchout_cannot_downgrade_severity():
    facts = _facts([_ch()], top1=0.4, top3=0.7)   # 确定性规则产出 P1 CONCENTRATION
    assert any(w["code"] == "CONCENTRATION" and w["severity"] == "P1"
               for w in facts["watchouts"])
    report, rejected = verify_weekly_review(
        {"schema": "weekly_review_v1", "watchouts": [
            {"code": "CONCENTRATION", "severity": "P2", "statement": "还行",
             "evidence_ids": ["portfolio.concentration.top1_share"],
             "numbers": [{"name": "portfolio.concentration.top1_share",
                          "value": 0.4}]}]},
        facts, set(facts["evidence_ids"]), channel_meta={}, base_equity=200.0)
    assert report["watchouts"][0]["severity"] == "P1"
    assert any(r["reason"].startswith("severity_downgrade_blocked") for r in rejected)


def test_gate_watchout_without_evidence_dropped():
    report, rejected = _verify({"schema": "weekly_review_v1", "watchouts": [
        {"code": "X", "severity": "P0", "statement": "无证据的危言耸听"}]})
    assert report["watchouts"] == []
    assert any(r["reason"].startswith("watchout_ungated") for r in rejected)


def test_gate_exploratory_requires_stop_condition():
    report, rejected = _verify({"schema": "weekly_review_v1", "exploratory": [
        {"question": "q", "data_source": "kline_720d",
         "falsifiable_test": "t", "stop_condition": ""},
        {"question": "q2", "data_source": "made_up",
         "falsifiable_test": "t", "stop_condition": "s"},
        {"question": "q3", "data_source": "kline_720d",
         "falsifiable_test": "t", "stop_condition": "s", "n_estimate": 300},
    ]})
    assert [e["question"] for e in report["exploratory"]] == ["q3"]
    assert len([r for r in rejected if r["reason"] == "exploratory_incomplete"]) == 2


def test_gate_sizing_cannot_widen():
    report, rejected = _verify({"schema": "weekly_review_v1", "sizing": [
        {"channel": "c1", "pct": 0.01, "rationale": "降仓"},       # 200×1% = 2 ≤ 5
        {"channel": "c1", "pct": 0.05, "rationale": "加仓"},       # 10 > 5 → 拒
        {"channel": "c1", "pct": 0.5, "rationale": "离谱"},        # 超区间 → 拒
    ]})
    assert [s["pct"] for s in report["sizing"]] == [0.01]
    assert any(r["reason"] == "sizing_widen_forbidden" for r in rejected)
    assert any(r["reason"].startswith("sizing_pct_out_of_range") for r in rejected)


def test_gate_sizing_accepts_downgrade_from_existing_pct():
    facts = _facts([_ch(channel="c1", amount_pct=0.02, rule_stake=4.0,
                        amount_usdt=5.0)])
    report, rejected = verify_weekly_review(
        {"schema": "weekly_review_v1", "sizing": [
            {"channel": "c1", "pct": 0.015, "rationale": "降到 1.5%"}]},
        facts, set(facts["evidence_ids"]), channel_meta={}, base_equity=200.0)
    assert report["sizing"][0]["pct"] == 0.015
    assert report["sizing"][0]["proposed_amount"] == 3.0


def test_gate_returns_prose_but_claims_are_gated():
    report, _ = _verify({"schema": "weekly_review_v1",
                         "headline": "本周小赚，风险集中在 S4",
                         "overall": {"narrative": "叙述可以自由写"}})
    assert report["headline"].startswith("本周小赚")
    assert report["overall"]["narrative"] == "叙述可以自由写"
    assert report["overall"]["claims"] == []


@pytest.mark.parametrize("payload", [None, [], "str"])
def test_gate_non_dict_payload(payload):
    report, rejected = _verify(payload)  # type: ignore[arg-type]
    assert report is None and rejected


# ======================================================================
# 编排（空数据库 + 降级路径）
# ======================================================================

class _EmptySession:
    """execute(...).scalars().all() → [] 的最小会话桩（无真实 DB）。"""

    async def execute(self, stmt):
        res = MagicMock()
        res.scalars.return_value.all.return_value = []
        return res


@pytest.mark.asyncio
async def test_collect_facts_empty_week():
    facts = await collect_facts(
        _EmptySession(), channel_meta={"c1": {"display_name": "C1"}},
        base_equity=200.0)
    assert facts["schema"] == "weekly_review_facts_v1"
    assert facts["portfolio"]["n"] == 0
    assert facts["coverage"]["channels"] == 1
    assert facts["ch"]["c1"]["n"] == 0
    assert "portfolio.n" in facts["evidence_ids"]
    assert "ch.c1.amount_usdt" in facts["evidence_ids"]
    assert facts["portfolio"]["equity"]["base_equity_usdt"] == 200.0


@pytest.mark.asyncio
async def test_run_weekly_review_degrades_without_llm(monkeypatch):
    """LLM 关闭 → 仍产出确定性报告（注意点 + 组合摘要），不报错。"""
    from binance_predict.services import weekly_review as wr

    fixture = _facts([_ch(n=5, pnl=3.0, settled_n=5)], top1=0.4, top3=0.7)

    async def _fake_collect(*_a, **_kw):
        return fixture

    monkeypatch.setattr(wr, "collect_facts", _fake_collect)
    monkeypatch.setattr(wr.settings, "weekly_review_llm_enabled", False)

    review = await run_weekly_review(
        "MANUAL", channel_meta={"c1": {"display_name": "C1"}}, persist=False)
    assert review is not None
    assert review["status"] == "DONE"
    assert review["verdict"] == "OK"
    assert review["report"]["llm"] is None
    codes = {w["code"] for w in review["report"]["deterministic"]["watchouts"]}
    assert "CONCENTRATION" in codes and "SMALL_SAMPLE_RANK" in codes


@pytest.mark.asyncio
async def test_run_weekly_review_insufficient_data(monkeypatch):
    from binance_predict.services import weekly_review as wr

    fixture = _facts([])
    fixture["portfolio"]["n"] = 0          # 空周：无成交订单 → 样本不足

    async def _fake_collect(*_a, **_kw):
        return fixture

    monkeypatch.setattr(wr, "collect_facts", _fake_collect)

    review = await run_weekly_review("MANUAL", channel_meta={}, persist=False)
    assert review is not None and review["verdict"] == "INSUFFICIENT_DATA"


# ======================================================================
# 审核回归（2026-10-30 P0/P1/P2 修复）
# ======================================================================

def test_gate_sizing_falls_back_to_fixed_amount_when_rule_missing():
    """pct 开启但 rule_stake 缺失 → 用固定金额做基数，仍不得放大。"""
    facts = _facts([_ch(channel="c1", amount_pct=0.02, rule_stake=None,
                        amount_usdt=5.0)])
    report, rejected = verify_weekly_review(
        {"schema": "weekly_review_v1", "sizing": [
            {"channel": "c1", "pct": 0.05, "rationale": "想加仓"}]},
        facts, set(facts["evidence_ids"]), channel_meta={}, base_equity=200.0)
    assert report["sizing"] == []
    assert any(r["reason"] == "sizing_widen_forbidden" for r in rejected)


def test_gate_sizing_rejects_when_basis_missing():
    """当前金额与基准都不可得 → 无法证明不放大 → 整条拒。"""
    facts = _facts([_ch(channel="c1", amount_pct=0.02, rule_stake=None,
                        amount_usdt=None)])
    report, rejected = verify_weekly_review(
        {"schema": "weekly_review_v1", "sizing": [
            {"channel": "c1", "pct": 0.01, "rationale": "降仓"}]},
        facts, set(facts["evidence_ids"]), channel_meta={}, base_equity=200.0)
    assert report["sizing"] == []
    assert any(r["reason"].startswith("sizing_basis_missing") for r in rejected)


@pytest.mark.asyncio
async def test_run_weekly_review_threads_runtime_base_equity(monkeypatch):
    """P0：基准本金必须来自运行时值，而不是 env 默认。"""
    from binance_predict.services import weekly_review as wr

    seen: dict = {}
    fixture = _facts([_ch()], drift=0.0)

    async def _fake_collect(*_a, **kw):
        seen.update(kw)
        return fixture

    monkeypatch.setattr(wr, "collect_facts", _fake_collect)
    monkeypatch.setattr(wr.settings, "weekly_review_llm_enabled", False)

    review = await run_weekly_review("MANUAL", channel_meta={},
                                     base_equity=350.0, persist=False)
    assert seen["base_equity"] == 350.0
    assert review is not None


class _CapturingSession:
    """捕获 SQL 语句的会话桩（不连 DB）。"""

    def __init__(self) -> None:
        self.stmts: list[str] = []

    async def execute(self, stmt):
        self.stmts.append(str(stmt))
        res = MagicMock()
        res.scalars.return_value.all.return_value = []
        return res


@pytest.mark.asyncio
async def test_load_orders_buckets_by_window_start():
    """P1：周报分桶按 window_start（与性能面板同口径），created_at 只做兜底。"""
    from binance_predict.services.weekly_review import _load_orders

    sess = _CapturingSession()
    await _load_orders(sess, 1_000_000, 2_000_000, ["c1"])
    sql = sess.stmts[0].lower()
    assert "window_start >=" in sql and "window_start <" in sql
    assert "created_at >=" in sql          # 仅 NULL 回落分支
