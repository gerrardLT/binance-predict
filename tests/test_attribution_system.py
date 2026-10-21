"""LLM 归因系统单测：证据包纯函数 + claim 校验闸 + 引擎纯函数层。"""
from __future__ import annotations

import json

from binance_predict.services.llm_panel import (
    TEST_MENU,
    _extract_json,
    build_user_prompt,
    verify_analysis,
    verify_strategy_advice,
)
from binance_predict.services.order_context_recorder import (
    assemble_pack,
    compute_order_facts,
)
from binance_predict.services.attribution_engine import (
    compute_channel_facts,
    consensus_map,
    execute_test,
    similar,
)


# ======================================================================
# 证据包组装（纯函数）
# ======================================================================

def _bar(t, o=100.0):
    return {"open_time": t, "open": o, "high": o + 1, "low": o - 1,
            "close": o + 0.5, "volume": 10.0}


def _order(**kw):
    base = {
        "id": 1, "signal_version": "ch_a", "direction": "DOWN", "side": "BUY",
        "status": "FILLED", "market_period": "5m", "window_start": 1_700_000_000_000,
        "amount_in": 10_000_000_000_000_000_000, "quote_json": {
            "averagePrice": "0.55", "quotedAvgPrice": "0.5415", "fillSource": "binance_history_confirm"},
        "settle_outcome": "UP", "win": False, "pnl": -10.0, "settle_price": 0.99,
        "error_message": None, "created_at": "2026-10-19T00:00:00+00:00",
        "settled_at": "2026-10-19T00:05:00+00:00",
    }
    base.update(kw)
    return base


class TestAssemblePack:
    def test_full_pack_built(self):
        pack, missing = assemble_pack(
            _order(), {"trigger_ts": 1, "quote_ts": 2, "config_snapshot": {}},
            [{"ts": 1, "up": 0.5, "down": 0.5}], [{"ts": 2, "up": 0.9, "down": 0.1}],
            [_bar(i * 60_000) for i in range(60)],
            [_bar(i * 300_000) for i in range(50)],
            [_bar(i * 60_000) for i in range(10)],
            [{"channel": "ch_b", "direction": "DOWN", "win": True, "pnl": 8.0}],
            {"n_prior": 20, "rolling20_win_rate": 0.62, "rolling20_pnl": -5.0},
        )
        assert not missing
        assert pack["schema_version"] == "v1"
        assert len(pack["pre"]["klines_1m"]) == 60
        assert pack["pre"]["channel_state"]["rolling20_win_rate"] == 0.62
        assert pack["exec"]["quote"]["averagePrice"] == "0.55"
        assert pack["post"]["settle"]["win"] is False
        assert pack["post"]["siblings"][0]["channel"] == "ch_b"
        assert pack["pre"]["regime"]["ret_1h"] is not None

    def test_missing_dims_partial(self):
        pack, missing = assemble_pack(
            _order(), None, [], [], None, None, None, [], {"n_prior": 0},
        )
        assert set(missing) == {"klines_1m_before", "klines_5m_before", "quotes_pre",
                                "assessment", "klines_1m_window"}
        assert pack["exec"]["order"]["id"] == 1

    def test_quote_downsample(self):
        from binance_predict.services.order_context_recorder import _downsample
        pts = [{"ts": i} for i in range(1000)]
        out = _downsample(pts, cap=480)
        assert len(out) <= 481 and out[-1]["ts"] == 999

    def test_compute_order_facts(self):
        f = compute_order_facts(_order())  # 10U 输单
        assert f["amount_usdt"] == 10.0
        assert f["entry_avg_price"] == 0.55
        assert f["tca"]["pnl_theoretical"] == -10.0
        assert abs(f["tca"]["slippage_bp"] - 156.8) < 0.5


# ======================================================================
# LLM 输出 JSON 提取 + claim 校验闸
# ======================================================================

FACTS = {
    "channel": "scene_momentum_fade",
    "streak": {"verdict": "ANOMALOUS_LOSING", "n": 21, "wins": 8,
               "observed_rate": 0.381, "binom_p_le": 0.023, "longest_losing_run": 5},
    "q4_execution": {"slippage_bp_mean": 12.3, "slippage_bp_max": 88.8},
}
EV_IDS = {"channel", "streak", "streak.verdict", "streak.observed_rate",
          "streak.binom_p_le", "q4_execution.slippage_bp_mean"}


class TestExtractJson:
    def test_plain(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert _extract_json('前言```json\n{"a": [1,2]}\n```后语') == {"a": [1, 2]}

    def test_nested_with_noise(self):
        text = '输出如下 {"hypotheses": [{"x": {"y": 2}}]} 完毕'
        assert _extract_json(text)["hypotheses"][0]["x"]["y"] == 2

    def test_garbage(self):
        assert _extract_json("no json") is None
        assert _extract_json("") is None


class TestVerifyAnalysis:
    def _parsed(self, **overrides):
        h = {
            "statement": "连败与持续上涨行情中同向敞口聚集有关",
            "category": "EXPOSURE", "confidence": 0.7,
            "test": {"name": "direction_clustering", "params": {}},
            "claims": [{
                "text": "近 21 单胜率 38.1%",
                "evidence_ids": ["streak"],
                "numbers": [{"name": "streak.observed_rate", "value": 0.381}],
            }],
            "action_suggestion": "考虑同向敞口上限",
        }
        parsed = {"hypotheses": [h]}
        for k, v in overrides.items():
            if k == "hyp":
                parsed["hypotheses"] = [v]
        return parsed

    def test_valid_passes_gate(self):
        verified, rejected = verify_analysis(self._parsed(), FACTS, EV_IDS)
        assert len(verified) == 2  # 1 条 LLM + 系统补的 RANDOM？——不：已有 RANDOM 才不补
        # 本例无 RANDOM 假设 → 系统应补一条
        cats = [h["category"] for h in verified]
        assert "RANDOM" in cats and "EXPOSURE" in cats
        assert not rejected

    def test_number_mismatch_rejected_claim_kept_hypothesis(self):
        parsed = self._parsed(hyp={
            "statement": "滑点异常大", "category": "EXECUTION", "confidence": 0.5,
            "test": {"name": "bucket_bonferroni", "params": {"dim": "entry_band"}},
            "claims": [{"text": "平均滑点 99bp", "evidence_ids": ["streak"],
                        "numbers": [{"name": "q4_execution.slippage_bp_mean", "value": 99.0}]}],
        })
        verified, rejected = verify_analysis(parsed, FACTS, EV_IDS)
        assert len(verified) == 2  # 假设保留（claims 清空）+ 补的 RANDOM
        assert verified[0]["claims"] == []
        assert rejected and rejected[0]["reason"] == "number_mismatch"

    def test_evidence_id_not_in_facts_rejected(self):
        parsed = self._parsed(hyp={
            "statement": "虚构证据", "category": "OTHER", "confidence": 0.1,
            "test": {"name": "cusum_changepoint", "params": {}},
            "claims": [{"text": "x", "evidence_ids": ["nonexistent_id"], "numbers": []}],
        })
        verified, rejected = verify_analysis(parsed, FACTS, EV_IDS)
        assert "nonexistent_id" in rejected[0]["reason"]
        assert verified[0]["statement"] == "虚构证据"

    def test_test_not_in_menu_whole_hypothesis_rejected(self):
        parsed = self._parsed(hyp={
            "statement": "自造检验", "category": "OTHER", "confidence": 0.9,
            "test": {"name": "my_custom_test", "params": {}},
            "claims": [],
        })
        verified, rejected = verify_analysis(parsed, FACTS, EV_IDS)
        assert all(h["statement"] != "自造检验" for h in verified)
        assert rejected[0]["reason"].startswith("test_not_in_menu")

    def test_confidence_clamped(self):
        parsed = self._parsed(hyp={
            "statement": "置信超界", "category": "CROWD", "confidence": 5.0,
            "test": {"name": "cusum_changepoint", "params": {}}, "claims": [],
        })
        verified, _ = verify_analysis(parsed, FACTS, EV_IDS)
        assert verified[0]["confidence"] == 1.0

    def test_random_present_not_appended(self):
        parsed = self._parsed(hyp={
            "statement": "随机零假设", "category": "RANDOM", "confidence": 0.4,
            "test": {"name": "streak_binomial", "params": {}}, "claims": [],
        })
        verified, _ = verify_analysis(parsed, FACTS, EV_IDS)
        randoms = [h for h in verified if h["category"] == "RANDOM"]
        assert len(randoms) == 1

    def test_prompt_includes_error_memory_and_menu(self):
        prompt = build_user_prompt(FACTS, "ch", [], ["已被证伪的故事A"])
        assert "已被证伪的故事A" in prompt
        assert all(k in prompt for k in TEST_MENU)


# ======================================================================
# 引擎纯函数：事实卡 / 检验执行 / 共识
# ======================================================================

def _mk_orders(n_hist=40, hist_rate=0.62, recent_wins=None, base_ts=1_700_000_000_000):
    """构造通道订单序列：n_hist 历史单（胜率 hist_rate）+ recent 段。"""
    import random
    rng = random.Random(7)
    orders = []
    ts = base_ts
    for i in range(n_hist):
        win = rng.random() < hist_rate
        orders.append({
            "id": i, "channel": "ch", "direction": "DOWN", "win": win,
            "pnl": 8.0 if win else -10.0, "amount_usdt": 10.0,
            "created_ts": ts, "avg_price": 0.55, "quoted_avg_price": 0.548,
        })
        ts += 3600_000
    for j, win in enumerate(recent_wins or []):
        orders.append({
            "id": 1000 + j, "channel": "ch", "direction": "DOWN", "win": win,
            "pnl": 8.0 if win else -10.0, "amount_usdt": 10.0,
            "created_ts": ts, "avg_price": 0.55, "quoted_avg_price": 0.548,
        })
        ts += 3600_000
    return orders


class TestChannelFacts:
    def test_anomalous_losing_detected(self):
        orders = _mk_orders(recent_wins=[True] * 8 + [False] * 13)
        facts, ev_ids = compute_channel_facts(orders, window_n=21)
        assert facts["streak"]["verdict"] == "ANOMALOUS_LOSING"
        assert facts["base_rate"] and abs(facts["base_rate"] - 0.62) < 0.2
        assert "channel" in ev_ids

    def test_insufficient_base(self):
        orders = _mk_orders(n_hist=5, recent_wins=[False] * 8)
        facts, _ = compute_channel_facts(orders, window_n=8)
        assert facts["streak"]["verdict"] == "INSUFFICIENT_DATA"
        assert facts["base_n"] < 30

    def test_execution_dims_populated(self):
        orders = _mk_orders(recent_wins=[True] * 3 + [False] * 3)
        facts, _ = compute_channel_facts(orders, window_n=6)
        assert facts["q4_execution"]["slippage_bp_mean"] is not None
        assert facts["entry_band_ev"]["n_buckets"] >= 1


class TestExecuteTest:
    def test_direction_clustering(self):
        orders = _mk_orders(n_hist=10, recent_wins=[False] * 6)
        res = execute_test("direction_clustering", {}, orders)
        assert res["verdict"] == "CLUSTERED"  # 全部亏损落在数小时内 → 高集中
        assert res["significant"]

    def test_cusum_on_flat(self):
        orders = _mk_orders(recent_wins=[True] * 5)
        res = execute_test("cusum_changepoint", {}, orders)
        assert res["name"] == "cusum_changepoint"

    def test_unknown_test(self):
        assert execute_test("made_up", {}, [])[0 if False else "verdict"] == "UNKNOWN_TEST"

    def test_composition_shift_untestable(self):
        res = execute_test("composition_shift", {}, [])
        assert res["verdict"] == "UNTESTABLE" and not res["significant"]


class TestConsensus:
    def test_consensus_and_divergence(self):
        m1 = [{"statement": "连败源于同向敞口聚集", "category": "EXPOSURE"},
              {"statement": "市场定价吃掉了边际", "category": "EDGE_DECAY"}]
        m2 = [{"statement": "连败源于同向敞口聚集效应", "category": "EXPOSURE"},
              {"statement": "纯属随机波动", "category": "RANDOM"}]
        cons = consensus_map([m1, m2])
        assert len(cons["consensus"]) == 1   # 敞口聚集两边都提了（包含式相似）
        assert len(cons["divergent"]) == 2   # 定价挤光 / 随机 各一家

    def test_similar_contains(self):
        assert similar("连败源于同向敞口聚集", "连败源于同向敞口聚集（近期）")
        assert not similar("连败源于同向敞口聚集", "滑点过大导致亏损")


class TestRandomnessGateLogic:
    """随机性前置门：RANDOM_CONSISTENT 时 _execute 不应调 LLM（集成层行为在 API 测试覆盖），
    这里验证事实卡对正常波动通道给出 RANDOM_CONSISTENT。"""

    def test_normal_channel_consistent(self):
        import random
        rng = random.Random(11)
        orders = _mk_orders(recent_wins=[rng.random() < 0.62 for _ in range(20)])
        facts, _ = compute_channel_facts(orders, window_n=20)
        # 基准 62%、20 单正常抽样 → 绝大多数构型应为 RANDOM_CONSISTENT
        assert facts["streak"]["verdict"] in ("RANDOM_CONSISTENT", "INSUFFICIENT_DATA")


# ======================================================================
# 结算钩子（交易链路改动配套测试）
# ======================================================================

class TestSettlementHook:
    def _row(self):
        from types import SimpleNamespace
        return SimpleNamespace(id=42, signal_version="scene_momentum_fade")

    def test_callback_fires_with_order_and_channel(self):
        from binance_predict.services.trade_settler import TradeSettler
        seen = []
        s = TradeSettler(on_settled=lambda oid, ch: seen.append((oid, ch)))
        s._fire_on_settled(self._row())
        assert seen == [(42, "scene_momentum_fade")]

    def test_callback_exception_swallowed(self):
        from binance_predict.services.trade_settler import TradeSettler

        def boom(oid, ch):
            raise RuntimeError("hook bug")
        s = TradeSettler(on_settled=boom)
        s._fire_on_settled(self._row())  # 不得抛出

    def test_no_callback_is_noop(self):
        from binance_predict.services.trade_settler import TradeSettler
        TradeSettler()._fire_on_settled(self._row())  # 默认 None 不报错

    def test_dispatcher_disabled_short_circuits(self, monkeypatch):
        import binance_predict.main as m
        from binance_predict.config.settings import settings
        monkeypatch.setattr(settings, "attribution_enabled", False)
        fired = []
        monkeypatch.setattr(m.order_context_recorder, "enqueue",
                            lambda oid: fired.append(oid))
        m._attribution_on_settled(1, "ch")
        assert not fired  # 总开关关闭时完全不 dispatch


# ======================================================================
# 回归：归因 DB 端点的 sa_select 局部导入（生产 500 事故 2026-10-20）
# main.py 惯例是端点内 `from sqlalchemy import select as sa_select`，
# 模块级不存在该名字——漏 import 只在运行时 NameError（500）。
# ======================================================================

def test_attribution_endpoints_have_local_sa_select_import():
    import re
    from pathlib import Path
    src = Path("src/binance_predict/main.py").read_text(encoding="utf-8")
    # 找到每个 attribution 端点函数体，检查用了 sa_select 的都有局部导入
    fn_pattern = re.compile(
        r'async def (list_attribution_runs|get_attribution_run|list_hypotheses|'
        r'list_order_context_packs|get_order_context_pack)\(.*?\n(?=@app|\Z)',
        re.S)
    checked = 0
    for m in fn_pattern.finditer(src):
        body = m.group(0)
        if "sa_select(" in body:
            assert "select as sa_select" in body, \
                f"{m.group(1)} 使用 sa_select 但缺局部导入（生产 NameError→500 同型）"
            checked += 1
    assert checked >= 4  # 五个 DB 端点里至少四个用 sa_select


# ======================================================================
# 三家族面板（2026-10-20：deepseek 官方 + gemini 中转 + qwen 百炼）
# ======================================================================

def test_provider_entry_gemini(monkeypatch):
    from binance_predict.config.settings import settings
    from binance_predict.services.llm_panel import _provider_entry
    monkeypatch.setattr(settings, "gemini_api_key", "sk-test")
    monkeypatch.setattr(settings, "gemini_base_url", "https://relay.example/v1")
    entry = _provider_entry("gemini", "ag/gemini-3.8-flash-high")
    assert entry == {"provider": "gemini", "model": "ag/gemini-3.8-flash-high",
                     "family": "gemini", "api_key": "sk-test",
                     "base_url": "https://relay.example/v1"}
    monkeypatch.setattr(settings, "gemini_api_key", "")
    assert _provider_entry("gemini", "x") is None  # 无 key → 不进面板


def test_default_panel_respects_override(monkeypatch):
    from binance_predict.config.settings import settings
    from binance_predict.services.llm_panel import default_panel
    monkeypatch.setattr(settings, "attribution_panel_models",
                        "gemini:ag/gemini-3.8-flash-high,deepseek:deepseek-v4-flash")
    monkeypatch.setattr(settings, "gemini_api_key", "sk-g")
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-d")
    panel = default_panel()
    assert [(p["provider"], p["model"], p["family"]) for p in panel] == [
        ("gemini", "ag/gemini-3.8-flash-high", "gemini"),
        ("deepseek", "deepseek-v4-flash", "deepseek"),
    ]


def test_engine_no_orm_dict_expansion():
    """回归：'Hypothesis' object is not iterable（生产 run#4 FAILED 根因）。

    SQLAlchemy 2.0 ORM 对象不可 dict()/迭代展开；台账去重必须显式取列。
    """
    from pathlib import Path
    src = Path("src/binance_predict/services/attribution_engine.py").read_text(encoding="utf-8")
    assert "dict(h) for h in" not in src


# ======================================================================
# 策略建议确定性闸（2026-10-22）：单调收紧 + 幅度上限 + 随机性门
# ======================================================================

STRATEGY_FACTS = {
    "channel": "ch_a",
    "randomness_first": {"verdict": "ANOMALOUS_LOSING"},
    "current_params": {"amount_usdt": 2.0, "max_daily_orders": 100,
                       "max_exec_price": 0.5, "enabled": True},
    "pre_tests": [
        {"name": "streak_binomial", "significant": True, "verdict": "ANOMALOUS_LOSING"},
        {"name": "ev_block_bootstrap", "significant": True, "verdict": "SIGNIFICANT_NEGATIVE"},
        {"name": "cusum_changepoint", "significant": True, "verdict": "BREAK_DETECTED"},
    ],
    "trend": {"level": "L2", "cusum": 9.3, "n": 60},
}
STRATEGY_EVID = {"channel", "randomness_first.verdict", "current_params.amount_usdt",
                 "current_params.max_daily_orders", "current_params.max_exec_price",
                 "current_params.enabled", "trend.level", "trend.cusum"}


def _advice_parsed(verdict="CUT_AMOUNT", param="amount_usdt", from_v=2.0, to_v=1.0,
                   evidence=None, counter="反方", kill="撤回",
                   st_name="streak_binomial", st_sig=True, rnd="ANOMALOUS_LOSING",
                   to_value_override=None):
    to_value = to_value_override if to_value_override is not None else to_v
    return {
        "schema": "strategy_advice_v1",
        "channel": "ch_a",
        "verdict": verdict,
        "diagnosis": {
            "primary_cause": "EDGE_DECAY",
            "evidence": evidence or [],
            "supporting_test": {"name": st_name, "verdict": "ANOMALOUS_LOSING",
                                "significant": st_sig},
            "randomness_check": {"verdict": rnd},
        },
        "action": {"param": param, "from_value": from_v, "to_value": to_value,
                   "duration_orders": 30, "expiry_condition": ""},
        "confidence": 0.7,
        "counter_hypothesis": counter,
        "kill_criteria": kill,
        "expected_effect": {},
        "requires_human_review": True,
    }


class TestVerifyStrategyAdvice:
    def test_valid_cut_amount(self):
        spec, rejected = verify_strategy_advice(_advice_parsed(), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is not None
        assert spec["verdict"] == "CUT_AMOUNT"
        assert spec["action"]["to_value"] == 1.0
        assert spec["applyable"] is True
        assert rejected == []

    def test_widen_forbidden_voids_whole_advice(self):
        spec, rejected = verify_strategy_advice(
            _advice_parsed(to_v=3.0), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is None
        assert any(r["reason"] == "widen_forbidden" for r in rejected)

    def test_ratio_clamped_to_fifty_percent(self):
        # 2.0 → 0.5（降 75%）应截断到 1.0 并标 clamped
        spec, _ = verify_strategy_advice(
            _advice_parsed(to_v=0.5), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is not None
        assert spec["action"]["to_value"] == 1.0
        assert spec["clamped"] is True

    def test_guard_step_clamped(self):
        spec, _ = verify_strategy_advice(
            _advice_parsed(verdict="TIGHTEN_GUARD", param="max_exec_price",
                           from_v=0.5, to_v=0.05),
            STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is not None
        assert spec["action"]["to_value"] == 0.4  # max(0.05, 0.5-0.10)
        assert spec["clamped"] is True

    def test_from_value_must_match_facts(self):
        spec, rejected = verify_strategy_advice(
            _advice_parsed(from_v=1.5), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is None
        assert any(r["reason"] == "from_value_mismatch" for r in rejected)

    def test_param_must_match_verdict(self):
        spec, rejected = verify_strategy_advice(
            _advice_parsed(param="max_exec_price"), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is None
        assert any("param_mismatch" in r["reason"] for r in rejected)

    def test_missing_counter_or_kill_voids(self):
        spec, rejected = verify_strategy_advice(
            _advice_parsed(counter=""), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is None
        assert rejected[0]["reason"] == "missing_counter_or_kill"

    def test_random_consistent_forces_hold(self):
        facts = dict(STRATEGY_FACTS, randomness_first={"verdict": "RANDOM_CONSISTENT"})
        spec, rejected = verify_strategy_advice(
            _advice_parsed(rnd="RANDOM_CONSISTENT"), facts, STRATEGY_EVID)
        assert spec is not None
        assert spec["verdict"] == "HOLD"
        assert spec["action"] is None
        assert spec["applyable"] is False
        assert any(r["reason"] == "random_consistent_forced_hold" for r in rejected)

    def test_supporting_test_must_be_significant(self):
        # LLM 引用 ev_block_bootstrap 降仓，但事实卡 pre_tests 显示该检验不显著
        facts = dict(STRATEGY_FACTS, pre_tests=[
            {"name": "streak_binomial", "significant": True,
             "verdict": "ANOMALOUS_LOSING"},
            {"name": "ev_block_bootstrap", "significant": False,
             "verdict": "NOT_SIGNIFICANT"},
        ])
        spec, rejected = verify_strategy_advice(
            _advice_parsed(st_name="ev_block_bootstrap"), facts, STRATEGY_EVID)
        assert spec is not None
        assert spec["verdict"] == "HOLD"
        assert any("supporting_test_not_significant" in r["reason"] for r in rejected)

    def test_enabled_widening_forbidden(self):
        parsed = _advice_parsed(verdict="PAUSE", param="enabled",
                                from_v=1.0, to_value_override=True)
        spec, rejected = verify_strategy_advice(parsed, STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is None
        assert any(r["reason"] == "enable_widening_forbidden" for r in rejected)

    def test_pause_without_recovery_condition_not_applyable(self):
        parsed = _advice_parsed(verdict="PAUSE", param="enabled",
                                from_v=1.0, to_value_override=False)
        spec, _ = verify_strategy_advice(parsed, STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is not None
        assert spec["verdict"] == "PAUSE"
        assert spec["applyable"] is False  # 缺恢复条件 → 不给一键按钮

    def test_evidence_number_gate(self):
        evidence = [{"claim": "金额是 2.0", "evidence_ids": ["current_params.amount_usdt"],
                     "numbers": [{"name": "current_params.amount_usdt", "value": 2.0}]}]
        spec, _ = verify_strategy_advice(
            _advice_parsed(evidence=evidence), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec is not None
        assert len(spec["diagnosis"]["evidence"]) == 1
        # 编造数字 → claim 被剔除，但建议本体保留（只展示过闸部分）
        bad = [{"claim": "编造", "evidence_ids": ["current_params.amount_usdt"],
                "numbers": [{"name": "current_params.amount_usdt", "value": 9.9}]}]
        spec2, rejected2 = verify_strategy_advice(
            _advice_parsed(evidence=bad), STRATEGY_FACTS, STRATEGY_EVID)
        assert spec2 is not None
        assert spec2["diagnosis"]["evidence"] == []
        assert any(r["reason"] == "number_mismatch" for r in rejected2)


def test_ev_block_bootstrap_negative_direction():
    """负向 EV 检验：全亏损单应判 SIGNIFICANT_NEGATIVE（降仓的合法依据）。"""
    orders = [{"win": False, "pnl": -10.0, "amount_usdt": 10.0,
               "created_ts": i * 86_400_000} for i in range(30)]
    res = execute_test("ev_block_bootstrap", {"direction": "negative"}, orders)
    assert res["verdict"] == "SIGNIFICANT_NEGATIVE"
    assert res["significant"] is True
    # 正向对照：同一批单不构成「显著为正」
    res_pos = execute_test("ev_block_bootstrap", {}, orders)
    assert res_pos["significant"] is False
