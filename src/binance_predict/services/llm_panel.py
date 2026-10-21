"""多 LLM 归因面板（跨厂商双家族）+ claim 级确定性校验闸。

科学骨架（2026-10-19 方案，文献 ExperiGen/CauScientist/VerityGate）：
- LLM 是假设生成器，不是裁决者：输出必须绑定检验菜单中的可证伪检验；
- 强制 ≥3 个备选假设且必含「随机零假设」（对冲叙事谬误）；
- 叙事 claim 过三道确定性闸：证据ID存在 / 数值与事实卡精确匹配 / 方向符号一致；
  不过闸的句子剔除进 rejected_claims（保留审计，不展示为结论）；
- 错误记忆（已证伪假设）注入 prompt，防止重复讲故事；
- 共识≠正确：同源模型约 60% 同错；分歧是一等输出（触发统计复核）。

安全边界：只做分析调用，永不进入交易决策路径；调用失败/超时记录 error 不阻塞。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import openai
from loguru import logger

from binance_predict.config.settings import settings

PROMPT_VERSION = "attribution_prompt_v1"
STRATEGY_PROMPT_VERSION = "strategy_advice_prompt_v1"

# 检验菜单：LLM 假设必须绑定其中之一（由引擎确定性执行，LLM 不能自造检验）
TEST_MENU: dict[str, str] = {
    "streak_binomial": "对通道基准胜率检验最近 N 笔胜负数是否异常（二项检验）",
    "ev_block_bootstrap": "按日块自助检验某子集费后 EV 是否显著为正/负（去簇；params.direction=positive|negative）",
    "bucket_bonferroni": "分桶（时段/入场价带/点差/波动档）对比 + Bonferroni 校正",
    "direction_clustering": "连败单是否聚集在同方向/同走势段（方向聚集检验）",
    "cusum_changepoint": "EV/胜率序列变化点定位（边何时变了）",
    "composition_shift": "触发特征分布漂移检验（KS 型分位对比，引擎执行）",
}

HYPOTHESIS_CATEGORIES = ["RANDOM", "EXPOSURE", "EDGE_DECAY", "EXECUTION",
                         "COMPOSITION", "CROWD", "OTHER"]

SYSTEM_PROMPT = """你是量化预测市场系统的归因分析专家。你的任务不是下结论，而是提出可检验的归因假设。

铁律：
1. 你会收到一份「事实卡」（全部由确定性代码计算），它是你唯一可以引用的数字来源。禁止使用事实卡之外的任何数字。
2. 必须提出至少 3 个互相竞争的假设，其中必须包含一个 RANDOM（随机零假设：观察到的连败/连胜与正常抽样波动一致）假设。
3. 每个假设必须绑定检验菜单中的一项检验（test.name 必须来自检验菜单），并给出检验参数建议。
4. 每个叙述性论断（claims）必须引用事实卡中的 evidence_id，且 numbers 中每个数值必须从事实卡对应字段原样复制（不得换算、四舍五入或推断）。
5. 已被统计证伪的假设在「错误记忆」列表中：不要重复提出等价假设。
6. 可以给出 action_suggestion（行动建议，例如仓位/护栏调整方向），但它只是建议，人类是终审。

输出必须是单个 JSON 对象，结构：
{
  "hypotheses": [
    {
      "statement": "假设陈述（一句话）",
      "category": "RANDOM|EXPOSURE|EDGE_DECAY|EXECUTION|COMPOSITION|CROWD|OTHER",
      "confidence": 0.0,
      "test": {"name": "<检验菜单键>", "params": {"...": "..."}},
      "claims": [
        {"text": "论断", "evidence_ids": ["<事实卡证据ID>"], "numbers": [{"name": "<事实卡字段名>", "value": 0}]}
      ],
      "action_suggestion": "可空"
    }
  ],
  "narrative": "整体叙述（必须只引用已过闸的事实）"
}"""


# ======================================================================
# 策略顾问（2026-10-22）：结构化动作 + 单调收紧 + 服务端硬闸
# ======================================================================

# 动作白名单：只允许「更保守」的调整；放开类建议在服务端被整条作废
STRATEGY_VERDICTS = {"HOLD", "TIGHTEN_GUARD", "CUT_AMOUNT", "CUT_DAILY_LIMIT",
                     "PAUSE", "ESCALATE_HUMAN"}
STRATEGY_PARAM_BY_VERDICT = {
    "TIGHTEN_GUARD": "max_exec_price",
    "CUT_AMOUNT": "amount_usdt",
    "CUT_DAILY_LIMIT": "max_daily_orders",
    "PAUSE": "enabled",
}
STRATEGY_APPLYABLE_VERDICTS = set(STRATEGY_PARAM_BY_VERDICT)
RATIO_MAX_STEP = 0.5       # 金额/日限单次降幅上限
GUARD_MAX_STEP = 0.10      # 护栏单次下调上限
GUARD_MIN_VALUE = 0.05     # 护栏下限（更低等于永久停火，须走 PAUSE）
AMOUNT_MIN_VALUE = 0.1     # 单笔金额硬界（与 live_channels 校验一致）
DAILY_MIN_VALUE = 1        # 日限硬界

STRATEGY_SYSTEM_PROMPT = """你是量化预测市场系统的策略顾问。你的任务不是讲故事，而是基于确定性事实卡，判断某实盘通道是否需要调整运行参数，并给出可被机器逐条校验的结构化建议。

铁律：
1. 事实卡是你唯一可以引用的数字来源；禁止使用事实卡之外的任何数字。
2. verdict 只能从动作白名单选择：HOLD（保持不动）/ TIGHTEN_GUARD（收紧执行价护栏）/ CUT_AMOUNT（下调单笔金额）/ CUT_DAILY_LIMIT（下调日限）/ PAUSE（停火观察）/ ESCALATE_HUMAN（信息不足转人工）。
3. 只允许“更保守”的调整：金额/日限/护栏只能下调，开火与放大任何参数一律禁止，写了也会被服务端整条作废。
4. 事实卡 randomness_first.verdict=RANDOM_CONSISTENT 时，只允许 HOLD 或 ESCALATE_HUMAN——与随机波动一致时任何参数变更都是过度调参。
5. verdict 不是 HOLD/ESCALATE_HUMAN 时：
   - action.param 必须与 verdict 对应（TIGHTEN_GUARD→max_exec_price，CUT_AMOUNT→amount_usdt，CUT_DAILY_LIMIT→max_daily_orders，PAUSE→enabled）；
   - action.from_value 必须与事实卡 current_params 对应字段完全一致（不得换算/四舍五入）；
   - diagnosis.supporting_test 必须引用 pre_tests 中 significant=true 的检验（name 必须在检验菜单）；
   - 必须给出 counter_hypothesis（如果我错了，最可能的另一种解释）与 kill_criteria（出现什么证据应该撤回本建议）；
   - PAUSE 必须在 action.expiry_condition 里写清恢复条件（如“n≥20 且 p_edge_negative<0.5”）。
6. 幅度上限：金额/日限单次降幅不超过 50%；护栏单次下调不超过 0.10 且结果不低于 0.05。
7. HOLD 是一等输出：没有显著证据时必须选它；禁止为了给建议而给建议。
8. 每个证据断言（claims）必须引用事实卡中的 evidence_id，且 numbers 中每个数值必须从事实卡对应字段原样复制。

输出必须是单个 JSON 对象，结构：
{
  "schema": "strategy_advice_v1",
  "channel": "通道名",
  "verdict": "HOLD|TIGHTEN_GUARD|CUT_AMOUNT|CUT_DAILY_LIMIT|PAUSE|ESCALATE_HUMAN",
  "diagnosis": {
    "primary_cause": "RANDOM|EDGE_DECAY|BUY_EXPENSIVE|EXECUTION|EXPOSURE|COMPOSITION|UNKNOWN",
    "evidence": [
      {"claim": "断言", "evidence_ids": ["<事实卡证据ID>"],
       "numbers": [{"name": "<事实卡字段名>", "value": 0}]}
    ],
    "supporting_test": {"name": "<检验菜单键>", "verdict": "...", "significant": true},
    "randomness_check": {"verdict": "RANDOM_CONSISTENT|ANOMALOUS_LOSING|..."}
  },
  "action": {"param": "none|amount_usdt|max_daily_orders|max_exec_price|enabled",
             "from_value": 0, "to_value": 0, "duration_orders": 20,
             "expiry_condition": "..."},
  "confidence": 0.0,
  "counter_hypothesis": "...",
  "kill_criteria": "...",
  "expected_effect": {"metric": "unit_ev", "direction": "up", "min_detectable_effect": 0.02},
  "requires_human_review": true
}"""


def build_strategy_prompt(facts: dict, channel: str, ledger: list[dict],
                          error_memory: list[str],
                          current_params: dict | None = None) -> str:
    """策略建议用户提示词：事实卡 + 当前参数 + 检验结果 + 台账 + 错误记忆。"""
    supported = [h for h in ledger if h.get("status") == "SUPPORTED"][:10]
    params = current_params or facts.get("current_params") or {}
    parts = [
        f"## 通道\n{channel}\n",
        "## 当前参数（action.from_value 必须取这里的原值）\n"
        + json.dumps(params, ensure_ascii=False) + "\n",
        "## 检验菜单（supporting_test.name 只能取以下键）\n"
        + "\n".join(f"- {k}: {v}" for k, v in TEST_MENU.items()) + "\n",
        "## 事实卡（唯一数字来源；evidence_id 已标注）\n"
        + json.dumps(facts, ensure_ascii=False, indent=1) + "\n",
    ]
    if supported:
        parts.append("## 已被统计支持的假设（避免重复提出）\n"
                     + "\n".join(f"- {h['statement']}" for h in supported) + "\n")
    if error_memory:
        parts.append("## 错误记忆（已被统计证伪，禁止再提等价假设）\n"
                     + "\n".join(f"- {s}" for s in error_memory[:20]) + "\n")
    parts.append("## 任务\n基于事实卡给出本通道的策略建议（HOLD 是合法且常见的输出），"
                 "按系统提示词的 JSON 结构输出。")
    return "\n".join(parts)


def default_panel() -> list[dict]:
    """默认双家族阵容：DeepSeek + Qwen（跨厂商，均走 OpenAI 兼容接口）。"""
    override = settings.attribution_panel_models.strip()
    if override:
        out = []
        for item in override.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            provider, model = item.split(":", 1)
            out.append(_provider_entry(provider.strip(), model.strip()))
        return [p for p in out if p]
    return [
        _provider_entry("deepseek", settings.decision_model),
        _provider_entry("dashscope", settings.review_model),
    ]


def _provider_entry(provider: str, model: str) -> dict | None:
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            return None
        return {"provider": provider, "model": model, "family": "deepseek",
                "api_key": settings.deepseek_api_key, "base_url": settings.deepseek_base_url}
    if provider == "dashscope":
        if not settings.dashscope_api_key:
            return None
        return {"provider": provider, "model": model, "family": "qwen",
                "api_key": settings.dashscope_api_key, "base_url": settings.dashscope_base_url}
    if provider == "gemini":
        if not settings.gemini_api_key:
            return None
        return {"provider": provider, "model": model, "family": "gemini",
                "api_key": settings.gemini_api_key, "base_url": settings.gemini_base_url}
    logger.warning("归因面板：未知 provider {}（忽略）", provider)
    return None


def build_user_prompt(facts: dict, channel: str, ledger: list[dict],
                      error_memory: list[str]) -> str:
    """组装用户提示词：事实卡 + 知识库 + 错误记忆 + 检验菜单 + 通道上下文。"""
    supported = [h for h in ledger if h.get("status") == "SUPPORTED"][:10]
    parts = [
        f"## 通道\n{channel}\n",
        "## 检验菜单（test.name 只能取以下键）\n"
        + "\n".join(f"- {k}: {v}" for k, v in TEST_MENU.items()) + "\n",
        "## 事实卡（唯一数字来源；evidence_id 已标注）\n"
        + json.dumps(facts, ensure_ascii=False, indent=1) + "\n",
    ]
    if supported:
        parts.append("## 已被统计支持的假设（知识库，避免重复提出）\n"
                     + "\n".join(f"- {h['statement']}" for h in supported) + "\n")
    if error_memory:
        parts.append("## 错误记忆（已被统计证伪，禁止再提等价假设）\n"
                     + "\n".join(f"- {s}" for s in error_memory[:20]) + "\n")
    parts.append("## 任务\n基于事实卡提出至少 3 个竞争假设（含 RANDOM），按系统提示词的 JSON 结构输出。")
    return "\n".join(parts)


def _extract_json(text: str) -> dict | None:
    """从模型输出中稳健提取第一个 JSON 对象（容忍 markdown 围栏与前后缀文本）。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ======================================================================
# claim 级确定性校验闸（VerityGate 模式）
# ======================================================================

def _flatten_facts(facts: dict, prefix: str = "") -> dict[str, float]:
    """事实卡展平成 {字段路径: 数值}，供 claim 数值精确匹配。"""
    out: dict[str, float] = {}
    for k, v in facts.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[path] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_facts(v, path))
        # evidence id 集合单独处理，不进数值表
    return out


def verify_analysis(parsed: dict, facts: dict,
                    evidence_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """校验 LLM 输出的假设列表。

    返回 (verified_hypotheses, rejected_claims)。规则：
    - hypothesis 级：test.name 必须在检验菜单、category 合法 → 否则整条剔除（进 rejected）；
    - claim 级：evidence_ids 必须都存在；numbers 每个 value 与事实卡数值表精确匹配
      （浮点容差 1e-9，且字段名必须存在）→ 否则该 claim 剔除，假设保留（其余 claim 仍可过）。
    - RANDOM 假设缺失 → 追加系统默认随机假设（保底，LLM 不许跳过零假设）。
    """
    flat = _flatten_facts(facts)
    verified: list[dict] = []
    rejected: list[dict] = []
    has_random = False

    for h in parsed.get("hypotheses") or []:
        if not isinstance(h, dict):
            continue
        stmt = str(h.get("statement") or "").strip()
        if not stmt:
            continue
        test_name = ((h.get("test") or {}).get("name") or "")
        if test_name not in TEST_MENU:
            rejected.append({"statement": stmt,
                             "reason": f"test_not_in_menu:{test_name}"})
            continue
        category = str(h.get("category") or "OTHER").upper()
        if category not in HYPOTHESIS_CATEGORIES:
            category = "OTHER"
        if category == "RANDOM":
            has_random = True
        kept_claims: list[dict] = []
        for c in h.get("claims") or []:
            if not isinstance(c, dict):
                continue
            ev_ids = [str(e) for e in (c.get("evidence_ids") or [])]
            bad_ev = [e for e in ev_ids if e not in evidence_ids]
            number_bad = False
            for n in c.get("numbers") or []:
                if not isinstance(n, dict):
                    number_bad = True
                    break
                name, val = n.get("name"), n.get("value")
                if name not in flat:
                    number_bad = True
                    break
                try:
                    if abs(float(val) - flat[name]) > 1e-9:
                        number_bad = True
                        break
                except (TypeError, ValueError):
                    number_bad = True
                    break
            if bad_ev or number_bad:
                rejected.append({
                    "statement": stmt,
                    "claim": c,
                    "reason": ("evidence_id_not_in_facts:" + ",".join(bad_ev)) if bad_ev else "number_mismatch",
                })
                continue
            kept_claims.append({"text": c.get("text"), "evidence_ids": ev_ids,
                                "numbers": c.get("numbers") or []})
        try:
            confidence = max(0.0, min(1.0, float(h.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        verified.append({
            "statement": stmt,
            "category": category,
            "confidence": confidence,
            "test": {"name": test_name, "params": (h.get("test") or {}).get("params") or {}},
            "claims": kept_claims,
            "action_suggestion": h.get("action_suggestion") or None,
        })

    if not has_random and verified:
        verified.append({
            "statement": "随机零假设：近期表现波动与通道基准胜率下的正常抽样波动一致",
            "category": "RANDOM", "confidence": None,
            "test": {"name": "streak_binomial", "params": {}},
            "claims": [], "action_suggestion": None,
            "system_appended": True,
        })
    return verified, rejected


# ======================================================================
# 策略建议确定性校验闸（服务端硬闸，不靠 prompt 自律）
# ======================================================================

def _verify_evidence(evidence: list, flat: dict[str, float],
                     evidence_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """证据断言逐条过闸（与 verify_analysis 的 claim 闸同口径）。"""
    kept: list[dict] = []
    rejected: list[dict] = []
    for c in evidence or []:
        if not isinstance(c, dict):
            continue
        ev_ids = [str(e) for e in (c.get("evidence_ids") or [])]
        bad_ev = [e for e in ev_ids if e not in evidence_ids]
        number_bad = False
        for n in c.get("numbers") or []:
            if not isinstance(n, dict):
                number_bad = True
                break
            name, val = n.get("name"), n.get("value")
            if name not in flat:
                number_bad = True
                break
            try:
                if abs(float(val) - flat[name]) > 1e-9:
                    number_bad = True
                    break
            except (TypeError, ValueError):
                number_bad = True
                break
        if bad_ev or number_bad:
            rejected.append({"claim": c,
                             "reason": ("evidence_id_not_in_facts:" + ",".join(bad_ev))
                             if bad_ev else "number_mismatch"})
            continue
        kept.append({"claim": c.get("claim") or c.get("text"),
                     "evidence_ids": ev_ids, "numbers": c.get("numbers") or []})
    return kept, rejected


def _clamp_action(param: str, from_v: float,
                  to_v: float) -> tuple[float | None, bool, str | None]:
    """单调收紧 + 幅度上限。返回 (final_value, clamped, reject_reason)。"""
    if to_v > from_v:
        return None, False, "widen_forbidden"
    if param in ("amount_usdt", "max_daily_orders"):
        hard_floor = AMOUNT_MIN_VALUE if param == "amount_usdt" else DAILY_MIN_VALUE
        floor = max(from_v * (1.0 - RATIO_MAX_STEP), hard_floor)
        final = max(to_v, floor)
        clamped = abs(final - to_v) > 1e-12
        if abs(final - from_v) < 1e-9:
            return None, False, "noop_action"
        return final, clamped, None
    if param == "max_exec_price":
        floor = max(GUARD_MIN_VALUE, from_v - GUARD_MAX_STEP)
        final = max(to_v, floor)
        clamped = abs(final - to_v) > 1e-12
        if abs(final - from_v) < 1e-9:
            return None, False, "noop_action"
        return final, clamped, None
    return None, False, f"unknown_param:{param}"


def verify_strategy_advice(parsed: dict, facts: dict,
                           evidence_ids: set[str]) -> tuple[dict | None, list[dict]]:
    """校验 LLM 策略建议（服务端硬闸）。返回 (advice_spec, rejected)。

    硬闸（违反后的处置一律是「整条作废」，不是降级）：
    - schema / verdict / counter_hypothesis / kill_criteria 结构性缺失；
    - 放开类动作：加金额、加日限、放宽护栏、开火；
    - action.from_value 与事实卡 current_params 不精确匹配；
    - 收紧幅度超限 → 截断到上限并标 clamped（不整条作废）；
    - RANDOM_CONSISTENT → 强制降为 HOLD；supporting_test 不可验证 → 强制降为 HOLD。
    """
    rejected: list[dict] = []
    if not isinstance(parsed, dict):
        return None, [{"reason": "not_dict"}]
    if str(parsed.get("schema") or "") != "strategy_advice_v1":
        return None, [{"reason": "schema_mismatch"}]
    verdict = str(parsed.get("verdict") or "").upper()
    if verdict not in STRATEGY_VERDICTS:
        return None, [{"reason": f"verdict_not_allowed:{verdict}"}]
    counter = str(parsed.get("counter_hypothesis") or "").strip()
    kill = str(parsed.get("kill_criteria") or "").strip()
    if not counter or not kill:
        return None, [{"reason": "missing_counter_or_kill"}]

    flat = _flatten_facts(facts)
    diag = parsed.get("diagnosis") if isinstance(parsed.get("diagnosis"), dict) else {}
    primary = str(diag.get("primary_cause") or "UNKNOWN").upper()
    if primary not in HYPOTHESIS_CATEGORIES and primary != "BUY_EXPENSIVE":
        primary = "OTHER"
    kept_ev, ev_rej = _verify_evidence(diag.get("evidence"), flat, evidence_ids)
    rejected.extend(ev_rej)

    # ---- 随机性门：与随机一致只允许 HOLD / ESCALATE_HUMAN ----
    rnd = str((diag.get("randomness_check") or {}).get("verdict") or "")
    facts_random = str((facts.get("randomness_first") or {}).get("verdict") or "")
    if rnd == "RANDOM_CONSISTENT" or facts_random == "RANDOM_CONSISTENT":
        if verdict not in ("HOLD", "ESCALATE_HUMAN"):
            rejected.append({"reason": "random_consistent_forced_hold",
                             "original_verdict": verdict})
            verdict = "HOLD"

    # ---- supporting_test：必须在菜单内且被 pre_tests 证明显著 ----
    st = diag.get("supporting_test") if isinstance(diag.get("supporting_test"), dict) else {}
    st_name = str(st.get("name") or "")
    if verdict not in ("HOLD", "ESCALATE_HUMAN"):
        if st_name not in TEST_MENU:
            rejected.append({"reason": f"test_not_in_menu:{st_name}"})
            verdict = "HOLD"
        else:
            pre = facts.get("pre_tests") or []
            match = next((t for t in pre if t.get("name") == st_name), None)
            if pre and (match is None or not match.get("significant")):
                rejected.append({"reason": f"supporting_test_not_significant:{st_name}"})
                verdict = "HOLD"

    # ---- 动作校验（单调收紧 + 幅度上限 + 原值匹配）----
    action_in = parsed.get("action") if isinstance(parsed.get("action"), dict) else {}
    action: dict | None = None
    clamped = False
    if verdict in ("HOLD", "ESCALATE_HUMAN"):
        action = None
    else:
        want_param = STRATEGY_PARAM_BY_VERDICT[verdict]
        cur = facts.get("current_params") or {}
        param = str(action_in.get("param") or "")
        if param != want_param:
            return None, rejected + [{"reason": f"param_mismatch:{param}!={want_param}"}]
        cur_v = cur.get(want_param)
        if cur_v is None:
            return None, rejected + [{"reason": f"current_param_missing:{want_param}"}]
        try:
            from_v = float(action_in.get("from_value"))
        except (TypeError, ValueError):
            return None, rejected + [{"reason": "from_value_not_numeric"}]
        if abs(from_v - float(cur_v)) > 1e-9:
            return None, rejected + [{"reason": "from_value_mismatch"}]
        if want_param == "enabled":
            if action_in.get("to_value") not in (False, 0, 0.0, "false", "False"):
                return None, rejected + [{"reason": "enable_widening_forbidden"}]
            action = {"param": "enabled", "from_value": bool(cur_v), "to_value": False,
                      "duration_orders": 0,
                      "expiry_condition": str(action_in.get("expiry_condition") or "").strip()}
        else:
            try:
                to_v = float(action_in.get("to_value"))
            except (TypeError, ValueError):
                return None, rejected + [{"reason": "action_value_not_numeric"}]
            final, clamped, reason = _clamp_action(want_param, from_v, to_v)
            if reason:
                return None, rejected + [{"reason": reason}]
            try:
                duration = int(action_in.get("duration_orders")
                               or settings.strategy_advice_tracking_n)
            except (TypeError, ValueError):
                duration = settings.strategy_advice_tracking_n
            action = {"param": want_param, "from_value": round(from_v, 4),
                      "to_value": round(final, 4), "duration_orders": max(1, duration),
                      "expiry_condition": str(action_in.get("expiry_condition") or "").strip()}

    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0

    applyable = (verdict in STRATEGY_APPLYABLE_VERDICTS and action is not None
                 and (verdict != "PAUSE" or bool(action.get("expiry_condition"))))

    spec = {
        "schema": "strategy_advice_v1",
        "channel": str(parsed.get("channel") or facts.get("channel") or ""),
        "verdict": verdict,
        "diagnosis": {
            "primary_cause": primary,
            "evidence": kept_ev,
            "supporting_test": {"name": st_name, "verdict": st.get("verdict"),
                                "significant": bool(st.get("significant"))},
            "randomness_check": {"verdict": rnd or facts_random or None},
        },
        "action": action,
        "confidence": confidence,
        "counter_hypothesis": counter,
        "kill_criteria": kill,
        "expected_effect": parsed.get("expected_effect") or {},
        "requires_human_review": bool(parsed.get("requires_human_review", True)),
        "clamped": clamped,
        "applyable": bool(applyable),
    }
    return spec, rejected


class LLMPanel:
    """跨家族多模型并行分析。单模型失败/超时只记 error，不影响其他模型。"""

    def __init__(self, panel: list[dict] | None = None) -> None:
        self.panel: list[dict] = panel if panel is not None else default_panel()

    async def analyze(self, user_prompt: str) -> list[dict]:
        if not self.panel:
            return [{"provider": "none", "model": "none", "family": "none",
                     "hypotheses": [], "rejected_claims": [], "error": "panel_empty"}]
        results = await asyncio.gather(*[
            self._call_one(p, user_prompt) for p in self.panel])
        return list(results)

    async def _call_one(self, p: dict, user_prompt: str) -> dict:
        out: dict[str, Any] = {"provider": p["provider"], "model": p["model"],
                               "family": p["family"], "hypotheses": [],
                               "rejected_claims": [], "error": None}
        try:
            client = openai.AsyncOpenAI(api_key=p["api_key"], base_url=p["base_url"])
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=p["model"],
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        # 思考模型（deepseek-flash/qwen3.8 等）思考消耗输出预算：
                        # 16000 保证思考 + JSON 正文都放得下（不禁思考，2026-10-20 拍板）
                        max_tokens=16000,
                    ),
                    timeout=settings.attribution_llm_timeout_s,
                )
            finally:
                await client.close()
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            out["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            out["output_tokens"] = getattr(usage, "completion_tokens", None)
            parsed = _extract_json(text)
            if parsed is None:
                out["error"] = "json_parse_failed"
                out["raw_text_head"] = text[:500]
                return out
            out["parsed"] = parsed
            return out
        except asyncio.TimeoutError:
            out["error"] = "timeout"
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out
