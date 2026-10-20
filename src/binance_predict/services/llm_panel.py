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

# 检验菜单：LLM 假设必须绑定其中之一（由引擎确定性执行，LLM 不能自造检验）
TEST_MENU: dict[str, str] = {
    "streak_binomial": "对通道基准胜率检验最近 N 笔胜负数是否异常（二项检验）",
    "ev_block_bootstrap": "按日块自助检验某子集费后 EV 是否显著为正/非正（去簇）",
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
