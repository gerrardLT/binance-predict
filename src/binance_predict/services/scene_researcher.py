"""场景研究员（M2，LLM 自进化体系的假设生成器）。

角色铁律：LLM 只产生候选假设，不产生知识——
- 它的输出止步于 ResearchAssessment（结构化建议），无任何线上权限
- 裁决权在科学回测引擎（hypothesis_arbiter，纯代码），终审在人（promote API）
- 报告中的数字必须来自引擎输出（上下文喂什么它引用什么，不得编造）

调用形态：照抄 llm_service.agent_learn 的 Instructor + Pydantic +
max_retries=2 + 禁用 thinking 已验证形态；审计写 llm_traces(phase=SCENE_RESEARCH)。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import LLMTrace

RESEARCH_SYSTEM_PROMPT = """你是量化策略研究员，负责评估一个 BTC 预测市场信号系统的两个场景模式并提出改进假设。

## 模式定义（不可修改语义，只能调参数）
- 场景① bull_exhaust：15m 周期破 4h 阻力位势 + 周期收阳 + 光头收盘（close_pos ≥ 阈值）
  → 次周期开盘买 DOWN。机制：破位动能收盘未回吐（买力耗尽），次周期兑现反转。
- 场景② bear_exhaust：15m 周期破 4h 支撑 + 周期收阴 + 放量（vol_ratio ≥ 阈值）
  → 次周期开盘买 UP。机制：恐慌杀跌空头耗尽，次周期反弹。

## 可调参数（param_overrides 只允许这 5 个键）
- close_pos_min（0.5~0.98）：场景①光头程度阈值，当前 0.85
- vol_ratio_min（1.0~5.0）：场景②放量阈值，当前 2.0
- vol_ma_window（10~60）：均量窗口根数，当前 20
- eps（0.0001~0.002）：破位幅度阈值，当前 0.0005
- level_lookbacks（{"4h": 24~96}）：位势回看窗口，当前 48

## 科学纪律（必须遵守）
1. 每个假设必须给出机制理由（为什么这个参数变化预期改善次周期反转概率），不接受纯数据拟合理由
2. expected_impact_pp 要诚实：参考上下文中的可检测下限，小样本期声称 >5pp 需强机制支撑
3. 优先建议 maintain_status_quo=true：没有充分机制理由时维持现状是对的
4. 假设最多 3 个（多重检验预算有限，每轮最多 1 个会被采纳）
5. 你看到的数字来自统计引擎，引用它们，不要自己心算新数字"""


class ResearchHypothesis(BaseModel):
    """单条改进假设（可回测的参数化定义）。"""

    change_suggestion: str = Field(description="改什么（人类可读，一句话）")
    mechanism_reason: str = Field(description="机制理由：为什么预期改善次周期反转概率")
    expected_impact_pp: float = Field(
        description="预期胜率改善（百分点，正数；供功效预检对照）",
        ge=0.0, le=30.0,
    )
    param_overrides: dict[str, Any] = Field(
        description=(
            "SceneParams 覆盖子集，只允许键：close_pos_min/vol_ratio_min/"
            "vol_ma_window/eps/level_lookbacks；未提及的键保持现状"
        ),
    )


class ResearchAssessment(BaseModel):
    """结构化评估结论。"""

    assessment: str = Field(description="当前模式健康度评估（引用上下文数字，说明理由）")
    maintain_status_quo: bool = Field(
        default=True,
        description="是否建议维持现状（默认且优先为 true）",
    )
    hypotheses: list[ResearchHypothesis] = Field(
        default_factory=list,
        description="改进假设（≤3 条；maintain_status_quo=true 时可为空）",
    )


class SceneResearcher:
    """LLM 研究员：消费评估上下文，产出结构化假设。"""

    def __init__(self, decision_client: Instructor) -> None:
        self._client = decision_client

    async def evaluate(self, context: dict, timeout: float = 180.0) -> ResearchAssessment:
        """评估当前场景模式。context 由 research_scheduler 构建（含基线/实况/失败案例）。"""
        user_message = self._build_user_msg(context)
        logger.info(
            "SCENE_RESEARCH LLM 调用 | model={} | timeout={}s",
            settings.decision_model, timeout,
        )
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._client.create_with_completion(
                response_model=ResearchAssessment,
                messages=[
                    {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=4096,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        await self._trace(RESEARCH_SYSTEM_PROMPT, user_message, result, t0, completion)
        logger.info(
            "SCENE_RESEARCH 完成 | maintain_status_quo={} | hypotheses={}",
            result.maintain_status_quo, len(result.hypotheses),
        )
        return result

    def _build_user_msg(self, ctx: dict) -> str:
        parts = ["## 评估上下文\n"]
        parts.append(f"当前 ACTIVE 版本：{ctx.get('active_version')}")
        parts.append(f"当前参数：{ctx.get('active_params')}\n")

        parts.append("## 基线成绩（180 天官方数据，统计引擎输出）")
        parts.append(f"全样本：{ctx.get('baseline_full')}")
        parts.append(f"样本外（后 60 天盲验）：{ctx.get('baseline_validation')}")
        parts.append(f"验证集可检测下限：{ctx.get('min_detectable')}pp\n")

        parts.append("## 实盘成绩（线上已结算信号）")
        parts.append(f"{ctx.get('live_stats')}\n")

        parts.append("## 失败案例特征（输的信号，脱敏聚合）")
        for line in ctx.get("failure_profile", [])[:20]:
            parts.append(f"- {line}")
        parts.append("")
        return "\n".join(parts)

    async def _trace(self, system_prompt: str, user_message: str, result: ResearchAssessment, t0: float, completion: Any) -> None:
        """审计落库 llm_traces（phase=SCENE_RESEARCH）；失败仅告警不影响主流程。"""
        try:
            usage = getattr(completion, "usage", None)
            async with async_session_factory() as session:
                session.add(LLMTrace(
                    phase="SCENE_RESEARCH",
                    model=settings.decision_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    assistant_output=json.loads(result.model_dump_json()),
                    result_summary=f"maintain={result.maintain_status_quo} hypotheses={len(result.hypotheses)}",
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    latency_s=round(time.monotonic() - t0, 2),
                ))
                await session.commit()
        except Exception as exc:
            logger.warning("SCENE_RESEARCH 轨迹落库失败（不影响主流程）| {}", exc)
