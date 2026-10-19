"""LLM 客户端服务（2026-10-19 瘦身版）。

系统B（SentimentAgent 四阶段闭环）已于 2026-10-19 整体移除，本文件只保留：
1. 跨厂商 OpenAI 兼容客户端构造（决策 DeepSeek）——场景研究 SceneResearcher
   依赖 `_decision_client`（instructor 封装）；LLM 归因面板 llm_panel 自建
   客户端，不经本服务。
2. llm_traces 轨迹写入助手（fire-and-forget；表仍服务 SCENE_RESEARCH 审计）。
"""
from __future__ import annotations

import asyncio
import time

import instructor
import openai
from loguru import logger

from ..config.settings import settings


class LLMService:
    """LLM 客户端与轨迹服务。

    - 决策模型（decision_model）→ DeepSeek 原生 API（instructor Tool Calling 封装）
    - 复盘模型（review_model）→ 百炼 DashScope（配置保留；面板自建客户端）
    """

    def __init__(self) -> None:
        decision_openai_client = openai.AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        # 场景研究直接使用该 Instructor 实例（main.py 装配处传入）
        self._decision_client = instructor.from_openai(
            decision_openai_client,
            model=settings.decision_model,
        )
        self._decision_model = settings.decision_model
        logger.info(
            "LLM 服务初始化完成 | 决策={} @ {}",
            settings.decision_model,
            settings.deepseek_base_url,
        )
        # LLM 轨迹落库任务引用集合（fire-and-forget，防止 task 被 GC）
        self._trace_tasks: set = set()

    # =================================================================
    # LLM 轨迹（llm_traces 表，fire-and-forget，绝不影响主流程）
    # =================================================================

    @staticmethod
    def _extract_usage(completion: object | None) -> tuple[int | None, int | None]:
        usage = getattr(completion, "usage", None) if completion is not None else None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        try:
            return int(prompt_tokens), int(completion_tokens)
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # 中文混合文本的粗估：字符数 / 2（与瘦身前实现同口径）
        return max(1, len(text) // 2)

    def _spawn_trace(
        self,
        phase: str,
        system_prompt: str,
        user_message: str,
        result: object,
        start_monotonic: float,
        completion: object | None,
    ) -> None:
        """组装并异步落库一条 LLM 轨迹（fire-and-forget，绝不影响主流程）。"""
        try:
            latency = time.monotonic() - start_monotonic
            prompt_tokens, completion_tokens = self._extract_usage(completion)
            if prompt_tokens is None:
                prompt_tokens = self._estimate_tokens(user_message)
            if completion_tokens is None:
                completion_tokens = max(1, int(prompt_tokens * 0.3))
            est_cost = (
                prompt_tokens * settings.llm_input_price_per_1m
                + completion_tokens * settings.llm_output_price_per_1m
            ) / 1_000_000
            try:
                output_dict = result.model_dump()  # type: ignore[attr-defined]
            except Exception:
                output_dict = None
            reasoning = getattr(result, "reasoning", None)

            task = asyncio.create_task(
                self._write_trace(
                    phase=phase,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    assistant_output=output_dict,
                    reasoning=reasoning,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_yuan=est_cost,
                    latency_s=latency,
                )
            )
            self._trace_tasks.add(task)
            task.add_done_callback(self._trace_tasks.discard)
        except Exception as e:  # 组装/入队失败也不能影响主流程
            logger.warning("LLM 轨迹入队失败（忽略，不影响主流程）: {}", e)

    async def _write_trace(self, **fields: object) -> None:
        """将一条 LLM 轨迹写入 llm_traces 表（独立会话，失败仅告警）。"""
        from ..db.engine import async_session_factory
        from ..db.models import LLMTrace

        try:
            async with async_session_factory() as session:
                session.add(LLMTrace(model=self._decision_model, **fields))
                await session.commit()
        except Exception as e:
            logger.warning(
                "LLM 轨迹落库失败（忽略）: phase={} err={}",
                fields.get("phase"), e,
            )


# 全局单例（scene_researcher 取 _decision_client；归因面板不经此实例）
llm_service = LLMService()
