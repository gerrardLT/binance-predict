"""
BTC 5min LLM 预测系统 V2 - LLM 决策服务（核心模块）

核心逻辑：每次决策前通过 prompt + LLM 多轮验证反馈沟通得到可靠决策。

实现方式：
1. 使用 Instructor 的 Tool Calling 模式（Mode.Tools）调用 deepseek-v4-flash / qwen3.7-max
   - 两个模型均不支持原生 Structured Output，但支持 Function Calling
   - Instructor 通过 Tool Calling + Pydantic 验证实现等效的结构化输出
2. 自动重试机制：Pydantic 验证失败时，Instructor 会将验证错误回传给 LLM 让其自修复
3. 多轮验证闭环：
   - 第 1 轮：LLM 输出决策结论
   - Pydantic 自动校验（枚举值、数值范围、必填字段）
   - 校验失败 → 自动回传错误 → LLM 修正 → 最多重试 max_retries 次
   - 全部通过 → HARD 规则代码兜底校验
   - HARD 规则违反 → 强制改写 NO_TRADE
4. reasoning-first 模式：Schema 中 reasoning 字段在答案字段之前，强制 LLM 先推理再决策

百炼 DashScope API 通过 OpenAI 兼容接口统一调用：
- base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
- 决策模型: deepseek-v4-flash（输入 1 元/百万Token，输出 2 元/百万Token）
- 复盘模型: qwen3.7-max（输入 12 元/百万Token，输出 36 元/百万Token）
"""

from __future__ import annotations

import asyncio
import json
import time

import instructor
import openai
from loguru import logger

from ..config.settings import settings
from ..models.schemas import (
    ArbitrateOutput,
    DiscoveryOutput,
    EvolveOutput,
    LearnOutput,
)
from ..prompts.agent_templates import (
    ARBITRATE_SYSTEM_PROMPT,
    DISCOVERY_SYSTEM_PROMPT,
    EVOLVE_SYSTEM_PROMPT,
    LEARN_SYSTEM_PROMPT,
)
from .metrics import metrics_collector


class LLMService:
    """
    LLM 决策与复盘服务

    两个模型走不同的 API 通道：
    - 决策模型 deepseek-v4-flash → DeepSeek 原生 API (api.deepseek.com)
    - 复盘模型 qwen3.7-max → 百炼 DashScope API (dashscope.aliyuncs.com)

    两个模型均不支持原生 Structured Output，
    因此使用 Instructor 默认的 Mode.Tools（Tool Calling 模式）。

    注意：instructor.from_provider("openai/...", base_url=...) 会忽略 base_url，
    因此改用手动创建 AsyncOpenAI 客户端 + instructor.from_openai() 包装。
    """

    def __init__(self) -> None:
        # --- 决策 LLM 客户端（deepseek-v4-flash → DeepSeek 原生 API）---
        decision_openai_client = openai.AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._raw_decision_client = decision_openai_client  # 原生客户端，供情绪分析使用
        self._decision_client = instructor.from_openai(
            decision_openai_client,
            model=settings.decision_model,
        )

        # 缓存模型名和超时配置（供情绪分析路径使用）
        self._decision_model = settings.decision_model
        self._decision_timeout = 120  # 秒，情绪分析允许更长时间

        logger.info(
            "LLM 服务初始化完成 | 决策={} @ {}",
            settings.decision_model,
            settings.deepseek_base_url,
        )

        # LLM 轨迹落库任务引用集合（fire-and-forget，防止 task 被 GC）
        self._trace_tasks: set = set()

    # =================================================================
    # Sentiment_Agent 三阶段结构化 LLM 调用（Learn / Predict / Evolve）
    #
    # 对应 spec `sentiment-agent-loop` 的 design.md「LLM 结构化输出设计」。
    # 三方法均复用 decide() 已验证的 Instructor 调用形态（instructor.from_openai
    # 的 Tool Calling 模式 + Pydantic 校验 + 自动重试 max_retries=2 + 禁用 thinking），
    # 并用 asyncio.wait_for 施加 LLM 内层超时。失败/超时/重试耗尽时直接抛异常，
    # 不在本层降级——由上层 SentimentAgent 按「无静默降级」策略处置（Req 7.4）。
    # =================================================================

    async def agent_learn(
        self,
        windows: list[dict],
        active_patterns: list[dict],
        timeout: float,
    ) -> LearnOutput:
        """
        学习阶段（Learn Phase）结构化 LLM 调用（Req 2.4 / 7.1 / 7.2 / 7.3 / 7.4）。

        分析最近若干个已归档情绪窗口与当前 ACTIVE 模式库，让 LLM 发现新模式或
        更新已有模式，返回结构化的 LearnOutput（reasoning-first + discoveries）。

        Args:
            windows: 最近 N 个 outcome 非空的情绪窗口（每个 dict 含
                curve_up_pct / curve_down_pct（[{t, v}, ...]）/ outcome / actual_return）
            active_patterns: 当前 Pattern_Memory 中所有 ACTIVE 模式（dict 列表，含 id 与特征）
            timeout: LLM 内层超时（秒），由上层按 settings.agent_llm_timeouts["LEARN"] 传入

        Returns:
            LearnOutput: 结构化学习结论（reasoning + discoveries[PatternDiscovery]）

        Raises:
            asyncio.TimeoutError: LLM 调用超过 timeout。
            Exception: 网络错误 / 重试 2 次后 Pydantic 校验仍失败等，均直接向上抛出。
        """
        # Learn 为重载阶段：对每个窗口的 UP/DOWN 曲线按分钟下采样以压缩 token
        user_message = self._build_learn_user_msg(windows, active_patterns)
        logger.info(
            "开始 Learn LLM 调用 | model={} | windows={} | active_patterns={} | timeout={}s",
            settings.decision_model,
            len(windows),
            len(active_patterns),
            timeout,
        )
        # 照搬 decide() 已验证形态：Instructor + Pydantic 校验 + 自动重试 + 禁用 thinking
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=LearnOutput,
                messages=[
                    {"role": "system", "content": LEARN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=4096,
                temperature=0.1,
                # 禁用 thinking mode，避免与 Instructor 的 tool_choice 冲突（与 decide() 一致）
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        self._record_llm_usage("LEARN", user_message, t0, completion)
        self._spawn_trace("LEARN", LEARN_SYSTEM_PROMPT, user_message, result, t0, completion)
        return result

    async def agent_deep_learn(
        self,
        symbolized_windows: list[dict],
        timeout: float,
        feedback: dict | None = None,
        hints: list[dict] | None = None,
    ) -> DiscoveryOutput:
        """
        深度模式发现（科学发现轨，Phase 2）：LLM 作为假设生成器。

        与旧轨的区别（宪法 Q1/Q5 与第〇条角色分离）：
        - 输入：发现集窗口的符号串 + 几何摘要（三通道，每通道独立分箱），
          不再是原始曲线全量点
        - 输出：谓词 DSL 假设（DiscoveryOutput.hypotheses），不再是自由文本特征
        - 统计审判由程序在验证集上完成（discovery.screen_hypotheses），LLM 不得自我验证

        Args:
            symbolized_windows: 已符号化的发现集窗口 payload（每项含 start_time /
                outcome / channels{channel: {symbols, geometry}}）
            timeout: LLM 超时秒数
            feedback: Q7-2 反馈包（负样本全量 / 正样本摘要 / 存活期统计），
                None 时按冷启动渲染（首轮发现无历史审判记录）
            hints: 程序预筛线索榜单（hypothesis_miner.mine_hints 产出，
                train 集穷举统计），None 时退化为纯直觉发现

        Returns:
            DiscoveryOutput: reasoning + hypotheses[PredicateHypothesis]
        """
        user_message = self._build_discovery_user_msg(symbolized_windows, feedback, hints)
        logger.info(
            "开始科学发现 LLM 调用 | model={} | symbolized_windows={} | timeout={}s",
            settings.decision_model,
            len(symbolized_windows),
            timeout,
        )
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=DiscoveryOutput,
                messages=[
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=settings.agent_deep_learn_max_tokens,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        self._record_llm_usage("DEEP_LEARN", user_message, t0, completion)
        self._spawn_trace("DEEP_LEARN", DISCOVERY_SYSTEM_PROMPT, user_message, result, t0, completion)
        return result

    async def agent_deep_learn_stream(
        self,
        symbolized_windows: list[dict],
        idle_timeout: float,
        feedback: dict | None = None,
        hints: list[dict] | None = None,
    ):
        """深度模式发现（流式版，科学发现轨）：逐 token 推送 LLM 输出，供前端实时打字机展示。

        与 agent_deep_learn 的差异：
        - 使用 Instructor create_partial 流式生成，逐步产出「部分完整」的 DiscoveryOutput
        - 不施加一次性总超时，改为「空闲超时」：仅当相邻两次分片间隔超过 idle_timeout
          才判定超时（只要模型在持续吐字就不算超时），彻底规避旧的 100s 硬超时被掐问题
        - 迭代结束后用最终对象落一条 DEEP_LEARN 轨迹（token 用量为估算值）

        产出事件（dict）：
        - {"type": "reasoning", "delta": str}  reasoning 新增片段（打字机增量）
        - {"type": "progress", "hypotheses": int}   当前已解析出的假设条数
        - {"type": "done", "result": DiscoveryOutput}  最终完整结构化结果
        - {"type": "error", "message": str}  空闲超时或流式异常

        Raises: 不向上抛异常，所有失败均以 {"type": "error"} 事件产出，交由上层转成 SSE。
        """
        user_message = self._build_discovery_user_msg(symbolized_windows, feedback, hints)
        logger.info(
            "开始科学发现 LLM 流式调用 | model={} | symbolized_windows={} | idle_timeout={}s",
            settings.decision_model,
            len(symbolized_windows),
            idle_timeout,
        )
        t0 = time.monotonic()
        try:
            stream = self._decision_client.create_partial(
                response_model=DiscoveryOutput,
                messages=[
                    {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=settings.agent_deep_learn_max_tokens,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            logger.error(
                "Deep Learn(stream): 建流失败 | error_type={} | error={}",
                type(exc).__name__,
                str(exc),
            )
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            return

        it = stream.__aiter__()
        last_reasoning = ""
        last_count = 0
        final = None
        while True:
            try:
                partial = await asyncio.wait_for(it.__anext__(), timeout=idle_timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.error(
                    "Deep Learn(stream): 空闲超时 | idle_timeout={}s | 已收到 reasoning={} 字符",
                    idle_timeout,
                    len(last_reasoning),
                )
                yield {
                    "type": "error",
                    "message": f"LLM 空闲超时（{idle_timeout:.0f}s 内无新输出）",
                }
                return
            except Exception as exc:
                logger.error(
                    "Deep Learn(stream): 流式迭代异常 | error_type={} | error={}",
                    type(exc).__name__,
                    str(exc),
                )
                yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                return

            final = partial
            reasoning = getattr(partial, "reasoning", None) or ""
            if len(reasoning) > len(last_reasoning):
                delta = reasoning[len(last_reasoning):]
                last_reasoning = reasoning
                yield {"type": "reasoning", "delta": delta}
            cur_count = len(getattr(partial, "hypotheses", None) or [])
            if cur_count != last_count:
                last_count = cur_count
                yield {"type": "progress", "hypotheses": cur_count}

        if final is None:
            yield {"type": "error", "message": "LLM 未返回任何内容"}
            return

        # 迭代完成：记录用量 + 落轨迹（流式无 completion 对象，token 为估算值）
        self._record_llm_usage("DEEP_LEARN", user_message, t0, None)
        self._spawn_trace("DEEP_LEARN", DISCOVERY_SYSTEM_PROMPT, user_message, final, t0, None)
        yield {"type": "done", "result": final}

    async def agent_arbitrate(
        self,
        window_payload: dict,
        candidates: list[dict],
        remaining_seconds: int,
        timeout: float,
    ) -> ArbitrateOutput:
        """
        预测阶段仲裁调用（科学发现宪法第八条，Phase 3）。

        程序已完成确定性谓词匹配，仅在「多模式命中且方向冲突」时调用本方法
        请 LLM 消歧。LLM 只能从冲突候选中选定一个模式或放弃，direction 由
        程序从选定模式的 predicted_direction 推导，LLM 无权发明方向。

        Args:
            window_payload: 当前窗口符号化视图 payload（_view_to_payload 形态：
                start_time / channels{channel: {symbols, geometry}}）
            candidates: 冲突候选模式（dict 列表，含 id / pattern_name /
                predicted_direction / description / predicate / win_rate / sample_count）
            remaining_seconds: 距当前窗口结束的剩余秒数
            timeout: LLM 内层超时（秒），由上层按 settings.agent_llm_timeouts["PREDICT"] 传入

        Returns:
            ArbitrateOutput: 仲裁结论（selected_pattern_id 或放弃）。

        Raises:
            asyncio.TimeoutError: LLM 调用超过 timeout。
            Exception: 网络错误 / 重试 2 次后 Pydantic 校验仍失败等，均直接向上抛出。
        """
        user_message = self._build_arbitrate_user_msg(
            window_payload, candidates, remaining_seconds
        )
        logger.info(
            "开始仲裁 LLM 调用 | model={} | 冲突候选={} | remaining={}s | timeout={}s",
            settings.decision_model,
            len(candidates),
            remaining_seconds,
            timeout,
        )
        # 仲裁输出字段少，max_tokens 取 2048 与既有阶段调用一致
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=ArbitrateOutput,
                messages=[
                    {"role": "system", "content": ARBITRATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=2048,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        # 阶段标签保持 PREDICT：健康检查/成本核算按四阶段口径统计
        self._record_llm_usage("PREDICT", user_message, t0, completion)
        self._spawn_trace("PREDICT", ARBITRATE_SYSTEM_PROMPT, user_message, result, t0, completion)
        return result

    async def agent_evolve(
        self,
        all_patterns: list[dict],
        recent_predictions: list[dict],
        timeout: float,
    ) -> EvolveOutput:
        """
        进化阶段（Evolve Phase）结构化 LLM 调用（Req 5.4 / 7.1 / 7.2 / 7.3 / 7.4）。

        让 LLM 基于全部模式（含近期 RETIRED）与最近若干次预测的验证结果进行自我
        反思，返回结构化的 EvolveOutput（reasoning-first + 进化操作列表）。

        Args:
            all_patterns: 全部模式（含 ACTIVE 与近期 RETIRED，dict 列表，含 id 与最新统计）
            recent_predictions: 最近 N 次 Agent 预测记录及验证结果（dict 列表）
            timeout: LLM 内层超时（秒），由上层按 settings.agent_llm_timeouts["EVOLVE"] 传入

        Returns:
            EvolveOutput: 结构化进化结论（reasoning + operations[EvolveOperation]）。

        Raises:
            asyncio.TimeoutError: LLM 调用超过 timeout。
            Exception: 网络错误 / 重试 2 次后 Pydantic 校验仍失败等，均直接向上抛出。
        """
        # Evolve 为重载阶段：若预测记录内含原始曲线，按分钟下采样以压缩 token
        user_message = self._build_evolve_user_msg(all_patterns, recent_predictions)
        logger.info(
            "开始 Evolve LLM 调用 | model={} | all_patterns={} | recent_predictions={} | timeout={}s",
            settings.decision_model,
            len(all_patterns),
            len(recent_predictions),
            timeout,
        )
        # 照搬 decide() 已验证形态：Instructor + Pydantic 校验 + 自动重试 + 禁用 thinking
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=EvolveOutput,
                messages=[
                    {"role": "system", "content": EVOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=4096,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        self._record_llm_usage("EVOLVE", user_message, t0, completion)
        self._spawn_trace("EVOLVE", EVOLVE_SYSTEM_PROMPT, user_message, result, t0, completion)
        return result

    # ---- 三阶段 user message 组装辅助（无 I/O，纯文本拼装）----

    # ================================================================
    # LLM 轨迹落库（前端「LLM 轨迹」面板 / 流程审查）
    # 写入为 fire-and-forget：不阻塞主决策流程，失败仅告警。
    # ================================================================

    @staticmethod
    def _extract_usage(completion: object | None) -> tuple[int | None, int | None]:
        """从 raw completion 提取真实 token 用量，缺失返回 (None, None)。"""
        usage = getattr(completion, "usage", None) if completion is not None else None
        pt = ct = None
        if usage is not None:
            p = getattr(usage, "prompt_tokens", None)
            c = getattr(usage, "completion_tokens", None)
            if isinstance(p, int) and p > 0:
                pt = p
            if isinstance(c, int) and c >= 0:
                ct = c
        return pt, ct

    @staticmethod
    def _summarize_result(result: object) -> str | None:
        """生成关键结论摘要（供轨迹列表快速浏览）。"""
        direction = getattr(result, "direction", None)
        if direction is not None:
            conf = getattr(result, "confidence", 0.0) or 0.0
            timing = getattr(result, "entry_timing", "")
            return f"direction={direction} conf={conf:.2f} timing={timing}"[:200]
        # 仲裁输出（ArbitrateOutput）：无 direction 字段，按选定 id 摘要
        if hasattr(result, "selected_pattern_id"):
            conf = getattr(result, "confidence", 0.0) or 0.0
            timing = getattr(result, "entry_timing", "")
            return (
                f"selected={result.selected_pattern_id} "
                f"conf={conf:.2f} timing={timing}"
            )[:200]
        hypotheses = getattr(result, "hypotheses", None)
        if hypotheses is not None:
            return f"hypotheses={len(hypotheses)}"[:200]
        discoveries = getattr(result, "discoveries", None)
        if discoveries is not None:
            return f"discoveries={len(discoveries)}"[:200]
        operations = getattr(result, "operations", None)
        if operations is not None:
            return f"operations={len(operations)}"[:200]
        return None

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
            summary = self._summarize_result(result)

            task = asyncio.create_task(
                self._write_trace(
                    phase=phase,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    assistant_output=output_dict,
                    reasoning=reasoning,
                    result_summary=summary,
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

    def _record_llm_usage(
        self,
        phase: str,
        user_message: str,
        start_monotonic: float,
        completion: object | None = None,
    ) -> None:
        """记录 LLM 调用延迟与 token 用量到 MetricsCollector。

        Fix #18: 优先从 Instructor create_with_completion() 返回的 raw
        completion.usage 读取真实 token 数（prompt/completion）；仅当
        usage 不可用时回退到字符估算。估算采用 CJK 感知启发：
        中文符约 1 token/字，非中文约 4 字符/token，远比统一 len//4 准确。
        """
        latency = time.monotonic() - start_monotonic

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        # 优先取 API 返回的真实 usage
        usage = getattr(completion, "usage", None) if completion is not None else None
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            if isinstance(pt, int) and pt > 0:
                prompt_tokens = pt
            if isinstance(ct, int) and ct >= 0:
                completion_tokens = ct

        # usage 缺失时回退到 CJK 感知估算
        if prompt_tokens is None:
            prompt_tokens = self._estimate_tokens(user_message)
            is_estimated = True
        else:
            is_estimated = False
        if completion_tokens is None:
            completion_tokens = max(1, int(prompt_tokens * 0.3))

        est_cost = (
            prompt_tokens * settings.llm_input_price_per_1m
            + completion_tokens * settings.llm_output_price_per_1m
        ) / 1_000_000
        metrics_collector.record_llm_call(
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=est_cost,
            latency_s=latency,
        )
        logger.debug(
            "LLM 用量记录 | phase={} | latency={:.2f}s | tokens={}/{} ({}) | cost={:.6f}元",
            phase, latency, prompt_tokens, completion_tokens,
            "估算" if is_estimated else "真实", est_cost,
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """CJK 感知的 token 粗估（Fix #18）。

        中文/CJK 字符按 ~1 token/字计，其余（ASCII/标点/空白）按 ~4
        字符/token 计。较原本统一 len//4 在中英混合场景下更贴近实际。
        """
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        non_cjk = len(text) - cjk
        return max(1, int(cjk + non_cjk / 4))

    @staticmethod
    def _downsample_curve(curve: list[dict], step: int = 4) -> list[str]:
        """
        将情绪曲线按分钟下采样以压缩 token。

        采样点约每 15 秒一个，故每 step=4 个点约合 1 分钟取 1 点；始终保留末点以
        反映最新状态。返回形如 ["52.3%", ...] 的百分比字符串列表。
        沿用现有 sentiment_backtest 每分钟取点的做法。
        """
        if not curve:
            return []
        last_idx = len(curve) - 1
        sampled: list[str] = []
        for j, point in enumerate(curve):
            if j % step == 0 or j == last_idx:
                sampled.append(f"{point.get('v', 0):.1f}%")
        return sampled

    @staticmethod
    def _compute_curve_summary(
        up_curve: list[dict], down_curve: list[dict]
    ) -> str:
        """计算 UP/DOWN 曲线的统计摘要字符串，供 LLM 参考。

        提取趋势方向、变化幅度、均值/标准差、UP-DOWN 背离趋势等定性特征，
        减轻 LLM 的数值计算负担（Plan 步骤 6）。
        """
        up_vals = [p.get("v", 0) for p in up_curve if p.get("v") is not None]
        down_vals = [p.get("v", 0) for p in down_curve if p.get("v") is not None]

        if len(up_vals) < 2:
            return ""

        up_start, up_end = up_vals[0], up_vals[-1]
        up_change = up_end - up_start
        up_mean = sum(up_vals) / len(up_vals)
        up_std = (sum((v - up_mean) ** 2 for v in up_vals) / len(up_vals)) ** 0.5

        # 趋势方向判定
        if abs(up_change) < 2.0:
            up_trend = "横盘"
        elif up_change > 0:
            up_trend = "上升"
        else:
            up_trend = "下降"

        parts = [
            f"UP% 起始={up_start:.1f} 终止={up_end:.1f} 变化={up_change:+.1f}pp({up_trend})",
            f"均值={up_mean:.1f} 标准差={up_std:.1f}",
        ]

        # UP-DOWN 背离分析
        if down_vals and len(down_vals) >= 2:
            divergence_start = up_vals[0] - down_vals[0]
            divergence_end = up_vals[-1] - down_vals[-1]
            div_change = divergence_end - divergence_start
            if abs(div_change) < 2.0:
                div_trend = "平行"
            elif abs(div_change) > abs(divergence_start):
                div_trend = "背离扩大"
            else:
                div_trend = "收敛"
            parts.append(f"UP-DOWN 背离趋势={div_trend}")

        return "; ".join(parts)

    @staticmethod
    def _format_patterns_json(patterns: list[dict]) -> str:
        """
        将模式列表序列化为 JSON 文本，保留 id / 名称 / 特征 / 条件 / 统计等字段，
        供 LLM 逐一比对与在结论中引用（如 target_pattern_id）。
        """
        if not patterns:
            return "（空——模式库暂无相关模式）"
        # default=str 兜底处理 datetime 等非原生 JSON 类型
        return json.dumps(patterns, ensure_ascii=False, indent=2, default=str)

    def _format_predictions_json(self, predictions: list[dict]) -> str:
        """
        将最近预测记录序列化为 JSON 文本；若记录内含原始曲线（curve_up_pct /
        curve_down_pct），先按分钟下采样再序列化，以压缩 token。
        """
        if not predictions:
            return "（空——暂无最近预测记录）"
        compact: list[dict] = []
        for p in predictions:
            item = dict(p)
            for curve_key in ("curve_up_pct", "curve_down_pct"):
                if isinstance(item.get(curve_key), list):
                    item[curve_key] = self._downsample_curve(item[curve_key])
            compact.append(item)
        return json.dumps(compact, ensure_ascii=False, indent=2, default=str)

    def _build_learn_user_msg(
        self, windows: list[dict], active_patterns: list[dict]
    ) -> str:
        """组装 Learn 阶段 user message：历史窗口曲线（下采样）+ 统计摘要 + 当前 ACTIVE 模式库。"""
        lines: list[str] = [
            f"## 最近 {len(windows)} 个已归档情绪窗口（每约 1 分钟取 1 采样点，从早到晚）",
            "格式：窗口序号: UP%[...] / DOWN%[...] → outcome (实际收益)",
        ]
        for i, w in enumerate(windows, 1):
            up_curve = w.get("curve_up_pct", [])
            down_curve = w.get("curve_down_pct", [])
            up = self._downsample_curve(up_curve)
            down = self._downsample_curve(down_curve)
            outcome = w.get("outcome", "N/A")
            actual_return = w.get("actual_return", 0) or 0
            lines.append(
                f"窗口{i}: UP%[{','.join(up)}] / DOWN%[{','.join(down)}] "
                f"→ {outcome} (收益: {actual_return:+.4%})"
            )
            # 附加统计摘要（Plan 步骤 6）
            summary = self._compute_curve_summary(up_curve, down_curve)
            if summary:
                lines.append(f"  摘要: {summary}")
        lines.append("")
        lines.append(f"## 当前模式库 ACTIVE 模式（共 {len(active_patterns)} 个）")
        lines.append(self._format_patterns_json(active_patterns))
        lines.append("")
        lines.append(
            "请据此分析历史曲线形态，产出新建（CREATE）或更新（UPDATE）模式的结构化结论。"
        )
        return "\n".join(lines)

    def _build_discovery_user_msg(
        self,
        symbolized_windows: list[dict],
        feedback: dict | None = None,
        hints: list[dict] | None = None,
    ) -> str:
        """组装科学发现 user message：反馈区块（Q7-2）+ 三通道符号串 + 几何摘要 + outcome。

        输入为 symbolizer.build_window_view 的 dict 形态（宪法 Q1/Q2）：
        每项含 start_time / outcome / channels{channel: {symbols, geometry}}。
        LLM 只消费符号串与几何摘要，不再呈现原始曲线数值——原始数值的形态
        解释权归程序（分箱快照），LLM 只做符号层面的假设生成。

        feedback（Q7-2 反馈循环）置于窗口数据之前，让 LLM 带着排除约束读数据：
        - negatives：SPURIOUS 死亡假设的全量细节（波普尔排除，禁止重提）
        - positive_summary：存活模式统计摘要（不含谓词结构，防近亲繁殖）
        - lifespan_stats：EXPIRED 死亡的存活期分布（规律预期寿命元信息）
        """
        from datetime import datetime, timezone

        # P0-1 沿用：用实际窗口的 min/max start_time 计算真实跨度，
        # 避免"最近 N 个当全量历史"的误导。
        starts = [
            w.get("start_time", 0) for w in symbolized_windows if w.get("start_time")
        ]
        if starts:
            t_min = datetime.fromtimestamp(min(starts) / 1000, tz=timezone.utc)
            t_max = datetime.fromtimestamp(max(starts) / 1000, tz=timezone.utc)
            span_days = (max(starts) - min(starts)) / 86_400_000
            span_str = (
                f"覆盖 {t_min.strftime('%Y-%m-%d %H:%M')} ~ "
                f"{t_max.strftime('%Y-%m-%d %H:%M')} UTC（约 {span_days:.1f} 天）"
            )
        else:
            span_str = "时间范围未知"

        lines: list[str] = []

        # --- Q7-2 反馈区块（先于数据呈现）---
        lines.append("## 发现反馈（历史审判结果，务必遵守）")
        if not feedback:
            lines.append("暂无历史审判记录（首轮发现）。")
        else:
            negatives = feedback.get("negatives") or []
            if negatives:
                lines.append(
                    f"\n### 已被证伪的假设（{len(negatives)} 条，禁止重提相同或高度相似结构）"
                )
                lines.append(
                    "以下假设经程序统计审判确认为假规律（live 命中从未显著优于局部基准），"
                    "已被处决。与它们相同或仅参数微调（如计数阈值 ±1）的重提会被同样证伪，"
                    "不要浪费假设名额："
                )
                for i, neg in enumerate(negatives, 1):
                    pred_json = json.dumps(
                        neg.get("predicate"), ensure_ascii=False, separators=(",", ":")
                    )
                    lines.append(
                        f"{i}. 「{neg.get('pattern_name', '?')}」"
                        f"方向 {neg.get('predicted_direction', '?')}"
                        f" | live {neg.get('sample_count', 0)} 次命中"
                        f"胜率 {float(neg.get('win_rate') or 0.0):.2f}"
                        f"\n   谓词 {pred_json}"
                        f"\n   形态描述：{neg.get('description', '')}"
                    )
            else:
                lines.append("\n### 已被证伪的假设\n暂无（尚无假规律被处决）。")

            pos = feedback.get("positive_summary") or {}
            pos_count = int(pos.get("count") or 0)
            if pos_count:
                lines.append(
                    f"\n### 当前存活模式统计（结构对你不可见）\n"
                    f"存活 ACTIVE 模式 {pos_count} 个"
                    f"（UP {pos.get('up_count', 0)} / DOWN {pos.get('down_count', 0)}），"
                    f"平均 live 胜率 {float(pos.get('avg_win_rate') or 0.0):.2f}。\n"
                    "注意：为防全库同质化（近亲繁殖），存活模式的谓词结构不向你开放——"
                    "不要猜测或模仿它们，你的价值在于探索它们未覆盖的形态空间。"
                )
            else:
                lines.append("\n### 当前存活模式统计\n暂无存活模式（模式库为空）。")

            life = feedback.get("lifespan_stats") or {}
            life_count = int(life.get("count") or 0)
            if life_count:
                lines.append(
                    f"\n### 规律存活期分布（元信息）\n"
                    f"历史过期规律（曾显著后衰减）共 {life_count} 个，"
                    f"存活期：平均 {life.get('mean')} 天 / 中位 {life.get('median')} 天 / "
                    f"最长 {life.get('max')} 天。\n"
                    "这是规律的预期寿命量级：优先提出稳健、跨 regime 的结构，而非短期噪声。"
                )
        lines.append("")

        # --- 程序预筛线索榜单（假设矿机，置于窗口明细之前：先线索后下钻）---
        if hints:
            lines.extend([
                f"## 程序预筛线索榜单（训练集统计，按偏向强度降序，共 {len(hints)} 条）",
                "程序已在训练集上穷举全部单谓词组合（约 300 个）逐窗口执行，以下条目",
                "命中数达标且 outcome 偏向最强。统计口径与最终审判一致（局部基准 lift），",
                "但跑在训练集——榜单只是线索不是结论：你精选的每条假设仍会在留出集独立审判。",
                "格式：#编号 方向 lift(CI下界) 命中数(UP/DOWN/NOISE) 谓词JSON",
                "",
            ])
            for i, h in enumerate(hints, 1):
                pred_json = json.dumps(
                    h.get("predicate"), ensure_ascii=False, separators=(",", ":")
                )
                lines.append(
                    f"#{i:02d} {h.get('direction')} "
                    f"lift={h.get('lift')}({h.get('ci_lower')}) "
                    f"命中{h.get('hits')}(UP:{h.get('up_hits')} "
                    f"DOWN:{h.get('down_hits')} NOISE:{h.get('noise_hits')}) "
                    f"{pred_json}"
                )
            lines.append("")

        channel_names = ("sentiment", "price", "volume")
        lines.extend([
            f"## 发现集窗口（共 {len(symbolized_windows)} 个，按时间从旧到新）",
            f"时间范围：{span_str}",
            "格式：窗口序号 [时间] → outcome；随后三通道符号串与几何摘要",
            "注：「缺」= 该窗口此通道无有效数据",
            "",
        ])
        for i, w in enumerate(symbolized_windows, 1):
            outcome = w.get("outcome", "N/A")
            start_time = w.get("start_time", 0)
            ts = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
            time_str = ts.strftime("%m-%d %H:%M")
            lines.append(f"窗口{i} [{time_str}] → {outcome}")

            channels = w.get("channels") or {}
            for channel in channel_names:
                view = channels.get(channel) or {}
                symbols = view.get("symbols") or []
                if not symbols:
                    lines.append(f"  {channel}: 缺")
                else:
                    lines.append(f"  {channel}: {','.join(symbols)}")

            geo_parts: list[str] = []
            for channel in channel_names:
                view = channels.get(channel) or {}
                geo = view.get("geometry") or {}
                if not geo:
                    continue
                geo_parts.append(
                    f"{channel}(峰{geo.get('peak_count', 0)}"
                    f"/面积比{geo.get('area_ratio', 0.5):.2f}"
                    f"/卷曲{geo.get('curliness', 1.0):.1f}"
                    f"/间距{geo.get('extremum_spacing', 'insufficient')})"
                )
            if geo_parts:
                lines.append(f"  几何: {' '.join(geo_parts)}")
            lines.append("")

        if hints:
            lines.append(
                "请基于以上线索榜单与符号串明细提出可证伪的谓词假设（hypotheses 至多 20 条）："
                "优先从榜单精选有形态学意义的条目（rationale 引用 #编号），"
                "可对榜单谓词微调参数或做逻辑组合，也鼓励提出榜单外的新结构。"
            )
        else:
            lines.append(
                "请基于以上符号串与几何摘要提出可证伪的谓词假设（hypotheses 至多 20 条）。"
            )
        return "\n".join(lines)

    def _build_arbitrate_user_msg(
        self,
        window_payload: dict,
        candidates: list[dict],
        remaining_seconds: int,
    ) -> str:
        """组装仲裁 user message（宪法第八条规则 6）。

        当前窗口三通道符号串+几何摘要（与 Deep Learn 同口径，LLM 只消费符号
        层面信息）+ 冲突候选详情（含谓词定义与 live 统计）。候选序列化为
        JSON 呈现，谓词原样保留供 LLM 对照形态。
        """
        channel_names = ("sentiment", "price", "volume")
        channels = window_payload.get("channels") or {}

        lines: list[str] = [
            "## 当前窗口（进行中）",
            f"- 距窗口结束剩余：{remaining_seconds} 秒",
        ]
        for channel in channel_names:
            view = channels.get(channel) or {}
            symbols = view.get("symbols") or []
            if not symbols:
                lines.append(f"- {channel}: 缺")
            else:
                lines.append(f"- {channel}: {','.join(symbols)}")

        geo_parts: list[str] = []
        for channel in channel_names:
            view = channels.get(channel) or {}
            geo = view.get("geometry") or {}
            if not geo:
                continue
            geo_parts.append(
                f"{channel}(峰{geo.get('peak_count', 0)}"
                f"/面积比{geo.get('area_ratio', 0.5):.2f}"
                f"/卷曲{geo.get('curliness', 1.0):.1f}"
                f"/间距{geo.get('extremum_spacing', 'insufficient')})"
            )
        if geo_parts:
            lines.append(f"- 几何: {' '.join(geo_parts)}")

        lines.extend([
            "",
            f"## 冲突候选（谓词均已命中，共 {len(candidates)} 个，方向相异）",
            json.dumps(candidates, ensure_ascii=False, indent=2, default=str),
            "",
            "请仲裁：选定一个与当前窗口形态最契合的候选（selected_pattern_id），"
            "或判定信号矛盾放弃（留空）。",
        ])
        return "\n".join(lines)

    def _build_evolve_user_msg(
        self, all_patterns: list[dict], recent_predictions: list[dict]
    ) -> str:
        """组装 Evolve 阶段 user message：全部模式（含统计）+ 最近预测及验证结果。"""
        lines: list[str] = [
            f"## 全部模式（含 ACTIVE 与近期 RETIRED，共 {len(all_patterns)} 个，含最新统计）",
            self._format_patterns_json(all_patterns),
            "",
            f"## 最近 {len(recent_predictions)} 次 Agent 预测及验证结果",
            self._format_predictions_json(recent_predictions),
            "",
            "请据此进行自我反思，产出保留（RETAIN）/ 修正（MODIFY）/ 淘汰（RETIRE）/ 新增（CREATE）的结构化进化操作。",
        ]
        return "\n".join(lines)
