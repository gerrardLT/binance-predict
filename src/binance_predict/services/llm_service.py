"""
BTC 5min LLM 预测系统 V2 - LLM 决策服务（核心模块）

核心逻辑：每次决策前通过 prompt + LLM 多轮验证反馈沟通得到可靠决策。

实现方式：
1. 使用 Instructor 的 Tool Calling 模式（Mode.Tools）调用 deepseek-v4-pro / qwen3.7-max
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
- 决策模型: deepseek-v4-pro（输入 12 元/百万Token，输出 24 元/百万Token）
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
    EvolveOutput,
    LearnOutput,
    PredictOutput,
)
from ..prompts.agent_templates import (
    DEEP_LEARN_SYSTEM_PROMPT,
    EVOLVE_SYSTEM_PROMPT,
    LEARN_SYSTEM_PROMPT,
    PREDICT_SYSTEM_PROMPT,
)
from .metrics import metrics_collector


class LLMService:
    """
    LLM 决策与复盘服务

    两个模型走不同的 API 通道：
    - 决策模型 deepseek-v4-pro → DeepSeek 原生 API (api.deepseek.com)
    - 复盘模型 qwen3.7-max → 百炼 DashScope API (dashscope.aliyuncs.com)

    两个模型均不支持原生 Structured Output，
    因此使用 Instructor 默认的 Mode.Tools（Tool Calling 模式）。

    注意：instructor.from_provider("openai/...", base_url=...) 会忽略 base_url，
    因此改用手动创建 AsyncOpenAI 客户端 + instructor.from_openai() 包装。
    """

    def __init__(self) -> None:
        # --- 决策 LLM 客户端（deepseek-v4-pro → DeepSeek 原生 API）---
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
        return result

    async def agent_deep_learn(
        self,
        windows: list[dict],
        active_patterns: list[dict],
        timeout: float,
    ) -> LearnOutput:
        """
        深度模式发现（手动触发）：全量历史分析。

        与 agent_learn() 的区别：
        - 输入：全量原始窗口 dict（不压缩，保留完整曲线形态）
        - max_tokens：16384（基于实测，允许充分推理输出）
        - 专用 prompt：强调跨周期深度分析，NOISE 过滤由 prompt 引导

        Args:
            windows: 全量 SentimentWindow 原始数据（dict 列表）
            active_patterns: 当前 ACTIVE 模式库
            timeout: LLM 超时秒数

        Returns:
            LearnOutput: 结构化发现结果
        """
        user_message = self._build_deep_learn_user_msg(windows, active_patterns)
        logger.info(
            "开始深度分析 LLM 调用 | model={} | windows={} | active_patterns={} | timeout={}s",
            settings.decision_model,
            len(windows),
            len(active_patterns),
            timeout,
        )
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=LearnOutput,
                messages=[
                    {"role": "system", "content": DEEP_LEARN_SYSTEM_PROMPT},
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
        return result

    async def agent_predict(
        self,
        current_curve: list[dict],
        active_patterns: list[dict],
        remaining_seconds: int,
        timeout: float,
    ) -> PredictOutput:
        """
        预测阶段（Predict Phase）结构化 LLM 调用（Req 3.4 / 7.1 / 7.2 / 7.3 / 7.4）。

        将当前窗口的实时情绪曲线与已有 ACTIVE 模式匹配，返回结构化预测结果
        PredictOutput（reasoning-first + 方向 / 置信度 / 匹配模式 / 入场时机）。

        Args:
            current_curve: 当前窗口已采集的实时曲线（[{t, v}, ...]，v 为 UP%）
            active_patterns: 当前 Pattern_Memory 中所有 ACTIVE 模式（dict 列表，含 id 与特征）
            remaining_seconds: 距当前窗口结束的剩余秒数
            timeout: LLM 内层超时（秒），由上层按 settings.agent_llm_timeouts["PREDICT"] 传入

        Returns:
            PredictOutput: 结构化预测结论。

        Raises:
            asyncio.TimeoutError: LLM 调用超过 timeout。
            Exception: 网络错误 / 重试 2 次后 Pydantic 校验仍失败等，均直接向上抛出。
        """
        # Predict 时间敏感、当前曲线点数少（单窗口 ≤ 约 20 点），完整呈现不下采样
        user_message = self._build_predict_user_msg(
            current_curve, active_patterns, remaining_seconds
        )
        logger.info(
            "开始 Predict LLM 调用 | model={} | curve_points={} | active_patterns={} | remaining={}s | timeout={}s",
            settings.decision_model,
            len(current_curve),
            len(active_patterns),
            remaining_seconds,
            timeout,
        )
        # 照搬 decide() 已验证形态；max_tokens 取 2048（design.md 示例值）
        t0 = time.monotonic()
        result, completion = await asyncio.wait_for(
            self._decision_client.create_with_completion(
                response_model=PredictOutput,
                messages=[
                    {"role": "system", "content": PREDICT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_retries=2,
                max_tokens=2048,
                temperature=0.1,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=timeout,
        )
        self._record_llm_usage("PREDICT", user_message, t0, completion)
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
        return result

    # ---- 三阶段 user message 组装辅助（无 I/O，纯文本拼装）----

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

    def _build_deep_learn_user_msg(
        self, windows: list[dict], active_patterns: list[dict]
    ) -> str:
        """组装深度分析 user message：全量窗口原始曲线（不下采样）+ 统计摘要 + ACTIVE 模式库。

        不压缩、不筛选、不下采样，所有窗口全量原始点呈现给 LLM，让其自主判断。
        """
        from datetime import datetime, timezone

        lines: list[str] = [
            f"## 全量历史窗口（共 {len(windows)} 个，按时间从新到旧，每 15 秒一个采样点）",
            "格式：窗口序号 [时间]: UP%[...] / DOWN%[...] → outcome (实际收益)",
            "",
        ]
        for i, w in enumerate(windows, 1):
            up_curve = w.get("curve_up_pct", []) or []
            down_curve = w.get("curve_down_pct", []) or []
            # 全量原始点，不下采样
            up = [f"{p.get('v', 0):.1f}%" for p in up_curve]
            down = [f"{p.get('v', 0):.1f}%" for p in down_curve]
            outcome = w.get("outcome", "N/A")
            actual_return = w.get("actual_return", 0) or 0

            # 格式化时间
            start_time = w.get("start_time", 0)
            ts = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
            time_str = ts.strftime("%m-%d %H:%M")

            lines.append(
                f"窗口{i} [{time_str}]: UP%[{','.join(up)}] / DOWN%[{','.join(down)}] "
                f"→ {outcome} (收益: {actual_return:+.4%})"
            )
            # 附加统计摘要（减轻 LLM 数值计算负担）
            summary = self._compute_curve_summary(up_curve, down_curve)
            if summary:
                lines.append(f"  摘要: {summary}")
            lines.append("")

        lines.append(f"## 当前模式库 ACTIVE 模式（共 {len(active_patterns)} 个）")
        lines.append(self._format_patterns_json(active_patterns))
        lines.append("")
        lines.append(
            "请分析全量历史窗口的完整曲线形态，自主判断哪些窗口值得重点关注、"
            "哪些可以忽略，发现跨周期可复现的模式。"
        )
        return "\n".join(lines)

    def _build_predict_user_msg(
        self,
        current_curve: list[dict],
        active_patterns: list[dict],
        remaining_seconds: int,
    ) -> str:
        """组装 Predict 阶段 user message：当前实时曲线 + 统计摘要 + ACTIVE 模式库 + 剩余时间。"""
        up_values = [f"{p.get('up_pct', 0):.1f}%" for p in current_curve]
        down_values = [f"{p.get('down_pct', 0):.1f}%" for p in current_curve]

        # 构建统计摘要（Plan 步骤 6）
        up_curve_raw = [{"v": p.get("up_pct", 0)} for p in current_curve]
        down_curve_raw = [{"v": p.get("down_pct", 0)} for p in current_curve]
        summary = self._compute_curve_summary(up_curve_raw, down_curve_raw)

        lines: list[str] = [
            "## 当前窗口实时状态",
            f"- 已采集采样点：{len(current_curve)} 个（约每 15 秒 1 点）",
            f"- 距窗口结束剩余：{remaining_seconds} 秒",
            f"- UP% 实时曲线（从早到晚）：[{','.join(up_values)}]",
            f"- DOWN% 实时曲线（从早到晚）：[{','.join(down_values)}]",
        ]
        if summary:
            lines.append(f"- 统计摘要: {summary}")
        lines.extend([
            "",
            f"## 可参考的 ACTIVE 模式库（共 {len(active_patterns)} 个）",
            self._format_patterns_json(active_patterns),
            "",
            "请将当前曲线与上述模式匹配，给出方向预测、置信度与入场时机。",
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
