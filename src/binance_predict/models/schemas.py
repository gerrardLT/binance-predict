"""情绪曲线自进化 Agent Loop 的数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# 与 Agent Loop 共用的枚举。
FinalPrediction = Literal["UP", "DOWN", "NO_TRADE"]
ActualLabel = Literal["UP", "DOWN", "NOISE"]
PatternStatus = Literal["ACTIVE", "RETIRED", "EVOLVING"]
PatternDirection = Literal["UP", "DOWN"]
ChangeType = Literal["CREATE", "UPDATE", "RETIRE"]
EvolveAction = Literal["RETAIN", "MODIFY", "RETIRE", "CREATE"]
EntryTiming = Literal["NOW", "WAIT", "SKIP"]


# ============================================================
# LLM 输出契约（Instructor response_model，reasoning-first）
# ============================================================

class PatternDiscovery(BaseModel):
    """Learn 阶段单条模式发现或更新。"""

    operation: Literal["CREATE", "UPDATE"] = Field(
        description="操作类型：CREATE 新建模式 / UPDATE 更新既有模式"
    )
    target_pattern_id: int | None = Field(
        default=None, description="UPDATE 时指向被更新的 Pattern id；CREATE 时为空"
    )
    pattern_name: str = Field(description="LLM 自主命名的模式名称")
    description: str = Field(description="模式描述")
    curve_features: dict = Field(default_factory=dict, description="曲线特征")
    conditions: dict = Field(default_factory=dict, description="适用条件")
    predicted_direction: PatternDirection = Field(description="模式预测方向 UP | DOWN")
    confidence_score: float = Field(description="模式置信度 0~1", ge=0, le=1)
    change_reason: str = Field(description="创建或更新该模式的原因")


class LearnOutput(BaseModel):
    """Learn 阶段 LLM 结构化输出。"""

    reasoning: str = Field(description="历史曲线分析推理过程")
    discoveries: list[PatternDiscovery] = Field(
        default_factory=list, description="本次发现或更新的模式"
    )


class PredictOutput(BaseModel):
    """Predict 阶段 LLM 结构化输出。"""

    reasoning: str = Field(description="当前曲线与模式匹配的推理过程")
    direction: FinalPrediction = Field(description="预测方向 UP | DOWN | NO_TRADE")
    matched_pattern_name: str | None = Field(default=None, description="匹配的模式名称")
    matched_pattern_id: int | None = Field(default=None, description="匹配的模式 id")
    confidence: float = Field(description="预测置信度 0~1", ge=0, le=1)
    entry_timing: EntryTiming = Field(default="SKIP", description="入场时机 NOW | WAIT | SKIP")
    entry_reason: str = Field(default="", description="入场或跳过时机的说明")


class EvolveOperation(BaseModel):
    """Evolve 阶段单条进化操作。"""

    action: EvolveAction = Field(description="RETAIN | MODIFY | RETIRE | CREATE")
    target_pattern_id: int | None = Field(default=None, description="目标 Pattern id")
    modifications: dict | None = Field(default=None, description="MODIFY 时的字段增量")
    new_pattern: PatternDiscovery | None = Field(default=None, description="CREATE 时的新模式")
    reason: str = Field(description="该进化操作的理由")


class EvolveOutput(BaseModel):
    """Evolve 阶段 LLM 结构化输出。"""

    reasoning: str = Field(description="模式有效性的自我反思推理过程")
    operations: list[EvolveOperation] = Field(default_factory=list, description="进化操作列表")


# ============================================================
# DB ↔ API 记录模型
# ============================================================

class PatternMemoryRecord(BaseModel):
    """PatternMemory 的数据库与 API 记录。"""

    id: int | None = Field(default=None, description="模式唯一标识")
    pattern_name: str = Field(description="LLM 自主命名的模式名称")
    description: str = Field(description="模式描述")
    curve_features: dict = Field(default_factory=dict, description="曲线特征 JSONB")
    conditions: dict = Field(default_factory=dict, description="适用条件 JSONB")
    predicted_direction: PatternDirection = Field(description="预测方向 UP | DOWN")
    win_rate: float = Field(default=0.0, description="历史胜率 0~1", ge=0, le=1)
    sample_count: int = Field(default=0, description="已验证预测数", ge=0)
    correct_count: int = Field(default=0, description="命中数", ge=0)
    confidence_score: float = Field(default=0.5, description="模式置信度 0~1", ge=0, le=1)
    status: PatternStatus = Field(default="ACTIVE", description="ACTIVE | RETIRED | EVOLVING")
    created_at: datetime | None = Field(default=None, description="创建时间 UTC")
    updated_at: datetime | None = Field(default=None, description="最后更新时间 UTC")


class AgentPredictionRecord(BaseModel):
    """AgentPrediction 的数据库与 API 记录。"""

    id: int | None = Field(default=None, description="预测记录唯一标识")
    prediction_time: datetime = Field(description="预测生成时间 UTC")
    sentiment_window_id: int | None = Field(default=None, description="关联的情绪窗口 id")
    predicted_direction: FinalPrediction = Field(description="预测方向 UP | DOWN | NO_TRADE")
    matched_pattern_id: int | None = Field(default=None, description="匹配的模式 id")
    matched_pattern_name: str | None = Field(default=None, description="匹配的模式名称")
    confidence: float = Field(description="预测置信度 0~1", ge=0, le=1)
    entry_timing: EntryTiming = Field(default="SKIP", description="入场时机 NOW | WAIT | SKIP")
    reasoning: str = Field(description="LLM 推理过程")
    is_correct: bool | None = Field(default=None, description="预测是否正确；未验证为空")
    actual_outcome: ActualLabel | None = Field(default=None, description="实际结果")
    actual_return: float | None = Field(default=None, description="实际收益率")
    validated_at: datetime | None = Field(default=None, description="验证时间 UTC")
    trade_order_id: int | None = Field(default=None, description="关联的交易订单 id")
    skip_trade_reason: str | None = Field(default=None, description="跳过交易的原因")
    created_at: datetime | None = Field(default=None, description="记录创建时间 UTC")


class PatternChangeLogRecord(BaseModel):
    """PatternChangeLog 的数据库与 API 记录。"""

    id: int | None = Field(default=None, description="变更日志唯一标识")
    pattern_id: int = Field(description="发生变更的模式 id")
    change_type: ChangeType = Field(description="CREATE | UPDATE | RETIRE")
    phase: str = Field(description="触发阶段 LEARN | EVOLVE")
    before_snapshot: dict | None = Field(default=None, description="变更前完整快照")
    after_snapshot: dict | None = Field(default=None, description="变更后完整快照")
    change_reason: str = Field(description="变更原因")
    evolve_phase_id: str | None = Field(default=None, description="触发该变更的 Evolve 执行 ID")
    created_at: datetime | None = Field(default=None, description="变更时间 UTC")


# ============================================================
# API 请求模型
# ============================================================

class CommitDeepLearnRequest(BaseModel):
    """深度分析确认写入请求。"""

    discoveries: list[PatternDiscovery] = Field(
        default_factory=list,
        description="用户确认后的发现列表（来自 POST /api/sentiment/agent/deep-learn 的返回值）"
    )


# ============================================================
# LLM 调用轨迹（前端「LLM 轨迹」面板）
# ============================================================

class LLMTraceSummary(BaseModel):
    """LLM 轨迹列表项（不含完整 prompt，供 5s 轮询列表使用）。"""

    id: int = Field(description="轨迹唯一标识")
    phase: str = Field(description="LEARN | DEEP_LEARN | PREDICT | EVOLVE")
    model: str = Field(description="调用的模型名")
    reasoning: str | None = Field(default=None, description="LLM 推理文本")
    result_summary: str | None = Field(default=None, description="关键结论摘要")
    prompt_tokens: int | None = Field(default=None, description="输入 token")
    completion_tokens: int | None = Field(default=None, description="输出 token")
    estimated_cost_yuan: float | None = Field(default=None, description="估算成本（元）")
    latency_s: float | None = Field(default=None, description="调用耗时（秒）")
    created_at: datetime | None = Field(default=None, description="调用时间 UTC")


class LLMTraceRecord(LLMTraceSummary):
    """LLM 轨迹完整详情（含系统提示词、用户输入与完整输出）。"""

    system_prompt: str = Field(default="", description="完整系统提示词")
    user_message: str = Field(default="", description="完整用户输入")
    assistant_output: dict | None = Field(default=None, description="LLM 结构化输出完整 JSON")
