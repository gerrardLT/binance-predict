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
# 发现方法：纯 LLM 深度发现 / Python 确定性聚类 / 存量或 learn·evolve 产出
DiscoveryMethod = Literal["LLM_DEEP", "PY_CLUSTER", "LEGACY"]


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


class DeepLearnDiscovery(PatternDiscovery):
    """Deep Learn 预览/提交用发现项：在 LLM·聚类发现基础上附加发现方法与样本外统计。

    holdout_* 由程序在留出集上计算后附加（非 LLM 填写）；commit 时据此做准入闸门
    与写库。作为预览端点返回与 commit 请求的元素类型，不污染作为 LLM response_model 的 PatternDiscovery。

    科学发现轨（Phase 2）：predicate 非空即谓词假设，准入闸门切换为 Q6 双轨裁决
    （screen_verdict），holdout_* 三列不再由新轨填充（旧轨 pycluster 保留）。
    """

    discovery_method: DiscoveryMethod = Field(
        default="LLM_DEEP", description="发现方法 LLM_DEEP | PY_CLUSTER"
    )
    holdout_win_rate: float | None = Field(default=None, description="holdout 胜率 0~1")
    holdout_sample_count: int | None = Field(default=None, description="holdout 命中判定样本数")
    holdout_ci_lower: float | None = Field(
        default=None, description="holdout 胜率 Wilson 95% 置信下界"
    )
    # --- 科学发现轨（谓词假设，Q5/Q6；非空即走新轨闸门）---
    predicate: dict | None = Field(
        default=None, description="谓词 DSL JSON（Q5 白名单，程序确定性执行）"
    )
    binning_version: str | None = Field(
        default=None, description="发现时的分箱快照版本（Q4）"
    )
    screen_verdict: str | None = Field(
        default=None, description="初筛裁决 ACTIVE | OBSERVE | REJECT（Q6，程序计算）"
    )
    screen_lift: float | None = Field(default=None, description="初筛 lift 点估计")
    screen_ci_lower: float | None = Field(default=None, description="初筛 log-lift CI 下界")
    screen_ci_upper: float | None = Field(default=None, description="初筛 log-lift CI 上界")
    screen_hit_count: int | None = Field(default=None, description="初筛验证集命中窗口数")
    screen_reject_reason: str | None = Field(
        default=None, description="REJECT 原因（非静默，反馈 LLM 用）"
    )
    # --- 经济闸证据（V1.1，Q6 第 5 步；仅双轨 ACTIVE 候选计算，其余为 None）---
    screen_ev: float | None = Field(
        default=None, description="费后 EV 点估计（入场价口径，含溢价）"
    )
    screen_ev_ci_lower: float | None = Field(
        default=None, description="EV bootstrap 95% CI 下界"
    )
    screen_ev_ci_upper: float | None = Field(
        default=None, description="EV bootstrap 95% CI 上界"
    )
    screen_ev_fires: int | None = Field(
        default=None, description="决策点截断视图上的命中注数"
    )
    screen_ev_passed: bool | None = Field(
        default=None, description="经济闸是否通过（未过则裁决降级 OBSERVE）"
    )


# ============================================================
# 科学发现系统（scientific-discovery 宪法 Phase 2）——谓词假设 LLM 契约
# ============================================================

class PredicateHypothesis(BaseModel):
    """Deep Learn 新轨 LLM 输出的单条谓词假设（Q5 DSL）。

    LLM 是假设生成器：只产出 predicate + target_outcome + 命名/描述，
    统计审判由程序完成（discovery.screen_hypotheses），LLM 永远不得自我验证。
    """

    pattern_name: str = Field(description="LLM 自主命名的模式名称")
    description: str = Field(description="假设的自然语言描述（形态直觉、预期结构）")
    predicate: dict = Field(
        description="谓词 DSL JSON（Q5 白名单：L1 五谓词 + L2 两谓词 + AND/OR/NOT ≤2 层）"
    )
    target_outcome: Literal["UP", "DOWN"] = Field(
        description="假设预测的目标结果（谓词命中时期望 outcome 偏向它）"
    )
    confidence_score: float = Field(
        description="主观置信度 0~1（仅先验参考，准入以程序初筛为准）", ge=0, le=1
    )
    rationale: str = Field(description="提出该假设的形态学理由")


class DiscoveryOutput(BaseModel):
    """Deep Learn 新轨 LLM 结构化输出（科学发现轨）。"""

    reasoning: str = Field(description="符号串与几何摘要的分析推理过程（先于结论）")
    hypotheses: list[PredicateHypothesis] = Field(
        default_factory=list, description="谓词假设列表（≤20 条，Q7 发现预算）"
    )


class ArbitrateOutput(BaseModel):
    """Predict 阶段 LLM 仲裁输出（科学发现宪法第八条，Phase 3）。

    谓词命中冲突（多模式异向）时，LLM 仅作仲裁者消歧：从冲突候选中
    选定一个模式或全部放弃。direction 不在输出中——由程序从选中模式的
    predicted_direction 推导，LLM 无权发明方向（角色分离）。
    """

    reasoning: str = Field(description="冲突消歧的推理过程（先于结论）")
    selected_pattern_id: int | None = Field(
        default=None, description="选定的候选模式 id；None 表示冲突不可调和、放弃交易"
    )
    confidence: float = Field(description="对选定模式的把握 0~1", ge=0, le=1)
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
    discovery_method: DiscoveryMethod = Field(
        default="LEGACY", description="发现方法 LLM_DEEP | PY_CLUSTER | LEGACY"
    )
    holdout_win_rate: float | None = Field(default=None, description="发现时 holdout 胜率 0~1")
    holdout_sample_count: int | None = Field(default=None, description="发现时 holdout 样本数")
    holdout_ci_lower: float | None = Field(
        default=None, description="发现时 holdout 胜率 Wilson 95% 置信下界"
    )
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

    discoveries: list[DeepLearnDiscovery] = Field(
        default_factory=list,
        description="用户确认后的发现列表（来自 deep-learn / deep-learn/pycluster 预览返回，每条携带 discovery_method 与 holdout 统计）"
    )
    snapshot_token: str | None = Field(
        default=None, description="预览时返回的窗口快照标记（hash of window_ids），commit 时回传校验一致性"
    )


class ManualTradeTestRequest(BaseModel):
    """实盘链路人工测试单请求（POST /api/trade/test，小额验证全链路）。"""

    amount_usdt: float = Field(
        default=1.0, description="测试单金额（USDT），端点硬限 0.1~5"
    )
    prediction: str = Field(
        default="DOWN", description="方向 UP/DOWN（测试单随意，验证链路为目的）"
    )


class TransferInboundRequest(BaseModel):
    """现货→预测钱包划转入金请求（POST /api/prediction/transfer-in）。

    预测市场下单扣预测钱包内余额，现货余额充足仍报 -9000 时需先划转。
    """

    amount_usdt: float = Field(
        default=1.0, description="划转金额（USDT），端点硬限 0.1~20"
    )


class TransferOutboundRequest(BaseModel):
    """预测钱包→现货划出提走请求（POST /api/prediction/transfer-out，P1-1）。

    ⚠️ 官方端点命名反转：inbound=钱包→CEX 提走。响应含划转前后
    现货余额（spot_before/spot_after/direction_confirmed）自证方向。
    """

    amount_usdt: float = Field(
        default=1.0, description="划出金额（USDT），端点硬限 0.1~20（金丝雀先 0.1）"
    )


class ToggleLiveRequest(BaseModel):
    """POST /api/live/toggle（QuoteEdgeLiveTrader 运行时开关，P2-1）。

    实时控制实盘开火状态；重启后回落.env 默认（fail-safe）。
    """

    enabled: bool = Field(description="启用=True / 禁用=False")


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


# ============================================================
# Agent 运行健康监控（services/health.py + GET /api/agent/health）
# ============================================================

HealthStatus = Literal["OK", "WARN", "CRITICAL"]


class CalibrationBucket(BaseModel):
    """置信度分桶校准：某置信度区间的预测数、平均置信度与实际命中率。"""

    range: str = Field(description="置信度区间，如 '0.60~0.70'")
    count: int = Field(description="该区间已验证（is_correct 非空）预测数", ge=0)
    avg_confidence: float = Field(description="该区间平均置信度")
    hit_rate: float | None = Field(default=None, description="实际命中率 correct/count；该区间无样本为空")
    gap: float | None = Field(
        default=None, description="avg_confidence - hit_rate；正值=过度自信，负值=过度保守"
    )


class HealthAlert(BaseModel):
    """单条健康告警。"""

    level: Literal["WARN", "CRITICAL"] = Field(description="告警级别")
    code: str = Field(description="告警码，如 WINDOW_STALE | NO_MATCH | LLM_FAILURES")
    message: str = Field(description="人类可读告警说明")


class HealthReport(BaseModel):
    """Agent 运行健康报告：5 类指标聚合 + 派生状态 + 告警 + 自然语言诊断。

    既作为 GET /api/agent/health 的响应体，也是后台轮询落库（HealthSnapshot）
    与 CLI 打印的统一数据结构。summary 为自然语言诊断，供人或 LLM 一眼读懂。
    """

    generated_at: datetime = Field(description="报告生成时间 UTC")
    overall_status: HealthStatus = Field(description="总体状态 OK | WARN | CRITICAL")
    alerts: list[HealthAlert] = Field(default_factory=list, description="当前触发的告警")
    window_continuity: dict = Field(
        default_factory=dict,
        description="窗口连续性：last_window_age_s / gap_count / recent_count / expected_interval_s",
    )
    predict_stats: dict = Field(
        default_factory=dict,
        description="predict 统计：total / matched / match_rate / direction_distribution / active_pattern_count",
    )
    calibration: list[CalibrationBucket] = Field(
        default_factory=list, description="置信度分桶校准表（样本不足时为空）"
    )
    scheduler: dict = Field(
        default_factory=dict,
        description="调度器：queue_depth / phase_ages_s / uptime_seconds（内存态，CLI --db-only 时为空）",
    )
    llm: dict = Field(
        default_factory=dict,
        description="LLM：call_count / total_cost / phase_success_rates / consecutive_failures（内存态）",
    )
    summary: str = Field(default="", description="自然语言诊断，供人/LLM 一眼读懂")
