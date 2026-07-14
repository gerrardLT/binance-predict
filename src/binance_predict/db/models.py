"""
BTC 5min LLM 预测系统 V2 - SQLAlchemy ORM 模型

对应 V2 PRD §16 数据库设计。
所有表结构与 Pydantic schemas（models/schemas.py）严格对齐，
确保 LLM 输出 = DB 列 = API 响应（对应用户规则 7/8）。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    pass


# ============================================================
# 交易订单表（Binance Prediction Trading）
# ============================================================

class TradeOrderModel(Base):
    """
    预测市场交易订单记录

    每次 Agent 预测后，若 agent_auto_trade=true 且 prediction!=NO_TRADE，
    则通过 Binance Prediction Trading API 下单，记录在此表中。
    """
    __tablename__ = "trade_orders"
    __table_args__ = (
        Index("ix_trade_orders_prediction_id", "prediction_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="关联的预测 ID（旧 K 线决策路径，退役后不再写入）"
    )
    # Agent 预测关联（新增）：与旧 prediction_id 并存、互不干扰。
    # 与 agent_predictions.trade_order_id 形成相互外键引用（循环依赖），
    # 故在本侧显式 use_alter=True + 具名约束，令 create_all 通过 ALTER TABLE
    # 追加该外键、drop_all 可按名 DROP CONSTRAINT，避免循环依赖排序报错。
    agent_prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "agent_predictions.id",
            use_alter=True,
            name="fk_trade_orders_agent_prediction_id",
        ),
        nullable=True,
        comment="关联的 Agent 预测 ID（新增，与 prediction_id 互斥使用）",
    )
    market_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="Binance 预测市场 ID"
    )
    token_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Outcome Token ID"
    )
    side: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="BUY | SELL"
    )
    amount_in: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="输入金额（wei 格式）"
    )
    amount_out: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="输出金额（wei 格式）"
    )
    order_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Binance 返回的订单 ID"
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING",
        comment="PENDING | FILLED | FAILED"
    )
    quote_json: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="报价响应 JSON"
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="错误信息（仅 FAILED 时）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 预测市场情绪采样表（持久化 UP/DOWN token 报价）
# ============================================================

class PredictionMarketSample(Base):
    """预测市场情绪采样：每 15s 记录 UP/DOWN token 价格"""
    __tablename__ = "prediction_market_samples"
    __table_args__ = (
        Index("ix_pm_samples_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="毫秒时间戳"
    )
    up_price: Mapped[float | None] = mapped_column(Float, nullable=True, comment="UP token 价格")
    down_price: Mapped[float | None] = mapped_column(Float, nullable=True, comment="DOWN token 价格")
    up_pct: Mapped[float | None] = mapped_column(Float, nullable=True, comment="UP 百分比")
    down_pct: Mapped[float | None] = mapped_column(Float, nullable=True, comment="DOWN 百分比")
    participants: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="参与人数")
    trade_volume: Mapped[float | None] = mapped_column(Float, nullable=True, comment="交易量")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 情绪窗口表：每 5 分钟归档一条（情绪曲线 + 实际涨跌结果）
# ============================================================

class SentimentWindow(Base):
    """
    5 分钟情绪窗口：将一个 5m 窗口内的所有采样点聚合为一条记录

    用于：
    1. LLM 回测分析（历史曲线形态 + 实际结果 → 模式挖掘）
    2. 实时预测（当前曲线 + 历史模式 → 方向 + 入场时机）
    """
    __tablename__ = "sentiment_windows"
    __table_args__ = (
        Index("ix_sw_start_time", "start_time"),
        Index("ix_sw_outcome", "outcome"),
        UniqueConstraint("start_time", "end_time", name="uq_sw_start_end"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    start_time: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="窗口开始时间戳（ms，5 分钟整点）"
    )
    end_time: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="窗口结束时间戳（ms）"
    )
    curve_up_pct: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="UP% 时间序列 [{t, v}, ...]"
    )
    curve_down_pct: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="DOWN% 时间序列 [{t, v}, ...]"
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="窗口内采样点数"
    )
    # BTC 实际结果（用于回测）
    entry_price: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="窗口开始时 BTC 价格"
    )
    exit_price: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="窗口结束时 BTC 价格"
    )
    actual_return: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="实际收益率 (exit/entry - 1)"
    )
    outcome: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="实际结果: UP / DOWN / NOISE"
    )
    # 窗口元数据
    avg_participants: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="窗口内平均参与人数"
    )
    avg_trade_volume: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="窗口内平均交易量"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 情绪曲线自进化 Agent Loop - 模式记忆表（sentiment-agent-loop Req 1）
# ============================================================

class PatternMemory(Base):
    """
    情绪曲线模式记忆：由 LLM 在 Learn/Evolve 阶段自主发现与命名（Req 1.1）

    curve_features / conditions 为 LLM 自由结构 JSON，程序不做语义校验（Req 1.3）。
    win_rate 以 Harness 维护的 correct_count / sample_count 为精确来源，
    LLM 不填写 win_rate / sample_count / correct_count / status 等统计与状态字段。
    """
    __tablename__ = "pattern_memory"
    __table_args__ = (
        Index("ix_pattern_memory_name", "pattern_name"),    # Req 1.2 名称检索
        Index("ix_pattern_memory_status", "status"),        # Req 1.2 状态筛选
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_name: Mapped[str] = mapped_column(
        String(120), nullable=False, comment="LLM 自主命名"
    )
    description: Mapped[str] = mapped_column(
        Text, nullable=False, comment="模式描述，LLM 自由填写"
    )
    curve_features: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="曲线特征，LLM 自由结构（程序不做语义校验，Req 1.3）"
    )
    conditions: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="适用条件，LLM 自由结构"
    )
    predicted_direction: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="UP | DOWN"
    )
    win_rate: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="历史胜率 0~1"
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="样本数"
    )
    correct_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="命中数（Harness 维护，win_rate=correct_count/sample_count 的精确来源）"
    )
    confidence_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, comment="置信度 0~1"
    )
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="ACTIVE",
        comment="ACTIVE | RETIRED | EVOLVING"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ============================================================
# 情绪曲线自进化 Agent Loop - Agent 预测记录表（Req 3.5 / 8.1）
# ============================================================

class AgentPrediction(Base):
    """
    Sentiment_Agent 单次方向预测记录

    Predict 阶段写入方向/置信度/匹配模式/推理；Validate 阶段回填验证结果
    （is_correct/actual_outcome/actual_return/validated_at，Req 4.3）。
    trade_order_id 与 trade_orders.agent_prediction_id 双向一致（Req 10.3）。
    """
    __tablename__ = "agent_predictions"
    __table_args__ = (
        Index("ix_agent_pred_time", "prediction_time"),           # Req 8.3 时间筛选
        Index("ix_agent_pred_direction", "predicted_direction"),  # Req 8.3 方向筛选
        Index("ix_agent_pred_window", "sentiment_window_id"),     # Validate 关联
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    sentiment_window_id: Mapped[int | None] = mapped_column(
        ForeignKey("sentiment_windows.id"), nullable=True,
        comment="关联的情绪窗口（预测时窗口尚未归档可为空，Validate 时回填/匹配）"
    )
    predicted_direction: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="UP | DOWN | NO_TRADE"
    )
    matched_pattern_id: Mapped[int | None] = mapped_column(
        ForeignKey("pattern_memory.id"), nullable=True,
        comment="匹配的模式；无匹配/冷启动为空"
    )
    matched_pattern_name: Mapped[str | None] = mapped_column(
        String(120), nullable=True
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="置信度 0~1"
    )
    entry_timing: Mapped[str] = mapped_column(
        String(10), nullable=False, default="SKIP", comment="NOW | WAIT | SKIP"
    )
    reasoning: Mapped[str] = mapped_column(
        Text, nullable=False, comment="LLM 推理过程"
    )
    # --- 验证结果（Validate 阶段回填，Req 4.3）---
    is_correct: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, comment="未验证为 NULL"
    )
    actual_outcome: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="UP | DOWN | NOISE"
    )
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # --- 交易关联（Req 10.3）---
    trade_order_id: Mapped[int | None] = mapped_column(
        ForeignKey("trade_orders.id"), nullable=True,
        comment="关联交易订单；未交易为空"
    )
    skip_trade_reason: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="跳过交易的原因（Req 10.2，非静默）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 情绪曲线自进化 Agent Loop - 模式变更日志表（Req 1.4 / 8.2）
# ============================================================

class PatternChangeLog(Base):
    """
    模式变更日志：记录 CREATE/UPDATE/RETIRE 的完整前后快照与变更原因

    每次模式变更恰生成一条日志（与变更在同一事务提交，保证"有变更必有日志"）。
    CREATE 的 before_snapshot 为空；RETIRE 的 after_snapshot.status 置为 RETIRED。
    """
    __tablename__ = "pattern_change_log"
    __table_args__ = (
        Index("ix_pcl_pattern_id", "pattern_id"),
        Index("ix_pcl_created_at", "created_at"),   # Req 8.5 时间正序
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_id: Mapped[int] = mapped_column(
        ForeignKey("pattern_memory.id"), nullable=False
    )
    change_type: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="CREATE | UPDATE | RETIRE"
    )
    phase: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="触发阶段 LEARN | EVOLVE"
    )
    before_snapshot: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="变更前完整快照；CREATE 为 NULL"
    )
    after_snapshot: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="变更后完整快照；RETIRE 时为置为 RETIRED 后的快照"
    )
    change_reason: Mapped[str] = mapped_column(
        Text, nullable=False, comment="变更原因，LLM 提供"
    )
    evolve_phase_id: Mapped[str | None] = mapped_column(
        String(40), nullable=True,
        comment="触发该变更的 Evolve 执行 ID（LEARN 触发时为 NULL，Req 8.2）",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# LLM 调用轨迹审计表（前端「LLM 轨迹」面板 / 流程审查用）
# ============================================================

class LLMTrace(Base):
    """
    LLM 单次调用的完整轨迹审计记录。

    覆盖 Sentiment_Agent 四个 LLM 阶段（LEARN / DEEP_LEARN / PREDICT / EVOLVE）
    每次调用的系统提示词、用户输入、结构化输出（含 reasoning）、token 用量、
    耗时与估算成本。用于人工审查「LLM 被喂了什么、想了什么、决定了什么」，
    判断整套自进化流程是否偏离预期。

    写入为 fire-and-forget（不阻塞主决策流程），失败仅告警不影响预测。
    """
    __tablename__ = "llm_traces"
    __table_args__ = (
        Index("ix_llm_traces_created_at", "created_at"),
        Index("ix_llm_traces_phase", "phase"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    phase: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="LEARN | DEEP_LEARN | PREDICT | EVOLVE"
    )
    model: Mapped[str] = mapped_column(
        String(60), nullable=False, comment="调用的模型名"
    )
    system_prompt: Mapped[str] = mapped_column(
        Text, nullable=False, comment="完整系统提示词"
    )
    user_message: Mapped[str] = mapped_column(
        Text, nullable=False, comment="完整用户输入（含曲线/模式库上下文）"
    )
    assistant_output: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="LLM 结构化输出完整 JSON（含 reasoning 与结论）"
    )
    reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="LLM 推理文本（从 assistant_output.reasoning 抽取，便于列表展示）"
    )
    result_summary: Mapped[str | None] = mapped_column(
        String(200), nullable=True,
        comment="关键结论摘要（如 direction=UP conf=0.72 / discoveries=3）"
    )
    prompt_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="输入 token（真实或估算）"
    )
    completion_tokens: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="输出 token"
    )
    estimated_cost_yuan: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="估算成本（元）"
    )
    latency_s: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="LLM 调用耗时（秒）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# Agent 运行健康快照表（监控系统定期落库，供趋势回看 / LLM 诊断）
# ============================================================

class HealthSnapshot(Base):
    """
    Agent 运行健康报告的持久化快照。

    后台监控循环按 settings.agent_health_snapshot_interval 周期，将
    HealthService.build_report 产出的完整报告落库一条。overall_status 与
    alert_count 单列冗余存储，便于按状态/时间快速筛选；report 保存完整 JSON
    以便回看当时的全部指标（窗口连续性/匹配率/校准/调度器/LLM）。
    """
    __tablename__ = "health_snapshots"
    __table_args__ = (
        Index("ix_health_snapshots_created_at", "created_at"),
        Index("ix_health_snapshots_status", "overall_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    overall_status: Mapped[str] = mapped_column(
        String(10), nullable=False, comment="OK | WARN | CRITICAL"
    )
    alert_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0", comment="当次告警条数"
    )
    report: Mapped[dict] = mapped_column(
        JSONB, nullable=False, comment="HealthReport 完整 JSON"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
