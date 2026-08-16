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
    """预测市场情绪采样：每 15s 记录 UP/DOWN token 价格（5m/15m 双市场）"""
    __tablename__ = "prediction_market_samples"
    __table_args__ = (
        Index("ix_pm_samples_timestamp", "timestamp"),
        Index("ix_pm_samples_period_ts", "market_period", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="毫秒时间戳"
    )
    market_period: Mapped[str] = mapped_column(
        String(5), nullable=False, default="5m", server_default="5m",
        comment="预测市场周期：5m | 15m（存量行回填 5m，语义不变）"
    )
    up_price: Mapped[float | None] = mapped_column(Float, nullable=True, comment="UP token 价格")
    down_price: Mapped[float | None] = mapped_column(Float, nullable=True, comment="DOWN token 价格")
    up_pct: Mapped[float | None] = mapped_column(Float, nullable=True, comment="UP 百分比")
    down_pct: Mapped[float | None] = mapped_column(Float, nullable=True, comment="DOWN 百分比")
    participants: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="参与人数")
    trade_volume: Mapped[float | None] = mapped_column(Float, nullable=True, comment="交易量")
    # BTC 现货中间价快照：验证“情绪领先还是滞后价格”的关键原始证据，
    # 与情绪采样同时刻记录，供局内领先/滞后与背离分析使用
    btc_price: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="采样时刻 BTC 现货中间价（spot bookTicker mid）"
    )
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
    # 价格曲线：下注那一刻锁定赔率的关键数据（采样表仅保留 1 小时，归档时永久化）
    curve_up_price: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="UP token 价格时间序列 [{t, v}, ...]，v 为 0~1"
    )
    curve_down_price: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="DOWN token 价格时间序列 [{t, v}, ...]，v 为 0~1"
    )
    # 参与者/交易量时序：momentum 类假设（资金流入速度、参与者增长率）的原始证据，
    # 归档时快照永久化，避免重演价格曲线仅剩均值的教训
    curve_participants: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="参与人数时间序列 [{t, v}, ...]"
    )
    curve_trade_volume: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="交易量时间序列 [{t, v}, ...]"
    )
    # BTC 局内价格曲线：与情绪曲线同步采样的现货中间价序列，
    # 用于情绪 vs 价格的领先/滞后、背离、加速度关系分析（归档时永久化）
    curve_btc_price: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="BTC 现货中间价时间序列 [{t, v}, ...]"
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
    [DEPRECATED 2026-08: 系统B已退役，只读存档]
    情绪曲线模式记忆：由 LLM 在 Learn/Evolve 阶段自主发现与命名（Req 1.1）

    curve_features / conditions 为 LLM 自由结构 JSON，程序不做语义校验（Req 1.3）。
    win_rate 以 Harness 维护的 correct_count / sample_count 为精确来源，
    LLM 不填写 win_rate / sample_count / correct_count / status 等统计与状态字段。
    """
    __tablename__ = "pattern_memory"
    __table_args__ = (
        Index("ix_pattern_memory_name", "pattern_name"),    # Req 1.2 名称检索
        Index("ix_pattern_memory_status", "status"),        # Req 1.2 状态筛选
        Index("ix_pattern_memory_discovery_method", "discovery_method"),  # 按发现方法聚合 live 指标
        Index("ix_pattern_memory_tier", "tier"),                            # 模式池分级筛选
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
    # --- 发现方法与样本外（holdout）统计（Deep Learn 双轨 A/B 对比）---
    # discovery_method: LLM_DEEP=纯 LLM 深度发现 / PY_CLUSTER=Python 确定性聚类 /
    # LEGACY=存量或 learn/evolve 产出（迁移默认值，存量行不受影响）。
    discovery_method: Mapped[str] = mapped_column(
        String(20), nullable=False, default="LEGACY", server_default="LEGACY",
        comment="发现方法：LLM_DEEP | PY_CLUSTER | LEGACY"
    )
    # 发现时在留出集(holdout)上的样本外统计，与 live 的 win_rate/sample_count/
    # correct_count 分开存、互不污染；仅 Deep Learn 双轨发现时回填。
    holdout_win_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="发现时 holdout 胜率 0~1"
    )
    holdout_sample_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="发现时 holdout 命中判定样本数"
    )
    holdout_ci_lower: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="发现时 holdout 胜率 Wilson 95% 置信下界"
    )
    # --- 模式池分级（S/A/B/C）：由定期重回测（pattern_backtest_runs）驱动
    # 晋级/降级；存量行默认 'C'。最有潜力的模式晋级到 S 池。---
    tier: Mapped[str] = mapped_column(
        String(2), nullable=False, default="C", server_default="C",
        comment="模式池分级 S | A | B | C"
    )
    # --- 科学发现系统（scientific-discovery 宪法）---
    # predicate: LLM 假设的谓词 DSL JSON（经 predicates.validate_predicate 白名单校验，
    # 程序可确定性执行）；旧自由文本模式此列为 NULL，与谓词轨并存。
    predicate: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True, comment="谓词 DSL JSON（科学发现轨，Q5）"
    )
    # binning_version: 模式"出生"时的分箱快照版本（Q4：测量结果必须注明仪器精度）
    binning_version: Mapped[str | None] = mapped_column(
        String(40), nullable=True, comment="发现时的分箱快照版本（Q4）"
    )
    # 双轨死因（Q7-1）：SPURIOUS=假规律（从未显著）/ EXPIRED=过期规律（曾显著后衰减）；
    # 存活为 NULL。RETIRE 时由 diagnose_death 回填。
    death_cause: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="死因：SPURIOUS | EXPIRED（Q7-1）"
    )
    lifespan_days: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="存活天数（RETIRE 时回填，供存活期分布反馈）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ============================================================
# 科学发现系统 - 分箱冻结快照表（Q4：每通道独立分位边界）
# ============================================================

class BinningSnapshotModel(Base):
    """
    [DEPRECATED 2026-08: 系统B已退役，只读存档]
    分位数分箱冻结快照：每 30 天按通道各冻结一版 20/40/60/80 分位边界。

    与 services/symbolizer.BinningSnapshot 纯数据结构对齐（version/edges/
    created_at_epoch/sample_count），DB 层额外加 channel 维度——三通道量纲不同，
    共用边界会让小量纲通道全部落入"平"档（Q4 量纲漏洞修订）。
    同一 version 覆盖 sentiment/price/volume 三行。
    """
    __tablename__ = "binning_snapshots"
    __table_args__ = (
        UniqueConstraint("version", "channel", name="uq_binning_version_channel"),
        Index("ix_binning_snapshots_channel", "channel"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(
        String(40), nullable=False, comment="快照版本号（同一版本覆盖三通道各一行）"
    )
    channel: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="sentiment | price | volume"
    )
    edges: Mapped[list] = mapped_column(
        JSONB, nullable=False, comment="[q20, q40, q60, q80] 分位边界"
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="计算边界时的差值样本数"
    )
    created_at_epoch: Mapped[float] = mapped_column(
        Float, nullable=False, comment="冻结时刻 epoch 秒（与 symbolizer 对齐）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 情绪曲线自进化 Agent Loop - Agent 预测记录表（Req 3.5 / 8.1）
# ============================================================

class AgentPrediction(Base):
    """
    [DEPRECATED 2026-08: 系统B已退役，只读存档]
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
    [DEPRECATED 2026-08: 系统B已退役，只读存档]
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
    [历史部分为系统B存档；表本体继续服务新模块的 LLM 调用审计（如 SCENE_RESEARCH）]
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
    [DEPRECATED 2026-08: 系统B已退役，只读存档；健康监控仍实时运行但不依赖本表]
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


# ============================================================
# 假突破信号表（日线阻力破位，暂不下注，只记录+提醒+到期结算回读）
# ============================================================

class FakeBreakoutSignal(Base):
    """
    日线阻力假突破信号：秒级检测 BTC 盘中冲高破位瞬间落表。

    策略口径（回测验证）：5m 瞬间入场 + 15m 兑现，BTC 方向胜率 80%。
    当前阶段不下注——只记录信号、邮件推送，到期后回读 BTC 价格回填结算方向，
    用于积累 15m 市场真实赔率与可成交性数据。
    """
    __tablename__ = "fake_breakout_signals"
    __table_args__ = (
        Index("ix_fbs_signal_time", "signal_time"),
        Index("ix_fbs_status", "status"),
        Index("ix_fbs_level_side", "level", "side"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(
        String(8), nullable=False, default="daily", server_default="daily",
        comment="破位级别：1h | 4h | daily（对应前 12/48/288 个 5m 窗口 closes 极值）"
    )
    side: Mapped[str] = mapped_column(
        String(4), nullable=False, default="high", server_default="high",
        comment="破位方向：high（冲过阻力→卖跌）| low（跌破支撑→买涨）"
    )
    signal_time: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="破位检测时刻（ms，秒级检测循环触发）"
    )
    resistance: Mapped[float] = mapped_column(
        Float, nullable=False, comment="当时日线阻力位（前 288 个 5m 窗口 closes 的 max）"
    )
    btc_price: Mapped[float] = mapped_column(
        Float, nullable=False, comment="破位时刻 BTC 现货中间价（仅破位审计，不参与结算判定）"
    )
    eps: Mapped[float] = mapped_column(
        Float, nullable=False, comment="触发阈值快照（破位幅度，如 0.0005）"
    )
    down_price_5m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="信号时刻 5m 市场 DOWN token 最近采样报价"
    )
    down_price_15m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="信号时刻 15m 市场 DOWN token 最近采样报价"
    )
    up_price_5m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="信号时刻 5m 市场 UP token 最近采样报价（支撑方向目标 token）"
    )
    up_price_15m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="信号时刻 15m 市场 UP token 最近采样报价"
    )
    market_end_15m: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="当时 15m 市场 end_date（ms，即到期结算时刻）"
    )
    market_start_15m: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="信号所在 15m 市场周期起点 start_date（ms）"
    )
    cycle_open_price_15m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="15m 周期开盘价 P(S)：周期锚点结算的判定基准"
    )
    market_start_5m: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="信号所在 5m 市场周期起点 start_date（ms）"
    )
    market_end_5m: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, comment="信号所在 5m 市场周期末 end_date（ms，5m 结算死线基准）"
    )
    cycle_open_price_5m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="5m 周期开盘价 P(S5)：5m 口径判定基准"
    )
    cycle_offset_sec_15m: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="信号触发时在 15m 周期内的偏移（秒，0~900）；过滤器 A（剩余时间）输入"
    )
    break_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="破位幅度 %：信号价 vs 15m 周期开盘价（破位方向）；过滤器 B 输入"
    )
    pattern: Mapped[str | None] = mapped_column(
        String(16), nullable=True,
        comment="收盘确认场景：bull_exhaust(破阻力+光头阳→次周期DOWN) | "
                "bear_exhaust(破支撑+收阴+放量→次周期UP)；旧 A+B 时代信号为 NULL"
    )
    version: Mapped[str | None] = mapped_column(
        String(40), nullable=True,
        comment="场景参数版本（M4 影子并行）：NULL/v1=现行 ACTIVE；其余为影子版本名"
    )
    close_pos: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="信号周期收盘位置 (C-L)/(H-L)：场景①判定输入与审计（阈值 0.85）"
    )
    vol_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="信号周期量比 = 本 15m 成交量 / 前 20 根均量：场景②判定输入与审计（阈值 2.0）"
    )
    settle_deadline: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="结算回读死线（ms）= signal_time + 15min + 缓冲"
    )
    settle_btc_price: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="15m 周期末时刻 BTC 现货中间价 P(E)（到期回读回填）"
    )
    settle_btc_price_5m: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="5m 周期末时刻回读的 BTC 现货中间价 P(E5)"
    )
    settle_outcome_5m: Mapped[str | None] = mapped_column(
        String(10), nullable=True,
        comment="5m 周期涨跌方向 UP | DOWN（周期锚点口径：P(E5) < cycle_open_price_5m → DOWN）"
    )
    settle_outcome: Mapped[str | None] = mapped_column(
        String(10), nullable=True,
        comment="15m 周期涨跌方向 UP | DOWN（周期锚点口径：P(E) < cycle_open_price_15m → DOWN，与币安市场真实结算一致）"
    )
    status: Mapped[str] = mapped_column(
        String(10), nullable=False, default="PENDING", server_default="PENDING",
        comment="PENDING | SETTLED | EXPIRED（数据缺失无法结算）"
    )
    email_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false(),
        comment="信号触发邮件是否已推送"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 模式回测快照表（每个模式每次回测的完整记录，支撑无限进化与前后对比）
# ============================================================

class PatternBacktestRun(Base):
    """
    [DEPRECATED 2026-08: 系统B已退役，只读存档]
    模式单次回测快照：定期重回测调度（新数据累积阈值触发）为每个模式落一条。

    前端展示两个维度：
    - 横向对比：同一时刻不同模式的回测指标对比
    - 纵向对比：同一模式随时间的回测指标漂移（胜率/CI/EV 变化曲线）
    """
    __tablename__ = "pattern_backtest_runs"
    __table_args__ = (
        Index("ix_pbr_pattern_id", "pattern_id"),
        Index("ix_pbr_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_id: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="关联 pattern_memory.id（软关联，不加外键避免模式删除牵连历史）"
    )
    data_start: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="本次回测数据范围起点（ms）"
    )
    data_end: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="本次回测数据范围终点（ms）"
    )
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="命中样本数")
    correct_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="命中且方向正确数")
    win_rate: Mapped[float] = mapped_column(Float, nullable=False, comment="胜率 0~1")
    wilson_lower: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Wilson 95% 置信下界"
    )
    wilson_upper: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Wilson 95% 置信上界"
    )
    ev_after_fee: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="费后 EV 估算（0.5 定价口径：(1-0.02)/0.51-1 ≈ +0.9216 / -1）"
    )
    segment_stats: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="分段细节 JSON（按行情段/月段的胜率与样本数，供纵向对比）"
    )
    delta_vs_prev: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="与上一次回测的细节对比差异（胜率漂移/新增样本分段表现等）"
    )
    trigger_reason: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="触发原因：SCHEDULED | DATA_THRESHOLD | MANUAL"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ============================================================
# 场景参数版本表（场景信号系统 LLM 自进化体系，2026-08-16）
# ============================================================

class SceneParamVersion(Base):
    """
    场景参数版本：场景①②判定参数的版本化载体（LLM 自进化体系）。

    生命周期：LLM 研究员提出假设（PENDING_REVIEW）→ 科学回测裁决（硬门禁）→
    人工放行 promote（SHADOW 影子并行，只落表不发邮件）→ 实盘积累且不劣于
    ACTIVE 后人工切换（原 ACTIVE 转 RETIRED）。任意时刻最多一个 ACTIVE，
    由服务层（hypothesis_arbiter / promote API）保证。
    多重检验预算基数 = 本表累计行数（持久化，不随重启清零）。
    """
    __tablename__ = "scene_param_versions"
    __table_args__ = (
        Index("ix_spv_status", "status"),
        UniqueConstraint("version", name="uq_spv_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(
        String(40), nullable=False,
        comment="版本号，如 v1-20260816（人工可读，全局唯一）"
    )
    params: Mapped[dict] = mapped_column(
        JSONB, nullable=False,
        comment="场景参数集：{close_pos_min, vol_ratio_min, vol_ma_window, eps, level_lookbacks}"
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="PENDING_REVIEW", server_default="PENDING_REVIEW",
        comment="PENDING_REVIEW | REJECTED | SHADOW | ACTIVE | RETIRED"
    )
    backtest_report: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True,
        comment="过闸时科学回测引擎的完整输出快照（四层检验结果）"
    )
    proposed_by: Mapped[str] = mapped_column(
        String(60), nullable=False, default="human", server_default="human",
        comment="提议者：llm-researcher | human"
    )
    reviewed_by: Mapped[str | None] = mapped_column(
        String(60), nullable=True,
        comment="放行人（promote API 调用时回填）"
    )
    review_note: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="审批备注（驳回理由 / 放行理由）"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="升为 ACTIVE 的时刻（人工 promote）"
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="退为 RETIRED 的时刻（被新版本接替或人工回退）"
    )
