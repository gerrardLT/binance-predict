"""add sentiment agent loop tables and trade_orders agent_prediction_id

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-07-09 13:45:48.507549

本迁移为「情绪曲线自进化 Agent Loop」新增三张表，并为既有 trade_orders
表追加与 Agent 预测关联的列/外键。手写脚本，字段/类型/nullable/server_default/
comment 均严格对齐 src/binance_predict/db/models.py（单一事实源，用户规则 7/8）。

关于 trade_orders ↔ agent_predictions 的循环外键：
- agent_predictions.trade_order_id  → trade_orders.id      （trade_orders 为既有表）
- trade_orders.agent_prediction_id  → agent_predictions.id （本迁移新增列）
升级时先建 agent_predictions（内联指向已存在的 trade_orders），最后再用
ALTER TABLE 为 trade_orders 追加外键；降级时先按名 DROP 该外键，解开环后逆序删表。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：建基础表（trade_orders / prediction_market_samples / sentiment_windows）
    + Agent Loop 三张表（pattern_memory / agent_predictions / pattern_change_log），
    并为 trade_orders 追加 agent_prediction_id 列与循环外键。"""

    # ------------------------------------------------------------------
    # 0a) 交易订单表 trade_orders
    # ------------------------------------------------------------------
    op.create_table(
        "trade_orders",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=True, comment="关联的预测 ID（旧 K 线决策路径，退役后不再写入）"),
        sa.Column("market_id", sa.BigInteger(), nullable=True, comment="Binance 预测市场 ID"),
        sa.Column("token_id", sa.String(length=50), nullable=True, comment="Outcome Token ID"),
        sa.Column("side", sa.String(length=10), nullable=False, comment="BUY | SELL"),
        sa.Column("amount_in", sa.String(length=50), nullable=False, comment="输入金额（wei 格式）"),
        sa.Column("amount_out", sa.String(length=50), nullable=True, comment="输出金额（wei 格式）"),
        sa.Column("order_id", sa.String(length=50), nullable=True, comment="Binance 返回的订单 ID"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="PENDING", comment="PENDING | FILLED | FAILED"),
        sa.Column("quote_json", postgresql.JSONB(), nullable=True, comment="报价响应 JSON"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="错误信息（仅 FAILED 时）"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_trade_orders_prediction_id", "trade_orders", ["prediction_id"], unique=False)

    # ------------------------------------------------------------------
    # 0b) 预测市场情绪采样表 prediction_market_samples
    # ------------------------------------------------------------------
    op.create_table(
        "prediction_market_samples",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.BigInteger(), nullable=False, comment="毫秒时间戳"),
        sa.Column("up_price", sa.Float(), nullable=True, comment="UP token 价格"),
        sa.Column("down_price", sa.Float(), nullable=True, comment="DOWN token 价格"),
        sa.Column("up_pct", sa.Float(), nullable=True, comment="UP 百分比"),
        sa.Column("down_pct", sa.Float(), nullable=True, comment="DOWN 百分比"),
        sa.Column("participants", sa.Integer(), nullable=True, comment="参与人数"),
        sa.Column("trade_volume", sa.Float(), nullable=True, comment="交易量"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pm_samples_timestamp", "prediction_market_samples", ["timestamp"], unique=False)

    # ------------------------------------------------------------------
    # 0c) 情绪窗口表 sentiment_windows
    # ------------------------------------------------------------------
    op.create_table(
        "sentiment_windows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("start_time", sa.BigInteger(), nullable=False, comment="窗口开始时间戳（ms，5 分钟整点）"),
        sa.Column("end_time", sa.BigInteger(), nullable=False, comment="窗口结束时间戳（ms）"),
        sa.Column("curve_up_pct", postgresql.JSONB(), nullable=True, comment="UP% 时间序列 [{t, v}, ...]"),
        sa.Column("curve_down_pct", postgresql.JSONB(), nullable=True, comment="DOWN% 时间序列 [{t, v}, ...]"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0", comment="窗口内采样点数"),
        sa.Column("entry_price", sa.Float(), nullable=True, comment="窗口开始时 BTC 价格"),
        sa.Column("exit_price", sa.Float(), nullable=True, comment="窗口结束时 BTC 价格"),
        sa.Column("actual_return", sa.Float(), nullable=True, comment="实际收益率 (exit/entry - 1)"),
        sa.Column("outcome", sa.String(length=10), nullable=True, comment="实际结果: UP / DOWN / NOISE"),
        sa.Column("avg_participants", sa.Float(), nullable=True, comment="窗口内平均参与人数"),
        sa.Column("avg_trade_volume", sa.Float(), nullable=True, comment="窗口内平均交易量"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("start_time", "end_time", name="uq_sw_start_end"),
    )
    op.create_index("ix_sw_start_time", "sentiment_windows", ["start_time"], unique=False)
    op.create_index("ix_sw_outcome", "sentiment_windows", ["outcome"], unique=False)

    # ------------------------------------------------------------------
    # 1) 模式记忆表 pattern_memory（Req 1.1/1.2/1.3）
    #    curve_features / conditions 为 LLM 自由结构 JSONB，程序不做语义校验。
    #    win_rate/sample_count/correct_count/confidence_score/status 在 ORM 侧
    #    有 Python 端 default（非 DDL server_default），故此处不设 server_default。
    # ------------------------------------------------------------------
    op.create_table(
        "pattern_memory",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pattern_name", sa.String(length=120), nullable=False, comment="LLM 自主命名"),
        sa.Column("description", sa.Text(), nullable=False, comment="模式描述，LLM 自由填写"),
        sa.Column(
            "curve_features",
            postgresql.JSONB(),
            nullable=False,
            comment="曲线特征，LLM 自由结构（程序不做语义校验，Req 1.3）",
        ),
        sa.Column("conditions", postgresql.JSONB(), nullable=False, comment="适用条件，LLM 自由结构"),
        sa.Column("predicted_direction", sa.String(length=10), nullable=False, comment="UP | DOWN"),
        sa.Column("win_rate", sa.Float(), nullable=False, comment="历史胜率 0~1"),
        sa.Column("sample_count", sa.Integer(), nullable=False, comment="样本数"),
        sa.Column(
            "correct_count",
            sa.Integer(),
            nullable=False,
            comment="命中数（Harness 维护，win_rate=correct_count/sample_count 的精确来源）",
        ),
        sa.Column("confidence_score", sa.Float(), nullable=False, comment="置信度 0~1"),
        sa.Column("status", sa.String(length=10), nullable=False, comment="ACTIVE | RETIRED | EVOLVING"),
        # created_at/updated_at 对应 Mapped[datetime]（非 Optional）→ NOT NULL + server_default now()
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # 名称检索（Req 1.2）与状态筛选（Req 1.2）索引
    op.create_index("ix_pattern_memory_name", "pattern_memory", ["pattern_name"], unique=False)
    op.create_index("ix_pattern_memory_status", "pattern_memory", ["status"], unique=False)

    # ------------------------------------------------------------------
    # 2) Agent 预测记录表 agent_predictions（Req 3.5/8.1/10.3）
    #    内联外键：sentiment_window_id→sentiment_windows.id、
    #             matched_pattern_id→pattern_memory.id、trade_order_id→trade_orders.id
    #    entry_timing 在 ORM 侧有 Python 端 default="SKIP"，非 DDL server_default。
    # ------------------------------------------------------------------
    op.create_table(
        "agent_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "sentiment_window_id",
            sa.Integer(),
            nullable=True,
            comment="关联的情绪窗口（预测时窗口尚未归档可为空，Validate 时回填/匹配）",
        ),
        sa.Column("predicted_direction", sa.String(length=10), nullable=False, comment="UP | DOWN | NO_TRADE"),
        sa.Column("matched_pattern_id", sa.Integer(), nullable=True, comment="匹配的模式；无匹配/冷启动为空"),
        sa.Column("matched_pattern_name", sa.String(length=120), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, comment="置信度 0~1"),
        sa.Column("entry_timing", sa.String(length=10), nullable=False, comment="NOW | WAIT | SKIP"),
        sa.Column("reasoning", sa.Text(), nullable=False, comment="LLM 推理过程"),
        # --- 验证结果（Validate 阶段回填，Req 4.3）---
        sa.Column("is_correct", sa.Boolean(), nullable=True, comment="未验证为 NULL"),
        sa.Column("actual_outcome", sa.String(length=10), nullable=True, comment="UP | DOWN | NOISE"),
        sa.Column("actual_return", sa.Float(), nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        # --- 交易关联（Req 10.3）---
        sa.Column("trade_order_id", sa.Integer(), nullable=True, comment="关联交易订单；未交易为空"),
        sa.Column("skip_trade_reason", sa.String(length=200), nullable=True, comment="跳过交易的原因（Req 10.2，非静默）"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["sentiment_window_id"], ["sentiment_windows.id"]),
        sa.ForeignKeyConstraint(["matched_pattern_id"], ["pattern_memory.id"]),
        sa.ForeignKeyConstraint(["trade_order_id"], ["trade_orders.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    # 时间筛选 / 方向筛选（Req 8.3）与 Validate 关联索引
    op.create_index("ix_agent_pred_time", "agent_predictions", ["prediction_time"], unique=False)
    op.create_index("ix_agent_pred_direction", "agent_predictions", ["predicted_direction"], unique=False)
    op.create_index("ix_agent_pred_window", "agent_predictions", ["sentiment_window_id"], unique=False)

    # ------------------------------------------------------------------
    # 3) 模式变更日志表 pattern_change_log（Req 1.4/8.2）
    #    外键：pattern_id→pattern_memory.id
    # ------------------------------------------------------------------
    op.create_table(
        "pattern_change_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pattern_id", sa.Integer(), nullable=False),
        sa.Column("change_type", sa.String(length=10), nullable=False, comment="CREATE | UPDATE | RETIRE"),
        sa.Column("phase", sa.String(length=10), nullable=False, comment="触发阶段 LEARN | EVOLVE"),
        sa.Column("before_snapshot", postgresql.JSONB(), nullable=True, comment="变更前完整快照；CREATE 为 NULL"),
        sa.Column(
            "after_snapshot",
            postgresql.JSONB(),
            nullable=True,
            comment="变更后完整快照；RETIRE 时为置为 RETIRED 后的快照",
        ),
        sa.Column("change_reason", sa.Text(), nullable=False, comment="变更原因，LLM 提供"),
        sa.Column(
            "evolve_phase_id",
            sa.String(length=40),
            nullable=True,
            comment="触发该变更的 Evolve 执行 ID（LEARN 触发时为 NULL，Req 8.2）",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["pattern_id"], ["pattern_memory.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pcl_pattern_id", "pattern_change_log", ["pattern_id"], unique=False)
    # 时间正序（Req 8.5）索引
    op.create_index("ix_pcl_created_at", "pattern_change_log", ["created_at"], unique=False)

    # ------------------------------------------------------------------
    # 4) trade_orders 新增列 agent_prediction_id（Req 10.3）
    #    与旧 prediction_id 并存、互不干扰；可空。
    # 5) 追加循环外键（此时 agent_predictions 已建好，可安全引用）
    # ------------------------------------------------------------------
    op.add_column(
        "trade_orders",
        sa.Column(
            "agent_prediction_id",
            sa.Integer(),
            nullable=True,
            comment="关联的 Agent 预测 ID（新增，与 prediction_id 互斥使用）",
        ),
    )
    op.create_foreign_key(
        "fk_trade_orders_agent_prediction_id",
        "trade_orders",
        "agent_predictions",
        ["agent_prediction_id"],
        ["id"],
    )


def downgrade() -> None:
    """降级：逆序回滚。先解开 trade_orders→agent_predictions 循环外键，
    再删列，最后按依赖逆序删除所有表（连同各自索引）。"""

    # 1) 先按名 DROP 循环外键，解开 trade_orders ↔ agent_predictions 的环
    op.drop_constraint("fk_trade_orders_agent_prediction_id", "trade_orders", type_="foreignkey")
    # 2) 删除 trade_orders 新增列
    op.drop_column("trade_orders", "agent_prediction_id")

    # 3) pattern_change_log（+ 索引）
    op.drop_index("ix_pcl_created_at", table_name="pattern_change_log")
    op.drop_index("ix_pcl_pattern_id", table_name="pattern_change_log")
    op.drop_table("pattern_change_log")

    # 4) agent_predictions（+ 索引）
    op.drop_index("ix_agent_pred_window", table_name="agent_predictions")
    op.drop_index("ix_agent_pred_direction", table_name="agent_predictions")
    op.drop_index("ix_agent_pred_time", table_name="agent_predictions")
    op.drop_table("agent_predictions")

    # 5) pattern_memory（+ 索引）
    op.drop_index("ix_pattern_memory_status", table_name="pattern_memory")
    op.drop_index("ix_pattern_memory_name", table_name="pattern_memory")
    op.drop_table("pattern_memory")

    # 6) 基础表（逆序删除，先删有外键引用的表）
    op.drop_index("ix_sw_outcome", table_name="sentiment_windows")
    op.drop_index("ix_sw_start_time", table_name="sentiment_windows")
    op.drop_table("sentiment_windows")

    op.drop_index("ix_pm_samples_timestamp", table_name="prediction_market_samples")
    op.drop_table("prediction_market_samples")

    op.drop_index("ix_trade_orders_prediction_id", table_name="trade_orders")
    op.drop_table("trade_orders")
