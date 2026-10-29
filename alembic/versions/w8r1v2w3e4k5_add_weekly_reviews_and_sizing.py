"""add weekly reviews + dynamic sizing config

Revision ID: w8r1v2w3e4k5
Revises: s7a8b9c0d1e2
Create Date: 2026-10-30

2026-10-30「LLM 每周复盘 + 动态仓位（%）」：
1. live_channel_overrides 新增 amount_pct / amount_cap_usdt（NULL=固定金额，零行为变化）；
2. live_sizing_config 单行表：仓位分母（手动固定基准本金，不随权益浮动）；
3. weekly_reviews：组合级周报事实卡 + LLM 报告 + 应用日志（只读分析层，
   不被下单/结算路径引用）。

回滚：三处均可独立撤销；两张新表不承载交易状态，drop 不影响订单与结算。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "w8r1v2w3e4k5"
down_revision = "s7a8b9c0d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "live_channel_overrides",
        sa.Column("amount_pct", sa.Float(), nullable=True,
                  comment="动态仓位百分比（NULL=用固定 amount_usdt）"),
    )
    op.add_column(
        "live_channel_overrides",
        sa.Column("amount_cap_usdt", sa.Float(), nullable=True,
                  comment="% 模式下的单笔金额上限（NULL=全局默认）"),
    )
    op.create_table(
        "live_sizing_config",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("base_equity_usdt", sa.Float(), nullable=False,
                  server_default=sa.text("200.0"),
                  comment="仓位分母（$10~$5000，手动设定）"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "weekly_reviews",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("week_key", sa.String(16), nullable=False, comment="ISO 周，如 2026-W44"),
        sa.Column("period_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("period_end_ms", sa.BigInteger(), nullable=False),
        sa.Column("trigger_type", sa.String(12), nullable=False, comment="WEEKLY|MANUAL"),
        sa.Column("status", sa.String(12), nullable=False, server_default="RUNNING"),
        sa.Column("verdict", sa.String(24), nullable=True,
                  comment="OK|INSUFFICIENT_DATA|LLM_ONLY_FAILED"),
        sa.Column("facts", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("report", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("rejected", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("models", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("llm_error", sa.Text(), nullable=True),
        sa.Column("application_log", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_weekly_reviews_week_key", "weekly_reviews", ["week_key"])
    op.create_index("ix_weekly_reviews_created", "weekly_reviews", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_weekly_reviews_created", table_name="weekly_reviews")
    op.drop_index("ix_weekly_reviews_week_key", table_name="weekly_reviews")
    op.drop_table("weekly_reviews")
    op.drop_table("live_sizing_config")
    op.drop_column("live_channel_overrides", "amount_cap_usdt")
    op.drop_column("live_channel_overrides", "amount_pct")
