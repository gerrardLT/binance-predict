"""add LLM attribution system tables (order context + attribution runs/analyses/hypotheses)

Revision ID: c4d5e6f7a8b9
Revises: i7d8e9f0a1b2
Create Date: 2026-10-19

LLM 归因闭环（2026-10-19 方案）：确定性统计先行裁决，LLM 只做假设生成器，
随机性是一等假设。四张表均为事后分析层，不被下单/结算决策路径引用。
旧系统B（agent_predictions/pattern_memory 等 deprecated 表）按用户拍板保留数据不删。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "c4d5e6f7a8b9"
down_revision = "i7d8e9f0a1b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "order_context_packs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("order_id", sa.Integer(),
                  sa.ForeignKey("trade_orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", sa.String(80), nullable=False),
        sa.Column("market_period", sa.String(8), nullable=False),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("direction", sa.String(8), nullable=True),
        sa.Column("schema_version", sa.String(16), nullable=False),
        sa.Column("status", sa.String(8), nullable=False),
        sa.Column("missing", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("pack", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("facts", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("order_id", name="uq_order_context_packs_order"),
    )
    op.create_index("ix_order_context_packs_channel_window", "order_context_packs",
                    ["channel", "window_start"])

    op.create_table(
        "attribution_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("trigger_type", sa.String(16), nullable=False),
        sa.Column("channel", sa.String(80), nullable=False),
        sa.Column("scope", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("facts", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("verdict", sa.String(24), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_attribution_runs_channel_created", "attribution_runs",
                    ["channel", "created_at"])
    op.create_index("ix_attribution_runs_trigger", "attribution_runs", ["trigger_type"])

    op.create_table(
        "attribution_analyses",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("run_id", sa.BigInteger(),
                  sa.ForeignKey("attribution_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("family", sa.String(24), nullable=False),
        sa.Column("raw_output", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("hypotheses", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("rejected_claims", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_attribution_analyses_run", "attribution_analyses", ["run_id"])

    op.create_table(
        "hypotheses",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("channel", sa.String(80), nullable=True),
        sa.Column("category", sa.String(24), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("test_spec", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("last_test", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("first_run_id", sa.BigInteger(), nullable=True),
        sa.Column("support_count", sa.Integer(), nullable=False, default=0),
        sa.Column("reject_count", sa.Integer(), nullable=False, default=0),
        sa.Column("in_error_memory", sa.Boolean(), nullable=False, default=False),
        sa.Column("action_suggestion", sa.Text(), nullable=True),
        sa.Column("action_status", sa.String(10), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_hypotheses_channel_status", "hypotheses", ["channel", "status"])
    op.create_index("ix_hypotheses_status", "hypotheses", ["status"])


def downgrade() -> None:
    op.drop_index("ix_hypotheses_status", table_name="hypotheses")
    op.drop_index("ix_hypotheses_channel_status", table_name="hypotheses")
    op.drop_table("hypotheses")
    op.drop_index("ix_attribution_analyses_run", table_name="attribution_analyses")
    op.drop_table("attribution_analyses")
    op.drop_index("ix_attribution_runs_trigger", table_name="attribution_runs")
    op.drop_index("ix_attribution_runs_channel_created", table_name="attribution_runs")
    op.drop_table("attribution_runs")
    op.drop_index("ix_order_context_packs_channel_window", table_name="order_context_packs")
    op.drop_table("order_context_packs")
