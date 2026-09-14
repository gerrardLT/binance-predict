"""add unified shadow execution assessment ledger

Revision ID: b0c1d2e3f4a5
Revises: a9b0c1d2e3f4
Create Date: 2026-09-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b0c1d2e3f4a5"
down_revision = "a9b0c1d2e3f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_execution_assessments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_type", sa.String(length=24), nullable=False),
        sa.Column("source_id", sa.BigInteger(), nullable=True),
        sa.Column("signal_version", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("config_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("source_window_start", sa.BigInteger(), nullable=True),
        sa.Column("target_window_start", sa.BigInteger(), nullable=False),
        sa.Column("market_period", sa.String(length=8), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("trigger_ts", sa.BigInteger(), nullable=True),
        sa.Column("evaluation_ts", sa.BigInteger(), nullable=True),
        sa.Column("strategy_eligible", sa.Boolean(), nullable=True),
        sa.Column("operational_eligible", sa.Boolean(), nullable=True),
        sa.Column("execution_eligible", sa.Boolean(), nullable=True),
        sa.Column("terminal_stage", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=48), nullable=False),
        sa.Column("legacy_status", sa.String(length=20), nullable=True),
        sa.Column("live_channel", sa.String(length=64), nullable=True),
        sa.Column("channel_mapped", sa.Boolean(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("retired", sa.Boolean(), nullable=True),
        sa.Column("order_type", sa.String(length=12), nullable=True),
        sa.Column("amount_usdt", sa.Float(), nullable=True),
        sa.Column("max_daily_orders", sa.Integer(), nullable=True),
        sa.Column("effective_max_exec_price", sa.Float(), nullable=True),
        sa.Column("guard_applied", sa.Boolean(), nullable=True),
        sa.Column("exclusive_blocker_channel", sa.String(length=64), nullable=True),
        sa.Column("market_id", sa.BigInteger(), nullable=True),
        sa.Column("token_id", sa.String(length=128), nullable=True),
        sa.Column("balance_available", sa.Float(), nullable=True),
        sa.Column("balance_required", sa.Float(), nullable=True),
        sa.Column("quote_average_price", sa.Float(), nullable=True),
        sa.Column("quote_ts", sa.BigInteger(), nullable=True),
        sa.Column("source_status", sa.String(length=20), nullable=True),
        sa.Column("theoretical_win", sa.Boolean(), nullable=True),
        sa.Column("theoretical_outcome", sa.String(length=16), nullable=True),
        sa.Column("theoretical_entry_price", sa.Float(), nullable=True),
        sa.Column("theoretical_realized_return", sa.Float(), nullable=True),
        sa.Column("theoretical_settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "config_snapshot",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "input_snapshot",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "decision_snapshot",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "quote_snapshot",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "signal_version",
            "target_window_start",
            "market_period",
            "policy_version",
            name="uq_shadow_exec_event_policy",
        ),
    )
    op.create_index(
        "uq_shadow_exec_source_policy",
        "shadow_execution_assessments",
        ["source_type", "source_id", "policy_version"],
        unique=True,
        postgresql_where=sa.text("source_id IS NOT NULL"),
    )
    op.create_index(
        "ix_shadow_exec_version_window",
        "shadow_execution_assessments",
        ["signal_version", "target_window_start"],
    )
    op.create_index(
        "ix_shadow_exec_stage_reason",
        "shadow_execution_assessments",
        ["terminal_stage", "reason_code"],
    )
    op.create_index(
        "ix_shadow_exec_policy_period",
        "shadow_execution_assessments",
        ["policy_version", "market_period"],
    )
    op.add_column(
        "trade_orders",
        sa.Column("assessment_id", sa.BigInteger(), nullable=True),
    )
    op.create_foreign_key(
        "fk_trade_orders_assessment_id",
        "trade_orders",
        "shadow_execution_assessments",
        ["assessment_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_trade_orders_assessment_id",
        "trade_orders",
        ["assessment_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_trade_orders_assessment_id", table_name="trade_orders")
    op.drop_constraint(
        "fk_trade_orders_assessment_id",
        "trade_orders",
        type_="foreignkey",
    )
    op.drop_column("trade_orders", "assessment_id")
    op.drop_index("ix_shadow_exec_policy_period", table_name="shadow_execution_assessments")
    op.drop_index("ix_shadow_exec_stage_reason", table_name="shadow_execution_assessments")
    op.drop_index("ix_shadow_exec_version_window", table_name="shadow_execution_assessments")
    op.drop_index("uq_shadow_exec_source_policy", table_name="shadow_execution_assessments")
    op.drop_table("shadow_execution_assessments")
