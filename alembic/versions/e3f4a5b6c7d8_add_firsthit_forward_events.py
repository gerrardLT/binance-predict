"""add frozen firsthit forward record-only events

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7

5m q<=0.10 首触母事件的 DOWN recovery 主候选、q-only 对照与 UP 镜像。
研究表只记录不下注，不被交易链路引用。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "firsthit_forward_events",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("schema_version", sa.String(48), nullable=False),
        sa.Column("capture_mode", sa.String(20), nullable=False),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("trigger_ts", sa.BigInteger(), nullable=False),
        sa.Column("td_sec", sa.Integer(), nullable=False),
        sa.Column("trigger_q", sa.Float(), nullable=False),
        sa.Column("opposite_q", sa.Float(), nullable=True),
        sa.Column("quote_sum", sa.Float(), nullable=True),
        sa.Column("btc_open", sa.Float(), nullable=True),
        sa.Column("btc_adverse_extreme", sa.Float(), nullable=True),
        sa.Column("btc_trigger", sa.Float(), nullable=True),
        sa.Column("btc_signed_return_bps", sa.Float(), nullable=True),
        sa.Column("btc_signed_min_bps", sa.Float(), nullable=True),
        sa.Column("btc_recovery_bps", sa.Float(), nullable=True),
        sa.Column("btc_recovery_frac", sa.Float(), nullable=True),
        sa.Column("recovery_pass", sa.Boolean(), nullable=False),
        sa.Column("atr_bps", sa.Float(), nullable=True),
        sa.Column("current_move_atr", sa.Float(), nullable=True),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("trigger_features", postgresql.JSONB(), nullable=False),
        sa.Column("future_quotes", postgresql.JSONB(), nullable=False),
        sa.Column("execution_quotes", postgresql.JSONB(), nullable=False),
        sa.Column("post_features", postgresql.JSONB(), nullable=False),
        sa.Column("retrospective", postgresql.JSONB(), nullable=False),
        sa.Column("data_missing", postgresql.JSONB(), nullable=False),
        sa.Column("settle_outcome", sa.String(10), nullable=True),
        sa.Column("win", sa.Boolean(), nullable=True),
        sa.Column("ev_at_entry", sa.Float(), nullable=True),
        sa.Column("status", sa.String(12), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("side", "window_start", name="uq_fh_forward_side_window"),
    )
    op.create_index("ix_fh_forward_window_start", "firsthit_forward_events", ["window_start"])
    op.create_index("ix_fh_forward_status", "firsthit_forward_events", ["status"])


def downgrade() -> None:
    op.drop_index("ix_fh_forward_status", table_name="firsthit_forward_events")
    op.drop_index("ix_fh_forward_window_start", table_name="firsthit_forward_events")
    op.drop_table("firsthit_forward_events")
