"""add firsthit research ledger tables (Phase 1)

Revision ID: z1a2b3c4d5e6
Revises: u9v0w1x2y3z4
Create Date: 2026-09-08

研究规范：docs/superpowers/specs/2026-09-08-g0-g1-g3-scientific-optimization-design.md §12
只读研究表（全窗口审计层 + G0 母事件层），不被下单链路引用。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "z1a2b3c4d5e6"
down_revision = "u9v0w1x2y3z4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "firsthit_window_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("audit_version", sa.String(32), nullable=False),
        sa.Column("expected_samples", sa.Integer(), nullable=False),
        sa.Column("actual_samples", sa.Integer(), nullable=False),
        sa.Column("down_pts", sa.Integer(), nullable=False),
        sa.Column("up_pts", sa.Integer(), nullable=False),
        sa.Column("btc_pts", sa.Integer(), nullable=False),
        sa.Column("max_gap_ms", sa.Integer(), nullable=True),
        sa.Column("raw_firsthit_detected", sa.Boolean(), nullable=False),
        sa.Column("firsthit_detected", sa.Boolean(), nullable=False),
        sa.Column("exclusion_reason", sa.String(32), nullable=True),
        sa.Column("outcome", sa.String(10), nullable=True),
        sa.Column("settled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("window_start", name="uq_fh_audit_window"),
    )
    op.create_index("ix_fh_audit_window_start", "firsthit_window_audit", ["window_start"])
    op.create_table(
        "firsthit_mother_event",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column("window_start", sa.BigInteger(), nullable=False),
        sa.Column("window_end", sa.BigInteger(), nullable=False),
        sa.Column("trigger_ts", sa.BigInteger(), nullable=False),
        sa.Column("feature_version", sa.String(32), nullable=False),
        sa.Column("q", sa.Float(), nullable=False),
        sa.Column("q_prev", sa.Float(), nullable=True),
        sa.Column("dt_prev_ms", sa.Integer(), nullable=True),
        sa.Column("up_price_at_trigger", sa.Float(), nullable=True),
        sa.Column("sum_gap", sa.Float(), nullable=True),
        sa.Column("btc_open", sa.Float(), nullable=False),
        sa.Column("btc_trigger", sa.Float(), nullable=False),
        sa.Column("path_hi", sa.Float(), nullable=False),
        sa.Column("path_lo", sa.Float(), nullable=False),
        sa.Column("chg_bps", sa.Float(), nullable=True),
        sa.Column("body_r", sa.Float(), nullable=True),
        sa.Column("wick01", sa.Float(), nullable=True),
        sa.Column("upper_wick_bps", sa.Float(), nullable=True),
        sa.Column("rng_bps", sa.Float(), nullable=True),
        sa.Column("npts", sa.Integer(), nullable=False),
        sa.Column("td_sec", sa.Integer(), nullable=False),
        sa.Column("dvol", sa.Float(), nullable=True),
        sa.Column("dpar", sa.Float(), nullable=True),
        sa.Column("dvol_missing", sa.Boolean(), nullable=False),
        sa.Column("dpar_missing", sa.Boolean(), nullable=False),
        sa.Column("labels", postgresql.JSONB(), nullable=False),
        sa.Column("stratum", sa.String(12), nullable=False),
        sa.Column("settle_outcome", sa.String(10), nullable=True),
        sa.Column("win", sa.Boolean(), nullable=True),
        sa.Column("intention_ev", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("window_start", name="uq_fh_mother_window"),
    )
    op.create_index("ix_fh_mother_window_start", "firsthit_mother_event", ["window_start"])
    op.create_index("ix_fh_mother_stratum", "firsthit_mother_event", ["stratum"])


def downgrade() -> None:
    op.drop_index("ix_fh_mother_stratum", table_name="firsthit_mother_event")
    op.drop_index("ix_fh_mother_window_start", table_name="firsthit_mother_event")
    op.drop_table("firsthit_mother_event")
    op.drop_index("ix_fh_audit_window_start", table_name="firsthit_window_audit")
    op.drop_table("firsthit_window_audit")
