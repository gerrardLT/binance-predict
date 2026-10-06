"""add signal_health_snapshots (信号体检日快照)

Revision ID: g5b6c7d8e9f0
Revises: f4a5b6c7d8e9
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "g5b6c7d8e9f0"
down_revision = "f4a5b6c7d8e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signal_health_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("key", sa.String(96), nullable=False),
        sa.Column("snap_date", sa.String(10), nullable=False),
        sa.Column("light", sa.String(8), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reasons", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("scope", "key", "snap_date", name="uq_signal_health_scope_key_date"),
    )
    op.create_index("ix_signal_health_scope_key", "signal_health_snapshots", ["scope", "key"])


def downgrade() -> None:
    op.drop_index("ix_signal_health_scope_key", table_name="signal_health_snapshots")
    op.drop_table("signal_health_snapshots")
