"""widen shadow version identifiers for tagged candlestick events

Revision ID: c1d2e3f4a5b6
Revises: b0c1d2e3f4a5
"""
from alembic import op
import sqlalchemy as sa

revision = "c1d2e3f4a5b6"
down_revision = "b0c1d2e3f4a5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "kline_shadow_signals", "version",
        existing_type=sa.String(length=24), type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "kline_shadow_signals", "discovery_id",
        existing_type=sa.String(length=16), type_=sa.String(length=32),
        existing_nullable=False,
    )
    op.alter_column(
        "shadow_version_overrides", "version",
        existing_type=sa.String(length=24), type_=sa.String(length=80),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "shadow_version_overrides", "version",
        existing_type=sa.String(length=80), type_=sa.String(length=24),
        existing_nullable=False,
    )
    op.alter_column(
        "kline_shadow_signals", "discovery_id",
        existing_type=sa.String(length=32), type_=sa.String(length=16),
        existing_nullable=False,
    )
    op.alter_column(
        "kline_shadow_signals", "version",
        existing_type=sa.String(length=64), type_=sa.String(length=24),
        existing_nullable=False,
    )
