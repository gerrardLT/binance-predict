"""add K10 audit features to firsthit shadow signals

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
"""
from alembic import op
import sqlalchemy as sa

revision = "f4a5b6c7d8e9"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("firsthit_shadow_signals", sa.Column("k10_z", sa.Float(), nullable=True))
    op.add_column("firsthit_shadow_signals", sa.Column("k10_remaining_seconds", sa.Float(), nullable=True))
    op.add_column("firsthit_shadow_signals", sa.Column("k10_remaining_sigma_bps", sa.Float(), nullable=True))
    op.add_column("firsthit_shadow_signals", sa.Column("k10_energy_expands", sa.Boolean(), nullable=True))
    op.add_column("firsthit_shadow_signals", sa.Column("k10_upper_wick_ratio", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("firsthit_shadow_signals", "k10_upper_wick_ratio")
    op.drop_column("firsthit_shadow_signals", "k10_energy_expands")
    op.drop_column("firsthit_shadow_signals", "k10_remaining_sigma_bps")
    op.drop_column("firsthit_shadow_signals", "k10_remaining_seconds")
    op.drop_column("firsthit_shadow_signals", "k10_z")
