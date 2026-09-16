"""widen live channel identifiers for candlestick logical versions

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
"""
from alembic import op
import sqlalchemy as sa

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("trade_orders", "signal_version", 40),
    ("live_channel_overrides", "channel", 64),
    ("notification_channel_overrides", "channel", 64),
    ("shadow_execution_assessments", "signal_version", 64),
    ("shadow_execution_assessments", "live_channel", 64),
    ("shadow_execution_assessments", "exclusive_blocker_channel", 64),
)

def upgrade() -> None:
    for table, column, old_length in _COLUMNS:
        op.alter_column(
            table, column,
            existing_type=sa.String(length=old_length),
            type_=sa.String(length=80),
            existing_nullable=column not in {"channel", "signal_version"} or table == "trade_orders",
        )

def downgrade() -> None:
    for table, column, old_length in reversed(_COLUMNS):
        op.alter_column(
            table, column,
            existing_type=sa.String(length=80),
            type_=sa.String(length=old_length),
            existing_nullable=column not in {"channel", "signal_version"} or table == "trade_orders",
        )
