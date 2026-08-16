"""add fake_breakout_signals.version (M4 shadow parallel)

Revision ID: p4g7h8i9j0k1
Revises: o5f6g7h8i9j0
Create Date: 2026-08-16

M4 影子并行：信号表加 version 列标记判定所用参数版本。
NULL/v1 = 现行 ACTIVE 参数；其余值 = 影子版本名（只落表不发邮件，
供实盘对照 SHADOW 与 ACTIVE 的表现差异，作为人工 promote 的依据）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "p4g7h8i9j0k1"
down_revision = "o5f6g7h8i9j0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "version", sa.String(length=40), nullable=True,
            comment="场景参数版本（M4 影子并行）：NULL/v1=现行 ACTIVE；其余为影子版本名",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "version")
