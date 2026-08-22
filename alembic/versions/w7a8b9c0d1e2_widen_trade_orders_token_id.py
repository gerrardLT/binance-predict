"""widen trade_orders.token_id for prediction outcome token (78-char hex)

Revision ID: w7a8b9c0d1e2
Revises: u6f7g8h9i0j1
Create Date: 2026-08-23

预测市场 outcome tokenId 为 78 字符 hex，原 VARCHAR(50) 写入超长触发
StringDataRightTruncationError → 成交订单 UPDATE 失败、行卡 PENDING
（钱已花出但本地无记录）。扩到 VARCHAR(128) 修复。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "w7a8b9c0d1e2"
down_revision = "u6f7g8h9i0j1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "trade_orders", "token_id",
        existing_type=sa.String(length=50),
        type_=sa.String(length=128),
        existing_nullable=True,
        comment="Outcome Token ID（预测市场 78 字符 hex）",
    )


def downgrade() -> None:
    op.alter_column(
        "trade_orders", "token_id",
        existing_type=sa.String(length=128),
        type_=sa.String(length=50),
        existing_nullable=True,
    )
