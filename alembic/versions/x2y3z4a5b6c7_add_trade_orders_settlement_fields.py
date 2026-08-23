"""add trade_orders settlement fields (direction + win/pnl settle loop)

Revision ID: x2y3z4a5b6c7
Revises: w7a8b9c0d1e2
Create Date: 2026-08-23

结算闭环（P0-2）：direction 从未落库（旧链路 prediction 只用于选 token），
判赢无从谈起。新增 direction（占位时写入）+ 结算五字段
（settle_outcome/win/settle_price/pnl/settled_at），status comment 同步扩充。

settled_at IS NULL 是 TradeSettler 扫描锚点；部分索引只覆盖待结算行
（status='FILLED' AND settled_at IS NULL AND window_start IS NOT NULL），
空转扫描亚毫秒。

comment 与 models.py 严格对齐（单一事实源，避免 autogenerate 漂移）。
downgrade 直接删列（结算数据可由 TradeSettler 从 SentimentWindow 重算）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "x2y3z4a5b6c7"
down_revision = "w7a8b9c0d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("trade_orders") as batch:
        batch.add_column(sa.Column(
            "direction", sa.String(length=8), nullable=True,
            comment="下单方向 UP/DOWN（占位时写入；NULL=旧数据）",
        ))
        batch.add_column(sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="窗口结算结果 UP/DOWN/NOISE/EXPIRED（本地 SentimentWindow 口径）",
        ))
        batch.add_column(sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="是否赢（settle_outcome==direction；NOISE/EXPIRED 为 NULL）",
        ))
        batch.add_column(sa.Column(
            "settle_price", sa.Float(), nullable=True,
            comment="结算参考价（窗口 exit_price，本地口径）",
        ))
        batch.add_column(sa.Column(
            "pnl", sa.Float(), nullable=True,
            comment="结算盈亏 USDT（本地估算：赢=amount/均价-amount，输=-amount）",
        ))
        batch.add_column(sa.Column(
            "settled_at", sa.DateTime(timezone=True), nullable=True,
            comment="本地结算时间（NULL=未结算，扫描锚点）",
        ))
        batch.alter_column(
            "status",
            existing_type=sa.String(length=20),
            existing_nullable=False,
            comment="PENDING | FILLED | FAILED（订单生命周期）；结算结果见 settle_outcome/win/pnl",
        )
        batch.create_index(
            "ix_trade_orders_settle_pending", ["window_start"],
            unique=False,
            postgresql_where=sa.text(
                "status = 'FILLED' AND settled_at IS NULL AND window_start IS NOT NULL"),
        )


def downgrade() -> None:
    with op.batch_alter_table("trade_orders") as batch:
        batch.drop_index("ix_trade_orders_settle_pending")
        batch.alter_column(
            "status",
            existing_type=sa.String(length=20),
            existing_nullable=False,
            comment="PENDING | FILLED | FAILED",
        )
        batch.drop_column("settled_at")
        batch.drop_column("pnl")
        batch.drop_column("settle_price")
        batch.drop_column("win")
        batch.drop_column("settle_outcome")
        batch.drop_column("direction")
