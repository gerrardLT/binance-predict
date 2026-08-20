"""add trade_orders signal linkage columns (quote edge live trading)

Revision ID: u6f7g8h9i0j1
Revises: v1k2l3m4n5o5
Create Date: 2026-08-20

报价 edge 实盘下单（quote_momentum_v1 LIVE）：trade_orders 增加信号关联字段。
signal_version + window_start 联合唯一 = 每窗每版本至多一单（重启/并发兜底）；
signal_id 窗口结算后回填，实盘订单与影子信号（misalignment_signals）对账。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "u6f7g8h9i0j1"
down_revision = "v1k2l3m4n5o5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("trade_orders") as batch:
        batch.add_column(sa.Column(
            "signal_version", sa.String(length=40), nullable=True,
            comment="触发信号版本（quote_momentum_v1 等）；NULL=非信号驱动订单（旧路径）",
        ))
        batch.add_column(sa.Column(
            "window_start", sa.BigInteger(), nullable=True,
            comment="目标 5m 窗口起始 ms（与 signal_version 联合唯一）",
        ))
        batch.add_column(sa.Column(
            "signal_id", sa.Integer(), nullable=True,
            comment="窗口结算后回填的 misalignment_signals.id（实盘对账影子）",
        ))
        batch.create_unique_constraint(
            "uq_trade_orders_version_window", ["signal_version", "window_start"],
        )
        batch.create_index("ix_trade_orders_signal_version", ["signal_version"])


def downgrade() -> None:
    with op.batch_alter_table("trade_orders") as batch:
        batch.drop_index("ix_trade_orders_signal_version")
        batch.drop_constraint("uq_trade_orders_version_window", type_="unique")
        batch.drop_column("signal_id")
        batch.drop_column("window_start")
        batch.drop_column("signal_version")
