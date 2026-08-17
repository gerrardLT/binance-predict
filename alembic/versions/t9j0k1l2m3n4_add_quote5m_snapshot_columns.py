"""add quote5m snapshot columns for S5 confirm entry

Revision ID: t9j0k1l2m3n4
Revises: r8i9j0k1l2m3
Create Date: 2026-08-18

S5 bull_exhaust_confirm 确认入场模式上线（2026-08-18）：
- S1 信号后 5 分钟（次周期第 1 根 5m 收盘）回落确认 → 买 DOWN（360 天回测
  确认组 n=591 胜率 78.5% [75.0,81.6]，盈亏平衡入场价 0.77）
- 新增 3 列：quote5m_down_15m / quote5m_up_15m / quote5m_ts_15m——
  信号后 5min 时刻的 15m 市场报价快照（S5 真实入场价 + 定价对照）
（main.py 内联幂等迁移与此等价，存量 dev 库安全网双轨）
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "t9j0k1l2m3n4"
down_revision = "r8i9j0k1l2m3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "quote5m_down_15m",
            sa.Float(),
            nullable=True,
            comment="信号后 5min（次周期 1/3 处）15m 市场 DOWN 报价：S5 确认入场真实价；NULL=未抓到",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "quote5m_up_15m",
            sa.Float(),
            nullable=True,
            comment="信号后 5min 时刻 15m 市场 UP 报价（对称快照）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "quote5m_ts_15m",
            sa.BigInteger(),
            nullable=True,
            comment="+5min 报价快照时刻（ms）",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "quote5m_ts_15m")
    op.drop_column("fake_breakout_signals", "quote5m_up_15m")
    op.drop_column("fake_breakout_signals", "quote5m_down_15m")
