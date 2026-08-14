"""add fake_breakout filter metrics (A 周期内偏移 / B 破位幅度)

Revision ID: l2d3e4f5g6h7
Revises: k1d2e3f4g5h6
Create Date: 2026-08-14

过滤器落地（scripts/local_combo_filter_lab.py 回测结论，4 场景方向一致）：
- A 剩余时间：信号在 15m 周期内偏移 <6min 才可行动（尾段桶 140 注仅 2 胜）
- B 破位幅度：信号价偏离周期开盘价 <0.2%（>0.3% 桶 144 注 0 胜）
两指标在 fire 时计算落表，邮件/API/前端据此标注「可行动 / 被过滤」；
信号本身全部照常落表（积累过滤组 vs 未过滤组的实盘对照数据）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "l2d3e4f5g6h7"
down_revision = "k1d2e3f4g5h6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "cycle_offset_sec_15m",
            sa.Integer(),
            nullable=True,
            comment="信号触发时在 15m 周期内的偏移（秒，0~900）；过滤器 A（剩余时间）输入",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "break_pct",
            sa.Float(),
            nullable=True,
            comment="破位幅度 %：信号价 vs 15m 周期开盘价（破位方向）；过滤器 B 输入",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "break_pct")
    op.drop_column("fake_breakout_signals", "cycle_offset_sec_15m")
