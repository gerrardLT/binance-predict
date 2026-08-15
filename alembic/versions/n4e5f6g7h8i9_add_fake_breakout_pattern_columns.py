"""add fake_breakout pattern/close_pos/vol_ratio (场景①②收盘确认)

Revision ID: n4e5f6g7h8i9
Revises: m3e4f5g6h7i8
Create Date: 2026-08-15

场景信号系统模式升级（旧 A+B 过滤整体替换）：破位记 pending → 15m 周期
收盘质量确认 → 次周期信号（scripts/local_continuation_discovery.py 180 天
发现集→验证集盲验）：
- 场景① bull_exhaust：破 4h 阻力 + 收阳 + 光头（close_pos ≥ 0.85）→ 次周期 DOWN（验证集 63.6%）
- 场景② bear_exhaust：破 4h 支撑 + 收阴 + 放量（vol_ratio ≥ 2.0）→ 次周期 UP（验证集 57.8%）
三列在周期边界确认时落表，邮件/API/stats 按场景区分胜率；旧 A+B 时代信号为 NULL。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "n4e5f6g7h8i9"
down_revision = "m3e4f5g6h7i8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "pattern",
            sa.String(length=16),
            nullable=True,
            comment="收盘确认场景：bull_exhaust(破阻力+光头阳→次周期DOWN) | "
                    "bear_exhaust(破支撑+收阴+放量→次周期UP)；旧 A+B 时代信号为 NULL",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "close_pos",
            sa.Float(),
            nullable=True,
            comment="信号周期收盘位置 (C-L)/(H-L)：场景①判定输入与审计（阈值 0.85）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "vol_ratio",
            sa.Float(),
            nullable=True,
            comment="信号周期量比 = 本 15m 成交量 / 前 20 根均量：场景②判定输入与审计（阈值 2.0）",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "vol_ratio")
    op.drop_column("fake_breakout_signals", "close_pos")
    op.drop_column("fake_breakout_signals", "pattern")
