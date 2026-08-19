"""add misalignment_signals table (X4 shadow parallel)

Revision ID: v1k2l3m4n5o5
Revises: t9j0k1l2m3n4
Create Date: 2026-08-19

X4 情绪错位影子信号表：错位假设工厂唯一全闸门存活信号
（收阳 & end≤40 → 押次窗 DOWN，IS 65.6%/OOS 57.8%，EV+0.254 CI 下界>0）。
M4 影子并行：只记录不下注，次窗归档后回读真实报价与结算方向定案经济账。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "v1k2l3m4n5o5"
down_revision = "t9j0k1l2m3n4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "misalignment_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=40), nullable=False,
            server_default="x4_v1",
            comment="信号口径版本：x4_v1 = 收阳 & end≤40 → 次窗 DOWN（端点口径）",
        ),
        sa.Column(
            "window_start", sa.BigInteger(), nullable=False,
            comment="触发窗（5m 情绪窗）start_time（ms）",
        ),
        sa.Column(
            "window_end", sa.BigInteger(), nullable=False,
            comment="触发窗 end_time（ms）= 信号判定时刻",
        ),
        sa.Column(
            "end_pct", sa.Float(), nullable=False,
            comment="触发窗末 UP% 采样值（curve_up_pct 排序后末点）",
        ),
        sa.Column(
            "outcome_base", sa.String(length=10), nullable=False,
            comment="触发窗结算方向（X4 定义恒为 UP，审计冗余）",
        ),
        sa.Column(
            "direction", sa.String(length=4), nullable=False,
            server_default="DOWN", comment="押注方向（X4 定义恒为 DOWN）",
        ),
        sa.Column(
            "target_window_start", sa.BigInteger(), nullable=False,
            comment="目标窗（次 5m 窗）start_time（ms）= window_end",
        ),
        sa.Column(
            "entry_down_price", sa.Float(), nullable=True,
            comment="次窗 150s 决策点 DOWN token 真实价（curve_down_price ≤150s 末点）",
        ),
        sa.Column(
            "entry_up_price", sa.Float(), nullable=True,
            comment="同时刻 UP token 价（对称快照，定价对照用）",
        ),
        sa.Column(
            "entry_quote_ts", sa.BigInteger(), nullable=True,
            comment="入场报价采样点时刻（ms，距决策点最近且 ≤150s）",
        ),
        sa.Column(
            "entry_quote_kind", sa.String(length=8), nullable=True,
            comment="报价来源：real（token 价）/ proxy（chance/100）/ NULL（缺失）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="次窗结算方向 UP | DOWN（NOISE 无法判向）",
        ),
        sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="押 DOWN 命中 = 次窗 DOWN；NOISE/缺结算为 NULL",
        ),
        sa.Column(
            "ev_at_entry", sa.Float(), nullable=True,
            comment="单注 EV：赢 0.98/(entry+0.01)−1 / 输 −1（entry 截断 [0.01,0.99]）；无报价 NULL",
        ),
        sa.Column(
            "status", sa.String(length=10), nullable=False,
            server_default="PENDING",
            comment="PENDING（等次窗归档）| SETTLED（已结算）| EXPIRED（次窗缺数据/超时未结算）",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "window_start", name="uq_mis_version_window"),
    )
    op.create_index("ix_mis_status", "misalignment_signals", ["status"])
    op.create_index("ix_mis_target_window", "misalignment_signals", ["target_window_start"])


def downgrade() -> None:
    op.drop_index("ix_mis_target_window", table_name="misalignment_signals")
    op.drop_index("ix_mis_status", table_name="misalignment_signals")
    op.drop_table("misalignment_signals")
