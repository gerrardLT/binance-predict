"""add kline_shadow_signals table (KREV shadow parallel)

Revision ID: n3g4h5i6j7k8
Revises: m2f3g4h5i6j7
Create Date: 2026-08-28

K 线科学发现影子信号表：720d 发现流水线冻结注册表条件的实时重放
（KREV-A holdout 64.2%/EV+0.234，KREV-B 63.4%/EV+0.219）。
只记录不下注，次根收盘回读 OHLC 按回测口径结算，攒样本复核后人工 promote。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "n3g4h5i6j7k8"
down_revision = "m2f3g4h5i6j7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "kline_shadow_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=24), nullable=False,
            comment="信号口径版本：krev_a_v1 / krev_b_v1（与冻结注册表条件一一对应）",
        ),
        sa.Column(
            "discovery_id", sa.String(length=16), nullable=False,
            comment="冻结注册表 discovery_id（krev_a=fd191c44fb5c36 / krev_b=5c5e4c78ab4c3f）",
        ),
        sa.Column(
            "condition_text", sa.Text(), nullable=False,
            comment="注册表条件原文（逐字复制，审计口径保真用）",
        ),
        sa.Column(
            "timeframe", sa.String(length=4), nullable=False,
            server_default="15m", comment="信号周期（本族恒 15m）",
        ),
        sa.Column(
            "signal_bar_start", sa.BigInteger(), nullable=False,
            comment="信号根（15m）open_time（ms）",
        ),
        sa.Column(
            "signal_bar_end", sa.BigInteger(), nullable=False,
            comment="信号根 close_time（ms）= 判定时刻",
        ),
        sa.Column(
            "direction", sa.String(length=4), nullable=False,
            server_default="UP",
            comment="押注方向（reversal_1 信号根为阴线，恒押次根收阳 = UP）",
        ),
        sa.Column(
            "target_bar_start", sa.BigInteger(), nullable=False,
            comment="目标根（次 15m 根）open_time（ms）= signal_bar_end",
        ),
        sa.Column(
            "feature_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True,
            comment="触发时特征实际值快照（审计：实时值与离线口径对照）",
        ),
        sa.Column(
            "settle_open", sa.Float(), nullable=True,
            comment="次根开盘价（结算回读）",
        ),
        sa.Column(
            "settle_close", sa.Float(), nullable=True,
            comment="次根收盘价（结算回读）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="次根方向 UP | DOWN | NOISE（平盘无法判向）",
        ),
        sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="回测口径：次根收阳（close>open）即赢；NOISE/缺数据为 NULL",
        ),
        sa.Column(
            "status", sa.String(length=10), nullable=False,
            server_default="PENDING",
            comment="PENDING（等次根收盘）| SETTLED（已结算）| EXPIRED（缺数据/超时未结算）",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
        ),
        sa.Column(
            "settled_at", sa.DateTime(timezone=True), nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "signal_bar_start", name="uq_kshadow_version_bar"),
    )
    op.create_index("ix_kshadow_status", "kline_shadow_signals", ["status"])
    op.create_index("ix_kshadow_target_bar", "kline_shadow_signals", ["target_bar_start"])


def downgrade() -> None:
    op.drop_index("ix_kshadow_target_bar", table_name="kline_shadow_signals")
    op.drop_index("ix_kshadow_status", table_name="kline_shadow_signals")
    op.drop_table("kline_shadow_signals")
