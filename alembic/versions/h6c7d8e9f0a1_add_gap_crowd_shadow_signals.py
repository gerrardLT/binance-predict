"""gap_crowd 影子信号表（2026-10-10 研究轮 promote）

Revision ID: h6c7d8e9f0a1
Revises: g5b6c7d8e9f0
Create Date: 2026-10-10

研究出处：output/research_runs/20261009T191824Z-reversal-state-certainty/
（盲测确认的报价-人群错位规则，用户拍板注册影子+实盘、实盘默认 OFF）
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "h6c7d8e9f0a1"
down_revision = "g5b6c7d8e9f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gap_crowd_shadow_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False,
                  comment="信号口径版本：gap_crowd_5m_v1 / gap_crowd_15m_v1"),
        sa.Column("market_period", sa.String(length=5), server_default="5m", nullable=False,
                  comment="市场周期：5m | 15m"),
        sa.Column("window_start", sa.BigInteger(), nullable=False, comment="触发窗 start_time（ms）"),
        sa.Column("window_end", sa.BigInteger(), nullable=False, comment="触发窗 end_time（ms）"),
        sa.Column("trigger_ts", sa.BigInteger(), nullable=False, comment="触发采样时刻（ms）"),
        sa.Column("elapsed_s", sa.Float(), nullable=False, comment="触发时点窗内已耗时（秒）"),
        sa.Column("remain_s", sa.Float(), nullable=False, comment="触发时点距收盘剩余（秒）"),
        sa.Column("gap", sa.Float(), nullable=False, comment="gap = up_price − (1 − down_pct/100)"),
        sa.Column("down_pct", sa.Float(), nullable=True, comment="触发时刻人群看跌倾向（%）"),
        sa.Column("direction", sa.String(length=4), server_default="UP", nullable=False,
                  comment="押注方向：恒 UP"),
        sa.Column("up_price", sa.Float(), nullable=True, comment="触发时刻 UP 报价"),
        sa.Column("down_price", sa.Float(), nullable=True, comment="触发时刻 DOWN 报价"),
        sa.Column("participants", sa.Float(), nullable=True, comment="触发时刻参与人数"),
        sa.Column("trade_volume", sa.Float(), nullable=True, comment="触发时刻成交量"),
        sa.Column("entry_up_price", sa.Float(), nullable=True, comment="入场 UP 真实价"),
        sa.Column("entry_down_price", sa.Float(), nullable=True, comment="入场 DOWN 真实价"),
        sa.Column("entry_quote_ts", sa.BigInteger(), nullable=True, comment="入场报价采样时刻（ms）"),
        sa.Column("entry_quote_kind", sa.String(length=8), nullable=True, comment="报价来源 real"),
        sa.Column("settle_outcome", sa.String(length=10), nullable=True, comment="结算方向 UP|DOWN"),
        sa.Column("win", sa.Boolean(), nullable=True, comment="命中"),
        sa.Column("ev_at_entry", sa.Float(), nullable=True, comment="单注 EV（真实价口径）"),
        sa.Column("status", sa.String(length=10), server_default="PENDING", nullable=False,
                  comment="PENDING | SETTLED | VOID"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "window_start", name="uq_gap_crowd_version_window"),
    )
    op.create_index("ix_gap_crowd_status", "gap_crowd_shadow_signals", ["status"])
    op.create_index("ix_gap_crowd_window_start", "gap_crowd_shadow_signals", ["window_start"])


def downgrade() -> None:
    op.drop_index("ix_gap_crowd_window_start", table_name="gap_crowd_shadow_signals")
    op.drop_index("ix_gap_crowd_status", table_name="gap_crowd_shadow_signals")
    op.drop_table("gap_crowd_shadow_signals")
