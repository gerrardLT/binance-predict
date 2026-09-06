"""add absorption_shadow_signals table (absorption_follow_v1 shadow family)

Revision ID: t4u5v6w7x8y9
Revises: r8s9t0u1v2w3
Create Date: 2026-09-04

吸收/欠反应跟随影子信号表：5m 窗内报价对 BTC 位移「欠反应」（有人犹豫/知情者吸筹）
→ 跟随 btc 方向补涨押注的影子重放。双 variant（TD=120 / TD=150）各维护独立滚动标定。
只记录不下注，物理隔离于下单路径（本表不被任何下单代码引用，不进 X4_VERSIONS/LIVE_CHANNELS）。
归档后处理（同 quote_edge_detector）：窗已结算 → 直接落 SETTLED，无 PENDING 阶段。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "t4u5v6w7x8y9"
down_revision = "r8s9t0u1v2w3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "absorption_shadow_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=32), nullable=False,
            comment="信号口径版本：absorption_follow_td120_v1 / absorption_follow_td150_v1",
        ),
        sa.Column(
            "window_start", sa.BigInteger(), nullable=False,
            comment="触发窗（5m 情绪窗）start_time（ms）",
        ),
        sa.Column(
            "window_end", sa.BigInteger(), nullable=False,
            comment="触发窗 end_time（ms）",
        ),
        sa.Column(
            "td", sa.Integer(), nullable=False,
            comment="判定时点（窗开后秒）：120 / 150",
        ),
        sa.Column(
            "direction", sa.String(length=4), nullable=False,
            comment="follow 押注方向：btc涨→UP / btc跌→DOWN",
        ),
        sa.Column(
            "k", sa.Float(), nullable=True,
            comment="标定弹性 k（pp/bp）：up_move ≈ k·btc_move + b",
        ),
        sa.Column(
            "b", sa.Float(), nullable=True,
            comment="标定截距 b（pp）",
        ),
        sa.Column(
            "disp_gate", sa.Float(), nullable=True,
            comment="位移门 = 标定缓冲 p50(|btc_move|)（bp）",
        ),
        sa.Column(
            "under_gate", sa.Float(), nullable=True,
            comment="欠反应门 = 标定缓冲 p80(under)（pp，过位移门子集）",
        ),
        sa.Column(
            "calib_n", sa.Integer(), nullable=True,
            comment="标定缓冲样本窗数（trailing ~14 天，ex-ante）",
        ),
        sa.Column(
            "btc_move_bp", sa.Float(), nullable=True,
            comment="BTC 位移 (btc@TD−open)/open×1e4（bp）",
        ),
        sa.Column(
            "up_move_pp", sa.Float(), nullable=True,
            comment="UP 真实价位移 (up_price@TD−open)×100（pp）",
        ),
        sa.Column(
            "resid_pp", sa.Float(), nullable=True,
            comment="残差 resid = up_move − (k·btc_move + b)（pp）",
        ),
        sa.Column(
            "under_pp", sa.Float(), nullable=True,
            comment="欠反应 under = −sign(btc_move)·resid（pp，≥under_gate 触发）",
        ),
        sa.Column(
            "dvol", sa.Float(), nullable=True,
            comment="Δtrade_volume（@TD−open，soft 记录维度，不作门）",
        ),
        sa.Column(
            "dpar", sa.Float(), nullable=True,
            comment="Δparticipants（@TD−open，soft 记录维度，不作门）",
        ),
        sa.Column(
            "entry_up_price", sa.Float(), nullable=True,
            comment="TD 时刻 UP token 真实价（押 UP 的入场价）",
        ),
        sa.Column(
            "entry_down_price", sa.Float(), nullable=True,
            comment="TD 时刻 DOWN token 真实价（押 DOWN 的入场价）",
        ),
        sa.Column(
            "entry_quote_ts", sa.BigInteger(), nullable=True,
            comment="入场报价采样时刻（ms，≤TD 最晚点）",
        ),
        sa.Column(
            "entry_quote_kind", sa.String(length=8), nullable=True,
            comment="报价来源：real（token 价）/ NULL（缺失不落表）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="触发窗结算方向 UP | DOWN",
        ),
        sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="follow 命中 = 窗 outcome == direction",
        ),
        sa.Column(
            "ev_at_entry", sa.Float(), nullable=True,
            comment="单注 EV（真实价口径）：赢 0.98/q−1 / 输 −1（费 2% 无溢价）",
        ),
        sa.Column(
            "status", sa.String(length=10), nullable=False,
            server_default="SETTLED",
            comment="SETTLED（归档后处理落表即结算，无 PENDING 阶段）",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "window_start", name="uq_abs_version_window"),
    )
    op.create_index("ix_abs_status", "absorption_shadow_signals", ["status"])
    op.create_index("ix_abs_window_start", "absorption_shadow_signals", ["window_start"])


def downgrade() -> None:
    op.drop_index("ix_abs_window_start", table_name="absorption_shadow_signals")
    op.drop_index("ix_abs_status", table_name="absorption_shadow_signals")
    op.drop_table("absorption_shadow_signals")
