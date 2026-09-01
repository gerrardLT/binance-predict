"""add pattern_shadow_signals table (hm_touch_down_v1 shadow entry)

Revision ID: p7i0j1k2l3m4
Revises: y3z4a5b6c7d8
Create Date: 2026-09-01

K 线形态入场影子信号表：弱收盘上吊线 → 次 15m 周期内等反弹触及 +0.25×ATR
记录押 DOWN 的虚拟入场（快照触及时刻真实报价）。只记录不下注，
仅 TOUCHED 行按目标根收阴结算，攒影子期真实样本复核后人工决定 LIVE 转正。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "p7i0j1k2l3m4"
down_revision = "y3z4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pattern_shadow_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=24), nullable=False,
            comment="信号口径版本：hm_touch_down_v1（与冻结规则一一对应）",
        ),
        sa.Column(
            "signal_bar_start", sa.BigInteger(), nullable=False,
            comment="信号根（15m 上吊线）open_time（ms）",
        ),
        sa.Column(
            "signal_bar_end", sa.BigInteger(), nullable=False,
            comment="信号根 close_time（ms）= 判定时刻",
        ),
        sa.Column(
            "target_bar_start", sa.BigInteger(), nullable=False,
            comment="目标周期（次 15m）open_time（ms）= signal_bar_end",
        ),
        sa.Column(
            "atr_snapshot", sa.Float(), nullable=True,
            comment="信号根 ATR20 快照（atr_series 口径：前 20 根 range% 均值 ×open，ex-ante）",
        ),
        sa.Column(
            "signal_bar_open", sa.Float(), nullable=True,
            comment="信号根开盘价（审计快照）",
        ),
        sa.Column(
            "signal_bar_close", sa.Float(), nullable=True,
            comment="信号根收盘价（审计快照）",
        ),
        sa.Column(
            "clv", sa.Float(), nullable=True,
            comment="信号根 CLV=(close−low)/(high−low)（≤0.75 触发，range≤0 不触发）",
        ),
        sa.Column(
            "target_open", sa.Float(), nullable=True,
            comment="目标周期开盘价 O（入场锚点，fetch_kline_open 回读）",
        ),
        sa.Column(
            "up_level", sa.Float(), nullable=True,
            comment="上障碍 = O+0.25×ATR（触及即虚拟入场）",
        ),
        sa.Column(
            "dn_level", sa.Float(), nullable=True,
            comment="下障碍 = O−0.25×ATR（先破即放弃）",
        ),
        sa.Column(
            "entry_state", sa.String(length=16), nullable=False,
            server_default="WAITING",
            comment=("入场状态机：WAITING | TOUCHED | ABANDON_LOWER | ABANDON_LATE | "
                     "NOT_TOUCHED | FEED_GAP | NO_DATA | RESTART_GAP"),
        ),
        sa.Column(
            "touch_ts", sa.BigInteger(), nullable=True,
            comment="触及时刻（ms）；实时裁决为采样时刻，1m 重建为所在 1m 棒 open_time 近似",
        ),
        sa.Column(
            "touch_price", sa.Float(), nullable=True,
            comment="触及价格；实时裁决为当时 mid，1m 重建为上障碍价近似",
        ),
        sa.Column(
            "entry_down_quote", sa.Float(), nullable=True,
            comment="TOUCHED 时刻 15m 市场 DOWN token 真实报价（未来护栏定标数据；缺失/重建为 NULL）",
        ),
        sa.Column(
            "settle_open", sa.Float(), nullable=True,
            comment="目标根开盘价（结算回读）",
        ),
        sa.Column(
            "settle_close", sa.Float(), nullable=True,
            comment="目标根收盘价（结算回读）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="目标根方向 DOWN（赢）| UP（输）| NOISE（平盘）；仅 TOUCHED 行填写",
        ),
        sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="押 DOWN 命中 = 目标根收阴（close<open）；非 TOUCHED/NOISE 为 NULL",
        ),
        sa.Column(
            "status", sa.String(length=10), nullable=False,
            server_default="PENDING",
            comment="PENDING（入场裁决/等结算）| SETTLED（TOUCHED 已结算）| EXPIRED（未触/放弃/超时）",
        ),
        sa.Column(
            "rule_text", sa.Text(), nullable=False,
            comment="冻结规则原文（预注册口径逐字落库，审计用）",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(),
        ),
        sa.Column(
            "settled_at", sa.DateTime(timezone=True), nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "signal_bar_start", name="uq_pshadow_version_bar"),
    )
    op.create_index("ix_pshadow_status", "pattern_shadow_signals", ["status"])
    op.create_index("ix_pshadow_target_bar", "pattern_shadow_signals", ["target_bar_start"])


def downgrade() -> None:
    op.drop_index("ix_pshadow_target_bar", table_name="pattern_shadow_signals")
    op.drop_index("ix_pshadow_status", table_name="pattern_shadow_signals")
    op.drop_table("pattern_shadow_signals")
