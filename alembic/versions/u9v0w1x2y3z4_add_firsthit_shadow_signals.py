"""add firsthit_shadow_signals table (firsthit_down shadow family)

Revision ID: u9v0w1x2y3z4
Revises: t4u5v6w7x8y9
Create Date: 2026-09-07

5m DOWN 首触反转影子信号表：窗内 DOWN token 报价**首次**进入 (0.005,0.1] → 以该
报价买 DOWN 的前向重放。三 version 同表隔离（firsthit_down_v1 基底 / _body_v1
G1 body_r≤0.35 / _chg_v1 G3 chg≤+2.82bp），全特征落库供事后重构交叉门。
归档后处理（同 absorption/quote_edge）：窗已结算 → 直接落 SETTLED，无 PENDING 阶段。
EV = 赢 0.98/q−1 / 输 −1（费 2% 无溢价，逐事件真实触发价）。
只记录不下注，物理隔离于下单路径（本表不被任何下单代码引用，不进 X4_VERSIONS/LIVE_CHANNELS）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "u9v0w1x2y3z4"
down_revision = "t4u5v6w7x8y9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "firsthit_shadow_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=32), nullable=False,
            comment="firsthit_down_v1 / firsthit_down_body_v1 / firsthit_down_chg_v1",
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
            "trigger_ts", sa.BigInteger(), nullable=False,
            comment="首触采样时刻（ms，DOWN 首次进入 (0.005,0.1]）",
        ),
        sa.Column(
            "td_sec", sa.Integer(), nullable=False,
            comment="首触距窗开秒数",
        ),
        sa.Column(
            "entry_down_price", sa.Float(), nullable=False,
            comment="首触 DOWN token 真实报价 q（EV 的入场价）",
        ),
        sa.Column(
            "chg_bps", sa.Float(), nullable=True,
            comment="BTC 相对开盘涨跌 (btc@触−开盘)/开盘×1e4（bp）",
        ),
        sa.Column(
            "body_r", sa.Float(), nullable=True,
            comment="路径归一实体 |btc@触−开盘|/(触发前路径 max−min)",
        ),
        sa.Column(
            "wick01", sa.Float(), nullable=True,
            comment="上影二元（路径高点>max(btc@触,开盘)=1，供 G6/G7 重构）",
        ),
        sa.Column(
            "rng_bps", sa.Float(), nullable=True,
            comment="触发前路径振幅 (max−min)/开盘×1e4（bp）",
        ),
        sa.Column(
            "npts", sa.Integer(), nullable=False,
            comment="触发前 btc 采样点数（≥8 才落表 = v2 主分析口径）",
        ),
        sa.Column(
            "dvol", sa.Float(), nullable=True,
            comment="Δtrade_volume（@触−open，soft 记录维度，不作门）",
        ),
        sa.Column(
            "dpar", sa.Float(), nullable=True,
            comment="Δparticipants（@触−open，soft 记录维度，不作门）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="触发窗结算方向 UP | DOWN",
        ),
        sa.Column(
            "win", sa.Boolean(), nullable=True,
            comment="买 DOWN 命中 = 窗 outcome == DOWN",
        ),
        sa.Column(
            "ev_at_entry", sa.Float(), nullable=True,
            comment="单注 EV（真实触发价口径）：赢 0.98/q−1 / 输 −1（费 2% 无溢价）",
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
        sa.UniqueConstraint("version", "window_start", name="uq_firsthit_version_window"),
    )
    op.create_index("ix_firsthit_status", "firsthit_shadow_signals", ["status"])
    op.create_index("ix_firsthit_window_start", "firsthit_shadow_signals", ["window_start"])


def downgrade() -> None:
    op.drop_index("ix_firsthit_window_start", table_name="firsthit_shadow_signals")
    op.drop_index("ix_firsthit_status", table_name="firsthit_shadow_signals")
    op.drop_table("firsthit_shadow_signals")
