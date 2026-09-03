"""add entry quote columns to kline_shadow_signals (KREV/reversal/nextbar real-EV)

Revision ID: q9l2m3n4p5r6
Revises: p7i0j1k2l3m4
Create Date: 2026-09-03

K 线族影子（KREV / 反转 P1/P2 / nextbar）过去只按次根 K 线涨跌结算、不记报价，
面板 EV/累计 EV 恒空。本迁移给共表 kline_shadow_signals 增三列，存信号落库时刻
（目标窗开盘后首次轮询 ~0~60s）快照的目标窗 UP/DOWN 真实报价与快照时刻，使聚合层
能按既有 `_shadow_realized_ev` 口径（赢 0.98/q−1 / 输 −1）现算真实 EV。全部可空，
存量行与冷启动回补行为 NULL（EV 不计，与旧口径一致），无回填、无数据风险。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "q9l2m3n4p5r6"
down_revision = "p7i0j1k2l3m4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "kline_shadow_signals",
        sa.Column(
            "entry_up_price", sa.Float(), nullable=True,
            comment="目标窗开盘后首次轮询快照的 UP token 报价（押 UP 的入场价；窗口未对齐/缺失为 NULL）",
        ),
    )
    op.add_column(
        "kline_shadow_signals",
        sa.Column(
            "entry_down_price", sa.Float(), nullable=True,
            comment="同时刻 DOWN token 报价（押 DOWN 的入场价，对称快照；缺失为 NULL）",
        ),
    )
    op.add_column(
        "kline_shadow_signals",
        sa.Column(
            "entry_quote_ts", sa.BigInteger(), nullable=True,
            comment="入场报价快照时刻（ms）；offset=entry_quote_ts−target_bar_start 可审计入场时点",
        ),
    )


def downgrade() -> None:
    op.drop_column("kline_shadow_signals", "entry_quote_ts")
    op.drop_column("kline_shadow_signals", "entry_down_price")
    op.drop_column("kline_shadow_signals", "entry_up_price")
