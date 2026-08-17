"""add scene stats columns (pattern_type / EV / cumulative stats)

Revision ID: r8i9j0k1l2m3
Revises: q5h8i9j0k1l2
Create Date: 2026-08-17

真 OOS 修正版上线（2026-08-17）：
- S1 bull_exhaust 升级完整终验口径 F22×F18×F25（补 4h 区间上沿 pos4h≥0.9 条件）
- S4 momentum_fade 新上线（F40 连阳≥3 × F06 光头阳，无破位要求，每周期独立判定）
- 新增统计维度 6 列：pattern_type 场景类型 / ev_at_entry 入场时刻预期 EV /
  cumulative_winrate + cumulative_ev 累计实盘指标（结算时回填最新行）/
  n_events_last_7d 近 7 日频率 / max_drawdown_curves 按周六归档的收益曲线回撤快照
（main.py 内联幂等迁移与此等价，存量 dev 库安全网双轨）
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "r8i9j0k1l2m3"
down_revision = "q5h8i9j0k1l2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "pattern_type",
            sa.String(32),
            nullable=True,
            comment="场景类型：bull_exhaust(破4h高·光头阳·4h上沿) | bear_exhaust(破4h低·收阴·放量) | "
                    "momentum_fade(连阳≥3·光头阳，无破位要求)；旧行/旧 A+B 时代信号为 NULL",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "ev_at_entry",
            sa.Float(),
            nullable=True,
            comment="@entry 价计算的 EV=p×(1-FEE)/entry−1（p=真 OOS 胜率点估计）；结算时回填",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "cumulative_winrate",
            sa.Float(),
            nullable=True,
            comment="累计胜率（该场景全部已结算正式信号；回填到最新一条 SETTLED 行）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "cumulative_ev",
            sa.Float(),
            nullable=True,
            comment="累计实现 EV/事件（1 USDT 本金口径：赢 0.98/entry−1，输 −1）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "n_events_last_7d",
            sa.Integer(),
            nullable=True,
            comment="近 7 日该场景事件计数（频率监控，含未结算）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "max_drawdown_curves",
            JSONB(),
            nullable=True,
            comment="回撤曲线快照 {周六yymmdd: {equity_curve, peak_equity, dd}}——结算时按周覆盖更新",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "max_drawdown_curves")
    op.drop_column("fake_breakout_signals", "n_events_last_7d")
    op.drop_column("fake_breakout_signals", "cumulative_ev")
    op.drop_column("fake_breakout_signals", "cumulative_winrate")
    op.drop_column("fake_breakout_signals", "ev_at_entry")
    op.drop_column("fake_breakout_signals", "pattern_type")
