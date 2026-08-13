"""fake_breakout_signals 增加级别/方向/UP token 快照列

Revision ID: j0c1d2e3f4g5
Revises: i9b0c1d2e3f4
Create Date: 2026-08-13 17:00:00.000000

检测器从「单级别（日线）单向（阻力）」升级为「三级别 × 双向」：
- level：1h（12 个 5m 窗口 closes）| 4h（48）| daily（288）
- side：high（盘中冲过阻力→卖跌信号）| low（盘中跌破支撑→买涨信号）
- up_price_5m / up_price_15m：UP token 快照（支撑方向的目标 token 是 UP）

回测依据（本地一个月，scripts/local_combo_level_matrix_check.py）：
- 1h 破阻力→DOWN：421 注，入场 0.107，方向胜率 65.1%
- 4h 破阻力→DOWN：211 注，入场 0.116，方向胜率 73.5%
- 日线破阻力→DOWN：80 注，入场 0.146，方向胜率 80.0%
- 支撑方向对称成立（1h 支撑→UP：447 注，方向胜率 62.4%）

存量行回填 level='daily' side='high'（旧检测器只有日线阻力），语义不变。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "j0c1d2e3f4g5"
down_revision = "i9b0c1d2e3f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "level", sa.String(length=8), nullable=False, server_default="daily",
            comment="破位级别：1h | 4h | daily（存量行回填 daily）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "side", sa.String(length=4), nullable=False, server_default="high",
            comment="破位方向：high（冲过阻力→卖跌）| low（跌破支撑→买涨）（存量行回填 high）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "up_price_5m", sa.Float(), nullable=True,
            comment="信号时刻 5m 市场 UP token 最近采样报价（支撑方向目标 token）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "up_price_15m", sa.Float(), nullable=True,
            comment="信号时刻 15m 市场 UP token 最近采样报价",
        ),
    )
    op.create_index(
        "ix_fbs_level_side", "fake_breakout_signals", ["level", "side"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_fbs_level_side", table_name="fake_breakout_signals")
    op.drop_column("fake_breakout_signals", "up_price_15m")
    op.drop_column("fake_breakout_signals", "up_price_5m")
    op.drop_column("fake_breakout_signals", "side")
    op.drop_column("fake_breakout_signals", "level")
