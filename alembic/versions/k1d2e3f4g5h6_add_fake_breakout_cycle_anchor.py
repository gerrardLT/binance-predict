"""fake_breakout_signals 切换周期锚点结算口径（清表重积累）

Revision ID: k1d2e3f4g5h6
Revises: j0c1d2e3f4g5
Create Date: 2026-08-14 16:30:00.000000

背景：旧口径用「信号时刻价」做输赢判定锚点，与币安预测市场真实结算规则
（UP 赢 ⟺ 周期末价 > 周期起点价）脱节——信号在周期中段触发，P(信号) ≠ P(周期起点)，
导致胜率系统性高估。本迁移为周期锚点口径加列：

- market_start_15m / cycle_open_price_15m：信号所在 15m 周期起点与其开盘价 P(S15)
- market_start_5m / market_end_5m / cycle_open_price_5m：所在 5m 周期 [S5,E5] 与开盘价

用户已拍板：旧口径历史数据全部废弃，清空重积累。
安全措施：先建备份表 fake_breakout_signals_pre_anchor_bak 再 TRUNCATE。

注意：downgrade 只删列，不恢复行数据；如需恢复历史行，手工从备份表
INSERT SELECT，确认无误后可人工 DROP 备份表。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "k1d2e3f4g5h6"
down_revision = "j0c1d2e3f4g5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "market_start_15m", sa.BigInteger(), nullable=True,
            comment="信号所在 15m 市场周期起点 start_date（ms）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "cycle_open_price_15m", sa.Float(), nullable=True,
            comment="15m 周期开盘价 P(S)：周期锚点结算的判定基准",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "market_start_5m", sa.BigInteger(), nullable=True,
            comment="信号所在 5m 市场周期起点 start_date（ms）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "market_end_5m", sa.BigInteger(), nullable=True,
            comment="信号所在 5m 市场周期末 end_date（ms，5m 结算死线基准）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "cycle_open_price_5m", sa.Float(), nullable=True,
            comment="5m 周期开盘价 P(S5)：5m 口径判定基准",
        ),
    )
    # 清表重积累：先备份再清空（同事务，失败整体回滚）。
    # IF NOT EXISTS：重跑时保留首次备份（旧口径数据只有一份，最珍贵）
    op.execute(
        "CREATE TABLE IF NOT EXISTS fake_breakout_signals_pre_anchor_bak AS SELECT * FROM fake_breakout_signals"
    )
    op.execute("TRUNCATE TABLE fake_breakout_signals RESTART IDENTITY")


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "cycle_open_price_5m")
    op.drop_column("fake_breakout_signals", "market_end_5m")
    op.drop_column("fake_breakout_signals", "market_start_5m")
    op.drop_column("fake_breakout_signals", "cycle_open_price_15m")
    op.drop_column("fake_breakout_signals", "market_start_15m")
