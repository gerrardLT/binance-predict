"""fake_breakout_signals 增加 5 分钟兑现结算回读列

Revision ID: i9b0c1d2e3f4
Revises: h8b9c0d1e2f3
Create Date: 2026-08-13 16:30:00.000000

每条假突破信号同时验证两种兑现周期的实盘表现（离线回测：5m 兑现 77.4%
vs 15m 兑现 80.0%，均无实盘验证）：
- settle_btc_price_5m / settle_outcome_5m：信号时刻 +5min 回读 BTC 价与方向
- 原 settle_btc_price / settle_outcome 保持 15m 口径（对齐 15m 市场到期）

两列均可空，存量信号行为不变（5m 列保持 NULL）。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "i9b0c1d2e3f4"
down_revision = "h8b9c0d1e2f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "settle_btc_price_5m", sa.Float(), nullable=True,
            comment="信号时刻 +5min 回读的 BTC 现货中间价（5m 兑现口径验证）",
        ),
    )
    op.add_column(
        "fake_breakout_signals",
        sa.Column(
            "settle_outcome_5m", sa.String(length=10), nullable=True,
            comment="5m 兑现方向 UP | DOWN（只看符号：settle_btc_5m < btc_price → DOWN 赢）",
        ),
    )


def downgrade() -> None:
    op.drop_column("fake_breakout_signals", "settle_outcome_5m")
    op.drop_column("fake_breakout_signals", "settle_btc_price_5m")
