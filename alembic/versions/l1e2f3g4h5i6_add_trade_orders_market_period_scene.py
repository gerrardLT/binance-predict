"""trade_orders 加 market_period / scene_signal_id（多通道实盘，15m 市场结算分流）

Revision ID: l1e2f3g4h5i6
Revises: l2e3f4g5h6i7
Create Date: 2026-08-24 12:00:00.000000

背景：多通道实盘（MultiLiveTrader）打通 15m 市场下单（场景信号 S1/S5/S2/S4）。
15m 周期起点（900s 网格）与 5m 窗口起点（300s 网格）数值重合——若不加区分，
trade_settler 会把 15m 订单错配到同名 5m SentimentWindow 用错误口径结算输赢。

- market_period：下单市场周期（'5m' | '15m'），存量行 server_default 回填 '5m'
- scene_signal_id：场景订单关联的 fake_breakout_signals.id（15m 结算回读）；
  与 signal_id 同为纯整型逻辑关联（不加 FK，fake_breakout_signals 历史上有
  TRUNCATE 重积累运维，FK 约束会卡此类操作——与 signal_id 现状一致）
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "l1e2f3g4h5i6"
down_revision = "l2e3f4g5h6i7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trade_orders",
        sa.Column(
            "market_period", sa.String(length=8), nullable=False,
            server_default="5m",
            comment="下单市场周期 5m | 15m（结算分流依据：15m 走 FakeBreakoutSignal 结算）",
        ),
    )
    op.add_column(
        "trade_orders",
        sa.Column(
            "scene_signal_id", sa.Integer(), nullable=True,
            comment="场景订单关联的 fake_breakout_signals.id（15m 结算回读；仅 scene_* 通道写入）",
        ),
    )


def downgrade() -> None:
    op.drop_column("trade_orders", "scene_signal_id")
    op.drop_column("trade_orders", "market_period")
