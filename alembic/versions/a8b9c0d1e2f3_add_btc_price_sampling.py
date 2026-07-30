"""add btc price sampling (采样与归档记录 BTC 现货中间价)

Revision ID: a8b9c0d1e2f3
Revises: a7b8c9d0e1f2
Create Date: 2026-07-29 12:00:00.000000

背景：验证"情绪领先还是滞后价格"需要与情绪采样同时刻的 BTC 现货价序列，
但此前 prediction_market_samples 只存预测市场报价（up/down price/pct），
sentiment_windows 只有首尾两个价格点（entry/exit_price），局内 BTC 走势
从未被记录，领先/滞后、背离、加速度关系分析无原始数据可用。

本迁移新增两列（均可空，存量行不受影响，历史无法回补）：
  prediction_market_samples.btc_price  FLOAT NULL  采样时刻 BTC 现货中间价
  sentiment_windows.curve_btc_price    JSONB NULL  BTC 中间价时间序列 [{t, v}, ...]

手写脚本，字段/类型/nullable/comment 严格对齐 src/binance_predict/db/models.py
的 PredictionMarketSample / SentimentWindow（单一事实源）。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "a8b9c0d1e2f3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：采样表加 btc_price，归档表加 curve_btc_price（均可空）。"""
    op.add_column(
        "prediction_market_samples",
        sa.Column(
            "btc_price",
            sa.Float(),
            nullable=True,
            comment="采样时刻 BTC 现货中间价（spot bookTicker mid）",
        ),
    )
    op.add_column(
        "sentiment_windows",
        sa.Column(
            "curve_btc_price",
            JSONB(),
            nullable=True,
            comment="BTC 现货中间价时间序列 [{t, v}, ...]",
        ),
    )


def downgrade() -> None:
    """降级：逆序删除两列。"""
    op.drop_column("sentiment_windows", "curve_btc_price")
    op.drop_column("prediction_market_samples", "btc_price")
