"""add sentiment_windows price curves (归档时永久化下注价格曲线)

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-17 12:00:00.000000

背景：prediction_market_samples 采样表仅保留最近 1 小时（main.py 归档循环清理），
历史 up_price/down_price 被永久删除，导致经济账（EV=胜率/买入价-1）只能用
price≈chance 近似。本迁移为 sentiment_windows 新增两列 JSONB 价格曲线，
归档时从采样表快照永久化。两列均可空，存量行不受影响。手写脚本，字段/类型/
nullable/comment 严格对齐 src/binance_predict/db/models.py 的 SentimentWindow
（单一事实源）。

新增列：
  curve_up_price    JSONB NULL  UP token 价格时间序列 [{t, v}, ...]，v 为 0~1
  curve_down_price  JSONB NULL  DOWN token 价格时间序列 [{t, v}, ...]，v 为 0~1
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：为 sentiment_windows 追加 2 列价格曲线（可空，存量行不受影响）。"""
    op.add_column(
        "sentiment_windows",
        sa.Column(
            "curve_up_price",
            JSONB(),
            nullable=True,
            comment="UP token 价格时间序列 [{t, v}, ...]，v 为 0~1",
        ),
    )
    op.add_column(
        "sentiment_windows",
        sa.Column(
            "curve_down_price",
            JSONB(),
            nullable=True,
            comment="DOWN token 价格时间序列 [{t, v}, ...]，v 为 0~1",
        ),
    )


def downgrade() -> None:
    """降级：逆序删除 2 列价格曲线。"""
    op.drop_column("sentiment_windows", "curve_down_price")
    op.drop_column("sentiment_windows", "curve_up_price")
