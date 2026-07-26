"""add sentiment_windows participants/volume curves (归档时永久化参与者与交易量时序)

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-26 12:00:00.000000

背景：此前归档仅保存参与者/交易量的窗口均值（avg_participants/avg_trade_volume），
时序曲线随 prediction_market_samples 清理策略永久丢失——momentum 类假设
（参与者增长率、交易量加速度）的原始证据无法回溯，重演了价格曲线仅剩近似值
的教训。本迁移为 sentiment_windows 新增两列 JSONB 时序曲线，归档时从采样表
快照永久化。两列均可空，存量行不受影响。手写脚本，字段/类型/nullable/comment
严格对齐 src/binance_predict/db/models.py 的 SentimentWindow（单一事实源）。

同期变更（无 schema 影响）：新增 settings.sample_retention_hours（默认 0=永不
删除原始采样），取代硬编码的 1 小时清理。

新增列：
  curve_participants  JSONB NULL  参与人数时间序列 [{t, v}, ...]
  curve_trade_volume  JSONB NULL  交易量时间序列 [{t, v}, ...]
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：为 sentiment_windows 追加 2 列参与者/交易量时序（可空，存量行不受影响）。"""
    op.add_column(
        "sentiment_windows",
        sa.Column(
            "curve_participants",
            JSONB(),
            nullable=True,
            comment="参与人数时间序列 [{t, v}, ...]",
        ),
    )
    op.add_column(
        "sentiment_windows",
        sa.Column(
            "curve_trade_volume",
            JSONB(),
            nullable=True,
            comment="交易量时间序列 [{t, v}, ...]",
        ),
    )


def downgrade() -> None:
    """降级：逆序删除 2 列时序曲线。"""
    op.drop_column("sentiment_windows", "curve_trade_volume")
    op.drop_column("sentiment_windows", "curve_participants")
