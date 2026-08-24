"""live_channel_overrides 实盘通道配置覆盖表（toggle 持久化，重启不丢设定）

Revision ID: m2f3g4h5i6j7
Revises: l1e2f3g4h5i6
Create Date: 2026-08-24 14:00:00.000000

背景：多通道实盘的 toggle 端点原本是纯内存态，每次部署/重启回落到
LIVE_CHANNELS_JSON——用户在面板上设定的开关组合会静默丢失（历史痛点 R6
只解决了一半：env 持久化了初始集，运行时变更仍会丢）。

启动配置分层：代码默认 → LIVE_CHANNELS_JSON（env）→ 本表（最高优先级）。
toggle 端点在运行时生效成功后 upsert 本表；删行即回落到 env 层配置。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "m2f3g4h5i6j7"
down_revision = "l1e2f3g4h5i6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "live_channel_overrides",
        sa.Column(
            "channel", sa.String(length=64), primary_key=True,
            comment="通道名（LIVE_CHANNELS 白名单，如 quote_contrarian_v2）",
        ),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False,
            server_default=sa.false(), comment="通道开关",
        ),
        sa.Column(
            "amount_usdt", sa.Float(), nullable=False,
            comment="单笔金额 USDT（硬上限见 MAX_ORDER_AMOUNT_USDT）",
        ),
        sa.Column(
            "max_daily_orders", sa.Integer(), nullable=False,
            comment="日单量上限（硬上限见 MAX_DAILY_ORDERS_CAP）",
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
            comment="最后一次 toggle 生效时刻（配置变更审计）",
        ),
    )


def downgrade() -> None:
    op.drop_table("live_channel_overrides")
