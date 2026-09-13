"""notification_channel_overrides 逐信号通知配置表

Revision ID: a9b0c1d2e3f4
Revises: z1a2b3c4d5e6
Create Date: 2026-09-13

缺行/缺键默认全开，部署零影响；物理凭据仍只在 .env。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "a9b0c1d2e3f4"
down_revision = "z1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "notification_channel_overrides",
        sa.Column(
            "channel", sa.String(length=64), primary_key=True,
            comment="LIVE_CHANNELS 通道名；__global__ 为全局渠道/低余额配置",
        ),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default=sa.true(),
            comment="该信号全部业务通知总开关",
        ),
        sa.Column(
            "config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb"),
            comment="经服务端白名单校验的事件、传输渠道与字段选择",
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
            comment="最后配置时间",
        ),
    )


def downgrade() -> None:
    op.drop_table("notification_channel_overrides")
