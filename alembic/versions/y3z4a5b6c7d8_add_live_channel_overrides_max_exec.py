"""live_channel_overrides 加 max_exec_price（执行价护栏自定义覆盖持久化）

Revision ID: y3z4a5b6c7d8
Revises: n3g4h5i6j7k8
Create Date: 2026-08-31 12:00:00.000000

背景：通道护栏此前只有代码预设值（ChannelSpec.auto_max_exec，可被
LIVE_CHANNELS_JSON 覆盖），toggle 端点不暴露护栏热调。本迁移为
live_channel_overrides 加 max_exec_price 列，使前端通道面板可改护栏：
- NULL = 未自定义（回落通道预设 auto_max_exec）
- 非 NULL = 用户最后一次保存的护栏值（重启恢复的就是它）

存量数据全为 NULL（历史从未自定义），无需回填。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "y3z4a5b6c7d8"
down_revision = "n3g4h5i6j7k8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "live_channel_overrides",
        sa.Column(
            "max_exec_price", sa.Float(), nullable=True,
            comment="执行价护栏自定义覆盖（NULL=回落通道预设 auto_max_exec）",
        ),
    )


def downgrade() -> None:
    op.drop_column("live_channel_overrides", "max_exec_price")
