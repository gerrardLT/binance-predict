"""shadow_version_overrides 影子版本开关表（前端手动下线，重启不丢）

Revision ID: r8s9t0u1v2w3
Revises: q9l2m3n4p5r6
Create Date: 2026-09-04

背景：影子信号版本只增不减（26 个），部分版本零触发或研究已翻篇（如 HM
上吊线族），但删代码成本高且族间有结算器寄生耦合（S5 深档借用 HM 结算循环）。
本表提供 version 级运行时开关：前端手动下线 → 检测器停止采集该版本新信号 +
面板置灰；历史已落库信号不受影响（下线≠删数据）；上线即恢复。

语义与 live_channel_overrides 同构（影子版）：无行 → 默认在线（回落代码默认，
部署零影响）；有行且 enabled=False → 下线；删行即恢复默认。与实盘通道表唯一
区别是默认值方向（实盘默认关、影子默认开）。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "r8s9t0u1v2w3"
down_revision = "q9l2m3n4p5r6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shadow_version_overrides",
        sa.Column(
            "version", sa.String(length=24), primary_key=True,
            comment="影子版本名（SHADOW_BENCH 白名单，如 hm_touch_down_v1 / combo_p1_v1）",
        ),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False,
            server_default=sa.true(),
            comment="在线=True（默认，采集+面板正常）/ 下线=False（停采集+面板置灰）",
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now(),
            comment="最后一次 toggle 时刻（审计）",
        ),
    )


def downgrade() -> None:
    op.drop_table("shadow_version_overrides")
