"""add health_snapshots table

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-13 11:00:00.000000

本迁移新增一张 Agent 运行健康快照表 health_snapshots，供监控系统后台轮询
按周期落库（趋势回看 / LLM 诊断）。手写脚本，字段/类型/nullable/server_default/
comment 均严格对齐 src/binance_predict/db/models.py 的 HealthSnapshot
（单一事实源，用户规则 7/8）。

该表与其他表无外键关联（独立监控表），写入失败不影响监控主循环与决策流程。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：创建 health_snapshots 表 + created_at / overall_status 索引。"""
    op.create_table(
        "health_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("overall_status", sa.String(length=10), nullable=False, comment="OK | WARN | CRITICAL"),
        sa.Column("alert_count", sa.Integer(), server_default="0", nullable=False, comment="当次告警条数"),
        sa.Column("report", postgresql.JSONB(), nullable=False, comment="HealthReport 完整 JSON"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_health_snapshots_created_at", "health_snapshots", ["created_at"], unique=False)
    op.create_index("ix_health_snapshots_status", "health_snapshots", ["overall_status"], unique=False)


def downgrade() -> None:
    """降级：删除 health_snapshots 表（先删索引再删表）。"""
    op.drop_index("ix_health_snapshots_status", table_name="health_snapshots")
    op.drop_index("ix_health_snapshots_created_at", table_name="health_snapshots")
    op.drop_table("health_snapshots")
