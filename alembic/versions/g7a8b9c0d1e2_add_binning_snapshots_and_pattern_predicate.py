"""add binning_snapshots table + pattern_memory 科学发现四列

Revision ID: g7a8b9c0d1e2
Revises: f6a7b8c9d0e1
Create Date: 2026-08-10 12:00:00.000000

本迁移为「科学发现系统 Phase 2」（.kiro/specs/scientific-discovery/design.md）：
1. 新增 binning_snapshots 表——Q4 分箱规则修订为每通道独立分位边界
   （三通道量纲不同，共用边界会让小量纲通道全部落入"平"档），
   每 30 天按通道各冻结一版 20/40/60/80 分位边界。
2. pattern_memory 新增 4 列（科学发现轨），全部可空、存量行不受影响：
   predicate        JSONB NULL        谓词 DSL JSON（Q5，程序可确定性执行）
   binning_version  VARCHAR(40) NULL  模式"出生"时的分箱快照版本（Q4）
   death_cause      VARCHAR(10) NULL  双轨死因 SPURIOUS | EXPIRED（Q7-1）
   lifespan_days    FLOAT NULL        存活天数（RETIRE 时回填）

手写脚本，字段/类型/nullable/comment 严格对齐
src/binance_predict/db/models.py（单一事实源，用户规则 7/8）。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "g7a8b9c0d1e2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：建 binning_snapshots 表 + pattern_memory 追加科学发现四列。"""
    # --- binning_snapshots（Q4 每通道独立分箱冻结快照）---
    op.create_table(
        "binning_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=40), nullable=False,
            comment="快照版本号（同一版本覆盖三通道各一行）",
        ),
        sa.Column(
            "channel", sa.String(length=20), nullable=False,
            comment="sentiment | price | volume",
        ),
        sa.Column("edges", JSONB(), nullable=False, comment="[q20, q40, q60, q80] 分位边界"),
        sa.Column(
            "sample_count", sa.Integer(), nullable=False,
            comment="计算边界时的差值样本数",
        ),
        sa.Column(
            "created_at_epoch", sa.Float(), nullable=False,
            comment="冻结时刻 epoch 秒（与 symbolizer 对齐）",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", "channel", name="uq_binning_version_channel"),
    )
    op.create_index(
        "ix_binning_snapshots_channel", "binning_snapshots", ["channel"], unique=False
    )

    # --- pattern_memory 科学发现四列（全部可空，存量行不受影响）---
    op.add_column(
        "pattern_memory",
        sa.Column("predicate", JSONB(), nullable=True, comment="谓词 DSL JSON（科学发现轨，Q5）"),
    )
    op.add_column(
        "pattern_memory",
        sa.Column(
            "binning_version", sa.String(length=40), nullable=True,
            comment="发现时的分箱快照版本（Q4）",
        ),
    )
    op.add_column(
        "pattern_memory",
        sa.Column(
            "death_cause", sa.String(length=10), nullable=True,
            comment="死因：SPURIOUS | EXPIRED（Q7-1）",
        ),
    )
    op.add_column(
        "pattern_memory",
        sa.Column(
            "lifespan_days", sa.Float(), nullable=True,
            comment="存活天数（RETIRE 时回填，供存活期分布反馈）",
        ),
    )


def downgrade() -> None:
    """降级：逆序删除 pattern_memory 四列与 binning_snapshots 表。"""
    op.drop_column("pattern_memory", "lifespan_days")
    op.drop_column("pattern_memory", "death_cause")
    op.drop_column("pattern_memory", "binning_version")
    op.drop_column("pattern_memory", "predicate")
    op.drop_index("ix_binning_snapshots_channel", table_name="binning_snapshots")
    op.drop_table("binning_snapshots")
