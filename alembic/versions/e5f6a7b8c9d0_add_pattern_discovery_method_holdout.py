"""add pattern_memory discovery_method + holdout stats (Deep Learn 双轨 A/B 对比)

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-13 12:00:00.000000

本迁移为「Deep Learn 整改」在 pattern_memory 表新增 4 列，用于区分模式的发现
方法（纯 LLM 深度发现 / Python 确定性聚类 / 存量）并存储发现时的样本外(holdout)
统计，供双轨 A/B 对比。所有新列均可空/带默认，存量行不受影响。手写脚本，
字段/类型/nullable/server_default/comment 严格对齐 src/binance_predict/db/models.py
的 PatternMemory（单一事实源，用户规则 7/8）。

新增列：
  discovery_method     VARCHAR(20) NOT NULL DEFAULT 'LEGACY'  发现方法标签
  holdout_win_rate     FLOAT NULL                             发现时 holdout 胜率
  holdout_sample_count INTEGER NULL                           发现时 holdout 样本数
  holdout_ci_lower     FLOAT NULL                             发现时 holdout Wilson 下界
新增索引：
  ix_pattern_memory_discovery_method 便于按发现方法聚合 live 指标
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：为 pattern_memory 追加 discovery_method 与 holdout 统计列 + 索引。

    discovery_method 带 server_default='LEGACY'，存量行自动回填为 LEGACY；
    holdout_* 三列可空，仅 Deep Learn 双轨发现时回填。"""
    op.add_column(
        "pattern_memory",
        sa.Column(
            "discovery_method",
            sa.String(length=20),
            nullable=False,
            server_default="LEGACY",
            comment="发现方法：LLM_DEEP | PY_CLUSTER | LEGACY",
        ),
    )
    op.add_column(
        "pattern_memory",
        sa.Column("holdout_win_rate", sa.Float(), nullable=True, comment="发现时 holdout 胜率 0~1"),
    )
    op.add_column(
        "pattern_memory",
        sa.Column(
            "holdout_sample_count",
            sa.Integer(),
            nullable=True,
            comment="发现时 holdout 命中判定样本数",
        ),
    )
    op.add_column(
        "pattern_memory",
        sa.Column(
            "holdout_ci_lower",
            sa.Float(),
            nullable=True,
            comment="发现时 holdout 胜率 Wilson 95% 置信下界",
        ),
    )
    op.create_index(
        "ix_pattern_memory_discovery_method", "pattern_memory", ["discovery_method"], unique=False
    )


def downgrade() -> None:
    """降级：逆序删除索引与 4 列。"""
    op.drop_index("ix_pattern_memory_discovery_method", table_name="pattern_memory")
    op.drop_column("pattern_memory", "holdout_ci_lower")
    op.drop_column("pattern_memory", "holdout_sample_count")
    op.drop_column("pattern_memory", "holdout_win_rate")
    op.drop_column("pattern_memory", "discovery_method")
