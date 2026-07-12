"""drop path A legacy tables (K线 + 决策 LLM 链路已退役)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-09 16:10:00.000000

本迁移删除「路径A（K线特征 + 决策 LLM 链路）」遗留的 7 张表。路径A 已整体
退役，其数据结构不再被任何在用代码引用，故一次性 DROP。

破坏性说明：upgrade() 为破坏性删除，被删表中的历史数据不可恢复（用户已明确
确认）。downgrade() 仅按删除前的 models.py 精确列定义重建这 7 张表的「结构」，
不恢复任何数据。

upgrade 删除的 7 张表：
  predictions / prediction_results / feature_snapshots / custom_rules /
  rule_versions / prompt_versions / review_memories
downgrade 逆序重建的 7 张表：
  review_memories / prompt_versions / rule_versions / custom_rules /
  feature_snapshots / prediction_results / predictions
（这 7 张表之间无跨表外键，删除/重建顺序不敏感；每张表先处理索引再处理表。）
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：删除路径A 遗留的 7 张表（先删各自索引，再删表）。破坏性操作。"""

    # 1) predictions（3 个自定义索引，加 if_exists 容错）
    op.drop_index("ix_predictions_target_time", table_name="predictions", if_exists=True)
    op.drop_index("ix_predictions_prediction_time", table_name="predictions", if_exists=True)
    op.drop_index("ix_predictions_symbol", table_name="predictions", if_exists=True)
    op.drop_table("predictions", if_exists=True)

    # 2) prediction_results（1 个自定义索引）
    op.drop_index("ix_prediction_results_prediction_id", table_name="prediction_results", if_exists=True)
    op.drop_table("prediction_results", if_exists=True)

    # 3) feature_snapshots（无自定义索引）
    op.drop_table("feature_snapshots", if_exists=True)

    # 4) custom_rules（1 个自定义索引）
    op.drop_index("ix_custom_rules_enabled", table_name="custom_rules", if_exists=True)
    op.drop_table("custom_rules", if_exists=True)

    # 5) rule_versions（无自定义索引；rules_version 列带 unique 约束，随表一并删除）
    op.drop_table("rule_versions", if_exists=True)

    # 6) prompt_versions（无自定义索引；prompt_version 列带 unique 约束，随表一并删除）
    op.drop_table("prompt_versions", if_exists=True)

    # 7) review_memories（1 个自定义索引）
    op.drop_index("ix_review_memories_confirmed", table_name="review_memories", if_exists=True)
    op.drop_table("review_memories", if_exists=True)


def downgrade() -> None:
    """降级：仅重建 7 张表的结构（不恢复数据），顺序与 upgrade 相反。

    列定义严格对齐删除前的 src/binance_predict/db/models.py：
    JSONB 使用 postgresql.JSONB()；带时区时间使用 sa.DateTime(timezone=True)；
    created_at/updated_at 使用 server_default=sa.func.now()。
    """

    # 7) review_memories
    op.create_table(
        "review_memories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("error_type", sa.String(length=30), nullable=False),
        sa.Column("lesson", sa.Text(), nullable=False),
        sa.Column("rule_suggestions_json", postgresql.JSONB(), nullable=False),
        sa.Column("confirmed", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_review_memories_confirmed", "review_memories", ["confirmed"], unique=False)

    # 6) prompt_versions（prompt_version 唯一）
    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("prompt_content", sa.Text(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column("expected_improvement", sa.Text(), nullable=True),
        sa.Column("rollback_condition", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prompt_version"),
    )

    # 5) rule_versions（rules_version 唯一）
    op.create_table(
        "rule_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("rules_version", sa.String(length=50), nullable=False),
        sa.Column("rules_snapshot_json", postgresql.JSONB(), nullable=False),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("rules_version"),
    )

    # 4) custom_rules
    op.create_table(
        "custom_rules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("rule_type", sa.String(length=10), nullable=False, comment="HARD | SOFT"),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column("scope", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_custom_rules_enabled", "custom_rules", ["enabled"], unique=False)

    # 3) feature_snapshots
    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_json", postgresql.JSONB(), nullable=False),
        sa.Column("anomaly_flags_json", postgresql.JSONB(), nullable=False),
        sa.Column("data_quality_score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # 2) prediction_results
    op.create_table(
        "prediction_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=False),
        sa.Column("actual_return", sa.Float(), nullable=False),
        sa.Column("actual_label", sa.String(length=10), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=30), nullable=True),
        sa.Column("lesson", sa.Text(), nullable=True),
        sa.Column("rule_suggestions_json", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prediction_results_prediction_id", "prediction_results", ["prediction_id"], unique=False)

    # 1) predictions
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("prediction_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("final_prediction", sa.String(length=20), nullable=False),
        sa.Column("p_up", sa.Float(), nullable=False),
        sa.Column("p_down", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=10), nullable=False),
        sa.Column("market_state", sa.String(length=30), nullable=False),
        sa.Column("reason_json", postgresql.JSONB(), nullable=False),
        sa.Column("reverse_risk_json", postgresql.JSONB(), nullable=False),
        sa.Column("invalid_conditions_json", postgresql.JSONB(), nullable=False),
        sa.Column("applied_rule_ids_json", postgresql.JSONB(), nullable=False),
        sa.Column("rules_version", sa.String(length=50), nullable=False),
        sa.Column("prompt_version", sa.String(length=50), nullable=False),
        sa.Column("feature_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_symbol", "predictions", ["symbol"], unique=False)
    op.create_index("ix_predictions_prediction_time", "predictions", ["prediction_time"], unique=False)
    op.create_index("ix_predictions_target_time", "predictions", ["target_time"], unique=False)
