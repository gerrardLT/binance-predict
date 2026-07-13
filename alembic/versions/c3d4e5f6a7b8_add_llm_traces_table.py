"""add llm_traces audit table

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-13 10:00:00.000000

本迁移新增一张 LLM 调用轨迹审计表 llm_traces，用于前端「LLM 轨迹」面板与
人工流程审查。手写脚本，字段/类型/nullable/server_default/comment 均严格对齐
src/binance_predict/db/models.py 的 LLMTrace（单一事实源，用户规则 7/8）。

该表与其他表无外键关联（独立审计表），写入为 fire-and-forget，
失败不影响主决策流程。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：创建 llm_traces 表 + created_at / phase 索引。"""
    op.create_table(
        "llm_traces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("phase", sa.String(length=20), nullable=False, comment="LEARN | DEEP_LEARN | PREDICT | EVOLVE"),
        sa.Column("model", sa.String(length=60), nullable=False, comment="调用的模型名"),
        sa.Column("system_prompt", sa.Text(), nullable=False, comment="完整系统提示词"),
        sa.Column("user_message", sa.Text(), nullable=False, comment="完整用户输入（含曲线/模式库上下文）"),
        sa.Column("assistant_output", postgresql.JSONB(), nullable=True, comment="LLM 结构化输出完整 JSON（含 reasoning 与结论）"),
        sa.Column("reasoning", sa.Text(), nullable=True, comment="LLM 推理文本（从 assistant_output.reasoning 抽取，便于列表展示）"),
        sa.Column("result_summary", sa.String(length=200), nullable=True, comment="关键结论摘要（如 direction=UP conf=0.72 / discoveries=3）"),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True, comment="输入 token（真实或估算）"),
        sa.Column("completion_tokens", sa.Integer(), nullable=True, comment="输出 token"),
        sa.Column("estimated_cost_yuan", sa.Float(), nullable=True, comment="估算成本（元）"),
        sa.Column("latency_s", sa.Float(), nullable=True, comment="LLM 调用耗时（秒）"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_traces_created_at", "llm_traces", ["created_at"], unique=False)
    op.create_index("ix_llm_traces_phase", "llm_traces", ["phase"], unique=False)


def downgrade() -> None:
    """降级：删除 llm_traces 表（先删索引再删表）。"""
    op.drop_index("ix_llm_traces_phase", table_name="llm_traces")
    op.drop_index("ix_llm_traces_created_at", table_name="llm_traces")
    op.drop_table("llm_traces")
