"""add strategy advices (LLM 策略建议 + 应用闭环 + 30 单追踪)

Revision ID: s7a8b9c0d1e2
Revises: c4d5e6f7a8b9
Create Date: 2026-10-22

2026-10-22「信号胜率趋势 + LLM 策略顾问」：L2 趋势预警/红灯/手动触发时，
LLM 输出结构化策略建议（动作白名单 + 单调收紧 + 服务端硬闸），落本表；
一键应用走服务端二次校验后调用既有 set_channel/persist_channel，并在此表
记录应用日志与 30 单效果追踪。本表为事后分析层，不被下单/结算决策路径引用。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "s7a8b9c0d1e2"
down_revision = "c4d5e6f7a8b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_advices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column("channel", sa.String(80), nullable=False),
        sa.Column("trigger_type", sa.String(16), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("advice_spec", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("rejected", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("pre_metrics", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(12), nullable=False),
        sa.Column("application_log", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("tracking", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_strategy_advices_channel_created", "strategy_advices",
                    ["channel", "created_at"])
    op.create_index("ix_strategy_advices_status", "strategy_advices", ["status"])


def downgrade() -> None:
    op.drop_index("ix_strategy_advices_status", table_name="strategy_advices")
    op.drop_index("ix_strategy_advices_channel_created", table_name="strategy_advices")
    op.drop_table("strategy_advices")
