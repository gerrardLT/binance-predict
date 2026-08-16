"""retire system B tables (archive) + add scene_param_versions (v1 seed)

Revision ID: o5f6g7h8i9j0
Revises: n4e5f6g7h8i9
Create Date: 2026-08-16

系统B（情绪 Agent Loop）退役拍板（2026-08-16）：
- 7 张系统B专属表只加 DEPRECATED 注释做存档隔离，不改任何行数据
  （llm_traces 表本体继续服务新模块的 LLM 审计，注释说明历史归属）
- 新建 scene_param_versions 场景参数版本表（LLM 自进化体系载体），
  并 INSERT 一条 ACTIVE v1 = 现行线上常量（0.85/2.0/20/0.0005/48），
  场景参数自此有版本化起点；多重检验预算基数 = 本表累计行数。
- sentiment_windows / prediction_market_samples / fake_breakout_signals
  为共享数据源，绝不触碰。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "o5f6g7h8i9j0"
down_revision = "n4e5f6g7h8i9"
branch_labels = None
depends_on = None

# 系统B专属表：只读存档隔离
_DEPRECATED_TABLES = [
    "pattern_memory",
    "agent_predictions",
    "pattern_change_log",
    "binning_snapshots",
    "pattern_backtest_runs",
    "health_snapshots",
]
_DEP_COMMENT = "DEPRECATED 2026-08: 系统B已退役，只读存档"
_LLMT_COMMENT = (
    "2026-08: 系统B退役；历史行为系统B存档，表本体继续服务新模块的 LLM 调用审计"
    "（如 SCENE_RESEARCH 阶段）"
)

_V1_PARAMS = {
    "close_pos_min": 0.85,
    "vol_ratio_min": 2.0,
    "vol_ma_window": 20,
    "eps": 0.0005,
    "level_lookbacks": {"4h": 48},
}


def upgrade() -> None:
    for tbl in _DEPRECATED_TABLES:
        op.execute(f"COMMENT ON TABLE {tbl} IS '{_DEP_COMMENT}'")
    op.execute(f"COMMENT ON TABLE llm_traces IS '{_LLMT_COMMENT}'")

    op.create_table(
        "scene_param_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "version", sa.String(length=40), nullable=False,
            comment="版本号，如 v1-20260816（人工可读，全局唯一）",
        ),
        sa.Column(
            "params", JSONB(), nullable=False,
            comment="场景参数集：{close_pos_min, vol_ratio_min, vol_ma_window, eps, level_lookbacks}",
        ),
        sa.Column(
            "status", sa.String(length=16), nullable=False,
            server_default="PENDING_REVIEW",
            comment="PENDING_REVIEW | REJECTED | SHADOW | ACTIVE | RETIRED",
        ),
        sa.Column(
            "backtest_report", JSONB(), nullable=True,
            comment="过闸时科学回测引擎的完整输出快照（四层检验结果）",
        ),
        sa.Column(
            "proposed_by", sa.String(length=60), nullable=False, server_default="human",
            comment="提议者：llm-researcher | human",
        ),
        sa.Column(
            "reviewed_by", sa.String(length=60), nullable=True,
            comment="放行人（promote API 调用时回填）",
        ),
        sa.Column("review_note", sa.Text(), nullable=True, comment="审批备注（驳回理由 / 放行理由）"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "activated_at", sa.DateTime(timezone=True), nullable=True,
            comment="升为 ACTIVE 的时刻（人工 promote）",
        ),
        sa.Column(
            "retired_at", sa.DateTime(timezone=True), nullable=True,
            comment="退为 RETIRED 的时刻（被新版本接替或人工回退）",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version", name="uq_spv_version"),
    )
    op.create_index("ix_spv_status", "scene_param_versions", ["status"])

    # v1 种子：现行线上常量的版本化起点（180 天发现集→验证集盲验口径的参数）
    op.execute(
        "INSERT INTO scene_param_versions (version, params, status, proposed_by, activated_at) "
        f"VALUES ('v1-20260816', '{_json_literal()}'::jsonb, 'ACTIVE', 'human', now())"
    )


def _json_literal() -> str:
    import json

    return json.dumps(_V1_PARAMS, ensure_ascii=False)


def downgrade() -> None:
    op.drop_index("ix_spv_status", table_name="scene_param_versions")
    op.drop_table("scene_param_versions")
    for tbl in _DEPRECATED_TABLES:
        op.execute(f"COMMENT ON TABLE {tbl} IS ''")
    op.execute("COMMENT ON TABLE llm_traces IS ''")
