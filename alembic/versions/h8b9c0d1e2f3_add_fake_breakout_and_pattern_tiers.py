"""add 15m 采样周期 + 假突破信号表 + 模式池分级 + 模式回测快照表

Revision ID: h8b9c0d1e2f3
Revises: a8b9c0d1e2f3, g7a8b9c0d1e2
Create Date: 2026-08-13 12:00:00.000000

本迁移为「假突破信号系统 + 模式池分级无限进化」：

1. prediction_market_samples 新增 market_period 列（'5m'/'15m'）：
   采样器扩展为同时记录 5m 与 15m 两个市场的 UP/DOWN 报价，
   存量行回填 '5m'，语义不变。

2. 新增 fake_breakout_signals 表：日线阻力假突破信号记录。
   信号触发（秒级检测 BTC 盘中冲高破位）即落表，暂不下注；
   到期后由检测器回读 BTC 价格回填结算方向（UP/DOWN，只看符号）。

3. pattern_memory 新增 tier 列（S/A/B/C 模式池分级）：
   存量行默认 'C'，由定期重回测（pattern_backtest_runs）驱动晋级/降级。

4. 新增 pattern_backtest_runs 表：每个模式每一次回测的完整快照
   （胜率/CI/费后EV/分段细节/与前次对比差异），
   支撑前端横向（模式间）与纵向（同一模式随时间）对比展示。

手写脚本，字段/类型/nullable/comment 严格对齐
src/binance_predict/db/models.py（单一事实源，用户规则 7/8）。
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "h8b9c0d1e2f3"
down_revision = ("a8b9c0d1e2f3", "g7a8b9c0d1e2")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """升级：合并双头 + 四处变更。"""
    # --- 1. prediction_market_samples.market_period（5m/15m 双市场采样）---
    op.add_column(
        "prediction_market_samples",
        sa.Column(
            "market_period", sa.String(length=5), nullable=False,
            server_default="5m",
            comment="预测市场周期：5m | 15m（存量行回填 5m，语义不变）",
        ),
    )
    op.create_index(
        "ix_pm_samples_period_ts",
        "prediction_market_samples",
        ["market_period", "timestamp"],
        unique=False,
    )

    # --- 2. fake_breakout_signals（日线阻力假突破信号，暂不下注）---
    op.create_table(
        "fake_breakout_signals",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "signal_time", sa.BigInteger(), nullable=False,
            comment="破位检测时刻（ms，秒级检测循环触发）",
        ),
        sa.Column(
            "resistance", sa.Float(), nullable=False,
            comment="当时日线阻力位（前 288 个 5m 窗口 closes 的 max）",
        ),
        sa.Column(
            "btc_price", sa.Float(), nullable=False,
            comment="破位时刻 BTC 现货中间价",
        ),
        sa.Column(
            "eps", sa.Float(), nullable=False,
            comment="触发阈值快照（破位幅度，如 0.0005）",
        ),
        sa.Column(
            "down_price_5m", sa.Float(), nullable=True,
            comment="信号时刻 5m 市场 DOWN token 最近采样报价",
        ),
        sa.Column(
            "down_price_15m", sa.Float(), nullable=True,
            comment="信号时刻 15m 市场 DOWN token 最近采样报价",
        ),
        sa.Column(
            "market_end_15m", sa.BigInteger(), nullable=True,
            comment="当时 15m 市场 end_date（ms，即到期结算时刻）",
        ),
        sa.Column(
            "settle_deadline", sa.BigInteger(), nullable=False,
            comment="结算回读死线（ms）= signal_time + 15min + 缓冲",
        ),
        sa.Column(
            "settle_btc_price", sa.Float(), nullable=True,
            comment="结算时刻 BTC 现货中间价（到期回读回填）",
        ),
        sa.Column(
            "settle_outcome", sa.String(length=10), nullable=True,
            comment="结算方向 UP | DOWN（只看符号：settle_btc < btc_price → DOWN 赢）",
        ),
        sa.Column(
            "status", sa.String(length=10), nullable=False, server_default="PENDING",
            comment="PENDING | SETTLED | EXPIRED（数据缺失无法结算）",
        ),
        sa.Column(
            "email_sent", sa.Boolean(), nullable=False, server_default=sa.false(),
            comment="信号触发邮件是否已推送",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_fbs_signal_time", "fake_breakout_signals", ["signal_time"], unique=False
    )
    op.create_index(
        "ix_fbs_status", "fake_breakout_signals", ["status"], unique=False
    )

    # --- 3. pattern_memory.tier（S/A/B/C 模式池分级）---
    op.add_column(
        "pattern_memory",
        sa.Column(
            "tier", sa.String(length=2), nullable=False,
            server_default="C",
            comment="模式池分级 S | A | B | C（由定期重回测驱动晋级/降级）",
        ),
    )
    op.create_index(
        "ix_pattern_memory_tier", "pattern_memory", ["tier"], unique=False
    )

    # --- 4. pattern_backtest_runs（模式每次回测的完整快照）---
    op.create_table(
        "pattern_backtest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "pattern_id", sa.Integer(), nullable=False,
            comment="关联 pattern_memory.id（软关联，不加外键避免模式删除牵连历史）",
        ),
        sa.Column(
            "data_start", sa.BigInteger(), nullable=False,
            comment="本次回测数据范围起点（ms）",
        ),
        sa.Column(
            "data_end", sa.BigInteger(), nullable=False,
            comment="本次回测数据范围终点（ms）",
        ),
        sa.Column("sample_count", sa.Integer(), nullable=False, comment="命中样本数"),
        sa.Column("correct_count", sa.Integer(), nullable=False, comment="命中且方向正确数"),
        sa.Column("win_rate", sa.Float(), nullable=False, comment="胜率 0~1"),
        sa.Column(
            "wilson_lower", sa.Float(), nullable=True,
            comment="Wilson 95% 置信下界",
        ),
        sa.Column(
            "wilson_upper", sa.Float(), nullable=True,
            comment="Wilson 95% 置信上界",
        ),
        sa.Column(
            "ev_after_fee", sa.Float(), nullable=True,
            comment="费后 EV 估算（0.5 定价口径：(1-0.02)/0.51-1 ≈ +0.9216 / -1）",
        ),
        sa.Column(
            "segment_stats", JSONB(), nullable=True,
            comment="分段细节 JSON（按行情段/月段的胜率与样本数，供纵向对比）",
        ),
        sa.Column(
            "delta_vs_prev", JSONB(), nullable=True,
            comment="与上一次回测的细节对比差异（胜率漂移/新增样本分段表现等）",
        ),
        sa.Column(
            "trigger_reason", sa.String(length=20), nullable=False,
            comment="触发原因：SCHEDULED | DATA_THRESHOLD | MANUAL",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_pbr_pattern_id", "pattern_backtest_runs", ["pattern_id"], unique=False
    )
    op.create_index(
        "ix_pbr_created_at", "pattern_backtest_runs", ["created_at"], unique=False
    )


def downgrade() -> None:
    """降级：逆序删除全部变更（不拆分双头，双头各自独立 downgrade）。"""
    op.drop_index("ix_pbr_created_at", table_name="pattern_backtest_runs")
    op.drop_index("ix_pbr_pattern_id", table_name="pattern_backtest_runs")
    op.drop_table("pattern_backtest_runs")
    op.drop_index("ix_pattern_memory_tier", table_name="pattern_memory")
    op.drop_column("pattern_memory", "tier")
    op.drop_index("ix_fbs_status", table_name="fake_breakout_signals")
    op.drop_index("ix_fbs_signal_time", table_name="fake_breakout_signals")
    op.drop_table("fake_breakout_signals")
    op.drop_index("ix_pm_samples_period_ts", table_name="prediction_market_samples")
    op.drop_column("prediction_market_samples", "market_period")
