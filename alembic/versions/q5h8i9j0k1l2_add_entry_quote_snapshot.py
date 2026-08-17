"""入场报价快照列（2026-08-17）：次周期开盘后延迟抓取真实 15m 市场报价。

背景：此前信号只落 fire 时刻报价（即将结算的旧市场残值 0.01~0.99），
入场 EV 只能按理论 @0.50 假设计算。实测（已记录数据 z 曲面）显示开盘
瞬间市场定价中位 DOWN=0.615/UP=0.385——盈亏平衡胜率 p*=(e+0.01)/0.98
随报价漂移（@0.615 需 63.8% vs @0.50 需 52%），胜率优势可能被报价吃掉。

新增 6 列：
- entry_down/up_price_15m + entry_quote_ts_15m：开盘后 ~8s 快照（15m 边界加速采样 2s 粒度，市场切换确认守卫）
- add_down/up_price_15m + add_trigger_ts_15m：场景①反弹加仓（mid≥开盘×1.001）触发快照

Revision ID: q5h8i9j0k1l2
Revises: p4g7h8i9j0k1
Create Date: 2026-08-17
"""
from alembic import op
import sqlalchemy as sa

revision = "q5h8i9j0k1l2"
down_revision = "p4g7h8i9j0k1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("fake_breakout_signals") as batch:
        batch.add_column(sa.Column(
            "entry_down_price_15m", sa.Float(), nullable=True,
            comment="入场报价快照：次周期开盘后~8s 的 15m 市场 DOWN token 价（市场切换确认后落）"))
        batch.add_column(sa.Column(
            "entry_up_price_15m", sa.Float(), nullable=True,
            comment="入场报价快照：次周期开盘后~8s 的 15m 市场 UP token 价"))
        batch.add_column(sa.Column(
            "entry_quote_ts_15m", sa.BigInteger(), nullable=True,
            comment="入场报价快照时刻（ms，距开盘 offset 可由此计算）"))
        batch.add_column(sa.Column(
            "add_down_price_15m", sa.Float(), nullable=True,
            comment="场景①加仓触发时（mid≥开盘×1.001）的 15m 市场 DOWN token 报价；NULL=未触发/未监测"))
        batch.add_column(sa.Column(
            "add_up_price_15m", sa.Float(), nullable=True,
            comment="场景①加仓触发时的 15m 市场 UP token 报价"))
        batch.add_column(sa.Column(
            "add_trigger_ts_15m", sa.BigInteger(), nullable=True,
            comment="场景①反弹加仓触发时刻（ms）；NULL=周期内未触发"))


def downgrade() -> None:
    with op.batch_alter_table("fake_breakout_signals") as batch:
        batch.drop_column("add_trigger_ts_15m")
        batch.drop_column("add_up_price_15m")
        batch.drop_column("add_down_price_15m")
        batch.drop_column("entry_quote_ts_15m")
        batch.drop_column("entry_up_price_15m")
        batch.drop_column("entry_down_price_15m")
