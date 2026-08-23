"""trade_orders 加 redeemed_at（获胜订单奖金领取标记）

Revision ID: l2e3f4g5h6i7
Revises: x2y3z4a5b6c7
Create Date: 2026-08-23 17:30:00.000000

背景：预测市场赢单的奖金以 outcome token 形式留在钱包，
需要调 POST /sapi/v1/w3w/wallet/prediction/batch-redeem 领取成 USDT。
本迁移为 trade_orders 加 redeemed_at 标记领取状态：
- NULL = 未领取（win 单的默认态）
- 非 NULL = 已领取（batch-redeem 成功后按 token_id 匹配标记）

存量数据全为 NULL（历史从未领取过），无需回填。
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "l2e3f4g5h6i7"
down_revision = "x2y3z4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trade_orders",
        sa.Column(
            "redeemed_at", sa.DateTime(timezone=True), nullable=True,
            comment="奖金领取时间（batch-redeem 成功后标记；NULL=未领取）",
        ),
    )


def downgrade() -> None:
    op.drop_column("trade_orders", "redeemed_at")
