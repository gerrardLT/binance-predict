"""narrow fake breakout to 4h-only A+B: backup + truncate

Revision ID: m3e4f5g6h7i8
Revises: l2d3e4f5g6h7
Create Date: 2026-08-15

模式收窄（2026-08-15 用户拍板）：只保留回测统计显著的 4h 双向 × A+B 过滤组合
（scripts/local_combo_filter_lab.py：4h 破阻力→DOWN 胜率 41.7% 费后 EV +7.06 [+0.35,+15.7]；
4h 破支撑→UP 53.3% +6.52 [+1.12,+13.9]，CI 均不含 0）。
1h / 日线级别与未过过滤的信号彻底移除（不落表不结算不邮件）。
本迁移备份并清空历史信号（含旧模式与过渡期数据），从 0 积累纯 4h A+B 样本。
"""
from __future__ import annotations

from alembic import op

revision = "m3e4f5g6h7i8"
down_revision = "l2d3e4f5g6h7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 清表重积累：先备份再清空（IF NOT EXISTS 保留首次备份，幂等可重跑）
    op.execute(
        "CREATE TABLE IF NOT EXISTS fake_breakout_signals_pre_4h_only_bak "
        "AS SELECT * FROM fake_breakout_signals"
    )
    op.execute("TRUNCATE TABLE fake_breakout_signals RESTART IDENTITY")


def downgrade() -> None:
    # 无结构变更可回退；数据恢复需手工从 fake_breakout_signals_pre_4h_only_bak INSERT SELECT
    pass
