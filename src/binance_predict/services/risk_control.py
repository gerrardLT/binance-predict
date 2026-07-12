"""
风控状态管理器 —— 日内统计查询

从 DB 查询交易记录，维护日内统计（交易次数、PnL、连续亏损），
供 evaluate_trade_gate 使用。

设计约束：
- 每次 refresh_daily_stats() 从 DB 重新加载当日数据，不依赖内存缓存
- 连续亏损基于 AgentPrediction.is_correct（方向命中率），
  统一语义，避免与交易 PnL 混淆（方向正确≠盈利，含手续费/滑点）
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func, select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import AgentPrediction, TradeOrderModel


class RiskController:
    """
    风控状态管理：从 DB 查询交易记录，维护日内统计。

    职责：提供日内统计数据（交易次数、PnL、连续亏损）
    供 evaluate_trade_gate 使用。
    """

    def __init__(self) -> None:
        self._daily_trade_count: int = 0
        self._daily_pnl: float = 0.0
        self._recent_loss_streak: int = 0
        # Fix #20: TTL 缓存，避免短时间内重复全量查询
        self._last_refresh_ts: float = 0.0

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def recent_loss_streak(self) -> int:
        return self._recent_loss_streak

    async def refresh_daily_stats(self, *, force: bool = False) -> None:
        """从 trade_orders + agent_predictions 查询当日交易统计。

        Fix #20: 优化为（a）TTL 缓存， min_interval 内复用上次结果；
        （b）交易计数改用 SQL COUNT 聚合，避免将当日全部订单加载入内存。

        Args:
            force: 为 True 时忽略 TTL 强制刷新。
        """
        min_interval = settings.risk_stats_cache_ttl_sec
        now_mono = time.monotonic()
        if not force and (now_mono - self._last_refresh_ts) < min_interval:
            return

        try:
            now = datetime.now(tz=timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

            async with async_session_factory() as session:
                # 当日交易数量：使用 SQL COUNT 聚合，不加载全部行
                count_stmt = select(func.count()).select_from(TradeOrderModel).where(
                    TradeOrderModel.created_at >= today_start
                )
                self._daily_trade_count = int(
                    (await session.execute(count_stmt)).scalar() or 0
                )

                # PnL 估算：仅拉取当日订单的 amount 字段（非全列）
                pnl_stmt = select(
                    TradeOrderModel.amount_in,
                    TradeOrderModel.amount_out,
                ).where(TradeOrderModel.created_at >= today_start)
                pnl_rows = (await session.execute(pnl_stmt)).all()
                pnl_sum = 0.0
                for amt_in_raw, amt_out_raw in pnl_rows:
                    try:
                        amt_in = float(amt_in_raw) if amt_in_raw else 0.0
                        amt_out = float(amt_out_raw) if amt_out_raw else 0.0
                        if amt_in > 0 and amt_out > 0:
                            pnl_sum += amt_out - amt_in
                    except (ValueError, TypeError):
                        pass
                self._daily_pnl = pnl_sum

                # 查询最近已验证的预测，计算连续亏损（基于 is_correct）
                pred_stmt = (
                    select(AgentPrediction.is_correct)
                    .where(AgentPrediction.is_correct.isnot(None))
                    .order_by(AgentPrediction.validated_at.desc())
                    .limit(50)
                )
                pred_result = await session.execute(pred_stmt)
                recent_flags = pred_result.scalars().all()

                streak = 0
                for is_correct in recent_flags:
                    if is_correct is False:
                        streak += 1
                    else:
                        break
                self._recent_loss_streak = streak

            self._last_refresh_ts = now_mono
            logger.debug(
                "RiskController: 日内统计刷新完成 | trades={} pnl={:.2f} loss_streak={}",
                self._daily_trade_count,
                self._daily_pnl,
                self._recent_loss_streak,
            )
        except Exception as exc:
            logger.error(
                "RiskController: 日内统计刷新失败 | error={} | 使用旧数据",
                exc,
            )
