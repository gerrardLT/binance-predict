"""P2-5：风控连续亏损（loss_streak）语义回归测试。

验证 RiskController.refresh_daily_stats 的连亏计数改为基于**实际交易盈亏**
（TradeOrderModel.amount_out < amount_in），而非方向命中（is_correct）：
- 方向对也可能因手续费/滑点亏钱 → 应计入 streak；
- NO_TRADE 未实际成交不产生订单 → 天然不计入；
- 未结算订单（amount_out 缺失/为 0）→ 跳过，不中断也不计入。

不导入 sentiment_agent/main（避免可选依赖 instructor），仅 patch
async_session_factory 注入伪 session。refresh_daily_stats 内 execute 顺序：
  ① count_stmt -> .scalar()
  ② pnl_stmt   -> .all()
  ③ order_stmt -> .all()（created_at 降序，最近在前）
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from binance_predict.services.risk_control import RiskController


class _FakeSessionCtx:
    """伪 `async with async_session_factory() as session:` 上下文。"""

    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _make_session(order_rows: list[tuple], *, count: int = 0,
                  pnl_rows: list[tuple] | None = None) -> MagicMock:
    """构造 execute 按序返回 [count.scalar, pnl.all, order.all] 的伪 session。"""
    count_result = MagicMock()
    count_result.scalar.return_value = count

    pnl_result = MagicMock()
    pnl_result.all.return_value = pnl_rows if pnl_rows is not None else []

    order_result = MagicMock()
    order_result.all.return_value = order_rows

    session = MagicMock()
    session.execute = AsyncMock(side_effect=[count_result, pnl_result, order_result])
    return session


async def _run(order_rows: list[tuple], *, count: int = 0,
               pnl_rows: list[tuple] | None = None) -> RiskController:
    session = _make_session(order_rows, count=count, pnl_rows=pnl_rows)
    factory = MagicMock(return_value=_FakeSessionCtx(session))
    with patch(
        "binance_predict.services.risk_control.async_session_factory", factory
    ):
        rc = RiskController()
        await rc.refresh_daily_stats(force=True)
    return rc


@pytest.mark.asyncio
async def test_three_losses_then_win_streak_is_three() -> None:
    # 最近在前：连亏 3 单（out<in）后遇 1 单盈利（out>in）中断
    rows = [(10.0, 8.0), (10.0, 9.0), (10.0, 7.0), (10.0, 12.0), (10.0, 5.0)]
    rc = await _run(rows, count=5)
    assert rc.recent_loss_streak == 3


@pytest.mark.asyncio
async def test_first_order_profit_streak_zero() -> None:
    # 最近一单盈利 → 立即中断，streak=0（即使更早有亏损）
    rows = [(10.0, 12.0), (10.0, 8.0), (10.0, 7.0)]
    rc = await _run(rows, count=3)
    assert rc.recent_loss_streak == 0


@pytest.mark.asyncio
async def test_no_orders_streak_zero() -> None:
    # 全 NO_TRADE / 无成交订单 → 无 order 行 → streak=0
    rc = await _run([], count=0)
    assert rc.recent_loss_streak == 0


@pytest.mark.asyncio
async def test_unsettled_order_skipped_not_breaking_streak() -> None:
    # 首行未结算（amount_out=0）应跳过，不中断也不计入；
    # 其后连亏 2 单，再遇盈利中断 → streak=2
    rows = [(10.0, 0.0), (10.0, 8.0), (10.0, 9.0), (10.0, 12.0)]
    rc = await _run(rows, count=4)
    assert rc.recent_loss_streak == 2


@pytest.mark.asyncio
async def test_break_even_counts_as_non_loss() -> None:
    # 打平（out == in）不算亏损 → 中断 streak
    rows = [(10.0, 10.0), (10.0, 8.0)]
    rc = await _run(rows, count=2)
    assert rc.recent_loss_streak == 0
