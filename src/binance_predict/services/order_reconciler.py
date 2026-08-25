"""订单自动对账器（R2，2026-08-25 风险评审）：回读币安订单历史订正卡 PENDING 行。

背景：下单响应超时/状态未知时本地行停留 PENDING，但钱可能已出去——
旧路径仅 CRITICAL 日志无人跟进；trade_settler 只结算 FILLED，PENDING
行永不结算且污染「在途持仓」视图。本服务周期性回读币安侧订单历史，
把能确认终态的 PENDING 行订正为 FILLED/FAILED。

匹配策略（保守优先，宁可不匹配也不错配）：
1. orderId 精确匹配：本地行已有 orderId（下单响应带回但状态未知）→
   币安历史同 orderId 即唯一事实，直接按币安 status 订正。
2. 窗口无歧义匹配：同 window_start 下本地 PENDING 与币安订单均只有
   一笔 → 一一对应可安全匹配（同窗多通道并行时此条件不成立，自动
   退化为不匹配——多通道错配正是 sync-binance 单键匹配的教训 R5）。
3. 均不满足 → 保持 PENDING 等人工（/api/trades/binance-history 对账）。

出清：超 abandon_after 仍无法确认的 PENDING 行标 FAILED（钱未出去
的概率压倒性高：FOK 要么立即成交要么拒绝，币安历史长期无此单即未
成交）；error_message 留痕可追溯。

结构克隆 trade_settler 范式（轮询 + per-row 独立事务 + 幂等守卫
WHERE status='PENDING' + 异常不中断）。只回读 + 订正状态，不发起任何
资金操作，零资金风险。
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import select as sa_select, update as sa_update

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import TradeOrderModel

logger = logging.getLogger(__name__)

POLL_INTERVAL = 60.0                      # 轮询间隔（秒）
RECON_DELAY = timedelta(minutes=2)        # 行龄门槛：刚下单的 PENDING 不扫（给响应回传留时间）
ABANDON_AFTER = timedelta(minutes=30)     # 超期无法确认 → FAILED 出清
HISTORY_LIMIT = 50                        # 单轮回读币安历史条数
SCAN_BATCH = 20                           # 单轮最多处理本地行数

# 币安侧终态映射：FILLED → 本地 FILLED；其余已知终态 → FAILED
_BN_FILLED = {"FILLED"}
_BN_FAILED = {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "FAILED", "CANCELLING"}


class OrderReconciler:
    """卡 PENDING 订单自动对账器：回读币安历史订正终态。"""

    def __init__(self, trader) -> None:
        """Args:
        trader: PredictionTrader 实例（复用其签名客户端调 query_order_history）。
        """
        self._trader = trader
        self._running = False
        self._task: asyncio.Task | None = None
        self._reconciled_count = 0
        self._last_poll_at: float | None = None
        self._last_result: dict = {}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="order_reconciler")
        logger.info(
            "订单对账器启动 | %ds 轮询 | 行龄门槛 %s | 超期出清 %s",
            int(POLL_INTERVAL), RECON_DELAY, ABANDON_AFTER,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("订单对账器已停止 | 累计订正 %d 单", self._reconciled_count)

    def status(self) -> dict:
        """供 /api/live/status 类端点观测对账器运行状态。"""
        return {
            "running": self._running,
            "reconciled_total": self._reconciled_count,
            "last_poll_at": self._last_poll_at,
            "last_result": self._last_result,
        }

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("订单对账器：循环异常 | %s | %s", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def poll_once(self) -> dict:
        """执行一轮对账，返回 {pending, matched, abandoned, skipped}。"""
        import time as _time
        self._last_poll_at = _time.time()
        summary = {"pending": 0, "matched": 0, "abandoned": 0, "skipped": 0}

        now_dt = datetime.now(timezone.utc)
        stmt = (
            sa_select(TradeOrderModel)
            .where(
                TradeOrderModel.status == "PENDING",
                TradeOrderModel.created_at < now_dt - RECON_DELAY,
            )
            .order_by(TradeOrderModel.created_at.asc())
            .limit(SCAN_BATCH)
        )
        async with async_session_factory() as session:
            rows = list((await session.execute(stmt)).scalars().all())
        summary["pending"] = len(rows)
        if not rows:
            self._last_result = summary
            return summary

        history = await self._trader.query_order_history(limit=HISTORY_LIMIT)
        if history is None:
            # API 不可达（限频/网络）：本轮跳过，下轮重试——不错判
            logger.warning(
                "订单对账器：币安历史查询失败（本轮跳过）| %s",
                self._trader.last_api_error)
            summary["skipped"] = len(rows)
            self._last_result = summary
            return summary

        bn_index = self._index_history(history)
        for row in rows:
            try:
                outcome = await self._reconcile_row(row, bn_index, now_dt)
                summary[outcome] += 1
            except Exception as exc:
                logger.warning(
                    "订单对账器：单行对账失败（下轮重试）| id=%s | %s", row.id, exc)
                summary["skipped"] += 1
        if summary["matched"] or summary["abandoned"]:
            logger.info(
                "订单对账器：本轮订正 %d 单（FILLED/FAILED）| 出清 %d | 待确认 %d",
                summary["matched"], summary["abandoned"], summary["pending"])
        self._reconciled_count += summary["matched"] + summary["abandoned"]
        self._last_result = summary
        return summary

    # ------------------------------------------------------------------
    # 单行对账
    # ------------------------------------------------------------------

    @staticmethod
    def _index_history(history: list) -> dict:
        """币安历史建双索引：orderId → 订单；window_start(ms) → [订单]。"""
        by_order_id: dict = {}
        by_window: dict[int, list] = {}
        for o in history:
            if not isinstance(o, dict):
                continue
            oid = o.get("orderId")
            if oid:
                by_order_id[str(oid)] = o
            m = re.search(r"-(\d{10})$", o.get("slug") or "")
            if m:
                by_window.setdefault(int(m.group(1)) * 1000, []).append(o)
        return {"by_order_id": by_order_id, "by_window": by_window}

    async def _reconcile_row(self, row: TradeOrderModel, bn_index: dict,
                             now_dt: datetime) -> str:
        """返回本轮结果：matched / abandoned / skipped。"""
        # 策略 1：orderId 精确匹配（最高置信）
        bn = bn_index["by_order_id"].get(str(row.order_id)) if row.order_id else None

        # 策略 2：窗口无歧义匹配（同窗本地 PENDING 与币安订单均唯一）
        if bn is None and row.window_start is not None:
            candidates = bn_index["by_window"].get(row.window_start) or []
            if len(candidates) == 1:
                sibling = await self._count_pending_in_window(row)
                if sibling == 1:
                    bn = candidates[0]

        if bn is not None:
            filled = str(bn.get("status") or "") in _BN_FILLED
            await self._apply_terminal(row, bn, filled)
            return "matched"

        # 无法确认：超期出清 FAILED（FOK 语义下币安长期无此单 ≈ 未成交）
        created = self._aware(row.created_at)
        if created is not None and now_dt - created > ABANDON_AFTER:
            await self._apply_abandon(row)
            return "abandoned"
        return "skipped"

    async def _count_pending_in_window(self, row: TradeOrderModel) -> int:
        """同 window_start 的本地 PENDING 行数（含本行）。"""
        from sqlalchemy import func
        stmt = (
            sa_select(func.count(TradeOrderModel.id))
            .where(TradeOrderModel.status == "PENDING")
            .where(TradeOrderModel.window_start == row.window_start)
        )
        async with async_session_factory() as session:
            return (await session.execute(stmt)).scalar_one() or 0

    async def _apply_terminal(self, row: TradeOrderModel, bn: dict, filled: bool) -> None:
        """按币安订单终态订正本地行；幂等守卫 WHERE status='PENDING'。"""
        values: dict = {
            "status": "FILLED" if filled else "FAILED",
            "error_message": None if filled else f"对账订正：币安侧状态 {bn.get('status')}",
        }
        if bn.get("orderId"):
            values["order_id"] = str(bn["orderId"])
        if filled:
            try:
                filled_usdt = float(bn.get("filledUsdtAmount") or 0)
                if filled_usdt > 0:
                    values["amount_in"] = str(int(filled_usdt * (10 ** 18)))
                values["quote_json"] = {
                    **(row.quote_json or {}),
                    "averagePrice": float(bn.get("price") or 0),
                    "filledShareQty": bn.get("filledShareQty"),
                    "source": "order_reconciler",
                }
            except (TypeError, ValueError):
                pass
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.status == "PENDING",
                )
                .values(**values)
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount:
            logger.info(
                "订单对账 | id=%s | window=%s | → %s（币安 orderId=%s status=%s）",
                row.id, row.window_start, values["status"],
                bn.get("orderId"), bn.get("status"))
            # 资金事实确认：FILLED 表示钱已出去——CRITICAL 留痕供日志健康检查
            if filled:
                logger.critical(
                    "订单对账确认成交（曾卡 PENDING）| id=%s | window=%s | "
                    "orderId=%s —— 请核对日限与结算链",
                    row.id, row.window_start, bn.get("orderId"))

    async def _apply_abandon(self, row: TradeOrderModel) -> None:
        """超期无法确认 → FAILED 出清（幂等守卫同上）。"""
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.status == "PENDING",
                )
                .values(
                    status="FAILED",
                    error_message=(
                        f"对账出清：超 {int(ABANDON_AFTER.total_seconds() // 60)}min "
                        f"币安历史无对应订单，FOK 判定未成交"),
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount:
            logger.warning(
                "订单对账出清 | id=%s | window=%s | 币安历史长期无此单 → FAILED",
                row.id, row.window_start)

    @staticmethod
    def _aware(dt: datetime | None) -> datetime | None:
        """naive datetime（测试桩/驱动差异）按 UTC 补 tzinfo。"""
        if dt is None:
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
