"""报价 edge 实盘执行器（quote_momentum_v1 LIVE）。

实时触发：5m 窗口内 DOWN 报价首次进入 [0.69, 0.75) 且 t∈[90,120)s → 真单押 DOWN。
规则复用 quote_edge_detector.QUOTE_EDGE_RULES 冻结口径（回测同源，勿改）；
影子轨继续归档记录（对照），实盘轨实时开火，二者互不干扰。

安全护栏：
1. 每窗至多一单：内存 fired 集合 + **先占位后下单**（CodeReview High#1：
   place_order 前先插 PENDING 行占住 (signal_version, window_start) 唯一键，
   重复窗口在花钱前被拒）；失败尝试（含护栏弃单）更新为 FAILED 仍占键，
   同一窗口不重试，与回测"首个命中点入场"语义一致。
2. 执行价护栏：报价 averagePrice > max_exec_price → 弃单；且下单滑点按护栏价
   动态收紧，成交价无法突破护栏（不追贵，保护回测 EV 口径）。
3. 日单量护栏：当日 FILLED 达上限停火（防极端行情密度暴涨散口）。
4. 不回灌：只盯活跃窗口，重启不补已过去的窗口（区别于影子 backscan）。
5. 下单任务 create_task 脱离采样循环，不阻塞 15s 采样；stop() 只取消辅助任务，
   在途下单任务等待完成（钱已出去的临界区绝不打断）。
6. signal_id 回填双通道：窗口结算后即时回填 + 周期自愈扫描（重启/影子延迟不丢对账）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import func as sa_func
from sqlalchemy import select as sa_select
from sqlalchemy import update as sa_update

from binance_predict.config.settings import settings
from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import MisalignmentSignal, TradeOrderModel

from .quote_edge_detector import QUOTE_EDGE_RULES

logger = logging.getLogger(__name__)

LIVE_VERSION = "quote_momentum_v1"   # 首个上实盘的信号（用户圈定，2026-08-20）
SIGNAL_BACKFILL_DELAY_MS = 180_000  # 窗口结束后 180s 回读影子信号（归档+结算已就绪）
HEAL_INTERVAL_S = 300.0             # signal_id 自愈扫描间隔（Low#4：重启/延迟不丢对账）
MAX_ORDER_AMOUNT_USDT = 50.0        # 单笔金额硬上限（Low#5：配置误写拒绝启动，不靠自律）


class QuoteEdgeLiveTrader:
    """报价 momentum 实盘执行器：采样循环喂价 → 区间命中 → 真单 DOWN。"""

    def __init__(self, trader) -> None:
        self._trader = trader  # BinancePredictionTrader（复用签名/报价/下单链路）
        t_lo, t_hi, q_lo, q_hi = QUOTE_EDGE_RULES[LIVE_VERSION]
        self._t_lo, self._t_hi, self._q_lo, self._q_hi = t_lo, t_hi, q_lo, q_hi
        self._amount = settings.quote_momentum_live_amount_usdt
        self._max_exec = settings.quote_momentum_live_max_exec_price
        self._max_daily = settings.quote_momentum_live_max_daily_orders
        self._fired: set[int] = set()          # 本进程已开火/尝试过的 window_start
        self._tasks: set[asyncio.Task] = set()  # 在途任务（下单/回填/自愈）
        self._fire_total = 0
        self._healed_total = 0
        self._stopped = False                   # stop 后拒绝派生新下单任务（High#1）
        self._running = False

    async def start(self) -> None:
        """启动 signal_id 自愈扫描（下单触发由 tracker 喂价，无需自循环）。"""
        if self._running:
            return
        self._running = True
        task = asyncio.create_task(self._heal_loop(), name="quote_edge_live_heal")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def check(self, window_start_ms: int, window_end_ms: int,
              ts_ms: int, down_price: float | None) -> bool:
        """每次 5m 采样调用一次；命中规则区间则派生下单任务，返回是否开火。"""
        if self._stopped:
            return False
        if not settings.quote_momentum_live_enabled:
            return False
        if down_price is None:
            return False
        if window_start_ms in self._fired:
            return False
        t_rel = (ts_ms - window_start_ms) / 1000.0
        if not (self._t_lo <= t_rel < self._t_hi):
            return False
        if not (self._q_lo <= float(down_price) < self._q_hi):
            return False
        # 命中：立即占位（内存），防同窗后续采样重复派生
        self._fired.add(window_start_ms)
        task = asyncio.create_task(
            self._fire(window_start_ms, window_end_ms, t_rel, float(down_price)),
            name=f"quote_edge_live_{window_start_ms}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    # ------------------------------------------------------------------
    # 下单主流程（护栏 → 唯一性预检 → 真单 → 回填调度）
    # ------------------------------------------------------------------

    async def _fire(self, window_start: int, window_end: int,
                    t_rel: float, down_price: float) -> None:
        win_label = datetime.fromtimestamp(window_start / 1000, tz=timezone.utc).strftime(
            "%m-%d %H:%M")
        try:
            # 日单量护栏（DB 口径，重启不清零）
            filled_today = await self._count_filled_today()
            if filled_today >= self._max_daily:
                logger.warning(
                    "报价 edge 实盘：日单量护栏停火 | 窗口 {} | 今日已成交 {} ≥ {}",
                    win_label, filled_today, self._max_daily)
                return

            # 重启防重：DB 已有本窗尝试记录（FILLED/FAILED 皆算）则跳过
            if await self._has_attempt(window_start):
                logger.info("报价 edge 实盘：本窗已有订单记录，跳过 | 窗口 {}", win_label)
                return

            logger.info(
                "报价 edge 实盘开火 | {} | LIVE 下单押 DOWN | t=+{:.0f}s q={:.3f}"
                " | 金额 {} USDT | 执行价上限 {}",
                win_label, t_rel, down_price, self._amount, self._max_exec)

            order = await self._trader.execute_signal_trade(
                prediction="DOWN",
                amount_usdt=self._amount,
                signal_version=LIVE_VERSION,
                window_start=window_start,
                max_exec_price=self._max_exec,
            )
            if order is None:
                # 同窗已有占位（重启/并发重复）或前置配置缺失，未花钱，正常路径
                logger.info("报价 edge 实盘：未产生订单（重复窗口或前置失败）| 窗口 {}", win_label)
                return

            self._fire_total += 1
            if order.status == "FILLED":
                logger.info(
                    "报价 edge 实盘成交 | {} | order_id={} | token={} | amount_in={}",
                    win_label, order.order_id, order.token_id, order.amount_in)
                self._schedule(self._backfill_signal_link(window_start), window_end)
            else:
                logger.warning(
                    "报价 edge 实盘未成交 | {} | status={} | {}",
                    win_label, order.status, order.error_message)
        except Exception as exc:
            logger.warning("报价 edge 实盘：下单任务异常 | 窗口 {} | {} | {}",
                           win_label, type(exc).__name__, exc)

    def _schedule(self, coro, window_end_ms: int) -> None:
        """延迟执行回填（窗口结束后等待归档+影子结算就绪）。"""
        delay_s = max(0.0, (window_end_ms + SIGNAL_BACKFILL_DELAY_MS - time.time() * 1000) / 1000.0)

        async def _delayed():
            await asyncio.sleep(delay_s)
            await coro

        task = asyncio.create_task(_delayed(), name="quote_edge_live_backfill")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _backfill_signal_link(self, window_start: int) -> None:
        """窗口结算后把订单关联回影子信号（实盘 vs 影子一一对账）。"""
        try:
            async with async_session_factory() as session:
                sig = (await session.execute(
                    sa_select(MisalignmentSignal.id).where(
                        MisalignmentSignal.version == LIVE_VERSION,
                        MisalignmentSignal.window_start == window_start,
                    )
                )).first()
                if sig is None:
                    logger.debug("报价 edge 实盘：影子信号未就绪，跳过回填 | 窗口 {}", window_start)
                    return
                await session.execute(
                    sa_update(TradeOrderModel).where(
                        TradeOrderModel.signal_version == LIVE_VERSION,
                        TradeOrderModel.window_start == window_start,
                        TradeOrderModel.signal_id.is_(None),
                    ).values(signal_id=sig[0])
                )
                await session.commit()
                logger.info("报价 edge 实盘：订单已关联影子信号 | 窗口 {} | signal_id={}",
                            window_start, sig[0])
        except Exception as exc:
            logger.warning("报价 edge 实盘：signal_id 回填失败 | 窗口 {} | {}", window_start, exc)

    # ------------------------------------------------------------------
    # DB 查询
    # ------------------------------------------------------------------

    async def _count_filled_today(self) -> int:
        # Low#3：date_trunc 取 PG 会话时区（容器内为 UTC），即"日"按 UTC 自然日计，
        # 北京时间 08:00 翻日；护栏语义自洽，对账时注明口径即可。
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(sa_func.count(TradeOrderModel.id)).where(
                    TradeOrderModel.signal_version == LIVE_VERSION,
                    TradeOrderModel.status == "FILLED",
                    TradeOrderModel.created_at >= sa_func.date_trunc("day", sa_func.now()),
                )
            )).scalar()
        return int(row or 0)

    async def _has_attempt(self, window_start: int) -> bool:
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(TradeOrderModel.id).where(
                    TradeOrderModel.signal_version == LIVE_VERSION,
                    TradeOrderModel.window_start == window_start,
                ).limit(1)
            )).first()
        return row is not None

    # ------------------------------------------------------------------
    # signal_id 自愈扫描（Low#4：即时回填只一次机会，重启/影子延迟会永久丢对账）
    # ------------------------------------------------------------------

    async def _heal_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(HEAL_INTERVAL_S)
                await self._heal_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("报价 edge 实盘：自愈扫描异常 | {}", exc)

    async def _heal_once(self) -> None:
        """为超过 10 分钟仍缺 signal_id 的订单重试回填（幂等，可重复执行）。"""
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
            async with async_session_factory() as session:
                orders = (await session.execute(
                    sa_select(TradeOrderModel.window_start).where(
                        TradeOrderModel.signal_version == LIVE_VERSION,
                        TradeOrderModel.signal_id.is_(None),
                        TradeOrderModel.created_at < cutoff,
                    ).order_by(TradeOrderModel.window_start.asc()).limit(20)
                )).scalars().all()
            for ws in orders:
                await self._backfill_signal_link(int(ws))
            if orders:
                self._healed_total += len(orders)
        except Exception as exc:
            logger.warning("报价 edge 实盘：自愈扫描查询失败 | {}", exc)

    # ------------------------------------------------------------------
    # 状态 / 生命周期
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "enabled": settings.quote_momentum_live_enabled,
            "version": LIVE_VERSION,
            "amount_usdt": self._amount,
            "max_exec_price": self._max_exec,
            "max_daily_orders": self._max_daily,
            "fired_windows": sorted(self._fired)[-10:],
            "fire_total": self._fire_total,
            "healed_total": self._healed_total,
            "pending_tasks": len(self._tasks),
        }

    async def stop(self) -> None:
        """停止：先拒新单，只取消辅助任务（回填/自愈）；在途下单任务不 cancel、
        等待完成——钱已出去的临界区被打断会造成"交易所成交、DB 无终态"（High#1）。"""
        self._stopped = True
        self._running = False
        for task in list(self._tasks):
            if task.get_name() in ("quote_edge_live_heal", "quote_edge_live_backfill"):
                task.cancel()
        for task in list(self._tasks):
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=45)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._tasks.clear()
        logger.info("报价 edge 实盘执行器已停止 | 开火次数 {}", self._fire_total)
