"""交易结算器（P0-2）：回读结算源回填 FILLED 订单的输赢/盈亏。

结算源分流（两口径，_settle_row 入口硬编码，防错配）：
1. 5m 订单（默认）：回读 SentimentWindow——每 5 分钟归档器按 entry/exit
   价判定 outcome（UP/DOWN/NOISE），start_time 与订单 window_start 同为
   5m 窗口起点 ms，可直接对齐。
2. 15m 场景订单（market_period='15m'）：回读 FakeBreakoutSignal
   .settle_outcome（周期锚点口径：P(E) vs P(S)，与币安 15m 市场真实
   结算一致）。**为何必须分流**：15m 周期起点与 5m 窗口起点数值重合
   （900s 网格 ⊂ 300s 网格），若走 SentimentWindow 会被同名 5m 窗
   错口径结算输赢——这是多通道改造最隐蔽的坑。

扫描锚点 = trade_orders.settled_at IS NULL（部分索引
ix_trade_orders_settle_pending 只覆盖待结算行，空转亚毫秒）。

结算规则：
    outcome ∈ {UP, DOWN}：win = (direction == outcome)；
        pnl = amount/avg_price − amount（赢）/ −amount（输），
        amount = amount_in/1e18，avg_price = quote_json.averagePrice。
    outcome == NOISE：win=None, pnl=0.0（return=0 极罕见 16/3522，
        真实赔付规则未知，保守记 0；settled_at 锚点天然终止重扫）。
    无匹配窗口（归档缺失/未归档/outcome NULL）且 created_at 超 24h：
        settle_outcome=EXPIRED, win=None, pnl=0.0（防无限重扫；
        归档器恢复后可手动重算）。
    无匹配窗口且未超 24h：跳过，下轮重试。

pnl 口径注释：本地估算（成交均价口径），币安侧验证阶段校正——
不用 amount_out（落库的是 quote 预估值非成交实际值）。

结构克隆 quote_edge_detector 范式（60s 轮询 + per-row 独立
commit/rollback + 异常不中断）；无 backscan——锚点是 settled_at
IS NULL，启动首次 poll_once 即全量回补历史未结算行，无需水位。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select as sa_select, update as sa_update

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import FakeBreakoutSignal, SentimentWindow, TradeOrderModel

logger = logging.getLogger(__name__)

POLL_INTERVAL = 60.0                    # 轮询间隔（秒）
SETTLE_DELAY = timedelta(minutes=7)     # 等 5m 窗口归档（boundary+15s）+ 冗余
EXPIRE_AFTER = timedelta(hours=24)      # 无归档超时 → EXPIRED
SCAN_BATCH = 50                         # 单轮最多结算行数


class TradeSettler:
    """FILLED 未结算订单结算器：回读 SentimentWindow 回填输赢/盈亏。

    只读窗口 + 回填结算字段，零资金风险（lifespan 无开关常开）。
    """

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._settled_count = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="trade_settler")
        logger.info(
            "交易结算器启动 | %ds 轮询 | 锚点 settled_at IS NULL | 延迟 %s 等归档",
            int(POLL_INTERVAL), SETTLE_DELAY,
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
        logger.info("交易结算器已停止 | 累计结算 %d 单", self._settled_count)

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
                logger.warning("交易结算器：循环异常 | %s | %s", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def poll_once(self) -> int:
        """扫描并结算一批未结算订单，返回本轮结算条数（settle-scan 端点复用）。"""
        cutoff = datetime.now(timezone.utc) - SETTLE_DELAY
        stmt = (
            sa_select(TradeOrderModel)
            .where(
                TradeOrderModel.status == "FILLED",
                TradeOrderModel.settled_at.is_(None),
                TradeOrderModel.window_start.isnot(None),
                TradeOrderModel.created_at < cutoff,
            )
            .order_by(TradeOrderModel.window_start.asc())
            .limit(SCAN_BATCH)
        )
        async with async_session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        settled = 0
        for row in rows:
            try:
                if await self._settle_row(row):
                    settled += 1
            except Exception as exc:
                logger.warning(
                    "交易结算器：单行结算失败（下轮重试）| id=%s | window=%s | %s",
                    row.id, row.window_start, exc)
        return settled

    # ------------------------------------------------------------------
    # 单行结算
    # ------------------------------------------------------------------

    async def _settle_row(self, row: TradeOrderModel) -> bool:
        # 口径分流：15m 场景订单走 FakeBreakoutSignal 结算（防错配，见模块 docstring）
        if row.market_period == "15m" or row.scene_signal_id is not None:
            return await self._settle_scene_row(row)

        if row.direction is None:
            # 旧数据（direction 落库前的订单）：无法判赢，且字段无回填机制
            #（direction 与下单同事务写入）——等待毫无意义，被扫出（超 7min
            # 延迟）即 EXPIRED 出清（win=None/pnl=0 不计统计）。防永久卡
            # 「在途持仓」+ 每 60s 空扫（2026-08-23 生产实锤：4 笔 8/22 旧单
            # direction=NULL 卡在途 3.85 USDT，每轮空扫告警刷屏；首版等 24h
            # 反而让旧单多挂一天，收敛为立即出清）。
            now_dt = datetime.now(timezone.utc)
            async with async_session_factory() as session:
                stmt = (
                    sa_update(TradeOrderModel)
                    .where(
                        TradeOrderModel.id == row.id,
                        TradeOrderModel.settled_at.is_(None),
                    )
                    .values(
                        settle_outcome="EXPIRED",
                        win=None,
                        settle_price=None,
                        pnl=0.0,
                        settled_at=now_dt,
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
            if result.rowcount == 0:
                return False
            self._settled_count += 1
            logger.info(
                "订单结算 | id=%s | direction=NULL（旧数据）→ EXPIRED 出清 | window=%s",
                row.id, row.window_start)
            return True

        now_dt = datetime.now(timezone.utc)
        window = await self._find_window(row.window_start)
        outcome = window.outcome if window is not None else None

        if outcome in ("UP", "DOWN"):
            win = row.direction == outcome
            amount = self._amount_usdt(row)
            avg_price = self._avg_price(row)
            settle_price = self._to_float(window.exit_price)
            if win and amount is not None and avg_price is not None:
                pnl = amount / avg_price - amount
            elif not win and amount is not None:
                pnl = -amount
            else:
                pnl = None  # 均价/金额缺失：输向无需均价，赢向无法估算
        elif outcome == "NOISE":
            win = None
            settle_price = self._to_float(window.exit_price)
            pnl = 0.0
        else:
            # 窗口未归档 / outcome NULL（归档缺结算价）：超 24h 兜底 EXPIRED，否则重试
            created = self._aware(row.created_at)
            if created is not None and created > now_dt - EXPIRE_AFTER:
                return False  # 未超 24h：归档可能迟到，下轮重试
            win = None
            settle_price = None
            pnl = 0.0
            outcome = "EXPIRED"

        # 幂等守卫：WHERE settled_at IS NULL（并发/手动补算竞争时只生效一次）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False  # 已被并发结算（幂等守卫生效）
        self._settled_count += 1
        logger.info(
            "订单结算 | id=%s | window=%s | direction=%s → %s | win=%s | pnl=%s | settle_price=%s",
            row.id, row.window_start, row.direction, outcome, win,
            f"{pnl:+.4f}" if pnl is not None else "N/A", settle_price,
        )
        return True

    async def _settle_scene_row(self, row: TradeOrderModel) -> bool:
        """15m 场景订单结算：回读 FakeBreakoutSignal.settle_outcome。

        结算判定与 pnl 公式与 5m 路径同口径（win = direction == outcome；
        赢 amount/avg_price−amount / 输 −amount）；settle_price 取信号行
        settle_btc_price（15m 周期末 BTC 中间价 P(E)）。信号未结算
        （PENDING）时 15m 检测器结算可能迟到：settle_deadline+24h 内
        下轮重试，超期 EXPIRED 出清（与 5m 兜底一致）。
        """
        now_dt = datetime.now(timezone.utc)
        if row.scene_signal_id is None:
            # 理论不可达（scene_signal_id 与占位同事务落库）：出清防每 60s
            # 空扫；CRITICAL 留痕供日志健康检查排查数据链异常
            logger.critical(
                "订单结算 | id=%s | 15m 订单缺 scene_signal_id（数据异常）"
                "→ EXPIRED 出清 | window=%s", row.id, row.window_start)
            return await self._expire_row(row, now_dt)

        async with async_session_factory() as session:
            sig = await session.get(FakeBreakoutSignal, int(row.scene_signal_id))
        if sig is None:
            # 信号行被运维 TRUNCATE/删除：无法判赢，出清不计统计
            logger.critical(
                "订单结算 | id=%s | 场景信号 %s 缺失（被清理？）→ EXPIRED 出清",
                row.id, row.scene_signal_id)
            return await self._expire_row(row, now_dt)

        outcome = sig.settle_outcome
        if outcome in ("UP", "DOWN"):
            win = row.direction == outcome
            amount = self._amount_usdt(row)
            avg_price = self._avg_price(row)
            settle_price = self._to_float(sig.settle_btc_price)
            if win and amount is not None and avg_price is not None:
                pnl = amount / avg_price - amount
            elif not win and amount is not None:
                pnl = -amount
            else:
                pnl = None  # 均价/金额缺失：输向无需均价，赢向无法估算
        elif outcome == "NOISE":
            # 与 5m 路径同口径：NOISE 即时结算（win=None/pnl=0），避免空扫重扫
            # 直到 settle_deadline+24h 才被误记为 EXPIRED
            win = None
            settle_price = self._to_float(sig.settle_btc_price)
            pnl = 0.0
        elif sig.settle_deadline is not None and datetime.fromtimestamp(
                sig.settle_deadline / 1000.0, tz=timezone.utc
        ) > now_dt - EXPIRE_AFTER:
            return False  # 检测器结算迟到（PENDING）：deadline+24h 内重试
        else:
            win = None
            settle_price = None
            pnl = 0.0
            outcome = "EXPIRED"

        # 幂等守卫：WHERE settled_at IS NULL（与 _settle_row 同模式）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False  # 已被并发结算（幂等守卫生效）
        self._settled_count += 1
        logger.info(
            "订单结算 | id=%s | scene_signal=%s | window=%s | direction=%s → %s"
            " | win=%s | pnl=%s | settle_price=%s",
            row.id, row.scene_signal_id, row.window_start, row.direction, outcome,
            win, f"{pnl:+.4f}" if pnl is not None else "N/A", settle_price,
        )
        return True

    async def _expire_row(self, row: TradeOrderModel, now_dt: datetime) -> bool:
        """异常行出清（EXPIRED/win=None/pnl=0）：防永久卡「在途持仓」+ 空扫。"""
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome="EXPIRED",
                    win=None,
                    settle_price=None,
                    pnl=0.0,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False
        self._settled_count += 1
        return True

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    async def _find_window(self, window_start: int) -> SentimentWindow | None:
        stmt = sa_select(SentimentWindow).where(
            SentimentWindow.start_time == window_start)
        async with async_session_factory() as session:
            return (await session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    def _amount_usdt(row: TradeOrderModel) -> float | None:
        """amount_in（wei 字符串）→ USDT；无效/零返回 None。"""
        try:
            v = int(row.amount_in) / 1e18
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _avg_price(row: TradeOrderModel) -> float | None:
        """quote_json.averagePrice → 成交均价；无效/零返回 None。"""
        try:
            v = float((row.quote_json or {}).get("averagePrice") or 0)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(v) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _aware(dt: datetime | None) -> datetime | None:
        """naive datetime（测试桩/驱动差异）按 UTC 补 tzinfo。"""
        if dt is None:
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
