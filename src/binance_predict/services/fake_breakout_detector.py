"""
假突破信号系统：日线阻力破位的秒级检测 + 信号落表 + 到期结算回读。

策略口径（本地一个月数据回测验证，scripts/local_combo_5m_15m_check.py）：
- 触发：BTC 盘中冲高，现价 > 日线阻力 × (1 + eps)
- 日线阻力 = 前 288 个 5m 窗口 closes 的 max（与回测 LOOKBACK=288 对齐）
- 兑现：15 分钟（币安 15m 预测市场到期按 BTC 方向结算，只看符号）
- 回测成绩：BTC 方向胜率 80.0%（80 注），对照组均值回归基线 65.4%

当前阶段【不下注】：
- 破位瞬间落表 fake_breakout_signals（含 5m/15m DOWN 当时报价快照）
- 邮件推送提醒（复用 agent_alert_* SMTP 配置）
- 到期后回读 BTC 价格回填结算方向（UP/DOWN 符号）
- 积累 15m 市场真实赔率数据，回答"冲高瞬间 15m DOWN 真实报价"这一回测遗留疑虑

风控（不下注阶段）：
- 同一阻力位信号冷却（默认 900s，覆盖一个兑现周期，一波冲高只报一次）
- 日内信号上限（默认 50，超限后仍落表但不再发邮件，防轰炸）

生命周期由 main.py lifespan 管理：start() 启动循环，stop() 优雅停止。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import desc, select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import FakeBreakoutSignal, SentimentWindow
from .alerting import send_plain_email
from .data_collector import BinanceDataCollector

# 兑现窗口（15 分钟，与回测 HOLD_MS 对齐）
HOLD_MS = 900_000
# 停机积压判定：到期后超过此宽限仍未结算的信号置 EXPIRED（现价已不能代表结算时刻）
SETTLE_EXPIRE_GRACE_MS = 300_000


class FakeBreakoutDetector:
    """
    日线阻力假突破秒级检测器。

    报价快照通过两个 dict 引用注入（main.py 模块级变量，tracker 单写者）：
    - pm_15m_latest: 15m 市场最新报价 {"down_price", "up_price", "end_date", "updated_ts"}
    - pm_market_info: 5m 市场最新元数据（含 down_price）
    """

    def __init__(
        self,
        collector: BinanceDataCollector,
        pm_15m_latest: dict,
        pm_market_info: dict,
    ) -> None:
        self._collector = collector
        self._pm_15m = pm_15m_latest
        self._pm_5m = pm_market_info

        self._running = False
        self._task: asyncio.Task | None = None

        # 日线阻力缓存
        self._resistance: float | None = None
        self._resistance_refreshed_at: float = 0.0

        # 风控状态
        self._last_signal_at_ms: int = 0
        self._daily_count: int = 0
        self._daily_date: str = ""  # UTC 日期串，跨天重置

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._refresh_resistance(force=True)
        self._task = asyncio.create_task(self._loop(), name="fake_breakout_detector")
        logger.info(
            "假突破检测器启动 | eps={} | 检测间隔={}s | 阻力回看={}窗 | 冷却={}s",
            settings.fake_breakout_eps,
            settings.fake_breakout_check_interval,
            settings.fake_breakout_resistance_lookback,
            settings.fake_breakout_cooldown_seconds,
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
        logger.info("假突破检测器已停止")

    # ==================================================================
    # 日线阻力计算（前 288 个 5m 窗口 closes 的 max，定期刷新）
    # ==================================================================

    async def _refresh_resistance(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._resistance_refreshed_at) < settings.fake_breakout_resistance_refresh_seconds:
            return
        self._resistance_refreshed_at = now

        lookback = settings.fake_breakout_resistance_lookback
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(SentimentWindow.exit_price)
                    .where(SentimentWindow.exit_price.isnot(None))
                    .order_by(desc(SentimentWindow.end_time))
                    .limit(lookback)
                )
                rows = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("假突破检测器：阻力位刷新失败 | {}", exc)
            return

        if len(rows) < lookback // 2:
            # 数据太少（冷启动初期）不更新，避免阻力位失真误触发
            logger.debug(
                "假突破检测器：历史窗口不足（{}/{}），阻力位暂不更新",
                len(rows), lookback,
            )
            return

        new_resistance = max(float(r) for r in rows)
        if new_resistance != self._resistance:
            logger.info(
                "假突破检测器：日线阻力更新 {} → {}（基于 {} 个窗口 closes）",
                self._resistance, new_resistance, len(rows),
            )
            self._resistance = new_resistance

    # ==================================================================
    # 主循环：秒级检测 + 顺带处理到期结算
    # ==================================================================

    async def _loop(self) -> None:
        logger.debug("假突破检测器：循环开始")
        while self._running:
            try:
                await self._refresh_resistance()

                mid = self._collector.store.mid_price
                now_ms = int(time.time() * 1000)

                if mid > 0 and self._resistance is not None:
                    await self._check_breakout(now_ms, mid)

                await self._settle_due_signals(now_ms)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "假突破检测器：循环异常 | error_type={} | error={}",
                    type(exc).__name__, exc,
                )

            try:
                await asyncio.sleep(settings.fake_breakout_check_interval)
            except asyncio.CancelledError:
                break
        logger.debug("假突破检测器：循环结束")

    # ==================================================================
    # 破位检测与信号落表
    # ==================================================================

    def _daily_rollover(self, now_ms: int) -> None:
        """UTC 跨天重置日内计数。"""
        today = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_count = 0

    async def _check_breakout(self, now_ms: int, mid: float) -> None:
        assert self._resistance is not None
        eps = settings.fake_breakout_eps
        if mid <= self._resistance * (1.0 + eps):
            return

        # 风控 1：冷却（一波冲高只报一次）
        if now_ms - self._last_signal_at_ms < settings.fake_breakout_cooldown_seconds * 1000:
            return

        self._daily_rollover(now_ms)
        # 风控 2：日内上限（超限仍落表，但不发邮件）
        over_daily_limit = self._daily_count >= settings.fake_breakout_max_daily_signals

        down_5m = self._pm_5m.get("down_price")
        down_15m = self._pm_15m.get("down_price")
        end_15m = self._pm_15m.get("end_date")
        # 结算死线对齐所报价 15m 市场的真实到期时刻（CodeReview Minor-3a）：
        # 信号若在该市场后段触发，按 signal+15min 回读不代表所记录 DOWN 报价的实际盈亏。
        # market_end_15m 缺失或已过期时退回 signal+15min 口径。
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        if end_15m and int(end_15m) > now_ms:
            settle_deadline = int(end_15m) + buffer_ms
        else:
            settle_deadline = now_ms + HOLD_MS + buffer_ms

        signal = FakeBreakoutSignal(
            signal_time=now_ms,
            resistance=self._resistance,
            btc_price=mid,
            eps=eps,
            down_price_5m=down_5m,
            down_price_15m=down_15m,
            market_end_15m=end_15m,
            settle_deadline=settle_deadline,
            status="PENDING",
            email_sent=False,
        )
        try:
            async with async_session_factory() as session:
                session.add(signal)
                await session.commit()
                await session.refresh(signal)
        except Exception as exc:
            logger.error("假突破信号落表失败 | {}", exc)
            return

        self._last_signal_at_ms = now_ms
        self._daily_count += 1

        logger.info(
            "假突破信号触发 #{} | BTC {:.0f} > 阻力 {:.0f}×(1+{:.4f}) | "
            "5m DOWN={} 15m DOWN={} | 15m到期={} | 日内第 {} 条{}",
            signal.id, mid, self._resistance, eps,
            down_5m, down_15m, end_15m,
            self._daily_count,
            "（超日限，不发邮件）" if over_daily_limit else "",
        )

        # 邮件推送（未超日限时）
        if settings.fake_breakout_email_enabled and not over_daily_limit:
            sent = await self._send_signal_email(signal)
            if sent:
                try:
                    async with async_session_factory() as session:
                        row = await session.get(FakeBreakoutSignal, signal.id)
                        if row is not None:
                            row.email_sent = True
                            await session.commit()
                except Exception as exc:
                    logger.warning("假突破信号 email_sent 回填失败 | {}", exc)

    async def _send_signal_email(self, signal: FakeBreakoutSignal) -> bool:
        t_str = datetime.fromtimestamp(
            signal.signal_time / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        end_str = (
            datetime.fromtimestamp(signal.market_end_15m / 1000, tz=timezone.utc).strftime(
                "%H:%M:%S UTC"
            )
            if signal.market_end_15m
            else "未知"
        )
        subject = (
            f"[假突破信号] BTC {signal.btc_price:.0f} 冲过日线阻力 {signal.resistance:.0f}"
        )
        body = (
            f"信号时间：{t_str}\n"
            f"日线阻力：{signal.resistance:.2f}\n"
            f"破位价格：{signal.btc_price:.2f}（+{(signal.btc_price / signal.resistance - 1) * 100:.3f}%）\n"
            f"触发阈值：{signal.eps:.4f}\n\n"
            f"当时报价：\n"
            f"  5m  DOWN token：{signal.down_price_5m}\n"
            f"  15m DOWN token：{signal.down_price_15m}\n"
            f"  15m 市场到期：{end_str}\n\n"
            f"玩法提示（回测口径）：冲高瞬间买 15m DOWN，持有到期按方向结算。\n"
            f"历史回测：BTC 方向胜率 80.0%（80 注，对照基线 65.4%）。\n"
            f"当前阶段：系统不下注，仅记录信号并到期回读结算方向。\n"
        )
        return await send_plain_email(subject, body)

    # ==================================================================
    # 到期结算回读
    # ==================================================================

    async def _settle_due_signals(self, now_ms: int) -> None:
        """回填到期信号的结算方向（settle_btc < btc_price → DOWN 赢，只看符号）。

        CodeReview Minor-3b：停机积压超过宽限的信号，现价已不能代表结算时刻，
        置 EXPIRED 而非按现价结算（避免失真）。
        """
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(FakeBreakoutSignal)
                    .where(FakeBreakoutSignal.status == "PENDING")
                    .where(FakeBreakoutSignal.settle_deadline <= now_ms)
                    .limit(20)
                )
                due = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("假突破结算查询失败 | {}", exc)
            return

        if not due:
            return

        # 先分离停机积压信号（逾期超宽限 → EXPIRED，不按现价结算）
        expired_ids = [s.id for s in due if now_ms - s.settle_deadline > SETTLE_EXPIRE_GRACE_MS]
        if expired_ids:
            try:
                async with async_session_factory() as session:
                    for sid in expired_ids:
                        row = await session.get(FakeBreakoutSignal, sid)
                        if row is not None and row.status == "PENDING":
                            row.status = "EXPIRED"
                            logger.warning(
                                "假突破信号 #{} 停机积压逾期 {}s，置 EXPIRED（不按现价结算）",
                                row.id, (now_ms - row.settle_deadline) // 1000,
                            )
                    await session.commit()
            except Exception as exc:
                logger.warning("假突破信号 EXPIRED 回填失败 | {}", exc)
            due = [s for s in due if s.id not in set(expired_ids)]
            if not due:
                return

        settle_price = await self._collector.fetch_mid_price()
        if settle_price <= 0:
            logger.warning("假突破结算：BTC 现价不可用，本轮 {} 条顺延", len(due))
            return

        try:
            async with async_session_factory() as session:
                for s in due:
                    row = await session.get(FakeBreakoutSignal, s.id)
                    if row is None or row.status != "PENDING":
                        continue
                    row.settle_btc_price = settle_price
                    if settle_price < row.btc_price:
                        row.settle_outcome = "DOWN"
                    elif settle_price > row.btc_price:
                        row.settle_outcome = "UP"
                    else:
                        row.settle_outcome = "NOISE"
                    row.status = "SETTLED"
                    logger.info(
                        "假突破信号结算 #{} | 入场 BTC {:.0f} → 结算 {:.0f} | 方向 {} | 15m DOWN 入场价 {}",
                        row.id, row.btc_price, settle_price,
                        row.settle_outcome, row.down_price_15m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("假突破结算回填失败 | {}", exc)

    # ==================================================================
    # 只读状态（供 API 查询）
    # ==================================================================

    @property
    def status_snapshot(self) -> dict:
        return {
            "running": self._running,
            "resistance": self._resistance,
            "daily_count": self._daily_count,
            "daily_date": self._daily_date,
            "eps": settings.fake_breakout_eps,
            "cooldown_seconds": settings.fake_breakout_cooldown_seconds,
            "max_daily_signals": settings.fake_breakout_max_daily_signals,
        }
