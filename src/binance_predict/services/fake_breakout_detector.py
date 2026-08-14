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
# 5 分钟兑现口径（与 15m 并行验证：离线回测 5m 77.4% vs 15m 80.0%）
HOLD_5M_MS = 300_000
# 停机积压判定：到期后超过此宽限仍未结算的信号置 EXPIRED（现价已不能代表结算时刻）
SETTLE_EXPIRE_GRACE_MS = 300_000

# 级别定义（与回测 scripts/local_combo_level_matrix_check.py 对齐）：
# 级别名 → 回看窗口数（5m 窗口 closes 极值）
LEVEL_LOOKBACKS: dict[str, int] = {
    "1h": 12,
    "4h": 48,
    "daily": 288,
}


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

        # 三级别位势缓存：level → {"resistance": float, "support": float}
        self._levels: dict[str, dict[str, float]] = {}
        self._levels_refreshed_at: float = 0.0

        # 风控状态：按 (side, level) 独立冷却（一波冲高/冲低每级别只报一次）
        self._last_signal_at: dict[tuple[str, str], int] = {}
        self._daily_count: int = 0
        self._daily_date: str = ""  # UTC 日期串，跨天重置

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._refresh_levels(force=True)
        self._task = asyncio.create_task(self._loop(), name="fake_breakout_detector")
        logger.info(
            "假突破检测器启动 | 级别={} | eps={} | 检测间隔={}s | 冷却={}s",
            "/".join(LEVEL_LOOKBACKS.keys()),
            settings.fake_breakout_eps,
            settings.fake_breakout_check_interval,
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
    # 三级别位势计算（1h/4h/日线 closes 极值，定期刷新）
    # ==================================================================

    async def _refresh_levels(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._levels_refreshed_at) < settings.fake_breakout_resistance_refresh_seconds:
            return
        self._levels_refreshed_at = now

        max_lookback = max(LEVEL_LOOKBACKS.values())  # 288
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(SentimentWindow.exit_price)
                    .where(SentimentWindow.exit_price.isnot(None))
                    .order_by(desc(SentimentWindow.end_time))
                    .limit(max_lookback)
                )
                rows = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("假突破检测器：位势刷新失败 | {}", exc)
            return

        # closes 按时间倒序返回，反转为升序后切片算各级别
        closes = [float(r) for r in reversed(rows)]
        if len(closes) < LEVEL_LOOKBACKS["1h"]:
            logger.debug("假突破检测器：历史窗口不足（{}），位势暂不更新", len(closes))
            return

        new_levels: dict[str, dict[str, float]] = {}
        for level, lookback in LEVEL_LOOKBACKS.items():
            window = closes[-lookback:] if len(closes) >= lookback else closes
            # 冷启动保护：大级别数据不足一半时跳过该级别（防失真误触发）
            if len(window) < lookback // 2:
                continue
            new_levels[level] = {
                "resistance": max(window),
                "support": min(window),
            }

        if new_levels != self._levels:
            logger.info(
                "假突破检测器：位势更新 | {}",
                " | ".join(
                    f"{lv} 阻力 {v['resistance']:.0f} 支撑 {v['support']:.0f}"
                    for lv, v in new_levels.items()
                ),
            )
            self._levels = new_levels

    # ==================================================================
    # 主循环：秒级检测 + 顺带处理到期结算
    # ==================================================================

    async def _loop(self) -> None:
        logger.debug("假突破检测器：循环开始")
        while self._running:
            try:
                await self._refresh_levels()

                mid = self._collector.store.mid_price
                now_ms = int(time.time() * 1000)

                if mid > 0 and self._levels:
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
        """遍历 三级别 × 双向 检查破位，每个 (side, level) 独立冷却。"""
        eps = settings.fake_breakout_eps
        for level, lv in self._levels.items():
            # 冲过阻力 → 卖跌信号（买 DOWN）
            if mid > lv["resistance"] * (1.0 + eps):
                await self._fire_signal(now_ms, mid, level, "high", lv["resistance"])
            # 跌破支撑 → 买涨信号（买 UP）
            if mid < lv["support"] * (1.0 - eps):
                await self._fire_signal(now_ms, mid, level, "low", lv["support"])

    async def _fire_signal(
        self, now_ms: int, mid: float, level: str, side: str, broken_level: float
    ) -> None:
        """单个 (level, side) 破位信号的冷却检查、落表与推送。"""
        eps = settings.fake_breakout_eps
        key = (side, level)

        # 风控 1：冷却（同一级别同一方向，一波冲高/冲低只报一次）
        if now_ms - self._last_signal_at.get(key, 0) < settings.fake_breakout_cooldown_seconds * 1000:
            return

        self._daily_rollover(now_ms)
        # 风控 2：日内上限（超限仍落表，但不发邮件）
        over_daily_limit = self._daily_count >= settings.fake_breakout_max_daily_signals

        down_5m = self._pm_5m.get("down_price")
        up_5m = self._pm_5m.get("up_price")
        down_15m = self._pm_15m.get("down_price")
        up_15m = self._pm_15m.get("up_price")
        end_15m = self._pm_15m.get("end_date")
        # 结算死线对齐所报价 15m 市场的真实到期时刻（CodeReview Minor-3a）：
        # market_end_15m 缺失或已过期时退回 signal+15min 口径。
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        if end_15m and int(end_15m) > now_ms:
            settle_deadline = int(end_15m) + buffer_ms
        else:
            settle_deadline = now_ms + HOLD_MS + buffer_ms

        signal = FakeBreakoutSignal(
            level=level,
            side=side,
            signal_time=now_ms,
            resistance=broken_level,
            btc_price=mid,
            eps=eps,
            down_price_5m=down_5m,
            down_price_15m=down_15m,
            up_price_5m=up_5m,
            up_price_15m=up_15m,
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

        self._last_signal_at[key] = now_ms
        self._daily_count += 1

        direction = "DOWN" if side == "high" else "UP"
        entry_15m = down_15m if side == "high" else up_15m
        logger.info(
            "假突破信号触发 #{} [{} {}] | BTC {:.0f} 破 {} {:.0f} | "
            "目标 {} | 15m {}价={} | 日内第 {} 条{}",
            signal.id, level, side, mid, "阻力" if side == "high" else "支撑",
            broken_level, direction, direction, entry_15m,
            self._daily_count,
            "（超日限，不发邮件）" if over_daily_limit else "",
        )

        # 邮件推送（未超日限时）：fire-and-forget，绝不阻塞检测循环。
        # 实测教训：SMTP 连接被防火墙丢包时同步等待会卡死整个循环 16 分钟，
        # 导致结算回读停摆（信号 #1 事故）。
        if settings.fake_breakout_email_enabled and not over_daily_limit:
            asyncio.create_task(
                self._send_signal_email_bg(signal.id),
                name=f"fbs_email_{signal.id}",
            )

    async def _send_signal_email_bg(self, signal_id: int) -> None:
        """后台邮件发送：重新查库拿完整信号，发送成功后回填 email_sent。"""
        try:
            async with async_session_factory() as session:
                signal = await session.get(FakeBreakoutSignal, signal_id)
            if signal is None:
                return
            sent = await self._send_signal_email(signal)
            if sent:
                async with async_session_factory() as session:
                    row = await session.get(FakeBreakoutSignal, signal_id)
                    if row is not None:
                        row.email_sent = True
                        await session.commit()
        except Exception as exc:
            logger.warning("假突破信号邮件后台发送异常 #{} | {}", signal_id, exc)

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
        is_high = signal.side == "high"
        direction = "DOWN" if is_high else "UP"
        level_name = "阻力" if is_high else "支撑"
        entry_15m = signal.down_price_15m if is_high else signal.up_price_15m
        entry_5m = signal.down_price_5m if is_high else signal.up_price_5m
        backtest_wr = {"1h": "65.1%", "4h": "73.5%", "daily": "80.0%"}.get(signal.level, "—")
        subject = (
            f"[假突破信号·{signal.level}] BTC {signal.btc_price:.0f} "
            f"破{level_name} {signal.resistance:.0f} → 看{direction}"
        )
        body = (
            f"信号时间：{t_str}\n"
            f"级别：{signal.level}（{'冲过阻力' if is_high else '跌破支撑'}，方向 {direction}）\n"
            f"{level_name}位：{signal.resistance:.2f}\n"
            f"破位价格：{signal.btc_price:.2f}（{'+' if is_high else ''}{(signal.btc_price / signal.resistance - 1) * 100:.3f}%）\n"
            f"触发阈值：{signal.eps:.4f}\n\n"
            f"当时报价（{direction} token）：\n"
            f"  5m：{entry_5m}\n"
            f"  15m：{entry_15m}\n"
            f"  15m 市场到期：{end_str}\n\n"
            f"玩法提示（回测口径）：破位瞬间买 15m {direction}，持有到期按方向结算。\n"
            f"历史回测（{signal.level} 级 {direction} 方向）：胜率 {backtest_wr}。\n"
            f"当前阶段：系统不下注，仅记录信号并到期回读结算方向。\n"
        )
        return await send_plain_email(subject, body)

    # ==================================================================
    # 到期结算回读（5m + 15m 双口径并行验证）
    # ==================================================================

    async def _settle_due_signals(self, now_ms: int) -> None:
        """双时点回读：5m 口径（信号+5min）与 15m 口径（对齐市场到期）独立回填。

        CodeReview Minor-3b：停机积压超过宽限的 15m 信号置 EXPIRED 而非按现价结算；
        5m 口径逾期不回填（保持 NULL，不影响 15m 状态机）。
        """
        await self._settle_5m(now_ms)
        await self._settle_15m(now_ms)

    async def _settle_5m(self, now_ms: int) -> None:
        """5m 兑现口径回读：信号时刻 +5min + 缓冲后回填 BTC 价与方向。

        不限定 status：信号在 15m 市场临到期前 5 分钟内触发时（约占 1/3），
        15m 死线早于 5m 死线，15m 先把 status 推进 SETTLED；若限定 PENDING，
        这些信号的 5m 口径将永远卡 NULL（前端"待结算"常驻）。
        SQL 下界排除超宽限旧信号（防卡死信号累积反复占位 limit 20），
        settle_outcome_5m IS NULL 已保证幂等。
        """
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        due_5m_before = now_ms - HOLD_5M_MS - buffer_ms
        earliest = due_5m_before - SETTLE_EXPIRE_GRACE_MS
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(FakeBreakoutSignal)
                    .where(FakeBreakoutSignal.settle_outcome_5m.is_(None))
                    .where(FakeBreakoutSignal.signal_time <= due_5m_before)
                    .where(FakeBreakoutSignal.signal_time > earliest)
                    .limit(20)
                )
                due = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("假突破 5m 结算查询失败 | {}", exc)
            return

        if not due:
            return

        settle_price = await self._collector.fetch_mid_price()
        if settle_price <= 0:
            logger.warning("假突破 5m 结算：BTC 现价不可用，本轮 {} 条顺延", len(due))
            return

        try:
            async with async_session_factory() as session:
                for s in due:
                    # 逾期超宽限（停机积压）：不回填，保持 NULL 避免失真
                    if now_ms - (s.signal_time + HOLD_5M_MS + buffer_ms) > SETTLE_EXPIRE_GRACE_MS:
                        continue
                    row = await session.get(FakeBreakoutSignal, s.id)
                    if row is None or row.settle_outcome_5m is not None:
                        continue
                    row.settle_btc_price_5m = settle_price
                    if settle_price < row.btc_price:
                        row.settle_outcome_5m = "DOWN"
                    elif settle_price > row.btc_price:
                        row.settle_outcome_5m = "UP"
                    else:
                        row.settle_outcome_5m = "NOISE"
                    logger.info(
                        "假突破信号 5m 结算 #{} | 入场 {:.0f} → +5min {:.0f} | {}",
                        row.id, row.btc_price, settle_price, row.settle_outcome_5m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("假突破 5m 结算回填失败 | {}", exc)

    async def _settle_15m(self, now_ms: int) -> None:
        """15m 兑现口径回读：对齐所报价 15m 市场到期时刻回填（原 _settle_due_signals）。"""
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
            "levels": self._levels,
            "daily_count": self._daily_count,
            "daily_date": self._daily_date,
            "eps": settings.fake_breakout_eps,
            "cooldown_seconds": settings.fake_breakout_cooldown_seconds,
            "max_daily_signals": settings.fake_breakout_max_daily_signals,
        }
