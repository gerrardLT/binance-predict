"""
假突破信号系统：4h 阻力/支撑破位秒级检测 + A+B 过滤 + 信号落表 + 周期结算回读。

模式收窄（2026-08-15 拍板）：只跑回测统计显著的 4h 双向 × A+B 组合
（scripts/local_combo_filter_lab.py：4h 破阻力→DOWN 胜率 41.7% 费后 EV +7.06 [+0.35,+15.7]；
4h 破支撑→UP 53.3% EV +6.52 [+1.12,+13.9]，CI 均不含 0）。
1h/日线级别与未过 A+B 过滤的信号一律不落表、不结算、不邮件（旧模式彻底移除）。

结算口径【周期锚点，与币安预测市场真实结算规则一致】：
- 信号落在哪个市场周期，就按那个周期的涨跌判定：
  UP 赢 ⟺ P(周期末) > P(周期开盘价)；DOWN 赢 ⟺ P(周期末) < P(周期开盘价)
- 15m 口径：信号所在 15m 市场 [market_start_15m, market_end_15m]
- 5m 口径：信号所在 5m 市场 [market_start_5m, market_end_5m]（兑现窗=剩余周期 0~5min）
- 周期开盘价 P(S)：tracker 周期切换时快照（冷启动 klines 精确回读）
- 周期末价 P(E)：到期+buffer 后现价；超宽限（停机积压）用 klines 精确补，
  历史时点必然可得，停机无损；仅周期坐标缺失或 klines 也失败才 EXPIRED

当前阶段【不下注】：
- 破位且通过 A+B 过滤才落表 fake_breakout_signals（含 5m/15m 目标 token 报价快照 + 周期坐标）
- 邮件推送提醒（复用 agent_alert_* SMTP 配置，fire-and-forget 不阻塞检测循环）
- 到期回读回填周期涨跌方向（UP/DOWN 符号），stats 按 side 换算策略胜负

风控（不下注阶段）：
- 同一 (side, level) 信号冷却（默认 900s，一波冲高/冲低只报一次）
- 日内信号上限（超限后仍落表但不再发邮件，防轰炸）

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
from . import clock_sync
from .alerting import send_plain_email
from .data_collector import BinanceDataCollector

# 15m 死线兜底：15m 报价快照缺失时退回 signal+15min（对齐 HOLD_MS 语义）
HOLD_MS = 900_000
# 超宽限阈值：到期后超过此宽限未结算的信号转 klines 精确补结算路径
# （周期锚点口径下 P(S)/P(E) 均为历史时点，klines 必然可得，停机无损）
SETTLE_EXPIRE_GRACE_MS = 300_000

# 级别定义（与回测 scripts/local_combo_filter_lab.py 对齐）：
# 级别名 → 回看窗口数（5m 窗口 closes 极值）。仅保留统计显著的 4h（2026-08-15 收窄）
LEVEL_LOOKBACKS: dict[str, int] = {
    "4h": 48,
}

# 行动过滤器阈值（scripts/local_combo_filter_lab.py 回测，4 个 级别×方向 场景方向一致）：
# A 剩余时间：信号落在 15m 周期前 6 分钟才可行动（12~15min 尾段桶 140 注仅 2 胜）
# B 破位幅度：信号价偏离周期开盘价 <0.2%（>0.3% 桶 144 注 0 胜——周期内回吐整段破位几乎不可能）
FILTER_A_MAX_OFFSET_SEC = 360
FILTER_B_MAX_BREAK_PCT = 0.20


def evaluate_signal_filter(
    level: str,
    side: str,
    offset_sec: int | float | None,
    break_pct: float | None,
) -> tuple[bool, str]:
    """A(周期内偏移) + B(破位幅度) + 方向 三重过滤判定（邮件/API/前端共用的单一事实源）。

    返回 (是否可行动, 逐项中文说明)。数据缺失的项按不通过处理（保守：宁可放弃也不错动）。
    """
    parts: list[str] = []
    ok_all = True

    # A：周期内偏移（剩余时间越多，价格越有机会回归周期开盘价另一侧）
    if offset_sec is None:
        ok_all = False
        parts.append("A 周期偏移未知❌")
    else:
        a_ok = offset_sec < FILTER_A_MAX_OFFSET_SEC
        ok_all = ok_all and a_ok
        parts.append(f"A 周期第 {offset_sec / 60:.1f}min（<6min {'✅' if a_ok else '❌'}）")

    # B：破位幅度（破位越深，周期内回吐全部幅度越不可能）
    if break_pct is None:
        ok_all = False
        parts.append("B 破位幅度未知❌")
    else:
        b_ok = break_pct < FILTER_B_MAX_BREAK_PCT
        ok_all = ok_all and b_ok
        parts.append(f"B {break_pct:.3f}%（<0.2% {'✅' if b_ok else '❌'}）")

    # 方向：daily 破支撑→UP 回测费后 EV 为负（-0.12），回避
    dir_ok = not (level == "daily" and side == "low")
    ok_all = ok_all and dir_ok
    parts.append(f"方向 {level} {side}{'✅' if dir_ok else '❌回避'}")

    return ok_all, "｜".join(parts)


class FakeBreakoutDetector:
    """
    4h 阻力/支撑假突破秒级检测器（A+B 过滤前置：未过过滤不落表）。

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
    # 4h 位势计算（48 个 5m 窗口 closes 极值，定期刷新）
    # ==================================================================

    async def _refresh_levels(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._levels_refreshed_at) < settings.fake_breakout_resistance_refresh_seconds:
            return
        self._levels_refreshed_at = now

        max_lookback = max(LEVEL_LOOKBACKS.values())  # 48
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
        if len(closes) < min(LEVEL_LOOKBACKS.values()):
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
                # 统一用币安服务器时钟：与市场 end_date（币安时钟）比较无时钟偏差
                now_ms = clock_sync.now_ms()

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
        """遍历 4h 级别 × 双向 检查破位，每个 (side, level) 独立冷却。"""
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
        start_15m = self._pm_15m.get("start_date")
        open_15m = self._pm_15m.get("cycle_open_price")
        open_15m_end = self._pm_15m.get("cycle_open_end")
        start_5m = self._pm_5m.get("start_date")
        end_5m = self._pm_5m.get("end_date")
        open_5m = self._pm_5m.get("cycle_open_price")
        open_5m_end = self._pm_5m.get("cycle_open_end")

        # 结算死线对齐所报价 15m 市场的真实到期时刻（CodeReview Minor-3a）：
        # market_end_15m 缺失或已过期时退回 signal+15min 口径（该条周期坐标落空，
        # 到期后无锚点可判，将由 EXPIRED 路径收场）。
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        if end_15m and int(end_15m) > now_ms:
            settle_deadline = int(end_15m) + buffer_ms
            # 周期坐标与开盘价配对守卫：cycle_open_end 必须与当前周期 end_date 一致，
            # 否则是 tracker 写入跨轮错配，宁可落空也不错配
            m_start_15m = int(start_15m) if start_15m else int(end_15m) - 900_000
            m_open_15m = (
                float(open_15m)
                if open_15m and open_15m_end is not None and int(open_15m_end) == int(end_15m)
                else None
            )
            m_end_15m: int | None = int(end_15m)
        else:
            settle_deadline = now_ms + HOLD_MS + buffer_ms
            m_start_15m = None
            m_open_15m = None
            m_end_15m = None
            logger.warning("15m 报价快照缺失/过期，信号周期坐标落空 | {} {}", level, side)

        # 5m 周期坐标：快照缺失或已过期（staleness 防御）则该条 5m 口径不结算
        if end_5m and int(end_5m) > now_ms:
            m_start_5m = int(start_5m) if start_5m else int(end_5m) - 300_000
            m_end_5m: int | None = int(end_5m)
            m_open_5m = (
                float(open_5m)
                if open_5m and open_5m_end is not None and int(open_5m_end) == int(end_5m)
                else None
            )
        else:
            m_start_5m = None
            m_end_5m = None
            m_open_5m = None

        # 过滤器输入（A/B 回测口径）：fire 时计算落表，邮件/API/前端据此标注可行动性
        m_offset_sec: int | None = (
            int(round((now_ms - m_start_15m) / 1000.0)) if m_start_15m else None
        )
        if m_open_15m and m_open_15m > 0:
            raw = mid / m_open_15m - 1.0
            m_break_pct: float | None = round(raw * 100.0 if side == "high" else -raw * 100.0, 4)
        else:
            m_break_pct = None
        filter_pass, filter_reason = evaluate_signal_filter(level, side, m_offset_sec, m_break_pct)
        # 模式收窄：未过 A+B 过滤的信号彻底丢弃（不落表/不结算/不邮件）。
        # 不占冷却——若后续周期前段再次破位且满足过滤，仍会正常报出
        if not filter_pass:
            logger.debug("假突破被过滤丢弃 [{} {}] | {}", level, side, filter_reason)
            return

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
            market_end_15m=m_end_15m,
            market_start_15m=m_start_15m,
            cycle_open_price_15m=m_open_15m,
            market_start_5m=m_start_5m,
            market_end_5m=m_end_5m,
            cycle_open_price_5m=m_open_5m,
            cycle_offset_sec_15m=m_offset_sec,
            break_pct=m_break_pct,
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
            "目标 {} | 15m {}价={} | 日内第 {} 条 | 过滤 {}{}",
            signal.id, level, side, mid, "阻力" if side == "high" else "支撑",
            broken_level, direction, direction, entry_15m,
            self._daily_count,
            "✅" if filter_pass else "❌",
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
        filter_pass, filter_label = evaluate_signal_filter(
            signal.level, signal.side, signal.cycle_offset_sec_15m, signal.break_pct
        )
        # A+B 过滤组合回测（scripts/local_combo_filter_lab.py，8809 窗口，CI 不含 0）：
        # 低胜率高赔率——赢需周期末价回到周期开盘价另一侧；仅剩 4h 双向（其余已移除）
        backtest = {
            ("4h", "high"): "A+B 组合胜率 41.7%，费后 EV +7.06 [+0.35,+15.7]",
            ("4h", "low"): "A+B 组合胜率 53.3%，费后 EV +6.52 [+1.12,+13.9]",
        }.get((signal.level, signal.side), "—")
        subject = (
            f"[假突破信号·{signal.level} {'✅可行动' if filter_pass else '❌被过滤'}] "
            f"BTC {signal.btc_price:.0f} 破{level_name} {signal.resistance:.0f} → 看{direction}"
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
            f"过滤判定（A 剩余时间 <6min ｜ B 破位幅度 <0.2% ｜ 方向）：\n"
            f"  {filter_label}\n"
            f"  结论：{'✅ 可行动（A+B 组合 4h 口径回测胜率 42~53%，费后 EV +6.5~+7.1）' if filter_pass else '❌ 被过滤，建议放弃本注（尾段/深破/回避方向回测 EV 为负或 0 胜）'}\n\n"
            f"玩法提示（回测口径）：破位瞬间买 15m {direction}，持有到周期到期，\n"
            f"按周期末价 vs 周期开盘价结算（与市场真实规则一致）。\n"
            f"历史回测（{signal.level} 级 {direction} 方向基线，未过滤）：{backtest}。\n"
            f"当前阶段：系统不下注，仅记录信号并到期回读结算方向。\n"
        )
        return await send_plain_email(subject, body)

    # ==================================================================
    # 到期结算回读（5m + 15m 双口径并行验证）
    # ==================================================================

    async def _settle_due_signals(self, now_ms: int) -> None:
        """双口径回读：5m 口径（所在 5m 周期到期）与 15m 口径（所在 15m 周期到期）。

        周期锚点口径：方向 = 周期末价 P(E) vs 周期开盘价 P(S)，与币安市场真实
        结算规则一致。锚点缺失时 klines 精确回读；超宽限（停机积压）同样走
        klines 补 P(E)——历史时点必然可得，停机无损，永不按失真现价结算。
        """
        await self._settle_5m(now_ms)
        await self._settle_15m(now_ms)

    async def _klines_open(self, interval: str, start_ms: int | None) -> float:
        """klines 回读某周期边界的开盘价；start_ms 缺失或失败返回 0.0。"""
        if not start_ms:
            return 0.0
        return await self._collector.fetch_kline_open(interval, int(start_ms))

    async def _settle_5m(self, now_ms: int) -> None:
        """5m 周期锚点回读：信号所在 5m 周期到期 + 缓冲后，按 P(E5) vs P(S5) 判定。

        - 锚点 P(S5)：fire 时快照的 cycle_open_price_5m；缺失则 klines 回读并补列
        - P(E5)：统一 klines 精确读（下一根 kline 开盘价 = 周期末时刻价格，时点零误差）；
          klines 暂时失败时宽限内可现价兜底，超宽限必须 klines（下轮重试，历史必然可得）
        - 周期坐标缺失（fire 时 5m 报价快照落空）的条目不结算，保持 NULL
        - 不限定 status：15m 死线可能更早先把 status 推进 SETTLED，两口径独立回填
        """
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(FakeBreakoutSignal)
                    .where(FakeBreakoutSignal.settle_outcome_5m.is_(None))
                    .where(FakeBreakoutSignal.market_end_5m.isnot(None))
                    .where(FakeBreakoutSignal.market_end_5m + buffer_ms <= now_ms)
                    # 锚点与周期起点都缺失的条目永远无法结算，SQL 层排除防占位
                    .where(
                        (FakeBreakoutSignal.cycle_open_price_5m.isnot(None))
                        | (FakeBreakoutSignal.market_start_5m.isnot(None))
                    )
                    .limit(20)
                )
                due = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.warning("假突破 5m 结算查询失败 | {}", exc)
            return

        if not due:
            return

        live_price: float | None = None  # 惰性 fallback：仅 klines 失败且宽限内才取现价
        try:
            async with async_session_factory() as session:
                for s in due:
                    end_5m = int(s.market_end_5m)
                    # P(E5)：klines 精确读周期末价；失败时宽限内现价兜底，超宽限下轮重试
                    close_price = await self._klines_open("5m", end_5m)
                    if close_price <= 0:
                        if now_ms - (end_5m + buffer_ms) > SETTLE_EXPIRE_GRACE_MS:
                            continue
                        if live_price is None:
                            live_price = await self._collector.fetch_mid_price()
                        close_price = live_price
                        if close_price <= 0:
                            continue

                    row = await session.get(FakeBreakoutSignal, s.id)
                    if row is None or row.settle_outcome_5m is not None:
                        continue
                    # 锚点 P(S5)：快照缺失时 klines 回读并补列
                    anchor = row.cycle_open_price_5m
                    if not anchor or anchor <= 0:
                        anchor = await self._klines_open("5m", row.market_start_5m)
                        if anchor <= 0:
                            continue  # 下轮重试
                        row.cycle_open_price_5m = anchor

                    row.settle_btc_price_5m = close_price
                    if close_price < anchor:
                        row.settle_outcome_5m = "DOWN"
                    elif close_price > anchor:
                        row.settle_outcome_5m = "UP"
                    else:
                        row.settle_outcome_5m = "NOISE"
                    logger.info(
                        "假突破信号 5m 周期结算 #{} | 周期开盘 {:.0f} → 周期末 {:.0f} | {}",
                        row.id, anchor, close_price, row.settle_outcome_5m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("假突破 5m 结算回填失败 | {}", exc)

    async def _settle_15m(self, now_ms: int) -> None:
        """15m 周期锚点回读：对齐信号所在 15m 市场到期，按 P(E) vs P(S15) 判定。
    
        - 锚点 P(S15)：fire 时快照；缺失则 klines 回读并补列
        - P(E)：统一 klines 精确读（时点零误差）；klines 暂时失败时宽限内现价兜底，
          超宽限（停机积压）必须 klines——历史必然可得，停机无损，不再置 EXPIRED
        - EXPIRED 仅用于周期坐标缺失（fire 时 15m 报价快照落空，无锚点可判）
        """
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
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

        # 周期坐标缺失的条目无锚点可判，直接 EXPIRED（fire 时 15m 快照落空）
        expired_ids = [s.id for s in due if not s.market_start_15m or not s.market_end_15m]
        if expired_ids:
            try:
                async with async_session_factory() as session:
                    for sid in expired_ids:
                        row = await session.get(FakeBreakoutSignal, sid)
                        if row is not None and row.status == "PENDING":
                            row.status = "EXPIRED"
                            logger.warning(
                                "假突破信号 #{} 周期坐标缺失（15m 报价快照落空），置 EXPIRED",
                                row.id,
                            )
                    await session.commit()
            except Exception as exc:
                logger.warning("假突破信号 EXPIRED 回填失败 | {}", exc)
            due = [s for s in due if s.id not in set(expired_ids)]
            if not due:
                return

        live_price: float | None = None  # 惰性 fallback：仅 klines 失败且宽限内才取现价
        try:
            async with async_session_factory() as session:
                for s in due:
                    end_15m = int(s.market_end_15m)
                    # P(E)：klines 精确读周期末价；失败时宽限内现价兜底，超宽限下轮重试
                    close_price = await self._klines_open("15m", end_15m)
                    if close_price <= 0:
                        if now_ms - s.settle_deadline > SETTLE_EXPIRE_GRACE_MS:
                            continue
                        if live_price is None:
                            live_price = await self._collector.fetch_mid_price()
                        close_price = live_price
                        if close_price <= 0:
                            continue

                    row = await session.get(FakeBreakoutSignal, s.id)
                    if row is None or row.status != "PENDING":
                        continue
                    # 锚点 P(S15)：快照缺失时 klines 回读并补列
                    anchor = row.cycle_open_price_15m
                    if not anchor or anchor <= 0:
                        anchor = await self._klines_open("15m", row.market_start_15m)
                        if anchor <= 0:
                            continue  # 下轮重试
                        row.cycle_open_price_15m = anchor

                    row.settle_btc_price = close_price
                    if close_price < anchor:
                        row.settle_outcome = "DOWN"
                    elif close_price > anchor:
                        row.settle_outcome = "UP"
                    else:
                        row.settle_outcome = "NOISE"
                    row.status = "SETTLED"
                    logger.info(
                        "假突破信号 15m 周期结算 #{} | 周期开盘 {:.0f} → 周期末 {:.0f} | {} | 入场价 {}",
                        row.id, anchor, close_price, row.settle_outcome, row.down_price_15m,
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
