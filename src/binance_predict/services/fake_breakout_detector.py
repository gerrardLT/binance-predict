"""
场景信号系统：4h 位势破位检测 + 15m 周期收盘质量确认 + 次周期入场提醒。

模式升级（2026-08-15 拍板）：旧 A+B 过滤方案（破位瞬间当周期行动）整体替换为
经 180 天官方数据样本外盲验的两个高确定性场景（scripts/local_continuation_discovery.py、
local_lowside_validation.py：前 120 天发现集筛选 → 后 60 天验证集盲验）：

场景① bull_exhaust（多头耗尽，验证集 63.6% [59.3,68.0] n=462，EV@0.50 +0.22/事件）：
  15m 周期内刺破 4h 阻力位势 + 周期收盘收阳 + 光头收盘（收盘位置 (C-L)/(H-L) ≥ 0.85）
  → 次周期开盘买 DOWN。入场方案：开盘半仓 @~0.50 + 次周期内反弹至 +0.10% 加仓半仓 @~0.27
    （180 天回测：开盘全仓 EV/事件 +0.223，触价加仓 +0.270，深等 z≥1 为负 EV 陷阱）

场景② bear_exhaust（镜像，验证集 57.8% [53.5,62.1] n=512，EV@0.50 +0.11/事件）：
  15m 周期内跌破 4h 支撑位势 + 周期收盘收阴 + 放量（量 ≥ 2× 前 20 根 15m 均量）
  → 次周期开盘买 UP。入场方案：只开盘买——跌态中 UP token 无折扣（0.79~0.90），等待无价值

检测流程：
  1. 秒级循环：mid 破位势 → 记 pending（仅内存，不报警不落表，每方向每周期记首次）
  2. 15m 周期边界：拉上一根完整 15m K（OHLCV）+ 前 20 根均量，判定收盘质量
  3. 场景命中 → 落表 fake_breakout_signals（pattern 列区分）+ 邮件 + 次周期锚点结算

结算口径【周期锚点，与币安预测市场真实结算规则一致】：
- 目标周期 = 破位确认后的下一个 15m 周期 [market_start_15m, market_end_15m]
- UP 赢 ⟺ P(周期末) > P(周期开盘价)；DOWN 赢 ⟺ P(周期末) < P(周期开盘价)
- 周期开盘价 P(S)：fire 时快照（cycle_open_end 配对守卫）；缺失则结算时 klines 精确回读
- 周期末价 P(E)：到期+buffer 后 klines 精确读；超宽限（停机积压）下轮重试，停机无损

当前阶段【不下注】：落表 + 邮件提醒 + 到期回读结算方向，积累实盘口径胜率/EV。

风控（不下注阶段）：
- pending 按 (side, 周期) 天然去重，一波冲高/冲低每周期最多一条信号
- 日内信号上限（超限后仍落表但不再发邮件，防轰炸）
- 确认重试：klines 拉取失败最多重试 5 次/60s，超时放弃（次周期已走远，入场价假设失效）

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

# 超宽限阈值：到期后超过此宽限未结算的信号转 klines 精确补结算路径
# （周期锚点口径下 P(S)/P(E) 均为历史时点，klines 必然可得，停机无损）
SETTLE_EXPIRE_GRACE_MS = 300_000

# 级别定义（与回测 scripts/local_combo_filter_lab.py 对齐）：
# 级别名 → 回看窗口数（5m 窗口 closes 极值）。仅保留统计显著的 4h（2026-08-15 收窄）
LEVEL_LOOKBACKS: dict[str, int] = {
    "4h": 48,
}

# 场景定义（180 天官方数据，发现集筛选 → 验证集盲验，scripts/local_continuation_discovery.py）：
# 场景① bull_exhaust：破 4h 阻力 + 周期收阳 + 光头收盘 → 次周期 DOWN（验证集 63.6%）
# 场景② bear_exhaust：破 4h 支撑 + 周期收阴 + 放量 → 次周期 UP（验证集 57.8%）
CLOSE_POS_MIN = 0.85   # 场景①：收盘位置 (C-L)/(H-L) 下限（上影 ≤ 15% 振幅）
VOL_RATIO_MIN = 2.0    # 场景②：量比下限（本周期 15m 量 / 前 20 根均量）
VOL_MA_WINDOW = 20     # 场景②：均量窗口（根，不含当前周期）
# 确认重试上限：klines 拉取失败时每轮循环重试，超过此时限放弃（次周期走远，入场价假设失效）
CONFIRM_RETRY_MAX = 5
CONFIRM_RETRY_TIMEOUT_MS = 60_000


def classify_close_pattern(
    side: str,
    o: float,
    h: float,
    l: float,
    c: float,
    volume: float,
    vol_ma: float | None,
) -> tuple[bool, float | None, float | None]:
    """信号周期收盘质量判定（纯函数，邮件/API/测试共用的单一事实源）。

    Args:
        side: 破位方向（high=破阻力→场景① | low=破支撑→场景②）
        o/h/l/c: 信号周期 15m K 线 OHLC
        volume: 信号周期 15m 成交量
        vol_ma: 前 VOL_MA_WINDOW 根 15m 均量（不含当前根）；数据不足时传 None（场景②保守不通过）

    Returns:
        (是否命中场景, close_pos, vol_ratio)
    """
    rng = h - l
    if rng <= 0 or o <= 0:
        return False, None, None
    close_pos = (c - l) / rng
    vol_ratio = volume / vol_ma if vol_ma and vol_ma > 0 else None
    if side == "high":
        # 场景①：收阳 + 光头（收盘贴自身最高，多头满仓无剩余买力）
        ok = c > o and close_pos >= CLOSE_POS_MIN
    else:
        # 场景②：收阴 + 放量（数据不足时保守不通过）
        ok = c < o and vol_ratio is not None and vol_ratio >= VOL_RATIO_MIN
    return ok, close_pos, vol_ratio


class FakeBreakoutDetector:
    """
    场景信号检测器：4h 位势破位秒级记录 → 15m 周期收盘质量确认 → 次周期信号。

    报价快照通过 dict 引用注入（main.py 模块级变量，tracker 单写者）：
    - pm_15m_latest: 15m 市场最新报价 {"down_price", "up_price", "end_date",
      "cycle_open_price", "cycle_open_end"}
    """

    def __init__(
        self,
        collector: BinanceDataCollector,
        pm_15m_latest: dict,
    ) -> None:
        self._collector = collector
        self._pm_15m = pm_15m_latest

        self._running = False
        self._task: asyncio.Task | None = None

        # 三级别位势缓存：level → {"resistance": float, "support": float}
        self._levels: dict[str, dict[str, float]] = {}
        self._levels_refreshed_at: float = 0.0

        # 风控状态：按 (side, level) 独立冷却（双保险；主去重靠 pending 周期覆盖）
        self._last_signal_at: dict[tuple[str, str], int] = {}
        self._daily_count: int = 0
        self._daily_date: str = ""  # UTC 日期串，跨天重置

        # 场景确认状态：
        # - _pending_breaks: side → 本周期首次破位记录（仅内存，周期收盘时判定）
        # - _last_cycle_id: 上次循环所在的 15m 周期号（边界检测用；冷启动为 None）
        # - _confirm_retries: klines 拉取失败的确认重试队列
        self._pending_breaks: dict[str, dict] = {}
        self._last_cycle_id: int | None = None
        self._confirm_retries: list[dict] = []

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
            "场景检测器启动 | 级别={} | eps={} | 检测间隔={}s | 冷却={}s | 场景①close_pos≥{} 场景②量比≥{}",
            "/".join(LEVEL_LOOKBACKS.keys()),
            settings.fake_breakout_eps,
            settings.fake_breakout_check_interval,
            settings.fake_breakout_cooldown_seconds,
            CLOSE_POS_MIN, VOL_RATIO_MIN,
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
        logger.info("场景检测器已停止")

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
            logger.warning("场景检测器：位势刷新失败 | {}", exc)
            return

        # closes 按时间倒序返回，反转为升序后切片算各级别
        closes = [float(r) for r in reversed(rows)]
        if len(closes) < min(LEVEL_LOOKBACKS.values()):
            logger.debug("场景检测器：历史窗口不足（{}），位势暂不更新", len(closes))
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
                "场景检测器：位势更新 | {}",
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
        logger.debug("场景检测器：循环开始")
        while self._running:
            try:
                await self._refresh_levels()

                mid = self._collector.store.mid_price
                # 统一用币安服务器时钟：与市场 end_date（币安时钟）比较无时钟偏差
                now_ms = clock_sync.now_ms()

                # 确认重试队列优先处理（klines 之前拉取失败的周期收盘确认）
                await self._drain_confirm_retries(now_ms)

                # 15m 周期边界检测：跨周期 → 对上一周期做收盘质量确认
                cycle_id = now_ms // 900_000
                if self._last_cycle_id is not None and cycle_id != self._last_cycle_id:
                    await self._on_cycle_boundary(self._last_cycle_id, cycle_id, now_ms)
                self._last_cycle_id = cycle_id

                if mid > 0 and self._levels:
                    self._record_breakout(now_ms, mid)

                await self._settle_due_signals(now_ms)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "场景检测器：循环异常 | error_type={} | error={}",
                    type(exc).__name__, exc,
                )

            try:
                await asyncio.sleep(settings.fake_breakout_check_interval)
            except asyncio.CancelledError:
                break
        logger.debug("场景检测器：循环结束")

    # ==================================================================
    # 破位检测与信号落表
    # ==================================================================

    def _daily_rollover(self, now_ms: int) -> None:
        """UTC 跨天重置日内计数。"""
        today = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_count = 0

    def _record_breakout(self, now_ms: int, mid: float) -> None:
        """遍历 4h 级别 × 双向检测破位：只记 pending（每方向每周期首次），不报警不落表。"""
        eps = settings.fake_breakout_eps
        cycle_id = now_ms // 900_000
        for level, lv in self._levels.items():
            for side, broken in (("high", mid > lv["resistance"] * (1.0 + eps)),
                                 ("low", mid < lv["support"] * (1.0 - eps))):
                if broken and side not in self._pending_breaks:
                    self._pending_breaks[side] = {
                        "cycle_id": cycle_id,
                        "level": level,
                        "broken_level": lv["resistance"] if side == "high" else lv["support"],
                        "break_price": mid,
                        "break_time": now_ms,
                    }
                    logger.debug(
                        "破位记 pending [{} {}] | BTC {:.0f} 破 {:.0f} | 周期 {}",
                        level, side, mid,
                        lv["resistance"] if side == "high" else lv["support"], cycle_id,
                    )

    async def _on_cycle_boundary(self, prev_cycle: int, cur_cycle: int, now_ms: int) -> None:
        """15m 周期边界：取出上一周期的 pending 破位，做收盘质量确认。

        冷启动边界（_last_cycle_id 跳变超过 1 个周期）不做确认——停机期间的
        pending 无法区分真实时序，且次周期早已走远，入场价假设失效。
        """
        due = {
            side: rec for side, rec in self._pending_breaks.items()
            if rec["cycle_id"] == prev_cycle
        }
        # 清理过期 pending（只保留当前及未来周期的记录）
        self._pending_breaks = {
            side: rec for side, rec in self._pending_breaks.items()
            if rec["cycle_id"] >= cur_cycle
        }
        if cur_cycle - prev_cycle > 1:
            logger.info("跨周期边界跳变（{} → {}），停机期间 pending 不确认", prev_cycle, cur_cycle)
            return
        if not due:
            return
        await self._confirm_and_fire(due, prev_cycle, cur_cycle, now_ms, retry=0)

    async def _drain_confirm_retries(self, now_ms: int) -> None:
        """处理确认重试队列：klines 之前拉取失败的周期收盘确认。"""
        if not self._confirm_retries:
            return
        retries, self._confirm_retries = self._confirm_retries, []
        for item in retries:
            if now_ms - item["at"] > CONFIRM_RETRY_TIMEOUT_MS:
                logger.warning(
                    "场景确认超时放弃 | 周期 {} 破位 {}（次周期已走远，入场价假设失效）",
                    item["prev_cycle"], "/".join(item["due"].keys()),
                )
                continue
            await self._confirm_and_fire(
                item["due"], item["prev_cycle"], item["cur_cycle"], now_ms, item["retry"]
            )

    async def _confirm_and_fire(
        self,
        due: dict[str, dict],
        prev_cycle: int,
        cur_cycle: int,
        now_ms: int,
        retry: int,
    ) -> None:
        """拉上一周期 15m K + 均量，判定收盘质量，命中场景则 fire 信号。"""
        klines = await self._collector.fetch_recent_klines("15m", VOL_MA_WINDOW + 1)
        sig_open_time = prev_cycle * 900_000
        sig_k = next((k for k in klines if k["open_time"] == sig_open_time), None)
        if sig_k is None:
            # 周期刚收盘 klines 可能尚未就绪：进重试队列（上限 5 次 / 60s）
            if retry < CONFIRM_RETRY_MAX:
                self._confirm_retries.append({
                    "due": due, "prev_cycle": prev_cycle, "cur_cycle": cur_cycle,
                    "retry": retry + 1, "at": now_ms,
                })
            else:
                logger.warning(
                    "场景确认放弃：klines 拉不到上一周期 K | 周期 {}", prev_cycle,
                )
            return

        hist = [k for k in klines if k["open_time"] < sig_open_time]
        vol_ma: float | None = None
        if len(hist) >= VOL_MA_WINDOW // 2:
            vol_ma = sum(k["volume"] for k in hist[-VOL_MA_WINDOW:]) / len(hist[-VOL_MA_WINDOW:])

        for side, rec in due.items():
            ok, close_pos, vol_ratio = classify_close_pattern(
                side, sig_k["open"], sig_k["high"], sig_k["low"], sig_k["close"],
                sig_k["volume"], vol_ma,
            )
            if not ok:
                logger.debug(
                    "收盘质量未命中 [{} {}] | 收盘位置 {} 量比 {}",
                    rec["level"], side,
                    f"{close_pos:.3f}" if close_pos is not None else "N/A",
                    f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A",
                )
                continue
            await self._fire_confirmed_signal(
                side, rec, sig_k, close_pos, vol_ratio, cur_cycle, now_ms
            )

    async def _fire_confirmed_signal(
        self,
        side: str,
        rec: dict,
        sig_k: dict,
        close_pos: float,
        vol_ratio: float | None,
        cur_cycle: int,
        now_ms: int,
    ) -> None:
        """场景命中信号：冷却/日限检查、落表与推送（目标周期 = 次周期 cur_cycle）。"""
        eps = settings.fake_breakout_eps
        level = rec["level"]
        pattern = "bull_exhaust" if side == "high" else "bear_exhaust"
        key = (side, level)

        # 风控 1：冷却（双保险；pending 每周期覆盖已保证每方向每周期最多一条）
        if now_ms - self._last_signal_at.get(key, 0) < settings.fake_breakout_cooldown_seconds * 1000:
            return

        self._daily_rollover(now_ms)
        # 风控 2：日内上限（超限仍落表，但不发邮件）
        over_daily_limit = self._daily_count >= settings.fake_breakout_max_daily_signals

        # 目标周期 = 次周期（时间网格推算，坐标必然可得，不走 EXPIRED 路径）
        next_start = cur_cycle * 900_000
        next_end = next_start + 900_000
        buffer_ms = settings.fake_breakout_settle_buffer_seconds * 1000
        settle_deadline = next_end + buffer_ms

        # 次周期开盘价快照（cycle_open_end 配对守卫，防 tracker 跨轮错配；
        # 缺失不阻塞——结算时 klines 精确回读兜底）
        open_15m = self._pm_15m.get("cycle_open_price")
        open_15m_end = self._pm_15m.get("cycle_open_end")
        m_open_15m = (
            float(open_15m)
            if open_15m and open_15m_end is not None and int(open_15m_end) == next_end
            else None
        )

        signal = FakeBreakoutSignal(
            level=level,
            side=side,
            signal_time=now_ms,
            resistance=rec["broken_level"],
            btc_price=self._collector.store.mid_price or sig_k["close"],
            eps=eps,
            down_price_15m=self._pm_15m.get("down_price"),
            up_price_15m=self._pm_15m.get("up_price"),
            market_end_15m=next_end,
            market_start_15m=next_start,
            cycle_open_price_15m=m_open_15m,
            pattern=pattern,
            close_pos=round(close_pos, 4),
            vol_ratio=round(vol_ratio, 3) if vol_ratio is not None else None,
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
            logger.error("场景信号落表失败 | {}", exc)
            return

        self._last_signal_at[key] = now_ms
        self._daily_count += 1

        direction = "DOWN" if side == "high" else "UP"
        logger.info(
            "场景信号触发 #{} [{} {}] | 周期 {} 破{} {:.0f} | 信号K收{} 收盘位置 {:.2f} 量比 {} | "
            "次周期看 {} | 日内第 {} 条{}",
            signal.id, level, pattern, rec["cycle_id"],
            "阻力" if side == "high" else "支撑", rec["broken_level"],
            "阳" if sig_k["close"] > sig_k["open"] else "阴", close_pos,
            f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A",
            direction, self._daily_count,
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
            logger.warning("场景信号邮件后台发送异常 #{} | {}", signal_id, exc)

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
        if signal.pattern == "bull_exhaust":
            pattern_label = "场景①多头耗尽"
            backtest = "180 天验证集胜率 63.6% [59.3,68.0] n=462，EV@0.50 +0.22/事件；按月最差 54%"
            entry_plan = (
                "次周期开盘即买 DOWN，半仓 @~0.50；\n"
                "  若次周期内价格反弹至 +0.10%（相对次周期开盘），加仓半仓 @~0.27；\n"
                "  不要深等（次周期前 10 分钟仍在涨 = 场景已被证伪，胜率崩至 4~14%）"
            )
        elif signal.pattern == "bear_exhaust":
            pattern_label = "场景②空头耗尽"
            backtest = "180 天验证集胜率 57.8% [53.5,62.1] n=512，EV@0.50 +0.11/事件"
            entry_plan = (
                "次周期开盘即买 UP，只开盘买；\n"
                "  跌态中 UP token 无折扣（0.79~0.90），等待无价值"
            )
        else:
            pattern_label = "旧信号"
            backtest = "—"
            entry_plan = "—"
        subject = (
            f"[场景信号·{pattern_label}] BTC 破4h{level_name}后收盘确认 → 次周期看 {direction}"
        )
        close_pos_str = (
            f"收盘质量：收盘位置 (C-L)/(H-L) = {signal.close_pos:.3f}"
            if signal.close_pos is not None else "收盘质量：收盘位置 N/A"
        )
        vol_str = (
            f"量比（本周期/前20根均量）= {signal.vol_ratio:.2f}"
            if signal.vol_ratio is not None else "量比：N/A"
        )
        body = (
            f"确认时间：{t_str}\n"
            f"场景：{pattern_label}（pattern={signal.pattern or 'N/A'}）\n"
            f"破位：15m 周期内{'冲过' if is_high else '跌破'} 4h {level_name} {signal.resistance:.2f}\n"
            f"{close_pos_str}\n"
            f"{vol_str}\n\n"
            f"目标周期：下一个 15m 市场（到期 {end_str}）\n"
            f"当时 {direction} token 报价：{entry_15m}\n\n"
            f"入场方案：\n  {entry_plan}\n\n"
            f"回测依据（scripts/local_continuation_discovery.py，发现集→验证集盲验）：\n"
            f"  {backtest}\n"
            f"机制：破位动能收盘未回吐（买力/卖力耗尽），次周期兑现反转。\n"
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
            logger.warning("场景信号 5m 结算查询失败 | {}", exc)
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
                        "场景信号 5m 周期结算 #{} | 周期开盘 {:.0f} → 周期末 {:.0f} | {}",
                        row.id, anchor, close_price, row.settle_outcome_5m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("场景信号 5m 结算回填失败 | {}", exc)

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
            logger.warning("场景结算查询失败 | {}", exc)
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
                                "场景信号 #{} 周期坐标缺失（15m 报价快照落空），置 EXPIRED",
                                row.id,
                            )
                    await session.commit()
            except Exception as exc:
                logger.warning("场景信号 EXPIRED 回填失败 | {}", exc)
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
                        "场景信号 15m 周期结算 #{} | 周期开盘 {:.0f} → 周期末 {:.0f} | {} | 入场价 {}",
                        row.id, anchor, close_price, row.settle_outcome, row.down_price_15m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("场景结算回填失败 | {}", exc)

    # ==================================================================
    # 只读状态（供 API 查询）
    # ==================================================================

    @property
    def status_snapshot(self) -> dict:
        return {
            "running": self._running,
            "levels": self._levels,
            "pending_breaks": {
                side: {"cycle_id": rec["cycle_id"], "broken_level": rec["broken_level"]}
                for side, rec in self._pending_breaks.items()
            },
            "confirm_retries": len(self._confirm_retries),
            "last_cycle_id": self._last_cycle_id,
            "daily_count": self._daily_count,
            "daily_date": self._daily_date,
            "eps": settings.fake_breakout_eps,
            "cooldown_seconds": settings.fake_breakout_cooldown_seconds,
            "max_daily_signals": settings.fake_breakout_max_daily_signals,
        }
