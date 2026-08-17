"""
场景信号系统：4h 位势破位检测 + 15m 周期收盘质量确认 + 次周期入场提醒。

模式升级（2026-08-15 拍板）：旧 A+B 过滤方案（破位瞬间当周期行动）整体替换为
场景信号系统；2026-08-17 升级为 360 天真 OOS 修正版（scripts/local_full_history_discovery.py，
前 240 天发现集 → 后 120 天验证集盲验，evaluate 切片 bug 修正后的口径）：

场景① bull_exhaust = F22×F18×F25（多头耗尽，真 OOS 64.4% [58.8,69.5] n=303，EV@0.51 +0.237）：
  15m 周期内刺破 4h 阻力位势 + 周期收盘收阳 + 光头收盘（(C-L)/(H-L) ≥ 0.85）
  + 4h 区间上沿（收盘在最近 16 根 15m 收盘区间位置 pos4h ≥ 0.9，F25）
  → 次周期开盘买 DOWN。入场方案：开盘半仓 @~0.50 + 次周期内反弹至 +0.10% 加仓半仓 @~0.27
    （180 天回测：开盘全仓 EV/事件 +0.223，触价加仓 +0.270，深等 z≥1 为负 EV 陷阱）

场景② bear_exhaust = F20×放量（空头耗尽，真 OOS 53.6% n=1097，EV@0.51 +0.066）：
  15m 周期内跌破 4h 支撑位势 + 周期收盘收阴 + 放量（量 ≥ 2× 前 20 根 15m 均量）
  → 次周期开盘买 UP。入场方案：只开盘买——跌态中 UP token 无折扣（0.79~0.90），等待无价值

场景④ momentum_fade = F40×F06（动量衰竭，真 OOS 55.4% [50.7,60.0] n=433，EV@0.51 +0.065）：
  连阳 ≥ 3 根（含信号 K 本身）+ 信号 K 光头阳（(C-L)/(H-L) ≥ 0.85），无破位要求
  → 次周期开盘买 DOWN。仅开盘入场，无加仓方案（EV 薄，边缘场景）
  （F40×F04 大实体组合 OOS 50.5% FAILED——不带大实体条件；
   S3/F23「S1+24h新高」强化标记 2026-08-17 拍板不上线，与 S1 重叠度高 n=118⊂303）

场景⑤ bull_exhaust_confirm = S1+5m 确认（2026-08-18 上线，360 天双周期结合回测）：
  S1 信号后不立即行动；等次周期第 1 根 5m K 收盘，若收盘价 < 次周期开盘价
  （z5 < 0，回测 build_events 口径 nop = 次周期第 1 根 5m 开盘价）
  → 确认买 DOWN，持有至同一次周期 15m 到期结算；反向上涨则放弃
  回测（360 天）：确认组 n=591 胜率 78.5% [75.0,81.6]（对照组 60.9%，
  反向上涨组 34.0%）；盈亏平衡入场价 0.77——真实 +5min 市场报价
  （quote5m_down_15m 快照）预计 0.6~0.7，EV 约 +0.1~+0.28，不劣于开盘入基准 +0.171

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
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import desc, select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import FakeBreakoutSignal, SceneParamVersion, SentimentWindow
from . import clock_sync
from .alerting import send_plain_email
from .data_collector import BinanceDataCollector
from .scene_params import DEFAULT_SCENE_PARAMS, SceneParams

# 超宽限阈值：到期后超过此宽限未结算的信号转 klines 精确补结算路径
# （周期锚点口径下 P(S)/P(E) 均为历史时点，klines 必然可得，停机无损）
SETTLE_EXPIRE_GRACE_MS = 300_000

# 级别定义（与回测 scripts/local_combo_filter_lab.py 对齐）：
# 级别名 → 回看窗口数（5m 窗口 closes 极值）。仅保留统计显著的 4h（2026-08-15 收窄）
# （M4 起：常量由 DEFAULT_SCENE_PARAMS 派生，版本化参数见 scene_param_versions）
LEVEL_LOOKBACKS: dict[str, int] = dict(DEFAULT_SCENE_PARAMS.level_lookbacks)

# 场景定义（360 天真 OOS 修正版，scripts/local_full_history_discovery.py 终验）：
# S1 bull_exhaust = F22×F18×F25：破4h阻力 + 收阳 + 光头(close_pos≥0.85) + 4h区间上沿(pos4h≥0.9)
#     → DOWN（真 OOS 64.4% [58.8,69.5] n=303 EV+0.237 Kelly 0.257）
# S2 bear_exhaust：破4h支撑 + 收阴 + 放量(vol_ratio≥2.0)
#     → UP（真 OOS 53.6% n=1097 EV+0.066）
# S4 momentum_fade = F40×F06：连阳≥3(含信号K) + 光头(close_pos≥0.85)，无破位要求
#     → DOWN（真 OOS 55.4% [50.7,60.0] n=433 EV+0.065 Kelly 0.071）
#     （F40×F04 大实体版 OOS 50.5% FAILED 不采用；S3/F23 强化版不上线，见文件头）
CLOSE_POS_MIN = DEFAULT_SCENE_PARAMS.close_pos_min   # S1/S4: 收盘位置下限 (>=0.85 光头)
VOL_RATIO_MIN = DEFAULT_SCENE_PARAMS.vol_ratio_min   # S2: 量比下限 (>=2.0 放量)
VOL_MA_WINDOW = DEFAULT_SCENE_PARAMS.vol_ma_window   # S2: 均量窗口 (前 20 根不含当前)
STREAK_BULL_MIN = 3                                  # S4: 连阳最小根数（含信号K本身，回测 streak 口径）
POS4H_WINDOW = 16                                    # S1: 4h 区间窗口（16 根 15m 收盘，含当前根）
POS4H_MIN = 0.9                                      # S1: F25 4h 区间上沿阈值
CONFIRM_RETRY_MAX = 5
CONFIRM_RETRY_TIMEOUT_MS = 60_000

# 入场报价快照 (次周期开盘后延迟抓取真实 15m 市场报价，替代理论 @0.50 假设)
ENTRY_SNAPSHOT_DELAY_MS = 8_000
ENTRY_SNAPSHOT_RETRY_INTERVAL_S = 5
ENTRY_SNAPSHOT_MAX_WAIT_MS = 90_000
ADD_TRIGGER_PCT = 0.001                              # S1/S3加仓触发 (mid ≥ open×(1+0.1%))
ADD_MONITOR_INTERVAL_S = 10

# S5 确认入场（2026-08-18）：S1 信号后 +5min 回落确认才买 DOWN
S5_CONFIRM_DELAY_MS = 300_000                        # 次周期第 1 根 5m 收盘时刻
S5_CONFIRM_GRACE_MS = 8_000                          # 收盘后缓冲（等 klines 落库）
S5_CONFIRM_MAX_WAIT_MS = 90_000                      # 5m K 拉不到的放弃上限
QUOTE_5M_DELAY_MS = 300_000                          # +5min 报价快照时点（与 S5 确认同时刻）

# 交易定价常数 (与 EV 计算一致：费 2%+0.01 溢价，赔率 b≈0.922，打平胜率≈52.0%)
FEE = 0.02
PREMIUM = 0.01
ODDS = (1 - FEE) / (0.50 + PREMIUM) - 1.0            # 赔率 b
BREAKEVEN = 1.0 / (1.0 + ODDS)                        # 打平胜率 (EV=0 临界点)

# 各场景真 OOS 胜率点估计（ev_at_entry = p×(1-FEE)/entry − 1 的 p；结算统计/邮件引用）
RESEARCH_WIN_RATES: dict[str, float] = {
    "bull_exhaust": 0.644,
    "bear_exhaust": 0.536,
    "momentum_fade": 0.554,
    # S5 = S1 子集“+5min 已回落”组（360 天双周期结合回测 2026-08-18）
    "bull_exhaust_confirm": 0.785,
}
# pattern_type → pattern 旧列粗类映射（String(16) 上限；旧列保留兼容历史查询）
PATTERN_GROUP: dict[str, str] = {
    "bull_exhaust": "bull_exhaust",
    "bear_exhaust": "bear_exhaust",
    "momentum_fade": "momentum_fade",
    "bull_exhaust_confirm": "bull_exhaust",
}


def classify_close_pattern(
    side: str,
    o: float,
    h: float,
    l: float,
    c: float,
    volume: float,
    vol_ma: float | None,
    pos4h: float | None = None,
    params: SceneParams | None = None,
) -> tuple[bool, str | None, float | None, float | None]:
    """信号周期收盘质量判定——破位侧场景 S1/S2（纯函数，邮件/API/测试/影子并行共用的单一事实源）。

    Args:
        side: 破位方向（high=破阻力→S1 | low=破支撑→S2）
        o/h/l/c: 信号周期 15m K 线 OHLC
        volume: 信号周期 15m 成交量
        vol_ma: 前 20 根 15m 均量（不含当前；None/0 = 数据不足）
        pos4h: 收盘价在最近 16 根（含当前根）15m 收盘区间中的位置 (c-min)/(max-min)
               —— S1 的 F25「4h 区间上沿」条件；None = 历史不足（保守判 False）
        params: 场景参数集（M4 影子并行注入；None = DEFAULT_SCENE_PARAMS）

    Returns:
        (命中, pattern_type, close_pos, vol_ratio)
        - pattern_type: "bull_exhaust" | "bear_exhaust" | None
    """
    p = params or DEFAULT_SCENE_PARAMS
    rng = h - l
    if rng <= 0 or o <= 0:
        return False, None, None, None

    close_pos = (c - l) / rng
    vol_ratio = volume / vol_ma if vol_ma and vol_ma > 0 else None

    if side == "high":
        # S1 bull_exhaust = F22×F18×F25：收阳 + 光头 + 4h 区间上沿
        is_s1 = (
            (c > o)
            and (close_pos >= p.close_pos_min)
            and (pos4h is not None)
            and (pos4h >= POS4H_MIN)
        )
        if is_s1:
            return True, "bull_exhaust", close_pos, vol_ratio
    else:
        # S2 bear_exhaust：收阴 + 放量
        is_s2 = (c < o) and (vol_ratio is not None) and (vol_ratio >= p.vol_ratio_min)
        if is_s2:
            return True, "bear_exhaust", close_pos, vol_ratio

    return False, None, None, None


def is_momentum_fade(
    o: float,
    h: float,
    l: float,
    c: float,
    prev_dir: list[int] | None,
    params: SceneParams | None = None,
) -> tuple[bool, float | None]:
    """S4 momentum_fade 独立判定（F40 连阳≥3 × F06 光头阳，无破位要求）。

    连阳≥3 含信号 K 本身（回测 streak 口径）：信号 K 收阳 + 前 STREAK_BULL_MIN-1 根均收阳。
    大实体条件不采用（F40×F04 组合 OOS 50.5% FAILED）。
    调用方须仅在无 high 侧破位 pending 的周期检查（破位周期命中 S1 时优先 S1，
    避免同一周期重复计数；S4 事件与 S1 重叠部分在实盘中被 S1 覆盖，属预期取舍）。

    Returns:
        (命中, close_pos)
    """
    p = params or DEFAULT_SCENE_PARAMS
    rng = h - l
    if rng <= 0 or o <= 0:
        return False, None
    need = STREAK_BULL_MIN - 1  # 信号 K 之前需要的连阳根数
    streak_ok = (
        prev_dir is not None
        and len(prev_dir) >= need
        and all(d == 1 for d in prev_dir[-need:])
    )
    close_pos = (c - l) / rng
    hit = (c > o) and (close_pos >= p.close_pos_min) and streak_ok
    return hit, close_pos if hit else None


def confirm_bull_exhaust_5m(c5_close: float, anchor: float) -> tuple[bool, str]:
    """S5 确认判定（纯函数）：次周期第 1 根 5m 收盘价 vs 次周期开盘价。

    回测口径（backtest/events.py 的 z5）：d1 = c5[nidx[0]][4] / nop − 1，
    nop = 次周期第 1 根 5m 开盘价（= 次周期开盘价 P(S)）；确认 ⟺ z5 < 0。
    锚点与 15m 结算同源（P(S)），确认组胜率 78.5%、反向上涨组 34.0%。

    Args:
        c5_close: 次周期第 1 根 5m K 的收盘价
        anchor: 次周期开盘价（父信号 cycle_open_price_15m，或该 5m K 的 open）

    Returns:
        (confirmed, reason)：reason ∈ CONFIRM | RISING（反向上涨，放弃）|
        FLAT（平价 NOISE，放弃）| INVALID（数据异常，放弃）
    """
    if not anchor or anchor <= 0 or not c5_close or c5_close <= 0:
        return False, "INVALID"
    if c5_close < anchor:
        return True, "CONFIRM"
    if c5_close > anchor:
        return False, "RISING"
    return False, "FLAT"


def compute_pattern_stats(rows: list) -> dict:
    """按 pattern_type 已结算正式信号计算实盘统计（结算回填与 stats API 共用的纯函数）。

    口径：每笔 1 USDT 本金；entry 价取入场报价快照（按方向选 DOWN/UP），
    缺失回退 0.51（含溢价理论价）；单笔实现收益 = 赢 (1-FEE)/entry-1 / 输 -1。

    Returns:
        {n, wins, winrate, cumulative_ev, avg_ev_at_entry,
         equity_curve, peak_equity, max_drawdown}
    """
    wins = 0
    rets: list[float] = []
    ev_entries: list[float] = []
    for row in rows:
        entry = row.entry_down_price_15m if row.side == "high" else row.entry_up_price_15m
        entry = float(entry) if entry and entry > 0 else 0.50 + PREMIUM
        ret = (1.0 - FEE) / entry - 1.0
        won = row.settle_outcome == ("DOWN" if row.side == "high" else "UP")
        if won:
            wins += 1
            rets.append(ret)
        else:
            rets.append(-1.0)
        p = RESEARCH_WIN_RATES.get(row.pattern_type or "")
        if p is not None:
            ev_entries.append(p * (1.0 + ret) - 1.0)
    n = len(rows)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    curve: list[float] = []
    for r in rets:
        cum += r
        curve.append(round(cum, 6))
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return {
        "n": n,
        "wins": wins,
        "winrate": wins / n if n else None,
        "cumulative_ev": cum / n if n else None,
        "avg_ev_at_entry": sum(ev_entries) / len(ev_entries) if ev_entries else None,
        "equity_curve": curve,
        "peak_equity": peak,
        "max_drawdown": max_dd,
    }


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
        # 累计观测计数（进程生命周期内，供 status API 确认系统在干活）：
        # - _pending_count: 记过的破位 pending 总数（含未命中形态的）
        # - _confirm_miss_count: 周期收盘确认时形态未命中的次数
        self._pending_count: int = 0
        self._confirm_miss_count: int = 0
        # M4 影子并行：ACTIVE 版本名 + 可影子判定的 SHADOW 版本列表
        # （[{"version", "params": SceneParams}]；仅 classify 层参数差异的版本）
        self._active_version: str = "v1"
        self._shadow_versions: list[dict] = []

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._refresh_levels(force=True)
        await self._load_scene_versions()
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
                    await self._load_scene_versions()  # 每周期边界重查版本（轻量）
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
                    self._pending_count += 1
                    logger.info(
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
        # due 为空也要跑收盘检查：S4 momentum_fade 无破位要求，每周期独立判定
        # （_confirm_and_fire 内部对空 due 只走 S4 路径，不误判 S1/S2）
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

    async def _load_scene_versions(self) -> None:
        """加载 ACTIVE + 可影子判定的 SHADOW 版本（M4）。

        无 ACTIVE 行（create_all 环境/迁移未跑）→ 回退 v1 默认参数，
        服务层必须容忍无种子行（M0 测试已声明）。
        """
        from .scene_params import is_shadow_supported
        try:
            async with async_session_factory() as session:
                from sqlalchemy import select as sa_select
                stmt = sa_select(SceneParamVersion).where(
                    SceneParamVersion.status.in_(["ACTIVE", "SHADOW"])
                )
                rows = (await session.execute(stmt)).scalars().all()
        except Exception as exc:
            logger.debug("场景版本加载失败（回退 v1 默认）| {}", exc)
            return
        active = [r for r in rows if r.status == "ACTIVE"]
        if active:
            row = max(active, key=lambda r: r.activated_at or r.created_at)
            self._active_version = row.version
            base_json = dict(row.params)
        else:
            self._active_version = "v1"
            base_json = DEFAULT_SCENE_PARAMS.to_params_json()
        shadows = []
        for r in rows:
            if r.status != "SHADOW":
                continue
            if is_shadow_supported(dict(r.params), base_json):
                shadows.append({"version": r.version, "params": SceneParams.from_params_json(r.params)})
            else:
                logger.warning(
                    "SHADOW 版本 {} 含破位层参数差异（eps/lookback），影子层跳过"
                    "（实盘判定与回测口径一致性保护）", r.version,
                )
        if shadows != self._shadow_versions:
            self._shadow_versions = shadows
            logger.info(
                "场景版本加载 | ACTIVE={} | SHADOW={}",
                self._active_version, [s["version"] for s in shadows],
            )

    async def _confirm_and_fire(
        self,
        due: dict[str, dict],
        prev_cycle: int,
        cur_cycle: int,
        now_ms: int,
        retry: int,
    ) -> None:
        """拉上一周期 15m K + 均量，判定收盘质量，命中场景则 fire 信号。

        除 due 中的破位侧场景（S1/S2）外，还独立检查 S4 momentum_fade
        （连阳≥3×光头，无破位要求；仅在无 high 侧 pending 时——破位周期
        S1 优先，避免同一周期重复计数）。
        """
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

        # 信号 K 之前的历史（升序）：均量 / 4h 区间位置 / 连阳序列
        hist = [k for k in klines if k["open_time"] < sig_open_time]
        vol_ma = (
            sum(k["volume"] for k in hist[-VOL_MA_WINDOW:]) / VOL_MA_WINDOW
            if len(hist) >= VOL_MA_WINDOW else None
        )
        # pos4h：收盘在最近 16 根（含当前根）15m 收盘区间中的位置
        # （回测 F25 口径 roll_max/roll_min(c15,16) 含当前根）
        pos4h = None
        win = [k["close"] for k in hist[-(POS4H_WINDOW - 1):]] + [sig_k["close"]]
        if len(win) >= POS4H_WINDOW:
            hi_, lo_ = max(win), min(win)
            if hi_ > lo_:
                pos4h = (sig_k["close"] - lo_) / (hi_ - lo_)
        # 连阳序列（信号 K 之前各根方向，升序；S4 取最后 STREAK_BULL_MIN-1 根）
        prev_dir = [
            1 if k["close"] > k["open"] else (-1 if k["close"] < k["open"] else 0)
            for k in hist
        ]

        for side, rec in due.items():
            # 主场景判定（ACTIVE 版本）
            ok, pattern_type, close_pos, vol_ratio = classify_close_pattern(
                side, sig_k["open"], sig_k["high"], sig_k["low"], sig_k["close"],
                sig_k["volume"], vol_ma, pos4h=pos4h,
            )
            if ok:
                await self._fire_confirmed_signal(
                    side, rec, sig_k, close_pos, vol_ratio, cur_cycle, now_ms,
                    version=self._active_version, shadow=False,
                    pattern_type=pattern_type or "bull_exhaust",
                )
            else:
                self._confirm_miss_count += 1
                logger.info(
                    "收盘质量未命中 [{} {}] | 收盘位置 {} 量比 {} pos4h {}",
                    rec["level"], side,
                    f"{close_pos:.3f}" if close_pos is not None else "N/A",
                    f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A",
                    f"{pos4h:.2f}" if pos4h is not None else "N/A",
                )
            # M4 影子并行
            for sv in self._shadow_versions:
                s_ok, s_pt, s_cp, s_vr = classify_close_pattern(
                    side, sig_k["open"], sig_k["high"], sig_k["low"], sig_k["close"],
                    sig_k["volume"], vol_ma, pos4h=pos4h, params=sv["params"],
                )
                if s_ok:
                    await self._fire_confirmed_signal(
                        side, rec, sig_k, s_cp, s_vr, cur_cycle, now_ms,
                        version=sv["version"], shadow=True,
                        pattern_type=s_pt or "bull_exhaust",
                    )

        # S4 momentum_fade 独立检查：仅无 high 侧破位 pending 的周期执行
        # （破位周期若 K 同时满足 S4 定义必为 S1 子集，已被上面优先处理）
        if "high" not in due:
            m_hit, m_close_pos = is_momentum_fade(
                sig_k["open"], sig_k["high"], sig_k["low"], sig_k["close"], prev_dir,
            )
            if m_hit:
                # 合成 rec：S4 无位势破位，level 固定 momentum（8 字符）区分索引统计；
                # broken_level 记信号 K 高点仅作审计参考
                synthetic_rec = {
                    "cycle_id": prev_cycle,
                    "level": "momentum",
                    "broken_level": sig_k["high"],
                    "break_price": sig_k["close"],
                    "break_time": sig_open_time,
                }
                await self._fire_confirmed_signal(
                    "high", synthetic_rec, sig_k, m_close_pos, None, cur_cycle, now_ms,
                    version=self._active_version, shadow=False,
                    pattern_type="momentum_fade",
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
        version: str = "v1",
        shadow: bool = False,
        pattern_type: str = "bull_exhaust",  # bull_exhaust | bear_exhaust | momentum_fade
    ) -> None:
        """场景命中信号：冷却/日限检查、落表与推送（目标周期 = 次周期 cur_cycle）。

        M4：version 标记参数版本；shadow=True 时只落表不发邮件、不计日限、
        独立冷却键——影子信号仅用于实盘对照，不影响正式信号流。
        
        pattern_type：具体模式类型（由 classify_close_pattern 返回）
        """
        eps = settings.fake_breakout_eps
        level = rec["level"]
        key = (side, level, version)

        # 风控 1：冷却（双保险；pending 每周期覆盖已保证每方向每周期最多一条）
        if now_ms - self._last_signal_at.get(key, 0) < settings.fake_breakout_cooldown_seconds * 1000:
            return

        self._daily_rollover(now_ms)
        # 风控 2：日内上限（超限仍落表，但不发邮件）；影子信号不占日限
        over_daily_limit = (not shadow) and self._daily_count >= settings.fake_breakout_max_daily_signals

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
            pattern=PATTERN_GROUP.get(pattern_type, "bull_exhaust"),
            pattern_type=pattern_type,
            close_pos=round(close_pos, 4),
            vol_ratio=round(vol_ratio, 3) if vol_ratio is not None else None,
            version=version,
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
        if not shadow:
            self._daily_count += 1

        # 入场报价快照（fire-and-forget，与邮件后台任务同模式）：
        # 次周期开盘后抓真实 15m 市场报价 + 场景①加仓触发监测。
        # 影子信号同样抓——影子对照必须与正式信号同口径才有可比性。
        asyncio.create_task(
            self._capture_entry_quote(signal.id, next_start, next_end, pattern_type, m_open_15m),
            name=f"fbs_entry_{signal.id}",
        )

        # S5 确认入场（2026-08-18）：仅正式 S1 信号派生——+5min 回落确认才买 DOWN；
        # 影子不派生（影子对照仅覆盖收盘判定层，S5 属入场时机层）
        if not shadow and pattern_type == "bull_exhaust":
            asyncio.create_task(
                self._confirm_s5_entry(signal.id, next_start, next_end),
                name=f"fbs_s5_{signal.id}",
            )

        direction = "DOWN" if side == "high" else "UP"
        logger.info(
            "场景信号触发 #{} [{} {}{}] | 周期 {} 破{} {:.0f} | 信号K收{} 收盘位置 {:.2f} 量比 {} | "
            "次周期看 {} | 日内第 {} 条{}",
            signal.id, level, pattern_type, f"·{version}" if version != "v1" else "", rec["cycle_id"],
            "阻力" if side == "high" else "支撑", rec["broken_level"],
            "阳" if sig_k["close"] > sig_k["open"] else "阴", close_pos,
            f"{vol_ratio:.2f}" if vol_ratio is not None else "N/A",
            direction, self._daily_count,
            "（影子，不发邮件）" if shadow else ("（超日限，不发邮件）" if over_daily_limit else ""),
        )

        # 邮件推送（未超日限且非影子）：fire-and-forget，绝不阻塞检测循环。
        # 实测教训：SMTP 连接被防火墙丢包时同步等待会卡死整个循环 16 分钟，
        # 导致结算回读停摆（信号 #1 事故）。
        if not shadow and settings.fake_breakout_email_enabled and not over_daily_limit:
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

    # ==================================================================
    # 入场报价快照（次周期开盘后延迟抓取，替代理论 @0.50 假设）
    # ==================================================================

    async def _sleep_until(self, target_ms: int) -> None:
        """睡到目标时刻（币安服务器时钟；分段睡防长阻塞 stop()）。"""
        while self._running:
            wait_s = (target_ms - clock_sync.now_ms()) / 1000
            if wait_s <= 0:
                return
            await asyncio.sleep(min(wait_s, 30))

    async def _update_signal(self, signal_id: int, **cols) -> bool:
        """回填信号列（与 email_sent 回填同模式）；失败不抛，返回 False。"""
        try:
            async with async_session_factory() as session:
                row = await session.get(FakeBreakoutSignal, signal_id)
                if row is None:
                    return False
                for k, v in cols.items():
                    setattr(row, k, v)
                await session.commit()
            return True
        except Exception as exc:
            logger.warning("信号列回填失败 #{} {} | {}", signal_id, list(cols), exc)
            return False

    async def _capture_entry_quote(
        self,
        signal_id: int,
        next_start: int,
        next_end: int,
        pattern_type: str | None,
        open_price: float | None,
    ) -> None:
        """入场报价快照 + 场景①加仓触发监测（次周期生命周期内的后台任务）。

        单循环同时处理三件事（时序可能重叠：开盘后 20s 内即可反弹触发加仓）：
        1. 入场快照：等 tracker 切到次周期市场（start_date 守卫，防旧市场残值），
           开盘后 ~20s 首试，+90s 截止，失败置 NULL（不阻塞结算）
        2. 加仓监测（仅 bull_exhaust）：mid ≥ 开盘价×(1+0.10%) 时抓报价落 add 列；
           周期结束未触发保持 NULL（= 未触发，本身是有效信息：@0.27 假设未兑现）
        3. +5min 报价快照（2026-08-18）：次周期 1/3 处的 15m 市场报价落 quote5m_* 列，
           S5 确认入场的真实可得价与提前离场定价对照；周期结束未抓到保持 NULL
        """
        try:
            is_s1 = pattern_type == "bull_exhaust"
            entry_done = False
            entry_abandoned = False
            entry_deadline = next_start + ENTRY_SNAPSHOT_MAX_WAIT_MS
            quote_done = False

            # 加仓触发价：开盘价缺失时 klines 回读一次；仍缺失则放弃监测
            trigger_price: float | None = None
            if is_s1:
                op = float(open_price) if open_price and open_price > 0 else 0.0
                if op <= 0:
                    op = await self._klines_open("15m", next_start)
                if op > 0:
                    trigger_price = op * (1 + ADD_TRIGGER_PCT)
                else:
                    logger.warning("加仓监测放弃 #{}：次周期开盘价不可得", signal_id)
            add_done = not is_s1 or trigger_price is None

            await self._sleep_until(next_start + ENTRY_SNAPSHOT_DELAY_MS)
            while self._running and clock_sync.now_ms() < next_end:
                now = clock_sync.now_ms()
                q = dict(self._pm_15m)
                matched = q.get("start_date") == next_start

                if not entry_done and not entry_abandoned:
                    if now > entry_deadline:
                        entry_abandoned = True
                        logger.warning(
                            "入场报价快照放弃 #{}（+{}s 内市场切换未确认）",
                            signal_id, ENTRY_SNAPSHOT_MAX_WAIT_MS // 1000,
                        )
                    elif matched:
                        ok = await self._update_signal(
                            signal_id,
                            entry_down_price_15m=q.get("down_price"),
                            entry_up_price_15m=q.get("up_price"),
                            entry_quote_ts_15m=int(q.get("updated_ts") or now),
                        )
                        if ok:
                            entry_done = True
                            logger.info(
                                "入场报价快照 #{} | DOWN {} UP {} | 开盘后 {}s",
                                signal_id, q.get("down_price"), q.get("up_price"),
                                (int(q.get("updated_ts") or now) - next_start) // 1000,
                            )

                if not add_done and matched:
                    mid = self._collector.store.mid_price
                    if mid and mid >= trigger_price:
                        ok = await self._update_signal(
                            signal_id,
                            add_down_price_15m=q.get("down_price"),
                            add_up_price_15m=q.get("up_price"),
                            add_trigger_ts_15m=now,
                        )
                        if ok:
                            add_done = True
                            logger.info(
                                "加仓触发快照 #{} | mid {:.0f} ≥ 触发价 {:.0f} | DOWN {} UP {}",
                                signal_id, mid, trigger_price,
                                q.get("down_price"), q.get("up_price"),
                            )

                # +5min 报价快照：S5 确认入场真实价 + 提前离场定价对照
                if not quote_done and now >= next_start + QUOTE_5M_DELAY_MS and matched:
                    if q.get("down_price") is not None:
                        ok = await self._update_signal(
                            signal_id,
                            quote5m_down_15m=q.get("down_price"),
                            quote5m_up_15m=q.get("up_price"),
                            quote5m_ts_15m=int(q.get("updated_ts") or now),
                        )
                        if ok:
                            quote_done = True
                            logger.info(
                                "+5min 报价快照 #{} | DOWN {} UP {}",
                                signal_id, q.get("down_price"), q.get("up_price"),
                            )

                if (entry_done or entry_abandoned) and add_done and quote_done:
                    return
                await asyncio.sleep(
                    ENTRY_SNAPSHOT_RETRY_INTERVAL_S if not entry_done else ADD_MONITOR_INTERVAL_S
                )
            # 周期自然结束：未触发即未触发（add 保持 NULL），无额外日志
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("入场报价快照任务异常 #{} | {}", signal_id, exc)

    # ==================================================================
    # S5 确认入场（2026-08-18）：S1 信号 +5min 回落确认才买 DOWN
    # ==================================================================

    async def _confirm_s5_entry(self, parent_id: int, next_start: int, next_end: int) -> None:
        """S5 确认判定：+5min 拉次周期第 1 根 5m K，回落确认则落 S5 信号。

        时序：睡到 next_start + 5min + 缓冲 → 重试循环（上限 90s）拉
        open_time == next_start 的 5m K（此时必已收盘）→ 按 confirm_bull_exhaust_5m
        判定（锚点 = 该 5m K 的 open，即次周期开盘价 P(S)，与回测 z5 的 nop 同源）。
        确认 → _fire_s5_signal 落行+邮件；反向上涨/平价 → 放弃（无 S5 行）；
        5m K 拉不到（停机/klines 异常）→ 放弃，不影响父 S1 行。
        """
        try:
            await self._sleep_until(next_start + S5_CONFIRM_DELAY_MS + S5_CONFIRM_GRACE_MS)
            deadline = next_start + S5_CONFIRM_DELAY_MS + S5_CONFIRM_MAX_WAIT_MS
            while self._running and clock_sync.now_ms() < min(deadline, next_end):
                try:
                    klines = await self._collector.fetch_recent_klines("5m", 3)
                except Exception:
                    klines = []
                k5 = next((k for k in klines if k["open_time"] == next_start), None)
                if k5 is None or clock_sync.now_ms() < next_start + S5_CONFIRM_DELAY_MS:
                    await asyncio.sleep(ENTRY_SNAPSHOT_RETRY_INTERVAL_S)
                    continue
                confirmed, reason = confirm_bull_exhaust_5m(
                    float(k5["close"]), float(k5["open"]),
                )
                if not confirmed:
                    logger.info(
                        "S5 确认放弃 #{}（{}）| 5m 收盘 {:.2f} vs 周期开盘 {:.2f} | 对照：反向上涨组胜率 34%",
                        parent_id, reason, k5["close"], k5["open"],
                    )
                    return
                await self._fire_s5_signal(
                    parent_id, next_start, next_end, float(k5["close"]), float(k5["open"]),
                )
                return
            if self._running:
                logger.warning(
                    "S5 确认放弃 #{}（+{}s 内次周期第 1 根 5m K 不可得）",
                    parent_id, S5_CONFIRM_MAX_WAIT_MS // 1000,
                )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("S5 确认任务异常 #{} | {}", parent_id, exc)

    async def _fire_s5_signal(
        self,
        parent_id: int,
        next_start: int,
        next_end: int,
        c5_close: float,
        anchor: float,
    ) -> None:
        """S5 确认命中：复制父 S1 行坐标落 pattern_type=bull_exhaust_confirm 新行。

        入场价 = 确认时刻 15m 市场 DOWN 报价（start_date 守卫，防旧市场残值；
        报价缺失置 NULL，统计回退 0.51 理论价）。结算与父行同周期同锚点
        （cycle_open_price_15m 缺失时用第 1 根 5m open 兜底，即 P(S)）。每父信号
        至多一条，天然去重；计入日限，邮件受日限控制（同正式信号风控）。
        """
        async with async_session_factory() as session:
            parent = await session.get(FakeBreakoutSignal, parent_id)
        if parent is None:
            return

        q = dict(self._pm_15m)
        matched = q.get("start_date") == next_start
        now_ms = clock_sync.now_ms()
        self._daily_rollover(now_ms)
        over_daily_limit = self._daily_count >= settings.fake_breakout_max_daily_signals

        signal = FakeBreakoutSignal(
            level=parent.level,
            side=parent.side,
            signal_time=now_ms,
            resistance=parent.resistance,
            btc_price=self._collector.store.mid_price or c5_close,
            eps=parent.eps,
            down_price_15m=q.get("down_price") if matched else None,
            up_price_15m=q.get("up_price") if matched else None,
            market_end_15m=parent.market_end_15m,
            market_start_15m=parent.market_start_15m,
            cycle_open_price_15m=(
                parent.cycle_open_price_15m
                if parent.cycle_open_price_15m and parent.cycle_open_price_15m > 0
                else anchor
            ),
            pattern=PATTERN_GROUP["bull_exhaust_confirm"],
            pattern_type="bull_exhaust_confirm",
            close_pos=parent.close_pos,
            vol_ratio=parent.vol_ratio,
            version=parent.version,
            # S5 入场即 +5min 确认时刻，entry 列语义对齐“入场时报价”
            entry_down_price_15m=q.get("down_price") if matched else None,
            entry_up_price_15m=q.get("up_price") if matched else None,
            entry_quote_ts_15m=int(q.get("updated_ts") or now_ms) if matched else None,
            settle_deadline=parent.settle_deadline,
            status="PENDING",
            email_sent=False,
        )
        try:
            async with async_session_factory() as session:
                session.add(signal)
                await session.commit()
                await session.refresh(signal)
        except Exception as exc:
            logger.error("S5 信号落表失败 #{} | {}", parent_id, exc)
            return

        self._daily_count += 1
        logger.info(
            "S5 确认信号触发 #{}（父 #{}）| 5m 收盘 {:.2f} < 周期开盘 {:.2f} | "
            "DOWN 入场 {} | 360 天回测：确认组 78.5% [75.0,81.6] 盈亏平衡 0.77 | 日内第 {} 条{}",
            signal.id, parent_id, c5_close, anchor,
            q.get("down_price") if matched else "N/A",
            self._daily_count,
            "（超日限，不发邮件）" if over_daily_limit else "",
        )

        if settings.fake_breakout_email_enabled and not over_daily_limit:
            asyncio.create_task(
                self._send_signal_email_bg(signal.id),
                name=f"fbs_email_{signal.id}",
            )

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
        pt = signal.pattern_type or signal.pattern
        if pt == "bull_exhaust":
            pattern_label = "场景①多头耗尽"
            backtest = "360 天真 OOS 胜率 64.4% [58.8,69.5] n=303，EV@0.51 +0.237/事件（F22×F18×F25）；按月最差 57%"
            entry_plan = (
                "次周期开盘即买 DOWN，半仓 @~0.50；\n"
                "  若次周期内价格反弹至 +0.10%（相对次周期开盘），加仓半仓 @~0.27；\n"
                "  不要深等（次周期前 10 分钟仍在涨 = 场景已被证伪，胜率崩至 4~14%）"
            )
        elif pt == "bear_exhaust":
            pattern_label = "场景②空头耗尽"
            backtest = "360 天真 OOS 胜率 53.6% n=1097，EV@0.51 +0.066/事件"
            entry_plan = (
                "次周期开盘即买 UP，只开盘买；\n"
                "  跌态中 UP token 无折扣（0.79~0.90），等待无价值"
            )
        elif pt == "momentum_fade":
            pattern_label = "场景④动量衰竭"
            backtest = "360 天真 OOS 胜率 55.4% [50.7,60.0] n=433，EV@0.51 +0.065/事件（F40连阳≥3×F06光头阳，无破位要求）"
            entry_plan = (
                "次周期开盘即买 DOWN，仅开盘入场；\n"
                "  EV 较薄（+0.065），观察为主，不加仓不深等"
            )
        elif pt == "bull_exhaust_confirm":
            pattern_label = "场景⑤确认入场（S1+5m 确认）"
            entry_d = signal.entry_down_price_15m
            ev_hint = (
                f"当前入场价 {entry_d:.3f} → 预期 EV {RESEARCH_WIN_RATES['bull_exhaust_confirm'] * (1 - FEE) / (entry_d + PREMIUM) - 1:+.3f}"
                if entry_d and entry_d > 0 else "入场价快照缺失（统计时回退 0.51 理论价）"
            )
            backtest = (
                "360 天双周期结合回测：S1 信号 +5min 回落确认组胜率 78.5% [75.0,81.6] n=591"
                "（全样本 60.9%；+5min 反向上涨组仅 34.0%）；\n"
                f"  盈亏平衡入场价 0.77，{ev_hint}"
            )
            entry_plan = (
                "已过 +5min 确认时点（次周期第 1 根 5m 收盘 < 周期开盘），现价买 DOWN 持有至本期到期；\n"
                "  若实际报价 > 0.77 放弃（负 EV）；\n"
                "  对照：父 S1 开盘入基准 EV@0.51 +0.171"
            )
        else:
            pattern_label = "旧信号"
            backtest = "—"
            entry_plan = "—"
        if pt == "momentum_fade":
            subject = f"[场景信号·{pattern_label}] BTC 连阳≥3+光头阳收盘确认 → 次周期看 {direction}"
        elif pt == "bull_exhaust_confirm":
            subject = f"[场景信号·{pattern_label}] BTC S1 信号后 5 分钟已回落 → 买 {direction}（胜率 78.5%）"
        else:
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
        break_line = (
            f"破位：15m 周期内{'冲过' if is_high else '跌破'} 4h {level_name} {signal.resistance:.2f}"
            if pt not in ("momentum_fade", "bull_exhaust_confirm")
            else (
                f"确认依据：父 S1 信号破 4h 阻力 {signal.resistance:.2f} 收盘确认后，"
                f"次周期第 1 根 5m 收盘低于周期开盘价（回落确认）"
                if pt == "bull_exhaust_confirm"
                else f"动量背景：连阳 ≥3 根 + 光头阳，信号 K 高点 {signal.resistance:.2f}（无破位要求）"
            )
        )
        body = (
            f"确认时间：{t_str}\n"
            f"场景：{pattern_label}（pattern_type={pt or 'N/A'}）\n"
            f"{break_line}\n"
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
        settled_pts: set[str] = set()  # 本次结算涉及的场景类型（结算后统一回填统计）
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
                    settled_pts.add(row.pattern_type or row.pattern or "")
                    logger.info(
                        "场景信号 15m 周期结算 #{} | 周期开盘 {:.0f} → 周期末 {:.0f} | {} | 入场价 {}",
                        row.id, anchor, close_price, row.settle_outcome, row.down_price_15m,
                    )
                await session.commit()
        except Exception as exc:
            logger.error("场景结算回填失败 | {}", exc)
            return
        # 统计维度回填（2026-08-17）：按本次结算涉及的场景类型更新累计指标
        for pt in sorted(settled_pts):
            if pt:
                await self._update_pattern_stats(pt)

    async def _update_pattern_stats(self, pattern_type: str) -> None:
        """结算后按 pattern_type 回填统计列（ev_at_entry 补齐 + 最新行累计指标）。

        仅正式信号（version NULL/v1，排除影子）；统计口径见 compute_pattern_stats。
        ev_at_entry 按入场报价与真 OOS 胜率点估计补算（快照缺失回退 0.51 理论价）。
        """
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(FakeBreakoutSignal)
                    .where(FakeBreakoutSignal.pattern_type == pattern_type)
                    .where(FakeBreakoutSignal.status == "SETTLED")
                    .where(
                        (FakeBreakoutSignal.version.is_(None))
                        | (FakeBreakoutSignal.version == "v1")
                    )
                    .order_by(FakeBreakoutSignal.signal_time)
                )
                rows = (await session.execute(stmt)).scalars().all()
                if not rows:
                    return
                # ev_at_entry 补齐（入场时刻预期 EV = p×(1-FEE)/entry − 1）
                p_research = RESEARCH_WIN_RATES.get(pattern_type)
                for row in rows:
                    if row.ev_at_entry is not None or p_research is None:
                        continue
                    entry = (
                        row.entry_down_price_15m if row.side == "high"
                        else row.entry_up_price_15m
                    )
                    entry = float(entry) if entry and entry > 0 else 0.50 + PREMIUM
                    row.ev_at_entry = round(p_research * (1.0 - FEE) / entry - 1.0, 6)
                stats = compute_pattern_stats(rows)
                latest = rows[-1]
                latest.cumulative_winrate = (
                    round(stats["winrate"], 4) if stats["winrate"] is not None else None
                )
                latest.cumulative_ev = (
                    round(stats["cumulative_ev"], 6)
                    if stats["cumulative_ev"] is not None else None
                )
                now_ms = clock_sync.now_ms()
                stmt7 = (
                    select(FakeBreakoutSignal.id)
                    .where(FakeBreakoutSignal.pattern_type == pattern_type)
                    .where(FakeBreakoutSignal.signal_time >= now_ms - 7 * 86_400_000)
                )
                latest.n_events_last_7d = len((await session.execute(stmt7)).all())
                # 回撤曲线快照：按周六归档 key 覆盖式更新（曲线截尾防 JSONB 膨胀）
                d = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
                week_key = (d + timedelta(days=(5 - d.weekday()) % 7)).strftime("%y%m%d")
                snaps = dict(latest.max_drawdown_curves or {})
                snaps[week_key] = {
                    "equity_curve": stats["equity_curve"][-200:],
                    "peak_equity": round(stats["peak_equity"], 6),
                    "dd": round(stats["max_drawdown"], 6),
                }
                latest.max_drawdown_curves = snaps
                await session.commit()
                logger.info(
                    "场景统计回填 [{}] | n={} 胜率 {} 累计EV/事件 {} 近7日 {} | 峰值 {} 回撤 {}",
                    pattern_type, stats["n"],
                    f"{stats['winrate']:.1%}" if stats["winrate"] is not None else "N/A",
                    f"{stats['cumulative_ev']:+.3f}" if stats["cumulative_ev"] is not None else "N/A",
                    latest.n_events_last_7d, stats["peak_equity"], stats["max_drawdown"],
                )
        except Exception as exc:
            logger.warning("场景统计回填失败 [{}] | {}", pattern_type, exc)

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
            "pending_count": self._pending_count,
            "confirm_miss_count": self._confirm_miss_count,
            "active_version": self._active_version,
            "shadow_versions": [s["version"] for s in self._shadow_versions],
            "daily_count": self._daily_count,
            "daily_date": self._daily_date,
            "eps": settings.fake_breakout_eps,
            "cooldown_seconds": settings.fake_breakout_cooldown_seconds,
            "max_daily_signals": settings.fake_breakout_max_daily_signals,
        }
