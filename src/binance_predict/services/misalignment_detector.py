"""情绪错位影子检测器（X4 / M4 影子并行）。

信号定义（与 scripts/local_misalignment_scan.py 回测口径逐字段对齐）：
    本窗（5m 情绪窗）结算 UP（收阳）且窗口末 UP% ≤ 40 → 押次窗 DOWN。
    回测成绩：IS 65.6%(n=122) / OOS 57.8%(n=45) / 合并 63.5%，
    EV +0.254 CI(0.038, 0.493)（费 2%+溢 0.01；瑕疵：实价覆盖 19%）。

影子纪律（M4）：只记录不下注、不占风控配额。邮件推送（2026-08-28 改为
结算后复盘口径）：触发落表不再发预告邮件；结算置 SETTLED 落库后按
新鲜度闸(sig.window_end) → 通道开关(is_live_enabled) → 实盘成交闸
(has_live_filled_order，x4 订单 window_start=次窗起点即 target_window_start，
决策点 +150s 已下单、此时早已终态，查询时序成立) 推送复盘邮件（含
win/entry/EV），被实盘门禁拦下的信号不再推。新鲜度闸自动静默冷启动
回补/污染重扫；总开关 signal_push_email_enabled，全局日限防轰炸。
数据流：完全复用 sentiment_windows 归档——
    1. 每 60s 轮询新归档窗口 W_n；
    2. W_n 收阳 & end_pct≤40 → 落 PENDING 信号（target = W_n.end_time 起的次窗）；
    3. 已有 PENDING 且 target_window_start == W_n.start_time → 回读 W_n 归档曲线
       结算：150s 决策点真实 DOWN/UP token 价 + 次窗 outcome → SETTLED/EXPIRED。
冷启动回补：启动时重扫最近 12 个已归档窗口（幂等，唯一约束防重）。

x4_v2 平静市门禁版（2026-08-22 5m 粒度归因落地，只加不改；v1 冻结口径原样）：
    触发条件 = v1 全部条件 + |触发前 1h BTC 累计涨跌幅| < 0.5%（±10min 容差查
    entry_price，严格 ex-ante）。归因依据（51 笔已结算 x4_v1）：平静市段
    wr 57.6%/EV+9.70 vs 单边段 wr 23%/EV−13.8；门禁数据缺失 → v2 不触发。
    结算通道与 v1 完全共用（PENDING → 次窗归档结算，版本无关）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import MisalignmentSignal, SentimentWindow
from .signal_notify import (
    fire_signal_email, fmt_bjt, has_live_filled_order, is_fresh_signal, is_live_enabled,
)

logger = logging.getLogger(__name__)

# ---- X4 冻结口径（回测 local_misalignment_scan.py 同源，勿动）----
X4_END_MAX = 40.0        # 触发窗末 UP% 上限（端点口径，非 30/50 档）
DECISION_T_SEC = 150.0   # 决策点：次窗开窗后 150s（宪法 V1.1）
FEE, PREM = 0.02, 0.01   # 单注成本口径（费 2% + 溢价 0.01）
POLL_INTERVAL = 60.0     # 轮询间隔（秒）
BACKSCAN_WINDOWS = 12    # 冷启动回补窗口数（1 小时）
PENDING_TIMEOUT_MS = 20 * 60 * 1000  # PENDING 超时（次窗归档最迟 ~6min）

# ---- x4_v2 平静市门禁（2026-08-22 5m 归因落地，只加不改；v1 冻结口径原样）----
# 触发前 1h（±10min 容差）BTC 累计涨跌幅 |chg|<0.5% 才触发：
# 51 笔已结算 x4_v1 归因——平静市段 wr 57.6%/EV+9.70 vs 单边段 wr 23%/EV−13.8。
X4_V2_VERSION = "x4_v2"
X4_V2_PAST1H_MAX_ABS_PCT = 0.5   # |过去 1h 涨幅| 上限（%）
X4_V2_LOOKBACK_MS = 3_600_000    # 回看 1h
X4_V2_TOL_MS = 600_000           # 历史窗 start_time 容差 ±10min（兜数据缺口）


def _window_open_price(w: SentimentWindow) -> float | None:
    """窗口开盘 BTC 价：entry_price 优先，回退 curve_btc_price 首个有效采样点。"""
    p = getattr(w, "entry_price", None)
    if p is not None and float(p) > 0:
        return float(p)
    for pt in sorted((getattr(w, "curve_btc_price", None) or []), key=lambda x: x.get("t") or 0):
        v = pt.get("v")
        if v is not None and float(v) > 0:
            return float(v)
    return None


async def _past_1h_chg_pct(session, w: SentimentWindow) -> float | None:
    """触发前 1h 的 BTC 累计涨跌幅（%）。

    基准 = start−1h（±10min 容差）内最晚已归档窗口的 entry_price；
    当前 = 本窗开盘价（entry_price 优先回退 curve 首点）。
    任一缺失 → None（门禁数据不足，v2 不触发）。
    """
    target = int(w.start_time) - X4_V2_LOOKBACK_MS
    base = (await session.execute(
        sa_select(SentimentWindow.entry_price)
        .where(
            SentimentWindow.start_time >= target - X4_V2_TOL_MS,
            SentimentWindow.start_time <= target + X4_V2_TOL_MS,
            SentimentWindow.entry_price.isnot(None),
        )
        .order_by(SentimentWindow.start_time.desc())
        .limit(1)
    )).scalar_one_or_none()
    if base is None:
        return None
    cur = _window_open_price(w)
    if cur is None:
        return None
    return (cur - float(base)) / float(base) * 100.0


def _curve_end_pct(curve: list | None) -> float | None:
    """窗口末 UP%：curve_up_pct 按 t 排序后的末点（回测 _line_feats['end'] 同口径）。"""
    pts = sorted((curve or []), key=lambda p: p.get("t", 0))
    for p in reversed(pts):
        v = p.get("v")
        if v is not None:
            return float(v)
    return None


def _price_at(curve: list | None, start_ms: int) -> tuple[float | None, int | None]:
    """≤150s 内最晚采样点（回测 price_at 同口径）：(价格, 采样点时刻 ms)。"""
    best_v, best_t = None, None
    for p in sorted((curve or []), key=lambda x: x.get("t", 0)):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if (t - start_ms) / 1000.0 <= DECISION_T_SEC:
            best_v, best_t = float(v), int(t)
    return best_v, best_t


def _entry_quote(
    down_curve: list | None, up_curve: list | None, pct_curve: list | None, start_ms: int,
) -> tuple[float | None, float | None, int | None, str | None]:
    """决策点入场报价：真实 DOWN/UP token 价优先，DOWN 缺失回退 chance/100（proxy）。"""
    d, _ = _price_at(down_curve, start_ms)
    u, _ = _price_at(up_curve, start_ms)
    if d is not None and d > 0:
        return d, u, _price_at(down_curve, start_ms)[1], "real"
    c, ct = _price_at(pct_curve, start_ms)
    if c is not None and c > 0:
        return c / 100.0, u, ct, "proxy"
    return None, u, None, None


def _ev_at_entry(win: bool | None, price: float | None) -> float | None:
    """单注 EV：赢 0.98/(entry)−1（entry=min(max(p+0.01,0.01),0.99)）/ 输 −1。"""
    if win is None or price is None:
        return None
    if win:
        return 0.98 / min(max(price + PREM, 0.01), 0.99) - 1.0
    return -1.0


class MisalignmentDetector:
    """X4 影子信号检测器：轮询情绪窗归档 → 触发/结算，全程只落表不下注。"""

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_window_end: int | None = None  # 已处理过的最大窗口 end_time
        self._trigger_count = 0
        self._settle_count = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._backscan()
        except Exception as exc:
            logger.warning("X4 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="misalignment_detector")
        logger.info(
            "X4 影子检测器启动 | 口径 end≤{} 收阳→次窗DOWN | 决策点 +{}s | 费{}+溢{}"
            " | x4_v2 并行：|past1h|<{}%（平静市门禁，缺数据不触发）"
            "（影子模式：只记录不下注）",
            X4_END_MAX, DECISION_T_SEC, FEE, PREM, X4_V2_PAST1H_MAX_ABS_PCT,
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
        logger.info("X4 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
                await self._expire_stale_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("X4 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        """取归档窗口（比 _last_window_end 新的，升序），逐窗处理触发与结算。"""
        stmt = (
            sa_select(SentimentWindow)
            .where(SentimentWindow.end_time > (self._last_window_end or 0))
            .order_by(SentimentWindow.end_time.asc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = (await session.execute(stmt)).scalars().all()
        for w in wins:
            try:
                await self._process_window(w)
            except Exception as exc:
                logger.warning(
                    "X4 影子：窗口处理失败 | window {} | {}", w.start_time, exc,
                )
            self._last_window_end = max(self._last_window_end or 0, int(w.end_time))

    # ------------------------------------------------------------------
    # 核心：单窗口处理（触发判定 + 目标结算）
    # ------------------------------------------------------------------

    async def _process_window(self, w: SentimentWindow) -> None:
        start_ms, end_ms = int(w.start_time), int(w.end_time)

        # --- 1) 结算：本窗是某 PENDING 信号的目标窗 ---
        await self._settle_pending_for(w)

        # --- 2) 触发：本窗收阳 & end_pct≤40 → 新 PENDING 信号 ---
        if (w.outcome or "") != "UP":
            return
        end_pct = _curve_end_pct(w.curve_up_pct)
        if end_pct is None or end_pct > X4_END_MAX:
            return
        async with async_session_factory() as session:
            dup = await session.execute(
                sa_select(MisalignmentSignal.id).where(
                    MisalignmentSignal.version == "x4_v1",
                    MisalignmentSignal.window_start == start_ms,
                )
            )
            if dup.first() is None:
                session.add(MisalignmentSignal(
                    version="x4_v1",
                    window_start=start_ms,
                    window_end=end_ms,
                    end_pct=end_pct,
                    outcome_base="UP",
                    direction="DOWN",
                    target_window_start=end_ms,
                    status="PENDING",
                ))
                await session.commit()
                self._trigger_count += 1
                logger.info(
                    "X4 影子触发 | 窗口 {}~{} | end_pct={:.1f} → 押次窗 DOWN（目标 {}）",
                    datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M"),
                    datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%H:%M"),
                    end_pct, end_ms,
                )
                # 邮件推送已挪到结算回读后（_settle_pending_for，2026-08-28）：
                # 落表预告改为结算复盘，且只推实盘已成交的信号（FILLED 闸）。

            # --- x4_v2：v1 条件全同 + 平静市门禁（独立幂等/独立 commit，只加不改）---
            dup2 = await session.execute(
                sa_select(MisalignmentSignal.id).where(
                    MisalignmentSignal.version == X4_V2_VERSION,
                    MisalignmentSignal.window_start == start_ms,
                )
            )
            if dup2.first() is not None:
                return
            past1h = await _past_1h_chg_pct(session, w)
            if past1h is None or abs(past1h) >= X4_V2_PAST1H_MAX_ABS_PCT:
                return  # 单边市/门禁数据缺失 → v2 不触发（v1 不受影响）
            session.add(MisalignmentSignal(
                version=X4_V2_VERSION,
                window_start=start_ms,
                window_end=end_ms,
                end_pct=end_pct,
                outcome_base="UP",
                direction="DOWN",
                target_window_start=end_ms,
                status="PENDING",
            ))
            await session.commit()
        self._trigger_count += 1
        logger.info(
            "X4v2 影子触发 | 窗口 {} | end_pct={:.1f} past1h={:+.2f}%（平静市）→ 押次窗 DOWN（目标 {}）",
            datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M"),
            end_pct, past1h, end_ms,
        )
    async def _settle_pending_for(self, w: SentimentWindow) -> None:
        """本窗 start == 信号 target_window_start → 回读曲线结算。"""
        start_ms = int(w.start_time)
        async with async_session_factory() as session:
            stmt = sa_select(MisalignmentSignal).where(
                MisalignmentSignal.status == "PENDING",
                MisalignmentSignal.target_window_start == start_ms,
            )
            sigs = (await session.execute(stmt)).scalars().all()
            if not sigs:
                return
            # 回测 ev_eval 同口径：actual_return 为 None/0 → 无法判向（EXPIRED）
            ret = w.actual_return
            outcome = w.outcome if (ret is not None and float(ret) != 0.0) else None

            for sig in sigs:
                price_d, price_u, quote_ts, kind = _entry_quote(
                    w.curve_down_price, w.curve_up_price, w.curve_down_pct, start_ms,
                )
                if outcome is None:
                    # NOISE / 缺结算价：胜负无法判定 → EXPIRED（不进统计）
                    sig.status = "EXPIRED"
                    sig.settle_outcome = w.outcome  # 保留原始标注（多为 NOISE）供审计
                    sig.entry_down_price, sig.entry_up_price = price_d, price_u
                    sig.entry_quote_ts, sig.entry_quote_kind = quote_ts, kind
                else:
                    win = outcome == "DOWN"
                    sig.status = "SETTLED"
                    sig.settle_outcome = outcome
                    sig.win = win
                    sig.entry_down_price, sig.entry_up_price = price_d, price_u
                    sig.entry_quote_ts, sig.entry_quote_kind = quote_ts, kind
                    sig.ev_at_entry = _ev_at_entry(win, price_d)
                self._settle_count += 1
                logger.info(
                    "X4 影子结算 | 信号窗 {} | 次窗 {} → {} | win={} | entry={}({}) ev={}",
                    sig.window_start, start_ms, sig.status,
                    sig.win if sig.status == "SETTLED" else "N/A",
                    price_d if price_d is not None else "N/A", kind or "missing",
                    f"{sig.ev_at_entry:+.3f}" if sig.ev_at_entry is not None else "N/A",
                )
            await session.commit()

            # 邮件推送（2026-08-28 挪到结算回读后，fire-and-forget）：结算置
            # SETTLED 落库后，对每条信号按闸链推复盘邮件——新鲜度闸
            # (sig.window_end，回补/重扫静默) → 通道开关(is_live_enabled) →
            # 实盘成交闸(has_live_filled_order：x4 订单 window_start=次窗起点
            # 即 target_window_start，决策点 +150s 已下单，此时早已终态；被
            # 实盘门禁拦下的信号无 FILLED 行 → 不推)。expire_on_commit=False，
            # commit 后 sig 属性仍可安全读取。EXPIRED 行不推。边界：下单确认
            # 暂落 PENDING、事后由对账翻成 FILLED 的信号会漏推（闸只在结算
            # 时刻评估一次，宁少勿多）。
            for sig in sigs:
                if sig.status != "SETTLED":
                    continue
                if not (is_fresh_signal(int(sig.window_end))
                        and is_live_enabled(sig.version)
                        and await has_live_filled_order(
                            sig.version, int(sig.target_window_start))):
                    continue
                win_str = "赢" if sig.win else "输"
                entry = sig.entry_down_price
                entry_str = f"{float(entry):.3f}" if entry is not None else "N/A"
                ev = sig.ev_at_entry
                ev_str = f"{float(ev):+.3f}" if ev is not None else "N/A"
                fire_signal_email(
                    "x4",
                    f"[信号·实盘] {sig.version} | 押次窗DOWN {win_str} | 触发窗 "
                    f"{fmt_bjt(int(sig.window_start))} 北京时间",
                    f"版本: {sig.version}（实盘已成交，本邮件为结算复盘）\n"
                    f"触发窗: {fmt_bjt(int(sig.window_start))}"
                    f"~{fmt_bjt(int(sig.window_end), with_date=False)} 北京时间\n"
                    f"条件: 本窗收阳 & 窗末 UP%={float(sig.end_pct):.1f} ≤ 40\n"
                    f"押注: 次窗 DOWN（次窗 {fmt_bjt(int(sig.target_window_start))} 北京时间）\n"
                    f"入场: 决策点 +{DECISION_T_SEC:.0f}s DOWN报价 {entry_str}\n"
                    f"结算: {sig.settle_outcome} → {win_str}\n"
                    f"EV: {ev_str}（0.98/(p+0.01)−1 / −1，费 2%+溢 0.01）",
                )

    async def _expire_stale_pending(self) -> None:
        """超时未结算的 PENDING（次窗归档缺失/停机跨窗）→ EXPIRED，防永久挂起。"""
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            stmt = sa_select(MisalignmentSignal).where(
                MisalignmentSignal.status == "PENDING",
                MisalignmentSignal.target_window_start < now_ms - PENDING_TIMEOUT_MS,
            )
            sigs = (await session.execute(stmt)).scalars().all()
            for sig in sigs:
                sig.status = "EXPIRED"
                logger.warning(
                    "X4 影子：PENDING 超时 EXPIRED | 窗口 {}（次窗归档缺失）", sig.window_start,
                )
            if sigs:
                await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        """重扫最近 N 个已归档窗口：补触发 + 补结算（停机期间漏掉的）。"""
        stmt = (
            sa_select(SentimentWindow)
            .order_by(SentimentWindow.end_time.desc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = list(reversed((await session.execute(stmt)).scalars().all()))
        for w in wins:
            await self._process_window(w)
        self._last_window_end = max(
            (int(w.end_time) for w in wins), default=None,
        )
        if wins:
            logger.info(
                "X4 影子：冷启动回补 {} 窗（{}~{}）完成",
                len(wins), wins[0].start_time, wins[-1].end_time,
            )

    # ------------------------------------------------------------------
    # 状态（status API 用）
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_window_end": self._last_window_end,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
        }
