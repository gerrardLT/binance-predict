"""报价 edge 影子检测器（quote_momentum / quote_contrarian 影子落表）。

信号定义（与 scripts/local_quote_bin_winrate.py + local_edge_cell_constraints.py
回测口径逐字段对齐，规则冻结勿动）：
    quote_momentum_v1（A 格顺势）：
        5m 窗内 t∈[90,210)s，DOWN token 报价首次进入 [0.69, 0.75) → 押 DOWN。
        回测：胜率 79.9%（Holdout 79%），EV +0.097，日频 13.5；
        约束来源：0.55-0.75 黄金区 + 唯一 OOS 存活约束 qdepth≥19（q≥0.69）。
    quote_contrarian_v1（B 格逆势）：
        5m 窗内 t∈[45,60)s，DOWN token 报价首次进入 [0.15, 0.25) → 押 DOWN。
        回测：胜率 24.0%，EV +0.155，日频 13.7；
        裸条件（两轮 30+ 假设 OOS 全灭，无可信约束）。

影子纪律（M4 同款）：只记录不下注、不发邮件、不占风控配额。
数据流（归档后处理，区别于 X4 的次窗结算）：
    1. 每 60s 轮询新归档 SentimentWindow；
    2. 扫 curve_down_price 曲线找规则区间内首个命中点（时点+报价）；
    3. 命中且窗口结算可判（actual_return 非 0 非 NULL）→ 直接落 SETTLED 记录
       （entry = 首个命中点真实报价，win = 本窗 outcome == DOWN，
       EV = 0.98/q−1 / −1，费 2% 无溢价，与回测完全一致）。
    4. 幂等：(version, window_start) 唯一约束防重。
冷启动回补：启动时重扫最近 12 个已归档窗口。

字段语义映射（复用 misalignment_signals 表）：
    end_pct → 触发时刻 DOWN 报价（X4 语义"触发窗末 UP%"的扩展，注释在案）；
    target_window_start → window_start（本窗即目标窗，无次窗）；
    outcome_base → 触发窗结算方向（审计冗余）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import MisalignmentSignal, SentimentWindow

logger = logging.getLogger(__name__)

# ---- 冻结口径（回测同源，勿动）----
# rule -> (t_lo_s, t_hi_s, q_lo, q_hi)
QUOTE_EDGE_RULES: dict[str, tuple[float, float, float, float]] = {
    "quote_momentum_v1": (90.0, 210.0, 0.69, 0.75),
    "quote_contrarian_v1": (45.0, 60.0, 0.15, 0.25),
}
FEE_RET = 0.98            # EV = 赢 0.98/q−1 / 输 −1（费 2%，无溢价，回测口径）
POLL_INTERVAL = 60.0      # 轮询间隔（秒）
BACKSCAN_WINDOWS = 12     # 冷启动回补窗口数（1 小时）


def _outcome_of(w: SentimentWindow) -> str | None:
    """可判定的结算方向：actual_return None/0 → None（与 X4 结算同口径）。"""
    ret = w.actual_return
    if ret is None or float(ret) == 0.0:
        return None
    return w.outcome if (w.outcome or "") in ("UP", "DOWN") else None


def _find_first_hit(
    curve: list | None, start_ms: int,
    t_lo: float, t_hi: float, q_lo: float, q_hi: float,
) -> tuple[float, int] | None:
    """曲线内首个命中点：(报价, 采样时刻 ms)。

    按时刻升序扫描，返回第一个满足 t_rel∈[t_lo,t_hi)s 且 v∈[q_lo,q_hi) 的点，
    与回测"每周期首个命中报价"同口径。曲线乱序时按 t 排序。
    """
    if not curve:
        return None
    for p in sorted(curve, key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        t_rel = (int(t) - start_ms) / 1000.0
        if t_rel < 0:
            continue
        if t_rel >= t_hi:  # 已过规则窗口，后续更晚
            break
        if t_rel < t_lo:
            continue
        v = float(v)
        if q_lo <= v < q_hi:
            return v, int(t)
    return None


def _up_price_at_or_before(curve: list | None, ts_ms: int) -> float | None:
    """UP 曲线中 ≤ ts_ms 的最晚采样价（决策时刻已知信息，不含未来）。"""
    best = None
    for p in sorted((curve or []), key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts_ms:
            best = float(v)
    return best


def _ev_at_entry(win: bool, price: float) -> float:
    """单注 EV（回测口径，无溢价）：赢 0.98/q−1 / 输 −1。"""
    return (FEE_RET / price - 1.0) if win else -1.0


class QuoteEdgeDetector:
    """报价 edge 影子检测器：轮询窗口归档 → 首个命中报价 → 直接落 SETTLED。

    归档后处理模式：落表时窗口已结算，无 PENDING 阶段（区别于 X4 的次窗结算）。
    """

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_window_end: int | None = None  # 已处理过的最大窗口 end_time
        self._trigger_count = 0

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
            logger.warning("报价 edge 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="quote_edge_detector")
        logger.info(
            "报价 edge 影子检测器启动 | 规则 {} | EV=0.98/q−1（费 2%，影子模式只记录不下注）",
            {k: f"t∈[{v[0]:.0f},{v[1]:.0f})s q∈[{v[2]:.2f},{v[3]:.2f})"
             for k, v in QUOTE_EDGE_RULES.items()},
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
        logger.info("报价 edge 影子检测器已停止 | 触发（含回补）{}", self._trigger_count)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("报价 edge 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
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
                logger.warning("报价 edge 影子：窗口处理失败 | window {} | {}", w.start_time, exc)
            self._last_window_end = max(self._last_window_end or 0, int(w.end_time))

    # ------------------------------------------------------------------
    # 核心：单窗口处理（扫曲线 → 首个命中 → 落 SETTLED）
    # ------------------------------------------------------------------

    async def _process_window(self, w: SentimentWindow) -> None:
        start_ms, end_ms = int(w.start_time), int(w.end_time)
        outcome = _outcome_of(w)
        if outcome is None:
            return  # NOISE/缺结算：胜负不可判，不产生信号

        async with async_session_factory() as session:
            for version, (t_lo, t_hi, q_lo, q_hi) in QUOTE_EDGE_RULES.items():
                hit = _find_first_hit(w.curve_down_price, start_ms, t_lo, t_hi, q_lo, q_hi)
                if hit is None:
                    continue
                price, quote_ts = hit
                dup = await session.execute(
                    sa_select(MisalignmentSignal.id).where(
                        MisalignmentSignal.version == version,
                        MisalignmentSignal.window_start == start_ms,
                    )
                )
                if dup.first() is not None:
                    continue
                win = outcome == "DOWN"
                session.add(MisalignmentSignal(
                    version=version,
                    window_start=start_ms,
                    window_end=end_ms,
                    end_pct=price,               # 语义扩展：触发时刻 DOWN 报价
                    outcome_base=outcome,          # 触发窗结算方向（审计冗余）
                    direction="DOWN",
                    target_window_start=start_ms,  # 本窗即目标窗
                    entry_down_price=price,
                    entry_up_price=_up_price_at_or_before(w.curve_up_price, quote_ts),
                    entry_quote_ts=quote_ts,
                    entry_quote_kind="real",
                    settle_outcome=outcome,
                    win=win,
                    ev_at_entry=_ev_at_entry(win, price),
                    status="SETTLED",
                ))
                self._trigger_count += 1
                logger.info(
                    "报价 edge 影子触发+结算 | {} | 窗口 {} | t=+{:.0f}s q={:.3f}"
                    " → {} | win={} ev={:+.3f}",
                    version,
                    datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                    .strftime("%m-%d %H:%M"),
                    (quote_ts - start_ms) / 1000.0, price, outcome, win,
                    _ev_at_entry(win, price),
                )
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        stmt = (
            sa_select(SentimentWindow)
            .order_by(SentimentWindow.end_time.desc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = list(reversed((await session.execute(stmt)).scalars().all()))
        for w in wins:
            await self._process_window(w)
        self._last_window_end = max((int(w.end_time) for w in wins), default=None)
        if wins:
            logger.info(
                "报价 edge 影子：冷启动回补 {} 窗（{}~{}）完成",
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
        }
