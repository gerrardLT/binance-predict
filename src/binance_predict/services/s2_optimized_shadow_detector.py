"""S2 空头耗尽优化版影子：复用父 S2，仅叠加量比上界与 14 日位置门。"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from inspect import isawaitable

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.shadow_version_gate import shadow_gate

S2_OPT_VERSION = "scene_bear_exhaust_opt_v1"
DISCOVERY_ID = "s2_opt14d"
BAR_MS_15M = 900_000
RANGE_BARS = 14 * 24 * 4
VOL_RATIO_MIN = 2.0
VOL_RATIO_MAX = 4.0
POS14D_MIN = 0.33
POLL_INTERVAL = 60.0
SETTLE_BARS = 40
PENDING_EXPIRE_MS = 4 * 3_600_000
RULE_TEXT = (
    "S2空头耗尽优化版：复用 bear_exhaust（跌破4h支撑+15m收阴+量比≥2）父信号，"
    "叠加 2≤vol_ratio<4 且 pos14d=(信号收盘-最近14日最低)/(最近14日最高-最低)≥0.33，"
    "最近14日仅使用截至信号15m收盘的完整K线；押下一15m周期 UP。"
)


def should_fire_s2_optimized(vol_ratio: float | None, pos14d: float | None) -> bool:
    return (
        vol_ratio is not None
        and pos14d is not None
        and VOL_RATIO_MIN <= vol_ratio < VOL_RATIO_MAX
        and pos14d >= POS14D_MIN
    )


def range_position(bars: list[dict], close: float) -> float | None:
    """完整 14 日 15m K 线区间位置；数据不足或平区间时保守拒绝。"""
    if len(bars) < RANGE_BARS:
        return None
    window = bars[-RANGE_BARS:]
    try:
        low = min(float(bar["low"]) for bar in window)
        high = max(float(bar["high"]) for bar in window)
    except (KeyError, TypeError, ValueError):
        return None
    return (close - low) / (high - low) if high > low else None


class S2OptimizedShadowDetector:
    """父 S2 命中时评估优化门；影子持续采集，实盘开火由独立通道开关控制。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        self._pm_15m_latest = pm_15m_latest
        self._on_live_fire = None
        self._trigger_count = 0
        self._settle_count = 0
        self._running = False
        self._task: asyncio.Task | None = None

    async def evaluate(self, sig: dict) -> None:
        try:
            if sig.get("pattern_type") != "bear_exhaust":
                return
            vol_ratio = self._number(sig.get("vol_ratio"))
            if vol_ratio is None or not VOL_RATIO_MIN <= vol_ratio < VOL_RATIO_MAX:
                return
            signal_start = int(sig["signal_bar_start"])
            target_start = int(sig["market_start_15m"])
            if target_start != signal_start + BAR_MS_15M:
                return
            bars = await self._collector.fetch_klines_ending_at(
                "15m", RANGE_BARS, target_start,
            )
            if (
                len(bars) != RANGE_BARS
                or int(bars[-1]["open_time"]) != signal_start
                or any(
                    int(bars[i]["open_time"]) - int(bars[i - 1]["open_time"]) != BAR_MS_15M
                    for i in range(1, len(bars))
                )
            ):
                return
            close = self._number(sig.get("signal_close"))
            if close is None:
                return
            pos14d = range_position(bars, close)
            if not should_fire_s2_optimized(vol_ratio, pos14d):
                return
            up, down, quote_ts = self._snapshot_quote(target_start)
            payload = {
                "id": sig.get("id"),
                "pattern_type": "bear_exhaust_opt_v1",
                "side": "low",
                "market_start_15m": target_start,
                "market_end_15m": int(sig["market_end_15m"]),
            }
            hook = self._on_live_fire
            if hook is not None:
                try:
                    result = hook(payload)
                    if isawaitable(result):
                        await result
                except Exception as exc:
                    logger.warning("S2优化版：实盘派发异常（不影响影子采集）| {}", exc)
            await self._record(
                sig, signal_start, target_start, close, vol_ratio, pos14d,
                up, down, quote_ts,
            )
        except Exception as exc:
            logger.warning("S2优化版：评估异常（不影响父S2）| {}", exc)

    async def _record(
        self,
        sig: dict,
        signal_start: int,
        target_start: int,
        close: float,
        vol_ratio: float,
        pos14d: float,
        up: float | None,
        down: float | None,
        quote_ts: int | None,
    ) -> None:
        if not shadow_gate.is_enabled(S2_OPT_VERSION):
            return
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == S2_OPT_VERSION,
                    KlineShadowSignal.signal_bar_start == signal_start,
                )
            )).scalar_one_or_none()
            if exists is not None:
                return
            session.add(KlineShadowSignal(
                version=S2_OPT_VERSION,
                discovery_id=DISCOVERY_ID,
                condition_text=RULE_TEXT,
                timeframe="15m",
                signal_bar_start=signal_start,
                signal_bar_end=target_start,
                direction="UP",
                target_bar_start=target_start,
                feature_snapshot={
                    "parent_id": sig.get("id"),
                    "signal_close": close,
                    "vol_ratio": vol_ratio,
                    "pos14d": round(pos14d, 6),
                },
                entry_up_price=up,
                entry_down_price=down,
                entry_quote_ts=quote_ts,
                status="PENDING",
            ))
            await session.commit()
        self._trigger_count += 1
        logger.info(
            "S2优化版影子触发 | 父信号 #{} | 量比 {:.3f} pos14d {:.3f} | UP报价 {}",
            sig.get("id"), vol_ratio, pos14d, up if up is not None else "N/A",
        )

    async def settle(self, closed_15m: list[dict]) -> None:
        """用完整 15m K 线结算本版本的 PENDING 行。"""
        by_start = {int(bar["open_time"]): bar for bar in closed_15m}
        if not by_start:
            return
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == S2_OPT_VERSION,
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(list(by_start)),
                )
            )).scalars().all()
            for row in rows:
                bar = by_start[int(row.target_bar_start)]
                open_price, close_price = float(bar["open"]), float(bar["close"])
                row.settle_open, row.settle_close = open_price, close_price
                row.settled_at = datetime.now(timezone.utc)
                if close_price == open_price:
                    row.settle_outcome, row.win, row.status = "NOISE", None, "EXPIRED"
                else:
                    row.settle_outcome = "UP" if close_price > open_price else "DOWN"
                    row.win = row.settle_outcome == row.direction
                    row.status = "SETTLED"
            if rows:
                self._settle_count += len(rows)
                await session.commit()

    async def _expire_stale(self) -> None:
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == S2_OPT_VERSION,
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            for row in rows:
                row.status = "EXPIRED"
            if rows:
                await session.commit()

    async def _poll_once(self) -> None:
        closed = await self._collector.fetch_recent_klines("15m", SETTLE_BARS)
        if closed:
            await self.settle(closed)
        await self._expire_stale()

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("S2优化版：结算轮询异常 | {}", exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._poll_once()
        except Exception as exc:
            logger.warning("S2优化版：冷启动结算失败（循环内重试）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="s2_optimized_shadow_detector")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def _snapshot_quote(self, target_start: int) -> tuple[float | None, float | None, int | None]:
        return snapshot_entry_quote(self._pm_15m_latest, target_start)

    @staticmethod
    def _number(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number

    def status(self) -> dict:
        return {
            "version": S2_OPT_VERSION,
            "running": self._running,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
        }
