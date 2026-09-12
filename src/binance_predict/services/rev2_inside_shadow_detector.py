"""经典孕线上吊线/倒垂线反转影子检测器与实盘驱动（rev2 族）。

15m 版本保持原有孕线几何口径，5m 新增精选 HM_DOWN 版本。每个实例只处理
一个周期，确保 K 线、报价缓存、PENDING 结算和实盘开火窗口不会跨周期混用。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import numpy as np
from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from binance_predict.discovery.data import Klines
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.shadow_version_gate import shadow_gate

from binance_predict.services.wechat_notifier import wechat_notifier
from binance_predict.config.settings import settings

REV2_15M_SPECS: list[dict] = [
    {
        "version": "hm_inside_15m_v2",
        "discovery_id": "rev2_hm_15m",
        "timeframe": "15m",
        "direction": "DOWN",
        "short_wick_max": 0.10,
        "main_wick_min": 0.45,
        "main_wick_max": None,
        "condition_text": (
            "prev_is_green == True AND prev_body_r >= 0.40 AND is_prominent_high "
            "AND is_longest_body AND is_full_inside AND lower_r >= 0.45 AND body_r <= 0.45 AND upper_r <= 0.10"
        ),
    },
    {
        "version": "ih_inside_15m_v2",
        "discovery_id": "rev2_ih_15m",
        "timeframe": "15m",
        "direction": "UP",
        "short_wick_max": 0.10,
        "main_wick_min": 0.45,
        "main_wick_max": None,
        "condition_text": (
            "prev_is_green == False AND prev_body_r >= 0.40 AND is_prominent_low "
            "AND is_longest_body AND is_full_inside AND upper_r >= 0.45 AND body_r <= 0.45 AND lower_r <= 0.10"
        ),
    },
]

REV2_5M_SPECS: list[dict] = [
    {
        "version": "hm_inside_5m_v2",
        "discovery_id": "rev2_hm_5m",
        "timeframe": "5m",
        "direction": "DOWN",
        "short_wick_max": 0.10,
        "main_wick_min": 0.75,
        "main_wick_max": 0.90,
        "condition_text": (
            "prev_is_green == True AND prev_body_r >= 0.40 AND is_prominent_high "
            "AND is_longest_body AND is_full_inside AND lower_r >= 0.75 AND lower_r < 0.90 "
            "AND body_r <= 0.45 AND upper_r <= 0.10"
        ),
    },
]

REV2_SHADOW_SPECS: list[dict] = REV2_15M_SPECS + REV2_5M_SPECS
REV2_VERSIONS = [s["version"] for s in REV2_SHADOW_SPECS]
VERSIONS_BY_TF = {
    "5m": [s["version"] for s in REV2_5M_SPECS],
    "15m": [s["version"] for s in REV2_15M_SPECS],
}
BAR_MS = {"5m": 300_000, "15m": 900_000}
BAR_MS_15M = BAR_MS["15m"]
POLL_INTERVAL = 60.0
WARMUP_BARS = 40
BACKSCAN_BARS = 12
PENDING_EXPIRE_MS = {"5m": 3_600_000, "15m": 4 * 3_600_000}
REV2_LIVE_MAX_LAG_MS = 90_000  # 目标根开盘 <=90s 的新鲜信号才驱动实盘


def _to_klines(rows: list[dict], bar_ms: int) -> Klines:
    t = np.asarray([r["open_time"] for r in rows], dtype=np.int64)
    kl = Klines(
        t=t,
        o=np.asarray([r["open"] for r in rows], dtype=np.float64),
        h=np.asarray([r["high"] for r in rows], dtype=np.float64),
        l=np.asarray([r["low"] for r in rows], dtype=np.float64),
        c=np.asarray([r["close"] for r in rows], dtype=np.float64),
        v=np.asarray([r["volume"] for r in rows], dtype=np.float64),
        cont=np.ones(len(rows), dtype=bool),
    )
    if len(t) > 1:
        kl.cont[1:] = (t[1:] - t[:-1]) == bar_ms
        kl.cont[0] = False
    return kl


def evaluate_rev2_patterns(
    kl: Klines, n_tail: int, specs: list[dict] | None = None
) -> list[dict]:
    """按给定周期规格求值孕线反转形态，返回末 n_tail 根内的触发点。"""
    specs = REV2_15M_SPECS if specs is None else specs
    specs_by_version = {spec["version"]: spec for spec in specs}
    hm_spec = next((spec for version, spec in specs_by_version.items() if version.startswith("hm_inside_")), None)
    ih_spec = next((spec for version, spec in specs_by_version.items() if version.startswith("ih_inside_")), None)
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    n = len(c)
    if n < 5:
        return []

    rng = h - l
    rng_safe = np.where(rng > 0, rng, 1e-8)
    body = np.abs(c - o)
    body_r = body / rng_safe
    upper_w = h - np.maximum(o, c)
    lower_w = np.minimum(o, c) - l
    upper_r = upper_w / rng_safe
    lower_r = lower_w / rng_safe
    is_green = c >= o

    hits = []
    start_eval = max(4, n - n_tail)

    for i in range(start_eval, n):
        prev_is_green = is_green[i - 1]
        prev_body = body[i - 1]
        prev_body_r = body_r[i - 1]
        prev_h = h[i - 1]
        prev_l = l[i - 1]

        is_prominent_high = prev_h >= np.max(h[max(0, i - 4):i])
        is_prominent_low = prev_l <= np.min(l[max(0, i - 4):i])
        is_longest_body = prev_body >= np.max(body[max(0, i - 4):i - 1])
        is_full_inside = (h[i] <= prev_h * 1.0002) and (l[i] >= prev_l * 0.9998)

        if not (is_longest_body and is_full_inside):
            continue

        if (
            hm_spec is not None
            and prev_is_green
            and prev_body_r >= 0.40
            and is_prominent_high
            and lower_r[i] >= hm_spec["main_wick_min"]
            and (hm_spec["main_wick_max"] is None or lower_r[i] < hm_spec["main_wick_max"])
            and body_r[i] <= 0.45
            and upper_r[i] <= hm_spec["short_wick_max"]
        ):
            hits.append({
                "spec": hm_spec,
                "idx": i,
                "snapshot": {
                    "prev_body_r": round(float(prev_body_r), 4),
                    "body_r": round(float(body_r[i]), 4),
                    "lower_r": round(float(lower_r[i]), 4),
                    "upper_r": round(float(upper_r[i]), 4),
                }
            })

        if (
            ih_spec is not None
            and (not prev_is_green)
            and prev_body_r >= 0.40
            and is_prominent_low
            and upper_r[i] >= ih_spec["main_wick_min"]
            and (ih_spec["main_wick_max"] is None or upper_r[i] < ih_spec["main_wick_max"])
            and body_r[i] <= 0.45
            and lower_r[i] <= ih_spec["short_wick_max"]
        ):
            hits.append({
                "spec": ih_spec,
                "idx": i,
                "snapshot": {
                    "prev_body_r": round(float(prev_body_r), 4),
                    "body_r": round(float(body_r[i]), 4),
                    "upper_r": round(float(upper_r[i]), 4),
                    "lower_r": round(float(lower_r[i]), 4),
                }
            })

    return hits


class Rev2InsideShadowDetector:
    """单周期 Rev2 孕线反转检测器：轮询收盘、落影子信号并驱动实盘。"""

    def __init__(
        self,
        collector,
        pm_latest: dict | None = None,
        timeframe: str = "15m",
        pm_15m_latest: dict | None = None,
    ) -> None:
        if timeframe not in BAR_MS:
            raise ValueError(f"不支持的 Rev2 周期: {timeframe}")
        self._collector = collector
        self._pm_latest = pm_latest if pm_latest is not None else (pm_15m_latest or {})
        self._timeframe = timeframe
        self._bar_ms = BAR_MS[timeframe]
        self._specs = REV2_5M_SPECS if timeframe == "5m" else REV2_15M_SPECS
        self._versions = VERSIONS_BY_TF[timeframe]
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: int | None = None
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._backscan()
        except Exception as exc:
            logger.warning("Rev2Inside 影子：{} 冷启动回补失败（忽略，循环内自愈）| {}", self._timeframe, exc)
        self._task = asyncio.create_task(
            self._loop(), name=f"rev2_inside_shadow_detector_{self._timeframe}"
        )
        logger.info(
            "Rev2Inside {} 孕线反转检测器启动 | {} | 影子落表+实盘支持就绪",
            self._timeframe, "/".join(self._versions),
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
        logger.info(
            "Rev2Inside {} 影子检测器已停止 | 触发 {} 结算 {}",
            self._timeframe, self._trigger_count, self._settle_count,
        )

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Rev2Inside 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        closed = await self._collector.fetch_recent_klines(self._timeframe, WARMUP_BARS)
        if len(closed) < WARMUP_BARS:
            return
        last_start = int(closed[-1]["open_time"])
        if self._last_evaluated_bar is None or last_start > self._last_evaluated_bar:
            await self._evaluate_new_bars(closed)
            self._last_evaluated_bar = last_start
        await self._settle_pending(closed)
        await self._expire_stale_pending()
        if self._timeframe == "15m":
            await self._check_radar()

    async def _check_radar(self) -> None:
        """提前预警雷达：探测当前正在走的 15m K 线是否初具孕线反转雏形。"""
        if self._timeframe != "15m" or not settings.wechat_radar_enabled or not settings.wechat_work_enabled:
            return
        try:
            raw_15m = await self._collector.fetch_klines_raw("15m", WARMUP_BARS)
            if len(raw_15m) < 5:
                return
            current_bar = raw_15m[-1]
            now_ms = int(time.time() * 1000)
            bar_start = int(current_bar["open_time"])
            bar_end = bar_start + self._bar_ms
            rem_sec = int((bar_end - now_ms) / 1000)

            if not (45 <= rem_sec <= 150):
                return

            kl15 = _to_klines(raw_15m, self._bar_ms)
            hits = evaluate_rev2_patterns(kl15, 1, self._specs)
            for hit in hits:
                spec = hit["spec"]
                channel = spec["version"]
                direction = spec["direction"]
                snap = hit["snapshot"]
                features = (
                    f"前根实体占比 {snap['prev_body_r']*100:.1f}%，"
                    f"当前孕线包裹，下影 {snap.get('lower_r', 0)*100:.1f}% / 上影 {snap.get('upper_r', 0)*100:.1f}%"
                )
                wechat_notifier.notify_pre_trade_radar(
                    radar_type="15m 经典孕线反转",
                    channel=channel,
                    direction=direction,
                    target_time_ms=bar_end,
                    lead_seconds=rem_sec,
                    max_exec_price=0.30,
                    features_desc=features,
                    dedup_key=f"radar_rev2_{channel}_{bar_start}",
                )
        except Exception as exc:
            logger.debug("Rev2Inside 雷达探测异常: {}", exc)

    async def _evaluate_new_bars(self, closed: list[dict]) -> None:
        kl = _to_klines(closed, self._bar_ms)
        if self._last_evaluated_bar is None:
            n_tail = BACKSCAN_BARS
        else:
            starts = [int(r["open_time"]) for r in closed]
            try:
                first_new = next(i for i, s in enumerate(starts) if s > self._last_evaluated_bar)
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)

        hits = evaluate_rev2_patterns(kl, n_tail, self._specs)
        if not hits:
            return

        live_payloads: list[dict] = []
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed[hit["idx"]]
                last_bar = bar
                if await self._record_signal(session, hit["spec"], bar, hit["snapshot"], live_payloads):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info(
                    "Rev2Inside 影子触发 +{} | {} | 信号根 {}",
                    added, self._timeframe, int(last_bar["open_time"]),
                )

        self._dispatch_live(live_payloads)

    def _dispatch_live(self, payloads: list[dict]) -> None:
        hook = self._on_live_fire
        if hook is None or not payloads:
            return
        for p in payloads:
            try:
                hook(p)
            except Exception as exc:
                logger.warning("Rev2Inside：实盘开火分派异常（不影响影子采集）| {}", exc)

    async def _record_signal(
        self, session, spec: dict, bar: dict, snapshot: dict, live_payloads: list[dict] | None = None
    ) -> bool:
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False

        target_bar_start = start_ms + self._bar_ms
        if live_payloads is not None and (
            0 <= int(time.time() * 1000) - target_bar_start <= REV2_LIVE_MAX_LAG_MS
        ):
            live_payloads.append({
                "version": spec["version"],
                "market_start": target_bar_start,
                "market_end": target_bar_start + self._bar_ms,
                "direction": spec["direction"],
                "signal_bar_start": start_ms,
            })

        if not shadow_gate.is_enabled(spec["version"]):
            return False

        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_latest, target_bar_start)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe=self._timeframe,
            signal_bar_start=start_ms,
            signal_bar_end=target_bar_start,
            direction=spec["direction"],
            target_bar_start=target_bar_start,
            feature_snapshot=snapshot,
            entry_up_price=up_q,
            entry_down_price=down_q,
            entry_quote_ts=q_ts,
            status="PENDING",
        ))
        return True

    async def _settle_pending(self, closed: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(self._versions),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(starts),
                )
            )).scalars().all()
            if not pendings:
                return
            for sig in pendings:
                bar = by_start[int(sig.target_bar_start)]
                o, c = float(bar["open"]), float(bar["close"])
                if c == o:
                    sig.settle_outcome, sig.win, sig.status = "NOISE", None, "EXPIRED"
                else:
                    up = c > o
                    sig.settle_outcome = "UP" if up else "DOWN"
                    sig.win = (up and sig.direction == "UP") or (not up and sig.direction == "DOWN")
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "Rev2Inside 影子结算 | {} | {} | 信号根 {} | 次根 {} -> {} | win={}",
                    sig.version, self._timeframe, int(sig.signal_bar_start),
                    int(sig.target_bar_start), sig.settle_outcome,
                    sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        cutoff = int(time.time() * 1000) - self._bar_ms - PENDING_EXPIRE_MS[self._timeframe]
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(self._versions),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("Rev2Inside 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    async def _backscan(self) -> None:
        closed = await self._collector.fetch_recent_klines(self._timeframe, WARMUP_BARS)
        if len(closed) < WARMUP_BARS:
            logger.warning(
                "Rev2Inside 影子：{} 冷启动回补数据不足（{} 根），跳过",
                self._timeframe, len(closed),
            )
            return
        await self._evaluate_new_bars(closed)
        self._last_evaluated_bar = int(closed[-1]["open_time"])
        await self._settle_pending(closed)
        await self._expire_stale_pending()

    def status(self) -> dict:
        return {
            "running": self._running,
            "timeframe": self._timeframe,
            "last_evaluated_bar": self._last_evaluated_bar,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "versions": list(self._versions),
        }
