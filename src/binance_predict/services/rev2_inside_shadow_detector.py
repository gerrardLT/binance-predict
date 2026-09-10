"""15m 经典孕线上吊线/倒垂线反转影子检测器与实盘驱动（rev2 族）。

信号定义（与 720d 回测严格对齐，禁止手抄/篡改阈值）：
    1. hm_inside_15m_v2（15m 顶部长下影上吊线被孕线包裹 -> 押次根 15m 收阴 DOWN）：
       - 前一根 K 线（Bar N-1）：实体阳线（is_green 且 body_r >= 0.40），且为近 4 根明显最高点，且实体长度大于前 3 根实体
       - 信号柱（Bar N）：完全被前一根包裹（high <= prev_high 且 low >= prev_low）
       - 几何特征：下影线 lower_r >= 0.45，实体 body_r <= 0.45，次影线 upper_r <= 0.25
       - 押注方向：次根 15m 押 DOWN（720d 胜率 56.6%，30d 胜率 65.4%）
       
    2. ih_inside_15m_v2（15m 底部长上影倒垂线被孕线包裹 -> 押次根 15m 收阳 UP）：
       - 前一根 K 线（Bar N-1）：实体阴线（not is_green 且 body_r >= 0.40），且为近 4 根明显最低点，且实体长度大于前 3 根实体
       - 信号柱（Bar N）：完全被前一根包裹（high <= prev_high 且 low >= prev_low）
       - 几何特征：上影线 upper_r >= 0.45，实体 body_r <= 0.45，次影线 lower_r <= 0.25
       - 押注方向：次根 15m 押 UP（720d 胜率 52.5%，30d 胜率 62.5%）

实盘下单执行与护栏策略：
    - 最佳挂单护栏：0.30（网格优化显示 0.30 兼顾 65.7% 极高成交率与 +18.3% 单笔期望 EV，累计收益与获利因子最高）
    - 驱动机制：新 15m 收盘判定命中后，触发 _on_live_fire 钩子直连 MultiLiveTrader
    - 幂等防重：(version, signal_bar_start) 数据库唯一约束，避免重复开火
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

REV2_SHADOW_SPECS: list[dict] = [
    {
        "version": "hm_inside_15m_v2",
        "discovery_id": "rev2_hm_15m",
        "timeframe": "15m",
        "direction": "DOWN",
        "condition_text": (
            "prev_is_green == True AND prev_body_r >= 0.40 AND is_prominent_high "
            "AND is_longest_body AND is_full_inside AND lower_r >= 0.45 AND body_r <= 0.45 AND upper_r <= 0.25"
        ),
    },
    {
        "version": "ih_inside_15m_v2",
        "discovery_id": "rev2_ih_15m",
        "timeframe": "15m",
        "direction": "UP",
        "condition_text": (
            "prev_is_green == False AND prev_body_r >= 0.40 AND is_prominent_low "
            "AND is_longest_body AND is_full_inside AND upper_r >= 0.45 AND body_r <= 0.45 AND lower_r <= 0.25"
        ),
    },
]

REV2_VERSIONS = [s["version"] for s in REV2_SHADOW_SPECS]
BAR_MS_15M = 900_000
POLL_INTERVAL = 60.0
WARMUP_BARS = 40
BACKSCAN_BARS = 12
PENDING_EXPIRE_MS = 4 * 3_600_000
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


def evaluate_rev2_patterns(kl: Klines, n_tail: int) -> list[dict]:
    """严谨求值 15m Ver 2 孕线反转形态。
    
    返回末 n_tail 根内的触发点列表。
    """
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

    # 逐根布尔数组
    hits = []
    start_eval = max(4, n - n_tail)

    for i in range(start_eval, n):
        prev_is_green = is_green[i - 1]
        prev_body = body[i - 1]
        prev_body_r = body_r[i - 1]
        prev_h = h[i - 1]
        prev_l = l[i - 1]

        # 1. 明显极值点：前置高点 >= 前4根最高，或前置低点 <= 前4根最低
        # 前 4 根切片: i-4 .. i-1
        is_prominent_high = prev_h >= np.max(h[max(0, i - 4):i])
        is_prominent_low = prev_l <= np.min(l[max(0, i - 4):i])

        # 2. 实体是附近最长：prev_body >= 前3根实体
        is_longest_body = prev_body >= np.max(body[max(0, i - 4):i - 1])

        # 3. 完全包裹孕线（允许 0.02% 贴线微差）
        is_full_inside = (h[i] <= prev_h * 1.0002) and (l[i] >= prev_l * 0.9998)

        if not (is_longest_body and is_full_inside):
            continue

        # 检查上吊线 HM
        if (
            prev_is_green
            and prev_body_r >= 0.40
            and is_prominent_high
            and lower_r[i] >= 0.45
            and body_r[i] <= 0.45
            and upper_r[i] <= 0.25
        ):
            hits.append({
                "spec": REV2_SHADOW_SPECS[0],
                "idx": i,
                "snapshot": {
                    "prev_body_r": round(float(prev_body_r), 4),
                    "body_r": round(float(body_r[i]), 4),
                    "lower_r": round(float(lower_r[i]), 4),
                    "upper_r": round(float(upper_r[i]), 4),
                }
            })

        # 检查倒垂线 IH
        if (
            (not prev_is_green)
            and prev_body_r >= 0.40
            and is_prominent_low
            and upper_r[i] >= 0.45
            and body_r[i] <= 0.45
            and lower_r[i] <= 0.25
        ):
            hits.append({
                "spec": REV2_SHADOW_SPECS[1],
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
    """15m Ver2 孕线反转形态影子检测器：轮询 15m 收盘 -> 判定落表 -> 驱动实盘。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        self._pm_15m = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: int | None = None
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None  # 实盘钩子，由 main 注入

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._backscan()
        except Exception as exc:
            logger.warning("Rev2Inside 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="rev2_inside_shadow_detector")
        logger.info(
            "Rev2Inside 15m 孕线反转检测器启动 | {} | 影子落表+实盘支持就绪",
            "/".join(REV2_VERSIONS),
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
        logger.info("Rev2Inside 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            return
        last_start = int(closed_15m[-1]["open_time"])
        if self._last_evaluated_bar is None or last_start > self._last_evaluated_bar:
            await self._evaluate_new_bars(closed_15m)
            self._last_evaluated_bar = last_start
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()
        await self._check_radar()

    async def _check_radar(self) -> None:
        """提前预警雷达：探测当前正在走的 15m K 线是否初具孕线反转雏形。"""
        if not settings.wechat_radar_enabled or not settings.wechat_work_enabled:
            return
        try:
            raw_15m = await self._collector.fetch_klines_raw("15m", WARMUP_BARS)
            if len(raw_15m) < 5:
                return
            current_bar = raw_15m[-1]
            now_ms = int(time.time() * 1000)
            bar_start = int(current_bar["open_time"])
            bar_end = bar_start + BAR_MS_15M
            rem_sec = int((bar_end - now_ms) / 1000)

            # 仅在收盘前 45s ~ 150s 区间进行雷达检测并推送预警
            if not (45 <= rem_sec <= 150):
                return

            kl15 = _to_klines(raw_15m, BAR_MS_15M)
            hits = evaluate_rev2_patterns(kl15, 1)
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

    async def _evaluate_new_bars(self, closed_15m: list[dict]) -> None:
        kl15 = _to_klines(closed_15m, BAR_MS_15M)
        if self._last_evaluated_bar is None:
            n_tail = BACKSCAN_BARS
        else:
            starts = [int(r["open_time"]) for r in closed_15m]
            try:
                first_new = next(i for i, s in enumerate(starts) if s > self._last_evaluated_bar)
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)

        hits = evaluate_rev2_patterns(kl15, n_tail)
        if not hits:
            return

        live_payloads: list[dict] = []
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed_15m[hit["idx"]]
                last_bar = bar
                if await self._record_signal(session, hit["spec"], bar, hit["snapshot"], live_payloads):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("Rev2Inside 影子触发 +{} | 信号根 {}", added, int(last_bar["open_time"]))

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

        target_bar_start = start_ms + BAR_MS_15M
        # 实盘开火收集：仅目标根刚开盘 <= 90s 内的新鲜命中
        if live_payloads is not None and (0 <= int(time.time() * 1000) - target_bar_start <= REV2_LIVE_MAX_LAG_MS):
            live_payloads.append({
                "version": spec["version"],
                "market_start": target_bar_start,
                "market_end": target_bar_start + BAR_MS_15M,
                "direction": spec["direction"],
                "signal_bar_start": start_ms,
            })

        if not shadow_gate.is_enabled(spec["version"]):
            return False

        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_15m, target_bar_start)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe="15m",
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

    async def _settle_pending(self, closed_15m: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed_15m}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(REV2_VERSIONS),
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
                    "Rev2Inside 影子结算 | {} | 信号根 {} | 次根 {} -> {} | win={}",
                    sig.version, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(REV2_VERSIONS),
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
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            logger.warning("Rev2Inside 影子：冷启动回补数据不足（{} 根），跳过", len(closed_15m))
            return
        await self._evaluate_new_bars(closed_15m)
        self._last_evaluated_bar = int(closed_15m[-1]["open_time"])
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_evaluated_bar": self._last_evaluated_bar,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "versions": REV2_VERSIONS,
        }
