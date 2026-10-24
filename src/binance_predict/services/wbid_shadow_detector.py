"""窗口内深价接盘影子检测器（wbid 族）：5m 窗口内位面触及的实时重放。

信号定义（阈值冻结自 output/research_runs/20261023T133520Z-intrawindow-path，
EXEC_SPEC.md §4，禁止手抄渲染值）：
    位面 level_d = open(W) - d * ATR5(t0)，ATR5 为前一根 5m bar 的 ATR14（shift(1)）；
    k = 窗口内第一根 low <= level_d 的 1m bar 序号（1..5）；
    speed = (open(W) - close(bar_{k-1})) / ATR5 / k；
    结算 win = close(W) > open(W)（押 UP）。

四个版本：
    wbid_d03_k2_5m_v1        d=0.3 k=2        挂价 0.21
    wbid_d03_k3_5m_v1        d=0.3 k=3        挂价 0.16
    wbid_d05_k2_5m_v1        d=0.5 k=2        挂价 0.14
    wbid_d03_k2_fast_5m_v1   d=0.3 k=2 急跌   挂价 0.21（speed > 0.032876）

影子纪律：只落表不下注；实盘开火由注入的 on_live_fire 钩子派发到
MultiLiveTrader.on_wbid_signal，通道 enabled 由 toggle 管（默认关闭）。
与 nextbar 族不同：wbid 的 target 就是**当前窗口**（signal_bar_start ==
target_bar_start），结算等当前 5m 根收盘。
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
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote

POLL_INTERVAL = 15.0
BAR_1M_MS = 60_000
BAR_5M_MS = 300_000
# ATR14 需要 t0 之前 ≥15 个完整 5m 窗口 = 75 根 1m，再加当前窗口余量；
# 40 根只够 8 个窗口 → _atr5_at 恒 None → 检测器静默不触发（CodeReview Blocker）。
WARMUP_1M = 90
WARMUP_5M = 6

# 冻结阈值（train 段中位数，EXEC_SPEC §4）
SPEED_THR = {0.3: 0.032876, 0.5: 0.082265}

WBID_SPECS: list[dict] = [
    {"version": "wbid_d03_k2_5m_v1", "d": 0.3, "k": 2, "speed": None, "limit": 0.21},
    {"version": "wbid_d03_k3_5m_v1", "d": 0.3, "k": 3, "speed": None, "limit": 0.16},
    {"version": "wbid_d05_k2_5m_v1", "d": 0.5, "k": 2, "speed": None, "limit": 0.14},
    {"version": "wbid_d03_k2_fast_5m_v1", "d": 0.3, "k": 2,
     "speed": "fast", "limit": 0.21},
]
WBID_VERSIONS = tuple(s["version"] for s in WBID_SPECS)


def _atr5_at(closed_1m: list[dict], t0: int) -> float | None:
    """前一根 5m bar 的 ATR14（shift(1)）；样本不足返回 None（保守不触发）。"""
    agg: dict[int, list[float]] = {}
    for b in closed_1m:
        w = int(b["open_time"]) // BAR_5M_MS * BAR_5M_MS
        o, h, l, c = (float(b["open"]), float(b["high"]),
                      float(b["low"]), float(b["close"]))
        a = agg.setdefault(w, [o, h, l, c])
        a[1] = max(a[1], h)
        a[2] = min(a[2], l)
        a[3] = c
    starts = sorted(agg)
    prev = [s for s in starts if s < t0]
    if len(prev) < 15:
        return None
    closes = [agg[s][3] for s in starts]
    trs = []
    for i, s in enumerate(starts):
        o, h, l, c = agg[s]
        pc = closes[i - 1] if i > 0 else np.nan
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    tr = np.asarray(trs, float)
    alpha = 1 / 14
    # shift(1)：只累积到 t0 前一根（索引 i），不含当前窗口的 TR
    idx = prev[-1]
    i = starts.index(idx)
    a = tr[0]
    for v in tr[1:i + 1]:
        a = alpha * v + (1 - alpha) * a
    return float(a)


def evaluate_window(open_price: float, atr5: float, bars: list[dict],
                    specs: list[dict]) -> list[dict]:
    """对当前窗口已收盘的 1m 根求值；返回命中 [{"spec","k","speed","level"}]。"""
    hits = []
    lo = [float(b["low"]) for b in bars]
    cl = [float(b["close"]) for b in bars]
    for spec in specs:
        level = open_price - spec["d"] * atr5
        k = 0
        for i, v in enumerate(lo):
            if v <= level:
                k = i + 1
                break
        if k != spec["k"]:
            continue
        if k < 2:
            continue
        speed = (open_price - cl[k - 2]) / atr5 / k
        if spec["speed"] == "fast" and speed <= SPEED_THR[spec["d"]]:
            continue
        hits.append({"spec": spec, "k": k, "speed": speed, "level": level})
    return hits


class WbidShadowDetector:
    """wbid 影子检测器：15s 轮询 1m 收盘根 → 位面触及判定 → 落表 + 实盘钩子。"""

    def __init__(self, collector, pm_5m_info: dict) -> None:
        self._collector = collector
        self._pm_5m = pm_5m_info
        self._running = False
        self._task: asyncio.Task | None = None
        self._fired: set[tuple[str, int]] = set()
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="wbid_shadow_detector")
        logger.info("wbid 窗口内接盘检测器启动 | {} | 影子落表 + 实盘钩子（默认关闭）",
                    "/".join(WBID_VERSIONS))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("wbid 检测器已停止 | 触发 {} 结算 {}",
                    self._trigger_count, self._settle_count)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("wbid 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        now_ms = int(time.time() * 1000)
        t0 = now_ms // BAR_5M_MS * BAR_5M_MS
        closed = await self._collector.fetch_recent_klines("1m", WARMUP_1M)
        if not closed:
            return
        bars = [b for b in closed
                if t0 <= int(b["open_time"]) < now_ms]
        # 窗口首根必须已收盘才有 open；k>=2 要求至少两根收盘
        if not bars or int(bars[0]["open_time"]) != t0 or len(bars) < 2:
            bars = None
        if bars is not None:
            atr5 = _atr5_at(closed, t0)
            if atr5:
                open_price = float(bars[0]["open"])
                for hit in evaluate_window(open_price, atr5, bars, WBID_SPECS):
                    await self._record(t0, open_price, atr5, hit)
        await self._settle_pending(now_ms)
        # _fired 上限清理：已结算/过期窗口不会再触发，剔除 2h 前的 key
        if len(self._fired) > 4096:
            cutoff = now_ms - 24 * BAR_5M_MS
            self._fired = {kk for kk in self._fired if kk[1] >= cutoff}

    async def _record(self, t0: int, open_price: float, atr5: float,
                      hit: dict) -> None:
        spec = hit["spec"]
        key = (spec["version"], t0)
        if key in self._fired:
            return
        # wbid 在 k≥2 才触发（offset ≥120s），默认 120s 近开盘守卫会把报价
        # 全部判 NULL → 放宽到整个窗口（CodeReview High）。
        up, down, qts = snapshot_entry_quote(self._pm_5m, t0,
                                             max_offset_ms=BAR_5M_MS)
        k = hit["k"]
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == spec["version"],
                    KlineShadowSignal.signal_bar_start == t0,
                ).limit(1))).first()
            if exists:
                self._fired.add(key)
                return
            session.add(KlineShadowSignal(
                version=spec["version"],
                discovery_id=f"wbid_5m_d{spec['d']:.1f}_k{k}",
                condition_text=(f"low <= open - {spec['d']}*ATR5 AND k == {k}"
                                + (" AND speed > thr" if spec["speed"] else "")),
                timeframe="5m",
                signal_bar_start=t0,
                signal_bar_end=t0 + k * BAR_1M_MS,
                direction="UP",
                target_bar_start=t0,
                feature_snapshot={"k": k, "speed": round(hit["speed"], 6),
                                  "atr5": round(atr5, 4),
                                  "level": round(hit["level"], 4),
                                  "open": round(open_price, 4),
                                  "limit_price": spec["limit"]},
                entry_up_price=up, entry_down_price=down, entry_quote_ts=qts,
                status="PENDING",
            ))
            await session.commit()
        # commit 成功后才标记，避免瞬态 DB 故障永久丢该窗口（CodeReview Medium）
        self._fired.add(key)
        self._trigger_count += 1
        logger.info("wbid 影子触发 | {} | 窗口 {} | k={} speed={:.4f} | ask={}",
                    spec["version"], t0, k, hit["speed"], up)
        if self._on_live_fire is not None:
            try:
                self._on_live_fire({
                    "version": spec["version"],
                    "market_start": t0,
                    "market_end": t0 + BAR_5M_MS,
                    "direction": "UP",
                    "signal_bar_start": t0,
                    "price_limit": spec["limit"],
                    "k": k, "speed": hit["speed"],
                })
            except Exception as exc:
                logger.warning("wbid 实盘钩子异常（不影响影子）| {}", exc)

    async def _settle_pending(self, now_ms: int) -> None:
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(WBID_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                ))).scalars().all()
            if not rows:
                return
            # 回读窗口按最老 PENDING 行动态扩大：修复部署前积压的行（如
            # close_time KeyError 期间卡死的 7 小时）也能补结算，而不是被
            # EXPIRED 兜底吞掉（老行结算完 limit 自动回落 WARMUP_5M；上限
            # 3 天防异常行无限拉取）。
            oldest = min(int(r.target_bar_start) for r in rows)
            need = max(WARMUP_5M, min((now_ms - oldest) // BAR_5M_MS + 2, 864))
            closed5 = await self._collector.fetch_recent_klines("5m", need)
            # fetch_recent_klines 的 bar 只有 open_time/open/high/low/close/volume，
            # 无 close_time 字段——收盘时刻 = open_time + 周期长度推导（2026-10-25
            # 生产修复：bar["close_time"] KeyError 曾使结算循环每 15s 崩溃，
            # wbid 触发行全部卡死 PENDING，无法进入任何统计视图）。
            by_start = {int(b["open_time"]): b for b in closed5}
            for r in rows:
                bar = by_start.get(int(r.target_bar_start))
                if bar is None or int(bar["open_time"]) + BAR_5M_MS > now_ms:
                    if now_ms - int(r.target_bar_start) > 10 * BAR_5M_MS:
                        r.status = "EXPIRED"
                    continue
                o, c = float(bar["open"]), float(bar["close"])
                r.settle_open, r.settle_close = o, c
                r.settle_outcome = "UP" if c > o else ("DOWN" if c < o else "NOISE")
                r.win = bool(c > o) if c != o else None
                r.status = "SETTLED"
                r.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
            await session.commit()
