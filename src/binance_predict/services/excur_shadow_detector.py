"""布防接盘影子检测器（excur 族）：15m 布防窗口内 UP 报价首次 ≤ 挂档的实时重放。

信号定义（阈值冻结自研究 run 20261024T092511Z-window-excursion-occurrence，
FROZEN_FINAL.json 哈希链内 FINAL_REPORT/ADDENDUM §C；改值须回研究轮重新冻结）：
    布防（窗口开盘前父 15m bar 序列，严格 ex-ante，只用 open_time < window_start 的根）：
      ret6_bp   = (close_prev − close_prev6) / close_prev × 1e4 < −31.033358   （TRAIN Q20）
      dist24l_bp = (close_prev − min(low, 前 96 根)) / close_prev × 1e4 < 60.987908 （TRAIN Q20）
    触发（窗口内 5s 对齐采样喂价）：up_price ≤ 0.30 且 采样时刻 < 窗口结束 − 120s
    结算 win = close(15m 目标窗) > open（押 UP；平局 NOISE）

研究依据（2026-07→10 PREDEXON 真实盘口，722 布防窗）：T−120s 撤单口径
布防 dd@k=0.30 EV +0.0215/窗 CI[+0.013,+0.081]（成交率 65.5%、成交后胜率 33.9%、
保本线 30.6%）——20 格校准中唯一 CI 下界>0 的探索性结论，先影子前向验证。
实盘为触发时市价单（护栏 0.31），非 GTC 挂单：「不晚于 T−120s 触发」与研究的
「挂单 T−120s 撤销」在成交集合上等价（此后不触发即不成交）。

影子纪律：每布防窗落一行（PENDING）；触发时回填 entry 价格与 touched 字段；
窗口收盘结算。只落表不下注；实盘开火由 on_live_fire 钩子派发到
MultiLiveTrader.on_excur_signal（通道默认 OFF，toggle 开闸）。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal

EXCUR_VERSION = "excur_bid30_15m_v1"
# 冻结阈值（TRAIN 2024-11-02→2025-10-28 15m 窗口 Q20；Q1低桶为严格小于，见研究
# 02_rank_factors.bucketize 的 np.digitize 语义）
RET6_Q20_BP = -31.033358
DIST24L_Q20_BP = 60.987908
BID_LEVEL = 0.30
CANCEL_BEFORE_END_S = 120
BAR_15M_MS = 900_000
# 布防需要 window_start 前至少 97 根 15m（96 根 24h 低 + 1 根前收）；留冗余
WARMUP_15M = 110
SETTLE_POLL_S = 15.0


def arm_15m(bars_15m: list[dict], window_start: int) -> dict | None:
    """用 window_start 之前的父 15m 根计算布防条件（严格 ex-ante）。

    bars_15m 为 collector.fetch_recent_klines("15m", ...) 返回的已收盘根
    （dict: open_time/open/high/low/close）。样本不足 / 价格非法 → None（保守不布防）。
    返回 {"ret6_bp", "dist24l_bp"}；两条件同时满足（严格 <，与训练分桶一致）才视为布防。
    """
    prev = [b for b in bars_15m if int(b["open_time"]) + BAR_15M_MS <= window_start]
    if len(prev) < 97:
        return None
    closes = [float(b["close"]) for b in prev]
    close_prev = closes[-1]
    close_prev6 = closes[-7]
    if close_prev <= 0 or close_prev6 <= 0:
        return None
    ret6_bp = (close_prev - close_prev6) / close_prev * 1e4
    lows = [float(b["low"]) for b in prev[-96:]]
    dist24l_bp = (close_prev - min(lows)) / close_prev * 1e4
    armed = ret6_bp < RET6_Q20_BP and dist24l_bp < DIST24L_Q20_BP
    return {"ret6_bp": ret6_bp, "dist24l_bp": dist24l_bp, "armed": armed}


class ExcurBidShadowDetector:
    """excur 布防接盘影子检测器：15m 采样喂价 → 布防判定（每窗一次缓存）→
    触发回填 + 实盘钩子；后台循环结算 PENDING。"""

    def __init__(self, collector) -> None:
        self._collector = collector
        self._running = False
        self._task: asyncio.Task | None = None
        self._arm_cache: dict[int, dict | None] = {}      # window_start -> arm info（None=未布防）
        self._row_ids: dict[int, int] = {}                # window_start -> 布防行 DB id
        self._touched: set[int] = set()                   # 已触发 window_start（内存去重）
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._settle_loop(), name="excur_shadow_detector")
        logger.info("excur 布防接盘检测器启动 | {} | 影子落表 + 实盘钩子（默认关闭）",
                    EXCUR_VERSION)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("excur 检测器已停止 | 触发 {} 结算 {}", self._trigger_count,
                    self._settle_count)

    # ---- 布防判定（每窗一次，含 DB 存在性核验防重启重复落表） ----
    async def _arm_of(self, window_start: int) -> dict | None:
        if window_start in self._arm_cache:
            arm = self._arm_cache[window_start]
            return arm if (arm and arm.get("armed")) else None
        bars = await self._collector.fetch_recent_klines("15m", WARMUP_15M)
        if not bars:
            return None              # 瞬态拉取失败：不写缓存，下轮采样重试（防整窗静默丢布防）
        arm = arm_15m(bars, window_start)
        self._arm_cache[window_start] = arm
        if len(self._arm_cache) > 512:
            cutoff = window_start - 8 * BAR_15M_MS
            self._arm_cache = {k: v for k, v in self._arm_cache.items() if k >= cutoff}
            self._row_ids = {k: v for k, v in self._row_ids.items() if k >= cutoff}
            self._touched = {k for k in self._touched if k >= cutoff}
        return arm if (arm and arm.get("armed")) else None

    async def observe_sample(self, window_start: int, window_end: int,
                             ts_ms: int, up_price: float | None,
                             down_price: float | None) -> None:
        """每次 15m 对齐采样调用（main._handle_15m_quote persist 路径）。"""
        arm = await self._arm_of(window_start)
        if arm is None:
            return
        row_id = await self._ensure_armed_row(window_start, arm)
        if row_id is None:
            return
        if up_price is None or float(up_price) > BID_LEVEL:
            return
        # 撤单语义：T−120s 之后不再触发（等价研究口径的挂单撤销）
        if ts_ms >= window_end - CANCEL_BEFORE_END_S * 1000:
            return
        t_rel = round((ts_ms - window_start) / 1000.0, 1)
        async with async_session_factory() as session:
            row = await session.get(KlineShadowSignal, row_id)
            if row is None:
                return
            # 持久化去重：重启后 _touched 内存丢失，以 DB 行状态为准——
            # 已触发过（touched/entry 价已写）则只回填内存标记，不重复触发/覆写
            if ((row.feature_snapshot or {}).get("touched")
                    or row.entry_up_price is not None):
                self._touched.add(window_start)
                return
            snap = dict(row.feature_snapshot or {})
            snap["touched"] = True
            snap["touched_t_rel_s"] = t_rel
            snap["bid_level"] = BID_LEVEL
            row.feature_snapshot = snap
            row.entry_up_price = float(up_price)
            row.entry_down_price = (float(down_price)
                                    if down_price is not None else None)
            row.entry_quote_ts = ts_ms
            await session.commit()
        self._touched.add(window_start)
        self._trigger_count += 1
        logger.info("excur 影子触发 | {} | 窗口 {} | t=+{}s up={} | 布防 ret6={:.1f}bp "
                    "dist24l={:.1f}bp", EXCUR_VERSION, window_start, t_rel,
                    up_price, arm["ret6_bp"], arm["dist24l_bp"])
        if self._on_live_fire is not None:
            try:
                self._on_live_fire({
                    "version": EXCUR_VERSION,
                    "market_start": window_start,
                    "market_end": window_end,
                    "direction": "UP",
                    "signal_bar_start": window_start,
                    "touch_price": float(up_price),
                    "t_rel": t_rel,
                })
            except Exception as exc:
                logger.warning("excur 实盘钩子异常（不影响影子）| {}", exc)

    async def _ensure_armed_row(self, window_start: int, arm: dict) -> int | None:
        """布防行只落一次（DB 存在性核验 + 内存缓存，防重启/重放重复落表）。"""
        if window_start in self._row_ids:
            return self._row_ids[window_start]
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == EXCUR_VERSION,
                    KlineShadowSignal.signal_bar_start == window_start,
                ).limit(1))).first()
            if exists:
                self._row_ids[window_start] = int(exists[0])
                return int(exists[0])
            row = KlineShadowSignal(
                version=EXCUR_VERSION,
                discovery_id="excur_bid30_15m",
                condition_text=(
                    f"ret6_bp < {RET6_Q20_BP} AND dist24l_bp < {DIST24L_Q20_BP}"
                    f" AND up_price <= {BID_LEVEL} AND t < end - {CANCEL_BEFORE_END_S}s"),
                timeframe="15m",
                signal_bar_start=window_start,
                signal_bar_end=window_start + BAR_15M_MS,
                direction="UP",
                target_bar_start=window_start,
                feature_snapshot={
                    "armed": True,
                    "ret6_bp": round(arm["ret6_bp"], 4),
                    "dist24l_bp": round(arm["dist24l_bp"], 4),
                    "cancel_before_end_s": CANCEL_BEFORE_END_S,
                },
                status="PENDING",
            )
            session.add(row)
            await session.commit()
            self._row_ids[window_start] = int(row.id)
            return int(row.id)

    async def _settle_loop(self) -> None:
        """后台结算：目标 15m 根收盘后按 close vs open 结算布防行（含未触发行）。"""
        while self._running:
            try:
                await self._settle_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("excur 影子：结算循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(SETTLE_POLL_S)
            except asyncio.CancelledError:
                break

    async def _settle_pending(self) -> None:
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == EXCUR_VERSION,
                    KlineShadowSignal.status == "PENDING",
                ))).scalars().all()
            if not rows:
                return
            # 回读窗口按最老 PENDING 行动态扩大（同 wbid：积压行补结算而非
            # EXPIRED 兜底；上限 3 天 = 288 根 15m）。
            oldest = min(int(r.target_bar_start) for r in rows)
            need = max(8, min((now_ms - oldest) // BAR_15M_MS + 2, 288))
            closed15 = await self._collector.fetch_recent_klines("15m", need)
            # bar 无 close_time 字段（同 wbid 2026-10-25 生产修复）：收盘时刻 =
            # open_time + 15m 推导；close_time KeyError 会使结算循环每 15s 崩溃。
            by_start = {int(b["open_time"]): b for b in closed15} if closed15 else {}
            for r in rows:
                bar = by_start.get(int(r.target_bar_start))
                if bar is None or int(bar["open_time"]) + BAR_15M_MS > now_ms:
                    if now_ms - int(r.target_bar_start) > 10 * BAR_15M_MS:
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
