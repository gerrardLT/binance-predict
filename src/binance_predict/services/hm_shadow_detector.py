"""弱收盘上吊线反弹入场影子检测器（hm_touch_down_v1 / v2）：窗内等价位触发 → 押 15m 收跌。

冻结规则（预注册口径，源自 720d 探索研究，审计锚点
scripts/hm_touch_entry_research_720d.py）：

触发（15m 收盘判定，几何阈值一律 ×ATR20）：
    实体≤0.3×ATR ∧ 下影≥2×实体 ∧ 下影≥0.3×ATR ∧ 上影≤0.15×ATR
    ∧ 收盘距近 20 根最高（含本根）≤0.75×ATR
    ∧ CLV=(close−low)/(high−low)≤0.75（range≤0 不触发）
    ∧ 信号根与前根连续（cont 守卫）
    ATR = discovery.features.atr_series（前 20 根 range% 均值 ×open，ex-ante；
    非经典 Wilder、非 ATR14，禁止本地重写）。

入场状态机（目标周期=信号次 15m，锚点 O=目标周期开盘价，
fetch_kline_open("15m", target_start) 回读）：
    触及：mid ≥ O+0.25×ATR 且发生在周期开始后 600s 内且此前未破下障碍 → 押 DOWN；
    放弃：先破 O−0.25×ATR（ABANDON_LOWER）；触及在 600s 后（ABANDON_LATE）；
    900s 未触（NOT_TOUCHED）。
    结算（仅 TOUCHED）：目标根 close<open = win；平盘 → NOISE/EXPIRED。
    实时裁决用 ~2s 轮询 collector.store.mid_price（bookTicker WS 已维护），
    禁止用 5m K 线判触（丢棒内先后顺序不可知）；喂价超龄 >10s → FEED_GAP 保守弃。
    重启跨窗：窗口未结束重派监控任务（已过时段用 1m K 线重建回放）；
    已结束整窗 1m 重建，缺棒/棒内双触不可判 → RESTART_GAP 弃，不回灌报价。

参考基准（720d，x=0.25 格子）：触及收跌率 58.7% vs 市场隐含 47.1%（n=46），
覆盖率 ~36%（127 次信号仅 ~46 次等到入场点）。注意：这是从 15 个探索性格子
中挑出的（精确二项 p≈0.06），前向验证正是影子期的目的。

v2 门禁（2026-09-01 切片分析后验假设，与 v1 并行双行落库）：
    触发时点（形态根收盘）额外要求：
    ① 非下跌段：ret24 = c[i]/c[i−96] − 1 > −1.0%（下跌段样本 41.7% < 基线，负边际）；
    ② 非低波：ATR / 前 24h ATR 中位数 ≥ 0.8（窗口有限值须 >20；低波样本 25%）。
    720d 冻结：门禁后触发 78 / 触价 29，触价收跌 69.0%（后验切片，非预注册，
    影子期前向验证决定是否保留）。入场/结算/放弃状态机与 v1 完全一致。
    审计锚点：.pytest_tmp/hm_slice_followup.py + hm_v2_freeze_counts.py。

影子纪律：只记录不下注、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，新表
pattern_shadow_signals 不被任何下单代码引用（物理隔离）。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view
from sqlalchemy import select as sa_select, update as sa_update

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import PatternShadowSignal
from binance_predict.discovery.data import Klines
from binance_predict.discovery.features import atr_series
from binance_predict.services import clock_sync

VERSION = "hm_touch_down_v1"
VERSION2 = "hm_touch_down_v2"
ALL_VERSIONS = (VERSION, VERSION2)  # 入场/裁决/结算双版本同步（v2 = v1 ∩ 门禁）
BAR_MS_15M = 900_000
BAR_MS_1M = 60_000
ATR_WINDOW = 20                 # ATR20：前 20 根 range% 均值（与 atr_series 默认 w 一致）
POLL_INTERVAL = 60.0            # 收盘轮询间隔（秒）
WARMUP_BARS = 120               # 15m 历史根数（v2 门禁需 96 根回看 + 12 回补余量）
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数（3 小时）
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED（兜底）

# ---- 冻结入场参数（预注册，勿动）----
ENTRY_X = 0.25                  # 对称障碍 ±0.25×ATR（720d 甜点，U 型曲线的底）
TOUCH_DEADLINE_S = 600          # 触及死线：周期开始 600s 内（前 10 分钟）才有效
FEED_STALE_S = 10.0             # bookTicker 喂价超龄阈值（秒）→ FEED_GAP 保守弃
MONITOR_POLL_S = 2.0            # 实时裁决轮询间隔（秒）
OPEN_RETRY_S = 5.0              # 目标周期开盘价回读重试间隔（秒）
QUOTE_MAX_AGE_MS = 20_000       # TOUCHED 报价快照可接受的最大龄（tracker 15s 采样）
REPLAY_1M_LIMIT = 60            # 重启重建最多回拉 60 根 1m（更早窗口 → RESTART_GAP）
CLV_MAX = 0.75                  # 弱收盘判据

# ---- v2 门禁参数（2026-09-01 切片分析冻结，勿动）----
V2_RET24_MIN = -0.01            # 非下跌段：过去 24h 涨幅 > −1.0%
V2_ATR_RATIO_MIN = 0.8          # 非低波：ATR / 前 24h ATR 中位数 ≥ 0.8
V2_REGIME_WINDOW = 96           # 24h = 96 根 15m

RULE_TEXT = (
    "hm_touch_down_v1 预注册冻结规则（720d 探索研究，2026-09 冻结）："
    "触发（15m 收盘，×ATR20，ATR=atr_series 前20根range%均值×open ex-ante）："
    "实体≤0.3×ATR ∧ 下影≥2×实体 ∧ 下影≥0.3×ATR ∧ 上影≤0.15×ATR ∧ "
    "收盘距近20根最高（含本根）≤0.75×ATR ∧ CLV=(close−low)/(high−low)≤0.75"
    "（range≤0 不触发）∧ 信号根与前根连续。"
    "入场状态机（目标=次15m，锚点 O=目标周期开盘 fetch_kline_open 回读）："
    "mid≥O+0.25×ATR 且发生于周期开始 600s 内且此前未破下障碍 → 押 DOWN（15m 市场）；"
    "先破 O−0.25×ATR（ABANDON_LOWER）/ 触及在 600s 后（ABANDON_LATE）/ "
    "900s 未触（NOT_TOUCHED）放弃；喂价超龄>10s 标 FEED_GAP 保守弃；"
    "重启 1m 重建，不确定标 RESTART_GAP 弃，不回灌。"
    "结算仅 TOUCHED：目标根 close<open = win；平盘 NOISE/EXPIRED。"
    "参考基准：触及收跌率 58.7% vs 隐含 47.1%（720d n=46），覆盖率 ~36%。"
)

RULE_TEXT_V2 = (
    "hm_touch_down_v2 冻结规则（2026-09-01 切片分析后验假设，与 v1 并行双行落库）："
    "v1 全部触发与入场规则之外，触发时点额外门禁："
    "① 非下跌段：ret24 = close/close[−24h] − 1 > −1.0%；"
    "② 非低波：ATR20 / 前 24h ATR 中位数 ≥ 0.8（窗口有限值 >20）。"
    "720d 冻结：门禁后触发 78 / 触价 29，触价收跌 69.0%（20/29）——"
    "后验切片非预注册，影子期前向验证决定是否保留。"
    "入场/结算/放弃状态机与 v1 完全一致（同锚点同障碍同裁决）。"
)


# ----------------------------------------------------------------------
# 纯函数（实时 / 回补 / 测试共用，口径单一事实源）
# ----------------------------------------------------------------------

def _to_klines(rows: list[dict], bar_ms: int) -> Klines:
    """data_collector 的 K 线 dict 列表 → discovery.data.Klines（升序、已收盘）。"""
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


def clv_series(kl: Klines) -> np.ndarray:
    """CLV=(close−low)/(high−low)；range≤0 → NaN（冻结规则：不触发）。"""
    rng = kl.h - kl.l
    clv = np.full(len(kl.c), np.nan)
    ok = rng > 0
    clv[ok] = (kl.c[ok] - kl.l[ok]) / rng[ok]
    return clv


def detect_weak_hm(kl: Klines, atr: np.ndarray) -> np.ndarray:
    """弱收盘上吊线触发判定（向量化，含 cont 守卫）。

    与 scripts/hanging_man_combo_720d.py detect_hm（L101-114）+
    hm_touch_entry_research_720d.py 弱收盘过滤逐位一致：
    top20 含本根（sliding_window_view(h,20).max），ATR 用 atr_series 原值。
    """
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    n = len(c)
    top20 = np.full(n, np.nan)
    if n >= ATR_WINDOW:
        top20[ATR_WINDOW - 1:] = sliding_window_view(h, ATR_WINDOW).max(axis=1)
    shape = ((body <= 0.3 * atr_s) & (lower >= 2.0 * body) & (lower >= 0.3 * atr_s)
             & (upper <= 0.15 * atr_s))
    pos = np.zeros(n, dtype=bool)
    if n >= ATR_WINDOW:
        pos[ATR_WINDOW - 1:] = c[ATR_WINDOW - 1:] >= top20[ATR_WINDOW - 1:] - 0.75 * atr_s[ATR_WINDOW - 1:]
    clv = clv_series(kl)
    weak = np.isfinite(clv) & (clv <= CLV_MAX)
    return shape & pos & np.isfinite(atr_s) & weak & kl.cont


def v2_gate_mask(kl: Klines, atr: np.ndarray) -> np.ndarray:
    """v2 门禁（向量化）：非下跌段 ∧ 非低波；历史不足一律不通过。

    与 .pytest_tmp/hm_slice_followup.py 口径逐字一致：
    ret24 = c[i]/c[i−96] − 1 > −0.01；atr_ratio = atr[i]/median(atr[i−96:i]) ≥ 0.8，
    中位数窗口内有限值须 >20 否则视为不可判（不通过）。
    """
    n = len(kl.c)
    m = np.zeros(n, dtype=bool)
    if n <= V2_REGIME_WINDOW:
        return m
    ret24 = np.full(n, np.nan)
    ret24[V2_REGIME_WINDOW:] = kl.c[V2_REGIME_WINDOW:] / kl.c[:-V2_REGIME_WINDOW] - 1
    # median(atr[i−96:i])：滑窗行 k = atr[k:k+96] → med[k+96]（窗口不含本根）
    rows = sliding_window_view(atr, V2_REGIME_WINDOW)[: n - V2_REGIME_WINDOW]
    fin = np.isfinite(rows).sum(axis=1)
    med = np.full(n, np.nan)
    good = fin > 20
    if good.any():
        med[V2_REGIME_WINDOW:][good] = np.nanmedian(rows[good], axis=1)
    ratio = np.full(n, np.nan)
    ok = np.isfinite(atr) & np.isfinite(med) & (med > 0)
    ratio[ok] = atr[ok] / med[ok]
    m[V2_REGIME_WINDOW:] = ((ret24[V2_REGIME_WINDOW:] > V2_RET24_MIN)
                            & (ratio[V2_REGIME_WINDOW:] >= V2_ATR_RATIO_MIN))
    return m


def entry_decision(mid: float, up_level: float, dn_level: float, t_rel_s: float) -> str:
    """单样本入场裁决（纯函数）。

    返回 WAIT / TOUCHED / ABANDON_LOWER / ABANDON_LATE。
    下障碍先判（保守）：同采样双触或退化区间（up≤dn）一律 ABANDON_LOWER。
    触及死线含边界：t_rel≤600s 为有效触及。
    """
    if up_level <= dn_level or mid <= dn_level:
        return "ABANDON_LOWER"
    if mid >= up_level:
        return "TOUCHED" if t_rel_s <= TOUCH_DEADLINE_S else "ABANDON_LATE"
    return "WAIT"


def atr_for_target(closed_rows: list[dict], target_open: float) -> float | None:
    """目标根位置的 ATR（= 前 20 根已收盘 range% 均值 × 目标根开盘价）。

    与 atr_series 在目标根下标处的值逐位一致（回测 lev=o15±x×atr[j15] 同口径）；
    不足 20 根 / 全平盘 / 非法开盘 → None。
    """
    tail = closed_rows[-ATR_WINDOW:]
    if len(tail) < ATR_WINDOW or target_open <= 0:
        return None
    vals = [(r["high"] - r["low"]) / r["open"] for r in tail
            if r["high"] > r["low"] and r["open"] > 0]
    if not vals:
        return None
    return float(np.mean(vals) * target_open)


class HmShadowDetector:
    """HM 反弹入场影子检测器：15m 收盘触发 → 窗内 2s 裁决 → 落表，全程不下注。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        self._pm_15m = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        self._watchers: set[asyncio.Task] = set()
        self._watched: set[int] = set()  # 已派监控任务的目标周期起点（防重派）
        self._last_evaluated_bar: int | None = None
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
            logger.warning("HM 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="hm_shadow_detector")
        logger.info(
            "HM 上吊线反弹影子检测器启动 | {} + {}（非下跌段∧非低波门禁，720d 冻结 78 触发/"
            "29 触价/69.0%）| 预注册基准 720d n=46 触价收跌 58.7% vs 隐含 47.1% | "
            "频率 ~5 天 1 触发（影子模式：只记录不下注）",
            VERSION, VERSION2,
        )

    async def stop(self) -> None:
        self._running = False
        tasks = [t for t in ([self._task, *self._watchers]) if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._task = None
        self._watchers.clear()
        self._watched.clear()
        logger.info("HM 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("HM 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            return  # 拉取失败/不足，下轮重试
        last_start = int(closed_15m[-1]["open_time"])
        if self._last_evaluated_bar is None or last_start > self._last_evaluated_bar:
            await self._evaluate_new_bars(closed_15m)
            self._last_evaluated_bar = last_start
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, closed_15m: list[dict]) -> None:
        """评估 _last_evaluated_bar 之后的新收盘根（含冷启动回补的末 12 根）。"""
        kl = _to_klines(closed_15m, BAR_MS_15M)
        atr = atr_series(kl)
        mask = detect_weak_hm(kl, atr)
        gate = v2_gate_mask(kl, atr)
        clv = clv_series(kl)
        if self._last_evaluated_bar is None:
            first_idx = max(0, len(closed_15m) - BACKSCAN_BARS)  # 冷启动：回补最近 12 根
        else:
            starts = [int(r["open_time"]) for r in closed_15m]
            try:
                first_idx = next(i for i, s in enumerate(starts) if s > self._last_evaluated_bar)
            except StopIteration:
                return
            # 长停机不追全史，最多回补 12 根
            first_idx = max(first_idx, len(closed_15m) - BACKSCAN_BARS)
        recorded: list[dict] = []
        async with async_session_factory() as session:
            added = 0
            for i in range(first_idx, len(closed_15m)):
                if not bool(mask[i]):
                    continue
                if await self._record_signal(session, closed_15m[i], float(atr[i]), float(clv[i])):
                    added += 1
                    recorded.append(closed_15m[i])
                    # v2 = v1 ∩ 门禁：门禁通过时同触发再落一行（同状态机同步裁决）
                    if bool(gate[i]):
                        await self._record_signal(
                            session, closed_15m[i], float(atr[i]), float(clv[i]),
                            version=VERSION2, rule_text=RULE_TEXT_V2,
                        )
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info(
                    "HM 影子触发 +{} | 信号根 {}",
                    added, [int(b["open_time"]) for b in recorded],
                )
        for bar in recorded:
            self._spawn_watcher(int(bar["open_time"]), int(bar["open_time"]) + BAR_MS_15M)

    async def _record_signal(self, session, bar: dict, atr_val: float, clv_val: float,
                             version: str = VERSION, rule_text: str = RULE_TEXT) -> bool:
        """幂等落 PENDING：唯一约束 (version, signal_bar_start) + 先查后插。"""
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(PatternShadowSignal.id).where(
                PatternShadowSignal.version == version,
                PatternShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        session.add(PatternShadowSignal(
            version=version,
            rule_text=rule_text,
            signal_bar_start=start_ms,
            signal_bar_end=start_ms + BAR_MS_15M,
            target_bar_start=start_ms + BAR_MS_15M,
            atr_snapshot=atr_val,
            signal_bar_open=float(bar["open"]),
            signal_bar_close=float(bar["close"]),
            clv=clv_val,
            entry_state="WAITING",
            status="PENDING",
        ))
        return True

    # ------------------------------------------------------------------
    # 入场监控（目标周期生命周期内的后台任务）
    # ------------------------------------------------------------------

    def _spawn_watcher(self, signal_bar_start: int, target_start: int) -> None:
        if target_start in self._watched:
            return
        task = asyncio.create_task(
            self._watch_entry(signal_bar_start, target_start),
            name=f"hm_shadow_watch_{target_start}",
        )
        self._watched.add(target_start)
        self._watchers.add(task)

        def _cleanup(_t: asyncio.Task) -> None:
            self._watchers.discard(_t)
            self._watched.discard(target_start)

        task.add_done_callback(_cleanup)

    async def _watch_entry(self, signal_bar_start: int, target_start: int) -> None:
        """单信号入场裁决：睡到周期开始 → 锚点/障碍 → 重建回放 → 2s 实时裁决。"""
        target_end = target_start + BAR_MS_15M
        try:
            # 1) 睡到目标周期开始（重启时已过起点则直接进入重建）
            while self._running:
                wait_ms = target_start - clock_sync.now_ms()
                if wait_ms <= 0:
                    break
                await asyncio.sleep(min(5.0, wait_ms / 1000))
            if not self._running:
                return
            # 2) 锚点 O 回读（失败每 5s 重试，直到周期结束）
            o = 0.0
            while self._running and clock_sync.now_ms() < target_end:
                o = await self._collector.fetch_kline_open("15m", target_start)
                if o > 0:
                    break
                await asyncio.sleep(OPEN_RETRY_S)
            if o <= 0:
                await self._finalize_entry(signal_bar_start, "NO_DATA")
                return
            # 3) 障碍 = O ± 0.25×ATR（目标根位置 ATR：前 20 根已收盘 range% 均值 ×O）
            closed = await self._collector.fetch_recent_klines("15m", ATR_WINDOW)
            atr_t = atr_for_target(closed, o) if closed else None
            if not atr_t:
                await self._finalize_entry(signal_bar_start, "NO_DATA")
                return
            up, dn = o + ENTRY_X * atr_t, o - ENTRY_X * atr_t
            await self._patch_levels(signal_bar_start, o, up, dn)
            # 4) 重启重建：已过时段（或整窗）用已收盘 1m 棒回放
            state, touch_ts, touch_price = await self._replay_1m(target_start, target_end, up, dn)
            if state is not None:
                await self._finalize_entry(signal_bar_start, state, touch_ts, touch_price)
                return
            # 5) 实时裁决：~2s 轮询 bookTicker mid（喂价超龄 → FEED_GAP 保守弃）
            while self._running:
                now = clock_sync.now_ms()
                if now >= target_end:
                    await self._finalize_entry(signal_bar_start, "NOT_TOUCHED")
                    return
                last_upd = self._collector.store.last_ws_spot_update or 0.0
                if time.time() - last_upd > FEED_STALE_S:
                    logger.warning("HM 影子：喂价超龄 >{}s，保守放弃 | 信号根 {}",
                                   FEED_STALE_S, signal_bar_start)
                    await self._finalize_entry(signal_bar_start, "FEED_GAP")
                    return
                mid = self._collector.store.mid_price
                if mid and mid > 0:
                    decision = entry_decision(mid, up, dn, (now - target_start) / 1000)
                    if decision != "WAIT":
                        is_touch = decision in ("TOUCHED", "ABANDON_LATE")
                        await self._finalize_entry(
                            signal_bar_start, decision,
                            touch_ts=now if is_touch else None,
                            touch_price=float(mid) if is_touch else None,
                        )
                        return
                await asyncio.sleep(MONITOR_POLL_S)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("HM 影子：入场监控任务异常 | 信号根 {} | {}", signal_bar_start, exc)

    async def _replay_1m(self, target_start: int, target_end: int,
                         up: float, dn: float) -> tuple[str | None, int | None, float | None]:
        """回放目标周期已过时段的已收盘 1m 棒（重启重建）。

        返回 (state, touch_ts, touch_price)：state=None 表示未决出（继续实时裁决）；
        RESTART_GAP = 1m 缺棒或棒内双触（先后顺序不可知，保守放弃）。
        触及时刻/价格按 1m 粒度近似：touch_ts=所在棒 open_time、touch_price=上障碍价；
        重建路径不回填 entry_down_quote（不回灌）。
        """
        now = clock_sync.now_ms()
        if now <= target_start:
            return None, None, None
        closed_ms = target_end if now >= target_end else now
        expected = int((closed_ms - target_start) // BAR_MS_1M)
        if expected <= 0:
            return None, None, None
        bars = await self._collector.fetch_recent_klines("1m", REPLAY_1M_LIMIT)
        seq = sorted(
            (b for b in bars
             if target_start <= int(b["open_time"]) < target_end
             and int(b["open_time"]) + BAR_MS_1M <= now),
            key=lambda b: int(b["open_time"]),
        )
        # 完整性：自周期起点连续无缺棒，否则棒间/棒内顺序不可知 → 保守放弃
        if (len(seq) < expected or int(seq[0]["open_time"]) != target_start
                or any(int(seq[i]["open_time"]) - int(seq[i - 1]["open_time"]) != BAR_MS_1M
                       for i in range(1, len(seq)))):
            return "RESTART_GAP", None, None
        for b in seq[:expected]:
            hit_up = float(b["high"]) >= up
            hit_dn = float(b["low"]) <= dn
            if hit_up and hit_dn:
                return "RESTART_GAP", None, None  # 棒内双触：先后顺序不可知
            if hit_dn:
                return "ABANDON_LOWER", None, None
            if hit_up:
                # 1m 棒与 600s 死线天然对齐：第 10 根起（≥600s）为迟到
                late = (int(b["open_time"]) - target_start) // BAR_MS_1M >= TOUCH_DEADLINE_S // 60
                return ("ABANDON_LATE" if late else "TOUCHED"), int(b["open_time"]), up
        if now >= target_end:
            return "NOT_TOUCHED", None, None
        return None, None, None

    async def _patch_levels(self, signal_bar_start: int, o: float, up: float, dn: float) -> None:
        async with async_session_factory() as session:
            await session.execute(sa_update(PatternShadowSignal).where(
                PatternShadowSignal.version.in_(ALL_VERSIONS),
                PatternShadowSignal.signal_bar_start == signal_bar_start,
                PatternShadowSignal.status == "PENDING",
            ).values(target_open=o, up_level=up, dn_level=dn))
            await session.commit()

    async def _finalize_entry(self, signal_bar_start: int, state: str,
                              touch_ts: int | None = None,
                              touch_price: float | None = None) -> None:
        """入场状态落库；TOUCHED 时顺带快照当时 15m 市场 DOWN 真实报价（护栏定标数据）。"""
        quote = None
        if state == "TOUCHED":
            q = dict(self._pm_15m)
            upd = q.get("updated_ts")
            if (q.get("start_date") == signal_bar_start + BAR_MS_15M
                    and q.get("down_price") is not None and upd is not None
                    and clock_sync.now_ms() - int(upd) <= QUOTE_MAX_AGE_MS):
                quote = float(q["down_price"])
        async with async_session_factory() as session:
            await session.execute(sa_update(PatternShadowSignal).where(
                PatternShadowSignal.version.in_(ALL_VERSIONS),
                PatternShadowSignal.signal_bar_start == signal_bar_start,
                PatternShadowSignal.status == "PENDING",
            ).values(entry_state=state, touch_ts=touch_ts, touch_price=touch_price,
                     entry_down_quote=quote))
            await session.commit()
        logger.info("HM 影子入场裁决 | 信号根 {} | {} | touch_ts={} quote={}",
                    signal_bar_start, state, touch_ts, quote)

    # ------------------------------------------------------------------
    # 结算（仅 TOUCHED：目标根收阴即赢）
    # ------------------------------------------------------------------

    async def _settle_pending(self, closed_15m: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed_15m}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(PatternShadowSignal).where(
                    PatternShadowSignal.status == "PENDING",
                    PatternShadowSignal.target_bar_start.in_(starts),
                )
            )).scalars().all()
            if not pendings:
                return
            for sig in pendings:
                bar = by_start[int(sig.target_bar_start)]
                o, c = float(bar["open"]), float(bar["close"])
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                if sig.entry_state != "TOUCHED":
                    # 入场阶段未触及/放弃：无结算，win 保持 NULL
                    sig.status = "EXPIRED"
                elif c < o:
                    sig.settle_outcome, sig.win, sig.status = "DOWN", True, "SETTLED"
                elif c > o:
                    sig.settle_outcome, sig.win, sig.status = "UP", False, "SETTLED"
                else:
                    sig.settle_outcome, sig.win, sig.status = "NOISE", None, "EXPIRED"
                self._settle_count += 1
                logger.info(
                    "HM 影子结算 | 信号根 {} | 入场={} | 目标根 {} → {} | win={}",
                    int(sig.signal_bar_start), sig.entry_state, int(sig.target_bar_start),
                    sig.settle_outcome or sig.entry_state,
                    sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        """目标根起点后 4h 仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(PatternShadowSignal).where(
                    PatternShadowSignal.status == "PENDING",
                    PatternShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("HM 影子：PENDING 超时转 EXPIRED | 目标根 {}",
                               int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补 / 重启恢复（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            logger.warning("HM 影子：冷启动回补数据不足（{} 根），跳过", len(closed_15m))
            return
        await self._evaluate_new_bars(closed_15m)
        self._last_evaluated_bar = int(closed_15m[-1]["open_time"])
        await self._resume_pending()
        # 顺带结算停机期间已到期信号
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()

    async def _resume_pending(self) -> None:
        """重启跨窗：WAITING 行重派监控任务（窗口未结束续跑；已结束由监控任务 1m 重建）。"""
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(
                    PatternShadowSignal.signal_bar_start,
                    PatternShadowSignal.target_bar_start,
                ).where(
                    PatternShadowSignal.status == "PENDING",
                    PatternShadowSignal.entry_state == "WAITING",
                )
            )).all()
        for r in rows:
            self._spawn_watcher(int(r.signal_bar_start), int(r.target_bar_start))
        if rows:
            logger.info("HM 影子：重启恢复重派入场监控 {} 个", len(rows))

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "version": VERSION,
            "last_evaluated_bar": self._last_evaluated_bar,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "active_watchers": len(self._watchers),
        }
