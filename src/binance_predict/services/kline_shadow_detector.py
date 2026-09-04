"""K 线科学发现影子检测器（KREV 族）：720d 冻结注册表条件的实时重放。

信号定义（与 output/kline_discovery_15m_720d_v2 冻结注册表逐字对齐，禁止手抄阈值）：
    KREV-A（Top3，discovery_id=fd191c44fb5c36，holdout n=137 胜率 64.2%，
    费后 EV@0.50=+0.234，月一致性 0.957）：
        dist_prior_low_atr_5 <= -0.0935059731 AND efficiency_5 >= 0.861468132
        AND path3_all_down == True  → 反转做多（押次根收阳）
    KREV-B（Top4，discovery_id=5c5e4c78ab4c3f，holdout n=134 胜率 63.4%，
    费后 EV@0.50=+0.219）：
        range_pos_prior_5 <= -0.0467509994 AND efficiency_5 >= 0.861468132
        AND path3_all_down == True  → 反转做多

口径保真（影子阶段的生命线）：
    特征计算直接调用离线流水线 discovery.features.build_feature_matrix（单一
    事实源），条件判定用 discovery.hypotheses.parse_condition/condition_mask
    执行注册表条件原文——实时与回测逐位一致，杜绝第二套口径。
    path3_all_down 需该 15m 周期内 3 根 5m 子 K 齐全才可判定（_cycle_path
    full3 守卫）；子根缺失/特征 NaN/缺特征一律保守不触发。

影子纪律：只记录不下注、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，新表不被
任何下单代码引用（物理隔离）。结算口径与回测 reversal_1 一致：次根收阳
（close>open）即赢；平盘 → NOISE/EXPIRED。

数据流：
    1. 每 60s 轮询；fetch_recent_klines 按币安服务器时间只返回已收盘 K，
       天然规避边界抢跑（无需本地时钟判断）；
    2. 出现新的已收盘 15m 根 → 拉 30 根 15m + 4 根 5m → 全量构建特征矩阵
       （微秒级）→ 条件求值 → 命中幂等落 PENDING（唯一约束防重），并快照目标窗
       开盘后首次轮询的 UP/DOWN 真实报价（entry_up/down_price，供聚合层现算真实 EV）；
    3. 已收盘 15m 根的 open_time 命中某 PENDING 的 target_bar_start →
       回读该根 OHLC 结算 → SETTLED/EXPIRED；
    4. 冷启动回补最近 12 根（幂等）；超时 PENDING → EXPIRED。
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
from binance_predict.discovery.features import build_feature_matrix
from binance_predict.discovery.hypotheses import condition_mask, parse_condition
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（来自 output/kline_discovery_15m_720d_v2/discovery_registry.csv，勿动）----
SHADOW_CONDITIONS = [
    {
        "version": "krev_a_v1",
        "discovery_id": "fd191c44fb5c36",
        "condition": (
            "dist_prior_low_atr_5 <= -0.0935059731 AND "
            "efficiency_5 >= 0.861468132 AND path3_all_down == True"
        ),
    },
    {
        "version": "krev_b_v1",
        "discovery_id": "5c5e4c78ab4c3f",
        "condition": (
            "range_pos_prior_5 <= -0.0467509994 AND "
            "efficiency_5 >= 0.861468132 AND path3_all_down == True"
        ),
    },
]

BAR_MS_15M = 900_000
BAR_MS_5M = 300_000
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
WARMUP_BARS = 40                # 15m 历史根数（条件最长窗口=ATR20 → 20 前置 + 12 回补余量）
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数（3 小时）
K5_REALTIME_BARS = 4            # 实时评估只需当周期 3 根 5m 子根 + 1 余量
K5_BACKSCAN_BARS = 40           # 回补窗口 12×3=36 根 5m + 余量
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED（数据缺失兜底）
# 审计快照特征（两条条件涉及的全部特征 + path3）
SNAPSHOT_FEATURES = ("dist_prior_low_atr_5", "efficiency_5", "range_pos_prior_5", "path3_all_down")
# 本检测器负责的 version（结算/超时只认这些）——kline_shadow_signals 表另被反转影子
# 检测器（reversal_shadow_detector，rev_p1_v1/rev_p2_v1）共用，各自 version 隔离，
# 杜绝本检测器的硬编码 UP 语义误结算 rev_p2_v1（DOWN）。
KREV_VERSIONS = [s["version"] for s in SHADOW_CONDITIONS]


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


def evaluate_conditions(fm, specs: list[dict], n_tail: int) -> list[dict]:
    """对特征矩阵末 n_tail 根逐条求值影子条件（纯函数，供实时/回补/测试共用）。

    返回命中列表：[{"spec", "idx", "bar_offset"}, ...]；缺特征/NaN 保守不触发。
    """
    hits: list[dict] = []
    first_col = next(iter(fm.cols.values()), None) if fm.cols else None
    n = len(first_col) if first_col is not None else 0
    for spec in specs:
        try:
            mask = condition_mask(fm, spec["parts"])
        except KeyError as exc:
            logger.warning("KREV 影子：条件特征缺失保守跳过 | {} | {}", spec["version"], exc)
            continue
        tail = mask[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off, "bar_offset": off})
    return hits


class KlineShadowDetector:
    """KREV 影子信号检测器：轮询 15m 收盘 → 条件求值/结算，全程只落表不下注。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        # 目标窗入场报价源（只读共享缓存 _pm_15m_latest）：信号落库时快照目标 15m 窗
        # UP/DOWN 真实报价（窗口对齐守卫），供聚合层现算真实 EV
        self._pm_15m = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: int | None = None  # 已评估过的最大 15m open_time
        self._trigger_count = 0
        self._settle_count = 0
        # 条件预解析（注册表原文 → 原子片段），启动时一次性暴露格式错误
        self._specs = [{**s, "parts": parse_condition(s["condition"])} for s in SHADOW_CONDITIONS]

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
            logger.warning("KREV 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="kline_shadow_detector")
        versions = "/".join(s["version"] for s in self._specs)
        logger.info(
            "KREV K线影子检测器启动 | {} | 冻结注册表条件原文求值（720d v2）"
            " | 理论频率 ~0.7 次/天（A 族）（影子模式：只记录不下注）",
            versions,
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
        logger.info("KREV 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("KREV 影子：循环异常 | {} | {}", type(exc).__name__, exc)
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
            closed_5m = await self._collector.fetch_recent_klines("5m", K5_REALTIME_BARS)
            if not closed_5m:
                return  # 5m 暂缺（path3 无法判定），下轮重试，不标记已评估
            await self._evaluate_new_bars(closed_15m, closed_5m)
            self._last_evaluated_bar = last_start
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, closed_15m: list[dict], closed_5m: list[dict]) -> None:
        """评估 _last_evaluated_bar 之后的新收盘根（含冷启动回补的末 12 根）。"""
        kl15 = _to_klines(closed_15m, BAR_MS_15M)
        kl5 = _to_klines(closed_5m, BAR_MS_5M)
        fm = build_feature_matrix(kl15, BAR_MS_15M, k5=kl5)
        if self._last_evaluated_bar is None:
            n_tail = BACKSCAN_BARS  # 冷启动：回补最近 12 根
        else:
            starts = [int(r["open_time"]) for r in closed_15m]
            try:
                first_new = next(
                    i for i, s in enumerate(starts) if s > self._last_evaluated_bar
                )
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)  # 长停机不追全史，最多回补 12 根
        hits = evaluate_conditions(fm, self._specs, n_tail)
        if not hits:
            return
        async with async_session_factory() as session:
            added = 0
            for hit in hits:
                bar = closed_15m[hit["idx"]]
                if await self._record_signal(session, hit["spec"], bar, fm, hit["idx"]):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("KREV 影子触发 +{} | 信号根 {}", added, int(bar["open_time"]))

    async def _record_signal(self, session, spec: dict, bar: dict, fm, idx: int) -> bool:
        """幂等落 PENDING：唯一约束 (version, signal_bar_start) + 先查后插。"""
        if not shadow_gate.is_enabled(spec["version"]):
            return False  # 手动下线：停止采集新信号（历史数据保留）
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        snapshot = {}
        for feat in SNAPSHOT_FEATURES:
            col = fm.cols.get(feat)
            if col is not None and idx < len(col):
                val = col[idx]
                snapshot[feat] = bool(val) if isinstance(val, (bool, np.bool_)) else float(val)
        target_bar_start = start_ms + BAR_MS_15M
        # 目标窗入场报价快照（窗口对齐+近开盘守卫；缺失/回补 → None，该笔 EV 不计）
        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_15m, target_bar_start)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition"],
            timeframe="15m",
            signal_bar_start=start_ms,
            signal_bar_end=target_bar_start,
            direction="UP",
            target_bar_start=target_bar_start,
            feature_snapshot=snapshot,
            entry_up_price=up_q,
            entry_down_price=down_q,
            entry_quote_ts=q_ts,
            status="PENDING",
        ))
        return True

    # ------------------------------------------------------------------
    # 结算（回测 reversal_1 口径：次根收阳即赢）
    # ------------------------------------------------------------------

    async def _settle_pending(self, closed_15m: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed_15m}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(KREV_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(starts),
                )
            )).scalars().all()
            if not pendings:
                return
            for sig in pendings:
                bar = by_start[int(sig.target_bar_start)]
                o, c = float(bar["open"]), float(bar["close"])
                if c > o:
                    sig.settle_outcome, sig.win = "UP", True
                elif c < o:
                    sig.settle_outcome, sig.win = "DOWN", False
                else:
                    sig.settle_outcome, sig.win, sig.status = "NOISE", None, "EXPIRED"
                if sig.status != "EXPIRED":
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "KREV 影子结算 | {} | 信号根 {} | 次根 {} → {} | win={}",
                    sig.version, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        """目标根起点后 4h 仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(KREV_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("KREV 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            logger.warning("KREV 影子：冷启动回补数据不足（{} 根），跳过", len(closed_15m))
            return
        closed_5m = await self._collector.fetch_recent_klines("5m", K5_BACKSCAN_BARS)
        if not closed_5m:
            logger.warning("KREV 影子：冷启动回补 5m 数据缺失，跳过")
            return
        await self._evaluate_new_bars(closed_15m, closed_5m)
        self._last_evaluated_bar = int(closed_15m[-1]["open_time"])
        # 顺带结算停机期间已到期信号
        await self._settle_pending(closed_15m)
        await self._expire_stale_pending()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_evaluated_bar": self._last_evaluated_bar,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "versions": [s["version"] for s in self._specs],
        }
