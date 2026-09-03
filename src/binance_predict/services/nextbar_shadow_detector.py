"""下一根 K 线方向影子检测器（nextbar 族）：H=1 方向研究冻结条件的实时重放。

信号定义（阈值全精度冻结，源自 output/nextbar_h1_720d/converge_registry.csv 与
阶段E误定价扫描，禁止手抄渲染值）：
    nb_zschamp_15m_v1（15m 冠军，converge_registry L3 ROBUST，holdout n=368
    P(up_1)=61.96%，月一致性 0.958 / walk-forward 1.00）：
        zscore_10 <= -1.65079327 AND zscore_5 <= -1.53756693 AND ret_3 <= -0.0039526084
        → 押次根 15m 收阳 UP（720d 次根收阳 58.92%，2006 触发；深超卖+急跌+卖盘衰竭）
    nb_smaslope_5m_v1（5m 误定价候选，阶段E：市场报价钝在 q̄0.500 而 Jul-Aug 真实
    P(UP)=0.534，B⁺ 逐笔 EV t=1.73 未达 t>3）：
        sma_slope_atr_5 >= 1.6605556162359245（7月发现段 q0.9 全精度复现，n_disc=8366）
        → 押次根 5m 收阳 UP（720d 次根收阳仅 47.43%：edge 依赖 Jul-Aug regime，长样本
        反指，影子期前向验证「动量误定价」是否持续——record-only 科学仪器，非背书）

口径保真（影子阶段的生命线）：
    特征计算直接调用离线流水线 discovery.features.build_feature_matrix（单一
    事实源，k5=None——本族两条条件均不含 path3 周期路径特征，故无需 5m 子根），
    条件判定用 discovery.hypotheses.parse_condition/condition_mask 执行冻结条件
    原文——实时与 720d 回测逐位一致，杜绝第二套口径。硬闸门测试复现全样本触发
    计数（5m 19597 / 15m 2006）与短窗末根特征对齐全量矩阵同位值。

与 KREV / 反转影子共表 kline_shadow_signals（version 区分），但结算/超时各自只认
自己 timeframe 的 version，且用本 timeframe 的 K 线结算——15m bar 起点同时也是某
5m bar 起点，若跨 timeframe 混用会用 5m 次根 close 错结 15m 信号，故按 tf 严格隔离。

影子纪律：只记录不下注、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，本表不被任何下单
代码引用（物理隔离）。攒 2~3 周真实样本复核后人工 promote 才可上线。

数据流：
    1. 每 60s 轮询；fetch_recent_klines 按币安服务器时间只返回已收盘 K，天然规避
       边界抢跑（无需本地时钟判断）；
    2. 每个 timeframe（5m/15m）独立：出现新的已收盘根 → 拉 40 根 → 构建特征矩阵
       （微秒级）→ 条件求值 → 命中幂等落 PENDING（唯一约束 (version, signal_bar_start)），
       并快照目标窗开盘后首次轮询的 UP/DOWN 真实报价（entry_up/down_price，供聚合层
       现算真实 EV；窗口未对齐/冷启动回补 → NULL 不计）；
    3. 该 timeframe 已收盘根的 open_time 命中某 PENDING 的 target_bar_start →
       回读该根 OHLC 按 direction 结算 → SETTLED/EXPIRED；
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

# ---- 冻结口径（converge_registry.csv L69 逐字 + 阶段E q0.9 全精度复现，勿手抄渲染值）----
NEXTBAR_SHADOW_SPECS: list[dict] = [
    {
        "version": "nb_zschamp_15m_v1",
        "discovery_id": "nb_15m_zs",       # 无冻结注册表 discovery_id → 占位标识，审计可辨
        "timeframe": "15m",
        "direction": "UP",                 # 深超卖反转做多：次根 15m 收阳即赢
        "condition": (
            "zscore_10 <= -1.65079327 AND zscore_5 <= -1.53756693 "
            "AND ret_3 <= -0.0039526084"
        ),
        "snapshot_features": ("zscore_10", "zscore_5", "ret_3"),
    },
    {
        "version": "nb_smaslope_5m_v1",
        "discovery_id": "nb_5m_sma",
        "timeframe": "5m",
        "direction": "UP",                 # 动量误定价候选：押次根 5m 收阳（regime 依赖）
        "condition": "sma_slope_atr_5 >= 1.6605556162359245",
        "snapshot_features": ("sma_slope_atr_5",),
    },
]
NEXTBAR_VERSIONS = [s["version"] for s in NEXTBAR_SHADOW_SPECS]
# 每个 timeframe 只认自己的 version 结算/超时（跨 tf 隔离，防 5m 次根错结 15m 信号）
VERSIONS_BY_TF: dict[str, list[str]] = {
    tf: [s["version"] for s in NEXTBAR_SHADOW_SPECS if s["timeframe"] == tf]
    for tf in ("5m", "15m")
}

BAR_MS = {"5m": 300_000, "15m": 900_000}
TIMEFRAMES = ("5m", "15m")
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
# ATR20（前 20 根 range% 均值）+ sma_slope_atr_5（sma5 及其 5 根前值）最长需 ~25 根；
# zscore_10 需 10 根。40 根足够 warmup + 12 根回补余量（末根特征与全量矩阵逐位一致）。
WARMUP_BARS = 40
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数
# 目标根起点后仍未结算的超时兜底：5m 次根应在 ~10min 内结算，给 1h；15m 与 KREV 同 4h。
PENDING_EXPIRE_MS = {"5m": 3_600_000, "15m": 4 * 3_600_000}


def _to_klines(rows: list[dict], bar_ms: int) -> Klines:
    """data_collector 的 K 线 dict 列表 → discovery.data.Klines（升序、已收盘）。

    与 KREV / 反转影子检测器同源实现（各检测器自包含，口径一致）。
    """
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

    返回命中列表：[{"spec", "idx", "bar_offset"}, ...]；缺特征/NaN 保守不触发
    （condition_mask 对 NaN 比较恒 False，天然排除特征未预热的根）。
    """
    hits: list[dict] = []
    first_col = next(iter(fm.cols.values()), None) if fm.cols else None
    n = len(first_col) if first_col is not None else 0
    for spec in specs:
        try:
            mask = condition_mask(fm, spec["parts"])
        except KeyError as exc:
            logger.warning("nextbar 影子：条件特征缺失保守跳过 | {} | {}", spec["version"], exc)
            continue
        tail = mask[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off, "bar_offset": off})
    return hits


class NextbarShadowDetector:
    """nextbar 影子信号检测器：轮询 5m/15m 收盘 → 条件求值/结算，全程只落表不下注。"""

    def __init__(self, collector, pm_15m_latest: dict, pm_5m_info: dict) -> None:
        self._collector = collector
        # 目标窗入场报价源（只读共享缓存）：15m→_pm_15m_latest / 5m→_pm_market_info，
        # 信号落库时按 timeframe 取对应缓存快照 UP/DOWN 真实报价（窗口对齐守卫）
        self._pm_by_tf: dict[str, dict] = {"5m": pm_5m_info, "15m": pm_15m_latest}
        self._running = False
        self._task: asyncio.Task | None = None
        # 每个 timeframe 独立记录已评估的最大 open_time（5m/15m 收盘节奏不同）
        self._last_evaluated_bar: dict[str, int | None] = {"5m": None, "15m": None}
        self._trigger_count = 0
        self._settle_count = 0
        # 条件预解析（冻结原文 → 原子片段），启动时一次性暴露格式错误
        self._specs = [{**s, "parts": parse_condition(s["condition"])} for s in NEXTBAR_SHADOW_SPECS]

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
            logger.warning("nextbar 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="nextbar_shadow_detector")
        logger.info(
            "nextbar K线影子检测器启动 | {} | 冻结条件原文求值（15m 冠军 720d 次根收阳 "
            "58.92% / 5m sma_slope 47.43% regime 依赖）（影子模式：只记录不下注）",
            "/".join(NEXTBAR_VERSIONS),
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
        logger.info("nextbar 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("nextbar 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        """逐 timeframe 独立轮询：5m 与 15m 各自评估/结算/超时（跨 tf 严格隔离）。"""
        for tf in TIMEFRAMES:
            specs_tf = [s for s in self._specs if s["timeframe"] == tf]
            if not specs_tf:
                continue
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            # 拉取足够才评估/结算（不足则本轮跳过、不推进 _last_evaluated_bar，下轮重试）
            if len(closed) >= WARMUP_BARS:
                last_start = int(closed[-1]["open_time"])
                if self._last_evaluated_bar[tf] is None or last_start > self._last_evaluated_bar[tf]:
                    await self._evaluate_new_bars(tf, closed, specs_tf)
                    self._last_evaluated_bar[tf] = last_start
                await self._settle_pending(tf, closed)
            # 超时兜底只依赖 DB + 墙钟（不依赖 closed），即使本轮拉取失败也执行，
            # 避免某 tf 长时间拉取失败时 stale PENDING 的 EXPIRED 清理被无限延迟
            await self._expire_stale_pending(tf)

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, tf: str, closed: list[dict], specs_tf: list[dict]) -> None:
        """评估该 tf _last_evaluated_bar 之后的新收盘根（含冷启动回补的末 12 根）。"""
        bar_ms = BAR_MS[tf]
        kl = _to_klines(closed, bar_ms)
        # k5=None：本族条件不含 path3 周期路径特征，无需 5m 子根（口径与冻结一致）
        fm = build_feature_matrix(kl, bar_ms)
        if self._last_evaluated_bar[tf] is None:
            n_tail = BACKSCAN_BARS  # 冷启动：回补最近 12 根
        else:
            starts = [int(r["open_time"]) for r in closed]
            try:
                first_new = next(
                    i for i, s in enumerate(starts) if s > self._last_evaluated_bar[tf]
                )
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)  # 长停机不追全史，最多回补 12 根
        hits = evaluate_conditions(fm, specs_tf, n_tail)
        if not hits:
            return
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed[hit["idx"]]
                last_bar = bar
                if await self._record_signal(session, hit["spec"], bar, fm, hit["idx"]):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("nextbar 影子触发 +{} | {} | 信号根 {}", added, tf, int(last_bar["open_time"]))

    async def _record_signal(self, session, spec: dict, bar: dict, fm, idx: int) -> bool:
        """幂等落 PENDING：唯一约束 (version, signal_bar_start) + 先查后插。"""
        start_ms = int(bar["open_time"])
        bar_ms = BAR_MS[spec["timeframe"]]
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        snapshot: dict = {}
        for feat in spec["snapshot_features"]:
            col = fm.cols.get(feat)
            if col is not None and idx < len(col):
                fv = float(col[idx])
                snapshot[feat] = None if np.isnan(fv) else round(fv, 6)
        target_bar_start = start_ms + bar_ms
        # 目标窗入场报价快照（窗口对齐+近开盘守卫；缺失/回补 → None，该笔 EV 不计）
        up_q, down_q, q_ts = snapshot_entry_quote(
            self._pm_by_tf.get(spec["timeframe"]), target_bar_start
        )
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition"],
            timeframe=spec["timeframe"],
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

    # ------------------------------------------------------------------
    # 结算（按 direction 判 win：UP→次根收阳赢；仅认本 tf 的 version + 本 tf 的 K 线）
    # ------------------------------------------------------------------

    async def _settle_pending(self, tf: str, closed: list[dict]) -> None:
        """次根收盘结算：用本 tf 的 K 线按 direction 判 win，仅认本 tf 的 version。

        跨 tf 隔离是生命线：15m bar 起点也是某 5m bar 起点，若用 5m 次根 close 结
        15m 信号会错判（5m 次根 ≠ 15m 次根），故 version 与 K 线源都按 tf 过滤。
        """
        versions_tf = VERSIONS_BY_TF[tf]
        if not versions_tf:
            return
        by_start = {int(r["open_time"]): r for r in closed}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(versions_tf),
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
                    # direction=UP → 次根收阳赢；direction=DOWN → 次根收阴赢
                    sig.win = (up and sig.direction == "UP") or (not up and sig.direction == "DOWN")
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "nextbar 影子结算 | {} | {} | 信号根 {} | 次根 {} → {} | win={}",
                    sig.version, tf, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self, tf: str) -> None:
        """目标根起点后超时仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        versions_tf = VERSIONS_BY_TF[tf]
        if not versions_tf:
            return
        cutoff = int(time.time() * 1000) - BAR_MS[tf] - PENDING_EXPIRE_MS[tf]
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(versions_tf),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("nextbar 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        for tf in TIMEFRAMES:
            specs_tf = [s for s in self._specs if s["timeframe"] == tf]
            if not specs_tf:
                continue
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                logger.warning("nextbar 影子：{} 冷启动回补数据不足（{} 根），跳过", tf, len(closed))
                continue
            await self._evaluate_new_bars(tf, closed, specs_tf)
            self._last_evaluated_bar[tf] = int(closed[-1]["open_time"])
            # 顺带结算停机期间已到期信号
            await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_evaluated_bar": dict(self._last_evaluated_bar),
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "versions": list(NEXTBAR_VERSIONS),
        }
