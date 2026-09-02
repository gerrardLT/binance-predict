"""反转形态影子检测器（P1/P2 族）：15m 连跌/连涨后弱收盘的几何反转信号实时重放。

信号定义（几何口径与 .pytest_tmp/rev_common.compute_features 逐位一致，禁止手抄阈值）：
    rev_p1_v1（P1，15m 连跌4 + 弱阴收 + 量正常 → 押 UP）：
        streak <= -4 AND weak_close_dn AND vol_norm  → 次根 15m 收阳（close>open）即赢
        720d 回测 62.0% / oos 63.9%（纯 K 线中性价，EV=None）
    rev_p2_v1（P2，15m 连涨5 + 弱阳收贴最高 → 押 DOWN）：
        streak >= +5 AND weak_close_up  → 次根 15m 收阴（close<open）即赢
        720d 回测 62.4% / oos 61.3%（纯 K 线中性价，EV=None）

口径保真（影子阶段的生命线）：
    几何特征（streak / close_pos / weak_close_dn / weak_close_up / vol_ratio /
    vol_norm）直接移植研究库 rev_common.compute_features 的公式与 nan_to_num 口径，
    15m 的 w20=7（与 rev_common 同源窗口折算），实时与研究逐位一致，杜绝第二套口径。
    与 KREV 族（kline_shadow_detector）的区别：KREV 走 discovery 特征矩阵
    （build_feature_matrix），本族走 rev_common 几何口径，两套互不复用。

与 KREV 影子共表 kline_shadow_signals（version 区分），但结算/超时各自只认自己的
version（rev_p1_v1 / rev_p2_v1），杜绝跨 version 误结算——KREV 硬编码 UP 语义，
本族按 direction 判 win（UP→收阳赢 / DOWN→收阴赢），若不过滤会互相污染。

影子纪律：只记录不下注、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，本表不被任何下单
代码引用（物理隔离）。攒 2~3 周真实样本复核后人工 promote 才可上线。

数据流：
    1. 每 60s 轮询；fetch_recent_klines 按币安服务器时间只返回已收盘 K，
       天然规避边界抢跑（无需本地时钟判断）；
    2. 出现新的已收盘 15m 根 → 拉 40 根 15m → 几何特征求值 → 命中幂等落 PENDING
       （唯一约束 (version, signal_bar_start) 防重）；
    3. 已收盘 15m 根的 open_time 命中某 PENDING 的 target_bar_start →
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

# ---- 冻结口径（几何规则原文，源自 rev_common.compute_features，禁止手抄阈值）----
# discovery_id：KlineShadowSignal.discovery_id NOT NULL（KREV 用于关联冻结注册表），
# 本族无注册表来源 → 填占位标识（rev_p1 / rev_p2），审计可辨。
REVERSAL_SHADOW_SPECS: list[dict] = [
    {
        "version": "rev_p1_v1",
        "discovery_id": "rev_p1",
        "pattern": "P1",
        "direction": "UP",              # 连跌后反转做多：次根收阳即赢
        "streak_le": -4,
        "weak_close": "dn",             # weak_close_dn：阴线收在下沿（贴最低）
        "vol_norm": True,               # 量比 [1.0, 1.5)
        "condition_text": "streak <= -4 AND weak_close_dn AND vol_norm",
    },
    {
        "version": "rev_p2_v1",
        "discovery_id": "rev_p2",
        "pattern": "P2",
        "direction": "DOWN",            # 连涨后反转做空：次根收阴即赢
        "streak_ge": 5,
        "weak_close": "up",             # weak_close_up：阳线收在上沿（贴最高）
        "vol_norm": False,              # P2 不约束量能
        "condition_text": "streak >= +5 AND weak_close_up",
    },
]
REVERSAL_VERSIONS = [s["version"] for s in REVERSAL_SHADOW_SPECS]

BAR_MS_15M = 900_000
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
# P1/P2 仅依赖短窗口特征（vol_ratio 的 w20=7、streak 运行计数），不受研究 warm=300
# 约束（那是 pos288 等长窗口特征的统一预热）；40 根足够 warmup + 12 根回补余量。
WARMUP_BARS = 40
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数（3 小时）
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED（数据缺失兜底）
# 审计快照特征（几何实际值，供实时值与研究口径对照）
SNAPSHOT_FEATURES = ("streak", "close_pos", "vol_ratio")


# ---------------------------------------------------------------- 移植特征（审计锚点）
def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    """滚动均值（trailing，含当前根；前 w-1 根 NaN）。

    审计锚点：与 rev_common.roll_mean 逐位一致，禁止改写口径。
    """
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    cnt = np.concatenate([[0.0], np.cumsum(np.isfinite(x))])
    out[w - 1:] = (cs[w:] - cs[:-w]) / np.maximum(cnt[w:] - cnt[:-w], 1)
    return out


def compute_streak(c: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """收盘动量 streak：连续 c[i]>c[i-1] 为正计数，反之为负；断点重置。

    审计锚点：与 rev_common.compute_streak 逐位一致，禁止改写口径。
    """
    n = len(c)
    cd = np.zeros(n, dtype=np.int8)
    cd[1:] = np.sign(c[1:] - c[:-1]).astype(np.int8)
    streak = np.zeros(n, dtype=np.int32)
    cur = 0
    for i in range(n):
        if i > 0 and not cont[i]:
            cur = 0
        d = cd[i]
        if d > 0:
            cur = cur + 1 if cur > 0 else 1
        elif d < 0:
            cur = cur - 1 if cur < 0 else -1
        streak[i] = cur
    return streak


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


def compute_geometry(kl: Klines, bar_ms: int) -> dict:
    """P1/P2 判定所需最小几何特征集。

    审计锚点：与 .pytest_tmp/rev_common.compute_features 逐位一致——close_pos /
    weak_close_dn / weak_close_up / vol_ratio / vol_norm / streak 的公式、nan_to_num
    口径、w20 窗口折算全部照搬，禁止在此另立第二套阈值。
    """
    o, h, l, c, v = kl.o, kl.h, kl.l, kl.c, kl.v
    n = len(c)
    # w20（rev_common 同源公式）：bar_ms>300_000 → round(20*300_000/bar_ms)，下限 5；
    # 15m→7（≈1.75h）；bar_ms<=300_000（5m）→ 20。
    w20 = max(5, int(round(20 * (300_000 / bar_ms)))) if bar_ms > 300_000 else 20
    w20 = max(w20, 5)
    # streak：收盘环比连涨/连跌计数（rev_common.compute_streak）
    streak = compute_streak(c, kl.cont)
    # close_pos=(c-l)/(h-l)，range<=0 → nan；布尔判定用 nan_to_num（rev_common 口径）
    rng = h - l
    rng_safe = np.where(rng > 0, rng, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        close_pos = (c - l) / rng_safe
    cp = np.nan_to_num(close_pos)
    dir_ = np.sign(c - o)
    weak_close_dn = (dir_ < 0) & (cp <= 0.15)   # 阴线收在下沿（贴最低）
    weak_close_up = (dir_ > 0) & (cp >= 0.85)   # 阳线收在上沿（贴最高）
    # vol_ratio=v/roll_mean(v,w20)；vol_norm=[1.0,1.5)（rev_common 同源）
    vma = roll_mean(v, w20)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = np.where(vma > 0, v / vma, np.nan)
    vr = np.nan_to_num(vol_ratio)
    vol_norm = (vr >= 1.0) & (vr < 1.5)
    return {
        "n": n, "w20": w20, "streak": streak,
        "close_pos": close_pos, "vol_ratio": vol_ratio,
        "weak_close_dn": weak_close_dn, "weak_close_up": weak_close_up,
        "vol_norm": vol_norm,
    }


def _spec_mask(spec: dict, geo: dict) -> np.ndarray:
    """按 spec 构建逐根布尔命中掩码（几何口径，审计锚点见 compute_geometry）。"""
    streak = geo["streak"]
    if "streak_le" in spec:
        m = streak <= spec["streak_le"]
    elif "streak_ge" in spec:
        m = streak >= spec["streak_ge"]
    else:
        raise ValueError(f"spec 缺 streak 阈值: {spec['version']}")
    m = m & (geo["weak_close_dn"] if spec["weak_close"] == "dn" else geo["weak_close_up"])
    if spec.get("vol_norm"):
        m = m & geo["vol_norm"]
    return m


def evaluate_reversals(geo: dict, specs: list[dict], n_tail: int) -> list[dict]:
    """对末 n_tail 根逐条求值 P1/P2（纯函数，供实时/回补/测试共用）。

    返回命中列表：[{"spec", "idx", "bar_offset"}, ...]。
    """
    hits: list[dict] = []
    n = geo["n"]
    for spec in specs:
        mask = _spec_mask(spec, geo)
        tail = mask[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off, "bar_offset": off})
    return hits


class ReversalShadowDetector:
    """P1/P2 反转影子信号检测器：轮询 15m 收盘 → 几何求值/结算，全程只落表不下注。"""

    def __init__(self, collector) -> None:
        self._collector = collector
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: int | None = None  # 已评估过的最大 15m open_time
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
            logger.warning("反转影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="reversal_shadow_detector")
        logger.info(
            "反转 K线影子检测器启动 | {} | rev_common 几何口径求值（P1 连跌弱阴→UP / "
            "P2 连涨弱阳→DOWN）（影子模式：只记录不下注）",
            "/".join(REVERSAL_VERSIONS),
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
        logger.info("反转影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("反转影子：循环异常 | {} | {}", type(exc).__name__, exc)
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
        kl15 = _to_klines(closed_15m, BAR_MS_15M)
        geo = compute_geometry(kl15, BAR_MS_15M)
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
        hits = evaluate_reversals(geo, REVERSAL_SHADOW_SPECS, n_tail)
        if not hits:
            return
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed_15m[hit["idx"]]
                last_bar = bar
                if await self._record_signal(session, hit["spec"], bar, geo, hit["idx"]):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("反转影子触发 +{} | 信号根 {}", added, int(last_bar["open_time"]))

    async def _record_signal(self, session, spec: dict, bar: dict, geo: dict, idx: int) -> bool:
        """幂等落 PENDING：唯一约束 (version, signal_bar_start) + 先查后插。"""
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        snapshot: dict = {}
        for feat in SNAPSHOT_FEATURES:
            val = geo[feat][idx]
            if feat == "streak":
                snapshot[feat] = int(val)
            else:
                fv = float(val)
                snapshot[feat] = None if np.isnan(fv) else round(fv, 6)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe="15m",
            signal_bar_start=start_ms,
            signal_bar_end=start_ms + BAR_MS_15M,
            direction=spec["direction"],
            target_bar_start=start_ms + BAR_MS_15M,
            feature_snapshot=snapshot,
            status="PENDING",
        ))
        return True

    # ------------------------------------------------------------------
    # 结算（按 direction 判 win：UP→次根收阳赢 / DOWN→次根收阴赢）
    # ------------------------------------------------------------------

    async def _settle_pending(self, closed_15m: list[dict]) -> None:
        """次根收盘结算：按 direction 判 win，仅认本检测器的 version（rev_p1/p2）。

        与 KREV 影子检测器共表 kline_shadow_signals 但各管各的 version，杜绝 KREV
        硬编码 UP 语义误结算本族的 rev_p2_v1（DOWN）。
        """
        by_start = {int(r["open_time"]): r for r in closed_15m}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(REVERSAL_VERSIONS),
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
                    "反转影子结算 | {} | 信号根 {} | 次根 {} → {} | win={}",
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
                    KlineShadowSignal.version.in_(REVERSAL_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("反转影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        closed_15m = await self._collector.fetch_recent_klines("15m", WARMUP_BARS)
        if len(closed_15m) < WARMUP_BARS:
            logger.warning("反转影子：冷启动回补数据不足（{} 根），跳过", len(closed_15m))
            return
        await self._evaluate_new_bars(closed_15m)
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
            "versions": list(REVERSAL_VERSIONS),
        }
