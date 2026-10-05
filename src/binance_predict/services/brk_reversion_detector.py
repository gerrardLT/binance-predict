"""BRK 假突破回归影子检测器（2026-10-05 研究 run 20261005T051500Z-sentiment-kline-reversal）。

信号定义（与预注册研究 src/kline_features.py 的 BREAK_UP 原子逐位一致，禁止手抄第二套口径）：
    brkrv_brk8h_15m_v1（15m 收盘突破前 32 根（8h）高 → 押次根收阴 DOWN）：
        close > prior32 高（roll_max(high,32).shift(1)，不含当前根）
        训练 55.2%(n=1570) → 验证 58.8%(n=1198，Wilson95%下界 56.0%) → 盲测
        55.7%(n=79)；720D 56.8%(n=2768，下界 54.9%)；90D/30D 59.5%/55.6% 全正——
        全部六口径一致的「假突破回归」结构（本轮研究最强形态家族）。
    brkrv_brk8h_5m_v1（5m 收盘突破前 96 根（8h）高 → 押次根收阴 DOWN）：
        close > prior96 高
        K 线 rev2 三段：训练 56.4% → 验证 58.1% → 盲测 57.6%（n=144）；
        盲测市场结算口径（NOISE 计输，hard）56.9%（n=144）——5m 族唯一经济口径
        仍为正的强档。⚠ 横盘 regime（如 2026-07 市场 NOISE 率 50%）会摧毁 5m
        经济 EV，开启实盘前须确认近期 NOISE 率（影子 entry_down_price 可现算）。

口径保真（影子阶段的生命线）：
    特征公式逐位移植研究 kline_features.py（trailing 滚动窗、prior 高点显式
    shift(1)、min_periods=window 全窗才有效），NaN 一律保守不触发；测试
    tests/test_brk_reversion_detector.py 用独立 numpy 参照逐位对照尾部掩码。
    close > 前窗全部高点 ⇒ close > 前收 ⇒ 阳线，故研究口径不含显式阳线条件
    （与 CRV 的 up ∧ break 的差异：CRV 是研究冻结原文，本族研究原文即无 up）。

采集与执行解耦：影子信号照常落表 kline_shadow_signals（version+timeframe 双隔离，
与 KREV/CRV/nextbar 共表）；两个版本注册同名实盘通道（默认 OFF），仅正常轮询中
目标根开盘后 90 秒内的新鲜命中经钩子交给 MultiLiveTrader
（on_kline_reversal_signal → kline_reversal 族）；冷启动回补不追单。
结算口径与表内既有族一致：押 DOWN → 次根收阴赢；平盘 NOISE/EXPIRED（K 线口径，
市场真实 NOISE 阈值更宽，5m 经济 EV 由 entry_down_price 前向现算校准）。
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

# ---- 冻结口径（研究 run 20261005T051500Z discovery/promoted_*.json，勿动）----
BRK_SHADOW_SPECS: list[dict] = [
    {
        "version": "brkrv_brk8h_15m_v1",
        "tf": "15m",
        "discovery_id": "brkrv20261005_15m",
        "kind": "brk8h",
        "direction": "DOWN",           # 假突破阳线 → 次根收阴（均值回归）
        "condition_text": "close > prior32高(8h) → 次根收阴DOWN",
    },
    {
        "version": "brkrv_brk8h_5m_v1",
        "tf": "5m",
        "discovery_id": "brkrv20261005_5m",
        "kind": "brk8h",
        "direction": "DOWN",
        "condition_text": "close > prior96高(8h) → 次根收阴DOWN",
    },
]
BRK_VERSIONS = [s["version"] for s in BRK_SHADOW_SPECS]
BRK_VERSIONS_BY_TF: dict[str, list[str]] = {
    tf: [s["version"] for s in BRK_SHADOW_SPECS if s["tf"] == tf] for tf in ("15m", "5m")
}

BAR_MS = {"15m": 900_000, "5m": 300_000}
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
# prior 高点最大窗 96（5m）+ 12 根回补余量 + 2 根安全边 → 110（两周期统一）
WARMUP_BARS = 110
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED
BRK_LIVE_MAX_LAG_MS = 90_000    # 目标根开盘后 90s 内的新鲜命中才派实盘
BRK_WINDOWS = {"15m": 32, "5m": 96}   # 突破参照窗（根数；两周期均 8 小时）


# ---------------------------------------------------------------- 滚动特征（研究口径移植）
def roll_max(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 最大（含当前根；前 w-1 根 NaN；窗内含 NaN → NaN，对齐 pandas 默认）。"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        out[i] = np.nan if np.isnan(win).any() else float(win.max())
    return out


def _shift1(x: np.ndarray) -> np.ndarray:
    """shift(1)：x[i-1]，首根 NaN（对齐 pandas shift 语义）。"""
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = x[:-1]
    return out


def _gt(a, b) -> np.ndarray:
    """NaN 安全比较：任一侧 NaN → False（保守不触发）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    out = a > b
    out[np.isnan(a) | np.isnan(b)] = False
    return out


def compute_brk_features(kl: Klines, window: int) -> dict:
    """BRK 判定所需特征：prior 高点 = roll_max(high, window).shift(1)（不含当前根）。"""
    return {
        "close": kl.c,
        "prior_hi": _shift1(roll_max(kl.h, window)),
    }


def spec_mask(spec: dict, feat: dict) -> np.ndarray:
    """按 spec.kind 构建逐根布尔命中掩码（冻结条件，NaN 保守不触发）。

    研究原文即无显式阳线条件：close > 前窗全部高点 ⇒ close > 前收盘（隐含阳线）。
    """
    if spec["kind"] != "brk8h":
        raise ValueError(f"未知 BRK spec.kind: {spec['kind']}")
    return _gt(feat["close"], feat["prior_hi"])


def evaluate_brk(feat: dict, specs: list[dict], n_tail: int) -> list[dict]:
    """对末 n_tail 根逐条求值 BRK 条件（纯函数，供实时/回补/测试共用）。"""
    hits: list[dict] = []
    n = len(feat["close"])
    for spec in specs:
        mask = spec_mask(spec, feat)
        tail = mask[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off, "bar_offset": off})
    return hits


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


class BrkReversionDetector:
    """BRK 假突破回归影子检测器：轮询 5m/15m 收盘 → 冻结条件求值/结算，影子落表；
    两个版本同名实盘通道默认 OFF，新鲜命中经钩子派单。"""

    def __init__(self, collector, pm_15m_latest: dict, pm_5m_info: dict) -> None:
        self._collector = collector
        # 目标窗入场报价源（只读共享缓存）：15m 用 _pm_15m_latest、5m 用 _pm_market_info
        self._pm_by_tf: dict[str, dict] = {"15m": pm_15m_latest, "5m": pm_5m_info}
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: dict[str, int | None] = {"15m": None, "5m": None}
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

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
            logger.warning("BRK 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="brk_reversion_detector")
        logger.info(
            "BRK 假突破回归影子检测器启动 | 15m {} | 5m {} | 冻结条件实时求值"
            "（影子模式：只记录不下注；实盘通道默认 OFF）",
            "/".join(BRK_VERSIONS_BY_TF["15m"]), "/".join(BRK_VERSIONS_BY_TF["5m"]),
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
        logger.info("BRK 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("BRK 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        for tf in ("15m", "5m"):
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                continue  # 拉取失败/不足，下轮重试
            last_start = int(closed[-1]["open_time"])
            if (self._last_evaluated_bar[tf] is None
                    or last_start > self._last_evaluated_bar[tf]):
                await self._evaluate_new_bars(tf, closed)
                self._last_evaluated_bar[tf] = last_start
            await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, tf: str, closed: list[dict]) -> None:
        """评估 _last_evaluated_bar 之后的新收盘根（含冷启动回补的末 12 根）。"""
        bar_ms = BAR_MS[tf]
        kl = _to_klines(closed, bar_ms)
        feat = compute_brk_features(kl, BRK_WINDOWS[tf])
        specs = [s for s in BRK_SHADOW_SPECS if s["tf"] == tf]
        collect_live = self._last_evaluated_bar[tf] is not None
        if self._last_evaluated_bar[tf] is None:
            n_tail = BACKSCAN_BARS  # 冷启动：回补最近 12 根，不追真钱单
        else:
            starts = [int(r["open_time"]) for r in closed]
            try:
                first_new = next(
                    i for i, s in enumerate(starts) if s > self._last_evaluated_bar[tf])
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)  # 长停机不追全史，最多回补 12 根
        hits = evaluate_brk(feat, specs, n_tail)
        if not hits:
            return
        live_payloads: list[dict] = []
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed[hit["idx"]]
                last_bar = bar
                payloads = live_payloads if collect_live else None
                if await self._record_signal(tf, session, hit["spec"], bar, feat, hit["idx"], payloads):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("BRK 影子触发 +{} | {} | 信号根 {}", added, tf, int(last_bar["open_time"]))
        self._dispatch_live(live_payloads)

    def _dispatch_live(self, payloads: list[dict]) -> None:
        hook = self._on_live_fire
        if hook is None:
            return
        for payload in payloads:
            try:
                hook(payload)
            except Exception as exc:
                logger.warning("BRK：实盘开火分派异常（不影响影子采集）| {}", exc)

    async def _record_signal(self, tf: str, session, spec: dict, bar: dict, feat: dict,
                             idx: int, live_payloads: list[dict] | None = None) -> bool:
        """幂等落 PENDING，并为正常轮询中的新鲜命中收集实盘 payload。"""
        bar_ms = BAR_MS[tf]
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        target_bar_start = start_ms + bar_ms
        if live_payloads is not None and (
                0 <= int(time.time() * 1000) - target_bar_start <= BRK_LIVE_MAX_LAG_MS):
            live_payloads.append({
                "version": spec["version"],
                "market_start": target_bar_start,
                "market_end": target_bar_start + bar_ms,
                "direction": spec["direction"],
                "signal_bar_start": start_ms,
            })
        if not shadow_gate.is_enabled(spec["version"]):
            return False  # 手动下线只停影子采集；实盘由 trader enabled 独立控制
        # 审计快照：条件涉及的实际特征值（供实时值与研究口径对照）
        snapshot: dict = {}
        prior_hi = float(feat["prior_hi"][idx])
        close = float(feat["close"][idx])
        if not np.isnan(prior_hi):
            snapshot["prior_hi"] = round(prior_hi, 6)
            snapshot["dist_prior_hi"] = round(close - prior_hi, 6)  # 突破幅度
        # 目标窗入场报价快照（窗口对齐+近开盘守卫；缺失/回补 → None，该笔 EV 不计）
        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_by_tf[tf], target_bar_start)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe=tf,
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
    # 结算（按 direction 判 win：DOWN→次根收阴赢；只认本检测器 version）
    # ------------------------------------------------------------------

    async def _settle_pending(self, tf: str, closed: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(BRK_VERSIONS_BY_TF[tf]),
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
                    "BRK 影子结算 | {} | 信号根 {} | 次根 {} → {} | win={}",
                    sig.version, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self, tf: str) -> None:
        """目标根起点后 4h 仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        cutoff = int(time.time() * 1000) - BAR_MS[tf] - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(BRK_VERSIONS_BY_TF[tf]),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("BRK 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        for tf in ("15m", "5m"):
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                logger.warning("BRK 影子：冷启动回补 {} 数据不足（{} 根），跳过", tf, len(closed))
                continue
            await self._evaluate_new_bars(tf, closed)
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
            "versions": list(BRK_VERSIONS),
            "live_channels_registered": True,
        }
