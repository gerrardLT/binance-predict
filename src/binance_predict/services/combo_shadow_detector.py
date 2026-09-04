"""组合条件影子检测器（combo 族）：45 维条件大搜索存活组合的实时重放。

信号定义（冻结自 .pytest_tmp/grand_search_v2.py 搜索 + mr_freeze_bench.py 冻结口径
验证；720d 搜索池 → 1443 天样本外考试（2020-10~2024-09）→ 50 次置换检验三重过滤后
存活的 5 个组合，全部实时可实现口径）：
    combo_p1_v1  连阳3+ ∧ 周末(UTC) ∧ EMA20乖离≥+0.3%      → 押次根15m收阴 DOWN
                 720d 313/490=63.9% | oos 962/1588=60.6% | ≈1.5天/次
    combo_p2_v1  大实体(body_bp≥23.4633) ∧ 连阳3+ ∧ 周末   → DOWN
                 720d 137/206=66.5% | oos 579/957=60.5% | ≈3.5天/次
    combo_p3_v1  贴1天高(≤0.1%) ∧ 美盘(UTC h≥16) ∧ 7d涨≥4% → DOWN
                 720d 120/176=68.2% | oos 126/212=59.4% | ≈4.1天/次
    combo_p4_v1  收低位(pos≤0.25) ∧ 周末 ∧ RSI14≤25        → 押次根15m收阳 UP
                 720d 202/322=62.7% | oos 292/459=63.6% | ≈2.2天/次
    combo_p5_v1  近光脚(下影≤5%) ∧ 周末 ∧ RSI14≤25         → UP
                 720d 90/131=68.7% | oos 92/142=64.8% | ≈5.5天/次

口径保真（影子阶段的生命线）：
    研究特征不在 discovery 流水线内，故自包含实现——特征计算逐字照抄
    grand_search_v2.build_conditions 的相应片段（连阳 run doji 重置 / EMA20
    alpha=2/21 / RSI14 SMA / 含当前根 96 根滚动高 / 7d 收益 / 收盘位置 / 下影
    占比），np.nan_to_num(NaN→0) 的比较语义一并照抄。「大实体」为研究美元 P80
    （$207.2，全样本统计量实时不可重现）的冻结版 body_bp=|c-o|/open*1e4 ≥
    23.46330248119974（mr_freeze_bench.py 验证：冻结口径下 5 组合 oos 全线
    ≥59.4%）。硬闸门测试（tests/test_combo_shadow_detector.py）重放 720d 全
    样本触发计数 490/206/176/322/131 与 win 计数 313/137/120/202/90。

    轮询减重：每 60s 先 limit=2 探测是否有新收盘 15m 根，仅新根出现才拉
    WARMUP=700 根做特征求值（7d 收益需 673 根回看）；EMA20/连阳 run 从窗口头
    递推，700 根窗口下末根值与全量计算一致（EMA 尾差 <1e-16 相对量级、run 不
    可能超长连阳），硬闸门测试锁定短窗=全量同位值。

与 KREV / 反转 / nextbar 影子共表 kline_shadow_signals（version 隔离），次根
收盘按 direction 结算（DOWN→次根收阴赢 / UP→次根收阳赢），并快照目标窗开盘
后首次轮询的 UP/DOWN 真实报价（snapshot_entry_quote，供聚合层现算真实 EV）。

影子纪律：只记录不下注、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，本表不被任何
下单代码引用（物理隔离）。攒 2~3 周真实样本复核后人工 promote 才可上线。
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

# ---- 冻结口径（mr_freeze_bench.py：720d body_bp P80 全精度，勿手抄渲染值）----
BIG_BODY_BP = 23.46330248119974  # |c-o|/open*1e4 ≥ 此值 = 大实体（研究美元P80冻结版）

COMBO_SHADOW_SPECS: list[dict] = [
    {
        "version": "combo_p1_v1",
        "discovery_id": "gs2_p1",       # grand_search_v2 组合1（占位标识，审计可辨）
        "direction": "DOWN",            # 周末连阳过热 → 押次根 15m 收阴
        "condition_text": "连阳3+ ∧ 周末(UTC) ∧ EMA20乖离≥+0.3%",
        "features": ("up3", "weekend", "bias_pos"),
        "snapshot_features": ("run_len", "bias"),
    },
    {
        "version": "combo_p2_v1",
        "discovery_id": "gs2_p2",
        "direction": "DOWN",            # 周末大实体连阳 → 押次根 15m 收阴
        "condition_text": "大实体(body_bp≥23.4633) ∧ 连阳3+ ∧ 周末(UTC)",
        "features": ("big_body", "up3", "weekend"),
        "snapshot_features": ("run_len", "body_bp"),
    },
    {
        "version": "combo_p3_v1",
        "discovery_id": "gs2_p3",
        "direction": "DOWN",            # 高位+美盘+中期动量衰减 → 押次根 15m 收阴
        "condition_text": "贴1天高(≤0.1%) ∧ 美盘(UTC h≥16) ∧ 7d涨≥4%",
        "features": ("near1dh", "us_sess", "r7d_up"),
        "snapshot_features": ("dist_1dh_bp", "r7d_pct"),
    },
    {
        "version": "combo_p4_v1",
        "discovery_id": "gs2_p4",
        "direction": "UP",              # 周末超卖反弹 → 押次根 15m 收阳
        "condition_text": "收低位(pos≤0.25) ∧ 周末(UTC) ∧ RSI14≤25",
        "features": ("pos_low", "weekend", "rsi_lt25"),
        "snapshot_features": ("pos", "rsi"),
    },
    {
        "version": "combo_p5_v1",
        "discovery_id": "gs2_p5",
        "direction": "UP",              # P4 同簇严格版（近光脚）→ 押次根 15m 收阳
        "condition_text": "近光脚(下影≤5%) ∧ 周末(UTC) ∧ RSI14≤25",
        "features": ("no_lower", "weekend", "rsi_lt25"),
        "snapshot_features": ("lower", "rsi"),
    },
]
COMBO_VERSIONS = [s["version"] for s in COMBO_SHADOW_SPECS]

TIMEFRAME = "15m"
BAR_MS = 900_000
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
PROBE_BARS = 2                  # 探测拉取根数（仅判断是否有新收盘根）
# 7d 收益需 673 根回看；EMA20/run 从窗口头递推在 700 根下与全量收敛一致。
# 币安 klines 单次上限 1000，700 根一档（权重 5）仅在每 15m 新根时拉一次。
WARMUP_BARS = 700
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数
# 目标根起点后仍未结算的超时兜底（15m 次根应在 ~15min 内结算，与 KREV 同 4h）。
PENDING_EXPIRE_MS = 4 * 3_600_000


def compute_feature_tables(rows: list[dict]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """collector dict 行 → 组合特征（布尔掩码 + 数值快照），纯函数供实时/回补/测试共用。

    特征定义逐字照抄 grand_search_v2.build_conditions（720d 搜索口径）：
    NaN 一律走 np.nan_to_num(NaN→0) 后比较的研究语义；时间取 UTC（CSV timestamp
    为 +00:00，实时 open_time ms → UTC，同一时区语义）。窗口依赖特征在 idx≥672
    预热（r7d），实时仅评估末尾根（idx≥700-12=688）恒已预热。
    """
    n = len(rows)
    t = np.asarray([int(r["open_time"]) for r in rows], dtype=np.int64)
    o = np.asarray([float(r["open"]) for r in rows], dtype=np.float64)
    h = np.asarray([float(r["high"]) for r in rows], dtype=np.float64)
    l = np.asarray([float(r["low"]) for r in rows], dtype=np.float64)
    c = np.asarray([float(r["close"]) for r in rows], dtype=np.float64)

    body = c - o
    sign = np.sign(body)
    rng = h - l
    rng_safe = np.where(rng > 0, rng, np.nan)
    pos = (c - l) / rng_safe
    lower = (np.minimum(o, c) - l) / rng_safe

    # 连续同色 run（doji 重置）——研究 L82-91 逐字
    run = np.zeros(n, dtype=int)
    cur = 0
    for j in range(n):
        if sign[j] == 0:
            cur = 0
        elif j > 0 and sign[j] == sign[j - 1]:
            cur += 1
        else:
            cur = 1
        run[j] = cur

    # EMA20 乖离（alpha=2/21 从头递推）——研究 L151-156 逐字
    alpha = 2.0 / 21
    ema = np.empty(n)
    ema[0] = c[0]
    for j in range(1, n):
        ema[j] = alpha * c[j] + (1 - alpha) * ema[j - 1]
    bias = (c - ema) / ema

    # RSI14（SMA 版）——研究 L143-150 逐字
    diff = np.zeros(n)
    if n > 1:
        diff[1:] = c[1:] - c[:-1]
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    rsi = np.full(n, np.nan)
    if n >= 15:
        from numpy.lib.stride_tricks import sliding_window_view
        mu = sliding_window_view(up, 14).mean(axis=1)
        md = sliding_window_view(dn, 14).mean(axis=1)
        rsi[13:] = 100 * mu / np.where(mu + md == 0, np.nan, mu + md)

    # 含当前根 96 根滚动高（贴1天高基准）——研究 L94（rollmax(h,96)）逐字
    hi1d = np.full(n, np.nan)
    if n >= 96:
        from numpy.lib.stride_tricks import sliding_window_view
        hi1d[95:] = sliding_window_view(h, 96).max(axis=1)

    # 7d 收益——研究 L185-190（ret(672)）逐字
    r672 = np.full(n, np.nan)
    if n > 672:
        r672[672:] = c[672:] / c[:-672] - 1

    # 时间（UTC）：open_time ms → datetime（CSV +00:00 同一时区语义）——研究 L159-167
    dts = [datetime.fromtimestamp(ms / 1000, tz=timezone.utc) for ms in t]
    hours = np.array([d.hour for d in dts])
    weekdays = np.array([d.weekday() for d in dts])

    body_bp = np.abs(body) / o * 1e4
    dist_1dh = (c - hi1d) / hi1d  # 收盘距 1 天高（负=在下方，0=贴着）

    masks = {
        # 研究 L210/L225/L167/L245
        "up3": (sign == 1) & (run >= 3),
        "weekend": weekdays >= 5,
        "bias_pos": np.nan_to_num(bias) >= 0.003,
        # 冻结大实体（mr_freeze_bench 口径，替代研究美元 P80）
        "big_body": body_bp >= BIG_BODY_BP,
        # 研究 L212/L167/L236（布尔数组 nan_to_num 为防御性 no-op，照抄）
        "near1dh": np.nan_to_num(c >= hi1d * (1 - 1e-3), nan=False),
        "us_sess": hours >= 16,
        "r7d_up": np.nan_to_num(r672) >= 0.04,
        # 研究 L205/L203/L244
        "pos_low": np.nan_to_num(pos) <= 0.25,
        "no_lower": np.nan_to_num(lower) <= 0.05,
        "rsi_lt25": np.nan_to_num(rsi) <= 25,
    }
    values = {
        "run_len": run.astype(float),
        "bias": bias,
        "body_bp": body_bp,
        "dist_1dh_bp": dist_1dh * 1e4,
        "r7d_pct": r672 * 100.0,
        "pos": pos,
        "lower": lower,
        "rsi": rsi,
    }
    return masks, values


def evaluate_bars(masks: dict[str, np.ndarray], specs: list[dict], n_tail: int) -> list[dict]:
    """对特征掩码末 n_tail 根逐条求值组合条件（纯函数，供实时/回补/测试共用）。

    返回命中列表：[{"spec", "idx"}]；组合条件 = 全部特征掩码按位与。
    """
    first = next(iter(masks.values()), None)
    n = len(first) if first is not None else 0
    hits: list[dict] = []
    for spec in specs:
        m = np.ones(n, dtype=bool)
        for feat in spec["features"]:
            col = masks.get(feat)
            if col is None:
                logger.warning("combo 影子：特征缺失保守跳过 | {} | {}", spec["version"], feat)
                m = np.zeros(n, dtype=bool)
                break
            m &= col
        tail = m[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off})
    return hits


class ComboShadowDetector:
    """combo 组合条件影子检测器：轮询 15m 收盘 → 条件求值/结算，全程只落表不下注。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        # 目标窗入场报价源（只读共享缓存，与 KREV/反转/nextbar 15m 同源）
        self._pm_15m_latest = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        # 已评估的最大 open_time（15m 收盘水位）
        self._last_evaluated_bar: int | None = None
        self._trigger_count = 0
        self._settle_count = 0
        self._specs = [dict(s) for s in COMBO_SHADOW_SPECS]

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
            logger.warning("combo 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="combo_shadow_detector")
        logger.info(
            "combo 组合影子检测器启动 | {} | 冻结口径实时重放（P1-P3 押收阴 DOWN / P4-P5 押收阳 UP；"
            "720d 62.7~68.7% / oos 59.4~64.8%，影子模式：只记录不下注）",
            "/".join(COMBO_VERSIONS),
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
        logger.info("combo 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("combo 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        """轻探测轮询：limit=2 判断是否有新收盘根，仅新根出现才拉 700 根求值/结算。

        结算依赖次根收盘（=新根出现），故结算也只在新根轮次做；超时兜底只依赖
        DB + 墙钟，每轮执行（拉取失败也不延迟 stale PENDING 清理）。
        """
        probe = await self._collector.fetch_recent_klines(TIMEFRAME, PROBE_BARS)
        if probe:
            last_start = int(probe[-1]["open_time"])
            has_new = self._last_evaluated_bar is None or last_start > self._last_evaluated_bar
            if has_new:
                closed = await self._collector.fetch_recent_klines(TIMEFRAME, WARMUP_BARS)
                # 拉取足够才评估/结算（不足则本轮跳过、不推进水位，下轮重试）
                if len(closed) >= WARMUP_BARS:
                    last_start = int(closed[-1]["open_time"])
                    if self._last_evaluated_bar is None or last_start > self._last_evaluated_bar:
                        await self._evaluate_new_bars(closed)
                        self._last_evaluated_bar = last_start
                    await self._settle_pending(closed)
        await self._expire_stale_pending()

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, closed: list[dict]) -> None:
        """评估水位之后的新收盘根（含冷启动回补的末 12 根）。"""
        masks, values = compute_feature_tables(closed)
        if self._last_evaluated_bar is None:
            n_tail = BACKSCAN_BARS  # 冷启动：回补最近 12 根
        else:
            starts = [int(r["open_time"]) for r in closed]
            try:
                first_new = next(
                    i for i, s in enumerate(starts) if s > self._last_evaluated_bar
                )
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)  # 长停机不追全史，最多回补 12 根
        hits = evaluate_bars(masks, self._specs, n_tail)
        if not hits:
            return
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed[hit["idx"]]
                last_bar = bar
                if await self._record_signal(session, hit["spec"], bar, values, hit["idx"]):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info(
                    "combo 影子触发 +{} | 信号根 {}", added, int(last_bar["open_time"])
                )

    async def _record_signal(self, session, spec: dict, bar: dict, values: dict, idx: int) -> bool:
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
        for feat in spec["snapshot_features"]:
            col = values.get(feat)
            if col is not None and idx < len(col):
                fv = float(col[idx])
                snapshot[feat] = None if np.isnan(fv) else round(fv, 6)
        target_bar_start = start_ms + BAR_MS
        # 目标窗入场报价快照（窗口对齐+近开盘守卫；缺失/回补 → None，该笔 EV 不计）
        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_15m_latest, target_bar_start)
        session.add(KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe=TIMEFRAME,
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
    # 结算（按 direction 判 win：DOWN→次根收阴赢 / UP→次根收阳赢）
    # ------------------------------------------------------------------

    async def _settle_pending(self, closed: list[dict]) -> None:
        """次根收盘结算：回读次根 OHLC 按 direction 判 win，仅认本族 version。"""
        by_start = {int(r["open_time"]): r for r in closed}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(COMBO_VERSIONS),
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
                    # direction=DOWN → 次根收阴赢；direction=UP → 次根收阳赢
                    sig.win = (up and sig.direction == "UP") or (not up and sig.direction == "DOWN")
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "combo 影子结算 | {} | 信号根 {} | 次根 {} → {} | win={}",
                    sig.version, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        """目标根起点后超时仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        cutoff = int(time.time() * 1000) - BAR_MS - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(COMBO_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("combo 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        closed = await self._collector.fetch_recent_klines(TIMEFRAME, WARMUP_BARS)
        if len(closed) < WARMUP_BARS:
            logger.warning("combo 影子：冷启动回补数据不足（{} 根），跳过", len(closed))
            return
        await self._evaluate_new_bars(closed)
        self._last_evaluated_bar = int(closed[-1]["open_time"])
        # 顺带结算停机期间已到期信号
        await self._settle_pending(closed)
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
            "versions": list(COMBO_VERSIONS),
        }
