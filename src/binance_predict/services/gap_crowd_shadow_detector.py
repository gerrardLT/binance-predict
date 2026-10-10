"""报价-人群错位（gap_crowd）影子检测器 + 判定规则。

⛔ 2026-10-10 勘误（结论作废，勿据本文件交易）：
    原研究结论「窗后半段报价比人群倾向便宜 ≥5pp → 买 UP 有费后正 EV」被证伪——
    该触发条件**只**在 up_price/down_price 快照不一致的瞬间成立（盲测 948 事件
    100% 报价和 <0.95、中位 0.91；全样本仅 1.03% 不一致、中位和 1.000）。
    触发价是滞后/偏低值不可成交（生产首单记录价 0.40 vs 实际报价 0.62）。
    以一致化入场价重算 EV：+0.036 → −0.072（只留一致样本则事件数 0）。
    ⇒ 两版本已退役（shadow_version_gate.RETIRED_VERSIONS），规则保留仅为审计与
    防止未来复用；QUOTE_SUM_TOL 一致性门为硬前置。详见研究报告 ERRATUM。

原始研究出处（已被上条勘误推翻，保留供审计）：
    output/research_runs/20261009T191824Z-reversal-state-certainty/
    （FINAL_REPORT.md §2 与文末 ERRATUM；holdout/FROZEN_CANDIDATES.json OBS-1/OBS-2）

冻结规则（含 AMENDMENT-2 一致性门）：
    gap = up_price − (1 − down_pct/100)          # 市场隐含涨概率 − 人群看涨倾向
    触发 = 窗内 elapsed ≥ 窗长×0.5 的首个采样点，且：
             gap ≤ −0.05、距收盘 ≥60s、0.02 ≤ up_price ≤ 0.98、
             |up_price + down_price − 1| ≤ 0.05（AMENDMENT-2，防伪迹快照）
    入场 = 触发时刻真实 up_price；结算 = 触发窗 sign(close−open)（K 线口径）

影子纪律：只记录不下注。触发即落 PENDING，窗关闭后由结算循环拉 K 线判胜负 →
SETTLED。实盘通道同名（已退役、默认 OFF）。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import GapCrowdShadowSignal
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（研究报告 OBS-1/OBS-2，勿动）----
GAP_CROWD_SPECS: dict[str, str] = {
    "gap_crowd_5m_v1": "5m",
    "gap_crowd_15m_v1": "15m",
}
WINDOW_MS: dict[str, int] = {"5m": 300_000, "15m": 900_000}
GAP_THRESHOLD = -0.05        # gap ≤ 此值触发（市场报价比人群倾向便宜 ≥5pp）
LATE_FRACTION = 0.5          # 只取窗口后半段（elapsed ≥ 窗长×0.5）
MIN_REMAIN_S = 60.0          # 距收盘至少 60s（执行可行性门）
Q_LO, Q_HI = 0.02, 0.98      # 入场报价有效区间
# ⚠️ 2026-10-10 勘误后新增（AMENDMENT-2）：报价对一致性门。
# 生产首条信号暴露根因：触发条件（gap≤−0.05）**只会**在 up_price/down_price 快照
# 不一致的瞬间成立——盲测 948 个触发事件 100% 的 up+down < 0.95（中位 0.91），
# 而全样本仅 1.03% 的采样不一致（中位和 1.000）。触发时刻的 up_price 是滞后/偏低
# 值，真实不可成交（实盘首单取到 0.62 而记录价 0.40，被护栏弃单）。
# 用一致化入场价（1−down_price，均值 0.564 而非 0.458）重算：EV 由 +0.036 变 −0.072；
# 只保留一致样本则事件数为 0。⇒ 原「报价落后人群」结论系数据伪迹，规则作废
# （两版本已退役，见 shadow_version_gate.RETIRED_VERSIONS）。本门保留为硬前置：
# 任何未来复用本模块的调用都不会在伪迹快照上触发。
QUOTE_SUM_TOL = 0.05         # |up_price + down_price − 1| 超过此值视为快照不一致 → 不判定
DIRECTION = "UP"             # 固定买便宜侧（gap-low 规则的方向语义）
FEE_RET = 0.98               # EV = 赢 0.98/q−1 / 输 −1（费 2% 无溢价，repo 统一口径）
SETTLE_POLL_INTERVAL = 60.0  # 结算循环轮询间隔（秒）
SETTLE_GRACE_S = 20.0        # 窗关闭后等待结算的宽限（K 线落定）
PENDING_EXPIRE_S = 86_400.0  # PENDING 超 1 天仍无结算数据 → VOID（不污染统计）


def evaluate_gap_crowd(
    period: str,
    window_start_ms: int,
    window_end_ms: int,
    ts_ms: int,
    up_price: float | None,
    down_price: float | None,
    down_pct: float | None,
) -> dict | None:
    """冻结规则判定（影子与实盘共用单点事实源）。

    返回 None = 本采样点不触发（未到后半窗 / gap 未达阈值 / 报价越界 / 数据缺失）；
    返回 dict(eligible=True, gap, q, t_rel, remain_s, prediction) = 触发。

    调用方语义：对同一窗口逐采样调用，取首个非 None 即「首触发」（与研究的
    「窗内首个满足全部条件的采样点」逐字一致）。
    """
    if up_price is None or down_price is None or down_pct is None:
        return None
    span_ms = int(window_end_ms) - int(window_start_ms)
    if span_ms <= 0:
        return None
    elapsed_ms = int(ts_ms) - int(window_start_ms)
    if elapsed_ms < span_ms * LATE_FRACTION:
        return None                                   # 未进入后半窗
    remain_s = (int(window_end_ms) - int(ts_ms)) / 1000.0
    if remain_s < MIN_REMAIN_S:
        return None                                   # 距收盘过近（执行可行性）
    q = float(up_price)
    if not (Q_LO <= q <= Q_HI):
        return None                                   # 报价越界（无效/单边）
    # AMENDMENT-2（2026-10-10）：报价对一致性硬前置——不一致快照的 up_price 不可成交。
    # 容差带 1e-9：0.41+0.54=0.95 的浮点差为 0.050000000000000044，不加容差会把
    # 名义贴线点（和=0.95）误拒；本门为新规则无研究口径需逐字复刻，边界按名义值含贴线。
    if abs(q + float(down_price) - 1.0) > QUOTE_SUM_TOL + 1e-9:
        return None                                   # 报价对不一致（至少一侧滞后）
    gap = q - (1.0 - float(down_pct) / 100.0)
    # ⚠ down_pct 是**百分比**（0-100，与研究读的 prediction_market_samples.down_pct
    # 同口径）；调用方不得传 0-1 小数（main.py 的 down_chance 需 ×100，漏转会退化）。
    # 判定表达式与研究 b_round2.gap_sign 逐字同构（sign(gap)==-1 且 abs(gap)>=thr）：
    # 单边 `gap > GAP_THRESHOLD` 在「名义 −0.05 但浮点落在 −0.049999999999999996」
    # 的格点上与研究的 abs 比较同结果，但为免边界语义分叉，一律用 abs 复刻研究式。
    if gap >= 0.0 or abs(gap) < abs(GAP_THRESHOLD):
        return None                                   # 报价未低于人群倾向阈值
    return {
        "eligible": True,
        "gap": gap,
        "q": q,
        "t_rel": elapsed_ms / 1000.0,
        "remain_s": remain_s,
        "prediction": DIRECTION,
    }


def _ev_at_entry(win: bool, price: float) -> float:
    """单注 EV（repo 统一口径，无溢价）：赢 0.98/q−1 / 输 −1。"""
    return (FEE_RET / price - 1.0) if win else -1.0


class GapCrowdShadowDetector:
    """gap_crowd 影子检测器：采样循环喂入（与实盘同一采样点）→ 首触发落 PENDING
    → 结算循环拉 K 线判胜负 → SETTLED。只记录不下注。

    生命周期由 main lifespan 装配（settings.gap_crowd_shadow_enabled，默认开）。
    只处理「本进程启动后才开始」的窗口（_started_ms 守卫）：重启/中途启用时正在
    进行的窗口不做半途入场——研究验证的是「窗内首个满足全部门的采样点」，
    半途加入的触发点无研究证据，宁缺毋滥（与实盘 check_gap_crowd 同守卫）。
    """

    def __init__(self, *, collector=None) -> None:
        self._collector = collector          # 提供 fetch_klines_ending_at（结算用）
        self._running = False
        self._task: asyncio.Task | None = None
        self._fired: dict[tuple[str, int], int] = {}  # (period, window_start) → 触发 ts（幂等防重）
        self._started_ms = int(time.time() * 1000)  # 启动时刻（窗口守卫基准）
        self._trigger_count = 0
        self._settled_count = 0
        self._void_count = 0
        self._persist_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._started_ms = int(time.time() * 1000)
        self._task = asyncio.create_task(self._settle_loop(), name="gap_crowd_shadow_detector")
        logger.info(
            "报价-人群错位影子检测器启动 | variant {} | 规则 gap≤{} 后半窗(≥{}%) 距收盘≥{}s "
            "| 只记录不下注（研究 L0：5m EV+0.036 n=947 / 15m EV+0.077 n=250 盲测）",
            {v: p for v, p in GAP_CROWD_SPECS.items()}, GAP_THRESHOLD,
            LATE_FRACTION * 100, MIN_REMAIN_S,
        )

    async def stop(self) -> None:
        self._running = False
        for t in list(self._persist_tasks):
            t.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info(
            "报价-人群错位影子检测器已停止 | 触发 {} | 结算 {} | VOID {}",
            self._trigger_count, self._settled_count, self._void_count,
        )

    # ------------------------------------------------------------------
    # 喂入路径（main 采样循环同步调用，纯内存快速路径）
    # ------------------------------------------------------------------

    def observe_sample(
        self,
        period: str,
        window_start_ms: int,
        window_end_ms: int,
        ts_ms: int,
        up_price: float | None,
        down_price: float | None,
        down_pct: float | None,
        participants: float | None = None,
        trade_volume: float | None = None,
    ) -> None:
        """每个 15s 对齐采样点调用一次（5m 与 15m 各自）。命中即派生落表任务。"""
        if not self._running:
            return
        if int(window_start_ms) < self._started_ms:
            return                                   # 启动前已开始的窗口：半途入场无研究证据
        period = period if period in WINDOW_MS else None
        if period is None:
            return
        version = next((v for v, p in GAP_CROWD_SPECS.items() if p == period), None)
        if version is None:
            return
        key = (period, int(window_start_ms))
        if key in self._fired:
            return
        ext = evaluate_gap_crowd(
            period, window_start_ms, window_end_ms, ts_ms,
            up_price, down_price, down_pct,
        )
        if ext is None:
            return
        self._fired[key] = int(ts_ms)
        if len(self._fired) > 1024:
            # 按年龄裁剪（窗口已结束超 1 天的键不再需要；防内存无界）
            cutoff = int(ts_ms) - 86_400_000
            self._fired = {k: v for k, v in self._fired.items() if v >= cutoff}
        task = asyncio.create_task(
            self._persist_pending(
                version, period, int(window_start_ms), int(window_end_ms), int(ts_ms),
                ext, float(up_price), float(down_price), float(down_pct),
                participants, trade_volume,
            ),
            name=f"gap_crowd_pending_{version}_{window_start_ms}",
        )
        self._persist_tasks.add(task)
        task.add_done_callback(self._persist_tasks.discard)

    async def _persist_pending(
        self, version: str, period: str, window_start: int, window_end: int,
        trigger_ts: int, ext: dict, up_price: float, down_price: float, down_pct: float,
        participants: float | None, trade_volume: float | None,
    ) -> None:
        try:
            if not shadow_gate.is_enabled(version):
                return                               # 手动下线：停止采集该版本（历史保留）
            async with async_session_factory() as session:
                dup = await session.execute(
                    sa_select(GapCrowdShadowSignal.id).where(
                        GapCrowdShadowSignal.version == version,
                        GapCrowdShadowSignal.window_start == window_start,
                    )
                )
                if dup.first() is not None:
                    return                               # 幂等（重启后同窗重触发）
                session.add(GapCrowdShadowSignal(
                    version=version,
                    market_period=period,
                    window_start=window_start,
                    window_end=window_end,
                    trigger_ts=trigger_ts,
                    elapsed_s=ext["t_rel"],
                    remain_s=ext["remain_s"],
                    gap=ext["gap"],
                    down_pct=down_pct,
                    direction=DIRECTION,
                    up_price=up_price,
                    down_price=down_price,
                    participants=participants,
                    trade_volume=trade_volume,
                    entry_up_price=up_price,
                    entry_down_price=down_price,
                    entry_quote_ts=trigger_ts,
                    entry_quote_kind="real",
                    status="PENDING",
                ))
                await session.commit()
                self._trigger_count += 1
                logger.info(
                    "报价-人群错位影子触发 | {} | 窗口 {} | t=+{:.0f}s gap={:+.3f} "
                    "down_pct={:.1f}% up={:.3f} | 押{}（PENDING 待结算）",
                    version,
                    datetime.fromtimestamp(window_start / 1000, tz=timezone.utc)
                    .strftime("%m-%d %H:%M"),
                    ext["t_rel"], ext["gap"], down_pct, up_price, DIRECTION,
                )
        except Exception as exc:
            logger.warning("报价-人群错位影子：落表失败 | {} | window {} | {}",
                           version, window_start, exc)

    # ------------------------------------------------------------------
    # 结算循环（K 线口径，与研究的结算代理同定义）
    # ------------------------------------------------------------------

    async def _settle_loop(self) -> None:
        while self._running:
            try:
                await self._settle_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("报价-人群错位影子：结算循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(SETTLE_POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _settle_pending(self) -> None:
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(GapCrowdShadowSignal)
                .where(GapCrowdShadowSignal.status == "PENDING")
                .where(GapCrowdShadowSignal.window_end <= now_ms - int(SETTLE_GRACE_S * 1000))
                .order_by(GapCrowdShadowSignal.window_end.asc())
                .limit(200)
            )).scalars().all()
        for row in rows:
            try:
                await self._settle_one(row, now_ms)
            except Exception as exc:
                logger.warning("报价-人群错位影子：结算失败 | {} | window {} | {}",
                               row.version, row.window_start, exc)

    async def _settle_one(self, row: GapCrowdShadowSignal, now_ms: int) -> None:
        period = row.market_period
        # 超期无结算数据 → VOID（不污染统计；正常路径不应发生）
        if now_ms - int(row.window_end) > PENDING_EXPIRE_S * 1000:
            await self._finalize(row.id, status="VOID", outcome=None, win=None, ev=None)
            self._void_count += 1
            return
        if self._collector is None:
            return                                    # 无 K 线源：保持 PENDING 待重试
        klines = await self._collector.fetch_klines_ending_at(period, 2, int(row.window_end))
        bar = next((k for k in klines if int(k["open_time"]) == int(row.window_start)), None)
        if bar is None:
            return                                    # K 线未就绪：下轮重试
        o, c = float(bar["open"]), float(bar["close"])
        if c == o:
            await self._finalize(row.id, status="VOID", outcome=None, win=None, ev=None)
            self._void_count += 1
            return                                    # 精确平局（研究口径剔除）
        outcome = "UP" if c > o else "DOWN"
        win = outcome == row.direction
        ev = _ev_at_entry(win, float(row.entry_up_price or 0.0))
        await self._finalize(row.id, status="SETTLED", outcome=outcome, win=win, ev=ev)
        self._settled_count += 1
        logger.info(
            "报价-人群错位影子结算 | {} | 窗口 {} | outcome={} 押{} win={} ev={:+.3f}",
            row.version,
            datetime.fromtimestamp(int(row.window_start) / 1000, tz=timezone.utc)
            .strftime("%m-%d %H:%M"),
            outcome, row.direction, win, ev,
        )

    async def _finalize(self, row_id: int, *, status: str, outcome: str | None,
                        win: bool | None, ev: float | None) -> None:
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(GapCrowdShadowSignal).where(GapCrowdShadowSignal.id == row_id)
            )).scalar_one_or_none()
            if row is None or row.status != "PENDING":
                return
            row.status = status
            row.settle_outcome = outcome
            row.win = win
            row.ev_at_entry = ev
            await session.commit()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "trigger_count": self._trigger_count,
            "settled_count": self._settled_count,
            "void_count": self._void_count,
            "fired_windows": len(self._fired),
        }
