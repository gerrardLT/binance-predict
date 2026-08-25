"""报价 edge 影子检测器（quote_momentum / quote_contrarian 影子落表）。

信号定义（与 scripts/local_quote_bin_winrate.py + local_edge_cell_constraints.py
回测口径逐字段对齐，规则冻结勿动）：
    quote_momentum_v1（A 格顺势）：
        5m 窗内 t∈[90,120)s，DOWN token 报价首次进入 [0.69, 0.75) → 押 DOWN。
        回测：胜率 79.9%（Holdout 79%），EV +0.097，日频 13.5（均为 t∈[90,120) 口径）；
        约束来源：0.55-0.75 黄金区 + 唯一 OOS 存活约束 qdepth≥19（q≥0.69）。
        注：曾误写 [90,210)，致事件量 2.7 倍、胜率/EV 虚高（CodeReview High#1），已收敛。
    quote_contrarian_v1（B 格逆势）：
        5m 窗内 t∈[45,60)s，DOWN token 报价首次进入 [0.15, 0.25) → 押 DOWN。
        回测：胜率 24.0%，EV +0.155，日频 13.7；
        裸条件（两轮 30+ 假设 OOS 全灭，无可信约束）。

影子纪律（M4 同款）：只记录不下注、不占风控配额。邮件推送（2026-08-25）：
仅实时新信号且对应通道已开启实盘开火才推（v3a/v3b 不在实盘白名单永不推），
新鲜度闸自动静默回补/重扫；总开关 signal_push_email_enabled，全局日限防轰炸。
数据流（归档后处理，区别于 X4 的次窗结算）：
    1. 每 60s 轮询新归档 SentimentWindow；
    2. 扫 curve_down_price 曲线找规则区间内首个命中点（时点+报价）；
    3. 命中且窗口结算可判（actual_return 非 0 非 NULL）→ 直接落 SETTLED 记录
       （entry = 首个命中点真实报价，win = 本窗 outcome == DOWN，
       EV = 0.98/q−1 / −1，费 2% 无溢价，与回测完全一致）。
    4. 幂等：(version, window_start) 唯一约束防重。
冷启动回补：启动时重扫最近 12 个已归档窗口。

v2 价格门禁版（2026-08-22 5m 粒度归因落地，只加不改；v1 冻结口径原样）：
    quote_momentum_v2   = v1 区间 + 触发时点 BTC 已低于窗口开盘 ≥0.10%（剔假恐慌）；
    quote_contrarian_v2 = v1 区间 + 触发时点 BTC 未高于窗口开盘 ≥0.10%（只接假冲高）。
    门禁数据源 curve_btc_price（与报价曲线同步采样），只用 ≤触发时点采样点（严格 ex-ante）。

v3 环境门禁版（2026-08-24 交替/延续归因落地，只加不改；v1/v2 冻结口径原样）：
    quote_contrarian_v3a = v2 ∩ 前窗 outcome==DOWN（交替环境：前窗跌+本窗涨=V 反弹假冲高）；
    quote_contrarian_v3b = v3a ∩ 触发时点距当日高点回落≥0.30%（含边界，与归因分桶同口径；震荡日：冲高更易衰竭）。
    归因依据（scripts/local_contrarian_v2_brainstorm.py，142 笔）：前窗 DOWN 31.2%/+0.511
    vs 前窗 UP 18.5%/−0.140；叠加距日高≥0.3% → 34.5%/+0.682（Wilson 下界过盈亏平衡线）。
    距日高 = 当日（UTC）已归档窗口（含本窗）curve_btc_price 中 ≤触发时点的最大值；
    处理按 end_time 升序，当日更晚窗口未归档，天然无未来函数；检测器内维护当日
    running high 增量缓存（按 UTC 日重置，乱序窗口触发失效），避免每笔全天曲线回读。
    环境数据缺失（前窗未归档/日高缺失）→ 不落 v3（保守跳过，v1/v2 不受影响）。
    纪律：纯影子（只记录不下注），前向攒 ≥100 笔且 Wilson 下界过线才谈实盘。

深夜时段变体（2026-08-26 落地，只加不改；v1/v2/v3 冻结口径原样）：
    late_night_contrarian_v1 = 北京时间 22~24 时开窗 ∩ t∈[45,90)s DOWN 报价
    首次进入 [0.25,0.30)（报价带与 contrarian_v1 相邻上移）。
    依据：43 天线上数据发现（22~24时×0.25~0.30 EV+0.96）+ 180/360/720 天 K 线
    代理回测修正——历史胜率 ≈34.7%（三档一致），小样本 51~56% 系噪声
    （CI[40%,68%]）；费后 EV +0.21~+0.27 为赔率型边际（盈亏平衡胜率 ≈29%）。
    时段门禁按窗口 start_time 的北京时间 hour ∈[22,24)；未注册门禁的版本不受影响。
    纪律：纯影子前向攒样本，核对真实触发分布与 K 线代理回测一致后才谈实盘接入。

字段语义映射（复用 misalignment_signals 表）：
    end_pct → 触发时刻 DOWN 报价（X4 语义"触发窗末 UP%"的扩展，注释在案）；
    target_window_start → window_start（本窗即目标窗，无次窗）；
    outcome_base → 触发窗结算方向（审计冗余）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import MisalignmentSignal, SentimentWindow
from .signal_notify import fire_signal_email, fmt_bjt, is_fresh_signal, is_live_enabled

logger = logging.getLogger(__name__)

# ---- 冻结口径（回测同源，勿动）----
# rule -> (t_lo_s, t_hi_s, q_lo, q_hi)
QUOTE_EDGE_RULES: dict[str, tuple[float, float, float, float]] = {
    # t∈[90,120)：与回测脚本 Cell A 同口径（用户圈定），79.9%/EV+0.097 即该窗；
    # 曾误写 [90,210) 导致口径漂移（CodeReview High#1），已收敛回 120。
    "quote_momentum_v1": (90.0, 120.0, 0.69, 0.75),
    "quote_contrarian_v1": (45.0, 60.0, 0.15, 0.25),
    # 深夜时段变体：报价带相邻 contrarian_v1 上移；时段门禁见 HOUR_GUARDS。
    "late_night_contrarian_v1": (45.0, 90.0, 0.25, 0.30),
}
FEE_RET = 0.98            # EV = 赢 0.98/q−1 / 输 −1（费 2%，无溢价，回测口径）
POLL_INTERVAL = 60.0      # 轮询间隔（秒）
BACKSCAN_WINDOWS = 12     # 冷启动回补窗口数（1 小时）

# ---- 时段门禁（2026-08-26 深夜变体落地，只加不改）----
# version -> 北京时间 hour [lo, hi)（按窗口 start_time 判定；北京时间 = UTC+8
# 固定偏移无夏令时，直接位移计算）。未注册的版本 → 门禁恒过（存量规则不受影响）。
HOUR_GUARDS: dict[str, tuple[int, int]] = {
    "late_night_contrarian_v1": (22, 24),
}

# ---- v2 价格门禁（2026-08-22 5m 粒度归因落地，只加不改；触发区间与 v1 完全相同）----
# chg% = (触发时点 BTC − 窗口开盘 BTC) / 窗口开盘 × 100，只用 ≤触发时点采样点（严格 ex-ante）：
#   quote_momentum_v2   min_drop −0.10：触发时点已真跌 ≥0.10%（剔"假恐慌"：
#       归因 dip<0.15% 段 wr 40%/EV−0.43 vs dip≥0.15% 段 wr 85%/EV+0.17，双 regime 成立）；
#   quote_contrarian_v2 max_rise +0.10：触发时点未真涨 ≥0.10%（只接"假冲高"：
#       归因 |chg|<0.05% 平盘窗贡献 86% 利润，melt≥0.3% 段 wr 0~7%）。
# 门禁数据缺失（curve_btc_price/entry_price 无）→ 不落 v2（保守跳过，v1 不受影响）。
V2_PRICE_GUARDS: dict[str, tuple[str, str, float]] = {
    # v2 -> (base 规则, 门禁模式, 阈值%)；min_drop=chg≤阈值 / max_rise=chg<阈值
    "quote_momentum_v2": ("quote_momentum_v1", "min_drop", -0.10),
    "quote_contrarian_v2": ("quote_contrarian_v1", "max_rise", 0.10),
}

# ---- v3 环境门禁（2026-08-24 交替/延续归因落地，只加不改；v1/v2 冻结口径原样） ----
# v3 -> 是否额外要求距日高回落≥0.30%（含边界，与归因分桶同口径；v3a 否 / v3b 是）
V3_ENV_GUARDS: dict[str, bool] = {
    "quote_contrarian_v3a": False,
    "quote_contrarian_v3b": True,
}
V3_DD_THRESHOLD = -0.30   # 距日高回落门槛（%，dd ≤ 阈值即过，含边界）


def _pass_hour_guard(version: str, start_ms: int) -> bool:
    """时段门禁：窗口 start_time 的北京时间 hour 须落在 [lo, hi)。

    未注册门禁 → True（存量规则不受影响）。北京时间 = UTC+8 固定偏移，
    epoch 小时 +8 后 mod 24 即北京时间小时（跨天自然正确）。
    """
    guard = HOUR_GUARDS.get(version)
    if guard is None:
        return True
    hour = (start_ms // 3_600_000 + 8) % 24
    return guard[0] <= hour < guard[1]


def _outcome_of(w: SentimentWindow) -> str | None:
    """可判定的结算方向：actual_return None/0 → None（与 X4 结算同口径）。"""
    ret = w.actual_return
    if ret is None or float(ret) == 0.0:
        return None
    return w.outcome if (w.outcome or "") in ("UP", "DOWN") else None


def _find_first_hit(
    curve: list | None, start_ms: int,
    t_lo: float, t_hi: float, q_lo: float, q_hi: float,
) -> tuple[float, int] | None:
    """曲线内首个命中点：(报价, 采样时刻 ms)。

    按时刻升序扫描，返回第一个满足 t_rel∈[t_lo,t_hi)s 且 v∈[q_lo,q_hi) 的点，
    与回测"每周期首个命中报价"同口径。曲线乱序时按 t 排序。
    """
    if not curve:
        return None
    for p in sorted(curve, key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        t_rel = (int(t) - start_ms) / 1000.0
        if t_rel < 0:
            continue
        if t_rel >= t_hi:  # 已过规则窗口，后续更晚
            break
        if t_rel < t_lo:
            continue
        v = float(v)
        if q_lo <= v < q_hi:
            return v, int(t)
    return None


def _up_price_at_or_before(curve: list | None, ts_ms: int) -> float | None:
    """UP 曲线中 ≤ ts_ms 的最晚采样价（决策时刻已知信息，不含未来）。"""
    best = None
    for p in sorted((curve or []), key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts_ms:
            best = float(v)
    return best


def _window_open_btc_price(w: SentimentWindow) -> float | None:
    """窗口开盘 BTC 基准价：entry_price 优先，回退 curve_btc_price 首个有效点。"""
    p = getattr(w, "entry_price", None)
    if p is not None and float(p) > 0:
        return float(p)
    for pt in sorted((getattr(w, "curve_btc_price", None) or []), key=lambda x: x.get("t") or 0):
        v = pt.get("v")
        if v is not None and float(v) > 0:
            return float(v)
    return None


def _pass_v2_price_guard(version: str, w: SentimentWindow, quote_ts: int) -> bool | None:
    """v2 价格门禁：触发时点 BTC vs 窗口开盘的涨跌幅。

    momentum_v2 要求 chg ≤ −0.10%（已真跌）；contrarian_v2 要求 chg < +0.10%（未真涨）。
    门禁数据缺失 → None（保守不落表）。取价复用 ≤ts 最晚采样点逻辑（不含未来）。
    """
    mode, threshold = V2_PRICE_GUARDS[version][1], V2_PRICE_GUARDS[version][2]
    base = _window_open_btc_price(w)
    if base is None:
        return None
    cur = _up_price_at_or_before(getattr(w, "curve_btc_price", None), quote_ts)
    if cur is None:
        return None
    chg_pct = (cur - base) / base * 100.0
    return chg_pct <= threshold if mode == "min_drop" else chg_pct < threshold


async def _prev_window_outcome(session, window_start_ms: int) -> str | None:
    """前一个 5m 窗（window_start−300s）的结算方向；未归档/NOISE → None。

    limit(1) 防御：sentiment_windows 唯一约束是 (start_time, end_time) 对，
    历史导入库可能存在同 start_time 重复行，避免 scalar_one_or_none 抛
    MultipleResultsFound 被误读为"落表失败"。
    """
    prev = (await session.execute(
        sa_select(SentimentWindow.outcome).where(
            SentimentWindow.start_time == window_start_ms - 300_000).limit(1)
    )).scalar_one_or_none()
    return prev if prev in ("UP", "DOWN") else None


def _pass_v3_env_guard(version: str, w: SentimentWindow, quote_ts: int,
                       prev_outcome: str | None, day_high: float | None) -> bool:
    """v3 环境门禁（纯函数）：前窗 DOWN（交替环境）+ 可选距日高回落≥0.30%。

    prev_outcome / day_high 由调用方预查（同窗 v3a/v3b 复用，避免重复 DB 往返）；
    缺失（None）→ False（保守不落表，与 v2 的 None 语义等效）。
    """
    if prev_outcome != "DOWN":
        return False
    if not V3_ENV_GUARDS[version]:
        return True
    trig = _up_price_at_or_before(getattr(w, "curve_btc_price", None), quote_ts)
    if trig is None or day_high is None or day_high <= 0:
        return False
    dd_pct = (trig - day_high) / day_high * 100.0
    return dd_pct <= V3_DD_THRESHOLD


def _ev_at_entry(win: bool, price: float) -> float:
    """单注 EV（回测口径，无溢价）：赢 0.98/q−1 / 输 −1。"""
    return (FEE_RET / price - 1.0) if win else -1.0


class QuoteEdgeDetector:
    """报价 edge 影子检测器：轮询窗口归档 → 首个命中报价 → 直接落 SETTLED。

    归档后处理模式：落表时窗口已结算，无 PENDING 阶段（区别于 X4 的次窗结算）。
    """

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_window_end: int | None = None  # 已处理过的最大窗口 end_time
        self._trigger_count = 0
        # v3b 日高增量缓存：(UTC 日序号, running high, 已并入的最大 window_start)。
        # 窗口按 end_time 升序处理时增量生效；乱序窗口（如污染重扫）触发重置。
        self._dh_day: int | None = None
        self._dh_high: float | None = None
        self._dh_covered: int = -1

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
            logger.warning("报价 edge 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="quote_edge_detector")
        logger.info(
            "报价 edge 影子检测器启动 | v1 规则 {} + v2 价格门禁版 {} + v3 环境门禁版 {}"
            " + 时段门禁 {} | EV=0.98/q−1（费 2%，影子模式只记录不下注）",
            {k: f"t∈[{v[0]:.0f},{v[1]:.0f})s q∈[{v[2]:.2f},{v[3]:.2f})"
             for k, v in QUOTE_EDGE_RULES.items()},
            {k: f"{m}{p:+.2f}%" for k, (_b, m, p) in V2_PRICE_GUARDS.items()},
            {k: ("前窗DOWN" + ("+距日高≥0.30%(含边界)" if dd else ""))
             for k, dd in V3_ENV_GUARDS.items()},
            {k: f"北京{lo}~{hi}时" for k, (lo, hi) in HOUR_GUARDS.items()},
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
        logger.info("报价 edge 影子检测器已停止 | 触发（含回补）{}", self._trigger_count)

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
                logger.warning("报价 edge 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        stmt = (
            sa_select(SentimentWindow)
            .where(SentimentWindow.end_time > (self._last_window_end or 0))
            .order_by(SentimentWindow.end_time.asc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = (await session.execute(stmt)).scalars().all()
        for w in wins:
            try:
                await self._process_window(w)
            except Exception as exc:
                logger.warning("报价 edge 影子：窗口处理失败 | window {} | {}", w.start_time, exc)
            self._last_window_end = max(self._last_window_end or 0, int(w.end_time))

    # ------------------------------------------------------------------
    # v3b 日高（增量缓存）
    # ------------------------------------------------------------------

    async def _day_high_btc(self, session, w: SentimentWindow,
                            quote_ts: int) -> float | None:
        """当日（UTC）已归档窗口（含本窗）BTC 曲线中 ≤quote_ts 的最高价。

        与归因脚本 running high 口径一致；处理按 end_time 升序进行，
        处理本窗时当日更晚窗口尚未归档，天然无未来函数。
        增量缓存：当日首次查询取全范围后推进水位，之后每窗只增量
        拉取水位之后的新窗口，避免全天 JSONB 曲线随日内时间线性回读。
        """
        # 乱序窗口（污染重扫等）会让缓存已并入的曲线失效 → 重置后取全范围
        if self._dh_day is not None and int(w.start_time) < self._dh_covered:
            self._dh_day, self._dh_high, self._dh_covered = None, None, -1
        day_key = quote_ts // 86_400_000
        if self._dh_day == day_key:
            lo = self._dh_covered + 1      # 增量：只拉水位之后的窗口
            best = self._dh_high
        else:
            lo = day_key * 86_400_000
            best = None
        curves = (await session.execute(
            sa_select(SentimentWindow.curve_btc_price).where(
                SentimentWindow.start_time >= lo,
                SentimentWindow.start_time <= w.start_time,
            )
        )).scalars().all()
        for curve in curves:
            for p in curve or []:
                t, v = p.get("t"), p.get("v")
                if t is None or v is None:
                    continue
                if int(t) <= quote_ts and (best is None or float(v) > best):
                    best = float(v)
        # 推进水位：[lo, w.start] 范围窗口已全部并入（查询覆盖该范围）
        self._dh_day = day_key
        self._dh_high = best
        self._dh_covered = max(self._dh_covered, int(w.start_time))
        return best

    # ------------------------------------------------------------------
    # 核心：单窗口处理（扫曲线 → 首个命中 → 落 SETTLED）
    # ------------------------------------------------------------------

    async def _process_window(self, w: SentimentWindow) -> None:
        start_ms, end_ms = int(w.start_time), int(w.end_time)
        outcome = _outcome_of(w)
        if outcome is None:
            return  # NOISE/缺结算：胜负不可判，不产生信号

        # v1 冻结规则 + v2 门禁版 + v3 环境门禁版（同区间，命中后过对应门禁才落表）
        rules: dict[str, tuple[float, float, float, float]] = dict(QUOTE_EDGE_RULES)
        for v2, (base, _mode, _pct) in V2_PRICE_GUARDS.items():
            rules[v2] = QUOTE_EDGE_RULES[base]
        for v3 in V3_ENV_GUARDS:
            rules[v3] = QUOTE_EDGE_RULES["quote_contrarian_v1"]  # v3 基于 contrarian 区间

        # per-rule 独立 commit（CodeReview Low#4）：单规则落表失败不回滚、
        # 不影响另一规则；trigger_count 仅在 commit 成功后自增。
        # v3a/v3b 同窗复用：v2 价格门禁结果与前窗 outcome 只查一次（懒加载）。
        v3_v2_ok: bool | None = None
        v3_prev: str | None = None
        v3_prev_fetched = False
        v3_high: float | None = None
        v3_high_fetched = False
        async with async_session_factory() as session:
            for version, (t_lo, t_hi, q_lo, q_hi) in rules.items():
                try:
                    hit = _find_first_hit(w.curve_down_price, start_ms, t_lo, t_hi, q_lo, q_hi)
                    if hit is None:
                        continue
                    price, quote_ts = hit
                    if not _pass_hour_guard(version, start_ms):
                        continue  # 时段门禁未过 → 该规则跳过（其他规则不受影响）
                    if version in V2_PRICE_GUARDS and _pass_v2_price_guard(version, w, quote_ts) is not True:
                        continue  # 门禁未过/门禁数据缺失 → v2 不落（v1 不受影响）
                    if version in V3_ENV_GUARDS:
                        # v3 = v2 价格门禁 ∩ 环境门禁（chg 门禁先过，环境门禁再判）
                        if v3_v2_ok is None:
                            v3_v2_ok = _pass_v2_price_guard("quote_contrarian_v2", w, quote_ts) is True
                        if not v3_v2_ok:
                            continue
                        if not v3_prev_fetched:
                            v3_prev = await _prev_window_outcome(session, start_ms)
                            v3_prev_fetched = True
                        if V3_ENV_GUARDS[version] and not v3_high_fetched:
                            v3_high = await self._day_high_btc(session, w, quote_ts)
                            v3_high_fetched = True
                        if not _pass_v3_env_guard(version, w, quote_ts, v3_prev, v3_high):
                            continue  # 环境门禁未过/数据缺失 → v3 不落（v1/v2 不受影响）
                    dup = await session.execute(
                        sa_select(MisalignmentSignal.id).where(
                            MisalignmentSignal.version == version,
                            MisalignmentSignal.window_start == start_ms,
                        )
                    )
                    if dup.first() is not None:
                        continue
                    win = outcome == "DOWN"
                    session.add(MisalignmentSignal(
                        version=version,
                        window_start=start_ms,
                        window_end=end_ms,
                        end_pct=price,               # 语义扩展：触发时刻 DOWN 报价
                        outcome_base=outcome,          # 触发窗结算方向（审计冗余）
                        direction="DOWN",
                        target_window_start=start_ms,  # 本窗即目标窗
                        entry_down_price=price,
                        entry_up_price=_up_price_at_or_before(w.curve_up_price, quote_ts),
                        entry_quote_ts=quote_ts,
                        entry_quote_kind="real",
                        settle_outcome=outcome,
                        win=win,
                        ev_at_entry=_ev_at_entry(win, price),
                        status="SETTLED",
                    ))
                    await session.commit()
                    self._trigger_count += 1
                    logger.info(
                        "报价 edge 影子触发+结算 | {} | 窗口 {} | t=+{:.0f}s q={:.3f}"
                        " → {} | win={} ev={:+.3f}",
                        version,
                        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                        .strftime("%m-%d %H:%M"),
                        (quote_ts - start_ms) / 1000.0, price, outcome, win,
                        _ev_at_entry(win, price),
                    )
                    # 邮件推送（fire-and-forget）：仅实时新信号且该通道已开
                    # 实盘开火；回补/重扫被新鲜度闸静默，v3a/v3b 不在实盘
                    # 白名单永不推。
                    if is_fresh_signal(end_ms) and is_live_enabled(version):
                        win_str = "赢" if win else "输"
                        fire_signal_email(
                            "quote_edge",
                            f"[信号] {version} | 押DOWN {win_str} | 窗口 "
                            f"{fmt_bjt(start_ms)} 北京时间",
                            f"版本: {version}（影子，只记录不下注）\n"
                            f"窗口: {fmt_bjt(start_ms)}~{fmt_bjt(end_ms, with_date=False)} 北京时间\n"
                            f"触发: t=+{(quote_ts - start_ms) / 1000.0:.0f}s DOWN报价 q={price:.3f}\n"
                            f"结算: {outcome} → {win_str}\n"
                            f"EV: {_ev_at_entry(win, price):+.3f}（0.98/q−1 / −1，费 2%）",
                        )
                except Exception as exc:
                    await session.rollback()
                    logger.warning("报价 edge 影子：rule {} 落表失败 | window {} | {}",
                                   version, start_ms, exc)

    # ------------------------------------------------------------------
    # 冷启动回补
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        stmt = (
            sa_select(SentimentWindow)
            .order_by(SentimentWindow.end_time.desc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = list(reversed((await session.execute(stmt)).scalars().all()))
        # 单窗失败不中断整体；无论成败都推进水位——宁可跳过回补，
        # 绝不让水位停在 None 触发后续从最老窗口全量回灌历史（CodeReview Medium#2）。
        for w in wins:
            try:
                await self._process_window(w)
            except Exception as exc:
                logger.warning("报价 edge 影子：回补单窗失败（跳过） | window {} | {}",
                               getattr(w, "start_time", None), exc)
        self._last_window_end = max((int(w.end_time) for w in wins), default=None)
        if wins:
            logger.info(
                "报价 edge 影子：冷启动回补 {} 窗（{}~{}）完成",
                len(wins), wins[0].start_time, wins[-1].end_time,
            )

    # ------------------------------------------------------------------
    # 状态（status API 用）
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """状态（status API 用）。注：trigger_count 计全部版本落表行数，
        2026-08-24 起含 v3a/v3b 影子行，增速高于市场事件数属预期口径漂移。"""
        return {
            "running": self._running,
            "last_window_end": self._last_window_end,
            "trigger_count": self._trigger_count,
        }
