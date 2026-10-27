"""反向格影子检测器（cic 族）：15m 窗口第 8 分钟「浅跌/浅涨+震荡+反向格」判定重放。

信号定义（冻结自研究链 20261027T005754Z-5m-intra-structure →
20261027T020848Z-intra-progress-direction → 20261027T023421Z-15m-book-cell-pricing
→ 20261027T025526Z-cell-factor-buckets（含 ERRATUM 量纲修正 v1.2）；
SIGNAL_SPEC.md v1.2 同源，改值须回研究轮重新冻结）：

    决策时刻 t* = window_start + 8min（第 8 根 1m 收盘）+ 3.5s 计算时延；
    数据 = 窗口内前 8 根 1m（已收盘）+ 窗口开始前 288 个 15m 根（已收盘，严格 ex-ante）。

    反向格：cum = P_8 − P_open 与首 5m 桶方向相反
        cum < 0 且 b0 = B0_close − P_open > 0 → 买 UP（场景A：已跌+首桶阳）
        cum > 0 且 b0 < 0                        → 买 DOWN（场景B：已涨+首桶阴）
        平盘（cum==0 或 b0==0）无信号
    过滤①浅跌：z1 = |ln(P_8/P_open)×1e4| / vol24 < 0.5
        （vol24 = 窗口开始前 288 个 15m 窗口平均 |ln(close/open)|×1e4——研究
        实现口径即此（ERRATUM：字段名误标 5m/24h，实为 15m 窗口×288=72h；
        阈值 0.5 与全部研究数字都基于该实现，生产必须逐字对齐而非对齐名义口径））
    过滤②震荡：eff = |P_8 − P_open| / path8 < 0.1878（无量纲前缀效率，
        path8 = Σ|1m收盘差分|（7 项）；人口中位数冻结，量纲修正见 ERRATUM）
    护栏（下单层）：成交均价 < 0.45（含贴线弃单），由通道 auto_max_exec 承载。

研究依据（2y 人口 n=14,399 格子 + 本地 PREDEXON 盘口样本）：
    精筛子集（z1<0.5 ∧ eff<中位）人口胜率 41.4%（同期 43.5%）、盘口样本 n=130
    胜率 53.1%、一档均价 0.436、EV +0.095/份（分月 07/08/09/10 = +0.086/+0.014/
    +0.149/+0.101，四个月全正）；深跌格（z1≥0.5）为负 EV 陷阱（EV −0.048）。
    ⚠ EV 统计显著性未达成（样本内挖掘+样本小），影子前向验证为晋级必要条件。

影子纪律：每个反向格窗口落一行（PENDING，含过滤器否决的格子——分层对账）；
    过滤器通过时回填 entry 报价并触发实盘钩子；窗口收盘按 close vs open 结算
    （行方向 UP/DOWN 判 win）。只落表不下注；实盘开火由 on_cic_fire 钩子派发到
    MultiLiveTrader.on_cic_signal（通道默认 OFF，toggle 开闸）。
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal

CIC_VERSION = "cic_shallow_15m_v1"
# 冻结阈值（研究链 20261027T025526Z-cell-factor-buckets ERRATUM v1.2）
Z1_MAX = 0.5          # 浅跌/浅涨门（|cum| / vol24_5m，严格小于）
EFF_MAX = 0.1878      # 前缀效率门（无量纲 |P8−Popen|/path8，人口中位，严格小于）
DECIDE_T_REL_S = 483.5   # 决策时刻 = +8min 收盘 + 3.5s 时延（研究快照口径）
DECIDE_DEADLINE_S = 660.0  # 决策重试上限（K线暂缺时等下一采样；过线放弃本窗）
VOL_BARS = 288        # vol24 需要 288 根 15m（研究冻结口径，勿改成 5m/24h）
BAR_15M_MS = 900_000
BAR_1M_MS = 60_000
SETTLE_POLL_S = 15.0


def evaluate_cic8(bars_1m: list[dict], bars_15m: list[dict],
                  window_start: int) -> dict | None:
    """纯函数：反向格 + 过滤器判定（严格 ex-ante，只用已收盘根）。

    bars_1m：collector.fetch_recent_klines("1m", ...)（已收盘根，升序）；
    bars_15m：collector.fetch_recent_klines("15m", ...)（已收盘根，升序）。
    数据不足 / 价格非法 → None（保守无信号，调用方可重试至决策截止）。
    返回 {"cell": "A"|"B"|None, "direction": "UP"|"DOWN"|None, "cum_bps",
    "b0_bps", "z1", "eff", "path8", "vol24_5m", "filters_pass"}。
    """
    by_t = {int(b["open_time"]): b for b in bars_1m}
    m1 = [by_t.get(window_start + i * BAR_1M_MS) for i in range(8)]
    if any(b is None for b in m1):
        return None
    try:
        p_open = float(m1[0]["open"])
        closes = [float(b["close"]) for b in m1]
    except (TypeError, ValueError):
        return None
    if p_open <= 0 or any(c <= 0 for c in closes):
        return None
    p8, b0_close = closes[7], closes[4]
    # vol24：窗口开始前 288 个 15m 窗口平均 |收益| bps（研究冻结实现口径）
    prev15 = [b for b in bars_15m if int(b["open_time"]) + BAR_15M_MS <= window_start]
    if len(prev15) < VOL_BARS:
        return None
    rets = []
    for b in prev15[-VOL_BARS:]:
        o, c = float(b["open"]), float(b["close"])
        if o <= 0 or c <= 0:
            return None
        rets.append(abs(math.log(c / o)) * 1e4)
    vol24_5m = sum(rets) / len(rets)   # 名字保留研究字段名，口径=15m×288（见 ERRATUM）
    if vol24_5m <= 0:
        return None
    cum_bps = math.log(p8 / p_open) * 1e4
    b0_bps = math.log(b0_close / p_open) * 1e4
    path8 = sum(abs(closes[i] - closes[i - 1]) for i in range(1, 8))
    eff = abs(p8 - p_open) / path8 if path8 > 0 else 1.0
    z1 = abs(cum_bps) / vol24_5m
    if cum_bps < 0 and b0_bps > 0:
        cell, direction = "A", "UP"
    elif cum_bps > 0 and b0_bps < 0:
        cell, direction = "B", "DOWN"
    else:
        cell, direction = None, None
    filters_pass = bool(cell) and z1 < Z1_MAX and eff < EFF_MAX
    return {
        "cell": cell, "direction": direction,
        "cum_bps": cum_bps, "b0_bps": b0_bps,
        "z1": z1, "eff": eff, "path8": path8,
        "vol24": vol24_5m, "filters_pass": filters_pass,
    }


class CicCounterInertiaShadowDetector:
    """反向格影子检测器：15m 采样喂价 → 第 8 分钟决策（每窗一次）→
    格子落行 + 触发回填 + 实盘钩子；后台循环结算 PENDING。"""

    def __init__(self, collector) -> None:
        self._collector = collector
        self._running = False
        self._task: asyncio.Task | None = None
        self._evaluated: set[int] = set()        # 已完成判定的 window_start（含非格子）
        self._row_ids: dict[int, int] = {}       # window_start -> 格子行 DB id
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._settle_loop(), name="cic_shadow_detector")
        logger.info("反向格检测器启动 | {} | 影子落表 + 实盘钩子（默认关闭）",
                    CIC_VERSION)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("反向格检测器已停止 | 触发 {} 结算 {}", self._trigger_count,
                    self._settle_count)

    async def observe_sample(self, window_start: int, window_end: int,
                             ts_ms: int, up_price: float | None,
                             down_price: float | None) -> None:
        """每次 15m 对齐采样调用（main._handle_15m persist 路径）。"""
        t_rel = (ts_ms - window_start) / 1000.0
        if t_rel < DECIDE_T_REL_S or t_rel >= DECIDE_DEADLINE_S:
            return
        if window_start in self._evaluated:
            return
        bars1 = await self._collector.fetch_recent_klines("1m", 16)
        bars15 = await self._collector.fetch_recent_klines("15m", VOL_BARS + 3)
        if not bars1 or not bars15:
            return              # 瞬态拉取失败：不标记已判定，下轮采样重试
        info = evaluate_cic8(bars1, bars15, window_start)
        if info is None:
            return              # K 线未齐（第 8 根未收/15m 根数不足）：下轮重试
        self._evaluated.add(window_start)
        self._prune(window_start)
        if not info["cell"]:
            return              # 非反向格：不落行（人口出现率 20.9% 已知）
        row_id = await self._ensure_cell_row(window_start, info)
        if row_id is None:
            return
        if not info["filters_pass"]:
            logger.info("反向格影子：格子命中但被过滤器否决 | {} | 窗口 {} | "
                        "cell={} z1={:.3f} eff={:.3f}", CIC_VERSION, window_start,
                        info["cell"], info["z1"], info["eff"])
            return
        # 触发：回填 entry 报价 + 实盘钩子（价格缺失只落表不开火，防盲市价单）
        async with async_session_factory() as session:
            row = await session.get(KlineShadowSignal, row_id)
            if row is None:
                return
            if ((row.feature_snapshot or {}).get("touched")
                    or row.entry_quote_ts is not None):
                return
            snap = dict(row.feature_snapshot or {})
            snap["touched"] = True
            snap["touched_t_rel_s"] = round(t_rel, 1)
            snap["entry_quote_kind"] = "market_snapshot"
            row.feature_snapshot = snap
            if up_price is not None and down_price is not None:
                row.entry_up_price = float(up_price)
                row.entry_down_price = float(down_price)
            row.entry_quote_ts = ts_ms
            await session.commit()
        self._trigger_count += 1
        logger.info("反向格影子触发 | {} | 窗口 {} | cell={} 押 {} | t=+{}s | "
                    "z1={:.3f} eff={:.3f} cum={:.1f}bp", CIC_VERSION, window_start,
                    info["cell"], info["direction"], round(t_rel, 1),
                    info["z1"], info["eff"], info["cum_bps"])
        if (self._on_live_fire is not None
                and up_price is not None and down_price is not None):
            try:
                self._on_live_fire({
                    "version": CIC_VERSION,
                    "market_start": window_start,
                    "market_end": window_end,
                    "direction": info["direction"],
                    "signal_bar_start": window_start,
                    "t_rel": round(t_rel, 1),
                    "z1": round(info["z1"], 4),
                    "eff": round(info["eff"], 4),
                    "touch_up": float(up_price),
                    "touch_down": float(down_price),
                })
            except Exception as exc:
                logger.warning("反向格实盘钩子异常（不影响影子）| {}", exc)

    def _prune(self, window_start: int) -> None:
        if len(self._evaluated) <= 2048:
            return
        cutoff = window_start - 8 * BAR_15M_MS
        self._evaluated = {k for k in self._evaluated if k >= cutoff}
        self._row_ids = {k: v for k, v in self._row_ids.items() if k >= cutoff}

    async def _ensure_cell_row(self, window_start: int, info: dict) -> int | None:
        """格子行只落一次（DB 存在性核验 + 内存缓存，防重启重复落表）。"""
        if window_start in self._row_ids:
            return self._row_ids[window_start]
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == CIC_VERSION,
                    KlineShadowSignal.signal_bar_start == window_start,
                ).limit(1))).first()
            if exists:
                self._row_ids[window_start] = int(exists[0])
                return int(exists[0])
            row = KlineShadowSignal(
                version=CIC_VERSION,
                discovery_id="cic_counter_inertia_15m",
                condition_text=(
                    f"cell(cum,b0 opposite) AND z1<{Z1_MAX} AND eff<{EFF_MAX}"
                    f" AND decide@+{DECIDE_T_REL_S}s"),
                timeframe="15m",
                signal_bar_start=window_start,
                signal_bar_end=window_start + BAR_15M_MS,
                direction=info["direction"],
                target_bar_start=window_start,
                feature_snapshot={
                    "cell": info["cell"],
                    "cum_bps": round(info["cum_bps"], 4),
                    "b0_bps": round(info["b0_bps"], 4),
                    "z1": round(info["z1"], 4),
                    "eff": round(info["eff"], 4),
                    "vol24_15m_x288": round(info["vol24"], 4),
                    "path8": round(info["path8"], 6),
                    "filters_pass": info["filters_pass"],
                },
                status="PENDING",
            )
            session.add(row)
            await session.commit()
            self._row_ids[window_start] = int(row.id)
            return int(row.id)

    async def _settle_loop(self) -> None:
        """后台结算：窗口收盘后按 close vs open 结算格子行（行方向判 win）。"""
        while self._running:
            try:
                await self._settle_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("反向格影子：结算循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(SETTLE_POLL_S)
            except asyncio.CancelledError:
                break

    async def _settle_pending(self) -> None:
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == CIC_VERSION,
                    KlineShadowSignal.status == "PENDING",
                ))).scalars().all()
            if not rows:
                return
            oldest = min(int(r.target_bar_start) for r in rows)
            need = max(8, min((now_ms - oldest) // BAR_15M_MS + 2, 288))
            closed15 = await self._collector.fetch_recent_klines("15m", need)
            by_start = {int(b["open_time"]): b for b in closed15} if closed15 else {}
            for r in rows:
                bar = by_start.get(int(r.target_bar_start))
                if bar is None or int(bar["open_time"]) + BAR_15M_MS > now_ms:
                    if now_ms - int(r.target_bar_start) > 10 * BAR_15M_MS:
                        r.status = "EXPIRED"
                    continue
                o, c = float(bar["open"]), float(bar["close"])
                r.settle_open, r.settle_close = o, c
                outcome = "UP" if c > o else ("DOWN" if c < o else "NOISE")
                r.settle_outcome = outcome
                r.win = (outcome == r.direction) if outcome != "NOISE" else None
                r.status = "SETTLED"
                r.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
            await session.commit()
