"""5m/15m 长影短实体反转检测器。

每根 K 线、每周期只落一个物理事件；18 个冻结研究版本以
``feature_snapshot.matched_signal_ids`` 标签保存，避免主版本、严格子集、
归因组件和后验假设被误当成多笔独立信号。逻辑版本可注册实盘通道，但全部
默认关闭；检测器只分派新鲜命中，是否下单由 MultiLiveTrader 独立门禁。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal, PredictionMarketSample
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.shadow_version_gate import shadow_gate

BAR_MS = {"5m": 300_000, "15m": 900_000}
TIMEFRAMES = ("5m", "15m")
POLL_INTERVAL = 60.0
WARMUP_BARS = 60
BACKSCAN_BARS = 12
PENDING_EXPIRE_MS = {"5m": 3_600_000, "15m": 4 * 3_600_000}
QUOTE_MAX_OFFSET_MS = 120_000
QUOTE_DELAYS = (2, 15, 30, 60, 120)
QUOTE_SUM_MIN = 0.95
QUOTE_SUM_MAX = 1.05
EPS = 1e-12

EVENT_VERSION_BY_TF = {
    "5m": "candle_hm_bull_5m_event_v1",
    "15m": "candle_hm_bull_15m_event_v1",
}

CANDLESTICK_LOGICAL_SPECS: tuple[dict[str, str], ...] = (
    {"signal_id": "candle_hm_bull_5m_consensus2_shadow_v1", "timeframe": "5m", "tier": "DEPLOY_NOW", "role": "PRIMARY"},
    {"signal_id": "candle_hm_bull_5m_consensus3_shadow_v1", "timeframe": "5m", "tier": "DEPLOY_NOW", "role": "CONTROL"},
    {"signal_id": "candle_hm_bull_15m_union_shadow_v1", "timeframe": "15m", "tier": "DEPLOY_NOW", "role": "PRIMARY"},
    {"signal_id": "candle_hm_bull_15m_consensus3_shadow_v1", "timeframe": "15m", "tier": "DEPLOY_NOW", "role": "CONTROL"},
    {"signal_id": "candle_hm_bull_5m_ret5_majority_component_shadow_v1", "timeframe": "5m", "tier": "COMPONENT_NOW", "role": "COMPONENT"},
    {"signal_id": "candle_hm_bull_5m_ret3_component_shadow_v1", "timeframe": "5m", "tier": "COMPONENT_NOW", "role": "COMPONENT"},
    {"signal_id": "candle_hm_bull_15m_ret3_component_shadow_v1", "timeframe": "15m", "tier": "COMPONENT_NOW", "role": "COMPONENT"},
    {"signal_id": "candle_hm_bull_15m_ma10_component_shadow_v1", "timeframe": "15m", "tier": "COMPONENT_NOW", "role": "COMPONENT"},
    {"signal_id": "candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1", "timeframe": "5m", "tier": "POTENTIAL_STRONG", "role": "HYPOTHESIS"},
    {"signal_id": "candle_hm_bull_5m_trend4h_up_shadow_hyp_v1", "timeframe": "5m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candle_hm_bull_5m_asia_shadow_hyp_v1", "timeframe": "5m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candle_hm_bull_15m_q45_55_shadow_hyp_v1", "timeframe": "15m", "tier": "POTENTIAL_STRONG", "role": "HYPOTHESIS"},
    {"signal_id": "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1", "timeframe": "15m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candle_hm_bull_15m_asia_shadow_hyp_v1", "timeframe": "15m", "tier": "POTENTIAL_LOW_N", "role": "HYPOTHESIS"},
    {"signal_id": "candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1", "timeframe": "5m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1", "timeframe": "5m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1", "timeframe": "15m", "tier": "POTENTIAL", "role": "HYPOTHESIS"},
    {"signal_id": "candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1", "timeframe": "15m", "tier": "POTENTIAL_LOW_N", "role": "HYPOTHESIS"},
)
CANDLESTICK_SIGNAL_IDS = tuple(spec["signal_id"] for spec in CANDLESTICK_LOGICAL_SPECS)
LOGICAL_SPEC_BY_ID = {spec["signal_id"]: spec for spec in CANDLESTICK_LOGICAL_SPECS}
CANDLESTICK_BACKTEST: dict[str, tuple[float, float, int]] = {
    "candle_hm_bull_5m_consensus2_shadow_v1": (0.57425743, 0.14695252, 303),
    "candle_hm_bull_5m_consensus3_shadow_v1": (0.59067358, 0.17651603, 193),
    "candle_hm_bull_15m_union_shadow_v1": (0.66197183, 0.26975357, 71),
    "candle_hm_bull_15m_consensus3_shadow_v1": (0.73333333, 0.40197905, 30),
    "candle_hm_bull_5m_ret5_majority_component_shadow_v1": (0.58, 0.15746431, 250),
    "candle_hm_bull_5m_ret3_component_shadow_v1": (0.56804734, 0.13265524, 338),
    "candle_hm_bull_15m_ret3_component_shadow_v1": (0.70175439, 0.35381524, 57),
    "candle_hm_bull_15m_ma10_component_shadow_v1": (0.69811321, 0.3298541, 53),
    "candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1": (0.70322581, 0.39753211, 155),
    "candle_hm_bull_5m_trend4h_up_shadow_hyp_v1": (0.59585492, 0.18858136, 193),
    "candle_hm_bull_5m_asia_shadow_hyp_v1": (0.60169492, 0.19644278, 118),
    "candle_hm_bull_15m_q45_55_shadow_hyp_v1": (0.70588235, 0.41624235, 51),
    "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1": (0.66666667, 0.26751519, 51),
    "candle_hm_bull_15m_asia_shadow_hyp_v1": (0.75, 0.44002248, 24),
    "candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1": (0.58870968, 0.15388355, 124),
    "candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1": (0.55555556, 0.11314249, 135),
    "candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1": (0.6, 0.14612602, 40),
    "candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1": (0.61290323, 0.13982741, 31),
}
CANDLESTICK_DISPLAY_NAMES = {
    "candle_hm_bull_5m_consensus2_shadow_v1": "5m长下影·至少2趋势",
    "candle_hm_bull_5m_consensus3_shadow_v1": "5m长下影·三趋势对照",
    "candle_hm_bull_15m_union_shadow_v1": "15m长下影·任一趋势",
    "candle_hm_bull_15m_consensus3_shadow_v1": "15m长下影·三趋势对照",
    "candle_hm_bull_5m_ret5_majority_component_shadow_v1": "5m ret5归因组件",
    "candle_hm_bull_5m_ret3_component_shadow_v1": "5m ret3归因组件",
    "candle_hm_bull_15m_ret3_component_shadow_v1": "15m ret3归因组件",
    "candle_hm_bull_15m_ma10_component_shadow_v1": "15m ma10归因组件",
    "candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1": "5m长下影·中高量能",
    "candle_hm_bull_5m_trend4h_up_shadow_hyp_v1": "5m长下影·4h上涨",
    "candle_hm_bull_5m_asia_shadow_hyp_v1": "5m长下影·亚洲时段",
    "candle_hm_bull_15m_q45_55_shadow_hyp_v1": "15m长下影·报价0.45~0.55",
    "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1": "15m长下影·1h上涨",
    "candle_hm_bull_15m_asia_shadow_hyp_v1": "15m长下影·亚洲时段",
    "candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1": "5m上涨阳线长上影",
    "candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1": "5m严格零上影阴线长下影",
    "candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1": "15m ret5阳线长下影",
    "candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1": "15m下跌阳线长上影",
}
CANDLESTICK_DIRECTIONS = {
    signal_id: ("UP" if signal_id == "candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1" else "DOWN")
    for signal_id in CANDLESTICK_SIGNAL_IDS
}
VERSIONS_BY_TF = {tf: [EVENT_VERSION_BY_TF[tf]] for tf in TIMEFRAMES}
CANDLESTICK_LIVE_MAX_LAG_MS = 90_000


def _ratios(bar: dict) -> tuple[float, float, float] | None:
    o, h, l, c = (float(bar[key]) for key in ("open", "high", "low", "close"))
    rng = h - l
    if rng <= 0:
        return None
    return abs(c - o) / rng, (h - max(o, c)) / rng, (min(o, c) - l) / rng


def _trend_snapshot(rows: list[dict], i: int, bar_ms: int) -> dict[str, Any] | None:
    if i < 48:
        return None
    t = np.asarray([int(row["open_time"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(t[max(0, i - 48):i + 1]) != bar_ms):
        return None
    o = np.asarray([float(row["open"]) for row in rows], dtype=float)
    c = np.asarray([float(row["close"]) for row in rows], dtype=float)
    v = np.asarray([float(row["volume"]) for row in rows], dtype=float)
    ret3 = c[i - 1] / o[i - 3] - 1.0
    ret5 = c[i - 1] / o[i - 5] - 1.0
    dirs = np.sign(c[i - 5:i] - o[i - 5:i])
    bull5 = int(np.sum(dirs > 0))
    ma10_base = float(np.mean(c[i - 10:i]))
    ma10 = c[i - 1] / ma10_base - 1.0 if ma10_base > 0 else np.nan
    h1_bars = max(1, 3_600_000 // bar_ms)
    h4_bars = max(1, 14_400_000 // bar_ms)
    ret1h = c[i - 1] / o[i - h1_bars] - 1.0
    ret4h = c[i - 1] / o[i - h4_bars] - 1.0
    med20 = float(np.median(v[i - 20:i]))
    volume_ratio = v[i] / med20 if med20 > 0 else np.nan
    flags = {
        "ret3": bool(ret3 > 0),
        "ret5_majority": bool(ret5 > 0 and bull5 >= 3),
        "ma10": bool(ma10 > 0),
    }
    return {
        "flags": flags,
        "trend_hits": sum(flags.values()),
        "values": {
            "ret3": float(ret3), "ret5": float(ret5), "ret5_bull_count": bull5,
            "ma10_distance": float(ma10), "ret1h": float(ret1h),
            "ret4h": float(ret4h), "volume_ratio": float(volume_ratio),
        },
    }


def _shape(body: float, upper: float, lower: float, *, major: str, long: float, body_max: float, minor: float) -> bool:
    major_value, minor_value = (lower, upper) if major == "LOWER" else (upper, lower)
    return major_value + EPS >= long and body <= body_max + EPS and minor_value <= minor + EPS


def _enabled_labels(ids: list[str]) -> list[str]:
    return [signal_id for signal_id in ids if shadow_gate.is_enabled(signal_id)]


def evaluate_candlestick_patterns(rows: list[dict], timeframe: str, n_tail: int) -> list[dict]:
    """求值末 n_tail 根；每根最多返回一个事件，逻辑版本以标签列出。"""
    if timeframe not in BAR_MS:
        raise ValueError(f"不支持的周期: {timeframe}")
    bar_ms = BAR_MS[timeframe]
    hits: list[dict] = []
    for i in range(max(48, len(rows) - n_tail), len(rows)):
        trend = _trend_snapshot(rows, i, bar_ms)
        ratios = _ratios(rows[i])
        if trend is None or ratios is None:
            continue
        body, upper, lower = ratios
        o, c = float(rows[i]["open"]), float(rows[i]["close"])
        bull, bear = c > o, c < o
        flags, count = trend["flags"], trend["trend_hits"]
        values = trend["values"]
        hour = datetime.fromtimestamp(int(rows[i]["open_time"]) / 1000, tz=timezone.utc).hour
        labels: list[str] = []
        direction = "DOWN"

        main_shape = bull and _shape(body, upper, lower, major="LOWER", long=.5, body_max=.5, minor=.1)
        if main_shape:
            if timeframe == "5m":
                if count >= 2:
                    labels.append("candle_hm_bull_5m_consensus2_shadow_v1")
                if count == 3:
                    labels.append("candle_hm_bull_5m_consensus3_shadow_v1")
                if flags["ret5_majority"]:
                    labels.append("candle_hm_bull_5m_ret5_majority_component_shadow_v1")
                if flags["ret3"]:
                    labels.append("candle_hm_bull_5m_ret3_component_shadow_v1")
                if count >= 2 and .75 <= values["volume_ratio"] < 2.0:
                    labels.append("candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1")
                if count >= 2 and values["ret4h"] > 0:
                    labels.append("candle_hm_bull_5m_trend4h_up_shadow_hyp_v1")
                if count >= 2 and hour < 8:
                    labels.append("candle_hm_bull_5m_asia_shadow_hyp_v1")
            else:
                if count >= 1:
                    labels.append("candle_hm_bull_15m_union_shadow_v1")
                if count == 3:
                    labels.append("candle_hm_bull_15m_consensus3_shadow_v1")
                if flags["ret3"]:
                    labels.append("candle_hm_bull_15m_ret3_component_shadow_v1")
                if flags["ma10"]:
                    labels.append("candle_hm_bull_15m_ma10_component_shadow_v1")
                if count >= 1 and values["ret1h"] > 0:
                    labels.append("candle_hm_bull_15m_trend1h_up_shadow_hyp_v1")
                if count >= 1 and hour < 8:
                    labels.append("candle_hm_bull_15m_asia_shadow_hyp_v1")

        if timeframe == "5m":
            if bull and flags["ret5_majority"] and _shape(body, upper, lower, major="UPPER", long=.7, body_max=.3, minor=.1):
                labels.append("candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1")
            # M0 必须用原始 OHLC 严格等于零；比例容差只用于其他阈值边界。
            raw_upper = float(rows[i]["high"]) - max(o, c)
            if bear and flags["ret3"] and raw_upper == 0.0 and _shape(body, upper, lower, major="LOWER", long=.5, body_max=.5, minor=0.0):
                labels.append("candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1")
        else:
            if bull and flags["ret5_majority"] and _shape(body, upper, lower, major="LOWER", long=.5, body_max=.4, minor=.1):
                labels.append("candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1")
            if bull and not flags["ret3"] and values["ret3"] < 0 and _shape(body, upper, lower, major="UPPER", long=.5, body_max=.4, minor=.1):
                labels.append("candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1")
                direction = "UP"

        labels = list(dict.fromkeys(labels))
        if labels:
            hits.append({
                "idx": i,
                "timeframe": timeframe,
                "direction": direction,
                "matched_signal_ids": labels,
                "snapshot": {
                    "trend_flags": flags,
                    "trend_hits": count,
                    "trend_values": values,
                    "geometry": {
                        "body_ratio": float(body),
                        "upper_wick_ratio": float(upper),
                        "lower_wick_ratio": float(lower),
                        "candle_color": "BULL" if bull else "BEAR",
                    },
                    "utc_hour": hour,
                },
            })
    return hits


def _valid_quote(point: Any, target_start: int) -> dict[str, float | int] | None:
    try:
        ts = int(point.timestamp)
        up, down = float(point.up_price), float(point.down_price)
    except (AttributeError, TypeError, ValueError):
        return None
    offset = ts - target_start
    quote_sum = up + down
    if not (0 <= offset <= QUOTE_MAX_OFFSET_MS and 0 < up < 1 and 0 < down < 1):
        return None
    if not (QUOTE_SUM_MIN <= quote_sum <= QUOTE_SUM_MAX):
        return None
    return {
        "timestamp": ts,
        "offset_sec": offset / 1000,
        "up_price": up,
        "down_price": down,
        "quote_sum": quote_sum,
    }


def quote_snapshots(points: list[Any], target_start: int) -> dict[str, Any]:
    """返回首个完整报价及 2/15/30/60/120 秒时点之前的最新完整报价。"""
    valid = [q for point in sorted(points, key=lambda p: int(p.timestamp)) if (q := _valid_quote(point, target_start))]
    first = valid[0] if valid else None
    delays: dict[str, dict[str, Any] | None] = {}
    for delay in QUOTE_DELAYS:
        eligible = [quote for quote in valid if quote["timestamp"] <= target_start + delay * 1000]
        delays[str(delay)] = eligible[-1] if eligible else None
    return {"first": first, "delays": delays}


def apply_quote_labels(timeframe: str, labels: list[str], down_quote: float | None) -> list[str]:
    result = list(labels)
    if (
        timeframe == "15m"
        and "candle_hm_bull_15m_union_shadow_v1" in result
        and down_quote is not None
        and .45 <= down_quote < .55
    ):
        result.append("candle_hm_bull_15m_q45_55_shadow_hyp_v1")
    return list(dict.fromkeys(result))


class CandlestickShadowDetector:
    """轮询已收盘 K 线，单事件落库、分派新鲜逻辑命中并在次根收盘结算。"""

    def __init__(self, collector, pm_15m_latest: dict, pm_5m_info: dict) -> None:
        self._collector = collector
        self._pm_by_tf = {"5m": pm_5m_info, "15m": pm_15m_latest}
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: dict[str, int | None] = {tf: None for tf in TIMEFRAMES}
        self._trigger_count = 0
        self._settle_count = 0
        # 注册实盘通道后由 main 注入；通道默认全 OFF，未注入仍是纯影子。
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._backscan()
        except Exception as exc:
            logger.warning("蜡烛组合影子：冷启动回补失败（循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="candlestick_shadow_detector")
        logger.info("蜡烛组合影子检测器启动 | 18 个冻结逻辑版本 | 单事件多标签（实盘通道默认关闭）")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("蜡烛组合影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        for tf in TIMEFRAMES:
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) >= WARMUP_BARS:
                last_start = int(closed[-1]["open_time"])
                if self._last_evaluated_bar[tf] is None or last_start > self._last_evaluated_bar[tf]:
                    await self._evaluate_new_bars(tf, closed)
                    self._last_evaluated_bar[tf] = last_start
                await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    async def _evaluate_new_bars(self, tf: str, closed: list[dict]) -> None:
        # 冷启动回补只补影子事实，绝不追下真钱单；正常轮询才收集新鲜实盘 payload。
        collect_live = self._last_evaluated_bar[tf] is not None
        if self._last_evaluated_bar[tf] is None:
            n_tail = BACKSCAN_BARS
        else:
            starts = [int(row["open_time"]) for row in closed]
            try:
                first_new = next(i for i, start in enumerate(starts) if start > self._last_evaluated_bar[tf])
            except StopIteration:
                return
            n_tail = min(BACKSCAN_BARS, len(starts) - first_new)
        hits = evaluate_candlestick_patterns(closed, tf, n_tail)
        if not hits:
            return
        live_payloads: list[dict] = []
        async with async_session_factory() as session:
            added = 0
            for hit in hits:
                payloads = live_payloads if collect_live else None
                if await self._record_signal(session, hit, closed[hit["idx"]], payloads):
                    added += 1
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("蜡烛组合影子触发 +{} | {} | 单事件多标签", added, tf)
        self._dispatch_live(live_payloads)

    async def _record_signal(self, session, hit: dict, bar: dict,
                             live_payloads: list[dict] | None = None) -> bool:
        tf = hit["timeframe"]
        event_version = EVENT_VERSION_BY_TF[tf]
        start = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == event_version,
                KlineShadowSignal.signal_bar_start == start,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return False
        # 返回值仅表示物理事件是否可落库；实盘分派独立于影子 gate，避免「下线采集」
        # 意外变成真钱开关。即便全部逻辑版本暂停采集，新鲜命中仍交给 trader 的 enabled 门禁。
        target = start + BAR_MS[tf]
        up_q, down_q, quote_ts = snapshot_entry_quote(self._pm_by_tf.get(tf), target)
        if up_q is not None and down_q is not None and not (QUOTE_SUM_MIN <= up_q + down_q <= QUOTE_SUM_MAX):
            up_q = down_q = quote_ts = None
        live_labels = apply_quote_labels(tf, hit["matched_signal_ids"], down_q)
        if live_payloads is not None and 0 <= int(time.time() * 1000) - target <= CANDLESTICK_LIVE_MAX_LAG_MS:
            live_payloads.extend({
                "version": version,
                "market_start": target,
                "market_end": target + BAR_MS[tf],
                "direction": CANDLESTICK_DIRECTIONS[version],
                "signal_bar_start": start,
            } for version in live_labels)
        labels = _enabled_labels(live_labels)
        snapshot = {
            **hit["snapshot"],
            "matched_signal_ids": labels,
            "all_matched_signal_ids": live_labels,
            "quote_snapshots": {},
        }
        session.add(KlineShadowSignal(
            version=event_version,
            discovery_id="candle_hm_bull_v1",
            condition_text="single event; logical versions in feature_snapshot.matched_signal_ids",
            timeframe=tf,
            signal_bar_start=start,
            signal_bar_end=target,
            direction=hit["direction"],
            target_bar_start=target,
            feature_snapshot=snapshot,
            entry_up_price=up_q,
            entry_down_price=down_q,
            entry_quote_ts=quote_ts,
            status="PENDING",
        ))
        return True

    def _dispatch_live(self, payloads: list[dict]) -> None:
        hook = self._on_live_fire
        if hook is None:
            return
        for payload in payloads:
            try:
                hook(payload)
            except Exception as exc:
                logger.warning("蜡烛组合：实盘开火分派异常（不影响影子采集）| {} | {}", payload["version"], exc)

    async def _load_quote_points(self, tf: str, target_start: int) -> list[Any]:
        async with async_session_factory() as session:
            return list((await session.execute(
                sa_select(PredictionMarketSample)
                .where(
                    PredictionMarketSample.market_period == tf,
                    PredictionMarketSample.timestamp >= target_start,
                    PredictionMarketSample.timestamp <= target_start + QUOTE_MAX_OFFSET_MS,
                )
                .order_by(PredictionMarketSample.timestamp)
            )).scalars().all())

    async def _settle_pending(self, tf: str, closed: list[dict]) -> None:
        by_start = {int(row["open_time"]): row for row in closed}
        if not by_start:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == EVENT_VERSION_BY_TF[tf],
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(sorted(by_start)),
                )
            )).scalars().all()
            if not pendings:
                return
            for signal in pendings:
                target_start = int(signal.target_bar_start)
                bar = by_start[target_start]
                o, c = float(bar["open"]), float(bar["close"])
                snapshots = quote_snapshots(await self._load_quote_points(tf, target_start), target_start)
                feature_snapshot = dict(signal.feature_snapshot or {})
                first = snapshots["first"]
                labels = list(feature_snapshot.get("matched_signal_ids", []))
                all_labels = list(feature_snapshot.get("all_matched_signal_ids", labels))
                enriched = apply_quote_labels(tf, all_labels, first["down_price"] if first else None)
                newly_derived = [label for label in enriched if label not in all_labels]
                feature_snapshot["all_matched_signal_ids"] = enriched
                feature_snapshot["matched_signal_ids"] = list(dict.fromkeys(
                    labels + [label for label in newly_derived if shadow_gate.is_enabled(label)]
                ))
                feature_snapshot["quote_snapshots"] = snapshots
                signal.feature_snapshot = feature_snapshot
                if first:
                    signal.entry_up_price = first["up_price"]
                    signal.entry_down_price = first["down_price"]
                    signal.entry_quote_ts = first["timestamp"]
                if c == o:
                    signal.settle_outcome, signal.win, signal.status = "NOISE", None, "EXPIRED"
                else:
                    up = c > o
                    signal.settle_outcome = "UP" if up else "DOWN"
                    signal.win = signal.settle_outcome == signal.direction
                    signal.status = "SETTLED"
                signal.settle_open, signal.settle_close = o, c
                signal.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
            await session.commit()

    async def _expire_stale_pending(self, tf: str) -> None:
        cutoff = int(time.time() * 1000) - BAR_MS[tf] - PENDING_EXPIRE_MS[tf]
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == EVENT_VERSION_BY_TF[tf],
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for signal in stale:
                signal.status = "EXPIRED"
            await session.commit()

    async def _backscan(self) -> None:
        for tf in TIMEFRAMES:
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                continue
            await self._evaluate_new_bars(tf, closed)
            self._last_evaluated_bar[tf] = int(closed[-1]["open_time"])
            await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "last_evaluated_bar": dict(self._last_evaluated_bar),
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "event_versions": dict(EVENT_VERSION_BY_TF),
            "logical_versions": list(CANDLESTICK_SIGNAL_IDS),
            "record_only": True,
            "live_channels_registered": True,
        }
