"""冻结首触反转规则的前向 record-only 采集器。

主候选（规则冻结，实验期勿改）：
- firsthit_down_btc_recovery_fwd_v1：5m DOWN 首次 q<=0.10，且触发时
  BTC 从对 DOWN 最不利极值回收至少 25%。
- firsthit_down_q10_control_v1：同一 DOWN 首触母事件，不要求 recovery。
- firsthit_up_btc_recovery_mirror_v1：UP 完全镜像，用于方向不对称检验。

实时采样只负责在首次触达时落 PENDING 快照，并异步获取 1/5/10 USDT 的
只读可执行报价（只调用 get_quote，永不调用 place_order）。归档轮询补齐
15/30/60/120 秒路径、rebound/retest、结算和事后长影标签。所有 q10 母事件
都落库；recovery 未通过和数据缺失也保留，避免幸存者偏差。
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy import select as sa_select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import FirstHitForwardEvent, SentimentWindow
from binance_predict.services.prediction_trading import BinancePredictionTrader

SCHEMA_VERSION = "firsthit_reversal_forward_v1"
DOWN_CANDIDATE = "firsthit_down_btc_recovery_fwd_v1"
DOWN_CONTROL = "firsthit_down_q10_control_v1"
UP_MIRROR = "firsthit_up_btc_recovery_mirror_v1"

Q_THRESHOLD = 0.10
RECOVERY_THRESHOLD = 0.25
FEE_RET = 0.98
HORIZONS_SEC = (15, 30, 60, 120)
QUOTE_AMOUNTS_USDT = (1.0, 5.0, 10.0)
POLL_INTERVAL = 60.0
BACKSCAN_WINDOWS = 12


def _series(curve: list | None) -> list[dict]:
    """清洗并按时间升序，保留有限数值。"""
    out: list[dict] = []
    for point in curve or []:
        try:
            ts, value = int(point["t"]), float(point["v"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append({"t": ts, "v": value})
    return sorted(out, key=lambda point: point["t"])


def _at_or_before(points: list[dict], ts: int) -> dict | None:
    found = None
    for point in points:
        if point["t"] > ts:
            break
        found = point
    return found


def _at_or_after(points: list[dict], ts: int) -> dict | None:
    return next((point for point in points if point["t"] >= ts), None)


def _speed(points: list[dict], end_ts: int, lookback_sec: int) -> float | None:
    end = _at_or_before(points, end_ts)
    start = _at_or_before(points, end_ts - lookback_sec * 1000)
    if end is None or start is None or end["t"] == start["t"]:
        return None
    return (end["v"] - start["v"]) / ((end["t"] - start["t"]) / 1000.0)


def _efficiency(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    path = sum(abs(b - a) for a, b in zip(values, values[1:]))
    return abs(values[-1] - values[0]) / path if path else 0.0


def _turns(values: list[float]) -> int:
    signs = [math.copysign(1, d) for d in (b - a for a, b in zip(values, values[1:])) if d]
    return sum(a != b for a, b in zip(signs, signs[1:]))


def _second_adverse_extreme(values: list[float]) -> bool:
    """是否在首次低点反弹后再次创出更不利极值。"""
    if len(values) < 3:
        return False
    running_min = values[0]
    recovered = False
    for value in values[1:]:
        if value > running_min:
            recovered = True
        elif value < running_min:
            if recovered:
                return True
            running_min = value
    return False


def _labels(side: str, recovery_pass: bool) -> dict[str, bool]:
    if side == "DOWN":
        return {DOWN_CONTROL: True, DOWN_CANDIDATE: recovery_pass}
    return {UP_MIRROR: recovery_pass}


def extract_forward_trigger(
    *,
    side: str,
    window_start: int,
    window_end: int,
    btc_open: float | None,
    up_curve: list | None,
    down_curve: list | None,
    btc_curve: list | None,
    max_trigger_ts: int | None = None,
) -> dict | None:
    """提取冻结首次触达事件；只读取 trigger_ts 及之前的数据。

    BTC 缺失不会删除母事件：recovery 保持 None、候选标签为 False，并在
    data_missing 留痕。q=0/负数不是可执行报价，不视为触达。
    """
    if side not in {"UP", "DOWN"}:
        raise ValueError(f"unsupported side: {side}")
    up, down, btc = _series(up_curve), _series(down_curve), _series(btc_curve)
    quote = up if side == "UP" else down
    trigger = next(
        (
            point for point in quote
            if (max_trigger_ts is None or point["t"] <= max_trigger_ts)
            and 0.0 <= point["v"] <= Q_THRESHOLD
        ),
        None,
    )
    if trigger is None:
        return None

    trigger_ts = trigger["t"]
    q_prefix = [point for point in quote if point["t"] <= trigger_ts]
    btc_prefix = [point for point in btc if point["t"] <= trigger_ts]
    other = _at_or_before(down if side == "UP" else up, trigger_ts)
    missing: list[str] = []

    bo = float(btc_open) if btc_open is not None and float(btc_open) > 0 else None
    if bo is None and btc_prefix:
        bo = btc_prefix[0]["v"]
    if bo is None:
        missing.append("btc_open")

    btc_trigger = btc_prefix[-1]["v"] if btc_prefix else None
    if btc_trigger is None:
        missing.append("btc_trigger")

    signed: list[float] = []
    if bo is not None:
        side_sign = 1.0 if side == "UP" else -1.0
        signed = [side_sign * (point["v"] / bo - 1.0) * 10_000.0 for point in btc_prefix]
    if signed:
        signed_ret = signed[-1]
        signed_min = min(signed)
        recovery_bps = signed_ret - signed_min
        recovery_frac = recovery_bps / max(abs(signed_min), 1e-9)
        adverse_index = signed.index(signed_min)
        adverse_btc = btc_prefix[adverse_index]["v"]
    else:
        signed_ret = signed_min = recovery_bps = recovery_frac = adverse_btc = None
        missing.append("btc_recovery")

    recovery_pass = recovery_frac is not None and recovery_frac >= RECOVERY_THRESHOLD
    q_values = [point["v"] for point in q_prefix]
    trigger_features = {
        "q_start": q_values[0],
        "q_min": min(q_values),
        "q_efficiency": _efficiency(q_values),
        "q_turns": _turns(q_values),
        "q_dwell_frac": sum(value <= Q_THRESHOLD for value in q_values) / len(q_values),
        "q_speed_15": _speed(q_prefix, trigger_ts, 15),
        "q_speed_30": _speed(q_prefix, trigger_ts, 30),
        "q_speed_60": _speed(q_prefix, trigger_ts, 60),
        "btc_speed_15_bps": _speed(
            [{"t": p["t"], "v": v} for p, v in zip(btc_prefix, signed)], trigger_ts, 15
        ) if signed else None,
        "btc_speed_30_bps": _speed(
            [{"t": p["t"], "v": v} for p, v in zip(btc_prefix, signed)], trigger_ts, 30
        ) if signed else None,
        "btc_speed_60_bps": _speed(
            [{"t": p["t"], "v": v} for p, v in zip(btc_prefix, signed)], trigger_ts, 60
        ) if signed else None,
        "btc_second_adverse_extreme": _second_adverse_extreme(signed),
        "sample_count": len(q_prefix),
        "btc_sample_count": len(btc_prefix),
    }
    other_q = other["v"] if other else None
    if other_q is None:
        missing.append("opposite_quote")
    if trigger["v"] <= 0:
        missing.append("trigger_q_nonpositive")

    return {
        "schema_version": SCHEMA_VERSION,
        "window_start": int(window_start),
        "window_end": int(window_end),
        "side": side,
        "trigger_ts": trigger_ts,
        "td_sec": int((trigger_ts - int(window_start)) / 1000),
        "trigger_q": trigger["v"],
        "opposite_q": other_q,
        "quote_sum": trigger["v"] + other_q if other_q is not None else None,
        "btc_open": bo,
        "btc_adverse_extreme": adverse_btc,
        "btc_trigger": btc_trigger,
        "btc_signed_return_bps": signed_ret,
        "btc_signed_min_bps": signed_min,
        "btc_recovery_bps": recovery_bps,
        "btc_recovery_frac": recovery_frac,
        "recovery_pass": recovery_pass,
        "labels": _labels(side, recovery_pass),
        "trigger_features": trigger_features,
        "data_missing": sorted(set(missing)),
    }


def _post_features(snapshot: dict, up_curve: list | None, down_curve: list | None,
                   btc_curve: list | None, btc_close: float | None) -> tuple[dict, dict, dict]:
    side, trigger_ts = snapshot["side"], snapshot["trigger_ts"]
    up, down, btc = _series(up_curve), _series(down_curve), _series(btc_curve)
    quote = up if side == "UP" else down
    after = [point for point in quote if point["t"] >= trigger_ts]

    future: dict[str, dict | None] = {}
    for seconds in HORIZONS_SEC:
        target = trigger_ts + seconds * 1000
        side_point = _at_or_after(quote, target)
        other_point = _at_or_after(down if side == "UP" else up, target)
        future[str(seconds)] = None if side_point is None else {
            "ts": side_point["t"],
            "lag_ms": side_point["t"] - target,
            "q": side_point["v"],
            "opposite_q": other_point["v"] if other_point else None,
        }

    values = [point["v"] for point in after]
    running_min = values[0] if values else None
    rebound_ts = retest_ts = None
    for point in after[1:]:
        running_min = min(running_min, point["v"])
        if rebound_ts is None and point["v"] - running_min >= 0.03 and point["v"] <= 0.25:
            rebound_ts = point["t"]
        elif rebound_ts is not None and point["v"] <= Q_THRESHOLD:
            retest_ts = point["t"]
            break

    post = {
        "q_min_after_trigger": min(values) if values else None,
        "q_max_after_trigger": max(values) if values else None,
        "q_efficiency_after_trigger": _efficiency(values),
        "q_turns_after_trigger": _turns(values),
        "rebound_ts": rebound_ts,
        "retest_ts": retest_ts,
        "rebound": rebound_ts is not None,
        "retest": retest_ts is not None,
    }

    # retrospective（2026-10-05 修复 look-ahead）：仅用 ≤trigger_ts 的 BTC 前缀，
    # close 取触发时刻价（snapshot["btc_trigger"]）。旧实现把 window.exit_price
    # （结算价）当 close、hi/lo 用全窗曲线——body_ratio 编码了「结算距开盘多远」，
    # 与 win 机械相关（body<0.2 假胜率 45.5%），绝不可用于门设计。
    bo = snapshot.get("btc_open")
    btc_prefix = [
        point for point in (btc or [])
        if int(point["t"]) <= int(trigger_ts)
    ]
    close = float(btc_close) if btc_close is not None else None
    if bo and close is not None and btc_prefix:
        vals = [float(point["v"]) for point in btc_prefix]
        hi, lo = max([bo, *vals]), min([bo, *vals])
        span = hi - lo
        body_ratio = abs(close - bo) / span if span > 0 else 0.0
        major_wick = (min(bo, close) - lo) if side == "UP" else (hi - max(bo, close))
        wick_ratio = major_wick / span if span > 0 else 0.0
        retrospective = {
            "source": "btc_prefix_le_trigger",
            "body_ratio": body_ratio,
            "major_wick_ratio": wick_ratio,
            "long_wick": wick_ratio >= 0.5 and body_ratio <= 0.5,
        }
    else:
        retrospective = {"source": "btc_prefix_le_trigger", "long_wick": None}
    return future, post, retrospective


def _event_values(snapshot: dict, capture_mode: str) -> dict:
    return {
        **snapshot,
        "capture_mode": capture_mode,
        "atr_bps": None,
        "current_move_atr": None,
        "future_quotes": {},
        "execution_quotes": {},
        "post_features": {},
        "retrospective": {},
        "settle_outcome": None,
        "win": None,
        "ev_at_entry": None,
        "status": "PENDING",
    }


class FirstHitForwardRecorder:
    """实时首次触达 + 归档结算的独立研究采集器；没有任何下单入口。"""

    def __init__(self, *, collector=None, trader=None) -> None:
        self._collector = collector
        # 默认创建独立报价客户端：不复用真钱 trader 的状态或 _trade_lock，避免
        # 1/5/10U 探测阻塞任何真实下单临界区。
        self._trader = trader or BinancePredictionTrader()
        self._owns_trader = trader is None
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._seen: set[tuple[str, int]] = set()
        self._last_window_end: int | None = None
        self._counts = {"DOWN": 0, "UP": 0}
        self._abs_seen = set()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            async with async_session_factory() as session:
                self._last_window_end = (await session.execute(
                    sa_select(sa_func.max(SentimentWindow.end_time))
                )).scalar_one_or_none()
        except Exception as exc:
            logger.warning("首触前向采集：水位预热失败（循环内重试）| {}", exc)
        self._loop_task = asyncio.create_task(self._loop(), name="firsthit_forward_recorder")
        logger.info(
            "首触前向采集启动 | DOWN q≤{:.2f} control + recovery≥{:.0%} candidate + UP mirror"
            " | 1/5/10U只读报价 | record-only 永不下单",
            Q_THRESHOLD, RECOVERY_THRESHOLD,
        )

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
        if self._loop_task is not None:
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        pending = [task for task in self._tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._loop_task = None
        if self._owns_trader:
            await self._trader.aclose()
        logger.info("首触前向采集停止 | {}", self._counts)

    def observe_sample(
        self, *, window_start: int, window_end: int, ts_ms: int,
        btc_open: float | None, up_curve: list | None,
        down_curve: list | None, btc_curve: list | None,
    ) -> None:
        """采样循环非阻塞入口；每侧每窗只派一个持久化任务。"""
        if not self._running:
            return
        from .shadow_forward_evidence import OVERFLOW_COUNTS
        if len(self._tasks) >= 32:
            OVERFLOW_COUNTS["firsthit_recorder"] = OVERFLOW_COUNTS.get("firsthit_recorder", 0) + 1
            return
        try:
            self._observe_forward_sample(window_start=window_start, window_end=window_end, ts_ms=ts_ms,
                btc_open=btc_open, up_curve=up_curve, down_curve=down_curve, btc_curve=btc_curve)
        except Exception as exc:
            OVERFLOW_COUNTS["firsthit_recorder"] = OVERFLOW_COUNTS.get("firsthit_recorder", 0) + 1
            logger.warning("前向采样捕获失败（不影响交易）| {}", exc)

    def _observe_forward_sample(self, *, window_start, window_end, ts_ms, btc_open,
                                up_curve, down_curve, btc_curve):
        self._observe_absorption(window_start, window_end, ts_ms, btc_open, up_curve, down_curve, btc_curve)
        from .firsthit_shadow_detector import extract_firsthit_features, _gate_of
        ext = extract_firsthit_features(window_start, btc_open, down_curve, btc_curve,
                                       max_trigger_ts=ts_ms, pin_path_to_first_touch=True,
                                       pin_btc_to_first_touch=True)
        version = "firsthit_down_chg_v2"
        if ext and _gate_of(version, ext) and (version, window_start) not in self._abs_seen:
            from .shadow_forward_evidence import capture_event
            self._abs_seen.add((version, window_start))
            event = capture_event({"window_start": window_start, "window_end": window_end,
                "trigger_ts": ext["trigger_ts"], "capture_mode": "LIVE" if ext["trigger_ts"] == ts_ms else "RESTORED",
                "side": "DOWN", "trigger_q": ext["q"],
                "version": version, "features": ext}, version)
            task = asyncio.create_task(self._record_absorption(event, True))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        self._seen = {key for key in self._seen if key[1] >= window_start - 300_000}
        for side in ("DOWN", "UP"):
            snapshot = extract_forward_trigger(
                side=side, window_start=window_start, window_end=window_end,
                btc_open=btc_open, up_curve=up_curve, down_curve=down_curve,
                btc_curve=btc_curve, max_trigger_ts=ts_ms,
            )
            key = (side, window_start)
            if snapshot is None or key in self._seen:
                continue
            self._seen.add(key)
            mode = "LIVE" if snapshot["trigger_ts"] == ts_ms else "RESTORED"
            from .shadow_forward_evidence import capture_event
            snapshot["_evidence"] = {version: capture_event(snapshot, version) for version in
                                     ("firsthit_control", "firsthit_recovery")}
            task = asyncio.create_task(
                self._record_trigger(snapshot, mode),
                name=f"firsthit_fwd_{side.lower()}_{window_start}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def _observe_absorption(self, start, end, ts, btc_open, up_curve, down_curve, btc_curve):
        from .absorption_shadow_detector import ABSORPTION_SPECS
        calibrator = getattr(self, "absorption_calibrator", None)
        if calibrator is None:
            return
        from .live_execution_policy import evaluate_absorption_policy
        up, down, btc = _series(up_curve), _series(down_curve), _series(btc_curve)
        if not up or not down or not btc or not btc_open:
            return
        self._abs_seen = {key for key in self._abs_seen if key[1] >= start-300_000}
        for version, td in ABSORPTION_SPECS.items():
            key = (version, start)
            from .multi_live_trader import ABS_LIVE_JUDGE_GRACE_S
            if key in self._abs_seen or not 0 <= ts-(start+td*1000) <= ABS_LIVE_JUDGE_GRACE_S*1000:
                continue
            fitted = calibrator.live_calibration(version)
            snapshot = {"window_start": start, "window_end": end, "trigger_ts": ts,
                        "side": "DOWN", "trigger_q": down[-1]["v"], "version": version,
                        "calibration": fitted, "btc_open": btc_open, "btc_price": btc[-1]["v"],
                        "up_open": up[0]["v"], "up_price": up[-1]["v"]}
            selected = False
            if fitted is not None:
                k, b, disp, under = fitted
                policy = evaluate_absorption_policy(btc_price=btc[-1]["v"], btc_open=btc_open,
                    up_price=up[-1]["v"], up_open=up[0]["v"], k=k, b=b, disp_gate=disp, under_gate=under)
                selected = policy.eligible and policy.prediction == "DOWN"
            if not selected:
                continue  # Failed TD observation must not consume the first legal opportunity.
            self._abs_seen.add(key)
            from .shadow_forward_evidence import capture_event
            snapshot = capture_event(snapshot, version)
            task = asyncio.create_task(self._record_absorption(snapshot, selected))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _record_absorption(self, snapshot, selected):
        from .shadow_forward_evidence import freeze_event
        try:
            async with async_session_factory() as session:
                await freeze_event(session, snapshot, snapshot.get("capture_mode", "LIVE"), {}, snapshot["version"], selected)
                await session.commit()
            from .shadow_forward_evidence import probe_ladder
            execution = (await probe_ladder(snapshot["version"], "DOWN", snapshot["window_start"], "5m",
                         amount=snapshot["_frozen"]["profile"]["amount_usdt"])
                         if snapshot.get("capture_mode", "LIVE") == "LIVE" else {})
            async with async_session_factory() as session:
                from .shadow_forward_evidence import attach_quotes
                await attach_quotes(session, snapshot, snapshot["version"], execution)
                await session.commit()
        except Exception as exc:
            logger.warning("吸收前向冻结失败 | {} | {}", snapshot["version"], exc)

    async def _record_trigger(self, snapshot: dict, capture_mode: str) -> None:
        key = (snapshot["side"], snapshot["window_start"])
        try:
            async with async_session_factory() as session:
                await session.execute(
                    pg_insert(FirstHitForwardEvent)
                    .values(**_event_values({k: v for k, v in snapshot.items() if not k.startswith("_")}, capture_mode))
                    .on_conflict_do_nothing(index_elements=["side", "window_start"])
                )
                await session.commit()

            from .shadow_forward_evidence import freeze_firsthit
            async with async_session_factory() as session:
                await freeze_firsthit(session, snapshot, capture_mode, {})
                await session.commit()
            # 恢复历史中的旧触达不能伪装成实时可执行报价。LIVE 模式下报价阶梯
            # 与 ATR 并发，避免非关键 ATR REST 请求延迟时间敏感的执行价快照。
            if capture_mode == "LIVE":
                atr_bps, execution = await asyncio.gather(
                    self._fetch_atr_bps(snapshot.get("btc_open")),
                    self._probe_execution_quotes(
                        snapshot["side"], snapshot["window_start"], snapshot["trigger_q"]
                    ),
                )
            else:
                atr_bps = await self._fetch_atr_bps(snapshot.get("btc_open"))
                execution = {}
            missing = list(snapshot["data_missing"])
            if atr_bps is None:
                missing.append("atr20")
            if capture_mode != "LIVE" or not any(v.get("available") for v in execution.values()):
                missing.append("execution_quote_ladder")

            async with async_session_factory() as session:
                row = (await session.execute(
                    sa_select(FirstHitForwardEvent).where(
                        FirstHitForwardEvent.side == snapshot["side"],
                        FirstHitForwardEvent.window_start == snapshot["window_start"],
                    )
                )).scalar_one_or_none()
                if row is not None:
                    row.atr_bps = atr_bps
                    row.current_move_atr = (
                        snapshot["btc_signed_return_bps"] / atr_bps
                        if atr_bps and snapshot["btc_signed_return_bps"] is not None else None
                    )
                    # 归档 fallback 已先落表时，不把稍晚到达的报价伪装为触发时可执行；
                    # capture_mode 相等才说明此行由本次实时触达任务创建。
                    if row.capture_mode == capture_mode:
                        row.execution_quotes = execution
                        row.data_missing = sorted(set(missing))
                        from .shadow_forward_evidence import freeze_firsthit
                        await freeze_firsthit(session, snapshot, capture_mode, execution)
                    await session.commit()
            self._counts[snapshot["side"]] += 1
            logger.info(
                "首触前向事件 | {} | {} | q={:.4f} recovery={} pass={} | mode={}",
                snapshot["side"],
                datetime.fromtimestamp(snapshot["window_start"] / 1000, tz=timezone.utc)
                .strftime("%m-%d %H:%M"),
                snapshot["trigger_q"], snapshot["btc_recovery_frac"],
                snapshot["recovery_pass"], capture_mode,
            )
        except Exception as exc:
            self._seen.discard(key)
            logger.warning("首触前向采集：实时事件落库失败 | {} | {}", key, exc)

    async def _fetch_atr_bps(self, btc_open: float | None) -> float | None:
        if self._collector is None or not btc_open:
            return None
        klines = await self._collector.fetch_recent_klines("5m", 21)
        if len(klines) < 21:
            return None
        true_ranges = []
        for prev, current in zip(klines, klines[1:]):
            true_ranges.append(max(
                current["high"] - current["low"],
                abs(current["high"] - prev["close"]),
                abs(current["low"] - prev["close"]),
            ))
        atr = sum(true_ranges[-20:]) / 20.0
        return atr / btc_open * 10_000.0 if atr > 0 else None

    async def _probe_execution_quotes(self, side: str, window_start: int,
                                      trigger_q: float) -> dict[str, dict]:
        """获取报价阶梯；仅 get_quote，不存在 place_order 调用。"""
        from . import shadow_forward_evidence as evidence
        if evidence._quote_client is self._trader:
            return await evidence.probe_ladder("firsthit_down_chg_v2", side, window_start, "5m")
        trader = self._trader
        if trader is None:
            return {}
        # 独立 trader 实例不经实盘链路初始化：_wallet_address 为空时 get_quote
        # 恒被币安 -1102（walletAddress 缺失）拒收。触发时自动补取（只读 API，
        # 每次触发重试自愈；上线 2026-09-17 至 2026-10-04 该缺失致阶梯 100% 失败）。
        if not getattr(trader, "_wallet_address", ""):
            wallet = await trader.fetch_wallet_info()
            if wallet is None or not getattr(trader, "_wallet_address", ""):
                return {"error": {
                    "available": False,
                    "reason": "wallet_address_unavailable",
                }}
        results: dict[str, dict] = {}
        try:
            from .shadow_forward_evidence import QUOTE_QUEUE
            async with QUOTE_QUEUE, trader._trade_lock:
                await trader.list_markets()
                market_start = getattr(trader, "_5m_start_date", None)
                token_id = getattr(trader, "_down_token_id" if side == "DOWN" else "_up_token_id", None)
                if market_start is None or abs(int(market_start) - window_start) > 2_000 or not token_id:
                    return {"error": {"available": False, "reason": "5m_market_mismatch_or_token_missing"}}
                from .live_channels import parse_channel_config
                configs = parse_channel_config()
                current = configs.get("firsthit_down_chg_v2")
                amounts = sorted(set((*QUOTE_AMOUNTS_USDT, current.amount_usdt if current else 1.0)))
                deadline = time.monotonic() + 4
                for amount in amounts:
                    if time.monotonic() >= deadline:
                        results.setdefault("error", {"available": False, "reason": "quote_budget_exhausted"})
                        break
                    requested_at = int(time.time() * 1000)
                    quote = await asyncio.wait_for(trader.get_quote(token_id, "BUY", amount_usdt=amount),
                                                   timeout=max(.01, deadline-time.monotonic()))
                    if not quote:
                        results[str(int(amount)) if amount == int(amount) else str(amount)] = {
                            "available": False,
                            "requested_at": requested_at,
                            "error": getattr(trader, "last_api_error", None),
                        }
                        continue
                    try:
                        avg_price = float(quote.get("averagePrice"))
                    except (TypeError, ValueError):
                        avg_price = None
                    received_at = int(time.time() * 1000)
                    results[str(int(amount)) if amount == int(amount) else str(amount)] = {
                        "available": avg_price is not None and math.isfinite(avg_price) and 0 < avg_price < 1,
                        "requested_at": requested_at,
                        "received_at": received_at,
                        "latency_ms": received_at - requested_at,
                        "amount_usdt": amount,
                        "average_price": avg_price,
                        "amount_out": quote.get("amountOut"),
                        "slippage_bps": (
                            (avg_price / trigger_q - 1.0) * 10_000.0
                            if avg_price is not None and trigger_q > 0 else None
                        ),
                    }
        except Exception as exc:
            logger.warning("首触前向采集：只读报价阶梯失败 | {} {} | {}", side, window_start, exc)
            results.setdefault("error", {"available": False, "reason": str(exc)})
        return results

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("首触前向采集：归档循环异常 | {}", exc)
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
            windows = (await session.execute(stmt)).scalars().all()
        for window in windows:
            try:
                await self._settle_window(window)
            except Exception as exc:
                logger.warning("首触前向采集：窗口结算失败 | {} | {}", window.start_time, exc)
                break  # 保留水位，下轮重试；不得跨过失败窗口。
            self._last_window_end = max(self._last_window_end or 0, int(window.end_time))

    async def _settle_window(self, window: SentimentWindow) -> None:
        outcome = window.outcome if window.outcome in {"UP", "DOWN"} else None
        btc_open = window.entry_price if window.entry_price and window.entry_price > 0 else None
        from .shadow_forward_evidence import settle_firsthit
        async with async_session_factory() as session:
            await settle_firsthit(session, int(window.start_time), outcome)
            await session.commit()
        for side in ("DOWN", "UP"):
            snapshot = extract_forward_trigger(
                side=side, window_start=int(window.start_time), window_end=int(window.end_time),
                btc_open=btc_open, up_curve=window.curve_up_price,
                down_curve=window.curve_down_price, btc_curve=window.curve_btc_price,
            )
            if snapshot is None:
                continue
            future, post, retrospective = _post_features(
                snapshot, window.curve_up_price, window.curve_down_price,
                window.curve_btc_price, snapshot.get("btc_trigger"),
            )
            async with async_session_factory() as session:
                # 单语句 upsert 关闭实时触达任务与归档轮询的竞态：实时行存在则
                # 仅补结算字段；不存在则无选择写 ARCHIVE_FALLBACK 母事件。
                await _upsert_settled_event(
                    session, snapshot, outcome=outcome, future=future,
                    post=post, retrospective=retrospective,
                )
                from .shadow_forward_evidence import settle_firsthit
                await settle_firsthit(session, int(window.start_time), outcome)
                await session.commit()

    def status(self) -> dict:
        return {
            "running": self._running,
            "schema_version": SCHEMA_VERSION,
            "rules": {
                "q_max": Q_THRESHOLD,
                "recovery_min": RECOVERY_THRESHOLD,
                "down_candidate": DOWN_CANDIDATE,
                "down_control": DOWN_CONTROL,
                "up_mirror": UP_MIRROR,
            },
            "events": dict(self._counts),
            "last_window_end": self._last_window_end,
        }


async def _upsert_settled_event(session, snapshot: dict, *, outcome: str | None,
                                future: dict, post: dict, retrospective: dict) -> None:
    """结算 upsert；独立函数便于用无真实 DB 的测试锁定幂等与失败单口径。"""
    win = None if outcome is None else outcome == snapshot["side"]
    fallback = _event_values(snapshot, "ARCHIVE_FALLBACK")
    fallback.update(
        data_missing=sorted(set([*snapshot["data_missing"], "execution_quote_ladder"])),
        future_quotes=future,
        post_features=post,
        retrospective=retrospective,
        settle_outcome=outcome,
        win=win,
        ev_at_entry=(
            None if win is None or snapshot["trigger_q"] <= 0 else
            (FEE_RET / snapshot["trigger_q"] - 1.0 if win else -1.0)
        ),
        status="SETTLED" if outcome is not None else "UNSETTLED",
    )
    await session.execute(
        pg_insert(FirstHitForwardEvent).values(**fallback)
        .on_conflict_do_update(
            index_elements=["side", "window_start"],
            set_={
                "future_quotes": future,
                "post_features": post,
                "retrospective": retrospective,
                "settle_outcome": outcome,
                "win": win,
                "ev_at_entry": fallback["ev_at_entry"],
                "status": fallback["status"],
            },
        )
    )
