"""阶段5前根评分 × 当窗0.30深价候选（影子与实盘均默认关闭）。

冻结口径：上一根已收盘15m及更早的44维HGB评分 >= 0.54282；只在目标窗
前300秒观察有效UP展示价，首次 <=0.30 时记录。实盘通道触价后延迟1秒，
以10 USDT MARKET单买UP，执行价护栏0.32（含贴线弃单）。
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from importlib.resources import files

import joblib
import numpy as np
from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from .shadow_version_gate import shadow_gate

VERSION = "prebar_dn_bull_bid30_15m_v1"
THRESHOLD = 0.54282
BID_LEVEL = 0.30
MAX_T_REL_S = 300
BAR_MS = 900_000
WARMUP_15M = 220
SETTLE_POLL_S = 15.0
MODEL_SHA256 = "9d64d0f40afb9bb8a4ce1b68895f56c959af0280d499b113e206cf2415050d5d"
_MODEL = None


def _model_payload():
    global _MODEL
    if _MODEL is None:
        path = files("binance_predict.services").joinpath("prebar_dn_bull_hgb_v1.joblib")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise RuntimeError("阶段5模型资产哈希不匹配")
        _MODEL = joblib.load(path)
    return _MODEL


def _mean(a: np.ndarray, n: int, end: int | None = None) -> float:
    end = len(a) if end is None else end
    return float(a[end - n:end].mean())


def _rank_pct(a: np.ndarray) -> float:
    """等价 pandas rolling(...).rank(pct=True) 最后一项，含 ties=average。"""
    last = a[-1]
    return float(((a < last).sum() + ((a == last).sum() + 1) / 2) / len(a))


def build_feature_row(bars: list[dict]) -> np.ndarray | None:
    """用已收盘15m根构造冻结44维最后一行；数据不足/非连续则保守返回None。"""
    if len(bars) < WARMUP_15M:
        return None
    ordered = sorted({int(b["open_time"]): b for b in bars}.values(), key=lambda b: b["open_time"])
    ordered = ordered[-WARMUP_15M:]
    ts = np.array([int(b["open_time"]) for b in ordered], dtype=np.int64)
    if len(ordered) < WARMUP_15M or np.any(np.diff(ts) != BAR_MS):
        return None
    o, h, low, c, v = (np.array([float(b[k]) for b in ordered])
                       for k in ("open", "high", "low", "close", "volume"))
    if not all(np.isfinite(a).all() for a in (o, h, low, c, v)):
        return None
    r = h - low
    atr = np.convolve(r, np.ones(20) / 20, mode="valid")
    a = float(atr[-1] + 1e-9)
    values: dict[str, float] = {}
    ma: dict[int, float] = {}
    for n in (5, 10, 20, 60):
        ma[n] = _mean(c, n)
        values[f"c_ma{n}"] = (c[-1] - ma[n]) / a
    values["ma5_gt_ma10"] = (ma[5] - ma[10]) / a
    values["ma10_gt_ma20"] = (ma[10] - ma[20]) / a
    values["ma20_gt_ma60"] = (ma[20] - ma[60]) / a
    values["align"] = sum(np.sign(values[k]) for k in
                          ("ma5_gt_ma10", "ma10_gt_ma20", "ma20_gt_ma60"))
    values["ma5_slope"] = (ma[5] - _mean(c, 5, len(c) - 5)) / a
    values["ma20_slope"] = (ma[20] - _mean(c, 20, len(c) - 5)) / a
    for n in (1, 3, 5, 10, 20):
        values[f"ret{n}_atr"] = (c[-1] - c[-1 - n]) / a
    sign = int(np.sign(c[-1] - o[-1]))
    run = 0
    if sign:
        for oi, ci in zip(o[::-1], c[::-1]):
            if int(np.sign(ci - oi)) != sign:
                break
            run += sign
    values["consec"] = run
    values["body_ratio_t"] = (c[-1] - o[-1]) / (r[-1] + 1e-9)

    hours = ts // 3_600_000
    unique_hours = np.unique(hours)
    hh, hl, hc = [], [], []
    for hour in unique_hours:
        mask = hours == hour
        hh.append(h[mask].max())
        hl.append(low[mask].min())
        hc.append(c[mask][-1])
    hh, hl, hc = np.asarray(hh), np.asarray(hl), np.asarray(hc)
    if len(hc) < 21:
        return None
    h1atr = _mean(hh - hl, 20, len(hc) - 1)
    values["hour_ret_atr"] = (hc[-2] - hc[-3]) / max(h1atr, 1e-9)
    values["fourh_ret_atr"] = (hc[-2] - hc[-6]) / max(h1atr, 1e-9)

    values["range_t_atr"] = r[-1] / a
    values["range5_atr"] = _mean(r, 5) / a
    values["range_expand"] = r[-1] / (_mean(r, 5) + 1e-9)
    values["atr_pct100"] = _rank_pct(atr[-100:])
    values["volr5"] = v[-1] / (_mean(v, 5) + 1e-9)
    values["volr20"] = v[-1] / (_mean(v, 20) + 1e-9)
    values["vol5_slope"] = _mean(v, 5) / (_mean(v, 5, len(v) - 5) + 1e-9)
    values["vol_pct100"] = _rank_pct(v[-100:])
    for n in (20, 50, 100):
        values[f"d_hi{n}"] = (h[-1 - n:-1].max() - c[-1]) / a
        values[f"d_lo{n}"] = (c[-1] - low[-1 - n:-1].min()) / a
    for n in (50, 200):
        lo, hi = low[-n:].min(), h[-n:].max()
        values[f"pos{n}"] = float(np.clip((c[-1] - lo) / (hi - lo + 1e-9), 0, 1))
    values["gap"] = (o[-1] - c[-2]) / a
    values["upwick_t"] = (h[-1] - max(o[-1], c[-1])) / (r[-1] + 1e-9)
    values["dnwick_t"] = (min(o[-1], c[-1]) - low[-1]) / (r[-1] + 1e-9)
    values["close_pos_t"] = (c[-1] - low[-1]) / (r[-1] + 1e-9)
    utc = datetime.fromtimestamp(ts[-1] / 1000, tz=timezone.utc)
    values.update(sess_asia=int(utc.hour < 8), sess_europe=int(8 <= utc.hour < 16),
                  sess_us=int(utc.hour >= 16), tf_15m=1, weekend=int(utc.weekday() >= 5))
    row = np.array([[values[name] for name in _model_payload()["feature_names"]]])
    return row if np.isfinite(row).all() else None


def score_bars(bars: list[dict]) -> float | None:
    row = build_feature_row(bars)
    return None if row is None else float(_model_payload()["model"].predict_proba(row)[0, 1])


class PrebarDnBullDetector:
    def __init__(self, collector) -> None:
        self._collector = collector
        self._scores: dict[int, float | None] = {}
        self._touched: set[int] = set()
        self._live_fired: set[int] = set()
        self._on_live_fire = None
        self._task = None
        self._running = False
        self._trigger_count = 0
        self._settle_count = 0

    async def start(self) -> None:
        if self._running:
            return
        try:
            _model_payload()
        except Exception as exc:
            # 新候选默认双OFF；资产问题不可拖垮现有真钱系统启动。
            logger.error("阶段5检测器模型加载失败，保持停用 | {}", exc)
            return
        self._running = True
        self._task = asyncio.create_task(self._settle_loop(), name="prebar_dn_bull_detector")
        logger.info("阶段5检测器启动 | {} | 影子默认关闭 + 实盘默认关闭", VERSION)

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _score(self, window_start: int) -> float | None:
        if window_start in self._scores:
            return self._scores[window_start]
        bars = await self._collector.fetch_recent_klines("15m", WARMUP_15M)
        prior = [b for b in bars if int(b["open_time"]) + BAR_MS <= window_start]
        score = score_bars(prior[-WARMUP_15M:])
        self._scores[window_start] = score
        if len(self._scores) > 128:
            self._scores = {k: v for k, v in self._scores.items()
                            if k >= window_start - 8 * BAR_MS}
        return score

    async def observe_sample(self, window_start: int, window_end: int, ts_ms: int,
                             up_price: float | None, down_price: float | None) -> None:
        if window_start in self._touched and (
                window_start in self._live_fired or self._on_live_fire is None):
            return
        t_rel = (ts_ms - window_start) / 1000
        # 0/1/None/终局哨兵绝不视作触价；冻结有效期为 [0,300s)。
        if not (0 <= t_rel < MAX_T_REL_S) or up_price is None or not (
                0.02 < float(up_price) <= BID_LEVEL):
            return
        score = await self._score(window_start)
        if score is None or score < THRESHOLD:
            return
        if shadow_gate.is_enabled(VERSION) and window_start not in self._touched:
            async with async_session_factory() as session:
                exists = (await session.execute(sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == VERSION,
                    KlineShadowSignal.signal_bar_start == window_start,
                ))).scalar_one_or_none()
                if exists is None:
                    session.add(KlineShadowSignal(
                        version=VERSION, discovery_id="prebar_dn_bull_bid30",
                        condition_text=(f"prebar_hgb>={THRESHOLD} AND t<300s "
                                        f"AND up_price<={BID_LEVEL}"),
                        timeframe="15m", signal_bar_start=window_start,
                        signal_bar_end=window_end, direction="UP",
                        target_bar_start=window_start,
                        feature_snapshot={
                            "score": round(score, 8), "threshold": THRESHOLD,
                            "touched_t_rel_s": round(t_rel, 1), "bid_level": BID_LEVEL,
                            "entry_quote_kind": "market_snapshot",
                        },
                        entry_up_price=float(up_price),
                        entry_down_price=(float(down_price)
                                          if down_price is not None else None),
                        entry_quote_ts=ts_ms, status="PENDING",
                    ))
                    await session.commit()
                    self._trigger_count += 1
                self._touched.add(window_start)
        if self._on_live_fire and window_start not in self._live_fired:
            # 钩子无返回值，调用即视作派发；通道OFF由MultiLiveTrader自行门禁。
            self._live_fired.add(window_start)
            try:
                self._on_live_fire({
                    "version": VERSION, "market_start": window_start,
                    "market_end": window_end, "direction": "UP",
                    "signal_bar_start": window_start - BAR_MS,
                    "touch_price": float(up_price), "t_rel": t_rel, "score": score,
                })
            except Exception as exc:
                logger.warning("阶段5实盘钩子异常（不影响影子）| {}", exc)

    async def _settle_loop(self) -> None:
        while self._running:
            try:
                await self._settle_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("阶段5结算异常 | {}", exc)
            try:
                await asyncio.sleep(SETTLE_POLL_S)
            except asyncio.CancelledError:
                break

    async def _settle_pending(self) -> None:
        now = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(sa_select(KlineShadowSignal).where(
                KlineShadowSignal.version == VERSION,
                KlineShadowSignal.status == "PENDING",
            ))).scalars().all()
            if not rows:
                return
            oldest = min(int(row.target_bar_start) for row in rows)
            limit = min(288, max(8, (now - oldest) // BAR_MS + 2))
            bars = await self._collector.fetch_recent_klines("15m", limit)
            by_start = {int(bar["open_time"]): bar for bar in bars}
            for row in rows:
                bar = by_start.get(int(row.target_bar_start))
                if bar is None:
                    if now - int(row.target_bar_start) > 10 * BAR_MS:
                        row.status = "EXPIRED"
                    continue
                o, c = float(bar["open"]), float(bar["close"])
                row.settle_open, row.settle_close = o, c
                row.settle_outcome = "UP" if c > o else ("DOWN" if c < o else "NOISE")
                row.win = (c > o) if c != o else None
                row.status = "SETTLED" if c != o else "EXPIRED"
                row.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
            await session.commit()
