#!/usr/bin/env python3
"""预注册的 BTC 长影反转 / 趋势延续研究流水线。

只读本地 720d K 线与真实预测市场报价；不调用生产接口，不改交易代码。
阶段分离：baseline 只写 discovery/validation 与数据审计；final 读取冻结注册表后才触碰 holdout。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DAY_MS = 86_400_000
FEE_PAYOUT = 0.98
SEED = 20260917
BAR_MS = {"5m": 300_000, "15m": 900_000}
HORIZONS = range(1, 6)
SEGMENTS = ("discovery", "validation", "holdout")


@dataclass
class MarketData:
    tf: str
    ms: int
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    direction: np.ndarray
    body: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    range_atr: np.ndarray
    rel_volume: np.ndarray
    prior_ret3: np.ndarray
    prior_ret5: np.ndarray
    prior_ret10: np.ndarray
    prior_dir3: np.ndarray
    prior_dir5: np.ndarray
    prior_dir10: np.ndarray
    prior_bull5: np.ndarray
    prior_ma10: np.ndarray
    streak: np.ndarray
    breakout_high20: np.ndarray
    breakout_low20: np.ndarray
    failed_breakout_high20: np.ndarray
    failed_breakout_low20: np.ndarray
    close_loc: np.ndarray
    splits: tuple[int, int]


def shift(x: np.ndarray, k: int = 1, fill: float = np.nan) -> np.ndarray:
    out = np.full(len(x), fill, dtype=float)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(x, nan=0.0))]
    cnt = np.r_[0, np.cumsum(np.isfinite(x))]
    sums = cs[w:] - cs[:-w]
    counts = cnt[w:] - cnt[:-w]
    out[w - 1 :] = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts == w)
    return out


def rolling_max(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1 :] = np.lib.stride_tricks.sliding_window_view(x, w).max(axis=1)
    return out


def rolling_min(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1 :] = np.lib.stride_tricks.sliding_window_view(x, w).min(axis=1)
    return out


def load_market(path: Path, tf: str) -> MarketData:
    rows: list[list[str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["timestamp", "open", "high", "low", "close", "volume"]:
            raise ValueError(f"unexpected kline header: {path}")
        rows.extend(reader)
    t = np.array([int(datetime.fromisoformat(r[0]).timestamp() * 1000) for r in rows], np.int64)
    o, h, l, c, v = (np.array([float(r[j]) for r in rows]) for j in range(1, 6))
    ms = BAR_MS[tf]
    if len(set(t.tolist())) != len(t) or not np.all(np.diff(t) == ms):
        raise ValueError(f"{tf} klines must be unique and continuous")
    rng = h - l
    safe = np.where(rng > 0, rng, np.nan)
    direction = np.sign(c - o).astype(np.int8)
    body = np.abs(c - o) / safe
    upper = (h - np.maximum(o, c)) / safe
    lower = (np.minimum(o, c) - l) / safe
    range_pct = rng / o
    atr20 = shift(rolling_mean(range_pct, 20))
    rel_volume = v / shift(rolling_mean(v, 20))

    def prior_ret(k: int) -> np.ndarray:
        return shift(c) / shift(o, k) - 1

    def prior_dir(k: int) -> np.ndarray:
        s = rolling_mean(np.sign(c - o), k)
        return np.nan_to_num(np.sign(shift(s)), nan=0).astype(np.int8)

    bull = (direction > 0).astype(float)
    prior_bull5 = shift(rolling_mean(bull, 5) * 5)
    prior_ma10 = shift(c) / shift(rolling_mean(c, 10)) - 1
    streak = np.ones(len(t), dtype=np.int16)
    for i in range(1, len(t)):
        streak[i] = streak[i - 1] + 1 if direction[i] != 0 and direction[i] == direction[i - 1] else 1
    prior_high20 = rolling_max(shift(h), 20)
    prior_low20 = rolling_min(shift(l), 20)
    close_loc = (c - l) / safe
    i1, i2 = int(len(t) * 0.6), int(len(t) * 0.8)
    return MarketData(
        tf, ms, t, o, h, l, c, v, direction, body, upper, lower,
        range_pct / atr20, rel_volume, prior_ret(3), prior_ret(5), prior_ret(10),
        prior_dir(3), prior_dir(5), prior_dir(10), prior_bull5, prior_ma10, streak,
        h > prior_high20, l < prior_low20, (h > prior_high20) & (c < prior_high20),
        (l < prior_low20) & (c > prior_low20), close_loc, (i1, i2),
    )


def segment_mask(n: int, splits: tuple[int, int], name: str) -> np.ndarray:
    i1, i2 = splits
    mask = np.zeros(n, dtype=bool)
    lo, hi = {"discovery": (0, i1), "validation": (i1, i2), "holdout": (i2, n)}[name]
    mask[lo:hi] = True
    return mask


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, center - half), min(1.0, center + half)


def bh_qvalues(pvalues: list[float]) -> list[float]:
    out = [math.nan] * len(pvalues)
    valid = [(i, p) for i, p in enumerate(pvalues) if math.isfinite(p)]
    if not valid:
        return out
    valid.sort(key=lambda x: x[1])
    m = len(valid)
    running = 1.0
    for rank in range(m, 0, -1):
        idx, p = valid[rank - 1]
        running = min(running, p * m / rank)
        out[idx] = running
    return out


def normal_pvalue(k: int, n: int, p0: float) -> float:
    if not n or not 0 < p0 < 1:
        return math.nan
    z = (k - n * p0 - 0.5) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def longest_loss_streak(x: np.ndarray) -> int:
    run = best = 0
    for value in x:
        run = run + 1 if value < 0 else 0
        best = max(best, run)
    return best


def pnl_stats(x: np.ndarray) -> dict[str, float | int]:
    if not len(x):
        return {"ev": math.nan, "cum": 0.0, "max_drawdown": 0.0, "max_loss_streak": 0}
    curve = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, curve])
    dd = peak - np.r_[0.0, curve]
    return {
        "ev": float(x.mean()), "cum": float(x.sum()), "max_drawdown": float(dd.max()),
        "max_loss_streak": longest_loss_streak(x),
    }


def clustered_ci(values: np.ndarray, clusters: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    keys = np.unique(clusters)
    if len(keys) < 3:
        return math.nan, math.nan
    groups = [values[clusters == key] for key in keys]
    rng = np.random.default_rng(seed)
    sims = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        sims[i] = np.concatenate([groups[j] for j in pick]).mean()
    return tuple(float(x) for x in np.percentile(sims, [2.5, 97.5]))


def episode_ids(mask: np.ndarray) -> np.ndarray:
    ids = np.full(len(mask), -1, dtype=np.int64)
    active = np.flatnonzero(mask)
    if not len(active):
        return ids
    eid = 0
    ids[active[0]] = eid
    for prev, cur in zip(active, active[1:]):
        if cur != prev + 1:
            eid += 1
        ids[cur] = eid
    return ids


def reversal_conditions(d: MarketData) -> list[dict]:
    """有限、机制先验的反转族；方向由假设明确给出，不从结果挑选。"""
    shapes = {
        "upper": (d.upper >= 0.6) & (d.body <= 0.3) & (d.lower <= 0.15),
        "lower": (d.lower >= 0.6) & (d.body <= 0.3) & (d.upper <= 0.15),
    }
    out: list[dict] = []
    # H1：影线一侧本身代表该侧价格被拒绝。
    out.extend([
        {"family": "reversal", "condition": "upper_wick_rejection", "mask": shapes["upper"], "expected": -1},
        {"family": "reversal", "condition": "lower_wick_rejection", "mask": shapes["lower"], "expected": 1},
    ])
    # H2：真正的“反转”方向由前置趋势决定；影线侧只负责区分形态。
    for side, shape in shapes.items():
        for trend, trend_mask, expected in (
            ("prior_up", d.prior_ret3 > 0, -1),
            ("prior_down", d.prior_ret3 < 0, 1),
        ):
            out.append({"family": "reversal", "condition": f"{side}_{trend}_reversal", "mask": shape & trend_mask, "expected": expected})
            # 颜色是预注册的归因维度，不把两个颜色事后混成最优规则。
            for color, color_mask in (("bull", d.direction > 0), ("bear", d.direction < 0)):
                out.append({"family": "reversal", "condition": f"{side}_{trend}_{color}_reversal", "mask": shape & trend_mask & color_mask, "expected": expected})
    # H3：扫过 20 根极值后收回，比单纯长影更接近流动性扫单/假突破机制。
    out.extend([
        {"family": "reversal", "condition": "upper_failed_break20", "mask": shapes["upper"] & d.failed_breakout_high20, "expected": -1},
        {"family": "reversal", "condition": "lower_failed_break20", "mask": shapes["lower"] & d.failed_breakout_low20, "expected": 1},
        {"family": "reversal", "condition": "upper_prior_up_failed_break20", "mask": shapes["upper"] & (d.prior_ret3 > 0) & d.failed_breakout_high20, "expected": -1},
        {"family": "reversal", "condition": "lower_prior_down_failed_break20", "mask": shapes["lower"] & (d.prior_ret3 < 0) & d.failed_breakout_low20, "expected": 1},
    ])
    # 已上线 record-only 家族只作污染基准：定义曾看过全历史，不能当全新 holdout 证据。
    trend_votes = (d.prior_ret3 > 0).astype(np.int8)
    trend_votes += ((d.prior_ret5 > 0) & (d.prior_bull5 >= 3)).astype(np.int8)
    trend_votes += (d.prior_ma10 > 0).astype(np.int8)
    legacy_shape = (d.direction > 0) & (d.lower >= 0.5) & (d.body <= 0.5) & (d.upper <= 0.1)
    thresholds = (2, 3) if d.tf == "5m" else (1, 3)
    labels = ("consensus2", "consensus3") if d.tf == "5m" else ("union", "consensus3")
    for need, label in zip(thresholds, labels):
        out.append({"family": "reversal", "condition": f"legacy_{d.tf}_{label}", "mask": legacy_shape & (trend_votes >= need), "expected": -1})
    return out


def continuation_conditions(d: MarketData) -> list[dict]:
    strong = (d.body >= 0.6) & (d.range_atr >= 1.0) & (d.rel_volume >= 1.0)
    out: list[dict] = []
    for expected, label in ((1, "up"), (-1, "down")):
        same = d.direction == expected
        breakout = d.breakout_high20 if expected > 0 else d.breakout_low20
        close_extreme = d.close_loc >= 0.8 if expected > 0 else d.close_loc <= 0.2
        prior3 = d.prior_ret3 * expected > 0
        candidates = [
            ("streak2", same & (d.streak == 2)),
            ("streak3", same & (d.streak == 3)),
            ("strong_impulse", same & strong & close_extreme),
            ("breakout20", same & breakout & close_extreme),
            ("breakout20_volume", same & breakout & close_extreme & (d.rel_volume >= 1.25)),
            ("momentum3_impulse", same & prior3 & strong & close_extreme),
        ]
        for name, mask in candidates:
            out.append({"family": "continuation", "condition": f"{label}_{name}", "mask": mask, "expected": expected})
    return out


def outcome_arrays(d: MarketData, expected: int, hz: int) -> tuple[np.ndarray, np.ndarray]:
    future_dir = np.zeros(len(d.t), np.int8)
    future_dir[:-hz] = d.direction[hz:]
    valid = np.zeros(len(d.t), bool)
    valid[:-hz] = future_dir[:-hz] != 0
    win = future_dir == expected
    return valid, win


def condition_stats(d: MarketData, spec: dict, segment: str) -> list[dict]:
    seg = segment_mask(len(d.t), d.splits, segment)
    rows = []
    base_cache: dict[tuple[int, int], float] = {}
    for hz in HORIZONS:
        valid, win = outcome_arrays(d, spec["expected"], hz)
        mask = spec["mask"] & seg & valid
        n = int(mask.sum())
        k = int(win[mask].sum())
        base_key = (spec["expected"], hz)
        if base_key not in base_cache:
            base_mask = seg & valid
            base_cache[base_key] = float(win[base_mask].mean())
        base = base_cache[base_key]
        lo, hi = wilson(k, n)
        ids = episode_ids(mask)
        if n:
            observed = win[mask].astype(float)
            selected_ids = ids[mask]
            ep_values = np.array([observed[selected_ids == eid].mean() for eid in np.unique(selected_ids)], float)
            ep_lo, ep_hi = clustered_ci(ep_values, np.arange(len(ep_values)), SEED + hz)
        else:
            ep_lo = ep_hi = math.nan
        rows.append({
            "timeframe": d.tf, "family": spec["family"], "condition": spec["condition"],
            "expected_direction": "UP" if spec["expected"] > 0 else "DOWN", "segment": segment,
            "horizon": hz, "n": n, "episodes": int(ids.max() + 1) if n else 0,
            "wins": k, "win_rate": k / n if n else math.nan, "wilson_low": lo, "wilson_high": hi,
            "episode_win_ci_low": ep_lo, "episode_win_ci_high": ep_hi,
            "unconditional_rate": base, "lift_pp": (k / n - base) * 100 if n else math.nan,
            "p_one_sided": normal_pvalue(k, n, base),
        })
    return rows


def path_rows(d: MarketData, spec: dict, segment: str) -> list[dict]:
    seg = segment_mask(len(d.t), d.splits, segment)
    indices = np.flatnonzero(spec["mask"] & seg)
    rows = []
    for i in indices:
        if i + 5 >= len(d.t):
            continue
        future = d.direction[i + 1 : i + 6]
        if np.any(future == 0):
            continue
        expected = spec["expected"]
        first_expected = next((j + 1 for j, x in enumerate(future) if x == expected), 0)
        first_opposite = next((j + 1 for j, x in enumerate(future) if x == -expected), 0)
        record = {
            "timeframe": d.tf, "family": spec["family"], "condition": spec["condition"],
            "expected_direction": "UP" if expected > 0 else "DOWN", "segment": segment,
            "signal_index": int(i), "signal_open_ms": int(d.t[i]),
            "signal_open_utc": datetime.fromtimestamp(d.t[i] / 1000, timezone.utc).isoformat(),
            "signal_hour_utc": int((d.t[i] // 3_600_000) % 24),
            "volatility_regime": "HIGH" if d.range_atr[i] >= 1.25 else "LOW" if d.range_atr[i] < 0.75 else "MID",
            "volume_regime": "HIGH" if d.rel_volume[i] >= 1.25 else "LOW" if d.rel_volume[i] < 0.75 else "MID",
            "first_expected_bar": first_expected, "first_opposite_bar": first_opposite,
            "path": "".join("U" if x > 0 else "D" for x in future),
            "body_ratio": d.body[i], "upper_wick_ratio": d.upper[i], "lower_wick_ratio": d.lower[i],
            "range_atr": d.range_atr[i], "rel_volume": d.rel_volume[i], "prior_ret3": d.prior_ret3[i],
            "failed_breakout": bool(d.failed_breakout_high20[i] or d.failed_breakout_low20[i]),
        }
        for hz, value in enumerate(future, 1):
            record[f"bar_{hz}_win"] = bool(value == expected)
        rows.append(record)
    return rows


def load_quotes(path: Path, ms: int) -> dict[int, list[tuple]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    windows: dict[int, list[tuple]] = defaultdict(list)
    for r in raw:
        try:
            ts, up, down = int(r["timestamp"]), float(r["up_price"]), float(r["down_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < up < 1 and 0 < down < 1:
            windows[ts // ms * ms].append((ts, up, down))
    for points in windows.values():
        points.sort()
    return windows


def quote_for(points: list[tuple], start: int, direction: int, delay_s: int) -> tuple | None:
    eligible = [p for p in points if delay_s * 1000 <= p[0] - start <= min(delay_s * 1000 + 20_000, 120_000) and 0.95 <= p[1] + p[2] <= 1.05]
    if not eligible:
        return None
    p = eligible[0]
    return p[0], (p[1] if direction > 0 else p[2])


def quote_evaluation(d: MarketData, spec: dict, segment: str, quotes: dict[int, list[tuple]]) -> tuple[list[dict], list[dict]]:
    seg = segment_mask(len(d.t), d.splits, segment)
    indices = np.flatnonzero(spec["mask"] & seg)
    events: list[dict] = []
    summaries: list[dict] = []
    ep = episode_ids(spec["mask"] & seg)
    for delay in (0, 15, 30, 60, 120):
        delay_events = []
        for i in indices:
            if i + 1 >= len(d.t) or not d.direction[i + 1]:
                continue
            start = int(d.t[i] + d.ms)
            quote = quote_for(quotes.get(start, []), start, spec["expected"], delay)
            if not quote:
                continue
            q_ts, q = quote
            win = bool(d.direction[i + 1] == spec["expected"])
            ret = FEE_PAYOUT / q - 1 if win else -1.0
            point = next(p for p in quotes[start] if p[0] == q_ts)
            row = {
                "timeframe": d.tf, "family": spec["family"], "condition": spec["condition"],
                "segment": segment, "delay_sec": delay, "signal_open_ms": int(d.t[i]),
                "target_window_start": start, "quote_ts": int(q_ts), "quote": q,
                "up_quote": float(point[1]), "down_quote": float(point[2]), "quote_sum": float(point[1] + point[2]),
                "expected_direction": "UP" if spec["expected"] > 0 else "DOWN", "win": win,
                "return_r": ret, "episode_id": int(ep[i]), "utc_day": int(d.t[i] // DAY_MS),
                "signal_hour_utc": int((d.t[i] // 3_600_000) % 24),
                "volatility_regime": "HIGH" if d.range_atr[i] >= 1.25 else "LOW" if d.range_atr[i] < 0.75 else "MID",
                "volume_regime": "HIGH" if d.rel_volume[i] >= 1.25 else "LOW" if d.rel_volume[i] < 0.75 else "MID",
            }
            events.append(row)
            delay_events.append(row)
        values = np.array([r["return_r"] for r in delay_events], float)
        days = np.array([r["utc_day"] for r in delay_events], np.int64)
        episodes = np.array([r["episode_id"] for r in delay_events], np.int64)
        stats = pnl_stats(values)
        day_lo, day_hi = clustered_ci(values, days, SEED + delay)
        ep_lo, ep_hi = clustered_ci(values, episodes, SEED + 100 + delay)
        summaries.append({
            "timeframe": d.tf, "family": spec["family"], "condition": spec["condition"],
            "segment": segment, "delay_sec": delay, "signals": len(indices), "quote_n": len(delay_events),
            "independent_episodes": len(set(episodes.tolist())) if len(episodes) else 0,
            "weeks": len(set((days // 7).tolist())) if len(days) else 0,
            "win_rate": float(np.mean([r["win"] for r in delay_events])) if delay_events else math.nan,
            "mean_quote": float(np.mean([r["quote"] for r in delay_events])) if delay_events else math.nan,
            "ev_per_trade": stats["ev"], "cumulative_r": stats["cum"],
            "max_drawdown_r": stats["max_drawdown"], "max_loss_streak": stats["max_loss_streak"],
            "day_ci_low": day_lo, "day_ci_high": day_hi, "episode_ci_low": ep_lo, "episode_ci_high": ep_hi,
        })
    return events, summaries


def quote_threshold_rows(events: list[dict]) -> list[dict]:
    rows = []
    keys = sorted({(r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) for r in events})
    for key in keys:
        group = [r for r in events if (r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) == key]
        for cap in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.95):
            selected = [r for r in group if r["quote"] < cap]
            values = np.array([r["return_r"] for r in selected], float)
            clusters = np.array([r["episode_id"] for r in selected], np.int64)
            lo, hi = clustered_ci(values, clusters, SEED + int(cap * 100))
            stats = pnl_stats(values)
            rows.append({
                "timeframe": key[0], "family": key[1], "condition": key[2], "segment": key[3],
                "delay_sec": key[4], "max_quote_exclusive": cap, "n": len(selected),
                "episodes": len(set(clusters.tolist())) if len(clusters) else 0,
                "ev_per_trade": stats["ev"], "episode_ci_low": lo, "episode_ci_high": hi,
                "cumulative_r": stats["cum"], "max_drawdown_r": stats["max_drawdown"],
            })
    return rows


def episode_summary_rows(events: list[dict]) -> list[dict]:
    """每个触发 episode 等额逐根下注的相关风险口径。"""
    rows = []
    keys = sorted({(r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) for r in events})
    for key in keys:
        group = [r for r in events if (r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) == key]
        totals: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        starts: dict[int, int] = {}
        for r in group:
            eid = int(r["episode_id"])
            totals[eid] += float(r["return_r"])
            counts[eid] += 1
            starts[eid] = min(starts.get(eid, int(r["signal_open_ms"])), int(r["signal_open_ms"]))
        values = np.array([totals[eid] for eid in sorted(totals)], float)
        days = np.array([starts[eid] // DAY_MS for eid in sorted(totals)], np.int64)
        lo, hi = clustered_ci(values, days, SEED + 700 + int(key[4]))
        stats = pnl_stats(values)
        rows.append({
            "timeframe": key[0], "family": key[1], "condition": key[2], "segment": key[3], "delay_sec": key[4],
            "episodes": len(values), "bets": len(group), "mean_bets_per_episode": len(group) / len(values) if len(values) else math.nan,
            "ev_per_episode_r": stats["ev"], "cumulative_r": stats["cum"], "max_episode_drawdown_r": stats["max_drawdown"],
            "day_cluster_ci_low": lo, "day_cluster_ci_high": hi,
        })
    return rows


def monthly_regime_rows(events: list[dict]) -> list[dict]:
    rows = []
    for key in sorted({(r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) for r in events}):
        group = [r for r in events if (r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) == key]
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in group:
            month = datetime.fromtimestamp(r["signal_open_ms"] / 1000, timezone.utc).strftime("%Y-%m")
            buckets[month].append(r)
        total = sum(float(r["return_r"]) for r in group)
        for month, selected in sorted(buckets.items()):
            values = np.array([r["return_r"] for r in selected], float)
            rows.append({
                "timeframe": key[0], "family": key[1], "condition": key[2], "segment": key[3], "delay_sec": key[4],
                "month": month, "n": len(selected), "win_rate": float(np.mean([r["win"] for r in selected])),
                "ev_per_trade": float(values.mean()), "cumulative_r": float(values.sum()),
                "share_of_total_positive_pnl": float(values.sum() / total) if total > 0 and values.sum() > 0 else 0.0,
            })
    return rows


def regime_rows(events: list[dict]) -> list[dict]:
    rows = []
    dimensions = {
        "volatility": lambda r: r["volatility_regime"],
        "volume": lambda r: r["volume_regime"],
        "session": lambda r: "ASIA" if int(r["signal_hour_utc"]) < 8 else "EUROPE" if int(r["signal_hour_utc"]) < 16 else "US",
    }
    base_keys = sorted({(r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) for r in events})
    for key in base_keys:
        group = [r for r in events if (r["timeframe"], r["family"], r["condition"], r["segment"], r["delay_sec"]) == key]
        for dimension, getter in dimensions.items():
            buckets: dict[str, list[dict]] = defaultdict(list)
            for r in group:
                buckets[getter(r)].append(r)
            for bucket, selected in sorted(buckets.items()):
                values = np.array([r["return_r"] for r in selected], float)
                clusters = np.array([r["episode_id"] for r in selected], np.int64)
                lo, hi = clustered_ci(values, clusters, SEED + 900 + len(rows))
                rows.append({
                    "timeframe": key[0], "family": key[1], "condition": key[2], "segment": key[3], "delay_sec": key[4],
                    "dimension": dimension, "bucket": bucket, "n": len(selected),
                    "win_rate": float(np.mean([r["win"] for r in selected])), "ev_per_trade": float(values.mean()),
                    "episode_ci_low": lo, "episode_ci_high": hi,
                })
    return rows


def run_length_rows(d: MarketData, segment: str) -> list[dict]:
    seg = segment_mask(len(d.t), d.splits, segment)
    rows = []
    for direction, label in ((1, "UP"), (-1, "DOWN")):
        for length in (1, 2, 3, 4, 5):
            mask = seg & (d.direction == direction) & (d.streak == length)
            n = int(mask[:-1].sum())
            cont = int(((d.direction[1:] == direction) & mask[:-1]).sum())
            lo, hi = wilson(cont, n)
            rows.append({
                "timeframe": d.tf, "segment": segment, "direction": label, "observed_run_length": length,
                "n": n, "continued_next": cont, "survival_probability": cont / n if n else math.nan,
                "wilson_low": lo, "wilson_high": hi,
            })
    return rows


def data_audit(root: Path, markets: dict[str, MarketData], quote_sets: dict[str, dict]) -> dict:
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "klines": {}, "quotes": {}}
    for tf, d in markets.items():
        p = root / f"output/klines_{tf}_720d.csv"
        result["klines"][tf] = {
            "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(d.t),
            "start_utc": datetime.fromtimestamp(d.t[0] / 1000, timezone.utc).isoformat(),
            "end_open_utc": datetime.fromtimestamp(d.t[-1] / 1000, timezone.utc).isoformat(),
            "duplicates": int(len(d.t) - len(set(d.t.tolist()))), "gaps": int(np.sum(np.diff(d.t) != d.ms)),
            "discovery_end_open_utc": datetime.fromtimestamp(d.t[d.splits[0] - 1] / 1000, timezone.utc).isoformat(),
            "validation_end_open_utc": datetime.fromtimestamp(d.t[d.splits[1] - 1] / 1000, timezone.utc).isoformat(),
            "settlement": "target window close versus target window open; direction equals target K-line color",
            "holdout_status": "research holdout relative to this pre-registration only; existing legacy candlestick rules are historically contaminated",
        }
        qp = root / f"output/online_{tf}_samples_full_merged.json"
        q = quote_sets[tf]
        samples = sum(len(x) for x in q.values())
        starts = sorted(q)
        result["quotes"][tf] = {
            "path": str(qp), "sha256": hashlib.sha256(qp.read_bytes()).hexdigest(), "samples": samples,
            "windows": len(q), "first_window_utc": datetime.fromtimestamp(starts[0] / 1000, timezone.utc).isoformat(),
            "last_window_utc": datetime.fromtimestamp(starts[-1] / 1000, timezone.utc).isoformat(),
            "duplicate_timestamps": samples - len({p[0] for points in q.values() for p in points}),
            "entry_rule": "first valid quote in [delay, delay+20s], capped at 120s; sum guard 0.95..1.05",
            "availability_caveat": "quote history overlaps only the final K-line split, so EV evidence has no separate pre-holdout quote segment; treat as observational unless followed by new forward shadow data",
            "missing_policy": "drop; never interpolate",
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_registry(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("holdout_opened_utc"):
        raise RuntimeError("holdout already opened for this registry")
    candidates = payload.get("frozen_candidates") or []
    if not candidates:
        raise ValueError("registry has no frozen candidates")
    return candidates


def all_specs(d: MarketData) -> list[dict]:
    return reversal_conditions(d) + continuation_conditions(d)


def select_spec(d: MarketData, family: str, condition: str) -> dict:
    return next(s for s in all_specs(d) if s["family"] == family and s["condition"] == condition)


def baseline(root: Path, out: Path) -> None:
    markets = {tf: load_market(root / f"output/klines_{tf}_720d.csv", tf) for tf in BAR_MS}
    quotes = {tf: load_quotes(root / f"output/online_{tf}_samples_full_merged.json", BAR_MS[tf]) for tf in BAR_MS}
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(data_audit(root, markets, quotes), ensure_ascii=False, indent=2), encoding="utf-8")
    hypothesis_rows, paths, runs = [], [], []
    for d in markets.values():
        for spec in all_specs(d):
            for segment in ("discovery", "validation"):
                hypothesis_rows.extend(condition_stats(d, spec, segment))
                paths.extend(path_rows(d, spec, segment))
        for segment in ("discovery", "validation"):
            runs.extend(run_length_rows(d, segment))
    for segment in ("discovery", "validation"):
        idx = [i for i, r in enumerate(hypothesis_rows) if r["segment"] == segment]
        qvals = bh_qvalues([hypothesis_rows[i]["p_one_sided"] for i in idx])
        for i, q in zip(idx, qvals):
            hypothesis_rows[i]["fdr_q"] = q
    write_csv(out / "hypothesis_results_pre_holdout.csv", hypothesis_rows)
    write_csv(out / "path_events_pre_holdout.csv", paths)
    write_csv(out / "run_length_survival_pre_holdout.csv", runs)
    manifest = {
        "stage": "baseline", "holdout_opened": False, "rows": {"hypotheses": len(hypothesis_rows), "paths": len(paths), "runs": len(runs)},
        "next": "freeze candidates in candidate_registry.json, then run --stage final exactly once",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def final(root: Path, out: Path, registry_path: Path) -> None:
    candidates = read_registry(registry_path)
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload["holdout_opened_utc"] = datetime.now(timezone.utc).isoformat()
    registry_payload["holdout_data_hashes"] = {
        tf: hashlib.sha256((root / f"output/klines_{tf}_720d.csv").read_bytes()).hexdigest() for tf in BAR_MS
    }
    registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markets = {tf: load_market(root / f"output/klines_{tf}_720d.csv", tf) for tf in BAR_MS}
    quotes = {tf: load_quotes(root / f"output/online_{tf}_samples_full_merged.json", BAR_MS[tf]) for tf in BAR_MS}
    stats, paths, quote_events, quote_summary = [], [], [], []
    for frozen in candidates:
        d = markets[frozen["timeframe"]]
        spec = select_spec(d, frozen["family"], frozen["condition"])
        stats.extend(condition_stats(d, spec, "holdout"))
        paths.extend(path_rows(d, spec, "holdout"))
        for segment in ("validation", "holdout"):
            ev, sm = quote_evaluation(d, spec, segment, quotes[d.tf])
            quote_events.extend(ev)
            quote_summary.extend(sm)
    qvals = bh_qvalues([r["p_one_sided"] for r in stats])
    for row, q in zip(stats, qvals):
        row["fdr_q"] = q
    write_csv(out / "holdout_results.csv", stats)
    write_csv(out / "path_events_holdout.csv", paths)
    write_csv(out / "quote_events.csv", quote_events)
    write_csv(out / "quote_summary.csv", quote_summary)
    write_csv(out / "quote_threshold_curve.csv", quote_threshold_rows(quote_events))
    write_csv(out / "episode_summary.csv", episode_summary_rows(quote_events))
    write_csv(out / "monthly_regimes.csv", monthly_regime_rows(quote_events))
    write_csv(out / "regime_slices.csv", regime_rows(quote_events))
    manifest = {
        "stage": "final", "holdout_opened": True, "candidate_count": len(candidates),
        "rows": {"holdout": len(stats), "paths": len(paths), "quote_events": len(quote_events), "quote_summary": len(quote_summary)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def self_check() -> None:
    assert bh_qvalues([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
    ids = episode_ids(np.array([False, True, True, False, True, False]))
    assert ids.tolist() == [-1, 0, 0, -1, 1, -1]
    lo, hi = wilson(60, 100)
    assert lo < 0.6 < hi
    assert abs((FEE_PAYOUT / 0.5 - 1) - 0.96) < 1e-12
    points = [(1_000, 0.4, 0.6), (16_000, 0.5, 0.5)]
    assert quote_for(points, 0, 1, 0) == (1_000, 0.4)
    assert quote_for(points, 0, -1, 15) == (16_000, 0.5)
    print("self-check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("output/reversal_continuation_research_20260917"))
    parser.add_argument("--stage", choices=("baseline", "final", "self-check"), required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    if args.stage == "self-check":
        self_check()
    elif args.stage == "baseline":
        baseline(args.root, args.out)
    else:
        if not args.registry:
            parser.error("--registry is required for final")
        final(args.root, args.out, args.registry)


if __name__ == "__main__":
    main()
