#!/usr/bin/env python3
"""V2：预测窗口内极低报价后的情绪反转研究。

长影只作事后机制标签。所有入场特征均截断到事件时刻；入场价格就是事件点真实 token 报价。
baseline 只评估 discovery/validation；final 必须读取事先冻结的 registry，并只允许打开一次 holdout。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEE_PAYOUT = 0.98
SEED = 20260918
DAY_MS = 86_400_000
BAR_MS = {"5m": 300_000, "15m": 900_000}
THRESHOLDS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
D2_THRESHOLDS = (0.05, 0.10, 0.20)
TRIGGERS = ("first_hit", "rebound", "retest")
SIDES = (("UP", 1), ("DOWN", -1))

EVENT_COLUMNS = [
    "timeframe", "window_start", "window_utc", "segment", "side", "side_sign", "outcome", "win",
    "threshold", "trigger", "event_ts", "event_offset_sec", "progress", "q", "up_q", "down_q", "quote_sum",
    "first_hit_offset_sec", "seconds_since_first_hit", "q_start", "q_min", "q_rebound", "q_net", "q_speed",
    "q_last_speed", "q_accel", "q_efficiency", "q_turns", "q_cross50", "q_dwell_frac", "retest_count",
    "btc_available", "btc_signed_ret_bps", "btc_signed_min_bps", "btc_recovery_bps", "btc_recovery_frac",
    "btc_last_speed_bps", "btc_q_divergence", "current_move_atr", "prior_signed_ret1_atr", "prior_signed_ret3_atr",
    "prior_signed_ret5_atr", "prior_side_streak", "prior_compression_5_20", "prior_volatility_regime",
    "prior_1h_signed_atr", "prior_4h_signed_atr", "participants_growth", "trade_volume_growth",
    "final_major_wick_ratio", "final_body_ratio", "retrospective_long_wick", "utc_day", "utc_month",
]

ATOM_FORMULAS = {
    "base": "True",
    "early": "progress <= 0.33",
    "middle": "0.33 < progress <= 0.67",
    "late": "progress > 0.67",
    "fast_orderly_panic": "first_hit_offset/progress <= 0.33 and q_efficiency >= 0.70",
    "panic_decelerating": "q_accel > 0",
    "choppy_extreme": "q_turns >= 2",
    "btc_recovery": "btc_available and btc_recovery_frac >= 0.25",
    "btc_deep_recovery": "btc_available and current/worst move <= -0.50 ATR and btc_recovery_frac >= 0.25",
    "price_sentiment_divergence": "btc_available and btc_recovery_bps >= 3 while q_rebound <= 0.02",
    "prior_exhaustion": "prior 3-bar signed move <= -0.50 ATR",
    "prior_with_side": "prior 3-bar signed move > 0",
    "low_volatility": "prior ATR20 below its trailing-median proxy regime",
    "high_volatility": "prior ATR20 high regime",
    "clean_quote_pair": "0.98 <= UP+DOWN <= 1.02",
    "participant_stall": "participants growth <= 0.10 from window start",
    "volume_stall": "trade-volume growth <= 0.25 from window start",
}


@dataclass
class KlineContext:
    t: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    index: dict[int, int]
    atr20: np.ndarray
    compression: np.ndarray
    vol_regime: np.ndarray


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        cs = np.r_[0.0, np.cumsum(x)]
        out[w - 1 :] = (cs[w:] - cs[:-w]) / w
    return out


def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1 :] = np.median(np.lib.stride_tricks.sliding_window_view(x, w), axis=1)
    return out


def load_klines(path: Path, tf: str) -> KlineContext:
    df = pd.read_csv(path)
    t = pd.to_datetime(df["timestamp"], utc=True).to_numpy(dtype="datetime64[ms]").astype(np.int64)
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    ms = BAR_MS[tf]
    if len(np.unique(t)) != len(t) or not np.all(np.diff(t) == ms):
        values, counts = np.unique(np.diff(t), return_counts=True)
        raise ValueError(f"{tf} K-line timestamps are not unique/continuous: dtype={t.dtype}, diffs={list(zip(values[:5].tolist(), counts[:5].tolist()))}, expected={ms}")
    prev_c = shift(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr20 = shift(rolling_mean(np.nan_to_num(tr, nan=0.0), 20))
    range_pct = (h - l) / o
    rm5, rm20 = shift(rolling_mean(range_pct, 5)), shift(rolling_mean(range_pct, 20))
    compression = rm5 / rm20
    med100 = shift(rolling_median(np.nan_to_num(atr20 / o, nan=0.0), 100))
    ratio = (atr20 / o) / med100
    regime = np.where(ratio >= 1.25, "HIGH", np.where(ratio <= 0.75, "LOW", "MID"))
    return KlineContext(t, o, h, l, c, v, {int(x): i for i, x in enumerate(t)}, atr20, compression, regime)


def split_boundaries(starts: list[int]) -> tuple[int, int]:
    unique = np.array(sorted(set(starts)), np.int64)
    return int(unique[int(len(unique) * 0.5)]), int(unique[int(len(unique) * 0.75)])


def segment_of(ts: int, split1: int, split2: int) -> str:
    return "discovery" if ts < split1 else "validation" if ts < split2 else "holdout"


def _turns(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    signs = np.sign(np.diff(values))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def _crosses(values: np.ndarray, level: float = 0.5) -> int:
    if len(values) < 2:
        return 0
    signs = np.sign(values - level)
    return int(np.sum(signs[1:] * signs[:-1] < 0))


def _efficiency(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    path = float(np.abs(np.diff(values)).sum())
    return abs(float(values[-1] - values[0])) / path if path else 0.0


def _speed(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 2 or times[-1] == times[0]:
        return 0.0
    return float(values[-1] - values[0]) / float((times[-1] - times[0]) / 1000)


def _last_speed(values: np.ndarray, times: np.ndarray, back: int = 1) -> float:
    if len(values) <= back or times[-1] == times[-1 - back]:
        return 0.0
    return float(values[-1] - values[-1 - back]) / float((times[-1] - times[-1 - back]) / 1000)


def _growth(first, last) -> float:
    try:
        a, b = float(first), float(last)
        return (b - a) / max(abs(a), 1.0)
    except (TypeError, ValueError):
        return math.nan


def event_indices(q: np.ndarray, threshold: float) -> dict[str, int]:
    hits = np.flatnonzero(q <= threshold)
    if not len(hits):
        return {}
    first = int(hits[0])
    result = {"first_hit": first}
    running_min = q[first]
    rebound_idx = None
    for j in range(first + 1, len(q)):
        running_min = min(running_min, q[j])
        if q[j] - running_min >= 0.03 and q[j] <= 0.25:
            rebound_idx = j
            result["rebound"] = j
            break
    if rebound_idx is not None:
        for j in range(rebound_idx + 1, len(q)):
            if q[j] <= threshold:
                result["retest"] = j
                break
    return result


def context_features(k: KlineContext, i: int, side_sign: int) -> dict:
    atr = k.atr20[i] if i < len(k.atr20) else math.nan
    denom = atr if math.isfinite(atr) and atr > 0 else math.nan
    prev_close = k.c[i - 1] if i >= 1 else math.nan

    def signed_ret(bars: int) -> float:
        if i < bars or not math.isfinite(denom):
            return math.nan
        return side_sign * (prev_close - k.o[i - bars]) / denom

    streak = 0
    for j in range(i - 1, -1, -1):
        direction = np.sign(k.c[j] - k.o[j])
        if direction != side_sign:
            break
        streak += 1
    return {
        "atr": atr,
        "prior_signed_ret1_atr": signed_ret(1),
        "prior_signed_ret3_atr": signed_ret(3),
        "prior_signed_ret5_atr": signed_ret(5),
        "prior_side_streak": streak,
        "prior_compression_5_20": float(k.compression[i]) if math.isfinite(k.compression[i]) else math.nan,
        "prior_volatility_regime": str(k.vol_regime[i]),
        "prior_1h_signed_atr": signed_ret(max(1, 3_600_000 // (k.t[1] - k.t[0]))),
        "prior_4h_signed_atr": signed_ret(max(1, 14_400_000 // (k.t[1] - k.t[0]))),
    }


def make_event(tf: str, start: int, points: list[dict], side: str, side_sign: int, threshold: float,
               trigger: str, idx: int, first_idx: int, outcome: str, segment: str,
               k: KlineContext, ki: int) -> tuple:
    ms = BAR_MS[tf]
    subset = points[: idx + 1]
    ts = np.array([int(p["timestamp"]) for p in subset], np.int64)
    qkey = "up_price" if side == "UP" else "down_price"
    q = np.array([float(p[qkey]) for p in subset], float)
    up = float(subset[-1]["up_price"])
    down = float(subset[-1]["down_price"])
    event_ts = int(ts[-1])
    offset = (event_ts - start) / 1000
    first_offset = (int(points[first_idx]["timestamp"]) - start) / 1000
    first_q = float(points[first_idx][qkey])
    qmin = float(q.min())
    q_last_speed = _last_speed(q, ts)
    q_prev_speed = _last_speed(q[:-1], ts[:-1]) if len(q) >= 3 else 0.0
    dwell = float(np.mean(q <= threshold))
    retests = int(np.sum((q[1:] <= threshold) & (q[:-1] > threshold))) if len(q) > 1 else 0

    btc_values, btc_times = [], []
    for p in subset:
        try:
            value = float(p["btc_price"])
            if math.isfinite(value):
                btc_values.append(value)
                btc_times.append(int(p["timestamp"]))
        except (TypeError, ValueError):
            pass
    btc_available = len(btc_values) >= 2 and k.o[ki] > 0
    if btc_available:
        b = np.array(btc_values, float)
        bt = np.array(btc_times, np.int64)
        signed = side_sign * (b / k.o[ki] - 1) * 10_000
        signed_ret = float(signed[-1])
        signed_min = float(signed.min())
        recovery = signed_ret - signed_min
        recovery_frac = recovery / max(abs(signed_min), 1e-9)
        btc_last_speed = _last_speed(signed, bt)
    else:
        signed_ret = signed_min = recovery = recovery_frac = btc_last_speed = math.nan

    ctx = context_features(k, ki, side_sign)
    atr_bps = ctx["atr"] / k.o[ki] * 10_000 if math.isfinite(ctx["atr"]) and ctx["atr"] > 0 else math.nan
    current_move_atr = signed_ret / atr_bps if btc_available and atr_bps > 0 else math.nan
    q_rebound = float(q[-1] - qmin)
    divergence = recovery - max(q_rebound, 0) * 100

    rng = k.h[ki] - k.l[ki]
    body = abs(k.c[ki] - k.o[ki]) / rng if rng > 0 else math.nan
    major_wick = ((min(k.o[ki], k.c[ki]) - k.l[ki]) if side == "UP" else (k.h[ki] - max(k.o[ki], k.c[ki]))) / rng if rng > 0 else math.nan
    long_wick = bool(major_wick >= 0.5 and body <= 0.5)

    row = {
        "timeframe": tf, "window_start": start,
        "window_utc": datetime.fromtimestamp(start / 1000, timezone.utc).isoformat(), "segment": segment,
        "side": side, "side_sign": side_sign, "outcome": outcome, "win": outcome == side,
        "threshold": threshold, "trigger": trigger, "event_ts": event_ts, "event_offset_sec": offset,
        "progress": offset / (ms / 1000), "q": float(q[-1]), "up_q": up, "down_q": down, "quote_sum": up + down,
        "first_hit_offset_sec": first_offset, "seconds_since_first_hit": offset - first_offset,
        "q_start": float(q[0]), "q_min": qmin, "q_rebound": q_rebound, "q_net": float(q[-1] - q[0]),
        "q_speed": _speed(q, ts), "q_last_speed": q_last_speed, "q_accel": q_last_speed - q_prev_speed,
        "q_efficiency": _efficiency(q), "q_turns": _turns(q), "q_cross50": _crosses(q),
        "q_dwell_frac": dwell, "retest_count": retests, "btc_available": btc_available,
        "btc_signed_ret_bps": signed_ret, "btc_signed_min_bps": signed_min, "btc_recovery_bps": recovery,
        "btc_recovery_frac": recovery_frac, "btc_last_speed_bps": btc_last_speed,
        "btc_q_divergence": divergence, "current_move_atr": current_move_atr,
        **{x: ctx[x] for x in (
            "prior_signed_ret1_atr", "prior_signed_ret3_atr", "prior_signed_ret5_atr", "prior_side_streak",
            "prior_compression_5_20", "prior_volatility_regime", "prior_1h_signed_atr", "prior_4h_signed_atr")},
        "participants_growth": _growth(subset[0].get("participants"), subset[-1].get("participants")),
        "trade_volume_growth": _growth(subset[0].get("trade_volume"), subset[-1].get("trade_volume")),
        "final_major_wick_ratio": major_wick, "final_body_ratio": body, "retrospective_long_wick": long_wick,
        "utc_day": start // DAY_MS, "utc_month": datetime.fromtimestamp(start / 1000, timezone.utc).strftime("%Y-%m"),
    }
    return tuple(row[col] for col in EVENT_COLUMNS)


def build_events(root: Path, tf: str) -> tuple[pd.DataFrame, dict]:
    raw_path = root / f"output/online_{tf}_samples_full_merged.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    kpath = root / f"output/klines_{tf}_720d.csv"
    k = load_klines(kpath, tf)
    ms = BAR_MS[tf]
    windows: dict[int, list[dict]] = {}
    for point in raw:
        try:
            ts = int(point["timestamp"])
            up, down = float(point["up_price"]), float(point["down_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 < up < 1 and 0 < down < 1 and 0.95 <= up + down <= 1.05:
            windows.setdefault(ts // ms * ms, []).append(point)
    for pts in windows.values():
        pts.sort(key=lambda x: int(x["timestamp"]))
    split1, split2 = split_boundaries(list(windows))
    records: list[tuple] = []
    skipped = {"missing_kline": 0, "noise": 0, "short": 0}
    for start, points in sorted(windows.items()):
        ki = k.index.get(start)
        if ki is None:
            skipped["missing_kline"] += 1
            continue
        direction = np.sign(k.c[ki] - k.o[ki])
        if direction == 0:
            skipped["noise"] += 1
            continue
        if len(points) < 2:
            skipped["short"] += 1
            continue
        outcome = "UP" if direction > 0 else "DOWN"
        segment = segment_of(start, split1, split2)
        for side, side_sign in SIDES:
            qkey = "up_price" if side == "UP" else "down_price"
            q = np.array([float(p[qkey]) for p in points], float)
            for threshold in THRESHOLDS:
                idxs = event_indices(q, threshold)
                if not idxs:
                    continue
                first_idx = idxs["first_hit"]
                for trigger, idx in idxs.items():
                    records.append(make_event(tf, start, points, side, side_sign, threshold, trigger, idx,
                                              first_idx, outcome, segment, k, ki))
    df = pd.DataFrame.from_records(records, columns=EVENT_COLUMNS)
    audit = {
        "timeframe": tf, "bar_ms": ms, "quote_path": str(raw_path), "quote_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "kline_path": str(kpath), "kline_sha256": hashlib.sha256(kpath.read_bytes()).hexdigest(),
        "valid_windows": len(windows), "events": len(df), "split1_utc": datetime.fromtimestamp(split1 / 1000, timezone.utc).isoformat(),
        "split2_utc": datetime.fromtimestamp(split2 / 1000, timezone.utc).isoformat(), "skipped": skipped,
        "feature_clock": "all path features use points[:event_index+1]; final wick fields are retrospective labels only",
    }
    return df, audit


def atom_mask(df: pd.DataFrame, atom: str) -> pd.Series:
    true = pd.Series(True, index=df.index)
    if atom == "base": return true
    if atom == "early": return df.progress <= 0.33
    if atom == "middle": return (df.progress > 0.33) & (df.progress <= 0.67)
    if atom == "late": return df.progress > 0.67
    if atom == "fast_orderly_panic": return (df.first_hit_offset_sec <= (df.timeframe.map(BAR_MS) / 1000) * 0.33) & (df.q_efficiency >= 0.70)
    if atom == "panic_decelerating": return df.q_accel > 0
    if atom == "choppy_extreme": return df.q_turns >= 2
    if atom == "btc_recovery": return df.btc_available.astype(bool) & (df.btc_recovery_frac >= 0.25)
    if atom == "btc_deep_recovery": return df.btc_available.astype(bool) & (df.current_move_atr <= -0.50) & (df.btc_recovery_frac >= 0.25)
    if atom == "price_sentiment_divergence": return df.btc_available.astype(bool) & (df.btc_recovery_bps >= 3) & (df.q_rebound <= 0.02)
    if atom == "prior_exhaustion": return df.prior_signed_ret3_atr <= -0.50
    if atom == "prior_with_side": return df.prior_signed_ret3_atr > 0
    if atom == "low_volatility": return df.prior_volatility_regime == "LOW"
    if atom == "high_volatility": return df.prior_volatility_regime == "HIGH"
    if atom == "clean_quote_pair": return df.quote_sum.between(0.98, 1.02)
    if atom == "participant_stall": return df.participants_growth <= 0.10
    if atom == "volume_stall": return df.trade_volume_growth <= 0.25
    raise KeyError(atom)


def candidate_mask(df: pd.DataFrame, candidate: dict) -> pd.Series:
    mask = (df.timeframe == candidate["timeframe"]) & np.isclose(df.threshold, candidate["threshold"]) & (df.trigger == candidate["trigger"])
    if candidate.get("side", "BOTH") != "BOTH":
        mask &= df.side == candidate["side"]
    for atom in candidate.get("atoms", ["base"]):
        mask &= atom_mask(df, atom)
    return mask


def bootstrap_ci(values: np.ndarray, clusters: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    if not len(values): return math.nan, math.nan
    keys = np.unique(clusters)
    if len(keys) < 3: return math.nan, math.nan
    groups = [values[clusters == key] for key in keys]
    rng = np.random.default_rng(seed)
    sims = np.empty(draws)
    for i in range(draws):
        pick = rng.integers(0, len(groups), len(groups))
        sims[i] = np.concatenate([groups[j] for j in pick]).mean()
    return tuple(float(x) for x in np.percentile(sims, [2.5, 97.5]))


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n: return math.nan, math.nan
    z = 1.96; p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return c-h, c+h


def stats(group: pd.DataFrame, seed: int = SEED) -> dict:
    n = len(group)
    if not n:
        return {"n": 0, "wins": 0, "win_rate": math.nan, "wilson_low": math.nan, "wilson_high": math.nan,
                "mean_q": math.nan, "break_even_rate": math.nan, "calibration_residual_pp": math.nan,
                "ev_per_trade": math.nan, "day_ci_low": math.nan, "day_ci_high": math.nan,
                "window_ci_low": math.nan, "window_ci_high": math.nan, "weeks": 0, "months": 0,
                "retrospective_long_wick_rate": math.nan, "p_edge": math.nan}
    wins = group.win.astype(float).to_numpy()
    q = group.q.to_numpy(float)
    p0 = np.clip(q / FEE_PAYOUT, 1e-6, 1 - 1e-6)
    returns = np.where(wins > 0, FEE_PAYOUT / q - 1, -1.0)
    lo, hi = wilson(int(wins.sum()), n)
    z = float(np.sum(wins - p0) / math.sqrt(max(np.sum(p0 * (1-p0)), 1e-12)))
    p_edge = 0.5 * math.erfc(z / math.sqrt(2))
    day_lo, day_hi = bootstrap_ci(returns, group.utc_day.to_numpy(np.int64), seed)
    win_lo, win_hi = bootstrap_ci(returns, group.window_start.to_numpy(np.int64), seed + 1)
    return {
        "n": n, "wins": int(wins.sum()), "win_rate": float(wins.mean()), "wilson_low": lo, "wilson_high": hi,
        "mean_q": float(q.mean()), "break_even_rate": float(p0.mean()),
        "calibration_residual_pp": float((wins.mean() - p0.mean()) * 100), "ev_per_trade": float(returns.mean()),
        "day_ci_low": day_lo, "day_ci_high": day_hi, "window_ci_low": win_lo, "window_ci_high": win_hi,
        "weeks": int(group.utc_day.floordiv(7).nunique()), "months": int(group.utc_month.nunique()),
        "retrospective_long_wick_rate": float(group.retrospective_long_wick.mean()), "p_edge": p_edge,
    }


def bh_qvalues(values: list[float]) -> list[float]:
    out = [math.nan] * len(values)
    valid = sorted(((i, p) for i, p in enumerate(values) if math.isfinite(p)), key=lambda x: x[1])
    running = 1.0; m = len(valid)
    for rank in range(m, 0, -1):
        idx, p = valid[rank-1]; running = min(running, p*m/rank); out[idx] = running
    return out


def descriptive_map(df: pd.DataFrame, segments=("discovery", "validation")) -> list[dict]:
    rows = []
    z = df[df.segment.isin(segments)].copy()
    z["time_bin"] = pd.cut(z.progress, [-1, .2, .4, .6, .8, 2], labels=["0_20", "20_40", "40_60", "60_80", "80_100"])
    for key, group in z.groupby(["timeframe", "segment", "threshold", "trigger", "side", "time_bin"], observed=True):
        rows.append(dict(zip(("timeframe", "segment", "threshold", "trigger", "side", "time_bin"), key)) | stats(group))
    return rows


def d2_candidates() -> list[dict]:
    atoms = list(ATOM_FORMULAS)
    rows = []
    for tf in BAR_MS:
        for threshold in D2_THRESHOLDS:
            for trigger in TRIGGERS:
                for side in ("BOTH", "UP", "DOWN"):
                    for atom in atoms:
                        rows.append({
                            "hypothesis_id": f"R|{tf}|q{threshold:.2f}|{trigger}|{side}|{atom}",
                            "timeframe": tf, "threshold": threshold, "trigger": trigger, "side": side,
                            "atoms": [atom], "mechanism": ATOM_FORMULAS[atom],
                        })
    return rows


def evaluate_candidates(df: pd.DataFrame, candidates: list[dict], segments=("discovery", "validation")) -> list[dict]:
    rows = []
    cols = ["win", "q", "utc_day", "window_start", "utc_month", "retrospective_long_wick"]
    for candidate in candidates:
        base = candidate_mask(df, candidate)
        for segment in segments:
            group = df.loc[base & (df.segment == segment), cols]
            rows.append({k: v for k, v in candidate.items() if k != "atoms"} | {"atoms": "+".join(candidate["atoms"]), "segment": segment} | stats(group, SEED + len(rows)))
    for segment in segments:
        idx = [i for i, row in enumerate(rows) if row["segment"] == segment]
        qs = bh_qvalues([rows[i]["p_edge"] for i in idx])
        for i, q in zip(idx, qs): rows[i]["fdr_q"] = q
    return rows


def combo_candidates(d2_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(d2_rows)
    dev = df[(df.segment == "discovery") & (df.side == "BOTH") & (df.n >= 80) & (df.ev_per_trade > 0) & (df.fdr_q <= 0.10) & (df.atoms != "base")]
    val = df[df.segment == "validation"][["hypothesis_id", "ev_per_trade", "n"]].rename(columns={"ev_per_trade":"val_ev", "n":"val_n"})
    surv = dev.merge(val, on="hypothesis_id")
    surv = surv[(surv.val_n >= 40) & (surv.val_ev > 0)]
    candidates = []
    for key, group in surv.groupby(["timeframe", "threshold", "trigger"]):
        atoms = list(group.sort_values("day_ci_low", ascending=False).atoms.head(6))
        for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                candidates.append({
                    "hypothesis_id": f"R2|{key[0]}|q{key[1]:.2f}|{key[2]}|BOTH|{atoms[i]}+{atoms[j]}",
                    "timeframe": key[0], "threshold": float(key[1]), "trigger": key[2], "side": "BOTH",
                    "atoms": [atoms[i], atoms[j]], "mechanism": f"{ATOM_FORMULAS[atoms[i]]} AND {ATOM_FORMULAS[atoms[j]]}",
                })
    return candidates[:120]


def feature_dictionary() -> list[dict]:
    retrospective = {"final_major_wick_ratio", "final_body_ratio", "retrospective_long_wick", "outcome", "win"}
    return [{
        "feature": name,
        "availability": "retrospective_label_only" if name in retrospective else "available_at_event",
        "prohibited_for_live_rule": name in retrospective,
    } for name in EVENT_COLUMNS]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def self_check() -> None:
    q = np.array([.5,.3,.1,.06,.11,.07])
    assert event_indices(q,.08) == {"first_hit":3,"rebound":4,"retest":5}
    assert _turns(np.array([.5,.4,.3,.35,.2])) == 2
    assert abs(_efficiency(np.array([.5,.4,.3])) - 1) < 1e-12
    test = pd.DataFrame({"progress":[.2], "timeframe":["5m"], "first_hit_offset_sec":[30], "q_efficiency":[.8],
                         "q_accel":[1], "q_turns":[2], "btc_available":[True], "btc_recovery_frac":[.5],
                         "current_move_atr":[-.6], "btc_recovery_bps":[4], "q_rebound":[.01],
                         "prior_signed_ret3_atr":[-.6], "prior_volatility_regime":["LOW"], "quote_sum":[1.0],
                         "participants_growth":[0], "trade_volume_growth":[0]})
    assert all(bool(atom_mask(test, atom).iloc[0]) for atom in ("early","fast_orderly_panic","panic_decelerating","choppy_extreme","btc_recovery","btc_deep_recovery","price_sentiment_divergence","prior_exhaustion","low_volatility","clean_quote_pair","participant_stall","volume_stall"))
    print("self-check: PASS")


def baseline(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "intrawindow_event_band_preholdout.csv.gz"
    audit_path = out / "data_audit.json"
    if cache.exists() and audit_path.exists():
        pre = pd.read_csv(cache)
        audits = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        frames, audits = [], []
        for tf in BAR_MS:
            df, audit = build_events(root, tf); frames.append(df); audits.append(audit)
        events = pd.concat(frames, ignore_index=True)
        pre = events[events.segment != "holdout"].copy()
        pre.to_csv(cache, index=False, compression="gzip")
        write_json(audit_path, audits)
    write_json(out / "feature_dictionary.json", feature_dictionary())
    dmap_path = out / "descriptive_map_preholdout.csv"
    if not dmap_path.exists():
        pd.DataFrame(descriptive_map(pre)).to_csv(dmap_path, index=False)
    d2_defs = d2_candidates()
    d2 = evaluate_candidates(pre, d2_defs)
    pd.DataFrame(d2).to_csv(out / "single_factor_preholdout.csv", index=False)
    combos = combo_candidates(d2)
    combo_rows = evaluate_candidates(pre, combos) if combos else []
    pd.DataFrame(combo_rows).to_csv(out / "combo_preholdout.csv", index=False)
    registry = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "holdout_opened_utc": None,
        "data_hashes": {a["timeframe"]: {"quote":a["quote_sha256"],"kline":a["kline_sha256"]} for a in audits},
        "single_factor_definitions": d2_defs, "combo_definitions": combos,
        "frozen_candidates": [],
    }
    write_json(out / "hypothesis_registry.json", registry)
    manifest = {"stage":"baseline","holdout_opened":False,"events_preholdout":len(pre),"d2_tests":len(d2_defs),"combo_tests":len(combos)}
    write_json(out / "manifest.json", manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2))


def final(root: Path, out: Path, registry_path: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("holdout_opened_utc"):
        raise RuntimeError("holdout already opened")
    frozen = registry.get("frozen_candidates") or []
    if not frozen: raise ValueError("freeze candidates before final")
    registry["holdout_opened_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(registry_path, registry)
    frames = [build_events(root, tf)[0] for tf in BAR_MS]
    events = pd.concat(frames, ignore_index=True)
    holdout = events[events.segment == "holdout"].copy()
    results = evaluate_candidates(holdout, frozen, segments=("holdout",))
    pd.DataFrame(results).to_csv(out / "intrawindow_holdout_results.csv", index=False)
    selected = []
    for candidate in frozen:
        group = holdout[candidate_mask(holdout, candidate)].copy()
        group["hypothesis_id"] = candidate["hypothesis_id"]
        selected.append(group)
    selected_df = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    selected_df.to_csv(out / "intrawindow_holdout_events.csv.gz", index=False, compression="gzip")
    # Regime robustness uses only frozen candidates, never feeds back into selection.
    regime_rows = []
    if len(selected_df):
        selected_df["time_bin"] = pd.cut(selected_df.progress, [-1,.33,.67,2], labels=["EARLY","MIDDLE","LATE"])
        for key, group in selected_df.groupby(["hypothesis_id","side","time_bin","prior_volatility_regime","utc_month"], observed=True):
            regime_rows.append(dict(zip(("hypothesis_id","side","time_bin","volatility","month"),key)) | stats(group, SEED+len(regime_rows)))
    pd.DataFrame(regime_rows).to_csv(out / "intrawindow_holdout_regimes.csv", index=False)
    manifest = {"stage":"final","holdout_opened":True,"frozen":len(frozen),"holdout_events":len(selected_df),"result_rows":len(results)}
    write_json(out / "manifest.json", manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(".")); ap.add_argument("--out",type=Path,default=Path("output/intrawindow_reversal_v2_20260918")); ap.add_argument("--stage",choices=("self-check","baseline","final"),required=True); ap.add_argument("--registry",type=Path)
    args = ap.parse_args()
    if args.stage == "self-check": self_check()
    elif args.stage == "baseline": baseline(args.root,args.out)
    else:
        if not args.registry: ap.error("--registry required")
        final(args.root,args.out,args.registry)


if __name__ == "__main__": main()
