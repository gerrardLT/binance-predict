#!/usr/bin/env python
"""720d 四周期（5m/15m/1h/4h）「连续 ≥3 根同向 K 线」普查与科学探究。

用法：
    .venv\Scripts\python.exe -X utf8 scripts/streak_census_720d.py --tf all --stage all
    .venv\Scripts\python.exe -X utf8 scripts/streak_census_720d.py --tf 5m,4h --stage census,test

阶段（分阶段可重入，崩溃不重跑）：
    prepare  5m → 1h → 4h 严格聚合落盘 + 官方抽样交叉校验
    census   逐周期提取全部「极大同向连根段」（长度 ≥3），事件级落盘
    describe 描述性共同点统计（分布 / 前驱结构 / 基准率对照）
    test     预注册假设（config/discovery_rounds/r4_streak_origin.json）三段检验
    report   汇总 report.md + census_manifest.json

口径纪律：
- 方向 = sign(close - open)，doji（close==open）打断连根（与 discovery.features._streak 一致）
- 数据断点（cont=False）强制截断连根；聚合只保留桶内根齐且无断点的完整周期
- 事件锚定在末根（ex-ante 可观察：收盘时已知「连根 ≥3 进行中」）
- census 中的 label_* 列为未来信息，仅用于记录与描述；
  检验阶段走 0.6/0.2/0.2 三段切分 + BH-FDR + 冻结 holdout 只触碰一次
- 安慰剂：origin 类用随机非事件锚点重采样；outcome 类用条件向量置换
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from binance_predict.backtest.stats import exact_binomial_p, wilson  # noqa: E402
from binance_predict.discovery import (  # noqa: E402
    aggregate_to, build_feature_matrix, data_summary, load_klines_csv,
)
from binance_predict.discovery.features import atr_series  # noqa: E402
from binance_predict.discovery.l1_tester import bh_fdr  # noqa: E402
from binance_predict.discovery.oos_validator import run_block_ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, "output")
BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
HORIZONS = {"5m": [1, 2, 3], "15m": [1, 2, 3, 6, 12],
            "1h": [1, 2, 3, 6, 12, 24], "4h": [1, 2, 3, 6]}
TFS = ("5m", "15m", "1h", "4h")
R4_JSON = os.path.join(ROOT, "config", "discovery_rounds", "r4_streak_origin.json")
ROUND_JSON = R4_JSON  # 当前预注册轮文件，可用 --round 切换（哈希冻结纪律不变）
_FEAT_VERSION = 1  # 特征缓存版本（构建器修复时递增使缓存失效）

# 事件级前驱/上下文字段（位置语义见 build_events 注释）
PRIOR_FEATS = [
    "doji", "inside_bar", "outside_bar", "bullish_engulfing", "bearish_engulfing",
    "marubozu_bull", "marubozu_bear", "hammer_geometry", "shooting_star_geometry",
    "breakout_high_20", "breakout_low_20", "failed_breakout_high_20",
    "failed_breakout_low_20", "compression_5_20", "regime_vol_high", "regime_vol_low",
    "regime_trend_up", "regime_trend_down", "regime_trending", "regime_ranging",
    "atr_pctile_4320", "rel_volume_20", "range_atr", "body_atr",
]
CTX_FEATS = [
    "regime_trending", "regime_ranging", "regime_vol_high", "regime_vol_low",
    "regime_trend_up", "regime_trend_down",
    "atr_pctile_4320", "align_4h", "align_1h", "compression_5_20",
]
CFG = {
    "discovery_frac": 0.6, "validation_frac": 0.2, "holdout_frac": 0.2,
    "fdr_alpha": 0.1, "min_event_n": 30, "min_event_frac": 0.01,
    "origin_min_rel_lift": 0.10, "outcome_min_lift_pp": 2.0,
    "min_val_rel_lift": 0.05, "min_val_lift_pp": 0.5,
    "n_placebo": 20, "seed": 11, "min_holdout_n": 20,
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def _pkl(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _unpkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _round_sha256() -> str:
    with open(ROUND_JSON, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _clean(v):
    """NaN/inf → None（JSON/CSV 落盘防污染）。"""
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ============================ prepare：1h/4h 聚合 ============================

def write_klines_csv(kl, path: str) -> None:
    """与 klines_*_720d.csv 同格式：带 +00:00 偏移的 ISO 时间戳（防本地时区解析漂移）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(len(kl)):
            f.write(f"{iso_utc(kl.t[i])},{kl.o[i]:.8f},{kl.h[i]:.8f},"
                    f"{kl.l[i]:.8f},{kl.c[i]:.8f},{kl.v[i]:.8f}\n")


def _xcheck_official_sample(kl1h, n_windows: int = 3, days: int = 7) -> dict:
    """官方 1h 抽样交叉校验：随机取若干 7 天窗口逐桶比对 OHLC（只校验不替换）。"""
    import urllib.request
    rng = np.random.default_rng(CFG["seed"])
    span = int(len(kl1h))
    win = days * 24
    if span < win * 2:
        return {"skipped": True, "reason": "数据跨度不足"}
    starts = sorted(rng.integers(0, span - win, size=min(n_windows, span // win)))
    checked, mism, worst = 0, 0, 0.0
    for s0 in starts:
        t_a, t_b = int(kl1h.t[s0]), int(kl1h.t[s0 + win - 1]) + BAR_MS["1h"]
        url = ("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT"
               f"&interval=1h&startTime={t_a}&endTime={t_b}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                batch = json.loads(resp.read().decode())
        except Exception as ex:
            return {"skipped": True, "reason": f"{type(ex).__name__}: {ex}",
                    "note": "官方端点不可达；聚合审计仍构成正确性证据。"}
        off = {int(r[0]): r for r in batch}
        for i in range(s0, s0 + win):
            r = off.get(int(kl1h.t[i]))
            if r is None:
                continue
            checked += 1
            for fld, idx in (("o", 1), ("h", 2), ("l", 3), ("c", 4)):
                d = abs(float(r[idx]) - float(getattr(kl1h, fld)[i]))
                worst = max(worst, d)
                if d > 1e-8:
                    mism += 1
        time.sleep(0.3)
    return {"windows": len(starts), "buckets_checked": checked,
            "ohlc_mismatch_cells": mism, "worst_abs_diff": worst,
            "note": "官方端点仅作交叉校验；实际研究使用 5m 聚合出的 1h。"}


def stage_prepare(out: str, force: bool) -> dict:
    """5m → 1h（12×5m）→ 4h（4×1h，二级聚合保 cont 语义）。"""
    p5 = os.path.join(OUT0, "klines_5m_720d.csv")
    if not os.path.exists(p5):
        raise FileNotFoundError(f"缺少 5m 数据：{p5}（先跑 scripts/refresh_and_export_klines.py）")
    p1, p4 = os.path.join(out, "klines_1h_720d.csv"), os.path.join(out, "klines_4h_720d.csv")
    if os.path.exists(p1) and os.path.exists(p4) and not force:
        log(f"prepare：1h/4h CSV 已存在（{p1}），跳过（--force 重建）")
        return {"cached": True}
    kl5 = load_klines_csv(p5, BAR_MS["5m"])
    log(f"prepare：5m 载入 {len(kl5)} 根（{iso_utc(kl5.t[0])} ~ {iso_utc(kl5.t[-1])}）")
    kl1h = aggregate_to(kl5, BAR_MS["1h"])
    kl4h = aggregate_to(kl1h, BAR_MS["4h"])
    audit = {
        "rows_5m": int(len(kl5)), "rows_1h": int(len(kl1h)), "rows_4h": int(len(kl4h)),
        "expect_1h": int(len(kl5) // 12), "expect_4h": int(len(kl5) // 48),
        "cont_break_1h": int((~kl1h.cont[1:]).sum()),
        "cont_break_4h": int((~kl4h.cont[1:]).sum()),
        "data_summary_1h": data_summary(kl1h, BAR_MS["1h"]),
        "data_summary_4h": data_summary(kl4h, BAR_MS["4h"]),
    }
    assert abs(audit["rows_1h"] - audit["expect_1h"]) <= 2, "1h 行数异常（断点过多？）"
    assert abs(audit["rows_4h"] - audit["expect_4h"]) <= 2, "4h 行数异常（断点过多？）"
    # 包络自检：high >= max(o,c)、low <= min(o,c)
    for kl in (kl1h, kl4h):
        assert np.all((kl.h >= np.maximum(kl.o, kl.c)) & (kl.l <= np.minimum(kl.o, kl.c)))
    audit["ohlc_envelope_ok"] = True
    audit["official_crosscheck_1h"] = _xcheck_official_sample(kl1h)
    write_klines_csv(kl1h, p1)
    write_klines_csv(kl4h, p4)
    audit["csv_1h"] = os.path.relpath(p1, ROOT).replace("\\", "/")
    audit["csv_4h"] = os.path.relpath(p4, ROOT).replace("\\", "/")
    _pkl(os.path.join(out, "_stage_prepare.pkl"), audit)
    log(f"prepare：1h {audit['rows_1h']} 根、4h {audit['rows_4h']} 根；"
        f"交叉校验 {audit['official_crosscheck_1h']}")
    return audit


def data_csv_path(tf: str, out: str) -> str:
    if tf in ("5m", "15m"):
        return os.path.join(OUT0, f"klines_{tf}_720d.csv")
    return os.path.join(out, f"klines_{tf}_720d.csv")


# ============================ 特征缓存（复用 run_kline_discovery 模式） ============================

def _feat_fp(kl, bar_ms: int) -> str:
    return hashlib.sha256(
        f"streakfeat-v{_FEAT_VERSION}|{bar_ms}|{kl.t[0]}|{kl.t[-1]}|{len(kl.t)}".encode("utf-8")
    ).hexdigest()[:16]


def get_featmat(kl, bar_ms: int, out: str, tf: str):
    fp_path = os.path.join(out, f"_featmat_{tf}.npz")
    fp = _feat_fp(kl, bar_ms)
    if os.path.exists(fp_path):
        try:
            z = np.load(fp_path, allow_pickle=True)
            meta = json.loads(str(z["_meta"][0]))
            if meta["fp"] == fp:
                from binance_predict.discovery.features import FeatureMatrix
                fm = FeatureMatrix()
                for i, nm in enumerate(meta["names"]):
                    fm.names.append(nm)
                    fm.families.append(meta["families"][i])
                    fm.dtypes.append(meta["dtypes"][i])
                    fm.cols[nm] = z[f"col_{i}"]
                log(f"  特征缓存命中：{len(fm)} 列")
                return fm
        except Exception as ex:
            log(f"  特征缓存失效（{ex}），重建")
    t0 = time.time()
    fm = build_feature_matrix(kl, bar_ms, None)
    log(f"  特征构建 {len(fm)} 列 × {len(kl)} 根，耗时 {time.time() - t0:.1f}s")
    meta = json.dumps({"names": fm.names, "families": fm.families,
                       "dtypes": fm.dtypes, "fp": fp})
    np.savez_compressed(fp_path, _meta=np.array([meta]),
                        **{f"col_{i}": fm.cols[nm] for i, nm in enumerate(fm.names)})
    return fm


# ============================ census：连根事件提取 ============================

def _fwd_ok(cont: np.ndarray, h: int) -> np.ndarray:
    """t→t+h 全程连续：cont[t+1..t+h] 全 True（与 discovery.targets._fwd_ok 同构）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(cont)
    out = np.zeros(n, dtype=bool)
    if n > h:
        out[: n - h] = sliding_window_view(cont[1:], h).all(axis=1)
    return out


def build_events(tf: str, kl, fm, out: str) -> dict:
    """提取全部极大同向连根段（≥3 根），构建事件级数组并落盘。

    位置语义（全部 ex-ante 安全）：
    - prior_*  取 start−1 根（连根启动前的那根）
    - ctx_*    取末根 end（锚点，收盘时可见）
    - streak_* 为 [start, end] 段内聚合
    - label_*  为未来信息（仅记录用），基于末根后 h 根
    """
    bar_ms = BAR_MS[tf]
    n = len(kl)
    dir_ = np.sign(kl.c - kl.o)
    same = np.zeros(n, dtype=bool)
    same[1:] = (dir_[1:] == dir_[:-1]) & (dir_[1:] != 0) & kl.cont[1:]
    starts = np.flatnonzero(~same)
    grp = np.searchsorted(starts, np.arange(n), side="right") - 1
    st = (np.arange(n) - starts[grp]).astype(np.int64)
    cont_next = np.append(same[1:], False)          # i+1 是否延续当前段
    e = np.flatnonzero((st >= 3) & ~cont_next)      # 事件末根
    s = starts[grp][(st >= 3) & ~cont_next]         # 事件首根
    length = st[e]
    ev_dir = dir_[e]
    right_trunc = e == n - 1                        # 尾段截断（未观察到终结）
    ne = len(e)
    log(f"  事件提取：{ne} 个（≥3 根；右截断 {int(right_trunc.sum())} 个）")

    cols: dict[str, np.ndarray] = {}
    # ---- 前驱（start−1；要求首根与前一根严格相邻）----
    vp = (s >= 1) & kl.cont[s]
    pm = np.where(vp, s - 1, 0)
    cols["valid_prior"] = vp
    for feat in PRIOR_FEATS:
        cols[f"prior_{feat}"] = np.where(vp, fm.cols[feat][pm], np.nan)
    # 派生前驱：反向启动 / 前 5 根全反向 / 前 5 根方向串
    prior_rev = vp & (dir_[pm] != 0) & (dir_[pm] == -ev_dir)
    cols["prior_reverse"] = prior_rev.astype(np.float64)
    cols["prior_1_dir"] = np.where(vp, dir_[pm], np.nan)
    ok5 = (s >= 5) & np.all([kl.cont[s - k] for k in range(5)], axis=0)
    ar = np.ones(ne, dtype=bool)
    str5 = []
    for j in range(ne):
        if ok5[j]:
            d5 = dir_[s[j] - 5: s[j]]
            str5.append("".join("U" if x > 0 else ("D" if x < 0 else "F") for x in d5))
            ar[j] = bool(np.all(d5 != 0) and np.all(d5 == -ev_dir[j]))
        else:
            str5.append("")
            ar[j] = False
    cols["prior_all_reverse"] = np.where(ok5, ar, False).astype(np.float64)
    cols["prior_5_dirs"] = np.array(str5, dtype=object)
    # ---- 上下文（锚点=末根）----
    for feat in CTX_FEATS:
        cols[f"ctx_{feat}"] = fm.cols[feat][e].astype(np.float64)
    # ---- 段内聚合 ----
    cols["length"] = length.astype(np.float64)
    cols["dir_up"] = (ev_dir > 0).astype(np.float64)
    cols["dir_down"] = (ev_dir < 0).astype(np.float64)
    relv, batr = fm.cols["rel_volume_20"], fm.cols["body_atr"]
    streak_rel_vol, streak_body_atr = np.full(ne, np.nan), np.full(ne, np.nan)
    for j in range(ne):
        seg = slice(int(s[j]), int(e[j]) + 1)
        rv = relv[seg]
        streak_rel_vol[j] = np.nanmean(rv) if np.isfinite(rv).any() else np.nan
        ba = batr[seg]
        streak_body_atr[j] = np.nanmean(ba) if np.isfinite(ba).any() else np.nan
    cols["streak_rel_vol"] = streak_rel_vol
    cols["streak_body_atr"] = streak_body_atr
    atr0 = atr_series(kl)
    body_sum = np.array([float(np.abs(kl.c[s[j]: e[j] + 1] - kl.o[s[j]: e[j] + 1]).sum())
                           for j in range(ne)])
    cols["cum_body_atr"] = np.where(atr0[e] > 0, body_sum / np.where(atr0[e] > 0, atr0[e], 1),
                                    np.nan)
    cols["cum_ret_pct"] = (kl.c[e] / kl.o[s] - 1.0) * 100.0
    # ---- 时间上下文（启动根）----
    hours = (kl.t[s] // 3_600_000) % 24
    cols["hour_utc_start"] = hours.astype(np.float64)
    cols["day_of_week_start"] = ((kl.t[s] // 86_400_000 + 3) % 7).astype(np.float64)
    cols["session_asia_start"] = (hours <= 7).astype(np.float64)
    cols["session_europe_start"] = ((hours >= 8) & (hours <= 15)).astype(np.float64)
    cols["session_us_start"] = (hours >= 16).astype(np.float64)
    cols["funding_slot_start"] = np.isin(hours, [0, 8, 16]).astype(np.float64)
    # ---- 结果标签（未来信息，仅记录）----
    labels: dict[int, dict] = {}
    for h in HORIZONS[tf]:
        ok_f = _fwd_ok(kl.cont, h)
        valid = ok_f[e] & ~right_trunc & (e + h < n) & (dir_[np.minimum(e + h, n - 1)] != 0)
        eh = np.minimum(e + h, n - 1)
        cont_h = dir_[eh] == ev_dir
        ret_h = ev_dir * (kl.c[eh] - kl.c[e]) / kl.c[e]
        labels[h] = {"cont": np.where(valid, cont_h, False),
                     "ret": np.where(valid, ret_h, np.nan), "valid": valid}

    # ---- 三段切分（事件按末根时序排列，构造即有序）----
    i1, i2 = int(ne * CFG["discovery_frac"]), int(ne * (CFG["discovery_frac"]
                                                         + CFG["validation_frac"]))
    bi1, bi2 = int(n * CFG["discovery_frac"]), int(n * (CFG["discovery_frac"]
                                                         + CFG["validation_frac"]))
    ev = {
        "tf": tf, "bar_ms": bar_ms, "n_bars": n, "n_events": ne,
        "data_summary": data_summary(kl, bar_ms),
        "e": e, "s": s, "dir": ev_dir, "length": length,
        "right_truncated": right_trunc,
        "open_first": kl.o[s], "close_last": kl.c[e],
        "start_ts": kl.t[s], "end_ts": kl.t[e],
        "seg_events": (i1, i2), "seg_bars": (bi1, bi2),
        "cols": cols, "labels": labels,
        # 供安慰剂重采样：全量方向序列 + 事件末根掩码
        "dir_all": dir_, "cont_all": kl.cont,
        "non_event_mask": _non_event_mask(n, e, s),
    }
    # ---- census CSV 落盘 ----
    _write_census_csv(tf, ev, os.path.join(out, f"census_{tf}.csv"))
    slim = {k: v for k, v in ev.items() if k not in ("dir_all", "cont_all")}
    slim["dir_all"] = dir_.astype(np.int8)
    slim["cont_all"] = kl.cont
    _pkl(os.path.join(out, f"_events_{tf}.pkl"), slim)
    return ev


def _non_event_mask(n: int, e: np.ndarray, s: np.ndarray) -> np.ndarray:
    """非连根内部也非事件末根的棒位（安慰剂锚点候选）。"""
    m = np.ones(n, dtype=bool)
    for sj, ej in zip(s, e):
        m[sj: ej + 1] = False
    return m


def _write_census_csv(tf: str, ev: dict, path: str) -> None:
    """全量事件记录。label_* 为未来信息标签（记录用，非验证输入）。"""
    scalar_cols = [
        "dir_up", "cum_ret_pct", "cum_body_atr", "streak_rel_vol",
        "streak_body_atr", "hour_utc_start", "day_of_week_start",
        "session_asia_start", "session_europe_start", "session_us_start",
        "funding_slot_start", "prior_1_dir", "prior_reverse", "prior_all_reverse",
    ] + [f"prior_{f}" for f in PRIOR_FEATS] + [f"ctx_{f}" for f in CTX_FEATS]
    hdr = (["tf", "streak_id", "dir", "length", "start_ts", "end_ts",
            "right_truncated", "open_first", "close_last", "prior_5_dirs"]
           + scalar_cols)
    hdr += [k for h in HORIZONS[tf] for k in (f"label_cont_{h}", f"label_ret_{h}",
                                              f"label_valid_{h}")]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        cols, labels = ev["cols"], ev["labels"]
        for j in range(ev["n_events"]):
            row = [tf, f"{tf}-{j:06d}", int(ev["dir"][j]), int(ev["length"][j]),
                   iso_utc(ev["start_ts"][j]), iso_utc(ev["end_ts"][j]),
                   bool(ev["right_truncated"][j]),
                   f"{ev['open_first'][j]:.2f}", f"{ev['close_last'][j]:.2f}",
                   cols["prior_5_dirs"][j]]
            for c in scalar_cols:
                v = cols[c][j]
                if c == "dir_up":
                    row.append(bool(v))
                elif isinstance(v, (bool, np.bool_)):
                    row.append(bool(v))
                else:
                    row.append("" if not np.isfinite(float(v)) else f"{float(v):.6g}")
            for h in HORIZONS[tf]:
                lb = labels[h]
                row.append("" if not lb["valid"][j] else bool(lb["cont"][j]))
                row.append("" if not lb["valid"][j] else f"{lb['ret'][j]:.6g}")
                row.append(bool(lb["valid"][j]))
            w.writerow(row)
    log(f"  census CSV：{ev['n_events']} 行 -> {os.path.relpath(path, ROOT)}")


# ============================ 假设条件求值 ============================

def _col_for(ev: dict, scope: str, feat: str) -> np.ndarray:
    if scope in ("prior", "ctx"):
        return ev["cols"][f"{scope}_{feat}"].astype(np.float64)
    return ev["cols"][feat].astype(np.float64)


def _apply_op(col: np.ndarray, op: str, val) -> np.ndarray:
    v = 1.0 if val is True else (0.0 if val is False else float(val))
    finite = np.isfinite(col)
    if op == "==":
        return finite & (col == v)
    if op == ">=":
        return finite & (col >= v)
    if op == "<=":
        return finite & (col <= v)
    raise ValueError(f"不支持算子: {op}")


def eval_event_condition(ev: dict, cond: dict, thr: float | None) -> np.ndarray:
    """事件级条件求值（NaN → False）。quantile 条件用 census 发现段拟合的 thr。"""
    val = thr if cond.get("quantile") is not None else cond["value"]
    col = _col_for(ev, cond["scope"], cond["feature"])
    return _apply_op(col, cond["op"], val)


# ============================ census 阶段的假设预计算 ============================

def _derived_bar_cols(ev: dict) -> dict[str, np.ndarray]:
    """事件派生条件的棒级类比列（供基准率与安慰剂）。"""
    d = ev["dir_all"].astype(np.float64)
    n = len(d)
    rev = np.zeros(n, dtype=bool)
    rev[1:] = (d[1:] != 0) & (d[:-1] != 0) & (d[1:] == -d[:-1])
    ar5 = np.zeros(n, dtype=bool)
    if n > 5:
        ok = (d[5:] != 0)
        for k in range(1, 6):
            ok &= (d[5 - k: n - k] != 0) & (d[5 - k: n - k] == -d[5:])
        ar5[5:] = ok
    return {"reverse_start": rev, "all_reverse_5": ar5}


def _bar_analog_mask(fm, derived: dict, spec: dict, thr: float | None) -> np.ndarray:
    """棒级类比条件掩码（全量）。"""
    if spec.get("derived"):
        return derived[spec["derived"]]
    val = thr if spec.get("quantile") is not None else spec["value"]
    return _apply_op(fm.cols[spec["feature"]].astype(np.float64), spec["op"], val)


def prepare_hypotheses(tf: str, ev: dict, fm, out: str) -> None:
    """census 阶段：为 R4 每个假设做发现段阈值拟合 + 棒级基准/安慰剂预计算。

    纪律：阈值只在发现段事件拟合；holdout 段只留基础列，检验阶段才触碰一次。
    """
    if not os.path.exists(ROUND_JSON):
        raise FileNotFoundError(f"缺少预注册假设：{ROUND_JSON}")
    with open(ROUND_JSON, encoding="utf-8") as f:
        r4 = json.load(f)
    hyps = r4["hypotheses"]
    n = ev["n_bars"]
    i1e, i2e = ev["seg_events"]
    bi1, bi2 = ev["seg_bars"]
    seg_names = ("discovery", "validation", "holdout")
    ev_seg = [np.arange(0, i1e), np.arange(i1e, i2e), np.arange(i2e, ev["n_events"])]
    bar_seg = [np.arange(0, bi1), np.arange(bi1, bi2), np.arange(bi2, n)]
    derived = _derived_bar_cols(ev)
    rng = np.random.default_rng(CFG["seed"])
    non_ev_idx = np.flatnonzero(ev["non_event_mask"])
    prep: dict[str, dict] = {}
    for hp in hyps:
        cond = hp["condition"]
        entry: dict = {}
        # ---- 阈值拟合（仅发现段事件，防泄漏）----
        thr = None
        if cond.get("quantile") is not None:
            v = _col_for(ev, cond["scope"], cond["feature"])[ev_seg[0]]
            v = v[np.isfinite(v)]
            thr = float(np.quantile(v, cond["quantile"])) if len(v) >= 100 else None
            if thr is None:
                log(f"  {hp['id']}: 发现段有限值 <100，quantile 条件弃用")
        entry["thr"] = thr
        if cond.get("quantile") is not None and thr is None:
            condv = np.zeros(ev["n_events"], dtype=bool)
        else:
            condv = eval_event_condition(ev, cond, thr)
        entry["cond"] = condv
        if hp["kind"] == "origin":
            # ---- 棒级类比基准 + 安慰剂重采样 ----
            analog = hp.get("bar_analog")
            ref: dict[str, dict] = {}
            if analog is not None:
                amask = _bar_analog_mask(fm, derived, analog, thr)
                for si, seg_name in enumerate(seg_names):
                    bidx = bar_seg[si]
                    bar_rate = float(amask[bidx].mean()) if len(bidx) else np.nan
                    ne_seg = len(ev_seg[si])
                    cand = (non_ev_idx[(non_ev_idx >= bidx[0]) & (non_ev_idx <= bidx[-1])]
                            if len(bidx) else np.array([], dtype=int))
                    rates = []
                    if len(cand) and ne_seg:
                        for _ in range(CFG["n_placebo"]):
                            samp = rng.choice(cand, size=min(ne_seg, len(cand)), replace=False)
                            rates.append(float(amask[samp].mean()))
                    ref[seg_name] = {
                        "bar_rate": bar_rate,
                        "placebo_mean": float(np.mean(rates)) if rates else np.nan,
                        "placebo_ci": ([float(x) for x in np.percentile(rates, [2.5, 97.5])]
                                       if rates else [np.nan, np.nan]),
                    }
            entry["origin_ref"] = ref
            # ---- 方向偏置类：response 基准（全体事件内 response 率）----
            if hp.get("response") is not None:
                rc = _col_for(ev, "event", hp["response"]["feature"])
                entry["response_base"] = {
                    seg_names[si]: (float(np.nanmean(rc[ev_seg[si]])) if len(ev_seg[si]) else np.nan)
                    for si in range(3)}
        prep[hp["id"]] = entry
    ev["hyp_prep"] = prep
    ev["r4_sha256"] = _round_sha256()  # 键名为历史兼容：实为当前轮文件哈希
    ev["round_file"] = os.path.basename(ROUND_JSON)
    slim = dict(ev)
    slim["dir_all"] = ev["dir_all"].astype(np.int8)
    _pkl(os.path.join(out, f"_events_{tf}.pkl"), slim)


# ============================ test：三段检验 ============================

def _seg_stats_origin(ev, hp, prep, seg_name, seg_ev_idx) -> dict:
    condv = prep["cond"][seg_ev_idx]
    n_tot, n_hit = len(seg_ev_idx), int(condv.sum())
    rate = n_hit / n_tot if n_tot else np.nan
    ref = prep.get("origin_ref", {}).get(seg_name) or {}
    base = ref.get("bar_rate", np.nan)
    rel = (rate / base - 1.0) if (np.isfinite(base) and base > 0 and np.isfinite(rate)) else np.nan
    p = exact_binomial_p(n_hit, n_tot, base) if (n_tot and np.isfinite(base) and 0 < base < 1) else np.nan
    out = {"n": n_tot, "n_hit": n_hit, "rate": rate, "base_rate": base,
           "rel_lift": rel, "p_value": p}
    if hp.get("response") is not None:
        rc = _col_for(ev, "event", hp["response"]["feature"])[seg_ev_idx]
        m = condv & np.isfinite(rc)
        k = int(rc[m].sum()) if m.any() else 0
        n_m = int(m.sum())
        base_r = prep.get("response_base", {}).get(seg_name, np.nan)
        wr = k / n_m if n_m else np.nan
        out["response_n"], out["response_rate"] = n_m, wr
        out["response_base"] = base_r
        out["response_lift_pp"] = ((wr - base_r) * 100
                                   if np.isfinite(wr) and np.isfinite(base_r) else np.nan)
        out["response_p"] = (exact_binomial_p(k, n_m, base_r)
                             if n_m and np.isfinite(base_r) and 0 < base_r < 1 else np.nan)
    return out


def _seg_stats_outcome(ev, hp, prep, seg_name, seg_ev_idx) -> dict:
    h, fam = hp["horizon"], hp["target_family"]
    lb = ev["labels"][h]
    valid = lb["valid"][seg_ev_idx]
    win = (lb["cont"][seg_ev_idx] if fam == "continuation"
           else ~lb["cont"][seg_ev_idx]) & valid
    n_base = int(valid.sum())
    base = float(win.sum()) / n_base if n_base else np.nan
    condv = prep["cond"][seg_ev_idx] & valid
    n_hit = int(condv.sum())
    k = int(win[condv].sum()) if n_hit else 0
    wr = k / n_hit if n_hit else np.nan
    lo, hi = wilson(wr, n_hit) if n_hit else (np.nan, np.nan)
    p = exact_binomial_p(k, n_hit, base) if (n_hit and np.isfinite(base) and 0 < base < 1) else np.nan
    return {"n": n_hit, "n_base": n_base, "baseline": base, "win_rate": wr,
            "lift_pp": ((wr - base) * 100
                        if np.isfinite(wr) and np.isfinite(base) else np.nan),
            "ci_low": lo, "ci_high": hi, "p_value": p}


def stage_test(out: str, tfs: tuple[str, ...]) -> dict:
    with open(ROUND_JSON, encoding="utf-8") as f:
        r4 = json.load(f)
    hyps = {hp["id"]: hp for hp in r4["hypotheses"]}
    all_results: dict[str, dict] = {}
    for tf in tfs:
        p = os.path.join(out, f"_events_{tf}.pkl")
        if not os.path.exists(p):
            log(f"test：{tf} 缺少事件产物，跳过")
            continue
        ev = _unpkl(p)
        if ev.get("r4_sha256") != _round_sha256():
            raise RuntimeError(f"{tf}: 预注册假设在 census 后变更（哈希不符）——违反冻结纪律")
        prep_all = ev.get("hyp_prep", {})
        i1e, i2e = ev["seg_events"]
        ne = ev["n_events"]
        seg_ev = [np.arange(0, i1e), np.arange(i1e, i2e), np.arange(i2e, ne)]
        n_floor = [max(CFG["min_event_n"], int(CFG["min_event_frac"] * len(sx)))
                   if len(sx) else CFG["min_event_n"] for sx in seg_ev]
        rows: list[dict] = []
        for hid, hp in hyps.items():
            prep = prep_all.get(hid)
            if prep is None:
                continue
            row = {"id": hid, "family": hp["family"], "kind": hp["kind"],
                   "expect": hp.get("expect", ""), "mechanism": hp["mechanism"],
                   "condition_text": json.dumps(hp["condition"], ensure_ascii=False),
                   "fit_threshold": prep.get("thr")}
            seg_fn = _seg_stats_origin if hp["kind"] == "origin" else _seg_stats_outcome
            # holdout 段延迟到存活者才算（只触碰一次的纪律落点）
            d = seg_fn(ev, hp, prep, "discovery", seg_ev[0])
            v = seg_fn(ev, hp, prep, "validation", seg_ev[1])
            if hp["kind"] == "origin":
                n_gate = d["n_hit"] if hp.get("response") is None else d["response_n"]
                gated = (d["n"] >= n_floor[0]) and (n_gate or 0) >= CFG["min_event_n"]
            else:
                gated = d["n"] >= n_floor[0]
            row["gated"] = bool(gated)
            row.update({f"discovery_{k}": _clean(x) for k, x in d.items()})
            row.update({f"validation_{k}": _clean(x) for k, x in v.items()})
            if not gated:
                row["status"] = "INSUFFICIENT_SAMPLES"
                row["verdict"] = "INSUFFICIENT_SAMPLES"
                rows.append(row)
                continue
            # ---- 验证段方向一致性 ----
            if hp["kind"] == "origin":
                key = "response_lift_pp" if hp.get("response") else "rel_lift"
                dv, vv = d.get(key), v.get(key)
                min_val = CFG["min_val_lift_pp"] if key == "response_lift_pp" \
                    else CFG["min_val_rel_lift"]
                val_pass = (np.isfinite(dv) and np.isfinite(vv)
                            and np.sign(dv) == np.sign(vv) and abs(vv) >= min_val)
            else:
                dv, vv = d["lift_pp"], v["lift_pp"]
                val_pass = (np.isfinite(dv) and np.isfinite(vv)
                            and np.sign(dv) == np.sign(vv)
                            and abs(vv) >= CFG["min_val_lift_pp"])
            row["validation_pass"] = bool(val_pass)
            rows.append(row)
        # ---- BH-FDR（发现段 p 值族，按 kind 分组）----
        for kind in ("origin", "outcome"):
            idx = [i for i, r in enumerate(rows) if r.get("gated") and r["kind"] == kind
                   and r.get("discovery_p_value") is not None]
            pv = [rows[i]["discovery_p_value"] for i in idx]
            ok = bh_fdr(pv, q=CFG["fdr_alpha"]) if pv else []
            for i, flag in zip(idx, ok):
                rows[i]["fdr_pass"] = bool(flag)
        for r in rows:
            r.setdefault("fdr_pass", False)
        # ---- 安慰剂（仅发现段零分布；holdout 不触碰）----
        rng = np.random.default_rng(CFG["seed"] + 1)
        for r in rows:
            hp = hyps[r["id"]]
            prep = prep_all[r["id"]]
            if not r.get("gated"):
                continue
            if hp["kind"] == "outcome":
                lb = ev["labels"][hp["horizon"]]
                valid = lb["valid"][seg_ev[0]]
                win = ((lb["cont"][seg_ev[0]] if hp["target_family"] == "continuation"
                        else ~lb["cont"][seg_ev[0]]) & valid)
                base = float(win.sum()) / int(valid.sum()) if valid.any() else np.nan
                condv = prep["cond"][seg_ev[0]] & valid
                lifts = []
                if int(valid.sum()) and np.isfinite(base):
                    for _ in range(CFG["n_placebo"]):
                        perm = rng.permutation(condv)
                        if perm.any():
                            lifts.append((float(win[perm].mean()) - base) * 100)
                r["placebo_lift_ci"] = ([float(x) for x in np.percentile(lifts, [2.5, 97.5])]
                                        if lifts else None)
            else:
                ref = prep.get("origin_ref", {}).get("validation") or {}
                r["placebo_rate_ci"] = ref.get("placebo_ci")
                r["placebo_rate_mean"] = _clean(ref.get("placebo_mean"))
        # ---- 冻结 holdout 终验（只触碰一次：仅验证段存活者）----
        for r in rows:
            hp = hyps[r["id"]]
            prep = prep_all[r["id"]]
            if not (r.get("gated") and r.get("fdr_pass") and r.get("validation_pass")):
                r["verdict"] = r.get("status", "REJECT_PREHOLDOUT")
                continue
            seg_fn = _seg_stats_origin if hp["kind"] == "origin" else _seg_stats_outcome
            hstat = seg_fn(ev, hp, prep, "holdout", seg_ev[2])
            r.update({f"holdout_{k}": _clean(x) for k, x in hstat.items()})
            if hp["kind"] == "origin":
                key = "response_lift_pp" if hp.get("response") else "rel_lift"
                dv, hv = r[f"discovery_{key}"], hstat.get(key)
                n_h = hstat.get("response_n") if hp.get("response") else hstat.get("n_hit")
                same = (hv is not None and np.isfinite(hv) and np.isfinite(dv)
                        and np.sign(hv) == np.sign(dv))
                p_h = hstat.get("response_p" if hp.get("response") else "p_value")
                min_h = CFG["min_val_lift_pp"] if key == "response_lift_pp" \
                    else CFG["origin_min_rel_lift"]
                if n_h is None or n_h < CFG["min_holdout_n"] or not same:
                    r["verdict"] = ("REJECT" if (hv is not None and np.isfinite(hv)
                                                 and np.isfinite(dv) and hv != 0
                                                 and np.sign(hv) != np.sign(dv))
                                    else "WEAK")
                elif abs(hv) >= min_h and p_h is not None and p_h < 0.05:
                    r["verdict"] = "CONFIRMED"
                else:
                    r["verdict"] = "PROMISING"
            else:
                dv, hv = r["discovery_lift_pp"], hstat["lift_pp"]
                same = (np.isfinite(hv) and np.isfinite(dv) and np.sign(hv) == np.sign(dv))
                # run 级块自助 CI（事件末根位置 + 胜负，处理相邻事件依赖）
                lb = ev["labels"][hp["horizon"]]
                condv = prep["cond"][seg_ev[2]] & lb["valid"][seg_ev[2]]
                e_h = ev["e"][seg_ev[2]][condv]
                win_h = ((lb["cont"][seg_ev[2]] if hp["target_family"] == "continuation"
                          else ~lb["cont"][seg_ev[2]])[condv]).astype(float)
                ci_lo, ci_hi = run_block_ci(e_h, win_h) if len(e_h) else (np.nan, np.nan)
                r["holdout_block_ci"] = [_clean(ci_lo), _clean(ci_hi)]
                base_h = hstat["baseline"]
                ci_excl = bool(np.isfinite(ci_lo) and np.isfinite(base_h) and hv is not None
                               and np.isfinite(hv) and
                               ((hv > 0 and ci_lo > base_h) or (hv < 0 and ci_hi < base_h)))
                if hstat["n"] < CFG["min_holdout_n"] or not same:
                    r["verdict"] = ("REJECT" if (np.isfinite(hv) and np.isfinite(dv)
                                                 and np.sign(hv) != np.sign(dv))
                                    else "WEAK")
                elif abs(hv) >= CFG["outcome_min_lift_pp"] and ci_excl:
                    r["verdict"] = "CONFIRMED"
                elif abs(hv) >= CFG["outcome_min_lift_pp"] or ci_excl:
                    r["verdict"] = "PROMISING"
                else:
                    r["verdict"] = "WEAK"
        all_results[tf] = rows
        n_conf = sum(1 for r in rows if r.get("verdict") == "CONFIRMED")
        log(f"test：{tf} {len(rows)} 假设 → CONFIRMED {n_conf}")
    with open(os.path.join(out, "hypothesis_tests.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1, default=_clean)
    return all_results


# ============================ census 阶段编排 ============================

def stage_census(out: str, tfs: tuple[str, ...]) -> None:
    for tf in tfs:
        log(f"census：{tf}")
        kl = load_klines_csv(data_csv_path(tf, out), BAR_MS[tf])
        fm = get_featmat(kl, BAR_MS[tf], out, tf)
        ev = build_events(tf, kl, fm, out)
        prepare_hypotheses(tf, ev, fm, out)


# ============================ describe：描述性共同点 ============================

def _describe_tf(tf: str, ev: dict) -> dict:
    cols, ne = ev["cols"], ev["n_events"]
    length = ev["length"]
    up = int((ev["dir"] > 0).sum())
    hour = cols["hour_utc_start"].astype(int)
    dow = cols["day_of_week_start"].astype(int)
    # 长度分布（长尾合并入 10+）
    lh: dict[str, int] = {}
    for L in range(3, 10):
        lh[str(L)] = int((length == L).sum())
    lh["10+"] = int((length >= 10).sum())
    d = {
        "tf": tf, "n_events": ne, "n_bars": ev["n_bars"],
        "event_rate_per_1000bars": round(ne / ev["n_bars"] * 1000, 3),
        "dir_up_rate": round(up / ne, 4) if ne else None,
        "length_mean": round(float(length.mean()), 3) if ne else None,
        "length_p50": int(np.median(length)) if ne else None,
        "length_p95": int(np.percentile(length, 95)) if ne else None,
        "length_max": int(length.max()) if ne else None,
        "length_hist": lh,
        "right_truncated": int(ev["right_truncated"].sum()),
        "hour_utc_hist": {str(h): int((hour == h).sum()) for h in range(24)},
        "day_of_week_hist": {str(w): int((dow == w).sum()) for w in range(7)},
        "cum_ret_pct": {k: round(float(np.percentile(cols["cum_ret_pct"], q)), 4)
                        for k, q in (("p25", 25), ("p50", 50), ("p75", 75))},
        "cum_body_atr_p50": round(float(np.nanpercentile(cols["cum_body_atr"], 50)), 3),
        "top_prior_5_dirs": dict(sorted(
            ((k, int(v)) for k, v in zip(*np.unique(cols["prior_5_dirs"],
                                                     return_counts=True)) if k),
            key=lambda kv: -kv[1])[:10]),
    }
    # ---- 前驱结构：事件内条件率 vs 棒级类比基准（富集 = 共同点线索）----
    prep = ev.get("hyp_prep", {})
    prior: dict[str, dict] = {}
    for feat in PRIOR_FEATS:
        v = cols[f"prior_{feat}"]
        prior[f"prior_{feat}"] = {"event_rate": round(float(np.nanmean(v)), 4)}
    rc = cols["prior_reverse"]
    prior["prior_reverse"] = {"event_rate": round(float(np.nanmean(rc)), 4),
                              "note": "极大段定义逻辑蕴含（首根前一根必不同向或 doji），≈1 为 tautology，非共同点"}
    ar = cols["prior_all_reverse"]
    prior["prior_all_reverse"] = {"event_rate": round(float(np.nanmean(ar)), 4)}
    # 棒级基准来自 R4 预计算的 bar_analog（reverse_start / all_reverse_5）
    for hid, ref in (("R4-006", prior["prior_reverse"]),
                     ("R4-007", prior["prior_all_reverse"])):
        b = (prep.get(hid, {}).get("origin_ref", {}).get("discovery", {}) or {})
        ref["bar_base_rate"] = _clean(b.get("bar_rate"))
        if ref["bar_base_rate"]:
            ref["enrich"] = round(ref["event_rate"] / ref["bar_base_rate"], 3)
    d["prior_structure"] = prior
    # ---- 结果描述（仅描述；验证走三段纪律）----
    # h=1 延续由事件终结规则逻辑蕴含（必为 False），描述与检验一律用 h=2
    lb2 = ev["labels"].get(2)
    if lb2 is not None and lb2["valid"].any():
        d["outcome_cont_2_rate_all"] = round(float(lb2["cont"][lb2["valid"]].mean()), 4)
        m5 = lb2["valid"] & (length >= 5)
        if m5.any():
            d["outcome_cont_2_rate_len_ge5"] = round(float(lb2["cont"][m5].mean()), 4)
    return d


def stage_describe(out: str, tfs: tuple[str, ...]) -> dict:
    res: dict[str, dict] = {}
    for tf in tfs:
        p = os.path.join(out, f"_events_{tf}.pkl")
        if not os.path.exists(p):
            log(f"describe：{tf} 缺少事件产物，跳过")
            continue
        res[tf] = _describe_tf(tf, _unpkl(p))
        log(f"describe：{tf} {res[tf]['n_events']} 事件")
    # ---- 跨周期尺度不变性（描述性对照，不进检验账本）----
    cross = {
        "length_mean": {t: res[t]["length_mean"] for t in res},
        "length_p95": {t: res[t]["length_p95"] for t in res},
        "event_rate_per_1000bars": {t: res[t]["event_rate_per_1000bars"] for t in res},
        "cum_ret_pct_p50": {t: res[t]["cum_ret_pct"]["p50"] for t in res},
        "cum_body_atr_p50": {t: res[t]["cum_body_atr_p50"] for t in res},
        "outcome_cont_2_rate_all": {t: res[t].get("outcome_cont_2_rate_all") for t in res},
    }
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "per_tf": res, "cross_tf_scale_invariance": cross}
    with open(os.path.join(out, "describe_commonality.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=_clean)
    log(f"describe -> {os.path.join(out, 'describe_commonality.json')}")
    return payload


# ============================ report：汇总产出 ============================

def _update_manifest(out: str, tfs: tuple[str, ...]) -> None:
    mp = os.path.join(out, "census_manifest.json")
    manifest = {}
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            manifest = json.load(f)
    data = manifest.get("data", {})
    for tf in tfs:
        p = os.path.join(out, f"_events_{tf}.pkl")
        if os.path.exists(p):
            ev = _unpkl(p)
            data[tf] = {
                "n_bars": ev["n_bars"], "n_events": ev["n_events"],
                "data_summary": ev["data_summary"],
                "seg_events": list(ev["seg_events"]), "seg_bars": list(ev["seg_bars"]),
                "r4_sha256": ev.get("r4_sha256"),
                "census_csv": f"output/streak_research_720d/census_{tf}.csv",
            }
    manifest["data"] = data
    if os.path.exists(os.path.join(out, "hypothesis_tests.json")):
        with open(os.path.join(out, "hypothesis_tests.json"), encoding="utf-8") as f:
            tests = json.load(f)
        manifest["test_ledger"] = {
            tf: {v: sum(1 for r in rows if r.get("verdict") == v)
                 for v in ("CONFIRMED", "PROMISING", "WEAK", "REJECT",
                           "REJECT_PREHOLDOUT", "INSUFFICIENT_SAMPLES")}
            for tf, rows in tests.items()}
    manifest["config"] = CFG
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1, default=_clean)


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def stage_report(out: str, tfs: tuple[str, ...]) -> None:
    hp_path = os.path.join(out, "hypothesis_tests.json")
    ds_path = os.path.join(out, "describe_commonality.json")
    tests = json.load(open(hp_path, encoding="utf-8")) if os.path.exists(hp_path) else {}
    desc = json.load(open(ds_path, encoding="utf-8")) if os.path.exists(ds_path) else {}
    L: list[str] = ["# 720d 四周期「连续 ≥3 根同向 K 线」普查与科学探究报告", ""]
    L += [f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
          "## 1. 口径与方法", "",
          "- 事件定义：极大同向连根段（方向 = sign(close−open)，doji 打断，数据断点截断），长度 ≥3，锚定末根（ex-ante 可观察）。",
          "- 数据：5m/15m 原始 720d；1h = 5m 严格聚合（12×5m），4h = 1h 二级聚合（4×1h）。",
          "- 检验：0.6/0.2/0.2 三段时序切分；连续阈值仅发现段拟合；验证段方向一致 + BH-FDR(q=0.1)；冻结 holdout 只对存活者触碰一次；安慰剂对照（origin：非事件锚点重采样；outcome：条件置换）。",
          f"- 预注册：{os.path.relpath(ROUND_JSON, ROOT).replace(chr(92), '/')}（sha256 冻结，census 与 test 双端校验）。", ""]
    L += ["## 2. 普查概况", "",
          "| 周期 | K线数 | 事件数 | 事件率(‰) | 平均长度 | 长度P95 | 阳线占比 | 续2基准率 |",
          "|---|---|---|---|---|---|---|---|"]
    for tf, d in desc.get("per_tf", {}).items():
        L.append(f"| {tf} | {d['n_bars']} | {d['n_events']} | "
                 f"{d['event_rate_per_1000bars']} | {d['length_mean']} | {d['length_p95']} | "
                 f"{d['dir_up_rate']} | {_fmt(d.get('outcome_cont_2_rate_all'))} |")
    L += ["", "跨周期尺度不变性（描述性）：", "",
          "| 指标 | " + " | ".join(desc.get("per_tf", {}).keys()) + " |",
          "|---|" + "---|" * max(1, len(desc.get("per_tf", {})))]
    for k, row in desc.get("cross_tf_scale_invariance", {}).items():
        L.append(f"| {k} | " + " | ".join(_fmt(v) for v in row.values()) + " |")
    L += ["", "## 3. 描述性共同点（事件内条件率；富集度 = 事件率 ÷ 棒级类比基准）", ""]
    for tf, d in desc.get("per_tf", {}).items():
        L.append(f"### {tf}")
        ps = d.get("prior_structure", {})
        top = sorted((k for k in ps if ps[k].get("event_rate") is not None
                      and "note" not in ps[k]),
                     key=lambda k: -(ps[k]["event_rate"] or 0))[:8]
        for k in top:
            e = ps[k]
            extra = (f"；基准 {_fmt(e['bar_base_rate'])}，富集 {e['enrich']}×"
                     if e.get("enrich") else "")
            L.append(f"- `{k}`：事件内 {e['event_rate']}{extra}")
        if "prior_reverse" in ps:
            L.append(f"- `prior_reverse`：{ps['prior_reverse'].get('note', '')}")
        L.append("")
    L += ["## 4. 预注册假设检验（R4）", "",
          "裁决：CONFIRMED（holdout 方向一致 + 幅度达标 + 显著）> PROMISING > WEAK > "
          "REJECT / REJECT_PREHOLDOUT（未过验证段）/ INSUFFICIENT_SAMPLES。", ""]
    for tf, rows in tests.items():
        if not rows:
            continue
        L.append(f"### {tf}")
        L.append("| ID | 类别 | 机制 | 发现段 | 验证段 | FDR | holdout | 裁决 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            mech = (r.get("mechanism") or "")[:44]
            if r["kind"] == "origin":
                disc = (f"率 {_fmt(r.get('discovery_rate'))} vs 基准 "
                        f"{_fmt(r.get('discovery_base_rate'))}（n={r.get('discovery_n_hit')}"
                        f"/{r.get('discovery_n')}）")
                val = (f"{_fmt(r.get('validation_rate'))}，rel_lift "
                       f"{_fmt(r.get('validation_rel_lift'), 3)}")
            else:
                disc = (f"胜率 {_fmt(r.get('discovery_win_rate'))} vs 基准 "
                        f"{_fmt(r.get('discovery_baseline'))}（n={r.get('discovery_n')}）")
                val = (f"{_fmt(r.get('validation_win_rate'))}，lift "
                       f"{_fmt(r.get('validation_lift_pp'), 2)}pp")
            fdr = "✓" if r.get("fdr_pass") else "—"
            L.append(f"| {r['id']} | {r['kind']} | {mech} | {disc} | {val} | {fdr} | "
                     f"{_fmt(r.get('holdout_win_rate', r.get('holdout_rate')))} | "
                     f"**{r.get('verdict', '?')}** |")
        L.append("")
        for r in rows:
            if r.get("placebo_rate_mean") is not None or r.get("placebo_lift_ci"):
                ptxt = (f"安慰剂率均值 {_fmt(r['placebo_rate_mean'])}，CI "
                        f"{r.get('placebo_rate_ci')}" if r["kind"] == "origin"
                        else f"安慰剂 lift CI {r['placebo_lift_ci']}")
                L.append(f"- {r['id']} {ptxt}")
        L.append("")
    L += ["## 5. 结论与纪律声明", "",
          "- 描述性结论与可交易结论分开：上述富集/胜率差异是**条件概率描述**，未经成本、滑点与执行延迟检验，不直接构成可交易信号。",
          "- CONFIRMED 的假设通过了冻结 holdout 终验（全程只触碰一次）；其余均不应外推。",
          "- 全程防蕴含：前驱特征窗口严格位于连根启动之前（start−1 及更早），触发条件不含未来信息与标签同源变量。",
          "- 产物：census_{tf}.csv ×4（全量事件）、describe_commonality.json、hypothesis_tests.json、census_manifest.json。", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"report -> {os.path.join(out, 'report.md')}")


# ============================ main ============================

_STAGES = ("prepare", "census", "describe", "test", "report")


def main() -> int:
    ap = argparse.ArgumentParser(description="720d 四周期连根（≥3 同向）普查与科学探究")
    ap.add_argument("--tf", default="all", help="all 或逗号分隔子集，如 5m,4h")
    ap.add_argument("--stage", default="all", help="all 或逗号分隔：" + ",".join(_STAGES))
    ap.add_argument("--out", default=os.path.join(OUT0, "streak_research_720d"))
    ap.add_argument("--force", action="store_true", help="忽略 prepare 缓存重建 1h/4h")
    ap.add_argument("--round", default="r4_streak_origin.json",
                    help="预注册轮文件名（config/discovery_rounds/ 下）")
    args = ap.parse_args()
    global ROUND_JSON
    ROUND_JSON = (args.round if os.path.isabs(args.round) or os.path.exists(args.round)
                  else os.path.join(ROOT, "config", "discovery_rounds", args.round))
    tfs = TFS if args.tf == "all" else tuple(t.strip() for t in args.tf.split(",") if t.strip())
    bad = [t for t in tfs if t not in BAR_MS]
    if bad:
        raise SystemExit(f"未知周期：{bad}（可选 {list(BAR_MS)}）")
    stages = _STAGES if args.stage == "all" else tuple(
        s.strip() for s in args.stage.split(",") if s.strip())
    bad = [s for s in stages if s not in _STAGES]
    if bad:
        raise SystemExit(f"未知阶段：{bad}（可选 {list(_STAGES)}）")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for st in stages:
        if st == "prepare":
            stage_prepare(args.out, args.force)
        elif st == "census":
            stage_census(args.out, tfs)
        elif st == "describe":
            stage_describe(args.out, tfs)
        elif st == "test":
            stage_test(args.out, tfs)
        elif st == "report":
            _update_manifest(args.out, tfs)
            stage_report(args.out, tfs)
    log(f"完成，总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
