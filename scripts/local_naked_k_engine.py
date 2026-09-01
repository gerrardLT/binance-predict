#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K形态引擎：numpy 掩码构建 + 摆动状态机 + 无未来信息自检。

设计约束（见 docs/research/naked-k/REPORT.md 的口径声明）：
- **自持实现**，不 import discovery/features.py。理由：features 层对 1h 会调
  aggregate_to(kl, 3_600_000) 而在 n_sub<=1 时 raise（1h 跑不通）；且往生产特征层
  加形态列会扩大 tests/test_discovery_pipeline.py 的全列因果断言面。
- 所有判据严格只引用 <=t 的信息；摆动/订单块类概念通过「右侧确认根数右移」实现因果化。
- 量纲：本引擎的 body_r / up_r / lo_r / close_loc 一律除以 **绝对振幅 rng=h-l**，
  取值域 [0,1]。（discovery/features.py:191-202 的同名列除以的是比率 rng_pct=(h-l)/o，
  量纲不一致，详见 REPORT.md「与生产特征层的量纲差异」一节。）

用法（作为库被 local_naked_k_report.py 调用；单独执行跑自检）：
    uv run python scripts/local_naked_k_engine.py --selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from binance_predict.discovery.data import Klines, aggregate_to, load_klines_csv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "config", "naked_k_patterns.json")

BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
DAY_MS = 86_400_000
FUNDING_MS = 8 * 3_600_000
SWING_K = 2          # Williams 分形左右各 2 根
SWING_SHIFT = SWING_K  # 确认所需右移根数（防未来函数）
OB_LOOKBACK = 30     # 订单块回溯上限（根）
DIR_PIN_MIN_N = 30   # 判定「dir_ 被掩码钉死」的最小命中数（低于此视为稀疏伪钉死）


# ============================ 向量化原语（自持，复刻口径不 import） ============================

def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w).max(axis=1)
    return out


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w).min(axis=1)
    return out


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """NaN 容忍滑窗和（窗口内 NaN 按 0 计）。"""
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        xs = np.where(np.isnan(x), 0.0, x)
        out[w - 1:] = np.nansum(sliding_window_view(xs, w), axis=1)
    return out


def _roll_nanmean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        with np.errstate(invalid="ignore"):
            out[w - 1:] = np.nanmean(sliding_window_view(x, w), axis=1)
    return out


def _roll_nanstd(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        with np.errstate(invalid="ignore"):
            out[w - 1:] = np.nanstd(sliding_window_view(x, w), axis=1)
    return out


def _prev(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[1:] = x[:-1]
    return out


def _prev_k(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) > k:
        out[k:] = x[:-k]
    return out


def _roll_pctile(x: np.ndarray, w: int, chunk: int = 512) -> np.ndarray:
    """当前值在过去 w 根（不含当前）中的分位排名 [0,1]。分块控内存。"""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= w:
        return out
    xs = _prev(x)
    sv = sliding_window_view(xs, w)
    for s in range(w, n, chunk):
        e = min(s + chunk, n)
        win = sv[s - w + 1: e - w + 1]
        cur = x[s:e, None]
        with np.errstate(invalid="ignore"):
            out[s:e] = np.nanmean(win < cur, axis=1)
    return out


# ============================ 摆动状态机（唯一的 Python 循环） ============================

def _fractal_raw(x: np.ndarray, k: int, want_max: bool) -> np.ndarray:
    """局部极值的「原始」位置标记（未做右侧确认偏移）。

    要求窗口 [i-k, i+k] 内的极值首次出现在中心：want_max 取 argmax==k，
    want_min 取 argmin==k。 ties 归更早的那根，保证确定性。
    """
    n = len(x)
    out = np.zeros(n, dtype=bool)
    span = 2 * k + 1
    if n < span:
        return out
    w = sliding_window_view(x, span)
    idx = w.argmax(axis=1) if want_max else w.argmin(axis=1)
    ext = w.max(axis=1) if want_max else w.min(axis=1)
    ok = (idx == k) & (ext == x[k: n - k])
    out[k: n - k] = ok
    return out


@dataclass
class SwingState:
    """一次性前向扫描产出的摆动/结构派生列（严格只用 <=t 信息）。"""

    n: int
    swing_hi: np.ndarray = field(default=None)      # 确认位（已右移 SWING_SHIFT）
    swing_lo: np.ndarray = field(default=None)
    last_shi: np.ndarray = field(default=None)      # 最近确认摆动高（含 t）
    last_slo: np.ndarray = field(default=None)
    last_shi_pre: np.ndarray = field(default=None)  # 仅用 <=t-1 确认的摆动（破位类用）
    last_slo_pre: np.ndarray = field(default=None)
    last_shi_age: np.ndarray = field(default=None)
    last_slo_age: np.ndarray = field(default=None)
    trend_dir: np.ndarray = field(default=None)
    bos_up: np.ndarray = field(default=None)
    bos_dn: np.ndarray = field(default=None)
    choch_up: np.ndarray = field(default=None)
    choch_dn: np.ndarray = field(default=None)
    ob_bull_touch: np.ndarray = field(default=None)
    ob_bear_touch: np.ndarray = field(default=None)
    sr_brk_retest_up: np.ndarray = field(default=None)
    sr_brk_retest_dn: np.ndarray = field(default=None)
    qms_up: np.ndarray = field(default=None)
    qms_dn: np.ndarray = field(default=None)
    wfm_dn: np.ndarray = field(default=None)
    wfm_up: np.ndarray = field(default=None)


def _swings(h, l, c, dir_, rng, atr_abs, close_loc, k: int = SWING_K) -> SwingState:
    n = len(h)
    st = SwingState(n=n)
    raw_hi = _fractal_raw(h, k, True)
    raw_lo = _fractal_raw(l, k, False)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    if n > 2 * k:
        sh[k:] = raw_hi[: n - k]      # 极值出现在 t-k，右侧 k 根走完才在 t 确认
        sl[k:] = raw_lo[: n - k]
    st.swing_hi, st.swing_lo = sh, sl

    out = {kk: np.zeros(n, dtype=bool) for kk in
           ("bos_up", "bos_dn", "choch_up", "choch_dn", "ob_bull_touch", "ob_bear_touch",
            "sr_brk_retest_up", "sr_brk_retest_dn", "qms_up", "qms_dn", "wfm_up", "wfm_dn")}
    last_shi = np.full(n, np.nan)
    last_slo = np.full(n, np.nan)
    last_shi_pre = np.full(n, np.nan)
    last_slo_pre = np.full(n, np.nan)
    last_shi_age = np.full(n, np.inf)
    last_slo_age = np.full(n, np.inf)
    trend = np.zeros(n, dtype=np.int64)

    hh_hist: list[tuple[int, float]] = []
    ll_hist: list[tuple[int, float]] = []
    struct = 0
    prev_above_h = False
    prev_below_l = False
    ob_zone: tuple[float, float] | None = None       # 待回补的看涨订单块
    ob_zone_dn: tuple[float, float] | None = None
    retest_lvl: float | None = None                  # 待回踩的看涨破位水平
    retest_lvl_dn: float | None = None
    qms_lvl: float | None = None                     # 待收回的摆动低（QMS 看涨）
    qms_lvl_dn: float | None = None

    for t in range(n):
        # 1) 先并入本根确认的摆动点（其极值位于 t-k / t-k 处）
        if sh[t]:
            hh_hist.append((t, float(h[t - k])))
            if len(hh_hist) > 4:
                hh_hist.pop(0)
            retest_lvl = None            # 结构刷新后旧的"待回踩"失效
        if sl[t]:
            ll_hist.append((t, float(l[t - k])))
            if len(ll_hist) > 4:
                ll_hist.pop(0)
            retest_lvl_dn = None
        if hh_hist:
            last_shi[t] = hh_hist[-1][1]
            last_shi_age[t] = t - hh_hist[-1][0]
        if ll_hist:
            last_slo[t] = ll_hist[-1][1]
            last_slo_age[t] = t - ll_hist[-1][0]
        # 2) 趋势方向：HH+HL / LH+LL（需各两个确认摆动）
        if len(hh_hist) >= 2 and len(ll_hist) >= 2:
            dh = hh_hist[-1][1] - hh_hist[-2][1]
            dl = ll_hist[-1][1] - ll_hist[-2][1]
            trend[t] = 1 if (dh > 0 and dl > 0) else (-1 if (dh < 0 and dl < 0) else 0)

        # 3) 破位判定使用「严格早于本根已确认」的摆动水平，避免用当根确认的摆动自我破位
        lh = hh_hist[-1][1] if (hh_hist and hh_hist[-1][0] < t) else np.nan
        ls = ll_hist[-1][1] if (ll_hist and ll_hist[-1][0] < t) else np.nan
        last_shi_pre[t], last_slo_pre[t] = lh, ls
        a = atr_abs[t]
        at = a if (np.isfinite(a) and a > 0) else 0.0

        above_h = bool(np.isfinite(lh) and c[t] > lh)
        below_l = bool(np.isfinite(ls) and c[t] < ls)
        bu = above_h and not prev_above_h
        bd = below_l and not prev_below_l
        out["bos_up"][t] = bu
        out["bos_dn"][t] = bd
        if bu:
            out["choch_up"][t] = (struct == -1)
            struct = 1
        if bd:
            out["choch_dn"][t] = (struct == 1)
            struct = -1
        prev_above_h, prev_below_l = above_h, below_l

        # 4) 订单块：突破前最后一根反向 K 的 l..h 区间；首次回到区间内 = mitigation
        if bu:
            for j in range(t - 1, max(-1, t - 1 - OB_LOOKBACK), -1):
                if dir_[j] < 0:
                    ob_zone = (float(l[j]), float(h[j]))
                    break
        if bd:
            for j in range(t - 1, max(-1, t - 1 - OB_LOOKBACK), -1):
                if dir_[j] > 0:
                    ob_zone_dn = (float(l[j]), float(h[j]))
                    break
        if ob_zone is not None and l[t] <= ob_zone[1]:
            out["ob_bull_touch"][t] = True
            ob_zone = None
        if ob_zone_dn is not None and h[t] >= ob_zone_dn[0]:
            out["ob_bear_touch"][t] = True
            ob_zone_dn = None

        # 5) 破位回踩（S/R 角色互换）：破高后首次回踩该水平 0.25ATR 内且不收回下方、当根收阳
        if bu and np.isfinite(lh):
            retest_lvl = float(lh)
        if bd and np.isfinite(ls):
            retest_lvl_dn = float(ls)
        if retest_lvl is not None:
            if l[t] <= retest_lvl + 0.25 * at:
                if c[t] > retest_lvl and dir_[t] > 0:
                    out["sr_brk_retest_up"][t] = True
                    retest_lvl = None
                elif c[t] < retest_lvl - 0.25 * at:
                    retest_lvl = None      # 回踩失败，放弃该位
        if retest_lvl_dn is not None:
            if h[t] >= retest_lvl_dn - 0.25 * at:
                if c[t] < retest_lvl_dn and dir_[t] < 0:
                    out["sr_brk_retest_dn"][t] = True
                    retest_lvl_dn = None
                elif c[t] > retest_lvl_dn + 0.25 * at:
                    retest_lvl_dn = None

        # 6) Quasimodo：摆动高存在 → 破摆动低 → 首次收回该摆动低上方（看涨）
        if bd and np.isfinite(ls) and len(hh_hist) >= 1:
            qms_lvl = float(ls)
        if bu and np.isfinite(lh) and len(ll_hist) >= 1:
            qms_lvl_dn = float(lh)
        if qms_lvl is not None and c[t] > qms_lvl:
            out["qms_up"][t] = True
            qms_lvl = None
        if qms_lvl_dn is not None and c[t] < qms_lvl_dn:
            out["qms_dn"][t] = True
            qms_lvl_dn = None

        # 7) 影线失效模型：影线破摆动高但实体收回其下方（看跌）/ 对称
        if np.isfinite(lh) and np.isfinite(h[t]):
            if h[t] > lh and c[t] < lh and close_loc[t] <= 0.5:
                out["wfm_dn"][t] = True
        if np.isfinite(ls) and np.isfinite(l[t]):
            if l[t] < ls and c[t] > ls and close_loc[t] >= 0.5:
                out["wfm_up"][t] = True

    st.last_shi, st.last_slo = last_shi, last_slo
    st.last_shi_pre, st.last_slo_pre = last_shi_pre, last_slo_pre
    st.last_shi_age, st.last_slo_age = last_shi_age, last_slo_age
    st.trend_dir = trend
    for kk, vv in out.items():
        setattr(st, kk, vv)
    return st


# ============================ 命名空间构建 ============================

def build_namespace(kl: Klines, bar_ms: int,
                    parents: dict[str, Klines]) -> dict[str, Any]:
    """构建 expr 求值所需的受控命名空间（全部列均为因果：只引用 <=t）。"""
    o, h, l, c, v, t = kl.o, kl.h, kl.l, kl.c, kl.v, kl.t
    n = len(o)
    ns: dict[str, Any] = {"o": o, "h": h, "l": l, "c": c, "v": v}

    rng = np.where(h > l, h - l, np.nan)
    rng_pct = rng / o
    dir_ = np.sign(c - o)
    with np.errstate(invalid="ignore", divide="ignore"):
        body_r = np.abs(c - o) / rng
        up_r = (h - np.maximum(o, c)) / rng
        lo_r = (np.minimum(o, c) - l) / rng
        close_loc = (c - l) / rng
    ns.update(dir_=dir_, body_r=body_r, up_r=up_r, lo_r=lo_r,
              close_loc=close_loc, rng=rng, rng_pct=rng_pct)

    # 滞后列
    for k in range(1, 5):
        ns[f"o{k}"] = _prev_k(o, k)
        ns[f"h{k}"] = _prev_k(h, k)
        ns[f"l{k}"] = _prev_k(l, k)
        ns[f"c{k}"] = _prev_k(c, k)
        ns[f"dir{k}"] = _prev_k(dir_, k)
        ns[f"body_r{k}"] = _prev_k(body_r, k)
        ns[f"rng_pct{k}"] = _prev_k(rng_pct, k)
    ns["v1"] = _prev_k(v, 1)

    # ATR 与滚动窗口（不含当根的前置极值）
    atr20 = _prev(_roll_nanmean(rng_pct, 20))
    atr_abs = atr20 * o
    ns["atr_abs"] = atr_abs
    hp, lp = _prev(h), _prev(l)
    for w in (1, 10, 16, 20, 50):
        ns[f"hh{w}"] = _roll_max(hp, w)
        ns[f"ll{w}"] = _roll_min(lp, w)
    ns["hh20p"] = ns["hh20"]
    ns["ll20p"] = ns["ll20"]
    ph20, pl20 = ns["hh20"], ns["ll20"]
    ph5, pl5 = _roll_max(hp, 5), _roll_min(lp, 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        ns["rpos20"] = (c - pl20) / np.where(ph20 > pl20, ph20 - pl20, np.nan)
        ns["rpos5"] = (c - pl5) / np.where(ph5 > pl5, ph5 - pl5, np.nan)
        ns["dist_lo_atr5"] = (c - pl5) / atr_abs
        ns["dist_hi_atr10"] = (c - _roll_max(hp, 10)) / atr_abs
    ns["rpos20_hi"] = ns["rpos20"] >= 0.8
    ns["rpos20_lo"] = ns["rpos20"] <= 0.2

    w4h = max(2, 4 * 3_600_000 // bar_ms)
    pm16, pmi16 = _roll_max(hp, w4h), _roll_min(lp, w4h)
    ns["sweep_hi16"] = (h > pm16) & (close_loc <= 0.3)
    ns["sweep_lo16"] = (l < pmi16) & (close_loc >= 0.7)

    # 波动率状态
    w30d = max(200, 30 * DAY_MS // bar_ms)
    atrp = _roll_pctile(rng_pct, w30d)
    ns["atrp"] = atrp
    sd20 = _roll_nanstd(c, 20)
    bbw = 4 * sd20 / np.where(c > 0, c, np.nan)
    ns["sqz"] = _roll_pctile(bbw, 100)
    comp520 = _prev(_roll_nanmean(rng_pct, 5)) / _prev(_roll_nanmean(rng_pct, 20))
    ns["volexp"] = (comp520 >= 1.0) & (_prev(comp520) < 1.0)
    ns["regime_vol_high"] = atrp >= 0.75
    ns["regime_vol_low"] = atrp <= 0.25
    ns["regime_compression"] = np.nan_to_num(comp520, nan=1.0) <= 0.6
    ns["regime_expansion"] = np.nan_to_num(comp520, nan=0.0) >= 1.5

    # Kaufman 效率比（与 features.py efficiency_k 同式）
    abs_dc = np.abs(c - _prev(c))
    for k in (3, 5, 10, 20):
        ck = _prev_k(c, k)
        net = np.where(ck > 0, np.abs(c - ck), np.nan)
        path = _roll_sum(abs_dc, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            ns[f"er{k}"] = np.where((path > 0) & (ck > 0), net / path, np.nan)

    # 量能
    volma20 = _prev(_roll_nanmean(v, 20))
    ns["volma20"] = volma20
    with np.errstate(invalid="ignore", divide="ignore"):
        ns["volr"] = v / volma20
    ns["nr4"] = rng <= _roll_min(rng, 4)
    ns["nr7"] = rng <= _roll_min(rng, 7)

    # 摆动 / 结构状态机
    st = _swings(h, l, c, dir_, rng, atr_abs, close_loc)
    ns["swing_hi"] = st.swing_hi
    ns["swing_lo"] = st.swing_lo
    ns["last_shi"] = st.last_shi
    ns["last_slo"] = st.last_slo
    ns["last_shi_age"] = st.last_shi_age
    ns["last_slo_age"] = st.last_slo_age
    ns["trend_dir"] = st.trend_dir
    ns["bos_up"] = st.bos_up
    ns["bos_dn"] = st.bos_dn
    ns["choch_up"] = st.choch_up
    ns["choch_dn"] = st.choch_dn
    ns["ob_bull_touch"] = st.ob_bull_touch
    ns["ob_bear_touch"] = st.ob_bear_touch
    ns["sr_brk_retest_up"] = st.sr_brk_retest_up
    ns["sr_brk_retest_dn"] = st.sr_brk_retest_dn
    ns["qms_up"] = st.qms_up
    ns["qms_dn"] = st.qms_dn
    ns["wfm_up"] = st.wfm_up
    ns["wfm_dn"] = st.wfm_dn

    with np.errstate(invalid="ignore", divide="ignore"):
        ns["eqh"] = (np.abs(h - st.last_shi_pre) <= 0.15 * atr_abs) & (st.last_shi_age >= 5)
        ns["eql"] = (np.abs(l - st.last_slo_pre) <= 0.15 * atr_abs) & (st.last_slo_age >= 5)
    ns["eqh"] = ns["eqh"] & np.isfinite(st.last_shi_pre)
    ns["eql"] = ns["eql"] & np.isfinite(st.last_slo_pre)

    # FVG（三根式，纯 <=t，天然因果）
    ns["fvg_up"] = l > ns["h2"]
    ns["fvg_dn"] = h < ns["l2"]

    # sweep / reclaim 与威科夫机检变体（20 根区间版）
    brk_hi20 = h > ph20
    brk_lo20 = l < pl20
    shrink = v < 0.7 * volma20
    ns["sweep_rec_up"] = brk_lo20 & (dir_ > 0) & (close_loc >= 0.5)
    ns["sweep_rec_dn"] = brk_hi20 & (dir_ < 0) & (close_loc <= 0.5)
    ns["spring"] = brk_lo20 & (close_loc >= 0.5) & shrink & (dir_ > 0)
    ns["utad"] = brk_hi20 & (close_loc <= 0.5) & (v >= 1.2 * volma20)
    ns["shrink_brk_hi"] = brk_hi20 & shrink
    ns["shrink_brk_lo"] = brk_lo20 & shrink

    # 2B 法则（前一根创 20 根新低/高，当根收回）
    # 注意必须把 l1/h1 自身从比较基准中剔除：_roll_min(lp,20) 含 lp[t]=l[t-1]，
    # 直接写 `l1 < min(...)` 会因自比恒假而永不触发（本行曾为此零命中）。
    l1_is_lo = ns["l1"] < _roll_min(_prev(lp), 20)
    h1_is_hi = ns["h1"] > _roll_max(_prev(hp), 20)
    ns["b2b_up"] = l1_is_lo & (c > ns["l1"]) & (dir_ > 0)
    ns["b2b_dn"] = h1_is_hi & (c < ns["h1"]) & (dir_ < 0)

    # VSA
    ns["effort_noresult"] = (v >= 2.0 * volma20) & (rng <= 0.5 * atr_abs)
    ns["nd_bar"] = (dir_ > 0) & (body_r <= 0.3) & (v <= 0.7 * volma20)
    ns["ns_bar"] = (dir_ < 0) & (body_r <= 0.3) & (v <= 0.7 * volma20)
    ns["stop_vol"] = (v >= 1.5 * volma20) & (lo_r >= 0.4) & (dir_ < 0)

    # 38.2% 回撤机检（近 20 根单向波段 + 触及 38.2% 位 + 当根收回）
    # 波段高点取自严格前窗（不含当根），否则“回踩”会被当根自身新高污染。
    hi20i, lo20i = _roll_max(hp, 20), _roll_min(lp, 20)
    up_span = hi20i - lo20i
    lvl_up = hi20i - 0.382 * up_span
    lvl_dn = lo20i + 0.382 * up_span
    with np.errstate(invalid="ignore"):
        ns["pull382_up"] = ((up_span >= 2 * atr_abs)
                            & (l <= lvl_up + 0.25 * atr_abs) & (l >= lvl_up - 0.25 * atr_abs)
                            & (c > lvl_up) & (dir_ > 0) & (c > (hi20i + lo20i) / 2))
        ns["pull382_dn"] = ((up_span >= 2 * atr_abs)
                            & (h >= lvl_dn - 0.25 * atr_abs) & (h <= lvl_dn + 0.25 * atr_abs)
                            & (c < lvl_dn) & (dir_ < 0) & (c < (hi20i + lo20i) / 2))

    # 多周期共振（父根必须已收盘）
    tag1, tag2 = _parent_tags(bar_ms)
    ns["align_h"] = _align(kl, parents[tag1], bar_ms)
    ns["align_h2"] = _align(kl, parents[tag2], bar_ms)
    ns["align_h4"] = _align(kl, parents.get(_great_grand_tag(bar_ms)), bar_ms) \
        if _great_grand_tag(bar_ms) in parents else np.zeros(n, dtype=bool)
    par_ms = BAR_MS_OF(parents[tag1])      # 从父序列自身取步长（1h 的父是 4h，不在 BAR_MS 表内）
    ns["slot_boundary"] = ((t // bar_ms) % (par_ms // bar_ms)) == 0
    ns["hour"] = ((t // 3_600_000) % 24).astype(np.int64)
    ns["weekday"] = ((t // DAY_MS) + 3) % 7
    ns["funding_slot"] = np.isin(ns["hour"], [0, 8, 16])   # 对齐 features.py:378

    # regime（趋势/震荡）——与 features.py 同式
    n30 = max(100, 30 * DAY_MS // bar_ms)
    c30 = _prev_k(c, n30)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret30 = np.where(c30 > 0, c / c30 - 1.0, np.nan)
    ns["regime_trend_up"] = ret30 > 0.03
    ns["regime_trend_dn"] = ret30 < -0.03
    ns["regime_trend_down"] = ns["regime_trend_dn"]
    rng100 = _prev(_roll_nanmean(rng_pct, 100))
    c100 = _prev_k(c, 100)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret100 = np.where(c100 > 0, np.abs(c / c100 - 1.0), np.nan)
    ns["regime_trending"] = ret100 >= 4 * rng100
    ns["regime_ranging"] = ret100 <= rng100

    ns["abs"] = np.abs
    ns["max"] = np.maximum
    ns["min"] = np.minimum
    ns["_n"] = n
    return ns


def _parent_tags(bar_ms: int) -> tuple[str, str]:
    """本周期的一级/二级父周期标签（永不返回 <= 本周期者，避开 aggregate_to 的 n_sub<=1 raise）。"""
    if bar_ms == BAR_MS["5m"]:
        return "1h", "4h"
    if bar_ms == BAR_MS["15m"]:
        return "1h", "4h"
    return "4h", "1d"


def _great_grand_tag(bar_ms: int) -> str:
    return "1d" if bar_ms in (BAR_MS["5m"], BAR_MS["15m"]) else ""


def _align(kl: Klines, par: Klines | None, bar_ms: int) -> np.ndarray:
    """本根方向是否与「已收盘的父根」方向一致（父根 end_time <= 本根 start_time）。"""
    n = len(kl)
    if par is None or len(par) < 2:
        return np.zeros(n, dtype=bool)
    par_ms = BAR_MS_OF(par)
    pdir = np.sign(par.c - par.o)
    j = np.searchsorted(par.t, kl.t - par_ms, side="right") - 1
    ok = j >= 0
    hd = np.where(ok, pdir[np.clip(j, 0, len(par) - 1)], np.nan)
    return np.where(ok, np.sign(kl.c - kl.o) * hd > 0, False)


def BAR_MS_OF(par: Klines) -> int:
    return int(par.t[1] - par.t[0]) if len(par.t) > 1 else 3_600_000


def build_parents(k5_csv: str) -> dict[str, Klines]:
    """从 5m CSV 逐级聚合出 1h / 4h / 1d（全部走 aggregate_to，桶内根数必须齐全）。"""
    kl5 = load_klines_csv(k5_csv, BAR_MS["5m"])
    h1 = aggregate_to(kl5, BAR_MS["1h"])
    h4 = aggregate_to(h1, 4 * 3_600_000)
    d1 = aggregate_to(h4, DAY_MS)
    return {"5m": kl5, "15m": aggregate_to(kl5, BAR_MS["15m"]),
            "1h": h1, "4h": h4, "1d": d1}


# ============================ 表达式求值 ============================

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_FUNCS = {"abs", "max", "min"}
_LITERAL_NAMES = {"True", "False", "None"}


def eval_expr(expr: str, ns: dict[str, Any]) -> np.ndarray:
    """在受控命名空间内求值逐根布尔判据。任何缺列/类型错都直接抛出（不静默降级）。

    仅暴露 `abs/max/min` 的 **向量化** 版本（np.abs/maximum/minimum）：注册表里
    `max(o1, c1)` 这类写法若用 Python 内置 max 作用于数组会抛 ambiguous-truth。
    `True/False/None` 在 3.8+ 是 ast.Constant 而非 Name，但正则会把它们当标识符，须白名单。
    """
    bad = [nm for nm in _IDENT_RE.findall(expr)
           if nm not in ns and nm not in _ALLOWED_FUNCS and nm not in _LITERAL_NAMES]
    if bad:
        raise KeyError(f"expr 引用未定义列：{sorted(set(bad))}；expr={expr}")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = eval(expr, {"__builtins__": {}}, dict(ns))  # noqa: S307 —— 判据来自本仓库冻结注册表
    arr = np.asarray(out)
    if arr.dtype != np.bool_:
        raise TypeError(f"expr 未返回布尔数组：{expr} → dtype={arr.dtype} shape={arr.shape}")
    if arr.ndim == 0:
        raise TypeError(f"expr 返回标量（应为逐根数组）：{expr}")
    if len(arr) != ns["_n"]:
        raise ValueError(f"expr 长度不符：{expr} → {len(arr)} vs {ns['_n']}")
    return arr



# ============================ 掩码展开 + 语义去重 ============================

def expand_masks(reg: list[dict], ns: dict[str, Any], i1: int) -> list[dict]:
    """注册表条目 × context_gate → 掩码集合，并按 hit_sig 语义去重。

    返回 row 列表：含 pattern_id / mask_name / expr / _mask / aliases / n_* / dir_*。
    """
    n = ns["_n"]
    rows: list[dict] = []
    for pat in reg:
        base_expr = pat["expr"]
        gates = [""] + [g for g in pat.get("context_gate", [])]
        for gi, gate in enumerate(gates):
            full = base_expr if not gate else f"({base_expr}) & ({gate})"
            try:
                mask = eval_expr(full, ns)
            except Exception as ex:  # 不做静默跳过：记录后由调用方判定是否中止
                raise RuntimeError(f"{pat['id']} 判据求值失败（gate={gate or '—'}）：{ex}") from ex
            suffix = "" if gi == 0 else f"|{gate}"
            rows.append({
                "pattern_id": pat["id"],
                "mask_key": pat["id"] + suffix,
                "gate": gate,
                "expr_full": full,
                "target_family": pat["target_family"],
                "bet_side": pat["bet_side"],
                "_mask": mask,
                "hit_sig": _hit_sig(mask, i1),
                "n_all": int(mask.sum()),
                "n_discovery": int(mask[:i1].sum()),
                "aliases": [],
            })
    # hit_sig 去重：同掩码 **且同声明口径** 才合并，其余并入 aliases。
    # 去重键必须带 target_family/bet_side：两条判据若产出同一掩码但押注方向不同，
    # 它们的主判决目标不同，合并会把其中一条的结果静默算到另一条头上（错误归因）。
    by_sig: dict[tuple, dict] = {}
    kept: list[dict] = []
    for r in rows:
        key = (r["hit_sig"], r["target_family"], r["bet_side"])
        prev = by_sig.get(key)
        if prev is not None and np.array_equal(prev["_mask"], r["_mask"]):
            prev["aliases"].append(r["mask_key"])
            continue
        by_sig[key] = r
        kept.append(r)
    return kept


def _hit_sig(mask: np.ndarray, upto: int) -> str:
    """发现段命中指纹（复刻 discovery/combo_search.py 的 hit_sig 口径）。"""
    packed = np.packbits(mask[:upto])
    return hashlib.sha256(packed.tobytes()).hexdigest()[:16]


# ============================ 因果性自检 ============================

def causality_selfcheck(kl: Klines, bar_ms: int, parents: dict[str, Klines],
                        reg: list[dict], tail: int = 200,
                        seed: int = 20260830) -> dict:
    """扰动最后 tail 根的 OHLCV，断言 t < n-tail 处所有掩码逐位不变。

    复刻 tests/test_discovery_pipeline.py:60-76 的扰动法（那里针对 FeatureMatrix，
    这里针对本引擎的掩码集合——覆盖面更宽：含状态机派生列）。
    """
    t0 = time.time()
    n = len(kl)
    ns0 = build_namespace(kl, bar_ms, parents)
    i1 = int(n * 0.6)
    masks0 = {r["mask_key"]: r["_mask"] for r in expand_masks(reg, ns0, i1)}

    rng = np.random.default_rng(seed)
    kl_p = Klines(t=kl.t.copy(), o=kl.o.copy(), h=kl.h.copy(), l=kl.l.copy(),
                  c=kl.c.copy(), v=kl.v.copy(), cont=kl.cont.copy())
    sl = slice(n - tail, n)
    for arr in (kl_p.o, kl_p.h, kl_p.l, kl_p.c):
        fac = rng.uniform(0.9, 1.1, tail)
        arr[sl] = arr[sl] * fac
    kl_p.h[sl] = np.maximum.reduce([kl_p.o[sl], kl_p.h[sl], kl_p.l[sl], kl_p.c[sl]])
    kl_p.l[sl] = np.minimum.reduce([kl_p.o[sl], kl_p.h[sl], kl_p.l[sl], kl_p.c[sl]])
    kl_p.v[sl] = kl_p.v[sl] * rng.uniform(0.5, 1.5, tail)
    parents_p = _perturb_parents(parents, rng, tail)

    ns1 = build_namespace(kl_p, bar_ms, parents_p)
    masks1 = {r["mask_key"]: r["_mask"] for r in expand_masks(reg, ns1, i1)}

    leaked = []
    for key, m0 in masks0.items():
        m1 = masks1[key]
        cut = min(n - tail, i1)
        if not np.array_equal(m0[:cut], m1[:cut]):
            diff = np.flatnonzero(m0[:cut] ^ m1[:cut])
            leaked.append({"mask": key, "n_diff": int(len(diff)),
                           "first_idx": int(diff[0]) if len(diff) else None})
    return {
        "passed": not leaked,
        "tail_perturbed": tail,
        "checked_prefix": int(min(n - tail, i1)),
        "n_masks": len(masks0),
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
        "leaked": leaked,
    }


def _perturb_parents(parents: dict[str, Klines], rng, tail: int) -> dict[str, Klines]:
    """对父周期同样扰动其尾部（否则父根未受扰，align_* 检不出泄漏）。"""
    out: dict[str, Klines] = {}
    for tag, par in parents.items():
        npar = len(par)
        m = min(tail, npar)
        kp = Klines(t=par.t.copy(), o=par.o.copy(), h=par.h.copy(), l=par.l.copy(),
                    c=par.c.copy(), v=par.v.copy(), cont=par.cont.copy())
        sl = slice(npar - m, npar)
        fac = rng.uniform(0.9, 1.1, m)
        for arr in (kp.o, kp.h, kp.l, kp.c):
            arr[sl] = arr[sl] * fac
        kp.h[sl] = np.maximum.reduce([kp.o[sl], kp.h[sl], kp.l[sl], kp.c[sl]])
        kp.l[sl] = np.minimum.reduce([kp.o[sl], kp.h[sl], kp.l[sl], kp.c[sl]])
        out[tag] = kp
    return out


# ============================ 注册表预检（冻结前的逐条求值体检） ============================

def preflight(reg: list[dict], ns: dict[str, Any]) -> dict:
    """逐条求值 expr 与 context_gate，收集全部缺陷而非首个即中止。

    注册表进入冻结态前必须让 `errors` 与 `zero_hits` 归零/被解释：
      - errors      ：未定义列、`&` 与比较运算符优先级导致的类型错、非布尔返回
      - zero_hits   ：掩码在全体样本上零命中（多为恒假比较，如 bool >= 2.0）
      - rare_hits   ：命中数 < 60（低于 DEFAULTS['n_min_holdout']，只能出描述性数字）
    """
    n = ns["_n"]
    errors, rows = [], []
    for pat in reg:
        pid, gates = pat["id"], [""] + list(pat.get("context_gate", []))
        for gate in gates:
            full = pat["expr"] if not gate else f"({pat['expr']}) & ({gate})"
            try:
                mask = eval_expr(full, ns)
            except Exception as ex:
                errors.append({"id": pid, "gate": gate, "expr": full,
                               "error": f"{type(ex).__name__}: {ex}"})
                continue
            rows.append({"id": pid, "gate": gate, "n_hit": int(mask.sum()),
                         "rate": round(float(mask.sum()) / n, 6)})
    hit = {r["id"] for r in rows if r["n_hit"] > 0}
    zero_hits = [r for r in rows if r["n_hit"] == 0]
    rare_hits = [r for r in rows if 0 < r["n_hit"] < 60]
    return {
        "n": n, "n_patterns": len(reg), "n_exprs_evaluated": len(rows) + len(errors),
        "errors": errors, "zero_hits": zero_hits, "rare_hits": rare_hits,
        "patterns_with_gate": sorted({r["id"] for r in rows if r["gate"]}),
        "coverage": {"patterns_with_hits": len(hit), "patterns_zero": len(reg) - len(hit)},
    }


# ============================ 押注方向一致性审计 ============================

def lens_of(mask: np.ndarray, dir_: np.ndarray, bet_side: str, target_family: str) -> dict:
    """双 lens 归属的 **唯一口径源**（engine 是口径所有者；report 必须复用，不得另写一套）。

    abs lens（dirup_/dirdn_）与 rel lens（continuation_/reversal_）何时给出同一数字：
      掩码内信号根自身方向 dir_ 全同号 → 两 lens 只是同一问题的两种命名（等价）；
      dir_ 混号 → 两 lens 回答不同问题（不等价，绝非冗余）。

    两条纪律（都是本函数存在的理由）：
    1. **主判决 lens 只由注册表声明决定**（bet_side=up/down → abs；none → rel），
       绝不因数据稀疏或方向分布而漂移——否则同一形态会在 5m 走 abs、在 1h 走 rel，
       跨周期数字不可比，且「拿哪一列出判决」本身变成数据窥视的产物。
    2. **dir_pinned 带最小样本门槛 DIR_PIN_MIN_N**：稀疏掩码里几个命中恰好同号会被
       误判为钉死（实测 1h 的等价占比虚高到 0.52，而 5m 全量仅 2/90）。门槛之下
       记为 THIN，只作提示不作结论；THIN 层的「声明与实证反向」记为 THIN_DISAGREE
       而非 CONFLICT（小样本反号是噪声，不构成注册表腐烂）。
    """
    pos = int((mask & (dir_ > 0)).sum())
    neg = int((mask & (dir_ < 0)).sum())
    nn = pos + neg
    pinned = "up" if (nn and pos == nn) else "down" if (nn and neg == nn) else None
    implied = None
    if pinned is not None:
        implied = pinned if target_family == "continuation" else ("down" if pinned == "up" else "up")
    thin = 0 < nn < DIR_PIN_MIN_N
    primary = "abs" if bet_side in ("up", "down") else "rel"   # ← 只由声明决定
    if nn == 0:
        status = "ZERO_DIR"
    elif bet_side == "none":
        status = "REL_PRIMARY"
    elif implied is None:
        status = "ABS_MIXED_DIR"
    elif implied == bet_side:
        status = "ABS_OK_THIN_PIN" if thin else "ABS_OK"
    else:
        status = (f"THIN_DISAGREE(declared={bet_side},implies={implied},n={nn})" if thin
                  else f"CONFLICT(declared={bet_side},implies={implied})")
    return {"primary_lens": primary, "dir_pinned": pinned, "implied_bet_side": implied,
            "n_dir": nn, "n_dir_pos": pos, "n_dir_neg": neg, "lens_status": status,
            "thin_pin": thin,
            "lenses_equivalent": bool(pinned is not None and implied == bet_side and not thin)}


def bet_audit(reg: list[dict], ns: dict[str, Any], i1: int) -> list[dict]:
    """逐条审计 bet_side / target_family 在真实掩码上的自洽性（注册表冻结闸门）。

    双 lens 口径见注册表 semantics.bet_lens；判定逻辑全部下沉到 `lens_of`（唯一口径源），
    本函数只负责逐条跑掩码并汇总成可打印的审计行。声明为绝对押注（bet_side=up/down）
    且 dir_ 被足够样本钉死的形态，若 implied 与 bet_side 不一致即为注册表腐烂（CONFLICT）；
    声明为 none 的形态只有相对押注，主判决走 rel。
    """
    dir_ = ns["dir_"]
    out = []
    for pat in reg:
        mask = eval_expr(pat["expr"], ns) & (dir_ != 0)
        a = lens_of(mask, dir_, pat["bet_side"], pat["target_family"])
        out.append({"id": pat["id"], "name_cn": pat["name_cn"],
                    "bet_side": pat["bet_side"], "target_family": pat["target_family"],
                    "primary_lens": a["primary_lens"],
                    "n": a["n_dir"], "n_dir_pos": a["n_dir_pos"], "n_dir_neg": a["n_dir_neg"],
                    "dir_pinned": a["dir_pinned"], "implied_bet_side": a["implied_bet_side"],
                    "lenses_equivalent": a["lenses_equivalent"], "thin_pin": a["thin_pin"],
                    "audit": a["lens_status"]})
    return out


# ============================ 合成序列单测（右移正确性） ============================

def test_swing_shift() -> dict:
    """合成序列验证摆动确认位的右移语义（Step 3 的最易出错处）。

    构造：一根真正的分形高在 p_hi、一根真正的分形低在 p_lo、**最后一根是全场尖峰**。
    断言：
      1. p_hi 的原始分形成立，且确认位恰在 p_hi + k（而不是 p_hi 本身）；
      2. 末根为峰值时 swing_hi 不得为 True（右侧无确认根 → 任何 True 都是未来函数）；
      3. swing_hi/swing_lo 为 True 的位置必须能回溯到一个窗口完整的原始极值。
    """
    n, k = 100, SWING_K
    p_hi, p_lo = 80, 86
    base = 60_000.0
    h = np.full(n, base + 100.0)
    l = np.full(n, base)
    h[p_hi] = base + 500.0
    for d in range(1, k + 1):                    # 以 p_hi 为中心的尖
        h[p_hi - d] = base + 300.0 - 50.0 * d
        h[p_hi + d] = base + 300.0 - 60.0 * d
    l[p_lo] = base - 500.0
    for d in range(1, k + 1):
        l[p_lo - d] = base - 300.0 + 50.0 * d
        l[p_lo + d] = base - 300.0 + 60.0 * d
    h[-1] = base + 10_000.0                      # 末根全场最高（不可确认）
    o = np.full(n, base + 40.0)
    c = np.full(n, base + 60.0)
    c[-1] = base + 9_000.0
    v = np.ones(n)
    kl = Klines(t=(np.arange(n) * BAR_MS["5m"]).astype(np.int64), o=o, h=h, l=l, c=c,
                v=v, cont=np.r_[False, np.ones(n - 1, dtype=bool)])
    ns = build_namespace(kl, BAR_MS["5m"], {"1h": kl, "4h": kl})
    raw_hi = _fractal_raw(h, k, True)
    sh, sl = ns["swing_hi"], ns["swing_lo"]
    # 任何确认位都必须对应一个窗口完整的原始极值（右移 k 根）
    aligned_hi = np.zeros(n, dtype=bool)
    aligned_hi[k:] = raw_hi[: n - k]
    return {
        "k": k,
        "raw_hi_at_p": bool(raw_hi[p_hi]),
        "confirmed_exactly_at_p_plus_k": bool(sh[p_hi + k]) and not bool(sh[p_hi]),
        "last_bar_spike_unconfirmed": not bool(sh[-1]),
        "lo_confirmed_at_p_plus_k": bool(sl[p_lo + k]),
        "shift_consistent": bool(np.array_equal(sh, aligned_hi)),
        "ok": (bool(raw_hi[p_hi]) and bool(sh[p_hi + k]) and not bool(sh[p_hi])
               and not bool(sh[-1]) and bool(np.array_equal(sh, aligned_hi))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="裸K形态引擎自检")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="逐条求值注册表并汇总缺陷")
    ap.add_argument("--tf", default="5m", choices=list(BAR_MS))
    args = ap.parse_args()
    reg = json.load(open(REGISTRY, encoding="utf-8"))["hypotheses"]

    res = {"swing_shift_unit": test_swing_shift()}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res["swing_shift_unit"]["ok"]:
        print("[FAIL] 摆动右移单测未通过", file=sys.stderr)
        return 1

    if args.preflight or args.selfcheck:
        parents = build_parents(os.path.join(ROOT, "output", "klines_5m_720d.csv"))
        tag = args.tf
        kl = parents[tag]
        ns = build_namespace(kl, BAR_MS[tag], parents)
        i1 = int(len(kl) * 0.6)

    if args.preflight:
        pf = preflight(reg, ns)
        print("\n[%s] preflight: %s" % (tag, json.dumps(
            {k: v for k, v in pf.items() if k not in ("errors", "zero_hits", "rare_hits")},
            ensure_ascii=False)))
        for e in pf["errors"]:
            print("  ERROR", json.dumps(e, ensure_ascii=False))
        for z in pf["zero_hits"]:
            print("  ZERO ", json.dumps(z, ensure_ascii=False))
        for rz in pf["rare_hits"]:
            print("  RARE  ", json.dumps(rz, ensure_ascii=False))
        if pf["errors"]:
            return 1

    if args.selfcheck:
        audit = bet_audit(reg, ns, i1)
        from collections import Counter
        cnt = Counter(a["audit"].split("(")[0] for a in audit)
        print(f"\n[{tag}] bet 审计：{dict(cnt)}")
        print(f"  primary_lens 分布：{dict(Counter(a['primary_lens'] for a in audit))}")
        print(f"  两 lens 等价（dir-pinned）形态数："
              f"{sum(1 for a in audit if a['lenses_equivalent'])}")
        conflict = [a for a in audit if a["audit"].startswith("CONFLICT")]
        thin_dis = [a for a in audit if a["audit"].startswith("THIN_DISAGREE")]
        print(f"  稀疏伪钉死（THIN，n<{DIR_PIN_MIN_N}）提示数：{len(thin_dis)}")
        # 安静打印仅留「需要人看一眼」的状态：ABS_OK* / ABS_MIXED_DIR（dir_ 混号是常态，
        # 不是缺陷）不打印，避免 90 条里刷出几十个无信息行。
        for a in audit:
            if not a["audit"].startswith(("ABS_OK", "ABS_MIXED_DIR")):
                print("  ", json.dumps(a, ensure_ascii=False))
        if thin_dis:
            print("  [WARN] 以下为小样本方向反号（不阻断冻结，但须在报告披露）："
                  + ", ".join(a["id"] for a in thin_dis))
        if conflict:
            print(f"[FAIL] {len(conflict)} 条 bet 声明与掩码方向不自洽", file=sys.stderr)
            return 3
        cs = causality_selfcheck(kl, BAR_MS[tag], parents, reg)
        print("\n因果自检：", json.dumps({k: v for k, v in cs.items() if k != "leaked"},
                                       ensure_ascii=False))
        for le in cs["leaked"][:20]:
            print("   泄漏", le)
        return 0 if cs["passed"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
