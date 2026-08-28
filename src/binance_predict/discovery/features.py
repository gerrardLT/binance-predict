"""特征层：向量化特征矩阵构建（复刻批 183 特征 + 发散批新维度）。

防泄漏铁律：所有滚动特征只用「不含当前根」的前置窗口（prev_* 系列先右移一根）。
特征命名/家族对齐 output/kline_discovery_gen2_5m_720d/feature_manifest.csv，
保证与既有 720d 产物可对比；发散批以新名字新增，不改旧名语义。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .data import Klines, aggregate_to


@dataclass
class FeatureMatrix:
    names: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    cols: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.names)

    def add(self, name: str, family: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        if arr.dtype == np.bool_:
            dtype = "bool"
        elif arr.dtype in (np.int8, np.int16, np.int32, np.int64):
            dtype = "int64"
        else:
            dtype = "float64"
            arr = arr.astype(np.float64)
        if name in self.cols:
            raise ValueError(f"特征重名: {name}")
        self.names.append(name)
        self.families.append(family)
        self.dtypes.append(dtype)
        self.cols[name] = arr

    def manifest_rows(self) -> list[tuple[str, str, str]]:
        return list(zip(self.names, self.families, self.dtypes))


# ---------------- 向量化原语 ----------------

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
    """NaN 容忍滑窗和（nansum 口径：窗口内 NaN 按 0 计——cumsum 会被首个 NaN 毒化整条后缀）。"""
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        xs = np.where(np.isnan(x), 0.0, x)
        with np.errstate(invalid="ignore"):
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
    """右移一根（前置窗口防泄漏原语）。"""
    out = np.full(len(x), np.nan)
    out[1:] = x[:-1]
    return out


def _roll_pctile(x: np.ndarray, w: int, chunk: int = 256) -> np.ndarray:
    """当前值在过去 w 根（不含当前）中的分位排名 [0,1]（分块避免大内存）。"""
    n = len(x)
    out = np.full(n, np.nan)
    if n <= w:
        return out
    xs = _prev(x)
    sv = sliding_window_view(xs, w)
    for s in range(w, n, chunk):
        e = min(s + chunk, n)
        win = sv[s - w + 1: e - w + 1]  # 位置 i 的窗口 = x[i-w .. i-1]（过去 w 根）
        cur = x[s:e, None]
        with np.errstate(invalid="ignore"):
            out[s:e] = np.nanmean(win < cur, axis=1)
    return out


def _streak(dir_: np.ndarray) -> np.ndarray:
    """连续同向根数（含当前根）向量化。"""
    n = len(dir_)
    same = np.zeros(n, dtype=bool)
    same[1:] = (dir_[1:] == dir_[:-1]) & (dir_[1:] != 0)
    starts = np.flatnonzero(~same)
    grp = np.searchsorted(starts, np.arange(n), side="right") - 1
    return (np.arange(n) - starts[grp]).astype(np.int64)


def _bars_since(b: np.ndarray) -> np.ndarray:
    """距上次 True 的根数（从未发生 → inf）。"""
    idx = np.where(b, np.arange(len(b)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, np.arange(len(b)) - last, np.inf)
    return out.astype(np.float64)


# ---------------- 周期路径特征（5m 子根 → 周期内形态） ----------------

def _cycle_path(kl: Klines, sub: Klines, cycle_ms: int) -> dict[str, np.ndarray]:
    """以 cycle_ms 为周期，用 sub（更细周期）的 3 根路径构造形态特征并广播回 kl。

    path3 = 周期内 3 根子 K 的 U/D/F 方向串；last5_dir = 末根方向；
    er = 效率路径 |净位移|/路径长。仅取子根齐全的完整周期。
    """
    n = len(kl.t)
    path = np.full(n, -1, dtype=np.int8)  # -1=缺失 0=FFF.. 编码见下
    last5 = np.full(n, np.nan)
    er = np.full(n, np.nan)
    sub_cyc = sub.t // cycle_ms
    uniq, first = np.unique(sub_cyc, return_index=True)
    counts = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(counts, np.searchsorted(uniq, sub_cyc), 1)
    full3 = counts == 3
    cyc_of_kl = kl.t // cycle_ms
    pos = np.searchsorted(uniq, cyc_of_kl)
    ok_kl = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == cyc_of_kl) & full3[np.minimum(pos, len(full3) - 1)]
    enc = {"UUU": 0, "UUD": 1, "UDU": 2, "UDD": 3, "DUU": 4, "DUD": 5, "DDU": 6, "DDD": 7}
    for j in np.flatnonzero(full3):
        i0 = first[j]
        d = np.sign(sub.c[i0: i0 + 3] - sub.o[i0: i0 + 3])
        s = "".join("U" if x > 0 else ("D" if x < 0 else "F") for x in d)
        if "F" in s:
            continue
        pts = np.concatenate([[sub.o[i0]], sub.c[i0: i0 + 3]])
        plen = float(np.abs(np.diff(pts)).sum())
        mask = ok_kl & (pos == j)
        path[mask] = enc.get(s, -1)
        last5[mask] = d[-1]
        er[mask] = abs(pts[3] - pts[0]) / plen if plen > 0 else np.nan
    return {"path_code": path, "last_dir": last5, "er_path": er}


# ---------------- 主构建入口 ----------------

def atr_series(kl: Klines, w: int = 20) -> np.ndarray:
    """ATR 绝对值序列（前 w 根 range% 均值 × open，不含当前根）——供目标层 MFE/MAE 归一。"""
    rng_pct = np.where(kl.h > kl.l, kl.h - kl.l, np.nan) / kl.o
    atr = _prev(_roll_nanmean(rng_pct, w))
    return atr * kl.o


def build_feature_matrix(kl: Klines, bar_ms: int, k5: Klines | None = None) -> FeatureMatrix:
    """全量特征矩阵。k5：5m 数据（tf=15m 时供周期路径特征；tf=5m 时传 kl 自身）。"""
    fm = FeatureMatrix()
    N = len(kl)
    o, h, l, c, v, t = kl.o, kl.h, kl.l, kl.c, kl.v, kl.t

    rng_pct = np.where(h > l, h - l, np.nan) / o
    dir_ = np.sign(c - o)
    body_pct = np.abs(c - o) / o
    rng = np.where(h > l, h - l, np.nan)

    # ---- geometry（ATR = 前 20 根 range% 的均值，不含当前根）----
    atr20 = _prev(_roll_nanmean(rng_pct, 20))
    atr_abs = atr20 * o
    fm.add("body_ratio", "geometry", body_pct / rng_pct)
    fm.add("upper_wick_ratio", "geometry", (h - np.maximum(o, c)) / rng_pct)
    fm.add("lower_wick_ratio", "geometry", (np.minimum(o, c) - l) / rng_pct)
    fm.add("close_loc", "geometry", (c - l) / rng_pct)
    fm.add("open_loc", "geometry", (o - l) / rng_pct)
    fm.add("body_atr", "geometry", body_pct / atr20)
    fm.add("range_atr", "geometry", rng_pct / atr20)
    fm.add("upper_wick_atr", "geometry", (h - np.maximum(o, c)) / o / atr20)
    fm.add("lower_wick_atr", "geometry", (np.minimum(o, c) - l) / o / atr20)
    fm.add("gap_atr", "geometry", (o - _prev(c)) / _prev(c) / atr20)
    body_r = body_pct / rng_pct
    up_r = (h - np.maximum(o, c)) / rng_pct
    lo_r = (np.minimum(o, c) - l) / rng_pct
    fm.add("doji", "geometry", np.nan_to_num(body_r, nan=1.0) <= 0.1)
    fm.add("hammer_geometry", "geometry",
           (np.nan_to_num(lo_r, nan=-1) >= 0.6) & (np.nan_to_num(body_r, nan=1) <= 0.3)
           & (np.nan_to_num(up_r, nan=1) <= 0.15))
    fm.add("shooting_star_geometry", "geometry",
           (np.nan_to_num(up_r, nan=-1) >= 0.6) & (np.nan_to_num(body_r, nan=1) <= 0.3)
           & (np.nan_to_num(lo_r, nan=1) <= 0.15))
    fm.add("inside_bar", "geometry", (h <= _prev(h)) & (l >= _prev(l)))
    fm.add("outside_bar", "geometry", (h > _prev(h)) & (l < _prev(l)))
    pmin, pmax = np.minimum(_prev(o), _prev(c)), np.maximum(_prev(o), _prev(c))
    fm.add("bullish_engulfing", "other", (dir_ > 0) & (o <= pmin) & (c > pmax))
    fm.add("bearish_engulfing", "other", (dir_ < 0) & (o >= pmax) & (c < pmin))
    # 发散批：K 线解剖补全
    fm.add("marubozu_bull", "geometry", (dir_ > 0) & (np.nan_to_num(body_r, nan=0) >= 0.9))
    fm.add("marubozu_bear", "geometry", (dir_ < 0) & (np.nan_to_num(body_r, nan=0) >= 0.9))
    fm.add("tweezer_top", "geometry",
           (np.abs(h - _prev(h)) <= 0.1 * atr_abs) & ((c - l) / rng_pct < 0.5))
    fm.add("tweezer_bottom", "geometry",
           (np.abs(l - _prev(l)) <= 0.1 * atr_abs) & ((c - l) / rng_pct > 0.5))

    # ---- sequence ----
    st = _streak(dir_)
    fm.add("streak_signed", "sequence", (st * dir_).astype(np.int64))
    fm.add("streak_abs", "sequence", st)
    ret1 = _prev(c)
    ret1 = np.where(ret1 > 0, c / ret1 - 1.0, np.nan)
    abs_dc = np.abs(c - _prev(c))
    up_b = (dir_ > 0).astype(np.float64)
    dn_b = (dir_ < 0).astype(np.float64)
    for k in (2, 3, 5, 8, 13, 20, 21):
        ck = _prev_k(c, k)
        ret_k = np.where(ck > 0, c / ck - 1.0, np.nan)
        fm.add(f"ret_{k}", "sequence", ret_k)
        fm.add(f"absret_{k}", "sequence", np.abs(ret_k))
        fm.add(f"up_frac_{k}", "sequence", _roll_sum(up_b, k) / k)
        fm.add(f"down_frac_{k}", "sequence", _roll_sum(dn_b, k) / k)
        path_len = _roll_sum(abs_dc, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            eff = np.abs(ret_k * ck) / path_len
        fm.add(f"efficiency_{k}", "sequence", np.where(path_len > 0, eff, np.nan))
    for k in (3, 5, 10, 20):
        fm.add(f"hh_count_{k}", "sequence", _roll_sum((h > _prev(h)).astype(np.float64), k))
        fm.add(f"hl_count_{k}", "sequence", _roll_sum((l > _prev(l)).astype(np.float64), k))
        fm.add(f"lh_count_{k}", "sequence", _roll_sum((h < _prev(h)).astype(np.float64), k))
        fm.add(f"ll_count_{k}", "sequence", _roll_sum((l < _prev(l)).astype(np.float64), k))
    # 发散批：动量扩展
    fm.add("streak_accel", "sequence", (body_pct - _prev(body_pct)) / atr20)
    fm.add("momentum_z_8", "sequence", _zscore_of(ret1, 8) / np.sqrt(8.0))
    fm.add("pullback_pct_20", "sequence", _pullback_pct(h, l, c, 20))
    fm.add("ret_to_path_20", "sequence", _ret_to_path(c, 20))

    # ---- structure（prior = 严格前 w 根，不含当前）----
    hp, lp = _prev(h), _prev(l)
    for w in (5, 10, 20, 50, 100):
        ph = _roll_max(hp, w)
        pl = _roll_min(lp, w)
        fm.add(f"breakout_high_{w}", "structure", h > ph)
        fm.add(f"breakout_low_{w}", "structure", l < pl)
        fm.add(f"failed_breakout_high_{w}", "structure", (h > ph) & (c < ph))
        fm.add(f"failed_breakout_low_{w}", "structure", (l < pl) & (c > pl))
        fm.add(f"breakout_high_depth_atr_{w}", "structure", np.clip(h - ph, 0, None) / atr_abs)
        fm.add(f"breakout_low_depth_atr_{w}", "structure", np.clip(pl - l, 0, None) / atr_abs)
        fm.add(f"dist_prior_high_atr_{w}", "structure", (c - ph) / atr_abs)
        fm.add(f"dist_prior_low_atr_{w}", "structure", (c - pl) / atr_abs)
        fm.add(f"prior_range_atr_{w}", "structure", (ph - pl) / atr_abs)
        fm.add(f"range_pos_prior_{w}", "structure", (c - pl) / np.where(ph > pl, ph - pl, np.nan))
    # 发散批：结构补全（4h 位势破位 / 扫高低失败 / 24h 新高低）
    lb4 = max(2, 4 * 3_600_000 // bar_ms)
    ph4, pl4 = _roll_max(hp, lb4), _roll_min(lp, lb4)
    fm.add("break_4h_hi", "structure", h > ph4)
    fm.add("break_4h_lo", "structure", l < pl4)
    w16 = max(2, 4 * 3_600_000 // bar_ms)
    pm16, pmi16 = _roll_max(hp, w16), _roll_min(lp, w16)
    fm.add("sweep_hi_fail", "structure", (h > pm16) & ((c - l) / rng_pct <= 0.3))
    fm.add("sweep_lo_fail", "structure", (l < pmi16) & ((c - l) / rng_pct >= 0.7))
    n24 = max(2, 24 * 3_600_000 // bar_ms)
    fm.add("new_24h_hi", "structure", c > _roll_max(_prev(c), n24))
    fm.add("new_24h_lo", "structure", c < _roll_min(_prev(c), n24))
    # other 族：持续位与拒绝深度
    for w in (20, 50):
        ph = _roll_max(hp, w)
        pl = _roll_min(lp, w)
        above = c > ph
        below = c < pl
        fm.add(f"bars_above_ph{w}", "other", _roll_sum(np.concatenate([[0.0], above[:-1]]), w))
        fm.add(f"bars_below_pl{w}", "other", _roll_sum(np.concatenate([[0.0], below[:-1]]), w))
        fm.add(f"reject_up_depth_atr_{w}", "other", (_roll_max(hp, w) - c) / atr_abs)
        fm.add(f"reject_down_depth_atr_{w}", "other", (c - _roll_min(lp, w)) / atr_abs)

    # ---- volatility ----
    rv1 = ret1 * ret1
    for k in (5, 10, 20, 50):
        fm.add(f"rv_{k}", "volatility", np.sqrt(np.clip(_roll_nanmean(rv1, k), 0, None)))
    atr14 = _prev(_roll_nanmean(rng_pct, 14))
    atr50 = _prev(_roll_nanmean(rng_pct, 50))
    fm.add("atr_ratio_14_50", "volatility", atr14 / atr50)
    fm.add("atr_pct_100", "volatility", _roll_pctile(rng_pct, 100))
    fm.add("atr_pct_200", "volatility", _roll_pctile(rng_pct, 200))
    tr = np.maximum.reduce([rng_pct, np.abs(h - _prev(c)) / _prev(c), np.abs(l - _prev(c)) / _prev(c)])
    fm.add("tr_pct_100", "volatility", _roll_pctile(tr, 100))
    fm.add("range_pct_100", "volatility", rng_pct / _prev(_roll_nanmean(rng_pct, 100)))
    rv5, rv10, rv20, rv50 = (np.sqrt(np.clip(_roll_nanmean(rv1, k), 0, None)) for k in (5, 10, 20, 50))
    fm.add("volatility_ratio_5_20", "volatility", rv5 / rv20)
    fm.add("volatility_ratio_10_50", "volatility", rv10 / rv50)
    comp520 = _prev(_roll_nanmean(rng_pct, 5)) / _prev(_roll_nanmean(rng_pct, 20))
    comp1050 = _prev(_roll_nanmean(rng_pct, 10)) / _prev(_roll_nanmean(rng_pct, 50))
    fm.add("compression_5_20", "volatility", comp520)
    fm.add("compression_10_50", "volatility", comp1050)
    # 发散批：波动率状态
    w30d = max(200, 30 * 86_400_000 // bar_ms)
    fm.add("atr_pctile_4320", "volatility", _roll_pctile(rng_pct, w30d, chunk=128))
    sd20 = _roll_nanstd(c, 20)
    bbw = 4 * sd20 / np.where(c > 0, c, np.nan)
    fm.add("bb_width_pctile_100", "volatility", _roll_pctile(bbw, 100))
    fm.add("vol_expansion_turn", "volatility", (comp520 >= 1.0) & (_prev(comp520) < 1.0))

    # ---- momentum_trend / mean_reversion ----
    for k in (5, 10, 20, 50, 100):
        sma = _roll_sum(c, k) / k
        fm.add(f"sma_dist_atr_{k}", "momentum_trend", (c - sma) / atr_abs)
        sma_prev_k = _prev_k(sma, k)
        fm.add(f"sma_slope_atr_{k}", "momentum_trend", (sma - sma_prev_k) / atr_abs)
        sd = _roll_nanstd(c, k)
        fm.add(f"zscore_{k}", "mean_reversion", (c - sma) / np.where(sd > 0, sd, np.nan))
    for k in (7, 14, 28):
        dch = np.where(ret1 > 0, ret1, 0.0)
        dcl = np.where(ret1 < 0, -ret1, 0.0)
        ag, al = _roll_nanmean(dch, k), _roll_nanmean(dcl, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            rsi = 100.0 * ag / (ag + al)
        fm.add(f"rsi_{k}", "momentum_trend", np.where((ag + al) > 0, rsi, np.nan))
    c3, c8 = _prev_k(c, 3), _prev_k(c, 8)
    c5, c13 = _prev_k(c, 5), _prev_k(c, 13)
    with np.errstate(invalid="ignore", divide="ignore"):
        fm.add("momentum_accel_3_8", "momentum_trend",
               np.where((c3 > 0) & (c8 > 0), (c / c3 - 1) / 3 - (c / c8 - 1) / 8, np.nan))
        fm.add("momentum_accel_5_13", "momentum_trend",
               np.where((c5 > 0) & (c13 > 0), (c / c5 - 1) / 5 - (c / c13 - 1) / 13, np.nan))

    # ---- volume ----
    for k in (20, 50, 100):
        vmean = _prev(_roll_nanmean(v, k))
        vstd = _prev(_roll_nanstd(v, k))
        fm.add(f"rel_volume_{k}", "volume", v / vmean)
        with np.errstate(invalid="ignore", divide="ignore"):
            fm.add(f"volume_z_{k}", "volume", (v - vmean) / vstd)
    sv = np.sign(ret1) * v
    for k in (5, 20, 50):
        fm.add(f"signed_volume_imbalance_{k}", "volume",
               _roll_nanmean(sv, k) / np.where(_prev(_roll_nanmean(v, k)) > 0,
                                               _prev(_roll_nanmean(v, k)), np.nan))
    # 发散批：量能背离
    rv20 = fm.cols["rel_volume_20"]
    fm.add("shrink_breakout_high_20", "volume", fm.cols["breakout_high_20"] & (rv20 < 0.7))
    fm.add("shrink_breakout_low_20", "volume", fm.cols["breakout_low_20"] & (rv20 < 0.7))
    fm.add("vol_price_divergence_20", "volume", fm.cols["new_24h_hi"] & (rv20 < 0.7))
    fm.add("high_vol_stall", "volume", (rv20 >= 1.5) & (fm.cols["absret_3"] < 0.5 * atr20))

    # ---- time ----
    fm.add("hour_utc", "time", ((t // 3_600_000) % 24).astype(np.int16))
    fm.add("day_of_week", "time", ((t // 86_400_000 + 3) % 7).astype(np.int8))
    hours = (t // 3_600_000) % 24
    fm.add("session_asia", "time", hours <= 7)
    fm.add("session_europe", "time", (hours >= 8) & (hours <= 15))
    fm.add("session_us", "time", hours >= 16)
    # 发散批：时间扩展
    dt64 = t.astype("datetime64[ms]")
    dom = (dt64.astype("datetime64[D]") - dt64.astype("datetime64[M]")).astype(np.int64) + 1
    mon = (dt64.astype("datetime64[M]").astype(np.int64) % 12) + 1
    fm.add("day_of_month", "time", dom.astype(np.int16))
    fm.add("month_start", "time", dom <= 3)
    fm.add("month_end", "time", dom >= 28)
    yrday = (dt64.astype("datetime64[D]") - dt64.astype("datetime64[Y]")).astype(np.int64)
    qe = np.isin(mon, [3, 6, 9, 12]) & (dom >= 28)
    fm.add("quarter_end", "time", qe)
    fm.add("funding_slot", "time", np.isin(hours, [0, 8, 16]))

    # ---- regime（阈值用滚动分位/固定倍数，避免依赖切分）----
    ap30 = fm.cols["atr_pctile_4320"]
    fm.add("regime_vol_high", "regime", ap30 >= 0.75)
    fm.add("regime_vol_low", "regime", ap30 <= 0.25)
    n30 = max(100, 30 * 86_400_000 // bar_ms)
    c30 = _prev_k(c, n30)
    ret30 = np.where(c30 > 0, c / c30 - 1.0, np.nan)
    fm.add("regime_trend_up", "regime", ret30 > 0.03)
    fm.add("regime_trend_down", "regime", ret30 < -0.03)
    rng100 = _prev(_roll_nanmean(rng_pct, 100))
    c100 = _prev_k(c, 100)
    ret100 = np.where(c100 > 0, np.abs(c / c100 - 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        fm.add("regime_trending", "regime", ret100 >= 4 * rng100)
        fm.add("regime_ranging", "regime", ret100 <= rng100)
    fm.add("regime_compression", "regime", comp520 <= 0.6)
    fm.add("regime_expansion", "regime", comp520 >= 1.5)

    # ---- 发散批：极端事件后行为 ----
    sd100 = _prev(_roll_nanstd(ret1, 100))
    shock = np.abs(ret1) > 3 * sd100
    since = _bars_since(shock)
    fm.add("bars_since_shock", "extreme", np.where(np.isfinite(since), np.minimum(since, 999), 999))
    fm.add("in_shock_24", "extreme", since <= 24)
    last_shock_idx = np.where(np.isfinite(since), np.arange(N) - np.where(np.isfinite(since), since, 0), 0).astype(np.int64)
    fm.add("shock_dir", "extreme", np.sign(ret1[np.clip(last_shock_idx, 0, N - 1)]) * np.isfinite(since))

    # ---- 发散批：多周期共振（1h/4h 最后收盘方向与当前根同向）----
    for tf_ms, tag in ((3_600_000, "1h"), (14_400_000, "4h")):
        htf = aggregate_to(kl, tf_ms)
        if len(htf) > 1:
            hdir = np.sign(htf.c - htf.o)
            j = np.searchsorted(htf.t, t - tf_ms, side="right") - 1
            ok = j >= 0
            hd = np.where(ok, hdir[np.clip(j, 0, len(htf) - 1)], np.nan)
            fm.add(f"align_{tag}", "multiscale", np.where(ok, dir_ * hd > 0, False))
        else:
            fm.add(f"align_{tag}", "multiscale", np.zeros(N, dtype=bool))
    fm.add("slot_in_4h", "multiscale", ((t // bar_ms) % (14_400_000 // bar_ms)).astype(np.int16))
    fm.add("slot_in_1h", "multiscale", ((t // bar_ms) % (3_600_000 // bar_ms)).astype(np.int16))

    # ---- 发散批：周期路径（15m 周期内 3 根 5m 形态）----
    if k5 is not None and len(k5) >= 3:
        cp = _cycle_path(kl, k5, 900_000)
        fm.add("path3_uud", "path", cp["path_code"] == 1)
        fm.add("path3_duu", "path", cp["path_code"] == 4)
        fm.add("path3_all_up", "path", cp["path_code"] == 0)
        fm.add("path3_all_down", "path", cp["path_code"] == 7)
        fm.add("last5_dir_neg", "path", cp["last_dir"] < 0)
        fm.add("last5_dir_pos", "path", cp["last_dir"] > 0)
        fm.add("er_path", "path", cp["er_path"])
    return fm


def _prev_k(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def _zscore_of(x: np.ndarray, k: int) -> np.ndarray:
    """ret1 序列的滚动标准化累计（近似冲量）。"""
    mu = _roll_nanmean(x, k)
    sd = _prev(_roll_nanstd(x, k))
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = _roll_sum(np.where(np.isnan(x), 0.0, x), k)
    return np.where(sd > 0, cs / (sd * np.sqrt(k)), np.nan)


def _pullback_pct(h: np.ndarray, l: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """过去 k 根（不含当前）最高点回撤到当前收盘的位置 [0=最高点, 1=最低点]。"""
    ph, pl = _roll_max(_prev(h), k), _roll_min(_prev(l), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ph - c) / (ph - pl)
    return out


def _ret_to_path(c: np.ndarray, k: int) -> np.ndarray:
    """累计收益 / 路径总长（效率比，含当前根）。"""
    ck = _prev_k(c, k)
    net = np.where(ck > 0, np.abs(c / ck - 1.0), np.nan)
    path = _roll_sum(np.abs(c - _prev(c)), k)
    with np.errstate(invalid="ignore", divide="ignore"):
        cp = _prev_k(c, k)
        out = np.where((path > 0) & (cp > 0), np.abs(c - cp) / path, np.nan)
    return out
