#!/usr/bin/env python
"""shape_segment_knn_720d：BTC 5m/15m K线分段形状 kNN 因果检索离线研究。

口径纪律（ex-ante，冻结于 CFG，跑后改=数据窥探须全新切分重验）：
- 锚点=confirm_idx（ZigZag）/段末根（固定窗），决策时刻只含历史；
- 标签=绝对 UP/DOWN：label_up[i]=c[i+1]>o[i+1]（非相对延续），doji→invalid；
- 因果检索：参考段锚点 ra ≤ 查询段首根 qs - causal_lag(2)，消除特征-标签同源污染；
- 三基准（always-UP / 动量 / 多数类）同分母配对（eval_idx），弃权不计入分母；
- 0.6/0.2/0.2 三段时序切分 + BH-FDR(q=0.1) + 冻结 holdout 只触碰一次；
- 诚实声明：胜率差异是条件概率描述，按 0.50 双边口径计 EV（FEE=0.02+PREMIUM=0.01），
  未经滑点/报价可得性/执行延迟检验，不直接构成可交易信号；CONFIRMED 仅以冻结 holdout 为准。

用法：
    .venv\\Scripts\\python.exe -X utf8 scripts/shape_segment_knn_720d.py --tf all --stage all
    .venv\\Scripts\\python.exe -X utf8 scripts/shape_segment_knn_720d.py --tf 5m --stage segment,scan --limit-segments 500

阶段（分阶段可重入，崩溃不重跑）：
    segment  固定窗 + ZigZag 分段（npz 缓存）+ segments_{tf}.csv 普查
    scan     变体网格 walk-forward 因果检索 + kNN 投票 + 三基准 → _scan_{tf}.pkl
    patterns KMeans 形态簇（定性）→ patterns_{tf}.csv
    test     预注册三段统计裁决（BH-FDR + 安慰剂 + holdout 只触碰一次）→ variants_{tf}.csv
    report   run_config.json + report.md + knn_predictions_{tf}.csv
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
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from binance_predict.backtest.stats import (  # noqa: E402
    ev, exact_binomial_p, min_detectable_effect, multiple_testing_threshold,
    power_preflight, wilson,
)
from binance_predict.discovery import Klines, data_summary, load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402
from binance_predict.discovery.hypotheses import DEFAULTS  # noqa: E402
from binance_predict.discovery.l1_tester import bh_fdr  # noqa: E402
from binance_predict.discovery.oos_validator import breakeven_win_rate, run_block_ci  # noqa: E402
from binance_predict.discovery.targets import _fwd_ok, seg_bounds  # noqa: E402
from binance_predict.services.backtest import snapshot_token, wilson_lower_bound  # noqa: E402
from binance_predict.services.curve_features import cluster_windows, cosine_sim  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, "output")
BAR_MS = {"5m": 300_000, "15m": 900_000}
TFS = ("5m", "15m")
CSV_PATH = {"5m": os.path.join(OUT0, "klines_5m_720d.csv"),
            "15m": os.path.join(OUT0, "klines_15m_720d.csv")}
_ATR_WARMUP = 20        # ZigZag 热身伪影丢弃阈值（pivot_idx < 此值的段丢弃）
_PRED_CSV_CAP = 20000   # knn_predictions CSV 逐查询行确定性截断上限
_SEG_VERSION = 1        # 分段缓存版本

# ---- 冻结 CFG（30 项；cfg_sha256 入 run_config 与每个 pkl，双端校验）----
# 用户已确认 3 项对交接文档字面规格的收紧：causal_lag=2、功效只标注不终裁、knn_block=128。
CFG = {
    "seg_len": 12, "seg_step": 1, "query_stride": 12,
    "zigzag_atr": 1.5, "zigzag_min_bars": 4, "zigzag_max_bars": 60,
    "resample_m": 12, "norm_mode": "zscore", "sim": "cosine",
    "k_neighbors": 20, "consensus_margin": 0.5,
    "dtw_band": 2, "dtw_recall_n": 100, "dtw_query_stride": 7,
    "min_refs": 200, "causal_lag": 2, "knn_block": 128,
    "discovery_frac": 0.6, "validation_frac": 0.2, "holdout_frac": 0.2,
    "fdr_alpha": 0.1, "min_samples": 80, "min_sample_frac": 0.01,
    "min_lift_pp": 2.0, "min_validation_lift_pp": 0.5,
    "n_min_holdout": 60, "entry_price": 0.5,
    "n_clusters": 12, "seed": 11, "n_placebo": 20,
}

# 启动断言：9 项统计预算键逐项 == DEFAULTS，把房规漂移变硬失败
_BUDGET_KEYS = ("discovery_frac", "validation_frac", "holdout_frac", "fdr_alpha",
                "min_samples", "min_sample_frac", "min_lift_pp",
                "min_validation_lift_pp", "n_min_holdout")
for _k in _BUDGET_KEYS:
    assert CFG[_k] == DEFAULTS[_k], \
        f"房规漂移：CFG[{_k}]={CFG[_k]} != DEFAULTS[{_k}]={DEFAULTS[_k]}"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def iso_utc(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def _clean(v):
    """NaN/inf → None；np 标量 → py 标量（JSON/CSV 落盘防污染）。"""
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _pkl(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _unpkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def cfg_sha256() -> str:
    return hashlib.sha256(json.dumps(CFG, sort_keys=True).encode("utf-8")).hexdigest()


def data_csv_path(tf: str) -> str:
    return CSV_PATH[tf]


# ============================ 层1：分段（两函数返回同构 dict，下游零分支）============================

def segment_fixed(kl, L: int, step: int) -> dict:
    """固定长度滑窗分段。返回 {start,anchor,end,n_bars,valid,kind,query_idx,segmode,confirm_lag}。

    - anchors=arange(L-1,n,step)，追加 anchors<=n-2 守卫（标签需 anchor+1<=n-1）；
    - valid=valid_start[start]（按段首根对齐，非 anchor——最易写错一格处）；
    - query_idx=非重叠查询（每 query_stride 个段取 1）。
    """
    n = len(kl)
    c = kl.c
    cont = kl.cont
    empty = {
        "start": np.zeros(0, dtype=np.int64), "anchor": np.zeros(0, dtype=np.int64),
        "end": np.zeros(0, dtype=np.int64), "n_bars": np.zeros(0, dtype=np.int64),
        "valid": np.zeros(0, dtype=bool), "kind": np.zeros(0, dtype=np.int64),
        "query_idx": np.zeros(0, dtype=np.int64), "segmode": "fixed",
        "confirm_lag": np.zeros(0, dtype=np.int64),
    }
    if n < L or L < 2:
        return empty
    anchors = np.arange(L - 1, n, step, dtype=np.int64)
    anchors = anchors[anchors <= n - 2]
    if len(anchors) == 0:
        return empty
    start = anchors - L + 1
    end = anchors  # 固定窗：段末根 == 锚点
    # valid_start[j] = 段 [j, j+L-1] 内连续：cont[j+1..j+L-1] 全 True
    #   = sliding_window_view(cont[1:], L-1)[j]
    n_vs = n - L + 1
    if n_vs > 0 and (n - 1) >= (L - 1):
        valid_start = sliding_window_view(cont[1:], L - 1).all(axis=1)
    else:
        valid_start = np.zeros(max(0, n_vs), dtype=bool)
    valid = valid_start[start] if len(valid_start) else np.zeros(len(start), dtype=bool)
    kind = np.sign(c[anchors] - c[start]).astype(np.int64)
    n_bars = np.full(len(anchors), L, dtype=np.int64)
    qstride = max(1, int(CFG["query_stride"]))
    query_idx = np.arange(0, len(anchors), qstride, dtype=np.int64)
    return {
        "start": start.astype(np.int64), "anchor": anchors.astype(np.int64),
        "end": end.astype(np.int64), "n_bars": n_bars,
        "valid": np.asarray(valid, dtype=bool), "kind": kind,
        "query_idx": query_idx, "segmode": "fixed",
        "confirm_lag": np.zeros(len(anchors), dtype=np.int64),
    }


def zigzag_pivots(c, atr, theta, cont) -> list:
    """单遍 O(n) ATR 阈值 ZigZag（逐行迁自已手验 no-repaint 的 zigzag_dryrun.py，禁改逻辑）。

    返回 [(pivot_idx, pivot_px, kind, confirm_idx)]。kind=-1 峰 / +1 谷。
    锚点(决策时刻)=confirm_idx，严格晚于 pivot_idx；cont[i]==False 终止当前腿重启极值；
    ATR 非有限只推进极值不确认；last_kind 保证峰谷交替；末段未确认腿丢弃。
    """
    n = len(c)
    pivots: list = []
    if n < 2:
        return pivots
    hi_i = 0
    hi_px = float(c[0])
    lo_i = 0
    lo_px = float(c[0])
    last_kind = 0  # 0=尚无 / +1=上一个是谷(接下来找峰) / -1=上一个是峰(接下来找谷)
    for i in range(1, n):
        if not cont[i]:
            # 断点：当前腿作废，极值追踪从 i 重启（不 emit 未确认腿）
            hi_i = lo_i = i
            hi_px = lo_px = float(c[i])
            last_kind = 0
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            # ATR 未热身/异常：无法定阈，仅推进极值追踪，不做确认
            if c[i] > hi_px:
                hi_px = float(c[i]); hi_i = i
            if c[i] < lo_px:
                lo_px = float(c[i]); lo_i = i
            continue
        thr_i = theta * float(a)
        # 先用当前根推进运行极值
        if c[i] > hi_px:
            hi_px = float(c[i]); hi_i = i
        if c[i] < lo_px:
            lo_px = float(c[i]); lo_i = i
        # 峰确认：价格自运行高点回落 >= thr_i（且上一个不是峰，保证交替）
        if last_kind != -1 and (hi_px - c[i]) >= thr_i:
            pivots.append((hi_i, hi_px, -1, i))
            last_kind = -1
            lo_i = i; lo_px = float(c[i])
        elif last_kind != 1 and (c[i] - lo_px) >= thr_i:
            pivots.append((lo_i, lo_px, 1, i))
            last_kind = 1
            hi_i = i; hi_px = float(c[i])
    return pivots


def segment_zigzag(kl, atr, theta: float, min_bars: int, max_bars: int) -> dict:
    """ZigZag 自适应分段：段=相邻确认摆动点对。返回与 segment_fixed 同构 dict。

    - start=pivot_idx[k]、end=pivot_idx[k+1]、anchor=confirm_idx[k+1]（★非 pivot_idx）；
    - kind=sign[k+1]、confirm_lag=anchor-end；
    - 固化两条手验发现：① 丢弃 start<_ATR_WARMUP(20) 的热身伪影段；
      ② min_bars(4)<=n_bars<=max_bars(60)；追加 anchor<=n-2 守卫；
    - 查询集=全部段（ZigZag 天然非重叠）；valid=段内 cont 全 True。
    """
    c = kl.c
    cont = kl.cont
    n = len(c)
    empty = {
        "start": np.zeros(0, dtype=np.int64), "anchor": np.zeros(0, dtype=np.int64),
        "end": np.zeros(0, dtype=np.int64), "n_bars": np.zeros(0, dtype=np.int64),
        "valid": np.zeros(0, dtype=bool), "kind": np.zeros(0, dtype=np.int64),
        "query_idx": np.zeros(0, dtype=np.int64), "segmode": "zigzag",
        "confirm_lag": np.zeros(0, dtype=np.int64),
    }
    pivots = zigzag_pivots(c, atr, theta, cont)
    starts, anchors, ends, n_bars, kinds, clags, valid = [], [], [], [], [], [], []
    for k in range(len(pivots) - 1):
        p0 = pivots[k][0]
        p1, _, kind1, cf1 = pivots[k + 1]
        if p0 < _ATR_WARMUP:          # ① 丢弃热身伪影段
            continue
        nb = p1 - p0
        if not (min_bars <= nb <= max_bars):   # ② 腿长过滤
            continue
        if cf1 > n - 2:               # 标签守卫 anchor+1<=n-1
            continue
        seg_valid = bool(cont[p0 + 1: p1 + 1].all()) if p1 > p0 else True
        starts.append(p0); ends.append(p1); anchors.append(cf1)
        n_bars.append(nb); kinds.append(int(kind1)); clags.append(cf1 - p1)
        valid.append(seg_valid)
    if not starts:
        return empty
    start = np.asarray(starts, dtype=np.int64)
    end = np.asarray(ends, dtype=np.int64)
    anchor = np.asarray(anchors, dtype=np.int64)
    return {
        "start": start, "anchor": anchor, "end": end,
        "n_bars": np.asarray(n_bars, dtype=np.int64),
        "valid": np.asarray(valid, dtype=bool),
        "kind": np.asarray(kinds, dtype=np.int64),
        "query_idx": np.arange(0, len(start), dtype=np.int64),  # 全部段为查询
        "segmode": "zigzag",
        "confirm_lag": np.asarray(clags, dtype=np.int64),
    }


def _seg_fp(tf: str, segmode: str, p: dict) -> str:
    """分段缓存指纹（只含分段参数；改 norm/k/sim 不重跑分段）。"""
    key = (f"shapeknn-seg-v{_SEG_VERSION}|{tf}|{segmode}|"
           f"{p.get('seg_len')}|{p.get('seg_step')}|{p.get('query_stride')}|"
           f"{p.get('zigzag_atr')}|{p.get('zigzag_min_bars')}|{p.get('zigzag_max_bars')}")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _save_segs_npz(path: str, segs: dict, fp: str, meta: dict) -> None:
    payload = {k: segs[k] for k in ("start", "anchor", "end", "n_bars", "valid",
                                    "kind", "query_idx", "confirm_lag")}
    payload["_meta"] = np.array([json.dumps({"fp": fp, "segmode": segs["segmode"], **meta},
                                            ensure_ascii=False)], dtype=object)
    np.savez(path, **payload)


def _load_segs_npz(path: str, fp: str):
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["_meta"][0]))
        if meta.get("fp") != fp:
            return None
        segs = {k: z[k] for k in ("start", "anchor", "end", "n_bars", "valid",
                                  "kind", "query_idx", "confirm_lag")}
        segs["segmode"] = meta["segmode"]
        return segs
    except Exception as ex:  # noqa: BLE001
        log(f"  分段缓存失效（{ex}），重建")
        return None


def get_segments(tf: str, kl, atr, segmode: str, p: dict, cache_dir: str | None) -> dict:
    """按 segmode + 参数取分段（npz 磁盘缓存命中则复用，否则重算并落盘）。"""
    fp = _seg_fp(tf, segmode, p)
    if cache_dir:
        path = os.path.join(cache_dir, f"_segments_{tf}_{segmode}_{fp}.npz")
        cached = _load_segs_npz(path, fp)
        if cached is not None:
            return cached
    else:
        path = None
    if segmode == "fixed":
        segs = segment_fixed(kl, int(p["seg_len"]), int(p["seg_step"]))
    elif segmode == "zigzag":
        segs = segment_zigzag(kl, atr, float(p["zigzag_atr"]),
                              int(p["zigzag_min_bars"]), int(p["zigzag_max_bars"]))
    else:
        raise ValueError(f"未知 segmode: {segmode}")
    if path:
        try:
            _save_segs_npz(path, segs, fp, {"n": len(kl)})
        except Exception as ex:  # noqa: BLE001
            log(f"  分段缓存写入失败（{ex}），跳过")
    return segs


# ============================ 层2：归一化表征 + npz 缓存（必须去均值）============================

def resample_path(closes, m: int) -> np.ndarray:
    """参考实现：np.interp 把变长路径重采样到 m 点（供 _resample_matrix 断言对拍）。"""
    closes = np.asarray(closes, dtype=np.float64)
    L = len(closes)
    if L == 0:
        return np.zeros(m, dtype=np.float64)
    if L == 1:
        return np.full(m, float(closes[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, L)
    x_new = np.linspace(0.0, 1.0, m)
    return np.interp(x_new, x_old, closes)


def _resample_matrix(W, m: int) -> np.ndarray:
    """向量化等价重采样：W (n_seg, L) → (n_seg, m)。gather: pos=j*(L-1)/(m-1) 线性插值。"""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W[None, :]
    n_seg, L = W.shape
    if L == 0:
        return np.zeros((n_seg, m), dtype=np.float64)
    if L == 1:
        return np.repeat(W, m, axis=1)
    pos = np.linspace(0.0, L - 1, m)          # (m,) 旧坐标系位置
    idx0 = np.floor(pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, L - 1)
    frac = pos - idx0
    v0 = W[:, idx0]
    v1 = W[:, idx1]
    return v0 * (1.0 - frac)[None, :] + v1 * frac[None, :]


def normalize_path(path, atr0, mode: str):
    """去均值归一化 + 末行 L2 归一（零向量保持零 + 标 degenerate）。

    path: (m,) 或 (n_seg, m)；atr0: 标量或 (n_seg,)。返回 (out, degenerate)。
    - zscore: (p-mean)/std（std<1e-12→全0，degenerate）
    - atr:    (p-p[0])/atr0（atr0 非有限或<=0→全0，degenerate）
    - logret: cumsum(diff(log p)) 前补 0
    三模式一律再去均值 + L2 归一。
    """
    p = np.asarray(path, dtype=np.float64)
    single = (p.ndim == 1)
    if single:
        p = p[None, :]
    a0 = np.atleast_1d(np.asarray(atr0, dtype=np.float64)).astype(np.float64)
    if a0.size == 1 and p.shape[0] != 1:
        a0 = np.full(p.shape[0], float(a0[0]))
    degenerate = np.zeros(p.shape[0], dtype=bool)
    if mode == "zscore":
        mu = p.mean(axis=1, keepdims=True)
        sd = p.std(axis=1, keepdims=True)
        small = sd < 1e-12
        out = np.where(small, 0.0, (p - mu) / np.where(small, 1.0, sd))
        degenerate |= small.squeeze(1)
    elif mode == "atr":
        bad = ~np.isfinite(a0) | (a0 <= 0)
        a_safe = np.where(bad, 1.0, a0)
        out = (p - p[:, :1]) / a_safe[:, None]
        out = np.where(bad[:, None], 0.0, out)
        degenerate |= bad
    elif mode == "logret":
        with np.errstate(invalid="ignore", divide="ignore"):
            lr = np.diff(np.log(p), axis=1)
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
    else:
        raise ValueError(f"未知 norm_mode: {mode}")
    # 一律去均值（zscore 已去均值，幂等）+ L2 归一
    out = out - out.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    tiny = nrm < 1e-12
    out = np.where(tiny, 0.0, out / np.where(tiny, 1.0, nrm))
    degenerate |= tiny.squeeze(1)
    if single:
        return out[0], bool(degenerate[0])
    return out, degenerate


def _gather_zigzag_paths(c, start, end, m) -> np.ndarray:
    """ZigZag 变长段向量化 gather 插值到 m 点：idx0=start+floor, idx1=min(idx0+1,end)。"""
    n_seg = len(start)
    lengths = (end - start + 1).astype(np.float64)     # (n_seg,)
    pos = np.linspace(0.0, 1.0, m)                      # 归一位置 (m,)
    lm1 = np.maximum(lengths - 1.0, 0.0)                # (n_seg,)
    fpos = pos[None, :] * lm1[:, None]                  # (n_seg, m) 段内浮点偏移
    idx0 = np.floor(fpos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, np.maximum(lengths - 1, 0).astype(np.int64)[:, None])
    frac = fpos - idx0
    g0 = start[:, None] + idx0
    g1 = start[:, None] + idx1
    v0 = c[g0]
    v1 = c[g1]
    return v0 * (1.0 - frac) + v1 * frac


def build_shape_matrix(kl, segs, atr, m: int, mode: str):
    """构建 (X, repr_valid)。X: (n_seg, m) L2 归一 float64；repr_valid: (n_seg,) bool。

    - 固定窗：sliding_window_view(c,L) 零拷贝 → fancy-index → 广播归一 → L2（全向量化）；
    - ZigZag 变长：向量化 gather 插值；
    - atr0=atr[start]（ex-ante）；std<eps/atr0 非法 → 行置零 + repr_valid=False 剔出检索池。
    """
    c = kl.c
    start = segs["start"]
    end = segs["end"]
    n_seg = len(start)
    if n_seg == 0:
        return np.zeros((0, m), dtype=np.float64), np.zeros(0, dtype=bool)
    seg_valid = np.asarray(segs["valid"], dtype=bool)
    if segs["segmode"] == "fixed":
        L = int(segs["n_bars"][0])
        W = sliding_window_view(c, L)[start]        # (n_seg, L) 零拷贝 fancy-index
        R = _resample_matrix(W, m)
    else:
        R = _gather_zigzag_paths(c, start, end, m)
    atr0 = atr[start]
    X, degen = normalize_path(R, atr0, mode)
    repr_valid = seg_valid & (~degen)
    return np.asarray(X, dtype=np.float64), repr_valid


def build_desc10_matrix(kl, segs, atr, m: int, disc_mask):
    """repr_desc10 臂：惰性 import _series_features（私有 API），整臂 try/except。

    逐维 mean/std 只在 discovery 段拟合（防泄漏）。返回 (X, repr_valid) 或抛异常。
    """
    from binance_predict.services.curve_features import _series_features  # 惰性私有 API
    c = kl.c
    start = segs["start"]
    end = segs["end"]
    n_seg = len(start)
    if n_seg == 0:
        return np.zeros((0, 10)), np.zeros(0, dtype=bool)
    R = _gather_zigzag_paths(c, start, end, m) if segs["segmode"] == "zigzag" \
        else _resample_matrix(sliding_window_view(c, int(segs["n_bars"][0]))[start], m)
    feats = np.array([_series_features(R[i]) for i in range(n_seg)], dtype=np.float64)
    # 逐维 mean/std 只在 discovery 段拟合
    dfeats = feats[disc_mask]
    mu = dfeats.mean(axis=0) if len(dfeats) else feats.mean(axis=0)
    sd = dfeats.std(axis=0) if len(dfeats) else feats.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (feats - mu) / sd
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    tiny = nrm < 1e-12
    X = np.where(tiny, 0.0, Z / np.where(tiny, 1.0, nrm))
    repr_valid = np.asarray(segs["valid"], dtype=bool) & (~tiny.squeeze(1))
    return X, repr_valid


# ============================ 层3：因果 kNN 检索（最关键）============================

def cosine_topk(Xq, Xr, qa, qs, ra, k, min_refs, block):
    """因果 kNN 余弦检索（分块矩阵乘，绝不物化 N×N）。返回 (nn_idx, nn_sim, n_avail)。

    - 统一因果：ra <= qs - causal_lag(2)；ra 升序 → cuts=searchsorted 裁剪，块内只乘 Xr[:max_cut]；
    - L2 归一 ⇒ 点积=余弦（float32）；n_avail<min_refs 由下游弃权；
    - 每行 argpartition 取 topk → sort + stable argsort 破平局（确定性：sim 降序，平局 ref 索引升序）；
    - 出口硬断言：① ra<=qs-lag ② 无自匹配 ③ 标签根 ra+1<qs。
    nn_idx: (n_q,k) 索引到 Xr/ra（-1=空槽）；nn_sim: (n_q,k) 余弦（-inf=空槽）。
    """
    n_q = Xq.shape[0]
    n_r = Xr.shape[0]
    nn_idx = np.full((n_q, k), -1, dtype=np.int64)
    nn_sim = np.full((n_q, k), -np.inf, dtype=np.float64)
    n_avail = np.zeros(n_q, dtype=np.int64)
    if n_q == 0 or n_r == 0:
        return nn_idx, nn_sim, n_avail
    assert np.all(np.diff(ra) >= 0), "参考池 ra 必须升序（cuts 裁剪前提）"
    Xq32 = np.ascontiguousarray(Xq, dtype=np.float32)
    Xr32 = np.ascontiguousarray(Xr, dtype=np.float32)
    lim = (qs - CFG["causal_lag"]).astype(np.int64)
    cuts = np.searchsorted(ra, lim, side="right").astype(np.int64)  # #{ra <= lim}
    n_avail = cuts.copy()
    for b in range(0, n_q, block):
        e = min(b + block, n_q)
        blk = e - b
        max_cut = int(cuts[b:e].max())
        if max_cut <= 0:
            continue
        S = Xq32[b:e] @ Xr32[:max_cut].T  # (blk, max_cut) float32；ra[:cr] 即因果候选
        for r in range(blk):
            cr = int(cuts[b + r])
            if cr <= 0:
                continue
            Sr = S[r, :cr].astype(np.float64)
            kbr = min(k, cr)
            if kbr < cr:
                pr = np.argpartition(-Sr, kbr - 1)[:kbr]
            else:
                pr = np.arange(cr)
            pr = np.sort(pr)                        # 升序，stable argsort 破平局前提
            ps = Sr[pr]
            order = np.argsort(-ps, kind="stable")  # sim 降序，平局保持 pr 升序
            pr = pr[order]
            ps = ps[order]
            fin = np.isfinite(ps)
            pr = pr[fin][:k]
            ps = ps[fin][:k]
            nn_idx[b + r, :len(pr)] = pr
            nn_sim[b + r, :len(ps)] = ps
    # ---- 出口硬断言（致命 off-by-one 防线）----
    valid = nn_idx >= 0
    if valid.any():
        safe = np.where(valid, nn_idx, 0)
        ra_nn = ra[safe]
        lim_k = np.repeat(lim[:, None], k, axis=1)
        qa_k = np.repeat(qa[:, None], k, axis=1)
        qs_k = np.repeat(qs[:, None], k, axis=1)
        assert bool((ra_nn[valid] <= lim_k[valid]).all()), "因果违规：存在 ra > qs-causal_lag"
        assert bool((ra_nn[valid] != qa_k[valid]).all()), "自匹配违规：参考锚点==查询锚点"
        assert bool((ra_nn[valid] + 1 < qs_k[valid]).all()), "标签根未严格早于查询段首根"
    return nn_idx, nn_sim, n_avail


def dtw_band_cost(q, C, band) -> np.ndarray:
    """纯 numpy 带状 Sakoe-Chiba DTW，一对多跨候选向量化。q:(m,) C:(n_cand,m) → (n_cand,) 成本。"""
    q = np.asarray(q, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[None, :]
    n_c, m = C.shape
    D = np.full((n_c, m + 1, m + 1), np.inf)
    D[:, 0, 0] = 0.0
    for i in range(1, m + 1):
        jlo = max(1, i - band)
        jhi = min(m, i + band)
        for j in range(jlo, jhi + 1):
            cost = np.abs(C[:, j - 1] - q[i - 1])
            D[:, i, j] = cost + np.minimum(
                np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, m, m]


def dtw_rerank(Xq, Xr, qa, qs, ra, k, min_refs, recall_n, band, block):
    """先 cosine_topk(recall_n) 召回 → 带状 DTW 只重排取 topk（N² DTW 禁止）。

    返回与 cosine_topk 同构 (nn_idx, nn_sim, n_avail)；nn_sim=-DTW成本（降序=越相似）。
    查询子集 [::dtw_query_stride] 的抽取在调用方（臂级）完成，保证与余弦臂配对。
    """
    nn_idx, nn_sim, n_avail = cosine_topk(Xq, Xr, qa, qs, ra, recall_n, min_refs, block)
    n_q = Xq.shape[0]
    out_idx = np.full((n_q, k), -1, dtype=np.int64)
    out_sim = np.full((n_q, k), -np.inf, dtype=np.float64)
    for r in range(n_q):
        cand = nn_idx[r]
        cand = cand[cand >= 0]
        if len(cand) == 0:
            continue
        C = Xr[cand]
        cost = dtw_band_cost(Xq[r], C, band)
        order = np.lexsort((cand, cost))       # 成本升序，平局 cand 索引升序
        sel = order[:min(k, len(order))]
        out_idx[r, :len(sel)] = cand[sel]
        out_sim[r, :len(sel)] = -cost[sel]
    return out_idx, out_sim, n_avail


# ============================ 层4：标签 + kNN 投票 + 三基准 ============================

def build_labels(kl) -> dict:
    """绝对 UP/DOWN 标签（自建，不复用 build_targets 的相对 continuation_1）。

    label_up[i]=c[i+1]>o[i+1]；label_valid[i]=(i+1<n)&cont[i+1]&(c[i+1]!=o[i+1])；
    连续性口径 cont[i+1] 照抄 _fwd_ok(cont,1)；doji→invalid。
    """
    n = len(kl)
    o, c, cont = kl.o, kl.c, kl.cont
    label_up = np.zeros(n, dtype=bool)
    label_valid = np.zeros(n, dtype=bool)
    label_ret = np.full(n, np.nan)
    if n > 1:
        fwd = _fwd_ok(cont, 1)                 # fwd[i]=cont[i+1]（i<n-1）
        neq = c[1:] != o[1:]
        label_up[:n - 1] = c[1:] > o[1:]
        label_valid[:n - 1] = fwd[:n - 1] & neq
        with np.errstate(invalid="ignore", divide="ignore"):
            label_ret[:n - 1] = c[1:] / o[1:] - 1.0
    return {"label_up": label_up, "label_valid": label_valid, "label_ret": label_ret}


def knn_vote(ref_up, ref_valid, nn_idx, nn_sim, n_avail, min_refs, margin) -> dict:
    """kNN 共识投票（对齐 feature_bench 三条弃权，弃权不计入分母）。

    弃权：① n_avail<min_refs；② 邻居 label_valid=False(NOISE) 不计票→无有效票；
    ③ up/(up+dn) 未达 margin 或平票。返回 {predicted,abstain,n_up,n_dn,top1_sim,mean_sim_k,consensus}。
    """
    n_q, kk = nn_idx.shape
    valid_slot = (nn_idx >= 0) & np.isfinite(nn_sim)
    safe_idx = np.where(nn_idx >= 0, nn_idx, 0)
    rv = ref_valid[safe_idx] & valid_slot      # 计票资格
    ru = ref_up[safe_idx]
    n_up = (rv & ru).sum(axis=1).astype(np.int64)
    n_dn = (rv & (~ru)).sum(axis=1).astype(np.int64)
    tot = n_up + n_dn
    sim_pos = np.where(valid_slot, nn_sim, 0.0)
    sim_cnt = valid_slot.sum(axis=1)
    if kk > 0:
        top1_sim = np.where(sim_cnt > 0, sim_pos[:, 0], np.nan)
    else:
        top1_sim = np.full(n_q, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_sim_k = np.where(sim_cnt > 0,
                              sim_pos.sum(axis=1) / np.where(sim_cnt > 0, sim_cnt, 1), np.nan)
        consensus = np.where(tot > 0, n_up / np.where(tot > 0, tot, 1), np.nan)
    predicted = np.zeros(n_q, dtype=np.int8)
    abstain = np.ones(n_q, dtype=bool)
    enough = n_avail >= min_refs
    has_vote = tot > 0
    tie = (n_up == n_dn)
    up_ok = enough & has_vote & (~tie) & (consensus >= margin)
    dn_ok = enough & has_vote & (~tie) & ((1.0 - consensus) >= margin)
    predicted[up_ok] = 1
    abstain[up_ok] = False
    predicted[dn_ok] = -1
    abstain[dn_ok] = False
    return {"predicted": predicted, "abstain": abstain, "n_up": n_up, "n_dn": n_dn,
            "top1_sim": top1_sim, "mean_sim_k": mean_sim_k, "consensus": consensus}


def momentum_baseline(segs, c) -> np.ndarray:
    """动量基准：sign(c[anchor]-c[start])，==0 弃权（每段一个预测，1/-1/0）。"""
    start = segs["start"]
    anchor = segs["anchor"]
    if len(start) == 0:
        return np.zeros(0, dtype=np.int8)
    return np.sign(c[anchor] - c[start]).astype(np.int8)


# ============================ 层5：变体网格 + scan 编排 ============================

# OFAT 冻结网格（15 臂）：每臂 {"name","overrides"(CFG 浅覆盖),"segmode"}。
# 任何变体若被采纳须重新预注册并在全新切分重验，不得直接认领这里的数字。
VARIANTS = [
    {"name": "PRIMARY", "overrides": {}, "segmode": "fixed"},
    {"name": "seg_zigzag_1.5atr", "overrides": {}, "segmode": "zigzag"},
    {"name": "seg_zigzag_1.0atr", "overrides": {"zigzag_atr": 1.0}, "segmode": "zigzag"},
    {"name": "seg_zigzag_2.0atr", "overrides": {"zigzag_atr": 2.0}, "segmode": "zigzag"},
    {"name": "seg_fixed_L8", "overrides": {"seg_len": 8}, "segmode": "fixed"},
    {"name": "seg_fixed_L24", "overrides": {"seg_len": 24}, "segmode": "fixed"},
    {"name": "seg_fixed_overlap6", "overrides": {"seg_step": 2, "query_stride": 6}, "segmode": "fixed"},
    {"name": "norm_atr", "overrides": {"norm_mode": "atr"}, "segmode": "fixed"},
    {"name": "norm_logret", "overrides": {"norm_mode": "logret"}, "segmode": "fixed"},
    {"name": "repr_desc10", "overrides": {"repr": "desc10"}, "segmode": "fixed"},
    {"name": "sim_dtw", "overrides": {"sim": "dtw"}, "segmode": "fixed"},
    {"name": "k5", "overrides": {"k_neighbors": 5}, "segmode": "fixed"},
    {"name": "k50", "overrides": {"k_neighbors": 50}, "segmode": "fixed"},
    {"name": "margin_0.6", "overrides": {"consensus_margin": 0.6}, "segmode": "fixed"},
    {"name": "margin_0.7", "overrides": {"consensus_margin": 0.7}, "segmode": "fixed"},
]


def _truncate_segs(segs, limit) -> dict:
    """确定性截断到前 limit 个段（探索性冒烟用；query_idx 同步过滤）。"""
    n = len(segs["start"])
    if limit is None or n <= limit:
        return segs
    idx = np.arange(0, limit, dtype=np.int64)
    out = {}
    for key, val in segs.items():
        out[key] = val[idx] if isinstance(val, np.ndarray) else val
    out["query_idx"] = segs["query_idx"][segs["query_idx"] < limit]
    return out


def _retrieve_arm(segs, X, repr_valid, labels, mom_all, cfg_arm) -> dict:
    """walk-forward 因果检索 + 投票（discovery/validation refs=锚点<i1；holdout refs=锚点<i2）。

    返回每查询数组（全 split 拼接）+ discovery 安慰剂所需 (nn_idx/n_avail/ref 标签)。
    """
    n_seg = len(segs["start"])
    anchor = segs["anchor"]
    start = segs["start"]
    qidx = segs["query_idx"]
    if cfg_arm.get("sim") == "dtw":
        qidx = qidx[::max(1, int(cfg_arm["dtw_query_stride"]))]  # 臂级确定性子集
    n_q = len(qidx)
    i1, i2 = seg_bounds(n_seg, cfg_arm["discovery_frac"], cfg_arm["validation_frac"])
    qpos = qidx
    q_split = np.where(qpos < i1, 0, np.where(qpos < i2, 1, 2)).astype(np.int8)
    qa = anchor[qidx]
    qs = start[qidx]
    label_up = labels["label_up"]
    label_valid = labels["label_valid"]
    predicted = np.zeros(n_q, dtype=np.int8)
    abstain = np.ones(n_q, dtype=bool)
    n_avail = np.zeros(n_q, dtype=np.int64)
    consensus = np.full(n_q, np.nan)
    top1_sim = np.full(n_q, np.nan)
    mom_q = mom_all[qidx]
    disc_placebo = None
    ref_caps = {0: i1, 1: i1, 2: i2}
    use_dtw = cfg_arm.get("sim") == "dtw"
    for sp in (0, 1, 2):
        q_sel = np.flatnonzero(q_split == sp)
        if len(q_sel) == 0:
            continue
        cap = ref_caps[sp]
        ref_seg = np.arange(0, min(cap, n_seg), dtype=np.int64)
        ref_ok = repr_valid[ref_seg]
        ref_seg_ok = ref_seg[ref_ok]
        ra = anchor[ref_seg_ok]
        Xr = X[ref_seg_ok]
        ref_up = label_up[ra]
        ref_valid = label_valid[ra]
        Xq = X[qidx[q_sel]]
        qa_s = qa[q_sel]
        qs_s = qs[q_sel]
        if use_dtw:
            nn_idx, nn_sim, nav = dtw_rerank(
                Xq, Xr, qa_s, qs_s, ra, int(cfg_arm["k_neighbors"]), int(cfg_arm["min_refs"]),
                int(cfg_arm["dtw_recall_n"]), int(cfg_arm["dtw_band"]), int(cfg_arm["knn_block"]))
        else:
            nn_idx, nn_sim, nav = cosine_topk(
                Xq, Xr, qa_s, qs_s, ra, int(cfg_arm["k_neighbors"]), int(cfg_arm["min_refs"]),
                int(cfg_arm["knn_block"]))
        vote = knn_vote(ref_up, ref_valid, nn_idx, nn_sim, nav,
                        int(cfg_arm["min_refs"]), float(cfg_arm["consensus_margin"]))
        predicted[q_sel] = vote["predicted"]
        abstain[q_sel] = vote["abstain"]
        n_avail[q_sel] = nav
        consensus[q_sel] = vote["consensus"]
        top1_sim[q_sel] = vote["top1_sim"]
        if sp == 0:
            disc_placebo = {
                "nn_idx": nn_idx, "n_avail": nav, "ref_up": ref_up,
                "ref_valid": ref_valid, "q_sel": q_sel,
                "label_up_q": label_up[qa_s], "label_valid_q": label_valid[qa_s],
                "mom_q": mom_q[q_sel], "qa_s": qa_s,
            }
    return {
        "qa": qa, "qs": qs, "q_split": q_split, "q_segpos": qpos,
        "predicted": predicted, "abstain": abstain, "mom_pred": mom_q,
        "label_up_q": label_up[qa], "label_valid_q": label_valid[qa],
        "n_avail": n_avail, "consensus": consensus, "top1_sim": top1_sim,
        "i1": int(i1), "i2": int(i2), "n_seg": int(n_seg),
        "disc_placebo": disc_placebo,
    }


def scan_tf(tf: str, kl, atr, labels, out: str, limit_segments=None) -> list:
    """对全部 VARIANTS 臂做 walk-forward 因果检索 + 投票，返回臂级结果列表。"""
    c = kl.c
    arms = []
    for var in VARIANTS:
        ov = var["overrides"]
        segmode = var["segmode"]
        cfg_arm = {**CFG, **ov}
        t0 = time.time()
        segp = {"seg_len": cfg_arm["seg_len"], "seg_step": cfg_arm["seg_step"],
                "query_stride": cfg_arm["query_stride"], "zigzag_atr": cfg_arm["zigzag_atr"],
                "zigzag_min_bars": cfg_arm["zigzag_min_bars"],
                "zigzag_max_bars": cfg_arm["zigzag_max_bars"]}
        segs = get_segments(tf, kl, atr, segmode, segp, out)
        segs = _truncate_segs(segs, limit_segments)
        n_seg = len(segs["start"])
        if n_seg == 0:
            arms.append({"name": var["name"], "segmode": segmode, "overrides": ov,
                         "skipped": "NO_SEGMENTS", "q": None, "cfg_sha256": cfg_sha256()})
            continue
        mom_all = momentum_baseline(segs, c)
        i1, _ = seg_bounds(n_seg, CFG["discovery_frac"], CFG["validation_frac"])
        skipped = None
        X = repr_valid = None
        if ov.get("repr") == "desc10":
            try:
                disc_mask = np.zeros(n_seg, dtype=bool)
                disc_mask[:i1] = True
                X, repr_valid = build_desc10_matrix(kl, segs, atr, cfg_arm["resample_m"], disc_mask)
            except Exception as ex:  # noqa: BLE001
                skipped = "SKIPPED_PRIVATE_API"
                log(f"  [{tf}] {var['name']}: 私有 API 失败（{ex}）→ {skipped}")
        else:
            X, repr_valid = build_shape_matrix(kl, segs, atr, cfg_arm["resample_m"],
                                               cfg_arm["norm_mode"])
        if skipped is not None:
            arms.append({"name": var["name"], "segmode": segmode, "overrides": ov,
                         "skipped": skipped, "q": None, "cfg_sha256": cfg_sha256()})
            continue
        q = _retrieve_arm(segs, X, repr_valid, labels, mom_all, cfg_arm)
        arms.append({
            "name": var["name"], "segmode": segmode, "overrides": ov, "skipped": None,
            "q": q, "cfg_sha256": cfg_sha256(), "n_seg": n_seg,
            "n_repr_valid": int(repr_valid.sum()), "elapsed_s": round(time.time() - t0, 2),
        })
        log(f"  [{tf}] {var['name']}: n_seg={n_seg} queries={len(q['qa'])} "
            f"abstain={float(q['abstain'].mean()):.3f} 耗时 {time.time()-t0:.1f}s")
    return arms


# ============================ 层6：统计裁决（逐行仿写 streak_census:601-755，顺序不可换）============================

def _write_csv(path: str, header: list, rows: list) -> None:
    """确定性 CSV：newline='\n'，固定表头，行由调用方按 anchor 升序排好。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(v, spec=".6g") -> str:
    """固定浮点格式（字节复现）；None/非有限 → 空串。"""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return format(float(v), spec)
    return str(v)


def eval_arm(pred, abstain, label_up_q, label_valid_q, mom_pred, qa) -> dict:
    """在给定查询子集上算 kNN 臂统计 + 三基准同分母配对（陷阱 C 防线）。

    eval_idx=(~abstain)&label_valid；base_up/base_mom/base_majority 均在同一 eval_idx 上算；
    另存 abstain_rate + base_up_allqueries 对照。power_note 只标注不裁决（冻结决策2）。
    """
    n_q = int(len(pred))
    abstain_rate = float(abstain.mean()) if n_q else np.nan
    lv_all = label_valid_q
    base_up_allqueries = float(label_up_q[lv_all].mean()) if lv_all.any() else np.nan
    eval_idx = (~abstain) & label_valid_q
    n = int(eval_idx.sum())
    out = {"n": n, "n_queries": n_q, "abstain_rate": abstain_rate,
           "base_up_allqueries": base_up_allqueries}
    if n == 0:
        out.update({"k": 0, "win_rate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "base_up": np.nan, "base_mom": np.nan, "base_majority": np.nan,
                    "lift_pp": np.nan, "lift_vs_mom_pp": np.nan, "p_up": np.nan,
                    "p_mom": np.nan, "p_method": None, "ev_at": np.nan, "ev_ci_low": np.nan,
                    "above_breakeven": False, "block_ci": [np.nan, np.nan],
                    "power_note": None})
        return out
    pred_e = pred[eval_idx]
    lab_e = label_up_q[eval_idx]
    correct = ((pred_e == 1) & lab_e) | ((pred_e == -1) & (~lab_e))
    k = int(correct.sum())
    win_rate = k / n
    lo, hi = wilson(win_rate, n)
    ci_low = wilson_lower_bound(k, n)
    base_up = float(lab_e.mean())                 # always-UP（同分母）
    base_majority = max(base_up, 1.0 - base_up)   # 多数类
    mom_e = mom_pred[eval_idx]
    mom_correct = ((mom_e == 1) & lab_e) | ((mom_e == -1) & (~lab_e))
    base_mom = float(mom_correct.mean())          # 动量（同分母；mom==0 计为未命中）
    p_up = exact_binomial_p(k, n, base_up) if 0 < base_up < 1 else np.nan
    p_mom = exact_binomial_p(k, n, base_mom) if 0 < base_mom < 1 else np.nan
    lift_pp = (win_rate - base_up) * 100.0
    lift_vs_mom_pp = (win_rate - base_mom) * 100.0
    ev_at = ev(win_rate, CFG["entry_price"])
    ev_ci_low = ev(ci_low, CFG["entry_price"])
    above_breakeven = bool(ci_low > breakeven_win_rate(CFG["entry_price"]))
    qa_e = qa[eval_idx]
    order = np.argsort(qa_e, kind="stable")
    hit_pos = qa_e[order]
    wins = correct[order].astype(float)
    block_ci = run_block_ci(hit_pos, wins, b=3000, seed=CFG["seed"])
    power_note = power_preflight(n, abs(lift_pp), base_up if np.isfinite(base_up) else 0.5)
    # exact_binomial_p 内部在 n>2000 自动退化为正态近似（含连续性校正）——显式落字段供 report 注明
    p_method = "exact" if n <= 2000 else "normal_approx"
    out.update({"k": k, "win_rate": win_rate, "ci_low": ci_low, "ci_high": hi,
                "base_up": base_up, "base_mom": base_mom, "base_majority": base_majority,
                "lift_pp": lift_pp, "lift_vs_mom_pp": lift_vs_mom_pp, "p_up": p_up,
                "p_mom": p_mom, "p_method": p_method, "ev_at": ev_at, "ev_ci_low": ev_ci_low,
                "above_breakeven": above_breakeven,
                "block_ci": [float(block_ci[0]), float(block_ci[1])],
                "power_note": power_note})
    return out


def _placebo_lifts(dp, cfg_arm, rng, n_placebo) -> list:
    """安慰剂（仅 discovery）：邻居标签置换 n_placebo 次 → lift 零分布。"""
    if dp is None:
        return []
    ref_up0 = dp["ref_up"]
    ref_valid0 = dp["ref_valid"]
    nn_idx = dp["nn_idx"]
    nav = dp["n_avail"]
    lab_up_q = dp["label_up_q"]
    lab_valid_q = dp["label_valid_q"]
    mom_q = dp["mom_q"]
    min_refs = int(cfg_arm["min_refs"])
    margin = float(cfg_arm["consensus_margin"])
    nn_sim_dummy = np.zeros_like(nn_idx, dtype=np.float64)
    lifts = []
    for _ in range(n_placebo):
        perm = rng.permutation(ref_up0)
        vote = knn_vote(perm, ref_valid0, nn_idx, nn_sim_dummy, nav, min_refs, margin)
        ev_idx = (~vote["abstain"]) & lab_valid_q
        nn_ = int(ev_idx.sum())
        if nn_ == 0:
            continue
        pe = vote["predicted"][ev_idx]
        le = lab_up_q[ev_idx]
        corr = ((pe == 1) & le) | ((pe == -1) & (~le))
        lifts.append((float(corr.mean()) - float(le.mean())) * 100.0)
    return lifts


def _verdict_ladder(row, h, lift_d) -> str:
    """裁决阶梯（冻结 holdout 六条硬闸全与才 CONFIRMED；power_note 不进阶梯）。"""
    n_h = h["n"]
    lift_h = h["lift_pp"]
    p_h = h["p_up"]
    wr_h = h["win_rate"]
    ev_ci_low_h = h["ev_ci_low"]
    lift_mom_h = h["lift_vs_mom_pp"]
    block_lo = h["block_ci"][0]
    base_up_h = h["base_up"]
    be = breakeven_win_rate(CFG["entry_price"])
    ci_excl = bool(np.isfinite(lift_h) and lift_h > 0 and np.isfinite(block_lo)
                   and np.isfinite(base_up_h) and block_lo > base_up_h)
    row["ci_excl"] = ci_excl
    if n_h < CFG["n_min_holdout"]:
        return "INSUFFICIENT_SAMPLES"
    if np.isfinite(lift_h) and np.isfinite(lift_d) and np.sign(lift_h) != np.sign(lift_d):
        return "REJECT"  # 方向翻转
    confirmed = (np.isfinite(lift_h) and lift_h >= CFG["min_lift_pp"]
                 and np.isfinite(p_h) and p_h < 0.05 and ci_excl
                 and np.isfinite(ev_ci_low_h) and ev_ci_low_h > 0
                 and np.isfinite(wr_h) and wr_h > be
                 and np.isfinite(lift_mom_h) and lift_mom_h > 0)
    if confirmed:
        return "CONFIRMED"
    if (np.isfinite(lift_h) and lift_h > 0
            and ((np.isfinite(p_h) and p_h < 0.10)
                 or (np.isfinite(ev_ci_low_h) and ev_ci_low_h > 0))):
        return "PROMISING"
    if np.isfinite(lift_h) and lift_h > 0:
        return "WEAK"
    return "REJECT"


def stage_test(out: str, tfs: tuple, exploratory: bool = False) -> dict:
    """预注册三段统计裁决：哈希双端校验→准入→val 方向→BH-FDR→安慰剂→holdout 只触碰一次。"""
    all_results: dict = {}
    for tf in tfs:
        p = os.path.join(out, f"_scan_{tf}.pkl")
        if not os.path.exists(p):
            log(f"test：{tf} 缺少 scan 产物，跳过")
            continue
        scan = _unpkl(p)
        if scan.get("cfg_sha256") != cfg_sha256():
            raise RuntimeError(f"{tf}: CFG 在 scan 后变更（哈希不符）——违反冻结纪律")
        arms = scan["arms"]
        rows: list = []
        for arm in arms:
            base = {"name": arm["name"], "segmode": arm["segmode"],
                    "overrides": json.dumps(arm["overrides"], ensure_ascii=False)}
            if arm.get("skipped") or arm.get("q") is None:
                base.update({"skipped": True, "gated": False, "fdr_pass": False,
                             "val_pass": False, "verdict": arm.get("skipped", "SKIPPED")})
                rows.append(base)
                continue
            q = arm["q"]
            sp = q["q_split"]
            d_sel, v_sel = sp == 0, sp == 1
            d = eval_arm(q["predicted"][d_sel], q["abstain"][d_sel], q["label_up_q"][d_sel],
                         q["label_valid_q"][d_sel], q["mom_pred"][d_sel], q["qa"][d_sel])
            v = eval_arm(q["predicted"][v_sel], q["abstain"][v_sel], q["label_up_q"][v_sel],
                         q["label_valid_q"][v_sel], q["mom_pred"][v_sel], q["qa"][v_sel])
            n_disc_total = int(d_sel.sum())
            n_floor = max(CFG["min_samples"], int(CFG["min_sample_frac"] * n_disc_total))
            gated = bool(d["n"] >= n_floor)
            base.update({"skipped": False, "gated": gated, "n_floor": n_floor,
                         "discovery": {k: _clean(x) for k, x in d.items()},
                         "validation": {k: _clean(x) for k, x in v.items()}})
            if not gated:
                base.update({"val_pass": False, "fdr_pass": False,
                             "verdict": "INSUFFICIENT_SAMPLES"})
                rows.append(base)
                continue
            ld, lv = d["lift_pp"], v["lift_pp"]
            if CFG["validation_frac"] <= 0:
                # --split 70/30 对照旗标：validation 段为空，方向一致性无从校验 → 退化为仅准入闸门
                val_pass = bool(gated)
            else:
                val_pass = bool(np.isfinite(ld) and np.isfinite(lv)
                                and np.sign(ld) == np.sign(lv)
                                and abs(lv) >= CFG["min_validation_lift_pp"])
            base["val_pass"] = val_pass
            rows.append(base)
        # ---- BH-FDR（discovery p_up 按 tf 一族）----
        idx = [i for i, r in enumerate(rows)
               if r.get("gated") and r.get("discovery", {}).get("p_up") is not None]
        pv = [rows[i]["discovery"]["p_up"] for i in idx]
        ok = bh_fdr(pv, q=CFG["fdr_alpha"]) if pv else []
        for i, flag in zip(idx, ok):
            rows[i]["fdr_pass"] = bool(flag)
        for r in rows:
            r.setdefault("fdr_pass", False)
        # ---- 安慰剂（仅 discovery；holdout 不触碰）----
        rng = np.random.default_rng(CFG["seed"] + 1)
        for arm, row in zip(arms, rows):
            if row.get("skipped") or not row.get("gated"):
                continue
            cfg_arm = {**CFG, **arm["overrides"]}
            lifts = _placebo_lifts(arm["q"].get("disc_placebo"), cfg_arm, rng, CFG["n_placebo"])
            row["placebo_lift_ci"] = ([float(x) for x in np.percentile(lifts, [2.5, 97.5])]
                                      if lifts else None)
            row["placebo_lift_mean"] = _clean(float(np.mean(lifts))) if lifts else None
        # ---- 冻结 holdout 终验（只触碰一次：仅 gated∧fdr_pass∧val_pass）----
        for arm, row in zip(arms, rows):
            if row.get("skipped"):
                continue
            if exploratory:
                # --limit-segments / --k / --sim / --split 70/30：未预注册，硬跳 holdout
                row["verdict"] = "EXPLORATORY_NO_HOLDOUT"
                continue
            if not (row.get("gated") and row.get("fdr_pass") and row.get("val_pass")):
                row["verdict"] = row.get("verdict", "REJECT_PREHOLDOUT")
                continue
            q = arm["q"]
            h_sel = q["q_split"] == 2
            h = eval_arm(q["predicted"][h_sel], q["abstain"][h_sel], q["label_up_q"][h_sel],
                         q["label_valid_q"][h_sel], q["mom_pred"][h_sel], q["qa"][h_sel])
            row["holdout"] = {k: _clean(x) for k, x in h.items()}
            row["verdict"] = _verdict_ladder(row, h, row["discovery"]["lift_pp"])
        all_results[tf] = rows
        n_conf = sum(1 for r in rows if r.get("verdict") == "CONFIRMED")
        log(f"test：{tf} {len(rows)} 臂 → CONFIRMED {n_conf}")
        _write_variants_csv(out, tf, rows)
    with open(os.path.join(out, "hypothesis_tests.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=1, default=_clean)
    return all_results


_VAR_COLS = ("n", "win_rate", "base_up", "base_mom", "base_majority", "lift_pp",
             "lift_vs_mom_pp", "p_up", "ci_low", "ci_high", "ev_at", "ev_ci_low",
             "above_breakeven", "abstain_rate", "base_up_allqueries")


def _write_variants_csv(out: str, tf: str, rows: list) -> None:
    header = (["tf", "variant", "segmode", "overrides", "gated", "fdr_pass", "val_pass",
               "placebo_lift_ci", "ci_excl"]
              + [f"discovery_{c}" for c in _VAR_COLS]
              + [f"validation_{c}" for c in _VAR_COLS]
              + [f"holdout_{c}" for c in _VAR_COLS]
              + ["holdout_block_ci", "holdout_power_note", "verdict"])
    lines = []
    for r in rows:
        row = [tf, r["name"], r["segmode"], r.get("overrides", ""),
               bool(r.get("gated")), bool(r.get("fdr_pass")), bool(r.get("val_pass")),
               json.dumps(r.get("placebo_lift_ci")) if r.get("placebo_lift_ci") else "",
               _clean(r.get("ci_excl"))]
        for grp in ("discovery", "validation", "holdout"):
            g = r.get(grp) or {}
            for c in _VAR_COLS:
                row.append(_fmt(g.get(c)))
        h = r.get("holdout") or {}
        row.append(json.dumps(h.get("block_ci")) if h.get("block_ci") else "")
        pn = h.get("power_note")
        row.append(pn.get("verdict", "") if isinstance(pn, dict) else "")
        row.append(r.get("verdict", ""))
        lines.append(row)
    _write_csv(os.path.join(out, f"variants_{tf}.csv"), header, lines)
    log(f"test：{tf} variants -> {os.path.relpath(os.path.join(out, f'variants_{tf}.csv'), ROOT)}")


# ============================ 层7：KMeans 形态簇（定性交付）============================

def stage_patterns(out: str, tfs: tuple, limit_segments=None) -> None:
    """PRIMARY 固定窗 discovery 段形状 KMeans → 簇普查 + OOS 余弦归属。"""
    for tf in tfs:
        kl = load_klines_csv(data_csv_path(tf), BAR_MS[tf])
        atr = atr_series(kl)
        labels = build_labels(kl)
        segp = {"seg_len": CFG["seg_len"], "seg_step": CFG["seg_step"],
                "query_stride": CFG["query_stride"], "zigzag_atr": CFG["zigzag_atr"],
                "zigzag_min_bars": CFG["zigzag_min_bars"], "zigzag_max_bars": CFG["zigzag_max_bars"]}
        segs = get_segments(tf, kl, atr, "fixed", segp, out)
        segs = _truncate_segs(segs, limit_segments)
        n_seg = len(segs["start"])
        if n_seg < 2:
            log(f"patterns：{tf} 段数不足，跳过")
            continue
        X, repr_valid = build_shape_matrix(kl, segs, atr, CFG["resample_m"], CFG["norm_mode"])
        i1, i2 = seg_bounds(n_seg, CFG["discovery_frac"], CFG["validation_frac"])
        start, end, anchor = segs["start"], segs["end"], segs["anchor"]
        n_bars = segs["n_bars"]
        c = kl.c
        net_ret = (c[end] / c[start] - 1.0) * 100.0
        lab_up = labels["label_up"][anchor]
        lab_valid = labels["label_valid"][anchor]
        disc_pos = np.arange(n_seg) < i1
        fit_mask = repr_valid & disc_pos
        X_fit = X[fit_mask]
        if len(X_fit) < 2:
            log(f"patterns：{tf} discovery 有效形状不足，跳过")
            continue
        n_cl = max(1, min(CFG["n_clusters"], len(X_fit)))
        cl = cluster_windows(X_fit, n_cl, random_state=42)
        fit_idx = np.flatnonzero(fit_mask)
        # 自写质心（原始形状空间，再 L2 归一）
        centroids = np.zeros((n_cl, X.shape[1]))
        for j in range(n_cl):
            mem = X_fit[cl == j]
            cen = mem.mean(axis=0) if len(mem) else np.zeros(X.shape[1])
            nrm = np.linalg.norm(cen)
            centroids[j] = cen / nrm if nrm > 1e-12 else cen
        rows = []
        for j in range(n_cl):
            mem_pos = fit_idx[cl == j]
            nn = len(mem_pos)
            if nn == 0:
                continue
            up_m = lab_valid[mem_pos]
            up_rate = float(lab_up[mem_pos][up_m].mean()) if up_m.any() else np.nan
            lo, hi = wilson(up_rate, int(up_m.sum())) if up_m.any() else (np.nan, np.nan)
            # OOS 归属：validation/holdout 段 cosine 原始空间 argmax
            oos_pos = np.flatnonzero(repr_valid & (np.arange(n_seg) >= i1))
            oos_sel = []
            if len(oos_pos):
                sims = X[oos_pos] @ centroids.T
                am = sims.argmax(axis=1)
                oos_sel = oos_pos[am == j]
            oos_valid = lab_valid[oos_sel]
            oos_up = float(lab_up[oos_sel][oos_valid].mean()) if len(oos_sel) and oos_valid.any() else np.nan
            olo, ohi = wilson(oos_up, int(oos_valid.sum())) if len(oos_sel) and oos_valid.any() else (np.nan, np.nan)
            rows.append([
                tf, j, nn, _fmt(float(net_ret[mem_pos].mean())),
                _fmt(float(n_bars[mem_pos].mean())),
                _fmt(up_rate), _fmt(lo), _fmt(hi), int(up_m.sum()),
                len(oos_sel), _fmt(oos_up), _fmt(olo), _fmt(ohi),
                "cosine_raw",
            ])
        header = ["tf", "cluster", "n_disc", "mean_net_ret_pct", "mean_leg_bars",
                  "insample_up_rate", "insample_ci_low", "insample_ci_high", "insample_n_valid",
                  "n_oos", "oos_up_rate", "oos_ci_low", "oos_ci_high", "assign_metric"]
        _write_csv(os.path.join(out, f"patterns_{tf}.csv"), header, rows)
        log(f"patterns：{tf} {n_cl} 簇 -> {os.path.relpath(os.path.join(out, f'patterns_{tf}.csv'), ROOT)}")


# ============================ 层8：落盘 + report ============================

_SPLIT_NAMES = ("discovery", "validation", "holdout")
# knn_predictions 逐查询落盘的臂（头条余弦臂 + 配对 DTW 臂；其余臂在 _scan pkl 内全量保留）
_PRED_ARMS = ("PRIMARY", "sim_dtw")


def _load_tf(tf: str):
    """载入 K线 + ATR（前置，不含当前根）+ 绝对 UP/DOWN 标签。"""
    kl = load_klines_csv(data_csv_path(tf), BAR_MS[tf])
    atr = atr_series(kl)
    labels = build_labels(kl)
    return kl, atr, labels


def _seg_params(cfg=None) -> dict:
    cfg = CFG if cfg is None else cfg
    return {"seg_len": cfg["seg_len"], "seg_step": cfg["seg_step"],
            "query_stride": cfg["query_stride"], "zigzag_atr": cfg["zigzag_atr"],
            "zigzag_min_bars": cfg["zigzag_min_bars"],
            "zigzag_max_bars": cfg["zigzag_max_bars"]}


def _segment_range_atr(kl, start, end, atr0) -> np.ndarray:
    """段内 (max high - min low)/atr[start]（atr0 锚定段首根，ex-ante）。

    定长段走 sliding_window_view 零拷贝向量化；变长段（ZigZag）逐段循环。
    """
    n_seg = len(start)
    out = np.full(n_seg, np.nan)
    if n_seg == 0:
        return out
    lens = end - start + 1
    uniq = np.unique(lens)
    if len(uniq) == 1 and len(kl) >= int(uniq[0]):
        L = int(uniq[0])
        hi = sliding_window_view(kl.h, L)[start].max(axis=1)
        lo = sliding_window_view(kl.l, L)[start].min(axis=1)
        safe = np.where(np.isfinite(atr0) & (atr0 > 0), atr0, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (hi - lo) / safe
    for i in range(n_seg):
        s, e = int(start[i]), int(end[i])
        a = float(atr0[i])
        if e < s or not np.isfinite(a) or a <= 0:
            continue
        out[i] = (float(kl.h[s:e + 1].max()) - float(kl.l[s:e + 1].min())) / a
    return out


def stage_segment(out: str, tfs: tuple, limit_segments=None) -> None:
    """两种 segmode 段普查 → segments_{tf}.csv（全局按 anchor 升序；label_* 仅记录用）。"""
    header = ["tf", "seg_id", "segmode", "start_ts", "end_ts", "anchor_ts", "n_bars",
              "kind", "confirm_lag", "net_ret_pct", "range_atr", "label_up", "label_valid",
              "label_ret", "split"]
    for tf in tfs:
        kl, atr, labels = _load_tf(tf)
        t, c = kl.t, kl.c
        lu, lv, lr = labels["label_up"], labels["label_valid"], labels["label_ret"]
        segp = _seg_params()
        modes, acc = [], {}
        for segmode in ("fixed", "zigzag"):
            segs = get_segments(tf, kl, atr, segmode, segp, out)
            segs = _truncate_segs(segs, limit_segments)
            n_seg = len(segs["start"])
            log(f"segment：{tf}/{segmode} {n_seg} 段（valid={int(segs['valid'].sum())}）")
            if n_seg == 0:
                continue
            i1, i2 = seg_bounds(n_seg, CFG["discovery_frac"], CFG["validation_frac"])
            pos = np.arange(n_seg)
            split = np.where(pos < i1, 0, np.where(pos < i2, 1, 2)).astype(np.int8)
            with np.errstate(invalid="ignore", divide="ignore"):
                net_ret = (c[segs["end"]] / c[segs["start"]] - 1.0) * 100.0
            acc[segmode] = {
                "start": segs["start"], "end": segs["end"], "anchor": segs["anchor"],
                "n_bars": segs["n_bars"], "kind": segs["kind"],
                "confirm_lag": segs["confirm_lag"], "net_ret": net_ret,
                "range_atr": _segment_range_atr(kl, segs["start"], segs["end"],
                                                atr[segs["start"]]),
                "split": split,
            }
            modes.append(segmode)
        if not modes:
            log(f"segment：{tf} 无段，跳过")
            continue
        # 全局 anchor 升序（并列时按 segmode 序号再按 seg_id，确定性）
        a_all = np.concatenate([acc[m]["anchor"] for m in modes])
        m_all = np.concatenate([np.full(len(acc[m]["anchor"]), mi, dtype=np.int64)
                                for mi, m in enumerate(modes)])
        i_all = np.concatenate([np.arange(len(acc[m]["anchor"]), dtype=np.int64)
                                for m in modes])
        order = np.lexsort((i_all, m_all, a_all))
        path = os.path.join(out, f"segments_{tf}.csv")
        n_cap = 0
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            w = csv.writer(f)
            w.writerow(header)
            for oi in order:
                mi = int(m_all[oi])
                j = int(i_all[oi])
                d = acc[modes[mi]]
                a = int(d["anchor"][j])
                w.writerow([
                    tf, j, modes[mi], iso_utc(t[int(d["start"][j])]),
                    iso_utc(t[int(d["end"][j])]), iso_utc(t[a]),
                    int(d["n_bars"][j]), int(d["kind"][j]), int(d["confirm_lag"][j]),
                    _fmt(float(d["net_ret"][j])), _fmt(float(d["range_atr"][j])),
                    bool(lu[a]), bool(lv[a]), _fmt(float(lr[a])),
                    _SPLIT_NAMES[int(d["split"][j])],
                ])
                n_cap += 1
        log(f"segment：{tf} {n_cap} 行 -> {os.path.relpath(path, ROOT)}")


def stage_scan(out: str, tfs: tuple, limit_segments=None) -> None:
    """变体网格 walk-forward 因果检索 → _scan_{tf}.pkl（内必存 cfg_sha256，双端校验）。"""
    for tf in tfs:
        kl, atr, labels = _load_tf(tf)
        t0 = time.time()
        arms = scan_tf(tf, kl, atr, labels, out, limit_segments)
        _pkl(os.path.join(out, f"_scan_{tf}.pkl"), {
            "cfg_sha256": cfg_sha256(), "tf": tf, "arms": arms,
            "data_summary": data_summary(kl, BAR_MS[tf]),
            "limit_segments": limit_segments,
            "exploratory": bool(limit_segments is not None),
            "variants": [v["name"] for v in VARIANTS],
        })
        log(f"scan：{tf} {len(arms)} 臂 -> _scan_{tf}.pkl 耗时 {time.time() - t0:.1f}s")


def _write_predictions_csv(out: str, tf: str, scan: dict, kl, labels) -> int:
    """knn_predictions_{tf}.csv：逐查询，按 anchor 升序 + 确定性截断 cap 防爆。

    浮点固定 .6g；label_* / hit 为未来信息，仅记录用（不进任何决策）。
    """
    header = ["tf", "variant", "seg_pos", "split", "start_ts", "anchor_ts", "predicted",
              "abstain", "consensus", "top1_sim", "n_avail", "mom_pred", "label_up",
              "label_valid", "hit"]
    t = kl.t
    lu, lv = labels["label_up"], labels["label_valid"]
    path = os.path.join(out, f"knn_predictions_{tf}.csv")
    n_rows, capped = 0, False
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(header)
        for arm in scan["arms"]:
            if arm["name"] not in _PRED_ARMS or arm.get("q") is None:
                continue
            q = arm["q"]
            order = np.argsort(q["qa"], kind="stable")
            for oi in order:
                if n_rows >= _PRED_CSV_CAP:
                    capped = True
                    break
                oi = int(oi)
                a = int(q["qa"][oi])
                good = bool(lv[a])
                hit = ""
                if good:
                    p_, l_ = int(q["predicted"][oi]), bool(lu[a])
                    hit = bool(((p_ == 1) and l_) or ((p_ == -1) and not l_))
                w.writerow([
                    tf, arm["name"], int(q["q_segpos"][oi]),
                    _SPLIT_NAMES[int(q["q_split"][oi])],
                    iso_utc(t[int(q["qs"][oi])]), iso_utc(t[a]),
                    int(q["predicted"][oi]), bool(q["abstain"][oi]),
                    _fmt(float(q["consensus"][oi])), _fmt(float(q["top1_sim"][oi])),
                    int(q["n_avail"][oi]), int(q["mom_pred"][oi]),
                    bool(lu[a]), good, hit,
                ])
                n_rows += 1
            if capped:
                break
    if capped:
        log(f"report：{tf} knn_predictions 触达 cap {_PRED_CSV_CAP}，按 anchor 升序确定性截断")
    log(f"report：{tf} knn_predictions {n_rows} 行 -> {os.path.relpath(path, ROOT)}")
    return n_rows


def _plot_shapes(out: str, tf: str, kl, atr, labels, limit_segments=None) -> None:
    """仅 --plot：discovery 段 PRIMARY 形状 KMeans 质心图（matplotlib 缺失则静默跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as ex:
        log(f"report：{tf} matplotlib 不可用（{ex}），跳过绘图")
        return
    segs = _truncate_segs(get_segments(tf, kl, atr, "fixed", _seg_params(), out), limit_segments)
    n_seg = len(segs["start"])
    if n_seg < 2:
        return
    X, repr_valid = build_shape_matrix(kl, segs, atr, CFG["resample_m"], CFG["norm_mode"])
    i1, _ = seg_bounds(n_seg, CFG["discovery_frac"], CFG["validation_frac"])
    fit = repr_valid & (np.arange(n_seg) < i1)
    X_fit = X[fit]
    if len(X_fit) < 2:
        return
    n_cl = max(1, min(CFG["n_clusters"], len(X_fit)))
    cl = cluster_windows(X_fit, n_cl, random_state=42)
    m = X.shape[1]
    xs = np.arange(m)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for j in range(n_cl):
        mem = X_fit[cl == j]
        if len(mem) == 0:
            continue
        cen = mem.mean(axis=0)
        nrm = np.linalg.norm(cen)
        if nrm > 1e-12:
            cen = cen / nrm
        ax.plot(xs, cen, marker="o", ms=3, lw=1.2, label=f"c{j} (n={len(mem)})")
    ax.axhline(0.0, color="0.7", lw=0.8)
    # 标题用 ASCII：DejaVu Sans 无 CJK 字形，中文会刷一屏 Glyph missing 警告
    ax.set_title(f"shape_segment_knn_720d - {tf} discovery shape cluster centroids "
                 f"(L2-normalized, m={m})")
    ax.set_xlabel("resampled position")
    ax.set_ylabel("normalized shape")
    ax.legend(fontsize=6, ncol=3, loc="best")
    fig.tight_layout()
    path = os.path.join(out, f"shapes_{tf}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log(f"report：{tf} 形态图 -> {os.path.relpath(path, ROOT)}")


def _segment_census(out: str, tf: str, kl, atr, limit_segments=None) -> dict:
    """分段普查统计（供 report；缓存命中不重算）。"""
    res = {}
    for segmode in ("fixed", "zigzag"):
        segs = _truncate_segs(get_segments(tf, kl, atr, segmode, _seg_params(), out),
                              limit_segments)
        n = len(segs["start"])
        if n == 0:
            res[segmode] = None
            continue
        nb = segs["n_bars"]
        cl = segs["confirm_lag"]
        res[segmode] = {
            "n_seg": int(n), "valid_rate": float(segs["valid"].mean()),
            "n_bars_median": float(np.median(nb)), "n_bars_mean": float(nb.mean()),
            "n_bars_p10": float(np.percentile(nb, 10)),
            "n_bars_p90": float(np.percentile(nb, 90)),
            "confirm_lag_median": float(np.median(cl)), "confirm_lag_max": int(cl.max()),
            "up_kind_rate": float((segs["kind"] > 0).mean()),
            "n_queries": int(len(segs["query_idx"])),
        }
    return res


def _paired_cosine_dtw(scan: dict) -> dict | None:
    """余弦 vs DTW 在完全相同查询子集上的配对对比（DTW 臂跑在 qa[::dtw_query_stride]）。"""
    arms = {a["name"]: a for a in scan["arms"]}
    ca, da = arms.get("PRIMARY"), arms.get("sim_dtw")
    if not ca or not da or ca.get("q") is None or da.get("q") is None:
        return None
    qc, qd = ca["q"], da["q"]
    common = np.isin(qc["q_segpos"], qd["q_segpos"])
    if not common.any():
        return None
    one = np.ones(len(qd["qa"]), dtype=bool)
    cs = eval_arm(qc["predicted"][common], qc["abstain"][common], qc["label_up_q"][common],
                  qc["label_valid_q"][common], qc["mom_pred"][common], qc["qa"][common])
    ds = eval_arm(qd["predicted"][one], qd["abstain"][one], qd["label_up_q"][one],
                  qd["label_valid_q"][one], qd["mom_pred"][one], qd["qa"][one])
    both = (~qc["abstain"][common]) & (~qd["abstain"][one])
    agree = (float((qc["predicted"][common][both] == qd["predicted"][one][both]).mean())
             if both.any() else np.nan)
    return {"n_paired": int(common.sum()), "cosine": cs, "dtw": ds,
            "n_both_voted": int(both.sum()), "agree_rate": agree}


def _power_rows(tfs: tuple, per_tf: dict, hyp: dict) -> list:
    """每臂功效明细（只标注、不裁决）：holdout 查询数与 holdout 有效评估数各自的 MDE。

    MDE 口径 = min_detectable_effect(n, p0=0.5, 单侧 α=0.05, 功效 0.8) 正态近似；
    两个分母分开报：n_holdout_queries（全部 holdout 查询，含弃权）是设计上限，
    n_holdout_eval（弃权与 label_valid 过滤后的实际分母）才是裁决真正吃到的功效。
    """
    rows: list = []
    for tf in tfs:
        d = per_tf.get(tf)
        if d is None:
            continue
        hrows = {r["name"]: r for r in (hyp.get(tf) or [])}
        for a in d["scan"]["arms"]:
            q = a.get("q")
            n_hq = int((q["q_split"] == 2).sum()) if q is not None else 0
            r = hrows.get(a["name"], {})
            n_he = (r.get("holdout") or {}).get("n")
            rows.append({
                "tf": tf, "variant": a["name"], "segmode": a["segmode"],
                "skipped": a.get("skipped"),
                "n_holdout_queries": n_hq,
                "mde_holdout_queries_pp": (round(min_detectable_effect(n_hq) * 100.0, 2)
                                           if n_hq else None),
                "n_holdout_eval": n_he,
                "mde_holdout_eval_pp": (round(min_detectable_effect(int(n_he)) * 100.0, 2)
                                        if n_he else None),
                "verdict": r.get("verdict"),
            })
    return rows


def _write_run_config(out: str, tfs: tuple, per_tf: dict, hyp: dict, split_mode: str,
                      exploratory: bool) -> None:
    n_tf = max(1, len(per_tf))
    total_tests = len(VARIANTS) * n_tf
    tokens = {}
    for tf in tfs:
        d = per_tf.get(tf)
        if d is None:
            continue
        pr = next((a for a in d["scan"]["arms"]
                   if a["name"] == "PRIMARY" and a.get("q") is not None), None)
        if pr is not None:
            tokens[tf] = snapshot_token(pr["q"]["qa"].tolist())
    power_rows = _power_rows(tfs, per_tf, hyp)
    cfgj = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/shape_segment_knn_720d.py",
        "cfg_sha256": cfg_sha256(),
        "split_mode": split_mode,
        "exploratory": bool(exploratory),
        "note": ("探索性：未预注册，不触碰 holdout，不得直接认领" if exploratory else None),
        "cfg": CFG,
        "data_summary": {tf: per_tf[tf]["scan"]["data_summary"] for tf in tfs if tf in per_tf},
        "total_tests": total_tests,
        "snapshot_token": tokens,
        "power": power_rows,
        "power_statement": (
            "本设计在 holdout 上对 2pp 效应功效不足：各臂 holdout 查询数的 MDE 普遍高于 "
            "min_lift_pp=2.0pp，因此 null 结果不具信息量（不能据此断言无效应）。"
            "power_note 只标注、不进入裁决阶梯；终裁硬闸 = n_h >= n_min_holdout(60)。"
        ),
        "multiple_testing": multiple_testing_threshold(CFG["min_lift_pp"], total_tests),
        "multiple_testing_note": "仅供参考；正式控假走 bh_fdr(q=0.1) 按 tf 分族（见 variants_*.csv fdr_pass）。",
        "discipline": [
            "胜率差异是条件概率描述，不直接构成可交易信号。",
            "EV 按 0.50 双边口径计（FEE=0.02 + PREMIUM=0.01），打平胜率 ≈52.0408%。",
            "未经滑点 / 报价可得性 / 执行延迟检验。",
            "CONFIRMED 仅以冻结 holdout 为准，holdout 只触碰一次（仅 gated∧fdr_pass∧val_pass）。",
            "三基准（always-UP / 动量 / 多数类）同分母配对于 eval_idx；主 lift 用配对口径。",
            "锚点=confirm_idx / 段末根（ex-ante）；因果规则 ra ≤ qs - causal_lag(2)。",
            "CFG 冻结于 cfg_sha256；跑后改参 = 数据窥探，须全新切分重验。",
        ],
    }
    path = os.path.join(out, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfgj, f, ensure_ascii=False, indent=1, default=_clean)
    log(f"report：run_config -> {os.path.relpath(path, ROOT)}（cfg_sha256={cfg_sha256()[:16]}）")


def _mdn(v, nd=4) -> str:
    """markdown 数值格式化（非有限/None → 破折号）。"""
    if v is None or isinstance(v, bool):
        return ("—" if v is None else ("是" if v else "否"))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:.{nd}f}"


def _verdict_counts(hyp: dict, tfs: tuple) -> dict:
    cnt: dict = {}
    for tf in tfs:
        for r in (hyp.get(tf) or []):
            v = r.get("verdict", "MISSING")
            cnt[v] = cnt.get(v, 0) + 1
    return cnt


def _write_report_md(out: str, tfs: tuple, per_tf: dict, hyp: dict, census: dict,
                     paired: dict, split_mode: str, exploratory: bool) -> None:
    be = breakeven_win_rate(CFG["entry_price"])
    L: list = []
    L.append("# shape_segment_knn_720d 离线研究报告")
    L.append("")
    L.append(f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}")
    L.append(f"- cfg_sha256：`{cfg_sha256()}`")
    L.append(f"- 切分：{split_mode}（discovery={CFG['discovery_frac']} / "
             f"validation={CFG['validation_frac']} / holdout={CFG['holdout_frac']}）")
    L.append(f"- 模式：{'**探索性**（未预注册，holdout 未触碰，不得直接认领）' if exploratory else '预注册'}")
    L.append(f"- 变体臂数：{len(VARIANTS)}；周期：{', '.join(tfs)}")
    L.append("")
    L.append("## 1. 口径与方法")
    L.append("")
    L.append("1. **分段**：固定长度滑窗（L=seg_len、step=seg_step）与 ATR 阈值 ZigZag 摆动段双轨；"
             "ZigZag 锚点 = **confirm_idx**（非 pivot_idx），已手验 no-repaint（θ=1.0/1.5/2.0 违规 0、lag≥1）。")
    L.append("2. **表征**：段内收盘路径重采样到 m=resample_m 点 → 去均值归一（zscore/atr/logret）"
             "→ 末行 L2 归一；平坦段（std<1e-12）置零并标 repr_valid=False 剔出检索池。")
    L.append("3. **因果检索**：分块矩阵乘余弦（L2 归一⇒点积=余弦），硬约束 `ra ≤ qs - causal_lag(2)`，"
             "等价于参考段标签根 `ra+1` 严格早于查询段首根，消除特征-标签同源污染；"
             "绝不物化 N×N，cuts=searchsorted 裁剪，block=knn_block(128) 冻结（保字节复现）。")
    L.append("4. **标签**：绝对 UP/DOWN，`label_up[i]=c[i+1]>o[i+1]`；doji 与断点→invalid（不计入分母）。")
    L.append("5. **投票**：k 近邻中 label_valid 的邻居计票，up/(up+dn) 未达 consensus_margin 或平票则弃权；"
             "弃权不计入分母。")
    L.append("6. **裁决**：三段时序切分 → 准入闸门（n_disc ≥ max(min_samples, min_sample_frac·n_disc_total)）"
             "→ validation 方向一致 → BH-FDR(q=fdr_alpha) 按 tf 分族 → 安慰剂（仅 discovery）"
             "→ **冻结 holdout 只触碰一次**（仅 gated∧fdr_pass∧val_pass）。")
    L.append(f"7. **CONFIRMED 六条硬闸（全与）**：lift_h ≥ {CFG['min_lift_pp']}pp ∧ p_h < 0.05 ∧ "
             f"block_ci_low > base_up ∧ ev_ci_low > 0 ∧ win_rate > {be:.6f}（打平胜率）∧ "
             "lift_vs_mom_pp > 0。power_note 只标注、不进阶梯。")
    L.append("")
    L.append("## 2. 数据体检")
    L.append("")
    L.append("| tf | rows | start | end | median_bar_s | gaps>1.5×median |")
    L.append("|---|---|---|---|---|---|")
    for tf in tfs:
        ds = per_tf.get(tf, {}).get("scan", {}).get("data_summary")
        if not ds:
            continue
        L.append(f"| {tf} | {ds['rows']} | {ds['start']} | {ds['end']} | "
                 f"{_mdn(ds['median_bar_seconds'], 1)} | {ds['gap_count_gt_1_5x_median']} |")
    L.append("")
    L.append("## 3. 分段对照（固定窗 vs ZigZag）")
    L.append("")
    L.append("| tf | segmode | n_seg | valid率 | 腿长中位 | 腿长均值 | 腿长p10 | 腿长p90 | "
             "confirm_lag中位 | confirm_lag最大 | UP腿占比 | 查询数 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tf in tfs:
        for segmode in ("fixed", "zigzag"):
            s = (census.get(tf) or {}).get(segmode)
            if not s:
                continue
            L.append(f"| {tf} | {segmode} | {s['n_seg']} | {_mdn(s['valid_rate'])} | "
                     f"{_mdn(s['n_bars_median'], 1)} | {_mdn(s['n_bars_mean'], 2)} | "
                     f"{_mdn(s['n_bars_p10'], 1)} | {_mdn(s['n_bars_p90'], 1)} | "
                     f"{_mdn(s['confirm_lag_median'], 1)} | {s['confirm_lag_max']} | "
                     f"{_mdn(s['up_kind_rate'])} | {s['n_queries']} |")
    L.append("")
    L.append("固定窗 confirm_lag 恒为 0（段末根即锚点）；ZigZag 的 confirm_lag 中位数即「摆动点确认延迟」成本，"
             "也是它不重绘的代价。")
    L.append("")
    L.append("## 4. 变体账本（三段 + 裁决）")
    L.append("")
    L.append("主 lift 均用**同分母配对口径**（base_up 与 kNN 在同一 eval_idx 上算）；"
             "`base_up_allqueries` 仅作对照，不得用于 lift。")
    n_approx = n_exact = 0
    for tf in tfs:
        for r in (hyp.get(tf) or []):
            for s in ("discovery", "validation", "holdout"):
                pm = (r.get(s) or {}).get("p_method")
                if pm == "normal_approx":
                    n_approx += 1
                elif pm == "exact":
                    n_exact += 1
    L.append("")
    L.append(f"**p 值方法注明**：`p_up` / `p_mom` 由 `exact_binomial_p(k, n, base)` 计算——"
             f"n ≤ 2000 走**精确二项双侧**，n > 2000 **自动退化为正态近似（含连续性校正）**。"
             f"本案共 {n_exact} 个段统计走精确法、{n_approx} 个走正态近似"
             f"（逐臂实际方法见 `hypothesis_tests.json` 的 `p_method` 字段）。正态近似在 n 大时保守性"
             f"略差，因此裁决不只看 p 值：头条 CI 一律以 `run_block_ci` 的 block_ci_low > base_up 为准。")
    for tf in tfs:
        rows = hyp.get(tf) or []
        if not rows:
            L.append("")
            L.append(f"### {tf}：无 test 阶段产物（未跑 --stage test）")
            continue
        L.append("")
        L.append(f"### {tf}（{len(rows)} 臂）")
        L.append("")
        L.append("| 臂 | segmode | gated | fdr | val | n_d | lift_d(pp) | n_v | lift_v(pp) | "
                 "n_h | lift_h(pp) | p_h | block_ci_low | base_up_h | win_rate_h | "
                 "ev_ci_low | vs动量(pp) | 安慰剂CI | 裁决 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            d, v, h = r.get("discovery") or {}, r.get("validation") or {}, r.get("holdout") or {}
            pl = r.get("placebo_lift_ci")
            pl_s = (f"[{pl[0]:.2f}, {pl[1]:.2f}]" if isinstance(pl, list) and len(pl) == 2
                    else "—")
            bc = h.get("block_ci") or [None, None]
            L.append(
                f"| {r['name']} | {r['segmode']} | {'✓' if r.get('gated') else '✗'} | "
                f"{'✓' if r.get('fdr_pass') else '✗'} | {'✓' if r.get('val_pass') else '✗'} | "
                f"{_mdn(d.get('n'))} | {_mdn(d.get('lift_pp'), 2)} | "
                f"{_mdn(v.get('n'))} | {_mdn(v.get('lift_pp'), 2)} | "
                f"{_mdn(h.get('n'))} | {_mdn(h.get('lift_pp'), 2)} | {_mdn(h.get('p_up'))} | "
                f"{_mdn(bc[0])} | {_mdn(h.get('base_up'))} | {_mdn(h.get('win_rate'))} | "
                f"{_mdn(h.get('ev_ci_low'))} | {_mdn(h.get('lift_vs_mom_pp'), 2)} | "
                f"{pl_s} | **{r.get('verdict', '—')}** |")
    L.append("")
    cnt = _verdict_counts(hyp, tfs)
    L.append(f"裁决分布：{json.dumps(cnt, ensure_ascii=False) if cnt else '（无）'}")
    L.append("")
    L.append("## 5. 余弦 vs DTW 配对对比（同一查询子集）")
    L.append("")
    L.append("DTW 只在余弦召回 recall_n 后重排（band=dtw_band），且只跑在 `qa[::dtw_query_stride]` "
             "确定性子集；下表余弦臂已**限定到完全相同的子集**，两者同分母可比。")
    L.append("")
    L.append("| tf | n_paired | 臂 | n_eval | win_rate | base_up | lift(pp) | vs动量(pp) | 弃权率 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tf in tfs:
        pd = paired.get(tf)
        if not pd:
            continue
        for tag in ("cosine", "dtw"):
            s = pd[tag]
            L.append(f"| {tf} | {pd['n_paired']} | {tag} | {_mdn(s['n'])} | "
                     f"{_mdn(s['win_rate'])} | {_mdn(s['base_up'])} | "
                     f"{_mdn(s['lift_pp'], 2)} | {_mdn(s['lift_vs_mom_pp'], 2)} | "
                     f"{_mdn(s['abstain_rate'])} |")
        L.append(f"| {tf} | — | 两臂同时投票数 {pd['n_both_voted']}，方向一致率 "
                 f"{_mdn(pd['agree_rate'])} | | | | | | |")
    if not any(paired.values()):
        L.append("| — | — | 无配对结果（sim_dtw 臂未跑或被跳过） | | | | | | |")
    L.append("")
    L.append("注：本对比覆盖**全量查询（跨三段）**，属描述性；DTW 是否优于余弦**仅以第 4 节 holdout 裁决为准**。")
    L.append("")
    L.append("## 6. 形态簇（定性交付）")
    L.append("")
    for tf in tfs:
        ppath = os.path.join(out, f"patterns_{tf}.csv")
        L.append(f"### {tf}")
        L.append("")
        if not os.path.exists(ppath):
            L.append("无 patterns 产物（未跑 --stage patterns）。")
            L.append("")
            continue
        with open(ppath, encoding="utf-8") as f:
            pr = list(csv.reader(f))
        if len(pr) < 2:
            L.append("簇为空。")
            L.append("")
            continue
        L.append("| " + " | ".join(pr[0]) + " |")
        L.append("|" + "---|" * len(pr[0]))
        for row in pr[1:]:
            L.append("| " + " | ".join(x if x else "—" for x in row) + " |")
        L.append("")
        L.append("`insample_*` 为 **discovery 段内描述性**（不可外推）；`oos_*` 为 validation+holdout 段"
                 "按质心余弦归属（`assign_metric=cosine_raw`）的**可外推**口径。"
                 "几何不一致已显式声明：cluster_windows 内部用 StandardScaler+欧氏，归属用余弦+原始空间。")
        L.append("")
    L.append("## 7. 动量基准对照（防平凡复读）")
    L.append("")
    L.append("| tf | 臂 | 段 | win_rate | base_up | base_mom | base_majority | lift_vs_mom(pp) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tf in tfs:
        for r in (hyp.get(tf) or []):
            for seg in ("discovery", "validation", "holdout"):
                g = r.get(seg)
                if not g or not g.get("n"):
                    continue
                L.append(f"| {tf} | {r['name']} | {seg} | {_mdn(g.get('win_rate'))} | "
                         f"{_mdn(g.get('base_up'))} | {_mdn(g.get('base_mom'))} | "
                         f"{_mdn(g.get('base_majority'))} | {_mdn(g.get('lift_vs_mom_pp'), 2)} |")
    L.append("")
    L.append("CONFIRMED 必须 `lift_vs_mom_pp > 0`：跑不赢「段内动量延续」这个零成本基准的形状相似性，不值得认领。")
    L.append("")
    L.append("## 8. 负结果与局限")
    L.append("")
    for tf in tfs:
        inside, tot, names = 0, 0, []
        for r in (hyp.get(tf) or []):
            pl = r.get("placebo_lift_ci")
            d = (r.get("discovery") or {}).get("lift_pp")
            if isinstance(pl, list) and len(pl) == 2 and d is not None:
                tot += 1
                if pl[0] <= d <= pl[1]:
                    inside += 1
                    names.append(r["name"])
        if tot:
            L.append(f"- **安慰剂对照（{tf}）**：{tot} 个有零分布的臂中，**{inside} 个**的 discovery 观测 lift "
                     f"落在「邻居标签随机置换」零分布的 95% 区间内（即与噪声不可区分）"
                     + (f"：{', '.join(names)}。" if names else "。")
                     + "安慰剂仅在 discovery 段跑，holdout 未触碰。")
    L.append(f"- **功效结构性不足**：下表为每臂 holdout 的可检测下限（MDE = "
             f"`min_detectable_effect(n, p0=0.5, 单侧 α=0.05, 功效 0.8)` 正态近似）。"
             f"各臂普遍高于 min_lift_pp={CFG['min_lift_pp']}pp，因此「未 CONFIRMED」**不等于「无效应」**。"
             f"power_note 只标注、**不进入裁决阶梯**（房内先例 l1_tester）；"
             f"终裁硬闸 = n_h ≥ n_min_holdout({CFG['n_min_holdout']})。")
    L.append("")
    L.append("| tf | 臂 | segmode | holdout查询数 | MDE(pp) | holdout有效n | MDE(pp) | 裁决 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for pr in _power_rows(tfs, per_tf, hyp):
        L.append(f"| {pr['tf']} | {pr['variant']} | {pr['segmode']} | {pr['n_holdout_queries']} | "
                 f"{_mdn(pr['mde_holdout_queries_pp'], 2)} | {_mdn(pr['n_holdout_eval'])} | "
                 f"{_mdn(pr['mde_holdout_eval_pp'], 2)} | {pr['verdict'] or '—'} |")
    L.append("")
    L.append("两个分母分开读：`holdout查询数` 是设计上限（含弃权），`holdout有效n` 是弃权与 "
             "`label_valid` 过滤后裁决真正吃到的样本——功效以后者为准。未触碰 holdout 的臂"
             "（REJECT_PREHOLDOUT）有效 n 为空，MDE 只按查询数给出。")
    L.append("")
    L.append("- **重叠窗 p 值反保守**：主规格 query_stride=seg_len 非重叠；`seg_fixed_overlap6` 臂存在重叠，"
             "其头条 CI 一律认 `run_block_ci` 的 block_ci_low > base_up，不认二项 p 值。")
    L.append("- **多重检验**：OFAT 冻结网格已计入 total_tests，BH-FDR 按 tf 分族；"
             "任何跑后新增变体须重新预注册并在全新切分重验。")
    L.append("- **市场状态依赖**：720d 单一标的（BTC）单一数据集，未做跨标的/跨时期稳健性。")
    L.append("- **执行可行性未检验**：未建模滑点、报价可得性、成交延迟、资金费率变化。")
    n_prom = sum(1 for tf in tfs for r in (hyp.get(tf) or []) if r.get("verdict") == "PROMISING")
    n_conf = sum(1 for tf in tfs for r in (hyp.get(tf) or []) if r.get("verdict") == "CONFIRMED")
    L.append(f"- **本案结果**：CONFIRMED {n_conf} 个、PROMISING {n_prom} 个。PROMISING 意味着未同时满足六条硬闸"
             "（典型卡点：holdout 胜率未越过 52.0408% 打平线 → ev_ci_low < 0），**不得当作可交易结论**。")
    L.append("")
    L.append("## 9. 诚实声明")
    L.append("")
    L.append("> 胜率差异是条件概率描述，按 0.50 双边口径计 EV（`ev(p, 0.5)`，FEE=0.02 + PREMIUM=0.01），"
             "未经滑点 / 报价可得性 / 执行延迟检验，**不直接构成可交易信号**；"
             "CONFIRMED 仅以冻结 holdout 为准。")
    L.append("")
    L.append(f"- 打平胜率 = {be:.6f}；`above_breakeven` 按 Wilson 下界 > 打平胜率判定。")
    L.append("- CSV 中 `label_*` / `hit` 列为**未来信息**，仅记录与描述用，不参与任何 ex-ante 决策。")
    L.append("- 字节复现：`generated_at` 只进 json/md，**不进 CSV**；CSV 一律 `newline='\\n'` + "
             "anchor 升序 + 固定浮点格式 `.6g`。")
    L.append("")
    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))
    log(f"report：report.md {len(L)} 行 -> {os.path.relpath(path, ROOT)}")


def stage_report(out: str, tfs: tuple, split_mode: str = "60/20/20",
                 limit_segments=None, plot: bool = False, exploratory: bool = False) -> None:
    """run_config.json + report.md + knn_predictions_{tf}.csv（+ 可选 shapes_{tf}.png）。"""
    hyp = {}
    hp = os.path.join(out, "hypothesis_tests.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            hyp = json.load(f)
    exploratory = bool(exploratory or limit_segments is not None)
    per_tf, census, paired = {}, {}, {}
    for tf in tfs:
        p = os.path.join(out, f"_scan_{tf}.pkl")
        if not os.path.exists(p):
            log(f"report：{tf} 缺少 scan 产物，跳过")
            continue
        scan = _unpkl(p)
        if scan.get("cfg_sha256") != cfg_sha256():
            raise RuntimeError(f"{tf}: CFG 在 scan 后变更（哈希不符）——违反冻结纪律")
        if scan.get("limit_segments") != limit_segments:
            log(f"report：{tf} scan 的 limit_segments={scan.get('limit_segments')} 与当前"
                f" {limit_segments} 不一致，以 pkl 为准")
            limit_segments = scan.get("limit_segments")
            exploratory = bool(limit_segments is not None)
        kl, atr, labels = _load_tf(tf)
        per_tf[tf] = {"scan": scan, "kl": kl, "labels": labels}
        census[tf] = _segment_census(out, tf, kl, atr, limit_segments)
        paired[tf] = _paired_cosine_dtw(scan)
        _write_predictions_csv(out, tf, scan, kl, labels)
        if plot:
            _plot_shapes(out, tf, kl, atr, labels, limit_segments)
    if not per_tf:
        log("report：无可用 scan 产物，跳过")
        return
    _write_run_config(out, tfs, per_tf, hyp, split_mode, exploratory)
    _write_report_md(out, tfs, per_tf, hyp, census, paired, split_mode, exploratory)


# ============================ 层9：main + 阶段化 + 自检 ============================

_STAGES = ("segment", "scan", "patterns", "test", "report")
_TMP_DIR = os.path.join(ROOT, ".pytest_tmp")


def _synthetic_klines(n=240, seed=7, bar_ms=300_000, gap_at=None):
    """内联合成 Klines（禁 import tests/）：随机游走 + 可选断点。"""
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0025, n)))
    o = np.empty(n)
    o[0] = c[0]
    o[1:] = c[:-1]
    wick = np.abs(rng.normal(0.0, 0.0012, n)) * c
    h = np.maximum(o, c) + wick
    l = np.minimum(o, c) - wick
    v = rng.uniform(1.0, 10.0, n)
    t = np.arange(n, dtype=np.int64) * bar_ms + 1_700_000_000_000
    cont = np.ones(n, dtype=bool)
    cont[0] = False
    if gap_at is not None and 0 < gap_at < n:
        t[gap_at:] += bar_ms * 5
        cont[gap_at] = False
    return Klines(t=t, o=o, h=h, l=l, c=c, v=v, cont=cont)


def _self_check() -> None:
    """启动即跑的硬校验（任一失败 raise）。全部基于内联合成 Klines，禁 import tests/。

    覆盖 6 类：① 因果硬校验 ② 无自匹配/标签根不重叠 ③ L2 归一 ④ cfg 双端
    ⑤ 字节复现 ⑥ 数学恒等（+ ZigZag no-repaint 前缀不变性、valid 掩码对齐）。
    """
    t0 = time.time()
    m = int(CFG["resample_m"])

    # ---- ⑥ 数学恒等 ----
    v = np.array([1.0, -2.0, 0.5, 3.0])
    assert abs(cosine_sim(v, v) - 1.0) < 1e-12, "cosine_sim(v,v) != 1.0"
    assert abs(cosine_sim(np.zeros_like(v), v)) < 1e-12, "cosine_sim(0,v) != 0.0"
    a = np.array([1.0, 1.4, 1.1, 0.8, 1.2, 1.6])
    d0 = dtw_band_cost(a, a[None, :], int(CFG["dtw_band"]))
    assert d0.shape == (1,) and abs(float(d0[0])) < 1e-9, "dtw_band_cost(a,a) != 0"
    be0 = breakeven_win_rate(0.5)
    assert abs(ev(be0, 0.5)) < 1e-9, f"打平胜率 {be0} 处 EV != 0"
    W = np.random.default_rng(3).normal(100.0, 2.0, (7, 12))
    Rm = _resample_matrix(W, m)
    for i in range(len(W)):
        assert np.allclose(Rm[i], resample_path(W[i], m), atol=1e-12), \
            "_resample_matrix 与 resample_path 不等价"
    zp, dg = normalize_path(W, np.full(len(W), 1.5), "zscore")
    assert not dg.any(), "正常段被误标 degenerate"
    assert np.allclose(zp.mean(axis=1), 0.0, atol=1e-9), "z-score 后 mean != 0"
    raw = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
    assert np.allclose(raw.std(axis=1), 1.0, atol=1e-9), "z-score 后 std != 1"
    assert np.allclose(np.linalg.norm(zp, axis=1), 1.0, atol=1e-9), "L2 归一失败"
    zf, dgf = normalize_path(np.ones((1, m)) * 5.0, np.array([1.0]), "zscore")
    assert bool(dgf[0]) and np.allclose(zf[0], 0.0), "平坦段未标 degenerate / 未置零"
    for mode in ("atr", "logret"):
        zo, dgo = normalize_path(W, np.full(len(W), 1.5), mode)
        assert np.allclose(np.linalg.norm(zo[~dgo], axis=1), 1.0, atol=1e-9), f"{mode} L2 失败"
        assert np.allclose(zo.mean(axis=1), 0.0, atol=1e-9), f"{mode} 未去均值"

    # ---- ⑥ segment_fixed 的 valid 掩码按 start 对齐（30 根合成手算比对）----
    klg = _synthetic_klines(n=30, seed=11, gap_at=20)
    Lg = 4
    sfg = segment_fixed(klg, Lg, 1)
    exp = [bool(klg.cont[int(s) + 1: int(s) + Lg].all()) for s in sfg["start"]]
    assert sfg["valid"].tolist() == exp, "segment_fixed valid 未按段首根对齐"
    assert (sfg["start"] == sfg["anchor"] - Lg + 1).all(), "start != anchor-L+1"
    assert (sfg["anchor"] <= 28).all(), "anchor 越界（标签需 anchor+1<=n-1）"
    assert (sfg["query_idx"] < len(sfg["start"])).all(), "query_idx 越界"
    gap_bad = [j for j in range(len(sfg["start"]))
               if int(sfg["start"][j]) < 20 <= int(sfg["anchor"][j])]
    assert gap_bad and not sfg["valid"][gap_bad].any(), "跨断点段未标 invalid"

    # ---- ③ L2 归一 + ZigZag 结构守卫（合成数据）----
    kl = _synthetic_klines(n=400, seed=5)
    atr = atr_series(kl)
    labels = build_labels(kl)
    sf = segment_fixed(kl, int(CFG["seg_len"]), int(CFG["seg_step"]))
    X, rv = build_shape_matrix(kl, sf, atr, m, CFG["norm_mode"])
    assert X.shape == (len(sf["start"]), m), "build_shape_matrix 形状错"
    assert rv.shape == (len(sf["start"]),)
    if rv.any():
        assert np.allclose(np.linalg.norm(X[rv], axis=1), 1.0, atol=1e-5), \
            "build_shape_matrix 输出未 L2 归一"
    pv = zigzag_pivots(kl.c, atr, float(CFG["zigzag_atr"]), kl.cont)
    if pv:
        assert all(int(p[3]) > int(p[0]) for p in pv), "confirm_idx 未严格晚于 pivot_idx（重绘）"
        ks = [int(p[2]) for p in pv]
        assert all(ks[i] != ks[i + 1] for i in range(len(ks) - 1)), "ZigZag 峰谷未交替"
        cf = [int(p[3]) for p in pv]
        assert all(cf[i] <= cf[i + 1] for i in range(len(cf) - 1)), "confirm_idx 非单调"
        # no-repaint 前缀不变性：截断到 confirm_idx 重跑，已确认摆动点必须完全一致
        for probe in sorted({len(pv) // 2, len(pv) - 1}):
            cut_i = int(pv[probe][3])
            sub = Klines(t=kl.t[:cut_i + 1], o=kl.o[:cut_i + 1], h=kl.h[:cut_i + 1],
                         l=kl.l[:cut_i + 1], c=kl.c[:cut_i + 1], v=kl.v[:cut_i + 1],
                         cont=kl.cont[:cut_i + 1])
            pv2 = zigzag_pivots(sub.c, atr_series(sub), float(CFG["zigzag_atr"]), sub.cont)

            def _key(seq, lim_i=cut_i):
                return [(int(x[0]), int(x[3]), int(x[2])) for x in seq if int(x[3]) <= lim_i]

            assert _key(pv2) == _key(pv), "ZigZag 重绘：前缀截断后历史摆动点发生改变"
    sz = segment_zigzag(kl, atr, float(CFG["zigzag_atr"]), int(CFG["zigzag_min_bars"]),
                        int(CFG["zigzag_max_bars"]))
    if len(sz["start"]):
        assert (sz["start"] >= _ATR_WARMUP).all(), "ZigZag 热身伪影未丢弃"
        assert (sz["anchor"] >= sz["end"]).all(), "ZigZag 锚点早于摆动点（重绘风险）"
        assert (sz["confirm_lag"] >= 0).all(), "confirm_lag < 0"
        assert (sz["anchor"] <= len(kl) - 2).all(), "ZigZag anchor 越界"
        assert (sz["n_bars"] >= CFG["zigzag_min_bars"]).all()
        assert (sz["n_bars"] <= CFG["zigzag_max_bars"]).all()
        Xz, rvz = build_shape_matrix(kl, sz, atr, m, CFG["norm_mode"])
        if rvz.any():
            assert np.allclose(np.linalg.norm(Xz[rvz], axis=1), 1.0, atol=1e-5)

    # ---- ①② 因果硬校验 + 无自匹配 + 标签根不重叠 + ⑤ 确定性 ----
    n_seg = len(sf["start"])
    cut = int(n_seg * 0.4)
    ref = np.arange(0, cut, dtype=np.int64)
    ref = ref[rv[ref]]
    qsel = sf["query_idx"][sf["query_idx"] >= cut]
    assert len(ref) > 0 and len(qsel) > 0, "自检合成数据不足以构造参考/查询集"
    ra = sf["anchor"][ref]
    qa = sf["anchor"][qsel]
    qs = sf["start"][qsel]
    kk = int(CFG["k_neighbors"])
    nn_idx, nn_sim, nav = cosine_topk(X[qsel], X[ref], qa, qs, ra, kk, 1, int(CFG["knn_block"]))
    nn2, sim2, nav2 = cosine_topk(X[qsel], X[ref], qa, qs, ra, kk, 1, int(CFG["knn_block"]))
    assert np.array_equal(nn_idx, nn2) and np.array_equal(nav, nav2) \
        and np.array_equal(nn_sim, sim2), "cosine_topk 非确定性（分块边界/破平局不稳定）"
    ok = nav >= kk
    assert ok.any(), "自检：无任何查询达到 k 个因果参考"
    sel = nn_idx[ok]
    assert (sel >= 0).all(), "nav>=k 的行不应有空槽"
    assert bool((ra[sel].max(axis=1) <= qs[ok] - CFG["causal_lag"]).all()), \
        "① 因果违规：ra > qs - causal_lag"
    assert bool((ra[sel] != qa[ok][:, None]).all()), "② 自匹配泄漏"
    assert bool((ra[sel] + 1 < qs[ok][:, None]).all()), "② 参考标签根未严格早于查询段首根"
    assert bool((ra[sel] < qs[ok][:, None]).all()), "参考段与查询段位置重叠"
    assert bool(np.isfinite(nn_sim[ok]).all()), "nn_sim 出现非有限值"
    assert bool((nn_sim[ok] <= 1.0 + 1e-5).all() and (nn_sim[ok] >= -1.0 - 1e-5).all()), \
        "余弦超出 [-1,1]"
    # 行内 sim 降序 + 平局按参考索引升序（确定性破平局）
    srt = np.sort(nn_sim[ok], axis=1)[:, ::-1]
    assert np.allclose(nn_sim[ok], srt, atol=1e-12), "nn_sim 未降序"
    # DTW 重排臂同构 + 确定性
    di, ds, dv = dtw_rerank(X[qsel], X[ref], qa, qs, ra, min(kk, 5), 1,
                            int(CFG["dtw_recall_n"]), int(CFG["dtw_band"]), int(CFG["knn_block"]))
    di2, ds2, dv2 = dtw_rerank(X[qsel], X[ref], qa, qs, ra, min(kk, 5), 1,
                               int(CFG["dtw_recall_n"]), int(CFG["dtw_band"]), int(CFG["knn_block"]))
    assert np.array_equal(di, di2) and np.array_equal(ds, ds2) and np.array_equal(dv, dv2), \
        "dtw_rerank 非确定性"
    dok = dv >= 1
    if dok.any() and (di[dok] >= 0).any():
        dsel = di[dok]
        dm = dsel >= 0
        ra_d = ra[dsel]
        qs_d = np.broadcast_to(qs[dok][:, None], ra_d.shape)
        assert bool((ra_d[dm] < qs_d[dm]).all()), "DTW 臂因果违规"

    # ---- 投票 / 基准 / eval_arm 同分母配对 ----
    ref_up = labels["label_up"][ra]
    ref_valid = labels["label_valid"][ra]
    v1 = knn_vote(ref_up, ref_valid, nn_idx, nn_sim, nav, 1, float(CFG["consensus_margin"]))
    v2 = knn_vote(ref_up, ref_valid, nn2, sim2, nav2, 1, float(CFG["consensus_margin"]))
    assert (v1["predicted"] == v2["predicted"]).all() and (v1["abstain"] == v2["abstain"]).all()
    assert set(np.unique(v1["predicted"]).tolist()) <= {-1, 0, 1}
    assert bool((v1["abstain"] | (v1["predicted"] != 0)).all()), "未弃权却 predicted==0"
    mom = momentum_baseline(sf, kl.c)
    e = eval_arm(v1["predicted"], v1["abstain"], labels["label_up"][qa],
                 labels["label_valid"][qa], mom[qsel], qa)
    if e["n"] > 0:
        assert 0.0 <= e["win_rate"] <= 1.0
        assert abs(e["lift_pp"] - (e["win_rate"] - e["base_up"]) * 100.0) < 1e-9
        assert abs(e["lift_vs_mom_pp"] - (e["win_rate"] - e["base_mom"]) * 100.0) < 1e-9
        assert e["base_majority"] >= e["base_up"] - 1e-12, "多数类基准低于 always-UP"
        assert e["k"] <= e["n"]
    e0 = eval_arm(v1["predicted"], np.ones(len(qa), bool), labels["label_up"][qa],
                  labels["label_valid"][qa], mom[qsel], qa)
    assert e0["n"] == 0 and not np.isfinite(e0["win_rate"]), "全弃权应 n==0 且 win_rate=nan"

    # ---- 层5 编排端到端（合成数据，min_refs 降为 1）----
    q5 = _retrieve_arm(sf, X, rv, labels, mom, {**CFG, "min_refs": 1, "k_neighbors": 5})
    assert len(q5["qa"]) == len(sf["query_idx"]), "_retrieve_arm 查询数不符"
    assert set(np.unique(q5["q_split"]).tolist()) <= {0, 1, 2}
    assert q5["i1"] < q5["i2"] <= q5["n_seg"]

    # ---- ④ cfg 双端校验（写 pkl 再读回）----
    os.makedirs(_TMP_DIR, exist_ok=True)
    pc = os.path.join(_TMP_DIR, "_selfcheck_cfg.pkl")
    _pkl(pc, {"cfg_sha256": cfg_sha256()})
    assert _unpkl(pc)["cfg_sha256"] == cfg_sha256(), "④ cfg 双端校验失败"

    # ---- ⑤ 字节复现（同参跑两次 CSV 逐字节 ==）----
    hdr = ["i", "anchor", "predicted", "abstain", "consensus", "top1_sim", "n_avail"]
    rows = [[i, int(qa[i]), int(v1["predicted"][i]), bool(v1["abstain"][i]),
             _fmt(float(v1["consensus"][i])), _fmt(float(v1["top1_sim"][i])), int(nav[i])]
            for i in range(len(qa))]
    pa = os.path.join(_TMP_DIR, "_selfcheck_a.csv")
    pb = os.path.join(_TMP_DIR, "_selfcheck_b.csv")
    _write_csv(pa, hdr, rows)
    _write_csv(pb, hdr, rows)
    with open(pa, "rb") as f1, open(pb, "rb") as f2:
        assert f1.read() == f2.read(), "⑤ CSV 字节复现失败"
    with open(pa, encoding="utf-8", newline="") as f1:
        assert f1.readline().rstrip("\r\n") == ",".join(hdr), "CSV 表头不符"
    log(f"_self_check 全过（因果/无自匹配/L2/cfg双端/字节复现/数学恒等/ZigZag无重绘）"
        f" 耗时 {time.time() - t0:.1f}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="BTC 5m/15m K线分段形状 kNN 因果检索离线研究（9 层纯函数 + 三段裁决）")
    ap.add_argument("--tf", default="all", help="all | 5m | 15m（逗号可多选）")
    ap.add_argument("--stage", default="all", help="all | 逗号列表：" + ",".join(_STAGES))
    ap.add_argument("--out", default=os.path.join(OUT0, "shape_segment_knn_720d"))
    ap.add_argument("--split", default="60/20/20", choices=("60/20/20", "70/30"),
                    help="70/30 仅作对照旗标：改 CFG 三段比例 → cfg_sha256 随之变 → 缓存自动失效")
    ap.add_argument("--plot", action="store_true",
                    help="额外落 shapes_{tf}.png（matplotlib 缺失则 try/except 跳过）")
    ap.add_argument("--force", action="store_true", help="强制重建分段 npz / scan pkl 缓存")
    ap.add_argument("--limit-segments", type=int, default=None,
                    help="探索性：确定性截断段数（冒烟）；打 note 并硬跳 holdout")
    ap.add_argument("--k", type=int, default=None, help="探索性：覆盖 k_neighbors")
    ap.add_argument("--sim", default=None, choices=("cosine", "dtw"), help="探索性：覆盖主相似度")
    args = ap.parse_args(argv)

    tfs = TFS if args.tf.strip().lower() == "all" else \
        tuple(x.strip() for x in args.tf.split(",") if x.strip())
    for tf in tfs:
        if tf not in TFS:
            raise SystemExit(f"未知 --tf {tf!r}（支持 all|{'|'.join(TFS)}）")
    want = _STAGES if args.stage.strip().lower() == "all" else \
        tuple(x.strip() for x in args.stage.split(",") if x.strip())
    for s in want:
        if s not in _STAGES:
            raise SystemExit(f"未知 --stage {s!r}（支持 all|{'|'.join(_STAGES)}）")
    stages = tuple(s for s in _STAGES if s in want)   # 固定执行顺序（依赖关系不可换）

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(_TMP_DIR, exist_ok=True)

    # 探索性覆盖：改 CFG → cfg_sha256 变 → 缓存自动失效；产物打 note 并硬跳 holdout
    exploratory = bool(args.limit_segments is not None or args.k is not None
                       or args.sim is not None or args.split == "70/30")
    if args.split == "70/30":
        CFG["discovery_frac"], CFG["validation_frac"], CFG["holdout_frac"] = 0.7, 0.0, 0.3
    if args.k is not None:
        CFG["k_neighbors"] = int(args.k)
    if args.sim is not None:
        CFG["sim"] = args.sim

    if args.force:
        removed = 0
        for fn in sorted(os.listdir(args.out)):
            if ((fn.startswith("_segments_") and fn.endswith(".npz"))
                    or (fn.startswith("_scan_") and fn.endswith(".pkl"))):
                os.remove(os.path.join(args.out, fn))
                removed += 1
        log(f"--force：清理 {removed} 个阶段缓存")

    log(f"cfg_sha256={cfg_sha256()[:16]} split={args.split} tf={','.join(tfs)} "
        f"stage={','.join(stages)} out={os.path.relpath(args.out, ROOT)}")
    if exploratory:
        log("探索性模式：未预注册，不触碰 holdout，不得直接认领")

    _self_check()

    t0 = time.time()
    if "segment" in stages:
        stage_segment(args.out, tfs, args.limit_segments)
    if "scan" in stages:
        stage_scan(args.out, tfs, args.limit_segments)
    if "patterns" in stages:
        stage_patterns(args.out, tfs, args.limit_segments)
    if "test" in stages:
        stage_test(args.out, tfs, exploratory)
    if "report" in stages:
        stage_report(args.out, tfs, args.split, args.limit_segments, args.plot, exploratory)
    log(f"完成，总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
