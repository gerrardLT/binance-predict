#!/usr/bin/env python
"""「双顶拒绝」形态探测（用户观察形态的形式化检验，720d × 5m/15m/1h/4h）。

形态（锚定在 B 根——触发短阴线的收盘时刻，ex-ante 可观察）：
    H（首顶） → 回撤 → 再冲高接近 H（次顶）
    A 根 = B 前一根：大实体阳线 + 短上影（冲顶根）
    B 根：短实体阴线 + 上影极短（类光头）+ 明显下影线（拒绝根）
    目标：B 之后紧跟 ≥3 根连续阴线

纪律：
- PARAMS 在任何数据观察前冻结（哈希入 manifest）；敏感性网格仅作探索性标注
- 0.6/0.2/0.2 三段时序切分；验证段方向一致才触碰一次冻结 holdout
- 基准 = 同周期「任意棒后 ≥3 连阴」的段内发生率；安慰剂 = 随机平移锚点
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from binance_predict.backtest.stats import exact_binomial_p, wilson  # noqa: E402
from binance_predict.discovery import load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, "output")
BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
TFS = ("5m", "15m", "1h", "4h")

# ---- 预注册参数 v2（用户图样校正后重新冻结；哈希入 manifest）----
# v1→v2 校正：次顶容差 0.5→1.5×ATR（允许“略低次顶/接近高点”）；
# A 实体 0.8→0.6（中大阳即可）；B 上影 0.15→0.25、下影 0.25→0.2；
# 新增 b_a_ratio：阴线实体须短于前阳（相对“较短”）；
# 新增 rally_atr：首顶须由上涨形成（排除下跌途中的反弹高）。
PARAMS = {
    "lookback": 20,        # 首顶 H 的回看窗口（根）
    "top_atr_tol": 1.5,    # 次顶接近 H 的容差（×ATR）
    "pullback_atr": 0.8,   # 两顶之间最小回撤深度（×ATR）
    "a_body_atr": 0.6,     # A 根最小实体（×ATR）
    "a_upper_atr": 0.3,    # A 根最大上影（×ATR）
    "b_body_atr": 0.7,     # B 根最大实体（×ATR）
    "b_min_body_atr": 0.05,  # B 根最小实体（排除 doji）
    "b_lower_atr": 0.2,    # B 根最小下影（×ATR）
    "b_upper_atr": 0.25,   # B 根最大上影（×ATR，类光头）
    "b_a_ratio": 0.7,      # B 实体 ≤ 0.7×A 实体（阴线相对较短）
    "rally_atr": 1.5,      # 首顶 H 须由上涨形成：H 前 12 根最低点到 H 的涨幅 ≥1.5×ATR
    "min_streak": 3,       # 目标：后续连续阴线数
    "seed": 11, "n_placebo": 20, "min_holdout_n": 15,
    "min_val_rel_lift": 0.10, "min_holdout_rel_lift": 0.10,
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def iso_utc(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def params_sha256() -> str:
    return hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode("utf-8")).hexdigest()


def data_csv_path(tf: str) -> str:
    if tf in ("5m", "15m"):
        return os.path.join(OUT0, f"klines_{tf}_720d.csv")
    return os.path.join(OUT0, "streak_research_720d", f"klines_{tf}_720d.csv")


# ============================ 形态检测 ============================

def detect_pattern(kl, bar_ms: int) -> dict:
    """向量化主体 + 候选环回查回撤时序，返回锚点（B 根）索引与几何量。"""
    n = len(kl)
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    atr = atr_series(kl)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    atr_safe = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    # ---- A 根：大实体阳线 + 短上影 ----
    a_ok = ((c > o) & np.isfinite(atr_safe)
            & (body >= PARAMS["a_body_atr"] * atr_safe)
            & (upper <= PARAMS["a_upper_atr"] * atr_safe))
    # ---- B 根：短实体阴线 + 类光头上影 + 明显下影 ----
    b_ok = ((c < o) & np.isfinite(atr_safe)
            & (body <= PARAMS["b_body_atr"] * atr_safe)
            & (body >= PARAMS["b_min_body_atr"] * atr_safe)
            & (upper <= PARAMS["b_upper_atr"] * atr_safe)
            & (lower >= PARAMS["b_lower_atr"] * atr_safe))
    if "b_a_ratio" in PARAMS:
        prev_body = np.roll(body, 1)
        b_ok = b_ok & (body <= PARAMS["b_a_ratio"] * prev_body)
    L = PARAMS["lookback"]
    # ---- 双顶几何（窗口 [i-L-1, i-2] 内的首顶 H）----
    cand0 = np.flatnonzero(b_ok & np.roll(a_ok, 1) & kl.cont & np.roll(kl.cont, 1))
    cand0 = cand0[(cand0 >= L + 1) & (cand0 <= n - 1)]
    hw = sliding_window_view(h, L)          # hw[k] = h[k:k+L]
    lw = sliding_window_view(l, L)
    anchors, geom = [], []
    for i in cand0:
        k = i - L - 1                        # 窗口 [i-L-1, i-2]
        j = int(np.argmax(hw[k]))            # 首顶位置（窗口内）
        Hj = float(hw[k][j])
        top2 = max(h[i - 1], c[i - 1])       # 次顶 = A 根最高/收盘
        if top2 < Hj - PARAMS["top_atr_tol"] * atr[i]:
            continue                         # 未接近首顶
        if h[i] > Hj + PARAMS["top_atr_tol"] * atr[i]:
            continue                         # B 根越过首顶过多 → 是突破不是双顶
        j_abs = k + j
        if j_abs + 1 > i - 2:
            continue                         # 首顶紧贴 A，无回撤空间
        seg_low = float(l[j_abs + 1: i - 1].min())
        depth = Hj - seg_low
        if depth < PARAMS["pullback_atr"] * atr[i]:
            continue                         # 回撤不够深
        rally = Hj - float(l[max(0, j_abs - 12): j_abs + 1].min())
        if rally < PARAMS.get("rally_atr", 0.0) * atr[i]:
            continue                         # H 不是上涨形成的高点（下跌途中的反弹高）
        anchors.append(i)
        geom.append({"H": Hj, "H_ts": int(kl.t[j_abs]), "top2": float(top2),
                     "depth_atr": depth / atr[i],
                     "a_body_atr": float(body[i - 1] / atr[i]),
                     "b_body_atr": float(body[i] / atr[i]),
                     "b_lower_atr": float(lower[i] / atr[i])})
    return {"anchors": np.array(anchors, dtype=int), "geom": geom,
            "dir": np.sign(c - o), "n": n}


def outcome_flags(kl, dir_: np.ndarray) -> np.ndarray:
    """每根棒 j：j+1..j+3 全阴且数据连续（≥3 连阴起始于下一根）。"""
    n = len(kl)
    f = np.zeros(n, dtype=bool)
    m = PARAMS["min_streak"]
    if n > m:
        d = dir_ < 0
        ok = d[1: n - m + 1].copy()
        for k in range(2, m + 1):
            ok &= d[k: n - m + k]
        f[: n - m] = ok & np.all(
            sliding_window_view(kl.cont[1:], m)[: n - m], axis=1)
    return f


def streak_len_from(kl, dir_: np.ndarray, j: int) -> int:
    """从 j 根起的阴线连根长度（数据断点截断）。"""
    n = len(kl)
    L = 0
    while j + L < n and dir_[j + L] < 0 and (L == 0 or kl.cont[j + L]):
        L += 1
    return L


# ============================ 敏感性网格（探索性，不触碰 holdout） ============================

# 主规格之外的变体：只报告发现/验证段富集度；任何变体若被采纳，
# 必须重新预注册并在全新数据/切分上检验，不得直接认领这里的数字。
SENSITIVITY_GRID = [
    {"name": "loose_tol", "overrides": {"top_atr_tol": 0.8, "pullback_atr": 0.5}},
    {"name": "deep_pullback", "overrides": {"pullback_atr": 1.5}},
    {"name": "loose_a", "overrides": {"a_body_atr": 0.6, "a_upper_atr": 0.4}},
    {"name": "loose_b", "overrides": {"b_body_atr": 0.7, "b_lower_atr": 0.15,
                                       "b_upper_atr": 0.3}},
    {"name": "wide_lookback", "overrides": {"lookback": 40}},
]


def run_sensitivity(tf: str, kl, base_flags: np.ndarray, out: str) -> list:
    global PARAMS
    saved = dict(PARAMS)
    n, m = len(kl), PARAMS["min_streak"]
    bi1, bi2 = int(n * 0.6), int(n * 0.8)
    rows = []
    for spec in SENSITIVITY_GRID:
        PARAMS = {**saved, **spec["overrides"]}
        det = detect_pattern(kl, BAR_MS[tf])
        anchors, na = det["anchors"], len(det["anchors"])
        i1 = int(na * 0.6)
        d = _stats(base_flags[anchors[:i1]], base_flags[:bi1][: n - m]) if i1 else {}
        v = _stats(base_flags[anchors[i1:int(na * 0.8)]],
                   base_flags[bi1:bi2][: n - m]) if na > i1 else {}
        rows.append({"tf": tf, "variant": spec["name"], "overrides": spec["overrides"],
                     "n_anchors": na,
                     "discovery": {k: _clean(x) for k, x in d.items()},
                     "validation": {k: _clean(x) for k, x in v.items()},
                     "note": "探索性：未预注册，不触碰 holdout，不得直接认领"})
        log(f"{tf} 敏感性 {spec['name']}: n={na} disc_lift={d.get('rel_lift')} "
            f"val_lift={v.get('rel_lift')}")
    PARAMS = saved
    return rows


# ============================ 统计 ============================

def _stats(hit: np.ndarray, base_flags: np.ndarray) -> dict:
    n, k = len(hit), int(hit.sum())
    rate = k / n if n else np.nan
    base = float(base_flags.mean()) if len(base_flags) else np.nan
    lo, hi = wilson(rate, n) if n else (np.nan, np.nan)
    p = exact_binomial_p(k, n, base) if (n and np.isfinite(base) and 0 < base < 1) else np.nan
    return {"n": int(n), "hits": k, "rate": rate, "base": base,
            "rel_lift": (rate / base - 1.0) if (np.isfinite(base) and base > 0
                                                 and np.isfinite(rate)) else np.nan,
            "ci_low": lo, "ci_high": hi, "p_value": p}


def _clean(v):
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def run_tf(tf: str, out: str) -> dict:
    kl = load_klines_csv(data_csv_path(tf), BAR_MS[tf])
    det = detect_pattern(kl, BAR_MS[tf])
    anchors, dir_ = det["anchors"], det["dir"]
    n, m = det["n"], PARAMS["min_streak"]
    base_flags = outcome_flags(kl, dir_)
    log(f"{tf}：K线 {n} 根，形态命中 {len(anchors)} 次；全量 ≥3 连阴基准率 "
        f"{base_flags[: n - m].mean():.4f}")

    # ---- 三段切分（锚点时序）----
    na = len(anchors)
    i1, i2 = int(na * 0.6), int(na * 0.8)
    seg_idx = [np.arange(0, i1), np.arange(i1, i2), np.arange(i2, na)]
    bi1, bi2 = int(n * 0.6), int(n * 0.8)
    bar_seg = [np.arange(0, bi1), np.arange(bi1, bi2), np.arange(bi2, n)]
    seg_names = ("discovery", "validation", "holdout")

    # ---- census 落盘（含未来标签，仅记录用）----
    cpath = os.path.join(out, f"pattern_census_{tf}.csv")
    with open(cpath, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["tf", "anchor_id", "b_ts", "b_price_close", "H_ts", "H_price",
                    "top2_price", "depth_atr", "a_body_atr", "b_body_atr",
                    "b_lower_atr", "seg", "follow_streak_len", "follow_ret_pct",
                    "hit_ge3_bear"])
        for r, (i, g) in enumerate(zip(anchors, det["geom"])):
            seg = seg_names[0] if r < i1 else (seg_names[1] if r < i2 else seg_names[2])
            L = streak_len_from(kl, dir_, i + 1) if i + 1 < n else 0
            ret = (kl.c[i + L] / kl.c[i] - 1.0) * 100 if (L and i + L < n) else np.nan
            w.writerow([tf, f"{tf}-P{r:05d}", iso_utc(kl.t[i]), f"{kl.c[i]:.2f}",
                        iso_utc(g["H_ts"]), f"{g['H']:.2f}", f"{g['top2']:.2f}",
                        f"{g['depth_atr']:.3f}", f"{g['a_body_atr']:.3f}",
                        f"{g['b_body_atr']:.3f}", f"{g['b_lower_atr']:.3f}", seg,
                        L, "" if not np.isfinite(ret) else f"{ret:.4g}",
                        bool(L >= m)])
    log(f"{tf}：census -> {os.path.relpath(cpath, ROOT)}")

    # ---- 三段统计（holdout 延迟到存活才触碰）----
    def seg_stat(si: int) -> dict:
        a = anchors[seg_idx[si]]
        b = bar_seg[si]
        hit = base_flags[a] if len(a) else np.array([], dtype=bool)
        return _stats(hit, base_flags[b[:(n - m)]])

    res: dict = {"tf": tf, "n_anchors": na, "n_bars": n,
                 "params_sha256": params_sha256(), "params": PARAMS}
    d = seg_stat(0)
    v = seg_stat(1)
    res["discovery"], res["validation"] = ({k: _clean(x) for k, x in d.items()},
                                           {k: _clean(x) for k, x in v.items()})
    gated = na >= 20 and d["n"] >= 10
    val_pass = (gated and np.isfinite(d["rel_lift"]) and np.isfinite(v["rel_lift"])
                and np.sign(d["rel_lift"]) == np.sign(v["rel_lift"]) > 0
                and v["rel_lift"] >= PARAMS["min_val_rel_lift"])
    res["gated"], res["validation_pass"] = bool(gated), bool(val_pass)

    # ---- 安慰剂（仅发现段零分布）----
    rng = np.random.default_rng(PARAMS["seed"])
    forb = np.zeros(n, dtype=bool)
    for i in anchors:
        forb[max(0, i - 5): min(n, i + 6)] = True
    pool = np.flatnonzero(~forb)
    pool = pool[pool <= n - m - 1]
    rates = []
    if len(pool) and d["n"]:
        for _ in range(PARAMS["n_placebo"]):
            s = rng.choice(pool, size=d["n"], replace=False)
            rates.append(float(base_flags[s].mean()))
    res["placebo"] = {"mean": _clean(np.mean(rates)) if rates else None,
                      "ci": ([float(x) for x in np.percentile(rates, [2.5, 97.5])]
                             if rates else None)}

    # ---- 冻结 holdout 终验（只触碰一次）----
    if val_pass:
        hstat = seg_stat(2)
        res["holdout"] = {k: _clean(x) for k, x in hstat.items()}
        hl, p_h, n_h = hstat["rel_lift"], hstat["p_value"], hstat["n"]
        ci_excl = np.isfinite(hstat["ci_low"]) and np.isfinite(hstat["base"]) \
            and hstat["ci_low"] > hstat["base"]
        if n_h < PARAMS["min_holdout_n"]:
            res["verdict"] = "INSUFFICIENT_SAMPLES"
        elif (np.isfinite(hl) and hl >= PARAMS["min_holdout_rel_lift"]
              and np.isfinite(p_h) and p_h < 0.05 and ci_excl):
            res["verdict"] = "CONFIRMED"
        elif np.isfinite(hl) and hl > 0 and np.isfinite(p_h) and p_h < 0.10:
            res["verdict"] = "PROMISING"
        elif np.isfinite(hl) and hl > 0:
            res["verdict"] = "WEAK"
        else:
            res["verdict"] = "REJECT"
    else:
        res["verdict"] = "REJECT_PREHOLDOUT" if gated else "INSUFFICIENT_SAMPLES"
    log(f"{tf}：裁决 {res['verdict']}（disc lift={d['rel_lift']:+.2f}, "
        f"val lift={v['rel_lift']:+.2f}）")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="双顶拒绝形态探测（用户观察形态形式化检验）")
    ap.add_argument("--tf", default="all")
    ap.add_argument("--out", default=os.path.join(OUT0, "double_top_probe_720d"))
    ap.add_argument("--sensitivity", action="store_true",
                    help="追加敏感性网格（仅 5m，探索性，不触碰 holdout）")
    args = ap.parse_args()
    tfs = TFS if args.tf == "all" else tuple(t.strip() for t in args.tf.split(",") if t.strip())
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for tf in tfs:
        if tf not in BAR_MS:
            raise SystemExit(f"未知周期：{tf}")
        results[tf] = run_tf(tf, args.out)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "params_sha256": params_sha256(), "params": PARAMS,
               "discipline": "参数预注册冻结；三段 0.6/0.2/0.2；holdout 只触碰一次；"
                             "安慰剂=随机非形态锚点重采样",
               "results": results}
    if args.sensitivity:
        kl = load_klines_csv(data_csv_path("5m"), BAR_MS["5m"])
        base_flags = outcome_flags(kl, np.sign(kl.c - kl.o))
        payload["sensitivity_5m"] = run_sensitivity("5m", kl, base_flags, args.out)
    with open(os.path.join(args.out, "probe_report.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=_clean)
    log(f"汇总 -> {os.path.join(args.out, 'probe_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
