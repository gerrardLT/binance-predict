#!/usr/bin/env python
"""上吊线 × 结构条件组合探测（720d，预注册假设集，三段切分+安慰剂+holdout一次）。

主结局（预注册）：P(下一根阴线 | 上吊线+条件X) —— 对比对象是**同段上吊线本身**的
基线率（增量价值），全局基准只作背景参考。
次结局（探索性标注）：P(连2阴)、P(连3阴)。

上吊线定义（与前期探索一致）：实体<=0.3×ATR、下影>=2×实体且>=0.3×ATR、
上影<=0.15×ATR、收盘距近20根最高 <=0.75×ATR（上涨后高位）。

裁决规则（冻结）：
- 门槛：发现段 n>=30
- 筛选：发现段 delta(k1) >= +2pp 且 精确二项 p < 0.10（vs 同段上吊线基线）
- 验证：验证段 delta(k1) > 0（方向一致）
- holdout（存活者只触碰一次）：CONFIRMED: delta>=+1pp 且 p<0.05；
  PROMISING: delta>0 且 p<0.15；WEAK: delta>0；否则 REJECT
- 安慰剂：从同段上吊线总体重采样 20 次（n 相同），报告 k1 率 95% 带
诚实性注记：上吊线定义与阳吊/阴吊拆分在预注册前已被观察过（H06 属半确认性）；
其余 H01-H05/H07-H15 均为新鲜假设。
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

from binance_predict.backtest.stats import exact_binomial_p  # noqa: E402
from binance_predict.discovery import load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, "output")
BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
TFS = ("5m", "15m", "1h", "4h")

CFG = {"seg": (0.6, 0.2, 0.2), "min_disc_n": 30, "disc_delta_pp": 0.02,
       "disc_p": 0.10, "hold_delta_pp": 0.01, "seed": 11, "n_placebo": 20}

# ---- 预注册假设集（冻结；sha256 入报告）----
# ATR 均指锚点处的 atr[anchor]；dir=sign(close-open)。
HYPOTHESES = {
    "H01_at_top":       "h 在近20根最高点 0.5×ATR 以内（上吊线就是最高点）",
    "H02_dt_geometry":  "双顶几何：[i-21,i-2] 内有首顶H(由>=1.5×ATR上涨形成)，"
                        "两顶回撤>=0.8×ATR，次顶(max(h[i-1],h[i]))距H<=1.5×ATR，h[i]未破H+1.5×ATR",
    "H03_yang_run3":    "前3根全为阳线（连阳耗尽）",
    "H04_runup_2atr":   "前5根(至i-1)涨幅>=2×ATR（急涨后的顶）",
    "H05_prev_big_yang":"前一根为中大阳线(实体>=0.6×ATR)（A阳冲顶+吊人触发）",
    "H06_yin_hm":       "上吊线本身收阴（阴吊）",
    "H07_lower3":       "下影>=3×实体（深探回收）",
    "H08_vol_surge":    "当根成交量>=1.2×20根均量（放量吊人）",
    "H09_session_us":   "美盘时段(UTC 16-24)",
    "H10_funding":      "资金费时点(UTC 0/8/16)",
    "H11_atr_high":     "ATR 分位>=0.75（高波环境）",
    "H12_atr_low":      "ATR 分位<=0.25（低波压缩）",
    "H13_dt_yin":       "H02 双顶几何 + H06 阴吊（拒绝共振）",
    "H14_run3_vol":     "H03 连阳耗尽 + H08 放量（量价耗尽）",
    "H15_yang_lower":   "H05 前中大阳 + H07 深下影（冲顶-深探结构）",
}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def hyp_sha256() -> str:
    blob = json.dumps({"cfg": CFG, "hyp": HYPOTHESES}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def data_csv_path(tf):
    if tf in ("5m", "15m"):
        return os.path.join(OUT0, f"klines_{tf}_720d.csv")
    return os.path.join(OUT0, "streak_research_720d", f"klines_{tf}_720d.csv")


def iso(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def rate_k(idx, dir_, cont, n, k):
    idx = np.asarray(idx, dtype=int)
    idx = idx[idx + k < n]
    if not len(idx):
        return float("nan"), 0
    ok = np.ones(len(idx), dtype=bool)
    for d in range(1, k + 1):
        ok &= dir_[idx + d] < 0
        ok &= cont[idx + d]
    return float(ok.mean()), len(idx)


def detect_hm(kl, atr) -> np.ndarray:
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    n = len(c)
    top20 = np.full(n, np.nan)
    top20[19:] = sliding_window_view(h, 20).max(axis=1)
    shape = ((body <= 0.3 * atr_s) & (lower >= 2.0 * body) & (lower >= 0.3 * atr_s)
             & (upper <= 0.15 * atr_s))
    pos = np.zeros(n, dtype=bool)
    pos[19:] = c[19:] >= top20[19:] - 0.75 * atr_s[19:]
    return shape & pos & np.isfinite(atr_s)


def build_conditions(kl, atr, hm: np.ndarray) -> dict:
    o, h, l, c, v, t = kl.o, kl.h, kl.l, kl.c, kl.v, kl.t
    n = len(c)
    body = np.abs(c - o)
    lower = np.minimum(o, c) - l
    dir_ = np.sign(c - o)
    yang = dir_ > 0
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    top20 = np.full(n, np.nan)
    top20[19:] = sliding_window_view(h, 20).max(axis=1)
    hour = (t // 3_600_000) % 24
    vmean20 = np.full(n, np.nan)
    vmean20[19:] = sliding_window_view(v, 20).mean(axis=1)
    prev_min5 = np.full(n, np.nan)
    prev_min5[5:] = sliding_window_view(l, 5).min(axis=1)[: n - 5]   # 窗口 [i-5, i-1]
    fin = np.isfinite(atr)
    pct = np.full(n, np.nan)
    pct[fin] = np.argsort(np.argsort(atr[fin])) / max(fin.sum() - 1, 1)

    cond = {
        "H01_at_top": h >= top20 - 0.5 * atr_s,
        "H03_yang_run3": yang & np.roll(yang, 1) & np.roll(yang, 2) & np.roll(yang, 3),
        "H04_runup_2atr": np.roll(c, 1) - prev_min5 >= 2.0 * atr_s,
        "H05_prev_big_yang": np.roll(yang, 1) & (np.roll(body, 1) >= 0.6 * atr_s),
        "H06_yin_hm": dir_ < 0,
        "H07_lower3": lower >= 3.0 * body,
        "H08_vol_surge": v >= 1.2 * vmean20,
        "H09_session_us": (hour >= 16) & (hour < 24),
        "H10_funding": np.isin(hour, (0, 8, 16)),
        "H11_atr_high": pct >= 0.75,
        "H12_atr_low": pct <= 0.25,
    }
    # H02 双顶几何（逐锚点回查）
    dt = np.zeros(n, dtype=bool)
    hw = sliding_window_view(h, 20)
    for i in np.flatnonzero(hm):
        if i < 21:
            continue
        k0 = i - 21                                   # 窗口 [i-21, i-2]
        j = int(np.argmax(hw[k0]))
        Hj = float(hw[k0][j])
        top2 = max(float(h[i - 1]), float(h[i]))
        if top2 < Hj - 1.5 * atr[i]:
            continue
        if h[i] > Hj + 1.5 * atr[i]:
            continue
        j_abs = k0 + j
        if j_abs + 1 > i - 2:
            continue
        if Hj - float(l[j_abs + 1: i - 1].min()) < 0.8 * atr[i]:
            continue
        if Hj - float(l[max(0, j_abs - 12): j_abs + 1].min()) < 1.5 * atr[i]:
            continue
        dt[i] = True
    cond["H02_dt_geometry"] = dt
    cond["H13_dt_yin"] = dt & cond["H06_yin_hm"]
    cond["H14_run3_vol"] = cond["H03_yang_run3"] & cond["H08_vol_surge"]
    cond["H15_yang_lower"] = cond["H05_prev_big_yang"] & cond["H07_lower3"]
    return cond


def _clean(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def seg_stats(idx, dir_, cont, n, k, p_hm):
    r, m = rate_k(idx, dir_, cont, n, k)
    hits = int(round(r * m)) if m else 0
    p = exact_binomial_p(hits, m, p_hm) if (m and 0 < p_hm < 1) else float("nan")
    return {"n": m, "rate": r, "delta": r - p_hm, "p": p}


def run_tf(tf: str, out: str) -> dict:
    kl = load_klines_csv(data_csv_path(tf), BAR_MS[tf])
    atr = atr_series(kl)
    dir_, cont, n = np.sign(kl.c - kl.o), kl.cont, len(kl.c)
    hm = detect_hm(kl, atr)
    anchors = np.flatnonzero(hm)
    cond = build_conditions(kl, atr, hm)
    bi1, bi2 = int(n * CFG["seg"][0]), int(n * (CFG["seg"][0] + CFG["seg"][1]))
    segs = (("disc", 0, bi1), ("val", bi1, bi2), ("hold", bi2, n))
    in_seg = [(anchors >= a) & (anchors < b) for _, a, b in segs]

    # 上吊线本身的三段基线（参照行）
    base = {}
    for (nm, a, b), msk in zip(segs, in_seg):
        base[nm] = {f"k{k}": rate_k(anchors[msk], dir_, cont, n, k)[0]
                    for k in (1, 2, 3)}
        base[nm]["n"] = int(msk.sum())
    log(f"{tf}: bars={n} HM={len(anchors)} 基线k1 disc/val/hold = "
        f"{base['disc']['k1']:.3f}/{base['val']['k1']:.3f}/{base['hold']['k1']:.3f}")

    # census 落盘
    cpath = os.path.join(out, f"hm_census_{tf}.csv")
    with open(cpath, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        cols = ["tf", "ts", "close", "seg"] + list(HYPOTHESES) + ["k1", "k2", "k3"]
        w.writerow(cols)
        for i in anchors:
            seg = "disc" if i < bi1 else ("val" if i < bi2 else "hold")
            row = [tf, iso(kl.t[i]), f"{kl.c[i]:.2f}", seg]
            row += [bool(cond[h][i]) for h in HYPOTHESES]
            row += [bool(rate_k([i], dir_, cont, n, k)[0]) if i + k < n else False
                    for k in (1, 2, 3)]
            w.writerow(row)
    log(f"{tf}: census -> {os.path.relpath(cpath, ROOT)}")

    # 逐假设检验
    results = {}
    for hid, desc in HYPOTHESES.items():
        idx_all = anchors[cond[hid][anchors]]
        r = {"desc": desc, "n_total": int(len(idx_all))}
        d_msk = (idx_all < bi1)
        v_msk = (idx_all >= bi1) & (idx_all < bi2)
        h_msk = idx_all >= bi2
        d = seg_stats(idx_all[d_msk], dir_, cont, n, 1, base["disc"]["k1"])
        v = seg_stats(idx_all[v_msk], dir_, cont, n, 1, base["val"]["k1"])
        r["disc"], r["val"] = {k: _clean(x) for k, x in d.items()}, \
                              {k: _clean(x) for k, x in v.items()}
        for k in (2, 3):
            r[f"disc_k{k}"] = _clean(rate_k(idx_all[d_msk], dir_, cont, n, k)[0]
                                     - base["disc"][f"k{k}"])
            r[f"val_k{k}"] = _clean(rate_k(idx_all[v_msk], dir_, cont, n, k)[0]
                                    - base["val"][f"k{k}"])
        gated = d["n"] >= CFG["min_disc_n"]
        screened = (gated and np.isfinite(d["delta"]) and d["delta"] >= CFG["disc_delta_pp"]
                    and np.isfinite(d["p"]) and d["p"] < CFG["disc_p"])
        val_pass = screened and np.isfinite(v["delta"]) and v["delta"] > 0
        r["gated"], r["screened"], r["val_pass"] = bool(gated), bool(screened), bool(val_pass)

        if screened:                                   # 安慰剂（发现段）
            rng = np.random.default_rng(CFG["seed"])
            pool = anchors[in_seg[0]]
            rates = []
            if len(pool) >= d["n"] and d["n"] > 0:
                for _ in range(CFG["n_placebo"]):
                    s = rng.choice(pool, size=d["n"], replace=False)
                    rates.append(rate_k(s, dir_, cont, n, 1)[0])
            r["placebo"] = {"mean": _clean(np.mean(rates)) if rates else None,
                            "ci": ([float(x) for x in np.percentile(rates, [2.5, 97.5])]
                                   if rates else None)}

        if val_pass:                                   # holdout 只触碰一次
            hh = seg_stats(idx_all[h_msk], dir_, cont, n, 1, base["hold"]["k1"])
            r["hold"] = {k: _clean(x) for k, x in hh.items()}
            for k in (2, 3):
                r[f"hold_k{k}"] = _clean(rate_k(idx_all[h_msk], dir_, cont, n, k)[0]
                                         - base["hold"][f"k{k}"])
            dl, p_h = hh["delta"], hh["p"]
            if hh["n"] < CFG["min_disc_n"]:
                r["verdict"] = "INSUFFICIENT_SAMPLES"
            elif dl >= CFG["hold_delta_pp"] and p_h < 0.05:
                r["verdict"] = "CONFIRMED"
            elif dl > 0 and p_h < 0.15:
                r["verdict"] = "PROMISING"
            elif dl > 0:
                r["verdict"] = "WEAK"
            else:
                r["verdict"] = "REJECT"
        else:
            r["verdict"] = "REJECT_PREHOLDOUT" if gated else "INSUFFICIENT_SAMPLES"
        results[hid] = r
        log(f"{tf} {hid}: n={r['n_total']} d={d['delta']:+.3f} v={v['delta']:+.3f} "
            f"-> {r['verdict']}")
    return {"tf": tf, "n_bars": n, "hm_total": int(len(anchors)),
            "hm_baseline": base, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="上吊线×结构组合探测（预注册）")
    ap.add_argument("--tf", default="all")
    ap.add_argument("--out", default=os.path.join(OUT0, "hm_combo_720d"))
    args = ap.parse_args()
    tfs = TFS if args.tf == "all" else tuple(x.strip() for x in args.tf.split(","))
    os.makedirs(args.out, exist_ok=True)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "spec_sha256": hyp_sha256(), "cfg": CFG, "hypotheses": HYPOTHESES,
               "primary": "P(next yin | HM+X) vs 同段 HM 基线；k2/k3 为探索性次结局",
               "results": {}}
    for tf in tfs:
        payload["results"][tf] = run_tf(tf, args.out)
    path = os.path.join(args.out, "hm_combo_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=_clean)
    log(f"汇总 -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
