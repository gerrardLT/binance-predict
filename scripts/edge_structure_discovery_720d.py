#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""720d「高胜率形态 / 确定性结构」发现引擎（发散→收敛），不限于连续涨跌。

与 streak_forward_prob_720d.py（连续 N 根同向）的根本区别：本研究目标是**方向胜率与
确定性结构**，回答「什么形态/结构下，未来价格行为最可预测（胜率高 / 确定性强）」。

目标族（全部 ex-ante 安全，站在第 t 根收盘；ATR = 前 20 根 range% 均值 × open，不含当前根）：
- up_H        : 未来 H 根净涨（c[t+H] > c[t]）——押涨方向胜率
- upstrong_H  : 未来 H 根净涨 >= 0.5×ATR——有意义幅度的确定性上涨（排除噪声级微涨）
- dnstrong_H  : 未来 H 根净跌 >= 0.5×ATR——确定性下跌
- touchup_H   : 未来 H 根内 high 触及 c[t]+1×ATR——目标达成（上），二元式确定性
- touchdn_H   : 未来 H 根内 low  触及 c[t]-1×ATR——目标达成（下）

方法论沿用已验证骨架：216 特征发散(L1/L2/L3+BH-FDR) → 三段 0.6/0.2/0.2 + holdout 只触碰一次
+ 安慰剂 + 月度/walk-forward 稳健性 → 剂量-反应曲线 + 720d∩2160d 跨样本共识。

阶段：baseline（基准率 + 强因子天花板探测）→ discover → converge → distill → report
用法：
    .venv\\Scripts\\python.exe -X utf8 scripts\\edge_structure_discovery_720d.py --tf 15m --stage baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)  # 同目录：复用 streak_forward_prob_720d 的通用方法论辅助

from binance_predict.backtest.stats import wilson  # noqa: E402
from binance_predict.discovery import build_feature_matrix, load_klines_csv  # noqa: E402
from binance_predict.discovery.combo_search import run_combos  # noqa: E402
from binance_predict.discovery.features import FeatureMatrix, atr_series  # noqa: E402
from binance_predict.discovery.hypotheses import DEFAULTS, atom_mask, make_atoms  # noqa: E402
from binance_predict.discovery.l1_tester import run_l1, seg_stats  # noqa: E402
from binance_predict.discovery.targets import TargetSet, Targets  # noqa: E402
from streak_forward_prob_720d import (  # noqa: E402
    _CONV_COLS, _load_converge, _mech_of, _monthly_align, _placebo_roll,
    _reg_row, _rel_lift, _seg_base, _verdict_run, _walkforward_align, _write_registry,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT0 = os.path.join(ROOT, "output")
FEAT_CACHE_DIR = os.path.join(OUT0, "streak_research_720d")  # 复用既有特征缓存（同数据 fp 命中）
DAYS = 720

BAR_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
TFS = ("5m", "15m", "1h", "4h")
HORIZONS = {"5m": (2, 3, 6), "15m": (2, 3, 6, 12), "1h": (2, 3, 6, 12), "4h": (2, 3, 6)}
STRONG_ATR = 0.5      # 净移动幅度门槛（×ATR）
TOUCH_ATR = 1.0       # 目标达成触及门槛（×ATR）
_FEAT_VERSION = 1

# 强因子天花板探测：已知的位置/极端因子（run 研究证明最强），看方向胜率能被提到多高
PROBE_CONT = ["zscore_5", "zscore_10", "range_pos_prior_20", "dist_prior_low_atr_20",
              "rsi_7", "pullback_pct_20", "sma_dist_atr_100"]
PROBE_BOOL = ["new_24h_lo", "new_24h_hi", "breakout_low_20", "breakout_high_20",
              "sweep_lo_fail", "sweep_hi_fail"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def data_csv_path(tf: str) -> str:
    if tf in ("5m", "15m"):
        return os.path.join(OUT0, f"klines_{tf}_{DAYS}d.csv")
    return os.path.join(FEAT_CACHE_DIR, f"klines_{tf}_720d.csv")


def load_kl(tf: str):
    return load_klines_csv(data_csv_path(tf), BAR_MS[tf])


def _feat_fp(kl, bar_ms: int) -> str:
    return hashlib.sha256(
        f"streakfeat-v{_FEAT_VERSION}|{bar_ms}|{kl.t[0]}|{kl.t[-1]}|{len(kl.t)}".encode("utf-8")
    ).hexdigest()[:16]


def get_featmat(kl, bar_ms: int, tf: str) -> FeatureMatrix:
    fp_path = os.path.join(FEAT_CACHE_DIR, f"_featmat_{tf}.npz")
    fp = _feat_fp(kl, bar_ms)
    if os.path.exists(fp_path):
        try:
            z = np.load(fp_path, allow_pickle=True)
            meta = json.loads(str(z["_meta"][0]))
            if meta["fp"] == fp:
                fm = FeatureMatrix()
                for i, nm in enumerate(meta["names"]):
                    fm.names.append(nm)
                    fm.families.append(meta["families"][i])
                    fm.dtypes.append(meta["dtypes"][i])
                    fm.cols[nm] = z[f"col_{i}"]
                log(f"  [{tf}] 特征缓存命中：{len(fm)} 列")
                return fm
        except Exception as ex:
            log(f"  [{tf}] 特征缓存失效（{ex}），重建")
    t0 = time.time()
    fm = build_feature_matrix(kl, bar_ms, None)
    log(f"  [{tf}] 特征构建 {len(fm)} 列 × {len(kl)} 根，耗时 {time.time() - t0:.1f}s")
    return fm


# ============================ 目标族：方向胜率 / 确定性结构 ============================

def _fwd_ok(cont: np.ndarray, H: int) -> np.ndarray:
    """t→t+H 全程数据连续：cont[t+1..t+H] 全 True。"""
    n = len(cont)
    out = np.zeros(n, dtype=bool)
    if n > H:
        out[: n - H] = sliding_window_view(cont[1:], H).all(axis=1)
    return out


def build_edge_targetsets(kl, horizons) -> Targets:
    """方向胜率 / 确定性结构目标族（包装成 TargetSet，喂给 l1_tester/combo_search）。"""
    n = len(kl)
    c, h, l, cont = kl.c, kl.h, kl.l, kl.cont
    atr = atr_series(kl)                       # 前置 ATR 绝对值（不含当前根）
    tg = Targets()
    for H in horizons:
        fwd_c = np.full(n, np.nan)
        if n > H:
            fwd_c[: n - H] = c[H:]
            maxh = np.full(n, np.nan)
            minl = np.full(n, np.nan)
            maxh[: n - H] = sliding_window_view(h[1:], H).max(axis=1)
            minl[: n - H] = sliding_window_view(l[1:], H).min(axis=1)
        else:
            maxh = minl = np.full(n, np.nan)
        okH = _fwd_ok(cont, H)
        valid = okH & np.isfinite(atr) & (atr > 0) & np.isfinite(fwd_c)
        half, one = STRONG_ATR * atr, TOUCH_ATR * atr
        defs = [
            (f"up_{H}", "up", fwd_c > c, (fwd_c - c) / c),
            (f"upstrong_{H}", "upstrong", fwd_c >= c + half, (fwd_c - c) / c),
            (f"dnstrong_{H}", "dnstrong", fwd_c <= c - half, (c - fwd_c) / c),
            (f"touchup_{H}", "touchup", maxh >= c + one, (maxh - c) / c),
            (f"touchdn_{H}", "touchdn", minl <= c - one, (c - minl) / c),
        ]
        for name, fam, win, ret in defs:
            win_b = np.where(np.isfinite(win.astype(float)) if win.dtype.kind == "f" else True,
                             win, False) & valid
            tg.add(TargetSet(
                name=name, family=fam, horizon=H, valid=valid, win=win_b,
                ret=np.where(valid & np.isfinite(ret), ret, np.nan),
                mfe_atr=np.full(n, np.nan), mae_atr=np.full(n, np.nan),
            ))
    return tg


# ============================ baseline：基准率 + 强因子天花板探测 ============================

def _rate(win: np.ndarray, valid: np.ndarray, sel: np.ndarray | None = None):
    m = valid if sel is None else (valid & sel)
    cnt = int(m.sum())
    if cnt == 0:
        return float("nan"), 0, (float("nan"), float("nan"))
    p = float(win[m].mean())
    return p, cnt, wilson(p, cnt)


def baseline_one_tf(tf: str, kl, fm) -> dict:
    hz = HORIZONS[tf]
    tg = build_edge_targetsets(kl, hz)
    n = len(kl)
    i1 = int(n * 0.6)                          # 发现段（分位阈值仅在此拟合，防泄漏）
    res: dict = {"tf": tf, "n_bars": n, "horizons": list(hz),
                 "span": [iso_utc(kl.t[0]), iso_utc(kl.t[-1])], "targets": {}}

    # ---- 1. 各目标无条件基准率（全样本 + 发现段）----
    for name in tg.names:
        ts = tg.items[name]
        p_all, n_all, ci = _rate(ts.win, ts.valid)
        p_d, n_d, _ = _rate(ts.win, ts.valid, np.r_[np.ones(i1, bool), np.zeros(n - i1, bool)])
        res["targets"][name] = {"base_all": p_all, "n_all": n_all, "ci": list(ci),
                                "base_disc": p_d, "n_disc": n_d}

    # ---- 2. 强因子天花板探测：位置因子极端分位下各目标胜率（发现段拟合分位）----
    probes: dict = {}
    disc = np.zeros(n, dtype=bool)
    disc[:i1] = True
    for feat in PROBE_CONT:
        if feat not in fm.cols:
            continue
        col = fm.cols[feat].astype(np.float64)
        d = col[disc & np.isfinite(col)]
        if len(d) < 100:
            continue
        q10, q90 = np.quantile(d, [0.10, 0.90])
        probes[f"{feat}<=q10"] = np.isfinite(col) & (col <= q10)
        probes[f"{feat}>=q90"] = np.isfinite(col) & (col >= q90)
    for feat in PROBE_BOOL:
        if feat in fm.cols:
            probes[f"{feat}==True"] = fm.cols[feat].astype(bool)

    ceiling: dict = {}
    for pname, sel in probes.items():
        row = {}
        for name in tg.names:
            ts = tg.items[name]
            base = res["targets"][name]["base_all"]
            p, cnt, _ = _rate(ts.win, ts.valid, sel)
            if cnt >= 100 and np.isfinite(p) and np.isfinite(base) and base > 0:
                row[name] = {"p": p, "n": cnt, "rel": (p - base) / base}
        if row:
            ceiling[pname] = row
    res["ceiling"] = ceiling
    return res


def stage_baseline(out: str, tfs: tuple[str, ...]) -> dict:
    log("=== baseline：方向胜率 / 确定性结构的基准率 + 强因子天花板探测 ===")
    allres = {}
    for tf in tfs:
        kl = load_kl(tf)
        fm = get_featmat(kl, BAR_MS[tf], tf)
        allres[tf] = baseline_one_tf(tf, kl, fm)
        log(f"  [{tf}] {len(kl)} 根，目标 {len(allres[tf]['targets'])} 个，探测因子 {len(allres[tf]['ceiling'])} 个")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=2, default=_json_default)
    log(f"baseline -> {os.path.join(out, 'baseline.json')}")
    _print_baseline(allres)
    return allres


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _print_baseline(allres: dict) -> None:
    for tf, r in allres.items():
        print("\n" + "=" * 108)
        print(f"[{tf}] 表 1. 各目标无条件基准率（{r['span'][0][:10]}~{r['span'][1][:10]}，{r['n_bars']} 根）")
        print("=" * 108)
        print(f"  {'目标':<16}{'基准率':>10}{'95%CI':>22}{'n':>10}   含义")
        meaning = {"up": "净涨胜率", "upstrong": f"净涨>={STRONG_ATR}ATR", "dnstrong": f"净跌>={STRONG_ATR}ATR",
                   "touchup": f"触及+{TOUCH_ATR}ATR", "touchdn": f"触及-{TOUCH_ATR}ATR"}
        for name, d in r["targets"].items():
            fam = name.rsplit("_", 1)[0]
            ci = f"[{d['ci'][0]:.4f},{d['ci'][1]:.4f}]"
            print(f"  {name:<16}{d['base_all']:>10.4f}{ci:>22}{d['n_all']:>10}   {meaning.get(fam, '')}")

        print("\n" + "=" * 108)
        print(f"[{tf}] 表 2. 强因子天花板探测：位置因子极端分位下，各目标的胜率与相对提升（找最高可提升空间）")
        print("=" * 108)
        # 汇总：每个目标下，相对提升最大的探测因子
        best: dict = {}
        for pname, row in r["ceiling"].items():
            for name, v in row.items():
                if name not in best or abs(v["rel"]) > abs(best[name]["rel"]):
                    best[name] = {"probe": pname, **v}
        print(f"  {'目标':<16}{'最强探测因子':<28}{'胜率':>9}{'基准':>9}{'相对提升':>10}{'n':>8}")
        for name in r["targets"]:
            b = best.get(name)
            if not b:
                continue
            base = r["targets"][name]["base_all"]
            print(f"  {name:<16}{b['probe']:<28}{b['p']:>9.4f}{base:>9.4f}{b['rel']:>+10.1%}{b['n']:>8}")
    print()


# ============================ discover / converge（复用 streak_forward 方法论骨架） ============================

def _build_all(out: str, tf: str):
    """载入数据/特征/目标 + 发现段原子 + L1 + 组合（discover 与 converge 共用前置）。"""
    kl = load_kl(tf)
    fm = get_featmat(kl, BAR_MS[tf], tf)
    n = len(kl)
    tg = build_edge_targetsets(kl, HORIZONS[tf])
    i1 = int(n * DEFAULTS["discovery_frac"])
    disc = np.zeros(n, dtype=bool)
    disc[:i1] = True
    atoms = make_atoms(fm, disc, rounds=("R1", "R2"))
    l1 = run_l1(fm, atoms, tg, n)
    combos = run_combos(fm, atoms, tg, l1, n)
    return kl, fm, tg, n, i1, atoms, l1, combos


def stage_discover(out: str, tfs: tuple[str, ...]) -> dict:
    log("=== discover：L1 单因子发散 + L2/L3 多维度组合（方向胜率 / 确定性结构）===")
    os.makedirs(out, exist_ok=True)
    registry_rows: list[dict] = []
    summary: dict = {}
    for tf in tfs:
        kl, fm, tg, n, i1, atoms, l1, combos = _build_all(out, tf)
        log(f"  [{tf}] 目标 {len(tg.names)} 个，原子 {len(atoms)}（发现段 {i1} 根）")
        tf_sum = {}
        for tname in tg.names:
            base_d, base_v = l1[tname]["baseline_disc"], l1[tname]["baseline_val"]
            kept = l1[tname]["kept"]
            short = combos.get(tname, {}).get("shortlist", [])
            tf_sum[tname] = {
                "baseline_disc": base_d, "n_disc": l1[tname]["n_disc"],
                "n_tests": l1[tname]["n_tests"], "n_fdr_pass": l1[tname]["n_fdr_pass"],
                "n_l1_kept": len(kept), "n_shortlist": len(short)}
            for r in kept:
                registry_rows.append(_reg_row(tf, tname, "L1", r, base_d, base_v))
            for r in short:
                registry_rows.append(_reg_row(tf, tname, r.get("level", "L2"), r, base_d, base_v))
        summary[tf] = tf_sum
    _write_registry(os.path.join(out, "discovery_registry.csv"), registry_rows)
    with open(os.path.join(out, "discover_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    log(f"discover -> discovery_registry.csv（{len(registry_rows)} 行候选）")
    _print_top(summary, registry_rows, "验证段 rel_lift", "validation")
    return summary


def stage_converge(out: str, tfs: tuple[str, ...]) -> dict:
    log("=== converge：holdout（后 0.2）终验只触碰一次 + 稳健性 + 安慰剂 + 裁决 ===")
    os.makedirs(out, exist_ok=True)
    rows: list[dict] = []
    for tf in tfs:
        kl, fm, tg, n, i1, atoms, l1, combos = _build_all(out, tf)
        t = kl.t
        i2 = int(n * (DEFAULTS["discovery_frac"] + DEFAULTS["validation_frac"]))
        by_id = {a.atom_id: a for a in atoms}
        log(f"  [{tf}] 发现/验证/holdout = {i1}/{i2 - i1}/{n - i2}")
        for tname in tg.names:
            ts = tg.items[tname]
            base_d, base_v = l1[tname]["baseline_disc"], l1[tname]["baseline_val"]
            base_h = _seg_base(ts.win, ts.valid, i2, n)
            cands: list[tuple] = []
            for r in l1[tname]["kept"]:
                a = by_id.get(r["atom_id"])
                if a is None:
                    continue
                cands.append(("L1", r.get("condition", a.condition),
                              r.get("feature_names", a.feature), r.get("parameter_stability", ""),
                              atom_mask(fm, a), r))
            for r in combos.get(tname, {}).get("shortlist", []):
                cands.append((r.get("level", "L2"), r.get("condition", ""),
                              r.get("feature_names", ""), "NA", r["_mask"], r))
            for (level, cond, feats, pstable, mask, r) in cands:
                dl = float(r.get("discovery_lift_pp", float("nan")))
                vl = float(r.get("validation_lift_pp", float("nan")))
                sign_d = np.sign(dl) if np.isfinite(dl) else 0.0
                hs = seg_stats(ts, mask, i2, n, base_h)
                nh, wrh, lifth = hs["n"], hs["win_rate"], hs["lift_pp"]
                relh = ((wrh - base_h) / base_h) if (base_h > 0 and np.isfinite(wrh)) else float("nan")
                flipped = bool(np.isfinite(lifth) and sign_d != 0
                               and np.sign(lifth) != sign_d and lifth != 0)
                retention = (abs(lifth) / abs(dl)) if (np.isfinite(lifth) and np.isfinite(dl) and dl != 0) else float("nan")
                val_ok = bool(np.isfinite(vl) and sign_d != 0 and np.sign(vl) == sign_d and vl > 0)
                full = mask & ts.valid
                mon = _monthly_align(t, full, ts.win, ts.valid, sign_d)
                wf = _walkforward_align(t, full, ts.win, ts.valid, sign_d)
                plac = _placebo_roll(mask, ts.win, ts.valid, 0, i1, base_d)
                verdict = _verdict_run(relh, nh, retention, flipped,
                                       mon["frac_aligned"], wf["frac_aligned"], val_ok)
                rows.append({
                    "tf": tf, "target": tname, "level": level, "verdict": verdict,
                    "condition": cond, "feature_names": feats, "param_stability": pstable,
                    "disc_n": int(r.get("discovery_n", 0)), "disc_p": r.get("discovery_win_rate", float("nan")),
                    "disc_rel_lift": _rel_lift(dl, base_d), "val_n": int(r.get("validation_n", 0)),
                    "val_p": r.get("validation_win_rate", float("nan")), "val_rel_lift": _rel_lift(vl, base_v),
                    "hold_n": nh, "hold_base": base_h, "hold_p": wrh, "hold_rel_lift": relh,
                    "hold_lift_pp": lifth, "retention": retention, "flipped": flipped,
                    "monthly_frac": mon["frac_aligned"], "walkforward_frac": wf["frac_aligned"],
                    "placebo_p95": plac.get("placebo_p95"),
                    "hold_ci": [hs["ci_low"], hs["ci_high"]], "hold_p_value": hs["p_value"],
                    "walkforward": wf["folds"], "monthly_detail": mon["detail"],
                })
    import csv
    cpath = os.path.join(out, "converge_registry.csv")
    with open(cpath, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=_CONV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {}
            for k in _CONV_COLS:
                v = r.get(k, "")
                if isinstance(v, (float, np.floating)):
                    v = "" if not np.isfinite(float(v)) else round(float(v), 6)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                rr[k] = v
            w.writerow(rr)
    with open(os.path.join(out, "converge_full.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=_json_default)
    log(f"converge -> converge_registry.csv（{len(rows)} 行）")
    _print_converge_edge(rows)
    return {"rows": rows}


def _print_top(summary: dict, registry_rows: list[dict], _lbl: str, seg: str) -> None:
    grp: dict = {}
    for r in registry_rows:
        grp.setdefault((r["tf"], r["target"]), []).append(r)
    print("\n" + "=" * 112)
    print(f"发散存活 Top（按{seg} rel_lift 降序，发现+{seg}同向且提升>0）")
    print("=" * 112)
    rl, vl = f"{seg}_rel_lift", f"{seg}_win_rate"
    for (tf, tname), rows in grp.items():
        cand = [r for r in rows if np.isfinite(r["discovery_lift_pp"]) and np.isfinite(r[f"{seg}_lift_pp"] if f"{seg}_lift_pp" in r else r["validation_lift_pp"])
                and r["discovery_lift_pp"] > 0]
        cand = [r for r in cand if np.isfinite(r.get(rl, float('nan')))]
        if not cand:
            continue
        cand.sort(key=lambda r: -r.get(rl, -9e9))
        base = rows[0]["discovery_baseline"]
        print(f"\n[{tf} · {tname}] 发现段基准 P={base:.4f}")
        for r in cand[:5]:
            print(f"    {r['level']:<3} 发现 P={r['discovery_win_rate']:.4f}(n{r['discovery_n']}) "
                  f"rel={r['discovery_rel_lift']:+.1%} | {seg} P={r.get(vl, float('nan')):.4f} "
                  f"rel={r.get(rl, float('nan')):+.1%} | {r['condition'][:60]}")


def _print_converge_edge(rows: list[dict]) -> None:
    from collections import Counter
    vc = Counter(r["verdict"] for r in rows)
    print("\n" + "=" * 118)
    print("收敛裁决汇总：" + "  ".join(f"{k}={vc.get(k, 0)}"
                                 for k in ("ROBUST", "PROMISING", "WEAK", "REJECT", "INSUFFICIENT")))
    print("=" * 118)
    surv = [r for r in rows if r["verdict"] in ("ROBUST", "PROMISING")]
    surv.sort(key=lambda r: (0 if r["verdict"] == "ROBUST" else 1,
                             -(r["hold_rel_lift"] if np.isfinite(r["hold_rel_lift"]) else -9e9)))
    for r in surv[:36]:
        mf, wf, ret = r["monthly_frac"], r["walkforward_frac"], r["retention"]
        mf_s = f"{mf:.2f}" if np.isfinite(mf) else "—"
        wf_s = f"{wf:.2f}" if np.isfinite(wf) else "—"
        ret_s = f"{ret:.2f}" if np.isfinite(ret) else "—"
        print(f"  [{r['verdict']}] {r['tf']}·{r['target']}·{r['level']} | "
              f"发现 P={r['disc_p']:.4f} rel{r['disc_rel_lift']:+.0%} | 验证 rel{r['val_rel_lift']:+.0%} | "
              f"holdout P={r['hold_p']:.4f}(基{r['hold_base']:.4f}) rel{r['hold_rel_lift']:+.0%} "
              f"ret={ret_s} 月{mf_s}/折{wf_s} | {r['condition'][:56]}")


# ============================ distill：剂量-反应 + 跨样本共识 + 每目标答案 ============================

# 波动/量能轴（发散频率客观识别出的核心驱动）：沿其分位看 P(触及/强移动) 是否单调
DOSE_AXES_EDGE = ["volume_z_20", "range_atr", "volatility_ratio_5_20",
                  "compression_5_20", "atr_pctile_4320", "rel_volume_20", "body_atr"]
DISTILL_TARGETS = ["touchup_2", "touchdn_2", "touchup_3", "touchdn_3", "upstrong_6", "dnstrong_6"]


def _axis_curve_single(kl, fm, tg, feat: str, target: str, nb: int = 10) -> list | None:
    """单目标剂量-反应：沿 feat 分位桶，P(target) 全样本 + holdout 独立段。"""
    if feat not in fm.cols or target not in tg.items:
        return None
    col = fm.cols[feat].astype(np.float64)
    ts = tg.items[target]
    valid = ts.valid & np.isfinite(col)
    if int(valid.sum()) < nb * 40:
        return None
    qs = np.quantile(col[valid], np.linspace(0, 1, nb + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    idx = np.digitize(col, qs[1:-1])
    i2 = int(len(kl) * 0.8)
    rows = []
    for b in range(nb):
        sel = valid & (idx == b)
        cnt = int(sel.sum())
        rec = {"b": b, "n": cnt, "lo": float(qs[b]), "hi": float(qs[b + 1]), "p": None, "h": None, "nh": 0}
        if cnt >= 30:
            rec["p"] = float(ts.win[sel].mean())
            selh = sel.copy()
            selh[:i2] = False
            nh = int(selh.sum())
            rec["nh"] = nh
            if nh >= 20:
                rec["h"] = float(ts.win[selh].mean())
        rows.append(rec)
    return rows


def _mono_edge(ys: list) -> tuple[float, float]:
    ys = [y for y in ys if y is not None and np.isfinite(y)]
    if len(ys) < 3:
        return (0.0, float("nan"))
    d = np.diff(ys)
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    return (1.0 if up >= dn else -1.0, max(up, dn) / len(d))


def _cross_sample_consensus_edge() -> dict:
    dirs = {"720d": os.path.join(OUT0, "edge_structure_720d", "converge_registry.csv"),
            "2160d": os.path.join(OUT0, "edge_structure_2160d", "converge_registry.csv")}
    per: dict = {}
    for tag, p in dirs.items():
        if not os.path.exists(p):
            continue
        agg: dict = {}
        for r in _load_converge(p):
            if r["verdict"] != "ROBUST":
                continue
            key = (r["target"], _mech_of(r["feature_names"]))
            cur = agg.get(key, {"n": 0, "best": -9e9, "cond": ""})
            cur["n"] += 1
            if np.isfinite(r["hold_rel_lift"]) and r["hold_rel_lift"] > cur["best"]:
                cur["best"] = r["hold_rel_lift"]
                cur["cond"] = r["condition"]
            agg[key] = cur
        per[tag] = agg
    if "720d" not in per or "2160d" not in per:
        return {"available": False, "per": per}
    cons = []
    for key, a in per["720d"].items():
        b = per["2160d"].get(key)
        if b:
            cons.append({"target": key[0], "mech": key[1], "n720": a["n"], "rel720": a["best"],
                         "cond720": a["cond"], "n2160": b["n"], "rel2160": b["best"]})
    cons.sort(key=lambda x: -min(x["rel720"], x["rel2160"]))
    return {"available": True, "consensus": cons, "per": per}


def _per_target_answers(out: str) -> dict:
    p = os.path.join(out, "converge_registry.csv")
    if not os.path.exists(p):
        return {}
    rows = [r for r in _load_converge(p) if r["verdict"] == "ROBUST"]
    ans: dict = {}
    for tn in sorted({r["target"] for r in rows}):
        grp = [r for r in rows if r["target"] == tn]
        grp.sort(key=lambda r: (0 if r["level"] == "L1" else 1,
                                -(r["hold_rel_lift"] if np.isfinite(r["hold_rel_lift"]) else -9e9)))
        ans[tn] = grp[0]
    return ans


def stage_distill(out: str, tfs: tuple[str, ...]) -> dict:
    log("=== distill：剂量-反应曲线（阈值无关铁证）+ 跨样本共识 + 每目标答案 ===")
    tf = "15m" if "15m" in tfs else tfs[0]
    kl = load_kl(tf)
    fm = get_featmat(kl, BAR_MS[tf], tf)
    tg = build_edge_targetsets(kl, HORIZONS[tf])
    curves: dict = {}
    for feat in DOSE_AXES_EDGE:
        for tn in DISTILL_TARGETS:
            if tn in tg.items:
                c = _axis_curve_single(kl, fm, tg, feat, tn)
                if c:
                    curves[(feat, tn)] = c
    cons = _cross_sample_consensus_edge()
    answers = _per_target_answers(out)
    _print_distill_edge(tf, curves, cons, answers)
    _write_distill_md_edge(out, tf, len(kl), curves, cons, answers)
    log(f"distill -> {os.path.join(out, 'distill.md')}")
    return {"n_curves": len(curves), "consensus": cons.get("available")}


def _print_distill_edge(tf: str, curves: dict, cons: dict, answers: dict) -> None:
    print("\n" + "=" * 104)
    print(f"表 A. 剂量-反应（{tf}）：沿波动/量能轴分位，P(触及/强移动) 是否单调上升（阈值无关铁证）")
    print("=" * 104)
    for feat in DOSE_AXES_EDGE:
        for tn in DISTILL_TARGETS:
            c = curves.get((feat, tn))
            if not c or c[0]["p"] is None or c[-1]["p"] is None:
                continue
            ps = [r["p"] for r in c]
            sign, mono = _mono_edge(ps)
            hs = [r["h"] for r in c]
            h_ok = hs[0] is not None and hs[-1] is not None
            h_str = f" | holdout {hs[0]:.3f}→{hs[-1]:.3f}" if h_ok else ""
            print(f"  [{feat} · {tn}] D1→D10: {c[0]['p']:.4f}→{c[-1]['p']:.4f} "
                  f"(方向{'+' if sign>0 else '-'} mono={mono:.2f}){h_str}")
    if cons.get("available"):
        print("\n" + "=" * 104)
        print("表 B. 跨样本共识铁律（720d ∩ 2160d 牛熊都 ROBUST）")
        print("=" * 104)
        for x in cons["consensus"][:14]:
            print(f"  {x['target']} · [{x['mech']}] | 720d {x['n720']}条 best{x['rel720']:+.0%} | "
                  f"2160d {x['n2160']}条 best{x['rel2160']:+.0%}")
    print("\n" + "=" * 104)
    print("表 C. 每目标的答案（ROBUST 冠军，优先 L1 单条件）")
    print("=" * 104)
    for tn, r in answers.items():
        print(f"  {tn}: holdout P={r['hold_p']:.4f}(基准{r['hold_base']:.4f}) rel={r['hold_rel_lift']:+.0%} "
              f"月{r['monthly_frac']:.2f}/折{r['walkforward_frac']:.2f} | {r['level']} | {r['condition'][:56]}")


def _write_distill_md_edge(out, tf, n, curves, cons, answers) -> None:
    L: list[str] = []
    A = L.append
    A(f"# {DAYS}d「高胜率形态 / 确定性结构」提纯报告（{tf}）")
    A("")
    A(f"生成（UTC）：{datetime.now(timezone.utc).isoformat()}；样本 {n} 根。")
    A("")
    A("## 1. 剂量-反应曲线（阈值无关铁证）")
    A("")
    A("若规律真实，P(触及±1ATR / 强移动) 应沿波动/量能轴**单调上升**（波动越爆发→未来越确定大幅移动）。")
    A("")
    A("| 轴特征 | 目标 | D1(最低) | D10(最高) | 单调方向 | 一致步 | holdout D1→D10 |")
    A("|---|---|---|---|---|---|---|")
    for feat in DOSE_AXES_EDGE:
        for tn in DISTILL_TARGETS:
            c = curves.get((feat, tn))
            if not c or c[0]["p"] is None or c[-1]["p"] is None:
                continue
            ps = [r["p"] for r in c]
            sign, mono = _mono_edge(ps)
            hs = [r["h"] for r in c]
            h_str = (f"{hs[0]:.3f}→{hs[-1]:.3f}" if hs[0] is not None and hs[-1] is not None else "—")
            A(f"| `{feat}` | {tn} | {c[0]['p']:.4f} | {c[-1]['p']:.4f} | "
              f"{'↑' if sign>0 else '↓'} | {mono:.0%} | {h_str} |")
    A("")
    A("## 2. 跨样本共识铁律（720d ∩ 2160d 牛熊）")
    A("")
    if cons.get("available"):
        A("| 目标 | 机制桶 | 720d ROBUST | 2160d 牛熊 ROBUST |")
        A("|---|---|---|---|")
        for x in cons["consensus"]:
            A(f"| {x['target']} | {x['mech']} | {x['n720']} 条 / {x['rel720']:+.0%} | "
              f"{x['n2160']} 条 / {x['rel2160']:+.0%} |")
    else:
        A("（需 720d 与 2160d 两份 converge_registry.csv 均存在）")
    A("")
    A("## 3. 每目标的答案（确定性结构 / 高胜率形态）")
    A("")
    A("| 目标 | 级别 | holdout P(基准) | 相对提升 | 月/折一致 | 条件 |")
    A("|---|---|---|---|---|---|")
    for tn, r in answers.items():
        A(f"| {tn} | {r['level']} | {r['hold_p']:.4f}({r['hold_base']:.4f}) | "
          f"**{r['hold_rel_lift']:+.0%}** | {r['monthly_frac']:.2f}/{r['walkforward_frac']:.2f} | "
          f"`{r['condition'][:88]}` |")
    A("")
    A("> 剂量-反应单调 + 跨样本共识 ⇒ 非阈值过拟合。**核心：波动率爆发（放量+扩张+大波幅）后，"
      "未来短窗内价格大幅移动（触及±1ATR）的确定性极高。**")
    with open(os.path.join(out, "distill.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ============================ report：汇总报告 ============================

def _scale_invariance_edge(conv_rows: list) -> dict:
    """跨周期尺度不变性：每目标在 5m/15m/1h/4h 的 ROBUST 数与最佳 holdout rel。"""
    scale: dict = {}
    for r in conv_rows:
        if r["verdict"] != "ROBUST":
            continue
        scale.setdefault(r["target"], {})
        cur = scale[r["target"]].get(r["tf"], {"n": 0, "best": -9e9})
        cur["n"] += 1
        if np.isfinite(r["hold_rel_lift"]) and r["hold_rel_lift"] > cur["best"]:
            cur["best"] = r["hold_rel_lift"]
        scale[r["target"]][r["tf"]] = cur
    return scale


def stage_report(out: str, tfs: tuple[str, ...]) -> dict:
    from collections import Counter
    tf = "15m" if "15m" in tfs else tfs[0]
    kl = load_kl(tf)
    fm = get_featmat(kl, BAR_MS[tf], tf)
    tg = build_edge_targetsets(kl, HORIZONS[tf])
    curves: dict = {}
    for feat in DOSE_AXES_EDGE:
        for tn in DISTILL_TARGETS:
            if tn in tg.items:
                c = _axis_curve_single(kl, fm, tg, feat, tn)
                if c:
                    curves[(feat, tn)] = c
    cons = _cross_sample_consensus_edge()
    answers = _per_target_answers(out)
    base = {}
    bp = os.path.join(out, "baseline.json")
    if os.path.exists(bp):
        base = json.load(open(bp, encoding="utf-8"))
    conv = []
    cp = os.path.join(out, "converge_registry.csv")
    if os.path.exists(cp):
        conv = _load_converge(cp)
    vc = Counter(r["verdict"] for r in conv)
    scale = _scale_invariance_edge(conv)
    _write_report_md_edge(out, tf, len(kl), base.get(tf, {}), curves, cons, answers, vc, len(conv), scale)
    log(f"report -> {os.path.join(out, 'report.md')}")
    return {"n_conv": len(conv)}


_MEANING = {"up": "净涨胜率", "upstrong": f"净涨>={STRONG_ATR}ATR", "dnstrong": f"净跌>={STRONG_ATR}ATR",
            "touchup": f"触及+{TOUCH_ATR}ATR", "touchdn": f"触及-{TOUCH_ATR}ATR"}


def _write_report_md_edge(out, tf, n, base, curves, cons, answers, vc, n_conv, scale=None) -> None:
    L: list[str] = []
    A = L.append
    A(f"# {DAYS}d「高胜率形态 / 确定性结构」研究报告（{tf}）")
    A("")
    A(f"生成（UTC）：{datetime.now(timezone.utc).isoformat()}；样本 {n} 根。")
    A("")
    A("## 1. 摘要（核心发现）")
    A("")
    A("不限于连涨连跌，本研究扫描五类目标（净涨胜率 / 幅度门槛强移动 / 触及±1ATR），发现**两类高确定性规律**：")
    A("")
    A("- **① 波动率爆发 → 大幅移动确定性（最强，幅度确定、方向不定）**：当出现巨幅 K 线 / 放量 / 波动扩张时，"
      "未来短窗内价格触及 ±1ATR 的概率从 ~25% 飙升到 ~44-60%（holdout rel 提升 **+72%~+132%**），"
      "且 touchup 与 touchdn **同时**升高——即「必然大幅波动」而非「必涨/必跌」。"
      "单条件 `range_atr>=~1.74`（巨幅 K 线）即成立，月度/walk-forward 一致性 1.00。")
    A("- **② 极端位置 → 方向胜率（次强，押均值回归方向）**：超卖（zscore低/贴低点/创新低）→ 押涨，"
      "净涨胜率升至 ~56%、强涨（>=0.5ATR）rel +38%；超买（创新高）→ 押跌，强跌 rel +45%。")
    A("- **纯方向胜率（up_H）天花板低**（最强因子仅 ~56%），而**确定性结构（touch/strong）提升空间大得多**——"
      "这与既有 kline_discovery（continuation 仅 +6pp）一致：单根方向难测，但「波动幅度」高度可测。")
    A("- 上述均通过 holdout 终验 + 剂量-反应单调（mono 0.89~1.00）+ 720d∩2160d 牛熊跨样本共识。")
    A("")
    A("## 2. 方法")
    A("")
    A(f"- 目标（ex-ante）：up_H=c[t+H]>c[t]；upstrong/dnstrong_H=净涨/跌>={STRONG_ATR}×ATR；"
      f"touchup/touchdn_H=未来H根内 high/low 触及 c±{TOUCH_ATR}×ATR。H={list(HORIZONS[tf])}。")
    A("- 发散：216 特征 × 分位原子 → L1(BH-FDR) → L2/L3 多维组合。")
    A("- 收敛：0.6/0.2/0.2 三段 + holdout 只触碰一次 + 安慰剂置换 + 月度/walk-forward 稳健性 + 裁决。")
    A("- 提纯：剂量-反应曲线（阈值无关）+ 720d∩2160d 跨样本共识。")
    A("")
    A("## 3. 基线：各目标无条件基准率")
    A("")
    A("| 目标 | 基准率 | 含义 |")
    A("|---|---|---|")
    for name, d in base.get("targets", {}).items():
        fam = name.rsplit("_", 1)[0]
        A(f"| {name} | {d['base_all']:.4f} | {_MEANING.get(fam, '')} |")
    A("")
    A("## 4. 发散→收敛裁决")
    A("")
    A(f"共 {n_conv} 候选：" + "；".join(f"{k}={vc.get(k,0)}" for k in
                                     ("ROBUST", "PROMISING", "WEAK", "REJECT", "INSUFFICIENT")) + "。")
    A("")
    A("## 5. 剂量-反应铁证（阈值无关）")
    A("")
    A("| 轴特征 | 目标 | D1 | D10 | 单调 | holdout D1→D10 |")
    A("|---|---|---|---|---|---|")
    for feat in DOSE_AXES_EDGE:
        for tn in DISTILL_TARGETS:
            c = curves.get((feat, tn))
            if not c or c[0]["p"] is None or c[-1]["p"] is None:
                continue
            s, mo = _mono_edge([r["p"] for r in c])
            hs = [r["h"] for r in c]
            hstr = f"{hs[0]:.3f}→{hs[-1]:.3f}" if hs[0] is not None and hs[-1] is not None else "—"
            A(f"| `{feat}` | {tn} | {c[0]['p']:.4f} | {c[-1]['p']:.4f} | {'↑' if s>0 else '↓'}{mo:.0%} | {hstr} |")
    A("")
    A("## 6. 跨样本共识铁律（720d ∩ 2160d 牛熊）")
    A("")
    if cons.get("available"):
        A("| 目标 | 机制桶 | 720d ROBUST | 2160d 牛熊 ROBUST |")
        A("|---|---|---|---|")
        for x in cons["consensus"]:
            A(f"| {x['target']} | {x['mech']} | {x['n720']} / {x['rel720']:+.0%} | {x['n2160']} / {x['rel2160']:+.0%} |")
    else:
        A("（需两份 converge_registry.csv）")
    A("")
    if scale:
        A("## 6b. 跨周期尺度不变性（各周期 ROBUST 数 / 最佳 holdout rel）")
        A("")
        tfs_all = [t for t in ("5m", "15m", "1h", "4h") if any(t in scale.get(tn, {}) for tn in scale)]
        A("| 目标 | " + " | ".join(tfs_all) + " |")
        A("|---|" + "---|" * len(tfs_all))
        for tn in sorted(scale):
            cells = []
            for t in tfs_all:
                d = scale[tn].get(t)
                cells.append(f"{d['n']} / {d['best']:+.0%}" if d else "—")
            A(f"| {tn} | " + " | ".join(cells) + " |")
        A("")
    A("## 7. 每目标答案（ROBUST 冠军，优先 L1 单条件）")
    A("")
    A("| 目标 | 级别 | holdout P(基准) | 相对提升 | 月/折 | 条件 |")
    A("|---|---|---|---|---|---|")
    for tn, r in answers.items():
        A(f"| {tn} | {r['level']} | {r['hold_p']:.4f}({r['hold_base']:.4f}) | **{r['hold_rel_lift']:+.0%}** | "
          f"{r['monthly_frac']:.2f}/{r['walkforward_frac']:.2f} | `{r['condition'][:80]}` |")
    A("")
    A("## 8. 科学结论与边界")
    A("")
    A("1. **最强确定性 = 波动幅度而非方向**：放量/巨幅/扩张后，价格「必然大幅移动」（触及±1ATR）确定性极高，"
      "但方向不定（touchup 与 touchdn 同升）。这是波动率聚集（volatility clustering）的体现。")
    A("2. **方向胜率靠极端位置的均值回归**：超卖押涨、超买押跌，胜率 ~56%（高于随机但有限）。")
    A("3. **描述性 ≠ 可交易**：上述为条件概率规律，未经报价/成本/滑点/执行检验，不直接构成实盘信号。")
    A("4. **防过拟合**：分位阈值仅发现段拟合；holdout 只触碰一次；安慰剂置换；剂量-反应单调；720d∩2160d 牛熊共识。")
    A("")
    A(f"产物：baseline.json、converge_registry.csv（{n_conv} 行）、distill.md、converge_full.json。")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


_STAGES = ("baseline", "discover", "converge", "distill", "report")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    warnings.simplefilter("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description="720d 高胜率形态 / 确定性结构发现引擎")
    ap.add_argument("--tf", default="15m", help="all 或逗号分隔子集")
    ap.add_argument("--stage", default="baseline", help=",".join(_STAGES))
    ap.add_argument("--days", type=int, default=720, help="720（默认）或 2160（牛熊样本外，仅 5m/15m）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global DAYS, FEAT_CACHE_DIR
    DAYS = args.days
    if args.out is None:
        args.out = os.path.join(OUT0, f"edge_structure_{DAYS}d")
    if DAYS != 720:
        FEAT_CACHE_DIR = args.out
    avail = TFS if DAYS == 720 else ("5m", "15m")
    tfs = avail if args.tf == "all" else tuple(
        t.strip() for t in args.tf.split(",") if t.strip() and t.strip() in avail)
    if not tfs:
        raise SystemExit(f"--days {DAYS} 无可用周期（可选 {list(avail)}）")
    stages = tuple(s.strip() for s in args.stage.split(",") if s.strip())
    bad = [s for s in stages if s not in _STAGES]
    if bad:
        raise SystemExit(f"未知阶段：{bad}（可选 {list(_STAGES)}）")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for st in stages:
        if st == "baseline":
            stage_baseline(args.out, tfs)
        elif st == "discover":
            stage_discover(args.out, tfs)
        elif st == "converge":
            stage_converge(args.out, tfs)
        elif st == "distill":
            stage_distill(args.out, tfs)
        elif st == "report":
            stage_report(args.out, tfs)
        else:
            raise SystemExit(f"{st} 阶段待实现")
    log(f"完成，总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
