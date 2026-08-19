#!/usr/bin/env python3
"""纯情绪曲线维度扫描：完全抛开 K 线，只用 p(t) 与 {e_k} 找规律（L1→对称性→FDR→L2→OOS→经济账）。

纯度声明（对齐已入库规范）：
  - 主线特征只读 curve_up_pct（当前窗完整曲线 p(t)）与历史窗 end 序列 {e_k}；
  - out/streak 属预测市场输出（非 K 线），按纯度分级仅作「对照家族 / 调节变量」，不参与主信号；
  - actual_return 及其派生（ret_ratio/big_ret 类）一律排除；
  - mis_*（情绪-结算连续错位）用到 out，标注为对照家族，FDR 与主线分开校正。

维度族（预注册，每个挂机制假设，阈值冻结不滑）：
  E  路径效率 eff=|delta|/Σ|dp|   单边有序推进 vs 拉锯分歧（反应不足→延续）
  T  末段动能 d_tail90             尾盘确认（A1 近亲，决策可用）
  H  早冲晚泄/早泄晚冲             首段×末段动能衰竭
  W  停留时间 hi/lo_share          共识驻留深度（自我实现 vs 拥挤）
  C  首穿 50 时刻（早/晚）         共识翻转被检验的程度
  S  情绪序列连升/连降（estreak）  情绪动量衰竭（纯情绪版 streak）
  M  连续错位 mis_hi2/lo2（对照）  高估/低估存量修正
  R  out 连击（对照）              验证主线是否优于价格动量搬运

纪律：70/30 单一切分；IS 上 L1 入场券 n≥150 且 |dev|≥1.5pp 且 p<0.05；
      主线/对照分家族 BH-FDR(q=0.1)；镜像对强制对称性检验（只有一半显著=可疑）；
      OOS 只验不改（n≥40 且方向一致）；经济账费 2%+溢 0.01，EV bootstrap CI 下界>0。

用法：
    python scripts/local_curve_pure_dim.py [--from-file sentiment_windows.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

from binance_predict.services.verification import bh_fdr, wilson_bounds  # noqa: E402
from local_sentiment_curve_discovery import (  # noqa: E402
    HOLDOUT_RATIO, build_records, curve_arrays, entry_quote, eval_mask, ev_eval,
)

MIN_N = 150            # L1 入场券最小样本
MIN_DEV = 0.015        # L1 入场券最小偏离（pp）
P_SIG = 0.05           # L1 名义显著性
OOS_MIN_N = 40         # OOS 终验最小样本
L2_FLOOR = 60          # L2 组合最小样本
TAIL_T = 210.0         # 末段动能起点（最后 90s）
HEAD_T = 150.0         # 首段终点（决策点对齐）
PHI_WIN, PHI_MIN = 24, 16   # φ regime：过去 24 窗 lag-1 自相关，最少 16 对
PHI_BAND = 0.15        # φ regime 显著带宽
OUT_PATH = "output/curve_pure_dim.json"


# ============================================================
# 纯情绪特征：窗内 p(t) 几何/时间结构 + 跨窗 {e_k} 序列
# ============================================================

def pure_window_feats(w: dict) -> dict | None:
    """单窗完整曲线 → 路径效率 / 首末段动能 / 停留占比 / 首穿50时刻。"""
    ca = curve_arrays(w, "curve_up_pct")
    if ca is None:
        return None
    v, t = ca
    if len(v) < 4:
        return None
    d = np.diff(v)
    path = float(np.abs(d).sum())
    delta = float(v[-1] - v[0])
    vh = v[t <= HEAD_T]
    vt = v[t <= TAIL_T]
    side0 = "up" if v[0] >= 50 else "dn"
    t_cross = float("nan")
    for j in range(1, len(v)):
        if side0 == "dn" and v[j] >= 50:
            t_cross = float(t[j])
            break
        if side0 == "up" and v[j] < 50:
            t_cross = float(t[j])
            break
    return {
        "delta": delta,
        "eff": abs(delta) / path if path >= 3.0 else float("nan"),
        "d_head": float(vh[-1] - v[0]) if len(vh) else float("nan"),
        "d_tail": float(v[-1] - vt[-1]) if len(vt) else float("nan"),
        "hi_share": float((v >= 60).mean()),
        "lo_share": float((v <= 40).mean()),
        "t_cross": t_cross,
        "cross_dir": "up" if side0 == "dn" else "dn",
    }


def build_pure(W: list[dict]) -> list[dict]:
    """build_records（label/上下文）+ 纯特征 + 跨窗 {e_k} 序列特征（estreak / φ / mis）。"""
    recs = build_records(W)
    for r, w in zip(recs, W):
        r["pf"] = pure_window_feats(w)
    ends: list[float | None] = [r["f"]["end"] if r.get("f") else None for r in recs]

    # estreak（含当前窗，要求 300s 连续链）+ Δe
    su = sd = 0
    de: list[float | None] = [None] * len(recs)
    for i, r in enumerate(recs):
        if i > 0 and r["prev_ok"] and ends[i] is not None and ends[i - 1] is not None:
            de[i] = ends[i] - ends[i - 1]
            su = su + 1 if de[i] > 0 else 0
            sd = sd + 1 if de[i] < 0 else 0
        else:
            su = sd = 0
        r["estreak_up"], r["estreak_dn"] = su, sd

    # φ regime：过去 PHI_WIN 窗 Δe 的 lag-1 自相关
    for i in range(len(recs)):
        pairs = [(de[j - 1], de[j]) for j in range(max(1, i - PHI_WIN + 1), i + 1)
                 if de[j] is not None and de[j - 1] is not None]
        if len(pairs) >= PHI_MIN:
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            if x.std() > 0 and y.std() > 0:
                recs[i]["phi"] = float(np.corrcoef(x, y)[0, 1])
                continue
        recs[i]["phi"] = float("nan")

    # mis：连续 2 窗情绪-结算同向错位（用到 out → 对照家族）
    for i, r in enumerate(recs):
        p = recs[i - 1] if i > 0 else None
        r["mis_hi2"] = bool(
            r["prev_ok"] and p is not None and p.get("f")
            and r.get("f") and r["out"] == "DOWN" and p["out"] == "DOWN"
            and r["f"]["end"] >= 55 and p["f"]["end"] >= 55)
        r["mis_lo2"] = bool(
            r["prev_ok"] and p is not None and p.get("f")
            and r.get("f") and r["out"] == "UP" and p["out"] == "UP"
            and r["f"]["end"] <= 45 and p["f"]["end"] <= 45)
    return recs


# ============================================================
# 假设注册表：(name, layer, family, mech, fn, dir)
#   layer: geo/tail/dwell/cross/eseq/ctx  family: main=纯 / xcheck=对照(用到 out)
# ============================================================

def register(recs) -> list[tuple]:
    def g(r, path, default=float("nan")):
        cur = r
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    ok = lambda x: x == x  # noqa: E731  非 NaN
    H: list[tuple] = []

    def h(name, layer, family, mech, fn, direction):
        H.append((name, layer, family, mech,
                  lambda R, fn=fn: np.array([bool(fn(r)) for r in R]), direction))

    # ---- E 路径效率（单边有序推进 → 反应不足延续）----
    h("E1_eff60_up", "geo", "main", "|Δ|≥6 且 eff≥0.6 且 Δ>0：单边有序上推 → 延续",
      lambda r: g(r, "pf.delta") >= 6 and ok(g(r, "pf.eff")) and g(r, "pf.eff") >= 0.6, "UP")
    h("E1_eff60_dn", "geo", "main", "对称：单边有序下推",
      lambda r: g(r, "pf.delta") <= -6 and ok(g(r, "pf.eff")) and g(r, "pf.eff") >= 0.6, "DOWN")
    h("E2_eff75_up", "geo", "main", "强单边（eff≥0.75）",
      lambda r: g(r, "pf.delta") >= 6 and ok(g(r, "pf.eff")) and g(r, "pf.eff") >= 0.75, "UP")
    h("E2_eff75_dn", "geo", "main", "对称：强单边下推",
      lambda r: g(r, "pf.delta") <= -6 and ok(g(r, "pf.eff")) and g(r, "pf.eff") >= 0.75, "DOWN")

    # ---- T 末段动能（尾盘确认，A1 近亲）----
    for t in (5, 10, 15):
        h(f"T_tail{t}_up", "tail", "main", f"末 90s 动能 ≥ +{t}pp → 次窗延续",
          lambda r, t=t: g(r, "pf.d_tail") >= t, "UP")
        h(f"T_tail{t}_dn", "tail", "main", f"末 90s 动能 ≤ -{t}pp",
          lambda r, t=t: g(r, "pf.d_tail") <= -t, "DOWN")

    # ---- H 首末段背离（动能衰竭）----
    h("H_fade_up", "tail", "main", "早冲晚泄（d_head>3 且 d_tail<-3）→ 上冲动能耗竭",
      lambda r: g(r, "pf.d_head") > 3 and g(r, "pf.d_tail") < -3, "DOWN")
    h("H_fade_dn", "tail", "main", "早泄晚冲（d_head<-3 且 d_tail>3）→ 下冲动能耗竭",
      lambda r: g(r, "pf.d_head") < -3 and g(r, "pf.d_tail") > 3, "UP")

    # ---- W 停留时间（共识驻留）----
    h("W_dwell_hi", "dwell", "main", "hi_share≥0.7（驻留多头带）→ 共识自我实现",
      lambda r: g(r, "pf.hi_share") >= 0.7, "UP")
    h("W_dwell_lo", "dwell", "main", "lo_share≥0.7（驻留空头带）",
      lambda r: g(r, "pf.lo_share") >= 0.7, "DOWN")

    # ---- C 首穿 50 时刻（早穿=已被检验 / 晚穿=未检验）----
    h("C_early_x_up", "cross", "main", "自下而上早穿（≤100s）且收尾≥55 → 新共识延续",
      lambda r: g(r, "pf.cross_dir") == "up" and ok(g(r, "pf.t_cross"))
      and g(r, "pf.t_cross") <= 100 and g(r, "f.end") >= 55, "UP")
    h("C_early_x_dn", "cross", "main", "自上而下早穿且收尾≤45",
      lambda r: g(r, "pf.cross_dir") == "dn" and ok(g(r, "pf.t_cross"))
      and g(r, "pf.t_cross") <= 100 and g(r, "f.end") <= 45, "DOWN")
    h("C_late_x_up", "cross", "main", "自下而上晚穿（≥210s）且收尾≥55 → 尾盘急转延续",
      lambda r: g(r, "pf.cross_dir") == "up" and ok(g(r, "pf.t_cross"))
      and g(r, "pf.t_cross") >= 210 and g(r, "f.end") >= 55, "UP")
    h("C_late_x_dn", "cross", "main", "自上而下晚穿且收尾≤45",
      lambda r: g(r, "pf.cross_dir") == "dn" and ok(g(r, "pf.t_cross"))
      and g(r, "pf.t_cross") >= 210 and g(r, "f.end") <= 45, "DOWN")

    # ---- S 情绪序列连升/连降（纯情绪版 streak）----
    h("S_estk2_up", "eseq", "main", "{e_k} 连升≥2 → 情绪动量延续",
      lambda r: r.get("estreak_up", 0) >= 2, "UP")
    h("S_estk2_dn", "eseq", "main", "{e_k} 连降≥2",
      lambda r: r.get("estreak_dn", 0) >= 2, "DOWN")
    h("S_estk3_up", "eseq", "main", "{e_k} 连升≥3（衰竭 vs 延续）",
      lambda r: r.get("estreak_up", 0) >= 3, "UP")
    h("S_estk3_dn", "eseq", "main", "{e_k} 连降≥3",
      lambda r: r.get("estreak_dn", 0) >= 3, "DOWN")

    # ---- M/R 对照家族（用到 out，非纯；FDR 分开校正）----
    h("M_mis_hi2", "ctx", "xcheck", "连续 2 窗 end≥55 但结算 DOWN → 高估存量修正",
      lambda r: r.get("mis_hi2", False), "UP")
    h("M_mis_lo2", "ctx", "xcheck", "连续 2 窗 end≤45 但结算 UP → 低估存量修正",
      lambda r: r.get("mis_lo2", False), "DOWN")
    h("R_stk3_up", "ctx", "xcheck", "out 3 连阳 → 对照：价格动量搬运上限",
      lambda r: r.get("streak_up", 0) >= 3, "UP")
    h("R_stk3_dn", "ctx", "xcheck", "out 3 连阴 → 对照",
      lambda r: r.get("streak_dn", 0) >= 3, "DOWN")
    return H


# 镜像对（对称性检验）：押 DOWN 侧的等效 UP 偏离 = -dev
SYMMETRY_PAIRS = [
    ("E1_eff60_up", "E1_eff60_dn"), ("E2_eff75_up", "E2_eff75_dn"),
    ("T_tail5_up", "T_tail5_dn"), ("T_tail10_up", "T_tail10_dn"),
    ("T_tail15_up", "T_tail15_dn"), ("H_fade_up", "H_fade_dn"),
    ("W_dwell_hi", "W_dwell_lo"), ("C_early_x_up", "C_early_x_dn"),
    ("C_late_x_up", "C_late_x_dn"), ("S_estk2_up", "S_estk2_dn"),
    ("S_estk3_up", "S_estk3_dn"), ("M_mis_hi2", "M_mis_lo2"),
    ("R_stk3_up", "R_stk3_dn"),
]


# ============================================================
# 调节因子（预注册，L2 用）：φ regime / 时段 / 前窗 / grid / streak(对照)
# ============================================================

def moderators(recs) -> dict[str, callable]:
    def m(fn):
        return lambda R: np.array([bool(fn(r)) for r in R])
    ok = lambda x: x == x  # noqa: E731
    return {
        "phi_pos(情绪动量期)": m(lambda r: ok(r.get("phi", float("nan"))) and r["phi"] > PHI_BAND),
        "phi_neg(回归期)": m(lambda r: ok(r.get("phi", float("nan"))) and r["phi"] < -PHI_BAND),
        "grid_first": m(lambda r: r["grid"] == 0),
        "grid_last": m(lambda r: r["grid"] == 2),
        "asia": m(lambda r: r["hour"] <= 7),
        "us": m(lambda r: r["hour"] >= 16),
        "prev_end_hi": m(lambda r: ok(r.get("prev_end_s", float("nan"))) and r["prev_end_s"] >= 60),
        "prev_end_lo": m(lambda r: ok(r.get("prev_end_s", float("nan"))) and r["prev_end_s"] <= 40),
        "streak_up2*": m(lambda r: r.get("streak_up", 0) >= 2),
        "streak_dn2*": m(lambda r: r.get("streak_dn", 0) >= 2),
    }


# ============================================================
# 主流程
# ============================================================

def run(W: list[dict], out_path: str) -> dict:
    W = sorted(W, key=lambda w: int(w["start_time"]))
    split_ts = int(W[int(len(W) * (1 - HOLDOUT_RATIO))]["start_time"])
    recs = build_pure(W)
    arr = np.array(recs, dtype=object)

    pool = np.array([r["has_next"] for r in recs])
    tgt_up = np.array([1.0 if (r["next_out"] or "") == "UP" else
                       (0.0 if (r["next_out"] or "") == "DOWN" else np.nan) for r in recs])
    is_oos = np.array([r["start"] >= split_ts for r in recs])
    base_is = float(np.nanmean(tgt_up[~is_oos & pool]))
    base_oos = float(np.nanmean(tgt_up[is_oos & pool]))
    print(f"[数据] {len(W)} 窗 | 切分 {time.strftime('%m-%d %H:%M', time.gmtime(split_ts/1000))} UTC | "
          f"IS={int((~is_oos & pool).sum())} OOS={int((is_oos & pool).sum())} | 基准 UP: IS {base_is:.1%} / OOS {base_oos:.1%}")

    H = register(recs)
    fns = {n: fn for n, _, _, _, fn, _ in H}
    layer_of = {n: ly for n, ly, _, _, _, _ in H}
    fam_of = {n: fm for n, _, fm, _, _, _ in H}
    mech_of = {n: mc for n, _, _, mc, _, _ in H}
    dir_of = {n: d for n, _, _, _, _, d in H}

    # ---------- L1（分家族 FDR + 入场券）----------
    l1: list[dict] = []
    for name, layer, family, mech, fn, td in H:
        r_is = eval_mask(fn(arr), tgt_up, base_is, pool & ~is_oos)
        if r_is is None:
            continue
        r_is.update({"name": name, "layer": layer, "family": family, "mech": mech, "dir": td})
        l1.append(r_is)
    # 分家族 BH-FDR(q=0.1) + 入场券（主线/对照分开校正，防对照稀释主线）
    by_fam: dict[str, list] = {"main": [], "xcheck": []}
    for family in ("main", "xcheck"):
        rows = [x for x in l1 if x["family"] == family]
        passed = bh_fdr([x["pval"] for x in rows], q=0.1)
        for x, ok_ in zip(rows, passed):
            x["fdr"] = ok_
            x["ticket"] = x["n"] >= MIN_N and abs(x["dev"]) >= MIN_DEV and x["pval"] < P_SIG
            by_fam[family].append(x)

    print(f"\n===== L1 单因子（IS，{len(l1)} 检验 | 主线 {len(by_fam['main'])} 对照 {len(by_fam['xcheck'])}）=====")
    for r in sorted(l1, key=lambda x: -abs(x["dev"])):
        star = "*" if r.get("ticket") else " "
        fdr = "F" if r.get("fdr") else " "
        print(f" {star}{fdr} [{r['family'][:4]}|{r['layer']:<5}] {r['name']:<14} n={r['n']:>5} "
              f"押{r['dir']:<4}{r['p']:6.1%} (dev {r['dev']:+.1%}pp) "
              f"CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}] p={r['pval']:.3f}")

    # ---------- 对称性检验（镜像对，IS 口径）----------
    l1map = {r["name"]: r for r in l1}
    print("\n===== 对称性检验（a 押 UP dev vs b 押 DOWN 等效 dev = -dev_b）=====")
    sym_rows = []
    for a, b in SYMMETRY_PAIRS:
        ra, rb = l1map.get(a), l1map.get(b)
        if not ra or not rb:
            continue
        dev_b_eq = -rb["dev"]
        same_sign = np.sign(ra["dev"]) == np.sign(dev_b_eq) and ra["dev"] != 0
        sig_a, sig_b = ra["pval"] < P_SIG, rb["pval"] < P_SIG
        verdict = "对称" if same_sign and (sig_a == sig_b) else ("方向同/显著性不齐" if same_sign else "不对称!")
        sym_rows.append({"a": a, "b": b, "dev_a": ra["dev"], "dev_b_eq": dev_b_eq,
                         "sig_a": sig_a, "sig_b": sig_b, "verdict": verdict})
        print(f"  {a:<14} dev {ra['dev']:+.1%} ({'显著' if sig_a else '不显著'})  |  "
              f"{b:<14} 等效 dev {dev_b_eq:+.1%} ({'显著' if sig_b else '不显著'})  → {verdict}")

    # ---------- L2：主线存活者 × 调节 + 跨层主线组合 ----------
    surv = sorted([r for r in by_fam["main"] if r["ticket"]], key=lambda x: -abs(x["dev"]))
    mods = moderators(recs)
    l2: list[dict] = []
    for r in surv[:6]:
        for mn, mfn in mods.items():
            m = fns[r["name"]](arr) & mfn(arr)
            rr = eval_mask(m, tgt_up, base_is, pool & ~is_oos)
            if rr and rr["n"] >= L2_FLOOR:
                rr.update({"name": f"{r['name']} × {mn}", "parents": [r["name"]], "mod": mn,
                           "layer": r["layer"], "family": "main", "dir": r["dir"]})
                l2.append(rr)
    # 跨层组合（同方向才组合，机制可叠加）
    mains = surv[:6]
    for i, ra in enumerate(mains):
        for rb in mains[i + 1:]:
            if layer_of[ra["name"]] == layer_of[rb["name"]] or ra["dir"] != rb["dir"]:
                continue
            m = fns[ra["name"]](arr) & fns[rb["name"]](arr)
            rr = eval_mask(m, tgt_up, base_is, pool & ~is_oos)
            if rr and rr["n"] >= L2_FLOOR:
                rr.update({"name": f"{ra['name']} × {rb['name']}", "parents": [ra["name"], rb["name"]],
                           "mod": None, "layer": layer_of[ra["name"]], "family": "main", "dir": ra["dir"]})
                l2.append(rr)
    l2.sort(key=lambda x: -abs(x["dev"]))
    print(f"\n===== L2 组合（IS，{len(l2)} 组 | floor n≥{L2_FLOOR}）=====")
    for r in l2[:12]:
        print(f"  [{r['layer']:<5}] {r['name']:<44} n={r['n']:>5} 押{r['dir']:<4}"
              f"{r['p']:6.1%} (dev {r['dev']:+.1%}pp) p={r['pval']:.3f}")

    # ---------- OOS 终验（只验不改；主线配额 8 / 对照 2）----------
    print(f"\n===== OOS 终验（后 {HOLDOUT_RATIO:.0%}，n≥{OOS_MIN_N} 且方向一致）=====")
    cand_main = sorted([r for r in surv] + [r for r in l2 if r["n"] >= L2_FLOOR],
                       key=lambda x: -abs(x["dev"]))
    cand_x = sorted([r for r in by_fam["xcheck"] if r["ticket"]], key=lambda x: -abs(x["dev"]))
    finals = []
    for family, cands, quota in (("main", cand_main, 8), ("xcheck", cand_x, 2)):
        picked = 0
        for r in cands:
            if picked >= quota:
                break
            names = r.get("parents") or [r["name"]]
            if r.get("mod"):
                names = names + [r["mod"]]
            m = np.ones(len(recs), dtype=bool)
            ok_mask = True
            for nm in names:
                if nm in fns:
                    m &= fns[nm](arr)
                elif nm in mods:
                    m &= mods[nm](arr)
                else:
                    ok_mask = False
                    break
            if not ok_mask:
                continue
            r_oos = eval_mask(m, tgt_up, base_oos, pool & is_oos)
            if r_oos is None or r_oos["n"] == 0:
                continue
            # 押 DOWN 的等效方向一致：dev 变号
            dev_oos_eq = r_oos["dev"] if r["dir"] == "UP" else -r_oos["dev"]
            dev_is_eq = r["dev"] if r["dir"] == "UP" else -r["dev"]
            oos_ok = r_oos["n"] >= OOS_MIN_N and np.sign(dev_oos_eq) == np.sign(dev_is_eq)
            k_all = r["k"] + r_oos["k"]
            n_all = r["n"] + r_oos["n"]
            lo_all, hi_all = wilson_bounds(k_all, n_all)
            # 经济账（OOS 段逐注真实报价）：次窗入场，押注方向 dir
            ev = _econ(m, pool & is_oos, r["dir"], recs, W)
            finals.append({
                "name": r["name"], "family": family, "layer": r["layer"], "dir": r["dir"],
                "mech": mech_of.get(r["parents"][0] if r.get("parents") else r["name"], ""),
                "is": {"n": r["n"], "p": r["p"], "dev": r["dev"]},
                "oos": {"n": r_oos["n"], "p": r_oos["p"], "dev": r_oos["dev"]},
                "oos_pass": bool(oos_ok),
                "combined": {"n": n_all, "p": k_all / n_all, "ci": [lo_all, hi_all]},
                "ev": ev,
            })
            picked += 1
            flag = "✓" if oos_ok else "✗"
            print(f"  {flag} [{family[:4]}] {r['name']}")
            print(f"      IS n={r['n']} 押{r['dir']} p={r['p']:.1%} ({r['dev']:+.1%}pp) | "
                  f"OOS n={r_oos['n']} p={r_oos['p']:.1%} ({r_oos['dev']:+.1%}pp) | "
                  f"合并 CI[{lo_all:.1%},{hi_all:.1%}]")
            if ev.get("n"):
                print(f"      EV(费2+溢1)={ev.get('ev_2_1', float('nan')):+.4f} "
                      f"CI{ev.get('ev_2_1_ci')} | 实价覆盖 {ev.get('n_real', 0)}/{ev['n']} "
                      f"均价={ev.get('avg_price', 0):.3f}")

    report = {
        "meta": {"windows": len(W), "split_ts": split_ts, "base_is": base_is,
                 "base_oos": base_oos, "families": {"main": "纯 p(t)+{e_k}", "xcheck": "用到 out 的对照"}},
        "l1": l1, "symmetry": sym_rows, "l2": l2[:30], "finals": finals,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[已写入] {out_path}")
    return report


def _econ(mask: np.ndarray, oos_pool: np.ndarray, direction: str, recs, W) -> dict:
    """OOS 段逐注经济账：当前窗命中 → 次窗 +150s 真实 token 报价入场。"""
    idxs, dirs = [], []
    for j, r in enumerate(recs):
        if not (mask[j] and oos_pool[j]):
            continue
        d = direction
        if d not in ("UP", "DOWN"):  # 组合后方向缺失时跳过
            continue
        idxs.append(j + 1)
        dirs.append(d)
    if not idxs:
        return {"n": 0}
    return ev_eval(idxs, dirs, W)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="纯情绪曲线维度扫描（无 K 线特征）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    with open(args.from_file, encoding="utf-8") as f:
        W = json.load(f)
    run(W, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
