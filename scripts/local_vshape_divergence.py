#!/usr/bin/env python3
"""情绪曲线深V背离分类学：5min 窗内结构提取 → 细致分类 → 斜率 → 次窗预测力 → OOS + 经济账。

研究假设（用户见解，本轮操作化）：
  高赔率模式必然出现在「背离」中，且背离一定伴随情绪的深 V 反转——
  右臂加速度极快（谷后 60s 冲击斜率）的深 V 是人群被暴力反转的形态签名。

结构提取（每窗全窗曲线，20 采样 / 300s，pp = up_pct 百分点）：
  - 主导极值：谷 i*=argmin / 峰 j*=argmax，取深度更大者定向
  - 深度 depth = min(左肩,右肩) - 谷     （V）；倒 V 镜像
  - 左臂斜率 s_left：左肩→谷 OLS（pp/min）
  - 右臂斜率 s_right：谷→右肩 OLS（pp/min）
  - 冲击速度 s_fast：谷后前 4 采样（~60s）OLS ——「加速度极快」的操作化
  - 谷位 trough_pos ∈ [0,1]、谷水平 trough_val、收尾 end_val

背离类型学（结算 out vs 曲线）：
  A0 无背离（对照）
  A1 收尾背离：sign(delta) 与结算相反且 |delta|≥10（X3/X4 的幅度推广）
  A2 V-结算背离：V 形修复成功（结算 UP）vs 失败（结算 DOWN）
  A3 极谷背离：谷 ≤20 却收阳（人群 80% 看跌过、结果 UP）

纪律：分位阈值只在发现集冻结；OOS 只验不改；经济账真实报价 + 费2%+溢价0.01。

用法：python scripts/local_vshape_divergence.py [--from-file sentiment_windows.json]
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
    HOLDOUT_RATIO,
    build_records,
    entry_quote,
    ev_eval,
)

MIN_DEPTH = 12.0   # V/倒V 判定的最小深度（pp）；低于此不称为反转结构
TREND_D = 12.0     # 单边趋势判定的最小 |delta|
FLAT_R = 8.0       # 平坦判定的最大 range
FAST_SEC = 60.0    # 冲击速度窗口：谷后前 60s
OOS_MIN_N = 25     # V 结构类样本稀疏，OOS 门槛按类调低并如实报告


# ============================================================
# 结构提取
# ============================================================

def _ols_slope(t: np.ndarray, v: np.ndarray) -> float:
    """OLS 斜率，单位 pp/min。样本<2 或零时间跨度返回 nan。"""
    if len(t) < 2 or float(t[-1] - t[0]) <= 0:
        return float("nan")
    return float(np.polyfit(t, v, 1)[0] * 60.0)


def extract_structure(v: np.ndarray, t: np.ndarray) -> dict | None:
    """全窗曲线 → 结构 dict（V/倒V/双V/单边/平坦/其他 + 两臂斜率族）。"""
    n = len(v)
    if n < 6:
        return None
    i_min, i_max = int(np.argmin(v)), int(np.argmax(v))
    lo, hi = float(v.min()), float(v.max())

    def varm(v_arr, i_ext, side):
        """肩高：极值向外到端点的最远同侧值。side='L'左肩 / 'R'右肩。"""
        seg = v_arr[: i_ext + 1] if side == "L" else v_arr[i_ext:]
        return float(seg.max()) if side == "L" else float(seg.max())

    d_v = min(varm(v, i_min, "L"), varm(v, i_min, "R")) - lo       # V 深度
    d_iv = hi - max(v[: i_max + 1].min(), v[i_max:].min())          # 倒V 深度

    st: dict = {"start": float(v[0]), "end": float(v[-1]),
                "delta": float(v[-1] - v[0]), "range": hi - lo,
                "trough_val": lo, "peak_val": hi, "depth": 0.0,
                "s_left": float("nan"), "s_right": float("nan"),
                "s_fast": float("nan"), "trough_pos": float("nan"),
                "ext_pos": float("nan"), "type": "OTHER"}

    kind = None
    if d_v >= MIN_DEPTH and d_v >= d_iv:
        kind = "V"
    elif d_iv >= MIN_DEPTH and d_iv > d_v:
        kind = "INV_V"
    elif d_v >= MIN_DEPTH and d_iv >= MIN_DEPTH:
        kind = "DOUBLE"

    if kind in ("V", "INV_V"):
        if kind == "V":
            i_ext = i_min
            l_sh = int(np.argmax(v[: i_ext + 1]))
            r_sh = i_ext + int(np.argmax(v[i_ext:]))
            ext_val, depth = lo, d_v
        else:
            i_ext = i_max
            l_sh = int(np.argmin(v[: i_ext + 1]))
            r_sh = i_ext + int(np.argmin(v[i_ext:]))
            ext_val, depth = hi, d_iv
        st.update({
            "type": kind, "depth": float(depth),
            "ext_pos": float(i_ext / (n - 1)),
            "trough_pos": float(i_min / (n - 1)),
            "ext_val": float(ext_val),
            "s_left": _ols_slope(t[l_sh:i_ext + 1], v[l_sh:i_ext + 1]),
            "s_right": _ols_slope(t[i_ext:r_sh + 1], v[i_ext:r_sh + 1]),
        })
        # 冲击速度：极值点后 FAST_SEC 内的 OLS
        m = (t >= t[i_ext]) & (t <= t[i_ext] + FAST_SEC)
        if int(m.sum()) >= 3:
            st["s_fast"] = _ols_slope(t[m], v[m])
        else:
            st["s_fast"] = st["s_right"]
    elif kind == "DOUBLE":
        st["type"] = "DOUBLE"
        st["depth"] = float(max(d_v, d_iv))
    elif abs(st["delta"]) >= TREND_D:
        st["type"] = "TREND_UP" if st["delta"] > 0 else "TREND_DN"
        st["s_right"] = _ols_slope(t, v)
    elif st["range"] <= FLAT_R:
        st["type"] = "FLAT"
    return st


# ============================================================
# 统计工具
# ============================================================

def binom_p(k: int, n: int, p0: float) -> float:
    if n <= 0:
        return 1.0
    z = (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def stat(mask: np.ndarray, y: np.ndarray, base: float, pool: np.ndarray) -> dict | None:
    m = mask & pool & np.isfinite(y)
    n = int(m.sum())
    if n == 0:
        return None
    k = int(y[m].sum())
    p = k / n
    lo, hi = wilson_bounds(k, n)
    return {"n": n, "k": k, "p": p, "ci": (lo, hi), "dev": p - base,
            "pval": binom_p(k, n, base)}


def fmt(r: dict | None, base: float, label: str, extra: str = "") -> str:
    if r is None:
        return f"  {label:<44} n=0"
    return (f"  {label:<44} n={r['n']:>5} {r['p']:6.1%} (dev {r['dev']:+.1%}pp) "
            f"CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}] p={r['pval']:.3f}{extra}")


# ============================================================
# 主流程
# ============================================================

def run(W: list[dict], out_path: str) -> dict:
    W = sorted(W, key=lambda w: int(w["start_time"]))
    split_ts = int(W[int(len(W) * (1 - HOLDOUT_RATIO))]["start_time"])
    recs = build_records(W)
    print(f"[数据] {len(W)} 窗 | 切分 {time.strftime('%m-%d %H:%M', time.gmtime(split_ts/1000))} UTC")

    # 结构提取（全窗曲线；次窗交易发生在本窗结束后，无泄漏）
    structs: list[dict | None] = []
    for w in W:
        pts = sorted(w["curve_up_pct"], key=lambda p: p["t"])
        v = np.array([float(p["v"]) for p in pts])
        t = np.array([(p["t"] - pts[0]["t"]) / 1000.0 for p in pts])
        structs.append(extract_structure(v, t))
    ok = [s for s in structs if s]
    is_oos = np.array([int(w["start_time"]) >= split_ts for w in W])

    # ---- 发现集分布 → 冻结阈值（物理可解释的固定值优先，分位兜底）----
    S = [s for s, o in zip(structs, is_oos) if s and not o]
    vs = [s for s in S if s["type"] == "V"]
    ivs = [s for s in S if s["type"] == "INV_V"]
    dep75 = float(np.quantile([s["depth"] for s in vs + ivs], 0.75))
    fast75 = float(np.quantile([s["s_fast"] for s in vs + ivs
                                if s["s_fast"] == s["s_fast"]], 0.75))
    D_DEEP = max(20.0, round(dep75, 0))       # 深V阈值：≥20pp 且不低于 p75
    F_FAST = max(30.0, round(fast75, 0))      # 快右臂：≥30 pp/min 且不低于 p75
    print(f"[结构分布·发现集] V={len(vs)} 倒V={len(ivs)} "
          f"双V={sum(1 for s in S if s['type']=='DOUBLE')} "
          f"单边={sum(1 for s in S if s['type'].startswith('TREND'))} "
          f"平坦={sum(1 for s in S if s['type']=='FLAT')} 其他={sum(1 for s in S if s['type']=='OTHER')}")
    print(f"[冻结阈值] depth p75={dep75:.1f}pp → 深V≥{D_DEEP:.0f}pp | "
          f"s_fast p75={fast75:.1f}pp/min → 快臂≥{F_FAST:.0f}pp/min")

    # ---- 目标 ----
    y_up = np.array([1.0 if (r["next_out"] or "") == "UP" else
                     (0.0 if (r["next_out"] or "") == "DOWN" else np.nan) for r in recs])
    y_same = np.array([1.0 if r["next_same"] else (0.0 if r["has_next"] and not r["next_same"]
                       else np.nan) for r in recs])
    pool = np.array([r["has_next"] for r in recs])
    base_up = float(np.nanmean(y_up[pool & ~is_oos]))
    base_same = float(np.nanmean(y_same[pool & ~is_oos]))
    base_up_oos = float(np.nanmean(y_up[pool & is_oos]))
    print(f"[基准·发现集] 次窗UP={base_up:.1%} 延续={base_same:.1%} | OOS 次窗UP={base_up_oos:.1%}")

    out = np.array([(r["out"] if r["out"] in ("UP", "DOWN") else None) for r in recs], dtype=object)

    def M(cond) -> np.ndarray:
        """结构掩码：cond(s, rec) 接收结构与同窗记录（含结算/次窗目标）。"""
        return np.array([bool(s) and cond(s, recs[j]) if s else False
                         for j, s in enumerate(structs)])

    # ============ 分类学总表（发现集） ============
    print(f"\n===== 一、结构类型 × 结算关系 分类学总表（发现集）=====")
    rows = []
    types = [("V 深V反转(≥%.0fpp)" % D_DEEP, lambda s: s["type"] == "V" and s["depth"] >= D_DEEP),
             ("V 浅V(%d-%.0fpp)" % (MIN_DEPTH, D_DEEP), lambda s: s["type"] == "V" and s["depth"] < D_DEEP),
             ("倒V 深倒V(≥%.0fpp)" % D_DEEP, lambda s: s["type"] == "INV_V" and s["depth"] >= D_DEEP),
             ("倒V 浅倒V", lambda s: s["type"] == "INV_V" and s["depth"] < D_DEEP),
             ("双V 双重摆动", lambda s: s["type"] == "DOUBLE"),
             ("单边上行", lambda s: s["type"] == "TREND_UP"),
             ("单边下行", lambda s: s["type"] == "TREND_DN"),
             ("平坦", lambda s: s["type"] == "FLAT")]
    # 每类再按「结算与曲线背离/一致」拆分（背离 = 结算方向 != sign(delta)，|delta|≥10 才有意义）
    for name, cond in types:
        m = M(lambda s, r, c=cond: c(s))
        for tag, div_cond in (("全部", None),
                              ("·收尾背离(结算逆delta≥10pp)",
                               lambda s, r: abs(s["delta"]) >= 10
                               and ((s["delta"] > 0) != (r["out"] == "UP"))),
                              ("·收尾一致",
                               lambda s, r: abs(s["delta"]) < 10
                               or ((s["delta"] > 0) == (r["out"] == "UP")))):
            mm = np.array([mm_ and (div_cond is None or div_cond(s, recs[j]))
                           for j, (mm_, s) in enumerate(zip(m, structs))])
            r1 = stat(mm, y_up, base_up, pool & ~is_oos)
            if r1 and r1["n"] >= 30:
                rows.append({"cell": f"{name} {tag}", **{k: r1[k] for k in ("n", "k", "p", "dev", "pval")},
                             "ci": r1["ci"]})
                print(fmt(r1, base_up, f"{name} {tag}"))
    # 中位斜率参考
    if vs:
        print(f"[斜率参考·V类] s_left 中位={np.nanmedian([s['s_left'] for s in vs]):.1f} "
              f"s_right 中位={np.nanmedian([s['s_right'] for s in vs]):.1f} "
              f"s_fast(谷后60s) 中位={np.nanmedian([s['s_fast'] for s in vs]):.1f} pp/min")
    if ivs:
        print(f"[斜率参考·倒V类] s_left 中位={np.nanmedian([s['s_left'] for s in ivs]):.1f} "
              f"s_right 中位={np.nanmedian([s['s_right'] for s in ivs]):.1f} "
              f"s_fast(峰后60s) 中位={np.nanmedian([s['s_fast'] for s in ivs]):.1f} pp/min")

    # ============ 二、用户核心假设：深V × 快右臂 × 背离 ============
    print(f"\n===== 二、核心假设检验：深V × 快臂 × 背离（发现集）=====")
    hyps: list[tuple] = []
    # 背离的操作化族：
    # divA 收尾背离 |divB V修复失败（V且结算DOWN）| divC 极谷背离（谷≤20且收阳）
    def h(name, desc, cond, dirn):
        hyps.append((name, desc, cond, dirn))

    h("V1_深V", "深V→次窗UP(修复延续)",
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP, "UP")
    h("V2_深V×快臂", "深V且右臂≥%.0fpp/min" % F_FAST,
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP and s["s_fast"] >= F_FAST, "UP")
    h("V3_深V×快臂×深谷", "+谷≤25",
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP and s["s_fast"] >= F_FAST
      and s["ext_val"] <= 25, "UP")
    h("V4_深V×快臂×谷后置", "+极值位≥0.5（V完成于后半窗）",
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP and s["s_fast"] >= F_FAST
      and s["ext_pos"] >= 0.5, "UP")
    h("IV1_深倒V", "深倒V→次窗DOWN", lambda s, r: s["type"] == "INV_V" and s["depth"] >= D_DEEP, "DOWN")
    h("IV2_深倒V×快臂", "深倒V且砸落≥%.0fpp/min" % F_FAST,
      lambda s, r: s["type"] == "INV_V" and s["depth"] >= D_DEEP and s["s_fast"] <= -F_FAST, "DOWN")
    h("IV3_深倒V×快臂×高顶", "+顶≥75",
      lambda s, r: s["type"] == "INV_V" and s["depth"] >= D_DEEP and s["s_fast"] <= -F_FAST
      and s["ext_val"] >= 75, "DOWN")
    h("D1_深V×修复失败", "深V但结算DOWN（修复失败）→次窗DOWN",
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP and r["out"] == "DOWN", "DOWN")
    h("D2_深V×修复成功", "深V且结算UP→次窗UP",
      lambda s, r: s["type"] == "V" and s["depth"] >= D_DEEP and r["out"] == "UP", "UP")
    h("D3_极谷背离", "谷≤20且收阳（人群80%看跌过却UP）→次窗DOWN?",
      lambda s, r: s["type"] == "V" and s["ext_val"] <= 20 and r["out"] == "UP", "DOWN")
    h("D4_极顶背离", "顶≥80且收阴→次窗UP?",
      lambda s, r: s["type"] == "INV_V" and s["ext_val"] >= 80 and r["out"] == "DOWN", "UP")
    h("A1_收尾背离增强", "|delta|≥20且结算反向（X系推广）→次窗顺delta",
      lambda s, r: abs(s["delta"]) >= 20 and ((s["delta"] > 0) != (r["out"] == "UP")), "DELTA")

    # A1（DELTA 方向）专用目标：次窗是否顺 delta 方向
    y_align = np.array([
        (1.0 if r["next_out"] == ("UP" if structs[j]["delta"] > 0 else "DOWN") else 0.0)
        if r["has_next"] and structs[j] and abs(structs[j]["delta"]) >= 10 else np.nan
        for j, r in enumerate(recs)])

    l1v: list[dict] = []
    for name, desc, cond, dirn in hyps:
        m = M(cond)
        if dirn == "DELTA":
            r1 = stat(m, y_align, 0.5, pool & ~is_oos)
        else:
            r1 = stat(m, y_up, base_up, pool & ~is_oos)
        if r1 is None:
            continue
        rc = stat(m, y_same, base_same, pool & ~is_oos)
        r1.update({"name": name, "desc": desc, "dir": dirn,
                   "cont": (rc["p"] if rc else None)})
        l1v.append(r1)
        cont_s = f" 延续={rc['p']:.1%}" if rc else ""
        print(fmt(r1, base_up, f"{name} [{desc}]"[:44] + cont_s))
    fdr = bh_fdr([r["pval"] for r in l1v], q=0.1)
    for r, okk in zip(l1v, fdr):
        r["fdr"] = bool(okk)
    print(f"[FDR] {sum(fdr)}/{len(l1v)} 通过 BH-FDR(q=0.1)")

    # ============ 三、OOS 终验 + 经济账 ============
    print(f"\n===== 三、OOS 终验 + 经济账（仅 OOS 真实报价，费2%%+溢1）=====")
    finals = []
    for r in sorted(l1v, key=lambda x: -abs(x["dev"])):
        m = M(dict((h2[0], h2[2]) for h2 in hyps)[r["name"]])
        if r["dir"] == "DELTA":
            r_oos = stat(m, y_align, 0.5, pool & is_oos)
        else:
            r_oos = stat(m, y_up, base_up_oos, pool & is_oos)
        if r_oos is None:
            continue
        k_all = r["k"] + r_oos["k"]
        n_all = r["n"] + r_oos["n"]
        lo_all, hi_all = wilson_bounds(k_all, n_all)
        # 交易方向解析与逐注 EV（次窗 +150s 入场，仅 OOS 段）
        idxs, ds = [], []
        for j, s in enumerate(structs):
            if not s or not m[j] or not recs[j]["has_next"] or not is_oos[j]:
                continue
            d = r["dir"]
            if d == "DELTA":
                d = "UP" if s["delta"] > 0 else "DOWN"
            idxs.append(j + 1)
            ds.append(d)
        ev = ev_eval(idxs, ds, W)
        # 按日稳定性
        byday: dict[str, list] = {}
        y_view = y_align if r["dir"] == "DELTA" else y_up
        for j in np.where(m & is_oos & pool)[0]:
            byday.setdefault(time.strftime("%m-%d", time.gmtime(recs[j]["start"] / 1000)), []) \
                .append(y_view[j])
        daily = {d: (len(v2), float(np.nanmean(v2))) for d, v2 in sorted(byday.items())}
        oos_ok = r_oos["n"] >= OOS_MIN_N and np.sign(r_oos["dev"]) == np.sign(r["dev"])
        # 方向口径换算：dir=DOWN 时胜率/CI 取镜像；假设方向与数据相反时标注 ⚠
        wr_up = k_all / n_all
        if r["dir"] == "DOWN":
            wr, lo_d, hi_d = 1 - wr_up, 1 - hi_all, 1 - lo_all
        else:
            wr, lo_d, hi_d = wr_up, lo_all, hi_all
        wrong_way = (r["dir"] == "UP" and r["dev"] < 0) or (r["dir"] == "DOWN" and r["dev"] > 0)
        finals.append({"name": r["name"], "desc": r["desc"], "dir": r["dir"], "fdr": r["fdr"],
                       "is": {k: r[k] for k in ("n", "p", "dev", "pval")},
                       "oos": {"n": r_oos["n"], "p": r_oos["p"], "dev": r_oos["dev"]},
                       "merged": {"n": n_all, "p": wr, "ci": [lo_d, hi_d],
                                  "p_up": wr_up, "ci_up": [lo_all, hi_all]},
                       "oos_pass": bool(oos_ok), "hypo_wrong_way": bool(wrong_way),
                       "ev": ev,
                       "daily_oos": {d: list(v2) for d, v2 in daily.items()}})
        flag = "✓" if oos_ok else "✗"
        warn = " ⚠假设方向与数据相反" if wrong_way else ""
        print(f"  {flag} {r['name']:<26} 发现 n={r['n']:>4} dev{r['dev']:+.1%}pp | "
              f"OOS n={r_oos['n']:>3} dev{r_oos['dev']:+.1%}pp | 合并WR(方向)={wr:.1%} "
              f"CI[{lo_d:.1%},{hi_d:.1%}] {'FDR✓' if r['fdr'] else ''}{warn}")
        if ev.get("n"):
            vet = ev.get("ev_2_1_ci") or [float("nan")]
            px = ev.get("avg_price") or 0.5
            be = 1 / (1 + (0.98 / min(max(px + .01, .01), .99) - 1))
            print(f"      EV(费2+溢1)={ev.get('ev_2_1', float('nan')):+.4f} CI[{vet[0]:+.3f},{vet[1]:+.3f}] "
                  f"| 实价覆盖 {ev.get('n_real', 0)}/{ev['n']} 均价={px:.3f} 打平={be:.1%}")
        if daily:
            print("      按日(方向胜率): " + " ".join(f"{d}:{p:.0%}({n})" for d, (n, p) in daily.items()))

    report = {"meta": {"windows": len(W), "split_ts": split_ts,
                       "thresholds": {"D_DEEP": D_DEEP, "F_FAST": F_FAST, "MIN_DEPTH": MIN_DEPTH},
                       "bases": {"up_is": base_up, "same_is": base_same, "up_oos": base_up_oos}},
              "taxonomy": rows, "hypotheses": l1v, "finals": finals}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[已写入] {out_path}")
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="深V背离分类学（结构提取→分类→斜率→OOS→经济账）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default="output/vshape_divergence.json")
    args = ap.parse_args()
    with open(args.from_file, encoding="utf-8") as f:
        W = json.load(f)
    run(W, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
