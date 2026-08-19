#!/usr/bin/env python3
"""错位项大扫描：以「曲线 × 结算」错位为基础的假设工厂（系统派穷举 + 涌现派预感）。

背景（local_curve_pure_dim.py 的否定性结论）：
  纯曲线形状（eff/tail/dwell/cross/estreak）对次窗方向零预测力；
  可预测的信息源在「情绪定价 × 结算现实」的分歧瞬间（A1/X3/X4 谱系）。
  本脚本把错位项作为第一性维度做穷举 + 发散扫描。

错位源枚举（系统派 D 系）：
  D1 收尾档位 × 结算（X 系网格 + 一致性对照）
  D3 全窗动量 × 结算（A1 剂量反应 10/15/20 + 同向对照）
  D4 末段动能 × 结算（A1 的尾段版）
  D5 情绪连击 vs 结算 / D6 结算连击 vs 情绪位置
  D7 楔子 = up_token价×100 − up_pct（流动性/拥挤错位，覆盖 22.5%）
  D8 周期末窗错位（grid=2 加权版）

涌现派（E 系，允许机制牵强，先射后验；E3 狼来了同条件双押修正论/惯性论）：
  E1 守门员晚扑救（C_late 宽化）E2 不信邪第二击 E3 狼来了连续错位
  E4 价格不撒谎（touch 极端+反结算）E5 半信半疑（低拥挤延续）E7 钉子户（死水+明确结算）
  E8 抢跑失败 E9 教学日 regime E10 延迟修正（次次窗，S5'' 谱系）
  E11 开盘否定 E12 双向拉扯 E13 共识瓦解前兆

纪律：70/30 切分（同 discovery）；入场券 n≥100（错位交乘天然稀疏，较 150 放宽，
      声明在案）且 |dev|≥1.5pp 且 p<0.05；sys/evo/xcheck 三家族分立 BH-FDR(q=0.1)；
      镜像对强制对称性；L2 = 存活 × 调节 + 系统派×涌现派跨流组合；
      OOS 只验不改（n≥40 方向一致）；配额按 parent 去重（档位/调节变体共享名额）；
      经济账费 2%+溢 0.01。

用法：
    python scripts/local_misalignment_scan.py [--from-file sentiment_windows.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
    HOLDOUT_RATIO, curve_arrays, entry_quote, eval_mask, ev_eval,
)
from local_curve_pure_dim import build_pure  # noqa: E402  pf/estreak/phi/mis 已含

MIN_N = 100           # L1 入场券（错位交乘稀疏，自 150 放宽，声明在案）
MIN_DEV = 0.015
P_SIG = 0.05
OOS_MIN_N = 40
L2_FLOOR = 50
TEACH_T = 0.55        # 教学日 regime 阈值（过去 48 窗方向性错位率）
WEDGE_T = 2.0         # 楔子阈值（p90≈1.5，取 2 为显著带）
OUT_PATH = "output/misalignment_scan.json"


# ============================================================
# 错位特征扩展：wedge / 方向性错位 / 连续错位 / 教学日 / 次次窗
# ============================================================

def extend(recs: list[dict], W: list[dict]) -> list[dict]:
    for r, w in zip(recs, W):
        # 楔子：up token 真实价 ×100 − up_pct（买盘为概率付出的额外溢价）
        ups = sorted(w.get("curve_up_price") or [], key=lambda p: p.get("t", 0))
        pcs = sorted(w.get("curve_up_pct") or [], key=lambda p: p.get("t", 0))
        r["wedge_end"] = r["wedge_delta"] = float("nan")
        if len(ups) >= 2 and len(pcs) >= 2 and ups[0].get("v") is not None \
                and pcs[0].get("v") is not None:
            we_s = ups[0]["v"] * 100 - pcs[0]["v"]
            we_e = ups[-1]["v"] * 100 - pcs[-1]["v"]
            r["wedge_end"], r["wedge_delta"] = we_e, we_e - we_s
        end = r["f"]["end"] if r.get("f") else None
        r["x3"] = bool(r["out"] == "DOWN" and end is not None and end >= 60)
        r["x4"] = bool(r["out"] == "UP" and end is not None and end <= 40)
        # 方向性错位（不看档位）：情绪偏一侧但结算反向
        r["mis_up"] = bool(end is not None and end > 50 and r["out"] == "DOWN")
        r["mis_dn"] = bool(end is not None and end < 50 and r["out"] == "UP")

    # 前窗错位链 + 教学日 regime（只含已结算窗）
    for i, r in enumerate(recs):
        p = recs[i - 1] if i > 0 and r["prev_ok"] else None
        r["prev_x3"] = bool(p and p["x3"])
        r["prev_x4"] = bool(p and p["x4"])
        r["prev_mis_up"] = bool(p and p["mis_up"])
        r["prev_mis_dn"] = bool(p and p["mis_dn"])
        hist = [recs[j] for j in range(max(0, i - 47), i + 1)]
        flags = [1 for h in hist if h.get("mis_up") or h.get("mis_dn")]
        r["misrate48"] = (sum(flags) / len(hist)) if len(hist) >= 24 else float("nan")

    # 次次窗目标（E10 延迟修正）
    for i in range(len(recs) - 2):
        r, r1, r2 = recs[i], recs[i + 1], recs[i + 2]
        if (r1["start"] - r["start"] == 300_000 and r2["start"] - r1["start"] == 300_000
                and r["out"] in ("UP", "DOWN") and r2["out"] in ("UP", "DOWN")):
            r["has_next2"] = True
            r["next2_out"] = r2["out"]
    return recs


# ============================================================
# 假设注册表：(name, school, mech, fn, dir, target)
#   school: sys=系统派穷举 / evo=涌现派预感 / xcheck=对照锚点
#   target: next=次窗（默认）/ next2=次次窗
# ============================================================

def register(recs) -> list[tuple]:
    def g(r, path, default=float("nan")):
        cur = r
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    ok = lambda x: x == x  # noqa: E731
    H: list[tuple] = []

    def h(name, school, mech, fn, direction, target="next"):
        H.append((name, school, mech,
                  lambda R, fn=fn: np.array([bool(fn(r)) for r in R]), direction, target))

    # ================= 系统派 D：错位源 × 档位网格 =================
    # D1 收尾档位 × 结算（X 系网格 + 一致性对照）
    h("D1_x3_60", "sys", "收阴但情绪≥60（X3 原版）→ 修正 UP", lambda r: r["x3"], "UP")
    h("D1_x3_70", "sys", "收阴但情绪≥70（强档）", lambda r: g(r, "f.end") >= 70 and r["out"] == "DOWN", "UP")
    h("D1_x4_40", "sys", "收阳但情绪≤40（X4 原版）→ 修正 DOWN", lambda r: r["x4"], "DOWN")
    h("D1_x4_30", "sys", "收阳但情绪≤30（强档）", lambda r: g(r, "f.end") <= 30 and r["out"] == "UP", "DOWN")
    h("D1_cons_up", "sys", "一致性对照：收阳且情绪≥60 → 延续", lambda r: g(r, "f.end") >= 60 and r["out"] == "UP", "UP")
    h("D1_cons_dn", "sys", "一致性对照：收阴且情绪≤40", lambda r: g(r, "f.end") <= 40 and r["out"] == "DOWN", "DOWN")
    # D3 全窗动量 × 结算（A1 剂量反应 10/15/20 + 同向对照）
    for t in (10, 15, 20):
        h(f"D3_a{t}_up", "sys", f"A1 剂量：Δ≥+{t} 且收阴 → 次窗 UP",
          lambda r, t=t: g(r, "pf.delta") >= t and r["out"] == "DOWN", "UP")
        h(f"D3_a{t}_dn", "sys", f"A1 剂量：Δ≤-{t} 且收阳 → 次窗 DOWN",
          lambda r, t=t: g(r, "pf.delta") <= -t and r["out"] == "UP", "DOWN")
    h("D3_s20_up", "sys", "同向对照：Δ≥+20 且收阳 → 延续 UP",
      lambda r: g(r, "pf.delta") >= 20 and r["out"] == "UP", "UP")
    h("D3_s20_dn", "sys", "同向对照：Δ≤-20 且收阴",
      lambda r: g(r, "pf.delta") <= -20 and r["out"] == "DOWN", "DOWN")
    # D4 末段动能 × 结算（A1 尾段版）
    for t in (5, 10):
        h(f"D4_t{t}_up", "sys", f"末段冲 +{t} 但收阴 → 顺尾段 UP",
          lambda r, t=t: g(r, "pf.d_tail") >= t and r["out"] == "DOWN", "UP")
        h(f"D4_t{t}_dn", "sys", f"末段跌 -{t} 但收阳 → 顺尾段 DOWN",
          lambda r, t=t: g(r, "pf.d_tail") <= -t and r["out"] == "UP", "DOWN")
    # D5 情绪连击 vs 结算（mis 的 estreak 版）
    h("D5_estk_mis_up", "sys", "情绪连升≥2 却收阴 → 修正 UP",
      lambda r: r.get("estreak_up", 0) >= 2 and r["out"] == "DOWN", "UP")
    h("D5_estk_mis_dn", "sys", "情绪连降≥2 却收阳 → 修正 DOWN",
      lambda r: r.get("estreak_dn", 0) >= 2 and r["out"] == "UP", "DOWN")
    # D6 结算连击 vs 情绪位置（人群不信连涨）
    h("D6_disbel_up", "sys", "out 2 连阳但情绪≤45（不信涨）→ 补涨 UP",
      lambda r: r.get("streak_up", 0) >= 2 and g(r, "f.end") <= 45, "UP")
    h("D6_disbel_dn", "sys", "out 2 连阴但情绪≥55（不信跌）→ 补跌 DOWN",
      lambda r: r.get("streak_dn", 0) >= 2 and g(r, "f.end") >= 55, "DOWN")
    # D7 楔子（token 价 ×100 − pct：买盘拥挤溢价）
    h("D7_wedge_hi", "sys", "收尾楔子≥2（UP token 贵过概率=买盘拥挤）→ DOWN",
      lambda r: ok(g(r, "wedge_end")) and r["wedge_end"] >= WEDGE_T, "DOWN")
    h("D7_wedge_lo", "sys", "收尾楔子≤-2（卖压折价）→ UP",
      lambda r: ok(g(r, "wedge_end")) and r["wedge_end"] <= -WEDGE_T, "UP")
    h("D7_wedgrow_hi", "sys", "楔子日内扩大≥2（流动性撤离）→ DOWN",
      lambda r: ok(g(r, "wedge_delta")) and r["wedge_delta"] >= WEDGE_T, "DOWN")
    h("D7_wedgrow_lo", "sys", "楔子收窄≥2 → UP",
      lambda r: ok(g(r, "wedge_delta")) and r["wedge_delta"] <= -WEDGE_T, "UP")
    # D8 周期末窗错位（grid=2 加权版 X 系）
    h("D8_cyc_x3", "sys", "周期末窗收阴但情绪≥60 → 次窗 UP",
      lambda r: r["grid"] == 2 and r["x3"], "UP")
    h("D8_cyc_x4", "sys", "周期末窗收阳但情绪≤40 → 次窗 DOWN",
      lambda r: r["grid"] == 2 and r["x4"], "DOWN")

    # ================= 涌现派 E：预感清单（允许牵强，先射后验）=================
    h("E1_lateflip_dn", "evo", "守门员晚扑救：开局偏多、前半仍正、收尾≤45 → DOWN",
      lambda r: g(r, "f.start") >= 50 and g(r, "pf.d_head") > 0 and g(r, "f.end") <= 45, "DOWN")
    h("E1_lateflip_up", "evo", "对称：开局偏空、前半仍负、收尾≥55 → UP",
      lambda r: g(r, "f.start") < 50 and g(r, "pf.d_head") < 0 and g(r, "f.end") >= 55, "UP")
    h("E2_double_x4", "evo", "不信邪第二击：连续两窗收阳但情绪都≤40 → 宣泄 DOWN",
      lambda r: r["prev_x4"] and r["x4"], "DOWN")
    h("E2_double_x3", "evo", "对称：连续两窗收阴但情绪都≥60 → UP",
      lambda r: r["prev_x3"] and r["x3"], "UP")
    h("E3_wolf_fix_up", "evo", "狼来了·修正论：连续 2 窗高估（>50 收阴）→ 修正 UP",
      lambda r: r["prev_mis_up"] and r["mis_up"], "UP")
    h("E3_wolf_mom_dn", "evo", "狼来了·惯性论：同条件 → 惯性 DOWN（双押对照）",
      lambda r: r["prev_mis_up"] and r["mis_up"], "DOWN")
    h("E3_wolf_fix_dn", "evo", "对称·修正论：连续 2 窗低估（<50 收阳）→ DOWN",
      lambda r: r["prev_mis_dn"] and r["mis_dn"], "DOWN")
    h("E3_wolf_mom_up", "evo", "对称·惯性论：同条件 → UP",
      lambda r: r["prev_mis_dn"] and r["mis_dn"], "UP")
    h("E4_truth_dn", "evo", "价格不撒谎：touch≥90 却收阴 → 顺结算 DOWN",
      lambda r: g(r, "f.max") >= 90 and r["out"] == "DOWN", "DOWN")
    h("E4_truth_up", "evo", "对称：touch≤10 却收阳 → UP",
      lambda r: g(r, "f.min") <= 10 and r["out"] == "UP", "UP")
    h("E5_hedged_up", "evo", "半信半疑：45≤end≤55 且收阳（低拥挤上涨）→ 延续 UP",
      lambda r: 45 <= g(r, "f.end") <= 55 and r["out"] == "UP", "UP")
    h("E5_hedged_dn", "evo", "对称：45≤end≤55 且收阴 → DOWN",
      lambda r: 45 <= g(r, "f.end") <= 55 and r["out"] == "DOWN", "DOWN")
    h("E7_dead_up", "evo", "钉子户：死水窗（range≤5）却收阳（人群没定价的运动）→ UP",
      lambda r: g(r, "f.range") <= 5 and r["out"] == "UP", "UP")
    h("E7_dead_dn", "evo", "对称：死水窗收阴 → DOWN",
      lambda r: g(r, "f.range") <= 5 and r["out"] == "DOWN", "DOWN")
    h("E8_spoil_dn", "evo", "抢跑失败：冲≥90 回落到≤80 且收阴 → DOWN",
      lambda r: g(r, "f.max") >= 90 and g(r, "f.end") <= 80 and r["out"] == "DOWN", "DOWN")
    h("E8_spoil_up", "evo", "对称：探≤10 回升到≥20 且收阳 → UP",
      lambda r: g(r, "f.min") <= 10 and g(r, "f.end") >= 20 and r["out"] == "UP", "UP")
    h("E9_teach_dn", "evo", "教学日（48 窗错位率≥55%）且情绪≥60 → 反情绪 DOWN",
      lambda r: ok(g(r, "misrate48")) and r["misrate48"] >= TEACH_T and g(r, "f.end") >= 60, "DOWN")
    h("E9_teach_up", "evo", "教学日且情绪≤40 → 反情绪 UP",
      lambda r: ok(g(r, "misrate48")) and r["misrate48"] >= TEACH_T and g(r, "f.end") <= 40, "UP")
    h("E10_delay_x4", "evo", "延迟修正：X4 窗的次次窗 DOWN（S5'' 谱系）",
      lambda r: r["x4"], "DOWN", target="next2")
    h("E10_delay_x3", "evo", "对称：X3 窗的次次窗 UP",
      lambda r: r["x3"], "UP", target="next2")
    h("E11_deny_up", "evo", "开盘否定：前窗收阴但本窗开局≥55（人群否定）→ UP",
      lambda r: r.get("prev_out") == "DOWN" and g(r, "f.start") >= 55, "UP")
    h("E11_deny_dn", "evo", "对称：前窗收阳但开局≤45 → DOWN",
      lambda r: r.get("prev_out") == "UP" and g(r, "f.start") <= 45, "DOWN")
    h("E12_tug_flip_dn", "evo", "双向拉扯：窗内多空双驻（hi/lo_share≥0.2）且收阳 → 次窗反转 DOWN",
      lambda r: g(r, "pf.hi_share") >= 0.2 and g(r, "pf.lo_share") >= 0.2 and r["out"] == "UP", "DOWN")
    h("E12_tug_flip_up", "evo", "对称：同条件收阴 → UP",
      lambda r: g(r, "pf.hi_share") >= 0.2 and g(r, "pf.lo_share") >= 0.2 and r["out"] == "DOWN", "UP")
    h("E13_crack_dn", "evo", "共识瓦解前兆：情绪≥70 但末段回落 → DOWN",
      lambda r: g(r, "f.end") >= 70 and g(r, "pf.d_tail") < 0, "DOWN")
    h("E13_crack_up", "evo", "对称：情绪≤30 但末段回升 → UP",
      lambda r: g(r, "f.end") <= 30 and g(r, "pf.d_tail") > 0, "UP")

    # ================= 对照锚点（复现已知结果，校验口径）=================
    h("R1_stk3_up", "xcheck", "out 3 连阳 → UP（动量搬运上限）",
      lambda r: r.get("streak_up", 0) >= 3, "UP")
    h("R2_stk3_dn", "xcheck", "out 3 连阴 → DOWN",
      lambda r: r.get("streak_dn", 0) >= 3, "DOWN")
    h("R3_end70_up", "xcheck", "无条件 end≥70 → UP（L1 档位锚点）",
      lambda r: g(r, "f.end") >= 70, "UP")
    h("R4_estk2_up", "xcheck", "无条件情绪连升≥2 → UP（无错位交互锚点）",
      lambda r: r.get("estreak_up", 0) >= 2, "UP")
    return H


# 镜像对（对称性检验）
SYMMETRY_PAIRS = [
    ("D1_x3_60", "D1_x4_40"), ("D1_x3_70", "D1_x4_30"),
    ("D3_a10_up", "D3_a10_dn"), ("D3_a15_up", "D3_a15_dn"), ("D3_a20_up", "D3_a20_dn"),
    ("D3_s20_up", "D3_s20_dn"),
    ("D4_t5_up", "D4_t5_dn"), ("D4_t10_up", "D4_t10_dn"),
    ("D5_estk_mis_up", "D5_estk_mis_dn"), ("D6_disbel_up", "D6_disbel_dn"),
    ("D7_wedge_hi", "D7_wedge_lo"), ("D7_wedgrow_hi", "D7_wedgrow_lo"),
    ("D8_cyc_x3", "D8_cyc_x4"),
    ("E1_lateflip_dn", "E1_lateflip_up"), ("E2_double_x4", "E2_double_x3"),
    ("E4_truth_dn", "E4_truth_up"), ("E5_hedged_up", "E5_hedged_dn"),
    ("E7_dead_up", "E7_dead_dn"), ("E8_spoil_dn", "E8_spoil_up"),
    ("E9_teach_dn", "E9_teach_up"), ("E10_delay_x4", "E10_delay_x3"),
    ("E11_deny_up", "E11_deny_dn"), ("E12_tug_flip_dn", "E12_tug_flip_up"),
    ("E13_crack_dn", "E13_crack_up"),
]


def moderators(recs) -> dict[str, callable]:
    def m(fn):
        return lambda R: np.array([bool(fn(r)) for r in R])
    ok = lambda x: x == x  # noqa: E731
    return {
        "phi_pos": m(lambda r: ok(r.get("phi", float("nan"))) and r["phi"] > 0.15),
        "phi_neg": m(lambda r: ok(r.get("phi", float("nan"))) and r["phi"] < -0.15),
        "grid_first": m(lambda r: r["grid"] == 0),
        "grid_last": m(lambda r: r["grid"] == 2),
        "asia": m(lambda r: r["hour"] <= 7),
        "us": m(lambda r: r["hour"] >= 16),
        "teach_day": m(lambda r: ok(r.get("misrate48", float("nan"))) and r["misrate48"] >= TEACH_T),
        "calm_day": m(lambda r: ok(r.get("misrate48", float("nan"))) and r["misrate48"] <= 0.45),
        "streak_up2*": m(lambda r: r.get("streak_up", 0) >= 2),
        "streak_dn2*": m(lambda r: r.get("streak_dn", 0) >= 2),
    }


# ============================================================
# 主流程
# ============================================================

def run(W: list[dict], out_path: str) -> dict:
    W = sorted(W, key=lambda w: int(w["start_time"]))
    split_ts = int(W[int(len(W) * (1 - HOLDOUT_RATIO))]["start_time"])
    recs = extend(build_pure(W), W)
    arr = np.array(recs, dtype=object)

    pool = np.array([r["has_next"] for r in recs])
    pool2 = np.array([r.get("has_next2", False) for r in recs])
    tgt_up = np.array([1.0 if (r["next_out"] or "") == "UP" else
                       (0.0 if (r["next_out"] or "") == "DOWN" else np.nan) for r in recs])
    tgt2_up = np.array([1.0 if (r.get("next2_out") or "") == "UP" else
                        (0.0 if (r.get("next2_out") or "") == "DOWN" else np.nan) for r in recs])
    is_oos = np.array([r["start"] >= split_ts for r in recs])
    base_is = float(np.nanmean(tgt_up[~is_oos & pool]))
    base_oos = float(np.nanmean(tgt_up[is_oos & pool]))
    base2_is = float(np.nanmean(tgt2_up[~is_oos & pool2]))
    base2_oos = float(np.nanmean(tgt2_up[is_oos & pool2]))
    print(f"[数据] {len(W)} 窗 | 切分 {time.strftime('%m-%d %H:%M', time.gmtime(split_ts/1000))} UTC | "
          f"next: IS {int((~is_oos & pool).sum())}/OOS {int((is_oos & pool).sum())} "
          f"基准 {base_is:.1%}/{base_oos:.1%} | next2 池 {int(pool2.sum())} 基准 {base2_is:.1%}/{base2_oos:.1%}")

    H = register(recs)
    fns = {n: fn for n, _, _, fn, _, _ in H}
    dir_of = {n: d for n, _, _, _, d, _ in H}
    mech_of = {n: mc for n, _, mc, _, _, _ in H}
    tgt_of = {n: t for n, _, _, _, _, t in H}
    school_of = {n: s for n, s, _, _, _, _ in H}

    # ---------- L1：三家族分立 FDR + 入场券 ----------
    l1: list[dict] = []
    for name, school, mech, fn, td, target in H:
        tgt = tgt2_up if target == "next2" else tgt_up
        pl = pool2 if target == "next2" else pool
        bs = base2_is if target == "next2" else base_is
        r_is = eval_mask(fn(arr), tgt, bs, pl & ~is_oos)
        if r_is is None:
            continue
        r_is.update({"name": name, "school": school, "mech": mech, "dir": td, "target": target})
        l1.append(r_is)
    for school in ("sys", "evo", "xcheck"):
        rows = [x for x in l1 if x["school"] == school]
        passed = bh_fdr([x["pval"] for x in rows], q=0.1)
        for x, ok_ in zip(rows, passed):
            x["fdr"] = ok_
            x["ticket"] = x["n"] >= MIN_N and abs(x["dev"]) >= MIN_DEV and x["pval"] < P_SIG
    print(f"\n===== L1（IS，{len(l1)} 检验 | sys {sum(1 for x in l1 if x['school']=='sys')} "
          f"evo {sum(1 for x in l1 if x['school']=='evo')} "
          f"xcheck {sum(1 for x in l1 if x['school']=='xcheck')}）=====")
    for r in sorted(l1, key=lambda x: -abs(x["dev"])):
        star = "*" if r.get("ticket") else " "
        fdr = "F" if r.get("fdr") else " "
        print(f" {star}{fdr} [{r['school']:<3}|{r['target']:<5}] {r['name']:<16} n={r['n']:>5} "
              f"押{r['dir']:<4}{r['p']:6.1%} (dev {r['dev']:+.1%}pp) "
              f"CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}] p={r['pval']:.3f}")

    # ---------- 对称性检验 ----------
    l1map = {r["name"]: r for r in l1}
    print("\n===== 对称性检验（只列有信号的镜像对；全零对自动对称）=====")
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
        if abs(ra["dev"]) >= 0.02 or abs(rb["dev"]) >= 0.02 or sig_a or sig_b:
            print(f"  {a:<16} dev {ra['dev']:+.1%} ({'显著' if sig_a else '不显著'}) | "
                  f"{b:<16} 等效 dev {dev_b_eq:+.1%} ({'显著' if sig_b else '不显著'}) → {verdict}")

    # ---------- L2：存活 × 调节 + 系统派×涌现派跨流组合 ----------
    surv = sorted([r for r in l1 if r["ticket"] and r["school"] != "xcheck"],
                  key=lambda x: -abs(x["dev"]))
    mods = moderators(recs)
    l2: list[dict] = []
    for r in surv[:6]:
        if r["target"] != "next":
            continue  # next2 目标不做调节（池太小）
        for mn, mfn in mods.items():
            m = fns[r["name"]](arr) & mfn(arr)
            rr = eval_mask(m, tgt_up, base_is, pool & ~is_oos)
            if rr and rr["n"] >= L2_FLOOR:
                rr.update({"name": f"{r['name']} × {mn}", "parents": [r["name"]], "mod": mn,
                           "school": r["school"], "dir": r["dir"], "target": "next"})
                l2.append(rr)
    mains = [r for r in surv[:6] if r["target"] == "next"]
    for i, ra in enumerate(mains):
        for rb in mains[i + 1:]:
            if school_of[ra["name"]] == school_of[rb["name"]] or ra["dir"] != rb["dir"]:
                continue  # 跨流派、同方向才组合（机制可叠加）
            m = fns[ra["name"]](arr) & fns[rb["name"]](arr)
            rr = eval_mask(m, tgt_up, base_is, pool & ~is_oos)
            if rr and rr["n"] >= L2_FLOOR:
                rr.update({"name": f"{ra['name']} × {rb['name']}", "parents": [ra["name"], rb["name"]],
                           "mod": None, "school": "mix", "dir": ra["dir"], "target": "next"})
                l2.append(rr)
    l2.sort(key=lambda x: -abs(x["dev"]))
    print(f"\n===== L2 组合（IS，{len(l2)} 组 | floor n≥{L2_FLOOR}）=====")
    for r in l2[:12]:
        print(f"  [{r['school']:<3}] {r['name']:<40} n={r['n']:>5} 押{r['dir']:<4}"
              f"{r['p']:6.1%} (dev {r['dev']:+.1%}pp) p={r['pval']:.3f}")

    # ---------- OOS 终验（sys 5 / evo 5 / mix 并入所属最强流派）----------
    print(f"\n===== OOS 终验（n≥{OOS_MIN_N} 且方向一致）=====")
    cand = sorted(surv + [r for r in l2 if r["n"] >= L2_FLOOR], key=lambda x: -abs(x["dev"]))
    quota = {"sys": 5, "evo": 5, "mix": 2}
    finals = []
    for school, cap in quota.items():
        picked = 0
        used: set[str] = set()
        for r in cand:
            if picked >= cap or r["school"] != school:
                continue
            ps = r.get("parents")
            if ps and len(ps) > 1:
                key = tuple(sorted(parent_of(p) for p in ps))  # 跨流组合：独立名额
            else:
                key = parent_of(ps[0] if ps else r["name"])   # 原版/调节变体：同享名额
            if key in used:
                continue  # 同 parent 变体已占名额，防三胞胎挤占
            names = (r.get("parents") or [r["name"]]) + ([r["mod"]] if r.get("mod") else [])
            m = np.ones(len(recs), dtype=bool)
            good = True
            for nm in names:
                if nm in fns:
                    m &= fns[nm](arr)
                elif nm in mods:
                    m &= mods[nm](arr)
                else:
                    good = False
                    break
            if not good:
                continue
            used.add(key)
            tgt = tgt2_up if r["target"] == "next2" else tgt_up
            pl = pool2 if r["target"] == "next2" else pool
            bs = base2_oos if r["target"] == "next2" else base_oos
            r_oos = eval_mask(m, tgt, bs, pl & is_oos)
            if r_oos is None or r_oos["n"] == 0:
                continue
            dev_is_eq = r["dev"] if r["dir"] == "UP" else -r["dev"]
            dev_oos_eq = r_oos["dev"] if r["dir"] == "UP" else -r_oos["dev"]
            oos_ok = r_oos["n"] >= OOS_MIN_N and np.sign(dev_oos_eq) == np.sign(dev_is_eq)
            k_all, n_all = r["k"] + r_oos["k"], r["n"] + r_oos["n"]
            lo_all, hi_all = wilson_bounds(k_all, n_all)
            ev = _econ(m, is_oos, r["dir"], recs, W, step=2 if r["target"] == "next2" else 1)
            finals.append({
                "name": r["name"], "school": school, "dir": r["dir"], "target": r["target"],
                "mech": mech_of.get(r["parents"][0] if r.get("parents") else r["name"], ""),
                "is": {"n": r["n"], "p": r["p"], "dev": r["dev"]},
                "oos": {"n": r_oos["n"], "p": r_oos["p"], "dev": r_oos["dev"]},
                "oos_pass": bool(oos_ok),
                "combined": {"n": n_all, "p": k_all / n_all, "ci": [lo_all, hi_all]},
                "ev": ev,
            })
            picked += 1
            flag = "✓" if oos_ok else "✗"
            print(f"  {flag} [{school:<3}] {r['name']} (目标 {r['target']})")
            print(f"      IS n={r['n']} 押{r['dir']} {r['p']:.1%} ({r['dev']:+.1%}pp) | "
                  f"OOS n={r_oos['n']} {r_oos['p']:.1%} ({r_oos['dev']:+.1%}pp) | "
                  f"合并 CI[{lo_all:.1%},{hi_all:.1%}]")
            if ev.get("n"):
                print(f"      EV(费2+溢1)={ev.get('ev_2_1', float('nan')):+.4f} "
                      f"CI{ev.get('ev_2_1_ci')} | 实价 {ev.get('n_real', 0)}/{ev['n']} "
                      f"均价={ev.get('avg_price', 0):.3f}")

    report = {
        "meta": {"windows": len(W), "split_ts": split_ts, "base_is": base_is,
                 "base_oos": base_oos, "base2_is": base2_is, "base2_oos": base2_oos,
                 "min_n": MIN_N, "schools": {"sys": "系统派穷举", "evo": "涌现派预感",
                                             "xcheck": "对照锚点"}},
        "l1": l1, "symmetry": sym_rows, "l2": l2[:40], "finals": finals,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[已写入] {out_path}")
    return report


def parent_of(name: str) -> str:
    """父假设归并：档位/调节变体共享 OOS 名额（D1_x4_40 ≡ D1_x4，D3_a10_dn ≡ D3_a_dn）。"""
    m = re.match(r"^(D1_x3|D1_x4|D3_a\d+_(?:up|dn)|D4_t\d+_(?:up|dn))", name)
    if m:
        return re.sub(r"\d+(?=_)", "", m.group(1))
    return name


def _econ(mask: np.ndarray, is_oos: np.ndarray, direction: str, recs, W, step: int) -> dict:
    """OOS 段逐注经济账：next → 次窗入场；next2 → 次次窗入场。"""
    idxs, dirs = [], []
    for j, r in enumerate(recs):
        if not mask[j] or not is_oos[j] or j + step >= len(recs):
            continue
        idxs.append(j + step)
        dirs.append(direction)
    if not idxs:
        return {"n": 0}
    return ev_eval(idxs, dirs, W)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="错位项假设工厂大扫描（系统派+涌现派）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    with open(args.from_file, encoding="utf-8") as f:
        W = json.load(f)
    run(W, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
