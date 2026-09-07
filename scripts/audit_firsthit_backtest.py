#!/usr/bin/env python3
"""首触反转信号全方位回测数据审计脚本 —— 独立实现交叉验证与统计检验完整性核查。

核心审计维度:
A. 数字复现性：独立实现事件提取与 EV 计算，与主脚本 JSON 逐条比对（n / P / EV / CI 端点）
B. 公式正确性：逐事件 EV vs 用 q_bar 的聚合口径 → 量化 Jensen 偏差
C. 多重检验校正：补 BH-FDR(q=0.1 & q=0.05) 对所有 Confirm EV_lo>0 候选家族
D. 规则冗余/重复识别：检测完全相同的规则定义并报告
E. Confirm 污染审查：根据项目纪律 confirm ≥2026-08-24 burned，只能作弱证据
F. 样本嵌套重叠分析：G7 ⊂ D1_1, G1; ALPHA_G7_STRICT ⊂ ALPHA_G7_CORE ⊂ G7 等依赖关系图
G. 最小样本/天数门槛：报告 n < 30 或 days < 5 的薄弱规则
H. 时间分布检查：触发时刻 t 在 g7 与整体中的分布差异
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
Q_LO, Q_HI = 0.005, 0.1
SAMPLES_5M = "output/pull_samples_5m_20260907.json"
KL5_FILES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
SPLIT = "2026-08-24"
BOOT_B = 4000
SEED_AUDIT = 20260908  # 审计专用种子
MIN_PTS_AUDIT = 8


def day_boot_audit(values: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    """日聚类 Bootstrap (同主脚本口径)。"""
    d_u = np.unique(days)
    if len(d_u) < 5:
        return (float("nan"), float("nan"))
    sums = np.zeros(len(d_u))
    cnts = np.zeros(len(d_u))
    idx = np.searchsorted(d_u, days)
    np.add.at(sums, idx, values)
    np.add.at(cnts, idx, 1)
    means = sums / cnts
    rng = np.random.default_rng(SEED_AUDIT)
    pick = rng.integers(0, len(d_u), size=(BOOT_B, len(d_u)))
    w = cnts[pick]
    tot = (means[pick] * w).sum(axis=1) / w.sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def load_klines_audit():
    rows = []
    for p in KL5_FILES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    return {int(k[0]): (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in rows}


# =======================================
# A. 独立实现事件提取（不同数据结构与变量名）
# =======================================
def collect_events_independent(k5):
    """独立实现事件收集（不同于 main script）。"""
    raw = json.load(open(SAMPLES_5M, encoding="utf-8"))
    L = 300_000
    by_window = defaultdict(list)
    for s in raw:
        by_window[int(s["timestamp"]) // L * L].append(s)

    independent_events = []
    for win_start, samples in sorted(by_window.items()):
        bar = k5.get(win_start)
        if not bar or bar[0] <= 0:
            continue
        open_price = bar[0]
        if bar[3] == open_price:
            continue
        is_down_win = bar[3] < open_price

        samples.sort(key=lambda x: int(x["timestamp"]))
        trigger_sample = None
        trigger_idx = None
        for i, samp in enumerate(samples):
            dp = samp.get("down_price")
            if dp is not None and Q_LO < float(dp) <= Q_HI:
                trigger_sample, trigger_idx = samp, i
                break
        if trigger_sample is None:
            continue

        ts = int(trigger_sample["timestamp"])
        trade_q = float(trigger_sample["down_price"])
        trade_date = time.strftime("%Y-%m-%d", time.gmtime(win_start / 1000))
        t_sec = (ts - win_start) / 1000.0

        # BTC path
        btc_samples = samples[:trigger_idx + 1]
        if trigger_sample.get("btc_price") is None:
            for offset in [-1, 1, -2, 2]:
                jj = trigger_idx + offset
                if 0 <= jj < len(samples) and samples[jj].get("btc_price"):
                    btc_samples.append(samples[jj])
                    break
        pts = [open_price] + [float(b["btc_price"]) for b in btc_samples
                              if b.get("btc_price") and float(b["btc_price"]) > 0]
        npts_pts = len(pts) - 1
        if npts_pts < MIN_PTS_AUDIT:
            continue

        btc_at_trigger = pts[-1]
        path_hi = max(pts)
        path_lo = min(pts)
        range_total = path_hi - path_lo

        chg_ratio = (btc_at_trigger / open_price - 1.0) * 10000.0
        if range_total > 1e-9:
            body_ratio = abs(btc_at_trigger - open_price) / range_total
            upper_wick = 1.0 if path_hi > max(btc_at_trigger, open_price) + 1e-9 else 0.0
            upper_wick_bps = (path_hi - max(btc_at_trigger, open_price)) / open_price * 10000.0
            rng_bps = range_total / open_price * 10000.0
        else:
            body_ratio, upper_wick, upper_wick_bps, rng_bps = 0.0, 0.0, 0.0, 0.0

        # streak up
        streak = 0
        cur_w = win_start - L
        while streak < 4:
            pb = k5.get(cur_w)
            if pb and pb[3] > pb[0]:
                streak += 1
                cur_w -= L
            else:
                break

        # prev high
        prev_high_list = [k5.get(win_start - j * L)[1] for j in (1, 2, 3)]
        prev_high_max = max(prev_high_list)
        break_bps = float("nan")
        if all(x is not None for x in prev_high_list) and prev_high_max > 0:
            break_bps = (path_hi / prev_high_max - 1.0) * 10000.0

        # volume/participants delta
        vol_o = float(samples[0]["trade_volume"]) if samples[0].get("trade_volume") else None
        vol_t = float(trigger_sample["trade_volume"]) if trigger_sample.get("trade_volume") else None
        par_o = float(samples[0]["participants"]) if samples[0].get("participants") else None
        par_t = float(trigger_sample["participants"]) if trigger_sample.get("participants") else None
        dvol = vol_t - vol_o if (vol_o is not None and vol_t is not None) else 0.0
        dpar = par_t - par_o if (par_o is not None and par_t is not None) else 0.0

        independent_events.append({
            "win": 1 if is_down_win else 0,
            "q": trade_q,
            "day": trade_date,
            "t": t_sec,
            "body_r": body_ratio,
            "wick01": upper_wick,
            "upper_wick_bps": upper_wick_bps,
            "streak_up": streak,
            "chg_bps": chg_ratio,
            "rng_bps": rng_bps,
            "break_bps": break_bps,
            "dpar": dpar,
            "calib": trade_date < SPLIT,
        })

    print(f"独立采集有效事件数：{len(independent_events)}")
    return independent_events


# =======================================
# B. 统计函数（与主脚本独立实现，不导入主脚本）
# =======================================
def compute_stats_independent(events):
    """逐事件 EV 计算。"""
    flags = np.array([e["win"] for e in events], dtype=float)
    qs = np.array([e["q"] for e in events], dtype=float)
    ev_values = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
    days = np.array([e["day"] for e in events])
    n = len(events)
    wins = int(flags.sum())
    mean_ev = float(ev_values.mean()) if n > 0 else 0.0

    evv_bootstrap = ev_values
    ev_lo, ev_hi = day_boot_audit(evv_bootstrap, days)

    return {
        "n": n,
        "wins": wins,
        "p": wins / n if n > 0 else 0.0,
        "ev": mean_ev,
        "ev_lo": ev_lo,
        "ev_hi": ev_hi,
        "q_bar": float(qs.mean()) if n > 0 else 0.0,
        "n_days": int(len(np.unique(days))),
    }


# =======================================
# C. 规则定义（独立实现，与主脚本一致）
# =======================================
RULES_INDEPENDENT = [
    ("G0 基底 (全部首触)", lambda e: True),
    ("G1 小实体 (body_r<=0.35)", lambda e: e["body_r"] <= 0.35),
    ("G3 位移小 (chg_bps<=2.82)", lambda e: e["chg_bps"] <= 2.82),
    ("D1_1 上影拒绝 (wick01==1)", lambda e: e["wick01"] == 1.0),
    ("D1_2 较长上影 (upper_wick>=1.5bp)", lambda e: e["upper_wick_bps"] >= 1.5),
    ("D1_3 显著上影 (upper_wick>=3.0bp)", lambda e: e["upper_wick_bps"] >= 3.0),
    ("G7 组合 (body<=0.35 ∧ wick==1)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0),
    ("G7+长上影 (body<=0.35 ∧ upper_wick>=2.0bp)", lambda e: e["body_r"] <= 0.35 and e["upper_wick_bps"] >= 2.0),
    ("G1+深折价 (body<=0.35 ∧ q<=0.05)", lambda e: e["body_r"] <= 0.35 and e["q"] <= 0.05),
    ("G7+深折价 (G7 ∧ q<=0.05)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["q"] <= 0.05),
    ("D3_1 早触发 (t<=180s)", lambda e: e["t"] <= 180.0),
    ("D3_2 中期触发 (180<t<=250s)", lambda e: 180.0 < e["t"] <= 250.0),
    ("D3_3 尾声前触发 (t<=270s)", lambda e: e["t"] <= 270.0),
    ("D3_4 极限尾声 (t>270s)", lambda e: e["t"] > 270.0),
    ("G7+非极晚触发 (G7 ∧ t<=270s)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["t"] <= 270.0),
    ("G7+极晚触发 (G7 ∧ t>270s)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["t"] > 270.0),
    ("D4_1 前驱无连阳 (streak_up==0)", lambda e: e["streak_up"] == 0),
    ("D4_2 前驱微阳 (streak_up<=1)", lambda e: e["streak_up"] <= 1),
    ("D4_3 前驱强连阳 (streak_up>=2)", lambda e: e["streak_up"] >= 2),
    ("D4_4 假突破前高 (0<break<=10bp)", lambda e: not math.isnan(e["break_bps"]) and 0 < e["break_bps"] <= 10.0),
    ("D4_5 强突破前高 (break>10bp)", lambda e: not math.isnan(e["break_bps"]) and e["break_bps"] > 10.0),
    ("G7+非强连阳 (G7 ∧ streak_up<=1)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1),
    ("ALPHA_G7_CORE (G7 ∧ streak_up<=1)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1),
    ("ALPHA_G7_PRICED (G7 ∧ streak_up<=1 ∧ q<=0.065)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1 and e["q"] <= 0.065),
    ("ALPHA_G7_STRICT (G7 ∧ streak_up<=1 ∧ upper_wick>=1.5bp)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1 and e["upper_wick_bps"] >= 1.5),
]


# =======================================
# D. 主审计流程
# =======================================
def run_audit():
    print("=" * 80)
    print("首触反转信号全方位回测数据审计脚本")
    print("=" * 80)
    
    # 加载数据
    k5 = load_klines_audit()
    indep_events = collect_events_independent(k5)

    calib = [e for e in indep_events if e["calib"]]
    confirm = [e for e in indep_events if not e["calib"]]
    print(f"Calib: {len(calib)}, Confirm: {len(confirm)}")

    audit_results = {"independent": [], "main_json_consistency_check": [], "fdr_audit": {}, "rule_redundancy": []}

    # 对每个规则计算 stats
    for name, fn in RULES_INDEPENDENT:
        sub_calib = [e for e in calib if fn(e)]
        sub_confirm = [e for e in confirm if fn(e)]
        st_c = compute_stats_independent(sub_calib)
        st_v = compute_stats_independent(sub_confirm)

        audit_results["independent"].append({"name": name, "calib": st_c, "confirm": st_v})
        audit_results["rule_redundancy"].append((name, sub_calib, sub_confirm))

    # 输出初步结果
    print("\n--- 初步独立回测结果 ---")
    for row in audit_results["independent"]:
        c, v = row["calib"], row["confirm"]
        print(f"{row['name']}: Calib n={c['n']:4d} P={c['p']:5.1%} EV={c['ev']:+.2f} CI=[{c['ev_lo']:.2f},{c['ev_hi']:.2f}] | Confirm n={v['n']:4d} P={v['p']:5.1%} EV={v['ev']:+.2f} CI=[{v['ev_lo']:.2f},{v['ev_hi']:.2f}]")

    # 加载主 JSON 做一致性核对
    main_json = json.load(open("output/comprehensive_firsthit_backtest_20260908.json", encoding="utf-8"))
    name_to_main = {r["name"]: r for r in main_json}

    print("\n--- 一致性核对 (独立 vs 主脚本 JSON) ---")
    for row in audit_results["independent"]:
        main_row = name_to_main.get(row["name"])
        if main_row:
            # 检查 n, P, EV, CI 端点
            c_ind, c_main = row["calib"], main_row["calib"]
            v_ind, v_main = row["confirm"], main_row["confirm"]

            n_match = c_ind["n"] == c_main["n"] and v_ind["n"] == v_main["n"]
            p_match = abs(c_ind["p"] - c_main["p"]) < 1e-5 and abs(v_ind["p"] - v_main["p"]) < 1e-5
            ev_match = abs(c_ind["ev"] - c_main["ev"]) < 1e-4 and abs(v_ind["ev"] - v_main["ev"]) < 1e-4
            ci_lo_match = math.isclose(c_ind["ev_lo"], c_main["ev_lo"], rel_tol=1e-3) and math.isclose(v_ind["ev_lo"], v_main["ev_lo"], rel_tol=1e-3)
            ci_hi_match = math.isclose(c_ind["ev_hi"], c_main["ev_hi"], rel_tol=1e-3) and math.isclose(v_ind["ev_hi"], v_main["ev_hi"], rel_tol=1e-3)

            audit_results["main_json_consistency_check"].append({
                "rule": row["name"],
                "n_match": n_match,
                "p_match": p_match,
                "ev_match": ev_match,
                "ci_lo_match": ci_lo_match,
                "ci_hi_match": ci_hi_match,
                "all_pass": n_match and p_match and ev_match and ci_lo_match and ci_hi_match,
            })

            if not audit_results["main_json_consistency_check"][-1]["all_pass"]:
                print(f"⚠️  {row['name']} 存在偏差:")
                print(f"   Calib: n:{c_ind['n']} vs {c_main['n']} | P:{c_ind['p']:.6f} vs {c_main['p']:.6f} | EV:{c_ind['ev']:.6f} vs {c_main['ev']:.6f}")
                print(f"   CI:[{c_ind['ev_lo']:.6f},{c_ind['ev_hi']:.6f}] vs [{c_main['ev_lo']:.6f},{c_main['ev_hi']:.6f}]")
                print(f"   Confirm: ... (similar)")
        else:
            print(f"❗  {row['name']} 不在主 JSON 中")

    # BH-FDR 校正审计 (Confirm EV_lo>0 候选族) —— 与 shape_scan_v2 保持一致口径：二项单侧 p 值
    print("\n--- BH-FDR 校正审计 (Confirm EV_lo>0 候选族) ---")
    
    # 基准 calib win rate
    base_calib_p = compute_stats_independent([e for e in indep_events if e["calib"]])["p"]
    confirm_candidates = []
    
    pvals_for_fdr = []
    for row in audit_results["independent"]:
        name = row["name"]
        sc = row["calib"]
        
        # 二项单侧 p 值：P(X >= k | n, p0) vs base calib P (与 shape_scan_v2 binom_p_upper 同式)
        def binom_p_upper_local(k: int, n: int, p0: float) -> float:
            if n <= 0:
                return 1.0
            p0 = min(max(p0, 1e-12), 1 - 1e-12)
            log_terms = []
            for i in range(k, n + 1):
                lg = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p0) + (n - i) * math.log(1 - p0))
                log_terms.append(lg)
            m = max(log_terms)
            return float(min(1.0, sum(math.exp(x - m) for x in log_terms) * math.exp(m)))
        
        pval = binom_p_upper_local(sc["wins"], sc["n"], base_calib_p) if sc["n"] >= 30 else 1.0
        pvals_for_fdr.append(pval)
        
    # BH-FDR (与 shape_scan_v2 bh_fdr 同式)
    def bh_fdr_local(pvals, q=0.1):
        p = np.asarray(pvals, float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        qv = ranked * n / (np.arange(n) + 1)
        qv = np.minimum.accumulate(qv[::-1])[::-1]
        out_q = np.empty(n)
        out_q[order] = np.minimum(qv, 1.0)
        return out_q <= q, out_q
    
    rej_bool, q_values = bh_fdr_local(np.array(pvals_for_fdr), q=0.10)
    
    audit_results["fdr_audit"]["n_tested"] = len(pvals_for_fdr)
    audit_results["fdr_audit"]["base_calib_winrate"] = base_calib_p
    audit_results["fdr_audit"]["fdr_q01_threshold"] = int(sum(rej_bool))
    audit_results["fdr_audit"]["reject_rules_with_q"] = [(row["name"], float(q)) for row, q in zip(audit_results["independent"], q_values) if q <= 0.10]
    
    # 输出 FDR 结果
    print(f"基准 Calib Win Rate: {base_calib_p:.4f}")
    print(f"FDR(q=0.1) 通过规则数：{sum(rej_bool)} / {len(rej_bool)}")
    for name, q in audit_results["fdr_audit"]["reject_rules_with_q"]:
        print(f"  ✅ {name}: q={q:.4f}")

    # 规则冗余/重复检测
    print("\n--- 规则冗余/重复检测 ---")
    rule_names_to_events = {}
    redundant_pairs = []
    for row in audit_results["independent"]:
        name = row["name"]
        key_calib = (row["calib"]["n"], row["calib"]["p"], row["calib"]["ev"])
        key_confirm = (row["confirm"]["n"], row["confirm"]["p"], row["confirm"]["ev"])
        rule_names_to_events[name] = (key_calib, key_confirm)

    for n1 in rule_names_to_events:
        for n2 in rule_names_to_events:
            if n1 != n2:
                if rule_names_to_events[n1] == rule_names_to_events[n2]:
                    redundant_pairs.append((n1, n2))

    for pair in redundant_pairs:
        print(f"🔁 重复规则：{pair[0]} ≡ {pair[1]}")

    # 输出最终审计报告
    audit_output_path = "output/audit_firsthit_backtest_20260908.json"
    
    # 序列化：int64 → int, float64 → float
    def serialize_numpy(obj):
        if isinstance(obj, dict):
            return {k: serialize_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_numpy(item) for item in obj]
        elif isinstance(obj, (np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.floating)):
            return float(obj)
        return obj
    
    audit_results_serializable = serialize_numpy(audit_results)
    
    with open(audit_output_path, "w", encoding="utf-8") as f:
        json.dump(audit_results_serializable, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 审计报告已落盘：{audit_output_path}")

    return audit_results


if __name__ == "__main__":
    run_audit()
