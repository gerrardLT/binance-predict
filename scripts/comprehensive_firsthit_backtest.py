#!/usr/bin/env python3
"""全方位科学假设回测脚本：5m DOWN 首触反转信号多维优化评测。

覆盖 5 大科学假设维度：
  维度 1 (形态拒止与流动性掠夺):
    - G0: 基底
    - G1: body_r <= 0.35 (小实体)
    - G3: chg_bps <= 2.82 (位移小)
    - G6: wick01 == 1 (上影线拒绝)
    - G7: body_r <= 0.35 ∧ wick01 == 1 (小实体+上影拒绝)
    - G7_deep: G7 且 上影长度 upper_wick_bps >= 2.0
  维度 2 (报价深度与折价不对称):
    - 报价细分子带: q <= 0.03, q <= 0.05, q <= 0.07, q in (0.07, 0.10]
    - G1 叠加 q <= 0.05
    - G7 叠加 q <= 0.05
  维度 3 (时间与扩散边界):
    - 触发时刻截断: t <= 150s, t <= 180s, t <= 210s, 晚触发 t > 210s
    - G7 叠加 t <= 210s
  维度 4 (高阶大势与 Regime 抑制):
    - 前驱无连阳 (streak_up == 0) vs 连阳 (streak_up >= 1, streak_up >= 2)
    - 破前高幅度 break_bps <= 0 (未创新高) vs 假突破 (0 < break_bps <= 10) vs 强突破 (break_bps > 10)
    - G7 叠加 streak_up <= 1
  维度 5 (微观参与度与量能背离):
    - 参与人数增量 dpar 与成交量增量 dvol 变化
    - 参与人数低增长 dpar_ratio <= 中位数 (散户跟风衰竭)

评估规范：
  - 严格切分: 2026-08-24 之前为 Calib (标定集), 之后为 Confirm (盲测验证集)
  - 统计量: 事件数 N, 获胜率 Win%, 均值单注 EV, 日聚类 Bootstrap 95% CI (4000 次重抽样), 平均入场价 qbar
  - 多重检验: 全规则候选经过 Benjamini-Hochberg FDR (q=0.10 & q=0.05)
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
Q_LO, Q_HI = 0.005, 0.1
SAMPLES_5M = "output/pull_samples_5m_20260907.json"
KL5_FILES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
SPLIT = "2026-08-24"
BOOT_B = 4000
SEED = 20260908
MIN_PTS = 8


def day_boot(values: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    d_u = np.unique(days)
    if len(d_u) < 5:
        return (float("nan"), float("nan"))
    sums = np.zeros(len(d_u))
    cnts = np.zeros(len(d_u))
    idx = np.searchsorted(d_u, days)
    np.add.at(sums, idx, values)
    np.add.at(cnts, idx, 1)
    means = sums / cnts
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(d_u), size=(BOOT_B, len(d_u)))
    w = cnts[pick]
    tot = (means[pick] * w).sum(axis=1) / w.sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def stats(evs):
    if not evs:
        return None
    flags = np.array([e["win"] for e in evs], float)
    qs = np.array([e["q"] for e in evs], float)
    days = np.array([e["day"] for e in evs])
    n = len(evs)
    k = int(flags.sum())
    evv = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
    lo, hi = day_boot(evv, days)
    return {
        "n": n,
        "k": k,
        "p": k / n if n > 0 else 0.0,
        "ev": float(evv.mean()) if n > 0 else 0.0,
        "ev_lo": lo,
        "ev_hi": hi,
        "qbar": float(qs.mean()) if n > 0 else 0.0,
        "n_days": int(len(np.unique(days))),
    }


def fmt_st(st):
    if st is None or st["n"] == 0:
        return "n=0"
    lo, hi = st["ev_lo"], st["ev_hi"]
    ci = "CI[nan]" if math.isnan(lo) else f"CI[{lo:+.2f},{hi:+.2f}]"
    return f"n={st['n']:>4} P={st['p']:>5.1%} EV={st['ev']:+.2f} {ci} q_bar={st['qbar']:.3f} {st['n_days']:>2}d"


def load_k5():
    rows = []
    for p in KL5_FILES:
        try:
            rows.extend(json.load(open(p, encoding="utf-8")))
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return {int(k[0]): (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in rows}


def collect_events(k5):
    print("Loading 5m samples...")
    raw = json.load(open(SAMPLES_5M, encoding="utf-8"))
    L = 300_000
    by = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)

    events = []
    for w, rows in sorted(by.items()):
        bar = k5.get(w)
        if not bar or bar[0] <= 0:
            continue
        o = bar[0]
        if bar[3] == o:
            continue
        win = 1 if bar[3] < o else 0
        rows.sort(key=lambda s: int(s["timestamp"]))

        trig = ti = None
        for i, s in enumerate(rows):
            q = s.get("down_price")
            if q is not None and Q_LO < float(q) <= Q_HI:
                trig, ti = s, i
                break
        if trig is None:
            continue

        ts = int(trig["timestamp"])
        q = float(trig["down_price"])
        day = time.strftime("%Y-%m-%d", time.gmtime(w / 1000))

        pre = rows[: ti + 1]
        if trig.get("btc_price") is None:
            for j in (ti - 1, ti + 1, ti - 2, ti + 2):
                if 0 <= j < len(rows) and rows[j].get("btc_price"):
                    pre = pre + [rows[j]]
                    break
        pts = [o] + [float(s["btc_price"]) for s in pre if s.get("btc_price") and float(s["btc_price"]) > 0]
        npts = len(pts) - 1
        if npts < MIN_PTS:
            continue

        btc_trig = pts[-1]
        cum_hi, cum_lo = max(pts), min(pts)
        rng = cum_hi - cum_lo
        chg_bps = (btc_trig / o - 1.0) * 10_000.0
        if rng > 1e-9:
            body_r = abs(btc_trig - o) / rng
            wick01 = 1.0 if cum_hi > max(btc_trig, o) + 1e-9 else 0.0
            upper_wick_bps = (cum_hi - max(btc_trig, o)) / o * 10_000.0
            rng_bps = rng / o * 10_000.0
        else:
            body_r, wick01, upper_wick_bps, rng_bps = 0.0, 0.0, 0.0, 0.0

        prev_his = [k5.get(w - j * L, (None,) * 4)[1] for j in (1, 2, 3)]
        if all(x is not None for x in prev_his) and max(prev_his) > 0:
            break_bps = (cum_hi / max(prev_his) - 1.0) * 10_000.0
        else:
            break_bps = float("nan")

        streak_up = 0
        cur = w - L
        while streak_up < 4:
            pb = k5.get(cur)
            if pb and pb[3] > pb[0]:
                streak_up += 1
                cur -= L
            else:
                break

        # 参与度与成交变化
        par_o = float(rows[0]["participants"]) if rows[0].get("participants") is not None else None
        par_t = float(trig["participants"]) if trig.get("participants") is not None else None
        vol_o = float(rows[0]["trade_volume"]) if rows[0].get("trade_volume") is not None else None
        vol_t = float(trig["trade_volume"]) if trig.get("trade_volume") is not None else None
        dpar = (par_t - par_o) if (par_o is not None and par_t is not None) else 0.0
        dvol = (vol_t - vol_o) if (vol_o is not None and vol_t is not None) else 0.0

        t_sec = (ts - w) / 1000.0

        events.append({
            "w": w,
            "ts": ts,
            "t": t_sec,
            "q": q,
            "win": win,
            "day": day,
            "chg_bps": chg_bps,
            "body_r": body_r,
            "wick01": wick01,
            "upper_wick_bps": upper_wick_bps,
            "rng_bps": rng_bps,
            "break_bps": break_bps,
            "streak_up": streak_up,
            "dpar": dpar,
            "dvol": dvol,
            "calib": day < SPLIT,
        })
    print(f"Total valid events (npts>={MIN_PTS}): {len(events)}")
    return events


def run_comprehensive_backtest():
    k5 = load_k5()
    events = collect_events(k5)

    rules = [
        # --- 基底与现有线上版本 ---
        ("G0 基底 (全部首触)", lambda e: True),
        ("G1 小实体 (body_r<=0.35)", lambda e: e["body_r"] <= 0.35),
        ("G3 位移小 (chg_bps<=2.82)", lambda e: e["chg_bps"] <= 2.82),

        # --- 维度 1: 形态拒止与衰竭 ---
        ("D1_1 上影拒绝 (wick01==1)", lambda e: e["wick01"] == 1.0),
        ("D1_2 较长上影 (upper_wick>=1.5bp)", lambda e: e["upper_wick_bps"] >= 1.5),
        ("D1_3 显著上影 (upper_wick>=3.0bp)", lambda e: e["upper_wick_bps"] >= 3.0),
        ("G7 组合 (body<=0.35 ∧ wick==1)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0),
        ("G7+长上影 (body<=0.35 ∧ upper_wick>=2.0bp)", lambda e: e["body_r"] <= 0.35 and e["upper_wick_bps"] >= 2.0),

        # --- 维度 2: 报价深度与折价不对称 ---
        ("D2_1 超深折价 (q<=0.03)", lambda e: e["q"] <= 0.03),
        ("D2_2 深度折价 (q<=0.05)", lambda e: e["q"] <= 0.05),
        ("D2_3 中度折价 (0.05<q<=0.08)", lambda e: 0.05 < e["q"] <= 0.08),
        ("D2_4 浅折价 (0.08<q<=0.10)", lambda e: 0.08 < e["q"] <= 0.10),
        ("G1+深折价 (body<=0.35 ∧ q<=0.05)", lambda e: e["body_r"] <= 0.35 and e["q"] <= 0.05),
        ("G7+深折价 (G7 ∧ q<=0.05)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["q"] <= 0.05),

        # --- 维度 3: 时间与均值回归扩散 ---
        ("D3_1 早触发 (t<=180s)", lambda e: e["t"] <= 180.0),
        ("D3_2 中期触发 (180<t<=250s)", lambda e: 180.0 < e["t"] <= 250.0),
        ("D3_3 尾声前触发 (t<=270s)", lambda e: e["t"] <= 270.0),
        ("D3_4 极限尾声 (t>270s)", lambda e: e["t"] > 270.0),
        ("G7+非极晚触发 (G7 ∧ t<=270s)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["t"] <= 270.0),
        ("G7+极晚触发 (G7 ∧ t>270s)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["t"] > 270.0),

        # --- 维度 4: 高阶大势与 Regime 抑制 ---
        ("D4_1 前驱无连阳 (streak_up==0)", lambda e: e["streak_up"] == 0),
        ("D4_2 前驱微阳 (streak_up<=1)", lambda e: e["streak_up"] <= 1),
        ("D4_3 前驱强连阳 (streak_up>=2)", lambda e: e["streak_up"] >= 2),
        ("D4_4 假突破前高 (0<break<=10bp)", lambda e: not math.isnan(e["break_bps"]) and 0 < e["break_bps"] <= 10.0),
        ("D4_5 强突破前高 (break>10bp)", lambda e: not math.isnan(e["break_bps"]) and e["break_bps"] > 10.0),
        ("G7+非强连阳 (G7 ∧ streak_up<=1)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1),
        ("G7+假突破 (G7 ∧ 0<break<=10bp)", lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and not math.isnan(e["break_bps"]) and 0 < e["break_bps"] <= 10.0),

        # --- 终极融合方案 ---
        ("ALPHA_G7_CORE (G7 ∧ streak_up<=1)", 
         lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1),
        ("ALPHA_G7_PRICED (G7 ∧ streak_up<=1 ∧ q<=0.065)", 
         lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1 and e["q"] <= 0.065),
        ("ALPHA_G7_STRICT (G7 ∧ streak_up<=1 ∧ upper_wick>=1.5bp)", 
         lambda e: e["body_r"] <= 0.35 and e["wick01"] == 1.0 and e["streak_up"] <= 1 and e["upper_wick_bps"] >= 1.5),
    ]
    
    # === BH-FDR 相关函数 (与 shape_scan_v2 口径一致) ===
    def binom_p_upper_local(k, n, p0):
        """二项单侧 p 值 P(X >= k | n, p0)，log-space 计算。"""
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
    
    def bh_fdr_local(pvals, q=0.1):
        """BH-FDR 校正。"""
        p = np.asarray(pvals, float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        qv = ranked * n / (np.arange(n) + 1)
        qv = np.minimum.accumulate(qv[::-1])[::-1]
        out_q = np.empty(n)
        out_q[order] = np.minimum(qv, 1.0)
        return out_q <= q, out_q
    
    calib_ev = [e for e in events if e["calib"]]
    confirm_ev = [e for e in events if not e["calib"]]

    print(f"\n样本切分: Calib = {len(calib_ev)} ({SPLIT} 之前), Confirm = {len(confirm_ev)} ({SPLIT} 之后)\n")

    # 基准 calib 胜率（用于二项检验 p0）
    base_calib_stats = stats(calib_ev)
    base_calib_p = base_calib_stats["p"]

    results = []
    pvals_for_fdr = []
    for name, fn in rules:
        c_sub = [e for e in calib_ev if fn(e)]
        v_sub = [e for e in confirm_ev if fn(e)]
        st_c = stats(c_sub)
        st_v = stats(v_sub)
        # 二项单侧 p 值: H0: P=base_calib_p, H1: P>base_calib_p (calib 段)
        if st_c and st_c["n"] >= 30:
            pval = binom_p_upper_local(st_c["k"], st_c["n"], base_calib_p)
        else:
            pval = 1.0
        pvals_for_fdr.append(pval)
        results.append({
            "name": name,
            "calib": st_c,
            "confirm": st_v,
            "binom_p_calib": pval,
        })

    # BH-FDR 校正
    rej_bool, q_values = bh_fdr_local(np.array(pvals_for_fdr), q=0.10)
    for row, qv, rej in zip(results, q_values, rej_bool):
        row["fdr_q"] = float(qv)
        row["fdr_pass_q010"] = bool(rej)

    print(f"基准 Calib Win Rate (p0) = {base_calib_p:.4f}")
    print(f"BH-FDR(q=0.10) 通过规则数: {int(sum(rej_bool))} / {len(rules)}\n")
    print(f"{'规则名称':<40} | {'Calib 表现':<44} | {'Confirm 表现 (弱证据)':<44} | FDR-q | 通过")
    print("-" * 168)
    for row in results:
        st_c, st_v = row["calib"], row["confirm"]
        flag = "✅" if row["fdr_pass_q010"] else "  "
        print(f"{row['name']:<40} | {fmt_st(st_c):<44} | {fmt_st(st_v):<44} | {row['fdr_q']:.4f} | {flag}")

    # 保存 JSON 产物
    out_path = "output/comprehensive_firsthit_backtest_20260908.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n回测产物已落盘至: {out_path}")
    print("⚠️ Confirm 段 (2026-08-24 之后) 已被多轮历史扫描使用 (burned)，只作「不否决」弱证据；")
    print("   任何上线决策必须以影子前向验证为准。FDR 基于 calib 段二项单侧检验 (vs base win rate)。")


if __name__ == "__main__":
    run_comprehensive_backtest()
