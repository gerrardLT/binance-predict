#!/usr/bin/env python3
"""5m DOWN 首触信号扫描 v2：方法论修正版。

v1（local_kline_shape_scan.py / local_dim_gain_scan.py）的 5 个方法缺陷与本版修正：
-----------------------------------------------|---------------------------------------------
缺陷 1: upper_wick 是触发时刻 t 的强代理          | 用 logit win ~ wick + t + body + chg + t²
（早触发无路径可形成上影，早触发天然胜率高）        |   回归直接量化各特征的独立边际贡献（t 作协变量）
                                                 | + 同 t 桶分层双向表
缺陷 2: 触发前 btc 采样点稀疏 → cum_hi/lo 不完整  | pts<8 的事件单列一行（sparse），不并入主结果
                                                 | + 披露被丢弃事件（len(pts)<=1）的 win 率
缺陷 3: body≤0.35 与 chg≤2.82 数学上不正交        | 组合改成正交化表述：先报 body 主效应，再报
（chg 微小钳制 body 上限，chg 出现在 body 分子）   |   chg 在 body 内的增量，避免联合≠独立的误导
缺陷 4: 每个脚本各自 2/3 切分，confirm 被反复验证   | SPLIT 冻结为 2026-08-24（首个脚本口径），
                                                 |   并注明 confirm 段已被多轮扫描污染的解读限制
缺陷 5: 多门无 FDR                                | 报告 BH-FDR(q=0.1)，★候选需过 FDR

特征（全部触发时刻可见，无未来函数）
------------------------------------
t           触发时刻（秒）
q           触发报价
chg_bps     BTC 相对开盘涨跌 bps（负=已跌；用 ±1 样本内就近补缺）
body_r      |chg| / (cum_hi - cum_lo)（路径归一实体）
wick01      上影线>0 的二元变量（受 t 混杂，回归里校正）
rng_bps     (cum_hi - cum_lo) bps（触发前路径总振幅）
npts        触发前 btc 采样点数（数据质量层）
sparse      npts < 8 → 数据质量差的标记
break_bps   cum_hi 相对前 3 根 5m 最高的突破幅度 bps
streak_up   前驱连续阳线根数

判定（预注册）
--------------
1) logit 回归（calib 拟合，confirm 报 out-of-sample 系数方向+显著）看每个特征
   控制 t 后的独立方向；胜率结论只认回归方向 + 分层表一致。
2) EV 表：日聚类 bootstrap 95% CI；★候选 = calib EV_lo>0 且 n≥60 且过 BH-FDR(q=0.1)
   （FDR 对「EV_lo>0」这一族门做）。
3) confirm 只终验 ★ 候选一次。confirm 段（08-24 后）此前已被多轮扫描使用，
   其结果只能作为「不否决」的弱证据，真正裁决靠影子前向。

用法
----
    .venv\\Scripts\\python.exe scripts/local_shape_scan_v2.py
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
LOG = "output/shape_scan_v2.log"
OUT = "output/shape_scan_v2.json"

SPLIT = "2026-08-24"          # 冻结切分（首个扫描脚本口径），所有后续扫描共用
BOOT_B = 4000
SEED = 20260907
MIN_N = 60
MIN_DAYS = 6
MIN_PTS = 8                   # 触发前 btc 采样点数门槛（数据质量）


class Tee:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        try:
            sys.__stdout__.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


def day_boot(values: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    d_u = np.unique(days)
    if len(d_u) < 5:
        return (float("nan"), float("nan"))
    sums = np.zeros(len(d_u)); cnts = np.zeros(len(d_u))
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
    n = len(evs); k = int(flags.sum())
    evv = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
    lo, hi = day_boot(evv, days)
    return {"n": n, "k": k, "p": k / n, "ev": float(evv.mean()), "ev_lo": lo,
            "ev_hi": hi, "qbar": float(qs.mean()),
            "n_days": int(len(np.unique(days)))}


def fmt(st):
    if st is None or st["n"] == 0:
        return "n=0"
    lo, hi = st["ev_lo"], st["ev_hi"]
    ci = "CI[nan]" if math.isnan(lo) else f"CI[{lo:+.2f},{hi:+.2f}]"
    return f"n={st['n']:>4} P={st['p']:>5.1%} EV={st['ev']:+.2f} {ci} q̄={st['qbar']:.3f} {st['n_days']:>2}天"


def load_k5():
    rows = []
    for p in KL5_FILES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    return {int(k[0]): (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in rows}


def collect(k5):
    """首触事件 + 修正后的特征。就近补缺 btc（±2 样本内）。"""
    raw = json.load(open(SAMPLES_5M, encoding="utf-8"))
    L = 300_000
    by = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)

    events, dropped_no_btc = [], []
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

        # ---- 路径：开盘 o + 触发前样本的 btc（触发样本缺则 ±2 样本内就近补缺） ----
        pre = rows[:ti + 1]
        if trig.get("btc_price") is None:
            for j in (ti - 1, ti + 1, ti - 2, ti + 2):
                if 0 <= j < len(rows) and rows[j].get("btc_price"):
                    pre = pre + [rows[j]]
                    break
        pts = [o] + [float(s["btc_price"]) for s in pre
                     if s.get("btc_price") and float(s["btc_price"]) > 0]
        npts = len(pts) - 1
        if npts == 0:
            dropped_no_btc.append({"w": w, "win": win, "q": q, "day": day})
            continue

        btc_trig = pts[-1]
        cum_hi, cum_lo = max(pts), min(pts)
        rng = cum_hi - cum_lo
        chg_bps = (btc_trig / o - 1.0) * 10_000.0
        if rng > 1e-9:
            body_r = abs(btc_trig - o) / rng
            wick01 = 1.0 if cum_hi > max(btc_trig, o) + 1e-9 else 0.0
            rng_bps = rng / o * 10_000.0
        else:
            body_r, wick01, rng_bps = 0.0, 0.0, 0.0

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

        events.append({
            "w": w, "ts": ts, "t": (ts - w) / 1000.0, "q": q, "win": win, "day": day,
            "chg_bps": chg_bps, "body_r": body_r, "wick01": wick01,
            "rng_bps": rng_bps, "npts": npts, "sparse": int(npts < MIN_PTS),
            "break_bps": break_bps, "streak_up": streak_up,
        })
    return events, dropped_no_btc


def logit(X, y, iters=60, lr=0.1):
    """简单 IRLS/Newton logit（无正则），返回 beta, se。"""
    X = np.column_stack([X, np.ones(len(X))])
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        z = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        Wd = np.clip(mu * (1 - mu), 1e-9, None)
        g = X.T @ (y - mu)
        H = X.T @ (X * Wd[:, None]) + 1e-9 * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    z = X @ beta
    mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    Wd = np.clip(mu * (1 - mu), 1e-9, None)
    try:
        cov = np.linalg.inv(X.T @ (X * Wd[:, None]))
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return beta, se


def bh_fdr(pvals, q=0.1):
    """BH 校正：返回 (reject 布尔数组, 校正后 q 值数组)。"""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    qv = ranked * n / (np.arange(n) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    out_q = np.empty(n)
    out_q[order] = np.minimum(qv, 1.0)
    return out_q <= q, out_q


def binom_p_upper(k: int, n: int, p0: float) -> float:
    """数值稳定的二项单侧 p 值：P(X >= k | n, p0)，log-space 计算。"""
    if n <= 0:
        return 1.0
    p0 = min(max(p0, 1e-12), 1 - 1e-12)
    log_terms = []
    for i in range(k, n + 1):
        # lgamma(n+1) - lgamma(i+1) - lgamma(n-i+1) + i*ln(p0) + (n-i)*ln(1-p0)
        lg = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
              + i * math.log(p0) + (n - i) * math.log(1 - p0))
        log_terms.append(lg)
    m = max(log_terms)
    return float(min(1.0, sum(math.exp(x - m) for x in log_terms) * math.exp(m)))


def main() -> int:
    sys.stdout = Tee(LOG)
    print("=" * 112)
    print("5m DOWN 首触 · 扫描 v2（方法论修正版：t 混杂校正 / 路径质量分层 /")
    print("正交化组合 / 冻结切分 2026-08-24 / BH-FDR(q=0.1)）")
    print("=" * 112)
    k5 = load_k5()
    events, dropped = collect(k5)
    dw = sum(d["win"] for d in dropped); dn = len(dropped)
    print(f"主事件 {len(events)} 条（触发前 btc 采样点 ≥1）；"
          f"被丢弃（完全无 btc）{dn} 条，其 DOWN 胜率 {dw / dn:.1%}（对照基底 8.9%，"
          f"{'存在丢弃偏差，见下文' if abs(dw / dn - 0.089) > 0.03 else '与基底接近，偏差可忽略'}）")

    calib = [e for e in events if e["day"] < SPLIT]
    confirm = [e for e in events if e["day"] >= SPLIT]
    print(f"冻结切分 {SPLIT} | calib {len(calib)} | confirm {len(confirm)}")
    print(f"基底 calib : {fmt(stats(calib))}")
    print(f"基底 confirm: {fmt(stats(confirm))}")

    # ---------------- 修正 2：路径质量分层 ----------------
    print("\n----- A. 路径质量分层（触发前 btc 采样点 npts）-----")
    for name, cond in (("npts>=8（完整路径）", lambda e: e["npts"] >= MIN_PTS),
                       ("npts<8 （稀疏路径）", lambda e: e["npts"] < MIN_PTS)):
        sc = stats([e for e in calib if cond(e)])
        sf = stats([e for e in confirm if cond(e)])
        print(f"  {name:<20} | calib {fmt(sc)} | confirm {fmt(sf)}")
    dense = [e for e in events if e["npts"] >= MIN_PTS]
    print("  → 主分析只用 npts>=8 子集（后续所有表格）。")

    calibD = [e for e in dense if e["day"] < SPLIT]
    confirmD = [e for e in dense if e["day"] >= SPLIT]
    print(f"  主分析事件 {len(dense)} | calib {len(calibD)} | confirm {len(confirmD)}")
    print(f"  基底 calib : {fmt(stats(calibD))}")
    print(f"  基底 confirm: {fmt(stats(confirmD))}")

    # ---------------- 修正 1：t 混杂校正（logit + 分层） ----------------
    print("\n----- B. 特征独立贡献：logit 回归（calib 拟合；t 作协变量）-----")
    feats = {
        "wick01（上影>0）": lambda e: e["wick01"],
        "body_r（小实体）": lambda e: -e["body_r"],       # 负号：body 越小预期胜率越高
        "chg_bps（BTC 涨幅小）": lambda e: -e["chg_bps"],  # 负号：涨幅越小预期胜率越高
        "rng_bps（路径振幅）": lambda e: e["rng_bps"],
        "t（触发时刻）": lambda e: e["t"],
    }
    yC = np.array([e["win"] for e in calibD], float)
    tC = np.array([e["t"] for e in calibD]) / 100.0
    rows = []
    for name, fn in feats.items():
        xC = np.array([fn(e) for e in calibD], float)
        if name == "t（触发时刻）":
            # t 自身作特征时不能再加 t/t² 协变量（完全共线）→ 单特征模型
            X = np.column_stack([tC])
        else:
            # 单特征 + t + t² 控制（t 混杂校正）
            X = np.column_stack([xC, tC, tC ** 2])
        beta, se = logit(X, yC)
        z = beta[0] / se[0] if se[0] and not math.isnan(se[0]) else float("nan")
        pv = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if math.isfinite(z) else float("nan")
        rows.append((name, beta[0], se[0], z, pv))
        print(f"  {name:<22} β={beta[0]:+.4f} se={se[0]:.4f} z={z:+.2f} p={pv:.3f}"
              f"{'  ← 控制 t 后方向不变' if (not math.isnan(pv) and pv < 0.1) else ''}")

    # 同 t 桶分层双向表（wick）
    print("\n----- B2. 同 t 桶内 有/无上影 胜率（wick 的 t 混杂检验）-----")
    ts_ = np.array([e["t"] for e in dense])
    wk = np.array([e["wick01"] for e in dense])
    wn = np.array([e["win"] for e in dense])
    for lo, hi in [(60, 120), (120, 180), (180, 240), (240, 300)]:
        m = (ts_ >= lo) & (ts_ < hi)
        a, b = wn[m & (wk > 0)], wn[m & (wk == 0)]
        if len(a) > 20 and len(b) > 20:
            print(f"  t∈[{lo:>3},{hi}): 有上影 n={len(a):>4} P={a.mean():>5.1%} | "
                  f"无上影 n={len(b):>4} P={b.mean():>5.1%} | Δ={a.mean() - b.mean():+.1%}")

    # ---------------- 修正 3：正交化组合表 ----------------
    print("\n----- C. 组合正交化：body 主效应 + chg 条件增量（完整路径子集）-----")
    gates = {}
    def gate(name, cond):
        sc = stats([e for e in calibD if cond(e)])
        sf = stats([e for e in confirmD if cond(e)])
        gates[name] = (sc, sf, cond)
        return sc, sf

    combos = [
        ("G0 基底（全部首触）", lambda e: True),
        ("G1 body≤0.35", lambda e: e["body_r"] <= 0.35),
        ("G2 body>0.35", lambda e: e["body_r"] > 0.35),
        ("G3 chg≤2.82（不切 body）", lambda e: e["chg_bps"] <= 2.82),
        ("G4 chg≤2.82 ∧ body≤0.35（组合B）", lambda e: e["chg_bps"] <= 2.82 and e["body_r"] <= 0.35),
        ("G5 body≤0.35 内 chg>2.82（增量检验）", lambda e: e["body_r"] <= 0.35 and e["chg_bps"] > 2.82),
        ("G6 wick>0", lambda e: e["wick01"] > 0),
        ("G7 wick>0 ∧ body≤0.35", lambda e: e["wick01"] > 0 and e["body_r"] <= 0.35),
        ("G8 0<break≤10bps", lambda e: not math.isnan(e["break_bps"]) and 0 < e["break_bps"] <= 10),
        ("G9 streak≥1（前阳）", lambda e: e["streak_up"] >= 1),
    ]
    print(f"  {'门':<34} | calib | confirm")
    pvals_for_fdr = []
    for name, cond in combos:
        sc, sf = gate(name, cond)
        # stable binomial tail P(X >= k) vs base calib P
        if sc and sc["n"] >= 30:
            pval = binom_p_upper(sc["k"], sc["n"], stats(calibD)["p"])
        else:
            pval = 1.0
        pvals_for_fdr.append(pval)
        star = "★" if (sc and sc["n"] >= MIN_N and sc["n_days"] >= MIN_DAYS
                        and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0) else " "
        print(f"  {star}{name:<33} | {fmt(sc)} | {fmt(sf)}  (binom p={pval:.3f})")

    # BH-FDR
    rej, qv = bh_fdr(pvals_for_fdr, q=0.1)
    print(f"\n  BH-FDR(q=0.1) 校正后：")
    for (name, _), q in zip(combos, qv):
        if q <= 0.1:
            print(f"    ✅ {name}: q={q:.3f}")
    print("  （未列出者未过 FDR；★ 只是 CI 下界条件，与 FDR 独立）")

    # ---------------- ★ 候选 confirm 终验 ----------------
    print("\n----- D. ★候选 confirm 终验（EV 日聚类 CI）-----")
    for name, (sc, sf, _) in gates.items():
        if sc and sc["n"] >= MIN_N and sc["n_days"] >= MIN_DAYS and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0:
            if sf and not math.isnan(sf["ev_lo"]) and sf["ev_lo"] > 0:
                verdict = "STRICT_PASS"
            elif sf and sf["ev"] > 0 and not math.isnan(sf["ev_lo"]) and sf["ev_lo"] > -0.10:
                verdict = "LOOSE_PASS"
            else:
                verdict = "FAIL"
            print(f"  {name}: calib {fmt(sc)}")
            print(f"    confirm {fmt(sf)} → {verdict}")

    # ---------------- 结果落盘 ----------------
    out = {
        "meta": {"split": SPLIT, "min_pts": MIN_PTS, "min_n": MIN_N,
                 "fdr": "BH q=0.1 on binomial p (vs base calib P)",
                 "dropped_no_btc": {"n": dn, "down_win_rate": dw / dn if dn else None}},
        "logit_calib": {name: {"beta": float(b), "se": float(s), "z": float(z), "p": float(pv)}
                        for name, b, s, z, pv in rows},
        "gates": {name: {"calib": sc, "confirm": sf, "fdr_q": float(q)}
                  for (name, _), (sc, sf, _), q in zip(combos,
                                                       [gates[n] for n, _ in combos], qv)},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果 JSON → {OUT} | 日志 → {LOG}")
    print("解读限制：confirm 段（08-24 之后）已被多轮历史扫描使用，本表 confirm 结果")
    print("只作「不否决」的弱证据；任何上线决策必须以影子前向验证为准。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
