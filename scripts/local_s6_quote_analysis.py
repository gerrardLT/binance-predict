#!/usr/bin/env python3
"""S6 定价行为实证：用 4 天 5m 市场 15s 报价采样验证"逐分钟 EV 扫描"假设（2026-08-18）。

数据：prediction_market_samples.json（2026-07-26 ~ 07-30，~15s/条，5m 市场）
     + output/klines_5m_cache_720d.json（结算方向判定，K 线 close vs open）

S6 假设的可检验推论（在 5m 市场上实证）：
  1) 校准曲线：P(win | down_price=q) vs q —— 市场是否系统偏差（edge 来源）
  2) 分钟 EV(t)：周期内按偏移分桶，EV = mean[win×0.98/q − 1] —— 何时入场 EV 最高
  3) 回落确认类比：后半周期价格 < 周期开盘（回落确认）vs 前半 —— S5 逻辑的定价证据
  4) 反应滞后：价格大幅跳动后，报价需要几个 15s 才跟上 —— S6 窗口宽度

注意：采样点在市场内自相关，各桶有效样本按"独立市场数"计。
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

FEE = 0.02
SAMPLES = "prediction_market_samples.json"
KLINES = "output/klines_5m_cache_720d.json"
LOG = "output/s6_quote_analysis.log"


class Tee:
    def __init__(self):
        self.f = open(LOG, "w", encoding="utf-8")
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


def main() -> int:
    sys.stdout = Tee()

    with open(SAMPLES, encoding="utf-8") as f:
        raw = json.load(f)
    with open(KLINES, encoding="utf-8") as f:
        kl5 = json.load(f)
    kmap = {}
    for k in kl5:
        kmap[int(k[0])] = (float(k[1]), float(k[4]))   # open_time -> (open, close)

    # ---------- 样本对齐到 5m 市场周期 ----------
    pts = []
    no_kline = no_quote = 0
    for s in raw:
        ts, q = int(s["timestamp"]), s.get("down_price")
        if q is None or q <= 0.01 or q >= 0.99:
            no_quote += 1
            continue
        cyc = ts // 300_000
        kc = kmap.get(int(cyc) * 300_000)
        if kc is None or kc[0] <= 0:
            no_kline += 1
            continue
        pts.append({
            "ts": ts, "q": float(q), "off": (ts - cyc * 300_000) / 1000.0,  # 秒
            "win": kc[1] < kc[0],            # DOWN 赢 = 周期收盘 < 开盘
            "btc": s.get("btc_price"),
            "open": kc[0], "cyc": int(cyc),
        })
    n_pts = len(pts)
    markets = sorted({p["cyc"] for p in pts})
    t0 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(pts[0]["ts"] / 1000))
    t1 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(pts[-1]["ts"] / 1000))
    btc_cov = sum(1 for p in pts if p["btc"])
    print(f"样本 {len(raw)} 条 → 有效 {n_pts}（弃 {no_quote} 极端报价 / {no_kline} 无K线）")
    print(f"覆盖 {t0} ~ {t1}（UTC），市场周期 {len(markets)} 个，btc_price 覆盖 {btc_cov / n_pts:.0%}")
    base = sum(p["win"] for p in pts) / n_pts
    print(f"DOWN 基准占比 {base:.1%}")

    # ---------- 1) 校准曲线：P(win | q) ----------
    print("\n===== 1) 市场校准：报价 q → 实际 DOWN 概率（有效样本=市场数）=====")
    print("  q 桶      n(市场)  实际P(DOWN)  [q 中位]   偏差(P−q)")
    calib = []
    edges = [round(0.05 + 0.1 * i, 2) for i in range(10)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        grp = [p for p in pts if lo <= p["q"] < hi]
        if not grp:
            continue
        p_hat = sum(p["win"] for p in grp) / len(grp)
        q_mid = float(np.median([p["q"] for p in grp]))
        n_mkt = len({p["cyc"] for p in grp})
        # 每市场取一点的保守估计
        seen: dict[int, list] = {}
        for p in grp:
            seen.setdefault(p["cyc"], []).append(p)
        one = [v[0] for v in seen.values()]
        p_one = sum(p["win"] for p in one) / len(one)
        calib.append({"q": q_mid, "p": p_hat, "p_one": p_one, "n_mkt": n_mkt})
        print(f"  [{lo:.2f},{hi:.2f})  {n_mkt:>5}   {p_hat:7.1%}   {q_mid:5.2f}   {p_hat - q_mid:+.1%}")

    # ---------- 2) 分钟 EV(t) ----------
    print("\n===== 2) 周期内分钟 EV(t)：EV = mean[win×0.98/q − 1]（30s 桶）=====")
    print("  t(分)   n(市场)   EV        平均q    P(DOWN)")
    ev_rows = []
    for b in range(10):
        lo, hi = b * 30.0, (b + 1) * 30.0
        grp = [p for p in pts if lo <= p["off"] < hi]
        if not grp:
            continue
        seen: dict[int, list] = {}
        for p in grp:
            seen.setdefault(p["cyc"], []).append(p)
        ev = sum((0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in grp) / len(grp)
        n_mkt = len(seen)
        qm = float(np.mean([p["q"] for p in grp]))
        pw = sum(p["win"] for p in grp) / len(grp)
        # 保守标准误：按独立市场重采样
        by_mkt = [[(0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in v] for v in seen.values()]
        m_per = [float(np.mean(v)) for v in by_mkt]
        se = float(np.std(m_per) / math.sqrt(n_mkt)) if n_mkt > 1 else float("nan")
        ev_rows.append({"t_min": hi / 60.0, "ev": ev, "se": se, "n_mkt": n_mkt, "q": qm, "p": pw})
        print(f"  {lo / 60:4.1f}-{hi / 60:4.1f}  {n_mkt:>5}  {ev:+7.3f}±{1.96 * se:.3f}  {qm:5.2f}  {pw:5.1%}")

    # ---------- 3) 回落确认类比（btc_price 非空段）----------
    btc_pts = [p for p in pts if p["btc"]]
    print(f"\n===== 3) 回落确认定价：btc_price 覆盖段（n={len(btc_pts)}）=====")
    if btc_pts:
        for half, tag in ((False, "前半周期(off<150s)"), (True, "后半周期(off≥150s)")):
            for state, stag in ((True, "价<开盘(回落)"), (False, "价≥开盘")):
                grp = [p for p in btc_pts
                       if ((p["off"] >= 150) == half)
                       and ((p["btc"] < p["open"]) == state)]
                if not grp:
                    continue
                ev = sum((0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in grp) / len(grp)
                pw = sum(p["win"] for p in grp) / len(grp)
                qm = float(np.mean([p["q"] for p in grp]))
                nm = len({p["cyc"] for p in grp})
                print(f"  {tag} × {stag}: n={len(grp):>5}（市场 {nm:>3}）P(DOWN)={pw:5.1%} q̄={qm:5.2f} EV={ev:+.3f}")

    # ---------- 4) 反应滞后：报价对价格跳动的跟随速度 ----------
    print(f"\n===== 4) 反应滞后：|Δbtc|≥15bp 的 15s 步后，DOWN 报价的跟随 =====")
    # 按周期内顺序找连续采样对（同市场内 15s 相邻）
    by_mkt: dict[int, list] = {}
    for p in btc_pts:
        by_mkt.setdefault(p["cyc"], []).append(p)
    lag_steps = {1: [], 2: [], 3: [], 4: []}   # k 步后的报价变化
    imm = []
    for cyc, arr in by_mkt.items():
        arr.sort(key=lambda p: p["ts"])
        for i in range(len(arr) - 4):
            dp = arr[i + 1]["btc"] / arr[i]["btc"] - 1.0
            if abs(dp) < 0.0015:
                continue
            down_move = dp < 0   # 价跌 → DOWN 概率应升 → q 升
            for k in lag_steps:
                dq = arr[i + 1 + k]["q"] - arr[i + 1]["q"]
                lag_steps[k].append(dq if down_move else -dq)   # 方向对齐：正=正确方向跟随
            dq0 = arr[i + 1]["q"] - arr[i]["q"]
            imm.append(dq0 if down_move else -dq0)
    if imm:
        print(f"  触发步数 n={len(imm)}")
        print(f"  同步(0步):  Δq(方向对齐) 均值 {float(np.mean(imm)):+.4f}  中位 {float(np.median(imm)):+.4f}")
        for k, vals in lag_steps.items():
            if vals:
                print(f"  +{k * 15}s 后累计:  均值 {float(np.mean(vals)):+.4f}  中位 {float(np.median(vals)):+.4f}")

    # ---------- 结论 ----------
    print("\n===== 要点 =====")
    best_row = max(ev_rows, key=lambda r: r["ev"])
    print(f"  EV(t) 最优桶：t∈[{best_row['t_min'] - 0.5:.1f},{best_row['t_min']:.1f}] 分 "
          f"EV={best_row['ev']:+.3f}±{1.96 * best_row['se']:.3f}（n 市场={best_row['n_mkt']}）")
    print(f"  EV(t) 最差桶：EV={min(r['ev'] for r in ev_rows):+.3f}")
    mid = [r for r in ev_rows if 2.0 <= r["t_min"] <= 4.5]
    early = [r for r in ev_rows if r["t_min"] <= 1.5]
    if mid and early:
        print(f"  中后段(2~4.5分) EV均值 {float(np.mean([r['ev'] for r in mid])):+.3f}"
              f" vs 早期(≤1.5分) {float(np.mean([r['ev'] for r in early])):+.3f}")

    with open("output/s6_quote_analysis_result.json", "w", encoding="utf-8") as f:
        json.dump({"calib": calib, "ev_rows": ev_rows}, f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s6_quote_analysis_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
