#!/usr/bin/env python3
"""S6 定价行为实证（15m 市场版）：线上 5 天 15s 报价 × 720d 5m K 缓存（2026-08-18）。

数据：output/online_15m_samples_full.json（线上 prediction_market_samples，15m 市场，
     08-13 08:43 ~ 08-18 06:36 UTC，26,098 条，btc_price 100% 覆盖）
     + output/klines_5m_cache_720d.json（聚合 15m 判结算：close vs open）

在 15m 市场上直接检验 S6 假设（S5 的真实交易市场）：
  1) 校准曲线：P(win | q) vs q
  2) 分钟 EV(t)：周期内逐分钟 EV（60s 桶）
  3) 三段 × 回落态定价：前/中/后 1/3 周期 × 价<开盘——直接对应 S5 的 +5min 确认窗口
  4) 反应滞后：价格跳变后报价的跟随速度（btc_price 全覆盖）
  5) S1 事件次周期的报价路径实录（8/17 后 bull_exhaust 信号）
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

SAMPLES = "output/online_15m_samples_full.json"
KLINES = "output/klines_5m_cache_720d.json"
LOG = "output/s6_quote_analysis_15m.log"


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
    # 5m K 聚合 15m：cyc → (open, close)
    agg: dict[int, tuple[float, float]] = {}
    byc: dict[int, list[list]] = {}
    for k in kl5:
        byc.setdefault(int(k[0]) // 900_000, []).append(k)
    for cyc, arr in byc.items():
        if len(arr) == 3:
            arr.sort(key=lambda k: int(k[0]))
            agg[cyc] = (float(arr[0][1]), float(arr[-1][4]))

    pts = []
    no_kline = no_quote = 0
    for s in raw:
        ts, q = int(s["timestamp"]), s.get("down_price")
        if q is None or q <= 0.01 or q >= 0.99:
            no_quote += 1
            continue
        cyc = ts // 900_000
        oc = agg.get(int(cyc))
        if oc is None or oc[0] <= 0:
            no_kline += 1
            continue
        pts.append({
            "ts": ts, "q": float(q), "off": (ts - cyc * 900_000) / 1000.0,
            "win": oc[1] < oc[0], "btc": s.get("btc_price"),
            "open": oc[0], "cyc": int(cyc),
        })
    n_pts = len(pts)
    markets = sorted({p["cyc"] for p in pts})
    t0 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(pts[0]["ts"] / 1000))
    t1 = time.strftime("%Y-%m-%d %H:%M", time.gmtime(pts[-1]["ts"] / 1000))
    btc_cov = sum(1 for p in pts if p["btc"])
    base = sum(p["win"] for p in pts) / n_pts
    print(f"样本 {len(raw)} → 有效 {n_pts}（弃 {no_quote} 极端报价 / {no_kline} 无K线）")
    print(f"覆盖 {t0} ~ {t1}（UTC），15m 周期 {len(markets)} 个，btc 覆盖 {btc_cov / n_pts:.0%}，DOWN 基准 {base:.1%}")

    # ---------- 1) 校准 ----------
    print("\n===== 1) 市场校准：报价 q → 实际 DOWN 概率 =====")
    print("  q 桶          n(点)  实际P(DOWN)  q̄    偏差(P−q̄)")
    calib = []
    for i in range(9):
        lo, hi = round(0.1 + 0.1 * i, 2), round(0.2 + 0.1 * i, 2)
        grp = [p for p in pts if lo <= p["q"] < hi]
        if not grp:
            continue
        p_hat = sum(p["win"] for p in grp) / len(grp)
        qm = float(np.mean([p["q"] for p in grp]))
        calib.append({"bucket": [lo, hi], "n": len(grp), "p": p_hat, "q": qm, "dev": p_hat - qm})
        print(f"  [{lo:.1f},{hi:.1f})  {len(grp):>6}   {p_hat:7.1%}   {qm:4.2f}   {p_hat - qm:+.1%}")

    # ---------- 2) 分钟 EV(t)（60s 桶，无条件）----------
    print("\n===== 2) 周期内逐分钟 EV(t)（无条件）=====")
    print("  t(分)   n(点)   EV         q̄     P(DOWN)")
    ev_rows = []
    for b in range(15):
        lo, hi = b * 60.0, (b + 1) * 60.0
        grp = [p for p in pts if lo <= p["off"] < hi]
        if not grp:
            continue
        seen: dict[int, list] = {}
        for p in grp:
            seen.setdefault(p["cyc"], []).append(p)
        ev = sum((0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in grp) / len(grp)
        m_per = [float(np.mean([(0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in v]))
                 for v in seen.values()]
        se = float(np.std(m_per) / math.sqrt(len(seen))) if len(seen) > 1 else float("nan")
        qm = float(np.mean([p["q"] for p in grp]))
        pw = sum(p["win"] for p in grp) / len(grp)
        ev_rows.append({"t": hi / 60.0, "ev": ev, "se": se, "n": len(grp), "q": qm, "p": pw})
        print(f"  {lo / 60:4.0f}-{hi / 60:4.0f}  {len(grp):>6}  {ev:+7.3f}±{1.96 * se:.3f}  {qm:4.2f}  {pw:5.1%}")

    # ---------- 3) 三段 × 回落态（S5 确认窗口的直接定价）----------
    print("\n===== 3) 三段 × 回落态 EV（btc 100% 覆盖）=====")
    print("  段落           状态        n(点)  n(周期)  P(DOWN)  q̄     EV")
    seg_rows = []
    for name, lo, hi in (("前1/3(0-5分)", 0, 300), ("中1/3(5-10分)", 300, 600), ("后1/3(10-15分)", 600, 900)):
        for state, stag in ((True, "价<开盘(回落)"), (False, "价≥开盘")):
            grp = [p for p in pts if lo <= p["off"] < hi and ((p["btc"] < p["open"]) == state)]
            if not grp:
                continue
            ev = sum((0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in grp) / len(grp)
            pw = sum(p["win"] for p in grp) / len(grp)
            qm = float(np.mean([p["q"] for p in grp]))
            nc = len({p["cyc"] for p in grp})
            seg_rows.append({"seg": name, "state": stag, "n": len(grp), "nc": nc,
                             "p": pw, "q": qm, "ev": ev})
            print(f"  {name:<13} {stag:<11} {len(grp):>6}  {nc:>5}   {pw:5.1%}  {qm:4.2f}  {ev:+.3f}")

    # ---------- 4) 逐分钟 × 回落态（S6 核心表）----------
    print("\n===== 4) 逐分钟 × 回落态 EV（S6 择时核心表）=====")
    print("  t(分) | 回落: n  P(DOWN)  q̄     EV     | 未回落: n  P(DOWN)  q̄     EV")
    s6_rows = []
    for b in range(15):
        lo, hi = b * 60.0, (b + 1) * 60.0
        cells = []
        for state in (True, False):
            grp = [p for p in pts if lo <= p["off"] < hi and ((p["btc"] < p["open"]) == state)]
            if grp:
                ev = sum((0.98 / p["q"] - 1.0) if p["win"] else -1.0 for p in grp) / len(grp)
                pw = sum(p["win"] for p in grp) / len(grp)
                qm = float(np.mean([p["q"] for p in grp]))
                cells.append((len(grp), pw, qm, ev))
            else:
                cells.append((0, float("nan"), float("nan"), float("nan")))
        (n1, p1, q1, e1), (n0, p0, q0, e0) = cells
        s6_rows.append({"t": hi / 60.0, "fall": cells[0], "rise": cells[1]})
        print(f"  {lo / 60:4.0f}-{hi / 60:4.0f} | {n1:>5} {p1:6.1%} {q1:4.2f} {e1:+6.3f}"
              f" | {n0:>5} {p0:6.1%} {q0:4.2f} {e0:+6.3f}")

    # ---------- 5) 反应滞后 ----------
    print("\n===== 5) 反应滞后：|Δbtc|≥10bp 的 15s 步后 DOWN 报价跟随 =====")
    by_mkt: dict[int, list] = {}
    for p in pts:
        by_mkt.setdefault(p["cyc"], []).append(p)
    lag = {1: [], 2: [], 4: []}
    imm = []
    for cyc, arr in by_mkt.items():
        arr.sort(key=lambda p: p["ts"])
        for i in range(len(arr) - 4):
            if not (arr[i]["btc"] and arr[i + 1]["btc"]):
                continue
            dp = arr[i + 1]["btc"] / arr[i]["btc"] - 1.0
            if abs(dp) < 0.0010:
                continue
            down_move = dp < 0
            dq0 = arr[i + 1]["q"] - arr[i]["q"]
            imm.append(dq0 if down_move else -dq0)
            for k in lag:
                dq = arr[i + 1 + k]["q"] - arr[i + 1]["q"]
                lag[k].append(dq if down_move else -dq)
    if imm:
        print(f"  触发步数 n={len(imm)}")
        print(f"  同步(+15s):  Δq 均值 {float(np.mean(imm)):+.4f} 中位 {float(np.median(imm)):+.4f}")
        for k, vals in lag.items():
            print(f"  +{k * 15}s 累计:  均值 {float(np.mean(vals)):+.4f} 中位 {float(np.median(vals)):+.4f}")

    # ---------- 6) S1 事件次周期实录 ----------
    print("\n===== 6) S1 事件次周期报价路径（线上 bull_exhaust 信号）=====")
    sig_times = [1787071801000]   # 2026-08-17 23:30:01 S1（q5=0.11 那条）
    for st in sig_times:
        cyc = st // 900_000 + 1
        arr = sorted([p for p in pts if p["cyc"] == cyc], key=lambda p: p["ts"])
        if not arr:
            print(f"  信号 {time.strftime('%m-%d %H:%M', time.gmtime(st / 1000))}: 次周期无采样")
            continue
        print(f"  信号 {time.strftime('%m-%d %H:%M', time.gmtime(st / 1000))} 次周期 "
              f"({len(arr)} 点, 开盘 {arr[0]['open']:.0f}, 结算 {'DOWN赢' if arr[0]['win'] else 'UP赢'}):")
        for p in arr[::4]:
            rel = "回落" if p["btc"] < p["open"] else "上涨"
            print(f"    +{p['off'] / 60:5.1f}分  btc {p['btc']:.0f} ({rel})  q(DOWN) {p['q']:.2f}")

    # ---------- 要点 ----------
    print("\n===== 要点 =====")
    if seg_rows:
        f_mid = next((r for r in seg_rows if r["seg"].startswith("前") and r["state"].startswith("价<")), None)
        if f_mid:
            print(f"  S5 确认窗口（前1/3回落）：P={f_mid['p']:.1%} q̄={f_mid['q']:.2f} EV={f_mid['ev']:+.3f}"
                  f"（S1 形态前提下真实 P=78.5% → EV≈{0.785 * 0.98 / f_mid['q'] - 1:+.3f}）")
    valid = [r for r in s6_rows if r["fall"][0] >= 100]
    if valid:
        best = max(valid, key=lambda r: r["fall"][3])
        print(f"  回落态逐分钟 EV 最优：t={best['t'] - 0.5:.0f}分 EV={best['fall'][3]:+.3f}"
              f"（n={best['fall'][0]}）vs t=5 分 EV={next((r['fall'][3] for r in s6_rows if abs(r['t'] - 5.0) < 0.1), float('nan')):+.3f}")

    with open("output/s6_quote_analysis_15m_result.json", "w", encoding="utf-8") as f:
        json.dump({"calib": calib, "ev_rows": ev_rows, "seg_rows": seg_rows, "s6_rows": s6_rows},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s6_quote_analysis_15m_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
