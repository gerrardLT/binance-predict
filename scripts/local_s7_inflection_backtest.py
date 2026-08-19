#!/usr/bin/env python3
"""S7 拐点择时回测（用户假设）：S1 形态下，次周期"先涨后跌拐点"入场的 EV。

检验两个主张（2026-08-18，探索性分析）：
  A. 放弃组（+5min 仍在涨）可能因报价便宜而是正 EV —— 胜率 34% ≠ 负 EV
  B. 最优入场时点是"先涨后跌的拐点"——涨势衰竭、DOWN 报价被压低时买入

数据（全部复用既有资产，零新拉取）：
  - output/klines_5m_cache_720d.json     720d 5m K（S1 识别 + 次周期结算）
  - output/klines_1m_s1_windows.json     2294 个 S1 事件次周期 1m 路径（已缓存）
  - output/online_15m_samples_full.json  5 天 15s 采样 → 双态报价表
    q̂(分钟, 状态∈{上涨,回落}, 深度档)：上涨态 DOWN 报价被压低，是拐点策略的核心

拐点定义：运行最高点（1m close 的 running max）回落 ≥θ 的首个分钟 t_c，
θ ∈ {10bp, 15bp, 20bp, 30bp} 扫描；变体：
  B1 无限制（任何拐点）
  B2 纯"先涨后跌"（t_c 时价格仍在开盘上方）
  对照：t=1 逐分钟主表 EV +0.242 / S5(t=5 回落) +0.104
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

FEE = 0.02
EPS = 0.0005
LOOKBACK = 48
KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "output/online_15m_samples_full.json"
W1M = "output/klines_1m_s1_windows.json"
LOG = "output/s7_inflection_backtest.log"
DEPTH_EDGES = (0.0005, 0.0015)
THETAS = (0.0010, 0.0015, 0.0020, 0.0030)


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def dbucket(depth: float) -> int:
    if depth < DEPTH_EDGES[0]:
        return 0
    if depth < DEPTH_EDGES[1]:
        return 1
    return 2


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- S1 识别（口径 = 720d 锁定）----------
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5]); o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5]); l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])

    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list, ks = [], {}
    for cyc, idxs in buckets.items():
        if len(idxs) != 3 or (cyc + 1) * 900_000 > now_ms:
            continue
        idxs.sort()
        cyc_list.append(cyc)
        ks[cyc] = idxs
    cyc_list.sort()
    N = len(cyc_list)
    o15 = np.array([c5[ks[c][0]][1] for c in cyc_list])
    h15 = np.array([max(h5[i] for i in ks[c]) for c in cyc_list])
    l15 = np.array([min(l5[i] for i in ks[c]) for c in cyc_list])
    c15 = np.array([c5[ks[c][-1]][4] for c in cyc_list])
    dir15 = np.sign(c15 - o15)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

    def roll_max(x, w):
        from numpy.lib.stride_tricks import sliding_window_view
        out = np.full(len(x), np.nan)
        out[w - 1:] = sliding_window_view(x, w).max(axis=1)
        return out

    def roll_min(x, w):
        from numpy.lib.stride_tricks import sliding_window_view
        out = np.full(len(x), np.nan)
        out[w - 1:] = sliding_window_view(x, w).min(axis=1)
        return out

    w16_hi = roll_max(c15, 16); w16_lo = roll_min(c15, 16)
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)
    lvl_hi = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]) and broke_hi5[i]:
                broke_hi15[j] = True
    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)

    a_end = now_ms - 360 * 86_400_000
    pos_in = {c: j for j, c in enumerate(cyc_list)}
    events = []
    for j in range(N - 1):
        if not s1[j] or cyc_list[j + 1] != cyc_list[j] + 1:
            continue
        if o15[j + 1] <= 0:
            continue
        events.append({"start": (cyc_list[j] + 1) * 900_000, "open": float(o15[j + 1]),
                       "win": bool(c15[j + 1] < o15[j + 1]),
                       "seg": "A" if (cyc_list[j] + 1) * 900_000 < a_end else "B"})

    with open(W1M, encoding="utf-8") as f:
        cache = json.load(f)
    paths = []
    for e in events:
        arr = cache.get(str(e["start"]), [])
        rows = [r for r in arr if int(r[0]) < e["start"] + 900_000]
        if len(rows) != 15:
            continue
        closes = [float(r[4]) for r in rows]
        if (closes[-1] < e["open"]) != e["win"]:
            continue
        paths.append({**e, "closes": closes})
    print(f"S1 事件 {len(events)} → 有效 1m 路径 {len(paths)}"
          f"（A {sum(1 for p in paths if p['seg'] == 'A')} / B {sum(1 for p in paths if p['seg'] == 'B')}）")

    # ---------- 双态报价表 q̂(桶, 状态, 深度档) ----------
    with open(SAMPLES, encoding="utf-8") as f:
        raw = json.load(f)
    agg: dict[int, tuple[float, float]] = {}
    byc: dict[int, list[list]] = {}
    for k in kl:
        byc.setdefault(int(k[0]) // 900_000, []).append(k)
    for cyc, arr2 in byc.items():
        if len(arr2) == 3:
            arr2.sort(key=lambda k: int(k[0]))
            agg[cyc] = (float(arr2[0][1]), float(arr2[-1][4]))
    qt = {b: {st: {d: [] for d in range(3)} for st in (0, 1)} for b in range(15)}
    qb_all = {b: {st: [] for st in (0, 1)} for b in range(15)}
    for s in raw:
        q = s.get("down_price"); ts = int(s["timestamp"]); btc = s.get("btc_price")
        cyc = ts // 900_000
        oc = agg.get(int(cyc))
        if q is None or q <= 0.01 or q >= 0.99 or btc is None or oc is None or oc[0] <= 0:
            continue
        b = int((ts - cyc * 900_000) // 60_000)
        if b >= 15:
            continue
        st = 0 if btc < oc[0] else 1
        qt[b][st][dbucket(abs(math.log(btc / oc[0])))].append(float(q))
        qb_all[b][st].append(float(q))

    def qhat(b: int, st: int, d: int) -> float:
        vals = qt[b][st][d]
        if len(vals) >= 30:
            return float(np.mean(vals))
        allv = qb_all[b][st]
        return float(np.mean(allv)) if allv else float("nan")

    print("\n双态报价表 q̂(t, 状态)（DOWN 报价；上涨态 = 价≥开盘）")
    print("  t(分) | 回落态: 浅    中    深    n | 上涨态: 浅    中    深    n")
    for b in range(15):
        cells = [f"{qhat(b, 0, d):.3f}" for d in range(3)] + [f"{qhat(b, 1, d):.3f}" for d in range(3)]
        print(f"  {b + 1:>4}  | {cells[0]} {cells[1]} {cells[2]} {len(qb_all[b][0]):>4}"
              f" | {cells[3]} {cells[4]} {cells[5]} {len(qb_all[b][1]):>4}")

    def price_of(p: dict, t: int) -> float:
        """事件 t 分钟（1-based）入场时刻的报价：按当时状态+深度档。"""
        px = p["closes"][t - 1]
        st = 0 if px < p["open"] else 1
        return qhat(t - 1, st, dbucket(abs(math.log(px / p["open"]))))

    # ---------- 分析 1：上涨态逐分钟 EV(t)（对称主表，检验主张 A）----------
    print("\n===== 1) S1 形态下，上涨态（价≥开盘）逐分钟买 DOWN 的 EV =====")
    print("  t(分)   n   P(DOWN)  Wilson95     q̂eff   EV")
    rise_rows = []
    for t in range(1, 15):
        grp = [p for p in paths if p["closes"][t - 1] >= p["open"]]
        if len(grp) < 20:
            continue
        k = sum(p["win"] for p in grp)
        lo, hi = wilson(k, len(grp))
        qs = [price_of(p, t) for p in grp]
        ev = float(np.mean([(0.98 / q - 1.0) if p["win"] else -1.0 for p, q in zip(grp, qs)]))
        rise_rows.append({"t": t, "n": len(grp), "p": k / len(grp), "q": float(np.mean(qs)), "ev": ev})
        print(f"  {t:>4}  {len(grp):>5}  {k / len(grp):6.1%}  [{lo:.1%},{hi:.1%}]  "
              f"{float(np.mean(qs)):.3f}  {ev:+.3f}")
    r5 = next((r for r in rise_rows if r["t"] == 5), None)
    if r5:
        print(f"  → 主张A检验：+5min 仍在涨（放弃组）P={r5['p']:.1%} q̂={r5['q']:.3f} EV={r5['ev']:+.3f}")

    # ---------- 分析 2：拐点策略（主张 B）----------
    print("\n===== 2) 拐点策略：运行高点回落 ≥θ 首分钟入场 =====")
    print("  θ     | B1 无限制: n   P     q̂     EV    | B2 拐点在开盘上方: n   P     q̂     EV")
    infl_rows = []
    for th in THETAS:
        cells = []
        for mode in ("B1", "B2"):
            evs = []
            ws = []
            qs_ = []
            for p in paths:
                peak = -1.0
                tc = None
                for t in range(1, 15):
                    px = p["closes"][t - 1]
                    peak = max(peak, px)
                    if peak > 0 and (peak - px) / peak >= th:
                        if mode == "B2" and px < p["open"]:
                            break        # 已跌破开盘 → 不是"先涨后跌在上方拐头"
                        tc = t
                        break
                if tc is None:
                    continue
                q = price_of(p, tc)
                evs.append((0.98 / q - 1.0) if p["win"] else -1.0)
                ws.append(p["win"])
                qs_.append(q)
            if len(evs) >= 30:
                pw_ = float(np.mean(ws))
                ev_ = float(np.mean(evs))
                se = float(np.std(evs) / math.sqrt(len(evs)))
                cells.append((len(evs), pw_, float(np.mean(qs_)), ev_, se))
            else:
                cells.append((len(evs), float("nan"), float("nan"), float("nan"), float("nan")))
        infl_rows.append({"theta": th, "B1": cells[0], "B2": cells[1]})
        (n1, p1, q1, e1, s1_), (n2, p2, q2, e2, s2_) = cells
        print(f"  {th * 1e4:>4.0f}bp | {n1:>5} {p1:6.1%} {q1:.3f} {e1:+6.3f}±{1.96 * s1_:.3f}"
              f" | {n2:>5} {p2:6.1%} {q2:.3f} {e2:+6.3f}±{1.96 * s2_:.3f}")

    # ---------- 分析 3：最优单点复核 + A/B 分段 ----------
    best = max((r for r in rise_rows if r["n"] >= 100), key=lambda r: r["ev"], default=None)
    if best:
        print(f"\n===== 3) 上涨态 EV 最优行：t={best['t']} P={best['p']:.1%} q̂={best['q']:.3f} EV={best['ev']:+.3f} =====")
        for seg in ("A", "B"):
            grp = [p for p in paths if p["seg"] == seg and p["closes"][best['t'] - 1] >= p["open"]]
            if len(grp) >= 30:
                qs = [price_of(p, best["t"]) for p in grp]
                ev_ = float(np.mean([(0.98 / q - 1.0) if p["win"] else -1.0
                                     for p, q in zip(grp, qs)]))
                print(f"  {seg} 段: n={len(grp)} P={sum(p['win'] for p in grp) / len(grp):.1%} EV={ev_:+.3f}")

    # ---------- 分析 4：涨后回落深度对最终胜率的影响（机制拆解）----------
    print("\n===== 4) 机制：t=1~5 内出现过的最大上涨深度 vs 最终 P(DOWN) =====")
    for lo_, hi_ in ((0.0, 0.0005), (0.0005, 0.0015), (0.0015, 0.004), (0.004, 1.0)):
        grp = []
        for p in paths:
            up5 = max((px - p["open"]) / p["open"] for px in p["closes"][:5])
            if lo_ <= up5 < hi_:
                grp.append(p)
        if len(grp) >= 30:
            k = sum(p["win"] for p in grp)
            print(f"  前5分最大涨幅 [{lo_ * 1e4:.1f},{hi_ * 1e4:.1f})bp: n={len(grp):>4}  P(DOWN)={k / len(grp):6.1%}"
                  f"  （S1 无条件基准 60.5% → 涨得越深跌回概率越{'高' if k / len(grp) > 0.605 else '低'}）")

    with open("output/s7_inflection_backtest_result.json", "w", encoding="utf-8") as f:
        json.dump({"rise_rows": rise_rows, "infl_rows": infl_rows,
                   "quote_up": {str(b): {str(d): [float(np.mean(qt[b][1][d])), len(qt[b][1][d])]
                                          for d in range(3)} for b in range(15)}},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s7_inflection_backtest_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
