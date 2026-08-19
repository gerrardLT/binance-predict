#!/usr/bin/env python3
"""S5 真实报价 EV 修正 + 5m/15m 市场分钟级定价效率（回应两项质疑）。

质疑 1（EV 口径）：S5 回测 EV@0.51 假设次周期开盘价入场，但 S5 实际在
  第 5 分钟确认后才入场，此时 DOWN 报价已被市场重新定价
  （实盘 3 注入场价 0.89/0.46/0.84，均值 0.73）。→ 分档重算真实报价 EV。

质疑 2（分钟赔率未分析）：新落地样本含 5m 市场 24 天（136,236 条，
  6,850 周期）+ 15m 市场 6 天（31,466 条）。→
  a) 5m 市场逐分钟定价效率：市场消化周期内价格信息的速度（t=1..4 双态 EV）；
  b) 15m 市场新样本双态报价表 q̂(t, 状态, 深度) 与 S7 旧表对照；
  c) S1 直入场（开盘桶 b=0/1）真实报价 EV。

数据：output/klines_5m_cache_720d.json + prediction_market_samples_online_20260819.json
口径：S1/S5 事件 = 720d 锁定口径；EV = 赢 0.98/q−1 / 输 −1；费 2%。
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
EPS = 0.0005
LOOKBACK = 48
KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
LOG = "output/s5_real_quote_ev.log"
DEPTH_EDGES = (0.0005, 0.0015)


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


def dbucket(depth: float) -> int:
    if depth < DEPTH_EDGES[0]:
        return 0
    if depth < DEPTH_EDGES[1]:
        return 1
    return 2


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- 15m 周期 + S1/S5 事件（720d 锁定口径） ----------
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
    events = []
    for j in range(N - 1):
        if not s1[j] or cyc_list[j + 1] != cyc_list[j] + 1:
            continue
        if o15[j + 1] <= 0:
            continue
        nxt = cyc_list[j + 1]
        c1 = c5[ks[nxt][0]][4]  # 次周期第 1 根 5m 收盘（t=5min 时刻价）
        z5 = math.log(c1 / o15[j + 1])
        events.append({
            "start": (cyc_list[j] + 1) * 900_000, "open": float(o15[j + 1]),
            "win": bool(c15[j + 1] < o15[j + 1]),
            "z5": float(z5),
            "d5": dbucket(abs(z5)),
            "seg": "A" if (cyc_list[j] + 1) * 900_000 < a_end else "B",
        })
    s5e = [e for e in events if e["z5"] < 0]
    print(f"S1 事件 {len(events)}（A {sum(1 for e in events if e['seg'] == 'A')}"
          f" / B {sum(1 for e in events if e['seg'] == 'B')}）"
          f" | S5 确认子集（z5<0）{len(s5e)}"
          f"（A {sum(1 for e in s5e if e['seg'] == 'A')} / B {sum(1 for e in s5e if e['seg'] == 'B')}）")

    # ---------- 15m 市场报价样本 → 双态报价表 ----------
    with open(SAMPLES, encoding="utf-8") as f:
        raw_all = json.load(f)
    raw15 = [s for s in raw_all if s.get("market_period") == "15m"]
    raw5 = [s for s in raw_all if s.get("market_period") == "5m"]
    # 15m 周期开盘价（用 5m K 聚合）
    agg15: dict[int, tuple[float, float]] = {}
    for cyc, idxs in buckets.items():
        if len(idxs) == 3:
            idxs.sort()
            agg15[cyc] = (float(c5[idxs[0]][1]), float(c5[idxs[-1]][4]))
    qt15 = {b: {st: {d: [] for d in range(3)} for st in (0, 1)} for b in range(15)}
    qb15 = {b: {st: [] for st in (0, 1)} for b in range(15)}
    for s in raw15:
        q = s.get("down_price"); ts = int(s["timestamp"]); btc = s.get("btc_price")
        cyc = ts // 900_000
        oc = agg15.get(int(cyc))
        if q is None or q <= 0.01 or q >= 0.99 or btc is None or oc is None or oc[0] <= 0:
            continue
        b = int((ts - cyc * 900_000) // 60_000)
        if b >= 15:
            continue
        st = 0 if btc < oc[0] else 1
        qt15[b][st][dbucket(abs(math.log(btc / oc[0])))].append(float(q))
        qb15[b][st].append(float(q))

    def qhat15(b: int, st: int, d: int) -> float:
        vals = qt15[b][st][d]
        if len(vals) >= 30:
            return float(np.mean(vals))
        allv = qb15[b][st]
        return float(np.mean(allv)) if allv else float("nan")

    print(f"\n15m 市场 {len(raw15)} 条样本（新数据，08-13~08-19）双态报价表 q̂(t, 状态)（DOWN）")
    print("  t(分) | 回落态: 浅    中    深   n | 上涨态: 浅    中    深   n")
    for b in range(15):
        cells = [f"{qhat15(b, 0, d):.3f}" for d in range(3)] + [f"{qhat15(b, 1, d):.3f}" for d in range(3)]
        print(f"  {b + 1:>4}  | {cells[0]} {cells[1]} {cells[2]} {len(qb15[b][0]):>4}"
              f" | {cells[3]} {cells[4]} {cells[5]} {len(qb15[b][1]):>4}")

    # ---------- 1) S5 真实报价 EV（分档） ----------
    print("\n===== 1) S5 真实报价 EV（入场 = 次周期第 5 分钟确认后，b=5 桶 300~360s） =====")
    print("  深度档 |    n   P(DOWN)  q̂真实   EV真实    EV@0.51(回测口径)")
    tot_ev = {5: [], 4: []}
    for d in range(3):
        grp = [e for e in s5e if e["d5"] == d]
        if not grp:
            continue
        k = sum(e["win"] for e in grp)
        p = k / len(grp)
        q5 = qhat15(5, 0, d)
        evs = [(0.98 / q5 - 1.0) if e["win"] else -1.0 for e in grp]
        seg_evs = {"A": [], "B": []}
        for e in grp:
            seg_evs[e["seg"]].append((0.98 / q5 - 1.0) if e["win"] else -1.0)
        ab = " | ".join(f"{s}段 EV={float(np.mean(seg_evs[s])):+.3f}(n={len(seg_evs[s])})"
                        for s in ("A", "B") if len(seg_evs[s]) >= 30)
        print(f"  {'浅中深'[d]}(<{DEPTH_EDGES[0] if d == 0 else DEPTH_EDGES[1] if d == 1 else '∞'},{DEPTH_EDGES[1] * 1e4 if d == 0 else ''})"
              f" | {len(grp):>5}  {p:6.1%}  {q5:.3f}  {float(np.mean(evs)):+.3f}"
              f"     {p * (0.98 / 0.51 - 1) - (1 - p):+.3f}")
        print(f"        {ab}")
    for b in (5, 4):
        evs = []
        for e in s5e:
            q = qhat15(b, 0, e["d5"])
            if math.isnan(q):
                continue
            evs.append((0.98 / q - 1.0) if e["win"] else -1.0)
        tot_ev[b] = evs
    ev5 = float(np.mean(tot_ev[5])); ev4 = float(np.mean(tot_ev[4]))
    p_all = sum(e["win"] for e in s5e) / len(s5e)
    ev051 = p_all * (0.98 / 0.51 - 1) - (1 - p_all)
    se5 = float(np.std(tot_ev[5]) / math.sqrt(len(tot_ev[5])))
    print(f"  合计   | {len(s5e):>5}  {p_all:6.1%}  —      {ev5:+.3f}±{1.96 * se5:.3f}"
          f"     {ev051:+.3f}")
    print(f"  对照 b=4（240~300s 确认前报价）EV = {ev4:+.3f}")
    for seg in ("A", "B"):
        grp = [e for e in s5e if e["seg"] == seg]
        evs = [(0.98 / qhat15(5, 0, e["d5"]) - 1.0) if e["win"] else -1.0 for e in grp]
        print(f"  {seg} 段: n={len(grp)} P={sum(e['win'] for e in grp) / len(grp):.1%}"
              f" EV={float(np.mean(evs)):+.3f}")
    qmix = float(np.mean([qhat15(5, 0, e["d5"]) for e in s5e]))
    print(f"  S5 加权入场价 q̄ = {qmix:.3f}（vs 回测假设 0.51）")
    print(f"  盈亏平衡入场价 q* = P×0.98 = {p_all * 0.98:.3f}"
          f" → 实际 q̄ {'<' if qmix < p_all * 0.98 else '≥'} q*，{'仍为正 EV' if qmix < p_all * 0.98 else '已转负'}")

    # ---------- 2) S1 直入场（开盘附近）真实报价 EV ----------
    print("\n===== 2) S1 直入场真实报价 EV（开盘桶 b=0/1 的 DOWN 报价，两态混合） =====")
    for b in (0, 1):
        allv = qb15[b][0] + qb15[b][1]
        if not allv:
            continue
        q0 = float(np.mean(allv))
        evs = [(0.98 / q0 - 1.0) if e["win"] else -1.0 for e in events]
        p1 = sum(e["win"] for e in events) / len(events)
        print(f"  b={b}（开盘后 {b * 60}~{(b + 1) * 60}s）: q̄={q0:.3f} n={len(allv)}"
              f" | S1 全事件 P={p1:.1%} EV={float(np.mean(evs)):+.3f}"
              f"（@0.51 口径 {p1 * (0.98 / 0.51 - 1) - (1 - p1):+.3f}）")
    print("  分解（b=1）：回落态 q̂=" + f"{float(np.mean(qb15[1][0])):.3f}"
          f" / 上涨态 q̂={float(np.mean(qb15[1][1])):.3f}"
          f"（S1 开盘后 1 分钟内的市场分歧）")

    # ---------- 3) 5m 市场逐分钟定价效率（24 天，6,850 周期） ----------
    print("\n===== 3) 5m 市场定价效率：周期内逐分钟买 DOWN 的 EV（双态） =====")
    # 5m K 对齐：klines 的 5m 桶（开盘=K.open，结算=K.close）
    k5map = {int(k[0]): (float(k[1]), float(k[4])) for k in kl}
    rows: dict[tuple, list] = defaultdict(list)
    seen: set[tuple] = set()  # 每周期每 (t,st,d) 只取首条样本（避免重复计数）
    n5cyc = 0
    for s in raw5:
        q = s.get("down_price"); ts = int(s["timestamp"]); btc = s.get("btc_price")
        cyc = ts // 300_000
        k5 = k5map.get(int(cyc * 300_000))
        if k5 is None or q is None or q <= 0.01 or q >= 0.99 or btc is None:
            continue
        ko, kc = k5
        if ko <= 0:
            continue
        b = int((ts - cyc * 300_000) // 60_000)
        if b >= 5:
            continue
        n5cyc += 1
        st = 0 if btc < ko else 1
        d = dbucket(abs(math.log(btc / ko)))
        key = (cyc, b, st, d)
        if key in seen:
            continue
        seen.add(key)
        win = kc < ko
        rows[(b, st, d)].append((float(q), win))
    print(f"  有效样本周期-桶 {len(seen)}（去重后），覆盖 24 天")
    print("  t(分) | 回落态(btc<开盘): n    P     q̄     EV   | 上涨态: n    P     q̄     EV")
    for b in range(5):
        cells = []
        for st in (0, 1):
            grp = rows.get((b, st, 0), []) + rows.get((b, st, 1), []) + rows.get((b, st, 2), [])
            if len(grp) >= 30:
                evs = [(0.98 / q - 1.0) if w else -1.0 for q, w in grp]
                p_ = sum(w for _, w in grp) / len(grp)
                q_ = float(np.mean([q for q, _ in grp]))
                cells.append(f"{len(grp):>5} {p_:5.1%} {q_:.3f} {float(np.mean(evs)):+.3f}")
            else:
                cells.append(f"{len(grp):>5}   —     —      —  ")
        print(f"  {b + 1:>4}  | {cells[0]} | {cells[1]}")
    # 深度档分解（t=2 回落态）：市场对深度的定价
    print("  深度分解（回落态 t=2）：")
    for d in range(3):
        grp = rows.get((1, 0, d), [])
        if len(grp) >= 30:
            evs = [(0.98 / q - 1.0) if w else -1.0 for q, w in grp]
            p_ = sum(w for _, w in grp) / len(grp)
            q_ = float(np.mean([q for q, _ in grp]))
            print(f"    深度{'浅中深'[d]}: n={len(grp):>5} P={p_:5.1%} q̄={q_:.3f} EV={float(np.mean(evs)):+.3f}"
                  f" | 隐含 P(q̄)={q_:.3f} 定价偏差={p_ - q_:+.3f}")
    # 时间稳定性：前后半样本（含输赢，按样本 ts 分半；每周期每桶首条去重）
    mid_ts = (1785067501000 + 1786610626000) / 2
    stab: dict[str, list] = {"前": [], "后": []}
    seen2: set[tuple] = set()
    for s in raw5:
        ts = int(s["timestamp"])
        q = s.get("down_price"); btc = s.get("btc_price")
        cyc = ts // 300_000
        k5 = k5map.get(int(cyc * 300_000))
        if k5 is None or q is None or q <= 0.01 or q >= 0.99 or btc is None:
            continue
        ko, kc = k5
        b = int((ts - cyc * 300_000) // 60_000)
        if b != 1 or btc >= ko:
            continue
        key = (cyc, b)
        if key in seen2:
            continue
        seen2.add(key)
        ev = (0.98 / q - 1.0) if kc < ko else -1.0
        stab["前" if ts < mid_ts else "后"].append(ev)
    print("  时间稳定性（回落态 t=2 逐周期，含输赢）：")
    for half in ("前", "后"):
        if len(stab[half]) >= 30:
            se_ = float(np.std(stab[half]) / math.sqrt(len(stab[half])))
            print(f"    {half}半: n={len(stab[half])} EV={float(np.mean(stab[half])):+.3f}±{1.96 * se_:.3f}")

    # ---------- 4) 实盘对照（线上 27 注快照） ----------
    print("\n===== 4) 实盘对照（fake_breakout_signals 报价快照，08-17 新版起） =====")
    live = [
        # (pattern, entry_dn, win_down)  # id
        ("S1", 0.51, False), ("S1", 0.58, True), ("S1", 0.47, False),
        ("S1", 0.48, True), ("S1", 0.69, False), ("S1", 0.55, True),
        ("S5", 0.89, True), ("S5", 0.46, False), ("S5", 0.84, True),
        ("S4", 0.46, False), ("S4", 0.49, True), ("S4", 0.75, True),
        ("S4", 0.46, False), ("S4", 0.52, True), ("S4", 0.62, True),
    ]
    by = defaultdict(list)
    for pat, q, w in live:
        by[pat].append((q, w))
    for pat in ("S1", "S5", "S4"):
        grp = by[pat]
        evs = [(0.98 / q - 1.0) if w else -1.0 for q, w in grp]
        print(f"  {pat}: n={len(grp)} 入场价 {[f'{q:.2f}' for q, _ in grp]}"
              f" q̄={float(np.mean([q for q, _ in grp])):.3f}"
              f" 胜率={sum(w for _, w in grp) / len(grp):.1%} 实盘 EV={float(np.mean(evs)):+.3f}")
    print("  （S1 id14 案例同一事件：S1 直入 0.58 → +69%，若等确认 q5=0.11 反弹……结局仍 DOWN；")
    print("   S1 id17/18 案例：S1 直入 0.48 → +104% vs S5 确认后 0.89 → +10%）")

    with open("output/s5_real_quote_ev_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "s5_real_ev": ev5, "s5_ev_b4": ev4, "s5_ev_at_051": ev051,
            "s5_n": len(s5e), "s5_p": p_all, "s5_qbar": qmix,
        }, f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s5_real_quote_ev_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
