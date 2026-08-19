#!/usr/bin/env python3
"""S7 稳健性检查：极低报价（<0.15/0.20）的 EV 巨尾截断/剔除 + 中位数 + A/B 分段。

背景：上涨态 EV +0.66~+0.74 由 q̂≈0.05~0.15 的尾部事件主导（Jensen 不等式：
mean(0.98/q) ≫ 0.98/mean(q)）。三口径检验哪部分 EV 是"可指望的"：
  原始      中间价口径全部计入（乐观上界）
  q截断0.15 假设低于 0.15 的单只能以 0.15 成交（保守滑点假设）
  剔除q<0.15 报价低于 0.15 的机会直接放弃（不成交）
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "output/online_15m_samples_full.json"
W1M = "output/klines_1m_s1_windows.json"
DEPTH_EDGES = (0.0005, 0.0015)
EPS = 0.0005
LOOKBACK = 48
import time


def dbucket(d: float) -> int:
    if d < DEPTH_EDGES[0]:
        return 0
    if d < DEPTH_EDGES[1]:
        return 1
    return 2


def main() -> int:
    now_ms = int(time.time() * 1000)

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
        cyc_list.append(cyc); ks[cyc] = idxs
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
        if not s1[j] or cyc_list[j + 1] != cyc_list[j] + 1 or o15[j + 1] <= 0:
            continue
        events.append({"start": (cyc_list[j] + 1) * 900_000, "open": float(o15[j + 1]),
                       "win": bool(c15[j + 1] < o15[j + 1]),
                       "seg": "A" if (cyc_list[j] + 1) * 900_000 < a_end else "B"})
    with open(W1M, encoding="utf-8") as f:
        cache = json.load(f)
    paths = []
    for e in events:
        rows = [r for r in cache.get(str(e["start"]), []) if int(r[0]) < e["start"] + 900_000]
        if len(rows) != 15:
            continue
        closes = [float(r[4]) for r in rows]
        if (closes[-1] < e["open"]) == e["win"]:
            paths.append({**e, "closes": closes})

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

    def price_of(p: dict, t: int) -> float:
        px = p["closes"][t - 1]
        st = 0 if px < p["open"] else 1
        return qhat(t - 1, st, dbucket(abs(math.log(px / p["open"]))))

    def report(tag: str, evs: list[tuple[float, bool]], seg: str | None = None) -> None:
        """evs = [(q, win)]；三口径 EV + 中位数 + A/B。"""
        rows = [(q, w) for q, w in evs if q == q and 0.01 < q < 0.99]
        if len(rows) < 30:
            print(f"  {tag}: n<30 跳过")
            return
        def ev_of(pairs):
            return float(np.mean([(0.98 / q - 1.0) if w else -1.0 for q, w in pairs]))
        raw = ev_of(rows)
        cap = [(max(q, 0.15), w) for q, w in rows]
        cap_ev = ev_of(cap)
        keep = [(q, w) for q, w in rows if q >= 0.15]
        keep_ev = ev_of(keep) if len(keep) >= 30 else float("nan")
        med = float(np.median([(0.98 / q - 1.0) if w else -1.0 for q, w in rows]))
        n_deep = sum(1 for q, _ in rows if q < 0.15)
        pw = sum(w for _, w in rows) / len(rows)
        seg_s = f" [{seg}]" if seg else ""
        print(f"  {tag}{seg_s}: n={len(rows)} P={pw:.1%} | EV原始 {raw:+.3f} | 截断0.15 {cap_ev:+.3f}"
              f" | 剔除<0.15(弃{n_deep}) {keep_ev:+.3f} | 单事件EV中位 {med:+.3f}")

    print("===== 上涨态各分钟 三口径 EV（q<0.15 视为不可成交的保守检验）=====")
    for t in (2, 3, 4, 5, 6, 8, 10, 12):
        grp = [p for p in paths if p["closes"][t - 1] >= p["open"]]
        report(f"涨态 t={t:>2}", [(price_of(p, t), p["win"]) for p in grp])

    print("\n===== 上涨态 t=2~6 合并（早期上涨窗口）三口径 + A/B =====")
    for seg in (None, "A", "B"):
        evs = []
        for p in paths:
            if seg and p["seg"] != seg:
                continue
            for t in (2, 3, 4, 5, 6):
                if p["closes"][t - 1] >= p["open"]:
                    evs.append((price_of(p, t), p["win"]))
                    break          # 每事件只取首个进入窗口的分钟，避免重复计数
        report("涨态早窗 t*=首个", evs, seg)

    print("\n===== 拐点 B1（θ=15bp）三口径 + A/B =====")
    for seg in (None, "A", "B"):
        evs = []
        for p in paths:
            if seg and p["seg"] != seg:
                continue
            peak = -1.0
            for t in range(1, 15):
                px = p["closes"][t - 1]
                peak = max(peak, px)
                if peak > 0 and (peak - px) / peak >= 0.0015:
                    evs.append((price_of(p, t), p["win"]))
                    break
        report("拐点θ=15bp", evs, seg)

    print("\n===== 对照：回落态主表行（前次 +0.242/+0.104）三口径 =====")
    for t in (1, 5):
        grp = [p for p in paths if p["closes"][t - 1] < p["open"]]
        report(f"跌态 t={t}", [(price_of(p, t), p["win"]) for p in grp])
    return 0


if __name__ == "__main__":
    sys.exit(main())
