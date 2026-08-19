#!/usr/bin/env python3
"""次周期逐分钟真实报价决策表：每分钟 × 状态 → 报价 + 胜率 + EV。

用户核心需求（最终口径）：所有 EV 必须用逐分钟真实报价配对计算，
不用固定入场价假设。

三个口径，全部输出：
  表1  15m 市场全体次周期（6 天真实配对）：t=1..15 × 双态 × 深度档，
       每周期每分钟取首条样本（状态+报价+该周期结算胜负）→ n/P/q̄/EV
  表2  720d 混合口径：胜率用 720d 全体 15m 周期（5m K 判 t=5/10/15 状态），
       报价用表1 的 q̂ → 大样本胜率 × 真实报价
  表3  S1 形态条件版（720d 事件池 1m 路径 + 6 天报价表）：t=1..15 × 双态，
       上轮 S7 只给了上涨态，这里补全两态

数据：prediction_market_samples_online_20260819.json（15m 31,466 条），
      klines_5m_cache_720d.json（结算判定），klines_1m_s1_windows.json（S1 路径）。
口径：EV = 赢 0.98/q−1 / 输 −1（费 2%）；down_price 视为可成交价。
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
W1M = "output/klines_1m_s1_windows.json"
LOG = "output/minute_quote_ev.log"
DEPTH_EDGES = (0.0005, 0.0015)
FEE_RET = 0.98


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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- K 线载入与 15m 周期聚合 ----------
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(c5):
        buckets[int(k[0]) // 900_000].append(i)
    agg15: dict[int, tuple[float, float]] = {}
    for cyc, idxs in buckets.items():
        if len(idxs) == 3:
            idxs.sort()
            agg15[cyc] = (float(c5[idxs[0]][1]), float(c5[idxs[-1]][4]))
    print(f"15m 周期 {len(agg15)} 个（720d，K 线聚合）")

    with open(SAMPLES, encoding="utf-8") as f:
        raw_all = json.load(f)
    raw15 = [s for s in raw_all if s.get("market_period") == "15m"]
    print(f"15m 市场报价样本 {len(raw15)} 条（08-13~08-19，6 天）")

    # ---------- 表1：全体次周期逐分钟双态三档（6 天真实配对） ----------
    # 每周期每分钟首条样本：状态 = 该分钟开头 btc vs 开盘
    pair: dict[tuple, list] = defaultdict(list)  # (t, st, d) -> [(q, win)]
    seen: set[tuple] = set()
    for s in raw15:
        q = s.get("down_price"); ts = int(s["timestamp"]); btc = s.get("btc_price")
        cyc = int(ts // 900_000)
        oc = agg15.get(cyc)
        if q is None or q <= 0.01 or q >= 0.99 or btc is None or oc is None or oc[0] <= 0:
            continue
        b = int((ts - cyc * 900_000) // 60_000)
        if b >= 15:
            continue
        key = (cyc, b)
        if key in seen:
            continue
        seen.add(key)
        st = 0 if btc < oc[0] else 1
        d = dbucket(abs(math.log(btc / oc[0])))
        win = oc[1] < oc[0]
        pair[(b, st, d)].append((float(q), win))
    print(f"配对周期-分钟 {len(seen)} 个\n")

    print("===== 表1  15m 次周期逐分钟真实配对 EV（全体周期，无形态条件，6 天） =====")
    print("  t(分) | 回落态(btc<开盘): n    P(DOWN) [Wilson]   q̄     EV     | 上涨态: n    P(DOWN) [Wilson]   q̄     EV")
    t1_rows = []
    for b in range(15):
        cells = []
        for st in (0, 1):
            grp = []
            for d in range(3):
                grp.extend(pair.get((b, st, d), []))
            if len(grp) >= 30:
                k_ = sum(w for _, w in grp)
                p_ = k_ / len(grp)
                lo, hi = wilson(k_, len(grp))
                q_ = float(np.mean([q for q, _ in grp]))
                ev_ = float(np.mean([(FEE_RET / q - 1.0) if w else -1.0 for q, w in grp]))
                cells.append(f"{len(grp):>4} {p_:6.1%} [{lo:.0%},{hi:.0%}] {q_:.3f} {ev_:+.3f}")
                t1_rows.append({"t": b + 1, "st": st, "n": len(grp), "p": p_,
                                "q": q_, "ev": ev_, "lo": lo, "hi": hi})
            else:
                cells.append(f"{len(grp):>4}    —                —      —   ")
        print(f"  {b + 1:>4}  | {cells[0]} | {cells[1]}")

    print("\n  深度档明细（回落态 / 上涨态，n≥30 才显示）：")
    for st, name in ((0, "回落"), (1, "上涨")):
        for b in range(15):
            parts = []
            for d in range(3):
                grp = pair.get((b, st, d), [])
                if len(grp) >= 30:
                    k_ = sum(w for _, w in grp)
                    p_ = k_ / len(grp)
                    q_ = float(np.mean([q for q, _ in grp]))
                    ev_ = float(np.mean([(FEE_RET / q - 1.0) if w else -1.0 for q, w in grp]))
                    parts.append(f"{'浅中深'[d]}:n={len(grp):>4} P={p_:5.1%} q={q_:.3f} EV={ev_:+.3f}")
            if parts:
                print(f"    t={b + 1:>2} {name}态 | " + " | ".join(parts))

    # ---------- 表2：720d 混合口径（胜率 720d，报价 6 天） ----------
    # 720d 全体 15m 周期，用第 k 根 5m K 收盘判 t=5k 的状态+深度
    print("\n===== 表2  720d 混合口径：胜率(720d 全体周期) × 报价(6 天真实) =====")
    print("  （状态/深度用第 k 根 5m K 收盘判定，t = 5k 分钟；报价 q̂ 取表1 同组均值）")
    # 720d 状态分布
    p720: dict[tuple, list] = defaultdict(list)  # (k5, st, d) -> [win]
    for cyc, (o_, c_) in agg15.items():
        idxs = buckets[cyc]
        for k5 in range(3):
            px = c5[idxs[k5]][4]
            if o_ <= 0:
                continue
            st = 0 if px < o_ else 1
            d = dbucket(abs(math.log(px / o_)))
            p720[(k5, st, d)].append(c_ < o_)
    print("  t(分) | 回落态: n720   P720    q̂6d   EV混合  | 上涨态: n720   P720    q̂6d   EV混合")
    for k5 in range(3):
        b = k5 * 5 + 4  # t=5/10/15 → 分钟桶 b=4/9/14
        cells = []
        for st in (0, 1):
            n720 = sum(len(p720.get((k5, st, d), [])) for d in range(3))
            if n720 < 100:
                cells.append(f"{n720:>6}    —       —      —   ")
                continue
            wins720 = sum(sum(v) for d in range(3) for v in [p720.get((k5, st, d), [])])
            p_ = wins720 / n720
            # 报价：同 (b, st) 的 6 天样本按深度分布加权
            qs, ws_ = [], []
            for d in range(3):
                g6 = pair.get((b, st, d), [])
                g72 = p720.get((k5, st, d), [])
                if g6 and g72:
                    qs.append(float(np.mean([q for q, _ in g6])))
                    ws_.append(len(g72))
            if not qs:
                cells.append(f"{n720:>6} {p_:6.1%}    —      —   ")
                continue
            q_ = float(np.average(qs, weights=ws_))
            ev_ = p_ * (FEE_RET / q_ - 1.0) - (1 - p_)
            cells.append(f"{n720:>6} {p_:6.1%} {q_:.3f} {ev_:+.3f}")
        print(f"  {(k5 + 1) * 5:>4}  | {cells[0]} | {cells[1]}")

    # ---------- 表3：S1 形态条件（720d 事件池 1m 路径 + 6 天报价表） ----------
    print("\n===== 表3  S1 形态下次周期逐分钟 EV（720d 事件池 × 6 天报价表，两态全表） =====")
    # 重建 S1 事件（720d 锁定口径）
    t5 = np.array([r[0] for r in c5]); o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5]); cl5 = np.array([r[4] for r in c5])
    cyc_list = sorted(agg15.keys())
    cyc_idx = {c: i for i, c in enumerate(cyc_list)}
    o15a = np.array([agg15[c][0] for c in cyc_list])
    c15a = np.array([agg15[c][1] for c in cyc_list])
    # 高低（用 5m 聚合）
    h15a = np.array([max(c5[i][2] for i in buckets[c]) for c in cyc_list])
    l15a = np.array([min(c5[i][3] for i in buckets[c]) for c in cyc_list])
    dir15 = np.sign(c15a - o15a)
    close_pos = (c15a - l15a) / np.where(h15a > l15a, h15a - l15a, np.nan)

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

    w16_hi = roll_max(c15a, 16); w16_lo = roll_min(c15a, 16)
    pos4h = (c15a - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)
    LOOKBACK = 48
    EPS = 0.0005
    lvl_hi = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(len(cyc_list), dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in buckets[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]) and broke_hi5[i]:
                broke_hi15[j] = True
    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)

    a_end = now_ms - 360 * 86_400_000
    events = []
    for j in range(len(cyc_list) - 1):
        if not s1[j] or cyc_list[j + 1] != cyc_list[j] + 1:
            continue
        if o15a[j + 1] <= 0:
            continue
        events.append({"start": (cyc_list[j] + 1) * 900_000, "open": float(o15a[j + 1]),
                       "win": bool(c15a[j + 1] < o15a[j + 1]),
                       "seg": "A" if (cyc_list[j] + 1) * 900_000 < a_end else "B"})
    with open(W1M, encoding="utf-8") as f:
        cache = json.load(f)
    paths = []
    for e in events:
        arr = cache.get(str(e["start"]), [])
        rows_ = [r for r in arr if int(r[0]) < e["start"] + 900_000]
        if len(rows_) != 15:
            continue
        closes = [float(r[4]) for r in rows_]
        if (closes[-1] < e["open"]) != e["win"]:
            continue
        paths.append({**e, "closes": closes})
    print(f"S1 事件 {len(events)} → 有效 1m 路径 {len(paths)}")

    # 6 天报价表 q̂(b, st, d)（表1 的 pair 就是，直接用组内均值）
    def qhat(b: int, st: int, d: int) -> float:
        g = pair.get((b, st, d), [])
        if len(g) >= 30:
            return float(np.mean([q for q, _ in g]))
        # 档内不足 → 同 (b, st) 全体
        ga = []
        for dd in range(3):
            ga.extend(pair.get((b, st, dd), []))
        return float(np.mean([q for q, _ in ga])) if ga else float("nan")

    print("  t(分) | 回落态: n    P(DOWN) [Wilson]  q̂     EV     | 上涨态: n    P(DOWN) [Wilson]  q̂     EV")
    t3_rows = []
    for t in range(1, 15):
        cells = []
        for st in (0, 1):
            grp = [p for p in paths if (p["closes"][t - 1] < p["open"]) == (st == 0)]
            if len(grp) < 20:
                cells.append(f"{len(grp):>4}    —               —      —   ")
                continue
            k_ = sum(p["win"] for p in grp)
            p_ = k_ / len(grp)
            lo, hi = wilson(k_, len(grp))
            qs = []
            for p in grp:
                px = p["closes"][t - 1]
                d = dbucket(abs(math.log(px / p["open"])))
                qs.append(qhat(t - 1, st, d))
            evs = [(FEE_RET / q - 1.0) if p["win"] else -1.0 for p, q in zip(grp, qs)]
            cells.append(f"{len(grp):>4} {p_:6.1%} [{lo:.0%},{hi:.0%}] {float(np.mean(qs)):.3f} {float(np.mean(evs)):+.3f}")
            t3_rows.append({"t": t, "st": st, "n": len(grp), "p": p_,
                            "q": float(np.mean(qs)), "ev": float(np.mean(evs))})
        print(f"  {t:>4}  | {cells[0]} | {cells[1]}")

    # ---------- 最优行汇总 ----------
    print("\n===== 汇总：各口径下 EV 最高的入场时点（n≥100） =====")
    for name, rows in (("表1 全体周期(6d配对)", t1_rows), ("表3 S1形态(720d×6d)", t3_rows)):
        ok = [r for r in rows if r["n"] >= 100]
        if not ok:
            continue
        top = sorted(ok, key=lambda r: -r["ev"])[:5]
        print(f"  {name}:")
        for r in top:
            stn = "回落" if r["st"] == 0 else "上涨"
            print(f"    t={r['t']:>2}分 {stn}态: n={r['n']:>5} P={r['p']:.1%} q={r['q']:.3f} EV={r['ev']:+.3f}")

    with open("output/minute_quote_ev_result.json", "w", encoding="utf-8") as f:
        json.dump({"table1_all_cycles": t1_rows, "table3_s1_events": t3_rows},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/minute_quote_ev_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
