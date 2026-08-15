#!/usr/bin/env python3
"""入场时机 v2（修正版）：z 空间定价，修复 5m 曲面→15m 周期的时间缩放错误。

v1 错误：直接按"剩余时间比例 τ"套用 5m 市场价格，忽略了同样 τ 下绝对剩余时间
差 3 倍（5m 市场 τ=0.3 只剩 90s，15m 周期 τ=0.3 还剩 4.5min），反转概率被低估。

v2 方法：
  状态 = (z, τ半区)，z = 周期内领先幅度 d / (σ_cycle × √τ_rem)  [布朗缩放的等价状态]
  曲面：5m 市场样本 → 每 (τ半区, z桶) 的 down_price 中位 + 实际收阴频率（按周期去重）
  事件：180 天场景①②，次周期 t=5/10min 的 z 映射到曲面价格
  假设声明：15m 市场参与者行为 ≈ 5m 市场（同一批人的追涨杀跌），按 z 等价状态转移定价。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48
DAYS = 180
API = "https://data-api.binance.vision/api/v3/klines"

Z_EDGES = [-4.0, -2.0, -1.0, -0.33, 0.33, 1.0, 2.0, 4.0]
Z_NAMES = ["z<-4", "-4~-2", "-2~-1", "-1~-0.33", "-0.33~+0.33", "+0.33~+1", "+1~+2", "z>+4"]


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    batch = json.loads(resp.read().decode())
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  重试 {attempt + 1}/3: {e}")
                time.sleep(2)
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 1
        time.sleep(0.2)
    return out


def ev(p: float, e: float) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def zbin(z: float) -> int:
    for i, (lo, hi) in enumerate(zip([-99] + Z_EDGES, Z_EDGES + [99])):
        if lo <= z < hi:
            return i
    return 3


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    # ---------- A. 5m 市场样本 → z 空间定价曲面 ----------
    with open(os.path.join(ROOT, "prediction_market_samples.json"), encoding="utf-8") as f:
        samples = json.load(f)
    samples = [s for s in samples if s.get("down_price") is not None
               and 0.02 < float(s["down_price"]) < 0.98]
    lo_t, hi_t = min(int(s["timestamp"]) for s in samples), max(int(s["timestamp"]) for s in samples)

    k1 = fetch_klines("1m", lo_t - 600_000, hi_t + 600_000)
    p1 = {int(k[0]): float(k[4]) for k in k1}
    k5s = fetch_klines("5m", lo_t - 900_000, hi_t + 900_000)
    cyc5 = {int(k[0]) // 300_000: (float(k[1]), float(k[4])) for k in k5s}
    sigma5 = float(np.std([(c - o) / o for o, c in cyc5.values() if c != o]))
    print(f"样本期 σ(5m 振幅) = {sigma5:.4%}")

    rows = []
    for s in samples:
        ts = int(s["timestamp"])
        cid = ts // 300_000
        if cid not in cyc5:
            continue
        op = cyc5[cid][0]
        p = p1.get((ts // 60_000 - 1) * 60_000)
        if p is None or op <= 0:
            continue
        tau = (cid * 300_000 + 300_000 - ts) / 300_000
        if tau <= 0.03:
            continue
        z = (p / op - 1) / (sigma5 * tau ** 0.5)
        rows.append((0 if tau < 0.5 else 1, z, float(s["down_price"]), cid))
    print(f"对齐样本 {len(rows)}")

    # 定价 & 频率（每周期每格子只取首个样本，避免重复计数）
    price_tab: dict[tuple, list] = {}
    freq_seen: dict[tuple, dict] = {}
    for tg, z, dp, cid in rows:
        key = (tg, zbin(z))
        price_tab.setdefault(key, []).append(dp)
        freq_seen.setdefault(key, {}).setdefault(cid, (z, None))
    down_of = {c: (cyc5[c][1] < cyc5[c][0]) if cyc5[c][1] != cyc5[c][0] else None for c in cyc5}

    print("\n===== A. 5m 市场 z 空间定价（price=DOWN中位价 | freq=实际收阴 | n周期）=====")
    print("格式: τ<0.5 [price|freq|n]  |  τ≥0.5 [price|freq|n]   （z>0 = 当前上涨态）")
    surf = {}
    for zi in range(8):
        cells = []
        for tg in (0, 1):
            key = (tg, zi)
            ps = price_tab.get(key, [])
            cf = freq_seen.get(key, {})
            downs = [down_of[c] for c in cf if down_of.get(c) is not None]
            if len(ps) >= 15 and len(downs) >= 15:
                pr = float(np.median(ps))
                fr = float(np.mean(downs))
                cells.append(f"{pr:.3f}|{fr:.3f}|{len(downs):4d}")
                surf[key] = {"price": pr, "freq": fr}
            else:
                cells.append(f"{'-':>18}")
        print(f"  {Z_NAMES[zi]:>13}: {cells[0]}  |  {cells[1]}")

    zc = np.array([(Z_EDGES[i] + Z_EDGES[i + 1]) / 2 for i in range(7)] + [4.8])
    zc = np.concatenate([[Z_EDGES[0] - 0.8], zc[:-1]])  # 8 个桶中心

    def e_down(z: float, tg: int) -> float:
        xs, ys = [], []
        for zi in range(8):
            cell = surf.get((tg, zi))
            if cell:
                xs.append(zc[zi])
                ys.append(cell["price"])
        return float(np.interp(z, xs, ys)) if xs else 0.5

    # ---------- B. 180 天场景事件 ----------
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - DAYS * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    o5, h5, l5, cl5, v5 = (np.array([r[i] for r in c5]) for i in (1, 2, 3, 4, 5))

    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list = sorted(c for c in buckets if len(buckets[c]) == 3 and (c + 1) * 900_000 <= now_ms)
    cyc_set = set(cyc_list)
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([c5[buckets[c][0]][1] for c in cyc_list])
    h15 = np.array([max(c5[i][2] for i in buckets[c]) for c in cyc_list])
    l15 = np.array([min(c5[i][3] for i in buckets[c]) for c in cyc_list])
    c15 = np.array([c5[buckets[c][-1]][4] for c in cyc_list])
    v15 = np.array([sum(c5[i][5] for i in buckets[c]) for c in cyc_list])
    dir15 = np.sign(c15 - o15)
    rng15 = np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / rng15
    sigma15 = float(np.std([(c - o) / o for o, c in zip(o15, c15) if c != o]))
    print(f"\n180天 σ(15m 振幅) = {sigma15:.4%}（≈σ5×√3 = {sigma5 * 3 ** 0.5:.4%}）")

    cs = np.concatenate([[0.0], np.cumsum(v15)])
    mv = np.full(N, np.nan)
    mv[19:] = (cs[20:] - cs[:-20]) / 20
    vratio = v15 / np.concatenate([[np.nan], mv[:-1]])

    from numpy.lib.stride_tricks import sliding_window_view
    rm = np.full(len(c5), np.nan)
    rmin = np.full(len(c5), np.nan)
    rm[LOOKBACK - 1:] = sliding_window_view(cl5, LOOKBACK).max(axis=1)
    rmin[LOOKBACK - 1:] = sliding_window_view(cl5, LOOKBACK).min(axis=1)
    lvl_hi = np.concatenate([[np.nan], rm[:-1]])
    lvl_lo = np.concatenate([[np.nan], rmin[:-1]])
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi = np.zeros(N, dtype=bool)
    broke_lo = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in buckets[cyc]:
            if cont[i] and i >= LOOKBACK:
                if h5[i] > lvl_hi[i] * (1 + EPS):
                    broke_hi[j] = True
                if l5[i] < lvl_lo[i] * (1 - EPS):
                    broke_lo[j] = True

    pool1 = broke_hi & (dir15 > 0) & (close_pos >= 0.85)
    pool2 = broke_lo & (dir15 < 0) & (vratio >= 2)

    s15 = sigma15
    events = {1: [], 2: [], 0: []}  # (z5, z10, hi12, lo12, next_red)
    for j in range(N - 1):
        nxt = cyc_arr[j] + 1
        if cyc_arr[j + 1] != nxt or nxt not in cyc_set:
            continue
        idxs = buckets[int(nxt)]
        op = c5[idxs[0]][1]
        if op <= 0 or c15[j + 1] == o15[j + 1]:
            continue
        d1, d2 = c5[idxs[0]][4] / op - 1, c5[idxs[1]][4] / op - 1
        rec = (d1 / (s15 * (10 / 15) ** 0.5), d2 / (s15 * (5 / 15) ** 0.5),
               max(c5[idxs[0]][2], c5[idxs[1]][2]) / op - 1,
               min(c5[idxs[0]][3], c5[idxs[1]][3]) / op - 1,
               bool(c15[j + 1] < o15[j + 1]))
        events[0].append(rec)
        if pool1[j]:
            events[1].append(rec)
        if pool2[j]:
            events[2].append(rec)
    print(f"事件池：场景① {len(events[1])} | 场景② {len(events[2])} | 全体 {len(events[0])}")

    rng_np = np.random.default_rng(7)

    def strat(name, pool, pick):
        filled, wins, es = 0, [], []
        for r in pool:
            out = pick(r)
            if out is None:
                continue
            e, is_down = out
            filled += 1
            es.append(e)
            wins.append(r[4] if is_down else not r[4])
        if filled < 20:
            print(f"  {name}: 成交不足 {filled}")
            return
        p = sum(wins) / filled
        e_avg = float(np.mean(es))
        evf = ev(p, e_avg)
        lo, hi = np.percentile(rng_np.binomial(filled, p, size=3000) / filled, [2.5, 97.5])
        print(f"  {name}: 成交 {filled}/{len(pool)} ({filled / len(pool):.0%}) | 均价 {e_avg:.3f} | "
              f"胜率 {p:.1%} [{lo:.1%},{hi:.1%}] | EV/成交 {evf:+.3f} | EV/事件 {evf * filled / len(pool):+.3f}")

    # 场景①：买 DOWN（z>0 涨态便宜）
    print("\n===== 场景① → 次周期买DOWN（n=%d）=====" % len(events[1]))
    pool = events[1]
    for zmin, tag in ((0.5, "z≥0.5"), (1.0, "z≥1.0"), (2.0, "z≥2.0")):
        strat(f"S5  [{tag}] t=5min 涨态买", pool,
              lambda r, zm=zmin: (e_down(r[0], 1), True) if r[0] >= zm else None)
        strat(f"S10 [{tag}] t=10min 涨态买", pool,
              lambda r, zm=zmin: (e_down(r[1], 0), True) if r[1] >= zm else None)
    for th in (0.0005, 0.0010):
        zt = th / (s15 * 0.55 ** 0.5)
        strat(f"T{int(th * 10000)} 触价+{th * 100:.2f}%即买", pool,
              lambda r, zz=zt, th_=th: (e_down(zz, 1), True) if r[2] >= th_ else None)
    strat("S0  开盘即买@0.50", pool, lambda r: (0.50, True))

    # 场景②：买 UP（z<0 跌态便宜）e_up = 1 - e_down(-z) - 0.02
    print("\n===== 场景② → 次周期买UP（n=%d）=====" % len(events[2]))
    pool = events[2]
    for zmin, tag in ((0.5, "z≤-0.5"), (1.0, "z≤-1.0")):
        strat(f"S5  [{tag}] t=5min 跌态买UP", pool,
              lambda r, zm=zmin: (1 - e_down(-r[0], 1) - 0.02, False) if r[0] <= -zm else None)
        strat(f"S10 [{tag}] t=10min 跌态买UP", pool,
              lambda r, zm=zmin: (1 - e_down(-r[1], 0) - 0.02, False) if r[1] <= -zm else None)
    for th in (0.0005,):
        zt = th / (s15 * 0.55 ** 0.5)
        strat(f"T05 触价-{th * 100:.2f}%即买UP", pool,
              lambda r, zz=zt, th_=th: (1 - e_down(zz, 1) - 0.02, False) if r[3] <= -th_ else None)
    strat("S0  开盘即买@0.50", pool, lambda r: (0.50, False))

    # 对照：全体
    print("\n===== 对照：全体周期（无场景信息）=====")
    pool = events[0]
    strat("S10 [z≥1.0] t=10min 涨态买DOWN", pool,
          lambda r: (e_down(r[1], 0), True) if r[1] >= 1.0 else None)
    strat("S10 [z≥2.0] t=10min 涨态买DOWN", pool,
          lambda r: (e_down(r[1], 0), True) if r[1] >= 2.0 else None)
    strat("S0  开盘即买@0.50", pool, lambda r: (0.50, True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
