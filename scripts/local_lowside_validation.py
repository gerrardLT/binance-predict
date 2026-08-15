#!/usr/bin/env python3
"""低位侧候选补验：发现集上冒头的做多场景，在后60天验证集上复检。

候选（发现集结论）：
  L1 F16 收盘创24h新低 → 次根↑ 56.5%
  L2 F13×F29 破4h低·收阴×放量 → 次根↑ 58.2%
  L1 F10 破4h低位势 → 次根↑ 54.8%
  L1 F05 阴·收最低区(用户形态组件) → 次根↑ 52.6%
高位侧对照：F15 24h新高→↓ / F17 4h上沿→↓ / F20 连阳≥3→↓
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48
DAYS = 180
API = "https://data-api.binance.vision/api/v3/klines"


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
        time.sleep(0.25)
    return out


def roll_extreme(x, w, fn):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    out[w - 1:] = getattr(np, fn)(sliding_window_view(x, w), axis=1)
    return out


def ev_at(p: float) -> float:
    return p * ((1 - FEE) / (0.50 + PREMIUM) - 1.0) - (1 - p)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - DAYS * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    h5 = np.array([r[2] for r in c5])
    l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])
    v5 = np.array([r[5] for r in c5])

    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list = sorted(c for c in buckets if len(buckets[c]) == 3 and (c + 1) * 900_000 <= now_ms)
    N = len(cyc_list)
    o15 = np.array([c5[buckets[c][0]][1] for c in cyc_list])
    h15 = np.array([max(c5[i][2] for i in buckets[c]) for c in cyc_list])
    l15 = np.array([min(c5[i][3] for i in buckets[c]) for c in cyc_list])
    c15 = np.array([c5[buckets[c][-1]][4] for c in cyc_list])
    v15 = np.array([sum(c5[i][5] for i in buckets[c]) for c in cyc_list])
    cyc_arr = np.array(cyc_list)
    dir15 = np.sign(c15 - o15)
    rng15 = np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / rng15

    prev_max96 = np.full(N, np.nan)
    prev_min96 = np.full(N, np.nan)
    prev_max96[1:] = roll_extreme(c15, 96, "max")[:-1]
    prev_min96[1:] = roll_extreme(c15, 96, "min")[:-1]
    new_24h_lo = c15 < prev_min96
    new_24h_hi = c15 > prev_max96

    rm = roll_extreme(cl5, LOOKBACK, "max")
    rmin = roll_extreme(cl5, LOOKBACK, "min")
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_lo5 = l5 < np.concatenate([[np.nan], rmin[:-1]]) * (1 - EPS)
    broke_lo15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in buckets[cyc]:
            if cont[i] and i >= LOOKBACK and broke_lo5[i]:
                broke_lo15[j] = True

    cs = np.concatenate([[0.0], np.cumsum(v15)])
    mv_full = np.full(N, np.nan)
    mv_full[19:] = (cs[20:] - cs[:-20]) / 20
    vratio = v15 / np.concatenate([[np.nan], mv_full[:-1]])
    streak = np.ones(N)
    for i in range(1, N):
        if dir15[i] == dir15[i - 1] and dir15[i] != 0:
            streak[i] = streak[i - 1] + 1

    has_next = np.zeros(N, dtype=bool)
    nxt_down = np.zeros(N, dtype=bool)
    for j in range(N - 1):
        if cyc_arr[j + 1] == cyc_arr[j] + 1 and dir15[j + 1] != 0:
            has_next[j] = True
            nxt_down[j] = dir15[j + 1] < 0

    split = int(N * 2 / 3)
    rng_np = np.random.default_rng(7)

    def rep(name, m, want_up: bool, pool_end):
        mm = m[:pool_end] & has_next[:pool_end]
        n = int(mm.sum())
        if n < 30:
            print(f"  {name} |验证: 样本不足 {n}")
            return
        pd = float(nxt_down[:pool_end][mm].mean())
        p = 1 - pd if want_up else pd
        lo, hi = np.percentile(rng_np.binomial(n, p, size=3000) / n, [2.5, 97.5])
        mark = "✅" if lo > 0.52 else ("❌" if hi < 0.52 else "~")
        side = "次根↑" if want_up else "次根↓"
        print(f"  {name} |验证集: {n} 根 {side} {p:.1%} [{lo:.1%},{hi:.1%}] "
              f"EV@0.50 {ev_at(p):+.3f} {mark}")
        return n, p

    print("===== 低位侧候选（后60天验证集）=====")
    rep("收盘创24h新低 → ↑", new_24h_lo, True, N)
    rep("破4h低·收阴 × 放量 → ↑", broke_lo15 & (dir15 < 0) & (vratio >= 2), True, N)
    rep("破4h低位势 → ↑", broke_lo15, True, N)
    rep("阴·收最低区(用户形态) → ↑", (dir15 < 0) & (close_pos <= 0.15), True, N)
    print("===== 高位侧对照（后60天验证集）=====")
    rep("收盘创24h新高 → ↓", new_24h_hi, False, N)
    w16_hi = roll_extreme(c15, 16, "max")
    rep("4h区间上沿 → ↓", c15 >= w16_hi - 0.1 * (roll_extreme(c15, 16, "max") - roll_extreme(c15, 16, "min")), False, N)
    rep("连阳≥3 → ↓", (dir15 > 0) & (streak >= 3), False, N)

    print("\n按月（收盘创24h新低 → 次根↑）:")
    months: dict[str, list[bool]] = {}
    for j in range(N):
        if new_24h_lo[j] and has_next[j]:
            months.setdefault(time.strftime("%Y-%m", time.gmtime(cyc_arr[j] * 900)),
                              []).append(not nxt_down[j])
    for m in sorted(months):
        b = months[m]
        print(f"  {m}: {sum(b)}/{len(b)} = {sum(b) / len(b):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
