#!/usr/bin/env python3
"""用户定义扫描："长上影实体下跌K" = 情绪反转 + 收盘零反抗。

用户澄清：不仅上影长（情绪反转已发生），且收盘时基本没有反抗力量
（收在整根K最低区，下影≈0，光脚阴线收盘）。

量化定义（阴系）：
  1) 收阴 body < 0
  2) 长上影（情绪反转）: upper/rng >= wick_min
  3) 收盘零反抗: lower/rng <= close_eps
  4) 实体 >= 15% 振幅（排除十字）
阳系镜像：收阳 + 长下影 + 收在最高区（上影≈0）。

数据：官方 15m klines 最近 180 天（~1.7万根）。
检验：形态 → 下一根 15m K 方向延续（收阴/收阳）。
打平线：@0.50 入场，费 2% + 溢价 0.01 → 52.0%。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np

FEE = 0.02
PREMIUM = 0.01
DAYS = 180
BODY_FRAC = 0.15  # 实体至少占振幅15%
API = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        batch = None
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


def ev_at(p: float) -> float:
    return p * ((1 - FEE) / (0.50 + PREMIUM) - 1.0) - (1 - p)


def feats(o: float, h: float, l: float, c: float):
    rng = h - l
    if rng <= 0 or o <= 0:
        return None
    return (c - o) / o, (h - max(o, c)) / o, (min(o, c) - l) / o, rng / o


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("15m", now_ms - DAYS * 86_400_000, now_ms)
    ks = {}
    for k in kl:
        cyc = int(k[0]) // 900_000
        if (cyc + 1) * 900_000 > now_ms:
            continue  # 剔除未收盘
        ks[cyc] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
    cycles = sorted(ks)
    print(f"官方 15m klines {len(cycles)} 根（"
          f"{time.strftime('%Y-%m-%d', time.gmtime(cycles[0] * 900))} ~ "
          f"{time.strftime('%Y-%m-%d', time.gmtime(cycles[-1] * 900))}）")

    directed = [c for c in cycles if ks[c][3] != ks[c][0]]
    base_down = sum(1 for c in directed if ks[c][3] < ks[c][0]) / len(directed)
    red_next = [c for c in cycles[:-1] if ks[c][3] < ks[c][0]]
    print(f"基准：15m 收阴 {base_down:.1%} | 打平 52.0%\n")

    rng_np = np.random.default_rng(7)

    def run(label: str, match_fn, want_down: bool, monthly: bool = False):
        wins: list[tuple[int, bool]] = []
        for i in range(len(cycles) - 1):
            c0, c1 = cycles[i], cycles[i + 1]
            if c1 != c0 + 1:
                continue
            f = feats(*ks[c0])
            if f is None or not match_fn(*f):
                continue
            o1, _, _, cl1 = ks[c1]
            if cl1 == o1:
                continue
            wins.append((c0, (cl1 < o1) == want_down))
        n = len(wins)
        if n < 5:
            print(f"  {label}: 样本不足 {n}")
            return
        hits = sum(1 for _, w in wins if w)
        p = hits / n
        lo, hi = np.percentile(rng_np.binomial(n, p, size=5000) / n, [2.5, 97.5])
        mark = "✅超打平" if lo > 0.52 else ("❌显著低于" if hi < 0.52 else "~区间内")
        print(f"  {label}: {hits}/{n} = {p:.1%} [{lo:.1%},{hi:.1%}] "
              f"EV@0.50 {ev_at(p):+.3f} {mark}")
        if monthly:
            buckets: dict[str, list[bool]] = {}
            for c0, w in wins:
                buckets.setdefault(time.strftime("%Y-%m", time.gmtime(c0 * 900)), []).append(w)
            for m in sorted(buckets):
                b = buckets[m]
                print(f"      {m}: {sum(b)}/{len(b)} = {sum(b) / len(b):.1%}")

    def bear(wick_min: float, close_eps: float):
        return lambda b, u, l, r: (b < 0 and u >= wick_min * r and l <= close_eps * r
                                   and abs(b) >= BODY_FRAC * r)

    def bull(wick_min: float, close_eps: float):
        return lambda b, u, l, r: (b > 0 and l >= wick_min * r and u <= close_eps * r
                                   and abs(b) >= BODY_FRAC * r)

    print("===== 用户定义网格：阴系（长上影+光脚收最低 → 次根收阴）=====")
    for close_eps in (0.05, 0.10, 0.15):
        for wick_min in (0.30, 0.40, 0.50):
            run(f"上影≥{wick_min:.0%}振幅 下影≤{close_eps:.0%}振幅",
                bear(wick_min, close_eps), True)
        print()

    print("===== 用户定义网格：阳系（长下影+光头收最高 → 次根收阳）=====")
    for close_eps in (0.05, 0.10, 0.15):
        for wick_min in (0.30, 0.40, 0.50):
            run(f"下影≥{wick_min:.0%}振幅 上影≤{close_eps:.0%}振幅",
                bull(wick_min, close_eps), False)
        print()

    print("===== 单因子隔离（阴系，全部收阴K为池）=====")
    run("收阴全部（无形态要求）→ 次根收阴", lambda b, u, l, r: b < 0, True)
    run("仅收盘零反抗（下影≤10%）→ 次根收阴", lambda b, u, l, r: b < 0 and l <= 0.10 * r, True)
    run("仅长上影（上影≥40%，不限收盘位置）→ 次根收阴",
        lambda b, u, l, r: b < 0 and u >= 0.40 * r and abs(b) >= BODY_FRAC * r, True)
    run("收盘零反抗+长上影（用户定义 40%/10%）按月 →",
        bear(0.40, 0.10), True, monthly=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
