#!/usr/bin/env python3
"""入场时机扫描：场景①②在次周期的哪个时刻、什么价格下注 EV 最高？

定价：真实报价曲面 pm_surface.json（5m 市场），d 轴按 √3 缩放到 15m 并线性插值。
EV = p×(0.98/(e+0.01)−1) − (1−p)。

输出：
  A. 条件状态表：场景①次周期在 t=5/10min 各涨跌状态下 → 收阴概率 | 曲面价 | EV
  B. 策略对比：S0 开盘买 / S5、S10 涨态等待 / T05、T10 触价即买（场景①② + 全体对照）
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
SQRT3 = 3 ** 0.5


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


def roll(x, w, fn):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    out[w - 1:] = getattr(np, fn)(sliding_window_view(x, w), axis=1)
    return out


def ev(p: float, e: float) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(ROOT, ".pytest_tmp", "pm_surface.json"), encoding="utf-8") as f:
        surface = json.load(f)

    D_KEYS = ["d<-0.08%", "-0.08~-0.02%", "-0.02~+0.02%", "+0.02~+0.08%", "d>+0.08%"]
    D_MID_15 = np.array([-0.0012, -0.0005, 0.0, 0.0005, 0.0012]) * SQRT3
    TAU_KEYS = ["<0.15", "0.15~0.35", "0.35~0.55", "0.55~0.75", "0.75~0.95"]

    def e_down(tau_key: str, d: float) -> float:
        row = [surface.get(f"{tau_key}|{k}") for k in D_KEYS]
        xs, ys = [], []
        for cell, x in zip(row, D_MID_15):
            if cell:
                xs.append(x)
                ys.append(cell["median_price"])
        return float(np.interp(d, xs, ys)) if xs else 0.5

    def e_up(tau_key: str, d: float) -> float:
        """d 为相对开盘涨跌幅（跌态 d<0 时 UP 便宜）。近似 up = 1 - down - 0.02价差。"""
        return 1.0 - e_down(tau_key, -d) - 0.02

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

    cs = np.concatenate([[0.0], np.cumsum(v15)])
    mv = np.full(N, np.nan)
    mv[19:] = (cs[20:] - cs[:-20]) / 20
    vratio = v15 / np.concatenate([[np.nan], mv[:-1]])

    rm, rmin = roll(cl5, LOOKBACK, "max"), roll(cl5, LOOKBACK, "min")
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

    events = {1: [], 2: [], 0: []}  # (d1, d2, hi12, lo12, next_red)
    for j in range(N - 1):
        nxt = cyc_arr[j] + 1
        if cyc_arr[j + 1] != nxt or nxt not in cyc_set:
            continue
        idxs = buckets[int(nxt)]
        op = c5[idxs[0]][1]
        if op <= 0 or c15[j + 1] == o15[j + 1]:
            continue
        rec = (c5[idxs[0]][4] / op - 1, c5[idxs[1]][4] / op - 1,
               max(c5[idxs[0]][2], c5[idxs[1]][2]) / op - 1,
               min(c5[idxs[0]][3], c5[idxs[1]][3]) / op - 1,
               bool(c15[j + 1] < o15[j + 1]))
        events[0].append(rec)
        if pool1[j]:
            events[1].append(rec)
        if pool2[j]:
            events[2].append(rec)
    print(f"事件池：场景① {len(events[1])} | 场景② {len(events[2])} | 全体对照 {len(events[0])}")

    rng_np = np.random.default_rng(7)
    EDGES = [-0.0014, -0.00035, 0.00035, 0.0014]
    BNAMES = ["深跌<-0.14%", "跌", "平±0.035%", "涨", "深涨>+0.14%"]

    def bucket_of(d: float) -> int:
        for i, (lo, hi) in enumerate(zip([-9] + EDGES, EDGES + [9])):
            if lo <= d < hi:
                return i
        return 2

    def cond_table(pool: list, t_idx: int, tau_key: str, down_bet: bool):
        t_name = "t=5min" if t_idx == 1 else "t=10min"
        print(f"\n  状态表 @{t_name}（买{'DOWN' if down_bet else 'UP'}）:")
        for bi in range(5):
            sub = [r for r in pool if bucket_of(r[t_idx]) == bi]
            if len(sub) < 15:
                print(f"    {BNAMES[bi]}: n={len(sub)} 不足")
                continue
            if down_bet:
                p = sum(r[4] for r in sub) / len(sub)
                e = e_down(tau_key, D_MID_15[bi] / SQRT3 * SQRT3)  # 桶中心价
            else:
                p = sum(not r[4] for r in sub) / len(sub)
                e = e_up(tau_key, D_MID_15[bi])
            fill = len(sub) / len(pool)
            print(f"    {BNAMES[bi]}: n={len(sub)}({fill:.0%}) | p={p:.1%} | e≈{e:.3f} | "
                  f"EV/成交 {ev(p, e):+.3f} | EV/事件 {fill * ev(p, e):+.3f}")

    def strat(name: str, pool: list, pick):
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
              f"胜率 {p:.1%} [{lo:.1%},{hi:.1%}] | EV/成交 {evf:+.3f} | EV/事件 {filled / len(pool) * evf:+.3f}")

    # ===== 场景①：买 DOWN =====
    print("\n===== 场景① 破4h高+光头阳 → 次周期买DOWN（n=%d）=====" % len(events[1]))
    pool = events[1]
    cond_table(pool, 1, "0.55~0.75", True)
    cond_table(pool, 2, "0.15~0.35", True)
    print()
    strat("S0   开盘即买@0.50", pool, lambda r: (0.50, True))
    strat("S5↑  t=5 涨态买(放弃跌/平)", pool,
          lambda r: (e_down("0.55~0.75", r[1]), True) if r[1] > 0.00035 else None)
    strat("S10↑ t=10 涨态买(放弃跌/平)", pool,
          lambda r: (e_down("0.15~0.35", r[2]), True) if r[2] > 0.00035 else None)
    strat("T05  触价+0.05%即买", pool,
          lambda r: (e_down("0.35~0.55", 0.0005), True) if r[2] >= 0.0005 or r[3] >= 0.0005 else None)
    strat("T10  触价+0.10%即买", pool,
          lambda r: (e_down("0.35~0.55", 0.0010), True) if r[2] >= 0.0010 else None)

    # ===== 场景②：买 UP =====
    print("\n===== 场景② 破4h低+收阴+放量 → 次周期买UP（n=%d）=====" % len(events[2]))
    pool = events[2]
    cond_table(pool, 1, "0.55~0.75", False)
    cond_table(pool, 2, "0.15~0.35", False)
    print()
    strat("S0   开盘即买@0.50", pool, lambda r: (0.50, False))
    strat("S5↓  t=5 跌态买UP(放弃涨/平)", pool,
          lambda r: (e_up("0.55~0.75", r[1]), False) if r[1] < -0.00035 else None)
    strat("S10↓ t=10 跌态买UP", pool,
          lambda r: (e_up("0.15~0.35", r[2]), False) if r[2] < -0.00035 else None)
    strat("T05↓ 触价-0.05%即买UP", pool,
          lambda r: (e_up("0.35~0.55", -0.0005), False) if r[3] <= -0.0005 else None)

    # ===== 对照：全体周期裸等待 =====
    print("\n===== 对照：全体周期裸等待买DOWN（无场景信息，n=%d）=====" % len(events[0]))
    pool = events[0]
    strat("S0   开盘即买@0.50", pool, lambda r: (0.50, True))
    strat("S5↑  t=5 涨态买", pool,
          lambda r: (e_down("0.55~0.75", r[1]), True) if r[1] > 0.00035 else None)
    strat("S10↑ t=10 涨态买", pool,
          lambda r: (e_down("0.15~0.35", r[2]), True) if r[2] > 0.00035 else None)
    strat("T05  触价+0.05%即买", pool,
          lambda r: (e_down("0.35~0.55", 0.0005), True) if r[2] >= 0.0005 else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
