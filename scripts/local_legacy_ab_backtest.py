#!/usr/bin/env python3
"""旧方案（4h 破位 + A/B 过滤 → 当周期行动）180 天重审回测。

背景：旧 A+B 方案 2026-08-15 退役（实盘 6 条 2 胜 4 负）。当时支撑它的回测
（local_combo_filter_lab.py）有三个缺陷：token 价用 5m 市场近似（自认"估值有偏"）、
样本仅数周（桶入选门槛 MIN_N=8）、无样本外验证。本脚本用场景①②同级的纪律重审：

  - 数据：180 天官方 5m K（data-api.binance.vision），前 120 天发现集 / 后 60 天验证集
  - 结算：周期锚点（当周期 P(E)=周期末 vs P(S)=周期开盘，与币安市场真实规则一致）
  - EV：z 空间定价曲面（prediction_market_samples.json 真实报价构建，对齐
    local_entry_timing_v2 的方法）+ 0.50 开盘价对照

旧方案定义（对齐线上 229a79c 版 fake_breakout_detector）：
  - 破位：5m high > 前 48 根 close max × 1.0005（low < min × 0.9995），每方向每周期首次
  - 过滤 A：信号 offset（周期内偏移秒数）< 360（6 分钟）
      5m bar 粒度近似：宽松档 = 破位 bar 起始偏移 <360（bar0/bar1 过），
      严格档 = 仅 bar0（起始偏移 0，必然 <360）过；另附按 bar 序号分解
  - 过滤 B：break_pct =（触发价 − 周期开盘）/ 开盘 < 0.2%（方向性取绝对偏离）
      触发价 = 位势线 × (1±EPS)（秒级触发瞬间刚越线，对齐线上语义）
  - 行动：破阻力 → 买 DOWN / 破支撑 → 买 UP，持有到当周期结束
"""
from __future__ import annotations

import json
import math
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


def wilson(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    ph = p + z * z / (2 * n)
    denom = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, (ph - half * denom) / denom), min(1.0, (ph + half * denom) / denom))


def build_surface() -> tuple[dict, float]:
    """5m 市场样本 → z 空间定价曲面（对齐 local_entry_timing_v2 的 A 部分）。"""
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

    price_tab: dict[tuple, list] = {}
    freq_seen: dict[tuple, dict] = {}
    for tg, z, dp, cid in rows:
        key = (tg, zbin(z))
        price_tab.setdefault(key, []).append(dp)
        freq_seen.setdefault(key, {}).setdefault(cid, (z, None))
    down_of = {c: (cyc5[c][1] < cyc5[c][0]) if cyc5[c][1] != cyc5[c][0] else None for c in cyc5}

    surf = {}
    for zi in range(8):
        for tg in (0, 1):
            key = (tg, zi)
            ps = price_tab.get(key, [])
            cf = freq_seen.get(key, {})
            downs = [down_of[c] for c in cf if down_of.get(c) is not None]
            if len(ps) >= 15 and len(downs) >= 15:
                surf[key] = {"price": float(np.median(ps)), "freq": float(np.mean(downs))}
    print(f"曲面就绪：{len(surf)} 个有效格子")
    return surf, sigma5


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("===== 旧方案（A+B 过滤）180 天重审 =====")
    print(f"方法论：官方 5m K | 前 120 天发现集 / 后 60 天验证集 | 周期锚点结算 | z 曲面定价\n")

    # ---------- A. 定价曲面 ----------
    surf, sigma5 = build_surface()
    zc = np.array([(Z_EDGES[i] + Z_EDGES[i + 1]) / 2 for i in range(7)] + [4.8])
    zc = np.concatenate([[Z_EDGES[0] - 0.8], zc[:-1]])

    def e_down(z: float, tg: int) -> float:
        xs, ys = [], []
        for zi in range(8):
            cell = surf.get((tg, zi))
            if cell:
                xs.append(zc[zi])
                ys.append(cell["price"])
        return float(np.interp(z, xs, ys)) if xs else 0.5

    # ---------- B. 180 天破位事件（对齐线上旧 detector 语义） ----------
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - DAYS * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    o5, h5, l5, cl5 = (np.array([r[i] for r in c5]) for i in (1, 2, 3, 4))

    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list = sorted(c for c in buckets if len(buckets[c]) == 3 and (c + 1) * 900_000 <= now_ms)
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([c5[buckets[c][0]][1] for c in cyc_list])
    c15 = np.array([c5[buckets[c][-1]][4] for c in cyc_list])
    sigma15 = float(np.std([(c - o) / o for o, c in zip(o15, c15) if c != o]))
    print(f"\n180天 σ(15m 振幅) = {sigma15:.4%}")

    from numpy.lib.stride_tricks import sliding_window_view
    rm = np.full(len(c5), np.nan)
    rmin = np.full(len(c5), np.nan)
    rm[LOOKBACK - 1:] = sliding_window_view(cl5, LOOKBACK).max(axis=1)
    rmin[LOOKBACK - 1:] = sliding_window_view(cl5, LOOKBACK).min(axis=1)
    lvl_hi = np.concatenate([[np.nan], rm[:-1]])
    lvl_lo = np.concatenate([[np.nan], rmin[:-1]])
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000

    # 事件：每周期每方向首次破位（bar 序号、触发价、offset、break_pct）
    events: list[dict] = []
    for j, cyc in enumerate(cyc_list):
        op = o15[j]
        if op <= 0 or c15[j] == op:
            continue
        settle_down = bool(c15[j] < op)
        month = time.strftime("%Y-%m", time.gmtime(cyc * 900))
        for side in ("high", "low"):
            for k, i in enumerate(buckets[cyc]):  # k = 0/1/2 → bar 起始 offset = k*300s
                if not cont[i] or i < LOOKBACK or np.isnan(lvl_hi[i]):
                    continue
                if side == "high" and h5[i] > lvl_hi[i] * (1 + EPS):
                    trig = lvl_hi[i] * (1 + EPS)
                elif side == "low" and l5[i] < lvl_lo[i] * (1 - EPS):
                    trig = lvl_lo[i] * (1 - EPS)
                else:
                    continue
                bp = abs(trig / op - 1)
                tau_rem = 1 - (k * 300 + 150) / 900  # bar 中点作 offset 点估计
                d = (trig / op - 1) * (1 if side == "high" else -1)  # 方向性偏离（涨为正）
                z = d / (sigma15 * max(tau_rem, 0.05) ** 0.5)
                price = (e_down(z, 0 if tau_rem >= 0.5 else 1) if side == "high"
                         else 1 - e_down(-z, 0 if tau_rem >= 0.5 else 1) - 0.02)
                events.append({
                    "j": j, "side": side, "bar": k, "bp": bp, "z": z,
                    "price": max(0.02, min(0.98, price)),
                    # side=high 买 DOWN：赢 ⟺ 当周期收阴；side=low 买 UP：赢 ⟺ 收阳
                    "win": settle_down if side == "high" else (not settle_down),
                    "month": month, "cyc": cyc,
                })
                break  # 每方向每周期首次
    print(f"破位事件（每方向每周期首次）：{len(events)}  "
          f"(high {sum(e['side'] == 'high' for e in events)} / low {sum(e['side'] == 'low' for e in events)})")

    split = int(N * 2 / 3)
    split_cyc = cyc_arr[split]
    for e in events:
        e["is_val"] = e["cyc"] >= split_cyc
    print(f"切分：发现集周期 ~{time.strftime('%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f" ~ {time.strftime('%m-%d', time.gmtime((split_cyc - 1) * 900))} | "
          f"验证集 {time.strftime('%m-%d', time.gmtime(split_cyc * 900))}"
          f" ~ {time.strftime('%m-%d', time.gmtime((cyc_arr[-1]) * 900))}")

    # ---------- C. 统计 ----------
    def report(name: str, sel: list[dict]) -> None:
        if not sel:
            print(f"  {name:<34} n=0")
            return
        n = len(sel)
        p = sum(e["win"] for e in sel) / n
        pr = float(np.mean([e["price"] for e in sel]))
        lo, hi = wilson(p, n)
        print(f"  {name:<34} n={n:>4} | 胜率 {p:.1%} [{lo:.1%},{hi:.1%}] | "
              f"入场均价 {pr:.3f} | EV曲面 {ev(p, pr):+.3f} | EV@0.50 {ev(p, 0.5):+.3f}")

    print("\n===== 发现集（前 120 天）=====")
    disc = [e for e in events if not e["is_val"]]
    report("全部破位（无过滤）", disc)
    report("  high 买DOWN", [e for e in disc if e["side"] == "high"])
    report("  low  买UP", [e for e in disc if e["side"] == "low"])
    for tag, a, b in (
        ("A宽松(offset<360: bar0/1)", lambda e: e["bar"] <= 1, lambda e: True),
        ("A严格(仅bar0)", lambda e: e["bar"] == 0, lambda e: True),
        ("B(break_pct<0.2%)", lambda e: True, lambda e: e["bp"] < 0.002),
        ("A宽松+B", lambda e: e["bar"] <= 1, lambda e: e["bp"] < 0.002),
        ("A严格+B", lambda e: e["bar"] == 0, lambda e: e["bp"] < 0.002),
    ):
        sub = [e for e in disc if a(e) and b(e)]
        report(f"{tag}", sub)

    print("\n===== 验证集（后 60 天，盲验）=====")
    val = [e for e in events if e["is_val"]]
    report("全部破位（无过滤）", val)
    report("  high 买DOWN", [e for e in val if e["side"] == "high"])
    report("  low  买UP", [e for e in val if e["side"] == "low"])
    for tag, a, b in (
        ("A宽松+B（旧方案口径·宽）", lambda e: e["bar"] <= 1, lambda e: e["bp"] < 0.002),
        ("A严格+B（旧方案口径·严）", lambda e: e["bar"] == 0, lambda e: e["bp"] < 0.002),
        ("仅A宽松", lambda e: e["bar"] <= 1, lambda e: True),
        ("仅B", lambda e: True, lambda e: e["bp"] < 0.002),
    ):
        sub = [e for e in val if a(e) and b(e)]
        report(f"{tag}", sub)

    print("\n===== 按月稳定性（A宽松+B，全 180 天）=====")
    ab = [e for e in events if e["bar"] <= 1 and e["bp"] < 0.002]
    months = sorted({e["month"] for e in ab})
    for m in months:
        report(m, [e for e in ab if e["month"] == m])

    print("\n===== 按 bar 序号分解（A 边界的直接证据，全 180 天）=====")
    for k in (0, 1, 2):
        report(f"bar{k}（offset {k * 300}~{k * 300 + 300}s）", [e for e in events if e["bar"] == k])

    print("\n===== 结论锚点 =====")
    print(f"  打平胜率@0.50 ≈ 52.0% | 曲面均价打平 ≈ 52%~56%（视入场价）")
    print(f"  参照：场景① 63.6% [59.3,68.0] / 场景② 57.8% [53.5,62.1]（验证集）")
    print(f"  旧方案实盘：6 条 2 胜 4 负（33%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
