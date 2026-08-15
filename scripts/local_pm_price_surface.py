#!/usr/bin/env python3
"""预测市场真实定价曲面：DOWN token 价格 = f(周期内涨跌幅 d, 剩余时间比例 τ)。

数据：prediction_market_samples.json（15s 间隔的真实报价快照，~22k 条）
     + 官方 1m klines（价格轴）+ 5m klines（周期开盘价/收盘方向）

产出：
  1. 定价表：τ × d 分桶 → down_price 中位数/样本数（等待策略的真实入场价）
  2. 有效性校验：down_price vs 同状态实际收阴频率（市场贵了还是便宜了）
  3. 曲面存 .pytest_tmp/pm_surface.json 供入场时机扫描用
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 1
        time.sleep(0.2)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    with open(os.path.join(ROOT, "prediction_market_samples.json"), encoding="utf-8") as f:
        samples = json.load(f)
    samples = [s for s in samples if s.get("down_price") is not None
               and 0.02 < float(s["down_price"]) < 0.98]
    ts_all = [int(s["timestamp"]) for s in samples]
    lo, hi = min(ts_all), max(ts_all)
    print(f"有效报价样本 {len(samples)}（{time.strftime('%m-%d %H:%M', time.gmtime(lo / 1000))}"
          f" ~ {time.strftime('%m-%d %H:%M', time.gmtime(hi / 1000))} UTC）")

    k1 = fetch_klines("1m", lo - 600_000, hi + 600_000)
    p1 = {int(k[0]): float(k[4]) for k in k1}  # 1m 收盘价（分钟键）
    k5 = fetch_klines("5m", lo - 900_000, hi + 900_000)
    cyc = {int(k[0]) // 300_000: (float(k[1]), float(k[4])) for k in k5}  # 桶号 → (open, close)
    print(f"1m klines {len(k1)} | 5m klines {len(k5)}")

    # 价格查找：ts 所在分钟的"最近已收盘 1m close"
    minutes = sorted(p1)

    def price_at(ts_ms: int) -> float | None:
        m = (ts_ms // 60_000 - 1) * 60_000  # 上一根已收盘的 1m
        return p1.get(m)

    rows = []  # (tau, d, down_price, cycle)
    for s in samples:
        ts = int(s["timestamp"])
        cid = ts // 300_000
        if cid not in cyc:
            continue
        op = cyc[cid][0]
        p = price_at(ts)
        if p is None or op <= 0:
            continue
        tau = (cid * 300_000 + 300_000 - ts) / 300_000  # 剩余时间比例
        rows.append((tau, p / op - 1, float(s["down_price"]), cid))
    print(f"可对齐样本 {len(rows)}")
    taus = np.array([r[0] for r in rows])
    ds = np.array([r[1] for r in rows])
    dps = np.array([r[2] for r in rows])
    print(f"d 分布(相对周期开盘): p5 {np.percentile(ds, 5):+.3%} p25 {np.percentile(ds, 25):+.3%} "
          f"p50 {np.percentile(ds, 50):+.3%} p75 {np.percentile(ds, 75):+.3%} p95 {np.percentile(ds, 95):+.3%}")

    tau_edges = [0.15, 0.35, 0.55, 0.75, 0.95]  # 剩余时间比例桶
    d_edges = [-0.0008, -0.0002, 0.0002, 0.0008]  # 5 桶: 深跌/跌/平/涨/深涨
    d_names = ["d<-0.08%", "-0.08~-0.02%", "-0.02~+0.02%", "+0.02~+0.08%", "d>+0.08%"]

    # 实际收阴频率 per 格子（用周期最终 close vs open）
    print("\n===== 定价表：down_price 中位数（真实市场报价）| 同状态实际收阴频率 =====")
    surface = {}
    header = "τ\\d     " + "".join(f"{n:>16}" for n in d_names)
    print(header)
    for ti in range(len(tau_edges) + 1):
        if ti == 0:
            tm = taus < tau_edges[0]
            tlabel = f"<{tau_edges[0]:.2f}"
        elif ti == len(tau_edges):
            tm = taus >= tau_edges[-1]
            tlabel = f">={tau_edges[-1]:.2f}"
        else:
            tm = (taus >= tau_edges[ti - 1]) & (taus < tau_edges[ti])
            tlabel = f"{tau_edges[ti - 1]:.2f}~{tau_edges[ti]:.2f}"
        cells = []
        for di in range(len(d_edges) + 1):
            if di == 0:
                dm = ds < d_edges[0]
            elif di == len(d_edges):
                dm = ds >= d_edges[-1]
            else:
                dm = (ds >= d_edges[di - 1]) & (ds < d_edges[di])
            m = tm & dm
            n = int(m.sum())
            if n >= 20:
                med = float(np.median(dps[m]))
                # 该状态下周期的最终收阴频率
                cids = {}
                for i in np.where(m)[0]:
                    cids[rows[i][3]] = None
                downs = []
                for c in cids:
                    o5, c5c = cyc[c]
                    if c5c != o5:
                        downs.append(c5c < o5)
                freq = float(np.mean(downs)) if downs else float("nan")
                cells.append(f"{med:.2f}|{freq:.2f}|{n}")
                surface[f"{tlabel}|{d_names[di]}"] = {
                    "median_price": med, "emp_freq": freq, "n": n,
                    "tau_mid": (tau_edges[ti - 1] + tau_edges[ti]) / 2 if 0 < ti < len(tau_edges)
                    else (tau_edges[0] / 2 if ti == 0 else (tau_edges[-1] + 1) / 2),
                    "d_mid": (d_edges[di - 1] + d_edges[di]) / 2 if 0 < di < len(d_edges)
                    else (d_edges[0] - 0.0004 if di == 0 else d_edges[-1] + 0.0004),
                }
            else:
                cells.append(f"{'-':>16}")
        print(f"{tlabel:>7} " + "".join(f"{c:>16}" for c in cells))

    print("\n（格式：down_price中位 | 实际收阴频率 | 样本数；τ=剩余时间比例）")
    out = os.path.join(ROOT, ".pytest_tmp", "pm_surface.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(surface, f, ensure_ascii=False, indent=1)
    print(f"曲面已存 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
