#!/usr/bin/env python3
"""回测数据审计：curve_btc_price 重建 15m K线的正确性验证。

1. 覆盖度：5m 窗口总数 vs 理论值，缺口清单
2. 分辨率：每窗口点数分布，btc_all 最大点间隔
3. 对齐：start_time 是否 5m 时钟对齐；15m 桶的窗口完整性
4. 金标准：重建 15m OHLC vs 币安官方 klines（随机抽样对比）
5. 连续性风险：跨越 15m 周期边界的缺口统计
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import SentimentWindow  # noqa: E402


def _sorted_pairs(curve):
    return sorted(
        (int(p.get("t", 0)), float(p["v"])) for p in (curve or []) if p.get("v") is not None
    )


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(SentimentWindow.start_time, SentimentWindow.curve_btc_price)
            .order_by(SentimentWindow.start_time)
        )).all()
        null_curve = (await session.execute(
            select(func.count(SentimentWindow.id)).where(SentimentWindow.curve_btc_price.is_(None))
        )).scalar()
        total_windows = (await session.execute(
            select(func.count(SentimentWindow.id))
        )).scalar()

    # 1. 覆盖度
    starts = [int(r.start_time) for r in rows]
    span_days = (starts[-1] - starts[0]) / 86_400_000
    expected = (starts[-1] - starts[0]) / 300_000 + 1
    print(f"库窗口总数 {total_windows}（curve_btc_price 为 NULL: {null_curve}）")
    print(f"取回带曲线窗口 {len(rows)} | 时间跨度 {span_days:.1f} 天 | 理论应有 {expected:.0f} 个 → 缺失 {expected - len(rows):.0f}")

    miss_align = sum(1 for s in starts if s % 300_000 != 0)
    print(f"时钟对齐检查：start_time 非 5m 对齐的窗口 {miss_align} 个")

    # 2. 分辨率与最大间隔
    lens = [len(_sorted_pairs(r.curve_btc_price)) for r in rows]
    arr = np.asarray(lens)
    print(f"每窗口点数: 中位 {np.median(arr):.0f} p5 {np.percentile(arr, 5):.0f} "
          f"最小 {arr.min()}（<6 点的窗口 {(arr < 6).sum()} 个）")
    btc_all: list[tuple[int, float]] = []
    for r in rows:
        btc_all.extend(_sorted_pairs(r.curve_btc_price))
    btc_all.sort()
    gaps = np.diff([t for t, _ in btc_all])
    big = sorted(range(len(gaps)), key=lambda i: -gaps[i])[:5]
    print("btc_all 最大 5 个采样间隔:")
    for i in big:
        print(f"  {gaps[i] / 1000:.0f}s @ {time.strftime('%m-%d %H:%M', time.gmtime(btc_all[i][0] / 1000))} UTC")

    # 3. 15m 桶完整性
    buckets: dict[int, int] = {}
    for s in starts:
        buckets[s // 900_000] = buckets.get(s // 900_000, 0) + 1
    full = sum(1 for v in buckets.values() if v == 3)
    partial = {k: v for k, v in buckets.items() if v < 3}
    print(f"15m 桶总数 {len(buckets)}：完整(3窗口) {full} | 不完整 {len(partial)}")

    # 跨 15m 边界的大缺口（>60s）
    boundary_gaps = []
    for i in range(len(gaps)):
        if gaps[i] > 60_000:
            t0, t1 = btc_all[i][0], btc_all[i + 1][0]
            if t0 // 900_000 != t1 // 900_000:
                boundary_gaps.append((gaps[i], t0))
    print(f">60s 缺口中跨越 15m 边界的: {len(boundary_gaps)} 个")

    # 4. 金标准：vs 币安官方 klines（取中段连续 192 个完整桶 ≈ 2 天，klines 单次可取全）
    full_keys = sorted(k for k, v in buckets.items() if v == 3)
    mid = len(full_keys) // 2
    sample = full_keys[max(0, mid - 96): mid + 96]
    t_lo, t_hi = sample[0] * 900_000, (sample[-1] + 1) * 900_000
    url = (f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m"
           f"&startTime={t_lo}&endTime={t_hi}&limit=1000")
    with urllib.request.urlopen(url, timeout=20) as resp:
        kl = json.loads(resp.read().decode())
    official = {int(k[0]) // 900_000: (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl}
    print(f"官方 klines 取回 {len(kl)} 根（覆盖样本桶 {sum(1 for c in sample if c in official)}/{len(sample)}）")

    d_open, d_close, d_high, d_low = [], [], [], []
    by_cyc: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_cyc.setdefault(int(r.start_time) // 900_000, []).extend(_sorted_pairs(r.curve_btc_price))
    for cyc in sample:
        if cyc not in official:
            continue
        pts = sorted(by_cyc.get(cyc, []))
        if len(pts) < 6:
            continue
        o, c = pts[0][1], pts[-1][1]
        h = max(p for _, p in pts)
        l = min(p for _, p in pts)
        oo, oh, ol, oc = official[cyc]
        d_open.append(abs(o - oo) / oo * 100)
        d_close.append(abs(c - oc) / oc * 100)
        d_high.append(abs(h - oh) / oh * 100)
        d_low.append(abs(l - ol) / ol * 100)

    def _s(x):
        if not x:
            return "无匹配样本"
        a = np.asarray(x)
        return f"中位 {np.median(a):.4f}% p90 {np.percentile(a, 90):.4f}% 最大 {a.max():.4f}% ({len(a)}根)"

    print(f"\n重建 OHLC vs 币安官方 klines（抽样 {len(d_open)} 根）:")
    print(f"  开盘偏差: {_s(d_open)}")
    print(f"  收盘偏差: {_s(d_close)}")
    print(f"  最高偏差: {_s(d_high)}（重建只能低于等于真实）")
    print(f"  最低偏差: {_s(d_low)}（重建只能高于等于真实）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
