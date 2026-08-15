#!/usr/bin/env python3
"""金标准复验：全部用币安官方 klines 重跑三个命题（规避采样欠采样/缺口影响）。

数据：官方 5m klines（level 与生产一致：最近 48 个 5m 收盘价极值）
      官方 15m K 由 5m klines 精确聚合（open/high/low/close 无损）
事件：5m 蜡烛 high/low 破 4h 位势（官方极值，能抓到采样漏掉的快速针刺）

复验：
1. 模式一（无位势条件）：官方长上影实体下跌K → 下根收阴？
2. 融合：破4h阻力 → 次15m周期延续DOWN？
3. 拆分：信号周期收阳/收阴 → 次周期方向？
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

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48
API = "https://data-api.binance.vision/api/v3/klines"

# 形态参数（与 local_candle_pattern_check 一致）
MIN_RANGE_PCT = 0.04
WICK_BODY_RATIO = 2.0
WICK_RANGE_MIN = 0.35
BODY_RANGE_MIN = 0.15


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as resp:
            batch = json.loads(resp.read().decode())
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 1
        time.sleep(0.3)
    return out


def ev_at(p: float, e: float) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def classify(o: float, h: float, l: float, c: float) -> str | None:
    rng = h - l
    if rng <= 0 or rng / o * 100 < MIN_RANGE_PCT:
        return None
    body = c - o
    upper = h - max(o, c)
    lower = min(o, c) - l
    if body < 0 and upper >= WICK_BODY_RATIO * abs(body) and upper >= WICK_RANGE_MIN * rng and abs(body) >= BODY_RANGE_MIN * rng:
        return "bear_reject"
    if body > 0 and lower >= WICK_BODY_RATIO * abs(body) and lower >= WICK_RANGE_MIN * rng and abs(body) >= BODY_RANGE_MIN * rng:
        return "bull_reject"
    return None


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        lo = (await session.execute(select(func.min(SentimentWindow.start_time)))).scalar()
        hi = (await session.execute(select(func.max(SentimentWindow.start_time)))).scalar()
    print(f"取官方 klines 覆盖 {int(lo)} ~ {int(hi) + 300_000}")

    k5 = fetch_klines("5m", int(lo), int(hi) + 300_000)
    print(f"官方 5m klines: {len(k5)} 根")
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in k5]
    closes5 = [c[4] for c in c5]
    times5 = [c[0] for c in c5]

    # 官方 15m K（由 5m 精确聚合）
    agg: dict[int, list] = {}
    for t, o, h, l, c in c5:
        agg.setdefault(t // 900_000, []).append((t, o, h, l, c))
    k15: dict[int, tuple[float, float, float, float]] = {}
    for cyc, ws in agg.items():
        if len(ws) != 3:
            continue
        ws.sort()
        k15[cyc] = (ws[0][1], max(w[2] for w in ws), min(w[3] for w in ws), ws[2][4])
    print(f"完整官方 15m K: {len(k15)} 根")

    cyc_order = sorted(k15)

    # ---- 1. 模式一：官方形态 → 下根方向 ----
    print("\n===== 命题1：官方K形态 → 下一根方向（@0.50，打平52.0%）=====")
    n_dir = sum(1 for c in cyc_order if k15[c][3] != k15[c][0])
    base_down = sum(1 for c in cyc_order if k15[c][3] < k15[c][0]) / n_dir
    for pat, want_down, label in (
        ("bear_reject", True, "长上影实体下跌→下根收阴"),
        ("bull_reject", False, "长下影实体上涨→下根收阳"),
    ):
        hits, total = 0, 0
        for i in range(len(cyc_order) - 1):
            c0, c1 = cyc_order[i], cyc_order[i + 1]
            if c1 != c0 + 1:
                continue
            o, h, l, cl = k15[c0]
            if classify(o, h, l, cl) != pat:
                continue
            o1, _, _, cl1 = k15[c1]
            if cl1 == o1:
                continue
            total += 1
            hits += 1 if (cl1 < o1) == want_down else 0
        p = hits / total if total else 0.0
        lo_ci, hi_ci = np.percentile(
            np.random.default_rng(7).binomial(total, p, size=5000) / total, [2.5, 97.5]
        ) if total else (0, 0)
        print(f"  {label}: {hits}/{total} = {p:.1%} [{lo_ci:.1%},{hi_ci:.1%}] @0.50 EV {ev_at(p, 0.5):+.3f} | 基准 {base_down if want_down else 1 - base_down:.1%}")

    # ---- 2+3. 破位事件 → 次周期方向 / 按信号周期收向拆分 ----
    print("\n===== 命题2：破4h位势 → 次周期延续方向 =====")
    groups = {"high 全部": [], "low 全部": [], "high·收阳": [], "high·收阴": [], "high·收阴+长上影": []}
    for idx in range(LOOKBACK, len(c5)):
        t, o, h, l, cl = c5[idx]
        hist = closes5[idx - LOOKBACK: idx]
        if len(hist) < LOOKBACK // 2:
            continue
        cyc = t // 900_000
        nxt_cyc = cyc + 1
        if cyc not in k15 or nxt_cyc not in k15:
            continue
        _, _, _, sig_close = k15[cyc]
        sig_open = k15[cyc][0]
        nxt_open, _, _, nxt_close = k15[nxt_cyc]
        if nxt_close == nxt_open:
            continue
        if h > max(hist) * (1 + EPS):
            win = nxt_close < nxt_open
            groups["high 全部"].append(win)
            if sig_close > sig_open:
                groups["high·收阳"].append(win)
            else:
                groups["high·收阴"].append(win)
                body = abs(sig_close - sig_open)
                upper = k15[cyc][1] - max(sig_open, sig_close)
                if body > 0 and upper >= 2 * body:
                    groups["high·收阴+长上影"].append(win)
        elif l < min(hist) * (1 - EPS):
            groups["low 全部"].append(nxt_close > nxt_open)
    for name, res in groups.items():
        n = len(res)
        if n < 5:
            print(f"  {name}: 样本不足 {n}")
            continue
        p = sum(res) / n
        lo_ci, hi_ci = np.percentile(
            np.random.default_rng(7).binomial(n, p, size=5000) / n, [2.5, 97.5]
        )
        print(f"  {name}: {sum(res)}/{n} = {p:.1%} [{lo_ci:.1%},{hi_ci:.1%}] @0.50 EV {ev_at(p, 0.5):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
