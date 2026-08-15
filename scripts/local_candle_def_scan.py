#!/usr/bin/env python3
"""形态定义敏感性扫描："长上影实体下跌K"到底该怎么定义？

质疑：此前测试用的 D1 定义偏影线主导（上影≥2×实体、实体≥15%振幅），
可能偏离用户强调的"实体下跌K"本意。本扫描用 6 种定义（纯实体 → 影线主导）
在币安官方 15m klines 上测"形态确认 → 下一根 K 延续方向"的胜率，
看"无边缘"的结论是否随定义翻转。

打平线：@0.50 入场，费 2% + 溢价 0.01 → 52.0%。
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
API = "https://data-api.binance.vision/api/v3/klines"


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


def ev_at(p: float, e: float = 0.50) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def feats(o: float, h: float, l: float, c: float):
    """全部折算为相对开盘价的比例（0.0005 = 0.05%）。"""
    rng = h - l
    if rng <= 0 or o <= 0:
        return None
    body = (c - o) / o
    upper = (h - max(o, c)) / o
    lower = (min(o, c) - l) / o
    return body, upper, lower, rng / o


def make(min_rng, min_body, body_frac_rng, wick_min, wick_max=None):
    """生成阴/阳两个判定函数。wick_min/max 为影线相对实体倍数的下/上限。"""
    def bear(b, u, l, r):
        if r < min_rng or b > -min_body or abs(b) < body_frac_rng * r:
            return False
        if u < wick_min * abs(b):
            return False
        return wick_max is None or u <= wick_max * abs(b)

    def bull(b, u, l, r):
        if r < min_rng or b < min_body or abs(b) < body_frac_rng * r:
            return False
        if l < wick_min * abs(b):
            return False
        return wick_max is None or l <= wick_max * abs(b)

    return bear, bull


DEFS = [
    ("D0 纯实体(无影线要求) 实体≥0.05% 且≥50%振幅", *make(0.0004, 0.0005, 0.50, 0.0)),
    ("D1 影线主导(近似原定义) 影≥2×实体 实体≥15%振幅", *make(0.0004, 0.0, 0.15, 2.0)),
    ("D2 均衡 影≥实体 实体≥25%振幅且≥0.03%", *make(0.0004, 0.0003, 0.25, 1.0)),
    ("D3 实体主导 实体>影≥0.5×实体 实体≥0.04%", *make(0.0004, 0.0004, 0.40, 0.5, 1.0)),
    ("D4 均衡+中幅度 振幅≥0.06% 实体≥0.04%", *make(0.0006, 0.0004, 0.25, 1.0)),
    ("D5 均衡+大幅度 振幅≥0.10% 实体≥0.06%", *make(0.0010, 0.0006, 0.25, 1.0)),
]


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        lo = (await session.execute(select(func.min(SentimentWindow.start_time)))).scalar()
        hi = (await session.execute(select(func.max(SentimentWindow.start_time)))).scalar()
    kl = fetch_klines("15m", int(lo), int(hi) + 300_000)
    k15 = {int(k[0]) // 900_000: (float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl}
    # 剔除未收盘的最后一根
    now_ms = int(time.time() * 1000)
    cycles = sorted(c for c in k15 if (c + 1) * 900_000 <= now_ms)
    print(f"官方 15m klines {len(cycles)} 根（"
          f"{time.strftime('%m-%d %H:%M', time.gmtime(cycles[0] * 900))} ~ "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(cycles[-1] * 900))} UTC）")

    directed = [c for c in cycles if k15[c][3] != k15[c][0]]
    base_down = sum(1 for c in directed if k15[c][3] < k15[c][0]) / len(directed)
    print(f"基准：收阴 {base_down:.1%} / 收阳 {1 - base_down:.1%} | @0.50 打平 52.0%\n")

    rng = np.random.default_rng(7)
    for name, bear_fn, bull_fn in DEFS:
        print(f"--- {name} ---")
        for label, fn, want_down, base in (
            ("长上影实体下跌→次根收阴", bear_fn, True, base_down),
            ("长下影实体上涨→次根收阳", bull_fn, False, 1 - base_down),
        ):
            hits = total = 0
            for i in range(len(cycles) - 1):
                c0, c1 = cycles[i], cycles[i + 1]
                if c1 != c0 + 1:
                    continue
                f = feats(*k15[c0])
                if f is None or not fn(*f):
                    continue
                o1, _, _, cl1 = k15[c1]
                if cl1 == o1:
                    continue
                total += 1
                hits += 1 if (cl1 < o1) == want_down else 0
            if total < 5:
                print(f"  {label}: 样本不足 {total}")
                continue
            p = hits / total
            lo_ci, hi_ci = np.percentile(rng.binomial(total, p, size=5000) / total, [2.5, 97.5])
            mark = "✅超打平" if lo_ci > 0.52 else ("❌显著低于" if hi_ci < 0.52 else "~区间内")
            print(f"  {label}: {hits}/{total} = {p:.1%} [{lo_ci:.1%},{hi_ci:.1%}] "
                  f"EV@0.50 {ev_at(p):+.3f} | 基准 {base:.1%} {mark}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
