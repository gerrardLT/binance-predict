#!/usr/bin/env python3
"""扩大样本 + 样本外验证：破4h+收阳→次周期DOWN 的边缘是不是31天的运气？

数据：官方 5m klines 最近 180 天（≈5.2万根），15m K 由 5m 精确聚合。
事件：5m 蜡烛 high/low 破最近 48 个 5m 收盘价极值 ×(1±0.05%)（与生产一致）。
去重：每个 15m 周期每侧只计首次破位（同一周期多根 5m 重复破位只算一次）。

判定规则（事先声明）：
  样本外 high·收阳→次周期DOWN 的 95%CI 下界 > 52% → 边缘可信，可落地；
  否则视为 31 天样本的运气，不落地。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict

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
DAYS = 180
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


def ev_at(p: float, e: float = 0.50) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def stat_line(name: str, wins: list[bool]) -> str:
    n = len(wins)
    if n < 5:
        return f"  {name}: 样本不足 {n}"
    p = sum(wins) / n
    lo, hi = np.percentile(
        np.random.default_rng(7).binomial(n, p, size=5000) / n, [2.5, 97.5]
    )
    mark = "✅" if lo > 0.52 else ("❌" if hi < 0.52 else "~")
    return (f"  {name}: {sum(wins)}/{n} = {p:.1%} [{lo:.1%},{hi:.1%}] "
            f"EV@0.50 {ev_at(p):+.3f} {mark}")


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        disc_lo = (await session.execute(select(func.min(SentimentWindow.start_time)))).scalar()
        disc_hi = (await session.execute(select(func.max(SentimentWindow.start_time)))).scalar()
    disc_lo, disc_hi = int(disc_lo), int(disc_hi) + 300_000
    print(f"发现窗口（本地库跨度）: {time.strftime('%m-%d %H:%M', time.gmtime(disc_lo / 1000))}"
          f" ~ {time.strftime('%m-%d %H:%M', time.gmtime(disc_hi / 1000))} UTC")

    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - DAYS * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()  # 剔除未收盘的最后一根 5m
    print(f"官方 5m klines {len(c5)} 根（{time.strftime('%Y-%m-%d', time.gmtime(c5[0][0] / 1000))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(c5[-1][0] / 1000))}）")
    closes5 = [c[4] for c in c5]
    times5 = [c[0] for c in c5]

    agg: dict[int, list] = {}
    for row in c5:
        agg.setdefault(row[0] // 900_000, []).append(row)
    k15: dict[int, tuple[float, float, float, float]] = {}
    for cyc, ws in agg.items():
        if len(ws) != 3 or (cyc + 1) * 900_000 > now_ms:
            continue
        ws.sort()
        k15[cyc] = (ws[0][1], max(w[2] for w in ws), min(w[3] for w in ws), ws[2][4])
    print(f"完整 15m K: {len(k15)} 根")

    directed = [c for c in k15 if k15[c][3] != k15[c][0]]
    base_down = sum(1 for c in directed if k15[c][3] < k15[c][0]) / len(directed)
    print(f"基准：15m 收阴 {base_down:.1%} | @0.50 打平 52.0%")

    # 事件提取
    events = []  # (cyc, side, sig_up, next_down, dup, sig_doji)
    seen: set[tuple[int, str]] = set()
    for idx in range(LOOKBACK, len(c5)):
        t, o, h, l, cl = c5[idx]
        if times5[idx - 1] != t - 300_000:
            continue  # 5m 不连续，位势失真
        if t - times5[idx - LOOKBACK] > (LOOKBACK + 4) * 300_000:
            continue  # 4h 窗口有空洞
        cyc = t // 900_000
        nxt = cyc + 1
        if cyc not in k15 or nxt not in k15:
            continue
        hist = closes5[idx - LOOKBACK:idx]
        broke_high = h > max(hist) * (1 + EPS)
        broke_low = l < min(hist) * (1 - EPS)
        if not (broke_high or broke_low):
            continue
        so, _, _, sc = k15[cyc]
        no, _, _, nc = k15[nxt]
        if nc == no:
            continue
        for side, broke in (("high", broke_high), ("low", broke_low)):
            if not broke:
                continue
            dup = (cyc, side) in seen
            seen.add((cyc, side))
            events.append((cyc, side, sc > so, nc < no, dup, sc == so))

    ded = [e for e in events if not e[4]]
    print(f"破位事件：去重后 {len(ded)}（原始 {len(events)}）")

    def wins_of(side: str, sig_up: bool | None, pool) -> list[bool]:
        out = []
        for e in pool:
            if e[1] != side:
                continue
            if sig_up is not None and (e[5] or e[2] != sig_up):
                continue
            out.append(e[3] if side == "high" else not e[3])
        return out

    print("\n===== 全样本 180 天（去重）=====")
    for name, side, sig_up in (
        ("high 全部→DOWN", "high", None),
        ("high·收阳→DOWN ★边缘", "high", True),
        ("high·收阴→DOWN", "high", False),
        ("low 全部→UP", "low", None),
        ("low·收阴→UP ★镜像", "low", False),
        ("low·收阳→UP", "low", True),
    ):
        print(stat_line(name, wins_of(side, sig_up, ded)))

    print("\n===== 发现窗口内 vs 样本外（去重）=====")
    in_pool = [e for e in ded if disc_lo <= e[0] * 900_000 <= disc_hi]
    out_pool = [e for e in ded if not (disc_lo <= e[0] * 900_000 <= disc_hi)]
    print(f"样本量：窗口内 {len(in_pool)} | 样本外 {len(out_pool)}")
    for name, side, sig_up in (
        ("high·收阳→DOWN ★", "high", True),
        ("high 全部→DOWN", "high", None),
        ("low·收阴→UP ★镜像", "low", False),
    ):
        print(stat_line(f"{name} | 窗口内", wins_of(side, sig_up, in_pool)))
        print(stat_line(f"{name} | 样本外", wins_of(side, sig_up, out_pool)))

    print("\n===== 按月稳定性：high·收阳→DOWN（去重）=====")
    monthly: dict[str, list[bool]] = defaultdict(list)
    for e in ded:
        if e[1] == "high" and not e[5] and e[2]:
            monthly[time.strftime("%Y-%m", time.gmtime(e[0] * 900))].append(e[3])
    for m in sorted(monthly):
        w = monthly[m]
        print(f"  {m}: {sum(w)}/{len(w)} = {sum(w) / len(w):.1%}")

    print("\n===== 与发现期口径可比（不去重）=====")
    print(stat_line("high·收阳→DOWN 原始", wins_of("high", True, events)))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
