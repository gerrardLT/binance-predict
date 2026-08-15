#!/usr/bin/env python3
"""融合命题检验：4h 位势破位（结构性条件）→ 下一个 15m 周期是否延续方向？

用户模式一/二无位势条件时胜率≈基准（无边缘，见 local_candle_pattern_check）。
本检验回答：把"长上影下跌K"换成"刺破 4h 阻力后收阴的K"（流动性收割已发生），
下一根 15m K 延续方向的概率是否 >52%（0.50 入场的打平线）。

事件定义与 local_combo_filter_lab 一致：窗口内极值破 48 窗口 closes 极值 ×(1±eps)。
下一周期方向：P(cyc15_end+900k) vs P(cyc15_end)。
入场价假设 0.50（下一周期开盘秒级 token ≈ 1:1，可执行性待实盘确认）。
"""
from __future__ import annotations

import asyncio
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
from sqlalchemy import select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import SentimentWindow  # noqa: E402

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48  # 4h


def _sorted_pairs(curve):
    pts = [(int(p.get("t", 0)), float(p["v"])) for p in (curve or []) if p.get("v") is not None]
    pts.sort()
    return pts


def _price_at(pairs, t_ms):
    if not pairs:
        return None
    if t_ms <= pairs[0][0]:
        return pairs[0][1]
    if t_ms >= pairs[-1][0]:
        return pairs[-1][1]
    for (t0, v0), (t1, v1) in zip(pairs, pairs[1:]):
        if t0 <= t_ms <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t_ms - t0) / (t1 - t0)
    return pairs[-1][1]


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(SentimentWindow.start_time, SentimentWindow.curve_btc_price)
            .order_by(SentimentWindow.start_time)
        )).all()

    wins = [{"btc": _sorted_pairs(r.curve_btc_price)} for r in rows]
    btc_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(w["btc"])
    btc_all.sort()
    t_max = btc_all[-1][0]
    closes = [w["btc"][-1][1] if w["btc"] else None for w in wins]
    print(f"5m 窗口 {len(wins)}")

    for side, side_cn in (("high", "破4h阻力→次周期延续DOWN"), ("low", "破4h支撑→次周期延续UP")):
        hits, total = 0, 0
        for idx, w in enumerate(wins):
            if idx <= LOOKBACK or len(w["btc"]) < 2:
                continue
            hist = [c for c in closes[idx - LOOKBACK: idx] if c is not None]
            if len(hist) < LOOKBACK // 2:
                continue
            if side == "high":
                ext_t, ext_v = max(w["btc"], key=lambda p: p[1])
                broke = ext_v > max(hist) * (1 + EPS)
            else:
                ext_t, ext_v = min(w["btc"], key=lambda p: p[1])
                broke = ext_v < min(hist) * (1 - EPS)
            if not broke:
                continue
            cyc_end = ext_t - (ext_t % 900_000) + 900_000
            nxt_end = cyc_end + 900_000
            if nxt_end > t_max:
                continue
            p0, p1 = _price_at(btc_all, cyc_end), _price_at(btc_all, nxt_end)
            if p0 is None or p1 is None or p1 == p0:
                continue
            total += 1
            hits += 1 if (p1 < p0) == (side == "high") else 0
        if total >= 5:
            p = hits / total
            lo, hi = np.percentile(
                np.random.default_rng(7).binomial(total, p, size=5000) / total, [2.5, 97.5]
            )
            ev = p * ((1 - FEE) / (0.50 + PREMIUM) - 1.0) - (1 - p)
            print(f"{side_cn}: {hits}/{total} = {p:.1%} [{lo:.1%},{hi:.1%}] | @0.50 EV {ev:+.3f}（打平 52.0%）")
        else:
            print(f"{side_cn}: 样本不足 {total}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
