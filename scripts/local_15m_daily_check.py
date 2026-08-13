#!/usr/bin/env python3
"""本地一个月数据验证：15 分钟周期 + 日线阻力假突破（与长周期口径一致）。

与 final_15m_check.py（长周期，LOOKBACK=96 档）完全同口径，仅数据源不同：
- 信号数据 = sentiment_windows 的 curve_btc_price（1m close 回填曲线）
- 入场价 = 下一 15m 窗口第一个 5m 窗口的真实 token 开盘价（up_price/down_price）
- 结算 = 15m 窗口 BTC 收益符号（c/o-1）

信号（收盘点位口径）：
- 3 个连续 5m 窗口聚合为一个 15m 窗口，closes = 拼接曲线
- 支撑/阻力 = 前 96 个 15m 窗口 closes 的 min/max（日线级别）
- 假突破：本窗口 closes 击穿支撑 0.05% 且窗口收盘收回 → 反向
- 连续 3 窗 = 前 3 个 15m 窗口实体同向（开收差）
- 逐注盈亏：赢 → (1-0.02)/(price+0.01)-1；输 → -1

用法：
    python scripts/local_15m_daily_check.py
"""
from __future__ import annotations

import asyncio
import json
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
WINDOW_MS = 300_000  # 5 分钟
LOOKBACK = 96  # 96 个 15m 窗口 = 24 小时（日线级别）
EPS = 0.0005
K = 3
MIN_BETS_FOR_CI = 8


def _curve_vals(curve: list | None) -> list[float]:
    pts = [
        (p.get("t", 0), p.get("v"))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return [float(v) for _, v in pts]


def _open_price_early(curve: list | None, start_ms: int) -> float | None:
    best, best_v = None, None
    for p in curve or []:
        if p.get("v") is None:
            continue
        rel = (p.get("t", 0) - start_ms) / 1000.0
        if rel < 0:
            continue
        if best is None or rel < best:
            best, best_v = rel, float(p["v"])
    return best_v


def _bet_pnl(win: bool, price: float) -> float:
    if not win:
        return -1.0
    return (1.0 - FEE) / min(max(price + PREMIUM, 0.01), 0.99) - 1.0


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(
                SentimentWindow.start_time, SentimentWindow.actual_return,
                SentimentWindow.curve_btc_price,
                SentimentWindow.curve_up_pct,
                SentimentWindow.curve_up_price,
                SentimentWindow.curve_down_price,
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {
            "start": r.start_time,
            "btc": r.curve_btc_price,
            "up_pct": r.curve_up_pct,
            "up_price": r.curve_up_price,
            "down_price": r.curve_down_price,
        }
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}，按时间排序")

    # 3 个连续 5m 窗口 → 1 个 15m 窗口（不重叠：每 3 个一组）
    agg: list[dict] = []
    for i in range(0, len(wins) - 2, 3):
        a, b, c = wins[i], wins[i + 1], wins[i + 2]
        if b["start"] - a["start"] != WINDOW_MS or c["start"] - b["start"] != WINDOW_MS:
            continue
        vals = _curve_vals(a["btc"]) + _curve_vals(b["btc"]) + _curve_vals(c["btc"])
        if len(vals) < 2:
            continue
        agg.append({
            "t": a["start"],
            "o": vals[0],
            "c": vals[-1],
            "closes": vals,
            "first": a,
        })
    print(f"聚合 15m 窗口 {len(agg)} 个")

    def stats(bets: list[tuple[bool, float, str]]) -> str:
        if not bets:
            return "0 注"
        n = len(bets)
        n_real = sum(1 for _, _, kk in bets if kk == "real")
        wr = sum(1 for w, _, _ in bets if w) / n
        pnls = [_bet_pnl(w, p) for w, p, _ in bets]
        ev = float(np.mean(pnls))
        if n >= MIN_BETS_FOR_CI:
            rng = np.random.default_rng(7)
            ix = rng.integers(0, n, size=(2000, n))
            ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
            return (f"注数 {n}(真实价 {n_real}) 胜率 {wr:.1%} "
                    f"费后EV {ev:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        return f"注数 {n}(真实价 {n_real}) 胜率 {wr:.1%} 费后EV {ev:+.3f}"

    def collect(side: str, direction: str, require_streak: bool = True) -> list[tuple[bool, float, str]]:
        bets: list[tuple[bool, float, str]] = []
        for idx, w in enumerate(agg):
            if idx <= LOOKBACK or idx >= len(agg) - 1:
                continue
            hist = agg[idx - LOOKBACK: idx]
            support = min(min(x["closes"]) for x in hist)
            resistance = max(max(x["closes"]) for x in hist)
            if side == "down":
                broke = min(w["closes"]) < support * (1.0 - EPS)
                reclaimed = w["c"] >= support
            else:
                broke = max(w["closes"]) > resistance * (1.0 + EPS)
                reclaimed = w["c"] <= resistance
            if not (broke and reclaimed):
                continue
            if require_streak:
                streak_ok = True
                for j in range(1, K + 1):
                    pw = agg[idx - j]
                    d = pw["c"] - pw["o"]
                    if side == "down" and d >= 0:
                        streak_ok = False
                        break
                    if side == "up" and d <= 0:
                        streak_ok = False
                        break
                if not streak_ok:
                    continue
            nxt = agg[idx + 1]
            d = nxt["c"] - nxt["o"]
            if d == 0:
                continue
            win = (d > 0) if direction == "UP" else (d < 0)
            first = nxt["first"]
            price = _open_price_early(
                first["down_price"] if direction == "DOWN" else first["up_price"],
                first["start"],
            )
            kind_p = "real"
            if price is None:
                p = _open_price_early(first["up_pct"], first["start"])
                if p is None:
                    continue
                price = p / 100.0 if direction == "UP" else max(1.0 - p / 100.0, 0.01)
                kind_p = "proxy"
            bets.append((win, price, kind_p))
        return bets

    print(f"\n===== 本地一个月 · 15m 周期 · 日线支撑/阻力（{LOOKBACK} 窗口） =====")
    for side, direction in (("down", "UP"), ("up", "DOWN")):
        bets = collect(side, direction)
        print(f"  {side:>4}→{direction}: {stats(bets)}")

    print("\n===== 对照基线（无连续窗条件，仅破位+收回） =====")
    for side, direction in (("down", "UP"), ("up", "DOWN")):
        bets = collect(side, direction, require_streak=False)
        print(f"  {side:>4}→{direction}: {stats(bets)}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
