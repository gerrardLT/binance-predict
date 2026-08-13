#!/usr/bin/env python3
"""市场状态（regime）拆解验证：假突破信号的方向不对称是形态本身，还是行情驱动？

用户假设：当前是空头市场，所以"冲高回落卖跌"强、"跌破收回买涨"弱。
若成立，则：多头段里方向应反转——跌破收回买涨变强、冲高回落卖跌变弱。

方法：
- 每个窗口的 regime = 该窗口开盘价 vs 24 小时前开盘价的涨跌幅
  BEAR：< -0.3%；BULL：> +0.3%；FLAT：介于两者
- 在各 regime 内分别统计「连续3窗 + 假突破」两方向的胜率/费后 EV
- 对照：无附加条件的基线假突破同样分段

模拟口径与前两轮完全一致（支撑=2h 滚动极值、下周期开盘入场、
sign(return) 结算、费 2% + 溢价 0.01）。
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
WINDOW_MS = 300_000
DAY_MS = 24 * 3600 * 1000
LOOKBACK = 24  # 2 小时支撑/阻力
EPS = 0.0005
REGIME_THRESHOLD = 0.003  # 24h 涨跌 ±0.3% 为分界
MIN_BETS_FOR_CI = 8


def _window_vals(curve: list | None) -> list[float]:
    pts = [
        (p.get("t", 0), p.get("v"))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return [float(v) for _, v in pts]


def _window_extremes(curve: list | None) -> tuple[float, float] | None:
    vals = [float(p["v"]) for p in (curve or []) if p.get("v") is not None]
    if len(vals) < 2:
        return None
    return min(vals), max(vals)


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


def _streak_ok(pairs: list[tuple[dict, dict]], idx: int, k: int, side: str) -> bool:
    for j in range(1, k + 1):
        if idx - j < 0:
            return False
        ret = pairs[idx - j][0].get("ret")
        if ret is None or float(ret) == 0.0:
            return False
        r = float(ret)
        if side == "down" and r >= 0:
            return False
        if side == "up" and r <= 0:
            return False
    return True


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(
                SentimentWindow.start_time, SentimentWindow.actual_return,
                SentimentWindow.outcome,
                SentimentWindow.curve_btc_price,
                SentimentWindow.curve_up_pct,
                SentimentWindow.curve_up_price,
                SentimentWindow.curve_down_price,
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {
            "start": r.start_time, "ret": r.actual_return, "outcome": r.outcome,
            "btc": r.curve_btc_price, "up_pct": r.curve_up_pct,
            "up_price": r.curve_up_price, "down_price": r.curve_down_price,
        }
        for r in rows
    ]
    print(f"窗口总数 {len(wins)}")

    pairs: list[tuple[dict, dict]] = []
    for a, b in zip(wins, wins[1:]):
        if b["start"] - a["start"] == WINDOW_MS:
            pairs.append((a, b))

    # 每个窗口开盘价（BTC 曲线首点）
    opens: list[float | None] = []
    for w in wins:
        vals = _window_vals(w["btc"])
        opens.append(vals[0] if vals else None)

    # 首尾价格看整体行情
    first_p = next((p for p in opens if p is not None), None)
    last_p = next((p for p in reversed(opens) if p is not None), None)
    if first_p and last_p:
        print(f"BTC 首尾价: {first_p:.0f} → {last_p:.0f} "
              f"({last_p / first_p - 1.0:+.2%}，{len(wins)} 窗口期间)")

    # 24h 收益：窗口开盘价 vs 24h 前最近窗口开盘价
    regimes: list[str] = []
    for i, w in enumerate(wins):
        o = opens[i]
        if o is None:
            regimes.append("NA")
            continue
        # 找 start_time <= w.start - 24h 的最大 idx
        j = i - 1
        while j >= 0 and wins[j]["start"] > w["start"] - DAY_MS:
            j -= 1
        base = opens[j] if j >= 0 else None
        if base is None or base <= 0:
            regimes.append("NA")
            continue
        r = o / base - 1.0
        regimes.append("BULL" if r > REGIME_THRESHOLD
                       else "BEAR" if r < -REGIME_THRESHOLD else "FLAT")

    from collections import Counter
    print(f"regime 分布（24h 涨跌，±{REGIME_THRESHOLD:.1%} 分界）: "
          f"{dict(Counter(r for r in regimes if r != 'NA'))}")

    # 分段验证：连续3窗+假突破 与 无附加基线
    for label, kind in (("连续3窗+假突破", "streak"), ("无附加基线", None)):
        print(f"\n===== {label}（支撑/阻力=2h）=====")
        for regime in ("BULL", "BEAR", "FLAT"):
            for side, direction in (("down", "UP"), ("up", "DOWN")):
                bets: list[tuple[bool, float]] = []
                for idx, (a, b) in enumerate(pairs):
                    if idx < LOOKBACK or idx >= len(regimes):
                        continue
                    if regimes[idx] != regime:
                        continue
                    hist = [pairs[j][0] for j in range(idx - LOOKBACK, idx)]
                    hist_ext = [e for e in (_window_extremes(h["btc"]) for h in hist) if e]
                    if len(hist_ext) < LOOKBACK // 2:
                        continue
                    support = min(e[0] for e in hist_ext)
                    resistance = max(e[1] for e in hist_ext)
                    if support <= 0 or resistance <= 0 or resistance <= support:
                        continue

                    vals = _window_vals(a["btc"])
                    if len(vals) < 2:
                        continue
                    low, high, last = min(vals), max(vals), vals[-1]
                    if side == "down":
                        broke = low < support * (1.0 - EPS)
                        reclaimed = last >= support
                    else:
                        broke = high > resistance * (1.0 + EPS)
                        reclaimed = last <= resistance
                    if not (broke and reclaimed):
                        continue
                    if kind == "streak" and not _streak_ok(pairs, idx, 3, side):
                        continue

                    if b["ret"] is None or float(b["ret"]) == 0.0:
                        continue
                    b_ret = float(b["ret"])
                    win = (b_ret > 0) if direction == "UP" else (b_ret < 0)

                    price = _open_price_early(
                        b["down_price"] if direction == "DOWN" else b["up_price"], b["start"]
                    )
                    if price is None:
                        p = _open_price_early(b["up_pct"], b["start"])
                        if p is None:
                            continue
                        price = p / 100.0 if direction == "UP" else max(1.0 - p / 100.0, 0.01)
                    bets.append((win, price))

                if not bets:
                    print(f"  {regime:>4} {side:>4}→{direction}: 0 注")
                    continue
                n = len(bets)
                win_rate = sum(1 for w, _ in bets if w) / n
                pnls = [_bet_pnl(w, p) for w, p in bets]
                ev = float(np.mean(pnls))
                if n >= MIN_BETS_FOR_CI:
                    rng = np.random.default_rng(7)
                    ix = rng.integers(0, n, size=(2000, n))
                    ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
                    lo, hi = float(ci[0]), float(ci[1])
                else:
                    lo, hi = float("nan"), float("nan")
                print(
                    f"  {regime:>4} {side:>4}→{direction}: 注数 {n} 胜率 {win_rate:.1%} "
                    f"费后EV {ev:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
