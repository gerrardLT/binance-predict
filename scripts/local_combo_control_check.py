#!/usr/bin/env python3
"""对照组：区分"日线阻力假突破"信号 vs "买在局部高点"的均值回归优势。

逻辑：所有 5m 窗口（不论是否冲过日线阻力），取盘中高点时刻入场买 DOWN，
持有 15 分钟，统计方向胜率/token 回升/平均入场价。
再按"是否冲过日线阻力"分组，看信号有没有增量。

用法：
    python scripts/local_combo_control_check.py
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
LOOKBACK = 288
EPS = 0.0005
HOLD_MS = 900_000
MIN_BETS_FOR_CI = 8


def _sorted_pairs(curve: list | None) -> list[tuple[int, float]]:
    pts = [
        (int(p.get("t", 0)), float(p["v"]))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return pts


def _price_at(pairs: list[tuple[int, float]], t_ms: int) -> float | None:
    if not pairs:
        return None
    if t_ms <= pairs[0][0]:
        return pairs[0][1]
    if t_ms >= pairs[-1][0]:
        return pairs[-1][1]
    for (t0, v0), (t1, v1) in zip(pairs, pairs[1:]):
        if t0 <= t_ms <= t1:
            if t1 == t0:
                return v0
            w = (t_ms - t0) / (t1 - t0)
            return v0 + (v1 - v0) * w
    return pairs[-1][1]


def _bet_pnl(win: bool, price: float) -> float:
    if not win:
        return -1.0
    return (1.0 - FEE) / min(max(price + PREMIUM, 0.01), 0.99) - 1.0


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(
                SentimentWindow.start_time,
                SentimentWindow.curve_btc_price,
                SentimentWindow.curve_down_price,
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {"btc": r.curve_btc_price, "down": r.curve_down_price}
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    btc_all: list[tuple[int, float]] = []
    down_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(_sorted_pairs(w["btc"]))
        down_all.extend(_sorted_pairs(w["down"]))
    btc_all.sort()
    down_all.sort()
    t_max = btc_all[-1][0]

    def stats(pnls: list[float], wins_list: list[bool]) -> str:
        n = len(pnls)
        if not n:
            return "0 注"
        wr = sum(wins_list) / n
        ev = float(np.mean(pnls))
        if n >= MIN_BETS_FOR_CI:
            rng = np.random.default_rng(7)
            ix = rng.integers(0, n, size=(2000, n))
            ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
            return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f}"

    groups: dict[str, list[dict]] = {"全部高点": [], "冲过日线阻力": [], "未冲过日线阻力": []}
    for idx, w in enumerate(wins):
        if idx <= LOOKBACK:
            continue
        btc = _sorted_pairs(w["btc"])
        down = _sorted_pairs(w["down"])
        if len(btc) < 2:
            continue
        peak_t, peak_v = max(btc, key=lambda p: p[1])
        if peak_t + HOLD_MS > t_max:
            continue
        entry = _price_at(down, peak_t)
        exit_dn = _price_at(down_all, peak_t + HOLD_MS)
        btc_exit = _price_at(btc_all, peak_t + HOLD_MS)
        if entry is None or exit_dn is None or btc_exit is None or entry <= 0:
            continue
        # 阻力（前 288 窗口 closes max）
        hist = wins[idx - LOOKBACK: idx]
        hist_closes: list[float] = []
        ok = True
        for h in hist:
            pairs = _sorted_pairs(h["btc"])
            if not pairs:
                ok = False
                break
            hist_closes.append(pairs[-1][1])
        if not ok:
            continue
        resistance = max(hist_closes)
        broke = peak_v > resistance * (1.0 + EPS)

        evt = {
            "entry": entry,
            "exit_dn": exit_dn,
            "btc_peak": peak_v,
            "btc_exit": btc_exit,
        }
        groups["全部高点"].append(evt)
        if broke:
            groups["冲过日线阻力"].append(evt)
        else:
            groups["未冲过日线阻力"].append(evt)

    print(f"\n对照组样本: 全部 {len(groups['全部高点'])} / "
          f"冲过 {len(groups['冲过日线阻力'])} / "
          f"未冲过 {len(groups['未冲过日线阻力'])}")

    for label, evts in groups.items():
        if not evts:
            print(f"\n===== {label}: 0 注 =====")
            continue
        entries = [e["entry"] for e in evts]
        tok_w = [e["exit_dn"] > e["entry"] for e in evts]
        tok_p = [_bet_pnl(w_, p) for w_, p in zip(tok_w, entries)]
        btc_w = [e["btc_exit"] < e["btc_peak"] for e in evts]
        btc_p05 = [_bet_pnl(w_, 0.5) for w_ in btc_w]
        print(f"\n===== {label}（平均入场 DOWN {float(np.mean(entries)):.3f}） =====")
        print(f"  BTC 方向（高点→+15m）: {stats(btc_p05, btc_w)}")
        print(f"  token 回升（真实入场）: {stats(tok_p, tok_w)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
