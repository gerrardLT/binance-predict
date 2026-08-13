#!/usr/bin/env python3
"""本地数据验证：5 分钟周期 · 假突破瞬间下注的动态赔率玩法。

与 local_15m_daily_token_check.py 的区别：
- 不聚合：每个 5m 窗口直接作为一个预测周期
- 日线阻力 = 前 288 个 5m 窗口 closes 的 max
- 结算原则（用户规定）：只关心周期内涨跌方向，与涨跌幅度无关

验证内容（本地一个月数据，15s 采样 token 价）：
1. up→DOWN 假突破：盘中冲过日线阻力 0.05% 且收盘收回（基线）
2. 每个事件：定位 BTC 盘中高点时刻 → 查该时刻 DOWN token 真实价（入场赔率）
3. 结算口径对比：
   a. token 结算：窗口末 DOWN token 价 vs 入场价
   b. BTC 符号结算：窗口末 BTC vs 窗口开 BTC（方向，与幅度无关）
4. 输出：平均入场赔率、token 胜率、费后期望 vs 0.5 入场对照

用法：
    python scripts/local_5m_daily_token_check.py
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
LOOKBACK = 288  # 日线阻力 = 前 288 个 5m 窗口
EPS = 0.0005
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
                SentimentWindow.curve_up_price,
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {
            "btc": r.curve_btc_price,
            "down": r.curve_down_price,
            "up": r.curve_up_price,
        }
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    events: list[dict] = []
    for idx, w in enumerate(wins):
        if idx <= LOOKBACK or idx >= len(wins) - 1:
            continue
        hist = wins[idx - LOOKBACK: idx]
        hist_closes: list[float] = []
        ok_hist = True
        for h in hist:
            pairs = _sorted_pairs(h["btc"])
            if not pairs:
                ok_hist = False
                break
            hist_closes.append(pairs[-1][1])
        if not ok_hist:
            continue
        resistance = max(hist_closes)

        btc = _sorted_pairs(w["btc"])
        down = _sorted_pairs(w["down"])
        if len(btc) < 2:
            continue
        o, c = btc[0][1], btc[-1][1]
        peak_t, peak_v = max(btc, key=lambda p: p[1])
        broke = peak_v > resistance * (1.0 + EPS)
        reclaimed = c <= resistance
        if not (broke and reclaimed):
            continue
        # 入场 = 假突破瞬间（盘中高点时刻）的 DOWN token 价
        entry = _price_at(down, peak_t)
        exitp = _price_at(down, btc[-1][0])
        if entry is None or exitp is None or entry <= 0:
            continue
        events.append({
            "peak_t": peak_t,
            "entry": entry,
            "exit": exitp,
            "btc_o": o,
            "btc_c": c,
        })
        print(
            f"  事件 {peak_t}: 盘中高点 {peak_v:.0f} 阻力 {resistance:.0f} "
            f"| DOWN入场 {entry:.3f} 窗口末 {exitp:.3f} "
            f"| BTC {o:.0f}→{c:.0f} ({'+' if c>o else ''}{(c/o-1)*100:.2f}%)"
        )

    if not events:
        print("无假突破事件")
        return 0

    n = len(events)
    entries = [e["entry"] for e in events]
    avg_entry = float(np.mean(entries))
    print(f"\n事件总数 {n}，平均入场赔率（DOWN token 价）{avg_entry:.3f}")

    token_wins = [e["exit"] > e["entry"] for e in events]
    token_wr = sum(token_wins) / n
    token_pnls = [_bet_pnl(w, p) for w, p in zip(token_wins, entries)]
    token_ev = float(np.mean(token_pnls))
    print(f"\n===== 口径A：token 结算（假突破瞬间买 DOWN，持有到窗口末） =====")
    print(f"  胜率 {token_wr:.1%} 费后EV {token_ev:+.3f}")

    btc_wins = [e["btc_c"] < e["btc_o"] for e in events]
    btc_wr = sum(btc_wins) / n
    btc_pnls = [_bet_pnl(w, p) for w, p in zip(btc_wins, entries)]
    btc_ev = float(np.mean(btc_pnls))
    print(f"\n===== 口径B：BTC 符号结算（同批事件，入场价用假突破瞬间 token 价） =====")
    print(f"  胜率 {btc_wr:.1%} 费后EV {btc_ev:+.3f}")

    fake_pnls = [_bet_pnl(w, 0.5) for w in btc_wins]
    print(f"\n===== 口径C：假设 0.5 入场（回测原口径对照） =====")
    print(f"  胜率 {btc_wr:.1%} 费后EV {float(np.mean(fake_pnls)):+.3f}")

    if n >= MIN_BETS_FOR_CI:
        rng = np.random.default_rng(7)
        ix = rng.integers(0, n, size=(2000, n))
        ci = np.percentile(np.asarray(token_pnls)[ix].mean(axis=1), [2.5, 97.5])
        print(f"  口径A 费后EV CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    print(f"\n入场价分布: 最低 {min(entries):.3f} 最高 {max(entries):.3f} "
          f"中位 {float(np.median(entries)):.3f}")
    print(f"  <0.4: {sum(1 for p in entries if p<0.4)}/{n}  "
          f"0.4-0.5: {sum(1 for p in entries if 0.4<=p<0.5)}/{n}  "
          f">=0.5: {sum(1 for p in entries if p>=0.5)}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
