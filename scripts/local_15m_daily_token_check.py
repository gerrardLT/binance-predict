#!/usr/bin/env python3
"""本地数据验证：假突破瞬间下注的动态赔率玩法（用户真实玩法）。

与回测口径的核心区别：
- 回测：假突破窗口收盘确认 → 下一根开盘入场，赔率假设 0.5
- 本玩法：盘中冲过日线阻力那一瞬间入场（情绪极端点，DOWN token 最便宜）

验证内容（本地一个月数据，15s 采样 token 价）：
1. 15m 聚合（3 个 5m 窗口不重叠）+ 日线阻力（前 96 组 closes max）
2. up→DOWN 假突破：盘中冲过阻力 0.05% 且收盘收回（基线，无连续窗）
3. 每个事件：定位 BTC 盘中高点时刻 → 查该时刻 DOWN token 真实价（入场赔率）
4. 两种结算口径对比：
   a. 持有到窗口末：窗口末 DOWN token 价 vs 入场价
   b. BTC 符号结算（回测口径）：窗口末 BTC vs 窗口开 BTC
5. 输出：平均入场赔率、token 胜率、token 费后期望 vs 回测口径期望

用法：
    python scripts/local_15m_daily_token_check.py
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
WINDOW_MS = 300_000
LOOKBACK = 96  # 日线阻力 = 前 96 个 15m 窗口
EPS = 0.0005
MIN_BETS_FOR_CI = 8


def _sorted_pairs(curve: list | None) -> list[tuple[int, float]]:
    """曲线按时间排序的 (t, v) 对。"""
    pts = [
        (int(p.get("t", 0)), float(p["v"]))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return pts


def _price_at(pairs: list[tuple[int, float]], t_ms: int) -> float | None:
    """曲线在 t_ms 时刻的最近采样价（前后各取一个线性插值）。"""
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
            "start": r.start_time,
            "btc": r.curve_btc_price,
            "down": r.curve_down_price,
            "up": r.curve_up_price,
        }
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    # 3 个连续 5m 窗口 → 1 个 15m 窗口（不重叠）
    agg: list[dict] = []
    for i in range(0, len(wins) - 2, 3):
        a, b, c = wins[i], wins[i + 1], wins[i + 2]
        if b["start"] - a["start"] != WINDOW_MS or c["start"] - b["start"] != WINDOW_MS:
            continue
        btc = _sorted_pairs(a["btc"]) + _sorted_pairs(b["btc"]) + _sorted_pairs(c["btc"])
        down = _sorted_pairs(a["down"]) + _sorted_pairs(b["down"]) + _sorted_pairs(c["down"])
        if len(btc) < 2:
            continue
        agg.append({
            "btc": btc,
            "down": down,
            "o": btc[0][1],
            "c": btc[-1][1],
        })
    print(f"聚合 15m 窗口 {len(agg)} 个")

    events: list[dict] = []
    for idx, w in enumerate(agg):
        if idx <= LOOKBACK or idx >= len(agg) - 1:
            continue
        hist = agg[idx - LOOKBACK: idx]
        resistance = max(max(x["btc"], key=lambda p: p[1])[1] for x in hist)
        peak_t, peak_v = max(w["btc"], key=lambda p: p[1])
        broke = peak_v > resistance * (1.0 + EPS)
        reclaimed = w["c"] <= resistance
        if not (broke and reclaimed):
            continue
        # 入场 = 假突破瞬间（盘中高点时刻）的 DOWN token 价
        entry = _price_at(w["down"], peak_t)
        # 退出 = 窗口末 DOWN token 价
        exitp = _price_at(w["down"], w["btc"][-1][0])
        if entry is None or exitp is None or entry <= 0:
            continue
        events.append({
            "peak_t": peak_t,
            "entry": entry,
            "exit": exitp,
            "btc_o": w["o"],
            "btc_c": w["c"],
        })
        print(
            f"  事件 {peak_t}: 盘中高点 {peak_v:.0f} 阻力 {resistance:.0f} "
            f"| DOWN入场 {entry:.3f} 窗口末 {exitp:.3f} "
            f"| BTC {w['o']:.0f}→{w['c']:.0f} ({'+' if w['c']>w['o'] else ''}{(w['c']/w['o']-1)*100:.2f}%)"
        )

    if not events:
        print("无假突破事件")
        return 0

    n = len(events)
    entries = [e["entry"] for e in events]
    avg_entry = float(np.mean(entries))
    print(f"\n事件总数 {n}，平均入场赔率（DOWN token 价）{avg_entry:.3f}")

    # 口径 A：token 结算（持有到窗口末）
    token_wins = [e["exit"] > e["entry"] for e in events]
    token_wr = sum(token_wins) / n
    token_pnls = [_bet_pnl(w, p) for w, p in zip(token_wins, entries)]
    token_ev = float(np.mean(token_pnls))
    print(f"\n===== 口径A：token 结算（假突破瞬间买 DOWN，持有到窗口末） =====")
    print(f"  胜率 {token_wr:.1%} 费后EV {token_ev:+.3f}")

    # 口径 B：BTC 符号结算（回测口径对照）
    btc_wins = [e["btc_c"] < e["btc_o"] for e in events]
    btc_wr = sum(btc_wins) / n
    btc_pnls = [_bet_pnl(w, p) for w, p in zip(btc_wins, entries)]
    btc_ev = float(np.mean(btc_pnls))
    print(f"\n===== 口径B：BTC 符号结算（同批事件，入场价用假突破瞬间 token 价） =====")
    print(f"  胜率 {btc_wr:.1%} 费后EV {btc_ev:+.3f}")

    # 口径 C：假设 0.5 入场（回测原口径）
    fake_pnls = [_bet_pnl(w, 0.5) for w in btc_wins]
    print(f"\n===== 口径C：假设 0.5 入场（回测原口径对照） =====")
    print(f"  胜率 {btc_wr:.1%} 费后EV {float(np.mean(fake_pnls)):+.3f}")

    if n >= MIN_BETS_FOR_CI:
        rng = np.random.default_rng(7)
        ix = rng.integers(0, n, size=(2000, n))
        ci = np.percentile(np.asarray(token_pnls)[ix].mean(axis=1), [2.5, 97.5])
        print(f"  口径A 费后EV CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # 入场价分布
    print(f"\n入场价分布: 最低 {min(entries):.3f} 最高 {max(entries):.3f} "
          f"中位 {float(np.median(entries)):.3f}")
    print(f"  <0.4: {sum(1 for p in entries if p<0.4)}/{n}  "
          f"0.4-0.5: {sum(1 for p in entries if 0.4<=p<0.5)}/{n}  "
          f">=0.5: {sum(1 for p in entries if p>=0.5)}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
