#!/usr/bin/env python3
"""组合测试：5m 发现假突破瞬间入场 + 所在 15m 周期到期结算（周期锚点修正版）。

审查修正：
1. 事件集 = 所有盘中冲过日线阻力的 5m 窗口（不要求收盘收回，消除选择偏差）
   - 再分子组对比：收回组 vs 未收回组
2. 结算【周期锚点，与币安市场真实结算规则一致】：
   信号所在 15m 市场周期，DOWN 赢 ⟺ 周期末价 < 周期开盘价
   （周期边界按自然时钟对齐：peak_t - peak_t % 900_000）
3. 执行敏感性：peak_t 插值价 vs peak_t±60s 区间内 DOWN 最低价（最优执行上限）
4. 结算两口径（用户原则：只关心周期内方向，与幅度无关）：
   a. token 结算：周期末时刻 DOWN 价 vs 入场价（到期 token 收敛 0/1，交叉验证）
   b. BTC 方向结算：周期末 BTC 价 vs 周期开盘价（= 市场真实结算）

用法：
    python scripts/local_combo_5m_15m_check.py
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
LOOKBACK = 288  # 日线阻力 = 前 288 个 5m 窗口 closes 的 max
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
            ).order_by(SentimentWindow.start_time)
        )).all()

    wins: list[dict] = [
        {"btc": r.curve_btc_price, "down": r.curve_down_price}
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    # 全局连续时间序列（跨窗口取价用）
    btc_all: list[tuple[int, float]] = []
    down_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(_sorted_pairs(w["btc"]))
        down_all.extend(_sorted_pairs(w["down"]))
    btc_all.sort()
    down_all.sort()
    print(f"全局 BTC 采样点 {len(btc_all)}，DOWN 采样点 {len(down_all)}")
    t_min = btc_all[0][0]
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
            return f"注数 {n:>3} 胜率 {wr:.1%} 费后EV {ev:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        return f"注数 {n:>3} 胜率 {wr:.1%} 费后EV {ev:+.3f}"

    events: list[dict] = []
    for idx, w in enumerate(wins):
        if idx <= LOOKBACK:
            continue
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

        btc = _sorted_pairs(w["btc"])
        down = _sorted_pairs(w["down"])
        if len(btc) < 2:
            continue
        o, c = btc[0][1], btc[-1][1]
        peak_t, peak_v = max(btc, key=lambda p: p[1])
        broke = peak_v > resistance * (1.0 + EPS)
        if not broke:
            continue
        # 全部冲高破位事件（不要求收回）
        # 周期锚点：信号所在 15m 市场周期（自然时钟边界对齐）
        cyc15_start = peak_t - (peak_t % 900_000)
        cyc15_end = cyc15_start + 900_000
        if cyc15_end > t_max:
            continue
        entry = _price_at(down, peak_t)
        # 执行敏感性：peak_t±60s 区间内 DOWN 最低价（最优执行上限）
        window_min = min(
            (v for t, v in down if peak_t - 60_000 <= t <= peak_t + 60_000),
            default=None,
        )
        exit_dn = _price_at(down_all, cyc15_end)
        p_s15 = _price_at(btc_all, cyc15_start)
        p_e15 = _price_at(btc_all, cyc15_end)
        if entry is None or exit_dn is None or p_s15 is None or p_e15 is None or entry <= 0:
            continue
        events.append({
            "peak_t": peak_t,
            "peak_v": peak_v,
            "res": resistance,
            "o": o,
            "c": c,
            "reclaimed_5m": c <= resistance,
            "reclaimed_15m": p_e15 <= resistance,
            "entry": entry,
            "entry_min": window_min if window_min is not None else entry,
            "exit_dn": exit_dn,
            "p_s15": p_s15,
            "p_e15": p_e15,
        })

    print(f"\n全部冲高破位事件: {len(events)} 个")

    def report(label: str, evts: list[dict]) -> None:
        if not evts:
            print(f"\n===== {label}: 0 注 =====")
            return
        n = len(evts)
        entries = [e["entry"] for e in evts]
        # token 结算（真实入场价）：周期末 DOWN 价 vs 入场价
        tok_w = [e["exit_dn"] > e["entry"] for e in evts]
        tok_p = [_bet_pnl(w_, p) for w_, p in zip(tok_w, entries)]
        # BTC 方向结算（周期锚点）：周期末价 < 周期开盘价 → DOWN 赢
        btc_w = [e["p_e15"] < e["p_s15"] for e in evts]
        btc_p = [_bet_pnl(w_, p) for w_, p in zip(btc_w, entries)]
        # BTC 方向结算（0.5 假设入场对照）
        fake_p = [_bet_pnl(w_, 0.5) for w_ in btc_w]
        # 最优执行上限（token 结算）
        mins = [e["entry_min"] for e in evts]
        tok_min_w = [e["exit_dn"] > e["entry_min"] for e in evts]
        tok_min_p = [_bet_pnl(w_, p) for w_, p in zip(tok_min_w, mins)]
        print(f"\n===== {label}（{n} 注，平均入场 {float(np.mean(entries)):.3f}） =====")
        print(f"  token 结算（真实入场）: {stats(tok_p, tok_w)}")
        print(f"  BTC 方向（真实入场）  : {stats(btc_p, btc_w)}")
        print(f"  BTC 方向（0.5 入场）  : {stats(fake_p, btc_w)}")
        print(f"  token 结算（最优执行）: {stats(tok_min_p, tok_min_w)}")

    report("全部冲高破位", events)
    report("5m 末收回", [e for e in events if e["reclaimed_5m"]])
    report("5m 末未收回", [e for e in events if not e["reclaimed_5m"]])
    report("15m 末收回", [e for e in events if e["reclaimed_15m"]])
    report("15m 末未收回", [e for e in events if not e["reclaimed_15m"]])

    # 事件明细（供人工核对）
    print("\n===== 事件明细 =====")
    for e in events:
        print(
            f"  {e['peak_t']}: 高点 {e['peak_v']:.0f} 阻力 {e['res']:.0f} "
            f"| 入场 {e['entry']:.3f}(±60s最低 {e['entry_min']:.3f}) "
            f"→周期末 {e['exit_dn']:.3f} | 15m周期 开→末: "
            f"{'+' if e['p_e15'] > e['p_s15'] else ''}{(e['p_e15'] / e['p_s15'] - 1) * 100:.3f}% "
            f"| 5m收 {'是' if e['reclaimed_5m'] else '否'} 15m收 {'是' if e['reclaimed_15m'] else '否'}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
