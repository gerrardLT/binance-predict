#!/usr/bin/env python3
"""级别矩阵 × 双向假突破瞬间下注（最优赔率玩法，周期锚点口径）。

基于 local_combo_control_check.py 的对照设计扩展：
- 支撑/阻力级别：1h（12 个 5m 窗口 closes）/ 4h（48）/ 日线（288）
- 双向：
  a. 冲高破阻力瞬间 → 买 DOWN（DOWN token 被砸到白菜价）
  b. 冲低破支撑瞬间 → 买 UP（UP token 被砸到白菜价）
- 结算【周期锚点，与币安市场真实结算规则一致】：
  信号所在 15m/5m 市场周期，UP 赢 ⟺ 周期末价 > 周期开盘价
  （周期边界按自然时钟对齐：ext_t - ext_t % 900_000 / 300_000）

用法：
    python scripts/local_combo_level_matrix_check.py
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
MIN_BETS_FOR_CI = 8
LEVELS = (("1h", 12), ("4h", 48), ("日线", 288))


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
        {"btc": r.curve_btc_price, "down": r.curve_down_price, "up": r.curve_up_price}
        for r in rows
    ]
    print(f"5m 窗口总数 {len(wins)}")

    btc_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(_sorted_pairs(w["btc"]))
    btc_all.sort()
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

    # 预计算每窗口的 closes 末点（供支撑/阻力计算）
    closes = []
    for w in wins:
        pairs = _sorted_pairs(w["btc"])
        closes.append(pairs[-1][1] if pairs else None)

    def collect(lookback: int, side: str) -> list[dict]:
        """side='high': 盘中高点冲过阻力买 DOWN；side='low': 盘中低点破支撑买 UP。"""
        events: list[dict] = []
        for idx, w in enumerate(wins):
            if idx <= lookback:
                continue
            btc = _sorted_pairs(w["btc"])
            if len(btc) < 2:
                continue
            hist_closes = [c for c in closes[idx - lookback: idx] if c is not None]
            if len(hist_closes) < lookback // 2:
                continue

            if side == "high":
                ext_t, ext_v = max(btc, key=lambda p: p[1])
                level = max(hist_closes)
                broke = ext_v > level * (1.0 + EPS)
                token_curve = _sorted_pairs(w["down"])
            else:
                ext_t, ext_v = min(btc, key=lambda p: p[1])
                level = min(hist_closes)
                broke = ext_v < level * (1.0 - EPS)
                token_curve = _sorted_pairs(w["up"])

            if not broke:
                continue
            # 周期锚点：信号所在 15m / 5m 市场周期（自然时钟边界对齐）
            cyc15_start = ext_t - (ext_t % 900_000)
            cyc15_end = cyc15_start + 900_000
            cyc5_start = ext_t - (ext_t % 300_000)
            cyc5_end = cyc5_start + 300_000
            if cyc15_end > t_max:
                continue
            entry = _price_at(token_curve, ext_t)
            p_s15 = _price_at(btc_all, cyc15_start)
            p_e15 = _price_at(btc_all, cyc15_end)
            p_s5 = _price_at(btc_all, cyc5_start)
            p_e5 = _price_at(btc_all, cyc5_end)
            if (
                entry is None or entry <= 0
                or p_s15 is None or p_e15 is None
                or p_s5 is None or p_e5 is None
            ):
                continue
            events.append({
                "entry": entry,
                "p_s15": p_s15, "p_e15": p_e15,
                "p_s5": p_s5, "p_e5": p_e5,
                "side": side,
            })
        return events

    def report(label: str, evts: list[dict]) -> None:
        if not evts:
            print(f"  {label}: 0 注")
            return
        entries = [e["entry"] for e in evts]
        avg_entry = float(np.mean(entries))
        high = evts[0]["side"] == "high"
        # 周期锚点两口径：买 DOWN 赢 ⟺ 周期末价 < 周期开盘价；买 UP 反之
        w15 = [
            (e["p_e15"] < e["p_s15"]) if high else (e["p_e15"] > e["p_s15"])
            for e in evts
        ]
        p15 = [_bet_pnl(w_, p) for w_, p in zip(w15, entries)]
        w5 = [
            (e["p_e5"] < e["p_s5"]) if high else (e["p_e5"] > e["p_s5"])
            for e in evts
        ]
        p5 = [_bet_pnl(w_, p) for w_, p in zip(w5, entries)]
        print(f"  {label}（平均入场 {avg_entry:.3f}）")
        print(f"    15m 周期方向: {stats(p15, w15)}")
        print(f"    5m 周期方向 : {stats(p5, w5)}")

    for label, lookback in LEVELS:
        print(f"\n===== 级别 {label}（LOOKBACK={lookback} 个 5m 窗口） =====")
        report("冲高破阻力→买DOWN", collect(lookback, "high"))
        report("冲低破支撑→买UP  ", collect(lookback, "low"))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
