#!/usr/bin/env python3
"""级别矩阵 × 双向假突破瞬间下注（最优赔率玩法）。

基于 local_combo_control_check.py 的对照设计扩展：
- 支撑/阻力级别：1h（12 个 5m 窗口 closes）/ 4h（48）/ 日线（288）
- 双向：
  a. 冲高破阻力瞬间 → 买 DOWN（DOWN token 被砸到白菜价）
  b. 冲低破支撑瞬间 → 买 UP（UP token 被砸到白菜价）
- 兑现：入场后 +15 分钟（900s），跨窗口取全局时间序列
- 结算两口径：token 回升（兑现时刻 token 价 vs 入场价）+ BTC 方向（兑现时刻 BTC vs 入场时刻 BTC）
- 对照基线：全部盘中高点买 DOWN / 全部盘中低点买 UP（均值回归基线）

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
HOLD_MS = 900_000  # 15 分钟兑现
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
    down_all: list[tuple[int, float]] = []
    up_all: list[tuple[int, float]] = []
    for w in wins:
        btc_all.extend(_sorted_pairs(w["btc"]))
        down_all.extend(_sorted_pairs(w["down"]))
        up_all.extend(_sorted_pairs(w["up"]))
    btc_all.sort()
    down_all.sort()
    up_all.sort()
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
                token_all = down_all
            else:
                ext_t, ext_v = min(btc, key=lambda p: p[1])
                level = min(hist_closes)
                broke = ext_v < level * (1.0 - EPS)
                token_curve = _sorted_pairs(w["up"])
                token_all = up_all

            if not broke or ext_t + HOLD_MS > t_max:
                continue
            entry = _price_at(token_curve, ext_t)
            exit_tok = _price_at(token_all, ext_t + HOLD_MS)
            btc_exit = _price_at(btc_all, ext_t + HOLD_MS)
            if entry is None or exit_tok is None or btc_exit is None or entry <= 0:
                continue
            events.append({
                "entry": entry,
                "exit_tok": exit_tok,
                "btc_ext": ext_v,
                "btc_exit": btc_exit,
                "side": side,
            })
        return events

    def report(label: str, evts: list[dict]) -> None:
        if not evts:
            print(f"  {label}: 0 注")
            return
        entries = [e["entry"] for e in evts]
        avg_entry = float(np.mean(entries))
        # token 回升：高点买 DOWN 后 token 回升 / 低点买 UP 后 token 回升
        tok_w = [e["exit_tok"] > e["entry"] for e in evts]
        tok_p = [_bet_pnl(w_, p) for w_, p in zip(tok_w, entries)]
        # BTC 方向：高点后回落 / 低点后反弹
        if evts[0]["side"] == "high":
            btc_w = [e["btc_exit"] < e["btc_ext"] for e in evts]
        else:
            btc_w = [e["btc_exit"] > e["btc_ext"] for e in evts]
        btc_p = [_bet_pnl(w_, p) for w_, p in zip(btc_w, entries)]
        print(f"  {label}（平均入场 {avg_entry:.3f}）")
        print(f"    token 回升: {stats(tok_p, tok_w)}")
        print(f"    BTC 方向  : {stats(btc_p, btc_w)}")

    for label, lookback in LEVELS:
        print(f"\n===== 级别 {label}（LOOKBACK={lookback} 个 5m 窗口） =====")
        report("冲高破阻力→买DOWN", collect(lookback, "high"))
        report("冲低破支撑→买UP  ", collect(lookback, "low"))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
