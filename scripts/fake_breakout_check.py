#!/usr/bin/env python3
"""假突破反向下注验证（用户场景 2）第二阶段：叠加"连续跌/涨 + 接近支撑/阻力"条件。

基线（已跑）：任何位置破位+收回 → 反向下注。2 小时支撑 28 注胜率 71.4%、费后 EV +31.7%。
本阶段：在基线条件上叠加用户原话的两个要素，控制变量看胜率是否继续增强——
1. 破位前连续 K 窗下跌（down 方向）/ 连续 K 窗上涨（up 方向）
2. 破位窗口开盘价已接近支撑/阻力（距离 <= X）

模拟规则与场景 1 脚本同口径（结果可直接对比）：
- 支撑 = 过去 N 个窗口（不含当前）BTC 曲线最低价；阻力 = 过去 N 个窗口最高价
- 假突破向下：完整窗口盘中低点 < 支撑*(1-eps) 且窗口末价收回支撑之上 → 下注 UP
- 假突破向上：盘中高点 > 阻力*(1+eps) 且窗口末价回落到阻力之下 → 下注 DOWN
- 下注窗口 i+1，入场价 = 窗口 i+1 曲线最早采样价（真实 token 价，缺则 up_pct proxy）
- 结算 = sign(actual_return)，=0 剔除；逐注盈亏：赢 → (1-0.02)/(price+0.01)-1；输 → -1

用法：
    python scripts/fake_breakout_check.py
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
LOOKBACKS = (12, 24)  # 支撑/阻力滚动窗口数 = 1h / 2h
EPS = 0.0005  # 破位幅度阈值（样本量最多的一档）
MIN_BETS_FOR_CI = 10

# 条件变体：(标签, 条件种类, 连续窗数 K, 接近距离 X)
#   kind=None      → 基线（只破位+收回）
#   kind="streak"  → 破位前连续 K 窗同向
#   kind="near"    → 破位窗口开盘价距支撑/阻力 <= X
#   kind="streak_near" → 两者都要
VARIANTS = (
    ("无附加(基线)", None, 0, 0.0),
    ("连续2窗", "streak", 2, 0.0),
    ("连续3窗", "streak", 3, 0.0),
    ("接近0.2%", "near", 0, 0.002),
    ("连续2窗+接近0.2%", "streak_near", 2, 0.002),
    ("连续3窗+接近0.2%", "streak_near", 3, 0.002),
)


def _window_extremes(curve: list | None) -> tuple[float, float] | None:
    """窗口内 BTC 曲线的 (低, 高)。"""
    vals = [float(p["v"]) for p in (curve or []) if p.get("v") is not None]
    if len(vals) < 2:
        return None
    return min(vals), max(vals)


def _window_vals(curve: list | None) -> list[float]:
    """窗口完整曲线有效值（按时间序）。"""
    pts = [
        (p.get("t", 0), p.get("v"))
        for p in (curve or [])
        if p.get("v") is not None
    ]
    pts.sort()
    return [float(v) for _, v in pts]


def _open_price_early(curve: list | None, start_ms: int) -> float | None:
    """窗口最早采样价（开盘附近可观测价）。"""
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
    """破位窗口 a 之前连续 k 个窗口是否同向（down=连续跌，up=连续涨）。"""
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


def _near_ok(a_open: float, level: float, x: float, side: str) -> bool:
    """开盘价是否已贴近支撑/阻力（距离 <= x）。"""
    if side == "down":
        return a_open <= level * (1.0 + x)
    return a_open >= level * (1.0 - x)


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
    print(f"窗口总数 {len(wins)}，按时间排序")

    pairs: list[tuple[dict, dict]] = []
    for a, b in zip(wins, wins[1:]):
        if b["start"] - a["start"] == WINDOW_MS:
            pairs.append((a, b))
    print(f"精确相邻窗口对: {len(pairs)}")

    results: list[dict] = []
    for lookback in LOOKBACKS:
        print(f"\n===== 支撑/阻力 = 过去 {lookback} 窗口极值 "
              f"（{lookback * 5 // 60} 小时）=====")
        for label, kind, k, x in VARIANTS:
            for side, direction in (("down", "UP"), ("up", "DOWN")):
                bets: list[tuple[bool, float, str]] = []
                for idx, (a, b) in enumerate(pairs):
                    if idx < lookback:
                        continue
                    hist = [pairs[j][0] for j in range(idx - lookback, idx)]
                    hist_ext: list[tuple[float, float]] = []
                    for h in hist:
                        ex = _window_extremes(h["btc"])
                        if ex is not None:
                            hist_ext.append(ex)
                    if len(hist_ext) < lookback // 2:
                        continue
                    support = min(ex[0] for ex in hist_ext)
                    resistance = max(ex[1] for ex in hist_ext)
                    if support <= 0 or resistance <= 0 or resistance <= support:
                        continue

                    vals = _window_vals(a["btc"])
                    if len(vals) < 2:
                        continue
                    a_open, low, high, last = vals[0], min(vals), max(vals), vals[-1]

                    if side == "down":
                        broke = low < support * (1.0 - EPS)
                        reclaimed = last >= support
                        level = support
                    else:
                        broke = high > resistance * (1.0 + EPS)
                        reclaimed = last <= resistance
                        level = resistance
                    if not (broke and reclaimed):
                        continue

                    # 叠加条件
                    if kind in ("streak", "streak_near"):
                        if not _streak_ok(pairs, idx, k, side):
                            continue
                    if kind in ("near", "streak_near"):
                        if not _near_ok(a_open, level, x, side):
                            continue

                    # 结算 = 下一周期 return 符号
                    if b["ret"] is None or float(b["ret"]) == 0.0:
                        continue
                    b_ret = float(b["ret"])
                    win = (b_ret > 0) if direction == "UP" else (b_ret < 0)

                    price = _open_price_early(
                        b["down_price"] if direction == "DOWN" else b["up_price"], b["start"]
                    )
                    kind_p = "real"
                    if price is None:
                        p = _open_price_early(b["up_pct"], b["start"])
                        if p is None:
                            continue
                        price = p / 100.0 if direction == "UP" else max(1.0 - p / 100.0, 0.01)
                        kind_p = "proxy"
                    bets.append((win, price, kind_p))

                if not bets:
                    print(f"  {label:>16} {side:>4}→{direction}: 0 注（条件过严无样本）")
                    continue
                n = len(bets)
                n_real = sum(1 for _, _, kk in bets if kk == "real")
                win_rate = sum(1 for w, _, _ in bets if w) / n
                avg_price = sum(p for _, p, _ in bets) / n
                pnls = [_bet_pnl(w, p) for w, p, _ in bets]
                ev = float(np.mean(pnls))
                if n >= MIN_BETS_FOR_CI:
                    rng = np.random.default_rng(7)
                    ix = rng.integers(0, n, size=(2000, n))
                    ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
                    lo, hi = float(ci[0]), float(ci[1])
                else:
                    lo, hi = float("nan"), float("nan")
                results.append({
                    "lookback": lookback, "variant": label, "side": side,
                    "direction": direction, "n": n, "n_real": n_real,
                    "win_rate": round(win_rate, 4),
                    "avg_entry": round(avg_price, 4), "ev": round(ev, 4),
                    "ev_ci_lower": round(lo, 4), "ev_ci_upper": round(hi, 4),
                })
                print(
                    f"  {label:>16} {side:>4}→{direction}: 注数 {n}(真实价 {n_real}) "
                    f"胜率 {win_rate:.1%} 平均入场 {avg_price:.3f} "
                    f"费后EV {ev:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                )

    out = {
        "n_windows": len(wins),
        "n_pairs": len(pairs),
        "eps": EPS,
        "results": results,
    }
    with open("output/fake_breakout_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 output/fake_breakout_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
