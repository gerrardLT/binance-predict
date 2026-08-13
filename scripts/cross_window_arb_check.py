#!/usr/bin/env python3
"""跨窗口信息套利验证（用户场景 1）：本周期形态 → 下周期方向，50/50 定价假设检验。

要验证的三个问题：
1. 本周期急跌/急涨（前 150s 已形成的实体长阴/长阳）→ 下一周期同向的实际胜率
2. 下一周期开盘时 UP/DOWN token 的真实定价分布（是否如用户所说接近 50/50）
3. 两者合并的费后期望（口径与 ev_gate 一致：费 2% + 溢价 0.01）

模拟规则：
- 在窗口 i 的第 150 秒（决策点）做决定，下注窗口 i+1 的方向
- 入场价 = 窗口 i+1 自己的曲线里最早的采样价（最接近"刚开盘未定价"的可观测价）
- 结算 = sign(actual_return)，=0 剔除（平盘规则未知，与 ev_gate 一致）
- 逐注盈亏：赢 → (1-0.02)/(price+0.01)-1；输 → -1

用法：
    python scripts/cross_window_arb_check.py
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
DECISION_SEC = 150.0
WINDOW_MS = 300_000  # 5 分钟
# 信号阈值（本周期前 150s BTC 涨跌幅），负=长阴看跌下周期，正=长阳看涨
THRESHOLDS = (0.0005, 0.001, 0.0015, 0.002, 0.003)


def _first_last(curve: list | None, start_ms: int, t_sec: float) -> tuple[float, float] | None:
    """曲线截断到 t_sec 内的 (首值, 末值)。"""
    pts = [
        (p.get("t", 0), p.get("v"))
        for p in (curve or [])
        if (p.get("t", 0) - start_ms) / 1000.0 <= t_sec and p.get("v") is not None
    ]
    if len(pts) < 2:
        return None
    pts.sort()
    return float(pts[0][1]), float(pts[-1][1])


def _open_price_early(curve: list | None, start_ms: int) -> float | None:
    """窗口最早采样价：取 rel_t 最小的点（开盘附近价）。"""
    best, best_v = None, None
    for p in curve or []:
        if p.get("v") is None:
            continue
        rel = (p.get("t", 0) - start_ms) / 1000.0
        if rel < 0:  # 开窗前采样点跳过（归入本窗口曲线的点应 >= 0）
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

    # 构造精确相邻对（5 分钟无缺口）
    pairs: list[tuple[dict, dict]] = []
    for a, b in zip(wins, wins[1:]):
        if b["start"] - a["start"] == WINDOW_MS:
            pairs.append((a, b))
    print(f"精确相邻窗口对: {len(pairs)}")

    # ---- 验证 2：下一周期开盘定价分布 ----
    open_prices: list[tuple[float, float]] = []  # (up, down) 开盘价
    for a, b in pairs:
        up = _open_price_early(b["up_price"], b["start"])
        down = _open_price_early(b["down_price"], b["start"])
        if up and down:
            open_prices.append((up, down))
    print(f"\n== 下一周期开盘定价分布（真实 token 价，N={len(open_prices)}）==")
    if open_prices:
        ups = np.array([u for u, d in open_prices])
        downs = np.array([d for u, d in open_prices])
        for name, arr in (("UP", ups), ("DOWN", downs)):
            qs = np.percentile(arr, [5, 25, 50, 75, 95])
            in_5050 = float(np.mean((arr >= 0.45) & (arr <= 0.55)))
            print(
                f"  {name} 开盘价: 中位 {qs[2]:.3f} | p5~p95 [{qs[0]:.3f}, {qs[4]:.3f}] "
                f"| 0.45~0.55 占比 {in_5050:.0%}"
            )
        # 同窗口 UP+DOWN 是否互补（up+down≈1？溢价存在则 >1）
        s = ups + downs
        print(f"  UP+DOWN 和: 中位 {np.median(s):.3f}（>1 为溢价，越接近 1 市场越均衡）")

    # ---- 验证 1+3：信号 → 胜率 → 费后 EV ----
    print(f"\n== 本周期前 {DECISION_SEC:.0f}s 急跌/急涨 → 下周期同向 ==（结算=sign(return)）")
    results = []
    for thr in THRESHOLDS:
        for side, direction in (("down", "DOWN"), ("up", "UP")):
            bets: list[tuple[bool, float, str]] = []  # (win, price, price_kind)
            for a, b in pairs:
                if a["ret"] is None or float(a["ret"]) == 0.0 or b["ret"] is None or float(b["ret"]) == 0.0:
                    continue
                fl = _first_last(a["btc"], a["start"], DECISION_SEC)
                if fl is None:
                    continue
                d = fl[1] / fl[0] - 1.0 if fl[0] else None
                if d is None:
                    continue
                if side == "down" and d > -thr:
                    continue
                if side == "up" and d < thr:
                    continue
                b_ret = float(b["ret"])
                win = (b_ret > 0) if direction == "UP" else (b_ret < 0)
                price = _open_price_early(
                    b["down_price"] if direction == "DOWN" else b["up_price"], b["start"]
                )
                kind = "real"
                if price is None:
                    # proxy：下一周期开盘 chance（up_pct 首点 /100，down 用互补）
                    p = _open_price_early(b["up_pct"], b["start"])
                    if p is None:
                        continue
                    price = p / 100.0 if direction == "UP" else max(1.0 - p / 100.0, 0.01)
                    kind = "proxy"
                bets.append((win, price, kind))

            if not bets:
                continue
            n = len(bets)
            n_real = sum(1 for _, _, k in bets if k == "real")
            win_rate = sum(1 for w, _, _ in bets if w) / n
            avg_price = sum(p for _, p, _ in bets) / n
            pnls = [_bet_pnl(w, p) for w, p, _ in bets]
            ev = float(np.mean(pnls))
            if n >= 10:
                rng = np.random.default_rng(7)
                idx = rng.integers(0, n, size=(2000, n))
                ci = np.percentile(np.asarray(pnls)[idx].mean(axis=1), [2.5, 97.5])
                lo, hi = float(ci[0]), float(ci[1])
            else:
                lo, hi = float("nan"), float("nan")
            row = {
                "side": side, "threshold": thr, "direction": direction,
                "n": n, "n_real": n_real, "win_rate": round(win_rate, 4),
                "avg_entry": round(avg_price, 4), "ev": round(ev, 4),
                "ev_ci_lower": round(lo, 4), "ev_ci_upper": round(hi, 4),
            }
            results.append(row)
            print(
                f"  信号|{side}|{thr:.1%}| {direction}: 注数 {n}(真实价 {n_real}) "
                f"胜率 {win_rate:.1%} 平均入场 {avg_price:.3f} "
                f"费后EV {ev:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            )

    out = {
        "n_windows": len(wins),
        "n_pairs": len(pairs),
        "n_open_prices": len(open_prices),
        "results": results,
    }
    with open("output/cross_window_arb_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n报告已写入 output/cross_window_arb_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
