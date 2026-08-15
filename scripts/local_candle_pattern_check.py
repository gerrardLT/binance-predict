#!/usr/bin/env python3
"""用户策略命题检验：15m K线形态 → 下一根 K线方向/路径（第一性原理实证）。

命题 1（模式一）：长上影实体下跌K → 下一根收阴？（反向：长下影实体上涨K → 收阳？）
  入场假设：下一周期开盘/预开时 token ≈ 0.50（1:1 赔率），打平胜率 = 0.51/0.98 ≈ 52.0%
命题 2（模式二）：上根长上影实体下跌K → 本根"先涨后跌"？量化：
  - 上行冲击幅度分布（先涨多少）
  - 先涨后跌路径占比（t_high < t_low 且收阴）
  - 机械化入场 开盘价×(1+θ) 的成交率与胜率 → 打平 token 价

另检：15m token 在周期开盘后 30s 内的价格分布（模式一入场价的可执行性）。
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
from binance_predict.db.models import PredictionMarketSample, SentimentWindow  # noqa: E402

FEE = 0.02
PREMIUM = 0.01
MIN_RANGE_PCT = 0.04   # 噪音蜡烛过滤：全程振幅 <0.04% 不算形态
WICK_BODY_RATIO = 2.0  # 影线 ≥ 2×实体
WICK_RANGE_MIN = 0.35  # 影线占全程振幅 ≥35%
BODY_RANGE_MIN = 0.15  # 实体占全程振幅 ≥15%（排除十字星）


def _sorted_pairs(curve: list | None) -> list[tuple[int, float]]:
    pts = [(int(p.get("t", 0)), float(p["v"])) for p in (curve or []) if p.get("v") is not None]
    pts.sort()
    return pts


def ev_at(p: float, e: float) -> float:
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


async def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(SentimentWindow.start_time, SentimentWindow.curve_btc_price)
            .order_by(SentimentWindow.start_time)
        )).all()
        pm_rows = (await session.execute(
            select(PredictionMarketSample.timestamp, PredictionMarketSample.down_price)
            .where(PredictionMarketSample.market_period == "15m")
            .where(PredictionMarketSample.down_price.isnot(None))
        )).all()

    # ---- 重建 15m 蜡烛 ----
    by_cyc: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        cyc = int(r.start_time) // 900_000
        by_cyc.setdefault(cyc, []).extend(_sorted_pairs(r.curve_btc_price))

    candles: list[dict] = []
    for cyc, pts in sorted(by_cyc.items()):
        pts.sort()
        if len(pts) < 6:
            continue
        o, c = pts[0][1], pts[-1][1]
        hi_p = max(pts, key=lambda p: p[1])
        lo_p = min(pts, key=lambda p: p[1])
        candles.append({
            "cyc": cyc, "o": o, "c": c, "h": hi_p[1], "l": lo_p[1],
            "t_hi": hi_p[0], "t_lo": lo_p[0], "t0": pts[0][0], "t1": pts[-1][0],
            "pts": pts,
        })
    print(f"15m 蜡烛总数 {len(candles)}")

    def classify(cd: dict) -> str | None:
        rng = cd["h"] - cd["l"]
        if rng <= 0 or rng / cd["o"] * 100 < MIN_RANGE_PCT:
            return None
        body = cd["c"] - cd["o"]
        upper = cd["h"] - max(cd["o"], cd["c"])
        lower = min(cd["o"], cd["c"]) - cd["l"]
        if body < 0 and upper >= WICK_BODY_RATIO * abs(body) and upper >= WICK_RANGE_MIN * rng and abs(body) >= BODY_RANGE_MIN * rng:
            return "bear_reject"   # 长上影实体下跌
        if body > 0 and lower >= WICK_BODY_RATIO * abs(body) and lower >= WICK_RANGE_MIN * rng and abs(body) >= BODY_RANGE_MIN * rng:
            return "bull_reject"   # 长下影实体上涨
        return None

    n_down = sum(1 for cd in candles if cd["c"] < cd["o"])
    n_dir = sum(1 for cd in candles if cd["c"] != cd["o"])
    base_down = n_down / n_dir if n_dir else 0.5
    print(f"无条件基准：收阴占比 {base_down:.1%}（{n_dir} 根有向蜡烛）")

    # ---- 模式一：形态 → 下一根方向 ----
    print("\n===== 模式一：形态确认 → 下一根方向（入场按 0.50 计，打平胜率 52.0%）=====")
    for pat, want_down, label in (
        ("bear_reject", True, "长上影实体下跌 → 下根收阴"),
        ("bull_reject", False, "长下影实体上涨 → 下根收阳"),
    ):
        hits, total = 0, 0
        for i in range(len(candles) - 1):
            if classify(candles[i]) != pat:
                continue
            nxt = candles[i + 1]
            if nxt["c"] == nxt["o"]:
                continue
            total += 1
            hits += 1 if (nxt["c"] < nxt["o"]) == want_down else 0
        if total >= 5:
            p = hits / total
            lo, hi = np.percentile(
                np.random.default_rng(7).binomial(total, p, size=5000) / total, [2.5, 97.5]
            )
            print(f"  {label}: {hits}/{total} = {p:.1%} [{lo:.1%},{hi:.1%}] | "
                  f"@0.50 EV {ev_at(p, 0.50):+.3f} | 基准 {base_down if want_down else 1 - base_down:.1%}")
        else:
            print(f"  {label}: 样本不足（{total}）")

    # ---- 模式二：上根 bear_reject → 本根路径 ----
    print("\n===== 模式二：上根长上影实体下跌 → 本根路径分析 =====")
    ups, paths, both = [], {"先涨后跌": 0, "直接跌": 0, "涨不回头": 0, "其他": 0}, 0
    entries: dict[float, list[bool]] = {0.02: [], 0.04: [], 0.06: []}
    for i in range(len(candles) - 1):
        if classify(candles[i]) != "bear_reject":
            continue
        nxt = candles[i + 1]
        if nxt["c"] == nxt["o"]:
            continue
        both += 1
        o = nxt["o"]
        up_exc = (nxt["h"] / o - 1.0) * 100
        ups.append(up_exc)
        closed_down = nxt["c"] < o
        first_up = nxt["t_hi"] < nxt["t_lo"] and up_exc >= 0.02
        if first_up and closed_down:
            paths["先涨后跌"] += 1
        elif closed_down and up_exc < 0.02:
            paths["直接跌"] += 1
        elif not closed_down and up_exc >= 0.02:
            paths["涨不回头"] += 1
        else:
            paths["其他"] += 1
        # 机械化入场：价格首次触及 o×(1+θ) 买 DOWN，赢 = 收阴
        for th in entries:
            fill_px = o * (1 + th / 100)
            if nxt["h"] >= fill_px:
                entries[th].append(closed_down)
    if both:
        arr = np.asarray(ups)
        print(f"  样本 {both} | 上行冲击%: 中位 {np.median(arr):.3f} p25 {np.percentile(arr, 25):.3f} "
              f"p75 {np.percentile(arr, 75):.3f} 最大 {arr.max():.3f}")
        print(f"  路径分布: " + " | ".join(f"{k} {v} ({v / both:.0%})" for k, v in paths.items()))
        print("  机械化入场（触及 开盘价×(1+θ) 即买 DOWN）：")
        for th, res in entries.items():
            if not res:
                print(f"    θ={th}%: 0 次成交")
                continue
            p = sum(res) / len(res)
            be = (1 - FEE) * p - PREMIUM
            print(f"    θ={th}%: 成交率 {len(res) / both:.0%} 胜率 {p:.1%} → 打平 token 价 ≤ {be:.3f}")

    # ---- 15m token 周期开盘后 30s 价格分布（模式一入场价可执行性）----
    early = [float(r.down_price) for r in pm_rows if int(r.timestamp) % 900_000 <= 30_000]
    if early:
        arr = np.asarray(early)
        print(f"\n15m DOWN token 周期开盘后≤30s 报价（{len(arr)} 个采样）: "
              f"中位 {np.median(arr):.3f} p10 {np.percentile(arr, 10):.3f} p90 {np.percentile(arr, 90):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
