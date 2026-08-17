#!/usr/bin/env python3
"""行情分段对比实验：360 天全量 vs 已记录数据（真实赔率 EV）。

Part A（360 天）：
- 行情分段 = 事件周期开盘时刻的**过去 24h 收益**（无未来函数）：
  ret24 > +1% → 上行段；< -1% → 下行段；否则震荡段
- 线上 ACTIVE 参数（v1）跑 build_events，每段分别统计两场景胜率 + Wilson CI
- EV 只能按固定 @0.50 假设（360 天无真实报价）

Part B（已记录数据 prediction_market_samples.json，5m 市场真实报价）：
- 重建记录窗口内的场景事件（前置 10 天预热）
- 入场价 = z 曲面在 z=0（次周期开盘瞬间）的市场隐含价（e_down_factory）
- 胜率 + 真实赔率 EV（stats.ev 口径：赎回 0.98，成本 e+0.01）
- 同时给 @0.50 假设口径作对照

用法：python scripts/regime_ev_compare.py [--days 360]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

from binance_predict.backtest import (
    build_events, build_surface, e_down_factory, ev, fetch_klines, wilson,
)
from binance_predict.services.scene_params import SceneParams

ROOT = "."
TREND_TH = 0.01   # ±1% 过去24h收益 → 趋势段门槛


def regime_of(ret24: float | None) -> str:
    if ret24 is None:
        return "未知"
    if ret24 > TREND_TH:
        return "上行"
    if ret24 < -TREND_TH:
        return "下行"
    return "震荡"


def trailing_24h(c5: list[tuple], cycs: list[int]) -> dict[int, float | None]:
    """每个 15m 周期开盘时刻的过去 24h 收益（无未来函数）。"""
    close_by_ts = {r[0]: r[4] for r in c5}
    out: dict[int, float | None] = {}
    for cyc in cycs:
        open_ms = cyc * 900_000
        base = close_by_ts.get(open_ms - 86_400_000)  # 24h 前那根 5m 的收盘
        op = close_by_ts.get(open_ms - 300_000)       # 开盘前最后一根 5m 收盘 ≈ 开盘价基准
        if base and op:
            out[cyc] = op / base - 1
        else:
            out[cyc] = None
    return out


def scene_rows(events: list[dict], ret24: dict[int, float | None]) -> dict:
    """按 regime 分组统计两场景：n / 胜率 / Wilson CI / 按月。"""
    out = {}
    for tag, win_of in (("scene1", lambda e: bool(e["next_down"])),
                        ("scene2", lambda e: not bool(e["next_down"]))):
        hits = [e for e in events if e.get(tag) and e.get("has_next")]
        groups: dict[str, list[bool]] = defaultdict(list)
        by_month: dict[str, list[bool]] = defaultdict(list)
        for e in hits:
            w = win_of(e)
            groups[regime_of(ret24.get(e["cyc"]))].append(w)
            by_month[e["month"]].append(w)
        segs = {}
        for g, ws in sorted(groups.items()):
            n = len(ws)
            p = sum(ws) / n
            lo, hi = wilson(p, n)
            segs[g] = {"n": n, "k": sum(ws), "p": round(p, 4),
                       "ci": [round(lo, 4), round(hi, 4)],
                       "ev_050": round(ev(p, 0.5), 4)}
        months = {m: {"n": len(ws), "p": round(sum(ws) / len(ws), 4)}
                  for m, ws in sorted(by_month.items())}
        out[tag] = {"total": len(hits), "segments": segs, "monthly": months}
    return out


def part_a(days: int) -> tuple[dict, list[tuple], int]:
    now_ms = int(time.time() * 1000)
    cache = f"{ROOT}/output/regime_cache_{days}d.pkl"
    import os, pickle
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            c5 = pickle.load(f)
        print(f"[A] 命中缓存 {cache}（{len(c5)} 根）")
    else:
        print(f"[A] 拉取 {days} 天 5m K 线 ...")
        t0 = time.monotonic()
        kl = fetch_klines("5m", now_ms - days * 86_400_000, now_ms)
        c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
        if c5 and c5[-1][0] + 300_000 > now_ms:
            c5.pop()
        with open(cache, "wb") as f:
            pickle.dump(c5, f)
        print(f"[A] K 线就绪 {len(c5)} 根 | {time.monotonic() - t0:.0f}s")

    res = build_events(c5, SceneParams(), now_ms)
    ret24 = trailing_24h(c5, res["agg"]["cycs"])
    stats = scene_rows(res["events"], ret24)
    return stats, c5, now_ms


def part_b(c5_hint: list[tuple] | None = None) -> None:
    import json
    with open(f"{ROOT}/prediction_market_samples.json", encoding="utf-8") as f:
        samples = json.load(f)
    ts_all = [int(s["timestamp"]) for s in samples]
    lo_t, hi_t = min(ts_all), max(ts_all)
    lo_cyc, hi_cyc = lo_t // 900_000, hi_t // 900_000
    print(f"\n[B] 已记录数据窗口：{time.strftime('%m-%d %H:%M', time.gmtime(lo_t / 1000))} → "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(hi_t / 1000))} UTC | {len(samples)} 条采样")

    # 重建事件（前置 10 天预热，保证 vol_ma / 4h 级别位势成熟）
    now_ms = hi_t + 900_000
    start_ms = lo_t - 10 * 86_400_000
    # 360 天数据已覆盖该窗口则复用，否则单独拉
    if c5_hint and c5_hint[0][0] <= start_ms:
        c5 = c5_hint
        print("[B] 复用 Part A 的 K 线")
    else:
        kl = fetch_klines("5m", start_ms, hi_t + 900_000)
        c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]

    res = build_events(c5, SceneParams(), now_ms)
    ret24 = trailing_24h(c5, res["agg"]["cycs"])
    events = [e for e in res["events"] if lo_cyc <= e["cyc"] <= hi_cyc]

    # z 曲面（真实报价 → 市场隐含价），入场 = 次周期开盘 z=0
    surf, _sigma5 = build_surface(ROOT)
    e_down = e_down_factory(surf)
    e0_down = e_down(0.0, 1)   # τ后半区（开盘瞬间 τ→1）z=0 的 DOWN 市场价
    e0_up = 1.0 - e_down(0.0, 1)
    fallback = 0.5 if not surf else None

    print(f"[B] 曲面就绪 {len(surf)} 格 | z=0 市场价 DOWN={e0_down:.3f} UP={e0_up:.3f}"
          + (" | ⚠ 曲面为空，回退 0.5" if fallback else ""))
    e_s1 = e0_down if surf else 0.5   # 场景①买 DOWN
    e_s2 = e0_up if surf else 0.5     # 场景②买 UP

    stats = scene_rows(events, ret24)
    for tag, e_entry, side in (("scene1", e_s1, "DOWN"), ("scene2", e_s2, "UP")):
        st = stats[tag]
        print(f"\n[B] {tag}（买 {side}，入场价={e_entry:.3f} 真实市场价口径）| 窗口内事件 n={st['total']}")
        for g, seg in st["segments"].items():
            if seg["n"] == 0:
                continue
            ev_real = ev(seg["p"], e_entry)
            print(f"    {g}段 n={seg['n']:>3} 胜率={seg['p']:.1%} CI={seg['ci']} | "
                  f"EV(真实价)={ev_real:+.3f}  EV(@0.50假设)={seg['ev_050']:+.3f}")
    return stats


def print_part_a(stats: dict) -> None:
    for tag, label in (("scene1", "场景① bull_exhaust（买 DOWN）"),
                       ("scene2", "场景② bear_exhaust（买 UP）")):
        st = stats[tag]
        print(f"\n[A] {label} | 360 天事件总数 n={st['total']}（EV 按 @0.50 固定假设）")
        for g, seg in st["segments"].items():
            print(f"    {g}段 n={seg['n']:>4} 胜率={seg['p']:.1%} CI={seg['ci']} EV@0.50={seg['ev_050']:+.3f}")
        m = st["monthly"]
        worst = min(m.items(), key=lambda kv: kv[1]["p"])
        best = max(m.items(), key=lambda kv: kv[1]["p"])
        print(f"    月度范围：最差 {worst[0]} {worst[1]['p']:.1%}(n={worst[1]['n']}) ~ "
              f"最好 {best[0]} {best[1]['p']:.1%}(n={best[1]['n']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=360)
    args = ap.parse_args()

    print("=" * 72)
    print(f"Part A：{args.days} 天全量 × 行情分段（过去24h收益 ±{TREND_TH:.0%} 分趋势/震荡）")
    print("=" * 72)
    stats_a, c5, _ = part_a(args.days)
    print_part_a(stats_a)

    print("\n" + "=" * 72)
    print("Part B：已记录数据（真实市场报价）回测 —— 胜率 + 真实赔率 EV")
    print("=" * 72)
    part_b(c5_hint=c5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
