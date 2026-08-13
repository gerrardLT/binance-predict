#!/usr/bin/env python3
"""口径对照实验：同一段时间（2026-07-13 ~ 08-13），同一信号，两种口径。

本地一个月数据跑出 85% 胜率（20 注），全周期 K 线口径只有 53%。
已核对出的条件差异：
- 支撑/阻力：本地用"回填曲线收盘点位极值"（1m K 线 close 的 min/max），
  全周期脚本用"K 线盘中 low/high 极值"
- 连续 3 窗：本地用"窗口开收差符号"（实体阴阳），全周期用"收盘 vs 前收盘"

本脚本拉 1m K 线 → 聚合 5m 窗口，四种口径组合在同一区间上跑：
A. 支撑=盘中 low/high，连续=收盘vs前收盘（全周期口径）
B. 支撑=收盘点位 min/max，连续=窗口开收差（本地口径）
C. 支撑=盘中 low/high，连续=窗口开收差（混合）
D. 支撑=收盘点位 min/max，连续=收盘vs前收盘（混合）

结算统一 = 下一窗口开收差符号（与本地 actual_return 口径一致）。
目的：定位 85% vs 53% 的差异来源（口径 or 时段）。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_LOOKBACK = 24  # 2h = 24 个 5m 窗口
_EPS = 0.0005
_K = 3


def _fetch(start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{_KLINES_URL}?symbol=BTCUSDT&interval=1m"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                break
            except Exception:
                time.sleep(2)
        else:
            break
        if not data:
            break
        out.extend(data)
        cursor = data[-1][0] + 60_000
        if len(data) < 1000:
            break
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    start = datetime(2026, 7, 13, tzinfo=timezone.utc)
    end = datetime(2026, 8, 13, tzinfo=timezone.utc)
    print(f"拉取 1m K 线: {start} → {end} ...")
    rows = _fetch(int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    print(f"共 {len(rows)} 根 1m K 线")

    # 聚合 5m 窗口（每 5 根 1m），并保留窗口内 1m close 序列（模拟本地回填曲线）
    wins: list[dict] = []
    for i in range(0, len(rows) - 4, 5):
        grp = rows[i: i + 5]
        wins.append({
            "t": grp[0][0],
            "o": float(grp[0][1]),
            "h": max(float(g[2]) for g in grp),
            "l": min(float(g[3]) for g in grp),
            "c": float(grp[-1][4]),
            "closes": [float(g[4]) for g in grp],  # 本地口径的"曲线点"
        })
    print(f"聚合 5m 窗口 {len(wins)} 个")

    def run(support_from_low: bool, streak_body: bool, label: str) -> None:
        for side, direction in (("down", "UP"), ("up", "DOWN")):
            bets: list[bool] = []
            for idx, w in enumerate(wins):
                if idx <= _LOOKBACK or idx >= len(wins) - 1:
                    continue
                hist = wins[idx - _LOOKBACK: idx]
                if support_from_low:
                    support = min(x["l"] for x in hist)
                    resistance = max(x["h"] for x in hist)
                else:
                    support = min(min(x["closes"]) for x in hist)
                    resistance = max(max(x["closes"]) for x in hist)

                if side == "down":
                    probe = w["l"] if support_from_low else min(w["closes"])
                    broke = probe < support * (1.0 - _EPS)
                    reclaimed = w["c"] >= support
                else:
                    probe = w["h"] if support_from_low else max(w["closes"])
                    broke = probe > resistance * (1.0 + _EPS)
                    reclaimed = w["c"] <= resistance
                if not (broke and reclaimed):
                    continue

                streak_ok = True
                for j in range(1, _K + 1):
                    pw = wins[idx - j]
                    if streak_body:
                        d = pw["c"] - pw["o"]
                    else:
                        d = pw["c"] - wins[idx - j - 1]["c"]
                    if side == "down" and d >= 0:
                        streak_ok = False
                        break
                    if side == "up" and d <= 0:
                        streak_ok = False
                        break
                if not streak_ok:
                    continue

                nxt = wins[idx + 1]
                d = nxt["c"] - nxt["o"]
                if d == 0:
                    continue
                win = (d > 0) if direction == "UP" else (d < 0)
                bets.append(win)

            if not bets:
                print(f"    {side:>4}→{direction}: 0 注")
                continue
            n = len(bets)
            wr = sum(bets) / n
            print(f"    {side:>4}→{direction}: 注数 {n:>3} 胜率 {wr:.1%}")

    print("\n===== A: 支撑=盘中low/high + 连续=收盘vs前收盘（全周期口径）=====")
    run(support_from_low=True, streak_body=False, label="A")
    print("\n===== B: 支撑=收盘点位min/max + 连续=窗口开收差（本地口径）=====")
    run(support_from_low=False, streak_body=True, label="B")
    print("\n===== C: 支撑=盘中low/high + 连续=窗口开收差（混合）=====")
    run(support_from_low=True, streak_body=True, label="C")
    print("\n===== D: 支撑=收盘点位min/max + 连续=收盘vs前收盘（混合）=====")
    run(support_from_low=False, streak_body=False, label="D")
    return 0


if __name__ == "__main__":
    sys.exit(main())
