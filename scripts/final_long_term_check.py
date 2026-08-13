#!/usr/bin/env python3
"""最终裁决：本地口径（收盘点位支撑 + 实体连续窗）在全周期上的表现。

对照实验已确认：
- 连续窗两种写法等价
- "收盘点位支撑"（不触盘中极值）比"盘中 low/high 支撑"更挑剔、质量更高
- 本地一个月真实水平 67%（18 注），85% 是采样粒度噪音

本脚本把本地口径放到整段空头周期（2025-10 ~ 2026-08，46 万根 1m K 线）验证：
1m K 线 → 聚合 5m 窗口（保留 closes 序列）→ 本地口径信号 → 分段胜率/EV。

信号（本地口径 B）：
- 支撑/阻力 = 前 24 个 5m 窗口内所有 1m 收盘价的 min/max
- 假突破：本窗口 1m close 序列击穿支撑 0.05% 且窗口收盘收回 → 反向
- 连续 3 窗 = 前 3 窗口开收差同向（实体）
- 结算 = 下一窗口开收差符号；定价 0.5（盈亏平衡 52.0%）
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_LOOKBACK = 288  # 日线支撑 = 前 288 个 5m 窗口（收盘点位口径）
_EPS = 0.0005
_K = 3
_WIN_PNL = (1.0 - 0.02) / (0.5 + 0.01) - 1.0  # +0.9216

_SEGMENTS = (
    ("主跌段(12.6w→6w)", datetime(2025, 10, 6, tzinfo=timezone.utc), datetime(2026, 2, 10, tzinfo=timezone.utc)),
    ("反弹段(6w→7.9w)", datetime(2026, 2, 10, tzinfo=timezone.utc), datetime(2026, 4, 20, tzinfo=timezone.utc)),
    ("二次下跌(7.9w→5.8w)", datetime(2026, 4, 20, tzinfo=timezone.utc), datetime(2026, 6, 25, tzinfo=timezone.utc)),
    ("修复段(5.8w→6.3w)", datetime(2026, 6, 25, tzinfo=timezone.utc), datetime(2026, 8, 13, tzinfo=timezone.utc)),
)


def _fetch(start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cursor = start_ms
    n_req = 0
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
        n_req += 1
        if n_req % 50 == 0:
            print(f"  已拉 {len(out)} 根 ...", flush=True)
            time.sleep(1.0)  # 防限流
        if len(data) < 1000:
            break
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    start = datetime(2025, 10, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, 13, tzinfo=timezone.utc)
    print(f"拉取 1m K 线: {start} → {end} ...")
    rows = _fetch(int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    print(f"共 {len(rows)} 根 1m K 线")

    wins: list[dict] = []
    for i in range(0, len(rows) - 4, 5):
        grp = rows[i: i + 5]
        wins.append({
            "t": grp[0][0],
            "o": float(grp[0][1]),
            "c": float(grp[-1][4]),
            "closes": [float(g[4]) for g in grp],
        })
    print(f"聚合 5m 窗口 {len(wins)} 个")

    seg_of = {}
    for idx, w in enumerate(wins):
        t = datetime.fromtimestamp(w["t"] / 1000, tz=timezone.utc)
        for name, s, e in _SEGMENTS:
            if s <= t < e:
                seg_of[idx] = name
                break

    def stats(bets: list[bool]) -> str:
        if not bets:
            return "0 注"
        n = len(bets)
        wr = sum(bets) / n
        pnls = [_WIN_PNL if x else -1.0 for x in bets]
        ev = float(np.mean(pnls))
        if n >= 8:
            rng = np.random.default_rng(7)
            ix = rng.integers(0, n, size=(2000, n))
            ci = np.percentile(np.asarray(pnls)[ix].mean(axis=1), [2.5, 97.5])
            return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        return f"注数 {n:>4} 胜率 {wr:.1%} 费后EV {ev:+.3f}"

    print("\n===== 5m 粒度 · 日线支撑 · 基线（无连续窗）· 分段 =====")
    for seg_name, _, _ in _SEGMENTS:
        for side, direction in (("down", "UP"), ("up", "DOWN")):
            bets: list[bool] = []
            for idx, w in enumerate(wins):
                if idx <= _LOOKBACK or idx >= len(wins) - 1:
                    continue
                if seg_of.get(idx) != seg_name:
                    continue
                hist = wins[idx - _LOOKBACK: idx]
                support = min(min(x["closes"]) for x in hist)
                resistance = max(max(x["closes"]) for x in hist)

                if side == "down":
                    broke = min(w["closes"]) < support * (1.0 - _EPS)
                    reclaimed = w["c"] >= support
                else:
                    broke = max(w["closes"]) > resistance * (1.0 + _EPS)
                    reclaimed = w["c"] <= resistance
                if not (broke and reclaimed):
                    continue

                nxt = wins[idx + 1]
                d = nxt["c"] - nxt["o"]
                if d == 0:
                    continue
                win = (d > 0) if direction == "UP" else (d < 0)
                bets.append(win)
            print(f"  {seg_name:<18} {side:>4}→{direction}: {stats(bets)}")

    # 全周期合并
    print("\n===== 5m 粒度 · 日线支撑 · 基线（无连续窗）· 全周期 =====")
    for side, direction in (("down", "UP"), ("up", "DOWN")):
        bets: list[bool] = []
        for idx, w in enumerate(wins):
            if idx <= _LOOKBACK or idx >= len(wins) - 1:
                continue
            hist = wins[idx - _LOOKBACK: idx]
            support = min(min(x["closes"]) for x in hist)
            resistance = max(max(x["closes"]) for x in hist)
            if side == "down":
                broke = min(w["closes"]) < support * (1.0 - _EPS)
                reclaimed = w["c"] >= support
            else:
                broke = max(w["closes"]) > resistance * (1.0 + _EPS)
                reclaimed = w["c"] <= resistance
            if not (broke and reclaimed):
                continue
            nxt = wins[idx + 1]
            d = nxt["c"] - nxt["o"]
            if d == 0:
                continue
            win = (d > 0) if direction == "UP" else (d < 0)
            bets.append(win)
        print(f"  {side:>4}→{direction}: {stats(bets)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
