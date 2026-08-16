#!/usr/bin/env python3
"""线上场景信号纸面盈亏审计（按邮件入场方案，不下注阶段的实盘口径积累）。

入场方案（与回测/邮件一致）：
- 场景① bull_exhaust：次周期开盘买 DOWN 半仓 @0.50；若次周期内 BTC 反弹
  ≥+0.10%（high ≥ open×1.001）加仓半仓 @0.27
- 场景② bear_exhaust：次周期开盘买 UP 全仓 @0.50（只开盘买）

成本口径（与回测 ev() 一致）：入场价 e 的实际成本 = e+0.01（溢价），
赢赎回 0.98（2% 手续费）。每 1 单位本金的盈亏：
  @0.50 赢 +0.9216 / 输 -1.0；@0.27 赢 +2.5 / 输 -1.0

用法：python scripts/local_live_pnl_audit.py [--api URL]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone

API = "http://165.154.147.155:8082"
KLINE = "https://data-api.binance.vision/api/v3/klines"

WIN_050 = (0.98 - 0.51) / 0.51   # +0.9216
WIN_027 = (0.98 - 0.28) / 0.28   # +2.5


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def cycle_path(start_ms: int) -> dict:
    """拉目标周期 3 根 5m K，重建 open/high/close。"""
    bars = fetch_json(f"{KLINE}?symbol=BTCUSDT&interval=5m&startTime={start_ms}&limit=3")
    o = float(bars[0][1])
    h = max(float(b[2]) for b in bars)
    c = float(bars[-1][4])
    return {"open": o, "high": h, "close": c}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=API)
    args = ap.parse_args()

    sigs = [s for s in fetch_json(f"{args.api}/api/fake-breakout/signals?limit=200")["signals"]
            if s["pattern"]]
    sigs.sort(key=lambda s: s["signal_time"])
    if not sigs:
        print("无场景信号")
        return 0

    total_pnl = 0.0
    wins = 0
    print(f"===== 场景信号纸面盈亏审计 | {len(sigs)} 条 =====\n")
    for s in sigs:
        path = cycle_path(s["market_start_15m"])
        op = path["open"]
        is_s1 = s["pattern"] == "bull_exhaust"
        direction = "DOWN" if is_s1 else "UP"
        won = (s["settle_outcome"] == "DOWN") if is_s1 else (s["settle_outcome"] == "UP")
        wins += int(won)

        if is_s1:
            # 半仓 @0.50；反弹 +0.10% 触发加仓半仓 @0.27
            added = path["high"] >= op * 1.001
            pnl = 0.5 * (WIN_050 if won else -1.0)
            if added:
                pnl += 0.5 * (WIN_027 if won else -1.0)
            detail = f"半仓@0.50{' + 加仓@0.27' if added else ''}"
        else:
            pnl = WIN_050 if won else -1.0
            detail = "全仓@0.50"

        total_pnl += pnl
        t = datetime.fromtimestamp(s["signal_time"] / 1000, tz=timezone.utc)
        print(f"#{s['id']} {s['pattern']:>13} {t:%m-%d %H:%M} | 买{direction} {detail} | "
              f"{'赢' if won else '输'} | 盈亏 {pnl:+.3f}")

    n = len(sigs)
    wr = wins / n
    print(f"\n合计：{n} 条 | {wins} 胜 {n - wins} 负 | 胜率 {wr:.1%}")
    print(f"纸面总盈亏（每信号 1 单位本金）：{total_pnl:+.3f} | 平均 {total_pnl / n:+.3f}/信号")
    print(f"\n基准对照：场景①验证集 62.0% / 场景② 56.4%（样本外）；"
          f"EV@0.50 全仓 ≈ +0.19/+0.08 每事件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
