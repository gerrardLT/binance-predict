#!/usr/bin/env python3
"""拉取币安 BTCUSDT 1h K 线近一年，打印月度走势，定位 12.5w→6w 空头段起止。

只做诊断，不写库。输出：
1. 月度 OHLC 表
2. 12.5w 附近高点 / 6w 附近低点的精确时间
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta

_KLINES_URL = "https://api.binance.com/api/v3/klines"


def _fetch(start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"{_KLINES_URL}?symbol=BTCUSDT&interval=1h"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        if not data:
            break
        out.extend(data)
        cursor = data[-1][0] + 3600_000
        if len(data) < 1000:
            break
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 380 * 24 * 3600 * 1000  # 约一年零半个月
    print(f"拉取 1h K 线: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)} 起 ...")
    rows = _fetch(start_ms, now_ms)
    print(f"共 {len(rows)} 根")

    # 月度 OHLC
    months: dict[str, dict] = {}
    for r in rows:
        t = datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc)
        key = t.strftime("%Y-%m")
        m = months.setdefault(key, {"o": float(r[1]), "h": float(r[2]),
                                    "l": float(r[3]), "c": float(r[4])})
        m["h"] = max(m["h"], float(r[2]))
        m["l"] = min(m["l"], float(r[3]))
        m["c"] = float(r[4])
    print("\n月度走势（UTC）:")
    print(f"{'月份':<9}{'开盘':>10}{'最高':>10}{'最低':>10}{'收盘':>10}{'涨跌':>8}")
    for k, m in sorted(months.items()):
        chg = (m["c"] / m["o"] - 1.0) * 100
        print(f"{k:<9}{m['o']:>10.0f}{m['h']:>10.0f}{m['l']:>10.0f}{m['c']:>10.0f}{chg:>+7.1f}%")

    # 全期最高/最低
    hi = max(rows, key=lambda r: float(r[2]))
    lo = min(rows, key=lambda r: float(r[3]))
    print(f"\n全期最高 {float(hi[2]):.0f} @ {datetime.fromtimestamp(hi[0]/1000, tz=timezone.utc)}")
    print(f"全期最低 {float(lo[3]):.0f} @ {datetime.fromtimestamp(lo[0]/1000, tz=timezone.utc)}")

    # 最近收盘
    last = rows[-1]
    print(f"最新收盘 {float(last[4]):.0f} @ {datetime.fromtimestamp(last[6]/1000, tz=timezone.utc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
