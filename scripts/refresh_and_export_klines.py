#!/usr/bin/env python3
"""刷新 720 天 BTCUSDT 5m klines 缓存，并导出 5m / 15m OHLCV CSV。

口径与 scripts/local_720d_validation.py 保持一致：
- 数据源：官方 data-api.binance.vision（5m klines）
- 缓存：output/klines_5m_cache_720d.json（增量补拉）
- 15m = 精确聚合 5m（bucket = ts // 900_000，open=首/high=max/low=min/close=末/volume=sum）

纪律：
- 只保留已收盘 K 线（close_time <= now），未收盘当前柱直接丢弃。
- 末尾不足 3 根 5m 成分的 15m 桶丢弃，并打印行数校验。
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DAYS = 720
API = "https://data-api.binance.vision/api/v3/klines"
CACHE = Path("output/klines_5m_cache_720d.json")
CSV_5M = Path("output/klines_5m_720d.csv")
CSV_15M = Path("output/klines_15m_720d.csv")
HEADER = "timestamp,open,high,low,close,volume\n"


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval=5m&startTime={cur}&endTime={end_ms}&limit=1000"
        batch = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    batch = json.loads(resp.read().decode())
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  重试 {attempt + 1}/3: {e}")
                time.sleep(2)
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 1
        if len(out) % 20000 < 1000:
            print(f"  已拉取 {len(out)} 根 5m ...")
        time.sleep(0.2)
    return out


def load_or_fetch(now_ms: int) -> list[list]:
    start_ms = now_ms - DAYS * 86_400_000
    kl: list[list] = []
    try:
        with open(CACHE, encoding="utf-8") as f:
            kl = json.load(f)
        print(f"缓存命中：{len(kl)} 根 5m")
    except Exception:
        pass
    last = int(kl[-1][0]) if kl else 0
    if last < now_ms - 2 * 86_400_000:       # 缓存过期 → 全量
        print("缓存过期，全量重拉 ...")
        kl = fetch_klines(start_ms, now_ms)
    elif last < now_ms - 300_000:            # 增量补拉
        print(f"增量补拉：{last} -> {now_ms} ...")
        kl += fetch_klines(last + 1, now_ms)
    else:
        print("缓存已新鲜（末根距今 <5 分钟），跳过拉取")
    # 只保留已收盘 K 线（close_time <= now）且位于 720 天窗口内
    kl = [k for k in kl if int(k[6]) <= now_ms and int(k[0]) >= start_ms]
    seen = {}
    for k in kl:
        seen[int(k[0])] = k
    kl = [seen[t] for t in sorted(seen)]
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(kl, f)
    return kl


def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def export_5m(kl: list[list]) -> None:
    with open(CSV_5M, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        for k in kl:
            f.write(f"{iso_utc(int(k[0]))},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]}\n")
    print(f"5m CSV：{len(kl)} 根 -> {CSV_5M}")


def export_15m(kl: list[list]) -> None:
    buckets: dict[int, list[list]] = {}
    order: list[int] = []
    for k in kl:
        b = int(k[0]) // 900_000 * 900_000
        if b not in buckets:
            buckets[b] = []
            order.append(b)
        buckets[b].append(k)
    rows, dropped = [], 0
    for b in order:
        bars = buckets[b]
        if len(bars) < 3:  # 末尾不完整 15m 柱
            dropped += 1
            continue
        rows.append(
            (
                b,
                float(bars[0][1]),
                max(float(x[2]) for x in bars),
                min(float(x[3]) for x in bars),
                float(bars[-1][4]),
                sum(float(x[5]) for x in bars),
            )
        )
    with open(CSV_15M, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER)
        for ts, o, h, l, c, v in rows:
            f.write(f"{iso_utc(ts)},{o:.8f},{h:.8f},{l:.8f},{c:.8f},{v:.8f}\n")
    print(f"15m CSV：{len(rows)} 根（丢弃不完整桶 {dropped} 个）-> {CSV_15M}")
    expect = len(kl) // 3
    assert abs(len(rows) - expect) <= 1, f"15m 行数校验失败：{len(rows)} vs 预期 ~{expect}"


def main() -> int:
    now_ms = int(time.time() * 1000)
    kl = load_or_fetch(now_ms)
    if not kl:
        raise SystemExit("无数据")
    print(f"数据范围：{int(kl[0][0])} -> {int(kl[-1][0])}（共 {len(kl)} 根 5m）")
    export_5m(kl)
    export_15m(kl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
