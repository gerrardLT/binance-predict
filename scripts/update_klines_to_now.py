#!/usr/bin/env python3
"""增量拉取 2026-09-04 至 2026-09-09 的 5m K 线并合并更新 720d CSV 文件，确保 30d/90d 绝对最新。"""

import json
import time
import httpx
from datetime import datetime, timezone
from pathlib import Path

API = "https://data-api.binance.vision/api/v3/klines"
CSV_5M = Path("output/klines_5m_720d.csv")
CSV_15M = Path("output/klines_15m_720d.csv")

# 读取 5m CSV 末尾时间
with open(CSV_5M, 'r', encoding='utf-8') as f:
    lines = [l.strip().split(',') for l in f if l.strip()]
last_dt_str = lines[-1][0]
last_ms = int(datetime.fromisoformat(last_dt_str).timestamp() * 1000)

print(f"当前 5m CSV 最新时间: {last_dt_str}")

# 拉取增量
now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
cur_ms = last_ms + 300_000

new_bars_5m = []
with httpx.Client(timeout=30) as client:
    while cur_ms < now_ms:
        url = f"{API}?symbol=BTCUSDT&interval=5m&startTime={cur_ms}&limit=1000"
        resp = client.get(url)
        data = resp.json()
        if not data:
            break
        # 只取已收盘
        closed = [d for d in data if d[6] < now_ms]
        if not closed:
            break
        for b in closed:
            ts_iso = datetime.fromtimestamp(b[0]/1000, tz=timezone.utc).isoformat()
            new_bars_5m.append(f"{ts_iso},{float(b[1]):.8f},{float(b[2]):.8f},{float(b[3]):.8f},{float(b[4]):.8f},{float(b[5]):.8f}\n")
        cur_ms = int(closed[-1][0]) + 300_000
        if len(closed) < 1000:
            break

print(f"拉取到 {len(new_bars_5m)} 根新 5m K 线")

if new_bars_5m:
    with open(CSV_5M, 'a', encoding='utf-8') as f:
        f.writelines(new_bars_5m)
    print("已追加到 5m CSV")

# 重新生成 15m CSV
# 读取全量 5m
with open(CSV_5M, 'r', encoding='utf-8') as f:
    f.readline() # header
    all_5m = [l.strip().split(',') for l in f if l.strip()]

# 聚合 15m
buckets_15m = {}
for r in all_5m:
    dt = datetime.fromisoformat(r[0])
    ts_ms = int(dt.timestamp() * 1000)
    b_ms = (ts_ms // 900_000) * 900_000
    o, h, l, c, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
    if b_ms not in buckets_15m:
        buckets_15m[b_ms] = {"ts": b_ms, "o": o, "h": h, "l": l, "c": c, "v": v, "count": 1}
    else:
        b = buckets_15m[b_ms]
        b["h"] = max(b["h"], h)
        b["l"] = min(b["l"], l)
        b["c"] = c
        b["v"] += v
        b["count"] += 1

# 写入 15m CSV (必须满 3 根)
lines_15m = ["timestamp,open,high,low,close,volume\n"]
for b_ms in sorted(buckets_15m.keys()):
    b = buckets_15m[b_ms]
    if b["count"] == 3:
        iso_str = datetime.fromtimestamp(b_ms/1000, tz=timezone.utc).isoformat()
        lines_15m.append(f"{iso_str},{b['o']:.8f},{b['h']:.8f},{b['l']:.8f},{b['c']:.8f},{b['v']:.8f}\n")

with open(CSV_15M, 'w', encoding='utf-8') as f:
    f.writelines(lines_15m)

print(f"已重新聚合生成 15m CSV，总行数: {len(lines_15m)-1}, 最新: {lines_15m[-1].strip().split(',')[0]}")
