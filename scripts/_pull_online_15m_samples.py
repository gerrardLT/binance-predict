#!/usr/bin/env python3
"""一次性拉取线上 15m 采样全量历史（since/limit 翻页）。"""
import json
import time
import urllib.request

BASE = "http://165.154.147.155:8082/api/chart/prediction-market/15m"
all_pts = []
since = 1
page = 0
while True:
    url = f"{BASE}?since={since}&limit=50000"
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.loads(r.read().decode())
    pts = d["points"]
    if not pts:
        print(f"page {page + 1}: empty, stop")
        break
    all_pts.extend(pts)
    page += 1
    fmt = lambda ts: time.strftime("%m-%d %H:%M", time.gmtime(ts / 1000))
    print(f"page {page}: {len(pts)} pts, {fmt(pts[0]['timestamp'])} ~ {fmt(pts[-1]['timestamp'])}")
    if len(pts) < 50000:
        break
    since = pts[-1]["timestamp"] + 1

with open("output/online_15m_samples_full.json", "w", encoding="utf-8") as f:
    json.dump(all_pts, f)
print("TOTAL:", len(all_pts))
btc = sum(1 for p in all_pts if p.get("btc_price"))
if all_pts:
    print(f"btc covered: {btc} ({btc / len(all_pts):.0%})")
