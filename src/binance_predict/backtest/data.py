"""官方 K 线数据管道（data-api.binance.vision，合并三脚本实现）。

与 scripts/local_* 系列同一数据源与限速策略；本模块为正式包组件，
供科学回测引擎（CLI 与 hypothesis_arbiter）共用。
"""
from __future__ import annotations

import json
import time
import urllib.request

API = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    """分页拉取 K 线（升序原始数组：[open_time, o, h, l, c, v, ...]）。"""
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
        batch: list = []
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
        time.sleep(0.2)
    return out


def aggregate_15m(c5: list[tuple]) -> dict:
    """5m K → 完整 15m 周期聚合（只保留 3 根齐全的周期）。

    Args:
        c5: [(open_time_ms, o, h, l, c, v), ...] 升序

    Returns:
        {"cycs": [周期号...], "o15"/"h15"/"l15"/"c15"/"v15": np-ready list,
         "buckets": {周期号: [c5 下标...]}, "cont": [5m 连续性 bool...]}
    """
    t5 = [r[0] for r in c5]
    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t // 900_000 for t in t5):
        buckets.setdefault(cyc, []).append(i)
    cyc_list = sorted(c for c in buckets if len(buckets[c]) == 3)
    o15 = [c5[buckets[c][0]][1] for c in cyc_list]
    h15 = [max(c5[i][2] for i in buckets[c]) for c in cyc_list]
    l15 = [min(c5[i][3] for i in buckets[c]) for c in cyc_list]
    c15 = [c5[buckets[c][-1]][4] for c in cyc_list]
    v15 = [sum(c5[i][5] for i in buckets[c]) for c in cyc_list]
    cont = [False] * len(c5)
    for i in range(1, len(c5)):
        cont[i] = (t5[i] - t5[i - 1]) == 300_000
    return {
        "cycs": cyc_list, "o15": o15, "h15": h15, "l15": l15, "c15": c15, "v15": v15,
        "buckets": buckets, "cont": cont,
    }


def load_pm_samples(root: str) -> list[dict]:
    """读 prediction_market_samples.json 的有效 DOWN 报价样本（曲面原料）。"""
    with open(f"{root}/prediction_market_samples.json", encoding="utf-8") as f:
        samples = json.load(f)
    return [
        s for s in samples
        if s.get("down_price") is not None and 0.02 < float(s["down_price"]) < 0.98
    ]
