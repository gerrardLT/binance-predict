#!/usr/bin/env python3
"""S2 条件单影子信号 720d 冻结计数 + 胜率审计锚点（预注册口径事实源）。

复现两版影子信号的 720d 触发计数与「价-only」条件组胜率（纯价格结果 c>o，硬数字，
不受报价表口径偏差影响），供 main.py SHADOW_BENCH 与 settings 注释引用：
    s2_cond_t4_v1  : S2(bear_exhaust) 次周期 t=4(240s) 1m收盘 < 周期开盘（全深度）→ 押 UP
    s2_cond_t5d_v1 : S2 次周期 t=5(300s) 0<ln(开盘/px5)<15bp（中度回落剔深）→ 押 UP

数据：output/klines_5m_cache_720d.json（S2 检测 + 胜率）+ output/klines_1m_cache_720d.json
（t=4/t=5 逐分钟价格）。口径与 .pytest_tmp/verify_s2_freq.py 逐字一致（研究事实源）；
研究 EV（+0.237/+0.283）属报价表乐观上界，本脚本只钉触发数与价-only 胜率，EV 由
生产真实报价前向现算（面板聚合），故此处不算 EV。

用法：.venv\\Scripts\\python.exe scripts/s2_cond_freeze_counts_720d.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

KL5 = "output/klines_5m_cache_720d.json"
KL1 = "output/klines_1m_cache_720d.json"
# ---- S2(bear_exhaust) 冻结检测参数（与 fake_breakout 场景研究同源）----
LOOKBACK = 48
EPS = 0.0005
VOL_RATIO_MIN = 2.0
VOL_MA_WINDOW = 20
DIP_MAX = 0.0015  # 15bp：t=5 剔深上界


def roll_min(x: np.ndarray, w: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    out = np.full(len(x), np.nan)
    out[w - 1:] = sliding_window_view(x, w).min(axis=1)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    now_ms = int(time.time() * 1000)
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    # 5m 行格式：[openTime, open, high, low, close, volume, ...]
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()  # 丢未收盘末根
    # 5m → 15m 聚合（仅取满 3 根的完整周期）
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(c5):
        buckets[int(k[0]) // 900_000].append(i)
    agg: dict[int, dict] = {}
    for cyc, idxs in buckets.items():
        if len(idxs) == 3:
            idxs.sort()
            agg[cyc] = {"o": float(c5[idxs[0]][1]), "l": min(c5[i][3] for i in idxs),
                        "c": float(c5[idxs[-1]][4]), "v": sum(c5[i][5] for i in idxs)}
    cyc_list = sorted(agg)
    o = np.array([agg[c]["o"] for c in cyc_list])
    c = np.array([agg[c]["c"] for c in cyc_list])
    v = np.array([agg[c]["v"] for c in cyc_list])
    # 量比 = 本周期量 / 前 20 周期均量
    cs = np.concatenate([[0.0], np.cumsum(v)])
    vol_ma = np.full(len(cyc_list), np.nan)
    for j in range(1, len(cyc_list)):
        lo = max(0, j - VOL_MA_WINDOW)
        if j - lo >= VOL_MA_WINDOW // 2:
            vol_ma[j] = (cs[j] - cs[lo]) / (j - lo)
    vol_ratio = v / np.where(vol_ma > 0, vol_ma, np.nan)
    # 破 4h 支撑：5m 收盘 rolling-min(48) 前值 × (1−EPS) 被本根 low 击穿
    cl5 = np.array([r[4] for r in c5])
    l5 = np.array([r[3] for r in c5])
    t5 = np.array([r[0] for r in c5])
    lvl_lo = np.full(len(c5), np.nan)
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    broke_lo5 = l5 < lvl_lo * (1 - EPS)
    cont = np.zeros(len(c5), bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_lo = np.zeros(len(cyc_list), bool)
    for j, cyc in enumerate(cyc_list):
        for i in buckets[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_lo[i]) and broke_lo5[i]:
                broke_lo[j] = True
    s2 = broke_lo & (c < o) & (np.nan_to_num(vol_ratio, nan=0) >= VOL_RATIO_MIN)

    # 1m close 字典（行格式 slim：[openTime, open, high, low, close]）
    with open(KL1, encoding="utf-8") as f:
        k1 = json.load(f)
    close1 = {int(r[0]): float(r[4]) for r in k1}

    # 事件池：S2 命中周期 → 次周期入场（要求次周期连续存在、t=4/t=5 1m 价齐全）
    span_days = (cyc_list[-1] - cyc_list[0]) * 900_000 / 86_400_000
    events: list[dict] = []
    for j in range(len(cyc_list) - 1):
        if not s2[j] or cyc_list[j + 1] != cyc_list[j] + 1:
            continue
        a = agg[cyc_list[j + 1]]
        if a["o"] <= 0:
            continue
        op_ts = cyc_list[j + 1] * 900_000
        px = {m: close1.get(op_ts + (m - 1) * 60_000) for m in (4, 5)}
        if any(p is None for p in px.values()):
            continue
        events.append({"open": a["o"], "px": px, "win": bool(a["c"] > a["o"])})

    N = len(events)
    print(f"720d 跨度 {span_days:.0f} 天 | S2 基线事件 {N} 个 | 基线频率 {N / span_days:.2f} 次/天")
    print(f"基线（无条件）胜率 {sum(e['win'] for e in events) / N:.1%}")

    def dip(e: dict, m: int) -> float:
        return math.log(e["open"] / e["px"][m])  # >0 表示价<开盘（回落）

    variants = [
        ("s2_cond_t4_v1", "t=4 价<开盘(全深度)", lambda e: dip(e, 4) > 0),
        ("s2_cond_t5d_v1", "t=5 剔深(0<回落<15bp)", lambda e: 0 < dip(e, 5) < DIP_MAX),
    ]
    print(f"\n{'version':<16}{'口径':<24}{'触发':>6}{'占基线':>9}{'频率/天':>9}{'胜率(价-only)':>15}")
    for ver, name, fn in variants:
        sub = [e for e in events if fn(e)]
        k = len(sub)
        wr = sum(e["win"] for e in sub) / k if k else float("nan")
        print(f"{ver:<16}{name:<24}{k:>6}{k / N:>9.1%}{k / span_days:>9.2f}{wr:>15.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
