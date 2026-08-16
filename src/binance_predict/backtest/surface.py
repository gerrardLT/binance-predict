"""z 空间定价曲面（市场隐含概率估计器，对齐 local_entry_timing_v2 方法）。

曲面 = (τ半区, z桶) → {"price": DOWN 中位价, "freq": 实际收阴频率}，
构建自真实预测市场报价样本 + 官方 1m/5m K 对齐。
角色：L2 双零假设的市场定价基准——edge 的最终定义是跑赢本曲面而非跑赢 50%。
"""
from __future__ import annotations

from .data import fetch_klines, load_pm_samples
from .stats import zbin

import numpy as np


def build_surface(root: str) -> tuple[dict, float]:
    """构建 z 曲面。Returns: (surf, sigma5)。

    surf: {(τ半区 0/1, z桶 0~7): {"price", "freq"}}；σ5 为样本期 5m 振幅波动率。
    """
    samples = load_pm_samples(root)
    lo_t, hi_t = min(int(s["timestamp"]) for s in samples), max(int(s["timestamp"]) for s in samples)

    k1 = fetch_klines("1m", lo_t - 600_000, hi_t + 600_000)
    p1 = {int(k[0]): float(k[4]) for k in k1}
    k5s = fetch_klines("5m", lo_t - 900_000, hi_t + 900_000)
    cyc5 = {int(k[0]) // 300_000: (float(k[1]), float(k[4])) for k in k5s}
    sigma5 = float(np.std([(c - o) / o for o, c in cyc5.values() if c != o]))

    rows = []
    for s in samples:
        ts = int(s["timestamp"])
        cid = ts // 300_000
        if cid not in cyc5:
            continue
        op = cyc5[cid][0]
        p = p1.get((ts // 60_000 - 1) * 60_000)
        if p is None or op <= 0:
            continue
        tau = (cid * 300_000 + 300_000 - ts) / 300_000
        if tau <= 0.03:
            continue
        z = (p / op - 1) / (sigma5 * tau ** 0.5)
        rows.append((0 if tau < 0.5 else 1, z, float(s["down_price"]), cid))

    price_tab: dict[tuple, list] = {}
    freq_seen: dict[tuple, dict] = {}
    for tg, z, dp, cid in rows:
        key = (tg, zbin(z))
        price_tab.setdefault(key, []).append(dp)
        freq_seen.setdefault(key, {}).setdefault(cid, (z, None))
    down_of = {c: (cyc5[c][1] < cyc5[c][0]) if cyc5[c][1] != cyc5[c][0] else None for c in cyc5}

    surf = {}
    for tg in (0, 1):
        for zi in range(8):
            key = (tg, zi)
            ps = price_tab.get(key, [])
            cf = freq_seen.get(key, {})
            downs = [down_of[c] for c in cf if down_of.get(c) is not None]
            if len(ps) >= 15 and len(downs) >= 15:
                surf[key] = {
                    "price": float(np.median(ps)),
                    "freq": float(np.mean(downs)),
                    "n": len(downs),
                }
    return surf, sigma5


def e_down_factory(surf: dict) -> "callable":
    """返回 e_down(z, tg) -> DOWN 价格插值函数（曲面缺格回退 0.5）。"""
    edges_mid = [-4.8, -3.0, -1.5, -0.665, 0.665, 1.5, 3.0, 4.4]

    def e_down(z: float, tg: int) -> float:
        xs, ys = [], []
        for zi in range(8):
            cell = surf.get((tg, zi))
            if cell:
                xs.append(edges_mid[zi])
                ys.append(cell["price"])
        if not xs:
            return 0.5
        return float(np.interp(z, xs, ys))

    return e_down
