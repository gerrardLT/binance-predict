"""确定性归因统计库（LLM 归因系统的「裁决者」层）。

设计原则（2026-10-19 方案）：
1. 纯 numpy + math，不依赖 pandas（CI 环境无 pandas 的教训）。
2. 随机性是一等假设：streak_assessment 是任何深归因的前置门——
   与随机波动一致时 verdict=RANDOM_CONSISTENT，系统禁止 LLM 继续编故事。
3. 显著性默认按「日」去簇（同一走势段/同一天的订单高度相关，不去簇会把
   显著性虚增几个量级——2026-10 形态研究的教训）；块自助法实现。
4. 所有含随机的检验带固定默认种子，结果可复现。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# 聚类标签粒度：同一 UTC 日内的订单视为相关簇（走势段级去簇的务实近似）
_BLOCK_DAY_MS = 86_400_000


# ======================================================================
# 基础分布
# ======================================================================

def binom_p_le(k: int, n: int, p: float) -> float:
    """精确二项分布 P(X <= k)。胜负数检验的基础（连败是否异常）。"""
    if k < 0:
        return 0.0
    if n <= 0 or p <= 0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    k = max(0, min(k, n))
    # 从 P(X=0) 递推，避免大 comb 溢出/慢
    q = 1.0 - p
    pmf = q ** n
    total = pmf
    for i in range(1, k + 1):
        pmf *= (n - i + 1) / i * p / q
        total += pmf
    return min(1.0, max(0.0, total))


def binom_p_ge(k: int, n: int, p: float) -> float:
    """精确二项分布 P(X >= k)（连胜检验）。"""
    return 1.0 - binom_p_le(k - 1, n, p)


def longest_run(flags: Sequence[bool], value: bool = False) -> int:
    """序列中 value 的最长连续段长度（默认数连败）。"""
    best = cur = 0
    for f in flags:
        if bool(f) is value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def runs_test_z(flags: Sequence[bool]) -> float | None:
    """Wald–Wolfowitz runs 检验 z 值（胜负交替是否异常；None=不可计算）。"""
    seq = [bool(f) for f in flags]
    n = len(seq)
    n1 = sum(seq)
    n2 = n - n1
    if n1 == 0 or n2 == 0 or n < 8:
        return None
    runs = 1 + sum(1 for i in range(1, n) if seq[i] != seq[i - 1])
    mu = 2.0 * n1 * n2 / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    if var <= 0:
        return None
    return (runs - mu) / math.sqrt(var)


def normal_sf_two_sided(z: float) -> float:
    """标准正态双侧 p 值（用 erfc，无需 scipy）。"""
    return math.erfc(abs(z) / math.sqrt(2.0))


# ======================================================================
# 随机性前置裁决（任何深归因的第一道门）
# ======================================================================

def streak_assessment(recent_wins: Sequence[bool | None],
                      base_rate: float,
                      alpha: float = 0.05) -> dict:
    """对通道基准胜率检验最近 N 笔是否异常。

    recent_wins: 按时间升序的胜负序列（None=未结算/不计入，先过滤）。
    base_rate: 通道基准胜率（历史全时段或冻结基准）。
    返回 verdict: RANDOM_CONSISTENT | ANOMALOUS_LOSING | ANOMALOUS_WINNING | INSUFFICIENT_DATA
    贴线口径：p 恰好等于 alpha 判 RANDOM_CONSISTENT（严格小于才异常）。
    """
    wins = [bool(w) for w in recent_wins if w is not None]
    n = len(wins)
    k = sum(wins)
    out: dict = {"n": n, "wins": k, "base_rate": base_rate, "alpha": alpha}
    if n < 8 or not (0.0 < base_rate < 1.0):
        out["verdict"] = "INSUFFICIENT_DATA"
        return out
    p_loss = binom_p_le(k, n, base_rate)   # 输得异常多（连败侧）
    p_win = binom_p_ge(k, n, base_rate)    # 赢得异常多（连胜侧）
    z_runs = runs_test_z(wins)
    out.update({
        "binom_p_le": round(p_loss, 5),
        "binom_p_ge": round(p_win, 5),
        "observed_rate": round(k / n, 4),
        "longest_losing_run": longest_run(wins, False),
        "longest_winning_run": longest_run(wins, True),
        "runs_z": round(z_runs, 3) if z_runs is not None else None,
        "runs_p": round(normal_sf_two_sided(z_runs), 5) if z_runs is not None else None,
    })
    if p_loss < alpha:
        out["verdict"] = "ANOMALOUS_LOSING"
    elif p_win < alpha:
        out["verdict"] = "ANOMALOUS_WINNING"
    else:
        out["verdict"] = "RANDOM_CONSISTENT"
    return out


# ======================================================================
# 日块自助（去簇显著性——同日订单相关的稳健处理）
# ======================================================================

def day_blocks(timestamps_ms: Sequence[int]) -> list[int]:
    """把时间戳映射为日簇 id（同 UTC 日=同簇）。"""
    return [int(ts // _BLOCK_DAY_MS) for ts in timestamps_ms]


def block_bootstrap_mean_p(values: Sequence[float],
                           timestamps_ms: Sequence[int],
                           n_boot: int = 2000,
                           seed: int = 20261019) -> dict:
    """按日块自助检验 mean <= 0（H0: 均值非正）。

    聚类原理：同一天的订单共享同一段行情，不是独立样本；自助时以「日」为
    重采样单位，保留簇内相关。p = bootstrap 均值中 <=0 的比例（加 1 平滑）。
    """
    vals = np.asarray([float(v) for v in values], dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0, "p_le_zero": None}
    blocks: dict[int, list[float]] = {}
    for ts, v in zip(timestamps_ms, vals):
        blocks.setdefault(int(ts // _BLOCK_DAY_MS), []).append(v)
    keys = sorted(blocks)
    block_sums = np.array([sum(blocks[b]) for b in keys], dtype=np.float64)
    block_sizes = np.array([len(blocks[b]) for b in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_blocks = len(keys)
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, n_blocks)
        stats[i] = block_sums[idx].sum() / block_sizes[idx].sum()
    p = float((np.sum(stats <= 0.0) + 1.0) / (n_boot + 1.0))
    return {
        "n": int(len(vals)),
        "n_day_blocks": n_blocks,
        "mean": round(float(vals.mean()), 6),
        "p_le_zero": round(p, 4),
        "significant_positive": bool(p < 0.05),
    }


def bucket_compare(groups: dict[str, dict]) -> dict:
    """分桶对比 + Bonferroni 校正（多重比较防护）。

    groups: {bucket_name: {"values": [...], "timestamps_ms": [...]}}
    每桶跑日块自助（vs 0），p 值 × 桶数做 Bonferroni；任何结论只报校正后值。
    """
    m = len([g for g in groups.values() if g.get("values")])
    out: dict = {"n_buckets": m, "buckets": {}}
    if m == 0:
        return out
    for name, g in groups.items():
        vals = g.get("values") or []
        if not vals:
            continue
        res = block_bootstrap_mean_p(vals, g.get("timestamps_ms") or [0] * len(vals))
        res["p_bonferroni"] = round(min(1.0, res["p_le_zero"] * m), 4) if res["p_le_zero"] is not None else None
        out["buckets"][name] = res
    return out


# ======================================================================
# 变化点检测（边衰减的正规统计版：不猜原因，确认「什么时候变了」）
# ======================================================================

def cusum_changepoint(values: Sequence[float]) -> dict:
    """单变化点 CUSUM 估计：|累计和| 最大处为候选断点，返回位置与两侧均值。

    用于 EV/胜率序列的「边何时变了」检测；不做因果解释，只定位结构变化。
    """
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    n = len(arr)
    if n < 10:
        return {"n": n, "index": None}
    d = arr - arr.mean()
    cs = np.cumsum(d)
    idx = int(np.argmax(np.abs(cs[: n - 1])))
    left, right = arr[: idx + 1], arr[idx + 1:]
    # 断点显著性：两侧均值差的日块自助超纲，这里给幅度描述，p 由调用方按需补
    return {
        "n": n,
        "index": idx,
        "mean_before": round(float(left.mean()), 6) if len(left) else None,
        "mean_after": round(float(right.mean()), 6) if len(right) else None,
        "delta": round(float(right.mean() - left.mean()), 6) if len(left) and len(right) else None,
    }


# ======================================================================
# 单笔订单 TCA 式确定性分解（facts 卡的 per-order 部分）
# ======================================================================

def tca_decompose(order: dict) -> dict:
    """单笔订单的确定性盈亏分解。

    order 需含：win(bool|None), avg_price(实际成交), quoted_avg_price(报价),
    amount_usdt, pnl, settle_price(可空)。
    分解口径：
      slippage_bp   — 实际成交价 vs 报价（执行环境维度的直接证据）
      entry_margin  — 成交价 vs 保本价（1 - avg 的赔率空间；0=贴保本）
      pnl_breakdown — 赢单: (1/avg - 1)*amount；输单: -amount；均以 USDT 计
    """
    avg = float(order.get("avg_price") or 0.0)
    quoted = float(order.get("quoted_avg_price") or 0.0)
    amount = float(order.get("amount_usdt") or 0.0)
    pnl = order.get("pnl")
    win = order.get("win")
    out: dict = {
        "slippage_bp": round((avg / quoted - 1.0) * 10_000.0, 2)
        if avg > 0 and quoted > 0 else None,
        "entry_margin": round(1.0 - avg, 4) if avg > 0 else None,
    }
    if avg > 0 and amount > 0 and win is not None:
        theoretical = (1.0 / avg - 1.0) * amount if win else -amount
        out["pnl_theoretical"] = round(theoretical, 4)
        out["pnl_gap"] = round(float(pnl) - theoretical, 4) if pnl is not None else None
    return out
