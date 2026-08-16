"""科学回测统计内核（纯函数，无 I/O）——四层检验框架的数学组件。

L1 随机漫步零假设：Lo & MacKinlay (1988) 方差比检验 VR(q)，含异方差稳健 z*(q)
L3 统计推断：Wilson 置信区间 / 精确二项检验 / 功效预检 / 多重检验门槛
EV 口径：p×((1-FEE)/(e+PREMIUM)-1)-(1-p)，与线上入场模型一致
"""
from __future__ import annotations

import math

FEE = 0.02
PREMIUM = 0.01

# z 空间分桶（与 scripts/local_entry_timing_v2.py 的曲面口径一致）
Z_EDGES = [-4.0, -2.0, -1.0, -0.33, 0.33, 1.0, 2.0, 4.0]


def ev(p: float, e: float) -> float:
    """事件 EV：p=胜率，e=入场价（赎回 1.0 扣 FEE，成本 e+PREMIUM）。"""
    return p * ((1 - FEE) / (e + PREMIUM) - 1.0) - (1 - p)


def wilson(p: float, n: int) -> tuple[float, float]:
    """Wilson score 置信区间（95%）。"""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054
    ph = p + z * z / (2 * n)
    denom = 1 + z * z / n
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, (ph - half * denom) / denom), min(1.0, (ph + half * denom) / denom))


def zbin(z: float) -> int:
    """z 值 → 曲面桶序号（0~7；末桶并掉 [4,+∞)，避免 8 边 9 区间越界）。"""
    for i in range(7):
        lo, hi = Z_EDGES[i], Z_EDGES[i + 1]
        if lo <= z < hi:
            return i + 1
    return 0 if z < Z_EDGES[0] else 7


def exact_binomial_p(k: int, n: int, p0: float = 0.5) -> float:
    """精确二项检验双侧 p 值（观测 k 胜 / n 次 vs 零假设胜率 p0）。

    n ≤ 2000 用精确法（求和所有不比观测更可能的尾部概率）；
    更大 n 退化为正态近似（连续性校正）。
    """
    if n <= 0:
        return 1.0
    if n <= 2000:
        pk = math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k)
        # 双侧：所有概率 ≤ 观测概率的结局之和（小样本精确，无须对称假设）
        p_sum = 0.0
        for i in range(n + 1):
            pi = math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
            if pi <= pk * (1 + 1e-9):
                p_sum += pi
        return min(1.0, p_sum)
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    zstat = abs(k - mu) - 0.5
    if sd <= 0:
        return 1.0
    return min(1.0, 2 * (1 - 0.5 * (1 + math.erf(zstat / (sd * math.sqrt(2))))))


def min_detectable_effect(n: int, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """给定 n 的最小可检测效应（单侧 α、目标功效下的 pp 上限，正态近似）。

    解释：真实优势小于该值时，本样本量大概率检不出来——测了也白测。
    """
    if n <= 0:
        return 1.0
    z_a, z_b = 1.6448536269514722, 0.8416212335729143  # α=0.05 单侧 / 功效 0.8
    return (z_a + z_b) * math.sqrt(p0 * (1 - p0) / n)


def power_preflight(n: int, claimed_effect_pp: float, p0: float = 0.5) -> dict:
    """功效预检：样本量能否检出假设声称的效应幅度。

    Args:
        n: 验证集样本量
        claimed_effect_pp: 假设声称的胜率改善（百分点）

    Returns:
        {"n", "min_detectable_pp", "claimed_pp", "verdict": "OK" | "INSUFFICIENT_POWER", "note"}
    判定：可检测下限 ≤ 声称效应 → OK（80% 功效下检得出）；否则样本不足不下结论。
    """
    mde = min_detectable_effect(n, p0)
    ok = mde * 100 <= max(claimed_effect_pp, 0.01)
    return {
        "n": n,
        "min_detectable_pp": round(mde * 100, 2),
        "claimed_pp": claimed_effect_pp,
        "verdict": "OK" if ok else "INSUFFICIENT_POWER",
        "note": (
            f"n={n} 可检测下限 {mde*100:.2f}pp vs 声称 {claimed_effect_pp}pp（80% 功效）"
            + ("" if ok else "——声称效应测不出，不下结论（先攒样本或提高声称门槛）")
        ),
    }


def multiple_testing_threshold(base_pp: float, n_hypotheses: int) -> dict:
    """多重检验预算：门槛随累计假设数上调（Bonferroni 近似）。

    LLM 每轮可提多个假设，跑多了总有碰巧达标的——门槛按
    α/n_hypotheses 校正（效应量门槛近似放大 √(n_hyp) 倍）。
    """
    n_hyp = max(1, n_hypotheses)
    factor = math.sqrt(n_hyp)
    return {
        "base_pp": base_pp,
        "n_hypotheses": n_hyp,
        "required_pp": round(base_pp * factor, 2),
        "note": f"累计 {n_hyp} 个假设 → 改善门槛 {base_pp}pp → {base_pp*factor:.2f}pp（Bonferroni 近似）",
    }


def variance_ratio(returns: list[float], q: int) -> dict:
    """Lo-MacKinlay 方差比检验：VR(q) = Var(r_q)/(q·Var(r_1))。

    随机漫步零假设下 VR(q)→1；VR>1 动量（正自相关），VR<1 均值回归。
    z 为同方差统计量，z_star 为异方差稳健统计量（Lo & MacKinlay 1988, RB 2.3）。
    """
    n = len(returns)
    if n < 4 * q or q < 2:
        return {"q": q, "vr": None, "z": None, "z_star": None, "n": n}

    mu = sum(returns) / n
    m = q * (n - q + 1) * (1 - q / n)
    var1 = sum((r - mu) ** 2 for r in returns) / (n - 1)
    if var1 <= 0:
        return {"q": q, "vr": None, "z": None, "z_star": None, "n": n}

    # q 期收益（重叠采样，对齐 LM 原文）；σ̂²(q) 已按 m 归一到单期尺度
    varq = sum(
        (sum(returns[i + j] for j in range(q)) - q * mu) ** 2 for i in range(n - q + 1)
    ) / m
    vr = varq / var1

    # 同方差 z
    z = math.sqrt(n * q) * (vr - 1) / math.sqrt(2 * (q - 1) * (2 * q - 1) / (3 * q)) if vr == vr else None

    # 异方差稳健 z*(q)（delta 法，theta(k) 为 k 阶自相关系数的样本估计）
    def delta(k: int) -> float:
        num = sum((returns[i] - mu) * (returns[i + k] - mu) for i in range(n - k))
        den = sum((r - mu) ** 2 for r in returns)
        return num / den if den > 0 else 0.0

    theta = sum((2 * (q - k) / q) ** 2 * delta(k) ** 2 for k in range(1, q))
    if theta >= 0:
        z_star = math.sqrt(n) * (vr - 1) / math.sqrt(theta) if theta > 0 else None
    else:
        z_star = None

    return {"q": q, "vr": round(vr, 4), "z": round(z, 3) if z is not None else None,
            "z_star": round(z_star, 3) if z_star is not None else None, "n": n}
