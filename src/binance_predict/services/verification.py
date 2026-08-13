"""科学发现系统 —— 假设验证内核（宪法 Q6/Q7）。

对 LLM 提出的谓词假设做统计审判。全程无 LLM、纯函数：同一输入必得同一输出。

核心设计（对应 .kiro/specs/scientific-discovery/design.md）：
- lift 体系替代绝对胜率：lift = P(outcome|命中) / P(outcome|局部基准)
- 局部时间基准（Q6-b）：每个命中窗口前后各 3 天为对照组，合并对照池——
  "永远猜多数方向" lift 恒=1，天然免疫 regime 漂移
- 双轨准入（Q6-a）：lift≥1.4 且 CI 下界>1 → ACTIVE；1.15≤lift<1.4 → OBSERVE；
  其余 → REJECT
- BH-FDR 控制多重检验（q=0.1），拒绝 Bonferroni 的过度保守
- 双轨死因（Q7-1）：SPURIOUS（假规律，从未显著）/ EXPIRED（过期规律，曾显著后衰减）
- 物理硬约束：288 窗口/天 ⇒ lift<1.2 统计上不可验证，不作为任何判定阈值
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- 宪法参数默认值 ---
LOCAL_BASELINE_DAYS: float = 3.0  # 局部基准前后各取天数（Q6-b）
STRONG_LIFT: float = 1.4  # 强模式 lift 阈值（Q6-a）
WEAK_LIFT: float = 1.15  # 弱模式 lift 下限（观察仓准入，Q6-a）
FDR_Q: float = 0.1  # Benjamini-Hochberg 假阳性控制水平
MIN_DEATH_HITS: int = 20  # 死因判定最小命中样本（Q7-1）
EXPIRE_PEAK_FLOOR: float = 1.3  # 过期判定：历史峰值须曾超过此值（Q7-1）
EXPIRE_RECENT_CEILING: float = 1.1  # 过期判定：近期 lift 须低于此值（Q7-1）

_DAY_MS = 86_400_000


# ============================================================
# Wilson 置信区间（双侧）
# ============================================================


def wilson_bounds(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 双侧置信区间（默认 95%）。

    total<=0 返回 (0.0, 0.0)。与 backtest.wilson_lower_bound 同源，
    本模块自包含实现以满足内核独立性（Phase 1 不动旧链路）。
    """
    if total <= 0:
        return 0.0, 0.0
    correct = max(0, min(int(correct), int(total)))
    n = float(total)
    phat = correct / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return lower, upper


# ============================================================
# 局部时间基准 + lift 检验（Q6-a/b）
# ============================================================


@dataclass(frozen=True)
class LiftResult:
    """一次 lift 检验的完整结果。"""

    lift: float  # P(outcome|命中) / P(outcome|局部基准)
    ci_lower: float  # log-lift 置信区间下界（exp 变换后）
    ci_upper: float
    p_value: float  # 单侧 p 值（H1: lift > 1）
    hit_events: int  # 命中组中 outcome==target 的窗口数
    hit_total: int  # 命中组窗口数
    base_events: int  # 对照池中 outcome==target 的窗口数
    base_total: int  # 对照池窗口数（去重）


def pooled_local_baseline(
    hit_start_times: list[int],
    all_windows: list[dict],
    target_outcome: str,
    window_days: float = LOCAL_BASELINE_DAYS,
) -> tuple[int, int]:
    """构建局部时间基准对照池，返回 (base_events, base_total)。

    对每个命中窗口，取 start_time ∈ [t-3d, t+3d] 内的所有非命中窗口进对照池
    （按 start_time 去重）。命中窗口自身永远不进对照池，避免分子污染分母。

    Args:
        hit_start_times: 命中窗口的 start_time（毫秒）列表
        all_windows: 全部候选窗口 dict，需含 start_time / outcome
        target_outcome: 目标结果（"UP" | "DOWN"）
        window_days: 局部基准半径（天，Q6-b 默认 3）

    Returns:
        (对照池中 target 事件数, 对照池窗口总数)
    """
    hit_set = set(hit_start_times)
    span_ms = int(window_days * _DAY_MS)
    pool: set[int] = set()
    events = 0
    for t in hit_start_times:
        lo, hi = t - span_ms, t + span_ms
        for w in all_windows:
            st = int(w.get("start_time", 0))
            if st in hit_set or st in pool:
                continue
            if lo <= st <= hi:
                pool.add(st)
                if (w.get("outcome") or "").upper() == target_outcome.upper():
                    events += 1
    return events, len(pool)


def _norm_cdf(x: float) -> float:
    """标准正态 CDF（math.erf 实现，避免引入 scipy 依赖）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lift_test(
    hit_events: int,
    hit_total: int,
    base_events: int,
    base_total: int,
    z: float = 1.96,
) -> LiftResult:
    """lift 点估计 + log 变换置信区间 + 单侧 p 值。

    log(lift) 标准误（delta 方法）：
        SE = sqrt(1/hit_events - 1/hit_total + 1/base_events - 1/base_total)
    CI = exp(log(lift) ± z·SE)；p = 1 - Φ(log(lift)/SE)。

    退化防御：任一组为空或零事件时返回保守值（lift=0, p=1.0），
    绝不产生假显著（无静默降级原则）。
    """
    if (
        hit_total <= 0
        or base_total <= 0
        or hit_events <= 0
        or base_events <= 0
    ):
        return LiftResult(
            lift=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            hit_events=hit_events,
            hit_total=hit_total,
            base_events=base_events,
            base_total=base_total,
        )

    p_hit = hit_events / hit_total
    p_base = base_events / base_total
    lift = p_hit / p_base

    # 事件数等于总数时该项方差贡献为 0（phi 系数边界），SE 由剩余项给出
    var_hit = 1.0 / hit_events - 1.0 / hit_total
    var_base = 1.0 / base_events - 1.0 / base_total
    se = math.sqrt(max(var_hit + var_base, 1e-12))

    log_lift = math.log(lift)
    ci_lower = math.exp(log_lift - z * se)
    ci_upper = math.exp(log_lift + z * se)
    p_value = 1.0 - _norm_cdf(log_lift / se)

    return LiftResult(
        lift=lift,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        hit_events=hit_events,
        hit_total=hit_total,
        base_events=base_events,
        base_total=base_total,
    )


def classify_candidate(
    result: LiftResult,
    strong_lift: float = STRONG_LIFT,
    weak_lift: float = WEAK_LIFT,
) -> str:
    """双轨准入分类（Q6-a）。

    Returns:
        "ACTIVE":  lift >= strong_lift 且 CI 下界 > 1（严格验证，直上线）
        "OBSERVE": weak_lift <= lift < strong_lift（观察仓纸面跟踪攒样本）
        "REJECT":  其余
    """
    if result.lift >= strong_lift and result.ci_lower > 1.0:
        return "ACTIVE"
    if weak_lift <= result.lift < strong_lift:
        return "OBSERVE"
    return "REJECT"


# ============================================================
# BH-FDR 多重检验控制（Q6-c 初筛）
# ============================================================


def bh_fdr(p_values: list[float], q: float = FDR_Q) -> list[bool]:
    """Benjamini-Hochberg 假阳性控制。

    对 m 个 p 值排序，找最大 k 使 p_(k) <= (k/m)·q，p_(1..k) 全部判显著。
    返回与输入等长的 bool 列表（True=通过 FDR 控制）。空输入返回空列表。
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    # 找最大 k（1-based）满足 p_(k) <= k/m * q
    max_k = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * q:
            max_k = rank
    passed = [False] * m
    for rank in range(1, max_k + 1):
        passed[indexed[rank - 1][0]] = True
    return passed


# ============================================================
# 双轨死因判定（Q7-1）
# ============================================================

DEATH_SPURIOUS = "SPURIOUS"  # 假规律：从未显著（发现器责任，全量反馈 LLM）
DEATH_EXPIRED = "EXPIRED"  # 过期规律：曾显著后衰减（regime 变迁，触发再发现）
DEATH_ALIVE = "ALIVE"  # 存活（或样本不足以判死）


def diagnose_death(
    recent_lift: float,
    recent_ci_lower: float,
    recent_ci_upper: float,
    peak_lift: float,
    hit_count: int,
    min_hits: int = MIN_DEATH_HITS,
    peak_floor: float = EXPIRE_PEAK_FLOOR,
    recent_ceiling: float = EXPIRE_RECENT_CEILING,
) -> str:
    """模式死因判定（Q7-1 双轨）。

    判定优先级：
    1. 命中数 < min_hits → ALIVE（样本不足，不判死）
    2. 历史峰值 lift >= peak_floor(1.3) 且近期 lift < recent_ceiling(1.1)
       → EXPIRED（曾真实存在，regime 变了；归档 + 记录存活期 + 触发再发现）
    3. 近期 CI 覆盖 1（recent_ci_lower <= 1 <= recent_ci_upper）且历史从未
       显著（peak_lift < peak_floor）→ SPURIOUS（假规律；全量细节反馈 LLM）
    4. 其余 → ALIVE

    Args:
        recent_lift: 最近 min_hits 次命中的 live lift 点估计
        recent_ci_lower / recent_ci_upper: 对应的 log-lift CI
        peak_lift: 上线以来滚动 live lift 的历史峰值
        hit_count: 累计命中次数
    """
    if hit_count < min_hits:
        return DEATH_ALIVE
    if peak_lift >= peak_floor and recent_lift < recent_ceiling:
        return DEATH_EXPIRED
    if (
        recent_ci_lower <= 1.0 <= recent_ci_upper
        and peak_lift < peak_floor
    ):
        return DEATH_SPURIOUS
    return DEATH_ALIVE


# ============================================================
# live 命中序列 lift 汇总（Q7-1 死因判定的数据准备）
# ============================================================

MIN_PEAK_PREFIX: int = 5  # 峰值扫描的最小前缀长度（与 Q6 初筛最小命中数对齐）


@dataclass(frozen=True)
class LiveLiftSummary:
    """live 命中序列的 lift 汇总（diagnose_death 的直接输入）。"""

    recent: LiftResult  # 最近 recent_n 次命中的 lift 检验（含 CI / p 值）
    peak_lift: float  # 前缀 live lift 的历史峰值（前缀长 >= min_prefix）
    hit_count: int  # 累计已结算命中数


def live_lift_summary(
    hits_chrono: list[bool],
    base_events: int,
    base_total: int,
    recent_n: int = MIN_DEATH_HITS,
    min_prefix: int = MIN_PEAK_PREFIX,
) -> LiveLiftSummary:
    """由 live 命中正确性序列计算死因判定所需的 recent / peak lift。

    Args:
        hits_chrono: 模式历次命中的正确性（is_correct），按时间升序
        base_events / base_total: 局部基准对照池（pooled_local_baseline 产出，
            全期共享一份——基准率固定后 lift 比较退化为正确率比较，峰值可 O(N) 扫描）
        recent_n: 「近期」窗口长度（Q7-1：最近 20 次命中）
        min_prefix: 峰值扫描的最小前缀长度（小于此的前缀不参与峰值，防小样本假峰）

    防御：基准池为空 / 零事件时 recent 与 peak 均退化为保守值（lift=0），
    配合 diagnose_death 的 min_hits 门槛，绝不会据此判死（无静默降级）。
    """
    hit_count = len(hits_chrono)
    recent = hits_chrono[-recent_n:] if recent_n > 0 else list(hits_chrono)
    recent_result = lift_test(
        sum(1 for h in recent if h), len(recent), base_events, base_total
    )

    peak_lift = 0.0
    if base_total > 0 and base_events > 0 and hit_count >= min_prefix:
        baseline_rate = base_events / base_total
        cum_correct = 0
        for n, hit in enumerate(hits_chrono, start=1):
            if hit:
                cum_correct += 1
            if n >= min_prefix:
                peak_lift = max(peak_lift, (cum_correct / n) / baseline_rate)

    return LiveLiftSummary(
        recent=recent_result, peak_lift=peak_lift, hit_count=hit_count
    )
