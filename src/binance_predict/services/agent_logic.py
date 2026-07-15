"""
情绪曲线自进化 Agent Loop —— 可测纯函数（无 I/O）

本模块抽离 Sentiment_Agent 四阶段中**不依赖任何 I/O**的决策逻辑，作为属性测试
（Property-Based Test）的直接目标（见 design.md「可测纯函数」与「决策 6 验证边界/
判定规则」两节）。

设计约束：
- 纯函数：仅依赖入参，不读全局状态、不做 DB/LLM/网络调用、不打印日志、无副作用。
- 确定性：相同输入恒得相同输出（涉及排序处均带稳定次级键 id，消除并列歧义）。
- 单一事实源：方向/结果/状态枚举复用 models.schemas，避免字面量漂移（用户规则 7）。

各函数对应的 Correctness Property（design.md）：
- compute_is_correct        -> Property 1  （验证判定真值表，Req 4.2/4.3）
- recompute_win_rate        -> Property 2  （win_rate 不变式与重算，Req 4.4）
- should_trade              -> Property 3  （交易门控，Req 10.1/10.2）
- select_retire_candidates  -> Property 5  （进化淘汰策略，Req 5.8/11.3）
- plan_active_patterns      -> Property 10 （ACTIVE 模式选择，Req 2.3/3.3）
- plan_learn_windows        -> Property 11 （Learn 最近窗口选择，Req 2.2）
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..models.schemas import ActualLabel, FinalPrediction, PatternStatus
from .curve_features import cosine_sim

__all__ = [
    "PatternStat",
    "PatternRow",
    "WindowRow",
    "TradeGateContext",
    "compute_is_correct",
    "should_trade",
    "evaluate_trade_gate",
    "recompute_win_rate",
    "select_retire_candidates",
    "plan_active_patterns",
    "plan_learn_windows",
    "is_prediction_stale",
    "compute_pattern_fingerprint",
    "detect_duplicate_pattern",
]


# ============================================================
# 轻量输入结构（承载纯函数所需的最小字段，与 ORM/DB 解耦）
#
# 由调用方（SentimentAgent）从对应 ORM 行映射而来，使纯逻辑不依赖 SQLAlchemy
# 会话即可独立测试（design.md 组件 3「纯业务逻辑与 LLM/DB I/O 分离」）。
# 均为 frozen（不可变，强化"无副作用"）+ slots（轻量）。
# ============================================================


@dataclass(frozen=True, slots=True)
class PatternStat:
    """
    模式统计精简视图 —— 供 select_retire_candidates 使用。

    仅承载淘汰决策所需字段（id/status/win_rate/sample_count），
    对应 pattern_memory 表同名列。
    """

    id: int
    status: PatternStatus
    win_rate: float = 0.0
    sample_count: int = 0
    recent_win_rate: float | None = None  # 最近 N 次预测的胜率（Plan 步骤 13）


@dataclass(frozen=True, slots=True)
class PatternRow:
    """
    模式行精简视图 —— 供 plan_active_patterns 使用（输入组装）。

    仅承载筛选与后续 LLM 序列化所需的标识字段；status 为筛选依据。
    """

    id: int
    status: PatternStatus
    pattern_name: str = ""
    predicted_direction: str = ""


@dataclass(frozen=True, slots=True)
class WindowRow:
    """
    情绪窗口精简视图 —— 供 plan_learn_windows 使用（输入组装）。

    start_time 为毫秒时间戳，对齐 SentimentWindow.start_time（BigInteger）。
    outcome 为 None 或空串表示窗口尚未产生实际结果（不合格，Req 2.2）。
    """

    id: int
    start_time: int
    outcome: ActualLabel | None = None


# ============================================================
# Property 1：验证判定真值表（Req 4.2 / 4.3）
# ============================================================


def compute_is_correct(direction: FinalPrediction, outcome: ActualLabel | None) -> bool:
    """
    计算单次预测是否正确（决策 6 完整真值表）。

    判定规则：
    - direction ∈ {UP, DOWN}：is_correct = (direction == outcome)
      方向命中为真；方向相反或结果为 NOISE 均为假。
    - direction == NO_TRADE：is_correct = (outcome == "NOISE")
      正确规避非行情为真；实际有 UP/DOWN 行情却未预测方向为假（错失机会）。

    参数
    ----
    direction : FinalPrediction
        预测方向，取值 UP | DOWN | NO_TRADE。
    outcome : ActualLabel | None
        实际结果，取值 UP | DOWN | NOISE；未知/未归档时为 None（恒判为不正确）。

    返回
    ----
    bool
        本次预测是否正确。

    说明：direction 为上述枚举以外的非法值时保守返回 False（不静默误判为正确）。
    """
    if direction in ("UP", "DOWN"):
        return direction == outcome
    if direction == "NO_TRADE":
        return outcome == "NOISE"
    return False


# ============================================================
# Property 3：交易门控（Req 10.1 / 10.2）
# ============================================================


@dataclass(frozen=True, slots=True)
class TradeGateContext:
    """扩展交易门控的完整上下文（Plan 步骤 8）。"""

    direction: FinalPrediction
    confidence: float
    auto_trade_enabled: bool
    threshold: float
    matched_pattern_win_rate: float | None = None
    matched_pattern_sample_count: int | None = None
    recent_loss_streak: int = 0
    daily_trade_count: int = 0
    daily_pnl: float = 0.0
    # Fix #19: 告警熟断器阻断标志（由 AlertService 根据成本/失败告警置位）
    alert_blocked: bool = False


def should_trade(
    direction: FinalPrediction,
    confidence: float,
    threshold: float = 0.6,
    auto_trade_enabled: bool = False,
) -> tuple[bool, str]:
    """
    交易门控（向后兼容的简单版）：判定是否应触发交易。

    当且仅当自动交易总开关开启、direction ∈ {UP, DOWN} 且 confidence 严格大于
    threshold 时执行交易；其余情形一律跳过。

    扩展版门控请使用 evaluate_trade_gate(TradeGateContext)。
    """
    if not auto_trade_enabled:
        return False, "自动交易总开关未开启"
    if direction not in ("UP", "DOWN"):
        return False, f"方向为 {direction}，非可交易方向（仅 UP/DOWN 触发交易）"
    if confidence <= threshold:
        return False, f"置信度 {confidence:.4f} 未超过阈值 {threshold}（需严格大于）"
    return True, f"方向 {direction} 且置信度 {confidence:.4f} > 阈值 {threshold}，执行交易"


def evaluate_trade_gate(
    ctx: TradeGateContext,
    min_pattern_samples: int = 5,
    min_pattern_win_rate: float = 0.4,
    max_consecutive_losses: int = 5,
    max_daily_trades: int = 20,
    max_daily_loss_usdt: float = 10.0,
) -> tuple[bool, str]:
    """
    扩展交易门控：在基础门控之上增加模式证据、连续亏损、日内限额检查。

    规则链（依次检查，任一不通过则拒绝）：
    1. 总开关
    2. 方向可交易性
    3. 置信度阈值
    4. 模式证据不足（sample_count 非空且 < min_pattern_samples）
    5. 模式胜率过低（win_rate 非空且 < min_pattern_win_rate）
    6. 连续亏损熔断（recent_loss_streak >= max_consecutive_losses）
    7. 日内交易次数上限（daily_trade_count >= max_daily_trades）
    8. 日内亏损限额（daily_pnl <= -max_daily_loss_usdt）
    """
    # 1. 总开关
    if not ctx.auto_trade_enabled:
        return False, "自动交易总开关未开启"
    # 1.5 告警熔断器阻断（Fix #19）：成本超限/阶段连续失败时拒绝新交易
    if ctx.alert_blocked:
        return False, "告警熔断器已触发（成本超限或阶段连续失败），暂停交易"
    # 2. 方向
    if ctx.direction not in ("UP", "DOWN"):
        return False, f"方向为 {ctx.direction}，非可交易方向"
    # 3. 置信度
    if ctx.confidence <= ctx.threshold:
        return False, f"置信度 {ctx.confidence:.4f} 未超过阈值 {ctx.threshold}"
    # 4. 模式证据不足
    if (
        ctx.matched_pattern_sample_count is not None
        and ctx.matched_pattern_sample_count < min_pattern_samples
    ):
        return False, (
            f"模式样本不足（{ctx.matched_pattern_sample_count} < {min_pattern_samples}）"
        )
    # 5. 模式胜率过低
    if (
        ctx.matched_pattern_win_rate is not None
        and ctx.matched_pattern_win_rate < min_pattern_win_rate
    ):
        return False, (
            f"模式胜率过低（{ctx.matched_pattern_win_rate:.2f} < {min_pattern_win_rate}）"
        )
    # 6. 连续亏损熔断
    if ctx.recent_loss_streak >= max_consecutive_losses:
        return False, (
            f"连续亏损熔断（{ctx.recent_loss_streak} >= {max_consecutive_losses}）"
        )
    # 7. 日内交易次数上限
    if ctx.daily_trade_count >= max_daily_trades:
        return False, (
            f"日内交易次数上限（{ctx.daily_trade_count} >= {max_daily_trades}）"
        )
    # 8. 日内亏损限额
    if ctx.daily_pnl <= -max_daily_loss_usdt:
        return False, (
            f"日内亏损限额（{ctx.daily_pnl:.2f} <= -{max_daily_loss_usdt}）"
        )
    return True, (
        f"方向 {ctx.direction} 且置信度 {ctx.confidence:.4f} > 阈值 {ctx.threshold}，"
        f"全部风控检查通过，执行交易"
    )


# ============================================================
# Property 2：win_rate 不变式与重算（Req 4.4）
# ============================================================


def recompute_win_rate(correct_count: int, sample_count: int) -> float:
    """
    依据命中数与样本数重算历史胜率。

    - sample_count == 0：返回 0.0（无样本时胜率约定为 0）。
    - 否则：返回 correct_count / sample_count。

    在合法输入（0 <= correct_count <= sample_count）下，结果天然落在 [0, 1]，
    这是 win_rate 不变式（design.md Property 2）；correct_count 由 Harness 与
    sample_count 同步递增维护（命中时二者同增、未命中仅 sample_count 增），
    因此二者不会越界。

    参数
    ----
    correct_count : int
        命中次数（is_correct 为真的已验证预测数）。
    sample_count : int
        样本数（已验证预测总数）。

    返回
    ----
    float
        历史胜率，合法输入下范围为 [0, 1]。
    """
    if sample_count == 0:
        return 0.0
    return min(1.0, correct_count / sample_count)


# ============================================================
# Property 5：进化淘汰策略（Req 5.8 / 11.3）
# ============================================================


def select_retire_candidates(
    patterns: Sequence[PatternStat],
    active_cap: int = 30,
    min_sample: int = 5,
) -> list[int]:
    """
    Evolve 阶段淘汰候选选择：返回应被淘汰（RETIRE）的模式 id 列表。

    规则（决策 6 / Property 5）：
    1. 冷启动保护：ACTIVE 模式数 < 3 时不产生任何淘汰（侧重发现，Req 11.3）。
    2. 未超上限：ACTIVE 模式数 <= active_cap 时不因上限淘汰任何模式（Req 5.8）。
    3. 超出上限：仅在 sample_count > min_sample 的 ACTIVE 模式中，按 win_rate 升序
       （最差优先）选出「超额部分」= (ACTIVE 数 - active_cap) 个；样本不足
       （sample_count <= min_sample）的新模式绝不因上限被淘汰。
    4. 合格可淘汰者不足以填满超额时，仅返回全部合格者（不足额淘汰，保护新模式）。

    排序采用 (win_rate, id) 升序，id 作为稳定次级键，保证并列 win_rate 时结果确定。

    参数
    ----
    patterns : Sequence[PatternStat]
        全体模式统计视图（可含 ACTIVE/RETIRED/EVOLVING，仅 ACTIVE 参与计数与淘汰）。
    active_cap : int, 默认 30
        ACTIVE 模式数上限（Req 5.8）。
    min_sample : int, 默认 5
        淘汰保护最小样本数，sample_count <= 该值的模式不因上限被淘汰。

    返回
    ----
    list[int]
        应淘汰的模式 id 列表（可能为空）。
    """
    active = [p for p in patterns if p.status == "ACTIVE"]
    active_count = len(active)

    # 规则 1：冷启动保护（ACTIVE < 3 不淘汰，侧重发现）
    if active_count < 3:
        return []
    # 规则 2：未超上限不淘汰
    if active_count <= active_cap:
        return []

    excess = active_count - active_cap
    # 规则 3：仅在样本充足的 ACTIVE 模式中按综合评分升序（最差优先）淘汰超额部分
    eligible = [p for p in active if p.sample_count > min_sample]

    def _score(p: PatternStat) -> float:
        """多维度淘汰评分（Plan 步骤 13）"""
        effective = p.recent_win_rate if p.recent_win_rate is not None else p.win_rate
        return 0.6 * p.win_rate + 0.4 * effective

    eligible_sorted = sorted(eligible, key=lambda p: (_score(p), p.id))
    # 规则 4：最多淘汰 excess 个，合格可淘汰者不足则全取（excess>=1，切片安全）
    return [p.id for p in eligible_sorted[:excess]]


# ============================================================
# Property 10：ACTIVE 模式选择（Req 2.3 / 3.3）
# ============================================================


def plan_active_patterns(patterns: Sequence[PatternRow]) -> list[PatternRow]:
    """
    输入组装：从混合状态的模式集合中筛出 ACTIVE 子集，供 Learn/Predict 使用。

    结果恰为 status == "ACTIVE" 的模式（不含 RETIRED / EVOLVING），并保持输入原始
    顺序（确定性）。

    参数
    ----
    patterns : Sequence[PatternRow]
        混合状态的模式集合。

    返回
    ----
    list[PatternRow]
        仅含 ACTIVE 状态的模式，顺序与输入一致。
    """
    return [p for p in patterns if p.status == "ACTIVE"]


# ============================================================
# Property 11：Learn 最近窗口选择（Req 2.2）
# ============================================================


def plan_learn_windows(windows: Sequence[WindowRow], limit: int = 50) -> list[WindowRow]:
    """
    输入组装：为 Learn 阶段选取最近的合格情绪窗口。

    合格条件为 outcome 非空（既非 None 也非空串）；在合格窗口中按 start_time 降序
    （最新在前）取前 min(limit, 合格数) 个。排序采用 (start_time, id) 降序，id 作为
    稳定次级键，保证 start_time 并列时结果仍然确定。

    参数
    ----
    windows : Sequence[WindowRow]
        情绪窗口集合（可含 outcome 为空的未归档窗口）。
    limit : int, 默认 50
        选取上限（Req 2.2）；非正值时返回空列表。

    返回
    ----
    list[WindowRow]
        按 start_time 降序排列的前 limit 个合格窗口。
    """
    eligible = [w for w in windows if w.outcome is not None and w.outcome != ""]
    eligible_sorted = sorted(eligible, key=lambda w: (w.start_time, w.id), reverse=True)
    return eligible_sorted[: max(limit, 0)]


# ============================================================
# 预测时效性检查（Plan 步骤 10）
# ============================================================


def is_prediction_stale(
    remaining_seconds: int,
    min_remaining: int = 30,
) -> bool:
    """
    预测距窗口结束是否仍有足够时间执行交易。

    Args:
        remaining_seconds: 距窗口结束的剩余秒数
        min_remaining: 最小剩余秒数阈值（默认 30s）

    Returns:
        True 表示预测已过时（剩余时间不足）
    """
    return remaining_seconds < min_remaining


# ============================================================
# 深度分析窗口压缩（双模式架构）
# ============================================================


@dataclass(frozen=True, slots=True)
class WindowSummary:
    """窗口统计摘要（供深度分析 LLM 输入）。

    将原始曲线压缩为统计特征 + 5 个关键点，大幅减少 token 消耗。
    """

    id: int
    start_time: int
    outcome: str
    actual_return: float
    # UP% 曲线统计
    up_mean: float
    up_std: float
    up_slope: float  # 首尾差值，正=上升
    # DOWN% 曲线统计
    down_mean: float
    down_std: float
    down_slope: float
    # UP% 最大回撤
    max_drawdown: float
    sample_count: int
    # 保留 5 个关键采样点（首、25%、50%、75%、末）
    up_key_points: tuple[str, ...]
    down_key_points: tuple[str, ...]


def _curve_stats(curve: list[dict]) -> tuple[float, float, float, float, tuple[str, ...]]:
    """计算曲线统计：mean, std, slope, max_drawdown, 5 个关键点。"""
    if not curve:
        return 0.0, 0.0, 0.0, 0.0, ()

    values = [p.get("v", 0) for p in curve]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 0 else 0
    std = variance ** 0.5
    slope = values[-1] - values[0] if n >= 2 else 0.0

    # 最大回撤（UP% 下降幅度）
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    # 5 个关键点：首、25%、50%、75%、末
    if n <= 5:
        key_points = tuple(f"{v:.1f}%" for v in values)
    else:
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        key_points = tuple(f"{values[i]:.1f}%" for i in indices)

    return mean, std, slope, max_dd, key_points


def compress_windows_for_deep_learn(
    windows: list[dict],
    target: int = 25,
) -> list[WindowSummary]:
    """
    将全量情绪窗口压缩为代表性样本（供深度分析 LLM 输入）。

    策略：
    1. 按 outcome 分组（UP / DOWN / NOISE）
    2. 每组内按 |actual_return| 排序，均匀取样（确保极端值 + 中间值）
    3. 若总数超过 target，按各组比例抽取

    Args:
        windows: 全量 SentimentWindow 列表（dict 格式）
        target: 压缩后目标窗口数

    Returns:
        压缩后的 WindowSummary 列表
    """
    if not windows:
        return []

    # 按 outcome 分组
    groups: dict[str, list[dict]] = {"UP": [], "DOWN": [], "NOISE": []}
    for w in windows:
        outcome = w.get("outcome", "NOISE") or "NOISE"
        if outcome not in groups:
            outcome = "NOISE"
        groups[outcome].append(w)

    # 计算各组应抽取数量（按比例，确保至少 1 个（如果该组有数据））
    total = len(windows)
    quotas = {}
    for outcome, group in groups.items():
        if not group:
            quotas[outcome] = 0
        else:
            quota = max(1, int(target * len(group) / total))
            quotas[outcome] = min(quota, len(group))

    # 调整配额使总数不超过 target
    while sum(quotas.values()) > target:
        # 从最大组减 1
        max_outcome = max(quotas, key=quotas.get)
        quotas[max_outcome] -= 1

    # 从各组抽取
    result: list[WindowSummary] = []
    for outcome, group in groups.items():
        if not group or quotas.get(outcome, 0) == 0:
            continue

        # 按 |actual_return| 排序，均匀取样
        sorted_group = sorted(
            group, key=lambda w: abs(w.get("actual_return") or 0), reverse=True
        )
        n_select = quotas[outcome]
        if n_select >= len(sorted_group):
            selected = sorted_group
        else:
            # 均匀取样：确保首尾和中间都有
            step = len(sorted_group) / n_select
            selected = [sorted_group[int(i * step)] for i in range(n_select)]

        for w in selected:
            up_curve = w.get("curve_up_pct", []) or []
            down_curve = w.get("curve_down_pct", []) or []
            up_mean, up_std, up_slope, max_dd, up_kp = _curve_stats(up_curve)
            down_mean, down_std, down_slope, _, down_kp = _curve_stats(down_curve)

            result.append(
                WindowSummary(
                    id=w.get("id", 0),
                    start_time=w.get("start_time", 0),
                    outcome=outcome,
                    actual_return=w.get("actual_return") or 0.0,
                    up_mean=up_mean,
                    up_std=up_std,
                    up_slope=up_slope,
                    down_mean=down_mean,
                    down_std=down_std,
                    down_slope=down_slope,
                    max_drawdown=max_dd,
                    sample_count=w.get("sample_count", 0),
                    up_key_points=up_kp,
                    down_key_points=down_kp,
                )
            )

    # 按 start_time 排序（从早到晚）
    result.sort(key=lambda s: s.start_time)
    return result


# ============================================================
# 特征指纹去重（Plan 步骤 14）
# ============================================================

_FINGERPRINT_KEYS = ("trend_direction", "volatility", "start_level", "divergence")

# P1-3：确定性特征向量去重的余弦相似度阈值（同方向且 >= 此值视为重复）
_DEDUP_COSINE_THRESHOLD = 0.95


def compute_pattern_fingerprint(
    curve_features: dict,
    predicted_direction: str,
) -> str | None:
    """
    基于结构化特征的模式指纹（用于去重比对）。

    提取 curve_features 中的结构化键（trend_direction, volatility,
    start_level, divergence）+ predicted_direction 拼接为规范化字符串。

    若所有结构化键均缺失（值为空），返回 None 表示"特征不足以判定"，
    避免空指纹导致的误判。
    """
    parts = []
    for key in _FINGERPRINT_KEYS:
        val = curve_features.get(key, "")
        parts.append(str(val).lower().strip() if val else "")

    # 所有特征均缺失 → 无法生成有效指纹
    if all(p == "" for p in parts):
        return None

    return predicted_direction.upper() + "|" + "|".join(parts)


def detect_duplicate_pattern(
    new_features: dict,
    new_direction: str,
    existing_patterns: list[dict],
) -> int | None:
    """
    方向相同前提下判重，返回重复模式 id（无重复返回 None）。

    P1-3：若新旧模式的 curve_features 均含确定性特征向量 `_feature_vector`，
    用 cosine_sim >= 阈值判重（比对 LLM 自由 JSONB 更稳健）；否则回退到
    结构化字符串指纹（向后兼容 LLM/存量无向量模式）。

    Args:
        new_features: 新模式的 curve_features
        new_direction: 新模式的 predicted_direction
        existing_patterns: [{id, curve_features, predicted_direction}, ...]

    Returns:
        重复模式的 id，无重复返回 None
    """
    new_vec = new_features.get("_feature_vector") if isinstance(new_features, dict) else None
    new_fp = compute_pattern_fingerprint(new_features, new_direction)
    # 既无向量又无有效指纹 → 无法判定
    if new_vec is None and new_fp is None:
        return None
    for pat in existing_patterns:
        if pat.get("predicted_direction", "").upper() != new_direction.upper():
            continue
        pat_features = pat.get("curve_features", {}) or {}
        pat_vec = (
            pat_features.get("_feature_vector")
            if isinstance(pat_features, dict)
            else None
        )
        # 优先用确定性特征向量
        if new_vec is not None and pat_vec is not None:
            if cosine_sim(new_vec, pat_vec) >= _DEDUP_COSINE_THRESHOLD:
                return pat.get("id")
            continue
        # 回退字符串指纹
        if new_fp is None:
            continue
        existing_fp = compute_pattern_fingerprint(
            pat_features,
            pat.get("predicted_direction", ""),
        )
        if existing_fp is not None and new_fp == existing_fp:
            return pat.get("id")
    return None
