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

__all__ = [
    "PatternStat",
    "PatternRow",
    "WindowRow",
    "compute_is_correct",
    "should_trade",
    "recompute_win_rate",
    "select_retire_candidates",
    "plan_active_patterns",
    "plan_learn_windows",
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


def should_trade(
    direction: FinalPrediction,
    confidence: float,
    threshold: float = 0.6,
    auto_trade_enabled: bool = False,
) -> tuple[bool, str]:
    """
    交易门控：判定是否应触发交易，并给出可追溯的原因说明。

    当且仅当自动交易总开关开启、direction ∈ {UP, DOWN} 且 confidence 严格大于
    threshold 时执行交易；
    其余情形一律跳过，并返回**非空**的跳过原因（记录而非静默降级，用户规则 3）。

    参数
    ----
    direction : FinalPrediction
        预测方向 UP | DOWN | NO_TRADE。
    confidence : float
        预测置信度（约定 0~1）。
    threshold : float, 默认 0.6
        置信度阈值，需严格大于该值方可交易（Req 10.1）。
    auto_trade_enabled : bool, 默认 False
        自动交易总开关；未显式开启时始终不允许下单。

    返回
    ----
    tuple[bool, str]
    (是否交易, 原因说明)。交易与跳过均返回非空说明，便于落库追溯。
    """
    if not auto_trade_enabled:
        return False, "自动交易总开关未开启"
    if direction not in ("UP", "DOWN"):
        return False, f"方向为 {direction}，非可交易方向（仅 UP/DOWN 触发交易）"
    if confidence <= threshold:
        return False, f"置信度 {confidence:.4f} 未超过阈值 {threshold}（需严格大于）"
    return True, f"方向 {direction} 且置信度 {confidence:.4f} > 阈值 {threshold}，执行交易"


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
    # 规则 3：仅在样本充足的 ACTIVE 模式中按 win_rate 升序（最差优先）淘汰超额部分
    eligible = [p for p in active if p.sample_count > min_sample]
    eligible_sorted = sorted(eligible, key=lambda p: (p.win_rate, p.id))
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
