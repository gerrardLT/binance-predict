"""
LLM 输出语义验证器 —— 纯函数、分 HARD/SOFT 两级

对 Learn / Predict / Evolve 三阶段 LLM 结构化输出进行业务语义校验，
补充 Instructor 的 Pydantic schema 校验（仅检查枚举值/数值范围）。

分级策略：
- HARD_FAIL：必须拒绝的结构性错误（如引用不存在的模式 ID），调用方应跳过该条操作
- SOFT_WARN：质量不足但非致命的问题（如 reasoning 过短），调用方可选择记录或阻断

所有函数为纯函数：不依赖全局状态、不做 I/O、不打印日志。
"""

from __future__ import annotations

from ..models.schemas import EvolveOutput, LearnOutput, PredictOutput


def validate_learn_output(
    output: LearnOutput,
    active_pattern_ids: set[int],
) -> tuple[list[str], list[str]]:
    """
    校验 Learn 阶段 LLM 输出。

    Args:
        output: LLM 返回的 LearnOutput
        active_pattern_ids: 当前 ACTIVE 模式的 ID 集合

    Returns:
        (hard_failures, soft_warnings) 两个字符串列表
    """
    hard: list[str] = []
    soft: list[str] = []

    # SOFT: reasoning 长度检查
    if len(output.reasoning) < 50:
        soft.append(f"reasoning 过短（{len(output.reasoning)} 字符，建议 >= 50）")

    for i, disc in enumerate(output.discoveries):
        prefix = f"discoveries[{i}]"

        # HARD: UPDATE 的 target_pattern_id 必须存在
        if disc.operation == "UPDATE":
            if disc.target_pattern_id is None:
                hard.append(f"{prefix}: UPDATE 操作缺少 target_pattern_id")
            elif disc.target_pattern_id not in active_pattern_ids:
                hard.append(
                    f"{prefix}: UPDATE target_pattern_id={disc.target_pattern_id} "
                    f"不在 ACTIVE 模式集合中"
                )

        # HARD: CREATE 的 curve_features 不能为空
        if disc.operation == "CREATE":
            if not disc.curve_features:
                hard.append(f"{prefix}: CREATE 的 curve_features 为空")

        # SOFT: description 过短
        if len(disc.description) < 10:
            soft.append(f"{prefix}: description 过短（{len(disc.description)} 字符）")

        # SOFT: change_reason 非空
        if not disc.change_reason.strip():
            soft.append(f"{prefix}: change_reason 为空")

    # SOFT: confidence_score 分布检查（不全为 1.0）
    if output.discoveries and all(d.confidence_score >= 1.0 for d in output.discoveries):
        soft.append("所有 discoveries 的 confidence_score 均为 1.0，可能过度自信")

    return hard, soft


def validate_predict_output(
    output: PredictOutput,
    active_pattern_ids: set[int],
) -> tuple[list[str], list[str]]:
    """
    校验 Predict 阶段 LLM 输出。

    Args:
        output: LLM 返回的 PredictOutput
        active_pattern_ids: 当前 ACTIVE 模式的 ID 集合

    Returns:
        (hard_failures, soft_warnings) 两个字符串列表
    """
    hard: list[str] = []
    soft: list[str] = []

    # HARD: matched_pattern_id 非空时必须存在于 ACTIVE 模式
    if output.matched_pattern_id is not None:
        if output.matched_pattern_id not in active_pattern_ids:
            hard.append(
                f"matched_pattern_id={output.matched_pattern_id} "
                f"不在 ACTIVE 模式集合中（可能已退役或不存在）"
            )

    # SOFT: reasoning 非空
    if not output.reasoning.strip():
        soft.append("reasoning 为空")

    # SOFT: 非 NO_TRADE 时 confidence 不应过低
    if output.direction != "NO_TRADE" and output.confidence < 0.3:
        soft.append(
            f"direction={output.direction} 但 confidence={output.confidence:.4f} < 0.3"
        )

    # SOFT: NO_TRADE 时 matched_pattern 应为空
    if output.direction == "NO_TRADE" and output.matched_pattern_id is not None:
        soft.append("direction=NO_TRADE 但 matched_pattern_id 非空")

    return hard, soft


def validate_evolve_output(
    output: EvolveOutput,
    all_pattern_ids: set[int],
) -> tuple[list[str], list[str]]:
    """
    校验 Evolve 阶段 LLM 输出。

    Args:
        output: LLM 返回的 EvolveOutput
        all_pattern_ids: 全部模式（ACTIVE + 近期 RETIRED）的 ID 集合

    Returns:
        (hard_failures, soft_warnings) 两个字符串列表
    """
    hard: list[str] = []
    soft: list[str] = []

    # SOFT: reasoning 长度检查
    if len(output.reasoning) < 50:
        soft.append(f"reasoning 过短（{len(output.reasoning)} 字符，建议 >= 50）")

    for i, op in enumerate(output.operations):
        prefix = f"operations[{i}]"

        # RETAIN 无需额外校验
        if op.action == "RETAIN":
            continue

        # HARD: MODIFY/RETIRE 的 target_pattern_id 必须存在
        if op.action in ("MODIFY", "RETIRE"):
            if op.target_pattern_id is None:
                hard.append(f"{prefix}: {op.action} 操作缺少 target_pattern_id")
            elif op.target_pattern_id not in all_pattern_ids:
                hard.append(
                    f"{prefix}: {op.action} target_pattern_id={op.target_pattern_id} "
                    f"不在模式集合中"
                )

        # HARD: CREATE 的 new_pattern 不能为空
        if op.action == "CREATE":
            if op.new_pattern is None:
                hard.append(f"{prefix}: CREATE 操作缺少 new_pattern")

        # SOFT: reason 非空
        if not op.reason.strip():
            soft.append(f"{prefix}: reason 为空")

    return hard, soft
