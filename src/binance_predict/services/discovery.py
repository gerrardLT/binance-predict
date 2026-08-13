"""科学发现系统 —— 假设初筛编排内核（宪法 Q6 操作化）。

把 LLM 提出的谓词假设在验证集上做统计审判的固定流水线。全程无 LLM、无 DB、
纯函数：同一输入必得同一输出。

流水线（design.md Q6「初筛操作化」，顺序不可换）：
1. 谓词 DSL 白名单校验（Q5）→ 失败直接 REJECT
2. 逐假设执行谓词 → 命中窗口集合（命中数 <5 直接 REJECT，样本不足不做统计）
3. 局部基准合并对照池 → lift_test 得 lift / CI / p 值
4. 全部假设的 p 值过 BH-FDR（q=0.1）得到通过标记
5. 合成裁决：ACTIVE = 双轨 ACTIVE 且 FDR 通过；双轨 ACTIVE 但 FDR 未通过 →
   降级 OBSERVE（统计功效不足而非模式无效）；OBSERVE = 双轨 OBSERVE
   （不要求 FDR，纸面跟踪本身即攒样本过程）；其余 REJECT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from binance_predict.services.predicates import evaluate_predicate, validate_predicate
from binance_predict.services.symbolizer import WindowView
from binance_predict.services.verification import (
    FDR_Q,
    LiftResult,
    bh_fdr,
    classify_candidate,
    lift_test,
    pooled_local_baseline,
)

# 初筛最小命中数（宪法参数表：低于此值直接 REJECT，不进 lift 检验）
MIN_SCREEN_HITS: int = 5

VERDICT_ACTIVE = "ACTIVE"
VERDICT_OBSERVE = "OBSERVE"
VERDICT_REJECT = "REJECT"

# lift 检验的合法目标结果（NOISE 不构成可检验假设）
_VALID_TARGETS = ("UP", "DOWN")


@dataclass(frozen=True)
class ScreenedHypothesis:
    """一条假设的初筛结果（含全部审判证据，供落库与反馈）。"""

    index: int  # 在输入 hypotheses 列表中的位置
    predicate: dict | None  # 通过 Q5 校验的谓词 DSL；校验失败为 None
    target_outcome: str
    hit_start_times: list[int] = field(default_factory=list)
    lift_result: LiftResult | None = None  # REJECT（未进统计）时为 None
    fdr_passed: bool = False
    verdict: str = VERDICT_REJECT
    reject_reason: str | None = None  # REJECT 原因（非静默，反馈 LLM 用）


def _compose_verdict(track: str, fdr_passed: bool) -> str:
    """双轨判定 × FDR 标记 → 最终裁决（Q6 操作化第 4 步）。

    ACTIVE 必须同时满足双轨 ACTIVE 与 FDR 通过；双轨 ACTIVE 但 FDR 未通过
    降级 OBSERVE（功效不足攒样本，而非判模式无效）。
    """
    if track == "ACTIVE" and fdr_passed:
        return VERDICT_ACTIVE
    if track in ("ACTIVE", "OBSERVE"):
        return VERDICT_OBSERVE
    return VERDICT_REJECT


def screen_hypotheses(
    hypotheses: list[dict],
    views: list[WindowView],
    min_hits: int = MIN_SCREEN_HITS,
    fdr_q: float = FDR_Q,
) -> list[ScreenedHypothesis]:
    """假设初筛流水线（Q6 操作化，顺序不可换）。

    Args:
        hypotheses: LLM 输出的假设列表，每条含：
            - predicate: dict，谓词 DSL（Q5）
            - target_outcome: "UP" | "DOWN"，假设预测的目标结果
            其余字段（name/rationale 等）原样忽略，不影响审判。
        views: 验证集的符号化窗口视图（须带 start_time 与 outcome）
        min_hits: 初筛最小命中数（宪法默认 5）
        fdr_q: BH-FDR 控制水平（宪法默认 0.1）

    Returns:
        与 hypotheses 等长、同序的 ScreenedHypothesis 列表
    """
    # 局部基准对照池原料（与命中判定同一验证集；命中窗口自身永不被纳入对照池）
    pool_windows = [
        {"start_time": v.start_time, "outcome": v.outcome or ""} for v in views
    ]

    results: list[ScreenedHypothesis] = []
    stat_slots: list[int] = []  # 进入统计检验的假设在 results 中的下标
    p_values: list[float] = []

    for i, hyp in enumerate(hypotheses):
        predicate = hyp.get("predicate")
        target = str(hyp.get("target_outcome") or "").upper()

        # 第 1 步：DSL 白名单校验（Q5）
        try:
            validate_predicate(predicate)
        except (ValueError, TypeError, AttributeError) as exc:
            results.append(ScreenedHypothesis(
                index=i, predicate=None, target_outcome=target,
                reject_reason=f"predicate_invalid: {exc}",
            ))
            continue

        # 目标结果非法：不构成可检验假设
        if target not in _VALID_TARGETS:
            results.append(ScreenedHypothesis(
                index=i, predicate=predicate, target_outcome=target,
                reject_reason=f"invalid_target: {target!r}（仅 UP | DOWN 可检验）",
            ))
            continue

        # 第 2 步：谓词执行 → 命中集合；命中不足直接 REJECT（样本不足不做统计）
        hits = [v.start_time for v in views if evaluate_predicate(predicate, v)]
        if len(hits) < min_hits:
            results.append(ScreenedHypothesis(
                index=i, predicate=predicate, target_outcome=target,
                hit_start_times=hits,
                reject_reason=f"insufficient_hits: {len(hits)} < {min_hits}",
            ))
            continue

        # 第 3 步：局部基准 lift 检验
        hit_events = sum(
            1 for v in views
            if v.start_time in set(hits) and (v.outcome or "").upper() == target
        )
        base_events, base_total = pooled_local_baseline(hits, pool_windows, target)
        lr = lift_test(hit_events, len(hits), base_events, base_total)

        results.append(ScreenedHypothesis(
            index=i, predicate=predicate, target_outcome=target,
            hit_start_times=hits, lift_result=lr,
        ))
        stat_slots.append(len(results) - 1)
        p_values.append(lr.p_value)

    # 第 4 步：BH-FDR（仅对进入统计的假设）
    fdr_flags = bh_fdr(p_values, q=fdr_q)

    # 第 5 步：合成裁决
    final: list[ScreenedHypothesis] = []
    for pos, r in enumerate(results):
        if r.lift_result is None:
            final.append(r)  # 已 REJECT，保持原因
            continue
        slot = stat_slots.index(pos)
        track = classify_candidate(r.lift_result)
        verdict = _compose_verdict(track, fdr_flags[slot])
        final.append(ScreenedHypothesis(
            index=r.index,
            predicate=r.predicate,
            target_outcome=r.target_outcome,
            hit_start_times=r.hit_start_times,
            lift_result=r.lift_result,
            fdr_passed=fdr_flags[slot],
            verdict=verdict,
            reject_reason=None if verdict != VERDICT_REJECT else (
                f"track={track}: lift={r.lift_result.lift:.3f} 未达观察仓下限"
            ),
        ))
    return final
