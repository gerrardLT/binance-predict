"""进化有效性看板指标（Item 1）。

把「Agent 是否真的在进化」量化为可观测、可证伪的数字，用于区分
「持续变化（change）」与「持续变好（improve）」——后者才是自进化的本质。
全程只读、无 LLM，同一输入必得同一输出。

核心口径：
- 仅统计已验证预测（is_correct IS NOT NULL）。
- 「决策样本」(decisive) = predicted_direction ∈ {UP, DOWN}，即真正下注的方向判断；
  NO_TRADE 属于弃权，单独计数、不计入胜率。
- 随机基线 = 0.5（方向二选一）。判定「跑赢随机」以 Wilson 95% 置信下界 > 0.5 为准，
  样本越少下界越低，天然抑制「运气当本事」。

四类指标：
1. overall     —— 总体决策胜率、Wilson 下界、超额、是否跑赢随机（acid test）。
2. trend_daily —— 按天分桶的样本外胜率序列（看是否随时间上行）。
3. generations —— 前半程 vs 近半程胜率对比（change ≠ improvement 的判据）。
4. by_method   —— 按发现方法（LLM_DEEP / PY_CLUSTER / LEGACY / UNMATCHED）拆分胜率，
                  回答「哪条轨道真的产出 alpha」。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AgentPrediction, PatternMemory
from .backtest import wilson_lower_bound

# 方向二选一的随机基线
RANDOM_BASELINE = 0.5
# 低于此决策样本数时不下「是否跑赢随机」结论（样本外统计不可信）
MIN_JUDGE_SAMPLES = 30
# generations 两半各自需达到的最小样本数才做「是否改善」判断
MIN_GENERATION_SAMPLES = 15
# UNMATCHED 桶标识：决策预测无匹配模式（冷启动/无命中）时归入
UNMATCHED_METHOD = "UNMATCHED"


def _bucket_stats(correct: int, total: int) -> dict:
    """把 (命中数, 样本数) 折算为一组标准统计（胜率 / Wilson 下界 / 超额 / 是否跑赢随机）。"""
    win_rate = (correct / total) if total > 0 else 0.0
    ci_lower = wilson_lower_bound(correct, total)
    return {
        "sample_count": total,
        "correct": correct,
        "win_rate": round(win_rate, 4),
        "ci_lower": round(ci_lower, 4),
        "excess_over_random": round(win_rate - RANDOM_BASELINE, 4),
        "beats_random": ci_lower > RANDOM_BASELINE,
    }


def _overall_verdict(stats: dict) -> str:
    """总体结论：样本不足 / 已跑赢随机 / 尚未显著。"""
    if stats["sample_count"] < MIN_JUDGE_SAMPLES:
        return "INSUFFICIENT_SAMPLES"
    if stats["beats_random"]:
        return "BEATS_RANDOM"
    return "INCONCLUSIVE"


def _build_summary(overall: dict, verdict: str, generations: dict) -> str:
    """自然语言诊断摘要，供人和 LLM 直接读取判断闭环是否真的在进化。"""
    n = overall["sample_count"]
    if verdict == "INSUFFICIENT_SAMPLES":
        base = (
            f"决策样本仅 {n} 条（<{MIN_JUDGE_SAMPLES}），暂无法判定是否跑赢随机，"
            f"继续积累已验证预测。"
        )
    elif verdict == "BEATS_RANDOM":
        base = (
            f"总体已显著跑赢随机：决策胜率 {overall['win_rate']:.1%}，"
            f"Wilson 95% 下界 {overall['ci_lower']:.1%} > 50%（{n} 样本）。"
        )
    else:
        base = (
            f"总体尚未显著跑赢随机：决策胜率 {overall['win_rate']:.1%}，"
            f"Wilson 95% 下界 {overall['ci_lower']:.1%} ≤ 50%（{n} 样本）——"
            f"当前更像在噪声内波动。"
        )

    gen = generations
    if gen.get("comparable"):
        older = gen["older_half"]["win_rate"]
        newer = gen["newer_half"]["win_rate"]
        delta = gen["win_rate_delta"]
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        gen_txt = (
            f" 代际对比：前半程 {older:.1%} {arrow} 近半程 {newer:.1%}"
            f"（Δ={delta:+.1%}）。"
        )
        if gen.get("significant_improvement"):
            gen_txt += "近半程保守下界已超前半程点估计，是可信的改善信号。"
        elif delta > 0:
            gen_txt += "有改善迹象但未达显著，可能仍是波动。"
        else:
            gen_txt += "未见改善——警惕「只在变化、并未变好」。"
    else:
        gen_txt = " 代际对比：两半程样本不足，暂不下改善结论。"

    return base + gen_txt


def aggregate_evolution(
    records: list[dict],
    method_map: dict[int, str],
    window_days: int,
) -> dict:
    """纯聚合函数（无 I/O，便于单测）。

    Args:
        records: 已验证预测记录，按 prediction_time 升序；每条含
            {prediction_time: datetime, predicted_direction: str,
             is_correct: bool, matched_pattern_id: int|None}
        method_map: pattern_id -> discovery_method 映射
        window_days: 统计时间窗（天），仅用于回填到返回结构

    Returns:
        看板报告 dict（见模块 docstring）。
    """
    total_validated = len(records)
    decisive = [r for r in records if r["predicted_direction"] in ("UP", "DOWN")]
    no_trade_count = total_validated - len(decisive)

    # --- 1. overall ---
    dec_correct = sum(1 for r in decisive if r["is_correct"])
    overall = _bucket_stats(dec_correct, len(decisive))
    verdict = _overall_verdict(overall)
    overall["verdict"] = verdict

    # --- 2. trend_daily（按 UTC 日期分桶）---
    daily: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # date -> [correct, total]
    for r in decisive:
        day = r["prediction_time"].astimezone(timezone.utc).date().isoformat()
        daily[day][1] += 1
        if r["is_correct"]:
            daily[day][0] += 1
    trend_daily = [
        {"date": day, **_bucket_stats(c, t)}
        for day, (c, t) in sorted(daily.items())
    ]

    # --- 3. generations（前半程 vs 近半程，按样本数对半切）---
    n = len(decisive)
    half = n // 2
    older, newer = decisive[:half], decisive[half:]
    older_stats = _bucket_stats(sum(1 for r in older if r["is_correct"]), len(older))
    newer_stats = _bucket_stats(sum(1 for r in newer if r["is_correct"]), len(newer))
    comparable = (
        len(older) >= MIN_GENERATION_SAMPLES and len(newer) >= MIN_GENERATION_SAMPLES
    )
    generations = {
        "comparable": comparable,
        "older_half": older_stats,
        "newer_half": newer_stats,
        "win_rate_delta": round(newer_stats["win_rate"] - older_stats["win_rate"], 4),
        # 严格改善判据：近半程保守下界 > 前半程点估计
        "significant_improvement": (
            comparable and newer_stats["ci_lower"] > older_stats["win_rate"]
        ),
    }

    # --- 4. by_discovery_method ---
    method_buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in decisive:
        pid = r["matched_pattern_id"]
        method = method_map.get(pid, UNMATCHED_METHOD) if pid is not None else UNMATCHED_METHOD
        method_buckets[method][1] += 1
        if r["is_correct"]:
            method_buckets[method][0] += 1
    by_discovery_method = {
        method: _bucket_stats(c, t)
        for method, (c, t) in sorted(method_buckets.items())
    }

    summary = _build_summary(overall, verdict, generations)

    return {
        "window_days": window_days,
        "total_validated": total_validated,
        "decisive_count": len(decisive),
        "no_trade_count": no_trade_count,
        "random_baseline": RANDOM_BASELINE,
        "overall": overall,
        "trend_daily": trend_daily,
        "generations": generations,
        "by_discovery_method": by_discovery_method,
        "summary": summary,
    }


async def build_evolution_report(
    db: AsyncSession,
    days: int = 30,
    max_samples: int = 20000,
) -> dict:
    """读取已验证预测并计算进化有效性看板报告。

    Args:
        db: 异步会话
        days: 统计时间窗（天），仅统计该窗内 prediction_time 的已验证预测
        max_samples: 安全上限，防止极端数据量下全表加载

    Returns:
        看板报告 dict（含 generated_at）。
    """
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)

    stmt = (
        select(
            AgentPrediction.prediction_time,
            AgentPrediction.predicted_direction,
            AgentPrediction.is_correct,
            AgentPrediction.matched_pattern_id,
        )
        .where(
            AgentPrediction.is_correct.is_not(None),
            AgentPrediction.prediction_time >= since,
        )
        .order_by(AgentPrediction.prediction_time.asc())
        .limit(max_samples)
    )
    rows = (await db.execute(stmt)).all()
    records = [
        {
            "prediction_time": row[0],
            "predicted_direction": row[1],
            "is_correct": row[2],
            "matched_pattern_id": row[3],
        }
        for row in rows
    ]

    # pattern_id -> discovery_method 映射（模式数量有限，一次性载入）
    method_rows = (
        await db.execute(select(PatternMemory.id, PatternMemory.discovery_method))
    ).all()
    method_map = {pid: method for pid, method in method_rows}

    report = aggregate_evolution(records, method_map, window_days=days)
    report["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
    return report
