"""进化有效性看板纯聚合函数回归测试（evolution_metrics.aggregate_evolution）。

覆盖：决策/弃权拆分、Wilson 下界与「跑赢随机」判定、样本不足短路、
代际对比、按发现方法拆分、按天趋势分桶、确定性。全程无 LLM、无 DB。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from binance_predict.services.evolution_metrics import (
    MIN_JUDGE_SAMPLES,
    RANDOM_BASELINE,
    UNMATCHED_METHOD,
    aggregate_evolution,
)

_BASE = datetime(2026, 7, 1, tzinfo=timezone.utc)


def _rec(
    idx: int,
    direction: str,
    correct: bool | None,
    pattern_id: int | None = None,
    day_offset: int = 0,
) -> dict:
    return {
        "prediction_time": _BASE + timedelta(days=day_offset, minutes=5 * idx),
        "predicted_direction": direction,
        "is_correct": correct,
        "matched_pattern_id": pattern_id,
    }


def test_empty_records() -> None:
    rep = aggregate_evolution([], {}, window_days=30)
    assert rep["total_validated"] == 0
    assert rep["decisive_count"] == 0
    assert rep["overall"]["verdict"] == "INSUFFICIENT_SAMPLES"
    assert rep["trend_daily"] == []
    assert rep["by_discovery_method"] == {}


def test_no_trade_excluded_from_decisive() -> None:
    records = [_rec(i, "NO_TRADE", None) for i in range(5)]
    records += [_rec(i + 10, "UP", True) for i in range(3)]
    rep = aggregate_evolution(records, {}, window_days=30)
    assert rep["total_validated"] == 8
    assert rep["decisive_count"] == 3
    assert rep["no_trade_count"] == 5


def test_beats_random() -> None:
    # 40 决策样本，30 命中 → 胜率 0.75，Wilson 下界应 > 0.5
    records = [_rec(i, "UP", i < 30) for i in range(40)]
    rep = aggregate_evolution(records, {}, window_days=30)
    o = rep["overall"]
    assert o["win_rate"] == 0.75
    assert o["ci_lower"] > RANDOM_BASELINE
    assert o["beats_random"] is True
    assert o["verdict"] == "BEATS_RANDOM"
    assert o["excess_over_random"] == round(0.75 - RANDOM_BASELINE, 4)


def test_insufficient_samples_short_circuit() -> None:
    # 全对但样本 < 判定阈值 → 不下跑赢随机结论
    n = MIN_JUDGE_SAMPLES - 1
    records = [_rec(i, "DOWN", True) for i in range(n)]
    rep = aggregate_evolution(records, {}, window_days=30)
    assert rep["overall"]["verdict"] == "INSUFFICIENT_SAMPLES"


def test_inconclusive_when_ci_covers_half() -> None:
    # 胜率恰 0.5，下界必 < 0.5 → INCONCLUSIVE
    records = [_rec(i, "UP", i % 2 == 0) for i in range(MIN_JUDGE_SAMPLES + 10)]
    rep = aggregate_evolution(records, {}, window_days=30)
    assert rep["overall"]["beats_random"] is False
    assert rep["overall"]["verdict"] == "INCONCLUSIVE"


def test_by_discovery_method_split_and_unmatched() -> None:
    method_map = {1: "LLM_DEEP", 2: "PY_CLUSTER"}
    records = [
        _rec(0, "UP", True, pattern_id=1),
        _rec(1, "UP", False, pattern_id=1),
        _rec(2, "DOWN", True, pattern_id=2),
        _rec(3, "UP", True, pattern_id=None),   # 无匹配 → UNMATCHED
        _rec(4, "UP", True, pattern_id=99),     # 未知 id → UNMATCHED
    ]
    rep = aggregate_evolution(records, method_map, window_days=30)
    bm = rep["by_discovery_method"]
    assert bm["LLM_DEEP"]["sample_count"] == 2
    assert bm["LLM_DEEP"]["correct"] == 1
    assert bm["PY_CLUSTER"]["sample_count"] == 1
    assert bm[UNMATCHED_METHOD]["sample_count"] == 2


def test_generations_comparable_and_delta() -> None:
    # 前半程 20 条胜率 0.5，后半程 20 条胜率 1.0 → 近半程更好
    older = [_rec(i, "UP", i % 2 == 0) for i in range(20)]
    newer = [_rec(i + 100, "UP", True) for i in range(20)]
    rep = aggregate_evolution(older + newer, {}, window_days=30)
    gen = rep["generations"]
    assert gen["comparable"] is True
    assert gen["older_half"]["win_rate"] == 0.5
    assert gen["newer_half"]["win_rate"] == 1.0
    assert gen["win_rate_delta"] == 0.5


def test_generations_not_comparable_when_small() -> None:
    records = [_rec(i, "UP", True) for i in range(10)]
    rep = aggregate_evolution(records, {}, window_days=30)
    assert rep["generations"]["comparable"] is False
    assert rep["generations"]["significant_improvement"] is False


def test_trend_daily_bucketing() -> None:
    records = [
        _rec(0, "UP", True, day_offset=0),
        _rec(1, "UP", False, day_offset=0),
        _rec(0, "DOWN", True, day_offset=1),
    ]
    rep = aggregate_evolution(records, {}, window_days=30)
    trend = rep["trend_daily"]
    assert len(trend) == 2
    assert trend[0]["date"] == "2026-07-01"
    assert trend[0]["sample_count"] == 2
    assert trend[1]["date"] == "2026-07-02"
    assert trend[1]["sample_count"] == 1


def test_determinism() -> None:
    records = [_rec(i, "UP", i < 15) for i in range(30)]
    r1 = aggregate_evolution(records, {}, window_days=30)
    r2 = aggregate_evolution(records, {}, window_days=30)
    assert r1 == r2
