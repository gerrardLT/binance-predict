"""方案对比摘要 _summarize_discovery_group 的回归测试（M2-api / P0-3 闸门聚合）。"""
from __future__ import annotations

from binance_predict.config.settings import settings
from binance_predict.main import _summarize_discovery_group


def _disc(direction: str, ci_lower: float, samples: int, win_rate: float = 0.7, conf: float = 0.6) -> dict:
    return {
        "predicted_direction": direction,
        "holdout_win_rate": win_rate,
        "holdout_ci_lower": ci_lower,
        "holdout_sample_count": samples,
        "confidence_score": conf,
    }


def test_summarize_empty_result() -> None:
    s = _summarize_discovery_group({"method": "PY_CLUSTER", "discoveries": []})
    assert s["discovery_count"] == 0
    assert s["passed_gate_ratio"] == 0.0
    assert s["avg_holdout_win_rate"] == 0.0
    assert s["method"] == "PY_CLUSTER"


def test_summarize_counts_direction_and_samples() -> None:
    min_s = settings.agent_deep_learn_min_holdout_samples
    result = {
        "method": "LLM_DEEP",
        "snapshot_token": "abc123",
        "train_count": 30,
        "holdout_count": 12,
        "discoveries": [
            _disc("UP", ci_lower=0.6, samples=min_s + 5),      # 过闸门
            _disc("DOWN", ci_lower=0.4, samples=min_s + 5),     # ci_lower<=0.5 不过
            _disc("UP", ci_lower=0.7, samples=max(0, min_s - 1)),  # 样本不足不过
        ],
    }
    s = _summarize_discovery_group(result)
    assert s["discovery_count"] == 3
    assert s["direction_up"] == 2
    assert s["direction_down"] == 1
    assert s["passed_gate_count"] == 1
    assert s["passed_gate_ratio"] == round(1 / 3, 4)
    assert s["snapshot_token"] == "abc123"
    assert s["train_count"] == 30 and s["holdout_count"] == 12
    assert s["total_holdout_samples"] == (min_s + 5) + (min_s + 5) + max(0, min_s - 1)


def test_summarize_gate_needs_both_ci_and_samples() -> None:
    min_s = settings.agent_deep_learn_min_holdout_samples
    # ci_lower 恰好 0.5（非严格大于）不过闸门
    result = {
        "method": "PY_CLUSTER",
        "discoveries": [_disc("UP", ci_lower=0.5, samples=min_s + 10)],
    }
    assert _summarize_discovery_group(result)["passed_gate_count"] == 0
