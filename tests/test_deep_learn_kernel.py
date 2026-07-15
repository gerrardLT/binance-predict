"""Deep Learn 确定性内核回归测试（curve_features + backtest）。

覆盖 P0-2 特征提取 / KMeans 聚类、P0-3 Wilson 下界 / time_split / holdout 回测、
P1-3 余弦去重、P2-2 snapshot_token 一致性。全程无 LLM、无 DB，纯确定性断言。
"""
from __future__ import annotations

import numpy as np
import pytest

from binance_predict.services.backtest import (
    HOLDOUT_MATCH_THRESHOLD,
    evaluate_on_holdout,
    snapshot_token,
    time_split,
    wilson_lower_bound,
)
from binance_predict.services.curve_features import (
    FEATURE_DIM,
    cluster_windows,
    cosine_sim,
    extract_features,
)


def _ramp_curve(start: float, end: float, n: int = 6) -> list[dict]:
    """构造一条从 start 线性升/降到 end 的 [{t, v}] 曲线。"""
    vals = np.linspace(start, end, n)
    return [{"t": i, "v": float(v)} for i, v in enumerate(vals)]


# ------------------------- curve_features -------------------------

def test_extract_features_dim_and_determinism() -> None:
    up = _ramp_curve(40.0, 60.0)
    down = _ramp_curve(60.0, 40.0)
    f1 = extract_features(up, down)
    f2 = extract_features(up, down)
    assert f1.shape == (FEATURE_DIM,)
    assert np.array_equal(f1, f2)  # 同输入必得同输出


def test_extract_features_empty_is_zero_vector() -> None:
    f = extract_features([], [])
    assert f.shape == (FEATURE_DIM,)
    assert np.all(f == 0.0)


def test_extract_features_captures_net_change() -> None:
    up = _ramp_curve(40.0, 60.0)
    down = _ramp_curve(50.0, 50.0)
    f = extract_features(up, down)
    # up 序列第 3 维为净变化(last-first)=+20
    assert f[2] == 60.0 - 40.0


def test_cosine_sim_identical_and_zero() -> None:
    v = extract_features(_ramp_curve(40.0, 60.0), _ramp_curve(60.0, 40.0))
    assert cosine_sim(v, v) == pytest.approx(1.0)
    assert cosine_sim(v, np.zeros(FEATURE_DIM)) == 0.0


def test_cosine_sim_opposite_shapes_lower_than_identical() -> None:
    up_a = extract_features(_ramp_curve(40.0, 60.0), _ramp_curve(60.0, 40.0))
    up_b = extract_features(_ramp_curve(60.0, 40.0), _ramp_curve(40.0, 60.0))
    assert cosine_sim(up_a, up_b) < cosine_sim(up_a, up_a)


def test_cluster_windows_trivial_cases() -> None:
    # 空矩阵 / 单行 → 全归簇 0
    assert cluster_windows(np.zeros((0, FEATURE_DIM)), 3).tolist() == []
    assert cluster_windows(np.zeros((1, FEATURE_DIM)), 3).tolist() == [0]


def test_cluster_windows_deterministic_and_separates_groups() -> None:
    rng = np.random.default_rng(0)
    group_a = rng.normal(0.0, 0.01, size=(5, FEATURE_DIM)) + 0.0
    group_b = rng.normal(0.0, 0.01, size=(5, FEATURE_DIM)) + 10.0
    X = np.vstack([group_a, group_b])
    labels1 = cluster_windows(X, 2)
    labels2 = cluster_windows(X, 2)
    assert np.array_equal(labels1, labels2)  # random_state 固定 → 可复现
    # 两组各自内部标签一致（被分到不同簇）
    assert len(set(labels1[:5].tolist())) == 1
    assert len(set(labels1[5:].tolist())) == 1
    assert labels1[0] != labels1[5]


# ------------------------- backtest: wilson -------------------------

def test_wilson_lower_bound_zero_total() -> None:
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_lower_bound_discounts_small_samples() -> None:
    # 同为 100% 胜率，样本越少下界越低
    lb_small = wilson_lower_bound(2, 2)
    lb_large = wilson_lower_bound(50, 50)
    assert 0.0 < lb_small < lb_large < 1.0


def test_wilson_lower_bound_clamps_correct() -> None:
    # correct > total 被夹到 total
    assert wilson_lower_bound(10, 5) == wilson_lower_bound(5, 5)


# ------------------------- backtest: snapshot_token -------------------------

def test_snapshot_token_deterministic_and_order_independent() -> None:
    a = snapshot_token([3, 1, 2])
    b = snapshot_token([2, 3, 1])
    assert a == b  # 顺序无关
    assert len(a) == 16


def test_snapshot_token_dedup_and_differs_on_change() -> None:
    assert snapshot_token([1, 1, 2]) == snapshot_token([1, 2])
    assert snapshot_token([1, 2]) != snapshot_token([1, 2, 3])


# ------------------------- backtest: time_split -------------------------

def test_time_split_too_few_windows() -> None:
    train, holdout = time_split([{"start_time": 1}], 0.3)
    assert len(train) == 1 and holdout == []


def test_time_split_latest_goes_to_holdout() -> None:
    windows = [{"start_time": t, "id": t} for t in [10, 40, 20, 30]]
    train, holdout = time_split(windows, 0.5)
    assert len(train) == 2 and len(holdout) == 2
    # holdout 为时间最新的两条
    assert {w["id"] for w in holdout} == {30, 40}
    assert {w["id"] for w in train} == {10, 20}


def test_time_split_ratio_clamped() -> None:
    windows = [{"start_time": t} for t in range(5)]
    train, holdout = time_split(windows, 5.0)  # ratio 被夹到 0.9
    assert len(holdout) == 4 and len(train) == 1


# ------------------------- backtest: evaluate_on_holdout -------------------------

def test_evaluate_on_holdout_matches_same_shape_and_direction() -> None:
    pattern = extract_features(_ramp_curve(40.0, 60.0), _ramp_curve(60.0, 40.0))
    holdout = [
        {
            "curve_up_pct": _ramp_curve(40.0, 60.0),
            "curve_down_pct": _ramp_curve(60.0, 40.0),
            "outcome": "UP",
        }
    ]
    res = evaluate_on_holdout(pattern, "UP", holdout)
    assert res["sample_count"] == 1
    assert res["win_rate"] == 1.0
    assert 0.0 <= res["ci_lower"] <= 1.0


def test_evaluate_on_holdout_no_match_when_dissimilar() -> None:
    pattern = extract_features(_ramp_curve(40.0, 60.0), _ramp_curve(60.0, 40.0))
    # 反向形态，余弦相似度低于阈值 → 不触发
    holdout = [
        {
            "curve_up_pct": _ramp_curve(90.0, 10.0),
            "curve_down_pct": _ramp_curve(10.0, 90.0),
            "outcome": "UP",
        }
    ]
    res = evaluate_on_holdout(pattern, "UP", holdout, match_threshold=HOLDOUT_MATCH_THRESHOLD)
    assert res["sample_count"] == 0
    assert res["win_rate"] == 0.0
    assert res["ci_lower"] == 0.0


def test_evaluate_on_holdout_wrong_direction_counts_but_misses() -> None:
    pattern = extract_features(_ramp_curve(40.0, 60.0), _ramp_curve(60.0, 40.0))
    holdout = [
        {
            "curve_up_pct": _ramp_curve(40.0, 60.0),
            "curve_down_pct": _ramp_curve(60.0, 40.0),
            "outcome": "DOWN",  # 触发但方向不符 → 计入分母不计命中
        }
    ]
    res = evaluate_on_holdout(pattern, "UP", holdout)
    assert res["sample_count"] == 1
    assert res["win_rate"] == 0.0
