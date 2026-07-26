"""feature_bench 离线评估台架的确定性回归测试。

守护「改-验-评闭环」的地基：台架本身必须确定性、指标口径正确、判别度量可靠。
全程无 LLM、无 DB、纯断言，风格对齐 test_deep_learn_kernel.py。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# scripts/ 非包，注入路径后导入台架模块
_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import feature_bench as fb  # noqa: E402


def _ramp(start: float, end: float, n: int = 12) -> list[dict]:
    vals = np.linspace(start, end, n)
    return [{"t": i, "v": float(v)} for i, v in enumerate(vals)]


def _win(wid: int, up, down, outcome: str, start_time: int | None = None) -> dict:
    return {
        "id": wid,
        "start_time": start_time if start_time is not None else wid * 300_000,
        "curve_up_pct": up,
        "curve_down_pct": down,
        "outcome": outcome,
    }


# ------------------------- _auc -------------------------

def test_auc_perfect_and_reversed_and_tied() -> None:
    assert fb._auc([3.0, 4.0, 5.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert fb._auc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)
    # 完全并列 → 0.5
    assert fb._auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_auc_empty_returns_none() -> None:
    assert fb._auc([], [1.0]) is None
    assert fb._auc([1.0], []) is None


# ------------------------- _bucket_stats -------------------------

def test_bucket_stats_matches_wilson() -> None:
    from binance_predict.services.backtest import wilson_lower_bound

    s = fb._bucket_stats(30, 40)
    assert s["sample_count"] == 40
    assert s["correct"] == 30
    assert s["win_rate"] == pytest.approx(0.75)
    assert s["ci_lower"] == pytest.approx(round(wilson_lower_bound(30, 40), 4))
    assert s["beats_random"] is True


def test_bucket_stats_zero_total() -> None:
    s = fb._bucket_stats(0, 0)
    assert s["sample_count"] == 0
    assert s["win_rate"] == 0.0
    assert s["ci_lower"] == 0.0
    assert s["beats_random"] is False


# ------------------------- run_scorecard: 确定性 -------------------------

def test_run_scorecard_deterministic() -> None:
    windows = fb.synthetic_windows(n_per_group=15)
    c1 = fb.run_scorecard(windows, holdout_ratio=0.3, n_clusters=6)
    c2 = fb.run_scorecard(windows, holdout_ratio=0.3, n_clusters=6)
    assert c1 == c2  # 同输入必得同输出（含指纹、诊断、逐簇明细）


def test_run_scorecard_empty_windows_no_crash() -> None:
    card = fb.run_scorecard([], holdout_ratio=0.3, n_clusters=6)
    assert card["data_health"]["total_windows"] == 0
    assert card["primary_metric"]["sample_count"] == 0
    assert card["clusters"] == []


# ------------------------- 指纹：数据漂移检测 -------------------------

def test_fingerprint_token_changes_on_data_change() -> None:
    windows = fb.synthetic_windows(n_per_group=10)
    base = fb.run_scorecard(windows, n_clusters=6)["fingerprint"]["snapshot_token"]
    # 同一批数据 → token 稳定
    again = fb.run_scorecard(windows, n_clusters=6)["fingerprint"]["snapshot_token"]
    assert base == again
    # 增加一个窗口 → token 改变
    extra = windows + [_win(999999, _ramp(45, 60), _ramp(55, 40), "UP")]
    changed = fb.run_scorecard(extra, n_clusters=6)["fingerprint"]["snapshot_token"]
    assert changed != base


# ------------------------- 汇总一致性 -------------------------

def test_primary_metric_aggregates_cluster_holdout() -> None:
    windows = fb.synthetic_windows(n_per_group=20)
    card = fb.run_scorecard(windows, holdout_ratio=0.3, n_clusters=8)
    pm = card["primary_metric"]
    assert pm["sample_count"] == sum(c["holdout_matched"] for c in card["clusters"])
    assert pm["correct"] == sum(c["holdout_correct"] for c in card["clusters"])


# ------------------------- 判别分离度度量有效性（守护优化点 A 的靶标）-------------------------

def test_discrimination_separates_easy_from_hard() -> None:
    """可分数据集的分离度 gap 应显著高于「形态相反但绝对水平接近」的难数据集。

    难数据集刻意让 UP/DOWN 形态相反却都贴近 50% 水平——这正是原始 cosine 被
    绝对水平绑架的场景：即使形态相反，异类对的 cosine 仍逆近 1.0，与同类对
    几乎无法拉开（gap≈0）。gap 直接反映「幅度可分性」，是水平绑架的直接靶标；
    度量必须能把两者区分开，否则后续 A 轮无法客观验证「几何统一」是否真的提升判别力。
    """
    # 可分：UP 明显上行、DOWN 明显下行，形态与水平差异都大
    easy = []
    for k in range(8):
        easy.append(_win(k * 2 + 1, _ramp(20, 80), _ramp(80, 20), "UP"))
        easy.append(_win(k * 2 + 2, _ramp(80, 20), _ramp(20, 80), "DOWN"))
    d_easy = fb._discrimination(easy, fb.default_feature_fn, fb.default_sim_fn)

    # 难：形态相反但都在 ~50 附近微幅波动（绝对水平绑架 cosine）
    hard = []
    for k in range(8):
        hard.append(_win(k * 2 + 1, _ramp(49.5, 50.5), _ramp(50.5, 49.5), "UP"))
        hard.append(_win(k * 2 + 2, _ramp(50.5, 49.5), _ramp(49.5, 50.5), "DOWN"))
    d_hard = fb._discrimination(hard, fb.default_feature_fn, fb.default_sim_fn)
    assert d_easy["comparable"] and d_hard["comparable"]
    # 可分集的幅度分离显著大于难集
    assert d_easy["gap"] > d_hard["gap"]
    # 难集几乎不可分：gap≈0，且异类对仍高度相似（水平绑架的铁证）
    assert d_hard["gap"] == pytest.approx(0.0, abs=0.05)
    assert d_hard["mean_diff_sim"] > 0.9


def test_discrimination_insufficient_windows() -> None:
    two = [_win(1, _ramp(40, 60), _ramp(60, 40), "UP")]
    res = fb._discrimination(two, fb.default_feature_fn, fb.default_sim_fn)
    assert res["comparable"] is False


# ------------------------- 冗余度度量 -------------------------

def test_redundancy_detects_duplicate_dims() -> None:
    # 构造：后半维度与前半完全重复（高冗余）→ mean_abs_corr 高、有效秩远小于维度数
    rng = np.random.default_rng(0)
    half = rng.normal(0, 1, size=(30, 4))
    dup = np.hstack([half, half])  # 8 维但只有 4 维独立信息
    red = fb._redundancy(dup)
    assert red["comparable"] is True
    assert red["n_dims"] == 8
    assert red["effective_rank"] < 6  # 有效秩显著低于 8


def test_redundancy_insufficient() -> None:
    assert fb._redundancy(np.zeros((0, 24)))["comparable"] is False
    assert fb._redundancy(np.zeros((5, 1)))["comparable"] is False


# ------------------------- 接缝可替换（守护 B/C 轮换特征/几何）-------------------------

def test_scorecard_accepts_custom_seams_with_different_dim() -> None:
    """自定义 4 维 feature_fn + 欧氏相似度 sim_fn，台架应正常产出（维度无关）。"""

    def feat4(curve_up, curve_down):
        u = fb.extract_features(curve_up, curve_down)
        # 取前 4 维构造一个不同维度的特征，验证台架不写死 24 维
        return np.asarray(u[:4], dtype=float)

    def neg_l2(a, b, std_ctx=None):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return 1.0 / (1.0 + float(np.linalg.norm(a - b)))

    windows = fb.synthetic_windows(n_per_group=12)
    card = fb.run_scorecard(
        windows, feature_fn=feat4, sim_fn=neg_l2, n_clusters=5, match_threshold=0.5
    )
    assert card["diagnostics"]["redundancy"]["n_dims"] == 4
    # 两次运行仍确定性
    card2 = fb.run_scorecard(
        windows, feature_fn=feat4, sim_fn=neg_l2, n_clusters=5, match_threshold=0.5
    )
    assert card == card2


# ------------------------- evaluate_holdout（参数化版）-------------------------

def test_evaluate_holdout_matches_and_scores() -> None:
    pattern = fb.default_feature_fn(_ramp(40, 60), _ramp(60, 40))
    holdout = [
        _win(1, _ramp(40, 60), _ramp(60, 40), "UP"),   # 同形态同方向 → 命中
        _win(2, _ramp(40, 60), _ramp(60, 40), "DOWN"),  # 同形态异方向 → 触发不命中
    ]
    ev = fb.evaluate_holdout(
        pattern, "UP", holdout, fb.default_feature_fn, fb.default_sim_fn, 0.8
    )
    assert ev["matched"] == 2
    assert ev["correct"] == 1
    assert set(ev["matched_ids"]) == {1, 2}


# ------------------------- 优化点 A：标准化几何接缝 -------------------------

def test_standardized_cosine_uses_train_stats() -> None:
    """标准化几何：用 train 统计 z-score 后再算余弦；std_ctx=None 退化为原始余弦。"""
    a = np.array([10.0, 20.0])
    b = np.array([12.0, 24.0])
    # 无上下文 → 与原始 cosine_sim 一致
    assert fb.standardized_cosine_sim(a, b, None) == pytest.approx(
        fb.cosine_sim(a, b)
    )
    # 有上下文：逐维 (x-mean)/std 后再余弦（手算可验）
    ctx = {"mean": np.array([10.0, 20.0]), "std": np.array([2.0, 4.0])}
    za = (a - ctx["mean"]) / ctx["std"]  # [0, 0]
    zb = (b - ctx["mean"]) / ctx["std"]  # [1, 1]
    assert fb.standardized_cosine_sim(a, b, ctx) == pytest.approx(fb.cosine_sim(za, zb))


def test_make_std_ctx_guards_constant_dim() -> None:
    # 第 2 维为常量（std=0）→ 以 1.0 兜底，不除零
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    ctx = fb.make_std_ctx(X)
    assert ctx["std"][1] == pytest.approx(1.0)
    assert ctx["mean"][0] == pytest.approx(3.0)
    # 空矩阵 → 安全返回
    empty = fb.make_std_ctx(np.zeros((0, 4)))
    assert empty["mean"] is None


def test_standardize_seam_reduces_level_hijack() -> None:
    """水平绑架场景：原始 cosine 被绝对水平拉高（异类也逆近 1），标准化后应拉开。

    守护优化点 A 的有效性：同一批「形态相反但水平接近」的数据，标准化几何的
    异类相似度应显著低于原始几何（即 gap 变大），否则 A 轮无法客观验证。
    """
    hard = []
    for k in range(8):
        hard.append(_win(k * 2 + 1, _ramp(49.5, 50.5), _ramp(50.5, 49.5), "UP"))
        hard.append(_win(k * 2 + 2, _ramp(50.5, 49.5), _ramp(49.5, 50.5), "DOWN"))
    d_raw = fb._discrimination(hard, fb.default_feature_fn, fb.default_sim_fn)

    # 用该批数据自身拟合标准化上下文（模拟 run_scorecard 内部行为）
    feats = np.vstack([
        fb.default_feature_fn(w["curve_up_pct"], w["curve_down_pct"]) for w in hard
    ])
    ctx = fb.make_std_ctx(feats)
    d_std = fb._discrimination(
        hard, fb.default_feature_fn, fb.standardized_cosine_sim, std_ctx=ctx
    )
    assert d_raw["comparable"] and d_std["comparable"]
    # 原始几何被水平绑架：异类相似度逆近 1
    assert d_raw["mean_diff_sim"] > 0.9
    # 标准化后异类相似度显著下降（水平绑架被解除）
    assert d_std["mean_diff_sim"] < d_raw["mean_diff_sim"]
    assert d_std["gap"] > d_raw["gap"]


# ------------------------- 优化点 A3：k-NN 多数投票接缝 -------------------------

def test_knn_evaluate_separable_high_accuracy() -> None:
    """可分数据上 k-NN 应接近全对：验证机制本身正确（形态可分⇒邻居方向可预测）。"""
    windows = fb.synthetic_windows(n_per_group=15)
    train, holdout = fb.time_split(windows, 0.3)
    train_feats = [
        fb.default_feature_fn(w["curve_up_pct"], w["curve_down_pct"]) for w in train
    ]
    ev = fb._knn_evaluate(
        train, train_feats, holdout, fb.default_feature_fn, fb.default_sim_fn, k=3
    )
    assert ev["matched"] > 0
    assert ev["correct"] / ev["matched"] > 0.7  # 可分集应远高于随机


def test_knn_evaluate_empty_and_noise_only() -> None:
    # 空 holdout → 零样本
    ev = fb._knn_evaluate([], [], [], fb.default_feature_fn, fb.default_sim_fn, k=3)
    assert ev == {"matched": 0, "correct": 0}
    # train 全 NOISE → 邻居无方向票，holdout 全弃权 → matched=0
    noise_train = [_win(1, _ramp(50, 50), _ramp(50, 50), "NOISE")]
    nf = [fb.default_feature_fn(w["curve_up_pct"], w["curve_down_pct"]) for w in noise_train]
    holdout = [_win(2, _ramp(40, 60), _ramp(60, 40), "UP")]
    ev2 = fb._knn_evaluate(
        noise_train, nf, holdout, fb.default_feature_fn, fb.default_sim_fn, k=3
    )
    assert ev2["matched"] == 0


def test_run_scorecard_knn_overrides_primary() -> None:
    """knn_k 启用时 primary 来自 k-NN；指纹记录 knn_k；两次运行确定性。"""
    windows = fb.synthetic_windows(n_per_group=15)
    card = fb.run_scorecard(windows, n_clusters=6, knn_k=3)
    assert card["fingerprint"]["knn_k"] == 3
    card2 = fb.run_scorecard(windows, n_clusters=6, knn_k=3)
    assert card == card2  # 确定性


# ------------------------- 优化点 A4：k-NN 信号来源诊断 -------------------------

def test_random_pair_sim_baseline_empty_and_deterministic() -> None:
    assert fb._random_pair_sim_baseline([], [], fb.default_feature_fn, fb.default_sim_fn) == 0.0
    windows = fb.synthetic_windows(n_per_group=10)
    train, holdout = fb.time_split(windows, 0.3)
    train_feats = [
        fb.default_feature_fn(w["curve_up_pct"], w["curve_down_pct"]) for w in train
    ]
    b1 = fb._random_pair_sim_baseline(train_feats, holdout, fb.default_feature_fn, fb.default_sim_fn)
    b2 = fb._random_pair_sim_baseline(train_feats, holdout, fb.default_feature_fn, fb.default_sim_fn)
    assert b1 == b2  # 确定性（固定 seed）


def test_knn_diagnosis_shape_matching_lifts_similarity() -> None:
    """可分集上：k-NN 邻居相似度应显著高于随机配对（证明邻居是按形态选出）。"""
    windows = fb.synthetic_windows(n_per_group=20)
    card = fb.run_scorecard(windows, n_clusters=6, knn_k=3)
    kd = card["diagnostics"]["knn"]
    assert kd["predictions"] > 0
    # 邻居是按形态相似选出 → 相似度应高于随机配对（lift>0；合成 ramp 整体相近，
    # 随机基线本就偏高，故只断言严格为正，幅度交由真实数据判读）
    assert kd["neighbor_sim_mean"] > kd["random_pair_sim_mean"]
    assert kd["sim_lift_vs_random"] > 0.0


# ------------------------- load_windows_from_file（服务器导出桥接取数）-------------------------

def test_load_windows_from_file_json_array(tmp_path) -> None:
    import json

    rows = [
        _win(3, _ramp(40, 60), _ramp(60, 40), "UP", start_time=300),
        _win(1, _ramp(60, 40), _ramp(40, 60), "DOWN", start_time=100),
        _win(2, _ramp(50, 50), _ramp(50, 50), "NOISE", start_time=200),
    ]
    p = tmp_path / "win.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    loaded = fb.load_windows_from_file(str(p))
    # 按 start_time 升序（与直连 DB 口径一致的确定性）
    assert [w["id"] for w in loaded] == [1, 2, 3]


def test_load_windows_from_file_jsonl(tmp_path) -> None:
    import json

    rows = [
        _win(1, _ramp(40, 60), _ramp(60, 40), "UP", start_time=100),
        _win(2, _ramp(60, 40), _ramp(40, 60), "DOWN", start_time=200),
    ]
    p = tmp_path / "win.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = fb.load_windows_from_file(str(p))
    assert [w["id"] for w in loaded] == [1, 2]


def test_load_windows_from_file_empty_and_bad(tmp_path) -> None:
    import json

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert fb.load_windows_from_file(str(empty)) == []

    # psql json_agg 空表返回 '[]'
    empty_agg = tmp_path / "empty_agg.json"
    empty_agg.write_text("[]", encoding="utf-8")
    assert fb.load_windows_from_file(str(empty_agg)) == []

    # 缺必需字段 → 明确报错
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": 1, "outcome": "UP"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        fb.load_windows_from_file(str(bad))


def test_load_windows_from_file_feeds_scorecard(tmp_path) -> None:
    """从文件取数 → run_scorecard 与直接喂内存窗口结果逐字段一致。"""
    import json

    windows = fb.synthetic_windows(n_per_group=12)
    p = tmp_path / "win.json"
    p.write_text(json.dumps(windows), encoding="utf-8")
    loaded = fb.load_windows_from_file(str(p))
    assert fb.run_scorecard(loaded, n_clusters=6) == fb.run_scorecard(windows, n_clusters=6)


# ------------------------- 标签置换安慰剂 -------------------------

def test_placebo_shuffles_outcome_deterministic() -> None:
    """打乱 outcome 后：确定性（同 seed 两次相同）且胜率回落至随机附近。"""
    windows = fb.synthetic_windows(n_per_group=20)

    # 模拟 placebo：固定 seed 打乱 outcome
    def _placebo_windows(ws, seed=42):
        import copy
        ws2 = copy.deepcopy(ws)
        rng = np.random.default_rng(seed)
        outcomes = [w["outcome"] for w in ws2]
        shuffled = rng.permutation(outcomes).tolist()
        for w, o in zip(ws2, shuffled):
            w["outcome"] = o
        return ws2

    pw1 = _placebo_windows(windows, seed=42)
    pw2 = _placebo_windows(windows, seed=42)
    # 确定性
    card1 = fb.run_scorecard(pw1, n_clusters=6, knn_k=3)
    card2 = fb.run_scorecard(pw2, n_clusters=6, knn_k=3)
    assert card1 == card2

    # 打乱后 k-NN 胜率应回落至随机附近（<0.65），远低于未打乱的可分集（>0.9）
    pm = card1["primary_metric"]
    if pm["sample_count"] > 0:
        assert pm["win_rate"] < 0.65, f"placebo win_rate {pm['win_rate']} 太高，打乱可能无效"
