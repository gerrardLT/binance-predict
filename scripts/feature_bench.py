#!/usr/bin/env python3
"""LEARN 环③「特征表征」离线评估台架（确定性、无 LLM、无网络）。

用真实历史情绪窗口（sentiment_windows）量化「特征表征」的质量，产出一张评分卡：
以「样本外 holdout 胜率的 Wilson 95% 置信下界（WLB）」为单一裁决指标，辅以诊断面板
（判别分离度 / holdout 覆盖率 / 簇纯度 / 特征冗余度）解释「为什么好/不好」。

设计要点（服务于「改-验-评闭环，一次只改一个变量」）：
- 台架复用与线上 deep_learn_pycluster 完全同构的确定性内核：
  time_split -> feature_fn -> cluster_windows(seed) -> 每簇 outcome 多数投票定方向
  -> 质心作代表向量 -> 在 holdout 上用 sim_fn 匹配回测。
- 两个可替换「接缝」：
    * feature_fn：曲线 -> 特征向量（默认 = curve_features.extract_features）。B/C 轮替换。
    * sim_fn：两向量相似度（默认 = curve_features.cosine_sim）。A 轮替换为标准化几何。
  台架自带参数化的 holdout 评估，绝不改动 backtest.evaluate_on_holdout（避免污染线上）。
- 纯逻辑（run_scorecard 及各指标函数）无 I/O，可离线单测；DB 取数与 CLI 是薄封装。

用法示例：
    python scripts/feature_bench.py                       # 直连 DB，全量历史，冻结 baseline
    python scripts/feature_bench.py --days-back 30         # 仅近 30 天
    python scripts/feature_bench.py --from-file win.json   # 从服务器导出的 JSON 文件取数（DB 不可直连时）
    python scripts/feature_bench.py --synthetic            # 强制合成数据（仅验证台架管道）
    python scripts/feature_bench.py --compare output/feature_bench_baseline.json
    python scripts/feature_bench.py --out output/feature_bench_A.json --variant A-zscore

服务器导出（DB 在 Docker 内网、本地不可直连时的桥接方案）：
    在部署服务器 /www/wwwroot/binance-predict/ 下执行，把窗口表导成 JSON，再传回本地：
    docker compose -f docker/docker-compose.prod.yml exec -T db \
      psql -U postgres -d binance_predict -t -A -c \
      "SELECT COALESCE(json_agg(t), '[]') FROM (
         SELECT id, start_time, end_time, curve_up_pct, curve_down_pct,
                outcome, actual_return, sample_count
         FROM sentiment_windows
         WHERE outcome IS NOT NULL AND curve_up_pct IS NOT NULL
           AND curve_down_pct IS NOT NULL
         ORDER BY start_time ASC) t" > sentiment_windows.json

退出码：0=正常产出，2=数据不足（裁决指标不可信），3=执行异常。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# 允许以 `python scripts/feature_bench.py` 直接运行（补齐 src 到 import 路径）
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from binance_predict.services.backtest import (  # noqa: E402
    HOLDOUT_MATCH_THRESHOLD,
    snapshot_token,
    time_split,
    wilson_lower_bound,
)
from binance_predict.services.curve_features import (  # noqa: E402
    FEATURE_DIM,
    cluster_windows,
    cosine_sim,
    extract_features,
)

# ------------------------- 常量 -------------------------

# 当前台架默认评的是「原始 24 维特征 + 原始 cosine」这版表征
DEFAULT_FEATURE_VERSION = "baseline-24d-rawcos"
DEFAULT_SEED = 42
RANDOM_BASELINE = 0.5  # 方向二选一的随机基线
MIN_JUDGE_SAMPLES = 30  # holdout 决策样本低于此值则裁决指标不可信
SEPARATION_SAMPLE_CAP = 400  # 判别分离度两两比较的窗口数上限（确定性取时间最早的前 N 个）


# ------------------------- 可替换接缝（默认 = 当前线上实现）-------------------------

def default_feature_fn(curve_up, curve_down) -> np.ndarray:
    """默认特征接缝：与线上 curve_features.extract_features 完全一致（24 维）。"""
    return extract_features(curve_up, curve_down)


def default_sim_fn(a, b, std_ctx=None) -> float:
    """默认相似度接缝：与线上 curve_features.cosine_sim 完全一致（原始向量余弦）。

    三参签名（第三参 std_ctx 为几何上下文，默认实现忽略），以便 A 轮的标准化
    几何变体 standardized_cosine_sim 可无缝替换同一接缝。
    """
    return cosine_sim(a, b)


def make_std_ctx(matrix: np.ndarray) -> dict:
    """由 train 特征矩阵拟合固定 z-score 统计（逐维 mean/std），供标准化几何用。

    与 KMeans 的 StandardScaler 同构（逐维标准化）；std≈0 的常量维以 1.0 兜底避免除零。
    关键：统计量只在 train 上拟合，holdout 复用同一套（防泄漏、保证确定性）。
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return {"mean": None, "std": None}
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {"mean": mean, "std": std}


def standardized_cosine_sim(a, b, std_ctx=None) -> float:
    """优化点 A 的几何：先用 train 的 z-score 统计标准化两向量，再算余弦。

    把 holdout 匹配/去重的相似度统一到与 KMeans 一致的标准化空间，消除「绝对水平
    维主宰余弦方向」的绑架（缺陷 A）。std_ctx 为 None 时退化为原始余弦（安全兜底）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if std_ctx is not None and std_ctx.get("mean") is not None:
        a = (a - std_ctx["mean"]) / std_ctx["std"]
        b = (b - std_ctx["mean"]) / std_ctx["std"]
    return cosine_sim(a, b)


# ------------------------- 纯统计工具 -------------------------

def _bucket_stats(correct: int, total: int) -> dict:
    """把 (命中数, 样本数) 折算为标准统计口径（对齐 evolution_metrics._bucket_stats）。"""
    win_rate = (correct / total) if total > 0 else 0.0
    ci_lower = wilson_lower_bound(correct, total)
    return {
        "sample_count": int(total),
        "correct": int(correct),
        "win_rate": round(win_rate, 4),
        "ci_lower": round(ci_lower, 4),
        "excess_over_random": round(win_rate - RANDOM_BASELINE, 4),
        "beats_random": bool(ci_lower > RANDOM_BASELINE),
    }


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并列值），确定性（mergesort 稳定排序）。供 AUC 计算用。"""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = a.argsort(kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        # 秩从 1 起，区间 [i, j] 取平均秩
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney U 对应的 AUC = P(pos 分数 > neg 分数)（含并列 0.5 计）。

    在本台架语义：pos=同 outcome 窗口对的相似度，neg=异 outcome 窗口对的相似度。
    AUC>0.5 表示「同类更像、异类更不像」，即特征/几何具备形态判别力。
    """
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    allv = np.concatenate([np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)])
    ranks = _rankdata_average(allv)
    sum_pos = float(ranks[:n_pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ------------------------- 台架内参数化 holdout 评估 -------------------------

def evaluate_holdout(
    pattern_features,
    direction: str,
    holdout_windows: list[dict],
    feature_fn,
    sim_fn,
    match_threshold: float,
    std_ctx=None,
) -> dict:
    """参数化版 holdout 回测：对每个 holdout 窗口用 feature_fn 提特征，与代表向量
    用 sim_fn 算相似度，>=阈值视为触发；触发窗口中 outcome==direction 记命中
    （NOISE 计入分母不计命中）。返回 {matched, correct, matched_ids}。

    与 backtest.evaluate_on_holdout 语义一致，但接缝可替换、且回传命中的窗口 id
    以便台架计算 holdout 覆盖率。绝不改动线上 evaluate_on_holdout。
    """
    pf = np.asarray(pattern_features, dtype=float)
    matched = 0
    correct = 0
    matched_ids: list[int] = []
    for w in holdout_windows:
        feat = feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
        if float(sim_fn(pf, feat, std_ctx)) >= match_threshold:
            matched += 1
            matched_ids.append(int(w.get("id", -1)))
            if (w.get("outcome") or "").upper() == direction.upper():
                correct += 1
    return {"matched": matched, "correct": correct, "matched_ids": matched_ids}


# ------------------------- 诊断指标 -------------------------

def _discrimination(
    windows: list[dict], feature_fn, sim_fn, sample_cap: int = SEPARATION_SAMPLE_CAP,
    std_ctx=None,
) -> dict:
    """判别分离度：仅取 UP/DOWN 窗口，两两算相似度，比较「同 outcome 对」vs
    「异 outcome 对」的相似度分布。返回 mean_same/mean_diff/gap/auc/pair 计数。

    直接量化「相似度是否被绝对水平绑架」——若绑架，则同类与异类相似度都高、几乎
    不可分，gap≈0、auc≈0.5（这正是优化点 A 的靶标）。
    """
    decisive = [
        w for w in windows if (w.get("outcome") or "").upper() in ("UP", "DOWN")
    ]
    # 确定性截断：按 start_time 升序取前 sample_cap 个，控制 O(n^2)
    decisive = sorted(decisive, key=lambda w: w.get("start_time", 0))[:sample_cap]
    n = len(decisive)
    if n < 3:
        return {
            "comparable": False,
            "reason": "UP/DOWN 窗口不足（<3），无法比较",
            "up_down_count": n,
        }
    feats = [
        feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
        for w in decisive
    ]
    outcomes = [(w.get("outcome") or "").upper() for w in decisive]
    same_scores: list[float] = []
    diff_scores: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_fn(feats[i], feats[j], std_ctx))
            if outcomes[i] == outcomes[j]:
                same_scores.append(s)
            else:
                diff_scores.append(s)
    mean_same = float(np.mean(same_scores)) if same_scores else 0.0
    mean_diff = float(np.mean(diff_scores)) if diff_scores else 0.0
    auc = _auc(same_scores, diff_scores)
    return {
        "comparable": bool(same_scores and diff_scores),
        "up_down_count": n,
        "same_pairs": len(same_scores),
        "diff_pairs": len(diff_scores),
        "mean_same_sim": round(mean_same, 4),
        "mean_diff_sim": round(mean_diff, 4),
        "gap": round(mean_same - mean_diff, 4),
        "auc": round(auc, 4) if auc is not None else None,
    }


def _redundancy(matrix: np.ndarray) -> dict:
    """特征冗余度：平均绝对相关系数 + 有效秩（奇异值熵）。优化点 B 的靶标。

    - mean_abs_corr：特征维两两相关的平均绝对值（越低越不冗余）。常量列跳过。
    - effective_rank：exp(奇异值分布的香农熵)，衡量「有效独立维度数」（越接近维度数越好）。
    """
    X = np.asarray(matrix, dtype=float)
    n, d = X.shape if X.ndim == 2 else (0, 0)
    if n < 2 or d < 2:
        return {"comparable": False, "reason": "样本或维度不足", "n_dims": int(d)}

    # 平均绝对相关（跳过零方差列，避免 nan 污染）
    stds = X.std(axis=0)
    valid = stds > 1e-12
    Xv = X[:, valid]
    if Xv.shape[1] >= 2:
        corr = np.corrcoef(Xv.T)
        off = corr[~np.eye(corr.shape[0], dtype=bool)]
        mean_abs_corr = float(np.mean(np.abs(off)))
    else:
        mean_abs_corr = 0.0

    # 有效秩：对中心化矩阵做 SVD，奇异值归一化后取熵指数
    Xc = X - X.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        eff_rank = 0.0
    else:
        p = sv / sv.sum()
        entropy = float(-np.sum(p * np.log(p)))
        eff_rank = float(np.exp(entropy))

    return {
        "comparable": True,
        "n_dims": int(d),
        "constant_dims": int(np.sum(~valid)),
        "mean_abs_corr": round(mean_abs_corr, 4),
        "effective_rank": round(eff_rank, 4),
    }


def _holdout_coverage(
    representatives: list[dict], holdout_windows: list[dict], feature_fn, sim_fn,
    match_threshold: float, std_ctx=None,
) -> dict:
    """holdout 覆盖率：被 >=1 个模式（代表向量）以阈值匹配的 holdout 窗口占比。

    覆盖率异常高（接近 1）往往是「阈值失效、几乎无差别匹配」的信号（优化点 A 的旁证）。
    """
    total = len(holdout_windows)
    if total == 0 or not representatives:
        return {"total": total, "covered": 0, "coverage": 0.0}
    covered_ids: set[int] = set()
    # 预提 holdout 特征，避免每个代表向量重复提特征
    hfeats = [
        (idx, feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or []))
        for idx, w in enumerate(holdout_windows)
    ]
    for rep in representatives:
        pf = np.asarray(rep["centroid"], dtype=float)
        for idx, feat in hfeats:
            if float(sim_fn(pf, feat, std_ctx)) >= match_threshold:
                covered_ids.add(idx)
    covered = len(covered_ids)
    return {"total": total, "covered": covered, "coverage": round(covered / total, 4)}


# ------------------------- k-NN 评估（优化点 A3：换匹配机制）-------------------------

def _knn_evaluate(
    train_windows: list[dict],
    train_feats: list,
    holdout_windows: list[dict],
    feature_fn,
    sim_fn,
    k: int,
    std_ctx=None,
    collect_neighbors: bool = False,
) -> dict:
    """k 近邻多数投票 holdout 评估（替代「质心单向量匹配」的另一种裁决机制）。

    对每个 holdout 窗口：在全部 train 窗口中按 sim_fn 取相似度 top-k 邻居，以其
    outcome 多数投票预测方向（NOISE 邻居不计票；全 NOISE 或平票则弃权不计样本）；
    预测==holdout 真实 outcome（UP/DOWN）记命中。返回 {matched, correct}。

    直接检验「个体邻居的方向」是否比「簇平均质心的方向」更能预测样本外结果。
    """
    k = max(1, int(k))
    n_train = len(train_windows)
    train_outcomes = [(w.get("outcome") or "NOISE").upper() for w in train_windows]
    matched = 0
    correct = 0
    neighbor_detail: list[dict] = []
    for w in holdout_windows:
        actual = (w.get("outcome") or "").upper()
        if actual not in ("UP", "DOWN"):
            continue  # NOISE holdout 不计入裁决样本
        feat = feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
        sims = [
            (float(sim_fn(feat, train_feats[i], std_ctx)), i) for i in range(n_train)
        ]
        sims.sort(key=lambda x: x[0], reverse=True)
        votes = Counter(
            train_outcomes[i]
            for _, i in sims[: min(k, n_train)]
            if train_outcomes[i] in ("UP", "DOWN")
        )
        if not votes:
            continue  # 邻居全为 NOISE，弃权
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # UP/DOWN 平票，弃权
        matched += 1
        if top[0][0] == actual:
            correct += 1
        if collect_neighbors:
            nb = sims[: min(k, n_train)]
            ht = w.get("start_time")
            neighbor_detail.append({
                "sims": [s for s, _ in nb],
                "time_gaps": [
                    abs(int(ht) - int(train_windows[i].get("start_time", ht)))
                    for _, i in nb
                ] if ht is not None else [],
                "correct": top[0][0] == actual,
            })
    out = {"matched": matched, "correct": correct}
    if collect_neighbors:
        out["neighbor_detail"] = neighbor_detail
    return out


def _random_pair_sim_baseline(
    train_feats: list, holdout_windows: list[dict], feature_fn, sim_fn, std_ctx=None,
    cap: int = 300, seed: int = 1234,
) -> float:
    """随机「holdout×train」配对的平均相似度（确定性采样）——作形态匹配强度的基线。

    k-NN 邻居相似度若显著高于此基线，说明邻居确是按形态相似选出（而非随便抓的
    时间相邻窗口）；两者接近则提示邻居只是时间上挨得近（形态信号弱）。
    """
    n_train = len(train_feats)
    n_hold = len(holdout_windows)
    if n_train == 0 or n_hold == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    h_idx = rng.choice(n_hold, size=min(cap, n_hold), replace=False)
    t_idx = rng.choice(n_train, size=min(cap, n_train), replace=False)
    sims = []
    for hi, ti in zip(h_idx, t_idx):
        w = holdout_windows[int(hi)]
        feat = feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
        sims.append(float(sim_fn(feat, train_feats[int(ti)], std_ctx)))
    return float(np.mean(sims)) if sims else 0.0


# ------------------------- 核心：评分卡 -------------------------

def run_scorecard(
    windows: list[dict],
    *,
    feature_fn=default_feature_fn,
    sim_fn=default_sim_fn,
    holdout_ratio: float = 0.3,
    n_clusters: int = 25,
    match_threshold: float = HOLDOUT_MATCH_THRESHOLD,
    seed: int = DEFAULT_SEED,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    decisive: bool = True,
    std_ctx=None,
    knn_k=None,
) -> dict:
    """对一批窗口跑出确定性评分卡（无 I/O，可离线单测）。

    Args:
        windows: 窗口 dict 列表，每条含 curve_up_pct/curve_down_pct/outcome/start_time[/id]。
        feature_fn/sim_fn: 可替换接缝（默认 = 线上实现）。
        holdout_ratio/n_clusters/match_threshold/seed: 确定性参数。
        feature_version: 变体标签，写入指纹用于 A/B/C 对比溯源。
        decisive: 是否为「可裁决」运行（合成数据降级时置 False）。

    Returns:
        评分卡 dict（含 fingerprint / data_health / primary / diagnostics / clusters）。
    """
    # --- 指纹（一致性溯源）---
    ids = [int(w.get("id", i)) for i, w in enumerate(windows)]
    fingerprint = {
        "feature_version": feature_version,
        "seed": int(seed),
        "holdout_ratio": float(holdout_ratio),
        "n_clusters": int(n_clusters),
        "match_threshold": float(match_threshold),
        "window_count": len(windows),
        "snapshot_token": snapshot_token(ids) if ids else snapshot_token([]),
        "decisive": bool(decisive),
        "knn_k": knn_k,
    }

    # --- 数据体检 ---
    outcome_dist = Counter((w.get("outcome") or "NOISE").upper() for w in windows)
    train_windows, holdout_windows = time_split(windows, holdout_ratio)
    data_health = {
        "total_windows": len(windows),
        "outcome_dist": dict(sorted(outcome_dist.items())),
        "train_count": len(train_windows),
        "holdout_count": len(holdout_windows),
        "min_judge_samples": MIN_JUDGE_SAMPLES,
    }

    # --- train 提特征 + 聚类 ---
    train_feats = [
        np.asarray(
            feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or []),
            dtype=float,
        )
        for w in train_windows
    ]
    dim = len(train_feats[0]) if train_feats else FEATURE_DIM
    matrix = np.vstack(train_feats) if train_feats else np.zeros((0, dim))
    # 标准化几何上下文：由 train 特征拟合（A 轮启用；默认 None 时 sim_fn 退化为原始余弦）
    if std_ctx is None:
        std_ctx = make_std_ctx(matrix)
    labels = cluster_windows(matrix, n_clusters, random_state=seed)
    label_list = [int(x) for x in labels.tolist()]

    # --- 每簇多数投票 + holdout 回测 ---
    clusters: list[dict] = []
    representatives: list[dict] = []
    agg_matched = 0
    agg_correct = 0
    purity_shares: list[float] = []
    for cid in sorted(set(label_list)):
        member_idx = [i for i, lb in enumerate(label_list) if lb == cid]
        if not member_idx:
            continue
        votes = Counter(
            (train_windows[i].get("outcome") or "NOISE").upper() for i in member_idx
        )
        direction, top_votes = votes.most_common(1)[0]
        n_members = len(member_idx)
        purity_shares.append(top_votes / n_members)
        if direction == "NOISE":
            # NOISE 主导簇无可交易方向，丢弃（与线上一致）
            continue
        centroid = np.mean(np.vstack([train_feats[i] for i in member_idx]), axis=0)
        ev = evaluate_holdout(
            centroid, direction, holdout_windows, feature_fn, sim_fn, match_threshold,
            std_ctx,
        )
        agg_matched += ev["matched"]
        agg_correct += ev["correct"]
        representatives.append({"centroid": centroid.tolist(), "direction": direction})
        cstat = _bucket_stats(ev["correct"], ev["matched"])
        clusters.append({
            "cluster_id": cid,
            "direction": direction,
            "member_count": n_members,
            "vote_up": int(votes.get("UP", 0)),
            "vote_down": int(votes.get("DOWN", 0)),
            "vote_noise": int(votes.get("NOISE", 0)),
            "holdout_matched": ev["matched"],
            "holdout_correct": ev["correct"],
            "holdout_win_rate": cstat["win_rate"],
            "holdout_ci_lower": cstat["ci_lower"],
        })

    # --- 裁决指标（primary）：跨簇汇总 holdout 命中 → WLB ---
    primary = _bucket_stats(agg_correct, agg_matched)
    if not decisive:
        primary_verdict = "PIPELINE_CHECK_ONLY"
    elif primary["sample_count"] < MIN_JUDGE_SAMPLES:
        primary_verdict = "INSUFFICIENT_SAMPLES"
    elif primary["beats_random"]:
        primary_verdict = "BEATS_RANDOM"
    else:
        primary_verdict = "INCONCLUSIVE"
    primary["verdict"] = primary_verdict

    # --- 优化点 A3：k-NN 裁决（启用时以全局 k 近邻投票替代质心匹配作 primary）---
    if knn_k:
        knn_ev = _knn_evaluate(
            train_windows, train_feats, holdout_windows, feature_fn, sim_fn,
            knn_k, std_ctx, collect_neighbors=True,
        )
        primary = _bucket_stats(knn_ev["correct"], knn_ev["matched"])
        if not decisive:
            primary["verdict"] = "PIPELINE_CHECK_ONLY"
        elif primary["sample_count"] < MIN_JUDGE_SAMPLES:
            primary["verdict"] = "INSUFFICIENT_SAMPLES"
        elif primary["beats_random"]:
            primary["verdict"] = "BEATS_RANDOM"
        else:
            primary["verdict"] = "INCONCLUSIVE"
        # A4 信号来源诊断：邻居相似度 vs 随机基线 + 邻居时间间隔
        detail = knn_ev.get("neighbor_detail", [])
        all_sims = [s for d in detail for s in d["sims"]]
        all_gaps = [g for d in detail for g in d["time_gaps"]]
        knn_diag = {
            "predictions": len(detail),
            "abstentions": sum(
                1 for w in holdout_windows
                if (w.get("outcome") or "").upper() in ("UP", "DOWN")
            ) - len(detail),
            "neighbor_sim_mean": round(float(np.mean(all_sims)), 4) if all_sims else None,
            "neighbor_sim_min": round(float(np.min(all_sims)), 4) if all_sims else None,
            "random_pair_sim_mean": round(
                _random_pair_sim_baseline(
                    train_feats, holdout_windows, feature_fn, sim_fn, std_ctx
                ), 4),
            "neighbor_time_gap_median_min": (
                round(float(np.median(all_gaps)) / 60_000, 1) if all_gaps else None
            ),
        }
        knn_diag["sim_lift_vs_random"] = (
            round(knn_diag["neighbor_sim_mean"] - knn_diag["random_pair_sim_mean"], 4)
            if knn_diag["neighbor_sim_mean"] is not None else None
        )

    # --- 诊断面板 ---
    diagnostics = {
        "discrimination": _discrimination(windows, feature_fn, sim_fn, std_ctx=std_ctx),
        "holdout_coverage": _holdout_coverage(
            representatives, holdout_windows, feature_fn, sim_fn, match_threshold,
            std_ctx=std_ctx,
        ),
        "cluster_purity_mean": (
            round(float(np.mean(purity_shares)), 4) if purity_shares else 0.0
        ),
        "redundancy": _redundancy(matrix),
        "non_noise_clusters": len(clusters),
        "total_clusters": len(set(label_list)) if label_list else 0,
    }
    if knn_k:
        diagnostics["knn"] = knn_diag

    return {
        "fingerprint": fingerprint,
        "data_health": data_health,
        "primary_metric": primary,
        "diagnostics": diagnostics,
        "clusters": clusters,
    }


# ------------------------- 合成数据（降级路径 / 冒烟）-------------------------

def _ramp_curve(start: float, end: float, n: int = 12) -> list[dict]:
    """构造一条从 start 线性变到 end 的 [{t, v}] 曲线。"""
    vals = np.linspace(start, end, n)
    return [{"t": i, "v": float(v)} for i, v in enumerate(vals)]


def synthetic_windows(n_per_group: int = 20) -> list[dict]:
    """构造可分的合成窗口集（UP 单边上行 / DOWN 单边下行 / NOISE 横盘），
    用于「真实数据不足」时验证台架管道正确性（非裁决）。确定性、无随机。"""
    windows: list[dict] = []
    wid = 0
    t0 = 1_700_000_000_000
    for k in range(n_per_group):
        # UP：UP% 上行、DOWN% 下行
        windows.append({
            "id": (wid := wid + 1),
            "start_time": t0 + wid * 300_000,
            "curve_up_pct": _ramp_curve(45.0 + k * 0.1, 62.0 + k * 0.1),
            "curve_down_pct": _ramp_curve(55.0 - k * 0.1, 38.0 - k * 0.1),
            "outcome": "UP",
        })
        # DOWN：UP% 下行、DOWN% 上行
        windows.append({
            "id": (wid := wid + 1),
            "start_time": t0 + wid * 300_000,
            "curve_up_pct": _ramp_curve(55.0 - k * 0.1, 38.0 - k * 0.1),
            "curve_down_pct": _ramp_curve(45.0 + k * 0.1, 62.0 + k * 0.1),
            "outcome": "DOWN",
        })
        # NOISE：两条曲线横盘
        windows.append({
            "id": (wid := wid + 1),
            "start_time": t0 + wid * 300_000,
            "curve_up_pct": _ramp_curve(50.0, 50.5),
            "curve_down_pct": _ramp_curve(50.0, 49.5),
            "outcome": "NOISE",
        })
    return windows


# ------------------------- DB 取数（薄封装）-------------------------

async def fetch_windows(days_back: int | None, limit: int | None) -> list[dict]:
    """直连数据库拉取 outcome 非空且曲线非空的窗口，按 start_time 升序（确定性）。"""
    import time

    from sqlalchemy import select

    from binance_predict.db.engine import async_session_factory
    from binance_predict.db.models import SentimentWindow

    async with async_session_factory() as session:
        stmt = select(SentimentWindow).where(
            SentimentWindow.outcome.isnot(None),
            SentimentWindow.curve_up_pct.isnot(None),
            SentimentWindow.curve_down_pct.isnot(None),
        )
        if days_back:
            cutoff = int(time.time() * 1000) - int(days_back) * 86_400_000
            stmt = stmt.where(SentimentWindow.start_time >= cutoff)
        stmt = stmt.order_by(SentimentWindow.start_time.asc())
        if limit:
            stmt = stmt.limit(int(limit))
        rows = (await session.execute(stmt)).scalars().all()
        return [
            {
                "id": w.id,
                "start_time": w.start_time,
                "end_time": w.end_time,
                "curve_up_pct": w.curve_up_pct,
                "curve_down_pct": w.curve_down_pct,
                "outcome": w.outcome,
                "actual_return": w.actual_return,
                "sample_count": w.sample_count,
            }
            for w in rows
        ]


# 台架内核只依赖这几个键；其余原样透传（便于溯源）
_REQUIRED_WINDOW_KEYS = ("curve_up_pct", "curve_down_pct", "outcome")


def load_windows_from_file(path: str) -> list[dict]:
    """从服务器导出的文件读取窗口（DB 不可直连时的桥接取数），确定性、无网络。

    支持两种格式（自动识别）：
    - JSON 数组：`[{...}, {...}]`（推荐，psql `json_agg` 直出）。
    - JSONL：每行一个 JSON 对象。

    每条至少需含 curve_up_pct / curve_down_pct / outcome；曲线为 [{t, v}] 结构。
    读取后按 start_time 升序排序（缺失则按 id）以保证与直连 DB 口径一致的确定性。
    """
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []

    data: object
    if text[0] == "[":
        data = json.loads(text)
    else:
        # JSONL 兜底：逐行解析（跳过空行）
        data = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(data, list):
        raise ValueError(
            f"文件内容应为 JSON 数组或 JSONL，实际顶层类型={type(data).__name__}"
        )

    windows: list[dict] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"第 {i} 条记录不是对象：{type(row).__name__}")
        missing = [k for k in _REQUIRED_WINDOW_KEYS if k not in row]
        if missing:
            raise ValueError(f"第 {i} 条记录缺少必需字段 {missing}")
        windows.append(row)

    windows.sort(key=lambda w: (w.get("start_time") is None, w.get("start_time", w.get("id", 0))))
    return windows


# ------------------------- 输出 / 对比 -------------------------

def _print_scorecard(card: dict) -> None:
    """人类可读评分卡。"""
    fp = card["fingerprint"]
    dh = card["data_health"]
    pm = card["primary_metric"]
    dg = card["diagnostics"]

    print("=" * 68)
    print(f"特征表征评分卡  |  variant={fp['feature_version']}  seed={fp['seed']}")
    print(f"指纹 snapshot_token={fp['snapshot_token']}  window_count={fp['window_count']}")
    print("=" * 68)

    print("\n[数据体检]")
    print(f"  总窗口={dh['total_windows']}  outcome 分布={dh['outcome_dist']}")
    print(f"  train={dh['train_count']}  holdout={dh['holdout_count']}  "
          f"(裁决门槛 holdout>={dh['min_judge_samples']})")

    print("\n[裁决指标] 样本外 holdout 胜率 Wilson 95% 下界（WLB）")
    print(f"  结论: {pm['verdict']}")
    print(f"  决策样本={pm['sample_count']}  命中={pm['correct']}  "
          f"胜率={pm['win_rate']:.1%}  WLB={pm['ci_lower']:.4f}  "
          f"超额={pm['excess_over_random']:+.4f}  跑赢随机={pm['beats_random']}")

    print("\n[诊断面板]")
    disc = dg["discrimination"]
    if disc.get("comparable"):
        print(f"  判别分离度: gap={disc['gap']:+.4f}  auc={disc['auc']}  "
              f"(同类均值={disc['mean_same_sim']} / 异类均值={disc['mean_diff_sim']}, "
              f"UP/DOWN 窗口={disc['up_down_count']})")
    else:
        print(f"  判别分离度: 不可比（{disc.get('reason', '?')}）")
    cov = dg["holdout_coverage"]
    print(f"  holdout 覆盖率: {cov['coverage']:.1%}  ({cov['covered']}/{cov['total']})")
    print(f"  簇纯度均值: {dg['cluster_purity_mean']}")
    red = dg["redundancy"]
    if red.get("comparable"):
        print(f"  特征冗余度: mean_abs_corr={red['mean_abs_corr']}  "
              f"effective_rank={red['effective_rank']}/{red['n_dims']}  "
              f"(常量维={red['constant_dims']})")
    else:
        print(f"  特征冗余度: 不可比（{red.get('reason', '?')}）")
    print(f"  非 NOISE 簇={dg['non_noise_clusters']}/{dg['total_clusters']}")

    if "knn" in dg:
        kd = dg["knn"]
        print("\n[k-NN 信号来源诊断]（A4：形态匹配 vs 时间泄漏）")
        print(f"  预测={kd['predictions']}  弃权={kd['abstentions']}")
        print(f"  邻居相似度: 均值={kd['neighbor_sim_mean']}  最低={kd['neighbor_sim_min']}  "
              f"随机基线={kd['random_pair_sim_mean']}  抬升={kd['sim_lift_vs_random']}")
        print(f"  邻居时间间隔中位={kd['neighbor_time_gap_median_min']} 分钟")
        if kd["sim_lift_vs_random"] is not None and kd["sim_lift_vs_random"] > 0.2:
            print("  判读: 邻居相似度显著高于随机 → 以形态匹配为主，信号较可信。")
        else:
            print("  判读: 邻居相似度接近随机 → 警惕时间自相关泄漏，需挤泄漏后重测。")

    print("\n[逐簇明细]（仅非 NOISE）")
    if card["clusters"]:
        for c in card["clusters"]:
            print(f"  C{c['cluster_id']:<3} {c['direction']:<4} "
                  f"members={c['member_count']:<3} "
                  f"(UP={c['vote_up']}/DOWN={c['vote_down']}/NOISE={c['vote_noise']}) "
                  f"holdout {c['holdout_correct']}/{c['holdout_matched']} "
                  f"WLB={c['holdout_ci_lower']:.4f}")
    else:
        print("  （无非 NOISE 簇）")
    print()


def _diff_scorecards(baseline: dict, current: dict) -> None:
    """逐字段 diff：先校验指纹（数据是否同一批），再对比关键指标。"""
    b_fp = baseline.get("fingerprint", {})
    c_fp = current.get("fingerprint", {})
    print("-" * 68)
    print("对比 baseline")
    print("-" * 68)
    if b_fp.get("snapshot_token") != c_fp.get("snapshot_token"):
        print("  [警告] snapshot_token 不一致 —— 两次运行数据批次不同，指标对比无效！")
        print(f"    baseline={b_fp.get('snapshot_token')}  current={c_fp.get('snapshot_token')}")
        return
    print(f"  variant: {b_fp.get('feature_version')} -> {c_fp.get('feature_version')}")

    b_pm = baseline.get("primary_metric", {})
    c_pm = current.get("primary_metric", {})
    d_wlb = round(c_pm.get("ci_lower", 0.0) - b_pm.get("ci_lower", 0.0), 4)
    print(f"  裁决 WLB: {b_pm.get('ci_lower')} -> {c_pm.get('ci_lower')}  (Δ={d_wlb:+.4f})")
    print(f"  样本外胜率: {b_pm.get('win_rate')} -> {c_pm.get('win_rate')}")
    print(f"  决策样本: {b_pm.get('sample_count')} -> {c_pm.get('sample_count')}")

    b_dg = baseline.get("diagnostics", {})
    c_dg = current.get("diagnostics", {})
    b_disc = b_dg.get("discrimination", {})
    c_disc = c_dg.get("discrimination", {})
    print(f"  判别 gap: {b_disc.get('gap')} -> {c_disc.get('gap')}")
    print(f"  判别 auc: {b_disc.get('auc')} -> {c_disc.get('auc')}")
    print(f"  holdout 覆盖率: {b_dg.get('holdout_coverage', {}).get('coverage')} -> "
          f"{c_dg.get('holdout_coverage', {}).get('coverage')}")
    b_red = b_dg.get("redundancy", {})
    c_red = c_dg.get("redundancy", {})
    print(f"  冗余 mean_abs_corr: {b_red.get('mean_abs_corr')} -> {c_red.get('mean_abs_corr')}")
    print(f"  有效秩: {b_red.get('effective_rank')} -> {c_red.get('effective_rank')}")

    # 门禁提示（WLB 不下降 eps=0.005）
    eps = 0.005
    if c_pm.get("sample_count", 0) < MIN_JUDGE_SAMPLES:
        print("  [门禁] 决策样本不足，不下裁决结论。")
    elif d_wlb >= -eps:
        print(f"  [门禁] WLB 未下降（Δ={d_wlb:+.4f} >= -{eps}）→ 满足保留前提，再看诊断项。")
    else:
        print(f"  [门禁] WLB 下降（Δ={d_wlb:+.4f} < -{eps}）→ 建议回退。")
    print()


def truncate_windows(windows: list[dict], keep_seconds: float) -> list[dict]:
    """决策点截断：只保留每条曲线开窗后前 keep_seconds 的采样点。

    outcome 保持整窗结果不变——模拟「在第 keep_seconds 秒下注、等整窗结算」，
    用于剔除曲线尾部的「答案读取」成分，检验前段是否存在真实领先信号。
    """
    out = []
    for w in windows:
        w2 = dict(w)
        for key in ("curve_up_pct", "curve_down_pct"):
            curve = w.get(key) or []
            if not curve:
                w2[key] = curve
                continue
            pts = sorted(curve, key=lambda p: p.get("t", 0))
            t0 = pts[0].get("t", 0)
            kept = [p for p in pts if (p.get("t", 0) - t0) <= keep_seconds * 1000.0]
            if len(kept) < 2:  # 特征提取至少需要 2 个点
                kept = pts[:2]
            w2[key] = kept
        out.append(w2)
    return out


# ------------------------- CLI -------------------------

async def _run(args: argparse.Namespace) -> int:
    from binance_predict.config.settings import settings

    holdout_ratio = (
        args.holdout_ratio
        if args.holdout_ratio is not None
        else settings.agent_deep_learn_holdout_ratio
    )
    n_clusters = (
        args.clusters
        if args.clusters is not None
        else settings.agent_deep_learn_target_clusters
    )

    # 优化点 A：几何统一——holdout 匹配/去重的相似度改用 train z-score 标准化空间
    sim_fn = standardized_cosine_sim if args.standardize else default_sim_fn

    decisive = True
    if args.synthetic:
        windows = synthetic_windows()
        decisive = False
        print("[提示] --synthetic：使用合成数据，仅验证台架管道，不用于裁决。")
    elif args.from_file:
        try:
            windows = load_windows_from_file(args.from_file)
        except Exception as exc:  # noqa: BLE001 —— CLI 顶层兜底
            print(f"[ERROR] 从文件取数失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 3
        print(f"[提示] --from-file：从 {args.from_file} 读取 {len(windows)} 个窗口。")
    else:
        try:
            windows = await fetch_windows(args.days_back, args.limit)
        except Exception as exc:  # noqa: BLE001 —— CLI 顶层兜底
            print(f"[ERROR] 数据库取数失败: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("        DB 不可直连时可用 --from-file <服务器导出的 JSON>，"
                  "或 --synthetic 仅验证台架管道。", file=sys.stderr)
            return 3

    # 决策点截断：剔除曲线尾部答案读取成分（须在特征提取前、安慰剂前完成）
    if args.truncate_sec is not None:
        windows = truncate_windows(windows, args.truncate_sec)
        print(f"[截断] 仅保留每条曲线开窗后前 {args.truncate_sec}s 的采样点；"
              "outcome 仍为整窗结果（模拟提前下注）。")

    # 标签置换安慰剂：随机打乱 outcome，切断「形态→结果」的真实映射。
    # 若准确率掉回 ~50%，则证明真实运行的高准确率来自形态-结果关系，非隐藏泄漏。
    if args.placebo:
        rng = np.random.default_rng(args.seed)  # 固定 seed 保证确定性
        outcomes = [w.get("outcome") for w in windows]
        shuffled = rng.permutation(outcomes).tolist()
        for w, o in zip(windows, shuffled):
            w["outcome"] = o
        decisive = False
        print(f"[安慰剂] 已随机打乱 {len(windows)} 个窗口的 outcome 标签（seed={args.seed}）。")
        print("         若胜率回落至 ~50%，证明真实信号来自形态-结果关系，非泄漏。")

    card = run_scorecard(
        windows,
        holdout_ratio=holdout_ratio,
        n_clusters=n_clusters,
        match_threshold=args.threshold,
        seed=args.seed,
        feature_version=args.variant,
        decisive=decisive,
        sim_fn=sim_fn,
        knn_k=args.knn,
    )

    # 真实数据但 holdout 不足：降级——补跑一次合成管道校验（明确标注非裁决）
    degraded = (
        not args.synthetic
        and card["primary_metric"]["sample_count"] < MIN_JUDGE_SAMPLES
    )

    card["generated_at"] = datetime.now(tz=timezone.utc).isoformat()

    if args.json:
        print(json.dumps(card, ensure_ascii=False, indent=2))
    else:
        _print_scorecard(card)

    if degraded:
        print("[降级] 真实 holdout 决策样本不足（<{}），裁决指标不可信；"
              "补跑合成管道校验以证明台架逻辑正确 ↓".format(MIN_JUDGE_SAMPLES))
        syn = run_scorecard(
            synthetic_windows(),
            holdout_ratio=holdout_ratio,
            n_clusters=n_clusters,
            match_threshold=args.threshold,
            seed=args.seed,
            feature_version=args.variant + "-synthetic",
            decisive=False,
        )
        if not args.json:
            _print_scorecard(syn)

    # 对比 baseline（后续 A/B/C 轮次用）
    if args.compare:
        try:
            with open(args.compare, encoding="utf-8") as f:
                baseline = json.load(f)
            _diff_scorecards(baseline, card)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 读取 baseline 失败: {type(exc).__name__}: {exc}", file=sys.stderr)

    # 写快照
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)
        print(f"[已写入] {args.out}")

    if decisive and card["primary_metric"]["sample_count"] < MIN_JUDGE_SAMPLES:
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LEARN 环③ 特征表征离线评估台架")
    parser.add_argument("--days-back", type=int, default=None,
                        help="仅统计近 N 天窗口；缺省=全量历史")
    parser.add_argument("--limit", type=int, default=None, help="窗口数上限（调试用）")
    parser.add_argument("--from-file", default=None,
                        help="从服务器导出的 JSON/JSONL 文件取数（DB 不可直连时的桥接方案）")
    parser.add_argument("--holdout-ratio", type=float, default=None,
                        help="holdout 比例，缺省取 settings.agent_deep_learn_holdout_ratio")
    parser.add_argument("--clusters", type=int, default=None,
                        help="KMeans 目标簇数，缺省取 settings.agent_deep_learn_target_clusters")
    parser.add_argument("--threshold", type=float, default=HOLDOUT_MATCH_THRESHOLD,
                        help=f"holdout 匹配相似度阈值（默认 {HOLDOUT_MATCH_THRESHOLD}）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="KMeans random_state")
    parser.add_argument("--standardize", action="store_true",
                        help="优化点 A：相似度改用 train z-score 标准化几何（与 KMeans 空间一致）")
    parser.add_argument("--knn", type=int, default=None,
                        help="优化点 A3：裁决改用全局 k 近邻多数投票（替代质心匹配）")
    parser.add_argument("--variant", default=DEFAULT_FEATURE_VERSION,
                        help="变体标签，写入指纹用于 A/B/C 溯源")
    parser.add_argument("--out", default="output/feature_bench_baseline.json",
                        help="评分卡 JSON 输出路径（空字符串则不写）")
    parser.add_argument("--compare", default=None, help="与指定 baseline JSON 逐字段 diff")
    parser.add_argument("--truncate-sec", type=float, default=None,
                        help="决策点截断：只用开窗后前 N 秒的曲线预测整窗 outcome（剔除答案读取）")
    parser.add_argument("--placebo", action="store_true",
                        help="标签置换安慰剂：随机打乱 outcome 后重跑，验证信号来自真实形态-结果关系")
    parser.add_argument("--synthetic", action="store_true",
                        help="强制用合成数据（仅验证台架管道）")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON 评分卡")
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
