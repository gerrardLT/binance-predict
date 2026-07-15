"""Deep Learn 样本外校验内核（P0-3 准入闸门与双轨对比复用）。

提供 Wilson 置信下界、按时间的 train/holdout 切分、以及用确定性特征相似度
将候选模式在 holdout 窗口上回测胜率。全程无 LLM，同一输入必得同一输出。
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .curve_features import cosine_sim, extract_features

# holdout 匹配相似度阈值：cosine_sim >= 此值才算某 holdout 窗口"触发"该模式形态
HOLDOUT_MATCH_THRESHOLD = 0.8


def snapshot_token(window_ids) -> str:
    """对参与本次发现的窗口 id 集合生成稳定指纹（P2-2 预览/提交一致性校验）。

    对 id 去重排序后取 sha256 前 16 位；同一窗口集合必得同一 token。
    """
    ids = sorted({int(i) for i in window_ids})
    raw = ",".join(str(i) for i in ids)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    """二项比例的 Wilson 置信下界（默认 95%）。total<=0 返回 0.0。

    用于把"少量样本上的高胜率"折价——样本越少，下界越低，天然抑制过拟合。
    """
    if total <= 0:
        return 0.0
    correct = max(0, min(int(correct), int(total)))
    n = float(total)
    phat = correct / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def time_split(windows: list[dict], holdout_ratio: float) -> tuple[list[dict], list[dict]]:
    """按 start_time 升序排序，切出最新 holdout_ratio 比例作 holdout，其余为 train。

    返回 (train_windows, holdout_windows)。窗口不足 2 条或算得 holdout 为 0 时
    holdout 为空、train 为全部。holdout_ratio 夹在 [0, 0.9]。
    """
    if not windows:
        return [], []
    ordered = sorted(windows, key=lambda w: w.get("start_time", 0))
    n = len(ordered)
    if n < 2:
        return ordered, []
    ratio = min(max(float(holdout_ratio), 0.0), 0.9)
    n_holdout = max(0, min(int(n * ratio), n - 1))
    if n_holdout == 0:
        return ordered, []
    return ordered[: n - n_holdout], ordered[n - n_holdout:]


def evaluate_on_holdout(
    pattern_features,
    direction: str,
    holdout_windows: list[dict],
    match_threshold: float = HOLDOUT_MATCH_THRESHOLD,
) -> dict:
    """用特征相似度将模式匹配 holdout 窗口，按 direction vs outcome 判命中。

    对每个 holdout 窗口提特征并与 pattern_features 算 cosine_sim，>= 阈值视为
    该窗口触发本模式；触发窗口中 outcome==direction 记命中（NOISE 计入分母不计命中）。
    返回 {win_rate, sample_count, ci_lower}；无触发窗口时 sample_count=0、其余 0.0。
    """
    pf = np.asarray(pattern_features, dtype=float)
    matched = 0
    correct = 0
    for w in holdout_windows:
        feat = extract_features(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
        if cosine_sim(pf, feat) >= match_threshold:
            matched += 1
            if (w.get("outcome") or "").upper() == direction.upper():
                correct += 1
    win_rate = (correct / matched) if matched > 0 else 0.0
    ci_lower = wilson_lower_bound(correct, matched)
    return {"win_rate": win_rate, "sample_count": matched, "ci_lower": ci_lower}
