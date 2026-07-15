"""Deep Learn 确定性曲线特征内核（P0-2 基座，被 P0-3 holdout 匹配 / P1-3 去重复用）。

全程无 LLM：从情绪窗口的 UP%/DOWN% 时间序列提取固定维度、可复现的特征向量，
供 KMeans 聚类、样本外 holdout 相似度匹配与模式去重共用。同一输入必得同一输出。

曲线数据结构对齐 SentimentWindow.curve_up_pct/curve_down_pct：list[{t, v}]，
v 为该采样点的 UP% / DOWN% 数值（百分比，如 55.0）。
"""
from __future__ import annotations

import numpy as np

# 特征向量维度（固定，供调用方断言）：up 10 + down 10 + 交叉 4 = 24
FEATURE_DIM = 24


def _series_values(curve) -> np.ndarray:
    """从 [{t, v}, ...] 曲线抽取 v 序列为 float ndarray；空/非法点跳过。"""
    if not curve:
        return np.array([], dtype=float)
    vals: list[float] = []
    for p in curve:
        v = p.get("v") if isinstance(p, dict) else p
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return np.array(vals, dtype=float)


def _series_features(v: np.ndarray) -> list[float]:
    """单条序列的 10 维确定性特征。空序列全 0。

    维度：首值 / 末值 / 净变化 / 均值 / 标准差 / 最小 / 最大 / 斜率 /
    极大值归一化位置 / 极小值归一化位置。
    """
    n = len(v)
    if n == 0:
        return [0.0] * 10
    first = float(v[0])
    last = float(v[-1])
    net = last - first
    mean = float(np.mean(v))
    std = float(np.std(v))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if n >= 2:
        x = np.linspace(0.0, 1.0, n)
        slope = float(np.polyfit(x, v, 1)[0])
        argmax_pos = float(np.argmax(v)) / (n - 1)
        argmin_pos = float(np.argmin(v)) / (n - 1)
    else:
        slope = 0.0
        argmax_pos = 0.0
        argmin_pos = 0.0
    return [first, last, net, mean, std, vmin, vmax, slope, argmax_pos, argmin_pos]


def extract_features(curve_up, curve_down) -> np.ndarray:
    """从 UP%/DOWN% 曲线提取固定维度（FEATURE_DIM）确定性特征向量。

    构成：up 10 维 + down 10 维 + 交叉 4 维（末段 up-down 差、diff 均值、
    diff 符号变化次数=交叉次数、末段 diff 符号）。同输入必得同输出。
    """
    up = _series_values(curve_up)
    down = _series_values(curve_down)
    feats = _series_features(up) + _series_features(down)

    m = min(len(up), len(down))
    if m >= 1:
        diff = up[:m] - down[:m]
        last_diff = float(diff[-1])
        mean_diff = float(np.mean(diff))
        if m >= 2:
            signs = np.sign(diff)
            cross_count = float(np.sum(signs[1:] * signs[:-1] < 0))
        else:
            cross_count = 0.0
        last_sign = float(np.sign(last_diff))
    else:
        last_diff = mean_diff = cross_count = last_sign = 0.0
    feats += [last_diff, mean_diff, cross_count, last_sign]
    return np.array(feats, dtype=float)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两特征向量余弦相似度，范围 [-1,1]；任一零向量返回 0.0。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_windows(
    feature_matrix: np.ndarray, n_clusters: int, random_state: int = 42
) -> np.ndarray:
    """对特征矩阵做 KMeans 聚类，返回每行簇标签（int ndarray）。

    实际 k = max(1, min(n_clusters, 窗口数))；特征先 StandardScaler 标准化避免
    量纲主导。random_state 固定保证可复现。窗口数<=1 或 k<=1 时全部归簇 0。
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(feature_matrix, dtype=float)
    n = X.shape[0]
    k = max(1, min(int(n_clusters), n))
    if n <= 1 or k <= 1:
        return np.zeros(n, dtype=int)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(Xs)
