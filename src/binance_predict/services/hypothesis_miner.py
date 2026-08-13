"""科学发现系统 —— 假设矿机（程序预筛轨，PY_CLUSTER 发现轨的谓词化实现）。

定位：LLM 假设生成的「侦察兵」。V1 谓词空间有限（约 300 个组合），程序
在 train 集上穷举执行 + 局部基准 lift 统计，把统计异常的谓词编成线索
榜单喂给 LLM——LLM 从「肉眼找规律」变为「按榜单精选」，每条猜想自带
数据出处（rationale 引用榜单编号），解决纯直觉猜想数量少、无依据的问题。

统计口径与 Q6 审判完全一致（同一 evaluate_predicate / pooled_local_baseline /
lift_test），唯一差异：矿机跑在 train 集（LLM 可见），审判跑在 holdout。
榜单仅作发现线索排序，不构成任何统计结论——最终裁决权永远在 holdout 审判。

全程无 LLM、无 DB、纯函数：同一输入必得同一输出。
"""

from __future__ import annotations

from itertools import permutations, product

from binance_predict.services.predicates import (
    ALLOWED_SEGMENTS,
    evaluate_predicate,
    validate_predicate,
)
from binance_predict.services.symbolizer import SYMBOLS, WindowView
from binance_predict.services.verification import lift_test, pooled_local_baseline

# --- 枚举空间参数（V1 白名单内的保守子集，控制总量 ~300） ---
_CHANNELS = ("sentiment", "price", "volume")
_COUNT_CMPS = (">=", "<=")  # >= 捕捉密集，<= 捕捉稀缺
_COUNT_VALUES = (1, 2, 3, 4)
_PEAK_CMPS = (">=", "==")
_PEAK_VALUES = (1, 2, 3, 4)
_SPACING_TRENDS = ("shrinking", "expanding", "mixed")
_SYNC_VALUES = (0.5, 0.6, 0.7)
_LEAD_KS = (1, 2)
_LEAD_MIN_MATCHES = 1

# 榜单默认规模与入选门槛（min_hits 与 Q6 初筛 MIN_SCREEN_HITS=5 对齐）
DEFAULT_MAX_HINTS = 60
DEFAULT_MIN_HITS = 5


def enumerate_predicates() -> list[dict]:
    """穷举 V1 谓词空间（单谓词，无逻辑组合），返回合法谓词 JSON 列表。

    空间规模：symbol_at 45 + count_symbol 120 + has_subseq 75 +
    peak_count 24 + extremum_spacing 9 + sync 9 + lead 12 ≈ 294。
    组合谓词（AND/OR/NOT）不在穷举范围——那是 LLM 基于榜单的增量价值。
    """
    predicates: list[dict] = []

    for channel, segment, symbol in product(_CHANNELS, ALLOWED_SEGMENTS, SYMBOLS):
        predicates.append({
            "pred": "symbol_at", "channel": channel,
            "segment": segment, "symbol": symbol,
        })

    for channel, symbol, cmp_op, value in product(
        _CHANNELS, SYMBOLS, _COUNT_CMPS, _COUNT_VALUES
    ):
        predicates.append({
            "pred": "count_symbol", "channel": channel, "symbol": symbol,
            "cmp": cmp_op, "value": value,
        })

    for channel, sym_a, sym_b in product(_CHANNELS, SYMBOLS, SYMBOLS):
        predicates.append({
            "pred": "has_subseq", "channel": channel, "symbols": [sym_a, sym_b],
        })

    for channel, cmp_op, value in product(_CHANNELS, _PEAK_CMPS, _PEAK_VALUES):
        predicates.append({
            "pred": "peak_count", "channel": channel, "cmp": cmp_op, "value": value,
        })

    for channel, trend in product(_CHANNELS, _SPACING_TRENDS):
        predicates.append({
            "pred": "extremum_spacing", "channel": channel, "trend": trend,
        })

    channel_pairs = [(a, b) for a, b in permutations(_CHANNELS, 2)]
    for (ch_a, ch_b), value in product(
        [(a, b) for a, b in channel_pairs if a < b], _SYNC_VALUES
    ):
        predicates.append({
            "pred": "sync", "channel_a": ch_a, "channel_b": ch_b,
            "cmp": ">=", "value": value,
        })

    for (ch_a, ch_b), k in product(channel_pairs, _LEAD_KS):
        predicates.append({
            "pred": "lead", "channel_a": ch_a, "channel_b": ch_b,
            "k": k, "min_matches": _LEAD_MIN_MATCHES,
        })

    # 枚举器自检：穷举产物必须全部通过 Q5 白名单校验（防御未来枚举参数越界）
    for p in predicates:
        validate_predicate(p)
    return predicates


def mine_hints(
    views: list[WindowView],
    max_hints: int = DEFAULT_MAX_HINTS,
    min_hits: int = DEFAULT_MIN_HITS,
) -> list[dict]:
    """在 train 符号化视图上穷举谓词，输出按偏向强度排序的线索榜单。

    Args:
        views: 训练集符号化窗口视图（须带 start_time 与 outcome）
        max_hints: 榜单条数上限
        min_hits: 入选最小命中数（与 Q6 初筛门槛对齐，低于此不做统计）

    Returns:
        线索榜单（按 lift 降序），每条含：
        - predicate: 谓词 JSON（可直接作为假设的 predicate）
        - direction: "UP" | "DOWN"（命中窗口的偏向方向，取两方向中 lift 较大者）
        - hits: 命中窗口数
        - up_hits / down_hits / noise_hits: 命中窗口的 outcome 分布
        - lift: 命中组偏向率 / 局部基准偏向率（与审判同口径，但跑在 train）
        - ci_lower: lift 置信区间下界
    """
    pool_windows = [
        {"start_time": v.start_time, "outcome": v.outcome or ""} for v in views
    ]

    hints: list[dict] = []
    for predicate in enumerate_predicates():
        hit_views = [v for v in views if evaluate_predicate(predicate, v)]
        if len(hit_views) < min_hits:
            continue

        up_hits = sum(1 for v in hit_views if (v.outcome or "").upper() == "UP")
        down_hits = sum(1 for v in hit_views if (v.outcome or "").upper() == "DOWN")
        noise_hits = len(hit_views) - up_hits - down_hits
        hit_times = [v.start_time for v in hit_views]

        # 两方向各算一次 lift（与审判同口径：局部基准对照池），取偏向强者
        best: tuple[str, float, float] | None = None  # (direction, lift, ci_lower)
        for target, target_hits in (("UP", up_hits), ("DOWN", down_hits)):
            base_events, base_total = pooled_local_baseline(
                hit_times, pool_windows, target
            )
            lr = lift_test(target_hits, len(hit_views), base_events, base_total)
            if best is None or lr.lift > best[1]:
                best = (target, lr.lift, lr.ci_lower)

        assert best is not None  # 两方向循环必然产生结果
        hints.append({
            "predicate": predicate,
            "direction": best[0],
            "hits": len(hit_views),
            "up_hits": up_hits,
            "down_hits": down_hits,
            "noise_hits": noise_hits,
            "lift": round(best[1], 3),
            "ci_lower": round(best[2], 3),
        })

    # 偏向强度降序；同分时命中多者优先（更稳的统计基础）
    hints.sort(key=lambda h: (h["lift"], h["hits"]), reverse=True)
    return hints[:max_hints]
