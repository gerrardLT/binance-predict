"""假设矿机（hypothesis_miner）单元测试。

覆盖：
- enumerate_predicates：枚举空间规模与 Q5 白名单合法性自检
- mine_hints：注入强信号的合成数据上榜单排序正确 / min_hits 过滤 /
  缺通道安全 / max_hints 截断 / 纯函数确定性
"""

from __future__ import annotations

import pytest

from binance_predict.services.hypothesis_miner import (
    enumerate_predicates,
    mine_hints,
)
from binance_predict.services.predicates import validate_predicate
from binance_predict.services.symbolizer import ChannelView, WindowView

_BASE_TS = 1_780_000_000_000  # 合成起始毫秒时间戳
_STEP_MS = 300_000  # 5 分钟一个窗口


def _make_view(
    idx: int,
    outcome: str,
    sentiment_symbols: list[str],
    with_price: bool = True,
) -> WindowView:
    """构造合成窗口视图（geometry 给默认值，仅符号串驱动谓词）。"""
    geo = {
        "peak_count": 1,
        "extremum_spacing": "mixed",
        "area_ratio": 0.5,
        "curliness": 1.0,
    }
    channels = {
        "sentiment": ChannelView(
            symbols=list(sentiment_symbols), geometry=dict(geo),
            point_count=len(sentiment_symbols) + 1,
        ),
    }
    if with_price:
        channels["price"] = ChannelView(
            symbols=["平"] * len(sentiment_symbols), geometry=dict(geo),
            point_count=len(sentiment_symbols) + 1,
        )
    return WindowView(
        start_time=_BASE_TS + idx * _STEP_MS,
        window_id=idx + 1,
        outcome=outcome,
        channels=channels,
    )


def _signal_dataset() -> list[WindowView]:
    """注入强信号的合成数据集（UP/DOWN 混合，避免对照池零事件触发 lift 退化）：

    - 组 A（i 0-19）sentiment 含连续三个「缓降」：18 DOWN + 2 UP（DOWN 偏向 90%）
    - 组 B（i 20-39）sentiment 为「平/缓升」交替：8 DOWN + 12 UP（UP 偏向 60%）
    """
    views: list[WindowView] = []
    for i in range(20):
        outcome = "DOWN" if i < 18 else "UP"
        views.append(_make_view(i, outcome, ["缓降", "缓降", "缓降", "平"]))
    for i in range(20, 40):
        outcome = "DOWN" if i < 28 else "UP"
        views.append(_make_view(i, outcome, ["平", "缓升", "平", "缓升"]))
    return views


# ============================================================
# enumerate_predicates
# ============================================================

class TestEnumeratePredicates:
    def test_all_predicates_pass_whitelist_validation(self):
        predicates = enumerate_predicates()
        assert predicates, "枚举空间不应为空"
        for p in predicates:
            validate_predicate(p)  # 不抛异常即合法

    def test_space_size_matches_design(self):
        # symbol_at 45 + count_symbol 120 + has_subseq 75 +
        # peak_count 24 + extremum_spacing 9 + sync 9 + lead 12 = 294
        assert len(enumerate_predicates()) == 294

    def test_no_duplicates(self):
        import json
        seen = {
            json.dumps(p, sort_keys=True, ensure_ascii=False)
            for p in enumerate_predicates()
        }
        assert len(seen) == len(enumerate_predicates())


# ============================================================
# mine_hints
# ============================================================

class TestMineHints:
    def test_injected_signal_ranks_top(self):
        """注入的「缓降密集→DOWN」信号应被挖出：方向正确、统计准确、偏向显著。"""
        hints = mine_hints(_signal_dataset())
        assert hints, "合成强信号数据集不应产出空榜单"

        # 榜首有真实偏向（不指定方向——UP/DOWN 两信号谁强取决于数据配比）
        assert hints[0]["lift"] > 1.5

        # 注入的缓降 DOWN 信号必在榜单中，且统计与数据集事实一致
        dip_hints = [
            h for h in hints
            if h["direction"] == "DOWN" and "缓降" in str(h["predicate"])
        ]
        assert dip_hints, "注入的缓降 DOWN 信号应被挖出"
        best_dip = dip_hints[0]
        assert best_dip["hits"] == 20       # 命中全部组 A 窗口
        assert best_dip["down_hits"] == 18
        assert best_dip["up_hits"] == 2
        assert best_dip["lift"] > 1.5       # 0.9 / 0.4 = 2.25

    def test_hints_sorted_by_lift_descending(self):
        hints = mine_hints(_signal_dataset())
        lifts = [h["lift"] for h in hints]
        assert lifts == sorted(lifts, reverse=True)

    def test_min_hits_filters_rare_predicates(self):
        """命中数 < min_hits 的谓词不得上榜。"""
        views = _signal_dataset()
        # 追加 3 个特殊窗口：只有它们含「急升」（命中数 3 < 5）
        for i in range(40, 43):
            views.append(_make_view(i, "UP", ["急升", "平", "平", "平"]))

        hints = mine_hints(views, min_hits=5)
        for h in hints:
            assert h["hits"] >= 5
            # 「急升密集」类谓词（symbol_at / has_subseq / count>=）最多命中 3 次，
            # 不得上榜；「急升稀缺」（count<=）命中面天然大，上榜合法
            pred = h["predicate"]
            if "急升" in str(pred):
                assert pred["pred"] == "count_symbol" and pred["cmp"] == "<="

    def test_missing_channel_is_safe(self):
        """缺 price/volume 通道的视图不应导致异常，跨通道谓词返回 False。"""
        views = [_make_view(i, "UP", ["平", "缓升"], with_price=False) for i in range(10)]
        views += [_make_view(i, "DOWN", ["缓降", "平"], with_price=False) for i in range(10, 20)]
        hints = mine_hints(views)  # 不抛异常即可
        # 跨通道谓词（lead/sync）在此数据集一律不命中，不应上榜
        for h in hints:
            assert h["predicate"]["pred"] not in ("lead", "sync")

    def test_max_hints_truncates(self):
        hints = mine_hints(_signal_dataset(), max_hints=5)
        assert len(hints) <= 5

    def test_deterministic_same_input_same_output(self):
        views = _signal_dataset()
        assert mine_hints(views) == mine_hints(views)

    def test_empty_views_returns_empty(self):
        assert mine_hints([]) == []

    def test_hint_fields_complete(self):
        hints = mine_hints(_signal_dataset())
        required = {
            "predicate", "direction", "hits",
            "up_hits", "down_hits", "noise_hits", "lift", "ci_lower",
        }
        for h in hints:
            assert required <= set(h.keys())
            assert h["direction"] in ("UP", "DOWN")
