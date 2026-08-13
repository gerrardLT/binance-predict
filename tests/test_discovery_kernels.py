"""科学发现系统 Phase 1 内核测试（宪法 Q1/Q4/Q5/Q6/Q7）。

覆盖三个纯函数内核：
- services/symbolizer.py  分位数分箱 / 冻结快照 / 符号化 / 几何摘要 / 窗口视图
- services/predicates.py  L1/L2 谓词语义 + DSL 校验器
- services/verification.py Wilson 区间 / 局部基准 lift / 三段式分类 / BH-FDR / 死因判定
"""

from __future__ import annotations

import pytest

from binance_predict.services.symbolizer import (
    BinningSnapshot,
    build_window_view,
    compute_bin_edges,
    geometric_summary,
    should_freeze,
    symbolize_delta,
    symbolize_series,
)
from binance_predict.services.predicates import (
    evaluate_predicate,
    validate_predicate,
)
from binance_predict.services.verification import (
    DEATH_ALIVE,
    DEATH_EXPIRED,
    DEATH_SPURIOUS,
    LiftResult,
    bh_fdr,
    classify_candidate,
    diagnose_death,
    lift_test,
    pooled_local_baseline,
    wilson_bounds,
)

# ============================================================
# 共用构造
# ============================================================

# 固定分箱边界：(-2.0, -0.5, 0.5, 2.0)
SNAP = BinningSnapshot(
    version="test-v1",
    edges=(-2.0, -0.5, 0.5, 2.0),
    created_at_epoch=1_000_000.0,
    sample_count=100,
)


def _curve(values: list[float]) -> list[dict]:
    return [{"t": i, "v": v} for i, v in enumerate(values)]


# ============================================================
# symbolizer：分箱与快照（Q4）
# ============================================================


class TestBinning:
    def test_quantile_edges_linear_interpolation(self):
        # 0..99 的 20/40/60/80 分位（线性插值）
        snap = compute_bin_edges([float(x) for x in range(100)], version="v")
        assert snap.edges == pytest.approx((19.8, 39.6, 59.4, 79.2))
        assert snap.sample_count == 100

    def test_insufficient_samples_rejected(self):
        with pytest.raises(ValueError, match="样本不足"):
            compute_bin_edges([1.0] * 9, version="v")

    def test_symbolize_delta_five_bins(self):
        assert symbolize_delta(-3.0, SNAP) == "急降"   # < q20
        assert symbolize_delta(-1.0, SNAP) == "缓降"   # [q20, q40)
        assert symbolize_delta(0.0, SNAP) == "平"      # [q40, q60)
        assert symbolize_delta(1.0, SNAP) == "缓升"    # [q60, q80)
        assert symbolize_delta(3.0, SNAP) == "急升"    # >= q80

    def test_symbolize_delta_boundary_belongs_to_upper_bin(self):
        # 边界值归入上一档（delta < q 才降档）
        assert symbolize_delta(-0.5, SNAP) == "平"
        assert symbolize_delta(2.0, SNAP) == "急升"

    def test_symbolize_series_length_is_n_minus_1(self):
        symbols = symbolize_series(_curve([50.0, 51.0, 52.0, 52.0]), SNAP)
        # deltas = [1, 1, 0] → 缓升、缓升、平
        assert symbols == ["缓升", "缓升", "平"]

    def test_snapshot_serialization_roundtrip(self):
        restored = BinningSnapshot.from_dict(SNAP.to_dict())
        assert restored == SNAP

    def test_should_freeze(self):
        assert should_freeze(None) is True
        # 30 天内不冻结
        assert should_freeze(SNAP, now_epoch=1_000_000.0 + 10 * 86_400) is False
        # 超过 30 天冻结
        assert should_freeze(SNAP, now_epoch=1_000_000.0 + 31 * 86_400) is True


# ============================================================
# symbolizer：几何摘要（Q1 辅助通道）
# ============================================================


class TestGeometricSummary:
    def test_straight_line(self):
        g = geometric_summary(_curve([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert g["peak_count"] == 0
        assert g["area_ratio"] == 0.5  # 与首末连线重合
        assert g["curliness"] == pytest.approx(1.0)  # 直线卷曲度为 1
        assert g["extremum_spacing"] == "insufficient"

    def test_single_peak(self):
        g = geometric_summary(_curve([1.0, 3.0, 5.0, 3.0, 1.0]))
        assert g["peak_count"] == 1
        assert g["extrema"][0]["kind"] == "peak"
        assert g["extrema"][0]["pos"] == pytest.approx(0.5)
        assert g["area_ratio"] == pytest.approx(1.0)  # 完全凸起于首末连线之上
        assert g["curliness"] == pytest.approx(8.0)  # 净位移 0 → 封顶为总变差

    def test_empty_curve_is_safe(self):
        g = geometric_summary([])
        assert g["peak_count"] == 0
        assert g["curliness"] == 1.0


# ============================================================
# symbolizer：窗口视图（Q2 三通道）
# ============================================================


class TestWindowView:
    def test_three_channels_and_missing_volume_skipped(self):
        window = {
            "id": 7,
            "start_time": 12345,
            "outcome": "UP",
            "curve_up_pct": _curve([50.0, 51.0, 52.0]),      # → 缓升,缓升
            "curve_btc_price": _curve([100.0, 99.0, 98.0]),  # → 缓降,缓降
            "curve_trade_volume": [],                        # 缺数据 → 跳过
        }
        view = build_window_view(window, SNAP)
        assert view.window_id == 7
        assert view.has_channel("sentiment")
        assert view.has_channel("price")
        assert not view.has_channel("volume")  # 空通道被跳过
        assert view.channels["sentiment"].symbols == ["缓升", "缓升"]
        assert view.channels["price"].symbols == ["缓降", "缓降"]


# ============================================================
# predicates：L1 谓词语义
# ============================================================


def _view(symbols: list[str], channel: str = "sentiment", geometry: dict | None = None):
    from binance_predict.services.symbolizer import ChannelView, WindowView

    return WindowView(
        start_time=0,
        channels={
            channel: ChannelView(
                symbols=symbols,
                geometry=geometry or {"peak_count": 0, "extremum_spacing": "mixed"},
                point_count=len(symbols) + 1,
            )
        },
    )


class TestL1Predicates:
    def test_has_subseq(self):
        view = _view(["平", "急升", "平", "急降"])
        assert evaluate_predicate(
            {"pred": "has_subseq", "channel": "sentiment", "symbols": ["急升", "平"]},
            view,
        )
        assert not evaluate_predicate(
            {"pred": "has_subseq", "channel": "sentiment", "symbols": ["急降", "急升"]},
            view,
        )

    def test_symbol_at_segments(self):
        # 6 符号：early=[0,2) mid=[2,4) late=[4,6)
        view = _view(["急升", "急升", "平", "平", "急降", "急降"])
        assert evaluate_predicate(
            {"pred": "symbol_at", "channel": "sentiment", "segment": "early", "symbol": "急升"},
            view,
        )
        assert evaluate_predicate(
            {"pred": "symbol_at", "channel": "sentiment", "segment": "late", "symbol": "急降"},
            view,
        )
        assert not evaluate_predicate(
            {"pred": "symbol_at", "channel": "sentiment", "segment": "early", "symbol": "急降"},
            view,
        )

    def test_count_symbol(self):
        view = _view(["急升", "平", "急升", "平"])
        assert evaluate_predicate(
            {"pred": "count_symbol", "channel": "sentiment", "symbol": "急升", "cmp": ">=", "value": 2},
            view,
        )
        assert not evaluate_predicate(
            {"pred": "count_symbol", "channel": "sentiment", "symbol": "急升", "cmp": "==", "value": 3},
            view,
        )

    def test_peak_count_uses_geometry(self):
        view = _view(["平"], geometry={"peak_count": 2, "extremum_spacing": "shrinking"})
        assert evaluate_predicate(
            {"pred": "peak_count", "channel": "sentiment", "cmp": "==", "value": 2},
            view,
        )

    def test_extremum_spacing(self):
        view = _view(["平"], geometry={"peak_count": 0, "extremum_spacing": "shrinking"})
        assert evaluate_predicate(
            {"pred": "extremum_spacing", "channel": "sentiment", "trend": "shrinking"},
            view,
        )
        assert not evaluate_predicate(
            {"pred": "extremum_spacing", "channel": "sentiment", "trend": "expanding"},
            view,
        )

    def test_missing_channel_returns_false(self):
        view = _view(["急升"])  # 只有 sentiment
        assert not evaluate_predicate(
            {"pred": "has_subseq", "channel": "volume", "symbols": ["急升"]},
            view,
        )


# ============================================================
# predicates：L2 关系谓词（Q2 变量间结构）
# ============================================================


class TestL2Predicates:
    def test_lead_with_tolerance(self):
        from binance_predict.services.symbolizer import ChannelView, WindowView

        # A 转移点 {1,2}；B 转移点 {2,3}；k=1 → 每个 A 转移点 +1±1 均有 B 对位
        view = WindowView(
            start_time=0,
            channels={
                "sentiment": ChannelView(["平", "急升", "平"], {}, 4),
                "price": ChannelView(["平", "平", "急升", "平"], {}, 5),
            },
        )
        assert evaluate_predicate(
            {"pred": "lead", "channel_a": "sentiment", "channel_b": "price", "k": 1, "min_matches": 2},
            view,
        )
        # k=3 超出对位容差 → False
        assert not evaluate_predicate(
            {"pred": "lead", "channel_a": "sentiment", "channel_b": "price", "k": 3, "min_matches": 2},
            view,
        )

    def test_sync_direction_class(self):
        from binance_predict.services.symbolizer import ChannelView, WindowView

        # A: U,U,F,D；B: U,U,D,D → 同步率 3/4=0.75
        view = WindowView(
            start_time=0,
            channels={
                "sentiment": ChannelView(["急升", "缓升", "平", "急降"], {}, 5),
                "price": ChannelView(["缓升", "急升", "缓降", "缓降"], {}, 5),
            },
        )
        assert evaluate_predicate(
            {"pred": "sync", "channel_a": "sentiment", "channel_b": "price", "cmp": ">=", "value": 0.7},
            view,
        )
        assert not evaluate_predicate(
            {"pred": "sync", "channel_a": "sentiment", "channel_b": "price", "cmp": ">=", "value": 0.9},
            view,
        )


# ============================================================
# predicates：DSL 组合与校验（Q5 自由度约束）
# ============================================================


class TestDslValidation:
    def test_logic_combination(self):
        view = _view(["急升", "平", "急降"])
        node = {
            "op": "AND",
            "args": [
                {"pred": "has_subseq", "channel": "sentiment", "symbols": ["急升", "平"]},
                {"op": "NOT", "arg": {"pred": "symbol_at", "channel": "sentiment", "segment": "late", "symbol": "急升"}},
            ],
        }
        assert evaluate_predicate(node, view)

    def test_reject_unknown_predicate(self):
        with pytest.raises(ValueError, match="白名单"):
            validate_predicate({"pred": "fft_energy", "channel": "sentiment"})

    def test_reject_depth_over_limit(self):
        # 3 层逻辑嵌套超过 MAX_LOGIC_DEPTH=2
        node = {"op": "AND", "args": [
            {"op": "AND", "args": [
                {"op": "OR", "args": [
                    {"pred": "has_subseq", "channel": "sentiment", "symbols": ["平"]},
                ]},
            ]},
        ]}
        with pytest.raises(ValueError, match="深度超限"):
            validate_predicate(node)

    def test_reject_illegal_channel(self):
        with pytest.raises(ValueError, match="非法通道"):
            validate_predicate(
                {"pred": "has_subseq", "channel": "funding_rate", "symbols": ["平"]}
            )

    def test_reject_non_enum_k(self):
        with pytest.raises(ValueError, match="枚举"):
            validate_predicate(
                {"pred": "lead", "channel_a": "sentiment", "channel_b": "price", "k": 5, "min_matches": 1}
            )

    def test_reject_same_channels(self):
        with pytest.raises(ValueError, match="不能相同"):
            validate_predicate(
                {"pred": "sync", "channel_a": "price", "channel_b": "price", "cmp": ">=", "value": 0.7}
            )


# ============================================================
# verification：Wilson 与 lift（Q6）
# ============================================================


class TestWilsonAndLift:
    def test_wilson_bounds_contain_point_estimate(self):
        lo, hi = wilson_bounds(60, 100)
        assert lo < 0.6 < hi
        assert wilson_bounds(0, 0) == (0.0, 0.0)

    def test_lift_unity_when_groups_identical(self):
        # 命中组与对照池同分布 → lift=1，CI 必覆盖 1
        r = lift_test(50, 100, 50, 100)
        assert r.lift == pytest.approx(1.0)
        assert r.ci_lower < 1.0 < r.ci_upper
        assert r.p_value == pytest.approx(0.5)

    def test_lift_strong_signal(self):
        # 8/10 vs 40/100 → lift=2.0，log-lift SE=0.2，CI 下界 ≈1.35 > 1
        r = lift_test(8, 10, 40, 100)
        assert r.lift == pytest.approx(2.0)
        assert r.ci_lower > 1.0
        assert r.p_value < 0.001

    def test_lift_degenerate_is_conservative(self):
        # 零事件绝不产生假显著
        r = lift_test(0, 10, 40, 100)
        assert r.lift == 0.0
        assert r.p_value == 1.0


class TestLocalBaseline:
    def test_pool_excludes_hits_and_dedupes(self):
        day = 86_400_000
        windows = [
            {"start_time": 0 * day, "outcome": "UP"},
            {"start_time": 1 * day, "outcome": "DOWN"},
            {"start_time": 2 * day, "outcome": "UP"},    # 命中窗口（自身排除）
            {"start_time": 3 * day, "outcome": "DOWN"},
            {"start_time": 4 * day, "outcome": "UP"},
            {"start_time": 10 * day, "outcome": "UP"},   # 超出 ±3 天，不进池
        ]
        events, total = pooled_local_baseline([2 * day], windows, "UP")
        # 池 = {0,1,3,4}（去重、排命中、排超界），其中 UP 为 {0,4}
        assert total == 4
        assert events == 2


# ============================================================
# verification：三段式分类 / BH-FDR / 死因（Q6/Q7）
# ============================================================


def _lift(lift: float, ci_lower: float, ci_upper: float) -> LiftResult:
    return LiftResult(
        lift=lift, ci_lower=ci_lower, ci_upper=ci_upper,
        p_value=0.01, hit_events=10, hit_total=10,
        base_events=10, base_total=20,
    )


class TestClassifyAndFdr:
    def test_classify_active(self):
        assert classify_candidate(_lift(2.0, 1.35, 3.0)) == "ACTIVE"

    def test_classify_observe(self):
        assert classify_candidate(_lift(1.25, 0.95, 1.6)) == "OBSERVE"

    def test_classify_reject(self):
        assert classify_candidate(_lift(1.05, 0.9, 1.2)) == "REJECT"

    def test_bh_fdr_all_significant(self):
        assert bh_fdr([0.001, 0.002, 0.003], q=0.1) == [True, True, True]

    def test_bh_fdr_partial(self):
        # rank1: 0.01 <= 1/3*0.1 ✓；rank2: 0.5 <= 2/3*0.1 ✗ → 仅第一个通过
        assert bh_fdr([0.01, 0.5, 0.9], q=0.1) == [True, False, False]

    def test_bh_fdr_empty(self):
        assert bh_fdr([]) == []


class TestDiagnoseDeath:
    def test_insufficient_hits_stays_alive(self):
        assert diagnose_death(1.0, 0.8, 1.2, peak_lift=1.0, hit_count=10) == DEATH_ALIVE

    def test_expired_when_decayed_after_significance(self):
        # 曾显著（峰值 1.5 ≥1.3）且近期衰减（1.05 <1.1）→ 过期规律
        assert diagnose_death(1.05, 0.9, 1.2, peak_lift=1.5, hit_count=30) == DEATH_EXPIRED

    def test_spurious_when_never_significant(self):
        # 从未显著（峰值 1.1 <1.3）且近期 CI 覆盖 1 → 假规律
        assert diagnose_death(1.0, 0.8, 1.2, peak_lift=1.1, hit_count=30) == DEATH_SPURIOUS

    def test_alive_when_still_significant(self):
        assert diagnose_death(1.5, 1.2, 1.9, peak_lift=1.6, hit_count=30) == DEATH_ALIVE
