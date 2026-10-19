"""attribution_stats 检验库单测：已知分布验证 + 贴线边界正反例。"""
from __future__ import annotations

import math

from binance_predict.services import attribution_stats as astats


# ======================================================================
# 二项精确检验
# ======================================================================

class TestBinomial:
    def test_p_le_known_value(self):
        # P(X<=8 | n=21, p=65/105) 此前手算 ≈0.023（S4 连败案例）
        p = astats.binom_p_le(8, 21, 65 / 105)
        assert 0.02 < p < 0.03

    def test_p_le_symmetry_with_p_ge(self):
        # P(X<=k) + P(X>=k+1) = 1
        for k in (0, 3, 10, 20):
            assert abs(astats.binom_p_le(k, 20, 0.6) + astats.binom_p_ge(k + 1, 20, 0.6) - 1) < 1e-9

    def test_boundaries(self):
        assert astats.binom_p_le(21, 21, 0.6) > 1 - 1e-9  # 浮点累加容差
        assert astats.binom_p_ge(0, 21, 0.6) == 1.0
        assert astats.binom_p_le(-1, 21, 0.6) == 0.0
        assert astats.binom_p_le(5, 0, 0.6) == 1.0  # n=0 边界


# ======================================================================
# 随机性前置裁决（贴线：p 恰等于 alpha 判 RANDOM_CONSISTENT）
# ======================================================================

class TestStreakAssessment:
    def test_s4_case_anomalous_losing(self):
        wins = [True] * 8 + [False] * 13
        res = astats.streak_assessment(wins, base_rate=65 / 105)
        assert res["verdict"] == "ANOMALOUS_LOSING"
        assert res["n"] == 21 and res["wins"] == 8

    def test_normal_luck_random_consistent(self):
        # 12/20 胜 @ 基准 0.62 —— 正常波动
        wins = [True] * 12 + [False] * 8
        assert astats.streak_assessment(wins, 0.62)["verdict"] == "RANDOM_CONSISTENT"

    def test_insufficient_data(self):
        assert astats.streak_assessment([True] * 7, 0.6)["verdict"] == "INSUFFICIENT_DATA"
        assert astats.streak_assessment([True] * 20, 1.0)["verdict"] == "INSUFFICIENT_DATA"

    def test_none_entries_filtered(self):
        wins = [True] * 8 + [False] * 13 + [None] * 5
        res = astats.streak_assessment(wins, 65 / 105)
        assert res["n"] == 21 and res["verdict"] == "ANOMALOUS_LOSING"

    def test_tie_line_p_equals_alpha(self):
        # 构造 p 恰好 = alpha 的贴线：找一个 (k,n,p) 组合使 binom_p_le 精确落在 0.05
        # n=20, p=0.8: P(X<=12) ≈ 0.0321（异常）；P(X<=13) ≈ 0.0867（贴线内不异常）
        wins = [True] * 13 + [False] * 7
        res = astats.streak_assessment(wins, 0.8)
        assert res["binom_p_le"] > 0.05
        assert res["verdict"] == "RANDOM_CONSISTENT"
        wins12 = [True] * 12 + [False] * 8
        assert astats.streak_assessment(wins12, 0.8)["verdict"] == "ANOMALOUS_LOSING"

    def test_anomalous_winning_side(self):
        # 20 单 19 胜 @ 基准 0.55 —— 连胜侧异常
        wins = [True] * 19 + [False]
        assert astats.streak_assessment(wins, 0.55)["verdict"] == "ANOMALOUS_WINNING"


# ======================================================================
# runs test / longest run
# ======================================================================

class TestRuns:
    def test_longest_run(self):
        flags = [True, False, False, False, True, True]
        assert astats.longest_run(flags, False) == 3
        assert astats.longest_run(flags, True) == 2

    def test_runs_z_sign(self):
        # 交替序列 runs 多 → z 为正；扎堆序列 runs 少 → z 为负
        alt = [i % 2 == 0 for i in range(20)]
        clumped = [True] * 10 + [False] * 10
        z_alt = astats.runs_test_z(alt)
        z_clump = astats.runs_test_z(clumped)
        assert z_alt > 0 and z_clump < 0
        assert astats.runs_test_z([True] * 10) is None  # 单侧序列不可算


# ======================================================================
# 日块自助（去簇 + 可复现）
# ======================================================================

class TestBlockBootstrap:
    def test_deterministic_same_seed(self):
        vals = [0.1, -0.05, 0.2, -0.3, 0.15, 0.05, -0.1, 0.25, 0.3, -0.02]
        ts = [1_700_000_000_000 + i * 3_600_000 for i in range(10)]
        a = astats.block_bootstrap_mean_p(vals, ts)
        b = astats.block_bootstrap_mean_p(vals, ts)
        assert a == b
        assert 0 < a["p_le_zero"] < 1

    def test_strong_positive_not_significant_le_zero(self):
        # 均值明显为正：p_le_zero 应很小
        vals = [0.5, 0.6, 0.4, 0.7, 0.3, 0.55]
        ts = [1_700_000_000_000 + i * 86_400_000 for i in range(6)]  # 各自一天
        res = astats.block_bootstrap_mean_p(vals, ts)
        assert res["p_le_zero"] < 0.05 and res["significant_positive"]

    def test_clustered_days_dont_inflate(self):
        # 全部同一天的强负值 vs 分散强负值：同日簇的自由度更少，
        # p 值应不小于分散情形（去簇保守性）
        vals = [-0.2] * 10
        ts_same = [1_700_000_000_000 + i * 60_000 for i in range(10)]
        ts_spread = [1_700_000_000_000 + i * 86_400_000 for i in range(10)]
        p_same = astats.block_bootstrap_mean_p(vals, ts_same)["p_le_zero"]
        p_spread = astats.block_bootstrap_mean_p(vals, ts_spread)["p_le_zero"]
        assert p_same >= p_spread

    def test_empty(self):
        assert astats.block_bootstrap_mean_p([], [])["n"] == 0


# ======================================================================
# 分桶对比 / CUSUM / TCA 分解
# ======================================================================

class TestBucketAndCusum:
    def test_bucket_bonferroni(self):
        groups = {
            "a": {"values": [0.5] * 5, "timestamps_ms": [i * 86_400_000 for i in range(5)]},
            "b": {"values": [-0.5] * 5, "timestamps_ms": [i * 86_400_000 for i in range(5)]},
            "c": {"values": [], "timestamps_ms": []},
        }
        res = astats.bucket_compare(groups)
        assert res["n_buckets"] == 2
        for b in res["buckets"].values():
            assert b["p_bonferroni"] <= 1.0

    def test_cusum_finds_break(self):
        vals = [0.1] * 15 + [-0.2] * 15
        res = astats.cusum_changepoint(vals)
        assert res["index"] is not None
        assert 10 <= res["index"] <= 19  # 断点在真断附近
        assert res["delta"] < 0

    def test_cusum_short_series(self):
        assert astats.cusum_changepoint([0.1] * 5)["index"] is None


class TestTcaDecompose:
    def _order(self, win=True, avg=0.55, quoted=0.54, amount=20.0, pnl=None):
        return {"win": win, "avg_price": avg, "quoted_avg_price": quoted,
                "amount_usdt": amount, "pnl": pnl if pnl is not None else
                ((1 / avg - 1) * amount if win else -amount)}

    def test_slippage_bp(self):
        res = astats.tca_decompose(self._order(avg=0.55, quoted=0.5415))
        assert abs(res["slippage_bp"] - 156.8) < 0.5  # (0.55/0.5415-1)*1e4 ≈ 156.9

    def test_pnl_breakdown_win(self):
        res = astats.tca_decompose(self._order(win=True))
        assert abs(res["pnl_theoretical"] - (1 / 0.55 - 1) * 20) < 1e-3  # 结果按 4 位小数舍入
        assert res["pnl_gap"] == 0.0

    def test_pnl_breakdown_loss(self):
        res = astats.tca_decompose(self._order(win=False))
        assert res["pnl_theoretical"] == -20.0

    def test_missing_fields_tolerated(self):
        res = astats.tca_decompose({"win": None})
        assert res["slippage_bp"] is None and "pnl_theoretical" not in res
