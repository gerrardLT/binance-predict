"""通道 EV 趋势报警器单测（2026-10-22）。

覆盖：MAD 稳健性 / CUSUM 正反例 / 分级边界 / 覆盖不足禁用 / CUSUM 曲线。
口径依据：μ_ref=0 单侧 CUSUM，σ=1.4826×MAD，k=0.5σ，h=5σ。
"""
from __future__ import annotations

from binance_predict.services import channel_trend as ct


def _evs(values, be: float | None = 0.55, start_ts: int = 1_700_000_000_000,
         step: int = 300_000):
    """EV 序列夹具：win 由 EV 正负近似（趋势测试只需要胜负与 EV 同向）。"""
    return [{"ts": start_ts + i * step, "win": float(v) > 0, "ev": float(v),
             "be": be, "q": be}
            for i, v in enumerate(values)]


class TestMadSigma:
    def test_constant_series_is_zero(self):
        # 中位数与偏差均为 0 → 退化序列（调用方会用 floor 兜底）
        assert ct.mad_sigma([0.02] * 20) == 0.0

    def test_outlier_does_not_inflate(self):
        # 19 个 0.02 + 1 个 -1.0：MAD 仍为 0（极端值不撑大方差）
        vals = [0.02] * 19 + [-1.0]
        assert ct.mad_sigma(vals) == 0.0
        # 有噪声时 MAD 明显小于标准差
        vals = [0.01, 0.02, 0.03, 0.02, -1.0]
        import statistics
        sd = statistics.pstdev(vals)
        assert ct.mad_sigma(vals) < sd

    def test_insufficient(self):
        assert ct.mad_sigma([]) is None
        assert ct.mad_sigma([0.1]) is None


class TestEvCusum:
    def test_small_sample_no_alarm(self):
        out = ct.ev_cusum_alarm(_evs([-0.5] * (ct.TREND_MIN_N - 1)))
        assert out["alarm"] is False
        assert out["n"] == ct.TREND_MIN_N - 1

    def test_downtrend_alarms_with_confirmation_window(self):
        # 40 单 +0.05 → 15 单 -0.05（低噪声，σ 走 floor）：约 2 单内越界
        out = ct.ev_cusum_alarm(_evs([0.05] * 40 + [-0.05] * 15))
        assert out["alarm"] is True
        assert out["since_alarm_n"] is not None
        assert out["since_alarm_n"] >= ct.L1_CONFIRM_N
        assert out["cusum"] > 0

    def test_stable_positive_no_alarm(self):
        out = ct.ev_cusum_alarm(_evs([0.05] * 60))
        assert out["alarm"] is False

    def test_alternating_noise_no_alarm(self):
        out = ct.ev_cusum_alarm(_evs([0.1, -0.1] * 30))
        assert out["alarm"] is False


class TestClassifyTrend:
    def test_coverage_floor_blocks_conclusion(self):
        level, reasons = ct.classify_trend(_evs([0.05] * 50), coverage_ratio=0.5)
        assert level == "L0"
        assert "覆盖率" in reasons[0]

    def test_small_sample_gray(self):
        level, reasons = ct.classify_trend(_evs([0.05] * (ct.TREND_MIN_N - 1)))
        assert level == "L0"
        assert reasons == []

    def test_l1_requires_ten_after_change(self):
        # 40 正 + 12 负：变化点后约 10-11 单，但不足 L2 的 15 单
        level, reasons = ct.classify_trend(_evs([0.05] * 40 + [-0.05] * 12))
        assert level == "L1"
        assert any("CUSUM" in r for r in reasons)

    def test_l2_when_after_edge_negative(self):
        # 40 正 + 20 负：变化后窗口胜率 0 < 保本 → L2（后验仍高，不会升 L3）
        level, reasons = ct.classify_trend(_evs([0.05] * 40 + [-0.05] * 20))
        assert level == "L2"
        assert any("边际" in r or "后验" in r for r in reasons)

    def test_l3_matches_red_light(self):
        # 50 单仅 5 胜且保本 0.9 → 优势已变负后验 ≈1 → L3（与红灯同口径）
        values = [0.05] * 5 + [-0.05] * 45
        evs = _evs(values, be=0.9)
        level, reasons = ct.classify_trend(evs)
        assert level == "L3"
        assert any("优势已变负" in r for r in reasons)


class TestTrendPayload:
    def test_payload_fields(self):
        evs = _evs([0.05] * 40 + [-0.05] * 20)
        p = ct.trend_payload("ch_a", evs, coverage_ratio=1.0)
        for key in ("channel", "level", "reasons", "n", "win_rate",
                    "p_edge_negative", "mean_breakeven", "edge",
                    "rolling20_win_rate", "cusum", "cusum_sigma",
                    "since_alarm_n", "coverage_ratio", "level_rank"):
            assert key in p, f"trend_payload 缺少字段 {key}"
        assert p["channel"] == "ch_a"
        assert p["level_rank"] == ct.LEVEL_RANK[p["level"]]

    def test_cusum_series_monotone_after_break(self):
        evs = _evs([0.05] * 40 + [-0.05] * 20)
        series = ct.cusum_series(evs)
        assert len(series) == 60
        assert series[-1]["cusum"] > series[39]["cusum"]
        assert series[-1]["threshold"] > 0


class TestFormatAlert:
    def test_alert_contains_core_facts(self):
        p = ct.trend_payload("ch_a", _evs([0.05] * 40 + [-0.05] * 20))
        text = ct.format_trend_alert(p, "测试通道", advice_hint="下调金额 2→1")
        assert "测试通道" in text and "ch_a" in text
        assert "AI 策略建议" in text and "下调金额" in text
        assert "仅为建议" in text
