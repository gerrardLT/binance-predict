"""时段胜率分析脚本（local_timeslice_winrate）的纯函数确定性测试。

零网络 / 零 DB / 零数据文件依赖：只测时间标签器边界、口径纯函数与
分组统计的三层采信链。数据资产与重放口径由脚本运行时对账（Step 5）。
"""
from __future__ import annotations

import datetime as dt
import os
import sys

# scripts/ 非包，注入路径后导入分析模块（与 test_feature_bench.py 同模式）
_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import local_timeslice_winrate as tw  # noqa: E402


def _ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp() * 1000)


# ----------------------------------------------------------------------
# 时间标签器边界
# ----------------------------------------------------------------------

def test_hour_boundaries_utc():
    assert tw.hour_utc(_ms(2026, 8, 24, 0, 0)) == "00"
    assert tw.hour_utc(_ms(2026, 8, 24, 23, 59)) == "23"
    assert tw.session_utc(_ms(2026, 8, 24, 0, 0)) == "asia"
    assert tw.session_utc(_ms(2026, 8, 24, 6, 59)) == "asia"
    assert tw.session_utc(_ms(2026, 8, 24, 7, 0)) == "europe"
    assert tw.session_utc(_ms(2026, 8, 24, 13, 0)) == "america"
    assert tw.session_utc(_ms(2026, 8, 24, 21, 0)) == "late"
    assert tw.session_utc(_ms(2026, 8, 24, 23, 59)) == "late"


def test_utc_1600_is_bjt_midnight_next_day():
    # 2026-08-24（周一）16:00 UTC = 2026-08-25（周二）00:00 北京时间
    ts = _ms(2026, 8, 24, 16, 0)
    assert tw.hour_bjt(ts) == "00"
    assert tw.dow_bjt(ts) == "Tue"
    assert tw.dow_utc(ts) == "Mon"


def test_sunday_2300_utc_crosses_to_monday_bjt():
    # 2026-08-23（周日）23:00 UTC = 2026-08-24（周一）07:00 北京时间（周界跨界）
    ts = _ms(2026, 8, 23, 23, 0)
    assert tw.weekend_utc(ts) == "weekend"
    assert tw.weekend_bjt(ts) == "weekday"
    assert tw.dow_bjt(ts) == "Mon"
    assert tw.session_bjt(ts) == "europe"  # 北京 07:00 落入 europe 段 [07,13)


def test_dom_phase_month_season():
    assert tw.dom_phase(_ms(2026, 1, 10)) == "early(01-10)"
    assert tw.dom_phase(_ms(2026, 1, 11)) == "mid(11-20)"
    assert tw.dom_phase(_ms(2026, 1, 31)) == "late(21-31)"
    assert tw.month(_ms(2026, 3, 15)) == "03"
    assert tw.season(_ms(2026, 3, 1)) == "spring"
    assert tw.season(_ms(2026, 7, 1)) == "summer"
    assert tw.season(_ms(2026, 10, 1)) == "autumn"
    assert tw.season(_ms(2026, 12, 1)) == "winter"
    assert tw.season(_ms(2026, 2, 28)) == "winter"


# ----------------------------------------------------------------------
# 口径纯函数：分版本盈亏平衡 / EV
# ----------------------------------------------------------------------

def test_breakeven_of_version_split():
    # x4 与场景族含溢 0.01；quote 族无溢价
    assert abs(tw.breakeven_of("x4_v1", 0.50) - 0.51 / 0.98) < 1e-12
    assert abs(tw.breakeven_of("S1", 0.50) - 0.51 / 0.98) < 1e-12
    assert abs(tw.breakeven_of("quote_momentum_v1", 0.70) - 0.70 / 0.98) < 1e-12


def test_ev_of_win_lose():
    # 赢：0.98/(q[+0.01])−1；输：−1
    assert abs(tw.ev_of("x4_v1", True, 0.50) - (0.98 / 0.51 - 1.0)) < 1e-9
    assert abs(tw.ev_of("quote_contrarian_v1", True, 0.20) - (0.98 / 0.20 - 1.0)) < 1e-9
    assert tw.ev_of("x4_v1", False, 0.50) == -1.0
    assert tw.ev_of("S4", False, 0.4) == -1.0


# ----------------------------------------------------------------------
# 分组统计：n/wins/wr/Wilson 数值 + 三层采信链
# ----------------------------------------------------------------------

def _rec(src: str, key: str, win: bool, ts: int | None = None) -> tw.Record:
    return tw.Record(src, key, ts or _ms(2026, 8, 24, 3, 0), "DOWN", win, None, None)


def test_group_stats_small_sample_insufficient():
    recs = [_rec("replay", "quote_momentum_v1", i < 7) for i in range(10)]
    rows, candidates, adopted = tw.group_stats(recs, {}, 0.5)
    row = rows[("replay", "quote_momentum_v1", "hour_utc", "03")]
    assert row["n"] == 10
    assert row["wins"] == 7
    assert abs(row["wr"] - 0.7) < 1e-9
    # Wilson CI 数值与统计内核一致
    from binance_predict.backtest.stats import wilson
    lo, hi = wilson(0.7, 10)
    assert abs(row["ci_lo"] - round(lo, 4)) < 1e-9
    assert abs(row["ci_hi"] - round(hi, 4)) < 1e-9
    assert row["verdict"] == "INSUFFICIENT"   # n<30 不解读
    assert not candidates and not adopted


def test_group_stats_full_chain_candidate_to_adopted():
    # 双源各 200 笔：03 时格 100 笔 70 胜 / 04 时格 100 笔 50 胜 →
    # 自身全量 wr=60%；03 格 vs 自身 +10pp（≥Bonferroni 6.93pp）∧
    # 双源同向 → 采信（正/负向偏离均可，时段分化双向捕捉）
    baseline = {"hour_utc": {"03": {"n": 1000, "p_down": 0.30},
                             "04": {"n": 1000, "p_down": 0.30}}}

    def _batch(src: str) -> list[tw.Record]:
        return ([_rec(src, "quote_momentum_v1", i < 70) for i in range(100)]
                + [_rec(src, "quote_momentum_v1", i < 50, _ms(2026, 8, 24, 4, 0))
                   for i in range(100)])

    recs = _batch("replay") + _batch("online")
    rows, candidates, adopted = tw.group_stats(recs, baseline, 0.5)
    for src in ("replay", "online"):
        hi = rows[(src, "quote_momentum_v1", "hour_utc", "03")]
        lo_row = rows[(src, "quote_momentum_v1", "hour_utc", "04")]
        assert hi["n"] == 100 and hi["wins"] == 70
        assert abs(hi["dev_pp"] - 40.0) < 1e-9        # vs DOWN 基线 30%
        assert abs(hi["dev_own_pp"] - 10.0) < 1e-9    # vs 自身全量 60%
        assert hi["verdict"] == "ADOPTED" and hi["dual_source"] is True
        assert lo_row["verdict"] == "ADOPTED"          # 负向分化同样采信
        assert lo_row["dev_own_pp"] < 0
    # 4 = hour_utc 03/04 × 2 源；+2 = hour_bjt 联动格（03/04 UTC = 11/12 BJT，
    # 11 格 lo>wr 进候选且同向偏离）
    assert len(adopted) == 6


def test_time_gated_excluded_from_discovery():
    # late_night 族本身时段限定 → hour 维度标 TIME_GATED 不进候选
    baseline = {"hour_bjt": {"22": {"n": 500, "p_down": 0.30}}}
    recs = [_rec("replay", "late_night_contrarian_v1", i < 40,
                 _ms(2026, 8, 24, 14, 0)) for i in range(50)]  # 14 UTC = 22 BJT
    rows, candidates, _ = tw.group_stats(recs, baseline, 0.5)
    row = rows[("replay", "late_night_contrarian_v1", "hour_bjt", "22")]
    assert row["verdict"] == "TIME_GATED"
    assert not candidates
    # 同批数据在非时段维度（周末）仍可正常参评
    row_w = rows[("replay", "late_night_contrarian_v1", "weekend_bjt", "weekday")]
    assert row_w["verdict"] in ("EXPLORE", "CANDIDATE")


def test_bonferroni_threshold_scales_with_hypotheses():
    from binance_predict.backtest.stats import multiple_testing_threshold
    assert multiple_testing_threshold(2.0, 1)["required_pp"] == 2.0
    assert multiple_testing_threshold(2.0, 9)["required_pp"] == 6.0
    # 本脚本先验检验数 = session 4×2 + weekend 2×2 = 12
    assert tw._n_prior_tests() == 12
