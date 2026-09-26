import pytest
from scipy.stats import beta

from binance_predict.backtest.stats import wilson
from binance_predict.services.live_channel_benchmarks import get_channel_benchmark
from binance_predict.services.live_performance import (
    LiveOrder,
    actual_cost,
    aggregate_windows,
    analyze_live_performance,
    break_even,
    channel_series,
    duplicate_risk,
    ewma,
    market_window_key,
    rolling_binary_stats,
    rolling_realized_ev,
    segment_orders,
)


def order(**kwargs):
    defaults = dict(channel="x4_v2", market_period="5m", window_start=1, direction="DOWN")
    defaults.update(kwargs)
    return LiveOrder(**defaults)


@pytest.mark.parametrize("n", [0, 1, 19, 20, 49, 50, 51])
def test_rolling_20_50_boundaries(n):
    stats = rolling_binary_stats([True] * n)
    assert stats["20"]["n"] == min(n, 20)
    assert stats["50"]["n"] == min(n, 50)
    assert stats["20"]["full_window"] is (n >= 20)
    assert stats["50"]["full_window"] is (n >= 50)


@pytest.mark.parametrize("value", [True, False])
def test_all_wins_and_losses_match_wilson_and_jeffreys(value):
    stats = rolling_binary_stats([value] * 20)["20"]
    wins = 20 if value else 0
    assert stats["win_rate"] == float(value)
    assert stats["wilson_95"] == pytest.approx(wilson(float(value), 20))
    assert stats["jeffreys_95"] == pytest.approx(
        [beta.ppf(0.025, wins + 0.5, 20 - wins + 0.5), beta.ppf(0.975, wins + 0.5, 20 - wins + 0.5)]
    )


def test_ewma_uses_first_valid_value_and_span_20_alpha():
    values = ewma([None, 1.0, 0.0])
    assert values[:2] == [None, 1.0]
    assert values[2] == pytest.approx(19 / 21)


def test_reconciled_order_amount_and_shares_take_precedence():
    row = order(
        amount_in=str(2 * 10**18),
        quote_json={
            "amountIn": str(9 * 10**18),
            "filledShareQty": 2.5,
            "averagePrice": 0.7,
            "quotedAvgPrice": 0.6,
            "fillSource": "binance_history_confirm",
        },
    )
    assert actual_cost(row) == 2.0
    assert break_even(row) == {
        "break_even_probability": 0.8,
        "quality": "CAPTURED_FILL",
        "method": "ACTUAL_AMOUNT_OVER_SHARES",
    }


@pytest.mark.parametrize("amount_in", [2e18, "2e18", "2000000000000000000.0"])
def test_quote_amount_uses_fixed_wei_contract_and_legacy_unknown(amount_in):
    assert actual_cost(order(quote_json={"amountIn": amount_in})) == 2.0
    assert break_even(order(quote_json={"averagePrice": 0.12})) == {
        "break_even_probability": None,
        "quality": "LEGACY_UNKNOWN",
        "method": "LEGACY_UNKNOWN",
    }


def test_confirmed_average_price_is_fee_adjusted_without_shares():
    result = break_even(order(quote_json={"averagePrice": 0.12, "fillSource": "binance_history_confirm"}))
    assert result["break_even_probability"] == pytest.approx(0.12 / 0.98)
    assert result["quality"] == "CAPTURED_FILL_ESTIMATED_FEE"
    assert result["method"] == "FEE_ADJUSTED_ACTUAL_AVERAGE_PRICE"


def test_market_window_key_prefers_period_window_and_uses_market_id_only_as_fallback():
    assert market_window_key(order(market_id=123)) == ("period_window", "5m", 1)
    assert market_window_key(order(market_id=None)) == ("period_window", "5m", 1)
    assert market_window_key(order(market_id=None, market_period="15m")) != market_window_key(order())
    assert market_window_key(order(market_id=123, market_period=None, window_start=None)) == (
        "market_id", 123,
    )
    assert market_window_key(order(market_id=None, market_period=None, window_start=None)) is None


def test_same_window_aggregates_real_pnl_and_deduplicates_win_rate():
    rows = [
        order(channel="a", pnl=2, settle_outcome="DOWN", amount_in=str(2 * 10**18)),
        order(channel="b", pnl=-1, settle_outcome="DOWN", amount_in=str(3 * 10**18)),
    ]
    windows = aggregate_windows(rows)
    assert windows[0]["net_pnl"] == 1
    assert windows[0]["gross_cost"] == 5
    assert windows[0]["window_win"] is True
    report = analyze_live_performance(rows)
    assert report["window_stats"]["20"]["n"] == 1
    assert report["order_stats"]["20"]["n"] == 2


def test_opposite_orders_use_net_pnl_not_per_order_win_consistency():
    rows = [
        order(channel="up", direction="UP", settle_outcome="UP", pnl=3, amount_in=str(2 * 10**18)),
        order(channel="down", direction="DOWN", settle_outcome="UP", pnl=-1, amount_in=str(2 * 10**18)),
    ]
    window = aggregate_windows(rows)[0]
    assert window["window_win"] is True
    assert duplicate_risk(rows)["opposite_direction_overlap_count"] == 1


def test_settlement_conflict_excludes_win_but_keeps_money():
    rows = [
        order(channel="a", settle_outcome="UP", pnl=2, amount_in=str(2 * 10**18)),
        order(channel="b", settle_outcome="DOWN", pnl=-1, amount_in=str(3 * 10**18)),
    ]
    window = aggregate_windows(rows)[0]
    assert window["settlement_conflict"] is True
    assert window["window_win"] is None
    assert window["net_pnl"] == 1
    assert window["gross_cost"] == 5
    assert analyze_live_performance(rows)["window_stats"]["20"]["n"] == 0


def test_realized_ev_is_ratio_of_sums_not_mean_of_order_ev():
    rows = [
        order(pnl=1, amount_in=str(1 * 10**18)),
        order(window_start=2, pnl=1, amount_in=str(9 * 10**18)),
    ]
    assert rolling_realized_ev(rows, 20)["realized_ev"] == pytest.approx(0.2)


def test_g7_duplicate_exposure_and_overlap_cost():
    channels = ["firsthit_down_g7_v1", "g7_streak_v1", "g7_wick20_v1"]
    rows = [order(channel=channel, pnl=1, settle_outcome="DOWN", amount_in=str(value * 10**18))
            for channel, value in zip(channels, [2, 3, 5])]
    risk = duplicate_risk(rows)
    assert risk["duplicate_window_ratio"] == 1
    assert risk["duplicate_overlap_cost"] == 5
    assert risk["duplicate_investment_ratio"] == 0.5
    assert risk["max_single_window_cost"] == 10
    assert risk["max_channels_per_window"] == 3
    assert risk["same_direction_overlap_count"] == 1


def test_explicit_exclusive_group_conflict_and_benchmark_registry():
    rows = [
        order(channel="s2_cond_t4_v1", market_period="15m", amount_in=str(2 * 10**18)),
        order(channel="s2_cond_t5d_v1", market_period="15m", amount_in=str(2 * 10**18)),
    ]
    assert duplicate_risk(rows)["explicit_exclusive_conflict_count"] == 1
    assert get_channel_benchmark("scene_bull_exhaust").win_rate == 0.644
    assert get_channel_benchmark("s1_dyn_sq_v1").win_rate == 0.6493
    assert get_channel_benchmark("s1_dyn_sq_v1").sample_size == 1782
    assert get_channel_benchmark("x4_v2").win_rate == 0.553
    assert get_channel_benchmark("firsthit_down_g7_v1").win_rate == 0.192
    assert get_channel_benchmark("krev_a_v1").win_rate == 0.642
    assert get_channel_benchmark("krev_b_v1").sample_size == 134
    assert get_channel_benchmark("rev_p1_v1").win_rate == 0.620
    assert get_channel_benchmark("rev_p2_v1").win_rate == 0.624
    assert get_channel_benchmark("unknown") is None


def test_average_price_requires_confirmed_fill_source_whitelist():
    assert break_even(order(quote_json={"averagePrice": 0.2, "fillSource": "other"}))[
        "break_even_probability"
    ] is None
    synced = order(quote_json={
        "averagePrice": 0.3,
        "source": "binance_history_sync",
        "binanceOrderStatus": "FILLED",
    })
    assert break_even(synced)["break_even_probability"] == pytest.approx(0.3 / 0.98)
    unfilled = order(quote_json={
        "averagePrice": 0.4,
        "source": "binance_history_sync",
        "binanceOrderStatus": "NEW",
    })
    assert break_even(unfilled)["break_even_probability"] is None


def test_explicit_win_takes_precedence_over_direction_outcome():
    report = analyze_live_performance([
        order(win=False, direction="DOWN", settle_outcome="DOWN"),
        order(window_start=2, win=True, direction="DOWN", settle_outcome="UP"),
    ])
    assert report["order_stats"]["20"]["wins"] == 1


def test_realized_ev_full_window_counts_only_usable_rows():
    rows = [order(id=i, window_start=i, pnl=1, amount_in=str(10**18)) for i in range(20)]
    rows[-1] = order(id=19, window_start=19, pnl=None, amount_in=str(10**18))
    stats = rolling_realized_ev(rows, 20)
    assert stats["n"] == 19
    assert stats["full_window"] is False


def test_channel_series_is_sorted_and_has_exact_20_50_boundaries():
    rows = [
        order(
            id=i,
            window_start=i,
            win=i % 2 == 0,
            pnl=1 if i % 2 == 0 else -1,
            amount_in=str(2 * 10**18),
            quote_json={"filledShareQty": 4, "amountIn": str(2 * 10**18)},
        )
        for i in reversed(range(50))
    ]
    series = channel_series(rows)
    assert [point["id"] for point in series] == list(range(50))
    assert series[18]["rolling"]["20"]["full_window"] is False
    assert series[19]["rolling"]["20"]["full_window"] is True
    assert series[48]["rolling"]["50"]["full_window"] is False
    assert series[49]["rolling"]["50"]["full_window"] is True
    assert series[49]["rolling_realized_ev"]["50"]["full_window"] is True
    assert series[49]["rolling_mean_break_even_probability"]["20"] == 0.5
    assert series[49]["benchmark_win_rate"] == 0.553
    assert series[49]["benchmark_gap"]["50"] == pytest.approx(0.5 - 0.553)


def test_portfolio_series_uses_window_time_not_market_id():
    from binance_predict.services.live_performance import portfolio_window_series

    rows = [
        order(market_id=10, window_start=2000, pnl=1, amount_in=str(10**18)),
        order(market_id=99, window_start=1000, pnl=2, amount_in=str(10**18)),
    ]
    series = portfolio_window_series(rows)
    assert [point["t"] for point in series] == [1000, 2000]
    assert [point["market_window_key"] for point in series] == [
        ["period_window", "5m", 1000], ["period_window", "5m", 2000],
    ]


def test_quote_and_exec_price_fine_bucketing():
    # 模拟两笔订单：一笔 break_even 0.52，一笔 break_even 0.63
    rows = [
        order(
            id=1,
            amount_in=str(2 * 10**18),
            win=True,
            pnl=1.5,
            quote_json={
                "amountIn": str(2 * 10**18),
                "filledShareQty": 3.846,  # 2.0 / 3.846 ≈ 0.52 -> [0.50,0.55)
                "averagePrice": 0.48,
                "fillSource": "binance_history_confirm",
            },
        ),
        order(
            id=2,
            amount_in=str(2 * 10**18),
            win=False,
            pnl=-2.0,
            quote_json={
                "amountIn": str(2 * 10**18),
                "filledShareQty": 3.174,  # 2.0 / 3.174 ≈ 0.63 -> [0.60,0.65)
                "averagePrice": 0.61,
                "fillSource": "binance_history_confirm",
            },
        ),
        order(
            id=3,
            amount_in=str(2 * 10**18),
            win=True,
            pnl=1.0,
            quote_json={
                "amountIn": str(2 * 10**18),
                "filledShareQty": 7.0,  # 2.0 / 7.0 ≈ 0.285 -> <0.30
                "averagePrice": 0.27,
                "source": "binance_history_sync",
                "binanceOrderStatus": "FILLED",
            },
        ),
        order(
            id=4,
            amount_in=str(2 * 10**18),
            win=True,
            pnl=0.5,
            quote_json={
                "amountIn": str(2 * 10**18),
                "filledShareQty": 2.5,  # 2.0 / 2.5 = 0.80 -> >=0.70
                "averagePrice": 0.76,
                "fillSource": "binance_history_confirm",
            },
        ),
        order(
            id=5,
            amount_in=str(2 * 10**18),
            win=False,
            pnl=-2.0,
            quote_json={
                "amountIn": str(2 * 10**18),
                "averagePrice": 0.55,  # 报价估算，不得混入确认成交价分桶
            },
        ),
    ]

    # 测试保本率分桶 (quote)
    quote_segments = {s["segment"]: s for s in segment_orders(rows, "quote")}
    assert "<0.30" in quote_segments
    assert "[0.50,0.55)" in quote_segments
    assert "[0.60,0.65)" in quote_segments
    assert ">=0.70" in quote_segments
    assert quote_segments["[0.50,0.55)"]["n"] == 1
    assert quote_segments["[0.50,0.55)"]["wins"] == 1
    assert quote_segments["[0.60,0.65)"]["wins"] == 0

    # 测试成交均价分桶 (exec_price)
    exec_segments = {s["segment"]: s for s in segment_orders(rows, "exec_price")}
    assert "<0.30" in exec_segments
    assert "[0.45,0.50)" in exec_segments  # 0.48 落在 [0.45,0.50)
    assert "[0.60,0.65)" in exec_segments  # 0.61 落在 [0.60,0.65)
    assert ">=0.70" in exec_segments      # 0.76 落在 >=0.70
    assert exec_segments["[0.45,0.50)"]["wins"] == 1
    assert exec_segments["LEGACY_UNKNOWN"]["n"] == 1


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (0.299999, "<0.30"),
        (0.30, "[0.30,0.35)"),
        (0.349999, "[0.30,0.35)"),
        (0.35, "[0.35,0.40)"),
        (0.699999, "[0.65,0.70)"),
        (0.70, ">=0.70"),
    ],
)
def test_price_bucket_boundaries(price, expected):
    from binance_predict.services.live_performance import _price_bucket

    assert _price_bucket(price) == expected


# ---- 信号真相（剔除动态下单金额）----

def _settled(id_, stake, win, price, **kw):
    """构造已结算订单：赢 pnl=stake*(0.98/price-1)，输 pnl=-stake；成交确认价=price。"""
    pnl = stake * (0.98 / price - 1) if win else -stake
    return order(
        id=id_, window_start=id_, amount_in=str(int(stake * 10**18)), win=win, pnl=pnl,
        quote_json={"averagePrice": price, "fillSource": "binance_history_confirm"}, **kw,
    )


def test_signal_truth_separates_equal_weight_ev_from_capital_roi():
    from binance_predict.services.live_performance import signal_truth, unit_return

    # 3 笔 1U 小注全赢 + 1 笔 50U 大注输：信号本身赚钱，钱包亏钱
    rows = [_settled(i, 1.0, True, 0.5) for i in range(1, 4)] + [_settled(4, 50.0, False, 0.5)]
    assert unit_return(rows[0]) == pytest.approx(0.96)
    truth = signal_truth(rows)
    assert truth["signal_n"] == 4
    assert truth["signal_ev"] == pytest.approx((0.96 * 3 - 1) / 4)
    assert truth["capital_roi"] == pytest.approx((0.96 * 3 - 50) / 53)
    assert truth["sizing_distortion"] == pytest.approx(truth["capital_roi"] - truth["signal_ev"])
    assert truth["sizing_distortion"] < 0
    assert truth["stake_max"] == 50.0
    assert truth["stake_median"] == 1.0
    assert truth["top3_stake_share"] == pytest.approx(52 / 53)
    assert truth["equal_stake_pnl"] == pytest.approx(truth["signal_ev"] * 4 * 1.0)
    # 保本率 = 0.5/0.98；edge = 平均(win - 保本率)
    be = 0.5 / 0.98
    assert truth["mean_break_even"] == pytest.approx(be)
    assert truth["edge"] == pytest.approx(0.75 - be)
    assert truth["edge_n"] == 4
    lo, hi = truth["signal_ev_ci95"]
    assert lo < truth["signal_ev"] < hi


def test_signal_truth_equal_stakes_have_zero_distortion():
    from binance_predict.services.live_performance import signal_truth

    rows = [_settled(1, 5.0, True, 0.4), _settled(2, 5.0, False, 0.4)]
    truth = signal_truth(rows)
    assert truth["sizing_distortion"] == pytest.approx(0.0)
    assert truth["equal_stake_pnl"] == pytest.approx(sum(r.pnl for r in rows))


def test_signal_truth_empty_and_missing_cost_are_none_safe():
    from binance_predict.services.live_performance import signal_truth

    empty = signal_truth([])
    assert empty["signal_ev"] is None and empty["capital_roi"] is None
    assert empty["sizing_distortion"] is None and empty["edge"] is None
    assert empty["signal_ev_ci95"] is None and empty["equal_stake_pnl"] is None
    # 无成本行不进信号 EV，也不进资金 ROI 分子（口径一致）
    no_cost = order(id=9, win=True, pnl=3.0)
    truth = signal_truth([no_cost, _settled(1, 2.0, False, 0.5)])
    assert truth["signal_n"] == 1
    assert truth["capital_roi"] == pytest.approx(-1.0)
    assert truth["signal_ev_ci95"] is None  # 单样本不给区间


@pytest.mark.parametrize(
    ("stake", "expected"),
    [(2.999, "<3U"), (3.0, "3-8U"), (7.99, "3-8U"), (8.0, "8-15U"), (15.0, "15-30U"),
     (29.99, "15-30U"), (30.0, ">=30U"), (50.0, ">=30U")],
)
def test_stake_bucket_boundaries(stake, expected):
    from binance_predict.services.live_performance import _stake_bucket

    assert _stake_bucket(stake) == expected
    assert _stake_bucket(None) == "LEGACY_UNKNOWN"


def test_stake_segments_are_ordered_and_carry_signal_fields():
    rows = [_settled(1, 50.0, False, 0.6), _settled(2, 1.0, True, 0.5), _settled(3, 10.0, True, 0.5)]
    segments = segment_orders(rows, "stake")
    assert [s["segment"] for s in segments] == ["<3U", "8-15U", ">=30U"]
    big = segments[-1]
    assert big["signal_ev"] == pytest.approx(-1.0)
    assert big["realized_ev"] == pytest.approx(-1.0)
    assert big["sizing_distortion"] == pytest.approx(0.0)
    assert big["stake_max"] == 50.0
    assert big["low_sample"] is True
    assert big["mean_break_even"] == pytest.approx(0.6 / 0.98)
    assert big["win_rate_wilson_95"] is not None


def test_channel_series_exposes_rolling_signal_ev():
    rows = [_settled(i, 1.0 if i % 2 else 40.0, i % 2 == 1, 0.5) for i in range(1, 23)]
    last = channel_series(rows)[-1]
    sig20 = last["rolling_signal_ev"]["20"]
    assert sig20["n"] == 20 and sig20["full_window"] is True
    # 最近 20 笔：10 赢(1U) + 10 输(40U) → 等权 EV = (0.96*10 - 10)/20
    assert sig20["signal_ev"] == pytest.approx((0.96 * 10 - 10) / 20)
    assert last["rolling_realized_ev"]["20"]["realized_ev"] < sig20["signal_ev"]
    assert last["unit_return"] == pytest.approx(-1.0)
