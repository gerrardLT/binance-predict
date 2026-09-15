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


def test_actual_amount_and_shares_take_precedence():
    row = order(
        amount_in=str(9 * 10**18),
        quote_json={
            "amountIn": str(2 * 10**18),
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


def test_market_window_key_prefers_market_id_and_period_fallback_separates_5m_15m():
    assert market_window_key(order(market_id=123)) == ("market_id", 123)
    assert market_window_key(order(market_id=None)) == ("period_window", "5m", 1)
    assert market_window_key(order(market_id=None, market_period="15m")) != market_window_key(order())
    assert market_window_key(order(market_period=None, window_start=None)) is None


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
    assert get_channel_benchmark("x4_v2").win_rate == 0.553
    assert get_channel_benchmark("firsthit_down_g7_v1").win_rate == 0.192
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
        ["market_id", 99], ["market_id", 10],
    ]
