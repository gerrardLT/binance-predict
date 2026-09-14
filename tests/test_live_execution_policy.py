import hashlib
import json

import pytest

from binance_predict.services.live_execution_policy import (
    config_fingerprint,
    evaluate_absorption_policy,
    evaluate_quote_execution_policy,
)


BANDS = ((0.10, 0.20), (0.30, 0.40))


def test_absorption_up_golden_values_calculated_independently() -> None:
    # BTC: +20 bp; UP: +1.5 pp; expected response: 2*20+1=41 pp.
    result = evaluate_absorption_policy(
        btc_price=100.2,
        btc_open=100.0,
        up_price=0.515,
        up_open=0.5,
        k=2.0,
        b=1.0,
        disp_gate=10.0,
        under_gate=39.0,
    )

    assert result.btc_move == pytest.approx(20.0)
    assert result.up_move == pytest.approx(1.5)
    assert result.residual == pytest.approx(-39.5)
    assert result.under == pytest.approx(39.5)
    assert result.prediction == "UP"
    assert result.eligible is True
    assert result.reason == "PASSED"


def test_absorption_down_golden_values_calculated_independently() -> None:
    # BTC: -30 bp; UP: +0.5 pp; expected response: -59 pp, so residual/under=59.5.
    result = evaluate_absorption_policy(
        btc_price=99.7,
        btc_open=100.0,
        up_price=0.505,
        up_open=0.5,
        k=2.0,
        b=1.0,
        disp_gate=20.0,
        under_gate=50.0,
    )

    assert result.btc_move == pytest.approx(-30.0)
    assert result.up_move == pytest.approx(0.5)
    assert result.residual == pytest.approx(59.5)
    assert result.under == pytest.approx(59.5)
    assert result.prediction == "DOWN"
    assert result.eligible is True


def test_absorption_boundary_equality_passes() -> None:
    # +1 bp and zero UP move with k=1 gives residual=-1 and under=1.
    result = evaluate_absorption_policy(
        btc_price=10001.0,
        btc_open=10000.0,
        up_price=0.5,
        up_open=0.5,
        k=1.0,
        b=0.0,
        disp_gate=1.0,
        under_gate=1.0,
    )

    assert result.btc_move == 1.0
    assert result.under == 1.0
    assert result.eligible is True


@pytest.mark.parametrize("btc_open", [0.0, -1.0, float("nan"), float("inf")])
def test_absorption_rejects_invalid_btc_open(btc_open: float) -> None:
    result = evaluate_absorption_policy(
        btc_price=100.0,
        btc_open=btc_open,
        up_price=0.5,
        up_open=0.5,
        k=1.0,
        b=0.0,
        disp_gate=1.0,
        under_gate=1.0,
    )

    assert result.eligible is False
    assert result.prediction is None
    assert result.reason == "INVALID_BTC_OPEN"


def test_absorption_rejects_zero_move() -> None:
    result = evaluate_absorption_policy(
        btc_price=100.0,
        btc_open=100.0,
        up_price=0.5,
        up_open=0.5,
        k=1.0,
        b=0.0,
        disp_gate=0.0,
        under_gate=0.0,
    )

    assert result.btc_move == 0.0
    assert result.eligible is False
    assert result.prediction is None
    assert result.reason == "ZERO_OR_INVALID_BTC_MOVE"


def test_absorption_rejects_each_strict_gate_failure() -> None:
    displacement = evaluate_absorption_policy(
        btc_price=100.05, btc_open=100.0, up_price=0.5, up_open=0.5,
        k=1.0, b=0.0, disp_gate=6.0, under_gate=0.0,
    )
    under = evaluate_absorption_policy(
        btc_price=10002.0, btc_open=10000.0, up_price=0.5, up_open=0.5,
        k=1.0, b=0.0, disp_gate=2.0, under_gate=2.1,
    )

    assert displacement.reason == "DISPLACEMENT_GATE"
    assert displacement.eligible is False
    assert under.reason == "UNDER_REACTION_GATE"
    assert under.eligible is False


@pytest.mark.parametrize(
    ("avg_price", "eligible"),
    [(0.49, True), (0.50, False), (0.51, False)],
)
def test_market_guard_under_equal_above(avg_price: float, eligible: bool) -> None:
    result = evaluate_quote_execution_policy(
        order_type="MARKET", avg_price=avg_price, max_exec_price=0.50
    )

    assert result.eligible is eligible
    assert result.guard_applied is True
    assert result.reason == ("PASSED" if eligible else "EXEC_PRICE_GUARD")


def test_market_slippage_cap_matches_production_formula() -> None:
    result = evaluate_quote_execution_policy(
        order_type="MARKET", avg_price=0.45, max_exec_price=0.50
    )
    uncapped = evaluate_quote_execution_policy(
        order_type="MARKET", avg_price=0.20, max_exec_price=0.50
    )

    assert result.slippage_bps == int((0.50 / 0.45 - 1.0) * 10000)
    assert uncapped.slippage_bps == 1200


@pytest.mark.parametrize(
    ("avg_price", "eligible"),
    [(0.10, True), (0.199999, True), (0.20, False), (0.30, True), (0.40, False)],
)
def test_market_whitelist_is_lower_inclusive_upper_exclusive(
    avg_price: float, eligible: bool
) -> None:
    result = evaluate_quote_execution_policy(
        order_type="MARKET",
        avg_price=avg_price,
        max_exec_price=None,
        entry_bands=BANDS,
    )

    assert result.eligible is eligible
    assert result.whitelist_applied is True
    assert result.reason == ("PASSED" if eligible else "ENTRY_BAND_REJECTED")


@pytest.mark.parametrize("avg_price", [0.0, 0.50, 0.99])
def test_limit_skips_quote_guard_and_whitelist(avg_price: float) -> None:
    result = evaluate_quote_execution_policy(
        order_type="LIMIT",
        avg_price=avg_price,
        max_exec_price=0.50,
        entry_bands=BANDS,
    )

    assert result.eligible is True
    assert result.reason == "PASSED"
    assert result.guard_applied is False
    assert result.whitelist_applied is False
    assert result.slippage_bps == 1200


def test_config_fingerprint_uses_sanitized_canonical_json() -> None:
    config = {"z": 1, "api_key": "secret", "nested": {"enabled": True}}
    canonical = json.dumps(
        {"api_key": "[REDACTED]", "nested": {"enabled": True}, "z": 1},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    assert config_fingerprint(config) == hashlib.sha256(canonical.encode()).hexdigest()
