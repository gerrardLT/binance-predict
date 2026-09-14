import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from binance_predict.services.shadow_execution_analytics import (
    build_execution_comparison,
    build_execution_comparison_from_rows,
)


SCOPE = {
    "policy_version": None, "from": None, "to": None,
    "market_period": None, "include_retired": False,
}


def assessment(id, **overrides):
    values = dict(
        id=id, source_id=id, signal_version="quote_momentum_v1",
        policy_version="policy-a", target_window_start=id * 1_000,
        market_period="5m", direction="UP", strategy_eligible=True,
        operational_eligible=True, execution_eligible=True,
        terminal_stage="EXECUTION", reason_code="PASSED",
        channel_mapped=True, retired=False, theoretical_win=True,
        theoretical_outcome="UP", theoretical_entry_price=0.5,
        theoretical_realized_return=1.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def order(id, assessment_id, **overrides):
    values = dict(
        id=id, assessment_id=assessment_id, status="FILLED",
        order_id=f"BINANCE-{id}",
        amount_in="1000000000000000000", pnl=0.5, settled_at=object(),
        created_at=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def aggregate(rows=(), orders=()):
    return build_execution_comparison_from_rows(rows, orders, scope=SCOPE.copy())


class Result:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def all(self):
        return self.rows


@pytest.mark.asyncio
async def test_empty_response_has_null_rates_and_is_json_serializable():
    session = SimpleNamespace(execute=AsyncMock(return_value=Result([])))

    result = await build_execution_comparison(session)

    assert result["funnel"] == {key: 0 for key in result["funnel"]}
    assert all(gap == {"n": 0, "rate": None} for gap in result["gaps"].values())
    assert result["returns"]["theoretical_fixed_1u"] == {"n": 0, "sum": 0, "avg": None}
    assert result["returns"]["actual"] == {"settled_n": 0, "stake": 0.0, "pnl": 0.0, "roi": None}
    assert result["versions"] == []
    assert result["curves"] == []
    json.dumps(result)
    session.execute.assert_awaited_once()


def test_unknown_and_no_mapping_are_isolated_from_rejection_rates():
    rows = [
        assessment(1, terminal_stage="UNKNOWN", operational_eligible=None, execution_eligible=None),
        assessment(2, channel_mapped=False, reason_code="NO_LIVE_MAPPING",
                   strategy_eligible=False, operational_eligible=None, execution_eligible=None),
    ]

    result = aggregate(rows)

    assert result["funnel"]["unknown"] == 1
    assert result["funnel"]["no_live_mapping"] == 1
    assert result["gaps"]["selection"] == {"n": 0, "rate": 0.0}
    assert result["gaps"]["coverage"] == {"n": 1, "rate": 1.0}


def test_null_mapping_is_unknown_not_mapped_or_no_mapping():
    result = aggregate([
        assessment(1, channel_mapped=None, terminal_stage="SOURCE",
                   strategy_eligible=False, operational_eligible=None,
                   execution_eligible=None),
    ])

    assert result["funnel"]["unknown"] == 1
    assert result["funnel"]["no_live_mapping"] == 0
    assert result["gaps"]["selection"] == {"n": 0, "rate": None}
    assert result["gaps"]["coverage"] == {"n": 0, "rate": None}


def test_executable_returns_are_a_paired_subset():
    rows = [
        assessment(1, theoretical_realized_return=2.0, execution_eligible=True),
        assessment(2, theoretical_realized_return=-1.0, execution_eligible=False),
        assessment(3, theoretical_realized_return=None, execution_eligible=True),
    ]

    returns = aggregate(rows)["returns"]

    assert returns["theoretical_fixed_1u"] == {"n": 2, "sum": 1.0, "avg": 0.5}
    assert returns["executable_fixed_1u"] == {"n": 1, "sum": 2.0, "avg": 2.0}


def test_actual_return_converts_wei_and_uses_stake_weighted_roi():
    rows = [assessment(1), assessment(2)]
    orders = [
        order(1, 1, amount_in="2000000000000000000", pnl=1.0),
        order(2, 2, amount_in="500000000000000000", pnl=-0.25),
    ]

    actual = aggregate(rows, orders)["returns"]["actual"]

    assert actual == {"settled_n": 2, "stake": 2.5, "pnl": 0.75, "roi": 0.3}


def test_gap_denominators_follow_each_funnel_transition():
    rows = [
        assessment(1, strategy_eligible=False, operational_eligible=None, execution_eligible=None),
        assessment(2, operational_eligible=False, execution_eligible=None),
        assessment(3, execution_eligible=False),
        assessment(4), assessment(5), assessment(6),
    ]
    orders = [
        order(4, 4, status="FAILED", settled_at=None, pnl=None),
        order(5, 5, status="FILLED", settled_at=None, pnl=None),
        order(6, 6),
    ]

    gaps = aggregate(rows, orders)["gaps"]

    assert gaps["selection"] == {"n": 1, "rate": 1 / 6}
    assert gaps["operational"] == {"n": 1, "rate": 0.2}
    assert gaps["execution"] == {"n": 1, "rate": 0.25}
    assert gaps["fill"] == {"n": 1, "rate": 1 / 3}
    assert gaps["settlement"] == {"n": 1, "rate": 0.5}
    assert gaps["coverage"] == {"n": 0, "rate": 0.0}


def test_pre_submission_placeholder_is_not_counted_as_submitted_or_fill_gap():
    rows = [assessment(
        1,
        execution_eligible=False,
        terminal_stage="EXECUTION",
        reason_code="EXEC_PRICE_GUARD",
    )]
    orders = [order(
        1,
        1,
        status="FAILED",
        order_id=None,
        amount_in="0",
        pnl=None,
        settled_at=None,
    )]

    result = aggregate(rows, orders)

    assert result["funnel"]["submitted"] == 0
    assert result["funnel"]["filled"] == 0
    assert result["gaps"]["fill"] == {"n": 0, "rate": None}


def test_ghost_settlement_without_exchange_order_is_excluded_from_actuals():
    result = aggregate(
        [assessment(1)],
        [order(1, 1, order_id=None, amount_in="9000000000000000000", pnl=8.0)],
    )

    assert result["funnel"]["submitted"] == 0
    assert result["funnel"]["filled"] == 0
    assert result["funnel"]["settled"] == 0
    assert result["gaps"]["fill"] == {"n": 0, "rate": None}
    assert result["gaps"]["settlement"] == {"n": 0, "rate": None}
    assert result["returns"]["actual"] == {
        "settled_n": 0, "stake": 0.0, "pnl": 0.0, "roi": None,
    }
    assert result["curves"][0]["actual_stake"] == 0.0
    assert result["curves"][0]["actual_pnl"] == 0.0
    assert result["curves"][0]["actual_roi"] is None


def test_exchange_backed_order_outranks_ghost_settlement_for_same_assessment():
    result = aggregate(
        [assessment(1)],
        [
            order(99, 1, order_id=None, amount_in="9000000000000000000", pnl=8.0),
            order(1, 1, status="PENDING", settled_at=None, pnl=None),
        ],
    )

    assert result["funnel"]["submitted"] == 1
    assert result["funnel"]["filled"] == 0
    assert result["funnel"]["settled"] == 0
    assert result["returns"]["actual"]["settled_n"] == 0


def test_settlement_fields_require_filled_status_for_actuals():
    result = aggregate([assessment(1)], [order(1, 1, status="FAILED")])

    assert result["funnel"]["submitted"] == 1
    assert result["funnel"]["filled"] == 0
    assert result["funnel"]["settled"] == 0
    assert result["returns"]["actual"]["settled_n"] == 0


def test_duplicate_orders_choose_settled_then_filled_then_latest_id():
    rows = [assessment(1), assessment(2), assessment(3)]
    orders = [
        order(99, 1, status="FAILED", settled_at=None, pnl=None),
        order(1, 1, status="FILLED", pnl=0.4),
        order(2, 2, status="FILLED", settled_at=None, pnl=None),
        order(3, 2, status="FAILED", settled_at=None, pnl=None),
        order(4, 3, status="FAILED", settled_at=None, pnl=None),
        order(5, 3, status="PENDING", settled_at=None, pnl=None),
    ]

    result = aggregate(rows, orders)

    assert result["funnel"]["submitted"] == 3
    assert result["funnel"]["filled"] == 2
    assert result["returns"]["actual"]["settled_n"] == 1
    assert result["returns"]["actual"]["pnl"] == 0.4


def test_duplicate_exchange_order_is_counted_once_across_assessments():
    rows = [assessment(1), assessment(2)]
    orders = [
        order(1, 1, order_id="SAME-BINANCE", pnl=0.5),
        order(2, 2, order_id="SAME-BINANCE", pnl=0.5),
    ]

    result = aggregate(rows, orders)

    assert result["funnel"]["submitted"] == 1
    assert result["funnel"]["filled"] == 1
    assert result["funnel"]["settled"] == 1
    assert result["returns"]["actual"] == {
        "settled_n": 1, "stake": 1.0, "pnl": 0.5, "roi": 0.5,
    }


def test_versions_separate_policies_and_follow_registry_then_unknown_order():
    rows = [
        assessment(1, signal_version="zzz_unknown", policy_version="policy-a"),
        assessment(2, signal_version="quote_momentum_v1", policy_version="policy-b"),
        assessment(3, signal_version="quote_momentum_v1", policy_version="policy-a"),
    ]

    versions = aggregate(rows)["versions"]

    assert [(v["signal_version"], v["policy_version"]) for v in versions] == [
        ("quote_momentum_v1", "policy-a"),
        ("quote_momentum_v1", "policy-b"),
        ("zzz_unknown", "policy-a"),
    ]
    assert [v["funnel"]["theoretical"] for v in versions] == [1, 1, 1]


def test_curves_are_deterministic_and_actual_only_advances_on_settlement():
    rows = [
        assessment(20, target_window_start=1_000, signal_version="zzz_unknown",
                   theoretical_realized_return=2.0),
        assessment(10, target_window_start=1_000, signal_version="quote_momentum_v1",
                   theoretical_realized_return=1.0),
        assessment(30, target_window_start=2_000, signal_version="quote_momentum_v1",
                   theoretical_realized_return=-1.0, execution_eligible=False),
    ]
    orders = [
        order(1, 10, amount_in="2000000000000000000", pnl=1.0),
        order(2, 20, status="FILLED", settled_at=None, pnl=None),
        order(3, 30, status="FAILED", settled_at=None, pnl=None),
    ]

    curves = aggregate(rows, orders)["curves"]

    assert [point["signal_version"] for point in curves] == [
        "quote_momentum_v1", "zzz_unknown", "quote_momentum_v1",
    ]
    assert curves[0]["theoretical_fixed_1u"] == 1.0
    assert curves[1]["theoretical_fixed_1u"] == 3.0
    assert curves[2]["theoretical_fixed_1u"] == 2.0
    assert curves[2]["executable_fixed_1u"] == 3.0
    assert curves[0]["actual_pnl"] == curves[2]["actual_pnl"] == 1.0
    assert curves[0]["actual_stake"] == curves[2]["actual_stake"] == 2.0
    assert curves[2]["actual_roi"] == 0.5
