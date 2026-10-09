import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from binance_predict.services.shadow_forward_evidence import PROFILE_HASH, quote_returns


@pytest.mark.asyncio
async def test_freeze_first_write_preserved(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as module
    row = SimpleNamespace(id=1, config_snapshot={})
    monkeypatch.setattr(module, "get_or_create_runtime_assessment", AsyncMock(return_value=row))
    monkeypatch.setattr(module, "shadow_capacity", AsyncMock())
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalar_one=lambda: row)))
    snapshot = {"window_start": 1000, "window_end": 301000, "trigger_ts": 2000, "side": "DOWN"}
    await module.freeze_event(session, snapshot, "LIVE", {"1": {"available": False}}, "absorption_follow_td120_v1", True)
    original = dict(row.input_snapshot)
    await module.freeze_event(session, {**snapshot, "trigger_ts": 9999}, "RESTORED", {}, "absorption_follow_td120_v1", False)
    assert row.input_snapshot == original
    assert row.strategy_eligible is True
    assert row.quote_snapshot["fill"] == "NOT_OBSERVED"


@pytest.mark.asyncio
async def test_quote_queue_capacity_and_cancellation():
    from binance_predict.services.shadow_forward_evidence import QuoteBudget
    budget = QuoteBudget()
    budget.pending = 8
    with pytest.raises(RuntimeError, match="quote_queue_full"):
        await budget.__aenter__()
    assert budget.pending == 8
    budget.pending = 0
    await budget.__aenter__()
    waiter = asyncio.create_task(budget.__aenter__())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await budget.__aexit__()
    assert budget.pending == 0


def test_operations_reject_unknown_and_no_mutation():
    from binance_predict.services.shadow_forward_evidence import operational_constraints
    inputs = dict(amount=5, available_balance=10, daily_count=0, max_daily_orders=20, exclusive_known=True)
    assert operational_constraints(**inputs)["eligible"] is True
    for field, value, reason in (("available_balance", 4, "INSUFFICIENT_BALANCE"),
                                 ("daily_count", 20, "DAILY_CAPACITY_REACHED"),
                                 ("exclusive_blocker", "other", "EXCLUSIVE_BLOCKED")):
        result = operational_constraints(**{**inputs, field: value})
        assert result["eligible"] is False and reason in result["reasons"]
        assert result["mutates_live"] is False
    assert operational_constraints(**{**inputs, "daily_count": None})["eligible"] is None
    assert inputs["daily_count"] == 0


@pytest.mark.asyncio
async def test_daily_cache_unknown_refresh_and_day_roll(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as module
    monkeypatch.setattr(module, "_daily_counts", {})
    monkeypatch.setattr(module, "_daily_tasks", {})
    clock = [86400.]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    counter = AsyncMock(return_value=3)
    assert module.cached_daily_count("channel", counter) is None
    await asyncio.gather(*module._daily_tasks.values())
    assert module.cached_daily_count("channel", counter) == 3
    clock[0] += 86400
    assert module.cached_daily_count("channel", counter) is None
    await asyncio.gather(*module._daily_tasks.values())
    assert counter.await_count == 2


@pytest.mark.asyncio
async def test_cached_quote_preserves_original_timestamp(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as module
    client = SimpleNamespace(_wallet_address="address", _5m_start_date=1000,
        _down_token_id="token", list_markets=AsyncMock(),
        get_quote=AsyncMock(return_value={"averagePrice": .2}))
    monkeypatch.setattr(module, "_quote_client", client)
    monkeypatch.setattr(module, "_quote_cache", {})
    monkeypatch.setattr(module, "_live_busy", None)
    monkeypatch.setattr(module, "current_amount", lambda version: None)
    clock = [1.1]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    first = await module.probe_ladder("a", "DOWN", 1000, "5m")
    clock[0] = 1.2
    second = await module.probe_ladder("b", "DOWN", 1000, "5m")
    assert client.get_quote.await_count == 3
    assert first["1"]["received_at"] == second["1"]["received_at"] == 1100
    from binance_predict.services.shadow_forward_evidence import PROFILE
    row = {"config_snapshot": {"profile": PROFILE, "hash": PROFILE_HASH},
           "input_snapshot": {"capture_mode": "LIVE"}, "strategy_eligible": True,
           "decision_snapshot": {"shadow_capacity": {"accepted": True}},
           "theoretical_outcome": "DOWN", "direction": "DOWN", "trigger_ts": 1200,
           "quote_snapshot": {"ladder": second}}
    assert module.quote_returns(row) == {}  # Cached pre-trigger quote cannot become new evidence.


def test_capture_runtime_profile_changes_cohort(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as module
    config = SimpleNamespace(amount_usdt=3, max_exec_price=.4, max_daily_orders=20, order_type="MARKET")
    monkeypatch.setattr(module, "_runtime_configs", lambda: {"firsthit_down_chg_v2": config})
    event = {"window_start": 1000, "trigger_ts": 1100}
    first = module.capture_event(event, "firsthit_down_chg_v2")
    config.amount_usdt = 5
    second = module.capture_event(event, "firsthit_down_chg_v2")
    assert first["_frozen"]["cohort"] != second["_frozen"]["cohort"]
    assert first["_frozen"]["profile"]["amount_usdt"] == 3


def test_overflow_and_capture_exception_never_escape(monkeypatch):
    from binance_predict.services import shadow_forward_evidence as module
    monkeypatch.setattr(module, "OVERFLOW_COUNTS", {})
    monkeypatch.setattr(module, "_FORWARD_TASKS", set(range(32)))
    module.observe_kline({"version": "krev_a_v1"})
    assert module.OVERFLOW_COUNTS["krev_a_v1"] == 1
    monkeypatch.setattr(module, "_FORWARD_TASKS", set())
    monkeypatch.setattr(module, "capture_event", lambda *args: (_ for _ in ()).throw(ValueError("injected")))
    module.observe_kline({"version": "krev_a_v1", "market_start": 1000})
    assert module.OVERFLOW_COUNTS["krev_a_v1"] == 2
    assert not module._FORWARD_TASKS


def test_amount_out_requires_positive_integer():
    from binance_predict.services.shadow_forward_evidence import valid_amount_out
    assert valid_amount_out("1000000000000000000")
    for value in (None, "NaN", "Infinity", "-1", "0", "1.5", "invalid"):
        assert not valid_amount_out(value)


def test_quotes_are_not_fills_and_archive_is_not_forward():
    from binance_predict.services.shadow_forward_evidence import PROFILE
    row = {"config_snapshot": {"hash": PROFILE_HASH, "profile": PROFILE},
           "decision_snapshot": {"shadow_capacity": {"accepted": True}}, "input_snapshot": {"capture_mode": "LIVE"},
           "strategy_eligible": True, "direction": "DOWN", "theoretical_outcome": "DOWN",
           "trigger_ts": 1000, "quote_snapshot": {"ladder": {"5": {
               "available": True, "average_price": .2, "amount_usdt": 5, "received_at": 1100}}}}
    result = quote_returns(row)["5"]
    assert abs(result["pnl"] - 19.5) < 1e-10
    assert result["stress_3pct"] < result["stress_1pct"] < result["pnl"]
    assert result["fill"] is False
    assert result["counterfactual_amount"] is True
    assert result["capacity_passed_for_amount"] is False
    assert result["amount_out_valid"] is False
    assert result["fees_verified"] is False and result["admission"] is False
    row["input_snapshot"]["capture_mode"] = "ARCHIVE_FALLBACK"
    assert quote_returns(row) == {}
    row["input_snapshot"]["capture_mode"] = "LIVE"
    for price in (.5, .51, float("nan"), float("inf"), 0):
        row["quote_snapshot"]["ladder"]["5"]["average_price"] = price
        assert quote_returns(row) == {}
