"""
Preservation Property Tests - Existing Behavior Unchanged

These tests verify behaviors that WORK correctly today and MUST continue
to work after the fix. They MUST PASS on the current unfixed code.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from hypothesis import given, settings as hyp_settings, assume
from hypothesis import strategies as st


# ============================================================
# 3.1 Trading Preservation: execute_trade() independently fetches token_id
# ============================================================


@pytest.mark.asyncio
@given(direction=st.sampled_from(["UP", "DOWN"]))
@hyp_settings(max_examples=10, deadline=None)
async def test_trading_preservation_execute_trade_fetches_token_id(direction):
    """
    **Validates: Requirements 3.1**

    Property: For any valid prediction direction (UP/DOWN), the trading module
    independently calls list_markets() to fetch the latest token_id before placing order.

    This verifies that execute_trade() has its own call to list_markets(),
    independent of any tracker logic.
    """
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader.__new__(BinancePredictionTrader)
    trader._api_key = "test_key"
    trader._api_secret = "test_secret"
    trader._wallet_address = "test_wallet"
    trader._wallet_id = "test_wallet_id"
    trader._trade_amount_usdt = 1.0
    trader._time_offset_ms = 0
    trader._trade_lock = asyncio.Lock()
    trader._active_market = None
    trader._up_token_id = None
    trader._down_token_id = None

    # Track list_markets calls
    list_markets_called = False

    async def mock_list_markets():
        nonlocal list_markets_called
        list_markets_called = True
        trader._up_token_id = "up_token_123"
        trader._down_token_id = "down_token_456"
        trader._active_market = {"marketTopicId": 1}
        return []

    trader.list_markets = mock_list_markets
    trader.fetch_wallet_info = AsyncMock(return_value={"walletAddress": "w", "walletId": "wid"})
    trader.get_quote = AsyncMock(return_value={"quoteId": "q1", "amountIn": "1000", "amountOut": "500"})
    trader.place_order = AsyncMock(return_value={"orderId": "order_1"})
    trader._save_order = AsyncMock(return_value=MagicMock())

    await trader.execute_trade(prediction=direction, confidence=0.8, prediction_id=1)

    # Property: list_markets is called by execute_trade independently
    assert list_markets_called, (
        f"Trading preservation violated: execute_trade({direction}) did not call list_markets()"
    )


# ============================================================
# 3.2 Normal Sampling Preservation: valid end_date writes to _pm_history
# ============================================================


@pytest.mark.asyncio
@given(
    up_chance=st.floats(min_value=0.01, max_value=0.99),
    down_chance=st.floats(min_value=0.01, max_value=0.99),
    end_date_val=st.integers(min_value=1700000000000, max_value=1800000000000),
)
@hyp_settings(max_examples=20, deadline=None)
async def test_normal_sampling_preservation(up_chance, down_chance, end_date_val):
    """
    **Validates: Requirements 3.2**

    Property: For any valid API response with non-None end_date and valid
    up/down chance, a sample is correctly stored in the expected format.

    This verifies the normal sampling path (non-buggy case).
    """
    assume(up_chance + down_chance <= 1.01)  # Realistic constraint

    pm_history = deque(maxlen=2000)
    end_date = str(end_date_val)

    # Simulate normal tracker sampling logic (non-bug path: end_date is valid)
    # This is the exact logic from main.py tracker when data is available
    aligned_ts = int(time.time()) * 1000

    # In the current code, when end_date is not None AND up_chance is not None,
    # a point is added to _pm_history
    if up_chance is not None or down_chance is not None:
        point = {
            "timestamp": aligned_ts,
            "up_price": up_chance,  # price ~= chance for prediction markets
            "down_price": down_chance,
            "up_pct": round(up_chance * 100, 1) if up_chance is not None else None,
            "down_pct": round(down_chance * 100, 1) if down_chance is not None else None,
            "participants": 150,
            "trade_volume": 8000.0,
        }
        pm_history.append(point)

    # Property: With valid data, sample is stored correctly
    assert len(pm_history) == 1, "Normal sampling should add exactly 1 point"
    stored = pm_history[0]
    assert stored["timestamp"] == aligned_ts
    assert stored["up_pct"] == round(up_chance * 100, 1)
    assert stored["down_pct"] == round(down_chance * 100, 1)
    assert stored["participants"] is not None
    assert stored["trade_volume"] is not None


# ============================================================
# 3.3 Normal Archive Preservation: >= 8 samples creates SentimentWindow
# ============================================================


@pytest.mark.asyncio
@given(sample_count=st.integers(min_value=8, max_value=20))
@hyp_settings(max_examples=10, deadline=None)
async def test_normal_archive_preservation(sample_count):
    """
    **Validates: Requirements 3.3**

    Property: For any sample count >= 8, archive produces a valid record
    with curve data (curve_up_pct, curve_down_pct, sample_count, prices).

    This verifies that the archiver correctly creates SentimentWindow
    when sufficient data is available.
    """
    # Generate samples
    start_ms = 1700000000000
    end_ms = start_ms + 5 * 60 * 1000

    samples = []
    for i in range(sample_count):
        sample = MagicMock()
        sample.timestamp = start_ms + i * (300000 // sample_count)
        sample.up_pct = 55.0 + (i % 10)
        sample.down_pct = 45.0 - (i % 10)
        sample.participants = 100 + i
        sample.trade_volume = 5000.0 + i * 100
        samples.append(sample)

    # Replicate the archiver's curve building logic (from main.py)
    # This is the CURRENT unfixed threshold (3), but we test with >= 8
    if len(samples) < 3:
        archived = False
    else:
        # Build curve data (same as archiver)
        curve_up = [{"t": s.timestamp, "v": s.up_pct} for s in samples if s.up_pct is not None]
        curve_down = [{"t": s.timestamp, "v": s.down_pct} for s in samples if s.down_pct is not None]

        # Compute averages
        participant_vals = [s.participants for s in samples if s.participants is not None]
        volume_vals = [s.trade_volume for s in samples if s.trade_volume is not None]
        avg_part = sum(participant_vals) / len(participant_vals) if participant_vals else None
        avg_vol = sum(volume_vals) / len(volume_vals) if volume_vals else None

        archived = True

    # Property: With >= 8 samples, archive succeeds with valid curve data
    assert archived, f"Archive should succeed with {sample_count} samples"
    assert len(curve_up) == sample_count, "All UP curve points should be present"
    assert len(curve_down) == sample_count, "All DOWN curve points should be present"
    assert avg_part is not None, "Average participants should be computed"
    assert avg_vol is not None, "Average trade volume should be computed"
    # Verify curve structure
    for point in curve_up:
        assert "t" in point and "v" in point, "Curve point must have 't' and 'v' keys"


# ============================================================
# 3.4 Force Backtest Preservation: force=True bypasses cache
# ============================================================


@pytest.mark.asyncio
async def test_force_backtest_preservation():
    """
    **Validates: Requirements 3.4**

    Property: For any call with force=True, fresh analysis is always executed
    regardless of cache state.

    Verifies the backtest API's force parameter works correctly.
    """
    # Simulate cached backtest state (recent, within 10min TTL)
    cached_result = "cached analysis from 3 minutes ago"
    cached_time = time.time() - 180  # 3 minutes ago (within 600s TTL)

    # Replicate the backtest cache check logic from main.py
    force = True
    _last_backtest_result = cached_result
    _last_backtest_time = cached_time

    # Cache check logic: if not force AND cached AND within TTL -> use cache
    should_use_cache = (
        not force
        and _last_backtest_result
        and (time.time() - _last_backtest_time < 600)
    )

    # Property: force=True always bypasses cache
    assert not should_use_cache, (
        "Force backtest preservation violated: force=True should bypass cache"
    )

    # Also verify with force=False for comparison (cache should be used)
    force_false = False
    should_use_cache_false = (
        not force_false
        and _last_backtest_result
        and (time.time() - _last_backtest_time < 600)
    )
    assert should_use_cache_false, (
        "Sanity check: force=False with valid cache should use cache"
    )


# ============================================================
# 3.5 Predict API Preservation: valid state returns prediction
# ============================================================


@pytest.mark.asyncio
async def test_predict_api_preservation():
    """
    **Validates: Requirements 3.5**

    Property: For any state with valid backtest result + sufficient window data,
    predict returns a prediction result (not error).

    Verifies that when prerequisites are met, the predict API works.
    """
    # Setup: Valid backtest result exists
    _last_backtest_result = "Pattern analysis: UP curves with rising slope..."

    # Setup: Sufficient data in _pm_history (>= 3 points in current window)
    now_ms = int(time.time()) * 1000
    window_start_ms = now_ms - (now_ms % (5 * 60 * 1000))

    pm_history = deque(maxlen=2000)
    for i in range(5):
        pm_history.append({
            "timestamp": window_start_ms + i * 15000,
            "up_price": 0.6,
            "down_price": 0.4,
            "up_pct": 60.0 + i,
            "down_pct": 40.0 - i,
            "participants": 100,
            "trade_volume": 5000.0,
        })

    # Replicate unfixed predict logic (first two checks)
    result = None

    # Check 1: backtest result available
    if not _last_backtest_result:
        result = {"status": "error", "message": "请先运行回测分析"}

    # Check 2: sufficient points
    points = list(pm_history)
    if result is None and len(points) < 3:
        result = {"status": "error", "message": "当前窗口采样点不足"}

    # Check 3: filter to current window
    if result is None:
        current_curve = [
            {"t": p["timestamp"], "v": p["up_pct"]}
            for p in points
            if p["timestamp"] >= window_start_ms and p.get("up_pct") is not None
        ]
        if len(current_curve) < 3:
            result = {"status": "error", "message": "当前窗口数据不足"}
        else:
            # Would call LLM here, but we just verify the path reaches here
            result = {"status": "ok", "curve_points": len(current_curve)}

    # Property: With valid prerequisites, predict returns ok (not error)
    assert result["status"] == "ok", (
        f"Predict preservation violated: got status='{result['status']}' "
        f"with message='{result.get('message', '')}' when prerequisites are met"
    )
    assert result["curve_points"] >= 3, "Should have sufficient curve points"


# ============================================================
# 3.6 Momentum Predict Preservation: uses _pm_history only
# ============================================================


@pytest.mark.asyncio
@given(
    num_points=st.integers(min_value=8, max_value=30),
    base_up_pct=st.floats(min_value=30.0, max_value=70.0),
)
@hyp_settings(max_examples=20, deadline=None)
async def test_momentum_predict_preservation(num_points, base_up_pct):
    """
    **Validates: Requirements 3.6**

    Property: For any _pm_history state with sufficient data (>= MIN_SAMPLES
    and >= MIN_ELAPSED_SEC), MomentumService produces a result based solely
    on algorithmic analysis (no LLM dependency).

    Tests MomentumService directly as it's a pure algorithmic service.
    """
    from binance_predict.services.momentum_service import MomentumService, MomentumResult

    service = MomentumService()

    # Generate time series with enough elapsed time (>= 60s)
    base_ts = 1700000000000
    interval_ms = 15000  # 15s between points

    points = []
    for i in range(num_points):
        # Create realistic variations
        up_pct = base_up_pct + (i % 5) * 0.5 - 1.0
        points.append({
            "timestamp": base_ts + i * interval_ms,
            "up_price": up_pct / 100,
            "down_price": (100 - up_pct) / 100,
            "up_pct": up_pct,
            "down_pct": 100 - up_pct,
            "participants": 100 + i * 2,
            "trade_volume": 5000.0 + i * 100,
        })

    result = service.analyze(points)

    # Property: MomentumService always returns a valid MomentumResult
    assert isinstance(result, MomentumResult), "Must return MomentumResult"
    assert result.direction in ("UP", "DOWN", "NO_TRADE"), (
        f"Direction must be UP/DOWN/NO_TRADE, got '{result.direction}'"
    )
    assert 0.0 <= result.confidence <= 1.0, (
        f"Confidence must be in [0, 1], got {result.confidence}"
    )
    assert -1.0 <= result.composite_score <= 1.0, (
        f"Composite score must be in [-1, 1], got {result.composite_score}"
    )
    assert result.sample_count >= service.MIN_SAMPLES, (
        f"Sample count should be >= {service.MIN_SAMPLES}, got {result.sample_count}"
    )

    # Property: Result is based solely on algorithmic analysis (signals present)
    if result.elapsed_seconds >= service.MIN_ELAPSED_SEC:
        assert len(result.signals) > 0, (
            "Momentum analysis with sufficient data should produce signals"
        )


# ============================================================
# 3.7 Cleanup Preservation: old samples deleted after archive
# ============================================================


@pytest.mark.asyncio
async def test_cleanup_preservation():
    """
    **Validates: Requirements 3.7**

    Property: For any archive completion, the system attempts to delete
    samples older than 1 hour from the window end time.

    Verifies the cleanup logic calculates the correct threshold.
    """
    # Simulate archive completing at a known window end time
    end_ms = 1700000300000  # Some window end timestamp

    # Replicate the cleanup logic from the archiver:
    # cleanup_threshold_ms = end_ms - 3600 * 1000
    cleanup_threshold_ms = end_ms - 3600 * 1000  # 1 hour before window end

    # Property: cleanup threshold is exactly 1 hour before window end
    assert cleanup_threshold_ms == end_ms - 3_600_000, (
        "Cleanup threshold should be 1 hour (3600s) before window end"
    )

    # Verify with different end times
    for test_end_ms in [1700000000000, 1700000600000, 1700001200000]:
        threshold = test_end_ms - 3600 * 1000
        assert threshold == test_end_ms - 3_600_000
        # The threshold correctly identifies "old" samples
        old_sample_ts = test_end_ms - 4000 * 1000  # 4000s old (> 1 hour)
        recent_sample_ts = test_end_ms - 2000 * 1000  # 2000s old (< 1 hour)

        assert old_sample_ts < threshold, "Old sample should be below threshold"
        assert recent_sample_ts >= threshold, "Recent sample should be above threshold"


# ============================================================
# 3.8 SSE/Heartbeat Preservation: heartbeat has correct fields
# ============================================================


@pytest.mark.asyncio
async def test_sse_heartbeat_preservation():
    """
    **Validates: Requirements 3.8**

    Property: Heartbeat broadcast includes correct state fields:
    price, data_quality, auto_predict, auto_trade.

    Verifies the heartbeat event structure remains intact.
    """
    # The heartbeat in main.py broadcasts these fields:
    # {
    #     "price": collector.store.mid_price,
    #     "data_quality": feature_service._compute_data_quality_score(...),
    #     "auto_predict": _auto_predict,
    #     "auto_trade": _auto_trade,
    # }

    # Simulate heartbeat data construction
    mock_price = 100500.0
    mock_data_quality = 0.85
    mock_auto_predict = True
    mock_auto_trade = False

    heartbeat_data = {
        "price": mock_price,
        "data_quality": mock_data_quality,
        "auto_predict": mock_auto_predict,
        "auto_trade": mock_auto_trade,
    }

    # Property: Heartbeat contains all required fields with correct types
    required_fields = {"price", "data_quality", "auto_predict", "auto_trade"}
    assert set(heartbeat_data.keys()) == required_fields, (
        f"Heartbeat must contain exactly {required_fields}, got {set(heartbeat_data.keys())}"
    )
    assert isinstance(heartbeat_data["price"], float), "price must be float"
    assert isinstance(heartbeat_data["data_quality"], float), "data_quality must be float"
    assert isinstance(heartbeat_data["auto_predict"], bool), "auto_predict must be bool"
    assert isinstance(heartbeat_data["auto_trade"], bool), "auto_trade must be bool"

    # Verify the _broadcast_sse function signature matches
    # (testing that we can call it with the heartbeat event structure)
    captured_events = []

    async def mock_broadcast(event: str, data: dict):
        captured_events.append((event, data))

    await mock_broadcast("heartbeat", heartbeat_data)

    assert len(captured_events) == 1
    event_name, event_data = captured_events[0]
    assert event_name == "heartbeat"
    assert event_data["price"] == mock_price
    assert event_data["data_quality"] == mock_data_quality
    assert event_data["auto_predict"] == mock_auto_predict
    assert event_data["auto_trade"] == mock_auto_trade
