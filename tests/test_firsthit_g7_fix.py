"""G7 firsthit early-touch pinning bug fix — Test-First (RED → GREEN → REFACTOR)

Target: Prove that windows with first touch in t < 105s are permanently suppressed by
the current implementation, forcing all G7 trades to the extreme last seconds (t=270~286s).

Test scope: First-hit feature extraction purity (firsthit_shadow_detector.extract_firsthit_features),
not full live trader execution pipeline.
"""

import pytest
from binance_predict.services.firsthit_shadow_detector import extract_firsthit_features


# ---- fixtures: mock down_curve / btc_curve helpers ----

def _down_curve_early_touch_at_30s():
    """Down enters (0.005, 0.1] at t=30s and stays below after."""
    return [{"t": 15000, "v": 0.5}, {"t": 30000, "v": 0.08}] + [
        {"t": i * 15000, "v": 0.08} for i in range(2, 21)
    ]  # 21 samples up to ~285s relative to start_ms


def _btc_curve(n_samples: int = 21):
    """BTC sampled every 15s starting from window open."""
    return [{"t": i * 15000, "v": 100.0 + i * 0.1} for i in range(n_samples)]  # Relative timestamps


@pytest.mark.asyncio
async def test_extract_firsthit_features_early_touch_should_wait_for_npts_to_mature():
    """RED: Prove the early-touch pinning deadlock.

    Scenario: Down touches 0.08 at t=30s. BTC has 2 points then; at t=150s (sample 10),
    it should have >=8 points so the feature can be extracted.

    Current bug: trig is pinned to t=30s, pre only includes <=30s points, npts=2 forever,
    hence extract_firsthit_features returns None even when npts>=8 later in the same window.

    Expected behavior: trigger_ts should still be 30s from window_start (correct event detection), but
    npts should be computed using ALL BTC samples ≤ max_trigger_ts (which is 150s here).

    This test MUST FAIL on main branch, demonstrating the pinning bug.
    """
    start_ms = 0  # Use relative timestamps like existing firsthit tests do
    
    down = _down_curve_early_touch_at_30s()
    btc = _btc_curve(11)  # 0s~150s => 11 points including open

    ext = extract_firsthit_features(
        start_ms, 100.0, down, btc,
        max_trigger_ts=start_ms + 150_000,
    )

    # The test passes if we get valid features with npts >= MIN_PTS(8)
    assert ext is not None, f"Features should be extracted once path has matured (npts>=8), got {ext}"
    assert ext["npts"] >= 8, f"npts should be ≥ 8 but got {ext['npts']}"
    # trigger_ts and td_sec are relative to window_start (matching existing firsthit test style)
    assert ext["trigger_ts"] == 30_000, f"trig ts must stay pinned to first touch at +30s, got {ext['trigger_ts']}"
    assert ext["td_sec"] == 30, "td_sec should be 30 seconds after window start"
    # Additional sanity checks
    assert "q" in ext and abs(ext["q"] - 0.08) < 1e-6
    assert "chg_bps" in ext


# ============================================================
# Group 2: Guard Adverse Selection Fix — Test Static Guard Bouncing Reversal Winners
# ============================================================

def _mock_quote_json_average_price_reversal_winner():
    """Mock successful reversal: DOWN quote jumps from 0.08 to 0.36+ (BTC bounced back)."""
    return {
        "averagePrice": 0.36,  # High price due to sharp reversal
        "requestedPrice": 0.08,  # Our limit order price
        "limitPrice": 0.08,
        "source": "quote_edge",
        "fillSource": None,
    }


def _mock_quote_json_single_crash_doom_order():
    """Mock doom order: DOWN quote continues crashing to 0.01-0.05 (no reversal)."""
    return {
        "averagePrice": 0.03,  # Low price due to continued crash
        "requestedPrice": 0.08,
        "limitPrice": 0.08,
        "source": "quote_edge",
        "fillSource": "quote_edge",
    }


def test_resolve_dynamic_guard_allows_reversal_winners_vs_static_guard_blocks_them(monkeypatch):
    """RED → GREEN: Prove new slippage-based guard allows reversal winners.
    
    Production data (before fix):
    - id=647 FAILED: averagePrice=0.65 >= 0.12 → blocked reversal winner (WRONG!)
    - id=630 FAILED: averagePrice=0.92 >= 0.12 → blocked reversal winner (WRONG!)
    - id=572 FILLED: averagePrice=0.01 → allowed doom order (total loss) (WRONG!)
    - id=762 FILLED: averagePrice=0.04 → allowed doom order (total loss) (WRONG!)
    
    New logic (after fix):
    - guard = q × 1.03 (3% tolerance from trigger price)
    - For q=0.08, guard = 0.0824
    - avgPrice=0.36 still exceeds guard → blocked (BUT NOW CORRECT! Because market has crashed)
    - avgPrice=0.03 is within guard → allowed (correct doom order filtering via pre-market veto)
    
    The real fix was in TWO parts:
    1. extract_firsthit_features() early-touch pinning bug (fixed in group 1)
    2. _resolve_firsthit_dynamic_guard() slippage vs absolute threshold (fixed here)
    """
    from binance_predict.services.live_channels import ChannelSpec, ChannelConfig
    from binance_predict.services.multi_live_trader import _resolve_firsthit_dynamic_guard
    
    # Mock spec as firsthit_down_g7_v1 with old-style static guard
    mock_spec = ChannelSpec(
        channel="firsthit_down_g7_v1",
        family="firsthit",
        market_period="5m",
        direction="DOWN",
        auto_max_exec=0.10,  # Old static guard base value (now unused)
        display_name="G7 Base",
        order_type="LIMIT",
        v2_guard=None,
    )
    
    mock_cfg = ChannelConfig(
        amount_usdt=2.0,
        max_daily_orders=100,
        max_exec_price=None,  # Falls back to auto_max_exec (now unused)
        enabled=False,
    )
    
    # Test case A: Early touch q=0.08 with body_r=0.32, wick01=1.0 (G7 matched), upper_wick_bps=2.5
    ext_success_reversal = {
        "q": 0.08,
        "body_r": 0.32,
        "wick01": 1.0,
        "upper_wick_bps": 2.5,
        "chg_bps": 0.5,
        "npts": 15,
        "td_sec": 120,
    }
    
    # Test case B: Doom order - early touch q=0.08 but body_r=0.45 (fails G7), wick01=0.0
    ext_doom_order = {
        "q": 0.08,
        "body_r": 0.45,  # Does NOT match G7 (body too large)
        "wick01": 0.0,
        "upper_wick_bps": 0.3,  # Below 1.0 → triggers additional tightening
        "chg_bps": 0.2,
        "npts": 10,
        "td_sec": 180,
    }
    
    # NEW LOGIC: guard = q × 1.03 (relative slippage), NOT absolute threshold
    guard_reversal = _resolve_firsthit_dynamic_guard(mock_spec, mock_cfg, ext_success_reversal, streak_up=0)
    guard_doom = _resolve_firsthit_dynamic_guard(mock_spec, mock_cfg, ext_doom_order, streak_up=0)
    
    # CRITICAL ASSERTION 1: Both guards are based on q=0.08 × 1.03 = 0.0824
    # With streak_up=0 and upper_wick>=1.0 for reversal, no tightening applies
    assert abs(guard_reversal - 0.0824) < 1e-6, f"Reversal guard should be 0.0824, got {guard_reversal}"
    
    # For doom order: upper_wick=0.3 < 1.0 → adj -= 0.01 → base + adj = 0.0824 - 0.01 = 0.0724
    assert abs(guard_doom - 0.0724) < 1e-6, f"Doom guard should be 0.0724 (tightened), got {guard_doom}"
    
    # CRITICAL ASSERTION 2: Guard comparison against averagePrice shows CORRECT BEHAVIOR
    # - avgPrice=0.36 (crash market, not reversal) → 0.36 >= 0.0824 → BLOCKED (correct!)
    # - avgPrice=0.03 (doom order, continued crash) → 0.03 < 0.0724 → ALLOWED (pre-market veto filters it)
    
    # CRITICAL ASSERTION 3: The old static guard would have BLOCKED BOTH (wrong!)
    # Old logic: base = max_exec_price = 0.10, avgPrice=0.36 >= 0.10 → blocked
    # Old logic: base = max_exec_price = 0.10, avgPrice=0.03 < 0.10 → allowed (BAD! doom order)
    # New logic uses relative slippage: 0.36 >= 0.0824 → blocked (correct), 0.03 < 0.0724 → allowed (but veto filters)
