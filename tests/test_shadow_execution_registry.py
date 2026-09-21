from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from binance_predict.services.shadow_execution_adapters import SHADOW_SOURCE_ADAPTERS
from binance_predict.services.shadow_execution_registry import (
    SHADOW_VERSION_SPECS,
    SHADOW_VERSIONS,
)
from binance_predict.services.shadow_execution_types import (
    EXECUTION_POLICY_VERSION,
    SourceType,
)


EXPECTED_ANALYTICS_VERSIONS = (
    "x4_v1", "quote_momentum_v1", "quote_contrarian_v1", "x4_v2",
    "quote_momentum_v2", "quote_contrarian_v2", "x4_v3",
    "quote_contrarian_v3a", "quote_contrarian_v3b", "quote_contrarian_v4",
    "late_night_contrarian_v1", "late_night_contrarian_v2", "krev_a_v1",
    "krev_b_v1", "hm_touch_down_v1", "hm_touch_down_v2", "rev_p1_v1",
    "rev_p2_v1", "nb_zschamp_15m_v1", "nb_smaslope_5m_v1", "combo_p1_v1",
    "combo_p2_v1", "combo_p3_v1", "combo_p4_v1", "combo_p5_v1",
    "s5_deep_z20_v1", "quote_momentum_v3", "absorption_follow_td120_v1",
    "absorption_follow_td150_v1", "s2_cond_t4_v1", "s2_cond_t5d_v1",
    "scene_bear_exhaust_opt_v1", "firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1",
    "firsthit_down_g4_v1", "firsthit_down_g7_v1", "g7_streak_v1",
    "g7_wick20_v1", "g7_strict_v1", "g7_q05_v1", "g7_t270_v1",
    "hm_inside_15m_v2", "ih_inside_15m_v2", "hm_inside_5m_v2",
    "candle_hm_bull_5m_event_v1", "candle_hm_bull_15m_event_v1",
)


def test_registry_matches_all_physical_shadow_versions_in_order() -> None:
    assert SHADOW_VERSIONS == EXPECTED_ANALYTICS_VERSIONS
    assert len(SHADOW_VERSION_SPECS) == 47
    for version in ("candle_hm_bull_5m_event_v1", "candle_hm_bull_15m_event_v1"):
        spec = SHADOW_VERSION_SPECS[version]
        assert spec.family == "candlestick_reversal"
        assert spec.role == "EVENT"
        assert spec.live_channel is None


def test_registry_covers_every_source_adapter() -> None:
    assert {adapter.source_type for adapter in SHADOW_SOURCE_ADAPTERS} == set(SourceType)
    assert {spec.source_type for spec in SHADOW_VERSION_SPECS.values()} == set(SourceType)


def test_registry_separates_signal_and_execution_policy_versions() -> None:
    assert all(spec.version == version for version, spec in SHADOW_VERSION_SPECS.items())
    assert {spec.policy_version for spec in SHADOW_VERSION_SPECS.values()} == {
        EXECUTION_POLICY_VERSION
    }
    assert EXECUTION_POLICY_VERSION not in SHADOW_VERSION_SPECS


def test_rev2_versions_are_not_excluded() -> None:
    assert SHADOW_VERSION_SPECS["hm_inside_15m_v2"].market_period == "15m"
    assert SHADOW_VERSION_SPECS["ih_inside_15m_v2"].market_period == "15m"
    assert SHADOW_VERSION_SPECS["hm_inside_5m_v2"].market_period == "5m"


def test_adapter_rejects_unregistered_version() -> None:
    row = SimpleNamespace(version="not_registered")
    with pytest.raises(ValueError, match="unregistered"):
        SHADOW_SOURCE_ADAPTERS[0].normalize(row)


def test_adapter_projection_query_is_database_backed_and_bounded() -> None:
    stmt = SHADOW_SOURCE_ADAPTERS[0].statement(EXECUTION_POLICY_VERSION, 17)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    assert "LEFT OUTER JOIN shadow_execution_assessments" in sql
    assert "shadow_execution_assessments.id IS NULL" in sql
    assert f"shadow_execution_assessments.policy_version = '{EXECUTION_POLICY_VERSION}'" in sql
    assert "misalignment_signals.version IN" in sql
    assert "LIMIT 17" in sql


def test_x4_fallback_uses_fixed_one_unit_cost_formula() -> None:
    row = SimpleNamespace(
        id=1,
        version="x4_v1",
        window_start=1_000_000,
        window_end=1_300_000,
        target_window_start=1_300_000,
        direction="UP",
        entry_up_price=0.49,
        entry_quote_ts=1_450_000,
        ev_at_entry=None,
        win=True,
        status="SETTLED",
        settle_outcome="UP",
        created_at=None,
    )

    event = SHADOW_SOURCE_ADAPTERS[0].normalize(row)

    assert event.signal_version == "x4_v1"
    assert event.target_window_start == 1_300_000
    assert event.trigger_ts == 1_300_000
    assert event.evaluation_ts == 1_450_000
    assert event.theoretical_settled_at is None
    assert event.theoretical_realized_return == pytest.approx(0.98 / (0.49 + 0.01) - 1)


def test_kline_and_pattern_adapters_preserve_explicit_timestamps() -> None:
    settled_at = datetime(2026, 9, 14, tzinfo=timezone.utc)
    kline = SimpleNamespace(
        id=2,
        version="krev_a_v1",
        signal_bar_start=2_000_000,
        signal_bar_end=2_900_000,
        target_bar_start=2_900_000,
        direction="UP",
        entry_up_price=0.42,
        entry_quote_ts=None,
        win=None,
        status="PENDING",
        settle_outcome=None,
        created_at=None,
        settled_at=settled_at,
    )
    pattern = SimpleNamespace(
        id=3,
        version="hm_touch_down_v1",
        signal_bar_start=3_000_000,
        signal_bar_end=3_900_000,
        target_bar_start=3_900_000,
        entry_down_quote=0.31,
        touch_ts=4_150_000,
        win=True,
        status="SETTLED",
        settle_outcome="DOWN",
        created_at=None,
        settled_at=settled_at,
    )

    kline_event = SHADOW_SOURCE_ADAPTERS[1].normalize(kline)
    pattern_event = SHADOW_SOURCE_ADAPTERS[2].normalize(pattern)

    assert kline_event.trigger_ts == 2_900_000
    assert kline_event.evaluation_ts is None
    assert kline_event.theoretical_settled_at == settled_at
    assert pattern_event.trigger_ts == 3_900_000
    assert pattern_event.evaluation_ts == 4_150_000
    assert pattern_event.theoretical_settled_at == settled_at


def test_absorption_and_firsthit_adapters_use_recorded_event_times_only() -> None:
    absorption = SimpleNamespace(
        id=4,
        version="absorption_follow_td150_v1",
        window_start=5_000_000,
        td=150,
        direction="DOWN",
        entry_down_price=0.27,
        entry_quote_ts=5_149_000,
        ev_at_entry=-1.0,
        win=False,
        status="SETTLED",
        settle_outcome="UP",
        created_at=None,
    )
    firsthit = SimpleNamespace(
        id=5,
        version="firsthit_down_v1",
        window_start=6_000_000,
        trigger_ts=6_087_000,
        entry_down_price=0.08,
        ev_at_entry=11.25,
        win=True,
        status="SETTLED",
        settle_outcome="DOWN",
        created_at=None,
    )

    absorption_event = SHADOW_SOURCE_ADAPTERS[3].normalize(absorption)
    firsthit_event = SHADOW_SOURCE_ADAPTERS[4].normalize(firsthit)

    assert absorption_event.trigger_ts == 5_150_000
    assert absorption_event.evaluation_ts == 5_149_000
    assert absorption_event.theoretical_settled_at is None
    assert firsthit_event.trigger_ts == 6_087_000
    assert firsthit_event.evaluation_ts == 6_087_000
    assert firsthit_event.theoretical_settled_at is None
