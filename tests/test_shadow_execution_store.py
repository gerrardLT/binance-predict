from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from binance_predict.services.shadow_execution_registry import SHADOW_VERSION_SPECS
from binance_predict.services.shadow_execution_store import (
    build_assessment_values,
    upsert_assessment,
)
from binance_predict.services.shadow_execution_types import (
    EXECUTION_POLICY_VERSION,
    NormalizedShadowEvent,
    ReasonCode,
    SourceType,
    TerminalStage,
    sanitize_json,
    stage_advances,
)


def _event(version: str = "quote_contrarian_v2") -> NormalizedShadowEvent:
    settled_at = datetime(2026, 9, 14, tzinfo=timezone.utc)
    return NormalizedShadowEvent(
        source_type=SHADOW_VERSION_SPECS[version].source_type,
        source_id=7,
        signal_version=version,
        source_window_start=1_000,
        target_window_start=2_000,
        market_period=SHADOW_VERSION_SPECS[version].market_period,
        direction="DOWN",
        trigger_ts=1_100,
        evaluation_ts=1_150,
        source_status="SETTLED",
        theoretical_win=True,
        theoretical_outcome="DOWN",
        theoretical_entry_price=0.25,
        theoretical_realized_return=2.92,
        theoretical_settled_at=settled_at,
    )


@pytest.mark.asyncio
async def test_source_projector_does_not_query_orders_and_uses_explicit_policy() -> None:
    session = SimpleNamespace(execute=AsyncMock())

    values = await build_assessment_values(session, _event())

    session.execute.assert_not_awaited()
    assert values["signal_version"] == "quote_contrarian_v2"
    assert values["policy_version"] == EXECUTION_POLICY_VERSION
    assert values["channel_mapped"] is True
    assert values["retired"] is False
    assert values["terminal_stage"] == TerminalStage.UNKNOWN.value
    assert values["reason_code"] == ReasonCode.LEGACY_UNKNOWN.value
    assert values["strategy_eligible"] is None
    assert values["operational_eligible"] is None
    assert values["execution_eligible"] is None
    assert values["trigger_ts"] == 1_100
    assert values["evaluation_ts"] == 1_150
    assert values["theoretical_settled_at"] == datetime(
        2026, 9, 14, tzinfo=timezone.utc
    )


@pytest.mark.asyncio
async def test_source_projector_classifies_no_mapping_and_retired() -> None:
    session = SimpleNamespace(execute=AsyncMock())
    no_mapping = await build_assessment_values(session, _event("krev_a_v1"))
    retired_version = next(
        version for version, spec in SHADOW_VERSION_SPECS.items() if spec.live_retired
    )
    retired = await build_assessment_values(session, _event(retired_version))

    assert no_mapping["channel_mapped"] is False
    assert no_mapping["retired"] is False
    assert no_mapping["terminal_stage"] == TerminalStage.SOURCE.value
    assert no_mapping["reason_code"] == ReasonCode.NO_LIVE_MAPPING.value
    assert retired["channel_mapped"] is True
    assert retired["retired"] is True
    assert retired["terminal_stage"] == TerminalStage.UNKNOWN.value
    assert retired["reason_code"] == ReasonCode.RETIRED_CHANNEL.value
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_uses_event_policy_and_monotonic_stage_case() -> None:
    session = SimpleNamespace(execute=AsyncMock())
    values = await build_assessment_values(session, _event())

    await upsert_assessment(session, values)

    stmt = session.execute.await_args.args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT ON CONSTRAINT uq_shadow_exec_event_policy DO UPDATE" in sql
    assert "CASE WHEN" in sql
    assert "excluded.terminal_stage" in sql
    assert "shadow_execution_assessments.terminal_stage" in sql


@pytest.mark.asyncio
async def test_source_upsert_binds_runtime_first_row_without_overwriting_runtime_fields() -> None:
    session = SimpleNamespace(execute=AsyncMock())
    values = await build_assessment_values(session, _event())

    await upsert_assessment(session, values)

    stmt = session.execute.await_args.args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "source_id = excluded.source_id" in sql
    assert "source_type = excluded.source_type" in sql
    assert "theoretical_win = excluded.theoretical_win" in sql
    assert "theoretical_settled_at = excluded.theoretical_settled_at" in sql
    assert "trigger_ts = coalesce(shadow_execution_assessments.trigger_ts, excluded.trigger_ts)" in sql
    assert "evaluation_ts = coalesce(shadow_execution_assessments.evaluation_ts, excluded.evaluation_ts)" in sql
    assert "strategy_eligible = excluded.strategy_eligible" not in sql
    assert "config_fingerprint = excluded.config_fingerprint" not in sql
    assert "enabled = excluded.enabled" not in sql
    assert "quote_snapshot = excluded.quote_snapshot" not in sql
    assert "decision_snapshot = excluded.decision_snapshot" not in sql
    assert "input_snapshot = (shadow_execution_assessments.input_snapshot || excluded.input_snapshot)" in sql


def test_unknown_can_advance_and_equal_rank_can_refresh_reason() -> None:
    assert stage_advances(TerminalStage.UNKNOWN, TerminalStage.STRATEGY)
    assert stage_advances(TerminalStage.ORDER, TerminalStage.ORDER)
    assert not stage_advances(TerminalStage.ORDER, TerminalStage.EXECUTION)


@dataclass
class _Snapshot:
    status: ReasonCode
    created_at: datetime
    opaque: object


def test_sanitizer_redacts_nested_secrets_and_normalizes_values() -> None:
    clean = sanitize_json({
        "API_KEY": "raw-key",
        "nested": {
            "authorization_header": "Bearer raw-token",
            "items": [{"passwordHash": "raw-password"}, _Snapshot(
                ReasonCode.PASSED,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                object(),
            )],
        },
    })

    assert clean["API_KEY"] == "[REDACTED]"
    assert clean["nested"]["authorization_header"] == "[REDACTED]"
    assert clean["nested"]["items"][0]["passwordHash"] == "[REDACTED]"
    assert clean["nested"]["items"][1]["status"] == ReasonCode.PASSED.value
    assert clean["nested"]["items"][1]["created_at"] == "2026-01-02T00:00:00+00:00"
    assert isinstance(clean["nested"]["items"][1]["opaque"], str)
