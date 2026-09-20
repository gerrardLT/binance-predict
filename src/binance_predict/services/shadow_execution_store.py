"""Persistence helpers for the unified shadow execution assessment ledger."""
from __future__ import annotations

from typing import Any

from sqlalchemy import case, func, literal, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from binance_predict.db.models import ShadowExecutionAssessment

from .shadow_execution_registry import SHADOW_VERSION_SPECS
from .shadow_execution_types import (
    EXECUTION_POLICY_VERSION,
    AssessmentPatch,
    NormalizedShadowEvent,
    ReasonCode,
    TerminalStage,
    sanitize_json,
    stage_rank,
)


async def build_assessment_values(
    session: AsyncSession,
    event: NormalizedShadowEvent,
    policy_version: str = EXECUTION_POLICY_VERSION,
) -> dict[str, Any]:
    """Build source-owned values without reconstructing runtime execution facts."""
    del session
    spec = SHADOW_VERSION_SPECS[event.signal_version]
    if spec.live_channel is None:
        stage = TerminalStage.SOURCE
        reason = ReasonCode.NO_LIVE_MAPPING
    elif spec.live_retired:
        stage = TerminalStage.UNKNOWN
        reason = ReasonCode.RETIRED_CHANNEL
    else:
        stage = TerminalStage.UNKNOWN
        reason = ReasonCode.LEGACY_UNKNOWN

    return {
        "source_type": event.source_type.value,
        "source_id": event.source_id,
        "signal_version": event.signal_version,
        "policy_version": policy_version,
        "source_window_start": event.source_window_start,
        "target_window_start": event.target_window_start,
        "market_period": event.market_period,
        "direction": event.direction,
        "trigger_ts": event.trigger_ts,
        "evaluation_ts": event.evaluation_ts,
        "terminal_stage": stage.value,
        "reason_code": reason.value,
        "legacy_status": event.source_status,
        "live_channel": spec.live_channel,
        "channel_mapped": spec.live_channel is not None,
        "retired": spec.live_retired,
        "strategy_eligible": None,
        "operational_eligible": None,
        "execution_eligible": None,
        "source_status": event.source_status,
        "theoretical_win": event.theoretical_win,
        "theoretical_outcome": event.theoretical_outcome,
        "theoretical_entry_price": event.theoretical_entry_price,
        "theoretical_realized_return": event.theoretical_realized_return,
        "theoretical_settled_at": event.theoretical_settled_at,
        "source_created_at": event.source_created_at,
        "input_snapshot": sanitize_json({
            "source": event.source_payload,
            "theoretical": event.theoretical_payload,
        }),
    }


def _rank_case(column):
    return case(
        *[(column == stage.value, rank) for stage, rank in {
            TerminalStage.UNKNOWN: 0,
            TerminalStage.SOURCE: 10,
            TerminalStage.STRATEGY: 20,
            TerminalStage.OPERATIONAL: 30,
            TerminalStage.EXECUTION: 40,
            TerminalStage.ORDER: 50,
            TerminalStage.FILL: 60,
            TerminalStage.SETTLEMENT: 70,
        }.items()],
        else_=0,
    )


async def upsert_assessment(session: AsyncSession, values: dict[str, Any]) -> None:
    """Merge a source event into its event-policy row without regressing runtime facts."""
    table = ShadowExecutionAssessment.__table__
    clean_values = {key: value for key, value in values.items() if key in table.c}
    for key in ("config_snapshot", "input_snapshot", "decision_snapshot", "quote_snapshot"):
        if key in clean_values:
            clean_values[key] = sanitize_json(clean_values[key])
    stmt = insert(table).values(**clean_values)
    excluded = stmt.excluded
    advances = _rank_case(excluded.terminal_stage) >= _rank_case(table.c.terminal_stage)
    source_owned = {
        "source_type", "source_id", "source_window_start", "direction", "legacy_status",
        "live_channel", "channel_mapped", "retired", "source_status", "theoretical_win",
        "theoretical_outcome", "theoretical_entry_price", "theoretical_realized_return",
        "theoretical_settled_at", "source_created_at",
    }
    update_values = {key: excluded[key] for key in source_owned if key in clean_values}
    for timestamp_field in ("trigger_ts", "evaluation_ts"):
        if timestamp_field in clean_values:
            update_values[timestamp_field] = func.coalesce(
                table.c[timestamp_field], excluded[timestamp_field]
            )
    if "input_snapshot" in clean_values:
        update_values["input_snapshot"] = table.c.input_snapshot.op("||")(
            excluded.input_snapshot
        )
    update_values.update({
        "terminal_stage": case((advances, excluded.terminal_stage), else_=table.c.terminal_stage),
        "reason_code": case((advances, excluded.reason_code), else_=table.c.reason_code),
        "updated_at": func.now(),
    })
    stmt = stmt.on_conflict_do_update(
        constraint="uq_shadow_exec_event_policy",
        set_=update_values,
    )
    await session.execute(stmt)


async def get_or_create_runtime_assessment(
    session: AsyncSession,
    *,
    source_type: str,
    signal_version: str,
    target_window_start: int,
    market_period: str,
    direction: str,
    policy_version: str = EXECUTION_POLICY_VERSION,
) -> ShadowExecutionAssessment:
    """Get or create the runtime-first row for one event-policy key without committing."""
    table = ShadowExecutionAssessment.__table__
    key = {
        "signal_version": signal_version,
        "target_window_start": target_window_start,
        "market_period": market_period,
        "policy_version": policy_version,
    }
    stmt = insert(table).values(
        source_type=source_type,
        source_id=None,
        direction=direction,
        terminal_stage=TerminalStage.UNKNOWN.value,
        reason_code=ReasonCode.SOURCE_MISSING.value,
        **key,
    ).on_conflict_do_nothing(constraint="uq_shadow_exec_event_policy")
    await session.execute(stmt)
    result = await session.execute(select(ShadowExecutionAssessment).filter_by(**key))
    return result.scalar_one()


async def patch_assessment(session: AsyncSession, patch: AssessmentPatch) -> None:
    """Best-effort monotonic runtime patch; the caller owns transaction boundaries."""
    table = ShadowExecutionAssessment.__table__
    values = patch.values()
    candidate_rank = stage_rank(values["terminal_stage"])
    advances = literal(candidate_rank) >= _rank_case(table.c.terminal_stage)
    # Stage/reason advance monotonically. A lower-stage late patch may fill facts
    # that are still NULL, but must not overwrite facts captured at a later stage.
    guarded = {
        key: (
            case((advances, value), else_=table.c[key])
            if key in {"terminal_stage", "reason_code"}
            else case((advances, value), else_=func.coalesce(table.c[key], value))
        )
        for key, value in values.items()
    }
    guarded["updated_at"] = func.now()
    await session.execute(update(table).where(table.c.id == patch.assessment_id).values(**guarded))
