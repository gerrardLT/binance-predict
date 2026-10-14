from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import NullType

from binance_predict.services.shadow_execution_registry import SHADOW_VERSION_SPECS
from binance_predict.services.shadow_execution_store import (
    backfill_eligibility,
    build_assessment_values,
    patch_assessment,
    upsert_assessment,
)
from binance_predict.services.shadow_execution_types import (
    EXECUTION_POLICY_VERSION,
    AssessmentPatch,
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
    no_mapping = await build_assessment_values(session, _event("combo_p1_v1"))
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

@pytest.mark.asyncio
async def test_late_fact_patch_fills_eligibility_without_regressing_terminal_stage() -> None:
    session = SimpleNamespace(execute=AsyncMock())

    await patch_assessment(session, AssessmentPatch(
        assessment_id=9,
        stage=TerminalStage.FILL,
        reason=ReasonCode.ORDER_FILLED,
        strategy_eligible=True,
        operational_eligible=True,
        execution_eligible=True,
    ))

    stmt = session.execute.await_args.args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    compact = sql.replace(" ", "").lower()
    assert "strategy_eligible=casewhen" in compact
    assert "operational_eligible=casewhen" in compact
    assert "execution_eligible=casewhen" in compact
    assert "coalesce(shadow_execution_assessments.strategy_eligible" in compact
    assert "terminal_stage=casewhen" in compact


@pytest.mark.asyncio
async def test_patch_assessment_binds_jsonb_with_column_type() -> None:
    """回归（2026-09-20 71a57fc 引入）：CASE/COALESCE 内的绑定值必须带列类型。

    裸字面量在 CASE 里推断成 NullType → asyncpg 收到裸 dict 报
    `DataError: 'dict' object has no attribute 'encode'`，整条评估写入随事务回滚
    （订单 assessment_id 恒 NULL、弃单审计记录全丢、对账器无行可绑）。
    """
    session = SimpleNamespace(execute=AsyncMock())

    await patch_assessment(session, AssessmentPatch(
        assessment_id=11,
        stage=TerminalStage.OPERATIONAL,
        reason=ReasonCode.PASSED,
        config_snapshot={"entry_band_whitelist": [[0.5, 1.01]]},
        input_snapshot={"side": "low"},
        amount_usdt=2.0,
    ))

    stmt = session.execute.await_args.args[0]
    compiled = stmt.compile(dialect=postgresql.dialect())
    binds = list(compiled.binds.values())

    dict_binds = [bp for bp in binds if isinstance(bp.value, dict)]
    assert len(dict_binds) >= 2  # config_snapshot / input_snapshot
    assert all(isinstance(bp.type, JSONB) for bp in dict_binds), (
        "JSONB 绑定丢了类型: " + str([type(bp.type).__name__ for bp in dict_binds]))
    # 标量字段同样不得退化为 NullType（否则 asyncpg 也收不到可编码的值）
    scalar_binds = [bp for bp in binds if isinstance(bp.value, float)]
    assert scalar_binds and all(not isinstance(bp.type, NullType) for bp in scalar_binds)


@dataclass
class _Snapshot:
    status: ReasonCode
    created_at: datetime
    opaque: object


@pytest.mark.asyncio
async def test_backfill_eligibility_only_fills_nulls_and_never_touches_stage() -> None:
    """对账回填（2026-10-14）：只在列为 NULL 时写资格，SET 子句不碰 stage/reason。"""
    session = SimpleNamespace(execute=AsyncMock())

    await backfill_eligibility(
        session, assessment_id=12, strategy_eligible=True,
        operational_eligible=True, execution_eligible=False,
    )

    stmt = session.execute.await_args.args[0]
    compiled = stmt.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    compact = sql.replace(" ", "").lower()
    assert "strategy_eligible=casewhen" in compact
    assert "operational_eligible=casewhen" in compact
    assert "execution_eligible=casewhen" in compact
    assert compact.count("isnull") == 3  # 每列都带 IS NULL 守卫，不覆盖已有值
    assert "terminal_stage" not in sql and "reason_code" not in sql
    boolean_binds = [
        bp for bp in compiled.binds.values()
        if isinstance(bp.value, bool)
    ]
    # 每个资格列一个带类型的布尔绑定（编译产物会同时保留匿名与具名两种参数键）。
    assert len({bp.value for bp in boolean_binds}) == 2
    assert len(boolean_binds) >= 3
    assert all(not isinstance(bp.type, NullType) for bp in boolean_binds)

    idle = SimpleNamespace(execute=AsyncMock())
    await backfill_eligibility(idle, assessment_id=12)
    idle.execute.assert_not_awaited()


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
