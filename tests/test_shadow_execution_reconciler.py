from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import binance_predict.services.shadow_execution_reconciler as reconciler_mod
from binance_predict.services.shadow_execution_reconciler import (
    _legacy_order_statement,
    _order_terminal,
    reconcile_shadow_execution_orders,
)
from binance_predict.services.shadow_execution_types import ReasonCode, TerminalStage


def test_legacy_order_query_filters_registry_versions_before_bounded_batch() -> None:
    sql = str(_legacy_order_statement(17).compile(compile_kwargs={"literal_binds": True}))

    assert "trade_orders.signal_version IN" in sql
    assert "quote_contrarian_v2" in sql
    assert "LEFT OUTER JOIN shadow_execution_assessments" in sql
    assert "strategy_eligible IS NULL" in sql
    assert "LIMIT 17" in sql


def _order(**overrides):
    values = {
        "id": 1,
        "assessment_id": None,
        "signal_version": "quote_contrarian_v2",
        "window_start": 1_000,
        "market_period": "5m",
        "direction": "DOWN",
        "status": "PENDING",
        "settled_at": None,
        "settle_outcome": None,
        "order_id": "ORD-1",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (_order(status="PENDING"), (TerminalStage.ORDER, ReasonCode.ORDER_PENDING)),
        (
            _order(status="PENDING", order_id=None),
            (TerminalStage.ORDER, ReasonCode.ORDER_PENDING),
        ),
        (_order(status="FAILED"), (TerminalStage.FILL, ReasonCode.ORDER_FAILED)),
        # 无交易所单号的 FAILED=执行前弃单（护栏/市场/报价）：必须是执行阶段事实，
        # 否则资格列永远补不上、对账游标卡死（2026-10-14 修复）。
        (
            _order(status="FAILED", order_id=None),
            (TerminalStage.EXECUTION, ReasonCode.ORDER_FAILED),
        ),
        (_order(status="FILLED"), (TerminalStage.FILL, ReasonCode.ORDER_FILLED)),
        (
            _order(
                status="FILLED",
                settled_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
                settle_outcome="DOWN",
            ),
            (TerminalStage.SETTLEMENT, ReasonCode.SETTLED),
        ),
        (
            _order(
                status="FILLED",
                settled_at=datetime(2026, 9, 14, tzinfo=timezone.utc),
                settle_outcome="EXPIRED",
            ),
            (TerminalStage.SETTLEMENT, ReasonCode.SETTLEMENT_EXPIRED),
        ),
    ],
)
def test_order_terminal_uses_only_proven_order_facts(order, expected) -> None:
    assert _order_terminal(order) == expected


class _Result:
    def __init__(self, *, rows=None, rowcount=None, scalar=None):
        self._rows = rows
        self.rowcount = rowcount
        self._scalar = scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one(self):
        return self._scalar


class _Session:
    def __init__(self, rows, *, update_rowcounts=(), duplicates=0):
        self._results = [_Result(rows=rows)]
        self._results.extend(_Result(rowcount=value) for value in update_rowcounts)
        self._results.append(_Result(scalar=duplicates))
        self.nested_entries = 0

    async def execute(self, _statement):
        return self._results.pop(0)

    @asynccontextmanager
    async def begin_nested(self):
        self.nested_entries += 1
        yield


@pytest.mark.asyncio
async def test_reconcile_links_exact_key_and_advances_proven_terminal(monkeypatch) -> None:
    order = _order(status="FILLED")
    session = _Session([order], update_rowcounts=[1], duplicates=2)
    assessment = SimpleNamespace(id=41, terminal_stage="UNKNOWN", reason_code="SOURCE_MISSING")
    get_or_create = AsyncMock(return_value=assessment)
    patch = AsyncMock()
    backfill = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)
    monkeypatch.setattr(reconciler_mod, "patch_assessment", patch)
    monkeypatch.setattr(reconciler_mod, "backfill_eligibility", backfill)

    report = await reconcile_shadow_execution_orders(session)

    assert report.scanned == 1
    assert report.linked == 1
    assert report.backfilled == 0
    assert report.skipped == 0
    assert report.duplicate_exchange_orders == 2
    assert session.nested_entries == 1
    backfill.assert_not_awaited()
    kwargs = get_or_create.await_args.kwargs
    assert kwargs["signal_version"] == "quote_contrarian_v2"
    assert kwargs["target_window_start"] == 1_000
    assert kwargs["market_period"] == "5m"
    assert kwargs["direction"] == "DOWN"
    written = patch.await_args.args[1]
    assert written.assessment_id == 41
    assert written.stage is TerminalStage.FILL
    assert written.reason is ReasonCode.ORDER_FILLED
    assert written.strategy_eligible is True
    assert written.operational_eligible is True
    assert written.execution_eligible is True


@pytest.mark.asyncio
async def test_reconcile_backfills_guard_reject_without_regressing_reason(monkeypatch) -> None:
    """护栏弃单（有评估、无单号、已写具体原因）只补资格列，不改 stage/reason。"""
    order = _order(status="FAILED", order_id=None)
    session = _Session([order], update_rowcounts=[0], duplicates=0)
    assessment = SimpleNamespace(
        id=43, terminal_stage="EXECUTION", reason_code="EXEC_PRICE_GUARD"
    )
    get_or_create = AsyncMock(return_value=assessment)
    patch = AsyncMock()
    backfill = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)
    monkeypatch.setattr(reconciler_mod, "patch_assessment", patch)
    monkeypatch.setattr(reconciler_mod, "backfill_eligibility", backfill)

    report = await reconcile_shadow_execution_orders(session)

    assert (report.scanned, report.linked, report.backfilled, report.skipped) == (1, 0, 1, 0)
    patch.assert_not_awaited()
    kwargs = backfill.await_args.kwargs
    assert kwargs["assessment_id"] == 43
    assert kwargs["strategy_eligible"] is True
    assert kwargs["operational_eligible"] is True
    assert kwargs["execution_eligible"] is False


@pytest.mark.asyncio
async def test_reconcile_advances_legacy_reject_to_execution_stage(monkeypatch) -> None:
    """旧账本（stage=UNKNOWN）里的无单号弃单：推进到 EXECUTION/ORDER_FAILED。"""
    order = _order(status="FAILED", order_id=None)
    session = _Session([order], update_rowcounts=[1], duplicates=0)
    assessment = SimpleNamespace(id=44, terminal_stage="UNKNOWN", reason_code="LEGACY_UNKNOWN")
    get_or_create = AsyncMock(return_value=assessment)
    patch = AsyncMock()
    backfill = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)
    monkeypatch.setattr(reconciler_mod, "patch_assessment", patch)
    monkeypatch.setattr(reconciler_mod, "backfill_eligibility", backfill)

    report = await reconcile_shadow_execution_orders(session)

    assert (report.scanned, report.linked, report.backfilled, report.skipped) == (1, 1, 0, 0)
    backfill.assert_not_awaited()
    written = patch.await_args.args[1]
    assert written.stage is TerminalStage.EXECUTION
    assert written.reason is ReasonCode.ORDER_FAILED
    assert written.execution_eligible is False


@pytest.mark.asyncio
async def test_reconcile_savepoint_failure_does_not_block_later_order(monkeypatch) -> None:
    rows = [_order(id=1, order_id="ORD-1"), _order(id=2, order_id="ORD-2")]
    session = _Session(rows, update_rowcounts=[1], duplicates=0)
    get_or_create = AsyncMock(
        side_effect=[
            RuntimeError("conflict"),
            SimpleNamespace(id=42, terminal_stage="UNKNOWN", reason_code="SOURCE_MISSING"),
        ]
    )
    patch = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)
    monkeypatch.setattr(reconciler_mod, "patch_assessment", patch)
    monkeypatch.setattr(reconciler_mod, "backfill_eligibility", AsyncMock())

    report = await reconcile_shadow_execution_orders(session)

    assert report.scanned == 2
    assert report.linked == 1
    assert report.skipped == 1
    assert session.nested_entries == 2
    assert get_or_create.await_count == 2
    written = patch.await_args.args[1]
    assert written.assessment_id == 42
    assert written.reason is ReasonCode.ORDER_PENDING


@pytest.mark.asyncio
async def test_reconcile_skips_unregistered_version_without_guessing(monkeypatch) -> None:
    session = _Session([_order(signal_version="unknown_v9")], duplicates=0)
    get_or_create = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)

    report = await reconcile_shadow_execution_orders(session)

    assert report == reconciler_mod.ReconcileReport(scanned=1, linked=0, skipped=1)
    assert session.nested_entries == 0
    get_or_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_scans_registered_scene_and_candle_orders(monkeypatch) -> None:
    """2026-10-14 P1：场景通道与蜡烛实盘通道的订单不再被视为 unregistered 跳过。"""
    rows = [
        _order(id=1, signal_version="scene_bull_exhaust", order_id="EX-S1", status="FILLED"),
        _order(id=2, signal_version="candle_hm_bull_5m_consensus2_shadow_v1", order_id=None, status="FAILED"),
    ]
    session = _Session(rows, update_rowcounts=[1, 1], duplicates=0)
    assessment = SimpleNamespace(id=55, terminal_stage="UNKNOWN", reason_code="SOURCE_MISSING")
    get_or_create = AsyncMock(return_value=assessment)
    patch = AsyncMock()
    backfill = AsyncMock()
    monkeypatch.setattr(reconciler_mod, "get_or_create_runtime_assessment", get_or_create)
    monkeypatch.setattr(reconciler_mod, "patch_assessment", patch)
    monkeypatch.setattr(reconciler_mod, "backfill_eligibility", backfill)

    report = await reconcile_shadow_execution_orders(session)

    assert report.scanned == 2
    assert report.skipped == 0
    assert report.linked == 2
    assert get_or_create.await_count == 2
