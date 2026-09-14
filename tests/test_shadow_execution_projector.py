from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import binance_predict.services.shadow_execution_projector as projector_mod
from binance_predict.services.shadow_execution_projector import ShadowExecutionProjector
from binance_predict.services.shadow_execution_types import SourceType


class _Adapter:
    def __init__(self, source_type, rows, refresh_rows=None):
        self.source_type = source_type
        self.rows = rows
        self.refresh_rows = refresh_rows or []
        self.statement_calls = []
        self.refresh_calls = []

    def statement(self, policy_version, batch_size):
        self.statement_calls.append((policy_version, batch_size))
        return ("new", self.source_type, policy_version, batch_size)

    def refresh_statement(self, policy_version, batch_size):
        self.refresh_calls.append((policy_version, batch_size))
        return ("refresh", self.source_type, policy_version, batch_size)

    def normalize(self, row):
        return SimpleNamespace(signal_version=row.version, source_id=row.id)


class _Db:
    def __init__(self, adapters):
        self.adapters = {a.source_type: a for a in adapters}
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    async def execute(self, stmt):
        query_kind, source_type = stmt[:2]
        result = MagicMock()
        rows = (self.adapters[source_type].rows if query_kind == "new"
                else self.adapters[source_type].refresh_rows)
        result.scalars.return_value.all.return_value = rows
        return result


@asynccontextmanager
async def _factory(db):
    yield db


@pytest.mark.asyncio
async def test_start_is_non_blocking(monkeypatch):
    projector = ShadowExecutionProjector(interval_seconds=60)
    entered = False

    async def _loop():
        nonlocal entered
        entered = True

    monkeypatch.setattr(projector, "_loop", _loop)
    await projector.start()
    await projector._task

    assert projector._running is True
    assert entered is True
    await projector.stop()


@pytest.mark.asyncio
async def test_project_once_is_one_batch_per_adapter(monkeypatch):
    adapters = [
        _Adapter(SourceType.MISALIGNMENT, [SimpleNamespace(id=1, version="x4_v1")]),
        _Adapter(SourceType.KLINE, [SimpleNamespace(id=2, version="krev_a_v1")]),
    ]
    db = _Db(adapters)
    projector = ShadowExecutionProjector(batch_size=7)
    monkeypatch.setattr(projector_mod, "SHADOW_SOURCE_ADAPTERS", adapters)
    monkeypatch.setattr(projector_mod, "async_session_factory", lambda: _factory(db))
    monkeypatch.setattr(projector_mod, "build_assessment_values", AsyncMock(return_value={}))
    monkeypatch.setattr(projector_mod, "upsert_assessment", AsyncMock())
    reconcile = AsyncMock(return_value=SimpleNamespace(
        scanned=0, linked=0, skipped=0, duplicate_exchange_orders=0))
    monkeypatch.setattr(projector_mod, "reconcile_shadow_execution_orders", reconcile)

    assert await projector.project_once() == 2
    assert [a.statement_calls for a in adapters] == [
        [(projector_mod.EXECUTION_POLICY_VERSION, 7)],
        [(projector_mod.EXECUTION_POLICY_VERSION, 7)],
    ]
    assert [a.refresh_calls for a in adapters] == [
        [(projector_mod.EXECUTION_POLICY_VERSION, 7)],
        [(projector_mod.EXECUTION_POLICY_VERSION, 7)],
    ]
    assert db.commit.await_count == 5
    reconcile.assert_awaited_once_with(db, batch_size=7)


@pytest.mark.asyncio
async def test_project_once_refreshes_changed_source_status(monkeypatch):
    settled = SimpleNamespace(id=3, version="x4_v1")
    adapter = _Adapter(SourceType.MISALIGNMENT, [], refresh_rows=[settled])
    db = _Db([adapter])
    projector = ShadowExecutionProjector(batch_size=4)
    build = AsyncMock(return_value={"source_status": "SETTLED"})
    upsert = AsyncMock()
    monkeypatch.setattr(projector_mod, "SHADOW_SOURCE_ADAPTERS", [adapter])
    monkeypatch.setattr(projector_mod, "async_session_factory", lambda: _factory(db))
    monkeypatch.setattr(projector_mod, "build_assessment_values", build)
    monkeypatch.setattr(projector_mod, "upsert_assessment", upsert)
    monkeypatch.setattr(
        projector_mod,
        "reconcile_shadow_execution_orders",
        AsyncMock(return_value=SimpleNamespace(
            scanned=0, linked=0, skipped=0, duplicate_exchange_orders=0)),
    )

    assert await projector.project_once() == 1
    assert adapter.statement_calls == [
        (projector_mod.EXECUTION_POLICY_VERSION, 4)]
    assert adapter.refresh_calls == [
        (projector_mod.EXECUTION_POLICY_VERSION, 4)]
    build.assert_awaited_once()
    upsert.assert_awaited_once_with(db, {"source_status": "SETTLED"})


@pytest.mark.asyncio
async def test_commit_failure_keeps_cursor_and_continues_next_adapter(monkeypatch):
    adapters = [
        _Adapter(SourceType.MISALIGNMENT, [SimpleNamespace(id=11, version="x4_v1")]),
        _Adapter(SourceType.KLINE, [SimpleNamespace(id=22, version="krev_a_v1")]),
    ]
    db = _Db(adapters)
    db.commit = AsyncMock(side_effect=[RuntimeError("commit failed"), None, None, None])
    projector = ShadowExecutionProjector(batch_size=5)
    monkeypatch.setattr(projector_mod, "SHADOW_SOURCE_ADAPTERS", adapters)
    monkeypatch.setattr(projector_mod, "async_session_factory", lambda: _factory(db))
    monkeypatch.setattr(projector_mod, "build_assessment_values", AsyncMock(return_value={}))
    monkeypatch.setattr(projector_mod, "upsert_assessment", AsyncMock())
    monkeypatch.setattr(
        projector_mod,
        "reconcile_shadow_execution_orders",
        AsyncMock(return_value=SimpleNamespace(
            scanned=0, linked=0, skipped=0, duplicate_exchange_orders=0)),
    )

    assert await projector.project_once() == 1
    assert [a.statement_calls for a in adapters] == [
        [(projector_mod.EXECUTION_POLICY_VERSION, 5)],
        [(projector_mod.EXECUTION_POLICY_VERSION, 5)],
    ]
    assert adapters[0].refresh_calls == []
    assert adapters[1].refresh_calls == [(projector_mod.EXECUTION_POLICY_VERSION, 5)]
    db.rollback.assert_awaited_once()
