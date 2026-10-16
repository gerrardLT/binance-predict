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
        result = MagicMock()
        if not isinstance(stmt, tuple):
            # 真实 SQLAlchemy 语句（标签行回填查询）：按「无待回填行」返回
            result.scalars.return_value.all.return_value = []
            return result
        query_kind, source_type = stmt[:2]
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
        scanned=0, linked=0, backfilled=0, skipped=0, duplicate_exchange_orders=0))
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
    # 提交次数 = 2 adapter × (new + refresh) 4 次 + 标签行回填 1 次 + 旧订单对账 1 次
    assert db.commit.await_count == 6
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
            scanned=0, linked=0, backfilled=0, skipped=0, duplicate_exchange_orders=0)),
    )

    assert await projector.project_once() == 1
    assert adapter.statement_calls == [
        (projector_mod.EXECUTION_POLICY_VERSION, 4)]
    assert adapter.refresh_calls == [
        (projector_mod.EXECUTION_POLICY_VERSION, 4)]
    build.assert_awaited_once()
    upsert.assert_awaited_once_with(db, {"source_status": "SETTLED"})


@pytest.mark.asyncio
async def test_project_once_settles_label_forward_rows(monkeypatch):
    """物理事件带标签时逐标签补结算：否则标签行的 1/5/10U 报价阶梯永远停在 unsettled。"""
    event_version = "candle_hm_bull_15m_event_v1"
    labels = ["candle_hm_bull_15m_ma10_component_shadow_v1",
              "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1"]

    class _LabelAdapter:
        source_type = SourceType.KLINE

        def __init__(self):
            self.rows = [SimpleNamespace(id=1, version=event_version)]
            self.refresh_rows = []

        def statement(self, policy_version, batch_size):
            return ("new", self.source_type)

        def refresh_statement(self, policy_version, batch_size):
            return ("refresh", self.source_type)

        def normalize(self, row):
            return SimpleNamespace(
                signal_version=event_version, source_id=1, target_window_start=900_000,
                market_period="15m", theoretical_outcome="DOWN",
                source_payload={"matched_signal_ids": [event_version, *labels],
                                "all_matched_signal_ids": [event_version, *labels]})

    adapter = _LabelAdapter()
    db = _Db([adapter])
    projector = ShadowExecutionProjector(batch_size=5)
    monkeypatch.setattr(projector_mod, "SHADOW_SOURCE_ADAPTERS", [adapter])
    monkeypatch.setattr(projector_mod, "async_session_factory", lambda: _factory(db))
    monkeypatch.setattr(projector_mod, "build_assessment_values", AsyncMock(return_value={}))
    monkeypatch.setattr(projector_mod, "upsert_assessment", AsyncMock())
    monkeypatch.setattr(
        projector_mod, "reconcile_shadow_execution_orders",
        AsyncMock(return_value=SimpleNamespace(
            scanned=0, linked=0, backfilled=0, skipped=0, duplicate_orders=0,
            duplicate_exchange_orders=0)))
    from binance_predict.services import shadow_forward_evidence as fe
    settle = AsyncMock()
    monkeypatch.setattr(fe, "settle_firsthit", settle)

    assert await projector.project_once() == 1
    assert [call.args[1:] for call in settle.await_args_list] == [
        (900_000, "DOWN", "15m", event_version),
        (900_000, "DOWN", "15m", labels[0]),
        (900_000, "DOWN", "15m", labels[1]),
    ]


@pytest.mark.asyncio
async def test_backfill_settles_legacy_label_rows_from_source_event(monkeypatch):
    """存量回填：源事件已结算但标签行仍 unsettled → 用源事件 outcome 补写（幂等、只认命中标签）。"""
    from binance_predict.services.shadow_execution_projector import (
        _backfill_candle_label_forward_rows,
    )

    label = "candle_hm_bull_15m_ma10_component_shadow_v1"
    other = "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1"
    pending = [
        SimpleNamespace(signal_version=label, target_window_start=900_000, market_period="15m",
                        direction="DOWN", theoretical_outcome=None, theoretical_win=None,
                        theoretical_settled_at=None, decision_snapshot={}),
        # 源事件未命中该标签（不在 all_matched_signal_ids）→ 必须跳过，不得猜结果
        SimpleNamespace(signal_version="candle_hm_bull_5m_asia_shadow_hyp_v1",
                        target_window_start=900_000, market_period="15m",
                        direction="DOWN", theoretical_outcome=None, theoretical_win=None,
                        theoretical_settled_at=None, decision_snapshot={}),
    ]
    source = SimpleNamespace(
        version="candle_hm_bull_15m_event_v1", target_bar_start=900_000, status="SETTLED",
        settle_outcome="DOWN",
        feature_snapshot={"all_matched_signal_ids": [label, other]})
    results = [pending, [source]]

    class _Session:
        def __init__(self, queue):
            self.queue = queue

        async def execute(self, _stmt):
            result = MagicMock()
            result.scalars.return_value.all.return_value = self.queue.pop(0) if self.queue else []
            return result

    filled = await _backfill_candle_label_forward_rows(_Session(results))
    assert filled == 1
    assert pending[0].theoretical_outcome == "DOWN" and pending[0].theoretical_win is True
    assert pending[1].theoretical_outcome is None
    # 幂等：重复运行不再有可回填行（结果集为空）
    assert await _backfill_candle_label_forward_rows(_Session([])) == 0


@pytest.mark.asyncio
async def test_commit_failure_keeps_cursor_and_continues_next_adapter(monkeypatch):
    adapters = [
        _Adapter(SourceType.MISALIGNMENT, [SimpleNamespace(id=11, version="x4_v1")]),
        _Adapter(SourceType.KLINE, [SimpleNamespace(id=22, version="krev_a_v1")]),
    ]
    db = _Db(adapters)
    failed = {"done": False}

    async def _commit():
        # 只让「第一个 adapter 的首次提交」失败，其余提交照常成功
        # （不写死提交次数，避免后续新增投影阶段时脆断）
        if not failed["done"]:
            failed["done"] = True
            raise RuntimeError("commit failed")

    db.commit = AsyncMock(side_effect=_commit)
    projector = ShadowExecutionProjector(batch_size=5)
    monkeypatch.setattr(projector_mod, "SHADOW_SOURCE_ADAPTERS", adapters)
    monkeypatch.setattr(projector_mod, "async_session_factory", lambda: _factory(db))
    monkeypatch.setattr(projector_mod, "build_assessment_values", AsyncMock(return_value={}))
    monkeypatch.setattr(projector_mod, "upsert_assessment", AsyncMock())
    monkeypatch.setattr(
        projector_mod,
        "reconcile_shadow_execution_orders",
        AsyncMock(return_value=SimpleNamespace(
            scanned=0, linked=0, backfilled=0, skipped=0, duplicate_exchange_orders=0)),
    )

    assert await projector.project_once() == 1
    assert [a.statement_calls for a in adapters] == [
        [(projector_mod.EXECUTION_POLICY_VERSION, 5)],
        [(projector_mod.EXECUTION_POLICY_VERSION, 5)],
    ]
    assert adapters[0].refresh_calls == []
    assert adapters[1].refresh_calls == [(projector_mod.EXECUTION_POLICY_VERSION, 5)]
    db.rollback.assert_awaited_once()
