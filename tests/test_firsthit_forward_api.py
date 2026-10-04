"""首触前向事件只读导出端点测试（2026-10-04）：
GET /api/signals/firsthit-forward（main.get_firsthit_forward_events）

不触网络/真实 DB：db 为 AsyncMock 替身，事件行用 SimpleNamespace；
SQL 过滤/分页通过编译产物断言（literal_binds），响应映射逐字段核对。
端点定位：G3 实盘归因后的可执行 EV 校准数据源——LIVE 行的 execution_quotes
（1/5/10 USDT 报价阶梯）必须在响应中原样透传，供离线 spread 折损校准。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


def _event_row(**over) -> SimpleNamespace:
    """前向事件行替身：默认 DOWN 侧 LIVE 触达、已结算、带 1U 报价阶梯。"""
    base = dict(
        id=1, schema_version="firsthit_reversal_forward_v1", capture_mode="LIVE",
        window_start=1_790_000_000_000, window_end=1_790_000_300_000,
        side="DOWN", trigger_ts=1_790_000_045_000, td_sec=45, trigger_q=0.09,
        opposite_q=0.91, quote_sum=1.0,
        btc_open=85_000.0, btc_adverse_extreme=85_020.0, btc_trigger=85_005.0,
        btc_signed_return_bps=0.6, btc_signed_min_bps=-2.5,
        btc_recovery_bps=1.4, btc_recovery_frac=0.3, recovery_pass=False,
        atr_bps=12.0, current_move_atr=0.05,
        labels={"down_q10_control": True},
        trigger_features={"q_min": 0.08},
        future_quotes={"15": {"q": 0.1}},
        execution_quotes={"1": {"available": True, "average_price": 0.093,
                                "slippage_bps": 333.3, "latency_ms": 210}},
        post_features={}, retrospective={}, data_missing=[],
        settle_outcome="DOWN", win=True, ev_at_entry=0.98 / 0.093 - 1,
        status="SETTLED", created_at=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _make_db(rows) -> AsyncMock:
    db = AsyncMock()
    res = MagicMock()
    res.scalars.return_value.all.return_value = rows
    db.execute = AsyncMock(return_value=res)
    return db


def _compiled_sql(db) -> str:
    stmt = db.execute.await_args.args[0]
    return str(stmt.compile(compile_kwargs={"literal_binds": True})).upper()


@pytest.mark.asyncio
async def test_forward_export_maps_fields_and_last_id() -> None:
    """字段全量透传（含 execution_quotes 阶梯原样）+ last_id 取末行 id。"""
    import binance_predict.main as m

    rows = [_event_row(id=3), _event_row(id=7, side="UP", win=False)]
    out = await m.get_firsthit_forward_events(db=_make_db(rows))

    assert out["last_id"] == 7
    assert len(out["events"]) == 2
    first = out["events"][0]
    assert first["id"] == 3 and first["side"] == "DOWN"
    assert first["trigger_q"] == 0.09 and first["win"] is True
    # 可执行报价阶梯原样透传（校准数据的核心字段，不得重整/截断）
    assert first["execution_quotes"]["1"]["average_price"] == 0.093
    assert first["execution_quotes"]["1"]["slippage_bps"] == 333.3
    assert first["labels"] == {"down_q10_control": True}
    assert out["events"][1]["side"] == "UP"


@pytest.mark.asyncio
async def test_forward_export_empty_rows_keeps_after_id() -> None:
    """翻页到尾页（无行）→ last_id 回落为请求的 after_id，便于客户端终止。"""
    import binance_predict.main as m

    out = await m.get_firsthit_forward_events(after_id=99, db=_make_db([]))
    assert out["events"] == [] and out["last_id"] == 99


@pytest.mark.asyncio
async def test_forward_export_limit_clamped_to_200() -> None:
    """limit>200 → 200；limit<1 → 1（影子信号接口 ≤200 条口径）。"""
    import binance_predict.main as m

    db = _make_db([])
    await m.get_firsthit_forward_events(limit=5000, db=db)
    assert "LIMIT 200" in _compiled_sql(db)

    db2 = _make_db([])
    await m.get_firsthit_forward_events(limit=0, db=db2)
    assert "LIMIT 1" in _compiled_sql(db2)


@pytest.mark.asyncio
async def test_forward_export_filters_compiled_into_sql() -> None:
    """after_id/side/status 过滤条件进入 SQL；默认无过滤（全量升序翻页）。"""
    import binance_predict.main as m

    db = _make_db([])
    await m.get_firsthit_forward_events(
        after_id=123, side="DOWN", status="SETTLED", db=db)
    sql = _compiled_sql(db)
    assert "ID > 123" in sql or "ID >123" in sql
    assert "SIDE = 'DOWN'" in sql
    assert "STATUS = 'SETTLED'" in sql

    db2 = _make_db([])
    await m.get_firsthit_forward_events(db=db2)
    sql2 = _compiled_sql(db2)
    assert " WHERE " not in sql2  # 默认无过滤（全量升序翻页）


@pytest.mark.asyncio
async def test_forward_export_rejects_invalid_side_and_status() -> None:
    """非法 side/status → 422，且不触 DB。"""
    import binance_predict.main as m

    for kwargs in ({"side": "up"}, {"side": "x"}, {"status": "settled"}):
        with pytest.raises(HTTPException) as exc:
            await m.get_firsthit_forward_events(db=_make_db([]), **kwargs)
        assert exc.value.status_code == 422
