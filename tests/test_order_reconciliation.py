"""R2 订单状态校验 + 自动对账的单元测试（2026-08-25 风险评审）。

覆盖：
- execute_signal_trade 下单响应状态分流：FILLED 才记成交；已知失败终态落
  FAILED；未知/缺失状态与响应超时保持 PENDING（交对账器回读确认）
- OrderReconciler.poll_once：orderId 精确匹配 / 窗口无歧义匹配 /
  多候选不错配 / 超期出清 / 币安历史不可达全跳过

不触网络/真实 DB：async_session_factory 用 @asynccontextmanager 桩
（仿 test_trade_settler 模式）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Select, Update

import binance_predict.services.order_reconciler as or_mod
from binance_predict.services.order_reconciler import OrderReconciler
from binance_predict.services.prediction_trading import BinancePredictionTrader

WS = 1_787_418_600_000          # 5m 窗口起点 ms（对应 slug btc-updown-5m-1787418600）
SLUG = f"btc-updown-5m-{WS // 1000}"


# ============================================================
# execute_signal_trade：下单响应状态分流（R2 ①）
# ============================================================

def _make_trader(monkeypatch, place_result) -> BinancePredictionTrader:
    """构造可下单的 trader 替身：锁内链路除 place_order 外全桩。"""
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    async def _list():
        trader._down_token_id = "T-DOWN"
        return []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return SimpleNamespace(id=1, status="PENDING")

    async def _quote(*a, **k):
        return {"averagePrice": 0.5, "amountIn": "5", "amountOut": "10",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        return place_result

    monkeypatch.setattr(trader, "list_markets", _list)
    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    return trader


def _capture_updates(trader, monkeypatch) -> list[tuple]:
    updates: list[tuple] = []

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {"status": status, **kwargs}

    monkeypatch.setattr(trader, "_update_signal_order", _update)
    return updates


@pytest.mark.asyncio
async def test_signal_trade_filled_status_records_filled(monkeypatch) -> None:
    """响应 status=FILLED → 记成交（R2 状态校验正向路径）。"""
    trader = _make_trader(monkeypatch, {"orderId": "O1", "status": "FILLED"})
    updates = _capture_updates(trader, monkeypatch)
    out = await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    assert out["status"] == "FILLED"
    assert updates[-1][0] == "FILLED"


@pytest.mark.asyncio
async def test_signal_trade_rejected_status_records_failed(monkeypatch) -> None:
    """响应 status=CANCELED（币安侧未成交终态）→ FAILED，不记 FILLED。"""
    trader = _make_trader(monkeypatch, {"orderId": "O2", "status": "CANCELED"})
    updates = _capture_updates(trader, monkeypatch)
    out = await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    assert out["status"] == "FAILED"
    assert "status=CANCELED" in updates[-1][1]["error_message"]


@pytest.mark.asyncio
async def test_signal_trade_unknown_status_keeps_pending(monkeypatch) -> None:
    """响应缺 status（未知态）→ 保持 PENDING 交对账器，绝不臆断 FILLED。"""
    trader = _make_trader(monkeypatch, {"orderId": "O3"})
    updates = _capture_updates(trader, monkeypatch)
    out = await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    assert out["status"] == "PENDING"
    assert updates[-1][1].get("order_id") == "O3"  # orderId 留痕供回读


@pytest.mark.asyncio
async def test_signal_trade_timeout_keeps_pending(monkeypatch) -> None:
    """place_order 返回 None（超时/异常）→ 保持 PENDING 等回读，不落 FAILED。"""
    trader = _make_trader(monkeypatch, None)
    updates = _capture_updates(trader, monkeypatch)
    out = await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    assert out["status"] == "PENDING"
    assert "对账" in updates[-1][1]["error_message"]


# ============================================================
# OrderReconciler：自动对账循环（R2 ②）
# ============================================================

def _row(**over) -> SimpleNamespace:
    base = dict(
        id=21, status="PENDING", order_id=None, window_start=WS,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        amount_in="0", quote_json=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Db:
    """按语句类型路由：首个 Select → PENDING 行；后续 Select → 同窗计数。"""

    def __init__(self, rows, sibling_count: int = 1, update_rowcount: int = 1):
        self.updates: list[Update] = []
        self._rows = rows
        self._sibling_count = sibling_count
        self._update_rowcount = update_rowcount
        self._select_calls = 0

        async def _execute(stmt):
            if isinstance(stmt, Update):
                self.updates.append(stmt)
                res = MagicMock()
                res.rowcount = self._update_rowcount
                return res
            assert isinstance(stmt, Select)
            self._select_calls += 1
            res = MagicMock()
            if self._select_calls == 1:
                res.scalars.return_value.all.return_value = self._rows
            else:
                res.scalar_one.return_value = self._sibling_count
            return res

        self.execute = AsyncMock(side_effect=_execute)
        self.commit = AsyncMock()


def _stub_db(monkeypatch, db: _Db) -> None:
    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(or_mod, "async_session_factory", _factory)


def _trader_with_history(history) -> SimpleNamespace:
    return SimpleNamespace(
        query_order_history=AsyncMock(return_value=history),
        last_api_error=None,
    )


@pytest.mark.asyncio
async def test_reconcile_by_order_id_exact_match(monkeypatch) -> None:
    """本地行带 orderId 且币安历史同 ID FILLED → 订正 FILLED + 回填金额。"""
    db = _Db([_row(order_id="B-1")])
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([
        {"orderId": "B-1", "slug": SLUG, "status": "FILLED",
         "filledUsdtAmount": "2", "price": "0.55", "filledShareQty": "3.6"},
    ]))

    summary = await rec.poll_once()
    assert summary["matched"] == 1
    p = db.updates[0].compile().params
    assert p["status"] == "FILLED"
    assert p["order_id"] == "B-1"
    assert p["amount_in"] == str(2 * 10 ** 18)


@pytest.mark.asyncio
async def test_reconcile_by_window_unambiguous(monkeypatch) -> None:
    """无 orderId 但同窗本地/币安各唯一 → 无歧义匹配订正 FAILED（币安侧 CANCELED）。"""
    db = _Db([_row()], sibling_count=1)
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([
        {"orderId": "B-2", "slug": SLUG, "status": "CANCELED"},
    ]))

    summary = await rec.poll_once()
    assert summary["matched"] == 1
    p = db.updates[0].compile().params
    assert p["status"] == "FAILED"


@pytest.mark.asyncio
async def test_reconcile_ambiguous_window_not_matched(monkeypatch) -> None:
    """同窗币安侧多笔订单（多通道并行）→ 不错配，保持 PENDING 等人工。"""
    db = _Db([_row()], sibling_count=1)
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([
        {"orderId": "B-3", "slug": SLUG, "status": "FILLED"},
        {"orderId": "B-4", "slug": SLUG, "status": "FILLED"},
    ]))

    summary = await rec.poll_once()
    assert summary["matched"] == 0 and summary["skipped"] == 1
    assert db.updates == []


@pytest.mark.asyncio
async def test_reconcile_ambiguous_local_siblings_not_matched(monkeypatch) -> None:
    """同窗本地多笔 PENDING → 即使币安侧唯一也不错配（R5 错配教训）。"""
    db = _Db([_row()], sibling_count=2)
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([
        {"orderId": "B-5", "slug": SLUG, "status": "FILLED"},
    ]))

    summary = await rec.poll_once()
    assert summary["matched"] == 0
    assert db.updates == []


@pytest.mark.asyncio
async def test_reconcile_abandon_after_timeout(monkeypatch) -> None:
    """超 30 分钟币安历史仍无对应订单 → FAILED 出清（FOK 未成交判定）。"""
    db = _Db([_row(created_at=datetime.now(timezone.utc) - timedelta(minutes=40))])
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([]))

    summary = await rec.poll_once()
    assert summary["abandoned"] == 1
    p = db.updates[0].compile().params
    assert p["status"] == "FAILED"
    assert "对账出清" in p["error_message"]


@pytest.mark.asyncio
async def test_reconcile_history_unavailable_all_skipped(monkeypatch) -> None:
    """币安历史查询失败（限频/网络）→ 全跳过不错判，下轮重试。"""
    db = _Db([_row()])
    _stub_db(monkeypatch, db)
    trader = SimpleNamespace(
        query_order_history=AsyncMock(return_value=None),
        last_api_error="HTTP 429",
    )
    rec = OrderReconciler(trader)

    summary = await rec.poll_once()
    assert summary["skipped"] == summary["pending"] == 1
    assert db.updates == []


@pytest.mark.asyncio
async def test_reconcile_idempotent_guard(monkeypatch) -> None:
    """UPDATE rowcount=0（已被人工订正）→ 匹配计数不重复。"""
    db = _Db([_row(order_id="B-1")], update_rowcount=0)
    _stub_db(monkeypatch, db)
    rec = OrderReconciler(_trader_with_history([
        {"orderId": "B-1", "slug": SLUG, "status": "FILLED"},
    ]))

    summary = await rec.poll_once()
    # matched 语义 = 本轮确认匹配（UPDATE 幂等守卫只防重复写入）
    assert summary["matched"] == 1
    assert rec.status()["reconciled_total"] == 1
