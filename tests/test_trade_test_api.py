"""实盘链路人工测试单端点单元测试（2026-08-22）：
POST /api/trade/test（main.manual_trade_test）。

不触网络/真实 DB：prediction_trader.execute_signal_trade 用替身。
注：execute_signal_trade 返回 dict 快照（非 ORM 对象，避免脱离会话
访问属性报 DetachedInstanceError 导致端点 500 + 行卡 PENDING）。
核心分支：
- 金额越界（<0.1 / >5）与方向非法 → 参数校验拒绝（不碰 trader）
- order=None（key 未配/钱包失败/同窗已有测试单）→ error 提示
- FILLED → 关键字段回显（status/order_id/average_price）
- FAILED（如未找到 token/报价失败）→ status=FAILED + error_message 透传
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from binance_predict.models.schemas import ManualTradeTestRequest


def _order(**over) -> dict:
    """execute_signal_trade 返回的 dict 快照替身。"""
    base = dict(
        id=1, status="FILLED", order_id="ORD-1", signal_version="manual_test",
        window_start=1_000_000_000_000, token_id="TOKEN-DOWN",
        amount_in="1000000", average_price=0.5, error_message=None,
    )
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_trade_test_amount_bounds_rejected(monkeypatch) -> None:
    """金额越界（0.05 / 6.0）→ 拒绝且不调 trader。"""
    import binance_predict.main as m

    called = []

    async def _exec(**kw):
        called.append(kw)
        return _order()

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out1 = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=0.05), _=None)
    out2 = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=6.0), _=None)
    assert "error" in out1 and "error" in out2
    assert called == []


@pytest.mark.asyncio
async def test_trade_test_bad_prediction_rejected(monkeypatch) -> None:
    import binance_predict.main as m

    async def _exec(**kw):  # pragma: no cover - 不应被调用
        raise AssertionError("非法方向不应触达 trader")

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0, prediction="SIDEWAYS"), _=None)
    assert "仅允许 UP/DOWN" in out["error"]


@pytest.mark.asyncio
async def test_trade_test_success_fields(monkeypatch) -> None:
    """FILLED → 关键字段回显；window_start 是 5m 对齐值、版本固定 manual_test。"""
    import binance_predict.main as m

    seen = {}

    async def _exec(**kw):
        seen.update(kw)
        return _order()

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0, prediction="DOWN"), _=None)

    assert out["status"] == "FILLED"
    assert out["order_id"] == "ORD-1"
    assert out["average_price"] == 0.5
    assert out["error_message"] is None
    assert seen["prediction"] == "DOWN"
    assert seen["amount_usdt"] == 1.0
    assert seen["signal_version"] == "manual_test"
    assert seen["window_start"] % 300_000 == 0  # 5m 窗口对齐
    assert "max_exec_price" not in seen or seen["max_exec_price"] is None


@pytest.mark.asyncio
async def test_trade_test_order_none_returns_error(monkeypatch) -> None:
    """trader 返回 None（key 未配/钱包失败/同窗占位）→ error + window_start。"""
    import binance_predict.main as m

    async def _exec(**kw):
        return None

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0), _=None)
    assert "下单未执行" in out["error"]
    assert out["window_start"] % 300_000 == 0


@pytest.mark.asyncio
async def test_trade_test_failed_order_passthrough(monkeypatch) -> None:
    """FAILED（如未找到 token）→ status=FAILED + error_message 透传。"""
    import binance_predict.main as m

    async def _exec(**kw):
        return _order(status="FAILED", order_id=None, token_id="",
                      average_price=None, amount_in=None,
                      error_message="未找到 DOWN 方向的 token")

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0), _=None)
    assert out["status"] == "FAILED"
    assert "未找到 DOWN" in out["error_message"]


# ============================================================
# GET /api/trades/recent（实盘面板订单历史）
# ============================================================

@pytest.mark.asyncio
async def test_recent_trades_fields_and_limit() -> None:
    """订单行字段透传（含 quote_json 提取的 average_price）；limit 截断到 1~100。"""
    import binance_predict.main as m

    row = SimpleNamespace(
        id=1, signal_version="manual_test", window_start=1_787_412_600_000,
        status="FAILED", order_id=None, token_id=None, amount_in=None,
        quote_json={"averagePrice": 0.5},
        error_message="获取报价失败 | HTTP 400: not enough USDT",
        created_at=datetime(2026, 8, 22, 15, 35, tzinfo=timezone.utc),
    )
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [row]
    db.execute = AsyncMock(return_value=result)

    out = await m.get_recent_trades(limit=200, _=None, db=db)
    assert len(out["orders"]) == 1
    o = out["orders"][0]
    assert o["signal_version"] == "manual_test"
    assert o["status"] == "FAILED"
    assert o["average_price"] == 0.5
    assert "HTTP 400" in o["error_message"]
    assert o["created_at"].startswith("2026-08-22")


@pytest.mark.asyncio
async def test_recent_trades_empty() -> None:
    import binance_predict.main as m

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    out = await m.get_recent_trades(limit=20, _=None, db=db)
    assert out["orders"] == []


# ============================================================
# GET /api/prediction-wallet 附现货 USDT 余额
# ============================================================

@pytest.mark.asyncio
async def test_prediction_wallet_includes_spot_balance(monkeypatch) -> None:
    """prediction-wallet 返回体携带 spot_usdt_free（余额查询失败时为 None）。"""
    import binance_predict.main as m

    async def _wallet():
        return {"walletAddress": "0xddfe00000000000000000000000000000000d1a2",
                "walletId": "WID", "registeredTime": 1_787_410_421_564}

    async def _bal():
        return 12.34

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    out = await m.get_prediction_wallet(_=None)
    assert out["wallet_id"] == "WID"
    assert out["spot_usdt_free"] == 12.34


@pytest.mark.asyncio
async def test_fetch_spot_usdt_balance_parses_free(monkeypatch) -> None:
    """服务层：从 /api/v3/account balances 取 USDT 的 free。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"

    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"balances": [
        {"asset": "BTC", "free": "0.1", "locked": "0"},
        {"asset": "USDT", "free": "12.34", "locked": "1"},
    ]}
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    assert await trader.fetch_spot_usdt_balance() == 12.34


# ============================================================
# POST /api/prediction/transfer-in（现货→预测钱包划转）
# ============================================================

@pytest.mark.asyncio
async def test_transfer_in_amount_bounds(monkeypatch) -> None:
    """金额越界（0.05 / 21）→ 拒绝且不触达 transfer_in。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferInboundRequest

    async def _never(_amount):  # pragma: no cover - 不应被调用
        raise AssertionError("越界金额不应触达 transfer_in")

    monkeypatch.setattr(m.prediction_trader, "transfer_in", _never)
    out1 = await m.prediction_transfer_in(TransferInboundRequest(amount_usdt=0.05), _=None)
    out2 = await m.prediction_transfer_in(TransferInboundRequest(amount_usdt=21.0), _=None)
    assert "error" in out1 and "error" in out2


@pytest.mark.asyncio
async def test_transfer_in_success_and_failure(monkeypatch) -> None:
    """成功 → SUCCESS + 刷新后现货余额；失败 → FAILED + last_api_error 透传。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferInboundRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _ok(amount):
        assert amount == 5.0
        return {"transferId": "T-1"}

    async def _bal():
        return 103.83

    monkeypatch.setattr(m.prediction_trader, "transfer_in", _ok)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    out = await m.prediction_transfer_in(TransferInboundRequest(amount_usdt=5.0), _=None)
    assert out["status"] == "SUCCESS"
    assert out["spot_usdt_free"] == 103.83

    async def _fail(_amount):
        return None

    monkeypatch.setattr(m.prediction_trader, "transfer_in", _fail)
    monkeypatch.setattr(m.prediction_trader, "last_api_error", "HTTP 400: -9000")
    out2 = await m.prediction_transfer_in(TransferInboundRequest(amount_usdt=5.0), _=None)
    assert out2["status"] == "FAILED"
    assert "-9000" in out2["error"]


# ============================================================
# POST /api/trades/sync-binance（对账回填卡 PENDING 行）
# ============================================================

@pytest.mark.asyncio
async def test_sync_binance_backfills_pending_rows(monkeypatch) -> None:
    """币安侧 FILLED 订单按 slug 窗口秒匹配 → PENDING 行回填 FILLED；
    币安侧无对应的行保持 PENDING 不动。"""
    import binance_predict.main as m

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")

    async def _history(limit=50):
        return [{"orderId": "B-1", "slug": "btc-updown-5m-1787418600",
                 "status": "FILLED", "filledUsdtAmount": "1",
                 "price": "0.55", "filledShareQty": "1.79"}]

    monkeypatch.setattr(m.prediction_trader, "query_order_history", _history)

    matched = SimpleNamespace(
        id=7, window_start=1_787_418_600_000, status="PENDING",
        order_id=None, amount_in="0", quote_json=None, error_message=None)
    unmatched = SimpleNamespace(
        id=9, window_start=1_787_419_200_000, status="PENDING",
        order_id=None, amount_in="0", quote_json=None, error_message=None)

    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = [matched, unmatched]
    db.execute = AsyncMock(return_value=result)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(m, "async_session_factory", _factory)

    out = await m.sync_binance_orders(_=None)
    assert out["synced"] == 1
    assert out["details"][0]["id"] == 7
    assert matched.status == "FILLED"
    assert matched.order_id == "B-1"
    assert matched.amount_in == str(10 ** 18)
    assert matched.quote_json["averagePrice"] == 0.55
    assert matched.quote_json["source"] == "binance_history_sync"
    assert unmatched.status == "PENDING"  # 无对应币安订单 → 不动
    db.commit.assert_awaited_once()


# ============================================================
# GET /api/prediction/quote-preview（报价预览：倒计时/指示价）
# ============================================================

def _fresh_lock():
    """新锁隔离事件循环：模块级 _state_lock 可能已绑定其他测试的 loop
    （Python 3.10+ asyncio.Lock 首次 await 时绑定 loop，跨 loop 复用会报错）。"""
    import asyncio
    return asyncio.Lock()


@pytest.mark.asyncio
async def test_quote_preview_snapshot_fields(monkeypatch) -> None:
    """_pm_market_info 有数据 → 快照字段 + server_now_ms + stale=False。"""
    import binance_predict.main as m

    monkeypatch.setattr(m, "_state_lock", _fresh_lock())
    monkeypatch.setattr(m, "_pm_market_info", {
        "start_date": 1_000_000_000_000,
        "end_date": 1_000_000_300_000,
        "up_price": 0.45,
        "down_price": 0.55,
        "participant_count": 123,
    })
    out = await m.get_quote_preview(_=None)
    assert out["stale"] is False
    assert out["window_start"] == 1_000_000_000_000
    assert out["window_end"] == 1_000_000_300_000
    assert out["up_price"] == 0.45
    assert out["down_price"] == 0.55
    assert isinstance(out["server_now_ms"], int)
    # 纯快照：不携带多余字段（participant_count 等仅供图表 API）
    assert "participant_count" not in out


@pytest.mark.asyncio
async def test_quote_preview_stale_when_empty(monkeypatch) -> None:
    """_pm_market_info 为空（启动初期/采样失败）→ stale=True 其余 None。"""
    import binance_predict.main as m

    monkeypatch.setattr(m, "_state_lock", _fresh_lock())
    monkeypatch.setattr(m, "_pm_market_info", {})
    out = await m.get_quote_preview(_=None)
    assert out["stale"] is True
    assert out["window_start"] is None
    assert out["window_end"] is None
    assert out["up_price"] is None
    assert out["down_price"] is None
    assert isinstance(out["server_now_ms"], int)


@pytest.mark.asyncio
async def test_quote_preview_none_dates(monkeypatch) -> None:
    """有部分数据但 start/end 缺失 → 日期字段 None 但 stale=False（防御分支）。"""
    import binance_predict.main as m

    monkeypatch.setattr(m, "_state_lock", _fresh_lock())
    monkeypatch.setattr(m, "_pm_market_info", {
        "up_price": 0.5, "down_price": 0.5,
    })
    out = await m.get_quote_preview(_=None)
    assert out["stale"] is False
    assert out["window_start"] is None
    assert out["window_end"] is None
    assert out["up_price"] == 0.5
