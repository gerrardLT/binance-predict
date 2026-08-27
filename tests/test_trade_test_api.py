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
        direction="DOWN",
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
        ManualTradeTestRequest(amount_usdt=51.0), _=None)
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
    assert out["direction"] == "DOWN"  # 3b：direction 落库后回显
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
async def test_trade_test_success_invalidates_balance_cache(monkeypatch) -> None:
    """下单成功后作废 prediction-wallet 余额 TTL 缓存（前端实时余额依赖）。"""
    import time

    import binance_predict.main as m

    async def _exec(**kw):
        return _order()

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    # 模拟新鲜缓存（未过期）
    m._wallet_view_ts["balance"] = time.time()
    await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0), _=None)
    assert m._wallet_view_ts["balance"] == 0.0  # 已作废


@pytest.mark.asyncio
async def test_trade_test_order_none_keeps_balance_cache(monkeypatch) -> None:
    """未产生订单（None）不作废余额缓存（未花钱）。"""
    import time

    import binance_predict.main as m

    async def _exec(**kw):
        return None

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    fresh = time.time()
    m._wallet_view_ts["balance"] = fresh
    await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0), _=None)
    assert m._wallet_view_ts["balance"] == fresh  # 未被作废


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
        direction=None, settle_outcome=None, win=None, pnl=None, settled_at=None,
        redeemed_at=None,
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
    assert o["direction"] is None      # 旧数据无 direction：透传 null
    assert o["settled_at"] is None     # 未结算 → null
    assert o["redeemed_at"] is None    # 未领取 → null（新字段透传）
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
# GET /api/prediction-wallet 附余额（TTL 缓存 + 预测钱包余额探索）
# ============================================================

def _stub_wallet_view(monkeypatch, m):
    """隔离模块级钱包视图缓存（每测新 dict/新锁，防跨测污染与跨 loop 绑定）。"""
    import asyncio
    monkeypatch.setattr(m, "_wallet_view", {})
    monkeypatch.setattr(m, "_wallet_view_ts", {"static": 0.0, "balance": 0.0})
    monkeypatch.setattr(m, "_wallet_view_lock", asyncio.Lock())


@pytest.mark.asyncio
async def test_prediction_wallet_includes_spot_balance(monkeypatch) -> None:
    """prediction-wallet 返回体携带 spot_usdt_free（余额查询失败时为 None）。"""
    import binance_predict.main as m

    async def _wallet():
        return {"walletAddress": "0xddfe00000000000000000000000000000000d1a2",
                "walletId": "WID", "registeredTime": 1_787_410_421_564}

    async def _bal():
        return 12.34

    async def _pred(wallet=None):
        return {"usdt_free": None, "assets": None}

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    monkeypatch.setattr(m.prediction_trader, "fetch_prediction_wallet_balance", _pred)
    _stub_wallet_view(monkeypatch, m)
    out = await m.get_prediction_wallet(_=None)
    assert out["wallet_id"] == "WID"
    assert out["spot_usdt_free"] == 12.34
    # 预测余额探索失败 → 降级 None 不阻塞其余字段
    assert out["prediction_usdt_free"] is None
    assert out["wallet_assets"] is None
    assert out["prediction_balance_available"] is False


@pytest.mark.asyncio
async def test_prediction_wallet_includes_prediction_balance(monkeypatch) -> None:
    """预测余额探索成功 → prediction_usdt_free/wallet_assets/available 透传。"""
    import binance_predict.main as m

    async def _wallet():
        return {"walletAddress": "0x" + "a" * 40, "walletId": "WID", "registeredTime": 1}

    async def _bal():
        return 12.34

    assets = [{"asset": "USDT", "free": "5.5"}, {"asset": "BTC-UP-TOKEN", "balance": "2"}]

    async def _pred(wallet=None):
        return {"usdt_free": 5.5, "assets": assets}

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    monkeypatch.setattr(m.prediction_trader, "fetch_prediction_wallet_balance", _pred)
    _stub_wallet_view(monkeypatch, m)
    out = await m.get_prediction_wallet(_=None)
    assert out["prediction_usdt_free"] == 5.5
    assert out["wallet_assets"] == assets
    assert out["prediction_balance_available"] is True


@pytest.mark.asyncio
async def test_prediction_wallet_cache_ttl(monkeypatch) -> None:
    """TTL 缓存：新鲜期内二次调用零 API；余额时间戳置零（划转作废）后仅重拉余额。"""
    import binance_predict.main as m

    calls = {"wallet": 0, "spot": 0, "pred": 0}

    async def _wallet():
        calls["wallet"] += 1
        return {"walletAddress": "0x" + "a" * 40, "walletId": "WID", "registeredTime": 1}

    async def _bal():
        calls["spot"] += 1
        return 1.0

    async def _pred(wallet=None):
        calls["pred"] += 1
        return {"usdt_free": 2.0, "assets": None}

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    monkeypatch.setattr(m.prediction_trader, "fetch_prediction_wallet_balance", _pred)
    _stub_wallet_view(monkeypatch, m)
    ts = m._wallet_view_ts  # monkeypatch 后的同一个 dict

    out1 = await m.get_prediction_wallet(_=None)
    assert calls == {"wallet": 1, "spot": 1, "pred": 1}
    assert out1["prediction_usdt_free"] == 2.0

    # 新鲜期内（静态 300s / 余额 20s）：全缓存命中，零 API
    out2 = await m.get_prediction_wallet(_=None)
    assert calls == {"wallet": 1, "spot": 1, "pred": 1}
    assert out2 == out1

    # 余额缓存作废（transfer-in 成功路径同款操作）：仅余额重新拉，静态仍命中
    ts["balance"] = 0.0
    await m.get_prediction_wallet(_=None)
    assert calls == {"wallet": 1, "spot": 2, "pred": 2}


@pytest.mark.asyncio
async def test_prediction_wallet_static_failure_backoff(monkeypatch) -> None:
    """静态信息失败：返回 error；退避期内重试一次后不再打 API。"""
    import binance_predict.main as m

    calls = {"wallet": 0}

    async def _wallet():
        calls["wallet"] += 1
        return None

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", AsyncMock(return_value=1.0))
    monkeypatch.setattr(m.prediction_trader, "fetch_prediction_wallet_balance",
                        AsyncMock(return_value={"usdt_free": None, "assets": None}))
    _stub_wallet_view(monkeypatch, m)

    out = await m.get_prediction_wallet(_=None)
    assert "未找到预测钱包" in out["error"]
    # 退避期内（60s）：不再重试 wallet/list，仍返回 error
    out2 = await m.get_prediction_wallet(_=None)
    assert "error" in out2
    assert calls["wallet"] == 1


# ============================================================
# 服务层：fetch_prediction_wallet_balance 两级探索
# ============================================================

@pytest.mark.asyncio
async def test_fetch_prediction_balance_from_wallet_dict(monkeypatch) -> None:
    """①级探索：wallet dict 内嵌余额字段命中（usdtBalance）。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"

    async def _assets():
        return None  # asset/list 探索失败（降级）

    monkeypatch.setattr(trader, "_fetch_prediction_asset_list", _assets)
    wallet = {"walletAddress": "0xW", "walletId": "WID", "usdtBalance": "5.5"}
    out = await trader.fetch_prediction_wallet_balance(wallet=wallet)
    assert out["usdt_free"] == 5.5
    assert out["assets"] is None


@pytest.mark.asyncio
async def test_fetch_prediction_balance_from_asset_list(monkeypatch) -> None:
    """②级探索：wallet 无内嵌字段时从 asset/list 的 USDT 条目兑底。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"

    assets = [{"asset": "BTC-UP-TOKEN", "balance": "1.5"},
              {"asset": "USDT", "free": "3.25"}]

    async def _assets():
        return assets

    monkeypatch.setattr(trader, "_fetch_prediction_asset_list", _assets)
    out = await trader.fetch_prediction_wallet_balance(wallet={"walletId": "WID"})
    assert out["usdt_free"] == 3.25
    assert out["assets"] == assets


@pytest.mark.asyncio
async def test_fetch_prediction_balance_all_failed(monkeypatch) -> None:
    """两级全失败 → (None, None)，不抛异常（探索型端点天然 fallback）。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"

    async def _wallet():
        return None

    async def _assets():
        return None

    monkeypatch.setattr(trader, "fetch_wallet_info", _wallet)
    monkeypatch.setattr(trader, "_fetch_prediction_asset_list", _assets)
    out = await trader.fetch_prediction_wallet_balance()
    assert out == {"usdt_free": None, "assets": None}


def test_balance_endpoint_path_is_official() -> None:
    """余额端点必须是官方 payment-options（旧 asset/list 实测 404，防回退）。"""
    from binance_predict.services import prediction_trading as pt

    assert pt._ASSET_LIST_PATH == (
        "/sapi/v1/w3w/wallet/prediction/balance/payment-options"
    )


@pytest.mark.asyncio
async def test_payment_options_response_parsing(monkeypatch) -> None:
    """payment-options 官方响应结构（2026-08-23 生产实测收敛）：
    {"items": [{accountType/availableBalanceDisplay/enabled}...]}，
    prediction_usdt_free 取 CeDeFi（预测钱包）那项。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"items": [
        {"accountType": "CeDeFi", "availableBalanceDisplay": "1.94", "enabled": True},
        {"accountType": "SPOT", "availableBalanceDisplay": "104.83", "enabled": True},
        {"accountType": "FUNDING", "availableBalanceDisplay": "0.00", "enabled": True},
    ]}
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    out = await trader.fetch_prediction_wallet_balance(wallet={"walletId": "WID"})
    assert out["usdt_free"] == 1.94  # CeDeFi（预测钱包）条目命中，非 SPOT
    assert out["assets"] == resp.json()["items"]


@pytest.mark.asyncio
async def test_payment_options_cedefi_disabled_falls_back(monkeypatch) -> None:
    """CeDeFi 条目 enabled=False 时跳过（不误取不可用支付方式）。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"items": [
        {"accountType": "CeDeFi", "availableBalanceDisplay": "9.9", "enabled": False},
        {"accountType": "SPOT", "availableBalanceDisplay": "104.83", "enabled": True},
    ]}
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    out = await trader.fetch_prediction_wallet_balance(wallet={"walletId": "WID"})
    assert out["usdt_free"] is None  # CeDeFi 禁用 + 无 USDT 条目 → 不误取 SPOT


@pytest.mark.asyncio
async def test_payment_options_signature_matches_sent_query(monkeypatch) -> None:
    """-1022 回归锁：发送的 query string 必须与签名字符串逐字节一致。

    根因（2026-08-23 生产实锤）：_sign_request 按字母序签名，但 httpx
    params= 按 dict 插入序发送，币安按收到的原文验签 → -1022。
    修复后必须走 _build_signed_url（手动拼 URL，签名串=发送串）。
    """
    import hashlib
    import hmac as hmac_mod
    from urllib.parse import parse_qsl

    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    captured: dict = {}
    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"options": []}

    async def _get(url, headers=None):
        captured["url"] = str(url)
        return resp

    client = MagicMock()
    client.get = _get
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    await trader._fetch_prediction_asset_list()
    qs = captured["url"].split("?", 1)[1]
    pairs = parse_qsl(qs)
    sig = dict(pairs)["signature"]
    no_sig = [f"{k}={v}" for k, v in pairs if k != "signature"]
    # ① 发送串本身必须按字母序（签名时即此序）
    assert no_sig == sorted(no_sig)
    # ② signature 必须是对发送串原文的 HMAC-SHA256（逐字节一致才验签通过）
    expected = hmac_mod.new(
        b"s", "&".join(no_sig).encode(), hashlib.sha256
    ).hexdigest()
    assert sig == expected
    # ③ 业务参数确实在请求里（walletId/walletAddress）
    assert dict(pairs)["walletId"] == "WID"
    assert dict(pairs)["walletAddress"] == "0xW"


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
# POST /api/prediction/transfer-out（预测钱包→现货划出，P1-1）
# ============================================================

@pytest.mark.asyncio
async def test_transfer_out_amount_bounds(monkeypatch) -> None:
    """金额越界（0.05 / 21）→ 拒绝且不触达 transfer_out。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferOutboundRequest

    async def _never(_amount):  # pragma: no cover - 不应被调用
        raise AssertionError("越界金额不应触达 transfer_out")

    monkeypatch.setattr(m.prediction_trader, "transfer_out", _never)
    out1 = await m.prediction_transfer_out(TransferOutboundRequest(amount_usdt=0.05), _=None)
    out2 = await m.prediction_transfer_out(TransferOutboundRequest(amount_usdt=21.0), _=None)
    assert "error" in out1 and "error" in out2


@pytest.mark.asyncio
async def test_transfer_out_success_direction_confirmed(monkeypatch) -> None:
    """成功且现货余额增加 → direction_confirmed=True（官方命名反转下的方向自证）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferOutboundRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _ok(amount):
        assert amount == 0.1
        return {"transferId": "T-2"}

    balances = iter([100.0, 100.1])  # 划转前 → 划转后

    async def _bal():
        return next(balances)

    monkeypatch.setattr(m.prediction_trader, "transfer_out", _ok)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    out = await m.prediction_transfer_out(TransferOutboundRequest(amount_usdt=0.1), _=None)
    assert out["status"] == "SUCCESS"
    assert out["spot_before"] == 100.0
    assert out["spot_after"] == 100.1
    assert out["direction_confirmed"] is True
    assert "warning" not in out


@pytest.mark.asyncio
async def test_transfer_out_direction_unconfirmed_warns(monkeypatch) -> None:
    """划转返回成功但现货未见增加 → direction_confirmed=False + warning 人工核对。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferOutboundRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _ok(_amount):
        return {"transferId": "T-3"}

    async def _bal():
        return 100.0  # 前后不变（入账延迟/方向异常）

    monkeypatch.setattr(m.prediction_trader, "transfer_out", _ok)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    out = await m.prediction_transfer_out(TransferOutboundRequest(amount_usdt=1.0), _=None)
    assert out["status"] == "SUCCESS"
    assert out["direction_confirmed"] is False
    assert "人工核对" in out["warning"]


@pytest.mark.asyncio
async def test_transfer_out_failure_passthrough(monkeypatch) -> None:
    """失败 → FAILED + last_api_error 透传。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import TransferOutboundRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _fail(_amount):
        return None

    async def _bal():
        return 100.0

    monkeypatch.setattr(m.prediction_trader, "transfer_out", _fail)
    monkeypatch.setattr(m.prediction_trader, "fetch_spot_usdt_balance", _bal)
    monkeypatch.setattr(m.prediction_trader, "last_api_error", "HTTP 400: -9000")
    out = await m.prediction_transfer_out(TransferOutboundRequest(amount_usdt=1.0), _=None)
    assert out["status"] == "FAILED"
    assert "-9000" in out["error"]


@pytest.mark.asyncio
async def test_transfer_paths_direction_guard(monkeypatch) -> None:
    """防方向反转回归（服务层）：transfer_in → transfer/outbound（入金），
    transfer_out → transfer/inbound（提走，官方命名反转）。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_id = "WID"
    trader._wallet_address = "0xW"

    seen: list[str] = []

    def _signed(path, params):
        seen.append(path)
        return f"https://api.binance.com{path}?signed"

    monkeypatch.setattr(trader, "_build_signed_url", _signed)

    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"transferId": "T"}
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    assert await trader.transfer_in(1.0) == {"transferId": "T"}
    assert await trader.transfer_out(1.0) == {"transferId": "T"}
    assert seen == [
        "/sapi/v1/w3w/wallet/prediction/transfer/outbound",
        "/sapi/v1/w3w/wallet/prediction/transfer/inbound",
    ]


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


# ============================================================
# 奖金领取（GET redeemable / POST redeem，batch-redeem）
# ============================================================

@pytest.mark.asyncio
async def test_fetch_pending_claim_positions_parsing(monkeypatch) -> None:
    """PENDING_CLAIM 查询：{positions: [...]} 包裹解析 + 原文透传。

    端点响应结构未实测（探索型）：兼容 positions/items/orders/data/list 包裹，
    last_pending_raw 必须写入供生产收敛。
    """
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"positions": [
        {"tokenId": "T-13", "amount": "1"},
        {"tokenId": "T-14", "amount": "2"},
    ]}
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    out = await trader.fetch_pending_claim_positions()
    assert out == [{"tokenId": "T-13", "amount": "1"}, {"tokenId": "T-14", "amount": "2"}]
    assert trader.last_pending_error is None
    assert "T-13" in (trader.last_pending_raw or "")


@pytest.mark.asyncio
async def test_fetch_pending_claim_positions_fail_degrades(monkeypatch) -> None:
    """HTTP 失败 → None + last_pending_error（降级不抛，端点兑底 DB 源）。"""
    import httpx
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    resp = SimpleNamespace(status_code=400)
    resp.raise_for_status = lambda: (_ for _ in ()).throw(
        httpx.HTTPStatusError("400", request=MagicMock(), response=resp)
    )
    resp.text = '{"code":-1121,"msg":"Invalid symbol."}'
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    out = await trader.fetch_pending_claim_positions()
    assert out is None
    assert "-1121" in (trader.last_pending_error or "")


def test_extract_token_ids_variants() -> None:
    """tokenId/token_id/id 键兼容 + 去重保序 + 非字典项跳过。"""
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    out = trader._extract_token_ids([
        {"tokenId": "T-1"},
        {"token_id": "T-2"},
        {"id": "T-3"},
        {"tokenId": "T-1"},   # 重复
        "not-a-dict",          # 跳过
        {"tokenId": ""},       # 空串跳过
    ])
    assert out == ["T-1", "T-2", "T-3"]


@pytest.mark.asyncio
async def test_redeem_tokens_signature_matches_sent_query(monkeypatch) -> None:
    """-1022 回归锁（batch-redeem）：POST 的 query 必须字母序且签名=发送串原文；
    tokenIds 逗号拼接在 query 里（与 transfer 同口径）。"""
    import hashlib
    import hmac as hmac_mod
    from urllib.parse import parse_qsl

    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    captured: dict = {}
    resp = SimpleNamespace(status_code=200)
    resp.raise_for_status = lambda: None
    resp.json = lambda: {"success": True}

    async def _post(url, headers=None):
        captured["url"] = str(url)
        return resp

    client = MagicMock()
    client.post = _post
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    out = await trader.redeem_tokens(["T-13", "T-14"])
    assert out == {"success": True}
    assert captured["url"].startswith(
        "https://api.binance.com/sapi/v1/w3w/wallet/prediction/batch-redeem?"
    )
    pairs = parse_qsl(captured["url"].split("?", 1)[1])
    d = dict(pairs)
    assert d["tokenIds"] == "T-13,T-14"
    sig = d["signature"]
    no_sig = [f"{k}={v}" for k, v in pairs if k != "signature"]
    assert no_sig == sorted(no_sig)  # 发送串字母序（签名时即此序）
    expected = hmac_mod.new(b"s", "&".join(no_sig).encode(), hashlib.sha256).hexdigest()
    assert sig == expected


@pytest.mark.asyncio
async def test_redeem_tokens_empty_and_fail(monkeypatch) -> None:
    """空列表 → NOOP+last_api_error；HTTP 失败 → None+错误透传。"""
    import httpx
    from binance_predict.services.prediction_trading import BinancePredictionTrader

    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_id = "WID"
    trader._wallet_address = "0xW"

    assert await trader.redeem_tokens([]) is None
    assert "token_ids" in (trader.last_api_error or "")

    resp = SimpleNamespace(status_code=400)
    resp.raise_for_status = lambda: (_ for _ in ()).throw(
        httpx.HTTPStatusError("400", request=MagicMock(), response=resp)
    )
    resp.text = '{"code":-1022,"msg":"Signature for this request is not valid."}'
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    monkeypatch.setattr(trader, "_get_client", lambda: client)

    assert await trader.redeem_tokens(["T-1"]) is None
    assert "-1022" in (trader.last_api_error or "")


@pytest.mark.asyncio
async def test_prediction_redeemable_merges_sources(monkeypatch) -> None:
    """redeemable 端点：钱包查询成功时链上事实为权威（DB 独有 token 不并入，
    防止送已不可赎的 token 给 batch-redeem 触发币安 400 -9000）；
    官方端点降级时 wallet_source=degraded 且 DB 源兜底。"""
    import binance_predict.main as m

    async def _positions():
        return [{"tokenId": "T-13"}]

    monkeypatch.setattr(m.prediction_trader, "fetch_pending_claim_positions", _positions)
    monkeypatch.setattr(m.prediction_trader, "last_pending_error", None)
    monkeypatch.setattr(m.prediction_trader, "last_pending_raw", '{"positions":[...]}')

    row = SimpleNamespace(id=13, token_id="T-13")   # DB 与钱包重叠
    row2 = SimpleNamespace(id=14, token_id="T-14")  # DB 独有（链上无 → 不得计入可领）
    db = AsyncMock()
    result = MagicMock()
    result.all.return_value = [row, row2]
    db.execute = AsyncMock(return_value=result)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(m, "async_session_factory", _factory)

    out = await m.prediction_redeemable(_=None)
    # 钱包源成功：链上事实为权威，DB 独有的 T-14 不并入
    assert out["claimable_tokens"] == ["T-13"]
    assert out["claimable_count"] == 1
    assert out["wallet_source"] == "ok"
    assert out["db_win_unclaimed_ids"] == [13, 14]

    # 官方端点降级：DB 源兜底
    async def _fail():
        return None

    monkeypatch.setattr(m.prediction_trader, "fetch_pending_claim_positions", _fail)
    out2 = await m.prediction_redeemable(_=None)
    assert out2["wallet_source"] == "degraded"
    assert out2["claimable_tokens"] == ["T-13", "T-14"]


@pytest.mark.asyncio
async def test_prediction_redeem_success_marks_db_and_cache(monkeypatch) -> None:
    """redeem 端点：自动收集 token → batch-redeem 成功 → 标记 DB redeemed_at
    + 作废余额缓存（领取入 CeDeFi，前端下次轮询即新值）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import RedeemRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _positions():
        return [{"tokenId": "T-13"}, {"token_id": "T-14"}]

    redeemed: list = []

    async def _redeem(token_ids):
        redeemed.append(list(token_ids))
        return {"success": True}

    monkeypatch.setattr(m.prediction_trader, "fetch_pending_claim_positions", _positions)
    monkeypatch.setattr(m.prediction_trader, "redeem_tokens", _redeem)

    db = AsyncMock()
    db.execute = AsyncMock()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(m, "async_session_factory", _factory)

    m._wallet_view_ts["balance"] = 123.0  # 预置非零，验证被作废
    out = await m.prediction_redeem(RedeemRequest(), _=None)

    assert out["status"] == "SUCCESS"
    assert out["redeemed"] == 2
    assert redeemed == [["T-13", "T-14"]]
    db.execute.assert_awaited()  # update redeemed_at 语句已发
    db.commit.assert_awaited_once()
    assert m._wallet_view_ts["balance"] == 0.0  # 余额缓存已作废


@pytest.mark.asyncio
async def test_prediction_redeem_noop(monkeypatch) -> None:
    """两源皆空 → NOOP（不触达 batch-redeem）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import RedeemRequest

    monkeypatch.setattr(m.prediction_trader, "_api_key", "k")
    monkeypatch.setattr(m.prediction_trader, "_wallet_address", "0xW")
    monkeypatch.setattr(m.prediction_trader, "_wallet_id", "WID")

    async def _positions():
        return []

    async def _never(_ids):  # pragma: no cover - 不应被调用
        raise AssertionError("无 token 不应触达 batch-redeem")

    monkeypatch.setattr(m.prediction_trader, "fetch_pending_claim_positions", _positions)
    monkeypatch.setattr(m.prediction_trader, "redeem_tokens", _never)

    db = AsyncMock()
    result = MagicMock()
    result.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(m, "async_session_factory", _factory)

    out = await m.prediction_redeem(RedeemRequest(), _=None)
    assert out["status"] == "NOOP"
