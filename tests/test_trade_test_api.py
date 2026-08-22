"""实盘链路人工测试单端点单元测试（2026-08-22）：
POST /api/trade/test（main.manual_trade_test）。

不触网络/真实 DB：prediction_trader.execute_signal_trade 用替身。
核心分支：
- 金额越界（<0.1 / >5）与方向非法 → 参数校验拒绝（不碰 trader）
- order=None（key 未配/钱包失败/同窗已有测试单）→ error 提示
- FILLED → 关键字段回显（status/order_id/average_price）
- FAILED（如未找到 token/报价失败）→ status=FAILED + error_message 透传
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from binance_predict.models.schemas import ManualTradeTestRequest


def _order(**over) -> SimpleNamespace:
    base = dict(
        status="FILLED", order_id="ORD-1", signal_version="manual_test",
        window_start=1_000_000_000_000, token_id="TOKEN-DOWN",
        quote_json={"averagePrice": 0.5, "amountIn": "1000000"},
        error_message=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


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
                      quote_json={}, error_message="未找到 DOWN 方向的 token")

    monkeypatch.setattr(m.prediction_trader, "execute_signal_trade", _exec)
    out = await m.manual_trade_test(
        ManualTradeTestRequest(amount_usdt=1.0), _=None)
    assert out["status"] == "FAILED"
    assert "未找到 DOWN" in out["error_message"]
