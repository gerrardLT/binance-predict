"""报价 edge 实盘执行器单元测试（quote_momentum_v1 LIVE）。

覆盖：规则区间守卫（时点/报价/None/开关）、每窗一条、日单量护栏、
重启防重（DB 已有尝试）、成交/未成交路径、execute_signal_trade 执行价护栏。

不触网络/真实 DB：trader 与 DB 查询方法全部用替身/monkeypatch。
"""
from __future__ import annotations

import asyncio

import pytest

from binance_predict.config.settings import settings
from binance_predict.services import quote_edge_live_trader as qelt
from binance_predict.services.prediction_trading import BinancePredictionTrader
from binance_predict.services.quote_edge_live_trader import QuoteEdgeLiveTrader

WINDOW_START = 1_000_000_000_000          # 5m 窗口起点（ms）
WINDOW_END = WINDOW_START + 300_000


class _FakeOrder:
    def __init__(self, status="FILLED", error_message=None):
        self.status = status
        self.order_id = "ORD-1"
        self.token_id = "TOKEN-DOWN"
        self.amount_in = "5000000000000000000"
        self.error_message = error_message


class _FakeTrader:
    """execute_signal_trade 替身：记录调用、返回可配置结果。"""

    def __init__(self, result=object()):
        self.calls: list[dict] = []
        # 哨兵区分"未传参（默认 FILLED）"与"显式传 None（落库异常）"
        self.result = _FakeOrder() if isinstance(result, object) and type(result) is object else result

    async def execute_signal_trade(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _make_trader(monkeypatch, trader: _FakeTrader) -> QuoteEdgeLiveTrader:
    monkeypatch.setattr(settings, "quote_momentum_live_enabled", True)
    t = QuoteEdgeLiveTrader(trader)

    async def _no_filled(self):
        return 0

    async def _no_attempt(self, _ws):
        return False

    async def _no_backfill(self, _ws):
        return None

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_count_filled_today", _no_filled)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_has_attempt", _no_attempt)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_backfill_signal_link", _no_backfill)
    return t


async def _drain(t: QuoteEdgeLiveTrader) -> None:
    """等待在途任务（下单/回填）跑完。"""
    while t._tasks:
        await asyncio.sleep(0.01)


# ============================================================
# check() 守卫：区间/开关/None
# ============================================================

def test_check_disabled_no_fire(monkeypatch) -> None:
    monkeypatch.setattr(settings, "quote_momentum_live_enabled", False)
    t = QuoteEdgeLiveTrader(_FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is False


def test_check_none_price_no_fire(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, None) is False


def test_check_time_window_guards(monkeypatch) -> None:
    """t<90s 或 t≥120s → 不命中（半开区间 [90,120)）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 89_999, 0.71) is False
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 120_000, 0.71) is False


def test_check_quote_band_guards(monkeypatch) -> None:
    """q∉[0.69,0.75) → 不命中（上界开区间）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.68) is False
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.75) is False


def test_check_rule_bounds_frozen(monkeypatch) -> None:
    """规则常量必须与冻结口径一致（防再次口径漂移）。"""
    assert qelt.QUOTE_EDGE_RULES[qelt.LIVE_VERSION] == (90.0, 120.0, 0.69, 0.75)


@pytest.mark.asyncio
async def test_check_hit_fires_once_per_window(monkeypatch) -> None:
    """区间内首个采样开火；同窗后续采样不再开火（内存占位）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is True
    assert t.check(WINDOW_START, WINDOW_END, ts + 15_000, 0.72) is False
    assert WINDOW_START in t._fired
    await _drain(t)


# ============================================================
# _fire 路径：成交/护栏/防重
# ============================================================

@pytest.mark.asyncio
async def test_fire_success_calls_trader_and_backfills(monkeypatch) -> None:
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == "quote_momentum_v1"
    assert call["window_start"] == WINDOW_START
    assert call["amount_usdt"] == settings.quote_momentum_live_amount_usdt
    assert call["max_exec_price"] == settings.quote_momentum_live_max_exec_price
    assert t._fire_total == 1


@pytest.mark.asyncio
async def test_fire_daily_cap_blocks(monkeypatch) -> None:
    """日单量护栏：当日成交达上限 → 不调 trader。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)

    async def _full(self):
        return t._max_daily

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_count_filled_today", _full)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []
    assert t._fire_total == 0


@pytest.mark.asyncio
async def test_fire_restart_dedup_blocks(monkeypatch) -> None:
    """重启防重：DB 已有本窗尝试（FILLED/FAILED）→ 跳过不再下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)

    async def _has(self, _ws):
        return True

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_has_attempt", _has)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_fire_failed_order_no_crash(monkeypatch) -> None:
    """trader 返回 FAILED（如护栏弃单）→ 计入 fire_total、无回填、不抛异常。"""
    fake = _FakeTrader(result=_FakeOrder(status="FAILED", error_message="执行价护栏弃单"))
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert len(fake.calls) == 1
    assert t._fire_total == 1


@pytest.mark.asyncio
async def test_fire_trader_none_no_crash(monkeypatch) -> None:
    """trader 返回 None（落库异常）→ 不崩、不回填。"""
    fake = _FakeTrader(result=None)
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert t._fire_total == 0


def test_status_shape(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    s = t.status()
    assert s["enabled"] is True
    assert s["version"] == "quote_momentum_v1"
    assert s["fire_total"] == 0


# ============================================================
# execute_signal_trade：执行价护栏（真 trader，方法级 monkeypatch）
# ============================================================

def _make_real_trader(monkeypatch) -> BinancePredictionTrader:
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xWALLET"
    trader._wallet_id = "WID"

    async def _list():
        trader._down_token_id = "TOKEN-DOWN"
        return []

    monkeypatch.setattr(trader, "list_markets", _list)
    return trader


@pytest.mark.asyncio
async def test_signal_trade_price_guard_rejects(monkeypatch) -> None:
    """报价均价 0.82 > 上限 0.78 → 弃单落 FAILED，不 place_order。"""
    trader = _make_real_trader(monkeypatch)

    async def _quote(_token, _side, amount_usdt=None):
        return {"averagePrice": 0.82, "amountIn": "1", "quoteId": "Q1"}

    async def _place(_q):
        raise AssertionError("护栏弃单不应走到 place_order")

    saved: list[dict] = []

    async def _save_failed(prediction_id, error_msg, quote=None,
                           agent_prediction_id=None, signal_version=None,
                           window_start=None):
        saved.append({"error": error_msg, "signal_version": signal_version,
                      "window_start": window_start})
        return _FakeOrder(status="FAILED", error_message=error_msg)

    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "_save_failed_order", _save_failed)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order is not None and order.status == "FAILED"
    assert "执行价护栏" in saved[0]["error"]
    assert saved[0]["signal_version"] == "quote_momentum_v1"
    assert saved[0]["window_start"] == WINDOW_START


@pytest.mark.asyncio
async def test_signal_trade_success_writes_signal_fields(monkeypatch) -> None:
    """报价 0.71 ≤ 上限 → 下单成功，FILLED 订单带 signal_version/window_start。"""
    trader = _make_real_trader(monkeypatch)

    async def _quote(_token, _side, amount_usdt=None):
        assert amount_usdt == 5.0  # 自定义金额透传到报价
        return {"averagePrice": 0.71, "amountIn": "5", "amountOut": "7", "quoteId": "Q1"}

    async def _place(_q):
        return {"orderId": "ORD-9"}

    saved: list[dict] = []

    async def _save(**kwargs):
        saved.append(kwargs)
        return _FakeOrder(status="FILLED")

    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "_save_order", _save)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order is not None and order.status == "FILLED"
    assert saved[0]["status"] == "FILLED"
    assert saved[0]["signal_version"] == "quote_momentum_v1"
    assert saved[0]["window_start"] == WINDOW_START
    assert saved[0]["order_id"] == "ORD-9"
