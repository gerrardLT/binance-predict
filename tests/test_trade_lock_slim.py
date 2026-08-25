"""R7 锁瘦身的单元测试（2026-08-25 风险评审）。

覆盖：
- execute_signal_trade 获锁后复查决策点时限：超期不落占位直接放弃（不烧窗口槽位）
- 行情拉取移出 trade_lock：TTL 内复用缓存；空结果（拉取失败）不刷时间戳，
  下次调用立即重试——故障不被 TTL 掩盖

不触网络/真实 DB：锁内链路除被测逻辑外全桩（仿 test_order_reconciliation 模式）。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import binance_predict.services.prediction_trading as pt
from binance_predict.services.prediction_trading import BinancePredictionTrader

WS = 1_787_418_600_000          # 5m 窗口起点 ms


def _make_trader(monkeypatch) -> tuple[BinancePredictionTrader, list, list]:
    """构造可下单的 trader 替身，返回 (trader, list_calls, reserve_calls)。"""
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xW"
    trader._wallet_id = "WID"

    list_calls: list[int] = []
    reserve_calls: list[int] = []

    async def _list():
        list_calls.append(1)
        trader._down_token_id = "T-DOWN"
        return [{"marketTopicId": "M1"}]   # 非空 → _ensure_markets_fresh 刷时间戳

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        reserve_calls.append(1)
        return SimpleNamespace(id=1, status="PENDING")

    async def _quote(*a, **k):
        return {"averagePrice": 0.5, "amountIn": "5", "amountOut": "10",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        return {"orderId": "O1", "status": "FILLED"}

    async def _update(order, status, **kwargs):
        return {"status": status, **kwargs}

    monkeypatch.setattr(trader, "list_markets", _list)
    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    return trader, list_calls, reserve_calls


# ============================================================
# 获锁后决策点时限复查
# ============================================================

@pytest.mark.asyncio
async def test_deadline_expired_after_lock_aborts_without_slot(monkeypatch) -> None:
    """deadline_ms 已过（等锁期间越过决策点）→ 返回 None，不占窗口槽位。"""
    trader, _lists, reserves = _make_trader(monkeypatch)
    out = await trader.execute_signal_trade(
        "DOWN", 5.0, "sig", WS, deadline_ms=int(time.time() * 1000) - 1)
    assert out is None
    assert reserves == []   # 未落占位：窗口槽位未烧，不追单


@pytest.mark.asyncio
async def test_deadline_in_future_proceeds(monkeypatch) -> None:
    """deadline_ms 未到 → 正常走完下单链路（复查不误伤）。"""
    trader, _lists, reserves = _make_trader(monkeypatch)
    out = await trader.execute_signal_trade(
        "DOWN", 5.0, "sig", WS, deadline_ms=int(time.time() * 1000) + 60_000)
    assert out is not None and out["status"] == "FILLED"
    assert reserves == [1]


@pytest.mark.asyncio
async def test_deadline_none_keeps_legacy_behavior(monkeypatch) -> None:
    """不传 deadline_ms（场景/人工测试单）→ 无时限复查，行为同旧版。"""
    trader, _lists, reserves = _make_trader(monkeypatch)
    out = await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    assert out is not None and out["status"] == "FILLED"
    assert reserves == [1]


# ============================================================
# 行情 TTL 缓存（锁外拉取）
# ============================================================

@pytest.mark.asyncio
async def test_execute_signal_trade_reuses_markets_cache(monkeypatch) -> None:
    """两次下单都在 TTL 内 → list_markets 只拉一次（缓存复用，锁内不再拉行情）。"""
    trader, lists, _reserves = _make_trader(monkeypatch)
    monkeypatch.setattr(pt, "_MARKETS_CACHE_TTL_S", 600.0)  # 避免测试期跨 30s 边界
    await trader.execute_signal_trade("DOWN", 5.0, "sig", WS)
    await trader.execute_signal_trade("DOWN", 5.0, "sig", WS + 300_000)
    assert len(lists) == 1


@pytest.mark.asyncio
async def test_ensure_markets_fresh_refetches_after_ttl(monkeypatch) -> None:
    """缓存过期 → 重新拉取（时间戳推进由真实 monotonic 驱动，TTL 调小验证）。"""
    trader, lists, _reserves = _make_trader(monkeypatch)
    monkeypatch.setattr(pt, "_MARKETS_CACHE_TTL_S", 0.0)   # 每次都算过期
    await trader._ensure_markets_fresh()
    await trader._ensure_markets_fresh()
    assert len(lists) == 2


@pytest.mark.asyncio
async def test_ensure_markets_fresh_empty_result_no_ts(monkeypatch) -> None:
    """拉取结果为空（网络故障/翻页全败）→ 不刷时间戳，下次立即重试。"""
    trader, lists, _reserves = _make_trader(monkeypatch)
    monkeypatch.setattr(pt, "_MARKETS_CACHE_TTL_S", 600.0)

    async def _empty():
        lists.append(1)
        return []

    monkeypatch.setattr(trader, "list_markets", _empty)
    await trader._ensure_markets_fresh()
    await trader._ensure_markets_fresh()
    assert len(lists) == 2   # 故障不被 TTL 掩盖
    assert trader._markets_fetched_at == float("-inf")  # 哨兵不刷新（非 0：防刚启动容器误判）
