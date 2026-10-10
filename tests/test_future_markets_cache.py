"""未来窗缓存：空结果短 TTL（防弹窗被 45s 长缓存锁死在「未创建」）。"""
import pytest

from binance_predict.services.prediction_trading import BinancePredictionTrader


@pytest.mark.asyncio
async def test_empty_scan_uses_short_ttl(monkeypatch):
    t = BinancePredictionTrader()
    calls = {"n": 0}

    async def fake_scan(period, count=8):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [{"window_start": 111, "window_end": 222}]

    monkeypatch.setattr(t, "scan_future_markets", fake_scan)

    # 首次扫描返回空 → 空结果按短 TTL 缓存（不是 45s 长缓存）
    windows, age = await t.get_future_markets_cached("5m", count=8)
    assert windows == [] and age == 0.0
    assert t._future_cache["5m"][2] == t._future_empty_ttl_ms

    # 缓存未过期：直接命中，不重扫
    await t.get_future_markets_cached("5m", count=8)
    assert calls["n"] == 1

    # 短 TTL 到期（把缓存时间戳回拨到 TTL 之外）→ 重扫拿到非空，改按长 TTL 缓存
    ts, w, ttl = t._future_cache["5m"]
    t._future_cache["5m"] = (ts - ttl - 1, w, ttl)
    windows, age = await t.get_future_markets_cached("5m", count=8)
    assert calls["n"] == 2
    assert windows == [{"window_start": 111, "window_end": 222}]
    assert t._future_cache["5m"][2] == t._future_cache_ttl_ms


@pytest.mark.asyncio
async def test_future_windows_stable_across_scans(monkeypatch):
    """记忆池：首次扫描到的窗口，即使后续扫描返回空也要保留（列表不再随扫描漂移）。"""
    import time

    t = BinancePredictionTrader()
    start = int(time.time() * 1000) + 300_000
    row = {"window_start": start, "window_end": start + 300_000, "ahead_sec": 300.0,
           "trading_status": "OPEN", "up_price": 0.5, "down_price": 0.5,
           "up_token": "u", "down_token": "d", "topic_id": 1, "available": True}
    calls = {"n": 0}

    async def fake_list_markets():
        t._future_markets = {}
        return []

    async def fake_scan(period, count, budget, now_ms):
        calls["n"] += 1
        return [dict(row)] if calls["n"] == 1 else []

    monkeypatch.setattr(t, "list_markets", fake_list_markets)
    monkeypatch.setattr(t, "_scan_future_via_detail", fake_scan)

    first = await t.scan_future_markets("5m", count=8)
    assert [r["window_start"] for r in first] == [start]

    second = await t.scan_future_markets("5m", count=8)   # 本轮扫描为空
    assert [r["window_start"] for r in second] == [start]
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_force_bypasses_cache(monkeypatch):
    """手动刷新（force=True）绕过后端 TTL 缓存强制重扫。"""
    t = BinancePredictionTrader()
    calls = {"n": 0}

    async def fake_scan(period, count=8):
        calls["n"] += 1
        return [{"window_start": calls["n"]}]

    monkeypatch.setattr(t, "scan_future_markets", fake_scan)

    await t.get_future_markets_cached("5m", count=8)
    assert calls["n"] == 1
    await t.get_future_markets_cached("5m", count=8)          # 命中缓存，不重扫
    assert calls["n"] == 1

    windows, age = await t.get_future_markets_cached("5m", count=8, force=True)
    assert calls["n"] == 2 and age == 0.0
    assert windows == [{"window_start": 2}]
