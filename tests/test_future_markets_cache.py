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
