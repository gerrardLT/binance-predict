"""btc_regime.ret_at 锚点口径单测（x4_v3 C3 趋势门禁的回测对齐底线）。

不触网络：直接填充进程级缓存（_opens/_closes/_fetched_at，TTL 内 _ensure_fresh
不刷新）；冷缓存/陈旧缓存用例把 _refresh 替换为 no-op（防 _ensure_fresh 触发
真 REST 请求）。

锚点语义（与 .pytest_tmp/x4v2_analysis/report5 公式逐位对齐）：
    i = bisect_right(opens, ts) − 1；ts = 触发窗 start_time → i = 触发窗所在 K
    ret_at(ts, bars) = closes[i−1] / closes[i−1−bars] − 1
    closes[i−1] = 触发窗开盘价（前一根 K 在触发时点已收盘，严格 ex-ante）。
缓存构造：opens = [k·300_000]，closes = [100.0 + k]（线性，期望值可手算）。
"""

from __future__ import annotations

import time

import pytest

from binance_predict.services import btc_regime as br

INTERVAL = 300_000  # 5m K 开盘间隔（ms）


def _make_feed(n: int) -> br.BtcRegimeFeed:
    """n 根线性缓存：opens=[0,300k,600k,…]，closes=[100.0,101.0,…]，TTL 内新鲜。"""
    feed = br.BtcRegimeFeed()
    feed._opens = [k * INTERVAL for k in range(n)]
    feed._closes = [100.0 + k for k in range(n)]
    feed._fetched_at = time.monotonic()   # _ensure_fresh TTL 内直接读，不触发刷新
    return feed


async def _noop_refresh() -> None:
    return None


@pytest.mark.asyncio
async def test_ret_at_4h_anchor_formula() -> None:
    """bars=48（x4_v3 4h 门禁）：closes[i−1]/closes[i−49]−1。

    closes[i]（触发时点所在 K，未收盘）绝不参与——ex-ante 口径钉死：
    与「错用当根 close」的 orig 口径必须不同（防回归回未来函数版本）。
    """
    feed = _make_feed(340)
    got = await feed.ret_at(300 * INTERVAL, 48)
    assert got == pytest.approx((100.0 + 299) / (100.0 + 251) - 1.0)
    orig_wrong = (100.0 + 300) / (100.0 + 252) - 1.0
    assert got != pytest.approx(orig_wrong)


@pytest.mark.asyncio
async def test_ret_at_24h_anchor_formula() -> None:
    """bars=288（v4 24h 门禁共用实现）：closes[i−1]/closes[i−289]−1。"""
    feed = _make_feed(340)
    got = await feed.ret_at(320 * INTERVAL, 288)
    assert got == pytest.approx((100.0 + 319) / (100.0 + 31) - 1.0)


@pytest.mark.asyncio
async def test_ret24_at_is_thin_wrapper() -> None:
    """ret24_at(ts) ≡ ret_at(ts, 288)（既有调用方零改动的等价保证）。"""
    feed = _make_feed(340)
    ts = 320 * INTERVAL
    assert await feed.ret24_at(ts) == pytest.approx(await feed.ret_at(ts, 288))


@pytest.mark.asyncio
async def test_ret_at_insufficient_samples_returns_none() -> None:
    """样本不足（i < bars+1）→ None：bars=48 需 i≥49，恰 49 根边界有效。"""
    feed = _make_feed(100)
    assert await feed.ret_at(48 * INTERVAL, 48) is None     # i=48 < 49
    got = await feed.ret_at(49 * INTERVAL, 48)              # i=49 ≥ 49
    assert got == pytest.approx((100.0 + 48) / 100.0 - 1.0)


@pytest.mark.asyncio
async def test_ret24_at_needs_289_bars() -> None:
    """bars=288 需 i≥289：289 根恰有效、差一根 None。"""
    feed = _make_feed(340)
    assert await feed.ret24_at(200 * INTERVAL) is None      # i=200 < 289
    got = await feed.ret24_at(289 * INTERVAL)               # i=289 ≥ 289
    assert got == pytest.approx((100.0 + 288) / 100.0 - 1.0)


@pytest.mark.asyncio
async def test_ret_at_bisect_within_window() -> None:
    """ts 落在窗内（start+120s）→ bisect 归到本窗 i，与窗起点取值一致。"""
    feed = _make_feed(340)
    a = await feed.ret_at(300 * INTERVAL, 48)
    b = await feed.ret_at(300 * INTERVAL + 120_000, 48)
    assert a == pytest.approx(b)


@pytest.mark.asyncio
async def test_ret_at_cold_cache_returns_none(monkeypatch) -> None:
    """冷启动无缓存（从未成功拉取）→ None。"""
    feed = br.BtcRegimeFeed()          # _opens=[]、_fetched_at=0
    monkeypatch.setattr(feed, "_refresh", _noop_refresh)
    assert await feed.ret_at(300 * INTERVAL, 48) is None


@pytest.mark.asyncio
async def test_ret_at_stale_cache_returns_none(monkeypatch) -> None:
    """缓存陈旧（> 600s 上限）→ 保守 None（陈旧数据不等于保守数据）。"""
    feed = _make_feed(340)
    feed._fetched_at = time.monotonic() - 601.0
    monkeypatch.setattr(feed, "_refresh", _noop_refresh)   # 防 _ensure_fresh 触网
    assert await feed.ret_at(300 * INTERVAL, 48) is None
    assert await feed.ret24_at(320 * INTERVAL) is None


@pytest.mark.asyncio
async def test_ret_at_refresh_failure_keeps_old_cache(monkeypatch) -> None:
    """TTL 过期触发刷新且刷新失败 → 陈旧度未超限前沿用旧缓存（影子/实盘同源）。"""
    feed = _make_feed(340)
    feed._fetched_at = time.monotonic() - 100.0   # >TTL(60s) 触发刷新，<STALE(600s) 仍可用

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr(
        "binance_predict.services.btc_regime.httpx.AsyncClient", _Boom)
    ts = 320 * INTERVAL
    expect = feed._closes[319] / feed._closes[31] - 1.0
    assert await feed.ret24_at(ts) == pytest.approx(expect)   # 旧缓存仍可用
    assert feed.status()["last_error"]                        # 失败可观测
