"""btc_regime BtcRegimeFeed 单元测试：ret24 口径钉死（严格 ex-ante，closes[i−1]）。

不触网络：直接注入内存 K 线序列，验证与 Part 3 回测 ex-ante 口径逐位一致、
样本不足保守 None、刷新失败沿用旧缓存（影子/实盘同源，不允许口径漂移）。
"""
from __future__ import annotations

import time

import pytest

from binance_predict.services.btc_regime import (
    RET24_BARS,
    STALE_MAX_S,
    BtcRegimeFeed,
)


def _feed_with_bars(n: int) -> BtcRegimeFeed:
    """n 根 5m K 线（open 每 5min 递增，close 逐根 +1），缓存视为新鲜（免网络）。"""
    f = BtcRegimeFeed()
    f._opens = [i * 300_000 for i in range(n)]
    f._closes = [100.0 + float(i) for i in range(n)]
    f._fetched_at = time.monotonic()
    return f


@pytest.mark.asyncio
async def test_ret24_exante_uses_prev_closed_bar() -> None:
    """ts 落在第 300 根内 → i=300，ret = closes[299]/closes[299−288] − 1。

    closes[i]（触发时点所在 K，未收盘）绝不参与——ex-ante 口径钉死。
    """
    f = _feed_with_bars(400)
    ts = 300 * 300_000 + 50_000          # 第 300 根内（open + 50s）
    expect = f._closes[299] / f._closes[299 - RET24_BARS] - 1.0
    assert await f.ret24_at(ts) == pytest.approx(expect)
    # 与「错用当根 close」的 orig 口径必须不同（防回归回未来函数版本）
    orig_wrong = f._closes[300] / f._closes[300 - RET24_BARS] - 1.0
    assert expect != pytest.approx(orig_wrong)


@pytest.mark.asyncio
async def test_ret24_insufficient_samples_none() -> None:
    """样本不足（覆盖不满触发点前 24h）→ None，调用方保守拒绝。"""
    f = _feed_with_bars(RET24_BARS)      # 288 根：i 最大 287 < 289
    assert await f.ret24_at(287 * 300_000 + 50_000) is None
    f2 = _feed_with_bars(RET24_BARS + 1)  # 289 根：i 最大 288，仍差一根
    assert await f2.ret24_at(288 * 300_000 + 50_000) is None
    f3 = _feed_with_bars(RET24_BARS + 2)  # 290 根：i=289 → i−1−288=0 恰可用
    ts = 289 * 300_000 + 50_000
    assert await f3.ret24_at(ts) == pytest.approx(
        f3._closes[288] / f3._closes[0] - 1.0)


@pytest.mark.asyncio
async def test_ret24_empty_feed_none(monkeypatch) -> None:
    """从未成功拉取（空缓存）→ None 且不抛（mock 刷新，不触网）。"""

    async def _no_refresh(self):
        self._last_error = "mocked-offline"

    monkeypatch.setattr(BtcRegimeFeed, "_refresh", _no_refresh)
    f = BtcRegimeFeed()
    assert await f.ret24_at(1_000_000_000) is None
    assert f.status()["stale"] is True


@pytest.mark.asyncio
async def test_refresh_failure_keeps_old_cache(monkeypatch) -> None:
    """TTL 过期触发刷新且刷新失败 → 陈旧度未超限前沿用旧缓存。"""
    f = _feed_with_bars(300)
    f._fetched_at = time.monotonic() - 100.0   # >TTL(60s) 触发刷新，<STALE(600s) 仍可用

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    monkeypatch.setattr(
        "binance_predict.services.btc_regime.httpx.AsyncClient", _Boom)
    ts = 289 * 300_000 + 50_000
    expect = f._closes[288] / f._closes[0] - 1.0
    assert await f.ret24_at(ts) == pytest.approx(expect)   # 旧缓存仍可用
    assert f.status()["last_error"]                        # 失败可观测


@pytest.mark.asyncio
async def test_stale_beyond_limit_returns_none(monkeypatch) -> None:
    """刷新连续失败超过陈旧度上限 → None（陈旧数据不等于保守数据，宁可拒绝）。"""

    async def _no_refresh(self):
        self._last_error = "mocked-offline"

    monkeypatch.setattr(BtcRegimeFeed, "_refresh", _no_refresh)
    f = _feed_with_bars(300)
    f._fetched_at = time.monotonic() - (STALE_MAX_S + 1)   # 强制超限
    assert await f.ret24_at(289 * 300_000 + 50_000) is None
    assert f.status()["stale"] is True
