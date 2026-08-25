"""R4 统一限频退避/熔断的单元测试（2026-08-25 风险评审）。

覆盖 services/rate_limit.py：
- binance_request 正常响应直通、429 退避后重试成功、重试耗尽返回限频
  响应（不改变调用方错误语义）、418 默认 ban 窗口
- Retry-After 解析（秒 / epoch ms 启发式 / 缺失）
- 熔断降速：连续限频指数退避 + 封顶、成功后逐步回收不立刻清零
- 主动闸：退避窗口内 wait_if_throttled 挂起到窗口结束

不触网络：client.request 用 AsyncMock 序列桩；time.monotonic/asyncio.sleep
用联动假时钟（sleep 推进 monotonic），断言等待时长而非真实睡眠。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import binance_predict.services.rate_limit as rl
from binance_predict.services.rate_limit import (
    BinanceRateGuard,
    _parse_retry_after,
    binance_request,
    rate_guard,
)


@pytest.fixture(autouse=True)
def _reset_guard():
    rate_guard.reset()
    yield
    rate_guard.reset()


@pytest.fixture
def fake_clock(monkeypatch):
    """联动假时钟：monotonic 由字典持有，sleep 推进时间。返回累计 sleep 列表。"""
    state = {"t": 1000.0}
    sleeps: list[float] = []

    monkeypatch.setattr(rl.time, "monotonic", lambda: state["t"])

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["t"] += seconds

    monkeypatch.setattr(rl.asyncio, "sleep", _sleep)
    return sleeps


def _resp(status_code: int, headers: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(status_code=status_code, headers=headers or {})


def _client(*responses) -> SimpleNamespace:
    req = AsyncMock(side_effect=list(responses))
    return SimpleNamespace(request=req, get=req, post=req)


# ============================================================
# binance_request：重试与语义保持
# ============================================================

@pytest.mark.asyncio
async def test_request_success_passthrough(fake_clock) -> None:
    """200 → 直通返回，不等待不重试。"""
    client = _client(_resp(200))
    resp = await binance_request(client, "GET", "https://x/api")
    assert resp.status_code == 200
    assert client.get.await_count == 1
    assert fake_clock == []


@pytest.mark.asyncio
async def test_request_429_respects_retry_after_then_succeeds(fake_clock) -> None:
    """429（Retry-After=10）→ 睡满 10s 后重试成功。"""
    client = _client(
        _resp(429, {"Retry-After": "10"}),
        _resp(200),
    )
    resp = await binance_request(client, "GET", "https://x/api")
    assert resp.status_code == 200
    assert client.get.await_count == 2
    # 等待总时长 ≥ Retry-After（首次 2s 指数基线被 Retry-After 抬升）
    assert sum(fake_clock) >= 10


@pytest.mark.asyncio
async def test_request_retries_exhausted_returns_throttled(fake_clock) -> None:
    """重试耗尽仍限频 → 返回限频响应本身（交调用方 raise_for_status 报错）。"""
    client = _client(
        _resp(429, {"Retry-After": "1"}),
        _resp(429, {"Retry-After": "1"}),
        _resp(429, {"Retry-After": "1"}),
    )
    resp = await binance_request(client, "GET", "https://x/api")
    assert resp.status_code == 429
    assert client.get.await_count == rl.MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_request_418_without_retry_after_uses_ban_default(fake_clock) -> None:
    """418 无 Retry-After → 默认 ban 窗口（≥120s），重试后成功。"""
    client = _client(_resp(418), _resp(200))
    resp = await binance_request(client, "POST", "https://x/api")
    assert resp.status_code == 200
    assert sum(fake_clock) >= 120


@pytest.mark.asyncio
async def test_pre_gate_suspends_request_within_backoff_window(fake_clock) -> None:
    """主动闸：窗口内发起的新请求先挂起到窗口结束才放行。"""
    await rate_guard.record_throttle(429, {"Retry-After": "7"})
    client = _client(_resp(200))
    resp = await binance_request(client, "GET", "https://x/api")
    assert resp.status_code == 200
    assert sum(fake_clock) >= 7  # 先等完退避窗口再发请求


# ============================================================
# 熔断降速状态机
# ============================================================

@pytest.mark.asyncio
async def test_consecutive_throttles_exponential_backoff(fake_clock) -> None:
    """连续限频：退避 2s → 4s → 8s（指数增长）。"""
    g = BinanceRateGuard()
    w1 = await g.record_throttle(429, None)
    w2 = await g.record_throttle(429, None)
    w3 = await g.record_throttle(429, None)
    assert (w1, w2, w3) == (2.0, 4.0, 8.0)


@pytest.mark.asyncio
async def test_backoff_capped(fake_clock) -> None:
    """退避封顶 120s，不无限膨胀。"""
    g = BinanceRateGuard()
    wait = 0.0
    for _ in range(20):
        wait = await g.record_throttle(429, None)
    assert wait == 120.0


@pytest.mark.asyncio
async def test_success_recovers_gradually(fake_clock) -> None:
    """成功后逐步回收连续限频计数（不立刻清零，保持降速惯性）。"""
    g = BinanceRateGuard()
    await g.record_throttle(429, None)
    await g.record_throttle(429, None)  # consecutive=2
    await g.record_success()             # → 1
    w = await g.record_throttle(429, None)  # → 2
    assert w == 4.0  # 若成功清零则此处应为 2.0


@pytest.mark.asyncio
async def test_retry_after_epoch_ms_heuristic(fake_clock) -> None:
    """418 的 Retry-After 可能是 epoch ms：折算为剩余秒数而非睡千年。"""
    import time as _t
    future_ms = (_t.time() + 60) * 1000
    g = BinanceRateGuard()
    w = await g.record_throttle(418, {"Retry-After": str(int(future_ms))})
    assert 55 <= w <= 120  # 60s ban 与 2s 基线取大，容差防时钟抖动


def test_parse_retry_after_variants() -> None:
    assert _parse_retry_after({"Retry-After": "15"}) == 15.0
    assert _parse_retry_after({"retry-after": " 3 "}) == 3.0
    assert _parse_retry_after({}) is None
    assert _parse_retry_after({"Retry-After": "garbage"}) is None


@pytest.mark.asyncio
async def test_status_snapshot(fake_clock) -> None:
    """status() 诊断快照：窗口内 throttled=True，窗口过后恢复。"""
    assert rate_guard.status()["throttled"] is False
    before = rate_guard.status()["total_throttles"]
    await rate_guard.record_throttle(429, {"Retry-After": "5"})
    st = rate_guard.status()
    assert st["throttled"] is True and st["total_throttles"] == before + 1
    assert st["backoff_remaining_s"] > 0
