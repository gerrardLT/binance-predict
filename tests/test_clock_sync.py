"""clock_sync 加固测试（2026-10-25 -1021 全天拒收事故）。

覆盖：
- 签名时间戳安全余量只作用于 signed_timestamp_ms，不影响 now_ms
  （影子检测器用 now_ms 做窗口对齐，后移会让整个影子舰队统一晚触发）
- 多样本校准取最小 RTT 样本（NTP 风格）
- 60s 节流短路（不发起网络）
- 校准失败沿用上次偏差
- -1021 响应触发强制重校准

不触网络：httpx.AsyncClient / time.time 全部替身。
"""

from __future__ import annotations

import asyncio

import pytest

from binance_predict.services import clock_sync


@pytest.fixture(autouse=True)
def _fresh_clock_state():
    """每个用例重置模块级状态（含跨事件循环的 asyncio.Lock）。"""
    clock_sync._sync_lock = asyncio.Lock()
    clock_sync._time_offset_ms = 0
    clock_sync._last_sync_ts = 0.0
    yield
    clock_sync._sync_lock = asyncio.Lock()
    clock_sync._time_offset_ms = 0
    clock_sync._last_sync_ts = 0.0


class _FakeResp:
    def __init__(self, server_time: int):
        self._server_time = server_time

    def json(self):
        return {"serverTime": self._server_time}


class _FakeTimeClient:
    """按序返回预设 serverTime 的假 /api/v3/time 客户端。"""

    def __init__(self, server_times: list[int]):
        self._times = list(server_times)
        self.calls = 0

    async def get(self, _url):
        self.calls += 1
        return _FakeResp(self._times.pop(0))


class _ClientCtx:
    def __init__(self, client):
        self._client = client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *exc):
        return False


class _BrokenClient:
    async def get(self, _url):
        raise clock_sync.httpx.ConnectError("boom")


def test_signed_timestamp_margin_not_in_now_ms(monkeypatch):
    monkeypatch.setattr(clock_sync.time, "time", lambda: 2000.0)
    clock_sync._time_offset_ms = 0
    base = 2_000_000
    assert clock_sync.now_ms() == base
    assert (
        clock_sync.signed_timestamp_ms()
        == base - clock_sync._AHEAD_SAFETY_MARGIN_MS
    )
    # 余量必须覆盖币安「超前 >1000ms」拒收阈值
    assert clock_sync._AHEAD_SAFETY_MARGIN_MS >= 1000


def test_signed_timestamp_includes_offset(monkeypatch):
    monkeypatch.setattr(clock_sync.time, "time", lambda: 2000.0)
    clock_sync._time_offset_ms = 300
    base = 2_000_300
    assert clock_sync.now_ms() == base
    assert (
        clock_sync.signed_timestamp_ms()
        == base - clock_sync._AHEAD_SAFETY_MARGIN_MS
    )


@pytest.mark.asyncio
async def test_sync_picks_min_rtt_sample(monkeypatch):
    # 墙钟序列：sync 内 now=0.0；样本1 before/after=1000.0→1000.5（RTT 500ms）；
    # 样本2 before/after=1000.0→1000.1（RTT 100ms）
    wall = iter([0.0, 1000.0, 1000.5, 1000.0, 1000.1])
    monkeypatch.setattr(clock_sync.time, "time", lambda: next(wall))
    # 样本1 server=1000600 → offset +350；样本2 server=1000100 → offset +50
    fake = _FakeTimeClient([1000600, 1000100])
    monkeypatch.setattr(
        clock_sync.httpx, "AsyncClient", lambda **_kw: _ClientCtx(fake)
    )

    offset = await clock_sync.sync_server_time(force=True, samples=2)

    assert fake.calls == 2
    assert offset == 50  # RTT 100ms 样本胜出，RTT 500ms 的 +350 被丢弃
    assert clock_sync._time_offset_ms == 50


@pytest.mark.asyncio
async def test_sync_throttled_within_interval(monkeypatch):
    clock_sync._time_offset_ms = 321
    clock_sync._last_sync_ts = 1_000_000.0
    monkeypatch.setattr(clock_sync.time, "time", lambda: 1_000_030.0)  # 30s 后 < 60s

    def _boom(**_kw):
        raise AssertionError("60s 内不应重新发起网络校准")

    monkeypatch.setattr(clock_sync.httpx, "AsyncClient", _boom)
    assert await clock_sync.sync_server_time() == 321


@pytest.mark.asyncio
async def test_sync_failure_keeps_last_offset(monkeypatch):
    clock_sync._time_offset_ms = 777
    monkeypatch.setattr(
        clock_sync.httpx, "AsyncClient", lambda **_kw: _ClientCtx(_BrokenClient())
    )
    offset = await clock_sync.sync_server_time(force=True)
    assert offset == 777
    assert clock_sync._time_offset_ms == 777


@pytest.mark.asyncio
async def test_heal_triggers_force_resync_only_on_1021(monkeypatch):
    forced = []

    async def _fake_sync(*, force: bool = False, **_kw):
        forced.append(force)
        return 0

    monkeypatch.setattr(clock_sync, "sync_server_time", _fake_sync)

    assert await clock_sync.heal_after_timestamp_ahead_reject(
        '{"code":-1021,"msg":"Timestamp for this request was 1000ms ahead"}'
    )
    assert not await clock_sync.heal_after_timestamp_ahead_reject(
        '{"code":-1003,"msg":"x"}'
    )
    assert forced == [True]  # 仅 -1021 触发，且为强制重校准
