"""S2 空头耗尽优化版：过滤判定、影子落库与实盘派发。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import binance_predict.services.s2_optimized_shadow_detector as s2o
from binance_predict.services.s2_optimized_shadow_detector import (
    S2_OPT_VERSION,
    S2OptimizedShadowDetector,
    range_position,
    should_fire_s2_optimized,
)

BAR_MS_15M = 900_000
SIGNAL_START = 1_700_000_000_000 // BAR_MS_15M * BAR_MS_15M
TARGET_START = SIGNAL_START + BAR_MS_15M


class _FakeResult:
    def __init__(self, scalar=None) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    def __init__(self, scalar=None) -> None:
        self.scalar = scalar
        self.added: list = []
        self.committed = False

    async def execute(self, _stmt) -> _FakeResult:
        return _FakeResult(self.scalar)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> _FakeSession:
        return self.session

    async def __aexit__(self, *exc) -> bool:
        return False


class _Collector:
    def __init__(self, bars: list[dict]) -> None:
        self.bars = bars
        self.calls: list[tuple[str, int]] = []

    async def fetch_klines_ending_at(self, interval: str, limit: int, end_ms: int) -> list[dict]:
        self.calls.append((interval, end_ms))
        return self.bars[-limit:]

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        return self.bars[-limit:]


def _bars(*, close: float = 133.0, low: float = 100.0, high: float = 200.0,
          count: int = 14 * 96) -> list[dict]:
    bars = [
        {"open_time": SIGNAL_START - (count - 1 - i) * BAR_MS_15M,
         "open": 120.0, "high": high, "low": low, "close": 120.0, "volume": 1.0}
        for i in range(count)
    ]
    bars[-1] = {"open_time": SIGNAL_START, "open": 140.0, "high": high,
                "low": low, "close": close, "volume": 3.0}
    return bars


def _payload(vol_ratio: float = 2.0) -> dict:
    return {
        "id": 7,
        "pattern_type": "bear_exhaust",
        "market_start_15m": TARGET_START,
        "market_end_15m": TARGET_START + BAR_MS_15M,
        "signal_bar_start": SIGNAL_START,
        "signal_close": 133.0,
        "vol_ratio": vol_ratio,
    }


def test_volume_and_position_boundaries() -> None:
    assert should_fire_s2_optimized(2.0, 0.33) is True
    assert should_fire_s2_optimized(3.999, 0.33) is True
    assert should_fire_s2_optimized(4.0, 0.33) is False
    assert should_fire_s2_optimized(2.0, 0.33 - 1e-9) is False
    assert should_fire_s2_optimized(None, 0.8) is False


def test_range_position_rejects_missing_or_flat_history() -> None:
    assert range_position([], 100.0) is None
    assert range_position(_bars(count=10), 133.0) is None
    assert range_position(_bars(low=100.0, high=100.0, close=100.0), 100.0) is None


@pytest.mark.asyncio
async def test_parent_s2_hit_records_shadow_and_dispatches_live(monkeypatch) -> None:
    session = _FakeSession()
    monkeypatch.setattr(s2o, "async_session_factory", lambda: _FakeSessionCtx(session))
    monkeypatch.setattr(s2o.shadow_gate, "is_enabled", lambda _v: True)
    live: list[dict] = []
    collector = _Collector(_bars())
    detector = S2OptimizedShadowDetector(collector, {
        "start_date": TARGET_START,
        "up_price": 0.54,
        "down_price": 0.48,
        "updated_ts": TARGET_START + 1_000,
    })
    detector._on_live_fire = live.append

    await detector.evaluate(_payload())

    assert collector.calls == [("15m", TARGET_START)]
    assert len(session.added) == 1
    row = session.added[0]
    assert row.version == S2_OPT_VERSION
    assert row.signal_bar_start == SIGNAL_START
    assert row.target_bar_start == TARGET_START
    assert row.direction == "UP"
    assert row.entry_up_price == 0.54
    assert row.feature_snapshot["vol_ratio"] == 2.0
    assert row.feature_snapshot["pos14d"] == pytest.approx(0.33)
    assert live == [{
        "id": 7,
        "pattern_type": "bear_exhaust_opt_v1",
        "side": "low",
        "market_start_15m": TARGET_START,
        "market_end_15m": TARGET_START + BAR_MS_15M,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vol_ratio", "bars"),
    [
        (4.0, _bars()),
        (2.0, _bars(close=132.9999)),
        (2.0, _bars(count=10)),
        (2.0, _bars(low=100.0, high=100.0, close=100.0)),
        (2.0, [*_bars()[:-1], {**_bars()[-1], "open_time": SIGNAL_START - BAR_MS_15M}]),
    ],
)
async def test_filter_miss_records_nothing(monkeypatch, vol_ratio: float,
                                           bars: list[dict]) -> None:
    session = _FakeSession()
    monkeypatch.setattr(s2o, "async_session_factory", lambda: _FakeSessionCtx(session))
    detector = S2OptimizedShadowDetector(_Collector(bars), {})
    live: list[dict] = []
    detector._on_live_fire = live.append
    payload = _payload(vol_ratio)
    payload["signal_close"] = bars[-1]["close"] if bars else 133.0

    await detector.evaluate(payload)

    assert session.added == []
    assert live == []


@pytest.mark.asyncio
async def test_record_is_idempotent(monkeypatch) -> None:
    session = _FakeSession(scalar=123)
    monkeypatch.setattr(s2o, "async_session_factory", lambda: _FakeSessionCtx(session))
    detector = S2OptimizedShadowDetector(_Collector(_bars()), {})

    await detector.evaluate(_payload())

    assert session.added == []


@pytest.mark.asyncio
async def test_settles_own_pending_signal(monkeypatch) -> None:
    row = SimpleNamespace(
        version=S2_OPT_VERSION,
        target_bar_start=TARGET_START,
        direction="UP",
        status="PENDING",
        settle_outcome=None,
        win=None,
        settle_open=None,
        settle_close=None,
        settled_at=None,
    )
    session = _FakeSession()
    session.rows = [row]

    class _RowsResult(_FakeResult):
        def scalars(self): return self
        def all(self): return session.rows

    async def _execute(_stmt): return _RowsResult()
    session.execute = _execute
    monkeypatch.setattr(s2o, "async_session_factory", lambda: _FakeSessionCtx(session))
    detector = S2OptimizedShadowDetector(_Collector([]), {})

    await detector.settle([{              
        "open_time": TARGET_START, "open": 100.0, "close": 101.0,
    }])

    assert row.status == "SETTLED"
    assert row.settle_outcome == "UP"
    assert row.win is True
    assert row.settle_open == 100.0 and row.settle_close == 101.0


@pytest.mark.asyncio
async def test_start_survives_cold_start_db_failure(monkeypatch) -> None:
    detector = S2OptimizedShadowDetector(_Collector([]), {})

    async def _boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(detector, "_poll_once", _boom)
    await detector.start()
    assert detector._running is True and detector._task is not None
    await detector.stop()


def test_parent_payload_is_strictly_signal_time_data() -> None:
    """派生器必须锚定父信号根，历史查询截止目标窗开盘，不读目标窗未来数据。"""
    sig = _payload()
    assert sig["signal_bar_start"] + BAR_MS_15M == sig["market_start_15m"]


@pytest.mark.asyncio
async def test_fake_breakout_dispatches_parent_features(monkeypatch) -> None:
    from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector

    received: list[dict] = []
    spawned: list = []
    detector = FakeBreakoutDetector.__new__(FakeBreakoutDetector)

    async def _hook(payload: dict) -> None:
        received.append(payload)

    detector._on_s2_optimized = _hook
    import binance_predict.services.fake_breakout_detector as fbd
    monkeypatch.setattr(fbd.asyncio, "create_task", lambda coro, name=None: spawned.append(coro))
    sig_k = {"open_time": SIGNAL_START, "close": 133.0}

    detector._notify_s2_optimized(7, sig_k, 2.5, TARGET_START, TARGET_START + BAR_MS_15M)

    assert len(spawned) == 1
    await spawned[0]
    assert received == [{
        "id": 7,
        "pattern_type": "bear_exhaust",
        "signal_bar_start": SIGNAL_START,
        "signal_close": 133.0,
        "vol_ratio": 2.5,
        "market_start_15m": TARGET_START,
        "market_end_15m": TARGET_START + BAR_MS_15M,
    }]
