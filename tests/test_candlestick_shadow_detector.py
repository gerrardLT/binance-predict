"""蜡烛长下影/长上影组合影子检测器：冻结口径、单事件多标签与实盘隔离。"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import binance_predict.services.candlestick_shadow_detector as csd
from binance_predict.services.candlestick_shadow_detector import (
    BAR_MS,
    CANDLESTICK_LOGICAL_SPECS,
    CANDLESTICK_SIGNAL_IDS,
    EVENT_VERSION_BY_TF,
    CandlestickShadowDetector,
    evaluate_candlestick_patterns,
    quote_snapshots,
)


def _rows(tf: str, *, falling: bool = False, hour: int = 4, n: int = 60) -> list[dict]:
    ms = BAR_MS[tf]
    signal_start = int(datetime(2026, 9, 16, hour, tzinfo=timezone.utc).timestamp() * 1000)
    t0 = signal_start // ms * ms - (n - 1) * ms
    rows = []
    for i in range(n):
        base = 200.0 - i if falling else 100.0 + i
        rows.append({
            "open_time": t0 + i * ms,
            "open": base,
            "high": base + 1.2,
            "low": base - 0.2,
            "close": base + (0.8 if not falling else -0.8),
            "volume": 100.0,
        })
    return rows


def _bull_lower_signal(rows: list[dict]) -> None:
    base = float(rows[-2]["close"])
    rows[-1].update(open=base + 0.2, high=base + 1.2, low=base - 9.8, close=base + 1.0, volume=100.0)


def test_main_5m_event_is_one_row_with_all_matching_labels() -> None:
    rows = _rows("5m")
    _bull_lower_signal(rows)

    hits = evaluate_candlestick_patterns(rows, "5m", n_tail=1)

    assert len(hits) == 1
    ids = set(hits[0]["matched_signal_ids"])
    assert ids == {
        "candle_hm_bull_5m_consensus2_shadow_v1",
        "candle_hm_bull_5m_consensus3_shadow_v1",
        "candle_hm_bull_5m_ret5_majority_component_shadow_v1",
        "candle_hm_bull_5m_ret3_component_shadow_v1",
        "candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1",
        "candle_hm_bull_5m_trend4h_up_shadow_hyp_v1",
        "candle_hm_bull_5m_asia_shadow_hyp_v1",
    }
    assert hits[0]["direction"] == "DOWN"
    assert hits[0]["snapshot"]["trend_hits"] == 3
    assert hits[0]["snapshot"]["geometry"]["lower_wick_ratio"] >= 0.5


def test_main_15m_event_adds_quote_band_only_after_real_quote() -> None:
    rows = _rows("15m")
    _bull_lower_signal(rows)
    hit = evaluate_candlestick_patterns(rows, "15m", n_tail=1)[0]
    q_id = "candle_hm_bull_15m_q45_55_shadow_hyp_v1"

    assert q_id not in hit["matched_signal_ids"]
    assert set(hit["matched_signal_ids"]) == {
        "candle_hm_bull_15m_union_shadow_v1",
        "candle_hm_bull_15m_consensus3_shadow_v1",
        "candle_hm_bull_15m_ret3_component_shadow_v1",
        "candle_hm_bull_15m_ma10_component_shadow_v1",
        "candle_hm_bull_15m_trend1h_up_shadow_hyp_v1",
        "candle_hm_bull_15m_asia_shadow_hyp_v1",
        "candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1",
    }

    assert csd.apply_quote_labels("15m", hit["matched_signal_ids"], 0.54)[-1] == q_id
    assert q_id not in csd.apply_quote_labels("15m", hit["matched_signal_ids"], 0.55)


def test_independent_candidate_geometries_are_recorded_as_labels() -> None:
    upper = _rows("5m")
    base = float(upper[-2]["close"])
    upper[-1].update(open=base, high=base + 9.0, low=base - 0.1, close=base + 1.0)
    upper_ids = set(evaluate_candlestick_patterns(upper, "5m", 1)[0]["matched_signal_ids"])
    assert "candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1" in upper_ids

    strict = _rows("5m")
    base = float(strict[-2]["close"])
    strict[-1].update(open=base + 1.0, high=base + 1.0, low=base - 9.0, close=base)
    strict_ids = set(evaluate_candlestick_patterns(strict, "5m", 1)[0]["matched_signal_ids"])
    assert "candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1" in strict_ids
    assert strict[-1]["high"] == strict[-1]["open"]  # 原始 OHLC 严格 0 上影

    counter = _rows("15m", falling=True)
    base = float(counter[-2]["close"])
    counter[-1].update(open=base, high=base + 9.0, low=base - 0.1, close=base + 1.0)
    counter_hit = evaluate_candlestick_patterns(counter, "15m", 1)[0]
    assert counter_hit["direction"] == "UP"
    assert counter_hit["matched_signal_ids"] == [
        "candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1"
    ]


def test_quote_snapshots_use_complete_real_quotes_and_fixed_delays() -> None:
    start = 1_700_000_000_000
    points = [
        SimpleNamespace(timestamp=start + 1_000, up_price=0.80, down_price=0.10),  # sum guard fail
        SimpleNamespace(timestamp=start + 3_000, up_price=0.51, down_price=0.50),
        SimpleNamespace(timestamp=start + 14_000, up_price=0.50, down_price=0.49),
        SimpleNamespace(timestamp=start + 61_000, up_price=0.47, down_price=0.54),
        SimpleNamespace(timestamp=start + 121_000, up_price=0.45, down_price=0.56),
    ]

    out = quote_snapshots(points, start)

    assert out["first"] == {
        "timestamp": start + 3_000, "offset_sec": 3.0,
        "up_price": 0.51, "down_price": 0.50, "quote_sum": 1.01,
    }
    assert out["delays"]["2"] is None
    assert out["delays"]["15"]["timestamp"] == start + 14_000
    assert out["delays"]["60"]["timestamp"] == start + 14_000
    assert out["delays"]["120"]["timestamp"] == start + 61_000


class _FakeResult:
    def __init__(self, *, scalar=None, rows=None) -> None:
        self.scalar = scalar
        self.rows = rows or []

    def scalar_one_or_none(self):
        return self.scalar

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _FakeSession:
    def __init__(self, *, scalar=None, rows=None) -> None:
        self.result = _FakeResult(scalar=scalar, rows=rows)
        self.added = []
        self.committed = False

    async def execute(self, _stmt):
        return self.result

    def add(self, row) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.committed = True


class _SessionCtx:
    def __init__(self, session) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_record_signal_is_idempotent_and_stores_one_physical_event(monkeypatch) -> None:
    rows = _rows("5m")
    _bull_lower_signal(rows)
    hit = evaluate_candlestick_patterns(rows, "5m", 1)[0]
    existing = _FakeSession(scalar=123)
    detector = CandlestickShadowDetector(None, {}, {})

    assert await detector._record_signal(existing, hit, rows[-1]) is False
    assert existing.added == []

    fresh = _FakeSession(scalar=None)
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _version: True)
    assert await detector._record_signal(fresh, hit, rows[-1]) is True
    assert len(fresh.added) == 1
    event = fresh.added[0]
    assert event.version == EVENT_VERSION_BY_TF["5m"]
    assert event.feature_snapshot["matched_signal_ids"] == hit["matched_signal_ids"]
    assert event.feature_snapshot["all_matched_signal_ids"] == hit["matched_signal_ids"]
    assert event.direction == "DOWN"


@pytest.mark.asyncio
async def test_settlement_updates_labels_quotes_and_down_result(monkeypatch) -> None:
    target = 1_700_000_100_000
    q_id = "candle_hm_bull_15m_q45_55_shadow_hyp_v1"
    signal = SimpleNamespace(
        version=EVENT_VERSION_BY_TF["15m"], timeframe="15m",
        signal_bar_start=target - BAR_MS["15m"], target_bar_start=target,
        direction="DOWN", status="PENDING", settle_outcome=None, win=None,
        settle_open=None, settle_close=None, settled_at=None,
        entry_up_price=None, entry_down_price=None, entry_quote_ts=None,
        feature_snapshot={
            "matched_signal_ids": ["candle_hm_bull_15m_union_shadow_v1"],
            "all_matched_signal_ids": ["candle_hm_bull_15m_union_shadow_v1"],
            "quote_snapshots": {},
        },
    )
    session = _FakeSession(rows=[signal])
    monkeypatch.setattr(csd, "async_session_factory", lambda: _SessionCtx(session))
    detector = CandlestickShadowDetector(None, {}, {})
    detector._load_quote_points = AsyncMock(return_value=[
        SimpleNamespace(timestamp=target + 5_000, up_price=0.47, down_price=0.54)
    ])

    await detector._settle_pending("15m", [{
        "open_time": target, "open": 100.0, "high": 101.0,
        "low": 98.0, "close": 99.0, "volume": 1.0,
    }])

    assert signal.status == "SETTLED" and signal.win is True
    assert signal.entry_down_price == 0.54 and signal.entry_quote_ts == target + 5_000
    assert q_id in signal.feature_snapshot["matched_signal_ids"]
    assert session.committed


def test_all_18_logical_versions_are_record_only_and_physically_isolated() -> None:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    from binance_predict.services.multi_live_trader import X4_VERSIONS

    assert len(CANDLESTICK_SIGNAL_IDS) == 18
    assert len(CANDLESTICK_LOGICAL_SPECS) == 18
    assert set(CANDLESTICK_SIGNAL_IDS) <= set(LIVE_CHANNELS)
    assert not (set(CANDLESTICK_SIGNAL_IDS) & set(X4_VERSIONS))
    assert not (set(EVENT_VERSION_BY_TF.values()) & set(LIVE_CHANNELS))
    detector = CandlestickShadowDetector(None, {}, {})
    assert hasattr(detector, "_on_live_fire")
    assert detector._on_live_fire is None


@pytest.mark.asyncio
async def test_cold_start_backscan_never_dispatches_live(monkeypatch) -> None:
    rows = _rows("5m")
    _bull_lower_signal(rows)
    session = _FakeSession(scalar=None)
    detector = CandlestickShadowDetector(None, {}, {})
    detector._on_live_fire = AsyncMock()
    monkeypatch.setattr(csd, "async_session_factory", lambda: _SessionCtx(session))
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _version: True)
    monkeypatch.setattr(csd.time, "time", lambda: (int(rows[-1]["open_time"]) + BAR_MS["5m"] + 30_000) / 1000)

    await detector._evaluate_new_bars("5m", rows)

    detector._on_live_fire.assert_not_called()
    assert len(session.added) == 1


@pytest.mark.asyncio
async def test_live_payloads_are_dispatched_once_per_logical_label(monkeypatch) -> None:
    rows = _rows("5m")
    _bull_lower_signal(rows)
    hit = evaluate_candlestick_patterns(rows, "5m", 1)[0]
    fresh = _FakeSession(scalar=None)
    detector = CandlestickShadowDetector(None, {}, {})
    fired: list[dict] = []
    detector._on_live_fire = fired.append
    monkeypatch.setattr(csd.shadow_gate, "is_enabled", lambda _version: True)
    monkeypatch.setattr(csd.time, "time", lambda: (int(rows[-1]["open_time"]) + BAR_MS["5m"] + 30_000) / 1000)

    payloads: list[dict] = []
    assert await detector._record_signal(fresh, hit, rows[-1], payloads) is True
    detector._dispatch_live(payloads)
    assert {payload["version"] for payload in fired} == set(hit["matched_signal_ids"])
    assert {payload["market_start"] for payload in fired} == {
        int(rows[-1]["open_time"]) + BAR_MS["5m"]
    }
    assert all(payload["direction"] == "DOWN" for payload in fired)
