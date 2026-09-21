from __future__ import annotations

import pytest

import binance_predict.services.data_collector as dc


class _Response:
    def __init__(self, payload: list[list]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> list[list]:
        return self.payload


class _Client:
    def __init__(self, payload: list[list] | list[list[list]], holder: dict) -> None:
        self.payload = payload
        self.holder = holder
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, _url, params=None):
        self.holder.setdefault("calls", []).append(params)
        if self.payload and isinstance(self.payload[0][0], list):
            payload = self.payload[self.index]
            self.index += 1
        else:
            payload = self.payload
        return _Response(payload)


def _row(start: int, close: float) -> list:
    return [start, "100", "110", "90", str(close), "5", start + 899_999]


@pytest.mark.asyncio
async def test_fetch_klines_ending_at_excludes_target_window(monkeypatch) -> None:
    target_start = 1_700_000_100_000
    previous_start = target_start - 900_000
    holder: dict = {}
    payload = [_row(previous_start, 101), _row(target_start, 102)]
    monkeypatch.setattr(
        dc.httpx,
        "AsyncClient",
        lambda **_kwargs: _Client(payload, holder),
    )

    bars = await dc.BinanceDataCollector().fetch_klines_ending_at(
        "15m", 2, target_start,
    )

    assert [bar["open_time"] for bar in bars] == [previous_start]
    assert holder["calls"][0]["endTime"] == target_start - 1
    assert holder["calls"][0]["limit"] == 2


@pytest.mark.asyncio
async def test_fetch_klines_ending_at_pages_past_binance_limit(monkeypatch) -> None:
    end = 1_700_100_000_000
    older = [_row(end - (1100 - i) * 900_000, 100 + i) for i in range(100)]
    newer = [_row(end - (1000 - i) * 900_000, 200 + i) for i in range(1000)]
    holder: dict = {}
    monkeypatch.setattr(
        dc.httpx,
        "AsyncClient",
        lambda **_kwargs: _Client([newer, older], holder),
    )

    bars = await dc.BinanceDataCollector().fetch_klines_ending_at("15m", 1100, end)

    assert len(bars) == 1100
    assert bars[0]["open_time"] == older[0][0]
    assert bars[-1]["open_time"] == newer[-1][0]
    assert [call["limit"] for call in holder["calls"]] == [1000, 100]
    assert holder["calls"][1]["endTime"] == newer[0][0] - 1
