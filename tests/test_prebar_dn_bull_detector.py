from __future__ import annotations

import pytest

from binance_predict.services.live_channels import LIVE_CHANNELS, parse_channel_config
from binance_predict.services.prebar_dn_bull_detector import VERSION, PrebarDnBullDetector
from binance_predict.services.shadow_version_gate import ShadowVersionGate


def test_prebar_candidate_defaults_off(monkeypatch):
    from binance_predict.config.settings import settings
    monkeypatch.setattr(settings, "live_channels_json", "")
    assert LIVE_CHANNELS[VERSION].default_enabled is False
    cfg = parse_channel_config()[VERSION]
    assert cfg.enabled is False
    assert cfg.amount_usdt == 10.0
    assert cfg.max_exec_price is None  # resolve_max_exec 回落冻结 spec 0.32
    assert ShadowVersionGate().is_enabled(VERSION) is False


@pytest.mark.asyncio
async def test_prebar_detector_uses_only_first_300_seconds(monkeypatch):
    detector = PrebarDnBullDetector(object())
    fired = []
    detector._on_live_fire = fired.append
    monkeypatch.setattr(
        "binance_predict.services.prebar_dn_bull_detector.shadow_gate.is_enabled",
        lambda version: True,
    )
    monkeypatch.setattr(detector, "_score", lambda window_start: _async_value(0.60))

    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def execute(self, stmt):
            class Result:
                def scalar_one_or_none(self): return None
            return Result()
        def add(self, row): self.row = row
        async def commit(self): pass

    monkeypatch.setattr(
        "binance_predict.services.prebar_dn_bull_detector.async_session_factory",
        lambda: Session(),
    )
    ws = 1_800_000_000_000
    await detector.observe_sample(ws, ws + 900_000, ws + 300_000, 0.20, 0.80)
    await detector.observe_sample(ws, ws + 900_000, ws + 100_000, 0.0, 1.0)
    assert fired == []  # 300s贴线过期；0/1终局哨兵也永不成交
    await detector.observe_sample(ws, ws + 900_000, ws + 299_999, 0.30, 0.70)
    assert fired[0]["version"] == VERSION
    assert fired[0]["market_start"] == ws
    assert fired[0]["direction"] == "UP"


async def _async_value(value):
    return value
