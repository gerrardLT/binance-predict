"""逐信号通知配置：默认全开、通道隔离、事件/字段闸、API 与发送挂钩。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from binance_predict.config.settings import settings
from binance_predict.main import app
from binance_predict.services import notification_config as nc
from binance_predict.services.notification_config import (
    GLOBAL_CHANNEL,
    NotificationConfigService,
)
from binance_predict.services.wechat_notifier import WeChatNotifier
from binance_predict.services import signal_notify as sn
from binance_predict.services import fake_breakout_detector as fbd_mod
from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector

PASSWORD = "s3cret-login"
CHANNEL = "quote_contrarian_v2"
OTHER = "x4_v2"


class _FakeResult:
    def __init__(self, rows=None, scalar=None) -> None:
        self._rows, self._scalar = rows or [], scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    def __init__(self, rows=None, scalar=None) -> None:
        self.rows, self.scalar = rows or [], scalar
        self.added: list = []
        self.committed = False
        self.deleted = False

    async def execute(self, _stmt):
        return _FakeResult(self.rows, self.scalar)

    def add(self, obj) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.committed = True


class _FakeSessionCtx:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.fixture
def svc() -> NotificationConfigService:
    return NotificationConfigService()


def test_default_all_on(svc) -> None:
    assert svc.channel_enabled(CHANNEL) is True
    assert svc.is_event_enabled(CHANNEL, "radar", "wechat") is True
    assert svc.is_event_enabled(CHANNEL, "settled", "email") is True
    assert svc.is_field_enabled(CHANNEL, "filled", "avg_price") is True
    assert svc.is_low_balance_enabled() is True
    snap = svc.snapshot()
    assert "wxpusher_spt" not in str(snap)
    assert "smtp" not in str(snap.get("physical", {})).lower() or "smtp_configured" in str(snap)
    assert "password" not in str(snap).lower()
    assert snap["physical"]["wechat"]["wxpusher_configured"] in (True, False)


def test_unknown_channel_keeps_sending(svc) -> None:
    assert svc.is_event_enabled("manual_test", "filled", "wechat") is True
    assert svc.is_event_enabled("quote_momentum_v1", "radar", "wechat") is True


def test_channel_isolation(svc) -> None:
    svc._store_row(CHANNEL, False, nc.default_channel_config())
    assert svc.is_event_enabled(CHANNEL, "filled", "wechat") is False
    assert svc.is_event_enabled(OTHER, "filled", "wechat") is True


def test_event_and_field_isolation(svc) -> None:
    cfg = nc.default_channel_config()
    cfg["events"]["radar"]["wechat"] = False
    cfg["events"]["filled"]["fields"]["shares"] = False
    svc._store_row(CHANNEL, True, cfg)
    assert svc.is_event_enabled(CHANNEL, "radar", "wechat") is False
    assert svc.is_event_enabled(CHANNEL, "filled", "wechat") is True
    assert svc.is_field_enabled(CHANNEL, "filled", "shares") is False
    assert svc.is_field_enabled(CHANNEL, "filled", "avg_price") is True


def test_global_email_off_blocks_settled_email(svc) -> None:
    g = nc.default_global_config()
    g["email_enabled"] = False
    svc._store_row(GLOBAL_CHANNEL, True, g)
    assert svc.is_event_enabled(CHANNEL, "settled", "email") is False
    assert svc.is_event_enabled(CHANNEL, "filled", "wechat") is True


def test_normalize_rejects_unknown_keys() -> None:
    out = nc.normalize_channel_config({
        "events": {"filled": {"wechat": False, "bogus": True, "fields": {"avg_price": False, "nope": False}}},
        "secret": "x",
    })
    assert "secret" not in out
    assert "bogus" not in out["events"]["filled"]
    assert "nope" not in out["events"]["filled"]["fields"]
    assert out["events"]["filled"]["wechat"] is False
    assert out["events"]["filled"]["fields"]["avg_price"] is False
    assert out["events"]["filled"]["fields"]["amount_usdt"] is True


@pytest.mark.asyncio
async def test_upsert_and_reset(monkeypatch, svc) -> None:
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(nc, "async_session_factory", lambda: _FakeSessionCtx(session))
    await svc.upsert(CHANNEL, enabled=False)
    assert svc.channel_enabled(CHANNEL) is False
    assert session.committed
    session2 = _FakeSession()
    monkeypatch.setattr(nc, "async_session_factory", lambda: _FakeSessionCtx(session2))
    await svc.reset(CHANNEL)
    assert svc.channel_enabled(CHANNEL) is True


@pytest.mark.asyncio
async def test_refresh_db_failure_conservative(monkeypatch, svc) -> None:
    svc._store_row(CHANNEL, False, nc.default_channel_config())

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(nc, "async_session_factory", _boom)
    await svc.refresh()
    assert svc.channel_enabled(CHANNEL) is False


def test_validate_unknown_channel() -> None:
    with pytest.raises(ValueError):
        nc.validate_channel_id("not_a_channel")


def test_wechat_event_and_field_gates(svc, monkeypatch) -> None:
    monkeypatch.setattr(nc, "notify_config", svc)
    import binance_predict.services.wechat_notifier as wn
    monkeypatch.setattr(wn, "notify_config", svc)
    n = WeChatNotifier()
    with patch.object(settings, "wechat_radar_enabled", True), \
         patch.object(n, "push_markdown") as mock_push:
        cfg = nc.default_channel_config()
        cfg["events"]["radar"]["wechat"] = False
        svc._store_row("hm_inside_15m_v2", True, cfg)
        n.notify_pre_trade_radar(
            radar_type="15m 经典孕线反转",
            channel="hm_inside_15m_v2",
            direction="DOWN",
            target_time_ms=1725900000000,
            lead_seconds=90,
            max_exec_price=0.30,
            features_desc="前根大实体",
            dedup_key="radar_cfg_off",
        )
        mock_push.assert_not_called()

        cfg2 = nc.default_channel_config()
        cfg2["events"]["filled"]["fields"]["shares"] = False
        svc._store_row(CHANNEL, True, cfg2)
        n.notify_order_filled(
            channel=CHANNEL, direction="DOWN", window_start=1725900000000,
            avg_price=0.25, amount_usdt=5.0, shares=20.0, order_id=123,
        )
        body = mock_push.call_args[0][0]
        assert "策略通道" in body
        assert "成交均价" in body
        assert "获得股数" not in body


def test_wechat_low_balance_gate(svc, monkeypatch) -> None:
    import binance_predict.services.wechat_notifier as wn
    monkeypatch.setattr(wn, "notify_config", svc)
    g = nc.default_global_config()
    g["low_balance"]["wechat"] = False
    svc._store_row(GLOBAL_CHANNEL, True, g)
    n = WeChatNotifier()
    with patch.object(n, "push_markdown") as mock_push:
        n.notify_low_balance(3.0, 12.0)
        mock_push.assert_not_called()


@pytest.mark.asyncio
async def test_push_email_respects_channel_gate(monkeypatch, svc) -> None:
    monkeypatch.setattr(sn, "notify_config", svc)
    sent = []

    async def _fake(sub, body):
        sent.append(sub)
        return True

    monkeypatch.setattr(sn, "send_plain_email", _fake)
    monkeypatch.setattr(settings, "signal_push_email_enabled", True)
    cfg = nc.default_channel_config()
    cfg["events"]["settled"]["email"] = False
    svc._store_row(CHANNEL, True, cfg)
    assert await sn.push_signal_email("quote_edge", "s", "b", 1000, channel=CHANNEL) is False
    assert await sn.push_signal_email("x4", "s2", "b", 1001, channel=OTHER) is True
    assert sent == ["s2"]


@pytest.mark.asyncio
async def test_scene_email_blocked_by_notify_config(monkeypatch, svc) -> None:
    from contextlib import asynccontextmanager

    class _Session:
        def __init__(self, rows):
            self._rows = rows

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, _stmt):
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(self._rows)))

        async def commit(self):
            pass

    row = SimpleNamespace(
        id=1, status="PENDING", pattern="S1", pattern_type="bull_exhaust",
        market_start_15m=1_700_000_900_000, market_end_15m=1_700_001_800_000,
        cycle_open_price_15m=43210.0, settle_deadline=1_700_001_800_000,
        settle_btc_price=None, settle_outcome=None, down_price_15m=0.55,
        version=None, signal_time=1_700_000_000_000,
    )

    @asynccontextmanager
    async def _factory():
        yield _Session([row])

    monkeypatch.setattr(fbd_mod, "async_session_factory", _factory)
    monkeypatch.setattr(settings, "fake_breakout_email_enabled", True)
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: True)
    monkeypatch.setattr(fbd_mod, "has_scene_filled_order", AsyncMock(return_value=True))
    monkeypatch.setattr(sn, "notify_config", svc)
    monkeypatch.setattr(fbd_mod, "is_settled_email_enabled", lambda ch: svc.is_event_enabled(ch, "settled", "email"))
    cfg = nc.default_channel_config()
    cfg["events"]["settled"]["email"] = False
    svc._store_row("scene_bull_exhaust", True, cfg)
    det = FakeBreakoutDetector(
        collector=SimpleNamespace(store=SimpleNamespace(mid_price=None)),
        pm_15m_latest={},
    )
    det._update_pattern_stats = AsyncMock()
    mail = AsyncMock()
    det._send_signal_email_bg = mail  # type: ignore[method-assign]

    async def _kopen(_interval, _start_ms):
        return 43190.0

    monkeypatch.setattr(det, "_klines_open", _kopen)
    await det._settle_15m(1_700_001_800_000 + 60_000)
    mail.assert_not_awaited()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "login_password", PASSWORD)
    return TestClient(app)


def _auth() -> dict:
    return {"Authorization": f"Bearer {PASSWORD}"}


def test_notify_config_api_requires_auth(client) -> None:
    assert client.get("/api/notify/config").status_code == 401


def test_notify_config_get_and_put(client, monkeypatch) -> None:
    svc = NotificationConfigService()
    monkeypatch.setattr("binance_predict.main.notify_config", svc)
    monkeypatch.setattr(nc, "notify_config", svc)
    session = _FakeSession(scalar=None)
    monkeypatch.setattr(nc, "async_session_factory", lambda: _FakeSessionCtx(session))
    r = client.get("/api/notify/config", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert "channels" in body and "physical" in body
    assert "spt" not in str(body).lower()
    bad = client.put(
        "/api/notify/config",
        headers=_auth(),
        json={"channels": [{"channel": "not_real", "enabled": False}]},
    )
    assert bad.status_code == 422
    ok = client.put(
        "/api/notify/config",
        headers=_auth(),
        json={
            "global_config": {"wechat_enabled": True, "email_enabled": False},
            "channels": [{
                "channel": CHANNEL,
                "enabled": True,
                "config": {"events": {"radar": {"wechat": False, "fields": {"note": False}}}},
            }],
        },
    )
    assert ok.status_code == 200
    saved = ok.json()
    ch = next(c for c in saved["channels"] if c["channel"] == CHANNEL)
    assert ch["config"]["events"]["radar"]["wechat"] is False
    assert ch["config"]["events"]["radar"]["fields"]["note"] is False
    assert saved["global"]["config"]["email_enabled"] is False
