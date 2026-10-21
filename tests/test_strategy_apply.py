"""一键应用策略建议端点的安全测试（2026-10-22）。

覆盖：合法降金额应用并写日志 / param 与 to_value 不可篡改 / 非 PENDING 拒绝 /
applyable=False 拒绝 / 停火需强确认 / 同通道冷冻期限流。

铁律：LLM 永远不能直接执行——所有校验在服务端，前端不得直连 /api/live/toggle。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import binance_predict.main as main_mod
from binance_predict.config.settings import settings
from binance_predict.db.engine import get_db
from binance_predict.db.models import StrategyAdvice
from binance_predict.main import app

PASSWORD = "s3cret-advice"
CH = "quote_contrarian_v2"
PATH = "/api/attribution/advices"


class FakeTrader:
    """多通道执行器替身：只记录热调调用，不碰任何真实外部接口。"""

    def __init__(self) -> None:
        self.set_calls: list[tuple[str, dict]] = []
        self.persisted: list[str] = []

    def set_channel(self, channel: str, **kwargs) -> None:
        self.set_calls.append((channel, kwargs))

    async def persist_channel(self, channel: str) -> None:
        self.persisted.append(channel)

    async def status_async(self) -> dict:
        return {"channels": [{"channel": CH, "enabled": True, "amount_usdt": 1.0,
                              "max_daily_orders": 100, "max_exec_price": 0.5}]}


def _advice(status: str = "PENDING", verdict: str = "CUT_AMOUNT",
            applyable: bool = True, param: str = "amount_usdt",
            from_v: float | bool = 2.0, to_v: float | bool = 1.0,
            expiry: str = "") -> StrategyAdvice:
    row = StrategyAdvice(
        channel=CH, trigger_type="MANUAL",
        prompt_version="strategy_advice_prompt_v1",
        advice_spec={
            "schema": "strategy_advice_v1", "channel": CH, "verdict": verdict,
            "diagnosis": {"primary_cause": "EDGE_DECAY", "evidence": [],
                          "supporting_test": {"name": "streak_binomial",
                                              "verdict": "ANOMALOUS_LOSING",
                                              "significant": True},
                          "randomness_check": {"verdict": "ANOMALOUS_LOSING"}},
            "action": {"param": param, "from_value": from_v, "to_value": to_v,
                       "duration_orders": 30, "expiry_condition": expiry},
            "confidence": 0.7, "counter_hypothesis": "反方", "kill_criteria": "撤回",
            "expected_effect": {}, "requires_human_review": True,
            "clamped": False, "applyable": applyable,
        },
        rejected=[], pre_metrics={}, status=status, application_log=[],
        tracking={"target_n": 30, "observed_n": 0, "verdict": None},
    )
    row.id = 1
    return row


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "login_password", PASSWORD)
    trader = FakeTrader()
    monkeypatch.setattr(main_mod, "multi_live_trader", trader)
    state: dict = {"row": _advice(), "applied": [], "others": []}

    async def _override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(side_effect=lambda model, pk: state["row"])
        calls = {"n": 0}

        async def _execute(stmt):
            # apply 端点两次 execute：①冷冻期查询（已 APPLIED）②同通道其他 PENDING
            calls["n"] += 1
            r = MagicMock()
            r.scalars.return_value.all.return_value = (
                state["applied"] if calls["n"] == 1 else state["others"])
            return r

        session.execute = AsyncMock(side_effect=_execute)
        session.commit = AsyncMock()
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {PASSWORD}"})
    c.trader = trader  # type: ignore[attr-defined]
    c.state = state  # type: ignore[attr-defined]
    yield c
    app.dependency_overrides.clear()


def test_apply_cut_amount_ok(client):
    resp = client.post(f"{PATH}/1/apply", json={"param": "amount_usdt", "to_value": 1.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert client.trader.set_calls == [(CH, {"amount_usdt": 1.0})]
    assert client.trader.persisted == [CH]
    assert client.state["row"].status == "APPLIED"
    log = client.state["row"].application_log
    assert log and log[-1]["param"] == "amount_usdt" and log[-1]["to_value"] == 1.0


def test_apply_param_mismatch_422(client):
    resp = client.post(f"{PATH}/1/apply",
                       json={"param": "max_exec_price", "to_value": 1.0})
    assert resp.status_code == 422
    assert client.trader.set_calls == []


def test_apply_to_value_cannot_be_changed(client):
    resp = client.post(f"{PATH}/1/apply", json={"param": "amount_usdt", "to_value": 0.5})
    assert resp.status_code == 422
    assert client.trader.set_calls == []


def test_apply_not_pending_rejected(client):
    client.state["row"] = _advice(status="APPLIED")
    resp = client.post(f"{PATH}/1/apply", json={"param": "amount_usdt", "to_value": 1.0})
    assert resp.status_code == 200
    assert "不可应用" in resp.json()["error"]
    assert client.trader.set_calls == []


def test_apply_not_applyable_rejected(client):
    client.state["row"] = _advice(applyable=False)
    resp = client.post(f"{PATH}/1/apply", json={"param": "amount_usdt", "to_value": 1.0})
    assert resp.status_code == 200
    assert "不可一键应用" in resp.json()["error"]
    assert client.trader.set_calls == []


def test_apply_pause_requires_confirm(client):
    client.state["row"] = _advice(verdict="PAUSE", param="enabled",
                                  from_v=True, to_v=False, expiry="n≥20 且 p<0.5")
    resp = client.post(f"{PATH}/1/apply", json={"param": "enabled"})
    assert resp.status_code == 422
    assert client.trader.set_calls == []
    # 强确认后才执行停火（enabled=False）
    resp2 = client.post(f"{PATH}/1/apply", json={"param": "enabled", "confirm": True})
    assert resp2.status_code == 200
    assert client.trader.set_calls == [(CH, {"enabled": False})]


def test_apply_freeze_period_limits_changes(client):
    # 近 7 天已有 2 次 APPLIED（默认上限 2）→ 参数变更被拒
    client.state["applied"] = [_advice(status="APPLIED"), _advice(status="APPLIED")]
    resp = client.post(f"{PATH}/1/apply", json={"param": "amount_usdt", "to_value": 1.0})
    assert resp.status_code == 429
    assert "冷冻期" in resp.json()["detail"]
    assert client.trader.set_calls == []
