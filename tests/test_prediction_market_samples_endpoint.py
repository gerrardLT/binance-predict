"""历史采样导出端点 /api/chart/prediction-market/samples 的冒烟测试。

该端点是「报价欠反应 × 量能」研究补覆盖的只读数据通道：按 period 从
prediction_market_samples 查真实 token 价 + 量能（participants/trade_volume），
与既有 /api/chart/prediction-market/15m 同源同翻页协议。

覆盖：
- 默认 period=5m，响应回显 period 且 points 含量能字段（区别于 15m 端点）；
- 非法 period 回落 5m（白名单 5m|15m）；
- since>0 走正序翻页分支、since=0 走 desc+reversed 分支，两者都按时间升序返回；
- oldest_ts = 首点 timestamp；
- 与其它 /api/* 一致受 LoginAuthMiddleware 保护（无 token 401）。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from binance_predict.config.settings import settings
from binance_predict.db.engine import get_db
from binance_predict.main import app

PASSWORD = "s3cret-samples"
PATH = "/api/chart/prediction-market/samples"


def _row(ts: int, up: float, vol: float, par: int) -> SimpleNamespace:
    """伪造一条 PredictionMarketSample（仅端点读取的 8 个字段）。"""
    return SimpleNamespace(
        timestamp=ts,
        up_price=up,
        down_price=round(1.0 - up, 4),
        up_pct=round(up * 100, 2),
        down_pct=round((1.0 - up) * 100, 2),
        participants=par,
        trade_volume=vol,
        btc_price=63000.0 + ts % 1000,
    )


@pytest.fixture
def authed_client(monkeypatch):
    """带 token 的 TestClient + get_db 覆盖（注入伪造 session）。"""
    monkeypatch.setattr(settings, "login_password", PASSWORD)

    # 忠实模拟 DB 排序：else 分支 SQL 是 ORDER BY timestamp DESC，端点再 reversed → 升序；
    # since>0 分支 SQL 是 ORDER BY timestamp ASC，端点直接透传 → 升序。
    desc_rows = [_row(300, 0.30, 3000.0, 30), _row(200, 0.20, 2000.0, 20), _row(100, 0.10, 1000.0, 10)]
    asc_rows = [_row(100, 0.11, 1100.0, 11), _row(200, 0.22, 2200.0, 22), _row(300, 0.33, 3300.0, 33)]

    def _result(rows):
        r = MagicMock()
        r.scalars.return_value.all.return_value = rows
        return r

    async def _override_get_db():
        session = AsyncMock()

        async def _execute(stmt):
            # 依据 SQL 的 ORDER BY 方向返回对应排序的 rows（与真实 DB 一致）
            return _result(desc_rows if "DESC" in str(stmt).upper() else asc_rows)

        session.execute = AsyncMock(side_effect=_execute)
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {PASSWORD}"})
    yield client
    app.dependency_overrides.clear()


def test_default_period_is_5m_and_has_volume_fields(authed_client):
    resp = authed_client.get(PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["period"] == "5m"
    assert body["poll_interval_sec"] == 15
    pts = body["points"]
    assert pts, "应返回采样点"
    # 量能字段是本端点相对 15m 端点的关键增量
    for key in ("participants", "trade_volume", "up_price", "down_price", "btc_price"):
        assert key in pts[0], f"points 缺少字段 {key}"


def test_invalid_period_falls_back_to_5m(authed_client):
    resp = authed_client.get(f"{PATH}?period=1h")
    assert resp.status_code == 200
    assert resp.json()["period"] == "5m"


def test_explicit_15m_period_echoed(authed_client):
    resp = authed_client.get(f"{PATH}?period=15m")
    assert resp.status_code == 200
    assert resp.json()["period"] == "15m"


def test_points_ascending_and_oldest_ts(authed_client):
    resp = authed_client.get(PATH)
    body = resp.json()
    ts = [p["timestamp"] for p in body["points"]]
    assert ts == sorted(ts), "points 必须按时间升序"
    assert body["oldest_ts"] == ts[0]


def test_since_paging_branch(authed_client):
    resp = authed_client.get(f"{PATH}?since=1&limit=10")
    assert resp.status_code == 200
    ts = [p["timestamp"] for p in resp.json()["points"]]
    assert ts == sorted(ts)


def test_requires_auth_without_token(monkeypatch):
    monkeypatch.setattr(settings, "login_password", PASSWORD)
    client = TestClient(app)  # 不带 Authorization
    resp = client.get(PATH)
    assert resp.status_code == 401
