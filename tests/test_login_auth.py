"""Web 登录认证测试（单一访问密码 + 全局中间件）。

覆盖：
- /api/auth/login：成功返回 token / 错误密码 401 / 未配置密码 401
- LoginAuthMiddleware：/api/* 未带或错误 Bearer 一律 401，正确 token 放行
- 豁免路径：/api/auth/login 本身、非 /api 路径
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from binance_predict.config.settings import settings
from binance_predict.main import app

PASSWORD = "s3cret-login"


@pytest.fixture
def client():
    # 不用 with 上下文：避免触发 lifespan 启动后台调度器
    return TestClient(app)


@pytest.fixture
def login_password(monkeypatch):
    monkeypatch.setattr(settings, "login_password", PASSWORD)
    return PASSWORD


# ------------------------------------------------------------
# 登录端点
# ------------------------------------------------------------

def test_login_success_returns_token(client, login_password):
    resp = client.post("/api/auth/login", json={"password": PASSWORD})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["token"] == PASSWORD


def test_login_wrong_password_401(client, login_password):
    resp = client.post("/api/auth/login", json={"password": "wrong"})
    assert resp.status_code == 401
    assert "密码错误" in resp.json()["detail"]


def test_login_without_server_password_401(client, monkeypatch):
    monkeypatch.setattr(settings, "login_password", "")
    resp = client.post("/api/auth/login", json={"password": "anything"})
    assert resp.status_code == 401
    assert "未配置" in resp.json()["detail"]


# ------------------------------------------------------------
# 全局认证中间件
# ------------------------------------------------------------

def test_api_without_token_401(client, login_password):
    resp = client.get("/api/fake-breakout/status")
    assert resp.status_code == 401


def test_api_with_wrong_token_401(client, login_password):
    resp = client.get(
        "/api/fake-breakout/status",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_api_with_valid_token_passes(client, login_password):
    resp = client.get(
        "/api/fake-breakout/status",
        headers={"Authorization": f"Bearer {PASSWORD}"},
    )
    assert resp.status_code != 401


def test_api_blocked_even_without_server_password(client, monkeypatch):
    """未配置 LOGIN_PASSWORD 时不做开发旁路：一律 401。"""
    monkeypatch.setattr(settings, "login_password", "")
    resp = client.get("/api/fake-breakout/status")
    assert resp.status_code == 401


def test_non_api_path_not_blocked(client, login_password):
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_health_probe_open_without_token(client, login_password):
    """/api/health 豁免：部署流水线/容器探针免认证探活。"""
    resp = client.get("/api/health")
    assert resp.status_code != 401
