"""资金端点鉴权 fail-closed 回归测试（2026-08-25 风险评审 R1）。

此前 _require_auth 在 API_AUTH_TOKEN 为空时直接放行（fail-open），
toggle/转账/redeem 等资金端点在空 token 下对公网完全开放。
现资金/写端点统一挂 _require_auth_strict：token 未配置 → 503，
错误/缺失 token → 401；只读端点继续用 _require_auth（开发态放行语义不变）。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from binance_predict.config.settings import settings
from binance_predict.main import _require_auth, _require_auth_strict


def _creds(token: str | None):
    return None if token is None else SimpleNamespace(credentials=token)


@pytest.mark.asyncio
async def test_strict_503_when_token_not_configured(monkeypatch) -> None:
    """fail-closed：token 未配置时资金端点拒绝服务而非放行。"""
    monkeypatch.setattr(settings, "api_auth_token", "")
    with pytest.raises(HTTPException) as ei:
        await _require_auth_strict(_creds(None))
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_strict_401_on_wrong_or_missing_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_auth_token", "s3cret")
    for creds in (_creds(None), _creds("wrong"), _creds("")):
        with pytest.raises(HTTPException) as ei:
            await _require_auth_strict(creds)
        assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_strict_passes_with_correct_token(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_auth_token", "s3cret")
    await _require_auth_strict(_creds("s3cret"))  # 不抛即通过


@pytest.mark.asyncio
async def test_readonly_auth_keeps_dev_passthrough(monkeypatch) -> None:
    """只读端点依赖：空 token 开发态放行语义不变（前端无 Bearer）。"""
    monkeypatch.setattr(settings, "api_auth_token", "")
    await _require_auth(_creds(None))  # 不抛即通过


@pytest.mark.asyncio
async def test_readonly_auth_rejects_wrong_token_when_set(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_auth_token", "s3cret")
    with pytest.raises(HTTPException) as ei:
        await _require_auth(_creds("wrong"))
    assert ei.value.status_code == 401
