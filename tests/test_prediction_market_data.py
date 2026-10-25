"""PredictionMarketDataService market/list 拉取测试（2026-10-25 -1021 加固）。

覆盖：
- fetch_market_data_multi 先驱动周期性时钟校准（60s 节流委托），
  采样循环每 5s 调用即实现持续刷新（事故根因：启动单次校准全天未刷新）
- market/list 遇 -1021：强制重校准 + 重签重试一次后成功
- 其他错误照旧失败（返回空 quotes，不重试）

不触网络：_get_client / clock_sync.sync_server_time 全部替身。
"""

from __future__ import annotations

import httpx
import pytest

from binance_predict.services import clock_sync
from binance_predict.services.prediction_market_data import (
    PredictionMarketDataService,
)

_LIST_REQ = httpx.Request(
    "GET", "https://api.binance.com/sapi/v1/w3w/wallet/prediction/market/list"
)


class _SeqClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def get(self, _url, params=None, headers=None):
        self.calls += 1
        return self._responses.pop(0)


def _btc_topic(period_mark: str, start_ms: int) -> dict:
    return {
        "chartType": "CRYPTO_UP_DOWN",
        "symbol": "BTCUSDT",
        "title": f"Bitcoin Up or Down {period_mark}",
        "slug": f"btc-{period_mark}",
        "marketTopicId": 15001,
        "startDate": start_ms,
        "endDate": start_ms + 300_000,
        "markets": [
            {
                "tradingStatus": "OPEN",
                "outcomes": [
                    {"name": "UP", "tokenId": "T-UP", "price": 0.4, "chance": 0.4},
                    {"name": "DOWN", "tokenId": "T-DOWN", "price": 0.6, "chance": 0.6},
                ],
            }
        ],
    }


def _svc_with_client(monkeypatch, client) -> PredictionMarketDataService:
    svc = PredictionMarketDataService()
    svc._api_key = "k"
    svc._api_secret = "s"
    monkeypatch.setattr(svc, "_get_client", lambda: client)
    return svc


@pytest.fixture(autouse=True)
def _stub_sync(monkeypatch):
    """记录同步调用并避免触网。"""

    calls = []

    async def _fake_sync(*, force: bool = False, **_kw):
        calls.append(force)
        return 0

    monkeypatch.setattr(clock_sync, "sync_server_time", _fake_sync)
    yield calls


@pytest.mark.asyncio
async def test_fetch_drives_periodic_sync(monkeypatch):
    ok = httpx.Response(
        200,
        json={"marketTopics": [_btc_topic("5m", 1_792_000_000_000)], "hasMore": False},
        request=_LIST_REQ,
    )
    svc = _svc_with_client(monkeypatch, _SeqClient([ok]))

    quotes = await svc.fetch_market_data_multi()

    assert "5m" in quotes  # 5m 报价解析成功（15m 缺席属正常：本页没有 15m 市场）
    assert quotes["5m"].down_price == 0.6


@pytest.mark.asyncio
async def test_fetch_heals_and_retries_on_1021(monkeypatch, _stub_sync):
    rejected = httpx.Response(
        400,
        json={"code": -1021, "msg": "Timestamp ahead"},
        request=_LIST_REQ,
    )
    ok = httpx.Response(
        200,
        json={"marketTopics": [_btc_topic("5m", 1_792_000_000_000)], "hasMore": False},
        request=_LIST_REQ,
    )
    client = _SeqClient([rejected, ok])
    svc = _svc_with_client(monkeypatch, client)

    quotes = await svc.fetch_market_data_multi()

    assert client.calls == 2
    assert _stub_sync == [False, True]  # 周期校准 + -1021 强制重校准
    assert quotes.get("5m") is not None


@pytest.mark.asyncio
async def test_fetch_no_retry_on_other_error(monkeypatch, _stub_sync):
    rejected = httpx.Response(
        400, json={"code": -1003, "msg": "other"}, request=_LIST_REQ
    )
    client = _SeqClient([rejected])
    svc = _svc_with_client(monkeypatch, client)

    quotes = await svc.fetch_market_data_multi()

    assert client.calls == 1
    assert _stub_sync == [False]  # 仅周期校准，无 -1021 强制重校准
    assert quotes == {}
