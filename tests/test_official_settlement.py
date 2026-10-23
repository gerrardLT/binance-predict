"""官方结算源（official_settlement）单元测试。

覆盖：ET 标题生成（夏令时/冬令时/整点/跨日）、winning_outcome 解析、
created_time 去重、缓存命中、PENDING 不缓存、查询失败不缓存。
不触网络：_request 全部用替身。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from binance_predict.services.official_settlement import (
    OfficialSettlementClient,
    eastern_time,
    market_title,
)

WS = 1_792_722_900_000  # 2026-10-23 02:35 UTC = 10:35PM EDT Oct 22
TITLE = "Bitcoin Up or Down - October 22, 10:35PM-10:40PM ET"


def _ms(y, mo, d, h, mi) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


# ------------------------------------------------------------------
# 标题 / 时区
# ------------------------------------------------------------------

def test_market_title_5m_edt() -> None:
    assert market_title("5m", WS) == TITLE


def test_market_title_15m_on_the_hour() -> None:
    # 02:30 UTC = 10:30PM EDT（整点省略 :00）
    assert market_title("15m", _ms(2026, 10, 23, 2, 30)) == (
        "Bitcoin Up or Down - October 22, 10:30PM-10:45PM ET"
    )


def test_market_title_cross_midnight() -> None:
    # 03:55 UTC Oct 23 = 23:55 EDT Oct 22，结束 00:00 次日
    assert market_title("5m", _ms(2026, 10, 23, 3, 55)) == (
        "Bitcoin Up or Down - October 22, 11:55PM-12AM ET"
    )


def test_market_title_winter_est() -> None:
    # 12 月为 EST（UTC-5）：14:35 UTC = 09:35
    assert market_title("5m", _ms(2026, 12, 1, 14, 35)) == (
        "Bitcoin Up or Down - December 1, 9:35AM-9:40AM ET"
    )


def test_market_title_dst_transition_window() -> None:
    # 2026-03-08 06:55 UTC 为 EST 01:55，窗口 5 分钟后结束（标题 1:55AM-2:00AM，
    # 不能跳到切换后的 3:00AM）
    assert market_title("5m", _ms(2026, 3, 8, 6, 55)) == (
        "Bitcoin Up or Down - March 8, 1:55AM-2AM ET"
    )


def test_eastern_time_dst_boundaries() -> None:
    # 2026-03-08 07:00 UTC 起 EDT；2026-11-01 06:00 UTC 起 EST
    before = eastern_time(datetime(2026, 3, 8, 6, 59, tzinfo=timezone.utc))
    after = eastern_time(datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc))
    assert before.utcoffset().total_seconds() == -5 * 3600
    assert after.utcoffset().total_seconds() == -4 * 3600
    assert eastern_time(
        datetime(2026, 11, 1, 5, 59, tzinfo=timezone.utc)
    ).utcoffset().total_seconds() == -4 * 3600
    assert eastern_time(
        datetime(2026, 11, 1, 6, 0, tzinfo=timezone.utc)
    ).utcoffset().total_seconds() == -5 * 3600


# ------------------------------------------------------------------
# 结果解析
# ------------------------------------------------------------------

def _payload(*markets: dict) -> dict:
    return {"markets": list(markets)}


@pytest.mark.asyncio
async def test_fetch_prefers_event_slug(monkeypatch) -> None:
    """同名跨年市场：event_slug 含窗口秒级时间戳，优先精确命中当期（不受标题缺年份影响）。"""
    client = OfficialSettlementClient(["k1"])
    payload = _payload(
        {  # 去年同名市场（干扰项）
            "title": TITLE, "event_slug": "btc-updown-5m-1729572600",
            "winning_outcome": 0, "created_time": "2025-10-22T02:36:38Z",
        },
        {
            "title": TITLE, "event_slug": "btc-updown-5m-1792722900",
            "winning_outcome": 1, "created_time": "2026-10-22T02:36:38Z",
        },
    )

    async def _request(title: str):
        assert title == TITLE
        return payload

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) == "DOWN"


@pytest.mark.asyncio
async def test_fetch_pending_cached_briefly(monkeypatch) -> None:
    """PENDING 短缓存 60s：轮询期不重复打 API，但过期后会重试。"""
    client = OfficialSettlementClient(["k1"])
    calls = []

    async def _request(title: str):
        calls.append(title)
        return _payload({"title": TITLE, "winning_outcome": None,
                         "created_time": "2026-10-23T02:36:38Z"})

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) == "PENDING"
    assert await client.fetch_outcome("5m", WS) == "PENDING"
    assert len(calls) == 1  # 60s TTL 内命中负缓存


@pytest.mark.asyncio
async def test_fetch_failure_cached_briefly(monkeypatch) -> None:
    """查询失败也短缓存 60s，避免限流期间每笔订单重试打爆 API。"""
    client = OfficialSettlementClient(["k1"])
    calls = []

    async def _request(title: str):
        calls.append(title)
        return None

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) is None
    assert await client.fetch_outcome("5m", WS) is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fetch_rejects_far_away_same_title_market(monkeypatch) -> None:
    """标题不含年份：同名旧市场（created_time 距窗口 >1h）必须拒绝，不能当当期结果。"""
    client = OfficialSettlementClient(["k1"])

    async def _request(title: str):
        return _payload({"title": TITLE, "winning_outcome": 1,
                         "created_time": "2025-10-22T02:36:38Z"})

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) is None


@pytest.mark.asyncio
async def test_fetch_naive_created_time_does_not_crash(monkeypatch) -> None:
    """created_time 无时区（naive）时按 UTC 处理，不能抛 TypeError 静默关掉官方源。"""
    client = OfficialSettlementClient(["k1"])

    async def _request(title: str):
        return _payload({"title": TITLE, "winning_outcome": 0,
                         "created_time": "2026-10-23T02:36:38"})

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) == "UP"


@pytest.mark.asyncio
async def test_fetch_success_cached(monkeypatch) -> None:
    client = OfficialSettlementClient(["k1"])
    calls = []

    async def _request(title: str):
        calls.append(title)
        return _payload({"title": TITLE, "winning_outcome": 0,
                         "created_time": "2026-10-23T02:36:38Z"})

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) == "UP"
    assert await client.fetch_outcome("5m", WS) == "UP"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_fetch_missing_market_returns_none(monkeypatch) -> None:
    client = OfficialSettlementClient(["k1"])

    async def _request(title: str):
        return _payload({"title": "Some Other Market", "winning_outcome": 1})

    monkeypatch.setattr(client, "_request", _request)
    assert await client.fetch_outcome("5m", WS) is None


@pytest.mark.asyncio
async def test_disabled_client_without_keys() -> None:
    client = OfficialSettlementClient([])
    assert client.enabled is False
    assert await client.fetch_outcome("5m", WS) is None
