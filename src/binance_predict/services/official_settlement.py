"""币安预测市场官方结算结果（winning_outcome）查询。

用途：TradeSettler 结算与每日对账的权威源；币安现货 K 线归档仅作失败回退。

为什么需要：币安 5m/15m Up-or-Down 市场的官方结算源是 **Chainlink BTC/USDT
top-of-book mid price**（mid = Binance 买一/卖一均价），不是币安现货 K 线
open/close。2026-08-25→10-23 实盘 1175 笔中 9 笔（0.77%）两者在
0.001~1.73bp 的微幅窗口方向相反，导致本地结算判反（订单 2865 等）。

数据源：PREDEXON /v2/predictfun/markets?search=<ET 标题>
- winning_outcome：0=UP，1=DOWN（链上 resolution，与币安赔付一致）
- 标题不含年份：用 created_time 与窗口起点距离去重
- 免费档每 key 1 rps，多 key 轮询；Cloudflare 要求浏览器 UA

返回语义（fetch_outcome）：
- "UP" / "DOWN"：官方已结算
- "PENDING"：市场已找到但尚无 winning_outcome（结算索引延迟）
- None：查询失败 / 未找到市场 / 未配置 key（调用方回退 K 线）
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from loguru import logger

from binance_predict.config.settings import settings

_BASE = "https://api.predexon.com"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
_CACHE_LIMIT = 4096
_MIN_INTERVAL = 1.05  # 单 key 1 rps（含余量）
_NEGATIVE_TTL = 60.0  # PENDING/查询失败短缓存（秒）：轮询期不重复打 API
_MAX_CREATED_DISTANCE = 14 * 86400.0  # 标题回退时 created_time 距窗口上限（14 天）


def _is_dst(dt_utc: datetime) -> bool:
    """美国东部夏令时（2007 起规则）：3 月第二个周日 07:00 UTC 起，
    11 月第一个周日 06:00 UTC 止。手写规则避免依赖容器 tzdata。"""
    year = dt_utc.year
    march1 = datetime(year, 3, 1, tzinfo=timezone.utc)
    first_sunday_march = 1 + (6 - march1.weekday()) % 7
    start = datetime(year, 3, first_sunday_march + 7, 7, tzinfo=timezone.utc)
    nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
    first_sunday_nov = 1 + (6 - nov1.weekday()) % 7
    end = datetime(year, 11, first_sunday_nov, 6, tzinfo=timezone.utc)
    return start <= dt_utc < end


def eastern_time(dt_utc: datetime) -> datetime:
    """UTC → 美东时间（自动判别 EDT/EST）。"""
    offset = -4 if _is_dst(dt_utc) else -5
    return dt_utc.astimezone(timezone(timedelta(hours=offset)))


def _hhmm(dt: datetime) -> str:
    hour = dt.strftime("%I").lstrip("0") or "12"
    ampm = dt.strftime("%p")
    return f"{hour}{ampm}" if dt.minute == 0 else f"{hour}:{dt.minute:02d}{ampm}"


def market_title(market_period: str, window_start_ms: int) -> str:
    """生成币安市场标题（ET 时区、整点省略 :00）。

    例：5m 窗口 1792722900000 → "Bitcoin Up or Down - October 22, 10:35PM-10:40PM ET"
    15m 整点 → "Bitcoin Up or Down - October 22, 10:30PM-10:45PM ET"

    end 在 start 的本地时间上加固定时长（不重新判 DST）：切换当口窗口
    与币安标题一致显示为 1:55AM-2:00AM，而非跳变后的 3:00AM。
    """
    minutes = 15 if market_period == "15m" else 5
    start_utc = datetime.fromtimestamp(int(window_start_ms) / 1000, tz=timezone.utc)
    start = eastern_time(start_utc)
    end = start + timedelta(minutes=minutes)
    return (f"Bitcoin Up or Down - {start.strftime('%B')} {start.day}, "
            f"{_hhmm(start)}-{_hhmm(end)} ET")


class OfficialSettlementClient:
    """官方结算结果查询客户端（内存缓存 + 多 key 轮询 + 1 rps 节流）。"""

    def __init__(self, keys: list[str], timeout: float = 15.0) -> None:
        self._keys = [k.strip() for k in keys if k and k.strip()]
        self._timeout = timeout
        self._cursor = 0
        # key -> (outcome|None, expires_at_monotonic)；正结果永久，负结果 60s
        self._cache: dict[tuple[str, int], tuple[str | None, float]] = {}
        self._lock = asyncio.Lock()
        # per-key 节流：不同 key 各自 1 rps，轮询可连续发（待免费档限制）
        self._last_call: list[float] = [0.0] * len(self._keys)

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    async def fetch_outcome(self, market_period: str, window_start_ms: int) -> str | None:
        """查询单个窗口的官方结果；语义见模块 docstring。"""
        if not self._keys or window_start_ms is None:
            return None
        ws = int(window_start_ms)
        cache_key = (market_period or "5m", ws)
        cached = self._cache.get(cache_key)
        if cached is not None and cached[1] > time.monotonic():
            return cached[0]

        title = market_title(market_period, ws)
        payload = await self._request(title)
        if payload is None:
            self._remember(cache_key, None, _NEGATIVE_TTL)
            return None

        markets = payload.get("markets") or []
        # event_slug 含窗口秒级时间戳（如 btc-updown-5m-1792722900）：唯一且
        # 不受标题缺年份影响，优先精确匹配（币安可能提前 1 天建市，
        # created_time 不能当主键）。
        slug_suffix = f"-{ws // 1000}"
        slug_hits = [m for m in markets
                     if slug_suffix in str(m.get("event_slug") or "")]
        if slug_hits:
            best = slug_hits[0]
        else:
            hits = [m for m in markets if m.get("title") == title]
            if not hits:
                logger.debug("官方结算：未找到市场 | {}", title)
                self._remember(cache_key, None, _NEGATIVE_TTL)
                return None

            target = datetime.fromtimestamp(ws / 1000, tz=timezone.utc)

            def _dist(m: dict) -> float:
                raw = m.get("created_time")
                if not raw:
                    return math.inf
                try:
                    created = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                except (TypeError, ValueError):
                    return math.inf
                return abs((created - target).total_seconds())

            best = min(hits, key=_dist)
            if _dist(best) > _MAX_CREATED_DISTANCE:
                # 标题不含年份：同名旧市场（跨年）不能被当期窗口采信（High#1）
                logger.warning(
                    "官方结算：同名市场 created_time 距窗口过远（>%s 天），忽略 | {}",
                    int(_MAX_CREATED_DISTANCE // 86400), title,
                )
                self._remember(cache_key, None, _NEGATIVE_TTL)
                return None

        winning = best.get("winning_outcome")
        if winning in (0, 1):
            outcome = "UP" if winning == 0 else "DOWN"
            self._remember(cache_key, outcome, None)
            return outcome
        # 市场已找到但还没出结果：短缓存，60s 后重试
        self._remember(cache_key, "PENDING", _NEGATIVE_TTL)
        return "PENDING"

    async def _request(self, title: str) -> dict[str, Any] | None:
        async with self._lock:
            idx = self._cursor % len(self._keys)
            self._cursor += 1
            wait = _MIN_INTERVAL - (time.monotonic() - self._last_call[idx])
            if wait > 0:
                await asyncio.sleep(wait)
            key = self._keys[idx]
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(
                        _BASE + "/v2/predictfun/markets",
                        params={"search": title},
                        headers={"x-api-key": key, "User-Agent": _UA},
                    )
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:  # noqa: BLE001 - 查询失败一律回退 K 线
                logger.warning(
                    "官方结算查询失败 | {} | {}: {}",
                    title, type(exc).__name__, exc,
                )
                return None
            finally:
                self._last_call[idx] = time.monotonic()

    def _remember(self, key: tuple[str, int], outcome: str | None, ttl: float | None) -> None:
        """写入缓存：ttl=None 永久（UP/DOWN 不可变），否则在秒数后过期。"""
        expires = math.inf if ttl is None else time.monotonic() + ttl
        self._cache[key] = (outcome, expires)
        while len(self._cache) > _CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))


_client: OfficialSettlementClient | None = None


def get_official_settlement_client() -> OfficialSettlementClient | None:
    """进程级单例；未配置 PREDEXON_API_KEY 时返回 None（调用方回退 K 线）。"""
    global _client
    if _client is None:
        keys = [k for k in (settings.predexon_api_key or "").split(",") if k.strip()]
        if not keys:
            return None
        _client = OfficialSettlementClient(keys)
    return _client
