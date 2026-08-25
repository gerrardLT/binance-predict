"""
BTC 5min LLM 预测系统 V2 - Prediction 只读市场数据服务

只读地从 Binance Prediction Markets API 获取 5 分钟 BTC 预测市场的报价数据，
供情绪曲线 tracker 采样使用。

与交易模块 `BinancePredictionTrader` 的关键区别：
- 本服务 **不持有** 任何 token_id / active_market 等交易执行状态
- 本服务 **不修改** 交易模块的任何字段
- 仅返回只读快照 dataclass `MarketQuoteData`

设计目的（读写分离）：
tracker 每 15s 通过本服务读取报价，交易模块在下单时才独立调用自己的
`list_markets()` 获取最新 token_id，两者互不干扰，避免竞态导致买入错误 token。

API 文档：
- 列出市场: GET /sapi/v1/w3w/wallet/prediction/market/list

签名机制与 `BinancePredictionTrader._sign_request()` 保持一致（HMAC-SHA256）。
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import httpx
from loguru import logger

from ..config.settings import settings
from . import clock_sync
from .rate_limit import binance_request


@dataclass(frozen=True)
class MarketQuoteData:
    """
    BTC 预测市场的只读报价快照（单周期）

    所有字段均为只读，不包含任何 token_id 或市场状态对象，
    仅供情绪曲线采样使用。period 标识市场周期（5m | 15m）。
    """

    period: str
    up_price: float | None
    down_price: float | None
    up_chance: float | None
    down_chance: float | None
    participants: int | None
    trade_volume: float | None
    end_date: int | None
    start_date: int | None


class PredictionMarketDataService:
    """
    Binance 预测市场只读数据服务

    独立于交易模块 `BinancePredictionTrader`，拥有自己的签名逻辑，
    每次调用 `fetch_market_data()` 都新建一个 httpx.AsyncClient。

    本服务不缓存 token_id，也不持有 active_market 状态。
    """

    BASE_URL = "https://api.binance.com"

    def __init__(self) -> None:
        self._api_key = settings.binance_api_key
        self._api_secret = settings.binance_api_secret

        # Fix #15: 复用同一 httpx 客户端（启用连接池/keep-alive），
        # 避免每 15s 采样都新建连接导致的 TCP/TLS 握手开销。
        self._client: httpx.AsyncClient | None = None

        if not self._api_key or not self._api_secret:
            logger.warning("Binance API Key/Secret 未配置，预测市场数据读取不可用")
        else:
            logger.info("Prediction 只读市场数据服务初始化完成")

    def _get_client(self) -> httpx.AsyncClient:
        """懒初始化并复用 httpx 客户端。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def aclose(self) -> None:
        """关闭复用的 httpx 客户端（供 lifespan 关闭时调用）。"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _sign_request(self, params: dict) -> dict:
        """
        HMAC-SHA256 签名（Binance 标准 API 签名，仅用于 GET 请求）

        1. 将所有参数按 key 字母升序拼接为 query string
        2. 使用 API Secret 对 query string 做 HMAC-SHA256
        3. 将签名添加到参数中
        """
        params.setdefault("recvWindow", 60000)  # 60s 容错时钟偏差
        # Fix #7: 使用进程级共享时钟偏差
        params["timestamp"] = clock_sync.now_ms()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    async def sync_server_time(self) -> None:
        """与 Binance 服务器校准时钟偏差（委托给共享时钟模块）。

        Fix #7: 不再维护实例级 _time_offset_ms，转而使用进程级共享偏差，
        与交易服务保持一致，避免重复请求与偏差不一致。
        """
        await clock_sync.sync_server_time()

    async def fetch_market_data(self) -> MarketQuoteData:
        """查询 5 分钟 BTC 预测市场报价（向后兼容包装，内部走 fetch_market_data_multi）。"""
        quotes = await self.fetch_market_data_multi()
        return quotes.get("5m") or MarketQuoteData(
            "5m", None, None, None, None, None, None, None, None
        )

    @staticmethod
    def _classify_period(market: dict) -> str | None:
        """按 title/slug 识别市场周期：'5m' | '15m' | None（其他周期暂不关注）。"""
        title = (market.get("title") or "").lower()
        slug = (market.get("slug") or "").lower()
        text = f"{title} {slug}"
        # 先判 15m 再判 5m（'15m' 包含 '5m' 子串）
        if "15m" in text:
            return "15m"
        if "5m" in text:
            return "5m"
        return None

    async def fetch_market_data_multi(self) -> dict[str, MarketQuoteData]:
        """
        查询活跃的 BTC 预测市场，返回 5m/15m 两个周期的只读报价快照

        筛选 chartType=CRYPTO_UP_DOWN 且 symbol=BTCUSDT 的市场，按 title/slug
        中的周期标记分类（'5m'/'15m'），直接提取 outcome 中的 price/chance 及元数据。

        分页拉取：币安市场列表默认按推荐排序（含全币种+事件市场），BTC 15m
        市场可能不在首页；按 hasMore 翻页确保短周期市场不遗漏。
        （实测：sortBy/orderBy 参数该 sapi 端点不接受，不可传）
        提前终止：两个周期都找到后不再翻页（控制采样频率下的 API 开销）。

        与交易模块不同，本方法：
        - 不缓存 token_id
        - 不修改任何外部状态
        - 无论成功与否都返回 dict（失败时对应周期缺席）

        Returns:
            dict[str, MarketQuoteData]: key 为 '5m'/'15m'；API 失败时返回空 dict。
        """
        quotes: dict[str, MarketQuoteData] = {}
        offset = 0
        limit = 100  # 官方上限 100

        for _page in range(5):  # 最多翻 5 页兑底
            params = self._sign_request({
                "limit": limit,
                "offset": offset,
            })

            try:
                client = self._get_client()
                resp = await binance_request(client, "GET",
                    f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/market/list",
                    params=params,
                    headers={"X-MBX-APIKEY": self._api_key},
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "读取预测市场数据失败 (HTTP {}): {}",
                    e.response.status_code,
                    e.response.text,
                )
                return quotes
            except Exception as e:
                logger.error("读取预测市场数据失败: {}", e)
                return quotes

            page = data.get("marketTopics", [])
            self._extract_period_quotes(page, quotes)

            # 提前终止：两周期均已找到，或无更多页
            if ("5m" in quotes and "15m" in quotes) or not data.get("hasMore") or not page:
                break
            offset += limit

        q5 = quotes.get("5m")
        q15 = quotes.get("15m")
        logger.debug(
            "读取预测报价 | 5m DOWN={} end={} | 15m DOWN={} end={}",
            q5.down_price if q5 else None,
            q5.end_date if q5 else None,
            q15.down_price if q15 else None,
            q15.end_date if q15 else None,
        )
        return quotes

    def _extract_period_quotes(
        self, markets: list[dict], quotes: dict[str, MarketQuoteData]
    ) -> None:
        """从一页市场列表中提取 5m/15m 报价，写入 quotes（已有的周期不覆盖）。"""
        for market in markets:
            if (
                market.get("chartType") != "CRYPTO_UP_DOWN"
                or market.get("symbol") != "BTCUSDT"
            ):
                continue

            period = self._classify_period(market)
            if period is None or period in quotes:
                continue

            up_price: float | None = None
            down_price: float | None = None
            up_chance: float | None = None
            down_chance: float | None = None

            # 提取 UP/DOWN 的 price/chance（不缓存 tokenId）
            for sub_market in market.get("markets", []):
                for outcome in sub_market.get("outcomes", []):
                    name = (outcome.get("name") or "").upper()
                    price = outcome.get("price")
                    chance = outcome.get("chance")

                    if name in ("UP", "YES"):
                        up_price = float(price) if price is not None else None
                        up_chance = float(chance) if chance is not None else None
                    elif name in ("DOWN", "NO"):
                        down_price = float(price) if price is not None else None
                        down_chance = float(chance) if chance is not None else None

            quotes[period] = MarketQuoteData(
                period=period,
                up_price=up_price,
                down_price=down_price,
                up_chance=up_chance,
                down_chance=down_chance,
                participants=market.get("participantCount"),
                trade_volume=market.get("tradeVolume"),
                end_date=market.get("endDate"),
                start_date=market.get("startDate"),
            )
