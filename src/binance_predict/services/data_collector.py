"""仅为情绪窗口提供现货中间价的 Binance 数据采集器。"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from dataclasses import dataclass

import httpx
import websockets
from loguru import logger

from ..config.settings import settings


@dataclass
class MarketDataStore:
    """当前现货最优报价与连接状态。"""

    best_bid: float = 0.0
    best_ask: float = 0.0
    last_ws_spot_update: float = 0.0
    ws_spot_connected: bool = False

    @property
    def mid_price(self) -> float:
        """中间价 = (最优买价 + 最优卖价) / 2；报价未就绪时为 0。"""
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return 0.0


class BinanceDataCollector:
    """订阅现货 ``bookTicker``，向 Agent 提供中间价。"""

    def __init__(self) -> None:
        self.store = MarketDataStore()
        self._running = False
        self._spot_ws: websockets.WebSocketClientProtocol | None = None

    async def start(self) -> None:
        """标记采集器开始运行；WebSocket 由 lifespan 后台任务连接。"""
        self._running = True
        logger.info("Binance 中间价采集服务启动")

    async def stop(self) -> None:
        """停止采集并关闭当前 WebSocket 连接。"""
        self._running = False
        self.store.ws_spot_connected = False
        if self._spot_ws is not None:
            await self._spot_ws.close()
        logger.info("Binance 中间价采集服务停止")

    async def connect_spot_ws(self) -> None:
        """连接现货组合流并订阅唯一需要的 ``bookTicker``。"""
        symbol_lower = settings.symbol.lower()
        ws_url = (
            f"{settings.binance_spot_ws_url.replace('/ws', '/stream')}"
            f"?streams={symbol_lower}@bookTicker"
        )
        reconnect_delay = 2

        while self._running:
            try:
                logger.info("连接现货 WebSocket: {}", ws_url[:100])
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=10,
                ) as ws:
                    self._spot_ws = ws
                    self.store.ws_spot_connected = True
                    reconnect_delay = 2
                    logger.info("现货 WebSocket 连接成功（bookTicker）")

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        await self._handle_spot_message(raw_msg)
                        self.store.last_ws_spot_update = time.time()

            except websockets.ConnectionClosed as exc:
                logger.warning("现货 WebSocket 断连: {}", exc)
            except (ssl.SSLError, OSError) as exc:
                logger.error("现货 WebSocket 网络/SSL 异常: {}", exc)
            except Exception as exc:
                logger.error("现货 WebSocket 异常: {}", exc)
            finally:
                self._spot_ws = None
                self.store.ws_spot_connected = False

            if self._running:
                logger.info("{} 秒后重连现货 WebSocket", reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    async def _handle_spot_message(self, raw_msg: str) -> None:
        """解析组合流中的 ``bookTicker`` 最优买卖报价。"""
        try:
            msg = json.loads(raw_msg)
            stream = msg.get("stream", "")
            if "bookticker" not in stream.lower():
                return

            data = msg.get("data", msg)
            self.store.best_bid = float(data["b"])
            self.store.best_ask = float(data["a"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("现货 bookTicker 消息解析异常: {} | raw={}", exc, raw_msg[:200])

    async def fetch_mid_price(self) -> float:
        """通过 REST API 获取最新 bookTicker 并更新 mid_price。

        WebSocket bookTicker 流可能失效，此方法作为可靠后备，
        在窗口切换和归档时按需调用（约每 5 分钟 2 次）。

        Returns:
            中间价浮点数；若 REST 和缓存均无法获取有效价格则返回 0.0，
            调用方应检查返回值 > 0 以避免除零错误。
        """
        url = "https://api.binance.com/api/v3/ticker/bookTicker"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params={"symbol": settings.symbol})
                resp.raise_for_status()
                data = resp.json()
                bid = float(data["bidPrice"])
                ask = float(data["askPrice"])
                if bid <= 0 or ask <= 0:
                    logger.error("REST bookTicker 返回无效价格: bid={} ask={}", bid, ask)
                    return self._safe_cached_mid_price()
                self.store.best_bid = bid
                self.store.best_ask = ask
                self.store.last_ws_spot_update = time.time()
                mid = (bid + ask) / 2
                logger.debug("REST mid_price 更新 | bid={} ask={} mid={:.2f}", bid, ask, mid)
                return mid
        except Exception as exc:
            logger.warning("REST fetch_mid_price 失败: {} | 回退使用缓存 mid_price={:.2f}", exc, self.store.mid_price)
            return self._safe_cached_mid_price()

    def _safe_cached_mid_price(self) -> float:
        """返回缓存的 mid_price，若为 0.0 则记录严重警告。"""
        cached = self.store.mid_price
        if cached <= 0:
            logger.critical(
                "fetch_mid_price 缓存 mid_price 为 0.0，"
                "WebSocket 可能从未连接成功。请检查 Binance WS 连接状态。"
            )
        return cached
