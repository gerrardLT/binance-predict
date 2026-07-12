"""
BTC 5min LLM 预测系统 V2 - Binance Prediction Trading 服务

通过 Binance Prediction Markets API 实现预测结果的自动交易。

API 文档：
- 列出市场: GET /sapi/v1/w3w/wallet/prediction/market/list
- 获取报价: POST /sapi/v1/w3w/wallet/prediction/trade/get-quote
- 下单: POST /sapi/v1/w3w/wallet/prediction/trade/place-order-bundle
- WebSocket 订单簿: wss://api.binance.com/sapi/wss

前置条件：
1. Binance API Key 开启 Prediction Trading 权限
2. 通过 Binance App 创建预测账户 + SAS 授权
3. 底层做市商: Predict.fun
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Any

import httpx
from loguru import logger

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import TradeOrderModel
from . import clock_sync


class BinancePredictionTrader:
    """
    Binance 预测市场交易服务

    负责：
    1. 查询活跃的 BTC 预测市场
    2. 根据 LLM 预测结果获取报价
    3. 执行交易下单
    4. 记录订单到数据库
    """

    BASE_URL = "https://api.binance.com"

    def __init__(self) -> None:
        self._api_key = settings.binance_api_key
        self._api_secret = settings.binance_api_secret
        self._wallet_address = settings.prediction_wallet_address  # 可通过 API 自动获取
        self._wallet_id = settings.prediction_wallet_id            # 可通过 API 自动获取
        self._trade_amount_usdt = settings.prediction_trade_amount_usdt

        # 时钟偏差补偿（毫秒），启动时通过 Binance /api/v3/time 校准
        # Fix #7: 不再维护实例级偏差，使用 clock_sync 进程级共享偏差

        # Fix #15: 复用同一 httpx 客户端（启用连接池/keep-alive）
        self._client: httpx.AsyncClient | None = None

        # Fix #21: 交易串行锁。list_markets() 会修改实例级 token_id/市场状态，
        # 若多个 execute_trade 并发执行会交错覆写导致买入错误 token。
        # 本锁保证 list_markets + token 选择 + 下单 整体串行。
        self._trade_lock = asyncio.Lock()

        # 缓存当前活跃的 BTC 预测市场信息
        self._active_market: dict | None = None
        self._up_token_id: str | None = None
        self._down_token_id: str | None = None

        # 5 分钟市场实时数据（由 list_markets 更新）
        self._5m_up_price: float | None = None
        self._5m_down_price: float | None = None
        self._5m_up_chance: float | None = None
        self._5m_down_chance: float | None = None
        self._5m_participant_count: int | None = None
        self._5m_trade_volume: float | None = None
        self._5m_liquidity: float | None = None
        self._5m_market_question: str | None = None
        self._5m_start_date: int | None = None
        self._5m_end_date: int | None = None

        if not self._api_key or not self._api_secret:
            logger.warning("Binance API Key/Secret 未配置，预测交易功能不可用")
        else:
            logger.info(
                "Binance Prediction Trading 服务初始化 | 单笔金额={} USDT",
                self._trade_amount_usdt,
            )

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
        HMAC-SHA256 签名（Binance 标准 API 签名）

        1. 将所有参数按 key 字母升序拼接为 query string
        2. 使用 API Secret 对 query string 做 HMAC-SHA256
        3. 将签名添加到参数中

        注意: 仅用于 GET 请求（params= 方式）。
        POST 请求请使用 _build_signed_url() 手动构建完整 URL。
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

    def _build_signed_url(self, path: str, params: dict) -> str:
        """
        构建带签名的完整 URL（用于 POST 请求）

        httpx.post(url, params=dict) 会对值做 URL 编码，
        导致编码后的 query string 与签名用的原始 string 不匹配（-1022）。
        解决方案：手动拼接 URL，确保签名与发送的 query string 完全一致。
        """
        params.setdefault("recvWindow", 60000)
        # Fix #7: 使用进程级共享时钟偏差
        params["timestamp"] = clock_sync.now_ms()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        signature = hmac.new(
            self._api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{self.BASE_URL}{path}?{query_string}&signature={signature}"

    async def sync_server_time(self) -> None:
        """与 Binance 服务器校准时钟偏差（委托给共享时钟模块）。

        Fix #7: 不再维护实例级 _time_offset_ms，使用进程级共享偏差，
        与只读数据服务保持一致，避免重复请求与偏差不一致。
        """
        await clock_sync.sync_server_time(force=True)

    async def fetch_wallet_info(self) -> dict | None:
        """
        自动获取预测钱包地址和 ID

        调用 GET /sapi/v1/w3w/wallet/prediction/wallet/list
        无需手动填写 walletAddress / walletId，API 自动返回。
        """
        params = self._sign_request({})

        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/wallet/list",
                params=params,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("获取预测钱包列表失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            logger.error("获取预测钱包列表失败: {}", e)
            return None

        wallets = data.get("wallets", [])
        if not wallets:
            logger.warning("未找到预测钱包，请先在 Binance App 中开通预测市场")
            return None

        # 取第一个钱包
        wallet = wallets[0]
        self._wallet_address = wallet.get("walletAddress", "")
        self._wallet_id = wallet.get("walletId", "")

        logger.info(
            "预测钱包信息获取成功 | address={} | walletId={}",
            self._wallet_address[:10] + "..." if self._wallet_address else "",
            self._wallet_id[:8] + "..." if self._wallet_id else "",
        )
        return wallet

    async def list_markets(self) -> list[dict]:
        """
        查询活跃的 BTC 预测市场

        筛选 chartType=CRYPTO_UP_DOWN 且 symbol=BTCUSDT 且 title 含 '5m' 的市场。
        直接提取 outcome 中的 price/chance，无需调用 get_quote。
        缓存 tokenId 供后续交易使用。
        """
        params = self._sign_request({
            "limit": 50,
            "offset": 0,
        })

        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/market/list",
                params=params,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("查询预测市场失败: {}", e)
            return []

        markets = data.get("marketTopics", [])
        btc_markets = []

        for market in markets:
            if (
                market.get("chartType") == "CRYPTO_UP_DOWN"
                and market.get("symbol") == "BTCUSDT"
            ):
                btc_markets.append(market)

                # 筛选 5 分钟市场（title 含 '5m' 或 slug 含 '5m'）
                is_5m = "5m" in (market.get("title") or "").lower() or "5m" in (market.get("slug") or "").lower()

                if is_5m:
                    # 更新 5 分钟市场元数据
                    self._5m_participant_count = market.get("participantCount")
                    self._5m_trade_volume = market.get("tradeVolume")
                    self._5m_liquidity = market.get("liquidity")
                    self._5m_market_question = market.get("question")
                    self._5m_start_date = market.get("startDate")
                    self._5m_end_date = market.get("endDate")

                    # 提取 UP/DOWN 的 tokenId + price/chance
                    for sub_market in market.get("markets", []):
                        for outcome in sub_market.get("outcomes", []):
                            name = outcome.get("name", "").upper()
                            token_id = outcome.get("tokenId")
                            price = outcome.get("price")
                            chance = outcome.get("chance")

                            if name in ("UP", "YES"):
                                self._up_token_id = token_id
                                self._5m_up_price = float(price) if price is not None else None
                                self._5m_up_chance = float(chance) if chance is not None else None
                            elif name in ("DOWN", "NO"):
                                self._down_token_id = token_id
                                self._5m_down_price = float(price) if price is not None else None
                                self._5m_down_chance = float(chance) if chance is not None else None

                    if not self._active_market and market.get("status") == "REGISTERED":
                        self._active_market = market
                else:
                    # 非 5 分钟市场：仍提取 tokenId 作为备用（交易用）
                    for sub_market in market.get("markets", []):
                        for outcome in sub_market.get("outcomes", []):
                            name = outcome.get("name", "").upper()
                            token_id = outcome.get("tokenId")
                            if name in ("UP", "YES") and not self._up_token_id:
                                self._up_token_id = token_id
                            elif name in ("DOWN", "NO") and not self._down_token_id:
                                self._down_token_id = token_id

        logger.info(
            "查询到 {} 个 BTC 预测市场 | 5m UP={}({:.1%}) / DOWN={}({:.1%}) | 参与者={} | 交易量={}",
            len(btc_markets),
            self._5m_up_price,
            self._5m_up_chance or 0,
            self._5m_down_price,
            self._5m_down_chance or 0,
            self._5m_participant_count,
            self._5m_trade_volume,
        )
        return btc_markets

    async def get_quote(
        self,
        token_id: str,
        side: str = "BUY",
        amount_usdt: float | None = None,
    ) -> dict | None:
        """
        获取交易报价

        Args:
            token_id: 预测 outcome token ID
            side: BUY 或 SELL
            amount_usdt: 交易金额（USDT），默认使用配置值

        Returns:
            报价响应 dict，包含 quoteId 等信息；失败返回 None
        """
        amount = amount_usdt or self._trade_amount_usdt
        # 转换为 wei 格式（18 位小数）
        amount_wei = str(int(amount * 10**18))

        # 使用 _build_signed_url 手动构建完整 URL
        # httpx.post(params=dict) 的 URL 编码会导致签名不匹配（-1022）
        signed_url = self._build_signed_url(
            "/sapi/v1/w3w/wallet/prediction/trade/get-quote",
            {
                "walletAddress": self._wallet_address,
                "tokenId": token_id,
                "side": side,
                "amountIn": amount_wei,
                "orderType": "MARKET",
                "slippageBps": 1200,  # 12% 滑点容忍
            },
        )

        logger.debug(
            "get_quote 签名 | wallet='{}' | token_len={} | offset_ms={}",
            self._wallet_address[:20] + "..." if self._wallet_address else "EMPTY",
            len(token_id),
            clock_sync.get_offset_ms(),
        )

        try:
            client = self._get_client()
            resp = await client.post(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            quote = resp.json()
            logger.info(
                "获取报价成功 | token={} | side={} | avgPrice={} | quoteId={}",
                token_id,
                side,
                quote.get("averagePrice"),
                quote.get("quoteId"),
            )
            return quote
        except httpx.HTTPStatusError as e:
            logger.error("获取报价失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            logger.error("获取报价异常: {}", e)
            return None

    async def place_order(self, quote: dict) -> dict | None:
        """
        执行下单

        Args:
            quote: get_quote 返回的报价响应

        Returns:
            下单响应 dict，包含 orderId；失败返回 None
        """
        # 使用 _build_signed_url 手动构建完整 URL
        signed_url = self._build_signed_url(
            "/sapi/v1/w3w/wallet/prediction/trade/place-order-bundle",
            {
                "walletAddress": self._wallet_address,
                "walletId": self._wallet_id,
                "quoteId": quote["quoteId"],
                "timeInForce": "FOK",
                "accountType": "SPOT",
                "orderType": "MARKET",
                "slippageBps": 1200,
            },
        )

        try:
            client = self._get_client()
            resp = await client.post(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info("下单成功 | orderId={}", result.get("orderId"))
            return result
        except httpx.HTTPStatusError as e:
            logger.error("下单失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            logger.error("下单异常: {}", e)
            return None

    async def execute_trade(
        self,
        prediction: str,
        confidence: float,
        prediction_id: int | None = None,
        agent_prediction_id: int | None = None,
    ) -> TradeOrderModel | None:
        """
        执行完整的交易流程

        根据预测方向（UP/DOWN）选择对应的 token，获取报价并下单。

        Args:
            prediction: 预测方向 UP/DOWN/NO_TRADE
            confidence: 预测置信度
            prediction_id: 关联的（旧 K 线决策路径）预测记录 ID
            agent_prediction_id: 关联的 Agent 预测记录 ID（新增，与 prediction_id 并存、
                互不干扰）；由 SentimentAgent.predict 传入，用于写入
                trade_orders.agent_prediction_id 并回填 AgentPrediction.trade_order_id

        Returns:
            TradeOrderModel 记录；NO_TRADE 或失败时返回 None
        """
        # NO_TRADE 不下单
        if prediction == "NO_TRADE":
            logger.info("预测为 NO_TRADE，跳过交易")
            return None

        # 检查必要配置
        if not self._api_key or not self._api_secret:
            logger.warning("API Key/Secret 未配置，无法执行交易")
            return await self._save_failed_order(
                prediction_id, "API Key/Secret 未配置",
                agent_prediction_id=agent_prediction_id,
            )

        # 确保有钱包信息（自动获取）
        if not self._wallet_address or not self._wallet_id:
            wallet = await self.fetch_wallet_info()
            if not wallet:
                return await self._save_failed_order(
                    prediction_id, "钱包信息获取失败，请先在 Binance App 开通预测市场",
                    agent_prediction_id=agent_prediction_id,
                )

        # Fix #21: 串行锁保护 list_markets + token 选择 + 下单，
        # 避免并发交易交错覆写 token_id 导致买入错误方向。
        async with self._trade_lock:
            # 确保有最新的 tokenId（每次交易前刷新）
            await self.list_markets()

            # 根据预测方向选择 token
            if prediction == "UP":
                token_id = self._up_token_id
            elif prediction == "DOWN":
                token_id = self._down_token_id
            else:
                logger.warning("未知预测方向: {}", prediction)
                return None

            if not token_id:
                logger.warning("未找到对应方向的 token_id | prediction={}", prediction)
                return await self._save_failed_order(
                    prediction_id, f"未找到 {prediction} 方向的 token",
                    agent_prediction_id=agent_prediction_id,
                )

            # 1. 获取报价
            quote = await self.get_quote(token_id, "BUY")
            if not quote:
                return await self._save_failed_order(
                    prediction_id, "获取报价失败",
                    agent_prediction_id=agent_prediction_id,
                )

            # 2. 下单
            order_result = await self.place_order(quote)
            if not order_result:
                return await self._save_failed_order(
                    prediction_id, "下单失败", quote=quote,
                    agent_prediction_id=agent_prediction_id,
                )

            # 3. 保存订单记录
            return await self._save_order(
                prediction_id=prediction_id,
                agent_prediction_id=agent_prediction_id,
                token_id=token_id,
                side="BUY",
                amount_in=str(quote.get("amountIn", "")),
                amount_out=str(quote.get("amountOut", "")),
                order_id=order_result.get("orderId"),
                status="FILLED",
                quote_json=quote,
            )

    async def _save_order(
        self,
        prediction_id: int | None,
        token_id: str,
        side: str,
        amount_in: str,
        amount_out: str | None,
        order_id: str | None,
        status: str,
        quote_json: dict | None = None,
        agent_prediction_id: int | None = None,
        error_message: str | None = None,
    ) -> TradeOrderModel | None:
        """
        保存订单到数据库

        agent_prediction_id 写入 trade_orders.agent_prediction_id，用于与 Agent 预测
        双向关联（旧 prediction_id 路径不传该值，行为保持不变）；
        error_message 仅在失败落库（status=FAILED）时写入，保证失败可追溯（规则 3，无静默降级）。
        """
        try:
            async with async_session_factory() as db:
                order = TradeOrderModel(
                    prediction_id=prediction_id,
                    agent_prediction_id=agent_prediction_id,
                    market_id=self._active_market.get("marketTopicId") if self._active_market else None,
                    token_id=token_id,
                    side=side,
                    amount_in=amount_in,
                    amount_out=amount_out,
                    order_id=order_id,
                    status=status,
                    quote_json=quote_json,
                    error_message=error_message,
                )
                db.add(order)
                await db.commit()
                logger.info(
                    "订单已保存 | id={} | status={} | agent_prediction_id={}",
                    order.id, status, agent_prediction_id,
                )
                return order
        except Exception as e:
            logger.error("保存订单失败: {}", e)
            return None

    async def _save_failed_order(
        self,
        prediction_id: int | None,
        error_msg: str,
        quote: dict | None = None,
        agent_prediction_id: int | None = None,
    ) -> TradeOrderModel | None:
        """
        保存失败订单到数据库

        落库 status=FAILED + error_message，不伪造成交（规则 3，无静默降级）；
        同样透传 agent_prediction_id，使失败订单亦可与 Agent 预测双向关联。
        """
        logger.warning(
            "交易失败 | prediction_id={} | agent_prediction_id={} | error={}",
            prediction_id, agent_prediction_id, error_msg,
        )
        return await self._save_order(
            prediction_id=prediction_id,
            agent_prediction_id=agent_prediction_id,
            token_id="",
            side="BUY",
            amount_in="0",
            amount_out=None,
            order_id=None,
            status="FAILED",
            quote_json=quote,
            error_message=error_msg,
        )
