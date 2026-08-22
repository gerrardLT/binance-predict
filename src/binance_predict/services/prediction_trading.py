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

        # 最近一次 API 调用的错误详情（诊断透传：人工测试单/日志排查用；
        # get_quote/place_order 失败时写入，成功时清空）
        self.last_api_error: str | None = None

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

        # 15 分钟市场实时数据（假突破信号系统：到期结算周期，暂不下注）
        self._15m_up_token_id: str | None = None
        self._15m_down_token_id: str | None = None
        self._15m_up_price: float | None = None
        self._15m_down_price: float | None = None
        self._15m_start_date: int | None = None
        self._15m_end_date: int | None = None

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

    async def fetch_spot_usdt_balance(self) -> float | None:
        """查询现货账户 USDT 可用余额（下单扣款来源）。

        调用 GET /api/v3/account（签名），从 balances 里取 USDT 的 free。
        资金从 Web3 钱包转回交易所时可能落在资金账户，此查询用于确认
        现货账户是否真正有可用 USDT。失败返回 None。
        """
        params = self._sign_request({})
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.BASE_URL}/api/v3/account",
                params=params,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("查询现货余额失败 (HTTP {}): {}", e.response.status_code, e.response.text[:300])
            return None
        except Exception as e:
            logger.error("查询现货余额失败: {}", e)
            return None

        for b in data.get("balances", []):
            if b.get("asset") == "USDT":
                try:
                    return float(b.get("free", 0))
                except (TypeError, ValueError):
                    return None
        return 0.0

    async def transfer_in(self, amount_usdt: float) -> dict | None:
        """从现货账户划转 USDT 入预测钱包（下单扣款来源）。

        预测市场下单扣的是预测钱包内余额（现货余额充足仍报 -9000
        即未划转），调用 POST /sapi/v1/w3w/wallet/prediction/transfer/inbound。
        失败返回 None，详情写入 last_api_error。
        """
        amount_wei = str(int(round(amount_usdt * 10**18)))
        params = {
            "walletId": self._wallet_id,
            "walletAddress": self._wallet_address,
            "fromTokenAmount": amount_wei,
            "accountType": "SPOT",
            "sourceBiz": "USER_TRANSFER",
        }
        # 官方示例参数走 --data 请求体：签名覆盖全部参数（含 timestamp/
        # recvWindow），业务参数以 JSON body 发送；全放 query 会报 -9000。
        signed = self._sign_request(dict(params))
        timestamp = signed["timestamp"]
        recv_window = signed["recvWindow"]
        signature = signed["signature"]

        try:
            client = self._get_client()
            resp = await client.post(
                (
                    f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/transfer/inbound"
                    f"?timestamp={timestamp}&recvWindow={recv_window}&signature={signature}"
                ),
                json=params,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_api_error = None
            logger.info("预测钱包入金成功 | amount={} USDT | resp={}", amount_usdt, data)
            return data
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("预测钱包入金失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("预测钱包入金异常: {}", e)
            return None

    @staticmethod
    def _classify_period(market: dict) -> str | None:
        """按 title/slug 识别市场周期：'5m' | '15m' | None。先判 15m（含 '5m' 子串）。"""
        text = f"{(market.get('title') or '').lower()} {(market.get('slug') or '').lower()}"
        if "15m" in text:
            return "15m"
        if "5m" in text:
            return "5m"
        return None

    async def list_markets(self) -> list[dict]:
        """
        查询活跃的 BTC 预测市场

        筛选 chartType=CRYPTO_UP_DOWN 且 symbol=BTCUSDT 的市场，按周期分类
        （5m 用于交易执行缓存 tokenId；15m 缓存报价供假突破信号系统读取）。
        直接提取 outcome 中的 price/chance，无需调用 get_quote。

        分页拉取（15m 市场不在首页）：币安市场列表默认按推荐排序，
        含全币种+事件市场，BTC 15m 可能被挤到后面的页；按 hasMore 翻页
        确保不遗漏。（实测：sortBy/orderBy 参数该 sapi 端点不接受，不可传）
        """
        markets: list[dict] = []
        offset = 0
        limit = 100  # 官方上限 100
        for _page in range(5):  # 最多翻 5 页兑底
            params = self._sign_request({
                "limit": limit,
                "offset": offset,
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
            except httpx.HTTPStatusError as e:
                logger.error(
                    "查询预测市场失败 (HTTP {}): {}",
                    e.response.status_code, e.response.text,
                )
                break
            except Exception as e:
                logger.error("查询预测市场失败: {}", e)
                break

            page = data.get("marketTopics", [])
            markets.extend(page)
            if not data.get("hasMore") or len(page) == 0:
                break
            offset += limit

        btc_markets = []

        for market in markets:
            if (
                market.get("chartType") == "CRYPTO_UP_DOWN"
                and market.get("symbol") == "BTCUSDT"
            ):
                btc_markets.append(market)

                # 按周期分类（先判 15m 再判 5m）
                period = self._classify_period(market)

                if period == "5m":
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
                elif period == "15m":
                    # 15 分钟市场：缓存 tokenId + 报价（假突破信号系统用，暂不下注）
                    self._15m_start_date = market.get("startDate")
                    self._15m_end_date = market.get("endDate")
                    for sub_market in market.get("markets", []):
                        for outcome in sub_market.get("outcomes", []):
                            name = outcome.get("name", "").upper()
                            token_id = outcome.get("tokenId")
                            price = outcome.get("price")
                            if name in ("UP", "YES"):
                                self._15m_up_token_id = token_id
                                self._15m_up_price = float(price) if price is not None else None
                            elif name in ("DOWN", "NO"):
                                self._15m_down_token_id = token_id
                                self._15m_down_price = float(price) if price is not None else None
                else:
                    # 非 5m/15m 市场：仍提取 tokenId 作为备用（交易用）
                    for sub_market in market.get("markets", []):
                        for outcome in sub_market.get("outcomes", []):
                            name = outcome.get("name", "").upper()
                            token_id = outcome.get("tokenId")
                            if name in ("UP", "YES") and not self._up_token_id:
                                self._up_token_id = token_id
                            elif name in ("DOWN", "NO") and not self._down_token_id:
                                self._down_token_id = token_id

        logger.info(
            "查询到 {} 个 BTC 预测市场 | 5m UP={}({:.1%}) / DOWN={}({:.1%}) | 15m DOWN={} | 参与者={} | 交易量={}",
            len(btc_markets),
            self._5m_up_price,
            self._5m_up_chance or 0,
            self._5m_down_price,
            self._5m_down_chance or 0,
            self._15m_down_price,
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
            self.last_api_error = None
            logger.info(
                "获取报价成功 | token={} | side={} | avgPrice={} | quoteId={}",
                token_id,
                side,
                quote.get("averagePrice"),
                quote.get("quoteId"),
            )
            return quote
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("获取报价失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("获取报价异常: {}", e)
            return None

    async def place_order(self, quote: dict, slippage_bps: int = 1200) -> dict | None:
        """
        执行下单

        Args:
            quote: get_quote 返回的报价响应
            slippage_bps: 滑点容忍（基点，默认 1200=12%）；信号实盘通道按
                执行价护栏动态收紧，防成交价突破护栏价（CodeReview Medium#2）

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
                "slippageBps": slippage_bps,
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
        amount_usdt: float | None = None,
        signal_version: str | None = None,
        window_start: int | None = None,
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
            amount_usdt: 自定义单笔金额（None 用配置值）；报价 edge 实盘灰度用
            signal_version: 触发信号版本（quote_momentum_v1 等），写入订单供实盘对账
            window_start: 目标 5m 窗口起始 ms（与 signal_version 联合唯一防重复开火）

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
            quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt)
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
                signal_version=signal_version,
                window_start=window_start,
            )

    async def execute_signal_trade(
        self,
        prediction: str,
        amount_usdt: float,
        signal_version: str,
        window_start: int,
        max_exec_price: float | None = None,
    ) -> TradeOrderModel | None:
        """
        信号驱动实盘专用通道（报价 edge LIVE）：报价 → 执行价护栏 → 下单 → 落表。

        与 execute_trade 的差异：
        1. 金额/信号版本/窗口由调用方显式传入（非配置默认）；
        2. **先占位后下单**（CodeReview High#1）：place_order 前先插 PENDING 行占住
           (signal_version, window_start) 唯一键，重复窗口在下单前即拒绝；
           成功/失败 UPDATE 该行——重启防重不再依赖"钱出去后"的事后提交；
        3. 报价后、下单前检查 averagePrice ≤ max_exec_price，超限弃单（不追贵）；
           且按护栏价动态收紧 slippageBps，成交价无法突破 max_exec_price。
        钱已出去而落库/更新失败时记 CRITICAL 日志，可追溯、不静默降级。
        """
        if not self._api_key or not self._api_secret:
            logger.warning("信号实盘：API Key/Secret 未配置 | signal={}", signal_version)
            return None

        if not self._wallet_address or not self._wallet_id:
            wallet = await self.fetch_wallet_info()
            if not wallet:
                logger.warning("信号实盘：钱包信息获取失败（未开通预测市场？）| signal={}",
                               signal_version)
                return None

        async with self._trade_lock:
            # 先占位后下单：PENDING 行占住唯一键；重复窗口（含重启/并发）在花钱前拒绝。
            pending = await self._reserve_order_slot(signal_version, window_start)
            if pending is None:
                logger.info("信号实盘：窗口 {} 已有订单占位，跳过（每窗一单）", window_start)
                return None

            await self.list_markets()

            if prediction == "UP":
                token_id = self._up_token_id
            elif prediction == "DOWN":
                token_id = self._down_token_id
            else:
                logger.warning("未知预测方向: {}", prediction)
                await self._update_signal_order(pending, "FAILED",
                                                error_message=f"未知预测方向: {prediction}")
                return pending

            if not token_id:
                await self._update_signal_order(
                    pending, "FAILED", error_message=f"未找到 {prediction} 方向的 token")
                return pending

            quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt)
            if not quote:
                detail = self.last_api_error or "无详情（网络异常？）"
                await self._update_signal_order(
                    pending, "FAILED",
                    error_message=f"获取报价失败 | {detail}",
                )
                return pending

            # 执行价护栏：报价均价超阈弃单（不追贵，保护回测 EV 口径）
            try:
                avg_price = float(quote.get("averagePrice") or 0.0)
            except (TypeError, ValueError):
                avg_price = 0.0
            if max_exec_price is not None and (avg_price <= 0 or avg_price > max_exec_price):
                await self._update_signal_order(
                    pending, "FAILED",
                    error_message=f"执行价护栏弃单 | averagePrice={avg_price} > {max_exec_price}",
                    quote_json=quote)
                return pending

            # 动态滑点收紧（CodeReview Medium#2）：FOK 成交价不得突破护栏价，
            # 否则 slippageBps=1200 会让 0.78 的护栏形同虚设（最高可成交 ~0.87）。
            slippage_bps = 1200
            if max_exec_price is not None and avg_price > 0:
                cap = int((max_exec_price / avg_price - 1.0) * 10000)
                slippage_bps = max(0, min(1200, cap))

            order_result = await self.place_order(quote, slippage_bps=slippage_bps)
            if not order_result:
                await self._update_signal_order(pending, "FAILED",
                                                error_message="下单失败", quote_json=quote)
                return pending

            await self._update_signal_order(
                pending, "FILLED",
                token_id=token_id,
                amount_in=str(quote.get("amountIn", "")),
                amount_out=str(quote.get("amountOut", "")),
                order_id=order_result.get("orderId"),
                quote_json=quote,
            )
            logger.info("信号实盘成交 | signal={} | window={} | orderId={} | slippageBps={}",
                        signal_version, window_start, order_result.get("orderId"), slippage_bps)
            return pending

    async def _reserve_order_slot(
        self, signal_version: str, window_start: int,
    ) -> TradeOrderModel | None:
        """先占位后下单（CodeReview High#1）：place_order 前先插 PENDING 行。

        占住 (signal_version, window_start) 唯一键，令每窗一单在花钱前生效：
        重复窗口（重启/并发）捕获 IntegrityError 返回 None，调用方放弃下单。
        """
        from sqlalchemy.exc import IntegrityError
        try:
            async with async_session_factory() as db:
                order = TradeOrderModel(
                    prediction_id=None,
                    market_id=self._active_market.get("marketTopicId") if self._active_market else None,
                    token_id="",
                    side="BUY",
                    amount_in="0",
                    status="PENDING",
                    signal_version=signal_version,
                    window_start=window_start,
                )
                db.add(order)
                await db.commit()
                return order
        except IntegrityError:
            return None
        except Exception as e:
            logger.error("信号实盘：订单占位失败（保守放弃本窗）| window {} | {}",
                         window_start, e)
            return None

    async def _update_signal_order(
        self,
        order: TradeOrderModel,
        status: str,
        *,
        token_id: str | None = None,
        amount_in: str | None = None,
        amount_out: str | None = None,
        order_id: str | None = None,
        quote_json: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """把占位订单更新为终态（FILLED/FAILED）。

        交易所已成交而此处更新失败是最坏路径（行卡在 PENDING）：记 CRITICAL
        供日志健康检查报警，禁止静默（CodeReview High#1 修复配套）。
        """
        try:
            async with async_session_factory() as db:
                order.status = status
                if token_id is not None:
                    order.token_id = token_id
                if amount_in is not None:
                    order.amount_in = amount_in
                if amount_out is not None:
                    order.amount_out = amount_out
                if order_id is not None:
                    order.order_id = order_id
                if quote_json is not None:
                    order.quote_json = quote_json
                if error_message is not None:
                    order.error_message = error_message
                db.add(order)
                await db.commit()
        except Exception as e:
            logger.critical(
                "信号实盘：订单终态落库失败 | window={} | status={} | order_id={} | {}",
                order.window_start, status, order_id, e)

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
        signal_version: str | None = None,
        window_start: int | None = None,
    ) -> TradeOrderModel | None:
        """
        保存订单到数据库

        agent_prediction_id 写入 trade_orders.agent_prediction_id，用于与 Agent 预测
        双向关联（旧 prediction_id 路径不传该值，行为保持不变）；
        error_message 仅在失败落库（status=FAILED）时写入，保证失败可追溯（规则 3，无静默降级）。
        signal_version/window_start 为报价 edge 实盘关联字段，旧路径不传保持 NULL。
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
                    signal_version=signal_version,
                    window_start=window_start,
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
        signal_version: str | None = None,
        window_start: int | None = None,
    ) -> TradeOrderModel | None:
        """
        保存失败订单到数据库

        落库 status=FAILED + error_message，不伪造成交（规则 3，无静默降级）；
        同样透传 agent_prediction_id，使失败订单亦可与 Agent 预测双向关联。
        signal_version/window_start 供报价 edge 实盘失败（含护栏弃单）追溯。
        """
        logger.warning(
            "交易失败 | prediction_id={} | agent_prediction_id={} | signal={} | error={}",
            prediction_id, agent_prediction_id, signal_version, error_msg,
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
            signal_version=signal_version,
            window_start=window_start,
        )
