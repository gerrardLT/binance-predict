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
import math
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from loguru import logger

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import TradeOrderModel
from . import clock_sync

# ---------------------------------------------------------------------------
# 预测钱包余额常量（P0-1：余额看不全；2026-08-23 agent-reach 调研收敛）
# ① wallet/list 响应内嵌余额字段候选 key（零额外 API 成本）；
# ② 官方文档确认的余额端点：balance/payment-options（旧 asset/list 实测 404
#    已废弃；返回结构未知，解析兼容多种包裹形态，响应原文打日志供收敛）。
# ---------------------------------------------------------------------------
_WALLET_BALANCE_KEYS: tuple[str, ...] = (
    "balance", "usdtBalance", "availableBalance", "available",
    "usdtFree", "free", "assetBalance",
)
_ASSET_LIST_PATH = "/sapi/v1/w3w/wallet/prediction/balance/payment-options"

# 预测钱包划转端点（⚠️ 官方命名反转陷阱：
#   outbound = CEX 现货→预测钱包（入金，transfer_in 用）
#   inbound  = 预测钱包→CEX 现货（提走，transfer_out 用））
_TRANSFER_OUTBOUND_PATH = "/sapi/v1/w3w/wallet/prediction/transfer/outbound"
_TRANSFER_INBOUND_PATH = "/sapi/v1/w3w/wallet/prediction/transfer/inbound"

# 奖金领取（2026-08-23：官方 agentic wallet 文档确认：PENDING_CLAIM 查可领 token
# → batch-redeem 领取；响应结构未实测，防御式解析 + 原文透传收敛）
_POSITION_LIST_PATH = "/sapi/v1/w3w/wallet/prediction/position/list"
_BATCH_REDEEM_PATH = "/sapi/v1/w3w/wallet/prediction/batch-redeem"


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
        # 最近一次 payment-options 余额查询错误（诊断透传：前端"暂不可查"时定位原因）
        self.last_assets_error: str | None = None
        # 最近一次 payment-options 成功响应的原文前 500 字符（诊断透传：解析未命中时收敛字段）
        self.last_assets_raw: str | None = None
        # 可领持仓（PENDING_CLAIM）查询诊断：错误 + 成功响应原文
        self.last_pending_error: str | None = None
        self.last_pending_raw: str | None = None

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
        # 多周期 15m 市场表：startDate(ms) → {up/down_token, up/down_price, end_date}。
        # API 同时列出当前周期 + 未来数小时的多个 15m 市场，下单必须按
        # startDate 精确匹配信号窗口——旧"最后一个覆盖"单值缓存会拿到未来
        # 市场的 token，场景单必押错周期（审计发现）
        self._15m_markets: dict[int, dict] = {}

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

    async def fetch_prediction_wallet_balance(self, wallet: dict | None = None) -> dict:
        """查询预测钱包 USDT 余额 + 资产列表（P0-1：余额看不全）。

        两级探索（不猜参数，失败降级 None 不阻塞）：
        ① 优先从 wallet/list 响应内嵌的候选字段取余额（_WALLET_BALANCE_KEYS，
           零额外 API 成本）；
        ② 资产列表候选端点 _ASSET_LIST_PATH（同时返回 outcome token 持仓，
           P2-2② 复用）——实测后改模块常量一处即可。

        返回 {"usdt_free": float | None, "assets": list | None}；
        全部失败时两字段均为 None（探索型端点天然 fallback 语义）。
        """
        out: dict = {"usdt_free": None, "assets": None}
        if wallet is None:
            wallet = await self.fetch_wallet_info()

        # ① wallet/list 内嵌余额字段
        if wallet:
            for key in _WALLET_BALANCE_KEYS:
                v = wallet.get(key)
                if v is None:
                    continue
                try:
                    out["usdt_free"] = float(v)
                    break
                except (TypeError, ValueError):
                    continue

        # ② 官方余额端点（payment-options；失败降级 None）
        assets = await self._fetch_prediction_asset_list()
        if assets is not None:
            out["assets"] = assets
            # ①未命中时兑底：官方形态 items[].accountType=CeDeFi（预测钱包，
            # 生产实测 2026-08-23）或旧形态 asset/symbol==USDT 条目
            if out["usdt_free"] is None:
                for a in assets:
                    if not isinstance(a, dict):
                        continue
                    if str(a.get("accountType", "")).upper() == "CEDEFI":
                        if a.get("enabled") is False:
                            continue
                        v = a.get("availableBalanceDisplay")
                        try:
                            out["usdt_free"] = float(v)
                            break
                        except (TypeError, ValueError):
                            continue
                    symbol = str(a.get("asset") or a.get("symbol") or "").upper()
                    if symbol != "USDT":
                        continue
                    free = a.get("free", a.get("balance", a.get("amount")))
                    if free is None:
                        continue
                    try:
                        out["usdt_free"] = float(free)
                    except (TypeError, ValueError):
                        pass
                    break
        return out

    async def _fetch_prediction_asset_list(self) -> list | None:
        """预测钱包余额/资产列表（官方 balance/payment-options 端点）。

        失败返回 None + last_api_error（上层降级不阻塞其余字段）。
        Fix -1022（2026-08-23 生产实锤 assets_api_error）：_sign_request 按
        字母序签名，而 httpx params= 按 dict 插入序发送；带 walletId/
        walletAddress 时两序不同，币安按收到的 query string 原文验签 →
        -1022。改用 _build_signed_url 手动拼 URL（签名串=发送串，
        与下单端点修 -1022 同款方案）。
        响应结构未知：兼容 list 直返 / {"options"/"assets"/...} 包裹形态，
        并把响应原文前 300 字符打进日志（生产实测后收敛解析字段）。
        """
        req_params: dict = {}
        if self._wallet_id:
            req_params["walletId"] = self._wallet_id
        if self._wallet_address:
            req_params["walletAddress"] = self._wallet_address
        url = self._build_signed_url(_ASSET_LIST_PATH, req_params)

        try:
            client = self._get_client()
            resp = await client.get(
                url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            self.last_assets_error = self.last_api_error
            logger.info(
                "预测钱包资产列表查询失败（探索型端点，降级） | HTTP {} | {}",
                e.response.status_code, e.response.text[:200],
            )
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            self.last_assets_error = self.last_api_error
            logger.info("预测钱包资产列表查询异常（探索型端点，降级）: {}", e)
            return None

        # 响应结构未知：兼容 list 直返 / {"options": [...]} 等包裹形态
        self.last_assets_error = None
        self.last_assets_raw = str(data)[:500]
        logger.info(
            "预测钱包 payment-options 原始响应（收敛用）：{}",
            str(data)[:300],
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # 生产实测（2026-08-23）：{"items": [{accountType/availableBalanceDisplay/enabled}...]}
            for key in ("items", "options", "assets", "balances", "data", "list"):
                v = data.get(key)
                if isinstance(v, list):
                    return v
        return None

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
        即未划转），调用 POST /sapi/v1/w3w/wallet/prediction/transfer/outbound
        （注意：官方 outbound = CEX→预测钱包入金，inbound 方向相反）。
        失败返回 None，详情写入 last_api_error。
        """
        amount_wei = str(int(round(amount_usdt * 10**18)))
        # 全参数走 query（与 place-order 同口径，签名可过）；早期的 -9000
        # 实为 API Key 缺万向划转权限，非编码方式问题（form/JSON 报 -1022）。
        signed_url = self._build_signed_url(
            _TRANSFER_OUTBOUND_PATH,
            {
                "walletId": self._wallet_id,
                "walletAddress": self._wallet_address,
                "fromTokenAmount": amount_wei,
                "accountType": "SPOT",
                "sourceBiz": "USER_TRANSFER",
            },
        )

        try:
            client = self._get_client()
            resp = await client.post(
                signed_url,
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

    async def transfer_out(self, amount_usdt: float) -> dict | None:
        """从预测钱包划转 USDT 回现货账户（提走/回笼通道，P1-1）。

        ⚠️ 方向反转陷阱（与直觉相反，逐行镜像 transfer_in）：
        官方端点 inbound = 预测钱包→CEX 现货（提走），
        outbound = CEX 现货→预测钱包（入金，见 transfer_in）。
        调用 POST /sapi/v1/w3w/wallet/prediction/transfer/inbound。
        上层端点用划转前后现货余额差自证方向（金丝雀 0.1U 先行）。
        失败返回 None，详情写入 last_api_error。
        """
        amount_wei = str(int(round(amount_usdt * 10**18)))
        # 全参数走 query（与 transfer_in 同口径，签名可过）
        signed_url = self._build_signed_url(
            _TRANSFER_INBOUND_PATH,
            {
                "walletId": self._wallet_id,
                "walletAddress": self._wallet_address,
                "fromTokenAmount": amount_wei,
                "accountType": "SPOT",
                "sourceBiz": "USER_TRANSFER",
            },
        )

        try:
            client = self._get_client()
            resp = await client.post(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_api_error = None
            logger.info("预测钱包提走成功 | amount={} USDT | resp={}", amount_usdt, data)
            return data
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("预测钱包提走失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("预测钱包提走异常: {}", e)
            return None

    # ------------------------------------------------------------------
    # 奖金领取（batch-redeem）：赢单 token → USDT
    # ------------------------------------------------------------------

    async def fetch_pending_claim_positions(self, limit: int = 50) -> list | None:
        """查询可领取的获胜持仓（GET position/list?tab=PENDING_CLAIM）。

        官方 agentic wallet 文档：--tab PENDING_CLAIM 返回已结算、赢了但
        未赎回的 token 持仓。响应结构未实测：兼容 list 直返 /
        {positions|items|orders|data} 包裹，原文透传 last_pending_raw 供收敛。
        失败返回 None + last_pending_error（上层降级不阻塞）。
        """
        req_params: dict = {"tab": "PENDING_CLAIM", "limit": limit}
        if self._wallet_id:
            req_params["walletId"] = self._wallet_id
        if self._wallet_address:
            req_params["walletAddress"] = self._wallet_address
        url = self._build_signed_url(_POSITION_LIST_PATH, req_params)

        try:
            client = self._get_client()
            resp = await client.get(
                url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            self.last_pending_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.info(
                "可领持仓查询失败（探索型端点，降级） | HTTP {} | {}",
                e.response.status_code, e.response.text[:200],
            )
            return None
        except Exception as e:
            self.last_pending_error = f"{type(e).__name__}: {e}"
            logger.info("可领持仓查询异常（探索型端点，降级）: {}", e)
            return None

        self.last_pending_error = None
        self.last_pending_raw = str(data)[:500]
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("positions", "items", "orders", "data", "list"):
                v = data.get(key)
                if isinstance(v, list):
                    return v
        return None

    def _extract_token_ids(self, positions: list) -> list[str]:
        """从持仓列表提取 tokenId（兼容 tokenId/token_id 键，去重保序）。"""
        seen: set[str] = set()
        out: list[str] = []
        for p in positions:
            if not isinstance(p, dict):
                continue
            tid = p.get("tokenId") or p.get("token_id") or p.get("id")
            if isinstance(tid, str) and tid and tid not in seen:
                seen.add(tid)
                out.append(tid)
        return out

    async def redeem_tokens(self, token_ids: list[str]) -> dict | None:
        """领取获胜 token 奖金（POST batch-redeem，tokenIds 逗号拼接）。

        官方端点（文档确认）：在链上赎回已结算的预测 token。
        全参数走 query 签名（与 transfer 同口径，防 -1022）。
        失败返回 None，详情写入 last_api_error。
        """
        if not token_ids:
            self.last_api_error = "token_ids 为空（无可领取持仓）"
            return None
        signed_url = self._build_signed_url(
            _BATCH_REDEEM_PATH,
            {
                "walletId": self._wallet_id,
                "walletAddress": self._wallet_address,
                "tokenIds": ",".join(token_ids),
            },
        )

        try:
            client = self._get_client()
            resp = await client.post(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            self.last_api_error = None
            logger.info(
                "奖金领取成功 | tokens={} | resp={}",
                len(token_ids), str(data)[:300],
            )
            return data
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("奖金领取失败 (HTTP {}): {}", e.response.status_code, e.response.text)
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("奖金领取异常: {}", e)
            return None

    async def query_order_history(self, limit: int = 20) -> list | None:
        """查询预测钱包订单历史（GET order/history）。

        用途：本地落库卡 PENDING 时对账，确认币安侧是否真实成交（钱是否
        已花出）。失败返回 None，详情写入 last_api_error（参数口径待与
        官方文档核对，错误回显便于定位）。
        """
        self.last_api_error = None
        # 文档示例仅 walletAddress；全参数走 query 手动拼 URL（httpx params=
        # 的 URL 编码会导致签名串与实际 query 不一致，报 -1022）
        signed_url = self._build_signed_url(
            "/sapi/v1/w3w/wallet/prediction/order/history",
            {
                "walletAddress": self._wallet_address,
                "limit": limit,
            },
        )
        try:
            client = self._get_client()
            resp = await client.get(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("查询订单历史失败 (HTTP {}): {}", e.response.status_code, e.response.text[:300])
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("查询订单历史失败: {}", e)
            return None

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            orders = data.get("orders") or data.get("items")
            if isinstance(orders, list):
                return orders
        return [data]

    async def confirm_order_status(
        self, order_id: str | None, attempts: int = 3, delay: float = 1.0,
    ) -> dict | None:
        """回查刚下单订单的币安侧终态：命中订单历史行 dict / None。

        place-order-bundle 返回 200 + orderId 不代表成交：FOK 单受理后仍可能
        翻转为 FAILED（流动性不足等）。曾以"收到响应即成交"落库，产生幽灵成交
        （2026-08 对账发现 2 笔：本地 FILLED + win/pnl，币安侧 filled=0）。
        此处轮询订单历史以币安终态为准；None = 历史中暂无该单（调用方应落
        PENDING，交由 /api/trades/sync-binance 对账，不伪造成交）。

        2026-08-28 语义变更：返回值由终态 str 改为命中的历史行 dict（终态取
        row["status"]，另含 price/filledUsdtAmount 等），顺手把历史行的实际
        成交价带回落库侧（execute_trade/execute_signal_trade 回填 quote_json）。
        """
        if not order_id:
            return None
        for i in range(attempts):
            history = await self.query_order_history(limit=20)
            if history:
                for o in history:
                    if isinstance(o, dict) and order_id in (
                        o.get("orderId"), o.get("vendorOrderId"),
                    ):
                        status = o.get("status")
                        if status in ("FILLED", "FAILED"):
                            return o
            if i < attempts - 1:
                await asyncio.sleep(delay)
        return None

    @staticmethod
    def _merge_fill_into_quote(quote: dict, fill_row: dict) -> dict:
        """用币安订单历史行的实际成交数据增强 quote_json（落库口径升级，2026-08-28）。

        averagePrice ← 历史行 price（实际成交均价，币安结算口径）；原报价
        预估均价保留在 quotedAvgPrice，fillSource 标记来源。落库后
        trade_settler._avg_price 读 quote_json.averagePrice 即自动升级为
        实际成交口径（结算/统计无需改动）。price 缺失/非法/≤0 时原样返回
        （无可靠成交价时不做半吊子回填，保持报价预估口径）。
        2026-08-28 补丁：amountIn ← filledUsdtAmount（实际成交金额）——
        FOK 只吃到部分深度时实际成交 < 请求金额（生产实锤：请求 3U 只成交 2U，
        本地仍按 3U 估算赢向盈亏虚高 55%）；原请求金额保留在 requestedAmountIn。
        """
        try:
            fill_price = float(fill_row.get("price"))
        except (TypeError, ValueError):
            return quote
        if fill_price <= 0:
            return quote
        merged = dict(quote)
        merged["quotedAvgPrice"] = quote.get("averagePrice")
        merged["averagePrice"] = fill_price
        merged["fillSource"] = "binance_history_confirm"
        try:
            filled_usdt = Decimal(str(fill_row.get("filledUsdtAmount")))
        except (InvalidOperation, TypeError, ValueError):
            filled_usdt = Decimal(-1)
        if filled_usdt > 0:
            merged["requestedAmountIn"] = quote.get("amountIn")
            merged["amountIn"] = str(int(filled_usdt * (10 ** 18)))
        return merged

    @staticmethod
    def _classify_period(market: dict) -> str | None:
        """按 title/slug 识别市场周期：'5m' | '15m' | None。先判 15m（含 '5m' 子串）。"""
        text = f"{(market.get('title') or '').lower()} {(market.get('slug') or '').lower()}"
        if "15m" in text:
            return "15m"
        if "5m" in text:
            return "5m"
        return None

    @staticmethod
    def _parse_15m_entry(market: dict) -> tuple[int, dict] | None:
        """解析单个 15m 市场 → (startDate, {token/报价/end_date})；startDate 缺失/非法返回 None。

        单位与 fake_breakout 的 market_start_15m 一致（epoch ms，检测器
        已用 start_date == next_start 守卫在生产验证）。下单按 startDate 精确
        匹配，防多周期列表下押错市场（审计发现）。
        """
        start_raw = market.get("startDate")
        end_raw = market.get("endDate")
        try:
            start_ms = int(start_raw)
        except (TypeError, ValueError):
            return None
        try:
            end_ms = int(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end_ms = None
        entry = {"end_date": end_ms, "up_token": None, "down_token": None,
                 "up_price": None, "down_price": None}
        for sub_market in market.get("markets", []):
            for outcome in sub_market.get("outcomes", []):
                name = outcome.get("name", "").upper()
                token_id = outcome.get("tokenId")
                price = outcome.get("price")
                if name in ("UP", "YES"):
                    entry["up_token"] = token_id
                    entry["up_price"] = float(price) if price is not None else None
                elif name in ("DOWN", "NO"):
                    entry["down_token"] = token_id
                    entry["down_price"] = float(price) if price is not None else None
        return start_ms, entry

    async def _fetch_market_via_detail(
        self, period: str, start_ms: int
    ) -> dict | None:
        """
        list 端点兜底：按 marketTopicId 锚点 + market/detail 向前扫描预取市场。

        背景（2026-08-28 实测修正）：market/list 只返回已开盘市场，而币安
        未来周期市场提前创建且可直接交易（15m 提前约 22 分钟、5m 提前 30+
        分钟，tradingStatus=OPEN）。开火点在边界后 0~2s 时新周期尚未进入
        list，锚定守卫拒单——本方法经 detail 端点拿回未来市场条目。

        定位方式：list 首页取当前周期 BTC 市场的 marketTopicId 作锚点，
        相邻 ID 向后扫描（相邻 ID 交错 BTC/ETH/BNB 与 5m/15m），按可预测
        slug `btc-updown-{period}-{epoch}` 精确匹配；命中即解析返回（同
        _parse_15m_entry 结构），并要求 tradingStatus=OPEN。
        """
        slug_want = f"btc-updown-{period}-{start_ms // 1000}"
        client = self._get_client()

        async def _detail(topic_id: int) -> dict | None:
            try:
                signed = self._sign_request({"marketTopicId": topic_id})
                resp = await client.get(
                    f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/market/detail",
                    params=signed,
                    headers={"X-MBX-APIKEY": self._api_key},
                )
                return resp.json() if resp.status_code == 200 else None
            except Exception:
                return None

        # 锚点：list 首页找当前已开盘的同周期 BTC 市场
        anchor = None
        try:
            signed = self._sign_request({"limit": 100, "offset": 0})
            resp = await client.get(
                f"{self.BASE_URL}/sapi/v1/w3w/wallet/prediction/market/list",
                params=signed,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            for m in resp.json().get("marketTopics", []):
                if (m.get("chartType") == "CRYPTO_UP_DOWN"
                        and m.get("symbol") == "BTCUSDT"
                        and self._classify_period(m) == period):
                    anchor = m.get("marketTopicId")
                    break
        except Exception as e:
            logger.warning("detail 预取：锚点查询失败 | {}", e)
        if anchor is None:
            return None

        for tid in range(anchor + 1, anchor + 48):
            d = await _detail(tid)
            if not d or d.get("slug") != slug_want:
                continue
            subs = d.get("markets", [])
            trading = subs[0].get("tradingStatus") if subs else None
            if trading != "OPEN":
                logger.warning("detail 预取：命中但不可交易 | tid={} trading={}",
                               tid, trading)
                return None
            parsed = self._parse_15m_entry(d)
            if parsed is None:
                return None
            entry_start, entry = parsed
            if abs(entry_start - start_ms) > 2000:
                logger.warning("detail 预取：slug 命中但 startDate 偏差 {}ms | tid={}",
                               entry_start - start_ms, tid)
                return None
            logger.info("detail 预取成功 | {} | topicId={} | 距开盘 {:.0f}s",
                        period, tid, (entry_start - int(time.time() * 1000)) / 1000)
            return entry
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
                    # 15 分钟市场：API 同时列出多个周期（当前 + 未来数小时），
                    # 逐周期按 startDate 建表，下单时精确匹配（见 _15m_markets 注释）
                    parsed = self._parse_15m_entry(market)
                    if parsed is None:
                        continue  # startDate 缺失/非法：无法锚定，不入表
                    start_ms, entry = parsed
                    self._15m_markets[start_ms] = entry
                    # 标量缓存（展示/日志用）只认当前周期 start ≤ now < end，
                    # 不再被列表里最后一个（往往是未来）市场覆盖
                    now_ms = int(time.time() * 1000)
                    end_ms = entry["end_date"]
                    if end_ms is not None and start_ms <= now_ms < end_ms:
                        self._15m_start_date = start_ms
                        self._15m_end_date = end_ms
                        self._15m_up_token_id = entry["up_token"]
                        self._15m_down_token_id = entry["down_token"]
                        self._15m_up_price = entry["up_price"]
                        self._15m_down_price = entry["down_price"]
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

        # 清退已结束的 15m 周期（防缓存随运行时间无界增长）
        now_ms = int(time.time() * 1000)
        self._15m_markets = {
            s: e for s, e in self._15m_markets.items()
            if e["end_date"] is None or e["end_date"] > now_ms
        }

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
            # 防御：成功响应缺 quoteId 时后续 place_order 会 KeyError 报 500，
            # 提前拦下并透传响应内容供诊断（响应结构变更可及时发现）。
            if not quote.get("quoteId"):
                self.last_api_error = f"报价响应缺少 quoteId | resp={str(quote)[:300]}"
                logger.error("获取报价异常：响应缺少 quoteId | resp={}", quote)
                return None
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
                    direction=prediction,
                )

            # 1. 获取报价
            quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt)
            if not quote:
                return await self._save_failed_order(
                    prediction_id, "获取报价失败",
                    agent_prediction_id=agent_prediction_id,
                    direction=prediction,
                )

            # 2. 下单
            order_result = await self.place_order(quote)
            if not order_result:
                return await self._save_failed_order(
                    prediction_id, "下单失败", quote=quote,
                    agent_prediction_id=agent_prediction_id,
                    direction=prediction,
                )

            # 3. 币安侧终态回查（FOK 受理后仍可能翻转 FAILED；防幽灵成交）
            eff_window_start = window_start if window_start is not None else (
                int(time.time() * 1000) // 300_000 * 300_000)
            order_id = order_result.get("orderId")
            confirmed = await self.confirm_order_status(order_id)
            if confirmed is not None and confirmed.get("status") == "FAILED":
                # 币安侧订单未成交：落 FAILED，不伪造成交（规则 3）
                return await self._save_order(
                    prediction_id=prediction_id,
                    agent_prediction_id=agent_prediction_id,
                    token_id=token_id,
                    side="BUY",
                    amount_in="0",
                    amount_out=None,
                    order_id=order_id,
                    status="FAILED",
                    quote_json=quote,
                    error_message="币安侧订单终态 FAILED（FOK 未成交）",
                    signal_version=signal_version,
                    window_start=eff_window_start,
                    direction=prediction,
                )
            # FILLED → 落成交（quote_json 回填实际成交均价，2026-08-28）；
            # 暂未确认（历史尚无该单）→ PENDING 交对账兜底
            status = ("FILLED" if confirmed is not None
                      and confirmed.get("status") == "FILLED" else "PENDING")
            if status == "PENDING":
                logger.warning("币安侧终态暂未确认，落 PENDING 待对账 | orderId={}", order_id)
            else:
                quote = self._merge_fill_into_quote(quote, confirmed)
            return await self._save_order(
                prediction_id=prediction_id,
                agent_prediction_id=agent_prediction_id,
                token_id=token_id,
                side="BUY",
                amount_in=str(quote.get("amountIn", "")),
                amount_out=str(quote.get("amountOut", "")),
                order_id=order_id,
                status=status,
                quote_json=quote,
                signal_version=signal_version,
                window_start=eff_window_start,
                direction=prediction,
            )

    async def execute_signal_trade(
        self,
        prediction: str,
        amount_usdt: float,
        signal_version: str,
        window_start: int,
        max_exec_price: float | None = None,
        market_period: str = "5m",
        scene_signal_id: int | None = None,
    ) -> dict | None:
        """
        信号驱动实盘专用通道（多通道 LIVE：quote_edge/x4 用 5m，场景用 15m）：
        报价 → 执行价护栏 → 下单 → 落表。

        返回 dict 快照（非 ORM 对象）：会话关闭后对象即脱离，调用方再访问
        属性会抛 DetachedInstanceError（旧实现导致端点 500 + 行卡 PENDING）。

        与 execute_trade 的差异：
        1. 金额/信号版本/窗口由调用方显式传入（非配置默认）；
        2. **先占位后下单**（CodeReview High#1）：place_order 前先插 PENDING 行占住
           (signal_version, window_start) 唯一键，重复窗口在下单前即拒绝；
           成功/失败 UPDATE 该行——重启防重不再依赖"钱出去后"的事后提交；
        3. 报价后、下单前检查 averagePrice ≤ max_exec_price，超限弃单（不追贵）；
           且按护栏价动态收紧 slippageBps，成交价无法突破 max_exec_price；
        4. market_period 分流（'5m'|'15m'）：15m 场景订单用 15m 市场 token
           （结算走 FakeBreakoutSignal，trade_settler 按 market_period 分流）；
           scene_signal_id 与占位同事务落库（下单即关联场景信号行，无需回填）。
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

        # 纯入参校验前置：非法周期在占位/拉行情之前拒绝——否则会烧掉
        # (signal_version, window_start) 窗口槽位，且在持有全局 trade_lock 期间
        # 白白消耗一次分页行情拉取、阻塞其它通道下单
        if market_period not in ("5m", "15m"):
            logger.warning("信号实盘：未知市场周期 {} | signal={}", market_period,
                           signal_version)
            return None

        async with self._trade_lock:
            # 先占位后下单：PENDING 行占住唯一键；重复窗口（含重启/并发）在花钱前拒绝。
            pending = await self._reserve_order_slot(
                signal_version, window_start, direction=prediction,
                market_period=market_period, scene_signal_id=scene_signal_id)
            if pending is None:
                logger.info("信号实盘：窗口 {} 已有订单占位，跳过（每窗一单）", window_start)
                return None

            await self.list_markets()

            if prediction not in ("UP", "DOWN"):
                logger.warning("未知预测方向: {}", prediction)
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction,
                    error_message=f"未知预测方向: {prediction}")

            if market_period == "15m":
                # 锚定校验：列表含多个 15m 周期，必须取本单窗口
                # （window_start == startDate）的 token；缺失即拒单——
                # 宁可错过不押错周期（旧"最后一个覆盖"行为会拿未来市场）
                entry = self._15m_markets.get(window_start)
                
                # 容差匹配：允许 ±2 秒偏差（应对 K 线对齐 vs 市场创建时间的微小差异）
                if entry is None:
                    tolerance_ms = 2 * 1000  # 2 秒，而不是之前的 2 分钟
                    candidates = []
                    for start_ms, e in self._15m_markets.items():
                        diff = abs(start_ms - window_start)
                        if diff <= tolerance_ms:
                            candidates.append((diff, start_ms, e))
                    if candidates:
                        # 选最接近的一个
                        candidates.sort(key=lambda x: x[0])
                        _, best_start_ms, best_entry = candidates[0]
                        entry = best_entry
                        logger.info(
                            "信号实盘：15m 周期 {} 容差匹配到 {} | 偏差 {}ms",
                            window_start, best_start_ms, abs(best_start_ms - window_start))
                    else:
                        # 详细日志：打印所有可用周期
                        logger.warning(
                            "信号实盘：15m 周期 {} 未在市场列表中找到（锚定守卫拒单）\n"
                            "   ├─ 可用周期 (start_date): {}\n"
                            "   └─ 最近偏差：{} ms",
                            window_start,
                            sorted(self._15m_markets.keys()),
                            min(abs(s - window_start) for s in self._15m_markets.keys()) if self._15m_markets else 0
                        )
                        # list 只返回已开盘市场：新周期边界刚过时尚未可见。
                        # 经 market/detail 预取未来市场（提前创建且可交易，2026-08-28 实测）
                        entry = await self._fetch_market_via_detail("15m", window_start)
                        if entry:
                            self._15m_markets[window_start] = entry
                
                token_id = (entry["up_token"] if prediction == "UP"
                            else entry["down_token"]) if entry else None
                if entry is None:
                    logger.warning(
                        "信号实盘：15m 周期 {} 未在市场列表中找到（锚定守卫拒单，含 detail 预取）| 列表周期 {}",
                        window_start, sorted(self._15m_markets))
            else:
                # 5m 锚定：缓存 token 属于当前可见 5m 市场，开火瞬间新周期
                # 可能尚未进入 list——startDate 不匹配本单窗口时同样走
                # detail 预取，防押到上一周期（旧行为无锚定直接取缓存）
                entry = None
                cache_start = self._5m_start_date
                try:
                    cache_start = int(cache_start) if cache_start else None
                except (TypeError, ValueError):
                    cache_start = None
                if cache_start is not None and abs(cache_start - window_start) <= 2000:
                    token_id = (self._up_token_id if prediction == "UP"
                                else self._down_token_id)
                else:
                    entry = await self._fetch_market_via_detail("5m", window_start)
                    token_id = (entry["up_token"] if prediction == "UP"
                                else entry["down_token"]) if entry else None
                    if entry is None:
                        logger.warning(
                            "信号实盘：5m 窗口 {} 锚定失配且 detail 预取失败 | 缓存周期 {}",
                            window_start, cache_start)

            if not token_id:
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction,
                    error_message=f"未找到 {market_period} 市场 {prediction} 方向的 token")

            quote = await self.get_quote(token_id, "BUY", amount_usdt=amount_usdt)
            if not quote:
                detail = self.last_api_error or "无详情（网络异常？）"
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction,
                    error_message=f"获取报价失败 | {detail}",
                )

            # 执行价护栏：报价均价超阈弃单（不追贵，保护回测 EV 口径）
            try:
                avg_price = float(quote.get("averagePrice") or 0.0)
            except (TypeError, ValueError):
                avg_price = 0.0
            if max_exec_price is not None and (avg_price <= 0 or avg_price > max_exec_price):
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction,
                    error_message=f"执行价护栏弃单 | averagePrice={avg_price} > {max_exec_price}",
                    quote_json=quote)

            # 动态滑点收紧（CodeReview Medium#2）：FOK 成交价不得突破护栏价，
            # 否则 slippageBps=1200 会让 0.78 的护栏形同虚设（最高可成交 ~0.87）。
            slippage_bps = 1200
            if max_exec_price is not None and avg_price > 0:
                cap = int((max_exec_price / avg_price - 1.0) * 10000)
                slippage_bps = max(0, min(1200, cap))

            order_result = await self.place_order(quote, slippage_bps=slippage_bps)
            if not order_result:
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction,
                    error_message="下单失败", quote_json=quote)

            # 币安侧终态回查（FOK 受理后仍可能翻转 FAILED；防幽灵成交，
            # 2026-08 对账发现 2 笔本地 FILLED 而币安 filled=0）
            order_id = order_result.get("orderId")
            confirmed = await self.confirm_order_status(order_id)
            if confirmed is not None and confirmed.get("status") == "FAILED":
                return await self._update_signal_order(
                    pending, "FAILED", direction=prediction, token_id=token_id,
                    order_id=order_id, quote_json=quote,
                    error_message="币安侧订单终态 FAILED（FOK 未成交）")

            # FILLED → 落成交（quote_json 回填实际成交均价，2026-08-28）；
            # 暂未确认 → 保持 PENDING 交 sync-binance 对账
            status = ("FILLED" if confirmed is not None
                      and confirmed.get("status") == "FILLED" else "PENDING")
            if status == "FILLED":
                quote = self._merge_fill_into_quote(quote, confirmed)
            snapshot = await self._update_signal_order(
                pending, status,
                direction=prediction,
                token_id=token_id,
                amount_in=str(quote.get("amountIn", "")),
                amount_out=str(quote.get("amountOut", "")),
                order_id=order_id,
                quote_json=quote,
            )
            if status == "PENDING":
                logger.warning("信号实盘：币安侧终态暂未确认，落 PENDING 待对账 | signal={} | window={} | orderId={}",
                               signal_version, window_start, order_id)
            else:
                logger.info("信号实盘成交 | signal={} | window={} | orderId={} | slippageBps={} | 成交价 {}（fillSource={}）",
                            signal_version, window_start, order_id, slippage_bps,
                            quote.get("averagePrice"), quote.get("fillSource", "quote_estimate"))
            return snapshot

    async def _reserve_order_slot(
        self, signal_version: str, window_start: int,
        direction: str | None = None,
        market_period: str = "5m",
        scene_signal_id: int | None = None,
    ) -> TradeOrderModel | None:
        """先占位后下单（CodeReview High#1）：place_order 前先插 PENDING 行。

        占住 (signal_version, window_start) 唯一键，令每窗一单在花钱前生效：
        重复窗口（重启/并发）捕获 IntegrityError 返回 None，调用方放弃下单。
        direction 占位即落库（结算判赢依赖；NULL 仅限旧数据）；
        market_period/scene_signal_id 同事务落库（15m 结算分流依据）。
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
                    direction=direction,
                    market_period=market_period,
                    scene_signal_id=scene_signal_id,
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
        direction: str | None = None,
        token_id: str | None = None,
        amount_in: str | None = None,
        amount_out: str | None = None,
        order_id: str | None = None,
        quote_json: dict | None = None,
        error_message: str | None = None,
    ) -> dict:
        """把占位订单更新为终态（FILLED/FAILED），返回 dict 快照。

        交易所已成交而此处更新失败是最坏路径（行卡在 PENDING）：记 CRITICAL
        供日志健康检查报警，禁止静默（CodeReview High#1 修复配套）。

        Fix（2026-08-23）：占位对象来自已关闭会话（脱离态），旧实现 db.add()
        会当 INSERT 撞唯一键静默失败（行卡 PENDING）且调用方再访问属性报
        DetachedInstanceError（端点 500）；改 merge() 按主键 UPDATE 并返回
        dict 快照，调用方不再触碰 ORM 对象。
        """
        try:
            async with async_session_factory() as db:
                order.status = status
                if direction is not None:
                    order.direction = direction
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
                merged = await db.merge(order)
                await db.commit()
                return {
                    "id": merged.id,
                    "status": merged.status,
                    "signal_version": merged.signal_version,
                    "window_start": merged.window_start,
                    "direction": merged.direction,
                    "token_id": merged.token_id,
                    "order_id": merged.order_id,
                    "amount_in": merged.amount_in,
                    "average_price": (merged.quote_json or {}).get("averagePrice"),
                    "error_message": merged.error_message,
                    "persisted": True,  # 真实落库成功（False=只返回快照未落库）
                }
        except Exception as e:
            logger.critical(
                "信号实盘：订单终态落库失败 | window={} | status={} | order_id={} | {}",
                order.window_start, status, order_id, e)
            return {
                "id": getattr(order, "id", None),
                "status": status,
                "signal_version": getattr(order, "signal_version", None),
                "window_start": getattr(order, "window_start", None),
                "direction": direction if direction is not None else getattr(order, "direction", None),
                "token_id": token_id,
                "order_id": order_id,
                "amount_in": amount_in,
                "average_price": (quote_json or {}).get("averagePrice"),
                "error_message": error_message,
                "persisted": False,  # 兜底快照：DB 未更新（行卡 PENDING）
            }

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
        direction: str | None = None,
    ) -> TradeOrderModel | None:
        """
        保存订单到数据库

        agent_prediction_id 写入 trade_orders.agent_prediction_id，用于与 Agent 预测
        双向关联（旧 prediction_id 路径不传该值，行为保持不变）；
        error_message 仅在失败落库（status=FAILED）时写入，保证失败可追溯（规则 3，无静默降级）。
        signal_version/window_start 为报价 edge 实盘关联字段，旧路径不传保持 NULL。
        direction 下单方向（UP/DOWN），结算判赢依赖（P0-2）。
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
                    direction=direction,
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
        direction: str | None = None,
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
            direction=direction,
        )

    # ------------------------------------------------------------------
    # 订单核对功能（对账/验证）
    # ------------------------------------------------------------------

    async def verify_order_by_id(self, order_id: str) -> dict | None:
        """通过订单 ID 查询单笔订单详情（用于对账）。

        Args:
            order_id: 币安返回的 orderId

        Returns:
            {
                'order_id': str,
                'token_id': str,
                'side': 'BUY' | 'SELL',
                'status': 'EXPIRED' | 'PENDING' | 'FILLED' | 'CANCELLED',
                'average_price': float | None,
                'amount_in': str | None,
                'created_at': int (epoch ms),
                'verified_at': datetime (UTC),
                'raw_data': dict
            }
        """
        self.last_api_error = None
        signed_url = self._build_signed_url(
            "/sapi/v1/w3w/wallet/prediction/order/query-by-orderId",
            {
                "walletAddress": self._wallet_address,
                "orderId": order_id,
            },
        )
        try:
            client = self._get_client()
            resp = await client.get(
                signed_url,
                headers={"X-MBX-APIKEY": self._api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            self.last_api_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
            logger.error("查询订单详情失败 (HTTP {}): {}", e.response.status_code, e.response.text[:300])
            return None
        except Exception as e:
            self.last_api_error = f"{type(e).__name__}: {e}"
            logger.error("查询订单详情失败：{}", e)
            return None

        if isinstance(data, dict):
            return {
                'order_id': data.get('orderId'),
                'token_id': data.get('tokenId'),
                'side': data.get('side'),
                'status': data.get('status'),
                'average_price': data.get('averagePrice'),
                'amount_in': data.get('amountIn'),
                'created_at': data.get('createTime'),
                'verified_at': int(time.time() * 1000),
                'raw_data': data,
            }
        return None
