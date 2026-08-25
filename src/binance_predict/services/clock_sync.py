"""
共享时钟同步模块（Fix #7）

多个服务（PredictionMarketDataService / BinancePredictionTrader）此前各自持有
独立的 `_time_offset_ms` 并分别调用 `/api/v3/time` 校准，存在以下问题：

1. 重复网络请求，浪费启动时间
2. 两个服务的偏差值可能不一致，导致签名时间戳出现细微差异
3. 无统一刷新机制，长时间运行后偏差可能漂移

本模块提供进程级单例的时钟偏差，所有需要 Binance 签名时间戳的服务共享同一来源。
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger

_BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"

# 进程级共享时钟偏差（毫秒）
_time_offset_ms: int = 0
_sync_lock = asyncio.Lock()
_last_sync_ts: float = 0.0


async def sync_server_time(*, force: bool = False, min_interval_sec: float = 60.0) -> int:
    """与 Binance 服务器校准时钟偏差（进程级共享）。

    Args:
        force: 为 True 时忽略最小间隔限制强制校准。
        min_interval_sec: 两次校准的最小间隔，避免频繁请求。

    Returns:
        当前时钟偏差（毫秒）。校准失败时保留上一次的有效值。
    """
    global _time_offset_ms, _last_sync_ts

    async with _sync_lock:
        now = time.time()
        if not force and (now - _last_sync_ts) < min_interval_sec:
            return _time_offset_ms

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                before = int(time.time() * 1000)
                resp = await client.get(_BINANCE_TIME_URL)
                after = int(time.time() * 1000)
                server_time = resp.json()["serverTime"]
                local_mid = (before + after) // 2
                _time_offset_ms = server_time - local_mid
                _last_sync_ts = now
                logger.info(
                    "共享时钟校准完成 | 偏差={}ms | local_mid={} | server={}",
                    _time_offset_ms, local_mid, server_time,
                )
        except Exception as e:
            logger.warning("共享时钟校准失败（沿用上次偏差 {}ms）: {}", _time_offset_ms, e)

        return _time_offset_ms


def get_offset_ms() -> int:
    """返回当前共享时钟偏差（毫秒）。"""
    return _time_offset_ms


def now_ms() -> int:
    """返回补偿了服务器偏差的当前毫秒时间戳，供签名使用。"""
    return int(time.time() * 1000) + _time_offset_ms
