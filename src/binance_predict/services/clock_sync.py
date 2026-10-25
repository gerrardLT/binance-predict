"""
共享时钟同步模块（Fix #7）

多个服务（PredictionMarketDataService / BinancePredictionTrader）此前各自持有
独立的 `_time_offset_ms` 并分别调用 `/api/v3/time` 校准，存在以下问题：

1. 重复网络请求，浪费启动时间
2. 两个服务的偏差值可能不一致，导致签名时间戳出现细微差异
3. 无统一刷新机制，长时间运行后偏差可能漂移

本模块提供进程级单例的时钟偏差，所有需要 Binance 签名时间戳的服务共享同一来源。

2026-10-25 生产事故加固（-1021 全天拒收，实盘订单全部「未找到 token」）：
- 币安对「时间戳超前服务器 >1000ms」无条件拒收（-1021）；recvWindow 只保护
  「落后」方向。当日启动校准测得 +1173ms 后全天无人刷新，VPS 时钟向前追赶，
  本地 + 陈旧偏差反超服务器 1s+，所有签名请求（market/list 等）全灭。
- 三层防御：
  1) signed_timestamp_ms()：签名时间戳整体后移 _AHEAD_SAFETY_MARGIN_MS，
     宁可落后（recvWindow 60s 兜底）绝不超前；
  2) 多样本校准取最小 RTT（NTP 风格），降低非对称延迟把偏差测偏的风险；
  3) heal_after_timestamp_ahead_reject()：收到 -1021 时强制重校准自愈，
     供各 market/list 调用方重签重试一次。
- 周期性刷新由采样循环驱动（fetch_market_data_multi 每 5s 调用、内部 60s 节流），
  不再依赖启动单次校准。

⚠️ now_ms() 不含安全余量：影子检测器用它做窗口对齐/等待计时，后移会让整个
影子舰队统一晚触发；余量只作用于 signed_timestamp_ms()（签名层）。
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger

_BINANCE_TIME_URL = "https://api.binance.com/api/v3/time"

# 签名时间戳安全余量：后移 1.5s。即使校准偏差 +1s（非对称延迟极端值）也仍
# 落后服务器 0.5s，落在 recvWindow=60000 容忍区间内，杜绝「超前 >1000ms」拒收
_AHEAD_SAFETY_MARGIN_MS = 1500

# 进程级共享时钟偏差（毫秒）
_time_offset_ms: int = 0
_sync_lock = asyncio.Lock()
_last_sync_ts: float = 0.0


async def _measure_once(client: httpx.AsyncClient) -> tuple[int, int]:
    """单次时钟测量，返回 (offset_ms, rtt_ms)。"""
    before = int(time.time() * 1000)
    resp = await client.get(_BINANCE_TIME_URL)
    after = int(time.time() * 1000)
    server_time = resp.json()["serverTime"]
    rtt = max(after - before, 0)
    return server_time - (before + rtt // 2), rtt


async def sync_server_time(
    *, force: bool = False, min_interval_sec: float = 60.0, samples: int = 5
) -> int:
    """与 Binance 服务器校准时钟偏差（进程级共享）。

    NTP 风格多样本：连续采样 samples 次，取 RTT 最小的样本计算偏差——单次
    测量在非对称/抖动网络下 midpoint 会系统性偏离（RTT 越大偏离越大），
    min-RTT 样本受网络非对称性影响最小。

    Args:
        force: 为 True 时忽略最小间隔限制强制校准。
        min_interval_sec: 两次校准的最小间隔，避免频繁请求。
        samples: 每次校准的测量样本数。

    Returns:
        当前时钟偏差（毫秒）。校准失败时保留上一次的有效值。
    """
    global _time_offset_ms, _last_sync_ts

    async with _sync_lock:
        now = time.time()
        if not force and (now - _last_sync_ts) < min_interval_sec:
            return _time_offset_ms

        try:
            best_offset: int | None = None
            best_rtt: int | None = None
            async with httpx.AsyncClient(timeout=10) as client:
                for _ in range(samples):
                    offset, rtt = await _measure_once(client)
                    if best_rtt is None or rtt < best_rtt:
                        best_offset, best_rtt = offset, rtt
            _time_offset_ms = best_offset  # type: ignore[assignment]
            _last_sync_ts = now
            logger.info(
                "共享时钟校准完成 | 偏差={}ms | 最小RTT={}ms | 样本={}",
                _time_offset_ms, best_rtt, samples,
            )
        except Exception as e:
            logger.warning("共享时钟校准失败（沿用上次偏差 {}ms）: {}", _time_offset_ms, e)

        return _time_offset_ms


def get_offset_ms() -> int:
    """返回当前共享时钟偏差（毫秒）。"""
    return _time_offset_ms


def now_ms() -> int:
    """返回补偿了服务器偏差的当前毫秒时间戳。

    ⚠️ 窗口对齐/等待计时等场景需要真实的服务器对齐时间，本函数**不含**
    签名安全余量；仅供 Binance 签名时应改用 signed_timestamp_ms()。
    """
    return int(time.time() * 1000) + _time_offset_ms


def signed_timestamp_ms() -> int:
    """Binance 签名专用时间戳：now_ms() 再后移安全余量。

    币安对超前服务器 >1000ms 的时间戳无条件拒收（-1021），落后则由
    recvWindow（60s）容忍；后移余量后即便校准偏差极端 +1s 也仍落后，
    从源头杜绝超前拒收。
    """
    return now_ms() - _AHEAD_SAFETY_MARGIN_MS


async def heal_after_timestamp_ahead_reject(response_text: str) -> bool:
    """-1021（时间戳超前拒收）自愈入口：强制重校准共享时钟。

    Args:
        response_text: 币安错误响应体（JSON 文本）。

    Returns:
        是否识别为 -1021 并触发了强制重校准；调用方可据此重签（新时间戳）
        并重试一次请求。
    """
    if '"code":-1021' not in response_text:
        return False
    logger.warning("币安 -1021 时间戳超前拒收，强制重校准共享时钟")
    await sync_server_time(force=True)
    return True
