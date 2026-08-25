"""Binance API 统一限频守卫（R4，2026-08-25 风险评审）。

背景：全仓此前对 429/418 零处理——触发限频后继续同速率打，
418 IP ban 后数据/信号/下单/结算同时瘫痪，需人工介入。

本模块提供进程级单例 rate_guard + 包装函数 binance_request()：
- 响应侧：429/418 → 解析 Retry-After，否则指数退避（2s 起、120s 封顶）；
  连续限频使退避窗口指数增长（熔断降速），成功后逐步回收不立刻清零。
- 请求侧：发起请求前若仍在退避窗口内，先 sleep 到窗口结束再放行，
  避免"明知会被拒还继续打"。
- binance_request() 内部最多重试 2 次；重试耗尽仍限频则返回原响应，
  由调用方的 raise_for_status / 既有错误分支按原语义处理。

所有对 Binance 域名的 httpx 请求点（交易/只读数据/现货 REST 后备/校时）
统一走 binance_request()；webhook 等第三方端点不适用。
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger

# 限频状态码：429 速率超限 / 418 IP 被 ban（Binance 语义）
_THROTTLE_CODES: frozenset[int] = frozenset({429, 418})
# 单请求内部最大重试次数（不含首次）
MAX_RETRIES: int = 2
# 指数退避参数
_BASE_BACKOFF_S: float = 2.0
_MAX_BACKOFF_S: float = 120.0
# 418 无 Retry-After 时的默认禁发窗口
_BAN_DEFAULT_S: float = 120.0
# 单次 sleep 上限（分段睡，便于日志可观测、防止时钟异常下不可控长睡）
_SLEEP_CHUNK_S: float = 30.0


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    """解析 Retry-After 头（秒）。

    Binance 对 418 的 Retry-After 可能是 epoch 毫秒（ban 截止时间），
    启发式：数值大于 1e12 视为 epoch ms，折算成剩余秒数。
    """
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value > 1e12:  # 疑似 epoch 毫秒
        value = value / 1000.0 - time.time()
    return max(0.0, value)


class BinanceRateGuard:
    """进程级限频状态机：记录限频窗口 + 连续限频计数（熔断降速）。

    状态全部基于 time.monotonic()，不依赖墙钟；进程重启后清零（可接受：
    重启本身即天然退避）。并发安全：asyncio.Lock 保护状态迁移。
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._throttled_until: float = 0.0      # monotonic 截止时间
        self._consecutive_throttles: int = 0
        self._total_throttles: int = 0

    def status(self) -> dict:
        """诊断快照（供 /api/health 类端点透传）。"""
        remaining = self._throttled_until - time.monotonic()
        return {
            "throttled": remaining > 0,
            "backoff_remaining_s": round(max(0.0, remaining), 1),
            "consecutive_throttles": self._consecutive_throttles,
            "total_throttles": self._total_throttles,
        }

    def _exponential_backoff_s(self) -> float:
        """指数退避窗口：2s → 4s → 8s …，封顶 120s。"""
        exp = max(0, self._consecutive_throttles - 1)
        return min(_BASE_BACKOFF_S * (2 ** exp), _MAX_BACKOFF_S)

    async def record_throttle(
        self, status_code: int, headers: httpx.Headers | None
    ) -> float:
        """记录一次 429/418，返回本次应等待的秒数并推进禁发窗口。"""
        async with self._lock:
            self._consecutive_throttles += 1
            self._total_throttles += 1
            retry_after = _parse_retry_after(headers) if headers is not None else None
            if retry_after is None:
                retry_after = _BAN_DEFAULT_S if status_code == 418 else self._exponential_backoff_s()
            wait_s = max(retry_after, self._exponential_backoff_s())
            self._throttled_until = max(self._throttled_until, time.monotonic() + wait_s)
            logger.warning(
                "[RATE-LIMIT] Binance API 限频 | status={} 第{}次连续限频 "
                "退避{:.0f}s（Retry-After={}）",
                status_code, self._consecutive_throttles, wait_s,
                retry_after if retry_after == wait_s else "被指数退避覆盖",
            )
            return wait_s

    async def record_success(self) -> None:
        """记录成功：逐步回收连续限频计数（不立刻清零，保持熔断降速惯性）。"""
        async with self._lock:
            if self._consecutive_throttles > 0:
                self._consecutive_throttles = max(0, self._consecutive_throttles - 1)

    async def wait_if_throttled(self) -> float:
        """若仍在禁发窗口内则 sleep 到窗口结束。返回实际等待秒数。"""
        waited = 0.0
        while True:
            remaining = self._throttled_until - time.monotonic()
            if remaining <= 0:
                return waited
            chunk = min(remaining, _SLEEP_CHUNK_S)
            logger.warning(
                "[RATE-LIMIT] 退避窗口内挂起请求，还需等待 {:.0f}s",
                remaining,
            )
            await asyncio.sleep(chunk)
            waited += chunk

    def reset(self) -> None:
        """测试/运维复位。"""
        self._throttled_until = 0.0
        self._consecutive_throttles = 0.0


# 进程级单例：所有 Binance 请求点共享同一限频状态
rate_guard = BinanceRateGuard()


async def binance_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    """统一限频包装：退避窗口内先挂起 → 发请求 → 429/418 退避后重试。

    重试耗尽（MAX_RETRIES 次）仍限频时返回最后的限频响应，
    由调用方既有的 raise_for_status / 错误分支处理——不改变各调用点
    的对外错误语义，只把"限频后继续同速率打"变成"退避 + 有限重试"。

    按方法分发到 client.get/post（而非 client.request）：与既有测试桩
    （桩 get/post）及 httpx 客户端行为双兼容。
    """
    _method = method.upper()
    for attempt in range(MAX_RETRIES + 1):
        await rate_guard.wait_if_throttled()
        if _method == "GET":
            resp = await client.get(url, **kwargs)
        elif _method == "POST":
            resp = await client.post(url, **kwargs)
        else:
            resp = await client.request(method, url, **kwargs)
        if resp.status_code not in _THROTTLE_CODES:
            if resp.status_code < 500:
                await rate_guard.record_success()
            return resp
        await rate_guard.record_throttle(resp.status_code, resp.headers)
        if attempt >= MAX_RETRIES:
            logger.error(
                "[RATE-LIMIT] 重试 {} 次后仍被限频，放弃本次请求 | method={} url={}",
                MAX_RETRIES, method, url,
            )
            return resp
    return resp  # 不可达（防御性）
