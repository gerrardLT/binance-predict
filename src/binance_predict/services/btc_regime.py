"""BTC 24h regime 数据源（ret24 门禁共用：影子检测器与实盘执行器同口径）。

口径（与 Part 3 Predexon 长周期回测逐位对齐；v4 采用严格 ex-ante 版）：
    i = bisect_right(opens, ts) − 1（触发时点所在 5m K）
    ret24 = closes[i−1] / closes[i−1−288] − 1
    —— 前一根**已收盘** 5m close vs 其前 288 根（24h）close。
    为什么用 i−1：触发发生在窗内 45~60s，触发时点所在 K 尚未收盘，其 close
    是未来数据（Part 3 回测 orig 口径 closes[i] 的未来函数已由
    .pytest_tmp/predexon_bt_followup.py 量化修正）；前一根 K 的 close 在触发
    时点必然已知，影子/实盘/回测三方零漂移。
触发判定含边界：ret24% ≤ 阈值（−1.0）即过门禁（阈值见 quote_edge_detector.REGIME_GUARDS）。
数据源：settings.binance_api_base 公共 K 线 REST（免 key，与 data_collector 同
host 可经 binance_api_base 指向镜像），limit 覆盖触发点前 24h + 缓冲。
进程级 TTL 缓存（60s）+ 锁内单飞刷新：同窗多通道触发共享一次 REST 往返。
拉取失败 → 沿用旧缓存（陈旧度 ≤ 10min 内仍可用；连续失败超限或从未成功
→ ret24 返回 None）；样本不足/基准价非法 → None。调用方保守拒绝
（同 v2/v3「门禁数据缺失不落表/不触发」口径）——陈旧数据不等于保守数据，
超限宁可拒绝（防 REST 与预测市场 WS 分离故障期误放行/错标样本）。
"""
from __future__ import annotations

import asyncio
import bisect
import time

import httpx
from loguru import logger

from binance_predict.config.settings import settings

RET24_BARS = 288        # 24h / 5m
FETCH_LIMIT = 340       # 290(=288+1) + 触发点距 now 缓冲（≤50 根 ≈ 4.2h，冗余充分）
CACHE_TTL_S = 60.0
STALE_MAX_S = 600.0     # 刷新连续失败超 10min → 数据视为不可用（保守 None，防陈旧误放行）
KLINE_INTERVAL_MS = 300_000


class BtcRegimeFeed:
    """最近 5m K 线缓存 + 触发时点 ret24 计算（多调用方共享单例）。"""

    def __init__(self) -> None:
        self._opens: list[int] = []
        self._closes: list[float] = []
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()
        self._last_error: str | None = None

    async def ret24_at(self, ts_ms: int) -> float | None:
        """触发时点过去 24h 收益（严格 ex-ante 口径）；数据不可用 → None。"""
        await self._ensure_fresh()
        if (not self._opens
                or time.monotonic() - self._fetched_at > STALE_MAX_S):
            # 冷启动失败 / 刷新连续失败超陈旧度上限 → 保守 None
            # （实盘重查后弃单、影子不落；陈旧 ret24 可能误放行/错标样本）
            return None
        i = bisect.bisect_right(self._opens, ts_ms) - 1
        # i−1−288 ≥ 0 且基准价合法；样本不足说明缓存刚冷启动或数据源残缺 → 保守 None
        if i < RET24_BARS + 1:
            return None
        base = self._closes[i - 1 - RET24_BARS]
        if base <= 0:
            return None
        return self._closes[i - 1] / base - 1.0

    async def _ensure_fresh(self) -> None:
        """TTL 缓存 + 锁内单飞：并发首个调用刷新，其余直接读。"""
        if self._opens and time.monotonic() - self._fetched_at < CACHE_TTL_S:
            return
        async with self._lock:
            if self._opens and time.monotonic() - self._fetched_at < CACHE_TTL_S:
                return
            await self._refresh()

    async def _refresh(self) -> None:
        url = f"{settings.binance_api_base}/api/v3/klines"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params={
                    "symbol": settings.symbol,
                    "interval": "5m",
                    "limit": FETCH_LIMIT,
                })
                resp.raise_for_status()
                data = resp.json()
            kl = [(int(k[0]), float(k[4])) for k in data
                  if isinstance(k, list) and len(k) > 4]
            kl.sort(key=lambda x: x[0])
            # 丢弃最后一根未收盘 K（与 fetch_recent_klines 同规），保证 close 完整
            if kl and kl[-1][0] + KLINE_INTERVAL_MS > int(time.time() * 1000):
                kl = kl[:-1]
            if len(kl) <= RET24_BARS:
                raise ValueError(f"kline 样本不足：{len(kl)} ≤ {RET24_BARS}")
            self._opens = [k[0] for k in kl]
            self._closes = [k[1] for k in kl]
            self._fetched_at = time.monotonic()
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            # 保留旧缓存（若曾成功，过期数据好过无数据；从未成功则调用方收 None）
            logger.warning("btc_regime K 线刷新失败（沿用旧缓存/无数据）| {}", exc)

    def status(self) -> dict:
        """观测用状态（排障：缓存根数/新鲜度/降级态/最近错误）。"""
        age_s = (round(time.monotonic() - self._fetched_at, 1)
                 if self._fetched_at else None)
        return {
            "bars": len(self._opens),
            "last_open": self._opens[-1] if self._opens else None,
            "age_s": age_s,
            "stale": age_s is None or age_s > STALE_MAX_S,
            "last_error": self._last_error,
        }


regime_feed = BtcRegimeFeed()   # 影子检测器与实盘执行器共享（进程级单例）
