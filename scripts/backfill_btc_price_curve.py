#!/usr/bin/env python3
"""回填 sentiment_windows.curve_btc_price（币安公开 1m K 线，幂等）。

背景：本地库 5014 个窗口中 curve_btc_price 有效的仅 20 个，导致分箱
price 通道（price <- curve_btc_price，见 symbolizer.py）整体缺数据，
跨通道谓词（lead/sync）无从检验。币安公开 K 线 API 无需 key 即可拉取
历史 1m K 线，可把缺失窗口的 BTC 局内价格曲线补齐。

粒度说明：原采集为 15s 一点（约 20 点/窗口）；回填用 1m K 线收盘价
（约 5 点/窗口），粒度更粗但保留窗口内趋势方向，分箱符号化可用。

幂等：只回填 curve_btc_price 缺失（NULL 或非数组/空数组）的窗口；
已有效的窗口一律跳过，重复执行安全。

用法：
    python scripts/backfill_btc_price_curve.py            # 实际回填
    python scripts/backfill_btc_price_curve.py --dry-run  # 只统计不写库
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import httpx  # noqa: E402
from sqlalchemy import select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import SentimentWindow  # noqa: E402

_KLINES_URL = "https://api.binance.com/api/v3/klines"
_SYMBOL = "BTCUSDT"
_INTERVAL = "1m"
_LIMIT = 1000  # API 单次上限
_MINUTE_MS = 60_000


def _is_missing(curve: object) -> bool:
    """curve_btc_price 是否缺失（NULL / 非数组 / 空数组）。"""
    return not (isinstance(curve, list) and len(curve) > 0)


async def _fetch_klines(client: httpx.AsyncClient, start_ms: int, end_ms: int) -> dict[int, float]:
    """拉取 [start_ms, end_ms] 内全部 1m K 线，返回 {开盘时间ms: 收盘价}。"""
    closes: dict[int, float] = {}
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": _SYMBOL,
            "interval": _INTERVAL,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": _LIMIT,
        }
        resp = await client.get(_KLINES_URL, params=params, timeout=30.0)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for k in batch:
            closes[int(k[0])] = float(k[4])  # openTime -> close
        last_open = int(batch[-1][0])
        cursor = last_open + _MINUTE_MS
        if len(batch) < _LIMIT:
            break
        time.sleep(0.2)  # 温和限速，避免触发权重限制
    return closes


def _build_curve(closes: dict[int, float], start_ms: int, end_ms: int) -> list[dict]:
    """用 1m K 线收盘价构造窗口内价格曲线 [{t, v}, ...]（按时间升序）。"""
    return [
        {"t": t, "v": closes[t]}
        for t in sorted(closes)
        if start_ms <= t < end_ms
    ]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只统计缺失窗口，不拉取不写库")
    args = parser.parse_args()

    # 1. 找出所有缺 curve_btc_price 的窗口
    async with async_session_factory() as session:
        rows = (await session.execute(
            select(SentimentWindow.id, SentimentWindow.start_time,
                   SentimentWindow.end_time, SentimentWindow.curve_btc_price)
            .order_by(SentimentWindow.start_time)
        )).all()
    missing = [(r.id, r.start_time, r.end_time) for r in rows if _is_missing(r.curve_btc_price)]
    print(f"总窗口 {len(rows)}，缺 curve_btc_price {len(missing)} 个")
    if not missing or args.dry_run:
        return 0

    # 2. 一次性拉取覆盖全时间段的 1m K 线
    min_start = min(m[1] for m in missing)
    max_end = max(m[2] for m in missing)
    print(f"拉取 BTCUSDT 1m K 线: [{min_start}, {max_end}] ...")
    async with httpx.AsyncClient() as client:
        closes = await _fetch_klines(client, min_start, max_end)
    print(f"K 线获取完成: {len(closes)} 根")

    # 3. 逐窗口构造曲线并写库（单事务批量提交）
    filled = 0
    skipped_empty = 0
    async with async_session_factory() as session:
        for wid, start_ms, end_ms in missing:
            curve = _build_curve(closes, start_ms, end_ms)
            if not curve:
                skipped_empty += 1
                continue
            obj = await session.get(SentimentWindow, wid)
            if obj is None or not _is_missing(obj.curve_btc_price):
                continue  # 并发安全：已被其他流程填过则跳过
            obj.curve_btc_price = curve
            filled += 1
        await session.commit()

    print(f"回填完成: {filled} 窗口写入曲线，{skipped_empty} 窗口区间内无 K 线（跳过）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
