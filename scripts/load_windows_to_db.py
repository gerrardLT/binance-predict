#!/usr/bin/env python3
"""把 sentiment_windows.json 导出文件灌入本地 DB（幂等，供本地跑 deep_learn 用）。

唯一约束 (start_time, end_time) 去重：已存在的窗口跳过，重复执行安全。
不带 id / created_at（由 DB 自增/默认），保留全部曲线与结算字段。

用法：
    python scripts/load_windows_to_db.py --from-file sentiment_windows.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from sqlalchemy import func, select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import SentimentWindow  # noqa: E402

_COLUMNS = (
    "start_time", "end_time",
    "curve_up_pct", "curve_down_pct", "curve_up_price", "curve_down_price",
    "curve_participants", "curve_trade_volume", "curve_btc_price",
    "sample_count", "entry_price", "exit_price", "actual_return", "outcome",
    "avg_participants", "avg_trade_volume",
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default="sentiment_windows.json")
    args = ap.parse_args()

    with open(args.from_file, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"读取 {len(rows)} 个窗口（{args.from_file}）")

    # 应用层去重（本地库历史悠久，uq_sw_start_end 约束可能未物理建立）
    async with async_session_factory() as session:
        existing = set(
            (await session.execute(
                select(SentimentWindow.start_time, SentimentWindow.end_time)
            )).all()
        )
    todo = [
        r for r in rows
        if (r.get("start_time"), r.get("end_time")) not in existing
    ]
    print(f"已存在 {len(rows) - len(todo)} 行，待插 {len(todo)} 行")

    inserted = 0
    async with async_session_factory() as session:
        for r in todo:
            session.add(SentimentWindow(**{c: r.get(c) for c in _COLUMNS}))
            inserted += 1
        await session.commit()
    print(f"灌入完成：新增 {inserted} 行")

    # 复核
    async with async_session_factory() as session:
        total = (await session.execute(
            select(func.count(SentimentWindow.id))
        )).scalar()
        with_price = (await session.execute(
            select(func.count(SentimentWindow.id)).where(
                SentimentWindow.curve_up_price.isnot(None)
            )
        )).scalar()
    print(f"DB 现共 {total} 个窗口，其中 {with_price} 个带真实价格曲线")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
