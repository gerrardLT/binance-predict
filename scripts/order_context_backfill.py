"""历史订单证据包回填脚本（一次性/可重跑，非运行时依赖）。

用法（本地或 VPS 容器内）：
    .venv/Scripts/python.exe scripts/order_context_backfill.py \
        [--channel scene_momentum_fade] [--limit 200] [--rebuild]

- 默认只补「已结算 FILLED 且尚无 pack」的订单；--rebuild 强制重建（幂等覆盖）。
- K 线由 data-api.binance.vision 回放（与生产 collector 同源）。
- 只写 order_context_packs，不触碰任何交易/结算数据。
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select as sa_select  # noqa: E402

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import OrderContextPack, TradeOrderModel  # noqa: E402
from binance_predict.services.data_collector import BinanceDataCollector  # noqa: E402
from binance_predict.services.order_context_recorder import recorder  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default=None, help="只回填该通道（默认全部）")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--rebuild", action="store_true", help="已有 pack 的也重建")
    args = ap.parse_args()

    collector = BinanceDataCollector()
    recorder.kline_fetcher = collector.fetch_klines_ending_at

    async with async_session_factory() as db:
        stmt = (
            sa_select(TradeOrderModel.id)
            .where(
                TradeOrderModel.status == "FILLED",
                TradeOrderModel.win.isnot(None),
                TradeOrderModel.window_start.isnot(None),
            )
            .order_by(TradeOrderModel.created_at.desc())
            .limit(args.limit)
        )
        if args.channel:
            stmt = stmt.where(TradeOrderModel.signal_version == args.channel)
        ids = [r[0] for r in (await db.execute(stmt)).all()]
        if not args.rebuild:
            have = set(r[0] for r in (await db.execute(
                sa_select(OrderContextPack.order_id).where(
                    OrderContextPack.order_id.in_(ids))
            )).all())
            ids = [i for i in ids if i not in have]
        total = len(ids)

    print(f"待回填: {total} 单（channel={args.channel or 'ALL'}, rebuild={args.rebuild}）")
    ok = partial = fail = 0
    for i, oid in enumerate(ids, 1):
        try:
            status = await recorder.build_and_store(oid)
            if status == "BUILT":
                ok += 1
            elif status.startswith("PARTIAL"):
                partial += 1
            else:
                fail += 1
        except Exception as exc:
            fail += 1
            print(f"  [{i}/{total}] order={oid} 失败: {exc}")
        if i % 25 == 0 or i == total:
            print(f"  进度 {i}/{total} | BUILT={ok} PARTIAL={partial} FAIL={fail}")
    print(f"完成: BUILT={ok} PARTIAL={partial} FAIL={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
