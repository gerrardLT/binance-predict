#!/usr/bin/env python3
"""首触研究账本回填（Phase 1，只读 SentimentWindow → 写研究表）。

幂等：按 window_start upsert（审计行必写，母事件行仅有首触时写）。
只读研究路径：不触下单链路、不获取 _trade_lock；DB 失败即退出，不影响实盘。

注意：生产 DB 本地不可达（expose-only）——生产建表/回填走 GitHub Actions psql；
本脚本面向本地库或 Actions 内执行。

用法：
    .venv\\Scripts\\python.exe scripts/build_firsthit_ledger.py [--since-ms 0] [--limit 0]
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from loguru import logger
from sqlalchemy import select as sa_select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, ".")

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import (  # noqa: E402
    FirstHitMotherEvent, FirstHitWindowAudit, SentimentWindow,
)
from binance_predict.research.firsthit_ledger import rows_from_window  # noqa: E402

CHUNK = 500  # 与预热水位同款分块，限制 JSONB 曲线峰值内存


async def upsert_window(session, window) -> tuple[bool, bool]:
    """单窗 upsert：返回 (audit_written, mother_written)。"""
    audit, mother = rows_from_window(window)
    await session.execute(
        pg_insert(FirstHitWindowAudit).values(**audit)
        .on_conflict_do_update(index_elements=["window_start"], set_={
            k: audit[k] for k in audit if k != "created_at"})
    )
    if mother is not None:
        await session.execute(
            pg_insert(FirstHitMotherEvent).values(**mother)
            .on_conflict_do_update(index_elements=["window_start"], set_={
                k: mother[k] for k in mother if k != "created_at"})
        )
        return True, True
    # 窗口从「有母事件」退化为「无」不可能发生（特征只增不改）；保守起见不删旧行。
    return True, False


async def main(since_ms: int, limit: int) -> int:
    written = mothers = 0
    async with async_session_factory() as session:
        stmt = sa_select(SentimentWindow).where(
            SentimentWindow.start_time >= since_ms).order_by(SentimentWindow.start_time)
        if limit > 0:
            stmt = stmt.limit(limit)
        batch = []
        stream = await session.stream_scalars(stmt)
        async for window in stream:
            batch.append(window)
            if len(batch) < CHUNK:
                continue
            for win in batch:
                a, m = await upsert_window(session, win)
                written += int(a)
                mothers += int(m)
            await session.commit()
            logger.info("账本回填进度 | 窗 {} | 母事件 {}", written, mothers)
            batch = []
        for win in batch:
            a, m = await upsert_window(session, win)
            written += int(a)
            mothers += int(m)
        await session.commit()
    logger.info("账本回填完成 | 窗 {} | 母事件 {}", written, mothers)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-ms", type=int, default=0, help="只回填 start_time >= 该毫秒的窗")
    ap.add_argument("--limit", type=int, default=0, help="最多处理窗数（0=不限）")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.since_ms, args.limit)))
