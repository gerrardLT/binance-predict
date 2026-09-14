"""Low-frequency, restart-safe projection from legacy shadow tables."""
from __future__ import annotations

import asyncio

from loguru import logger

from binance_predict.db.engine import async_session_factory

from .shadow_execution_adapters import SHADOW_SOURCE_ADAPTERS
from .shadow_execution_reconciler import reconcile_shadow_execution_orders
from .shadow_execution_registry import SHADOW_VERSION_SPECS
from .shadow_execution_store import build_assessment_values, upsert_assessment
from .shadow_execution_types import EXECUTION_POLICY_VERSION


class ShadowExecutionProjector:
    def __init__(self, interval_seconds: float = 300.0, batch_size: int = 200) -> None:
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="shadow_execution_projector")
        logger.info("统一影子执行账本 projector 已启动 | interval={}s", self.interval_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def project_once(self) -> int:
        projected = 0
        async with async_session_factory() as session:
            for adapter in SHADOW_SOURCE_ADAPTERS:
                try:
                    rows = (await session.execute(
                        adapter.statement(
                            EXECUTION_POLICY_VERSION, self.batch_size)
                    )).scalars().all()
                    batch_projected = 0
                    for row in rows:
                        if str(row.version) not in SHADOW_VERSION_SPECS:
                            logger.warning(
                                "统一影子执行账本跳过未注册版本 | source={} version={} id={}",
                                adapter.source_type.value, row.version, row.id,
                            )
                            continue
                        event = adapter.normalize(row)
                        values = await build_assessment_values(session, event)
                        await upsert_assessment(session, values)
                        batch_projected += 1
                    await session.commit()
                    projected += batch_projected
                except Exception as exc:
                    await session.rollback()
                    logger.warning(
                        "统一影子执行账本 adapter 失败（继续其他来源）| source={} error={}",
                        adapter.source_type.value, exc,
                    )
                    continue
                try:
                    changed_rows = (await session.execute(
                        adapter.refresh_statement(
                            EXECUTION_POLICY_VERSION, self.batch_size)
                    )).scalars().all()
                    for row in changed_rows:
                        if str(row.version) not in SHADOW_VERSION_SPECS:
                            continue
                        event = adapter.normalize(row)
                        values = await build_assessment_values(session, event)
                        await upsert_assessment(session, values)
                    await session.commit()
                    projected += len(changed_rows)
                except Exception as exc:
                    await session.rollback()
                    logger.warning(
                        "统一影子执行账本终态刷新失败（继续其他来源）| source={} error={}",
                        adapter.source_type.value, exc,
                    )
            try:
                report = await reconcile_shadow_execution_orders(
                    session, batch_size=self.batch_size)
                await session.commit()
                if report.linked or report.skipped or report.duplicate_exchange_orders:
                    logger.info(
                        "统一影子执行账本旧订单对账完成 | scanned={} linked={} skipped={} duplicates={}",
                        report.scanned, report.linked, report.skipped,
                        report.duplicate_exchange_orders,
                    )
            except Exception as exc:
                await session.rollback()
                logger.warning(
                    "统一影子执行账本旧订单对账失败（不影响 source 投影）| {}", exc)
        return projected

    async def _loop(self) -> None:
        while self._running:
            try:
                count = await self.project_once()
                if count:
                    logger.info("统一影子执行账本刷新完成 | projected={}", count)
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("统一影子执行账本刷新失败（不影响交易链路）| {}", exc)


shadow_execution_projector = ShadowExecutionProjector()
