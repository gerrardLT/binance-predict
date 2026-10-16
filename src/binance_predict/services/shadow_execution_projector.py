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


async def _settle_event_and_labels(session, event) -> None:
    """事件行 + 同物理事件下的逻辑标签前向行一起补结算（2026-10-15）。

    蜡烛族按 timeframe 只落一行物理事件（version=candle_hm_bull_*_event_v1），18 个
    逻辑标签各自另有前向行——observe_kline 在触发当刻建行并挂 1/5/10U 可执行报价阶梯。
    投影器按 row.version 建账只会覆盖物理事件版本，标签行的 ladder 因此永远停在
    unsettled、面板「可执行报价」恒空；此处按 (signal_version, target_window_start)
    回填标签结算（幂等，含部署前遗留的存量行）。方向按标签自身落库方向判定，
    故与物理事件的 DOWN 方向解耦（candidate_*_ret3_down_upper_* 为 UP）。
    """
    outcome = getattr(event, "theoretical_outcome", None)
    if outcome not in {"UP", "DOWN"}:
        return
    from .shadow_forward_evidence import settle_firsthit
    await settle_firsthit(session, event.target_window_start, outcome,
                          event.market_period, event.signal_version)
    payload = getattr(event, "source_payload", None) or {}
    labels = list(payload.get("all_matched_signal_ids") or payload.get("matched_signal_ids") or [])
    for label in labels:
        if label == event.signal_version:
            continue
        await settle_firsthit(session, event.target_window_start, outcome,
                              event.market_period, label)


async def _backfill_candle_label_forward_rows(session, limit: int = 200) -> int:
    """回填「源事件已结算、标签前向行仍 unsettled」的存量行（蜡烛族）。

    refresh_statement 只重扫「状态刚变化」的源行，因此本修正部署前遗留的标签行不会被
    自动带出；此处按 (事件版本, 目标窗) 回查已结算的物理事件行，用其 settle_outcome
    补写标签行（含该行 1/5/10U 阶梯的结算估计）。只认蜡烛物理事件版本，其它族不受影响；
    查不到源事件/标签不在命中集 → 跳过（绝不猜结果）。按目标窗升序、有界批量，反复运行
    直至收敛（幂等）。
    """
    from datetime import datetime, timezone

    from sqlalchemy import select as sa_select

    from binance_predict.db.models import KlineShadowSignal, ShadowExecutionAssessment

    from .candlestick_shadow_detector import EVENT_VERSION_BY_TF
    from .shadow_forward_evidence import apply_settlement

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    # 下沿 45 天与面板取数窗口一致；上沿留 30 分钟，避免抢在正常结算路径前动手
    floor_ms = now_ms - 45 * 86_400_000
    ceiling_ms = now_ms - 30 * 60_000
    pending = (await session.execute(
        sa_select(ShadowExecutionAssessment).where(
            ShadowExecutionAssessment.policy_version.like("forward-v2-%"),
            ShadowExecutionAssessment.theoretical_outcome.is_(None),
            ShadowExecutionAssessment.target_window_start >= floor_ms,
            ShadowExecutionAssessment.target_window_start < ceiling_ms,
        ).order_by(ShadowExecutionAssessment.target_window_start.asc()).limit(limit)
    )).scalars().all()
    if not pending:
        return 0

    windows_by_period: dict[str, set[int]] = {}
    for row in pending:
        windows_by_period.setdefault(row.market_period, set()).add(int(row.target_window_start))
    settled: dict[tuple[str, int], tuple[str | None, set[str]]] = {}
    for period, targets in windows_by_period.items():
        event_version = EVENT_VERSION_BY_TF.get(period)
        if event_version is None:
            continue
        rows = (await session.execute(
            sa_select(KlineShadowSignal).where(
                KlineShadowSignal.version == event_version,
                KlineShadowSignal.target_bar_start.in_(sorted(targets)),
                KlineShadowSignal.status == "SETTLED",
            )
        )).scalars().all()
        for row in rows:
            snapshot = row.feature_snapshot or {}
            labels = set(snapshot.get("all_matched_signal_ids")
                         or snapshot.get("matched_signal_ids") or [])
            settled[(event_version, int(row.target_bar_start))] = (row.settle_outcome, labels)

    filled = 0
    for row in pending:
        source = settled.get((EVENT_VERSION_BY_TF.get(row.market_period, ""),
                              int(row.target_window_start)))
        if source is None:
            continue
        outcome, labels = source
        if outcome not in {"UP", "DOWN"} or str(row.signal_version) not in labels:
            continue
        apply_settlement(row, outcome)
        filled += 1
    return filled


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
                        await _settle_event_and_labels(session, event)
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
                        await _settle_event_and_labels(session, event)
                    await session.commit()
                    projected += len(changed_rows)
                except Exception as exc:
                    await session.rollback()
                    logger.warning(
                        "统一影子执行账本终态刷新失败（继续其他来源）| source={} error={}",
                        adapter.source_type.value, exc,
                    )
            # 存量标签行回填（蜡烛族）：refresh 只重扫状态刚变化的源行，
            # 故本修正部署前遗留的 unsettled 标签行需显式回查源事件补结算。
            try:
                filled = await _backfill_candle_label_forward_rows(session)
                await session.commit()
                if filled:
                    logger.info("统一影子执行账本标签行回填完成 | filled={}", filled)
            except Exception as exc:
                await session.rollback()
                logger.warning("统一影子执行账本标签行回填失败（不影响其它投影）| {}", exc)
            try:
                report = await reconcile_shadow_execution_orders(
                    session, batch_size=self.batch_size)
                await session.commit()
                if report.linked or report.backfilled or report.skipped or report.duplicate_exchange_orders:
                    logger.info(
                        "统一影子执行账本旧订单对账完成 | scanned={} linked={} backfilled={} skipped={} duplicates={}",
                        report.scanned, report.linked, report.backfilled, report.skipped,
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
