"""Best-effort reconciliation from proven trade-order facts into assessments."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from binance_predict.db.models import ShadowExecutionAssessment, TradeOrderModel

from .shadow_execution_registry import SHADOW_VERSION_SPECS
from .shadow_execution_store import get_or_create_runtime_assessment, patch_assessment
from .shadow_execution_types import AssessmentPatch, ReasonCode, TerminalStage


@dataclass(frozen=True)
class ReconcileReport:
    scanned: int = 0
    linked: int = 0
    skipped: int = 0
    duplicate_exchange_orders: int = 0


def _order_terminal(order: TradeOrderModel) -> tuple[TerminalStage, ReasonCode]:
    if not str(order.order_id or "").strip():
        return TerminalStage.UNKNOWN, ReasonCode.SOURCE_MISSING
    status = str(order.status or "").upper()
    if status == "PENDING":
        return TerminalStage.ORDER, ReasonCode.ORDER_PENDING
    if status == "FAILED":
        return TerminalStage.FILL, ReasonCode.ORDER_FAILED
    if status == "FILLED" and order.settled_at is not None:
        outcome = str(order.settle_outcome or "").upper()
        reason = (
            ReasonCode.SETTLED
            if outcome in {"UP", "DOWN"}
            else ReasonCode.SETTLEMENT_EXPIRED
        )
        return TerminalStage.SETTLEMENT, reason
    if status == "FILLED":
        return TerminalStage.FILL, ReasonCode.ORDER_FILLED
    return TerminalStage.UNKNOWN, ReasonCode.SOURCE_MISSING


def _legacy_order_statement(batch_size: int):
    return (
        select(TradeOrderModel)
        .outerjoin(
            ShadowExecutionAssessment,
            TradeOrderModel.assessment_id == ShadowExecutionAssessment.id,
        )
        .where(
            TradeOrderModel.signal_version.in_(tuple(SHADOW_VERSION_SPECS)),
            TradeOrderModel.window_start.isnot(None),
            TradeOrderModel.market_period.isnot(None),
            TradeOrderModel.direction.isnot(None),
            (
                TradeOrderModel.assessment_id.is_(None)
                | ShadowExecutionAssessment.strategy_eligible.is_(None)
                | ShadowExecutionAssessment.operational_eligible.is_(None)
                | ShadowExecutionAssessment.execution_eligible.is_(None)
            ),
        )
        .order_by(TradeOrderModel.id.asc())
        .limit(batch_size)
    )


async def reconcile_shadow_execution_orders(
    session: AsyncSession,
    *,
    batch_size: int = 200,
) -> ReconcileReport:
    """Bind legacy orders and backfill lifecycle/eligibility facts proven by an order."""
    rows = list((await session.execute(
        _legacy_order_statement(batch_size)
    )).scalars().all())

    linked = skipped = 0
    for order in rows:
        version = str(order.signal_version)
        spec = SHADOW_VERSION_SPECS.get(version)
        if spec is None:
            skipped += 1
            logger.warning(
                "统一影子执行账本：旧订单版本未注册，跳过绑定 | order={} version={}",
                order.id, version,
            )
            continue
        try:
            async with session.begin_nested():
                assessment = await get_or_create_runtime_assessment(
                    session,
                    source_type=spec.source_type.value,
                    signal_version=version,
                    target_window_start=int(order.window_start),
                    market_period=str(order.market_period),
                    direction=str(order.direction),
                )
                stage, reason = _order_terminal(order)
                if stage is not TerminalStage.UNKNOWN:
                    # 订单已经存在即证明策略与操作门均通过；是否进入执行由
                    # 报价/下单事实决定。旧账本缺这三列时在对账阶段补齐。
                    await patch_assessment(
                        session,
                        AssessmentPatch(
                            assessment_id=int(assessment.id),
                            stage=stage,
                            reason=reason,
                            strategy_eligible=True,
                            operational_eligible=True,
                            execution_eligible=bool(order.order_id),
                        ),
                    )
                result = await session.execute(
                    update(TradeOrderModel)
                    .where(
                        TradeOrderModel.id == order.id,
                        TradeOrderModel.assessment_id.is_(None),
                    )
                    .values(assessment_id=int(assessment.id))
                )
                linked += int(result.rowcount or 0)
        except Exception as exc:
            skipped += 1
            logger.warning(
                "统一影子执行账本：旧订单绑定失败 | order={} version={} | {}",
                order.id, version, exc,
            )

    duplicates = int((await session.execute(
        select(func.count()).select_from(
            select(TradeOrderModel.order_id)
            .where(TradeOrderModel.order_id.isnot(None))
            .group_by(TradeOrderModel.order_id)
            .having(func.count(TradeOrderModel.id) > 1)
            .subquery()
        )
    )).scalar_one())
    if duplicates:
        logger.error(
            "统一影子执行账本：检测到重复 exchange order ownership | groups={}",
            duplicates,
        )

    return ReconcileReport(
        scanned=len(rows),
        linked=linked,
        skipped=skipped,
        duplicate_exchange_orders=duplicates,
    )
