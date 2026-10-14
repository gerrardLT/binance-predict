"""Best-effort reconciliation from proven trade-order facts into assessments."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from binance_predict.db.models import ShadowExecutionAssessment, TradeOrderModel

from .shadow_execution_registry import SHADOW_VERSION_SPECS
from .shadow_execution_store import (
    backfill_eligibility,
    get_or_create_runtime_assessment,
    patch_assessment,
)
from .shadow_execution_types import (
    AssessmentPatch,
    ReasonCode,
    TerminalStage,
    stage_rank,
)


@dataclass(frozen=True)
class ReconcileReport:
    scanned: int = 0
    linked: int = 0
    backfilled: int = 0
    skipped: int = 0
    duplicate_exchange_orders: int = 0


def _order_terminal(order: TradeOrderModel) -> tuple[TerminalStage, ReasonCode]:
    """由订单事实推导终态；先看 status 再看是否有交易所单号。

    2026-10-14 修复：旧实现把「无 order_id」一律判为 UNKNOWN/SOURCE_MISSING，
    于是护栏/市场/报价类弃单（有评估绑定、无单号）永远补不上资格列，
    对账每轮重复扫同一批最早旧单、游标卡死（生产实测 scanned=200 linked=0）。
    """
    has_order_id = bool(str(order.order_id or "").strip())
    status = str(order.status or "").upper()
    if status == "PENDING":
        return TerminalStage.ORDER, ReasonCode.ORDER_PENDING
    if status == "FAILED":
        # 有单号=已提交后被拒/取消；无单号=执行前弃单（护栏/市场/报价）。
        # 两者都是执行阶段事实，区别只在是否进入交易所。
        return (
            (TerminalStage.FILL, ReasonCode.ORDER_FAILED)
            if has_order_id
            else (TerminalStage.EXECUTION, ReasonCode.ORDER_FAILED)
        )
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

    linked = backfilled = skipped = 0
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
                # 订单已经存在即证明策略与操作门均通过；是否进入执行由交易所
                # 单号事实决定（无单号=执行前弃单，execution_eligible=False）。
                has_order_id = bool(str(order.order_id or "").strip())
                eligibility = {
                    "strategy_eligible": True,
                    "operational_eligible": True,
                    "execution_eligible": has_order_id,
                }
                if (
                    stage is not TerminalStage.UNKNOWN
                    and stage_rank(stage.value) > stage_rank(assessment.terminal_stage)
                ):
                    await patch_assessment(
                        session,
                        AssessmentPatch(
                            assessment_id=int(assessment.id),
                            stage=stage,
                            reason=reason,
                            **eligibility,
                        ),
                    )
                else:
                    # 不推进（同级或更低，含运行时已写更具体原因的护栏弃单）：
                    # 只补仍为 NULL 的资格列，保留已有 stage/reason。
                    await backfill_eligibility(
                        session, assessment_id=int(assessment.id), **eligibility
                    )
                    backfilled += 1
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
        backfilled=backfilled,
        skipped=skipped,
        duplicate_exchange_orders=duplicates,
    )
