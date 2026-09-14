#!/usr/bin/env python3
"""Read-only coverage audit for the unified shadow execution ledger."""
from __future__ import annotations

import asyncio
import json
import os
import sys

from sqlalchemy import func, select

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from binance_predict.db.engine import async_session_factory  # noqa: E402
from binance_predict.db.models import (  # noqa: E402
    ShadowExecutionAssessment,
    TradeOrderModel,
)
from binance_predict.services.shadow_execution_adapters import (  # noqa: E402
    SHADOW_SOURCE_ADAPTERS,
)
from binance_predict.services.shadow_execution_registry import (  # noqa: E402
    SHADOW_VERSION_SPECS,
)
from binance_predict.services.shadow_execution_types import (  # noqa: E402
    EXECUTION_POLICY_VERSION,
    ReasonCode,
    SourceType,
)


async def _count(session, statement) -> int:
    return int((await session.execute(statement)).scalar_one() or 0)


async def main() -> int:
    registered_by_source = {
        source_type: sorted(
            version
            for version, spec in SHADOW_VERSION_SPECS.items()
            if spec.source_type is source_type
        )
        for source_type in SourceType
    }
    source_counts: dict[str, int] = {}
    projected_counts: dict[str, int] = {}
    unregistered_versions: dict[str, list[str]] = {}

    async with async_session_factory() as session:
        for adapter in SHADOW_SOURCE_ADAPTERS:
            versions = registered_by_source[adapter.source_type]
            source_counts[adapter.source_type.value] = await _count(
                session,
                select(func.count(adapter.model.id)).where(
                    adapter.model.version.in_(versions)
                ),
            )
            projected_counts[adapter.source_type.value] = await _count(
                session,
                select(func.count(ShadowExecutionAssessment.id)).where(
                    ShadowExecutionAssessment.policy_version == EXECUTION_POLICY_VERSION,
                    ShadowExecutionAssessment.source_type == adapter.source_type.value,
                    ShadowExecutionAssessment.source_id.isnot(None),
                ),
            )
            found_versions = set((await session.execute(
                select(adapter.model.version).distinct()
            )).scalars().all())
            unregistered_versions[adapter.source_type.value] = sorted(
                str(version) for version in found_versions
                if str(version) not in SHADOW_VERSION_SPECS
            )

        total_assessments = await _count(
            session,
            select(func.count(ShadowExecutionAssessment.id)).where(
                ShadowExecutionAssessment.policy_version == EXECUTION_POLICY_VERSION
            ),
        )
        known_execution = await _count(
            session,
            select(func.count(ShadowExecutionAssessment.id)).where(
                ShadowExecutionAssessment.policy_version == EXECUTION_POLICY_VERSION,
                ShadowExecutionAssessment.execution_eligible.isnot(None),
            ),
        )
        legacy_unknown = await _count(
            session,
            select(func.count(ShadowExecutionAssessment.id)).where(
                ShadowExecutionAssessment.policy_version == EXECUTION_POLICY_VERSION,
                ShadowExecutionAssessment.reason_code == ReasonCode.LEGACY_UNKNOWN.value,
            ),
        )
        source_missing = await _count(
            session,
            select(func.count(ShadowExecutionAssessment.id)).where(
                ShadowExecutionAssessment.policy_version == EXECUTION_POLICY_VERSION,
                ShadowExecutionAssessment.source_id.is_(None),
            ),
        )
        signal_orders = await _count(
            session,
            select(func.count(TradeOrderModel.id)).where(
                TradeOrderModel.signal_version.isnot(None)
            ),
        )
        linked_orders = await _count(
            session,
            select(func.count(TradeOrderModel.id)).where(
                TradeOrderModel.signal_version.isnot(None),
                TradeOrderModel.assessment_id.isnot(None),
            ),
        )
        settled_orders = await _count(
            session,
            select(func.count(TradeOrderModel.id)).where(
                TradeOrderModel.signal_version.isnot(None),
                TradeOrderModel.settled_at.isnot(None),
            ),
        )
        linked_settled_orders = await _count(
            session,
            select(func.count(TradeOrderModel.id)).where(
                TradeOrderModel.signal_version.isnot(None),
                TradeOrderModel.settled_at.isnot(None),
                TradeOrderModel.assessment_id.isnot(None),
            ),
        )
        duplicate_exchange_orders = await _count(
            session,
            select(func.count()).select_from(
                select(TradeOrderModel.order_id)
                .where(TradeOrderModel.order_id.isnot(None))
                .group_by(TradeOrderModel.order_id)
                .having(func.count(TradeOrderModel.id) > 1)
                .subquery()
            ),
        )

    source_total = sum(source_counts.values())
    projected_total = sum(projected_counts.values())
    report = {
        "policy_version": EXECUTION_POLICY_VERSION,
        "registry": {
            "versions": len(SHADOW_VERSION_SPECS),
            "source_types": len(registered_by_source),
            "versions_by_source": {
                source_type.value: len(versions)
                for source_type, versions in registered_by_source.items()
            },
        },
        "sources": {
            "rows": source_counts,
            "projected": projected_counts,
            "total_rows": source_total,
            "total_projected": projected_total,
            "coverage": projected_total / source_total if source_total else None,
            "unregistered_versions": unregistered_versions,
        },
        "assessments": {
            "total": total_assessments,
            "known_execution": known_execution,
            "known_execution_coverage": (
                known_execution / total_assessments if total_assessments else None
            ),
            "source_missing": source_missing,
            "legacy_unknown": legacy_unknown,
        },
        "orders": {
            "signal_orders": signal_orders,
            "linked": linked_orders,
            "link_rate": linked_orders / signal_orders if signal_orders else None,
            "settled": settled_orders,
            "linked_settled": linked_settled_orders,
            "settlement_link_rate": (
                linked_settled_orders / settled_orders if settled_orders else None
            ),
            "duplicate_exchange_order_groups": duplicate_exchange_orders,
        },
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
