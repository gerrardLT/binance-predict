"""Aggregate the unified shadow/live execution ledger.

Gap denominators deliberately exclude unmapped rows. Selection uses mapped
assessment rows with a bound source and known strategy eligibility. Operational
uses rows that passed strategy and have known operational eligibility. Execution
uses rows that passed operational and have known execution eligibility. Fill uses
submitted orders; settlement uses filled orders. Coverage uses all mapped rows
with a bound source and counts rows missing any eligibility value needed to reach
an execution classification. A zero denominator always produces a null rate.

``unknown`` is a non-double-counted row count: terminal stage UNKNOWN, or a
mapped, active row whose operational or execution eligibility is null. Fixed-1U
returns use non-null theoretical realized returns; the executable series is the
paired subset whose execution eligibility is true. Actual returns use one
representative settled order per assessment and stake-weighted ROI.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from binance_predict.db.models import ShadowExecutionAssessment, TradeOrderModel

from .shadow_execution_registry import SHADOW_VERSION_SPECS, SHADOW_VERSIONS

_WEI = Decimal(10) ** 18


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _rate(n: int, denominator: int) -> dict[str, int | float | None]:
    return {"n": n, "rate": n / denominator if denominator else None}


def _stake(order: Any) -> float:
    try:
        value = Decimal(str(_value(order, "amount_in", "0"))) / _WEI
        return float(value) if value.is_finite() else 0.0
    except (InvalidOperation, ValueError, TypeError):
        return 0.0


def _exchange_order_key(order: Any) -> str | None:
    value = str(_value(order, "order_id", "") or "").strip()
    return value or None


def _is_submitted(order: Any) -> bool:
    return _exchange_order_key(order) is not None


def _is_filled(order: Any) -> bool:
    return _is_submitted(order) and str(_value(order, "status", "")).upper() == "FILLED"


def _is_settled(order: Any) -> bool:
    return (
        _is_filled(order)
        and _value(order, "settled_at") is not None
        and _value(order, "pnl") is not None
    )


def _order_rank(order: Any) -> tuple[int, int, int, int]:
    return (
        int(_is_settled(order)),
        int(_is_filled(order)),
        int(_is_submitted(order)),
        int(_value(order, "id", 0) or 0),
    )


def _representative_orders(orders: Iterable[Any]) -> dict[int, Any]:
    placeholders: list[Any] = []
    exchange_orders: dict[str, Any] = {}
    for order in orders:
        if _value(order, "assessment_id") is None:
            continue
        exchange_key = _exchange_order_key(order)
        if exchange_key is None:
            placeholders.append(order)
            continue
        current = exchange_orders.get(exchange_key)
        if current is None or _order_rank(order) > _order_rank(current):
            exchange_orders[exchange_key] = order

    selected: dict[int, Any] = {}
    for order in [*placeholders, *exchange_orders.values()]:
        assessment_id = int(_value(order, "assessment_id"))
        current = selected.get(assessment_id)
        if current is None or _order_rank(order) > _order_rank(current):
            selected[assessment_id] = order
    return selected


def _empty_funnel() -> dict[str, int]:
    return {key: 0 for key in (
        "theoretical", "strategy_eligible", "operational_eligible",
        "execution_eligible", "submitted", "filled", "settled", "unknown",
        "no_live_mapping",
    )}


def _fixed(values: list[float]) -> dict[str, int | float | None]:
    total = sum(values)
    return {"n": len(values), "sum": total, "avg": total / len(values) if values else None}


def _aggregate(rows: list[Any], orders: dict[int, Any]) -> dict[str, Any]:
    funnel = _empty_funnel()
    theoretical_returns: list[float] = []
    executable_returns: list[float] = []
    actual_stake = 0.0
    actual_pnl = 0.0
    actual_n = 0
    reasons: Counter[str] = Counter()
    selection_den = operational_den = execution_den = 0
    selection_n = operational_n = execution_n = 0
    fill_den = fill_n = settlement_den = settlement_n = 0
    coverage_den = coverage_n = 0

    for row in rows:
        source = _value(row, "source_id") is not None
        channel_mapped = _value(row, "channel_mapped")
        reason = str(_value(row, "reason_code", "") or "")
        no_mapping = channel_mapped is False or reason == "NO_LIVE_MAPPING"
        mapped = channel_mapped is True and not no_mapping
        retired = _value(row, "retired") is True
        strategy = _value(row, "strategy_eligible")
        operational = _value(row, "operational_eligible")
        execution = _value(row, "execution_eligible")
        order = orders.get(int(_value(row, "id", 0) or 0))
        reasons[reason] += 1

        funnel["theoretical"] += int(source)
        funnel["strategy_eligible"] += int(strategy is True)
        funnel["operational_eligible"] += int(operational is True)
        funnel["execution_eligible"] += int(execution is True)
        submitted = order is not None and _is_submitted(order)
        funnel["submitted"] += int(submitted)
        filled = order is not None and _is_filled(order)
        settled = order is not None and _is_settled(order)
        funnel["filled"] += int(filled)
        funnel["settled"] += int(settled)
        funnel["no_live_mapping"] += int(no_mapping)
        unknown = channel_mapped is None or str(_value(row, "terminal_stage", "")).upper() == "UNKNOWN" or (
            mapped and not retired and (operational is None or execution is None)
        )
        funnel["unknown"] += int(unknown)

        realized = _value(row, "theoretical_realized_return")
        if realized is not None:
            theoretical_returns.append(float(realized))
            if execution is True:
                executable_returns.append(float(realized))
        if settled:
            actual_n += 1
            actual_stake += _stake(order)
            actual_pnl += float(_value(order, "pnl"))

        if source and mapped and strategy is not None:
            selection_den += 1
            selection_n += int(strategy is False)
        if source and mapped and strategy is True and operational is not None:
            operational_den += 1
            operational_n += int(operational is False)
        if source and mapped and operational is True and execution is not None:
            execution_den += 1
            execution_n += int(execution is False)
        if submitted:
            fill_den += 1
            fill_n += int(not filled)
        if filled:
            settlement_den += 1
            settlement_n += int(not settled)
        if source and mapped:
            coverage_den += 1
            coverage_n += int(strategy is None or (strategy is True and operational is None) or (
                operational is True and execution is None
            ))

    return {
        "funnel": funnel,
        "returns": {
            "theoretical_fixed_1u": _fixed(theoretical_returns),
            "executable_fixed_1u": _fixed(executable_returns),
            "actual": {
                "settled_n": actual_n, "stake": actual_stake, "pnl": actual_pnl,
                "roi": actual_pnl / actual_stake if actual_stake else None,
            },
        },
        "gaps": {
            "selection": _rate(selection_n, selection_den),
            "operational": _rate(operational_n, operational_den),
            "execution": _rate(execution_n, execution_den),
            "fill": _rate(fill_n, fill_den),
            "settlement": _rate(settlement_n, settlement_den),
            "coverage": _rate(coverage_n, coverage_den),
        },
        "reason_counts": [
            {"reason": reason, "count": count}
            for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def _version_order(version: str) -> tuple[int, str]:
    try:
        return (SHADOW_VERSIONS.index(version), version)
    except ValueError:
        return (len(SHADOW_VERSIONS), version)


def _curves(rows: list[Any], orders: dict[int, Any]) -> list[dict[str, Any]]:
    theoretical = executable = pnl = stake = 0.0
    points = []
    ordered = sorted(rows, key=lambda row: (
        int(_value(row, "target_window_start", 0)), _version_order(str(_value(row, "signal_version", ""))),
        str(_value(row, "policy_version", "")), int(_value(row, "id", 0) or 0),
    ))
    for row in ordered:
        realized = _value(row, "theoretical_realized_return")
        if realized is not None:
            theoretical += float(realized)
            if _value(row, "execution_eligible") is True:
                executable += float(realized)
        order = orders.get(int(_value(row, "id", 0) or 0))
        if order is not None and _is_settled(order):
            pnl += float(_value(order, "pnl"))
            stake += _stake(order)
        points.append({
            "timestamp": int(_value(row, "target_window_start")),
            "signal_version": str(_value(row, "signal_version")),
            "policy_version": str(_value(row, "policy_version")),
            "theoretical_fixed_1u": theoretical,
            "executable_fixed_1u": executable,
            "actual_stake": stake,
            "actual_pnl": pnl,
            "actual_roi": pnl / stake if stake else None,
        })
    return points


def build_execution_comparison_from_rows(rows: Iterable[Any], orders: Iterable[Any], *, scope: dict[str, Any]) -> dict[str, Any]:
    """Pure aggregation entry point used by focused tests and offline callers."""
    rows_list = list(rows)
    selected_orders = _representative_orders(orders)
    result = {"scope": scope, **_aggregate(rows_list, selected_orders)}
    grouped: dict[tuple[str, str, str], list[Any]] = {}
    for row in rows_list:
        key = (str(_value(row, "signal_version")), str(_value(row, "policy_version")), str(_value(row, "market_period")))
        grouped.setdefault(key, []).append(row)
    result["versions"] = []
    for (version, policy, period), group in sorted(grouped.items(), key=lambda item: (_version_order(item[0][0]), item[0][1], item[0][2])):
        spec = SHADOW_VERSION_SPECS.get(version)
        result["versions"].append({
            "signal_version": version, "policy_version": policy, "market_period": period,
            "channel_mapped": any(_value(row, "channel_mapped") is True for row in group),
            "live_channel": spec.live_channel if spec else None,
            "retired": all(_value(row, "retired") is True for row in group),
            **_aggregate(group, selected_orders),
        })
    result["curves"] = _curves(rows_list, selected_orders)
    return result


async def build_execution_comparison(
    session: AsyncSession, *, policy_version: str | None = None,
    from_ts: int | None = None, to_ts: int | None = None,
    market_period: str | None = None, include_retired: bool = False,
) -> dict[str, Any]:
    """Load bounded ledger facts in two queries and return the comparison contract."""
    stmt = select(ShadowExecutionAssessment)
    if policy_version is not None:
        stmt = stmt.where(ShadowExecutionAssessment.policy_version == policy_version)
    if from_ts is not None:
        stmt = stmt.where(ShadowExecutionAssessment.target_window_start >= from_ts)
    if to_ts is not None:
        stmt = stmt.where(ShadowExecutionAssessment.target_window_start < to_ts)
    if market_period is not None:
        stmt = stmt.where(ShadowExecutionAssessment.market_period == market_period)
    if not include_retired:
        stmt = stmt.where(ShadowExecutionAssessment.retired.is_not(True))
    stmt = stmt.order_by(ShadowExecutionAssessment.target_window_start, ShadowExecutionAssessment.id)
    rows = list((await session.execute(stmt)).scalars().all())
    ids = [int(row.id) for row in rows]
    orders: list[Any] = []
    if ids:
        order_stmt = select(TradeOrderModel).where(TradeOrderModel.assessment_id.in_(ids))
        orders = list((await session.execute(order_stmt)).scalars().all())
    scope = {
        "policy_version": policy_version, "from": from_ts, "to": to_ts,
        "market_period": market_period, "include_retired": include_retired,
    }
    return build_execution_comparison_from_rows(rows, orders, scope=scope)
