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
            # Compatibility alias: this is gate-subset settlement reference, not execution PnL.
            "executable_fixed_1u": _fixed(executable_returns),
            "gate_subset_reference_fixed_1u": _fixed(executable_returns),
            "execution_quote_estimate": {"n": 0, "sum": None, "avg": None},
            "semantics": {
                "executable_fixed_1u": "门禁子集结算参考（兼容字段），不是可执行收益",
                "execution_quote_estimate": "未提供已冻结金额报价证据，不估算成交收益",
                "fees": "参考公式费用假设未核实；实际收益仅取订单 pnl",
                "holding": "持有到结算，未验证提前退出",
            },
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
    from .shadow_forward_evidence import COHORT, PROFILE, quote_returns
    forward = [row for row in rows_list if str(_value(row, "policy_version", "")).startswith("forward-v2-")]
    estimates = [(row, quote_returns(row)) for row in forward]
    live = [row for row in forward if (_value(row, "input_snapshot", {}) or {}).get("capture_mode") == "LIVE"]
    evaluable = [row for row, estimate in estimates if estimate]
    windows = {(_value(row, "market_period"), _value(row, "target_window_start")) for row in evaluable}
    days = {int(_value(row, "target_window_start")) // 86_400_000 for row in evaluable}
    ladder = {}
    amounts = {"1", "5", "10"} | {amount for _, values in estimates for amount in values}
    for amount in sorted(amounts, key=float):
        values = [value[amount] for _, value in estimates if amount in value]
        stake = sum(value["stake"] for value in values)
        from statistics import mean, stdev
        daily = {}
        for row, estimate in estimates:
            if amount in estimate:
                day = int(_value(row, "target_window_start")) // 86_400_000
                daily[day] = daily.get(day, 0.0) + estimate[amount]["stress_3pct"]
        cluster = list(daily.values())
        lower = mean(cluster)-2.58*stdev(cluster)/len(cluster)**.5 if len(cluster) >= 14 else None
        ladder[amount] = {"daily_clusters": len(cluster), "daily_mean_lower_99pct": lower,
                          "uncertainty": "日聚类正态近似；非独立事件不得当独立样本",
                          "n": len(values), "stake": stake,
                          "pnl": sum(value["pnl"] for value in values),
                          "stress_1pct": sum(value["stress_1pct"] for value in values),
                          "stress_3pct": sum(value["stress_3pct"] for value in values),
                          "missing": len(live)-len(values), "actual_fill_n": 0}
    # Never sum candidate/control, different versions or changing profiles as a portfolio.
    grouped_evidence = {}
    for row, estimate in estimates:
        key = (str(_value(row, "signal_version")), str(_value(row, "policy_version")))
        grouped_evidence.setdefault(key, []).append((row, estimate))
    evidence_series = []
    for (version, policy), group in sorted(grouped_evidence.items()):
        for amount in sorted(amounts, key=float):
            usable = sorted([(row, values[amount]) for row, values in group if amount in values],
                            key=lambda item: int(_value(item[0], "target_window_start")))
            cumulative = peak = drawdown = 0.0
            buckets = {}
            daily = {}
            overlap = Counter()
            for row, value in usable:
                cumulative += value["pnl"]
                peak = max(peak, cumulative)
                drawdown = max(drawdown, peak-cumulative)
                start = int(_value(row, "target_window_start"))
                daily[start//86_400_000] = daily.get(start//86_400_000, 0.0)+value["stress_3pct"]
                overlap[start] += 1
                price = (( _value(row, "quote_snapshot", {}) or {}).get("ladder", {}).get(amount, {})).get("average_price", 0)
                bucket = "<0.10" if price < .1 else ("0.10–0.25" if price < .25 else "≥0.25")
                item = buckets.setdefault(bucket, {"n": 0, "pnl": 0.0})
                item["n"] += 1
                item["pnl"] += value["pnl"]
            wins = sorted([value["pnl"] for _, value in usable if value["pnl"] > 0], reverse=True)
            missing = Counter()
            for row, values in group:
                if amount in values:
                    continue
                if _value(row, "strategy_eligible") is not True:
                    missing["strategy_rejected_or_unknown"] += 1
                elif not (_value(row, "decision_snapshot", {}) or {}).get("shadow_capacity", {}).get("accepted"):
                    missing["capacity_rejected_or_unknown"] += 1
                elif _value(row, "theoretical_outcome") not in {"UP", "DOWN"}:
                    missing["unsettled"] += 1
                else:
                    missing["quote_missing_stale_or_guard_rejected"] += 1
            clusters = list(daily.values())
            from statistics import mean, stdev
            evidence_series.append({"signal_version": version, "policy_version": policy, "amount": amount,
                "n": len(usable), "pnl": cumulative, "max_drawdown": drawdown,
                "without_top_win_pnl": cumulative-(wins[0] if wins else 0),
                "top_win_positive_share": wins[0]/sum(wins) if wins else None,
                "price_buckets": buckets, "missing": dict(missing), "overlapping_windows": sum(n>1 for n in overlap.values()),
                "independent_windows": len(overlap), "independent_days": len(daily),
                "daily_mean_lower_99pct": mean(clusters)-2.58*stdev(clusters)/len(clusters)**.5 if len(clusters)>=14 else None})
    ladder = {}  # Compatibility field: mixed-version totals deliberately unavailable.
    pairs = {}
    for row, value in estimates:
        key = (_value(row, "target_window_start"), _value(row, "direction"))
        pairs.setdefault(key, {})[_value(row, "signal_version")] = value
    paired = [pair for pair in pairs.values() if pair.get("firsthit_control") and pair.get("firsthit_recovery")]
    result["forward_evidence"] = {
        "cohort": "按series.policy_version独立分组", "profile": PROFILE, "live_n": len(live),
        "top_level_observation_only": True,
        "cohorts": sorted({str(_value(row, "policy_version")) for row in forward}),
        "overflow_counts": dict(__import__("binance_predict.services.shadow_forward_evidence", fromlist=["OVERFLOW_COUNTS"]).OVERFLOW_COUNTS),
        "archive_n": len(forward)-len(live), "independent_windows": len(windows),
        "independent_days": len(days), "ladder": ladder, "series": evidence_series,
        "pair_details": [{"window_start": key[0], "direction": key[1],
                          "versions": sorted(pair), "missing_versions": [v for v in
                          ("firsthit_control", "firsthit_recovery") if not pair.get(v)]}
                         for key, pair in sorted(pairs.items())],
        "paired_same_window_direction": len(paired), "pair_missing": len(pairs)-len(paired),
        "eligible": False, "auto_enable": False,
        "blocking_reasons": ["费用仍为假设，报价不证明成交", "运营门禁与全家族实时等价尚未覆盖"] +
            (["独立窗口不足"] if len(windows) < PROFILE["minimum_windows"] else []) +
            (["独立交易日不足"] if len(days) < PROFILE["minimum_days"] else []),
    }
    # Registry coverage is independent of observed samples: missing families remain visible.
    result["coverage_matrix"] = [
        {
            "signal_version": version, "family": spec.family,
            "market_period": spec.market_period, "retired": spec.live_retired,
            "rows": sum(_value(row, "signal_version") == version for row in rows_list),
            "forward_admission": False,
            "live_frozen_n": sum(_value(row, "signal_version") == version and
                (_value(row, "input_snapshot", {}) or {}).get("capture_mode") == "LIVE" and
                str(_value(row, "policy_version", "")).startswith("forward-v2-") for row in rows_list),
            "archive_or_legacy_n": sum(_value(row, "signal_version") == version and not
                str(_value(row, "policy_version", "")).startswith("forward-v2-") for row in rows_list),
            "capture_capability": "LIVE_ENTRY" if version in {
                "firsthit_down_chg_v2", "absorption_follow_td120_v1", "absorption_follow_td150_v1"
            } else ("LIVE_ENTRY" if any(_value(row, "signal_version") == version for row in live)
                    else "ARCHIVE_OR_LEGACY_ONLY"),
            "blocking_reasons": (["费用与成交未验证，实时入口不等于准入"] if version in {
                "firsthit_down_chg_v2", "absorption_follow_td120_v1", "absorption_follow_td150_v1"
            } or any(_value(row, "signal_version") == version for row in live)
                else ["该版本尚无已声明实时等价适配，归档阻塞；G3不能代表其他首触版本"]),
        }
        for version, spec in SHADOW_VERSION_SPECS.items()
        if not spec.live_retired
    ]
    result["admission"] = {
        "eligible": False, "auto_enable": False,
        "health_is_admission": False,
        "blocking_reasons": ["历史统一账本不能替代新 cohort 前向证据"],
    }
    return result


async def build_execution_comparison(
    session: AsyncSession, *, policy_version: str | None = None,
    from_ts: int | None = None, to_ts: int | None = None,
    market_period: str | None = None, include_retired: bool = False,
    signal_versions: set[str] | None = None,
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
    if signal_versions is not None:
        stmt = stmt.where(ShadowExecutionAssessment.signal_version.in_(signal_versions))
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
    result = build_execution_comparison_from_rows(rows, orders, scope=scope)
    from .shadow_forward_evidence import COHORT
    forward_stmt = select(ShadowExecutionAssessment).where(
        ShadowExecutionAssessment.policy_version.like("forward-v2-%"))
    if signal_versions is not None:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.signal_version.in_(signal_versions))
    if policy_version is not None:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.policy_version == policy_version)
    if not include_retired:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.retired.is_not(True))
    if from_ts is not None:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.target_window_start >= from_ts)
    if to_ts is not None:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.target_window_start < to_ts)
    if market_period is not None:
        forward_stmt = forward_stmt.where(ShadowExecutionAssessment.market_period == market_period)
    forward_rows = list((await session.execute(forward_stmt)).scalars().all())
    forward_result = build_execution_comparison_from_rows(forward_rows, [], scope=scope)
    result["forward_evidence"] = forward_result["forward_evidence"]
    result["coverage_matrix"] = build_execution_comparison_from_rows(
        rows + [row for row in forward_rows if row not in rows], [], scope=scope)["coverage_matrix"]
    return result
