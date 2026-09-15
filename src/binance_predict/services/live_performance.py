"""Pure statistics for evaluating live prediction-market orders."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from scipy.stats import beta

from binance_predict.backtest.stats import wilson
from binance_predict.services.live_channel_benchmarks import get_channel_benchmark
from binance_predict.services.live_channels import exclusive_group

WEI = 10**18


@dataclass(frozen=True)
class LiveOrder:
    """Small, storage-agnostic representation of a live order."""

    id: int | str | None = None
    channel: str | None = None
    market_id: int | str | None = None
    assessment_id: int | str | None = None
    market_period: str | None = None
    window_start: int | str | None = None
    window_time: int | None = None
    settled_at: int | None = None
    created_at: int | None = None
    direction: str | None = None
    settle_outcome: str | None = None
    win: bool | None = None
    pnl: float | None = None
    amount_in: str | int | float | None = None
    quote_json: Mapping[str, Any] | None = None
    status: str | None = None
    policy_version: str | None = None
    trigger_ts: int | None = None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) and parsed > 0 else None


def _quote_amount_usdt(value: Any) -> float | None:
    """Convert quote_json.amountIn from its fixed wei contract to USDT."""
    amount = _positive_float(value)
    return amount / WEI if amount is not None else None


def actual_cost(order: LiveOrder) -> float | None:
    quote = order.quote_json or {}
    if "amountIn" in quote:
        amount = _quote_amount_usdt(quote.get("amountIn"))
        if amount is not None:
            return amount
    fallback = _positive_float(order.amount_in)
    return fallback / WEI if fallback is not None else None


def execution_details(order: LiveOrder) -> dict[str, Any]:
    quote = order.quote_json or {}
    return {
        "average_price": _positive_float(quote.get("averagePrice")),
        "quoted_average_price": _positive_float(quote.get("quotedAvgPrice")),
        "fill_source": quote.get("fillSource"),
        "filled_share_qty": _positive_float(quote.get("filledShareQty")),
    }


def break_even(order: LiveOrder) -> dict[str, Any]:
    details = execution_details(order)
    quote = order.quote_json or {}
    cost = actual_cost(order)
    shares = details["filled_share_qty"]
    if cost is not None and shares is not None:
        return {
            "break_even_probability": cost / shares,
            "quality": "CAPTURED_FILL",
            "method": "ACTUAL_AMOUNT_OVER_SHARES",
        }
    confirmed_average = (
        details["fill_source"] == "binance_history_confirm"
        or (
            quote.get("source") == "binance_history_sync"
            and quote.get("binanceOrderStatus") == "FILLED"
        )
    )
    if details["average_price"] is not None and confirmed_average:
        return {
            "break_even_probability": details["average_price"] / 0.98,
            "quality": "CAPTURED_FILL_ESTIMATED_FEE",
            "method": "FEE_ADJUSTED_ACTUAL_AVERAGE_PRICE",
        }
    return {
        "break_even_probability": None,
        "quality": "LEGACY_UNKNOWN",
        "method": "LEGACY_UNKNOWN",
    }


def market_window_key(order: LiveOrder) -> tuple[Any, ...] | None:
    if order.market_id is not None and str(order.market_id).strip():
        return ("market_id", order.market_id)
    if order.market_period and order.window_start is not None:
        return ("period_window", order.market_period, order.window_start)
    return None


def normalize_order(order: LiveOrder) -> dict[str, Any]:
    result = asdict(order)
    result["actual_cost"] = actual_cost(order)
    result.update(execution_details(order))
    result.update(break_even(order))
    key = market_window_key(order)
    result["market_window_key"] = list(key) if key is not None else None
    return result


def rolling_binary_stats(values: Iterable[bool | None], windows: tuple[int, ...] = (20, 50)) -> dict[str, Any]:
    valid = [bool(value) for value in values if value is not None]
    result: dict[str, Any] = {}
    for size in windows:
        sample = valid[-size:]
        n = len(sample)
        wins = sum(sample)
        rate = wins / n if n else None
        wilson_interval = wilson(rate, n) if rate is not None else None
        jeffreys = (
            (float(beta.ppf(0.025, wins + 0.5, n - wins + 0.5)),
             float(beta.ppf(0.975, wins + 0.5, n - wins + 0.5)))
            if n else None
        )
        result[str(size)] = {
            "window_size": size,
            "n": n,
            "wins": wins,
            "full_window": n == size,
            "win_rate": rate,
            "wilson_95": list(wilson_interval) if wilson_interval else None,
            "jeffreys_95": list(jeffreys) if jeffreys else None,
        }
    return result


def ewma(values: Iterable[float | None], span: int = 20) -> list[float | None]:
    alpha = 2.0 / (span + 1)
    current: float | None = None
    output: list[float | None] = []
    for value in values:
        if value is not None:
            current = float(value) if current is None else alpha * float(value) + (1 - alpha) * current
        output.append(current)
    return output


def rolling_realized_ev(orders: Iterable[LiveOrder], window: int) -> dict[str, Any]:
    sample = list(orders)[-window:]
    pairs = [(order.pnl, actual_cost(order)) for order in sample]
    usable = [(float(pnl), cost) for pnl, cost in pairs if pnl is not None and cost is not None]
    total_cost = sum(cost for _, cost in usable)
    return {
        "window_size": window,
        "n": len(usable),
        "full_window": len(usable) == window,
        "realized_ev": sum(pnl for pnl, _ in usable) / total_cost if total_cost > 0 else None,
        "pnl": sum(pnl for pnl, _ in usable) if usable else None,
        "actual_cost": total_cost if usable else None,
    }


def _sort_component(value: Any) -> tuple[int, float | str]:
    if value is None:
        return (2, "")
    try:
        return (0, float(value))
    except (TypeError, ValueError, OverflowError):
        return (1, str(value))


def _order_win(order: LiveOrder) -> bool | None:
    if order.win is not None:
        return bool(order.win)
    if order.direction and order.settle_outcome in {"UP", "DOWN"}:
        return order.direction == order.settle_outcome
    return None


def channel_series(orders: Iterable[LiveOrder]) -> list[dict[str, Any]]:
    """Build plotting-ready per-order statistics in one bounded-memory pass."""
    rows = sorted(
        orders,
        key=lambda row: (_sort_component(row.window_start), _sort_component(row.id)),
    )
    win_windows = {size: deque(maxlen=size) for size in (20, 50)}
    order_windows = {size: deque(maxlen=size) for size in (20, 50)}
    break_even_windows = {size: deque(maxlen=size) for size in (20, 50)}
    cumulative_pnl = 0.0
    cumulative_cost = 0.0
    has_pnl = has_cost = False
    ewma_value: float | None = None
    alpha = 2.0 / 21.0
    output: list[dict[str, Any]] = []

    for row in rows:
        win = _order_win(row)
        cost = actual_cost(row)
        probability = break_even(row)["break_even_probability"]
        if row.pnl is not None:
            cumulative_pnl += float(row.pnl)
            has_pnl = True
        if cost is not None:
            cumulative_cost += cost
            has_cost = True
        if win is not None:
            numeric_win = float(win)
            ewma_value = numeric_win if ewma_value is None else alpha * numeric_win + (1 - alpha) * ewma_value
        for size in (20, 50):
            win_windows[size].append(win)
            order_windows[size].append(row)
            break_even_windows[size].append(probability)

        rolling = {
            str(size): rolling_binary_stats(win_windows[size], (size,))[str(size)]
            for size in (20, 50)
        }
        realized_ev = {
            str(size): rolling_realized_ev(order_windows[size], size)
            for size in (20, 50)
        }
        mean_break_even = {
            str(size): (
                sum(value for value in break_even_windows[size] if value is not None)
                / sum(value is not None for value in break_even_windows[size])
                if any(value is not None for value in break_even_windows[size]) else None
            )
            for size in (20, 50)
        }
        benchmark = get_channel_benchmark(row.channel or "")
        output.append({
            "id": row.id,
            "channel": row.channel,
            "market_id": row.market_id,
            "assessment_id": row.assessment_id,
            "window_start": row.window_start,
            "window_time": row.window_time,
            "t": row.window_start,
            "settled_at": row.settled_at,
            "created_at": row.created_at,
            "pnl": float(row.pnl) if row.pnl is not None else None,
            "cost": cost,
            "win": win,
            "direction": row.direction,
            "settle_outcome": row.settle_outcome,
            "policy_version": row.policy_version,
            "trigger_ts": row.trigger_ts,
            "trigger_offset_seconds": (
                (row.trigger_ts - int(row.window_start)) / 1000
                if row.trigger_ts is not None and row.window_start is not None else None
            ),
            "cumulative_pnl": cumulative_pnl if has_pnl else None,
            "cumulative_cost": cumulative_cost if has_cost else None,
            "rolling": rolling,
            "ewma_win_rate": ewma_value,
            "rolling_realized_ev": realized_ev,
            "break_even_probability": probability,
            "rolling_mean_break_even_probability": mean_break_even,
            "benchmark_win_rate": benchmark.win_rate if benchmark else None,
            "benchmark_gap": {
                str(size): (
                    rolling[str(size)]["win_rate"] - benchmark.win_rate
                    if benchmark and rolling[str(size)]["win_rate"] is not None else None
                )
                for size in (20, 50)
            },
        })
    return output


def aggregate_windows(orders: Iterable[LiveOrder]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[LiveOrder]] = {}
    for order in orders:
        key = market_window_key(order)
        if key is not None:
            grouped.setdefault(key, []).append(order)
    output = []
    for key, rows in grouped.items():
        pnls = [float(row.pnl) for row in rows if row.pnl is not None]
        costs = [cost for row in rows if (cost := actual_cost(row)) is not None]
        outcomes = {row.settle_outcome for row in rows if row.settle_outcome in {"UP", "DOWN"}}
        conflict = len(outcomes) > 1
        net_pnl = sum(pnls) if pnls else None
        window_starts = [int(row.window_start) for row in rows if row.window_start is not None]
        periods = {row.market_period for row in rows if row.market_period}
        output.append({
            "market_window_key": list(key),
            "window_start": min(window_starts) if window_starts else None,
            "market_period": next(iter(periods)) if len(periods) == 1 else None,
            "order_count": len(rows),
            "net_pnl": net_pnl,
            "gross_cost": sum(costs) if costs else None,
            "settle_outcome": next(iter(outcomes)) if len(outcomes) == 1 else None,
            "settlement_conflict": conflict,
            "window_win": (net_pnl > 0) if net_pnl is not None and not conflict else None,
        })
    return output


def portfolio_window_series(orders: Iterable[LiveOrder]) -> list[dict[str, Any]]:
    """Build chronological unique-window points without dropping duplicate capital."""
    windows = aggregate_windows(orders)
    windows.sort(key=lambda row: (
        _sort_component(row["window_start"]),
        tuple(str(part) for part in row["market_window_key"]),
    ))
    cumulative = 0.0
    output: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        if window["net_pnl"] is not None:
            cumulative += window["net_pnl"]
        point = dict(window)
        point["t"] = window["window_start"]
        point["cumulative_pnl"] = cumulative
        for size in (20, 50):
            sample = windows[max(0, index - size + 1):index + 1]
            usable = [row for row in sample if row["net_pnl"] is not None and row["gross_cost"] is not None]
            cost = sum(row["gross_cost"] for row in usable)
            point[f"rolling_realized_ev_{size}"] = (
                sum(row["net_pnl"] for row in usable) / cost if cost > 0 else None
            )
        output.append(point)
    return output


def _segment_label(order: LiveOrder, segment_by: str) -> str:
    if segment_by == "quote":
        probability = break_even(order)["break_even_probability"]
        if probability is None:
            return "LEGACY_UNKNOWN"
        index = min(9, max(0, int(probability * 10)))
        return f"[{index / 10:.1f},{(index + 1) / 10:.1f}{']' if index == 9 else ')'}"
    if segment_by == "trigger_offset":
        if order.trigger_ts is None or order.window_start is None:
            return "LEGACY_UNKNOWN"
        seconds = (order.trigger_ts - int(order.window_start)) / 1000
        if seconds < 60:
            return "<60"
        if seconds < 300:
            start = int(seconds // 60) * 60
            return f"{start}-{start + 59}"
        return ">=300"
    if segment_by == "policy_version":
        return order.policy_version or "LEGACY_UNKNOWN"
    return "LEGACY_UNKNOWN"


def segment_orders(orders: Iterable[LiveOrder], segment_by: str) -> list[dict[str, Any]]:
    """Aggregate one diagnostic dimension; regime/deployment facts are unknown in P0."""
    grouped: dict[str, list[LiveOrder]] = {}
    for order in orders:
        grouped.setdefault(_segment_label(order, segment_by), []).append(order)
    output = []
    for label, rows in grouped.items():
        valid_wins = [win for row in rows if (win := _order_win(row)) is not None]
        costs = [cost for row in rows if (cost := actual_cost(row)) is not None]
        pnls = [float(row.pnl) for row in rows if row.pnl is not None]
        total_cost = sum(costs)
        output.append({
            "segment": label,
            "n": len(rows),
            "wins": sum(valid_wins),
            "win_rate": sum(valid_wins) / len(valid_wins) if valid_wins else None,
            "total_cost": total_cost,
            "total_pnl": sum(pnls),
            "realized_ev": sum(pnls) / total_cost if total_cost > 0 else None,
            "unique_window_count": len({key for row in rows if (key := market_window_key(row)) is not None}),
            "data_quality": "LEGACY_UNKNOWN" if label == "LEGACY_UNKNOWN" else "CAPTURED",
        })
    output.sort(key=lambda row: (row["segment"] == "LEGACY_UNKNOWN", row["segment"]))
    return output


def duplicate_risk(orders: Iterable[LiveOrder]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[LiveOrder]] = {}
    for order in orders:
        key = market_window_key(order)
        if key is not None:
            grouped.setdefault(key, []).append(order)
    windows = list(grouped.values())
    duplicate = [rows for rows in windows if len(rows) > 1]
    all_costs = [cost for rows in windows for row in rows if (cost := actual_cost(row)) is not None]
    excess = 0.0
    max_window_cost: float | None = None
    same_direction = opposite_direction = explicit_conflicts = 0
    max_channels = 0
    for rows in windows:
        costs = [cost for row in rows if (cost := actual_cost(row)) is not None]
        gross = sum(costs)
        if costs:
            excess += gross - max(costs)
            max_window_cost = gross if max_window_cost is None else max(max_window_cost, gross)
        channels = {row.channel for row in rows if row.channel}
        max_channels = max(max_channels, len(channels))
        if len(rows) > 1:
            directions = {row.direction for row in rows if row.direction}
            if len(directions) == 1:
                same_direction += 1
            elif len(directions) > 1:
                opposite_direction += 1
            groups = [exclusive_group(channel) for channel in channels]
            if any(group and len(channels.intersection(group)) > 1 for group in groups):
                explicit_conflicts += 1
    total_cost = sum(all_costs)
    return {
        "window_count": len(windows),
        "duplicate_window_count": len(duplicate),
        "duplicate_window_ratio": len(duplicate) / len(windows) if windows else None,
        "duplicate_overlap_cost": excess if all_costs else None,
        "excess_overlap_cost": excess if all_costs else None,
        "duplicate_investment_ratio": excess / total_cost if total_cost > 0 else None,
        "max_single_window_cost": max_window_cost,
        "max_channels_per_window": max_channels if windows else None,
        "same_direction_overlap_count": same_direction,
        "opposite_direction_overlap_count": opposite_direction,
        "explicit_exclusive_conflict_count": explicit_conflicts,
    }


def analyze_live_performance(
    orders: Iterable[LiveOrder], *, include_channel_series: bool = True,
) -> dict[str, Any]:
    rows = list(orders)
    windows = aggregate_windows(rows)
    window_wins = [row["window_win"] for row in windows]
    report = {
        "orders": [normalize_order(row) for row in rows],
        "order_stats": rolling_binary_stats(_order_win(row) for row in rows),
        "window_stats": rolling_binary_stats(window_wins),
        "windows": windows,
        "rolling_realized_ev": {str(size): rolling_realized_ev(rows, size) for size in (20, 50)},
        "duplicate_risk": duplicate_risk(rows),
    }
    if include_channel_series:
        report["channel_series"] = channel_series(rows)
    return report
