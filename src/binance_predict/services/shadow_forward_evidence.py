"""Record-only frozen forward evidence; no order or balance mutations."""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

from sqlalchemy import select

from binance_predict.db.models import ShadowExecutionAssessment
from .live_execution_policy import config_fingerprint, evaluate_quote_execution_policy
from .shadow_execution_store import get_or_create_runtime_assessment

PROFILE = {"revision": "forward-v1", "order_type": "MARKET", "max_exec_price": 0.50,
           "amounts": [1, 5, 10], "fee_assumption": 0.02, "holding": "settlement_only",
           "minimum_windows": 100, "minimum_days": 14, "quote_max_age_ms": 3000}
PROFILE_HASH = config_fingerprint(PROFILE)
COHORT = "forward-v1-" + PROFILE_HASH[:16]
# ponytail: one process, bounded queue; multi-worker deployments need a shared limiter.
class QuoteBudget:
    """Bound waiting ladders and wall time; rejected work is recorded, not queued."""
    def __init__(self):
        self.pending = 0
        self.lock = asyncio.Semaphore(1)

    async def __aenter__(self):
        if self.pending >= 8:
            raise RuntimeError("quote_queue_full")
        self.pending += 1
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=2)
        except BaseException:
            self.pending -= 1
            raise
        return self

    async def __aexit__(self, *args):
        self.lock.release()
        self.pending -= 1


QUOTE_QUEUE = QuoteBudget()


def capture_event(snapshot, version):
    """Synchronous trigger-time copy, before persistence or quote awaits."""
    from copy import deepcopy
    from .shadow_execution_registry import SHADOW_VERSION_SPECS
    spec = SHADOW_VERSION_SPECS.get(version)
    config = (_runtime_configs() if _runtime_configs else {}).get(spec.live_channel if spec else version)
    profile = {**PROFILE, "signal_version": version, "amount_usdt": current_amount(version),
               "runtime_rules": {name: getattr(config, name, None) for name in
                                 ("order_type", "max_exec_price", "max_daily_orders")}}
    fingerprint = config_fingerprint(profile)
    return {**deepcopy(snapshot), "_frozen": {"profile": profile, "hash": fingerprint,
        "cohort": "forward-v2-"+fingerprint[:16],
        "operations": deepcopy(read_constraints(version, snapshot["window_start"]))}}


async def freeze_event(session, snapshot, capture_mode, quotes, version, selected, period="5m"):
    snapshot = snapshot if "_frozen" in snapshot else capture_event(snapshot, version)
    frozen = snapshot["_frozen"]
    row = await get_or_create_runtime_assessment(session, source_type="forward",
        signal_version=version, target_window_start=snapshot["window_start"], market_period=period,
        direction=snapshot["side"], policy_version=frozen["cohort"] + "-" + snapshot["side"])
    row = (await session.execute(select(ShadowExecutionAssessment).where(
        ShadowExecutionAssessment.id == row.id).with_for_update())).scalar_one()
    if row.config_snapshot:
        return
    row.config_fingerprint = frozen["hash"]
    row.config_snapshot = {key: frozen[key] for key in ("cohort", "profile", "hash")}
    row.input_snapshot = {"capture_mode": capture_mode, "frozen_at": int(time.time()*1000), "event": snapshot}
    row.quote_snapshot = {"ladder": quotes, "fill": "NOT_OBSERVED", "fees": "ASSUMPTION"}
    row.strategy_eligible = selected
    row.trigger_ts = snapshot["trigger_ts"]
    constraints = frozen["operations"]
    row.operational_eligible = constraints["eligible"]
    row.decision_snapshot = {"pure_profile": True, "operational_constraints": constraints,
        "shadow_assumptions": {"balance_usdt": 100, "daily_limit": 20, "exclusive": "per_version_window"},
        "real_state": {"balance": None, "daily_count": None, "exclusive_blocker": None},
        "assumptions_are_real_state": False}
    row.reason_code = "FORWARD_QUOTE_ONLY"
    await shadow_capacity(session, row)


async def attach_quotes(session, snapshot, version, quotes):
    row = (await session.execute(select(ShadowExecutionAssessment).where(
        ShadowExecutionAssessment.signal_version == version,
        ShadowExecutionAssessment.target_window_start == snapshot["window_start"],
        ShadowExecutionAssessment.policy_version == snapshot["_frozen"]["cohort"] + "-" + snapshot["side"]
    ).with_for_update())).scalar_one()
    if not (row.quote_snapshot or {}).get("ladder"):
        row.quote_snapshot = {"ladder": quotes, "fill": "NOT_OBSERVED", "fees": "ASSUMPTION"}


_FORWARD_TASKS = set()
_runtime_configs = None
_quote_client = None


_operational_reader = None


def configure_forward(*, configs, quote_client, operational_reader=None):
    global _runtime_configs, _quote_client, _operational_reader
    _runtime_configs = configs
    _quote_client = quote_client
    _operational_reader = operational_reader


_daily_counts = {}
_daily_tasks = {}


async def _refresh_daily(channel, counter, day):
    try:
        count = await counter(channel)
        _daily_counts[channel] = (day, time.monotonic(), count)
    except Exception:
        _daily_counts.pop(channel, None)
    finally:
        _daily_tasks.pop(channel, None)


def cached_daily_count(channel, counter):
    day = int(time.time()) // 86400
    cached = _daily_counts.get(channel)
    if (not cached or cached[0] != day or time.monotonic()-cached[1] >= 15) and channel not in _daily_tasks:
        _daily_tasks[channel] = asyncio.create_task(_refresh_daily(channel, counter, day))
    return cached[2] if cached and cached[0] == day and time.monotonic()-cached[1] < 15 else None


def read_constraints(version, start):
    inputs = _operational_reader(version, start) if _operational_reader else {}
    return operational_constraints(amount=current_amount(version), **inputs)


async def shadow_capacity(session, row):
    """Derive hypothetical capacity from committed ledger, never real reservations."""
    day = row.target_window_start // 86_400_000
    # Serialize shadow-only accounting; restart restores it from ledger rows.
    from sqlalchemy import text
    import hashlib
    key = int.from_bytes(hashlib.sha256((COHORT+row.signal_version+row.market_period).encode()).digest()[:8], "big", signed=True)
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
    others = (await session.execute(select(ShadowExecutionAssessment).where(
        ShadowExecutionAssessment.policy_version.like(row.config_snapshot["cohort"] + "%"),
        ShadowExecutionAssessment.signal_version == row.signal_version,
        ShadowExecutionAssessment.market_period == row.market_period,
        ShadowExecutionAssessment.target_window_start >= day*86_400_000,
        ShadowExecutionAssessment.target_window_start < (day+1)*86_400_000,
        ShadowExecutionAssessment.id != row.id))).scalars().all()
    accepted = [other for other in others if (other.decision_snapshot or {}).get("shadow_capacity", {}).get("accepted")]
    spent = sum((other.decision_snapshot or {})["shadow_capacity"]["amount_usdt"] for other in accepted)
    amount = row.config_snapshot["profile"].get("amount_usdt") or 1.0
    duplicate = any(other.target_window_start == row.target_window_start for other in accepted)
    reason = "DUPLICATE_WINDOW" if duplicate else ("DAILY_CAPACITY_REACHED" if len(accepted) >= 20 else
             ("INSUFFICIENT_BALANCE" if spent+amount > 100 else None))
    decision = dict(row.decision_snapshot or {})
    decision["shadow_capacity"] = {"accepted": row.strategy_eligible is True and reason is None,
        "reason": reason, "amount_usdt": amount, "day": day, "count_before": len(accepted),
        "balance_before": 100-spent, "assumption": True, "real_slot_reserved": False}
    row.decision_snapshot = decision


def operational_constraints(*, amount, available_balance=None, daily_count=None,
                            max_daily_orders=None, exclusive_blocker=None, exclusive_known=False):
    reasons = []
    unknown = []
    if amount is None or not math.isfinite(amount) or amount <= 0:
        reasons.append("INVALID_AMOUNT")
    if available_balance is None:
        unknown.append("BALANCE_UNKNOWN")
    elif amount is not None and available_balance < amount:
        reasons.append("INSUFFICIENT_BALANCE")
    if daily_count is None or max_daily_orders is None:
        unknown.append("DAILY_CAPACITY_UNKNOWN")
    elif daily_count >= max_daily_orders:
        reasons.append("DAILY_CAPACITY_REACHED")
    if not exclusive_known:
        unknown.append("EXCLUSIVE_STATE_UNKNOWN")
    elif exclusive_blocker:
        reasons.append("EXCLUSIVE_BLOCKED")
    return {"eligible": False if reasons else (None if unknown else True),
            "reasons": reasons, "missing": unknown, "amount_usdt": amount,
            "available_balance": available_balance, "daily_count": daily_count,
            "max_daily_orders": max_daily_orders, "exclusive_blocker": exclusive_blocker,
            "mutates_live": False}


def current_amount(version):
    from .shadow_execution_registry import SHADOW_VERSION_SPECS
    spec = SHADOW_VERSION_SPECS.get(version)
    channel = spec.live_channel if spec else version
    configs = _runtime_configs() if _runtime_configs else {}
    config = configs.get(channel)
    amount = getattr(config, "amount_usdt", None)
    return float(amount) if isinstance(amount, (int, float)) and math.isfinite(amount) and amount > 0 else None


_quote_cache = {}
_live_busy = None


async def probe_ladder(version, side, start, period, *, amount=None):
    client = _quote_client
    if client is None:
        return {"error": {"available": False, "reason": "quote_client_unavailable"}}
    result = {}
    if _live_busy and _live_busy():
        return {"error": {"available": False, "reason": "live_priority_backpressure"}}
    try:
        async with QUOTE_QUEUE:
            async with asyncio.timeout(4):
                if not getattr(client, "_wallet_address", ""):
                    await client.fetch_wallet_info()
                await client.list_markets()
                if period == "15m":
                    market = client._15m_markets.get(start, {})
                    token = market.get("up_token" if side == "UP" else "down_token")
                else:
                    token = getattr(client, "_up_token_id" if side == "UP" else "_down_token_id", None)
                    if getattr(client, "_5m_start_date", None) != start:
                        token = None
                if not token:
                    return {"error": {"available": False, "reason": "market_mismatch"}}
                amount = amount if amount is not None else current_amount(version)
                if amount is None:
                    result["error"] = {"available": False, "reason": "runtime_amount_unknown"}
                for stake in sorted(set([1., 5., 10.] + ([amount] if amount else []))):
                    if _live_busy and _live_busy():
                        result["error"] = {"available": False, "reason": "live_priority_backpressure"}
                        break
                    key = (period, start, side, token, stake)
                    cached = _quote_cache.get(key)
                    if cached and time.monotonic()-cached[0] < .5:
                        result[str(int(stake)) if stake == int(stake) else str(stake)] = dict(cached[1])
                        continue
                    requested = int(time.time()*1000)
                    quote = await client.get_quote(token, "BUY", amount_usdt=stake)
                    received = int(time.time()*1000)
                    try:
                        price = float((quote or {}).get("averagePrice"))
                    except (TypeError, ValueError):
                        price = None
                    result[str(int(stake)) if stake == int(stake) else str(stake)] = {
                        "amount_usdt": stake, "average_price": price,
                        "available": price is not None and math.isfinite(price) and 0 < price < 1,
                        "requested_at": requested, "received_at": received,
                        "latency_ms": received-requested, "token_id": token,
                        "amount_out": (quote or {}).get("amountOut")}
                    _quote_cache[key] = (time.monotonic(), dict(result[str(int(stake)) if stake == int(stake) else str(stake)]))
                    if len(_quote_cache) > 128:
                        _quote_cache.clear()
    except Exception as exc:
        result["error"] = {"available": False, "reason": type(exc).__name__ + ":" + str(exc)}
    return result


OVERFLOW_COUNTS = {}


def observe_kline(payload):
    """Bound all persistence tasks; observation failures never escape to live hooks."""
    version = str(payload.get("version") or payload.get("channel") or "unknown")
    try:
        if len(_FORWARD_TASKS) >= 32:
            OVERFLOW_COUNTS[version] = OVERFLOW_COUNTS.get(version, 0)+1
            return
        event = dict(payload)
        event["window_start"] = event.get("market_start", event.get("market_start_15m"))
        event["trigger_ts"] = int(time.time()*1000)
        event = capture_event(event, version)
        task = asyncio.create_task(_freeze_kline(event))
        _FORWARD_TASKS.add(task)
        task.add_done_callback(_FORWARD_TASKS.discard)
    except Exception:
        OVERFLOW_COUNTS[version] = OVERFLOW_COUNTS.get(version, 0)+1


async def _freeze_kline(payload):
    from binance_predict.db.engine import async_session_factory
    from .shadow_execution_registry import SHADOW_VERSION_SPECS
    from loguru import logger
    try:
        version = payload.get("version") or payload.get("channel")
        payload = dict(payload)
        payload.setdefault("market_start", payload.get("market_start_15m"))
        payload.setdefault("direction", "UP" if payload.get("side") == "low" else "DOWN")
        spec = SHADOW_VERSION_SPECS.get(version)
        if spec is None:
            return
        start = int(payload["market_start"])
        period_ms = 300_000 if spec.market_period == "5m" else 900_000
        now = int(time.time()*1000)
        if not start <= now < start+period_ms:
            return
        snapshot = {**payload, "window_start": start, "window_end": start+period_ms,
                    "trigger_ts": payload["trigger_ts"], "side": payload["direction"]}
        async with async_session_factory() as session:
            await freeze_event(session, snapshot, "LIVE", {}, version, True, spec.market_period)
            await session.commit()
        quotes = ({"error": {"available": False, "reason": "task_capacity_exceeded"}}
                  if payload.get("_overloaded") else
                  await probe_ladder(version, payload["direction"], start, spec.market_period,
                                     amount=snapshot["_frozen"]["profile"]["amount_usdt"]))
        async with async_session_factory() as session:
            await attach_quotes(session, snapshot, version, quotes)
            await session.commit()
    except Exception as exc:
        logger.warning("K线前向冻结失败（不影响交易）| {}", exc)


async def freeze_firsthit(session, snapshot, capture_mode, quotes):
    """Separate policy key isolates legacy rows; first write freezes the event."""
    for version, selected in (("firsthit_control", True),
                              ("firsthit_recovery", snapshot.get("recovery_pass") is True)):
        event = snapshot.get("_evidence", {}).get(version) or capture_event(snapshot, version)
        await freeze_event(session, event, capture_mode, {}, version, selected)
        if quotes and capture_mode == "LIVE":
            await attach_quotes(session, event, version, quotes)


def valid_amount_out(value):
    try:
        from decimal import Decimal
        amount = Decimal(str(value))
        return amount.is_finite() and amount > 0 and amount == amount.to_integral_value()
    except Exception:
        return False


def quote_returns(row):
    """Amount-weighted settlement estimate, never a fill or actual PnL."""
    snapshot = row if isinstance(row, dict) else vars(row)
    config = snapshot.get("config_snapshot") or {}
    inputs = snapshot.get("input_snapshot") or {}
    profile = config.get("profile") or {}
    if config.get("hash") != config_fingerprint(profile) or inputs.get("capture_mode") != "LIVE" or not (
        (snapshot.get("decision_snapshot") or {}).get("shadow_capacity", {}).get("accepted")):
        return {}
    outcome = snapshot.get("theoretical_outcome")
    if outcome not in {"UP", "DOWN"} or snapshot.get("strategy_eligible") is not True:
        return {}
    trigger = snapshot.get("trigger_ts")
    results = {}
    for amount, quote in ((snapshot.get("quote_snapshot") or {}).get("ladder") or {}).items():
        price = quote.get("average_price")
        stake = quote.get("amount_usdt")
        received = quote.get("received_at")
        if not (quote.get("available") and isinstance(price, (int, float)) and
                math.isfinite(price) and 0 < price < 1 and isinstance(stake, (int, float)) and
                math.isfinite(stake) and stake > 0 and trigger and received and
                0 <= received-trigger <= profile["quote_max_age_ms"]):
            continue
        gate = evaluate_quote_execution_policy(order_type="MARKET", avg_price=price,
                                                max_exec_price=profile.get("runtime_rules", {}).get("max_exec_price") or profile["max_exec_price"])
        if not gate.eligible:
            continue
        won = outcome == snapshot.get("direction")
        def pnl(extra):
            stressed = min(1.0, price*(1+extra))
            return stake*((1-profile["fee_assumption"])/stressed-1) if won else -stake
        results[amount] = {"stake": stake, "pnl": pnl(0), "stress_1pct": pnl(.01),
                           "stress_3pct": pnl(.03), "fill": False,
                           "capacity_passed_for_amount": stake == profile.get("amount_usdt"),
                           "counterfactual_amount": stake != profile.get("amount_usdt"),
                           "amount_out_valid": valid_amount_out(quote.get("amount_out")),
                           "admission": False, "fees_verified": False}
    return results


async def settle_firsthit(session, window_start, outcome, market_period="5m", signal_version=None):
    if outcome not in {"UP", "DOWN"}:
        return
    rows = (await session.execute(select(ShadowExecutionAssessment).where(
        ShadowExecutionAssessment.target_window_start == window_start,
        ShadowExecutionAssessment.market_period == market_period,
        ShadowExecutionAssessment.policy_version.like("forward-v2-%"),
        *([ShadowExecutionAssessment.signal_version == signal_version] if signal_version else [])
    ))).scalars().all()
    for row in rows:
        row.theoretical_outcome = outcome
        row.theoretical_win = row.direction == outcome
        row.theoretical_settled_at = datetime.now(timezone.utc)
        decision = dict(row.decision_snapshot or {})
        decision["settlement_quote_estimates"] = quote_returns(row)
        row.decision_snapshot = decision
