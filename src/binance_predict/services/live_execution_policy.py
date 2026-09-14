"""Pure evaluators for live execution policy characterization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal

from .shadow_execution_types import ReasonCode, sanitize_json


Prediction = Literal["UP", "DOWN"]
EntryBands = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AbsorptionPolicyResult:
    btc_move: float | None
    up_move: float | None
    residual: float | None
    under: float | None
    prediction: Prediction | None
    eligible: bool
    reason: str


@dataclass(frozen=True)
class QuoteExecutionPolicyResult:
    eligible: bool
    reason: str
    guard_applied: bool
    whitelist_applied: bool
    slippage_bps: int


def evaluate_absorption_policy(
    *,
    btc_price: float,
    btc_open: float,
    up_price: float,
    up_open: float,
    k: float,
    b: float,
    disp_gate: float,
    under_gate: float,
) -> AbsorptionPolicyResult:
    """Evaluate the frozen ``MultiLiveTrader.check`` absorption math."""
    if not math.isfinite(btc_open) or btc_open <= 0:
        return AbsorptionPolicyResult(
            None, None, None, None, None, False, "INVALID_BTC_OPEN"
        )

    btc_move = (btc_price - btc_open) / btc_open * 1e4
    up_move = (up_price - up_open) * 100.0
    if not math.isfinite(btc_move) or btc_move == 0.0:
        return AbsorptionPolicyResult(
            btc_move, up_move, None, None, None, False, "ZERO_OR_INVALID_BTC_MOVE"
        )

    residual = up_move - (k * btc_move + b)
    under = -math.copysign(1.0, btc_move) * residual
    prediction: Prediction = "UP" if btc_move > 0 else "DOWN"
    if abs(btc_move) < disp_gate:
        return AbsorptionPolicyResult(
            btc_move, up_move, residual, under, prediction, False, "DISPLACEMENT_GATE"
        )
    if under < under_gate:
        return AbsorptionPolicyResult(
            btc_move, up_move, residual, under, prediction, False, "UNDER_REACTION_GATE"
        )
    return AbsorptionPolicyResult(
        btc_move, up_move, residual, under, prediction, True, ReasonCode.PASSED.value
    )


def evaluate_quote_execution_policy(
    *,
    order_type: str,
    avg_price: float,
    max_exec_price: float | None,
    entry_bands: EntryBands | None = None,
) -> QuoteExecutionPolicyResult:
    """Characterize current MARKET and LIMIT quote-level execution behavior."""
    if order_type == "LIMIT":
        return QuoteExecutionPolicyResult(
            True, ReasonCode.PASSED.value, False, False, 1200
        )

    guard_applied = max_exec_price is not None
    whitelist_applied = entry_bands is not None
    if guard_applied and (avg_price <= 0 or avg_price >= max_exec_price):
        return QuoteExecutionPolicyResult(
            False, ReasonCode.EXEC_PRICE_GUARD.value, True, whitelist_applied, 1200
        )

    in_band = (
        True
        if entry_bands is None
        else avg_price > 0 and any(lo <= avg_price < hi for lo, hi in entry_bands)
    )
    if not in_band:
        return QuoteExecutionPolicyResult(
            False,
            ReasonCode.ENTRY_BAND_REJECTED.value,
            guard_applied,
            True,
            1200,
        )

    slippage_bps = 1200
    if max_exec_price is not None and avg_price > 0:
        cap = int((max_exec_price / avg_price - 1.0) * 10000)
        slippage_bps = max(0, min(1200, cap))
    return QuoteExecutionPolicyResult(
        True,
        ReasonCode.PASSED.value,
        guard_applied,
        whitelist_applied,
        slippage_bps,
    )


def config_fingerprint(config: Any) -> str:
    """Return a stable SHA-256 fingerprint of sanitized canonical JSON."""
    payload = json.dumps(
        sanitize_json(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
