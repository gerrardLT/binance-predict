"""Shared types for the unified shadow execution assessment ledger."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
import math
from typing import Any, Mapping


EXECUTION_POLICY_VERSION = "live-exec-v1"


class SourceType(str, Enum):
    MISALIGNMENT = "misalignment"
    KLINE = "kline"
    PATTERN = "pattern"
    ABSORPTION = "absorption"
    FIRSTHIT = "firsthit"


class ExecutionMode(str, Enum):
    SHADOW_ONLY = "SHADOW_ONLY"
    LIVE_ACTIVE = "LIVE_ACTIVE"
    LIVE_RETIRED = "LIVE_RETIRED"


class TerminalStage(str, Enum):
    UNKNOWN = "UNKNOWN"
    SOURCE = "SOURCE"
    STRATEGY = "STRATEGY"
    OPERATIONAL = "OPERATIONAL"
    EXECUTION = "EXECUTION"
    ORDER = "ORDER"
    FILL = "FILL"
    SETTLEMENT = "SETTLEMENT"


class ReasonCode(str, Enum):
    PASSED = "PASSED"
    LEGACY_UNKNOWN = "LEGACY_UNKNOWN"
    NO_LIVE_MAPPING = "NO_LIVE_MAPPING"
    RETIRED_CHANNEL = "RETIRED_CHANNEL"
    CHANNEL_DISABLED = "CHANNEL_DISABLED"
    DAILY_LIMIT_REACHED = "DAILY_LIMIT_REACHED"
    DUPLICATE_WINDOW = "DUPLICATE_WINDOW"
    EXCLUSIVE_GROUP_BLOCKED = "EXCLUSIVE_GROUP_BLOCKED"
    STRATEGY_REJECTED = "STRATEGY_REJECTED"
    STRATEGY_DATA_MISSING = "STRATEGY_DATA_MISSING"
    MARKET_NOT_FOUND = "MARKET_NOT_FOUND"
    BALANCE_INSUFFICIENT = "BALANCE_INSUFFICIENT"
    BALANCE_CHECK_UNKNOWN = "BALANCE_CHECK_UNKNOWN"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    QUOTE_INVALID = "QUOTE_INVALID"
    EXEC_PRICE_GUARD = "EXEC_PRICE_GUARD"
    ENTRY_BAND_REJECTED = "ENTRY_BAND_REJECTED"
    ORDER_SUBMIT_FAILED = "ORDER_SUBMIT_FAILED"
    ORDER_PENDING = "ORDER_PENDING"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_FAILED = "ORDER_FAILED"
    SETTLEMENT_PENDING = "SETTLEMENT_PENDING"
    SETTLED = "SETTLED"
    SETTLEMENT_EXPIRED = "SETTLEMENT_EXPIRED"
    SOURCE_MISSING = "SOURCE_MISSING"
    UNREGISTERED_VERSION = "UNREGISTERED_VERSION"


_STAGE_ORDER = {
    TerminalStage.UNKNOWN: 0,
    TerminalStage.SOURCE: 10,
    TerminalStage.STRATEGY: 20,
    TerminalStage.OPERATIONAL: 30,
    TerminalStage.EXECUTION: 40,
    TerminalStage.ORDER: 50,
    TerminalStage.FILL: 60,
    TerminalStage.SETTLEMENT: 70,
}


def stage_rank(stage: TerminalStage | str | None) -> int:
    if stage is None:
        return 0
    try:
        return _STAGE_ORDER[TerminalStage(stage)]
    except (ValueError, KeyError):
        return 0


def stage_advances(current: TerminalStage | str | None, candidate: TerminalStage | str) -> bool:
    return stage_rank(candidate) >= stage_rank(current)


@dataclass(frozen=True)
class NormalizedShadowEvent:
    source_type: SourceType
    source_id: int
    signal_version: str
    source_window_start: int
    target_window_start: int
    market_period: str
    direction: str
    trigger_ts: int | None = None
    evaluation_ts: int | None = None
    source_status: str | None = None
    source_created_at: Any = None
    theoretical_win: bool | None = None
    theoretical_outcome: str | None = None
    theoretical_entry_price: float | None = None
    theoretical_realized_return: float | None = None
    theoretical_settled_at: Any = None
    source_payload: dict[str, Any] = field(default_factory=dict)
    theoretical_payload: dict[str, Any] = field(default_factory=dict)


_UNSET = object()


@dataclass(frozen=True)
class AssessmentPatch:
    assessment_id: int
    stage: TerminalStage
    reason: ReasonCode
    strategy_eligible: Any = _UNSET
    operational_eligible: Any = _UNSET
    execution_eligible: Any = _UNSET
    config_fingerprint: Any = _UNSET
    evaluation_ts: Any = _UNSET
    live_channel: Any = _UNSET
    channel_mapped: Any = _UNSET
    enabled: Any = _UNSET
    retired: Any = _UNSET
    order_type: Any = _UNSET
    amount_usdt: Any = _UNSET
    max_daily_orders: Any = _UNSET
    effective_max_exec_price: Any = _UNSET
    guard_applied: Any = _UNSET
    exclusive_blocker_channel: Any = _UNSET
    market_id: Any = _UNSET
    token_id: Any = _UNSET
    balance_available: Any = _UNSET
    balance_required: Any = _UNSET
    quote_average_price: Any = _UNSET
    quote_ts: Any = _UNSET
    config_snapshot: Any = _UNSET
    input_snapshot: Any = _UNSET
    decision_snapshot: Any = _UNSET
    quote_snapshot: Any = _UNSET

    def values(self) -> dict[str, Any]:
        result = {"terminal_stage": self.stage.value, "reason_code": self.reason.value}
        for name in self.__dataclass_fields__:
            if name in {"assessment_id", "stage", "reason"}:
                continue
            value = getattr(self, name)
            if value is not _UNSET:
                result[name] = sanitize_json(value) if name.endswith("_snapshot") else value
        return result

    def update_mapping(self) -> dict[str, Any]:
        return self.values()


_SENSITIVE_PARTS = (
    "api_key", "apikey", "secret", "signature", "token",
    "authorization", "cookie", "password",
)


def sanitize_json(value: Any) -> Any:
    """Return a JSON-safe copy with nested authentication material redacted."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        return sanitize_json(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            lowered = normalized_key.casefold()
            clean[normalized_key] = (
                "[REDACTED]" if any(part in lowered for part in _SENSITIVE_PARTS)
                else sanitize_json(item)
            )
        return clean
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)
