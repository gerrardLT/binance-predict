"""Pure adapters from the five legacy shadow tables to normalized events."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import and_, select

from binance_predict.db.models import (
    AbsorptionShadowSignal,
    FirstHitShadowSignal,
    KlineShadowSignal,
    MisalignmentSignal,
    PatternShadowSignal,
    ShadowExecutionAssessment,
)

from .shadow_execution_registry import SHADOW_VERSION_SPECS
from .shadow_execution_types import NormalizedShadowEvent, SourceType


def _entry_price(row: Any, direction: str) -> float | None:
    if direction == "UP":
        value = getattr(row, "entry_up_price", None)
    else:
        value = getattr(row, "entry_down_price", None)
    return float(value) if value is not None else None


def _realized_return(row: Any, entry: float | None, signal_version: str) -> float | None:
    stored = getattr(row, "ev_at_entry", None)
    if stored is not None:
        return float(stored)
    win = getattr(row, "win", None)
    if win is None or entry is None or entry <= 0:
        return None
    if not win:
        return -1.0
    if signal_version.startswith("x4_"):
        return 0.98 / (entry + 0.01) - 1.0
    return 0.98 / entry - 1.0


@dataclass(frozen=True)
class ShadowSourceAdapter:
    source_type: SourceType
    model: type
    source_window: Callable[[Any], int]
    target_window: Callable[[Any], int]
    direction: Callable[[Any], str]
    entry: Callable[[Any, str], float | None]
    trigger_ts: Callable[[Any], int | None]
    evaluation_ts: Callable[[Any], int | None]
    settled_at: Callable[[Any], Any]

    def statement(self, policy_version: str, limit: int):
        assessment = ShadowExecutionAssessment
        registered_versions = [
            version
            for version, spec in SHADOW_VERSION_SPECS.items()
            if spec.source_type is self.source_type
        ]
        return (
            select(self.model)
            .outerjoin(
                assessment,
                and_(
                    assessment.source_type == self.source_type.value,
                    assessment.source_id == self.model.id,
                    assessment.policy_version == policy_version,
                ),
            )
            .where(
                assessment.id.is_(None),
                self.model.version.in_(registered_versions),
            )
            .order_by(self.model.id)
            .limit(limit)
        )

    def refresh_statement(self, policy_version: str, limit: int):
        assessment = ShadowExecutionAssessment
        return (
            select(self.model)
            .outerjoin(
                assessment,
                and_(
                    assessment.source_type == self.source_type.value,
                    assessment.source_id == self.model.id,
                    assessment.policy_version == policy_version,
                ),
            )
            .where(
                assessment.id.isnot(None),
                assessment.source_status.is_distinct_from(self.model.status),
            )
            .order_by(self.model.id)
            .limit(limit)
        )

    def normalize(self, row: Any) -> NormalizedShadowEvent:
        signal_version = str(row.version)
        spec = SHADOW_VERSION_SPECS.get(signal_version)
        if spec is None:
            raise ValueError(f"unregistered shadow signal version: {signal_version}")
        if spec.source_type is not self.source_type:
            raise ValueError(
                f"shadow signal version {signal_version} belongs to {spec.source_type.value}, "
                f"not {self.source_type.value}"
            )

        row_direction = self.direction(row).upper()
        direction = "DOWN" if spec.direction_mode == "down" else row_direction
        source_window_start = self.source_window(row)
        target_window_start = self.target_window(row)
        entry = self.entry(row, direction)
        return NormalizedShadowEvent(
            source_type=self.source_type,
            source_id=int(row.id),
            signal_version=signal_version,
            source_window_start=source_window_start,
            target_window_start=target_window_start,
            market_period=spec.market_period,
            direction=direction,
            trigger_ts=self.trigger_ts(row),
            evaluation_ts=self.evaluation_ts(row),
            source_status=getattr(row, "status", None),
            source_created_at=getattr(row, "created_at", None),
            theoretical_win=getattr(row, "win", None),
            theoretical_outcome=getattr(row, "settle_outcome", None),
            theoretical_entry_price=entry,
            theoretical_realized_return=_realized_return(row, entry, signal_version),
            theoretical_settled_at=self.settled_at(row),
            source_payload={"source_table": self.model.__tablename__},
        )


def _pattern_entry(row: Any, _direction: str) -> float | None:
    value = row.entry_down_quote
    return float(value) if value is not None else None


def _firsthit_entry(row: Any, _direction: str) -> float | None:
    value = row.entry_down_price
    return float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


SHADOW_SOURCE_ADAPTERS: tuple[ShadowSourceAdapter, ...] = (
    ShadowSourceAdapter(
        SourceType.MISALIGNMENT, MisalignmentSignal,
        lambda r: int(r.window_start), lambda r: int(r.target_window_start),
        lambda r: str(r.direction), _entry_price,
        lambda r: int(r.window_end),
        lambda r: _optional_int(r.entry_quote_ts),
        lambda _r: None,
    ),
    ShadowSourceAdapter(
        SourceType.KLINE, KlineShadowSignal,
        lambda r: int(r.signal_bar_start), lambda r: int(r.target_bar_start),
        lambda r: str(r.direction), _entry_price,
        lambda r: int(r.signal_bar_end),
        lambda r: _optional_int(r.entry_quote_ts),
        lambda r: r.settled_at,
    ),
    ShadowSourceAdapter(
        SourceType.PATTERN, PatternShadowSignal,
        lambda r: int(r.signal_bar_start), lambda r: int(r.target_bar_start),
        lambda _r: "DOWN", _pattern_entry,
        lambda r: int(r.signal_bar_end),
        lambda r: _optional_int(r.touch_ts),
        lambda r: r.settled_at,
    ),
    ShadowSourceAdapter(
        SourceType.ABSORPTION, AbsorptionShadowSignal,
        lambda r: int(r.window_start), lambda r: int(r.window_start),
        lambda r: str(r.direction), _entry_price,
        lambda r: int(r.window_start) + int(r.td) * 1000,
        lambda r: _optional_int(r.entry_quote_ts),
        lambda _r: None,
    ),
    ShadowSourceAdapter(
        SourceType.FIRSTHIT, FirstHitShadowSignal,
        lambda r: int(r.window_start), lambda r: int(r.window_start),
        lambda r: "DOWN", _firsthit_entry,
        lambda r: int(r.trigger_ts),
        lambda r: int(r.trigger_ts),
        lambda _r: None,
    ),
)
