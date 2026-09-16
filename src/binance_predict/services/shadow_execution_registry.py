"""Registry for every shadow version exposed by ``/api/signals/analytics``."""
from __future__ import annotations

from dataclasses import dataclass

from .live_channels import LIVE_CHANNELS, RETIRED_CHANNEL_SPECS
from .shadow_execution_types import EXECUTION_POLICY_VERSION, SourceType


@dataclass(frozen=True)
class ShadowVersionSpec:
    version: str
    source_type: SourceType
    market_period: str
    target_semantics: str
    direction_mode: str
    live_channel: str | None
    live_retired: bool = False
    policy_version: str = EXECUTION_POLICY_VERSION
    family: str = "legacy"
    role: str = "STRATEGY"
    display_name: str | None = None


# Order is API behavior: keep additions explicit rather than sorting this registry.
_VERSION_ROWS: tuple[tuple[str, SourceType, str, str, str], ...] = (
    ("x4_v1", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_momentum_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("x4_v2", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_momentum_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("x4_v3", SourceType.MISALIGNMENT, "5m", "next_window", "row"),
    ("quote_contrarian_v3a", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v3b", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("quote_contrarian_v4", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("late_night_contrarian_v1", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("late_night_contrarian_v2", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("krev_a_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("krev_b_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("hm_touch_down_v1", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("hm_touch_down_v2", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("rev_p1_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("rev_p2_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("nb_zschamp_15m_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("nb_smaslope_5m_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("combo_p1_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p2_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p3_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p4_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("combo_p5_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s5_deep_z20_v1", SourceType.PATTERN, "15m", "target_bar", "down"),
    ("quote_momentum_v3", SourceType.MISALIGNMENT, "5m", "same_window", "row"),
    ("absorption_follow_td120_v1", SourceType.ABSORPTION, "5m", "same_window", "dynamic"),
    ("absorption_follow_td150_v1", SourceType.ABSORPTION, "5m", "same_window", "dynamic"),
    ("s2_cond_t4_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("s2_cond_t5d_v1", SourceType.KLINE, "15m", "target_bar", "row"),
    ("firsthit_down_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_body_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_chg_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_g4_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("firsthit_down_g7_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_streak_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_wick20_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_strict_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_q05_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("g7_t270_v1", SourceType.FIRSTHIT, "5m", "same_window_first_touch", "down"),
    ("hm_inside_15m_v2", SourceType.KLINE, "15m", "target_bar", "row"),
    ("ih_inside_15m_v2", SourceType.KLINE, "15m", "target_bar", "row"),
    ("hm_inside_5m_v2", SourceType.KLINE, "5m", "target_bar", "row"),
    # 长影短实体检测器按 timeframe 只落一个物理事件；18 个冻结逻辑版本作为标签，
    # 不在此投影表重复展开，避免同窗主版本/严格子集/组件被统计成多次事件。
    ("candle_hm_bull_5m_event_v1", SourceType.KLINE, "5m", "target_bar", "row"),
    ("candle_hm_bull_15m_event_v1", SourceType.KLINE, "15m", "target_bar", "row"),
)


def _live_mapping(version: str) -> tuple[str | None, bool]:
    if version in LIVE_CHANNELS:
        return version, False
    if version in RETIRED_CHANNEL_SPECS:
        return version, True
    return None, False


def _family_of(version: str, source: SourceType) -> str:
    if version.startswith("candle_hm_bull_"):
        return "candlestick_reversal"
    if version.startswith(("quote_", "late_night_")):
        return "quote_edge"
    if version.startswith("x4_"):
        return "x4"
    if version.startswith(("firsthit_", "g7_")):
        return "firsthit"
    if version.startswith("absorption_"):
        return "absorption"
    if version.startswith("s2_cond_"):
        return "s2_cond"
    if version.startswith("combo_"):
        return "combo"
    if version.startswith(("nb_", "hm_inside_", "ih_inside_")):
        return "nextbar"
    if version.startswith(("krev_", "rev_")):
        return "kline_reversal"
    if version.startswith(("hm_touch_", "s5_deep_")):
        return "pattern"
    return source.value

SHADOW_VERSION_SPECS: dict[str, ShadowVersionSpec] = {}
for _version, _source, _period, _target, _direction in _VERSION_ROWS:
    _live, _retired = _live_mapping(_version)
    SHADOW_VERSION_SPECS[_version] = ShadowVersionSpec(
        _version, _source, _period, _target, _direction, _live, _retired,
        family=_family_of(_version, _source),
        role=("EVENT" if _version.startswith("candle_hm_bull_") else "STRATEGY"),
    )
SHADOW_VERSIONS: tuple[str, ...] = tuple(SHADOW_VERSION_SPECS)


def _validate() -> None:
    if len(_VERSION_ROWS) != len(SHADOW_VERSION_SPECS):
        raise RuntimeError("shadow execution registry must contain unique versions")
    if {spec.source_type for spec in SHADOW_VERSION_SPECS.values()} != set(SourceType):
        raise RuntimeError("shadow execution registry must cover every source type")
    valid_targets = {"same_window", "next_window", "target_bar", "same_window_first_touch"}
    valid_directions = {"row", "down", "dynamic"}
    for version, spec in SHADOW_VERSION_SPECS.items():
        if version != spec.version:
            raise RuntimeError(f"registry key/version mismatch: {version}")
        if spec.market_period not in {"5m", "15m"}:
            raise RuntimeError(f"invalid market period for {version}: {spec.market_period}")
        if spec.target_semantics not in valid_targets:
            raise RuntimeError(f"invalid target semantics for {version}: {spec.target_semantics}")
        if spec.direction_mode not in valid_directions:
            raise RuntimeError(f"invalid direction mode for {version}: {spec.direction_mode}")
        if spec.policy_version != EXECUTION_POLICY_VERSION:
            raise RuntimeError(f"invalid execution policy for {version}: {spec.policy_version}")
        expected_live, expected_retired = _live_mapping(version)
        if (spec.live_channel, spec.live_retired) != (expected_live, expected_retired):
            raise RuntimeError(f"invalid live mapping for {version}: {spec.live_channel}")


_validate()
