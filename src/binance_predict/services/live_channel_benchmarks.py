"""Frozen research benchmarks for live channels, with traceable code sources."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from binance_predict.services.candlestick_shadow_detector import CANDLESTICK_BACKTEST
from binance_predict.services.fake_breakout_detector import RESEARCH_WIN_RATES
from binance_predict.services.live_channels import LIVE_CHANNELS


@dataclass(frozen=True)
class ChannelBenchmark:
    channel: str
    win_rate: float
    source_ref: str
    method_version: str = "frozen_point_estimate_v1"
    quality: str = "FROZEN_RESEARCH"
    sample_size: int | None = None


# Keep these values aligned with main.py SHADOW_BENCH. Importing main would create
# an application-level dependency cycle, so this small registry repeats the frozen points.
_FROZEN_BENCHMARKS: dict[str, tuple[float, int | None]] = {
    "quote_contrarian_v2": (0.258, 155),
    "late_night_contrarian_v2": (0.440, 50),
    "x4_v2": (0.553, 94),
    "x4_v3": (0.425, 40),
    "s5_deep_z20_v1": (0.913, None),
    "s2_cond_t4_v1": (0.389, 1069),
    "s2_cond_t5d_v1": (0.448, 643),
    "scene_bear_exhaust_opt_v1": (0.5897, 975),
    "nb_smaslope_5m_v1": (0.4743, 19597),
    "krev_a_v1": (0.642, 137),
    "krev_b_v1": (0.634, 134),
    "rev_p1_v1": (0.620, None),
    "rev_p2_v1": (0.624, None),
    "absorption_follow_td120_v1": (0.798, 397),
    "absorption_follow_td150_v1": (0.885, 399),
    "firsthit_down_v1": (0.088, 4632),
    "firsthit_down_body_v1": (0.153, 418),
    "firsthit_down_chg_v1": (0.108, 1309),
    "firsthit_down_g4_v1": (0.157, 300),
    "firsthit_down_g7_v1": (0.192, 260),
    "g7_streak_v1": (0.193, 202),
    "g7_wick20_v1": (0.205, 127),
    "g7_strict_v1": (0.219, 114),
    "g7_q05_v1": (0.137, 51),
    "g7_t270_v1": (0.286, 84),
    "hm_inside_15m_v2": (0.557, 237),
    "ih_inside_15m_v2": (0.546, 183),
    "hm_inside_5m_v2": (0.585, 371),
    **{channel: (win_rate, sample_size)
       for channel, (win_rate, _ev, sample_size) in CANDLESTICK_BACKTEST.items()},
}

_SCENE_PATTERNS = {
    "scene_bull_exhaust": "bull_exhaust",
    "scene_bull_exhaust_confirm": "bull_exhaust_confirm",
    "scene_bear_exhaust": "bear_exhaust",
    "scene_momentum_fade": "momentum_fade",
}

CHANNEL_BENCHMARKS: dict[str, ChannelBenchmark] = {
    channel: ChannelBenchmark(
        channel=channel,
        win_rate=rate,
        sample_size=sample_size,
        source_ref="main.py:SHADOW_BENCH",
    )
    for channel, (rate, sample_size) in _FROZEN_BENCHMARKS.items()
    if channel in LIVE_CHANNELS
}
CHANNEL_BENCHMARKS.update({
    channel: ChannelBenchmark(
        channel=channel,
        win_rate=RESEARCH_WIN_RATES[pattern],
        source_ref="fake_breakout_detector.RESEARCH_WIN_RATES",
        method_version="scene_research_win_rates_v1",
    )
    for channel, pattern in _SCENE_PATTERNS.items()
    if channel in LIVE_CHANNELS and pattern in RESEARCH_WIN_RATES
})


def get_channel_benchmark(channel: str) -> ChannelBenchmark | None:
    return CHANNEL_BENCHMARKS.get(channel)


def benchmark_dict(channel: str) -> dict[str, object] | None:
    benchmark = get_channel_benchmark(channel)
    return asdict(benchmark) if benchmark else None
