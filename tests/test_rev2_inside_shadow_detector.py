"""15m 孕线反转（rev2 族）影子与实盘通道集成测试。

覆盖：
1. evaluate_rev2_patterns 纯函数几何求值正反例与冻结口径
2. Rev2InsideShadowDetector 状态机、幂等落 PENDING、实盘开火钩子分派
3. LIVE_CHANNELS 注册表对齐与 MultiLiveTrader 护栏调度
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from binance_predict.discovery.data import Klines
from binance_predict.services.live_channels import LIVE_CHANNELS
import binance_predict.services.rev2_inside_shadow_detector as rev2_mod
from binance_predict.services.rev2_inside_shadow_detector import (
    BAR_MS,
    BAR_MS_15M,
    REV2_5M_SPECS,
    REV2_SHADOW_SPECS,
    REV2_VERSIONS,
    Rev2InsideShadowDetector,
    _to_klines,
    evaluate_rev2_patterns,
)


def test_live_channels_registered() -> None:
    """验证 Rev2 三个通道的周期、方向与默认热调护栏。"""
    spec_5m = LIVE_CHANNELS["hm_inside_5m_v2"]
    assert spec_5m.family == "nextbar"
    assert spec_5m.market_period == "5m"
    assert spec_5m.direction == "DOWN"
    assert spec_5m.auto_max_exec == 0.30
    assert spec_5m.order_type == "LIMIT"

    spec_hm = LIVE_CHANNELS["hm_inside_15m_v2"]
    assert spec_hm.family == "nextbar"
    assert spec_hm.market_period == "15m"
    assert spec_hm.direction == "DOWN"
    assert spec_hm.auto_max_exec == 0.30

    spec_ih = LIVE_CHANNELS["ih_inside_15m_v2"]
    assert spec_ih.family == "nextbar"
    assert spec_ih.market_period == "15m"
    assert spec_ih.direction == "UP"
    assert spec_ih.auto_max_exec == 0.30


def test_evaluate_rev2_patterns_hanging_man() -> None:
    """构造标准 15m 上吊线孕线形态：前置大阳为高点 + 附近最长 + 信号柱完全包裹长下影。"""
    t0 = 1700000000000
    rows = [
        # i=0
        {"open_time": t0, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        # i=1
        {"open_time": t0 + BAR_MS_15M, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 10},
        # i=2
        {"open_time": t0 + 2 * BAR_MS_15M, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 10},
        # i=3: 前一根大阳线 (Bar N-1)，高点 115 为前 4 根最高，实体 112-103=9 大于前 3 根
        {"open_time": t0 + 3 * BAR_MS_15M, "open": 103, "high": 115, "low": 102.5, "close": 114, "volume": 20},
        # i=4: 信号柱短上影恰好 10%，主下影 80%
        {"open_time": t0 + 4 * BAR_MS_15M, "open": 113, "high": 114, "low": 104, "close": 112, "volume": 15},
    ]
    kl = _to_klines(rows, BAR_MS_15M)
    hits = evaluate_rev2_patterns(kl, n_tail=2)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["spec"]["version"] == "hm_inside_15m_v2"
    assert hit["idx"] == 4
    assert hit["spec"]["direction"] == "DOWN"


def test_evaluate_rev2_patterns_inverted_hammer() -> None:
    """构造标准 15m 倒垂线孕线形态：前置大阴为低点 + 附近最长 + 信号柱完全包裹长上影。"""
    t0 = 1700000000000
    rows = [
        # i=0
        {"open_time": t0, "open": 100, "high": 101, "low": 98, "close": 99, "volume": 10},
        # i=1
        {"open_time": t0 + BAR_MS_15M, "open": 99, "high": 100, "low": 97, "close": 98, "volume": 10},
        # i=2
        {"open_time": t0 + 2 * BAR_MS_15M, "open": 98, "high": 99, "low": 96, "close": 97, "volume": 10},
        # i=3: 前一根大阴线 (Bar N-1)，低点 85 为前 4 根最低，实体 97-86=11 大于前 3 根
        {"open_time": t0 + 3 * BAR_MS_15M, "open": 97, "high": 97.5, "low": 85, "close": 86, "volume": 20},
        # i=4: 信号柱短下影恰好 10%，主上影 80%
        {"open_time": t0 + 4 * BAR_MS_15M, "open": 87, "high": 96, "low": 86, "close": 88, "volume": 15},
    ]
    kl = _to_klines(rows, BAR_MS_15M)
    hits = evaluate_rev2_patterns(kl, n_tail=2)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["spec"]["version"] == "ih_inside_15m_v2"
    assert hit["idx"] == 4
    assert hit["spec"]["direction"] == "UP"


def test_evaluate_rev2_rejects_breakout() -> None:
    """突破前柱高低点的假孕线必须被严格拒绝。"""
    t0 = 1700000000000
    rows = [
        {"open_time": t0, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        {"open_time": t0 + BAR_MS_15M, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 10},
        {"open_time": t0 + 2 * BAR_MS_15M, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 10},
        {"open_time": t0 + 3 * BAR_MS_15M, "open": 103, "high": 115, "low": 102.5, "close": 114, "volume": 20},
        # 信号柱高点达到 116 (超过前高 115) -> 破位，不属于被包裹
        {"open_time": t0 + 4 * BAR_MS_15M, "open": 113, "high": 116, "low": 105, "close": 112, "volume": 15},
    ]
    kl = _to_klines(rows, BAR_MS_15M)
    hits = evaluate_rev2_patterns(kl, n_tail=2)
    assert len(hits) == 0


def _hm_rows(bar_ms: int, lower_r: float, upper_r: float) -> list[dict]:
    t0 = 1700000000000
    high, low = 114.0, 104.0
    return [
        {"open_time": t0, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        {"open_time": t0 + bar_ms, "open": 101, "high": 103, "low": 100, "close": 102, "volume": 10},
        {"open_time": t0 + 2 * bar_ms, "open": 102, "high": 104, "low": 101, "close": 103, "volume": 10},
        {"open_time": t0 + 3 * bar_ms, "open": 103, "high": 115, "low": 102.5, "close": 114, "volume": 20},
        {
            "open_time": t0 + 4 * bar_ms,
            "open": high - upper_r * (high - low),
            "high": high,
            "low": low,
            "close": low + lower_r * (high - low),
            "volume": 15,
        },
    ]


def _ih_rows(lower_r: float) -> list[dict]:
    t0 = 1700000000000
    high, low = 96.0, 86.0
    return [
        {"open_time": t0, "open": 100, "high": 101, "low": 98, "close": 99, "volume": 10},
        {"open_time": t0 + BAR_MS_15M, "open": 99, "high": 100, "low": 97, "close": 98, "volume": 10},
        {"open_time": t0 + 2 * BAR_MS_15M, "open": 98, "high": 99, "low": 96, "close": 97, "volume": 10},
        {"open_time": t0 + 3 * BAR_MS_15M, "open": 97, "high": 97.5, "low": 85, "close": 86, "volume": 20},
        {
            "open_time": t0 + 4 * BAR_MS_15M,
            "open": low + lower_r * (high - low),
            "high": high,
            "low": low,
            "close": 88.0,
            "volume": 15,
        },
    ]


def test_15m_short_wick_above_ten_percent_rejected() -> None:
    hm = _to_klines(_hm_rows(BAR_MS_15M, lower_r=0.75, upper_r=0.1001), BAR_MS_15M)
    ih = _to_klines(_ih_rows(lower_r=0.1001), BAR_MS_15M)
    assert evaluate_rev2_patterns(hm, 1) == []
    assert evaluate_rev2_patterns(ih, 1) == []


@pytest.mark.parametrize(
    ("lower_r", "expected"),
    [(0.7499, False), (0.75, True), (0.8999, True), (0.90, False)],
)
def test_5m_hm_main_wick_boundaries(lower_r: float, expected: bool) -> None:
    rows = _hm_rows(BAR_MS["5m"], lower_r=lower_r, upper_r=0.05)
    hits = evaluate_rev2_patterns(
        _to_klines(rows, BAR_MS["5m"]), 1, REV2_5M_SPECS
    )
    assert bool(hits) is expected
    if hits:
        assert hits[0]["spec"]["version"] == "hm_inside_5m_v2"


@pytest.mark.parametrize(("upper_r", "expected"), [(0.10, True), (0.1001, False)])
def test_5m_hm_short_wick_boundary(upper_r: float, expected: bool) -> None:
    rows = _hm_rows(BAR_MS["5m"], lower_r=0.80, upper_r=upper_r)
    hits = evaluate_rev2_patterns(
        _to_klines(rows, BAR_MS["5m"]), 1, REV2_5M_SPECS
    )
    assert bool(hits) is expected


@pytest.mark.asyncio
async def test_5m_record_signal_uses_matching_quote_and_live_window(monkeypatch) -> None:
    bar_ms = BAR_MS["5m"]
    signal_start = 1_700_000_000_000
    target_start = signal_start + bar_ms
    cache = {
        "start_date": target_start,
        "up_price": 0.68,
        "down_price": 0.32,
        "updated_ts": target_start + 15_000,
    }
    detector = Rev2InsideShadowDetector(MagicMock(), cache, timeframe="5m")
    session = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result)
    payloads: list[dict] = []
    monkeypatch.setattr(rev2_mod.time, "time", lambda: (target_start + 30_000) / 1000)

    added = await detector._record_signal(
        session,
        REV2_5M_SPECS[0],
        {"open_time": signal_start},
        {"lower_r": 0.8, "upper_r": 0.1},
        payloads,
    )

    assert added is True
    row = session.add.call_args.args[0]
    assert row.timeframe == "5m"
    assert row.version == "hm_inside_5m_v2"
    assert row.target_bar_start == target_start
    assert row.entry_up_price == pytest.approx(0.68)
    assert row.entry_down_price == pytest.approx(0.32)
    assert row.entry_quote_ts == target_start + 15_000
    assert payloads == [{
        "version": "hm_inside_5m_v2",
        "market_start": target_start,
        "market_end": target_start + bar_ms,
        "direction": "DOWN",
        "signal_bar_start": signal_start,
    }]
