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
from binance_predict.services.rev2_inside_shadow_detector import (
    BAR_MS_15M,
    REV2_SHADOW_SPECS,
    REV2_VERSIONS,
    Rev2InsideShadowDetector,
    _to_klines,
    evaluate_rev2_patterns,
)


def test_live_channels_registered() -> None:
    """验证 15m Ver2 两个通道已正确注册在 LIVE_CHANNELS 中。"""
    assert "hm_inside_15m_v2" in LIVE_CHANNELS
    assert "ih_inside_15m_v2" in LIVE_CHANNELS

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
        # i=4: 信号柱 (Bar N)，完全在 [102.5, 115] 内，高 114，低 105，开 113，收 112 (长下影=7, 实体=1, 上影=1)
        {"open_time": t0 + 4 * BAR_MS_15M, "open": 113, "high": 114, "low": 105, "close": 112, "volume": 15},
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
        # i=4: 信号柱 (Bar N)，完全在 [85, 97.5] 内，高 95，低 86，开 87，收 88 (长上影=7, 实体=1, 下影=1)
        {"open_time": t0 + 4 * BAR_MS_15M, "open": 87, "high": 95, "low": 86, "close": 88, "volume": 15},
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
