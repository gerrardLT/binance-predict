"""当前情绪曲线 Agent 路径的回归测试。"""

from __future__ import annotations

import inspect
import re

import pytest


def _get_tracker_source() -> str:
    """返回预测市场追踪器的源码。"""
    from binance_predict.main import _prediction_market_tracker

    return inspect.getsource(_prediction_market_tracker)


def _get_archiver_source() -> str:
    """返回情绪窗口归档器的源码。"""
    from binance_predict.main import _sentiment_window_archiver

    return inspect.getsource(_sentiment_window_archiver)


@pytest.mark.asyncio
async def test_tracker_does_not_mutate_trading_state() -> None:
    """报价追踪必须使用只读服务，不能刷新交易器状态。"""
    source = _get_tracker_source()

    assert "await prediction_trader.list_markets" not in source
    assert "market_data_service.fetch_market_data" in source

    import binance_predict.main as main_module
    from binance_predict.services.prediction_market_data import PredictionMarketDataService

    assert isinstance(main_module.market_data_service, PredictionMarketDataService)


@pytest.mark.asyncio
async def test_tracker_skips_quote_without_end_date() -> None:
    """缺失窗口结束时间的报价不得写入采样历史。"""
    source = _get_tracker_source()
    guard_position = max(
        source.find("end_date is None"),
        source.find("end_date == None"),
        source.find("quote is None"),
    )
    append_position = source.find("pm_history.append")

    assert guard_position >= 0
    assert append_position >= 0
    assert guard_position < append_position


@pytest.mark.asyncio
async def test_window_switch_resets_restore_flag() -> None:
    """每个新窗口都要允许从数据库恢复残留采样。"""
    source = _get_tracker_source()
    matches = list(re.finditer(r"_restored_current_window\s*=\s*False", source))

    assert len(matches) >= 2


@pytest.mark.asyncio
async def test_window_entry_price_uses_mid_price_snapshot() -> None:
    """窗口入场价来自窗口切换时的中间价快照。"""
    import binance_predict.main as main_module

    assert hasattr(main_module, "_window_entry_price")

    tracker_source = _get_tracker_source()
    archiver_source = _get_archiver_source()
    assert "_window_entry_price" in tracker_source
    assert "collector.store.mid_price" in tracker_source
    assert "_window_entry_price" in archiver_source
    assert re.search(r"for\s+k\s+in.*klines_5m.*entry_price", archiver_source, re.DOTALL) is None


@pytest.mark.asyncio
async def test_archiver_requires_eight_samples() -> None:
    """少于 8 个采样点的窗口不能归档。"""
    source = _get_archiver_source()

    assert re.search(r"len\(samples\)\s*<\s*8", source) is not None
    assert re.search(r"len\(samples\)\s*<\s*3", source) is None
