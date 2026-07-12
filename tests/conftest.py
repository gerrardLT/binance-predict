"""
Pytest configuration for sentiment-curve-fixes bug condition exploration tests.

Provides shared fixtures for mocking the main module's dependencies.
"""

from __future__ import annotations

import os

# CI 环境无 .env，设置 dummy API Key 避免模块级 LLMService() 初始化失败
os.environ.setdefault("DEEPSEEK_API_KEY", "test-dummy-key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-dummy-key")

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_prediction_trader():
    """Mock BinancePredictionTrader instance with default state."""
    trader = MagicMock()
    trader.sync_server_time = AsyncMock()
    trader.fetch_wallet_info = AsyncMock(return_value={"walletAddress": "test", "walletId": "test"})
    trader.list_markets = AsyncMock(return_value=[])
    trader.execute_trade = AsyncMock(return_value=None)
    trader._wallet_address = "test_wallet"
    trader._wallet_id = "test_wallet_id"
    trader._api_key = "test_key"
    trader._api_secret = "test_secret"
    trader._active_market = None
    trader._up_token_id = None
    trader._down_token_id = None
    trader._5m_up_price = None
    trader._5m_down_price = None
    trader._5m_up_chance = None
    trader._5m_down_chance = None
    trader._5m_participant_count = None
    trader._5m_trade_volume = None
    trader._5m_liquidity = None
    trader._5m_market_question = None
    trader._5m_start_date = None
    trader._5m_end_date = None
    return trader


@pytest.fixture
def mock_collector():
    """Mock BinanceDataCollector instance."""
    collector = MagicMock()
    collector.store = MagicMock()
    collector.store.mid_price = 100000.0
    collector.store.kline_5m = []
    return collector


@pytest.fixture
def mock_db_session():
    """Mock async DB session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def pm_history():
    """Fresh _pm_history deque for testing."""
    return deque(maxlen=2000)
