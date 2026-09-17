import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("src"))

from binance_predict.services.prediction_trading import BinancePredictionTrader
from binance_predict.services import clock_sync

async def test_probe():
    trader = BinancePredictionTrader()
    await clock_sync.sync_server_time(force=True)
    print("Clock synced offset:", clock_sync.get_offset_ms())
    
    wallet = await trader.fetch_wallet_info()
    print("Wallet fetched:", bool(wallet), wallet)
    
    # Check markets
    await trader.list_markets()
    print("15m markets available:", list(trader._15m_markets.keys()))
    print("Up token:", trader._up_token_id)
    print("Down token:", trader._down_token_id)

asyncio.run(test_probe())
