"""现货采集器仅保留中间价来源的回归测试。"""

import inspect

from binance_predict.services.data_collector import BinanceDataCollector


def test_spot_collector_subscribes_only_to_book_ticker() -> None:
    source = inspect.getsource(BinanceDataCollector.connect_spot_ws)

    assert "@bookTicker" in source
    assert "@kline" not in source
    assert "@aggTrade" not in source
    assert "@depth" not in source
