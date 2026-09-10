import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from binance_predict.config.settings import settings
from binance_predict.services.rev2_inside_shadow_detector import Rev2InsideShadowDetector
from binance_predict.services.wechat_notifier import wechat_notifier


@pytest.mark.asyncio
async def test_rev2_radar_early_warning():
    """测试 15m 孕线反转雷达在收盘前探测到雏形时触发企微预警。"""
    collector = MagicMock()
    now_ms = 1725900800000  # 假设当前时间
    # 目标 15m 柱开始时间：now_ms - 13.5 min = 810s 之前
    bar_start = now_ms - 810_000
    bar_end = bar_start + 900_000
    rem_sec = int((bar_end - now_ms) / 1000)  # 90 秒

    # 构造满足条件的 K 线列表
    # 前 5 根历史
    raw_klines = []
    # 0, 1, 2, 3
    base_price = 60000.0
    for i in range(4):
        raw_klines.append({
            "open_time": bar_start - (4 - i) * 900_000,
            "open": base_price + i * 10,
            "high": base_price + i * 10 + 20,
            "low": base_price + i * 10 - 20,
            "close": base_price + i * 10 + 10,
            "volume": 10.0,
        })
    # Bar N-1 (最长实体大阳线)
    raw_klines.append({
        "open_time": bar_start - 900_000,
        "open": 60050.0,
        "high": 60300.0,
        "low": 60040.0,
        "close": 60280.0,  # 实体 230，长阳
        "volume": 50.0,
    })
    # Bar N (当前未走完 15m 柱：孕线包裹，下影线很长 -> 上吊线 HM 雏形)
    raw_klines.append({
        "open_time": bar_start,
        "open": 60250.0,
        "high": 60260.0,
        "low": 60100.0,   # 完全包裹在 [60040, 60300] 内
        "close": 60240.0, # 实体 10，下影线 140，下影占比 > 0.45
        "volume": 20.0,
    })

    collector.fetch_klines_raw = AsyncMock(return_value=raw_klines)
    detector = Rev2InsideShadowDetector(collector, {})

    with patch.object(settings, "wechat_work_enabled", True), \
         patch.object(settings, "wechat_radar_enabled", True), \
         patch("time.time", return_value=now_ms / 1000.0), \
         patch.object(wechat_notifier, "notify_pre_trade_radar") as mock_notify:
        
        await detector._check_radar()

        assert mock_notify.call_count == 1
        call_kwargs = mock_notify.call_args[1]
        assert call_kwargs["channel"] == "hm_inside_15m_v2"
        assert call_kwargs["direction"] == "DOWN"
        assert call_kwargs["lead_seconds"] == rem_sec
        assert call_kwargs["max_exec_price"] == 0.30
