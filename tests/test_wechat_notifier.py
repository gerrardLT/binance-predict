import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from binance_predict.config.settings import settings
from binance_predict.services.wechat_notifier import WeChatNotifier


def test_format_channel_title():
    from binance_predict.services.live_channels import format_channel_title

    # 1. 正常活跃通道
    res_active = format_channel_title("hm_inside_15m_v2")
    assert res_active == "**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)"

    # 2. 退役通道兜底
    res_retired = format_channel_title("quote_momentum_v1")
    assert res_retired == "**报价动量（A格顺势）** (`quote_momentum_v1`)"

    # 3. 未知通道安全回退
    res_unknown = format_channel_title("unknown_channel_xyz")
    assert res_unknown == "`unknown_channel_xyz`"

    # 4. 空字符串或 None 安全兜底
    assert format_channel_title("") == "``"
    assert format_channel_title(None) == "``"


@pytest.fixture
def notifier():
    n = WeChatNotifier()
    n.reset_daily_count()
    return n


@pytest.mark.asyncio
async def test_wechat_notifier_disabled(notifier):
    with patch.object(settings, "wechat_work_enabled", False):
        res = await notifier._send_markdown_raw("test message")
        assert res is False


@pytest.mark.asyncio
async def test_wechat_notifier_empty_url(notifier):
    with patch.object(settings, "wechat_work_enabled", True), \
         patch.object(settings, "wechat_work_webhook_url", ""):
        res = await notifier._send_markdown_raw("test message")
        assert res is False


@pytest.mark.asyncio
async def test_wechat_notifier_send_success(notifier):
    with patch.object(settings, "wechat_work_enabled", True), \
         patch.object(settings, "wechat_work_webhook_url", "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=fake"):
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await notifier._send_markdown_raw("test message")
            assert res is True
            mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_wechat_notifier_daily_cap(notifier):
    with patch.object(settings, "wechat_work_enabled", True), \
         patch.object(settings, "wechat_work_webhook_url", "https://fake.url"), \
         patch.object(settings, "wechat_max_daily_messages", 2):
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            # 发送 2 条
            assert await notifier._send_markdown_raw("msg 1") is True
            assert await notifier._send_markdown_raw("msg 2") is True
            # 第 3 条超限
            assert await notifier._send_markdown_raw("msg 3") is False
            assert mock_post.call_count == 2


def test_wechat_radar_dedup(notifier):
    with patch.object(settings, "wechat_radar_enabled", True), \
         patch.object(notifier, "push_markdown") as mock_push:
        
        # 第一次触发
        notifier.notify_pre_trade_radar(
            radar_type="15m 经典孕线反转",
            channel="hm_inside_15m_v2",
            direction="DOWN",
            target_time_ms=1725900000000,
            lead_seconds=90,
            max_exec_price=0.30,
            features_desc="前根大实体，当前孕线",
            dedup_key="radar_test_1",
        )
        assert mock_push.call_count == 1
        call_arg = mock_push.call_args[0][0]
        assert "潜在限价单提前预警" in call_arg
        assert "hm_inside_15m_v2" in call_arg
        assert "DOWN" in call_arg
        assert "0.3000" in call_arg

        # 短时间内相同 key 触发应去重
        notifier.notify_pre_trade_radar(
            radar_type="15m 经典孕线反转",
            channel="hm_inside_15m_v2",
            direction="DOWN",
            target_time_ms=1725900000000,
            lead_seconds=90,
            max_exec_price=0.30,
            features_desc="前根大实体，当前孕线",
            dedup_key="radar_test_1",
        )
        assert mock_push.call_count == 1  # 依然是 1 次


def test_wechat_notifier_channel_formatting(notifier):
    with patch.object(settings, "wechat_radar_enabled", True), \
         patch.object(notifier, "push_markdown") as mock_push:
        
        # 1. 提前预警雷达
        notifier.notify_pre_trade_radar(
            radar_type="15m 经典孕线反转",
            channel="hm_inside_15m_v2",
            direction="DOWN",
            target_time_ms=1725900000000,
            lead_seconds=90,
            max_exec_price=0.30,
            features_desc="前根大实体",
            dedup_key="radar_test_fmt",
        )
        assert mock_push.called
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 2. 实盘订单成交
        notifier.notify_order_filled(
            channel="quote_contrarian_v2",
            direction="DOWN",
            window_start=1725900000000,
            avg_price=0.25,
            amount_usdt=5.0,
            shares=20.0,
            order_id=123,
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**报价反向·门禁版** (`quote_contrarian_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 3. 护栏弃单
        notifier.notify_order_abandoned(
            channel="x4_v2",
            direction="DOWN",
            window_start=1725900000000,
            quote_price=0.55,
            guard_price=0.50,
            reason="报价超出护栏",
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**情绪错位·平静市门禁版** (`x4_v2`)" in content

    with patch.object(notifier, "push_markdown") as mock_push:
        # 4. 结算复盘
        notifier.notify_order_settled(
            channel="s5_deep_z20_v1",
            window_start=1725900000000,
            direction="DOWN",
            outcome="DOWN",
            win=True,
            pnl=3.5,
            amount_usdt=5.0,
            settle_price=65000.0,
        )
        content = mock_push.call_args[0][0]
        assert "> **策略通道**：**S5深档·深回落门禁版** (`s5_deep_z20_v1`)" in content


@pytest.mark.asyncio
async def test_wxpusher_spt_send(notifier):
    with patch.object(settings, "wxpusher_enabled", True), \
         patch.object(settings, "wxpusher_spt", "spt_test123"):
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 1000, "msg": "处理成功"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await notifier._send_wxpusher_raw("wxpusher content")
            assert res is True
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert "simple-push" in call_url
            call_json = mock_post.call_args[1]["json"]
            assert call_json["spt"] == "spt_test123"
            assert call_json["contentType"] == 3


@pytest.mark.asyncio
async def test_wxpusher_app_token_send(notifier):
    with patch.object(settings, "wxpusher_enabled", True), \
         patch.object(settings, "wxpusher_spt", ""), \
         patch.object(settings, "wxpusher_app_token", "AT_test456"), \
         patch.object(settings, "wxpusher_uids", "UID_aaa, UID_bbb"):
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"code": 1000, "msg": "处理成功"}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            res = await notifier._send_wxpusher_raw("wxpusher content")
            assert res is True
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert call_url == "https://wxpusher.zjiecode.com/api/send/message"
            call_json = mock_post.call_args[1]["json"]
            assert call_json["appToken"] == "AT_test456"
            assert call_json["uids"] == ["UID_aaa", "UID_bbb"]
            assert call_json["contentType"] == 3

