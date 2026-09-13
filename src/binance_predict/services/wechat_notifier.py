"""企业微信群机器人推送与交易生命周期通知服务 (2026-09-10)。

特性：
1. 提前预警雷达 (Pre-Trade Radar)：在关键限价单挂单前（如 15m 孕线反转、5m x4 错位）
   提前 60~150 秒推送卡片，留下充足的人工看盘与核验决策缓冲时间。
2. 交易动作全闭环：覆盖挂单、成交、护栏弃单、窗口结算盈亏复盘、钱包低余额预警。
3. 零阻塞设计：网络请求全异步 (asyncio.create_task / httpx)，设置 5s 超时与全局日限。
4. 格式友好：使用企业微信 Markdown 卡片，高亮关键数据与方向。
5. 逐信号配置闸（2026-09-13）：notify_config 控制渠道/事件/字段；默认全开。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from loguru import logger

from ..config.settings import settings
from .live_channels import format_channel_title
from .notification_config import notify_config

TZ_BJT = timezone(timedelta(hours=8))


def fmt_bjt(ms: int | float | None, with_date: bool = True) -> str:
    """毫秒时间戳/秒时间戳 -> 北京时间字符串。"""
    if ms is None:
        return "N/A"
    sec = ms / 1000.0 if ms > 1e11 else float(ms)
    fmt = "%m-%d %H:%M:%S" if with_date else "%H:%M:%S"
    return datetime.fromtimestamp(sec, tz=TZ_BJT).strftime(fmt)


def _field_on(channel: str, event: str, field: str) -> bool:
    try:
        return notify_config.is_field_enabled(channel, event, field)
    except Exception:
        return True


def _event_on(channel: str, event: str) -> bool:
    try:
        return notify_config.is_event_enabled(channel, event, "wechat")
    except Exception:
        return True


class WeChatNotifier:
    """企业微信群机器人通知器。"""

    def __init__(self) -> None:
        # 全局日限计数: (UTC日序号, 发送计数)
        self._daily: tuple[int, int] = (-1, 0)
        # 预警去重集合: key -> timestamp，防止同一窗口同一事件被重复预警
        self._sent_warnings: dict[str, float] = {}

    def _try_bump_daily(self, now_ms: int) -> bool:
        """检查并自增日消息数。"""
        day = now_ms // 86_400_000
        cnt = self._daily[1] if self._daily[0] == day else 0
        if cnt >= settings.wechat_max_daily_messages:
            self._daily = (day, cnt)
            return False
        self._daily = (day, cnt + 1)
        return True

    def reset_daily_count(self) -> None:
        """测试或调试钩子：重置日计数与去重。"""
        self._daily = (-1, 0)
        self._sent_warnings.clear()

    async def _send_wxpusher_raw(self, content: str) -> bool:
        """底层异步发送 WxPusher 微信消息（永久免费，直达普通微信）。"""
        if not settings.wxpusher_enabled:
            return False

        # 方式 1: 极简 SPT 模式
        spt = settings.wxpusher_spt.strip()
        if spt:
            url = "https://wxpusher.zjiecode.com/api/send/message/simple-push"
            payload = {
                "spt": spt,
                "content": content,
                "contentType": 3,  # 3 代表 Markdown
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200 and resp.json().get("code") == 1000:
                        logger.info("[WXPUSHER] 微信极简消息推送成功")
                        return True
                    else:
                        logger.warning("[WXPUSHER] 微信极简消息推送返回: {}", resp.text[:200])
            except Exception as exc:
                logger.warning("[WXPUSHER] 微信推送网络异常: {} | {}", type(exc).__name__, exc)
            return False

        # 方式 2: 标准应用模式 (appToken + UID)
        token = settings.wxpusher_app_token.strip()
        uids = [u.strip() for u in settings.wxpusher_uids.split(",") if u.strip()]
        if token and uids:
            url = "https://wxpusher.zjiecode.com/api/send/message"
            payload = {
                "appToken": token,
                "content": content,
                "contentType": 3,  # 3 代表 Markdown
                "uids": uids,
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200 and resp.json().get("code") == 1000:
                        logger.info("[WXPUSHER] 微信应用消息推送成功 (uids={})", len(uids))
                        return True
                    else:
                        logger.warning("[WXPUSHER] 微信应用消息推送返回: {}", resp.text[:200])
            except Exception as exc:
                logger.warning("[WXPUSHER] 微信推送网络异常: {} | {}", type(exc).__name__, exc)
            return False

        return False

    async def _send_markdown_raw(self, content: str) -> bool:
        """底层异步发送企业微信 Markdown 消息。"""
        if not settings.wechat_work_enabled:
            return False
        url = settings.wechat_work_webhook_url.strip()
        if not url:
            return False

        now_ms = int(time.time() * 1000)
        if not self._try_bump_daily(now_ms):
            logger.info("[WECHAT] 企微推送已达全局日上限 ({})，本条跳过", settings.wechat_max_daily_messages)
            return False

        payload = {
            "msgtype": "markdown",
            "markdown": {
                "content": content
            }
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    res_json = resp.json()
                    if res_json.get("errcode") == 0:
                        logger.info("[WECHAT] 企微消息发送成功")
                        return True
                    else:
                        logger.warning("[WECHAT] 企微返回业务错误: {}", res_json)
                else:
                    logger.warning("[WECHAT] 企微 HTTP 请求失败: {} {}", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("[WECHAT] 企微推送网络异常: {} | {}", type(exc).__name__, exc)
        return False

    def push_markdown(self, content: str) -> None:
        """fire-and-forget 发送微信/企微 Markdown 消息（多通道聚合）。"""
        has_qy = settings.wechat_work_enabled and bool(settings.wechat_work_webhook_url.strip())
        has_wx = settings.wxpusher_enabled and (bool(settings.wxpusher_spt.strip()) or bool(settings.wxpusher_app_token.strip()))
        if not has_qy and not has_wx:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # 无运行中的事件循环（如纯同步单元测试环境），安全忽略

        async def _run() -> None:
            tasks = []
            if has_qy:
                tasks.append(self._send_markdown_raw(content))
            if has_wx:
                tasks.append(self._send_wxpusher_raw(content))
            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as exc:
                    logger.warning("[WECHAT] 消息推送任务异常: {}", exc)

        loop.create_task(_run(), name="wechat_push_markdown")

    # ------------------------------------------------------------------
    # 核心业务场景：提前预警雷达 (Pre-Trade Radar)
    # ------------------------------------------------------------------

    def notify_pre_trade_radar(
        self,
        radar_type: str,
        channel: str,
        direction: str,
        target_time_ms: int,
        lead_seconds: int,
        max_exec_price: float,
        features_desc: str,
        dedup_key: str,
        note: str = "系统将在倒计时结束后按策略自动挂单，请根据需要看盘或干预。"
    ) -> None:
        """发送提前预警通知（留出缓冲时间供人工决策）。"""
        if not settings.wechat_radar_enabled:
            return
        if not _event_on(channel, "radar"):
            return

        now = time.time()
        # 10 分钟内的同一 key 幂等去重
        if dedup_key in self._sent_warnings and (now - self._sent_warnings[dedup_key]) < 600:
            return
        self._sent_warnings[dedup_key] = now

        dir_color = "warning" if direction == "DOWN" else "info"
        dir_emoji = "📉 DOWN（看跌）" if direction == "DOWN" else "📈 UP（看涨）"

        lines = [
            f"### ⚡【潜在限价单提前预警】<{dir_color}>{radar_type}</{dir_color}>",
            f"> **策略通道**：{format_channel_title(channel)}",
        ]
        if _field_on(channel, "radar", "direction"):
            lines.append(f"> **预测方向**：**{dir_emoji}**")
        if _field_on(channel, "radar", "target_time"):
            lines.append(f"> **预计挂单时间**：`{fmt_bjt(target_time_ms)}`")
        if _field_on(channel, "radar", "lead_seconds"):
            lines.append(f"> **剩余决策缓冲**：<font color=\"comment\">约 {lead_seconds} 秒</font>")
        if _field_on(channel, "radar", "max_exec_price"):
            lines.append(f"> **限价单参考护栏**：**{max_exec_price:.4f}**")
        if _field_on(channel, "radar", "features"):
            lines.append(f"> **形态特征**：{features_desc}")
        if _field_on(channel, "radar", "note"):
            lines.append(f"\n💡 **提示**：{note}")
        self.push_markdown("\n".join(lines))

    # ------------------------------------------------------------------
    # 交易动作：挂单与成交
    # ------------------------------------------------------------------

    def notify_order_filled(
        self,
        channel: str,
        direction: str,
        window_start: int,
        avg_price: float,
        amount_usdt: float,
        shares: float | None = None,
        order_id: Any = None,
    ) -> None:
        """订单成交通知。"""
        if not _event_on(channel, "filled"):
            return
        dir_emoji = "📉 押 DOWN" if direction == "DOWN" else "📈 押 UP"
        shares_str = f"{shares:.2f}" if shares is not None else "待回读"
        lines = [
            "### 🎯【实盘订单已成交】",
            f"> **策略通道**：{format_channel_title(channel)}",
        ]
        if _field_on(channel, "filled", "window"):
            lines.append(f"> **标的周期**：`{fmt_bjt(window_start)}`")
        if _field_on(channel, "filled", "direction"):
            lines.append(f"> **下单方向**：<font color=\"info\">{dir_emoji}</font>")
        if _field_on(channel, "filled", "avg_price"):
            lines.append(f"> **成交均价**：`{avg_price:.4f}`")
        if _field_on(channel, "filled", "amount_usdt"):
            lines.append(f"> **下注金额**：`{amount_usdt:.2f} USDT`")
        if _field_on(channel, "filled", "shares"):
            lines.append(f"> **获得股数**：`{shares_str}` 股")
        if _field_on(channel, "filled", "order_id"):
            lines.append(f"> **订单 ID**：`#{order_id}`")
        if _field_on(channel, "filled", "fill_time"):
            lines.append(f"> **成交时间**：`{fmt_bjt(time.time()*1000)}`")
        self.push_markdown("\n".join(lines))

    # ------------------------------------------------------------------
    # 交易动作：护栏弃单拦截
    # ------------------------------------------------------------------

    def notify_order_abandoned(
        self,
        channel: str,
        direction: str,
        window_start: int,
        quote_price: float | None,
        guard_price: float | None,
        reason: str,
    ) -> None:
        """执行价护栏拦截弃单通知。"""
        if not _event_on(channel, "abandoned"):
            return
        q_str = f"{quote_price:.4f}" if quote_price is not None else "N/A"
        g_str = f"{guard_price:.4f}" if guard_price is not None else "N/A"
        lines = [
            "### 🛡️【护栏安全弃单保护】",
            f"> **策略通道**：{format_channel_title(channel)}",
        ]
        if _field_on(channel, "abandoned", "window"):
            lines.append(f"> **标的周期**：`{fmt_bjt(window_start)}`")
        if _field_on(channel, "abandoned", "direction"):
            lines.append(f"> **意向方向**：`{direction}`")
        if _field_on(channel, "abandoned", "quote_price") or _field_on(channel, "abandoned", "guard_price"):
            q_part = f"`{q_str}`" if _field_on(channel, "abandoned", "quote_price") else "N/A"
            g_part = f"`{g_str}`" if _field_on(channel, "abandoned", "guard_price") else "N/A"
            lines.append(f"> **市场报价**：{q_part} (护栏上限: {g_part})")
        if _field_on(channel, "abandoned", "reason"):
            lines.append(f"> **弃单原因**：<font color=\"comment\">{reason}</font>")
        if _field_on(channel, "abandoned", "safety_note"):
            lines.append("> **安全机制**：当前价格溢价过高或贴线，系统主动放弃该单以保护本金。")
        self.push_markdown("\n".join(lines))

    # ------------------------------------------------------------------
    # 结算动作：窗口复盘与盈亏
    # ------------------------------------------------------------------

    def notify_order_settled(
        self,
        channel: str,
        window_start: int,
        direction: str,
        outcome: str,
        win: bool | None,
        pnl: float | None,
        amount_usdt: float | None,
        settle_price: float | None = None,
    ) -> None:
        """订单结算复盘通知。"""
        if not _event_on(channel, "settled"):
            return
        if win is True:
            res_tag = "💰 获胜 (WIN)"
            res_color = "info"
        elif win is False:
            res_tag = "❌ 亏损 (LOSS)"
            res_color = "warning"
        else:
            res_tag = "⚖️ 平局/无效 (NOISE)"
            res_color = "comment"

        pnl_str = f"{pnl:+.4f} USDT" if pnl is not None else "0.0000 USDT"
        roi_str = ""
        if pnl is not None and amount_usdt and amount_usdt > 0:
            roi_str = f" ({pnl / amount_usdt * 100:+.1f}%)"

        s_price_str = f"{settle_price:.2f}" if settle_price is not None else "N/A"

        lines = [
            f"### 🏁【实盘订单结算复盘】<{res_color}>{res_tag}</{res_color}>",
            f"> **策略通道**：{format_channel_title(channel)}",
        ]
        if _field_on(channel, "settled", "window"):
            lines.append(f"> **所属窗口**：`{fmt_bjt(window_start)}`")
        if _field_on(channel, "settled", "direction_vs_outcome"):
            lines.append(f"> **下注 vs 结算**：下注 `{direction}` ➜ 最终 `{outcome}`")
        if _field_on(channel, "settled", "settle_price"):
            lines.append(f"> **结算 BTC 价格**：`{s_price_str}`")
        if _field_on(channel, "settled", "pnl"):
            roi_part = roi_str if _field_on(channel, "settled", "roi") else ""
            lines.append(f"> **本单净盈亏**：**{pnl_str}**{roi_part}")
        elif _field_on(channel, "settled", "roi") and roi_str:
            lines.append(f"> **收益率**：{roi_str.strip()}")
        if _field_on(channel, "settled", "settle_time"):
            lines.append(f"> **结算时间**：`{fmt_bjt(time.time()*1000)}`")
        self.push_markdown("\n".join(lines))

    # ------------------------------------------------------------------
    # 系统状态：预测钱包低余额预警
    # ------------------------------------------------------------------

    def notify_low_balance(self, balance_usdt: float, spot_usdt: float | None = None) -> None:
        """预测钱包余额过低警告。"""
        try:
            if not notify_config.is_low_balance_enabled():
                return
        except Exception:
            pass
        dedup_key = f"low_balance_{int(time.time() // 3600)}"  # 1小时最多推一次
        now = time.time()
        if dedup_key in self._sent_warnings:
            return
        self._sent_warnings[dedup_key] = now

        def _lb(field: str) -> bool:
            try:
                return notify_config.is_low_balance_field_enabled(field)
            except Exception:
                return True

        lines = ["### ⚠️【预测钱包可用余额不足提醒】"]
        if _lb("balance"):
            spot_str = ""
            if spot_usdt is not None and _lb("spot"):
                spot_str = f" (现货可用: {spot_usdt:.2f} USDT)"
            lines.append(
                f"> **预测钱包当前余额**：<font color=\"warning\">{balance_usdt:.2f} USDT</font>{spot_str}"
            )
        elif spot_usdt is not None and _lb("spot"):
            lines.append(f"> **现货可用**：`{spot_usdt:.2f} USDT`")
        if _lb("threshold"):
            lines.append("> **警戒线**：`10.00 USDT`")
        if _lb("advice"):
            lines.append("> **建议**：余额不足将导致后续信号无法成交挂单，请尽快前往管理页面或通过 API 划转入金。")
        self.push_markdown("\n".join(lines))


# 进程全局单例
wechat_notifier = WeChatNotifier()
