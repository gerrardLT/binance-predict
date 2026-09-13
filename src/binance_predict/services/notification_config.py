"""逐信号通知配置：默认全开覆盖层 + 热路径同步闸门。

物理凭据仍只在 .env（WxPusher SPT/SMTP）；本模块只管路由与字段显隐。
无覆盖行 / 缺键 / 首次 DB 失败 → 全部开启，兼容既有线上行为。
热路径 is_event_enabled / is_field_enabled 同步零延迟；toggle 写 DB 后
本进程立即生效，多 worker 靠 60s 后台刷新收敛。
"""
from __future__ import annotations

import asyncio
import copy
import time
from typing import Any

from loguru import logger
from sqlalchemy import delete as sa_delete
from sqlalchemy import select as sa_select

from binance_predict.config.settings import settings
from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import NotificationChannelOverride

REFRESH_INTERVAL = 60.0
GLOBAL_CHANNEL = "__global__"

TRANSPORTS = ("wechat", "email")

EVENT_CATALOG: dict[str, dict[str, Any]] = {
    "radar": {
        "label": "提前预警雷达",
        "hint": "挂单前 60~150 秒推送，留给人工看盘缓冲。",
        "transports": ("wechat",),
        "fields": {
            "direction": {"label": "预测方向", "hint": "UP/DOWN 及看涨看跌标识"},
            "target_time": {"label": "预计挂单时间", "hint": "北京时间决策点"},
            "lead_seconds": {"label": "剩余决策缓冲", "hint": "距挂单剩余秒数"},
            "max_exec_price": {"label": "限价单参考护栏", "hint": "执行价上限"},
            "features": {"label": "形态特征", "hint": "触发时的形态/门禁描述"},
            "note": {"label": "提示", "hint": "人工干预说明"},
        },
    },
    "filled": {
        "label": "实盘成交通知",
        "hint": "真实成交后推送金额、均价与股数。",
        "transports": ("wechat",),
        "fields": {
            "window": {"label": "标的周期", "hint": "下注窗口北京时间"},
            "direction": {"label": "下单方向", "hint": "押 UP/DOWN"},
            "avg_price": {"label": "成交均价", "hint": "实际成交价"},
            "amount_usdt": {"label": "下注金额", "hint": "USDT 投入"},
            "shares": {"label": "获得股数", "hint": "成交份额"},
            "order_id": {"label": "订单 ID", "hint": "本地订单号"},
            "fill_time": {"label": "成交时间", "hint": "北京时间"},
        },
    },
    "abandoned": {
        "label": "护栏弃单通知",
        "hint": "报价超护栏或入场价白名单拦截时推送。",
        "transports": ("wechat",),
        "fields": {
            "window": {"label": "标的周期", "hint": "拟下注窗口"},
            "direction": {"label": "意向方向", "hint": "原计划方向"},
            "quote_price": {"label": "市场报价", "hint": "被拦截时的报价"},
            "guard_price": {"label": "护栏上限", "hint": "通道执行价护栏"},
            "reason": {"label": "弃单原因", "hint": "拦截文案"},
            "safety_note": {"label": "安全机制说明", "hint": "弃单保护提示"},
        },
    },
    "settled": {
        "label": "结算复盘",
        "hint": "窗口结算后微信复盘与邮件复盘分开控制。",
        "transports": ("wechat", "email"),
        "fields": {
            "window": {"label": "所属窗口", "hint": "结算窗口北京时间"},
            "direction_vs_outcome": {"label": "下注 vs 结算", "hint": "方向对照"},
            "settle_price": {"label": "结算 BTC 价格", "hint": "周期锚点结算价"},
            "pnl": {"label": "本单净盈亏", "hint": "USDT 盈亏"},
            "roi": {"label": "收益率", "hint": "相对投入百分比"},
            "settle_time": {"label": "结算时间", "hint": "微信复盘时间戳"},
            "trigger": {"label": "触发报价/时点", "hint": "邮件：触发 t 与 q"},
            "condition": {"label": "触发条件", "hint": "邮件：形态/门禁条件"},
            "bet": {"label": "押注说明", "hint": "邮件：押注方向与目标窗"},
            "entry": {"label": "入场报价", "hint": "邮件：决策点报价"},
            "outcome": {"label": "结算结果", "hint": "赢/输/平及方向"},
            "ev": {"label": "入场 EV", "hint": "邮件：期望值公式结果"},
            "confirm_time": {"label": "确认时间", "hint": "场景邮件确认时刻"},
            "scene": {"label": "场景类型", "hint": "场景邮件形态名"},
            "break_line": {"label": "破位/确认依据", "hint": "场景邮件破位描述"},
            "close_pos": {"label": "收盘质量", "hint": "场景邮件 close_pos"},
            "vol_ratio": {"label": "量比", "hint": "场景邮件量比"},
            "target_window": {"label": "目标周期", "hint": "场景邮件次周期到期"},
            "entry_quote": {"label": "当时 token 报价", "hint": "场景邮件入场报价"},
            "entry_plan": {"label": "入场方案", "hint": "场景邮件操作建议"},
            "backtest": {"label": "回测依据", "hint": "场景邮件历史样本"},
            "mechanism": {"label": "机制说明", "hint": "场景邮件机制段"},
            "disclaimer": {"label": "阶段说明", "hint": "场景邮件「当前阶段」脚注"},
        },
    },
}

LOW_BALANCE_FIELDS: dict[str, dict[str, str]] = {
    "balance": {"label": "预测钱包余额", "hint": "当前可用 USDT"},
    "spot": {"label": "现货可用余额", "hint": "现货账户对照"},
    "threshold": {"label": "警戒线", "hint": "默认 10 USDT"},
    "advice": {"label": "建议", "hint": "划转入金提示"},
}

FAMILY_LABELS: dict[str, str] = {
    "quote_edge": "报价边缘",
    "x4": "情绪错位",
    "scene": "场景信号",
    "s2_cond": "S2 条件单",
    "nextbar": "下一根确认",
    "absorption": "吸收跟随",
    "firsthit": "首触反转",
}


def _all_true(keys: dict[str, Any]) -> dict[str, bool]:
    return {k: True for k in keys}


def default_event_block(event: str) -> dict[str, Any]:
    spec = EVENT_CATALOG[event]
    block: dict[str, Any] = {
        transport: True for transport in spec["transports"]
    }
    block["fields"] = _all_true(spec["fields"])
    return block


def default_channel_config() -> dict[str, Any]:
    return {
        "events": {event: default_event_block(event) for event in EVENT_CATALOG},
    }


def default_global_config() -> dict[str, Any]:
    return {
        "wechat_enabled": True,
        "email_enabled": True,
        "low_balance": {
            "wechat": True,
            "fields": _all_true(LOW_BALANCE_FIELDS),
        },
    }


def _as_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _merge_fields(allowed: dict[str, Any], incoming: Any) -> dict[str, bool]:
    src = incoming if isinstance(incoming, dict) else {}
    return {key: _as_bool(src.get(key), True) for key in allowed}


def normalize_channel_config(raw: Any) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    events_in = src.get("events") if isinstance(src.get("events"), dict) else {}
    events: dict[str, Any] = {}
    for event, spec in EVENT_CATALOG.items():
        block_in = events_in.get(event) if isinstance(events_in.get(event), dict) else {}
        block: dict[str, Any] = {
            transport: _as_bool(block_in.get(transport), True)
            for transport in spec["transports"]
        }
        block["fields"] = _merge_fields(spec["fields"], block_in.get("fields"))
        events[event] = block
    return {"events": events}


def normalize_global_config(raw: Any) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    lb_in = src.get("low_balance") if isinstance(src.get("low_balance"), dict) else {}
    return {
        "wechat_enabled": _as_bool(src.get("wechat_enabled"), True),
        "email_enabled": _as_bool(src.get("email_enabled"), True),
        "low_balance": {
            "wechat": _as_bool(lb_in.get("wechat"), True),
            "fields": _merge_fields(LOW_BALANCE_FIELDS, lb_in.get("fields")),
        },
    }


def _live_channels() -> dict:
    from binance_predict.services.live_channels import LIVE_CHANNELS
    return LIVE_CHANNELS


def validate_channel_id(channel: str) -> str:
    if channel == GLOBAL_CHANNEL:
        return channel
    if channel not in _live_channels():
        raise ValueError(f"未知信号通道: {channel}")
    return channel


def validate_event(event: str) -> str:
    if event not in EVENT_CATALOG:
        raise ValueError(f"未知通知事件: {event}")
    return event


def validate_transport(event: str, transport: str) -> str:
    if transport not in EVENT_CATALOG[event]["transports"]:
        raise ValueError(f"事件 {event} 不支持渠道 {transport}")
    return transport


def physical_status() -> dict[str, Any]:
    wx_spt = bool(settings.wxpusher_spt.strip())
    wx_app = bool(settings.wxpusher_app_token.strip()) and bool(
        settings.wxpusher_uids.strip()
    )
    email_to = [
        x.strip() for x in settings.agent_alert_email_to.split(",") if x.strip()
    ]
    return {
        "wechat": {
            "env_enabled": bool(settings.wxpusher_enabled or settings.wechat_work_enabled),
            "wxpusher_configured": wx_spt or wx_app,
            "wechat_work_configured": bool(settings.wechat_work_webhook_url.strip()),
            "radar_env_enabled": bool(settings.wechat_radar_enabled),
        },
        "email": {
            "env_enabled": bool(settings.signal_push_email_enabled),
            "smtp_configured": bool(settings.agent_alert_smtp_host.strip()) and bool(email_to),
            "scene_email_env_enabled": bool(settings.fake_breakout_email_enabled),
        },
    }


def catalog_payload() -> dict[str, Any]:
    return {
        "global_channel": GLOBAL_CHANNEL,
        "events": {
            event: {
                "label": spec["label"],
                "hint": spec["hint"],
                "transports": list(spec["transports"]),
                "fields": [
                    {"id": fid, "label": fmeta["label"], "hint": fmeta["hint"]}
                    for fid, fmeta in spec["fields"].items()
                ],
            }
            for event, spec in EVENT_CATALOG.items()
        },
        "low_balance_fields": [
            {"id": fid, "label": fmeta["label"], "hint": fmeta["hint"]}
            for fid, fmeta in LOW_BALANCE_FIELDS.items()
        ],
        "families": FAMILY_LABELS,
    }


class NotificationConfigService:
    """内存缓存 {channel: {enabled, config}}；缺行按默认全开。"""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._loaded_at = 0.0
        self._running = False
        self._task: asyncio.Task | None = None

    def _row(self, channel: str) -> dict[str, Any] | None:
        return self._rows.get(channel)

    def global_config(self) -> dict[str, Any]:
        row = self._row(GLOBAL_CHANNEL)
        if row is None:
            return default_global_config()
        return normalize_global_config(row.get("config"))

    def channel_enabled(self, channel: str) -> bool:
        row = self._row(channel)
        if row is None:
            return True
        return _as_bool(row.get("enabled"), True)

    def channel_config(self, channel: str) -> dict[str, Any]:
        row = self._row(channel)
        if row is None:
            return default_channel_config()
        return normalize_channel_config(row.get("config"))

    def is_transport_globally_enabled(self, transport: str) -> bool:
        cfg = self.global_config()
        if transport == "wechat":
            return bool(cfg["wechat_enabled"])
        if transport == "email":
            return bool(cfg["email_enabled"])
        return False

    def is_event_enabled(self, channel: str, event: str, transport: str) -> bool:
        if event not in EVENT_CATALOG:
            return True
        if transport not in EVENT_CATALOG[event]["transports"]:
            return False
        if not self.is_transport_globally_enabled(transport):
            return False
        if channel == GLOBAL_CHANNEL or channel not in _live_channels():
            return True
        if not self.channel_enabled(channel):
            return False
        block = self.channel_config(channel)["events"][event]
        return _as_bool(block.get(transport), True)

    def is_field_enabled(self, channel: str, event: str, field: str) -> bool:
        if event not in EVENT_CATALOG:
            return True
        if field not in EVENT_CATALOG[event]["fields"]:
            return True
        fields = self.channel_config(channel)["events"][event]["fields"]
        return _as_bool(fields.get(field), True)

    def is_low_balance_enabled(self) -> bool:
        if not self.is_transport_globally_enabled("wechat"):
            return False
        return _as_bool(self.global_config()["low_balance"].get("wechat"), True)

    def is_low_balance_field_enabled(self, field: str) -> bool:
        if field not in LOW_BALANCE_FIELDS:
            return True
        return _as_bool(self.global_config()["low_balance"]["fields"].get(field), True)

    def snapshot(self) -> dict[str, Any]:
        channels = []
        for channel, spec in _live_channels().items():
            row = self._row(channel)
            channels.append({
                "channel": channel,
                "display_name": spec.display_name,
                "family": spec.family,
                "family_label": FAMILY_LABELS.get(spec.family, spec.family),
                "market_period": spec.market_period,
                "enabled": self.channel_enabled(channel),
                "config": self.channel_config(channel),
                "overridden": row is not None,
                "updated_at": row.get("updated_at") if row else None,
            })
        g_row = self._row(GLOBAL_CHANNEL)
        return {
            "catalog": catalog_payload(),
            "physical": physical_status(),
            "global": {
                "channel": GLOBAL_CHANNEL,
                "config": self.global_config(),
                "overridden": g_row is not None,
                "updated_at": g_row.get("updated_at") if g_row else None,
            },
            "channels": channels,
        }

    def _store_row(
        self,
        channel: str,
        enabled: bool,
        config: dict[str, Any],
        updated_at: Any = None,
    ) -> None:
        self._rows[channel] = {
            "enabled": enabled,
            "config": copy.deepcopy(config),
            "updated_at": (
                updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at
            ),
        }

    async def upsert(
        self,
        channel: str,
        *,
        enabled: bool | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        channel = validate_channel_id(channel)
        is_global = channel == GLOBAL_CHANNEL
        current = (
            self.global_config() if is_global else self.channel_config(channel)
        )
        next_enabled = True if is_global else (
            self.channel_enabled(channel) if enabled is None else bool(enabled)
        )
        if config is None:
            next_config = current
        else:
            next_config = (
                normalize_global_config(config)
                if is_global
                else normalize_channel_config(config)
            )
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(NotificationChannelOverride).where(
                    NotificationChannelOverride.channel == channel
                )
            )).scalar_one_or_none()
            if row is None:
                row = NotificationChannelOverride(
                    channel=channel,
                    enabled=next_enabled,
                    config=next_config,
                )
                session.add(row)
            else:
                row.enabled = next_enabled
                row.config = next_config
            await session.commit()
            updated_at = row.updated_at
        self._store_row(channel, next_enabled, next_config, updated_at)
        self._loaded_at = time.monotonic()
        logger.info("通知配置已更新 | {} | enabled={}", channel, next_enabled)
        if is_global:
            return self.global_config()
        return {
            "channel": channel,
            "enabled": next_enabled,
            "config": next_config,
        }

    async def reset(self, channel: str | None = None) -> None:
        async with async_session_factory() as session:
            if channel is None:
                await session.execute(sa_delete(NotificationChannelOverride))
            else:
                channel = validate_channel_id(channel)
                await session.execute(
                    sa_delete(NotificationChannelOverride).where(
                        NotificationChannelOverride.channel == channel
                    )
                )
            await session.commit()
        if channel is None:
            self._rows = {}
        else:
            self._rows.pop(channel, None)
        self._loaded_at = time.monotonic()
        logger.info("通知配置已恢复默认 | {}", channel or "全部")

    async def refresh(self) -> None:
        try:
            async with async_session_factory() as session:
                rows = (await session.execute(
                    sa_select(NotificationChannelOverride)
                )).scalars().all()
            next_rows: dict[str, dict[str, Any]] = {}
            for row in rows:
                cfg = (
                    normalize_global_config(row.config)
                    if row.channel == GLOBAL_CHANNEL
                    else normalize_channel_config(row.config)
                )
                next_rows[str(row.channel)] = {
                    "enabled": True if row.channel == GLOBAL_CHANNEL else bool(row.enabled),
                    "config": cfg,
                    "updated_at": (
                        row.updated_at.isoformat()
                        if getattr(row, "updated_at", None) is not None
                        and hasattr(row.updated_at, "isoformat")
                        else None
                    ),
                }
            self._rows = next_rows
            self._loaded_at = time.monotonic()
        except Exception as exc:
            logger.warning("通知配置刷新失败（保守维持现状/默认全开）| {}", exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.refresh()
        self._task = asyncio.create_task(self._loop(), name="notification_config")
        logger.info("通知配置服务启动 | 覆盖 {} 行", len(self._rows))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(REFRESH_INTERVAL)
                await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("通知配置后台刷新异常 | {}", exc)


notify_config = NotificationConfigService()
