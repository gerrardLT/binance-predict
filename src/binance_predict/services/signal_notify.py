"""信号推送公共通道（2026-08-25：全信号族邮件推送统一入口）。

背景：此前只有场景信号（fake_breakout）有邮件推送；x4 / quote_edge 影子族
按「影子纪律不发邮件」静默。用户要求所有信号都推送、且暂停 agent 告警推送，
故将信号推送与告警推送解耦：

- SMTP 物理通道复用 alerting.send_plain_email（内部走 agent_alert_smtp_* 配置）；
- 开关独立：settings.signal_push_email_enabled（告警暂停不影响信号推送）；
- 全局日上限 signal_push_max_daily_emails 防轰炸（超限仅日志，不阻塞检测循环）。

调用约定：各检测器落表成功后 fire-and-forget（asyncio.create_task）调用，
与场景信号邮件同模式——SMTP 被防火墙丢包时绝不阻塞检测循环（信号 #1 事故教训）。
冷启动回补 / 污染自愈重扫属历史重放，调用方不得触发推送（notify=False 路径）。
"""
from __future__ import annotations

import asyncio
import time

from loguru import logger

from ..config.settings import settings
from .alerting import send_plain_email

# 新鲜度闸：窗口 end_time 早于 now−此阈值的信号视为历史重放（冷启动回补、
# 停机积压、污染自愈重扫），只落表不推邮件——推送只对实时新信号生效。
# 正常链路：窗口结束后 ~2min 归档、检测器 60s 内处理，10min 余量充足。
SIGNAL_FRESH_MS = 600_000


def is_fresh_signal(window_end_ms: int, now_ms: int | None = None) -> bool:
    """信号新鲜度判定：仅实时新信号才值得推送（历史重放静默）。"""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return (now - int(window_end_ms)) <= SIGNAL_FRESH_MS


def fire_signal_email(tag: str, subject: str, body: str) -> None:
    """fire-and-forget 推送（各检测器挂钩统一入口）。

    绝不阻塞检测循环（SMTP 丢包事故教训）；异常只日志不抛。
    """

    async def _run() -> None:
        try:
            await push_signal_email(tag, subject, body, int(time.time() * 1000))
        except Exception as exc:  # 推送失败不影响检测主流程
            logger.warning("[SIGNAL] 信号推送异常 | tag={} | {} | {}",
                           tag, type(exc).__name__, exc)

    asyncio.create_task(_run(), name=f"sig_email_{tag}")


# 全局日计数（进程内）：(UTC 日序号, 当日已发送数)。进程重启清零可接受——
# 宁可重启后多发，不做持久化（推送是通知不是账务）。
_daily: tuple[int, int] = (-1, 0)


def _try_bump_daily(now_ms: int) -> bool:
    """尝试占用一个当日推送名额；跨 UTC 日自动翻转。True=名额可用。"""
    global _daily
    day = now_ms // 86_400_000
    cnt = _daily[1] if _daily[0] == day else 0
    if cnt >= settings.signal_push_max_daily_emails:
        _daily = (day, cnt)
        return False
    _daily = (day, cnt + 1)
    return True


def reset_daily_count() -> None:
    """测试钩子：重置全局日计数。"""
    global _daily
    _daily = (-1, 0)


async def push_signal_email(tag: str, subject: str, body: str, now_ms: int) -> bool:
    """推送一条信号邮件（总开关 + 全局日限双闸）。

    Args:
        tag: 信号族标记（日志用），如 "quote_edge" / "x4" / "场景"
        subject/body: 邮件主题与正文（纯文本）
        now_ms: 调用方时钟（毫秒；日计数按 UTC 日翻转）

    Returns:
        True=已发出；False=总开关关闭/超日限/SMTP 未配置/发送失败。
        任何失败路径都只记录日志，不抛出。
    """
    if not settings.signal_push_email_enabled:
        return False
    if not _try_bump_daily(now_ms):
        logger.info("[SIGNAL] 信号推送超全局日限（{}），本条仅日志 | {}",
                    settings.signal_push_max_daily_emails, subject)
        return False
    ok = await send_plain_email(subject, body)
    logger.info("[SIGNAL] 信号邮件推送 | tag={} | subject={} | ok={}", tag, subject, ok)
    return ok
