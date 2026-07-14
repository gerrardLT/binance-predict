"""
Agent 阈值告警服务

基于 MetricsCollector 的聚合数据执行阈值检查，触发条件时通过 loguru 记录告警日志。
设计为轻量级、无外部依赖，后续可扩展 webhook/邮件等通知渠道。

告警条件：
- 阶段连续失败 >= N 次
- LLM 累计成本超过日限额
- 调度队列积压超阈值
"""

from __future__ import annotations

import asyncio
import smtplib
import time
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate
from typing import TYPE_CHECKING

import httpx
from loguru import logger

from ..config.settings import settings
from .metrics import MetricsCollector

if TYPE_CHECKING:
    from ..models.schemas import HealthAlert, HealthReport


class AlertService:
    """
    阈值告警服务。

    调用 check_and_alert() 检查各项指标是否超过预设阈值，
    超阈值时通过 loguru.warning 输出 [ALERT] 前缀的告警日志。
    """

    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        # 避免重复告警：记录已触发告警的阶段/类型
        self._alerted_phase_failures: dict[str, int] = {}
        # Fix #19: 熟断器阻断标志。任一严重告警（成本超限/阶段连续失败）
        # 触发时置 True，供 evaluate_trade_gate 查询以拒绝新交易。
        self._trading_blocked: bool = False

    @property
    def trading_blocked(self) -> bool:
        """当前是否因告警而需阻断交易（Fix #19）。

        仅在 settings.agent_alert_block_trades 开启时生效。
        """
        return settings.agent_alert_block_trades and self._trading_blocked

    def check_and_alert(self) -> None:
        """执行全部告警检查。"""
        if not settings.agent_alert_enabled:
            return

        # 每轮重新评估阻断状态：先复位，再由各 _check_* 根据当前指标置位，
        # 使指标恢复正常后能自动解除阻断（Fix #19）。
        self._trading_blocked = False
        self._check_phase_failures()
        self._check_cost_limit()
        self._check_queue_depth()

    def _check_phase_failures(self) -> None:
        """检查各阶段连续失败次数是否超过阈值。"""
        threshold = settings.agent_alert_consecutive_failures
        for phase in ("PREDICT", "VALIDATE", "LEARN", "EVOLVE"):
            count = self._metrics.get_consecutive_failures(phase)
            prev_alerted = self._alerted_phase_failures.get(phase, 0)
            if count >= threshold and count > prev_alerted:
                logger.warning(
                    "[ALERT] 阶段 {} 连续失败 {} 次（阈值 {}）| "
                    "请检查 LLM 服务/数据库连接/配置参数",
                    phase,
                    count,
                    threshold,
                )
                self._alerted_phase_failures[phase] = count
                # Fix #19: 阶段连续失败超阈，置位阻断标志
                self._trading_blocked = True
            elif count == 0 and prev_alerted > 0:
                # 恢复正常后重置告警状态
                self._alerted_phase_failures[phase] = 0

    def _check_cost_limit(self) -> None:
        """检查 LLM 累计成本是否超过日限额。"""
        snapshot = self._metrics.get_snapshot()
        total_cost = snapshot.get("llm", {}).get("total_cost", 0.0)
        limit = settings.agent_alert_daily_cost_limit_usd
        if total_cost >= limit:
            logger.warning(
                "[ALERT] LLM 累计成本 {:.4f} 元已达日限额 {:.4f} 元 | "
                "建议检查调用频率或调整限额配置",
                total_cost,
                limit,
            )
            # Fix #19: 成本超限，置位阻断标志（避免失控持续花费/下单）
            self._trading_blocked = True

    def _check_queue_depth(self) -> None:
        """检查调度队列积压是否超过阈值。"""
        snapshot = self._metrics.get_snapshot()
        latest_depth = snapshot.get("queue", {}).get("latest_depth", 0)
        threshold = settings.agent_alert_queue_depth_threshold
        if latest_depth >= threshold:
            logger.warning(
                "[ALERT] 调度队列深度 {} 超过阈值 {} | "
                "可能存在阶段执行阻塞，请检查 worker 消费速度",
                latest_depth,
                threshold,
            )


def _format_alert_lines(alerts: list[HealthAlert]) -> str:
    """将告警列表格式化为多行文本，供邮件正文/日志复用。"""
    return "\n".join(
        f"  [{a.level}] {a.code}: {a.message}" for a in alerts
    ) or "  （无具体告警条目）"


def _send_email_sync(subject: str, body: str, recipients: list[str]) -> None:
    """同步 SMTP 发信（阻塞）。由 send_email_alert 经线程池调用，避免阻塞事件循环。

    端口 465 走隐式 SSL（连接即加密，QQ/163 等常用）→ smtplib.SMTP_SSL；
    其余端口（如 587）走普通连接 + 可选 STARTTLS（agent_alert_smtp_use_tls）。
    """
    sender = settings.agent_alert_email_from or settings.agent_alert_smtp_user
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("Binance-Predict Agent", sender))
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)

    host = settings.agent_alert_smtp_host
    port = settings.agent_alert_smtp_port
    timeout = settings.agent_alert_email_timeout
    use_ssl = port == 465
    smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
    with smtp_cls(host, port, timeout=timeout) as server:
        # 465 连接已是 SSL，无需 STARTTLS；其余端口按配置尝试升级为 TLS
        if not use_ssl and settings.agent_alert_smtp_use_tls:
            server.starttls()
        if settings.agent_alert_smtp_user and settings.agent_alert_smtp_password:
            server.login(settings.agent_alert_smtp_user, settings.agent_alert_smtp_password)
        server.sendmail(sender, recipients, msg.as_string())


async def send_email_alert(report: HealthReport, alerts: list[HealthAlert]) -> None:
    """将健康报告的告警以邮件推送（主渠道）。

    仅在 agent_alert_email_enabled=True、SMTP host 与收件人均已配置时发送。
    阻塞的 smtplib 调用放入线程池执行，发送失败仅告警、不抛出。
    """
    if not settings.agent_alert_email_enabled:
        return
    recipients = [x.strip() for x in settings.agent_alert_email_to.split(",") if x.strip()]
    if not settings.agent_alert_smtp_host or not recipients:
        return

    subject = f"[Agent 告警] {report.overall_status} · {len(alerts)} 项"
    body = (
        f"总体状态：{report.overall_status}\n"
        f"生成时间：{report.generated_at.isoformat()}\n\n"
        f"诊断：{report.summary}\n\n"
        f"告警条目：\n{_format_alert_lines(alerts)}\n"
    )
    try:
        await asyncio.to_thread(_send_email_sync, subject, body, recipients)
        logger.info("[ALERT] 告警邮件已发送 | 收件人={} | 条目={}", len(recipients), len(alerts))
    except Exception as exc:
        logger.warning(
            "[ALERT] 告警邮件发送失败 | error_type={} | error={}",
            type(exc).__name__, str(exc),
        )


async def send_webhook_alert(
    report: HealthReport, alerts: list[HealthAlert] | None = None
) -> None:
    """将健康报告的告警以通用 JSON POST 推送到配置的 webhook（可选备用渠道）。

    仅在 settings.agent_alert_webhook_url 非空时发送。alerts 传入去重后的子集；
    留空则回退到 report.alerts 全量。payload 为通用结构，接入钉钉/飞书/Telegram
    自定义机器人时格式各异，需按平台适配。推送失败仅告警、不抛出。
    """
    url = settings.agent_alert_webhook_url
    if not url:
        return
    items = alerts if alerts is not None else report.alerts

    payload = {
        "source": "binance-predict-agent",
        "status": report.overall_status,
        "generated_at": report.generated_at.isoformat(),
        "summary": report.summary,
        "alerts": [
            {"level": a.level, "code": a.code, "message": a.message}
            for a in items
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=settings.agent_alert_webhook_timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    "[ALERT] webhook 返回非 2xx: {} | body={}",
                    resp.status_code, resp.text[:200],
                )
    except Exception as exc:
        logger.warning(
            "[ALERT] webhook 推送失败 | error_type={} | error={}",
            type(exc).__name__, str(exc),
        )


class AlertNotifier:
    """告警通知器：对告警按 code 去重抑制后，经配置渠道（邮件+webhook）主动推送。

    抑制：同一 code 在 settings.agent_alert_suppress_seconds 窗口内只推一次，
    避免 60s 轮询导致同一问题反复轰炸。overall_status=OK 或无新告警时不推送。
    进程内单例持有 code→最近推送时刻，进程重启后抑制状态清零（可接受）。
    """

    def __init__(self) -> None:
        self._last_sent: dict[str, float] = {}

    def filter_fresh(
        self, alerts: list[HealthAlert], now: float
    ) -> list[HealthAlert]:
        """返回未被抑制（超出静默窗口或首次出现）的告警子集。纯查询，不改状态。"""
        window = settings.agent_alert_suppress_seconds
        fresh: list[HealthAlert] = []
        for a in alerts:
            last = self._last_sent.get(a.code)
            if last is None or (now - last) >= window:
                fresh.append(a)
        return fresh

    def mark_sent(self, alerts: list[HealthAlert], now: float) -> None:
        """记录这批告警的推送时刻，用于后续抑制判定。"""
        for a in alerts:
            self._last_sent[a.code] = now

    async def notify(self, report: HealthReport) -> list[HealthAlert]:
        """按抑制窗口过滤后，经邮件+webhook 推送新告警。返回实际推送的告警子集。

        overall_status=OK 时直接跳过。仅当存在「新」告警（未在窗口内推过）才推送，
        推送成功与否都刷新该批 code 的推送时刻，避免失败时下一轮立即重试轰炸。
        """
        if report.overall_status == "OK":
            return []
        now = time.time()
        fresh = self.filter_fresh(report.alerts, now)
        if not fresh:
            return []
        await send_email_alert(report, fresh)
        await send_webhook_alert(report, fresh)
        self.mark_sent(fresh, now)
        return fresh


# 进程内单例：供后台监控 loop 复用抑制状态
alert_notifier = AlertNotifier()
