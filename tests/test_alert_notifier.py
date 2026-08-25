"""AlertNotifier 去重抑制逻辑的单元测试。

覆盖 services/alerting.py 的 AlertNotifier：同一告警 code 在抑制窗口内只推一次、
窗口过后可再推、OK 状态不推。邮件/webhook 的实际发送由 settings 门控（默认关闭），
本测试聚焦去重决策本身，不触发外部 I/O。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from binance_predict.config.settings import settings
from binance_predict.models.schemas import HealthAlert, HealthReport
from binance_predict.services.alerting import AlertNotifier


def _alert(code: str, level: str = "CRITICAL") -> HealthAlert:
    return HealthAlert(level=level, code=code, message=f"msg-{code}")


def _report(status: str, alerts: list[HealthAlert]) -> HealthReport:
    return HealthReport(
        generated_at=datetime.now(timezone.utc),
        overall_status=status,
        alerts=alerts,
        summary="test",
    )


def test_filter_fresh_first_time_all_pass() -> None:
    n = AlertNotifier()
    alerts = [_alert("WINDOW_STALE"), _alert("NO_MATCH", "WARN")]
    fresh = n.filter_fresh(alerts, now=1000.0)
    assert len(fresh) == 2


def test_filter_fresh_suppressed_within_window() -> None:
    n = AlertNotifier()
    alerts = [_alert("WINDOW_STALE")]
    now = 1000.0
    n.mark_sent(n.filter_fresh(alerts, now), now)
    # 窗口内再次判定 → 被抑制
    within = now + settings.agent_alert_suppress_seconds - 1
    assert n.filter_fresh(alerts, within) == []


def test_filter_fresh_returns_after_window() -> None:
    n = AlertNotifier()
    alerts = [_alert("WINDOW_STALE")]
    now = 1000.0
    n.mark_sent(n.filter_fresh(alerts, now), now)
    after = now + settings.agent_alert_suppress_seconds + 1
    fresh = n.filter_fresh(alerts, after)
    assert len(fresh) == 1


def test_filter_fresh_new_code_not_suppressed() -> None:
    n = AlertNotifier()
    now = 1000.0
    n.mark_sent(n.filter_fresh([_alert("WINDOW_STALE")], now), now)
    # 同一时刻出现的新 code 不受既有抑制影响
    fresh = n.filter_fresh([_alert("WINDOW_STALE"), _alert("LLM_FAILURES")], now + 10)
    codes = {a.code for a in fresh}
    assert codes == {"LLM_FAILURES"}


def test_warn_suppressed_longer_than_critical() -> None:
    """分级抑制（2026-08-15）：WARN 用 4h 窗口，CRITICAL 用 15min 窗口。

    慢性 WARN（如 NO_MATCH）在 CRITICAL 窗口过后仍被抑制，
    避免修复邮件配置后一天轰炸 96 封。
    """
    n = AlertNotifier()
    now = 1000.0
    n.mark_sent(n.filter_fresh([_alert("NO_MATCH", "WARN")], now), now)
    # 15 分钟（CRITICAL 窗口）后：WARN 仍被抑制
    after_critical_window = now + settings.agent_alert_suppress_seconds + 1
    assert n.filter_fresh([_alert("NO_MATCH", "WARN")], after_critical_window) == []
    # 4 小时（WARN 窗口）后：可重推
    after_warn_window = now + settings.agent_alert_suppress_warn_seconds + 1
    assert len(n.filter_fresh([_alert("NO_MATCH", "WARN")], after_warn_window)) == 1


def test_critical_still_uses_short_window() -> None:
    """CRITICAL 级不受 WARN 长窗口影响：15 分钟后可重推（真故障要及时知道）。"""
    n = AlertNotifier()
    now = 1000.0
    n.mark_sent(n.filter_fresh([_alert("WINDOW_STALE")], now), now)
    after = now + settings.agent_alert_suppress_seconds + 1
    assert len(n.filter_fresh([_alert("WINDOW_STALE")], after)) == 1


@pytest.mark.asyncio
async def test_notify_skips_ok_status(monkeypatch) -> None:
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", True)
    n = AlertNotifier()
    sent = await n.notify(_report("OK", []))
    assert sent == []


@pytest.mark.asyncio
async def test_notify_dedups_across_calls(monkeypatch) -> None:
    # 钉死总闸开启（本地 .env 可能已置 false 暂停告警，与本用例无关）
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", True)
    n = AlertNotifier()
    report = _report("CRITICAL", [_alert("WINDOW_STALE")])
    # 首次推送返回该告警（邮件/webhook 均未配置，不触发外部 I/O）
    first = await n.notify(report)
    assert len(first) == 1
    # 立即二次调用（远小于抑制窗口）→ 被抑制
    second = await n.notify(report)
    assert second == []


@pytest.mark.asyncio
async def test_notify_paused_by_master_switch(monkeypatch) -> None:
    """告警推送总闸（2026-08-25）：notify_enabled=False 时全暂停且不 mark_sent，
    恢复后新告警能及时推送；信号推送开关与之解耦（由 signal_notify 测试覆盖）。"""
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", False)
    n = AlertNotifier()
    report = _report("CRITICAL", [_alert("WINDOW_STALE")])
    assert await n.notify(report) == []
    # 暂停期间不 mark_sent：恢复后同一告警立即推送（不吞告警）
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", True)
    assert len(await n.notify(report)) == 1


@pytest.mark.asyncio
async def test_notify_fund_critical_bypasses_master_switch(monkeypatch) -> None:
    """资金类独立通道（R2）：总闸关闭时 ORDER_STUCK_PENDING 仍推送，
    常规告警（WINDOW_STALE）静默且不被 mark_sent（恢复后不吞告警）。"""
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", False)
    n = AlertNotifier()
    fund = _alert("ORDER_STUCK_PENDING")
    normal = _alert("WINDOW_STALE")
    report = _report("CRITICAL", [fund, normal])

    sent = await n.notify(report)
    assert [a.code for a in sent] == ["ORDER_STUCK_PENDING"]

    # 常规告警未被 mark_sent：恢复总闸后立即推送
    monkeypatch.setattr(settings, "agent_alert_notify_enabled", True)
    reopened = await n.notify(report)
    codes = {a.code for a in reopened}
    assert "WINDOW_STALE" in codes
    assert "ORDER_STUCK_PENDING" not in codes  # 已在窗口内推过


@pytest.mark.asyncio
async def test_send_plain_email_gated_by_signal_switch(monkeypatch) -> None:
    """send_plain_email 门控解耦：受 signal_push_email_enabled 控制，
    不再依赖 agent_alert_email_enabled（暂停告警不影响信号邮件）。"""
    from binance_predict.services import alerting

    monkeypatch.setattr(settings, "signal_push_email_enabled", False)
    monkeypatch.setattr(settings, "agent_alert_email_enabled", True)
    assert await alerting.send_plain_email("s", "b") is False

    # 信号开关开 + 告警开关关 → 照样能发（桩掉 SMTP）：
    # 门控与告警开关彻底解耦
    monkeypatch.setattr(settings, "signal_push_email_enabled", True)
    monkeypatch.setattr(settings, "agent_alert_email_enabled", False)
    monkeypatch.setattr(settings, "agent_alert_smtp_host", "smtp.test")
    monkeypatch.setattr(settings, "agent_alert_email_to", "a@b.c")
    monkeypatch.setattr(alerting, "_send_email_sync", lambda *a, **k: None)
    assert await alerting.send_plain_email("s", "b") is True

