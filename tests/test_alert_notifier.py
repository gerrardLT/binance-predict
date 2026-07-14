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


@pytest.mark.asyncio
async def test_notify_skips_ok_status() -> None:
    n = AlertNotifier()
    sent = await n.notify(_report("OK", []))
    assert sent == []


@pytest.mark.asyncio
async def test_notify_dedups_across_calls() -> None:
    n = AlertNotifier()
    report = _report("CRITICAL", [_alert("WINDOW_STALE")])
    # 首次推送返回该告警（邮件/webhook 均未配置，不触发外部 I/O）
    first = await n.notify(report)
    assert len(first) == 1
    # 立即二次调用（远小于抑制窗口）→ 被抑制
    second = await n.notify(report)
    assert second == []
