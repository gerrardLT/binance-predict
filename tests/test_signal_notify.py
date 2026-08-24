"""signal_notify 信号推送公共通道的单元测试（2026-08-25 全信号推送落地）。

覆盖：新鲜度闸（历史重放静默）、总开关、全局日限防轰炸、跨 UTC 日翻转、
fire-and-forget 入口异常隔离。不触发真实 SMTP（send_plain_email 被桩掉）。
"""
from __future__ import annotations

import pytest

from binance_predict.config.settings import settings
from binance_predict.services import signal_notify as sn


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """每个用例独立的全局日计数与开关状态。"""
    sn.reset_daily_count()
    monkeypatch.setattr(settings, "signal_push_email_enabled", True)
    monkeypatch.setattr(settings, "signal_push_max_daily_emails", 3)
    yield
    sn.reset_daily_count()


# ------------------------------------------------------------------
# 新鲜度闸：回补/积压/污染重扫的历史重放不得推送
# ------------------------------------------------------------------

def test_is_fresh_signal_boundary() -> None:
    now = 2_000_000_000_000
    # 恰在阈值内 → 新鲜；超 1ms → 历史重放
    assert sn.is_fresh_signal(now - sn.SIGNAL_FRESH_MS, now) is True
    assert sn.is_fresh_signal(now - sn.SIGNAL_FRESH_MS - 1, now) is False


# ------------------------------------------------------------------
# 总开关 + 全局日限
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_push_disabled_by_master_switch(monkeypatch) -> None:
    monkeypatch.setattr(settings, "signal_push_email_enabled", False)
    called = []

    async def _fake(sub, body):
        called.append(sub)
        return True

    monkeypatch.setattr(sn, "send_plain_email", _fake)
    assert await sn.push_signal_email("x4", "s", "b", 1000) is False
    assert called == []  # 总开关关闭时不占用 SMTP 也不计日限


@pytest.mark.asyncio
async def test_push_respects_daily_cap(monkeypatch) -> None:
    """日限 3：前 3 条发出，第 4 条仅日志；跨 UTC 日后名额恢复。"""
    sent = []

    async def _fake(sub, body):
        sent.append(sub)
        return True

    monkeypatch.setattr(sn, "send_plain_email", _fake)
    day_ms = 20_000 * 86_400_000  # 某 UTC 日零点
    for i in range(3):
        assert await sn.push_signal_email("x4", f"s{i}", "b", day_ms + i) is True
    # 同日第 4 条超日限
    assert await sn.push_signal_email("x4", "s3", "b", day_ms + 100) is False
    assert len(sent) == 3
    # 跨 UTC 日 → 计数翻转，恢复推送
    assert await sn.push_signal_email("x4", "s4", "b", day_ms + 86_400_000) is True
    assert len(sent) == 4


# ------------------------------------------------------------------
# fire-and-forget 入口：绝不向检测循环抛异常
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fire_signal_email_swallows_errors(monkeypatch) -> None:
    async def _boom(tag, subject, body, now_ms):
        raise RuntimeError("smtp down")

    monkeypatch.setattr(sn, "push_signal_email", _boom)
    # 同步调用不抛；后台 task 内的异常被吞（只日志）
    sn.fire_signal_email("x4", "s", "b")
    import asyncio
    await asyncio.sleep(0.05)  # 让后台 task 跑完，确认无未捕获异常上抛
