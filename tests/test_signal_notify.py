"""signal_notify 信号推送公共通道的单元测试（2026-08-25 全信号推送落地）。

覆盖：新鲜度闸（历史重放静默）、实盘开火闸（只推已开火通道，未注入
fail-safe 不推）、总开关、全局日限防轰炸、跨 UTC 日翻转、fire-and-forget
入口异常隔离。不触发真实 SMTP（send_plain_email 被桩掉）。
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


def test_fmt_bjt_converts_to_beijing_time() -> None:
    """邮件时间口径：ms → 北京时间（UTC+8），跨日期边界正确。"""
    # 2026-08-24 22:30 UTC = 2026-08-25 06:30 北京时间（跨日）
    ms = 1787610600000
    assert sn.fmt_bjt(ms) == "08-25 06:30"
    assert sn.fmt_bjt(ms, with_date=False) == "06:30"


def test_daily_cap_default_is_800() -> None:
    """日限默认值防回退：2026-08-25 用户拍板 80→800，误改回 80 应被 CI 拦下。"""
    from binance_predict.config.settings import Settings
    assert Settings().signal_push_max_daily_emails == 800


# ------------------------------------------------------------------
# 实盘开火闸：只推已开实盘开火通道的信号
# ------------------------------------------------------------------

def test_is_live_enabled_false_without_resolver(monkeypatch) -> None:
    """resolver 未注入（MultiLiveTrader 装配失败/测试环境）→ 一律不推。"""
    monkeypatch.setattr(sn, "_live_resolver", None)
    assert sn.is_live_enabled("x4_v1") is False
    assert sn.is_live_enabled("quote_momentum_v2") is False


def test_is_live_enabled_follows_resolver(monkeypatch) -> None:
    """注入后按运行时开关返回；未知通道（如 v3a/v3b）恒 False。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: ch == "x4_v1")
    assert sn.is_live_enabled("x4_v1") is True
    assert sn.is_live_enabled("x4_v2") is False
    assert sn.is_live_enabled("quote_contrarian_v3a") is False


def test_is_live_enabled_swallows_resolver_errors(monkeypatch) -> None:
    """resolver 抛异常 → 宁少勿多返回 False，不得把检测循环拖死。"""
    def _boom(ch):
        raise RuntimeError("configs not ready")

    monkeypatch.setattr(sn, "_live_resolver", _boom)
    assert sn.is_live_enabled("x4_v1") is False


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
