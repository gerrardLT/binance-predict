"""场景族邮件实盘开火闸回归测试（2026-08-25 评审发现 S5 路径漏挂闸）。

S5 确认信号（_fire_s5_signal）的邮件挂钩此前只查 fake_breakout_email_enabled
+ 日限，绕过实盘开火闸——通道全关时 S1 邮件被拦、S5 邮件照发。
此处用 stub session 驱动 _fire_s5_signal，断言邮件任务的对偶行为：
通道关（或未注入 resolver）→ 不创建邮件任务；通道开 → 创建。

不触真实 DB/SMTP/klines。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from binance_predict.config.settings import settings
from binance_predict.services import fake_breakout_detector as fbd_mod
from binance_predict.services import signal_notify as sn
from binance_predict.services.fake_breakout_detector import (
    FakeBreakoutDetector,
    FakeBreakoutSignal,
)

S5_CHANNEL = "scene_bull_exhaust_confirm"


class _Session:
    """最小 async session stub：get 返回预置父信号，add/commit/refresh 空操作。"""

    def __init__(self, parent: FakeBreakoutSignal) -> None:
        self._parent = parent

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def get(self, model, pk):
        return self._parent

    def add(self, obj) -> None:  # SQLAlchemy 的 add 是同步方法
        pass

    async def commit(self) -> None:
        pass

    async def refresh(self, obj) -> None:
        obj.id = 99  # create_task 命名与日志依赖 id


def _parent_signal() -> FakeBreakoutSignal:
    return FakeBreakoutSignal(
        level="4h", side="high", signal_time=1_700_000_000_000,
        resistance=43200.0, btc_price=43250.0, eps=0.001,
        market_start_15m=1_700_000_000_000, market_end_15m=1_700_000_900_000,
        cycle_open_price_15m=43210.0, settle_deadline=1_700_003_600_000,
        close_pos=0.9, vol_ratio=1.2, version="v1", pattern_type="bull_exhaust",
    )


def _detector(monkeypatch, parent: FakeBreakoutSignal) -> FakeBreakoutDetector:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _factory():
        yield _Session(parent)

    monkeypatch.setattr(fbd_mod, "async_session_factory", _factory)
    det = FakeBreakoutDetector(
        collector=SimpleNamespace(store=SimpleNamespace(mid_price=None)),
        pm_15m_latest={},
    )
    return det


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """邮件总开关开、日限放宽（本测试只验证实盘闸），resolver 默认未注入。"""
    monkeypatch.setattr(settings, "fake_breakout_email_enabled", True)
    monkeypatch.setattr(settings, "fake_breakout_max_daily_signals", 100)
    monkeypatch.setattr(sn, "_live_resolver", None)
    yield
    sn.reset_daily_count()


async def _fire(det: FakeBreakoutDetector) -> AsyncMock:
    """驱动 _fire_s5_signal，返回被桩掉的邮件任务入口，供断言调用次数。"""
    mail = AsyncMock()
    det._send_signal_email_bg = mail  # type: ignore[method-assign]
    await det._fire_s5_signal(
        parent_id=1, next_start=1_700_000_900_000, next_end=1_700_001_800_000,
        c5_close=43190.0, anchor=43210.0,
    )
    import asyncio
    await asyncio.sleep(0)  # 让可能的 create_task 排程执行
    return mail


@pytest.mark.asyncio
async def test_s5_email_blocked_when_channel_off(monkeypatch) -> None:
    """通道关（resolver 明确 False）→ 信号落表但不创建邮件任务。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: False)
    det = _detector(monkeypatch, _parent_signal())
    mail = await _fire(det)
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_s5_email_blocked_when_resolver_missing(monkeypatch) -> None:
    """resolver 未注入（装配失败 fail-safe）→ 不创建邮件任务。"""
    det = _detector(monkeypatch, _parent_signal())
    mail = await _fire(det)
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_s5_email_sent_when_channel_on(monkeypatch) -> None:
    """scene_bull_exhaust_confirm 通道开 → 创建一次邮件任务。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: ch == S5_CHANNEL)
    det = _detector(monkeypatch, _parent_signal())
    mail = await _fire(det)
    mail.assert_awaited_once_with(99)
