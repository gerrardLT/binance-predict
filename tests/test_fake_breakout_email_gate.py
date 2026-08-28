"""场景信号「结算推送闸」回归（2026-08-28 推送口径升级）。

改动前：主信号/S5 fire 时即发预告邮件（只查通道开关 + 日限），实盘门禁
拦下的信号照样推送，邮件胜率偏离实盘胜率。
改动后：触发不再发邮件；_settle_15m 结算置 SETTLED 后按闸链推送——
子开关(fake_breakout_email_enabled) → 通道开关(is_live_enabled) →
实盘成交闸(has_scene_filled_order：
trade_orders.scene_signal_id == 信号 id 且 FILLED)。影子信号无实盘订单
天然被 FILLED 闸过滤，被实盘门禁拦下的信号同样不推。

此处用 stub session 驱动 _settle_15m，断言三态：
通道关 → 不发；通道开但未成交 → 不发（闸按信号 id 查过一次）；
通道开 + FILLED → 发一次。另验 _send_signal_email_bg 的 email_sent 幂等守卫。

不触真实 DB/SMTP/klines。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from binance_predict.services import fake_breakout_detector as fbd_mod
from binance_predict.services import signal_notify as sn
from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector

CHANNEL = "scene_bull_exhaust"


class _Session:
    """最小 async session stub：execute 返回预置待结算行，get 按主键取同一对象。"""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, _stmt):  # 仅 _settle_15m 的 due 查询会走到这里
        return SimpleNamespace(
            scalars=lambda: SimpleNamespace(all=lambda: list(self._rows)))

    async def get(self, _model, pk):
        for r in self._rows:
            if r.id == pk:
                return r
        return None

    async def commit(self) -> None:
        pass


def _settled_candidate(sig_id: int = 1) -> SimpleNamespace:
    """待结算 PENDING 行（周期坐标/锚点齐全，bull_exhaust 正式信号）。"""
    return SimpleNamespace(
        id=sig_id, status="PENDING", pattern="S1", pattern_type="bull_exhaust",
        market_start_15m=1_700_000_900_000, market_end_15m=1_700_001_800_000,
        cycle_open_price_15m=43210.0, settle_deadline=1_700_001_800_000,
        settle_btc_price=None, settle_outcome=None, down_price_15m=0.55,
        version=None, signal_time=1_700_000_000_000,
    )


def _detector(monkeypatch, rows: list) -> FakeBreakoutDetector:
    @asynccontextmanager
    async def _factory():
        yield _Session(rows)

    monkeypatch.setattr(fbd_mod, "async_session_factory", _factory)
    det = FakeBreakoutDetector(
        collector=SimpleNamespace(store=SimpleNamespace(mid_price=None)),
        pm_15m_latest={},
    )
    det._update_pattern_stats = AsyncMock()  # 统计回填不触真实 DB
    return det


async def _settle(det: FakeBreakoutDetector, monkeypatch, filled: bool) -> AsyncMock:
    """驱动 _settle_15m：klines 固定收盘 < 锚点（结算 DOWN），返回邮件 stub。"""

    async def _kopen(_interval, _start_ms):
        return 43190.0  # < 锚点 43210 → settle_outcome=DOWN，置 SETTLED

    monkeypatch.setattr(det, "_klines_open", _kopen)
    monkeypatch.setattr(fbd_mod, "has_scene_filled_order",
                        AsyncMock(return_value=filled))
    mail = AsyncMock()
    det._send_signal_email_bg = mail  # type: ignore[method-assign]
    await det._settle_15m(1_700_001_800_000 + 60_000)
    import asyncio
    await asyncio.sleep(0)  # 让可能的 create_task 排程执行
    return mail


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """子开关默认开（本套聚焦通道闸与 FILLED 闸；子开关有独立关闭用例）。"""
    from binance_predict.config.settings import settings
    monkeypatch.setattr(settings, "fake_breakout_email_enabled", True)
    yield
    sn.reset_daily_count()


@pytest.mark.asyncio
async def test_settle_email_blocked_when_subswitch_off(monkeypatch) -> None:
    """子开关 fake_breakout_email_enabled=False → 结算落库但不发邮件。

    场景邮件直连 send_plain_email（不走 push_signal_email 全局日限），
    此开关即该路径的运维 kill switch（CodeReview Medium#1：开关须保持生效）。
    """
    from binance_predict.config.settings import settings
    monkeypatch.setattr(settings, "fake_breakout_email_enabled", False)
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: ch == CHANNEL)
    det = _detector(monkeypatch, [_settled_candidate()])
    mail = await _settle(det, monkeypatch, filled=True)
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_email_blocked_when_channel_off(monkeypatch) -> None:
    """通道关（resolver 明确 False）→ 结算落库但不创建邮件任务。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: False)
    det = _detector(monkeypatch, [_settled_candidate()])
    mail = await _settle(det, monkeypatch, filled=True)
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_email_blocked_when_resolver_missing(monkeypatch) -> None:
    """resolver 未注入（装配失败 fail-safe）→ 结算落库但不创建邮件任务。"""
    det = _detector(monkeypatch, [_settled_candidate()])
    mail = await _settle(det, monkeypatch, filled=True)
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_email_blocked_when_not_filled(monkeypatch) -> None:
    """通道开但无 FILLED 订单（被实盘门禁拦下/影子）→ 闸查过一次，不推。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: ch == CHANNEL)
    det = _detector(monkeypatch, [_settled_candidate()])
    gate = AsyncMock(return_value=False)
    monkeypatch.setattr(fbd_mod, "has_scene_filled_order", gate)
    mail = AsyncMock()
    det._send_signal_email_bg = mail  # type: ignore[method-assign]
    async def _kopen(_interval, _start_ms):
        return 43190.0
    monkeypatch.setattr(det, "_klines_open", _kopen)
    await det._settle_15m(1_700_001_800_000 + 60_000)
    gate.assert_awaited_once_with(1)   # FILLED 闸按信号 id 精确查询
    mail.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_email_sent_once_when_filled(monkeypatch) -> None:
    """通道开 + FILLED → 结算后创建一次邮件任务（推送点=结算回读后）。"""
    monkeypatch.setattr(sn, "_live_resolver", lambda ch: ch == CHANNEL)
    det = _detector(monkeypatch, [_settled_candidate()])
    mail = await _settle(det, monkeypatch, filled=True)
    mail.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_email_bg_skips_already_sent(monkeypatch) -> None:
    """email_sent=True 幂等守卫：已发过的信号不再重发。"""
    row = _settled_candidate()
    row.email_sent = True

    @asynccontextmanager
    async def _factory():
        yield _Session([row])

    monkeypatch.setattr(fbd_mod, "async_session_factory", _factory)
    det = FakeBreakoutDetector(
        collector=SimpleNamespace(store=SimpleNamespace(mid_price=None)),
        pm_15m_latest={},
    )
    send = AsyncMock()
    monkeypatch.setattr(det, "_send_signal_email", send)
    await det._send_signal_email_bg(1)
    send.assert_not_awaited()
