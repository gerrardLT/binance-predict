"""入场报价快照单元测试（2026-08-17）：次周期开盘后延迟抓取 + 加仓触发监测。

不触网络/真实 DB：async_session_factory / clock_sync / asyncio.sleep 全部替身。
核心分支：
- 市场切换守卫（start_date 匹配才落，防旧市场残值 0.01~0.99 污染入场价）
- 未切换重试 → 切换后落；超时放弃 → 保持 NULL 不阻塞
- 场景①加仓触发（mid ≥ 开盘×1.001）落 add 列；未触发保持 NULL
- fire 成功后调度快照后台任务
- 15m 通道共用处理（main._handle_15m_quote）：切换检测 / 幂等 / persist 分支 / 防御
"""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from binance_predict.services import fake_breakout_detector as fbd
from binance_predict.services.fake_breakout_detector import FakeBreakoutDetector

NEXT_START = 1_000_000_000_000
NEXT_END = NEXT_START + 900_000


class _Row:
    """信号行替身：未回填列 getattr 为 None（MagicMock 属性默认非 None，故用普通类）。"""


class _FakeStore:
    def __init__(self, mid: float) -> None:
        self.mid_price = mid

    def fresh_mid_price(self, max_age_s: float | None = None) -> float:
        # 替身模拟喂价永远新鲜（R3 新鲜度闸语义）
        return self.mid_price or 0.0


class _FakeCollector:
    def __init__(self, mid: float) -> None:
        self.store = _FakeStore(mid)


def _make_detector(monkeypatch, pm: dict, mid: float, row: _Row,
                   now_start_ms: int = NEXT_START + 5_000, step_ms: int = 10_000):
    """替身注入：DB factory / 服务器时钟（步进）/ asyncio.sleep（立即返回）。"""
    det = FakeBreakoutDetector(collector=_FakeCollector(mid), pm_15m_latest=pm)
    det._running = True

    session = AsyncMock()
    session.get = AsyncMock(return_value=row)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()  # 同步调用（session.add(signal) 不 await）
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(fbd, "async_session_factory", factory)

    clock = MagicMock()
    clock.now_ms = MagicMock(side_effect=itertools.count(now_start_ms, step_ms))
    monkeypatch.setattr(fbd, "clock_sync", clock)

    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    return det, session


@pytest.mark.asyncio
async def test_entry_snapshot_scene2(monkeypatch) -> None:
    """市场已切换 → 开盘后首试即落 entry 三列；场景②无加仓监测，任务即退。"""
    pm = {"start_date": NEXT_START, "end_date": NEXT_END,
          "down_price": 0.61, "up_price": 0.38, "updated_ts": NEXT_START + 25_000}
    row = _Row()
    det, session = _make_detector(monkeypatch, pm, mid=64_000.0, row=row)

    await det._capture_entry_quote(7, NEXT_START, NEXT_END, "bear_exhaust", None)

    assert row.entry_down_price_15m == 0.61
    assert row.entry_up_price_15m == 0.38
    assert row.entry_quote_ts_15m == NEXT_START + 25_000
    session.commit.await_count == 1


@pytest.mark.asyncio
async def test_entry_snapshot_retry_until_switch(monkeypatch) -> None:
    """tracker 未切换（旧市场残值）→ 不落；重试一轮后切换 → 落。"""
    pm = {"start_date": NEXT_START - 900_000, "down_price": 0.99, "up_price": 0.01,
          "updated_ts": NEXT_START - 60_000}
    row = _Row()

    # _sleep_until 阶段消耗两次 sleep；主循环第一次 sleep（第 3 次）时市场切换，
    # 验证「主循环首给旧市场不落 → 重试一轮后落」的重试路径
    sleep_calls = itertools.count()

    async def _switch_on_sleep(_s: float) -> None:
        if next(sleep_calls) >= 2:
            pm["start_date"] = NEXT_START
            pm["down_price"] = 0.55
            pm["up_price"] = 0.44
            pm["updated_ts"] = NEXT_START + 30_000

    det, _ = _make_detector(monkeypatch, pm, mid=64_000.0, row=row)
    monkeypatch.setattr(asyncio, "sleep", _switch_on_sleep)

    await det._capture_entry_quote(7, NEXT_START, NEXT_END, "bear_exhaust", None)

    # 第一轮未匹配不落（0.99 残值不得污染）；第二轮切换后落新价
    assert row.entry_down_price_15m == 0.55
    assert row.entry_up_price_15m == 0.44


@pytest.mark.asyncio
async def test_entry_snapshot_timeout_abandoned(monkeypatch) -> None:
    """市场永不切换 → +90s 截止放弃，entry 保持 NULL（不阻塞结算）。"""
    pm = {"start_date": NEXT_START - 900_000, "down_price": 0.99, "up_price": 0.01}
    row = _Row()
    det, session = _make_detector(monkeypatch, pm, mid=64_000.0, row=row)

    await det._capture_entry_quote(7, NEXT_START, NEXT_END, "bear_exhaust", None)

    assert getattr(row, "entry_down_price_15m", None) is None
    assert getattr(row, "entry_quote_ts_15m", None) is None
    session.get.assert_not_awaited()  # 放弃路径不触 DB


@pytest.mark.asyncio
async def test_add_trigger_snapshot_scene1(monkeypatch) -> None:
    """场景①：mid ≥ 开盘×1.001 → 抓当时报价落 add 三列（@0.27 假设的实盘对照）。"""
    pm = {"start_date": NEXT_START, "down_price": 0.31, "up_price": 0.68,
          "updated_ts": NEXT_START + 25_000}
    row = _Row()
    # 开盘 100000 → 触发价 100100；mid 100150 ≥ 触发价
    det, _ = _make_detector(monkeypatch, pm, mid=100_150.0, row=row)

    await det._capture_entry_quote(7, NEXT_START, NEXT_END, "bull_exhaust", 100_000.0)

    assert row.add_down_price_15m == 0.31
    assert row.add_up_price_15m == 0.68
    assert row.add_trigger_ts_15m is not None and row.add_trigger_ts_15m > NEXT_START
    assert row.entry_down_price_15m == 0.31  # 入场快照同轮也落


@pytest.mark.asyncio
async def test_add_not_triggered_keeps_null(monkeypatch) -> None:
    """场景①未反弹（mid 恒低于触发价）→ 周期结束 add 保持 NULL（=未触发事实）。"""
    pm = {"start_date": NEXT_START, "down_price": 0.50, "up_price": 0.49,
          "updated_ts": NEXT_START + 25_000}
    row = _Row()
    det, _ = _make_detector(monkeypatch, pm, mid=99_900.0, row=row)  # < 100100

    await det._capture_entry_quote(7, NEXT_START, NEXT_END, "bull_exhaust", 100_000.0)

    assert row.entry_down_price_15m == 0.50  # 入场快照正常落
    assert getattr(row, "add_down_price_15m", None) is None
    assert getattr(row, "add_trigger_ts_15m", None) is None


@pytest.mark.asyncio
async def test_fire_schedules_capture_task(monkeypatch) -> None:
    """fire 落表成功后调度 fbs_entry_* 后台任务（影子对照同口径）。"""
    pm = {"start_date": NEXT_START, "down_price": 0.5, "up_price": 0.5}
    det, session = _make_detector(monkeypatch, pm, mid=64_000.0, row=_Row())
    monkeypatch.setattr(det, "_send_signal_email_bg", AsyncMock())

    created: list[tuple] = []

    def fake_create_task(coro, name=None):
        created.append((name, coro))
        coro.close()
        return MagicMock()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    rec = {"level": "4h", "cycle_id": 111, "broken_level": 63_000.0,
           "break_price": 63_100.0, "break_time": 0}
    sig_k = {"open": 63_000.0, "high": 63_500.0, "low": 62_900.0,
             "close": 63_400.0, "volume": 120.0}
    await det._fire_confirmed_signal(
        "high", rec, sig_k, close_pos=0.92, vol_ratio=None,
        cur_cycle=NEXT_START // 900_000, now_ms=NEXT_START + 2_000,
    )

    assert any(name and name.startswith("fbs_entry_") for name, _ in created)


# ============================================================
# 15m 通道共用处理（main._handle_15m_quote）：tracker 与边界加速协程共用
# ============================================================


def _quote(end: int, down: float = 0.6, up: float = 0.38) -> SimpleNamespace:
    return SimpleNamespace(
        down_price=down, up_price=up, start_date=end - 900_000, end_date=end,
        up_chance=0.4, down_chance=0.6, participants=10, trade_volume=99.0,
    )


async def _run_handle(monkeypatch, quote, ts_ms: int, persist: bool) -> tuple:
    """注入替身后跑 _handle_15m_quote，返回 (main模块, session)。"""
    import binance_predict.main as m

    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(m, "async_session_factory", factory)

    col = MagicMock()
    col.store.mid_price = 64_000.0
    col.store.fresh_mid_price = MagicMock(return_value=64_000.0)  # R3 闸：模拟喂价新鲜
    col.fetch_mid_price = AsyncMock(return_value=64_001.0)
    col.fetch_kline_open = AsyncMock(return_value=63_999.0)
    monkeypatch.setattr(m, "collector", col)

    m._pm_15m_latest.clear()
    m._pm_15m_latest.update({"down_price": None, "up_price": None, "start_date": None,
                             "end_date": None, "updated_ts": None,
                             "cycle_open_price": None, "cycle_open_end": None})
    await m._handle_15m_quote(quote, ts_ms, persist=persist)
    return m, session


@pytest.mark.asyncio
async def test_handle_15m_first_quote_cold_start(monkeypatch) -> None:
    """冷启动（缓存空 end_date）：klines 回读开盘价 + 全量刷缓存 + 落库一次。"""
    end = NEXT_START + 900_000
    m, session = await _run_handle(monkeypatch, _quote(end), ts_ms=NEXT_START + 20_000, persist=True)

    assert m._pm_15m_latest["cycle_open_price"] == 63_999.0  # klines 回读分支
    assert m._pm_15m_latest["end_date"] == end
    assert m._pm_15m_latest["down_price"] == 0.6
    assert m._pm_15m_latest["updated_ts"] == NEXT_START + 20_000
    session.add.assert_called_once()  # persist=True 落库


@pytest.mark.asyncio
async def test_handle_15m_switch_then_idempotent(monkeypatch) -> None:
    """首次冷启动 klines 回读；同 end_date 再调不重复快照（幂等）；
    切换到新 end_date 走现价快照分支；persist=False 不落库。"""
    end1, end2 = NEXT_START + 900_000, NEXT_START + 1_800_000
    m, session = await _run_handle(monkeypatch, _quote(end1), ts_ms=1, persist=True)
    assert m._pm_15m_latest["cycle_open_price"] == 63_999.0  # 冷启动：klines 回读分支

    col = m.collector
    await m._handle_15m_quote(_quote(end1, down=0.7), ts_ms=2, persist=True)
    assert col.fetch_mid_price.await_count == 0  # 同 end_date 不再触发切换检测
    assert m._pm_15m_latest["down_price"] == 0.7  # 报价照常刷新
    assert m._pm_15m_latest["cycle_open_price"] == 63_999.0  # 开盘价未被覆盖

    await m._handle_15m_quote(_quote(end2), ts_ms=3, persist=False)
    assert m._pm_15m_latest["end_date"] == end2
    assert m._pm_15m_latest["cycle_open_end"] == end2
    assert m._pm_15m_latest["cycle_open_price"] == 64_000.0  # 正常切换：现价快照分支
    assert session.add.call_count == 2  # persist=False 不落库（前两次 True 各一条）


@pytest.mark.asyncio
async def test_handle_15m_defensive(monkeypatch) -> None:
    """None 报价 / 缺 end_date / 缺 down_price → 直接返回，缓存与 DB 均不动。"""
    m, session = await _run_handle(monkeypatch, None, ts_ms=1, persist=True)
    assert m._pm_15m_latest["end_date"] is None
    session.add.assert_not_called()

    bad = _quote(NEXT_START)
    bad.end_date = None
    await m._handle_15m_quote(bad, ts_ms=1, persist=True)
    bad2 = _quote(NEXT_START)
    bad2.down_price = None
    await m._handle_15m_quote(bad2, ts_ms=1, persist=True)
    session.add.assert_not_called()
