"""TradeSettler 单元测试（P0-2）：结算规则分支 + 幂等守卫。

不触网络/真实 DB：async_session_factory 用 @asynccontextmanager 桩
（仿 test_sync_binance_backfills_pending_rows 模式）。db.execute 按
stmt 类型路由：Select(TradeOrderModel) → 待结算行；Select(SentimentWindow)
→ 窗口桩；Update → 捕获语句（compile().params 断言结算字段）+
rowcount 控制幂等分支。

核心分支（对照 trade_settler.py 结算规则）：
- 赢/输：pnl 数值断言（赢=amount/avg_price-amount，输=-amount）
- NOISE：win=None + pnl=0.0 + settled_at 写入（终止重扫）
- 窗口未归档且未超 24h：跳过下轮重试（无 UPDATE）
- 窗口缺失超 24h：EXPIRED 兜底
- UPDATE rowcount=0（并发已结算）：幂等守卫，不计入 settled
- direction NULL（旧数据）：跳过，不查窗口不写 UPDATE
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Select, Update

import binance_predict.services.trade_settler as ts_mod
from binance_predict.db.models import SentimentWindow, TradeOrderModel
from binance_predict.services.trade_settler import TradeSettler

WS = 1_787_400_000_000  # 任意 5m 窗口起点 ms


def _row(**over) -> SimpleNamespace:
    """TradeOrderModel 待结算行替身（FILLED + 未结算 + 超 7min 延迟）。"""
    base = dict(
        id=11, status="FILLED", settled_at=None, window_start=WS,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        direction="DOWN", amount_in=str(10 ** 18),  # 1 USDT
        quote_json={"averagePrice": 0.5},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _window(outcome: str, exit_price: float = 43250.0) -> SimpleNamespace:
    """SentimentWindow 归档行替身。"""
    return SimpleNamespace(start_time=WS, outcome=outcome, exit_price=exit_price)


class _Db:
    """按 stmt 类型路由 execute：订单查询 / 窗口查询 / UPDATE 捕获。"""

    def __init__(self, rows, window, update_rowcount: int = 1) -> None:
        self.updates: list[Update] = []
        self._update_rowcount = update_rowcount

        orders_res = MagicMock()
        orders_res.scalars.return_value.all.return_value = rows
        window_res = MagicMock()
        window_res.scalar_one_or_none.return_value = window

        async def _execute(stmt):
            if isinstance(stmt, Update):
                self.updates.append(stmt)
                res = MagicMock()
                res.rowcount = self._update_rowcount
                return res
            assert isinstance(stmt, Select), f"意外语句类型: {stmt!r}"
            entity = stmt.column_descriptions[0]["entity"]
            if entity is TradeOrderModel:
                return orders_res
            if entity is SentimentWindow:
                return window_res
            raise AssertionError(f"未知查询实体: {entity}")

        self.execute = AsyncMock(side_effect=_execute)
        self.commit = AsyncMock()


def _stub_db(monkeypatch, db: _Db) -> None:
    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(ts_mod, "async_session_factory", _factory)


def _params(update_stmt) -> dict:
    return update_stmt.compile().params


# ------------------------------------------------------------------
# 赢 / 输（pnl 数值断言）
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_settle_win(monkeypatch) -> None:
    """赢：direction==outcome → win=True、pnl=1/0.5-1=1.0、settle_price=exit。"""
    db = _Db([_row(direction="DOWN")], _window("DOWN"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    assert len(db.updates) == 1
    p = _params(db.updates[0])
    assert p["settle_outcome"] == "DOWN"
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(1.0)
    assert p["settle_price"] == pytest.approx(43250.0)
    assert isinstance(p["settled_at"], datetime)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_settle_lose(monkeypatch) -> None:
    """输：direction!=outcome → win=False、pnl=-amount=-1.0（无需均价）。"""
    db = _Db([_row(direction="DOWN", quote_json=None)], _window("UP"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "UP"
    assert p["win"] is False
    assert p["pnl"] == pytest.approx(-1.0)


# ------------------------------------------------------------------
# NOISE / 窗口缺失
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_settle_noise(monkeypatch) -> None:
    """NOISE：win=None、pnl=0.0、settled_at 写入（锚点终止重扫）。"""
    db = _Db([_row()], _window("NOISE"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "NOISE"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] == pytest.approx(43250.0)
    assert p["settled_at"] is not None


@pytest.mark.asyncio
async def test_window_missing_retry(monkeypatch) -> None:
    """窗口未归档且未超 24h → 不写 UPDATE，下轮重试。"""
    db = _Db([_row(created_at=datetime.now(timezone.utc) - timedelta(hours=2))],
             None)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert db.updates == []


@pytest.mark.asyncio
async def test_window_missing_expired(monkeypatch) -> None:
    """窗口缺失超 24h → EXPIRED 兜底（win=None/pnl=0/settle_price=None）。"""
    old = datetime.now(timezone.utc) - timedelta(hours=25)
    db = _Db([_row(created_at=old)], None)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "EXPIRED"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] is None


@pytest.mark.asyncio
async def test_naive_created_at_treated_as_utc(monkeypatch) -> None:
    """naive created_at（测试桩/驱动差异）按 UTC 解释：未超 24h → 重试。"""
    naive_recent = (datetime.now(timezone.utc) - timedelta(hours=2)
                    ).replace(tzinfo=None)
    db = _Db([_row(created_at=naive_recent)], None)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert db.updates == []


# ------------------------------------------------------------------
# 幂等 / 旧数据
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idempotent_guard(monkeypatch) -> None:
    """UPDATE rowcount=0（已被并发结算）→ 幂等守卫生效，不计入 settled。"""
    db = _Db([_row()], _window("DOWN"), update_rowcount=0)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert len(db.updates) == 1  # 语句仍发出，只是守卫判定未生效


@pytest.mark.asyncio
async def test_direction_null_expired(monkeypatch) -> None:
    """旧数据 direction=None（被扫出即超 7min 延迟）→ 立即 EXPIRED 出清。

    生产实锤（2026-08-23）：4 笔 8/22 旧单 direction=NULL 无限重扫；
    字段无回填机制，等待毫无意义（首版等 24h 反而让旧单多挂一天）。
    扫描层已保证 created_at 超 7min，此处不查窗口直接出清。
    """
    db = _Db([_row(direction=None)], _window("DOWN"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "EXPIRED"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] is None
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_direction_null_created_at_missing(monkeypatch) -> None:
    """direction=None 且 created_at 缺失（防御）：同样出清不炸。"""
    db = _Db([_row(direction=None, created_at=None)], _window("DOWN"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1
    assert _params(db.updates[0])["settle_outcome"] == "EXPIRED"
