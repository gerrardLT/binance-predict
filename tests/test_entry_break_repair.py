"""冷启动入场价断链自愈（2026-08-27 误结算事故）回归测试。

背景：部署重启落在 5m 窗口中间时，旧代码用重启时刻 mid_price 充当窗口
entry_price → 归档 outcome 翻转 → TradeSettler 误结算（实锤：20:10~20:15
窗 entry 79512≠真开盘 79444，DOWN 单误判赢 +30.33，币安官方结果 UP）。

污染指纹（断链）：正常窗口切换时本窗 entry == 前窗 exit（同一边界快照），
污染窗 entry 是窗中价 → 与前窗 exit 显著偏离。

覆盖（不触网络/真实 DB：session 用 _Db 桩，仿 test_trade_settler 模式）：
- detect_entry_break：断链/正常/无前窗/无效价四分支
- correct_entry_break：归档钩子——未断链不动、断链 kline 回读、回读失败沿用
- resettle_window_orders：改判口径同 TradeSettler（输=-amount）、
  已一致行幂等跳过、NOISE、金额缺失跳过
- heal_entry_break_windows：启动自愈端到端——修 entry/actual_return/outcome
  + 订单重结算；幂等（无断链 0 命中）；kline 失败不动数据
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Select, Update

import binance_predict.services.archive_contamination_repair as acr
from binance_predict.db.models import SentimentWindow, TradeOrderModel

W = 5 * 60 * 1000
S1 = 1_787_832_600_000          # 事故窗起点（20:10 北京 / 12:10 UTC）
S0 = S1 - W                     # 前一窗起点

# 事故现场数值（生产实锤）：前窗 exit 在 20:05~20:10 边界采样，与真实
# 开盘几乎同价；冷启动污染窗 entry 是 20:14 重启时刻的窗中价 → 偏差 $68
BAD_ENTRY = 79_511.955          # 冷启动污染的窗中价
PREV_EXIT = 79_444.1            # 前窗 exit（边界价≈真实开盘，正确）
EXIT = 79_498.005               # 污染窗 exit（切换时刻真实价，正确）
KOPEN = 79_444.0                # klines 精确回读的真实开盘


# ------------------------------------------------------------------
# 公共桩
# ------------------------------------------------------------------

def _order(**over) -> SimpleNamespace:
    """已结算订单行替身（订单 73 实锤：DOWN 误判赢 +30.33，3 USDT @ 0.09）。"""
    base = dict(
        id=73, status="FILLED", settled_at=datetime.now(timezone.utc),
        window_start=S1, direction="DOWN", amount_in=str(3 * 10 ** 18),
        quote_json={"averagePrice": 0.09}, settle_outcome="DOWN",
        win=True, pnl=30.33, settle_price=EXIT,
        market_period="5m", scene_signal_id=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _window_row(**over) -> SimpleNamespace:
    """SentimentWindow 归档行替身。"""
    base = dict(
        id=1, start_time=S1, end_time=S1 + W,
        entry_price=BAD_ENTRY, exit_price=EXIT, outcome="DOWN",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Db:
    """按 stmt 路由：窗口全列查询 / 前窗 exit 列查询 / 订单查询 / UPDATE 捕获。"""

    def __init__(self, windows=None, prev_exit=None, orders=None) -> None:
        self.updates: list[Update] = []
        self._windows = windows or []
        self._prev_exit = prev_exit
        self._orders = orders or []

        wins_res = MagicMock()
        wins_res.scalars.return_value.all.return_value = self._windows
        prev_res = MagicMock()
        prev_res.scalar_one_or_none.return_value = self._prev_exit
        orders_res = MagicMock()
        orders_res.scalars.return_value.all.return_value = self._orders

        async def _execute(stmt):
            if isinstance(stmt, Update):
                self.updates.append(stmt)
                res = MagicMock()
                res.rowcount = 1
                return res
            assert isinstance(stmt, Select), f"意外语句类型: {stmt!r}"
            desc = stmt.column_descriptions[0]
            if desc["entity"] is TradeOrderModel:
                return orders_res
            if desc["entity"] is SentimentWindow:
                # 全实体查询（启动扫描） vs exit_price 列查询（断链判定）
                return wins_res if desc["expr"] is SentimentWindow else prev_res
            raise AssertionError(f"未知查询实体: {desc['entity']}")

        self.execute = AsyncMock(side_effect=_execute)
        self.commit = AsyncMock()


def _stub_factory(monkeypatch, db: _Db) -> None:
    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(acr, "async_session_factory", _factory)


def _collector(kopen: float | None = KOPEN) -> SimpleNamespace:
    return SimpleNamespace(fetch_kline_open=AsyncMock(return_value=kopen))


def _params(stmt) -> dict:
    return stmt.compile().params


# ------------------------------------------------------------------
# detect_entry_break：断链判定四分支
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detect_break_deviated_entry() -> None:
    """事故口径：entry 79512 vs 前窗 exit 79444 → 偏差 ~$68 ≫ 容差 → 断链。"""
    db = _Db(prev_exit=PREV_EXIT)
    assert await acr.detect_entry_break(db, S1, BAD_ENTRY) is True


@pytest.mark.asyncio
async def test_detect_break_normal_chain() -> None:
    """正常链：entry == 前窗 exit（同一边界快照）→ 不断链。"""
    db = _Db(prev_exit=PREV_EXIT)
    assert await acr.detect_entry_break(db, S1, PREV_EXIT) is False
    # 边界处 mid 与 kline 天然偏差 ≤$8（≈1e-4）在容差 2e-4 内 → 不断链
    db2 = _Db(prev_exit=PREV_EXIT)
    assert await acr.detect_entry_break(db2, S1, PREV_EXIT + 5.0) is False


@pytest.mark.asyncio
async def test_detect_break_no_prev_window() -> None:
    """库首窗（无前窗）→ 无从判定，不动。"""
    db = _Db(prev_exit=None)
    assert await acr.detect_entry_break(db, S1, BAD_ENTRY) is False


@pytest.mark.asyncio
async def test_detect_break_invalid_entry() -> None:
    db = _Db(prev_exit=PREV_EXIT)
    assert await acr.detect_entry_break(db, S1, None) is False
    assert await acr.detect_entry_break(db, S1, 0.0) is False


# ------------------------------------------------------------------
# correct_entry_break：归档钩子
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correct_hook_no_break_noop() -> None:
    """未断链 → 返回 None 且不打 kline（零成本）。"""
    db = _Db(prev_exit=PREV_EXIT)
    col = _collector()
    assert await acr.correct_entry_break(db, S1, PREV_EXIT, col) is None
    col.fetch_kline_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_correct_hook_replaces_with_kline_open() -> None:
    """断链 + kline 回读成功 → 返回真实开盘价。"""
    db = _Db(prev_exit=PREV_EXIT)
    col = _collector(KOPEN)
    got = await acr.correct_entry_break(db, S1, BAD_ENTRY, col)
    assert got == KOPEN
    col.fetch_kline_open.assert_awaited_once_with("5m", S1)


@pytest.mark.asyncio
async def test_correct_hook_kline_fail_keeps_original() -> None:
    """断链但 kline 回读失败（返回 0.0）→ None，调用方沿用原值。"""
    db = _Db(prev_exit=PREV_EXIT)
    col = _collector(0.0)
    assert await acr.correct_entry_break(db, S1, BAD_ENTRY, col) is None


# ------------------------------------------------------------------
# resettle_window_orders：重结算口径
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resettle_flips_false_win_to_loss() -> None:
    """事故口径：误判赢（+30.33）的 DOWN 单在真实 outcome=UP 下改判
    输 → win=False、pnl=-amount=-3.0（口径同 TradeSettler）。"""
    row = _order()
    db = _Db(orders=[row])
    changed = await acr.resettle_window_orders(db, S1, EXIT, "UP")

    assert changed == 1
    assert row.settle_outcome == "UP"
    assert row.win is False
    assert row.pnl == pytest.approx(-3.0)
    assert row.settle_price == pytest.approx(EXIT)


@pytest.mark.asyncio
async def test_resettle_keeps_correct_win() -> None:
    """方向与新 outcome 一致（原本判对的单）→ 幂等跳过不改。"""
    row = _order(direction="UP", settle_outcome="UP", win=True,
                 pnl=100.0, amount_in=str(10 ** 18),
                 quote_json={"averagePrice": 0.5})
    db = _Db(orders=[row])
    changed = await acr.resettle_window_orders(db, S1, EXIT, "UP")

    assert changed == 0
    assert row.pnl == pytest.approx(100.0)  # 原值不动


@pytest.mark.asyncio
async def test_resettle_win_recomputes_pnl() -> None:
    """改判后仍为赢（outcome 翻但方向一致侧）→ pnl 按 amount/avg-amount 重算。"""
    row = _order(direction="UP", settle_outcome="DOWN", win=False,
                 pnl=-3.0, amount_in=str(3 * 10 ** 18),
                 quote_json={"averagePrice": 0.09})
    db = _Db(orders=[row])
    changed = await acr.resettle_window_orders(db, S1, EXIT, "UP")

    assert changed == 1
    assert row.win is True
    assert row.pnl == pytest.approx(3 / 0.09 - 3)


@pytest.mark.asyncio
async def test_resettle_noise() -> None:
    """NOISE：win=None、pnl=0.0（口径同 TradeSettler）。"""
    row = _order()
    db = _Db(orders=[row])
    changed = await acr.resettle_window_orders(db, S1, EXIT, "NOISE")

    assert changed == 1
    assert row.win is None
    assert row.pnl == 0.0


@pytest.mark.asyncio
async def test_resettle_missing_amount_skips() -> None:
    """金额/均价缺失无法重算 → 保持原值，不计入改判。"""
    row = _order(amount_in=None)
    db = _Db(orders=[row])
    changed = await acr.resettle_window_orders(db, S1, EXIT, "UP")

    assert changed == 0
    assert row.win is True and row.pnl == pytest.approx(30.33)


@pytest.mark.asyncio
async def test_resettle_invalid_outcome_noop() -> None:
    db = _Db(orders=[_order()])
    assert await acr.resettle_window_orders(db, S1, EXIT, None) == 0
    assert await acr.resettle_window_orders(db, S1, EXIT, "EXPIRED") == 0


# ------------------------------------------------------------------
# heal_entry_break_windows：启动自愈端到端
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heal_repairs_accident_window(monkeypatch) -> None:
    """事故端到端：污染窗 entry 79512/结果 DOWN + 假赢单 →
    修复后 entry=79444、actual_return>0、outcome=UP，订单改判输 -3.0。"""
    win = _window_row()
    db = _Db(windows=[win], prev_exit=PREV_EXIT, orders=[_order()])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_entry_break_windows(_collector(KOPEN))

    assert stats == {"scanned": 1, "repaired": 1, "orders_resettled": 1}
    assert len(db.updates) == 1
    p = _params(db.updates[0])
    assert p["entry_price"] == KOPEN
    assert p["actual_return"] == pytest.approx(EXIT / KOPEN - 1)
    assert p["outcome"] == "UP"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_heal_no_break_is_noop(monkeypatch) -> None:
    """链完整（正常归档）→ 0 命中、不写 UPDATE、不打 kline（幂等 no-op）。"""
    win = _window_row(entry_price=PREV_EXIT, outcome="NOISE")
    db = _Db(windows=[win], prev_exit=PREV_EXIT)
    _stub_factory(monkeypatch, db)
    col = _collector()

    stats = await acr.heal_entry_break_windows(col)

    assert stats == {"scanned": 1, "repaired": 0, "orders_resettled": 0}
    assert db.updates == []
    col.fetch_kline_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_heal_kline_fail_defers(monkeypatch) -> None:
    """断链但 kline 回读失败 → 不修（下次启动重试），不误改数据。"""
    win = _window_row()
    db = _Db(windows=[win], prev_exit=PREV_EXIT, orders=[_order()])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_entry_break_windows(_collector(0.0))

    assert stats == {"scanned": 1, "repaired": 0, "orders_resettled": 0}
    assert db.updates == []
