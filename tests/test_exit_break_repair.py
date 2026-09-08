"""出口价污染自愈（2026-09-08 误结算事故）回归测试。

背景：归档 exit_price 原取「tracker 检测到窗口切换那一刻」的 mid_price
（main.py:377），而检测发生在窗口边界之后的下一轮轮询 —— 晚到多少就污染
多少，晚到的快照实际落在下一窗口内。

生产实锤（2026-09-07 22:55~23:00 UTC 窗，订单 id=469/470）：
    entry 78942.005  == kline open  78942.01   ✓ 入口干净
    末次采样 22:59:46 BTC=78964.485（UP +2.85bp，市场 UP 报价已到 95.5%）
    exit    78938.385 ← 该价出现在 23:00 那根 1m K 线内（下一窗口）
    kline close 78952.19（UP +1.29bp）
    → outcome 由 UP 翻成 DOWN，两笔 firsthit DOWN 单虚记 +43.13 USDT
      （币安侧 DOWN token 归零：/api/prediction/redeemable claimable_count=0）

关键教训（本文件存在的理由）：事故窗偏差仅 |78938.385−78952.19|/78952.19
= 1.75bp，落在 400 窗实测分布 p95(1.13bp) 与 p99(2.79bp) 之间 —— 任何
「>1.75bp 纯容差」判定都会漏掉它。故 heal 必须以 **outcome 改判** 为主触发
条件，偏差容差只是次要条件。

覆盖（不触网络/真实 DB：session 用 _Db 桩，仿 test_entry_break_repair 模式）：
- outcome_of：UP/DOWN/NOISE/无效价四分支
- detect_exit_break：污染/干净/无效价/kline 失败
- heal_exit_break_windows：事故端到端（改判 + 两笔假赢单改判为输）、
  同向但幅度失真仍修、干净窗 no-op、kline 失败不动数据、幂等
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
S1 = 1_788_821_700_000          # 事故窗起点（2026-09-07 22:55 UTC）

# ---- 生产实锤数值（勿改：这些是事故现场的真实价格）----
ENTRY = 78_942.005              # 归档 entry（与 kline open 78942.01 吻合，入口干净）
BAD_EXIT = 78_938.385           # 归档 exit（实为 23:00 那根 1m K 线内的价）
KCLOSE = 78_952.19              # kline 真实收盘（UP）
LAST_SAMPLE = 78_964.485        # 末次采样 22:59:46（UP +2.85bp）


# ------------------------------------------------------------------
# 公共桩
# ------------------------------------------------------------------

def _order(**over) -> SimpleNamespace:
    """已结算订单行替身。默认复刻事故单 id=470：
    firsthit_down_v1 DOWN 1.0 USDT @0.03，到手 32.73 股，被误判赢 +31.73。"""
    base = dict(
        id=470, status="FILLED", settled_at=datetime.now(timezone.utc),
        window_start=S1, direction="DOWN", amount_in=str(1 * 10 ** 18),
        quote_json={"averagePrice": 0.03, "filledShareQty": 32.73},
        settle_outcome="DOWN", win=True, pnl=31.73, settle_price=BAD_EXIT,
        market_period="5m", scene_signal_id=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _window_row(**over) -> SimpleNamespace:
    """SentimentWindow 归档行替身（默认 = 事故窗）。"""
    base = dict(
        id=16178, start_time=S1, end_time=S1 + W,
        entry_price=ENTRY, exit_price=BAD_EXIT, outcome="DOWN",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Db:
    """按 stmt 路由：窗口全列查询 / 订单查询 / UPDATE 捕获。"""

    def __init__(self, windows=None, orders=None) -> None:
        self.updates: list[Update] = []
        self._windows = windows or []
        self._orders = orders or []

        wins_res = MagicMock()
        wins_res.scalars.return_value.all.return_value = self._windows
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
                return wins_res
            raise AssertionError(f"未知查询实体: {desc['entity']}")

        self.execute = AsyncMock(side_effect=_execute)
        self.commit = AsyncMock()


def _stub_factory(monkeypatch, db: _Db) -> None:
    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(acr, "async_session_factory", _factory)


def _collector(kclose: float | None = KCLOSE) -> SimpleNamespace:
    return SimpleNamespace(fetch_kline_close=AsyncMock(return_value=kclose))


def _params(stmt) -> dict:
    return stmt.compile().params


# ------------------------------------------------------------------
# outcome_of：结算口径纯函数
# ------------------------------------------------------------------

def test_outcome_of_up_down_noise() -> None:
    assert acr.outcome_of(ENTRY, KCLOSE) == pytest.approx((KCLOSE / ENTRY - 1, "UP"), rel=1e-9)
    assert acr.outcome_of(ENTRY, BAD_EXIT)[1] == "DOWN"
    assert acr.outcome_of(100.0, 100.0) == (0.0, "NOISE")


def test_outcome_of_invalid_prices() -> None:
    """任一端价无效 → (None, None)，调用方不得据此结算。"""
    assert acr.outcome_of(None, KCLOSE) == (None, None)
    assert acr.outcome_of(ENTRY, None) == (None, None)
    assert acr.outcome_of(0.0, KCLOSE) == (None, None)
    assert acr.outcome_of(ENTRY, 0.0) == (None, None)


def test_outcome_of_reproduces_accident() -> None:
    """事故复现：同一 entry 下，污染 exit 判 DOWN、真实收盘判 UP。"""
    assert acr.outcome_of(ENTRY, BAD_EXIT)[1] == "DOWN"
    assert acr.outcome_of(ENTRY, KCLOSE)[1] == "UP"
    # 末次采样（22:59:46）本来就指向 UP，与 kline 一致 —— 证明污染发生在归档环节
    assert acr.outcome_of(ENTRY, LAST_SAMPLE)[1] == "UP"


# ------------------------------------------------------------------
# detect_exit_break：偏差判定（次要触发条件）
# ------------------------------------------------------------------

def test_detect_exit_break_deviated() -> None:
    """事故偏差 1.75bp > 容差 1bp → 检出。"""
    assert abs(BAD_EXIT - KCLOSE) / KCLOSE == pytest.approx(1.7485e-4, rel=1e-3)
    assert acr.detect_exit_break(BAD_EXIT, KCLOSE) is True


def test_detect_exit_break_clean() -> None:
    """已修复（exit == kline close）→ 不检出，保证幂等。"""
    assert acr.detect_exit_break(KCLOSE, KCLOSE) is False
    # 亚 bp 级噪音（mid vs kline 天然价差）→ 不检出
    assert acr.detect_exit_break(KCLOSE * (1 + 5e-5), KCLOSE) is False


def test_detect_exit_break_invalid_inputs() -> None:
    assert acr.detect_exit_break(None, KCLOSE) is False
    assert acr.detect_exit_break(0.0, KCLOSE) is False
    assert acr.detect_exit_break(BAD_EXIT, None) is False
    assert acr.detect_exit_break(BAD_EXIT, 0.0) is False


def test_tolerance_alone_would_miss_accident() -> None:
    """守卫测试：证明「只用 2bp 容差」会漏掉本次事故（heal 必须带 outcome 判定）。

    事故偏差 1.75bp < 2bp。若有人把 EXIT_BREAK_REL_TOL 放宽回 2e-4，
    本测试与 test_heal_repairs_accident_window 会同时暴露回归。
    """
    accident_dev = abs(BAD_EXIT - KCLOSE) / KCLOSE
    assert accident_dev < 2e-4, "事故偏差应小于 2bp（否则当初不会被漏判）"
    assert acr.EXIT_BREAK_REL_TOL <= accident_dev, (
        f"容差 {acr.EXIT_BREAK_REL_TOL} 必须 ≤ 事故偏差 {accident_dev:.3e}，"
        "否则纯容差判定漏掉本次事故")


# ------------------------------------------------------------------
# heal_exit_break_windows：启动自愈端到端
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heal_repairs_accident_window(monkeypatch) -> None:
    """事故端到端：污染窗 exit 78938.385/outcome DOWN + 两笔假赢单 →
    修复后 exit=78952.19、outcome=UP，两单改判为输（pnl = −成本）。"""
    win = _window_row()
    o470 = _order()                                    # 1.00 USDT，误记 +31.73
    o469 = _order(id=469, amount_in=str(88 * 10 ** 16),  # 0.88 USDT，误记 +11.40
                  quote_json={"averagePrice": 0.07, "filledShareQty": 12.28},
                  pnl=11.40)
    db = _Db(windows=[win], orders=[o470, o469])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_exit_break_windows(_collector(KCLOSE))

    assert stats["scanned"] == 1
    assert stats["repaired"] == 1
    assert stats["outcome_flipped"] == 1
    assert stats["orders_resettled"] == 2

    # 归档行被改写为 kline 口径
    assert len(db.updates) == 1
    p = _params(db.updates[0])
    assert p["exit_price"] == KCLOSE
    assert p["actual_return"] == pytest.approx(KCLOSE / ENTRY - 1)
    assert p["outcome"] == "UP"

    # 两笔假赢单改判为输：win=False、pnl=−成本（口径同 TradeSettler）
    for row, cost in ((o470, 1.0), (o469, 0.88)):
        assert row.settle_outcome == "UP"
        assert row.win is False
        assert row.pnl == pytest.approx(-cost)
        assert row.settle_price == pytest.approx(KCLOSE)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_heal_fixes_magnitude_without_flip(monkeypatch) -> None:
    """方向未变但幅度失真（偏差 3bp，同为 UP）→ 仍修 exit/actual_return，
    订单因方向一致而幂等跳过（不改判、不计入 resettled）。

    注：本窗真实涨幅仅 1.29bp（entry 78942.005 → close 78952.19），向下偏 3bp
    会跌破 entry 翻成 DOWN（那是 test_heal_repairs_accident_window 的场景），
    故此处向上偏离以构造「同向、幅度失真」。
    """
    deviated_exit = KCLOSE * (1 + 3e-4)                 # 比收盘高 3bp，仍 > entry
    assert deviated_exit > ENTRY                        # 确保同为 UP
    assert acr.outcome_of(ENTRY, deviated_exit)[1] == "UP"
    win = _window_row(exit_price=deviated_exit, outcome="UP")
    row = _order(direction="UP", settle_outcome="UP", win=True, pnl=5.0)
    db = _Db(windows=[win], orders=[row])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_exit_break_windows(_collector(KCLOSE))

    assert stats["repaired"] == 1
    assert stats["outcome_flipped"] == 0
    assert stats["orders_resettled"] == 0
    p = _params(db.updates[0])
    assert p["exit_price"] == KCLOSE
    assert p["outcome"] == "UP"
    assert row.pnl == pytest.approx(5.0)                # 原值不动


@pytest.mark.asyncio
async def test_heal_clean_window_is_noop(monkeypatch) -> None:
    """归档已等于 kline 口径 → 0 命中、不写 UPDATE（幂等 no-op）。"""
    win = _window_row(exit_price=KCLOSE, outcome="UP")
    db = _Db(windows=[win], orders=[_order(direction="UP", settle_outcome="UP", win=True)])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_exit_break_windows(_collector(KCLOSE))

    assert stats == {"scanned": 1, "repaired": 0, "orders_resettled": 0,
                     "kline_failed": 0, "outcome_flipped": 0}
    assert db.updates == []
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_heal_kline_fail_defers(monkeypatch) -> None:
    """kline 回读失败（0.0）→ 不修（下次启动重试），绝不用可疑值改数据。"""
    win = _window_row()
    row = _order()
    db = _Db(windows=[win], orders=[row])
    _stub_factory(monkeypatch, db)

    stats = await acr.heal_exit_break_windows(_collector(0.0))

    assert stats["scanned"] == 1
    assert stats["repaired"] == 0
    assert stats["kline_failed"] == 1
    assert db.updates == []
    assert row.win is True and row.pnl == pytest.approx(31.73)   # 原值不动


@pytest.mark.asyncio
async def test_heal_is_idempotent(monkeypatch) -> None:
    """修复后二次启动 → 0 命中（exit 已等于 kline close、outcome 已一致）。"""
    win = _window_row()
    db = _Db(windows=[win], orders=[_order()])
    _stub_factory(monkeypatch, db)
    col = _collector(KCLOSE)

    first = await acr.heal_exit_break_windows(col)
    assert first["repaired"] == 1

    # 模拟修复已落库：归档行变成 kline 口径
    fixed = _window_row(exit_price=KCLOSE, outcome="UP")
    db2 = _Db(windows=[fixed], orders=[_order(direction="DOWN", settle_outcome="UP",
                                              win=False, pnl=-1.0)])
    _stub_factory(monkeypatch, db2)
    second = await acr.heal_exit_break_windows(_collector(KCLOSE))

    assert second["repaired"] == 0
    assert db2.updates == []


# ------------------------------------------------------------------
# fetch_kline_close：真实代码路径（mock httpx transport）
#
# 币安 kline 行格式：[0]openTime [1]open [2]high [3]low [4]close
#                    [5]volume [6]closeTime ...
# 收盘守卫是本次修复的关键防线：进行中的 K 线其 close 是「当前价」而非窗口
# 终价，拿它当 exit 等于把旧 bug 换个来源重演。
# ------------------------------------------------------------------

import time as _time

import binance_predict.services.data_collector as dc


class _FakeResp:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """httpx.AsyncClient 替身：异步上下文管理器 + async get。"""

    def __init__(self, payload) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        self.calls.append({"url": url, "params": params})
        return _FakeResp(self._payload)


def _patch_httpx(monkeypatch, payload):
    holder = {}

    def _factory(**kwargs):
        c = _FakeClient(payload)
        holder["client"] = c
        return c

    monkeypatch.setattr(dc.httpx, "AsyncClient", _factory)
    return holder


def _kline_row(open_time: int, open_: float, close: float, close_time: int) -> list:
    return [open_time, str(open_), str(max(open_, close)), str(min(open_, close)),
            str(close), "123.45", close_time, "9999", 42, "1", "2", "0"]


@pytest.mark.asyncio
async def test_fetch_kline_close_returns_closed_candle_close(monkeypatch) -> None:
    """已收盘 K 线（closeTime 在过去）→ 返回 index[4] 收盘价。"""
    now_ms = int(_time.time() * 1000)
    row = _kline_row(S1, ENTRY, KCLOSE, close_time=now_ms - 60_000)
    h = _patch_httpx(monkeypatch, [row])

    got = await dc.BinanceDataCollector().fetch_kline_close("5m", S1)

    assert got == KCLOSE
    params = h["client"].calls[0]["params"]
    assert params["interval"] == "5m"
    assert params["startTime"] == S1
    assert params["limit"] == 1


@pytest.mark.asyncio
async def test_fetch_kline_close_rejects_unclosed_candle(monkeypatch) -> None:
    """收盘守卫：K 线未收盘（closeTime 在未来）→ 返回 0.0，绝不把当前价当窗口终价。"""
    now_ms = int(_time.time() * 1000)
    row = _kline_row(S1, ENTRY, LAST_SAMPLE, close_time=now_ms + 120_000)
    _patch_httpx(monkeypatch, [row])

    got = await dc.BinanceDataCollector().fetch_kline_close("5m", S1)

    assert got == 0.0


@pytest.mark.asyncio
async def test_fetch_kline_close_empty_response(monkeypatch) -> None:
    """空响应 → 0.0（调用方沿用快照，下轮/启动自愈重试）。"""
    _patch_httpx(monkeypatch, [])
    assert await dc.BinanceDataCollector().fetch_kline_close("5m", S1) == 0.0


@pytest.mark.asyncio
async def test_fetch_kline_close_invalid_price(monkeypatch) -> None:
    """收盘价非正 → 0.0。"""
    now_ms = int(_time.time() * 1000)
    _patch_httpx(monkeypatch, [_kline_row(S1, ENTRY, 0.0, close_time=now_ms - 60_000)])
    assert await dc.BinanceDataCollector().fetch_kline_close("5m", S1) == 0.0
