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
from binance_predict.db.models import (
    FakeBreakoutSignal,
    KlineShadowSignal,
    SentimentWindow,
    TradeOrderModel,
)
from binance_predict.services.trade_settler import TradeSettler

WS = 1_787_400_000_000  # 任意 5m 窗口起点 ms


def _row(**over) -> SimpleNamespace:
    """TradeOrderModel 待结算行替身（FILLED + 未结算 + 超 7min 延迟）。

    market_period/scene_signal_id 为 5m 默认值（15m 分流分支见下方
    「15m 分流」区块：scene 族走 FakeBreakoutSignal，K 线影子族走
    KlineShadowSignal）。
    """
    base = dict(
        id=11, status="FILLED", settled_at=None, window_start=WS,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        direction="DOWN", amount_in=str(10 ** 18),  # 1 USDT
        quote_json={"averagePrice": 0.5},
        market_period="5m", scene_signal_id=None,
        signal_version="firsthit_g7_base_v1",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _row15(**over) -> SimpleNamespace:
    """15m 且无 scene_signal_id 的订单行替身（K 线影子族：s2_cond / nextbar hm·ih）。"""
    base = dict(
        market_period="15m", scene_signal_id=None,
        signal_version="s2_cond_t4_v1", direction="UP",
    )
    base.update(over)
    return _row(**base)


def _window(outcome: str, exit_price: float = 43250.0) -> SimpleNamespace:
    """SentimentWindow 归档行替身。"""
    return SimpleNamespace(start_time=WS, outcome=outcome, exit_price=exit_price)


def _shadow(settle_outcome: str | None, *, status: str = "SETTLED",
            settle_open: float = 79104.0, settle_close: float = 78813.0,
            win: bool | None = None,
            created_at: datetime | None = None) -> SimpleNamespace:
    """KlineShadowSignal 影子行替身（检测器次根收盘结算后写入）。

    ⚠ 探测器约定：NOISE 行 status 记为 "EXPIRED" 但 settle_outcome 仍为
    "NOISE"（见 s2_cond_shadow_detector._settle_pending），故结算器只认
    settle_outcome 不认 status——test_kline_shadow_noise 即锁此约定。
    """
    return SimpleNamespace(
        version="s2_cond_t4_v1", target_bar_start=WS,
        settle_outcome=settle_outcome, win=win, status=status,
        settle_open=settle_open, settle_close=settle_close,
        created_at=created_at or (datetime.now(timezone.utc) - timedelta(hours=1)),
    )


class _Db:
    """按 stmt 类型路由 execute：订单 / 窗口 / 影子信号查询 + UPDATE 捕获。

    scene 路径用 session.get(FakeBreakoutSignal, id)，故另桩 get()。
    """

    def __init__(self, rows, window, update_rowcount: int = 1,
                 shadow=None, scene_signal=None) -> None:
        self.updates: list[Update] = []
        self._update_rowcount = update_rowcount

        orders_res = MagicMock()
        orders_res.scalars.return_value.all.return_value = rows
        window_res = MagicMock()
        window_res.scalar_one_or_none.return_value = window
        shadow_res = MagicMock()
        shadow_res.scalar_one_or_none.return_value = shadow

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
            if entity is KlineShadowSignal:
                return shadow_res
            raise AssertionError(f"未知查询实体: {entity}")

        self.execute = AsyncMock(side_effect=_execute)
        self.get = AsyncMock(return_value=scene_signal)
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
async def test_settle_win_prefers_filled_shares(monkeypatch) -> None:
    """赢向优先按币安实际到手股数算：pnl = filledShareQty − 成本（2026-08-28）。

    marketProviderFee 以少给股数体现（生产实锤：2U/0.16 → 12.28 股而非 12.5）；
    无费口径 2/0.16−2=10.5 比币安实现盈亏 10.27 高约费用额，
    股数口径 12.28−2=10.28 才对得上。
    """
    db = _Db([_row(direction="DOWN", amount_in=str(2 * 10 ** 18),
                   quote_json={"averagePrice": 0.16, "filledShareQty": 12.28})],
             _window("DOWN"))
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(12.28 - 2.0)


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


# ------------------------------------------------------------------
# K 线影子族 15m 订单（s2_cond / nextbar hm·ih）——Bug #1 修复验证
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kline_shadow_15m_win(monkeypatch) -> None:
    """K 线影子结算：15m+ 无 scene_signal_id → win=True、pnl=shares−amount。

    对照 Bug #1 生产实锤 id=466: s2_cond_t4_v1(UP)/窗口 DOWN 被误判 EXPIRED/pnl=0。
    正确应读 KlineShadowSignal(settle_outcome='UP',settle_close=78813)→win=True。
    """
    shadow = _shadow("UP", win=True, settle_close=78813.0)
    db = _Db([_row15(signal_version="s2_cond_t4_v1", amount_in=str(2 * 10**18),
                     quote_json={"averagePrice": 0.16, "filledShareQty": 12.28})],
             None, shadow=shadow)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "UP"
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(12.28 - 2.0)  # shares−cost
    assert p["settle_price"] == pytest.approx(78813.0)
    assert isinstance(p["settled_at"], datetime)


@pytest.mark.asyncio
async def test_kline_shadow_15m_lose(monkeypatch) -> None:
    """K 线影子结算：direction≠outcome → win=False、pnl=-amount（输）。"""
    shadow = _shadow("DOWN", win=False, settle_close=78500.0)
    db = _Db([_row15(direction="UP", amount_in=str(1 * 10**18))], None, shadow=shadow)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "DOWN"
    assert p["win"] is False
    assert p["pnl"] == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_kline_shadow_noise(monkeypatch) -> None:
    """NOISE：探测器 status="EXPIRED"但 settle_outcome="NOISE"→win=None/pnl=0.0。

    锁定探测器约定（见 s2_cond_shadow_detector._settle_pending line 364-370）：
    NOISE 行不写 status="SETTLED"而写"EXPIRED"，但 settle_outcome 仍为"NOISE"。
    结算器只认 outcome 不认 status，避免把平盘误判为「无结算源」。
    """
    shadow = _shadow("NOISE", status="EXPIRED", settle_close=79000.0)
    db = _Db([_row15()], None, shadow=shadow)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "NOISE"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] == pytest.approx(79000.0)


@pytest.mark.asyncio
async def test_kline_shadow_missing_retry(monkeypatch) -> None:
    """影子行缺失且未超 24h → 不写 UPDATE，下轮重试（gate 恢复/检测器补录后可结算）。"""
    db = _Db([_row15(created_at=datetime.now(timezone.utc) - timedelta(hours=2))],
             None, shadow=None)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert db.updates == []


@pytest.mark.asyncio
async def test_kline_shadow_missing_expired(monkeypatch) -> None:
    """影子行缺失超 24h → EXPIRED 兜底（无法判赢，防永挂「在途持仓」）。"""
    old = datetime.now(timezone.utc) - timedelta(hours=25)
    db = _Db([_row15(created_at=old)], None, shadow=None)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "EXPIRED"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] is None


@pytest.mark.asyncio
async def test_kline_shadow_pending_retry(monkeypatch) -> None:
    """影子行存在但无 outcome（PENDING 未结算）→ 24h 内重试。"""
    shadow = _shadow(None, status="PENDING")
    db = _Db([_row15(created_at=datetime.now(timezone.utc) - timedelta(hours=2))],
             None, shadow=shadow)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert db.updates == []


@pytest.mark.asyncio
async def test_kline_shadow_idempotent_guard(monkeypatch) -> None:
    """UPDATE rowcount=0（已被并发结算）→ 幂等守卫生效，不计入 settled。"""
    shadow = _shadow("UP", win=True, settle_close=78800.0)
    db = _Db([_row15()], None, shadow=shadow, update_rowcount=0)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert len(db.updates) == 1  # 语句仍发出，只是守卫判定未生效


@pytest.mark.asyncio
async def test_nextbar_15m_down_direction(monkeypatch) -> None:
    """nextbar hm_inside_15m_v2(DOWN) 路由到影子路径：outcome=DOWN→win=True。

    证明分流按「有无 scene_signal_id」而非硬编码通道名：
    nextbar 族同样落 kline_shadow_signals，无需修改 trade_settler 即可支持。
    """
    shadow = _shadow("DOWN", win=True, settle_close=78500.0)
    db = _Db([_row15(signal_version="hm_inside_15m_v2", direction="DOWN",
                     amount_in=str(1 * 10**18))], None, shadow=shadow)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "DOWN"
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(1.0 / 0.5 - 1.0)  # amount/avg_price−amount


@pytest.mark.asyncio
async def test_15m_with_scene_signal_id_uses_fake_breakout(monkeypatch) -> None:
    """15m + 有 scene_signal_id → 走 FakeBreakoutSignal（scene 族），非影子路径。

    回归保护：routing 改为「15m 且 scene_signal_id IS NULL → 影子路径」后，
    原 scene 订单仍走旧口径（周期锚点 P(E)/P(S)，settle_price=settle_btc_price）。
    """
    scene_sig = SimpleNamespace(
        id=42, settle_outcome="UP", settle_btc_price=79000.0,
        settle_deadline=int((datetime.now(timezone.utc)
                             + timedelta(hours=1)).timestamp() * 1000),
    )
    db = _Db(
        [_row(market_period="15m", scene_signal_id=42, direction="UP",
              signal_version="scene_bear_exhaust")],
        None, scene_signal=scene_sig)
    _stub_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "UP"
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(1.0 / 0.5 - 1.0)
    assert p["settle_price"] == pytest.approx(79000.0)
    db.get.assert_awaited_once()
