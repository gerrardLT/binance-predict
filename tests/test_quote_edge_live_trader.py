"""报价 edge 实盘执行器单元测试（LIVE 版本可配，momentum/contrarian 双版本）。

覆盖：规则区间守卫（时点/报价/None/开关）、每窗一条、日单量护栏、
重启防重（DB 已有尝试）、成交/未成交路径、execute_signal_trade 执行价护栏、
版本白名单校验与自动护栏推导。

不触网络/真实 DB：trader 与 DB 查询方法全部用替身/monkeypatch。
"""
from __future__ import annotations

import asyncio

import pytest

from binance_predict.config.settings import settings
from binance_predict.services import quote_edge_live_trader as qelt
from binance_predict.services.prediction_trading import BinancePredictionTrader
from binance_predict.services.quote_edge_live_trader import QuoteEdgeLiveTrader

WINDOW_START = 1_000_000_000_000          # 5m 窗口起点（ms）
WINDOW_END = WINDOW_START + 300_000


def _fake_order(status="FILLED", error_message=None) -> dict:
    """execute_signal_trade 返回的 dict 快照替身。"""
    return {
        "id": 1,
        "status": status,
        "signal_version": "quote_momentum_v1",
        "window_start": WINDOW_START,
        "order_id": "ORD-1",
        "token_id": "TOKEN-DOWN",
        "amount_in": "5000000000000000000",
        "average_price": None,
        "error_message": error_message,
    }


class _FakeTrader:
    """execute_signal_trade 替身：记录调用、返回可配置结果。"""

    def __init__(self, result=object()):
        self.calls: list[dict] = []
        # 哨兵区分"未传参（默认 FILLED）"与"显式传 None（落库异常）"
        self.result = _fake_order() if isinstance(result, object) and type(result) is object else result

    async def execute_signal_trade(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _make_trader(monkeypatch, trader: _FakeTrader,
                 version: str = "quote_momentum_v1") -> QuoteEdgeLiveTrader:
    monkeypatch.setattr(settings, "quote_momentum_live_enabled", True)
    monkeypatch.setattr(settings, "quote_edge_live_version", version)
    t = QuoteEdgeLiveTrader(trader)

    async def _no_filled(self):
        return 0

    async def _no_attempt(self, _ws):
        return False

    async def _no_backfill(self, _ws):
        return None

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_count_filled_today", _no_filled)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_has_attempt", _no_attempt)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_backfill_signal_link", _no_backfill)
    return t


async def _drain(t: QuoteEdgeLiveTrader) -> None:
    """等待在途任务（下单/回填）跑完。"""
    while t._tasks:
        await asyncio.sleep(0.01)


# ============================================================
# check() 守卫：区间/开关/None
# ============================================================

def test_check_disabled_no_fire(monkeypatch) -> None:
    monkeypatch.setattr(settings, "quote_momentum_live_enabled", False)
    t = QuoteEdgeLiveTrader(_FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is False


def test_check_none_price_no_fire(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, None) is False


def test_check_time_window_guards(monkeypatch) -> None:
    """t<90s 或 t≥120s → 不命中（半开区间 [90,120)）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 89_999, 0.71) is False
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 120_000, 0.71) is False


def test_check_quote_band_guards(monkeypatch) -> None:
    """q∉[0.69,0.75) → 不命中（上界开区间）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.68) is False
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.75) is False


def test_check_rule_bounds_frozen(monkeypatch) -> None:
    """规则常量必须与冻结口径一致（防再次口径漂移），两个 LIVE 候选都锁。"""
    assert qelt.QUOTE_EDGE_RULES["quote_momentum_v1"] == (90.0, 120.0, 0.69, 0.75)
    assert qelt.QUOTE_EDGE_RULES["quote_contrarian_v1"] == (45.0, 60.0, 0.15, 0.25)


@pytest.mark.asyncio
async def test_check_hit_fires_once_per_window(monkeypatch) -> None:
    """区间内首个采样开火；同窗后续采样不再开火（内存占位）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is True
    assert t.check(WINDOW_START, WINDOW_END, ts + 15_000, 0.72) is False
    assert WINDOW_START in t._fired
    await _drain(t)


# ============================================================
# _fire 路径：成交/护栏/防重
# ============================================================

@pytest.mark.asyncio
async def test_fire_success_calls_trader_and_backfills(monkeypatch) -> None:
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == "quote_momentum_v1"
    assert call["window_start"] == WINDOW_START
    assert call["amount_usdt"] == settings.quote_momentum_live_amount_usdt
    assert call["max_exec_price"] == 0.78  # momentum 自动推导（旧默认口径不变）
    assert t._fire_total == 1


@pytest.mark.asyncio
async def test_fire_daily_cap_blocks(monkeypatch) -> None:
    """日单量护栏：当日成交达上限 → 不调 trader。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)

    async def _full(self):
        return t._max_daily

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_count_filled_today", _full)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []
    assert t._fire_total == 0


@pytest.mark.asyncio
async def test_fire_restart_dedup_blocks(monkeypatch) -> None:
    """重启防重：DB 已有本窗尝试（FILLED/FAILED）→ 跳过不再下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake)

    async def _has(self, _ws):
        return True

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_has_attempt", _has)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_fire_failed_order_no_crash(monkeypatch) -> None:
    """trader 返回 FAILED（如护栏弃单）→ 计入 fire_total、无回填、不抛异常。"""
    fake = _FakeTrader(result=_fake_order(status="FAILED", error_message="执行价护栏弃单"))
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert len(fake.calls) == 1
    assert t._fire_total == 1


@pytest.mark.asyncio
async def test_fire_invalidates_balance_cache_hook(monkeypatch) -> None:
    """产生订单（FILLED）后触发余额缓存作废钩子（前端实时余额依赖）。"""
    fake = _FakeTrader()  # 默认 FILLED
    t = _make_trader(monkeypatch, fake)
    calls: list[None] = []
    t._on_balance_change = lambda: calls.append(None)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert calls == [None]  # 产生订单即作废（一次）


@pytest.mark.asyncio
async def test_fire_no_order_skips_balance_hook(monkeypatch) -> None:
    """未产生订单（None）不触发余额钩子（未花钱，缓存无需作废）。"""
    fake = _FakeTrader(result=None)
    t = _make_trader(monkeypatch, fake)
    calls: list[None] = []
    t._on_balance_change = lambda: calls.append(None)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert calls == []


@pytest.mark.asyncio
async def test_fire_trader_none_no_crash(monkeypatch) -> None:
    """trader 返回 None（落库异常）→ 不崩、不回填。"""
    fake = _FakeTrader(result=None)
    t = _make_trader(monkeypatch, fake)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert t._fire_total == 0


def test_status_shape(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    s = t.status()
    assert s["enabled"] is True
    assert s["version"] == "quote_momentum_v1"
    assert s["fire_total"] == 0


# ============================================================
# contrarian LIVE（2026-08-22 切换的默认版本）
# ============================================================

def test_default_live_version_is_contrarian() -> None:
    """settings 默认必须切到 contrarian（唯一正 EV，用户 2026-08-22 圈定）；
    防后续误改默认值静默回 momentum（负 EV）。"""
    assert settings.quote_edge_live_version == "quote_contrarian_v1"
    assert qelt.LIVE_ALLOWED_VERSIONS == ("quote_momentum_v1", "quote_contrarian_v1")


def test_contrarian_band_guards(monkeypatch) -> None:
    """contrarian 区间 t∈[45,60)s×q∈[0.15,0.25)（半开区间）。"""
    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    ts = WINDOW_START + 50_000
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 44_999, 0.20) is False
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 60_000, 0.20) is False
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.1499) is False
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.25) is False
    assert WINDOW_START not in t._fired


@pytest.mark.asyncio
async def test_contrarian_fire_derived_exec_guard(monkeypatch) -> None:
    """contrarian 开火：signal_version 透传、执行价护栏自动推导 0.28（0.25+0.03）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, version="quote_contrarian_v1")
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20) is True
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "quote_contrarian_v1"
    assert call["prediction"] == "DOWN"
    assert call["max_exec_price"] == 0.28
    assert t._fire_total == 1


def test_momentum_exec_guard_backcompat(monkeypatch) -> None:
    """momentum 未显式配置时推导 0.78（与旧默认完全一致，回测口径不变）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    assert t._max_exec == 0.78


def test_explicit_exec_guard_override(monkeypatch) -> None:
    """显式配置 max_exec_price 时覆盖推导（收紧场景）。"""
    monkeypatch.setattr(settings, "quote_momentum_live_max_exec_price", 0.26)
    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    assert t._max_exec == 0.26


def test_unknown_version_rejected(monkeypatch) -> None:
    """v2 门禁版/未知版本 → 构造抛 ValueError（配了会以 v1 区间裸下单丢门禁）。"""
    monkeypatch.setattr(settings, "quote_edge_live_version", "quote_momentum_v2")
    with pytest.raises(ValueError, match="白名单"):
        QuoteEdgeLiveTrader(_FakeTrader())
    monkeypatch.setattr(settings, "quote_edge_live_version", "nonexistent")
    with pytest.raises(ValueError, match="不支持"):
        QuoteEdgeLiveTrader(_FakeTrader())


def test_amount_over_hard_cap_rejected(monkeypatch) -> None:
    """Low#5：金额超硬上限 50 → 构造抛 ValueError（配置误写拒绝启动，不靠自律）。"""
    monkeypatch.setattr(settings, "quote_momentum_live_amount_usdt", 51.0)
    with pytest.raises(ValueError, match="硬上限"):
        QuoteEdgeLiveTrader(_FakeTrader())


async def _patch_db_noop(monkeypatch) -> None:
    async def _no_filled(self):
        return 0

    async def _no_attempt(self, _ws):
        return False

    async def _no_backfill(self, _ws):
        return None

    monkeypatch.setattr(QuoteEdgeLiveTrader, "_count_filled_today", _no_filled)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_has_attempt", _no_attempt)
    monkeypatch.setattr(QuoteEdgeLiveTrader, "_backfill_signal_link", _no_backfill)


@pytest.mark.asyncio
async def test_toggle_runtime_switch(monkeypatch) -> None:
    """P2-1 运行时开关：构造 OFF → 区间命中也不开火；toggle 置 True → 命中。

    toggle 只改实例标志位，不动 settings（重启回落 .env 默认，fail-safe）。"""
    monkeypatch.setattr(settings, "quote_momentum_live_enabled", False)
    monkeypatch.setattr(settings, "quote_edge_live_version", "quote_momentum_v1")
    fake = _FakeTrader()
    t = QuoteEdgeLiveTrader(fake)
    await _patch_db_noop(monkeypatch)
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is False  # OFF：命中也不开火
    assert not t._fired
    t._enabled = True  # /api/live/toggle 置位路径
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) is True
    s = t.status()
    assert s["enabled"] is True and s["enabled_at_startup"] is False  # 快照不受 toggle 影响
    await _drain(t)


def test_status_dual_enabled_fields(monkeypatch) -> None:
    """status 双字段：enabled=运行时状态 / enabled_at_startup=启动配置快照。"""
    t = _make_trader(monkeypatch, _FakeTrader())  # enabled=True 构造
    assert t.status()["enabled"] is True
    assert t.status()["enabled_at_startup"] is True
    t._enabled = False  # toggle 关闭后
    assert t.status()["enabled"] is False
    assert t.status()["enabled_at_startup"] is True  # 快照不变


# ============================================================
# execute_signal_trade：先占位后下单 + 执行价护栏 + 动态滑点（方法级 monkeypatch）
# ============================================================

def _make_real_trader(monkeypatch) -> BinancePredictionTrader:
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xWALLET"
    trader._wallet_id = "WID"

    async def _list():
        trader._down_token_id = "TOKEN-DOWN"
        return []

    monkeypatch.setattr(trader, "list_markets", _list)
    return trader


def _pending_order():
    return _fake_order(status="PENDING")


@pytest.mark.asyncio
async def test_signal_trade_price_guard_rejects(monkeypatch) -> None:
    """报价均价 0.82 > 上限 0.78 → 弃单，占位更新 FAILED，不 place_order。"""
    trader = _make_real_trader(monkeypatch)
    pending = _pending_order()
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None):
        return pending

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status}

    async def _quote(_token, _side, amount_usdt=None):
        return {"averagePrice": 0.82, "amountIn": "1", "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        raise AssertionError("护栏弃单不应走到 place_order")

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"  # 返回 dict 快照（非占位 ORM 对象）
    assert updates[0][0] == "FAILED"
    assert "执行价护栏" in updates[0][1]["error_message"]


@pytest.mark.asyncio
async def test_signal_trade_success_dynamic_slippage(monkeypatch) -> None:
    """报价 0.71 ≤ 上限 → 成交；滑点按护栏价收紧至 985bps（0.78/0.71−1），
    成交价无法突破护栏（Medium#2）；占位更新 FILLED 带 orderId。"""
    trader = _make_real_trader(monkeypatch)
    pending = _pending_order()
    updates: list[tuple] = []
    slippage_seen: list[int] = []

    async def _reserve(_v, _ws, direction=None):
        return pending

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, "order_id": kwargs.get("order_id")}

    async def _quote(_token, _side, amount_usdt=None):
        assert amount_usdt == 5.0  # 自定义金额透传到报价
        return {"averagePrice": 0.71, "amountIn": "5", "amountOut": "7", "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        slippage_seen.append(slippage_bps)
        return {"orderId": "ORD-9"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FILLED" and order["order_id"] == "ORD-9"
    assert slippage_seen == [985]
    assert updates[0][0] == "FILLED"
    assert updates[0][1]["order_id"] == "ORD-9"


@pytest.mark.asyncio
async def test_signal_trade_duplicate_window_skips(monkeypatch) -> None:
    """占位失败（同窗已有 PENDING/终态行）→ 花钱前拒绝，不取报价不下单（High#1）。"""
    trader = _make_real_trader(monkeypatch)

    async def _reserve(_v, _ws, direction=None):
        return None

    async def _quote(*a, **k):
        raise AssertionError("重复窗口不应走到取报价")

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "get_quote", _quote)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order is None


# ============================================================
# 生命周期与自愈（High#1 stop 拒新单 / Low#4 回填自愈）
# ============================================================

@pytest.mark.asyncio
async def test_stop_rejects_new_fires(monkeypatch) -> None:
    """stop 后 check 拒绝派生新下单任务（shutdown 窗口期保护，High#1）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    await t.stop()
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) is False


class _HealResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _HealRow:
    """sa_select(window_start, signal_version).all() 的 Row 替身（可解包两列）。"""

    def __init__(self, ws, ver):
        self.ws, self.ver = ws, ver

    def __iter__(self):
        return iter((self.ws, self.ver))


class _HealSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        return _HealResult(self._rows)


@pytest.mark.asyncio
async def test_heal_once_backfills_stale_orders(monkeypatch) -> None:
    """自愈扫描：缺 signal_id 的陈旧订单逐窗回填，按各自 signal_version
    （跨版本：热切换后旧版本订单同样自愈，不随当前绑定漂移）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    monkeypatch.setattr(qelt, "async_session_factory",
                        lambda: _HealSession([_HealRow(WINDOW_START, "quote_momentum_v1")]))
    calls: list[tuple[int, str | None]] = []

    async def _bf(ws, version=None):
        calls.append((ws, version))

    t._backfill_signal_link = _bf
    await t._heal_once()
    assert calls == [(WINDOW_START, "quote_momentum_v1")]
    assert t._healed_total == 1


@pytest.mark.asyncio
async def test_heal_once_no_stale_orders(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    monkeypatch.setattr(qelt, "async_session_factory", lambda: _HealSession([]))
    await t._heal_once()
    assert t._healed_total == 0


# ============================================================
# switch_version：运行时热切信号（2026-08-23 前端可选信号实盘开关）
# ============================================================

@pytest.mark.asyncio
async def test_switch_version_updates_band_and_guard(monkeypatch) -> None:
    """热切 momentum→contrarian：区间换为新版、护栏重算 0.28、
    _fired 记忆保留（同窗双版本双单防线）。"""
    monkeypatch.setattr(settings, "quote_momentum_live_max_exec_price", None)
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, version="quote_momentum_v1")
    t._fired.add(999)  # 模拟旧版本已开火窗口

    t.switch_version("quote_contrarian_v1")

    assert t._version == "quote_contrarian_v1"
    assert t._max_exec == 0.28
    assert 999 in t._fired  # 记忆保留
    # 旧区间（t=100s q=0.71 momentum 命中点）不再命中
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) is False
    # 新区间（t=50s q=0.20）命中（check 命中会派生任务，需异步环境）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20) is True
    await _drain(t)


@pytest.mark.asyncio
async def test_switch_version_fire_uses_new_version(monkeypatch) -> None:
    """热切后开火：signal_version/max_exec_price 均按新版本下单。"""
    monkeypatch.setattr(settings, "quote_momentum_live_max_exec_price", None)
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, version="quote_contrarian_v1")
    t.switch_version("quote_momentum_v1")
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) is True
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "quote_momentum_v1"
    assert call["max_exec_price"] == 0.78


def test_switch_version_unknown_rejected(monkeypatch) -> None:
    """非白名单版本（v2 门禁版/未知）→ ValueError，实例状态不变。"""
    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_momentum_v1")
    with pytest.raises(ValueError, match="白名单"):
        t.switch_version("quote_momentum_v2")
    with pytest.raises(ValueError, match="不支持"):
        t.switch_version("nonexistent")
    assert t._version == "quote_momentum_v1"  # 状态不变


def test_switch_version_explicit_exec_price_rejected(monkeypatch) -> None:
    """显式配置 max_exec_price（与特定版本绑定）→ 拒绝热切（语义不明）。"""
    monkeypatch.setattr(settings, "quote_momentum_live_max_exec_price", 0.26)
    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    with pytest.raises(ValueError, match="显式配置"):
        t.switch_version("quote_momentum_v1")
    assert t._version == "quote_contrarian_v1"


def test_available_versions_shape() -> None:
    """available_versions：白名单全集 + 区间 + 自动护栏（前端选择层渲染源）。"""
    avs = QuoteEdgeLiveTrader.available_versions()
    assert [a["version"] for a in avs] == ["quote_momentum_v1", "quote_contrarian_v1"]
    by_v = {a["version"]: a for a in avs}
    assert by_v["quote_momentum_v1"]["auto_max_exec_price"] == 0.78
    assert by_v["quote_contrarian_v1"]["auto_max_exec_price"] == 0.28


def test_status_lists_available_versions(monkeypatch) -> None:
    """status 携带 available_versions + version_at_startup（重启回落基准）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    s = t.status()
    assert [a["version"] for a in s["available_versions"]] == list(
        qelt.LIVE_ALLOWED_VERSIONS)
    assert s["version_at_startup"] == settings.quote_edge_live_version


@pytest.mark.asyncio
async def test_guard_sqls_are_cross_version(monkeypatch) -> None:
    """防重/日护栏 SQL 跨版本口径：in_(白名单) 而非 == 当前版本
    （唯一键按 (signal_version, window_start) 隔离，DB 层拦不住跨版本重复；
    日护栏不跨版本合计则切版本可重置计数继续打）。

    注：直接构造执行器（不走 _make_trader——它把 _has_attempt/_count_
    filled_today 桩掉了，验 SQL 必须走真方法）。
    """
    from sqlalchemy.dialects import postgresql

    monkeypatch.setattr(settings, "quote_momentum_live_enabled", False)
    monkeypatch.setattr(settings, "quote_edge_live_version", "quote_momentum_v1")
    t = QuoteEdgeLiveTrader(_FakeTrader())

    class _R:
        def first(self):
            return None

        def scalar(self):
            return 0

    class _S:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, stmt):
            compiled = str(stmt.compile(dialect=postgresql.dialect(),
                                        compile_kwargs={"literal_binds": True}))
            _S.seen.append(compiled)
            return _R()

    _S.seen = []
    monkeypatch.setattr(qelt, "async_session_factory", lambda: _S())

    await t._has_attempt(WINDOW_START)
    await t._count_filled_today()
    joined = "\n".join(_S.seen)
    assert "signal_version IN" in joined
    assert "quote_momentum_v1" in joined and "quote_contrarian_v1" in joined


# ============================================================
# POST /api/live/toggle（端点层：可选 version 热切）
# ============================================================

@pytest.mark.asyncio
async def test_toggle_endpoint_switches_version(monkeypatch) -> None:
    """带 version 开启：先切版本再置 enabled；响应携带新 status 与重启回落警示。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    t._enabled = False
    monkeypatch.setattr(settings, "quote_momentum_live_max_exec_price", None)
    monkeypatch.setattr(m, "quote_edge_live_trader", t)

    out = await m.live_toggle(ToggleLiveRequest(enabled=True, version="quote_momentum_v1"), _=None)
    assert "error" not in out
    assert t._version == "quote_momentum_v1"
    assert t._enabled is True
    assert out["status"]["version"] == "quote_momentum_v1"
    assert "quote_contrarian_v1" in out["warning"]  # 提示重启回落到启动默认


@pytest.mark.asyncio
async def test_toggle_endpoint_bad_version_atomic(monkeypatch) -> None:
    """非法 version → error 早退，enabled 不动（原子性：切不过就不开火）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    t._enabled = False
    monkeypatch.setattr(m, "quote_edge_live_trader", t)

    out = await m.live_toggle(
        ToggleLiveRequest(enabled=True, version="quote_momentum_v2"), _=None)
    assert "error" in out
    assert t._enabled is False  # 未被置位
    assert t._version == "quote_contrarian_v1"


@pytest.mark.asyncio
async def test_toggle_endpoint_without_version_keeps_binding(monkeypatch) -> None:
    """不带 version（旧前端/只开关）：维持当前绑定，行为与 P2-1 完全一致。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), version="quote_contrarian_v1")
    monkeypatch.setattr(m, "quote_edge_live_trader", t)

    out = await m.live_toggle(ToggleLiveRequest(enabled=True), _=None)
    assert "error" not in out
    assert t._version == "quote_contrarian_v1"
    assert t._enabled is True
