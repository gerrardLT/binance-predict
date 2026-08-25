"""多通道实盘执行器单元测试（MultiLiveTrader，2026-08-24 取代 QuoteEdgeLiveTrader）。

覆盖（对照计划 Step 10 八组用例）：
- live_channels 配置解析：默认值 / JSON 覆盖 / 非法值拒启（不靠自律靠拒启）
- quote_edge 多通道同窗独立开火（每通道每窗一单，互不阻塞）
- v2 门禁：BTC/entry 缺失不触发；chg 阈值边界（momentum_v2 −0.10% / contrarian_v2 +0.10%）
- per-channel 日限停火（A 打满不影响 B）；重启防重（DB 已有尝试）
- x4：PENDING 轮询幂等（seen set）；错过决策点不追；成交即回填 signal_id
- 场景钩子：S1→DOWN/15m、S2→UP/15m、S5 独立通道；未启用零下单；日限停火
- set_channel 运行时热调 / status 形状 / stop 拒新单
- execute_signal_trade：15m token 分流透传 + 旧 5m 行为迁移（护栏弃单/动态滑点/重复窗）
- 15m 结算分流：settle_outcome UP/DOWN → win/pnl；PENDING 重试；超 24h EXPIRED

不触网络/真实 DB：trader 用 FakeTrader、DB 查询方法全部方法级 monkeypatch
（延续旧 test_quote_edge_live_trader.py 风格，结算桩仿 test_trade_settler.py）。
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Select, Update

import binance_predict.services.multi_live_trader as milt
import binance_predict.services.trade_settler as ts_mod
from binance_predict.config.settings import settings
from binance_predict.db.models import FakeBreakoutSignal, TradeOrderModel
from binance_predict.services.live_channels import (
    LIVE_CHANNELS,
    parse_channel_config,
    scene_pattern_to_channel,
)
from binance_predict.services.multi_live_trader import MultiLiveTrader
from binance_predict.services.prediction_trading import BinancePredictionTrader
from binance_predict.services.trade_settler import TradeSettler

WINDOW_START = 1_000_000_000_000           # 5m 窗口起点（ms）
WINDOW_END = WINDOW_START + 300_000
MARKET_START_15M = 1_000_000_000_000       # 15m 周期起点（与 5m 窗起点数值重合是常态）


def _fake_order(status: str = "FILLED", error_message: str | None = None,
                signal_version: str = "quote_momentum_v1") -> dict:
    """execute_signal_trade 返回的 dict 快照替身。"""
    return {
        "id": 1,
        "status": status,
        "signal_version": signal_version,
        "window_start": WINDOW_START,
        "order_id": "ORD-1",
        "token_id": "TOKEN-DOWN",
        "amount_in": "5000000000000000000",
        "average_price": None,
        "error_message": error_message,
    }


class _FakeTrader:
    """execute_signal_trade 替身：记录调用、返回可配置结果。"""

    def __init__(self, result: dict | None | object = object()):
        self.calls: list[dict] = []
        self.result = (_fake_order() if isinstance(result, object)
                       and type(result) is object else result)

    async def execute_signal_trade(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _make_trader(monkeypatch, trader: _FakeTrader,
                 channels: list[str] | None = None,
                 overrides: dict | None = None) -> MultiLiveTrader:
    """构造执行器：live_channels_json 启用指定通道；DB 访问全部置空。"""
    monkeypatch.setattr(settings, "live_default_amount_usdt", 2.0)
    monkeypatch.setattr(settings, "live_default_max_daily_orders", 100)
    monkeypatch.setattr(settings, "live_channels_json", "")
    cfg: dict[str, dict] = {ch: {"enabled": True} for ch in (channels or [])}
    if overrides:
        cfg.update(overrides)
    if cfg:
        monkeypatch.setattr(settings, "live_channels_json", json.dumps(cfg))
    t = MultiLiveTrader(trader)

    async def _no_filled(self, version: str) -> int:
        return 0

    async def _no_attempt(self, version: str, ws: int) -> bool:
        return False

    async def _no_backfill(self, ws: int, version: str | None = None) -> None:
        return None

    async def _no_link(self, version: str, ws: int, sig_id: int) -> None:
        return None

    monkeypatch.setattr(MultiLiveTrader, "_count_filled_today", _no_filled)
    monkeypatch.setattr(MultiLiveTrader, "_has_attempt", _no_attempt)
    monkeypatch.setattr(MultiLiveTrader, "_backfill_signal_link", _no_backfill)
    monkeypatch.setattr(MultiLiveTrader, "_link_signal_id", _no_link)
    return t


async def _drain(t: MultiLiveTrader) -> None:
    """等待在途任务（下单/回填/轮询派生）跑完。"""
    while t._tasks:
        await asyncio.sleep(0.01)


class _SelectSession:
    """sa_select(...).all() 的 DB 会话桩（x4 轮询 / status 聚合同构）。"""

    def __init__(self, rows: list) -> None:
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        res = MagicMock()
        res.all.return_value = self._rows
        return res


def _stub_select_db(monkeypatch, rows: list) -> None:
    monkeypatch.setattr(milt, "async_session_factory",
                        lambda: _SelectSession(rows))


# ============================================================
# 组 1：live_channels 配置解析（纯函数，拒启哲学）
# ============================================================

def test_parse_defaults_all_off(monkeypatch) -> None:
    """默认：全 10 通道 OFF、金额/日限取全局默认（用户拍板 2U / 100 单）。"""
    monkeypatch.setattr(settings, "live_default_amount_usdt", 2.0)
    monkeypatch.setattr(settings, "live_default_max_daily_orders", 100)
    monkeypatch.setattr(settings, "live_channels_json", "")
    cfgs = parse_channel_config()
    assert len(cfgs) == 10
    assert all(not c.enabled for c in cfgs.values())
    assert all(c.amount_usdt == 2.0 for c in cfgs.values())
    assert all(c.max_daily_orders == 100 for c in cfgs.values())
    assert all(c.max_exec_price is None for c in cfgs.values())  # 缺省回落 auto


def test_parse_overrides_applied(monkeypatch) -> None:
    """JSON 覆盖：enabled/金额/日限/护栏逐通道生效，未提及通道保持默认。"""
    monkeypatch.setattr(settings, "live_channels_json", json.dumps({
        "x4_v1": {"enabled": True, "amount_usdt": 1.0,
                  "max_daily_orders": 50, "max_exec_price": 0.48},
        "quote_contrarian_v1": {"enabled": True},
    }))
    cfgs = parse_channel_config()
    x4 = cfgs["x4_v1"]
    assert x4.enabled and x4.amount_usdt == 1.0
    assert x4.max_daily_orders == 50 and x4.max_exec_price == 0.48
    assert cfgs["quote_contrarian_v1"].enabled is True
    assert cfgs["quote_momentum_v1"].enabled is False   # 未提及通道不受影响


def test_parse_unknown_channel_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"nonexistent": {"enabled": True}}))
    with pytest.raises(ValueError, match="未知通道"):
        parse_channel_config()


def test_parse_amount_over_cap_rejected(monkeypatch) -> None:
    """金额超 50 硬上限 → 拒启（配置误写不靠自律）。"""
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v1": {"amount_usdt": 51.0}}))
    with pytest.raises(ValueError, match="超界"):
        parse_channel_config()


def test_parse_daily_over_cap_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v1": {"max_daily_orders": 501}}))
    with pytest.raises(ValueError, match="日限"):
        parse_channel_config()


def test_parse_bad_exec_price_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v1": {"max_exec_price": 1.5}}))
    with pytest.raises(ValueError, match="护栏"):
        parse_channel_config()


def test_parse_bad_json_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json", "{not-json")
    with pytest.raises(ValueError, match="解析失败"):
        parse_channel_config()


def test_parse_null_amount_rejected(monkeypatch) -> None:
    """amount_usdt=null → 归一为 ValueError（若放 TypeError 穿透 lifespan
    会拖垮整个服务启动，违背“非法配置拒启但不拒服务”契约）。"""
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v1": {"amount_usdt": None}}))
    with pytest.raises(ValueError, match="amount_usdt 非法"):
        parse_channel_config()


def test_parse_non_int_daily_rejected(monkeypatch) -> None:
    """max_daily_orders 非整数（2.5）→ ValueError（防 int() 静默截断成 2）。"""
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v1": {"max_daily_orders": 2.5}}))
    with pytest.raises(ValueError, match="需为整数"):
        parse_channel_config()


def test_channels_registry_shape() -> None:
    """注册表形状：10 通道全集、市场周期/方向/护栏与计划表逐项对齐。"""
    assert set(LIVE_CHANNELS) == {
        "quote_momentum_v1", "quote_contrarian_v1",
        "quote_momentum_v2", "quote_contrarian_v2",
        "x4_v1", "x4_v2",
        "scene_bull_exhaust", "scene_bull_exhaust_confirm",
        "scene_bear_exhaust", "scene_momentum_fade",
    }
    by = {ch: s for ch, s in LIVE_CHANNELS.items()}
    assert by["quote_momentum_v1"].market_period == "5m"
    assert by["quote_momentum_v1"].auto_max_exec == 0.78
    assert by["quote_contrarian_v1"].auto_max_exec == 0.28
    assert by["quote_momentum_v2"].v2_guard == "min_drop"
    assert by["quote_contrarian_v2"].v2_guard == "max_rise"
    assert by["x4_v1"].auto_max_exec == 0.45
    assert by["x4_v2"].auto_max_exec == 0.50
    assert all(by[ch].market_period == "15m" for ch in
               ("scene_bull_exhaust", "scene_bull_exhaust_confirm",
                "scene_bear_exhaust", "scene_momentum_fade"))
    assert by["scene_bull_exhaust"].auto_max_exec == 0.60
    assert by["scene_bull_exhaust_confirm"].auto_max_exec == 0.75
    assert by["scene_bear_exhaust"].auto_max_exec == 0.55
    assert by["scene_momentum_fade"].auto_max_exec == 0.55


def test_scene_pattern_to_channel_mapping() -> None:
    """场景 pattern_type → 通道名映射（fake_breakout 钩子 payload 契约）。"""
    assert scene_pattern_to_channel("bull_exhaust") == "scene_bull_exhaust"
    assert scene_pattern_to_channel("bull_exhaust_confirm") == "scene_bull_exhaust_confirm"
    assert scene_pattern_to_channel("bear_exhaust") == "scene_bear_exhaust"
    assert scene_pattern_to_channel("momentum_fade") == "scene_momentum_fade"
    assert scene_pattern_to_channel("unknown_pattern") is None


# ============================================================
# 组 2：check() 守卫（quote_edge 族区间/开关/None/停机）
# ============================================================

def test_check_none_price_no_fire(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader(), channels=["quote_momentum_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, None) == []


def test_check_disabled_channel_no_fire(monkeypatch) -> None:
    """默认全 OFF：区间命中也不开火（需显式开启）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) == []


def test_check_time_band_guards(monkeypatch) -> None:
    """momentum t∉[90,120) / contrarian t∉[45,60) → 不命中（半开区间）。"""
    t = _make_trader(monkeypatch, _FakeTrader(),
                     channels=["quote_momentum_v1", "quote_contrarian_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 89_999, 0.71) == []
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 120_000, 0.71) == []
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 44_999, 0.20) == []
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 60_000, 0.20) == []


def test_check_quote_band_guards(monkeypatch) -> None:
    """momentum q∉[0.69,0.75) / contrarian q∉[0.15,0.25) → 不命中。"""
    t = _make_trader(monkeypatch, _FakeTrader(),
                     channels=["quote_momentum_v1", "quote_contrarian_v1"])
    ts_m = WINDOW_START + 100_000
    ts_c = WINDOW_START + 50_000
    assert t.check(WINDOW_START, WINDOW_END, ts_m, 0.68) == []
    assert t.check(WINDOW_START, WINDOW_END, ts_m, 0.75) == []
    assert t.check(WINDOW_START, WINDOW_END, ts_c, 0.1499) == []
    assert t.check(WINDOW_START, WINDOW_END, ts_c, 0.25) == []


@pytest.mark.asyncio
async def test_check_multi_channel_same_window(monkeypatch) -> None:
    """多通道同窗独立开火：contrarian（t=50s q=0.20）与 momentum（t=100s
    q=0.71）在同一 5m 窗内各下一单、互不阻塞（废除旧跨版本互斥）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["quote_momentum_v1", "quote_contrarian_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20) \
        == ["quote_contrarian_v1"]
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) \
        == ["quote_momentum_v1"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "quote_contrarian_v1", "quote_momentum_v1"]


@pytest.mark.asyncio
async def test_check_same_channel_once_per_window(monkeypatch) -> None:
    """每通道每窗至多一单：同窗后续采样（区间仍命中）不再开火。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])
    ts = WINDOW_START + 100_000
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) == ["quote_momentum_v1"]
    assert t.check(WINDOW_START, WINDOW_END, ts + 15_000, 0.72) == []
    await _drain(t)
    assert len(fake.calls) == 1
    assert WINDOW_START in t._configs["quote_momentum_v1"].fired


def test_check_stop_rejects(monkeypatch) -> None:
    """stop 后 check 拒绝派生新下单任务（shutdown 窗口期保护）。"""
    t = _make_trader(monkeypatch, _FakeTrader(), channels=["quote_momentum_v1"])
    t._stopped = True
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) == []


# ============================================================
# 组 3：v2 门禁（BTC 实时喂价，阈值复用 V2_PRICE_GUARDS）
# ============================================================

def test_v2_guard_momentum_threshold() -> None:
    """momentum_v2（min_drop ≤ −0.10%）：边界含等号；数据缺失不触发。"""
    g = MultiLiveTrader._pass_live_v2_guard
    assert g("quote_momentum_v2", None, 10000.0) is False          # BTC 缺失
    assert g("quote_momentum_v2", 9980.0, None) is False           # 开盘价缺失
    assert g("quote_momentum_v2", 9980.0, 10000.0) is True         # −0.20% 触发
    assert g("quote_momentum_v2", 9990.0, 10000.0) is True         # 恰 −0.10%（≤ 含边界）
    assert g("quote_momentum_v2", 9991.0, 10000.0) is False        # −0.09% 不触发


def test_v2_guard_contrarian_threshold() -> None:
    """contrarian_v2（max_rise < +0.10%）：边界不含等号（严格小于）。"""
    g = MultiLiveTrader._pass_live_v2_guard
    assert g("quote_contrarian_v2", 10009.0, 10000.0) is True      # +0.09% 触发
    assert g("quote_contrarian_v2", 10010.0, 10000.0) is False     # 恰 +0.10% 不触发
    assert g("quote_contrarian_v2", 10020.0, 10000.0) is False     # +0.20% 不触发


@pytest.mark.asyncio
async def test_v2_gate_blocks_fire_when_missing(monkeypatch) -> None:
    """v2 集成：区间命中但 BTC 门禁数据缺失 → v2 不开火，v1 不受影响。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["quote_momentum_v1", "quote_momentum_v2"])
    ts = WINDOW_START + 100_000
    # 门禁数据缺失：v2 不触发，v1（无门禁）照常开火
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) == ["quote_momentum_v1"]
    # 门禁通过（chg=−0.20%）：新窗口 v1+v2 同窗并行双单
    ws2 = WINDOW_START + 300_000
    assert t.check(ws2, ws2 + 300_000, ws2 + 100_000, 0.71,
                   btc_price=9980.0, window_entry_price=10000.0) \
        == ["quote_momentum_v1", "quote_momentum_v2"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "quote_momentum_v1", "quote_momentum_v1", "quote_momentum_v2"]


@pytest.mark.asyncio
async def test_v2_gate_blocks_fire_when_rising(monkeypatch) -> None:
    """momentum_v2 门禁不满足（BTC 反涨）→ 仅 v1 开火。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["quote_momentum_v1", "quote_momentum_v2"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71,
                   btc_price=10020.0, window_entry_price=10000.0) \
        == ["quote_momentum_v1"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == ["quote_momentum_v1"]


# ============================================================
# 组 4：_fire 路径（日限/防重/失败容错/余额钩子）
# ============================================================

@pytest.mark.asyncio
async def test_fire_success_kwargs(monkeypatch) -> None:
    """开火下单 kwargs 全量断言：版本/护栏/金额/市场周期透传。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == "quote_momentum_v1"
    assert call["window_start"] == WINDOW_START
    assert call["amount_usdt"] == 2.0            # settings 默认
    assert call["max_exec_price"] == 0.78        # momentum 自动推导（0.75+0.03）
    assert call["market_period"] == "5m"
    assert t._configs["quote_momentum_v1"].fire_total == 1


@pytest.mark.asyncio
async def test_fire_daily_cap_blocks(monkeypatch) -> None:
    """日单量护栏：当日 FILLED 达上限 → 不调 trader。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     overrides={"quote_momentum_v1": {"enabled": True,
                                                      "max_daily_orders": 5}})

    async def _full(self, version: str) -> int:
        return 5

    monkeypatch.setattr(MultiLiveTrader, "_count_filled_today", _full)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []
    assert t._configs["quote_momentum_v1"].fire_total == 0


@pytest.mark.asyncio
async def test_fire_daily_cap_per_channel(monkeypatch) -> None:
    """per-channel 日限：A 通道打满不影响 B 通道照常开火。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "quote_contrarian_v1", "quote_momentum_v1"])

    async def _filled(self, version: str) -> int:
        return 100 if version == "quote_contrarian_v1" else 0

    monkeypatch.setattr(MultiLiveTrader, "_count_filled_today", _filled)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20) \
        == ["quote_contrarian_v1"]   # 派生任务但被日限拦截
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) \
        == ["quote_momentum_v1"]     # B 通道不受 A 打满影响
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == ["quote_momentum_v1"]


@pytest.mark.asyncio
async def test_fire_restart_dedup_blocks(monkeypatch) -> None:
    """重启防重：DB 已有本通道本窗尝试记录 → 跳过不再下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])

    async def _has(self, version: str, ws: int) -> bool:
        return True

    monkeypatch.setattr(MultiLiveTrader, "_has_attempt", _has)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_fire_failed_order_no_crash(monkeypatch) -> None:
    """trader 返回 FAILED（护栏弃单等）→ 计入 fire_total、不回填、不抛。"""
    fake = _FakeTrader(result=_fake_order(status="FAILED",
                                          error_message="执行价护栏弃单"))
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert len(fake.calls) == 1
    assert t._configs["quote_momentum_v1"].fire_total == 1


@pytest.mark.asyncio
async def test_fire_trader_none_no_crash(monkeypatch) -> None:
    """trader 返回 None（占位失败/落库异常）→ 不计 fire_total、不抛。"""
    fake = _FakeTrader(result=None)
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert t._configs["quote_momentum_v1"].fire_total == 0


@pytest.mark.asyncio
async def test_fire_balance_hook(monkeypatch) -> None:
    """产生订单（FILLED/FAILED）后触发余额缓存作废钩子；None 不触发。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_momentum_v1"])
    calls: list[None] = []
    t._on_balance_change = lambda: calls.append(None)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t)
    assert calls == [None]


# ============================================================
# 组 5：x4 族（PENDING 轮询 → 决策点下单）
# ============================================================

@pytest.mark.asyncio
async def test_x4_poll_schedules_and_idempotent(monkeypatch) -> None:
    """PENDING 信号调度下单且幂等：重复轮询（同信号 id）不重下。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["x4_v1"])
    now_ms = int(time.time() * 1000)
    target = now_ms - 155_000  # 决策点（target+150s）在 5s 前，容差 30s 内
    _stub_select_db(monkeypatch, [(501, "x4_v1", target)])

    await t._x4_poll_once()
    assert 501 in t._x4_seen
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == "x4_v1"
    assert call["window_start"] == target
    assert call["max_exec_price"] == 0.45
    assert call["market_period"] == "5m"

    await t._x4_poll_once()   # 同一信号再次被捞到：seen 幂等
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_x4_missed_decision_not_chased(monkeypatch) -> None:
    """错过决策点（超 30s 容差）不追单——与影子 PENDING→EXPIRED 语义一致。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["x4_v1"])
    now_ms = int(time.time() * 1000)
    target = now_ms - 200_000  # 决策点在 50s 前（> 30s 容差）
    _stub_select_db(monkeypatch, [(502, "x4_v1", target)])

    await t._x4_poll_once()
    assert 502 in t._x4_seen   # 已标记（不再重试），但不派下单任务
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_x4_fire_respects_disabled(monkeypatch) -> None:
    """决策点到达时通道未启用 → 不下单（关停在途任务前生效）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[])
    now_ms = int(time.time() * 1000)
    target = now_ms - 155_000
    await t._fire_x4(503, "x4_v1", target)
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_x4_fill_links_signal_id(monkeypatch) -> None:
    """x4 下单成交即回填 signal_id（信号已存在，无需等结算）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["x4_v1"])
    now_ms = int(time.time() * 1000)
    target = now_ms - 155_000
    link_calls: list[tuple] = []

    async def _link(version: str, ws: int, sig_id: int) -> None:
        link_calls.append((version, ws, sig_id))

    t._link_signal_id = _link   # 实例级覆盖类级 stub
    await t._fire_x4(504, "x4_v1", target)
    await _drain(t)
    assert len(fake.calls) == 1
    assert link_calls == [("x4_v1", target, 504)]


# ============================================================
# 组 6：场景钩子（fake_breakout fire → 15m 市场下单）
# ============================================================

def _scene_sig(pattern_type: str, side: str, sig_id: int = 42) -> dict:
    return {"id": sig_id, "pattern_type": pattern_type, "side": side,
            "market_start_15m": MARKET_START_15M,
            "market_end_15m": MARKET_START_15M + 900_000}


@pytest.mark.asyncio
async def test_scene_s1_down_15m(monkeypatch) -> None:
    """S1（多头耗尽，side=high）→ 15m 市场押 DOWN，护栏 0.60。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bull_exhaust"])
    t.on_scene_signal(_scene_sig("bull_exhaust", "high"))
    await _drain(t)
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == "scene_bull_exhaust"
    assert call["window_start"] == MARKET_START_15M
    assert call["market_period"] == "15m"
    assert call["scene_signal_id"] == 42
    assert call["max_exec_price"] == 0.60
    assert call["amount_usdt"] == 2.0


@pytest.mark.asyncio
async def test_scene_s2_up_15m(monkeypatch) -> None:
    """S2（空头耗尽，side=low）→ 15m 市场押 UP，护栏 0.55。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bear_exhaust"])
    t.on_scene_signal(_scene_sig("bear_exhaust", "low"))
    await _drain(t)
    call = fake.calls[0]
    assert call["prediction"] == "UP"
    assert call["signal_version"] == "scene_bear_exhaust"
    assert call["market_period"] == "15m"
    assert call["max_exec_price"] == 0.55


@pytest.mark.asyncio
async def test_scene_s5_independent_channel(monkeypatch) -> None:
    """S5 确认信号走独立通道 scene_bull_exhaust_confirm（护栏 0.75）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bull_exhaust_confirm"])
    t.on_scene_signal(_scene_sig("bull_exhaust_confirm", "high", sig_id=77))
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "scene_bull_exhaust_confirm"
    assert call["prediction"] == "DOWN"
    assert call["max_exec_price"] == 0.75
    assert call["scene_signal_id"] == 77


@pytest.mark.asyncio
async def test_scene_disabled_channel_no_fire(monkeypatch) -> None:
    """通道未启用 → 零下单（影子检测继续落表，互不干扰）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[])
    t.on_scene_signal(_scene_sig("bull_exhaust", "high"))
    t.on_scene_signal(_scene_sig("bear_exhaust", "low"))
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_scene_daily_cap_blocks(monkeypatch) -> None:
    """场景通道日限停火：当日 FILLED 满 100 → 不下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bull_exhaust"])

    async def _full(self, version: str) -> int:
        return 100

    monkeypatch.setattr(MultiLiveTrader, "_count_filled_today", _full)
    t.on_scene_signal(_scene_sig("bull_exhaust", "high"))
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_scene_same_market_start_dedup(monkeypatch) -> None:
    """同一 15m 周期重复回调 → 每周期至多一单（内存 fired 防重）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bull_exhaust"])
    t.on_scene_signal(_scene_sig("bull_exhaust", "high", sig_id=1))
    t.on_scene_signal(_scene_sig("bull_exhaust", "high", sig_id=2))
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_scene_unknown_pattern_and_bad_payload_swallowed(monkeypatch) -> None:
    """未知 pattern_type / 缺字段 payload → 静默忽略，绝不阻塞检测循环。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "scene_bull_exhaust", "scene_bear_exhaust"])
    t.on_scene_signal(_scene_sig("unknown_pattern", "high"))
    t.on_scene_signal({"pattern_type": "bull_exhaust"})  # 缺 market_start_15m
    await _drain(t)
    assert fake.calls == []


# ============================================================
# 组 7：set_channel 热调 / status / toggle 端点
# ============================================================

def test_set_channel_toggle_and_hot_adjust(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    t.set_channel("x4_v1", enabled=True, amount_usdt=1.5, max_daily_orders=50)
    cfg = t._configs["x4_v1"]
    assert cfg.enabled and cfg.amount_usdt == 1.5 and cfg.max_daily_orders == 50
    # 其余通道不受影响
    assert t._configs["quote_momentum_v1"].enabled is False


def test_set_channel_unknown_rejected(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    with pytest.raises(ValueError, match="未知通道"):
        t.set_channel("nonexistent", enabled=True)


def test_set_channel_amount_over_cap_rejected(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    with pytest.raises(ValueError, match="超界"):
        t.set_channel("x4_v1", amount_usdt=99.0)


def test_set_channel_daily_over_cap_rejected(monkeypatch) -> None:
    t = _make_trader(monkeypatch, _FakeTrader())
    with pytest.raises(ValueError, match="日限"):
        t.set_channel("x4_v1", max_daily_orders=0)


def test_status_shape(monkeypatch) -> None:
    """status：10 通道全量、enabled_any/defaults/amount_cap、单通道字段。"""
    t = _make_trader(monkeypatch, _FakeTrader(), channels=["quote_contrarian_v1"])
    s = t.status()
    assert s["enabled_any"] is True
    assert s["amount_cap"] == 50
    assert s["defaults"]["amount_usdt"] == 2.0
    assert s["defaults"]["max_daily_orders"] == 100
    assert len(s["channels"]) == 10
    by = {c["channel"]: c for c in s["channels"]}
    c = by["quote_contrarian_v1"]
    assert c["enabled"] is True and c["enabled_at_startup"] is True
    assert c["amount_usdt"] == 2.0 and c["max_daily_orders"] == 100
    assert c["max_exec_price"] == 0.28 and c["auto_max_exec"] == 0.28
    assert c["market_period"] == "5m" and c["fire_total"] == 0
    assert by["scene_bull_exhaust"]["market_period"] == "15m"


@pytest.mark.asyncio
async def test_status_async_filled_today(monkeypatch) -> None:
    """status_async：GROUP BY 结果注入每通道 filled_today。"""
    t = _make_trader(monkeypatch, _FakeTrader(), channels=["quote_momentum_v1"])
    _stub_select_db(monkeypatch, [("quote_momentum_v1", 3)])
    s = await t.status_async()
    by = {c["channel"]: c for c in s["channels"]}
    assert by["quote_momentum_v1"]["filled_today"] == 3
    assert by["quote_contrarian_v1"]["filled_today"] == 0


@pytest.mark.asyncio
async def test_toggle_endpoint_channel_flow(monkeypatch) -> None:
    """POST /api/live/toggle：开通道 + 金额热调，返回全通道 status。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    monkeypatch.setattr(m, "multi_live_trader", t)

    async def _status_async(self):
        return self.status()

    monkeypatch.setattr(MultiLiveTrader, "status_async", _status_async)

    async def _persist(self, channel):
        return None  # 免碰真实 DB（持久化行为见组 10 专项用例）

    monkeypatch.setattr(MultiLiveTrader, "persist_channel", _persist)
    out = await m.live_toggle(
        ToggleLiveRequest(channel="x4_v1", enabled=True, amount_usdt=1.5),
        _=None)
    assert "error" not in out
    assert t._configs["x4_v1"].enabled is True
    assert t._configs["x4_v1"].amount_usdt == 1.5
    assert out["status"]["enabled_any"] is True
    assert "开启" in out["message"]


@pytest.mark.asyncio
async def test_toggle_endpoint_bad_channel_and_amount(monkeypatch) -> None:
    """非法通道/金额 → error 早退，状态不变（原子：改不过就不生效）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    monkeypatch.setattr(m, "multi_live_trader", t)

    async def _persist(self, channel):
        return None

    monkeypatch.setattr(MultiLiveTrader, "persist_channel", _persist)

    out = await m.live_toggle(
        ToggleLiveRequest(channel="nonexistent", enabled=True), _=None)
    assert "error" in out
    out = await m.live_toggle(
        ToggleLiveRequest(channel="x4_v1", enabled=True, amount_usdt=99.0),
        _=None)
    assert "error" in out
    assert t._configs["x4_v1"].enabled is False   # 金额校验失败未被置位


# ============================================================
# 组 10：DB 覆盖层（toggle 持久化，重启不丢设定）
# ============================================================

class _OverrideSession:
    """live_channel_overrides 会话桩：execute 回 rows / merge 记录 / commit 放行。"""

    def __init__(self, rows: list) -> None:
        self._rows = rows
        self.merged: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        res = MagicMock()
        res.scalars.return_value.all.return_value = self._rows
        return res

    async def merge(self, row):
        self.merged.append(row)
        return row

    async def commit(self):
        return None


def _stub_override_db(monkeypatch, rows: list) -> _OverrideSession:
    sess = _OverrideSession(rows)
    monkeypatch.setattr(milt, "async_session_factory", lambda: sess)
    return sess


@pytest.mark.asyncio
async def test_apply_db_overrides_applies_and_skips_bad(monkeypatch) -> None:
    """DB 覆盖层：合法行覆盖 env 设定；未知通道行跳过不崩启动；
    enabled_at_startup 重快照反映覆盖后的最终启动态。"""
    t = _make_trader(monkeypatch, _FakeTrader(),
                     channels=["quote_contrarian_v1"])
    rows = [
        SimpleNamespace(channel="x4_v1", enabled=True,
                        amount_usdt=1.5, max_daily_orders=50),
        SimpleNamespace(channel="ghost_channel", enabled=True,
                        amount_usdt=2.0, max_daily_orders=10),
        SimpleNamespace(channel="quote_contrarian_v1", enabled=False,
                        amount_usdt=2.0, max_daily_orders=100),
    ]
    _stub_override_db(monkeypatch, rows)

    applied = await t.apply_db_overrides()

    assert applied == ["x4_v1", "quote_contrarian_v1"]   # ghost 被跳过
    assert t._configs["x4_v1"].enabled is True
    assert t._configs["x4_v1"].amount_usdt == 1.5
    assert t._configs["x4_v1"].max_daily_orders == 50
    assert t._configs["quote_contrarian_v1"].enabled is False   # DB 胜过 env 层
    assert t._enabled_at_startup["quote_contrarian_v1"] is False


@pytest.mark.asyncio
async def test_persist_channel_writes_snapshot(monkeypatch) -> None:
    """持久化：upsert 该通道当前完整配置快照（重启后恢复的正是最后设定）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    sess = _stub_override_db(monkeypatch, [])

    t.set_channel("x4_v2", enabled=True, amount_usdt=3.0, max_daily_orders=7)
    await t.persist_channel("x4_v2")

    assert len(sess.merged) == 1
    row = sess.merged[0]
    assert row.channel == "x4_v2" and row.enabled is True
    assert row.amount_usdt == 3.0 and row.max_daily_orders == 7
    assert row.updated_at is not None


@pytest.mark.asyncio
async def test_toggle_endpoint_persists(monkeypatch) -> None:
    """toggle 端点：运行时生效成功后持久化该通道（重启不丢设定链路）。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    monkeypatch.setattr(m, "multi_live_trader", t)
    persisted: list[str] = []

    async def _persist(self, channel):
        persisted.append(channel)

    async def _status_async(self):
        return self.status()

    monkeypatch.setattr(MultiLiveTrader, "persist_channel", _persist)
    monkeypatch.setattr(MultiLiveTrader, "status_async", _status_async)
    out = await m.live_toggle(
        ToggleLiveRequest(channel="x4_v1", enabled=True), _=None)
    assert "error" not in out
    assert persisted == ["x4_v1"]


# ============================================================
# 组 8：execute_signal_trade（15m token 分流 + 5m 行为迁移）
# ============================================================

def _make_real_trader(monkeypatch, with_15m: bool = True) -> BinancePredictionTrader:
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xWALLET"
    trader._wallet_id = "WID"

    async def _list():
        trader._down_token_id = "TOKEN-DOWN"
        if with_15m:
            # 新锚定机制：按 startDate 建表，下单精确匹配 window_start
            trader._15m_markets[WINDOW_START] = {
                "end_date": WINDOW_START + 900_000,
                "up_token": "TOKEN-15M-UP", "down_token": "TOKEN-15M-DOWN",
                "up_price": 0.45, "down_price": 0.55,
            }
        return []

    monkeypatch.setattr(trader, "list_markets", _list)
    return trader


def _pending_order() -> dict:
    return _fake_order(status="PENDING")


@pytest.mark.asyncio
async def test_signal_trade_15m_token_and_scene_id(monkeypatch) -> None:
    """15m 市场：token 分流到 _15m_down_token_id，market_period/
    scene_signal_id 与占位同事务透传（结算分流依据）。"""
    trader = _make_real_trader(monkeypatch)
    reserve_calls: list[tuple] = []
    quote_tokens: list[str] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        reserve_calls.append((market_period, scene_signal_id))
        return _pending_order()

    async def _update(order, status, **kwargs):
        return {**order, "status": status, **kwargs}

    async def _quote(token_id, side, amount_usdt=None):
        quote_tokens.append(token_id)
        return {"averagePrice": 0.55, "amountIn": "5", "amountOut": "9",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        return {"orderId": "ORD-15M"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START,
        max_exec_price=0.60, market_period="15m", scene_signal_id=42)
    assert order["status"] == "FILLED"
    assert quote_tokens == ["TOKEN-15M-DOWN"]       # 15m token 分流
    assert reserve_calls == [("15m", 42)]            # 占位透传（落库依据）


@pytest.mark.asyncio
async def test_signal_trade_15m_token_missing_failed(monkeypatch) -> None:
    """15m 周期未在市场列表中找到（锚定守卫）→ FAILED 落库，不取报价。"""
    trader = _make_real_trader(monkeypatch, with_15m=False)
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(*a, **k):
        raise AssertionError("锚定拒单不应走到取报价")

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bear_exhaust", WINDOW_START,
        max_exec_price=0.55, market_period="15m", scene_signal_id=7)
    assert order["status"] == "FAILED"
    assert "未找到 15m 市场 DOWN 方向的 token" in str(updates[0][1]["error_message"])


@pytest.mark.asyncio
async def test_signal_trade_15m_anchor_picks_right_cycle(monkeypatch) -> None:
    """多周期列表（当前 + 未来）：token 必须取自 window_start 对应周期，
    而非列表里更晚的市场（旧"最后一个覆盖"行为的回归用例）。"""
    trader = _make_real_trader(monkeypatch, with_15m=False)
    future = WINDOW_START + 900_000
    quote_tokens: list[str] = []

    async def _list():
        # 目标周期先入表，未来周期后入表（旧行为会被后者覆盖）
        trader._15m_markets[WINDOW_START] = {
            "end_date": future, "up_token": "T-UP-N", "down_token": "T-DOWN-N",
            "up_price": 0.5, "down_price": 0.5}
        trader._15m_markets[future] = {
            "end_date": future + 900_000, "up_token": "T-UP-NEXT",
            "down_token": "T-DOWN-NEXT", "up_price": 0.5, "down_price": 0.5}
        return []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        return {**order, "status": status, **kwargs}

    async def _quote(token_id, side, amount_usdt=None):
        quote_tokens.append(token_id)
        return {"averagePrice": 0.55, "amountIn": "5", "amountOut": "9",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        return {"orderId": "ORD-ANCHOR"}

    monkeypatch.setattr(trader, "list_markets", _list)
    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START,
        max_exec_price=0.60, market_period="15m", scene_signal_id=42)
    assert order["status"] == "FILLED"
    assert quote_tokens == ["T-DOWN-N"]          # 取本单周期，非未来周期


def test_parse_15m_entry() -> None:
    """15m 市场解析：token/报价提取 + startDate 缺失/非法拒入表。"""
    market = {
        "startDate": WINDOW_START, "endDate": WINDOW_START + 900_000,
        "markets": [{"outcomes": [
            {"name": "Up", "tokenId": "T-U", "price": "0.4"},
            {"name": "Down", "tokenId": "T-D", "price": "0.6"},
        ]}],
    }
    start, entry = BinancePredictionTrader._parse_15m_entry(market)
    assert start == WINDOW_START
    assert entry["up_token"] == "T-U" and entry["down_token"] == "T-D"
    assert entry["up_price"] == 0.4 and entry["down_price"] == 0.6
    assert entry["end_date"] == WINDOW_START + 900_000
    # startDate 缺失/非法 → None（不入表，无法锚定的市场不参与下单）
    assert BinancePredictionTrader._parse_15m_entry({"endDate": 1}) is None
    assert BinancePredictionTrader._parse_15m_entry(
        {"startDate": "not-a-ts", "endDate": 1}) is None


@pytest.mark.asyncio
async def test_signal_trade_price_guard_rejects(monkeypatch) -> None:
    """报价均价 0.82 > 上限 0.78 → 弃单，占位更新 FAILED，不 place_order。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

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
    assert order["status"] == "FAILED"
    assert updates[0][0] == "FAILED"
    assert "执行价护栏" in updates[0][1]["error_message"]


@pytest.mark.asyncio
async def test_signal_trade_success_dynamic_slippage(monkeypatch) -> None:
    """报价 0.71 ≤ 上限 0.78 → 成交；滑点按护栏价收紧至 985bps。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    slippage_seen: list[int] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, "order_id": kwargs.get("order_id")}

    async def _quote(_token, _side, amount_usdt=None):
        assert amount_usdt == 5.0
        return {"averagePrice": 0.71, "amountIn": "5", "amountOut": "7",
                "quoteId": "Q1"}

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


@pytest.mark.asyncio
async def test_signal_trade_duplicate_window_skips(monkeypatch) -> None:
    """占位失败（同窗已有订单）→ 花钱前拒绝，不取报价不下单。"""
    trader = _make_real_trader(monkeypatch)

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return None

    async def _quote(*a, **k):
        raise AssertionError("重复窗口不应走到取报价")

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "get_quote", _quote)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order is None


# ============================================================
# 组 9：15m 结算分流（trade_settler._settle_scene_row，防错口径结算）
# ============================================================

WS_15M = 1_787_400_000_000  # 15m 周期起点（与 5m 窗起点数值重合是常态）


def _scene_row(**over) -> SimpleNamespace:
    """15m 场景订单待结算行替身（FILLED + 2U + scene_signal_id）。"""
    base = dict(
        id=21, status="FILLED", settled_at=None, window_start=WS_15M,
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        direction="DOWN", amount_in=str(2 * 10 ** 18),  # 2 USDT
        quote_json={"averagePrice": 0.6},
        market_period="15m", scene_signal_id=7,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _scene_signal(outcome: str | None, deadline_ms: int | None = None,
                  settle_price: float = 43250.0) -> SimpleNamespace:
    """FakeBreakoutSignal 结算字段替身（只读三字段）。"""
    return SimpleNamespace(settle_outcome=outcome, settle_btc_price=settle_price,
                           settle_deadline=deadline_ms)


class _SceneDb:
    """15m 结算 DB 桩：execute 路由 Update/订单查询 + get(FakeBreakoutSignal)。"""

    def __init__(self, rows, signal, update_rowcount: int = 1) -> None:
        self.updates: list[Update] = []
        self.get_pks: list[int] = []
        self._signal = signal
        self._update_rowcount = update_rowcount
        orders_res = MagicMock()
        orders_res.scalars.return_value.all.return_value = rows
        self._orders_res = orders_res

    async def execute(self, stmt):
        if isinstance(stmt, Update):
            self.updates.append(stmt)
            res = MagicMock()
            res.rowcount = self._update_rowcount
            return res
        assert isinstance(stmt, Select), f"意外语句类型: {stmt!r}"
        entity = stmt.column_descriptions[0]["entity"]
        assert entity is TradeOrderModel, f"未知查询实体: {entity}"
        return self._orders_res

    async def get(self, model, pk):
        assert model is FakeBreakoutSignal
        self.get_pks.append(pk)
        return self._signal

    async def commit(self):
        return None


def _stub_scene_db(monkeypatch, db: _SceneDb) -> None:
    @asynccontextmanager
    async def _factory():
        yield db

    monkeypatch.setattr(ts_mod, "async_session_factory", _factory)


def _params(update_stmt) -> dict:
    return update_stmt.compile().params


@pytest.mark.asyncio
async def test_settle_scene_win(monkeypatch) -> None:
    """15m 赢：direction==settle_outcome → win=True、pnl=2/0.6−2。"""
    db = _SceneDb([_scene_row(direction="DOWN")], _scene_signal("DOWN"))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    assert db.get_pks == [7]   # 回读 scene_signal_id
    p = _params(db.updates[0])
    assert p["settle_outcome"] == "DOWN"
    assert p["win"] is True
    assert p["pnl"] == pytest.approx(2 / 0.6 - 2)
    assert p["settle_price"] == pytest.approx(43250.0)


@pytest.mark.asyncio
async def test_settle_scene_lose(monkeypatch) -> None:
    """15m 输：direction≠settle_outcome → win=False、pnl=−amount。"""
    db = _SceneDb([_scene_row(direction="DOWN")], _scene_signal("UP"))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "UP"
    assert p["win"] is False
    assert p["pnl"] == pytest.approx(-2.0)


@pytest.mark.asyncio
async def test_settle_scene_noise(monkeypatch) -> None:
    """NOISE：即时结算 win=None/pnl=0/settle_price=信号结算价（与 5m 同口径），
    不再空扫重试直到 settle_deadline+24h 被误记 EXPIRED。"""
    db = _SceneDb([_scene_row(direction="DOWN")], _scene_signal("NOISE"))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "NOISE"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] == pytest.approx(43250.0)


@pytest.mark.asyncio
async def test_settle_scene_pending_retry(monkeypatch) -> None:
    """信号 PENDING 且 settle_deadline 未超 24h → 不写 UPDATE，下轮重试
    （15m 检测器结算可能迟到，与 5m 归档迟到同口径）。"""
    now_ms = int(time.time() * 1000)
    db = _SceneDb([_scene_row()],
                  _scene_signal(None, deadline_ms=now_ms + 3_600_000))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert db.updates == []


@pytest.mark.asyncio
async def test_settle_scene_expired(monkeypatch) -> None:
    """信号 PENDING 超 deadline+24h → EXPIRED 出清（win=None/pnl=0）。"""
    now_ms = int(time.time() * 1000)
    db = _SceneDb([_scene_row()],
                  _scene_signal(None, deadline_ms=now_ms - 25 * 3_600_000))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    p = _params(db.updates[0])
    assert p["settle_outcome"] == "EXPIRED"
    assert p["win"] is None
    assert p["pnl"] == 0.0
    assert p["settle_price"] is None


@pytest.mark.asyncio
async def test_settle_scene_missing_signal_id_expired(monkeypatch) -> None:
    """15m 行缺 scene_signal_id（数据异常）→ 直接 EXPIRED 出清
    （不查窗口——防错口径结算入口分流的 or 条件）。"""
    db = _SceneDb([_scene_row(scene_signal_id=None)], _scene_signal("DOWN"))
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1

    assert db.get_pks == []   # 无 id 可查
    p = _params(db.updates[0])
    assert p["settle_outcome"] == "EXPIRED"


@pytest.mark.asyncio
async def test_settle_scene_signal_row_missing_expired(monkeypatch) -> None:
    """信号行被运维 TRUNCATE/删除 → EXPIRED 出清不计统计。"""
    db = _SceneDb([_scene_row()], None)
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 1
    assert _params(db.updates[0])["settle_outcome"] == "EXPIRED"


@pytest.mark.asyncio
async def test_settle_scene_idempotent_guard(monkeypatch) -> None:
    """UPDATE rowcount=0（并发已结算）→ 幂等守卫生效，不计入 settled。"""
    db = _SceneDb([_scene_row()], _scene_signal("DOWN"), update_rowcount=0)
    _stub_scene_db(monkeypatch, db)

    assert await TradeSettler().poll_once() == 0
    assert len(db.updates) == 1
