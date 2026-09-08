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

import binance_predict.services.live_channels as lc
import binance_predict.services.multi_live_trader as milt
import binance_predict.services.prediction_trading as pt
import binance_predict.services.trade_settler as ts_mod
from binance_predict.config.settings import settings
from binance_predict.db.models import FakeBreakoutSignal, TradeOrderModel
from binance_predict.services.live_channels import (
    LIVE_CHANNELS,
    RETIRED_CHANNEL_SPECS,
    RETIRED_CHANNELS,
    RETIRED_SAME_WINDOW_EXCLUSIVE,
    ChannelConfig,
    parse_channel_config,
    resolve_max_exec,
    scene_pattern_to_channel,
)
from binance_predict.services.multi_live_trader import MultiLiveTrader
from binance_predict.services.prediction_trading import (
    CONFIRM_ATTEMPTS,
    CONFIRM_DELAY_S,
    CONFIRM_DEEP_ATTEMPTS,
    CONFIRM_DEEP_DELAY_S,
    LOW_BALANCE_ALERT_USDT,
    RETRYABLE_ORDER_STATUSES,
    TERMINAL_ORDER_STATUSES,
    BinancePredictionTrader,
)
from binance_predict.services.quote_edge_detector import (
    QUOTE_EDGE_RULES,
    REGIME_GUARDS,
)
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


def _with_retired(monkeypatch):
    """把已退役通道 spec 注回注册表（机制类用例的 fixture 源）。

    2026-09-04 退役 8 通道、2026-09-06 影子 promote 5 通道 + x4_v3 并行注册后
    LIVE_CHANNELS 共 13 个在线通道，quote_edge 族仅存 quote_contrarian_v2 一个
    ——「多通道同窗独立开火」/「v2 门禁双模式」/「v3 环境门禁」/「regime」/
    「streak」/「同窗互斥」这些机制无法只用在线通道构造。机制代码仍在生产
    （由 ChannelSpec 标志位驱动），故用退役 spec 当 fixture 继续覆盖；
    生产注册表现 13 通道。
    """
    merged = {**RETIRED_CHANNEL_SPECS, **LIVE_CHANNELS}
    monkeypatch.setattr(lc, "LIVE_CHANNELS", merged)
    monkeypatch.setattr(milt, "LIVE_CHANNELS", merged)
    monkeypatch.setattr(lc, "SAME_WINDOW_EXCLUSIVE",
                        lc.SAME_WINDOW_EXCLUSIVE + RETIRED_SAME_WINDOW_EXCLUSIVE)
    return merged


def _make_trader(monkeypatch, trader: _FakeTrader,
                 channels: list[str] | None = None,
                 overrides: dict | None = None) -> MultiLiveTrader:
    """构造执行器：live_channels_json 启用指定通道；DB 访问全部置空。"""
    _with_retired(monkeypatch)
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
    """默认：全 23 在线通道 OFF、金额/日限取全局默认（用户拍板 2U / 100 单）。"""
    monkeypatch.setattr(settings, "live_default_amount_usdt", 2.0)
    monkeypatch.setattr(settings, "live_default_max_daily_orders", 100)
    monkeypatch.setattr(settings, "live_channels_json", "")
    cfgs = parse_channel_config()
    assert len(cfgs) == 23  # G0/G1/G3+G4+G7 族 6 变体 = 10 firsthit + 其他 13
    assert all(not c.enabled for c in cfgs.values())
    assert all(c.amount_usdt == 2.0 for c in cfgs.values())
    assert all(c.max_daily_orders == 100 for c in cfgs.values())
    assert all(c.max_exec_price is None for c in cfgs.values())  # 缺省回落 auto


def test_parse_overrides_applied(monkeypatch) -> None:
    """JSON 覆盖：enabled/金额/日限/护栏逐通道生效，未提及通道保持默认。"""
    monkeypatch.setattr(settings, "live_channels_json", json.dumps({
        "x4_v2": {"enabled": True, "amount_usdt": 1.0,
                  "max_daily_orders": 50, "max_exec_price": 0.48},
        "quote_contrarian_v2": {"enabled": True},
    }))
    cfgs = parse_channel_config()
    x4 = cfgs["x4_v2"]
    assert x4.enabled and x4.amount_usdt == 1.0
    assert x4.max_daily_orders == 50 and x4.max_exec_price == 0.48
    assert cfgs["quote_contrarian_v2"].enabled is True
    assert cfgs["scene_bull_exhaust"].enabled is False   # 未提及通道不受影响


def test_parse_unknown_channel_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"nonexistent": {"enabled": True}}))
    with pytest.raises(ValueError, match="未知通道"):
        parse_channel_config()


def test_parse_retired_channel_skipped_not_raise(monkeypatch) -> None:
    """退役通道在 LIVE_CHANNELS_JSON 里残留 → 告警跳过，不 raise（部署阻断防护）。

    若这里 raise，main 装配只捕 ValueError 后会把 multi_live_trader 置 None，
    整个实盘（含 7 个保留通道）静默全停——生产 env JSON 内容不可见，退役前
    必须先排掉这个雷。未知拼写错误仍必须 raise（不丢防护）。
    """
    monkeypatch.setattr(settings, "live_channels_json", json.dumps({
        "x4_v1": {"enabled": True, "amount_usdt": 5.0},   # 退役：忽略
        "quote_momentum_v3": {"enabled": True},           # 退役：忽略
        "x4_v2": {"enabled": True, "amount_usdt": 3.0},   # 在线：正常生效
    }))
    cfgs = parse_channel_config()
    assert set(cfgs) == set(LIVE_CHANNELS)          # 退役通道不入配置表
    assert "x4_v1" not in cfgs and "quote_momentum_v3" not in cfgs
    assert cfgs["x4_v2"].enabled is True and cfgs["x4_v2"].amount_usdt == 3.0


def test_parse_amount_over_cap_rejected(monkeypatch) -> None:
    """金额超 50 硬上限 → 拒启（配置误写不靠自律）。"""
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v2": {"amount_usdt": 51.0}}))
    with pytest.raises(ValueError, match="超界"):
        parse_channel_config()


def test_parse_daily_over_cap_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v2": {"max_daily_orders": 501}}))
    with pytest.raises(ValueError, match="日限"):
        parse_channel_config()


def test_parse_bad_exec_price_rejected(monkeypatch) -> None:
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v2": {"max_exec_price": 1.5}}))
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
                        json.dumps({"x4_v2": {"amount_usdt": None}}))
    with pytest.raises(ValueError, match="amount_usdt 非法"):
        parse_channel_config()


def test_parse_non_int_daily_rejected(monkeypatch) -> None:
    """max_daily_orders 非整数（2.5）→ ValueError（防 int() 静默截断成 2）。"""
    monkeypatch.setattr(settings, "live_channels_json",
                        json.dumps({"x4_v2": {"max_daily_orders": 2.5}}))
    with pytest.raises(ValueError, match="需为整数"):
        parse_channel_config()


def test_channels_registry_shape() -> None:
    """注册表形状（2026-09-08 firsthit G4+G7 系注册后）：23 在线 + 8 退役，两者不交。"""
    assert set(LIVE_CHANNELS) == {
        "quote_contrarian_v2",
        "x4_v2",
        "x4_v3",
        "scene_bull_exhaust", "scene_bull_exhaust_confirm",
        "scene_bear_exhaust", "scene_momentum_fade",
        "s5_deep_z20_v1",
        # 2026-09-06 影子 promote 三族（默认全 OFF，面板/配置开启）
        "s2_cond_t4_v1", "s2_cond_t5d_v1",
        "nb_smaslope_5m_v1",
        "absorption_follow_td120_v1", "absorption_follow_td150_v1",
        # 2026-09-07 firsthit 三通道（默认全 OFF，用户确认独立下单）
        "firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1",
        # 2026-09-08 firsthit G4 交互门（G1∩G3，默认 OFF；⚠非独立暴露，见 live_channels 注释）
        "firsthit_down_g4_v1",
        # 2026-09-08 firsthit G7 系列六通道（默认全 OFF，用户确认独立下单）
        "firsthit_down_g7_v1", "g7_streak_v1", "g7_wick20_v1",
        "g7_strict_v1", "g7_q05_v1", "g7_t270_v1",
    }
    assert set(RETIRED_CHANNELS) == {
        "quote_momentum_v1", "quote_contrarian_v1",
        "quote_momentum_v2",
        "quote_contrarian_v3a", "quote_contrarian_v3b",
        "quote_contrarian_v4", "quote_momentum_v3",
        "x4_v1",
    }
    assert not (set(LIVE_CHANNELS) & set(RETIRED_CHANNELS))
    by = {**RETIRED_CHANNEL_SPECS, **LIVE_CHANNELS}
    # 在线通道：市场周期/方向/护栏逐项对齐
    assert by["quote_contrarian_v2"].market_period == "5m"
    assert by["quote_contrarian_v2"].v2_guard == "max_rise"
    assert by["quote_contrarian_v2"].auto_max_exec == 0.28
    assert by["x4_v2"].auto_max_exec == 0.50
    assert all(by[ch].market_period == "15m" for ch in
               ("scene_bull_exhaust", "scene_bull_exhaust_confirm",
                "scene_bear_exhaust", "scene_momentum_fade"))
    assert by["scene_bull_exhaust"].auto_max_exec == 0.70
    assert by["scene_bull_exhaust_confirm"].auto_max_exec == 0.75
    assert by["scene_bear_exhaust"].auto_max_exec == 0.65
    assert by["scene_momentum_fade"].auto_max_exec == 0.55
    # S5 深档：scene 族 15m 押 DOWN，护栏 0.88（盈亏平衡 0.895 略下方）
    s5d = by["s5_deep_z20_v1"]
    assert s5d.family == "scene" and s5d.market_period == "15m"
    assert s5d.direction == "DOWN" and s5d.auto_max_exec == 0.88
    # promote 三族（2026-09-06）：周期/族/护栏逐项对齐决定表（护栏 = 价-only
    # 胜率 wr×0.98 平衡价略下方；absorption spec 基准 UP，实际方向由 BTC 位移符号
    # 动态决定）
    for ch, period, fam, guard in (
        ("s2_cond_t4_v1", "15m", "s2_cond", 0.38),
        ("s2_cond_t5d_v1", "15m", "s2_cond", 0.44),
        ("nb_smaslope_5m_v1", "5m", "nextbar", 0.46),
        ("absorption_follow_td120_v1", "5m", "absorption", 0.78),
        ("absorption_follow_td150_v1", "5m", "absorption", 0.86),
    ):
        spec = by[ch]
        assert spec.market_period == period and spec.family == fam
        assert spec.auto_max_exec == guard
    # 在线通道一律不带门禁标志（v3_env/regime/streak 仅退役 spec 使用；
    # 机制代码保留，未来新通道可再启用）
    assert all(not s.v3_env and not s.regime_gate and not s.streak_gate
               for s in LIVE_CHANNELS.values())
    assert all(s.v2_guard in (None, "max_rise") for s in LIVE_CHANNELS.values())
    # 同窗互斥组：S5 组（既有）+ promote 新增 s2_cond/absorption 双变体两组；
    # momentum 组随三成员全退役而移除
    from binance_predict.services.live_channels import exclusive_group
    g_s5 = exclusive_group("s5_deep_z20_v1")
    assert g_s5 == frozenset({"scene_bull_exhaust_confirm", "s5_deep_z20_v1"})
    assert exclusive_group("s2_cond_t4_v1") == frozenset(
        {"s2_cond_t4_v1", "s2_cond_t5d_v1"})
    assert exclusive_group("absorption_follow_td150_v1") == frozenset(
        {"absorption_follow_td120_v1", "absorption_follow_td150_v1"})
    assert exclusive_group("scene_bull_exhaust") is None
    assert exclusive_group("quote_contrarian_v2") is None
    assert exclusive_group("quote_momentum_v3") is None   # 退役组不参与生产判定
    assert RETIRED_SAME_WINDOW_EXCLUSIVE == (
        frozenset({"quote_momentum_v1", "quote_momentum_v2", "quote_momentum_v3"}),)


def test_retired_channel_specs_match_retired_versions() -> None:
    """两处退役名单必须一致（live_channels.RETIRED_CHANNELS ⊆
    shadow_version_gate.RETIRED_VERSIONS）：通道退役了影子版本也必须退役，
    否则出现「影子还在采、实盘不可交易」的半退役态。
    反向不要求包含：late_night_contrarian_v1 / hm_touch_down_v1/v2 是纯影子
    版本，本就无实盘通道。
    """
    from binance_predict.services.shadow_version_gate import RETIRED_VERSIONS
    assert RETIRED_CHANNELS <= RETIRED_VERSIONS
    assert RETIRED_VERSIONS - RETIRED_CHANNELS == {
        "late_night_contrarian_v1", "hm_touch_down_v1", "hm_touch_down_v2"}


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
    """v2 集成：区间命中但 BTC 门禁数据缺失 → v2 不开火，v1 不受影响；
    门禁通过的同窗 v1+v2 双命中 → 同窗互斥至多一单成交（v1 先成交占坑）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["quote_momentum_v1", "quote_momentum_v2"])
    ts = WINDOW_START + 100_000
    # 门禁数据缺失：v2 不触发，v1（无门禁）照常开火
    assert t.check(WINDOW_START, WINDOW_END, ts, 0.71) == ["quote_momentum_v1"]
    # 门禁通过（chg=−0.20%）：新窗口 v1+v2 同窗双命中（同步预检都过），
    # 但互斥组内至多一单成交 → 仅 v1 成交，v2 弃单
    ws2 = WINDOW_START + 300_000
    assert t.check(ws2, ws2 + 300_000, ws2 + 100_000, 0.71,
                   btc_price=9980.0, window_entry_price=10000.0) \
        == ["quote_momentum_v1", "quote_momentum_v2"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "quote_momentum_v1", "quote_momentum_v1"]


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
# 组 3b：v3 环境门禁通道（v3a/v3b，异步环境核验后才下单）
# ============================================================

def test_v3_blocks_when_v2_gate_missing(monkeypatch) -> None:
    """v3：BTC 喂价缺失 → v2 价格门禁同步拦截，不派生环境核验任务。"""
    t = _make_trader(monkeypatch, _FakeTrader(),
                     channels=["quote_contrarian_v3a", "quote_contrarian_v3b"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20) == []


def test_v3_blocks_when_price_rising(monkeypatch) -> None:
    """v3：触发时 BTC 已涨 ≥0.10%（max_rise 不含边界）→ 不开火。"""
    t = _make_trader(monkeypatch, _FakeTrader(),
                     channels=["quote_contrarian_v3a"])
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20,
                   btc_price=10010.0, window_entry_price=10000.0) == []


@pytest.mark.asyncio
async def test_v3_fire_when_env_gate_passes(monkeypatch) -> None:
    """v3：区间命中 + 价格门禁过 + 环境核验过 → 真单押 DOWN，护栏 0.28。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_contrarian_v3a"])

    async def _env(self, channel, ws, ts, btc, curve=None):
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_v3_env", _env)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20,
                   btc_price=9995.0, window_entry_price=10000.0) \
        == ["quote_contrarian_v3a"]
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "quote_contrarian_v3a"
    assert call["prediction"] == "DOWN"
    assert call["max_exec_price"] == 0.28
    assert call["market_period"] == "5m"


@pytest.mark.asyncio
async def test_v3_env_gate_failure_aborts(monkeypatch) -> None:
    """v3：环境核验未过 → 弃单；fired 占位保留，同窗不重派。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_contrarian_v3b"])

    async def _env(self, channel, ws, ts, btc, curve=None):
        return False, "env_guard_reject"

    monkeypatch.setattr(MultiLiveTrader, "_check_v3_env", _env)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20,
                   btc_price=9995.0, window_entry_price=10000.0) \
        == ["quote_contrarian_v3b"]   # 同步预检已过，派生核验任务
    await _drain(t)
    assert fake.calls == []            # 核验未过 → 弃单
    # 同窗后续采样不再派生（fired 占位已生效）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 55_000, 0.21,
                   btc_price=9995.0, window_entry_price=10000.0) == []


@pytest.mark.asyncio
async def test_v3_prev_not_archived_retry(monkeypatch) -> None:
    """v3：前窗未归档（归档相位竞态）→ 延迟重查一次；第二次仍未过 → 弃单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_contrarian_v3a"])
    calls: list[str] = []

    async def _env(self, channel, ws, ts, btc, curve=None):
        calls.append(channel)
        return False, "prev_not_archived"

    monkeypatch.setattr(MultiLiveTrader, "_check_v3_env", _env)
    monkeypatch.setattr(milt, "V3_PREV_RETRY_DELAY_S", 0.0)  # 免等待
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20,
                   btc_price=9995.0, window_entry_price=10000.0) \
        == ["quote_contrarian_v3a"]
    await _drain(t)
    assert len(calls) == 2             # 首查 + 延迟重查一次
    assert fake.calls == []            # 重查仍未归档 → 弃单


@pytest.mark.asyncio
async def test_v3_prev_retry_then_fire(monkeypatch) -> None:
    """v3：首查 prev_not_archived → 重查通过 → 下单（归档追上的正常路径）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["quote_contrarian_v3a"])
    attempts = {"n": 0}

    async def _env(self, channel, ws, ts, btc, curve=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return False, "prev_not_archived"
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_v3_env", _env)
    monkeypatch.setattr(milt, "V3_PREV_RETRY_DELAY_S", 0.0)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20,
                   btc_price=9995.0, window_entry_price=10000.0) \
        == ["quote_contrarian_v3a"]
    await _drain(t)
    assert attempts["n"] == 2
    assert [c["signal_version"] for c in fake.calls] == ["quote_contrarian_v3a"]


class _V3EnvSession:
    """_check_v3_env 日高曲线查询会话桩（execute → 预设曲线列表）。

    前窗 outcome 已由 monkeypatch 的 _prev_window_outcome 接管，
    会话内实际只有日高曲线一次 execute。"""

    def __init__(self, curves: list) -> None:
        self._curves = curves

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, stmt):
        res = MagicMock()
        res.scalars.return_value.all.return_value = self._curves
        return res


@pytest.mark.asyncio
async def test_check_v3_env_branches(monkeypatch) -> None:
    """v3 实时环境门禁分支：BTC 缺失/前窗未归档/前窗 UP 各自拒；
    v3a 前窗过即通过；v3b 需距日高回落≤−0.30%（含边界）；
    本窗喂价时序计入日高（W1 对齐影子口径）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    ts = WINDOW_START + 50_000
    prev_holder = {"v": "DOWN"}

    async def _prev(session, ws):
        return prev_holder["v"]

    monkeypatch.setattr(milt, "_prev_window_outcome", _prev)
    # 当日已归档曲线：日高 10000（t ≤ quote_ts，ex-ante）
    curves = [[{"t": WINDOW_START - 120_000, "v": 10000.0}]]
    monkeypatch.setattr(milt, "async_session_factory",
                        lambda: _V3EnvSession(curves))

    # BTC 缺失 → btc_missing
    assert await t._check_v3_env(
        "quote_contrarian_v3a", WINDOW_START, ts, None) \
        == (False, "btc_missing")
    # 前窗未归档 → prev_not_archived（调用方延迟重查）
    prev_holder["v"] = None
    assert await t._check_v3_env(
        "quote_contrarian_v3a", WINDOW_START, ts, 9995.0) \
        == (False, "prev_not_archived")
    # 前窗 UP → prev_not_down（交替环境不成立）
    prev_holder["v"] = "UP"
    assert await t._check_v3_env(
        "quote_contrarian_v3a", WINDOW_START, ts, 9995.0) \
        == (False, "prev_not_down")
    prev_holder["v"] = "DOWN"
    # v3a：前窗 DOWN 即通过（无日高要求）
    assert await t._check_v3_env(
        "quote_contrarian_v3a", WINDOW_START, ts, 9995.0) == (True, "ok")
    # v3b：btc 9970 → 恰 −0.30% 含边界过；9975 → −0.25% 不过
    assert await t._check_v3_env(
        "quote_contrarian_v3b", WINDOW_START, ts, 9970.0) == (True, "ok")
    assert await t._check_v3_env(
        "quote_contrarian_v3b", WINDOW_START, ts, 9975.0) \
        == (False, "env_guard_reject")
    # W1：本窗喂价时序中的高点 10010 计入日高 → 9975 也过
    # （回落 −0.35% ≥ 0.30%；无喂价时按归档日高 10000 会被拒）
    assert await t._check_v3_env(
        "quote_contrarian_v3b", WINDOW_START, ts, 9975.0,
        window_btc_curve=[{"t": WINDOW_START + 20_000, "v": 10010.0}]) \
        == (True, "ok")
    # 喂价中 quote_ts 之后的点不计入（ex-ante）
    assert await t._check_v3_env(
        "quote_contrarian_v3b", WINDOW_START, ts, 9975.0,
        window_btc_curve=[{"t": ts + 1000, "v": 10010.0}]) \
        == (False, "env_guard_reject")
    # 日高缺失（当日无归档曲线+无喂价）→ 回落触发点自保：
    # 触发点即日高，距自身回落 0 < 0.30% → v3b 拒（保守口径）
    monkeypatch.setattr(milt, "async_session_factory",
                        lambda: _V3EnvSession([]))
    assert await t._check_v3_env(
        "quote_contrarian_v3b", WINDOW_START, ts, 9970.0) \
        == (False, "env_guard_reject")


# ============================================================
# 组 3.5：v4 regime 门禁（ret24 K 线核验，影子/实盘同口径，2026-08-31）
# ============================================================

V4_VERSION = "quote_contrarian_v4"


def test_v4_channel_registration() -> None:
    """v4 退役 spec 完整性：RETIRED_CHANNEL_SPECS/REGIME_GUARDS/QUOTE_EDGE_RULES 三方对齐。

    v4 通道已于 2026-09-04 退役（不在 LIVE_CHANNELS），但门禁机制代码与冻结
    回测口径仍在生产（regime_feed / REGIME_GUARDS / QUOTE_EDGE_RULES 未动），
    三方对齐关系继续钉住。
    """
    assert V4_VERSION in RETIRED_CHANNELS and V4_VERSION not in LIVE_CHANNELS
    spec = RETIRED_CHANNEL_SPECS[V4_VERSION]
    assert spec.family == "quote_edge" and spec.regime_gate is True
    assert spec.v2_guard is None and spec.v3_env is False   # v4 不叠加 v2/v3 门禁
    base, thr = REGIME_GUARDS[V4_VERSION]
    assert base == "quote_contrarian_v1"                    # 触发区间引用 v1 冻结口径
    assert QUOTE_EDGE_RULES[base][2:] == (0.15, 0.25)
    assert thr == -1.0                                      # ret24 阈值（含边界）
    assert spec.auto_max_exec == round(0.25 + 0.03, 4)      # 护栏同 v1 推导


@pytest.mark.asyncio
async def test_v4_regime_gate_fires_on_pass(monkeypatch) -> None:
    """v4：门禁通过 → 派生核验任务并下单；同窗后续采样不重复派生。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V4_VERSION])

    async def _ok(self, channel, quote_ts):
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_regime", _ok)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20)\
        == [V4_VERSION]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [V4_VERSION]
    # 同窗后续采样（fired 占位已生效）不再派生
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 55_000, 0.21) == []
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_v4_regime_gate_rejects(monkeypatch) -> None:
    """v4：门禁未过 → 同步预检已过、核验弃单；同窗不再重试。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V4_VERSION])

    async def _reject(self, channel, quote_ts):
        return False, "regime_reject"

    monkeypatch.setattr(MultiLiveTrader, "_check_regime", _reject)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20)\
        == [V4_VERSION]                 # 同步预检已过，派生核验任务
    await _drain(t)
    assert fake.calls == []             # 核验未过 → 弃单
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 55_000, 0.21) == []


@pytest.mark.asyncio
async def test_v4_regime_gate_missing_ret24_retry_once(monkeypatch) -> None:
    """v4：ret24 缺失 → 延迟重查一次；仍缺失 → 弃单；重查通过 → 下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V4_VERSION])
    calls = {"n": 0}

    async def _missing(self, channel, quote_ts):
        calls["n"] += 1
        return False, "ret24_missing"

    monkeypatch.setattr(MultiLiveTrader, "_check_regime", _missing)
    monkeypatch.setattr(milt, "REGIME_RETRY_DELAY_S", 0.0)  # 免等待
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20)\
        == [V4_VERSION]
    await _drain(t)
    assert calls["n"] == 2              # 首查 + 延迟重查一次
    assert fake.calls == []

    # 重查通过场景：首次 K 线拉取失败，重试后恢复
    fake2 = _FakeTrader()
    t2 = _make_trader(monkeypatch, fake2, channels=[V4_VERSION])
    attempts = {"n": 0}

    async def _flaky(self, channel, quote_ts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return False, "ret24_missing"
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_regime", _flaky)
    monkeypatch.setattr(milt, "REGIME_RETRY_DELAY_S", 0.0)
    assert t2.check(WINDOW_START, WINDOW_END, WINDOW_START + 50_000, 0.20)\
        == [V4_VERSION]
    await _drain(t2)
    assert attempts["n"] == 2
    assert [c["signal_version"] for c in fake2.calls] == [V4_VERSION]


@pytest.mark.asyncio
async def test_check_regime_branches(monkeypatch) -> None:
    """_check_regime 分支：贴线含边界（恰 −1.0% 过 / −0.9% 拒）；
    ret24 缺失保守拒；未注册 regime 门禁的通道拒绝。"""
    t = _make_trader(monkeypatch, _FakeTrader())

    class _Feed:
        def __init__(self, v):
            self._v = v

        async def ret24_at(self, ts_ms):
            return self._v

    feed = _Feed(-0.010)                # 恰 −1.0% → 含边界过
    monkeypatch.setattr(milt, "regime_feed", feed)
    assert await t._check_regime(V4_VERSION, 123) == (True, "ok")
    feed._v = -0.009                    # −0.9% → 拒
    assert await t._check_regime(V4_VERSION, 123)\
        == (False, "regime_reject")
    feed._v = None                      # K 线缺失 → 保守拒
    assert await t._check_regime(V4_VERSION, 123)\
        == (False, "ret24_missing")
    feed._v = -0.012
    assert await t._check_regime("quote_contrarian_v3a", 123)\
        == (False, "no_regime_guard")   # 未注册 regime 门禁的通道


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


@pytest.mark.asyncio
async def test_x4_poll_passes_v3_whitelist(monkeypatch) -> None:
    """x4_v3 PENDING 轮询 → 下单 kwargs 透传入场价白名单（ChannelSpec 声明式）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["x4_v3"])
    now_ms = int(time.time() * 1000)
    target = now_ms - 155_000
    _stub_select_db(monkeypatch, [(601, "x4_v3", target)])

    await t._x4_poll_once()
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["signal_version"] == "x4_v3"
    assert call["entry_band_whitelist"] == ((0.0, 0.2), (0.3, 0.4))
    assert call["max_exec_price"] == 0.50
    assert call["market_period"] == "5m"


# ============================================================
# 组 6：场景钩子（fake_breakout fire → 15m 市场下单）
# ============================================================

def _scene_sig(pattern_type: str, side: str, sig_id: int = 42) -> dict:
    return {"id": sig_id, "pattern_type": pattern_type, "side": side,
            "market_start_15m": MARKET_START_15M,
            "market_end_15m": MARKET_START_15M + 900_000}


@pytest.mark.asyncio
async def test_scene_s1_down_15m(monkeypatch) -> None:
    """S1（多头耗尽，side=high）→ 15m 市场押 DOWN，护栏 0.70（2026-08-30 盘口校准）。"""
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
    assert call["max_exec_price"] == 0.70
    assert call["amount_usdt"] == 2.0


@pytest.mark.asyncio
async def test_scene_s2_up_15m(monkeypatch) -> None:
    """S2（空头耗尽，side=low）→ 15m 市场押 UP，护栏 0.65（2026-08-30 盘口校准）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["scene_bear_exhaust"])
    t.on_scene_signal(_scene_sig("bear_exhaust", "low"))
    await _drain(t)
    call = fake.calls[0]
    assert call["prediction"] == "UP"
    assert call["signal_version"] == "scene_bear_exhaust"
    assert call["market_period"] == "15m"
    assert call["max_exec_price"] == 0.65


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
# 组 6b：影子 promote 三族钩子（s2_cond / nextbar / absorption，2026-09-06）
# ============================================================

def _s2_cond_sig(version: str = "s2_cond_t4_v1", parent_id: int | None = 9) -> dict:
    """S2CondShadowDetector._dispatch_live payload 替身。"""
    return {
        "version": version,
        "market_start_15m": MARKET_START_15M,
        "market_end_15m": MARKET_START_15M + 900_000,
        "parent_id": parent_id,
        "up_quote": 0.42,
    }


def _nextbar_sig(version: str = "nb_smaslope_5m_v1") -> dict:
    """NextbarShadowDetector._dispatch_live payload 替身（目标根 = WINDOW_START）。"""
    return {
        "version": version,
        "market_start": WINDOW_START,
        "market_end": WINDOW_START + 300_000,
        "direction": "UP",
        "signal_bar_start": WINDOW_START - 300_000,
    }


class _FakeCalibrator:
    """absorption_calibrator 替身：live_calibration 返回预设快照（None=标定缺失）。"""

    def __init__(self, fit: tuple | None):
        self._fit = fit

    def live_calibration(self, version: str):
        return self._fit


@pytest.mark.asyncio
async def test_s2_cond_hook_fires_15m_up(monkeypatch) -> None:
    """s2_cond 钩子：判价命中 → 15m 市场押 UP（冻结方向恒 UP），护栏取 spec。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["s2_cond_t4_v1"])
    t.on_s2_cond_signal(_s2_cond_sig("s2_cond_t4_v1"))
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["signal_version"] == "s2_cond_t4_v1"
    assert call["prediction"] == "UP"
    assert call["market_period"] == "15m"
    assert call["window_start"] == MARKET_START_15M
    assert call["max_exec_price"] == 0.38      # spec.auto_max_exec（38.9%×0.98 略下方）
    assert call["amount_usdt"] == 2.0


@pytest.mark.asyncio
async def test_s2_cond_disabled_and_stopped_no_fire(monkeypatch) -> None:
    """s2_cond：通道关 / stopped → 零下单（影子采集照常，互不干扰）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[])
    t.on_s2_cond_signal(_s2_cond_sig())
    await _drain(t)
    assert fake.calls == []
    fake2 = _FakeTrader()
    t2 = _make_trader(monkeypatch, fake2, channels=["s2_cond_t4_v1"])
    t2._stopped = True
    t2.on_s2_cond_signal(_s2_cond_sig())
    await _drain(t2)
    assert fake2.calls == []


@pytest.mark.asyncio
async def test_s2_cond_same_market_dedup(monkeypatch) -> None:
    """同一 S2 事件重复回调 → 每周期至多一单（内存 fired 防重）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["s2_cond_t4_v1"])
    t.on_s2_cond_signal(_s2_cond_sig(parent_id=1))
    t.on_s2_cond_signal(_s2_cond_sig(parent_id=2))
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_s2_cond_exclusive_t4_fill_blocks_t5d(monkeypatch) -> None:
    """同窗互斥：t4 成交占坑 → t5d 同窗拒单（同一 S2 事件至多一单成交）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["s2_cond_t4_v1", "s2_cond_t5d_v1"])
    t.on_s2_cond_signal(_s2_cond_sig("s2_cond_t4_v1"))
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == ["s2_cond_t4_v1"]
    t.on_s2_cond_signal(_s2_cond_sig("s2_cond_t5d_v1"))
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == ["s2_cond_t4_v1"]


@pytest.mark.asyncio
async def test_s2_cond_exclusive_reject_keeps_slot_free(monkeypatch) -> None:
    """同窗互斥：t4 弃单（非 FILLED）不占坑 → t5d 同窗仍可下单（互补重试）。"""
    fake = _FakeTrader(result=_fake_order(status="REJECTED",
                                          error_message="exec_price_guard",
                                          signal_version="s2_cond_t4_v1"))
    t = _make_trader(monkeypatch, fake,
                     channels=["s2_cond_t4_v1", "s2_cond_t5d_v1"])
    t.on_s2_cond_signal(_s2_cond_sig("s2_cond_t4_v1"))
    await _drain(t)
    t.on_s2_cond_signal(_s2_cond_sig("s2_cond_t5d_v1"))
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "s2_cond_t4_v1", "s2_cond_t5d_v1"]


@pytest.mark.asyncio
async def test_nextbar_hook_fires_5m(monkeypatch) -> None:
    """nextbar 钩子：新根命中 → 5m 市场押 UP（direction 透传），护栏 0.46。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["nb_smaslope_5m_v1"])
    t.on_nextbar_signal(_nextbar_sig())
    await _drain(t)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["signal_version"] == "nb_smaslope_5m_v1"
    assert call["prediction"] == "UP"
    assert call["market_period"] == "5m"
    assert call["window_start"] == WINDOW_START
    assert call["max_exec_price"] == 0.46


@pytest.mark.asyncio
async def test_nextbar_disabled_and_stopped_no_fire(monkeypatch) -> None:
    """nextbar：通道关 / stopped → 零下单（冷启动回补由检测器侧时间窗守卫排除）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[])
    t.on_nextbar_signal(_nextbar_sig())
    await _drain(t)
    assert fake.calls == []
    fake2 = _FakeTrader()
    t2 = _make_trader(monkeypatch, fake2, channels=["nb_smaslope_5m_v1"])
    t2._stopped = True
    t2.on_nextbar_signal(_nextbar_sig())
    await _drain(t2)
    assert fake2.calls == []


@pytest.mark.asyncio
async def test_absorption_td_not_reached_no_fire(monkeypatch) -> None:
    """absorption：TD 未到不判定；首个采样只建窗开快照。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    # 首采样建快照（不判定）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
                   btc_price=100.0, window_entry_price=100.0, up_price=0.50,
                   up_open=0.50) == []
    # TD=120s 前不判定
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 60_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) == []
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_absorption_missing_calibration_no_fire(monkeypatch) -> None:
    """absorption：标定源未注入 / live_calibration None → 保守不开火（fail-safe）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
            btc_price=100.0, window_entry_price=100.0, up_price=0.50,
            up_open=0.50)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 130_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) == []
    await _drain(t)
    assert fake.calls == []
    # 标定源在册但返回 None（缓冲 <MIN_CALIB）→ 同样不开火
    t.absorption_calibrator = _FakeCalibrator(None)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 135_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) == []
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_absorption_fires_up_when_btc_up(monkeypatch) -> None:
    """absorption：过位移门+欠反应门 → 开火押 UP（follow 补涨），护栏 0.78。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
            btc_price=100.0, window_entry_price=100.0, up_price=0.50,
            up_open=0.50)
    # TD 后：btc +30bp（过位移门 10bp）、UP 报价粘滞 under=+59.5pp（过欠反应门 0.5）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 130_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) \
        == ["absorption_follow_td120_v1"]
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "absorption_follow_td120_v1"
    assert call["prediction"] == "UP"
    assert call["market_period"] == "5m"
    assert call["window_start"] == WINDOW_START
    assert call["max_exec_price"] == 0.78
    assert call["amount_usdt"] == 2.0


@pytest.mark.asyncio
async def test_absorption_fires_down_when_btc_down(monkeypatch) -> None:
    """absorption：btc 跌（btc_move<0）过门 → 押 DOWN（方向随 btc_move 符号）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
            btc_price=100.0, window_entry_price=100.0, up_price=0.50,
            up_open=0.50)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 130_000, 0.5,
                   btc_price=99.7, window_entry_price=100.0, up_price=0.505) \
        == ["absorption_follow_td120_v1"]
    await _drain(t)
    assert fake.calls[0]["prediction"] == "DOWN"


@pytest.mark.asyncio
async def test_absorption_exclusive_td120_fill_blocks_td150(monkeypatch) -> None:
    """absorption 同窗互斥：td120 成交占坑 → td150 同窗同步开火但下单被拒。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "absorption_follow_td120_v1", "absorption_follow_td150_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
            btc_price=100.0, window_entry_price=100.0, up_price=0.50,
            up_open=0.50)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 130_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) \
        == ["absorption_follow_td120_v1"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "absorption_follow_td120_v1"]
    # TD150 时点：门照过、fired 占位生效（同步开火），但互斥槽拒单（td120 已成交）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 160_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505) \
        == ["absorption_follow_td150_v1"]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "absorption_follow_td120_v1"]


@pytest.mark.asyncio
async def test_absorption_midwindow_no_up_open_no_fire(monkeypatch) -> None:
    """absorption 中途建快照防线：拿不到窗开 up_open（冷启动/中途启用且本窗
    尚无 UP 报价入库）→ 本窗跳过，绝不用当前 up_price 当基准伪开火。

    回归 2026-09-06 审计 Critical：中途基准 up_move≈0 → under≈k·|btc_move|
    虚高必过欠反应门 → 在途窗口伪开火下真单。
    """
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    # t=200s 中途首采样（up_open 缺失）：不建快照、不判定
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 200_000, 0.5,
                   btc_price=100.0, window_entry_price=100.0,
                   up_price=0.50) == []
    # 后续采样数学上门必过（up_move≈0 → under=+59.95pp），仍不得开火
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 215_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0,
                   up_price=0.5005) == []
    await _drain(t)
    assert fake.calls == []
    assert WINDOW_START not in t._abs_open


@pytest.mark.asyncio
async def test_absorption_midwindow_enable_uses_window_open_basis(monkeypatch) -> None:
    """absorption 中途启用通道：up_open 由 main 从 _pm_history 传入（本窗归档
    首个有效 UP 采样，与影子 _first(up_p) 同源）→ 基准正确，在途窗口仍可判定。

    t=130s 启用后首采样建快照（up_open=0.50 窗开价、up_price=0.505 当前价），
    t=145s 判定：up_move=+0.5pp、btc_move=+30bp → under=+59.5pp 过门 → 开火
    （证明修复不损失在途窗口，且基准是窗开价而非中途价）。
    """
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 130_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505,
                   up_open=0.50) == []          # 快照刚建立，判定等后续采样
    assert t._abs_open[WINDOW_START] == (0.50, 100.0)   # 基准=窗开价非中途价
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 145_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.505,
                   up_open=0.50) == ["absorption_follow_td120_v1"]
    await _drain(t)
    assert fake.calls[0]["prediction"] == "UP"


@pytest.mark.asyncio
async def test_absorption_stale_judge_window_skipped(monkeypatch) -> None:
    """absorption 判定新鲜度：t_rel 超 TD+90s（重启后在途窗）→ 本窗放弃。

    当前价已非 TD 时刻价，与影子 ≤TD 末点口径不成立；即使数学过门也不下注
    （口径漂移防御，与中途建快照防线互补）。
    """
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["absorption_follow_td120_v1"])
    t.absorption_calibrator = _FakeCalibrator((2.0, 0.0, 10.0, 0.5))
    # t=5s 建快照（up_open 正确），随后进程「重启」错过判定点，t=250s 才恢复采样
    t.check(WINDOW_START, WINDOW_END, WINDOW_START + 5_000, 0.5,
            btc_price=100.0, window_entry_price=100.0, up_price=0.50,
            up_open=0.50)
    # 250s > TD120+90s：up_move=+0.05pp、under=+59.95pp 数学过门，但口径过期
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 250_000, 0.5,
                   btc_price=100.3, window_entry_price=100.0, up_price=0.5005,
                   up_open=0.50) == []
    await _drain(t)
    assert fake.calls == []


# ============================================================
# 组 8：firsthit G0/G1/G3 实时重放（down_curve 历史唯一触发源）
# ============================================================

def _firsthit_down_curve(trigger_q: float = 0.07, npts: int = 20, trigger_idx: int = 8) -> list[dict]:
    """构造 down_curve，使 trigger_idx 位置的点是第一个入区点（默认 t=120s）。"""
    pts = []
    for i in range(npts):
        ts = WINDOW_START + i * 15_000   # 15s 采样
        if i < trigger_idx:
            q = 0.5 + 0.4 * (i / trigger_idx)      # 前半段 >0.1
        else:
            q = trigger_q                          # 触发点及之后 ∈ (Q_LO, Q_HI]
        pts.append({"t": ts, "v": q})
    return pts


def _firsthit_btc_curve(btc_open: float = 100.0, btc_trig: float = 100.3, npts: int = 20) -> list[dict]:
    """btc_curve：开盘 + 连续上升至 btc_trig。"""
    pts = [{"t": WINDOW_START, "v": btc_open}]
    for i in range(1, npts):
        pts.append({"t": WINDOW_START + i * 15_000, "v": btc_open + (btc_trig - btc_open) * (i / (npts - 1))})
    return pts


@pytest.mark.asyncio
async def test_firsthit_enabled_channels_registered(monkeypatch) -> None:
    """firsthit G0/G1/G3 三个通道已注册，默认关闭，护栏按约定值。"""
    from binance_predict.services.live_channels import LIVE_CHANNELS
    assert all(ch in LIVE_CHANNELS for ch in ["firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1"])
    cfgs = parse_channel_config()
    assert not cfgs["firsthit_down_v1"].enabled and not cfgs["firsthit_down_body_v1"].enabled and not cfgs["firsthit_down_chg_v1"].enabled
    assert cfgs["firsthit_down_v1"].amount_usdt == 2.0
    assert cfgs["firsthit_down_v1"].max_daily_orders == 100
    assert cfgs["firsthit_down_v1"].max_exec_price is None       # 缺省回落 spec.auto_max_exec
    assert cfgs["firsthit_down_body_v1"].max_exec_price is None
    assert cfgs["firsthit_down_chg_v1"].max_exec_price is None


@pytest.mark.asyncio
async def test_firsthit_g0_match_real_first_touch(monkeypatch) -> None:
    """首次触价 G0：完整历史里有第一入区点 → 开火 DOWN，每通道每窗一单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["firsthit_down_v1"])
    dn_p = _firsthit_down_curve(trigger_q=0.07, npts=20)
    btc_p = _firsthit_btc_curve(btc_open=100.0, btc_trig=100.3, npts=20)
    # t=REL_T_SEC 采样时，首触点已完整记录
    REL_T_SEC = 120  # 120s
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                   down_price=0.5, btc_price=100.3,
                   window_entry_price=100.0,
                   window_btc_curve=btc_p, window_down_curve=dn_p) == ["firsthit_down_v1"]
    await _drain(t)
    call = fake.calls[0]
    assert call["signal_version"] == "firsthit_down_v1"
    assert call["prediction"] == "DOWN"
    assert call["window_start"] == WINDOW_START
    assert call["market_period"] == "5m"
    assert call["amount_usdt"] == 2.0
    assert call["max_exec_price"] == 0.08


@pytest.mark.asyncio
async def test_firsthit_g1_g3_gate_rejects_high_chg_or_large_body(monkeypatch) -> None:
    """G1/G3 门：chg≤+2.82bp / body_r≤0.35；不满足则不开火，G0 仍可开。"""
    fake = _FakeTrader()
    # chg 较大（+5bp）、body_r 较大（≈1.0）
    t = _make_trader(monkeypatch, fake, channels=["firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1"])
    dn_p = _firsthit_down_curve(trigger_q=0.07, npts=20)
    btc_p = _firsthit_btc_curve(btc_open=100.0, btc_trig=100.5, npts=20)  # chg≈+5bp
    REL_T_SEC = 120
    fired = t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                    down_price=0.5, btc_price=100.5,
                    window_entry_price=100.0,
                    window_btc_curve=btc_p, window_down_curve=dn_p)
    assert "firsthit_down_v1" in fired
    assert "firsthit_down_body_v1" not in fired
    assert "firsthit_down_chg_v1" not in fired
    await _drain(t)
    assert fake.calls[-1]["signal_version"] == "firsthit_down_v1"


@pytest.mark.asyncio
async def test_firsthit_three_channels_independent_same_window_fires_all_on(monkeypatch) -> None:
    """三通道同时开启且三门全中：派生三单，互不阻塞（用户确认独立下单）。
    
    三门全中需要：chg≤+2.82bp（G3）、body_r≤0.35（G1）、npts≥8（路径质量）。
    body_r<1 要求 btc_curve 有波动（中间有比 bo 更低或比 btc_trig 更高的点）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake,
                     channels=["firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1"],
                     overrides={"firsthit_down_body_v1": {"enabled": True, "max_exec_price": 0.12},
                                "firsthit_down_chg_v1": {"enabled": True, "max_exec_price": 0.09}})
    # chg=+2bp、body_r≈0.17 → 三门全中（i=8 触发点 v=100.02）
    dn_p = _firsthit_down_curve(trigger_q=0.07, npts=20, trigger_idx=8)
    # btc_curve：下跌→反弹，让 body_r<0.35 且 chg≤2.82bp
    btc_p = [{"t": WINDOW_START, "v": 100.0}]
    for i in range(1, 20):
        ts = WINDOW_START + i * 15_000
        if i <= 7:
            v = 100.0 - i * 0.0143  # 下跌段 100.0→99.90 (i=7 最低)
        elif i == 8:
            v = 100.02           # i=8 触发点：直接到 100.02（chg=+2bp）
        else:
            v = 100.02 + (i - 8) * 0.01  # 反弹后继续上行
        btc_p.append({"t": ts, "v": v})
    REL_T_SEC = 120
    fired = t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                    down_price=0.5, btc_price=100.02,
                    window_entry_price=100.0,
                    window_btc_curve=btc_p, window_down_curve=dn_p)
    # per-channel fired 防重复；三通道各自 open
    assert set(fired) == {"firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1"}
    await _drain(t)
    # 应有三单
    assert len(fake.calls) == 3
    versions = [c["signal_version"] for c in fake.calls]
    assert set(versions) == {"firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1"}
    # 检查护栏按 spec auto_max_exec 生效
    exec_prices = {c["signal_version"]: c.get("max_exec_price") for c in fake.calls}
    assert exec_prices.get("firsthit_down_v1") == 0.08
    assert exec_prices.get("firsthit_down_body_v1") == 0.12
    assert exec_prices.get("firsthit_down_chg_v1") == 0.09


@pytest.mark.asyncio
async def test_firsthit_no_down_curve_data_not_fire(monkeypatch) -> None:
    """缺 window_down_curve → 保守不开火（向后兼容旧调用方）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["firsthit_down_v1"])
    btc_p = _firsthit_btc_curve(btc_open=100.0, btc_trig=100.3, npts=20)
    REL_T_SEC = 120
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                   down_price=0.5, btc_price=100.3,
                   window_entry_price=100.0,
                   window_btc_curve=btc_p, window_down_curve=None) == []
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_firsthit_npts_under_min_not_fire(monkeypatch) -> None:
    """路径 npts<8 → 保守不开火（v2 主分析口径）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["firsthit_down_v1"])
    dn_p = _firsthit_down_curve(trigger_q=0.07, npts=6)       # npts=6 < MIN_PTS=8
    btc_p = _firsthit_btc_curve(btc_open=100.0, btc_trig=100.3, npts=6)
    REL_T_SEC = 120
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                   down_price=0.07, btc_price=100.3,
                   window_entry_price=100.0,
                   window_btc_curve=btc_p, window_down_curve=dn_p) == []
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_firsthit_does_not_look_ahead_past_current_sample(monkeypatch) -> None:
    """回归：check() 必须把当前采样时刻作为 max_trigger_ts 传入（防未来函数）。

    场景：t=15s 采样时 DOWN 尚未入区，真实首触发生在 t=30s。此时 btc 路径已有
    9 点（≥MIN_PTS），所以若调用点漏传 max_trigger_ts，纯函数会「偷看」t=30s
    的未来报价并立即开火——在本窗还没到首触时刻就下真单。
    本测试锁住调用点契约：早于首触的采样必须返回空。
    """
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=["firsthit_down_v1"])
    # down_curve：t≤15s 全 >0.1，t=30s 才首次入区
    dn_p = [
        {"t": WINDOW_START, "v": 0.50},
        {"t": WINDOW_START + 7_500, "v": 0.40},
        {"t": WINDOW_START + 15_000, "v": 0.30},
        {"t": WINDOW_START + 30_000, "v": 0.07},   # 真实首触（未来）
    ]
    # btc_curve：3.75s 间隔，t≤30s 共 9 点（满足 npts≥8，排除 npts 门兜底）
    btc_p = [{"t": WINDOW_START + i * 3_750, "v": 100.0 + i * 0.01} for i in range(9)]

    # t=15s：首触尚未发生 → 不得开火
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 15_000,
                   down_price=0.30, btc_price=100.04,
                   window_entry_price=100.0,
                   window_btc_curve=btc_p, window_down_curve=dn_p) == []
    await _drain(t)
    assert fake.calls == []

    # t=30s：首触已发生 → 正常开火（证明上面不是「永不开火」的假绿）
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 30_000,
                   down_price=0.07, btc_price=100.08,
                   window_entry_price=100.0,
                   window_btc_curve=btc_p, window_down_curve=dn_p) == ["firsthit_down_v1"]
    await _drain(t)
    assert len(fake.calls) == 1
    assert fake.calls[0]["prediction"] == "DOWN"


@pytest.mark.asyncio
async def test_firsthit_g7_series_all_fire(monkeypatch) -> None:
    """G7 纯组合基底及 5 个变体在满足条件时均能正确开火，独立下单，参数符合护栏。"""
    fake = _FakeTrader()
    g7_channels = [
        "firsthit_down_g7_v1", "g7_streak_v1", "g7_wick20_v1",
        "g7_strict_v1", "g7_q05_v1", "g7_t270_v1",
    ]
    t = _make_trader(monkeypatch, fake, channels=g7_channels)

    # 模拟满足所有 G7 条件的采样：
    # 1. q=0.04 (<=0.05，满足 q05);
    # 2. t=120s (<=270s，满足 t270);
    # 3. body_r=0.20 (<=0.35);
    # 4. wick01=1.0 且 upper_wick_bps=2.5bp (>=2.0bp, 满足 wick20 和 strict);
    # 5. npts >= 8.
    dn_p = _firsthit_down_curve(trigger_q=0.04, npts=20, trigger_idx=8)
    btc_p = [{"t": WINDOW_START, "v": 100.0}]
    for i in range(1, 20):
        ts = WINDOW_START + i * 15_000
        if i <= 5:
            v = 100.0 - i * 0.01  # 最低 99.95
        elif i == 6:
            v = 100.05           # 冲高留上影 (cum_hi = 100.05, 相对 100.0 涨 5bp)
        elif i == 8:
            v = 100.02           # 触发点 (upper_wick = 100.05 - 100.02 = 0.03 -> 3bp >= 2bp)
        else:
            v = 100.01
        btc_p.append({"t": ts, "v": v})

    # mock session 查前驱连阳返回 0 连阳（满足 streak_up <= 1）
    class _DummySession:
        async def execute(self, stmt):
            res = MagicMock()
            res.scalars.return_value.all.return_value = [-0.001]  # 前窗收阴 -> streak_up = 0
            return res
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    import binance_predict.services.multi_live_trader as mlt_mod
    monkeypatch.setattr(mlt_mod, "async_session_factory", lambda: _DummySession())

    REL_T_SEC = 120
    fired = t.check(WINDOW_START, WINDOW_END, WINDOW_START + REL_T_SEC * 1000,
                    down_price=0.04, btc_price=100.02,
                    window_entry_price=100.0,
                    window_btc_curve=btc_p, window_down_curve=dn_p)
    assert set(fired) == set(g7_channels)
    await _drain(t)

    assert len(fake.calls) == 6
    called_channels = [c["signal_version"] for c in fake.calls]
    assert set(called_channels) == set(g7_channels)
    # 验证各通道护栏
    guards = {c["signal_version"]: c.get("max_exec_price") for c in fake.calls}
    assert guards["firsthit_down_g7_v1"] == 0.10
    assert guards["g7_streak_v1"] == 0.10
    assert guards["g7_wick20_v1"] == 0.12
    assert guards["g7_strict_v1"] == 0.12
    assert guards["g7_q05_v1"] == 0.05
    assert guards["g7_t270_v1"] == 0.10


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
    """status：31 通道全量（23 在线 +8 退役 fixture）、enabled_any/defaults/amount_cap、单通道字段。"""
    t = _make_trader(monkeypatch, _FakeTrader(), channels=["quote_contrarian_v1"])
    s = t.status()
    assert s["enabled_any"] is True
    assert s["amount_cap"] == 50
    assert s["defaults"]["amount_usdt"] == 2.0
    assert s["defaults"]["max_daily_orders"] == 100
    assert len(s["channels"]) == 31   # 既有 24+G7 系列 6→30，加 G4 共 31
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
                        amount_usdt=1.5, max_daily_orders=50,
                        max_exec_price=0.55),
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
    assert t._configs["x4_v1"].max_exec_price == 0.55    # 护栏自定义覆盖（DB 层）
    assert t._configs["quote_contrarian_v1"].enabled is False   # DB 胜过 env 层
    assert t._enabled_at_startup["quote_contrarian_v1"] is False


@pytest.mark.asyncio
async def test_persist_channel_writes_snapshot(monkeypatch) -> None:
    """持久化：upsert 该通道当前完整配置快照（重启后恢复的正是最后设定）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    sess = _stub_override_db(monkeypatch, [])

    t.set_channel("x4_v2", enabled=True, amount_usdt=3.0, max_daily_orders=7,
                  max_exec_price=0.52)
    await t.persist_channel("x4_v2")

    assert len(sess.merged) == 1
    row = sess.merged[0]
    assert row.channel == "x4_v2" and row.enabled is True
    assert row.amount_usdt == 3.0 and row.max_daily_orders == 7
    assert row.max_exec_price == 0.52
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


@pytest.mark.asyncio
async def test_toggle_endpoint_max_exec_heat_tune(monkeypatch) -> None:
    """toggle 端点护栏热调：设置生效并持久化；reset_max_exec=True 回落预设。"""
    import binance_predict.main as m
    from binance_predict.models.schemas import ToggleLiveRequest

    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    monkeypatch.setattr(m, "multi_live_trader", t)

    async def _persist(self, channel):
        return None

    async def _status_async(self):
        return self.status()

    monkeypatch.setattr(MultiLiveTrader, "persist_channel", _persist)
    monkeypatch.setattr(MultiLiveTrader, "status_async", _status_async)

    auto = t._specs["x4_v1"].auto_max_exec  # 0.45（预设值）
    out = await m.live_toggle(
        ToggleLiveRequest(channel="x4_v1", enabled=False,
                          max_exec_price=0.58), _=None)
    assert "error" not in out
    assert t._configs["x4_v1"].max_exec_price == 0.58
    by = {c["channel"]: c for c in out["status"]["channels"]}
    assert by["x4_v1"]["max_exec_price"] == 0.58
    assert by["x4_v1"]["custom_max_exec"] is True

    out = await m.live_toggle(
        ToggleLiveRequest(channel="x4_v1", enabled=False,
                          reset_max_exec=True), _=None)
    assert "error" not in out
    assert t._configs["x4_v1"].max_exec_price is None     # 回落预设
    by = {c["channel"]: c for c in out["status"]["channels"]}
    assert by["x4_v1"]["max_exec_price"] == auto
    assert by["x4_v1"]["custom_max_exec"] is False


def test_set_channel_max_exec_bounds_atomic(monkeypatch) -> None:
    """护栏热调校验：超界拒改（原子），enabled 不被半置位。"""
    t = _make_trader(monkeypatch, _FakeTrader(), channels=[])
    with pytest.raises(ValueError, match="超界"):
        t.set_channel("x4_v1", enabled=True, max_exec_price=1.2)
    assert t._configs["x4_v1"].enabled is False   # 校验失败整体拒改
    with pytest.raises(ValueError, match="超界"):
        t.set_channel("x4_v1", max_exec_price=0.001)
    # 边界值合法：0.01 与 0.99
    t.set_channel("x4_v1", max_exec_price=0.01)
    assert t._configs["x4_v1"].max_exec_price == 0.01
    t.set_channel("x4_v1", max_exec_price=0.99)
    assert t._configs["x4_v1"].max_exec_price == 0.99


# ============================================================
# 组 11：盈亏趋势端点（GET /api/live/pnl-curve）
# ============================================================

@pytest.mark.asyncio
async def test_live_pnl_curve_endpoint(monkeypatch) -> None:
    """pnl-curve：按通道分组累计 pnl；排除未结算/manual_test；
    排序按总盈亏降序；执行器未装配时元数据回落注册表（enabled=False）。"""
    import binance_predict.main as m

    rows = [
        # (signal_version, window_start, settled_at, win, pnl) 按 window_start 升序
        ("quote_contrarian_v1", 1000,
         datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc), True, 4.0),
        ("scene_bull_exhaust", 1500, None, True, 1.5),
        ("quote_contrarian_v1", 2000,
         datetime(2026, 8, 30, 10, 5, tzinfo=timezone.utc), False, -2.0),
    ]
    _stub_select_db(monkeypatch, rows)
    monkeypatch.setattr(m, "multi_live_trader", None)  # 元数据回落注册表

    out = await m.live_pnl_curve(_=None, db=_SelectSession(rows))
    assert out["total"]["settled_count"] == 3
    assert out["total"]["total_pnl"] == 3.5
    assert out["total"]["win_rate"] == round(2 / 3, 4)
    # 口径自描述字段（2026-09-08 审计）：前端展示口径的事实源
    assert out["scope"]["filter"] == "LIVE_CHANNELS ∩ FILLED ∩ win≠NULL ∩ pnl≠NULL"
    assert "manual_test" in out["scope"]["excludes"]
    assert "fund-flow" in out["scope"]["note"]

    by = {c["channel"]: c for c in out["channels"]}
    # 排序：quote_contrarian_v1(+2.0) 在 scene_bull_exhaust(+1.5) 前
    assert out["channels"][0]["channel"] == "quote_contrarian_v1"

    q = by["quote_contrarian_v1"]
    assert q["settled_count"] == 2 and q["win_count"] == 1
    assert q["total_pnl"] == 2.0
    assert [p["cum"] for p in q["points"]] == [4.0, 2.0]   # 逐单累计
    assert q["points"][1]["n"] == 2
    assert q["enabled"] is False   # 执行器未装配 → 回落注册表默认

    s = by["scene_bull_exhaust"]
    assert s["total_pnl"] == 1.5
    assert s["points"][0]["t"] == 1500   # settled_at 缺失回落 window_start


@pytest.mark.asyncio
async def test_live_pnl_curve_endpoint_empty(monkeypatch) -> None:
    """无已结算订单：channels 空列表，total 归零（不抛 500）。"""
    import binance_predict.main as m

    _stub_select_db(monkeypatch, [])
    monkeypatch.setattr(m, "multi_live_trader", None)

    out = await m.live_pnl_curve(_=None, db=_SelectSession([]))
    assert out["channels"] == []
    assert out["total"] == {"settled_count": 0, "total_pnl": 0, "win_rate": None}


# ============================================================
# 组 8：execute_signal_trade（15m token 分流 + 5m 行为迁移）
# ============================================================

def _make_real_trader(monkeypatch, with_15m: bool = True) -> BinancePredictionTrader:
    trader = BinancePredictionTrader()
    trader._api_key = "k"
    trader._api_secret = "s"
    trader._wallet_address = "0xWALLET"
    trader._wallet_id = "WID"
    # 5m 锚定：缓存市场的 startDate 与本单窗口一致（默认路径不触发 detail 预取）
    trader._5m_start_date = WINDOW_START

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

    # detail 预取默认不命中（避免单测触网；命中路径有独立用例覆盖）
    async def _no_detail(_period, _start_ms):
        return None

    monkeypatch.setattr(trader, "_fetch_market_via_detail", _no_detail)

    # 默认币安侧终态回查为 FILLED（组 8 各用例聚焦下单分流逻辑；
    # 幽灵成交/待对账路径有独立用例覆盖）。
    # 2026-08-28 confirm_order_status 返回命中历史行 dict（status 取终态）。
    # 签名与 P0 后的常量对齐（attempts/delay 默认值已放宽）。
    async def _confirm(_oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        return {"orderId": _oid, "status": "FILLED", "price": "0.55"}

    monkeypatch.setattr(trader, "confirm_order_status", _confirm)

    # 余额预检默认放行（P1 新增）。不 stub 会因 _api_key="k" 真去签币安
    # balance 端点（单测触网红线）；余额不足/低余额告警有组 9 独立用例覆盖。
    async def _bal(_amount_usdt):
        return True, 1000.0, ""

    monkeypatch.setattr(trader, "_preflight_balance", _bal)
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
async def test_signal_trade_15m_detail_prefetch_fallback(monkeypatch) -> None:
    """list 未含目标周期（边界刚过新周期未可见）→ detail 预取未来市场命中，
    正常报价下单并入缓存（2026-08-28 修复路径）。"""
    trader = _make_real_trader(monkeypatch, with_15m=False)
    quote_tokens: list[str] = []
    fetched: list[tuple] = []

    async def _detail(period, start_ms):
        fetched.append((period, start_ms))
        return {"end_date": start_ms + 900_000, "up_token": "T-FUT-UP",
                "down_token": "T-FUT-DOWN", "up_price": 0.5, "down_price": 0.5}

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
        return {"orderId": "ORD-PREFETCH"}

    monkeypatch.setattr(trader, "_fetch_market_via_detail", _detail)
    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START,
        max_exec_price=0.60, market_period="15m", scene_signal_id=42)
    assert order["status"] == "FILLED"
    assert fetched == [("15m", WINDOW_START)]
    assert quote_tokens == ["T-FUT-DOWN"]
    # 预取结果入缓存：后续同窗口重入直接命中，不再走 detail
    assert trader._15m_markets[WINDOW_START]["down_token"] == "T-FUT-DOWN"


@pytest.mark.asyncio
async def test_signal_trade_5m_anchor_mismatch_prefetch(monkeypatch) -> None:
    """5m 缓存市场与本单窗口不一致（开火瞬间新周期未进 list）→ 不用旧周期
    token，走 detail 预取本窗口市场。"""
    trader = _make_real_trader(monkeypatch)
    trader._5m_start_date = WINDOW_START - 300_000  # 缓存属上一周期
    quote_tokens: list[str] = []

    async def _detail(period, start_ms):
        return {"end_date": start_ms + 300_000, "up_token": "T5-UP",
                "down_token": "T5-DOWN", "up_price": 0.5, "down_price": 0.5}

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
        return {"orderId": "ORD-5M-PREFETCH"}

    monkeypatch.setattr(trader, "_fetch_market_via_detail", _detail)
    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_edge_v1", WINDOW_START, max_exec_price=0.60)
    assert order["status"] == "FILLED"
    assert quote_tokens == ["T5-DOWN"]  # 本窗口市场，非上一周期缓存


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


@pytest.mark.asyncio
async def test_signal_trade_guard_boundary_touch_rejected(monkeypatch) -> None:
    """护栏含贴线（2026-08-29 id=100 实证）：报价均价=护栏价时弃单，
    不提交 slippageBps=0（币安 -1102 拒收）；低于护栏正常下单。"""
    trader = _make_real_trader(monkeypatch)
    placed: list[int] = []
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(token_id, side, amount_usdt=None):
        return {"averagePrice": 0.75, "amountIn": "5", "amountOut": "6",
                "quoteId": "Q-BND"}

    async def _place(_q, slippage_bps=1200):
        placed.append(slippage_bps)
        return {"orderId": "ORD-BND"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)

    # 贴线：0.75 >= 0.75 → FAILED，不下单
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust_confirm", WINDOW_START,
        max_exec_price=0.75, market_period="15m", scene_signal_id=1)
    assert order["status"] == "FAILED"
    assert "贴线无滑点空间" in str(updates[-1][1]["error_message"])
    assert placed == []

    # 低于护栏（有滑点空间）→ 正常下单，滑点按护栏价收紧（0.74→0.75 ≈ 135bps）
    async def _quote_ok(token_id, side, amount_usdt=None):
        return {"averagePrice": 0.74, "amountIn": "5", "amountOut": "6",
                "quoteId": "Q-BND2"}

    monkeypatch.setattr(trader, "get_quote", _quote_ok)
    trader._15m_markets[WINDOW_START + 300_000] = {
        "end_date": WINDOW_START + 1200_000,
        "up_token": "T-B2-UP", "down_token": "T-B2-DOWN",
        "up_price": 0.26, "down_price": 0.74,
    }
    order2 = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust_confirm", WINDOW_START + 300_000,
        max_exec_price=0.75, market_period="15m", scene_signal_id=2)
    assert order2["status"] == "FILLED"
    assert placed == [135]  # int((0.75/0.74−1)×10000)=135，非 0/非默认 1200


@pytest.mark.asyncio
async def test_signal_trade_s1_guard_070_calibration(monkeypatch) -> None:
    """S1 护栏校准后默认 0.70（2026-08-30 盘口数据：旧 0.60 误拦胜率 75% 的入场价段）：
    旧护栏误拦区间 [0.60,0.70) 内的报价 0.65 正常成交，贴线 0.70 仍弃单。"""
    trader = _make_real_trader(monkeypatch)
    guard = resolve_max_exec(LIVE_CHANNELS["scene_bull_exhaust"], ChannelConfig())
    assert guard == 0.70  # 无显式配置时回落新默认值，非旧 0.60
    placed: list[int] = []
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _place(_q, slippage_bps=1200):
        placed.append(slippage_bps)
        return {"orderId": "ORD-S1G"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "place_order", _place)

    # 报价 0.65：旧 0.60 护栏会拦（2026-08-29 id=99 实证），新护栏放行成交，
    # 滑点按护栏价收紧 int((0.70/0.65−1)×10000)=769bps
    async def _q1(token_id, side, amount_usdt=None):
        return {"averagePrice": 0.65, "amountIn": "5", "amountOut": "7",
                "quoteId": "Q-S1G1"}

    monkeypatch.setattr(trader, "get_quote", _q1)
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START,
        max_exec_price=guard, market_period="15m", scene_signal_id=1)
    assert order["status"] == "FILLED"
    assert placed == [769]

    # 贴线：0.70 >= 0.70 → FAILED，不下单
    async def _q2(token_id, side, amount_usdt=None):
        return {"averagePrice": 0.70, "amountIn": "5", "amountOut": "7",
                "quoteId": "Q-S1G2"}

    monkeypatch.setattr(trader, "get_quote", _q2)
    trader._15m_markets[WINDOW_START + 300_000] = {
        "end_date": WINDOW_START + 1200_000,
        "up_token": "T-S1G-UP", "down_token": "T-S1G-DOWN",
        "up_price": 0.30, "down_price": 0.70,
    }
    order2 = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START + 300_000,
        max_exec_price=guard, market_period="15m", scene_signal_id=2)
    assert order2["status"] == "FAILED"
    assert "贴线无滑点空间" in str(updates[-1][1]["error_message"])
    assert len(placed) == 1  # 第二单未下到 place_order


@pytest.mark.asyncio
async def test_signal_trade_s2_guard_065_calibration(monkeypatch) -> None:
    """S2 护栏校准后默认 0.65（2026-08-30 盘口数据：旧 0.55 拦掉 47% 信号，
    被拦段胜率 88%/EV+0.39）：旧误拦区间 [0.55,0.65) 报价 0.60 正常成交，
    贴线 0.65 仍弃单。S2 押 UP（与 S1 方向相反，验证 UP token 分流）。"""
    trader = _make_real_trader(monkeypatch)
    guard = resolve_max_exec(LIVE_CHANNELS["scene_bear_exhaust"], ChannelConfig())
    assert guard == 0.65  # 无显式配置时回落新默认值，非旧 0.55
    placed: list[int] = []
    quote_tokens: list[str] = []
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _place(_q, slippage_bps=1200):
        placed.append(slippage_bps)
        return {"orderId": "ORD-S2G"}

    async def _q1(token_id, side, amount_usdt=None):
        quote_tokens.append(token_id)
        return {"averagePrice": 0.60, "amountIn": "5", "amountOut": "8",
                "quoteId": "Q-S2G1"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "get_quote", _q1)

    # 报价 0.60：旧 0.55 护栏会拦，新护栏放行成交；押 UP → 取 up_token，
    # 滑点 int((0.65/0.60−1)×10000)=833bps
    order = await trader.execute_signal_trade(
        "UP", 5.0, "scene_bear_exhaust", WINDOW_START,
        max_exec_price=guard, market_period="15m", scene_signal_id=1)
    assert order["status"] == "FILLED"
    assert placed == [833]
    assert quote_tokens and quote_tokens[0].endswith("UP")  # UP 方向取 up_token

    # 贴线：0.65 >= 0.65 → FAILED，不下单
    async def _q2(token_id, side, amount_usdt=None):
        return {"averagePrice": 0.65, "amountIn": "5", "amountOut": "8",
                "quoteId": "Q-S2G2"}

    monkeypatch.setattr(trader, "get_quote", _q2)
    trader._15m_markets[WINDOW_START + 300_000] = {
        "end_date": WINDOW_START + 1200_000,
        "up_token": "T-S2G-UP", "down_token": "T-S2G-DOWN",
        "up_price": 0.65, "down_price": 0.35,
    }
    order2 = await trader.execute_signal_trade(
        "UP", 5.0, "scene_bear_exhaust", WINDOW_START + 300_000,
        max_exec_price=guard, market_period="15m", scene_signal_id=2)
    assert order2["status"] == "FAILED"
    assert "贴线无滑点空间" in str(updates[-1][1]["error_message"])
    assert len(placed) == 1  # 第二单未下到 place_order


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
    # 2026-08-28 FILLED 回填：averagePrice=实际成交价（历史行 price 0.55），
    # 原报价预估均价 0.71 保留在 quotedAvgPrice，来源标记 binance_history_confirm
    assert updates[0][1]["quote_json"]["averagePrice"] == 0.55
    assert updates[0][1]["quote_json"]["quotedAvgPrice"] == 0.71
    assert updates[0][1]["quote_json"]["fillSource"] == "binance_history_confirm"
    # 默认回查行无 filledUsdtAmount → amountIn 保持报价口径不做半吊子回填
    assert updates[0][1]["amount_in"] == "5"


def test_merge_fill_partial_fill_amount() -> None:
    """部分成交订正（2026-08-28）：amountIn ← filledUsdtAmount，
    携带 filledShareQty（扣 marketProviderFee 后的实际到手股数）。

    FOK 只吃到部分深度时实际成交 < 请求金额；不订正则结算按请求金额
    估算赢向盈亏虚高（生产实锤：请求 3U 只成交 2U，15.75 虚高 vs 实际 10.28）；
    费用以少给股数体现（12.5 → 12.28 股），赢向按股数算才对得上币安实现盈亏。
    """
    quote = {"averagePrice": 0.71, "amountIn": "5000000000000000000", "quoteId": "Q1"}
    merged = BinancePredictionTrader._merge_fill_into_quote(
        quote, {"orderId": "O1", "status": "FILLED", "price": "0.55",
                "filledUsdtAmount": "2", "filledShareQty": "12.28"})
    assert merged["averagePrice"] == 0.55
    assert merged["amountIn"] == "2000000000000000000"
    assert merged["requestedAmountIn"] == "5000000000000000000"
    assert merged["filledShareQty"] == 12.28
    # filledUsdtAmount 缺失/零/非法 → 保持报价金额（无可靠数据不做半吊子回填）
    for bad in (None, "0", "abc"):
        row = {"price": "0.55", "filledUsdtAmount": bad}
        unchanged = BinancePredictionTrader._merge_fill_into_quote(quote, row)
        assert unchanged["amountIn"] == "5000000000000000000"
        assert "requestedAmountIn" not in unchanged
    # filledShareQty 缺失/零/非法 → 不携带（结算回退无费估算口径）
    for bad in (None, "0", "abc"):
        row = {"price": "0.55", "filledUsdtAmount": "2", "filledShareQty": bad}
        no_shares = BinancePredictionTrader._merge_fill_into_quote(quote, row)
        assert "filledShareQty" not in no_shares


@pytest.mark.asyncio
async def test_signal_trade_ghost_fill_downgrade(monkeypatch) -> None:
    """币安侧终态回查为 FAILED（FOK 受理后翻转）→ 重试仍 FAILED → 落 FAILED。

    2026-08-30 起：终态 FAILED 会触发同槽位重试（最多 MAX_FOK_RETRIES 次）；
    本用例回查恒 FAILED → 共 3 次下单（首次+2 重试），最终仍不伪造成交。
    """
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[str] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        return {"averagePrice": 0.55, "amountIn": "5", "amountOut": "9",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        place_calls.append(_q["quoteId"])
        return {"orderId": "ORD-GHOST"}

    async def _confirm_failed(_oid, attempts=3, delay=1.0):
        return {"orderId": _oid, "status": "FAILED"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm_failed)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"
    assert len(place_calls) == 3  # 首次 + 2 次重试（重试报价同价仍过护栏）
    assert updates[0][0] == "FAILED"
    assert updates[0][1]["order_id"] == "ORD-GHOST"
    assert "FOK 未成交" in updates[0][1]["error_message"]
    assert "已重试 2 次" in updates[0][1]["error_message"]


@pytest.mark.asyncio
async def test_signal_trade_fok_retry_second_attempt_fills(monkeypatch) -> None:
    """FOK 首轮未成交 → 重新报价 0.52 重试（滑点放宽到护栏全空间）→ 成交。

    2026-08-30 S1 实证：0.49 报价 / 0.6 护栏，首轮滑点钳在 1200bps（限价只到 0.55）
    吃不满；重试轮滑点 = (0.6/0.52−1)=1538bps 不再钳 1200，覆盖到护栏价。
    """
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    slippage_seen: list[int] = []
    quotes = [
        {"averagePrice": 0.49, "amountIn": "5", "amountOut": "10", "quoteId": "Q1"},
        {"averagePrice": 0.52, "amountIn": "5", "amountOut": "9", "quoteId": "Q2"},
    ]
    places = [{"orderId": "ORD-1"}, {"orderId": "ORD-2"}]
    confirms = [
        {"orderId": "ORD-1", "status": "FAILED"},
        {"orderId": "ORD-2", "status": "FILLED", "price": "0.58"},
    ]

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        return quotes.pop(0)

    async def _place(_q, slippage_bps=1200):
        slippage_seen.append(slippage_bps)
        return places.pop(0)

    async def _confirm(_oid, attempts=3, delay=1.0):
        return next(c for c in confirms if c["orderId"] == _oid)

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START, max_exec_price=0.6)
    assert order["status"] == "FILLED"
    assert order["order_id"] == "ORD-2"  # 落库为重试单，非首次被拒单
    # 首轮钳 1200（护栏空间 2244）；重试轮放宽到护栏全空间 1538 不再钳 1200
    assert slippage_seen == [1200, 1538]
    # FILLED 回填口径：成交价取币安历史行 price，重试报价均价保留在 quotedAvgPrice
    assert updates[0][1]["quote_json"]["averagePrice"] == 0.58
    assert updates[0][1]["quote_json"]["quotedAvgPrice"] == 0.52


@pytest.mark.asyncio
async def test_signal_trade_fok_retry_guardrail_rejects_retry_quote(monkeypatch) -> None:
    """重试轮重新报价已涨破护栏（0.62 ≥ 0.6）→ 弃单不追贵，不再下第二单。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[str] = []
    quotes = [
        {"averagePrice": 0.49, "amountIn": "5", "amountOut": "10", "quoteId": "Q1"},
        {"averagePrice": 0.62, "amountIn": "5", "amountOut": "8", "quoteId": "Q2"},
    ]

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        return quotes.pop(0)

    async def _place(_q, slippage_bps=1200):
        place_calls.append(_q["quoteId"])
        return {"orderId": "ORD-1"}

    async def _confirm(_oid, attempts=3, delay=1.0):
        return {"orderId": _oid, "status": "FAILED"}

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "scene_bull_exhaust", WINDOW_START, max_exec_price=0.6)
    assert order["status"] == "FAILED"
    assert place_calls == ["Q1"]  # 护栏复检拦下重试单，未下第二单
    assert "重试执行价护栏弃单" in updates[0][1]["error_message"]
    assert updates[0][1]["order_id"] == "ORD-1"  # 首次单 orderId 保留供对账追溯


# ------------------------------------------------------------
# 入场价白名单（x4_v3 下单层主护栏，entry_band_whitelist 透传）
# ------------------------------------------------------------

_V3_BANDS = ((0.0, 0.2), (0.3, 0.4))   # 与 live_channels x4_v3 冻结口径一致


def _stub_whitelist_trade(monkeypatch, trader, quotes, confirms):
    """白名单用例替身装配：报价依次弹出、下单/币安回查按表，返回 (updates, place_calls)。"""
    updates: list[tuple] = []
    place_calls: list[str] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        return quotes.pop(0)

    async def _place(_q, slippage_bps=1200):
        place_calls.append(_q["quoteId"])
        return {"orderId": "ORD-1"}

    async def _confirm(_oid, attempts=3, delay=1.0):
        return next(c for c in confirms if c["orderId"] == _oid)

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm)
    return updates, place_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("avg,inside", [
    (0.05, True),    # [0, 0.2) 带内
    (0.19, True),    # 上界内侧（左闭右开）
    (0.20, False),   # 上界贴线（0.2 不含）
    (0.25, False),   # 两带间隙
    (0.30, True),    # 第二带下界（左闭）
    (0.40, False),   # 第二带上界贴线
    (0.49, False),   # 护栏内但带外
])
async def test_signal_trade_whitelist_band_matrix(monkeypatch, avg, inside) -> None:
    """x4_v3 白名单边界矩阵：[0,0.2)∪[0.3,0.4)，带外弃单且币安零请求。"""
    trader = _make_real_trader(monkeypatch)
    updates, place_calls = _stub_whitelist_trade(
        monkeypatch, trader,
        quotes=[{"averagePrice": avg, "amountIn": "5", "amountOut": "10",
                 "quoteId": "Q1"}],
        confirms=[{"orderId": "ORD-1", "status": "FILLED", "price": "0.55"}])
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "x4_v3", WINDOW_START, max_exec_price=0.50,
        market_period="5m", entry_band_whitelist=_V3_BANDS)
    if inside:
        assert order["status"] == "FILLED"
        assert place_calls == ["Q1"]
    else:
        assert order["status"] == "FAILED"
        assert "入场价白名单弃单" in updates[0][1]["error_message"]
        assert place_calls == []   # 下单前拦截：不向币安提交


@pytest.mark.asyncio
async def test_signal_trade_whitelist_guardrail_precedes_band(monkeypatch) -> None:
    """avg=0.50 贴护栏线：执行价护栏先拦（文案护栏非白名单），白名单不触达。"""
    trader = _make_real_trader(monkeypatch)
    updates, place_calls = _stub_whitelist_trade(
        monkeypatch, trader,
        quotes=[{"averagePrice": 0.50, "amountIn": "5", "amountOut": "10",
                 "quoteId": "Q1"}],
        confirms=[])
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "x4_v3", WINDOW_START, max_exec_price=0.50,
        market_period="5m", entry_band_whitelist=_V3_BANDS)
    assert order["status"] == "FAILED"
    assert "执行价护栏弃单" in updates[0][1]["error_message"]
    assert place_calls == []


@pytest.mark.asyncio
async def test_signal_trade_whitelist_nonpositive_avg_failsafe(monkeypatch) -> None:
    """avg=0（无护栏时）→ 白名单 fail-safe 恒 False 弃单（纯函数独立防线）。"""
    trader = _make_real_trader(monkeypatch)
    updates, place_calls = _stub_whitelist_trade(
        monkeypatch, trader,
        quotes=[{"averagePrice": 0, "amountIn": "5", "amountOut": "10",
                 "quoteId": "Q1"}],
        confirms=[])
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "x4_v3", WINDOW_START,      # max_exec_price=None：护栏整段跳过
        market_period="5m", entry_band_whitelist=_V3_BANDS)
    assert order["status"] == "FAILED"
    assert "入场价白名单弃单" in updates[0][1]["error_message"]
    assert place_calls == []


@pytest.mark.asyncio
async def test_signal_trade_fok_retry_whitelist_rechecks_retry_quote(monkeypatch) -> None:
    """首报 0.19 过白名单 → FOK 未成交；重试报价 0.22 跳出带 → 复检拦，不下第二单。"""
    trader = _make_real_trader(monkeypatch)
    updates, place_calls = _stub_whitelist_trade(
        monkeypatch, trader,
        quotes=[
            {"averagePrice": 0.19, "amountIn": "5", "amountOut": "10", "quoteId": "Q1"},
            {"averagePrice": 0.22, "amountIn": "5", "amountOut": "9", "quoteId": "Q2"},
        ],
        confirms=[{"orderId": "ORD-1", "status": "FAILED"}])
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "x4_v3", WINDOW_START, max_exec_price=0.50,
        market_period="5m", entry_band_whitelist=_V3_BANDS)
    assert order["status"] == "FAILED"
    assert "重试入场价白名单弃单" in updates[0][1]["error_message"]
    assert place_calls == ["Q1"]                  # 复检拦下重试单
    assert updates[0][1]["order_id"] == "ORD-1"   # 首单 id 保留供对账追溯


@pytest.mark.asyncio
async def test_signal_trade_whitelist_none_keeps_legacy_behavior(monkeypatch) -> None:
    """bands=None（x4_v2 等未配置通道）恒 True：0.25 不在 v3 带内也照常成交。"""
    trader = _make_real_trader(monkeypatch)
    updates, place_calls = _stub_whitelist_trade(
        monkeypatch, trader,
        quotes=[{"averagePrice": 0.25, "amountIn": "5", "amountOut": "10",
                 "quoteId": "Q1"}],
        confirms=[{"orderId": "ORD-1", "status": "FILLED", "price": "0.25"}])
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "x4_v2", WINDOW_START, max_exec_price=0.50,
        market_period="5m")   # 不传 entry_band_whitelist（默认 None）
    assert order["status"] == "FILLED"
    assert place_calls == ["Q1"]


@pytest.mark.asyncio
async def test_signal_trade_unconfirmed_stays_pending(monkeypatch) -> None:
    """币安侧终态暂未确认（回查 None）→ 落 PENDING 交对账，不当作成交。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []

    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        return {"averagePrice": 0.55, "amountIn": "5", "amountOut": "9",
                "quoteId": "Q1"}

    async def _place(_q, slippage_bps=1200):
        return {"orderId": "ORD-UNCONFIRMED"}

    async def _confirm_none(_oid, attempts=3, delay=1.0):
        return None  # None=币安历史暂无该单，语义不变（落 PENDING 交对账）

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm_none)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_momentum_v1", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "PENDING"
    assert updates[0][0] == "PENDING"
    assert updates[0][1]["order_id"] == "ORD-UNCONFIRMED"


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


# ============================================================
# 组 10：15m 周期边界空窗修复（2026-08-30 S1 漏单：A 后台预热 + B 锚点回退）
# ============================================================

class _FakeHttpResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpClient:
    """market/list + market/detail 双端点 HTTP 桓（记录 detail 扫描轨迹）。"""

    def __init__(self, list_topics: list, details: dict) -> None:
        self._list_topics = list_topics
        self._details = details
        self.detail_tids: list[int] = []

    async def get(self, url, params=None, headers=None):
        if url.endswith("/market/list"):
            return _FakeHttpResp({"marketTopics": self._list_topics})
        tid = int((params or {}).get("marketTopicId", -1))
        self.detail_tids.append(tid)
        d = self._details.get(tid)
        return _FakeHttpResp(d, 200 if d is not None else 404)


def _detail_trader(monkeypatch, client: _FakeHttpClient) -> BinancePredictionTrader:
    trader = BinancePredictionTrader()
    monkeypatch.setattr(trader, "_sign_request", lambda p: p)
    monkeypatch.setattr(trader, "_get_client", lambda: client)
    return trader


@pytest.mark.asyncio
async def test_market_warmup_loop_refreshes_periodically(monkeypatch) -> None:
    """后台预热循环：定时调 refresh_markets 预缓存未来 15m 周期（修复 A）；
    刷新异常只告警不打断循环（下轮照常预热）。"""
    fake = _FakeTrader()
    calls: list[int] = []

    async def _refresh():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("list 超时")

    fake.refresh_markets = _refresh
    t = _make_trader(monkeypatch, fake)
    monkeypatch.setattr(milt, "MARKET_WARMUP_INTERVAL_S", 0.005)
    t._running = True
    task = asyncio.create_task(t._market_warmup_loop())
    await asyncio.sleep(0.05)
    t._running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(calls) >= 2   # 首轮异常后循环仍继续（未被打断）


@pytest.mark.asyncio
async def test_start_registers_market_warmup_task(monkeypatch) -> None:
    """start() 注册预热循环为辅助任务；stop() 将其取消并清场。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    await t.start()
    names = {task.get_name() for task in t._tasks}
    assert "multi_live_market_warmup" in names
    await t.stop()
    assert t._stopped is True


@pytest.mark.asyncio
async def test_refresh_markets_holds_trade_lock(monkeypatch) -> None:
    """refresh_markets 持下单锁刷新（与下单临界区互斥，防预热途中覆写 token）。"""
    trader = BinancePredictionTrader()
    locked_seen: list[bool] = []

    async def _list():
        locked_seen.append(trader._trade_lock.locked())

    monkeypatch.setattr(trader, "list_markets", _list)
    await trader.refresh_markets()
    assert locked_seen == [True]


@pytest.mark.asyncio
async def test_detail_prefetch_fallback_anchor_backward_scan(monkeypatch) -> None:
    """列表无同周期锚点（15m 整体空窗，2026-08-30 事故形态）→ 回退 5m
    市场 tid 双向扫描；目标在锚点之前（tid 更小）也能命中（修复 B）。"""
    slug_want = f"btc-updown-15m-{MARKET_START_15M // 1000}"
    target_tid = 990   # 在 5m 锚点 1000 之前（目标比锚点先创建的场景）
    client = _FakeHttpClient(
        list_topics=[{"chartType": "CRYPTO_UP_DOWN", "symbol": "BTCUSDT",
                      "title": "", "slug": "btc-updown-5m-999",
                      "marketTopicId": 1000}],
        details={target_tid: {
            "slug": slug_want,
            "startDate": MARKET_START_15M,
            "endDate": MARKET_START_15M + 900_000,
            "markets": [{"tradingStatus": "OPEN", "outcomes": [
                {"name": "UP", "tokenId": "T-FB-UP", "price": 0.4},
                {"name": "DOWN", "tokenId": "T-FB-DOWN", "price": 0.6},
            ]}],
        }},
    )
    trader = _detail_trader(monkeypatch, client)

    entry = await trader._fetch_market_via_detail("15m", MARKET_START_15M)

    assert entry is not None
    assert entry["down_token"] == "T-FB-DOWN"
    assert entry["up_token"] == "T-FB-UP"
    assert target_tid in client.detail_tids   # 后向扫描覆盖到锚点之前的 ID


@pytest.mark.asyncio
async def test_detail_prefetch_no_anchor_gives_up(monkeypatch) -> None:
    """列表完全空窗（5m/15m 锚点都没有）→ 返回 None 不抛（等后台预热/下次开火）。"""
    client = _FakeHttpClient(list_topics=[], details={})
    trader = _detail_trader(monkeypatch, client)

    assert await trader._fetch_market_via_detail("15m", MARKET_START_15M) is None
    assert client.detail_tids == []   # 无锚点不扫描，不烧调用预算


# ============================================================
# 组 9：quote_momentum_v3 非连涨门禁 + s5_deep 通道 + 同窗互斥（2026-09-02）
# ============================================================

V3_STREAK = "quote_momentum_v3"
S5_DEEP = "s5_deep_z20_v1"
# _pass_streak_guard ex-ante 红线：bar_start=(quote_ts//900_000−1)*900_000
_STREAK_BAR = ((WINDOW_START + 100_000) // 900_000 - 1) * 900_000


def _k15(close_prev: float, close_bar: float) -> list[dict]:
    """15m K 线两根：bar_start 前一根 + 末收根（覆盖门禁定位所需）。"""
    return [
        {"open_time": _STREAK_BAR - 900_000, "close": close_prev},
        {"open_time": _STREAK_BAR, "close": close_bar},
    ]


def _deep_sig(sig_id: int = 91) -> dict:
    """S5 深档钩子 payload 替身（z5≤−20bp 深回落子集）。"""
    return {"id": sig_id, "pattern_type": S5_DEEP, "side": "high",
            "market_start_15m": MARKET_START_15M,
            "market_end_15m": MARKET_START_15M + 900_000,
            "z5": -0.0025, "down_quote": 0.85}


@pytest.mark.asyncio
async def test_v3_streak_gate_fires_on_pass(monkeypatch) -> None:
    """v3：非连涨核验通过 → 下单；同窗后续采样不重复派生。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V3_STREAK])

    async def _ok(self, channel, quote_ts):
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _ok)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) \
        == [V3_STREAK]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [V3_STREAK]
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 105_000, 0.72) == []
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_v3_streak_gate_rejects(monkeypatch) -> None:
    """v3：连涨 → 同步预检已过、核验弃单；同窗不再重试。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V3_STREAK])

    async def _reject(self, channel, quote_ts):
        return False, "streak_reject"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _reject)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) \
        == [V3_STREAK]
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_v3_streak_klines_missing_retry_once(monkeypatch) -> None:
    """v3：K 线拉取失败 → 延迟重查一次；仍缺失 → 弃单；重查通过 → 下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[V3_STREAK])
    calls = {"n": 0}

    async def _missing(self, channel, quote_ts):
        calls["n"] += 1
        return False, "klines_missing"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _missing)
    monkeypatch.setattr(milt, "STREAK_RETRY_DELAY_S", 0.0)
    assert t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71) \
        == [V3_STREAK]
    await _drain(t)
    assert calls["n"] == 2 and fake.calls == []

    # 重查通过场景：首次 K 线拉取失败，重试后恢复
    fake2 = _FakeTrader()
    t2 = _make_trader(monkeypatch, fake2, channels=[V3_STREAK])
    attempts = {"n": 0}

    async def _flaky(self, channel, quote_ts):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return False, "klines_missing"
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _flaky)
    monkeypatch.setattr(milt, "STREAK_RETRY_DELAY_S", 0.0)
    t2.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    await _drain(t2)
    assert attempts["n"] == 2
    assert [c["signal_version"] for c in fake2.calls] == [V3_STREAK]


@pytest.mark.asyncio
async def test_check_streak_branches(monkeypatch) -> None:
    """_check_streak：非连涨过 / 连涨拒 / 缺根不可判保守拒 / fetcher 缺失或空
    保守拒 / 未注册门禁通道拒（与影子 _pass_streak_guard 同口径）。"""
    t = _make_trader(monkeypatch, _FakeTrader())
    quote_ts = WINDOW_START + 100_000

    async def _fetch(interval, limit):
        return _k15(100.0, 99.0)      # close[j] ≤ close[j-1] → 非连涨

    t.kline_fetcher = _fetch
    assert await t._check_streak(V3_STREAK, quote_ts) == (True, "ok")

    async def _fetch_up(interval, limit):
        return _k15(100.0, 101.0)     # 连涨 → 拒

    t.kline_fetcher = _fetch_up
    assert await t._check_streak(V3_STREAK, quote_ts) == (False, "streak_reject")

    async def _fetch_gap(interval, limit):
        return [{"open_time": _STREAK_BAR, "close": 99.0}]  # 前一根缺失 → 不可判

    t.kline_fetcher = _fetch_gap
    assert await t._check_streak(V3_STREAK, quote_ts) == (False, "streak_unknown")

    async def _fetch_empty(interval, limit):
        return []

    t.kline_fetcher = _fetch_empty
    assert await t._check_streak(V3_STREAK, quote_ts) == (False, "klines_missing")

    t.kline_fetcher = None
    assert await t._check_streak(V3_STREAK, quote_ts) == (False, "klines_missing")
    assert await t._check_streak("quote_momentum_v1", quote_ts) \
        == (False, "no_streak_guard")


@pytest.mark.asyncio
async def test_s5_deep_hook_fires(monkeypatch) -> None:
    """S5 深档钩子：+5min 确认即 15m 市场押 DOWN，护栏 0.88，scene_signal_id 溯源；同周期防重。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[S5_DEEP])
    t.on_s5_deep_signal(_deep_sig())
    await _drain(t)
    call = fake.calls[0]
    assert call["prediction"] == "DOWN"
    assert call["signal_version"] == S5_DEEP
    assert call["window_start"] == MARKET_START_15M
    assert call["market_period"] == "15m"
    assert call["scene_signal_id"] == 91
    assert call["max_exec_price"] == 0.88
    # 同周期重复回调 → 至多一单
    t.on_s5_deep_signal(_deep_sig(sig_id=92))
    await _drain(t)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_s5_deep_disabled_no_fire(monkeypatch) -> None:
    """S5 深档未启用 → 零下单（影子落表继续，互不干扰）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[])
    t.on_s5_deep_signal(_deep_sig())
    await _drain(t)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_exclusive_confirm_filled_blocks_deep(monkeypatch) -> None:
    """同窗互斥：S5 确认已成交 → 深档同窗弃单（组内每窗至多一单成交）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "scene_bull_exhaust_confirm", S5_DEEP])
    t.on_scene_signal(_scene_sig("bull_exhaust_confirm", "high", sig_id=77))
    await _drain(t)
    t.on_s5_deep_signal(_deep_sig())
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "scene_bull_exhaust_confirm"]


@pytest.mark.asyncio
async def test_exclusive_unfilled_does_not_block(monkeypatch) -> None:
    """同窗互斥：未成交不占坑——确认单失败后深档同窗照常下单（互补价格带）。"""
    fake = _FakeTrader(result=_fake_order("FAILED"))
    t = _make_trader(monkeypatch, fake, channels=[
        "scene_bull_exhaust_confirm", S5_DEEP])
    t.on_scene_signal(_scene_sig("bull_exhaust_confirm", "high", sig_id=77))
    await _drain(t)
    t.on_s5_deep_signal(_deep_sig())
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [
        "scene_bull_exhaust_confirm", S5_DEEP]


@pytest.mark.asyncio
async def test_exclusive_momentum_same_window_one_fill(monkeypatch) -> None:
    """同窗互斥：v2/v3 同窗命中，v2 先成交占坑 → v3 弃单（防叠加敞口）。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "quote_momentum_v2", V3_STREAK])

    async def _ok(self, channel, quote_ts):
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _ok)
    # v2 min_drop 门禁过（chg −0.2% ≤ −0.10%）；q=0.71 同命中 v1 冻结区间（v2/v3 共享）
    fired = t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71,
                    btc_price=99.8, window_entry_price=100.0)
    assert sorted(fired) == ["quote_momentum_v2", V3_STREAK]
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == ["quote_momentum_v2"]


@pytest.mark.asyncio
async def test_exclusive_momentum_v2_blocked_v3_fires(monkeypatch) -> None:
    """同窗互斥：v2 门禁弃单（BTC 缺失）不占坑 → v3 核验通过下单。"""
    fake = _FakeTrader()
    t = _make_trader(monkeypatch, fake, channels=[
        "quote_momentum_v2", V3_STREAK])

    async def _ok(self, channel, quote_ts):
        return True, "ok"

    monkeypatch.setattr(MultiLiveTrader, "_check_streak", _ok)
    fired = t.check(WINDOW_START, WINDOW_END, WINDOW_START + 100_000, 0.71)
    assert fired == [V3_STREAK]     # v2 门禁数据缺失，同步预检即排除
    await _drain(t)
    assert [c["signal_version"] for c in fake.calls] == [V3_STREAK]


# ============================================================
# 组 9：P0 终态回查竞态（confirm_order_status / _confirm_with_backfill）
# 生产实证 id=285（2026-09-04 13:00）：币安 createTime→terminalTime 耗时
# 2.967s，旧轮询窗口 3×1.0s≈2.4s 差 471ms 返回 None → 落 PENDING 且
# FOK 重试入口因 `confirmed is not None` 短路，一次重试都没跑。
# ============================================================

def _bare_trader() -> BinancePredictionTrader:
    """不预置任何桩的 trader（终态回查/余额预检类用例自行按需 stub）。"""
    t = BinancePredictionTrader()
    t._api_key = "k"
    t._api_secret = "s"
    t._wallet_address = "0xWALLET"
    t._wallet_id = "WID"
    return t


def test_confirm_terminal_status_set() -> None:
    """终态集合形状：EXPIRED/CANCELLED 已纳入；可重试集为终态真子集且不含 FILLED。"""
    assert TERMINAL_ORDER_STATUSES == {"FILLED", "FAILED", "EXPIRED", "CANCELLED"}
    assert RETRYABLE_ORDER_STATUSES == {"FAILED", "EXPIRED", "CANCELLED"}
    assert RETRYABLE_ORDER_STATUSES < TERMINAL_ORDER_STATUSES


def test_confirm_poll_window_covers_production_latency() -> None:
    """常量守卫：轮询窗口必须覆盖生产实测 2.967s 终态延迟并留足余量。

    防回退：若将来有人把 attempts/delay 改回旧值（窗口 <2.967s），id=285
    那类竞态会静默复发（表现为偏尔 PENDING 卡单），此处直接拦下。
    """
    regular = (CONFIRM_ATTEMPTS - 1) * CONFIRM_DELAY_S
    deep = (CONFIRM_DEEP_ATTEMPTS - 1) * CONFIRM_DEEP_DELAY_S
    assert regular >= 6.0, f"常规轮询窗口 {regular}s 不足（生产 2.967s + 余量）"
    assert deep >= 6.0, f"深轮询窗口 {deep}s 不足"
    assert regular + deep >= 12.0          # 两轮兜底总窗口


@pytest.mark.asyncio
async def test_confirm_terminal_includes_expired_cancelled(monkeypatch) -> None:
    """EXPIRED/CANCELLED 也是终态 → 首轮即命中（旧实现会轮询到超时误报 None）。"""
    trader = _bare_trader()
    for st in sorted(TERMINAL_ORDER_STATUSES):
        async def _hist(limit=20, _st=st):
            return [{"orderId": "O-1", "status": _st, "price": "0.53"}]

        monkeypatch.setattr(trader, "query_order_history", _hist)
        row = await trader.confirm_order_status("O-1", attempts=1)
        assert row is not None and row["status"] == st, f"{st} 应立即判为终态"


@pytest.mark.asyncio
async def test_confirm_non_terminal_keeps_polling(monkeypatch) -> None:
    """反例：NEW 等非终态不得提前返回（继续轮询到超时 → None，不伪造终态）。"""
    trader = _bare_trader()
    polls: list[int] = []

    async def _hist(limit=20):
        polls.append(limit)
        return [{"orderId": "O-1", "status": "NEW"}]

    monkeypatch.setattr(trader, "query_order_history", _hist)
    assert await trader.confirm_order_status("O-1", attempts=3, delay=0.0) is None
    assert len(polls) == 3                 # 每轮都查，不提前退出

    polls.clear()
    assert await trader.confirm_order_status(None, attempts=3, delay=0.0) is None
    assert polls == []                     # orderId 为空 → 不发请求


@pytest.mark.asyncio
async def test_confirm_slow_terminal_caught_by_widened_window(monkeypatch) -> None:
    """竞态回归：终态在第 4 轮才出现 → 新窗口（5 轮）命中；旧窗口（3 轮）必漏。"""
    trader = _bare_trader()
    n = {"i": 0}

    async def _hist(limit=20):
        n["i"] += 1
        if n["i"] < 4:                     # 前 3 轮币安侧仍 NEW（旧窗口在此放弃）
            return [{"orderId": "O-1", "status": "NEW"}]
        return [{"orderId": "O-1", "status": "FAILED", "price": "0.53"}]

    monkeypatch.setattr(trader, "query_order_history", _hist)
    row = await trader.confirm_order_status(
        "O-1", attempts=CONFIRM_ATTEMPTS, delay=0.0)
    assert row is not None and row["status"] == "FAILED"
    assert n["i"] == 4

    # 旧口径（3 轮）在同一序列下必然漏 → 证明放宽是必要的，不是无意义加大
    n["i"] = 0
    assert await trader.confirm_order_status("O-1", attempts=3, delay=0.0) is None


@pytest.mark.asyncio
async def test_confirm_with_backfill_falls_back_to_deep_poll(monkeypatch) -> None:
    """常规轮询未命中 → 自动转深轮询（参数用 DEEP 常量）兜底命中。"""
    trader = _bare_trader()
    seen: list[tuple[int, float]] = []
    calls = {"i": 0}

    async def _confirm(oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        seen.append((attempts, delay))
        calls["i"] += 1
        return None if calls["i"] == 1 else {"orderId": oid, "status": "FAILED"}

    monkeypatch.setattr(trader, "confirm_order_status", _confirm)
    row = await trader._confirm_with_backfill("O-9")
    assert row["status"] == "FAILED"
    assert seen == [(CONFIRM_ATTEMPTS, CONFIRM_DELAY_S),
                    (CONFIRM_DEEP_ATTEMPTS, CONFIRM_DEEP_DELAY_S)]


@pytest.mark.asyncio
async def test_confirm_with_backfill_short_circuits_on_hit(monkeypatch) -> None:
    """常规轮询已命中 → 不进深轮询（不白多等 ≈6s 拖慢下单回路）。"""
    trader = _bare_trader()
    seen: list[tuple[int, float]] = []

    async def _confirm(oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        seen.append((attempts, delay))
        return {"orderId": oid, "status": "FILLED"}

    monkeypatch.setattr(trader, "confirm_order_status", _confirm)
    assert (await trader._confirm_with_backfill("O-9"))["status"] == "FILLED"
    assert seen == [(CONFIRM_ATTEMPTS, CONFIRM_DELAY_S)]


@pytest.mark.asyncio
async def test_confirm_with_backfill_two_rounds_none(monkeypatch) -> None:
    """两轮都未命中 → None（调用方落 PENDING 交对账，不伪造成交）。"""
    trader = _bare_trader()
    calls: list[int] = []

    async def _confirm(oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        calls.append(attempts)
        return None

    monkeypatch.setattr(trader, "confirm_order_status", _confirm)
    assert await trader._confirm_with_backfill("O-9") is None
    assert calls == [CONFIRM_ATTEMPTS, CONFIRM_DEEP_ATTEMPTS]


def _exec_stubs(monkeypatch, trader, updates: list, *,
                quote=None, place=None, confirm=None, balance=None,
                quote_calls: list | None = None,
                place_calls: list | None = None) -> None:
    """execute_signal_trade 公用桩（组 9/10 共用）：占位/更新/报价/下单/回查/预检。"""
    async def _reserve(_v, _ws, direction=None, market_period="5m",
                       scene_signal_id=None):
        return _pending_order()

    async def _update(order, status, **kwargs):
        updates.append((status, kwargs))
        return {**order, "status": status, **kwargs}

    async def _quote(_token, _side, amount_usdt=None):
        if quote_calls is not None:
            quote_calls.append(amount_usdt)
        return ({"averagePrice": 0.55, "amountIn": "5", "amountOut": "9",
                 "quoteId": "Q1"} if quote is None else
                (quote.pop(0) if isinstance(quote, list) else quote))

    async def _place(_q, slippage_bps=1200):
        if place_calls is not None:
            place_calls.append((_q.get("quoteId"), slippage_bps))
        return {"orderId": "ORD-1"} if place is None else (
            place.pop(0) if isinstance(place, list) else place)

    async def _confirm(_oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        return {"orderId": _oid, "status": "FILLED", "price": "0.55"} if confirm is None \
            else (confirm.pop(0) if isinstance(confirm, list) else confirm)

    async def _bal(_amount_usdt):
        return (True, 1000.0, "") if balance is None else balance

    monkeypatch.setattr(trader, "_reserve_order_slot", _reserve)
    monkeypatch.setattr(trader, "_update_signal_order", _update)
    monkeypatch.setattr(trader, "get_quote", _quote)
    monkeypatch.setattr(trader, "place_order", _place)
    monkeypatch.setattr(trader, "confirm_order_status", _confirm)
    monkeypatch.setattr(trader, "_preflight_balance", _bal)


@pytest.mark.asyncio
async def test_signal_trade_expired_terminal_enters_retry_path(monkeypatch) -> None:
    """P0：EXPIRED 纳入可重试终态 → 走 FOK 重试（与 FAILED 同路径）；
    仍 EXPIRED 则 FAILED 并在归因文案里点名真实终态（不再笼统写 FAILED）。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[tuple] = []
    _exec_stubs(monkeypatch, trader, updates, place_calls=place_calls,
                confirm={"orderId": "ORD-1", "status": "EXPIRED"})

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"
    assert len(place_calls) == 3            # 首次 + 2 次重试
    msg = updates[0][1]["error_message"]
    assert "EXPIRED" in msg and "已重试 2 次" in msg


@pytest.mark.asyncio
async def test_signal_trade_cancelled_terminal_enters_retry_path(monkeypatch) -> None:
    """P0：CANCELLED 同样纳入可重试终态（边界：三个非 FILLED 终态行为一致）。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[tuple] = []
    _exec_stubs(monkeypatch, trader, updates, place_calls=place_calls,
                confirm={"orderId": "ORD-1", "status": "CANCELLED"})

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"
    assert len(place_calls) == 3
    assert "CANCELLED" in updates[0][1]["error_message"]


@pytest.mark.asyncio
async def test_signal_trade_deep_poll_recovers_retry_path(monkeypatch) -> None:
    """P0 端到端：常规轮询未命中（币安慢终态）→ 深轮询兜底拿到 FAILED →
    FOK 重试路径恢复执行（旧实现此处直接落 PENDING，13 分钟后才被对账订正）。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[tuple] = []
    rounds = {"i": 0}

    async def _confirm(oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        rounds["i"] += 1
        # 第 1 轮（首次单常规）未命中；第 2 轮（首次单深轮询）拿到 FAILED；
        # 重试单两轮均 FILLED。
        if rounds["i"] == 1:
            return None
        if rounds["i"] == 2:
            assert (attempts, delay) == (CONFIRM_DEEP_ATTEMPTS, CONFIRM_DEEP_DELAY_S)
            return {"orderId": oid, "status": "FAILED"}
        return {"orderId": oid, "status": "FILLED", "price": "0.55"}

    _exec_stubs(monkeypatch, trader, updates, place_calls=place_calls,
                place=[{"orderId": "ORD-SLOW"}, {"orderId": "ORD-RETRY"}])
    monkeypatch.setattr(trader, "confirm_order_status", _confirm)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FILLED"
    assert order["order_id"] == "ORD-RETRY"      # 落库为重试单
    assert len(place_calls) == 2                 # 重试路径确实被执行


@pytest.mark.asyncio
async def test_signal_trade_unconfirmed_never_retries(monkeypatch) -> None:
    """两轮回查都 None → 落 PENDING 且绝不重试：终态未知时重复下单可能双花，
    必须交对账判定（深轮询只降低 None 概率，不改变 None 的保守语义）。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    place_calls: list[tuple] = []
    quote_calls: list[float] = []
    _exec_stubs(monkeypatch, trader, updates, place_calls=place_calls,
                quote_calls=quote_calls, confirm=None)

    async def _confirm_none(_oid, attempts=CONFIRM_ATTEMPTS, delay=CONFIRM_DELAY_S):
        return None

    monkeypatch.setattr(trader, "confirm_order_status", _confirm_none)

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "PENDING"
    assert len(place_calls) == 1 and len(quote_calls) == 1


# ============================================================
# 组 10：P1 资金安全（下单前余额预检 + 低余额告警 + 错误分类）
# 背景：用户在币安 App 手工下单把 CeDeFi 可用余额打穿（账户总资产看着够），
# 币安要到报价阶段才回 -9000，文案泛化难归因（2026-09-04 12:45 生产实证）。
# ============================================================

class _FakeLogger:
    """loguru 替身：只记录 error/warning 文案（低余额告警断言用）。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _fmt(self, msg, args) -> str:
        try:
            return msg.format(*args) if args else str(msg)
        except (IndexError, KeyError):
            return str(msg)

    def error(self, msg, *a) -> None:
        self.errors.append(self._fmt(msg, a))

    def warning(self, msg, *a) -> None:
        self.warnings.append(self._fmt(msg, a))

    def info(self, msg, *a) -> None:
        pass


@pytest.mark.asyncio
async def test_preflight_balance_insufficient_blocks(monkeypatch) -> None:
    """可用余额 < 下单额 → 拒单 + 可归因文案（未提交币安，防 -9000 废单）。"""
    trader = _bare_trader()

    async def _bal(wallet=None):
        return {"usdt_free": "1.2345"}

    monkeypatch.setattr(trader, "fetch_prediction_wallet_balance", _bal)
    ok, free, why = await trader._preflight_balance(2.0)
    assert ok is False and free == 1.2345
    assert "余额不足" in why and "1.2345" in why and "-9000" in why

    # 贴线边界：可用 == 下单额 → 放行（只拦「不够」，不预留缓冲：
    # 硬性预留会在余额刚好够时白停实盘，用户选定的防护档位是预检+告警）
    async def _bal_exact(wallet=None):
        return {"usdt_free": 2.0}

    monkeypatch.setattr(trader, "fetch_prediction_wallet_balance", _bal_exact)
    assert await trader._preflight_balance(2.0) == (True, 2.0, "")


@pytest.mark.asyncio
async def test_preflight_balance_low_alert_still_places(monkeypatch) -> None:
    """余额够本单但低于告警线 → 放行 + ERROR 告警（提前发现资金被抽走）。"""
    trader = _bare_trader()
    log = _FakeLogger()
    monkeypatch.setattr(pt, "logger", log)

    async def _bal(wallet=None):
        return {"usdt_free": LOW_BALANCE_ALERT_USDT - 0.5}

    monkeypatch.setattr(trader, "fetch_prediction_wallet_balance", _bal)
    ok, free, why = await trader._preflight_balance(2.0)
    assert ok is True and why == ""
    assert free == LOW_BALANCE_ALERT_USDT - 0.5
    assert any("低余额告警" in e for e in log.errors), log.errors

    # 反例：余额高于告警线 → 不告警（否则每单刷 ERROR，告警失效）
    log.errors.clear()

    async def _bal_hi(wallet=None):
        return {"usdt_free": LOW_BALANCE_ALERT_USDT + 1}

    monkeypatch.setattr(trader, "fetch_prediction_wallet_balance", _bal_hi)
    assert (await trader._preflight_balance(2.0))[0] is True
    assert log.errors == []


@pytest.mark.asyncio
async def test_preflight_balance_query_failure_conservative(monkeypatch) -> None:
    """查询异常 / 字段缺失 / 非数值 → 保守放行（预检是护栏不是闸门：
    查询通道故障不应停掉实盘，与影子 gate「DB 故障保守全在线」同口径）。"""
    trader = _bare_trader()

    async def _boom(wallet=None):
        raise RuntimeError("balance endpoint down")

    async def _empty(wallet=None):
        return {}

    async def _bad(wallet=None):
        return {"usdt_free": "not-a-number"}

    for stub in (_boom, _empty, _bad):
        monkeypatch.setattr(trader, "fetch_prediction_wallet_balance", stub)
        assert await trader._preflight_balance(2.0) == (True, None, "")


@pytest.mark.asyncio
async def test_signal_trade_balance_precheck_blocks_before_quote(monkeypatch) -> None:
    """预检不过 → 弃单发生在取报价之前（不产生币安废单，占位行落 FAILED 带归因）。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    quote_calls: list[float] = []
    place_calls: list[tuple] = []
    _exec_stubs(
        monkeypatch, trader, updates, quote_calls=quote_calls,
        place_calls=place_calls,
        balance=(False, 0.4242,
                 "预测钱包可用余额不足：0.4242 USDT < 下单额 5.0 USDT"
                 "（未提交币安，防 -9000 废单）"))

    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"
    assert quote_calls == [] and place_calls == []   # 未取报价、未下单
    msg = updates[0][1]["error_message"]
    assert "余额预检弃单" in msg and "0.4242" in msg


@pytest.mark.asyncio
async def test_signal_trade_quote_error_classifies_insufficient_balance(
        monkeypatch) -> None:
    """错误分类：报价阶段币安回 -9000（预检后被并发手工单抽走余额）→
    归因文案直接点名余额不足，不再只有泛化「获取报价失败」。"""
    trader = _make_real_trader(monkeypatch)
    updates: list[tuple] = []
    _exec_stubs(monkeypatch, trader, updates, quote=None)

    async def _quote_9000(_token, _side, amount_usdt=None):
        trader.last_api_error = (
            'HTTP 400: {"code":-9000,"msg":"You don\'t have enough USDT"}')
        return None

    monkeypatch.setattr(trader, "get_quote", _quote_9000)
    order = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order["status"] == "FAILED"
    msg = updates[-1][1]["error_message"]
    assert "余额不足（币安 -9000）" in msg and "获取报价失败" in msg

    # 反例：非余额类错误不加余额前缀（避免误归因把服务端故障当资金问题）
    async def _quote_500(_token, _side, amount_usdt=None):
        trader.last_api_error = "HTTP 500: internal error"
        return None

    monkeypatch.setattr(trader, "get_quote", _quote_500)
    order2 = await trader.execute_signal_trade(
        "DOWN", 5.0, "quote_contrarian_v2", WINDOW_START, max_exec_price=0.78)
    assert order2["status"] == "FAILED"
    msg2 = updates[-1][1]["error_message"]
    assert "余额不足" not in msg2 and "HTTP 500" in msg2
