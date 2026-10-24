"""wbid 族（窗口内深价接盘）注册与判定逻辑测试。

覆盖：通道注册 / 默认关闭 / 限价与挂单价 / 同窗互斥 / 位面触及判定（k、speed、
急跌过滤）/ ATR5 shift(1) / 实盘钩子在未启用时不派单。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from binance_predict.services import live_channels as lc
from binance_predict.services.multi_live_trader import MultiLiveTrader
from binance_predict.services.wbid_shadow_detector import (
    WBID_SPECS,
    WARMUP_1M,
    _atr5_at,
    evaluate_window,
)

WBID_IDS = [s["version"] for s in WBID_SPECS]


def _bars(lows, closes=None, opens=None, highs=None):
    closes = closes or [100.0] * len(lows)
    opens = opens or [100.0] * len(lows)
    highs = highs or [max(o, c, l) + 0.1 for o, c, l in zip(opens, closes, lows)]
    return [{"open": o, "high": h, "low": l, "close": c}
            for o, h, l, c in zip(opens, highs, lows, closes)]


# ---------------------------------------------------------------- 注册与开关
def test_wbid_channels_registered_with_limit_price():
    """LIMIT 单的挂单价由 auto_max_exec 承载（execute_signal_trade 内部
    limit_price = max_exec_price），须等于研究冻结的建议价。"""
    for spec_id, limit in [("wbid_d03_k2_5m_v1", 0.21),
                           ("wbid_d03_k3_5m_v1", 0.16),
                           ("wbid_d05_k2_5m_v1", 0.14),
                           ("wbid_d03_k2_fast_5m_v1", 0.21)]:
        spec = lc.LIVE_CHANNELS[spec_id]
        assert spec.family == "wbid"
        assert spec.market_period == "5m"
        assert spec.direction == "UP"
        assert spec.order_type == "LIMIT"
        assert spec.auto_max_exec == limit


def test_wbid_default_disabled():
    """下单开关默认关闭：注册即可见、但 enabled 初始为 False。"""
    for spec_id in WBID_IDS:
        assert lc.LIVE_CHANNELS[spec_id].default_enabled is False


def test_wbid_fast_subset_is_mutually_exclusive():
    grp = lc.exclusive_group("wbid_d03_k2_fast_5m_v1")
    assert grp is not None
    assert "wbid_d03_k2_5m_v1" in grp
    # 其余三个 k/d 互不重叠，不在互斥组内
    assert lc.exclusive_group("wbid_d03_k3_5m_v1") is None
    assert lc.exclusive_group("wbid_d05_k2_5m_v1") is None


# ---------------------------------------------------------------- 判定逻辑
def test_evaluate_window_detects_k2_touch():
    # 窗口 open=100，ATR5=10 → level(d=0.3)=97；第 2 根 low=96.5 触及
    bars = _bars([99.0, 96.5, 98.0], closes=[99.5, 96.8, 98.2],
                 opens=[100.0, 99.5, 96.8])
    hits = evaluate_window(100.0, 10.0, bars, WBID_SPECS)
    vers = {h["spec"]["version"] for h in hits}
    assert "wbid_d03_k2_5m_v1" in vers          # k=2 命中
    assert "wbid_d03_k3_5m_v1" not in vers      # k 不等于 3
    assert "wbid_d05_k2_5m_v1" not in vers      # d=0.5 level=95 未触及


def test_evaluate_window_no_touch_when_shallow():
    bars = _bars([99.0, 98.0, 98.5])
    assert evaluate_window(100.0, 10.0, bars, WBID_SPECS) == []


def test_evaluate_window_fast_filter():
    # 急跌：close_{k-1} 远低于 open → speed 大；slow 版不命中 fast 版命中
    bars = _bars([99.0, 96.5], closes=[99.0, 96.8], opens=[100.0, 99.0])
    hits = evaluate_window(100.0, 10.0, bars, WBID_SPECS)
    vers = {h["spec"]["version"] for h in hits}
    # speed = (100 - 99.0)/10/2 = 0.05 > 0.032876 → fast 命中
    assert "wbid_d03_k2_fast_5m_v1" in vers
    assert "wbid_d03_k2_5m_v1" in vers


def test_evaluate_window_slow_excludes_fast_version():
    # 缓跌：close_{k-1} 接近 open → speed 小 → fast 版不命中
    bars = _bars([99.0, 96.5], closes=[99.9, 96.8], opens=[100.0, 99.9])
    hits = evaluate_window(100.0, 10.0, bars, WBID_SPECS)
    vers = {h["spec"]["version"] for h in hits}
    assert "wbid_d03_k2_5m_v1" in vers
    assert "wbid_d03_k2_fast_5m_v1" not in vers


def test_atr5_uses_prior_bar_only():
    """ATR5 必须是 t0 之前一根的值（shift(1)），不含当前窗口。"""
    closed = []
    base = 0
    # 16 根完整 5m（t0 之前）+ 当前窗口 2 根 1m
    for i in range(16):
        w = base + i * 300_000
        for j in range(5):
            closed.append({"open_time": w + j * 60_000, "open": 100.0,
                           "high": 101.0, "low": 99.0, "close": 100.0})
    t0 = base + 16 * 300_000
    closed.append({"open_time": t0, "open": 100.0, "high": 100.5,
                   "low": 99.5, "close": 100.0})
    closed.append({"open_time": t0 + 60_000, "open": 100.0, "high": 100.5,
                   "low": 90.0, "close": 95.0})   # 当前窗暴跌，不应进 ATR
    a = _atr5_at(closed, t0)
    assert a is not None
    # 每根 TR=2 的平稳序列 → ATR≈2，与当前窗的暴跌无关
    assert a == pytest.approx(2.0, abs=0.05)


def test_warmup_covers_atr14_lookback():
    """WARMUP_1M 必须够 ATR14 的前置 15 个 5m 窗口（75 根 1m）+ 当前窗余量。

    回归防护：曾用 40 根（仅 8 个窗口）导致 _atr5_at 恒 None、检测器静默不触发。
    """
    assert WARMUP_1M >= 80


def test_atr5_returns_value_with_warmup_sized_input():
    closed = []
    for i in range(18):
        w = i * 300_000
        for j in range(5):
            closed.append({"open_time": w + j * 60_000, "open": 100.0,
                           "high": 101.0, "low": 99.0, "close": 100.0})
    t0 = 18 * 300_000
    closed.append({"open_time": t0, "open": 100.0, "high": 100.5,
                   "low": 99.5, "close": 100.0})
    closed.append({"open_time": t0 + 60_000, "open": 100.0, "high": 100.5,
                   "low": 97.0, "close": 98.0})
    assert _atr5_at(closed[-WARMUP_1M:], t0) is not None


# ---------------------------------------------------------------- 实盘门禁
def _bare_trader(enabled: bool):
    t = object.__new__(MultiLiveTrader)
    t._stopped = False
    ch = "wbid_d03_k2_5m_v1"
    t._specs = {ch: lc.LIVE_CHANNELS[ch]}
    t._configs = {ch: SimpleNamespace(enabled=enabled, fired=set())}
    return t, ch


def test_hook_gate_blocks_when_disabled():
    t, ch = _bare_trader(enabled=False)
    assert t._hook_gate(ch, "wbid", 1_000) is False


def test_hook_gate_passes_when_enabled_and_dedups():
    t, ch = _bare_trader(enabled=True)
    assert t._hook_gate(ch, "wbid", 1_000) is True
    assert t._hook_gate(ch, "wbid", 1_000) is False   # 同窗去重
    assert t._hook_gate(ch, "wbid", 2_000) is True


def test_hook_gate_rejects_family_mismatch():
    t, ch = _bare_trader(enabled=True)
    assert t._hook_gate(ch, "nextbar", 1_000) is False


def test_on_wbid_signal_no_order_when_disabled(monkeypatch):
    """未启用时 on_wbid_signal 不得派生下单任务。"""
    t, ch = _bare_trader(enabled=False)
    created = []
    monkeypatch.setattr(
        "binance_predict.services.multi_live_trader.asyncio.create_task",
        lambda *a, **k: created.append(a) or None)
    t.on_wbid_signal({"version": ch, "market_start": 1_000,
                      "market_end": 1_000 + 300_000, "direction": "UP",
                      "signal_bar_start": 1_000, "price_limit": 0.21,
                      "k": 2, "speed": 0.01})
    assert created == []
