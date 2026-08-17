"""影子并行单元测试（M4）：参数化 classify + 影子判定分流 + 不发邮件。

不触网络/真实 DB：collector 用最小替身，_fire_confirmed_signal 用捕获替身
验证分流逻辑（ACTIVE 未命中但 SHADOW 命中 → 仅影子 fire）。
"""

from __future__ import annotations

import asyncio

import pytest

from binance_predict.services.fake_breakout_detector import (
    FakeBreakoutDetector,
    classify_close_pattern,
)
from binance_predict.services.scene_params import (
    DEFAULT_SCENE_PARAMS,
    SceneParams,
    is_shadow_supported,
)


# ============================================================
# classify 参数化
# ============================================================

def test_classify_params_override() -> None:
    """同一根 K：默认参数不命中（cp=0.60<0.85），放宽后命中（pos4h 达标 1.0）。"""
    o, h, l, c, v = 100.0, 110.0, 100.0, 106.0, 50.0  # close_pos=0.60
    ok_default, _, cp, _ = classify_close_pattern("high", o, h, l, c, v, 40.0, pos4h=1.0)
    assert ok_default is False and cp is None  # 未命中不回传指标
    ok_loose, _, cp2, _ = classify_close_pattern(
        "high", o, h, l, c, v, 40.0, pos4h=1.0, params=SceneParams(close_pos_min=0.5),
    )
    assert ok_loose is True and cp2 == pytest.approx(0.60)


def test_classify_default_params_equivalence() -> None:
    """params=None 与 DEFAULT_SCENE_PARAMS 等价（向后兼容）。"""
    args = ("high", 100.0, 110.0, 99.0, 109.0, 50.0, 40.0)
    kw = {"pos4h": 1.0}
    assert (classify_close_pattern(*args, **kw)
            == classify_close_pattern(*args, **kw, params=DEFAULT_SCENE_PARAMS))


def test_is_shadow_supported_mapping() -> None:
    base = DEFAULT_SCENE_PARAMS.to_params_json()
    assert is_shadow_supported({**base, "close_pos_min": 0.9}, base) is True
    assert is_shadow_supported({**base, "vol_ratio_min": 2.5}, base) is True
    # 破位层差异 → 影子层不支持（口径一致性保护）
    assert is_shadow_supported({**base, "eps": 0.0008}, base) is False
    # 无差异 → 不支持（影子无意义）
    assert is_shadow_supported(dict(base), base) is False


# ============================================================
# 影子判定分流
# ============================================================

class _FakeStore:
    mid_price = 63000.0


class _FakeCollector:
    store = _FakeStore()

    async def fetch_recent_klines(self, interval: str, limit: int) -> list[dict]:
        # 升序 21 根：前 20 根历史（周期 80~99，收阳、量 40、收盘 100）
        # + 信号 K（周期 100，收阳 close_pos=0.60：默认不命中、放宽后命中）。
        # 历史 20 根满足 vol_ma（=40）；末 15 根收盘全 100 → pos4h=(106-100)/6=1.0 ≥ 0.9
        hist = [{
            "open_time": (80 + i) * 900_000, "open": 99.5, "high": 100.5,
            "low": 99.0, "close": 100.0, "volume": 40.0,
        } for i in range(20)]
        sig = [{
            "open_time": 100 * 900_000, "open": 100.0, "high": 110.0,
            "low": 100.0, "close": 106.0, "volume": 50.0,
        }]
        return hist + sig


@pytest.mark.asyncio
async def test_shadow_fires_when_active_misses(monkeypatch) -> None:
    """ACTIVE（0.85）未命中但 SHADOW（0.5）命中 → 仅影子 fire，不发邮件不占日限。"""
    d = FakeBreakoutDetector(collector=_FakeCollector(), pm_15m_latest={})
    d._shadow_versions = [{"version": "sh1", "params": SceneParams(close_pos_min=0.5)}]

    fired: list[dict] = []

    async def fake_fire(side, rec, sig_k, close_pos, vol_ratio, cur_cycle, now_ms,
                        version="v1", shadow=False, pattern_type=None):
        fired.append({"side": side, "version": version, "shadow": shadow})

    monkeypatch.setattr(d, "_fire_confirmed_signal", fake_fire)
    due = {"high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                    "break_price": 101.0, "break_time": 0}}
    await d._confirm_and_fire(due, prev_cycle=100, cur_cycle=101, now_ms=101 * 900_000, retry=0)

    # ACTIVE 未命中（cp=0.60<0.85）不 fire；SHADOW 命中 fire 一次且带影子标记
    assert len(fired) == 1
    assert fired[0] == {"side": "high", "version": "sh1", "shadow": True}


@pytest.mark.asyncio
async def test_no_shadow_no_fire_when_active_misses() -> None:
    """无 SHADOW 版本时，ACTIVE 未命中 → 零 fire（现状行为不变）。"""
    d = FakeBreakoutDetector(collector=_FakeCollector(), pm_15m_latest={})
    fired = []

    async def fake_fire(*args, **kwargs):
        fired.append(1)

    d._fire_confirmed_signal = fake_fire  # type: ignore[method-assign]
    due = {"high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                    "break_price": 101.0, "break_time": 0}}
    await d._confirm_and_fire(due, prev_cycle=100, cur_cycle=101, now_ms=101 * 900_000, retry=0)
    assert fired == []
