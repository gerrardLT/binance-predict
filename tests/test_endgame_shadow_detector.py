"""endgame 尾盘买热门影子检测器单测（纯函数 + observe_sample 门禁 + 落表）。"""
from __future__ import annotations

import pytest

from binance_predict.services.endgame_shadow_detector import (
    ENDGAME_VERSION,
    FAV_HI,
    FAV_LO,
    TRIGGER_FROM_S,
    pick_favorite,
)

WS = 1_790_000_000_000
BAR = 900_000


def test_pick_favorite_band_edges():
    """0.65 含、0.85 不含（与研究的 [0.65,0.85) 半开区间一致）。"""
    assert pick_favorite(FAV_LO, 1 - FAV_LO) == ("UP", FAV_LO)
    assert pick_favorite(0.84, 0.16) == ("UP", 0.84)
    assert pick_favorite(FAV_HI, 0.15) is None          # 上界贴线不触发
    assert pick_favorite(0.15, 0.84) == ("DOWN", 0.84)  # 热门在 DOWN 侧
    assert pick_favorite(0.45, 0.55) is None            # 两侧都不到 0.65
    assert pick_favorite(None, None) is None


@pytest.mark.asyncio
async def test_observe_sample_gates(monkeypatch):
    """未到 855s / 越窗 / 价位不在带内 → 不落表、不派钩子。"""
    from binance_predict.services import endgame_shadow_detector as mod

    det = mod.EndgameFavShadowDetector(collector=None)
    fired: list[dict] = []
    det._on_live_fire = lambda sig: fired.append(sig)

    calls: list[tuple] = []

    async def _ensure(*a, **kw):
        calls.append((a, kw))
        return None

    monkeypatch.setattr(det, "_ensure_row", _ensure)

    # 早于 855s
    await det.observe_sample(WS, WS + BAR, WS + (TRIGGER_FROM_S - 1) * 1000, 0.70, 0.30)
    # 窗口已结束
    await det.observe_sample(WS, WS + BAR, WS + BAR, 0.70, 0.30)
    # 价位不在带内
    await det.observe_sample(WS, WS + BAR, WS + 870_000, 0.55, 0.45)
    assert calls == [] and fired == []


@pytest.mark.asyncio
async def test_observe_sample_fires_and_dedups(monkeypatch):
    """855s 后热门价在带内 → 落一次表 + 派一次钩子；同窗后续采样被内存去重。"""
    from binance_predict.services import endgame_shadow_detector as mod

    det = mod.EndgameFavShadowDetector(collector=None)
    fired: list[dict] = []
    det._on_live_fire = lambda sig: fired.append(sig)

    created: list[dict] = []

    async def _ensure(ws, we, ts, direction, fav, up, down, t_rel):
        created.append(dict(ws=ws, direction=direction, fav=fav, up=up, down=down, t=t_rel))
        return 1

    monkeypatch.setattr(det, "_ensure_row", _ensure)

    await det.observe_sample(WS, WS + BAR, WS + 860_000, 0.72, 0.28)
    await det.observe_sample(WS, WS + BAR, WS + 870_000, 0.73, 0.27)

    assert len(created) == 1 and created[0]["direction"] == "UP"
    assert created[0]["fav"] == pytest.approx(0.72)
    assert len(fired) == 1
    sig = fired[0]
    assert sig["version"] == ENDGAME_VERSION
    assert sig["direction"] == "UP"
    assert sig["market_start"] == WS and sig["market_end"] == WS + BAR
    assert sig["t_rel"] == pytest.approx(860.0)


@pytest.mark.asyncio
async def test_observe_sample_down_side(monkeypatch):
    """热门在 DOWN 侧时方向为 DOWN，且记录的是 DOWN 价。"""
    from binance_predict.services import endgame_shadow_detector as mod

    det = mod.EndgameFavShadowDetector(collector=None)
    created: list[dict] = []

    async def _ensure(ws, we, ts, direction, fav, up, down, t_rel):
        created.append(dict(direction=direction, fav=fav, up=up, down=down))
        return 1

    monkeypatch.setattr(det, "_ensure_row", _ensure)
    await det.observe_sample(WS, WS + BAR, WS + 880_000, 0.12, 0.84)
    assert created and created[0]["direction"] == "DOWN"

    # UP=0.12 时 DOWN=0.88 超出上界 → 不触发
    created.clear()
    det._fired.clear()
    await det.observe_sample(WS + BAR, WS + 2 * BAR, WS + BAR + 880_000, 0.12, 0.88)
    assert created == []
