"""信号次周期路径端点单元测试（2026-08-18）：
GET /api/fake-breakout/signals/{id}/path（main.get_fake_breakout_signal_path）。

不触网络/真实 DB：db 为 AsyncMock 替身，信号行/样本行用 SimpleNamespace。
核心分支：
- 信号不存在 / 无周期锚点 → has_data=False 短路（不查 samples）
- 正常路径：off 换算（ms→s 保留 1 位）、btc/down 缺失行过滤、quote5m_off 区间内换算
- quote5m_ts 越界 → quote5m_off=None；side=low → quote5m_down 取 up 列
- 区间无样本 → has_data=False（points 空但锚点信息完整）
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

START = 1_000_000_000_000
END = START + 900_000


def _signal(**over) -> SimpleNamespace:
    """信号行替身：默认 side=high、+5m 确认点落在区间内第 300 秒。"""
    base = dict(
        market_start_15m=START, market_end_15m=END,
        cycle_open_price_15m=64_000.0, side="high", settle_outcome="DOWN",
        quote5m_ts_15m=START + 300_000, quote5m_down_15m=0.686, quote5m_up_15m=0.30,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _sample(ts: int, btc=64_100.0, down=0.62, up=0.36) -> SimpleNamespace:
    return SimpleNamespace(timestamp=ts, btc_price=btc, down_price=down, up_price=up)


def _make_db(signal, rows) -> AsyncMock:
    db = AsyncMock()
    db.get = AsyncMock(return_value=signal)
    result = MagicMock()
    result.scalars.return_value.all.return_value = rows
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_path_signal_not_found() -> None:
    """信号不存在 → has_data=False，短路不执行 samples 查询。"""
    import binance_predict.main as m

    db = _make_db(None, [])
    out = await m.get_fake_breakout_signal_path(1, db)

    assert out == {"signal_id": 1, "has_data": False, "points": []}
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_path_no_cycle_anchor() -> None:
    """信号存在但无 15m 周期锚点（早期信号未回填）→ has_data=False。"""
    import binance_predict.main as m

    db = _make_db(_signal(market_start_15m=None), [])
    out = await m.get_fake_breakout_signal_path(2, db)

    assert out["has_data"] is False and out["points"] == []
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_path_full_points() -> None:
    """正常路径：off 换算（15s 采样 → 15.0s）、btc/down 缺失行过滤、+5m 确认点换算。"""
    import binance_predict.main as m

    rows = [
        _sample(START + 15_000),
        _sample(START + 30_000, btc=None),        # btc 缺失 → 过滤
        _sample(START + 45_000, down=None),       # down 缺失 → 过滤
        _sample(END - 15_000, btc=63_900.0, down=0.71, up=0.27),
    ]
    db = _make_db(_signal(), rows)
    out = await m.get_fake_breakout_signal_path(3, db)

    assert out["has_data"] is True
    assert out["cycle_start"] == START and out["cycle_end"] == END
    assert out["open"] == 64_000.0 and out["settle"] == "DOWN"
    assert [p["off"] for p in out["points"]] == [15.0, 885.0]  # 仅 2 行存活
    assert out["points"][0] == {"off": 15.0, "btc": 64_100.0, "down": 0.62, "up": 0.36}
    assert out["quote5m_off"] == 300.0
    assert out["quote5m_down"] == 0.686  # side=high → DOWN 报价


@pytest.mark.asyncio
async def test_path_quote5m_out_of_range_side_low() -> None:
    """quote5m_ts 越界 → quote5m_off=None；side=low → quote5m_down 取 up 列。"""
    import binance_predict.main as m

    sig = _signal(side="low", quote5m_ts_15m=END + 60_000, quote5m_up_15m=0.42)
    db = _make_db(sig, [_sample(START + 15_000)])
    out = await m.get_fake_breakout_signal_path(4, db)

    assert out["quote5m_off"] is None
    assert out["quote5m_down"] == 0.42  # 低价突破 → UP 才是顺势方向


@pytest.mark.asyncio
async def test_path_no_samples() -> None:
    """锚点完整但区间无采样（8/13 前信号）→ has_data=False、points 空。"""
    import binance_predict.main as m

    db = _make_db(_signal(), [])
    out = await m.get_fake_breakout_signal_path(5, db)

    assert out["has_data"] is False
    assert out["points"] == [] and out["cycle_start"] == START
