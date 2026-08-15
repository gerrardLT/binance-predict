"""场景①②收盘质量判定（classify_close_pattern）与破位 pending 状态机回归测试。

对应 2026-08-15 模式升级：旧 A+B 过滤整体替换为收盘确认场景（180 天
发现集→验证集盲验）：
- 场景① bull_exhaust：破 4h 阻力 + 收阳 + 光头（close_pos ≥ 0.85）→ 次周期 DOWN
- 场景② bear_exhaust：破 4h 支撑 + 收阴 + 放量（vol_ratio ≥ 2.0）→ 次周期 UP

只测纯函数与内存状态机（不触 DB/邮件/klines）：
- classify_close_pattern：两场景通过/不通过/边界值/防御分支
- _record_breakout：每方向每周期只记首次破位
- _on_cycle_boundary：停机跳变（>1 周期）不确认且过期 pending 已清理
"""

from __future__ import annotations

import pytest

from binance_predict.services.fake_breakout_detector import (
    CLOSE_POS_MIN,
    VOL_RATIO_MIN,
    FakeBreakoutDetector,
    classify_close_pattern,
)


# ============================================================
# classify_close_pattern：场景①（high：收阳 + 光头）
# ============================================================

def test_pattern1_bull_exhaust_pass() -> None:
    """光头阳线（close_pos 0.909 ≥ 0.85）：命中，指标回传供审计。"""
    ok, close_pos, vol_ratio = classify_close_pattern(
        "high", o=100.0, h=110.0, l=99.0, c=109.0, volume=50.0, vol_ma=40.0,
    )
    assert ok is True
    assert close_pos == pytest.approx((109.0 - 99.0) / (110.0 - 99.0))
    assert vol_ratio == pytest.approx(50.0 / 40.0)


def test_pattern1_upper_shadow_fail() -> None:
    """长上影（close_pos 0.5 < 0.85）：多头未满仓，不命中。"""
    ok, close_pos, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=99.0, c=104.5, volume=50.0, vol_ma=40.0,
    )
    assert ok is False
    assert close_pos == pytest.approx(0.5)


def test_pattern1_red_close_fail() -> None:
    """收阴（c < o）：即使收盘位置高也不命中（场景①要求收阳）。"""
    ok, close_pos, _ = classify_close_pattern(
        "high", o=109.0, h=110.0, l=99.0, c=108.5, volume=50.0, vol_ma=40.0,
    )
    assert close_pos == pytest.approx(9.5 / 11.0)  # 位置高但方向错


def test_pattern1_close_pos_boundary() -> None:
    """收盘位置恰好 0.85（≥ 含等号）：命中。"""
    ok, close_pos, _ = classify_close_pattern(
        "high", o=100.0, h=110.0, l=100.0, c=100.0 + 10.0 * CLOSE_POS_MIN,
        volume=50.0, vol_ma=40.0,
    )
    assert ok is True
    assert close_pos == pytest.approx(CLOSE_POS_MIN)


# ============================================================
# classify_close_pattern：场景②（low：收阴 + 放量）
# ============================================================

def test_pattern2_bear_exhaust_pass() -> None:
    """收阴 + 放量（量比 2.67 ≥ 2.0）：命中。"""
    ok, close_pos, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=80.0, vol_ma=30.0,
    )
    assert close_pos == pytest.approx(1.0 / 11.0)
    assert vol_ratio == pytest.approx(80.0 / 30.0)


def test_pattern2_low_volume_fail() -> None:
    """收阴但缩量（量比 1.67 < 2.0）：不命中。"""
    ok, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=50.0, vol_ma=30.0,
    )
    assert ok is False
    assert vol_ratio == pytest.approx(50.0 / 30.0)


def test_pattern2_vol_ratio_boundary() -> None:
    """量比恰好 2.0（≥ 含等号）：命中。"""
    ok, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0,
        volume=30.0 * VOL_RATIO_MIN, vol_ma=30.0,
    )
    assert ok is True
    assert vol_ratio == pytest.approx(VOL_RATIO_MIN)


def test_pattern2_no_vol_ma_fail_conservative() -> None:
    """均量数据不足（vol_ma=None）：场景②保守不命中（无审计依据）。"""
    ok, _, vol_ratio = classify_close_pattern(
        "low", o=110.0, h=111.0, l=100.0, c=101.0, volume=500.0, vol_ma=None,
    )
    assert ok is False
    assert vol_ratio is None


def test_pattern2_green_close_fail() -> None:
    """收阳：即使巨量也不命中（场景②要求收阴）。"""
    ok, _, _ = classify_close_pattern(
        "low", o=100.0, h=111.0, l=100.0, c=110.0, volume=80.0, vol_ma=30.0,
    )
    assert ok is False


# ============================================================
# classify_close_pattern：防御分支
# ============================================================

def test_zero_range_defensive() -> None:
    """一字板 H==L（rng=0）：除零防御，不命中且指标为 None。"""
    ok, close_pos, vol_ratio = classify_close_pattern(
        "high", o=100.0, h=100.0, l=100.0, c=100.0, volume=50.0, vol_ma=40.0,
    )
    assert ok is False
    assert close_pos is None
    assert vol_ratio is None


def test_zero_open_defensive() -> None:
    """开盘价异常（o ≤ 0）：防御不命中。"""
    ok, _, _ = classify_close_pattern(
        "low", o=0.0, h=111.0, l=100.0, c=101.0, volume=80.0, vol_ma=30.0,
    )
    assert ok is False


# ============================================================
# 破位 pending 状态机（内存去重 / 边界跳变保护）
# ============================================================

def _make_detector() -> FakeBreakoutDetector:
    """detector 实例：不启动循环，只注入位势测内存状态机。"""
    d = FakeBreakoutDetector(collector=None, pm_15m_latest={})  # type: ignore[arg-type]
    d._levels = {"4h": {"resistance": 100.0, "support": 90.0}}
    return d


def test_record_breakout_first_per_side_per_cycle() -> None:
    """每方向每周期只记首次破位：同周期二次冲高不覆盖首破记录。"""
    d = _make_detector()
    t0 = 900_000 * 100 + 60_000  # 周期 100 内 1min
    d._record_breakout(t0, mid=100.2)
    d._record_breakout(t0 + 60_000, mid=101.5)  # 更高，但同周期同方向
    assert d._pending_breaks["high"]["break_price"] == 100.2
    assert d._pending_breaks["high"]["cycle_id"] == 100
    # 支撑方向未破：无记录
    assert "low" not in d._pending_breaks


def test_record_breakout_after_boundary_cleanup() -> None:
    """周期边界清理后，新周期破位可重新记录（每周期最多一条的另一半）。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    # 周期边界：清理只保留 cycle_id >= 101 的记录
    d._pending_breaks = {
        side: rec for side, rec in d._pending_breaks.items()
        if rec["cycle_id"] >= 101
    }
    assert d._pending_breaks == {}
    # 周期 101 内再次冲高：记新 pending
    d._record_breakout(900_000 * 101 + 30_000, mid=100.4)
    assert d._pending_breaks["high"]["cycle_id"] == 101
    assert d._pending_breaks["high"]["break_price"] == 100.4


@pytest.mark.asyncio
async def test_cycle_boundary_gap_skips_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """停机跳变（跨 >1 周期）：不确认（次周期已走远），过期 pending 已清理。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    fired: list[dict] = []

    async def fake_confirm(*args: object, **kwargs: object) -> None:
        fired.append({"args": args})

    monkeypatch.setattr(d, "_confirm_and_fire", fake_confirm)
    await d._on_cycle_boundary(prev_cycle=100, cur_cycle=103, now_ms=900_000 * 103)
    assert fired == []            # 跳变不确认
    assert d._pending_breaks == {}  # 过期 pending 已清理


@pytest.mark.asyncio
async def test_cycle_boundary_normal_dispatches_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常相邻边界（+1）：上一周期的 pending 派发给收盘确认。"""
    d = _make_detector()
    d._pending_breaks = {
        "high": {"cycle_id": 100, "level": "4h", "broken_level": 100.0,
                 "break_price": 100.2, "break_time": 0},
    }
    fired: list[tuple[object, ...]] = []

    async def fake_confirm(*args: object, **kwargs: object) -> None:
        fired.append(args)

    monkeypatch.setattr(d, "_confirm_and_fire", fake_confirm)
    await d._on_cycle_boundary(prev_cycle=100, cur_cycle=101, now_ms=900_000 * 101)
    assert len(fired) == 1
    due = fired[0][0]  # 第一个位置参数 due: dict[side, rec]
    assert "high" in due
    assert due["high"]["cycle_id"] == 100


@pytest.mark.asyncio
async def test_status_snapshot_exposes_scene_state() -> None:
    """status_snapshot 暴露 pending/重试/周期号，供线上排查（t15 验证用）。"""
    d = _make_detector()
    d._pending_breaks = {
        "low": {"cycle_id": 2071, "level": "4h", "broken_level": 90.0,
                "break_price": 89.9, "break_time": 0},
    }
    d._confirm_retries = [{"due": {}, "prev_cycle": 2070, "cur_cycle": 2071, "retry": 1, "at": 0}]
    d._last_cycle_id = 2071
    snap = d.status_snapshot
    assert snap["pending_breaks"]["low"]["cycle_id"] == 2071
    assert snap["pending_breaks"]["low"]["broken_level"] == 90.0
    assert snap["confirm_retries"] == 1
    assert snap["last_cycle_id"] == 2071
