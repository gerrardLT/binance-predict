"""shadow_entry_quote.snapshot_entry_quote 单元测试：K 线族影子入场报价快照守卫。

覆盖：窗口对齐/近开盘/报价合法三守卫的正反例，任一不满足 → (None, None, None)
（保守：无对齐报价 → 该笔 EV 不计，与旧纯 K 线口径兼容）。
"""
from __future__ import annotations

from binance_predict.services.shadow_entry_quote import (
    ENTRY_CLOCK_SKEW_TOLERANCE_MS,
    ENTRY_MAX_OFFSET_MS,
    snapshot_entry_quote,
)

TARGET = 1_700_000_000_000 // 900_000 * 900_000  # 某 15m 窗起点（ms）


def _cache(**over) -> dict:
    base = dict(start_date=TARGET, up_price=0.57, down_price=0.45,
                updated_ts=TARGET + 38_000)  # 开盘后 38s，近开盘守卫内
    base.update(over)
    return base


def test_aligned_fresh_quote_snapshots() -> None:
    """窗口对齐 + 近开盘 + 报价合法 → 返回 (up, down, ts)。"""
    up, down, ts = snapshot_entry_quote(_cache(), TARGET)
    assert (up, down, ts) == (0.57, 0.45, TARGET + 38_000)


def test_offset_boundary_inclusive() -> None:
    """offset 恰为上限 ENTRY_MAX_OFFSET_MS → 仍接受（含边界）。"""
    up, down, ts = snapshot_entry_quote(_cache(updated_ts=TARGET + ENTRY_MAX_OFFSET_MS), TARGET)
    assert up == 0.57 and ts == TARGET + ENTRY_MAX_OFFSET_MS


def test_empty_cache_returns_none() -> None:
    assert snapshot_entry_quote({}, TARGET) == (None, None, None)
    assert snapshot_entry_quote(None, TARGET) == (None, None, None)


def test_window_misalignment_returns_none() -> None:
    """缓存跟踪的窗口 != 目标窗（如冷启动回补/缓存停在下一窗）→ None。"""
    assert snapshot_entry_quote(_cache(start_date=TARGET + 900_000), TARGET) == (None, None, None)
    assert snapshot_entry_quote(_cache(start_date=TARGET - 900_000), TARGET) == (None, None, None)


def test_missing_keys_return_none() -> None:
    for key in ("start_date", "up_price", "down_price", "updated_ts"):
        c = _cache()
        c[key] = None
        assert snapshot_entry_quote(c, TARGET) == (None, None, None), f"{key} 缺失应返回 None"


def test_price_out_of_range_returns_none() -> None:
    """token 价须在开区间 (0,1)：贴 0/1 或越界 → None。"""
    for bad in (0.0, 1.0, -0.1, 1.5):
        assert snapshot_entry_quote(_cache(up_price=bad), TARGET) == (None, None, None)
        assert snapshot_entry_quote(_cache(down_price=bad), TARGET) == (None, None, None)


def test_offset_slightly_before_open_within_skew_tolerance_accepted() -> None:
    """offset 轻微为负（本地钟落后服务器的亚秒~秒级偏差；守卫1 已确保是目标窗）→ 容忍接受。"""
    up, down, ts = snapshot_entry_quote(
        _cache(updated_ts=TARGET - ENTRY_CLOCK_SKEW_TOLERANCE_MS), TARGET
    )
    assert up == 0.57 and down == 0.45 and ts == TARGET - ENTRY_CLOCK_SKEW_TOLERANCE_MS


def test_offset_before_open_beyond_tolerance_returns_none() -> None:
    """offset 负值超出时钟偏差容忍（真正陈旧/开盘前遗留）→ None。"""
    assert snapshot_entry_quote(
        _cache(updated_ts=TARGET - ENTRY_CLOCK_SKEW_TOLERANCE_MS - 1), TARGET
    ) == (None, None, None)


def test_offset_too_late_returns_none() -> None:
    """报价取自目标窗深处（offset>上限，非近开盘入场）→ None。"""
    assert snapshot_entry_quote(
        _cache(updated_ts=TARGET + ENTRY_MAX_OFFSET_MS + 1), TARGET
    ) == (None, None, None)


def test_non_numeric_values_return_none() -> None:
    """脏数据（非数值）→ try/except 兜底 None，不抛异常中断落库。"""
    assert snapshot_entry_quote(_cache(up_price="abc"), TARGET) == (None, None, None)
    assert snapshot_entry_quote(_cache(start_date="abc"), TARGET) == (None, None, None)
