"""归档污染自愈（archive_contamination_repair）回归测试。

背景：主循环每轮同写 5m + 15m 两条样本（同一时间戳），归档查询曾未过滤
market_period → 5m 情绪窗曲线混入 15m 报价，影子检测器扫出"更便宜一侧"
幻影触发，WR/EV 记账虚高。写入路径已由 commit 5fe6356 修复；
本模块负责历史数据自愈（重建曲线 + 重落信号）。

本测试覆盖纯函数口径（不触 DB）：
- 污染指纹识别（重复时间戳）；
- 受影响信号集合圈定（quote_edge 按本窗，x4 按触发窗或目标窗）；
- x4 重扫集合必含干净前一窗（触发源重建）；
- 重建口径（复刻归档器曲线构建）消解幻影触发。
"""

from __future__ import annotations

from types import SimpleNamespace

from binance_predict.services import archive_contamination_repair as acr
from binance_predict.services import quote_edge_detector as qed

# B 格（quote_contrarian_v1）冻结口径：t∈[45,60)s × q∈[0.15,0.25)
_B_RULE = (45.0, 60.0, 0.15, 0.25)
W = 5 * 60 * 1000


def _win(start: int, curve_down_price: list | None = None,
         curve_up_price: list | None = None,
         curve_up_pct: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        start_time=start, end_time=start + W,
        curve_down_price=curve_down_price, curve_up_price=curve_up_price,
        curve_up_pct=curve_up_pct,
    )


# ------------------------------------------------------------------
# 污染指纹
# ------------------------------------------------------------------

def test_duplicate_ts_is_contamination_fingerprint() -> None:
    contaminated = [{"t": 1000, "v": 0.20}, {"t": 1000, "v": 0.30}]
    clean = [{"t": 1000, "v": 0.30}, {"t": 2000, "v": 0.28}]
    assert acr.has_duplicate_ts(contaminated) is True
    assert acr.has_duplicate_ts(clean) is False
    assert acr.has_duplicate_ts(None) is False
    assert acr.has_duplicate_ts([]) is False


def test_window_is_contaminated_any_curve() -> None:
    dup = [{"t": 1, "v": 0.2}, {"t": 1, "v": 0.3}]
    assert acr.window_is_contaminated(_win(0, curve_down_price=dup)) is True
    assert acr.window_is_contaminated(_win(0, curve_up_price=dup)) is True
    assert acr.window_is_contaminated(_win(0, curve_up_pct=dup)) is True
    assert acr.window_is_contaminated(_win(0, curve_down_price=[{"t": 1, "v": 0.2}])) is False


# ------------------------------------------------------------------
# 受影响信号圈定
# ------------------------------------------------------------------

def test_quote_edge_affected_is_contaminated_windows_itself() -> None:
    # quote_edge 信号本窗即目标窗：window_start ∈ 污染集
    assert acr.quote_edge_affected_starts({100, 200}) == {100, 200}


def test_x4_reprocess_includes_clean_prev_window() -> None:
    # 干净前窗 C 触发、污染目标窗 T 结算的信号也被删除 →
    # 重扫集合必须含 C（重触发）与 T（重结算），升序保证先触发后结算
    t = 1_000_000
    got = acr.x4_reprocess_starts({t})
    assert got == {t - W, t}
    assert sorted(got)[0] == t - W


# ------------------------------------------------------------------
# 重建口径消解幻影（复刻 rebuild 的曲线构建 + 检测器扫描）
# ------------------------------------------------------------------

def _rebuild_curve_down_price(samples: list) -> list:
    """复刻 rebuild_window_from_raw_samples 的 curve_down_price 构建（5m 过滤后）。"""
    return [
        {"t": s.timestamp, "v": s.down_price}
        for s in samples if s.down_price is not None
    ]


def test_rebuild_removes_phantom_trigger() -> None:
    """t=45s 处 5m 报价 0.30（区间外）、15m 报价 0.20（区间内便宜价）。

    - 污染曲线：首个命中 = 15m 的 0.20 → 幻影信号（记账虚高源头）；
    - 重建后（仅 5m 样本）：曲线无重复时间戳、无命中 → 幻影消解。
    """
    ts = 45_000
    samples = [
        SimpleNamespace(timestamp=ts, down_price=0.30, market_period="5m"),
        SimpleNamespace(timestamp=ts, down_price=0.20, market_period="15m"),
    ]
    contaminated = [{"t": s.timestamp, "v": s.down_price} for s in samples]
    assert qed._find_first_hit(contaminated, 0, *_B_RULE) == (0.20, ts)

    kept = [s for s in samples if s.market_period == "5m"]
    rebuilt = _rebuild_curve_down_price(kept)
    assert acr.has_duplicate_ts(rebuilt) is False
    assert qed._find_first_hit(rebuilt, 0, *_B_RULE) is None


def test_rebuild_keeps_real_5m_trigger() -> None:
    """重建不丢真实触发：5m 报价本就在区间内 → 重建后仍命中（同价同刻）。"""
    ts = 50_000
    samples = [
        SimpleNamespace(timestamp=ts, down_price=0.18, market_period="5m"),
        SimpleNamespace(timestamp=ts, down_price=0.40, market_period="15m"),
    ]
    kept = [s for s in samples if s.market_period == "5m"]
    rebuilt = _rebuild_curve_down_price(kept)
    assert qed._find_first_hit(rebuilt, 0, *_B_RULE) == (0.18, ts)


def test_contamination_floor_bounds_scan() -> None:
    # 扫描下界 = 15m 采样开始积累日：之前不存在 15m 样本，不可能污染
    import datetime as _dt
    assert acr.CONTAMINATION_FLOOR_MS == int(
        _dt.datetime(2026, 8, 13, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
