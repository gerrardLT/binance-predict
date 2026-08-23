"""P1-2：低采样情绪窗口兜底归档回归测试。

修复前：_sentiment_window_archiver 对 `len(samples) < 8` 直接跳过归档，导致
低采样窗口永不归档 → 其内 AgentPrediction.is_correct 永久为 None（孤儿预测）。
修复后：采样点数仅作质量标注（warning），只要能取到有效首尾价格即照常归档，
归档决策改由 entry_price 有效性决定，而非采样量。

本测试复刻 main.py `_sentiment_window_archiver` 的归档决策逻辑（不导入 main，
避免可选依赖 instructor），验证：
- 采样量 < 8 但价格有效 → 归档（archived=True，outcome 由涨跌判定）；
- entry_price 无效 → 跳过归档（skipped，对应源码的 continue）；
- outcome 只取决于 entry/exit 价格，与采样量无关。

结算口径对齐后：outcome 按 actual_return 正负号标注（预测市场只按方向赔付），
仅恰好为 0 时标 NOISE。
"""

from __future__ import annotations

from types import SimpleNamespace

from binance_predict.services import quote_edge_detector as qed

# B 格（quote_contrarian_v1）冻结口径：t∈[45,60)s × q∈[0.15,0.25)
_B_RULE = (45.0, 60.0, 0.15, 0.25)


def _archive_samples_filter(samples: list) -> list:
    """复刻 main.py `_sentiment_window_archiver` 归档查询的 market_period 过滤。

    修复前：无 market_period 过滤 → 主循环每轮写入的 5m + 15m 两条样本
    （同一时间戳）全部混入 5m 情绪窗曲线，下游影子检测器扫出 15m 幻影触发，
    与只看 5m 的实盘执行器 / 回测脚本（local_quote_bin_winrate 等均
    market_period=="5m"）口径不一致。修复后：仅归档 5m 样本。
    """
    return [s for s in samples if s.market_period == "5m"]


def _curve_down_price(samples: list) -> list:
    """复刻归档器的曲线构建（main.py：curve_down_price = [{t,v} for s in samples]）。"""
    return [{"t": s.timestamp, "v": s.down_price} for s in samples if s.down_price is not None]


def test_archiver_filters_out_15m_samples() -> None:
    """同一时间戳的 5m + 15m 样本 → 过滤后曲线只留 5m，sample 数减半。

    回归守护：若归档查询丢失 market_period 过滤，sample_count 会是 2 倍
    （生产曾观测到 5m 窗 sample_count=40 = 20 轮询 × 2 市场）。
    """
    ts = 45_000
    mixed = [
        SimpleNamespace(timestamp=ts, down_price=0.30, market_period="5m"),
        SimpleNamespace(timestamp=ts, down_price=0.20, market_period="15m"),
    ]
    kept = _archive_samples_filter(mixed)
    assert len(kept) == 1
    assert kept[0].market_period == "5m"
    assert kept[0].down_price == 0.30


def test_15m_contamination_causes_phantom_trigger() -> None:
    """复刻生产 bug：t=45s 处 5m 报价 0.30（区间外）、15m 报价 0.20（区间内）。

    - 混合曲线（修复前）：影子扫出 15m 的 0.20 → 幻影触发；实盘只看 5m 的
      0.30 未触发 → 影子/实盘不一致（影子多报）。
    - 5m 过滤曲线（修复后）：无命中，与实盘一致。
    """
    ts = 45_000
    s5 = SimpleNamespace(timestamp=ts, down_price=0.30, market_period="5m")
    s15 = SimpleNamespace(timestamp=ts, down_price=0.20, market_period="15m")

    # 修复前：混合曲线 → 首个命中是 15m 的 0.20（幻影）
    contaminated = _curve_down_price([s5, s15])
    assert qed._find_first_hit(contaminated, 0, *_B_RULE) == (0.20, ts)

    # 修复后：5m 过滤曲线 → 无命中（5m 的 0.30 不在 [0.15,0.25)）
    clean = _curve_down_price(_archive_samples_filter([s5, s15]))
    assert qed._find_first_hit(clean, 0, *_B_RULE) is None


def _archive_decision(sample_count: int, entry_price: float | None,
                      exit_price: float | None) -> dict:
    """复刻 main.py 归档主体的决策（P1-2 后 + 结算口径标注）。

    返回 {"archived": bool, "outcome": str|None, "sample_count": int}。
    archived=False 表示源码走 continue 跳过归档。
    """
    # P1-2：采样量不再是闸门，仅告警；对任意采样量继续
    low_quality = sample_count < 8  # 仅质量标注

    # entry_price 无效 → 源码 continue 跳过
    if not entry_price or entry_price <= 0:
        return {"archived": False, "outcome": None,
                "sample_count": sample_count, "low_quality": low_quality}

    outcome = None
    if entry_price and exit_price and entry_price > 0:
        actual_return = exit_price / entry_price - 1
        if actual_return > 0:
            outcome = "UP"
        elif actual_return < 0:
            outcome = "DOWN"
        else:
            outcome = "NOISE"

    return {"archived": True, "outcome": outcome,
            "sample_count": sample_count, "low_quality": low_quality}


def test_low_sample_still_archives() -> None:
    # 采样仅 3 点但价格有效 → 仍归档（修复前会被跳过）
    res = _archive_decision(sample_count=3, entry_price=100.0, exit_price=101.0)
    assert res["archived"] is True
    assert res["low_quality"] is True
    assert res["outcome"] == "UP"


def test_zero_sample_with_valid_price_archives() -> None:
    # 极端：0 采样点但价格有效 → 仍归档（避免孤儿预测）
    res = _archive_decision(sample_count=0, entry_price=100.0, exit_price=99.0)
    assert res["archived"] is True
    assert res["outcome"] == "DOWN"


def test_invalid_entry_price_skips_regardless_of_samples() -> None:
    # entry_price 无效 → 跳过归档，即使采样充足
    res = _archive_decision(sample_count=20, entry_price=0.0, exit_price=100.0)
    assert res["archived"] is False
    assert res["outcome"] is None


def test_outcome_independent_of_sample_count() -> None:
    # 相同价格下，低采样与高采样得到相同 outcome（判定只看价格）
    low = _archive_decision(sample_count=2, entry_price=100.0, exit_price=100.02)
    high = _archive_decision(sample_count=50, entry_price=100.0, exit_price=100.02)
    # 结算口径：只要收涨就是 UP，不再有幅度门槛
    assert low["outcome"] == high["outcome"] == "UP"


def test_flat_price_is_noise() -> None:
    # 恰好分毫不动（无法结算方向）→ NOISE
    res = _archive_decision(sample_count=20, entry_price=100.0, exit_price=100.0)
    assert res["outcome"] == "NOISE"


def test_high_sample_not_flagged_low_quality() -> None:
    res = _archive_decision(sample_count=8, entry_price=100.0, exit_price=101.0)
    assert res["archived"] is True
    assert res["low_quality"] is False
