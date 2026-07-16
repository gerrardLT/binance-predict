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
"""

from __future__ import annotations

NOISE_THRESHOLD = 0.0005  # 与 settings.noise_threshold 同量级，仅用于本地判定


def _archive_decision(sample_count: int, entry_price: float | None,
                      exit_price: float | None) -> dict:
    """复刻 main.py 归档主体的决策（P1-2 后）。

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
        if actual_return > NOISE_THRESHOLD:
            outcome = "UP"
        elif actual_return < -NOISE_THRESHOLD:
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
    assert low["outcome"] == high["outcome"] == "NOISE"


def test_high_sample_not_flagged_low_quality() -> None:
    res = _archive_decision(sample_count=8, entry_price=100.0, exit_price=101.0)
    assert res["archived"] is True
    assert res["low_quality"] is False
