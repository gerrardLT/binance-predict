"""P0-1：EVOLVING 观察态晋升/淘汰决策边界测试。

Evolve 阶段对 EVOLVING 模式的裁决规则（见 sentiment_agent.py Step4）：
- 若 live 样本 sample_count < agent_deep_learn_min_holdout_samples(=50)：
  保持 EVOLVING 继续观察；
- 否则用 live 样本的 Wilson 置信下界裁决：
  wilson_lower_bound(correct, total) > 0.5 → 晋升 ACTIVE（confidence=下界）；
  否则 → RETIRE 淘汰。

本测试复刻该纯决策逻辑并断言其边界，验证代码所依赖的 Wilson 下界数学，
不导入 sentiment_agent（避免可选依赖 instructor）。
"""

from __future__ import annotations

from binance_predict.services.backtest import wilson_lower_bound

MIN_HOLDOUT = 50  # 对齐 settings.agent_deep_learn_min_holdout_samples


def _decide(correct: int, total: int) -> str:
    """复刻 Evolve Step4 对单个 EVOLVING 模式的裁决。"""
    if total < MIN_HOLDOUT:
        return "EVOLVING"
    return "ACTIVE" if wilson_lower_bound(correct, total) > 0.5 else "RETIRE"


def test_below_min_samples_keeps_evolving() -> None:
    # 样本不足阈值：即便命中率 100% 也不晋升，继续观察
    assert _decide(correct=40, total=40) == "EVOLVING"
    assert _decide(correct=49, total=49) == "EVOLVING"


def test_high_winrate_at_threshold_promotes() -> None:
    # 恰好达阈值且高胜率 → Wilson 下界 > 0.5 → 晋升 ACTIVE
    assert wilson_lower_bound(38, 50) > 0.5
    assert _decide(correct=38, total=50) == "ACTIVE"


def test_marginal_winrate_retires() -> None:
    # 达阈值但胜率贴近 0.5 → 下界 <= 0.5 → 淘汰
    assert wilson_lower_bound(30, 50) <= 0.5
    assert _decide(correct=30, total=50) == "RETIRE"


def test_low_winrate_retires() -> None:
    # 达阈值但胜率明显低 → 淘汰
    assert _decide(correct=20, total=50) == "RETIRE"


def test_large_sample_boundary() -> None:
    # 大样本下 0.6 胜率下界稳定 > 0.5 → 晋升；0.5 胜率 → 淘汰
    assert _decide(correct=120, total=200) == "ACTIVE"
    assert _decide(correct=100, total=200) == "RETIRE"
