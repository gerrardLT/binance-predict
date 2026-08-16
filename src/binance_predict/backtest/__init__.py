"""科学回测引擎包（binance_predict.backtest）。

四层检验框架：
- L1 随机漫步零假设（stats.variance_ratio）
- L2 市场定价双零假设（surface 曲面隐含概率）
- L3 统计推断 + 功效预检 + 多重检验预算（stats）
- L4 按月 / 波动率三分位稳健性 + 实盘衰减监测（engine 编排层）

消费方：scripts/backtest_engine.py（CLI）与 services/hypothesis_arbiter.py（裁决）。
"""
from .data import aggregate_15m, fetch_klines, load_pm_samples
from .events import build_events
from .stats import (
    Z_EDGES,
    ev,
    exact_binomial_p,
    min_detectable_effect,
    multiple_testing_threshold,
    power_preflight,
    variance_ratio,
    wilson,
    zbin,
)
from .surface import build_surface, e_down_factory

__all__ = [
    "aggregate_15m", "fetch_klines", "load_pm_samples",
    "build_events", "build_surface", "e_down_factory",
    "ev", "wilson", "zbin", "Z_EDGES", "exact_binomial_p",
    "min_detectable_effect", "power_preflight",
    "multiple_testing_threshold", "variance_ratio",
]
