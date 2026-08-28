"""K 线科学发现流水线（binance_predict.discovery）。

分层架构（层间单向依赖，纯函数无 I/O，I/O 只在 CLI 层）：
  data → features → hypotheses → l1_tester → combo_search → oos_validator → report

方法论对齐既有 720d 产物（output/kline_discovery_*_720d）：
- 切分：0.6/0.2/0.2（发现/验证/冻结 holdout，时序）
- 原子化：连续特征在发现段按 6 分位 [0.1,0.2,0.3,0.7,0.8,0.9] 取阈值
- 漏斗：L1 单因子（BH-FDR q=0.1）→ L2 双因子 → L3 三因子 → 冻结 → holdout 终验（只触碰一次）
- 经济口径：FEE=2% + PREMIUM=0.01，打平胜率 ≈52.04%（复用 backtest.stats）

消费方：scripts/run_kline_discovery.py（CLI 编排）。
"""
from .data import Klines, aggregate_to, data_summary, load_klines_csv
from .features import build_feature_matrix
from .hypotheses import condition_mask, load_rounds, make_atoms, parse_condition
from .l1_tester import bh_fdr, run_l1, seg_stats
from .combo_search import hit_sig, merge_r3, run_combos
from .oos_validator import breakeven_win_rate, run_block_ci, run_oos
from .report import write_outputs
from .targets import build_targets

__all__ = [
    "Klines", "load_klines_csv", "aggregate_to", "data_summary",
    "build_feature_matrix", "load_rounds", "make_atoms", "parse_condition",
    "condition_mask", "bh_fdr", "run_l1", "seg_stats",
    "hit_sig", "run_combos", "merge_r3",
    "run_oos", "run_block_ci", "breakeven_win_rate",
    "write_outputs", "build_targets",
]
