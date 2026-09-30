"""首触 G3/K10 的前3分钟研究版本；影子与实盘共用时间门。"""

G3_EARLY = "firsthit_down_chg_early180_v1"
K10_EARLY = "firsthit_down_k10_early180_v1"
VERSIONS = frozenset({G3_EARLY, K10_EARLY})
DECISION_CUTOFF_MS = 180_000


def before_cutoff(window_start: int, trigger_ts: int) -> bool:
    return int(trigger_ts) < int(window_start) + DECISION_CUTOFF_MS
