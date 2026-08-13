"""科学发现系统 —— 谓词 DSL 内核（宪法 Q5）。

LLM 提出的每条假设必须是本模块可执行的谓词 JSON，程序据此做确定性验证。
LLM 不得自我验证；本模块不得替 LLM 定义形态——谓词库是双方唯一的契约语言。

V1 自由度（先紧后松，Q5）：
- L1 原子谓词全开：has_subseq / symbol_at / count_symbol / peak_count / extremum_spacing
- L2 关系谓词两个：lead / sync
- L3 上下文谓词禁用（二期）
- 逻辑组合 AND/OR/NOT，嵌套深度 ≤2 层
- 数值参数枚举化：k∈{1,2,3}、min_matches∈{1,2,3}、count 值域 {1..10}

DSL 节点形态：
    逻辑节点：{"op": "AND"|"OR", "args": [node, ...]}
              {"op": "NOT", "arg": node}
    谓词节点：{"pred": "has_subseq", "channel": "sentiment", "symbols": ["急升", "平"]}
              {"pred": "symbol_at", "channel": "sentiment", "segment": "early", "symbol": "急升"}
              {"pred": "count_symbol", "channel": "volume", "symbol": "急升", "cmp": ">=", "value": 1}
              {"pred": "peak_count", "channel": "price", "cmp": "==", "value": 2}
              {"pred": "extremum_spacing", "channel": "sentiment", "trend": "shrinking"}
              {"pred": "lead", "channel_a": "sentiment", "channel_b": "price", "k": 1, "min_matches": 2}
              {"pred": "sync", "channel_a": "sentiment", "channel_b": "price", "cmp": ">=", "value": 0.7}

执行语义一律确定性：同一 WindowView 输入必得同一 bool 输出。
缺失通道 / 空符号串 → 该谓词返回 False（不产生异常，由校验器在结构上拦截非法引用）。
"""

from __future__ import annotations

from typing import Any

from .symbolizer import DIRECTION_CLASS, SYMBOLS, WindowView

# --- V1 白名单（Q5） ---
L1_PREDICATES = frozenset(
    {"has_subseq", "symbol_at", "count_symbol", "peak_count", "extremum_spacing"}
)
L2_PREDICATES = frozenset({"lead", "sync"})
ALLOWED_PREDICATES = L1_PREDICATES | L2_PREDICATES

ALLOWED_CHANNELS = frozenset({"sentiment", "price", "volume"})
ALLOWED_OPS = frozenset({"AND", "OR", "NOT"})
ALLOWED_SEGMENTS = frozenset({"early", "mid", "late"})
ALLOWED_CMPS = frozenset({">=", "<=", "=="})
ALLOWED_SPACING_TRENDS = frozenset({"shrinking", "expanding", "mixed"})

MAX_LOGIC_DEPTH = 2  # 逻辑节点嵌套上限（Q5）
ALLOWED_K = frozenset({1, 2, 3})  # lead 步长枚举
ALLOWED_MIN_MATCHES = frozenset({1, 2, 3})  # lead 最小对位成功数枚举
COUNT_VALUE_RANGE = range(1, 11)  # count_symbol / peak_count 值域 {1..10}
SYNC_VALUE_RANGE = (0.5, 0.95)  # sync 阈值闭区间


# ============================================================
# DSL 结构校验
# ============================================================


def validate_predicate(node: Any, _depth: int = 0) -> None:
    """校验 DSL 节点结构与 V1 自由度约束。非法时抛 ValueError。

    Args:
        node: DSL 节点（dict）
        _depth: 当前逻辑嵌套深度（内部递归用）

    Raises:
        ValueError: 结构非法 / 谓词不在白名单 / 参数越界 / 深度超限
    """
    if not isinstance(node, dict):
        raise ValueError(f"DSL 节点必须是 dict，得到 {type(node).__name__}")

    # --- 逻辑节点 ---
    if "op" in node:
        op = node["op"]
        if op not in ALLOWED_OPS:
            raise ValueError(f"非法逻辑运算符: {op!r}（允许 AND/OR/NOT）")
        if _depth >= MAX_LOGIC_DEPTH:
            raise ValueError(f"逻辑嵌套深度超限（>{MAX_LOGIC_DEPTH} 层）")
        if op == "NOT":
            if "arg" not in node:
                raise ValueError("NOT 节点缺少 arg")
            validate_predicate(node["arg"], _depth + 1)
        else:
            args = node.get("args")
            if not isinstance(args, list) or not args:
                raise ValueError(f"{op} 节点缺少非空 args 列表")
            for child in args:
                validate_predicate(child, _depth + 1)
        return

    # --- 谓词节点 ---
    pred = node.get("pred")
    if pred is None:
        raise ValueError("节点既无 op 也无 pred，无法识别")
    if pred not in ALLOWED_PREDICATES:
        raise ValueError(f"谓词 {pred!r} 不在 V1 白名单: {sorted(ALLOWED_PREDICATES)}")

    if pred in L2_PREDICATES:
        ch_a = node.get("channel_a")
        ch_b = node.get("channel_b")
        for ch in (ch_a, ch_b):
            if ch not in ALLOWED_CHANNELS:
                raise ValueError(f"非法通道: {ch!r}（允许 {sorted(ALLOWED_CHANNELS)}）")
        if ch_a == ch_b:
            raise ValueError(f"{pred} 的 channel_a 与 channel_b 不能相同")
    else:
        ch = node.get("channel")
        if ch not in ALLOWED_CHANNELS:
            raise ValueError(f"非法通道: {ch!r}（允许 {sorted(ALLOWED_CHANNELS)}）")

    # --- 谓词参数校验 ---
    if pred == "has_subseq":
        symbols = node.get("symbols")
        if (
            not isinstance(symbols, list)
            or not symbols
            or any(s not in SYMBOLS for s in symbols)
        ):
            raise ValueError(f"has_subseq.symbols 必须是非空合法符号列表（{list(SYMBOLS)}）")

    elif pred == "symbol_at":
        if node.get("segment") not in ALLOWED_SEGMENTS:
            raise ValueError(f"symbol_at.segment 非法（允许 {sorted(ALLOWED_SEGMENTS)}）")
        if node.get("symbol") not in SYMBOLS:
            raise ValueError(f"symbol_at.symbol 非法（允许 {list(SYMBOLS)}）")

    elif pred == "count_symbol":
        _validate_cmp(node)
        if node.get("symbol") not in SYMBOLS:
            raise ValueError(f"count_symbol.symbol 非法（允许 {list(SYMBOLS)}）")
        _validate_count_value(node)

    elif pred == "peak_count":
        _validate_cmp(node)
        _validate_count_value(node)

    elif pred == "extremum_spacing":
        if node.get("trend") not in ALLOWED_SPACING_TRENDS:
            raise ValueError(
                f"extremum_spacing.trend 非法（允许 {sorted(ALLOWED_SPACING_TRENDS)}）"
            )

    elif pred == "lead":
        if node.get("k") not in ALLOWED_K:
            raise ValueError(f"lead.k 必须枚举于 {sorted(ALLOWED_K)}")
        if node.get("min_matches") not in ALLOWED_MIN_MATCHES:
            raise ValueError(
                f"lead.min_matches 必须枚举于 {sorted(ALLOWED_MIN_MATCHES)}"
            )

    elif pred == "sync":
        _validate_cmp(node)
        value = node.get("value")
        lo, hi = SYNC_VALUE_RANGE
        if not isinstance(value, (int, float)) or not (lo <= float(value) <= hi):
            raise ValueError(f"sync.value 必须在 [{lo}, {hi}] 闭区间")


def _validate_cmp(node: dict) -> None:
    if node.get("cmp") not in ALLOWED_CMPS:
        raise ValueError(f"cmp 非法（允许 {sorted(ALLOWED_CMPS)}）")


def _validate_count_value(node: dict) -> None:
    value = node.get("value")
    if not isinstance(value, int) or value not in COUNT_VALUE_RANGE:
        raise ValueError(
            f"value 必须是整数且属于 {{{COUNT_VALUE_RANGE.start}..{COUNT_VALUE_RANGE.stop - 1}}}"
        )


# ============================================================
# 谓词执行（纯函数，确定性）
# ============================================================


def _cmp(a: float, op: str, b: float) -> bool:
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    return a == b


def _segment_range(length: int, segment: str) -> tuple[int, int]:
    """符号串三等分的 [start, end) 区间。"""
    third = length / 3.0
    if segment == "early":
        return 0, max(1, round(third))
    if segment == "mid":
        return round(third), max(round(third) + 1, round(2 * third))
    return min(length - 1, round(2 * third)), length


def _transition_points(symbols: list[str]) -> list[int]:
    """符号转移点：symbol[i] != symbol[i-1] 的位置 i 列表。"""
    return [i for i in range(1, len(symbols)) if symbols[i] != symbols[i - 1]]


def evaluate_predicate(node: dict, view: WindowView) -> bool:
    """对 WindowView 执行谓词 JSON，返回确定性 bool。

    执行前先做完整结构校验（validate_predicate），非法节点抛 ValueError。
    引用了视图缺失通道的谓词返回 False（防御：volume 通道可能缺数据）。
    """
    validate_predicate(node)
    return _eval(node, view)


def _eval(node: dict, view: WindowView) -> bool:
    # --- 逻辑节点 ---
    if "op" in node:
        op = node["op"]
        if op == "AND":
            return all(_eval(c, view) for c in node["args"])
        if op == "OR":
            return any(_eval(c, view) for c in node["args"])
        return not _eval(node["arg"], view)  # NOT

    pred = node["pred"]

    # --- L2 关系谓词 ---
    if pred in L2_PREDICATES:
        ch_a, ch_b = node["channel_a"], node["channel_b"]
        if not view.has_channel(ch_a) or not view.has_channel(ch_b):
            return False
        seq_a = view.channels[ch_a].symbols
        seq_b = view.channels[ch_b].symbols
        if pred == "lead":
            return _lead(seq_a, seq_b, int(node["k"]), int(node["min_matches"]))
        return _sync(seq_a, seq_b, node["cmp"], float(node["value"]))

    # --- L1 原子谓词 ---
    ch = node["channel"]
    if not view.has_channel(ch):
        return False
    cv = view.channels[ch]

    if pred == "has_subseq":
        return _has_subseq(cv.symbols, node["symbols"])
    if pred == "symbol_at":
        return _symbol_at(cv.symbols, node["segment"], node["symbol"])
    if pred == "count_symbol":
        count = sum(1 for s in cv.symbols if s == node["symbol"])
        return _cmp(count, node["cmp"], int(node["value"]))
    if pred == "peak_count":
        return _cmp(
            float(cv.geometry.get("peak_count", 0)), node["cmp"], int(node["value"])
        )
    # extremum_spacing
    return cv.geometry.get("extremum_spacing") == node["trend"]


def _has_subseq(seq: list[str], subseq: list[str]) -> bool:
    """seq 是否包含连续子序列 subseq。"""
    n, m = len(seq), len(subseq)
    if m == 0 or m > n:
        return False
    return any(seq[i : i + m] == list(subseq) for i in range(n - m + 1))


def _symbol_at(seq: list[str], segment: str, symbol: str) -> bool:
    """目标符号在指定段内出现且占比 >= 50%。

    段定义：符号串三等分（early/mid/late）。5min 窗口约 4~8 个符号，
    分段后每段 1~3 个符号；占比 ≥50% 且至少出现 1 次即匹配。
    """
    if not seq:
        return False
    start, end = _segment_range(len(seq), segment)
    seg = seq[start:end]
    if not seg:
        return False
    count = sum(1 for s in seg if s == symbol)
    return count >= 1 and count / len(seg) >= 0.5


def _lead(seq_a: list[str], seq_b: list[str], k: int, min_matches: int) -> bool:
    """A 的符号转移领先 B 的符号转移 k 位（容差 ±1），成功对位数 >= min_matches。

    对 A 的每个转移点 t_a，若 B 存在转移点 t_b 满足 (t_b - t_a) ∈ {k-1, k, k+1}，
    记一次成功对位。序列过短（无转移点）直接 False。
    """
    transitions_a = _transition_points(seq_a)
    transitions_b = set(_transition_points(seq_b))
    if not transitions_a or not transitions_b:
        return False
    matches = sum(
        1
        for t_a in transitions_a
        if any((t_a + k + delta) in transitions_b for delta in (-1, 0, 1))
    )
    return matches >= min_matches


def _sync(seq_a: list[str], seq_b: list[str], cmp_op: str, value: float) -> bool:
    """两符号串的方向类同步率（同位置方向类相同比例）与阈值比较。

    方向类归并：急升/缓升→U，平→F，缓降/急降→D（见 symbolizer.DIRECTION_CLASS）。
    对齐长度取两串较短者；任一串为空 → False。
    """
    if not seq_a or not seq_b:
        return False
    m = min(len(seq_a), len(seq_b))
    same = sum(
        1
        for i in range(m)
        if DIRECTION_CLASS.get(seq_a[i]) == DIRECTION_CLASS.get(seq_b[i])
    )
    return _cmp(same / m, cmp_op, value)
