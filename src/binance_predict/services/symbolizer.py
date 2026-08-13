"""科学发现系统 —— 符号化内核（宪法 Q1/Q2/Q4）。

将任意时序曲线转换为 LLM 可消费的符号串与几何摘要。全程无 LLM、无 DB、
纯函数：同一输入必得同一输出（分箱边界由 BinningSnapshot 显式传入）。

设计要点（对应 .kiro/specs/scientific-discovery/design.md）：
- 分箱边界来自数据自身：每通道独立的相邻点差值 20/40/60/80 分位切 5 档（Q4）
- 三通道量纲不同（up_pct 0~100 / btc_price 绝对价位 / volume 量级不定），
  共用边界会让小量纲通道全部落入"平"档——Q4 修订为每通道独立分箱
- 分箱快照定期冻结（默认 30 天），模式与其"出生"分箱版本绑定（Q4）
- 三通道同等符号化：sentiment / price / volume，不预设规律载体（Q2）
- 几何摘要供符号层面无解时下钻：极值序列 / 面积比 / 卷曲度 / 极值间距趋势（Q1）

曲线输入格式对齐 sentiment_windows 表：list[{"t": int, "v": float}]，v 允许 None（跳过）。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

# --- 符号集合（Q4：5 档分箱）---
# 对相邻点差值 delta 按分位边界映射：delta < q20 → 急降；[q20,q40) → 缓降；
# [q40,q60) → 平；[q60,q80) → 缓升；>= q80 → 急升。
SYMBOL_SURGE = "急升"
SYMBOL_RISE = "缓升"
SYMBOL_FLAT = "平"
SYMBOL_DIP = "缓降"
SYMBOL_DROP = "急降"
SYMBOLS: tuple[str, ...] = (
    SYMBOL_SURGE,
    SYMBOL_RISE,
    SYMBOL_FLAT,
    SYMBOL_DIP,
    SYMBOL_DROP,
)

# 方向归并（sync 谓词使用）：急升/缓升→U，平→F，缓降/急降→D
DIRECTION_CLASS: dict[str, str] = {
    SYMBOL_SURGE: "U",
    SYMBOL_RISE: "U",
    SYMBOL_FLAT: "F",
    SYMBOL_DIP: "D",
    SYMBOL_DROP: "D",
}

# 分位点（Q4）：20/40/60/80 分位
QUANTILES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)

# 冻结周期（Q4：30 天）
FREEZE_INTERVAL_SECONDS: float = 30 * 86_400.0

# V1 三通道（Q2）：窗口 dict 字段 → 通道名
CHANNEL_FIELDS: dict[str, str] = {
    "sentiment": "curve_up_pct",
    "price": "curve_btc_price",
    "volume": "curve_trade_volume",
}


# ============================================================
# 分箱快照（Q4）
# ============================================================


@dataclass(frozen=True)
class BinningSnapshot:
    """分箱参数冻结快照。

    模式必须记录其"出生"时的快照 version，如同物理测量注明仪器精度。
    edges 为 4 个分位边界值（升序）：(q20, q40, q60, q80)。
    """

    version: str  # 如 "2026-08"
    edges: tuple[float, float, float, float]
    created_at_epoch: float
    sample_count: int  # 参与计算分位的差值样本数

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edges"] = list(self.edges)
        return d

    @staticmethod
    def from_dict(d: dict) -> BinningSnapshot:
        return BinningSnapshot(
            version=str(d["version"]),
            edges=tuple(float(x) for x in d["edges"]),  # type: ignore[arg-type]
            created_at_epoch=float(d["created_at_epoch"]),
            sample_count=int(d["sample_count"]),
        )


def _series_values(curve: list | None) -> list[float]:
    """从 [{t, v}, ...] 曲线抽取 v 序列；空/非法点跳过。"""
    if not curve:
        return []
    vals: list[float] = []
    for p in curve:
        v = p.get("v") if isinstance(p, dict) else p
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return vals


def series_deltas(values: list[float]) -> list[float]:
    """相邻点差值序列。n 个点 → n-1 个差值。"""
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def compute_bin_edges(
    deltas: list[float],
    version: str,
    created_at_epoch: float | None = None,
) -> BinningSnapshot:
    """从全部历史相邻点差值计算分位数边界，生成分箱快照。

    边界来自数据自身分布（20/40/60/80 分位），不引入人为阈值（Q4）。
    样本不足（<10 个差值）时抛出 ValueError——分位数在小样本下无意义。

    Args:
        deltas: 全部历史窗口的相邻点差值汇总
        version: 快照版本标识（如 "2026-08"）
        created_at_epoch: 创建时间戳（默认当前时间）

    Returns:
        BinningSnapshot（edges 升序）
    """
    if len(deltas) < 10:
        raise ValueError(f"分箱样本不足: {len(deltas)} < 10")
    ordered = sorted(float(d) for d in deltas)
    n = len(ordered)

    def _quantile(q: float) -> float:
        # 线性插值分位数（与 numpy 默认 'linear' 一致）
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    edges = tuple(_quantile(q) for q in QUANTILES)
    return BinningSnapshot(
        version=version,
        edges=edges,  # type: ignore[arg-type]
        created_at_epoch=(
            created_at_epoch if created_at_epoch is not None else time.time()
        ),
        sample_count=n,
    )


def should_freeze(
    latest_snapshot: BinningSnapshot | None,
    now_epoch: float | None = None,
) -> bool:
    """判断是否应冻结新一版分箱快照（距上次冻结 >= 30 天，或从未冻结）。"""
    if latest_snapshot is None:
        return True
    now = now_epoch if now_epoch is not None else time.time()
    return (now - latest_snapshot.created_at_epoch) >= FREEZE_INTERVAL_SECONDS


def compute_channel_snapshots(
    windows: list[dict],
    version: str,
    channel_fields: dict[str, str] | None = None,
    created_at_epoch: float | None = None,
) -> dict[str, BinningSnapshot]:
    """从全部历史窗口为每个通道独立计算分箱快照（Q4 修订：每通道独立边界）。

    某通道差值样本 <10 时该通道缺席（分位数在小样本下无意义）；调用方应显式
    检查返回 dict 的通道覆盖，缺席通道的谓词一律求值 False，不静默降级。
    """
    mapping = channel_fields or CHANNEL_FIELDS
    result: dict[str, BinningSnapshot] = {}
    for channel, field_name in mapping.items():
        deltas: list[float] = []
        for w in windows:
            deltas.extend(series_deltas(_series_values(w.get(field_name))))
        if len(deltas) < 10:
            continue
        result[channel] = compute_bin_edges(deltas, version, created_at_epoch)
    return result


# ============================================================
# 符号化（Q1 主通道）
# ============================================================


def symbolize_delta(delta: float, snapshot: BinningSnapshot) -> str:
    """单个相邻点差值 → 符号。"""
    q20, q40, q60, q80 = snapshot.edges
    if delta < q20:
        return SYMBOL_DROP
    if delta < q40:
        return SYMBOL_DIP
    if delta < q60:
        return SYMBOL_FLAT
    if delta < q80:
        return SYMBOL_RISE
    return SYMBOL_SURGE


def symbolize_series(curve: list | None, snapshot: BinningSnapshot) -> list[str]:
    """曲线 [{t, v}, ...] → 符号串。n 个有效点 → n-1 个符号。"""
    values = _series_values(curve)
    return [symbolize_delta(d, snapshot) for d in series_deltas(values)]


# ============================================================
# 几何摘要（Q1 辅助通道）
# ============================================================


def _find_extrema(values: list[float]) -> list[dict]:
    """局部极值点序列（位置归一化到 0~1）。

    返回 [{"pos": float, "kind": "peak"|"trough"}, ...]，按时间升序。
    不做滤波去抖——保持领域无关的最小定义；噪声抑制由符号化层完成。
    """
    n = len(values)
    if n < 3:
        return []
    extrema: list[dict] = []
    denom = n - 1
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            extrema.append({"pos": i / denom, "kind": "peak"})
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            extrema.append({"pos": i / denom, "kind": "trough"})
    return extrema


def _area_ratio(values: list[float]) -> float:
    """曲线相对首末连线的上方面积占比（0~1）。

    上方面积 / (上方+下方面积)。>0.5 表示曲线主体在首末连线之上（凸起），
    <0.5 表示凹陷。总面积为 0（直线）时返回 0.5。
    """
    n = len(values)
    if n < 2:
        return 0.5
    first, last = values[0], values[-1]
    above = 0.0
    below = 0.0
    for i, v in enumerate(values):
        baseline = first + (last - first) * i / (n - 1)
        diff = v - baseline
        if diff > 0:
            above += diff
        else:
            below -= diff
    total = above + below
    if total <= 0:
        return 0.5
    return above / total


def _curliness(values: list[float]) -> float:
    """卷曲度 = 总变差 / |净位移|。直线 ≈1；往返越剧烈越大。

    净位移为 0（回到原点）时返回 min(总变差, 99.0) 的封顶值，避免除零。
    """
    if len(values) < 2:
        return 1.0
    total_variation = sum(abs(d) for d in series_deltas(values))
    net = abs(values[-1] - values[0])
    if net <= 1e-12:
        return min(total_variation, 99.0) if total_variation > 0 else 1.0
    return total_variation / net


def _extremum_spacing(extrema: list[dict]) -> str:
    """相邻极值间距趋势："shrinking" | "expanding" | "mixed" | "insufficient"。

    间距序列中递减占比 >60% 判 shrinking，递增占比 >60% 判 expanding，
    其余 mixed；极值数 <3（间距数 <2）判 insufficient。
    """
    if len(extrema) < 3:
        return "insufficient"
    positions = [e["pos"] for e in extrema]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    dec = sum(1 for i in range(len(gaps) - 1) if gaps[i + 1] < gaps[i])
    inc = sum(1 for i in range(len(gaps) - 1) if gaps[i + 1] > gaps[i])
    total = len(gaps) - 1
    if dec / total > 0.6:
        return "shrinking"
    if inc / total > 0.6:
        return "expanding"
    return "mixed"


def geometric_summary(curve: list | None) -> dict:
    """曲线的领域无关几何摘要（Q1 辅助下钻通道）。

    Returns:
        {
            "extrema": [{"pos", "kind"}, ...],   # 极值位置序列
            "peak_count": int,                   # 峰数（peak_count 谓词的直接输入）
            "area_ratio": float,                 # 上方面积占比（凸/凹）
            "curliness": float,                  # 卷曲度（总变差/净位移）
            "extremum_spacing": str,             # 极值间距趋势
        }
    """
    values = _series_values(curve)
    extrema = _find_extrema(values)
    return {
        "extrema": extrema,
        "peak_count": sum(1 for e in extrema if e["kind"] == "peak"),
        "area_ratio": _area_ratio(values),
        "curliness": _curliness(values),
        "extremum_spacing": _extremum_spacing(extrema),
    }


# ============================================================
# 窗口视图（Q2：三通道一键符号化）
# ============================================================


@dataclass
class ChannelView:
    """单通道的符号化视图：谓词执行的最小输入单元。"""

    symbols: list[str]
    geometry: dict
    point_count: int  # 原始有效采样点数（symbols 长度 + 1）


@dataclass
class WindowView:
    """一个 5min 窗口的完整符号化视图（三通道）。"""

    start_time: int
    window_id: int | None = None
    outcome: str | None = None  # UP | DOWN | NOISE（验证时使用，发现时对 LLM 可见）
    channels: dict[str, ChannelView] = field(default_factory=dict)

    def has_channel(self, channel: str) -> bool:
        return channel in self.channels and len(self.channels[channel].symbols) > 0


def build_window_view(
    window: dict,
    snapshots: BinningSnapshot | dict[str, BinningSnapshot],
    channel_fields: dict[str, str] | None = None,
) -> WindowView:
    """从窗口 dict（sentiment_windows 序列化形态）构建符号化视图。

    通道字段映射默认 CHANNEL_FIELDS（Q2）：sentiment←curve_up_pct、
    price←curve_btc_price、volume←curve_trade_volume。
    有效点 <2 的通道被跳过（无法产生任何符号），谓词对缺失通道一律返回 False。

    Args:
        window: 含 curve_* / start_time / id / outcome 的窗口 dict
        snapshots: 分箱快照。Q4 修订后应传 dict[str, BinningSnapshot] 按通道取
            独立边界（缺快照的通道跳过）；兼容传单个 BinningSnapshot 表示
            三通道共用同一边界（单通道场景 / Phase 1 既有测试）。
        channel_fields: 可覆盖的通道映射

    Returns:
        WindowView
    """
    mapping = channel_fields or CHANNEL_FIELDS
    channels: dict[str, ChannelView] = {}
    for channel, field_name in mapping.items():
        snap = snapshots.get(channel) if isinstance(snapshots, dict) else snapshots
        if snap is None:
            continue
        curve = window.get(field_name)
        values = _series_values(curve)
        if len(values) < 2:
            continue
        channels[channel] = ChannelView(
            symbols=[symbolize_delta(d, snap) for d in series_deltas(values)],
            geometry=geometric_summary(curve),
            point_count=len(values),
        )
    return WindowView(
        start_time=int(window.get("start_time", 0)),
        window_id=window.get("id"),
        outcome=window.get("outcome"),
        channels=channels,
    )
