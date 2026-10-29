"""动态仓位：% 基准本金 → 单笔金额（纯函数，无 IO、无状态）。

口径（docs/WEEKLY_REVIEW_PLAN.md §3.3 / D1–D2）：

    单笔 = 基准本金 × pct × 调用点乘数        （乘数 = 高波半仓等，见下）
    向下取整到 0.1 → clamp 到 [0.1, min(cap, 50)]

为什么用「向下取整」而不是 round：宁可少下 0.05U，也不让规则金额越线，
事后核对「实付 ≤ 规则」永远成立，审计无需容忍误差。

为什么保留调用点乘数：`highvol_amount_factor` 等减仓因子在开火点就已经乘进
`amount_usdt`。本模块只替换「基准金额」这一层，不推翻调用点的减仓决定——
传入的 requested/base_amount 之比即乘数。

pct 为 None（通道未启用 % 模式）→ `resolve_amount` 返回 None，调用方保持
固定金额，行为与引入本模块前 100% 一致。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MIN_AMOUNT_USDT = 0.1      # 币安可下单最小粒度（与 main/toggle 校验一致）
MAX_AMOUNT_CAP_USDT = 50.0  # 单笔硬上限（= live_channels.MAX_ORDER_AMOUNT_USDT）
DEFAULT_AMOUNT_CAP_USDT = 20.0
AMOUNT_QUANTUM = 0.1
_EPS = 1e-9                # 浮点地板除修正：0.3 → 不因 0.29999999 取到 0.2


@dataclass(frozen=True)
class SizingDecision:
    """一次仓位裁决的完整审计快照（写进 assessment patch / 日志）。"""

    amount: float
    pct: float | None
    base_equity: float | None
    cap: float
    ratio: float
    raw: float | None
    below_floor: bool
    rule: str

    def as_dict(self) -> dict:
        return asdict(self)


def quantize_amount(value: float) -> float:
    """向下取整到 0.1 并兜底最小金额。"""
    return round(max(MIN_AMOUNT_USDT, math.floor(value * 10 + _EPS) / 10), 4)


def resolve_amount(
    *,
    pct: float | None,
    base_equity: float | None,
    requested_amount: float,
    base_amount: float,
    cap_usdt: float | None = None,
) -> SizingDecision | None:
    """按 % 基准本金算出单笔金额；未启用（pct/基准缺失）返回 None。

    requested_amount/base_amount: 开火点已算好的金额（含高波半仓等乘数）与
    通道基准金额，两者之比即要保留的乘数。
    """
    if pct is None or base_equity is None:
        return None
    try:
        pct_v, equity_v = float(pct), float(base_equity)
        requested_v, base_v = float(requested_amount), float(base_amount)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(pct_v) and math.isfinite(equity_v)) or pct_v <= 0 or equity_v <= 0:
        return None
    cap = DEFAULT_AMOUNT_CAP_USDT if cap_usdt is None else float(cap_usdt)
    if not math.isfinite(cap):
        cap = DEFAULT_AMOUNT_CAP_USDT
    cap_eff = max(MIN_AMOUNT_USDT, min(cap, MAX_AMOUNT_CAP_USDT))

    ratio = 1.0
    if base_v > 0 and math.isfinite(requested_v) and requested_v > 0:
        # 乘数只降不升：上限 1.0（绝不用 floor 把 tiny 乘数抬回去放大金额）；
        # requested<=0/非法时保 1.0（与未启用乘数同义）。
        ratio = min(1.0, requested_v / base_v)

    raw = equity_v * pct_v * ratio
    amount = min(quantize_amount(raw), cap_eff)
    return SizingDecision(
        amount=round(amount, 4),
        pct=round(pct_v, 6),
        base_equity=round(equity_v, 2),
        cap=round(cap_eff, 2),
        ratio=round(ratio, 4),
        raw=round(raw, 4),
        below_floor=raw < MIN_AMOUNT_USDT,
        rule=f"{equity_v:.2f}x{pct_v:.4f}x{ratio:.2f}"
             f"{'' if cap_eff >= cap else f' cap={cap_eff:.2f}'}",
    )


def validate_pct(pct: float, lo: float, hi: float) -> float:
    """校验 pct 合法区间；越界抛 ValueError（配置误写拒绝生效，不静默裁剪）。"""
    value = float(pct)
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"仓位百分比 {pct!r} 超界 [{lo}, {hi}]")
    return value


def validate_base_equity(value: float, lo: float = 10.0, hi: float = 5000.0) -> float:
    """校验基准本金区间（$10~$5000）；越界抛 ValueError。"""
    v = float(value)
    if not (math.isfinite(v) and lo <= v <= hi):
        raise ValueError(f"基准本金 {value!r} 超界 [{lo}, {hi}] USDT")
    return round(v, 2)
