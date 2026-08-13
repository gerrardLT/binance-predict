"""科学发现系统 —— 经济闸内核（宪法 Q6-a 补充，V1.1）。

lift 检验回答「是否偏离局部基准」，本模块回答「是否跑赢入场价」——
二元预测市场里入场价 ≈ 群众隐含概率，是盈利的唯一诚实基准。

实证依据（2026-08-11 台架 output/predicate_ev_bench.json）：全窗 lift 初筛
48 个存活者费后 EV CI 下界>0 的为 0 个；lift 与 EV 相关性 r=0.16。
lift 是必要非充分条件，故 Q6 初筛第 5 步增修经济闸。

全程无 LLM、无 DB、纯函数：同一输入必得同一输出（bootstrap 固定 seed）。

口径（与 scripts/decision_bench.py / price_calibration_scan.py 对齐）：
- 谓词在决策点截断视图上命中 = 按该时刻真实 token 价买入 1 单位
  （curve_*_price 缺失的窗口回退 chance/100，并单列真实价覆盖率）
- 市场判定 = sign(actual_return)，=0 剔除（平盘结算规则未知）
- 逐注盈亏：赢 → (1-fee)/price - 1；输 → -1（price 已含溢价溢价防除零截断）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .predicates import evaluate_predicate
from .symbolizer import WindowView

# --- 宪法 V1.1 参数默认值 ---
DECISION_POINT_SEC: float = 150.0  # 决策点：开窗后 150s（第 10 采样点，规则 8）
FEE_RATE: float = 0.02  # 实测费率（2026-07-17 用户实盘确认）
PRICE_PREMIUM: float = 0.01  # 溢价保守情形（实测中位 +0.005 / p90 +0.015）
MIN_EV_FIRES: int = 10  # 经济闸最小注数（低于此 CI 无意义，判功效不足）
BOOT_N: int = 2000  # bootstrap 重抽次数
BOOT_SEED: int = 7  # 固定 seed（纯函数确定性）

# 符号化通道曲线（build_window_view 需 ≥2 点，截断后不足则保持原样由其防御跳过）
_CHANNEL_CURVE_KEYS = (
    "curve_up_pct",
    "curve_down_pct",
    "curve_btc_price",
    "curve_trade_volume",
)
# 入场价曲线：始终截断——单点亦是有效入场价，保持原样反而会让决策点后的
# 未来价格泄漏进 entry_price_at（经济账口径污染）
_PRICE_CURVE_KEYS = ("curve_up_price", "curve_down_price")


@dataclass(frozen=True)
class EvGateResult:
    """一条假设的经济账审判结果（经济闸的直接输入）。"""

    n_fires: int  # 截断视图上的命中注数
    n_real_price: int  # 其中使用真实 token 价的注数（其余为 chance 代理）
    win_rate: float | None  # 命中率（市场判定口径）
    avg_entry_price: float | None  # 平均入场价
    ev: float | None  # 费后 EV 点估计（含溢价）
    ev_ci_lower: float | None  # bootstrap 95% CI
    ev_ci_upper: float | None
    passed: bool  # 经济闸：n_fires >= min 且 CI 下界 > 0


def truncate_to_decision_point(
    windows: list[dict], keep_seconds: float = DECISION_POINT_SEC
) -> list[dict]:
    """把窗口的全部曲线截断到开窗后 keep_seconds（规则 8 决策点截断对齐）。

    三通道与价格曲线同步截断；有效点 <2 的曲线保持原样（由符号化层按
    既有防御跳过）。outcome / actual_return 保持整窗结算结果不变——
    模拟「在第 keep_seconds 秒决策、等整窗结算」。
    """
    out: list[dict] = []
    for w in windows:
        w2 = dict(w)
        for key in _CHANNEL_CURVE_KEYS + _PRICE_CURVE_KEYS:
            curve = w.get(key) or []
            if not curve:
                continue
            pts = sorted(curve, key=lambda p: p.get("t", 0))
            t0 = pts[0].get("t", 0)
            kept = [p for p in pts if (p.get("t", 0) - t0) <= keep_seconds * 1000.0]
            if key in _PRICE_CURVE_KEYS:
                w2[key] = kept  # 价格曲线始终截断（单点有效，防空也防泄漏）
            elif len(kept) >= 2:
                w2[key] = kept
        out.append(w2)
    return out


def _price_at(curve: list | None, start_ms: int, t_sec: float) -> float | None:
    """决策时刻价格：rel_t <= t_sec 的最后一个采样价（不偷看未来）。"""
    best = None
    for p in sorted(curve or [], key=lambda x: x.get("t", 0)):
        if (p.get("t", 0) - start_ms) / 1000.0 <= t_sec and p.get("v") is not None:
            best = float(p["v"])
    return best


def entry_price_at(
    window: dict, direction: str, t_sec: float = DECISION_POINT_SEC
) -> tuple[float | None, str]:
    """入场价提取：真实 token 价优先，缺失回退 chance/100。

    Returns:
        (price, kind)：kind ∈ {"real", "proxy", "missing"}；missing 时 price=None
    """
    start_ms = int(window.get("start_time", 0))
    if direction == "UP":
        p = _price_at(window.get("curve_up_price"), start_ms, t_sec)
        if p is not None and p > 0:
            return p, "real"
        c = _price_at(window.get("curve_up_pct"), start_ms, t_sec)
    else:
        p = _price_at(window.get("curve_down_price"), start_ms, t_sec)
        if p is not None and p > 0:
            return p, "real"
        c = _price_at(window.get("curve_down_pct"), start_ms, t_sec)
    if c is None or c <= 0:
        return None, "missing"
    return c / 100.0, "proxy"


def bet_pnl(win: bool, price: float, fee: float = FEE_RATE, premium: float = PRICE_PREMIUM) -> float:
    """逐注盈亏：赢 → (1-fee)/(price+premium) - 1；输 → -1。price 截断防除零。"""
    if not win:
        return -1.0
    return (1.0 - fee) / min(max(price + premium, 0.01), 0.99) - 1.0


def _bootstrap_ci(pnls: list[float]) -> tuple[float, float]:
    """EV 的 bootstrap 95% CI（对注单重抽，固定 seed 保证确定性）。"""
    a = np.asarray(pnls, dtype=float)
    if len(a) < MIN_EV_FIRES:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(a), size=(BOOT_N, len(a)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return (float(lo), float(hi))


def hypothesis_ev(
    predicate: dict,
    direction: str,
    views: list[WindowView],
    windows: list[dict],
    t_sec: float = DECISION_POINT_SEC,
    fee: float = FEE_RATE,
    premium: float = PRICE_PREMIUM,
    min_fires: int = MIN_EV_FIRES,
) -> EvGateResult:
    """一条谓词假设的逐注经济账（Q6 初筛第 5 步经济闸）。

    Args:
        predicate / direction: 假设的谓词 DSL 与目标方向（UP | DOWN）
        views: 决策点截断后的符号化视图（谓词执行输入）
        windows: 与 views 同序等长的截断窗口 dict（入场价与结算来源）
        t_sec: 决策点（开窗后秒数）
        fee / premium: 成本口径（宪法 V1.1：费 2% + 溢价 0.01 保守情形）
        min_fires: 最小注数，不足判经济功效不足（passed=False）

    防御：谓词求值异常视同不命中（与线上防御语义一致）；
    actual_return 缺失或为 0 的窗口剔除（平盘结算规则未知）。
    """
    fires: list[tuple[bool, float, str]] = []  # (win, price, price_kind)
    for view, w in zip(views, windows):
        try:
            if not evaluate_predicate(predicate, view):
                continue
        except Exception:
            continue
        ret = w.get("actual_return")
        if ret is None or float(ret) == 0.0:
            continue
        price, kind = entry_price_at(w, direction, t_sec)
        if price is None:
            continue
        resolution = "UP" if float(ret) > 0 else "DOWN"
        fires.append((direction == resolution, price, kind))

    n = len(fires)
    n_real = sum(1 for _, _, k in fires if k == "real")
    if n == 0:
        return EvGateResult(
            n_fires=0, n_real_price=0, win_rate=None, avg_entry_price=None,
            ev=None, ev_ci_lower=None, ev_ci_upper=None, passed=False,
        )

    wins = sum(1 for win, _, _ in fires if win)
    pnls = [bet_pnl(win, p, fee, premium) for win, p, _ in fires]
    ev = sum(pnls) / n
    lo, hi = _bootstrap_ci(pnls)
    passed = n >= min_fires and not math.isnan(lo) and lo > 0.0
    return EvGateResult(
        n_fires=n,
        n_real_price=n_real,
        win_rate=wins / n,
        avg_entry_price=sum(p for _, p, _ in fires) / n,
        ev=ev,
        ev_ci_lower=None if math.isnan(lo) else lo,
        ev_ci_upper=None if math.isnan(hi) else hi,
        passed=passed,
    )
