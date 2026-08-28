"""目标层：延续/反转标签与交易语义收益（胜率、收益、MFE/MAE）。

标签定义（与既有 720d 产物一致）：
- continuation_h：第 t+h 根方向 == 第 t 根方向（h 步延续）
- reversal_h：第 t+h 根方向 == 第 t 根方向的反向（h 步反转）
有效样本：两端方向均非零且 t→t+h 全程连续（无断点）。

交易语义（盈亏比原料）：信号根收盘按期望方向入场，第 t+h 根收盘离场：
- ret = 期望方向上的区间收益；win = ret > 0
- MFE/MAE = 持有期内期望方向最大顺逆波动（用 high/low 路径），以当根前置 ATR 归一
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class TargetSet:
    """单个目标（如 continuation_1）的全量数组。"""

    name: str
    family: str  # continuation | reversal
    horizon: int
    valid: np.ndarray  # bool
    win: np.ndarray  # bool（仅 valid 处有意义）
    ret: np.ndarray  # float64 期望方向区间收益
    mfe_atr: np.ndarray
    mae_atr: np.ndarray


@dataclass
class Targets:
    names: list[str] = field(default_factory=list)
    items: dict[str, TargetSet] = field(default_factory=dict)

    def add(self, ts: TargetSet) -> None:
        self.names.append(ts.name)
        self.items[ts.name] = ts


def _fwd_ok(cont: np.ndarray, h: int) -> np.ndarray:
    """t→t+h 全程连续：cont[t+1..t+h] 全 True。"""
    n = len(cont)
    out = np.zeros(n, dtype=bool)
    if n <= h:
        return out
    # cont[1:] 的位置 j 表示根 j 与 j-1 相邻；窗口覆盖 cont[i+1..i+h]
    out[: n - h] = sliding_window_view(cont[1:], h).all(axis=1)
    return out


def build_targets(t: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                  c: np.ndarray, cont: np.ndarray, horizons: list[int],
                  atr_abs: np.ndarray) -> Targets:
    """构建 continuation_h / reversal_h 全目标集（仅依赖未来信息，属标签非特征）。"""
    tg = Targets()
    n = len(t)
    dir_ = np.sign(c - o)
    for hz in horizons:
        ok_fwd = _fwd_ok(cont, hz)
        nxt_dir = np.zeros(n, dtype=np.float64)
        nxt_dir[: n - hz] = dir_[hz:]
        nxt_c = np.full(n, np.nan)
        nxt_c[: n - hz] = c[hz:]
        # 持有期路径极值（未来 hz 根的 high/low）
        if n > hz:
            max_h = np.full(n, np.nan)
            min_l = np.full(n, np.nan)
            max_h[: n - hz] = sliding_window_view(h[1:], hz).max(axis=1)
            min_l[: n - hz] = sliding_window_view(l[1:], hz).min(axis=1)
        else:
            max_h = min_l = np.full(n, np.nan)
        base_valid = (dir_ != 0) & (nxt_dir != 0) & ok_fwd & np.isfinite(atr_abs) & (atr_abs > 0)
        for fam, sign in (("continuation", 1.0), ("reversal", -1.0)):
            valid = base_valid
            if fam == "continuation":
                win = nxt_dir == dir_
            else:
                win = nxt_dir == -dir_
            d = sign * dir_  # 期望方向（+1 做多 / -1 做空）
            with np.errstate(invalid="ignore", divide="ignore"):
                ret = d * (nxt_c - c) / c
                mfe = np.where(d > 0, max_h - c, c - min_l) / (atr_abs * c)
                mae = np.where(d > 0, c - min_l, max_h - c) / (atr_abs * c)
            ts = TargetSet(
                name=f"{fam}_{hz}", family=fam, horizon=hz,
                valid=valid & np.isfinite(ret),
                win=np.where(valid, win, False),
                ret=np.where(np.isfinite(ret), ret, np.nan),
                mfe_atr=np.where(np.isfinite(mfe), mfe, np.nan),
                mae_atr=np.where(np.isfinite(mae), mae, np.nan),
            )
            tg.add(ts)
    return tg


def seg_bounds(n: int, discovery_frac: float = 0.6, validation_frac: float = 0.2) -> tuple[int, int]:
    """时序三段切分边界（发现/验证/冻结 holdout），对齐既有 720d 产物。"""
    i1 = int(n * discovery_frac)
    i2 = int(n * (discovery_frac + validation_frac))
    return i1, i2
