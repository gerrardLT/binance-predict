"""数据层：K 线 CSV 加载、泛化周期聚合、连续性检查、data_summary。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass
class Klines:
    """一个周期的 OHLCV 数组集（升序、已收盘）。"""

    t: np.ndarray  # int64 open_time ms
    o: np.ndarray  # float64
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    cont: np.ndarray  # bool：与前一根严格相邻（首根 False）

    def __len__(self) -> int:
        return len(self.t)


def load_klines_csv(path: str, bar_ms: int) -> Klines:
    """读 klines_*_720d.csv（timestamp,open,high,low,close,volume，ISO UTC 时间戳）。"""
    ts: list[int] = []
    rows: list[tuple[float, float, float, float, float]] = []
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        if not header.startswith("timestamp,open,high,low,close,volume"):
            raise ValueError(f"CSV 表头不符: {path}")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) != 6:
                continue
            ts.append(int(datetime.fromisoformat(parts[0]).timestamp() * 1000))
            rows.append((float(parts[1]), float(parts[2]), float(parts[3]),
                         float(parts[4]), float(parts[5])))
    t = np.asarray(ts, dtype=np.int64)
    ohlcv = np.asarray(rows, dtype=np.float64)
    cont = np.zeros(len(t), dtype=bool)
    if len(t) > 1:
        cont[1:] = (t[1:] - t[:-1]) == bar_ms
    return Klines(t=t, o=ohlcv[:, 0], h=ohlcv[:, 1], l=ohlcv[:, 2],
                  c=ohlcv[:, 3], v=ohlcv[:, 4], cont=cont)


def aggregate_to(kl: Klines, bar_ms: int) -> Klines:
    """基周期 → 更大周期的桶聚合（只保留基线根齐全的完整周期）。

    复刻 backtest.data.aggregate_15m 的桶映射：open=桶首根 open、high=max、
    low=min、close=桶末根 close、volume=sum。聚合后的 cont 要求桶间严格相邻
    且桶内全部相邻。
    """
    sub_ms = int(kl.t[1] - kl.t[0]) if len(kl.t) > 1 else 300_000
    n_sub = bar_ms // sub_ms
    if n_sub <= 1 or bar_ms % sub_ms != 0:
        raise ValueError(f"聚合周期 {bar_ms} 必须是基周期 {sub_ms} 的整数倍")
    bkt = kl.t // bar_ms
    uniq, first = np.unique(bkt, return_index=True)
    counts = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(counts, np.searchsorted(uniq, bkt), 1)
    full = counts == n_sub
    # 桶内连续性：桶内存在断点（cont=False）则该桶整体不可用
    inner_ok = np.ones(len(uniq), dtype=bool)
    if len(kl.t) > 1:
        gap_idx = np.nonzero(~kl.cont)[0]
        bad_bkt = np.unique(bkt[gap_idx])
        inner_ok[np.searchsorted(uniq, bad_bkt)] = False
    # 全桶 reduceat 先算段统计（段边界=桶边界），再按 keep 过滤——
    # 不能先过滤 first 再 reduceat（会把被丢弃桶的行并入相邻段）
    ends = np.append(first[1:], len(kl.t))
    h_all = np.maximum.reduceat(kl.h, first)
    l_all = np.minimum.reduceat(kl.l, first)
    c_all = kl.c[ends - 1]
    v_all = np.add.reduceat(kl.v, first)
    keep = full & inner_ok
    t = (uniq[keep] * bar_ms).astype(np.int64)
    cont = np.zeros(len(t), dtype=bool)
    if len(t) > 1:
        cont[1:] = (t[1:] - t[:-1]) == bar_ms
    return Klines(t=t, o=kl.o[first[keep]], h=h_all[keep], l=l_all[keep],
                  c=c_all[keep], v=v_all[keep], cont=cont)


def data_summary(kl: Klines, bar_ms: int) -> dict:
    """与既有产物 run_config.data_summary 同构的数据体检。"""
    median_gap = float(np.median(np.diff(kl.t))) if len(kl.t) > 1 else bar_ms
    gaps = int((np.diff(kl.t) > 1.5 * median_gap).sum()) if len(kl.t) > 1 else 0
    return {
        "rows": len(kl.t),
        "start": datetime.fromtimestamp(int(kl.t[0]) / 1000, tz=timezone.utc).isoformat(),
        "end": datetime.fromtimestamp(int(kl.t[-1]) / 1000, tz=timezone.utc).isoformat(),
        "median_bar_seconds": median_gap / 1000,
        "gap_count_gt_1_5x_median": gaps,
    }
