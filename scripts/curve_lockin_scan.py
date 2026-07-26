#!/usr/bin/env python3
"""情绪曲线「答案锁定度」逐时间点分析（离线、确定性、无 LLM、无网络）。

回答两个由数据决定、不能拍脑袋的问题：
  (1) 到窗口的第几个时间点（约第几秒），仅凭当下 up% 猜方向，就已经几乎必对？
      —— 即「结果何时算定」的时间地平线。
  (2) up% 冲到多少阈值时，最终结果就基本锁死（高精度）？
      —— 即「达到什么阈值算定」的数值地平线。

做法：
  - 把每条曲线重采样到统一的 N=20 格相对时间轴（线性插值），使各窗口可逐格对齐。
  - 只用 decisive 窗口（UP/DOWN）算方向准确率；NOISE 单独统计。
  - 逐格计算：方向准确率、UP/DOWN 分离度、达阈值后的精度与覆盖率。
  - 统计每条曲线「符号最终锁定」发生在第几格 -> 锁定时间分布。

用法：
    python scripts/curve_lockin_scan.py --from-file sentiment_windows.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

import feature_bench as fb  # noqa: E402

N_GRID = 20          # 统一时间格数
WINDOW_SECONDS = 300  # 5 分钟


def _curve_vals(curve: list) -> np.ndarray:
    if not curve:
        return np.array([], dtype=float)
    pts = sorted(curve, key=lambda p: p.get("t", 0))
    return np.array([float(p.get("v", 0.0)) for p in pts], dtype=float)


def _resample(vals: np.ndarray, n: int = N_GRID) -> np.ndarray:
    """把任意长度曲线线性插值到 n 格相对时间轴 [0,1]。"""
    if len(vals) == n:
        return vals.astype(float)
    src = np.linspace(0.0, 1.0, len(vals))
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, vals)


def _lock_index(vals: np.ndarray) -> int:
    """符号最终锁定的格位：从该格起，sign(up%-50) 与最终符号一致且不再翻转。

    返回 0..n-1；0 表示开窗即锁定，n-1 表示到最后一刻才锁定。
    """
    final_side = np.sign(vals[-1] - 50.0)
    if final_side == 0:
        return len(vals) - 1
    sides = np.sign(vals - 50.0)
    # 最后一个与最终符号不一致的位置 + 1
    mismatch = np.where(sides != final_side)[0]
    if len(mismatch) == 0:
        return 0
    return int(mismatch[-1]) + 1


def _pctiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {}
    ps = [0, 10, 25, 50, 75, 90, 100]
    return {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in ps}


def analyze(windows: list[dict]) -> dict:
    grids, outcomes = [], []
    for w in windows:
        vals = _curve_vals(w.get("curve_up_pct") or [])
        if len(vals) < 2:
            continue
        grids.append(_resample(vals))
        outcomes.append((w.get("outcome") or "NOISE").upper())
    G = np.vstack(grids)          # [n_windows, N_GRID]
    outc = np.array(outcomes)
    is_up = outc == "UP"
    is_dn = outc == "DOWN"
    decisive = is_up | is_dn

    # 决定方向的真值：UP=+1, DOWN=-1
    truth = np.zeros(len(outc))
    truth[is_up] = 1
    truth[is_dn] = -1

    seconds = [round(i / (N_GRID - 1) * WINDOW_SECONDS, 1) for i in range(N_GRID)]

    per_pos = []
    for i in range(N_GRID):
        col = G[:, i]
        up_vals = col[is_up]
        dn_vals = col[is_dn]
        # 方向准确率：仅在 decisive 上，预测 = sign(up%-50)
        dcol = col[decisive]
        dtruth = truth[decisive]
        pred = np.sign(dcol - 50.0)
        pred[pred == 0] = 1  # 恰好 50 记为押涨（边界，占比极小）
        acc = float(np.mean(pred == dtruth))
        # 分离度
        d = 0.0
        if len(up_vals) > 1 and len(dn_vals) > 1:
            va, vb = np.var(up_vals, ddof=1), np.var(dn_vals, ddof=1)
            pooled = np.sqrt(((len(up_vals) - 1) * va + (len(dn_vals) - 1) * vb)
                             / (len(up_vals) + len(dn_vals) - 2))
            if pooled > 0:
                d = float((np.mean(up_vals) - np.mean(dn_vals)) / pooled)
        # 阈值门：up%>=thr 时，decisive 中最终=UP 的精度与覆盖率
        gates = {}
        for thr in (60, 70, 80, 90):
            hi = decisive & (col >= thr)
            lo = decisive & (col <= 100 - thr)
            n_hi, n_lo = int(np.sum(hi)), int(np.sum(lo))
            p_up = float(np.mean(is_up[hi])) if n_hi else None
            p_dn = float(np.mean(is_dn[lo])) if n_lo else None
            gates[f"thr{thr}"] = {
                "up_ge": {"n": n_hi, "P_final_UP": round(p_up, 3) if p_up is not None else None,
                          "coverage": round(n_hi / max(1, int(np.sum(decisive))), 3)},
                "dn_le": {"n": n_lo, "P_final_DOWN": round(p_dn, 3) if p_dn is not None else None,
                          "coverage": round(n_lo / max(1, int(np.sum(decisive))), 3)},
            }
        per_pos.append({
            "pos": i,
            "t_sec": seconds[i],
            "dir_accuracy": round(acc, 4),
            "cohens_d": round(d, 3),
            "mean_UP": round(float(np.mean(up_vals)), 2) if len(up_vals) else None,
            "mean_DOWN": round(float(np.mean(dn_vals)), 2) if len(dn_vals) else None,
            "gates": gates,
        })

    # 符号锁定时间分布（仅 decisive）
    lock_idx = np.array([_lock_index(G[j]) for j in range(len(G)) if decisive[j]])
    lock_sec = lock_idx / (N_GRID - 1) * WINDOW_SECONDS

    return {
        "n_windows": int(len(G)),
        "n_decisive": int(np.sum(decisive)),
        "n_noise": int(np.sum(~decisive)),
        "per_position": per_pos,
        "lock_index_pctiles": _pctiles(lock_idx.astype(float)),
        "lock_seconds_pctiles": _pctiles(lock_sec),
    }


def _print(rep: dict) -> None:
    print("=" * 74)
    print("情绪曲线「答案锁定度」逐时间点分析")
    print("=" * 74)
    print(f"窗口 {rep['n_windows']}  decisive(UP/DOWN) {rep['n_decisive']}  NOISE {rep['n_noise']}")
    print()
    print("逐格：方向准确率 / 分离度 / UP·DOWN 均值 / (up%>=70 时最终UP精度, 覆盖率)")
    print("-" * 74)
    hdr = f"{'格':>3}{'秒':>7}{'方向准确率':>11}{'d':>8}{'均值UP':>9}{'均值DN':>9}{'P(UP|>=70)':>12}{'覆盖':>7}"
    print(hdr)
    for p in rep["per_position"]:
        g70 = p["gates"]["thr70"]["up_ge"]
        print(f"{p['pos']:>3}{p['t_sec']:>7}{p['dir_accuracy']:>11}{p['cohens_d']:>8}"
              f"{str(p['mean_UP']):>9}{str(p['mean_DOWN']):>9}"
              f"{str(g70['P_final_UP']):>12}{str(g70['coverage']):>7}")
    print()
    print(f"符号最终锁定的时间分布（decisive）: 格 {rep['lock_index_pctiles']}")
    print(f"                                    秒 {rep['lock_seconds_pctiles']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="情绪曲线答案锁定度分析")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default="output/curve_lockin_scan.json")
    args = ap.parse_args()

    windows = fb.load_windows_from_file(args.from_file)
    rep = analyze(windows)
    _print(rep)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
