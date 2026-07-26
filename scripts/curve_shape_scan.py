#!/usr/bin/env python3
"""情绪曲线形态分布扫描（离线、确定性、无 LLM、无网络）。

目的（对齐"数据现象 -> 统计验证 -> 才下结论"）：
把每条 up% 情绪曲线压成一组可量化的「形态描述量」（峰值位置 / 冲高幅度 /
回落幅度 / 穿越 50 次数 / 反转次数 / 单调性 / 末段斜率 ...），先看它们在
1748 条真实窗口上的**整体分布**，再按最终 outcome（UP/DOWN/NOISE）拆分，
量化「哪些形态量天生就能分开涨跌」（用 UP vs DOWN 的标准化均值差 Cohen's d）。

这是所有后续「形态规则」的地基：让阈值由数据本身决定，而不是拍脑袋。

用法：
    python scripts/curve_shape_scan.py --from-file sentiment_windows.json
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


def _curve_vals(curve: list) -> np.ndarray:
    """曲线 [{t, v}] -> 按 t 升序的 v 数组（确定性）。"""
    if not curve:
        return np.array([], dtype=float)
    pts = sorted(curve, key=lambda p: p.get("t", 0))
    return np.array([float(p.get("v", 0.0)) for p in pts], dtype=float)


def _shape_descriptors(vals: np.ndarray) -> dict:
    """一条 up% 曲线 -> 一组形态描述量（纯情绪曲线，不含价格）。"""
    n = len(vals)
    if n < 2:
        return {}
    first, last = float(vals[0]), float(vals[-1])
    peak_i = int(np.argmax(vals))
    trough_i = int(np.argmin(vals))
    peak, trough = float(vals[peak_i]), float(vals[trough_i])
    diffs = np.diff(vals)
    steps = len(diffs)
    up_steps = int(np.sum(diffs > 0))
    down_steps = int(np.sum(diffs < 0))
    # 方向反转次数（相邻步的符号变化）
    signs = np.sign(diffs)
    nz = signs[signs != 0]
    reversals = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
    # 穿越 50 次数
    side = np.sign(vals - 50.0)
    side_nz = side[side != 0]
    cross50 = int(np.sum(side_nz[1:] != side_nz[:-1])) if len(side_nz) > 1 else 0
    # 最大回撤（峰后回落）/ 最大升幅（谷后拉升）
    run_max = np.maximum.accumulate(vals)
    max_drawdown = float(np.max(run_max - vals))
    run_min = np.minimum.accumulate(vals)
    max_runup = float(np.max(vals - run_min))
    # 末段斜率（最后 3 点线性趋势，点数不足则退化）
    tail_n = min(3, n)
    tail = vals[-tail_n:]
    end_slope = float(tail[-1] - tail[0]) / max(1, tail_n - 1)
    return {
        "first": first,
        "last": last,
        "net": last - first,                       # 全程净漂移
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),                # 摇摆幅度
        "mean_dist_50": float(abs(np.mean(vals) - 50.0)),  # 共识强度
        "peak": peak,
        "peak_pos": peak_i / (n - 1),              # 峰值在窗口的相对位置 0~1
        "trough": trough,
        "trough_pos": trough_i / (n - 1),
        "rise_to_peak": peak - first,              # 起点到峰的冲高
        "fall_from_peak": peak - last,             # 峰到终点的回落（"冲高回落"）
        "max_drawdown": max_drawdown,
        "max_runup": max_runup,
        "reversals": reversals,                    # 犹豫程度
        "cross50": cross50,
        "monotonicity": max(up_steps, down_steps) / steps,  # 单调性 0.5~1
        "end_slope": end_slope,                    # 末段动量
        "end_dist_50": float(abs(last - 50.0)),    # 结算时共识强度
    }


DESC_KEYS = [
    "first", "last", "net", "mean", "std", "mean_dist_50",
    "peak", "peak_pos", "trough", "trough_pos",
    "rise_to_peak", "fall_from_peak", "max_drawdown", "max_runup",
    "reversals", "cross50", "monotonicity", "end_slope", "end_dist_50",
]


def _pctiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {}
    ps = [0, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(arr, ps)
    return {f"p{p}": round(float(v), 3) for p, v in zip(ps, vals)}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """UP vs DOWN 的标准化均值差：|d| 越大 = 该描述量越能分开涨跌。"""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def scan(windows: list[dict]) -> dict:
    rows = []
    lengths = []
    samples = []
    for w in windows:
        vals = _curve_vals(w.get("curve_up_pct") or [])
        if len(vals) < 2:
            continue
        d = _shape_descriptors(vals)
        d["outcome"] = (w.get("outcome") or "NOISE").upper()
        d["actual_return"] = w.get("actual_return")
        rows.append(d)
        lengths.append(len(vals))
        if w.get("sample_count") is not None:
            samples.append(int(w["sample_count"]))

    outcomes = [r["outcome"] for r in rows]
    n_total = len(rows)
    counts = {o: outcomes.count(o) for o in ("UP", "DOWN", "NOISE")}

    # 每个描述量：整体分布 + 按 outcome 均值 + UP/DOWN 分离度(Cohen's d)
    per_desc = {}
    for key in DESC_KEYS:
        allv = np.array([r[key] for r in rows], dtype=float)
        up = np.array([r[key] for r in rows if r["outcome"] == "UP"], dtype=float)
        dn = np.array([r[key] for r in rows if r["outcome"] == "DOWN"], dtype=float)
        no = np.array([r[key] for r in rows if r["outcome"] == "NOISE"], dtype=float)
        per_desc[key] = {
            "overall": _pctiles(allv),
            "mean_UP": round(float(np.mean(up)), 3) if len(up) else None,
            "mean_DOWN": round(float(np.mean(dn)), 3) if len(dn) else None,
            "mean_NOISE": round(float(np.mean(no)), 3) if len(no) else None,
            "cohens_d_UP_vs_DOWN": round(_cohens_d(up, dn), 3),
        }

    return {
        "n_windows": n_total,
        "outcome_counts": counts,
        "outcome_pct": {k: round(v / n_total, 3) for k, v in counts.items()},
        "curve_length": _pctiles(np.array(lengths, dtype=float)),
        "sample_count": _pctiles(np.array(samples, dtype=float)) if samples else {},
        "descriptors": per_desc,
    }


def _print_report(rep: dict) -> None:
    print("=" * 70)
    print("情绪曲线形态分布扫描")
    print("=" * 70)
    print(f"窗口总数: {rep['n_windows']}")
    print(f"outcome 分布: {rep['outcome_counts']}  占比: {rep['outcome_pct']}")
    print(f"曲线采样点数分布: {rep['curve_length']}")
    if rep["sample_count"]:
        print(f"参与人数(sample_count)分布: {rep['sample_count']}")
    print()
    print("-" * 70)
    print("形态描述量：分离度排行（|Cohen's d| 越大 = 越能分开 UP/DOWN）")
    print("-" * 70)
    ranked = sorted(
        rep["descriptors"].items(),
        key=lambda kv: abs(kv[1]["cohens_d_UP_vs_DOWN"]),
        reverse=True,
    )
    print(f"{'描述量':<16}{'d(UP-DOWN)':>12}{'均值UP':>10}{'均值DOWN':>10}{'均值NOISE':>10}")
    for key, v in ranked:
        print(f"{key:<16}{v['cohens_d_UP_vs_DOWN']:>12}"
              f"{str(v['mean_UP']):>10}{str(v['mean_DOWN']):>10}{str(v['mean_NOISE']):>10}")
    print()
    print("-" * 70)
    print("关键描述量整体分布（分位数，用于定阈值）")
    print("-" * 70)
    for key, v in ranked:
        print(f"{key:<16} {v['overall']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="情绪曲线形态分布扫描")
    parser.add_argument("--from-file", default="sentiment_windows.json",
                        help="服务器导出的窗口 JSON")
    parser.add_argument("--out", default="output/curve_shape_scan.json")
    args = parser.parse_args()

    windows = fb.load_windows_from_file(args.from_file)
    rep = scan(windows)
    _print_report(rep)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        print(f"\n[已写入] {args.out}")


if __name__ == "__main__":
    main()
