#!/usr/bin/env python3
"""决策点对照台架：k-NN vs 阈值笨规则，同口径同弃权 + 真实赔付经济账。

背景（由代码事实钉死的赔付规则）：
  Binance 5m 预测市场按「份额价格」下注——每个方向有 price(0~1)，判对每份结 1。
  下注那一刻价格锁定赔率：EV/注 = 胜率/price - 1（费前）。
  导出数据只有 chance 曲线（up%），本台架以 price ≈ up%/100 近似（价格即隐含
  概率，二者在此类市场基本贴合），并对手续费做敏感性标注。

对照设计（同一 holdout、同一弃权口径、同一决策时刻）：
  - naive(m): 决策时刻 up% >= 50+m 押 UP、<= 50-m 押 DOWN，否则弃权（m=0/5/10/15）
  - fade(m):  与人群反向——up% >= 50+m 押 DOWN、<= 50-m 押 UP（检验赔率不对称）
  - knn3:     24 维特征 + train z-score 标准化余弦 + top-3 多数投票，
              邻居全 NOISE / 平票弃权（与 feature_bench 满分线一致）

两套评估：
  (1) 准确率口径：holdout 中 outcome 为 UP/DOWN 的窗口，win = 押中 outcome。
  (2) 经济口径：holdout 中所有出手窗口（含 NOISE 标签，市场照样结算），
      以 sign(actual_return) 为市场判定，按锁定价格结算盈亏。

用法：
    python scripts/decision_bench.py --from-file sentiment_windows.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

import feature_bench as fb  # noqa: E402

DECISION_POINTS = (60.0, 90.0, 120.0)
NAIVE_MARGINS = (0.0, 5.0, 10.0, 15.0)
FADE_MARGINS = (10.0,)
# (手续费, 价格溢价) 组合：溢价=实际买入价高于 chance 的部分（实测中位 +0.005，p90 +0.015）
# (0,0)=理想情形；(0.02,0)=实测费率 2%（用户实盘确认）；(0.02,0.01)=费+溢价保守情形
COST_SCENARIOS = ((0.0, 0.0), (0.02, 0.0), (0.02, 0.01))
BOOT_N = 2000  # bootstrap 重抽次数
BOOT_SEED = 7


def _up_pct_at_decision(w: dict) -> float | None:
    """截断后曲线的最后一个采样点 = 决策时刻的 up%。"""
    curve = w.get("curve_up_pct") or []
    if not curve:
        return None
    pts = sorted(curve, key=lambda p: p.get("t", 0))
    return float(pts[-1].get("v", 0.0))


def _knn_decide(feat, train_feats, train_outcomes, std_ctx, k: int = 3) -> str | None:
    """与 feature_bench._knn_evaluate 同规则的单窗口裁决（返回方向或 None=弃权）。"""
    sims = [
        (fb.standardized_cosine_sim(feat, train_feats[i], std_ctx), i)
        for i in range(len(train_feats))
    ]
    sims.sort(key=lambda x: x[0], reverse=True)
    votes = Counter(
        train_outcomes[i] for _, i in sims[:k] if train_outcomes[i] in ("UP", "DOWN")
    )
    if not votes:
        return None
    top = votes.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def _boot_ci(arr: list[float], seed: int = BOOT_SEED) -> tuple[float, float]:
    """EV 的 bootstrap 95% 置信区间（对注单重抽，确定性 seed）。"""
    a = np.asarray(arr, dtype=float)
    if len(a) < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(BOOT_N, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _evaluate(holdout: list[dict], decisions: list[str | None]) -> dict:
    """双口径评估：准确率（decisive）+ 经济账（全出手，按锁定价结算）。"""
    # (1) 准确率口径
    n = correct = 0
    for w, side in zip(holdout, decisions):
        if side is None:
            continue
        actual = (w.get("outcome") or "").upper()
        if actual not in ("UP", "DOWN"):
            continue
        n += 1
        correct += int(side == actual)
    acc = fb._bucket_stats(correct, n)

    # (2) 经济口径：市场以 sign(actual_return) 结算；价格 ≈ 决策时刻 chance + 溢价
    profits: dict[tuple, list[float]] = {sc: [] for sc in COST_SCENARIOS}
    prices = []
    n_bets = 0
    for w, side in zip(holdout, decisions):
        if side is None:
            continue
        ret = w.get("actual_return")
        if ret is None or float(ret) == 0.0:
            continue  # 平盘极罕见且规则未知，不计
        v = _up_pct_at_decision(w)
        if v is None:
            continue
        resolution = "UP" if float(ret) > 0 else "DOWN"
        base_price = (v if side == "UP" else 100.0 - v) / 100.0
        prices.append(base_price)
        n_bets += 1
        win = side == resolution
        for fee, prem in COST_SCENARIOS:
            price = min(max(base_price + prem, 0.01), 0.99)  # 溢价+防除零
            profits[(fee, prem)].append((1.0 - fee) / price - 1.0 if win else -1.0)

    eco = {
        "n_bets": n_bets,
        "avg_price_paid": round(float(np.mean(prices)), 4) if prices else None,
        "market_win_rate": round(
            float(np.mean([p > 0 for p in profits[COST_SCENARIOS[0]]])), 4) if n_bets else None,
    }
    for fee, prem in COST_SCENARIOS:
        arr = profits[(fee, prem)]
        key = f"ev_fee{int(fee * 100)}_prem{int(prem * 100)}"
        if arr:
            lo, hi = _boot_ci(arr)
            eco[key] = round(float(np.mean(arr)), 4)
            eco[key + "_ci"] = [round(lo, 4), round(hi, 4)]
        else:
            eco[key] = None
            eco[key + "_ci"] = None
    return {
        "accuracy": {"n": n, "correct": correct,
                     "win_rate": acc.get("win_rate"), "ci_lower": acc.get("ci_lower")},
        "economics": eco,
        "abstained": sum(1 for s in decisions if s is None),
    }


def run(windows: list[dict], holdout_ratio: float, knn_k: int) -> dict:
    report: dict = {"holdout_ratio": holdout_ratio, "knn_k": knn_k, "points": {}}
    for t_sec in DECISION_POINTS:
        truncated = fb.truncate_windows(windows, t_sec)
        train, holdout = fb.time_split(truncated, holdout_ratio)
        # k-NN 上下文：train 全量特征 + z-score 几何（与满分线一致，只在 train 拟合）
        train_feats = [
            fb.default_feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or [])
            for w in train
        ]
        train_outcomes = [(w.get("outcome") or "NOISE").upper() for w in train]
        std_ctx = fb.make_std_ctx(np.asarray(train_feats, dtype=float))

        strategies: dict[str, list[str | None]] = {}
        # naive / fade
        ups = [_up_pct_at_decision(w) for w in holdout]
        for m in NAIVE_MARGINS:
            strategies[f"naive{int(m)}"] = [
                None if v is None else ("UP" if v >= 50 + m else "DOWN" if v <= 50 - m else None)
                for v in ups
            ]
        for m in FADE_MARGINS:
            strategies[f"fade{int(m)}"] = [
                None if v is None else ("DOWN" if v >= 50 + m else "UP" if v <= 50 - m else None)
                for v in ups
            ]
        # knn
        strategies[f"knn{knn_k}"] = [
            _knn_decide(
                fb.default_feature_fn(w.get("curve_up_pct") or [], w.get("curve_down_pct") or []),
                train_feats, train_outcomes, std_ctx, knn_k,
            )
            for w in holdout
        ]

        report["points"][f"t{int(t_sec)}"] = {
            "n_holdout": len(holdout),
            "strategies": {name: _evaluate(holdout, dec) for name, dec in strategies.items()},
        }
    return report


def _print(report: dict) -> None:
    for tkey, pdata in report["points"].items():
        print("=" * 88)
        print(f"决策点 {tkey}  (holdout={pdata['n_holdout']})")
        print("-" * 88)
        print(f"{'策略':<10}{'出手':>6}{'弃权':>6}{'胜率(决出)':>12}{'WLB':>8}"
              f"{'均价':>8}{'EV 理想':>22}{'EV 费2%':>10}{'EV 费+溢价':>10}")
        for name, ev in pdata["strategies"].items():
            a, e = ev["accuracy"], ev["economics"]
            ci = e.get("ev_fee0_prem0_ci") or ["-", "-"]
            ev0 = f"{e['ev_fee0_prem0']} [{ci[0]},{ci[1]}]"
            print(f"{name:<10}{e['n_bets']:>6}{ev['abstained']:>6}"
                  f"{str(a['win_rate']):>12}{str(a['ci_lower']):>8}"
                  f"{str(e['avg_price_paid']):>8}{ev0:>22}"
                  f"{str(e['ev_fee2_prem0']):>10}{str(e['ev_fee2_prem1']):>10}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="决策点对照台架：策略×经济账")
    ap.add_argument("--from-file", default="windows_with_price.json")
    ap.add_argument("--holdout-ratio", type=float, default=0.3)
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--out", default="output/decision_bench.json")
    args = ap.parse_args()

    windows = fb.load_windows_from_file(args.from_file)
    report = run(windows, args.holdout_ratio, args.knn)
    _print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[已写入] {args.out}")


if __name__ == "__main__":
    main()
