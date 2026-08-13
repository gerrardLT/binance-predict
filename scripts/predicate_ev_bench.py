#!/usr/bin/env python3
"""谓词假设的经济账台架：Q6 lift 初筛存活者，在入场价口径下 EV 为正吗？

背景（docs/runbook-decision-bench.md，2026-07-17 定论）：方向预测力真实存在，
但市场定价基本有效——naive/fade/kNN 费前 EV≈0、费后为负。科学发现系统
（宪法 Q0~Q8）是对"简单策略打不赢价格"的回应，但其验证基准是 lift（vs ±3 天
局部基准），从未与经济口径（入场价 EV）对接。本台架补上这一环。

方法（离线、确定性、无 LLM、无网络）：
  1. 枚举 V1 DSL 假设空间（Q5 白名单全组合）替代 LLM 提案——LLM 只是提案器，
     初筛才是被检验对象；枚举空间是 LLM 任何提案的超集，结论对提案分布稳健。
  2. time_split 70/30；分箱快照只从 train 全窗差值冻结（Q4，与线上一致）。
  3. 两版初筛对照（真实 screen_hypotheses 流水线，一字不改）：
     - Run A（现管线忠实版）：holdout 全窗符号化 → 初筛。即 deep_learn 今日行为。
     - Run B（截断错位修正版）：holdout 截断到决策点（前 150s ≈ 第 10 采样点）
       再符号化 → 初筛。predict 在线只能看到前 150s，A/B 之差即截断错位代价。
  4. 经济账（唯一诚实口径）：谓词在截断视图上命中 = 第 150s 下注 1 单位，
     入场价取该时刻真实 curve_up_price/curve_down_price（缺失窗口回退
     chance/100 并单列覆盖率）；市场判定 = sign(actual_return)，=0 剔除
     （与 decision_bench 一致）。EV/注 = (1-fee)/price - 1（赢）或 -1（输）。
     成本情形沿用 runbook 一票否决口径：(0,0)/(2%,0)/(2%+溢价0.01)。

回答的问题：
  - Q6 裁决 ACTIVE/OBSERVE 的假设，费 2%+溢价下 EV bootstrap CI 下界>0 的有几个？
  - 被 REJECT 的假设里有没有经济上的漏网之鱼（初筛方向性错误）？
  - Run A vs Run B 存活集合差异 = 全窗初筛的截断错位有多大？

用法：
    python scripts/predicate_ev_bench.py --from-file sentiment_windows.json
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
from binance_predict.services.discovery import screen_hypotheses  # noqa: E402
from binance_predict.services.predicates import evaluate_predicate  # noqa: E402
from binance_predict.services.symbolizer import (  # noqa: E402
    SYMBOLS,
    build_window_view,
    compute_channel_snapshots,
)

# --- 台架参数（与 runbook / decision_bench 对齐）---
DECISION_T_SEC = 150.0  # predict 触发点：第 10 采样点 ≈ 开窗后 150s
HOLDOUT_RATIO = 0.3  # 宪法参数表
COST_SCENARIOS = ((0.0, 0.0), (0.02, 0.0), (0.02, 0.01))  # (费率, 溢价)
VETO_SCENARIO = (0.02, 0.01)  # runbook 一票否决口径
BOOT_N = 2000
BOOT_SEED = 7
MIN_EV_FIRES = 10  # 低于此命中数不做 bootstrap（CI 无意义）

# 截断时需同步截断的全部曲线键（三通道 + 价格曲线；fb.truncate_windows 只截 pct）
_TRUNC_KEYS = (
    "curve_up_pct",
    "curve_down_pct",
    "curve_btc_price",
    "curve_trade_volume",
)


# ============================================================
# V1 DSL 假设空间枚举（Q5 白名单，替代 LLM 提案）
# ============================================================

def enumerate_predicates() -> list[dict]:
    """V1 合法谓词的全组合（单层原子 + L2 关系，不含逻辑组合节点）。

    组合深度 0~1 层已覆盖 LLM 提案的绝大多数形态；逻辑组合（AND/OR/NOT）
    的命集是原子命集的布尔函数，其统计性质由原子假设的联合分布决定，
    不在本轮枚举（控制多重检验规模，与发现预算 ≤20 假设同量级 × 枚举冗余）。
    """
    preds: list[dict] = []
    channels = ("sentiment", "price", "volume")
    ch_pairs = [(a, b) for a in channels for b in channels if a != b]

    # count_symbol：channel × symbol × >= × {1,2,3}
    for ch in channels:
        for sym in SYMBOLS:
            for k in (1, 2, 3):
                preds.append({"pred": "count_symbol", "channel": ch,
                              "symbol": sym, "cmp": ">=", "value": k})
    # symbol_at：channel × segment × symbol
    for ch in channels:
        for seg in ("early", "mid", "late"):
            for sym in SYMBOLS:
                preds.append({"pred": "symbol_at", "channel": ch,
                              "segment": seg, "symbol": sym})
    # has_subseq：channel × 有序符号对（含同符号=连续同向）
    for ch in channels:
        for s1 in SYMBOLS:
            for s2 in SYMBOLS:
                preds.append({"pred": "has_subseq", "channel": ch,
                              "symbols": [s1, s2]})
    # peak_count：channel × >= × {1,2,3}
    for ch in channels:
        for k in (1, 2, 3):
            preds.append({"pred": "peak_count", "channel": ch,
                          "cmp": ">=", "value": k})
    # extremum_spacing：channel × trend
    for ch in channels:
        for trend in ("shrinking", "expanding", "mixed"):
            preds.append({"pred": "extremum_spacing", "channel": ch, "trend": trend})
    # lead：有序通道对 × k{1,2,3} × min_matches{1,2,3}
    for a, b in ch_pairs:
        for k in (1, 2, 3):
            for mm in (1, 2, 3):
                preds.append({"pred": "lead", "channel_a": a, "channel_b": b,
                              "k": k, "min_matches": mm})
    # sync：有序通道对 × >= × {0.5..0.9}
    for a, b in ch_pairs:
        for v in (0.5, 0.6, 0.7, 0.8, 0.9):
            preds.append({"pred": "sync", "channel_a": a, "channel_b": b,
                          "cmp": ">=", "value": v})
    return preds


# ============================================================
# 截断 / 入场价 / EV
# ============================================================

def truncate3(windows: list[dict], keep_seconds: float) -> list[dict]:
    """三通道 + 价格曲线同步截断到开窗后 keep_seconds（fb.truncate_windows 扩展版）。"""
    out = []
    for w in windows:
        w2 = dict(w)
        for key in _TRUNC_KEYS + ("curve_up_price", "curve_down_price"):
            curve = w.get(key) or []
            if not curve:
                continue
            pts = sorted(curve, key=lambda p: p.get("t", 0))
            t0 = pts[0].get("t", 0)
            kept = [p for p in pts if (p.get("t", 0) - t0) <= keep_seconds * 1000.0]
            if len(kept) >= 2:
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


def entry_price(w: dict, direction: str, t_sec: float) -> tuple[float | None, str]:
    """入场价：真实 curve_*_price 优先；缺失回退 chance/100（返回口径标记）。"""
    start_ms = w.get("start_time", 0)
    if direction == "UP":
        p = _price_at(w.get("curve_up_price"), start_ms, t_sec)
        if p is not None and p > 0:
            return p, "real"
        c = _price_at(w.get("curve_up_pct"), start_ms, t_sec)
    else:
        p = _price_at(w.get("curve_down_price"), start_ms, t_sec)
        if p is not None and p > 0:
            return p, "real"
        c = _price_at(w.get("curve_down_pct"), start_ms, t_sec)
    if c is None or c <= 0:
        return None, "missing"
    return c / 100.0, "proxy"


def _boot_ci(arr: list[float]) -> tuple[float, float]:
    a = np.asarray(arr, dtype=float)
    if len(a) < MIN_EV_FIRES:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(a), size=(BOOT_N, len(a)))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return (float(lo), float(hi))


def ev_eval(pred: dict, direction: str, trunc_views, trunc_windows) -> dict:
    """谓词在截断视图上的逐注经济账：命中=按入场价买 1 单位 direction 方向。"""
    fires: list[tuple[bool, float, str]] = []  # (win, price, price_kind)
    for view, w in zip(trunc_views, trunc_windows):
        try:
            if not evaluate_predicate(pred, view):
                continue
        except Exception:
            continue  # 求值异常视同不命中（与线上防御语义一致，逐注计数不受影响）
        ret = w.get("actual_return")
        if ret is None or float(ret) == 0.0:
            continue  # 平盘结算规则未知，剔除（decision_bench 口径）
        price, kind = entry_price(w, direction, DECISION_T_SEC)
        if price is None:
            continue
        resolution = "UP" if float(ret) > 0 else "DOWN"
        fires.append((direction == resolution, price, kind))

    out: dict = {"n_fires": len(fires),
                 "n_real_price": sum(1 for _, _, k in fires if k == "real")}
    if not fires:
        return out
    wins = sum(1 for win, _, _ in fires if win)
    out["win_rate"] = round(wins / len(fires), 4)
    out["avg_entry_price"] = round(float(np.mean([p for _, p, _ in fires])), 4)
    for fee, prem in COST_SCENARIOS:
        arr = [
            (1.0 - fee) / min(max(p + prem, 0.01), 0.99) - 1.0 if win else -1.0
            for win, p, _ in fires
        ]
        key = f"ev_fee{int(fee * 100)}_prem{int(prem * 100)}"
        lo, hi = _boot_ci(arr)
        out[key] = round(float(np.mean(arr)), 4)
        out[key + "_ci"] = [round(lo, 4), round(hi, 4)]
    return out


# ============================================================
# 主流程
# ============================================================

def run(windows: list[dict]) -> dict:
    windows = [w for w in windows if w.get("outcome") in ("UP", "DOWN", "NOISE")]
    train, holdout = fb.time_split(windows, HOLDOUT_RATIO)
    snapshots = compute_channel_snapshots(train, "EVBENCH-v1")
    print(f"[数据] 总窗={len(windows)} train={len(train)} holdout={len(holdout)} "
          f"快照通道={sorted(snapshots.keys())}")

    # 全窗视图（Run A 初筛用，现管线忠实版）与截断视图（Run B 初筛 + 两版 EV 用）
    full_views = [build_window_view(w, snapshots) for w in holdout]
    trunc_windows = truncate3(holdout, DECISION_T_SEC)
    trunc_views = [build_window_view(w, snapshots) for w in trunc_windows]

    preds = enumerate_predicates()
    hypotheses = [{"predicate": p, "target_outcome": t}
                  for p in preds for t in ("UP", "DOWN")]
    print(f"[假设] 谓词={len(preds)} × 方向=2 → 共 {len(hypotheses)} 条进入初筛")

    report: dict = {"decision_t_sec": DECISION_T_SEC,
                    "n_windows": len(windows), "n_train": len(train),
                    "n_holdout": len(holdout), "n_hypotheses": len(hypotheses),
                    "runs": {}}

    for run_name, views in (("A_full_screen", full_views), ("B_trunc_screen", trunc_views)):
        screened = screen_hypotheses(hypotheses, views)
        rows = []
        for hyp, s in zip(hypotheses, screened):
            ev = ev_eval(hyp["predicate"], hyp["target_outcome"],
                         trunc_views, trunc_windows)
            lr = s.lift_result
            rows.append({
                "predicate": hyp["predicate"],
                "target": hyp["target_outcome"],
                "verdict": s.verdict,
                "fdr_passed": s.fdr_passed,
                "reject_reason": s.reject_reason,
                "screen_hits": len(s.hit_start_times),
                "screen_lift": round(lr.lift, 4) if lr else None,
                "screen_ci": [round(lr.ci_lower, 4), round(lr.ci_upper, 4)] if lr else None,
                "screen_p": round(lr.p_value, 6) if lr else None,
                **ev,
            })
        vc = {}
        for r in rows:
            vc[r["verdict"]] = vc.get(r["verdict"], 0) + 1
        survivors = [r for r in rows if r["verdict"] in ("ACTIVE", "OBSERVE")]
        veto_key = f"ev_fee{int(VETO_SCENARIO[0]*100)}_prem{int(VETO_SCENARIO[1]*100)}"

        def _veto_pass(r: dict) -> bool:
            lo = (r.get(veto_key + "_ci") or [None])[0]
            return lo is not None and lo == lo and lo > 0  # lo==lo 排除 NaN

        passed = [r for r in survivors if _veto_pass(r)]
        report["runs"][run_name] = {"verdict_counts": vc, "rows": rows,
                                    "survivors_veto_pass": len(passed)}
        _print_run(run_name, vc, survivors, rows, veto_key)
    return report


def _print_run(name: str, vc: dict, survivors: list[dict], rows: list[dict], veto_key: str) -> None:
    print("=" * 112)
    print(f"[{name}] 裁决分布: {vc}  （存活={len(survivors)}）")
    print("-" * 112)
    hdr = (f"{'谓词':<46}{'向':>3}{'裁决':>9}{'筛中':>5}{'lift':>7}{'p':>9}"
           f"{'实中':>5}{'胜率':>7}{'入场价':>8}{'EV费2+溢':>9}{'CI下':>8}")
    print(hdr)
    for r in sorted(survivors, key=lambda x: -(x.get(veto_key) or -9)):
        pred = json.dumps(r["predicate"], ensure_ascii=False)
        ci = r.get(veto_key + "_ci") or [None, None]
        print(f"{pred[:45]:<46}{r['target']:>3}{r['verdict']:>9}"
              f"{r['screen_hits']:>5}{str(r['screen_lift']):>7}{str(r['screen_p']):>9}"
              f"{r.get('n_fires', 0):>5}{str(r.get('win_rate')):>7}"
              f"{str(r.get('avg_entry_price')):>8}{str(r.get(veto_key)):>9}"
              f"{(f'{ci[0]:.3f}' if ci[0] is not None and ci[0] == ci[0] else '-'):>8}")
    # 反向检查：被 REJECT 但经济口径 EV 显著为正者（初筛漏杀方向）
    veto_ci = veto_key + "_ci"
    leaked = [r for r in rows if r["verdict"] == "REJECT"
              and r.get(veto_ci) and r[veto_ci][0] == r[veto_ci][0] and r[veto_ci][0] > 0]
    print(f"[反向] REJECT 中费2%+溢价 EV CI下界>0 的漏网假设: {len(leaked)} 条")
    for r in sorted(leaked, key=lambda x: -(x.get(veto_key) or 0))[:10]:
        pred = json.dumps(r["predicate"], ensure_ascii=False)
        print(f"  {pred[:60]} →{r['target']}  EV={r.get(veto_key)} CI={r.get(veto_ci)} "
              f"fires={r.get('n_fires')} 拒因={r['reject_reason']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="谓词假设的经济账台架（lift 初筛 × 入场价 EV）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default="output/predicate_ev_bench.json")
    args = ap.parse_args()

    windows = fb.load_windows_from_file(args.from_file)
    report = run(windows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[已写入] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
