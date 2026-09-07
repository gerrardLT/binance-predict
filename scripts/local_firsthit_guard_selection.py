#!/usr/bin/env python3
"""D4 地基检查：护栏选择效应量化（纯离线，复用冻结扫描的事件集与 EV 口径）。

问题（见 .pytest_tmp/firsthit_optimization_evidence_v2.md §3 Class D / D4）
--------------------------------------------------------------------------
影子路径无护栏：firsthit_shadow_detector._process_window() 直接
    ev = _ev_at_entry(win, q)     # q = 触发报价
实盘路径有护栏：成交均价 >= max_exec_price 即弃单（AGENTS.md：含贴线）。

=> 实盘样本是研究样本中「成交价足够低」的偏向子集。直接比对「影子 EV」与
   「实盘 EV」会误判。必须先量化这个选择效应有多大。

口径纪律
--------
- 事件集：直接 import scripts/local_shape_scan_v2.py 的 collect()/load_k5()，
  零公式漂移（与影子/实盘共用纯函数同一纪律）。
- EV：赢 0.98/q-1 / 输 -1，逐事件真实触发价，禁 q̄ 代理。
- 护栏近似声明：离线只有触发报价 q，没有真实成交均价。本脚本用 q 作为
  成交价的**一阶近似**，因此结论是「选择效应的下界估计」——真实滑点会
  让实盘成交价高于 q，选择效应只会更大，不会更小。
- CI：日聚类 bootstrap（与冻结扫描同法，B=4000，SEED 同源）。

用法
----
    .venv\\Scripts\\python.exe scripts/local_firsthit_guard_selection.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "scripts")
from local_shape_scan_v2 import (  # noqa: E402
    collect, load_k5, day_boot, Tee, MIN_PTS,
)

FEE_RET = 0.98
LOG = "output/firsthit_guard_selection.log"

# live_channels.py:157-168 的实盘护栏（max_exec_price）
GUARDS = {
    "G0 firsthit_down_v1": 0.08,
    "G1 firsthit_down_body_v1": 0.12,
    "G3 firsthit_down_chg_v1": 0.09,
}

# 门定义（与 firsthit_shadow_detector._gate_of / local_shape_scan_v2 combos 同式）
GATES = {
    "G0 firsthit_down_v1": lambda e: True,
    "G1 firsthit_down_body_v1": lambda e: e["body_r"] <= 0.35,
    "G3 firsthit_down_chg_v1": lambda e: e["chg_bps"] <= 2.82,
}


def ev_of(e: dict) -> float:
    """逐事件真实触发价 EV：赢 0.98/q-1 / 输 -1。"""
    return (FEE_RET / e["q"] - 1.0) if e["win"] else -1.0


def summarize(evs: list[dict]) -> dict:
    if not evs:
        return {"n": 0}
    v = np.array([ev_of(e) for e in evs])
    days = np.array([e["day"] for e in evs])
    k = sum(e["win"] for e in evs)
    lo, hi = day_boot(v, days)
    return {"n": len(evs), "k": k, "p": k / len(evs), "ev": float(v.mean()),
            "lo": lo, "hi": hi, "qbar": float(np.mean([e["q"] for e in evs])),
            "days": int(len(np.unique(days)))}


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return "n=0"
    ci = "CI[nan]" if math.isnan(s["lo"]) else f"CI[{s['lo']:+.2f},{s['hi']:+.2f}]"
    return (f"n={s['n']:>4} P={s['p']:>5.1%} EV={s['ev']:+.2f} {ci} "
            f"qbar={s['qbar']:.4f} {s['days']:>2}天")


def main() -> int:
    sys.stdout = Tee(LOG)
    print("=" * 100)
    print("D4 地基检查 · 护栏选择效应（影子全集 vs 护栏内子集）")
    print("=" * 100)
    k5 = load_k5()
    events, _dropped = collect(k5)
    dense = [e for e in events if e["npts"] >= MIN_PTS]
    print(f"主分析事件（npts>={MIN_PTS}）{len(dense)} 条 | 全事件 {len(events)} 条")
    print("护栏近似声明：用触发报价 q 代理成交均价 → 结论为选择效应的**下界**\n")

    verdicts = []
    for name, guard in GUARDS.items():
        gate = GATES[name]
        passed = [e for e in dense if gate(e)]          # 门命中 = 影子会落表的全集
        filled = [e for e in passed if e["q"] < guard]   # 护栏内 = 实盘会成交
        reject = [e for e in passed if e["q"] >= guard]  # 护栏外 = 实盘弃单

        s_all, s_fill, s_rej = summarize(passed), summarize(filled), summarize(reject)
        print(f"----- {name} | guard={guard} -----")
        print(f"  影子全集（无护栏）  : {fmt(s_all)}")
        print(f"  护栏内子集（实盘成交）: {fmt(s_fill)}")
        print(f"  被弃单子集          : {fmt(s_rej)}")

        if s_all["n"] and s_fill["n"]:
            d_ev = s_fill["ev"] - s_all["ev"]
            fill_rate = s_fill["n"] / s_all["n"]
            print(f"  → 成交率 {fill_rate:.1%} | EV 选择效应 {d_ev:+.3f}"
                  f"（护栏内 − 影子全集）")
            # 盈亏平衡护栏 q* = P × 0.98（live_channels.py:17 口径）
            qstar_all = s_all["p"] * FEE_RET
            qstar_fill = s_fill["p"] * FEE_RET
            margin = (qstar_all - guard) / qstar_all * 100 if qstar_all else float("nan")
            print(f"  → 盈亏平衡 q*：影子全集 P={s_all['p']:.1%} → q*={qstar_all:.4f}"
                  f" | guard={guard} 边际 {margin:+.1f}%")
            print(f"     护栏内子集 P={s_fill['p']:.1%} → q*={qstar_fill:.4f}"
                  f"（选择后胜率口径）")
            verdicts.append((name, guard, fill_rate, d_ev, qstar_all, margin))
        print()

    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"  {'通道':<28} {'guard':>6} {'成交率':>8} {'EV选择效应':>11} "
          f"{'q*(全集)':>9} {'护栏边际':>9}")
    for name, guard, fr, d_ev, qstar, margin in verdicts:
        flag = "  ⚠️边际<5%" if margin < 5 else ""
        print(f"  {name:<28} {guard:>6.3f} {fr:>7.1%} {d_ev:>+11.3f} "
              f"{qstar:>9.4f} {margin:>+8.1f}%{flag}")
    print("\n判定规则（预注册）：")
    print("  · 成交率 < 50% → 护栏显著改变信号性质，通道说明需标注")
    print("  · EV 选择效应 > +0.10 → 影子 EV 不可直接用于实盘预期")
    print("  · 护栏边际 < 5% → 结构性亏损风险，建议下调 guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
