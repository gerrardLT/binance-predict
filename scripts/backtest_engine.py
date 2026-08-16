#!/usr/bin/env python3
"""科学回测 CLI 引擎：场景①②参数假设的四层检验裁决（M1，2026-08-16）。

四层框架（方法论详见 src/binance_predict/backtest/__init__.py）：
  L1 随机漫步零假设：Lo-MacKinlay 方差比 VR(q)——确认条件可预测性存在的前提
  L2 市场定价双零假设：胜率 vs 50%（精确二项）+ vs 曲面隐含频率（z 状态定价）
  L3 统计推断：Wilson CI + 功效预检（样本不足输出 INSUFFICIENT_POWER）+ 多重检验预算
  L4 稳健性：按月 + 15m σ 三分位分组 + （可选）线上实盘衰减对照

用法：
    python scripts/backtest_engine.py                       # 默认参数（v1 基线复算）
    python scripts/backtest_engine.py --close-pos 0.88      # 假设参数检验
    python scripts/backtest_engine.py --json-out out/r.json # 结构化结果（M3 裁决消费）
    python scripts/backtest_engine.py --skip-surface        # 跳过曲面（快速、无 L2 定价）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from binance_predict.backtest import (  # noqa: E402
    build_events,
    build_surface,
    e_down_factory,
    ev,
    exact_binomial_p,
    fetch_klines,
    multiple_testing_threshold,
    power_preflight,
    variance_ratio,
    wilson,
    zbin,
)
from binance_predict.services.scene_params import SceneParams  # noqa: E402

import numpy as np  # noqa: E402

DAYS = 180


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="场景①②科学回测引擎")
    ap.add_argument("--close-pos", type=float, default=0.85)
    ap.add_argument("--vol-ratio", type=float, default=2.0)
    ap.add_argument("--vol-ma-window", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.0005)
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--hypotheses-total", type=int, default=1,
                    help="累计假设数（多重检验预算基数，M3 传 scene_param_versions 行数）")
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--skip-surface", action="store_true", help="跳过 z 曲面构建（免报价样本依赖）")
    args = ap.parse_args()

    params = SceneParams(
        close_pos_min=args.close_pos, vol_ratio_min=args.vol_ratio,
        vol_ma_window=args.vol_ma_window, eps=args.eps,
        level_lookbacks={"4h": args.lookback},
    )
    print(f"===== 科学回测 | params={params.to_params_json()} =====")

    # ---------- 数据 ----------
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - args.days * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    res = build_events(c5, params, now_ms)
    events = res["events"]
    sigma15 = res["sigma15"]
    cyc_arr = res["cyc_arr"]
    print(f"\n5m bars={len(c5)} 15m cycles={len(cyc_arr)} σ(15m)={sigma15:.4%} "
          f"破位事件={len(events)}")

    # 发现/验证切分（对齐 local_continuation_discovery：前 2/3 发现，后 1/3 盲验）
    N = len(cyc_arr)
    split = int(N * 2 / 3)
    split_cyc = int(cyc_arr[split])
    for e in events:
        e["is_val"] = e["cyc"] >= split_cyc

    # L4 分组标签：15m σ 三分位（按周期内振幅）
    agg = res["agg"]
    amp = np.array(agg["h15"]) - np.array(agg["l15"])
    t1, t2 = np.nanpercentile(amp, [33.3, 66.7])
    amp_of_cyc = {c: amp[j] for j, c in enumerate(cyc_arr)}
    for e in events:
        a = amp_of_cyc.get(e["cyc"])
        e["vol_tercile"] = 0 if a <= t1 else (1 if a <= t2 else 2)

    # ---------- L1 随机漫步零假设 ----------
    rets = [float(c / o - 1) for o, c in zip(agg["o15"], agg["c15"]) if o > 0 and c != o]
    vr_rows = [variance_ratio(rets, q) for q in (2, 3)]
    print("\n[L1] Lo-MacKinlay 方差比（随机漫步零假设：VR=1）")
    for r in vr_rows:
        if r["vr"] is not None:
            tag = "动量" if r["vr"] > 1 else "均值回归"
            print(f"  VR({r['q']}) = {r['vr']:.3f}  z*={r['z_star']}  → 拒绝随机漫步方向：{tag}")

    # ---------- 曲面（L2 定价基准，可选，只构建一次） ----------
    e_down = None
    surf = None
    if not args.skip_surface:
        try:
            surf, sigma5 = build_surface(ROOT)
            e_down = e_down_factory(surf)
            print(f"\n[L2] z 曲面就绪（{len(surf)} 格，σ5={sigma5:.4%}）")
        except Exception as exc:
            print(f"\n[L2] 曲面构建失败（降级为仅 50% 基准）：{exc}")

    # ---------- 场景统计（发现/验证 × L2/L3/L4） ----------
    def scene_rows(pool: list[dict]) -> dict:
        out = {}
        for tag, sel, win_of in (
            ("scene1", lambda e: e["scene1"] and e["has_next"], lambda e: bool(e["next_down"])),
            ("scene2", lambda e: e["scene2"] and e["has_next"], lambda e: not bool(e["next_down"])),
        ):
            es = [e for e in pool if sel(e)]
            if not es:
                out[tag] = {"n": 0}
                continue
            wins = [win_of(e) for e in es]
            n, k = len(wins), sum(wins)
            p = k / n
            lo, hi = wilson(p, n)
            row = {
                "n": n, "k": k, "p": round(p, 4), "ci": [round(lo, 4), round(hi, 4)],
                "binom_vs_50": round(exact_binomial_p(k, n), 5),
                "ev_open_050": round(ev(p, 0.50), 4),
                "power": power_preflight(n, claimed_effect_pp=abs(p - 0.5) * 100),
            }
            if surf is not None:
                # L2 市场隐含：次周期 t=10min 同状态的市场定价频率
                zs = [e["z10"] for e in es if e["z10"] is not None]
                implied = [
                    surf.get((0, zbin(z)), {}).get("freq")  # t=10min → τ≥0.5 半区
                    for z in zs
                ]
                implied = [x for x in implied if x is not None]
                if implied:
                    row["market_implied_freq"] = round(float(np.mean(implied)), 4)
                    row["edge_vs_market_pp"] = round((p - float(np.mean(implied))) * 100, 2)
            out[tag] = row
        return out

    disc = [e for e in events if not e["is_val"]]
    val = [e for e in events if e["is_val"]]
    stats = {
        "params": params.to_params_json(),
        "meta": {
            "days": args.days, "n_cycles": N, "sigma15": round(sigma15, 6),
            "n_events": len(events), "split_cyc": split_cyc,
            "surface": not args.skip_surface,
        },
        "l1_variance_ratio": vr_rows,
        "discovery": scene_rows(disc),
        "validation": scene_rows(val),
    }

    print("\n[L3] 场景统计（发现集 / 验证集盲验）")
    for phase in ("discovery", "validation"):
        for tag in ("scene1", "scene2"):
            r = stats[phase][tag]
            if r["n"] == 0:
                print(f"  [{phase}] {tag}: n=0")
                continue
            extra = ""
            if "market_implied_freq" in r:
                extra = f" | 市场隐含 {r['market_implied_freq']:.1%} → edge {r['edge_vs_market_pp']:+.1f}pp"
            print(f"  [{phase}] {tag}: n={r['n']} 胜率 {r['p']:.1%} CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}]"
                  f" | 二项p={r['binom_vs_50']} | EV@0.50 {r['ev_open_050']:+.3f}{extra}")
            print(f"           功效: {r['power']['note']}")

    # ---------- L4 稳健性（验证集，按月 + σ 三分位） ----------
    print("\n[L4] 稳健性（验证集场景命中，按月 / σ三分位）")
    l4: dict = {"by_month": {}, "by_vol_tercile": {}}
    for tag, win_of in (("scene1", lambda e: bool(e["next_down"])), ("scene2", lambda e: not bool(e["next_down"]))):
        es = [e for e in val if e.get(tag) and e["has_next"]]
        for key, grouper, store in (
            ("month", lambda e: e["month"], l4["by_month"]),
            ("tercile", lambda e: f"T{e['vol_tercile']}", l4["by_vol_tercile"]),
        ):
            g: dict = {}
            for e in es:
                g.setdefault(grouper(e), []).append(win_of(e))
            for gk, ws in sorted(g.items()):
                store.setdefault(gk, {})[tag] = {
                    "n": len(ws), "p": round(sum(ws) / len(ws), 4),
                }
    for gk, tags in sorted(l4["by_month"].items()):
        parts = [f"{t} {v['p']:.0%}(n={v['n']})" for t, v in tags.items()]
        print(f"  {gk}: " + " | ".join(parts))
    for gk, tags in sorted(l4["by_vol_tercile"].items()):
        parts = [f"{t} {v['p']:.0%}(n={v['n']})" for t, v in tags.items()]
        print(f"  {gk}: " + " | ".join(parts))
    stats["l4_robustness"] = l4

    # ---------- 多重检验预算 ----------
    mt = multiple_testing_threshold(2.0, args.hypotheses_total)
    stats["multiple_testing"] = mt
    print(f"\n[预算] {mt['note']}")

    # ---------- 基准对照 ----------
    print("\n[基准] 原始基准（local_continuation_discovery，2026-08-14 窗口）：")
    print("  场景① 63.6% [59.3,68.0] n=462 | 场景② 57.8% [53.5,62.1] n=512")
    print("  口径澄清（M1 对齐验证发现）：上述「验证集」标签实为 180 天全样本终验")
    print("  （report(m, N) 评估全数组）；本引擎已对齐——全样本加权应与基准一致。")
    print("  真正的样本外（后 60 天盲验）基准以本 CLI validation 段为准：")
    print("  v1 实测：场景① 62.0% [54.0,69.4] n=150 | 场景② 56.4% [48.4,64.1] n=149")
    print("  （场景② 样本外尚不显著，二项 p≈0.14——进化体系的 T3/M3 门禁以此为准）")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\nJSON → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
