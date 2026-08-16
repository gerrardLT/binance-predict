#!/usr/bin/env python3
"""全链路演练（M5）：评估→假设→回测→门禁→判决 的端到端验收（不触 DB）。

用途：
- 验收 M1-M3 各组件串联正确（事件引擎/门禁/affected 映射/多重检验预算）
- 给人工放行决策提供"预演"：任意参数假设先在这里看完整裁决报告

--offline：用固定样例假设（免 LLM）；不带 --offline 时可选接 LLM 研究员
（需本地 .env 有 API key；默认 offline）。

输出：stdout 报告 + --json-out 全链路审计 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from binance_predict.backtest import fetch_klines, multiple_testing_threshold  # noqa: E402
from binance_predict.services.hypothesis_arbiter import (  # noqa: E402
    HypothesisArbiter,
    _run_engine,
    _scene_stats,
)
from binance_predict.services.scene_params import DEFAULT_SCENE_PARAMS  # noqa: E402

# 固定样例假设（offline 模式）：覆盖三种受影响场景组合
SAMPLE_HYPOTHESES = [
    {"name": "H1 更严光头（仅场景①）", "params": {**DEFAULT_SCENE_PARAMS.to_params_json(), "close_pos_min": 0.90},
     "claimed_pp": 4.0},
    {"name": "H2 更严放量（仅场景②）", "params": {**DEFAULT_SCENE_PARAMS.to_params_json(), "vol_ratio_min": 2.5},
     "claimed_pp": 3.0},
    {"name": "H3 更宽破位（双场景）", "params": {**DEFAULT_SCENE_PARAMS.to_params_json(), "eps": 0.0008},
     "claimed_pp": 2.0},
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="场景进化体系全链路演练")
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    arbiter = HypothesisArbiter()
    base = DEFAULT_SCENE_PARAMS
    print("===== 全链路演练（M5）| 基线 v1 参数 =====")

    now_ms = int(time.time() * 1000)
    t0 = time.monotonic()
    kl = fetch_klines("5m", now_ms - args.days * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    print(f"数据就绪：{len(c5)} 根 5m | {time.monotonic() - t0:.0f}s")

    events_base = _run_engine(base, c5, now_ms)
    stats_base = _scene_stats(events_base)
    print(f"基线验证集：① n={stats_base['scene1']['n']} p={stats_base['scene1']['p']} | "
          f"② n={stats_base['scene2']['n']} p={stats_base['scene2']['p']}")

    audit = {"baseline_validation": stats_base, "hypotheses": []}
    n_hypotheses = len(SAMPLE_HYPOTHESES)
    mt = multiple_testing_threshold(2.0, n_hypotheses)
    print(f"\n多重检验预算：{mt['note']}\n")

    for h in SAMPLE_HYPOTHESES:
        from binance_predict.services.scene_params import SceneParams
        hypo = SceneParams.from_params_json(h["params"])
        affected = HypothesisArbiter._affected_scenes(h["params"], base.to_params_json())
        events_h = _run_engine(hypo, c5, now_ms)
        stats_h = _scene_stats(events_h)
        verdict = arbiter._apply_gates(stats_h, stats_base, mt, h["claimed_pp"], affected)
        status = "PASS→SHADOW 候选" if verdict.passed else "REJECTED"
        print(f"[{h['name']}] affected={sorted(affected)} → {status}")
        for g in verdict.gates:
            mark = "✓" if g["passed"] else "✗"
            print(f"   {mark} {g['gate']}" + (f" | {g['detail']}" if g["detail"] else ""))
        audit["hypotheses"].append({
            "name": h["name"], "params": h["params"], "affected": sorted(affected),
            "claimed_pp": h["claimed_pp"], "passed": verdict.passed,
            "gates": verdict.gates, "hypo_validation": stats_h,
        })

    print("\n演练结论：PASS 的假设在真实链路上会进入 SHADOW 影子并行，"
          "最终生效仍需人工 promote——LLM/人都不直接改线上参数。")
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=1)
        print(f"JSON → {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
