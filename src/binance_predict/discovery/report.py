"""报告层：机器五件套 + REPORT.md + hypotheses_registry.csv（全部显式 utf-8）。

产物（与既有 kline_discovery_*_720d 字段同构，目录加 _v2 后缀）：
- run_config.json / feature_manifest.csv / all_tests.csv /
  discovery_registry.csv / summary.md / REPORT.md / hypotheses_registry.csv
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from ..backtest.stats import ev
from .l1_tester import ALL_TESTS_HEADER

REGISTRY_HEADER: list[str] = [
    "discovery_id", "target", "horizon", "level", "condition",
    "feature_names", "families", "atom_ids",
    "discovery_n", "validation_n", "holdout_n",
    "discovery_baseline", "validation_baseline", "holdout_baseline",
    "discovery_win_rate", "validation_win_rate", "holdout_win_rate",
    "discovery_lift_pp", "validation_lift_pp", "holdout_lift_pp",
    "discovery_q", "discovery_ci_low", "discovery_ci_high",
    "holdout_ci_low", "holdout_ci_high",
    "discovery_payoff_ratio", "validation_payoff_ratio", "holdout_payoff_ratio",
    "discovery_expectancy", "validation_expectancy", "holdout_expectancy",
    "discovery_mfe_mean_atr", "validation_mfe_mean_atr", "holdout_mfe_mean_atr",
    "discovery_mae_mean_atr", "validation_mae_mean_atr", "holdout_mae_mean_atr",
    "temporal_consistency", "parameter_stability", "regime_consistency",
    "oos_degradation_pp", "oos_retention", "verdict", "score",
]

# 既有 720d 最强发现（REPORT.md 逐条对照基准）
LEGACY_BEST = {
    "5m": ("breakout_high_50 == True AND dist_prior_high_atr_10 >= -0.0148 "
           "AND efficiency_8 >= 0.692", "reversal holdout 61.5% (n=558, lift +10.3pp)"),
    "15m": ("ret_3 >= 0.0041 AND zscore_5 >= 1.37",
            "reversal 族 holdout ~63%"),
}


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return str(v)


def _write_csv(path: str, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(r.get(h, "")) for h in header])


def registry_rows(oos_results: dict) -> list[dict]:
    """OOS 结果 → registry 行（与既有 discovery_registry.csv 同构）。"""
    rows = []
    for tname, lst in oos_results.items():
        for r in lst:
            rows.append({k: r.get(k, "") for k in REGISTRY_HEADER})
    return rows


def all_test_rows(l1_results: dict, combo_results: dict) -> list[dict]:
    """L1 全量 + L2/L3 存活（内部字段 _ 前缀跳过）。"""
    rows: list[dict] = []
    for tname, res in l1_results.items():
        rows.extend({k: v for k, v in r.items() if not k.startswith("_")}
                    for r in res["rows"])
    for tname, res in combo_results.items():
        for key in ("l2_kept", "l3_kept"):
            rows.extend({k: v for k, v in r.items() if not k.startswith("_")}
                        for r in res[key])
    return rows


def _top_n(oos_results: dict, k: int = 10) -> list[dict]:
    flat = [r for lst in oos_results.values() for r in lst]
    flat.sort(key=lambda r: -r.get("score", 0.0))
    return flat[:k]


def _negatives(oos_results: dict, combo_results: dict) -> list[str]:
    lines = []
    for tname, res in combo_results.items():
        n_short = len(res["shortlist"])
        lst = oos_results.get(tname, [])
        n_pass = sum(1 for r in lst if r["verdict"] in ("ROBUST", "PROMISING"))
        if n_short == 0:
            lines.append(f"- **{tname}**：无任何组合通过漏斗（入场券/样本量/交互增益全未达标）——"
                         "该目标的确定性场景在当前特征空间内不存在，如实记录。")
        elif n_pass == 0:
            best = lst[0] if lst else None
            if best is not None:
                wr = best.get("holdout_win_rate", np.nan)
                lines.append(
                    f"- **{tname}**：{n_short} 条组合进入 shortlist，但 holdout 无一通过"
                    f"（最强 {best['condition']} 胜率 {_fmt(wr)}，"
                    f"Wilson 下界 {_fmt(best.get('holdout_ci_low', np.nan))}，"
                    f"打平线 {_fmt(best.get('breakeven', np.nan))}）——费后无正期望。")
    return lines


def _report_md(tf: str, run_config: dict, oos_results: dict,
               combo_results: dict, top: list[dict]) -> str:
    L: list[str] = []
    L.append(f"# {tf} 720d K 线科学发现报告（v2）\n")
    ds = run_config.get("data_summary", {})
    L.append(f"- 数据：{ds.get('rows', '?')} 根，{ds.get('start', '')} ~ {ds.get('end', '')}"
             f"（gap={ds.get('gap_count_gt_1_5x_median', '?')}）")
    L.append(f"- 切分：0.6/0.2/0.2 时序三段；漏斗预算：{json.dumps(run_config.get('budget', {}), ensure_ascii=False)}")
    L.append(f"- 总检验数：{run_config.get('total_tests', '?')}；holdout 只触碰一次（{run_config.get('final_holdout_rule', '')}）\n")

    L.append("## Top 10 发现（按 score = holdout_lift × √n × min(retention,1)）\n")
    if not top:
        L.append("_无任何发现通过冻结 shortlist——本轮为全负结果。_\n")
    for i, r in enumerate(top, 1):
        wr = r.get("holdout_win_rate", np.nan)
        L.append(f"### {i}. [{r['verdict']}] {r['target']} @ {r['level']}（score={_fmt(r.get('score'))}）")
        L.append(f"- 触发条件：`{r['condition']}`")
        L.append(f"- 机制：{r.get('mechanism', '（数据驱动，无预注册机制说明）')}")
        L.append(f"- 发现段：n={r['discovery_n']}，胜率 {_fmt(r.get('discovery_win_rate'))}"
                 f"（lift {_fmt(r.get('discovery_lift_pp'))}pp）")
        L.append(f"- holdout：n={r['holdout_n']}，胜率 {_fmt(wr)}"
                 f" [Wilson {_fmt(r.get('holdout_ci_low'))}~{_fmt(r.get('holdout_ci_high'))}]"
                 f"，lift {_fmt(r.get('holdout_lift_pp'))}pp，retention {_fmt(r.get('oos_retention'))}")
        L.append(f"- 盈亏比（MFE/MAE, ATR 口径）：{_fmt(r.get('holdout_payoff_ratio'))}；"
                 f"费后 EV@0.50：{_fmt(r.get('ev_at_0.50'))}；Kelly：{_fmt(r.get('kelly'))}")
        wf = r.get("walk_forward", [])
        if wf:
            seg = " → ".join(f"F{f['fold']}:{_fmt(f.get('win_rate'))}" for f in wf if f.get("n"))
            L.append(f"- walk-forward 逐折胜率：{seg}")
        L.append(f"- 一致性：月 {_fmt(r.get('temporal_consistency'))} / 波动regime {_fmt(r.get('regime_consistency'))}"
                 f"；run 块自助 CI [{_fmt(r.get('block_ci_low'))}, {_fmt(r.get('block_ci_high'))}]\n")

    L.append("## 与既有最强发现对照（R1 复刻可比性）\n")
    cond, note = LEGACY_BEST.get(tf, ("（无）", ""))
    L.append(f"- 旧产物基准：`{cond}` → {note}")
    repl = [r for r in (r for lst in oos_results.values() for r in lst) if r["condition"] == cond]
    if repl:
        r = repl[0]
        L.append(f"- 本轮重放：holdout 胜率 {_fmt(r.get('holdout_win_rate'))}（n={r['holdout_n']}）\n")
    else:
        L.append("- 本轮 shortlist 未包含该字面条件（由 `--replay-legacy` 单独对照）。\n")

    neg = _negatives(oos_results, combo_results)
    L.append("## 负结果（波普尔式保留）\n")
    L.extend(neg if neg else ["- 本轮所有目标均有存活组合，无全空目标。"])
    L.append("\n_所有裁决仅以冻结 holdout 为准；名义 lift 与费后 EV 相关性低（r≈0.16），"
             "每条结论均附经济账。_")
    return "\n".join(L)


def hypothesis_registry_rows(rounds_json: list[dict], atoms_by_round: dict[str, int],
                             oos_results: dict) -> list[dict]:
    """全部假设 × 预注册哈希 × 裁决 × 死因（供下轮头脑风暴负反馈）。"""
    rows = []
    for rnd, cnt in atoms_by_round.items():
        rows.append({"id": f"{rnd}-ATOMS", "round": rnd, "family": "atomized",
                     "atoms": "", "expect": "", "target_family": "",
                     "mechanism": "", "pre_reg_hash": "",
                     "status": f"{cnt} 个原子进入 L1 漏斗", "death_cause": ""})
    verdict_by_cond = {r["condition"]: r["verdict"]
                       for lst in oos_results.values() for r in lst}
    for h in rounds_json:
        cond = " AND ".join(f"{a[0]} {a[1]} {a[2]}" for a in h.get("atoms", []))
        rows.append({
            "id": h.get("id", ""), "round": h.get("round", ""),
            "family": h.get("family", ""), "atoms": cond,
            "expect": h.get("expect", ""), "target_family": h.get("target_family", ""),
            "mechanism": h.get("mechanism", ""),
            "pre_reg_hash": hashlib.sha256(cond.encode("utf-8")).hexdigest()[:16],
            "status": verdict_by_cond.get(cond, "NOT_TESTED"),
            "death_cause": "" if verdict_by_cond.get(cond) in ("ROBUST", "PROMISING")
            else "holdout 未确认" if cond in verdict_by_cond else "未进入冻结 shortlist",
        })
    return rows


HYPO_HEADER = ["id", "round", "family", "atoms", "expect", "target_family",
               "mechanism", "pre_reg_hash", "status", "death_cause"]


def write_outputs(outdir: str, *, run_config: dict, fm, l1_results: dict,
                  combo_results: dict, oos_results: dict,
                  rounds_json: list[dict] | None = None,
                  atoms_by_round: dict[str, int] | None = None) -> dict[str, str]:
    """落盘全部产物。返回 {产物名: 路径}。"""
    os.makedirs(outdir, exist_ok=True)
    paths = {}

    p = os.path.join(outdir, "run_config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=_fmt)
    paths["run_config"] = p

    p = os.path.join(outdir, "feature_manifest.csv")
    _write_csv(p, ["feature", "family", "dtype"],
               [{"feature": n, "family": fam, "dtype": dt}
                for n, fam, dt in fm.manifest_rows()])
    paths["feature_manifest"] = p

    p = os.path.join(outdir, "all_tests.csv")
    _write_csv(p, ALL_TESTS_HEADER, all_test_rows(l1_results, combo_results))
    paths["all_tests"] = p

    p = os.path.join(outdir, "discovery_registry.csv")
    _write_csv(p, REGISTRY_HEADER, registry_rows(oos_results))
    paths["discovery_registry"] = p

    p = os.path.join(outdir, "hypotheses_registry.csv")
    _write_csv(p, HYPO_HEADER,
               hypothesis_registry_rows(rounds_json or [], atoms_by_round or {},
                                        oos_results))
    paths["hypotheses_registry"] = p

    tf = run_config.get("tf", "?")
    p = os.path.join(outdir, "summary.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_summary_md(tf, run_config, l1_results, combo_results, oos_results))
    paths["summary"] = p

    p = os.path.join(outdir, "REPORT.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_report_md(tf, run_config, oos_results, combo_results,
                           _top_n(oos_results)))
    paths["report"] = p
    return paths


def _summary_md(tf: str, run_config: dict, l1_results: dict,
                combo_results: dict, oos_results: dict) -> str:
    L = [f"# {tf} 720d 发现漏斗摘要（v2，机器可读）\n",
         "| target | L1检验 | L1准入 | FDR通过 | L1存活 | L2检验 | L2存活 | L3检验 | L3存活 | shortlist | ROBUST | PROMISING | WEAK | REJECT |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tname in l1_results:
        l1 = l1_results[tname]
        cb = combo_results.get(tname, {})
        oo = oos_results.get(tname, [])
        vc = {v: sum(1 for r in oo if r["verdict"] == v)
              for v in ("ROBUST", "PROMISING", "WEAK", "REJECT")}
        L.append(f"| {tname} | {l1['n_tests']} | {l1['n_gated']} | {l1['n_fdr_pass']} "
                 f"| {len(l1['kept'])} | {cb.get('n_l2_tests', 0)} | {len(cb.get('l2_kept', []))} "
                 f"| {cb.get('n_l3_tests', 0)} | {len(cb.get('l3_kept', []))} "
                 f"| {len(cb.get('shortlist', []))} | {vc['ROBUST']} | {vc['PROMISING']} "
                 f"| {vc['WEAK']} | {vc['REJECT']} |")
    L.append("")
    return "\n".join(L)
