#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 5: compile_cross_td_report — 三 TD 形态报告合成（Week 1 交付物）。

输出 docs/research/curve-shape/report_morphology_summary_v001.md。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent


def load_td_data(td_sec: int):
    """加载单个 TD 的完整数据。"""
    tag = f"{td_sec}s"
    out_dir = ROOT / "output" / "curve_shape" / tag
    
    with open(out_dir / "cluster_profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    with open(ROOT / "docs" / "research" / "curve-shape" / "census_report_v001.json", "r", encoding="utf-8") as f:
        census = json.load(f)
    
    return {
        "tag": tag,
        "n_td": td_sec,
        "k_scan": json.load(open(out_dir / "k_scan.json", "r", encoding="utf-8")),
        "profiles": profiles,
        "census": census,
    }


def extract_pattern_families(cluster_stats_list: list[dict]) -> dict[str, list[int]]:
    """
    Heuristic pattern family grouping (not formal, just for human consumption).
    
    Group clusters by winrate similarity within a threshold. This is exploratory only.
    """
    families = []
    used = set()
    
    # Sort by winrate magnitude to find high-EV vs low-EV patterns
    sorted_clusters = sorted(cluster_stats_list, key=lambda c: abs(c["up_rate"] - 0.5), reverse=True)
    
    for c in sorted_clusters:
        if c["cluster_id"] in used:
            continue
        
        cid = c["cluster_id"]
        up_r = c["up_rate"]
        
        # Find similar winrate clusters (±0.05)
        group = [cid]
        for c2 in sorted_clusters:
            if c2["cluster_id"] != cid and c2["cluster_id"] not in used:
                if abs(c2["up_rate"] - up_r) <= 0.05:
                    group.append(c2["cluster_id"])
                    used.add(c2["cluster_id"])
        
        used.add(cid)
        
        # Classify by direction
        if up_r >= 0.6:
            direction = "high_UP"
        elif up_r <= 0.4:
            direction = "low_UP"
        else:
            direction = "neutral"
        
        families.append({
            "direction": direction,
            "cluster_ids": group,
            "avg_up_rate": round(sum(c["up_rate"] for i, c in enumerate(cluster_stats_list) if cluster_stats_list[i]["cluster_id"] in group) / len(group), 3),
        })
    
    return {"families": families}


def build_report(tds_data: list[dict]):
    """Build the comprehensive markdown report."""
    
    lines = [
        "# Curve Shape Discovery Morphology Report v001",
        "",
        f"Generated at UTC: {datetime.utcnow().isoformat()}",
        "",
        "---",
        "",
    ]
    
    # Executive Summary
    lines.extend([
        "## Executive Summary",
        "",
        "**Key findings:**",
        "",
        "1. **形态确实聚类**：三个 TD 各聚出 8 簇，silhouette 0.22~0.25，簇间胜率差异巨大",
        "   （如 225s：C6 胜率 84.3% vs C4 11.8%）。",
        "2. **但整簇胜率 ≈ 市场定价**：簇 winrate 与 TD 时刻市场隐含概率高度一致",
        "   （多数 |lift| < 3pp）——形态信息已被市场反映，直接整簇下注无显著 edge。",
        "3. **微弱系统性错配方向**：多个簇呈负 lift（形态胜率低于市场定价），押 DOWN 口径",
        "   EV +0.02~+0.05——与「群众 UP 偏好」一致，但属全样本事后均值，未过检验门。",
        "4. **Week 2 方向**：在簇内做条件子集细分（谓词约束 × 报价分层），",
        "   用 calib/confirm 切分 + BH-FDR 找 lift 显著 ≠ 0 的角落。",
        "",
        "详见 `mispricing_analysis_v001.md`（lift/EV 逐簇表）。",
        "",
    ])

    best_td_by_sil = max(tds_data, key=lambda x: x["k_scan"]["best_silhouette"])
    lines.extend([
        f"聚类质量最优 TD：{best_td_by_sil['n_td']}s（silhouette={best_td_by_sil['k_scan']['best_silhouette']:.3f}）",
        "",
        "---",
    ])
    
    # Census Recap
    census = tds_data[0]["census"]
    lines.extend([
        "## Data Overview",
        "",
        f"Time span: {census['time_span_days']} days",
        f"Total windows: {census['n_windows_total']}",
        f"Outcome distribution: UP={census['outcome_distribution']['UP']}, DOWN={census['outcome_distribution']['DOWN']}, NOISE={census['outcome_distribution']['NOISE']}",
        "",
    ])
    
    # Per-TD Sections
    for td in tds_data:
        lines.extend([
            "---",
            f"## TD = {td['n_td']} seconds",
            "",
            f"Optimal K = {td['k_scan']['best_k']} (silhouette={td['k_scan']['best_silhouette']:.4f})",
            "",
            "| Cluster | Size | UP Rate | Wilson 95% CI | Avg Down Quote@TD |",
            "|---------|------|---------|---------------|-------------------|",
        ])
        
        for c in sorted(td["profiles"]["clusters"], key=lambda x: x["n_members"], reverse=True):
            ci = c["wilson_ci_95"]
            lines.append(
                f"| C{c['cluster_id']} | {c['n_members']} | {c['up_rate']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {c.get('avg_down_quote_td') or 'N/A'} |"
            )
        
        lines.extend(["", "*Note: Avg Down Quote is the token's quoted price at decision point (lower q = deeper discount)*"])
        lines.extend(["", "---"])
    
    # Pattern Families
    lines.extend([
        "## Cross-TD Pattern Families (Exploratory)",
        "",
        "Clusters across different TD horizons may represent variant forms of the same underlying pattern. Below we group by winrate similarity (±5pp) to identify candidate families.",
        "",
    ])
    
    for i, td in enumerate(tds_data):
        fam = extract_pattern_families(td["profiles"]["clusters"])
        lines.extend([
            f"**TD={td['n_td']}s**:",
            "",
        ])
        for ff in fam["families"]:
            cid_str = ", ".join([str(cid) for cid in ff["cluster_ids"]])
            lines.append(f"- **{ff['direction']}** (avg UP rate={ff['avg_up_rate']:.3f}): Clusters {cid_str}")
        lines.append("")
    
    # Next Steps
    lines.extend([
        "## Next Steps",
        "",
        "The top candidates identified above should be manually inspected: inspect their member curve scatter plots (available at `output/curve_shape/<td>s/plots/`) to confirm homogeneity before converting to testable hypotheses in Week 2.",
        "",
        "Recommended workflow:",
        "",
        "1. Load `window_meta.json` + `curves_znorm.npy` for each candidate cluster",
        "2. Plot scatter of individual curves (not just median) to check consistency",
        "3. If homogeneous, hand-write predicate constraints in YAML (e.g., `peak_pos ≤ 0.3 ∧ fall_from_peak ≥ 15bp`)",
        "4. Run validation pipeline (BH-FDR correction + real-quote EV bootstrap) on calib/confirm splits",
        "",
        "Files generated:",
        "- `docs/research/curve-shape/census_report_v001.md` — data census summary",
        "- `docs/research/curve-shape/morphology_clusters_<td>s_v001.md` — per-TD cluster tables",
        "- `output/curve_shape/<td>s/plots/cluster_X.{png,pdf}` — visualization of top clusters",
        "- `output/curve_shape/<td>s/cluster_profiles.json` — per-cluster statistics",
        "",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compile cross-TD morphology report")
    parser.add_argument("--tds", type=int, nargs="+", default=[75, 150, 225], help="TD horizons to include")
    args = parser.parse_args()
    
    logger.info("Loading data...")
    tds_data = [load_td_data(td) for td in args.tds]
    
    logger.info("Compiling report...")
    report = build_report(tds_data)
    
    out_path = ROOT / "docs" / "research" / "curve-shape" / "report_morphology_summary_v001.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Wrote report to {out_path}")
    print(f"\nReport saved to: {out_path}")
    print("\n=== Quick Stats ===")
    for td in tds_data:
        print(f"  TD={td['n_td']}s: {td['profiles']['n_clusters']} clusters, best silhouette={td['k_scan']['best_silhouette']:.4f}")


if __name__ == "__main__":
    main()
