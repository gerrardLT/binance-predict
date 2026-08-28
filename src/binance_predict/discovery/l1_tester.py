"""L1 单因子检验：原子 × 目标逐一评估 + 准入门槛 + BH-FDR + 参数稳定性。

固定流水线（顺序不可换，对齐既有 720d 产物）：
1. 原子 × 目标：n、胜率、对段内基准偏离、精确二项 p、Wilson CI；
2. 准入门槛：n >= max(min_samples, 1% 段样本) 且 |lift| >= min_lift_pp；
3. 准入者全族 p 值过 BH-FDR q=fdr_alpha；
4. power_preflight 标注功效（INSUFFICIENT_POWER 不下结论）；
5. 参数稳定性：阈值 ±1 相邻分位重测；
6. 按 |lift| 取 top max_l1 作为 L2 入场券。

纯函数无 I/O；输出行字典键集 == ALL_TESTS_HEADER（与既有产物同构）。
"""
from __future__ import annotations

import math

import numpy as np

from ..backtest.stats import exact_binomial_p, power_preflight, wilson
from .hypotheses import DEFAULTS, Atom, atom_mask

# all_tests.csv 表头（与既有 kline_discovery_*_720d 产物同构）
_STAT_FIELDS = ("n", "baseline", "win_rate", "lift_pp", "ci_low", "ci_high",
                "p_value", "avg_win_return", "avg_loss_return", "payoff_ratio",
                "expectancy", "mfe_mean_atr", "mae_mean_atr")
ALL_TESTS_HEADER: list[str] = (
    ["target", "horizon", "level", "atom_ids", "atom_ids_text", "atom_id",
     "feature", "family", "op", "value", "value_numeric", "condition",
     "feature_names", "families"]
    + [f"discovery_{f}" for f in _STAT_FIELDS]
    + [f"validation_{f}" for f in _STAT_FIELDS]
    + ["discovery_q", "parameter_stability", "status_preholdout",
       "interaction_gain_discovery_pp"]
)

_Q_LIST = DEFAULTS["quantiles"]
# 相邻分位映射（向更极端方向移动一格，0.9/0.1 处折返）
_ADJ_Q = {0.1: 0.2, 0.2: 0.1, 0.3: 0.2, 0.7: 0.8, 0.8: 0.9, 0.9: 0.8}


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR（向量化，复刻 local_full_history_discovery.py 口径）。"""
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    thresh = q * np.arange(1, n + 1) / n
    below = ranked <= thresh
    if not below.any():
        return [False] * n
    cutoff = ranked[np.max(np.nonzero(below)[0])]
    return list(arr <= cutoff)


def seg_stats(ts, hit: np.ndarray, a: int, b: int, baseline: float) -> dict:
    """段内命中统计（13 字段）。hit 为全量布尔掩码，取 [a, b) 切片。"""
    m = hit[a:b] & ts.valid[a:b]
    n = int(m.sum())
    if n == 0:
        return {f: (0 if f == "n" else np.nan) for f in _STAT_FIELDS}
    ret = ts.ret[a:b][m]
    k = int(ts.win[a:b][m].sum())
    wr = k / n
    lo, hi = wilson(wr, n)
    p = exact_binomial_p(k, n, baseline if np.isfinite(baseline) else 0.5)
    pos, neg = ret[ret > 0], ret[ret < 0]
    avg_win = float(pos.mean()) if len(pos) else 0.0
    avg_loss = float(-neg.mean()) if len(neg) else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else (np.inf if avg_win > 0 else np.nan)
    return {
        "n": n, "baseline": baseline, "win_rate": wr,
        "lift_pp": (wr - baseline) * 100.0, "ci_low": lo, "ci_high": hi,
        "p_value": p, "avg_win_return": avg_win, "avg_loss_return": avg_loss,
        "payoff_ratio": payoff, "expectancy": wr * avg_win - (1 - wr) * avg_loss,
        "mfe_mean_atr": float(np.nanmean(ts.mfe_atr[a:b][m])),
        "mae_mean_atr": float(np.nanmean(ts.mae_atr[a:b][m])),
    }


def _empty_row(ts, atom: Atom, level: str = "L1") -> dict:
    row = {
        "target": ts.name, "horizon": ts.horizon, "level": level,
        "atom_ids": atom.atom_id, "atom_ids_text": atom.condition,
        "atom_id": atom.atom_id, "feature": atom.feature, "family": atom.family,
        "op": atom.op, "value": str(atom.value),
        "value_numeric": "" if isinstance(atom.value, bool) else atom.value,
        "condition": atom.condition, "feature_names": atom.feature,
        "families": atom.family,
        "discovery_q": "", "parameter_stability": "NA",
        "status_preholdout": "", "interaction_gain_discovery_pp": "",
    }
    for prefix in ("discovery_", "validation_"):
        for f in _STAT_FIELDS:
            row[prefix + f] = 0 if f == "n" else np.nan
    return row


def _adjacent_hit(fm, atom: Atom, feat_thr: dict[str, np.ndarray],
                  dmask: np.ndarray) -> np.ndarray | None:
    """相邻分位阈值的命中掩码（参数稳定性复测用）；非分位原子返回 None。"""
    if atom.quantile is None or atom.op not in (">=", "<="):
        return None
    thr = feat_thr.get(atom.feature)
    if thr is None:
        return None
    q2 = _ADJ_Q[atom.quantile]
    t2 = float(thr[_Q_LIST.index(q2)])
    col = np.where(np.isfinite(fm.cols[atom.feature].astype(np.float64)),
                   fm.cols[atom.feature], np.nan)
    hit = (col >= t2) if atom.op == ">=" else (col <= t2)
    return hit & dmask


def run_l1(fm, atoms: list[Atom], tg, n: int, cfg: dict | None = None) -> dict:
    """全目标 L1 检验。返回 {target: {baseline*/n_* 统计, rows: 全部行, kept: 存活行}}。"""
    c = {**DEFAULTS, **(cfg or {})}
    i1 = int(n * c["discovery_frac"])
    i2 = int(n * (c["discovery_frac"] + c["validation_frac"]))
    disc = np.zeros(n, dtype=bool)
    disc[:i1] = True
    val = np.zeros(n, dtype=bool)
    val[i1:i2] = True
    masks = [atom_mask(fm, a) for a in atoms]
    idx_by_id = {a.atom_id: j for j, a in enumerate(atoms)}
    # 发现段分位阈值网格（参数稳定性复测用；只在发现段拟合，防泄漏）
    feat_thr: dict[str, np.ndarray] = {}
    for name in {a.feature for a in atoms if a.quantile is not None}:
        d = fm.cols[name].astype(np.float64)[disc[:]]
        d = d[np.isfinite(d)]
        if len(d) >= 100:
            feat_thr[name] = np.quantile(d, _Q_LIST)

    results: dict[str, dict] = {}
    for tname in tg.names:
        ts = tg.items[tname]
        dmask = disc & ts.valid
        vmask = val & ts.valid
        nd, nv = int(dmask.sum()), int(vmask.sum())
        base_d = float(ts.win[dmask].sum()) / nd if nd else np.nan
        base_v = float(ts.win[vmask].sum()) / nv if nv else np.nan
        n_floor = max(c["min_samples"], int(c["min_sample_frac"] * nd))
        rows, gated_idx = [], []
        for j, atom in enumerate(atoms):
            hit = masks[j] & dmask
            row = _empty_row(ts, atom)
            row.update({f"discovery_{k}": v for k, v in seg_stats(ts, hit, 0, i1, base_d).items()})
            row["_gated"] = row["discovery_n"] >= n_floor
            rows.append(row)
            if row["_gated"]:
                gated_idx.append(j)
        # BH-FDR（仅对准入者的 p 值族做校正）
        fdr_pass = bh_fdr([rows[j]["discovery_p_value"] for j in gated_idx],
                          q=c["fdr_alpha"]) if gated_idx else []
        for j, ok in zip(gated_idx, fdr_pass):
            rows[j]["_fdr"] = ok
        kept = [r for j, r in enumerate(rows)
                if r.get("_gated") and r.get("_fdr")
                and abs(r["discovery_lift_pp"]) >= c["min_lift_pp"]]
        kept.sort(key=lambda r: (-abs(r["discovery_lift_pp"]), -r["discovery_n"]))
        kept = kept[: c["max_l1"]]
        # 存活者：验证段参考统计 + 参数稳定性 + 功效标注
        for r in kept:
            j = idx_by_id[r["atom_id"]]
            atom = atoms[j]
            vs = seg_stats(ts, masks[j] & val, i1, i2, base_v)
            r.update({f"validation_{k}": v for k, v in vs.items()})
            adj = _adjacent_hit(fm, atom, feat_thr, dmask)
            if adj is not None:
                a_lift = seg_stats(ts, adj, 0, i1, base_d)["lift_pp"]
                same = np.sign(a_lift) == np.sign(r["discovery_lift_pp"])
                r["parameter_stability"] = ("STABLE" if same and abs(a_lift) >= 1.0
                                            else "EDGE_SENSITIVE")
            pf = power_preflight(int(r["validation_n"]),
                                 abs(float(r["discovery_lift_pp"])),
                                 base_v if np.isfinite(base_v) else 0.5)
            if pf["verdict"] == "INSUFFICIENT_POWER":
                r["status_preholdout"] = "INSUFFICIENT_POWER"
            elif (np.isfinite(r["validation_lift_pp"])
                  and abs(r["validation_lift_pp"]) >= c["min_validation_lift_pp"]
                  and np.sign(r["validation_lift_pp"]) == np.sign(r["discovery_lift_pp"])):
                r["status_preholdout"] = "PASS"
            else:
                r["status_preholdout"] = "FAIL"
        for r in rows:
            r.pop("_gated", None)
            r.pop("_fdr", None)
        results[tname] = {
            "baseline_disc": base_d, "baseline_val": base_v,
            "n_disc": nd, "n_val": nv, "n_tests": len(rows),
            "n_gated": len(gated_idx), "n_fdr_pass": int(sum(fdr_pass)) if fdr_pass else 0,
            "rows": rows, "kept": kept,
        }
    return results
