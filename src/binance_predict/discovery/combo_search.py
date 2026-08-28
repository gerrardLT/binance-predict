"""L2/L3 组合搜索：L1 存活者的两两/三三组合（只在发现段执行）。

漏斗纪律（对齐既有 720d 产物）：
- 入场券：仅 L1 通过者（max_l1/目标）；
- 样本量下界剪枝：|A∩B| ≤ min(n_A, n_B)，min < n_min（L2=200/L3=120）直接跳过；
- 交互增益门槛：lift_AB − max(lift_A, lift_B) ≥ 0.5pp 才保留；
- 硬预算：max_l2=120 / max_l3=120 / max_l3_tests=50000；
- 命中集哈希去重（hit_sig）：同一组样本语义等价，只保留证据最强一条；
- continuation 与 reversal 目标同权执行（补上一轮的组合层空白）。

纯函数无 I/O。输出行字典键集 == ALL_TESTS_HEADER + discovery_id/_mask（内部字段）。
"""
from __future__ import annotations

import hashlib

import numpy as np

from .hypotheses import DEFAULTS, Atom, atom_mask, condition_mask
from .l1_tester import seg_stats


def hit_sig(mask: np.ndarray, upto: int) -> str:
    """命中集哈希（发现段切片打包后取前 16 位十六进制），复刻既有口径。"""
    return hashlib.sha256(np.packbits(mask[:upto]).tobytes()).hexdigest()[:16]


def _discovery_id(condition: str, target: str) -> str:
    return hashlib.sha256(f"{target}|{condition}".encode("utf-8")).hexdigest()[:14]


def _combo_row(ts, atoms_sel: list[dict], level: str, i1: int, base_d: float) -> dict | None:
    """组合行（发现段统计）。返回带 _mask 的 row；样本为 0 时返回 None。"""
    mask = None
    for a in atoms_sel:
        m = a["_mask"]
        mask = m if mask is None else (mask & m)
    st = seg_stats(ts, mask, 0, i1, base_d)
    if st["n"] == 0:
        return None
    cond = " AND ".join(a["condition"] for a in atoms_sel)
    row = {
        "target": ts.name, "horizon": ts.horizon, "level": level,
        "atom_ids": "|".join(a["atom_id"] for a in atoms_sel),
        "atom_ids_text": cond, "atom_id": "",
        "feature": atoms_sel[0]["feature"], "family": atoms_sel[0]["family"],
        "op": "", "value": "", "value_numeric": "",
        "condition": cond,
        "feature_names": "|".join(a["feature"] for a in atoms_sel),
        "families": "|".join(a["family"] for a in atoms_sel),
        "discovery_q": "", "parameter_stability": "NA",
        "status_preholdout": "", "interaction_gain_discovery_pp": "",
        "discovery_id": _discovery_id(cond, ts.name), "_mask": mask,
    }
    row.update({f"discovery_{k}": v for k, v in st.items()})
    for f in ("n", "baseline", "win_rate", "lift_pp", "ci_low", "ci_high",
              "p_value", "avg_win_return", "avg_loss_return", "payoff_ratio",
              "expectancy", "mfe_mean_atr", "mae_mean_atr"):
        row[f"validation_{f}"] = 0 if f == "n" else np.nan
    return row


def _attach_validation(ts, rows: list[dict], i1: int, i2: int, base_v: float) -> None:
    for r in rows:
        vs = seg_stats(ts, r["_mask"], i1, i2, base_v)
        r.update({f"validation_{k}": v for k, v in vs.items()})


def run_combos(fm, atoms: list[Atom], tg, l1_results: dict, n: int,
               cfg: dict | None = None) -> dict:
    """L2/L3 组合搜索 + shortlist 冻结。

    返回 {target: {"l2_kept": [...], "l3_kept": [...], "shortlist": [...],
                   "n_l2_tests", "n_l3_tests"}}，行带 _mask 供 OOS 使用。
    """
    c = {**DEFAULTS, **(cfg or {})}
    i1 = int(n * c["discovery_frac"])
    i2 = int(n * (c["discovery_frac"] + c["validation_frac"]))
    by_id = {a.atom_id: a for a in atoms}
    out: dict[str, dict] = {}
    for tname in tg.names:
        ts = tg.items[tname]
        kept_l1 = l1_results[tname]["kept"]
        base_d = l1_results[tname]["baseline_disc"]
        base_v = l1_results[tname]["baseline_val"]
        # 入场券：L1 存活原子 + 全量掩码缓存（按发现段 n 降序，供下界剪枝）
        ent = []
        for r in kept_l1:
            a = by_id[r["atom_id"]]
            m = atom_mask(fm, a)
            ent.append({"atom_id": a.atom_id, "condition": a.condition,
                        "feature": a.feature, "family": a.family,
                        "n_disc": int((m[:i1] & ts.valid[:i1]).sum()),
                        "lift": float(r["discovery_lift_pp"]), "_mask": m})
        ent.sort(key=lambda x: -x["n_disc"])

        # ---- L2：两两组合 ----
        l2, n_l2_tests = [], 0
        for i in range(len(ent)):
            a = ent[i]
            if a["n_disc"] < c["n_min_l2"]:
                break  # n 降序：后续更小，直接终止
            for j in range(i + 1, len(ent)):
                b = ent[j]
                if min(a["n_disc"], b["n_disc"]) < c["n_min_l2"]:
                    continue
                n_l2_tests += 1
                row = _combo_row(ts, [a, b], "L2", i1, base_d)
                if row is None or row["discovery_n"] < c["n_min_l2"]:
                    continue
                # 交互增益：仅同向组合才有意义（方向冲突直接剔除）
                gain = (row["discovery_lift_pp"] - max(a["lift"], b["lift"])
                        if np.sign(a["lift"]) == np.sign(b["lift"]) else -np.inf)
                if gain < c["min_interaction_gain_pp"]:
                    continue
                row["interaction_gain_discovery_pp"] = gain
                l2.append(row)
        l2.sort(key=lambda r: -abs(r["discovery_lift_pp"]))
        l2 = l2[: c["max_l2"]]

        # ---- L3：L2 存活 × L1 原子（命中集去重）----
        l3, n_l3_tests, seen = [], 0, {}
        for row2 in l2:
            ids2 = set(row2["atom_ids"].split("|"))
            for e in ent:
                if n_l3_tests >= c["max_l3_tests"]:
                    break
                if e["atom_id"] in ids2:
                    continue
                if min(row2["discovery_n"], e["n_disc"]) < c["n_min_l3"]:
                    continue
                n_l3_tests += 1
                row = _combo_row(ts, [
                    {"atom_id": row2["atom_ids"], "condition": row2["condition"],
                     "feature": row2["feature_names"], "family": row2["families"],
                     "_mask": row2["_mask"]},
                    e,
                ], "L3", i1, base_d)
                if row is None or row["discovery_n"] < c["n_min_l3"]:
                    continue
                gain = row["discovery_lift_pp"] - max(row2["discovery_lift_pp"], e["lift"])
                if np.sign(row["discovery_lift_pp"]) != np.sign(row2["discovery_lift_pp"]):
                    continue
                if gain < c["min_interaction_gain_pp"]:
                    continue
                sig = hit_sig(row["_mask"], i1)
                if sig in seen and abs(seen[sig]["discovery_lift_pp"]) >= abs(row["discovery_lift_pp"]):
                    continue
                row["interaction_gain_discovery_pp"] = gain
                seen[sig] = row
        l3 = sorted(seen.values(), key=lambda r: -abs(r["discovery_lift_pp"]))[: c["max_l3"]]

        # ---- shortlist 冻结（L2+L3 合并，命中集去重，top shortlist_per_target）----
        pool, merged = l2 + l3, {}
        for r in pool:
            sig = hit_sig(r["_mask"], i1)
            if sig not in merged or abs(r["discovery_lift_pp"]) > abs(merged[sig]["discovery_lift_pp"]):
                merged[sig] = r
        short = sorted(merged.values(), key=lambda r: -abs(r["discovery_lift_pp"]))
        short = short[: c["shortlist_per_target"]]
        _attach_validation(ts, short, i1, i2, base_v)
        out[tname] = {
            "l2_kept": l2, "l3_kept": l3, "shortlist": short,
            "n_l2_tests": n_l2_tests, "n_l3_tests": n_l3_tests,
        }
    return out


def _r3_val(raw) -> float | bool:
    if raw == "True":
        return True
    if raw == "False":
        return False
    return float(raw)


def merge_r3(fm, tg, rounds_json: list[dict], combo_results: dict, n: int,
             cfg: dict | None = None) -> dict:
    """R3 预注册假设 → 显式组合，按目标族并入各目标 shortlist（holdout 纪律不变）。

    预注册假设豁免数据驱动的入场券/交互增益门槛（先验已冻结），
    但仍须满足样本量下界并参与同一 holdout 终验。返回 {"merged": k, "skipped": [...]}。
    """
    c = {**DEFAULTS, **(cfg or {})}
    i1 = int(n * c["discovery_frac"])
    i2 = int(n * (c["discovery_frac"] + c["validation_frac"]))
    merged, skipped = 0, []
    for h in rounds_json:
        parts = [(feat, op, _r3_val(val)) for feat, op, val in h.get("atoms", [])]
        try:
            mask = condition_mask(fm, parts)
        except KeyError as ex:
            skipped.append(f"{h.get('id', '?')}: {ex}")
            continue
        cond = " AND ".join(f"{f} {o} {v}" for f, o, v in parts)
        feats = "|".join(p[0] for p in parts)
        for tname in tg.names:
            ts = tg.items[tname]
            if h.get("target_family") and ts.family != h["target_family"]:
                continue
            base_d = float(ts.win[:i1][ts.valid[:i1]].sum()) / max(1, int(ts.valid[:i1].sum()))
            st = seg_stats(ts, mask, 0, i1, base_d)
            if st["n"] < c["n_min_l2"]:
                skipped.append(f"{h.get('id', '?')}@{tname}: n={st['n']} < n_min_l2")
                continue
            vs = seg_stats(ts, mask, i1, i2,
                           float(ts.win[i1:i2][ts.valid[i1:i2]].sum()) / max(1, int(ts.valid[i1:i2].sum())))
            row = {
                "target": ts.name, "horizon": ts.horizon, "level": "R3",
                "atom_ids": h.get("id", ""), "atom_ids_text": cond,
                "atom_id": h.get("id", ""), "feature": feats,
                "family": h.get("family", ""), "op": "", "value": "",
                "value_numeric": "", "condition": cond,
                "feature_names": feats, "families": h.get("family", ""),
                "discovery_q": "", "parameter_stability": "NA",
                "status_preholdout": "", "interaction_gain_discovery_pp": "",
                "discovery_id": _discovery_id(cond, ts.name), "_mask": mask,
                "mechanism": h.get("mechanism", ""), "r3_id": h.get("id", ""),
            }
            row.update({f"discovery_{k}": v for k, v in st.items()})
            row.update({f"validation_{k}": v for k, v in vs.items()})
            bucket = combo_results.setdefault(tname, {
                "l2_kept": [], "l3_kept": [], "shortlist": [],
                "n_l2_tests": 0, "n_l3_tests": 0})
            sig = hit_sig(mask, i1)
            if any(hit_sig(r["_mask"], i1) == sig for r in bucket["shortlist"]):
                continue
            bucket["shortlist"].append(row)
            merged += 1
    return {"merged": merged, "skipped": skipped}
