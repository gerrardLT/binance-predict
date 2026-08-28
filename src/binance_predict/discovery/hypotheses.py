"""假设层：原子化、三轮头脑风暴注册表、既有产物条件解析（重放）。

轮次设计：
- R1 复刻轮：既有 183 特征 × 6 分位方向原子（与旧 720d 产物同空间，保证可比）
- R2 发散轮：发散批新特征 × 同原子化规则（新增维度）
- R3 条件化轮：JSON 显式预注册（regime × 强因子），冻结后不可改

防泄漏：连续特征分位阈值只在发现段拟合。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

# 与既有 720d 产物 run_config 对齐的全局预算
DEFAULTS: dict = {
    "discovery_frac": 0.6,
    "validation_frac": 0.2,
    "holdout_frac": 0.2,
    "quantiles": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    "min_samples": 80,
    "min_sample_frac": 0.01,
    "fdr_alpha": 0.1,
    "min_lift_pp": 2.0,
    "min_validation_lift_pp": 0.5,
    "min_interaction_gain_pp": 0.5,
    "max_l1": 60,
    "max_l2": 120,
    "max_l3": 120,
    "max_l3_tests": 50000,
    "n_min_l2": 200,
    "n_min_l3": 120,
    "n_min_holdout": 60,
    "shortlist_per_target": 40,
    "final_holdout_rule": "⚠️ Final Holdout 前，shortlist 必须完全冻结（只触碰一次）",
}

# R1 复刻特征家族（与 gen2 feature_manifest 一致；new_* 之外的家族属复刻批）
REPLICA_FAMILIES = {"geometry", "sequence", "structure", "volatility",
                    "momentum_trend", "mean_reversion", "volume", "time", "regime"}


@dataclass
class Atom:
    atom_id: str
    feature: str
    family: str
    op: str          # ">=" | "<=" | "=="
    value: float | bool
    quantile: float | None
    condition: str
    round: str       # R1 | R2 | R3

    def key(self) -> str:
        return f"{self.feature} {self.op} {self.value}"


def _atom_id(condition: str) -> str:
    return hashlib.sha256(condition.encode("utf-8")).hexdigest()[:12]


def _fmt_val(v: float | bool) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "True" if v else "False"
    return repr(float(v))


def make_atoms(fm, disc_mask: np.ndarray, rounds: tuple[str, ...] = ("R1", "R2")) -> list[Atom]:
    """按家族分轮原子化。连续特征：发现段分位阈值 × 方向算子；
    bool：== True；小域 int（唯一值 ≤12）：逐值 == v；大域 int：按连续处理。"""
    atoms: list[Atom] = []
    q_list = DEFAULTS["quantiles"]
    lo_q = [q for q in q_list if q < 0.5]
    hi_q = [q for q in q_list if q > 0.5]
    for name, fam, dtype in fm.manifest_rows():
        rnd = "R1" if fam in REPLICA_FAMILIES else "R2"
        if rnd not in rounds:
            continue
        col = fm.cols[name]
        d = col[disc_mask]
        if dtype == "bool":
            cond = f"{name} == True"
            atoms.append(Atom(_atom_id(cond), name, fam, "==", True, None, cond, rnd))
            continue
        if dtype == "int64":
            uniq = np.unique(d[np.isfinite(d.astype(float)) if d.dtype.kind == "f" else np.ones(len(d), bool)])
            if len(uniq) <= 12:
                for v in uniq:
                    cond = f"{name} == {int(v)}"
                    atoms.append(Atom(_atom_id(cond), name, fam, "==", float(int(v)), None, cond, rnd))
                continue
            d = d.astype(np.float64)
        finite = d[np.isfinite(d)] if np.issubdtype(d.dtype, np.floating) else d
        if len(finite) < 100:
            continue
        qs = np.quantile(finite, lo_q + hi_q)
        for q, thr in zip(lo_q, qs[: len(lo_q)]):
            cond = f"{name} <= {float(thr):.9g}"
            atoms.append(Atom(_atom_id(cond), name, fam, "<=", float(thr), q, cond, rnd))
        for q, thr in zip(hi_q, qs[len(lo_q):]):
            cond = f"{name} >= {float(thr):.9g}"
            atoms.append(Atom(_atom_id(cond), name, fam, ">=", float(thr), q, cond, rnd))
    return atoms


def atom_mask(fm, atom: Atom) -> np.ndarray:
    col = fm.cols[atom.feature]
    if atom.op == "==":
        if isinstance(atom.value, bool):
            return col == atom.value
        return col == atom.value
    finite_safe = np.where(np.isfinite(col.astype(np.float64)), col, np.nan)
    if atom.op == ">=":
        return finite_safe >= atom.value
    return finite_safe <= atom.value


def load_rounds(paths: list[str]) -> list[dict]:
    """读 R3 显式预注册 JSON（每条含 id/round/family/atoms/expect/mechanism）。"""
    out: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        out.extend(data.get("hypotheses", data if isinstance(data, list) else []))
    return out


_COND_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(==|>=|<=|>|<)\s*(.+?)\s*$")


def parse_condition(cond: str) -> list[tuple[str, str, float | bool]]:
    """解析既有产物字面量条件 'a >= 1.2 AND b == True' → [(feat, op, value), ...]。"""
    parts = []
    for piece in cond.split(" AND "):
        m = _COND_RE.match(piece)
        if not m:
            raise ValueError(f"无法解析条件片段: {piece!r}")
        feat, op, raw = m.group(1), m.group(2), m.group(3)
        if raw == "True":
            val: float | bool = True
        elif raw == "False":
            val = False
        else:
            val = float(raw)
        parts.append((feat, op, val))
    return parts


def condition_mask(fm, parts: list[tuple[str, str, float | bool]]) -> np.ndarray:
    """显式条件（R3 预注册 / 旧产物重放）→ 全量布尔掩码。缺特征抛 KeyError。"""
    m = None
    for feat, op, val in parts:
        if feat not in fm.cols:
            raise KeyError(f"特征不存在: {feat}")
        col = fm.cols[feat]
        if op == "==":
            sub = col == val
        else:
            fs = np.where(np.isfinite(col.astype(np.float64)), col, np.nan)
            sub = fs >= val if op in (">=", ">") else (fs <= val if op in ("<=", "<") else fs < val)
        m = sub if m is None else (m & sub)
    if m is None:
        raise ValueError("空条件")
    return m
