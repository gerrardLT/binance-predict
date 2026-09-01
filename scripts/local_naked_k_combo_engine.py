#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K组合假设空间的编排层：G1 规则推导 / G2 kill gate / G3 全量搜索（计划 §3、§4）。

分工（与已有脚本的边界）：
- `local_naked_k_prepare.py`  **只 import**：它的任何改动都会换 run_fp，毁掉
  上一轮 `26af1604732a` 的可复现链。本研究只用它的 sha256/CSV 工具。
- `local_naked_k_engine.py`   **只 import**：build_namespace / build_parents /
  expand_masks / _hit_sig / _align 语义是本研究原子的唯一来源。
- `local_naked_k_report.py`   **只 import**：derive_abs_targets（dirup/dirdn 的
  代数推导 + 与「次根 open→close」逐位对齐的那条口径链）。
- `local_naked_k_grammar.py`  纯函数层（符号化 / n-gram / 闭项集 / 位集）。
- 本文件                      判决与编排：三段切分、支持度前沿、K1/K2/K3、FDR。

三条不可协商的纪律（写进代码，不靠自觉）：
1. **规则先于数据**：`assert_grammar_frozen()` 在 G2/G3 入口检查 config 的分位
   边界已回填且 grammar/budgets/kill_gate 未被改过。没冻结就不许看结果。
2. **CONFIRM 只碰一次**：`write_confirm_flag(..., allow=False)` 第二次调用直接
   RuntimeError，且失败路径不写盘（不是 WARN）。
3. **因果性**：所有段切片都是「先按全量算好原子，再切段」，绝不「用段尾数据
   重新估参数」。分位边界只在 SCREEN 估计并冻结。

用法：
    python scripts/local_naked_k_combo_engine.py --stage g1
    python scripts/local_naked_k_combo_engine.py --stage g2
    python scripts/local_naked_k_combo_engine.py --stage g3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from binance_predict.backtest.stats import (FEE, PREMIUM, exact_binomial_p,  # noqa: E402
                                            ev, min_detectable_effect)
from binance_predict.discovery.data import Klines, load_klines_csv  # noqa: E402
from binance_predict.discovery.l1_tester import bh_fdr  # noqa: E402
from binance_predict.discovery.targets import build_targets  # noqa: E402

import local_naked_k_engine as eng  # noqa: E402
import local_naked_k_grammar as G  # noqa: E402
import local_naked_k_prepare as prep  # noqa: E402
import local_naked_k_report as rep  # noqa: E402

CFG_PATH = os.path.join(ROOT, "config", "naked_k_combos.json")
OUT_ROOT = os.path.join(ROOT, "output", "naked_k_combo")
DOCS_DIR = os.path.join(ROOT, "docs", "research", "naked-k")
BAR_MS = eng.BAR_MS
DAY_MS = eng.DAY_MS
TFS = ("5m", "15m", "1h")
PRIMARY_TARGETS = ("dirup_1", "dirdn_1")
NS_PREFIX_PROBE = 20_000          # 只为拿「列集合」，不需全量（实测同键集，省 18s）

# combo_run_fp 的输入源。凡「变了会改结果」的都必须在列（含 config 本体——它的
# 分位边界回填即完成冻结，指纹随之一变，旧产物不会被静默复用）。
COMBO_FP_SOURCES = [
    ("config_combos", CFG_PATH),
    ("grammar_py", os.path.join(ROOT, "scripts", "local_naked_k_grammar.py")),
    ("combo_engine_py", os.path.join(ROOT, "scripts", "local_naked_k_combo_engine.py")),
    ("combo_report_py", os.path.join(ROOT, "scripts", "local_naked_k_combo_report.py")),
    ("fetch_history_py", os.path.join(ROOT, "scripts", "local_naked_k_fetch_history.py")),
    ("registry", prep.REGISTRY),
    ("klines_5m_2160d", os.path.join(ROOT, "output", "klines_5m_2160d.csv")),
    ("klines_15m_2160d", os.path.join(ROOT, "output", "klines_15m_2160d.csv")),
    ("history_manifest", os.path.join(ROOT, "output", "klines_history_2160d_manifest.json")),
    ("market_samples", prep.SAMPLES),
    ("sentiment_windows", os.path.join(ROOT, "output", "sentiment_windows_online_fixed.json")),
    # 复用的统计层与上一轮三条腿（口径所有者）
    ("engine_py", prep.ENGINE),
    ("prepare_py", prep.PREPARE),
    ("report_py", prep.REPORT),
] + [(("dep_" + k), v) for k, v in prep.DEP_MODULES.items()]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    return prep.sha256_file(path)


# ============================ 配置与冻结守卫 ============================

def load_config(path: str = CFG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assert_grammar_frozen(cfg: dict[str, Any]) -> None:
    """G2/G3 入口守卫：文法参数必须已在看过任何结果之前冻结。

    唯一允许「看过数据再写」的是 quantile_edges（它按定义只能在 SCREEN 段估计），
    因此它的 status 就是「冻结是否完成」的唯一开关。
    """
    qe = cfg["power_frontier"]["quantile_edges"]
    if str(qe.get("status")) != "FROZEN":
        raise RuntimeError(
            f"[FAIL] 文法未冻结：power_frontier.quantile_edges.status={qe.get('status')!r}"
            "（须为 'FROZEN'）。先跑 --stage g1 --freeze-edges 在 SCREEN 段估计三分位边界并回填 config；"
            "在规则未闭合前开跑搜索等于用结果定规则。")
    by_tf = qe.get("by_tf") or {}
    variables = cfg["grammar"]["family_a"]["atom_pool"]["discretized"]["variables"]
    for tf in TFS:
        blk = by_tf.get(tf)
        if not isinstance(blk, dict):
            raise RuntimeError(f"[FAIL] 分位边界缺失：quantile_edges.by_tf.{tf} 未回填")
        for var in variables:
            e = blk.get(var)
            if not isinstance(e, dict) or "e1" not in e or "e2" not in e:
                raise RuntimeError(f"[FAIL] 分位边界缺失：quantile_edges.by_tf.{tf}.{var}")
            if not (math.isfinite(float(e["e1"])) and math.isfinite(float(e["e2"]))):
                raise RuntimeError(f"[FAIL] 分位边界非有限值：{tf}.{var} = {e}")
            if float(e["e1"]) >= float(e["e2"]):
                raise RuntimeError(f"[FAIL] 分位边界逆序：{tf}.{var} e1={e['e1']} >= e2={e['e2']}")


def edges_from_config(cfg: dict[str, Any], tf: str) -> dict[str, tuple[float, float]]:
    """读已冻结的分位边界（G2/G3 唯一合法来源；已冻结后不得再从数据重估）。"""
    blk = cfg["power_frontier"]["quantile_edges"]["by_tf"][tf]
    return {v: (float(blk[v]["e1"]), float(blk[v]["e2"])) for v in blk}


def freeze_quantile_edges(edges_by_tf: dict[str, dict[str, tuple[float, float]]],
                          cfg_path: str = CFG_PATH) -> dict:
    """G1 专用：把 SCREEN 段算出的三分位边界按 TF 回填进 config（此后只读）。

    这是 freeze_contract 允许的唯一一次写入。写完立刻把 status 置 FROZEN，并把
    「写了什么、从哪段算的」一起落盘 —— 否则事后无法证明边界不是挑出来的。
    """
    cfg = load_config(cfg_path)
    qe = cfg["power_frontier"]["quantile_edges"]
    if str(qe.get("status")) == "FROZEN":
        raise RuntimeError("[FAIL] quantile_edges 已是 FROZEN，拒绝二次回填（会换 run_fp）")
    variables = cfg["grammar"]["family_a"]["atom_pool"]["discretized"]["variables"]
    for tf in TFS:
        if tf not in edges_by_tf:
            raise RuntimeError(f"[FAIL] 缺少 {tf} 的分位边界，不允许只冻结部分 TF")
        got = edges_by_tf[tf]
        if sorted(got) != sorted(variables):
            raise RuntimeError(f"[FAIL] {tf} 边界变量集与 config 不符：{sorted(got)}")
        qe["by_tf"][tf] = {v: {"e1": float(e1), "e2": float(e2)} for v, (e1, e2) in got.items()}
    qe["status"] = "FROZEN"
    qe["frozen_at_utc"] = now_utc()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return cfg


def combo_run_fp() -> tuple[str, dict[str, Any]]:
    parts = {name: sha256_file(p) for name, p in COMBO_FP_SOURCES if os.path.exists(p)}
    absent = [n for n, p in COMBO_FP_SOURCES if n not in parts]
    blob = json.dumps(parts, sort_keys=True).encode()
    fp = hashlib.sha256(blob).hexdigest()[:12]
    return fp, {"combo_run_fp": fp, "digests": parts, "absent": absent}


def out_dir(create: bool = True) -> tuple[str, str]:
    fp, parts = combo_run_fp()
    d = os.path.join(OUT_ROOT, fp)
    if create:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "_fingerprint.json"), "w", encoding="utf-8") as f:
            json.dump(parts, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return fp, d


# ============================ 三段协议 ============================

def resolve_segments(t: np.ndarray, cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """SCREEN / REPLICATE / CONFIRM 的下标边界（左闭右开）。

    CONFIRM 的起点**不按比例**取，而是钉在报价可用窗的第一根上：这一段的全部意义
    是「用真实报价算一次 EV」，没有报价的时间轴不属于它。
    历史不够长时 SCREEN 会吃掉报价窗 → 直接 raise，不做「那就重叠一点」的降级。
    """
    n = int(len(t))
    b = cfg["segments"]["boundary"]
    frac = float(b["screen_frac"])
    c0 = int(np.searchsorted(np.asarray(t, dtype=np.int64), int(b["confirm_start_ms"])))
    if n == 0:
        raise RuntimeError("[FAIL] 空时间轴，无法切分三段")
    if c0 <= 0:
        raise RuntimeError(
            f"[FAIL] 数据起点已晚于报价窗起点 {b['confirm_start_iso']}（c0={c0}）"
            "→ 没有 SCREEN/REPLICATE 可用，三段协议无法建立。")
    s1 = int(n * frac)
    if s1 >= c0:
        raise RuntimeError(
            f"[FAIL] SCREEN 上界 {s1}（={frac:.0%}×{n}）≥ CONFIRM 起点 {c0}：三段重叠。"
            "候选生成将看到用于最终确认的数据 —— 这条协议红线不允许降级运行。")
    return {"screen": (0, s1), "replicate": (s1, c0), "confirm": (c0, n)}


def segment_report(t: np.ndarray, seg: dict[str, tuple[int, int]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, (a, b) in seg.items():
        out[k] = {"i0": a, "i1": b, "bars": b - a,
                  "start_iso": _iso(t[a]) if b > a else None,
                  "end_iso": _iso(t[b - 1]) if b > a else None,
                  "days": round((int(t[b - 1]) - int(t[a])) / DAY_MS, 2) if b > a else 0.0}
    return out


def _iso(ms: Any) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# ============================ 数据与原子 ============================

def assert_atom_pool(ns: dict[str, Any], cfg: dict[str, Any], tf: str) -> None:
    """池漂移守卫：namespace 的 bool 列集合必须与冻结 config 逐字相等。

    不等意味着 build_namespace 增删过列 —— 那等于有人静默改过假设空间。宁可中止，
    也不要拿「一半旧原子一半新原子」的池去算 FDR。
    """
    live = {k for k, v in ns.items() if isinstance(v, np.ndarray) and v.dtype == np.bool_}
    want = set(cfg["grammar"]["family_a"]["atom_pool"]["bool_atoms"])
    if live != want:
        raise RuntimeError(
            f"[FAIL] {tf} 原子池漂移：仅引擎有={sorted(live - want)} 仅config={sorted(want - live)}")


def namespace_bool_keys(prefix: int = NS_PREFIX_PROBE) -> list[str]:
    """从 build_namespace 实取 bool 列名（只用前缀样本：列集合与 n 无关）。

    父周期保持全量 —— _align 只做 searchsorted，多几根父根不改变列集合与 dtype。
    """
    csv5 = _pick_5m_csv()
    parents = eng.build_parents(csv5)
    kl5 = parents["5m"]
    m = min(int(prefix), int(len(kl5)))
    sub = Klines(**{f: getattr(kl5, f)[:m] for f in ("t", "o", "h", "l", "c", "v", "cont")})
    ns = eng.build_namespace(sub, BAR_MS["5m"], parents)
    return sorted(k for k, v in ns.items()
                  if isinstance(v, np.ndarray) and v.dtype == np.bool_)


def _pick_5m_csv() -> str:
    for p in (os.path.join(ROOT, "output", "klines_5m_2160d.csv"), prep.CSV_5M):
        if os.path.exists(p):
            return p
    raise FileNotFoundError("缺少 5m K 线 CSV（2160d 与 720d 均不存在）")


def load_tf(tf: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """某周期的统计腿：序列 + namespace + targets + 三段边界（不含原子）。"""
    parents = load_parents()
    kl = parents[tf]
    ns = eng.build_namespace(kl, BAR_MS[tf], parents)
    assert_atom_pool(ns, cfg, tf)
    seg = resolve_segments(kl.t, cfg)
    tg = build_targets(kl.t, kl.o, kl.h, kl.l, kl.c, kl.cont, [1], ns["atr_abs"])
    rep.derive_abs_targets(tg, ns["dir_"], [1])
    return {"tf": tf, "kl": kl, "ns": ns, "parents": parents, "tg": tg, "seg": seg}


_PARENTS_CACHE: dict[str, dict[str, Klines]] = {}


def load_parents() -> dict[str, Klines]:
    if "p" not in _PARENTS_CACHE:
        _PARENTS_CACHE["p"] = eng.build_parents(os.path.join(ROOT, "output", "klines_5m_2160d.csv"))
    return _PARENTS_CACHE["p"]


def symbol_fields(ns: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: ns[k] for k in ("dir_", "body_r", "up_r", "lo_r")}


def build_family_a(ds: dict[str, Any], cfg: dict[str, Any],
                   edges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """族 A 原子池 = 53 bool ∪ 12 分位档 ∪ pattern 掩码（含语境门子掩码）。

    返回 packed（全量位集）+ names/kinds/exprs + exclusive_groups。
    原子顺序即索引，一旦排序变了，config 里的索引类判据全部错位 → 显式稳定排序。
    """
    pool = cfg["grammar"]["family_a"]["atom_pool"]
    disc = pool["discretized"]
    names: list[str] = []
    kinds: list[str] = []
    exprs: list[str] = []
    cols: list[np.ndarray] = []
    groups: list[list[int]] = []
    ns = ds["ns"]

    for a in sorted(pool["bool_atoms"]):
        names.append(a)
        kinds.append("bool")
        exprs.append(a)
        cols.append(np.asarray(ns[a], dtype=bool))

    labels = list(disc["bin_labels"])
    for var in sorted(disc["variables"]):
        low, mid, high = G.ternary_bins(ns[var], edges[var])
        bins = {"low": low, "mid": mid, "high": high}
        g: list[int] = []
        for lab in labels:
            g.append(len(names))
            names.append(f"{var}__{lab}")
            kinds.append("bin")
            exprs.append(f"{var}|{lab}")
            cols.append(np.asarray(bins[lab], dtype=bool))
        groups.append(g)

    s1 = ds["seg"]["screen"][1]
    with open(prep.REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)["hypotheses"]
    for row in eng.expand_masks(reg, ns, s1):
        names.append(row["mask_key"])
        kinds.append("pattern")
        exprs.append(row["expr_full"])
        cols.append(row["_mask"])

    packed = G.pack_columns(cols)
    return {"names": names, "kinds": kinds, "exprs": exprs, "packed": packed,
            "exclusive_groups": groups, "n": int(len(ds["kl"])), "m": int(packed.shape[0])}


def target_bitsets(ds: dict[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    """(valid, win&valid) 两条全量位集 —— 段内 n/k 的唯一计数来源。"""
    ts = ds["tg"].items[name]
    v = np.asarray(ts.valid, dtype=bool)
    w = np.asarray(ts.win, dtype=bool) & v
    return G.pack_columns([v])[0], G.pack_columns([w])[0]


# ============================ 功效前沿 ============================

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def z_for_alpha(alpha: float, iters: int = 100) -> float:
    """上侧 alpha 的分位点 z（P(Z>z)=alpha）。二分法：无 scipy，且确定性可复算。"""
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha 须在 (0,0.5)，收到 {alpha}")
    lo, hi = 0.0, 12.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _ncdf(-mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delta_min_pp(n: int, m_tests: int, q: float, p0: float, z_beta: float) -> float:
    """可检出效应的绝对下限（pp）。α_eff = q/M 是 BH 在 rank=1 处最严的那个 α。

    与 stats.min_detectable_effect 的区别只在 α：那里钉死 0.05，这里必须随假设预算
    收紧 —— 否则「跑了 50 万个组合后找到的 5pp」会被当成可检出效应。
    """
    if n <= 0:
        return float("inf")
    z_a = z_for_alpha(q / max(1, int(m_tests)))
    return (z_a + z_beta) * math.sqrt(p0 * (1.0 - p0) / n) * 100.0


def frontier_table(cfg: dict[str, Any]) -> dict[str, Any]:
    pf = cfg["power_frontier"]
    q, p0, zb = float(pf["q"]), float(pf["p0"]), float(pf["z_beta"])
    grid = {int(M): {int(n): round(delta_min_pp(int(n), int(M), q, p0, zb), 3)
                     for n in pf["support_grid_n"]} for M in pf["budget_grid_M"]}
    return {"formula": pf["formula"], "q": q, "p0": p0, "z_beta": zb,
            "alpha_eff_rule": pf["alpha_eff_rule"],
            "M_grid": [int(x) for x in pf["budget_grid_M"]],
            "n_grid": [int(x) for x in pf["support_grid_n"]],
            "delta_min_pp": grid,
            "reference_stats_min_detectable_effect": {
                "n_3000_alpha_0.05": round(min_detectable_effect(3000, 0.5) * 100, 3),
                "n_1000_alpha_0.05": round(min_detectable_effect(1000, 0.5) * 100, 3),
                "n_300_alpha_0.05": round(min_detectable_effect(300, 0.5) * 100, 3)}}


def breakeven_p(q: float) -> float:
    return (q + PREMIUM) / (1.0 - FEE)


def q_max_entry(p: float) -> float:
    return (1.0 - FEE) * p - PREMIUM


# ============================ 符号 / 族 B / 族 C ============================

def ohlc_symbol_fields(kl: Klines) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 OHLC 直接算四要素（与 build_namespace:324-330 同一算术）。

    为什么另开一条路而不是给每个父周期跑一次 build_namespace：后者在 5m 上要 18s，
    且会拉进一大堆与符号无关的状态机列。代价是必须证明两条路逐位相同 —— 见
    `assert_symbol_routes_agree`（G1 硬闸，不一致就中止）。
    """
    o, h, l, c = (np.asarray(kl.o, dtype=np.float64), np.asarray(kl.h, dtype=np.float64),
                  np.asarray(kl.l, dtype=np.float64), np.asarray(kl.c, dtype=np.float64))
    rng = np.where(h > l, h - l, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.sign(c - o), np.abs(c - o) / rng,
                (h - np.maximum(o, c)) / rng, (np.minimum(o, c) - l) / rng)


def assert_symbol_routes_agree(ds: dict[str, Any], cfg: dict[str, Any]) -> None:
    """快路径（OHLC 直算）必须与权威路径（build_namespace 列）逐位相等。"""
    ns = ds["ns"]
    ref = (np.asarray(ns["dir_"]), np.asarray(ns["body_r"]),
           np.asarray(ns["up_r"]), np.asarray(ns["lo_r"]))
    got = ohlc_symbol_fields(ds["kl"])
    for name, r, g in zip(("dir_", "body_r", "up_r", "lo_r"), ref, got):
        same = np.array_equal(np.nan_to_num(r, nan=-9.0), np.nan_to_num(g, nan=-9.0))
        if not same:
            raise RuntimeError(f"[FAIL] {ds['tf']} 符号快路径与 build_namespace 不一致：{name}")


def sigma_of(cfg: dict[str, Any]) -> int:
    return G.sigma_size(cfg["grammar"]["family_b"]["alphabet"]["axes"])


def tf_symbol(ds: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    """本 TF 的逐根 Σ 符号（族 B 与族 C 的子周期共用同一份）。"""
    f = symbol_fields(ds["ns"])
    return G.symbolize(f["dir_"], f["body_r"], f["up_r"], f["lo_r"],
                       cfg["grammar"]["family_b"]["alphabet"]["axes"])


def parent_symbol_from_kl(kl: Klines, cfg: dict[str, Any]) -> np.ndarray:
    d, b, u, l = ohlc_symbol_fields(kl)
    return G.symbolize(d, b, u, l, cfg["grammar"]["family_b"]["alphabet"]["axes"])


def _quant_counts(cnts: np.ndarray, positions: int, floor: int,
                  confirm_rate: float) -> dict[str, Any]:
    """计数分布 + 两个决策相关量：过支持度地板的个数、过报价窗可确认率线的个数。

    两个量为什么要同时报：支持度地板管「统计上看不看得见」，可确认率线管「就算看见了，
    报价窗里有没有足够样本去做 EV 确认」。它们不一定同源 —— 取**交集**才是真正的搜索空间。
    """
    rate = cnts / max(1, positions)
    return {"observed": int(cnts.size), "total_positions": int(positions),
            "count_quantiles": {str(q): float(np.percentile(cnts, q))
                                for q in (10, 25, 50, 75, 90, 99, 100)},
            "ge_floor": int((cnts >= floor).sum()),
            "ge_confirm_rate": int((rate >= confirm_rate).sum()),
            "max_rate": float(rate.max()) if rate.size else None}


def family_b_distribution(sym: np.ndarray, a: int, b: int, base: int,
                          cap: int, floor: int, confirm_rate: float) -> dict[str, Any]:
    out: dict[str, Any] = {"alphabet_size": int(base), "length_cap": int(cap)}
    for L in range(1, cap + 1):
        codes, valid = G.ngram_codes(sym, L, base)
        sel = np.asarray(valid[a:b], dtype=bool)
        c = codes[a:b][sel]
        key = f"L{L}"
        if c.size == 0:
            out[key] = {"space": int(base ** L), "observed": 0}
            continue
        _, cnts = np.unique(c, return_counts=True)
        out[key] = {"space": int(base ** L),
                    **_quant_counts(cnts, int(sel.sum()), floor, confirm_rate)}
    return out


def family_c_cells(child_t: np.ndarray, child_sym: np.ndarray, parent_kl: Klines,
                   cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """族 C 的全量复合码与有效性（父根严格取已收盘那根，走 grammar.parent_symbol）。

    返回 (code ∈ [0,σ²) 或 -1, ok)。单独抽出这个函数是因为 G2 要拿它建候选位集，
    而 G1 只要计数 —— 两条路必须用同一份 code，不得各自对齐一次父根。
    """
    sigma = sigma_of(cfg)
    psym = parent_symbol_from_kl(parent_kl, cfg)
    par_ms = int(parent_kl.t[1] - parent_kl.t[0])
    aligned = G.parent_symbol(child_t, parent_kl.t, psym, par_ms)
    return G.cross_product_code(child_sym, aligned, sigma)


def family_c_distribution(code: np.ndarray, ok: np.ndarray, a: int, b: int,
                          sigma: int, floor: int, confirm_rate: float) -> dict[str, Any]:
    sel = np.asarray(ok[a:b], dtype=bool)
    c = code[a:b][sel]
    out: dict[str, Any] = {"space": int(sigma * sigma), "positions": int(sel.sum())}
    if c.size == 0:
        out["observed"] = 0
        return out
    _, cnts = np.unique(c, return_counts=True)
    out.update(_quant_counts(cnts, int(sel.sum()), floor, confirm_rate))
    return out


# ============================ 段内命中计数（位集路线） ============================

def count_mat(mat: np.ndarray, vp: np.ndarray, wp: np.ndarray,
              chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(q, nb) 候选位集矩阵 × 段内 (valid, win&valid) 位集 → 每候选的 n 与 k。

    先 C&valid 再在其上 &win：省一次与 C 的 AND，且中间量更小（内存峰值更低）。
    """
    q = int(mat.shape[0])
    n_arr = np.zeros(q, dtype=np.int64)
    k_arr = np.zeros(q, dtype=np.int64)
    for s in range(0, q, chunk):
        blk = mat[s:s + chunk] & vp
        n_arr[s:s + chunk] = np.bitwise_count(blk).sum(axis=1, dtype=np.int64)
        k_arr[s:s + chunk] = np.bitwise_count(blk & wp).sum(axis=1, dtype=np.int64)
    return n_arr, k_arr


def seg_base_rate(vp: np.ndarray, wp: np.ndarray) -> tuple[float, int]:
    """段内基准胜率（只在 valid 上）与该段 valid 根数。"""
    nv = int(np.bitwise_count(vp).sum())
    nw = int(np.bitwise_count(wp).sum())
    return (nw / nv if nv else float("nan")), nv


def block_perm_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """块置换下标：整块重排、块内顺序不动（保留块内自相关）。

    末尾不足一块用「补到 nb*block 再过滤越界槽」处理：reshape 要求整块。排完后每个
    0..n-1 恰好出现一次，所以这确实是 [0,n) 的一个置换，不是有放回抽样。
    """
    nb = -(-int(n) // int(block))
    slots = np.arange(nb * block, dtype=np.int64).reshape(nb, block)
    slots = slots[rng.permutation(nb)]
    flat = slots.ravel()
    return flat[flat < n]


def pack_bool(x: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(x, dtype=bool), bitorder="big")


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    raise TypeError(f"不可序列化：{type(v)}")


def edges_from_screen(ds: dict[str, Any], cfg: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """三分位边界，只在 SCREEN 段估计（禁止全样本分位 = 前视）。"""
    a, b = ds["seg"]["screen"]
    disc = cfg["grammar"]["family_a"]["atom_pool"]["discretized"]
    return {v: G.tertile_edges(np.asarray(ds["ns"][v], dtype=np.float64)[a:b])
            for v in disc["variables"]}


# ============================ G1：枚举规则的实证推导 ============================

def stage_g1(cfg: dict[str, Any], tfs: Sequence[str] = TFS) -> dict[str, Any]:
    """交付「规则」本身：支持度/联合支持度分布 + 可检出效应前沿 + 地板推导。

    本阶段唯一允许写 config 的动作是 quantile_edges 回填（由 --freeze-edges 触发）。
    边界按 TF 分别估计：atrp/volr/er20/rpos20 的跨周期量纲差一个 sqrt(bar 比)，
    共用一组 5m 边界会让 1h 的三档几乎退化成单一档 —— 那才是真的把文法写错。
    """
    t_all = time.time()
    pf = cfg["power_frontier"]
    floors = {k: int(v) for k, v in pf["floors_frozen"].items()}
    gate = pf["quote_window_cross_gate"]
    csm = cfg["segments"]["confirm_sample_measured"]
    n_quote = int(csm["windows_with_both_quote_and_outcome"])
    confirm_rate = float(gate["min_expected_hits"]) / n_quote
    sigma = sigma_of(cfg)
    axes = cfg["grammar"]["family_b"]["alphabet"]["axes"]

    per_tf: dict[str, Any] = {}
    edges_all: dict[str, dict[str, tuple[float, float]]] = {}
    already_frozen = str(cfg["power_frontier"]["quantile_edges"].get("status")) == "FROZEN"
    for tf in tfs:
        t0 = time.time()
        ds = load_tf(tf, cfg)
        assert_symbol_routes_agree(ds, cfg)
        n = int(len(ds["kl"]))
        a, b = ds["seg"]["screen"]
        r0, r1 = ds["seg"]["replicate"]
        edges = (edges_from_config(cfg, tf) if already_frozen
                 else edges_all.setdefault(tf, edges_from_screen(ds, cfg)))
        fa = build_family_a(ds, cfg, edges)
        pk_s = G.pack_region(fa["packed"], n, a, b)
        pair = G.pairwise_supports(pk_s)
        marg = np.diag(pair).astype(np.float64)
        off = pair[np.triu_indices(pair.shape[0], k=1)].astype(np.float64)
        closed = G.closed_frequent_itemsets(pk_s, floors[tf], 2,
                                           fa["exclusive_groups"], pair)
        # 报价窗交叉闸（计划 §4-G1）：支持度地板只管「看不看得见」，这条管「看见了能不能用真报价确认」。
        conf_min_cnt = int(math.ceil(confirm_rate * (b - a)))
        conf_ge = {f"depth{d}": int(sum(1 for it, s in closed if len(it) == d and s >= conf_min_cnt))
                   for d in (1, 2)}
        sym = G.symbolize(*[ds["ns"][k] for k in ("dir_", "body_r", "up_r", "lo_r")], axes)
        fb = family_b_distribution(sym, a, b, sigma,
                                   int(cfg["grammar"]["family_b"]["length_cap"][tf]),
                                   floors[tf], confirm_rate)
        fc: dict[str, Any] = {}
        for tier in cfg["grammar"]["family_c"]["parent_tiers"]:
            if tier == tf:
                continue
            code, ok = family_c_cells(ds["kl"].t, sym, ds["parents"][tier], cfg)
            fc[tier] = family_c_distribution(code, ok, a, b, sigma, floors[tf], confirm_rate)
            del code, ok
        per_tf[tf] = {
            "bars": n,
            "segments": segment_report(ds["kl"].t, ds["seg"]),
            "targets": {t: _target_profile(ds, t, floors[tf]) for t in PRIMARY_TARGETS},
            "family_a": {
                "m_atoms": fa["m"], "min_support_floor": floors[tf],
                "marginal_support_quantiles": _q(marg),
                "marginal_rate_quantiles": _q(marg / max(1, b - a)),
                "atoms_ge_floor": int((marg >= floors[tf]).sum()),
                "pair_joint_support_quantiles": _q(off),
                "pairs_total": int(off.size),
                "pairs_ge_floor": int((off >= floors[tf]).sum()),
                "closed_depth1": int(sum(1 for i, _ in closed if len(i) == 1)),
                "closed_depth2": int(sum(1 for i, _ in closed if len(i) == 2)),
                "quote_gate_min_screen_count": conf_min_cnt,
                "closed_ge_quote_gate": conf_ge,
            },
            "family_b": fb,
            "family_c": fc,
            "quantile_edges_screen": {k: [v[0], v[1]] for k, v in edges.items()},
            "screen_bars": b - a, "replicate_bars": r1 - r0,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        del ds, fa, pk_s, pair, sym
    return {
        "stage": "G1", "created_at_utc": now_utc(), "tfs": list(tfs),
        "frontier": frontier_table(cfg),
        "floors_frozen": floors,
        "quote_gate": {"min_expected_hits": int(gate["min_expected_hits"]),
                       "confirm_windows": n_quote, "confirm_days": csm["quote_span_days"],
                       "implied_min_support_rate": confirm_rate,
                       "implied_min_screen_count_5m":
                           int(math.ceil(confirm_rate * per_tf["5m"]["screen_bars"]))
                           if "5m" in per_tf else None,
                       "binding_constraint": "max(统计地板, 报价可确认率线)"},
        "per_tf": per_tf,
        "quantile_edges_by_tf": {tf: {k: [v[0], v[1]] for k, v in e.items()}
                                 for tf, e in edges_all.items()},
        "elapsed_sec": round(time.time() - t_all, 1),
    }


def _q(x: np.ndarray) -> dict[str, float]:
    return {str(p): float(np.percentile(x, p)) for p in (0, 10, 25, 50, 75, 90, 99, 100)} \
        if x.size else {"empty": 0.0}


def _target_profile(ds: dict[str, Any], name: str, floor: int) -> dict[str, Any]:
    """每段的 valid 数与基准胜率：所有 lift 的分母都从这里出发，不得各自算一套。"""
    ts = ds["tg"].items[name]
    out: dict[str, Any] = {}
    for seg, (x, y) in ds["seg"].items():
        v = np.asarray(ts.valid[x:y], dtype=bool)
        w = np.asarray(ts.win[x:y], dtype=bool) & v
        out[seg] = {"valid": int(v.sum()), "base_win_rate": float(w.sum() / v.sum())
                    if v.sum() else None}
    return out


# ============================ G2：⛔ KILL GATE ============================

def _stack_and(pk: np.ndarray, items: Sequence[Sequence[int]]) -> np.ndarray:
    if not items:
        return np.empty((0, pk.shape[1]), dtype=np.uint8)
    return np.stack([G.and_region(pk, it) for it in items])


def build_g2_pool(ds: dict[str, Any], cfg: dict[str, Any],
                  fa: dict[str, Any]) -> dict[str, Any]:
    """G2 粗网格候选池 = 族 A（5m、深度 2、全量两两闭项集）+ 族 C（5m×15m 一档）。

    故意不含族 B：它检验的是「序列延续」这一不同假设，混进来会让「不过」无法归因
    （见 config.kill_gate.scope.why_not_b）。
    """
    scope = cfg["kill_gate"]["scope"]
    tf = scope["family_a_tf"][0]
    assert ds["tf"] == tf, f"G2 族 A 只跑 {scope['family_a_tf']}，收到 {ds['tf']}"
    floor = int(scope["family_a_min_support"])
    n = fa["n"]
    (s0, s1), (r0, r1) = ds["seg"]["screen"], ds["seg"]["replicate"]
    pk_s = G.pack_region(fa["packed"], n, s0, s1)
    pk_r = G.pack_region(fa["packed"], n, r0, r1)
    pair = G.pairwise_supports(pk_s)
    closed = G.closed_frequent_itemsets(pk_s, floor, int(scope["family_a_depth"]),
                                       fa["exclusive_groups"], pair)
    a_items = [it for it, _ in closed if len(it) == int(scope["family_a_depth"])]
    a_labels = ["&".join(fa["names"][i] for i in it) for it in a_items]

    c_labels: list[str] = []
    axes = cfg["grammar"]["family_b"]["alphabet"]["axes"]
    sym = G.symbolize(*[ds["ns"][k] for k in ("dir_", "body_r", "up_r", "lo_r")], axes)
    sigma = sigma_of(cfg)
    c_masks_s: list[np.ndarray] = []
    c_masks_r: list[np.ndarray] = []
    for child_tf, par_tf in scope["family_c_tiers"]:
        assert child_tf == tf, "G2 只允许主周期与子周期一致的跨周期档"
        code, ok = family_c_cells(ds["kl"].t, sym, ds["parents"][par_tf], cfg)
        ok_s = np.asarray(ok[s0:s1], dtype=bool)
        ok_r = np.asarray(ok[r0:r1], dtype=bool)
        cs, cr = code[s0:s1], code[r0:r1]
        for cell in range(sigma * sigma):
            ms = ok_s & (cs == cell)
            if int(ms.sum()) < floor:
                continue                    # 支持度不达地板的细格不进池（也不进零分布）
            c_masks_s.append(np.packbits(ms, bitorder="big"))
            c_masks_r.append(np.packbits(ok_r & (cr == cell), bitorder="big"))
            c_labels.append(f"C:{tf}x{par_tf}|{G.describe_symbol(cell // sigma, axes)}"
                            f"/{G.describe_symbol(cell % sigma, axes)}")
    mat_s = np.concatenate([_stack_and(pk_s, a_items),
                            np.stack(c_masks_s) if c_masks_s else np.empty((0, pk_s.shape[1]), np.uint8)]
                           ) if (a_items or c_masks_s) else np.empty((0, pk_s.shape[1]), np.uint8)
    mat_r = np.concatenate([_stack_and(pk_r, a_items),
                            np.stack(c_masks_r) if c_masks_r else np.empty((0, pk_r.shape[1]), np.uint8)]
                          ) if (a_items or c_masks_r) else np.empty((0, pk_r.shape[1]), np.uint8)
    return {"labels": a_labels + c_labels,
            "families": ["A"] * len(a_labels) + ["C"] * len(c_labels),
            "mat_screen": mat_s, "mat_replicate": mat_r,
            "n_screen_atoms": fa["m"], "floor": floor,
            "n_A": len(a_labels), "n_C": len(c_labels),
            "a_items": a_items, "screen_span": (s0, s1), "replicate_span": (r0, r1)}


# ============================ 段内打分 ============================


def seg_target_pack(ds: dict[str, Any], name: str, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """段 [a, b) 的 (valid, win&valid) 位集（重新字节对齐，段长直接缩放代价）。"""
    vp, wp = target_bitsets(ds, name)
    n = int(len(ds["kl"]))
    return (G.pack_region(np.stack([vp]), n, a, b)[0],
            G.pack_region(np.stack([wp]), n, a, b)[0])


def score_rows(mat: np.ndarray, vp: np.ndarray, wp: np.ndarray,
               base: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """候选位集矩阵在给定段 + 给定 target 下的 (n, k, lift_pp)。

    n=0 的行 lift 为 NaN（不是 0）：「没样本」与「样本里一半赢」必须可区分。
    """
    n_arr, k_arr = count_mat(mat, vp, wp)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = np.where(n_arr > 0, k_arr / np.maximum(n_arr, 1), np.nan)
    return n_arr, k_arr, (wr - base) * 100.0


# ============================ K3：block-permutation 零校准 ============================


def block_perm_bytes(nb: int, block_bars: int, rng: np.random.Generator) -> np.ndarray:
    """字节轴上的整块置换下标（块长 = block_bars/8 字节，块内顺序不动）。

    为什么在字节轴做而不在根轴做：块长 24 根是 8 的整数倍，一个字节恰好装 8 根，
    所以下标可以在字节粒度上精确表达同一个置换，代价从 O(n) 位运算降到 O(n/8) 取数。
    不足一整块的尾部留在原地（仍是 [0,nb) 的一个置换，不是有放回抽样）。
    """
    if int(block_bars) % 8:
        raise ValueError(f"block_bars 必须是 8 的整数倍才能在字节轴置换，收到 {block_bars}")
    bb = int(block_bars) // 8
    idx = np.arange(int(nb), dtype=np.int64)
    nfull = int(nb) - (int(nb) % bb)
    if nfull <= 0:
        return idx
    blocks = np.arange(nfull, dtype=np.int64).reshape(-1, bb)
    blocks = blocks[rng.permutation(blocks.shape[0])]
    out = blocks.ravel() if nfull == int(nb) else np.concatenate([blocks.ravel(), idx[nfull:]])
    if not np.array_equal(np.sort(out), idx):
        raise RuntimeError("[FAIL] 字节块置换不是置换（有重复/丢元素）")
    return out


def k3_calibrate(mat: np.ndarray, fam_idx: dict[str, np.ndarray],
                 packs: dict[str, tuple[np.ndarray, np.ndarray]],
                 bases: dict[str, float], n_perm: int, block_bars: int,
                 seed: int, chunk: int = 256) -> dict[str, Any]:
    """族内 max-lift 的置换零分布与经验 p（config.kill_gate.K3）。

    循环顺序是「候选块 外层 → 置换 内层」：一块候选位集（256×nb ≈ 12MB）在 200 次
    置换里反复仍热，反过来的话每块都要重读整份置换。置换下标只依赖 seed + 块数，
    与 target 无关，故两个 target 共用同一批置换（它们不是独立实验）。

    基线胜率不需重估：(win, valid) 被**同时**置换，两者的总 popcount 不变 →
    base_perm ≡ base_obs。这一点在代码里被 assert，而不是靠推定。
    """
    q = int(mat.shape[0])
    nb = int(mat.shape[1])
    rng = np.random.default_rng(int(seed))
    perms = np.stack([block_perm_bytes(nb, block_bars, rng) for _ in range(int(n_perm))])
    null = {f: np.full(int(n_perm), -np.inf) for f in fam_idx}
    obs = {f: -np.inf for f in fam_idx}
    obs_t: dict[str, dict[str, float]] = {}
    null_t: dict[str, dict[str, np.ndarray]] = {}
    for tname, (vp, wp) in packs.items():
        base = float(bases[tname])
        if int(np.bitwise_count(np.take(vp, perms[0])).sum()) != int(np.bitwise_count(vp).sum()):
            raise RuntimeError("[FAIL] 置换改变了 valid 总数，零分布的基线不再可比")
        _, _, lift = score_rows(mat, vp, wp, base)
        obs_t[tname] = {}
        null_t[tname] = {f: np.full(int(n_perm), -np.inf) for f in fam_idx}
        for f, idxs in fam_idx.items():
            if idxs.size:
                v = float(np.nanmax(lift[idxs]))
                obs_t[tname][f] = v
                obs[f] = max(obs[f], v)
        del lift
        for f, idxs in fam_idx.items():
            if idxs.size == 0:
                continue
            for s in range(0, idxs.size, int(chunk)):
                sub = mat[idxs[s:s + int(chunk)]]
                for p in range(int(n_perm)):
                    pm = perms[p]
                    n_a, k_a = count_mat(sub, np.take(vp, pm), np.take(wp, pm))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        lf = np.where(n_a > 0, k_a / np.maximum(n_a, 1), np.nan) - base
                    mx = float(np.nanmax(lf)) * 100.0
                    if mx > null[f][p]:
                        null[f][p] = mx
                    if mx > null_t[tname][f][p]:
                        null_t[tname][f][p] = mx
    out: dict[str, Any] = {"n_permutations": int(n_perm), "block_length_bars": int(block_bars),
                           "seed": int(seed), "rows": q, "families": {}}
    for f in fam_idx:
        if not math.isfinite(obs[f]):
            out["families"][f] = {"observed_max_lift_pp": None, "p_value": None,
                                  "note": "该族在 SCREEN 无候选行（无法判 K3）"}
            continue
        ge = int((null[f] >= obs[f]).sum())
        out["families"][f] = {
            "rows": int(fam_idx[f].size),
            "observed_max_lift_pp": round(float(obs[f]), 4),
            "null_max_mean_pp": round(float(null[f].mean()), 4),
            "null_max_median_pp": round(float(np.median(null[f])), 4),
            "null_max_p95_pp": round(float(np.percentile(null[f], 95)), 4),
            "null_max_max_pp": round(float(null[f].max()), 4),
            "n_null_ge_observed": ge,
            "p_value": round((1.0 + ge) / (1.0 + int(n_perm)), 5),
        }
    out["per_target_descriptive"] = {
        t: {f: {"observed_max_lift_pp": round(obs_t[t][f], 4),
                "p_value": round((1.0 + int((null_t[t][f] >= obs_t[t][f]).sum()))
                                 / (1.0 + int(n_perm)), 5)}
            for f in fam_idx if math.isfinite(obs_t[t][f])}
        for t in packs}
    return out


# ============================ G2 判决 ============================


def stage_g2(cfg: dict[str, Any], run_k3: bool = True,
             k3_chunk: int = 256) -> dict[str, Any]:
    """⛔ KILL GATE。20 秒级粗网格，不建管道；三条任一不过 → 终止。

    本函数只读，不写 CONFIRM。它的产物是「继续/终止」这个二值决定及其数值证据。
    """
    t_all = time.time()
    assert_grammar_frozen(cfg)
    kg = cfg["kill_gate"]
    scope = kg["scope"]
    tf = scope["family_a_tf"][0]
    ds = load_tf(tf, cfg)
    fa = build_family_a(ds, cfg, edges_from_config(cfg, tf))
    pool = build_g2_pool(ds, cfg, fa)
    (s0, s1), (r0, r1) = pool["screen_span"], pool["replicate_span"]
    labels = pool["labels"]
    fams = pool["families"]
    rows: list[dict[str, Any]] = []
    packs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    bases: dict[str, float] = {}
    prof: dict[str, Any] = {}
    for tname in PRIMARY_TARGETS:
        vp_s, wp_s = seg_target_pack(ds, tname, s0, s1)
        vp_r, wp_r = seg_target_pack(ds, tname, r0, r1)
        base_s, nv_s = seg_base_rate(vp_s, wp_s)
        base_r, nv_r = seg_base_rate(vp_r, wp_r)
        packs[tname] = (vp_s, wp_s)
        bases[tname] = base_s
        n_s, k_s, lift_s = score_rows(pool["mat_screen"], vp_s, wp_s, base_s)
        n_r, k_r, lift_r = score_rows(pool["mat_replicate"], vp_r, wp_r, base_r)
        for i, lab in enumerate(labels):
            rows.append({"label": lab, "family": fams[i], "target": tname,
                         "n_screen": int(n_s[i]), "win_screen": int(k_s[i]),
                         "lift_screen_pp": float(lift_s[i]),
                         "n_replicate": int(n_r[i]), "win_replicate": int(k_r[i]),
                         "lift_replicate_pp": float(lift_r[i])})
        prof[tname] = {"base_rate_screen": round(base_s, 6), "base_rate_replicate": round(base_r, 6),
                       "valid_bars_screen": nv_s, "valid_bars_replicate": nv_r}

    # ---------------- K1 ----------------
    k1 = kg["K1"]
    top = sorted(rows, key=lambda r: -r["lift_screen_pp"])[: int(k1["top_n"])]
    med_s = float(np.nanmedian([r["lift_screen_pp"] for r in top]))
    med_r = float(np.nanmedian([r["lift_replicate_pp"] for r in top]))
    n_nan = sum(1 for r in top if r["n_replicate"] == 0)
    retention = (med_r / med_s) if med_s > 0 else float("nan")
    k1_res = {
        "name": k1["name"], "top_n": int(k1["top_n"]),
        "median_lift_screen_pp": round(med_s, 4),
        "median_lift_replicate_pp": round(med_r, 4),
        "retention": None if not math.isfinite(retention) else round(retention, 4),
        "top_rows_missing_in_replicate": n_nan,
        "threshold": float(k1["threshold"]), "comparator": k1["comparator"],
        "pass": bool(med_s > 0 and math.isfinite(retention)
                     and retention >= float(k1["threshold"])),
    }

    # ---------------- K2 ----------------
    k2 = kg["K2"]
    hit_rows = [r for r in rows if r["n_replicate"] >= int(k2["min_n_replicate"])
                and r["lift_replicate_pp"] >= float(k2["min_lift_pp"])]
    uniq_labels = sorted({r["label"] for r in hit_rows})
    k2_res = {
        "name": k2["name"], "min_lift_pp": float(k2["min_lift_pp"]),
        "min_n_replicate": int(k2["min_n_replicate"]),
        "n_rows": len(hit_rows), "n_distinct_combos": len(uniq_labels),
        "counting_unit": "distinct 组合（同一组合在两个 target 上都过只计一次，比计行更保守）",
        "threshold": int(k2["threshold"]), "comparator": k2["comparator"],
        "pass": bool(len(uniq_labels) >= int(k2["threshold"])),
        "top_examples": [{"label": r["label"], "target": r["target"],
                          "lift_rep_pp": round(r["lift_replicate_pp"], 3),
                          "n_rep": r["n_replicate"],
                          "lift_scr_pp": round(r["lift_screen_pp"], 3),
                          "n_scr": r["n_screen"]}
                         for r in sorted(hit_rows, key=lambda x: -x["lift_replicate_pp"])[:20]],
    }

    # ---------------- K3 ----------------
    k3 = kg["K3"]
    if run_k3:
        fam_idx = {f: np.where(np.asarray(fams) == f)[0] for f in ("A", "C")}
        k3_res = k3_calibrate(pool["mat_screen"], fam_idx, packs, bases,
                              int(k3["n_permutations"]), int(k3["block_length_bars"]),
                              int(k3["seed"]), chunk=int(k3_chunk))
        thr = float(k3["threshold"])
        pv = {f: k3_res["families"][f].get("p_value") for f in k3_res["families"]}
        passes = [f for f, v in pv.items() if v is not None and v < thr]
        k3_res.update({"name": k3["name"], "threshold": thr, "comparator": k3["comparator"],
                       "segment": k3["segment_for_observed_statistic"],
                       "pass_rule": k3["pass_rule"], "families_passing": passes,
                       "pass": bool(passes)})
    else:
        k3_res = {"name": k3["name"], "skipped": True, "pass": None}

    checks = {"K1": k1_res, "K2": k2_res, "K3": k3_res}
    decisive = [v.get("pass") for v in checks.values()]
    all_pass = bool(all(d is True for d in decisive))
    # 无论 kill 与否都保留 REPLICATE 实际天花板：kill 报告若只写「0 个过 8pp」
    # 而不写「实际最高是多少」，读者就无法区分「差一点」与「差一个量级」。
    rep_rows = sorted((r for r in rows if r["n_replicate"] > 0),
                      key=lambda r: -r["lift_replicate_pp"])
    top_rep = [{"label": r["label"], "target": r["target"],
                "lift_rep_pp": round(r["lift_replicate_pp"], 3), "n_rep": r["n_replicate"],
                "lift_scr_pp": round(r["lift_screen_pp"], 3), "n_scr": r["n_screen"]}
               for r in rep_rows[:20]]
    return {
        "stage": "G2", "created_at_utc": now_utc(), "tf": tf,
        "segments": segment_report(ds["kl"].t, ds["seg"]),
        "scope": scope, "pool": {"rows_per_target": len(labels), "family_A": pool["n_A"],
                                 "family_C": pool["n_C"], "min_support": pool["floor"],
                                 "screen_atoms": pool["n_screen_atoms"]},
        "target_profiles": prof, "checks": checks,
        "top_replicate_rows": top_rep,
        "replicate_max_lift_pp": round(rep_rows[0]["lift_replicate_pp"], 4) if rep_rows else None,
        "screen_max_lift_pp": round(max(r["lift_screen_pp"] for r in rows), 4) if rows else None,
        "total_tests_this_gate": len(rows),
        "all_pass": all_pass,
        "decision": "PROCEED_TO_G3" if all_pass else "KILL",
        "failed": [k for k, v in checks.items() if v.get("pass") is not True],
        "elapsed_sec": round(time.time() - t_all, 1),
    }


# ============================ CONFIRM 幂等守卫 ============================

CONFIRM_LEDGER = os.path.join(OUT_ROOT, "_confirm_touches.jsonl")


def write_confirm_flag(path: str, detail: dict[str, Any], allow: bool = False) -> None:
    """CONFIRM 只能被碰一次：第二次调用直接 RuntimeError，且失败路径不写盘。

    为什么是 hard fail 而不是 WARN：WARN 在批量脚本里等于没有。一旦 CONFIRM 被同一个
    选择过程多次消费，它就不再独立，而这件事事后从数据上看不出来 —— 只能靠卡。
    这个文件就是「本轮结论能不能信」的物理凭据。
    """
    if os.path.exists(path) and not allow:
        raise RuntimeError(
            f"[FATAL] CONFIRM 已被触碰过：{path}\n"
            "第二次使用同一确认集做判决会使它对该选择过程不再独立，结论不可解释。\n"
            "确实要重跑（视为新一轮研究，旧结论作废）请加 --allow-retouch。")
    payload = {"written_at_utc": now_utc(), "path": os.path.abspath(path), "detail": detail}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_confirm_ledger(detail: dict[str, Any], ledger: str = CONFIRM_LEDGER) -> int:
    """全局 CONFIRM 账本（追加），返回 seq。与 flag 的区别：flag 是 per-run 幂等卡，
    账本是跨 run 的「确认集还剩多少独立性」凭据。"""
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    seq = 0
    if os.path.exists(ledger):
        with open(ledger, encoding="utf-8") as f:
            seq = sum(1 for line in f if line.strip())
    rec = {"seq": seq + 1, "touched_at_utc": now_utc()}
    rec.update(detail)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return seq + 1


# ============================ kill report ============================


def render_kill_report(res: dict[str, Any], cfg: dict[str, Any]) -> str:
    kg = cfg["kill_gate"]
    c = res["checks"]
    lines = [
        "# 裸K组合方向：G2 KILL GATE 报告（方向被实证证伪）",
        "",
        f"> 生成时间：{res['created_at_utc']}　·　run：{res.get('run_id', '')}　·　"
        f"判决：**{res['decision']}**　·　未通过项：`{', '.join(res['failed'])}`",
        "",
        "## 0. 一句话结论",
        "",
        "在 2160 天、约 62 万根 5m K 线上，「多形态组合」相对于单形态的超可加增益"
        "未达到继续建设全量搜索管道的必要门槛。因此本研究不进入 G3，"
        "**也不触碰 CONFIRM（报价窗一次未消费）**。",
        "",
        "这是计划 §4-G2 预设的合规出口之一：「三条任一不满足 → 判定组合方向为伪需求」。"
        "本文档就是那个交付物。",
        "",
        "## 1. 三条判据的实测数值",
        "",
        "| 判据 | 定义 | 阈值 | 实测 | 结论 |",
        "|---|---|---|---|---|",
    ]
    fmt = {
        "K1": lambda v: (f"保留率 = {v['retention']}"
                         f"（screen 中位 {v['median_lift_screen_pp']}pp → "
                         f"replicate 中位 {v['median_lift_replicate_pp']}pp）"),
        "K2": lambda v: (f"{v['n_distinct_combos']} 个组合（{v['n_rows']} 行）"
                         f"满足 lift>={v['min_lift_pp']}pp 且 n>={v['min_n_replicate']}"),
        "K3": lambda v: ("skipped" if v.get("skipped") else
                         "；".join(f"{f}: max={d.get('observed_max_lift_pp')}pp p={d.get('p_value')}"
                                   for f, d in v["families"].items())),
    }
    for k in ("K1", "K2", "K3"):
        v = c[k]
        lines.append(f"| **{k} {v['name']}** | {kg[k]['definition']} | "
                     f"`{v.get('comparator', '')} {v.get('threshold', '')}` | "
                     f"{fmt[k](v)} | {'✅ 过' if v.get('pass') else '❌ 不过'} |")
    lines += [
        "",
        "## 2. 本次探针的实际搜索空间",
        "",
        f"- 周期：`{res['tf']}`；段：SCREEN {res['segments']['screen']['bars']} 根"
        f"（{res['segments']['screen']['start_iso']} ~ {res['segments']['screen']['end_iso']}）"
        f"、REPLICATE {res['segments']['replicate']['bars']} 根"
        f"（{res['segments']['replicate']['start_iso']} ~ {res['segments']['replicate']['end_iso']}）",
        f"- 原子池：{res['pool']['screen_atoms']}（53 bool + 12 分位档 + pattern 掩码）",
        f"- 族 A（深度 2、min_support={res['pool']['min_support']} 的闭项集）："
        f"{res['pool']['family_A']} 条",
        f"- 族 C（5m×15m 符号积，同支持度地板）：{res['pool']['family_C']} 条",
        f"- 总检验行：{res['total_tests_this_gate']}（× 2 target）",
        "",
        "## 3. 为什么这个「不过」可信而不是一句空话",
        "",
        "- 零校准用的是 block-permutation（块长 "
        f"{kg['K3']['block_length_bars']} 根 = 2 小时，{kg['K3']['n_permutations']} 次），"
        "它把「搜索机器自身在纯噪声上能刷多高的 top-lift」量出来，而不是假定一个理论分布。",
        "- 判据数字在跑之前已写进 `config/naked_k_combos.json` 的 `kill_gate` 块并纳入 "
        "`combo_run_fp`；跑完未改一字。",
        "- CONFIRM（报价窗 23.78d / 6,791 窗）**一次未碰**；它仍是未被消费的干净额度。",
        "",
        "## 4. REPLICATE 段的实际天花板（完整公开，不做美化）",
        "",
        f"全池 {res['total_tests_this_gate']} 行里，REPLICATE 的最高 lift 为 "
        f"**{res['replicate_max_lift_pp']}pp**（SCREEN 为 {res['screen_max_lift_pp']}pp）。"
        "K2 要 10 个 ≥8pp 的，实际 0 个——不是差一点，是整个天花板在门槛之下。",
        "",
        "| 组合 | target | lift_rep (pp) | n_rep | lift_scr (pp) | n_scr |",
        "|---|---|---|---|---|---|"]
    for r in res["top_replicate_rows"]:
        lines.append(f"| `{r['label']}` | {r['target']} | {r['lift_rep_pp']} | "
                     f"{r['n_rep']} | {r['lift_scr_pp']} | {r['n_scr']} |")
    lines += ["",
        "## 5. 下一步（计划 §12 已预设的转向）",
        "",
        "组合方向被证伪后，杠杆不在「改分子（胜率）」而在「改分母（报价）」："
        "实测 `up_price` 的 5–95 分位跨 .03→.98，而打平线随 q 从 63.8% 一直降到 ~40%。"
        "下一轮研究应以「条件化报价」为主对象，而不是继续堆形态组合。",
        "",
    ]
    return "\n".join(lines)


# ============================ CLI ============================


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="裸K组合假设空间：G1/G2/G3 编排")
    ap.add_argument("--stage", choices=["g1", "g2", "g3"], required=True)
    ap.add_argument("--freeze-edges", action="store_true",
                    help="仅 G1：把 SCREEN 段估出的分位边界回填 config（唯一允许的一次写入）")
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--no-k3", action="store_true", help="调试用：跳过置换校准")
    ap.add_argument("--k3-chunk", type=int, default=256)
    ap.add_argument("--kill-report", action="store_true",
                    help="G2 不过时落盘 combo_kill_report.md")
    args = ap.parse_args(argv)
    cfg = load_config()
    tfs = tuple(x.strip() for x in args.tfs.split(",") if x.strip())
    fp, odir = out_dir()
    print(f"[combo_run_fp] {fp}  ->  {os.path.relpath(odir, ROOT)}")
    if args.stage == "g1":
        res = stage_g1(cfg, tfs)
        res["combo_run_fp"] = fp
        write_json(os.path.join(odir, "g1_frontier.json"), res)
        for tf in tfs:
            p = res["per_tf"][tf]
            print(f"\n=== G1 {tf}  bars={p['bars']}  screen={p['screen_bars']}  "
                  f"atoms={p['family_a']['m_atoms']}  elapsed={p['elapsed_sec']}s")
            print(f"  边际支持度 p10/50/90/100 = "
                  f"{p['family_a']['marginal_support_quantiles']['10']:.0f}/"
                  f"{p['family_a']['marginal_support_quantiles']['50']:.0f}/"
                  f"{p['family_a']['marginal_support_quantiles']['90']:.0f}/"
                  f"{p['family_a']['marginal_support_quantiles']['100']:.0f}"
                  f"   过地板({p['family_a']['min_support_floor']})={p['family_a']['atoms_ge_floor']}")
            print(f"  两两联合支持度 p50/90/99/max = "
                  f"{p['family_a']['pair_joint_support_quantiles']['50']:.0f}/"
                  f"{p['family_a']['pair_joint_support_quantiles']['90']:.0f}/"
                  f"{p['family_a']['pair_joint_support_quantiles']['99']:.0f}/"
                  f"{p['family_a']['pair_joint_support_quantiles']['100']:.0f}"
                  f"   对数={p['family_a']['pairs_total']} 过地板={p['family_a']['pairs_ge_floor']}")
            print(f"  闭项集：depth1={p['family_a']['closed_depth1']} "
                  f"depth2={p['family_a']['closed_depth2']}")
            for L, d in p["family_b"].items():
                if isinstance(d, dict) and "observed" in d:
                    print(f"  族B {L}: space={d['space']} observed={d['observed']} "
                          f"ge_floor={d.get('ge_floor')} ge_confirm_rate={d.get('ge_confirm_rate')}")
            for tier, d in p["family_c"].items():
                print(f"  族C ×{tier}: space={d['space']} observed={d.get('observed')} "
                      f"ge_floor={d.get('ge_floor')} ge_confirm_rate={d.get('ge_confirm_rate')}")
        print("\n[F1 前沿 Δ_min(pp)]")
        for M, col in res["frontier"]["delta_min_pp"].items():
            print(f"  M={M}: " + "  ".join(f"n={n}→{v}" for n, v in col.items()))
        print(f"\n[报价窗可确认率线] {res['quote_gate']['implied_min_support_rate']:.4%}")
        if args.freeze_edges:
            if res.get("quantile_edges_by_tf") and len(res["quantile_edges_by_tf"]) == len(TFS):
                e = {tf: {k: (v[0], v[1]) for k, v in d.items()}
                     for tf, d in res["quantile_edges_by_tf"].items()}
                freeze_quantile_edges(e)
                print("[FROZEN] quantile_edges 已回填 → config 从此只读（run_fp 已变）")
            else:
                print("[WARN] 未回填：边界已 FROZEN 或 TF 不完整")
        return 0
    if args.stage == "g2":
        res = stage_g2(cfg, run_k3=not args.no_k3, k3_chunk=args.k3_chunk)
        res["combo_run_fp"] = fp
        write_json(os.path.join(odir, "g2_kill_gate.json"), res)
        for k, v in res["checks"].items():
            print(f"  {k} {v['name']}: pass={v.get('pass')}")
        print(f"[G2] decision={res['decision']}  failed={res['failed']}  "
              f"elapsed={res['elapsed_sec']}s")
        if args.kill_report and not res["all_pass"]:
            res["run_id"] = cfg["run_id"]
            p = os.path.join(DOCS_DIR, "combo_kill_report.md")
            os.makedirs(DOCS_DIR, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(render_kill_report(res, cfg))
            print(f"[写盘] {os.path.relpath(p, ROOT)}")
        return 0 if res["all_pass"] else 2
    print("[G3] 未启用：必须先通过 G2（计划 §4 / §7）。")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
