#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K形态胜率验证：双口径统计 + 窗口切片 + run_oos 终验 + 经济账 + 阈值敏感性。

对应计划 Step 4~8。上游：scripts/local_naked_k_prepare.py（基线锚定 + 1h 构造）。

口径声明（详见 docs/research/naked-k/REPORT.md）：
1. **双 lens 并行**（注册表 semantics.bet_lens）：
   - `rel` = 相对方向：continuation_h / reversal_h（discovery/targets.py 既有目标族），
     由 target_family 选族；win 语义 = 次根 open→close 方向是否等于（或反于）信号根方向。
   - `abs` = 绝对方向：dirup_h / dirdn_h（本脚本由 build_targets 输出 **代数推导**），
     由 bet_side 选边；win 语义 = 第 t+h 根自身 open→close 是否收阳/收阴。
     推导式：dirup.win = (cont.win & dir_>0) | (rev.win & dir_<0)；ret_dirup = ret_cont·dir_；
     valid 与 continuation_h.valid 逐位相同。不 import、不修改 discovery 特征层与目标层。
2. `win_rate`（相对/绝对方向的次根 open→close 命中）与 `cum_dir_win_rate`
   （= mean(sign(ret)>0)，即信号根收盘 → t+h 收盘的累计收益落在押注方向）**严格分列**：
   两者基准不同（targets.py:80-85），禁止合并称“胜率”。
3. 三段（discovery/validation/holdout）在**同一份全量 target** 上用 seg_stats 的
   [a,b) 任意区间切片得到；360d/180d 是「最近 N 天」后缀切片后重切 0.6/0.2/0.2，
   不重建目标、不重跑 bootstrap。
4. 判决只认 **720d 冻结 holdout**；360d/180d 仅估计「估计量随样本长度的稳定性」，
   三周期仅方向性佐证（共享同一 BTC 价格路径，非独立样本）。
5. run_oos 的 run 块自助在命中数过大时会 OOM（oos_validator.py:32 的
   size=(b, len(v))），本脚本以 in-process monkeypatch 降低 b（**不抽稀命中**，
   避免选择偏倚），代价是蒙特卡洛噪声增大；实际使用的 b 记入 block_ci_b 字段。

用法：
    uv run python scripts/local_naked_k_report.py                 # 全量（Step 4~8）
    uv run python scripts/local_naked_k_report.py --stage stats   # 只跑统计段
    uv run python scripts/local_naked_k_report.py --export-catalog
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from binance_predict.backtest.stats import (FEE, PREMIUM, ev,  # noqa: E402
                                            min_detectable_effect)
from binance_predict.discovery import oos_validator as oos_mod  # noqa: E402
from binance_predict.discovery.data import Klines, load_klines_csv  # noqa: E402
from binance_predict.discovery.hypotheses import DEFAULTS  # noqa: E402
from binance_predict.discovery.l1_tester import bh_fdr, seg_stats  # noqa: E402
from binance_predict.discovery.targets import (TargetSet,  # noqa: E402
                                               build_targets)

import local_naked_k_engine as eng  # noqa: E402
import local_naked_k_prepare as prep  # noqa: E402

BAR_MS = eng.BAR_MS
DAY_MS = eng.DAY_MS
TFS = ("5m", "15m", "1h")
HORIZONS: dict[str, list[int]] = {"5m": [1, 2, 3], "15m": [1, 2, 3, 6, 12],
                                  "1h": [1, 3, 6, 12, 24]}
WINDOWS: list[tuple[str, int | None]] = [("720d", None), ("360d", 360), ("180d", 180)]
LENSES = ("abs", "rel")
SEGS = ("discovery", "validation", "holdout")
MARKET_TFS = ("5m", "15m")            # prediction_trading.py:1285-1288 只接受 5m/15m
STAT_FIELDS = ("n", "baseline", "win_rate", "cum_dir_win_rate", "cum_dir_baseline",
               "lift_pp", "ci_low", "ci_high", "p_value", "avg_win_return",
               "avg_loss_return", "payoff_ratio", "expectancy", "mfe_mean_atr",
               "mae_mean_atr")
MDE_TESTABLE_PP = 5.0                 # 80% 功效可检测下限 > 5pp → 只给描述性数字
BLOCK_CI_ELEMS = 20_000_000           # run_block_ci 自助矩阵元素上限（≈320MB 峰值）
BLOCK_CI_B_MIN, BLOCK_CI_B_MAX = 100, 3000
QUOTE_LAYERS = (("L1_nominal_0.50", 0.50), ("L2_recorded_0.615", 0.615),
                ("L3_delayed_0.73", 0.73))
SENS_FACTORS = (0.8, 1.25)            # 阈值 ±1 档（比例阈乘性缩放并夹回 [0,1]）
CATALOG_PATH = os.path.join(ROOT, "docs", "research", "naked-k", "pattern_catalog.csv")
DOCS_REPORT = os.path.join(ROOT, "docs", "research", "naked-k", "REPORT.md")


# ============================ 通用工具 ============================

def san(v):
    """numpy / NaN / inf → JSON 与 CSV 可表示的值（NaN 与 inf 一律 None，不静默变 0）。"""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.ndarray):
        return [san(x) for x in v.tolist()]
    return v


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_runs(hit_idx: np.ndarray) -> int:
    """命中位置中「连续 run」的个数（run_block_ci 的自助抽样单位）。"""
    hit_idx = np.asarray(hit_idx)
    return 0 if len(hit_idx) == 0 else int(np.count_nonzero(np.diff(hit_idx) != 1) + 1)


def block_ci_b(n_runs: int) -> int:
    """run_block_ci 的实际自助次数（按 run 数反推，保持内存可控）。

    内存主导项是 oos_validator.py:32 的 `size=(b, len(v))`（v 为 run 序列），
    故抽样基数是 run 数而非命中数；连续命中的高频掩码 run 数远小于命中数，
    因此这里按 run 数取 b 比按命中数取 b 更省算力且不损失精度。
    """
    return int(max(BLOCK_CI_B_MIN, min(BLOCK_CI_B_MAX, BLOCK_CI_ELEMS // max(1, n_runs))))


def safe_run_block_ci(hit_idx, wins, b: int = 3000, seed: int = 11):
    """monkeypatch 版 run_block_ci：只按 run 数降低自助次数 b，不改点估计、不抽稀命中。

    代价是蒙特卡洛噪声增大（b 下限 100），故实际使用的 b 必须随结果一起落盘。
    """
    nn = count_runs(hit_idx)
    if nn == 0:
        return (float("nan"), float("nan"))
    return _ORIG_BLOCK_CI(hit_idx, wins, b=block_ci_b(nn), seed=seed)


_ORIG_BLOCK_CI = oos_mod.run_block_ci


# ============================ 基线与产物目录（run_fp 严格匹配） ============================

def resolve_outdir() -> tuple[str, dict, str]:
    """run_fp 由全部输入源码+数据哈希决定；缺 baseline.json 说明尚未跑 prepare。"""
    run_fp, parts = prep.compute_run_fp()
    if parts["absent"]:
        raise SystemExit(f"[FAIL] run_fp 输入源缺失（指纹不可复现）：{parts['absent']}")
    out_dir = os.path.join(prep.OUT_ROOT, run_fp)
    bp = os.path.join(out_dir, "baseline.json")
    if not os.path.exists(bp):
        raise SystemExit(
            f"[FAIL] 未找到 {os.path.relpath(bp, ROOT)}\n"
            f"       run_fp={run_fp} 与已锚定基线不一致（输入变更后必须先重跑 "
            f"scripts/local_naked_k_prepare.py 锚定基线）。")
    with open(bp, encoding="utf-8") as f:
        base = json.load(f)
    if base.get("registry_sha256") != parts["digests"].get("registry"):
        raise SystemExit("[FAIL] baseline.json 的注册表哈希与当前注册表不一致，拒绝继续。")
    return run_fp, base, out_dir


# ============================ 绝对方向目标族（代数推导，不改 discovery） ============================

def derive_abs_targets(tg, dir_: np.ndarray, horizons: list[int]) -> None:
    """向 tg 追加 dirup_h / dirdn_h：把「相对信号根」的目标改写成「绝对收阳/收阴」。

    依据 targets.py:76-95：base_valid 与 fam 无关（两族 valid 逐位相同），
    cont.win = (nxt_dir == dir_)、rev.win = (nxt_dir == -dir_)、ret = d·(c_{t+h}-c_t)/c_t。
    故 dirup.win = (cont.win & dir_>0) | (rev.win & dir_<0)；ret_dirup = ret_cont·dir_；
    MFE/MAE 按 dir_ 符号从两族中选取；dirdn 与 dirup 严格互补（mfe/mae 互换）。
    """
    for hz in horizons:
        cont = tg.items[f"continuation_{hz}"]
        rev = tg.items[f"reversal_{hz}"]
        if not np.array_equal(cont.valid, rev.valid):
            raise AssertionError(f"hz={hz}：continuation/reversal 的 valid 不一致（推导前提破坏）")
        valid = cont.valid
        win_up = (cont.win & (dir_ > 0)) | (rev.win & (dir_ < 0))
        with np.errstate(invalid="ignore"):
            ret_up = np.where(np.isfinite(cont.ret), cont.ret * dir_, np.nan)
            ret_dn = np.where(np.isfinite(ret_up), -ret_up, np.nan)
        mfe_up = np.where(dir_ > 0, cont.mfe_atr, rev.mfe_atr)
        mae_up = np.where(dir_ > 0, cont.mae_atr, rev.mae_atr)
        tg.add(TargetSet(name=f"dirup_{hz}", family="dirup", horizon=hz, valid=valid,
                         win=np.where(valid, win_up, False), ret=ret_up,
                         mfe_atr=mfe_up, mae_atr=mae_up))
        # 做空「绝对收阴」的最大顺波动 = 做多的最大逆波动（路径互补）
        tg.add(TargetSet(name=f"dirdn_{hz}", family="dirdn", horizon=hz, valid=valid,
                         win=np.where(valid, ~win_up, False), ret=ret_dn,
                         mfe_atr=mae_up, mae_atr=mfe_up))


def verify_abs_lens(kl, tg, horizons: list[int]) -> dict:
    """独立重算校验：dirup/dirdn 的 win 必须等于次根 open→close 符号（不经任何代数中转）。"""
    n = len(kl)
    bad = []
    for hz in horizons:
        nd = np.zeros(n, dtype=np.float64)
        nd[: n - hz] = np.sign(kl.c[hz:] - kl.o[hz:])     # 第 t+hz 根自身的 open→close 方向
        for fam, ref in (("dirup", nd > 0), ("dirdn", nd < 0)):
            ts = tg.items[f"{fam}_{hz}"]
            v = ts.valid
            if not np.array_equal(ts.win[v], ref[v]):
                bad.append(f"{fam}_{hz}: 不一致数={int((ts.win[v] != ref[v]).sum())}")
            if not np.all(nd[v] != 0):
                bad.append(f"{fam}_{hz}: valid 内存在 nxt_dir==0（base_valid 前提破坏）")
        # ret 口径独立校验：dirup.ret 必须 = (c_{t+h}-c_t)/c_t
        ts = tg.items[f"dirup_{hz}"]
        ref_ret = np.full(n, np.nan)
        ref_ret[: n - hz] = (kl.c[hz:] - kl.c[: n - hz]) / kl.c[: n - hz]
        v = ts.valid & np.isfinite(ref_ret)
        if not np.allclose(ts.ret[v], ref_ret[v], rtol=0, atol=1e-12):
            bad.append(f"dirup_{hz}: ret 与 (c_{{t+h}}-c_t)/c_t 不符")
    return {"passed": not bad, "errors": bad, "horizons": horizons}


def synthetic_target_check() -> dict:
    """合成 K 线手算对照（Test Plan「target 口径」行）。

    前 23 根同形态小 K（rng_pct=0.03）使前置 ATR 可算；信号根 p=23：
      t=23  o=100 h=112 l=98  c=110（dir=+1, atr_abs=0.03×100=3）
      t=24  o=110 h=115 l=104 c=105（dir=-1）
      t=25  o=100 h=102 l=99  c=101（dir=+1）
    手算（hz=1）：cont.win=False / rev.win=True；ret_cont=1×(105-110)/110=-1/22；
      dirup.win=False / dirdn.win=True；ret_dirup=ret_cont×dir_=-1/22；
      mfe_cont=(115-110)/(3×110)=5/330；mae_cont=(110-104)/(3×110)=6/330；dirup 取 dir_>0 侧。
    手算（hz=2，t+2=25）：cont.win=True（dir=+1 同号）；ret_cont=(101-110)/110=-9/110；
      dirup_2.win=True（第 25 根 o=100→c=101 收阳）且 ret_dirup=-9/110。
    """
    n, p = 26, 23
    o = np.full(n, 100.0); h = np.full(n, 102.0); l = np.full(n, 99.0); c = np.full(n, 101.0)
    o[p], h[p], l[p], c[p] = 100.0, 112.0, 98.0, 110.0
    o[p + 1], h[p + 1], l[p + 1], c[p + 1] = 110.0, 115.0, 104.0, 105.0
    kl = Klines(t=(np.arange(n) * BAR_MS["5m"]).astype(np.int64), o=o, h=h, l=l, c=c,
                v=np.ones(n), cont=np.r_[False, np.ones(n - 1, dtype=bool)])
    dir_ = np.sign(c - o)
    rng_pct = np.where(h > l, h - l, np.nan) / o
    atr_abs = eng._prev(eng._roll_nanmean(rng_pct, 20)) * o
    tg = build_targets(kl.t, kl.o, kl.h, kl.l, kl.c, kl.cont, [1, 2], atr_abs)
    derive_abs_targets(tg, dir_, [1, 2])
    got = {
        "atr_abs_at_p": (float(atr_abs[p]), 3.0),
        "cont1_win": (bool(tg.items["continuation_1"].win[p]), False),
        "rev1_win": (bool(tg.items["reversal_1"].win[p]), True),
        "cont1_ret": (float(tg.items["continuation_1"].ret[p]), -1 / 22),
        "rev1_ret": (float(tg.items["reversal_1"].ret[p]), 1 / 22),
        "dirup1_win": (bool(tg.items["dirup_1"].win[p]), False),
        "dirdn1_win": (bool(tg.items["dirdn_1"].win[p]), True),
        "dirup1_ret": (float(tg.items["dirup_1"].ret[p]), -1 / 22),
        "dirdn1_ret": (float(tg.items["dirdn_1"].ret[p]), 1 / 22),
        "dirup1_mfe": (float(tg.items["dirup_1"].mfe_atr[p]), 5 / 330),
        "dirup1_mae": (float(tg.items["dirup_1"].mae_atr[p]), 6 / 330),
        "dirdn1_mfe": (float(tg.items["dirdn_1"].mfe_atr[p]), 6 / 330),
        "dirdn1_mae": (float(tg.items["dirdn_1"].mae_atr[p]), 5 / 330),
        "valid1_eq_cont1": (bool(np.array_equal(tg.items["dirup_1"].valid,
                                                tg.items["continuation_1"].valid)), True),
        "cont2_win": (bool(tg.items["continuation_2"].win[p]), True),
        "dirup2_win": (bool(tg.items["dirup_2"].win[p]), True),
        "dirup2_ret": (float(tg.items["dirup_2"].ret[p]), -9 / 110),
    }
    bad = [k for k, (a, b) in got.items()
           if not (abs(a - b) < 1e-12 if isinstance(a, float) else a == b)]
    return {"passed": not bad, "checks": {k: list(v) for k, v in got.items()}, "failed": bad}


# ============================ 窗口切片与功效预筛 ============================

def window_bounds(n: int, bar_ms: int, days: int | None) -> dict:
    """days=None → 全量；否则取「最近 days 天」后缀并重切 0.6/0.2/0.2。"""
    if days is None:
        a, b = 0, n
    else:
        rows = min(int(days * DAY_MS // bar_ms), n)
        a, b = n - rows, n
    L = b - a
    return {"a": a, "b": b,
            "i1": a + int(L * DEFAULTS["discovery_frac"]),
            "i2": a + int(L * (DEFAULTS["discovery_frac"] + DEFAULTS["validation_frac"])),
            "rows": L}


def seg_ranges(wb: dict) -> dict[str, tuple[int, int]]:
    return {"discovery": (wb["a"], wb["i1"]), "validation": (wb["i1"], wb["i2"]),
            "holdout": (wb["i2"], wb["b"])}


def target_baselines(ts, wb: dict) -> dict[str, tuple[float, float, int]]:
    """每段的 (win 基准, cum_dir 基准, 有效样本数)——基准只由 valid 全集给出。"""
    out = {}
    for seg, (x, y) in seg_ranges(wb).items():
        v = ts.valid[x:y]
        nn = int(v.sum())
        w = float(ts.win[x:y][v].mean()) if nn else float("nan")
        r = ts.ret[x:y][v]
        cm = float((r > 0).mean()) if nn else float("nan")
        out[seg] = (w, cm, nn)
    return out


def stats_all_segs(ts, hit: np.ndarray, wb: dict, base: dict) -> dict[str, dict]:
    """三段各 15 字段；win_rate 与 cum_dir_win_rate 分列（口径不可混用）。"""
    out = {}
    for seg, (x, y) in seg_ranges(wb).items():
        st = seg_stats(ts, hit, x, y, base[seg][0])
        m = hit[x:y] & ts.valid[x:y]
        nn = int(m.sum())
        ret = ts.ret[x:y][m]
        st["cum_dir_win_rate"] = float((ret > 0).mean()) if nn else float("nan")
        st["cum_dir_baseline"] = base[seg][1]
        out[seg] = st
    return out


def power_tag(n_holdout: int, n_valid_holdout: int, base_h: float, n_all: int) -> dict:
    """功效预筛标签（低频形态仍出数但明确标注，不静默丢弃——与 merge_r3 行为相反）。"""
    mde = min_detectable_effect(n_holdout, base_h if math.isfinite(base_h) else 0.5) * 100 \
        if n_holdout > 0 else float("inf")
    if n_all == 0:
        tag, why = "NO_HITS", "掩码在全样本上零命中"
    elif n_holdout < DEFAULTS["n_min_holdout"]:
        tag, why = "UNTESTABLE", f"holdout 有效命中 {n_holdout} < n_min_holdout"
    elif mde > MDE_TESTABLE_PP:
        tag, why = "DESCRIPTIVE_ONLY", f"80% 功效可检测下限 {mde:.1f}pp > {MDE_TESTABLE_PP}pp"
    else:
        tag, why = "TESTABLE", ""
    # 可达裁决按 oos_validator._verdict 的门槛反推
    if n_holdout >= 120:
        reach = "ROBUST"
    elif n_holdout >= DEFAULTS["n_min_holdout"]:
        reach = "PROMISING"
    else:
        reach = "WEAK_OR_REJECT"
    return {"power_tag": tag, "power_note": why, "mde_pp": mde,
            "reachable_verdict": reach, "n_valid_holdout": n_valid_holdout,
            "min_hit_rate_for_robust": (120 / n_valid_holdout) if n_valid_holdout else float("nan")}


# ============================ 双 lens 归属（逐掩码） ============================
# lens_of 的唯一口径源在 local_naked_k_engine（口径所有者）。本脚本不得另写一套，
# 否则冻结闸门（engine --selfcheck）与实际出数（本脚本）可能用不同规则选主判决 lens。

lens_of = eng.lens_of


def target_name(lens: str, bet_side: str, target_family: str, hz: int) -> str | None:
    if lens == "rel":
        return f"{target_family}_{hz}"
    if bet_side == "up":
        return f"dirup_{hz}"
    if bet_side == "down":
        return f"dirdn_{hz}"
    return None          # bet_side=none 无绝对押注


# ============================ 序列装载与掩码准备（Step 4~5） ============================

def build_series(tf: str, out_dir: str, parents: dict[str, Klines]) -> tuple[Klines, dict]:
    """装载某周期的统计腿序列。

    数据源选择（刻意决策，报告须披露）：
      - 5m：冻结 output/klines_5m_720d.csv（与 output/kline_discovery_*_720d_v2 同源）。
      - 15m：冻结 output/klines_15m_720d.csv（**不换成 5m 聚合**），但与 5m 聚合出的 15m
        做逐根诊断；不一致只 WARN 不中止（两者都是真数据，差异来自导出时刻与缺根处理）。
      - 1h：<run_fp>/klines_1h_720d.csv（由 prepare 从 5m 严格聚合 + 逐桶硬校验产出）。
        若与内存聚合 parents["1h"] 不一致 → 直接中止（产物被污染或聚合不可复现）。
    """
    csv_map = {"5m": prep.CSV_5M, "15m": prep.CSV_15M}
    if tf in csv_map:
        kl = load_klines_csv(csv_map[tf], BAR_MS[tf])
    else:
        path = os.path.join(out_dir, "klines_1h_720d.csv")
        if not os.path.exists(path):
            raise SystemExit(f"[FAIL] 缺少 {path}：先运行 scripts/local_naked_k_prepare.py")
        kl = load_klines_csv(path, BAR_MS["1h"])
    ref = parents.get(tf)
    diag: dict = {"rows": int(len(kl)), "cont_breaks": int((~kl.cont[1:]).sum()),
                  "first_ts": str(kl.t[0]), "last_ts": str(kl.t[-1]),
                  "span_days": round(float(kl.t[-1] - kl.t[0]) / DAY_MS, 3)}
    if ref is None:
        diag["aggregate_reference"] = "N/A"
    elif tf == "1h":
        if len(ref) != len(kl) or not np.array_equal(ref.t, kl.t):
            raise SystemExit("[FAIL] 1h 产物与 5m 内存聚合结果不一致，拒绝继续（基线被污染？）")
        diag["aggregate_identical"] = True
        diag["ohlc_max_abs_diff"] = 0.0
    else:
        same_t = len(ref) == len(kl) and bool(np.array_equal(ref.t, kl.t))
        diag["rows_from_5m_aggregation"] = int(len(ref))
        diag["timestamps_identical"] = same_t
        if same_t:
            diag["ohlc_max_abs_diff"] = float(max(
                np.abs(ref.o - kl.o).max(), np.abs(ref.h - kl.h).max(),
                np.abs(ref.l - kl.l).max(), np.abs(ref.c - kl.c).max()))
        else:
            diag["ohlc_max_abs_diff"] = None
            diag["warn"] = (f"{tf}：冻结 CSV 与 5m 聚合行数/时间轴不一致，"
                            f"统计腿仍用冻结 CSV（已披露）")
    return kl, diag


def attach_lens(ns: dict, masks: list[dict]) -> list[str]:
    """给每个掩码挂上双 lens 归属，并返回 CONFLICT 清单（声明押注与掩码实态相反）。"""
    dir_ = ns["dir_"]
    conflicts: list[str] = []
    for m in masks:
        m.update(lens_of(m["_mask"], dir_, m["bet_side"], m["target_family"]))
        if str(m["lens_status"]).startswith("CONFLICT"):
            conflicts.append(f'{m["mask_key"]}: {m["lens_status"]}')
    return conflicts


def identity_fields(m: dict, pat: dict) -> dict:
    srcs = pat.get("sources") or []
    return {"pattern_id": m["pattern_id"], "mask_key": m["mask_key"],
            "name_cn": pat.get("name_cn"), "layer": pat.get("layer"),
            "family": pat.get("family"), "target_family": m["target_family"],
            "bet_side": m["bet_side"], "expr": pat.get("expr"),
            "expr_full": m["expr_full"], "gate": m["gate"],
            "tier": pat.get("tier"), "status": pat.get("status"),
            "overlap_with": pat.get("overlap_with") or "",
            "market_caliber": pat.get("market_caliber"),
            "freq_prior": pat.get("freq_prior"), "mechanism": pat.get("mechanism"),
            "n_sources": len(srcs), "sources": "|".join(srcs),
            "hit_sig": m["hit_sig"], "aliases": "|".join(m["aliases"]),
            "n_aliases": len(m["aliases"])}


def mask_power(ts, mask: np.ndarray, wb: dict, base: dict) -> dict:
    """该（掩码 × 目标 × 窗口）格的功效标签：以 holdout 有效命中数为准。"""
    x, y = seg_ranges(wb)["holdout"]
    v = ts.valid[x:y]
    va = ts.valid[wb["a"]:wb["b"]]
    n_all = int((mask[wb["a"]:wb["b"]] & va).sum())
    return power_tag(int((mask[x:y] & v).sum()), int(v.sum()), base["holdout"][0], n_all)


# ============================ Step 6：主口径三段双 lens 统计 ============================

def stage_stats(tf: str, kl: Klines, ns: dict, tg, masks: list[dict],
                reg_index: dict[str, dict]) -> tuple[list[dict], dict]:
    """每 掩码 × 窗口 × lens × horizon × 三段 全量出数（低频形态不丢弃）。"""
    n, bar = len(kl), BAR_MS[tf]
    t0 = time.time()
    records: list[dict] = []
    for wname, days in WINDOWS:
        wb = window_bounds(n, bar, days)
        for m in masks:
            ident = identity_fields(m, reg_index[m["pattern_id"]])
            for lens in LENSES:
                for hz in HORIZONS[tf]:
                    tname = target_name(lens, m["bet_side"], m["target_family"], hz)
                    if tname is None or tname not in tg.items:
                        continue
                    ts = tg.items[tname]
                    base = target_baselines(ts, wb)
                    st = stats_all_segs(ts, m["_mask"], wb, base)
                    rec = dict(ident)
                    rec.update({"tf": tf, "window": wname, "lens": lens, "horizon": hz,
                                "target": tname,
                                "primary_lens": m["primary_lens"],
                                "is_primary_lens": bool(lens == m["primary_lens"]),
                                "lens_status": m["lens_status"],
                                "lenses_equivalent": m["lenses_equivalent"],
                                "dir_pinned": m["dir_pinned"], "thin_pin": m["thin_pin"],
                                "n_dir": m["n_dir"],
                                "n_dir_pos": m["n_dir_pos"], "n_dir_neg": m["n_dir_neg"],
                                "window_rows": wb["rows"], "seg_i1": wb["i1"],
                                "seg_i2": wb["i2"]})
                    for seg in SEGS:
                        for f in STAT_FIELDS:
                            rec[f"{seg}_{f}"] = st[seg].get(f)
                        cw, cb = st[seg]["cum_dir_win_rate"], st[seg]["cum_dir_baseline"]
                        rec[f"{seg}_cum_dir_lift_pp"] = (
                            (cw - cb) * 100.0
                            if cw is not None and cb is not None
                            and math.isfinite(cw) and math.isfinite(cb) else None)
                    rec.update(mask_power(ts, m["_mask"], wb, base))
                    records.append(rec)
    return records, {"tf": tf, "n_records": len(records),
                     "elapsed_sec": round(time.time() - t0, 1)}


# ============================ BH-FDR 分族 + 全量对照 ============================

def apply_fdr(records: list[dict]) -> dict:
    """按 (tf, window, lens, segment) 分族做 BH-FDR，另做一族全量对照。

    两者都**不是**最终判决——最终判决只认 720d 冻结 holdout 的 run_oos 裁决（Step 7）。
    """
    by_family: dict[tuple, list[int]] = defaultdict(list)
    n_p = 0
    for i, r in enumerate(records):
        for seg in SEGS:
            p = r.get(f"{seg}_p_value")
            if p is None or not math.isfinite(p):
                r[f"{seg}_fdr_pass"] = None
                continue
            by_family[(r["tf"], r["window"], r["lens"], seg)].append(i)
            r[f"{seg}_fdr_pass"] = False        # 先置否，回填覆盖
            n_p += 1
    ledger: dict[str, int] = {}
    for fam, idxs in sorted(by_family.items(), key=lambda kv: str(kv[0])):
        seg = fam[3]
        passes = bh_fdr([records[i][f"{seg}_p_value"] for i in idxs],
                        q=DEFAULTS["fdr_alpha"])
        for i, ok in zip(idxs, passes):
            records[i][f"{seg}_fdr_pass"] = bool(ok)
        ledger["/".join(str(x) for x in fam)] = len(idxs)
    # 全量对照族：只对所有有限 p 再做一次 BH（最保守门槛）
    for seg in SEGS:
        idxs = [i for i, r in enumerate(records)
                if r.get(f"{seg}_p_value") is not None
                and math.isfinite(r[f"{seg}_p_value"])]
        for i in idxs:
            records[i][f"{seg}_fdr_pass_global"] = False
        if idxs:
            passes = bh_fdr([records[i][f"{seg}_p_value"] for i in idxs],
                            q=DEFAULTS["fdr_alpha"])
            for i, ok in zip(idxs, passes):
                records[i][f"{seg}_fdr_pass_global"] = bool(ok)
        ledger[f"GLOBAL/{seg}"] = len(idxs)
    return {"fdr_alpha": DEFAULTS["fdr_alpha"], "family_sizes": ledger,
            "n_families": len(by_family), "n_pvalues_total": n_p,
            "n_fdr_pass_holdout": sum(1 for r in records if r.get("holdout_fdr_pass")),
            "n_fdr_pass_holdout_global": sum(
                1 for r in records if r.get("holdout_fdr_pass_global"))}


# ============================ Step 7：run_oos 稳健性终验（holdout 只碰一次） ============================

HOLDOUT_FLAG = "_holdout_touched.flag"


def holdout_flag_state(out_dir: str, allow_retouch: bool) -> tuple[dict, str]:
    """幂等守卫：flag 存在即拒绝再跑 Step 7（防止 holdout 被多次窥视）。"""
    path = os.path.join(out_dir, HOLDOUT_FLAG)
    state = ({"touches": 0, "history": []}
             if not os.path.exists(path)
             else json.load(open(path, encoding="utf-8")))
    if int(state.get("touches", 0)) > 0 and not allow_retouch:
        raise SystemExit(
            f"[FAIL] {HOLDOUT_FLAG} 已存在（touches={state['touches']}）。\n"
            f"       Step 7 的冻结 holdout 只允许触碰一次；重复执行会污染终验。\n"
            f"       确因输入变更需重跑时显式加 --allow-retouch（会在 flag 中累计记录）。")
    return state, path


def write_holdout_flag(path: str, state: dict, detail: dict) -> None:
    state["touches"] = int(state.get("touches", 0)) + 1
    state.setdefault("history", []).append(detail)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


TOUCH_LEDGER = os.path.join(prep.OUT_ROOT, "_holdout_touches.jsonl")


def record_holdout_touch(run_fp: str, tfs: list[str]) -> int:
    """跨 run_fp 的 holdout 触碰总账（append-only）；返回本次是第几次物理执行 Step 7。

    为什么需要它：`_holdout_touched.flag` 落在 `<run_fp>/` 内，而「任何输入变更 → 新目录」
    恰好会把守卫清空 —— 守卫在最需要它的场合失效。总账不随指纹变化，报告据此如实披露
    Step 7 的物理执行次数，而不是让每个新目录都显示「只碰了一次」。
    """
    seq = 0
    if os.path.exists(TOUCH_LEDGER):
        with open(TOUCH_LEDGER, encoding="utf-8") as fh:
            seq = sum(1 for line in fh if line.strip())
    os.makedirs(prep.OUT_ROOT, exist_ok=True)
    with open(TOUCH_LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": seq + 1, "touched_at_utc": now_utc(),
                             "run_fp": run_fp, "tfs": list(tfs)},
                            ensure_ascii=False) + "\n")
    return seq + 1


OOS_SCALARS = {
    "holdout_n": "oos_holdout_n", "holdout_baseline": "oos_holdout_baseline",
    "holdout_win_rate": "oos_holdout_win_rate", "holdout_lift_pp": "oos_holdout_lift_pp",
    "holdout_ci_low": "oos_holdout_ci_low", "holdout_ci_high": "oos_holdout_ci_high",
    "holdout_p_value": "oos_holdout_p_value",
    "holdout_payoff_ratio": "oos_holdout_payoff_ratio",
    "holdout_expectancy": "oos_holdout_expectancy",
    "holdout_mfe_mean_atr": "oos_holdout_mfe_mean_atr",
    "holdout_mae_mean_atr": "oos_holdout_mae_mean_atr",
    "block_ci_low": "oos_block_ci_low", "block_ci_high": "oos_block_ci_high",
    "temporal_consistency": "oos_temporal_consistency",
    "regime_consistency": "oos_regime_consistency",
    "oos_degradation_pp": "oos_degradation_pp", "oos_retention": "oos_retention",
    "flipped": "oos_flipped", "holdout_n_ok": "oos_holdout_n_ok",
    "above_breakeven": "oos_above_breakeven", "breakeven": "oos_breakeven_nominal",
    "ev_at_0.50": "oos_ev_at_0.50", "kelly": "oos_kelly",
    "verdict": "oos_verdict", "score": "oos_score",
}
OOS_DETAIL_KEYS = ("walk_forward", "monthly_detail", "regime_detail")


def wf_encode(wf, field: str) -> str:
    """把 run_oos 的 walk_forward 折列表压成一个 CSV 单元格（分号分隔，避免与 md 表格撞 `|`）。

    唯一编码点：findings.csv 与报告正文同源，杜绝手抄出第二套口径。
    注意 walk_forward 的折边界是**全 720d** 等分（见 oos_validator.py:86-104），
    故折 7/8 与 holdout 重叠 —— 本列衡量估计量的时间稳定性，不是第 4 段独立复现。
    """
    out = []
    for fold in wf or []:
        v = fold.get(field)
        if v is None:
            out.append(f"F{fold.get('fold')}=NA")
        elif field == "win_rate":
            out.append(f"F{fold.get('fold')}:{float(v):.6f}")
        else:
            out.append(f"F{fold.get('fold')}={int(v)}")
    return ";".join(out)


def stage_oos(tf: str, kl: Klines, tg, ns: dict, masks: list[dict],
              rec_index: dict[tuple, dict]) -> tuple[list[dict], dict]:
    """绕开 merge_r3 的 n_min_l2 静默丢弃，直喂 run_oos（仅 720d、仅主 lens）。

    row 契约只需 `_mask` + `discovery_lift_pp`（oos_validator.py:143-152）+ 身份字段。
    OOM 防线：in-process 替换 run_block_ci，按 run 数降 b（不抽稀命中）。
    """
    n = len(kl)
    combo: dict[str, list[dict]] = {}
    skipped: list[str] = []
    for m in masks:
        lens = m["primary_lens"]
        for hz in HORIZONS[tf]:
            tname = target_name(lens, m["bet_side"], m["target_family"], hz)
            if tname is None:
                continue
            rec = rec_index.get((tf, m["mask_key"], "720d", lens, hz))
            if rec is None:
                skipped.append(f"{m['mask_key']}/{tname}: NO_STEP6_RECORD")
                continue
            d_lift = rec["discovery_lift_pp"]
            if d_lift is None or not math.isfinite(d_lift):
                skipped.append(f"{m['mask_key']}/{tname}: discovery 段零命中（不可判方向先验）")
                continue
            full_mask = m["_mask"] & tg.items[tname].valid
            n_runs = count_runs(np.flatnonzero(full_mask))
            combo.setdefault(tname, []).append({
                "_mask": m["_mask"], "discovery_lift_pp": float(d_lift),
                "mask_key": m["mask_key"], "pattern_id": m["pattern_id"],
                "tf": tf, "window": "720d", "lens": lens, "horizon": hz,
                "target": tname, "lens_status": m["lens_status"],
                "n_full_hits": int(full_mask.sum()), "n_full_runs": n_runs,
                "discovery_win_rate": rec["discovery_win_rate"],
                "block_ci_b": block_ci_b(n_runs)})
    wrapped = {t: {"shortlist": rows, "l2_kept": [], "l3_kept": [],
                   "n_l2_tests": 0, "n_l3_tests": 0} for t, rows in combo.items()}
    volp = ns.get("atrp")
    has_vol = isinstance(volp, np.ndarray) and len(volp) == n
    oos_mod.run_block_ci = safe_run_block_ci          # try/finally 必还原
    t0 = time.time()
    try:
        res = oos_mod.run_oos(tg, kl.t.astype("datetime64[ms]"), wrapped, n,
                              volp if has_vol else None, None)
    finally:
        oos_mod.run_block_ci = _ORIG_BLOCK_CI
    flat = [row for tname in res for row in res[tname]]
    return flat, {"tf": tf, "scope": "720d × primary_lens × 全 horizons",
                  "n_targets_with_rows": len([t for t in res if res[t]]),
                  "n_oos_rows": len(flat), "n_skipped": len(skipped), "skipped": skipped,
                  "min_block_ci_b": min([r["block_ci_b"] for r in flat], default=None),
                  "vol_pctile_source": ("engine.atrp（与生产 atr_pctile_4320 同口径）"
                                        if has_vol else "MISSING → run_oos 内部按命中集现算"),
                  "elapsed_sec": round(time.time() - t0, 1)}


def merge_oos_into_records(flat: list[dict],
                           rec_index: dict[tuple, dict]) -> tuple[list[dict], dict]:
    """把 run_oos 标量回填到 Step 6 记录，并交叉校验 holdout 数字必须逐位一致。"""
    audit: list[dict] = []
    matched = 0
    for row in flat:
        key = (row["tf"], row["mask_key"], row["window"], row["lens"], row["horizon"])
        kid = "/".join(str(x) for x in key)
        rec = rec_index.get(key)
        if rec is None:
            audit.append({"key": kid, "issue": "NO_STEP6_RECORD"})
            continue
        for k, col in OOS_SCALARS.items():
            rec[col] = san(row.get(k))
        rec["oos_block_ci_b"] = row["block_ci_b"]
        rec["oos_n_full_hits"] = row["n_full_hits"]
        rec["oos_n_full_runs"] = row["n_full_runs"]
        rec["oos_walk_forward"] = wf_encode(row.get("walk_forward"), "win_rate")
        rec["oos_walk_forward_n"] = wf_encode(row.get("walk_forward"), "n")
        rec["oos_detail"] = {k: row.get(k) for k in OOS_DETAIL_KEYS}
        matched += 1
        if rec.get("holdout_n") != row["holdout_n"]:
            audit.append({"key": kid, "issue": "HOLDOUT_N_MISMATCH",
                          "step6": rec.get("holdout_n"), "run_oos": row["holdout_n"]})
        for f6 in ("holdout_win_rate", "holdout_lift_pp"):
            a, b = rec.get(f6), row.get(f6)
            if (a is not None and b is not None and math.isfinite(float(a))
                    and math.isfinite(float(b)) and abs(float(a) - float(b)) > 1e-9):
                audit.append({"key": kid, "issue": f"{f6}_DRIFT",
                              "step6": a, "run_oos": b})
    return audit, {"n_oos_rows": len(flat), "n_matched": matched,
                   "n_audit_issues": len(audit)}


# ============================ Step 8：副口径经济账（四层报价） ============================

def q_max_entry(p: float) -> float:
    """可接受入场价上限：ev(p, q)=0 的解 → q = (1-FEE)·p - PREMIUM = 0.98p - 0.01。"""
    return (1.0 - FEE) * p - PREMIUM


def breakeven_p(q: float) -> float:
    """给定入场价 q 的费后打平胜率（与 q_max_entry 互为反函数）。"""
    return (q + PREMIUM) / (1.0 - FEE)


QUOTE_PER_MS = {"5m": 300_000, "15m": 900_000}
QUOTE_PCTS = (10, 25, 50, 75, 90)


def quote_surface(path: str = prep.SAMPLES) -> dict:
    """实测报价曲面 q̂(周期, 分钟桶, 状态) —— 只统计不推断。

    状态 = UP/DOWN token（=押注方向）；深度 = 分位数；入场时刻 = 桶 0（周期开盘）。
    必须随报告披露：样本跨度极短且集中于单一 regime，q̂ 仅作分档参考，
    主经济账用闭式（q_max / p*）而非曲面点估计。
    """
    rows = json.load(open(path, encoding="utf-8"))
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: {"up": [], "dn": []})
    cycles: dict[str, set] = defaultdict(set)
    span: dict[str, list[int]] = defaultdict(list)
    sums: dict[str, list[float]] = defaultdict(list)
    skipped_period, missing_price = 0, 0
    for s in rows:
        mp = s.get("market_period")
        if mp not in QUOTE_PER_MS:
            skipped_period += 1
            continue
        up, dn = s.get("up_price"), s.get("down_price")
        if up is None or dn is None:
            missing_price += 1
            continue
        ts = int(s["timestamp"])
        cyc = ts // QUOTE_PER_MS[mp]
        b = int((ts - cyc * QUOTE_PER_MS[mp]) // 60_000)
        acc[(mp, b)]["up"].append(float(up))
        acc[(mp, b)]["dn"].append(float(dn))
        sums[mp].append(float(up) + float(dn))
        cycles[mp].add(cyc)
        span[mp].append(ts)

    def dist(v: list[float]) -> dict:
        a = np.asarray(v, dtype=np.float64)
        out: dict = {"n": int(a.size), "mean": float(a.mean())}
        out.update({f"p{p}": float(np.percentile(a, p)) for p in QUOTE_PCTS})
        return out

    by_bucket = {mp: {str(b): {"up": dist(a["up"]), "dn": dist(a["dn"])}
                      for (m, b), a in sorted(acc.items()) if m == mp}
                 for mp in QUOTE_PER_MS}
    open_bucket = {mp: {"up": dist(acc[(mp, 0)]["up"]), "dn": dist(acc[(mp, 0)]["dn"])}
                   for mp in QUOTE_PER_MS if (mp, 0) in acc}

    def spread(mp: str) -> dict:
        """up+down 偏离 1 的分散度：均值≈1 不等价于逐条≈1，两个尺度必须同时出。"""
        raw = np.asarray(sums[mp], dtype=np.float64)
        dev = np.abs(raw - 1.0)
        return {"n_samples": int(raw.size), "n_cycles": len(cycles[mp]),
                "span_days": round((max(span[mp]) - min(span[mp])) / DAY_MS, 2),
                "first_ts": min(span[mp]), "last_ts": max(span[mp]),
                "up_plus_dn_mean": round(float(raw.mean()), 6),
                "up_plus_dn_median": round(float(np.median(raw)), 6),
                "up_plus_dn_min": round(float(raw.min()), 4),
                "up_plus_dn_max": round(float(raw.max()), 4),
                "up_plus_dn_max_abs_dev": round(float(dev.max()), 9),
                "frac_gt_1p01": round(float((raw > 1.01).mean()), 6),
                "frac_lt_0p99": round(float((raw < 0.99).mean()), 6),
                "frac_abs_dev_gt_0p01": round(float((dev > 0.01).mean()), 6),
                "frac_abs_dev_gt_0p05": round(float((dev > 0.05).mean()), 6)}

    return {
        "source_file": os.path.relpath(path, ROOT).replace("\\", "/"),
        "n_records": len(rows), "n_skipped_other_period": skipped_period,
        "n_missing_price": missing_price,
        "by_bucket": by_bucket, "open_bucket": open_bucket,
        "summary": {mp: spread(mp) for mp in QUOTE_PER_MS if span[mp]},
        "caveats": [
            "报价样本跨度极短（见 summary.span_days），且绝大多数数据点在 08-19 后单一 regime。",
            "up_price + down_price 并不逐条恒等于 1：中位数与均值都≈1，但分散度见 summary 的 "
            "frac_abs_dev_gt_0p01 / up_plus_dn_min / up_plus_dn_max。偏离方向大致对称（"
            "frac_gt_1p01 ≈ frac_lt_0p99），说明它是两侧报价快照不同步的噪声，而不是可提取的做市抽水；"
            "因此 PREMIUM=0.01 仍是外生滑点假设，不是市场事实。",
            "L4 用开盘桶的 q̂ 中位数，对上述离群不敏感；主经济账仍用闭式（q_max / p*）而非曲面点估计。",
            "1h 在预测市场无对应周期（prediction_trading.py:1285-1288 只接受 5m/15m）→ 均不折算 EV。",
            "本曲面只描述报价，不提供任何形态的条件报价；形态级 q̂ 需按周期×状态×深度分档。",
        ],
    }


def stage_economy(records: list[dict], qs: dict) -> dict:
    """副口径：把 holdout 方向命中率折算成四层报价下的可交易性（1h 一律 NO_MARKET）。"""
    measured = qs.get("open_bucket", {})
    counters: dict[str, int] = defaultdict(int)
    for r in records:
        r["market_exists"] = r["tf"] in MARKET_TFS
        if not r["market_exists"]:
            r["econ_scope"] = "NO_MARKET_1H_NOT_LISTED"
            continue
        if r["lens"] != "abs" or r["horizon"] != 1:
            r["econ_scope"] = "NOT_QUOTE_ALIGNED"      # 次根 open→close 才等价于市场结算口径
            continue
        side = "up" if r["target"].startswith("dirup") else "down"
        r["econ_scope"] = "EVALUATED"
        r["econ_side"] = side
        p = r.get("holdout_win_rate")
        if p is None or not math.isfinite(p):
            r["econ_scope"] = "NO_HOLDOUT_HITS"
            r["econ_note"] = "holdout 零命中，无法折算"
            if r["is_primary_lens"] and r["window"] == "720d":
                counters["econ_no_p"] += 1
            counters["econ_no_p_any_window_or_lens"] += 1
            continue
        qm = measured.get(r["tf"], {}).get(side, {}).get("p50")
        r["econ_p_holdout"] = p
        r["ev_nominal_0.50"] = ev(p, 0.5)
        r["econ_q_max"] = q_max_entry(p)
        ci = r.get("holdout_ci_low")
        r["econ_q_max_ci_low"] = q_max_entry(ci) if ci is not None and math.isfinite(ci) else None
        layers = list(QUOTE_LAYERS) + [("L4_measured_b0", qm)]
        # 计数器只累加「主 lens × 720d」的行：判决只认冻结 holdout，而 360d/180d 是
        # 720d 的时间子集，把三种窗口混进同一个计数会凭空括大“可交易形态数”。
        counted = bool(r["is_primary_lens"]) and r["window"] == "720d"
        pt, cit = [], []
        for tag, q in layers:
            if q is None:
                continue
            ps = breakeven_p(q)
            r[f"econ_qbar_{tag}"] = q
            r[f"econ_pstar_{tag}"] = ps
            r[f"econ_headroom_{tag}"] = p - ps
            ok = bool(p > ps)
            r[f"econ_tradable_{tag}"] = ok
            ci_ok = None
            if ci is not None and math.isfinite(ci):
                ci_ok = bool(ci > ps)               # 比点估计更保守：用 Wilson 下界
                r[f"econ_ci_tradable_{tag}"] = ci_ok
            pt.append(ok)
            cit.append(bool(ci_ok))
            if counted:
                counters[f"tradable_{tag}"] += int(ok)
                counters[f"ci_tradable_{tag}"] += int(bool(ci_ok))
        r["econ_q_measured_side"] = qm
        r["econ_tradable_any"] = any(pt) if pt else None
        # 报价层对 p* 单调（q 越高越严），故 all_layers 等价于「最严层 L3(q=0.73) 也过」
        r["econ_tradable_all_layers"] = all(pt) if pt else None
        r["econ_ci_tradable_all_layers"] = (all(cit) if (pt and len(cit) == len(pt))
                                            else None)
        if counted:
            counters["n_evaluated"] += 1
            counters["tradable_none"] += int(bool(pt) and not any(pt))
            counters["ci_tradable_none"] += int(bool(cit) and not any(cit))
            counters["tradable_all_layers"] += int(bool(pt) and all(pt))
            counters["ci_tradable_all_layers"] += int(bool(cit) and len(cit) == len(pt)
                                                      and all(cit))
    return {"layers": [{"tag": t, "q": q, "p_star": breakeven_p(q), "kind": "constant"}
                       for t, q in QUOTE_LAYERS]
            + [{"tag": f"L4_measured_b0/{tf}-{side}", "q": v.get("p50"),
                "p_star": (breakeven_p(v["p50"]) if v.get("p50") is not None else None),
                "kind": "measured", "n_samples": v.get("n")}
               for tf, d in sorted(measured.items()) for side, v in sorted(d.items())],
            "counters": dict(counters),
            "counter_scope": ("仅 is_primary_lens 且 window=720d 的行（holdout 只此一次判决）；"
                              "per-row 经列仍对全部窗口/双 lens 标注，供参考但不入计数"),
            "note": ("主判定用 p*>(实测报价)，0.5204 仅为名义参照（入场价 0.50 的理论假设，"
                     "被 db/models.py:761 的实测记录推翻）。L4 为样本内实测报价中位数（桶 0），"
                     "不是常数假设。ci_tradable_* 用 Wilson 下界代替点估计，是更保守的口径。")}


# ============================ 阈值敏感性（不碰 holdout） ============================

_DEC_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+\.\d+)")


def scale_expr(expr: str, factor: float) -> tuple[str, int, int]:
    """把判据里所有**带小数点的比例/倍数阈值**乘以 factor（0~1 的比率夹回 [0,1]）。

    只动带小数点的字面量：整数在判据里几乎全窗口长度/根数下标（如 h16、dir1、20），
    乘它们会变成另一个问题（窗口长度敏感性不是本节的标的）。
    返回 (新表达式, 改动数, 被夹回 [0,1] 的个数)。
    """
    hits = {"n": 0, "clamped": 0}

    def rep(m: re.Match) -> str:
        hits["n"] += 1
        v = float(m.group(1))
        nv = v * factor
        if 0.0 <= v <= 1.0 and not (0.0 <= nv <= 1.0):
            nv = min(1.0, max(0.0, nv))
            hits["clamped"] += 1
        return f"{nv:.10g}"

    return _DEC_RE.sub(rep, expr), hits["n"], hits["clamped"]


SENS_FACTORS_ALL = (1.0,) + SENS_FACTORS


def stage_sensitivity(tf: str, kl: Klines, ns: dict, masks: list[dict],
                      tg, reg_index: dict[str, dict]) -> tuple[list[dict], dict]:
    """关键阈值 ±1 档重跑（只算 discovery/validation 两段）。

    纪律：故意**不**在扰动版掩码上报 holdout 数字。阈值变体属多重规格比较，
    若把它放到冻结 holdout 上比较就构成窥视，会莫名其妙地提高“复现”假阳性。
    """
    n, bar = len(kl), BAR_MS[tf]
    wb = window_bounds(n, bar, None)                      # 720d
    hz = HORIZONS[tf][0]                                  # 副口径同一根（市场结算口径）
    rows: list[dict] = []
    for m in masks:
        lens = m["primary_lens"]
        tname = target_name(lens, m["bet_side"], m["target_family"], hz)
        if tname is None:
            continue
        ts = tg.items[tname]
        base = target_baselines(ts, wb)
        for factor in SENS_FACTORS_ALL:
            expr_s, nch, ncl = scale_expr(m["expr_full"], factor)
            if nch == 0:
                continue                                   # 无比例阈值→无敏感性可谈
            mask = eng.eval_expr(expr_s, ns)
            st = stats_all_segs(ts, mask, wb, base)
            pat = reg_index[m["pattern_id"]]
            rows.append({
                "tf": tf, "pattern_id": m["pattern_id"], "mask_key": m["mask_key"],
                "name_cn": pat.get("name_cn"), "layer": pat.get("layer"),
                "target": tname, "lens": lens, "horizon": hz,
                "factor": factor, "n_thresholds": nch, "n_clamped": ncl,
                "expr_variant": expr_s,
                "n_hits_all": int(mask.sum()),
                "discovery_n": st["discovery"]["n"],
                "discovery_win_rate": st["discovery"]["win_rate"],
                "discovery_lift_pp": st["discovery"]["lift_pp"],
                "validation_n": st["validation"]["n"],
                "validation_win_rate": st["validation"]["win_rate"],
                "validation_lift_pp": st["validation"]["lift_pp"],
                "holdout_reported": False,
            })
    flips: list[dict] = []
    by_key: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        by_key[(r["tf"], r["mask_key"], r["target"])][r["factor"]] = r
    for key, per in by_key.items():
        b = per.get(1.0)
        if b is None:
            continue
        for fac, r in per.items():
            if fac == 1.0:
                continue
            sg = lambda x, y: (0 if (x[y] is None or not math.isfinite(x[y]) or x[y] == 0)
                               else (1 if x[y] > 0 else -1))
            r["flip_discovery"] = bool(sg(b, "discovery_lift_pp") != 0
                                       and sg(r, "discovery_lift_pp") != 0
                                       and sg(b, "discovery_lift_pp") != sg(r, "discovery_lift_pp"))
            r["flip_validation"] = bool(sg(b, "validation_lift_pp") != 0
                                        and sg(r, "validation_lift_pp") != 0
                                        and sg(b, "validation_lift_pp") != sg(r, "validation_lift_pp"))
            r["n_change_ratio"] = ((r["n_hits_all"] - b["n_hits_all"]) / b["n_hits_all"]
                                   if b["n_hits_all"] else None)
            for seg in ("discovery", "validation"):
                if r[f"flip_{seg}"]:
                    flips.append({"tf": key[0], "mask_key": key[1], "target": key[2],
                                  "factor": fac, "segment": seg,
                                  "base_lift_pp": b[f"{seg}_lift_pp"],
                                  "var_lift_pp": r[f"{seg}_lift_pp"]})
    return rows, {"tf": tf, "n_rows": len(rows),
                  "n_masks_with_thresholds": len({r["mask_key"] for r in rows}),
                  "n_flips": len(flips), "flips": flips}


# ============================ 产物落盘 ============================

def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(san(obj), f, ensure_ascii=False)


def write_csv(path: str, rows: list[dict], preferred: tuple[str, ...] = ()) -> dict:
    """宽表落盘：列序 = 指定优先列 + 其余按名字排序；NaN/inf 已在 san 里变 None。"""
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    head = [c for c in preferred if c in cols]
    tail = sorted(c for c in cols if c not in head)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=head + tail, extrasaction="ignore",
                            restval="")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: san(v) for k, v in r.items() if k in cols})
    return {"path": os.path.relpath(path, ROOT).replace("\\", "/"),
            "rows": len(rows), "cols": len(head) + len(tail)}


HEAD_COLS = ("pattern_id", "mask_key", "name_cn", "layer", "family", "target_family",
             "bet_side", "primary_lens", "is_primary_lens", "lens_status",
             "lenses_equivalent", "dir_pinned", "thin_pin", "n_dir", "n_dir_pos", "n_dir_neg",
             "tf", "window", "lens", "horizon",
             "target", "tier", "status", "overlap_with", "aliases", "n_aliases",
             "expr", "gate", "market_caliber", "power_tag", "reachable_verdict",
             "mde_pp", "n_valid_holdout", "min_hit_rate_for_robust",
             "discovery_n", "discovery_win_rate", "discovery_cum_dir_win_rate",
             "discovery_lift_pp", "discovery_p_value", "discovery_fdr_pass",
             "validation_n", "validation_win_rate", "validation_cum_dir_win_rate",
             "validation_lift_pp", "validation_p_value", "validation_fdr_pass",
             "holdout_n", "holdout_win_rate", "holdout_cum_dir_win_rate",
             "holdout_lift_pp", "holdout_p_value", "holdout_fdr_pass",
             "oos_verdict", "oos_score", "oos_retention", "oos_temporal_consistency",
             "oos_regime_consistency", "oos_flipped", "oos_block_ci_low",
             "oos_block_ci_high", "oos_block_ci_b",
             "oos_walk_forward", "oos_walk_forward_n",
             "econ_scope", "econ_p_holdout", "econ_q_max", "econ_tradable_all_layers",
             "econ_ci_tradable_all_layers",
             "window_rows", "seg_i1", "seg_i2",
             "run_fp", "registry_sha256", "causality_passed", "abs_lens_verified")


def grid_rows(records: list[dict]) -> list[dict]:
    """9 格总览：每 (tf, window) 一格，按主 lens + 最短 horizon 的功效标签分布。"""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["is_primary_lens"] and r["horizon"] == HORIZONS[r["tf"]][0]:
            cells[(r["tf"], r["window"])].append(r)
    order = {tf: i for i, tf in enumerate(TFS)}
    worder = {w: i for i, (w, _) in enumerate(WINDOWS)}
    out = []
    for (tf, wname), rs in sorted(cells.items(), key=lambda kv: (order[kv[0][0]],
                                                                worder[kv[0][1]])):
        tags = Counter(r["power_tag"] for r in rs)
        out.append({"tf": tf, "window": wname, "masks": len(rs),
                    "rows_total": sum(1 for r in records if r["tf"] == tf and r["window"] == wname),
                    "TESTABLE": tags.get("TESTABLE", 0),
                    "DESCRIPTIVE_ONLY": tags.get("DESCRIPTIVE_ONLY", 0),
                    "UNTESTABLE": tags.get("UNTESTABLE", 0),
                    "NO_HITS": tags.get("NO_HITS", 0),
                    "seg_i1": rs[0]["seg_i1"], "seg_i2": rs[0]["seg_i2"],
                    "window_rows": rs[0]["window_rows"],
                    "median_mde_pp": (float(np.median([r["mde_pp"] for r in rs
                                                       if r["mde_pp"] is not None
                                                       and math.isfinite(r["mde_pp"])]))
                                      if any(math.isfinite(r["mde_pp"] or 0.0) for r in rs)
                                      else None),
                    "reachable_ROBUST": sum(1 for r in rs if r["reachable_verdict"] == "ROBUST"),
                    "n_verdict_given": sum(1 for r in rs if r.get("oos_verdict"))})
    return out


def md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_（无行）_"
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


TOP_COLS = ["pattern_id", "name_cn", "layer", "tf", "lens", "target", "oos_verdict",
            "oos_score", "holdout_n", "holdout_win_rate", "holdout_lift_pp",
            "oos_retention", "oos_temporal_consistency", "oos_regime_consistency",
            "oos_flipped", "oos_block_ci_low", "oos_block_ci_high", "oos_block_ci_b",
            "power_tag", "mde_pp",
            "econ_scope", "econ_tradable_L1_nominal_0.50", "econ_tradable_L4_measured_b0",
            "econ_ci_tradable_L4_measured_b0", "econ_tradable_L2_recorded_0.615",
            "econ_tradable_all_layers",
            "overlap_with"]

FDR_COLS = ["pattern_id", "mask_key", "name_cn", "layer", "tf", "lens", "target",
            "holdout_n", "holdout_win_rate", "holdout_lift_pp", "holdout_p_value",
            "holdout_ci_low", "oos_verdict", "oos_flipped", "power_tag",
            "econ_tradable_L1_nominal_0.50", "econ_tradable_all_layers", "overlap_with"]


def build_tables(ledger: dict, grid: list[dict], records: list[dict]) -> str:
    """机读证据表（供 docs/research/naked-k/REPORT.md 正文引用，不替代人写结论）。"""
    prim = [r for r in records if r["is_primary_lens"]]
    with_verdict = [r for r in prim if r.get("oos_verdict")]
    top = sorted(with_verdict, key=lambda r: -(r.get("oos_score") or 0.0))[:40]
    vdist: dict[str, Counter] = defaultdict(Counter)
    for r in with_verdict:
        vdist[r["tf"]][r["oos_verdict"]] += 1
    overl = [r for r in prim if r["window"] == "720d" and r["horizon"] == HORIZONS[r["tf"]][0]
             and r["overlap_with"]]
    lens_conf = [r for r in prim if str(r["lens_status"]).startswith("CONFLICT")]
    lines: list[str] = ["# 裸K验证机读证据表", "",
                        f"生成时间（UTC）：{ledger['generated_at_utc']}", "",
                        "## 1. 运行指纹与门控", "", md_table(
        [{"k": "run_fp", "v": ledger["run_fp"]},
         {"k": "causality_passed", "v": ledger["gates"]["causality_passed"]},
         {"k": "synthetic_passed", "v": ledger["gates"]["synthetic_passed"]},
         {"k": "abs_lens_verified", "v": ledger["gates"]["abs_lens_verified"]},
         {"k": "holdout_touches", "v": ledger["oos"].get("touches_after", "not-run")},
         {"k": "holdout_touches_all_time（本仓累计物理执行 Step 7 次数）", "v":
          ledger["oos"].get("touches_all_time", "not-run")},
         {"k": "registry_frozen_at", "v": ledger.get("registry_frozen_at")},
         {"k": "records_total", "v": len(records)},
         {"k": "records_primary_lens", "v": len(prim)}], ["k", "v"]), ""]
    lines += ["## 2. 9 格矩阵总览（3 周期 × 3 窗口；非 9 个独立样本）", "",
              md_table(grid, ["tf", "window", "window_rows", "seg_i1", "seg_i2", "masks",
                              "rows_total", "TESTABLE", "DESCRIPTIVE_ONLY", "UNTESTABLE",
                              "NO_HITS", "median_mde_pp", "reachable_ROBUST",
                              "n_verdict_given"]), ""]
    lines += ["## 3. 720d 主判决 Top 40（按 run_oos score）", "",
              md_table(top, TOP_COLS), ""]
    lines += ["### 3b. Top 10 的 walk-forward 逐折胜率", "",
              "_折边界由 `oos_validator.walk_forward` 在**全 720d 索引**上等分 8 段（≈90 天/折），_",
              "_故折 7/8 与 holdout 重叠：本表衡量估计量随时间的稳定性，**不是第 4 段独立复现**。_", "",
              md_table(top[:10], ["pattern_id", "name_cn", "tf", "lens", "target",
                                  "oos_verdict", "oos_temporal_consistency",
                                  "oos_walk_forward", "oos_walk_forward_n"]), ""]
    lines += ["## 4. 裁决分布（主 lens、720d）", "",
              md_table([dict(tf=tf, n_rows=sum(v.values()),
                             **{k: v.get(k, 0) for k in
                                ("ROBUST", "PROMISING", "WEAK", "REJECT")})
                        for tf, v in sorted(vdist.items(),
                                            key=lambda kv: TFS.index(kv[0]))],
                       ["tf", "ROBUST", "PROMISING", "WEAK", "REJECT", "n_rows"]), ""]
    prim720 = [r for r in prim if r["window"] == "720d"]
    fdr_fam = sorted([r for r in prim720 if r.get("holdout_fdr_pass")],
                     key=lambda r: -(r.get("holdout_lift_pp") or 0.0))
    fdr_glob = sorted([r for r in prim720 if r.get("holdout_fdr_pass_global")],
                      key=lambda r: -(r.get("holdout_lift_pp") or 0.0))
    n_glob_h = ledger["fdr"]["family_sizes"].get("GLOBAL/holdout")
    lines += ["## 5. 多重检验账本", "", md_table(
        [{"k": "fdr_alpha", "v": ledger["fdr"]["fdr_alpha"]},
         {"k": "n_families", "v": ledger["fdr"]["n_families"]},
         {"k": "n_pvalues_total", "v": ledger["fdr"]["n_pvalues_total"]},
         {"k": "n_pvalues_global_holdout", "v": n_glob_h},
         {"k": "n_fdr_pass_holdout", "v": ledger["fdr"]["n_fdr_pass_holdout"]},
         {"k": "n_fdr_pass_holdout_global", "v": ledger["fdr"]["n_fdr_pass_holdout_global"]},
         {"k": "n_with_oos_verdict", "v": len(with_verdict)}], ["k", "v"]), "",
        "### 5a. holdout 上过族内 BH-FDR（q=0.1）的行（主 lens × 720d）", "",
        md_table(fdr_fam, FDR_COLS), "",
        "### 5b. holdout 上过**全局** BH-FDR（q=0.1，"
        f"{n_glob_h} 个 holdout p 值同池）的行", "",
        "_全局族把三个周期、两个 lens、所有窗口、所有 horizon 的 holdout p 值汇在一池；_",
        "_这是最严的校正。若存活行的 lift 全为负，说明「形态的正向优势」经不起全量校正。_", "",
        md_table(fdr_glob, FDR_COLS), ""]
    lines += ["## 6. 副口径经济账计数", "",
              "_计数范围：主 lens × 720d × 5m/15m × horizon=1（holdout 只此一次判决）；_",
              "_`ci_tradable_*` 以 Wilson 下界代替点估计（更保守）。_", "", md_table(
        [{"k": k, "v": v} for k, v in sorted(ledger["economy"]["counters"].items())],
        ["k", "v"]), ""]
    lines += ["### 各报价层达到的形态数（holdout 口径、主 lens × 720d、5m/15m、horizon=1）", "",
              md_table(ledger["economy"]["layers"],
                       ["tag", "kind", "n_samples", "q", "p_star"]), ""]
    lines += ["## 7. 与线上实盘/已注册条目的重叠清单（非独立新证据）", "",
              md_table(overl, ["pattern_id", "mask_key", "name_cn", "layer", "tf", "target",
                               "overlap_with",
                               "holdout_n", "holdout_win_rate", "holdout_lift_pp",
                               "oos_verdict"]), ""]
    sens = ledger.get("sensitivity", {})
    flips = sens.get("flips", [])
    lines += ["## 8. 阈值 ±1 档敏感性：lift 符号翻转清单（只用 discovery/validation）", "",
              f"扰动行数：{sens.get('n_rows', 0)}（敏感性阶段已跳过）"
              if not sens else
              f"扰动行数：{sens.get('n_rows', 0)}；翻转事件数：{len(flips)}", "",
              md_table(flips[:120], ["tf", "mask_key", "target", "factor", "segment",
                                     "base_lift_pp", "var_lift_pp"]), ""]
    st_dist = Counter(str(r["lens_status"]).split("(")[0] for r in prim)
    lines += ["## 9. 双 lens 自洽性", "",
              "_主判决 lens 只由注册表声明（bet_side）决定，不因数据稀疏漂移；_",
              "_dir-pinned 等价判定带最小样本门槛 DIR_PIN_MIN_N=%d（低于此记 THIN，不作结论）。_" % eng.DIR_PIN_MIN_N,
              "", md_table(
        [{"k": "CONFLICT 数（声明与实证方向反号且样本达标）", "v": len(lens_conf)},
         {"k": "两 lens 等价（dir-pinned且 n_dir≥门槛）行占比", "v": round(
             sum(1 for r in prim if r["lenses_equivalent"]) / max(1, len(prim)), 4)},
         {"k": "稀疏伪钉死（thin_pin）行占比", "v": round(
             sum(1 for r in prim if r.get("thin_pin")) / max(1, len(prim)), 4)},
         {"k": "THIN_DISAGREE 数（小样本反号，不阻断）", "v": st_dist.get("THIN_DISAGREE", 0)},
         {"k": "lens_status 分布", "v": json.dumps(dict(st_dist), ensure_ascii=False)},
         {"k": "conflict_keys", "v": ", ".join(sorted({r["mask_key"] for r in lens_conf}))}],
        ["k", "v"]), ""]
    return "\n".join(lines) + "\n"


# ============================ pattern_catalog.csv 导出（A-3） ============================

_INT_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+)(?![A-Za-z0-9_.])")
_IDENT_TAIL_RE = re.compile(r"[A-Za-z_][A-Za-z_]*?(\d+)")


def lookback_hint(expr: str) -> int:
    """从判据文本提取「最多往回看多少根」的机械提示（非人工保证，仅供阅读）。"""
    nums = [int(x) for x in _INT_RE.findall(expr)]
    nums += [int(x) for x in _IDENT_TAIL_RE.findall(expr)]
    return max(nums) if nums else 0


CATALOG_COLS = ("id", "name_cn", "layer", "family", "direction_bet", "target_family",
                "bet_side", "criterion_expr", "context_gate", "lookback_hint_bars",
                "n_expressions", "tier", "status", "overlap_with", "market_caliber",
                "freq_prior", "sources", "mechanism", "masks_5m", "n_masks_5m",
                "dedup_aliases_5m", "hit_sig_5m")


def export_catalog(raw: dict, masks_ref: list[dict], path: str) -> dict:
    """由**冻结注册表**导出形态登记表（唯一口径源，不手工维护第二份定义）。"""
    per: dict[str, list[dict]] = defaultdict(list)
    for m in masks_ref:
        per[m["pattern_id"]].append(m)
    rows = []
    for p in raw["hypotheses"]:
        ms = per.get(p["id"], [])
        aliases = [a for m in ms for a in m["aliases"]]
        rows.append({
            "id": p["id"], "name_cn": p.get("name_cn"), "layer": p.get("layer"),
            "family": p.get("family"),
            "direction_bet": ("押收阳" if p.get("bet_side") == "up" else
                              "押收阴" if p.get("bet_side") == "down" else
                              ("押延续" if p.get("target_family") == "continuation"
                               else "押反转")),
            "target_family": p.get("target_family"), "bet_side": p.get("bet_side"),
            "criterion_expr": p.get("expr"),
            "context_gate": "|".join(p.get("context_gate") or []),
            "lookback_hint_bars": lookback_hint(
                " & ".join([p.get("expr") or ""] + list(p.get("context_gate") or []))),
            "n_expressions": len(ms), "tier": p.get("tier"), "status": p.get("status"),
            "overlap_with": p.get("overlap_with") or "",
            "market_caliber": p.get("market_caliber"), "freq_prior": p.get("freq_prior"),
            "sources": "|".join(p.get("sources") or []), "mechanism": p.get("mechanism"),
            "masks_5m": "|".join(m["mask_key"] for m in ms), "n_masks_5m": len(ms),
            "dedup_aliases_5m": "|".join(aliases),
            "hit_sig_5m": "|".join(m["hit_sig"] for m in ms),
        })
    info = write_csv(path, rows, CATALOG_COLS)
    info["unverified_note"] = ("本表只包含可机检化判据；UNSPECIFIED 术语及其不纳入理由"
                              "列于同目录 sources.md 的「不可验证清单」。")
    return info


# ============================ 主流程 ============================

def load_registry() -> tuple[list[dict], dict]:
    with open(prep.REGISTRY, encoding="utf-8") as f:
        raw = json.load(f)
    return raw["hypotheses"], raw


def main() -> int:
    ap = argparse.ArgumentParser(description="裸K形态胜率验证（Step 4~9）")
    ap.add_argument("--stage", choices=("all", "stats"), default="all",
                    help="stats = 只跑 Step 4/5/6/8 + 敏感性，不碰 holdout（不跑 run_oos）")
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--allow-retouch", action="store_true",
                    help="允许重复触碰冻结 holdout（仅因输入变更重跑时；会累计记录）")
    ap.add_argument("--export-catalog", action="store_true",
                    help="由冻结注册表导出 docs/research/naked-k/pattern_catalog.csv 后退出")
    ap.add_argument("--skip-selfcheck", action="store_true",
                    help="仅调试用：跳过因果自检（产物会标记，结论不可用于判决）")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()

    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    bad = [t for t in tfs if t not in TFS]
    if bad:
        raise SystemExit(f"[FAIL] 未知周期：{bad}（可选 {list(TFS)}）")

    run_fp, base, out_dir = resolve_outdir()
    reg, raw = load_registry()
    reg_index = {p["id"]: p for p in reg}
    print(f"[run_fp] {run_fp}  →  {os.path.relpath(out_dir, ROOT)}")
    parents = eng.build_parents(prep.CSV_5M)

    if args.export_catalog:
        kl5 = load_klines_csv(prep.CSV_5M, BAR_MS["5m"])
        ns5 = eng.build_namespace(kl5, BAR_MS["5m"], parents)
        m5 = eng.expand_masks(reg, ns5, int(len(kl5) * DEFAULTS["discovery_frac"]))
        os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
        info = export_catalog(raw, m5, CATALOG_PATH)
        print("[catalog]", json.dumps(info, ensure_ascii=False))
        return 0

    synth = synthetic_target_check()
    if not synth["passed"]:
        raise SystemExit(f"[FAIL] 合成 K 线手算对照失败：{synth['failed']}")

    ledger: dict = {
        "run_fp": run_fp, "generated_at_utc": now_utc(), "stage": args.stage,
        "tfs": tfs, "registry_frozen_at": raw.get("frozen_at"),
        "registry_sha256": base.get("registry_sha256"),
        "baseline_digests": base.get("digests"),
        "config": {"BAR_MS": BAR_MS, "HORIZONS": HORIZONS, "MDE_TESTABLE_PP": MDE_TESTABLE_PP,
                   "BLOCK_CI_ELEMS": BLOCK_CI_ELEMS, "BLOCK_CI_B_MIN": BLOCK_CI_B_MIN,
                   "BLOCK_CI_B_MAX": BLOCK_CI_B_MAX, "SENS_FACTORS": list(SENS_FACTORS),
                   "DEFAULTS": {k: san(v) for k, v in DEFAULTS.items()},
                   "FEE": FEE, "PREMIUM": PREMIUM, "quote_layers": list(QUOTE_LAYERS)},
        "gates": {"synthetic_passed": synth["passed"], "selfcheck_skipped": args.skip_selfcheck,
                  "causality_passed": None, "abs_lens_verified": None},
        "per_tf": {}, "oos": {"ran": False}, "sensitivity": {"n_rows": 0, "flips": []},
    }

    # ---- Step 3~6：逐周期（因果自检是 Step 6 的前置门）----
    records: list[dict] = []
    series: dict[str, dict] = {}
    causality: dict[str, dict] = {}
    for tf in tfs:
        t_start = time.time()
        kl, diag = build_series(tf, out_dir, parents)
        if not args.skip_selfcheck:
            cs = eng.causality_selfcheck(kl, BAR_MS[tf], parents, reg)
            causality[tf] = cs
            if not cs["passed"]:
                raise SystemExit(f"[FAIL] {tf} 因果自检泄漏，中止：{cs['leaked'][:5]}")
        ns = eng.build_namespace(kl, BAR_MS[tf], parents)
        n = len(kl)
        i1 = int(n * DEFAULTS["discovery_frac"])
        masks = eng.expand_masks(reg, ns, i1)
        conflicts = attach_lens(ns, masks)
        tg = build_targets(kl.t, kl.o, kl.h, kl.l, kl.c, kl.cont, HORIZONS[tf],
                           ns["atr_abs"])
        derive_abs_targets(tg, ns["dir_"], HORIZONS[tf])
        av = verify_abs_lens(kl, tg, HORIZONS[tf])
        if not av["passed"]:
            raise SystemExit(f"[FAIL] {tf} 绝对 lens 独立重算不吻合：{av['errors'][:5]}")
        recs, sinfo = stage_stats(tf, kl, ns, tg, masks, reg_index)
        records.extend(recs)
        series[tf] = {"kl": kl, "ns": ns, "tg": tg, "masks": masks}
        ledger["per_tf"][tf] = {**diag, "n_masks": len(masks),
                                "n_exprs_before_dedup": len(masks) + sum(
                                    len(m["aliases"]) for m in masks),
                                "n_aliases_merged": sum(len(m["aliases"]) for m in masks),
                                "n_unique_hit_sig": len({m["hit_sig"] for m in masks}),
                                "lens_conflicts": conflicts, "abs_verify": av,
                                "causality": {k: v for k, v in causality.get(tf, {}).items()
                                              if k != "leaked"},
                                "stats": sinfo}
        print(f"[{tf}] rows={diag['rows']} masks={len(masks)} records={len(recs)} "
              f"conflicts={len(conflicts)} {round(time.time() - t_start, 1)}s")
    ledger["gates"]["causality_passed"] = ("skipped" if args.skip_selfcheck else
                                           all(c["passed"] for c in causality.values()))
    ledger["gates"]["abs_lens_verified"] = True
    write_json(os.path.join(out_dir, "causality_selfcheck.json"),
               {"passed": ledger["gates"]["causality_passed"],
                "synthetic_target_check": synth, "per_tf": causality,
                "note": ("扰动末 200 根 OHLCV 后，t<n-200 处所有掩码逐位不变；"
                         "另外逐周期验证 dirup/dirdn 与次根 open→close 符号独立重算一致。")})

    # ---- 全局 FDR ----
    ledger["fdr"] = apply_fdr(records)

    # ---- Step 8：副口径经济账 ----
    qs = quote_surface()
    write_json(os.path.join(out_dir, "quote_surface.json"), qs)
    ledger["quote_summary"] = qs["summary"]
    ledger["economy"] = stage_economy(records, qs)

    # ---- 阈值敏感性 ----
    if not args.skip_sensitivity:
        sens_rows, all_flips = [], []
        for tf in tfs:
            s = series[tf]
            rows_tf, info = stage_sensitivity(tf, s["kl"], s["ns"], s["masks"], s["tg"],
                                              reg_index)
            sens_rows.extend(rows_tf)
            all_flips.extend(info["flips"])
            info.pop("flips")
            ledger.setdefault("sensitivity_by_tf", {})[tf] = info
        ledger["sensitivity"] = {"n_rows": len(sens_rows), "n_flips": len(all_flips),
                                 "flips": all_flips}
        write_csv(os.path.join(out_dir, "sensitivity.csv"), sens_rows,
                  ("tf", "pattern_id", "mask_key", "name_cn", "layer", "target", "lens",
                   "horizon", "factor", "n_thresholds", "n_clamped", "n_hits_all",
                   "discovery_n", "discovery_win_rate", "discovery_lift_pp",
                   "validation_n", "validation_win_rate", "validation_lift_pp",
                   "flip_discovery", "flip_validation", "n_change_ratio",
                   "holdout_reported", "expr_variant"))
        print(f"[sensitivity] rows={len(sens_rows)} flips={len(all_flips)}")

    # ---- Step 7：run_oos 终验（只 720d、只主 lens、holdout 只碰一次）----
    if args.stage == "all":
        state, flag_path = holdout_flag_state(out_dir, args.allow_retouch)
        detail = {"touched_at_utc": now_utc(), "run_fp": run_fp, "tfs": tfs,
                  "status": "in-progress", "allow_retouch": args.allow_retouch}
        write_holdout_flag(flag_path, state, detail)      # 先落盘，崩溃也不给静默重跑
        touch_seq = record_holdout_touch(run_fp, tfs)     # 跨目录总账（不随 run_fp 重置）
        rec_index = {(r["tf"], r["mask_key"], r["window"], r["lens"], r["horizon"]): r
                     for r in records if r["is_primary_lens"] and r["window"] == "720d"}
        flat_all: list[dict] = []
        audits: list[dict] = []
        oos_store: dict[str, dict] = {}
        for tf in tfs:
            s = series[tf]
            flat, info = stage_oos(tf, s["kl"], s["tg"], s["ns"], s["masks"], rec_index)
            au, minfo = merge_oos_into_records(flat, rec_index)
            info.update(minfo)
            audits.extend(au)
            flat_all.extend(flat)
            oos_store[tf] = {"info": info, "rows": san(flat)}
            ledger.setdefault("oos_skipped", {})[tf] = info["skipped"]
            info.pop("skipped", None)
            ledger.setdefault("oos_by_tf", {})[tf] = info
            print(f"[oos:{tf}] {json.dumps({k: v for k, v in info.items() if k != 'skipped'}, ensure_ascii=False)}")
        state2, _ = holdout_flag_state(out_dir, allow_retouch=True)
        state2["history"][-1].update({"status": "completed", "n_oos_rows": len(flat_all),
                                      "n_audit_issues": len(audits),
                                      "finished_at_utc": now_utc()})
        with open(flag_path, "w", encoding="utf-8") as f:
            json.dump(state2, f, ensure_ascii=False, indent=2)
        ledger["oos"] = {"ran": True, "n_rows": len(flat_all), "scope": "720d × primary lens",
                         "touches_after": state2["touches"],
                         "touches_all_time": touch_seq,
                         "crosscheck_issues": audits[:200],
                         "n_crosscheck_issues": len(audits),
                         # run_oos 原始行的裁决字段名是 "verdict"（写入 record 时才加 oos_ 前缀）
                         "verdict_dist": dict(Counter(str(r.get("verdict")) for r in flat_all)),
                         "verdict_dist_by_tf": {
                             tf: dict(Counter(str(r.get("verdict")) for r in oos_store[tf]["rows"]))
                             for tf in oos_store}}
        write_json(os.path.join(out_dir, "oos_720d.json"),
                   {"meta": {"run_fp": run_fp, "generated_at_utc": now_utc(),
                             "note": ("row 内含 walk_forward/monthly_detail/regime_detail；"
                                      "run_block_ci 已被降 b（见 block_ci_b）。")},
                    "by_tf": oos_store})

    # ---- Step 9：产物落盘 ----
    grid = grid_rows(records)
    for r in records:
        r.pop("oos_detail", None)
        # 元数据随行落盘：任何一行都能反查它出自哪次冻结输入 + 因果自检是否通过
        r["run_fp"] = run_fp
        r["registry_sha256"] = ledger.get("registry_sha256")
        r["causality_passed"] = ledger["gates"]["causality_passed"]
        r["abs_lens_verified"] = ledger["gates"]["abs_lens_verified"]
    artifacts = [write_csv(os.path.join(out_dir, "findings.csv"), records, HEAD_COLS)]
    write_json(os.path.join(out_dir, "findings.json"),
               {"run_fp": run_fp, "generated_at_utc": now_utc(),
                "ledger": {k: v for k, v in ledger.items() if k != "sensitivity"},
                "records": records})
    preflight_rows = [{k: r.get(k) for k in
                       ("tf", "window", "mask_key", "pattern_id", "name_cn", "layer", "lens",
                        "target", "horizon", "power_tag", "power_note", "mde_pp",
                        "reachable_verdict", "n_valid_holdout", "min_hit_rate_for_robust",
                        "holdout_n", "discovery_n", "validation_n", "n_dir_pos", "n_dir_neg",
                        "dir_pinned", "thin_pin", "n_dir", "lens_status",
                        "lenses_equivalent", "overlap_with")}
                      for r in records if r["is_primary_lens"]]
    artifacts.append(write_csv(os.path.join(out_dir, "power_preflight.csv"), preflight_rows,
                               ("tf", "window", "mask_key", "pattern_id", "name_cn",
                                "target", "power_tag", "mde_pp", "reachable_verdict")))
    tables = build_tables(ledger, grid, records)
    tpath = os.path.join(out_dir, "report_tables.md")
    with open(tpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(tables)
    ledger["artifacts"] = artifacts + [
        {"path": os.path.relpath(tpath, ROOT).replace("\\", "/"),
         "bytes": len(tables.encode())},
        {"path": os.path.relpath(os.path.join(out_dir, "findings.json"), ROOT)
            .replace("\\", "/")},
        {"path": os.path.relpath(os.path.join(out_dir, "quote_surface.json"), ROOT)
            .replace("\\", "/")},
        {"path": os.path.relpath(os.path.join(out_dir, "causality_selfcheck.json"), ROOT)
            .replace("\\", "/")},
        {"path": os.path.relpath(os.path.join(out_dir, "power_preflight.csv"), ROOT)
            .replace("\\", "/")}]
    write_json(os.path.join(out_dir, "ledger.json"), ledger)
    print("\n[grid]")
    for g in grid:
        print("  ", json.dumps(san(g), ensure_ascii=False))
    print(f"\n[OK] 产物 → {os.path.relpath(out_dir, ROOT)}")
    print(f"     待手写报告：{os.path.relpath(DOCS_REPORT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

