#!/usr/bin/env python3
"""情绪曲线科学发现：5m / 15m 周期中 K 线延续·反转确定性场景（L1→L2/L3→OOS→经济账）。

研究对象（第一性原理）：
  情绪曲线 = 预测市场参与者对「BTC 未来 5min 涨跌」的实时共识（15s 采样）。
  它是价格的前瞻代理：当情绪领先/滞后于价格、或情绪达到极端时，
  K 线的延续或反转概率可能系统性偏离 50%。

研究纪律（由简入繁 + 样本外验证）：
  1. 单一时间切分：前 70% = 发现集（L1/L2/L3 只在此做），后 30% = OOS（仅终验）。
  2. L1 单因子：每个假设在发现集上检验 n / 胜率 / Wilson CI / 双侧 p；
     方向族与延续族分别过 BH-FDR(q=0.1)。
     入场券：n≥150(5m)/60(15m) 且 |偏离基准|≥1.5pp 且 p<0.05。
  3. L2 双因子 / L3 三因子：入场券因子组合，发现集内搜高胜率场景。
  4. 终验：OOS 方向一致 + n≥40 + 合并 Wilson 下界>基准。
  5. 经济账（唯一诚实口径）：5m 场景按决策点（次窗开窗 150s）真实 token 价入场，
     费 2%+溢价0.01，EV bootstrap 95% CI 下界>0 才算可交易；15m 场景给 @0.50 口径。

防泄漏设计：
  - 跨窗场景（预测次窗）：特征只用当前窗完整曲线 + 已结束的历史窗，决策点=次窗+150s。
  - 窗内场景（早读）：特征只用当前窗前 150s 截断视图，预测整窗结算。
  - 滚动分位阈值只在发现集冻结，OOS 复用。

用法：
    python scripts/local_sentiment_curve_discovery.py [--from-file sentiment_windows.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

from binance_predict.services.verification import bh_fdr, wilson_bounds  # noqa: E402

DECISION_T_SEC = 150.0   # 决策点：开窗后 150s（与宪法 V1.1 对齐）
HOLDOUT_RATIO = 0.3
FEE, PREM = 0.02, 0.01   # 一票否决成本口径
BOOT_N, BOOT_SEED = 2000, 7
MIN_N_5M, MIN_N_15M = 150, 60          # L1 入场券最小样本
MIN_DEV = 0.015                        # L1 入场券最小偏离（绝对 pp）
OOS_MIN_N = 40                         # 终验最小 OOS 样本
P_SIG = 0.05                           # L1 名义显著性（未校正）


# ============================================================
# 数据加载与窗口级特征
# ============================================================

def curve_arrays(w: dict, key: str = "curve_up_pct", t_sec: float | None = None):
    """曲线 → (vals, rel_sec)，按时间排序；t_sec 截断（防未来）。样本<2 返回 None。"""
    pts = sorted(w.get(key) or [], key=lambda p: p.get("t", 0))
    if t_sec is not None:
        t0 = pts[0]["t"] if pts else 0
        pts = [p for p in pts if (p["t"] - t0) / 1000.0 <= t_sec]
    if len(pts) < 2:
        return None
    t0 = pts[0]["t"]
    v = np.array([float(p["v"]) for p in pts])
    t = np.array([(p["t"] - t0) / 1000.0 for p in pts])
    return v, t


def _line_feats(v: np.ndarray, t: np.ndarray) -> dict:
    """单条曲线的几何特征（水平/动量/形态），样本<4 时形态项置 NaN。"""
    n = len(v)
    f: dict = {}
    f["start"], f["end"] = float(v[0]), float(v[-1])
    f["mean"], f["max"], f["min"] = float(v.mean()), float(v.max()), float(v.min())
    f["delta"] = f["end"] - f["start"]
    f["range"] = f["max"] - f["min"]
    f["n"] = n
    if n >= 4:
        f["slope"] = float(np.polyfit(t, v, 1)[0] * 300.0)  # pp / 300s
        d = np.diff(v)
        ds = d[np.abs(d) > 0.5]
        f["chop"] = int(np.sum(np.sign(ds[1:] * ds[:-1]) < 0)) if len(ds) >= 2 else 0
        f["mono_up"] = bool(np.all(d >= -1.0))
        f["mono_dn"] = bool(np.all(d <= 1.0))
        half = n // 2
        f["peak_pos"] = float(np.argmax(v) / (n - 1))
        f["trough_pos"] = float(np.argmin(v) / (n - 1))
        f["v_shape"] = bool(f["trough_pos"] < 0.5 and f["delta"] > 3
                            and f["min"] < f["start"] - 3)
        f["inv_v"] = bool(f["peak_pos"] < 0.5 and f["delta"] < -3
                          and f["max"] > f["start"] + 3)
        f["close_hi"] = bool(f["end"] >= f["max"] - 2.0)
        f["close_lo"] = bool(f["end"] <= f["min"] + 2.0)
    else:
        for k in ("slope", "chop", "peak_pos", "trough_pos"):
            f[k] = float("nan")
        for k in ("mono_up", "mono_dn", "v_shape", "inv_v", "close_hi", "close_lo"):
            f[k] = False
    return f


def build_records(W: list[dict]) -> list[dict]:
    """逐窗构建：全窗特征 f_* / 截断特征 t_* / 上下文 ctx_* / 目标 next_*。"""
    recs: list[dict] = []
    ret_hist: list[float] = []  # |ret| 滚动窗（过去 48 窗 = 4h）
    for i, w in enumerate(W):
        out = (w.get("outcome") or "").upper()
        ret = float(w.get("actual_return") or 0.0)
        start = int(w["start_time"])
        gt = time.gmtime(start / 1000)
        r: dict = {"i": i, "start": start, "out": out, "ret": ret,
                   "hour": gt.tm_hour, "wd": gt.tm_wday,
                   "grid": (start % 900_000) // 300_000,
                   "has_next": False, "next_out": None, "next_same": None,
                   "prev_ok": False}

        ca = curve_arrays(w, "curve_up_pct")
        if ca:
            r["f"] = _line_feats(*ca)
        ca_t = curve_arrays(w, "curve_up_pct", DECISION_T_SEC)
        r["t"] = _line_feats(*ca_t) if ca_t and len(ca_t[0]) >= 4 else None

        # 上下文（只用已结束窗口）
        if i > 0:
            p = W[i - 1]
            r["prev_ok"] = (start - int(p["start_time"])) == 300_000
            r["prev_out"] = (p.get("outcome") or "").upper()
            r["prev_end_s"] = float(sorted(p["curve_up_pct"], key=lambda x: x["t"])[-1]["v"]) \
                if p.get("curve_up_pct") else float("nan")
            if i > 1:
                r["prev2_out"] = (W[i - 2].get("outcome") or "").upper()
        # 连击（含当前窗，需前窗连续）
        su = 1 if out == "UP" else 0
        sd = 1 if out == "DOWN" else 0
        if r.get("prev_ok") and r.get("prev_out") == "UP":
            su += recs[-1].get("streak_up", 0)
        if r.get("prev_ok") and r.get("prev_out") == "DOWN":
            sd += recs[-1].get("streak_dn", 0)
        r["streak_up"], r["streak_dn"] = su, sd
        med = float(np.median(ret_hist[-48:])) if len(ret_hist) >= 24 else float("nan")
        r["ret_med48"] = med
        r["ret_ratio"] = abs(ret) / med if med and med > 0 else float("nan")
        ret_hist.append(abs(ret))
        recs.append(r)

    # 目标（次窗），跨数据缺口失效
    for i, r in enumerate(recs[:-1]):
        nxt = recs[i + 1]
        if (nxt["start"] - r["start"]) == 300_000 and r["out"] in ("UP", "DOWN") \
                and nxt["out"] in ("UP", "DOWN"):
            r["has_next"] = True
            r["next_out"] = nxt["out"]
            r["next_same"] = nxt["out"] == r["out"]
    return recs


# ============================================================
# 15m 周期构建
# ============================================================

def build_cycles(W: list[dict], recs: list[dict]) -> list[dict]:
    """完整 3 连窗聚合为 15m 周期；特征只用周期内已完成数据。"""
    by_start = {int(w["start_time"]): w for w in W}
    rec_by_start = {r["start"]: r for r in recs}
    buckets: dict[int, list[int]] = {}
    for s in sorted(by_start):
        buckets.setdefault(s // 900_000, []).append(s)
    cycles: list[dict] = []
    for cyc, starts in sorted(buckets.items()):
        if len(starts) != 3 or starts[1] - starts[0] != 300_000 or starts[2] - starts[1] != 300_000:
            continue
        ws = [by_start[s] for s in starts]
        vals = []
        for w in ws:
            pts = sorted(w["curve_up_pct"], key=lambda p: p["t"])
            vals.extend(float(p["v"]) for p in pts)
        v = np.array(vals)
        s_start, s_end = float(v[0]), float(v[-1])
        entry, exit_ = float(ws[0]["entry_price"]), float(ws[2]["exit_price"])
        ret_c = exit_ / entry - 1.0
        out_c = "UP" if exit_ > entry else ("DOWN" if exit_ < entry else None)
        rs = [rec_by_start[s] for s in starts]
        wd_ends = [r["f"]["end"] for r in rs if r.get("f")]
        d_win = [wd_ends[k + 1] - wd_ends[k] for k in range(2)] if len(wd_ends) == 3 else []
        gt = time.gmtime(starts[0] / 1000)
        cycles.append({
            "cyc": cyc, "start": starts[0], "out": out_c, "ret": ret_c,
            "s_start": s_start, "s_end": s_end, "s_delta": s_end - s_start,
            "s_max": float(v.max()), "s_min": float(v.min()),
            "peak_pos": float(np.argmax(v) / (len(v) - 1)),
            "d_win": d_win, "path3": "".join("U" if r["out"] == "UP" else
                                             ("D" if r["out"] == "DOWN" else "F") for r in rs),
            "net_up": sum(1 for r in rs if r["out"] == "UP"),
            "hour": gt.tm_hour, "wd": gt.tm_wday,
            "has_next": False, "next_out": None, "next_same": None,
        })
    for j, c in enumerate(cycles[:-1]):
        if cycles[j + 1]["cyc"] == c["cyc"] + 1 and c["out"] and cycles[j + 1]["out"]:
            c["has_next"] = True
            c["next_out"] = cycles[j + 1]["out"]
            c["next_same"] = cycles[j + 1]["out"] == c["out"]
    # 周期连击
    for j, c in enumerate(cycles):
        c["streak_up"] = (1 if c["out"] == "UP" else 0) + (
            cycles[j - 1].get("streak_up", 0)
            if j > 0 and cycles[j - 1]["out"] == "UP" and c["out"] == "UP" else 0)
        c["streak_dn"] = (1 if c["out"] == "DOWN" else 0) + (
            cycles[j - 1].get("streak_dn", 0)
            if j > 0 and cycles[j - 1]["out"] == "DOWN" and c["out"] == "DOWN" else 0)
    return cycles


# ============================================================
# 统计与经济账
# ============================================================

def binom_p(k: int, n: int, p0: float) -> float:
    """双侧正态近似二项检验（H0: 真实率 = p0）。"""
    if n <= 0:
        return 1.0
    z = (k / n - p0) / math.sqrt(p0 * (1 - p0) / n)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def eval_mask(mask: np.ndarray, target: np.ndarray, base: float,
              pool: np.ndarray | None = None) -> dict | None:
    """mask 在 pool 上命中 → 目标胜率统计。target 为 0/1 数组。"""
    m = mask & np.isfinite(target.astype(float))
    if pool is not None:
        m = m & pool
    n = int(m.sum())
    if n == 0:
        return None
    k = int(target[m].sum())
    p = k / n
    lo, hi = wilson_bounds(k, n)
    return {"n": n, "k": k, "p": p, "ci": (lo, hi), "dev": p - base,
            "pval": binom_p(k, n, base)}


def price_at(curve: list | None, start_ms: int, t_sec: float) -> float | None:
    best = None
    for pnt in sorted(curve or [], key=lambda x: x.get("t", 0)):
        if (pnt.get("t", 0) - start_ms) / 1000.0 <= t_sec and pnt.get("v") is not None:
            best = float(pnt["v"])
    return best


def entry_quote(w_next: dict, direction: str) -> tuple[float | None, str]:
    """决策点（次窗+150s）入场价：真实 token 价优先，缺失回退 chance/100。"""
    key = "curve_up_price" if direction == "UP" else "curve_down_price"
    p = price_at(w_next.get(key), int(w_next["start_time"]), DECISION_T_SEC)
    if p is not None and p > 0:
        return p, "real"
    c = price_at(w_next.get("curve_up_pct" if direction == "UP" else "curve_down_pct"),
                 int(w_next["start_time"]), DECISION_T_SEC)
    if c is None or c <= 0:
        return None, "missing"
    return c / 100.0, "proxy"


def ev_eval(idx_list: list[int], dirs: list[str], W: list[dict]) -> dict:
    """逐注经济账：idx=目标窗索引（5m 场景=次窗，窗内场景=当前窗），dirs=下单方向。"""
    fires: list[tuple[bool, float, str]] = []
    for wi, d in zip(idx_list, dirs):
        w = W[wi]
        ret = w.get("actual_return")
        if ret is None or float(ret) == 0.0:
            continue
        price, kind = entry_quote(w, d)
        if price is None:
            continue
        fires.append((d == ("UP" if float(ret) > 0 else "DOWN"), price, kind))
    out: dict = {"n": len(fires), "n_real": sum(1 for _, _, k in fires if k == "real")}
    if not fires:
        return out
    wins = [w for w, _, _ in fires]
    out["win_rate"] = sum(wins) / len(wins)
    out["avg_price"] = float(np.mean([p for _, p, _ in fires]))
    for fee, prem in ((0.0, 0.0), (FEE, PREM)):
        arr = np.array([(1.0 - fee) / min(max(p + prem, 0.01), 0.99) - 1.0 if w else -1.0
                        for w, p, _ in fires])
        key = f"ev_{int(fee*100)}_{int(prem*100)}"
        out[key] = float(arr.mean())
        if len(arr) >= 10:
            rng = np.random.default_rng(BOOT_SEED)
            bs = rng.integers(0, len(arr), size=(BOOT_N, len(arr)))
            lo, hi = np.percentile(arr[bs].mean(axis=1), [2.5, 97.5])
            out[key + "_ci"] = [float(lo), float(hi)]
    return out


# ============================================================
# 假设注册表：L1 单因子（机制注释 = 第一性原理）
# ============================================================

def register_hypotheses(recs, cycles, split_ts, W):
    """返回 [(name, scale, mechanism, mask_fn, trade_dir)]。

    mask_fn(recs_or_cycles_array) → bool 数组；trade_dir：
      'UP'/'DOWN' = 固定方向注；'SAME' = 顺当前方向（延续注）；'FLIP' = 逆当前方向（反转注）。
    """
    R = np.array(recs, dtype=object)
    C = np.array(cycles, dtype=object)
    # 发现集冻结阈值（防泄漏：OOS 复用）
    big_ret = float(np.nanquantile([abs(r["ret"]) for r in recs
                                    if r["start"] < split_ts], 0.8))
    cyc_big = float(np.quantile([abs(c["ret"]) for c in cycles
                                 if c["start"] < split_ts], 0.8))
    H: list[tuple] = []

    def h5(name, mech, fn, td):
        H.append((name, "5m", mech, lambda R, fn=fn: np.array([bool(fn(r)) for r in R]), td))

    def h15(name, mech, fn, td):
        H.append((name, "15m", mech, lambda C, fn=fn: np.array([bool(fn(c)) for c in C]), td))

    def g(r, path, default=np.nan):
        """安全取嵌套特征。"""
        cur = r
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur or cur[k] is None:
                return default
            cur = cur[k]
        return cur

    nan_ok = lambda x: x == x  # noqa: E731  非 NaN

    # ---- D1 情绪水平（共识强度：共识是否即定局）----
    h5("L1_end≥70", "强看多共识 → 共识自我实现", lambda r: g(r, "f.end") >= 70, "UP")
    h5("L1_end≤30", "强看空共识", lambda r: g(r, "f.end") <= 30, "DOWN")
    h5("L1_mean≥65", "全窗高共识", lambda r: g(r, "f.mean") >= 65, "UP")
    h5("L1_mean≤35", "全窗低共识", lambda r: g(r, "f.mean") <= 35, "DOWN")
    h5("L1_touch90", "触及≥90 超买区（情绪极端→泄力）", lambda r: g(r, "f.max") >= 90, "FLIP")
    h5("L1_touch10", "触及≤10 超卖区", lambda r: g(r, "f.min") <= 10, "FLIP")
    h5("L1_end40-60", "无共识（对照，应为零边际）", lambda r: 40 <= g(r, "f.end") <= 60, "SAME")
    h5("L1_end70+touch90", "高位二次确认（收尾仍≥70 且曾≥90）",
       lambda r: g(r, "f.end") >= 70 and g(r, "f.max") >= 90, "UP")

    # ---- D2 情绪动量（趋势外推：人群惯性）----
    h5("M1_Δ≥15", "情绪涨≥15pp → 惯性延续", lambda r: g(r, "f.delta") >= 15, "UP")
    h5("M2_Δ≤-15", "情绪跌≥15pp", lambda r: g(r, "f.delta") <= -15, "DOWN")
    h5("M3_slope_hi", "斜率前 20% 分位（陡涨）",
       lambda r: nan_ok(g(r, "f.slope")) and g(r, "f.slope") >= 12, "UP")
    h5("M4_slope_lo", "斜率后 20%（陡跌）", lambda r: nan_ok(g(r, "f.slope")) and g(r, "f.slope") <= -12, "DOWN")
    h5("M5_mono_up", "近单调上行（全程回撤≤1pp）", lambda r: bool(g(r, "f.mono_up", False)), "UP")
    h5("M6_mono_dn", "近单调下行", lambda r: bool(g(r, "f.mono_dn", False)), "DOWN")
    h5("M7_v_shape", "V 形（探底回升）→ 次窗延续回升", lambda r: bool(g(r, "f.v_shape", False)), "UP")
    h5("M8_inv_v", "倒 V（冲高回落）", lambda r: bool(g(r, "f.inv_v", False)), "DOWN")

    # ---- D3 收尾形态（尾盘信息含量最高）----
    h5("P1_close_hi", "收在最高区（强尾盘）", lambda r: bool(g(r, "f.close_hi", False)), "UP")
    h5("P2_close_lo", "收在最低区（弱尾盘）", lambda r: bool(g(r, "f.close_lo", False)), "DOWN")
    h5("P3_peak_late", "峰在末 1/3 且回落≥5（冲高回落）",
       lambda r: nan_ok(g(r, "f.peak_pos")) and g(r, "f.peak_pos") >= 2 / 3
       and g(r, "f.end") <= g(r, "f.max") - 5, "DOWN")
    h5("P4_trough_late", "谷在末 1/3 且反抽≥5", lambda r: nan_ok(g(r, "f.trough_pos"))
       and g(r, "f.trough_pos") >= 2 / 3 and g(r, "f.end") >= g(r, "f.min") + 5, "UP")
    h5("P5_flat", "死水窗（range≤5pp）→ 无信息延续", lambda r: g(r, "f.range") <= 5, "SAME")
    h5("P6_choppy", "高噪声（≥4 次方向翻转）→ 分歧大反转",
       lambda r: g(r, "f.chop", 0) >= 4, "FLIP")

    # ---- D4 情绪×结果一致性（共识兑现 / 背离修正）----
    h5("X1_consist_up", "共识兑现（收阳且情绪≥60）→ 动量延续",
       lambda r: r["out"] == "UP" and g(r, "f.end") >= 60, "SAME")
    h5("X2_consist_dn", "共识兑现（收阴且情绪≤40）", lambda r: r["out"] == "DOWN" and g(r, "f.end") <= 40, "SAME")
    h5("X3_div_up", "背离（收阴但情绪≥60）→ 次窗修正上行",
       lambda r: r["out"] == "DOWN" and g(r, "f.end") >= 60, "UP")
    h5("X4_div_dn", "背离（收阳但情绪≤40）→ 次窗修正下行",
       lambda r: r["out"] == "UP" and g(r, "f.end") <= 40, "DOWN")
    h5("X5_cross_up", "情绪跨界 50↓→↑（空头回补启动）",
       lambda r: g(r, "f.start") < 50 <= g(r, "f.end") and g(r, "f.delta") >= 8, "UP")
    h5("X6_cross_dn", "情绪跨界 50↑→↓（多杀多启动）",
       lambda r: g(r, "f.start") > 50 >= g(r, "f.end") and g(r, "f.delta") <= -8, "DOWN")

    # ---- D5 价格上下文（马尔可夫 / 连击衰竭）----
    h5("C1_streak3_up", "3 连阳（含当前）→ 动量 vs 衰竭", lambda r: r["streak_up"] >= 3, "SAME")
    h5("C2_streak3_dn", "3 连阴", lambda r: r["streak_dn"] >= 3, "SAME")
    h5("C3_streak5_up", "5 连阳（极端延续）", lambda r: r["streak_up"] >= 5, "SAME")
    h5("C4_streak5_dn", "5 连阴", lambda r: r["streak_dn"] >= 5, "SAME")
    h5("C5_big_ret", "本窗大波动（|ret|≥发现集 p80=%.4f）→ 反转" % big_ret,
       lambda r: nan_ok(r["ret_ratio"]) and abs(r["ret"]) >= big_ret, "FLIP")
    h5("C6_tiny_ret", "微动窗（|ret|≤p80 的一半）→ 延续原方向",
       lambda r: abs(r["ret"]) <= big_ret / 2, "SAME")
    h5("C7_just_flipped", "刚反转（前窗与本窗反向）→ 再反转（钟摆）",
       lambda r: r.get("prev_ok") and r.get("prev_out") in ("UP", "DOWN")
       and r["out"] in ("UP", "DOWN") and r.get("prev_out") != r["out"], "FLIP")
    h5("C8_same2", "已 2 连同向 → 延续", lambda r: r.get("prev_ok")
       and r.get("prev_out") == r["out"] and r["out"] in ("UP", "DOWN"), "SAME")

    # ---- D6 前窗情绪背景（情绪的预测力本身）----
    h5("B1_prev_end≥70", "前窗情绪强多 → 本结果延续到次窗",
       lambda r: nan_ok(g(r, "prev_end_s")) and r.get("prev_end_s", 0) >= 70, "UP")
    h5("B2_prev_end≤30", "前窗情绪强空", lambda r: nan_ok(g(r, "prev_end_s")) and r.get("prev_end_s", 100) <= 30, "DOWN")

    # ---- D7 时段 / 周期位置 ----
    h5("S1_asia", "亚洲时段 0-7 UTC", lambda r: r["hour"] <= 7, "SAME")
    h5("S2_europe", "欧洲时段 8-15", lambda r: 8 <= r["hour"] <= 15, "SAME")
    h5("S3_us", "美洲时段 16-23", lambda r: r["hour"] >= 16, "SAME")
    h5("S4_weekend", "周末流动性差异", lambda r: r["wd"] >= 5, "SAME")
    h5("S5_grid_first", "周期首窗（15m 开盘对齐）", lambda r: r["grid"] == 0, "SAME")
    h5("S6_grid_last", "周期末窗（结算前对齐）", lambda r: r["grid"] == 2, "SAME")

    # ---- B 系列：窗内早读（前 150s 截断 → 整窗结算）----
    h5("T1_end≥70", "早读：150s 情绪≥70 → 整窗收阳", lambda r: g(r, "t.end") >= 70, "UP")
    h5("T2_end≤30", "早读：150s 情绪≤30 → 整窗收阴", lambda r: g(r, "t.end") <= 30, "DOWN")
    h5("T3_Δ≥10", "早读：150s 内情绪涨≥10pp", lambda r: g(r, "t.delta") >= 10, "UP")
    h5("T4_Δ≤-10", "早读：150s 内情绪跌≥10pp", lambda r: g(r, "t.delta") <= -10, "DOWN")
    h5("T5_touch85", "早读：触及≥85 → 整窗兑现", lambda r: g(r, "t.max") >= 85, "UP")
    h5("T6_touch15", "早读：触及≤15", lambda r: g(r, "t.min") <= 15, "DOWN")
    h5("T7_conf_up", "早读确认：前窗 UP 且 150s 情绪≥60 → 延续",
       lambda r: r.get("prev_out") == "UP" and g(r, "t.end") >= 60, "UP")
    h5("T8_conf_dn", "早读确认：前窗 DOWN 且 150s 情绪≤40 → 延续",
       lambda r: r.get("prev_out") == "DOWN" and g(r, "t.end") <= 40, "DOWN")
    h5("T9_deny_up", "早读否定：前窗 UP 但 150s 情绪≤40 → 反转",
       lambda r: r.get("prev_out") == "UP" and g(r, "t.end") <= 40, "DOWN")
    h5("T10_deny_dn", "早读否定：前窗 DOWN 但 150s 情绪≥60 → 反转",
       lambda r: r.get("prev_out") == "DOWN" and g(r, "t.end") >= 60, "UP")
    h5("T11_flat", "早读死水（range≤3）→ 无信息", lambda r: g(r, "t.range") <= 3, "SAME")

    # ---- 15m 周期假设 ----
    h15("N1_s_end≥70", "15m 周期末情绪≥70 → 次周期延续", lambda c: c["s_end"] >= 70, "UP")
    h15("N2_s_end≤30", "15m 周期末情绪≤30", lambda c: c["s_end"] <= 30, "DOWN")
    h15("N3_mono3_up", "窗间情绪两连升（d1>0 且 d2>0）",
        lambda c: len(c["d_win"]) == 2 and c["d_win"][0] > 0 and c["d_win"][1] > 0, "UP")
    h15("N4_mono3_dn", "窗间情绪两连降", lambda c: len(c["d_win"]) == 2 and c["d_win"][0] < 0 and c["d_win"][1] < 0, "DOWN")
    h15("N5_path_UUU", "三窗全阳 → 次周期（衰竭 vs 动量）", lambda c: c["path3"] == "UUU", "SAME")
    h15("N6_path_DDD", "三窗全阴", lambda c: c["path3"] == "DDD", "SAME")
    h15("N7_path_UUD", "两阳后阴（动能转弱）", lambda c: c["path3"] == "UUD", "DOWN")
    h15("N8_path_DDU", "两阴后阳", lambda c: c["path3"] == "DDU", "UP")
    h15("N9_peak_early", "周期情绪峰在首 1/3（早见顶 → 回落）",
        lambda c: c["peak_pos"] <= 1 / 3 and c["s_delta"] < 0, "DOWN")
    h15("N10_peak_late", "周期情绪峰在末 1/3（晚见顶 → 延续）",
        lambda c: c["peak_pos"] >= 2 / 3 and c["s_delta"] > 0, "UP")
    h15("N11_div_up", "情绪涨但周期收阴（背离修正）",
        lambda c: c["s_delta"] >= 5 and c["out"] == "DOWN", "UP")
    h15("N12_div_dn", "情绪跌但周期收阳", lambda c: c["s_delta"] <= -5 and c["out"] == "UP", "DOWN")
    h15("N13_streak2_up", "2 连阳周期", lambda c: c["streak_up"] >= 2, "SAME")
    h15("N14_streak2_dn", "2 连阴周期", lambda c: c["streak_dn"] >= 2, "SAME")
    h15("N15_big_move", "大波动周期（|ret|≥p80=%.4f）→ 反转" % cyc_big,
        lambda c: abs(c["ret"]) >= cyc_big, "FLIP")
    h15("N16_s_delta≥20", "周期情绪累计涨≥20pp", lambda c: c["s_delta"] >= 20, "UP")
    h15("N17_s_delta≤-20", "周期情绪累计跌≥20pp", lambda c: c["s_delta"] <= -20, "DOWN")
    h15("N18_hour Asia", "亚洲时段周期", lambda c: c["hour"] <= 7, "SAME")
    h15("N19_hour US", "美洲时段周期", lambda c: c["hour"] >= 16, "SAME")

    return H


# ============================================================
# 主流程
# ============================================================

def run(W: list[dict], out_path: str) -> dict:
    W = sorted(W, key=lambda w: int(w["start_time"]))
    split_ts = int(W[int(len(W) * (1 - HOLDOUT_RATIO))]["start_time"])
    recs = build_records(W)
    cycles = build_cycles(W, recs)
    print(f"[数据] {len(W)} 窗 / {len(cycles)} 完整 15m 周期 | "
          f"切分 {time.strftime('%m-%d %H:%M', time.gmtime(split_ts/1000))} UTC | "
          f"发现={sum(1 for r in recs if r['start'] < split_ts)} OOS={sum(1 for r in recs if r['start'] >= split_ts)}")

    arr_next = np.array(recs, dtype=object)
    pool_5m = np.array([r["has_next"] for r in recs])
    tgt_up = np.array([1.0 if (r["next_out"] or "") == "UP" else
                       (0.0 if (r["next_out"] or "") == "DOWN" else np.nan) for r in recs])
    tgt_same = np.array([1.0 if r["next_same"] else (0.0 if r["has_next"] and not r["next_same"]
                       else np.nan) for r in recs])
    # 窗内目标（早读系列）：整窗结算
    tgt_cur = np.array([1.0 if r["out"] == "UP" else (0.0 if r["out"] == "DOWN" else np.nan)
                        for r in recs])
    is_oos_5m = np.array([r["start"] >= split_ts for r in recs])

    pool_15 = np.array([c["has_next"] for c in cycles])
    t15_up = np.array([1.0 if (c["next_out"] or "") == "UP" else
                       (0.0 if (c["next_out"] or "") == "DOWN" else np.nan) for c in cycles])
    t15_same = np.array([1.0 if c["next_same"] else (0.0 if c["has_next"] and not c["next_same"]
                        else np.nan) for c in cycles])
    is_oos_15 = np.array([c["start"] >= split_ts for c in cycles])

    base_up_is = float(np.nanmean(tgt_up[~is_oos_5m & pool_5m]))
    base_same_is = float(np.nanmean(tgt_same[~is_oos_5m & pool_5m]))
    base_cur_is = float(np.nanmean(tgt_cur[~is_oos_5m]))
    base_up_oos = float(np.nanmean(tgt_up[is_oos_5m & pool_5m]))
    base_cur_oos = float(np.nanmean(tgt_cur[is_oos_5m]))
    b15_up_is = float(np.nanmean(t15_up[~is_oos_15 & pool_15]))
    b15_same_is = float(np.nanmean(t15_same[~is_oos_15 & pool_15]))
    b15_up_oos = float(np.nanmean(t15_up[is_oos_15 & pool_15]))
    print(f"[基准·发现集] 5m 次窗UP={base_up_is:.1%} 延续={base_same_is:.1%} "
          f"本窗UP={base_cur_is:.1%} | 15m 次周期UP={b15_up_is:.1%} 延续={b15_same_is:.1%}")

    H = register_hypotheses(recs, cycles, split_ts, W)
    print(f"[假设] 预注册 {len(H)} 条（5m×{sum(1 for x in H if x[1]=='5m')} + 15m×{sum(1 for x in H if x[1]=='15m')}）")

    # ---------- L1 ----------
    l1: list[dict] = []
    for name, scale, mech, fn, td in H:
        if scale == "5m":
            mask = fn(arr_next)
            is_early = name.startswith("T")
            tgt, base = (tgt_cur, base_cur_is) if is_early else (tgt_up, base_up_is)
            pool = np.ones(len(recs), dtype=bool) if is_early else pool_5m
        else:
            mask = fn(np.array(cycles, dtype=object))
            tgt, base, pool = t15_up, b15_up_is, pool_15
        r_is = eval_mask(mask, tgt, base, pool & ~is_oos_5m if scale == "5m" else pool & ~is_oos_15)
        if r_is is None:
            continue
        r_is.update({"name": name, "scale": scale, "mech": mech, "dir": td})
        # 延续率（早读系列同样报延续，供参考）
        if scale == "5m":
            cont = eval_mask(mask, tgt_same, base_same_is, pool_5m & ~is_oos_5m)
        else:
            cont = eval_mask(mask, t15_same, b15_same_is, pool_15 & ~is_oos_15)
        r_is["cont"] = cont
        l1.append(r_is)

    passed = bh_fdr([r["pval"] for r in l1], q=0.1)
    for r, ok in zip(l1, passed):
        r["fdr"] = ok

    def fam(r) -> str:
        """假设家族：early=窗内早读（T*，目标=本窗）/ cross5m=跨窗（目标=次窗）/ 15m。"""
        if r["scale"] == "15m":
            return "15m"
        return "early" if r["name"].startswith("T") else "cross5m"

    def ticket(r) -> bool:
        """家族化入场券：早读系列信息已被定价，胜率天花板高但边际小；
        跨窗/15m 系列是真正的延续·反转问题，按各自功效定门槛。"""
        f = fam(r)
        if f == "early":
            return r["n"] >= MIN_N_5M and abs(r["dev"]) >= MIN_DEV and r["pval"] < P_SIG
        if f == "cross5m":
            return r["n"] >= 100 and abs(r["dev"]) >= 0.015 and r["pval"] < 0.10
        return r["n"] >= MIN_N_15M and abs(r["dev"]) >= 0.02 and r["pval"] < 0.10

    surv = [r for r in l1 if ticket(r)]
    print(f"\n===== L1 单因子（发现集，{len(l1)} 检验 | FDR 通过 {sum(passed)} | 入场券 {len(surv)}）=====")
    for r in sorted(l1, key=lambda x: -abs(x["dev"]))[:25]:
        star = "*" if ticket(r) else " "
        fdr = "F" if r["fdr"] else " "
        print(f" {star}{fdr} [{r['scale']:>2}] {r['name']:<22} n={r['n']:>5} "
              f"{r['p']:6.1%} (dev {r['dev']:+.1%}pp) CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}] "
              f"p={r['pval']:.3f} 延续={('%4.1f%%' % (r['cont']['p']*100)) if r['cont'] else '  -  '}")

    # ---------- L2 / L3（家族内组合：早读/跨窗/15m 目标不同，不混排） ----------
    hmap = {name: fn for name, _, _, fn, _ in H}
    dir_of = {name: d for name, _, _, _, d in H}
    mech_of = {name: m for name, _, m, _, _ in H}

    def combo_mask(names, scale):
        arr = arr_next if scale == "5m" else np.array(cycles, dtype=object)
        m = np.ones(len(arr), dtype=bool)
        for nm in names:
            m &= hmap[nm](arr)
        return m

    _L2_FLOOR = {"early": 300, "cross5m": 60, "15m": 30}
    _L3_FLOOR = {"early": 200, "cross5m": 40, "15m": 20}

    def evaluate_combo(names, scale, family):
        """组合在发现集的统计；方向取 |dev| 最大父因子。"""
        if family == "early":
            tgt, base, pool, is_oos = tgt_cur, base_cur_is, np.ones(len(recs), bool), ~is_oos_5m
        elif family == "cross5m":
            tgt, base, pool, is_oos = tgt_up, base_up_is, pool_5m, ~is_oos_5m
        else:
            tgt, base, pool, is_oos = t15_up, b15_up_is, pool_15, ~is_oos_15
        r = eval_mask(combo_mask(names, scale), tgt, base, pool & is_oos)
        if r is None:
            return None
        parent_devs = [next((x["dev"] for x in l1 if x["name"] == nm), 0.0) for nm in names]
        main = names[int(np.argmax(np.abs(parent_devs)))]
        r.update({"name": " × ".join(names), "scale": scale, "family": family,
                  "dir": dir_of[main], "parents": list(names),
                  "boost": bool(abs(r["dev"]) > max(abs(d) for d in parent_devs))})
        return r

    fam_top: dict[str, list] = {}
    for f in ("early", "cross5m", "15m"):
        fam_top[f] = sorted([r for r in surv if fam(r) == f], key=lambda x: -abs(x["dev"]))[:10]
    # 调节因子（L1 已预注册，仅作为组合伙伴深化存活主因子，不单独入场）
    _COND = {"cross5m": ["S5_grid_first", "S6_grid_last", "C5_big_ret", "C6_tiny_ret",
                         "C7_just_flipped", "X5_cross_up", "X6_cross_dn", "B1_prev_end≥70",
                         "B2_prev_end≤30", "L1_touch90", "L1_touch10"],
             "15m": ["N18_hour Asia", "N19_hour US", "N15_big_move"],
             "early": []}
    for f, extra in _COND.items():
        have = {r["name"] for r in fam_top[f]}
        fam_top[f] += [r for r in l1 if r["name"] in extra and r["name"] not in have
                       and r["n"] >= (40 if f == "15m" else 100)]

    l2: list[dict] = []
    for f, tops in fam_top.items():
        mains = [r for r in tops if r["name"] not in _COND[f]][:8]
        conds = [r for r in tops if r["name"] in _COND[f]]
        pairs = [(a, b) for i, a in enumerate(mains) for b in mains[i + 1:]] \
            + [(a, c) for a in mains for c in conds]
        for a, b in pairs:
            r = evaluate_combo([a["name"], b["name"]], a["scale"], f)
            if r and r["n"] >= _L2_FLOOR[f]:
                l2.append(r)
    l2.sort(key=lambda x: -abs(x["dev"]))
    print(f"\n===== L2 双因子（家族内组合，发现集，{len(l2)} 组）=====")
    for r in l2[:14]:
        print(f"  {'↑' if r['boost'] else ' '} [{r['family']:>7}] {r['name']:<46} n={r['n']:>5} "
              f"{r['p']:6.1%} (dev {r['dev']:+.1%}pp) CI[{r['ci'][0]:.1%},{r['ci'][1]:.1%}] p={r['pval']:.3f}")

    l3: list[dict] = []
    _seen_sets = {frozenset(r["parents"]) for r in l2}
    for r2 in [x for x in l2 if x["boost"]][:6] or l2[:4]:
        for extra in fam_top[r2["family"]]:
            if extra["name"] in r2["parents"]:
                continue
            names = r2["parents"] + [extra["name"]]
            if frozenset(names) in _seen_sets:
                continue
            r = evaluate_combo(names, r2["scale"], r2["family"])
            if r and r["n"] >= _L3_FLOOR[r2["family"]] and abs(r["dev"]) >= abs(r2["dev"]) - 0.005:
                _seen_sets.add(frozenset(names))
                l3.append(r)
    l3.sort(key=lambda x: -abs(x["dev"]))
    print(f"\n===== L3 三因子（发现集，{len(l3)} 组）=====")
    for r in l3[:10]:
        print(f"    [{r['family']:>7}] {r['name']:<62} n={r['n']:>4} {r['p']:6.1%} "
              f"(dev {r['dev']:+.1%}pp) p={r['pval']:.3f}")

    # ---------- OOS 终验（家族配额：早读 4 / 跨窗 6 / 15m 6，防止早读系列挤占） ----------
    print(f"\n===== OOS 终验（后 {HOLDOUT_RATIO:.0%} 数据，只验不改）=====")
    quota = {"early": 4, "cross5m": 6, "15m": 6}
    pools = {}
    for f, cap in quota.items():
        if f == "15m":
            cands = [r for r in l3 if r["family"] == f and r["n"] >= 30] \
                + [r for r in l2 if r["family"] == f and r["n"] >= 40] \
                + [r for r in surv if fam(r) == f]
        else:
            floor = 150 if f == "early" else 60
            cands = [r for r in l3 if r["family"] == f and r["n"] >= floor] \
                + [r for r in l2 if r["family"] == f and r["n"] >= floor] \
                + [r for r in surv if fam(r) == f and r["n"] >= floor]
        pools[f] = sorted(cands, key=lambda x: -abs(x["dev"]))
    
    finals, seen = [], set()
    for f in ("cross5m", "15m", "early"):  # 跨窗家族优先呈现（用户核心问题）
        picked = 0
        for r in pools[f]:
            if picked >= quota[f]:
                break
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            names = r.get("parents") or [r["name"]]
            scale = r["scale"]
            if scale == "5m":
                if f == "early":
                    tgt, base, pool = tgt_cur, base_cur_oos, np.ones(len(recs), bool)
                else:
                    tgt, base, pool = tgt_up, base_up_oos, pool_5m
                mask = combo_mask(names, scale)
                r_oos = eval_mask(mask, tgt, base, pool & is_oos_5m)
                day_rows = [(recs[j], tgt[mask & is_oos_5m][i])
                            for i, j in enumerate(np.where(mask & is_oos_5m)[0])]
            else:
                mask = combo_mask(names, scale)
                r_oos = eval_mask(mask, t15_up, b15_up_oos, pool_15 & is_oos_15)
                day_rows = [(cycles[j], t15_up[mask & is_oos_15][i])
                            for i, j in enumerate(np.where(mask & is_oos_15)[0])]
            if r_oos is None or r_oos["n"] == 0:
                continue
            oos_ok = (r_oos["n"] >= OOS_MIN_N and np.sign(r_oos["dev"]) == np.sign(r["dev"]))
            k_all, n_all = r["k"] + r_oos["k"], r["n"] + r_oos["n"]
            lo_all, hi_all = wilson_bounds(k_all, n_all)
            ev = economic(r, W, recs, cycles, split_ts)
            # OOS 按日稳定性（OOS 通过者附上）
            daily = None
            if oos_ok:
                byday: dict[str, list] = {}
                for rec, tv in day_rows:
                    if not np.isfinite(tv):
                        continue
                    byday.setdefault(time.strftime("%m-%d", time.gmtime(rec["start"] / 1000)),
                                     []).append(tv)
                daily = {d: (len(vs), float(np.mean(vs))) for d, vs in sorted(byday.items())}
            finals.append({
                "name": r["name"], "scale": scale, "family": f, "dir": r.get("dir"),
                "mech": mech_of.get(names[-1], ""),
                "is": {"n": r["n"], "p": r["p"], "dev": r["dev"], "ci": r["ci"]},
                "oos": {"n": r_oos["n"], "p": r_oos["p"], "dev": r_oos["dev"],
                        "ci": r_oos["ci"]},
                "oos_pass": bool(oos_ok),
                "combined": {"n": n_all, "p": k_all / n_all, "ci": [lo_all, hi_all]},
                "ev": ev, "daily_oos": daily,
            })
            picked += 1
            flag = "✓" if oos_ok else "✗"
            print(f"  {flag} [{f:>7}] {r['name']}")
            print(f"      发现 n={r['n']} p={r['p']:.1%} ({r['dev']:+.1%}pp) | "
                  f"OOS n={r_oos['n']} p={r_oos['p']:.1%} ({r_oos['dev']:+.1%}pp) | "
                  f"合并 CI[{lo_all:.1%},{hi_all:.1%}]")
            if ev and ev.get("n") and "ev_2_1" in ev:
                print(f"      EV(费2+溢1)={ev['ev_2_1']:+.4f} CI{ev.get('ev_2_1_ci')} | "
                      f"实价覆盖 {ev.get('n_real', 0)}/{ev['n']} 均价={ev.get('avg_price', 0):.3f}")
            elif ev and ev.get("n"):
                print(f"      @0.50 假设口径 EV={ev.get('ev_050_assume', float('nan')):+.4f} "
                      f"(n={ev['n']})")
            if daily:
                ds = " ".join(f"{d}:{p:.0%}({n})" for d, (n, p) in daily.items())
                print(f"      OOS 按日: {ds}")

    report = {
        "meta": {"windows": len(W), "cycles": len(cycles), "split_ts": split_ts,
                 "decision_t_sec": DECISION_T_SEC, "bases_is": {
                     "5m_up": base_up_is, "5m_same": base_same_is, "5m_cur": base_cur_is,
                     "15m_up": b15_up_is, "15m_same": b15_same_is}},
        "l1": [{k: v for k, v in r.items() if k != "cont"} | 
               {"cont_p": r["cont"]["p"] if r["cont"] else None} for r in l1],
        "l2": l2, "l3": l3, "finals": finals,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[已写入] {out_path}")
    return report


def b15_up_is_oos(W, cycles, split_ts):
    """OOS 段 15m 次周期 UP 基准。"""
    vals = [1.0 if c["next_out"] == "UP" else 0.0
            for c in cycles if c["has_next"] and c["start"] >= split_ts]
    return sum(vals) / len(vals) if vals else 0.5


def economic(r, W, recs, cycles, split_ts) -> dict:
    """候选场景的逐注经济账（OOS 段）：5m=真实报价；15m=@0.50 口径。"""
    names = r.get("parents") or [r["name"]]
    scale = r["scale"]
    # 重建假设掩码（register_hypotheses 内的闭包通过名称重新求值）
    H = register_hypotheses(recs, cycles, split_ts, W)
    fns = {n: fn for n, _, _, fn, _ in H}
    dirs = {n: d for n, _, _, _, d in H}
    if scale == "5m":
        arr = np.array(recs, dtype=object)
        m = np.ones(len(arr), dtype=bool)
        for nm in names:
            m &= fns[nm](arr)
        is_early = any(nm.startswith("T") for nm in names)
        trade_dir = r.get("dir") or dirs[names[-1]]
        idxs, ds = [], []
        for j, rec in enumerate(recs):
            if not m[j] or rec["start"] < split_ts:
                continue
            if is_early:
                if rec["out"] not in ("UP", "DOWN"):
                    continue
                d = trade_dir if trade_dir in ("UP", "DOWN") else (
                    rec["out"] if trade_dir == "SAME" else
                    ("DOWN" if rec["out"] == "UP" else "UP"))
                idxs.append(j)
                ds.append(d)
            else:
                if not rec["has_next"]:
                    continue
                d = trade_dir if trade_dir in ("UP", "DOWN") else (
                    rec["out"] if trade_dir == "SAME" else
                    ("DOWN" if rec["out"] == "UP" else "UP"))
                idxs.append(j + 1)
                ds.append(d)
        return ev_eval(idxs, ds, W)
    # 15m：@0.50 假设口径（周期方向 → 押次周期首窗）
    arr = np.array(cycles, dtype=object)
    m = np.ones(len(arr), dtype=bool)
    for nm in names:
        m &= fns[nm](arr)
    trade_dir = r.get("dir") or dirs[names[-1]]
    wins, total = 0, 0
    for j, c in enumerate(cycles):
        if not m[j] or c["start"] < split_ts or not c["has_next"]:
            continue
        d = trade_dir if trade_dir in ("UP", "DOWN") else (
            c["out"] if trade_dir == "SAME" else ("DOWN" if c["out"] == "UP" else "UP"))
        total += 1
        wins += 1 if c["next_out"] == d else 0
    if total == 0:
        return {"n": 0}
    p = wins / total
    return {"n": total, "win_rate": p, "note": "15m EV 按 @0.50 口径",
            "ev_050_assume": p * (0.98 / 0.51 - 1) - (1 - p)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="情绪曲线 5m/15m 科学发现（L1→L2/L3→OOS→经济账）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default="output/sentiment_curve_discovery.json")
    args = ap.parse_args()
    with open(args.from_file, encoding="utf-8") as f:
        W = json.load(f)
    run(W, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
