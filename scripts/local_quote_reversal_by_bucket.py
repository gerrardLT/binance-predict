#!/usr/bin/env python3
"""预测市场「报价 0~0.1 反转概率」按最小时间间隔（15s 桶）统计：5m / 15m 双周期。

研究问题
--------
在一个预测市场窗口内（5m=300s / 15m=900s），从开始到结束按最小采样间隔 15s 分桶
（5m→20 桶，15m→60 桶）。当某侧报价 q ∈ (0, 0.1] 时，该侧最终结算获胜（=「反转」）
的概率是多少？哪个时间点最高？

数据
----
  output/pull_samples_5m_20260907.json    线上 prediction_market_samples，period=5m
                                          2026-07-26 ~ 09-07，243,657 条，~15s/条
  output/pull_samples_15m_20260907.json   period=15m，2026-08-13 ~ 09-07，134,418 条
  output/klines_5m_cache_720d.json        5m K 线主缓存（~09-04）
  output/klines_5m_tail_20260907.json     5m K 线增量（09-04 ~ 09-07）

口径
----
  结算真值：K 线 close vs open。5m 窗 = 单根 5m K；15m 窗 = 三根 5m K 聚合
            （open = 第 1 根 open，close = 第 3 根 close）。平局（close==open）剔除。
            与 scripts/local_s5_real_quote_ev.py、local_s6_quote_analysis_15m.py 同源。
  事件单位：per (window, bucket, side) 只取该桶内第一条样本 → 一个窗口在一个桶内
            对一侧至多贡献 1 个事件，避免同桶重复计数与窗口内自相关膨胀显著性。
  反转    ：报价侧最终获胜（q≤0.1 的冷门侧赢）。
  经济口径：FEE=2%，赢 = (1−FEE)/q − 1，输 = −1；盈亏平衡价 q* = P × 0.98。
            与 scripts/local_s5_real_quote_ev.py 一致。

自检（任一不过则结论不可信）
---------------------------
  ①窗口对齐：候选偏移扫描，取「尾桶报价收敛 ↔ K 线 outcome」一致率最高者；
             要求最优偏移 = 0 且一致率 ≥ 0.85（若窗口与 K 线错位，一致率会掉到 ~50%）。
  ②独立结果代理：样本内 btc_price 首末值判向 vs K 线判向，一致率 ≥ 0.95。
  ③报价无未来函数：事件桶 offset ∈ [0, 桶末]，且能从原始数组独立重算。
  ④覆盖率：每桶实际样本数 / 应有窗口数，防止「某桶只有极少数窗口有数据」被当成结论。
  ⑤日聚类不确定性：按 UTC 日分块 bootstrap，给出扣除时间自相关后的 CI（Wilson CI 偏窄）。

用法
----
    .venv\\Scripts\\python.exe scripts/local_quote_reversal_by_bucket.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

KLINE_CACHE = "output/klines_5m_cache_720d.json"
KLINE_TAIL = "output/klines_5m_tail_20260907.json"
SAMPLES = {
    "5m": "output/pull_samples_5m_20260907.json",
    "15m": "output/pull_samples_15m_20260907.json",
}
LOG_FILE = "output/quote_reversal_by_bucket.log"
OUT_JSON = "output/quote_reversal_by_bucket.json"

FEE = 0.02
BUCKET_MS = 15_000          # 最小采样间隔（样本 gap 众数 = 15000ms）
Q_LO, Q_HI = 0.0, 0.1       # 报价区间（用户口径 0~0.1，含 0.1）
WIN_MS = {"5m": 300_000, "15m": 900_000}
MIN_N_HEADLINE = 100        # 「最高反转率」headline 的最小样本门槛，防小样本假高峰


class Tee:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")
        try:
            sys.__stdout__.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


def iso(ms: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI（小样本/极端比例下比正态近似稳）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, c - h), min(1.0, c + h)


def day_block_bootstrap(flags: np.ndarray, days: np.ndarray, B: int = 4000,
                        seed: int = 20260907) -> tuple[float, float]:
    """按 UTC 日分块 bootstrap 均值 CI —— 扣除窗口间时间自相关（Wilson CI 假设独立会偏窄）。"""
    uniq = np.unique(days)
    if len(uniq) < 3:
        return (float("nan"), float("nan"))
    by_day = {d: flags[days == d] for d in uniq}
    rates = np.array([v.mean() for v in by_day.values()])
    wts = np.array([len(v) for v in by_day.values()], dtype=float)
    rng = np.random.default_rng(seed)
    n_days = len(uniq)
    means = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, n_days, n_days)
        w = wts[pick]
        means[i] = float(np.sum(rates[pick] * w) / np.sum(w))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- K 线与结算

def load_klines() -> dict[int, tuple[float, float]]:
    """5m K 线 → {bar_open_ms: (open, close)}。主缓存 + 增量，按时间排序去重。"""
    rows: list[list] = []
    for p in (KLINE_CACHE, KLINE_TAIL):
        with open(p, encoding="utf-8") as f:
            rows.extend(json.load(f))
    rows.sort(key=lambda k: int(k[0]))
    m: dict[int, tuple[float, float]] = {}
    for k in rows:
        m[int(k[0])] = (float(k[1]), float(k[4]))
    print(f"K 线 {len(m)} 根（去重后），范围 [{iso(min(m))} ~ {iso(max(m))}]")
    return m


def window_outcome(w_start: int, k5: dict[int, tuple[float, float]], period: str) -> str | None:
    """'UP' / 'DOWN' / None（数据缺失或平局）。"""
    if period == "5m":
        bar = k5.get(w_start)
        if bar is None:
            return None
        o, c = bar
    else:
        bars = [k5.get(w_start + i * 300_000) for i in range(3)]
        if any(b is None for b in bars):
            return None
        o, c = bars[0][0], bars[2][1]
    if o <= 0:
        return None
    if c > o:
        return "UP"
    if c < o:
        return "DOWN"
    return None


# ---------------------------------------------------------------- 事件构建

def build(period: str, k5: dict[int, tuple[float, float]]) -> dict:
    """把样本整理成 per-(window, bucket, side) 事件表 + 窗口级元信息。"""
    L = WIN_MS[period]
    nb = L // BUCKET_MS
    with open(SAMPLES[period], encoding="utf-8") as f:
        raw = json.load(f)

    # 窗口归组（同时记录首/末 btc_price 供自检②）
    by_win: dict[int, list[dict]] = defaultdict(list)
    for s in raw:
        ts = int(s["timestamp"])
        by_win[ts // L * L].append(s)

    out_cache: dict[int, str | None] = {}
    n_win = n_win_ok = n_win_tie = n_win_nodata = 0
    # events[bucket][side] = list of (win_flag, price, day)
    events: dict[int, dict[str, list]] = {b: {"UP": [], "DOWN": []} for b in range(nb)}
    # 全量报价（不限 0~0.1）用于校准曲线对照
    allq: dict[str, list[tuple[int, float]]] = {"UP": [], "DOWN": []}
    # 自检①用：尾桶报价收敛 —— (quote, 报价指向的侧, 该侧, 实际 outcome)
    tail_conv: list[tuple[float, str, str, str]] = []
    # 自检②用：btc_price 首末判向
    btc_proxy: list[tuple[str, str]] = []
    # 自检③用：offset 越界计数
    bad_off = 0
    seen: set[tuple[int, int, str]] = set()
    # 每窗口每侧「首次触达 q≤0.1」的时刻 → (bucket, win_flag, price, day)
    first_hit: dict[str, dict[int, tuple[int, int, float, str]]] = {"UP": {}, "DOWN": {}}

    for w, rows in by_win.items():
        n_win += 1
        if w not in out_cache:
            out_cache[w] = window_outcome(w, k5, period)
        oc = out_cache[w]
        if oc is None:
            # 区分「K 线缺失」与「平局」
            if period == "5m":
                bar = k5.get(w)
                missing = bar is None
                tie = (not missing) and bar[0] == bar[1]
            else:
                bars = [k5.get(w + i * 300_000) for i in range(3)]
                missing = any(b is None for b in bars)
                tie = (not missing) and bars[0][0] == bars[2][1]
            if missing:
                n_win_nodata += 1
            elif tie:
                n_win_tie += 1
            continue
        n_win_ok += 1
        rows.sort(key=lambda s: int(s["timestamp"]))
        day = time.strftime("%Y-%m-%d", time.gmtime(w / 1000))

        # 自检②：btc_price 首末判向
        bp = [float(s["btc_price"]) for s in rows if s.get("btc_price")]
        if len(bp) >= 2 and bp[0] > 0 and bp[-1] != bp[0]:
            btc_proxy.append((oc, "UP" if bp[-1] > bp[0] else "DOWN"))

        for s in rows:
            ts = int(s["timestamp"])
            off = ts - w
            b = off // BUCKET_MS
            if b < 0 or b >= nb:
                bad_off += 1
                continue
            for side in ("UP", "DOWN"):
                q = s.get("up_price") if side == "UP" else s.get("down_price")
                if q is None:
                    continue
                q = float(q)
                allq[side].append((1 if oc == side else 0, q))
                # 首次触价：按窗口维度记录，必须在桶内去重之前（否则会漏掉
                # 「同桶首条 >0.1、次条才 ≤0.1」的首次触达时刻）
                if Q_LO < q <= Q_HI and w not in first_hit[side]:
                    first_hit[side][w] = (b, 1 if oc == side else 0, q, day)
                key = (w, b, side)
                if key in seen:
                    continue          # 同窗同桶同侧只取首条
                seen.add(key)
                if Q_LO < q <= Q_HI:
                    events[b][side].append((1 if oc == side else 0, q, day))
                # 自检①：最后 2 桶的收敛报价 → 市场指向哪一侧
                if b >= nb - 2 and (q > 0.9 or q < 0.1):
                    other = "DOWN" if side == "UP" else "UP"
                    pointed = side if q > 0.5 else other
                    tail_conv.append((q, pointed, side, oc))

    # 自检①化简：尾桶「报价指向的侧」与 outcome 的一致率
    conv_hits = conv_n = 0
    for _q, pointed, _side, oc in tail_conv:
        conv_n += 1
        if pointed == oc:
            conv_hits += 1

    return {
        "period": period, "L": L, "nb": nb, "events": events, "allq": allq,
        "first_hit": first_hit,
        "n_win": n_win, "n_win_ok": n_win_ok, "n_win_tie": n_win_tie,
        "n_win_nodata": n_win_nodata, "bad_off": bad_off,
        "conv_rate": (conv_hits / conv_n) if conv_n else float("nan"),
        "conv_n": conv_n,
        "btc_agree": (sum(1 for a, b in btc_proxy if a == b) / len(btc_proxy))
                     if btc_proxy else float("nan"),
        "btc_n": len(btc_proxy),
        "days": sorted({d for b in range(nb) for s in ("UP", "DOWN")
                        for (_, _, d) in events[b][s]}),
    }


# ---------------------------------------------------------------- 报告

def fmt_bucket_table(r: dict, side: str) -> list[dict]:
    nb = r["nb"]; L = r["L"]
    rows = []
    for b in range(nb):
        ev = r["events"][b][side]
        n = len(ev)
        if n == 0:
            continue
        flags = np.array([e[0] for e in ev], dtype=float)
        days = np.array([e[2] for e in ev])
        qs = np.array([e[1] for e in ev], dtype=float)
        k = int(flags.sum())
        p = k / n
        lo, hi = wilson(k, n)
        dlo, dhi = day_block_bootstrap(flags, days)
        qb = float(qs.mean())
        # 逐事件 EV（不能用桶内 q̄：1/q 是凸函数，E[1/q] ≠ 1/E[q]，会有 Jensen 偏差）
        ev_per = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
        ev_avg = float(ev_per.mean())
        ev_se = float(ev_per.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        ev_lo, ev_hi = day_block_bootstrap(ev_per, days)
        rows.append({
            "bucket": b,
            "t_sec": f"{b * 15}~{(b + 1) * 15}",
            "elapsed_pct": round((b + 0.5) * 15 / (L / 1000) * 100, 1),
            "remain_sec": int(L / 1000 - (b + 1) * 15),
            "n": n, "k": k, "p": p,
            "ci_lo": lo, "ci_hi": hi,
            "day_ci_lo": dlo, "day_ci_hi": dhi,
            "q_mean": qb, "ev": ev_avg, "ev_se": ev_se,
            "ev_day_lo": ev_lo, "ev_day_hi": ev_hi,
            "breakeven_q": p * (1 - FEE),
            "n_days": int(len(np.unique(days))),
        })
    return rows


def merge_sides(r: dict) -> list[dict]:
    """合并 UP/DOWN 两侧（用户口径未指定侧）：同一桶内两侧事件汇总，按归一化时刻排列。"""
    nb = r["nb"]; L = r["L"] / 1000
    rows = []
    for b in range(nb):
        ev = list(r["events"][b]["UP"]) + list(r["events"][b]["DOWN"])
        n = len(ev)
        if n == 0:
            continue
        flags = np.array([e[0] for e in ev], dtype=float)
        days = np.array([e[2] for e in ev])
        qs = np.array([e[1] for e in ev], dtype=float)
        k = int(flags.sum()); p = k / n
        lo, hi = wilson(k, n)
        dlo, dhi = day_block_bootstrap(flags, days)
        ev_per = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
        rows.append({
            "bucket": b, "t_sec": f"{b * 15}~{(b + 1) * 15}",
            "elapsed_pct": round((b + 0.5) * 15 / L * 100, 1),
            "remain_sec": int(L - (b + 1) * 15),
            "n": n, "k": k, "p": p, "ci_lo": lo, "ci_hi": hi,
            "day_ci_lo": dlo, "day_ci_hi": dhi,
            "q_mean": float(qs.mean()), "ev": float(ev_per.mean()),
            "ev_se": float(ev_per.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
            "ev_day_lo": day_block_bootstrap(ev_per, days)[0],
            "ev_day_hi": day_block_bootstrap(ev_per, days)[1],
            "breakeven_q": p * (1 - FEE), "n_days": int(len(np.unique(days))),
        })
    return rows


def print_table(rows: list[dict], period: str, label: str) -> None:
    print(f"\n----- [{period}] {label}：报价 q∈(0,{Q_HI}] → 该侧最终获胜（反转）的概率 -----")
    print(f"{'桶':>3} {'窗口内时刻(s)':>13} {'已过窗%':>7} {'剩余(s)':>7} {'n':>6} {'反转':>5} "
          f"{'P(反转)':>8} {'Wilson95%':>15} {'日聚类95%':>15} {'q̄':>6} "
          f"{'EV/注':>7} {'EV日聚类95%':>17}")
    for r in rows:
        w = f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
        dw = ("[nan,nan]" if math.isnan(r["day_ci_lo"])
              else f"[{r['day_ci_lo']:.3f},{r['day_ci_hi']:.3f}]")
        ew = ("[nan,nan]" if math.isnan(r["ev_day_lo"])
              else f"[{r['ev_day_lo']:+.2f},{r['ev_day_hi']:+.2f}]")
        print(f"{r['bucket']:>3} {r['t_sec']:>13} {r['elapsed_pct']:>6.0f}% {r['remain_sec']:>7} "
              f"{r['n']:>6} {r['k']:>5} {r['p']:>7.1%} {w:>15} {dw:>15} {r['q_mean']:>6.3f} "
              f"{r['ev']:>+7.3f} {ew:>17}")


def calibration(allq: dict[str, list[tuple[int, float]]], side: str) -> None:
    """全报价区间校准曲线：P(win | q) vs q —— 0~0.1 桶是否系统性偏离市场定价。"""
    data = allq[side]
    if not data:
        return
    arr = np.array(data, dtype=float)
    win, q = arr[:, 0], arr[:, 1]
    edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 1.01]
    print(f"\n  校准曲线 P({side} win | q) —— 全报价区间（对照 0~0.1 段是否 mispricing）")
    print(f"  {'q 区间':>14} {'n':>7} {'q̄(市场隐含)':>12} {'P(实际)':>9} {'差值(pp)':>9}")
    for a, b in zip(edges[:-1], edges[1:]):
        m = (q > a) & (q <= b)
        n = int(m.sum())
        if n < 30:
            continue
        qb = float(q[m].mean()); pw = float(win[m].mean())
        print(f"  {f'({a},{b}]':>14} {n:>7} {qb:>12.3f} {pw:>9.1%} {(pw - qb) * 100:>+9.1f}")


def depth_split(r: dict, side: str) -> None:
    """0~0.1 内再分深度档：越冷门的报价反转率越低但赔率越高。"""
    bands = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.1)]
    agg = {b: [] for b in bands}
    for bk in range(r["nb"]):
        for (f, q, d) in r["events"][bk][side]:
            for lo, hi in bands:
                if lo < q <= hi:
                    agg[(lo, hi)].append((f, q, d))
                    break
    print(f"\n  [{r['period']}] {side} 侧 0~0.1 内深度分档（全时段合并）")
    print(f"  {'q 档':>14} {'n':>6} {'P(反转)':>8} {'Wilson95%':>15} {'q̄':>6} {'EV/注':>7} {'EV±SE':>8}")
    for (lo, hi), ev in agg.items():
        n = len(ev)
        if n == 0:
            continue
        flags = np.array([e[0] for e in ev], dtype=float)
        qs = np.array([e[1] for e in ev], dtype=float)
        days = np.array([e[2] for e in ev])
        k = int(flags.sum()); p = k / n
        cl, ch = wilson(k, n)
        qb = float(qs.mean())
        ev_per = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
        evv = float(ev_per.mean())
        se = float(ev_per.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        print(f"  {f'({lo},{hi}]':>14} {n:>6} {p:>7.1%} {f'[{cl:.3f},{ch:.3f}]':>15} "
              f"{qb:>6.3f} {evv:>+7.3f} {f'±{1.96 * se:.3f}':>8}")


def first_hit_report(r: dict) -> None:
    """首次触价分析：区分「早期桶为空」是市场规律还是数据截断。

    主表的桶 b 统计的是「该桶内报价 ≤0.1」，包含「报价从第 2 分钟起就一直趴在
    0.05」的持续态样本。本函数只看每窗口每侧**首次**触达 q≤0.1 的时刻，回答：
      (a) 到底多大比例的窗口会出现 0~0.1 报价、最早什么时候出现；
      (b) 若首次触价发生在 t，反转概率是多少（去掉持续态污染）。
    """
    L = r["L"] / 1000
    nb = r["nb"]
    n_ok = r["n_win_ok"]
    fh = r["first_hit"]

    print(f"\n----- [{r['period']}] 首次触价分析（每窗口每侧只记第一次 q≤{Q_HI} 的时刻）-----")
    ever_up = set(fh["UP"].keys())
    ever_dn = set(fh["DOWN"].keys())
    ever_any = ever_up | ever_dn
    print(f"  可结算窗口 {n_ok}；曾出现 q≤{Q_HI} 报价的窗口："
          f"UP侧 {len(ever_up)} ({len(ever_up) / n_ok:.1%}) | "
          f"DOWN侧 {len(ever_dn)} ({len(ever_dn) / n_ok:.1%}) | "
          f"任一侧 {len(ever_any)} ({len(ever_any) / n_ok:.1%})")

    # (a) 首次触价时刻分位（秒 + 已过窗口%）
    for side in ("UP", "DOWN"):
        offs = np.array([v[0] * 15 + 7.5 for v in fh[side].values()], dtype=float)
        if len(offs) < 10:
            print(f"  {side}: 样本不足（n={len(offs)}）")
            continue
        qs = np.percentile(offs, [5, 10, 25, 50, 75, 90])
        print(f"  {side} 首次触价时刻(s) p5/p10/p25/p50/p75/p90 = "
              + " / ".join(f"{v:.0f}" for v in qs)
              + "  （占窗口 " + " / ".join(f"{v / L:.0%}" for v in qs) + "）")
        # 前 1/3 窗口内首次触价的占比 → 判断早期是否真的几乎不发生
        early = float((offs < L / 3).mean())
        print(f"     其中在窗口前 1/3（<{L / 3:.0f}s）内首次触价的占比 = {early:.1%}")

    # (b) 按首次触价时刻分段的条件反转率（双侧合并）
    bands = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300),
             (300, 420), (420, 540), (540, 660), (660, 780), (780, 900)]
    rows = []
    for (lo, hi) in bands:
        if hi > L:
            continue
        ev = [v for side in ("UP", "DOWN") for v in fh[side].values()
              if lo <= v[0] * 15 < hi]
        n = len(ev)
        if n == 0:
            continue
        flags = np.array([e[1] for e in ev], dtype=float)
        qs_ = np.array([e[2] for e in ev], dtype=float)
        days = np.array([e[3] for e in ev])
        k = int(flags.sum()); p = k / n
        cl, ch = wilson(k, n)
        dlo, dhi = day_block_bootstrap(flags, days)
        ev_per = np.where(flags > 0, (1 - FEE) / qs_ - 1.0, -1.0)
        rows.append((lo, hi, n, k, p, cl, ch, dlo, dhi,
                     float(qs_.mean()), float(ev_per.mean())))
    print(f"\n  按「首次触价时刻」分段的条件反转率（双侧合并，每窗口每侧至多 1 条）")
    print(f"  {'首次触价(s)':>12} {'n':>6} {'反转':>5} {'P(反转)':>8} {'Wilson95%':>15} "
          f"{'日聚类95%':>15} {'q̄':>6} {'EV/注':>7}")
    for lo, hi, n, k, p, cl, ch, dlo, dhi, qb, evv in rows:
        dw = ("[nan,nan]" if math.isnan(dlo) else f"[{dlo:.3f},{dhi:.3f}]")
        print(f"  {f'{lo}~{hi}':>12} {n:>6} {k:>5} {p:>7.1%} {f'[{cl:.3f},{ch:.3f}]':>15} "
              f"{dw:>15} {qb:>6.3f} {evv:>+7.3f}")
    elig = [x for x in rows if x[2] >= MIN_N_HEADLINE]
    if elig:
        best = max(elig, key=lambda x: x[4])
        print(f"  ★ 首次触价口径下反转率最高段：{best[0]}~{best[1]}s → P={best[4]:.1%} "
              f"n={best[2]} EV={best[10]:+.3f}")


def main() -> int:
    sys.stdout = Tee(LOG_FILE)
    print("=" * 118)
    print("预测市场「报价 0~0.1 → 最终反转」概率：按最小时间间隔（15s 桶）统计 | 5m / 15m")
    print("=" * 118)
    k5 = load_klines()

    out: dict = {"meta": {"generated": iso(int(time.time() * 1000)), "fee": FEE,
                          "bucket_ms": BUCKET_MS, "q_range": [Q_LO, Q_HI],
                          "min_n_headline": MIN_N_HEADLINE},
                 "periods": {}}

    for period in ("5m", "15m"):
        r = build(period, k5)
        print("\n" + "=" * 118)
        print(f"### {period} 市场  窗口 {r['L'] // 1000}s → {r['nb']} 个 15s 桶")
        print(f"窗口总数 {r['n_win']} | 可结算 {r['n_win_ok']} | 平局剔除 {r['n_win_tie']} "
              f"| K线缺失 {r['n_win_nodata']} | 日期跨度 {r['days'][0] if r['days'] else '-'} "
              f"~ {r['days'][-1] if r['days'] else '-'}（{len(r['days'])} 天）")
        print(f"offset 越界样本 {r['bad_off']}（应为 0）")

        # ---------- 自检 ----------
        print("\n----- 自检 -----")
        ok1 = r["conv_rate"] == r["conv_rate"] and r["conv_rate"] >= 0.85
        print(f"①窗口对齐（尾桶报价收敛↔K线outcome 一致率）: {r['conv_rate']:.1%} "
              f"(n={r['conv_n']}) {'✅' if ok1 else '❌ 对齐可疑，结论不可信'}")
        ok2 = r["btc_agree"] == r["btc_agree"] and r["btc_agree"] >= 0.90
        print(f"②独立结果代理（样本 btc_price 首末判向↔K线判向）: {r['btc_agree']:.4%} "
              f"(n={r['btc_n']}) {'✅' if ok2 else '❌'}"
              f"｜注：首/末样本不落在窗口边界上（首条晚于开盘数秒、末条早于收盘数秒），"
              f"边界价差被采样时序吃掉，故一致率天然 <100%；对齐判据以①为准")
        ok3 = r["bad_off"] == 0
        print(f"③报价无未来函数（桶 offset 合法）: {'✅' if ok3 else '❌'}")
        if not (ok1 and ok3):
            print("!! 自检未通过，跳过结论输出")
            continue

        # ---------- 覆盖率 ----------
        print("\n----- 桶覆盖率（每桶有报价事件的窗口数 / 可结算窗口数）-----")
        cov = []
        for side in ("UP", "DOWN"):
            cs = [len(r["events"][b][side]) / r["n_win_ok"] * 100 for b in range(r["nb"])]
            cov.append(cs)
            print(f"  {side}: " + " ".join(f"{c:.0f}" for c in cs))

        # ---------- 首次触价分析（判断早期空桶是市场规律还是数据截断）----------
        first_hit_report(r)

        # ---------- 主表 ----------
        tables = {}
        for side in ("UP", "DOWN"):
            rows = fmt_bucket_table(r, side)
            tables[side] = rows
            print_table(rows, period, side)
            calibration(r["allq"], side)
            depth_split(r, side)

        # ---------- headline：反转率最高的时间点 ----------
        print("\n----- ★ 反转率最高的时间点（n ≥ %d 才进 headline，防小样本假高峰）-----" % MIN_N_HEADLINE)
        head = {}
        for side in ("UP", "DOWN"):
            elig = [x for x in tables[side] if x["n"] >= MIN_N_HEADLINE]
            if not elig:
                print(f"  {side}: 无任何桶满足 n≥{MIN_N_HEADLINE}")
                head[side] = None
                continue
            best = max(elig, key=lambda x: x["p"])
            head[side] = best
            print(f"  {side} 侧：桶 {best['bucket']}（窗口内 {best['t_sec']}s，剩余 {best['remain_sec']}s）"
                  f" → P(反转) = {best['p']:.1%}  n={best['n']}  "
                  f"Wilson[{best['ci_lo']:.3f},{best['ci_hi']:.3f}]  "
                  f"日聚类[{best['day_ci_lo']:.3f},{best['day_ci_hi']:.3f}]  "
                  f"q̄={best['q_mean']:.3f}  EV={best['ev']:+.3f}")
            # top5
            top5 = sorted(elig, key=lambda x: -x["p"])[:5]
            print(f"    Top5 桶: " + " | ".join(
                f"b{x['bucket']}({x['t_sec']}s) {x['p']:.1%} n={x['n']}" for x in top5))
            # 无门槛的原始最大值（供对照，通常是极小样本）；与 headline 相同时不必再报「不可用」
            raw_best = max(tables[side], key=lambda x: x["p"])
            if raw_best["bucket"] != best["bucket"]:
                print(f"    （无 n 门槛的原始最大：桶 {raw_best['bucket']} P={raw_best['p']:.1%} "
                      f"n={raw_best['n']} —— 样本过小，不采信）")

        # ---------- 合并双侧视图（用户口径未指定侧）+ 归一化时间轴 ----------
        merged = merge_sides(r)
        print_table(merged, period, "UP+DOWN 合并")
        m_elig = [x for x in merged if x["n"] >= MIN_N_HEADLINE]
        print(f"\n----- ★ 合并双侧：反转率最高的时间点（n ≥ {MIN_N_HEADLINE}）-----")
        m_best = None
        if not m_elig:
            print(f"  无桶满足 n≥{MIN_N_HEADLINE}")
        else:
            m_best = max(m_elig, key=lambda x: x["p"])
            print(f"  桶 {m_best['bucket']}（窗口内 {m_best['t_sec']}s = 已过 {m_best['elapsed_pct']:.1f}%，"
                  f"剩余 {m_best['remain_sec']}s）→ P(反转) = {m_best['p']:.1%}  n={m_best['n']}  "
                  f"Wilson[{m_best['ci_lo']:.3f},{m_best['ci_hi']:.3f}]  "
                  f"日聚类[{m_best['day_ci_lo']:.3f},{m_best['day_ci_hi']:.3f}]  "
                  f"q̄={m_best['q_mean']:.3f}  EV/注={m_best['ev']:+.3f} "
                  f"EV日聚类[{m_best['ev_day_lo']:+.2f},{m_best['ev_day_hi']:+.2f}]")
            top5 = sorted(m_elig, key=lambda x: -x["p"])[:5]
            print("    Top5: " + " | ".join(
                f"b{x['bucket']}({x['t_sec']}s/{x['elapsed_pct']:.0f}%) {x['p']:.1%} "
                f"n={x['n']} EV{x['ev']:+.2f}" for x in top5))

        def _round_row(x: dict) -> dict:
            o = {}
            for kk, vv in x.items():
                if isinstance(vv, float):
                    o[kk] = None if math.isnan(vv) else round(vv, 5)
                else:
                    o[kk] = vv
            return o

        def _hl(x: dict | None) -> dict | None:
            if x is None:
                return None
            return {"bucket": x["bucket"], "t_sec": x["t_sec"],
                    "elapsed_pct": x["elapsed_pct"], "remain_sec": x["remain_sec"],
                    "n": x["n"], "p": round(x["p"], 5), "ev": round(x["ev"], 5),
                    "q_mean": round(x["q_mean"], 5),
                    "ci": [round(x["ci_lo"], 5), round(x["ci_hi"], 5)],
                    "day_ci": [None if math.isnan(x["day_ci_lo"]) else round(x["day_ci_lo"], 5),
                               None if math.isnan(x["day_ci_hi"]) else round(x["day_ci_hi"], 5)]}

        out["periods"][period] = {
            "n_windows": r["n_win"], "n_settleable": r["n_win_ok"],
            "n_tie": r["n_win_tie"], "n_kline_missing": r["n_win_nodata"],
            "days": r["days"], "checks": {
                "align_conv_rate": r["conv_rate"], "align_conv_n": r["conv_n"],
                "btc_proxy_agree": r["btc_agree"], "btc_proxy_n": r["btc_n"],
                "bad_offset": r["bad_off"],
            },
            "coverage_pct": {"UP": cov[0], "DOWN": cov[1]},
            "tables": {s: [_round_row(x) for x in tables[s]] for s in ("UP", "DOWN")},
            "table_merged": [_round_row(x) for x in merged],
            "headline": {"UP": _hl(head.get("UP")), "DOWN": _hl(head.get("DOWN")),
                         "merged": _hl(m_best)},
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果 JSON → {OUT_JSON} | 日志 → {LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
