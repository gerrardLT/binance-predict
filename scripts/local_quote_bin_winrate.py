#!/usr/bin/env python3
"""细粒度（30s / 15s）× 报价区间的条件胜率回测（5m / 15m 双市场）。

用户问题（升级版）：采样间隔本来就是 15s，按 30s / 15s 分桶重做——
每一时点，报价在哪个范围内，最终延续（DOWN 赢）的可能性最高？

方法（全真实数据配对，无任何假设价）：
  1. 每条报价样本按 (周期, 时点桶) 去重取首条
  2. 胜负 = 该周期 K 线结算 close < open（DOWN 赢）
  3. 每个 (市场, step, 时点, 报价区间) 统计 n / P(DOWN) / Wilson CI / 定价偏差
     定价偏差 = P(DOWN) − q̄：>0 表示该报价区间系统性偏便宜

表格：行 = 时点桶，列 = 报价区间；单元格 = n:胜率(偏差)。
数据：prediction_market_samples_online（5m 自 07-26 / 15m 自 08-13），
      klines_5m_cache_720d.json（结算判定）。
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
LOG = "output/quote_bin_winrate_fine.log"
BINS = [(0.05, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 0.55),
        (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 0.95)]
STEPS = (30, 15)          # 分桶粒度（秒）
MIN_N = 50                # 表格显示最小样本
MAX_COLS_15M = 20         # 15m 市场只打印前 20 个时点桶（其余进 JSON）


class Tee:
    def __init__(self):
        self.f = open(LOG, "w", encoding="utf-8")
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bin_of(q: float) -> int:
    for i, (a, b) in enumerate(BINS):
        if a <= q < b:
            return i
    return -1


def load_cycles() -> dict[int, tuple[float, float]]:
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    now_ms = int(time.time() * 1000)
    c5 = [(int(k[0]), float(k[1]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    return {ts // 300_000: (o, c) for ts, o, c in c5}


def agg15_cycles(cyc5: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    agg: dict[int, list] = defaultdict(list)
    for cyc, oc in cyc5.items():
        agg[cyc // 3].append((cyc, oc))
    out = {}
    for c15, lst in agg.items():
        if len(lst) == 3:
            lst.sort()
            out[c15] = (lst[0][1][0], lst[-1][1][1])
    return out


def pair_market(raw: list[dict], cyc_oc: dict, win_ms: int, step_s: int):
    """(slot, bin) -> [(q, win)]；每周期每时点桶首条。"""
    pair: dict[tuple, list] = defaultdict(list)
    seen: set[tuple] = set()
    step_ms = step_s * 1000
    n_slots = win_ms // step_ms
    for s in raw:
        q = s.get("down_price"); ts = s.get("timestamp")
        if q is None or ts is None or q <= 0.01 or q >= 0.99:
            continue
        ts = int(ts)
        cyc = ts // win_ms
        oc = cyc_oc.get(cyc)
        if oc is None:
            continue
        b = (ts - cyc * win_ms) // step_ms
        if b >= n_slots:
            continue
        if (cyc, b) in seen:
            continue
        seen.add((cyc, b))
        pair[(b, bin_of(float(q)))].append((float(q), oc[1] < oc[0]))
    return pair, len(seen), n_slots


def cell(grp: list) -> tuple[str, dict | None]:
    if len(grp) < MIN_N:
        return f"{len(grp):>4}:  —    ", None
    k = sum(w for _, w in grp)
    p = k / len(grp)
    lo, hi = wilson(k, len(grp))
    qm = float(np.mean([q for q, _ in grp]))
    return f"{len(grp):>4}:{p:3.0%}{p-qm:+4.0%}", {
        "n": len(grp), "p": p, "q": qm, "dev": p - qm, "wilson": [lo, hi]}


def report(name: str, step_s: int, pair: dict, n_seen: int, n_slots: int,
           rows_out: list, max_cols: int | None = None):
    show = n_slots if max_cols is None else min(n_slots, max_cols)
    print(f"\n===== {name} | {step_s}s 分桶（配对 {n_seen} 个周期-时点；"
          f"单元格 = n:胜率(偏差)；n≥{MIN_N} 才显示）=====")
    if max_cols is not None and n_slots > max_cols:
        print(f"  （仅打印前 {max_cols} 个时点，完整 {n_slots} 桶见 JSON）")
    hdr = "时点   " + "".join(f"{a:.1f}-{b:.1f}     " for a, b in BINS)
    print("  " + hdr)
    for b in range(show):
        cells = []
        for bi in range(len(BINS)):
            txt, rec = cell(pair.get((b, bi), []))
            cells.append(f"{txt}  ")
            if rec:
                a_, b_ = BINS[bi]
                rows_out.append({"market": name, "step_s": step_s,
                                 "slot": b + 1, "sec": (b + 1) * step_s,
                                 "bin": f"{a_:.2f}-{b_:.2f}", **rec})
        t_sec = (b + 1) * step_s
        mm, ss = divmod(t_sec, 60)
        print(f"  {mm}:{ss:02d}   " + "".join(cells))


def main() -> int:
    sys.stdout = Tee()
    cyc5 = load_cycles()
    print(f"K 线覆盖 {datetime.fromtimestamp(min(cyc5)*300, timezone.utc):%Y-%m-%d}"
          f" ~ {datetime.fromtimestamp(max(cyc5)*300, timezone.utc):%Y-%m-%d}"
          f"（{len(cyc5)} 个 5m 周期）")
    with open(SAMPLES, encoding="utf-8") as f:
        S = json.load(f)
    s5 = [s for s in S if s.get("market_period") == "5m"]
    s15 = [s for s in S if s.get("market_period") == "15m"]
    cyc15 = agg15_cycles(cyc5)
    print(f"样本：5m {len(s5)} 条 / 15m {len(s15)} 条")

    rows: list[dict] = []
    for step in STEPS:
        pair5, n5, ns5 = pair_market(s5, cyc5, 300_000, step)
        report("5m市场", step, pair5, n5, ns5, rows)
        pair15, n15, ns15 = pair_market(s15, cyc15, 900_000, step)
        report("15m市场", step, pair15, n15, ns15, rows, MAX_COLS_15M)

    # ---------- 汇总 ----------
    ok = [r for r in rows if r["n"] >= 100]
    print("\n===== 定价偏差 TOP15（n≥100；偏差>0 = 该区间买 DOWN 便宜）=====")
    for r in sorted(ok, key=lambda x: -x["dev"])[:15]:
        ev = r["p"] * (0.98 / r["q"] - 1.0) - (1 - r["p"])
        print(f"  {r['market']} {r['step_s']}s桶 t={r['sec']:>3}s 报价{r['bin']}: "
              f"n={r['n']:>5} P={r['p']:.1%} q̄={r['q']:.3f} 偏差{r['dev']:+.1%} EV={ev:+.3f}")
    print("\n===== 30s 粒度 vs 60s 粒度对照：5m 市场 0.55-0.75 黄金区是否更清晰 =====")
    for step in (30, 15):
        seg = [r for r in rows if r["market"] == "5m市场" and r["step_s"] == step
               and r["bin"] in ("0.55-0.65", "0.65-0.75") and r["n"] >= 50]
        for r in sorted(seg, key=lambda x: x["sec"]):
            ev = r["p"] * (0.98 / r["q"] - 1.0) - (1 - r["p"])
            print(f"  {step}s桶 t={r['sec']:>3}s {r['bin']}: n={r['n']:>4} "
                  f"P={r['p']:.1%} q̄={r['q']:.3f} 偏差{r['dev']:+.1%} EV={ev:+.3f}")

    with open("output/quote_bin_winrate_fine_result.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/quote_bin_winrate_fine_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
