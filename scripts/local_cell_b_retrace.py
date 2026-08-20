#!/usr/bin/env python3
"""Cell B 第二代假设实验：回撤确认（新实验，独立于首轮约束检验）。

事件池：5m 市场 t∈[45,60)s × DOWN 报价 q∈[0.15,0.25)（329 事件，基线胜率 24.0%）。

冻结假设（测试前定义，不得事后修改）：
  H1 retrace>0：决策时刻 btc 已从本周期内此前采样高点回落 → 胜率 > 基线
     机制：尖峰停滞/回落 = 追涨动能耗尽，DOWN（均值回归）概率上升
  H2 last_step<0：最近一个采样步长转负 → 胜率 > 基线
  H3 retrace≥Discovery 中位数（仅 retrace>0 子集内）→ 进一步富集

纪律：时序 60/20/20；阈值仅由 Discovery 生成；Val/Hold 只验不改；
      样本薄（329），verdict 上限 PROMISING，负结果保留。
EV 口径：赢 0.98/q−1 / 输 −1（费 2%）。
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
LOG = "output/cell_b_retrace.log"
T_LO, T_HI, Q_LO, Q_HI = 45, 60, 0.15, 0.25
N_DAYS = 24.0
MIN_N = 30
FEE_RET = 0.98


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


def stat(evs: list[dict]) -> dict | None:
    if not evs:
        return None
    k = sum(e["win"] for e in evs)
    n = len(evs)
    lo, hi = wilson(k, n)
    pnl = [(FEE_RET / e["q"] - 1.0) if e["win"] else -1.0 for e in evs]
    return {"n": n, "k": k, "p": k / n, "lo": lo, "hi": hi,
            "ev": float(np.mean(pnl)),
            "q": float(np.mean([e["q"] for e in evs])),
            "freq_day": n / N_DAYS}


def daily_beat(evs: list[dict], fn) -> tuple[int, int]:
    sub = [e for e in evs if fn(e)]
    days: dict[int, list] = defaultdict(list)
    for e in sub:
        days[e["ts"] // 86_400_000].append(e)
    beat = total = 0
    for d in sorted(days):
        dd = days[d]
        if len(dd) < 3:
            continue
        total += 1
        p = sum(e["win"] for e in dd) / len(dd)
        q = float(np.mean([e["q"] for e in dd]))
        beat += p > q
    return beat, total


def main() -> int:
    sys.stdout = Tee()
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    now_ms = int(time.time() * 1000)
    rows = [(int(k[0]), float(k[1]), float(k[4])) for k in kl]
    if rows and rows[-1][0] + 300_000 > now_ms:
        rows.pop()
    idx = {r[0] // 300_000: i for i, r in enumerate(rows)}

    with open(SAMPLES, encoding="utf-8") as f:
        S = json.load(f)
    s5 = sorted((s for s in S if s.get("market_period") == "5m"),
                key=lambda s: int(s["timestamp"]))
    # 按周期分组（用于路径重建）
    bycyc: dict[int, list] = defaultdict(list)
    for s in s5:
        ts = int(s["timestamp"]); btc = s.get("btc_price")
        if btc is not None:
            bycyc[ts // 300_000].append((ts, float(btc)))

    evs, hit = [], set()
    for s in s5:
        q, ts, btc = s.get("down_price"), s.get("timestamp"), s.get("btc_price")
        if q is None or ts is None or btc is None or not (Q_LO <= float(q) < Q_HI):
            continue
        ts = int(ts); cyc = ts // 300_000
        if cyc in hit:
            continue
        t_sec = ts - cyc * 300_000
        if t_sec < T_LO * 1000 or t_sec >= T_HI * 1000:
            continue
        i = idx.get(cyc)
        if i is None:
            continue
        o, c = rows[i][1], rows[i][2]
        if o <= 0:
            continue
        hit.add(cyc)
        btc = float(btc)
        # 路径重建：此前采样点（< ts）
        prior = [(t_, b) for t_, b in bycyc.get(cyc, []) if t_ < ts]
        prior.sort()
        highs = [b for _, b in prior] + [btc]
        max_btc = max(highs)
        retrace_bp = (max_btc - btc) / o * 1e4
        last_step = (btc - prior[-1][1]) / o * 1e4 if prior else 0.0
        depth_bp = (btc - o) / o * 1e4
        evs.append({"ts": ts, "q": float(q), "win": c < o,
                    "retrace": retrace_bp, "last_step": last_step,
                    "depth": depth_bp})

    n = len(evs)
    base = stat(evs)
    print(f"Cell B 事件 {n}（日均 {base['freq_day']:.1f}）| 基线胜率 {base['p']:.1%} "
          f"[W {base['lo']:.0%},{base['hi']:.0%}] EV={base['ev']:+.3f}")
    n_ret = sum(1 for e in evs if e["retrace"] > 0)
    n_flip = sum(1 for e in evs if e["depth"] <= 0)
    print(f"retrace>0 占比 {n_ret}/{n}（{n_ret/n:.0%}）| 已跌回开盘下 depth<=0 占比 {n_flip}/{n}（{n_flip/n:.0%}）")

    evs.sort(key=lambda e: e["ts"])
    disc, val, hold = evs[:int(n*.6)], evs[int(n*.6):int(n*.8)], evs[int(n*.8):]
    bd, bv, bh = stat(disc), stat(val), stat(hold)
    print(f"切分：D n={bd['n']}（基线 {bd['p']:.1%}）/ V n={bv['n']}（{bv['p']:.1%}）"
          f"/ H n={bh['n']}（{bh['p']:.1%}）")

    med = float(np.median([e["retrace"] for e in disc if e["retrace"] > 0])) \
        if any(e["retrace"] > 0 for e in disc) else 0.0
    hyp = [
        ("H1 retrace>0", lambda e: e["retrace"] > 0),
        ("H1' retrace=0(对照组)", lambda e: e["retrace"] <= 0),
        ("H2 last_step<0", lambda e: e["last_step"] < 0),
        ("H2' last_step>=0(对照)", lambda e: e["last_step"] >= 0),
        (f"H3 retrace>={med:.2f}bp", lambda e: e["retrace"] >= med),
        ("H1×H2 双确认", lambda e: e["retrace"] > 0 and e["last_step"] < 0),
        ("H4 depth<=0(尖峰失败)", lambda e: e["depth"] <= 0),
        ("H4' depth>0(对照)", lambda e: e["depth"] > 0),
    ]
    print(f"\nDiscovery 回撤中位数（retrace>0 内）= {med:.2f} bp")
    print(f"\n{'假设':<22}{'D n':>5} {'D胜率':>7} {'lift':>7} {'EV':>8} | "
          f"{'V胜率(n)':>10} {'H胜率(n)':>10} | 日跑赢")
    verdicts = {}
    for hname, fn in hyp:
        sd = [e for e in disc if fn(e)]
        sv = [e for e in val if fn(e)]
        sh = [e for e in hold if fn(e)]
        s_d, s_v, s_h = stat(sd), stat(sv), stat(sh)
        if not s_d or s_d["n"] < MIN_N:
            print(f"{hname:<22}{len(sd):>5}  （样本不足 n<{MIN_N}，保留负结果，不判定）")
            verdicts[hname] = "INSUFFICIENT"
            continue
        lift = s_d["p"] - bd["p"]
        pv = f"{s_v['p']:.0%}({s_v['n']})" if s_v else "—"
        ph = f"{s_h['p']:.0%}({s_h['n']})" if s_h else "—"
        bt, tt = daily_beat(evs, fn)
        lv = (s_v["p"] - bv["p"]) if s_v and s_v["n"] >= 20 else None
        lh = (s_h["p"] - bh["p"]) if s_h and s_h["n"] >= 20 else None
        if lv is not None and lh is not None and lv >= 0.03 and lh >= 0.03 and lift >= 0.03:
            verdicts[hname] = "PROMISING"
        elif lv is not None and lh is not None and lv > 0 and lh > 0 and lift > 0:
            verdicts[hname] = "WEAK"
        else:
            verdicts[hname] = "REJECT"
        print(f"{hname:<22}{s_d['n']:>5} {s_d['p']:>6.1%} {lift:>+6.1%} "
              f"{s_d['ev']:>+7.3f} | {pv:>10} {ph:>10} | {bt}/{tt}  "
              f"[W {s_d['lo']:.0%},{s_d['hi']:.0%}]  {verdicts[hname]}")

    print("\n===== Verdict 汇总 =====")
    for hname, v in verdicts.items():
        print(f"  {hname}: {v}")
    print("\n（样本仅 329 事件，任何判定上限为 PROMISING；负结果已保留。）")
    with open("output/cell_b_retrace_result.json", "w", encoding="utf-8") as f:
        json.dump({"n": n, "base": base, "median_retrace_bp": med,
                   "verdicts": verdicts}, f, ensure_ascii=False, indent=2,
                  default=float)
    print("结果已存 output/cell_b_retrace_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
