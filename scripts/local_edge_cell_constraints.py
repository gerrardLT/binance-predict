#!/usr/bin/env python3
"""两个 edge 格子的约束强化（科学假设流程）。

目标格子（5m 市场，DOWN 方向，24 天真实配对）：
  Cell A 顺势：t∈[90,120)s × 报价 q∈[0.65,0.75)  基线胜率 ~75%，EV ~+0.06
  Cell B 逆势：t∈[45,60)s  × 报价 q∈[0.15,0.25)  基线胜率 ~29%，EV ~+0.39

流程（kline-scientific-discovery 纪律）：
  1. 事件池构建（每周期首个命中报价，win = 周期收盘 < 开盘）
  2. 日频率统计
  3. 决策时刻可得特征（禁用未来信息）：
     depth（当前 btc 相对开盘 log 幅度）/ prev_ret / prev2_ret / streak /
     pos12（开盘价在过去 12 周期 close 区间位置）/ vol12（前 12 周期平均振幅）/
     hour_utc / q 桶内位置
  4. 时序切分 60/20/20（Discovery/Validation/Holdout）；阈值仅由 Discovery 分位数生成
  5. L1 单因子：P(win|约束) vs 事件基线，Wilson CI + EV；
     Validation 与 Holdout 必须同方向才保留（holdout 只验不改）
  6. L2：存活 L1 因子两两组合（同一底层特征不得双阈值）
  7. 每个存活约束报日频率

口径：EV = 赢 0.98/q−1 / 输 −1（费 2%）。
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
LOG = "output/edge_cell_constraints.log"
CELLS = {
    "A_momentum": {"t_lo": 90, "t_hi": 120, "q_lo": 0.65, "q_hi": 0.75},
    "B_contrarian": {"t_lo": 45, "t_hi": 60, "q_lo": 0.15, "q_hi": 0.25},
}
N_DAYS = 24.0
MIN_N_DISC = 60          # Discovery 最小样本（24 天样本有限，声明在案）
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
            "ev": float(np.mean(pnl)), "q": float(np.mean([e["q"] for e in evs])),
            "freq_day": n / N_DAYS}


def daily_stability(evs: list[dict], fn) -> tuple[int, int, float]:
    """约束子集的日胜率稳定性：返回 (跑赢报价 q̄ 的天数, 总天数, 最大连亏天数)。"""
    sub = [e for e in evs if fn(e)]
    if not sub:
        return 0, 0, 0.0
    days: dict[int, list] = defaultdict(list)
    for e in sub:
        days[e["ts"] // 86_400_000].append(e)
    beat, total, worst, cur = 0, 0, 0, 0
    for d in sorted(days):
        dd = days[d]
        if len(dd) < 3:
            continue
        total += 1
        p = sum(e["win"] for e in dd) / len(dd)
        q = float(np.mean([e["q"] for e in dd]))
        if p > q:
            beat += 1; cur = 0
        else:
            cur += 1; worst = max(worst, cur)
    return beat, total, worst


def load_klines():
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    now_ms = int(time.time() * 1000)
    rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]))
            for k in kl]
    if rows and rows[-1][0] + 300_000 > now_ms:
        rows.pop()
    idx = {r[0] // 300_000: i for i, r in enumerate(rows)}
    return rows, idx


def build_events(rows, idx, cell: dict) -> list[dict]:
    """每周期首个命中报价 → 事件（含决策时刻特征）。"""
    with open(SAMPLES, encoding="utf-8") as f:
        S = json.load(f)
    s5 = sorted((s for s in S if s.get("market_period") == "5m"),
                key=lambda s: int(s["timestamp"]))
    hit_cyc: set[int] = set()
    evs = []
    for s in s5:
        q = s.get("down_price"); ts = s.get("timestamp"); btc = s.get("btc_price")
        if q is None or ts is None or btc is None or q <= 0.01 or q >= 0.99:
            continue
        q = float(q)
        if not (cell["q_lo"] <= q < cell["q_hi"]):
            continue
        ts = int(ts)
        cyc = ts // 300_000
        if cyc in hit_cyc:
            continue
        t_sec = ts - cyc * 300_000
        if t_sec < cell["t_lo"] * 1000 or t_sec >= cell["t_hi"] * 1000:
            continue
        i = idx.get(cyc)
        if i is None or i < 13:
            continue
        o, c = rows[i][1], rows[i][4]
        if o <= 0:
            continue
        hit_cyc.add(cyc)
        # ---- 决策时刻特征（全部 ≤ ts 可得）----
        po, pc = rows[i - 1][1], rows[i - 1][4]        # 前周期
        p2o, p2c = rows[i - 2][1], rows[i - 2][4]
        streak = 0
        d_prev = 1 if pc > po else -1
        for j in range(i - 1, max(0, i - 9), -1):
            d = 1 if rows[j][4] > rows[j][1] else -1
            if d == d_prev:
                streak += 1
            else:
                break
        closes12 = [rows[j][4] for j in range(i - 12, i)]
        hi12, lo12 = max(closes12), min(closes12)
        pos12 = (o - lo12) / (hi12 - lo12) if hi12 > lo12 else 0.5
        vol12 = float(np.mean([abs(rows[j][4] - rows[j][1]) / rows[j][1]
                               for j in range(i - 12, i)]))
        evs.append({
            "ts": ts, "q": q, "win": c < o,
            "depth": math.log(float(btc) / o) * 1e4,              # bp，决策时 btc 位置
            "qdepth": abs(q - 0.5) * 100,                         # 报价隐含方向强度
            "agree": 1 if ((q > 0.5) == (float(btc) < o)) else 0, # 报价方向与 btc 位置同向
            "prev_ret": math.log(pc / po) * 1e4 if po > 0 else 0.0,
            "prev2_ret": math.log(p2c / p2o) * 1e4 if p2o > 0 else 0.0,
            "prev_dir": 1 if pc > po else 0,
            "streak": streak * d_prev,                            # 带方向连击
            "pos12": pos12, "vol12": vol12 * 1e4,
            "hour": datetime.fromtimestamp(ts / 1000, timezone.utc).hour,
        })
    return evs


def split3(evs: list[dict]) -> tuple[list, list, list]:
    evs = sorted(evs, key=lambda e: e["ts"])
    n = len(evs)
    return evs[: int(n * .6)], evs[int(n * .6): int(n * .8)], evs[int(n * .8):]


# ---- L1 因子定义：name -> (feature_fn, 阈值方向说明) ----
def make_factors(evs_disc: list[dict]) -> list[tuple[str, callable]]:
    qs = lambda key, ps: [float(np.percentile([e[key] for e in evs_disc], p))
                          for p in ps]
    d_q = qs("depth", (25, 50, 75))
    pr_q = qs("prev_ret", (25, 50, 75))
    v_q = qs("vol12", (33, 66))
    F = []
    for t in d_q:
        F.append((f"depth<=<{t:.1f}bp", lambda e, t=t: e["depth"] <= t))
        F.append((f"depth>={t:.1f}bp", lambda e, t=t: e["depth"] >= t))
    for t in pr_q:
        F.append((f"prev_ret<=<{t:.1f}", lambda e, t=t: e["prev_ret"] <= t))
        F.append((f"prev_ret>={t:.1f}", lambda e, t=t: e["prev_ret"] >= t))
    F.append(("prev_dir=DOWN", lambda e: e["prev_dir"] == 0))
    F.append(("prev_dir=UP", lambda e: e["prev_dir"] == 1))
    F.append(("streak<=-2(连阴)", lambda e: e["streak"] <= -2))
    F.append(("streak>=2(连阳)", lambda e: e["streak"] >= 2))
    for t in (0.25, 0.5, 0.75):
        F.append((f"pos12<{t:.2f}", lambda e, t=t: e["pos12"] < t))
        F.append((f"pos12>={t:.2f}", lambda e, t=t: e["pos12"] >= t))
    for t in v_q:
        F.append((f"vol12<{t:.1f}", lambda e, t=t: e["vol12"] < t))
        F.append((f"vol12>={t:.1f}", lambda e, t=t: e["vol12"] >= t))
    F.append(("hour∈0-7(亚)", lambda e: 0 <= e["hour"] < 8))
    F.append(("hour∈8-15(欧)", lambda e: 8 <= e["hour"] < 16))
    F.append(("hour∈16-23(美)", lambda e: 16 <= e["hour"] < 24))
    a_q = np.percentile([e["qdepth"] for e in evs_disc], 50)
    F.append((f"qdepth>={a_q:.1f}", lambda e, t=float(a_q): e["qdepth"] >= t))
    F.append((f"qdepth<{a_q:.1f}", lambda e, t=float(a_q): e["qdepth"] < t))
    F.append(("agree=同向", lambda e: e["agree"] == 1))
    F.append(("agree=背离", lambda e: e["agree"] == 0))
    return F


def main() -> int:
    sys.stdout = Tee()
    rows, idx = load_klines()
    print(f"K 线 {len(rows)} 根；样本文件载入中…")

    results = {}
    for cname, cell in CELLS.items():
        evs = build_events(rows, idx, cell)
        if not evs:
            print(f"\n{cname}: 无事件"); continue
        base_all = stat(evs)
        disc, val, hold = split3(evs)
        base_d = stat(disc)
        print(f"\n{'='*72}\n{cname}  t∈[{cell['t_lo']},{cell['t_hi']})s × q∈[{cell['q_lo']},{cell['q_hi']})")
        print(f"事件总数 {base_all['n']}（日均 {base_all['freq_day']:.1f} 次）| "
              f"全期胜率 {base_all['p']:.1%} [W {base_all['lo']:.0%},{base_all['hi']:.0%}] "
              f"q̄={base_all['q']:.3f} EV={base_all['ev']:+.3f}")
        print(f"切分：Discovery n={len(disc)}（基线 {base_d['p']:.1%}）/ "
              f"Validation n={len(val)} / Holdout n={len(hold)}")

        # ---- L1 ----
        factors = make_factors(disc)
        base_v, base_h = stat(val), stat(hold)
        l1_pos, l1_neg = [], []
        print(f"\n  --- L1 单因子（Discovery 基线 {base_d['p']:.1%}；n≥{MIN_N_DISC}）---")
        print(f"  Val 基线 {base_v['p']:.1%}(n={base_v['n']}) / Hold 基线 {base_h['p']:.1%}(n={base_h['n']})")
        print(f"  {'约束':<20}{'n':>5} {'胜率':>7} {'lift':>7} {'EV':>8} {'日频':>6} | Val胜率(n) | Hold胜率(n)")
        for fname, fn in factors:
            sd = [e for e in disc if fn(e)]
            if len(sd) < MIN_N_DISC:
                continue
            s_d = stat(sd)
            sv = [e for e in val if fn(e)]
            sh = [e for e in hold if fn(e)]
            s_v, s_h = stat(sv), stat(sh)
            lift = s_d["p"] - base_d["p"]
            pv = f"{s_v['p']:.0%}({s_v['n']})" if s_v else "—"
            ph = f"{s_h['p']:.0%}({s_h['n']})" if s_h else "—"
            # 存活：Discovery |lift|≥3pp；Val/Hold 相对各自基线同方向且幅度≥2pp；
            # 候选(△)：幅度 1~2pp，展示但不进 L2
            mark = " "
            if abs(lift) >= 0.03:
                lv = (s_v["p"] - base_v["p"]) if s_v and s_v["n"] >= 30 else None
                lh = (s_h["p"] - base_h["p"]) if s_h and s_h["n"] >= 30 else None
                if lv is not None and lh is not None:
                    if lift > 0 and lv >= 0.02 and lh >= 0.02:
                        l1_pos.append((fname, fn, lift)); mark = "✓+"
                    elif lift < 0 and lv <= -0.02 and lh <= -0.02:
                        l1_neg.append((fname, fn, lift)); mark = "✓-"
                    elif lift > 0 and lv >= 0.01 and lh >= 0.01:
                        mark = "△+"
                    elif lift < 0 and lv <= -0.01 and lh <= -0.01:
                        mark = "△-"
            print(f"  {fname:<20}{s_d['n']:>5} {s_d['p']:>6.1%} {lift:>+6.1%} "
                  f"{s_d['ev']:>+7.3f} {s_d['freq_day']:>5.1f} | {pv:>10} | {ph:>10} {mark}")
        print(f"  L1 存活：正向强化 {len(l1_pos)} 个 / 负向规避 {len(l1_neg)} 个")

        # ---- 日稳定性审计：无约束基线 + 存活/候选约束 ----
        print(f"\n  --- 日稳定性审计（每日胜率 vs 当日报价 q̄，天数≥3 注才计）---")
        bt, tt, ws = daily_stability(evs, lambda e: True)
        print(f"  基线（无约束）: 跑赢天数 {bt}/{tt}，最大连输 {ws} 天")
        for fname, fn, _ in l1_pos + l1_neg:
            bt, tt, ws = daily_stability(evs, fn)
            print(f"  {fname}: 跑赢天数 {bt}/{tt}，最大连输 {ws} 天")

        # ---- L2：正向因子交乘强化；负向因子交乘找毒区 ----
        def root(nm: str) -> str:
            return nm.split("<")[0].split(">")[0].split("=")[0].split("∈")[0]

        def l2_scan(pool, tag, min_lift, min_seg_n, min_n):
            out = []
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    n1, f1, _ = pool[i]; n2, f2, _ = pool[j]
                    if root(n1) == root(n2):
                        continue
                    sd = [e for e in disc if f1(e) and f2(e)]
                    if len(sd) < min_n:
                        continue
                    s_d = stat(sd)
                    sv = [e for e in val if f1(e) and f2(e)]
                    sh = [e for e in hold if f1(e) and f2(e)]
                    s_v, s_h = stat(sv), stat(sh)
                    lift = s_d["p"] - base_d["p"]
                    lv = (s_v["p"] - base_v["p"]) if s_v and s_v["n"] >= min_seg_n else None
                    lh = (s_h["p"] - base_h["p"]) if s_h and s_h["n"] >= min_seg_n else None
                    if lv is None or lh is None:
                        continue
                    ok = (lift * min_lift > 0 and abs(lift) >= abs(min_lift)
                          and lv * min_lift > 0 and lh * min_lift > 0)
                    if ok:
                        pv = f"{s_v['p']:.0%}({s_v['n']})"
                        ph = f"{s_h['p']:.0%}({s_h['n']})"
                        print(f"  {n1} × {n2}: D n={s_d['n']} {s_d['p']:.1%} "
                              f"lift{lift:+.1%} EV{s_d['ev']:+.3f} 日频{s_d['freq_day']:.1f} "
                              f"| Val {pv} | Hold {ph}")
                        out.append({"cond": f"{n1} & {n2}", **s_d})
            if not out:
                print(f"  （{tag}无存活组合）")
            return out

        print(f"\n  --- L2 正向强化组合 ---")
        l2p = l2_scan(l1_pos, "正向", 0.04, 15, 40)
        print(f"\n  --- L2 负向规避组合（胜率毒区）---")
        l2n = l2_scan(l1_neg, "负向", -0.04, 15, 40)
        results[cname] = {
            "base": base_all,
            "l1_pos": [n for n, _, _ in l1_pos],
            "l1_neg": [n for n, _, _ in l1_neg],
            "l2_pos": l2p, "l2_neg": l2n,
        }

    with open("output/edge_cell_constraints_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print("\n结果已存 output/edge_cell_constraints_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
