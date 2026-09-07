#!/usr/bin/env python3
"""5m DOWN 首触信号的维度增益扫描（预注册 → calib 扫描 → confirm 终验一次）。

背景
----
local_firsthit_qdepth_grid.py 圈出的边际信号：5m 市场，DOWN 报价首次进入
(0.005, 0.1] 时买入 DOWN，全期 EV=+0.22 [+0.10,+0.35]（n=5608，44 天），
calib / confirm 两段都为正（+0.20 / +0.26）。本轮在该基底上测试：加哪些
过滤维度能把 P(反转) 从 8.9% 再抬高、EV 更稳。

预注册规则（运行前锁定；confirm 段只终验一次）
----------------------------------------------
基底   ：5m DOWN 首触事件（0.005 < q ≤ 0.1，每窗口每侧 1 条）
切分   ：按 UTC 日，前 2/3 为 calib（< 2026-08-24），后 1/3 为 confirm（≥）
门     ：10 个连续维度 × 3 分位 = 30 + streak 4 档 + hour 4 档 = 38
        + 预注册组合 chg×vol 3×3 = 9 → 共 47 个门
候选   ：calib n ≥ 60（组合格 ≥ 40）、覆盖天数 ≥ 8（格 ≥ 6）、
        P ≥ 基底 calib P + 3pp（格 +4pp）、EV 日聚类 CI 下界 > 0
通过   ：严格 = confirm EV 日聚类 CI 下界 > 0
        宽松 = confirm EV > 0 且 CI 下界 > −0.10
分位切点与 misalign 回归系数只用 calib 拟合（防泄漏）。
多重比较：47 门 × 0.05 ≈ 期望 2.4 个假阳性；headline 只认 calib→confirm 路径。

维度（全部为触发时刻可观测，无未来函数）
----------------------------------------
q            触发报价
t            触发时刻（窗口内秒数）
chg_bps      触发时 BTC 相对本窗开盘涨跌（负=已跌）→「要爬的距离」
misalign     q 对 chg 的线性回归残差（calib 拟合）→ 同跌幅下报价异常便宜
vol1h_bps    前 12 根 5m K 对数收益 std（bps）→ 波动率 regime
participants 触发样本参与人数
q15_div      覆盖同刻的 15m 市场 DOWN 报价 − 本触发报价（15m 数据 08-13 起）
speed_bps_s  |chg| / t → 尖峰砸下 vs 阴跌爬行
qdrop30      触发前 ≥20s 的最近报价 − 触发报价 → 报价下坠速度
spread       up + down − 1（点差）
streak       此前连续 DOWN 收盘的窗口数（0 / 1 / 2 / ≥3）
hour         UTC 时段（0-5 / 6-11 / 12-17 / 18-23）

用法
----
    .venv\\Scripts\\python.exe scripts/local_dim_gain_scan.py
"""
from __future__ import annotations

import bisect
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
Q_LO, Q_HI = 0.005, 0.1
SAMPLES_5M = "output/pull_samples_5m_20260907.json"
SAMPLES_15M = "output/pull_samples_15m_20260907.json"
KL5_FILES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
LOG = "output/dim_gain_scan.log"
OUT = "output/dim_gain_scan.json"

BOOT_B = 4000
SEED = 20260907
MIN_N_GATE = 60          # 单维门 calib 样本门槛
MIN_N_CELL = 40          # 组合格门槛
MIN_DAYS_GATE = 8
MIN_DAYS_CELL = 6
P_LIFT_GATE = 0.03       # 单维：比基底 calib P 高 3pp
P_LIFT_CELL = 0.04       # 组合：高 4pp

TER_DIMS = [
    ("q", "触发报价 q"),
    ("t", "触发时刻 t(s)"),
    ("chg_bps", "BTC 相对开盘涨跌(bps，负=已跌)"),
    ("misalign", "q 对 chg 回归残差(负=比同跌幅更便宜)"),
    ("vol1h_bps", "前 1h 波动率(bps/5m bar)"),
    ("participants", "触发时参与人数"),
    ("q15_div", "15m DOWN 报价 − 5m 触发报价"),
    ("speed_bps_s", "下砸速度 |chg|/t (bps/s)"),
    ("qdrop30", "前 ~30s 报价跌幅"),
    ("spread", "点差 up+down−1"),
]


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


def day_boot(values: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    d_u = np.unique(days)
    if len(d_u) < 5:
        return (float("nan"), float("nan"))
    sums = np.zeros(len(d_u)); cnts = np.zeros(len(d_u))
    idx = np.searchsorted(d_u, days)
    np.add.at(sums, idx, values)
    np.add.at(cnts, idx, 1)
    means = sums / cnts
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(d_u), size=(BOOT_B, len(d_u)))
    w = cnts[pick]
    tot = (means[pick] * w).sum(axis=1) / w.sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def stats(evs: list[dict]) -> dict | None:
    if not evs:
        return None
    flags = np.array([e["win"] for e in evs], float)
    qs = np.array([e["q"] for e in evs], float)
    days = np.array([e["day"] for e in evs])
    n = len(evs); k = int(flags.sum())
    evv = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
    lo, hi = day_boot(evv, days)
    return {"n": n, "k": k, "p": k / n, "ev": float(evv.mean()),
            "ev_lo": lo, "ev_hi": hi, "qbar": float(qs.mean()),
            "n_days": int(len(np.unique(days))), "per_day": n / len(np.unique(days))}


def fmt(st: dict | None) -> str:
    if st is None:
        return "n=0"
    lo, hi = st["ev_lo"], st["ev_hi"]
    ci = "CI[nan]" if math.isnan(lo) else f"CI[{lo:+.2f},{hi:+.2f}]"
    return (f"n={st['n']:>4} P={st['p']:>5.1%} EV={st['ev']:+.2f} {ci} "
            f"q̄={st['qbar']:.3f} {st['n_days']}天")


def load_k5() -> dict[int, tuple[float, float]]:
    rows = []
    for p in KL5_FILES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    return {int(k[0]): (float(k[1]), float(k[4])) for k in rows}


def collect_events(k5: dict[int, tuple[float, float]]) -> list[dict]:
    """5m DOWN 首触事件 + 触发时刻可观测特征。"""
    raw = json.load(open(SAMPLES_5M, encoding="utf-8"))
    L = 300_000
    by: dict[int, list[dict]] = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)

    # 15m 报价数组（nearest 查询用）
    r15 = json.load(open(SAMPLES_15M, encoding="utf-8"))
    r15 = [s for s in r15 if s.get("down_price") is not None]
    r15.sort(key=lambda s: int(s["timestamp"]))
    ts15 = [int(s["timestamp"]) for s in r15]
    q15dn = [float(s["down_price"]) for s in r15]

    def q15_at(ts: int) -> float:
        i = bisect.bisect_left(ts15, ts)
        best, bd = None, 45_000
        for j in (i - 1, i):
            if 0 <= j < len(ts15):
                d = abs(ts15[j] - ts)
                if d <= bd:
                    best, bd = q15dn[j], d
        return float("nan") if best is None else best

    events: list[dict] = []
    n_no_btc = 0
    for w, rows in sorted(by.items()):
        bar = k5.get(w)
        if bar is None or bar[0] <= 0:
            continue
        o, c = bar
        if o == c:
            continue
        win = 1 if c < o else 0          # DOWN 获胜
        rows.sort(key=lambda s: int(s["timestamp"]))
        # 首触：首个 down_price ∈ (0.005, 0.1]
        trig = None
        for s in rows:
            q = s.get("down_price")
            if q is not None and Q_LO < float(q) <= Q_HI:
                trig = s
                break
        if trig is None:
            continue
        ts = int(trig["timestamp"])
        q = float(trig["down_price"])
        day = time.strftime("%Y-%m-%d", time.gmtime(w / 1000))
        t = (ts - w) / 1000.0

        # btc at trigger（±30s 内就近补缺）
        btc = trig.get("btc_price")
        if btc is None:
            ti = rows.index(trig)
            for j in (ti - 1, ti + 1, ti - 2, ti + 2):
                if 0 <= j < len(rows) and rows[j].get("btc_price"):
                    btc = rows[j]["btc_price"]
                    break
        if btc:
            chg = (float(btc) / o - 1.0) * 10_000.0
        else:
            chg = float("nan")
            n_no_btc += 1

        # vol1h：前 12 根 5m K 的对数收益 std
        closes = [k5.get(w - i * L, (None, None))[1] for i in range(1, 13)]
        if all(x is not None for x in closes) and closes[-1] > 0:
            lr = np.diff(np.log(np.array([float(x) for x in closes])))
            vol = float(np.std(lr) * 10_000.0)
        else:
            vol = float("nan")

        # streak：此前连续 DOWN 收盘窗数（封顶 3）
        st = 0
        b = w - L
        while st < 3:
            bb = k5.get(b)
            if bb and bb[1] < bb[0]:
                st += 1
                b -= L
            else:
                break

        # qdrop30：触发前 ≥20s 最近一条 down_price
        qd = float("nan")
        for s in rows:
            if int(s["timestamp"]) > ts - 20_000:
                break
            if s.get("down_price") is not None:
                qd = float(s["down_price"])
        if not math.isnan(qd):
            qd -= q

        up = trig.get("up_price")
        spread = (float(up) + q - 1.0) if up is not None else float("nan")
        part = trig.get("participants")
        events.append({
            "w": w, "ts": ts, "t": t, "q": q, "win": win, "day": day,
            "chg_bps": chg, "vol1h_bps": vol,
            "participants": float(part) if part is not None else float("nan"),
            "q15_div": q15_at(ts) - q, "speed_bps_s": abs(chg) / max(t, 1.0)
            if not math.isnan(chg) else float("nan"),
            "qdrop30": qd, "spread": spread, "streak": st,
            "hour": time.gmtime(ts / 1000).tm_hour, "misalign": float("nan"),
        })
    print(f"基底事件 {len(events)} 条；btc 缺失（chg 相关维度不可用）{n_no_btc} 条")
    return events


def main() -> int:
    sys.stdout = Tee(LOG)
    print("=" * 112)
    print("5m DOWN 首触 · 维度增益扫描（预注册 47 门：38 单维 + chg×vol 9 格）")
    print("=" * 112)
    k5 = load_k5()
    events = collect_events(k5)

    days = sorted({e["day"] for e in events})
    split = days[int(len(days) * 2 / 3)]
    calib = [e for e in events if e["day"] < split]
    confirm = [e for e in events if e["day"] >= split]
    base_c, base_f = stats(calib), stats(confirm)
    print(f"\n切分日 {split} | calib {len(calib)} 事件 | confirm {len(confirm)} 事件")
    print(f"基底 calib : {fmt(base_c)}")
    print(f"基底 confirm: {fmt(base_f)}")
    p_base = base_c["p"]

    # ---- misalign：calib 拟合 q ~ chg ----
    xs = np.array([e["chg_bps"] for e in calib if not math.isnan(e["chg_bps"])])
    ys = np.array([e["q"] for e in calib if not math.isnan(e["chg_bps"])])
    b1, a1 = np.polyfit(xs, ys, 1)  # q = a1 + b1*chg
    print(f"\nmisalign 拟合（calib）: q = {a1:.5f} + {b1:.6f}×chg_bps "
          f"(n={len(xs)}，相关 {np.corrcoef(xs, ys)[0, 1]:+.3f})")
    for e in events:
        if not math.isnan(e["chg_bps"]):
            e["misalign"] = e["q"] - (a1 + b1 * e["chg_bps"])

    # ---- 单维三分位表 ----
    results: list[dict] = []
    print("\n" + "=" * 112)
    print("### 单维三分位（切点用 calib 拟合；★=calib 候选门）")
    for key, label in TER_DIMS:
        vals = [e[key] for e in calib if not math.isnan(e[key])]
        if len(vals) < MIN_N_GATE:
            print(f"\n── {label}：calib 可用样本 {len(vals)} < {MIN_N_GATE}，跳过")
            continue
        c1, c2 = (float(x) for x in np.nanquantile(
            [e[key] for e in calib if not math.isnan(e[key])], [1 / 3, 2 / 3]))
        print(f"\n── {label}｜calib 三分位切点 [{c1:.4g}, {c2:.4g}]")
        for lname, cond in (
            ("T1", lambda v: v <= c1), ("T2", lambda v: c1 < v <= c2),
            ("T3", lambda v: v > c2),
        ):
            ec = [e for e in calib if not math.isnan(e[key]) and cond(e[key])]
            ef = [e for e in confirm if not math.isnan(e[key]) and cond(e[key])]
            sc, sf = stats(ec), stats(ef)
            cand = (sc and sc["n"] >= MIN_N_GATE and sc["n_days"] >= MIN_DAYS_GATE
                    and sc["p"] >= p_base + P_LIFT_GATE
                    and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0)
            line = f"  {lname}{'★' if cand else ' '} {label[:2]:<3}: calib {fmt(sc)}"
            line += f" | confirm {fmt(sf)}"
            print(line)
            if cand:
                verdict = "FAIL"
                if sf and not math.isnan(sf["ev_lo"]):
                    if sf["ev_lo"] > 0:
                        verdict = "STRICT_PASS"
                    elif sf["ev"] > 0 and sf["ev_lo"] > -0.10:
                        verdict = "LOOSE_PASS"
                print(f"      → 候选门终验：{verdict}")
                results.append({"dim": key, "level": lname, "cut": [c1, c2],
                                "calib": sc, "confirm": sf, "verdict": verdict})

    # ---- 分类维度：streak / hour ----
    print("\n── streak（此前连续 DOWN 收盘窗数）")
    for lv in (0, 1, 2, 3):
        ec = [e for e in calib if e["streak"] == lv]
        ef = [e for e in confirm if e["streak"] == lv]
        sc, sf = stats(ec), stats(ef)
        cand = (sc and sc["n"] >= MIN_N_GATE and sc["n_days"] >= MIN_DAYS_GATE
                and sc["p"] >= p_base + P_LIFT_GATE
                and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0)
        nm = f"≥{lv}" if lv == 3 else str(lv)
        print(f"  {nm}{'★' if cand else ' '}: calib {fmt(sc)} | confirm {fmt(sf)}")
        if cand:
            verdict = "FAIL"
            if sf and not math.isnan(sf["ev_lo"]):
                if sf["ev_lo"] > 0:
                    verdict = "STRICT_PASS"
                elif sf["ev"] > 0 and sf["ev_lo"] > -0.10:
                    verdict = "LOOSE_PASS"
            print(f"      → 候选门终验：{verdict}")
            results.append({"dim": "streak", "level": nm, "calib": sc,
                            "confirm": sf, "verdict": verdict})

    print("\n── hour（UTC 时段）")
    for lo, hi in ((0, 5), (6, 11), (12, 17), (18, 23)):
        ec = [e for e in calib if lo <= e["hour"] <= hi]
        ef = [e for e in confirm if lo <= e["hour"] <= hi]
        sc, sf = stats(ec), stats(ef)
        cand = (sc and sc["n"] >= MIN_N_GATE and sc["n_days"] >= MIN_DAYS_GATE
                and sc["p"] >= p_base + P_LIFT_GATE
                and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0)
        print(f"  {lo:>2}-{hi:<2}{'★' if cand else ' '}: calib {fmt(sc)} | confirm {fmt(sf)}")
        if cand:
            verdict = "FAIL"
            if sf and not math.isnan(sf["ev_lo"]):
                if sf["ev_lo"] > 0:
                    verdict = "STRICT_PASS"
                elif sf["ev"] > 0 and sf["ev_lo"] > -0.10:
                    verdict = "LOOSE_PASS"
            print(f"      → 候选门终验：{verdict}")
            results.append({"dim": "hour", "level": f"{lo}-{hi}", "calib": sc,
                            "confirm": sf, "verdict": verdict})

    # ---- 预注册组合：chg × vol 3×3 ----
    print("\n" + "=" * 112)
    print("### 预注册组合 chg × vol（3×3，格子门槛 n≥40/天≥6/+4pp）")
    cvals = [e["chg_bps"] for e in calib if not math.isnan(e["chg_bps"])]
    vvals = [e["vol1h_bps"] for e in calib if not math.isnan(e["vol1h_bps"])]
    cc1, cc2 = (float(x) for x in np.quantile(cvals, [1 / 3, 2 / 3]))
    vc1, vc2 = (float(x) for x in np.quantile(vvals, [1 / 3, 2 / 3]))
    print(f"chg 切点 [{cc1:.1f}, {cc2:.1f}] bps | vol 切点 [{vc1:.1f}, {vc2:.1f}] bps")

    def chg_lv(v):
        return 1 if v <= cc1 else (2 if v <= cc2 else 3)

    def vol_lv(v):
        return 1 if v <= vc1 else (2 if v <= vc2 else 3)

    combo_results = []
    for cl in (1, 2, 3):
        row = []
        for vl in (1, 2, 3):
            ec = [e for e in calib if not math.isnan(e["chg_bps"])
                  and not math.isnan(e["vol1h_bps"])
                  and chg_lv(e["chg_bps"]) == cl and vol_lv(e["vol1h_bps"]) == vl]
            ef = [e for e in confirm if not math.isnan(e["chg_bps"])
                  and not math.isnan(e["vol1h_bps"])
                  and chg_lv(e["chg_bps"]) == cl and vol_lv(e["vol1h_bps"]) == vl]
            sc, sf = stats(ec), stats(ef)
            cand = (sc and sc["n"] >= MIN_N_CELL and sc["n_days"] >= MIN_DAYS_CELL
                    and sc["p"] >= p_base + P_LIFT_CELL
                    and not math.isnan(sc["ev_lo"]) and sc["ev_lo"] > 0)
            row.append((sc, sf, cand))
            tag = "★" if cand else " "
            s = (f"C{cl}V{vl}{tag} calib {fmt(sc)}" if sc else f"C{cl}V{vl} n=0")
            if cand:
                verdict = "FAIL"
                if sf and not math.isnan(sf["ev_lo"]):
                    if sf["ev_lo"] > 0:
                        verdict = "STRICT_PASS"
                    elif sf["ev"] > 0 and sf["ev_lo"] > -0.10:
                        verdict = "LOOSE_PASS"
                s += f" → {verdict}"
                combo_results.append({"chg_lv": cl, "vol_lv": vl, "calib": sc,
                                      "confirm": sf, "verdict": verdict})
            print(f"  {s}")
            if sf:
                print(f"      confirm {fmt(sf)}")

    # ---- 汇总 ----
    print("\n" + "=" * 112)
    print("### 汇总")
    n_gates = 38 + 9
    print(f"扫描门数 {n_gates}（38 单维 + 9 组合）→ 纯噪音期望假阳性 ≈ {0.05 * n_gates:.1f}")
    strict = [r for r in results + combo_results if r["verdict"] == "STRICT_PASS"]
    loose = [r for r in results + combo_results if r["verdict"] == "LOOSE_PASS"]
    print(f"calib 候选 {len(results) + len(combo_results)} 个 → "
          f"严格通过 {len(strict)}，宽松通过 {len(loose)}，"
          f"失败 {len(results) + len(combo_results) - len(strict) - len(loose)}")
    for r in strict + loose:
        d = r.get("dim") or f"chg×vol(C{r['chg_lv']}V{r['vol_lv']})"
        print(f"  [{r['verdict']}] {d} {r.get('level', '')}: "
              f"calib P={r['calib']['p']:.1%} EV={r['calib']['ev']:+.2f} | "
              f"confirm P={r['confirm']['p']:.1%} EV={r['confirm']['ev']:+.2f} "
              f"CI[{r['confirm']['ev_lo']:+.2f},{r['confirm']['ev_hi']:+.2f}]")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"split_day": split, "n_gates": n_gates,
                            "base_calib": base_c, "base_confirm": base_f,
                            "misalign_fit": {"a": a1, "b": b1}},
                   "candidates": [{**r, "calib": r["calib"], "confirm": r["confirm"]}
                                  for r in results + combo_results]},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果 JSON → {OUT} | 日志 → {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
