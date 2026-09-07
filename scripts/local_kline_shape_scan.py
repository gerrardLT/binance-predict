#!/usr/bin/env python3
"""5m DOWN 首触叠加 4 类微观 K 线形态特征扫描。

目标
----
验证在「5m 市场 DOWN 首次跌入 (0.005, 0.1]」基底下，叠加以下 4 个触发时刻可见的
微观 K 线特征，能否把胜率 P 从 8.9% 显著抬升、把 EV 进一步推高，并在 confirm 段站稳。

特征定义（严格保证触发时刻 ex-ante，无未来函数）
--------------------------------------------------
1. upper_wick_ratio（冲高回落/流星线特征）：
   窗口开盘至触发时刻这半根 K 线的上影线比例：
   (cum_high - max(btc_trig, open)) / (cum_high - cum_low)
   其中 cum_high/cum_low 是触发时刻前采样点里的最高/最低 btc_price。
   若 high == low 则为 0。该值越大，说明盘中曾大幅冲高但已被砸回。

2. body_ratio（实体/振幅比，衡量滞涨/小实体）：
   |btc_trig - open| / (cum_high - cum_low)
   该值越小，说明当前处于十字星/小纺锤线/冲高回落滞涨态；越接近 1 说明是大阳推进。

3. break_prev_high_bps（突破前驱高点幅度）：
   (cum_high - prev_3bar_high) / prev_3bar_high * 10000 (bps)
   正数 = 盘中曾刷新前 3 根 5m 的最高点（假突破/诱多测试）。

4. prev_bull_streak（前驱连阳数）：
   此前连续收阳的 5m K 线根数（0, 1, 2, >=3）。衡量多头惯性透支程度。

5. 交叉组合：
   结合上一轮验证通过的 chg <= 2.82 bps（BTC 微动但报价超跌），与形态特征交叉。

验证纪律
--------
- 前 2/3 天为 calib (< 2026-08-24)，后 1/3 天为 confirm (>= 2026-08-24)
- 逐事件真实赔率 EV，日聚类 bootstrap 95% CI
- 主候选标准：calib EV_lo > 0 且 n >= 40；confirm 段终验一次

用法
----
    .venv\\Scripts\\python.exe scripts/local_kline_shape_scan.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
Q_LO, Q_HI = 0.005, 0.1
SAMPLES_5M = "output/pull_samples_5m_20260907.json"
KL5_FILES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
LOG = "output/kline_shape_scan.log"
OUT = "output/kline_shape_scan.json"

BOOT_B = 4000
SEED = 20260907
MIN_N = 40
MIN_DAYS = 5


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
    if len(d_u) < MIN_DAYS:
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
            "n_days": int(len(np.unique(days)))}


def fmt(st: dict | None) -> str:
    if st is None or st["n"] == 0:
        return "n=0"
    lo, hi = st["ev_lo"], st["ev_hi"]
    ci = "CI[nan]" if math.isnan(lo) else f"CI[{lo:+.2f},{hi:+.2f}]"
    return (f"n={st['n']:>4} P={st['p']:>5.1%} EV={st['ev']:+.2f} {ci} "
            f"q̄={st['qbar']:.3f} {st['n_days']:>2}天")


def load_k5() -> dict[int, tuple[float, float, float, float]]:
    # ms -> (open, high, low, close)
    rows = []
    for p in KL5_FILES:
        rows.extend(json.load(open(p, encoding="utf-8")))
    m = {}
    for k in rows:
        m[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]))
    return m


def collect_shape_events(k5: dict[int, tuple[float, float, float, float]]) -> list[dict]:
    raw = json.load(open(SAMPLES_5M, encoding="utf-8"))
    L = 300_000
    by = defaultdict(list)
    for s in raw:
        by[int(s["timestamp"]) // L * L].append(s)

    events = []
    for w, rows in sorted(by.items()):
        bar = k5.get(w)
        if not bar or bar[0] <= 0:
            continue
        o, h_bar, l_bar, c = bar
        if o == c:
            continue
        win = 1 if c < o else 0
        rows.sort(key=lambda s: int(s["timestamp"]))

        # 找 DOWN 首触点
        trig = None
        ti = -1
        for i, s in enumerate(rows):
            q = s.get("down_price")
            if q is not None and Q_LO < float(q) <= Q_HI:
                trig = s
                ti = i
                break
        if trig is None:
            continue

        ts = int(trig["timestamp"])
        q = float(trig["down_price"])
        day = time.strftime("%Y-%m-%d", time.gmtime(w / 1000))
        t_sec = (ts - w) / 1000.0

        # 获取触发前及触发时的所有现货价格点（以窗开盘价 o 为起点）
        pts = [o]
        for s in rows[:ti + 1]:
            bp = s.get("btc_price")
            if bp and float(bp) > 0:
                pts.append(float(bp))
        if len(pts) <= 1:
            continue

        btc_trig = pts[-1]
        cum_hi = max(pts)
        cum_lo = min(pts)
        rng = cum_hi - cum_lo
        chg_bps = (btc_trig / o - 1.0) * 10_000.0

        # 特征 1: 上影线比例
        if rng > 1e-6:
            upper_wick = (cum_hi - max(btc_trig, o)) / rng
            body_r = abs(btc_trig - o) / rng
        else:
            upper_wick = 0.0
            body_r = 0.0

        # 特征 2: 突破前 3 根最高点幅度 (bps)
        prev_his = [k5.get(w - j * L, (None, None, None, None))[1] for j in (1, 2, 3)]
        if all(x is not None for x in prev_his):
            p3_hi = max(prev_his)
            break_hi_bps = (cum_hi / p3_hi - 1.0) * 10_000.0
        else:
            break_hi_bps = float("nan")

        # 特征 3: 前驱连阳数
        streak = 0
        cur_w = w - L
        while streak < 4:
            pb = k5.get(cur_w)
            if pb and pb[3] > pb[0]: # close > open 阳线
                streak += 1
                cur_w -= L
            else:
                break

        events.append({
            "w": w, "ts": ts, "t": t_sec, "q": q, "win": win, "day": day,
            "chg_bps": chg_bps,
            "upper_wick": upper_wick,
            "body_ratio": body_r,
            "break_hi_bps": break_hi_bps,
            "prev_bull_streak": streak,
        })
    return events


def eval_filter(name: str, cond_fn, calib: list[dict], confirm: list[dict]):
    ec = [e for e in calib if cond_fn(e)]
    ef = [e for e in confirm if cond_fn(e)]
    sc, sf = stats(ec), stats(ef)
    cand = (sc and sc["n"] >= MIN_N and sc["n_days"] >= MIN_DAYS and sc["ev_lo"] > 0)
    tag = "★" if cand else " "
    print(f"  {tag} {name:<42} | calib: {fmt(sc)} | confirm: {fmt(sf)}")
    return {"name": name, "calib": sc, "confirm": sf, "cand": cand}


def main():
    sys.stdout = Tee(LOG)
    print("=" * 115)
    print("5m DOWN 首触 · 4 类微观 K 线形态特征扫描（calib / confirm 时间严格终验）")
    print("=" * 115)

    k5 = load_k5()
    events = collect_shape_events(k5)
    print(f"提取出包含有效 K 线微观路径的首触事件: {len(events)} 条")

    days = sorted({e["day"] for e in events})
    split = days[int(len(days) * 2 / 3)]
    calib = [e for e in events if e["day"] < split]
    confirm = [e for e in events if e["day"] >= split]

    base_c, base_f = stats(calib), stats(confirm)
    print(f"切分日: {split} | calib {len(calib)} 条 ({base_c['n_days']}天) | confirm {len(confirm)} 条 ({base_f['n_days']}天)")
    print(f"基底 calib : {fmt(base_c)}")
    print(f"基底 confirm: {fmt(base_f)}")
    print("\n" + "=" * 115)

    # 1. 上影线比例 (Upper Wick Ratio) 三分位
    uw_vals = [e["upper_wick"] for e in calib]
    q_uw = np.quantile(uw_vals, [1/3, 2/3])
    print(f"### 一、上影线比例（冲高回落流星线）切点: [{q_uw[0]:.2f}, {q_uw[1]:.2f}]")
    eval_filter("W1: 弱上影 (<= %.2f)" % q_uw[0], lambda e: e["upper_wick"] <= q_uw[0], calib, confirm)
    eval_filter("W2: 中上影 (%.2f ~ %.2f)" % (q_uw[0], q_uw[1]), lambda e: q_uw[0] < e["upper_wick"] <= q_uw[1], calib, confirm)
    eval_filter("W3: 显著冲高回落 (> %.2f)" % q_uw[1], lambda e: e["upper_wick"] > q_uw[1], calib, confirm)
    eval_filter("W_ext: 极强长上影 (> 0.60)", lambda e: e["upper_wick"] > 0.60, calib, confirm)

    # 2. 实体振幅比 (Body Ratio)
    br_vals = [e["body_ratio"] for e in calib]
    q_br = np.quantile(br_vals, [1/3, 2/3])
    print(f"\n### 二、实体振幅比（滞涨吸收小实体）切点: [{q_br[0]:.2f}, {q_br[1]:.2f}]")
    eval_filter("B1: 极小实体/十字星 (<= %.2f)" % q_br[0], lambda e: e["body_ratio"] <= q_br[0], calib, confirm)
    eval_filter("B2: 中实体 (%.2f ~ %.2f)" % (q_br[0], q_br[1]), lambda e: q_br[0] < e["body_ratio"] <= q_br[1], calib, confirm)
    eval_filter("B3: 饱满推进大实体 (> %.2f)" % q_br[1], lambda e: e["body_ratio"] > q_br[1], calib, confirm)

    # 3. 突破前 3 根最高点幅度 (Break Margin bps)
    print(f"\n### 三、突破前驱 3 根高点（前高假突破/Sweep 猎杀）")
    eval_filter("H0: 未碰前高 (break <= 0 bps)", lambda e: not math.isnan(e["break_hi_bps"]) and e["break_hi_bps"] <= 0, calib, confirm)
    eval_filter("H1: 刺破假突破 (0 < break <= 10 bps)", lambda e: not math.isnan(e["break_hi_bps"]) and 0 < e["break_hi_bps"] <= 10, calib, confirm)
    eval_filter("H2: 强力突破前高 (> 10 bps)", lambda e: not math.isnan(e["break_hi_bps"]) and e["break_hi_bps"] > 10, calib, confirm)

    # 4. 前驱 5m 连阳数 (Bull Streak)
    print(f"\n### 四、多头惯性透支（前驱连续阳线）")
    eval_filter("S0: 前一根收阴 (streak = 0)", lambda e: e["prev_bull_streak"] == 0, calib, confirm)
    eval_filter("S1: 前一根单阳 (streak = 1)", lambda e: e["prev_bull_streak"] == 1, calib, confirm)
    eval_filter("S2: 前驱连阳2根 (streak = 2)", lambda e: e["prev_bull_streak"] == 2, calib, confirm)
    eval_filter("S3: 前驱连阳>=3根 (透支吸收)", lambda e: e["prev_bull_streak"] >= 3, calib, confirm)

    # 5. 黄金组合交叉：叠加 chg <= 2.8 bps (BTC 没真涨/假推)
    print("\n" + "=" * 115)
    print("### 五、形态 × 价格偏离（chg <= 2.8 bps）黄金交叉验证")
    eval_filter("基准参考: 仅 chg <= 2.8 bps", lambda e: e["chg_bps"] <= 2.82, calib, confirm)
    eval_filter("组合 A: chg<=2.8 且 长上影(>0.50)", lambda e: e["chg_bps"] <= 2.82 and e["upper_wick"] > 0.50, calib, confirm)
    eval_filter("组合 B: chg<=2.8 且 小实体(<=0.35)", lambda e: e["chg_bps"] <= 2.82 and e["body_ratio"] <= 0.35, calib, confirm)
    eval_filter("组合 C: chg<=2.8 且 刺破前高(0~10bps)", lambda e: e["chg_bps"] <= 2.82 and 0 < e["break_hi_bps"] <= 10, calib, confirm)
    eval_filter("组合 D: chg<=2.8 且 前驱连阳>=2根", lambda e: e["chg_bps"] <= 2.82 and e["prev_bull_streak"] >= 2, calib, confirm)
    eval_filter("组合 E: 长上影(>0.40) + 小实体(<=0.40)", lambda e: e["upper_wick"] > 0.40 and e["body_ratio"] <= 0.40, calib, confirm)

    print("\n" + "=" * 115)
    print("扫描完成，日志落盘至:", LOG)

if __name__ == "__main__":
    main()
