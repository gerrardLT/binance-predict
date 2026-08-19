#!/usr/bin/env python3
"""S6 假设检验：S1 事件后次周期内逐分钟入场 EV 扫描（2026-08-18）。

用户假设：
  - S6 = 次周期 15 分钟内每分钟监测赔率，按当时 EV 择时入场；
  - 直觉 1：逐分钟入场的胜率应低于 +5min 回落确认（S5）——晚确认 = 强信息 + 路径依赖；
  - 直觉 2：但 +5min 回落确认时赔率也变贵，所以 S6 的 EV 未必低于 S5。

数据限制（诚实声明）：历史逐分钟预测市场赔率不存在，报价 q(t) 只能模型化：
  q_λ(t) = p_rw(t) + λ·(p_emp(t) − p_rw(t))
  - p_rw(t)：无信息随机游走基准（对数收益正态，局部 σ），"朴素做市商"定价下界；
  - λ=0 完全无效 / λ=1 完全有效；真实 λ 用线上实测锚点校准：
    锚点1 开盘 q=0.51（B 段 p_emp=0.609, p_rw=0.5）→ λ≈0.09；
    锚点2 +5min 确认 q=0.58（2026-08-18 07:30 实测）→ 由本脚本回算。

输出：
  1) p(t) 条件胜率曲线（回落组/上涨组）vs p_rw(t) vs 超额信息 α(t)
  2) 首次回落即入 vs S5(t=5 固定) 对比
  3) EV(t) × λ 敏感性矩阵与最优入场时刻 t*
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request

import numpy as np

DAYS = 720
API = "https://data-api.binance.vision/api/v3/klines"
CACHE_5M = "output/klines_5m_cache_720d.json"
CACHE_1M = "output/klines_1m_cache_720d.json"
LOG = "output/s6_minute_scan.log"
FEE = 0.02
EPS = 0.0005
LOOKBACK = 48
Q_OPEN_ANCHOR = 0.51      # 实测锚点：S1 开盘时刻 DOWN 报价（线上经验值）
Q_5M_ANCHOR = 0.58        # 实测锚点：+5min 确认时刻 DOWN 报价（2026-08-18 07:30）


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = (f"{API}?symbol=BTCUSDT&interval={interval}"
               f"&startTime={cur}&endTime={end_ms}&limit=1000")
        batch = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    batch = json.loads(resp.read().decode())
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  重试 {attempt + 1}/3: {e}")
                time.sleep(2)
        if not batch:
            break
        out.extend(batch)
        cur = int(batch[-1][0]) + 1
        if len(out) % 100_000 < 1000:
            print(f"  已拉取 {len(out)} 根 {interval} ...")
        time.sleep(0.2)
    return out


def load_or_fetch(cache: str, interval: str, now_ms: int) -> list[list]:
    start_ms = now_ms - DAYS * 86_400_000
    kl: list[list] = []
    try:
        with open(cache, encoding="utf-8") as f:
            kl = json.load(f)
        print(f"缓存命中：{len(kl)} 根 {interval}")
    except Exception:
        pass
    last = int(kl[-1][0]) if kl else 0
    if last < now_ms - 2 * 86_400_000:
        kl = fetch_klines(interval, start_ms, now_ms)
    elif last < now_ms - 60_000:
        kl += fetch_klines(interval, last + 1, now_ms)
    kl = [k for k in kl if int(k[0]) >= start_ms]
    seen: dict[int, list] = {}
    for k in kl:
        seen[int(k[0])] = k
    kl = [seen[t] for t in sorted(seen)]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(kl, f)
    return kl


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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- 1) 5m 缓存 → 15m 聚合 → S1 事件（与 local_720d_validation 完全同口径）----------
    with open(CACHE_5M, encoding="utf-8") as f:
        kl5 = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl5]
    t5 = np.array([r[0] for r in c5])
    o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5])
    l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])
    v5 = np.array([r[5] for r in c5])

    cyc_ids = t5 // 900_000
    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(cyc_ids):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list, ks, cyc_to_idxs = [], {}, {}
    for cyc, idxs in buckets.items():
        if len(idxs) != 3 or (cyc + 1) * 900_000 > now_ms:
            continue
        idxs.sort()
        cyc_list.append(cyc)
        cyc_to_idxs[cyc] = idxs
        ks[cyc] = (o5[idxs[0]], max(h5[i] for i in idxs), min(l5[i] for i in idxs),
                   cl5[idxs[-1]], float(sum(v5[i] for i in idxs)))
    cyc_list.sort()
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([ks[c][0] for c in cyc_list])
    h15 = np.array([ks[c][1] for c in cyc_list])
    l15 = np.array([ks[c][2] for c in cyc_list])
    c15 = np.array([ks[c][3] for c in cyc_list])
    dir15 = np.sign(c15 - o15)
    rng15 = np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / rng15

    def roll_max(x, w):
        from numpy.lib.stride_tricks import sliding_window_view
        out = np.full(len(x), np.nan)
        out[w - 1:] = sliding_window_view(x, w).max(axis=1)
        return out

    def roll_min(x, w):
        from numpy.lib.stride_tricks import sliding_window_view
        out = np.full(len(x), np.nan)
        out[w - 1:] = sliding_window_view(x, w).min(axis=1)
        return out

    w16_hi = roll_max(c15, 16)
    w16_lo = roll_min(c15, 16)
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)

    lvl_hi = np.full(len(c5), np.nan)
    lvl_lo = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in cyc_to_idxs[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]) and broke_hi5[i]:
                broke_hi15[j] = True

    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)
    s1_idx = [j for j in range(N) if s1[j]
              and j + 1 < N and cyc_arr[j + 1] == cyc_arr[j] + 1]
    print(f"5m K {len(c5)} 根 → 15m {N} 根 → S1 事件（含连续次周期）n={len(s1_idx)}")

    # ---------- 2) 1m 数据 ----------
    kl1 = load_or_fetch(CACHE_1M, "1m", now_ms)
    c1 = {int(k[0]): (float(k[1]), float(k[4])) for k in kl1}   # open_time -> (open, close)
    t1_all = sorted(c1)
    import bisect
    print(f"1m K {len(c1)} 根")

    # ---------- 3) 事件分钟路径 ----------
    events = []
    skipped = 0
    for j in s1_idx:
        t0 = int(cyc_arr[j + 1]) * 900_000
        bars = [c1.get(t0 + i * 60_000) for i in range(15)]
        if any(b is None for b in bars):
            skipped += 1
            continue
        p_s = bars[0][0]
        closes = [b[1] for b in bars]
        if p_s <= 0:
            skipped += 1
            continue
        # 局部 σ：事件前 60 根 1m 对数收益 std（bisect 定位，避免全量扫描）
        i_lo = bisect.bisect_left(t1_all, t0 - 60 * 60_000)
        i_hi = bisect.bisect_left(t1_all, t0)
        hist = [c1[t][1] for t in t1_all[i_lo:i_hi]]
        if len(hist) < 50:
            skipped += 1
            continue
        lr = np.diff(np.log(np.array(hist)))
        sig = float(np.std(lr))
        if sig <= 0:
            skipped += 1
            continue
        events.append({
            "t0": t0, "p_s": p_s, "closes": closes, "sigma": sig,
            "win": closes[14] < p_s,
            "seg": "A" if t0 < now_ms - 360 * 86_400_000 else "B",
        })
    n_ev = len(events)
    print(f"有效事件 {n_ev}（跳过 {skipped}：1m 缺口/σ 无效）；A 段 {sum(1 for e in events if e['seg'] == 'A')} / B 段 {sum(1 for e in events if e['seg'] == 'B')}")
    overall = sum(e["win"] for e in events) / n_ev
    print(f"S1 全量基准胜率 {overall:.1%}（对照 720d 验证 ≈58%）")

    # ---------- 4) p(t) 曲线：条件回落/上涨 ----------
    print("\n===== p(t)：t 分钟收盘 vs 周期开盘 P(S) 的条件胜率（t=1..14）=====")
    print("  t | 回落组 n / 胜率 [Wilson] / p_rw | 上涨组 n / 胜率 / p_rw | α(回落)")
    rows_by_t: dict[int, dict] = {}
    for t in range(1, 15):
        rem = 15 - t
        fall = {"n": 0, "k": 0, "rw": []}
        rise = {"n": 0, "k": 0, "rw": []}
        for e in events:
            p_t = e["closes"][t - 1]
            p_rw = norm_cdf(math.log(e["p_s"] / p_t) / (e["sigma"] * math.sqrt(rem)))
            grp = fall if p_t < e["p_s"] else rise
            grp["n"] += 1
            grp["k"] += e["win"]
            grp["rw"].append(p_rw)
        fl, fh = wilson(fall["k"], fall["n"])
        rw_f = float(np.mean(fall["rw"])) if fall["rw"] else float("nan")
        rw_r = float(np.mean(rise["rw"])) if rise["rw"] else float("nan")
        p_f = fall["k"] / fall["n"] if fall["n"] else float("nan")
        alpha = p_f - rw_f if fall["n"] else float("nan")
        print(f"  {t:>2} | {fall['n']:>5} {p_f:6.1%} [{fl:.1%},{fh:.1%}] {rw_f:5.1%}"
              f" | {rise['n']:>5} {rise['k'] / max(rise['n'], 1):6.1%} {rw_r:5.1%} | {alpha:+.1%}")
        rows_by_t[t] = {"fall": fall, "rise": rise, "rw_f": rw_f, "alpha": alpha,
                        "p_f": p_f}

    # ---------- 5) 首次回落 vs S5(t=5 固定) ----------
    print("\n===== 入场规则对比（全量事件）=====")
    s5_k = s5_n = ft_k = ft_n = 0
    ft_times = []
    for e in events:
        c5_close = e["closes"][4]
        if c5_close < e["p_s"]:
            s5_n += 1
            s5_k += e["win"]
        first = next((t for t in range(1, 16) if e["closes"][t - 1] < e["p_s"]), None)
        if first is not None:
            ft_n += 1
            ft_k += e["win"]
            ft_times.append(first)
    lo5, hi5 = wilson(s5_k, s5_n)
    lof, hif = wilson(ft_k, ft_n)
    print(f"  S5（t=5 固定，回落才入） : n={s5_n} 胜率 {s5_k / s5_n:.1%} [{lo5:.1%},{hi5:.1%}]")
    print(f"  首次回落即入（t*=首触分钟）: n={ft_n} 胜率 {ft_k / ft_n:.1%} [{lof:.1%},{hif:.1%}]"
          f"  平均入场 t={np.mean(ft_times):.1f} 分钟")
    ft_hist = {t: ft_times.count(t) for t in sorted(set(ft_times))}
    print(f"  首触分钟分布: {ft_hist}")

    # ---------- 6) EV(t) × λ 敏感性（回落组逐事件）----------
    print("\n===== EV(t) 敏感性：q_λ(t) = p_rw(t) + λ·(p_emp(t) − p_rw(t))，逐事件结算 =====")
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    # 预计算逐事件 (win, p_rw) by t（回落组）
    per_t: dict[int, list[tuple[bool, float]]] = {}
    for t in range(1, 15):
        rem = 15 - t
        arr = []
        for e in events:
            p_t = e["closes"][t - 1]
            if p_t >= e["p_s"]:
                continue
            p_rw = norm_cdf(math.log(e["p_s"] / p_t) / (e["sigma"] * math.sqrt(rem)))
            arr.append((e["win"], p_rw))
        per_t[t] = arr

    def ev_at(t: int, lam: float) -> float:
        arr = per_t[t]
        if not arr:
            return float("nan")
        p_emp = sum(w for w, _ in arr) / len(arr)
        p_rwm = sum(r for _, r in arr) / len(arr)
        tot = 0.0
        for w, rw in arr:
            q = rw + lam * (p_emp - p_rwm)
            q = min(max(q, 0.02), 0.98)
            tot += (0.98 / q) if w else 0.0
        return tot / len(arr) - 1.0

    header = "  t   " + "".join(f"λ={lm:<5.1f}" for lm in lambdas)
    print(header)
    best = {}
    for t in range(1, 15):
        cells = "".join(f"{ev_at(t, lm):+7.3f}" for lm in lambdas)
        print(f"  {t:>2}  {cells}")
    for lm in lambdas:
        evs = {t: ev_at(t, lm) for t in range(1, 15)}
        t_star = max(evs, key=evs.get)
        best[lm] = (t_star, evs[t_star])
        print(f"  λ={lm:.1f} 最优 t*={t_star} 分钟，EV={evs[t_star]:+.3f}")

    # ---------- 7) 锚点校准 ----------
    print("\n===== 实测锚点校准 λ =====")
    lam_open = (Q_OPEN_ANCHOR - 0.50) / (overall - 0.50)
    print(f"  开盘锚：q={Q_OPEN_ANCHOR}，p_emp={overall:.1%}，p_rw=50% → λ≈{lam_open:.2f}")
    t5_arr = per_t[5]
    p_emp5 = sum(w for w, _ in t5_arr) / len(t5_arr)
    p_rwm5 = sum(r for _, r in t5_arr) / len(t5_arr)
    lam_5 = (Q_5M_ANCHOR - p_rwm5) / (p_emp5 - p_rwm5) if p_emp5 != p_rwm5 else float("nan")
    print(f"  +5min 锚：q={Q_5M_ANCHOR}，p_emp={p_emp5:.1%}，p_rw={p_rwm5:.1%} → λ≈{lam_5:.2f}")

    # ---------- 8) 结论 ----------
    print("\n===== 结论要点 =====")
    print(f"  1) 回落组胜率随 t 单调性：t=1 {rows_by_t[1]['p_f']:.1%} → t=5 {rows_by_t[5]['p_f']:.1%} → t=14 {rows_by_t[14]['p_f']:.1%}")
    print(f"  2) 超额信息 α(t) 峰值时刻：t={max(rows_by_t, key=lambda t: rows_by_t[t]['alpha'])}"
          f"（α={max(r['alpha'] for r in rows_by_t.values()):+.1%}）")
    for lm in (0.1, 0.3, 0.5):
        t_star, ev = best[lm]
        ev_s5 = ev_at(5, lm)
        print(f"  3) λ={lm:.1f}：最优 t*={t_star} EV={ev:+.3f} vs S5(t=5) EV={ev_s5:+.3f}（差 {ev - ev_s5:+.3f}）")

    out = {
        "n_events": n_ev,
        "rows_by_t": {str(t): {"p_f": r["p_f"], "rw_f": r["rw_f"], "alpha": r["alpha"],
                               "n_fall": r["fall"]["n"]} for t, r in rows_by_t.items()},
        "s5": {"n": s5_n, "k": s5_k},
        "first_touch": {"n": ft_n, "k": ft_k, "mean_t": float(np.mean(ft_times))},
        "best_by_lambda": {str(lm): best[lm] for lm in lambdas},
        "anchors": {"lam_open": lam_open, "lam_5m": lam_5},
    }
    with open("output/s6_minute_scan_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s6_minute_scan_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
