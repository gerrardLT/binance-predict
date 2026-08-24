#!/usr/bin/env python3
"""S5 分行情（regime）分段分析 —— 检验 76.4% 池化胜率的行情依赖性。

背景：S5（bull_exhaust_confirm）76.4% 是 720d 池化数字。已有分段证据：
  时间 8×90d：74%~80%（validation_720d.log，稳定）；A/B 半期 74.9%/78.3%；
  回落深度：浅 67.6% / 中 77.2% / 深 85.0%（s5_real_quote_ev.log）。
本脚本补上缺的行情状态（regime）维度 + 波动率维度。

regime 定义（同 local_cycle_regime_check.py，无未来函数，仅用截至事件根数据）：
  ER_7d[j] = |c15[j] − c15[j−672]| / Σ|Δc15|（672 根 15m = 7 天效率比）
  RV_7d[j] = std(Δlog c15, 672 根)（7 天已实现波动率）
  阈值取发现段（A 段前 360 天）分布分位数（防验证集泄漏进阈值）：
    趋势牛 = ER≥q75 且 7d 净位移>0 | 趋势熊 = ER≥q75 且 <0
    震荡 = ER<q50 | 过渡 = 其余
  波动档：低 RV<q33 / 中 / 高 RV≥q67

事件 = output/klines_5m_cache_720d.json 720d 锁定口径（同 local_s5_real_quote_ev.py）
报价 = prediction_market_samples_online_20260819.json → q̂(t=5, 回落态, 深度)
线上对照 = output/online_signals_now_20260824_1641.json 的 S5 结算行
EV 口径：赢 0.98/q̂−1 / 输 −1（无溢价快照口径，与 s5_real_quote_ev 一致）
纪律声明：exploratory 事后细分——用于理解周期依赖与仓位分配，不筛选新场景。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import time
from collections import defaultdict

import numpy as np

FEE = 0.02
EPS = 0.0005
LOOKBACK = 48
KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
LIVE = "output/online_signals_now_20260824_1641.json"
LOG = "output/s5_regime_analysis.log"
DEPTH_EDGES = (0.0005, 0.0015)
W = 672  # 7d × 96 根 15m
REGIMES = ("趋势牛", "趋势熊", "过渡", "震荡")
VBANDS = ("低", "中", "高")


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


def dbucket(depth: float) -> int:
    if depth < DEPTH_EDGES[0]:
        return 0
    if depth < DEPTH_EDGES[1]:
        return 1
    return 2


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% 区间。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- 15m 聚合 + S1 判定（720d 锁定口径，同 local_s5_real_quote_ev.py） ----------
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5]); o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5]); l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])

    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list, ks = [], {}
    for cyc, idxs in buckets.items():
        if len(idxs) != 3 or (cyc + 1) * 900_000 > now_ms:
            continue
        idxs.sort()
        cyc_list.append(cyc)
        ks[cyc] = idxs
    cyc_list.sort()
    N = len(cyc_list)
    o15 = np.array([c5[ks[c][0]][1] for c in cyc_list])
    h15 = np.array([max(h5[i] for i in ks[c]) for c in cyc_list])
    l15 = np.array([min(l5[i] for i in ks[c]) for c in cyc_list])
    c15 = np.array([c5[ks[c][-1]][4] for c in cyc_list])
    dir15 = np.sign(c15 - o15)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

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

    w16_hi = roll_max(c15, 16); w16_lo = roll_min(c15, 16)
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)
    lvl_hi = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]) and broke_hi5[i]:
                broke_hi15[j] = True
    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)

    # ---------- ER / RV 序列（无未来函数：截至根 j 收盘） ----------
    ad = np.abs(np.diff(c15))
    lr = np.diff(np.log(np.maximum(c15, 1e-12)))
    cs = np.concatenate([[0.0], np.cumsum(ad)])
    path_sum = np.full(N, np.nan)
    path_sum[W:] = cs[W:] - cs[:-W]          # sum|Δc15| over c15[j-W..j]
    disp = np.full(N, np.nan)
    disp[W:] = c15[W:] - c15[:-W]            # 7d 净位移
    er = np.abs(disp) / np.where(path_sum > 0, path_sum, np.nan)
    rv = np.full(N, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    rv[W:] = sliding_window_view(lr, W).std(axis=1)   # 7d 已实现波动率

    a_end = now_ms - 360 * 86_400_000
    close_t = (np.array(cyc_list, dtype=np.int64) + 1) * 900_000   # 根 j 收盘时刻（int64 防溢出）
    m_a = (close_t < a_end) & ~np.isnan(er) & ~np.isnan(rv)
    er50 = float(np.quantile(er[m_a], 0.50))
    er75 = float(np.quantile(er[m_a], 0.75))
    rv33 = float(np.quantile(rv[m_a], 1 / 3))
    rv67 = float(np.quantile(rv[m_a], 2 / 3))
    print(f"发现段（A 段）ER/RV 分位数阈值：ER q50={er50:.4f} q75={er75:.4f}"
          f" | RV q33={rv33:.6f} q67={rv67:.6f}")

    def regime(j: int) -> str | None:
        if j < W or np.isnan(er[j]):
            return None
        if er[j] >= er75:
            return "趋势牛" if disp[j] > 0 else "趋势熊"
        return "震荡" if er[j] < er50 else "过渡"

    def volband(j: int) -> str | None:
        if j < W or np.isnan(rv[j]):
            return None
        return "低" if rv[j] < rv33 else ("高" if rv[j] >= rv67 else "中")

    # ---------- S1/S5 事件重建（j>=W 才有 regime；前 7 天事件丢弃并计数） ----------
    events = []
    dropped = 0
    for j in range(N - 1):
        if not s1[j] or cyc_list[j + 1] != cyc_list[j] + 1:
            continue
        if o15[j + 1] <= 0:
            continue
        if j < W:
            dropped += 1
            continue
        nxt = cyc_list[j + 1]
        c1 = c5[ks[nxt][0]][4]  # 次周期第 1 根 5m 收盘（t=5min 时刻价）
        z5 = math.log(c1 / o15[j + 1])
        events.append({
            "start": (cyc_list[j] + 1) * 900_000, "open": float(o15[j + 1]),
            "win": bool(c15[j + 1] < o15[j + 1]),
            "z5": float(z5), "d5": dbucket(abs(z5)),
            "seg": "A" if (cyc_list[j] + 1) * 900_000 < a_end else "B",
            "rg": regime(j), "vb": volband(j), "j": j,
        })
    s5e = [e for e in events if e["z5"] < 0]
    print(f"S1 事件 {len(events)}（前 7 天无 regime 丢弃 {dropped}）"
          f" | S5 确认子集 {len(s5e)}"
          f"（A {sum(1 for e in s5e if e['seg'] == 'A')} / B {sum(1 for e in s5e if e['seg'] == 'B')}）")

    # ---------- 15m 报价表 q̂(t, 状态, 深度)（同 local_s5_real_quote_ev.py） ----------
    with open(SAMPLES, encoding="utf-8") as f:
        raw_all = json.load(f)
    raw15 = [s for s in raw_all if s.get("market_period") == "15m"]
    agg15: dict[int, tuple[float, float]] = {}
    for cyc, idxs in buckets.items():
        if len(idxs) == 3:
            idxs.sort()
            agg15[cyc] = (float(c5[idxs[0]][1]), float(c5[idxs[-1]][4]))
    qt15 = {b: {st: {d: [] for d in range(3)} for st in (0, 1)} for b in range(15)}
    qb15 = {b: {st: [] for st in (0, 1)} for b in range(15)}
    for s in raw15:
        q = s.get("down_price"); ts = int(s["timestamp"]); btc = s.get("btc_price")
        cyc = ts // 900_000
        oc = agg15.get(int(cyc))
        if q is None or q <= 0.01 or q >= 0.99 or btc is None or oc is None or oc[0] <= 0:
            continue
        b = int((ts - cyc * 900_000) // 60_000)
        if b >= 15:
            continue
        st = 0 if btc < oc[0] else 1
        qt15[b][st][dbucket(abs(math.log(btc / oc[0])))].append(float(q))
        qb15[b][st].append(float(q))

    def qhat15(b: int, st: int, d: int) -> float:
        vals = qt15[b][st][d]
        if len(vals) >= 30:
            return float(np.mean(vals))
        allv = qb15[b][st]
        return float(np.mean(allv)) if allv else float("nan")

    def ev_of(e: dict) -> float:
        q5 = qhat15(5, 0, e["d5"])
        return (0.98 / q5 - 1.0) if e["win"] else -1.0

    # ---------- A. 行情面貌：全期 regime 占比 + 8×90d 演变 ----------
    print("\n===== A. 720d 行情面貌（15m 根级 regime 分布） =====")
    t0 = close_t[0]
    seg_ms = 90 * 86_400_000
    seg8 = np.clip(((close_t - t0) // seg_ms).astype(int), 0, 7)
    labels = [f"{dt.datetime.fromtimestamp((t0 + i * seg_ms) / 1000, tz=dt.timezone.utc):%y-%m}" for i in range(8)]
    rg_arr = np.array([regime(j) or "未知" for j in range(N)])
    row = "  ".join(f"{r}:{sum(rg_arr == r) / N:5.1%}" for r in REGIMES)
    print(f"  全期 720d: {row}")
    print(f"  {'段':>8} | " + " | ".join(f"{r:>6}" for r in REGIMES))
    for i in range(8):
        m = seg8 == i
        cnt = {r: int(sum(rg_arr[m] == r)) for r in REGIMES}
        n_i = int(m.sum())
        cells = " | ".join(f"{cnt[r] / n_i:6.1%}" for r in REGIMES)
        print(f"  {labels[i]:>8} | {cells}  (n={n_i})")

    # ---------- B. 核心：regime × S5（胜率 + 真实报价 EV） ----------
    print("\n===== B. regime × S5 确认入场：n / P(DOWN) / EV真实报价 =====")
    print(f"  {'regime':<6} | {'n':>5} {'P':>6} {'95%CI':>15} | {'q̄真实':>6} "
          f"| {'EV真实':>8} {'±95%':>7} | {'EV@0.51':>8}")
    for r in REGIMES:
        grp = [e for e in s5e if e["rg"] == r]
        if not grp:
            print(f"  {r:<6} |    0     —               — |      — |        —       |        —")
            continue
        n = len(grp); k = sum(e["win"] for e in grp)
        p = k / n
        lo, hi = wilson(k, n)
        qbar = float(np.mean([qhat15(5, 0, e["d5"]) for e in grp]))
        evs = [ev_of(e) for e in grp]
        se = float(np.std(evs) / math.sqrt(n))
        print(f"  {r:<6} | {n:>5} {p:6.1%} [{lo:5.1%},{hi:5.1%}] | {qbar:6.3f}"
              f" | {float(np.mean(evs)):+8.3f} ±{1.96 * se:6.3f}"
              f" | {p * (0.98 / 0.51 - 1) - (1 - p):+8.3f}")
    n = len(s5e); k = sum(e["win"] for e in s5e); p = k / n
    lo, hi = wilson(k, n)
    evs = [ev_of(e) for e in s5e]
    se = float(np.std(evs) / math.sqrt(n))
    print(f"  {'合计':<6} | {n:>5} {p:6.1%} [{lo:5.1%},{hi:5.1%}] |"
          f"   —    | {float(np.mean(evs)):+8.3f} ±{1.96 * se:6.3f}"
          f" | {p * (0.98 / 0.51 - 1) - (1 - p):+8.3f}")

    # ---------- C. regime × 回落深度 交叉 EV ----------
    print("\n===== C. regime × 回落深度 交叉（n / P / EV真实） =====")
    print(f"  {'regime':<6} | " + " | ".join(f"{'深度' + d:>20}" for d in "浅中深"))
    for r in REGIMES:
        cells = []
        for d in range(3):
            grp = [e for e in s5e if e["rg"] == r and e["d5"] == d]
            if len(grp) >= 20:
                p_ = sum(e["win"] for e in grp) / len(grp)
                ev_ = float(np.mean([ev_of(e) for e in grp]))
                cells.append(f"{len(grp):>4} {p_:5.1%} {ev_:+7.3f}")
            else:
                cells.append(f"{len(grp):>4}     —        —")
        print(f"  {r:<6} | " + " | ".join(f"{c:>20}" for c in cells))

    # ---------- D. 波动率档 × S5 ----------
    print("\n===== D. 7d 已实现波动率档 × S5 =====")
    print(f"  {'波动档':<6} | {'n':>5} {'P':>6} {'95%CI':>15} | {'EV真实':>8} {'±95%':>7}")
    for v in VBANDS:
        grp = [e for e in s5e if e["vb"] == v]
        if not grp:
            print(f"  {v:<6} |    0     —               — |        —")
            continue
        n = len(grp); k = sum(e["win"] for e in grp)
        p = k / n
        lo, hi = wilson(k, n)
        evs = [ev_of(e) for e in grp]
        se = float(np.std(evs) / math.sqrt(n))
        print(f"  {v:<6} | {n:>5} {p:6.1%} [{lo:5.1%},{hi:5.1%}]"
              f" | {float(np.mean(evs)):+8.3f} ±{1.96 * se:6.3f}")

    # ---------- E. regime × S1 漏斗（触发 → 确认 → 筛选增量） ----------
    print("\n===== E. regime × S1→S5 漏斗（筛选效果是否各行情一致） =====")
    print(f"  {'regime':<6} | {'S1 n':>6} {'S1 P':>6} | {'确认率':>6} | "
          f"{'S5 P':>6} | {'放弃组P':>7} | {'增量pp':>7}")
    for r in REGIMES:
        g1 = [e for e in events if e["rg"] == r]
        g5 = [e for e in g1 if e["z5"] < 0]
        ga = [e for e in g1 if e["z5"] >= 0]
        if not g1:
            print(f"  {r:<6} |     0     — |      — |      — |       — |       —")
            continue
        p1 = sum(e["win"] for e in g1) / len(g1)
        cf = len(g5) / len(g1)
        p5 = sum(e["win"] for e in g5) / len(g5) if g5 else float("nan")
        pa = sum(e["win"] for e in ga) / len(ga) if ga else float("nan")
        print(f"  {r:<6} | {len(g1):>6} {p1:6.1%} | {cf:6.1%} | {p5:6.1%}"
              f" | {pa:7.1%} | {100 * (p5 - p1):+7.1f}")

    # ---------- F. 线上 S5 结算行 → regime 归属 ----------
    print("\n===== F. 线上 S5 结算 12 注的 regime 归属（对照） =====")
    try:
        with open(LIVE, encoding="utf-8") as f:
            live_all = json.load(f)
        live_s5 = [s for s in live_all.get("signals", [])
                   if s.get("pattern_type") == "bull_exhaust_confirm"
                   and s.get("settle_outcome")]
        live_s5.sort(key=lambda s: s["signal_time"])
        cyc_to_j = {c: j for j, c in enumerate(cyc_list)}
        cnt: dict[str, int] = defaultdict(int)
        for s in live_s5:
            q = s.get("entry_down_price_15m")
            ts = int(s["signal_time"])
            t = dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc)
            j = cyc_to_j.get(ts // 900_000)
            r = regime(j) if j is not None and j >= W else "缓存外/未知"
            win = s["settle_outcome"] == "DOWN"
            ev = (0.98 / q - 1.0) if (win and q) else (-1.0 if not win else float("nan"))
            cnt[r] += 1
            print(f"  id{s['id']:>4} {t:%m-%d %H:%M} q={q if q else '—':>5}"
                  f" {'赢' if win else '输'} EV={ev:+.3f} → {r}")
        print("  归属分布: " + ", ".join(f"{r}×{c}" for r, c in sorted(cnt.items())))
    except Exception as ex:
        print(f"  （线上对照跳过：{ex}）")

    # ---------- H. 深度档 / 波动档 × A/B 半期稳定性（防单段效应） ----------
    print("\n===== H. 深度档 & 波动档 × A/B 半期稳定性（防把单段效应当真信号） =====")
    print(f"  {'维度':<8} | {'档':<4} | {'A段  n / P / EV':>24} | {'B段  n / P / EV':>24}")

    def seg_cell(grp):
        if grp:
            p_ = sum(e["win"] for e in grp) / len(grp)
            ev_ = float(np.mean([ev_of(e) for e in grp]))
            return f"{len(grp):>5} {p_:5.1%} {ev_:+7.3f}"
        return "    —      —       —"

    for d in range(3):
        cells = [seg_cell([e for e in s5e if e["d5"] == d and e["seg"] == s]) for s in ("A", "B")]
        print(f"  {'回落深度':<8} | {'浅中深'[d]:<4} | {cells[0]:>24} | {cells[1]:>24}")
    for v in VBANDS:
        cells = [seg_cell([e for e in s5e if e["vb"] == v and e["seg"] == s]) for s in ("A", "B")]
        print(f"  {'7d波动':<8} | {v:<4} | {cells[0]:>24} | {cells[1]:>24}")

    # ---------- G. 报价表来源周的 regime 偏倚声明 ----------
    print("\n===== G. q̂ 报价表来源周（08-13~08-19）的 regime 背景 =====")
    ts15 = [int(s["timestamp"]) for s in raw15]
    if ts15:
        t_lo, t_hi = min(ts15), max(ts15)
        m_w = (close_t >= t_lo) & (close_t <= t_hi)
        if m_w.sum() > 0:
            dist = {r: float(np.mean(rg_arr[m_w] == r)) for r in REGIMES}
            print(f"  样本周 {dt.datetime.fromtimestamp(t_lo / 1000, tz=dt.timezone.utc):%m-%d}"
                  f"~{dt.datetime.fromtimestamp(t_hi / 1000, tz=dt.timezone.utc):%m-%d}"
                  f"（{int(m_w.sum())} 根 15m）: "
                  + ", ".join(f"{r} {dist[r]:.0%}" for r in REGIMES))
            print("  → q̂(t=5,回落态,深度) 表来自这一周；若该周 regime 与未来实盘 regime 不同，")
            print("    EV真实 的入场价假设存在 regime 偏倚（本表 t=5 回落态样本按深度加权后近似全期结构）。")

    # ---------- 存结果 ----------
    def pack(grp):
        evs = [ev_of(e) for e in grp]
        return {"n": len(grp),
                "p": sum(e["win"] for e in grp) / len(grp) if grp else None,
                "ev": float(np.mean(evs)) if grp else None}

    result = {
        "thresholds": {"er50": er50, "er75": er75, "rv33": rv33, "rv67": rv67},
        "regime_s5": {r: pack([e for e in s5e if e["rg"] == r]) for r in REGIMES},
        "vband_s5": {v: pack([e for e in s5e if e["vb"] == v]) for v in VBANDS},
        "regime_x_depth": {
            r: {d: pack([e for e in s5e if e["rg"] == r and e["d5"] == d])
                for d in range(3)} for r in REGIMES},
        "depth_x_seg": {
            d: {s: pack([e for e in s5e if e["d5"] == d and e["seg"] == s])
                for s in ("A", "B")} for d in range(3)},
        "vband_x_seg": {
            v: {s: pack([e for e in s5e if e["vb"] == v and e["seg"] == s])
                for s in ("A", "B")} for v in VBANDS},
    }
    with open("output/s5_regime_analysis_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s5_regime_analysis_result.json；日志 output/s5_regime_analysis.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
