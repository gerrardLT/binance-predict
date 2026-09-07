#!/usr/bin/env python3
"""「首次触价时刻 × q 深度」二维网格：寻找 EV 日聚类 CI 下界 > 0 的可上线格子。

背景
----
上一轮（scripts/local_quote_reversal_by_bucket.py）结论：0~0.1 报价的反转率随
窗口推进单调衰减、随 q 深度单调上升；但没有任何单桶的 EV 在日聚类 bootstrap 下
显著为正。本轮把两个维度交叉，扫描 (首次触价时刻 × 触价深度) 格子，回答：
是否存在 EV 日聚类 95% CI 下界 > 0 的格子——即「值得进 live_channels.py 的候选」。

数据
----
  output/pull_samples_5m_20260907.json / pull_samples_15m_20260907.json（线上全量采样）
  output/klines_5m_cache_720d.json + klines_5m_tail_20260907.json（结算真值）

事件定义
--------
  per (window, side)：报价**首次**进入 (0, 0.1] 的那条采样 →
    t = 距窗口起点秒数，q = 该条报价，win = 该侧最终结算获胜。
  结束真值：K 线 close vs open（5m 单根 / 15m 三根聚合，平局剔除）。
  经济口径：FEE=2%；赢 = (1-FEE)/q − 1，输 = −1（逐事件，不用 q̄，防 Jensen 偏差）。

多重比较纪律（预先写死，防数据窥探）
------------------------------------
  扫描格子 ≈ 5 时间带 × 4 深度带 × 3 视图(UP/DOWN/MERGED) × 2 周期 ≈ 120 格。
  纯噪音下按 95% CI 扫描，期望假阳性 ≈ 0.05 × 参与扫描格子数。因此：
    1) calib/confirm 按天时间切分（前 2/3 天 calib，后 1/3 天 confirm，不重叠）；
    2) 候选格子只在 calib 上产生：n_calib ≥ 30、天数 ≥ 5、且 EV 日聚类 CI 下界 > 0；
    3) confirm 只终验一次，通过标准分两档（预先写死）：
       宽松 = confirm EV 点估计 > 0 且 EV 日聚类 CI 下界 > -0.10
       严格 = confirm EV 日聚类 CI 下界 > 0
  全期格子表照样打印（描述性），但 headline 只认 calib→confirm 路径。
  注意：MERGED 视图与 UP/DOWN 视图样本重叠，候选间不独立，只作线索不作独立证据。

用法
----
    .venv\\Scripts\\python.exe scripts/local_firsthit_qdepth_grid.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict

import numpy as np

FEE = 0.02
Q_HI = 0.1
WIN_MS = {"5m": 300_000, "15m": 900_000}
SAMPLES = {
    "5m": "output/pull_samples_5m_20260907.json",
    "15m": "output/pull_samples_15m_20260907.json",
}
KLINE_FILES = ["output/klines_5m_cache_720d.json", "output/klines_5m_tail_20260907.json"]
LOG = "output/firsthit_qdepth_grid.log"
OUT = "output/firsthit_qdepth_grid.json"

T_BANDS = {
    "5m": [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300)],
    "15m": [(0, 300), (300, 450), (450, 600), (600, 750), (750, 900)],
}
Q_BANDS = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 0.10)]
VIEWS = ("UP", "DOWN", "MERGED")
MIN_N_CELL = 30      # 全期格子打印明细的门槛
MIN_N_CALIB = 30     # calib 候选门槛
MIN_DAYS = 5         # bootstrap 需要的最少天数
BOOT_B = 4000
SEED = 20260907


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
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return max(0.0, c - h), min(1.0, c + h)


def day_boot(values: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    """按天分块 bootstrap 加权均值 CI（向量化；扣除窗口间时间自相关）。"""
    d_u = np.unique(days)
    if len(d_u) < MIN_DAYS:
        return (float("nan"), float("nan"))
    sums = np.zeros(len(d_u))
    cnts = np.zeros(len(d_u))
    idx = np.searchsorted(d_u, days)
    np.add.at(sums, idx, values)
    np.add.at(cnts, idx, 1)
    means = sums / cnts
    rng = np.random.default_rng(SEED)
    pick = rng.integers(0, len(d_u), size=(BOOT_B, len(d_u)))
    wsel = cnts[pick]
    tot = (means[pick] * wsel).sum(axis=1) / wsel.sum(axis=1)
    return (float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5)))


def load_klines() -> dict[int, tuple[float, float]]:
    rows: list[list] = []
    for p in KLINE_FILES:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.load(f))
    rows.sort(key=lambda k: int(k[0]))
    m: dict[int, tuple[float, float]] = {}
    for k in rows:
        m[int(k[0])] = (float(k[1]), float(k[4]))
    return m


def window_outcome(w: int, k5: dict[int, tuple[float, float]], period: str) -> str | None:
    if period == "5m":
        bar = k5.get(w)
        if bar is None:
            return None
        o, c = bar
    else:
        bars = [k5.get(w + i * 300_000) for i in range(3)]
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


def collect_first_hits(period: str, k5: dict[int, tuple[float, float]]) -> list[dict]:
    """每窗口每侧取报价首次进入 (0,0.1] 的采样作为事件。"""
    L = WIN_MS[period]
    with open(SAMPLES[period], encoding="utf-8") as f:
        raw = json.load(f)
    by_win: dict[int, list[dict]] = defaultdict(list)
    for s in raw:
        ts = int(s["timestamp"])
        by_win[ts // L * L].append(s)

    events: list[dict] = []
    n_win = n_ok = n_none = 0
    for w, rows in by_win.items():
        n_win += 1
        oc = window_outcome(w, k5, period)
        if oc is None:
            n_none += 1
            continue
        n_ok += 1
        rows.sort(key=lambda s: int(s["timestamp"]))
        day = time.strftime("%Y-%m-%d", time.gmtime(w / 1000))
        hit: dict[str, dict | None] = {"UP": None, "DOWN": None}
        for s in rows:
            if hit["UP"] is not None and hit["DOWN"] is not None:
                break
            ts = int(s["timestamp"])
            off = (ts - w) / 1000.0
            for side in ("UP", "DOWN"):
                if hit[side] is not None:
                    continue
                q = s.get("up_price") if side == "UP" else s.get("down_price")
                if q is None:
                    continue
                q = float(q)
                if 0.005 < q <= Q_HI:
                    hit[side] = {"w": w, "side": side, "t": off, "q": q,
                                 "win": 1 if oc == side else 0, "day": day}
        for side in ("UP", "DOWN"):
            if hit[side] is not None:
                events.append(hit[side])  # type: ignore[arg-type]

    print(f"[{period}] 窗口 {n_win}（可结算 {n_ok}，剔除平局/缺K线 {n_none}）"
          f" → 首触事件 {len(events)} 条")
    return events


def stats(evs: list[dict]) -> dict | None:
    n = len(evs)
    if n == 0:
        return None
    flags = np.array([e["win"] for e in evs], dtype=float)
    qs = np.array([e["q"] for e in evs], dtype=float)
    days = np.array([e["day"] for e in evs])
    k = int(flags.sum())
    p = k / n
    lo, hi = wilson(k, n)
    evvals = np.where(flags > 0, (1 - FEE) / qs - 1.0, -1.0)
    ev = float(evvals.mean())
    evlo, evhi = day_boot(evvals, days)
    n_days = int(len(np.unique(days)))
    return {"n": n, "k": k, "p": p, "ci": (lo, hi), "ev": ev,
            "ev_ci": (evlo, evhi), "qbar": float(qs.mean()),
            "n_days": n_days, "per_day": n / n_days if n_days else float("nan")}


def cell_filter(evs: list[dict], tb: tuple[int, int], qb: tuple[float, float]) -> list[dict]:
    return [e for e in evs if tb[0] <= e["t"] < tb[1] and qb[0] < e["q"] <= qb[1]]


def fmt_ev(st: dict | None) -> str:
    if st is None:
        return "    —     "
    lo, hi = st["ev_ci"]
    if math.isnan(lo):
        return f"+{st['ev']:+.2f}(n{st['n']})"
    return f"{st['ev']:+.2f}[{lo:+.2f},{hi:+.2f}]n{st['n']}"


def detail_line(tb, qb, st, tag):
    lo, hi = st["ci"]
    el, eh = st["ev_ci"]
    ew = "nan" if math.isnan(el) else f"[{el:+.2f},{eh:+.2f}]"
    print(f"  {tag} t{tb[0]:>4}-{tb[1]:<4}s × q({qb[0]},{qb[1]}] | n={st['n']:>5} "
          f"k={st['k']:>4} P={st['p']:>5.1%} W[{lo:.3f},{hi:.3f}] | "
          f"EV={st['ev']:+.3f} {ew} | q̄={st['qbar']:.3f} "
          f"{st['n_days']}天 ≈{st['per_day']:.1f}次/天")


def view_events(events: list[dict], view: str) -> list[dict]:
    if view == "MERGED":
        return events
    return [e for e in events if e["side"] == view]


def print_grid(events: list[dict], period: str, view: str, title: str) -> list[dict]:
    """打印一个视图的全期网格；返回 n≥MIN_N_CELL 的格子明细。"""
    evs = view_events(events, view)
    print(f"\n----- [{period} | {view}] {title}（EV/注 [日聚类95%CI] (n)）-----")
    head = " " * 14 + "".join(f"q({a},{b}]".rjust(24) for a, b in Q_BANDS) + "全深度".rjust(24)
    print(head)
    details: list[dict] = []
    for tb in T_BANDS[period]:
        cells = []
        row_all: list[dict] = []
        for qb in Q_BANDS:
            cell = cell_filter(evs, tb, qb)
            st = stats(cell)
            cells.append(fmt_ev(st) if st else "—")
            if st:
                row_all.append(st)
        st_row = stats([e for e in evs if tb[0] <= e["t"] < tb[1]])
        print(f"t{tb[0]:>4}-{tb[1]:<4}s  " + "".join(c.rjust(24) for c in cells)
              + (fmt_ev(st_row) if st_row else "—").rjust(24))
        for qb, st in zip(Q_BANDS, [stats(cell_filter(evs, tb, qb)) for qb in Q_BANDS]):
            if st and st["n"] >= MIN_N_CELL:
                details.append({"tb": tb, "qb": qb, "st": st})
    # 全时刻行
    cells = [stats([e for e in evs if qb[0] < e["q"] <= qb[1]]) for qb in Q_BANDS]
    print(f"{'全时刻':>14}  " + "".join(fmt_ev(s).rjust(24) if s else "—".rjust(24) for s in cells)
          + fmt_ev(stats(evs) if evs else None).rjust(24))
    print("  明细（n ≥ %d）：" % MIN_N_CELL)
    for d in details:
        star = " ★" if (not math.isnan(d["st"]["ev_ci"][0]) and d["st"]["ev_ci"][0] > 0) else ""
        detail_line(d["tb"], d["qb"], d["st"], star)
    if not details:
        print("    （无格子满足 n 门槛）")
    return details


def main() -> int:
    sys.stdout = Tee(LOG)
    print("=" * 110)
    print("「首次触价时刻 × q 深度」二维网格 · EV 日聚类 CI 扫描 | 5m / 15m")
    print("=" * 110)
    k5 = load_klines()
    print(f"K 线 {len(k5)} 根 [{iso(min(k5))} ~ {iso(max(k5))}]")

    out: dict = {"meta": {"fee": FEE, "q_hi": Q_HI, "min_n_cell": MIN_N_CELL,
                         "min_n_calib": MIN_N_CALIB, "boot_b": BOOT_B,
                         "t_bands": T_BANDS, "q_bands": Q_BANDS,
                         "criteria": {
                             "candidate": "calib: n>=30 & days>=5 & EV日聚类CI下界>0",
                             "loose_pass": "confirm EV>0 且 EV日聚类CI下界>-0.10",
                             "strict_pass": "confirm EV日聚类CI下界>0"}},
                 "periods": {}}

    n_scanned = 0
    all_results: dict[str, list] = {}

    for period in ("5m", "15m"):
        events = collect_first_hits(period, k5)
        # ---- 首触 q / t 分布 ----
        qhist = Counter(round(e["q"], 2) for e in events)
        print(f"  首触 q 分布（UP+DOWN）: "
              + " ".join(f"{q:.2f}:{n}" for q, n in sorted(qhist.items())))
        ts_ = np.array([e["t"] for e in events])
        pct = np.percentile(ts_, [10, 50, 90])
        print(f"  首触时刻分位 p10/p50/p90 = {pct[0]:.0f}/{pct[1]:.0f}/{pct[2]:.0f}s "
              f"（窗口 {WIN_MS[period] // 1000}s）")

        # ---- 全期网格（描述性）----
        print(f"\n{'=' * 110}\n### {period} 全期网格")
        full_details = {}
        for view in VIEWS:
            full_details[view] = print_grid(events, period, view, "全期")

        # ---- calib / confirm 切分 ----
        days = sorted({e["day"] for e in events})
        split_day = days[int(len(days) * 2 / 3)]
        calib = [e for e in events if e["day"] < split_day]
        confirm = [e for e in events if e["day"] >= split_day]
        print(f"\n{'=' * 110}\n### {period} calib/confirm 切分")
        print(f"calib: {days[0]} ~ <{split_day}（{sum(1 for d in days if d < split_day)} 天，"
              f"事件 {len(calib)}）| confirm: {split_day} ~ {days[-1]}"
              f"（{sum(1 for d in days if d >= split_day)} 天，事件 {len(confirm)}）")

        # ---- calib 候选 ----
        cands = []
        for view in VIEWS:
            evs_c = view_events(calib, view)
            for tb in T_BANDS[period]:
                for qb in Q_BANDS:
                    cell = cell_filter(evs_c, tb, qb)
                    st = stats(cell)
                    if st is None or st["n"] < MIN_N_CALIB or st["n_days"] < MIN_DAYS:
                        continue
                    n_scanned += 1
                    lo = st["ev_ci"][0]
                    if not math.isnan(lo) and lo > 0:
                        cands.append({"view": view, "tb": tb, "qb": qb, "calib": st})
        cands.sort(key=lambda c: -c["calib"]["ev_ci"][0])
        print(f"  calib 候选（EV 日聚类 CI 下界 > 0）：{len(cands)} 个")
        results = []
        for i, c in enumerate(cands, 1):
            evs_f = view_events(confirm, c["view"])
            st_f = stats(cell_filter(evs_f, c["tb"], c["qb"]))
            cs = c["calib"]
            print(f"  候选{i}: [{period} | {c['view']}] t{c['tb'][0]}-{c['tb'][1]}s × "
                  f"q({c['qb'][0]},{c['qb'][1]}]")
            print(f"    calib  : n={cs['n']:>4} P={cs['p']:.1%} q̄={cs['qbar']:.3f} "
                  f"EV={cs['ev']:+.3f} CI[{cs['ev_ci'][0]:+.3f},{cs['ev_ci'][1]:+.3f}]")
            if st_f is None:
                print("    confirm: 无事件 ❌")
                verdict = "NO_CONFIRM_DATA"
            else:
                fl, fh = st_f["ev_ci"]
                fw = "nan" if math.isnan(fl) else f"[{fl:+.3f},{fh:+.3f}]"
                print(f"    confirm: n={st_f['n']:>4} P={st_f['p']:.1%} q̄={st_f['qbar']:.3f} "
                      f"EV={st_f['ev']:+.3f} {fw}")
                if not math.isnan(fl) and fl > 0:
                    verdict = "STRICT_PASS"
                elif st_f["ev"] > 0 and fl > -0.10:
                    verdict = "LOOSE_PASS"
                else:
                    verdict = "FAIL"
                print(f"    判定: {verdict}")
            results.append({**{k: v for k, v in c.items() if k != "calib"},
                            "calib": cs, "confirm": st_f, "verdict": verdict})

        all_results[period] = results
        out["periods"][period] = {
            "n_events": len(events),
            "days": days, "split_day": split_day,
            "full_grids": {v: [{"tb": list(d["tb"]), "qb": list(d["qb"]), **{
                kk: (round(vv, 5) if isinstance(vv, float) else vv)
                for kk, vv in d["st"].items() if kk in ("n", "k", "p", "qbar", "ev")}}
                for d in full_details[v]] for v in VIEWS},
            "candidates": [{**{k: v for k, v in r.items() if k not in ("calib", "confirm")},
                            "calib": {k2: (round(v2, 5) if isinstance(v2, float) else v2)
                                      for k2, v2 in r["calib"].items()},
                            "confirm": (None if r["confirm"] is None else
                                        {k2: (round(v2, 5) if isinstance(v2, float) else v2)
                                         for k2, v2 in r["confirm"].items()}),
                            "verdict": r["verdict"]} for r in results],
        }

    # ---- 汇总 ----
    print("\n" + "=" * 110)
    print("### 汇总")
    print(f"参与扫描的 calib 格子数（n_calib≥{MIN_N_CALIB}）：{n_scanned} 个"
          f" → 纯噪音下期望假阳性 ≈ {0.05 * n_scanned:.1f} 个")
    for period, rs in all_results.items():
        strict = sum(1 for r in rs if r["verdict"] == "STRICT_PASS")
        loose = sum(1 for r in rs if r["verdict"] == "LOOSE_PASS")
        print(f"[{period}] calib 候选 {len(rs)} → 严格通过 {strict}，宽松通过 {loose}，"
              f"失败 {len(rs) - strict - loose}")
    total_strict = sum(1 for rs in all_results.values()
                       for r in rs if r["verdict"] == "STRICT_PASS")
    print(f"\n结论：严格通过（confirm EV 日聚类 CI 下界 > 0）的格子共 {total_strict} 个")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON → {OUT} | 日志 → {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
