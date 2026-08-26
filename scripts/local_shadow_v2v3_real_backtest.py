#!/usr/bin/env python3
"""影子 v2/v3 五版本真实数据回测（2026-08-26）：x4_v2 / quote_momentum_v2 /
quote_contrarian_v2 / quote_contrarian_v3a / quote_contrarian_v3b。

目的：信号分析面板 SHADOW_BENCH 中这五版本 bench=None（只有归因描述，无独立
回测基准）。本脚本用两段真实数据补齐「回测胜率」：

  A 段（线上全历史，门禁归因同源=IS）：
     线上 v1 已结算影子行（真实触发+真实入场价+真实结算）∩ 门禁过滤。
     v2/v3 门禁是 v1 触发集的纯子集（同报价区间+附加门禁），故 v1 行过门禁
     即该版本的历史真实笔。门禁特征（chg/前窗/日高/past1h）全部 ex-ante。
  B 段（本地全曲线 07-13~07-30，门禁归因从未接触=OOS 盲验）：
     sentiment_windows.json 全字段重放检测器冻结口径（报价区间首命中+门禁+
     结算），与线上检测器逻辑逐字段对齐。

口径冻结（与 detector 源码一致，勿动）：
  quote_momentum   t∈[90,120)s q∈[0.69,0.75)  押本窗 DOWN；EV=0.98/q−1（无溢价）
  quote_contrarian t∈[45,60)s  q∈[0.15,0.25)  押本窗 DOWN；EV 同上
  v2 momentum: chg≤−0.10% / v2 contrarian: chg<+0.10%（chg=触发点BTC vs 窗开盘）
  v3a: v2 ∩ 前窗 outcome==DOWN；v3b: v3a ∩ 距日高回落≤−0.30%（含边界）
  x4: 本窗收阳 & 末点 UP%≤40 → 次窗 DOWN；决策点=次窗+150s 真实 DOWN 价；
      EV=0.98/(q+0.01)−1（含溢价）；x4_v2: |past1h|<0.5%（±10min 容差）

输出：stdout + output/shadow_v2v3_real_backtest.log；
      结果 JSON output/shadow_v2v3_real_backtest_result.json。
用法：python -X utf8 scripts/local_shadow_v2v3_real_backtest.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOCAL_WINDOWS = "sentiment_windows.json"
CACHE = "output/.shadow_v2v3_bt_cache.json"
LOG = "output/shadow_v2v3_real_backtest.log"
OUT = "output/shadow_v2v3_real_backtest_result.json"

VERSIONS = ["x4_v2", "quote_momentum_v2", "quote_contrarian_v2",
            "quote_contrarian_v3a", "quote_contrarian_v3b"]


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


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ----------------------------------------------------------------------
# 通用 ex-ante 取价工具（与 detector 同口径）
# ----------------------------------------------------------------------

def btc_at_or_before(curve: list | None, ts: int) -> float | None:
    best = None
    for p in sorted(curve or [], key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts:
            best = float(v)
    return best


def window_open_btc(w: dict) -> float | None:
    p = w.get("entry_price")
    if p is not None and float(p) > 0:
        return float(p)
    for pt in sorted(w.get("curve_btc_price") or [], key=lambda x: x.get("t") or 0):
        v = pt.get("v")
        if v is not None and float(v) > 0:
            return float(v)
    return None


def find_first_hit(curve: list | None, start_ms: int,
                   t_lo: float, t_hi: float, q_lo: float, q_hi: float):
    if not curve:
        return None
    for p in sorted(curve, key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        t_rel = (int(t) - start_ms) / 1000.0
        if t_rel < 0:
            continue
        if t_rel >= t_hi:
            break
        if t_rel < t_lo:
            continue
        v = float(v)
        if q_lo <= v < q_hi:
            return v, int(t)
    return None


def curve_end_pct(curve: list | None) -> float | None:
    pts = sorted(curve or [], key=lambda p: p.get("t", 0))
    for p in reversed(pts):
        v = p.get("v")
        if v is not None:
            return float(v)
    return None


def price_within(curve: list | None, start_ms: int, max_sec: float):
    """≤max_sec 内最晚采样点（x4 决策点同口径）。"""
    best = None
    for p in sorted(curve or [], key=lambda x: x.get("t", 0)):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if (int(t) - start_ms) / 1000.0 <= max_sec:
            best = float(v)
    return best


def build_day_high(wins: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """UTC 日 -> 按 t 升序的 (t, running_max) 前缀表（brainstorm 同口径）。"""
    by_day: dict[int, list[tuple[int, float]]] = {}
    for w in wins:
        for p in w.get("curve_btc_price") or []:
            t, v = p.get("t"), p.get("v")
            if t is not None and v is not None:
                by_day.setdefault(int(t) // 86_400_000, []).append((int(t), float(v)))
    out = {}
    for d, pts in by_day.items():
        pts.sort()
        run = -1.0
        out[d] = [(t, run := max(run, v)) for t, v in pts]
    return out


def day_high_before(prefix: dict, ts: int) -> float | None:
    pts = prefix.get(ts // 86_400_000)
    if not pts:
        return None
    best = None
    for t, m in pts:
        if t <= ts:
            best = m
        else:
            break
    return best


# ----------------------------------------------------------------------
# A 段：线上 v1 已结算行 ∩ 门禁
# ----------------------------------------------------------------------

def online_segment() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)
    else:
        def get(p):
            with urllib.request.urlopen(BASE + p, timeout=300) as r:
                return json.load(r)
        cache = {
            "x4_v1": get("/api/misalignment/signals?limit=5000&version=x4_v1")["signals"],
            "qm_v1": get("/api/misalignment/signals?limit=5000&version=quote_momentum_v1")["signals"],
            "qc_v1": get("/api/misalignment/signals?limit=5000&version=quote_contrarian_v1")["signals"],
            "wins": get("/api/sentiment/windows?limit=50000"),
        }
        os.makedirs("output", exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    wins = {int(w["start_time"]): w for w in cache["wins"]}
    prefix = build_day_high(list(wins.values()))
    t0 = min(w["start_time"] for w in wins.values())
    t1 = max(w["start_time"] for w in wins.values())

    def settled(rows):
        return [r for r in rows if r.get("status") == "SETTLED"
                and r.get("win") is not None]

    stats: dict[str, dict] = {}

    def agg(key: str, rows: list[dict]):
        n = len(rows)
        k = sum(1 for r in rows if r["win"])
        evs = [r["ev_at_entry"] for r in rows if r.get("ev_at_entry") is not None]
        stats[key] = {"n": n, "wins": k,
                      "wr": k / n if n else None,
                      "wilson": wilson(k, n) if n else (None, None),
                      "avg_ev": sum(evs) / len(evs) if evs else None}

    # ---- quote 族：v1 行过 v2/v3 门禁 ----
    for label, rows, mode, thr in (
            ("quote_momentum_v2", settled(cache["qm_v1"]), "min_drop", -0.10),
            ("quote_contrarian_v2", settled(cache["qc_v1"]), "max_rise", 0.10)):
        v2_rows, v3a_rows, v3b_rows = [], [], []
        for r in rows:
            ws = int(r["window_start"])
            ts = r.get("entry_quote_ts")
            w = wins.get(ws)
            if ts is None or w is None:
                continue
            base = window_open_btc(w)
            cur = btc_at_or_before(w.get("curve_btc_price"), int(ts))
            if base is None or cur is None:
                continue
            chg = (cur - base) / base * 100.0
            ok = chg <= thr if mode == "min_drop" else chg < thr
            if not ok:
                continue
            v2_rows.append(r)
            prev = wins.get(ws - 300_000)
            if (prev or {}).get("outcome") != "DOWN":
                continue
            v3a_rows.append(r)
            dh = day_high_before(prefix, int(ts))
            if dh and (cur - dh) / dh * 100.0 <= -0.30:
                v3b_rows.append(r)
        agg(label, v2_rows)
        if label == "quote_contrarian_v2":
            agg("quote_contrarian_v3a", v3a_rows)
            agg("quote_contrarian_v3b", v3b_rows)

    # ---- x4_v2：x4_v1 行过 past1h 门禁 ----
    x4v2 = []
    for r in settled(cache["x4_v1"]):
        ws = int(r["window_start"])
        w = wins.get(ws)
        if w is None:
            continue
        target = ws - 3_600_000
        base = _x4_base(wins, _X4_IDX, target)
        cur = window_open_btc(w)
        if base is None or cur is None:
            continue
        if abs((cur - base) / base * 100.0) < 0.5:
            x4v2.append(r)
    agg("x4_v2", x4v2)

    return {"stats": stats, "range": (t0, t1), "n_windows": len(wins)}


_X4_IDX: list[tuple[int, float]] = []


def _x4_base(wins: dict, idx: list[tuple[int, float]], target: int) -> float | None:
    """start−1h（±10min）内最晚有 entry_price 的窗口价（detector 同口径）。"""
    import bisect
    if not idx:
        items = sorted((int(w["start_time"]), w.get("entry_price"))
                       for w in wins.values())
        idx.extend(items)
    lo, hi = target - 600_000, target + 600_000
    i = bisect.bisect_right(idx, (hi, float("inf"))) - 1
    while i >= 0 and idx[i][0] >= lo:
        if idx[i][1] is not None and float(idx[i][1]) > 0:
            return float(idx[i][1])
        i -= 1
    return None


# ----------------------------------------------------------------------
# B 段：本地全曲线盲验重放（07-13~07-30，门禁归因未接触）
# ----------------------------------------------------------------------

def local_segment() -> dict:
    with open(LOCAL_WINDOWS, encoding="utf-8") as f:
        wins = json.load(f)
    wins = {int(w["start_time"]): w for w in wins}
    ordered = [wins[k] for k in sorted(wins)]
    prefix = build_day_high(ordered)
    t0, t1 = ordered[0]["start_time"], ordered[-1]["start_time"]

    evs: dict[str, list] = {v: [] for v in VERSIONS}
    wins_cnt: dict[str, list] = {v: [] for v in VERSIONS}

    for w in ordered:
        ws = int(w["start_time"])
        outcome = w.get("outcome")
        if outcome not in ("UP", "DOWN"):
            continue

        # ---- quote 族 ----
        for rule, t_lo, t_hi, q_lo, q_hi in (
                ("quote_momentum_v2", 90.0, 120.0, 0.69, 0.75),
                ("quote_contrarian_v2", 45.0, 60.0, 0.15, 0.25)):
            hit = find_first_hit(w.get("curve_down_price"), ws, t_lo, t_hi, q_lo, q_hi)
            if hit is None:
                continue
            q, ts = hit
            base = window_open_btc(w)
            cur = btc_at_or_before(w.get("curve_btc_price"), ts)
            if base is None or cur is None:
                continue
            chg = (cur - base) / base * 100.0
            if rule == "quote_momentum_v2":
                if not chg <= -0.10:
                    continue
            else:
                if not chg < 0.10:
                    continue
            win = outcome == "DOWN"
            wins_cnt[rule].append(win)
            evs[rule].append((0.98 / q - 1.0) if win else -1.0)
            if rule == "quote_contrarian_v2":
                prev = wins.get(ws - 300_000)
                if (prev or {}).get("outcome") == "DOWN":
                    win3 = win
                    wins_cnt["quote_contrarian_v3a"].append(win3)
                    evs["quote_contrarian_v3a"].append((0.98 / q - 1.0) if win3 else -1.0)
                    dh = day_high_before(prefix, ts)
                    if dh and (cur - dh) / dh * 100.0 <= -0.30:
                        wins_cnt["quote_contrarian_v3b"].append(win3)
                        evs["quote_contrarian_v3b"].append((0.98 / q - 1.0) if win3 else -1.0)

        # ---- x4 族 ----
        if outcome == "UP":
            end_pct = curve_end_pct(w.get("curve_up_pct"))
            if end_pct is not None and end_pct <= 40.0:
                base = _x4_base_local(wins, ws)
                cur = window_open_btc(w)
                if base is not None and cur is not None \
                        and abs((cur - base) / base * 100.0) < 0.5:
                    tgt = wins.get(ws + 300_000)
                    if tgt is not None and tgt.get("outcome") in ("UP", "DOWN"):
                        q = price_within(tgt.get("curve_down_price"), tgt["start_time"], 150.0)
                        if q is not None and q > 0:  # 干净口径：仅真实价
                            win = tgt["outcome"] == "DOWN"
                            wins_cnt["x4_v2"].append(win)
                            evs["x4_v2"].append(
                                (0.98 / min(max(q + 0.01, 0.01), 0.99) - 1.0)
                                if win else -1.0)

    stats = {}
    for v in VERSIONS:
        n = len(wins_cnt[v])
        k = sum(wins_cnt[v])
        stats[v] = {"n": n, "wins": k,
                    "wr": k / n if n else None,
                    "wilson": wilson(k, n) if n else (None, None),
                    "avg_ev": sum(evs[v]) / len(evs[v]) if evs[v] else None}
    return {"stats": stats, "range": (t0, t1), "n_windows": len(ordered)}


def _x4_base_local(wins: dict, ws: int) -> float | None:
    target = ws - 3_600_000
    best_t, best_p = None, None
    for s in range(target - 600_000, target + 600_000 + 1, 300_000):
        w = wins.get(s)
        if w and w.get("entry_price"):
            if best_t is None or s > best_t:
                best_t, best_p = s, float(w["entry_price"])
    return best_p


def main() -> int:
    sys.stdout = Tee()
    print("=" * 92)
    print("影子 v2/v3 五版本真实数据回测（A=线上全历史 IS / B=本地 07-13~07-30 OOS 盲验）")
    print("=" * 92)

    a = online_segment()
    b = local_segment()

    def fmt(ts_ms):
        import datetime as dt
        return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")

    print(f"\nA 段：线上窗口 {a['n_windows']} 个（{fmt(a['range'][0])} ~ {fmt(a['range'][1])}）"
          f"；v1 行过门禁=该版本历史真实笔")
    print(f"B 段：本地窗口 {b['n_windows']} 个（{fmt(b['range'][0])} ~ {fmt(b['range'][1])}）"
          f"；全曲线重放（门禁归因未接触，盲验）")
    print(f"\n{'版本':<22}{'段':<4}{'n':>5}{'胜率':>8}{'Wilson95%':>16}{'avgEV':>9}")
    report = {}
    for v in VERSIONS:
        ra, rb = a["stats"][v], b["stats"][v]
        n = ra["n"] + rb["n"]
        k = ra["wins"] + rb["wins"]
        lo, hi = wilson(k, n)
        report[v] = {"online": ra, "local": rb,
                     "combined": {"n": n, "wins": k,
                                  "wr": k / n if n else None,
                                  "wilson": (lo, hi)}}
        for tag, r in (("A", ra), ("B", rb)):
            wr = f"{r['wr']:.1%}" if r["wr"] is not None else "-"
            wlo, whi = r["wilson"]
            wci = f"[{wlo:.1%},{whi:.1%}]" if r["n"] else "-"
            ev = f"{r['avg_ev']:+.3f}" if r["avg_ev"] is not None else "-"
            print(f"{v:<22}{tag:<4}{r['n']:>5}{wr:>8}{wci:>16}{ev:>9}")
        wr = f"{k / n:.1%}" if n else "-"
        print(f"{v:<22}{'合':<4}{n:>5}{wr:>8}[{lo:.1%},{hi:.1%}]{'':>9}")

    print("\n---- 建议 SHADOW_BENCH 回填（合并真实数据胜率；EV 取 A 段线上实测口径）----")
    for v in VERSIONS:
        c = report[v]["combined"]
        ev = report[v]["online"]["avg_ev"]
        ev_s = f"{ev:.3f}" if ev is not None else "None"
        print(f'    "{v}": ({c["wr"]:.3f}, {ev_s})  # n={c["n"]}'
              f"（A {report[v]['online']['n']} + B {report[v]['local']['n']}）")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已存 {OUT}；日志 {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
