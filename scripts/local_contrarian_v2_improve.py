#!/usr/bin/env python3
"""contrarian_v2 胜率提升归因：在 141 笔可结算样本上找可行的 v3 门禁方向。

数据：
  1. 线上 v1 全量（178 可结算，v2⊂v1，含被门禁剔除的"真冲高"段）
  2. 线上 sentiment windows（08-10 起，curve_btc_price 15s 采样）
     → 逐笔重建：触发时点 chg（比落库门禁二值判断更细）、触发前路径形状

归因维度（按机制可信度排序）：
  A. chg_at_trigger 细分（门禁维度自身的细化——回测归因说平盘窗 |chg|<0.05%
     贡献 86% 利润，线上是否复现？被剔除的 chg≥0.10% 段之外，
     [−x, +0.10%) 内部是否还有更优切点？）
  B. 触发前路径形状：触发前最低点深度（min_chg）vs 触发时位置
     （V 型反弹中的回落 vs 单边阴跌中的低位）
  C. q 细分：[0.15,0.25) 内三档（越接近 0.25 = 市场越不那么恐慌）
  D. t_rel：触发秒位置（早触发 = 快速跌入区间；晚触发 = 阴跌慢入）
  E. 时段（UTC 小时）
  F. chg × q 交叉

纪律声明（必须正视的多重检验风险）：
  v2 本身已是"v1 → 归因 → 门禁"挖了一轮的产物；本分析在同一 141 笔上
  继续切分，每格 n≈30 时 Wilson CI 宽达 ±15pp，任何"提升"都可能是噪声。
  结论只能作为【影子 v3 假设】提出，需前向验证（只加记录不下注 ≥100 笔）
  后才能动实盘。EV 口径 = 0.98/q−1 / −1（与落库一致）。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/contrarian_v2_improve.log"


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


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=120) as r:
        return json.loads(r.read().decode())


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def btc_at_or_before(curve: list[dict], ts: int) -> float | None:
    best = None
    for p in curve or []:
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts:
            best = float(v)
    return best


def main() -> int:
    sys.stdout = Tee()

    # ---------- 数据拉取 ----------
    v1 = get("/api/misalignment/signals?version=quote_contrarian_v1&limit=300")["signals"]
    v2 = get("/api/misalignment/signals?version=quote_contrarian_v2&limit=300")["signals"]
    v1s = {int(r["window_start"]): r for r in v1 if r.get("settle_outcome") in ("UP", "DOWN")}
    v2s = {int(r["window_start"]): r for r in v2 if r.get("settle_outcome") in ("UP", "DOWN")}
    wins_raw = get("/api/sentiment/windows?limit=4000")
    wins = {int(w["start_time"]): w for w in wins_raw}
    print(f"v1 可结算 {len(v1s)} | v2 可结算 {len(v2s)} | windows {len(wins)}"
          f"（{dt.datetime.fromtimestamp(min(wins)/1000, tz=dt.timezone.utc):%m-%d}"
          f"~{dt.datetime.fromtimestamp(max(wins)/1000, tz=dt.timezone.utc):%m-%d}）")

    # ---------- 逐笔重建特征 ----------
    rows = []
    n_no_win = 0
    for ws, s in sorted(v1s.items()):
        w = wins.get(ws)
        if w is None or not w.get("curve_btc_price"):
            n_no_win += 1
            continue
        curve = sorted(w["curve_btc_price"], key=lambda p: p.get("t") or 0)
        base = w.get("entry_price") or (curve[0]["v"] if curve else None)
        ts = int(s["entry_quote_ts"]) if s.get("entry_quote_ts") else None
        if base is None or base <= 0 or ts is None:
            n_no_win += 1
            continue
        trig = btc_at_or_before(curve, ts)
        if trig is None:
            n_no_win += 1
            continue
        chg = (trig - base) / base * 100.0
        # 触发前路径：min/max（相对开盘）
        pre = [float(p["v"]) for p in curve if p.get("t") is not None and int(p["t"]) <= ts]
        pre = [v for v in pre if v > 0]
        min_chg = (min(pre) - base) / base * 100.0 if pre else chg
        max_chg = (max(pre) - base) / base * 100.0 if pre else chg
        rows.append({
            "ws": ws, "v2": ws in v2s,
            "q": s["entry_down_price"], "ev": s["ev_at_entry"],
            "win": s["settle_outcome"] == "DOWN",
            "chg": chg, "min_chg": min_chg, "max_chg": max_chg,
            "t_rel": (ts - ws) / 1000.0,
            "hour": dt.datetime.fromtimestamp(ws / 1000, tz=dt.timezone.utc).hour,
        })
    print(f"特征重建 {len(rows)} 笔（缺窗口/曲线 {n_no_win} 笔）")
    print(f"交叉校验：重建 chg<+0.10% 的笔数 = {sum(1 for r in rows if r['chg'] < 0.10)}"
          f" vs 落库 v2 笔数 = {sum(1 for r in rows if r['v2'])}"
          f"（应接近，差异=曲线采样 vs 门禁取价时点差）")

    def table(grp: list[dict], key_name: str, key_fn, order=None) -> None:
        print(f"\n----- {key_name} -----")
        buckets: dict = {}
        for r in grp:
            buckets.setdefault(key_fn(r), []).append(r)
        keys = order or sorted(buckets.keys(), key=str)
        for k_ in keys:
            g = buckets.get(k_)
            if not g:
                continue
            n = len(g); w = sum(r["win"] for r in g)
            p = w / n
            ev = sum(r["ev"] for r in g) / n
            lo, hi = wilson(w, n)
            print(f"  {str(k_):<18} n={n:>3} 胜率={p:6.1%} [{lo:5.1%},{hi:5.1%}]"
                  f" q̄={sum(r['q'] for r in g)/n:.3f} EV={ev:+7.3f}")
        # 合计
        n = len(grp); w = sum(r["win"] for r in grp)
        print(f"  {'合计':<18} n={n:>3} 胜率={w/n:6.1%} EV={sum(r['ev'] for r in grp)/n:+7.3f}")

    v2rows = [r for r in rows if r["v2"]]

    # ---------- A. chg_at_trigger 细分（含被剔除段，完整映射） ----------
    print()
    print("=" * 96)
    print("A. 触发时点 chg 完整映射（v1 全量 178→重建样本；v2 门禁 = chg<+0.10%）")
    print("=" * 96)
    chg_bins = [(-99, -0.30), (-0.30, -0.15), (-0.15, -0.05), (-0.05, 0.05), (0.05, 0.10), (0.10, 99)]
    def chg_key(r):
        for lo_, hi_ in chg_bins:
            if lo_ <= r["chg"] < hi_:
                return f"[{lo_:+.2f},{hi_:+.2f})"
        return "?"
    table(rows, "chg@触发（v1 全量）", chg_key,
          order=[f"[{a:+.2f},{b:+.2f})" for a, b in chg_bins])
    table(v2rows, "chg@触发（v2 门禁内 141）", chg_key,
          order=[f"[{a:+.2f},{b:+.2f})" for a, b in chg_bins])

    # ---------- B. 触发前路径形状 ----------
    print()
    print("=" * 96)
    print("B. 触发前路径形状（v2 门禁内）：单边阴跌 vs 冲高回落 vs V反弹回落")
    print("=" * 96)
    def shape_key(r):
        # max_chg：触发前最高点；min_chg：触发前最低点
        if r["max_chg"] >= 0.08:                      # 曾明显冲高（≥0.08%）
            return "曾冲高≥0.08%"
        if r["min_chg"] <= -0.15:                     # 曾深跌 ≥0.15%
            return "曾深跌≤-0.15%"
        return "窄幅内"
    table(v2rows, "路径形状", shape_key)
    # 反弹位置：触发价在 [min, max] 区间的位置（0=最低，1=最高）
    def pos_key(r):
        span = r["max_chg"] - r["min_chg"]
        if span < 1e-9:
            return "无波动"
        pos = (r["chg"] - r["min_chg"]) / span
        if pos < 0.33:
            return "位于区间下部"
        if pos < 0.66:
            return "位于区间中部"
        return "位于区间上部"
    table(v2rows, "触发点在窗内区间位置", pos_key)

    # ---------- C. q 细分 ----------
    print()
    print("=" * 96)
    print("C. q 报价细分（v2 门禁内）")
    print("=" * 96)
    q_bins = [(0.15, 0.18), (0.18, 0.21), (0.21, 0.25)]
    def q_key(r):
        for lo_, hi_ in q_bins:
            if lo_ <= r["q"] < hi_:
                return f"q[{lo_:.2f},{hi_:.2f})"
        return "?"
    table(v2rows, "q 档", q_key, order=[f"q[{a:.2f},{b:.2f})" for a, b in q_bins])

    # ---------- D. t_rel ----------
    print()
    print("=" * 96)
    print("D. 触发秒位置（v2 门禁内，规则窗 [45,60)s）")
    print("=" * 96)
    def t_key(r):
        t = r["t_rel"]
        if t < 50:
            return "45-50s 早触发"
        if t < 55:
            return "50-55s 中"
        return "55-60s 晚触发"
    table(v2rows, "t_rel", t_key)

    # ---------- E. 时段 ----------
    print()
    print("=" * 96)
    print("E. UTC 时段（v2 门禁内，4h 一档）")
    print("=" * 96)
    def hour_key(r):
        return f"{(r['hour'] // 4) * 4:02d}-{(r['hour'] // 4) * 4 + 4:02d}h"
    table(v2rows, "UTC 时段", hour_key)

    # ---------- F. chg × q 交叉 ----------
    print()
    print("=" * 96)
    print("F. chg × q 交叉（v2 门禁内）")
    print("=" * 96)
    def chg3(r):
        if r["chg"] < -0.05:
            return "chg<-0.05%"
        if r["chg"] < 0.05:
            return "|chg|<0.05%"
        return "chg≥+0.05%"
    def q2(r):
        return "q<0.21" if r["q"] < 0.21 else "q≥0.21"
    table(v2rows, "chg三档 × q两档", lambda r: f"{chg3(r)} & {q2(r)}")

    # ---------- G. 假设汇总与多重检验警示 ----------
    print()
    print("=" * 96)
    print("G. v3 假设清单（全部需前向影子验证后才可动实盘）")
    print("=" * 96)
    # 平盘假设统计
    flat = [r for r in v2rows if abs(r["chg"]) < 0.05]
    nonflat = [r for r in v2rows if abs(r["chg"]) >= 0.05]
    for name, g in (("平盘窗 |chg|<0.05%", flat), ("非平盘 |chg|≥0.05%", nonflat)):
        if g:
            n = len(g); w = sum(r["win"] for r in g)
            print(f"  {name}: n={n} 胜率={w/n:.1%} EV={sum(r['ev'] for r in g)/n:+.3f}")
    print()
    print("  多重检验警示：本分析在 141 笔上切了 6 个维度，每格 n≈20-50，")
    print("  Wilson CI 宽 ±13~18pp；任何单格'提升'都需视为假设而非结论。")
    print("  正确路径：把候选门禁加进 quote_edge_detector 作 v3 影子（只记录），")
    print("  攒 ≥100 笔前向样本且 CI 下界仍高于盈亏平衡线，才考虑实盘。")
    print("\n结果已存日志 output/contrarian_v2_improve.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
