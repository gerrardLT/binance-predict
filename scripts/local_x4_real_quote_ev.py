#!/usr/bin/env python3
"""X4 错位信号真实报价 EV 重估 + 线上全信号盘点。

X4 权威口径（local_misalignment_scan.py D1_x4_40）：
  当前窗收阳（outcome=UP）但收尾情绪 ≤40（curve_up_pct 末点）→ 次窗 DOWN。
  历史胜率 63.5%，盈亏平衡入场价 = 0.635×0.98 ≈ 0.622。

本脚本：
  1. 全量（10,558 窗）重算 X4 胜率 + Wilson CI + 前后半样本稳健性
  2. 与 5m 市场真实报价（07-26~08-19，136,236 条）配对：
     每个 X4 事件取次窗开盘后 b=0 桶（<1min）首条 down_price → 真实报价 EV
  3. 输出线上信号完整盘点表（S1/S2/S4/S5/X4）

口径：EV = 赢 0.98/q−1 / 输 −1（费 2%）；down_price 视为可成交价。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

WINDOWS = "output/sentiment_windows_online_fixed.json"
SAMPLES = "prediction_market_samples_online_20260819.json"
LOG = "output/x4_real_quote_ev.log"
FEE_RET = 0.98
END_T = 40


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


def last_v(pts) -> float | None:
    if not pts:
        return None
    v = pts[-1].get("v")
    return float(v) if v is not None else None


def main() -> int:
    sys.stdout = Tee()
    with open(WINDOWS, encoding="utf-8") as f:
        W = json.load(f)
    W.sort(key=lambda w: int(w["start_time"]))
    print(f"窗口总数 {len(W)}（{datetime.fromtimestamp(W[0]['start_time']/1000, timezone.utc):%m-%d}"
          f" ~ {datetime.fromtimestamp(W[-1]['start_time']/1000, timezone.utc):%m-%d}）")

    # ---------- 1. X4 识别 + 次窗胜负（全量胜率） ----------
    x4_idx = []
    for i, w in enumerate(W):
        if w.get("outcome") != "UP":
            continue
        end = last_v(w.get("curve_up_pct"))
        if end is not None and end <= END_T:
            x4_idx.append(i)
    print(f"X4 窗 {len(x4_idx)} 个（收阳 & 收尾情绪≤{END_T}）")

    events = []  # (start_next, win)
    for i in x4_idx:
        if i + 1 >= len(W):
            continue
        nx = W[i + 1]
        if int(nx["start_time"]) - int(W[i]["start_time"]) != 300_000:
            continue
        if nx.get("outcome") not in ("UP", "DOWN"):
            continue
        events.append({"t_next": int(nx["start_time"]),
                       "win": nx["outcome"] == "DOWN"})
    wins = sum(e["win"] for e in events)
    n = len(events)
    lo, hi = wilson(wins, n)
    print(f"\n===== 1. X4 全量胜率（次窗 DOWN）=====")
    print(f"  可判定事件 {n}，胜率 {wins/n:.1%}  Wilson [{lo:.1%}, {hi:.1%}]")
    print(f"  盈亏平衡入场价 = {wins/n:.3f} × 0.98 = {wins/n*FEE_RET:.3f}")
    half = n // 2
    for name, seg in (("前半", events[:half]), ("后半", events[half:])):
        k_ = sum(e["win"] for e in seg)
        print(f"  {name}样本: n={len(seg)} 胜率 {k_/len(seg):.1%}")

    # ---------- 2. 5m 市场真实报价配对 ----------
    with open(SAMPLES, encoding="utf-8") as f:
        S = json.load(f)
    s5 = [s for s in S if s.get("market_period") == "5m"]
    bywin: dict[int, list] = defaultdict(list)
    for s in s5:
        q = s.get("down_price"); ts = int(s["timestamp"])
        if q is None or q <= 0.01 or q >= 0.99:
            continue
        bywin[ts // 300_000].append((ts, float(q)))
    for v in bywin.values():
        v.sort()
    t0 = min(bywin) * 300_000
    print(f"\n===== 2. X4 × 5m 真实报价配对（{datetime.fromtimestamp(t0/1000, timezone.utc):%m-%d} 起） =====")

    pairs = []  # (q, win, b)
    for e in events:
        cyc = e["t_next"] // 300_000
        if cyc not in bywin:
            continue
        for ts, q in bywin[cyc]:
            b = (ts - cyc * 300_000) // 60_000
            if b > 1:
                break
            pairs.append((q, e["win"], b))
            break  # 每事件仅取首条（最早报价）
    print(f"  配对事件 {len(pairs)} / {n}（次窗开盘后 0-1 分钟首条报价）")
    if pairs:
        qs = np.array([p[0] for p in pairs])
        evs = np.array([(FEE_RET / q - 1.0) if w else -1.0 for q, w, _ in pairs])
        k_ = sum(p[1] for p in pairs)
        p_ = k_ / len(pairs)
        print(f"  胜率 {p_:.1%}，q̄={qs.mean():.3f} [min {qs.min():.2f} / max {qs.max():.2f}]")
        print(f"  真实报价 EV = {evs.mean():+.3f} ± {evs.std(ddof=1)/np.sqrt(len(evs)):.3f}")
        # b 分桶
        for b in (0, 1):
            seg = [(q, w) for q, w, bb in pairs if bb == b]
            if len(seg) >= 20:
                ev_ = float(np.mean([(FEE_RET/q - 1.0) if w else -1.0 for q, w in seg]))
                q_ = float(np.mean([q for q, _ in seg]))
                print(f"    b={b} 桶: n={len(seg)} q̄={q_:.3f} EV={ev_:+.3f}")
        # 前后半稳健性
        hh = len(pairs) // 2
        for name, seg in (("前半", pairs[:hh]), ("后半", pairs[hh:])):
            if not seg:
                continue
            ev_ = float(np.mean([(FEE_RET/q - 1.0) if w else -1.0 for q, w, _ in seg]))
            print(f"    {name}样本: n={len(seg)} EV={ev_:+.3f}")

    # ---------- 3. 线上信号完整盘点表 ----------
    print("\n===== 3. 线上信号真实报价 EV 完整盘点 =====")
    print("  信号 | 720d胜率    | 回测假设EV | 真实报价q̄ | 真实EV      | 实盘EV     | 判定")
    print("  " + "-" * 96)
    rows = [
        ("S1 假突破延续", "58.6%", "+0.14(@0.51)", "0.547(6注)", "+0.04~+0.31*", "-0.081",
         "边际薄：t=1回落EV+0.313最优但执行窗口≤1min；直入靠运气价"),
        ("S2 次周期回落", "WEAK", "—", "—", "≈0(市场1min定价)", "—",
         "否：5m市场上涨/回落信息瞬时定价，无edge"),
        ("S4 12min确认", "高", "+0.07(@0.51)", "0.550(6注)", "+0.129", "+0.129",
         "最稳：实盘验证，报价仍便宜"),
        ("S5 5min确认", "76.4%", "+0.468(@0.51)", "0.712(3注)", "+0.060", "-0.244",
         "从神坛跌落：确认后报价吃掉大部分edge，中档被过度定价"),
        (f"X4 收阳情绪≤40", f"{wins/n:.1%}" if n else "—", "+0.24(@0.622盈亏线)",
         "见上", "见上", "+0.531(1注)", "贴打平线入场，待真实报价结论"),
    ]
    for r in rows:
        print(f"  {r[0]:<14}| {r[1]:<10}| {r[2]:<13}| {r[3]:<12}| {r[4]:<12}| {r[5]:<10}| {r[6]}")

    with open("output/x4_real_quote_ev_result.json", "w", encoding="utf-8") as f:
        json.dump({"n_total": len(W), "n_x4": len(x4_idx), "n_events": n,
                   "winrate": wins / n if n else None,
                   "wilson": [lo, hi],
                   "pairs": [{"q": q, "win": w, "b": b} for q, w, b in pairs]},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/x4_real_quote_ev_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
