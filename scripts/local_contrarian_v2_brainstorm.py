#!/usr/bin/env python3
"""contrarian_v2 脑洞归因 round 2：跳出传统参数切片的框架。

前一轮（local_contrarian_v2_improve.py）结论：chg/q/t/时段等传统维度全部
过不了 Wilson 下界。本轮换数据源 + 换视角：

新数据源（此前未用过）：
  - curve_down_pct 情绪曲线（x4 家族的本源数据）→ 情绪 vs 报价错位
  - 窗口链（4000 窗全量）→ 前窗 streak、触发聚集、日内高点位置
  - 触发后 BTC 路径 → S5 确认思想的 5m 微观移植

脑洞维度（每个都带机制故事）：
  A. 情绪错位度 = down_pct@触发 − q：人看跌比例 vs 市场定价的差。
     错位大 = 报价跌得比情绪快（做市商主导），回归力应更强（x4 思想移植）
  B. 情绪跟随度 = down_pct 窗内变化：报价跌+情绪没跟（假恐慌）vs 情绪跟着跌（真恐慌）
  C. 前窗 streak：连续 UP 窗后散户看涨惯性 → DOWN 便宜的过度反应更真？
  D. S5 确认移植：触发后 15-30s BTC 方向（涨势衰竭确认 vs 动量仍在）
     注意：这是"延迟入场变体"——t+15s 决策时该信息 ex-ante 合法，
     但入场价会漂移（无 t+15s 报价数据，用 q0 代理并标注偏差）
  E. 窗内微波动率（前 45s BTC 已实现波动）：高频抖动 vs 平滑上涨的均值回归差异
  F. 15m 周期内位置（第 1/2/3 个 5m 窗）：跨时间尺度边界效应
  G. 日内高点回撤位置：触发时 BTC 距当日 running high 的距离（趋势日 vs 震荡日）
  H. 触发聚集：当日第几次触发 / 距上次触发间隔（模式重复衰减）
  I. 前窗幅度：前窗大涨 vs 小涨后的 contrarian

视角转换（不依赖统计的确定性改善）：
  Z. 盈亏平衡线 = q̄/0.98 是确定性的——等更深折扣（q 更低子集）不依赖
     胜率估计就能压低盈亏平衡线。q 档 × EV 的确定性数学。

纪律：141 笔切 9 维度，多重检验更严重；每格标 Wilson CI；
结论只作为 v3 影子假设排序，不动实盘。
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import urllib.request

BASE = "http://165.154.147.155:8082"
LOG = "output/contrarian_v2_brainstorm.log"
CACHE = "output/.cv2_bs_cache.json"


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


def at_or_before(curve: list[dict], ts: int) -> float | None:
    best = None
    for p in curve or []:
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts:
            best = float(v)
    return best


def next_after(curve: list[dict], ts: int) -> float | None:
    for p in sorted(curve or [], key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is not None and v is not None and int(t) > ts:
            return float(v)
    return None


def main() -> int:
    sys.stdout = Tee()

    # ---------- 数据（缓存避免重复拉 12s 的 windows） ----------
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        v1s = {int(k): v for k, v in cached["v1s"].items()}
        v2s = {int(k): v for k, v in cached["v2s"].items()}
        wins = {int(k): v for k, v in cached["wins"].items()}
    else:
        v1 = get("/api/misalignment/signals?version=quote_contrarian_v1&limit=300")["signals"]
        v2 = get("/api/misalignment/signals?version=quote_contrarian_v2&limit=300")["signals"]
        v1s = {int(r["window_start"]): r for r in v1 if r.get("settle_outcome") in ("UP", "DOWN")}
        v2s = {int(r["window_start"]): r for r in v2 if r.get("settle_outcome") in ("UP", "DOWN")}
        wins = {int(w["start_time"]): w for w in get("/api/sentiment/windows?limit=4000")}
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump({"v1s": v1s, "v2s": v2s, "wins": wins}, f, ensure_ascii=False)
    print(f"v1 可结算 {len(v1s)} | v2 可结算 {len(v2s)} | windows {len(wins)}")

    # ---------- 日内 running high（全部窗口 BTC 曲线拼接） ----------
    all_pts = []
    for w in wins.values():
        for p in w.get("curve_btc_price") or []:
            if p.get("t") is not None and p.get("v"):
                all_pts.append((int(p["t"]), float(p["v"])))
    all_pts.sort()
    day_max: dict[int, float] = {}
    for t, v in all_pts:
        d = t // 86_400_000
        day_max[d] = max(day_max.get(d, 0.0), v)
    # 触发时刻前的当日高点：用逐点扫描（简单法：按日分组后前缀 max）
    by_day: dict[int, list[tuple[int, float]]] = {}
    for t, v in all_pts:
        by_day.setdefault(t // 86_400_000, []).append((t, v))
    prefix_max: dict[int, list[tuple[int, float]]] = {}
    for d, pts in by_day.items():
        run = -1.0
        out = []
        for t, v in pts:
            run = max(run, v)
            out.append((t, run))
        prefix_max[d] = out

    def day_high_before(ts: int) -> float | None:
        d = ts // 86_400_000
        pts = prefix_max.get(d)
        if not pts:
            return None
        best = None
        for t, m in pts:      # 已按 t 升序
            if t <= ts:
                best = m
            else:
                break
        return best

    # ---------- 逐笔特征重建 ----------
    rows = []
    for ws, s in sorted(v1s.items()):
        w = wins.get(ws)
        if w is None or not w.get("curve_btc_price") or not w.get("curve_down_pct"):
            continue
        ts = int(s["entry_quote_ts"]) if s.get("entry_quote_ts") else None
        base = w.get("entry_price")
        if ts is None or not base or base <= 0:
            continue
        q = s["entry_down_price"]
        btc_curve = w["curve_btc_price"]
        sent_curve = w["curve_down_pct"]
        trig = at_or_before(btc_curve, ts)
        if trig is None:
            continue
        sent_at = at_or_before(sent_curve, ts)
        sent_0 = at_or_before(sent_curve, ws + 1000)  # 窗口开始处情绪
        # A. 情绪错位度
        misalign = (sent_at - q * 100.0) if sent_at is not None else None  # down_pct 是百分数
        # B. 情绪跟随（窗开→触发 down_pct 变化，pp）
        sent_chg = (sent_at - sent_0) if (sent_at is not None and sent_0 is not None) else None
        # D. S5 确认移植：触发后下一个采样（≈+15s）BTC 方向
        nxt = next_after(btc_curve, ts)
        post15_dir = (1 if nxt > trig else (-1 if nxt < trig else 0)) if nxt is not None else None
        # E. 窗内微波动率（前 45s 逐点收益率 std）
        pre = sorted([p for p in btc_curve if p.get("t") is not None and ws <= int(p["t"]) <= ts],
                     key=lambda x: x["t"])
        rets = []
        for a, b in zip(pre, pre[1:]):
            if a["v"] and b["v"] and a["v"] > 0:
                rets.append(b["v"] / a["v"] - 1)
        micro_vol = float(np_std(rets)) if len(rets) >= 3 else None
        # F. 15m 周期内位置
        pos15 = (ws // 300_000) % 3
        # G. 日内高点回撤
        dh = day_high_before(ts)
        dd_from_high = (trig - dh) / dh * 100.0 if dh else None
        # C/H/I. 前窗结构
        prev = wins.get(ws - 300_000)
        prev_out = (prev or {}).get("outcome")
        prev2 = wins.get(ws - 600_000)
        streak_up = 0
        pw = ws - 300_000
        while wins.get(pw, {}).get("outcome") == "UP":
            streak_up += 1
            pw -= 300_000
        prev_ret = (prev or {}).get("actual_return")
        # H. 触发聚集
        day = ws // 86_400_000
        rows.append({
            "ws": ws, "v2": ws in v2s, "q": q, "ev": s["ev_at_entry"],
            "win": s["settle_outcome"] == "DOWN", "day": day,
            "misalign": misalign, "sent_chg": sent_chg, "post15_dir": post15_dir,
            "micro_vol": micro_vol, "pos15": pos15, "dd_from_high": dd_from_high,
            "prev_out": prev_out, "prev2_out": (prev2 or {}).get("outcome"),
            "streak_up": streak_up, "prev_ret": prev_ret,
        })
    # 触发聚集：按天编号
    day_seq: dict[int, int] = {}
    for r in rows:
        day_seq[r["day"]] = day_seq.get(r["day"], 0) + 1
        r["trig_idx_day"] = day_seq[r["day"]]
    # 距上次触发间隔
    last_t = None
    for r in rows:
        r["gap_min"] = (r["ws"] - last_t) / 60_000 if last_t else None
        last_t = r["ws"]
    print(f"特征重建 {len(rows)} 笔")
    v2r = [r for r in rows if r["v2"]]

    def table(grp, key_name, key_fn, order=None):
        print(f"\n----- {key_name} -----")
        buckets: dict = {}
        for r in grp:
            k_ = key_fn(r)
            if k_ is None:
                continue
            buckets.setdefault(k_, []).append(r)
        keys = order or sorted(buckets.keys(), key=str)
        for k_ in keys:
            g = buckets.get(k_)
            if not g:
                continue
            n = len(g); w = sum(r["win"] for r in g)
            lo, hi = wilson(w, n)
            print(f"  {str(k_):<22} n={n:>3} 胜率={w/n:6.1%} [{lo:5.1%},{hi:5.1%}]"
                  f" q̄={sum(r['q'] for r in g)/n:.3f} EV={sum(r['ev'] for r in g)/n:+7.3f}")
        n = len(grp); w = sum(r["win"] for r in grp)
        print(f"  {'合计':<22} n={n:>3} 胜率={w/n:6.1%} EV={sum(r['ev'] for r in grp)/n:+7.3f}")

    # ===================== A. 情绪错位度 =====================
    print()
    print("=" * 96)
    print("A. 情绪错位度 = down_pct@触发 − q×100（人看跌比例 − 市场定价，pp）")
    print("   机制：错位>0 = 情绪比定价更悲观（报价跌过头，做市商主导）→ 回归应更强")
    print("=" * 96)
    table(v2r, "错位度（v2 内）", lambda r: (
        "错位>5pp（情绪更悲观）" if r["misalign"] is not None and r["misalign"] > 5 else
        "|错位|≤5pp（基本一致）" if r["misalign"] is not None and r["misalign"] >= -5 else
        "错位<-5pp（定价更悲观）" if r["misalign"] is not None else None))

    # ===================== B. 情绪跟随度 =====================
    print()
    print("=" * 96)
    print("B. 情绪跟随度 = down_pct 窗开→触发 变化（pp）：报价跌时人有没有跟着看跌")
    print("=" * 96)
    table(v2r, "情绪跟随（v2 内）", lambda r: (
        "情绪转跌>2pp（人跟着跌）" if r["sent_chg"] is not None and r["sent_chg"] > 2 else
        "情绪基本没动|Δ|≤2" if r["sent_chg"] is not None and r["sent_chg"] >= -2 else
        "情绪转涨<-2pp（人反而看涨）" if r["sent_chg"] is not None else None))

    # ===================== C. 前窗 streak =====================
    print()
    print("=" * 96)
    print("C. 前窗连续 UP 数（散户看涨惯性）：streak 越长，DOWN 便宜越是过度反应？")
    print("=" * 96)
    table(v2r, "前窗连续UP数（v2 内）", lambda r: (
        "streak≥2" if r["streak_up"] >= 2 else
        "streak=1" if r["streak_up"] == 1 else
        "streak=0（前窗DOWN）"))
    table(v2r, "前窗方向（v2 内）", lambda r: r["prev_out"] or None, order=["UP", "DOWN"])

    # ===================== D. S5 确认移植（延迟入场变体） =====================
    print()
    print("=" * 96)
    print("D. S5 确认思想移植：触发后 +15s BTC 方向（动量仍在 vs 衰竭确认）")
    print("   注意：这是延迟入场变体——t+15s 决策合法，但实际入场价≠q0（未建模，标注）")
    print("=" * 96)
    table(v2r, "触发后+15s BTC（v2 内）", lambda r: (
        "继续涨（动量仍在）" if r["post15_dir"] == 1 else
        "回落（衰竭确认）" if r["post15_dir"] == -1 else
        "持平" if r["post15_dir"] == 0 else None))

    # ===================== E. 微波动率 =====================
    print()
    print("=" * 96)
    print("E. 窗内微波动率（前 45s 逐采样收益率 std，bps）")
    print("=" * 96)
    def mv_key(r):
        if r["micro_vol"] is None:
            return None
        v = r["micro_vol"] * 1e4
        med = sorted(x["micro_vol"] for x in v2r if x["micro_vol"] is not None)
        m = med[len(med) // 2] * 1e4
        return "低微波动(<中位)" if v < m else "高微波动(≥中位)"
    table(v2r, "微波动率（v2 内）", mv_key)

    # ===================== F. 15m 周期内位置 =====================
    print()
    print("=" * 96)
    print("F. 15m 周期内第几个 5m 窗（跨时间尺度边界效应）")
    print("=" * 96)
    table(v2r, "15m 内位置（v2 内）", lambda r: f"第{r['pos15'] + 1}个5m窗")

    # ===================== G. 日内高点回撤 =====================
    print()
    print("=" * 96)
    print("G. 触发时 BTC 距当日 running high 的距离（趋势日 vs 震荡日微观版）")
    print("=" * 96)
    table(v2r, "距日高回撤（v2 内）", lambda r: (
        "距日高≤0.05%（贴着日高）" if r["dd_from_high"] is not None and r["dd_from_high"] > -0.05 else
        "距日高0.05~0.3%" if r["dd_from_high"] is not None and r["dd_from_high"] > -0.3 else
        "距日高>0.3%（已回落）" if r["dd_from_high"] is not None else None))

    # ===================== H. 触发聚集 =====================
    print()
    print("=" * 96)
    print("H. 触发聚集：当日第几次触发 / 距上次触发间隔")
    print("=" * 96)
    table(v2r, "当日触发序号（v2 内）", lambda r: (
        "当日首次" if r["trig_idx_day"] == 1 else
        "第2-3次" if r["trig_idx_day"] <= 3 else "第4+次"))
    table(v2r, "距上次触发（v2 内）", lambda r: (
        "≤30min（密集）" if r["gap_min"] is not None and r["gap_min"] <= 30 else
        "30-120min" if r["gap_min"] is not None and r["gap_min"] <= 120 else
        ">2h（孤立）" if r["gap_min"] is not None else None))

    # ===================== I. 前窗幅度 =====================
    print()
    print("=" * 96)
    print("I. 前窗涨幅（大涨后的 contrarian vs 小涨后）")
    print("=" * 96)
    table(v2r, "前窗 actual_return（v2 内）", lambda r: (
        "前窗大涨>0.15%" if r["prev_ret"] is not None and r["prev_ret"] > 0.0015 else
        "前窗小涨0~0.15%" if r["prev_ret"] is not None and r["prev_ret"] > 0 else
        "前窗跌" if r["prev_ret"] is not None else None))

    # ===================== Z. 确定性视角：q 档 × 盈亏平衡数学 =====================
    print()
    print("=" * 96)
    print("Z. 视角转换：盈亏平衡线 q̄/0.98 是确定性的 —— 等更深折扣的数学")
    print("=" * 96)
    print("  （胜率不是终极目标，EV 才是：EV = P×(0.98/q−1) − (1−P)，q 直接进公式）")
    for lo_, hi_ in ((0.15, 0.18), (0.18, 0.21), (0.21, 0.25), (0.15, 0.20), (0.20, 0.25)):
        g = [r for r in v2r if lo_ <= r["q"] < hi_]
        if not g:
            continue
        n = len(g); w = sum(r["win"] for r in g)
        p_ = w / n
        qb = sum(r["q"] for r in g) / n
        be = qb / 0.98
        lo, hi = wilson(w, n)
        print(f"  q∈[{lo_:.2f},{hi_:.2f}): n={n:>3} P={p_:5.1%} [{lo:5.1%},{hi:5.1%}]"
              f" q̄={qb:.3f} 盈亏平衡={be:5.1%} 实测EV={sum(r['ev'] for r in g)/n:+.3f}")
    print("  → 若胜率恒定，q̄ 每 −0.03 盈亏平衡线降 ~3pp；这是不依赖统计估计的确定性改善，")
    print("    代价是频率下降 + 深折扣样本可能自选择（越深越接近真跌）。")

    # ===================== 汇总排序 =====================
    print()
    print("=" * 96)
    print("总结：脑洞维度证据板（机制可信度 × CI 下界 vs 盈亏平衡线）")
    print("=" * 96)
    print("  详见上方各表；判定标准 = Wilson 下界 > 该格 q̄/0.98 才算过线。")
    print("  多重检验警示：本轮 9 个维度 ≈ 20+ 次比较，出现 1-2 个'过线'纯属噪声期望。")
    print("\n结果已存日志 output/contrarian_v2_brainstorm.log")
    return 0


def np_std(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


if __name__ == "__main__":
    sys.exit(main())
