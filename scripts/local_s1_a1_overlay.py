#!/usr/bin/env python3
"""S1×A1 执行层结合回放：S1 持仓周期内的 5min 情绪收尾背离止盈信号（方案 B）。

不动 S1 判定，只在其持仓周期（次 15m 周期）内的 5min 窗结算点检测 A1 型背离：
  决策点 = 周期内第 1/2 个 5min 窗收盘时（pos2 结算时周期已结束，无止盈意义）
  危险信号（对 DOWN 持仓，S1 用）= Δ ≥ +T 且该窗结算 DOWN
    （人群情绪大幅转多但价格收跌 → A1 口径预测后续 UP → 应止盈/放弃加仓）
  对称信号（对 UP 持仓，S2 用）= Δ ≤ −T 且该窗结算 UP

两层验证：
  L1 大样本通用检验（全部 15m 周期，不限 S1）：
    目标 a：下一 5min 窗顺 Δ 方向（A1 原口径复验）
    目标 b：本 15m 周期结算收 UP（执行层意义：止盈决策的输赢）
    阈值 T ∈ {10, 15, 20}（10=A1 冻结基础口径，20=增强口径，15=灵敏度）；
    对照池 = |Δ|≥10 的全部决策点（与 A1 池一致），背离增量即信号价值。
  L2 S1/S4 环境内佐证：情绪数据时段（17.2 天）内 S1/S4 命中的持仓周期逐条明细。

纪律：这是执行层回放不是新判定模式——T 直接沿用 A1 已冻结口径，不重新选阈值；
IS/OOS 沿用情绪曲线轮 70/30 时间切分；样本量如实报告。S1 判定复刻线上
detector（破位 eps=0.0005 × close_pos≥0.85 × pos4h≥0.9，4h=前48根5m closes）。

用法：python scripts/local_s1_a1_overlay.py [--from-file sentiment_windows.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

from binance_predict.backtest.data import aggregate_15m  # noqa: E402
from local_sentiment_curve_discovery import HOLDOUT_RATIO, build_records  # noqa: E402
from local_vshape_divergence import binom_p, fmt  # noqa: E402
from binance_predict.services.verification import wilson_bounds  # noqa: E402

EPS = 0.0005            # 破位阈值（对齐线上 DEFAULT_SCENE_PARAMS）
CLOSE_POS_MIN = 0.85    # S1/S4 光头下限
POS4H_WIN = 16          # S1: 4h 区间窗口（16 根 15m 收盘，含当前根）
POS4H_MIN = 0.9         # S1: F25 上沿阈值
STREAK_BULL_MIN = 3     # S4: 连阳根数（含信号 K）
LOOKBACK_4H = 48        # 4h 位势 = 前 48 根 5m closes
THRESHOLDS = (10, 15, 20)
KLINE_CACHE = "output/klines_5m_cache.json"
OUT = "output/s1_a1_overlay.json"


# ============================================================
# K 线与场景判定（复刻线上 detector 口径）
# ============================================================

def load_klines() -> list[tuple]:
    """读 360 天缓存 → [(open_time_ms, o, h, l, c, v), ...] 升序。"""
    with open(KLINE_CACHE, encoding="utf-8") as f:
        raw = json.load(f)
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in raw]


def detect_scenes(c5: list[tuple]) -> tuple[dict, dict]:
    """15m 聚合 + S1/S4 判定。

    Returns:
        (scenes, cyc_index)
        scenes[cyc] = {"s1"/"s4": bool, "broke_high": bool, "close_pos"/"pos4h": f,
                       "next": "UP"|"DOWN"|None}   # next = 次周期结算（周期锚点）
        cyc_index = 周期号 → o15/h15/l15/c15
    """
    agg = aggregate_15m(c5)
    cycs = agg["cycs"]
    o15, h15, l15, c15 = agg["o15"], agg["h15"], agg["l15"], agg["c15"]
    cyc_index = {c: j for j, c in enumerate(cycs)}

    cl5 = np.array([r[4] for r in c5], dtype=float)
    h5 = np.array([r[2] for r in c5], dtype=float)
    cont = agg["cont"]

    # 每 15m 周期破位检查（对齐 events.py：首破位 bar，要求 cont 且有完整 48 根历史；
    # 未破位继续看周期内下一根，破位即停——每级别每方向每周期首次）
    def broke_high(cyc: int) -> bool:
        for i in agg["buckets"][cyc]:
            if not cont[i] or i < LOOKBACK_4H:
                continue
            res = float(cl5[i - LOOKBACK_4H:i].max())
            if h5[i] > res * (1 + EPS):
                return True
        return False

    scenes: dict[int, dict] = {}
    for j, cyc in enumerate(cycs):
        o, h, l, c = o15[j], h15[j], l15[j], c15[j]
        rng = h - l
        cp = (c - l) / rng if rng > 0 and o > 0 else None
        # pos4h：最近 16 根 15m 收盘（含当前）区间位置
        p4 = None
        if j >= POS4H_WIN - 1:
            seg = c15[j - POS4H_WIN + 1: j + 1]
            if max(seg) > min(seg):
                p4 = (c - min(seg)) / (max(seg) - min(seg))
        bh = broke_high(cyc)
        green = c > o
        s1 = bh and green and cp is not None and cp >= CLOSE_POS_MIN \
            and p4 is not None and p4 >= POS4H_MIN
        # S4：无 high 破位的周期才查（对齐 detector：S1 优先，破位周期不重复计 S4）
        streak_ok = j >= STREAK_BULL_MIN - 1 and all(
            c15[j - k] > o15[j - k] for k in range(1, STREAK_BULL_MIN))
        s4 = (not bh) and green and cp is not None and cp >= CLOSE_POS_MIN and streak_ok
        nxt = None
        if j + 1 < len(cycs) and cycs[j + 1] == cyc + 1 and o15[j + 1] > 0:
            nxt = "DOWN" if c15[j + 1] < o15[j + 1] else "UP"
        scenes[cyc] = {"s1": bool(s1), "s4": bool(s4), "broke_high": bool(bh),
                       "close_pos": cp, "pos4h": p4, "next": nxt,
                       "c_open": o, "c_close": c}
    return scenes, cyc_index


# ============================================================
# 决策点构建（情绪窗 × 15m 周期对齐）
# ============================================================

def build_points(W: list[dict], recs: list[dict]) -> list[dict]:
    """每窗 → 决策点记录（pos∈{0,1}，曲线 Δ 与 5min 市场结算）。"""
    pts: list[dict] = []
    for j, (w, r) in enumerate(zip(W, recs)):
        cur = sorted(w["curve_up_pct"], key=lambda p: p["t"])
        if len(cur) < 2 or r["out"] not in ("UP", "DOWN"):
            continue
        delta = float(cur[-1]["v"] - cur[0]["v"])
        st = int(w["start_time"])
        pts.append({
            "j": j, "cyc": st // 900_000, "pos": (st % 900_000) // 300_000,
            "delta": delta, "out": r["out"], "start": st,
            "next_out": r["next_out"], "has_next": r["has_next"],
        })
    return pts


def stat_mask(sub: list, key: str, base: float) -> dict | None:
    """决策点子集 → {n,k,p,ci,dev,pval}（key = 目标 0/1 值）。"""
    vals = [p[key] for p in sub if p.get(key) is not None]
    n = len(vals)
    if n == 0:
        return None
    k = int(sum(vals))
    p = k / n
    lo, hi = wilson_bounds(k, n)
    return {"n": n, "k": k, "p": p, "ci": (lo, hi), "dev": p - base,
            "pval": binom_p(k, n, base)}


def run(W: list[dict], out_path: str) -> dict:
    W = sorted(W, key=lambda w: int(w["start_time"]))
    split_ts = int(W[int(len(W) * (1 - HOLDOUT_RATIO))]["start_time"])
    recs = build_records(W)
    t0, t1 = int(W[0]["start_time"]), int(W[-1]["end_time"])
    print(f"[数据] 情绪窗 {len(W)} | {time.strftime('%m-%d %H:%M', time.gmtime(t0/1000))} → "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(t1/1000))} UTC | 70/30 切分 "
          f"{time.strftime('%m-%d %H:%M', time.gmtime(split_ts/1000))}")

    c5 = load_klines()
    scenes, cyc_index = detect_scenes(c5)

    # 数据质量抽查：情绪窗结算 vs 同期 5m K 线方向一致率（对齐校验）
    k5 = {r[0]: r for r in c5}
    n_chk = n_agree = 0
    for w, r in zip(W, recs):
        if r["out"] not in ("UP", "DOWN"):
            continue
        bar = k5.get(int(w["start_time"]))
        if not bar or bar[1] <= 0 or bar[4] == bar[1]:
            continue
        n_chk += 1
        n_agree += int((bar[4] > bar[1]) == (r["out"] == "UP"))
    print(f"[对齐校验] 5min 市场结算 vs 5m K 线方向 一致率 {n_agree}/{n_chk}"
          f" = {n_agree / n_chk:.1%}" if n_chk else "[对齐校验] 无可比样本")

    pts = build_points(W, recs)
    for p in pts:
        sc = scenes.get(p["cyc"])
        # 本周期自身结算方向（= 持仓周期输赢；c_close/c_open 为该周期锚点）
        if sc and sc["c_open"] and sc["c_close"] != sc["c_open"]:
            p["cyc_dir"] = "DOWN" if sc["c_close"] < sc["c_open"] else "UP"
        else:
            p["cyc_dir"] = None
        # 目标 a：下一 5min 窗顺 Δ 方向（A1 原口径）
        if p["has_next"] and p["next_out"] in ("UP", "DOWN"):
            p["y_a"] = 1.0 if p["next_out"] == ("UP" if p["delta"] > 0 else "DOWN") else 0.0
        # 目标 b：本周期收 UP（危险信号的方向预测；对称信号用 1-y_b）
        if p["cyc_dir"] in ("UP", "DOWN"):
            p["y_b"] = 1.0 if p["cyc_dir"] == "UP" else 0.0

    dec = [p for p in pts if p["pos"] in (0, 1)]
    pool = [p for p in dec if abs(p["delta"]) >= 10]
    base_a = float(np.mean([p["y_a"] for p in pool if p.get("y_a") is not None]))
    base_b = float(np.mean([p["y_b"] for p in pool if p.get("y_b") is not None]))
    oos = [p for p in pool if p["start"] >= split_ts]
    base_a_oos = float(np.mean([p["y_a"] for p in oos if p.get("y_a") is not None]))
    base_b_oos = float(np.mean([p["y_b"] for p in oos if p.get("y_b") is not None]))
    print(f"[决策点] 全部 {len(dec)} | 对照池(|Δ|≥10) {len(pool)} "
          f"(OOS {len(oos)}) | 基准: 次窗顺Δ={base_a:.1%} 周期UP={base_b:.1%} "
          f"| OOS: 顺Δ={base_a_oos:.1%} 周期UP={base_b_oos:.1%}")

    # ============ L1：大样本通用检验 ============
    print(f"\n===== L1 通用检验：决策点背离信号（全周期，合并 IS+OOS）=====")
    l1: list[dict] = []
    for T in THRESHOLDS:
        for label, cond in (
            ("危险(DOWN持仓)", lambda p, T=T: p["delta"] >= T and p["out"] == "DOWN"),
            ("对称(UP持仓)", lambda p, T=T: p["delta"] <= -T and p["out"] == "UP"),
        ):
            sub = [p for p in pool if cond(p)]
            ra = stat_mask(sub, "y_a", base_a)
            rb = stat_mask(sub, "y_b", base_b)
            ro = stat_mask([p for p in sub if p["start"] >= split_ts], "y_b", base_b_oos)
            l1.append({"T": T, "sig": label, "n": len(sub),
                       "ya": ra, "yb": rb, "yb_oos": ro})
            print(f"  T≥{T:<3} {label:<14} n={len(sub):>4}")
            if ra:
                print(fmt(ra, base_a, f"    目标a 次窗顺Δ"))
            if rb:
                oo = (f" | OOS n={ro['n']} p={ro['p']:.1%}" if ro and ro["n"] else "")
                print(fmt(rb, base_b, f"    目标b 周期收UP{oo}"))

    # ============ L2：S1/S4 环境内明细 ============
    print(f"\n===== L2 S1/S4 持仓周期内的触发明细（情绪数据时段）=====")
    by_cyc: dict[int, list] = {}
    for p in pts:
        by_cyc.setdefault(p["cyc"], []).append(p)
    # 限定情绪数据时段（detect_scenes 覆盖全部 360 天 K 线，时段外无情绪窗）
    lo_cyc = (t0 - 900_000) // 900_000
    hi_cyc = t1 // 900_000
    env_rows: list[dict] = []
    for cyc in sorted(scenes):
        if not (lo_cyc <= cyc <= hi_cyc):
            continue
        sc = scenes[cyc]
        for tag in ("s1", "s4"):
            if not sc[tag] or sc["next"] is None:
                continue
            hold_cyc = cyc + 1  # 持仓周期 = 次周期
            wins = sorted(by_cyc.get(hold_cyc, []), key=lambda p: p["pos"])
            row = {"pattern": tag.upper(), "sig_cyc": cyc, "hold_cyc": hold_cyc,
                   "hold_dir": "DOWN" if tag == "s1" else "DOWN",  # S1/S4 均买 DOWN
                   "settle": sc["next"], "win": sc["next"] == "DOWN",
                   "windows": [{"pos": p["pos"], "delta": round(p["delta"], 1),
                                "out": p["out"],
                                "danger": [T for T in THRESHOLDS
                                           if p["delta"] >= T and p["out"] == "DOWN"]}
                               for p in wins],
                   "n_windows": len(wins)}
            env_rows.append(row)
            wins_s = " ".join(f"pos{p['pos']}:Δ{p['delta']:+.0f}/{p['out']}"
                              f"{'⚠' if any(p['delta'] >= T and p['out'] == 'DOWN'
                                           for T in THRESHOLDS) else ''}"
                              for p in wins) or "（无情绪窗）"
            print(f"  {tag.upper()} 周期{cyc} → 持仓{hold_cyc} 结算{sc['next']}"
                  f"{'✓赢' if row['win'] else '✗输'} | {wins_s}")
    n_s1 = sum(1 for r in env_rows if r["pattern"] == "S1")
    w_s1 = sum(1 for r in env_rows if r["pattern"] == "S1" and r["win"])
    n_s4 = sum(1 for r in env_rows if r["pattern"] == "S4")
    w_s4 = sum(1 for r in env_rows if r["pattern"] == "S4" and r["win"])
    trig = sum(1 for r in env_rows for wd in r["windows"]
               if wd["pos"] in (0, 1) and wd["danger"])
    print(f"\n  [环境统计] S1: {w_s1}/{n_s1} 赢 | S4: {w_s4}/{n_s4} 赢 | "
          f"持仓周期内危险信号触发 {trig} 次（T∈{THRESHOLDS} 任一）")

    report = {"meta": {"windows": len(W), "split_ts": split_ts,
                       "align_check": {"agree": n_agree, "total": n_chk},
                       "pool": {"n": len(pool), "oos": len(oos)},
                       "bases": {"ya": base_a, "yb": base_b,
                                 "ya_oos": base_a_oos, "yb_oos": base_b_oos}},
              "l1": l1, "l2_env": env_rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[已写入] {out_path}")
    return report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="S1×A1 执行层结合回放（方案 B）")
    ap.add_argument("--from-file", default="sentiment_windows.json")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    with open(args.from_file, encoding="utf-8") as f:
        W = json.load(f)
    run(W, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
