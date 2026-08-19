"""次次周期回测（2026-08-19）：信号后第二个 15 分钟周期的胜率与 EV。

事件定义完全复用 local_720d_validation.py（S1/S2/S4/S5 锁定口径），
目标从 j+1（次周期）扩展到 j+2（次次周期）：
- 次周期行 = 基准复现（应与 validation_720d_result.json 一致）
- 次次周期行 = 用户假设检验：形态余波是否延续到第二个周期
- S5 两个变体：入场延后（原确认+次次周期）/ 次次自身确认（第1根5m<开盘）
- EV 双口径：EV@0.51 理论价（b=0.9216）；EV@q̂ 用线上 26,098 条 15s 采样
  估计"次次周期开盘报价"（按前一周期动量分桶，捕捉市场记忆效应）
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

FEE = 0.02
ODDS = (1 - FEE) / 0.51 - 1.0          # 0.9216（理论价 0.51 口径，对齐 720d）
EPS = 0.0005
LOOKBACK = 48
DAYS = 720
CACHE = "output/klines_5m_cache_720d.json"
LOG = "output/next_next_cycle.log"


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


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


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)
    with open(CACHE, encoding="utf-8") as f:
        kl = json.load(f)
    start_ms = now_ms - DAYS * 86_400_000
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
          for k in kl if DAYS * 86_400_000 <= 0 or int(k[0]) >= start_ms]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5]); o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5]); l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5]); v5 = np.array([r[5] for r in c5])
    print(f"5m K {len(c5)} 根")

    # ---------- 聚合 15m（复用 720d 逻辑）----------
    cyc_ids = t5 // 900_000
    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(cyc_ids):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list, ks = [], {}
    for cyc, idxs in buckets.items():
        if len(idxs) != 3 or (cyc + 1) * 900_000 > now_ms - 1_800_000:  # 留足次次周期
            continue
        idxs.sort()
        cyc_list.append(cyc)
        ks[cyc] = (o5[idxs[0]], max(h5[i] for i in idxs), min(l5[i] for i in idxs),
                   cl5[idxs[-1]], float(sum(v5[i] for i in idxs)), idxs)
    cyc_list.sort()
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([ks[c][0] for c in cyc_list]); h15 = np.array([ks[c][1] for c in cyc_list])
    l15 = np.array([ks[c][2] for c in cyc_list]); c15 = np.array([ks[c][3] for c in cyc_list])
    v15 = np.array([ks[c][4] for c in cyc_list])
    print(f"15m K {N} 根")

    # ---------- 特征（对齐 720d）----------
    rng15 = np.where(h15 > l15, h15 - l15, np.nan)
    dir15 = np.sign(c15 - o15)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

    def roll_nanmean(x, w):
        from numpy.lib.stride_tricks import sliding_window_view
        out = np.full(len(x), np.nan)
        sw = sliding_window_view(x, w)
        with np.errstate(invalid="ignore"):
            out[w - 1:] = np.nanmean(sw, axis=1)
        return out

    vma_prev = np.concatenate([[np.nan], roll_nanmean(v15, 20)[:-1]])
    vratio = v15 / np.where(vma_prev > 0, vma_prev, np.nan)

    streak = np.ones(N)
    for i in range(1, N):
        if dir15[i] == dir15[i - 1] and dir15[i] != 0:
            streak[i] = streak[i - 1] + 1

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

    lvl_hi = np.full(len(c5), np.nan); lvl_lo = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    broke_lo5 = l5 < lvl_lo * (1 - EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool); broke_lo15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc][5]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]):
                if broke_hi5[i]:
                    broke_hi15[j] = True
                if broke_lo5[i]:
                    broke_lo15[j] = True

    # ---------- 次周期(j+1) 与 次次周期(j+2) 结果 ----------
    has_n1 = np.zeros(N, dtype=bool); n1_down = np.zeros(N, dtype=bool)
    d1 = np.full(N, np.nan)              # 次周期第 1 根 5m 收益
    has_n2 = np.zeros(N, dtype=bool); n2_down = np.zeros(N, dtype=bool)
    d1_n2 = np.full(N, np.nan)           # 次次周期第 1 根 5m 收益
    for j in range(N - 1):
        if cyc_arr[j + 1] != cyc_arr[j] + 1:
            continue
        i0 = ks[cyc_list[j + 1]][5][0]
        if o5[i0] > 0:
            d1[j] = cl5[i0] / o5[i0] - 1.0
        nd = dir15[j + 1]
        if nd != 0:
            has_n1[j] = True
            n1_down[j] = nd < 0
        if j + 2 < N and cyc_arr[j + 2] == cyc_arr[j] + 2:
            i02 = ks[cyc_list[j + 2]][5][0]
            if o5[i02] > 0:
                d1_n2[j] = cl5[i02] / o5[i02] - 1.0
            nd2 = dir15[j + 2]
            if nd2 != 0:
                has_n2[j] = True
                n2_down[j] = nd2 < 0

    # ---------- 谓词（锁定口径）----------
    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)
    s2 = broke_lo15 & (dir15 < 0) & (vratio >= 2.0)
    s4 = (dir15 > 0) & (streak >= 3) & (close_pos >= 0.85)
    s5 = s1 & (d1 < 0)
    s5_skip = s1 & ~(d1 < 0)
    s5_n2confirm = s1 & (d1_n2 < 0)      # 次次周期自身第 1 根 5m 回落

    # ---------- EV 报价基准：线上 15s 采样 → 次次周期开盘报价（按前周期动量分桶）----------
    print("\n===== 开盘报价基准（线上 15s 采样，按前一周期动量分桶）=====")
    samples = json.load(open("output/online_15m_samples_full.json", encoding="utf-8"))
    by_cyc: dict[int, list] = {}
    for s in samples:
        by_cyc.setdefault(int(s["timestamp"]) // 900_000, []).append(s)
    q_stat = {}                          # bucket -> (mean_open_down, mean_open_up, n)
    cyc_open = {}                        # cyc -> (open_down, open_up, ret_prev_bp)
    for cyc in sorted(by_cyc):
        pts = sorted(by_cyc[cyc], key=lambda x: x["timestamp"])
        head = pts[:2]
        od = float(np.mean([p["down_price"] for p in head]))
        ou = float(np.mean([p["up_price"] for p in head]))
        ret_prev = np.nan
        if cyc - 1 in by_cyc:
            prev = sorted(by_cyc[cyc - 1], key=lambda x: x["timestamp"])
            if prev[0]["btc_price"] and prev[-1]["btc_price"]:
                ret_prev = (prev[-1]["btc_price"] / prev[0]["btc_price"] - 1) * 10000
        cyc_open[cyc] = (od, ou, ret_prev)
    for name, cond in (("前周期大跌<-10bp", lambda r: r is not np.nan and r < -10),
                       ("前周期平盘±10bp", lambda r: r is not np.nan and abs(r) <= 10),
                       ("前周期大涨>+10bp", lambda r: r is not np.nan and r > 10),
                       ("无条件", lambda r: True)):
        vals = [(od, ou) for od, ou, r in cyc_open.values() if cond(r)]
        if vals:
            q_stat[name] = (float(np.mean([v[0] for v in vals])),
                            float(np.mean([v[1] for v in vals])), len(vals))
            print(f"  {name}: n={q_stat[name][2]:>3} 周期  开盘DOWN价 {q_stat[name][0]:.3f}  开盘UP价 {q_stat[name][1]:.3f}")

    Q_DOWN_PREVDOWN = q_stat.get("前周期大跌<-10bp", (0.5, 0.5, 0))[0]
    Q_UP_PREVUP = q_stat.get("前周期大涨>+10bp", (0.5, 0.5, 0))[1]
    Q_UNCOND = q_stat.get("无条件", (0.5, 0.5, 0))[0]

    # ---------- 主表 ----------
    MODES = [
        ("S1 多头耗尽", s1, "down", Q_DOWN_PREVDOWN),
        ("S2 空头耗尽", s2, "up", Q_UP_PREVUP),
        ("S4 动量衰竭", s4, "down", Q_UNCOND),
        ("S5 确认入场(原确认)", s5, "down", Q_DOWN_PREVDOWN),
        ("S5' 次次自身确认", s5_n2confirm, "down", Q_UNCOND),
        ("S5'' skip后第二机会", s5_skip & (d1_n2 < 0), "down", Q_UNCOND),
        ("S5''&双确认", s5 & (d1_n2 < 0), "down", Q_UNCOND),
        ("S5-skip 放弃组(对照)", s5_skip, "down", Q_DOWN_PREVDOWN),
    ]
    report = {"q_basis": {k: v for k, v in q_stat.items()}, "rows": {}}
    print(f"\n===== 主表：次周期(j+1) vs 次次周期(j+2)，720d =====")
    print(f"{'模式':<24}{'周期':<8}{'n':>6}{'胜率':>8}{'Wilson95':>16}"
          f"{'EV@0.51':>9}{'EV@q̂':>8}")
    for name, mask, expect, q in MODES:
        for tag, has, down_arr in (("次周期", has_n1, n1_down), ("次次周期", has_n2, n2_down)):
            m = mask & has
            n = int(m.sum())
            if n < 5:
                continue
            k_d = int(down_arr[m].sum())
            p = (k_d / n) if expect == "down" else (1 - k_d / n)
            lo, hi = wilson(round(p * n), n)
            ev51 = p * (1 + ODDS) - 1.0
            evq = p * (0.98 / q - 1) - (1 - p)
            print(f"{name:<24}{tag:<8}{n:>6}{p:>8.1%}  [{lo:.1%},{hi:.1%}]"
                  f"{ev51:>+9.3f}{evq:>+8.3f}")
            report["rows"][f"{name}|{tag}"] = {
                "n": n, "p": p, "wilson": [lo, hi], "ev51": ev51, "evq": evq, "q": q}

    # ---------- 次周期结果 × 次次周期（条件延续性）----------
    print("\n===== 条件分析：次周期输/赢后，次次周期表现（S1 族）=====")
    for name, mask in (("S1 全体", s1), ("S5 确认", s5)):
        for lab, cond in (("次周期赢(回落)", n1_down), ("次周期输(延续涨)", ~n1_down)):
            m = mask & has_n1 & has_n2 & cond
            n = int(m.sum())
            if n < 5:
                continue
            p = float(n2_down[m].mean())
            print(f"  {name} & {lab}: n={n:>4}  次次周期P(DOWN)={p:.1%}")

    with open("output/next_next_cycle_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("\n结果已存 output/next_next_cycle_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
