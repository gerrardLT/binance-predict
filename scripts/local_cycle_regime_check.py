#!/usr/bin/env python3
"""补充分析：行情周期维度 —— 8 个 OOS 场景 × 周期状态交互检验。

周期状态定义（无未来函数，仅用截至当前根的数据）：
  ER_7d = |c15[i] - c15[i-672]| / sum(|Δc15|, 过去672根)   （7天效率比）
  阈值取发现集 ER 分布的分位数（防止验证集信息泄漏进阈值）：
    趋势牛 = ER≥q75 且 7d 净位移>0
    趋势熊 = ER≥q75 且 7d 净位移<0
    震荡   = ER<q50
    过渡   = 其余（q50≤ER<q75）

分析内容：
  A. 周期状态按月占比（验证集各月处于什么周期态）
  B. 各周期态下的市场微观结构基准（次根阴率 / 延续率）
  C. 8 个 OOS 场景 × 周期态胜率矩阵（验证集内 + 全期）
  D. F09×F11 月度衰减 vs 当月趋势态占比对照

纪律声明：本分析是对已有 OOS 结论的【事后细分】（exploratory），
不用于筛选新场景，只用于理解既有信号的周期依赖。
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48
CACHE = "output/klines_5m_cache.json"
LOG = "output/cycle_regime_check.log"


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


def roll_sum(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)
    with open(CACHE, encoding="utf-8") as f:
        kl = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5])
    l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])

    # ---------- 聚合 15m ----------
    cyc_ids = t5 // 900_000
    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(cyc_ids):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_list, ks = [], {}
    for cyc, idxs in buckets.items():
        if len(idxs) != 3 or (cyc + 1) * 900_000 > now_ms:
            continue
        idxs.sort()
        cyc_list.append(cyc)
        ks[cyc] = (o5[idxs[0]], max(h5[i] for i in idxs), min(l5[i] for i in idxs),
                   cl5[idxs[-1]], idxs)
    cyc_list.sort()
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([ks[c][0] for c in cyc_list])
    h15 = np.array([ks[c][1] for c in cyc_list])
    l15 = np.array([ks[c][2] for c in cyc_list])
    c15 = np.array([ks[c][3] for c in cyc_list])
    print(f"15m K {N} 根（{time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[-1] * 900))}）")

    # ---------- 特征 ----------
    dir15 = np.sign(c15 - o15)
    body_frac = np.abs(c15 - o15) / np.where(h15 > l15, h15 - l15, np.nan)
    upper_frac = (h15 - np.maximum(o15, c15)) / np.where(h15 > l15, h15 - l15, np.nan)
    lower_frac = (np.minimum(o15, c15) - l15) / np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)
    prev_o = np.concatenate([[np.nan], o15[:-1]])
    prev_c = np.concatenate([[np.nan], c15[:-1]])
    bull_engulf = (dir15 > 0) & (o15 <= np.minimum(prev_o, prev_c)) & (c15 > np.maximum(prev_o, prev_c))
    marubozu_bull = (dir15 > 0) & (upper_frac + lower_frac <= 0.15) & (body_frac >= 0.7)

    streak = np.ones(N)
    for i in range(1, N):
        if dir15[i] == dir15[i - 1] and dir15[i] != 0:
            streak[i] = streak[i - 1] + 1

    lvl_hi = np.full(len(c5), np.nan)
    lvl_lo = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    broke_lo15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc][4]:
            if cont[i] and i >= LOOKBACK:
                if h5[i] > lvl_hi[i] * (1 + EPS):
                    broke_hi15[j] = True
                if l5[i] < lvl_lo[i] * (1 - EPS):
                    broke_lo15[j] = True
    pm = roll_max(c15, 96)
    pmi = roll_min(c15, 96)
    prev_max96 = np.full(N, np.nan)
    prev_min96 = np.full(N, np.nan)
    prev_max96[1:] = pm[:-1]
    prev_min96[1:] = pmi[:-1]

    # ---------- 周期状态（无未来函数）----------
    W = 672  # 7天
    disp = np.full(N, np.nan)
    disp[W:] = np.abs(c15[W:] - c15[:-W])
    pathl = roll_sum(np.abs(np.diff(c15, prepend=c15[0])), W)
    er7 = disp / pathl
    net7 = np.full(N, np.nan)
    net7[W:] = c15[W:] - c15[:-W]

    split = int(N * 2 / 3)
    q75, q50 = np.nanquantile(er7[:split], [0.75, 0.5])
    state = np.full(N, "过渡", dtype=object)
    state[er7 >= q75] = np.where(net7[er7 >= q75] > 0, "趋势牛", "趋势熊")
    state[er7 < q50] = "震荡"
    state[np.isnan(er7)] = "未知"

    # ---------- 目标 ----------
    nxt_down = np.zeros(N, dtype=bool)
    has_next = np.zeros(N, dtype=bool)
    nxt_same = np.zeros(N, dtype=bool)
    same_valid = np.zeros(N, dtype=bool)
    for j in range(N - 1):
        if cyc_arr[j + 1] == cyc_arr[j] + 1:
            nd = dir15[j + 1]
            has_next[j] = nd != 0
            nxt_down[j] = nd < 0
            if nd != 0 and dir15[j] != 0:
                same_valid[j] = True
                nxt_same[j] = nd == dir15[j]

    # ---------- 8 个 OOS 场景掩码 ----------
    SC = {
        "S1 F22 破4h高+光头阳→DOWN": (broke_hi15 & (dir15 > 0) & (close_pos >= 0.85), "down"),
        "S2 F22×24h新高→DOWN": (broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (c15 > prev_max96), "down"),
        "S3 连阳3+光头阳→DOWN": ((dir15 > 0) & (streak >= 3) & (close_pos >= 0.85), "down"),
        "S4 连阳3+大实体→DOWN": ((dir15 > 0) & (streak >= 3) & (body_frac >= 0.7), "down"),
        "S5 光脚阳+吞没→DOWN": (marubozu_bull & bull_engulf, "down"),
        "S6 破4h低+收阴→UP": (broke_lo15 & (dir15 < 0), "up"),
        "S7 24h新低→UP": (c15 < prev_min96, "up"),
        "S8 光脚阳+连阳3+大实体→DOWN": (marubozu_bull & (streak >= 3) & (body_frac >= 0.7), "down"),
    }

    print(f"\nER_7d 阈值：q50={q50:.3f} q75={q75:.3f}（发现集分位）")
    print(f"发现集状态占比：", {s: f"{(state[:split] == s).mean():.1%}" for s in ["趋势牛", "趋势熊", "震荡", "过渡"]})
    print(f"验证集状态占比：", {s: f"{(state[split:] == s).mean():.1%}" for s in ["趋势牛", "趋势熊", "震荡", "过渡"]})

    # ---------- A. 按月状态占比 ----------
    print("\n===== A. 周期状态按月占比（2026 验证期重点）=====")
    months = sorted({time.strftime("%Y-%m", time.gmtime(c * 900)) for c in cyc_arr})
    for mth in months:
        mm = np.array([time.strftime("%Y-%m", time.gmtime(c * 900)) == mth for c in cyc_arr])
        parts = []
        for s in ["趋势牛", "趋势熊", "震荡", "过渡"]:
            k = int((mm & (state == s)).sum())
            if k:
                parts.append(f"{s}{k / mm.sum():.0%}")
        tag = " ←验证集" if mm[split:].any() else ""
        print(f"  {mth}{tag}: " + " ".join(parts))

    # ---------- B. 周期态下的微观结构基准 ----------
    print("\n===== B. 各周期态的市场微观结构基准（发现集 / 验证集）=====")
    print(f"  {'状态':<6}{'发现集n':>9}{'次根↓':>8}{'延续率':>8} | {'验证集n':>8}{'次根↓':>8}{'延续率':>8}")
    for s in ["趋势牛", "趋势熊", "震荡", "过渡"]:
        row = []
        for lo, hi in ((0, split), (split, N)):
            m = (slice(lo, hi),)
            sel = (state[lo:hi] == s) & has_next[lo:hi]
            sel_s = (state[lo:hi] == s) & same_valid[lo:hi]
            n1, n2 = int(sel.sum()), int(sel_s.sum())
            pd_ = nxt_down[lo:hi][sel].mean() if n1 else np.nan
            ps_ = nxt_same[lo:hi][sel_s].mean() if n2 else np.nan
            row.append((n1, pd_, ps_))
        print(f"  {s:<6}{row[0][0]:>9}{row[0][1]:>8.1%}{row[0][2]:>8.1%} | "
              f"{row[1][0]:>8}{row[1][1]:>8.1%}{row[1][2]:>8.1%}")

    # ---------- C. 场景 × 周期态 胜率矩阵 ----------
    def win(mask, expect, sel):
        """sel: 布尔选择器（状态∩区间）。返回 (n, 胜率)。"""
        m = mask & sel
        n = int(m.sum())
        if n < 30:
            return n, np.nan
        p = nxt_down[m].mean() if expect == "down" else 1 - nxt_down[m].mean()
        return n, p

    print("\n===== C. 8 个 OOS 场景 × 周期态（验证集 120 天 / 全期 360 天）=====")
    states = ["趋势牛", "趋势熊", "震荡", "过渡"]
    for name, (mask, expect) in SC.items():
        cells = []
        for s in states:
            sel_v = has_next.copy()
            sel_v[:split] = False                      # 仅验证集
            n_v, p_v = win(mask, expect, sel_v & (state == s))
            sel_f = has_next.copy()
            sel_f[split:] = False                      # 仅发现集
            n_f, p_f = win(mask, expect, sel_f & (state == s))
            pv = f"{p_v:.0%}({n_v})" if p_v == p_v else f"-({n_v})"
            pf = f"{p_f:.0%}({n_f})" if p_f == p_f else f"-({n_f})"
            cells.append(f"{s}: 验{pv}/发{pf}")
        # 验证集整体
        sel_v = has_next.copy()
        sel_v[:split] = False
        n_all, p_all = win(mask, expect, sel_v)
        print(f"  {name}")
        print(f"    验证集整体 {p_all:.1%}(n={n_all}) | " + " | ".join(cells))

    # ---------- D. F09×F11 月度衰减 vs 趋势态 ----------
    print("\n===== D. S5（光脚阳+吞没）月度胜率 vs 当月趋势态占比 =====")
    mask5 = SC["S5 光脚阳+吞没→DOWN"][0]
    for mth in months:
        mm = np.array([time.strftime("%Y-%m", time.gmtime(c * 900)) == mth for c in cyc_arr])
        m = mask5 & has_next & mm
        trend_share = (mm & ((state == "趋势牛") | (state == "趋势熊"))).sum() / mm.sum()
        if m.sum() >= 10:
            print(f"  {mth}: S5↓ {nxt_down[m].mean():.0%}(n={m.sum()}) | 当月趋势态占比 {trend_share:.0%}")
        else:
            print(f"  {mth}: S5 样本不足(n={m.sum()}) | 趋势态占比 {trend_share:.0%}")

    # ---------- E. 趋势态下的信号选择性增强（探索性）----------
    print("\n===== E. 探索：王牌 S1 在「震荡态」验证集子集的表现 =====")
    mask1 = SC["S1 F22 破4h高+光头阳→DOWN"][0]
    sel_v = has_next.copy()
    sel_v[:split] = False
    for s in ["震荡", "趋势牛", "趋势熊", "过渡"]:
        n, p = win(mask1, "down", sel_v & (state == s))
        ps = f"{p:.1%}" if p == p else "-"
        print(f"  S1 × {s}: {ps} (n={n})")

    print("\n（分析为事后细分，不用于筛选，仅用于理解周期依赖）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
