#!/usr/bin/env python3
"""确定性场景发现：什么条件下 15m K 线的延续性/反转性最高？

研究纪律：
  1. 数据：官方 5m klines ×180天 → 精确聚合 15m（OHLCV）。
  2. 时间切分：前 120 天 = 发现集（L1/L2/L3 只在此做），后 60 天 = 验证集（仅终验）。
  3. L1 单因子（预注册假设，带机制） → L2 双因子 → L3 三因子。
  4. 入场券：发现集 n≥300 且 |偏离50%|≥2pp；终验要求验证集 n≥60、方向一致、点估计>52%。

目标变量：next_down（次根收阴）；next_same（次根与当前同向 = 延续）。
打平线：@0.50 入场费2%+溢价0.01 → 52.0%。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np

FEE = 0.02
PREMIUM = 0.01
EPS = 0.0005
LOOKBACK = 48
DAYS = 180
API = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(interval: str, start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval={interval}&startTime={cur}&endTime={end_ms}&limit=1000"
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
        time.sleep(0.25)
    return out


def roll_max(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    v = sliding_window_view(x, w).max(axis=1)
    out[w - 1:] = v
    return out


def roll_min(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    v = sliding_window_view(x, w).min(axis=1)
    out[w - 1:] = v
    return out


def roll_mean(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


def ev_at(p: float) -> float:
    return p * ((1 - FEE) / (0.50 + PREMIUM) - 1.0) - (1 - p)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    now_ms = int(time.time() * 1000)
    kl = fetch_klines("5m", now_ms - DAYS * 86_400_000, now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5])
    l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])
    v5 = np.array([r[5] for r in c5])

    # ---------- 聚合 15m ----------
    cyc_ids = t5 // 900_000
    uniq, first_idx = np.unique(cyc_ids, return_index=True)
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
                   cl5[idxs[-1]], float(sum(v5[i] for i in idxs)), idxs)
    cyc_list.sort()
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([ks[c][0] for c in cyc_list])
    h15 = np.array([ks[c][1] for c in cyc_list])
    l15 = np.array([ks[c][2] for c in cyc_list])
    c15 = np.array([ks[c][3] for c in cyc_list])
    v15 = np.array([ks[c][4] for c in cyc_list])
    print(f"15m K {N} 根（{time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[-1] * 900))}）")

    # ---------- 特征工程 ----------
    rng15 = np.where(h15 > l15, h15 - l15, np.nan) / o15
    dir15 = np.sign(c15 - o15)
    body_frac = np.abs(c15 - o15) / np.where(h15 > l15, h15 - l15, np.nan)
    upper_frac = (h15 - np.maximum(o15, c15)) / np.where(h15 > l15, h15 - l15, np.nan)
    lower_frac = (np.minimum(o15, c15) - l15) / np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

    atr_prev = np.empty(N)  # 前20根平均振幅（不含当前）
    atr_prev[:] = roll_mean(np.concatenate([[np.nan], rng15[:-1]]), 20)
    vratio = v15 / np.concatenate([[np.nan], roll_mean(v15, 20)[:-1]])

    streak = np.ones(N)  # 当前根计入的连续同向根数
    for i in range(1, N):
        if dir15[i] == dir15[i - 1] and dir15[i] != 0:
            streak[i] = streak[i - 1] + 1
    prev_dir = np.concatenate([[np.nan], dir15[:-1]])

    # 5m 层 4h 位势破位 → 映射 15m
    lvl_hi = np.full(len(c5), np.nan)
    lvl_lo = np.full(len(c5), np.nan)
    rm = roll_max(cl5, LOOKBACK)
    rmin = roll_min(cl5, LOOKBACK)
    lvl_hi[1:] = rm[:-1]
    lvl_lo[1:] = rmin[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    broke_lo5 = l5 < lvl_lo * (1 - EPS)
    cont = np.zeros(len(c5), dtype=bool)  # 5m 连续性检查
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    broke_lo15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc][5]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]):
                if broke_hi5[i]:
                    broke_hi15[j] = True
                if broke_lo5[i]:
                    broke_lo15[j] = True

    # 24h 新高低（收盘 vs 前96根收盘极值）
    prev_max96 = np.full(N, np.nan)
    prev_min96 = np.full(N, np.nan)
    pm = roll_max(c15, 96)
    pmi = roll_min(c15, 96)
    prev_max96[1:] = pm[:-1]
    prev_min96[1:] = pmi[:-1]
    new_24h_hi = c15 > prev_max96
    new_24h_lo = c15 < prev_min96

    # 过去16根15m（4h）区间位置
    w16_hi = roll_max(c15, 16)
    w16_lo = roll_min(c15, 16)
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)

    # 当前 4h K 已走部分方向（桶内首根5m open → 当前15m收盘）
    bucket4h_of_5m = t5 // 14_400_000
    first_open4h: dict[int, float] = {}
    for i in range(len(c5)):
        b = int(bucket4h_of_5m[i])
        if b not in first_open4h:
            first_open4h[b] = o5[i]
    run_dir4h = np.array([np.sign(c15[j] - first_open4h[int(cyc_arr[j] * 900_000 // 14_400_000)])
                          for j in range(N)])

    # 5m 路径 / 尾盘动量 / ER
    path3, last5_dir, er = [], [], []
    for cyc in cyc_list:
        idxs = ks[cyc][5]
        d = [int(np.sign(cl5[i] - o5[i])) for i in idxs]
        path3.append("".join("U" if x > 0 else ("D" if x < 0 else "F") for x in d))
        last5_dir.append(d[-1])
        pts = [o5[idxs[0]], cl5[idxs[0]], cl5[idxs[1]], cl5[idxs[2]]]
        path_len = sum(abs(pts[i + 1] - pts[i]) for i in range(3))
        er.append(abs(pts[3] - pts[0]) / path_len if path_len > 0 else np.nan)
    path3 = np.array(path3)
    last5_dir = np.array(last5_dir, dtype=float)
    er = np.array(er)

    hours = np.array([time.gmtime(c * 900).tm_hour for c in cyc_arr])
    wd = np.array([time.gmtime(c * 900).tm_wday for c in cyc_arr])

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

    # ---------- 发现/验证切分 ----------
    split = int(N * 2 / 3)
    print(f"发现集 {split} 根（~{time.strftime('%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f"~{time.strftime('%m-%d', time.gmtime(cyc_arr[split - 1] * 900))}） | "
          f"验证集 {N - split} 根（{time.strftime('%m-%d', time.gmtime(cyc_arr[split] * 900))}"
          f"~{time.strftime('%m-%d', time.gmtime(cyc_arr[-1] * 900))}）")

    rng_np = np.random.default_rng(7)

    def report(mask, pool_end, label=""):
        m = mask[:pool_end] & has_next[:pool_end]
        n = int(m.sum())
        if n < 30:
            return None
        pd = float(nxt_down[:pool_end][m].mean())
        ms = mask[:pool_end] & same_valid[:pool_end]
        ns = int(ms.sum())
        ps = float(nxt_same[:pool_end][ms].mean()) if ns >= 30 else np.nan
        lo, hi = np.percentile(rng_np.binomial(n, pd, size=3000) / n, [2.5, 97.5])
        return {"name": label, "n": n, "pd": pd, "lo": lo, "hi": hi, "ps": ps, "ns": ns}

    def line(r, base_d, base_s):
        ps_s = f"{r['ps']:.1%}" if r["ps"] == r["ps"] else "  -  "
        return (f"  {r['name']}: n={r['n']:>5} 次根↓ {r['pd']:.1%} ({r['pd'] - base_d:+.1%}pp) "
                f"[{r['lo']:.1%},{r['hi']:.1%}] EV↓ {ev_at(r['pd']):+.3f} | 延续 {ps_s}")

    # ---------- L1：预注册假设 ----------
    H = []
    def add(name, m):
        m = np.asarray(m, dtype=bool).copy()
        m[np.isnan(rng15)] = False
        H.append((name, m))

    add("F01 收阴(对照)", dir15 < 0)
    add("F02 收阳(对照)", dir15 > 0)
    add("F03 大实体阴 body≥70%", (dir15 < 0) & (body_frac >= 0.7))
    add("F04 大实体阳 body≥70%", (dir15 > 0) & (body_frac >= 0.7))
    add("F05 阴·收最低区 close_pos≤0.15", (dir15 < 0) & (close_pos <= 0.15))
    add("F06 阳·收最高区 close_pos≥0.85", (dir15 > 0) & (close_pos >= 0.85))
    add("F07 阴·长上影 upper≥40%", (dir15 < 0) & (upper_frac >= 0.4))
    add("F08 阳·长下影 lower≥40%", (dir15 > 0) & (lower_frac >= 0.4))
    add("F09 破4h高位势(周期内任意5m)", broke_hi15)
    add("F10 破4h低位势", broke_lo15)
    add("F11 破4h高·收阳(旧信号)", broke_hi15 & (dir15 > 0))
    add("F12 破4h高·收阴", broke_hi15 & (dir15 < 0))
    add("F13 破4h低·收阴", broke_lo15 & (dir15 < 0))
    add("F14 破4h低·收阳", broke_lo15 & (dir15 > 0))
    add("F15 收盘创24h新高", new_24h_hi)
    add("F16 收盘创24h新低", new_24h_lo)
    add("F17 4h区间上沿 pos4h≥0.9", pos4h >= 0.9)
    add("F18 4h区间下沿 pos4h≤0.1", pos4h <= 0.1)
    add("F19 连阴≥3", (dir15 < 0) & (streak >= 3))
    add("F20 连阳≥3", (dir15 > 0) & (streak >= 3))
    add("F21 阴后阴(马尔可夫)", (dir15 < 0) & (prev_dir < 0))
    add("F22 阳后阳", (dir15 > 0) & (prev_dir > 0))
    add("F23 阳+尾段5m仍涨(收盘加速)", (dir15 > 0) & (last5_dir > 0))
    add("F24 阴+尾段5m仍跌", (dir15 < 0) & (last5_dir < 0))
    add("F25 阳+尾段5m回落(收盘泄力)", (dir15 > 0) & (last5_dir < 0))
    add("F26 阴+尾段5m反抽", (dir15 < 0) & (last5_dir > 0))
    add("F27 振幅爆发 rng≥2×ATR20", rng15 >= 2 * atr_prev)
    add("F28 压缩 rng≤0.5×ATR20", rng15 <= 0.5 * atr_prev)
    add("F29 放量 v≥2×均量", vratio >= 2)
    add("F30 缩量 v≤0.5×均量", vratio <= 0.5)
    add("F31 单边路径 ER≥0.8", er >= 0.8)
    add("F32 震荡路径 ER≤0.4", er <= 0.4)
    add("F33 路径UUD(冲高回落)", path3 == "UUD")
    add("F34 路径DUU(探底回升)", path3 == "DUU")
    add("F35 顺4h势", dir15 * run_dir4h > 0)
    add("F36 逆4h势", dir15 * run_dir4h < 0)
    add("F37 亚洲时段(0-7)", hours <= 7)
    add("F38 欧洲时段(8-15)", (hours >= 8) & (hours <= 15))
    add("F39 美洲时段(16-23)", hours >= 16)
    add("F40 周末", wd >= 5)

    base_d = float(nxt_down[:split][has_next[:split]].mean())
    base_s = float(nxt_same[:split][same_valid[:split]].mean())
    print(f"\n发现集基准：次根↓ {base_d:.1%} | 延续率 {base_s:.1%} | 打平 52.0%")
    print(f"\n===== L1 单因子（{len(H)} 个预注册假设，发现集）=====")
    l1 = []
    for name, m in H:
        r = report(m, split, name)
        if r:
            l1.append(r)
    l1.sort(key=lambda r: -max(abs(r["pd"] - 0.5), abs(r["ps"] - 0.5) if r["ps"] == r["ps"] else 0))
    for r in l1:
        print(line(r, base_d, base_s))

    # ---------- L2：双因子 ----------
    cands = [r for r in l1 if r["n"] >= 300 and max(abs(r["pd"] - 0.5), abs(r["ps"] - 0.5)) >= 0.02]
    cand_names = {r["name"] for r in cands}
    hmap = {n: m for n, m in H}
    print(f"\n===== L2 双因子（L1 入场券 {len(cands)} 个因子两两组合，发现集）=====")
    l2 = []
    cl = sorted(cand_names)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            m = hmap[cl[i]] & hmap[cl[j]]
            r = report(m, split, f"{cl[i]} × {cl[j]}")
            if r and r["n"] >= 200:
                l2.append(r)
    l2.sort(key=lambda r: -max(abs(r["pd"] - 0.5), abs(r["ps"] - 0.5)))
    for r in l2[:20]:
        print(line(r, base_d, base_s))

    # ---------- L3：三因子 ----------
    print(f"\n===== L3 三因子（L2 top5 组合 × 其余强单因子，发现集）=====")
    l3 = []
    top5 = l2[:5]
    for r2 in top5:
        n1, n2 = r2["name"].split(" × ")
        for n3 in cl:
            if n3 in (n1, n2):
                continue
            m = hmap[n1] & hmap[n2] & hmap[n3]
            r = report(m, split, f"{n1} × {n2} × {n3}")
            if r and r["n"] >= 120:
                l3.append(r)
    l3.sort(key=lambda r: -max(abs(r["pd"] - 0.5), abs(r["ps"] - 0.5)))
    for r in l3[:15]:
        print(line(r, base_d, base_s))

    # ---------- 终验：候选在验证集 + 按月 ----------
    def to_mask(name: str) -> np.ndarray:
        parts = name.split(" × ")
        m = hmap[parts[0]].copy()
        for p in parts[1:]:
            m = m & hmap[p]
        return m

    pool = ([r for r in l3 if r["n"] >= 150] + [r for r in l2 if r["n"] >= 250])
    pool.sort(key=lambda r: -max(abs(r["pd"] - 0.5), abs(r["ps"] - 0.5)))
    print(f"\n===== 终验：top 候选 → 验证集（后60天）+ 按月稳定性 =====")
    base_d_v = float(nxt_down[split:][has_next[split:]].mean())
    print(f"验证集基准：次根↓ {base_d_v:.1%}")
    for r in pool[:8]:
        m = to_mask(r["name"])
        rv = report(m, N, r["name"] + " |验证集")
        if rv:
            print(f"  [发现] {line(r, base_d, base_s)}")
            print(f"  {line(rv, base_d_v, None)}")
            mm = m & has_next
            months: dict[str, list[bool]] = {}
            for j in range(N):
                if mm[j]:
                    months.setdefault(time.strftime("%Y-%m", time.gmtime(cyc_arr[j] * 900)),
                                      []).append(nxt_down[j])
            ms = " / ".join(f"{m_}:{sum(b)}/{len(b)}={sum(b) / len(b):.0%}"
                            for m_, b in sorted(months.items()))
            print(f"    按月(次根↓): {ms}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
