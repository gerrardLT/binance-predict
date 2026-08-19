#!/usr/bin/env python3
"""S6 逐分钟入场回测（S1 形态前提）：次周期内每分钟 t=1..14 回落确认即入场的 EV 曲线。

设计（2026-08-18，探索性分析，非预注册）：
  1. S1 事件识别：720d 5m 缓存，锁定口径同 local_720d_validation.py（broke_hi15 &
     dir15>0 & close_pos>=0.85 & pos4h>=0.9）。
  2. 次周期 1m 路径：仅拉 S1 事件次周期窗口的 1m K（15 根/事件，~2300 窗口，
     约为全量 1m 的 3%），缓存 output/klines_1m_s1_windows.json 断点续拉。
  3. 报价模型：线上 5 天 15s 采样（output/online_15m_samples_full.json）构建
     q̂(分钟桶, 回落深度档) 表——普通回落的定价（形态信息缺口假设：S1 形态
     不改变市场报价方式，市场看不见形态）。
  4. 主表：EV(t) = mean over {S1 事件: 价(t)<开盘} 的 win*0.98/q̂ − (1−win)，
     t=1..14（分钟末 1m close 判状态，报价取该分钟桶 × 深度档）。
  5. 对照与变体：t=5 行应复现 S5（5m 口径 78.5%）；S6-min = 首次回落 t* 即入场；
     A/B 段各 360 天分段；q̂ ±0.02 敏感性。
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request

import numpy as np

FEE = 0.02
EPS = 0.0005
LOOKBACK = 48
DAYS = 720
API = "https://data-api.binance.vision/api/v3/klines"
KL5 = "output/klines_5m_cache_720d.json"
SAMPLES = "output/online_15m_samples_full.json"
W1M = "output/klines_1m_s1_windows.json"
LOG = "output/s6_s1_minute_backtest.log"
DEPTH_EDGES = (0.0005, 0.0015)   # 回落深度档：浅 / 中 / 深（−ln(P/open)）


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


def dbucket(depth: float) -> int:
    if depth < DEPTH_EDGES[0]:
        return 0
    if depth < DEPTH_EDGES[1]:
        return 1
    return 2


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)

    # ---------- 1) 5m 缓存 → 15m 聚合 → S1 识别（口径 = 720d 验证锁定）----------
    with open(KL5, encoding="utf-8") as f:
        kl = json.load(f)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5]); o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5]); l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5]); v5 = np.array([r[5] for r in c5])

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
    cyc_arr = np.array(cyc_list)
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

    lvl_hi = np.full(len(c5), np.nan); lvl_lo = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    cont = np.zeros(len(c5), dtype=bool)
    cont[1:] = (t5[1:] - t5[:-1]) == 300_000
    broke_hi15 = np.zeros(N, dtype=bool)
    for j, cyc in enumerate(cyc_list):
        for i in ks[cyc]:
            if cont[i] and i >= LOOKBACK and not np.isnan(lvl_hi[i]) and broke_hi5[i]:
                broke_hi15[j] = True

    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)
    pos_in = {c: j for j, c in enumerate(cyc_list)}

    # 事件：S1 信号 j → 次周期 cyc+1 完整存在 → (次周期ms, open, win, A/B 段)
    a_end = now_ms - 360 * 86_400_000
    events = []
    n_s1 = n_nonext = 0
    for j in range(N - 1):
        if not s1[j]:
            continue
        n_s1 += 1
        cyc_n = cyc_list[j] + 1
        if cyc_list[j + 1] != cyc_n:
            n_nonext += 1
            continue
        o_n = o15[j + 1]
        if o_n <= 0:
            continue
        events.append({
            "start": cyc_n * 900_000, "open": float(o_n), "win": bool(c15[j + 1] < o_n),
            "seg": "A" if (cyc_n * 900_000) < a_end else "B",
        })
    print(f"S1 事件 {n_s1} 个（缺次周期 {n_nonext}）→ 有效 {len(events)}"
          f"（A 段 {sum(1 for e in events if e['seg'] == 'A')} / B 段 {sum(1 for e in events if e['seg'] == 'B')}）")

    # ---------- 2) 次周期 1m 窗口拉取（增量缓存）----------
    try:
        with open(W1M, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
    todo = [e for e in events if str(e["start"]) not in cache or len(cache[str(e["start"])]) < 15]
    print(f"1m 窗口缓存 {len(cache)}，待拉 {len(todo)}")
    for idx, e in enumerate(todo, 1):
        s_ms = e["start"]
        url = f"{API}?symbol=BTCUSDT&interval=1m&startTime={s_ms}&endTime={s_ms + 900_000}&limit=15"
        batch = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    batch = json.loads(resp.read().decode())
                break
            except Exception as ex:
                if attempt == 3:
                    print(f"  窗口 {s_ms} 拉取失败: {ex}")
                else:
                    time.sleep(1.5)
        cache[str(s_ms)] = batch or []
        if idx % 200 == 0:
            with open(W1M, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            print(f"  1m 已拉 {idx}/{len(todo)}")
        time.sleep(0.12)
    with open(W1M, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    print(f"1m 窗口缓存完成：{len(cache)}")

    # 事件路径矩阵：closes[t] t=1..15（分钟末价格）
    paths = []
    n_badwin = 0
    for e in events:
        arr = cache.get(str(e["start"]), [])
        rows = [r for r in arr if int(r[0]) < e["start"] + 900_000]
        if len(rows) != 15:
            n_badwin += 1
            continue
        closes = [float(r[4]) for r in rows]
        # 结算一致性：1m 最后 close vs 15m close（5m 聚合）
        if (closes[-1] < e["open"]) != e["win"]:
            n_badwin += 1
            continue
        paths.append({**e, "closes": closes})
    print(f"有效路径 {len(paths)}（剔除不完整/结算不一致 {n_badwin}）")

    # ---------- 3) 报价表 q̂(分钟桶, 深度档) ← 5 天 15s 采样 ----------
    with open(SAMPLES, encoding="utf-8") as f:
        raw = json.load(f)
    agg: dict[int, tuple[float, float]] = {}
    byc: dict[int, list[list]] = {}
    for k in kl:
        byc.setdefault(int(k[0]) // 900_000, []).append(k)
    for cyc, arr2 in byc.items():
        if len(arr2) == 3:
            arr2.sort(key=lambda k: int(k[0]))
            agg[cyc] = (float(arr2[0][1]), float(arr2[-1][4]))
    qt: dict[int, dict[int, list[float]]] = {b: {d: [] for d in range(3)} for b in range(15)}
    qb_all: dict[int, list[float]] = {b: [] for b in range(15)}
    for s in raw:
        q = s.get("down_price")
        ts = int(s["timestamp"])
        btc = s.get("btc_price")
        cyc = ts // 900_000
        oc = agg.get(int(cyc))
        if q is None or q <= 0.01 or q >= 0.99 or btc is None or oc is None or oc[0] <= 0:
            continue
        if not (btc < oc[0]):          # 只收回落态（入场态）
            continue
        b = int((ts - cyc * 900_000) // 60_000)
        if b >= 15:
            continue
        qt[b][dbucket(-math.log(btc / oc[0]))].append(float(q))
        qb_all[b].append(float(q))

    def qhat(b: int, d: int) -> tuple[float, int]:
        """桶 b 深度档 d 的报价（档样本不足回退桶均值）。"""
        vals = qt[b][d]
        if len(vals) >= 30:
            return float(np.mean(vals)), len(vals)
        allv = qb_all[b]
        return (float(np.mean(allv)) if allv else float("nan")), len(allv)

    print("\n报价表 q̂(t, depth) ← 5 天线上采样（回落态 DOWN 报价）：")
    print("  t(分)  浅(<5bp)  中(5-15bp)  深(>15bp)  桶合计n")
    for b in range(15):
        cells = []
        for d in range(3):
            m, n = qhat(b, d)
            cells.append(f"{m:.3f}({n})" if n >= 30 else f"  ~{m:.3f}")
        print(f"  {b + 1:>4}   {cells[0]:>10} {cells[1]:>10} {cells[2]:>10}  {len(qb_all[b]):>6}")

    # ---------- 4) 主表：EV(t) ----------
    print("\n===== 主表：S1 形态下逐分钟回落确认入场 EV(t) =====")
    print("  t(分)   n   P(DOWN)  Wilson95      q̂eff   EV      EV[q̂±0.02]")
    main_rows = []
    for t in range(1, 15):
        grp = [p for p in paths if p["closes"][t - 1] < p["open"]]
        if len(grp) < 20:
            continue
        k = sum(p["win"] for p in grp)
        pw = k / len(grp)
        lo, hi = wilson(k, len(grp))
        qs = []
        for p in grp:
            m, _ = qhat(t - 1, dbucket(-math.log(p["closes"][t - 1] / p["open"])))
            qs.append(m)
        qeff = float(np.mean(qs))
        ev = float(np.mean([(0.98 / q - 1.0) if p["win"] else -1.0
                            for p, q in zip(grp, qs)]))
        ev_lo = ev - pw * 0.98 * 0.02 / (qeff * qeff)   # q̂+0.02 → EV 降
        ev_hi = ev + pw * 0.98 * 0.02 / (qeff * qeff)
        main_rows.append({"t": t, "n": len(grp), "p": pw, "ci": [lo, hi],
                          "q": qeff, "ev": ev, "ev_band": [ev_lo, ev_hi]})
        print(f"  {t:>4}  {len(grp):>5}  {pw:6.1%}  [{lo:.1%},{hi:.1%}]  {qeff:.3f}  {ev:+.3f}  [{ev_lo:+.3f},{ev_hi:+.3f}]")

    # 5m 口径 S5 复现（对照）
    s5_paths = [p for p in paths if p["closes"][4] < p["open"]]
    if s5_paths:
        pw5 = sum(p["win"] for p in s5_paths) / len(s5_paths)
        print(f"  对照 S5（t=5 行）: n={len(s5_paths)} P={pw5:.1%}（720d 验证 5m 口径参考 78.5%）")

    # ---------- 5) S6-min：首次回落 t* 即入场 ----------
    print("\n===== S6-min：首次回落分钟 t* 即入场 =====")
    tstar_rows = []
    for t in range(1, 15):
        grp = [p for p in paths
               if p["closes"][t - 1] < p["open"]
               and all(p["closes"][i] >= p["open"] for i in range(t - 1))]
        if len(grp) < 10:
            tstar_rows.append({"t": t, "n": len(grp)})
            continue
        k = sum(p["win"] for p in grp)
        qs = []
        for p in grp:
            m, _ = qhat(t - 1, dbucket(-math.log(p["closes"][t - 1] / p["open"])))
            qs.append(m)
        ev = float(np.mean([(0.98 / q - 1.0) if p["win"] else -1.0
                            for p, q in zip(grp, qs)]))
        tstar_rows.append({"t": t, "n": len(grp), "p": k / len(grp),
                           "q": float(np.mean(qs)), "ev": ev})
        print(f"  t*={t:>2}  n={len(grp):>4}  P={k / len(grp):6.1%}  q̂={float(np.mean(qs)):.3f}  EV={ev:+.3f}")
    # S6-min 总体（每个事件只在其 t* 入场一次）
    evs = []
    for p in paths:
        tstar = next((t for t in range(1, 16) if p["closes"][t - 1] < p["open"]), None)
        if tstar is None or tstar > 14:
            continue
        m, _ = qhat(tstar - 1, dbucket(-math.log(p["closes"][tstar - 1] / p["open"])))
        evs.append((0.98 / m - 1.0) if p["win"] else -1.0)
    if evs:
        se = float(np.std(evs) / math.sqrt(len(evs)))
        print(f"  S6-min 总体: n={len(evs)}  EV={float(np.mean(evs)):+.3f}±{1.96 * se:.3f}")

    # ---------- 6) A/B 段稳健性（主表关键行）----------
    print("\n===== A/B 段分段（主表口径）=====")
    print("  t(分)   |  A: n  P(DOWN) EV   |  B: n  P(DOWN) EV")
    ab_rows = []
    for t in range(1, 15):
        cells = []
        for seg in ("A", "B"):
            grp = [p for p in paths if p["seg"] == seg and p["closes"][t - 1] < p["open"]]
            if len(grp) >= 10:
                pw_ = sum(p["win"] for p in grp) / len(grp)
                qs = [qhat(t - 1, dbucket(-math.log(p["closes"][t - 1] / p["open"])))[0] for p in grp]
                ev_ = float(np.mean([(0.98 / q - 1.0) if p["win"] else -1.0
                                     for p, q in zip(grp, qs)]))
                cells.append((len(grp), pw_, ev_))
            else:
                cells.append((len(grp), float("nan"), float("nan")))
        ab_rows.append({"t": t, "A": cells[0], "B": cells[1]})
        (na, pa, ea), (nb, pb, eb) = cells
        print(f"  {t:>4}   | {na:>5} {pa:6.1%} {ea:+6.3f} | {nb:>5} {pb:6.1%} {eb:+6.3f}")

    with open("output/s6_s1_minute_backtest_result.json", "w", encoding="utf-8") as f:
        json.dump({"events": len(events), "paths": len(paths),
                   "quote_table": {str(b): {str(d): [float(np.mean(qt[b][d])), len(qt[b][d])]
                                            for d in range(3)} for b in range(15)},
                   "main_rows": main_rows, "tstar_rows": tstar_rows, "ab_rows": ab_rows},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 output/s6_s1_minute_backtest_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
