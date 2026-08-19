#!/usr/bin/env python3
"""720 天（2024-08 ~ 2026-08）固定参数验证：S1/S2/S4/S5 线上口径跨状态稳健性。

预注册验证纪律（参数锁定，2026-08-18 执行前冻结，严禁事后调整）：
  1. 数据：官方 5m klines ×720 天 → 精确聚合 15m（OHLCV）。
     - A 段（盲验）= 前 360 天（2024-08~2025-08）：参数发现过程从未接触的数据
     - B 段（对照）= 后 360 天（2025-08~2026-08）：原 360 天回测窗口，应复现已知结果
  2. 模式 = 线上 detector 锁定口径（DEFAULT_SCENE_PARAMS + 真 OOS 完整口径）：
     - S1 bull_exhaust  = 破4h高 & 收阳 & close_pos≥0.85 & pos4h≥0.9  → 次根 DOWN
     - S2 bear_exhaust  = 破4h低 & 收阴 & vol_ratio≥2.0               → 次根 UP
     - S4 momentum_fade = 连阳≥3 & 光头阳(close_pos≥0.85)，无破位要求 → 次根 DOWN
     - S5 confirm       = S1 & 次周期第1根5m收盘 < 次周期开盘（d1<0） → 次根 DOWN
  3. 指标：n / 胜率 / Wilson95% / EV@0.51（赔率 b=0.9216）/ 对窗口基准偏离；8×90 天分段。
  4. 预注册判定：A 段 Wilson 下界 > 52.04%（打平线）= 跨状态稳健；
     下界 < 50% = 失效信号（对近一年行情状态过拟合）。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np

FEE = 0.02
PREMIUM = 0.01
ODDS = (1 - FEE) / (0.50 + PREMIUM) - 1.0   # b≈0.9216
BREAKEVEN = 1.0 / (1.0 + ODDS)              # ≈52.04%
EPS = 0.0005
LOOKBACK = 48
DAYS = 720
API = "https://data-api.binance.vision/api/v3/klines"
CACHE = "output/klines_5m_cache_720d.json"
LOG = "output/validation_720d.log"


def fetch_klines(start_ms: int, end_ms: int) -> list[list]:
    out, cur = [], start_ms
    while cur < end_ms:
        url = f"{API}?symbol=BTCUSDT&interval=5m&startTime={cur}&endTime={end_ms}&limit=1000"
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
        if len(out) % 20000 < 1000:
            print(f"  已拉取 {len(out)} 根 5m ...")
        time.sleep(0.2)
    return out


def load_or_fetch(now_ms: int) -> list[list]:
    start_ms = now_ms - DAYS * 86_400_000
    kl: list[list] = []
    try:
        with open(CACHE, encoding="utf-8") as f:
            kl = json.load(f)
        print(f"缓存命中：{len(kl)} 根 5m")
    except Exception:
        pass
    last = int(kl[-1][0]) if kl else 0
    if last < now_ms - 2 * 86_400_000:       # 缓存过期 → 全量
        kl = fetch_klines(start_ms, now_ms)
    elif last < now_ms - 300_000:            # 增量补拉
        kl += fetch_klines(last + 1, now_ms)
    kl = [k for k in kl if int(k[0]) >= start_ms]
    seen = {}
    for k in kl:
        seen[int(k[0])] = k
    kl = [seen[t] for t in sorted(seen)]
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(kl, f)
    return kl


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
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


MODES = [
    ("S1 多头耗尽(bull_exhaust)", "s1", "down", 0.644),
    ("S2 空头耗尽(bear_exhaust)", "s2", "up", 0.536),
    ("S4 动量衰竭(momentum_fade)", "s4", "down", 0.554),
    ("S5 确认入场(bull_exhaust_confirm)", "s5", "down", 0.785),
]


def main() -> int:
    sys.stdout = Tee()
    now_ms = int(time.time() * 1000)
    kl = load_or_fetch(now_ms)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])) for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    t5 = np.array([r[0] for r in c5])
    o5 = np.array([r[1] for r in c5])
    h5 = np.array([r[2] for r in c5])
    l5 = np.array([r[3] for r in c5])
    cl5 = np.array([r[4] for r in c5])
    v5 = np.array([r[5] for r in c5])
    print(f"5m K {len(c5)} 根")

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
                   cl5[idxs[-1]], float(sum(v5[i] for i in idxs)), idxs)
    cyc_list.sort()
    N = len(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15 = np.array([ks[c][0] for c in cyc_list])
    h15 = np.array([ks[c][1] for c in cyc_list])
    l15 = np.array([ks[c][2] for c in cyc_list])
    c15 = np.array([ks[c][3] for c in cyc_list])
    v15 = np.array([ks[c][4] for c in cyc_list])
    t0 = time.strftime("%Y-%m-%d", time.gmtime(cyc_arr[0] * 900))
    t1 = time.strftime("%Y-%m-%d", time.gmtime(cyc_arr[-1] * 900))
    print(f"15m K {N} 根（{t0} ~ {t1}）")

    # ---------- 特征（口径对齐 local_full_history_discovery / 线上 detector）----------
    rng15 = np.where(h15 > l15, h15 - l15, np.nan)
    dir15 = np.sign(c15 - o15)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

    # 均量（前 20 根，不含当前；NaN-safe）
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

    # 过去 16 根 15m（4h）收盘区间位置 pos4h（含当前根窗口，对齐 discovery）
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

    w16_hi = roll_max(c15, 16)
    w16_lo = roll_min(c15, 16)
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)

    # 5m 层 4h 位势破位（前 48 根 5m 收盘极值）→ 映射 15m
    lvl_hi = np.full(len(c5), np.nan)
    lvl_lo = np.full(len(c5), np.nan)
    lvl_hi[1:] = roll_max(cl5, LOOKBACK)[:-1]
    lvl_lo[1:] = roll_min(cl5, LOOKBACK)[:-1]
    broke_hi5 = h5 > lvl_hi * (1 + EPS)
    broke_lo5 = l5 < lvl_lo * (1 - EPS)
    cont = np.zeros(len(c5), dtype=bool)
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

    # 次周期目标 + S5 确认量 d1（次周期第 1 根 5m 收盘 vs 该根开盘）
    nxt_down = np.zeros(N, dtype=bool)
    has_next = np.zeros(N, dtype=bool)
    d1 = np.full(N, np.nan)
    for j in range(N - 1):
        if cyc_arr[j + 1] != cyc_arr[j] + 1:
            continue
        i0 = ks[cyc_list[j + 1]][5][0]
        if o5[i0] > 0:
            d1[j] = cl5[i0] / o5[i0] - 1.0
        nd = dir15[j + 1]
        if nd != 0:
            has_next[j] = True
            nxt_down[j] = nd < 0

    # ---------- 模式谓词（线上锁定口径；NaN 一律 False）----------
    s1 = broke_hi15 & (dir15 > 0) & (close_pos >= 0.85) & (np.nan_to_num(pos4h, nan=-1) >= 0.9)
    s2 = broke_lo15 & (dir15 < 0) & (vratio >= 2.0)
    s4 = (dir15 > 0) & (streak >= 3) & (close_pos >= 0.85)
    s5 = s1 & (d1 < 0)
    s5_skip = s1 & ~(d1 < 0)
    masks = {"s1": s1, "s2": s2, "s4": s4, "s5": s5, "s5_skip": s5_skip}

    # ---------- 窗口切分 ----------
    ms_arr = cyc_arr * 900_000
    a_end = now_ms - 360 * 86_400_000
    win_all = np.ones(N, dtype=bool)
    win_a = ms_arr < a_end
    win_b = ms_arr >= a_end
    print(f"\nA 段（盲验 360 天）：{time.strftime('%Y-%m-%d', time.gmtime(ms_arr[0] / 1000))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(a_end / 1000))}，{int(win_a.sum())} 根 15m")
    print(f"B 段（对照 360 天）：{time.strftime('%Y-%m-%d', time.gmtime(a_end / 1000))}"
          f" ~ {t1}，{int(win_b.sum())} 根 15m")

    def evaluate(mask: np.ndarray, expect: str, win: np.ndarray) -> dict | None:
        m = mask & win & has_next
        n = int(m.sum())
        if n < 5:
            return None
        k_d = int(nxt_down[m].sum())
        pd_ = k_d / n
        base = float(nxt_down[win & has_next].mean())
        if expect == "down":
            p_hat, k_hat = pd_, k_d
            base_hat = base
        else:
            p_hat, k_hat = 1 - pd_, n - k_d
            base_hat = 1 - base
        lo, hi = wilson(k_hat, n)
        return {"n": n, "p_hat": p_hat, "dev": p_hat - base_hat, "wilson": (lo, hi),
                "ev": p_hat * (1 + ODDS) - 1.0, "pd": pd_}

    def line(tag: str, r: dict) -> str:
        lo, hi = r["wilson"]
        return (f"    {tag}: n={r['n']:>5} 胜率 {r['p_hat']:.1%} ({r['dev']:+.1%}pp) "
                f"[{lo:.1%},{hi:.1%}] EV@0.51 {r['ev']:+.3f}")

    # ---------- 总表 ----------
    print(f"\n===== 各模式 × 窗口（打平线 {BREAKEVEN:.1%}，赔率 {ODDS:.3f}）=====")
    report = {"meta": {"days": DAYS, "n_15m": N, "t0": t0, "t1": t1,
                       "breakeven": BREAKEVEN, "odds": ODDS}, "modes": {}}
    for name, key, expect, p_ref in MODES:
        print(f"\n  ◆ {name}（线上参考胜率 {p_ref:.1%}）")
        entry = {}
        for tag, win in (("全量720d", win_all), ("A段盲验360d", win_a), ("B段对照360d", win_b)):
            r = evaluate(masks[key], expect, win)
            if r:
                print(line(tag, r))
                entry[tag] = r
        # S5 额外输出：对照组（反向上涨+平价放弃组）
        if key == "s5":
            for tag, win in (("A段盲验360d", win_a), ("B段对照360d", win_b)):
                r = evaluate(masks["s5_skip"], "down", win)
                if r:
                    print(line(f"    └ 放弃组(对照)", r) if False else f"    └ 放弃组 {tag}: n={r['n']:>5} 胜率 {r['p_hat']:.1%} [{r['wilson'][0]:.1%},{r['wilson'][1]:.1%}]")
                    entry[f"skip_{tag}"] = r
        report["modes"][key] = entry

    # ---------- 8 × 90 天分段稳定性 ----------
    print("\n===== 90 天分段稳定性 =====")
    seg_days = 90
    n_seg = DAYS // seg_days
    seg_stats = {k: [] for k in masks}
    seg_labels = []
    for si in range(n_seg):
        s_ms = now_ms - (n_seg - si) * seg_days * 86_400_000
        e_ms = s_ms + seg_days * 86_400_000
        w = (ms_arr >= s_ms) & (ms_arr < e_ms)
        seg_labels.append(time.strftime("%y-%m", time.gmtime(s_ms / 1000)))
        for _, k, expect, _p in MODES:
            r = evaluate(masks[k], expect, w)
            seg_stats[k].append(r)
    header = "  模式        " + " ".join(f"{lb:>9}" for lb in seg_labels)
    print(header)
    for name, key, _e, _p in MODES:
        cells = []
        for r in seg_stats[key]:
            cells.append(f"{r['p_hat'] * 100:4.0f}%/{r['n']:<4d}" if r else "    -    ")
        print(f"  {name[:12]:<12}" + " ".join(f"{c:>9}" for c in cells))
    report["segments"] = {"labels": seg_labels, "stats": seg_stats}

    # ---------- S5 确认率（S1 → S5 转化）----------
    print("\n===== S5 确认率（S1 事件中 d1<0 占比）=====")
    for tag, win in (("A段盲验", win_a), ("B段对照", win_b), ("全量", win_all)):
        m = s1 & win & has_next
        n1 = int(m.sum())
        n5 = int((s5 & win & has_next).sum())
        if n1:
            print(f"    {tag}: S1 n={n1} → 确认 n={n5}（{n5 / n1:.1%}）")
    report["s5_confirm_rate"] = {
        tag: {"s1": int((s1 & win & has_next).sum()), "s5": int((s5 & win & has_next).sum())}
        for tag, win in (("A", win_a), ("B", win_b))
    }

    # ---------- 预注册裁决 ----------
    print("\n===== 预注册裁决（A 段盲验 Wilson 下界 vs 打平线）=====")
    verdicts = {}
    for name, key, _, _ in MODES:
        r = report["modes"][key].get("A段盲验360d")
        if not r:
            print(f"  {name}: 样本不足（n<5）→ 无法裁决")
            verdicts[key] = "INSUFFICIENT"
            continue
        lo = r["wilson"][0]
        if lo > BREAKEVEN:
            v = "ROBUST"
        elif lo > 0.50:
            v = "WEAK"
        else:
            v = "FAILED"
        verdicts[key] = v
        print(f"  {name}: A段 n={r['n']} 胜率 {r['p_hat']:.1%} 下界 {lo:.1%} → {v}")
    report["verdicts"] = verdicts

    with open("output/validation_720d_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print("\n结果已存 output/validation_720d_result.json；日志 output/validation_720d.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
