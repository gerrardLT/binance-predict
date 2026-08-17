#!/usr/bin/env python3
"""360 天全量历史的系统性科学发现：15m K 线延续/反转确定性场景搜索。

预注册研究纪律（执行前锁定，严禁事后修改）：
  1. 数据：官方 5m klines ×360天 → 精确聚合 15m（OHLCV + 5m 路径细节）。
  2. 时间切分：前 240 天 = 发现集（L1/L2/L3 只在此做），后 120 天 = 验证集（仅终验盲测）。
  3. 假设族：56 个预注册假设（形态/结构/时间/动量/量能/regime/路径 7 个维度）。
  4. L1 单因子：发现集 n≥300 且方向性偏离基准≥2pp → 入场券；BH-FDR(q=0.10) 多重校正。
  5. L2 双因子：入场券两两组合（机制方向冲突的组合剔除），n≥200。
  6. L3 三因子：L2 top5 × 其余入场券因子，n≥120。
  7. 终验（OOS）：验证集 n≥60、方向与发现集一致、点估计>52%（含 2% 费+0.01 溢价）。
     稳健性：按月分布 / 波动率+趋势 regime 分组 / run 块自助 CI。
  8. 盈亏比：@0.51 买入 → 赔率 b=0.98/0.51-1≈0.922，EV=p(1+b)-1，Kelly=p-(1-p)/b。

市场语义：所有假设的终极目标 = 次根 15m 收盘方向（UP/DOWN token 二选一）。
  expect='down' → 赌次根阴（胜率=次根阴率）；'up' → 赌次根阳；'either' → 探索性对照。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np
from scipy.stats import binomtest

FEE = 0.02
PREMIUM = 0.01
ODDS = (1 - FEE) / (0.50 + PREMIUM) - 1.0   # 赔率 b≈0.9216（赢 0.92 / 输 1）
BREAKEVEN = 1.0 / (1.0 + ODDS)               # 打平胜率 ≈ 52.04%
EPS = 0.0005
LOOKBACK = 48      # 5m × 48 = 4h 位势窗口
DAYS = 360
API = "https://data-api.binance.vision/api/v3/klines"
CACHE = "output/klines_5m_cache.json"
LOG = "output/full_history_discovery.log"


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
    if last < now_ms - 2 * 86_400_000:   # 缓存过期 → 全量
        kl = fetch_klines("5m", start_ms, now_ms)
    elif last < now_ms - 300_000:        # 增量补拉
        kl += fetch_klines("5m", last + 1, now_ms)
    kl = [k for k in kl if int(k[0]) >= start_ms]
    # 去重 + 排序
    seen = {}
    for k in kl:
        seen[int(k[0])] = k
    kl = [seen[t] for t in sorted(seen)]
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(kl, f)
    return kl


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


def roll_mean(x, w):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    return out


def roll_nanmean(x, w):
    """NaN-safe 滑动均值（cumsum 会传播 NaN，必须窗口内忽略）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    sw = sliding_window_view(x, w)
    with np.errstate(invalid="ignore"):
        out[w - 1:] = np.nanmean(sw, axis=1)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bh_fdr(pvals: list[float], q: float = 0.10) -> list[bool]:
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    thresh = q * np.arange(1, n + 1) / n
    below = ranked <= thresh
    if not below.any():
        return [False] * n
    cutoff = ranked[np.max(np.nonzero(below)[0])]
    return list(arr <= cutoff)


def run_block_ci(hit_cycs: np.ndarray, hits: np.ndarray, b: int = 3000,
                 seed: int = 11) -> tuple[float, float]:
    """连续命中窗口合并为 run（处理相邻重叠依赖），run 级自助。"""
    runs_v, runs_n = [], []
    start = 0
    for i in range(1, len(hit_cycs) + 1):
        if i == len(hit_cycs) or hit_cycs[i] != hit_cycs[i - 1] + 1:
            runs_v.append(int(hits[start:i].sum()))
            runs_n.append(i - start)
            start = i
    v = np.asarray(runs_v, dtype=float)
    w = np.asarray(runs_n, dtype=float)
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, len(v), size=(b, len(v)))
    means = v[sel].sum(axis=1) / w[sel].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


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
    print(f"15m K {N} 根（{time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[-1] * 900))}）")

    # ---------- 特征工程 ----------
    rng15 = np.where(h15 > l15, h15 - l15, np.nan) / o15
    dir15 = np.sign(c15 - o15)
    body_frac = np.abs(c15 - o15) / np.where(h15 > l15, h15 - l15, np.nan)
    upper_frac = (h15 - np.maximum(o15, c15)) / np.where(h15 > l15, h15 - l15, np.nan)
    lower_frac = (np.minimum(o15, c15) - l15) / np.where(h15 > l15, h15 - l15, np.nan)
    close_pos = (c15 - l15) / np.where(h15 > l15, h15 - l15, np.nan)

    # 前置 NaN 会被 cumsum 传播 → 必须用 NaN-safe 滑动均值（旧脚本此处为隐藏 bug）
    atr_prev = np.empty(N)
    atr_prev[:] = np.concatenate([[np.nan], roll_nanmean(rng15, 20)[:-1]])
    vratio = v15 / np.concatenate([[np.nan], roll_nanmean(v15, 20)[:-1]])

    streak = np.ones(N)
    for i in range(1, N):
        if dir15[i] == dir15[i - 1] and dir15[i] != 0:
            streak[i] = streak[i - 1] + 1
    prev_dir = np.concatenate([[np.nan], dir15[:-1]])
    prev_o = np.concatenate([[np.nan], o15[:-1]])
    prev_c = np.concatenate([[np.nan], c15[:-1]])
    prev_h = np.concatenate([[np.nan], h15[:-1]])
    prev_l = np.concatenate([[np.nan], l15[:-1]])

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

    # 24h 新高低（收盘 vs 前 96 根收盘极值）
    pm = roll_max(c15, 96)
    pmi = roll_min(c15, 96)
    prev_max96 = np.full(N, np.nan)
    prev_min96 = np.full(N, np.nan)
    prev_max96[1:] = pm[:-1]
    prev_min96[1:] = pmi[:-1]
    new_24h_hi = c15 > prev_max96
    new_24h_lo = c15 < prev_min96

    # 过去 16 根 15m（4h）区间位置 + 扫高/扫低失败
    w16_hi = roll_max(c15, 16)
    w16_lo = roll_min(c15, 16)
    prev_max16 = np.full(N, np.nan)
    prev_min16 = np.full(N, np.nan)
    prev_max16[1:] = w16_hi[:-1]
    prev_min16[1:] = w16_lo[:-1]
    pos4h = (c15 - w16_lo) / np.where(w16_hi > w16_lo, w16_hi - w16_lo, np.nan)
    sweep_hi = (h15 > prev_max16) & (close_pos <= 0.3)
    sweep_lo = (l15 < prev_min16) & (close_pos >= 0.7)

    # 吞没 / 内含（实体与区间关系）
    bull_engulf = (dir15 > 0) & (o15 <= np.minimum(prev_o, prev_c)) & (c15 > np.maximum(prev_o, prev_c))
    bear_engulf = (dir15 < 0) & (o15 >= np.maximum(prev_o, prev_c)) & (c15 < np.minimum(prev_o, prev_c))
    inside_bar = (h15 <= prev_h) & (l15 >= prev_l)

    # 当前 4h K 已走部分方向（桶内首根 5m open → 当前 15m 收盘）
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
    slot_in_4h = cyc_arr % 16          # 4h 周期内第几根 15m（0..15）

    # regime：波动率（前 20 根 ATR，发现集中位数定阈值）与 30 天趋势
    split = int(N * 2 / 3)
    atr_med = float(np.nanmedian(atr_prev[:split]))
    hi_vol = atr_prev >= atr_med
    ret30d = np.full(N, np.nan)
    ret30d[30 * 96:] = c15[30 * 96:] / c15[:-30 * 96] - 1.0
    bear_regime = ret30d < 0

    # ---------- 目标：次根方向 ----------
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

    print(f"发现集 {split} 根（{time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[0] * 900))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[split - 1] * 900))}） | "
          f"验证集 {N - split} 根（{time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[split] * 900))}"
          f" ~ {time.strftime('%Y-%m-%d', time.gmtime(cyc_arr[-1] * 900))}）")

    # ---------- 预注册假设族（56 个）----------
    H = []  # (name, mask, expect, mech)
    ecode = {"down": 1, "up": -1, "either": 0}

    def add(name, m, expect, mech):
        m = np.asarray(m, dtype=bool).copy()
        m[np.isnan(rng15)] = False
        H.append((name, m, expect, mech))

    # 对照
    add("F01 收阴(对照)", dir15 < 0, "down", "马尔可夫基准")
    add("F02 收阳(对照)", dir15 > 0, "up", "马尔可夫基准")
    # 形态维度
    add("F03 大实体阴 body≥70%", (dir15 < 0) & (body_frac >= 0.7), "down", "强卖压延续")
    add("F04 大实体阳 body≥70%", (dir15 > 0) & (body_frac >= 0.7), "up", "强买压延续")
    add("F05 光脚阴 close_pos≤0.15", (dir15 < 0) & (close_pos <= 0.15), "down", "收盘零反抗")
    add("F06 光头阳 close_pos≥0.85", (dir15 > 0) & (close_pos >= 0.85), "up", "收盘最强")
    add("F07 阴·长上影 upper≥40%", (dir15 < 0) & (upper_frac >= 0.4), "down", "上攻被拒")
    add("F08 阳·长下影 lower≥40%", (dir15 > 0) & (lower_frac >= 0.4), "up", "下探被买")
    add("F09 光头光脚阳", (dir15 > 0) & (upper_frac + lower_frac <= 0.15) & (body_frac >= 0.7), "up", "单边碾压")
    add("F10 光头光脚阴", (dir15 < 0) & (upper_frac + lower_frac <= 0.15) & (body_frac >= 0.7), "down", "单边碾压")
    add("F11 看涨吞没", bull_engulf, "up", "买方接管")
    add("F12 看跌吞没", bear_engulf, "down", "卖方接管")
    add("F13 内含线", inside_bar, "either", "波动收缩待选择")
    add("F14 pin上 upper≥60%·body≤30%", (upper_frac >= 0.6) & (body_frac <= 0.3), "down", "高位拒绝")
    add("F15 pin下 lower≥60%·body≤30%", (lower_frac >= 0.6) & (body_frac <= 0.3), "up", "低位承接")
    # 结构维度
    add("F16 破4h高位势", broke_hi15, "down", "fade 突破买盘")
    add("F17 破4h低位势", broke_lo15, "up", "fade 突破卖盘")
    add("F18 破4h高·收阳", broke_hi15 & (dir15 > 0), "down", "旧信号")
    add("F19 破4h高·收阴", broke_hi15 & (dir15 < 0), "down", "冲高回落")
    add("F20 破4h低·收阴", broke_lo15 & (dir15 < 0), "up", "破位衰竭")
    add("F21 破4h低·收阳", broke_lo15 & (dir15 > 0), "up", "V形回收")
    add("F22 破4h高·光头阳(旧王牌)", broke_hi15 & (dir15 > 0) & (close_pos >= 0.85), "down", "买力耗尽")
    add("F23 收盘24h新高", new_24h_hi, "down", "极端位 fade")
    add("F24 收盘24h新低", new_24h_lo, "up", "极端位 fade")
    add("F25 4h区间上沿 pos≥0.9", pos4h >= 0.9, "down", "区间边回落")
    add("F26 4h区间下沿 pos≤0.1", pos4h <= 0.1, "up", "区间边反弹")
    add("F27 扫高失败", sweep_hi, "down", "冲高卖方接管")
    add("F28 扫低失败", sweep_lo, "up", "探底买方接管")
    # 时间维度
    add("F29 亚洲时段(0-7)", hours <= 7, "either", "时段结构")
    add("F30 欧洲时段(8-15)", (hours >= 8) & (hours <= 15), "either", "时段结构")
    add("F31 美洲时段(16-23)", hours >= 16, "either", "时段结构")
    add("F32 周末", wd >= 5, "either", "流动性结构")
    add("F33 4h首根15m", slot_in_4h == 0, "either", "周期开启效应")
    add("F34 4h末根15m", slot_in_4h == 15, "either", "周期结算效应")
    add("F35 周一", wd == 0, "either", "周界效应")
    add("F36 周五", wd == 4, "either", "周界效应")
    # 动量维度
    add("F37 连阴≥2", (dir15 < 0) & (streak >= 2), "down", "动量延续")
    add("F38 连阳≥2", (dir15 > 0) & (streak >= 2), "up", "动量延续")
    add("F39 连阴≥3", (dir15 < 0) & (streak >= 3), "down", "动量衰竭?")
    add("F40 连阳≥3", (dir15 > 0) & (streak >= 3), "up", "动量衰竭?")
    add("F41 阴后阴", (dir15 < 0) & (prev_dir < 0), "down", "马尔可夫")
    add("F42 阳后阳", (dir15 > 0) & (prev_dir > 0), "up", "马尔可夫")
    add("F43 阳·尾段5m回落", (dir15 > 0) & (last5_dir < 0), "down", "收盘泄力")
    add("F44 阴·尾段5m反抽", (dir15 < 0) & (last5_dir > 0), "up", "收盘反抽")
    # 量能/波动维度
    add("F45 振幅爆发 rng≥2×ATR20", rng15 >= 2 * atr_prev, "either", "波动冲击")
    add("F46 压缩 rng≤0.5×ATR20", rng15 <= 0.5 * atr_prev, "either", "波动压缩")
    add("F47 放量 v≥2×均量", vratio >= 2, "either", "量能冲击")
    add("F48 缩量 v≤0.5×均量", vratio <= 0.5, "either", "量能萎缩")
    add("F49 单边路径 ER≥0.8", er >= 0.8, "either", "效率路径")
    add("F50 震荡路径 ER≤0.4", er <= 0.4, "either", "噪声路径")
    # regime 维度
    add("F51 高波regime", hi_vol, "either", "波动环境")
    add("F52 熊市regime(30d跌)", bear_regime, "either", "趋势环境")
    # 路径维度
    add("F53 路径UUD(冲高回落)", path3 == "UUD", "down", "尾盘转弱")
    add("F54 路径DUU(探底回升)", path3 == "DUU", "up", "尾盘转强")
    add("F55 顺4h势", dir15 * run_dir4h > 0, "either", "顺势延续")
    add("F56 逆4h势", dir15 * run_dir4h < 0, "either", "逆势反转")

    # ---------- 评估函数 ----------
    base_d = float(nxt_down[:split][has_next[:split]].mean())
    base_s = float(nxt_same[:split][same_valid[:split]].mean())
    print(f"\n发现集基准：次根↓ {base_d:.1%} | 次根↑ {1 - base_d:.1%} | 延续率 {base_s:.1%} | 打平 {BREAKEVEN:.1%} | 赔率 {ODDS:.3f}")

    def evaluate(mask, expect, end, start=0):
        m = mask[start:end] & has_next[start:end]
        n = int(m.sum())
        if n < 5:
            return None
        k_d = int(nxt_down[start:end][m].sum())
        pd = k_d / n
        base = float(nxt_down[start:end][has_next[start:end]].mean())
        ms = mask[start:end] & same_valid[start:end]
        ps = float(nxt_same[start:end][ms].mean()) if ms.sum() >= 30 else np.nan
        # 假设命中率（expect 方向）
        if expect == "down":
            p_hat, k_hat, alt = pd, k_d, "greater"
        elif expect == "up":
            p_hat, k_hat, alt = 1 - pd, n - k_d, "less"
        else:
            p_hat, k_hat, alt = max(pd, 1 - pd), max(k_d, n - k_d), "two-sided"
        base_hat = base if expect == "down" else (1 - base if expect == "up" else 0.5)
        return {"n": n, "k_d": k_d, "pd": pd, "ps": ps, "p_hat": p_hat, "k_hat": k_hat,
                "base_hat": base_hat, "dev": p_hat - base_hat,
                "pval": binomtest(k_hat, n, base_hat, alternative=alt).pvalue if n >= 30 else 1.0,
                "wilson": wilson(k_hat, n)}

    def line(name, r, expect):
        tag = {"down": "↓", "up": "↑", "either": "~"}[expect]
        lo, hi = r["wilson"]
        return (f"  {name} [{tag}]: n={r['n']:>5} 胜率 {r['p_hat']:.1%} ({r['dev']:+.1%}pp) "
                f"[{lo:.1%},{hi:.1%}] EV {r['p_hat'] * (1 + ODDS) - 1:+.3f} | 次根↓ {r['pd']:.1%}")

    # ---------- L1：单因子 + BH-FDR ----------
    print(f"\n===== L1 单因子（{len(H)} 个预注册假设，发现集 {split} 根）=====")
    l1_all = []
    for name, m, expect, mech in H:
        r = evaluate(m, expect, split)
        if r and r["n"] >= 300 and abs(r["dev"]) >= 0.02:
            r.update({"name": name, "expect": expect, "mech": mech})
            l1_all.append(r)
    fdr = bh_fdr([r["pval"] for r in l1_all], q=0.10)
    for r, f in zip(l1_all, fdr):
        r["fdr"] = f
    l1_all.sort(key=lambda r: -abs(r["dev"]))
    for r in l1_all:
        print(line(r["name"], r, r["expect"]) + (f"  [FDR✓]" if r["fdr"] else "  [FDR✗]"))
    passed = [r for r in l1_all if r["fdr"]]
    print(f"L1 入场券（n≥300 & |dev|≥2pp & FDR✓）：{len(passed)} 个")

    # ---------- L2：双因子 ----------
    def combo_expect(e1, e2):
        c1, c2 = ecode[e1], ecode[e2]
        if c1 * c2 < 0:
            return None            # 机制方向冲突
        if c1 != 0:
            return e1 if c1 == c2 or c2 == 0 else e2
        return e2

    hmap = {n: m for n, m, _, _ in H}
    emap = {n: e for n, _, e, _ in H}
    cand_names = [r["name"] for r in passed]
    print(f"\n===== L2 双因子（{len(cand_names)} 个入场券因子两两组合，发现集）=====")
    l2 = []
    for i in range(len(cand_names)):
        for j in range(i + 1, len(cand_names)):
            n1, n2 = cand_names[i], cand_names[j]
            ce = combo_expect(emap[n1], emap[n2])
            if ce is None:
                continue
            r = evaluate(hmap[n1] & hmap[n2], ce, split)
            if r and r["n"] >= 200 and abs(r["dev"]) >= 0.02:
                r.update({"name": f"{n1} × {n2}", "expect": ce})
                l2.append(r)
    l2.sort(key=lambda r: -abs(r["dev"]))
    for r in l2[:25]:
        print(line(r["name"], r, r["expect"]))
    print(f"L2 通过（n≥200 & |dev|≥2pp）：{len(l2)} 个")

    # ---------- L3：三因子 ----------
    print(f"\n===== L3 三因子（L2 top5 × 其余强因子，发现集）=====")
    l3 = []
    for r2 in l2[:5]:
        parts = r2["name"].split(" × ")
        for n3 in cand_names:
            if n3 in parts:
                continue
            ce = combo_expect(combo_expect(emap[parts[0]], emap[parts[1]]), emap[n3])
            if ce is None:
                continue
            r = evaluate(hmap[parts[0]] & hmap[parts[1]] & hmap[n3], ce, split)
            if r and r["n"] >= 120 and abs(r["dev"]) >= 0.02:
                r.update({"name": f"{r2['name']} × {n3}", "expect": ce})
                l3.append(r)
    l3.sort(key=lambda r: -abs(r["dev"]))
    for r in l3[:15]:
        print(line(r["name"], r, r["expect"]))

    # ---------- 终验：候选 → 验证集（120 天 OOS）+ 稳健性 ----------
    def to_mask(name: str) -> np.ndarray:
        m = None
        for p in name.split(" × "):
            m = hmap[p].copy() if m is None else m & hmap[p]
        return m

    def to_expect(name: str) -> str:
        e = "either"
        for p in name.split(" × "):
            e2 = combo_expect(e, emap[p])
            if e2 is None:
                return "down"
            e = e2
        return e

    pool = {}
    for r in l3:
        if r["n"] >= 150:
            pool[r["name"]] = r
    for r in l2:
        if r["n"] >= 250:
            pool.setdefault(r["name"], r)
    for r in l1_all:                       # L1 强单因子全部进入终验
        pool.setdefault(r["name"], r)
    f22_name = "F22 破4h高·光头阳(旧王牌)"
    if f22_name not in pool:               # 旧王牌强制复验
        r22 = evaluate(hmap[f22_name], "down", split)
        if r22:
            r22.update({"name": f22_name, "expect": "down"})
            pool[f22_name] = r22
    ranked = sorted(pool.values(), key=lambda r: -abs(r.get("dev", 0)))
    print(f"\n===== 终验：{len(ranked)} 个候选（语义去重后取 top8）→ 验证集（后 120 天 OOS）=====")
    base_d_v = float(nxt_down[split:][has_next[split:]].mean())
    print(f"验证集基准：次根↓ {base_d_v:.1%} | 次根↑ {1 - base_d_v:.1%}")

    def hit_sig(mask, end) -> int:
        return hash(np.nonzero(mask[:end] & has_next[:end])[0].tobytes())

    finals, seen = [], set()
    for r in ranked:
        if len(finals) >= 8:
            break
        name = r["name"]
        m = to_mask(name)
        s = hit_sig(m, split)
        if s in seen:                      # 同一样本集的组合语义等价 → 去重
            continue
        seen.add(s)
        expect0 = r.get("expect") or to_expect(name)
        rd = evaluate(m, expect0, split)
        if not rd:
            continue
        # 方向校准：预注册方向与发现集数据方向相反时（dev<0），按发现集确定的方向评估
        #（方向只由发现集决定，验证集仍为盲测；报告标注 flipped）
        flipped = False
        expect = expect0
        if expect0 != "either" and np.sign(rd["dev"]) < 0:
            flipped = True
            expect = "up" if expect0 == "down" else "down"
            rd = evaluate(m, expect, split)
        rv = evaluate(m, expect, N, start=split)  # 严格验证集（不含发现集）
        if not rv:
            continue
        # 验证集命中明细（月度 / regime / run 块自助）
        mv = m & has_next
        hit_idx = np.nonzero(mv)[0]
        hit_cyc = cyc_arr[hit_idx]
        if expect == "down":
            hit_out = nxt_down[hit_idx].astype(float)
        else:
            hit_out = (~nxt_down[hit_idx]).astype(float)
        months: dict[str, list[float]] = {}
        for j, o in zip(hit_idx, hit_out):
            if j >= split:
                months.setdefault(time.strftime("%Y-%m", time.gmtime(cyc_arr[j] * 900)), []).append(float(o))
        hv = hit_out[hi_vol[hit_idx]]
        bv = hit_out[bear_regime[hit_idx]]
        lv = hit_out[~hi_vol[hit_idx] & ~np.isnan(atr_prev[hit_idx])]
        brv = hit_out[~bear_regime[hit_idx] & ~np.isnan(ret30d[hit_idx])]
        blo, bhi = run_block_ci(hit_cyc, hit_out)

        verdict = "FAILED"
        if rv["n"] >= 60 and np.sign(rv["dev"]) == np.sign(rd["dev"]) and rv["p_hat"] > BREAKEVEN:
            verdict = "CONFIRMED" if rv["wilson"][0] > BREAKEVEN - 0.02 else "WEAK_PASS"

        mech_map = {n: mech for n, _, _, mech in H}
        mech0 = mech_map.get(name.split(" × ")[0], "")
        flip_tag = " [方向翻转:预注册延续,发现集支持反转]" if flipped else ""
        fdr_tag = " FDR✓" if r.get("fdr") else ""
        print(f"\n  ◆ {name} [{expect}]{flip_tag}{fdr_tag} 机制:{mech0}")
        print(f"    {line('发现集', rd, expect)}")
        print(f"    {line('验证集', rv, expect)}")
        hv_s = f"{hv.mean():.1%}(n={len(hv)})" if len(hv) else "-"
        lv_s = f"{lv.mean():.1%}(n={len(lv)})" if len(lv) else "-"
        bv_s = f"{bv.mean():.1%}(n={len(bv)})" if len(bv) else "-"
        brv_s = f"{brv.mean():.1%}(n={len(brv)})" if len(brv) else "-"
        print(f"    run块自助CI [{blo:.1%},{bhi:.1%}] | 高波 {hv_s} 低波 {lv_s} | 熊 {bv_s} 牛 {brv_s}")
        print(f"    验证集按月: " + " / ".join(f"{m_}:{sum(b):.0f}/{len(b)}={sum(b) / len(b):.0%}" for m_, b in sorted(months.items())))
        print(f"    裁决: {verdict} | Kelly={rv['p_hat'] - (1 - rv['p_hat']) / ODDS:.3f}")
        finals.append({"name": name, "expect": expect, "flipped": flipped, "disc": rd, "valid": rv,
                       "block_ci": [float(blo), float(bhi)], "verdict": verdict,
                       "months": {k: [sum(v), len(v)] for k, v in months.items()}})

    # ---------- 汇总 ----------
    ok = [f for f in finals if f["verdict"] in ("CONFIRMED", "WEAK_PASS")]
    print(f"\n===== 结论汇总（360 天全量，240 发现 / 120 OOS）=====")
    print(f"通过 OOS 终验的场景：{len(ok)} / {len(finals)}")
    for f in ok:
        rv = f["valid"]
        flip = " [方向由发现集校准]" if f["flipped"] else ""
        print(f"  [{f['verdict']}] {f['name']} [{f['expect']}]{flip}：验证集 n={rv['n']} 胜率 {rv['p_hat']:.1%} "
              f"[{rv['wilson'][0]:.1%},{rv['wilson'][1]:.1%}] EV {rv['p_hat'] * (1 + ODDS) - 1:+.3f} "
              f"盈亏比 1:{ODDS:.2f}")
    with open("output/full_history_discovery_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"days": DAYS, "split": split, "n_15m": N, "breakeven": BREAKEVEN,
                     "odds": ODDS, "base_d_disc": base_d, "base_d_valid": base_d_v},
            "l1": [{k: (float(v) if isinstance(v, (np.floating, float)) and not isinstance(v, bool) else
                        [float(x) for x in v] if isinstance(v, tuple) else
                        bool(v) if isinstance(v, (bool, np.bool_)) else v)
                    for k, v in r.items()} for r in l1_all],
            "finals": finals,
        }, f, ensure_ascii=False, indent=2, default=str)
    print("\n结果已存 output/full_history_discovery_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
