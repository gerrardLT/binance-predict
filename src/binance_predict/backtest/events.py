"""破位事件与场景判定引擎（参数化，对齐线上 detector 与 180 天回测口径）。

事件语义（每级别每方向每 15m 周期首次破位）：
- 破位：5m high > 前 lookback 根 close 极值 ×(1+eps)（或 low < ×(1-eps)）
- 场景① bull_exhaust：破阻力 + 当周期收阳 + close_pos ≥ close_pos_min
  → 次周期看 DOWN（win = next_down）
- 场景② bear_exhaust：破支撑 + 当周期收阴 + vol_ratio ≥ vol_ratio_min
  → 次周期看 UP（win = not next_down）
结算 = 次周期周期锚点（次周期末价 vs 次周期开盘），与币安预测市场一致。
"""
from __future__ import annotations

import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..services.scene_params import SceneParams
from .data import aggregate_15m


def build_events(c5: list[tuple], params: SceneParams, now_ms: int) -> dict:
    """从 5m K 构建破位事件与场景命中标记。

    Args:
        c5: [(open_time_ms, o, h, l, c, v), ...] 升序（已剔未收盘根）
        params: 场景参数集
        now_ms: 当前时刻（过滤未完成周期）

    Returns:
        {"events": [...], "agg": aggregate_15m 结果, "sigma15": float,
         "cyc_set": set, "cyc_arr": np.array, "lookbacks": {...}}
    """
    agg = aggregate_15m(c5)
    cyc_list = agg["cycs"]
    cyc_set = set(cyc_list)
    cyc_arr = np.array(cyc_list)
    o15, h15, l15, c15, v15 = agg["o15"], agg["h15"], agg["l15"], agg["c15"], agg["v15"]
    N = len(cyc_list)
    sigma15 = float(np.std([(c - o) / o for o, c in zip(o15, c15) if c != o]))

    # 滚动位势：按最大 lookback 拉窗口，各级别切片取极值（对齐 detector 口径）
    cl5 = np.array([r[4] for r in c5], dtype=float)
    lookbacks = dict(params.level_lookbacks)
    max_lb = max(lookbacks.values()) if lookbacks else 48
    lvl_max = np.full(len(c5), np.nan)
    lvl_min = np.full(len(c5), np.nan)
    if len(c5) >= max_lb:
        lvl_max[max_lb - 1:] = sliding_window_view(cl5, max_lb).max(axis=1)
        lvl_min[max_lb - 1:] = sliding_window_view(cl5, max_lb).min(axis=1)
    cont = agg["cont"]
    h5 = np.array([r[2] for r in c5], dtype=float)
    l5 = np.array([r[3] for r in c5], dtype=float)

    # 15m 特征：close_pos / vol_ratio（对齐 detector classify_close_pattern 输入）
    rng15 = np.where(np.array(h15) > np.array(l15), np.array(h15) - np.array(l15), np.nan)
    close_pos = (np.array(c15) - np.array(l15)) / rng15
    vs = np.array(v15, dtype=float)
    # vma[j] = 前 vol_ma_window 根 15m 均量（不含当前根）；冷启动历史 ≥ 半窗才算
    # （对齐线上 _confirm_and_fire：len(hist) >= VOL_MA_WINDOW // 2 才计均量，否则 None）
    w = params.vol_ma_window
    cs = np.concatenate([[0.0], np.cumsum(vs)])  # cs[j] = sum(vs[:j])
    vma = np.full(N, np.nan)
    for j in range(1, N):
        lo_j = max(0, j - w)
        cnt = j - lo_j
        if cnt >= max(1, w // 2):
            vma[j] = (cs[j] - cs[lo_j]) / cnt
    vol_ratio = vs / np.where(vma > 0, vma, np.nan)

    events: list[dict] = []
    for j, cyc in enumerate(cyc_list):
        op = o15[j]
        if op <= 0 or c15[j] == op:
            continue
        red = bool(c15[j] < op)
        month = time.strftime("%Y-%m", time.gmtime(cyc * 900))
        # 次周期（连续性要求：紧邻下一周期存在且 3 根齐全）
        nxt = cyc + 1
        has_next = (j + 1 < N) and cyc_list[j + 1] == nxt
        next_down = None
        z5 = z10 = None
        if has_next:
            nidx = agg["buckets"][nxt]
            nop = c5[nidx[0]][1]
            if nop > 0 and c15[j + 1] != o15[j + 1]:
                next_down = bool(c15[j + 1] < o15[j + 1])
                # 次周期 t=5/10min 的 z 状态（定价对照用，对齐 local_entry_timing_v2）
                d1 = c5[nidx[0]][4] / nop - 1
                d2 = c5[nidx[1]][4] / nop - 1
                z5 = d1 / (sigma15 * (10 / 15) ** 0.5) if sigma15 > 0 else None
                z10 = d2 / (sigma15 * (5 / 15) ** 0.5) if sigma15 > 0 else None

        for level, lb in lookbacks.items():
            for side in ("high", "low"):
                for k, i in enumerate(agg["buckets"][cyc]):
                    if not cont[i] or i < lb or np.isnan(lvl_max[i]):
                        continue
                    # 级别切片：窗口极值需基于该级别 lookback（重算该 bar 的级别位势）
                    lo_i = i - lb
                    win = cl5[lo_i:i]
                    if len(win) < lb:
                        continue
                    res, sup = float(win.max()), float(win.min())
                    if side == "high":
                        if not h5[i] > res * (1 + params.eps):
                            continue
                        trig = res * (1 + params.eps)
                    else:
                        if not l5[i] < sup * (1 - params.eps):
                            continue
                        trig = sup * (1 - params.eps)
                    cp = float(close_pos[j]) if close_pos[j] == close_pos[j] else None
                    vr = float(vol_ratio[j]) if vol_ratio[j] == vol_ratio[j] else None
                    scene1 = side == "high" and (not red) and cp is not None and cp >= params.close_pos_min
                    scene2 = side == "low" and red and vr is not None and vr >= params.vol_ratio_min
                    events.append({
                        "cyc": cyc, "level": level, "side": side, "bar": k,
                        "bp": abs(trig / op - 1), "trig": trig,
                        "close_pos": cp, "vol_ratio": vr, "red": red,
                        "scene1": scene1, "scene2": scene2,
                        "has_next": has_next, "next_down": next_down,
                        "z5": z5, "z10": z10, "month": month,
                    })
                    break  # 每级别每方向每周期首次
    return {
        "events": events, "agg": agg, "sigma15": sigma15,
        "cyc_set": cyc_set, "cyc_arr": cyc_arr, "lookbacks": lookbacks,
    }
