#!/usr/bin/env python3
"""放宽光头小实体倒垂线与上吊线的参数梯度扫描（最近 180 天 5m 和 15m）。

参数维度探索：
- 短影线占比 max_minor_r: [0.05 (标准), 0.08 (微影线), 0.12 (轻微刺透), 0.15 (常规)]
- 实体占比 max_body_r: [0.25, 0.30, 0.35]
- 长影线/实体比 major_ratio: [2.0, 1.5]
- 长影线占比 min_major_r: [0.65, 0.55]
"""

import numpy as np
from datetime import datetime, timezone

def scan_relaxed_grid(tf="15m", days=180):
    csv_path = f"output/klines_{tf}_720d.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = [l.strip().split(',') for l in f if l.strip()]

    ts_list, o_list, h_list, l_list, c_list = [], [], [], [], []
    for row in lines:
        dt = datetime.fromisoformat(row[0])
        ts_list.append(int(dt.timestamp() * 1000))
        o_list.append(float(row[1]))
        h_list.append(float(row[2]))
        l_list.append(float(row[3]))
        c_list.append(float(row[4]))

    ts = np.array(ts_list)
    o = np.array(o_list)
    h = np.array(h_list)
    l = np.array(l_list)
    c = np.array(c_list)
    
    end_ms = ts[-1]
    start_ms = end_ms - days * 86400 * 1000
    idx_180 = np.where(ts >= start_ms)[0][0]
    start_eval = max(idx_180, 10)
    end_eval = len(ts) - 2

    print(f"============================================================")
    print(f"[{tf.upper()}] 最近 {days} 天形态放宽梯度扫描 (总根数: {end_eval - start_eval + 1})")
    print(f"============================================================")

    # 梯度档位设计：
    # 档位 1 (原标准型): minor <= 0.05, body <= 0.25, major_ratio >= 2.0, major_r >= 0.65
    # 档位 2 (温和放宽): minor <= 0.08, body <= 0.30, major_ratio >= 2.0, major_r >= 0.60
    # 档位 3 (中度放宽): minor <= 0.10, body <= 0.35, major_ratio >= 1.8, major_r >= 0.55
    # 档位 4 (大幅放宽): minor <= 0.15, body <= 0.35, major_ratio >= 1.5, major_r >= 0.50
    tiers = [
        ("档位1 (原标准型: 纯光头/体<=25%)", 0.05, 0.25, 2.0, 0.65),
        ("档位2 (温和放宽: 微影<=8%/体<=30%)", 0.08, 0.30, 2.0, 0.60),
        ("档位3 (中度放宽: 影<=10%/体<=35%)", 0.10, 0.35, 1.8, 0.55),
        ("档位4 (大幅放宽: 影<=15%/体<=35%/长影>=1.5体)", 0.15, 0.35, 1.5, 0.50),
    ]

    for name, max_minor, max_body, major_ratio, min_major in tiers:
        # 统计上吊线 (HM) - 上涨背景
        # 统计倒垂线 (IH) - 下跌背景
        n_hm, win_hm = 0, 0
        n_ih, win_ih = 0, 0
        
        # 同时统计不限背景的纯形态
        n_hm_all, win_hm_all = 0, 0
        n_ih_all, win_ih_all = 0, 0

        for i in range(start_eval, end_eval + 1):
            bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
            bar_range = bar_h - bar_l
            if bar_range <= 0: continue

            body = abs(bar_c - bar_o)
            body_r = body / bar_range
            body_top = max(bar_o, bar_c)
            body_bottom = min(bar_o, bar_c)
            upper_wick = bar_h - body_top
            lower_wick = body_bottom - bar_l
            
            next_c, next_o = c[i+1], o[i+1]
            next_is_up = next_c > next_o
            next_is_down = next_c < next_o

            ret3 = (c[i-1] - o[i-3]) / o[i-3]
            ma10 = np.mean(c[i-10:i])
            is_downtrend = (ret3 < -0.001) or (c[i] < ma10)
            is_uptrend = (ret3 > 0.001) or (c[i] > ma10)

            # 上吊线：长下影 + 小实体在上 + 微上影
            if (body_r <= max_body and 
                (upper_wick / bar_range) <= max_minor and 
                (lower_wick / bar_range) >= min_major and 
                lower_wick >= major_ratio * body):
                n_hm_all += 1
                if next_is_down: win_hm_all += 1
                if is_uptrend:
                    n_hm += 1
                    if next_is_down: win_hm += 1

            # 倒垂线：长上影 + 小实体在下 + 微下影
            if (body_r <= max_body and 
                (lower_wick / bar_range) <= max_minor and 
                (upper_wick / bar_range) >= min_major and 
                upper_wick >= major_ratio * body):
                n_ih_all += 1
                if next_is_up: win_ih_all += 1
                if is_downtrend:
                    n_ih += 1
                    if next_is_up: win_ih += 1

        wr_hm = (win_hm / n_hm * 100) if n_hm > 0 else 0
        wr_ih = (win_ih / n_ih * 100) if n_ih > 0 else 0
        wr_hm_all = (win_hm_all / n_hm_all * 100) if n_hm_all > 0 else 0
        wr_ih_all = (win_ih_all / n_ih_all * 100) if n_ih_all > 0 else 0

        print(f"\n>>> {name}")
        print(f"  [上吊线 押DOWN]")
        print(f"    * 上涨背景: 样本 {n_hm:>4} | 收跌 {win_hm:>4} | 胜率: {wr_hm:5.2f}%")
        print(f"    * 任意位置: 样本 {n_hm_all:>4} | 收跌 {win_hm_all:>4} | 胜率: {wr_hm_all:5.2f}%")
        print(f"  [倒垂线 押UP]")
        print(f"    * 下跌背景: 样本 {n_ih:>4} | 收阳 {win_ih:>4} | 胜率: {wr_ih:5.2f}%")
        print(f"    * 任意位置: 样本 {n_ih_all:>4} | 收阳 {win_ih_all:>4} | 胜率: {wr_ih_all:5.2f}%")

if __name__ == "__main__":
    scan_relaxed_grid("15m", 180)
    print("\n")
    scan_relaxed_grid("5m", 180)
