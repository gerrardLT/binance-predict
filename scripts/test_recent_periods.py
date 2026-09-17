#!/usr/bin/env python3
"""严谨测试最近 30 天与 90 天 5m 和 15m K 线中的倒垂线与上吊线表现。

支持梯度：
- 严格型 (档位1): minor <= 0.05, body <= 0.25, major_ratio >= 2.0, major_r >= 0.65
- 放宽型 (档位2): minor <= 0.08, body <= 0.30, major_ratio >= 2.0, major_r >= 0.60
- 大幅放宽 (档位4): minor <= 0.15, body <= 0.35, major_ratio >= 1.5, major_r >= 0.50
"""

import sys
import numpy as np
from datetime import datetime, timezone

def run_test_period(days=30):
    for tf in ["15m", "5m"]:
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
        idx_sub = np.where(ts >= start_ms)[0][0]
        start_eval = max(idx_sub, 10)
        end_eval = len(ts) - 2

        start_dt = datetime.fromtimestamp(ts[start_eval]/1000, tz=timezone.utc).isoformat()
        end_dt = datetime.fromtimestamp(ts[end_eval]/1000, tz=timezone.utc).isoformat()

        print(f"\n================================================================================")
        print(f"【{tf.upper()} 周期】 最近 {days:>2} 天回测 ({start_dt[:10]} 至 {end_dt[:10]} | 共 {end_eval - start_eval + 1} 根 K线)")
        print(f"================================================================================")

        tiers = [
            ("严格光头小实体 (标准: 影<=5%, 体<=25%, 影/体>=2)", 0.05, 0.25, 2.0, 0.65),
            ("放宽光头小实体 (放宽: 影<=8%, 体<=30%, 影/体>=2)", 0.08, 0.30, 2.0, 0.60),
            ("大幅放宽实体 (大宽: 影<=15%, 体<=35%, 影/体>=1.5)", 0.15, 0.35, 1.5, 0.50),
        ]

        for name, max_minor, max_body, major_ratio, min_major in tiers:
            n_hm_trend, win_hm_trend = 0, 0
            n_hm_all, win_hm_all = 0, 0

            n_ih_trend, win_ih_trend = 0, 0
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

                # 背景趋势判据
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
                        n_hm_trend += 1
                        if next_is_down: win_hm_trend += 1

                # 倒垂线：长上影 + 小实体在下 + 微下影
                if (body_r <= max_body and 
                    (lower_wick / bar_range) <= max_minor and 
                    (upper_wick / bar_range) >= min_major and 
                    upper_wick >= major_ratio * body):
                    n_ih_all += 1
                    if next_is_up: win_ih_all += 1
                    if is_downtrend:
                        n_ih_trend += 1
                        if next_is_up: win_ih_trend += 1

            wr_hm_trend = (win_hm_trend / n_hm_trend * 100) if n_hm_trend > 0 else 0
            wr_hm_all = (win_hm_all / n_hm_all * 100) if n_hm_all > 0 else 0
            wr_ih_trend = (win_ih_trend / n_ih_trend * 100) if n_ih_trend > 0 else 0
            wr_ih_all = (win_ih_all / n_ih_all * 100) if n_ih_all > 0 else 0

            print(f">>> {name}")
            print(f"    【上吊线 押DOWN】")
            print(f"       * 上涨背景: 样本 {n_hm_trend:>3} | 收跌 {win_hm_trend:>3} | 胜率: {wr_hm_trend:5.2f}%")
            print(f"       * 任意位置: 样本 {n_hm_all:>3} | 收跌 {win_hm_all:>3} | 胜率: {wr_hm_all:5.2f}%")
            print(f"    【倒垂线 押UP】")
            print(f"       * 下跌背景: 样本 {n_ih_trend:>3} | 收阳 {win_ih_trend:>3} | 胜率: {wr_ih_trend:5.2f}%")
            print(f"       * 任意位置: 样本 {n_ih_all:>3} | 收阳 {win_ih_all:>3} | 胜率: {wr_ih_all:5.2f}%")

if __name__ == "__main__":
    print("################################################################################")
    print("## 1. 最近 30 天样本实测 (2026-08-10 ~ 2026-09-09)")
    print("################################################################################")
    run_test_period(30)

    print("\n\n################################################################################")
    print("## 2. 最近 90 天样本实测 (2026-06-11 ~ 2026-09-09)")
    print("################################################################################")
    run_test_period(90)
