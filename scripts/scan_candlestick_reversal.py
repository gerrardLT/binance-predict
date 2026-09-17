#!/usr/bin/env python3
"""回测最近 180 天 5m 与 15m BTC K 线中「非常标准的光头小实体倒垂线与上吊线」的统计表现。

形态严格定义（纯蜡烛图经典几何规范，不掺杂任何复杂门禁）：
1. 实体占比 body_r = abs(close - open) / range <= 0.25（小实体）
2. 光头/光脚判定：
   - 倒垂线（Inverted Hammer）：
     * 长上影：upper_wick / range >= 0.65 且 upper_wick >= 2.0 * body
     * 无下影（光脚/平底）：lower_wick / range <= 0.05（严格光脚）
     * 出现背景：下跌过程中（前 3 根累积跌幅 < 0，或跌破 MA10）
     * 预期：反转看涨（次根收阳 close > open 押 UP）
   - 上吊线（Hanging Man）：
     * 长下影：lower_wick / range >= 0.65 且 lower_wick >= 2.0 * body
     * 无上影（光头/平顶）：upper_wick / range <= 0.05（严格光头）
     * 出现背景：上升过程中（前 3 根累积涨幅 > 0，或站上 MA10）
     * 预期：反转看跌（次根收阴 close < open 押 DOWN）
"""

import sys
import numpy as np
from datetime import datetime, timezone

def run_scan(tf="5m", days=180):
    csv_path = f"output/klines_{tf}_720d.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = [l.strip().split(',') for l in f if l.strip()]

    # 解析数据
    ts_list, o_list, h_list, l_list, c_list, v_list = [], [], [], [], [], []
    for row in lines:
        dt = datetime.fromisoformat(row[0])
        ts_list.append(int(dt.timestamp() * 1000))
        o_list.append(float(row[1]))
        h_list.append(float(row[2]))
        l_list.append(float(row[3]))
        c_list.append(float(row[4]))
        v_list.append(float(row[5]))

    ts = np.array(ts_list)
    o = np.array(o_list)
    h = np.array(h_list)
    l = np.array(l_list)
    c = np.array(c_list)
    
    # 截取最近 180 天
    end_ms = ts[-1]
    start_ms = end_ms - days * 86400 * 1000
    idx_180 = np.where(ts >= start_ms)[0]
    start_idx = idx_180[0]
    
    # 确保前置窗口足够（需要前 10 根算背景）
    start_eval = max(start_idx, 10)
    end_eval = len(ts) - 2 # 留一根看次根 outcome
    
    print(f"=== [{tf.upper()}] 最近 {days} 天回测扫描 ===")
    print(f"时间范围: {datetime.fromtimestamp(ts[start_eval]/1000, tz=timezone.utc).isoformat()} -> {datetime.fromtimestamp(ts[end_eval]/1000, tz=timezone.utc).isoformat()}")
    print(f"总根数: {end_eval - start_eval + 1}")

    # 两种严格度梯度扫描：
    # 宽松度 A: lower_wick/range <= 0.08, body/range <= 0.30, upper_wick >= 2*body & >= 0.60
    # 极严格 B: lower_wick/range <= 0.03 (几乎绝对光头), body/range <= 0.20, upper_wick >= 2.5*body & >= 0.70
    
    results = {}
    
    for mode_name, (max_minor_wick, max_body, min_major_wick_r, major_body_ratio) in [
        ("标准光头小实体 (标准型)", (0.05, 0.25, 0.65, 2.0)),
        ("极度光头极小实体 (严苛型)", (0.02, 0.18, 0.75, 3.0)),
    ]:
        # 1. 倒垂线 (Inverted Hammer) -> 预期次根反转 UP
        # 2. 上吊线 (Hanging Man) -> 预期次根反转 DOWN
        ih_signals_all = []
        ih_signals_downtrend = []
        
        hm_signals_all = []
        hm_signals_uptrend = []
        
        for i in range(start_eval, end_eval + 1):
            bar_o, bar_h, bar_l, bar_c = o[i], h[i], l[i], c[i]
            bar_range = bar_h - bar_l
            if bar_range <= 0:
                continue
            
            body = abs(bar_c - bar_o)
            body_r = body / bar_range
            body_top = max(bar_o, bar_c)
            body_bottom = min(bar_o, bar_c)
            upper_wick = bar_h - body_top
            lower_wick = body_bottom - bar_l
            
            next_c = c[i+1]
            next_o = o[i+1]
            next_ret = (next_c - next_o) / next_o
            next_is_up = next_c > next_o
            next_is_down = next_c < next_o
            
            # 背景趋势：
            # 过去 3 根涨跌幅
            ret3 = (c[i-1] - o[i-3]) / o[i-3]
            # 距 MA10
            ma10 = np.mean(c[i-10:i])
            is_downtrend = (ret3 < -0.001) or (c[i] < ma10)
            is_uptrend = (ret3 > 0.001) or (c[i] > ma10)
            
            # --- 倒垂线判断 ---
            # 长上影 + 小实体在下 + 几乎无下影 (光脚)
            if (body_r <= max_body and 
                (lower_wick / bar_range) <= max_minor_wick and 
                (upper_wick / bar_range) >= min_major_wick_r and 
                upper_wick >= major_body_ratio * body):
                
                sig = {
                    "ts": ts[i], "dt": datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).isoformat(),
                    "o": bar_o, "h": bar_h, "l": bar_l, "c": bar_c,
                    "range": bar_range, "body_r": body_r,
                    "upper_r": upper_wick / bar_range, "lower_r": lower_wick / bar_range,
                    "is_up": next_is_up, "next_ret": next_ret
                }
                ih_signals_all.append(sig)
                if is_downtrend:
                    ih_signals_downtrend.append(sig)

            # --- 上吊线判断 ---
            # 长下影 + 小实体在上 + 几乎无上影 (光头)
            if (body_r <= max_body and 
                (upper_wick / bar_range) <= max_minor_wick and 
                (lower_wick / bar_range) >= min_major_wick_r and 
                lower_wick >= major_body_ratio * body):
                
                sig = {
                    "ts": ts[i], "dt": datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).isoformat(),
                    "o": bar_o, "h": bar_h, "l": bar_l, "c": bar_c,
                    "range": bar_range, "body_r": body_r,
                    "upper_r": upper_wick / bar_range, "lower_r": lower_wick / bar_range,
                    "is_down": next_is_down, "next_ret": next_ret
                }
                hm_signals_all.append(sig)
                if is_uptrend:
                    hm_signals_uptrend.append(sig)

        print(f"\n--- {mode_name} ---")
        
        # 统计倒垂线
        n_ih_all = len(ih_signals_all)
        win_ih_all = sum(1 for s in ih_signals_all if s["is_up"])
        wr_ih_all = win_ih_all / n_ih_all if n_ih_all > 0 else 0
        
        n_ih_trend = len(ih_signals_downtrend)
        win_ih_trend = sum(1 for s in ih_signals_downtrend if s["is_up"])
        wr_ih_trend = win_ih_trend / n_ih_trend if n_ih_trend > 0 else 0
        
        print(f"【倒垂线 (看涨反转 UP)】")
        print(f"  * 任意位置: 触发 {n_ih_all:>4} 次 | 次根收阳 {win_ih_all:>4} 次 | 胜率: {wr_ih_all*100:5.2f}%")
        print(f"  * 下跌背景: 触发 {n_ih_trend:>4} 次 | 次根收阳 {win_ih_trend:>4} 次 | 胜率: {wr_ih_trend*100:5.2f}%")

        # 统计上吊线
        n_hm_all = len(hm_signals_all)
        win_hm_all = sum(1 for s in hm_signals_all if s["is_down"])
        wr_hm_all = win_hm_all / n_hm_all if n_hm_all > 0 else 0
        
        n_hm_trend = len(hm_signals_uptrend)
        win_hm_trend = sum(1 for s in hm_signals_uptrend if s["is_down"])
        wr_hm_trend = win_hm_trend / n_hm_trend if n_hm_trend > 0 else 0
        
        print(f"【上吊线 (看跌反转 DOWN)】")
        print(f"  * 任意位置: 触发 {n_hm_all:>4} 次 | 次根收跌 {win_hm_all:>4} 次 | 胜率: {wr_hm_all*100:5.2f}%")
        print(f"  * 上涨背景: 触发 {n_hm_trend:>4} 次 | 次根收跌 {win_hm_trend:>4} 次 | 胜率: {wr_hm_trend*100:5.2f}%")

if __name__ == "__main__":
    run_scan("5m", 180)
    print("\n" + "="*60 + "\n")
    run_scan("15m", 180)
