# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def extract_case_details():
    cases = [
        ("15m", "2025-10-10T21:15:00+00:00", "15M-720D/360D 历史第一多头探底神针 (102000大反转)"),
        ("15m", "2025-03-07T00:00:00+00:00", "15M-720D 历史第一空头诱多射击之星 (91283冲顶暴跌)"),
        ("15m", "2026-02-08T23:00:00+00:00", "15M-360D 近一年最强空头诱多流星 (72271假突破)"),
        ("5m", "2026-01-31T18:40:00+00:00", "5M-360D 极端探底金针 (下影占比94.4%)"),
        ("5m", "2025-08-24T19:35:00+00:00", "5M-720D 惊天大波动探底 (单根5m振幅3215刀)"),
        ("5m", "2025-03-07T00:10:00+00:00", "5M-720D 极端冲高受阻 (5m拉升砸盘暴跌2.18%)")
    ]
    
    for tf, ts, desc in cases:
        df = pd.read_csv(f"output/klines_{tf}_720d.csv")
        idx = df[df["timestamp"] == ts].index
        if len(idx) == 0:
            print(f"Not found: {ts}")
            continue
        i = idx[0]
        sub = df.iloc[max(0, i-3):min(len(df), i+4)].copy()
        sub["bar_rel"] = [f"N{k-i:+d}" if k != i else "★ SIGNAL" for k in sub.index]
        print(f"\n========================================================")
        print(f"【{desc}】({tf})")
        print(f"========================================================")
        for _, r in sub.iterrows():
            candle_type = "阳" if r["close"] >= r["open"] else "阴"
            rng = r["high"] - r["low"]
            body = abs(r["close"] - r["open"])
            up_w = r["high"] - max(r["open"], r["close"])
            dn_w = min(r["open"], r["close"]) - r["low"]
            ret_str = f"{(r['close']-r['open'])/r['open']*100:+.2f}%"
            print(f"{r['bar_rel']:>8} | {r['timestamp']} | [{candle_type}] O:{r['open']:8.1f} H:{r['high']:8.1f} L:{r['low']:8.1f} C:{r['close']:8.1f} | 涨跌:{ret_str:>6} 振幅:{rng:6.1f} (上影:{up_w:6.1f} 下影:{dn_w:6.1f}) Vol:{r['volume']:8.2f}")

if __name__ == "__main__":
    extract_case_details()
