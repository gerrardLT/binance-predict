import pandas as pd
import numpy as np

df = pd.read_csv("output/klines_15m_720d.csv")
df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("dt").reset_index(drop=True)

df["range"] = df["high"] - df["low"]
df["body"] = (df["close"] - df["open"]).abs()
df["body_r"] = df["body"] / np.where(df["range"] > 0, df["range"], 1.0)
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
df["upper_r"] = df["upper_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
df["lower_r"] = df["lower_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
df["is_green"] = df["close"] >= df["open"]

# Bar N-1 features
df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)
df["prev_open"] = df["open"].shift(1)
df["prev_close"] = df["close"].shift(1)

# Condition A: Bar N-1 是近 5 根明显高点 / 低点
df["prior_high_max_5"] = df["high"].shift(1).rolling(5).max()
df["prior_low_min_5"] = df["low"].shift(1).rolling(5).min()

df["is_local_top"] = df["prev_high"] >= df["prior_high_max_5"]
df["is_local_bot"] = df["prev_low"] <= df["prior_low_min_5"]

# Condition B: Bar N-1 实体是附近几根中最长 (大于前 3 根实体)
df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_prev_largest_body"] = df["prev_body"] >= df["prior_body_max_3"]

# Condition C: 被包裹住
# 上吊线: 高点不破前高 (high[N] <= prev_high) 且 上吊线实体在前柱实体区间内 (或整根在 prev_high 和 prev_low 之间)
# 倒垂线: 低点不破前低 (low[N] >= prev_low) 且 倒垂线实体在前柱实体区间内
df["hm_engulfed"] = (df["high"] <= df["prev_high"] * 1.0002) # 允许贴线
df["ih_engulfed"] = (df["low"] >= df["prev_low"] * 0.9998)

# Strict full inside (高不破高，低不破低)
df["hm_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)
df["ih_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# Next bar outcome
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)

# Base shapes
hm_base = (
    (df["prev_is_green"] == True) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["lower_r"] >= 0.50) & 
    (df["body_r"] <= 0.45) & 
    (df["upper_r"] <= 0.20)
)

ih_base = (
    (df["prev_is_green"] == False) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["upper_r"] >= 0.50) & 
    (df["body_r"] <= 0.45) & 
    (df["lower_r"] <= 0.20)
)

now = df["dt"].max()

print("Testing strictness of包裹住:")
for label, hm_wrap, ih_wrap in [
    ("极值包裹 (高不创新高 / 低不创新低)", df["hm_engulfed"], df["ih_engulfed"]),
    ("完全孕线包裹 (整根K线都在前柱范围内)", df["hm_full_inside"], df["ih_full_inside"]),
]:
    print(f"\n--- {label} ---")
    for days in [30, 90]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & hm_base & df["is_local_top"] & df["is_prev_largest_body"] & hm_wrap]
        sub_ih = df[(df["dt"] >= cutoff) & ih_base & df["is_local_bot"] & df["is_prev_largest_body"] & ih_wrap]
        
        hm_win = sub_hm["next_is_bear"].mean() if len(sub_hm) > 0 else 0
        ih_win = sub_ih["next_is_bull"].mean() if len(sub_ih) > 0 else 0
        
        print(f"  {days}天: 上吊线 n={len(sub_hm)} (胜率 {hm_win*100:.1f}%) | 倒垂线 n={len(sub_ih)} (胜率 {ih_win*100:.1f}%)")
