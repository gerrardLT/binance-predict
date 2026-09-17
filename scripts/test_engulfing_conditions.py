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

# Bar N-1
df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)
df["prev_open"] = df["open"].shift(1)
df["prev_close"] = df["close"].shift(1)

# Condition 1: Bar N-1 是个明显高点 (HM) / 明显低点 (IH)
# For HM: prev_high is the max of the prior 4-5 bars
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()

df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

# Condition 2: Bar N-1 实体是附近几根中最长的 (e.g. prior 4 bars)
df["prior_body_max_4"] = df["body"].shift(2).rolling(3).max() # bodies of N-4, N-3, N-2
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_4"]

# Condition 3: 上吊线/倒垂线被包裹住 (Inside / Harami)
# Option 1: 完全孕线 (Range inside: high[N] <= prev_high and low[N] >= prev_low)
df["is_range_inside"] = (df["high"] <= df["prev_high"]) & (df["low"] >= df["prev_low"])

# Option 2: 实体孕线 (Body inside: max(O, C) <= max(prev_O, prev_C) and min(O, C) >= min(prev_O, prev_C))
df["cur_body_top"] = df[["open", "close"]].max(axis=1)
df["cur_body_bot"] = df[["open", "close"]].min(axis=1)
df["prev_body_top"] = df[["prev_open", "prev_close"]].max(axis=1)
df["prev_body_bot"] = df[["prev_open", "prev_close"]].min(axis=1)
df["is_body_inside"] = (df["cur_body_top"] <= df["prev_body_top"]) & (df["cur_body_bot"] >= df["prev_body_bot"])

# Option 3: 不创极值 (HM: high[N] <= prev_high; IH: low[N] >= prev_low)
df["hm_not_new_high"] = df["high"] <= df["prev_high"]
df["ih_not_new_low"] = df["low"] >= df["prev_low"]

# Next bar outcomes
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

# Base Hanging Man & Inverted Hammer
base_hm = (
    (df["prev_is_green"] == True) &
    (df["prev_body_r"] >= 0.40) &
    (df["lower_r"] >= 0.50) &
    (df["body_r"] <= 0.45) &
    (df["upper_r"] <= 0.25)
)

base_ih = (
    (df["prev_is_green"] == False) &
    (df["prev_body_r"] >= 0.40) &
    (df["upper_r"] >= 0.50) &
    (df["body_r"] <= 0.45) &
    (df["lower_r"] <= 0.25)
)

now = df["dt"].max()

print("=======================================================================")
print("测试不同包裹与高低点组合条件在 30天 和 90天 的样本数与胜率")
print("=======================================================================")

configs = [
    ("组合 1【基准 + 明显高低点】", 
     df["is_prominent_high"], 
     df["is_prominent_low"]),
    
    ("组合 2【基准 + 明显高低点 + 附近最大实体】", 
     df["is_prominent_high"] & df["is_longest_body"], 
     df["is_prominent_low"] & df["is_longest_body"]),
     
    ("组合 3【基准 + 明显高低点 + 不破前柱极值(高不破高/低不破低)】", 
     df["is_prominent_high"] & df["hm_not_new_high"], 
     df["is_prominent_low"] & df["ih_not_new_low"]),
     
    ("组合 4【基准 + 明显高低点 + 实体被包裹(实体孕线)】", 
     df["is_prominent_high"] & df["is_body_inside"], 
     df["is_prominent_low"] & df["is_body_inside"]),
     
    ("组合 5【终极严格版：高低点 + 最大实体 + 实体被包裹】", 
     df["is_prominent_high"] & df["is_longest_body"] & df["is_body_inside"], 
     df["is_prominent_low"] & df["is_longest_body"] & df["is_body_inside"]),
     
    ("组合 6【极致完整包裹：高低点 + 最大实体 + 整根K线完全被包裹(Inside Bar)】", 
     df["is_prominent_high"] & df["is_longest_body"] & df["is_range_inside"], 
     df["is_prominent_low"] & df["is_longest_body"] & df["is_range_inside"]),
]

for name_cfg, hm_extra, ih_extra in configs:
    print(f"\n>>> {name_cfg}:")
    for days in [30, 90]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & base_hm & hm_extra]
        sub_ih = df[(df["dt"] >= cutoff) & base_ih & ih_extra]
        
        hm_win = sub_hm["next_is_bear"].mean() if len(sub_hm) > 0 else 0
        ih_win = sub_ih["next_is_bull"].mean() if len(sub_ih) > 0 else 0
        
        print(f"  {days}天: 上吊线 n={len(sub_hm):2d} (胜率 {hm_win*100:5.1f}%) | 倒垂线 n={len(sub_ih):2d} (胜率 {ih_win*100:5.1f}%)")
