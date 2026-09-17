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

# Prior high / low
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

# Prior longest body
df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

# Inside bar / 包裹住 (高不破高，低不破低)
df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)
df["is_high_low_inside"] = (df["high"] <= df["prev_high"] * 1.0002) # 上吊线只要高不破高

# Next bar outcome
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)

now = df["dt"].max()

# Test various thresholds of "光头" (upper_r for HM) and "光脚" (lower_r for IH)
# Level 1: 原基准（允许少量上影 <= 20%）
# Level 2: 宽松光头（<= 8%）
# Level 3: 标准光头（<= 5%）
# Level 4: 极度光头（<= 2% 或绝对 0）

print("==========================================================================================")
print("【回测对比：如果上吊线必须光头、倒垂线必须光脚】")
print("==========================================================================================")

for flat_label, max_minor_r in [
    ("原基准 (允许微量次影 <= 20%)", 0.20),
    ("准光头 (次影 <= 8%)", 0.08),
    ("标准光头 (次影 <= 5%)", 0.05),
    ("绝对光头 (次影 <= 2%)", 0.02),
]:
    print(f"\n>>> 档位: {flat_label} <<<")
    
    hm_cond = (
        (df["prev_is_green"] == True) &
        (df["prev_body_r"] >= 0.40) &
        (df["is_prominent_high"]) &
        (df["is_longest_body"]) &
        (df["is_full_inside"]) &
        (df["lower_r"] >= 0.50) &
        (df["body_r"] <= 0.45) &
        (df["upper_r"] <= max_minor_r) # 光头约束
    )
    
    ih_cond = (
        (df["prev_is_green"] == False) &
        (df["prev_body_r"] >= 0.40) &
        (df["is_prominent_low"]) &
        (df["is_longest_body"]) &
        (df["is_full_inside"]) &
        (df["upper_r"] >= 0.50) &
        (df["body_r"] <= 0.45) &
        (df["lower_r"] <= max_minor_r) # 光脚约束
    )
    
    for days in [30, 90, 180]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & hm_cond]
        sub_ih = df[(df["dt"] >= cutoff) & ih_cond]
        
        hm_win = sub_hm["next_is_bear"].mean() if len(sub_hm) > 0 else 0
        ih_win = sub_ih["next_is_bull"].mean() if len(sub_ih) > 0 else 0
        
        print(f"  {days}天 | 上吊线 n={len(sub_hm):2d} (看跌胜率 {hm_win*100:5.1f}%) | 倒垂线 n={len(sub_ih):2d} (看涨胜率 {ih_win*100:5.1f}%)")
