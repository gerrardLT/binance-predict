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

# Prior high/low
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

# Prior longest body
df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

# Wrap condition:
# Case 1: High doesn't break prev high (for HM) / Low doesn't break prev low (for IH)
df["hm_no_break"] = df["high"] <= df["prev_high"] * 1.0002
df["ih_no_break"] = df["low"] >= df["prev_low"] * 0.9998

# Case 2: Full inside (both high and low inside)
df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# Outcome
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

now = df["dt"].max()

print("Testing with Shaved Head (光头):")
# For HM: Shaved head means upper_r <= 0.05 (no upper shadow)
# For IH: Shaved flat base means lower_r <= 0.05 (no lower shadow)

for wrap_name, hm_wrap_cond, ih_wrap_cond in [
    ("高低点极值包裹 (高不破前高/低不破前低)", df["hm_no_break"], df["ih_no_break"]),
    ("完全孕线包裹 (整根完全在前柱内)", df["is_full_inside"], df["is_full_inside"]),
    ("不要求包裹 (仅前置高低点+最长大实体)", True, True),
]:
    print(f"\n==================== {wrap_name} ====================")
    for minor_thresh, minor_name in [(0.05, "标准光头 <=5%"), (0.08, "准光头 <=8%"), (0.12, "轻微影线 <=12%")]:
        hm_c = (
            (df["prev_is_green"] == True) &
            (df["prev_body_r"] >= 0.40) &
            (df["is_prominent_high"]) &
            (df["is_longest_body"]) &
            hm_wrap_cond &
            (df["lower_r"] >= 0.50) &
            (df["body_r"] <= 0.45) &
            (df["upper_r"] <= minor_thresh)
        )
        ih_c = (
            (df["prev_is_green"] == False) &
            (df["prev_body_r"] >= 0.40) &
            (df["is_prominent_low"]) &
            (df["is_longest_body"]) &
            ih_wrap_cond &
            (df["upper_r"] >= 0.50) &
            (df["body_r"] <= 0.45) &
            (df["lower_r"] <= minor_thresh)
        )
        
        for days in [30, 90]:
            cutoff = now - pd.Timedelta(days=days)
            n_hm = len(df[(df["dt"] >= cutoff) & hm_c])
            w_hm = df[(df["dt"] >= cutoff) & hm_c]["next_is_bear"].mean() if n_hm > 0 else 0
            
            n_ih = len(df[(df["dt"] >= cutoff) & ih_c])
            w_ih = df[(df["dt"] >= cutoff) & ih_c]["next_is_bull"].mean() if n_ih > 0 else 0
            
            print(f"[{minor_name}] {days}d: HM n={n_hm:2d} (胜率 {w_hm*100:5.1f}%) | IH n={n_ih:2d} (胜率 {w_ih*100:5.1f}%)")
