import pandas as pd
import numpy as np

# Let's inspect the candles in the 4 user crops to see what "包裹住" (engulfed/inside)
# and "明显高点/低点" and "附近最大实体" look like in the user's actual examples!

# Load 15m klines
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

# Let's test the 3 new conditions:
# Condition 1: Bar N-1 is a local high/low (e.g. highest in last 5 bars for HM, lowest for IH)
df["high_max_5"] = df["high"].rolling(5).max()
df["low_min_5"] = df["low"].rolling(5).min()

# Condition 2: Bar N-1 body is the largest among past 4-5 bars
df["body_max_5"] = df["body"].rolling(5).max()

# Condition 3: "包裹住" (Engulfed / Inside):
# Option A: Bar N high <= Bar N-1 high and Bar N low >= Bar N-1 low (Strict Inside Bar)
# Option B: Bar N body is inside Bar N-1 body (Harami / 孕线)
# Option C: Bar N doesn't make a new high (for HM: high[N] <= high[N-1]) or new low (for IH: low[N] >= low[N-1])
# Option D: Bar N range is mostly inside Bar N-1 (e.g. 80%+ overlap)

print("Script template ready to test.")
