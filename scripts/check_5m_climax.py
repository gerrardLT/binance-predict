import pandas as pd
import numpy as np

# Let's inspect 5m micro-structure:
# In 5m, why is Inverted Hammer winrate lower than 15m?
# Because in 5m, crypto downtrends frequently form "micro-continuation" cascade drops!
# What filters out the fake 5m inverted hammers?
# Let's test:
# 1. Bar N-1 range vs ATR: was N-1 an exhaustion dump (range >= 1.5 ATR)?
# 2. Moving Average filter: 5m EMA(20) / EMA(50) distance
# 3. Micro-double bottom / divergence: low[N] strictly above prior low?
df = pd.read_csv("output/klines_5m_720d.csv")
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

df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)

df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# ATR(14)
tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
df["atr_14"] = tr.rolling(14).mean()
df["prev_range_atr"] = (df["prev_high"] - df["prev_low"]) / df["atr_14"].shift(1)

# Outcomes
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

now = df["dt"].max()

print("Testing 5m Inverted Hammer with Exhaustion Climax (prev_range >= 1.3 ATR):")
for days in [90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    base_ih = (
        (df["dt"] >= cutoff) &
        (df["prev_is_green"] == False) & 
        (df["prev_body_r"] >= 0.40) & 
        (df["is_prominent_low"]) & 
        (df["is_longest_body"]) & 
        (df["is_full_inside"]) & 
        (df["upper_r"] >= 0.45) & 
        (df["body_r"] <= 0.45) & 
        (df["lower_r"] <= 0.25)
    )
    climax_ih = base_ih & (df["prev_range_atr"] >= 1.3)
    
    n_base = len(df[base_ih])
    w_base = df[base_ih]["next_is_bull"].mean() if n_base > 0 else 0
    
    n_climax = len(df[climax_ih])
    w_climax = df[climax_ih]["next_is_bull"].mean() if n_climax > 0 else 0
    
    print(f"  {days:3d}d: Base IH n={n_base:3d} (win={w_base*100:.1f}%) --> Climax Dump IH n={n_climax:3d} (win={w_climax*100:.1f}%)")
