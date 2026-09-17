import pandas as pd
import numpy as np

# Let's inspect the cross-timeframe effect:
# When 1h triggers Ver 2, what does the NEXT 15M bar do?
# In prediction markets, 1h signals can be used to trade the next 15m window!
df_15m = pd.read_csv("output/klines_15m_720d.csv")
df_15m["dt"] = pd.to_datetime(df_15m["timestamp"], utc=True)
df_15m = df_15m.sort_values("dt").reset_index(drop=True)

df_1h = pd.read_csv("output/klines_1h_720d.csv")
df_1h["dt"] = pd.to_datetime(df_1h["timestamp"], utc=True)
df_1h = df_1h.sort_values("dt").reset_index(drop=True)

df_1h["range"] = df_1h["high"] - df_1h["low"]
df_1h["body"] = (df_1h["close"] - df_1h["open"]).abs()
df_1h["body_r"] = df_1h["body"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["upper_wick"] = df_1h["high"] - df_1h[["open", "close"]].max(axis=1)
df_1h["lower_wick"] = df_1h[["open", "close"]].min(axis=1) - df_1h["low"]
df_1h["upper_r"] = df_1h["upper_wick"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["lower_r"] = df_1h["lower_wick"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["is_green"] = df_1h["close"] >= df_1h["open"]

df_1h["prev_is_green"] = df_1h["is_green"].shift(1)
df_1h["prev_body"] = df_1h["body"].shift(1)
df_1h["prev_body_r"] = df_1h["body_r"].shift(1)
df_1h["prev_high"] = df_1h["high"].shift(1)
df_1h["prev_low"] = df_1h["low"].shift(1)

df_1h["prior_high_max_4"] = df_1h["high"].shift(1).rolling(4).max()
df_1h["prior_low_min_4"] = df_1h["low"].shift(1).rolling(4).min()
df_1h["is_prominent_high"] = df_1h["prev_high"] >= df_1h["prior_high_max_4"]
df_1h["is_prominent_low"] = df_1h["prev_low"] <= df_1h["prior_low_min_4"]

df_1h["prior_body_max_3"] = df_1h["body"].shift(2).rolling(3).max()
df_1h["is_longest_body"] = df_1h["prev_body"] >= df_1h["prior_body_max_3"]

df_1h["is_full_inside"] = (df_1h["high"] <= df_1h["prev_high"] * 1.0002) & (df_1h["low"] >= df_1h["prev_low"] * 0.9998)

# 1H Ver 2
hm_1h = (df_1h["prev_is_green"] == True) & (df_1h["prev_body_r"] >= 0.40) & (df_1h["is_prominent_high"]) & (df_1h["is_longest_body"]) & (df_1h["is_full_inside"]) & (df_1h["lower_r"] >= 0.45) & (df_1h["body_r"] <= 0.45) & (df_1h["upper_r"] <= 0.25)
ih_1h = (df_1h["prev_is_green"] == False) & (df_1h["prev_body_r"] >= 0.40) & (df_1h["is_prominent_low"]) & (df_1h["is_longest_body"]) & (df_1h["is_full_inside"]) & (df_1h["upper_r"] >= 0.45) & (df_1h["body_r"] <= 0.45) & (df_1h["lower_r"] <= 0.25)

now = df_1h["dt"].max()
cutoff_180 = now - pd.Timedelta(days=180)

# Match each 1h signal with the immediate next 15m bar!
hm_signals = df_1h[(df_1h["dt"] >= cutoff_180) & hm_1h]
ih_signals = df_1h[(df_1h["dt"] >= cutoff_180) & ih_1h]

hm_15m_wins = []
for idx, row in hm_signals.iterrows():
    sig_end = row["dt"] + pd.Timedelta(hours=1)
    next_15m = df_15m[df_15m["dt"] == sig_end]
    if len(next_15m) > 0:
        bar15 = next_15m.iloc[0]
        hm_15m_wins.append(bar15["close"] < bar15["open"])

ih_15m_wins = []
for idx, row in ih_signals.iterrows():
    sig_end = row["dt"] + pd.Timedelta(hours=1)
    next_15m = df_15m[df_15m["dt"] == sig_end]
    if len(next_15m) > 0:
        bar15 = next_15m.iloc[0]
        ih_15m_wins.append(bar15["close"] > bar15["open"])

print(f"1H signal -> Next 15m Bar outcome (180d):")
print(f"  Hanging Man (DOWN on next 15m): n={len(hm_15m_wins)}, win={np.mean(hm_15m_wins)*100:.1f}%")
print(f"  Inverted Hammer (UP on next 15m): n={len(ih_15m_wins)}, win={np.mean(ih_15m_wins)*100:.1f}%")
