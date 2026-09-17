import pandas as pd
import numpy as np

# Let's inspect 180 days performance of Ver 1, Ver 2, Ver 4 to see long-term stability
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

df["green_count_5"] = df["is_green"].shift(1).rolling(5).sum()
df["ret_5"] = (df["close"].shift(1) - df["open"].shift(5)) / df["open"].shift(5)

df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

now = df["dt"].max()
cutoff_180 = now - pd.Timedelta(days=180)

# Ver 1 180d
v1_hm = df[(df["dt"] >= cutoff_180) & (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["green_count_5"] >= 3) & (df["ret_5"] > 0) & (df["lower_r"] >= 0.50) & (df["body_r"] <= 0.40) & (df["upper_r"] <= 0.20)]
v1_ih = df[(df["dt"] >= cutoff_180) & (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["green_count_5"] <= 2) & (df["ret_5"] < 0) & (df["upper_r"] >= 0.50) & (df["body_r"] <= 0.40) & (df["lower_r"] <= 0.20)]

# Ver 2 180d
v2_hm = df[(df["dt"] >= cutoff_180) & (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_high"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["lower_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["upper_r"] <= 0.25)]
v2_ih = df[(df["dt"] >= cutoff_180) & (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_low"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["upper_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["lower_r"] <= 0.25)]

print(f"180d Ver 1: Total n={len(v1_hm)+len(v1_ih)}, Win rate={(v1_hm['next_is_bear'].sum()+v1_ih['next_is_bull'].sum())/(len(v1_hm)+len(v1_ih))*100:.1f}% | HM n={len(v1_hm)} (win={v1_hm['next_is_bear'].mean()*100:.1f}%), IH n={len(v1_ih)} (win={v1_ih['next_is_bull'].mean()*100:.1f}%)")
print(f"180d Ver 2: Total n={len(v2_hm)+len(v2_ih)}, Win rate={(v2_hm['next_is_bear'].sum()+v2_ih['next_is_bull'].sum())/(len(v2_hm)+len(v2_ih))*100:.1f}% | HM n={len(v2_hm)} (win={v2_hm['next_is_bear'].mean()*100:.1f}%), IH n={len(v2_ih)} (win={v2_ih['next_is_bull'].mean()*100:.1f}%)")
