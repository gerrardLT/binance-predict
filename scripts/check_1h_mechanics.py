import pandas as pd
import numpy as np

# Let's inspect why 1h has lower next-bar reversal winrate
# In 1h klines:
# 1. Next bar is 60 minutes long!
# 2. What about the next 15m bar right after the 1h pattern?
# Let's check: does 1h pattern trigger a quick 15m reversal, but by the end of 1h the trend continues?
df_15m = pd.read_csv("output/klines_15m_720d.csv")
df_15m["dt"] = pd.to_datetime(df_15m["timestamp"], utc=True)
df_1h = pd.read_csv("output/klines_1h_720d.csv")
df_1h["dt"] = pd.to_datetime(df_1h["timestamp"], utc=True)

# Also let's check relaxed 1h definitions (Ver 1 on 1h)
df_1h["range"] = df_1h["high"] - df_1h["low"]
df_1h["body"] = (df_1h["close"] - df_1h["open"]).abs()
df_1h["body_r"] = df_1h["body"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["upper_wick"] = df_1h["high"] - df_1h[["open", "close"]].max(axis=1)
df_1h["lower_wick"] = df_1h[["open", "close"]].min(axis=1) - df_1h["low"]
df_1h["upper_r"] = df_1h["upper_wick"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["lower_r"] = df_1h["lower_wick"] / np.where(df_1h["range"] > 0, df_1h["range"], 1.0)
df_1h["is_green"] = df_1h["close"] >= df_1h["open"]

df_1h["prev_is_green"] = df_1h["is_green"].shift(1)
df_1h["prev_body_r"] = df_1h["body_r"].shift(1)
df_1h["green_count_5"] = df_1h["is_green"].shift(1).rolling(5).sum()
df_1h["ret_5"] = (df_1h["close"].shift(1) - df_1h["open"].shift(5)) / df_1h["open"].shift(5)

# Ver 1 conditions on 1h
hm_v1 = (df_1h["prev_is_green"] == True) & (df_1h["prev_body_r"] >= 0.40) & (df_1h["green_count_5"] >= 3) & (df_1h["ret_5"] > 0) & (df_1h["lower_r"] >= 0.50) & (df_1h["body_r"] <= 0.40) & (df_1h["upper_r"] <= 0.20)
ih_v1 = (df_1h["prev_is_green"] == False) & (df_1h["prev_body_r"] >= 0.40) & (df_1h["green_count_5"] <= 2) & (df_1h["ret_5"] < 0) & (df_1h["upper_r"] >= 0.50) & (df_1h["body_r"] <= 0.40) & (df_1h["lower_r"] <= 0.20)

df_1h["next_is_bull"] = df_1h["close"].shift(-1) > df_1h["open"].shift(-1)
df_1h["next_is_bear"] = df_1h["close"].shift(-1) < df_1h["open"].shift(-1)

now = df_1h["dt"].max()
for days in [90, 180, 360, 720]:
    c = now - pd.Timedelta(days=days)
    s_hm = df_1h[(df_1h["dt"] >= c) & hm_v1]
    s_ih = df_1h[(df_1h["dt"] >= c) & ih_v1]
    w_hm = s_hm["next_is_bear"].mean() if len(s_hm) > 0 else 0
    w_ih = s_ih["next_is_bull"].mean() if len(s_ih) > 0 else 0
    print(f"1H Ver 1 | {days}d: HM n={len(s_hm)} win={w_hm*100:.1f}% | IH n={len(s_ih)} win={w_ih*100:.1f}%")
