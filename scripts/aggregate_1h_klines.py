import pandas as pd
import numpy as np

# Load 15m data and aggregate into 1h klines
df_15m = pd.read_csv("output/klines_15m_720d.csv")
df_15m["dt"] = pd.to_datetime(df_15m["timestamp"], utc=True)
df_15m = df_15m.sort_values("dt").reset_index(drop=True)

# Group by 1-hour bucket
df_15m["dt_1h"] = df_15m["dt"].dt.floor("1h")

# Ensure each 1h bucket has 4 complete 15m candles
counts = df_15m.groupby("dt_1h")["close"].count()
complete_1h = counts[counts == 4].index

df_filtered = df_15m[df_15m["dt_1h"].isin(complete_1h)]

df_1h = df_filtered.groupby("dt_1h").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum")
).reset_index()

df_1h.rename(columns={"dt_1h": "timestamp"}, inplace=True)
df_1h["dt"] = pd.to_datetime(df_1h["timestamp"], utc=True)
df_1h = df_1h.sort_values("dt").reset_index(drop=True)

# Save to output/klines_1h_720d.csv
df_1h.to_csv("output/klines_1h_720d.csv", index=False)
print(f"Aggregated {len(df_1h)} complete 1h bars from {df_1h.iloc[0]['dt']} to {df_1h.iloc[-1]['dt']}")
