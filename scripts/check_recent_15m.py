import pandas as pd
from datetime import datetime, timezone

df = pd.read_csv("output/klines_15m_720d.csv")
df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
recent = df[df["dt"] >= "2026-09-01"].copy()
print(f"Total bars since 2026-09-01: {len(recent)}")
print(recent[["dt", "open", "high", "low", "close", "volume"]].tail(10).to_string())
