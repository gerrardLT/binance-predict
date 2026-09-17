import pandas as pd
import numpy as np

df = pd.read_csv("output/klines_15m_720d.csv")
df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
# Focus on recent 14 days (2026-08-27 to 2026-09-09)
sub = df[df["dt"] >= "2026-08-28"].copy().reset_index(drop=True)

# Calculate candlestick features
sub["range"] = sub["high"] - sub["low"]
sub["body"] = (sub["close"] - sub["open"]).abs()
sub["body_r"] = sub["body"] / np.where(sub["range"] > 0, sub["range"], 1.0)
sub["upper_shadow"] = sub["high"] - np.maximum(sub["open"], sub["close"])
sub["lower_shadow"] = np.minimum(sub["open"], sub["close"]) - sub["low"]
sub["upper_r"] = sub["upper_shadow"] / np.where(sub["range"] > 0, sub["range"], 1.0)
sub["lower_r"] = sub["lower_shadow"] / np.where(sub["range"] > 0, sub["range"], 1.0)
sub["is_bull"] = sub["close"] > sub["open"]

# Also calculate prior 3-bar and 5-bar trends
sub["ret_3"] = (sub["close"] - sub["close"].shift(3)) / sub["close"].shift(3)
sub["ret_5"] = (sub["close"] - sub["close"].shift(5)) / sub["close"].shift(5)

# Next bar return
sub["next_is_bull"] = sub["close"].shift(-1) > sub["open"].shift(-1)
sub["next_ret"] = (sub["close"].shift(-1) - sub["open"].shift(-1)) / sub["open"].shift(-1)

print(f"Total 15m bars in last 12 days: {len(sub)}")

# Let's search for pinbars (long shadow >= 0.50 of range, small body <= 0.35)
long_lower = sub[(sub["lower_r"] >= 0.50) & (sub["body_r"] <= 0.35)].copy()
long_upper = sub[(sub["upper_r"] >= 0.50) & (sub["body_r"] <= 0.35)].copy()

print(f"\nCandidates with long lower shadow (hammer / hanging man / pinbar): {len(long_lower)}")
print(long_lower[["dt", "open", "high", "low", "close", "lower_r", "upper_r", "body_r", "ret_5", "next_is_bull"]].to_string())

print(f"\nCandidates with long upper shadow (inverted hammer / shooting star): {len(long_upper)}")
print(long_upper[["dt", "open", "high", "low", "close", "upper_r", "lower_r", "body_r", "ret_5", "next_is_bull"]].to_string())
