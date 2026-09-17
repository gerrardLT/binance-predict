import pandas as pd
import numpy as np

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

hm_v2 = (
    (df["prev_is_green"] == True) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["is_prominent_high"]) & 
    (df["is_longest_body"]) & 
    (df["is_full_inside"]) & 
    (df["lower_r"] >= 0.45) & 
    (df["body_r"] <= 0.45) & 
    (df["upper_r"] <= 0.25)
)

ih_v2 = (
    (df["prev_is_green"] == False) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["is_prominent_low"]) & 
    (df["is_longest_body"]) & 
    (df["is_full_inside"]) & 
    (df["upper_r"] >= 0.45) & 
    (df["body_r"] <= 0.45) & 
    (df["lower_r"] <= 0.25)
)

df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)

sub_hm = df[hm_v2 & (df.index >= 5) & (df.index < len(df) - 3)].copy()
sub_hm["signal_type"] = "HM_DOWN"
sub_hm["is_win"] = sub_hm["next_is_bear"]
sub_hm["adverse_excursion"] = sub_hm["next_high_spike"]

sub_ih = df[ih_v2 & (df.index >= 5) & (df.index < len(df) - 3)].copy()
sub_ih["signal_type"] = "IH_UP"
sub_ih["is_win"] = sub_ih["next_is_bull"]
sub_ih["adverse_excursion"] = sub_ih["next_low_dip"]

all_signals = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
now = all_signals["dt"].max()

FEE = 0.98
PREMIUM = 0.01

fine_grid = [
    (0.20, 0.0012),
    (0.25, 0.0007),
    (0.28, 0.0005),
    (0.30, 0.0004),
    (0.32, 0.0003),
    (0.35, 0.0002),
]

print("=" * 80)
print("FINE GRID SCAN FOR OPTIMAL LIMIT ORDER QUOTE (720 DAYS)")
print("=" * 80)

for q, req_bps in fine_grid:
    payoff = FEE / (q + PREMIUM) - 1.0
    filled = all_signals[all_signals["adverse_excursion"] >= req_bps]
    n_fill = len(filled)
    fill_rate = n_fill / len(all_signals)
    win_rate = filled["is_win"].mean()
    pnl_series = np.where(filled["is_win"], payoff, -1.0)
    tot_pnl = np.sum(pnl_series)
    ev = tot_pnl / n_fill
    
    # Streaks
    cur_loss, max_loss = 0, 0
    for w in filled["is_win"]:
        if not w:
            cur_loss += 1
            if cur_loss > max_loss: max_loss = cur_loss
        else: cur_loss = 0
        
    print(f"Quote={q:.2f} | Fill Rate={fill_rate*100:4.1f}% ({n_fill:3d}单) | Win Rate={win_rate*100:4.1f}% | EV={ev*100:+5.1f}% | Total PnL={tot_pnl:+7.1f} R | Max Loss={max_loss}连亏")
