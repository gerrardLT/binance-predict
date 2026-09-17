import pandas as pd
import numpy as np

def get_filled_stats(tf="15m"):
    df = pd.read_csv(f"output/klines_{tf}_720d.csv")
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

    hm_cond = (
        (df["prev_is_green"] == True) & 
        (df["prev_body_r"] >= 0.40) & 
        (df["is_prominent_high"]) & 
        (df["is_longest_body"]) & 
        (df["is_full_inside"]) & 
        (df["lower_r"] >= 0.45) & 
        (df["body_r"] <= 0.45) & 
        (df["upper_r"] <= 0.25)
    )

    ih_cond = (
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

    sub_hm = df[hm_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_hm["signal_type"] = "HM_DOWN"
    sub_hm["is_win"] = sub_hm["next_is_bear"]
    sub_hm["adverse_excursion"] = sub_hm["next_high_spike"]

    sub_ih = df[ih_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_ih["signal_type"] = "IH_UP"
    sub_ih["is_win"] = sub_ih["next_is_bull"]
    sub_ih["adverse_excursion"] = sub_ih["next_low_dip"]

    all_signals = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
    return all_signals

sig_15m = get_filled_stats("15m")
now = sig_15m["dt"].max()

print("=" * 80)
print("15M Ver2: Signal Count vs Real Order Filled Count")
print("=" * 80)

for days in [30, 90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    sub = sig_15m[sig_15m["dt"] >= cutoff].copy()
    n_sig = len(sub)
    n_hm_sig = len(sub[sub["signal_type"] == "HM_DOWN"])
    n_ih_sig = len(sub[sub["signal_type"] == "IH_UP"])
    
    fill_030 = sub[sub["adverse_excursion"] >= 0.0004]
    n_fill_030 = len(fill_030)
    hm_fill_030 = len(fill_030[fill_030["signal_type"] == "HM_DOWN"])
    ih_fill_030 = len(fill_030[fill_030["signal_type"] == "IH_UP"])
    
    fill_020 = sub[sub["adverse_excursion"] >= 0.0012]
    n_fill_020 = len(fill_020)
    hm_fill_020 = len(fill_020[fill_020["signal_type"] == "HM_DOWN"])
    ih_fill_020 = len(fill_020[fill_020["signal_type"] == "IH_UP"])
    
    print(f"\n--- {days} Days ({cutoff.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}) ---")
    print(f"  Signal Printed      : {n_sig:4d} signals (Daily avg: {n_sig/days:4.2f} / day | HM={n_hm_sig}, IH={n_ih_sig})")
    print(f"  0.30 Real Filled    : {n_fill_030:4d} orders  (Daily avg: {n_fill_030/days:4.2f} / day | Every {days/n_fill_030:3.1f} days 1 order)")
    print(f"     Fill Rate        : {n_fill_030/n_sig*100:4.1f}% (HM filled: {hm_fill_030:3d}, IH filled: {ih_fill_030:3d})")
    print(f"  0.20 Real Filled    : {n_fill_020:4d} orders  (Daily avg: {n_fill_020/days:4.2f} / day | Every {days/n_fill_020:3.1f} days 1 order)")
    print(f"     Fill Rate        : {n_fill_020/n_sig*100:4.1f}% (HM filled: {hm_fill_020:3d}, IH filled: {ih_fill_020:3d})")
