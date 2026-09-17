import pandas as pd
import numpy as np

def extract_signals(tf="15m", use_rsi_ih=False):
    csv_path = f"output/klines_{tf}_720d.csv"
    df = pd.read_csv(csv_path)
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

    # Bar N-1
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

    # RSI(14)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / np.where(loss > 0, loss, 1.0)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Outcomes
    df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
    df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
    df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

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
    if use_rsi_ih:
        ih_cond = ih_cond & (df["rsi_14"].shift(1) <= 40)

    sub_hm = df[hm_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_hm["signal_type"] = "HM_DOWN"
    sub_hm["is_win"] = sub_hm["next_is_bear"]
    sub_hm["discount_opp"] = sub_hm["next_high_spike"] > 0.0003

    sub_ih = df[ih_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_ih["signal_type"] = "IH_UP"
    sub_ih["is_win"] = sub_ih["next_is_bull"]
    sub_ih["discount_opp"] = sub_ih["next_low_dip"] > 0.0003

    merged = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
    merged["tf"] = tf
    return merged

def analyze_drawdowns_and_streaks(pnl_series, is_win_series):
    cum_pnl = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    
    dd_duration = 0
    max_dd_duration = 0
    for dd in drawdown:
        if dd > 0:
            dd_duration += 1
            if dd_duration > max_dd_duration:
                max_dd_duration = dd_duration
        else:
            dd_duration = 0
            
    max_consecutive_losses = 0
    max_consecutive_wins = 0
    cur_losses = 0
    cur_wins = 0
    
    for win in is_win_series:
        if win:
            cur_wins += 1
            cur_losses = 0
            if cur_wins > max_consecutive_wins:
                max_consecutive_wins = cur_wins
        else:
            cur_losses += 1
            cur_wins = 0
            if cur_losses > max_consecutive_losses:
                max_consecutive_losses = cur_losses
                
    return max_dd, max_dd_duration, max_consecutive_losses, max_consecutive_wins

s_15m = extract_signals("15m", use_rsi_ih=False)
s_5m = extract_signals("5m", use_rsi_ih=True)
combined = pd.concat([s_15m, s_5m]).sort_values("dt").reset_index(drop=True)

FEE = 0.98
PREMIUM = 0.01
QUOTE = 0.20
PAYOFF = FEE / (QUOTE + PREMIUM) - 1.0 # +3.667

now = combined["dt"].max()

print("=" * 80)
print("DRAWDOWN AND CONSECUTIVE LOSS ANALYSIS (720 DAYS)")
print("=" * 80)

datasets = [
    ("COMBINED (15m + 5m)", combined),
    ("STANDALONE 15m", s_15m),
    ("STANDALONE  5m", s_5m),
]

for ds_name, ds in datasets:
    print(f"\n========================================================")
    print(f"Dataset: {ds_name}")
    print(f"========================================================")
    
    for days in [30, 90, 180, 360, 720]:
        cutoff = now - pd.Timedelta(days=days)
        sub = ds[ds["dt"] >= cutoff].copy()
        if len(sub) == 0: continue
        
        # 0.20 limit order simulation:
        sub_filled = sub[sub["discount_opp"] == True].copy().reset_index(drop=True)
        pnl_filled = np.where(sub_filled["is_win"], PAYOFF, -1.0)
        max_dd_limit, max_dd_dur_limit, max_loss_limit, max_win_limit = analyze_drawdowns_and_streaks(pnl_filled, sub_filled["is_win"].values)
        tot_pnl_limit = np.sum(pnl_filled)
        
        # Market order simulation:
        pnl_mkt = np.where(sub["is_win"], 0.96, -1.0)
        max_dd_mkt, max_dd_dur_mkt, max_loss_mkt, max_win_mkt = analyze_drawdowns_and_streaks(pnl_mkt, sub["is_win"].values)
        tot_pnl_mkt = np.sum(pnl_mkt)
        
        print(f"\n--- {days} Days ---")
        print(f"  [0.20 Limit Order (Payoff 1:4.67)]:")
        print(f"    Filled Trades     : {len(sub_filled)} trades | Net PnL = {tot_pnl_limit:+7.1f} R")
        print(f"    Max Consec Losses : {max_loss_limit} losses | Max Consec Wins = {max_win_limit} wins")
        print(f"    Max Drawdown      : {max_dd_limit:5.1f} R (Duration = {max_dd_dur_limit} trades)")
        
        print(f"  [Market Order (Payoff 1:0.96)]:")
        print(f"    Total Trades      : {len(sub)} trades | Net PnL = {tot_pnl_mkt:+7.1f} R")
        print(f"    Max Consec Losses : {max_loss_mkt} losses | Max Consec Wins = {max_win_mkt} wins")
        print(f"    Max Drawdown      : {max_dd_mkt:5.1f} R (Duration = {max_dd_dur_mkt} trades)")
