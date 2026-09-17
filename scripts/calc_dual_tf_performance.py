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

    # Next bar outcome & excursions
    df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
    df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
    df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
    df["settle_time"] = df["dt"] + pd.Timedelta(minutes=15 if tf=="15m" else 5) * 2

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
    sub_hm["adverse_excursion"] = sub_hm["next_high_spike"]

    sub_ih = df[ih_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_ih["signal_type"] = "IH_UP"
    sub_ih["is_win"] = sub_ih["next_is_bull"]
    sub_ih["adverse_excursion"] = sub_ih["next_low_dip"]

    merged = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
    merged["tf"] = tf
    return merged

def calc_curve_stats(pnl_series, is_win_series, dt_series):
    if len(pnl_series) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    cum_pnl = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = np.max(drawdown)
    
    # Drawdown duration in trades
    cur_dur = 0
    max_dur = 0
    for dd in drawdown:
        if dd > 0:
            cur_dur += 1
            if cur_dur > max_dur: max_dur = cur_dur
        else:
            cur_dur = 0
            
    # Consecutive streaks
    max_losses = 0
    max_wins = 0
    cur_l = 0
    cur_w = 0
    for w in is_win_series:
        if w:
            cur_w += 1
            cur_l = 0
            if cur_w > max_wins: max_wins = cur_w
        else:
            cur_l += 1
            cur_w = 0
            if cur_l > max_losses: max_losses = cur_l
            
    tot_pnl = cum_pnl[-1]
    gross_wins = np.sum(np.where(pnl_series > 0, pnl_series, 0))
    gross_losses = np.sum(np.where(pnl_series < 0, -pnl_series, 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else np.nan
    win_rate = np.mean(is_win_series)
    
    return tot_pnl, max_dd, max_dur, max_losses, max_wins, pf, win_rate, len(pnl_series)

# Extract 15m & 5m
s_15m = extract_signals("15m", use_rsi_ih=False)
s_5m = extract_signals("5m", use_rsi_ih=True)

# Combine chronological
combined = pd.concat([s_15m, s_5m]).sort_values("dt").reset_index(drop=True)
now = combined["dt"].max()

FEE = 0.98
PREMIUM = 0.01

# Scenarios:
# 1. Quote 0.30 (payoff = 0.98 / 0.31 - 1 = +2.161 R)
# 2. Quote 0.20 (payoff = 0.98 / 0.21 - 1 = +3.667 R)
# 3. Market Order (quote ~0.50, payoff = 0.98 / 0.51 - 1 = +0.922 R)

PAYOFF_030 = FEE / (0.30 + PREMIUM) - 1.0 # +2.161
PAYOFF_020 = FEE / (0.20 + PREMIUM) - 1.0 # +3.667
PAYOFF_MKT = 0.96

print("=" * 100)
print("DUAL-TIMEFRAME (15m + 5m) COMPREHENSIVE PERFORMANCE AND DRAWDOWN AUDIT")
print("=" * 100)

for days in [30, 90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    sub = combined[combined["dt"] >= cutoff].copy().reset_index(drop=True)
    
    print(f"\n==========================================================================================")
    print(f"--- PERIOD: RECENT {days:3d} DAYS ({cutoff.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}) ---")
    print(f"Total Signals Printed: {len(sub)} (15m: {len(sub[sub['tf']=='15m'])}, 5m: {len(sub[sub['tf']=='5m'])})")
    print(f"==========================================================================================")
    
    # ------------------ Mode 1: 0.30 Limit Order (Balanced / Recommended) ------------------
    # Trigger requirement: adverse excursion >= 0.0004 for 15m, >= 0.0003 for 5m
    filled_030 = sub[
        ((sub["tf"] == "15m") & (sub["adverse_excursion"] >= 0.0004)) |
        ((sub["tf"] == "5m") & (sub["adverse_excursion"] >= 0.0003))
    ].copy().reset_index(drop=True)
    
    pnl_030 = np.where(filled_030["is_win"], PAYOFF_030, -1.0)
    pnl_tot_030, dd_030, dur_030, ml_030, mw_030, pf_030, wr_030, n_030 = calc_curve_stats(pnl_030, filled_030["is_win"].values, filled_030["dt"])
    
    # ------------------ Mode 2: 0.20 Limit Order (Extreme Convexity) ------------------
    # Trigger requirement: adverse excursion >= 0.0012 for 15m, >= 0.0008 for 5m
    filled_020 = sub[
        ((sub["tf"] == "15m") & (sub["adverse_excursion"] >= 0.0012)) |
        ((sub["tf"] == "5m") & (sub["adverse_excursion"] >= 0.0008))
    ].copy().reset_index(drop=True)
    
    pnl_020 = np.where(filled_020["is_win"], PAYOFF_020, -1.0)
    pnl_tot_020, dd_020, dur_020, ml_020, mw_020, pf_020, wr_020, n_020 = calc_curve_stats(pnl_020, filled_020["is_win"].values, filled_020["dt"])
    
    # ------------------ Mode 3: Market Order (All In, No Discount) ------------------
    pnl_mkt = np.where(sub["is_win"], PAYOFF_MKT, -1.0)
    pnl_tot_mkt, dd_mkt, dur_mkt, ml_mkt, mw_mkt, pf_mkt, wr_mkt, n_mkt = calc_curve_stats(pnl_mkt, sub["is_win"].values, sub["dt"])

    # Table output
    res_df = pd.DataFrame([
        {
            "模式名称": "0.30 护栏挂单 (推荐)",
            "实际成交单量": f"{n_030} 笔 ({n_030/len(sub)*100:.1f}%)",
            "日均成交": f"{n_030/days:.2f} 笔/天",
            "真实胜率": f"{wr_030*100:.1f}%",
            "累计净利 (R)": f"{pnl_tot_030:+7.1f} R",
            "获利因子 PF": f"{pf_030:4.2f}",
            "最大回撤 (R)": f"{dd_030:5.1f} R",
            "最大连亏": f"{ml_030} 连亏",
            "回撤修复期": f"{dur_030} 笔交易",
        },
        {
            "模式名称": "0.20 极限挂单 (深折)",
            "实际成交单量": f"{n_020} 笔 ({n_020/len(sub)*100:.1f}%)",
            "日均成交": f"{n_020/days:.2f} 笔/天",
            "真实胜率": f"{wr_020*100:.1f}%",
            "累计净利 (R)": f"{pnl_tot_020:+7.1f} R",
            "获利因子 PF": f"{pf_020:4.2f}",
            "最大回撤 (R)": f"{dd_020:5.1f} R",
            "最大连亏": f"{ml_020} 连亏",
            "回撤修复期": f"{dur_020} 笔交易",
        },
        {
            "模式名称": "市价直接入场 (无折扣)",
            "实际成交单量": f"{n_mkt} 笔 (100%)",
            "日均成交": f"{n_mkt/days:.2f} 笔/天",
            "真实胜率": f"{wr_mkt*100:.1f}%",
            "累计净利 (R)": f"{pnl_tot_mkt:+7.1f} R",
            "获利因子 PF": f"{pf_mkt:4.2f}",
            "最大回撤 (R)": f"{dd_mkt:5.1f} R",
            "最大连亏": f"{ml_mkt} 连亏",
            "回撤修复期": f"{dur_mkt} 笔交易",
        },
    ])
    print(res_df.to_string(index=False))
