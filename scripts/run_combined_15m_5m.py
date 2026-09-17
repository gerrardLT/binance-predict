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
    df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
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

    # Filter out boundary bars
    sub_hm = df[hm_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_hm["signal_type"] = "HM_DOWN"
    sub_hm["direction"] = "DOWN"
    sub_hm["is_win"] = sub_hm["next_is_bear"]
    sub_hm["pnl_ret"] = -sub_hm["next_ret"]
    sub_hm["discount_opp"] = sub_hm["next_high_spike"] > 0.0003

    sub_ih = df[ih_cond & (df.index >= 5) & (df.index < len(df) - 3)].copy()
    sub_ih["signal_type"] = "IH_UP"
    sub_ih["direction"] = "UP"
    sub_ih["is_win"] = sub_ih["next_is_bull"]
    sub_ih["pnl_ret"] = sub_ih["next_ret"]
    sub_ih["discount_opp"] = sub_ih["next_low_dip"] > 0.0003

    merged = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
    merged["tf"] = tf
    return merged

# 15m signals (Ver 2 standard)
signals_15m = extract_signals("15m", use_rsi_ih=False)

# 5m signals (Ver 2 with IH RSI<=40 for robustness)
signals_5m = extract_signals("5m", use_rsi_ih=True)

# Combine both timelines
combined = pd.concat([signals_15m, signals_5m]).sort_values("dt").reset_index(drop=True)
now = combined["dt"].max()

print("=" * 90)
print("【15m + 5m 双周期合并回测综合报告】")
print(f"数据总跨度: {combined.iloc[0]['dt'].strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}")
print("=" * 90)

FEE = 0.98
PREMIUM = 0.01
QUOTE = 0.20
PAYOFF = FEE / (QUOTE + PREMIUM) - 1.0 # +3.667

for days in [30, 90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    sub = combined[combined["dt"] >= cutoff].copy()
    
    # By timeframe
    sub_15 = sub[sub["tf"] == "15m"]
    sub_5 = sub[sub["tf"] == "5m"]
    
    n_tot = len(sub)
    w_tot = sub["is_win"].sum()
    r_tot = w_tot / n_tot if n_tot > 0 else 0
    
    n_15 = len(sub_15)
    w_15 = sub_15["is_win"].sum()
    r_15 = w_15 / n_15 if n_15 > 0 else 0
    
    n_5 = len(sub_5)
    w_5 = sub_5["is_win"].sum()
    r_5 = w_5 / n_5 if n_5 > 0 else 0
    
    # 0.20 limit order simulation:
    # Only trades where discount_opp == True will fill!
    filled_trades = sub[sub["discount_opp"] == True]
    n_filled = len(filled_trades)
    w_filled = filled_trades["is_win"].sum()
    r_filled = w_filled / n_filled if n_filled > 0 else 0
    
    # EV & Net Profit per unit risked
    # Win -> +PAYOFF (+3.667), Loss -> -1.0
    pnl_units = w_filled * PAYOFF - (n_filled - w_filled) * 1.0
    ev_per_trade = pnl_units / n_filled if n_filled > 0 else 0
    
    # Profit factor
    gross_win = w_filled * PAYOFF
    gross_loss = (n_filled - w_filled) * 1.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan
    
    print(f"\n========================================================")
    print(f"【最近 {days:3d} 天】({cutoff.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')})")
    print(f"========================================================")
    print(f"1. 订单流与胜率分布:")
    print(f"   * 双周期合并总单量 : {n_tot:4d} 笔 (平均每日 {(n_tot/days):4.1f} 笔)")
    print(f"   * 综合总胜率       : {r_tot*100:5.2f}% ({w_tot} 胜 / {n_tot-w_tot} 负)")
    print(f"   * 15m 主力贡献     : {n_15:4d} 笔 (占比 {n_15/n_tot*100:4.1f}%) | 胜率: {r_15*100:5.2f}% ({w_15}胜/{n_15-w_15}负)")
    print(f"   * 5m  高频补充     : {n_5:4d} 笔 (占比 {n_5/n_tot*100:4.1f}%) | 胜率: {r_5*100:5.2f}% ({w_5}胜/{n_5-w_5}负)")
    
    print(f"\n2. 0.20 极限折价挂单实操模拟:")
    print(f"   * 成功成交订单数   : {n_filled:4d} 笔 (挂单成交率: {n_filled/n_tot*100:4.1f}%)")
    print(f"   * 成交单真实胜率   : {r_filled*100:5.2f}% ({w_filled} 胜 / {n_filled-w_filled} 负)")
    print(f"   * 盈亏比 / 赔率    : 1 : 4.67 (净盈利赔率 +366.7%)")
    print(f"   * 净收益点数(R倍数): {pnl_units:+6.1f} R (单笔平均期望 EV: {ev_per_trade*100:+5.1f}%)")
    print(f"   * 获利因子 (PF)    : {profit_factor:5.2f}")
