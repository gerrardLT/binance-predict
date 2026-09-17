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

# Bar N-1 features
df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)

# Ver 2 conditions:
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# Ver 2 filters
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

# Extract targets (Bar N+1)
# For Hanging Man (bet DOWN):
# How high did next bar spike up?
# relative to open price: spike_pct = (next_high - next_open) / next_open
df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
# Did next bar close down?
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

# For Inverted Hammer (bet UP):
# How low did next bar dip down?
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
# Did next bar close up?
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)

# Clean signals
sub_hm = df[hm_v2 & (df.index >= 5) & (df.index < len(df) - 3)].copy()
sub_hm["signal_type"] = "HM_DOWN"
sub_hm["is_win"] = sub_hm["next_is_bear"]
sub_hm["adverse_excursion"] = sub_hm["next_high_spike"] # 逆行幅度（假冲刺幅度）

sub_ih = df[ih_v2 & (df.index >= 5) & (df.index < len(df) - 3)].copy()
sub_ih["signal_type"] = "IH_UP"
sub_ih["is_win"] = sub_ih["next_is_bull"]
sub_ih["adverse_excursion"] = sub_ih["next_low_dip"] # 逆行幅度（假下探幅度）

all_signals = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
now = all_signals["dt"].max()

print("=" * 100)
print(f"15m Ver 2 全样本信号总数: {len(all_signals)} (HM: {len(sub_hm)}, IH: {len(sub_ih)})")
print("=" * 100)

# Micro-pricing relation in prediction market:
# In Binance 15m prediction markets:
# At bar open, initial quote is around 0.50 (± 0.05).
# As price moves in the original trend direction by X bps (adverse excursion):
# The quote for the counter-trend token (our bet) drops sharply!
# Typically:
# - excursion >= 0.0003 (3 bps)  --> quote touches ~0.35 ~ 0.40
# - excursion >= 0.0006 (6 bps)  --> quote touches ~0.30
# - excursion >= 0.0010 (10 bps) --> quote touches ~0.25
# - excursion >= 0.0015 (15 bps) --> quote touches ~0.20
# - excursion >= 0.0022 (22 bps) --> quote touches ~0.15
# - excursion >= 0.0030 (30 bps) --> quote touches ~0.10

# Let's test a comprehensive grid of limit order price levels:
# Price P in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
FEE = 0.98
PREMIUM = 0.01

quote_grid = [
    # (quote, min_adverse_bps, label)
    (0.10, 0.0030, "0.10 极度深渊价 (需大假突破 ~30bps)"),
    (0.15, 0.0020, "0.15 超深折扣价 (需假突破 ~20bps)"),
    (0.20, 0.0012, "0.20 经典折扣价 (需中假突破 ~12bps)"),
    (0.25, 0.0007, "0.25 适度折扣价 (需轻微假突破 ~7bps)"),
    (0.30, 0.0004, "0.30 温和折扣价 (微冲即成交 ~4bps)"),
    (0.35, 0.0002, "0.35 浅折扣价 (微幅扰动 ~2bps)"),
    (0.40, 0.0001, "0.40 微折扣价 (开盘几乎必成交 ~1bps)"),
    (0.48, 0.0000, "0.48 市价开盘直击 (100% 全部成交)"),
]

for days in [90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    sub = all_signals[all_signals["dt"] >= cutoff].copy()
    n_total = len(sub)
    
    print(f"\n==========================================================================================")
    print(f"【最近 {days:3d} 天网格扫描】总信号数: {n_total} 笔 (日均 {(n_total/days):.1f} 笔)")
    print(f"==========================================================================================")
    print(f"{'挂单价格':<12} | {'成交单量(成交率)':<16} | {'真实胜率':<8} | {'单笔赔率':<9} | {'单笔期望EV':<10} | {'累计净利润(R)':<13} | {'获利因子PF':<8} | {'最大连亏':<8}")
    print("-" * 100)
    
    best_ev = -999
    best_pnl = -999
    best_quote = None
    
    for q, req_bps, q_label in quote_grid:
        # Payoff calculation
        payoff = FEE / (q + PREMIUM) - 1.0
        
        # Which trades filled?
        filled = sub[sub["adverse_excursion"] >= req_bps]
        n_fill = len(filled)
        fill_rate = n_fill / n_total if n_total > 0 else 0
        
        if n_fill == 0:
            continue
            
        wins = filled["is_win"].sum()
        win_rate = wins / n_fill
        
        # PnL & EV
        pnl_series = np.where(filled["is_win"], payoff, -1.0)
        tot_pnl = np.sum(pnl_series)
        ev = tot_pnl / n_fill
        
        # Profit factor
        gross_w = wins * payoff
        gross_l = (n_fill - wins) * 1.0
        pf = gross_w / gross_l if gross_l > 0 else np.nan
        
        # Max consecutive losses
        cur_loss, max_loss = 0, 0
        for w in filled["is_win"]:
            if not w:
                cur_loss += 1
                if cur_loss > max_loss: max_loss = cur_loss
            else:
                cur_loss = 0
                
        print(f"{q:<12.2f} | {n_fill:4d} 笔 ({fill_rate*100:4.1f}%)   | {win_rate*100:5.1f}%  | 1:{payoff+1:4.2f}   | {ev*100:+7.1f}%   | {tot_pnl:+9.1f} R   | {pf:5.2f}    | {max_loss:2d} 连亏")
