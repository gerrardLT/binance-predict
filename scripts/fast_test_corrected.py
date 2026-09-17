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

# Prior 3-bar trend
df["ret_3"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3)

# Next bar outcome
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

# Next bar micro-inertia:
# For Top Reversal (bet DOWN): next bar has an upward spike (next_high > next_open)
# If next_high - next_open is large, it creates extreme discount for DOWN!
df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
# For Bottom Reversal (bet UP): next bar has a downward dip (next_open - next_low)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

for days in [30, 90, 180]:
    cutoff = df["dt"].max() - pd.Timedelta(days=days)
    sub = df[df["dt"] >= cutoff].copy().reset_index(drop=True)
    
    # 1. 顶部长上影反转（射击之星 / 冲高拒线）：
    # 条件：upper_r >= 0.55, body_r <= 0.30, lower_r <= 0.18, ret_3 > 0
    top_rev = sub[(sub["upper_r"] >= 0.55) & (sub["body_r"] <= 0.30) & (sub["lower_r"] <= 0.18) & (sub["ret_3"] > 0)]
    
    # 2. 底部长下影反转（探底锤子线 / 探底拒线）：
    # 条件：lower_r >= 0.55, body_r <= 0.30, upper_r <= 0.18, ret_3 < 0
    bot_rev = sub[(sub["lower_r"] >= 0.55) & (sub["body_r"] <= 0.30) & (sub["upper_r"] <= 0.18) & (sub["ret_3"] < 0)]
    
    print(f"\n==================== 最近 {days} 天统计表现 ====================")
    if len(top_rev) > 0:
        top_win = top_rev["next_is_bear"].mean()
        # micro inertia rate: next bar actually spiked up before reversing
        top_spike_rate = (top_rev["next_high_spike"] > 0.0005).mean()
        print(f"【顶部冲高反转 (射击之星)】样本数: {len(top_rev)}")
        print(f"  次根直接收跌胜率 (市价): {top_win*100:.2f}%")
        print(f"  次根存在冲高惯性比例 (给到折价挂单机会): {top_spike_rate*100:.2f}%")
        
    if len(bot_rev) > 0:
        bot_win = bot_rev["next_is_bull"].mean()
        bot_dip_rate = (bot_rev["next_low_dip"] > 0.0005).mean()
        print(f"【底部探底反转 (锤子线)】样本数: {len(bot_rev)}")
        print(f"  次根直接收涨胜率 (市价): {bot_win*100:.2f}%")
        print(f"  次根存在下探惯性比例 (给到折价挂单机会): {bot_dip_rate*100:.2f}%")
