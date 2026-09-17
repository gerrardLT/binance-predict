import pandas as pd
import numpy as np

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
df["prev_body_r"] = df["body_r"].shift(1)

df["green_count_5"] = df["is_green"].shift(1).rolling(5).sum()
df["ret_5"] = (df["close"].shift(1) - df["open"].shift(5)) / df["open"].shift(5)

df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)

df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

hm_cond = (
    (df["prev_is_green"] == True) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["green_count_5"] >= 3) & 
    (df["ret_5"] > 0) & 
    (df["lower_r"] >= 0.50) & 
    (df["body_r"] <= 0.40) & 
    (df["upper_r"] <= 0.20)
)
ih_cond = (
    (df["prev_is_green"] == False) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["green_count_5"] <= 2) & 
    (df["ret_5"] < 0) & 
    (df["upper_r"] >= 0.50) & 
    (df["body_r"] <= 0.40) & 
    (df["lower_r"] <= 0.20)
)

now = df["dt"].max()

for days in [30, 90]:
    cutoff = now - pd.Timedelta(days=days)
    sub_hm = df[(df["dt"] >= cutoff) & hm_cond].copy()
    sub_ih = df[(df["dt"] >= cutoff) & ih_cond].copy()
    
    sub_hm = sub_hm[(sub_hm.index >= 5) & (sub_hm.index < len(df) - 3)]
    sub_ih = sub_ih[(sub_ih.index >= 5) & (sub_ih.index < len(df) - 3)]
    
    hm_win = sub_hm["next_is_bear"].mean()
    hm_spike = (sub_hm["next_high_spike"] > 0.0003).mean()
    hm_avg_ret = (-sub_hm["next_ret"]).mean()
    
    ih_win = sub_ih["next_is_bull"].mean()
    ih_dip = (sub_ih["next_low_dip"] > 0.0003).mean()
    ih_avg_ret = sub_ih["next_ret"].mean()
    
    print(f"=== {days} Days ({cutoff.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}) ===")
    print(f"Hanging Man (HM): count={len(sub_hm)}, win={hm_win*100:.2f}%, spike_rate={hm_spike*100:.2f}%, avg_ret={hm_avg_ret*100:+.2f}%")
    print(f"Inverted Hammer (IH): count={len(sub_ih)}, win={ih_win*100:.2f}%, dip_rate={ih_dip*100:.2f}%, avg_ret={ih_avg_ret*100:+.2f}%")
