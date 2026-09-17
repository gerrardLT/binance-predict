import pandas as pd
import numpy as np

# Comprehensive performance comparison across all test versions
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

# High/Low & Longest body
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

df["green_count_5"] = df["is_green"].shift(1).rolling(5).sum()
df["ret_5"] = (df["close"].shift(1) - df["open"].shift(5)) / df["open"].shift(5)

# Wrap conditions
df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)
df["is_body_inside"] = (df[["open", "close"]].max(axis=1) <= df[["open", "close"]].shift(1).max(axis=1)) & \
                       (df[["open", "close"]].min(axis=1) >= df[["open", "close"]].shift(1).min(axis=1))

# Outcomes
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

now = df["dt"].max()

# Define versions to rank
versions = {}

# Ver 1: 3项核心基础版 (前置充实实体 + 趋势占优 + 宽容实体)
versions["Ver 1: 基础核心版 (前置实体+趋势占优)"] = {
    "hm": (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["green_count_5"] >= 3) & (df["ret_5"] > 0) & (df["lower_r"] >= 0.50) & (df["body_r"] <= 0.40) & (df["upper_r"] <= 0.20),
    "ih": (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["green_count_5"] <= 2) & (df["ret_5"] < 0) & (df["upper_r"] >= 0.50) & (df["body_r"] <= 0.40) & (df["lower_r"] <= 0.20),
}

# Ver 2: 完全包裹孕线版 (明显高低点 + 最长实体 + 完全包裹)
versions["Ver 2: 完全孕线包裹版 (明显极值+附近最长实体+完全包裹)"] = {
    "hm": (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_high"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["lower_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["upper_r"] <= 0.25),
    "ih": (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_low"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["upper_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["lower_r"] <= 0.25),
}

# Ver 3: 实体包裹孕线版 (明显极值 + 实体包裹)
versions["Ver 3: 实体孕线版 (明显极值+实体包裹)"] = {
    "hm": (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_high"]) & (df["is_body_inside"]) & (df["lower_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["upper_r"] <= 0.25),
    "ih": (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_low"]) & (df["is_body_inside"]) & (df["upper_r"] >= 0.45) & (df["body_r"] <= 0.45) & (df["lower_r"] <= 0.25),
}

# Ver 4: 严格光头光脚版
versions["Ver 4: 严格光头光脚版 (最长实体+包裹+光头)"] = {
    "hm": (df["prev_is_green"] == True) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_high"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["lower_r"] >= 0.50) & (df["body_r"] <= 0.45) & (df["upper_r"] <= 0.06),
    "ih": (df["prev_is_green"] == False) & (df["prev_body_r"] >= 0.40) & (df["is_prominent_low"]) & (df["is_longest_body"]) & (df["is_full_inside"]) & (df["upper_r"] >= 0.50) & (df["body_r"] <= 0.45) & (df["lower_r"] <= 0.06),
}

print("=== RANKING ALL VERSIONS ===")
for v_name, v_conds in versions.items():
    print(f"\n[{v_name}]")
    for days in [30, 90]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & v_conds["hm"]]
        sub_ih = df[(df["dt"] >= cutoff) & v_conds["ih"]]
        
        n_hm = len(sub_hm)
        w_hm = sub_hm["next_is_bear"].mean() if n_hm > 0 else 0
        spike_hm = (sub_hm["next_high_spike"] > 0.0003).mean() if n_hm > 0 else 0
        
        n_ih = len(sub_ih)
        w_ih = sub_ih["next_is_bull"].mean() if n_ih > 0 else 0
        dip_ih = (sub_ih["next_low_dip"] > 0.0003).mean() if n_ih > 0 else 0
        
        # Combined
        tot_n = n_hm + n_ih
        tot_wins = (sub_hm["next_is_bear"].sum() if n_hm > 0 else 0) + (sub_ih["next_is_bull"].sum() if n_ih > 0 else 0)
        tot_win_r = tot_wins / tot_n if tot_n > 0 else 0
        
        print(f"  {days:2d}天: 总样本={tot_n:3d} (综合胜率={tot_win_r*100:5.1f}%) | 上吊线 n={n_hm:2d} 胜率={w_hm*100:5.1f}%(折价惯性={spike_hm*100:4.1f}%) | 倒垂线 n={n_ih:2d} 胜率={w_ih*100:5.1f}%(折价惯性={dip_ih*100:4.1f}%)")
