import pandas as pd
import numpy as np

# Load 5m klines
df = pd.read_csv("output/klines_5m_720d.csv")
df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("dt").reset_index(drop=True)

# 1. Geometry
df["range"] = df["high"] - df["low"]
df["body"] = (df["close"] - df["open"]).abs()
df["body_r"] = df["body"] / np.where(df["range"] > 0, df["range"], 1.0)
df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
df["upper_r"] = df["upper_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
df["lower_r"] = df["lower_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
df["is_green"] = df["close"] >= df["open"]

# 2. Bar N-1
df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)

# 3. Ver 2 Core conditions
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# 4. RSI(14) calculation on 5m
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / np.where(loss > 0, loss, 1.0)
df["rsi_14"] = 100 - (100 / (1 + rs))

# 5. Outcomes
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

# Base Ver 2 Filters
hm_base = (
    (df["prev_is_green"] == True) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["is_prominent_high"]) & 
    (df["is_longest_body"]) & 
    (df["is_full_inside"]) & 
    (df["lower_r"] >= 0.45) & 
    (df["body_r"] <= 0.45) & 
    (df["upper_r"] <= 0.25)
)

ih_base = (
    (df["prev_is_green"] == False) & 
    (df["prev_body_r"] >= 0.40) & 
    (df["is_prominent_low"]) & 
    (df["is_longest_body"]) & 
    (df["is_full_inside"]) & 
    (df["upper_r"] >= 0.45) & 
    (df["body_r"] <= 0.45) & 
    (df["lower_r"] <= 0.25)
)

# Enhanced Combinations for 5m:
# Combo A: 上吊线原版 + 倒垂线(RSI<=40 或 收阳)
ih_enhanced_or = ih_base & ((df["rsi_14"].shift(1) <= 40) | (df["close"] > df["open"]))

# Combo B: 上吊线(RSI>=60) + 倒垂线(RSI<=40) (双向动量共振)
hm_rsi60 = hm_base & (df["rsi_14"].shift(1) >= 60)
ih_rsi40 = ih_base & (df["rsi_14"].shift(1) <= 40)

# Combo C: 上吊线原版 + 倒垂线仅限RSI<=40
ih_only_rsi = ih_base & (df["rsi_14"].shift(1) <= 40)

# Combo D: 上吊线原版 + 倒垂线仅限收阳
ih_only_green = ih_base & (df["close"] > df["open"])

now = df["dt"].max()

print("=" * 85)
print(f"[5M 周期] Ver 2 深度增强组合策略回测 (30d / 90d / 180d / 360d / 720d)")
print(f"数据总跨度: {df.iloc[0]['dt'].strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')} (总计 720+ 天)")
print("=" * 85)

schemes = [
    ("【方案 0】原 Ver 2 基准版 (无额外过滤)", hm_base, ih_base),
    ("【方案 1: 推荐组合】上吊线基准 + 倒垂线(RSI<=40 或 收阳)", hm_base, ih_enhanced_or),
    ("【方案 2: 纯动量共振】上吊线(RSI>=60) + 倒垂线(RSI<=40)", hm_rsi60, ih_rsi40),
    ("【方案 3: 严格超卖倒垂】上吊线基准 + 倒垂线(仅限RSI<=40)", hm_base, ih_only_rsi),
    ("【方案 4: 阳线控盘倒垂】上吊线基准 + 倒垂线(仅限自身收阳)", hm_base, ih_only_green),
]

for s_name, hm_c, ih_c in schemes:
    print(f"\n>>>>>>>>>>>>>>> {s_name} <<<<<<<<<<<<<<<")
    for days in [30, 90, 180, 360, 720]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & hm_c]
        sub_ih = df[(df["dt"] >= cutoff) & ih_c]
        
        n_hm = len(sub_hm)
        w_hm = sub_hm["next_is_bear"].sum()
        r_hm = w_hm / n_hm if n_hm > 0 else 0
        spike_hm = (sub_hm["next_high_spike"] > 0.0003).mean() if n_hm > 0 else 0
        
        n_ih = len(sub_ih)
        w_ih = sub_ih["next_is_bull"].sum()
        r_ih = w_ih / n_ih if n_ih > 0 else 0
        dip_ih = (sub_ih["next_low_dip"] > 0.0003).mean() if n_ih > 0 else 0
        
        tot_n = n_hm + n_ih
        tot_w = w_hm + w_ih
        tot_r = tot_w / tot_n if tot_n > 0 else 0
        
        print(f"  {days:3d}天: 总单={tot_n:4d} (日均{tot_n/days:4.1f}单) | 综合胜率={tot_r*100:5.2f}% ({tot_w}胜/{tot_n-tot_w}负) | 上吊HM: n={n_hm:3d}(胜率{r_hm*100:4.1f}%, 冲高折价{spike_hm*100:4.1f}%) | 倒垂IH: n={n_ih:3d}(胜率{r_ih*100:4.1f}%, 下探折价{dip_ih*100:4.1f}%)")
