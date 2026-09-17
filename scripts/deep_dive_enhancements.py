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

# Bar N-1
df["prev_is_green"] = df["is_green"].shift(1)
df["prev_body"] = df["body"].shift(1)
df["prev_body_r"] = df["body_r"].shift(1)
df["prev_high"] = df["high"].shift(1)
df["prev_low"] = df["low"].shift(1)
df["prev_open"] = df["open"].shift(1)
df["prev_close"] = df["close"].shift(1)

# Base Ver 2 Setup
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# Outcomes
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)

# Let's explore several deep hypotheses:
# Option A: Wick Dominance (主影线优势：影线不仅>=0.50，且影线长度 >= 2.0 * 实体)
df["hm_wick_dominance"] = (df["lower_wick"] >= 2.0 * df["body"]) & (df["lower_r"] >= 0.55)
df["ih_wick_dominance"] = (df["upper_wick"] >= 2.0 * df["body"]) & (df["upper_r"] >= 0.55)

# Option B: Body Position in Top/Bottom third (实体位置必须在整根K线的前1/3极值区)
df["hm_body_in_top_35"] = (df[["open", "close"]].min(axis=1) - df["low"]) >= df["range"] * 0.55
df["ih_body_in_bot_35"] = (df["high"] - df[["open", "close"]].max(axis=1)) >= df["range"] * 0.55

# Option C: Confirmation of Divergence (RSI(14) or Momentum Peak)
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / np.where(loss > 0, loss, 1.0)
df["rsi_14"] = 100 - (100 / (1 + rs))
df["hm_rsi_div"] = df["rsi_14"].shift(1) >= 60 # RSI在强势区出现上吊见顶
df["ih_rsi_div"] = df["rsi_14"].shift(1) <= 40 # RSI在超卖区出现倒垂见底

# Option D: Closing Color Match (上吊收红，倒垂收绿)
df["hm_color_match"] = df["close"] < df["open"]
df["ih_color_match"] = df["close"] > df["open"]

# Base Ver 2 Masks
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

now = df["dt"].max()

candidates = {
    "1. Ver 2 基准": (True, True),
    "2. 深度增强 A: 影线绝对主导 (影线>=2倍实体 & 影线>=55%)": (df["hm_wick_dominance"], df["ih_wick_dominance"]),
    "3. 深度增强 B: 实体强锁定在极值前35%区间": (df["hm_body_in_top_35"], df["ih_body_in_bot_35"]),
    "4. 深度增强 C: RSI动量势区配合 (上吊RSI>=60 / 倒垂RSI<=40)": (df["hm_rsi_div"], df["ih_rsi_div"]),
    "5. 深度增强 D: 信号柱收盘色同向确认 (上吊收红跌 / 倒垂收绿涨)": (df["hm_color_match"], df["ih_color_match"]),
    "6. 强强联手: 影线主导 A + 收盘确认 D": (df["hm_wick_dominance"] & df["hm_color_match"], df["ih_wick_dominance"] & df["ih_color_match"]),
    "7. 终极精英版: 影线主导 A + RSI势区 C": (df["hm_wick_dominance"] & df["hm_rsi_div"], df["ih_wick_dominance"] & df["ih_rsi_div"]),
}

for name, (c_hm, c_ih) in candidates.items():
    print("=" * 80)
    print(name)
    print("=" * 80)
    for days in [30, 90, 180, 360, 720]:
        cutoff = now - pd.Timedelta(days=days)
        s_hm = df[(df["dt"] >= cutoff) & hm_v2 & c_hm]
        s_ih = df[(df["dt"] >= cutoff) & ih_v2 & c_ih]
        
        n_hm, w_hm = len(s_hm), s_hm["next_is_bear"].sum()
        r_hm = w_hm / n_hm if n_hm > 0 else 0
        
        n_ih, w_ih = len(s_ih), s_ih["next_is_bull"].sum()
        r_ih = w_ih / n_ih if n_ih > 0 else 0
        
        tot_n = n_hm + n_ih
        tot_w = w_hm + w_ih
        tot_r = tot_w / tot_n if tot_n > 0 else 0
        
        print(f"  {days:3d}天: 总样本={tot_n:3d}, 综合胜率={tot_r*100:5.1f}% | 上吊线 n={n_hm:2d}(胜率{r_hm*100:5.1f}%) | 倒垂线 n={n_ih:2d}(胜率{r_ih*100:5.1f}%)")
