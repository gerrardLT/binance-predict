import pandas as pd
import numpy as np

# Run Ver 2 backtest on 1h klines
df = pd.read_csv("output/klines_1h_720d.csv")
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

# 1. 明显高点 / 低点（前 4 根内最高 / 最低）
df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]

# 2. 实体是附近几根中最长（大于前 3 根实体）
df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]

# 3. 完全包裹住 (整根 K 线都在前柱高低点内，即经典 Inside Bar / 孕线)
df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)

# Outcomes (次根 1h K 线表现)
df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)

# Ver 2 conditions:
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

now = df["dt"].max()
print("=" * 80)
print(f"[1H 周期] Ver 2 全量长期回测：30d / 90d / 180d / 360d / 720d")
print(f"数据跨度: {df.iloc[0]['dt'].strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')} (总计 {(now - df.iloc[0]['dt']).days} 天)")
print("=" * 80)

for days in [30, 90, 180, 360, 720]:
    cutoff = now - pd.Timedelta(days=days)
    sub_hm = df[(df["dt"] >= cutoff) & hm_cond].copy()
    sub_ih = df[(df["dt"] >= cutoff) & ih_cond].copy()

    sub_hm = sub_hm[(sub_hm.index >= 5) & (sub_hm.index < len(df) - 3)]
    sub_ih = sub_ih[(sub_ih.index >= 5) & (sub_ih.index < len(df) - 3)]

    n_hm = len(sub_hm)
    w_hm = sub_hm["next_is_bear"].sum()
    r_hm = w_hm / n_hm if n_hm > 0 else 0
    spike_hm = (sub_hm["next_high_spike"] > 0.0005).mean() if n_hm > 0 else 0
    ret_hm = (-sub_hm["next_ret"]).mean() if n_hm > 0 else 0

    n_ih = len(sub_ih)
    w_ih = sub_ih["next_is_bull"].sum()
    r_ih = w_ih / n_ih if n_ih > 0 else 0
    dip_ih = (sub_ih["next_low_dip"] > 0.0005).mean() if n_ih > 0 else 0
    ret_ih = sub_ih["next_ret"].mean() if n_ih > 0 else 0

    tot_n = n_hm + n_ih
    tot_w = w_hm + w_ih
    tot_r = tot_w / tot_n if tot_n > 0 else 0

    print(f"\n--- {days} 天跨度 ({cutoff.strftime('%Y-%m-%d')} 至 {now.strftime('%Y-%m-%d')}) ---")
    print(f"  综合表现: 总计触发 {tot_n:3d} 次 | 综合胜率 {tot_r*100:5.2f}% ({tot_w}胜 / {tot_n-tot_w}负)")
    print(f"  上吊线 (看跌 DOWN): 触发 {n_hm:3d} 次 | 胜率 {r_hm*100:5.2f}% ({w_hm}胜/{n_hm-w_hm}负) | 惯性冲高折价率={spike_hm*100:4.1f}% | 次根均幅={ret_hm*100:+.3f}%")
    print(f"  倒垂线 (看涨   UP): 触发 {n_ih:3d} 次 | 胜率 {r_ih*100:5.2f}% ({w_ih}胜/{n_ih-w_ih}负) | 惯性下探折价率={dip_ih*100:4.1f}% | 次根均幅={ret_ih*100:+.3f}%")
