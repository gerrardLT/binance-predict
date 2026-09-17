import pandas as pd
import numpy as np

def run_backtest_ver2(tf="15m"):
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

    # Outcomes
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
    print(f"[{tf.upper()}] Ver 2 Backtest: 30d / 90d / 180d / 360d / 720d")
    print(f"Time span: {df.iloc[0]['dt'].strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')} ({(now - df.iloc[0]['dt']).days} days)")
    print("=" * 80)

    for days in [30, 90, 180, 360, 720]:
        cutoff = now - pd.Timedelta(days=days)
        sub_hm = df[(df["dt"] >= cutoff) & hm_cond].copy()
        sub_ih = df[(df["dt"] >= cutoff) & ih_cond].copy()

        # 剔除边缘
        sub_hm = sub_hm[(sub_hm.index >= 5) & (sub_hm.index < len(df) - 3)]
        sub_ih = sub_ih[(sub_ih.index >= 5) & (sub_ih.index < len(df) - 3)]

        n_hm = len(sub_hm)
        w_hm = sub_hm["next_is_bear"].sum()
        r_hm = w_hm / n_hm if n_hm > 0 else 0
        spike_hm = (sub_hm["next_high_spike"] > 0.0003).mean() if n_hm > 0 else 0
        ret_hm = (-sub_hm["next_ret"]).mean() if n_hm > 0 else 0

        n_ih = len(sub_ih)
        w_ih = sub_ih["next_is_bull"].sum()
        r_ih = w_ih / n_ih if n_ih > 0 else 0
        dip_ih = (sub_ih["next_low_dip"] > 0.0003).mean() if n_ih > 0 else 0
        ret_ih = sub_ih["next_ret"].mean() if n_ih > 0 else 0

        tot_n = n_hm + n_ih
        tot_w = w_hm + w_ih
        tot_r = tot_w / tot_n if tot_n > 0 else 0

        print(f"\n--- {days} Days ({cutoff.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}) ---")
        print(f"  TOTAL: n={tot_n:4d} | Win Rate={tot_r*100:5.2f}% ({tot_w}W / {tot_n-tot_w}L)")
        print(f"  Hanging Man (DOWN)    : n={n_hm:4d} | Win Rate={r_hm*100:5.2f}% ({w_hm}W / {n_hm-w_hm}L) | Spike 0.20 Rate={spike_hm*100:4.1f}% | Avg Return={ret_hm*100:+.3f}%")
        print(f"  Inverted Hammer (UP)  : n={n_ih:4d} | Win Rate={r_ih*100:5.2f}% ({w_ih}W / {n_ih-w_ih}L) | Dip 0.20 Rate  ={dip_ih*100:4.1f}% | Avg Return={ret_ih*100:+.3f}%")

if __name__ == "__main__":
    run_backtest_ver2("15m")
    run_backtest_ver2("5m")
