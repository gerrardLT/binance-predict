import pandas as pd
import numpy as np

def analyze_5m_fill_frequency():
    df = pd.read_csv("output/klines_5m_720d.csv")
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

    # Next bar excursion
    df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
    df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
    df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)

    # 1. 5m Ver2 纯基准版
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

    # 2. 5m Ver2 稳健版（叠加倒垂线 RSI<=40）
    ih_robust = ih_base & (df["rsi_14"].shift(1) <= 40)

    for mode_name, hm_c, ih_c in [
        ("5M Ver2 纯基准版", hm_base, ih_base),
        ("5M Ver2 稳健版 (倒垂叠加 RSI<=40)", hm_base, ih_robust),
    ]:
        sub_hm = df[hm_c & (df.index >= 5) & (df.index < len(df) - 3)].copy()
        sub_hm["signal_type"] = "HM_DOWN"
        sub_hm["is_win"] = sub_hm["next_is_bear"]
        sub_hm["adverse_excursion"] = sub_hm["next_high_spike"]

        sub_ih = df[ih_c & (df.index >= 5) & (df.index < len(df) - 3)].copy()
        sub_ih["signal_type"] = "IH_UP"
        sub_ih["is_win"] = sub_ih["next_is_bull"]
        sub_ih["adverse_excursion"] = sub_ih["next_low_dip"]

        all_sig = pd.concat([sub_hm, sub_ih]).sort_values("dt").reset_index(drop=True)
        now = all_sig["dt"].max()

        print("=" * 85)
        print(f"[{mode_name}] 信号出现次数 vs 真实成交下单次数")
        print("=" * 85)

        for days in [30, 90, 180, 360, 720]:
            cutoff = now - pd.Timedelta(days=days)
            sub = all_sig[all_sig["dt"] >= cutoff].copy()
            n_sig = len(sub)
            n_hm = len(sub[sub["signal_type"] == "HM_DOWN"])
            n_ih = len(sub[sub["signal_type"] == "IH_UP"])

            # 5m 盘口特性：0.30 护栏价对应微观假突破 ~ 3-4 bps (>=0.0003)
            fill_030 = sub[sub["adverse_excursion"] >= 0.0003]
            n_f030 = len(fill_030)
            hm_f030 = len(fill_030[fill_030["signal_type"] == "HM_DOWN"])
            ih_f030 = len(fill_030[fill_030["signal_type"] == "IH_UP"])

            # 0.20 护栏价对应微观深假突破 ~ 8-10 bps (>=0.0008)
            fill_020 = sub[sub["adverse_excursion"] >= 0.0008]
            n_f020 = len(fill_020)
            hm_f020 = len(fill_020[fill_020["signal_type"] == "HM_DOWN"])
            ih_f020 = len(fill_020[fill_020["signal_type"] == "IH_UP"])

            print(f"\n--- {days} Days ({cutoff.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}) ---")
            print(f"  Signal Printed    : {n_sig:4d} signals (Daily avg: {n_sig/days:4.2f} / day | HM={n_hm:3d}, IH={n_ih:3d})")
            print(f"  0.30 Real Filled  : {n_f030:4d} orders  (Daily avg: {n_f030/days:4.2f} / day | Fill Rate: {n_f030/n_sig*100:4.1f}%)")
            print(f"     Breakdown      : HM filled={hm_f030:3d}, IH filled={ih_f030:3d}")
            print(f"  0.20 Real Filled  : {n_f020:4d} orders  (Daily avg: {n_f020/days:4.2f} / day | Fill Rate: {n_f020/n_sig*100:4.1f}%)")
            print(f"     Breakdown      : HM filled={hm_f020:3d}, IH filled={ih_f020:3d}")

if __name__ == "__main__":
    analyze_5m_fill_frequency()
