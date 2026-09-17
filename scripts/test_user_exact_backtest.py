import pandas as pd
import numpy as np

# Load 15m and 5m data
for tf in ["15m", "5m"]:
    df = pd.read_csv(f"output/klines_{tf}_720d.csv")
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
    
    # Prior bar (Bar N-1) features
    df["prev_is_green"] = df["is_green"].shift(1)
    df["prev_body_r"] = df["body_r"].shift(1)
    df["prev_range"] = df["range"].shift(1)
    # Check if prev bar was solid (e.g. body_r >= 0.45)
    # Check if prev bar had minimal wick in trend direction (flat top / flat bottom)
    df["prev_upper_r"] = df["upper_r"].shift(1)
    df["prev_lower_r"] = df["lower_r"].shift(1)
    
    # Trend dominance (e.g. prior 5 bars count of green/red)
    df["green_count_5"] = df["is_green"].shift(1).rolling(5).sum()
    df["ret_5"] = (df["close"].shift(1) - df["open"].shift(5)) / df["open"].shift(5)
    
    # Outcome: next bar (Bar N+1)
    df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
    df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
    
    # Next bar micro-inertia (for 0.20 entry):
    # For hanging man (bet DOWN): next bar dips or rallies first?
    # Next high spike:
    df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
    
    print(f"\n=================================================================")
    print(f"[{tf.upper()}] 回测结果（按您的 3 项核心细节）")
    print(f"=================================================================")
    
    for days in [30, 90, 180]:
        cutoff = df["dt"].max() - pd.Timedelta(days=days)
        sub = df[df["dt"] >= cutoff].copy().reset_index(drop=True)
        
        # 1. 上吊线 (Hanging Man):
        # - 前一根必须是实体阳线: prev_is_green == True, prev_body_r >= 0.40
        # - 上吊线前阳线占据优势: green_count_5 >= 3 且 ret_5 > 0
        # - 本根为上吊线: lower_r >= 0.50 (长下影), body_r <= 0.40 (实体不强求极小), upper_r <= 0.20 (上影较小)
        hm = sub[
            (sub["prev_is_green"] == True) & 
            (sub["prev_body_r"] >= 0.40) & 
            (sub["green_count_5"] >= 3) & 
            (sub["ret_5"] > 0) & 
            (sub["lower_r"] >= 0.50) & 
            (sub["body_r"] <= 0.40) & 
            (sub["upper_r"] <= 0.20)
        ]
        
        # 2. 倒垂线 (Inverted Hammer):
        # - 前一根必须是实体阴线: prev_is_green == False, prev_body_r >= 0.40
        # - 倒垂线前阴线占据优势: green_count_5 <= 2 且 ret_5 < 0
        # - 本根为倒垂线: upper_r >= 0.50 (长上影), body_r <= 0.40 (实体不强求极小), lower_r <= 0.20 (下影较小)
        ih = sub[
            (sub["prev_is_green"] == False) & 
            (sub["prev_body_r"] >= 0.40) & 
            (sub["green_count_5"] <= 2) & 
            (sub["ret_5"] < 0) & 
            (sub["upper_r"] >= 0.50) & 
            (sub["body_r"] <= 0.40) & 
            (sub["lower_r"] <= 0.20)
        ]
        
        hm_win = hm["next_is_bear"].mean() if len(hm) > 0 else 0
        ih_win = ih["next_is_bull"].mean() if len(ih) > 0 else 0
        
        print(f"\n--- 最近 {days} 天 ---")
        print(f"  【上吊线 Hanging Man (看跌 DOWN)】触发次数: {len(hm)}")
        if len(hm) > 0:
            print(f"    次根看跌胜率: {hm_win*100:.2f}%")
            print(f"    次根冲高惯性比例 (给到折价挂单点): {(hm['next_high_spike'] > 0.0003).mean()*100:.2f}%")
            
        print(f"  【倒垂线 Inverted Hammer (看涨 UP)】触发次数: {len(ih)}")
        if len(ih) > 0:
            print(f"    次根看涨胜率: {ih_win*100:.2f}%")
            print(f"    次根下探惯性比例 (给到折价挂单点): {(ih['next_low_dip'] > 0.0003).mean()*100:.2f}%")
