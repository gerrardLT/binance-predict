import pandas as pd
import numpy as np

def analyze_sentiment_reversals(tf="15m"):
    print(f"================================================================")
    print(f"             ANALYZING {tf.upper()} CANDLESTICK REVERSALS        ")
    print(f"================================================================")
    
    df = pd.read_csv(f"output/klines_{tf}_720d.csv")
    df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    
    # K 线几何特征
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_r"] = df["body"] / np.where(df["range"] > 0, df["range"], 1.0)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_r"] = df["upper_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
    df["lower_r"] = df["lower_wick"] / np.where(df["range"] > 0, df["range"], 1.0)
    df["is_green"] = df["close"] >= df["open"]
    
    # 波动率与成交量基准
    tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))))
    df["atr_14"] = tr.rolling(14).mean()
    df["vol_ma_14"] = df["volume"].rolling(14).mean()
    df["rel_range"] = df["range"] / np.where(df["atr_14"] > 0, df["atr_14"], 1.0)
    df["rel_vol"] = df["volume"] / np.where(df["vol_ma_14"] > 0, df["vol_ma_14"], 1.0)
    
    # 前一根 Bar N-1
    df["prev_is_green"] = df["is_green"].shift(1)
    df["prev_body"] = df["body"].shift(1)
    df["prev_range"] = df["range"].shift(1)
    df["prev_body_r"] = df["body_r"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_open"] = df["open"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["prev_vol"] = df["volume"].shift(1)
    df["prev_rel_range"] = df["prev_range"] / df["atr_14"].shift(1)
    
    # 趋势/背景特征
    df["ret_3"] = (df["close"].shift(1) - df["close"].shift(4)) / df["close"].shift(4)
    df["ret_5"] = (df["close"].shift(1) - df["close"].shift(6)) / df["close"].shift(6)
    df["prior_high_max_4"] = df["high"].shift(1).rolling(4).max()
    df["prior_low_min_4"] = df["low"].shift(1).rolling(4).min()
    df["is_prominent_high"] = df["prev_high"] >= df["prior_high_max_4"]
    df["is_prominent_low"] = df["prev_low"] <= df["prior_low_min_4"]
    df["prior_body_max_3"] = df["body"].shift(2).rolling(3).max()
    df["is_longest_body"] = df["prev_body"] >= df["prior_body_max_3"]
    
    # 孕线判定 (Inside Bar)
    df["is_full_inside"] = (df["high"] <= df["prev_high"] * 1.0002) & (df["low"] >= df["prev_low"] * 0.9998)
    
    # 吞没/包雷判定 (Engulfing / Outside Bar)
    df["is_engulfing"] = (df["high"] >= df["prev_high"]) & (df["low"] <= df["prev_low"])
    
    # 次根收益表现
    # 1. 纯方向 (下一根是阳线还是阴线)
    df["next_is_bull"] = df["close"].shift(-1) > df["open"].shift(-1)
    df["next_is_bear"] = df["close"].shift(-1) < df["open"].shift(-1)
    # 2. 次根收益率 (基于开盘到收盘)
    df["next_ret"] = (df["close"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    # 3. 次根极端冲高/下探 (微观折价空间)
    df["next_high_spike"] = (df["high"].shift(-1) - df["open"].shift(-1)) / df["open"].shift(-1)
    df["next_low_dip"] = (df["open"].shift(-1) - df["low"].shift(-1)) / df["open"].shift(-1)
    
    now = df["dt"].max()
    cutoffs = {
        "360D": now - pd.Timedelta(days=360),
        "720D": now - pd.Timedelta(days=720)
    }
    
    # 定义各类核心反转形态
    patterns = {}
    
    # 1. 经典单针探底 Pinbar / 锤子线 (Hammer)
    # 底部背景，长下影，小实体，几乎无上影
    patterns["1. 经典长下影探底锤子线 (Hammer)"] = {
        "direction": "UP",
        "cond": (df["ret_3"] < -0.001) & 
                (df["lower_r"] >= 0.60) & 
                (df["body_r"] <= 0.25) & 
                (df["upper_r"] <= 0.15) & 
                (df["rel_range"] >= 1.0)
    }
    
    # 2. 经典冲高受阻射击之星 (Shooting Star)
    # 顶部背景，长上影，小实体，几乎无下影
    patterns["2. 经典长上影冲高射击之星 (Shooting Star)"] = {
        "direction": "DOWN",
        "cond": (df["ret_3"] > 0.001) & 
                (df["upper_r"] >= 0.60) & 
                (df["body_r"] <= 0.25) & 
                (df["lower_r"] <= 0.15) & 
                (df["rel_range"] >= 1.0)
    }
    
    # 3. 巨幅爆量大长针 (Extreme Climax Pinbar - 上影空头)
    patterns["3. 极限量能天线长上影 (Climax Shooting Star)"] = {
        "direction": "DOWN",
        "cond": (df["upper_r"] >= 0.60) & 
                (df["body_r"] <= 0.30) & 
                (df["rel_range"] >= 1.5) & 
                (df["rel_vol"] >= 2.0)
    }
    
    # 4. 巨幅爆量大长针 (Extreme Climax Pinbar - 下影多头)
    patterns["4. 极限量能金针探底 (Climax Hammer)"] = {
        "direction": "UP",
        "cond": (df["lower_r"] >= 0.60) & 
                (df["body_r"] <= 0.30) & 
                (df["rel_range"] >= 1.5) & 
                (df["rel_vol"] >= 2.0)
    }
    
    # 5. 实盘验证的孕线上吊线 (Ver 2 Hanging Man inside Mother)
    patterns["5. 孕线长下影上吊线 (Ver 2 Inside Hanging Man)"] = {
        "direction": "DOWN",
        "cond": (df["prev_is_green"] == True) & 
                (df["prev_body_r"] >= 0.40) & 
                (df["is_prominent_high"]) & 
                (df["is_longest_body"]) & 
                (df["is_full_inside"]) & 
                (df["lower_r"] >= 0.45) & 
                (df["body_r"] <= 0.45) & 
                (df["upper_r"] <= 0.25)
    }

    # 6. 实盘验证的孕线倒垂线 (Ver 2 Inverted Hammer inside Mother)
    patterns["6. 孕线长上影倒垂线 (Ver 2 Inside Inverted Hammer)"] = {
        "direction": "UP",
        "cond": (df["prev_is_green"] == False) & 
                (df["prev_body_r"] >= 0.40) & 
                (df["is_prominent_low"]) & 
                (df["is_longest_body"]) & 
                (df["is_full_inside"]) & 
                (df["upper_r"] >= 0.45) & 
                (df["body_r"] <= 0.45) & 
                (df["lower_r"] <= 0.25)
    }
    
    # 7. 看涨吞没 (Bullish Engulfing) - 前大阴，本根实体包覆前根实体且收大阳
    patterns["7. 强力看涨多头吞没 (Bullish Engulfing)"] = {
        "direction": "UP",
        "cond": (df["prev_is_green"] == False) & 
                (df["prev_body_r"] >= 0.40) & 
                (df["is_green"] == True) & 
                (df["body_r"] >= 0.60) & 
                (df["open"] <= df["prev_close"]) & 
                (df["close"] >= df["prev_open"]) & 
                (df["rel_range"] >= 1.2)
    }

    # 8. 看跌吞没 (Bearish Engulfing) - 前大阳，本根实体包覆前根实体且收大阴
    patterns["8. 强力看跌空头吞没 (Bearish Engulfing)"] = {
        "direction": "DOWN",
        "cond": (df["prev_is_green"] == True) & 
                (df["prev_body_r"] >= 0.40) & 
                (df["is_green"] == False) & 
                (df["body_r"] >= 0.60) & 
                (df["open"] >= df["prev_close"]) & 
                (df["close"] <= df["prev_open"]) & 
                (df["rel_range"] >= 1.2)
    }
    
    # 9. 极端衰竭大阴线后的反转阳线 (Exhaustion Dump Reversal)
    # 前一根是巨幅放量暴跌大阴线（超跌），本根收十字星或探底回升
    patterns["9. 暴跌衰竭底分型 (Exhaustion Dump Reversal)"] = {
        "direction": "UP",
        "cond": (df["prev_is_green"] == False) & 
                (df["prev_rel_range"] >= 2.0) & 
                (df["prev_body_r"] >= 0.65) & 
                (df["lower_r"] >= 0.40)
    }

    # 10. 暴涨衰竭顶分型 (Exhaustion Pump Reversal)
    # 前一根是巨幅放量暴涨大阳线（超买），本根收长上影
    patterns["10. 暴涨力竭顶分型 (Exhaustion Pump Reversal)"] = {
        "direction": "DOWN",
        "cond": (df["prev_is_green"] == True) & 
                (df["prev_rel_range"] >= 2.0) & 
                (df["prev_body_r"] >= 0.65) & 
                (df["upper_r"] >= 0.40)
    }

    # 扫描与计算
    results = []
    for name, p_info in patterns.items():
        cond = p_info["cond"]
        direction = p_info["direction"]
        
        row = {"Pattern": name, "Target_Direction": direction}
        for period_name, cutoff in cutoffs.items():
            sub_cond = cond & (df["dt"] >= cutoff) & (df.index < len(df) - 1)
            sub = df[sub_cond]
            count = len(sub)
            if count > 0:
                if direction == "UP":
                    win_rate = sub["next_is_bull"].mean()
                    mean_ret = sub["next_ret"].mean() * 100
                    adv_opp = (sub["next_low_dip"] >= 0.0015).mean() * 100
                else:
                    win_rate = sub["next_is_bear"].mean()
                    mean_ret = -sub["next_ret"].mean() * 100
                    adv_opp = (sub["next_high_spike"] >= 0.0015).mean() * 100
            else:
                win_rate, mean_ret, adv_opp = 0, 0, 0
                
            row[f"{period_name}_Count"] = count
            row[f"{period_name}_WinRate"] = win_rate * 100
            row[f"{period_name}_MeanRet%"] = mean_ret
            row[f"{period_name}_Dip%"] = adv_opp
            
        results.append(row)
        
    res_df = pd.DataFrame(results)
    return res_df

if __name__ == "__main__":
    for tf in ["15m", "5m"]:
        res = analyze_sentiment_reversals(tf)
        print(res.to_string(index=False))
