import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from scripts.calc_dual_tf_performance import extract_signals, calc_curve_stats

s_15m = extract_signals("15m", use_rsi_ih=False)
s_5m = extract_signals("5m", use_rsi_ih=True)

now = s_15m["dt"].max()
FEE = 0.98
PREMIUM = 0.01
PAYOFF_030 = FEE / (0.30 + PREMIUM) - 1.0

print("=== 15m vs 5m Breakdown under 0.30 Execution ===")
for days in [90, 180, 360, 720]:
    c = now - pd.Timedelta(days=days)
    sub15 = s_15m[(s_15m["dt"] >= c) & (s_15m["adverse_excursion"] >= 0.0004)]
    sub5 = s_5m[(s_5m["dt"] >= c) & (s_5m["adverse_excursion"] >= 0.0003)]
    
    pnl15 = np.where(sub15["is_win"], PAYOFF_030, -1.0)
    pnl5 = np.where(sub5["is_win"], PAYOFF_030, -1.0)
    
    t15, dd15, _, ml15, _, pf15, wr15, n15 = calc_curve_stats(pnl15, sub15["is_win"].values, sub15["dt"])
    t5, dd5, _, ml5, _, pf5, wr5, n5 = calc_curve_stats(pnl5, sub5["is_win"].values, sub5["dt"])
    
    print(f"\n[{days} Days]")
    print(f"  15m Alone: n={n15:3d} | Win={wr15*100:4.1f}% | PnL={t15:+6.1f} R | MaxDD={dd15:4.1f} R | MaxLoss={ml15} losses | PF={pf15:4.2f}")
    print(f"   5m Alone: n={n5:3d} | Win={wr5*100:4.1f}% | PnL={t5:+6.1f} R | MaxDD={dd5:4.1f} R | MaxLoss={ml5} losses | PF={pf5:4.2f}")
