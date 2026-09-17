import pandas as pd
import numpy as np

# Let's inspect the EV of 5m Hanging Man & Inverted Hammer with 0.20 Limit Orders!
# In 5m:
# When Hanging Man prints:
# Next bar high spike > 0.0003 (giving discount to DOWN) is ~56%
# When Inverted Hammer prints:
# Next bar low dip > 0.0003 (giving discount to UP) is ~55%~62%
# If order fills at 0.20: Payoff is +366.7%!

FEE = 0.98
PREMIUM = 0.01
QUOTE = 0.20
PAYOFF = FEE / (QUOTE + PREMIUM) - 1.0

# If win rate is 53% ~ 57%:
# EV = win_rate * 3.667 - (1 - win_rate) * 1.0
print("=== 5M Expected Value (EV) Analysis with 0.20 Limit Orders ===")
for wr in [0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]:
    ev = wr * PAYOFF - (1.0 - wr)
    print(f"Win Rate = {wr*100:.0f}% --> Payoff = +{PAYOFF*100:.1f}%, Single-Trade EV = {ev*100:+.1f}%")
