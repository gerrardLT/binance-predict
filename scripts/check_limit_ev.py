import pandas as pd
import numpy as np

# Let's inspect EV for 0.20 limit order entries!
# Payoff formula:
# When entry quote = 0.20, fee = 0.98, premium = 0.01:
# Payoff = 0.98 / (0.20 + 0.01) - 1 = 0.98 / 0.21 - 1 = +3.667 (+366.7%)
# Loss = -1.0 (-100%)
# Net EV = Win_Rate * 3.667 - (1 - Win_Rate) * 1.0 = 4.667 * Win_Rate - 1.0
# Break-even win rate is only 1.0 / 4.667 = 21.43%!

FEE = 0.98
PREMIUM = 0.01
QUOTE = 0.20
PAYOFF = FEE / (QUOTE + PREMIUM) - 1.0

def calc_ev(win_rate):
    return win_rate * PAYOFF - (1.0 - win_rate)

print(f"Break-even win rate at 0.20 limit order: {1.0 / (PAYOFF + 1.0)*100:.2f}%")
print(f"EV at 50% win rate: {calc_ev(0.50):+.3f} (i.e. {calc_ev(0.50)*100:+.1f}%)")
print(f"EV at 55% win rate: {calc_ev(0.55):+.3f} (i.e. {calc_ev(0.55)*100:+.1f}%)")
print(f"EV at 60% win rate: {calc_ev(0.60):+.3f} (i.e. {calc_ev(0.60)*100:+.1f}%)")
print(f"EV at 65% win rate: {calc_ev(0.65):+.3f} (i.e. {calc_ev(0.65)*100:+.1f}%)")
