import pandas as pd
import numpy as np
from PIL import Image

# Let's inspect the user's screenshots with the exact user definition:
# 1. Hanging Man (上吊线):
#    - Preceded by bullish move (阳线占据优势)
#    - Immediate previous bar (Bar N-1) MUST be a solid bullish/green candle (实体阳线，最好是光头阳)
#    - Signal Bar (Bar N): Hanging Man (小到中等实体在上方，长下影线，无/小上影线)
#    - Prediction / Bet: 次周期看跌 (DOWN) / 挂 0.20 买跌
#
# 2. Inverted Hammer (倒垂线):
#    - Preceded by bearish move (阴线占据优势)
#    - Immediate previous bar (Bar N-1) MUST be a solid bearish/red candle (实体阴线，最好是光头实体阴)
#    - Signal Bar (Bar N): Inverted Hammer (小到中等实体在下方，长上影线，无/小下影线)
#    - Prediction / Bet: 次周期看涨 (UP) / 挂 0.20 买涨

# Let's inspect the 4 images' exact candles and see where the user drew the arrows!
print("User criteria understood completely:")
print("1. Bar N-1 MUST be a solid candle in the trend direction (光头实体阳 / 光头实体阴)")
print("2. Prior trend: trend direction dominates (阳线优势 / 阴线优势)")
print("3. Bar N: Hanging Man (长下影，实体不一定非常小) / Inverted Hammer (长上影，实体不一定非常小)")
