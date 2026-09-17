import cv2
import numpy as np
from PIL import Image

# Let's inspect each of the 4 images across the ENTIRE chart width to see the full price path
# For each image, let's find the y-center of candles from left to right (x=50 to x=650)
for idx, name in enumerate(["138_246", "139_246", "140_246", "141_246"]):
    path = rf"C:\Users\gerrard\Desktop\k\微信图片_20260909171949_{name}.png"
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    
    # Chart area y=200..800
    chart = arr[200:800, 50:650]
    
    # Arrow mask
    r, g, b = chart[:, :, 0].astype(int), chart[:, :, 1].astype(int), chart[:, :, 2].astype(int)
    arrow_mask = (r > 160) & (g < 90) & (b < 90) & ((r - g) > 80) & ((r - b) > 80)
    
    # Candle mask
    g_mask = (g > 150) & (r < 130) & (~arrow_mask)
    r_mask = (r > 170) & (g < 110) & (~arrow_mask)
    candle_mask = g_mask | r_mask
    
    # Find all candles across x
    col_sum = np.sum(candle_mask, axis=0)
    active_cols = np.where(col_sum > 5)[0]
    clusters = []
    if len(active_cols) > 0:
        cur = [active_cols[0]]
        for c in active_cols[1:]:
            if c <= cur[-1] + 2:
                cur.append(c)
            else:
                clusters.append(cur)
                cur = [c]
        clusters.append(cur)
        
    print(f"\n==================== Image {idx+1} ({name}) Full Chart Trend ====================")
    print(f"Total candles detected: {len(clusters)}")
    
    # Print each candle's (x, y_high, y_low, color)
    # Note: smaller y means HIGHER price (screen coordinates)
    candle_records = []
    for c in clusters:
        sub_mask = candle_mask[:, c]
        ys = np.where(sub_mask.any(axis=1))[0]
        if len(ys) == 0: continue
        high_y = ys.min() + 200
        low_y = ys.max() + 200
        mid_x = (c[0] + c[-1]) / 2 + 50
        n_g = np.sum(g_mask[:, c])
        n_r = np.sum(r_mask[:, c])
        color = "G" if n_g > n_r else "R"
        candle_records.append((mid_x, high_y, low_y, color))
        
    # Find which candle is closest to the arrow tip
    # We know arrow tips:
    # Image 1: x~240..300, y~630..800
    # Let's print the trajectory of prices around the arrow!
    print("Sample of candles:")
    for mid_x, high_y, low_y, color in candle_records[::max(1, len(candle_records)//15)]:
        print(f"  x={mid_x:.0f}: {color} high_y={high_y} low_y={low_y}")
