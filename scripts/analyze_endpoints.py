import cv2
import numpy as np
from PIL import Image

# Let's inspect the exact neighborhood of each arrow endpoint in each image
# We will inspect:
# 1. What candles are around x=200..350 in Image 1
# 2. What candles are around x=300..420 in Image 2
# 3. What candles are around x=300..420 in Image 3
# 4. What candles are around x=380..620 in Image 4

def analyze_neighborhood(im_path, x_center, y_center, radius_x=60, radius_y=120):
    im = Image.open(im_path).convert("RGB")
    arr = np.array(im)
    h, w, _ = arr.shape
    
    x1, x2 = max(0, x_center - radius_x), min(w, x_center + radius_x)
    y1, y2 = max(0, y_center - radius_y), min(h, y_center + radius_y)
    
    crop = arr[y1:y2, x1:x2]
    # Candle pixels
    g_mask = (crop[:, :, 1] > 160) & (crop[:, :, 0] < 120)
    r_mask = (crop[:, :, 0] > 180) & (crop[:, :, 1] < 100)
    # Check if there is red arrow pixels here:
    arrow_mask = (crop[:, :, 0] > 160) & (crop[:, :, 1] < 90) & (crop[:, :, 2] < 90)
    # Exclude arrow from candle mask
    g_mask = g_mask & (~arrow_mask)
    r_mask = r_mask & (~arrow_mask)
    
    return crop, g_mask, r_mask, arrow_mask, (x1, y1, x2, y2)

for idx, (name, arrow_pt) in enumerate([
    ("138_246", (241, 794)),
    ("138_246", (298, 634)),
    ("139_246", (368, 533)),
    ("139_246", (311, 334)),
    ("140_246", (374, 797)),
    ("140_246", (325, 588)),
    ("141_246", (405, 569)),
    ("141_246", (599, 623)),
]):
    path = rf"C:\Users\gerrard\Desktop\k\微信图片_20260909171949_{name}.png"
    crop, g_mask, r_mask, a_mask, bbox = analyze_neighborhood(path, arrow_pt[0], arrow_pt[1], 40, 80)
    print(f"Image {name} near ({arrow_pt[0]}, {arrow_pt[1]}): green_px={np.sum(g_mask)}, red_candle_px={np.sum(r_mask)}, arrow_px={np.sum(a_mask)}")
