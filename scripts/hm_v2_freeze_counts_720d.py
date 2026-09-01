"""冻结 hm_touch_down_v2 的 720d 硬闸门计数（门禁：非下跌段 ∧ 非低波）。

门禁口径（与 scripts/hm_slice_followup_720d.py 逐字一致）：
- ret24 = c[i]/c[i-96] − 1 > −0.01（排除下跌段：过去 24h 跌 ≤−1%）
- atr_ratio = atr[i] / median(atr[i-96:i]) ≥ 0.8（排除低波；窗口内有限值须 >20）
输出：门禁后触发数、触价数、胜率——供检测器硬闸门测试冻结。
"""
import io
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from binance_predict.discovery import load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402
from double_top_probe_720d import BAR_MS, data_csv_path  # noqa: E402

X = 0.25
W = 96


def main():
    kl = load_klines_csv(data_csv_path("15m"), BAR_MS["15m"])
    k5 = load_klines_csv(data_csv_path("5m"), BAR_MS["5m"])
    atr = atr_series(kl)
    o, h, l, c, t = kl.o, kl.h, kl.l, kl.c, kl.t
    cont, n = kl.cont, len(c)

    top20 = np.full(n, np.nan)
    top20[19:] = sliding_window_view(h, 20).max(axis=1)
    body, upper = np.abs(c - o), h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    hm = ((body <= 0.3 * atr_s) & (lower >= 2.0 * body) & (lower >= 0.3 * atr_s)
          & (upper <= 0.15 * atr_s))
    pos = np.zeros(n, dtype=bool)
    pos[19:] = c[19:] >= top20[19:] - 0.75 * atr_s[19:]
    hm &= pos & np.isfinite(atr_s)
    A = np.flatnonzero(hm & cont & np.roll(cont, 1))
    A = A[(A > 20) & (A < n - 8)]
    clv = (c - l) / np.maximum(h - l, 1e-9)
    Aw = A[clv[A] <= 0.75]
    print(f"v1 触发: {len(Aw)}")

    # 门禁（与切片跟进脚本逐字一致）
    ret24 = np.full(n, np.nan)
    ret24[W:] = c[W:] / c[:-W] - 1
    atr_med = np.full(n, np.nan)
    for i in range(W, n):
        w = atr[i - W:i]
        w = w[np.isfinite(w)]
        if len(w) > 20:
            atr_med[i] = np.median(w)
    gate = np.zeros(n, dtype=bool)
    finite = np.isfinite(ret24) & np.isfinite(atr_med) & (atr_med > 0) & np.isfinite(atr)
    gate[finite] = (ret24[finite] > -0.01) & (atr[finite] / atr_med[finite] >= 0.8)

    Ag = Aw[gate[Aw]]
    print(f"v2 门禁后触发: {len(Ag)}（冻结数）")

    s15 = np.searchsorted(k5.t, t)
    ok_map = np.zeros(n, dtype=bool)
    ok_map[:-2] = ((s15[:-2] + 3 <= len(k5.t))
                   & (k5.t[s15[:-2]] == t[:-2])
                   & (k5.t[s15[:-2] + 1] == t[:-2] + 300_000)
                   & (k5.t[s15[:-2] + 2] == t[:-2] + 600_000))

    def touch(j15):
        s = s15[j15]
        o15 = o[j15]
        lu, ld = o15 + X * atr[j15], o15 - X * atr[j15]
        hd = False
        for q in range(3):
            hu, hd2 = k5.h[s + q] >= lu, k5.l[s + q] <= ld
            if hu and not hd:
                return (q, (c[j15] - lu) / atr[j15])
            if hd2 and not hu:
                hd = True
            if hu:
                return None
            if hd2:
                hd = True
        return None

    nxt = Ag[(Ag + 1 < n) & ok_map[np.minimum(Ag + 1, n - 1)]]
    ys = []
    for j15 in nxt + 1:
        r = touch(int(j15))
        if r:
            ys.append(r[1] < 0)
    print(f"v2 触价数: {len(ys)}（冻结数）| 胜率 {np.mean(ys):.1%} "
          f"({sum(ys)}/{len(ys)})（基准 69.0%）")


if __name__ == "__main__":
    main()
