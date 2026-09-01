#!/usr/bin/env python
"""弱收盘上吊线(15m)后最优买跌时点：反弹触及+x×ATR 时的真实收阴胜率 vs 市场隐含胜率。

框架：
- 场景：A=弱收盘上吊线收盘后，下一根15m若先反弹（未先跌破-x）触及 +x×ATR，
  在触及时刻买「该15m收跌」。
- p_real(x,j)：A后样本中，第 j 根5m首次触及+x 者最终15m收阴的比例（j=0,1,2）。
- p_base(x,j)：全体15m棒同样条件（先触+x 未先触-x、触及5m序号 j）的收阴比例
  —— 近似预测市场在该时点的公允隐含胜率（价格位置定价、无形态信息）。
- 公平赔率下 EV = p_real / p_base - 1（正=有超额期望）。
- 另报触及点→15m收盘的平均变动（×ATR，做空视角）。
"""
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from binance_predict.discovery import load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402
from double_top_probe_720d import BAR_MS, data_csv_path  # noqa: E402

XS = (0.1, 0.15, 0.25, 0.4, 0.5)


def main():
    kl = load_klines_csv(data_csv_path("15m"), BAR_MS["15m"])
    k5 = load_klines_csv(data_csv_path("5m"), BAR_MS["5m"])
    atr = atr_series(kl)
    o, h, l, c, t = kl.o, kl.h, kl.l, kl.c, kl.t
    dir_, cont, n = np.sign(c - o), kl.cont, len(c)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    atr_s = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)
    top20 = np.full(n, np.nan)
    top20[19:] = sliding_window_view(h, 20).max(axis=1)
    hm = ((body <= 0.3 * atr_s) & (lower >= 2.0 * body) & (lower >= 0.3 * atr_s)
          & (upper <= 0.15 * atr_s))
    pos = np.zeros(n, dtype=bool)
    pos[19:] = c[19:] >= top20[19:] - 0.75 * atr_s[19:]
    hm &= pos & np.isfinite(atr_s)
    A = np.flatnonzero(hm & cont & np.roll(cont, 1))
    A = A[(A > 20) & (A < n - 8)]
    clv = (c - l) / np.maximum(h - l, 1e-9)
    Aw = A[clv[A] <= 0.75]

    s15 = np.searchsorted(k5.t, t)
    ok_map = np.zeros(n, dtype=bool)
    ok_map[:-2] = ((s15[:-2] + 3 <= len(k5.t))
                   & (k5.t[s15[:-2]] == t[:-2])
                   & (k5.t[s15[:-2] + 1] == t[:-2] + 300_000)
                   & (k5.t[s15[:-2] + 2] == t[:-2] + 600_000))

    def touch_info(j15, x):
        """15m棒 j15：先触+x（未先触-x）返回 (j5, 触点→收盘变动/ATR)；否则 None。"""
        s = s15[j15]
        o15 = o[j15]
        lev_up, lev_dn = o15 + x * atr[j15], o15 - x * atr[j15]
        hit_up = hit_dn = False
        for q in range(3):
            hu = k5.h[s + q] >= lev_up
            hd = k5.l[s + q] <= lev_dn
            if hu and not hit_dn:
                return (q, (c[j15] - lev_up) / atr[j15])
            if hd and not hit_up:
                hit_dn = True
            if hu:
                # 已先触下、后触上：不算干净反弹
                return None
            if hd:
                hit_dn = True
        return None

    # ---- 基线：全体 15m 棒（向量化近似：同 touch_info 逻辑，抽样 8000 根足够稳）----
    rng = np.random.default_rng(7)
    pool = np.flatnonzero(ok_map & (np.arange(n) < n - 1))
    base_idx = rng.choice(pool, size=min(8000, len(pool)), replace=False)
    base_hits = {x: {j: [] for j in range(3)} for x in XS}
    for j15 in base_idx:
        for x in XS:
            r = touch_info(int(j15), x)
            if r:
                base_hits[x][r[0]].append(r[1])

    print("A后下一根15m：先反弹触及+x 后买「跌」的性价比表")
    print(f"{'x(×ATR)':>8} {'j5':>3} {'n':>4} | {'p_real':>7} {'p_base':>7} {'EV':>7} | "
          f"{'触点→收均':>9}")
    print("-" * 66)
    nxt = Aw[(Aw + 1 < n) & ok_map[np.minimum(Aw + 1, n - 1)]]
    best = []
    for x in XS:
        for j in range(3):
            ys, ns = [], 0
            for j15 in nxt + 1:
                r = touch_info(int(j15), x)
                if r and r[0] == j:
                    ys.append(r[1])
                    ns += 1
            p_real = np.mean([v < 0 for v in ys]) if ys else float("nan")
            bv = base_hits[x][j]
            p_base = np.mean([v < 0 for v in bv]) if bv else float("nan")
            ev = p_real / p_base - 1 if (bv and ys) else float("nan")
            avg_move = np.mean(ys) if ys else float("nan")
            if ys and bv and len(ys) >= 5:
                print(f"{x:>8} {j:>3} {len(ys):>4} | {p_real:7.1%} {p_base:7.1%} "
                      f"{ev:+7.1%} | {avg_move:+9.3f}")
                best.append((ev, x, j, len(ys), p_real, p_base, avg_move))
    print("-" * 66)
    best.sort(reverse=True)
    print("EV 排行（n>=5）:")
    for ev, x, j, m, pr, pb, am in best[:6]:
        print(f"  x={x} j={j}: n={m} p_real={pr:.1%} p_base={pb:.1%} EV={ev:+.1%} "
              f"触点→收盘平均 {am:+.3f}×ATR")

    # 分组聚合（j 合并，增大格子）
    print("\n按 x 聚合（j=0..2 合并，触及越晚剩时间越少，谨慎合并仅作参考）:")
    for x in XS:
        ys, bs = [], []
        for j15 in nxt + 1:
            r = touch_info(int(j15), x)
            if r:
                ys.append(r[1])
        for j15 in base_idx:
            r = touch_info(int(j15), x)
            if r:
                bs.append(r[1])
        if len(ys) >= 5 and len(bs) > 50:
            pr, pb = np.mean([v < 0 for v in ys]), np.mean([v < 0 for v in bs])
            print(f"  x={x}: n={len(ys)} p_real={pr:.1%} p_base={pb:.1%} "
                  f"EV={pr / pb - 1:+.1%} 触点→收盘均 {np.mean(ys):+.3f}×ATR")

    # 对照：不等待反弹，A 收盘立刻买跌（下一根15m收阴）
    im = nxt + 1
    p_im = np.mean(dir_[im] < 0)
    print(f"\n对照：A收盘立即买跌（不等反弹）: p_real={p_im:.1%}  市场隐含≈50%  "
          f"EV≈{p_im / 0.5 - 1:+.1%}  (n={len(im)})")


if __name__ == "__main__":
    main()
