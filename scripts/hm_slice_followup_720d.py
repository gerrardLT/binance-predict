"""HM 切片跟进：①ret24×波动 交叉格 ②时序分段稳健性（480d/240d）。

假设（来自第一轮切片，后验生成，需分段验证）：
- 下跌段（ret24≤−1%）边际为负（41.7% < 基线 46.1%）——高位衰竭形态在下跌途中等
  于下跌中继，应排除；注意这与 v4 的 regime 门禁方向相反
- 低波环境（ATR/前24h中位<0.8）边际为负（25.0%）——触点位太近，信号退化为噪声
"""
import io
import os
import sys
from datetime import datetime, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from binance_predict.backtest.stats import wilson  # noqa: E402
from binance_predict.discovery import load_klines_csv  # noqa: E402
from binance_predict.discovery.features import atr_series  # noqa: E402
from double_top_probe_720d import BAR_MS, data_csv_path  # noqa: E402

X = 0.25


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

    ret24 = np.full(n, np.nan)
    ret24[96:] = c[96:] / c[:-96] - 1
    atr_med = np.full(n, np.nan)
    for i in range(96, n):
        w = atr[i - 96:i]
        w = w[np.isfinite(w)]
        if len(w) > 20:
            atr_med[i] = np.median(w)
    atr_ratio = np.where(atr_med > 0, atr / atr_med, np.nan)

    def regime(i):  # 形态根收盘时点
        return ("跌" if ret24[i] <= -0.01 else "涨" if ret24[i] >= 0.01 else "平")

    def volreg(i):
        r = atr_ratio[i]
        return ("低" if r < 0.8 else "高" if r > 1.2 else "常")

    nxt = Aw[(Aw + 1 < n) & ok_map[np.minimum(Aw + 1, n - 1)]]
    hm_rows = []
    for j15 in nxt + 1:
        r = touch(int(j15))
        if r:
            i = int(j15) - 1
            hm_rows.append((i, r[1] < 0))
    pool = np.flatnonzero(ok_map & (np.arange(n) < n - 1))
    base_rows = []
    for j15 in pool:
        r = touch(int(j15))
        if r:
            base_rows.append((int(j15) - 1, r[1] < 0))

    def stat(rows, sel):
        ws = [w for i, w in rows if sel(i)]
        nb = len(ws)
        if nb == 0:
            return None
        p = float(np.mean(ws))
        lo, hi = wilson(p, nb)
        return nb, p, lo, hi

    # ---- ① ret24 × 波动 交叉格 ----
    print("① ret24 × 波动 交叉格（HM 触价样本；括号内为基线池同格胜率）")
    print(f"{'':>6}" + "".join(f"{v+'波':>22}" for v in ("低", "常", "高")))
    for g in ("跌", "平", "涨"):
        cells = []
        for v in ("低", "常", "高"):
            sel = lambda i, g=g, v=v: regime(i) == g and volreg(i) == v  # noqa: E731
            s_hm = stat(hm_rows, sel)
            s_bs = stat(base_rows, sel)
            if s_hm is None:
                cells.append(f"{'—':>22}")
            else:
                nb, p, lo, hi = s_hm
                pb = s_bs[1] if s_bs else float("nan")
                txt = "n=%d %.0f%%[%.0f%%,%.0f%%]" % (nb, p * 100, lo * 100, hi * 100)
                txt += "(%.0f%%)" % (pb * 100)
                cells.append(f"{txt:>22}")
        print(f"{g+'段':>6}" + "".join(cells))

    # ---- 候选过滤：非下跌段 ∧ 非低波 ----
    sel_keep = lambda i: regime(i) != "跌" and volreg(i) != "低"  # noqa: E731
    s_hm, s_bs = stat(hm_rows, sel_keep), stat(base_rows, sel_keep)
    print(f"\n候选过滤（非下跌段 ∧ 非低波）: n={s_hm[0]} p_real={s_hm[1]:.1%} "
          f"[{s_hm[2]:.1%},{s_hm[3]:.1%}] | 基线 {s_bs[1]:.1%} "
          f"| EV={s_hm[1] / s_bs[1] - 1:+.1%}")
    sel_drop = lambda i: not sel_keep(i)  # noqa: E731
    s_hm2 = stat(hm_rows, sel_drop)
    print(f"被排除部分: n={s_hm2[0]} p_real={s_hm2[1]:.1%} "
          f"[{s_hm2[2]:.1%},{s_hm2[3]:.1%}]")

    # ---- ② 时序分段：480d 训练段 / 240d 验证段 ----
    t_cut = t[0] + 480 * 86_400_000
    cut_i = int(np.searchsorted(t, t_cut))

    def d(ts_ms):
        return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%y-%m-%d")

    print(f"\n② 时序分段（切分点 {d(t_cut)}）")
    for name, sel in (("前480d", lambda i: i < cut_i), ("后240d", lambda i: i >= cut_i)):
        s_all = stat(hm_rows, sel)
        s_kep = stat(hm_rows, lambda i, sel=sel: sel(i) and sel_keep(i))
        b_kep = stat(base_rows, lambda i, sel=sel: sel(i) and sel_keep(i))
        print(f"{name}: 总量 n={s_all[0]} p={s_all[1]:.1%} | "
              f"过滤后 n={s_kep[0]} p={s_kep[1]:.1%} [{s_kep[2]:.1%},{s_kep[3]:.1%}] "
              f"基线 {b_kep[1]:.1%}")


if __name__ == "__main__":
    main()
