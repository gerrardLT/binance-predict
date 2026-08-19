# -*- coding: utf-8 -*-
"""方案 A：4h 位势 regime 条件化 X4（探索性调节分析，非新闸门）。

问题：X4（收阳 & end≤40 → 次窗 DOWN）的 63.5% 胜率是否随 K 线位势变化？
假说（S1×X4 机制叠加）：收阳发生在 4h 高位（连续上攻后）→ 人群情绪却压
DOWN（end≤40）= 更强的"不信"错位 → 次窗 DOWN 命中更高；低位收阳的错位
则可能只是噪声。

位势口径（与线上 FakeBreakoutDetector 对齐）：
  closes = sentiment_windows.exit_price 序列（detector._refresh_levels 同源）
  pos4h = (exit − min48) / (max48 − min48)，roll 48 窗 = 4h，含当前窗
  （detector POS4H_WINDOW 为 15m×16 根；此处 5m×48 窗，等价 4h）

纪律：X4 已是主信号，本分析为调节分解——只看分桶胜率/EV 与剂量趋势，
不新增闸门不做多重校正选择；IS/OOS 分列展示但 OOS 桶 n 小仅供参考。
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from local_sentiment_curve_discovery import ev_eval  # noqa: E402
from local_curve_pure_dim import build_pure  # noqa: E402
from local_misalignment_scan import extend  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

POS_WIN = 48          # 4h = 48 个 5m 窗 closes
BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]  # 低位/中位/高位


def main() -> int:
    W = sorted(json.load(open("sentiment_windows.json", encoding="utf-8")),
               key=lambda w: int(w["start_time"]))
    recs = extend(build_pure(W), W)
    split = int(W[int(len(W) * 0.7)]["start_time"])
    n = len(recs)

    # ---- pos4h：roll 48 窗 closes（含当前），detector 同源 ----
    # exit_price 在原始窗 dict 上（recs 不保留该键；recs 与 W 等长同序，直接取 W）
    closes = np.array([w.get("exit_price") or np.nan for w in W], dtype=float)
    pos4h = np.full(n, np.nan)
    for j in range(n):
        lo = max(0, j - POS_WIN + 1)
        seg = closes[lo:j + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= POS_WIN // 2 and seg.max() > seg.min():
            pos4h[j] = (closes[j] - seg.min()) / (seg.max() - seg.min())

    # ---- X4 mask（主口径 end≤40 收阳）----
    x4 = np.array([r["x4"] for r in recs])
    nxt_up = np.array([1.0 if (r.get("next_out") or "") == "UP" else
                       (0.0 if (r.get("next_out") or "") == "DOWN" else np.nan)
                       for r in recs])
    has_next = np.array([r["has_next"] for r in recs])
    is_oos = np.array([r["start"] >= split for r in recs])
    valid = x4 & has_next & ~np.isnan(nxt_up) & ~np.isnan(pos4h)
    print(f"[数据] {n} 窗 | X4 且可判胜负且 pos4h 有效：{int(valid.sum())} 注"
          f"（pos4h 覆盖 {int((~np.isnan(pos4h)).sum())}/{n}）")

    def wr(m: np.ndarray) -> tuple[float, int]:
        k = int((nxt_up[m] == 0).sum())  # 押 DOWN 命中 = 次窗 DOWN
        nn = int(m.sum())
        return (k / nn if nn else float("nan")), nn

    # ---- 全样本基线 + 分桶 ----
    w0, n0 = wr(valid)
    print(f"\n[基线] X4 全部：押DOWN 命中 {w0:.1%} (n={n0})")

    for lo, hi in BUCKETS:
        m = valid & (pos4h >= lo) & (pos4h < hi)
        w_is, n_is = wr(m & ~is_oos)
        w_oos, n_oos = wr(m & is_oos)
        w_all, n_all = wr(m)
        ev = ev_eval([j + 1 for j in range(n) if m[j] and j + 1 < n],
                     ["DOWN"] * int(m.sum()), W)
        evs = ev.get("ev_2_1", float("nan"))
        print(f"  pos4h [{lo:.1f},{hi:.1f})  IS {w_is:.1%}(n={n_is:>3}) | "
              f"OOS {w_oos:.1%}(n={n_oos:>2}) | 全 {w_all:.1%}(n={n_all:>3}) | "
              f"EV={evs:+.3f} 实价{ev.get('n_real', 0)}/{ev.get('n', 0)}")

    # ---- 剂量趋势（Spearman：pos4h vs 次窗 DOWN 命中）----
    from scipy.stats import spearmanr
    rho, pv = spearmanr(pos4h[valid], 1 - nxt_up[valid])
    print(f"\n[剂量] Spearman(pos4h, 次窗DOWN命中) rho={rho:+.3f} p={pv:.3f}"
          f"（正 = 位势越高 X4 越灵）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
