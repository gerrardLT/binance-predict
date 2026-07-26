#!/usr/bin/env python3
"""预测市场价格校准扫描（离线、确定性、无 LLM、无网络）。

直接检验「热门-冷门偏差」（favorite-longshot bias）——fade 策略 EV 为正的
唯一合理成因。相比策略回测，校准检验统计效力更高：不引入策略自由度，
只问一个问题——

    在决策时刻 t，热门方价格为 p 时，热门方的真实获胜频率是否 == p？

若极端价格桶系统性出现「胜率 < 价格」，则热门被高估、反向（fade）存在
结构性溢价；反之若校准良好，fade 的正 EV 点估计只是噪声。

口径（与 decision_bench 经济口径一致）：
  - 市场判定 = sign(actual_return)（NOISE 标签窗口市场照样结算），=0 剔除。
  - 热门方 = 决策时刻 up_price 与 down_price 中较高的一侧；用真实采样价格
    （非 chance 近似），故 vig（up+down-1）被显式量化。
  - 每桶给出热门真实胜率的 Wilson 95% 区间；「胜率-价格」gap 的显著性以
    区间是否覆盖桶均价判定。
  - EV：热门 EV = win/p_fav - 1；fade EV = (1-win)/p_dog - 1（费前），
    另列扣 2% 手续费口径。

用法：
    python scripts/price_calibration_scan.py --from-file windows_with_price.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

DECISION_POINTS = (60.0, 90.0, 120.0)   # 秒（窗口内相对时刻）
BUCKET_EDGES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.001)
FEE = 0.02  # 实测费率（用户实盘确认，与 decision_bench 一致）


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例 Wilson 95% 置信区间。total<=0 返回 (0, 1)。"""
    if total <= 0:
        return 0.0, 1.0
    n = float(total)
    phat = correct / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _price_at(curve: list, start_ms: int, t_sec: float) -> float | None:
    """决策时刻价格：取 rel_t <= t_sec 的最后一个采样价（不偷看未来）。"""
    best = None
    for p in curve or []:
        rel = (p.get("t", 0) - start_ms) / 1000.0
        if rel <= t_sec and p.get("v") is not None:
            best = float(p["v"])
    return best


def scan(windows: list[dict], proxy_chance: bool = False) -> dict:
    """proxy_chance=True 时用 up%/100 近似价格（历史窗口无真实价格曲线时的
    全量口径，与 decision_bench 一致；真实买价通常略高，溢价另做敏感性）。"""
    out: dict = {}
    for t_sec in DECISION_POINTS:
        rows = []
        vig_sum = 0.0
        vig_n = 0
        for w in windows:
            ret = w.get("actual_return")
            if ret is None or ret == 0:
                continue  # 市场判定不明，剔除
            start_ms = w.get("start_time", 0)
            if proxy_chance:
                up_c = _price_at(w.get("curve_up_pct"), start_ms, t_sec)
                dn_c = _price_at(w.get("curve_down_pct"), start_ms, t_sec)
                up_p = up_c / 100.0 if up_c is not None else None
                dn_p = dn_c / 100.0 if dn_c is not None else None
            else:
                up_p = _price_at(w.get("curve_up_price"), start_ms, t_sec)
                dn_p = _price_at(w.get("curve_down_price"), start_ms, t_sec)
            if up_p is None or dn_p is None or up_p <= 0 or dn_p <= 0:
                continue
            vig_sum += up_p + dn_p - 1.0
            vig_n += 1
            if up_p >= dn_p:
                fav_price, dog_price, fav_win = up_p, dn_p, ret > 0
            else:
                fav_price, dog_price, fav_win = dn_p, up_p, ret < 0
            rows.append((fav_price, dog_price, fav_win))

        buckets = []
        for lo, hi in zip(BUCKET_EDGES, BUCKET_EDGES[1:]):
            sel = [r for r in rows if lo <= r[0] < hi]
            n = len(sel)
            if n == 0:
                continue
            wins = sum(1 for r in sel if r[2])
            win_rate = wins / n
            avg_fav = sum(r[0] for r in sel) / n
            avg_dog = sum(r[1] for r in sel) / n
            ci_lo, ci_hi = wilson_ci(wins, n)
            fav_ev = win_rate / avg_fav - 1.0
            dog_ev = (1.0 - win_rate) / avg_dog - 1.0
            buckets.append({
                "bucket": f"{lo:.2f}~{hi:.2f}",
                "n": n,
                "avg_fav_price": round(avg_fav, 4),
                "fav_win_rate": round(win_rate, 4),
                "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
                "gap": round(win_rate - avg_fav, 4),
                # gap 显著（价格在胜率 CI 之外）才可下「错价」结论
                "gap_significant": not (ci_lo <= avg_fav <= ci_hi),
                "fav_ev_raw": round(fav_ev, 4),
                "fav_ev_fee": round((win_rate * (1 - FEE)) / avg_fav - 1.0, 4),
                "avg_dog_price": round(avg_dog, 4),
                "fade_ev_raw": round(dog_ev, 4),
                "fade_ev_fee": round(((1 - win_rate) * (1 - FEE)) / avg_dog - 1.0, 4),
            })
        out[f"t{int(t_sec)}"] = {
            "n_windows": len(rows),
            "avg_vig": round(vig_sum / vig_n, 4) if vig_n else None,
            "buckets": buckets,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", required=True)
    ap.add_argument("--proxy-chance", action="store_true",
                    help="用 up%%/100 近似价格（历史窗口无真实价格曲线时）")
    ap.add_argument("--out", default="output/price_calibration_scan.json")
    args = ap.parse_args()

    with open(args.from_file, encoding="utf-8") as f:
        windows = json.load(f)
    result = scan(windows, proxy_chance=args.proxy_chance)

    for key, blk in result.items():
        print("=" * 100)
        print(f"决策点 {key}  n={blk['n_windows']}  平均vig(up+down-1)={blk['avg_vig']}")
        print("-" * 100)
        header = (f"{'价格桶':<12}{'n':>6}{'均价':>8}{'真实胜率':>10}"
                  f"{'CI95':>18}{'gap':>8}{'显著':>6}"
                  f"{'热门EV费后':>12}{'fade均价':>10}{'fadeEV费前':>12}{'fadeEV费后':>12}")
        print(header)
        for b in blk["buckets"]:
            print(f"{b['bucket']:<14}{b['n']:>6}{b['avg_fav_price']:>9.3f}"
                  f"{b['fav_win_rate']:>10.4f}"
                  f"{str(b['ci95']):>20}{b['gap']:>+9.4f}"
                  f"{'是' if b['gap_significant'] else '否':>5}"
                  f"{b['fav_ev_fee']:>+12.4f}{b['avg_dog_price']:>10.3f}"
                  f"{b['fade_ev_raw']:>+12.4f}{b['fade_ev_fee']:>+12.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[已写入] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
