"""OOS 验证：冻结 shortlist 的 holdout 终验（只触碰一次）+ 稳健性三件套 + 裁决。

- holdout 判据：方向与发现一致 + n≥n_min_holdout + 点估计 > 打平线 + Wilson 下界对照；
- 稳健性：按月一致性（≥2/3 月份同号）、波动率三分位分组、run 块自助 95% CI；
- 8×90 天 walk-forward 段报表（全局折边界只算一次，逐折段内统计）；
- 裁决：ROBUST / PROMISING / WEAK / REJECT（沿用既有词汇表），方向翻转标 flipped 降级。

纯函数无 I/O。
"""
from __future__ import annotations

import numpy as np

from ..backtest.stats import FEE, PREMIUM, ev
from .hypotheses import DEFAULTS
from .l1_tester import seg_stats


def run_block_ci(hit_idx: np.ndarray, wins: np.ndarray, b: int = 3000,
                 seed: int = 11) -> tuple[float, float] | tuple[float, float]:
    """连续命中合并为 run（处理相邻重叠依赖），run 级自助 95% CI。

    hit_idx 为命中位置的升序索引，wins 为对应胜负（0/1）。
    复刻 local_full_history_discovery.py L146-161 口径（向量化分段）。
    """
    if len(hit_idx) == 0:
        return (np.nan, np.nan)
    brk = np.flatnonzero(np.diff(hit_idx) != 1) + 1
    v = np.add.reduceat(wins.astype(float), np.r_[0, brk])
    w = np.diff(np.r_[0, brk, len(hit_idx)]).astype(float)
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, len(v), size=(b, len(v)))
    means = v[sel].sum(axis=1) / w[sel].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def breakeven_win_rate(entry: float = 0.5) -> float:
    """费后打平胜率：ev(p, e)=0 的解（(2%, 0.01) 口径下 ≈52.04%）。"""
    odds = (1 - FEE) / (entry + PREMIUM) - 1.0
    return 1.0 / (1.0 + odds)


def monthly_consistency(t: np.ndarray, mask: np.ndarray, wins: np.ndarray,
                        sign: float) -> dict:
    """按月一致性：命中月份中胜率偏离 0.5 与发现同号的比例。"""
    months = t.astype("datetime64[ms]").astype("datetime64[M]")
    uniq = np.unique(months[mask])
    if len(uniq) == 0:
        return {"n_periods": 0, "frac_aligned": np.nan, "detail": []}
    aligned, detail = 0, []
    for m in uniq:
        sel = mask & (months == m)
        n = int(sel.sum())
        if n < 20:
            continue
        lift = float(wins[sel].mean()) - 0.5
        ok = np.sign(lift) == sign and lift != 0
        aligned += int(ok)
        detail.append((str(m), n, round(lift * 100, 2)))
    k = len(detail)
    return {"n_periods": k, "frac_aligned": aligned / k if k else np.nan,
            "detail": detail}


def regime_consistency(vol_pctile: np.ndarray, mask: np.ndarray,
                       wins: np.ndarray, sign: float) -> dict:
    """波动率三分位分组一致性（低/中/高波各组的 lift 是否同号）。"""
    thirds = [(-np.inf, 1 / 3), (1 / 3, 2 / 3), (2 / 3, np.inf)]
    out = {"n_groups": 0, "frac_aligned": np.nan, "detail": []}
    aligned = 0
    for lo, hi in thirds:
        sel = mask & (vol_pctile >= lo) & (vol_pctile < hi)
        n = int(sel.sum())
        if n < 20:
            continue
        lift = float(wins[sel].mean()) - 0.5
        out["detail"].append((f"{lo:.2f}-{hi:.2f}", n, round(lift * 100, 2)))
        out["n_groups"] += 1
        aligned += int(np.sign(lift) == sign and lift != 0)
    if out["n_groups"]:
        out["frac_aligned"] = aligned / out["n_groups"]
    return out


def walk_forward(t: np.ndarray, mask: np.ndarray, wins: np.ndarray,
                 n_folds: int = 8) -> list[dict]:
    """8×90 天等长折报表（折边界全局只算一次，逐折段内统计）。"""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    edges = np.linspace(0, len(t), n_folds + 1).astype(int)
    folds = []
    for f in range(n_folds):
        a, b = edges[f], edges[f + 1]
        sel = mask.copy()
        sel[:a] = False
        sel[b:] = False
        n = int(sel.sum())
        if n == 0:
            folds.append({"fold": f + 1, "n": 0, "win_rate": np.nan, "lift_pp": np.nan})
            continue
        wr = float(wins[sel].mean())
        folds.append({"fold": f + 1, "n": n, "win_rate": wr,
                      "start": str(t[a].astype("datetime64[ms]").astype("datetime64[D]")),
                      "end": str(t[min(b, len(t) - 1)].astype("datetime64[ms]").astype("datetime64[D]"))})
    return folds


def _verdict(lift_h: float, n_h: int, retention: float, temporal: float,
             regime: float, flipped: bool) -> str:
    """沿用既有裁决词汇表。"""
    if n_h <= 0 or np.isnan(lift_h):
        return "REJECT"
    if flipped:
        return "WEAK" if lift_h > 0 else "REJECT"
    if (retention >= 0.5 and n_h >= 120
            and temporal >= 2 / 3 and regime >= 2 / 3 and lift_h > 0):
        return "ROBUST"
    if retention >= 0.3 and temporal >= 0.5 and lift_h > 0:
        return "PROMISING"
    if lift_h > 0:
        return "WEAK"
    return "REJECT"


def run_oos(tg, t: np.ndarray, combo_results: dict, n: int,
            vol_pctile: np.ndarray | None = None,
            cfg: dict | None = None) -> dict:
    """对每个目标的冻结 shortlist 做 holdout 终验 + 稳健性 + 裁决。

    vol_pctile：全量滚动波动分位列（regime 一致性用）；缺省则按命中集内波动现算。
    返回 {target: [row, ...]}，row 为 registry 行（含 holdout_*、verdict、经济账）。
    """
    c = {**DEFAULTS, **(cfg or {})}
    i1 = int(n * c["discovery_frac"])
    i2 = int(n * (c["discovery_frac"] + c["validation_frac"]))
    be = breakeven_win_rate()
    out: dict[str, list[dict]] = {}
    for tname in tg.names:
        ts = tg.items[tname]
        rows = []
        for r in combo_results.get(tname, {}).get("shortlist", []):
            sign_d = np.sign(r["discovery_lift_pp"])
            # ---- holdout（冻结后只触碰一次）----
            base_h = float(ts.win[i2:n][ts.valid[i2:n]].sum()) / max(1, int(ts.valid[i2:n].sum()))
            hs = seg_stats(ts, r["_mask"], i2, n, base_h)
            n_h, lift_h = hs["n"], hs["lift_pp"]
            wr_h = hs["win_rate"]
            flipped = bool(np.isfinite(lift_h) and np.sign(lift_h) != sign_d and lift_h != 0)
            retention = (abs(lift_h) / abs(r["discovery_lift_pp"])
                         if np.isfinite(lift_h) and r["discovery_lift_pp"] != 0 else np.nan)
            degradation = (float(r["discovery_lift_pp"]) - float(lift_h)
                           if np.isfinite(lift_h) else np.nan)
            hold_dir_ok = not flipped and n_h >= c["n_min_holdout"]
            above_be = bool(np.isfinite(wr_h) and wr_h > be)
            # ---- 稳健性三件套（全量命中上评估）----
            full_mask = r["_mask"] & ts.valid
            hit_idx = np.flatnonzero(full_mask)
            wins = ts.win[full_mask].astype(np.float64)
            ci_lo, ci_hi = run_block_ci(hit_idx, wins)
            temp = monthly_consistency(t, full_mask, ts.win.astype(np.float64), sign_d)
            vp = vol_pctile if vol_pctile is not None else np.full(n, np.nan)
            regime = regime_consistency(vp, full_mask, ts.win.astype(np.float64), sign_d)
            wf = walk_forward(t, full_mask, ts.win.astype(np.float64))
            temporal = temp["frac_aligned"] if np.isfinite(temp["frac_aligned"]) else 0.0
            regime_f = regime["frac_aligned"] if np.isfinite(regime["frac_aligned"]) else 0.0
            verdict = _verdict(float(lift_h) if np.isfinite(lift_h) else np.nan,
                               n_h, retention if np.isfinite(retention) else 0.0,
                               temporal, regime_f, flipped)
            # ---- 经济账（(2%, 0.01) 口径）----
            p = wr_h if np.isfinite(wr_h) else np.nan
            ev_val = ev(p, 0.5) if np.isfinite(p) else np.nan
            odds = (1 - FEE) / (0.5 + PREMIUM) - 1.0
            kelly = (p - (1 - p) / odds) if np.isfinite(p) else np.nan
            rows.append({
                **{k: v for k, v in r.items() if k != "_mask"},
                "discovery_baseline": np.nan,
                "holdout_n": n_h, "holdout_baseline": base_h,
                "holdout_win_rate": wr_h, "holdout_lift_pp": lift_h,
                "holdout_ci_low": hs["ci_low"], "holdout_ci_high": hs["ci_high"],
                "holdout_p_value": hs["p_value"],
                "holdout_payoff_ratio": hs["payoff_ratio"],
                "holdout_expectancy": hs["expectancy"],
                "holdout_mfe_mean_atr": hs["mfe_mean_atr"],
                "holdout_mae_mean_atr": hs["mae_mean_atr"],
                "block_ci_low": ci_lo, "block_ci_high": ci_hi,
                "temporal_consistency": round(temporal, 3),
                "regime_consistency": round(regime_f, 3),
                "oos_degradation_pp": degradation, "oos_retention": retention,
                "flipped": flipped, "holdout_n_ok": hold_dir_ok,
                "above_breakeven": above_be, "breakeven": be,
                "ev_at_0.50": ev_val, "kelly": kelly,
                "verdict": verdict,
                "score": (float(lift_h) * np.sqrt(n_h) * min(retention, 1.0)
                          if np.isfinite(lift_h) and np.isfinite(retention) else 0.0),
                "walk_forward": wf,
                "monthly_detail": temp["detail"], "regime_detail": regime["detail"],
            })
        rows.sort(key=lambda x: -x["score"])
        out[tname] = rows
    return out
