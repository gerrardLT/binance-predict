"""发现流水线测试：防泄漏 / 统计口径回归 / 对照实现一致性 / 端到端冒烟。

对照基准：
- bh_fdr / run_block_ci 复刻 scripts/local_full_history_discovery.py 的原实现；
- variance_ratio 对照朴素 O(n·q) 循环实现；
- 打平胜率 ≈52.04% 与 ev() 的 (2%, 0.01) 口径。
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from binance_predict.backtest.stats import ev, variance_ratio
from binance_predict.discovery import (
    Klines, aggregate_to, bh_fdr, build_feature_matrix, build_targets, make_atoms,
    merge_r3, run_block_ci, run_combos, run_l1, run_oos, write_outputs,
)
from binance_predict.discovery.hypotheses import DEFAULTS
from binance_predict.discovery.l1_tester import ALL_TESTS_HEADER
from binance_predict.discovery.oos_validator import breakeven_win_rate
from binance_predict.discovery.report import REGISTRY_HEADER


BAR_MS = 300_000


def synth_klines(n: int = 5760, seed: int = 7) -> Klines:
    """20 天 5m 随机游走合成 K 线。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.int64) * BAR_MS + 1_700_000_000_000
    ret = rng.normal(0, 0.0012, n)
    c = 60_000 * np.exp(np.cumsum(ret))
    o = np.concatenate([[c[0]], c[:-1]])
    wig = np.abs(rng.normal(0, 0.0008, n)) * c
    h = np.maximum(o, c) + wig
    l = np.minimum(o, c) - wig
    v = rng.uniform(10, 100, n)
    cont = np.ones(n, dtype=bool)
    cont[0] = False
    return Klines(t=t, o=o, h=h, l=l, c=c, v=v, cont=cont)


# ---------------- 防泄漏 ----------------

def test_key_columns_not_degenerate():
    """关键序列特征不得退化为全 NaN（回归：_roll_sum 曾被首个 NaN 毒化整条后缀）。"""
    kl = synth_klines()
    fm = build_feature_matrix(kl, BAR_MS)
    for name in ("efficiency_8", "ret_3", "up_frac_5", "sma_dist_atr_20"):
        col = fm.cols[name].astype(float)
        assert np.isfinite(col).sum() > len(kl) * 0.5, f"{name} 退化为 NaN"
    eff = fm.cols["efficiency_8"]
    finite = eff[np.isfinite(eff)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0 + 1e-9


def test_no_lookahead_in_features():
    """扰动未来数据不得影响任何历史位置的特征值（前置窗口因果性）。"""
    kl = synth_klines()
    fm1 = build_feature_matrix(kl, BAR_MS)
    n = len(kl)
    m = n - 200
    kl2 = Klines(t=kl.t.copy(), o=kl.o.copy(), h=kl.h.copy(), l=kl.l.copy(),
                 c=kl.c.copy(), v=kl.v.copy(), cont=kl.cont.copy())
    rng = np.random.default_rng(3)
    for arr2, arr1 in ((kl2.c, kl.c), (kl2.h, kl.h), (kl2.l, kl.l), (kl2.v, kl.v)):
        arr2[m:] = arr1[m:] * rng.uniform(0.9, 1.1, n - m)
    fm2 = build_feature_matrix(kl2, BAR_MS)
    assert fm1.names == fm2.names
    for name in fm1.names:
        a, b = fm1.cols[name][:m], fm2.cols[name][:m]
        eq = (a == b) | (np.isnan(a.astype(float)) & np.isnan(b.astype(float)))
        assert eq.all(), f"特征 {name} 泄漏未来信息"


# ---------------- 高周期输入守卫（1h/4h 自聚合防护） ----------------

def test_feature_matrix_htf_input_no_crash():
    """1h/4h 输入时多周期共振自聚合无意义：不得抛异常，自共振列填空语义。"""
    kl5 = synth_klines()
    for bar_ms in (3_600_000, 14_400_000):
        klh = aggregate_to(kl5, bar_ms)
        fm = build_feature_matrix(klh, bar_ms)
        n = len(klh)
        for tf_ms, tag in ((3_600_000, "1h"), (14_400_000, "4h")):
            assert len(fm.cols[f"align_{tag}"]) == n
            if tf_ms <= bar_ms:  # 自共振/降周期无意义 → 全 False/0
                assert not fm.cols[f"align_{tag}"].any()
                assert (fm.cols[f"slot_in_{tag}"] == 0).all()


def test_align_and_slot_unchanged_for_5m():
    """5m 输入下 align/slot 列口径与守卫前一致（回归）。"""
    kl = synth_klines()
    fm = build_feature_matrix(kl, BAR_MS)
    t = kl.t
    # slot_in_4h 口径：(t // bar_ms) % 48
    assert (fm.cols["slot_in_4h"] == ((t // BAR_MS) % 48).astype(np.int16)).all()
    assert (fm.cols["slot_in_1h"] == ((t // BAR_MS) % 12).astype(np.int16)).all()
    # align_1h 与手工聚合对照（末根已收盘的 1h 方向）
    kl1h = aggregate_to(kl, 3_600_000)
    hdir = np.sign(kl1h.c - kl1h.o)
    j = np.searchsorted(kl1h.t, t - 3_600_000, side="right") - 1
    ok = j >= 0
    hd = np.where(ok, hdir[np.clip(j, 0, len(kl1h) - 1)], np.nan)
    expect = np.where(ok, np.sign(kl.c - kl.o) * hd > 0, False)
    assert (fm.cols["align_1h"] == expect).all()


# ---------------- 统计口径回归 ----------------

def test_breakeven_and_ev_caliber():
    be = breakeven_win_rate()
    assert abs(be - 0.5204) < 2e-4  # 打平胜率 ≈52.04%
    assert abs(ev(be, 0.5)) < 1e-9
    assert ev(0.615, 0.5) > 0.05  # 既有最强发现口径应为显著正 EV


def test_variance_ratio_matches_naive_loop():
    rng = np.random.default_rng(11)
    r = rng.normal(0, 0.001, 800) + 0.0001
    q = 6

    # 朴素 O(n·q) 参考实现（原逐元素口径）
    n = len(r)
    mu = float(np.mean(r))
    d = r - mu
    var1 = float((d * d).sum()) / (n - 1)
    m = q * (n - q + 1) * (1 - q / n)
    rq = np.array([r[i:i + q].sum() - q * mu for i in range(n - q + 1)])
    varq = float((rq * rq).sum()) / m
    vr = varq / var1
    z = np.sqrt(n * q) * (vr - 1) / np.sqrt(2 * (q - 1) * (2 * q - 1) / (3 * q))
    denom = float((d * d).sum())
    theta = sum((2 * (q - k) / q) ** 2 * (float((d[:-k] * d[k:]).sum()) / denom) ** 2
                for k in range(1, q))
    zs = np.sqrt(n) * (vr - 1) / np.sqrt(theta)

    got = variance_ratio(list(r), q)
    assert got["vr"] == pytest.approx(vr, abs=5e-4)
    assert got["z"] == pytest.approx(z, abs=5e-3)
    assert got["z_star"] == pytest.approx(zs, abs=5e-3)


# ---------------- 对照脚本版实现 ----------------

def _ref_bh_fdr(pvals, q=0.10):
    n = len(pvals)
    if n == 0:
        return []
    arr = np.asarray(pvals, dtype=float)
    order = np.argsort(arr)
    ranked = arr[order]
    thresh = q * np.arange(1, n + 1) / n
    below = ranked <= thresh
    if not below.any():
        return [False] * n
    cutoff = ranked[np.max(np.nonzero(below)[0])]
    return list(arr <= cutoff)


def test_bh_fdr_matches_reference():
    rng = np.random.default_rng(5)
    pvals = list(rng.uniform(0, 1, 500)) + list(rng.uniform(0, 0.001, 30))
    assert bh_fdr(pvals, 0.1) == _ref_bh_fdr(pvals, 0.1)


def _ref_run_block_ci(hit_cycs, hits, b=3000, seed=11):
    runs_v, runs_n = [], []
    start = 0
    for i in range(1, len(hit_cycs) + 1):
        if i == len(hit_cycs) or hit_cycs[i] != hit_cycs[i - 1] + 1:
            runs_v.append(int(hits[start:i].sum()))
            runs_n.append(i - start)
            start = i
    v = np.asarray(runs_v, dtype=float)
    w = np.asarray(runs_n, dtype=float)
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, len(v), size=(b, len(v)))
    means = v[sel].sum(axis=1) / w[sel].sum(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def test_run_block_ci_matches_reference():
    rng = np.random.default_rng(9)
    idx = np.sort(rng.choice(3000, 400, replace=False))
    wins = rng.integers(0, 2, 400).astype(float)
    got = run_block_ci(idx, wins, b=800, seed=11)
    ref = _ref_run_block_ci(idx, wins, b=800, seed=11)
    assert got[0] == pytest.approx(ref[0], abs=1e-12)
    assert got[1] == pytest.approx(ref[1], abs=1e-12)


# ---------------- 端到端冒烟（20 天合成数据，宽松漏斗） ----------------

SMOKE_CFG = {
    **DEFAULTS,
    "min_lift_pp": 0.0, "fdr_alpha": 1.0, "min_validation_lift_pp": -1e9,
    "min_interaction_gain_pp": -1e9, "n_min_l2": 60, "n_min_l3": 40,
    "n_min_holdout": 20, "max_l1": 15, "max_l2": 20, "max_l3": 20,
    "shortlist_per_target": 8,
}


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom")
    warnings.filterwarnings("ignore", message="invalid value encountered")
    kl = synth_klines()
    n = len(kl)
    fm = build_feature_matrix(kl, BAR_MS)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    disc = np.zeros(n, dtype=bool)
    disc[:i1] = True
    atr = np.abs(kl.h - kl.l)
    atr_abs = np.maximum(atr, 1e-9)
    tg = build_targets(kl.t, kl.o, kl.h, kl.l, kl.c, kl.cont, [1, 2], atr_abs)
    atoms = make_atoms(fm, disc)
    l1 = run_l1(fm, atoms, tg, n, SMOKE_CFG)
    cb = run_combos(fm, atoms, tg, l1, n, SMOKE_CFG)
    r3 = [{"id": "R3-T1", "family": "冒烟", "target_family": "reversal",
           "expect": "down", "mechanism": "test",
           "atoms": [["doji", "==", "True"], ["inside_bar", "==", "True"]]}]
    merge_r3(fm, tg, r3, cb, n, SMOKE_CFG)
    oos = run_oos(tg, kl.t.astype("datetime64[ms]"), cb, n,
                  fm.cols.get("atr_pctile_4320"), SMOKE_CFG)
    outdir = str(tmp_path_factory.mktemp("disc_smoke"))
    run_config = {"tf": "5m", "data_summary": {"rows": n}, "budget": SMOKE_CFG,
                  "total_tests": sum(v["n_tests"] for v in l1.values())}
    paths = write_outputs(outdir, run_config=run_config, fm=fm, l1_results=l1,
                          combo_results=cb, oos_results=oos,
                          rounds_json=r3, atoms_by_round={"R1": len(atoms)})
    return paths


def test_smoke_artifacts_exist(smoke_run):
    for key in ("run_config", "feature_manifest", "all_tests",
                "discovery_registry", "summary", "report", "hypotheses_registry"):
        assert os.path.exists(smoke_run[key]), f"缺少产物 {key}"


def test_smoke_headers_isomorphic(smoke_run):
    with open(smoke_run["all_tests"], encoding="utf-8") as f:
        assert next(csv.reader(f)) == ALL_TESTS_HEADER
    with open(smoke_run["discovery_registry"], encoding="utf-8") as f:
        assert next(csv.reader(f)) == REGISTRY_HEADER


def test_smoke_funnel_executed(smoke_run):
    # continuation 目标必须进入组合层（补空白核心验收项）
    with open(smoke_run["all_tests"], encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        levels = {(r["target"], r["level"]) for r in rdr}
    assert any(t.startswith("continuation") and lv in ("L2", "L3")
               for t, lv in levels), "continuation 组合层未执行"
