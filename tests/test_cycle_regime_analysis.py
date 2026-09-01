"""tests for scripts/local_cycle_regime_analysis.py（纯函数：标签器边界 /
阈值冻结只吃 IS 段 / group 统计与朴素实现对拍 / 月份键边界）。"""
import datetime as dt
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import local_cycle_regime_analysis as cra  # noqa: E402
from binance_predict.backtest.stats import wilson  # noqa: E402

START = 1_750_000_000_000 - (1_750_000_000_000 % 300_000)  # 对齐 5m 网格
N_BARS = 20 * 288                                        # 20 天 5m 根


@pytest.fixture(autouse=True)
def _loosen_frozen_assert(monkeypatch):
    """build_labels 内含冻结阈值断言（真数据口径）；合成数据放宽容差，
    只测结构不测数值。"""
    monkeypatch.setattr(cra, "ER_TOL", 1e9)
    monkeypatch.setattr(cra, "RV_TOL", 1e9)


def _synth_c5(n_bars: int = N_BARS, seed: int = 7) -> list[tuple]:
    """合成 5m K 线（正弦 + 小噪声），首末根对齐 15m 周期边界。"""
    rng = np.random.default_rng(seed)
    n = n_bars - n_bars % 3                       # 整 15m 周期
    t = START + 300_000 * np.arange(n)
    price = 100.0 + 2.0 * np.sin(np.arange(n) / 50.0) \
        + rng.normal(0, 0.05, n).cumsum() * 0.01
    price = np.maximum(price, 1.0)
    c5 = []
    for i in range(n):
        o = float(price[i])
        c = float(price[min(i + 1, n - 1)])
        h = max(o, c) * 1.0005
        lo = min(o, c) * 0.9995
        c5.append((int(t[i]), o, h, lo, c, 1.0))
    return c5


# ---------------- B 区：标签器边界 ----------------

def test_build_labels_warmup_and_is_end():
    c5 = _synth_c5()
    labels, meta = cra.build_labels(c5)
    # 暖机：ER/RV 需 672 根 15m 回看 → 前 W 根必无 er/rv 标签
    assert meta["n_warm"] >= cra.W
    for j in range(cra.W):
        assert labels["er_band"][j] == ""
        assert labels["rv_band"][j] == ""
    # 暖机之后（+24h ret24 回看余量）标签齐全（合成数据无缺口）
    j = cra.W + 300
    assert labels["er_band"][j] in ("趋势牛", "趋势熊", "过渡", "震荡")
    assert labels["rv_band"][j] in ("低", "中", "高")
    assert labels["ret24"][j] in ("上行", "下行", "震荡")
    # IS/OOS 分界 = 数据窗起点 + 360d（确定性，不随运行时刻漂移）
    assert meta["is_end"] == int(c5[0][0]) + 360 * 86_400_000
    # session 恒有值（纯日历，无暖机）
    assert all(s in ("asia", "europe", "america", "late") for s in labels["session"])
    # label_at：命中 / 暖机 / 越界三类边界
    label_at = meta["label_at"]
    ts_valid = meta["ts0"] + j * 900_000             # 暖机后有效周期
    assert label_at("er_band", ts_valid) == labels["er_band"][j]
    assert label_at("session", ts_valid) == labels["session"][j]
    assert label_at("er_band", meta["ts0"]) is None  # 暖机 → None
    assert label_at("er_band", meta["ts1"] + 10 * 900_000) is None  # 越界
    assert label_at("session", meta["ts0"]) in ("asia", "europe", "america", "late")


def test_ret24_threshold_boundary():
    """ret24 口径（±1%）与手工逐根计算对拍，并抽查 ±1% 边界归类。"""
    c5 = _synth_c5()
    labels, meta = cra.build_labels(c5)
    close_by_ts = {r[0]: r[4] for r in c5}
    n_check = 0
    for j, lab in enumerate(labels["ret24"]):
        if lab == "":
            continue
        ts_open = meta["ts0"] + j * 900_000
        base = close_by_ts.get(ts_open - 86_400_000)
        op = close_by_ts.get(ts_open - 300_000)
        if base is None or op is None:
            continue
        r24 = op / base - 1.0
        expect = ("上行" if r24 > cra.TREND_TH
                  else ("下行" if r24 < -cra.TREND_TH else "震荡"))
        assert lab == expect
        n_check += 1
        if n_check > 300:
            break
    assert n_check > 100


def test_thresholds_isolated_to_is_segment():
    """阈值冻结只吃 IS 段：数据窗 >360d 时，尾部（OOS 段）追加新数据 →
    分位数逐位不变（is_end = t5[0]+360d 固定，m_a 掩码只用 IS 段）。
    370d 合成数据 ≈ 106k 根 5m，单次构建秒级。"""
    c5_a = _synth_c5(n_bars=370 * 288, seed=11)
    _, meta_a = cra.build_labels(c5_a)

    rng = np.random.default_rng(12)
    c5_b = list(c5_a)
    last_t, last_c = c5_a[-1][0], c5_a[-1][4]
    for k in range(1, 3 * 288 + 1):                 # 尾部追加 3 天随机游走
        o = last_c
        c = max(o * (1.0 + rng.normal(0, 0.002)), 1.0)
        c5_b.append((last_t + 300_000 * k, o, max(o, c) * 1.0005,
                     min(o, c) * 0.9995, c, 1.0))
        last_c = c
    _, meta_b = cra.build_labels(c5_b)
    assert meta_b["is_end"] == meta_a["is_end"]
    for k in ("er50", "er75", "rv33", "rv67"):
        assert meta_b["quantiles"][k] == pytest.approx(meta_a["quantiles"][k],
                                                         abs=0.0, rel=0.0), k


# ---------------- C/D 区：group 统计与朴素实现对拍 ----------------

def _naive_seg(rs, base_p):
    n = len(rs)
    if n == 0:
        return {"n": 0}
    k = sum(r.win for r in rs)
    wr = k / n
    lo, hi = wilson(wr, n)
    out = {"n": n, "wins": k, "wr": round(wr, 4),
           "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    if base_p is not None:
        out["dev_pp"] = round((wr - base_p) * 100, 2)
    return out


def _naive_verdict(pooled, seg_is, seg_oos, base_p, req_pp):
    if pooled["n"] < cra.MIN_N:
        return "INSUFFICIENT_POWER"
    dis = seg_is.get("dev_pp") if seg_is.get("n") else None
    dos = seg_oos.get("dev_pp") if seg_oos.get("n") else None
    same_dir = (dis is not None and dos is not None and dis * dos > 0
                and abs(dis) >= 1.0 and abs(dos) >= 1.0)
    oos_ok = seg_oos.get("n", 0) >= cra.MIN_N
    cross = False
    if oos_ok and base_p is not None and dos is not None:
        cross = seg_oos["ci_lo"] > base_p if dos > 0 else seg_oos["ci_hi"] < base_p
    big = pooled.get("dev_pp") is not None and abs(pooled["dev_pp"]) >= req_pp
    if same_dir and oos_ok and cross and big:
        return "ADOPTED"
    if same_dir and oos_ok and cross:
        return "CANDIDATE"
    return "EXPLORE"


def _mk_recs(n_is, wr_is, n_oos, wr_oos, ts0=START, spread=10 * 86_400_000):
    recs = []
    for n, wr, off in ((n_is, wr_is, 0), (n_oos, wr_oos, spread)):
        wins = int(round(n * wr))
        for j in range(n):
            recs.append(cra.Rec("sig", ts0 + off + j * 60_000, "DOWN",
                                j < wins, None, None))
    return recs


@pytest.mark.parametrize("n_is,wr_is,n_oos,wr_oos,base,req", [
    (80, 0.60, 80, 0.62, 0.50, 9.17),    # 双向同向显著 → ADOPTED
    (80, 0.60, 80, 0.62, 0.55, 9.17),    # 偏离不够大 → CANDIDATE
    (80, 0.60, 80, 0.45, 0.50, 9.17),    # IS/OOS 反向 → EXPLORE
    (10, 0.80, 12, 0.75, 0.50, 9.17),    # n<30 → INSUFFICIENT_POWER
    (80, 0.60, 20, 0.65, 0.50, 9.17),    # OOS n<30 → EXPLORE
])
def test_group_cell_matches_naive(n_is, wr_is, n_oos, wr_oos, base, req):
    rs = _mk_recs(n_is, wr_is, n_oos, wr_oos)
    is_end = START + 5 * 86_400_000
    row = cra.group_cell("sig", "dim", "bin", rs, base, is_end, req)
    is_rs = [r for r in rs if r.ts_ms < is_end]
    oos_rs = [r for r in rs if r.ts_ms >= is_end]
    p, si, so = _naive_seg(rs, base), _naive_seg(is_rs, base), _naive_seg(oos_rs, base)
    assert row["n"] == p["n"] and row["wr"] == p["wr"]
    assert row["ci_lo"] == p["ci_lo"] and row["ci_hi"] == p["ci_hi"]
    assert row["dev_pp"] == p["dev_pp"]
    assert row["is"]["n"] == si["n"] and row["is"]["dev_pp"] == si.get("dev_pp")
    assert row["oos"]["n"] == so["n"] and row["oos"]["dev_pp"] == so.get("dev_pp")
    assert row["verdict"] == _naive_verdict(p, si, so, base, req)


def test_group_cell_quote_half_split():
    """is_end=None（报价族）→ 按样本时间中位切前后两个半段。"""
    rs = [cra.Rec("sig", START + j * 60_000, "DOWN", j % 2 == 0, None, None)
          for j in range(80)]
    base, req = 0.50, 9.17
    row = cra.group_cell("sig", "dim", "bin", rs, base, None, req)
    split = sorted(r.ts_ms for r in rs)[40]
    is_rs = [r for r in rs if r.ts_ms < split]
    oos_rs = [r for r in rs if r.ts_ms >= split]
    assert row["is"]["n"] == len(is_rs) == 40
    assert row["oos"]["n"] == len(oos_rs) == 40
    p, si, so = _naive_seg(rs, base), _naive_seg(is_rs, base), _naive_seg(oos_rs, base)
    assert row["verdict"] == _naive_verdict(p, si, so, base, req)


def test_breakeven_and_ev_family_split():
    # x4 族含 0.01 溢价；其余报价族无溢价（同 tsw.breakeven_of 口径）
    assert cra.breakeven_q("x4_v1", 0.49) == pytest.approx(0.50 / 0.98)
    assert cra.breakeven_q("quote_momentum_v1", 0.49) == pytest.approx(0.49 / 0.98)
    assert cra._ev_real("x4_v1", True, 0.49) == pytest.approx(0.98 / 0.50 - 1.0)
    assert cra._ev_real("quote_momentum_v1", True, 0.49) == pytest.approx(0.98 / 0.49 - 1.0)
    assert cra._ev_real("x4_v1", False, 0.30) == -1.0


# ---------------- 月份键边界 ----------------

def test_month_key_utc_boundaries():
    t0 = dt.datetime(2025, 12, 31, 23, 59, tzinfo=dt.timezone.utc)
    t1 = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    assert cra.month_key(int(t0.timestamp() * 1000)) == "2025-12"
    assert cra.month_key(int(t1.timestamp() * 1000)) == "2026-01"
