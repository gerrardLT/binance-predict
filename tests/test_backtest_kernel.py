"""科学回测内核单元测试（M1）：统计纯函数 + 事件引擎，不触网络。

- stats：wilson 已知值 / zbin 边界 / 二项检验方向性 / 功效预检判定 /
  多重检验门槛 / VR 对合成随机漫步 ≈1 与对动量序列 >1
- events：构造 5m K 验证破位切分、场景①②命中与 win 语义
- scene_params：JSON 往返与覆盖子集
"""

from __future__ import annotations

import random

from binance_predict.backtest.data import aggregate_15m
from binance_predict.backtest.events import build_events
from binance_predict.backtest.stats import (
    exact_binomial_p,
    multiple_testing_threshold,
    power_preflight,
    variance_ratio,
    wilson,
    zbin,
)
from binance_predict.services.scene_params import DEFAULT_SCENE_PARAMS, SceneParams


# ============================================================
# stats 纯函数
# ============================================================

def test_wilson_known_value() -> None:
    # p=0.636, n=462 → 95% CI ≈ [0.593, 0.677]（验证集场景①基准的量级）
    lo, hi = wilson(0.636, 462)
    assert 0.58 < lo < 0.61
    assert 0.66 < hi < 0.70


def test_zbin_boundaries() -> None:
    assert zbin(-5.0) == 0
    assert zbin(0.0) == 4
    assert zbin(0.33) == 5
    assert zbin(5.0) == 7


def test_binomial_directionality() -> None:
    # 显著偏离 50% → 小 p；对称不偏 → 大 p
    assert exact_binomial_p(300, 462) < 0.001
    assert exact_binomial_p(231, 462) > 0.9


def test_power_preflight_verdicts() -> None:
    # n=462 可检测下限 ≈5.8pp：场景① 的 13.6pp 效应检得出；2pp 的微弱声称检不出
    assert power_preflight(462, claimed_effect_pp=13.6)["verdict"] == "OK"
    assert power_preflight(462, claimed_effect_pp=2.0)["verdict"] == "INSUFFICIENT_POWER"
    assert power_preflight(0, claimed_effect_pp=20.0)["verdict"] == "INSUFFICIENT_POWER"


def test_multiple_testing_threshold_scales() -> None:
    t1 = multiple_testing_threshold(2.0, 1)
    t9 = multiple_testing_threshold(2.0, 9)
    assert t1["required_pp"] == 2.0
    assert t9["required_pp"] == 6.0  # √9 = 3 倍


def test_variance_ratio_random_walk_and_momentum() -> None:
    rng = random.Random(42)
    rw = [rng.gauss(0, 1) for _ in range(2000)]
    vr_rw = variance_ratio(rw, 2)["vr"]
    assert vr_rw is not None and 0.9 < vr_rw < 1.1  # 随机漫步 ≈ 1

    mom = []
    x = 0.0
    for _ in range(2000):
        x = 0.4 * x + rng.gauss(0, 1)  # 正自相关（动量）
        mom.append(x)
    vr_mom = variance_ratio(mom, 2)["vr"]
    assert vr_mom is not None and vr_mom > 1.2       # 动量 > 1


# ============================================================
# events 事件引擎（构造 K 线）
# ============================================================

def _mk_c5(base: float = 63000.0, n_cycles: int = 60, seed: int = 7) -> list[tuple]:
    """构造 60 个 15m 周期（180 根 5m）：先横盘 4h（48 根 close 恒定），
    再在最后一个周期冲高破位（bar0 创新高 + 收盘光头收阳）。"""
    rng = random.Random(seed)
    rows: list[tuple] = []
    t0 = 1_700_000_000_000 // 900_000 * 900_000  # 对齐周期边界的起点
    for i in range(48):  # 4h 横盘：close 均为 base
        ot = t0 + i * 300_000
        o = base + rng.uniform(-5, 5)
        c = base
        rows.append((ot, o, max(o, c) + 3, min(o, c) - 3, c, 10.0))
    # 之后的周期：温和震荡（不破位）
    for i in range(48, 177):
        ot = t0 + i * 300_000
        o = base + rng.uniform(-5, 5)
        c = o + rng.uniform(-5, 5)
        rows.append((ot, o, max(o, c) + 3, min(o, c) - 3, c, 10.0))
    # 最后一个周期（cyc=59）bar0：冲高破 48 根 close max（=base）并收在高点
    ot = t0 + 177 * 300_000
    spike = base * 1.002  # 远超 eps
    rows.append((ot, base, spike, base, spike, 50.0))
    for k in (1, 2):  # bar1/bar2 高位横住 → 15m 收阳光头
        ot = t0 + (177 + k) * 300_000
        rows.append((ot, spike, spike + 1, spike - 1, spike, 10.0))
    return rows


def test_build_events_detects_scene1() -> None:
    c5 = _mk_c5()
    now_ms = c5[-1][0] + 300_000 + 1
    res = build_events(c5, DEFAULT_SCENE_PARAMS, now_ms)
    events = res["events"]
    assert events, "应至少捕获一个破位事件"
    last = [e for e in events if e["side"] == "high"]
    assert last, "应有破阻力事件"
    e = last[-1]
    assert e["scene1"] is True        # 破阻力 + 收阳 + 光头
    assert e["close_pos"] >= 0.85
    # 次周期结果存在性（最后一周期无次周期 → has_next False，scene1 事件 win 不可判）
    assert e["has_next"] is False


def test_build_events_scene_semantics_with_next() -> None:
    # 在破位周期后补一个次周期（收阳）→ scene1 win = 次周期收阴 = False
    c5 = _mk_c5()
    t0 = c5[-1][0] + 300_000
    spike = c5[-1][4]
    for k in range(3):
        ot = t0 + k * 300_000
        c = spike + 5 if k < 2 else spike + 8  # 次周期收阳
        rows = (ot, spike if k == 0 else c - 1, c + 1, spike if k == 0 else c - 2, c, 10.0)
        c5.append(rows)
    now_ms = c5[-1][0] + 300_000 + 1
    res = build_events(c5, DEFAULT_SCENE_PARAMS, now_ms)
    e = [x for x in res["events"] if x["side"] == "high"][-1]
    assert e["has_next"] is True
    assert e["next_down"] is False
    # 场景① win 语义 = next_down → 此处应为输（次周期收阳）
    assert e["scene1"] is True


def test_aggregate_15m_drops_incomplete() -> None:
    # 7 根 5m（2 个完整周期 + 1 根散根）→ 只保留 2 个周期
    base = 1_700_000_000_000 // 900_000 * 900_000
    rows = [(base + i * 300_000, 1.0, 1.1, 0.9, 1.0, 5.0) for i in range(7)]
    agg = aggregate_15m(rows)
    assert len(agg["cycs"]) == 2


# ============================================================
# scene_params
# ============================================================

def test_scene_params_roundtrip_and_overrides() -> None:
    p = SceneParams(close_pos_min=0.9)
    j = p.to_params_json()
    assert SceneParams.from_params_json(j).close_pos_min == 0.9
    p2 = p.with_overrides({"vol_ratio_min": 2.5, "bogus_key": 1})
    assert p2.close_pos_min == 0.9        # 未覆盖键保持
    assert p2.vol_ratio_min == 2.5
    assert not hasattr(p2, "bogus_key")   # 未知键被忽略
    assert SceneParams.from_params_json(None) == DEFAULT_SCENE_PARAMS
