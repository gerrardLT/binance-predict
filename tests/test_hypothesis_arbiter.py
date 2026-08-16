"""科学裁决器单元测试（M3）：四层硬门禁的边界判定（纯函数，不触网络/DB）。

G1 样本量 / G2 改善幅度（Bonferroni 校正门槛）/ G3 最差月稳健性 / G4 功效预检。
"""

from __future__ import annotations

from binance_predict.services.hypothesis_arbiter import (
    BASE_MARGIN_PP,
    MIN_VALIDATION_N,
    HypothesisArbiter,
)


def _stats(n: int, p: float, worst: float | None = None) -> dict:
    return {"n": n, "p": p, "ci": [p - 0.05, p + 0.05], "worst_month_p": worst}


def _pair(s1: dict, s2: dict) -> dict:
    return {"scene1": s1, "scene2": s2}


BASE1 = _stats(n=150, p=0.620, worst=0.54)
BASE2 = _stats(n=149, p=0.564, worst=0.44)


def _arbiter() -> HypothesisArbiter:
    return HypothesisArbiter()


def test_gate1_insufficient_n_rejects() -> None:
    # n=59 < 60 → G1 拒（即使胜率更高）
    hypo = _pair(_stats(59, 0.70, 0.60), BASE2)
    v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2),
                                {"required_pp": 2.0}, claimed_pp=10.0)
    assert v.passed is False
    assert any("n=59" in r for r in v.reasons)


def test_gate2_margin_boundary() -> None:
    mt = {"required_pp": 2.0}
    # 改善 +1.9pp（0.639）→ 拒；+2.1pp（0.641）→ 过（场景②与基线全同不设障）
    for p_exp, expect in ((0.639, False), (0.641, True)):
        hypo = _pair(_stats(150, p_exp, 0.54), BASE2)
        v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2), mt, claimed_pp=5.0)
        g2 = [g for g in v.gates if g["gate"] == "G2-scene1"]
        assert g2 and g2[0]["passed"] is expect, f"p={p_exp} 应为 {expect}"


def test_gate3_worst_month_regression_rejects() -> None:
    # 均值改善但最差月从 54% 崩到 46%（跌 8pp > 5pp 容差）→ G3 拒
    hypo = _pair(_stats(150, 0.66, worst=0.46), BASE2)
    v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2), {"required_pp": 2.0}, claimed_pp=5.0)
    assert any("最差月" in r for r in v.reasons)


def test_gate4_power_preflight_rejects_small_claim() -> None:
    # n=150 可检测下限 ~10.2pp，声称 3pp → G4 拒
    hypo = _pair(_stats(150, 0.66, 0.54), BASE2)
    v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2), {"required_pp": 2.0}, claimed_pp=3.0)
    assert any("测不出" in r for r in v.reasons)


def test_all_gates_pass() -> None:
    # 单场景假设（只改 close_pos_min → affected={scene1}）：改善 +8pp、
    # 最差月持平、声称 12pp → 全过；未受影响的场景②不设 G2 障（+0pp 不卡闸）
    hypo = _pair(_stats(150, 0.70, 0.54), BASE2)
    v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2), {"required_pp": 2.0},
                                claimed_pp=12.0, affected={"scene1"})
    assert v.passed is True, v.reasons
    assert not any(g["gate"] == "G2-scene2" for g in v.gates)  # 未受影响场景无 G2


def test_affected_scenes_mapping() -> None:
    # 参数差异 → 受影响场景映射
    base = {"close_pos_min": 0.85, "vol_ratio_min": 2.0, "vol_ma_window": 20,
            "eps": 0.0005, "level_lookbacks": {"4h": 48}}
    assert HypothesisArbiter._affected_scenes({**base, "close_pos_min": 0.88}, base) == {"scene1"}
    assert HypothesisArbiter._affected_scenes({**base, "vol_ratio_min": 2.5}, base) == {"scene2"}
    assert HypothesisArbiter._affected_scenes({**base, "eps": 0.0008}, base) == {"scene1", "scene2"}
    assert HypothesisArbiter._affected_scenes(dict(base), base) == {"scene1", "scene2"}  # 无差异保守全查


def test_multiple_hypotheses_raise_margin() -> None:
    # 门槛随假设数上调后，原 +2.1pp 的改善不再够（√4=2 倍 → 需 4pp）
    hypo = _pair(_stats(150, 0.641, 0.54), BASE2)
    v = _arbiter()._apply_gates(hypo, _pair(BASE1, BASE2), {"required_pp": 4.0}, claimed_pp=5.0)
    g2 = [g for g in v.gates if g["gate"] == "G2-scene1"]
    assert g2 and g2[0]["passed"] is False
