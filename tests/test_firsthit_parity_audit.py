"""Phase 0 审计纯函数单测：配对政策替换 estimand（规范 §8.2）与护栏代理。"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "audit_firsthit_parity", Path("scripts/audit_firsthit_parity.py"))
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

paired_policy_value = mod.paired_policy_value
samples_to_curves = mod.samples_to_curves
trigger_q_proxy_guard_curve = mod.trigger_q_proxy_guard_curve


def _ev(win, q):
    return 0.98 / q - 1.0 if win else -1.0


def test_samples_to_curves_splits_by_kind():
    rows = [
        {"timestamp": 1000, "down_price": 0.3, "btc_price": 100.0},
        {"timestamp": 2000, "down_price": 0.07, "btc_price": 100.5},
        {"timestamp": 3000, "down_price": None, "btc_price": None},
    ]
    curves = samples_to_curves(rows)
    assert curves["down"] == [{"t": 1000, "v": 0.3}, {"t": 2000, "v": 0.07}]
    assert curves["btc"] == [{"t": 1000, "v": 100.0}, {"t": 2000, "v": 100.5}]


def test_paired_policy_value_all_g0_denominator():
    # 3 个 G0 事件；child 命中前两个。A_p 恒 1 → diff = [0, 0, −1]，
    # 即子政策只在「父做子不做」的窗口产生 −R 贡献（规范 §8.2 政策替换语义）。
    events = [
        {"labels": {"c": True}, "r": _ev(True, 0.05)},   # +18.6（diff=0）
        {"labels": {"c": True}, "r": _ev(False, 0.08)},   # −1（diff=0）
        {"labels": {"c": False}, "r": _ev(True, 0.06)},   # +15.33（diff=−1 → 贡献 −R）
    ]
    out = paired_policy_value(events, child="c", parent=None)
    assert out["n_g0"] == 3
    assert out["contributing"] == 1  # (A_c − A_p) != 0 的窗口 = 子政策跳过者
    assert abs(out["delta_v"] - (-events[2]["r"] / 3)) < 1e-9
    assert abs(out["child_v"] - (events[0]["r"] + events[1]["r"]) / 3) < 1e-9
    assert abs(out["parent_v"] - (events[0]["r"] + events[1]["r"] + events[2]["r"]) / 3) < 1e-9


def test_paired_policy_value_child_vs_child():
    events = [
        {"labels": {"a": True, "b": True}, "r": 1.0},
        {"labels": {"a": True, "b": False}, "r": -1.0},
        {"labels": {"a": False, "b": False}, "r": 10.0},
    ]
    out = paired_policy_value(events, child="b", parent="a")
    # (A_b − A_a)·R = [0, −1·(−1), 0] → mean = 1/3
    assert abs(out["delta_v"] - (0 + 1.0 + 0) / 3) < 1e-9
    assert out["contributing"] == 1  # 只有第二个窗标签不同


def test_paired_policy_value_empty():
    out = paired_policy_value([], child="c", parent=None)
    assert out == {"n_g0": 0, "contributing": 0, "delta_v": None,
                   "child_v": None, "parent_v": None}


def test_trigger_q_proxy_guard_curve_labels_proxy():
    events = [
        {"q": 0.05, "win": True}, {"q": 0.07, "win": False},
        {"q": 0.09, "win": True}, {"q": 0.11, "win": True},
    ]
    curve = trigger_q_proxy_guard_curve(events, guards=(0.06, 0.10))
    assert curve["proxy"] == "trigger_q_proxy"
    g6 = curve["guards"]["0.06"]
    assert g6["eligible_n"] == 1 and g6["coverage"] == 0.25
    g10 = curve["guards"]["0.10"]
    assert g10["eligible_n"] == 3 and abs(g10["ev"] - (
        (0.98 / 0.05 - 1) - 1 + (0.98 / 0.09 - 1)) / 3) < 1e-9
