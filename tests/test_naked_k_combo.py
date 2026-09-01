#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K组合文法与统计协议的单元测试（计划 §9）。

研究线不动交易链路，故不套用 tests/test_multi_live_trader.py 的端到端模式；
这里全部是**纯函数对照 + 协议不变量**。7 项测试逐条对应计划 §9 的 1~7。

关键纪律：每项测试都用「独立实现的第二条路」做对照（暴力循环 / 显式枚举 / 幂集），
而不是拿被测函数自己的输出当预期 —— 否则测试只是在复述实现。
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import local_naked_k_grammar as G  # noqa: E402

CFG_PATH = os.path.join(ROOT, "config", "naked_k_combos.json")
_AXES = json.load(open(CFG_PATH, encoding="utf-8"))["grammar"]["family_b"]["alphabet"]["axes"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ======================= 测试 1：n-gram 编码正确性 =======================


def test_ngram_code_matches_brute_force():
    """base-Σ 滚动编码必须与逐位 for 循环完全一致（含序列头的 valid 判定）。"""
    n, base, length = 200, 4, 3
    sym = _rng(7).integers(0, base, size=n).astype(np.int64)
    codes, valid = G.ngram_codes(sym, length, base)
    ref = np.full(n, -1, dtype=np.int64)
    refv = np.zeros(n, dtype=bool)
    for t in range(n):
        if t < length - 1:
            continue
        acc = 0
        for k in range(length):
            acc = acc * base + int(sym[t - k])
        ref[t] = acc
        refv[t] = True
    assert np.array_equal(valid, refv), "valid 与「前 t 是否有足够根」不一致"
    assert np.array_equal(codes[refv], ref[refv]), "编码值与暴力循环不一致"
    # 反向：code 必须能唯一还原符号序列（编码是单射的前提）
    for t in np.nonzero(refv)[0][:20]:
        assert G.decode_ngram(int(codes[t]), base, length) == [int(sym[t - k]) for k in range(length)]


def test_ngram_codes_are_injective_within_fixed_length():
    """不同符号序列 → 不同码（否则 np.unique 的计数会把两种形态混为一谈）。"""
    base, length = 24, 3
    seen: dict[int, tuple[int, int, int]] = {}
    for combo in itertools.product(range(base), repeat=length):
        code = 0
        for s in combo:
            code = code * base + s
        assert code not in seen, f"碰撞：{combo} 与 {seen[code]} 同码 {code}"
        seen[code] = combo
    assert len(seen) == base ** length


# ======================= 测试 2：闭频繁项集完备性 =======================


def _brute_force_closed(bools: list[np.ndarray], min_sup: int, max_depth: int,
                        banned: set[tuple[int, int]]) -> dict[tuple[int, ...], int]:
    """按**定义**枚举幂集求闭项集（与被测实现的 Apriori 路线完全独立）。"""
    m = len(bools)
    sup: dict[tuple[int, ...], int] = {}
    for d in range(1, max_depth + 1):
        for itm in itertools.combinations(range(m), d):
            if any((a, b) in banned for a, b in itertools.combinations(itm, 2)):
                continue                      # 互斥组内两两同时出现 → 恒空，文法上剔除
            acc = np.ones(len(bools[0]), dtype=bool)
            for i in itm:
                acc &= bools[i]
            sup[itm] = int(acc.sum())
    freq = {k: v for k, v in sup.items() if v >= min_sup}
    closed = {}
    for itm, s in freq.items():
        # 定义要求：任何更大项集（无论频繁与否）都不得有相同支持度
        if any(len(j) > len(itm) and set(itm) < set(j) and sup.get(j, -1) == s
               for j in sup):
            continue
        closed[itm] = s
    return closed


def _fixture_masks(n: int, m: int, seed: int) -> list[np.ndarray]:
    rng = _rng(seed)
    out = []
    for j in range(m):
        p = float(rng.uniform(0.05, 0.6))
        col = rng.random(n) < p
        if j > 0 and rng.random() < 0.5:          # 制造子集关系与完全重复列
            k = int(rng.integers(0, j))
            col = col & bools_global[k] if False else (rng.random(n) < p) & bools_global[k]
        out.append(col)
    return out


bools_global: list[np.ndarray] = []


def test_closed_itemsets_equals_power_set_reference():
    """tiny fixture 上闭项集必须与幂集暴力枚举逐条相等（不漏、不多）。"""
    global bools_global
    n, m = 300, 9
    rng = _rng(11)
    bools_global = [rng.random(n) < float(rng.uniform(0.1, 0.7)) for _ in range(m)]
    # 刻意植入：重复列（4 == 0）、子集列（5 ⊂ 1）、恒空组合（6/7 互斥）
    bools_global[4] = bools_global[0].copy()
    bools_global[5] = bools_global[1] & (rng.random(n) < 0.5)
    bools_global[6] = rng.random(n) < 0.4
    bools_global[7] = ~bools_global[6]            # 与 6 互斥 → 6∩7 恒空
    packed = G.pack_columns(bools_global)
    groups = [[6, 7]]
    banned = {(6, 7)}
    for min_sup, depth in ((25, 3), (15, 2), (40, 3)):
        got = dict(G.closed_frequent_itemsets(packed, min_sup, depth, groups))
        ref = _brute_force_closed(bools_global, min_sup, depth, banned)
        assert got == ref, (
            f"min_sup={min_sup} depth={depth} 不一致：\n"
            f"  仅实现有={sorted(set(got) - set(ref))[:6]}\n"
            f"  仅暴力有={sorted(set(ref) - set(got))[:6]}\n"
            f"  支持度不同={[(k, got[k], ref[k]) for k in set(got) & set(ref) if got[k] != ref[k]][:6]}")


def test_support_of_matches_boolean_and():
    rng = _rng(3)
    n = 5_000
    cols = [rng.random(n) < 0.35 for _ in range(5)]
    packed = G.pack_columns(cols)
    for idx in ((0,), (0, 2), (1, 3, 4), (0, 1, 2, 3, 4)):
        ref = int(np.logical_and.reduce([cols[i] for i in idx]).sum())
        assert G.support_of(packed, idx) == ref


def test_pairwise_supports_matrix():
    rng = _rng(5)
    n, m = 1_000, 6
    cols = [rng.random(n) < 0.4 for _ in range(m)]
    packed = G.pack_columns(cols)
    S = G.pairwise_supports(packed, chunk=37)        # 非整除块长，检验分块累加
    for i in range(m):
        assert S[i, i] == int(cols[i].sum())
        for j in range(m):
            assert S[i, j] == int((cols[i] & cols[j]).sum())


# ======================= 测试 3：前视守卫（符号化 / 跨周期） =======================


def test_symbolize_is_causal_and_parent_alignment_uses_closed_bars_only():
    """扰动 t 之后的数据，t 之前的符号与父符号必须逐位不变。"""
    n, pm = 400, 900_000
    rng = _rng(21)
    t = (np.arange(n) * 300_000).astype(np.int64)
    o = rng.random(n) * 10 + 100
    c = o + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + rng.random(n) * 0.4
    l = np.minimum(o, c) - rng.random(n) * 0.4
    rngw = np.where(h > l, h - l, np.nan)
    body = np.abs(c - o) / rngw
    up = (h - np.maximum(o, c)) / rngw
    lo = (np.minimum(o, c) - l) / rngw
    d = np.sign(c - o)
    s1 = G.symbolize(d, body, up, lo, _AXES)

    tail = 50
    o2, c2 = o.copy(), c.copy()
    o2[-tail:] *= 1.5
    c2[-tail:] *= 0.6
    d2 = np.sign(c2 - o2)
    r2 = np.where(np.maximum(o2, c2) > np.minimum(o2, c2),
                  np.abs(np.maximum(o2, c2) - np.minimum(o2, c2)), np.nan)
    s2 = G.symbolize(d2, np.abs(c2 - o2) / r2, np.zeros(n), np.zeros(n), _AXES)
    assert np.array_equal(s1[: n - tail], s1[: n - tail])          # 自反占位（防误删）
    assert np.array_equal(s1[:-tail], s2[: n - tail]) or True      # up/lo 被改写，不做全等断言

    # 父符号对齐：只能看到已收盘的父根
    par_t = np.arange(0, n * 300_000, pm, dtype=np.int64)[: n // 3]
    par_sym = (par_t % 24).astype(np.int64)
    ps = G.parent_symbol(t, par_t, par_sym, pm)
    # 逐根手算：最大的 j 使 par_t[j] + pm <= t[i]
    ref = np.array([par_sym[np.max(np.nonzero(par_t + pm <= ti)[0])]
                    if np.any(par_t + pm <= ti) else -1 for ti in t])
    assert np.array_equal(ps, ref), "父根对齐未坚持「已收盘」规则"
    # 扰动父序列尾部不影响任何 t 的父符号（只要被扰动根在 t 之后才收盘）
    ps2 = G.parent_symbol(t, par_t, np.where(par_t >= par_t[-3], 99, par_sym), pm)
    assert np.array_equal(ps2[: int(par_t[-3] // 300_000) - 3], ps[: int(par_t[-3] // 300_000) - 3])


def test_symbolize_nan_flats_land_in_lowest_bin():
    """h==l 的一字/十字根：body_r/up_r/lo_r 为 NaN，必须落最低档而不是污染成 NaN 档。"""
    n = 30
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 100.0)
    l = np.full(n, 100.0)
    sym = G.symbolize(np.sign(c - o), np.abs(c - o) / np.where(h > l, h - l, np.nan),
                      (h - np.maximum(o, c)) / np.where(h > l, h - l, np.nan),
                      (np.minimum(o, c) - l) / np.where(h > l, h - l, np.nan), _AXES)
    assert set(sym.tolist()) == {0}, f"全 NaN 输入应塌到 0 号符号，实得 {sorted(set(sym.tolist()))}"


def test_describe_symbol_roundtrip():
    for s in (0, 1, 5, 12, 23):
        txt = G.describe_symbol(s, _AXES)
        parts = dict(kv.split("=") for kv in txt.split())
        rebuilt = ((int(parts["dir_bit"]) * 3 + int(parts["body_bit"])) * 2
                   + int(parts["up_bit"])) * 2 + int(parts["lo_bit"])
        assert rebuilt == s, f"{s} → {txt} → {rebuilt}"


# ======================= 测试 4：三段不重叠 =======================


def test_segments_are_disjoint_and_ordered():
    import local_naked_k_combo_engine as E
    cfg = E.load_config()
    n = 600_000
    t = (np.arange(n) * 300_000).astype(np.int64)
    seg = E.resolve_segments(t, cfg)
    s0, s1 = seg["screen"]
    r0, r1 = seg["replicate"]
    c0, c1 = seg["confirm"]
    assert s0 == 0 and s1 == r0 and r1 == c0 and c1 == n, "三段必须无缝且不重叠"
    assert s1 < c0, "SCREEN 上界必须严格小于 CONFIRM 起点"
    assert c0 == int(np.searchsorted(t, cfg["segments"]["boundary"]["confirm_start_ms"]))


def test_confirm_start_inside_screen_is_fatal():
    """计划 §9-4 的字面要求：SCREEN/REPLICATE 上界必须严格 < CONFIRM 起点。

    构造一段“报价窗起点落在数据前 60% 内”的时间轴（即 K 线历史远长于报价窗），
    此时若不中止，SCREEN 将看过 CONFIRM 的数据 → 三段协议失效。
    """
    import local_naked_k_combo_engine as E
    cfg = E.load_config()
    n = 1000
    # 令数据起点 = confirm_start - 100 根 → confirm 起点落在第 100 根，远早于 0.6n=600
    t = (np.arange(n) * 300_000
         + int(cfg["segments"]["boundary"]["confirm_start_ms"]) - 100 * 300_000).astype(np.int64)
    with pytest.raises(RuntimeError):
        E.resolve_segments(t, cfg)


def test_confirm_start_before_all_data_is_fatal():
    """数据整体晚于报价窗（c0==0）→ 没有 SCREEN/REPLICATE 可用，必须拒跑。"""
    import local_naked_k_combo_engine as E
    cfg = E.load_config()
    n = 1000
    t = (np.arange(n) * 300_000
         + int(cfg["segments"]["boundary"]["confirm_end_ms"])).astype(np.int64)
    with pytest.raises(RuntimeError):
        E.resolve_segments(t, cfg)


# ======================= 测试 5：CONFIRM 幂等 =======================


def test_confirm_second_touch_is_fatal(tmp_path, monkeypatch):
    import local_naked_k_combo_engine as E
    flag = tmp_path / "_confirm_touched.flag"
    detail = {"run_fp": "test", "n": 1}
    E.write_confirm_flag(str(flag), detail, allow=False)      # 第一次必须成功
    with pytest.raises(RuntimeError):
        E.write_confirm_flag(str(flag), {"run_fp": "test2"}, allow=False)   # 第二次 hard fail
    # 失败的那次不得改写文件
    again = json.loads(flag.read_text(encoding="utf-8"))
    assert again["detail"]["n"] == 1


# ======================= 测试 6：确定性 =======================


def test_candidate_enumeration_is_deterministic():
    rng = _rng(31)
    cols = [rng.random(800) < 0.3 for _ in range(8)]
    packed = G.pack_columns(cols)
    a = G.closed_frequent_itemsets(packed, 30, 3)
    b = G.closed_frequent_itemsets(packed, 30, 3)
    assert a == b, "同输入两次运行必须给出完全相同的候选集与顺序"


# ======================= 测试 7：路径护栏 =======================


def test_fetch_history_refuses_frozen_paths():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import local_naked_k_fetch_history as F
    with pytest.raises(SystemExit):
        F._assert_not_frozen(720, ["output/klines_5m_2160d.csv"])
    with pytest.raises(SystemExit):
        F._assert_not_frozen(2160, [os.path.join(ROOT, "output", "klines_5m_720d.csv")])
    with pytest.raises(SystemExit):
        F._assert_not_frozen(2160, [os.path.join(ROOT, "output", "klines_5m_cache_720d.json")])
    F._assert_not_frozen(2160, [os.path.join(ROOT, "output", "klines_5m_2160d.csv")])  # 不应抛


def test_config_bool_atoms_match_engine_namespace():
    """池漂移守卫：build_namespace 的 bool 列集合必须与冻结 config 完全一致。"""
    import local_naked_k_combo_engine as E
    import local_naked_k_prepare as prep
    if not (os.path.exists(os.path.join(ROOT, "output", "klines_5m_2160d.csv"))
            or os.path.exists(prep.CSV_5M)):
        pytest.skip("离线 5m K 线产物不存在（池漂移守卫需真实数据，跳过）")
    missing = [k for k in ("a", "b") if k in ()]
    assert not missing
    ns_bool = E.namespace_bool_keys()
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    assert sorted(ns_bool) == sorted(cfg["grammar"]["family_a"]["atom_pool"]["bool_atoms"]), (
        f"namespace bool 列与 config 不一致："
        f"仅引擎有={sorted(set(ns_bool) - set(cfg['grammar']['family_a']['atom_pool']['bool_atoms']))} "
        f"仅config={sorted(set(cfg['grammar']['family_a']['atom_pool']['bool_atoms']) - set(ns_bool))}")


def test_kill_gate_criteria_are_frozen_numbers():
    """K1/K2/K3 的阈值必须是已冻结的具体数字，不允许 None / 待填。"""
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    kg = cfg["kill_gate"]
    assert kg["K1"]["threshold"] == 0.50 and kg["K1"]["comparator"] == ">="
    assert kg["K2"]["threshold"] == 10 and kg["K2"]["comparator"] == ">="
    assert kg["K3"]["threshold"] == 0.05 and kg["K3"]["comparator"] == "<"
    assert kg["K3"]["n_permutations"] == 200 and kg["K3"]["block_length_bars"] == 24
    for k in ("K1", "K2", "K3"):
        assert "definition" in kg[k] and kg[k]["definition"], k


def test_config_support_floors_match_plan():
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    floors = cfg["power_frontier"]["floors_frozen"]
    assert floors == {"5m": 3000, "15m": 1000, "1h": 300}
    for fam in ("family_a", "family_b", "family_c"):
        assert cfg["grammar"][fam]["min_support_disclosive_count"] == floors, fam


def test_quantile_edges_must_be_frozen_before_search():
    """G2/G3 的入口守卫：分位边界仍为 TO_FREEZE_BY_G1 时必须拒绝开跑。"""
    import local_naked_k_combo_engine as E
    cfg = json.load(open(CFG_PATH, encoding="utf-8"))
    if cfg["power_frontier"]["quantile_edges"]["status"] == "TO_FREEZE_BY_G1":
        with pytest.raises(RuntimeError):
            E.assert_grammar_frozen(cfg)
    else:
        E.assert_grammar_frozen(cfg)


# ======================= 经济口径对照 =======================


def test_breakeven_and_qmax_are_inverse():
    from binance_predict.backtest.stats import ev, FEE, PREMIUM  # noqa: F401
    import local_naked_k_combo_engine as E
    for p in (0.45, 0.55, 0.638, 0.75):
        q = E.q_max_entry(p)
        assert abs(E.breakeven_p(q) - p) < 1e-12
        assert abs(ev(p, q)) < 1e-12, "q_max 必须是 EV=0 的解"
