#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""裸K组合文法的纯函数层（计划 §2 / §5）。

本模块**只做变换，不做判决**：符号化、n-gram 编码、位集支持度、闭频繁项集、
跨周期对齐取父符号。所有函数满足：
- 纯函数（输入 → 输出，无全局态、无 IO、无随机）
- numpy-only（无 pandas / mlxtend / scipy）
- 因果安全：任何用到未来信息的地方都会体现在输出上（valid=False / 码=-1），
  由调用方的前视守卫测试断言。

设计要点（为什么这样实现，而不是抄一个挖掘库）：
1. **位集 + popcount**：支持度 = |A ∩ B|，用 packed bitset 做 AND 再 np.bitwise_count。
   m≈160 个原子、n≈37 万根时，全 pairwise 只需 ~0.6GB 内存流量，无物化 n×m 矩阵。
2. **闭项集而非全幂集**：闭项集在「达到支持度下界的组合」上是**完备**的
   （任何非闭项集都有等支持度的超集，其统计量由超集代表），因此「深度≤3 的闭项集」
   是对『无限组合』有原则的有界化 —— 不是取前 N 个候选的那种任意截断。
3. **base-Σ 整数编码**：把 L 元序列映到一个 int64，一次 np.unique + np.bincount
   得到全空间计数与 outcome 和（O(n·L)），不必逐序列求值表达式。
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

# ============================ 族 B：字母表 Σ ============================


def symbolize(dir_: np.ndarray, body_r: np.ndarray, up_r: np.ndarray,
              lo_r: np.ndarray, axes: Sequence[dict[str, Any]]) -> np.ndarray:
    """把逐根裸K四要素编码成 [0, |Σ|) 的符号索引。

    |Σ| = Π axes[*].levels。NaN 先经 nan_to_num(0) 再分档：rng==0 的十字/一字根
    其 body/up/lo 比值在数学上就是 0/0，真实语义是「实体为 0、无上下影」，
    归最低档是**还原语义**，不是缺失填充（见 config.grammar.family_b.nan_handling）。
    """
    n = len(dir_)
    src = {"dir_": dir_, "body_r": body_r, "up_r": up_r, "lo_r": lo_r}
    code = np.zeros(n, dtype=np.int64)
    for ax in axes:
        x = np.nan_to_num(np.asarray(src[ax["source"]], dtype=np.float64), nan=0.0)
        code = code * int(ax["levels"]) + _bin_index(x, ax)
    return code


def _bin_index(x: np.ndarray, ax: dict[str, Any]) -> np.ndarray:
    """按 axis 定义把连续值映射到档号（档数 = levels）。

    支持两种写法：
      - "rule" 里给阈值（本项目用文本规则，解析在 config 侧完成，这里读 thresholds）
      - "thresholds": [t1, t2, ...] 递增，档号 = #(x >= ti)
    """
    ths = ax.get("thresholds")
    if ths is None:
        raise ValueError(f"axis {ax.get('key')} 缺少 thresholds（config 侧必须把文本规则解析成数值）")
    idx = np.zeros(len(x), dtype=np.int64)
    for t in ths:
        idx += (x >= float(t)).astype(np.int64)
    return idx


def sigma_size(axes: Sequence[dict[str, Any]]) -> int:
    out = 1
    for a in axes:
        out *= int(a["levels"])
    return out


def ngram_codes(sym: np.ndarray, length: int, base: int) -> tuple[np.ndarray, np.ndarray]:
    """滚动 base-Σ 编码：code_t = Σ_{k=0..L-1} sym_{t-k} · base^{L-1-k}。

    返回 (codes, valid)。valid=False 表示 t 之前不足 L-1 根（序列头），
    这些位置必须从统计中剔除 —— 不允许用「补零」伪装成有效样本。
    """
    n = len(sym)
    codes = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for k in range(length):
        lagged = np.full(n, -1, dtype=np.int64)
        if k < n:
            lagged[k:] = sym[: n - k]
        codes = codes * int(base) + np.where(lagged >= 0, lagged, 0)
        valid &= lagged >= 0
    codes = np.where(valid, codes, -1)
    return codes, valid


def ngram_table(codes: np.ndarray, valid: np.ndarray, win: np.ndarray,
                target_valid: np.ndarray, base: int,
                length: int) -> dict[str, np.ndarray]:
    """全 n-gram 空间的计数与命中和（一次 np.unique + 两次 np.bincount）。

    win / target_valid 为判决目标的逐根布尔与有效性（族 B 用 dirup_1 / dirdn_1）。
    返回的 code_to_sym[length] 可反解出每个符号档，供报告还原可读表达式。
    """
    sel = valid & target_valid
    c = codes[sel]
    if c.size == 0:
        return {"codes": np.empty(0, np.int64), "n": np.empty(0, np.int64),
                "k": np.empty(0, np.int64), "wr": np.empty(0, np.float64)}
    uniq, inv = np.unique(c, return_inverse=True)
    w = win[sel].astype(np.int64)
    n_arr = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    k_arr = np.bincount(inv, weights=w.astype(np.float64), minlength=len(uniq))
    k_arr = np.rint(k_arr).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = k_arr / n_arr
    return {"codes": uniq, "n": n_arr, "k": k_arr, "wr": wr}


def decode_ngram(code: int, base: int, length: int) -> list[int]:
    """code → [sym_t, sym_{t-1}, ...]（与 ngram_codes 的权重顺序严格互逆）。

    ngram_codes 是「先乘后加」，所以最新的根落在最高位、最老的根落在最低位；
    逐次取模得出来的是**反序**，必须翻回来。忘了这一步会把「先阴后阳」读成「先阳后阴」。
    """
    out = []
    x = int(code)
    for _ in range(length):
        out.append(x % base)
        x //= base
    out.reverse()
    return out


# ============================ 族 A：位集与闭频繁项集 ============================


def pack_columns(cols: Iterable[np.ndarray]) -> np.ndarray:
    """把 m 个等长布尔列打包成 (m, ceil(n/8)) 的 uint8 位集矩阵。

    用 np.packbits(bitorder='big')：同一列内位序一致即可，
    AND + popcount 的语义与位序无关，故不必额外处理。
    """
    arrs = [np.asarray(c, dtype=bool) for c in cols]
    if not arrs:
        return np.empty((0, 0), dtype=np.uint8)
    n = len(arrs[0])
    for a in arrs:
        if len(a) != n:
            raise ValueError("位集列长不一致，pack 会静默错位")
    packed = [np.packbits(a, bitorder="big") for a in arrs]
    nb = packed[0].size
    out = np.zeros((len(arrs), nb), dtype=np.uint8)
    for i, p in enumerate(packed):
        out[i] = p
    return out


def support_of(packed: np.ndarray, idx: Sequence[int]) -> int:
    """|∩_{i∈idx} A_i|（一次 AND + 一次 popcount，不物化中间掩码）。"""
    if len(idx) == 0:
        raise ValueError("空项集的支持度无定义")
    acc = packed[int(idx[0])]
    for i in idx[1:]:
        acc = acc & packed[int(i)]
    return int(np.bitwise_count(acc).sum())


def pairwise_supports(packed: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """全 pairwise 支持度矩阵（对称，含对角=单原子支持度）。

    分块遍历字节维以控制峰值内存：单次只物化 (m, m, chunk) 的 bool 立方。
    m=160、chunk=4096 → 160*160*4096 B ≈ 100MB/块（bool 视图为 1 字节/元素）。
    """
    m, nb = packed.shape
    out = np.zeros((m, m), dtype=np.int64)
    for s in range(0, nb, chunk):
        blk = packed[:, s: s + chunk]                      # (m, c) uint8
        cube = blk[:, None, :] & blk[None, :, :]           # (m, m, c)
        out += np.bitwise_count(cube).sum(axis=2, dtype=np.int64)
    return out


def closed_frequent_itemsets(packed: np.ndarray, min_support: int, max_depth: int,
                             exclusive_groups: Sequence[Sequence[int]] = (),
                             pair_support: np.ndarray | None = None,
                             ) -> list[tuple[tuple[int, ...], int]]:
    """深度 ≤ max_depth 的**闭**频繁项集（Apriori 下行剪枝 + 等支持度超集剔除）。

    返回 [(atoms_tuple, support), ...]，按 (支持度降序, 原子索引升序) 稳定排序。

    闭合性判定：I 闭 ⟺ 不存在被枚举到的频繁超集 J ⊋ I 使 sup(J) == sup(I)。
    只需查**直接超集**（深度 +1）：若存在 |J| ≥ |I|+2 且 sup(J)=sup(I)，则由支持度单调性，
    任何夹在中间的 K（|K|=|I|+1）也满足 sup(K)=sup(I) —— 查一层即完备，代价 O(|L_{d+1}|·(d+1))。
    深度 == max_depth 的项集无更深层可查 → 按定义闭（这是深度封顶下的相对闭合）。

    exclusive_groups：同组原子（如同一变量的 3 个互斥分位档）两两合取恒空，
      在 level-2 生成阶段结构性剔除（不生成、不计数、不进假设预算）。
    """
    m = packed.shape[0]
    if int(max_depth) < 1:
        raise ValueError(f"max_depth 必须 >= 1，收到 {max_depth}")
    if pair_support is None:
        pair_support = pairwise_supports(packed)
    banned = _banned_pairs(exclusive_groups)

    freq1 = [(int(i), int(pair_support[i, i])) for i in range(m) if pair_support[i, i] >= min_support]
    levels: dict[int, dict[tuple[int, ...], int]] = {1: {(i,): s for i, s in freq1}}
    prev_map = levels[1]
    for d in range(2, int(max_depth) + 1):
        items = sorted({a for itm in prev_map for a in itm})
        if len(items) < d:
            levels[d] = {}
            break
        prev_keyset = set(prev_map)
        cands: list[tuple[int, ...]] = []
        for itm in prev_map:
            for a in items:
                if a <= max(itm):
                    continue                      # 只向右扩张，保证每个候选只生成一次
                if d == 2 and (itm[0], a) in banned:
                    continue                      # 互斥档：恒空，不生成也不计数
                new = itm + (a,)
                if any(new[:i] + new[i + 1:] not in prev_keyset for i in range(len(new))):
                    continue                      # Apriori：所有 (d-1) 子集必须频繁
                cands.append(new)
        cur: dict[tuple[int, ...], int] = {}
        for c in cands:
            s = support_of(packed, c)
            if s >= min_support:
                cur[c] = s
        levels[d] = cur
        prev_map = cur

    nonclosed: set[tuple[int, ...]] = set()
    for d in range(1, int(max_depth)):
        deeper = levels.get(d + 1, {})
        for J, sJ in deeper.items():
            for i in range(len(J)):
                I = J[:i] + J[i + 1:]
                if I in levels[d] and levels[d][I] == sJ:
                    nonclosed.add(I)
    out: list[tuple[tuple[int, ...], int]] = [
        (itm, s) for d in levels for itm, s in levels[d].items() if itm not in nonclosed]
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def _banned_pairs(groups: Sequence[Sequence[int]]) -> set[tuple[int, int]]:
    banned: set[tuple[int, int]] = set()
    for g in groups:
        gl = sorted(int(x) for x in g)
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                banned.add((gl[i], gl[j]))
    return banned


def itemset_stats(packed: np.ndarray, itemsets: Sequence[Sequence[int]],
                  n: int) -> tuple[np.ndarray, np.ndarray]:
    """批量支持度与支持率（支持度用位运算，不走布尔物化）。"""
    sup = np.array([support_of(packed, it) for it in itemsets], dtype=np.int64)
    rate = sup / max(1, n)
    return sup, rate


def pack_region(packed: np.ndarray, n: int, a: int, b: int) -> np.ndarray:
    """把 (m, nb) 的全量位集裁成 [a, b) 的位集（重新字节对齐）。

    为什么要专门裁一段：三段协议下支持度/命中统计只发生在段内，而位集的字节边界
    是相对全数组的。区域字节数从 ceil(n/8) 降到 ceil((b-a)/8)，置换零校准的代价
    直接按段长缩放。不做「整字节近似」——那会把段边界挪动最多 7 根。
    """
    if not (0 <= a <= b <= n):
        raise ValueError(f"非法区间 [{a},{b}) vs n={n}")
    if packed.shape[0] == 0:
        return np.empty((0, 0), dtype=np.uint8)
    bits = np.unpackbits(packed, axis=1, bitorder="big")[:, :n]
    return np.packbits(bits[:, a:b], axis=1, bitorder="big")


def and_region(packed: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """若干位集的 AND 结果（仍为位集），用于把候选掩码批量存下来复用。"""
    if len(idx) == 0:
        raise ValueError("空项集无定义")
    acc = packed[int(idx[0])]
    for i in idx[1:]:
        acc = acc & packed[int(i)]
    return acc


def popcount_matrix(mat: np.ndarray) -> np.ndarray:
    """(q, nb) 位集矩阵 → 每行 popcount（uint8 上的 np.bitwise_count）。"""
    if mat.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    return np.bitwise_count(mat).sum(axis=1, dtype=np.int64)


# ============================ 连续原子三分位 ============================


def tertile_edges(x: np.ndarray) -> tuple[float, float]:
    """SCREEN 段上的三分位边界（NaN 不参与估计）。"""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        raise ValueError(f"有限样本不足（{a.size}）无法估计三分位")
    e1, e2 = np.percentile(a, [100 / 3, 200 / 3])
    return (float(e1), float(e2))


def ternary_bins(x: np.ndarray, edges: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """low / mid / high 三个互斥布尔列；NaN → 三档全 False（不强行归档）。"""
    a = np.asarray(x, dtype=np.float64)
    e1, e2 = float(edges[0]), float(edges[1])
    fin = np.isfinite(a)
    low = fin & (a < e1)
    mid = fin & (a >= e1) & (a < e2)
    high = fin & (a >= e2)
    return low, mid, high


# ============================ 族 C：跨周期已收盘父符号 ============================


def parent_symbol(child_t: np.ndarray, par_t: np.ndarray, par_sym: np.ndarray,
                  par_ms: int) -> np.ndarray:
    """取「在本 child 根开始时刻已收盘」的那根父根的符号；无则 -1。

    与 engine._align 用同一条 searchsorted 规则（par.t <= child.t - par_ms），
    不得另写一套 —— 两套对齐逻辑之间迟早会差一根，而那根就是前视。
    """
    j = np.searchsorted(np.asarray(par_t, dtype=np.int64),
                        np.asarray(child_t, dtype=np.int64) - int(par_ms), side="right") - 1
    ok = j >= 0
    return np.where(ok, np.asarray(par_sym, dtype=np.int64)[np.clip(j, 0, None)], -1)


def cross_product_code(child_sym: np.ndarray, par_sym: np.ndarray,
                       sigma: int) -> tuple[np.ndarray, np.ndarray]:
    """child × parent 的复合码 ∈ [0, sigma²)，并给出有效性（父根存在且子符号有效）。"""
    ok = (child_sym >= 0) & (par_sym >= 0)
    code = np.where(ok, child_sym * int(sigma) + par_sym, -1)
    return code, ok


def cross_code_to_parts(code: int, sigma: int) -> tuple[int, int]:
    c, p = divmod(int(code), int(sigma))
    return c, p


# ============================ 可读表达式还原 ============================


def describe_symbol(sym: int, axes: Sequence[dict[str, Any]]) -> str:
    """符号索引 → 人可读的四档描述（用于报告与机制预注册）。

    编码是「先乘后加」逐轴推进，故**最后一个轴变化最快** → 反解必须从末轴开始取模。
    """
    x = int(sym)
    parts: list[str] = []
    for ax in reversed(list(axes)):
        lv = int(ax["levels"])
        parts.append(f"{ax['key']}={x % lv}")
        x //= lv
    return " ".join(reversed(parts))


def symbol_axis_labels(axes: Sequence[dict[str, Any]]) -> list[list[str]]:
    return [[f"{a['key']}:{i}" for i in range(int(a["levels"]))] for a in axes]
