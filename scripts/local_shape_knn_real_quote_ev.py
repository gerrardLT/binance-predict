#!/usr/bin/env python
"""local_shape_knn_real_quote_ev：形状 kNN 信号 × 预测市场真实报价配对 EV 重估。

缘起（上一轮 shape_segment_knn_720d 的判据缺陷）
    该脚本用 CFG["entry_price"]=0.5 固定入场价判 EV，得出 5m PRIMARY 冻结 holdout
    胜率 50.57% / lift +3.01pp / EV 下界 -0.0633 → PROMISING（0 个 CONFIRMED）。
    但房内最终口径（local_minute_quote_ev.py:4「所有 EV 必须用逐分钟真实报价配对
    计算，不用固定入场价假设」）要求真实报价配对。本脚本补这一步。

核心问题（比 EV 数字更重要）
    形状相似性是否携带「市场开盘报价尚未反映」的增量信息？
    实测报价分布（prediction_market_samples_online_20260819.json，5m，6850 窗）：
      开盘首条 up_price  p25/p50/p75 = 0.50 / 0.51 / 0.54
      开盘首条 down_price p25/p50/p75 = 0.46 / 0.49 / 0.50
      up+down p50 = 1.0000（零价差）
    ——开盘报价几乎恒定、信息含量极低；而 kNN consensus 是有散布的。
    错配空间就在这个差值里，这也是房内三个在跑实盘的检测器
    （misalignment / absorption_shadow / quote_edge）共同的范式：
    赌的不是方向，是定价错误。

口径（ex-ante 冻结于 CFG；跑后改=数据窥探，须全新评估段重验）
- **复用不改**：分段/表征/因果检索/投票全部 import 自 shape_segment_knn_720d，
  9 项 kNN 参数逐项启动断言 == 原 CFG，杜绝手抄漂移；本脚本 0 改动现有文件。
- **评估段** = 报价覆盖的 23.8 天（2026-07-26~08-19，6850 个 5m 窗），
  **refs** = 该段之前全部段（约 690 天）。refs cap 按「最早查询的 qs-causal_lag」
  统一收紧：比逐查询 expanding 更保守（评估段内部互不为 refs），且 cap 跨运行恒定
  → nn_idx 索引恒定 → 字节复现。
- **决策时刻 = 窗口开盘**：kNN 只用 anchor 及之前的 K 线，t=0 决策因果正确；
  报价取 ≤decision_sec 内最晚采样（逐字复刻 local_timeslice_winrate.py:290
  price_within 口径）。主口径 30s（贴近开盘），对照 150s（房内宪法 V1.1 决策点）。
- **EV 双口径**：EV_A = 0.98/q − 1（无溢价，local_s5_real_quote_ev 口径）、
  EV_B = 0.98/(q+0.01) − 1（含吃单溢价，misalignment_detector 落库口径，**主口径**）。
  两者均与 backtest/stats.py 的 ev(p,e)=p·(1−FEE)/(e+PREMIUM)−(1−p) 同源，自检对齐。
- **错配度** misalign = model_prob − q，model_prob = consensus（押 UP）
  / 1−consensus（押 DOWN）。consensus = 邻居中 UP 票占比，天然是模型概率估计。
- **纪律**：评估段与上一轮冻结 holdout（2026-04-14~09-05）时间重叠 → 全程标注
  exploratory 事后分析，不声称 pre-registered。为恢复部分纪律，评估段内部再切
  calib(前 70%) / confirm(后 30%)：错配阈值 θ **只在 calib 段按预定规则选定**，
  confirm 段用该 θ **终裁一次**。

不做 15m 的理由
    上一轮 15m 全 15 臂 REJECT_PREHOLDOUT（discovery↔validation 方向不一致），
    holdout 零触碰 —— 没有稳定方向可供报价配对，配对结果无解释力。

用法
    .venv\\Scripts\\python.exe -X utf8 scripts/local_shape_knn_real_quote_ev.py
    .venv\\Scripts\\python.exe -X utf8 scripts/local_shape_knn_real_quote_ev.py --plot
    .venv\\Scripts\\python.exe -X utf8 scripts/local_shape_knn_real_quote_ev.py --self-check-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                       # import 同目录研究脚本（复用其冻结实现）

import shape_segment_knn_720d as sk             # noqa: E402

# sk 模块级已 sys.path.insert(.../src)，此处可安全 import 房内统计口径
from binance_predict.backtest.stats import (    # noqa: E402
    FEE as _STATS_FEE, PREMIUM as _STATS_PREM,
    ev, exact_binomial_p, min_detectable_effect, power_preflight, wilson,
)
from binance_predict.discovery.oos_validator import (  # noqa: E402
    breakeven_win_rate, run_block_ci,
)

OUT = os.path.join(ROOT, "output", "shape_knn_quote_pair")
QUOTE_JSON = os.path.join(ROOT, "prediction_market_samples_online_20260819.json")
TF = "5m"
BAR_MS = 300_000

# ---- 冻结 CFG -------------------------------------------------------------
# kNN 段 9 项必须逐项 == sk.CFG（启动断言），本脚本无权重新预注册检索口径。
CFG = {
    # 复用上一轮冻结的 kNN 口径（只换评估段与 EV 口径，不换检索）
    "seg_len": 12, "seg_step": 1, "resample_m": 12, "norm_mode": "zscore",
    "k_neighbors": 20, "consensus_margin": 0.5, "min_refs": 200,
    "causal_lag": 2, "knn_block": 128,
    # 本次新增：真实报价配对口径
    "decision_sec_main": 30.0,       # 主口径：开窗后 ≤30s 最晚报价（贴近开盘）
    "decision_sec_const": 150.0,     # 对照：房内宪法 V1.1 决策点
    "calib_frac": 0.7,               # 评估段内部再切：前 70% 定阈、后 30% 终裁一次
    "gate_grid": (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15),   # 错配阈值扫描
    "min_n_gate": 200,               # calib 段选 θ 的最小样本门槛
    "min_n_confirm": 60,             # 终裁最小样本（== sk.CFG["n_min_holdout"]）
    "q_floor": 0.01, "q_ceil": 0.99,  # 报价合法区间（越界视为不可成交，剔除并计数）
    "tail_min_sec": 270.0,           # 窗末结果代理下界（offset≥270s 最晚采样；实测最晚 286s）
    "btc_entry_sec": 5.0,            # btc_price 入场取样上界（offset≤5s 最晚采样）
    "xcheck_min_agree": 0.90,        # 自检④「K线标签 ↔ 市场结果代理」最低一致率
    "fee": 0.02, "premium": 0.01,
    "block_boot": 3000, "block_seed": 11,     # run_block_ci 口径（房内默认）
    "min_runs_advisory": 30,         # 自检⑩提示线：run 块低于此数时 bootstrap 自由度偏低
    "consensus_bins": (0.50, 0.55, 0.60, 0.65, 0.70, 0.80),   # 分箱诊断边界
}

# 启动断言：kNN 9 项逐项 == 上一轮冻结值，把口径漂移变硬失败
_KNN_KEYS = ("seg_len", "seg_step", "resample_m", "norm_mode", "k_neighbors",
             "consensus_margin", "min_refs", "causal_lag", "knn_block")
for _k in _KNN_KEYS:
    assert CFG[_k] == sk.CFG[_k], \
        f"kNN 口径漂移：CFG[{_k}]={CFG[_k]} != shape_segment_knn_720d.CFG[{_k}]={sk.CFG[_k]}"
assert CFG["min_n_confirm"] == sk.CFG["n_min_holdout"], "终裁样本门槛与房规不一致"
# 成本口径必须与房内唯一事实源逐字一致（EV 公式分母，错 1pp 即翻转结论）
assert CFG["fee"] == _STATS_FEE, f"FEE 漂移：{CFG['fee']} != {_STATS_FEE}"
assert CFG["premium"] == _STATS_PREM, f"PREMIUM 漂移：{CFG['premium']} != {_STATS_PREM}"

# θ 选择规则：**预先写死**，防「看了 confirm 再定阈」的数据窥探
GATE_SELECT_RULE = (
    "在 calib 段扫 gate_grid，取 EV_B 均值最大且 n ≥ min_n_gate(200) 的 θ；"
    "若无 θ 满足样本门槛，回退 θ=0.0（无门禁）。confirm 段只用此 θ 终裁一次。"
)

EXPLORATORY_NOTE = (
    "评估段（2026-07-26~08-19）与上一轮 shape_segment_knn_720d 的冻结 holdout "
    "（2026-04-14~09-05）时间重叠，且该 holdout 的固定价口径结果已被查看过。"
    "因此本轮全部结论为 exploratory 事后分析，不构成 pre-registered 验证；"
    "calib/confirm 内部再切只恢复部分纪律，不能消除「口径事后选择」的风险。"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def iso_utc(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()


def cfg_sha256() -> str:
    return hashlib.sha256(
        json.dumps(CFG, sort_keys=True, default=list).encode("utf-8")).hexdigest()


def _clean(v):
    """NaN/inf → None；np 标量 → py 标量（JSON/CSV 落盘防污染）。"""
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _fmt(v, spec=".6g") -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        return format(v, spec)
    return str(v)


def _write_csv(path: str, header: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(x) for x in r])


# ============================ 层1：报价加载与取样 ============================

def load_quotes(path: str, tf: str, bar_ms: int) -> dict:
    """报价 JSON → {window_start_ms: (n_samp,4) float64 [t, up_price, down_price, btc_price]}。

    - 只收 market_period==tf ∧ up_price/down_price 双非空且有限的采样；
    - btc_price 缺失记 nan，**不参与筛选**（仅供自检④独立结果交叉验证）；
    - 窗口网格 = t − (t % bar_ms)，与 SentimentWindow.start_time 同为整点 ms；
    - 每窗按 t 升序（quote_pick 取「≤max_sec 内最晚」依赖此序）。
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    acc: dict[int, list] = {}
    n_skip_period = n_skip_null = n_btc_null = 0
    for s in raw:
        if s.get("market_period") != tf:
            n_skip_period += 1
            continue
        t, up, dn = s.get("timestamp"), s.get("up_price"), s.get("down_price")
        if t is None or up is None or dn is None:
            n_skip_null += 1
            continue
        up, dn = float(up), float(dn)
        if not (np.isfinite(up) and np.isfinite(dn)):
            n_skip_null += 1
            continue
        btc = s.get("btc_price")
        try:
            btc = float(btc) if btc is not None else float("nan")
        except (TypeError, ValueError):
            btc = float("nan")
        if not np.isfinite(btc):
            n_btc_null += 1
        t = int(t)
        acc.setdefault(t - (t % bar_ms), []).append((t, up, dn, btc))
    out = {}
    for w in sorted(acc):                        # sorted → 落盘/切分确定性
        lst = sorted(acc[w], key=lambda z: z[0])
        out[w] = np.array(lst, dtype=np.float64)
    n_samp = sum(len(v) for v in out.values())
    log(f"报价加载：{tf} 窗口 {len(out)} 个 | 采样 {n_samp} 条 "
        f"| 跳过 非{tf}={n_skip_period} 缺值={n_skip_null} "
        f"| btc_price 缺失 {n_btc_null}（覆盖率 {1 - n_btc_null / max(1, n_samp):.4f}）")
    return out


def quote_pick(arr, w_ms: int, max_sec: float, col: int) -> tuple:
    """≤max_sec 内最晚采样点（逐字复刻 local_timeslice_winrate.py:290 price_within）。

    arr 已按 t 升序 → 掩码内最后一条即最晚。无采样/全超窗 → (None, None)。
    col: 1=up_price, 2=down_price, 3=btc_price。
    返回 (price, off_sec)：off_sec 为自检②「报价时刻不晚于决策点」留痕。
    """
    if arr is None or len(arr) == 0:
        return None, None
    off = (arr[:, 0] - float(w_ms)) / 1000.0
    hit = np.flatnonzero(off <= max_sec)
    if len(hit) == 0:
        return None, None
    j = int(hit[-1])
    return float(arr[j, col]), float(off[j])


def quote_tail(arr, w_ms: int, min_sec: float, col: int):
    """≥min_sec 内最晚采样点（窗末口径，供独立结果代理）。缺失/非有限 → None。"""
    if arr is None or len(arr) == 0:
        return None
    off = (arr[:, 0] - float(w_ms)) / 1000.0
    hit = np.flatnonzero(off >= min_sec)
    if len(hit) == 0:
        return None
    v = float(arr[int(hit[-1]), col])
    return v if np.isfinite(v) else None


def btc_window_ret(arr, w_ms: int, cfg: dict):
    """窗内 BTC 收益 exit/entry−1（entry=≤btc_entry_sec 最晚、exit=≥tail_min_sec 最晚）。

    **独立结果代理**：btc_price 与 up_price 是不同字段，一致即证明
    「查询段 anchor=i−1 ↔ 市场窗 t[i]」映射未反向。仅供自检④，不参与任何 EV/门禁。
    """
    if arr is None or len(arr) == 0:
        return None
    fin = np.isfinite(arr[:, 3])
    if not fin.any():
        return None
    sub = arr[fin]
    e = quote_pick(sub, w_ms, float(cfg["btc_entry_sec"]), 3)[0]
    x = quote_tail(sub, w_ms, float(cfg["tail_min_sec"]), 3)
    if e is None or x is None or not np.isfinite(e) or e <= 0:
        return None
    return x / e - 1.0


def quote_span_stats(quotes: dict, bar_ms: int) -> dict:
    """报价覆盖体检：窗数、时间跨度、每窗采样数、首条 offset 分位、btc 覆盖率。"""
    ws = np.array(sorted(quotes), dtype=np.int64)
    cnt = np.array([len(quotes[int(w)]) for w in ws], dtype=np.int64)
    first_off = np.array([(quotes[int(w)][0, 0] - float(w)) / 1000.0 for w in ws])
    last_off = np.array([(quotes[int(w)][-1, 0] - float(w)) / 1000.0 for w in ws])
    n_btc = sum(int(np.isfinite(quotes[int(w)][:, 3]).sum()) for w in ws)
    grid = np.diff(ws)
    return {
        "n_windows": int(len(ws)),
        "start_ms": int(ws[0]), "end_ms": int(ws[-1]),
        "start_iso": iso_utc(ws[0]), "end_iso": iso_utc(ws[-1]),
        "days": round(float(ws[-1] - ws[0]) / 86_400_000, 2),
        "samples_total": int(cnt.sum()),
        "samples_per_window_p50": float(np.median(cnt)),
        "samples_per_window_min": int(cnt.min()),
        "first_offset_sec_p50": float(np.median(first_off)),
        "first_offset_sec_p90": float(np.percentile(first_off, 90)),
        "last_offset_sec_p50": float(np.median(last_off)),
        "last_offset_sec_max": float(last_off.max()),
        "btc_price_coverage": round(n_btc / max(1, int(cnt.sum())), 4),
        "grid_gap_mode_ms": int(np.bincount(grid).argmax()) if len(grid) else None,
        "grid_gap_nonconsecutive": int((grid != bar_ms).sum()),
    }


# ============================ 层2：因果检索（复用 sk）============================

def build_retrieval(kl, segs, X, repr_valid, labels, quotes: dict, cfg: dict) -> dict:
    """评估段查询 × 评估段之前 refs 的因果 kNN 检索 + 投票。

    查询定义：报价窗口 W ⇔ K 线索引 i（kl.t[i]==W）⇔ 段 anchor=i−1，
    因 label_up[anchor]=c[anchor+1]>o[anchor+1] ⇔ 窗口 i 收 UP，与市场结算
    outcome（actual_return=exit/entry−1 的符号）同口径。

    refs cap 统一取「最早查询 qs − causal_lag」对应的段数上界 → 对全部查询因果安全，
    且评估段内部互不为 refs（比 expanding 更保守）；cap 只依赖数据边界 → 字节复现。
    """
    t = kl.t
    anchor = segs["anchor"]
    start = segs["start"]
    label_up = labels["label_up"]
    label_valid = labels["label_valid"]

    qw = np.array(sorted(quotes), dtype=np.int64)
    # 窗口 ms → K 线索引（t 升序，searchsorted 后精确校验，防缺口错位）
    pos = np.searchsorted(t, qw)
    inb = pos < len(t)
    match = inb & (t[np.minimum(pos, len(t) - 1)] == qw)
    w_ok, i_ok = qw[match], pos[match]

    # K 线索引 → 段索引（anchor 升序，同样精确校验）
    a_target = i_ok - 1
    spos = np.searchsorted(anchor, a_target)
    inb2 = spos < len(anchor)
    seg_match = inb2 & (anchor[np.minimum(spos, len(anchor) - 1)] == a_target)
    w_sel, i_sel, seg_sel = w_ok[seg_match], i_ok[seg_match], spos[seg_match]

    n0 = len(w_sel)
    # 有效性过滤：查询段表征有效 ∧ 标签有效（doji/断点剔除，与上一轮同口径）
    keep = repr_valid[seg_sel] & label_valid[anchor[seg_sel]]
    w_sel, i_sel, seg_sel = w_sel[keep], i_sel[keep], seg_sel[keep]

    qa = anchor[seg_sel]
    qs = start[seg_sel]
    qs_min = int(qs.min())
    ra_lim = qs_min - int(cfg["causal_lag"])
    cap = int(np.searchsorted(anchor, ra_lim, side="right"))

    ref_seg = np.arange(0, cap, dtype=np.int64)
    ref_ok = repr_valid[ref_seg]
    ref_seg_ok = ref_seg[ref_ok]
    ra = anchor[ref_seg_ok]
    Xr = X[ref_seg_ok]
    ref_up = label_up[ra]
    ref_valid = label_valid[ra]

    nn_idx, nn_sim, n_avail = sk.cosine_topk(
        X[seg_sel], Xr, qa, qs, ra,
        int(cfg["k_neighbors"]), int(cfg["min_refs"]), int(cfg["knn_block"]))
    vote = sk.knn_vote(ref_up, ref_valid, nn_idx, nn_sim, n_avail,
                       int(cfg["min_refs"]), float(cfg["consensus_margin"]))

    return {
        "w_sel": w_sel, "i_sel": i_sel, "seg_sel": seg_sel, "qa": qa, "qs": qs,
        "cap": cap, "ra_lim": ra_lim, "n_refs_pool": int(len(ref_seg_ok)),
        "ra_min": int(ra.min()) if len(ra) else None,
        "ra_max": int(ra.max()) if len(ra) else None,
        "ref_anchor_t_min": int(t[int(ra.min())]) if len(ra) else None,
        "ref_anchor_t_max": int(t[int(ra.max())]) if len(ra) else None,
        "qs_min": qs_min, "qs_min_t": int(t[qs_min]),
        "n_bar_matched": int(len(w_ok)),
        "n_window_matched": int(n0), "n_after_valid": int(len(w_sel)),
        "n_dropped_valid": int(n0 - len(w_sel)),
        "predicted": vote["predicted"], "abstain": vote["abstain"],
        "consensus": vote["consensus"], "n_up": vote["n_up"], "n_dn": vote["n_dn"],
        "top1_sim": vote["top1_sim"], "mean_sim_k": vote["mean_sim_k"],
        "n_avail": n_avail,
        "label_up_q": label_up[qa], "label_valid_q": label_valid[qa],
    }


# ============================ 层3：报价配对 + EV 双口径 ============================

def _ev_pair(win: bool, q: float | None, fee: float, prem: float) -> tuple:
    """(EV_A, EV_B)：A=0.98/q−1 无溢价（local_s5_real_quote_ev 口径）、
    B=0.98/(q+prem)−1 含吃单溢价（misalignment_detector 落库口径，主口径）。
    输均为 −1。q 缺失 → (None, None)。"""
    if q is None or not np.isfinite(q) or q <= 0:
        return None, None
    if win:
        return (1 - fee) / q - 1.0, (1 - fee) / (q + prem) - 1.0
    return -1.0, -1.0


def build_pairs(ret: dict, quotes: dict, kl, cfg: dict) -> dict:
    """逐查询配对真实报价 → 结构化数组（含双口径 EV、错配度、calib/confirm 切分）。"""
    w_sel = ret["w_sel"]
    predicted = ret["predicted"]
    abstain = ret["abstain"]
    consensus = ret["consensus"]
    label_up_q = ret["label_up_q"]
    fee, prem = float(cfg["fee"]), float(cfg["premium"])
    d_main, d_const = float(cfg["decision_sec_main"]), float(cfg["decision_sec_const"])
    q_floor, q_ceil = float(cfg["q_floor"]), float(cfg["q_ceil"])

    n = len(w_sel)
    rows: dict[str, list] = {k: [] for k in (
        "window_start", "seg_id", "anchor_idx", "bar_idx", "direction",
        "consensus", "model_prob", "n_up", "n_dn", "n_avail", "top1_sim", "mean_sim_k",
        "q_main", "q_const", "q_up_main", "q_up_const", "q_off_main", "label_up", "win",
        "misalign", "misalign_const", "ev_a", "ev_b", "ev_a_const", "ev_b_const",
        "ev_at_fixed05", "q_up_tail", "btc_ret", "q_ok", "drop_reason")}

    n_abstain = n_noquote = n_qbad = 0
    for j in range(n):
        W = int(w_sel[j])
        pred = int(predicted[j])
        if bool(abstain[j]) or pred == 0:
            n_abstain += 1
            continue
        arr = quotes.get(W)
        col = 1 if pred == 1 else 2               # 1=up_price, 2=down_price
        q_main, off_main = quote_pick(arr, W, d_main, col)
        q_const, _ = quote_pick(arr, W, d_const, col)
        q_up_main, _ = quote_pick(arr, W, d_main, 1)    # UP token 报价（诊断错配用，不论方向）
        q_up_const, _ = quote_pick(arr, W, d_const, 1)
        if q_main is None:
            n_noquote += 1
            continue
        if not (q_floor <= q_main <= q_ceil):
            n_qbad += 1
            continue
        direction = "UP" if pred == 1 else "DOWN"
        win = bool(label_up_q[j]) == (pred == 1)
        cp = float(consensus[j])
        model_prob = cp if pred == 1 else 1.0 - cp
        ev_a, ev_b = _ev_pair(win, q_main, fee, prem)
        ev_a_c, ev_b_c = _ev_pair(win, q_const, fee, prem)
        rows["window_start"].append(W)
        rows["seg_id"].append(int(ret["seg_sel"][j]))
        rows["anchor_idx"].append(int(ret["qa"][j]))
        rows["bar_idx"].append(int(ret["i_sel"][j]))
        rows["direction"].append(direction)
        rows["consensus"].append(cp)
        rows["model_prob"].append(model_prob)
        rows["n_up"].append(int(ret["n_up"][j]))
        rows["n_dn"].append(int(ret["n_dn"][j]))
        rows["n_avail"].append(int(ret["n_avail"][j]))
        rows["top1_sim"].append(_clean(float(ret["top1_sim"][j])))
        rows["mean_sim_k"].append(_clean(float(ret["mean_sim_k"][j])))
        rows["q_main"].append(q_main)
        rows["q_const"].append(_clean(q_const))
        rows["q_up_main"].append(_clean(q_up_main))
        rows["q_up_const"].append(_clean(q_up_const))
        rows["q_off_main"].append(_clean(off_main))
        rows["label_up"].append(bool(label_up_q[j]))
        rows["win"].append(bool(win))
        rows["misalign"].append(model_prob - q_main)
        rows["misalign_const"].append(_clean(None if q_const is None else model_prob - q_const))
        rows["ev_a"].append(_clean(ev_a))
        rows["ev_b"].append(_clean(ev_b))
        rows["ev_a_const"].append(_clean(ev_a_c))
        rows["ev_b_const"].append(_clean(ev_b_c))
        # 对照：上一轮固定 entry=0.5 口径（房内 ev(p,e)），同 win 序列
        rows["ev_at_fixed05"].append(ev(1.0, 0.5) if win else ev(0.0, 0.5))
        # 独立结果代理（仅供自检④，不参与 EV/门禁）
        rows["q_up_tail"].append(_clean(quote_tail(arr, W, float(cfg["tail_min_sec"]), 1)))
        rows["btc_ret"].append(_clean(btc_window_ret(arr, W, cfg)))
        rows["q_ok"].append(True)
        rows["drop_reason"].append("")

    n_paired = len(rows["window_start"])
    # calib/confirm 按时间切（前 calib_frac 定阈、后段终裁一次）
    ws = np.array(rows["window_start"], dtype=np.int64)
    order = np.argsort(ws, kind="stable")
    n_calib = int(round(n_paired * float(cfg["calib_frac"])))
    split = np.array(["confirm"] * n_paired, dtype=object)
    split[order[:n_calib]] = "calib"
    rows["split"] = split.tolist()

    log(f"配对：查询 {n} → 弃权 {n_abstain} / 无报价 {n_noquote} / 报价越界 {n_qbad} "
        f"→ 成对 {n_paired}（calib {int((split == 'calib').sum())} / "
        f"confirm {int((split == 'confirm').sum())}）")
    return {**rows, "_counts": {"n_query": n, "n_abstain": n_abstain,
                                "n_noquote": n_noquote, "n_qbad": n_qbad,
                                "n_paired": n_paired}}


# ============================ 层4：诊断统计 ============================

def _n_runs(run_key: np.ndarray) -> int:
    """run 块数（相邻 run_key 差 1 视为同一块）。==1 时 run 级 bootstrap 退化。"""
    rk = np.asarray(run_key, dtype=np.int64)
    if len(rk) == 0:
        return 0
    return int(np.flatnonzero(np.diff(rk) != 1).size + 1)


def _agg(win: np.ndarray, ev_b: np.ndarray, ev_a: np.ndarray, q: np.ndarray,
         run_key: np.ndarray, cfg: dict) -> dict:
    """一组配对样本的汇总统计（胜率 + Wilson + block CI + EV 双口径 + MDE）。

    run_key 必须是 **bar 索引**，不是本数组位置：房内 run_block_ci 把「相邻命中」
    合并为一个 run 以处理重叠依赖，而本脚本相邻窗口共享 11/12 根 K 线，依赖极强；
    弃权/缺报价造成的 bar 缺口才是正确的 run 边界。若误传数组位置，密集子集
    （如整个 confirm 段）会全部并成 1 个 run → bootstrap 无重采样自由度 → CI 塌为点估计。
    """
    prem_arg = float(cfg["premium"])
    n = int(len(win))
    if n == 0:
        return {"n": 0}
    k = int(win.sum())
    wr = k / n
    wlo, whi = wilson(wr, n)
    blo, bhi = run_block_ci(run_key, win.astype(np.int64),
                            b=int(cfg["block_boot"]), seed=int(cfg["block_seed"]))
    evb = np.asarray(ev_b, dtype=np.float64)
    eva = np.asarray(ev_a, dtype=np.float64)
    # EV 均值的 run 级 bootstrap CI（与胜率同口径处理相邻重叠依赖）
    ev_lo, ev_hi = _run_mean_ci(run_key, evb, int(cfg["block_boot"]),
                                int(cfg["block_seed"]))
    return {
        "n": n, "k_win": k, "win_rate": wr,
        "wilson_low": wlo, "wilson_high": whi,
        "block_ci_low": blo, "block_ci_high": bhi, "n_runs": _n_runs(run_key),
        "p_up_0.5": exact_binomial_p(k, n, 0.5),
        "p_method": "exact" if n <= 2000 else "normal_approx",
        "mde_pp": round(min_detectable_effect(n) * 100.0, 3),
        "ev_b_mean": float(evb.mean()), "ev_b_sd": float(evb.std(ddof=1)) if n > 1 else None,
        "ev_b_ci_low": ev_lo, "ev_b_ci_high": ev_hi,
        "ev_b_sum": float(evb.sum()),
        "ev_a_mean": float(eva.mean()),
        "q_mean": float(np.asarray(q, dtype=np.float64).mean()),
        "q_p50": float(np.median(np.asarray(q, dtype=np.float64))),
        # 打平胜率两口径（breakeven_win_rate(e)=(e+PREMIUM)/(1-FEE)）：
        #   主口径 EV_B 的成本是 q+PREMIUM → 传 q；对照 EV_A 成本是 q → 传 q-PREMIUM
        "be_b_at_qmean": float(breakeven_win_rate(float(np.asarray(q, dtype=np.float64).mean()))),
        "be_a_at_qmean": float(breakeven_win_rate(
            float(np.asarray(q, dtype=np.float64).mean()) - prem_arg)),
        "power_note": power_preflight(n, abs(wr - 0.5) * 100.0, 0.5),
    }


def _run_mean_ci(hit_pos: np.ndarray, vals: np.ndarray, b: int, seed: int) -> tuple:
    """run 级 bootstrap 均值 CI（run_block_ci 的连续量版本，同分组逻辑）。

    hit_pos 语义同 _agg 的 run_key（**bar 索引**）：相邻合并为 run，run 级重采样。
    """
    vals = np.asarray(vals, dtype=np.float64)
    hit_pos = np.asarray(hit_pos, dtype=np.int64)
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    brk = np.flatnonzero(np.diff(hit_pos) != 1) + 1
    idx = np.r_[0, brk, len(vals)]
    run_sum = np.add.reduceat(vals, np.r_[0, brk])
    run_n = np.diff(idx).astype(np.float64)
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, len(run_sum), size=(b, len(run_sum)))
    means = run_sum[sel].sum(axis=1) / run_n[sel].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def diagnose_bins(pairs: dict, cfg: dict) -> list:
    """consensus 分箱诊断（**本轮最有诊断价值的输出**）。

    每箱给：n / 实际胜率 + Wilson / 真实报价 q 均值 / 打平线 / EV_B / 错配度均值。
    直接回答「高 consensus 箱的胜率是否显著高于市场报价隐含概率」。
    只用 calib 段（confirm 段留给终裁，防分箱窥探）。
    """
    edges = list(cfg["consensus_bins"])
    cp = np.asarray(pairs["consensus"], dtype=np.float64)
    split = np.asarray(pairs["split"], dtype=object)
    m_cal = split == "calib"
    win = np.asarray(pairs["win"], dtype=bool)
    q = np.asarray(pairs["q_main"], dtype=np.float64)
    evb = np.asarray(pairs["ev_b"], dtype=np.float64)
    pos = np.asarray(pairs["bar_idx"], dtype=np.int64)   # run 分组键 = bar 索引

    bounds = sorted({0.5, *edges, 1.01})   # 去重：CFG 首边界就是 0.5，否则产生退化空箱
    rows = []
    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        m = m_cal & (cp >= lo) & (cp < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.2f},{hi:.2f})", "n": 0})
            continue
        st = _agg(win[m], evb[m], np.asarray(pairs["ev_a"], dtype=np.float64)[m],
                  q[m], pos[m], cfg)
        st["bin"] = f"[{lo:.2f},{hi:.2f})"
        st["misalign_mean"] = float(np.mean(cp[m] - q[m]))
        rows.append(st)
    return rows


def gate_scan(pairs: dict, cfg: dict, split_want: str) -> list:
    """错配门禁扫描：misalign ≥ θ 才入场。返回每 θ 的 n / 胜率 / EV_B / CI。"""
    split = np.asarray(pairs["split"], dtype=object)
    m = split == split_want
    mis = np.asarray(pairs["misalign"], dtype=np.float64)
    win = np.asarray(pairs["win"], dtype=bool)
    evb = np.asarray(pairs["ev_b"], dtype=np.float64)
    eva = np.asarray(pairs["ev_a"], dtype=np.float64)
    q = np.asarray(pairs["q_main"], dtype=np.float64)
    bidx = np.asarray(pairs["bar_idx"], dtype=np.int64)   # run 分组键 = bar 索引
    out = []
    for th in cfg["gate_grid"]:
        sel = m & (mis >= float(th))
        n = int(sel.sum())
        row = {"theta": float(th), "n": n}
        if n > 0:
            st = _agg(win[sel], evb[sel], eva[sel], q[sel], bidx[sel], cfg)
            row.update(st)
            row["n_frac"] = round(n / max(1, int(m.sum())), 4)
            row["dir_up_frac"] = float(np.mean(
                np.asarray(pairs["direction"], dtype=object)[sel] == "UP"))
        out.append(row)
    return out


def select_theta(scan: list, cfg: dict) -> tuple:
    """按 GATE_SELECT_RULE 在 calib 扫描结果上选 θ（规则预先写死，不看 confirm）。"""
    best, best_ev = None, None
    for r in scan:
        if r["n"] < int(cfg["min_n_gate"]):
            continue
        e = r.get("ev_b_mean")
        if e is None or not np.isfinite(e):
            continue
        if best_ev is None or e > best_ev:
            best, best_ev = r, e
    if best is None:
        return 0.0, "fallback_no_theta_met_min_n", None
    return float(best["theta"]), "max_ev_b_above_min_n", best


# ============================ 层5：confirm 终裁（只跑一次）============================

def _farr(pairs: dict, key: str) -> np.ndarray:
    """pairs 字段（list，可能含 None）→ float64 数组（None→nan）。"""
    return np.array([np.nan if v is None else float(v) for v in pairs[key]],
                    dtype=np.float64)


def _agg_subset(pairs: dict, cfg: dict, mask) -> dict:
    """按布尔掩码取子集做 _agg（run_key 用 **bar 索引**，而非筛后数组位置）。"""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    pos = np.asarray(pairs["bar_idx"], dtype=np.int64)[m]   # run 分组键 = bar 索引
    return _agg(np.asarray(pairs["win"], dtype=bool)[m], _farr(pairs, "ev_b")[m],
                _farr(pairs, "ev_a")[m], _farr(pairs, "q_main")[m], pos, cfg)


def _fixed05_agg(pairs: dict, mask) -> dict:
    """同一批样本换成上一轮 entry_price=0.5 固定口径的 EV（房内 stats.ev）。"""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    f = _farr(pairs, "ev_at_fixed05")[m]
    return {"n": n, "ev_fixed05_mean": float(f.mean()), "ev_fixed05_sum": float(f.sum())}


def _verdict_of(st: dict, cfg: dict) -> tuple:
    """终裁判定（阀值预先写死，不看结果调）。返回 (verdict, reason)。"""
    n = int(st.get("n", 0))
    if n < int(cfg["min_n_confirm"]):
        return ("INSUFFICIENT_SAMPLE",
                f"n={n} < min_n_confirm={cfg['min_n_confirm']}（房规 holdout 最小样本），不终裁")
    if int(st.get("n_runs", 0)) <= 1:
        return ("CI_DEGENERATE",
                f"n={n} 但 run 分组只有 {st.get('n_runs')} 块——run 级 bootstrap 无重采样"
                f"自由度，CI 塌为点估计，结论不可用")
    mu = float(st["ev_b_mean"])
    lo = float(st["ev_b_ci_low"])
    hi = float(st["ev_b_ci_high"])
    wr = float(st["win_rate"])
    be = float(st["be_b_at_qmean"])
    tail = f"胜率 {wr:.4f} vs 打平线 {be:.4f}（{'过' if wr > be else '未过'}）"
    if np.isfinite(lo) and lo > 0.0:
        return ("EV_POSITIVE_CI_EXCLUDES_ZERO",
                f"EV_B 均值 {mu:+.4f}，run 级 95% CI 下界 {lo:+.4f} > 0；{tail}")
    if mu > 0.0:
        return ("EV_MEAN_POSITIVE_CI_CROSSES_ZERO",
                f"EV_B 均值 {mu:+.4f} > 0 但 CI [{lo:+.4f}, {hi:+.4f}] 跨零，"
                f"样本量不足以排除零效应；{tail}")
    return ("EV_NOT_POSITIVE",
            f"EV_B 均值 {mu:+.4f} ≤ 0（CI [{lo:+.4f}, {hi:+.4f}]）；{tail}")


def confirm_verdict(pairs: dict, cfg: dict, theta: float, rule: str) -> dict:
    """confirm 段终裁（**只跑一次**）：θ 已在 calib 段按 GATE_SELECT_RULE 选定。

    同段并列三口径，但**只有 gate 行是终裁对象**，另两行是对照：
      gate     misalign ≥ θ（选定 θ）           —— 终裁
      no_gate  θ=0（confirm 全部配对样本）      —— 「不加门禁会怎样」基准
      fixed05  同 gate 样本、entry=0.5 固定价   —— 「换成真实报价改变了什么」
    """
    split = np.asarray(pairs["split"], dtype=object)
    mis = _farr(pairs, "misalign")
    ws = np.asarray(pairs["window_start"], dtype=np.int64)
    m_conf = split == "confirm"
    m_gate = m_conf & (mis >= float(theta))
    st_gate = _agg_subset(pairs, cfg, m_gate)
    st_base = _agg_subset(pairs, cfg, m_conf)
    verdict, reason = _verdict_of(st_gate, cfg)
    n_conf = int(m_conf.sum())
    return {
        "theta": float(theta),
        "theta_rule": rule,
        "theta_rule_text": GATE_SELECT_RULE,
        "min_n_confirm": int(cfg["min_n_confirm"]),
        "confirm_window": {
            "n_calib_paired": int((split == "calib").sum()),
            "n_confirm_paired": n_conf,
            "first_iso": iso_utc(int(ws[m_conf].min())) if n_conf else None,
            "last_iso": iso_utc(int(ws[m_conf].max())) if n_conf else None,
        },
        "gate": st_gate,
        "no_gate": st_base,
        "fixed05_on_gate_sample": _fixed05_agg(pairs, m_gate),
        "verdict": verdict,
        "reason": reason,
        "exploratory_note": EXPLORATORY_NOTE,
    }


# ============= 层6：自检（因果 / 无未来函数 / 口径一致 / 交叉验证）=============

def _self_check(kl, ret: dict, pairs: dict, quotes: dict, cfg: dict,
                theta: float, rule: str, verdict: dict) -> list:
    """自检清单（⓪~⑩，④ 拆 a/b）。任何一项 ok=False 都应视为「结论不可信」，
    而非「细节瑕疵」。

    ①因果：refs 标签根严格早于评估段首根（cosine_topk 出口断言之外的全局复核）
    ②报价无未来函数：决策价 offset∈[0,decision_sec] 且能从原始数组独立重算
    ③EV 公式与房内 stats.ev 逐点一致（双口径各自对齐）
    ④a K线标签 ↔ btc_price 独立结果代理（验映射未反向）
    ④b K线标签 ↔ 窗末 up_price 收敛代理（第二独立源）
    ⑤calib/confirm 无时间重叠
    ⑥窗口唯一 + window↔bar↔anchor 映射自洽
    ⑦UP 行取的就是 up_price 列（方向-列一致）
    ⑧θ 溯源合法（在 gate_grid 内、规则名合法、只由 calib 决定）
    ⑨consensus/model_prob/misalign 代数自洽 + n_avail ≥ min_refs
    ⑩run 分组非退化：bar_idx 严格升序（run_block_ci 前提）+ n_runs>1 + CI 未塌
      为点估计 + 落盘 n_runs 可从 pairs 独立重算（专防「误传数组位置」类隐性 bug）
    """
    checks: list = []

    def add(name: str, ok, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    t = kl.t
    ws = np.asarray(pairs["window_start"], dtype=np.int64)
    n_pair = len(ws)
    if n_pair == 0:
        add("⓪配对样本非空", False, "n_paired=0，后续全部自检无意义")
        return checks
    add("⓪配对样本非空", True, f"n_paired={n_pair}")

    # ---- ①因果 ----
    ra_max, qs_min = ret["ra_max"], int(ret["qs"].min())
    ok1 = ra_max is not None and int(ra_max) + 1 < qs_min
    add("①因果：refs 标签根严格早于评估段首根", ok1,
        f"refs 池 {ret['n_refs_pool']} 段（cap={ret['cap']}，ra_lim={ret['ra_lim']}）；"
        f"最大 ref anchor={ra_max}（t={iso_utc(t[int(ra_max)])}），其标签根 "
        f"{int(ra_max) + 1}（t={iso_utc(t[int(ra_max) + 1])}）< 最早查询段首根 "
        f"{qs_min}（t={iso_utc(t[qs_min])}）；评估段内部互不为 refs。"
        f"cosine_topk 出口三条硬断言（ra≤qs−lag / 无自匹配 / ra+1<qs）已通过")

    # ---- ②报价无未来函数（独立重算，不信 pairs 自己的记录）----
    d_main = float(cfg["decision_sec_main"])
    off = _farr(pairs, "q_off_main")
    qm = _farr(pairs, "q_main")
    dirn = np.asarray(pairs["direction"], dtype=object)
    bad_off = int(((off < 0.0) | (off > d_main) | ~np.isfinite(off)).sum())
    mism = 0
    for j in range(n_pair):
        W = int(ws[j])
        col = 1 if dirn[j] == "UP" else 2
        p2, o2 = quote_pick(quotes.get(W), W, d_main, col)
        if p2 is None or abs(p2 - qm[j]) > 1e-12 or o2 is None or abs(o2 - off[j]) > 1e-12:
            mism += 1
    add("②报价无未来函数：offset∈[0,decision_sec] 且可独立重算",
        bad_off == 0 and mism == 0,
        f"decision_sec={d_main}s；offset 越界 {bad_off} 条；独立重算不符 {mism} 条；"
        f"offset p50={float(np.median(off)):.1f}s max={float(off.max()):.1f}s"
        f"（对照 decision_sec_const={cfg['decision_sec_const']}s）")

    # ---- ③EV 公式与房内 stats.ev 一致 ----
    fee, prem = float(cfg["fee"]), float(cfg["premium"])
    worst = 0.0
    for e0 in np.linspace(0.02, 0.98, 97):
        e = float(e0)
        a, b = _ev_pair(True, e, fee, prem)
        a0, b0 = _ev_pair(False, e, fee, prem)
        worst = max(worst, abs(b - ev(1.0, e)), abs(b0 - ev(0.0, e)),
                    abs(a - ((1.0 - _STATS_FEE) / e - 1.0)), abs(a0 - (-1.0)))
    add("③EV 公式与房内 stats.ev 逐点一致", worst < 1e-12,
        f"97 点网格 e∈[0.02,0.98] 上最大偏差 {worst:.3e}；"
        f"EV_B(win,e)≡ev(1.0,e)=(1−FEE)/(e+PREMIUM)−1、EV_B(lose,e)≡ev(0.0,e)=−1、"
        f"EV_A(win,e)=(1−FEE)/e−1（无溢价口径，local_s5_real_quote_ev 同源）；"
        f"FEE={_STATS_FEE} PREMIUM={_STATS_PREM}（启动断言已锁）")

    # ---- ④a K线标签 ↔ btc_price 独立结果代理 ----
    lu = np.asarray(pairs["label_up"], dtype=bool)
    br = _farr(pairs, "btc_ret")
    msk = np.isfinite(br) & (np.abs(br) > 1e-12)
    agree_a = float(np.mean(lu[msk] == (br[msk] > 0.0))) if int(msk.sum()) else None
    thr = float(cfg["xcheck_min_agree"])
    add("④a K线标签 ↔ btc_price 独立结果代理",
        agree_a is not None and agree_a >= thr,
        f"可对照 {int(msk.sum())}/{n_pair} 窗（btc_price 双端齐且非零变动）；"
        f"一致率 {('%.4f' % agree_a) if agree_a is not None else 'N/A'} ≥ 门槛 {thr}。"
        f"label_up[i−1]=c[i]>o[i] 应与「窗内 BTC 涨」同号——一致即证明"
        f"「查询段 anchor=i−1 ↔ 市场窗 t[i]」映射未反向")

    # ---- ④b K线标签 ↔ 窗末 up_price 收敛代理 ----
    tq = _farr(pairs, "q_up_tail")
    msk2 = np.isfinite(tq) & ((tq > 0.9) | (tq < 0.1))
    agree_b = float(np.mean(lu[msk2] == (tq[msk2] > 0.9))) if int(msk2.sum()) else None
    add("④b K线标签 ↔ 窗末 up_price 收敛代理（第二独立源）",
        agree_b is not None and agree_b >= thr,
        f"可对照 {int(msk2.sum())}/{n_pair} 窗（窗末 offset≥{cfg['tail_min_sec']}s 且 "
        f"up_price>0.9 或 <0.1，即市场自身已收敛）；一致率 "
        f"{('%.4f' % agree_b) if agree_b is not None else 'N/A'} ≥ 门槛 {thr}")

    # ---- ⑤calib/confirm 无时间重叠 ----
    split = np.asarray(pairs["split"], dtype=object)
    cal, conf = ws[split == "calib"], ws[split == "confirm"]
    ok5 = len(cal) > 0 and len(conf) > 0 and int(cal.max()) < int(conf.min())
    add("⑤calib/confirm 无时间重叠（θ 只在 calib 选、confirm 只终裁一次）", ok5,
        f"calib n={len(cal)} [{iso_utc(int(cal.min())) if len(cal) else '-'} ~ "
        f"{iso_utc(int(cal.max())) if len(cal) else '-'}]；confirm n={len(conf)} "
        f"[{iso_utc(int(conf.min())) if len(conf) else '-'} ~ "
        f"{iso_utc(int(conf.max())) if len(conf) else '-'}]；calib_frac={cfg['calib_frac']}")

    # ---- ⑥窗口唯一 + window↔bar↔anchor 映射自洽 ----
    bidx = np.asarray(pairs["bar_idx"], dtype=np.int64)
    ok6 = (len(np.unique(ws)) == n_pair) and bool(np.all(t[bidx] == ws)) \
        and bool(np.all(np.asarray(pairs["anchor_idx"], dtype=np.int64) == bidx - 1))
    add("⑥窗口唯一 + window↔bar↔anchor 映射自洽", ok6,
        f"唯一窗口 {len(np.unique(ws))}/{n_pair}；t[bar_idx]==window_start 全等；"
        f"anchor_idx==bar_idx−1 全等（label_up[anchor] 即窗口结果）")

    # ---- ⑦UP 行取的就是 up_price 列 ----
    m_up = dirn == "UP"
    qu = _farr(pairs, "q_up_main")
    ok7 = bool(np.allclose(qm[m_up], qu[m_up], atol=1e-12)) if int(m_up.sum()) else False
    add("⑦方向-报价列一致（UP 行 q_main==up_price）", ok7,
        f"UP 行 {int(m_up.sum())} / DOWN 行 {int((~m_up).sum())}；"
        f"UP 行 q_main 与 q_up_main 全等；DOWN 行取 down_price（由②独立重算覆盖）")

    # ---- ⑧θ 溯源合法 ----
    grid = [float(x) for x in cfg["gate_grid"]]
    ok8 = rule in ("max_ev_b_above_min_n", "fallback_no_theta_met_min_n") \
        and float(theta) in grid
    add("⑧θ 溯源合法（gate_grid 内 + 规则名合法 + 只由 calib 决定）", ok8,
        f"θ={theta} 规则={rule}；gate_grid={grid}；min_n_gate={cfg['min_n_gate']}；"
        f"规则文本预先写死于 GATE_SELECT_RULE，confirm 段未参与选 θ")

    # ---- ⑨代数自洽 ----
    cp = _farr(pairs, "consensus")
    mp = _farr(pairs, "model_prob")
    mis = _farr(pairs, "misalign")
    nav = np.asarray(pairs["n_avail"], dtype=np.int64)
    e_mp = np.where(dirn == "UP", cp, 1.0 - cp)
    ok9 = bool(np.allclose(mp, e_mp, atol=1e-12)) \
        and bool(np.allclose(mis, mp - qm, atol=1e-12)) \
        and bool(np.all(nav >= int(cfg["min_refs"]))) \
        and bool(np.all((cp >= 0.0) & (cp <= 1.0)))
    add("⑨consensus/model_prob/misalign 代数自洽 + n_avail≥min_refs", ok9,
        f"model_prob==consensus(UP)/1−consensus(DOWN) 全等；"
        f"misalign==model_prob−q_main 全等；n_avail 最小 {int(nav.min())} ≥ "
        f"min_refs={cfg['min_refs']}；consensus∈[0,1]")

    # ---- ⑩run 分组非退化（CI 未塌）----
    # 本项专防一类隐性 bug：run_block_ci 要求 hit_idx 能反映时间邻接，若误传
    # 「筛后数组位置」，任何时间连续的子集（如整个 confirm 段）会并成 1 个 run，
    # bootstrap 失去重采样自由度，CI 静默塌为点估计——数字看上去完全正常。
    rec_g, rec_b = verdict["gate"], verdict["no_gate"]
    m_conf10 = split == "confirm"
    m_gate10 = m_conf10 & (mis >= float(theta))
    nr_gate = _n_runs(bidx[m_gate10])
    nr_base = _n_runs(bidx[m_conf10])
    ci_open = all(float(s[f"{p}_ci_low"]) < float(s[f"{p}_ci_high"])
                  for s in (rec_g, rec_b) for p in ("block", "ev_b")
                  if int(s.get("n", 0)))
    ok10 = (bool(np.all(np.diff(bidx) >= 1))
            and nr_gate > 1 and nr_base > 1
            and nr_gate == int(rec_g.get("n_runs", -1))
            and nr_base == int(rec_b.get("n_runs", -1))
            and ci_open)
    adv = int(cfg["min_runs_advisory"])
    add("⑩run 分组非退化（bar_idx 升序 + n_runs>1 + CI 未塌 + 落盘值可重算）", ok10,
        f"bar_idx 严格升序（np.diff≥1 全成立，run_block_ci 前提）；"
        f"独立重算 gate n_runs={nr_gate}（n={int(rec_g.get('n', 0))}）、"
        f"no_gate n_runs={nr_base}（n={int(rec_b.get('n', 0))}），"
        f"均 >1 且与落盘 n_runs 逐字一致；block/ev_b CI 上下界严格不等（未塌）。"
        + (f"⚠️ 提示：min(n_runs)={min(nr_gate, nr_base)} < {adv}，run 级 bootstrap "
           f"自由度偏低，CI 偏宽且不稳定，不宜当精确区间读"
           if min(nr_gate, nr_base) < adv
           else f"min(n_runs)={min(nr_gate, nr_base)} ≥ {adv}，bootstrap 自由度充足"))

    return checks


def self_check_pass(checks: list) -> bool:
    return all(bool(c["ok"]) for c in checks)


# ============================ 层7：落盘 + 报告 + main ============================

SK_CACHE = os.path.join(sk.OUT0, "shape_segment_knn_720d")   # 复用上一轮分段 npz 缓存

PAIR_COLS = [
    "window_start", "window_iso", "split", "seg_id", "anchor_idx", "bar_idx", "direction",
    "consensus", "model_prob", "misalign", "n_up", "n_dn", "n_avail", "top1_sim",
    "mean_sim_k", "q_main", "q_off_main", "q_const", "misalign_const", "q_up_main",
    "q_up_const", "q_up_tail", "btc_ret", "label_up", "win", "ev_a", "ev_b",
    "ev_a_const", "ev_b_const", "ev_at_fixed05", "q_ok", "drop_reason",
]

AGG_COLS = [
    "n", "k_win", "win_rate", "wilson_low", "wilson_high", "block_ci_low", "block_ci_high",
    "n_runs", "p_up_0.5", "p_method", "mde_pp", "ev_b_mean", "ev_b_sd", "ev_b_ci_low",
    "ev_b_ci_high", "ev_b_sum", "ev_a_mean", "q_mean", "q_p50", "be_b_at_qmean",
    "be_a_at_qmean",
]

# 产物清单的中文说明（report.md 第 10 节用；report.md 自身在写盘时才入 paths，故静态补一行）
_PATH_DESC = {
    "pairs_csv": "逐窗配对明细（每行=一个报价窗×一个 kNN 判决，含双口径 EV 与丢弃原因）",
    "bins_csv": "consensus 分箱诊断（**仅 calib 段**，本轮最有诊断价值的表）",
    "scan_calib_csv": "错配门禁 θ 扫描（calib 段）——**唯一参与选 θ 的表**",
    "scan_confirm_csv": "错配门禁 θ 扫描（confirm 段）——终裁后才算，仅透明留痕",
    "verdict_json": "confirm 终裁结果（gate / no_gate / fixed05 三口径 + 裁决与理由）",
    "selfcheck_json": "自检清单逐项结果与证据（all_pass=false 则全部结论不可信）",
    "run_config_json": "冻结口径快照：CFG + cfg_sha256 + 漏斗计数 + 复现命令",
    "plot_png": "诊断图：q 分布 / consensus 分箱胜率 vs 报价 / θ 扫描曲线",
    "report_md": "本报告（含生成时间戳，不参与字节复现比对）",
}


def _jsonable(o):
    """递归转 JSON 安全对象：np 标量→py、NaN/inf→None、tuple→list。

    NaN 写入 JSON 会产生非法字面量 `NaN`，必须在落盘前收敛为 null。
    """
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _dicts_to_csv(path: str, rows: list, lead: list, extra: list = ()) -> None:
    """dict 列表 → CSV（表头 = lead + extra + AGG_COLS，缺失列置空）。"""
    header = list(lead) + list(extra) + [c for c in AGG_COLS if c not in lead + list(extra)]
    out = [[r.get(c) for c in header] for r in rows]
    _write_csv(path, header, out)


def _md_table(header: list, rows: list) -> str:
    """markdown 表格（空表也要输出表头，方便 diff）。"""
    line = "| " + " | ".join(str(h) for h in header) + " |"
    sep = "|" + "|".join(["---"] * len(header)) + "|"
    body = ["| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in r) + " |"
            for r in rows]
    return "\n".join([line, sep] + body)


def _pct(v, nd: int = 2):
    return "" if v is None else f"{float(v) * 100:.{nd}f}%"


def _sgn(v, nd: int = 4):
    return "" if v is None else f"{float(v):+.{nd}f}"


def write_outputs(out: str, pairs: dict, bins: list, scan_cal: list, scan_conf: list,
                  verdict: dict, checks: list, run_cfg: dict) -> dict:
    """全部产物落盘（CSV 不含墙钟时间 → 可字节复现；run_config.json 含）。"""
    os.makedirs(out, exist_ok=True)
    paths = {}
    ws = np.asarray(pairs["window_start"], dtype=np.int64)

    rows = []
    for j in range(len(ws)):
        r = {c: pairs[c][j] for c in PAIR_COLS if c != "window_iso"}
        r["window_iso"] = iso_utc(int(ws[j]))
        rows.append([r[c] for c in PAIR_COLS])
    p = os.path.join(out, f"quote_pairs_{TF}.csv")
    _write_csv(p, PAIR_COLS, rows)
    paths["pairs_csv"] = p

    p = os.path.join(out, f"consensus_bins_calib_{TF}.csv")
    _dicts_to_csv(p, bins, lead=["bin"], extra=["misalign_mean"])
    paths["bins_csv"] = p

    p = os.path.join(out, f"gate_scan_calib_{TF}.csv")
    _dicts_to_csv(p, scan_cal, lead=["theta"], extra=["n_frac", "dir_up_frac"])
    paths["scan_calib_csv"] = p

    p = os.path.join(out, f"gate_scan_confirm_{TF}.csv")
    _dicts_to_csv(p, scan_conf, lead=["theta"], extra=["n_frac", "dir_up_frac"])
    paths["scan_confirm_csv"] = p

    p = os.path.join(out, f"confirm_verdict_{TF}.json")
    _write_json(p, verdict)
    paths["verdict_json"] = p

    p = os.path.join(out, f"self_check_{TF}.json")
    _write_json(p, {"all_pass": self_check_pass(checks), "n_checks": len(checks),
                    "checks": checks})
    paths["selfcheck_json"] = p

    p = os.path.join(out, "run_config.json")
    _write_json(p, run_cfg)
    paths["run_config_json"] = p

    log(f"产物已落盘 {out}（{len(paths)} 个文件）")
    return paths


def maybe_plot(out: str, bins: list, scan_cal: list, scan_conf: list,
               theta: float, verdict: dict) -> str | None:
    """诊断图：左=consensus 分箱胜率 vs 市场报价隐含概率；右=门禁扫描 EV_B。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:                                   # noqa: BLE001
        log(f"绘图跳过（matplotlib 不可用：{ex}）")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    ax = axes[0]
    b = [r for r in bins if int(r.get("n", 0)) > 0]
    if b:
        x = np.arange(len(b))
        wr = np.array([float(r["win_rate"]) for r in b])
        lo = np.array([float(r["wilson_low"]) for r in b])
        hi = np.array([float(r["wilson_high"]) for r in b])
        q = np.array([float(r["q_mean"]) for r in b])
        be = np.array([float(r["be_b_at_qmean"]) for r in b])
        ax.errorbar(x, wr, yerr=[wr - lo, hi - wr], fmt="o-", capsize=4,
                    label="kNN win_rate (Wilson 95%)", color="#1f77b4")
        ax.plot(x, q, "s--", label="market q_mean (real quote)", color="#ff7f0e")
        ax.plot(x, be, "^:", label="breakeven @ q_mean", color="#d62728")
        ax.axhline(0.5, color="#888", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['bin']}\nn={r['n']}" for r in b], fontsize=8)
        for i, r in enumerate(b):
            ax.annotate(f"{r['win_rate'] * 100:.2f}%\nq={r['q_mean']:.3f}",
                        (i, float(hi[i])), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    ax.set_title(f"consensus bins (calib) — win_rate vs market quote\n{TF}", fontsize=10)
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for scan, lab, col in ((scan_cal, "calib", "#1f77b4"), (scan_conf, "confirm", "#2ca02c")):
        s = [r for r in scan if int(r.get("n", 0)) > 0]
        if not s:
            continue
        th = np.array([float(r["theta"]) for r in s])
        mu = np.array([float(r["ev_b_mean"]) for r in s])
        lo = np.array([float(r["ev_b_ci_low"]) for r in s])
        hi = np.array([float(r["ev_b_ci_high"]) for r in s])
        ax.plot(th, mu, "o-", label=f"{lab} EV_B mean", color=col)
        ax.fill_between(th, lo, hi, alpha=0.18, color=col, label=f"{lab} run-boot 95% CI")
        for r in s:
            ax.annotate(f"n={r['n']}", (float(r["theta"]), float(r["ev_b_mean"])),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.axvline(float(theta), color="#d62728", ls="--", lw=1.2,
               label=f"selected theta={theta:g}")
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.set_title(f"misalignment gate scan — EV_B (fee={CFG['fee']}, prem={CFG['premium']})\n"
                 f"confirm verdict: {verdict['verdict']}", fontsize=10)
    ax.set_xlabel("theta (misalign >= theta)")
    ax.set_ylabel("EV_B per trade")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    p = os.path.join(out, f"quote_pair_diag_{TF}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log(f"绘图完成：{p}")
    return p


def build_report(ctx: dict) -> str:
    """report.md。所有数字均从 ctx 结构读出，不写死任何结果。"""
    pairs, span, ret = ctx["pairs"], ctx["span"], ctx["ret"]
    bins, scan_cal, scan_conf = ctx["bins"], ctx["scan_cal"], ctx["scan_conf"]
    theta, rule, verdict = ctx["theta"], ctx["rule"], ctx["verdict"]
    checks, counts, funnel = ctx["checks"], ctx["counts"], ctx["funnel"]
    run_cfg, paths = ctx["run_cfg"], ctx["paths"]
    n_ok = sum(1 for c in checks if c["ok"])
    gate, base = verdict["gate"], verdict["no_gate"]
    fx = verdict["fixed05_on_gate_sample"]
    L: list = []
    A = L.append

    A("# 形状 kNN × 预测市场真实报价配对 EV 重估（5m）")
    A("")
    A(f"> ⚠️ **EXPLORATORY（事后分析）**：{EXPLORATORY_NOTE}")
    A("")
    A(f"- 生成：`{run_cfg['generated_at_utc']}`　耗时 {run_cfg['elapsed_sec']}s")
    A(f"- `cfg_sha256` = `{run_cfg['cfg_sha256']}`　"
      f"`sk_cfg_sha256` = `{run_cfg['sk_cfg_sha256'][:16]}…`")
    A(f"- 自检：**{n_ok}/{len(checks)} 通过**"
      + ("" if n_ok == len(checks) else "　⚠️ 有未通过项，下列结论不可信"))
    A(f"- 产出目录：`{os.path.relpath(paths['pairs_csv'], ROOT).replace(os.sep, '/')}` 所在目录")
    A("")

    # ---- 1 结论 ----
    A("## 1. 结论")
    A("")
    A(f"**终裁：`{verdict['verdict']}`**")
    A("")
    A(f"{verdict['reason']}")
    A("")
    if int(gate.get("n", 0)):
        A(f"confirm 段在 θ={theta:g} 门禁下 n={gate['n']}、"
          f"胜率 {_pct(gate['win_rate'])}、真实报价均值 {gate['q_mean']:.4f}、"
          f"打平线 {_pct(gate['be_b_at_qmean'])}、"
          f"EV_B 均值 {_sgn(gate['ev_b_mean'])}"
          f"（run 级 95% CI [{_sgn(gate['ev_b_ci_low'])}, {_sgn(gate['ev_b_ci_high'])}]）。")
    else:
        A("confirm 段在选定 θ 下 **零样本**——门禁把信号全部滤掉了，"
          "这本身就是结论：calib 段的高错配区间在 confirm 段不复现。")
    A("")

    # ---- 1.1 核心问题的直接回答（全部由 bins 数据算出，不写死结论）----
    nz1 = [r for r in bins if int(r.get("n", 0)) > 0]
    if nz1:
        def _bl(rs: list) -> str:
            return "、".join("`" + str(r["bin"]) + "`" for r in rs)

        pos_bins = [r for r in nz1 if float(r["ev_b_ci_low"]) > 0.0]
        neg_bins = [r for r in nz1 if float(r["ev_b_ci_high"]) < 0.0]
        above_q = [r for r in nz1 if float(r["win_rate"]) > float(r["q_mean"])]
        A("### 1.1 核心问题的直接回答")
        A("")
        A("**问：形状相似性是否携带「市场开盘报价尚未反映」的增量信息？**")
        A("")
        A(f"- calib 段 {len(nz1)} 个非空 consensus 箱中，EV_B 的 run 级 95% CI "
          f"**完全在 0 以上**的有 {len(pos_bins)} 个"
          + (f"（{_bl(pos_bins)}）" if pos_bins else "")
          + f"；**完全在 0 以下**的有 {len(neg_bins)} 个"
          + (f"（{_bl(neg_bins)}）" if neg_bins else "") + "。")
        A(f"- 胜率 > 同箱 q_mean 的箱：{len(above_q)}/{len(nz1)}"
          + (f"（{_bl(above_q)}）" if above_q else "")
          + "——未构成系统性同向；且差值量级（见 §4）远小于 ~2-3pp 的交易成本。")
        if pos_bins:
            A(f"- **答：存在候选正向箱**（{_bl(pos_bins)}），但均为 calib 段事后观察，"
              "须在新的冻结段上 pre-registered 重验才能计入结论。")
        elif neg_bins:
            A(f"- **答：没有正向增量信息。**唯一 CI 脱离 0 的箱是负向的"
              f"（{_bl(neg_bins)}）：该区间实际胜率显著低于掷硬币、也显著低于同箱"
              "市场报价，即 kNN 在其**最自信的区间给出的方向是系统性错的**。"
              "算术上「反向使用」诱人，但反向是看过数据后的事后选择，"
              "本轮不给它任何验证权重（见 §9）。")
        else:
            A("- **答：无任何箱的 EV_B CI 脱离 0**——本轮数据不足以区分"
              "「无增量信息」与「有但太小、当前样本测不出」。")
        A("")

    # ---- 2 口径对照 ----
    A("## 2. 本轮到底换了什么（vs 上一轮 shape_segment_knn_720d）")
    A("")
    A(_md_table(
        ["维度", "上一轮", "本轮"],
        [["EV 入场价", "固定 `entry_price=0.5`（成本 0.51）",
          f"真实报价：开窗后 ≤{CFG['decision_sec_main']:g}s 最晚采样（对照 "
          f"{CFG['decision_sec_const']:g}s）"],
         ["评估段", "冻结 holdout（约 144 天，n=2891）",
          f"报价覆盖全段 {span['days']} 天（{span['n_windows']} 窗）"],
         ["refs 池", "三段各自 expanding（cap=i1/i1/i2）",
          f"统一 cap（评估段之前全部，{ret['n_refs_pool']} 段）"],
         ["判据", "胜率 > `breakeven_win_rate(0.5)`=52.0408%",
          "`misalign = model_prob − q ≥ θ` 且 EV_B 的 run 级 CI 下界 > 0"],
         ["统计口径", "Wilson + block CI + BH-FDR",
          "Wilson + block CI + run 级 EV bootstrap CI + MDE（不涉多重检验，只一个 θ）"],
         ["纪律", "pre-registered holdout 只触碰一次",
          f"exploratory + 内部再切 calib({CFG['calib_frac']:g})/confirm"
          f"({1 - CFG['calib_frac']:g})，θ 只在 calib 选"],
         ["kNN 检索口径", "9 项冻结", "**逐项启动断言 == 上轮，0 改动**"]]))
    A("")
    A("kNN 检索本身（分段/表征/因果 topk/投票）**未动**，启动断言逐项比对 "
      f"{len(_KNN_KEYS)} 个键；因此本轮差异只可能来自「评估段 + EV 口径」。")
    A("")

    # ---- 3 数据体检 ----
    A("## 3. 数据体检")
    A("")
    A("### 3.1 报价覆盖")
    A("")
    A(_md_table(["项", "值"], [
        ["报价文件", f"`{os.path.basename(ctx['quotes_path'])}`"],
        ["5m 窗口数", span["n_windows"]],
        ["时间跨度", f"{span['days']} 天（{span['start_iso']} ~ {span['end_iso']}）"],
        ["采样总数", span["samples_total"]],
        ["每窗采样 p50 / min", f"{span['samples_per_window_p50']:g} / "
                               f"{span['samples_per_window_min']}"],
        ["首条 offset p50 / p90",
         f"{span['first_offset_sec_p50']:.1f}s / {span['first_offset_sec_p90']:.1f}s"],
        ["末条 offset p50 / max",
         f"{span['last_offset_sec_p50']:.1f}s / {span['last_offset_sec_max']:.1f}s"],
        ["btc_price 覆盖率", f"{span['btc_price_coverage']:.4f}（仅供自检④）"],
        ["窗口网格众数 / 非连续跳变",
         f"{span['grid_gap_mode_ms']}ms / {span['grid_gap_nonconsecutive']} 处"],
    ]))
    A("")
    A("### 3.2 配对漏斗")
    A("")
    A(_md_table(["阶段", "数量", "相对上一步"], funnel))
    A("")
    A("### 3.3 refs 池与因果")
    A("")
    A(_md_table(["项", "值"], [
        ["K 线根数", f"{run_cfg['klines']['n_bars']}"
                     f"（{run_cfg['klines']['first_iso']} ~ {run_cfg['klines']['last_iso']}）"],
        ["分段总数", run_cfg["segments"]["n_seg"]],
        ["表征有效段", f"{run_cfg['repr']['n_repr_valid']} / "
                       f"{run_cfg['repr']['X_shape'][0]}"],
        ["refs cap / ra_lim", f"{ret['cap']} / {ret['ra_lim']}"],
        ["refs 池段数", ret["n_refs_pool"]],
        ["refs 最早 / 最晚 anchor",
         f"{iso_utc(int(ret['ref_anchor_t_min']))} / {iso_utc(int(ret['ref_anchor_t_max']))}"],
        ["最早查询段首根", iso_utc(int(ret["qs_min_t"]))],
        ["causal_lag", CFG["causal_lag"]],
    ]))
    A("")

    # ---- 4 分箱诊断 ----
    A("## 4. 核心诊断：consensus 分箱（calib 段）")
    A("")
    A("这张表直接回答本轮的核心问题：**形状相似性是否携带市场开盘报价尚未反映的"
      "增量信息？**若携带，高 consensus 箱的实际胜率应系统性高于同箱 `q_mean`"
      "（市场报价隐含概率），而不只是高于 50%。")
    A("")
    A(_md_table(
        ["consensus 箱", "n", "胜率", "Wilson 95%", "q_mean（市场隐含）",
         "胜率−q_mean", "p(vs 0.5)", "EV_B 均值", "EV_B run-boot 95% CI", "错配均值"],
        [[r["bin"], r.get("n", 0), _pct(r.get("win_rate")),
          _ci(r, "wilson"),
          (f"{r['q_mean']:.4f}" if int(r.get("n", 0)) else ""),
          (_sgn(float(r["win_rate"]) - float(r["q_mean"])) if int(r.get("n", 0)) else ""),
          _pv(r), _sgn(r.get("ev_b_mean")), _ci(r, "ev_b"),
          _sgn(r.get("misalign_mean"))] for r in bins]))
    A("")
    nz = [r for r in bins if int(r.get("n", 0)) > 0]
    if nz:
        gaps = sorted(((float(r["win_rate"]) - float(r["q_mean"]), r) for r in nz),
                      key=lambda z: -z[0])
        g0, r0 = gaps[0]
        gl, rl = gaps[-1]
        A(f"- 胜率−报价 差值最大的箱：`{r0['bin']}`（n={r0['n']}）差 {g0 * 100:+.2f}pp；"
          f"最小：`{rl['bin']}`（n={rl['n']}）差 {gl * 100:+.2f}pp。")
        A(f"- 最高非空箱 `{nz[-1]['bin']}`：n={nz[-1]['n']}、胜率 "
          f"{_pct(nz[-1]['win_rate'])}、q_mean {nz[-1]['q_mean']:.4f}、"
          f"EV_B {_sgn(nz[-1]['ev_b_mean'])}。")
        sig_bad = [r for r in nz
                   if float(r["p_up_0.5"]) < 0.05 and float(r["win_rate"]) < 0.5]
        if sig_bad:
            txt = "；".join(
                f"`{r['bin']}` n={r['n']}、胜率 {_pct(r['win_rate'])}、"
                f"p(vs 0.5)={_pv(r)}、q_mean {r['q_mean']:.4f}" for r in sig_bad)
            A(f"- ⚠️ **胜率显著低于 0.5 的箱**（consensus 过度自信、方向反了）：{txt}。"
              f"这些箱里 kNN 形状相似性不但没有携带报价尚未反映的增量信息，"
              f"而是给出了**系统性错误**的方向判断（声称的 consensus 远高于实际胜率）。"
              f"理论上可反向使用，但「反向」属于看过数据后的事后发现，"
              f"**必须在新的冻结段上重验才能计**。")
        A("- 判读口径：差值>0 且 Wilson 下界 > 同箱打平线，才算「报价尚未反映」；"
          "仅差值>0 但 CI 跨打平线，只是描述性。")
    else:
        A("- calib 段无非空箱。")
    A("")

    # ---- 5 门禁扫描 ----
    A("## 5. 错配门禁扫描与 θ 选定")
    A("")
    A(f"选择规则（**预先写死**）：{GATE_SELECT_RULE}")
    A("")
    A("### 5.1 calib 段")
    A("")
    A(_scan_table(scan_cal))
    A("")
    A(f"→ **选定 θ = {theta:g}**（规则 `{rule}`）。")
    A("")
    A("### 5.2 confirm 段全扫描（仅透明留痕，**不参与选 θ**）")
    A("")
    A(_scan_table(scan_conf))
    A("")

    # ---- 6 终裁 ----
    A("## 6. confirm 终裁（只跑一次）")
    A("")
    cw = verdict["confirm_window"]
    A(f"confirm 段时间：{cw['first_iso']} ~ {cw['last_iso']}（calib n="
      f"{cw['n_calib_paired']} / confirm n={cw['n_confirm_paired']}）")
    A("")
    A(_md_table(
        ["口径", "n", "胜率", "Wilson 95%", "run 块", "block CI", "q_mean", "打平线",
         "EV_B 均值", "EV_B run-boot 95% CI", "p(vs 0.5)", "MDE"],
        [["**gate（终裁对象）**", gate.get("n", 0), _pct(gate.get("win_rate")),
          _ci(gate, "wilson"), (gate.get("n_runs", "") if int(gate.get("n", 0)) else ""),
          _ci(gate, "block"),
          (f"{gate['q_mean']:.4f}" if int(gate.get("n", 0)) else ""),
          _pct(gate.get("be_b_at_qmean")), _sgn(gate.get("ev_b_mean")),
          _ci(gate, "ev_b"), _pv(gate), (f"{gate['mde_pp']}pp" if int(gate.get("n", 0)) else "")],
         ["no_gate（θ=0 基准）", base.get("n", 0), _pct(base.get("win_rate")),
          _ci(base, "wilson"), (base.get("n_runs", "") if int(base.get("n", 0)) else ""),
          _ci(base, "block"),
          (f"{base['q_mean']:.4f}" if int(base.get("n", 0)) else ""),
          _pct(base.get("be_b_at_qmean")), _sgn(base.get("ev_b_mean")),
          _ci(base, "ev_b"), _pv(base), (f"{base['mde_pp']}pp" if int(base.get("n", 0)) else "")]]))
    A("")
    if int(gate.get("n", 0)):
        A(f"功效：{gate['power_note']['note']}（verdict `{gate['power_note']['verdict']}`）。"
          f"EV_B 口径 p 方法：{gate['p_method']}。")
        A("")

    # ---- 7 固定价对照 ----
    A("## 7. 真实报价 vs 固定 0.5：**同一批样本**的口径差")
    A("")
    A("同 confirm∧gate 样本、同 win 序列，只把入场价从固定 0.5 换成真实报价——"
      "这才是「上一轮口径错在哪、错多少」的无混淆答案。")
    A("")
    if int(gate.get("n", 0)):
        d = float(gate["ev_b_mean"]) - float(fx["ev_fixed05_mean"])
        A(_md_table(["口径", "n", "EV 均值", "EV 总和", "说明"], [
            ["固定 entry=0.5（上一轮）", fx["n"], _sgn(fx["ev_fixed05_mean"]),
             _sgn(fx["ev_fixed05_sum"], 2), "房内 `stats.ev(p, 0.5)`"],
            ["真实报价 EV_B（本轮主口径）", gate["n"], _sgn(gate["ev_b_mean"]),
             _sgn(gate["ev_b_sum"], 2), f"0.98/(q+{CFG['premium']})−1"],
            ["真实报价 EV_A（无溢价对照）", gate["n"], _sgn(gate["ev_a_mean"]),
             "", "0.98/q−1（`local_s5_real_quote_ev` 口径）"],
            ["**差值（B − 固定）**", "", _sgn(d), "",
             "正=真实报价比固定假设更乐观，负=更悲观"],
        ]))
        A("")
        A(f"→ 口径切换对 EV 均值的净影响：**{d:+.4f}**。"
          f"真实成交价均值 {gate['q_mean']:.4f}（中位 {gate['q_p50']:.4f}）"
          f"vs 固定假设 0.5100。")
    else:
        A("gate 样本为空，无对照可算。")
    A("")

    # ---- 8 自检 ----
    A("## 8. 自检结果")
    A("")
    A(_md_table(["项", "结果", "证据"],
                [[c["name"], "✅ PASS" if c["ok"] else "❌ FAIL", c["detail"]]
                 for c in checks]))
    A("")

    # ---- 9 局限 ----
    A("## 9. 诚实局限")
    A("")
    for i, s in enumerate(ctx["limits"], 1):
        A(f"{i}. {s}")
    A("")

    # ---- 10 产物 ----
    A("## 10. 产物清单")
    A("")
    A(_md_table(["文件", "说明"], [
        [f"`{os.path.basename(v)}`", _PATH_DESC.get(k, k)]
        for k, v in sorted(paths.items())]
        + [["`report.md`", _PATH_DESC["report_md"]]]))
    A("")
    A("复现：`.venv\\Scripts\\python.exe -X utf8 "
      "scripts/local_shape_knn_real_quote_ev.py`。四个 CSV 不含墙钟时间，"
      "两轮重跑字节全等（run_config.json / report.md 含生成时间戳，不参与哈希比对）。")
    A("")
    return "\n".join(L)


def _ci(st: dict, kind: str) -> str:
    if not int(st.get("n", 0)):
        return ""
    if kind == "wilson":
        return f"[{_pct(st['wilson_low'])}, {_pct(st['wilson_high'])}]"
    if kind == "block":
        return f"[{_pct(st['block_ci_low'])}, {_pct(st['block_ci_high'])}]"
    return f"[{_sgn(st['ev_b_ci_low'])}, {_sgn(st['ev_b_ci_high'])}]"


def _pv(st: dict) -> str:
    if not int(st.get("n", 0)):
        return ""
    p = st["p_up_0.5"]
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def _scan_table(scan: list) -> str:
    return _md_table(
        ["θ", "n", "占比", "UP 方向占比", "胜率", "Wilson 95%", "q_mean", "打平线",
         "EV_B 均值", "run 块", "EV_B run-boot 95% CI", "p(vs 0.5)", "MDE"],
        [[f"{r['theta']:g}", r["n"],
          (_pct(r.get("n_frac"), 1) if r["n"] else ""),
          (_pct(r.get("dir_up_frac"), 1) if r["n"] else ""),
          _pct(r.get("win_rate")), _ci(r, "wilson"),
          (f"{r['q_mean']:.4f}" if r["n"] else ""), _pct(r.get("be_b_at_qmean")),
          _sgn(r.get("ev_b_mean")), (r.get("n_runs", "") if r["n"] else ""),
          _ci(r, "ev_b"), _pv(r),
          (f"{r['mde_pp']}pp" if r["n"] else "")] for r in scan])


def build_limits(span: dict, ret: dict, counts: dict, verdict: dict, theta: float) -> list:
    """静态诚实局限清单（不依赖结果好坏，跑前就能写全）。"""
    return [
        f"**exploratory 而非 pre-registered**：{EXPLORATORY_NOTE}",
        f"**θ 与 gate_grid 是本轮新引入的自由度**：grid "
        f"{list(CFG['gate_grid'])} 是在看过开盘报价分布（up_price IQR 0.50~0.54）之后定的；"
        f"虽在 calib 段选定、confirm 只终裁一次，但「grid 本身的选取」仍是事后判断。"
        f"本轮选定 θ={theta:g}。",
        f"**decision_sec_main={CFG['decision_sec_main']:g}s 也是本轮新选的自由度**（已配 "
        f"{CFG['decision_sec_const']:g}s 宪法对照口径，但未对两者做多重检验校正）。",
        "**报价是快照，不含可成交量**：实测开盘 up+down 中位数=1.0000（零价差），"
        "但快照不告知该价位的深度；EV_B 已含 FEE+PREMIUM，**未含吃单推动价格的影响**。"
        "房内 `local_ev_audit_transparent.py` 已记录过同类偏差（S5 回测 +0.47 vs 真实 +0.06）。",
        f"**时间跨度只有 {span['days']} 天 / {span['n_windows']} 窗**，未跨越不同波动率 regime；"
        f"且窗网格有 {span['grid_gap_nonconsecutive']} 处非连续跳变（缺失窗）。",
        "**EV 是 per-trade 口径，不是组合收益**：未建模每通道每窗口至多一单、"
        "资金占用、并发与撤单；也未做仓位（Kelly）与回撤约束。",
        "**未做 15m**：上一轮 15m 全 15 臂 REJECT_PREHOLDOUT（discovery↔validation "
        "方向不一致）、holdout 零触碰，没有稳定方向可供报价配对。",
        "**报价文件是单次导出快照**，无法验证导出时是否完整；"
        "同批的 `sentiment_windows_online_20260819.json` 已确认字节损坏，说明该批导出本身不可全信。",
        f"**弃权/缺失漏斗**：报价窗 {span['n_windows']} → 成对 {counts['n_paired']}"
        f"（弃权 {counts['n_abstain']} / 无报价 {counts['n_noquote']} / "
        f"报价越界 {counts['n_qbad']}）；丢弃非随机时会引入选择偏差。",
        "**未验证可下单性**：本轮全程离线，未调用任何交易接口，不构成上线建议。",
    ]


def _rel(a, b) -> str:
    a, b = int(a), int(b)
    return "—" if b <= 0 else f"{a / b * 100:.1f}%"


def main(argv=None) -> int:
    """入口：报价加载 → 因果检索 → 配对 → calib 选 θ → confirm 终裁 → 自检 → 落盘。"""
    ap = argparse.ArgumentParser(
        description="形状 kNN × 预测市场真实报价配对 EV 重估（5m，exploratory）")
    ap.add_argument("--out", default=OUT, help="产物目录（gitignored output/ 下）")
    ap.add_argument("--quotes", default=QUOTE_JSON, help="报价快照 JSON 路径")
    ap.add_argument("--plot", action="store_true", help="额外输出诊断 PNG")
    ap.add_argument("--self-check-only", action="store_true",
                    help="只跑管线+自检+CSV/JSON，不出 report.md / PNG")
    args = ap.parse_args(argv)

    t0 = time.time()
    log("=" * 78)
    log("local_shape_knn_real_quote_ev：形状 kNN × 真实报价配对 EV 重估")
    log("=" * 78)
    log(f"CFG（{len(CFG)} 项）cfg_sha256={cfg_sha256()}")
    log(f"kNN {len(_KNN_KEYS)} 项启动断言已通过（逐项 == shape_segment_knn_720d.CFG）")
    log(f"EXPLORATORY：{EXPLORATORY_NOTE}")

    if not os.path.exists(args.quotes):
        log(f"✗ 报价文件不存在：{args.quotes}")
        return 2

    # ---- 报价 ----
    quotes = load_quotes(args.quotes, TF, BAR_MS)
    span = quote_span_stats(quotes, BAR_MS)
    log(f"报价覆盖：{span['n_windows']} 窗 / {span['days']} 天 "
        f"[{span['start_iso']} ~ {span['end_iso']}] / 采样 {span['samples_total']} 条 "
        f"/ btc 覆盖 {span['btc_price_coverage']}")

    # ---- K 线 + 分段 + 表征（全部复用 sk）----
    kl, atr, labels = sk._load_tf(TF)
    log(f"K 线：{len(kl)} 根 [{iso_utc(int(kl.t[0]))} ~ {iso_utc(int(kl.t[-1]))}]")
    segs = sk.get_segments(TF, kl, atr, "fixed", sk._seg_params(), SK_CACHE)
    log(f"分段（fixed L={CFG['seg_len']} step={CFG['seg_step']}）："
        f"{len(segs['start'])} 段（缓存目录 {os.path.basename(SK_CACHE)}）")
    X, repr_valid = sk.build_shape_matrix(kl, segs, atr, int(CFG["resample_m"]),
                                          CFG["norm_mode"])
    log(f"表征矩阵：X={X.shape} repr_valid={int(repr_valid.sum())}/{len(repr_valid)}")

    # ---- 因果检索 ----
    log("因果检索（评估段查询 × 评估段之前 refs）…")
    ret = build_retrieval(kl, segs, X, repr_valid, labels, quotes, CFG)
    log(f"refs 池 {ret['n_refs_pool']} 段（cap={ret['cap']}）；"
        f"报价窗 {span['n_windows']} → bar 匹配 {ret['n_bar_matched']} → "
        f"分段匹配 {ret['n_window_matched']} → 有效 {ret['n_after_valid']}")
    log(f"refs anchor 最晚 {iso_utc(int(ret['ref_anchor_t_max']))} < "
        f"最早查询段首根 {iso_utc(int(ret['qs_min_t']))}（causal_lag={CFG['causal_lag']}）")
    n_pred = int((~ret["abstain"]).sum())
    log(f"投票：未弃权 {n_pred} / 弃权 {int(ret['abstain'].sum())}"
        f"（UP {int((ret['predicted'] == 1).sum())} / DOWN "
        f"{int((ret['predicted'] == -1).sum())}）")

    # ---- 配对 ----
    pairs = build_pairs(ret, quotes, kl, CFG)
    counts = pairs["_counts"]
    if counts["n_paired"] == 0:
        log("✗ 成对样本为 0，无法继续")
        return 2

    # ---- calib 选 θ → confirm 终裁一次 ----
    bins = diagnose_bins(pairs, CFG)
    scan_cal = gate_scan(pairs, CFG, "calib")
    theta, rule, _best = select_theta(scan_cal, CFG)
    log(f"calib 扫描 {len(scan_cal)} 个 θ → 选定 θ={theta:g}（规则 {rule}）")
    verdict = confirm_verdict(pairs, CFG, theta, rule)
    log(f"confirm 终裁：{verdict['verdict']} —— {verdict['reason']}")
    scan_conf = gate_scan(pairs, CFG, "confirm")   # 终裁后才算，仅透明留痕

    # ---- 自检 ----
    checks = _self_check(kl, ret, pairs, quotes, CFG, theta, rule, verdict)
    ok_all = self_check_pass(checks)
    for c in checks:
        log(f"  {'PASS' if c['ok'] else 'FAIL'} {c['name']}")
    log(f"自检：{sum(1 for c in checks if c['ok'])}/{len(checks)} 通过"
        + ("" if ok_all else "　⚠️ 有未通过项"))

    # ---- run_config ----
    n_noabst = counts["n_query"] - counts["n_abstain"]
    funnel = [
        ["报价文件 5m 窗口", span["n_windows"], "—"],
        ["匹配到 K 线整点 bar", ret["n_bar_matched"],
         _rel(ret["n_bar_matched"], span["n_windows"])],
        ["匹配到分段 anchor=i−1", ret["n_window_matched"],
         _rel(ret["n_window_matched"], ret["n_bar_matched"])],
        ["表征有效 ∧ 标签有效（成查询）", ret["n_after_valid"],
         _rel(ret["n_after_valid"], ret["n_window_matched"])],
        ["kNN 未弃权", n_noabst, _rel(n_noabst, counts["n_query"])],
        [f"有 ≤{CFG['decision_sec_main']:g}s 报价", n_noabst - counts["n_noquote"],
         _rel(n_noabst - counts["n_noquote"], n_noabst)],
        [f"报价∈[{CFG['q_floor']},{CFG['q_ceil']}]（最终成对）", counts["n_paired"],
         _rel(counts["n_paired"], n_noabst - counts["n_noquote"])],
    ]
    elapsed = round(time.time() - t0, 1)
    run_cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(os.path.abspath(__file__)),
        "tf": TF, "bar_ms": BAR_MS,
        "cfg": CFG, "cfg_sha256": cfg_sha256(),
        "sk_cfg_sha256": hashlib.sha256(
            json.dumps(sk.CFG, sort_keys=True, default=list).encode("utf-8")).hexdigest(),
        "knn_keys_asserted": list(_KNN_KEYS),
        "gate_select_rule": GATE_SELECT_RULE,
        "exploratory_note": EXPLORATORY_NOTE,
        "inputs": {
            "quotes_json": args.quotes,
            "quotes_bytes": os.path.getsize(args.quotes),
            "klines_csv": sk.data_csv_path(TF),
            "klines_bytes": os.path.getsize(sk.data_csv_path(TF)),
        },
        "quote_span": span,
        "klines": {"n_bars": int(len(kl)), "first_iso": iso_utc(int(kl.t[0])),
                   "last_iso": iso_utc(int(kl.t[-1]))},
        "segments": {"n_seg": int(len(segs["start"])), "segmode": "fixed",
                     "cache_dir": SK_CACHE, "params": sk._seg_params()},
        "repr": {"X_shape": [int(X.shape[0]), int(X.shape[1])],
                 "n_repr_valid": int(repr_valid.sum())},
        "retrieval": {k: ret[k] for k in (
            "cap", "ra_lim", "n_refs_pool", "ra_min", "ra_max", "ref_anchor_t_min",
            "ref_anchor_t_max", "qs_min", "qs_min_t", "n_bar_matched",
            "n_window_matched", "n_after_valid", "n_dropped_valid")},
        "vote": {"n_predicted": n_pred, "n_abstain": int(ret["abstain"].sum()),
                 "n_up_pred": int((ret["predicted"] == 1).sum()),
                 "n_down_pred": int((ret["predicted"] == -1).sum())},
        "funnel": funnel,
        "counts": counts,
        "theta": float(theta), "theta_rule": rule,
        "verdict": verdict["verdict"], "verdict_reason": verdict["reason"],
        "self_check": {"all_pass": ok_all, "n_checks": len(checks),
                       "failed": [c["name"] for c in checks if not c["ok"]]},
        "elapsed_sec": elapsed,
        "outputs": [f"quote_pairs_{TF}.csv", f"consensus_bins_calib_{TF}.csv",
                    f"gate_scan_calib_{TF}.csv", f"gate_scan_confirm_{TF}.csv",
                    f"confirm_verdict_{TF}.json", f"self_check_{TF}.json",
                    "run_config.json"]
                   + ([] if args.self_check_only else
                      ["report.md"] + ([f"quote_pair_diag_{TF}.png"] if args.plot else [])),
    }

    paths = write_outputs(args.out, pairs, bins, scan_cal, scan_conf,
                          verdict, checks, run_cfg)

    if args.plot and not args.self_check_only:
        pp = maybe_plot(args.out, bins, scan_cal, scan_conf, theta, verdict)
        if pp:
            paths["plot_png"] = pp

    if not args.self_check_only:
        ctx = {"pairs": pairs, "span": span, "ret": ret, "bins": bins,
               "scan_cal": scan_cal, "scan_conf": scan_conf, "theta": theta,
               "rule": rule, "verdict": verdict, "checks": checks, "counts": counts,
               "funnel": funnel, "run_cfg": run_cfg, "paths": paths,
               "quotes_path": args.quotes,
               "limits": build_limits(span, ret, counts, verdict, theta)}
        rp = os.path.join(args.out, "report.md")
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            f.write(build_report(ctx))
        paths["report_md"] = rp
        log(f"report.md 已写入：{rp}")

    log("-" * 78)
    log(f"完成：θ={theta:g} | 终裁 {verdict['verdict']} | 自检 "
        f"{sum(1 for c in checks if c['ok'])}/{len(checks)} | 耗时 {elapsed}s")
    log("-" * 78)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
