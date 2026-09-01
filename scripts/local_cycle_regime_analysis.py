#!/usr/bin/env python3
"""720 天信号周期/行情分段深度分析（定稿；草稿见 .pytest_tmp/cycle_regime_analysis_draft.py）。

目标（对齐已批准计划 720天信号周期分段分析）：
  对全部线上信号族做周期分段表现分析，补齐 5 个缺口——
  G1 regime 维度×全信号 / G2 KREV-A/B 720d 重放 / G3 逐分段 IS/OOS 双向同向采信 /
  G4 跨维度统一多重检验预算 / G5 短历史报价族与 720d 可重建族物理分区。

分段维度（决策已冻结）：
  ER_7d  趋势周期 4 段（效率比，672 根 15m 回看；阈值只用 IS 段分位数 + 冻结断言）
  RV_7d  波动率周期 3 档（7d 已实现波动率三分位）
  ret24  短期动量 3 段（过去 24h 收益 ±1%，口径照抄 regime_ev_compare.py L33-55）
  session_utc 4 段（亚/欧/美/深夜；唯一保留的日历维，其余 10 维引用 timeslice 报告）
  月份漂移 24 个月逐月 + 前后半斜率（explore，只报告不采信）

信号族：
  720d 可重建（纯 K 线）：S1/S2/S4/S5（build_events 重放）+ KREV-A/B（冻结条件重放）
  短历史报价族（~38 天情绪窗）：x4_v1/v2（自写重放）+ quote_edge 8 版（复用
    local_timeslice_winrate.replay_shadow_signals）——只切 session+ret24，
    min_detectable_effect 前置裁剪，与 720d 族物理分区禁止同表混排。

口径冻结（import 不手抄）：
  场景事件   ← local_timeslice_winrate.replay_scene_events（build_events + pos4h 补齐）
  KREV 条件  ← kline_shadow_detector.SHADOW_CONDITIONS（condition_mask 重放）
  quote 门禁 ← local_timeslice_winrate.replay_shadow_signals（QUOTE_EDGE_RULES 链）
  x4 常量    ← misalignment_detector（X4_END_MAX/DECISION_T_SEC/X4_V2_*）
  统计内核   ← backtest.stats（wilson/exact_binomial_p/MDE/multiple_testing_threshold/ev）
  regime 口径 ← local_s5_regime_analysis.py L149-182（ER/RV 序列 + IS 段分位数）

对账闸（不过不出报告）：
  ① KREV 时间切片：冻结发现窗口末（2026-08-27T21:00 UTC）前命中数与注册表
     总数（A=650/B=642，tests/test_kline_shadow_detector.py L37-40）之差 ∈[0,8]
     ——720d 窗口滚动 3 天导致头部 3 天数据出窗（期望缺口≈2.7，Poisson 95% 上界 8），
     超带即口径漂移硬中止。
  ② 场景族全量胜率对齐 timeslice_winrate_report_20260829_1152（偏差 >1pp 中止；
     n 因窗口滚动允许漂移只打印）。
  ③ regime 全期占比对齐 output/s5_regime_analysis.log（偏差 >3pp 中止）。

采信闸（D 区）：同向 ∧ OOS n≥30 ∧ OOS Wilson 越过格基线 ∧ |池化偏离|≥多重检验门槛
（分母 = 全表 prior 格数 14+7=21 统一记账）。n<30 一律 INSUFFICIENT_POWER 不解读。

用法：.venv\\Scripts\\python.exe -X utf8 scripts/local_cycle_regime_analysis.py
输出：output/cycle_regime_report_<stamp>.md / .json + output/cycle_regime_analysis.log
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time as _time
from dataclasses import asdict, dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from binance_predict.backtest.stats import (  # noqa: E402
    ev as ev_pq,
    min_detectable_effect,
    multiple_testing_threshold,
    wilson,
)
import local_timeslice_winrate as tsw  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output")
KL5_JSON = os.path.join(ROOT, "output", "klines_5m_cache_720d.json")
CSV_15M = os.path.join(ROOT, "output", "klines_15m_720d.csv")
CSV_5M = os.path.join(ROOT, "output", "klines_5m_720d.csv")
SENTIMENT_JSON = os.path.join(ROOT, "output", "sentiment_windows_online_fixed.json")

# ---- 冻结常量（出处注明，勿改动）----
W = 672                                    # 7d × 96 根 15m（local_s5_regime_analysis.py L42）
ER_FROZEN = (0.0316, 0.0554)               # s5_regime_analysis.log L1
RV_FROZEN = (0.001954, 0.002504)           # 同上
ER_TOL, RV_TOL = 0.002, 1e-4               # 窗口滚动 3 天的分位漂移容忍带
REGIME_SHARE_FROZEN = {"趋势牛": 0.145, "趋势熊": 0.122, "过渡": 0.261, "震荡": 0.462}
SHARE_TOL = 0.03                           # s5_regime_analysis.log L5
DISCOVERY_END_MS = 1788123600000           # 2026-08-27T21:00 UTC（发现窗口末，KREV 切片闸）
KREV_TOTAL = {"krev_a_v1": 650, "krev_b_v1": 642}   # REGISTRY_COUNTS 三段求和
KREV_GAP_MAX = 8                           # 头部出窗缺口容忍（期望 2.7，Poisson 95% 上界）
# 场景族对账基准：(n, wr)，timeslice_winrate_report_20260829_1152.md L77-83 replay 行
SCENE_BENCH = {"S1": (2295, 0.577), "S2": (2184, 0.553),
               "S4": (1774, 0.577), "S5": (1312, 0.766)}
WR_TOL_PP = 1.0
TREND_TH = 0.01                            # ret24 ±1%（regime_ev_compare.py L30）
DIMS_720 = ("er_band", "rv_band", "ret24", "session")   # 4+3+3+4 = 14 prior 格
DIMS_QUOTE = ("session", "ret24")                        # 4+3 = 7 prior 格（降维，计划 Step3）
MIN_N = 30
BONF_BASE_PP = 2.0
N_PRIOR_CELLS = 14 + 7                     # 全表 prior 格数统一记账（计划 D 区）
MONTHS_HALF = 12
LOG = os.path.join(OUT_DIR, "cycle_regime_analysis.log")


class Tee:
    """双产物：stdout + 日志文件（勿用重定向落盘，GBK 乱码先例）。"""

    def __init__(self):
        self.f = open(LOG, "w", encoding="utf-8")
        try:
            sys.__stdout__.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def write(self, s):
        try:
            sys.__stdout__.write(s)
        except Exception:
            pass
        self.f.write(s)

    def flush(self):
        try:
            sys.__stdout__.flush()
        except Exception:
            pass
        self.f.flush()


def _iso(ms) -> str:
    return dt.datetime.fromtimestamp(int(ms) / 1000, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M")


def _gm(ts_ms):
    return _time.gmtime(int(ts_ms) / 1000.0)


def month_key(ts_ms) -> str:
    g = _gm(ts_ms)
    return f"{g.tm_year}-{g.tm_mon:02d}"


def _ym(mk: str) -> tuple[int, int]:
    y, m = mk.split("-")
    return int(y), int(m)


@dataclass(frozen=True)
class Rec:
    """统一轻量事件记录（比 tsw.Record 多 disp/settle_open 供位移口径）。"""
    key: str
    ts_ms: int
    direction: str        # 押注方向：DOWN / UP
    win: bool
    q: float | None       # 入场报价（K 线族无报价 → None）
    disp: float | None    # 结算段方向化位移（K 线族 @基线口径辅助）


# ======================================================================
# A 区：事件表（三族构建函数 → 统一 Rec）
# ======================================================================

def load_c5_json() -> list[tuple]:
    """720d 5m K 线 JSON 快路径（照抄 local_s5_real_quote_ev.py L72-79，
    另保留 volume 第 6 元：aggregate_15m/build_events 需要 6 元组）。"""
    with open(KL5_JSON, encoding="utf-8") as f:
        kl = json.load(f)
    now_ms = int(_time.time() * 1000)
    c5 = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
          for k in kl]
    if c5 and c5[-1][0] + 300_000 > now_ms:
        c5.pop()
    return c5


def replay_scene_recs(c5: list[tuple]) -> tuple[list[Rec], dict]:
    """场景族 S1/S2/S4/S5：import 冻结重放（tsw.replay_scene_events）。

    disp = 结算 15m 根 (close/open−1)×方向符号（位移均值口径的原料）。
    """
    raw, meta = tsw.replay_scene_events(c5)
    o15 = {r[0]: r[1] for r in c5}
    c15 = {r[0]: r[4] for r in c5}
    recs = []
    for r in raw:
        o, c = o15.get(r.ts_ms + 900_000), c15.get(r.ts_ms + 900_000)
        disp = None
        if o and c and o > 0:
            disp = (c / o - 1.0) * (1.0 if r.direction == "UP" else -1.0)
        recs.append(Rec(r.signal_key, r.ts_ms, r.direction, r.win, None, disp))
    return recs, meta


def replay_krev_recs() -> tuple[list[Rec], dict]:
    """KREV-A/B 720d 重放（计划 G2）。

    条件原文/特征矩阵/目标全 import（kline_shadow_detector.SHADOW_CONDITIONS +
    discovery 管线，与 tests/test_kline_shadow_detector.py 同路径）：
      mask = condition_mask(fm, parse_condition(cond)) & reversal_1.valid
      win = reversal_1.win（path3_all_down → 信号根阴线 → 押次根收阳，方向 UP）
      disp = reversal_1.ret（期望方向化次根收益，与 targets.py L84-85 同口径）
    返回 meta 含逐版本全窗计数与冻结窗口末前切片计数（对账闸原料）。
    """
    from binance_predict.discovery.data import load_klines_csv
    from binance_predict.discovery.features import atr_series, build_feature_matrix
    from binance_predict.discovery.hypotheses import condition_mask, parse_condition
    from binance_predict.discovery.targets import build_targets
    from binance_predict.services.kline_shadow_detector import SHADOW_CONDITIONS

    kl15 = load_klines_csv(CSV_15M, 900_000)
    kl5 = load_klines_csv(CSV_5M, 300_000)
    fm = build_feature_matrix(kl15, 900_000, k5=kl5)
    tg = build_targets(kl15.t, kl15.o, kl15.h, kl15.l, kl15.c, kl15.cont,
                       [1], atr_series(kl15, 20))
    ts = tg.items["reversal_1"]

    recs: list[Rec] = []
    meta: dict = {"total": {}, "slice": {}}
    for spec in SHADOW_CONDITIONS:
        v = spec["version"]
        mask = condition_mask(fm, parse_condition(spec["condition"])) & ts.valid
        idx = np.flatnonzero(mask)
        bars = np.asarray(kl15.t, dtype=np.int64)
        meta["total"][v] = int(len(idx))
        meta["slice"][v] = int((bars[idx] <= DISCOVERY_END_MS).sum())
        for i in idx:
            recs.append(Rec(v, int(bars[i]), "UP", bool(ts.win[i]), None,
                            float(ts.ret[i])))
    return recs, meta


def _rec_from_shadow(raw: list) -> list[Rec]:
    """quote_edge 8 版（tsw.replay_shadow_signals 产物）→ Rec（disp 无 → None）。"""
    return [Rec(r.signal_key, r.ts_ms, r.direction, r.win, r.q, None) for r in raw]


def replay_x4_recs(wins: dict[int, dict], ordered: list[dict]) -> tuple[list[Rec], dict]:
    """x4_v1/v2 本地重放（timeslice 脚本无 x4 重放，此处自写；口径全 import）。

    触发（misalignment_detector 冻结）：本窗 outcome==UP ∧ curve_end_pct(curve_up_pct)
    ≤ X4_END_MAX → 押次窗 DOWN；入场 = 次窗开窗后 ≤DECISION_T_SEC 最晚 DOWN token 价
    （_price_at 同口径），缺失回退 curve_up_pct/100 proxy（_entry_quote 同口径）。
    v2 追加 |过去 1h 涨跌幅| < X4_V2_PAST1H_MAX_ABS_PCT（基准 = start−1h±10min
    entry_price，_x4_base_local 同口径）；门禁数据缺失 → v2 不触发。
    ts 锚 = 次窗起点（对齐线上 x4 window_start=target_window_start）。
    """
    from binance_predict.services.misalignment_detector import (
        DECISION_T_SEC, X4_END_MAX, X4_V2_PAST1H_MAX_ABS_PCT,
    )

    recs: list[Rec] = []
    n_no_entry = 0
    for w in ordered:
        ws = int(w["start_time"])
        if w.get("outcome") != "UP":
            continue
        end = tsw.curve_end_pct(w.get("curve_up_pct"))
        if end is None or end > X4_END_MAX:
            continue
        nw = wins.get(ws + 300_000)
        if not nw or nw.get("outcome") not in ("UP", "DOWN"):
            continue
        nxt = ws + 300_000
        win = nw["outcome"] == "DOWN"
        q = tsw.price_within(nw.get("curve_down_price"), nxt, DECISION_T_SEC)
        if q is None:
            pct = tsw.price_within(nw.get("curve_up_pct"), nxt, DECISION_T_SEC)
            q = pct / 100.0 if pct is not None else None
        if q is None or not (0.01 <= q <= 0.99):
            n_no_entry += 1
            continue
        recs.append(Rec("x4_v1", nxt, "DOWN", win, q, None))
        base = tsw._x4_base_local(wins, ws)
        cur = tsw.window_open_btc(w)
        if base is not None and cur is not None and base > 0:
            chg = (cur - base) / base * 100.0
            if abs(chg) < X4_V2_PAST1H_MAX_ABS_PCT:
                recs.append(Rec("x4_v2", nxt, "DOWN", win, q, None))
    meta = {"n_no_entry": n_no_entry,
            "n_v1": sum(1 for r in recs if r.key == "x4_v1"),
            "n_v2": sum(1 for r in recs if r.key == "x4_v2")}
    return recs, meta


# ======================================================================
# B 区：标签器（纯函数，全吃 trailing 数据；阈值只用 IS 段）
# ======================================================================

def build_labels(c5: list[tuple]) -> tuple[dict, dict]:
    """15m 根级周期标签：ER_7d 4 段 / RV_7d 3 档 / ret24 3 段 / session / month。

    ER/RV 序列口径照抄 local_s5_regime_analysis.py L149-160（无未来函数，截至根
    j 收盘）。阈值分位只用 IS 段（数据窗前 360d，确定性可复现），并与冻结值
    断言比对（容差带内容忍 3 天窗口滚动漂移，超带 SystemExit）。
    ret24 口径照抄 regime_ev_compare.py L43-55（开盘前最后一根 5m 收盘 / 24h 前
    同位收盘 − 1）。暖机期（j<W，前 7 天）标签置 None，事件侧丢弃并计数。
    """
    t5 = np.array([r[0] for r in c5], dtype=np.int64)
    cl5 = np.array([r[4] for r in c5], dtype=np.float64)
    buckets: dict[int, list[int]] = {}
    for i, cyc in enumerate(t5 // 900_000):
        buckets.setdefault(int(cyc), []).append(i)
    cyc_arr: list[int] = []
    o_list: list[float] = []
    c_list: list[float] = []
    for cyc in sorted(buckets):
        idxs = buckets[cyc]
        if len(idxs) != 3 or (cyc + 1) * 900_000 > t5[-1] + 300_000:
            continue
        idxs.sort()
        cyc_arr.append(cyc)
        o_list.append(c5[idxs[0]][1])
        c_list.append(c5[idxs[-1]][4])
    cycs = np.array(cyc_arr, dtype=np.int64)
    o15 = np.array(o_list)
    c15 = np.array(c_list)
    N = len(cycs)

    # ER / RV（截至根 j 收盘的 672 根回看）
    ad = np.abs(np.diff(c15))
    lr = np.diff(np.log(np.maximum(c15, 1e-12)))
    cs = np.concatenate([[0.0], np.cumsum(ad)])
    path_sum = np.full(N, np.nan)
    path_sum[W:] = cs[W:] - cs[:-W]
    disp = np.full(N, np.nan)
    disp[W:] = c15[W:] - c15[:-W]
    er = np.abs(disp) / np.where(path_sum > 0, path_sum, np.nan)
    rv = np.full(N, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    rv[W:] = sliding_window_view(lr, W).std(axis=1)

    # IS/OOS 分界 = 数据窗起点 + 360d（确定性，不随运行时刻漂移）
    is_end = int(t5[0]) + 360 * 86_400_000
    close_ts = (cycs + 1) * 900_000
    m_a = (close_ts < is_end) & ~np.isnan(er) & ~np.isnan(rv)
    er50 = float(np.quantile(er[m_a], 0.50))
    er75 = float(np.quantile(er[m_a], 0.75))
    rv33 = float(np.quantile(rv[m_a], 1.0 / 3.0))
    rv67 = float(np.quantile(rv[m_a], 2.0 / 3.0))
    for got, frz, tol, name in (
            (er50, ER_FROZEN[0], ER_TOL, "ER q50"), (er75, ER_FROZEN[1], ER_TOL, "ER q75"),
            (rv33, RV_FROZEN[0], RV_TOL, "RV q33"), (rv67, RV_FROZEN[1], RV_TOL, "RV q67")):
        if abs(got - frz) > tol:
            raise SystemExit(
                f"[对账闸] 冻结阈值漂移：{name}={got:.6f} vs 冻结 {frz}（容差 {tol}）")
        print(f"  [阈值] {name}={got:.6f}（冻结 {frz}，|Δ|={abs(got - frz):.6f} ≤ {tol} OK）")

    # ret24（cyc→close 哈希口径，等价于 regime_ev_compare.trailing_24h）
    close_by_ts = {int(t): float(v) for t, v in zip(t5, cl5)}
    ret24 = np.full(N, np.nan)
    for j, cyc in enumerate(cyc_arr):
        open_ms = cyc * 900_000
        base = close_by_ts.get(open_ms - 86_400_000)
        op = close_by_ts.get(open_ms - 300_000)
        if base and op:
            ret24[j] = op / base - 1.0

    er_band = np.full(N, "", dtype=object)
    rv_band = np.full(N, "", dtype=object)
    r24_band = np.full(N, "", dtype=object)
    session = np.full(N, "", dtype=object)
    monthk = np.full(N, "", dtype=object)
    n_warm = 0
    for j in range(N):
        ts_open = int(cycs[j]) * 900_000
        session[j] = tsw.session_utc(ts_open)
        monthk[j] = month_key(ts_open)
        if j < W or np.isnan(er[j]) or np.isnan(rv[j]) or np.isnan(ret24[j]):
            n_warm += 1
            continue
        er_band[j] = ("趋势牛" if disp[j] > 0 else "趋势熊") if er[j] >= er75 else (
            "震荡" if er[j] < er50 else "过渡")
        rv_band[j] = "低" if rv[j] < rv33 else ("高" if rv[j] >= rv67 else "中")
        r24_band[j] = "上行" if ret24[j] > TREND_TH else ("下行" if ret24[j] < -TREND_TH else "震荡")

    cyc_f = cycs.astype(np.float64)
    labels = {"er_band": er_band, "rv_band": rv_band, "ret24": r24_band,
              "session": session, "month": monthk}
    meta = {"is_end": is_end, "n_cycles": N, "n_warm": n_warm,
            "quantiles": {"er50": er50, "er75": er75, "rv33": rv33, "rv67": rv67},
            "ts0": int(cycs[0]) * 900_000, "ts1": int(cycs[-1]) * 900_000}

    def label_at(dim: str, ts_ms: int) -> str | None:
        """事件时点（15m 周期开盘锚）→ 该维标签；暖机/缺标签 → None。"""
        cyc = int(ts_ms) // 900_000
        j = int(np.searchsorted(cyc_f, float(cyc)))
        if j >= N or int(cycs[j]) != cyc:
            return None
        v = labels[dim][j]
        return v if v != "" else None

    meta["label_at"] = label_at
    return labels, meta


def compute_baseline(label_at, c5: list[tuple]) -> tuple[dict, float]:
    """各维格子的全市场次周期方向率（偏离口径基线）。

    结算口径与场景信号一致：信号根锚定，次 15m 根 close<open → DOWN；
    次根平盘/不连续跳过（同 tsw.compute_baseline L708-712）。KREV 押 UP，
    统计层取 1−p_down。ER/RV 暖机期无标签自然不入基线（与事件域一致）。
    """
    open_by_ts = {r[0]: r[1] for r in c5}
    close_by_ts = {r[0]: r[4] for r in c5}
    cycs_u = sorted({int(t) // 900_000 for t in open_by_ts})
    bl: dict[str, dict[str, dict]] = {d: {} for d in DIMS_720}
    n_total = n_down = 0
    for cyc in cycs_u:
        open_ms = cyc * 900_000
        o_n = open_by_ts.get(open_ms + 900_000)     # 次周期首根 5m 开盘 = 次 15m 开盘
        c_n = close_by_ts.get(open_ms + 600_000 + 300_000)  # 次周期末根 5m 收盘 = 次 15m 收盘
        if o_n is None or c_n is None or o_n <= 0 or c_n == o_n:
            continue
        down = c_n < o_n
        n_total += 1
        n_down += down
        for d in DIMS_720:
            lab = label_at(d, open_ms)
            if lab is None:
                continue
            cell = bl[d].setdefault(lab, {"n": 0, "down": 0})
            cell["n"] += 1
            cell["down"] += down
    baseline = {d: {b: {"n": v["n"], "p_down": v["down"] / v["n"]}
                    for b, v in cells.items()}
                for d, cells in bl.items()}
    global_p = n_down / n_total if n_total else 0.5
    return baseline, global_p


# ======================================================================
# C 区：分组统计（偏离口径；报价族真实报价，K 线族 @0.50 假设 + 位移双口径）
# ======================================================================

def breakeven_q(key: str, q: float) -> float:
    """逐笔盈亏平衡胜率：x4 族含溢 0.01 → (q+0.01)/0.98；quote 族无溢价 → q/0.98。
    （直接复用 tsw.SIGNALS 注册表判定，勿手抄；同 tsw.breakeven_of L216-222。）"""
    return (q + 0.01) / 0.98 if tsw.SIGNALS[key]["breakeven_mode"] == "x4" else q / 0.98


def _ev_real(key: str, win: bool, q: float) -> float:
    """逐笔已实现 EV（同 tsw.ev_of L225-230 口径）。"""
    if not win:
        return -1.0
    return (0.98 / min(max(q + 0.01, 0.01), 0.99) - 1.0) \
        if tsw.SIGNALS[key]["breakeven_mode"] == "x4" else (0.98 / q - 1.0)


def _seg_stats(rs: list[Rec], base_p: float | None) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0}
    k = sum(r.win for r in rs)
    wr = k / n
    lo, hi = wilson(wr, n)
    out = {"n": n, "wins": k, "wr": round(wr, 4),
           "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
           "mde_pp": round(min_detectable_effect(n, wr) * 100, 2)}
    if base_p is not None:
        out["dev_pp"] = round((wr - base_p) * 100, 2)
    return out


def group_cell(key: str, dim: str, bin_name: str, rs: list[Rec],
               base_p: float | None, is_end: int | None, req_pp: float) -> dict:
    """(信号 × 维 × 格) → 池化 + IS/OOS 分段统计 + 采信裁决（D 区闸门内嵌）。

    采信 = IS/OOS 两段偏离同向（各自 |dev|≥1pp 方向明确）∧ OOS n≥30 ∧
    OOS Wilson 越过格基线 ∧ |池化偏离| ≥ 多重检验门槛。
    is_end=None（报价族，计划 Step3）：按样本时间中位切前后两个半段，
    同向仅记弱证据（样本短，采信闸自然拦截）。
    """
    rs = sorted(rs, key=lambda r: r.ts_ms)
    pooled = _seg_stats(rs, base_p)
    split = rs[len(rs) // 2].ts_ms if is_end is None else is_end
    is_rs = [r for r in rs if r.ts_ms < split]
    oos_rs = [r for r in rs if r.ts_ms >= split]
    seg_is, seg_oos = _seg_stats(is_rs, base_p), _seg_stats(oos_rs, base_p)

    quoted = [(r.q, r.win) for r in rs if r.q is not None]
    avg_ev = (round(sum(_ev_real(key, w, q) for q, w in quoted) / len(quoted), 4)
              if quoted else None)
    qs = [q for q, _ in quoted]
    be = round(sum(breakeven_q(key, q) for q in qs) / len(qs), 4) if qs else None
    disps = [r.disp for r in rs if r.disp is not None]
    ev050 = round(sum(0.98 / 0.51 - 1.0 if r.win else -1.0 for r in rs) / len(rs), 4)

    row = {
        "signal": key, "dim": dim, "bin": bin_name,
        "n": pooled["n"], "wins": pooled.get("wins", 0), "wr": pooled.get("wr"),
        "ci_lo": pooled.get("ci_lo"), "ci_hi": pooled.get("ci_hi"),
        "mde_pp": pooled.get("mde_pp"),
        "baseline_p": round(base_p, 4) if base_p is not None else None,
        "dev_pp": pooled.get("dev_pp"),
        "avg_ev": avg_ev, "avg_breakeven": be,
        "avg_q": round(sum(qs) / len(qs), 4) if qs else None,
        "ev_at_050": ev050,
        "avg_disp_pct": round(sum(disps) / len(disps) * 100, 3) if disps else None,
        "is": seg_is, "oos": seg_oos,
    }

    # ---- 采信裁决 ----
    if pooled["n"] < MIN_N:
        row["verdict"] = "INSUFFICIENT_POWER"
        return row
    dis = seg_is.get("dev_pp") if seg_is.get("n") else None
    dos = seg_oos.get("dev_pp") if seg_oos.get("n") else None
    same_dir = (dis is not None and dos is not None
                and dis * dos > 0 and abs(dis) >= 1.0 and abs(dos) >= 1.0)
    oos_ok = seg_oos.get("n", 0) >= MIN_N
    wilson_cross = False
    if oos_ok and base_p is not None and dos is not None:
        wilson_cross = (seg_oos["ci_lo"] > base_p if dos > 0
                        else seg_oos["ci_hi"] < base_p)
    big = pooled.get("dev_pp") is not None and abs(pooled["dev_pp"]) >= req_pp
    row["verdict"] = "ADOPTED" if (same_dir and oos_ok and wilson_cross and big) else (
        "CANDIDATE" if (same_dir and oos_ok and wilson_cross) else "EXPLORE")
    return row


def group_table(recs: list[Rec], dims: tuple, baseline: dict | None,
                label_at, is_end: int | None, req_pp: float,
                dir_map: dict[str, str] | None = None) -> tuple[list[dict], int]:
    """records × dims 分组 → 行列表（含暖机/缺标签丢弃计数）。"""
    groups: dict[tuple, list[Rec]] = {}
    n_drop = 0
    for r in recs:
        for d in dims:
            lab = label_at(d, r.ts_ms) if d != "month" else month_key(r.ts_ms)
            if lab is None:
                n_drop += 1
                continue
            groups.setdefault((r.key, d, lab), []).append(r)
    rows = []
    for (key, d, b), rs in sorted(groups.items()):
        base_p = None
        if baseline is not None:
            pd = (baseline.get(d) or {}).get(b, {}).get("p_down")
            if pd is not None:
                direction = (dir_map or {}).get(key, "DOWN")
                base_p = pd if direction == "DOWN" else 1.0 - pd
        rows.append(group_cell(key, d, b, rs, base_p, is_end, req_pp))
    return rows, n_drop


def monthly_drift(recs: list[Rec], label_at) -> dict:
    """月份漂移（explore 只报告不采信）：逐月胜率 + 前后半斜率 + 逐月 regime 占比。"""
    by: dict[str, list[Rec]] = {}
    for r in recs:
        by.setdefault(r.key, []).append(r)
    out = {}
    for key, rs in sorted(by.items()):
        months: dict[str, list[Rec]] = {}
        for r in rs:
            months.setdefault(month_key(r.ts_ms), []).append(r)
        mks = sorted(months)
        monthly = {m: {"n": len(months[m]),
                       "wr": round(sum(x.win for x in months[m]) / len(months[m]), 4)}
                   for m in mks}
        # 前后半斜率：前半加权胜率 − 后半加权胜率（pp，正=前高后低衰减）
        half = max(1, len(mks) // 2)

        def _w(ms):
            n = sum(monthly[m]["n"] for m in ms)
            return (sum(monthly[m]["wr"] * monthly[m]["n"] for m in ms) / n
                    if n else None)

        w1, w2 = _w(mks[:half]), _w(mks[half:])
        slope = round((w1 - w2) * 100, 2) if w1 is not None and w2 is not None else None
        # 逐月 regime（er_band）占比：区分信号衰减与行情结构变化
        rg: dict[str, dict[str, float]] = {}
        for m in mks:
            cnt: dict[str, int] = {}
            for r in months[m]:
                lab = label_at("er_band", r.ts_ms) or "暖机/无"
                cnt[lab] = cnt.get(lab, 0) + 1
            tot = sum(cnt.values())
            rg[m] = {k: round(v / tot, 3) for k, v in cnt.items()}
        out[key] = {"monthly": monthly, "slope_first_minus_last_pp": slope,
                    "regime_share": rg}
    return out


# ======================================================================
# E 区：报告（草稿期落 .pytest_tmp/）
# ======================================================================

HDR = ("| 信号 | 维 | 格 | n | 胜率 | Wilson95% | 基线 | 偏离pp | IS段 | OOS段 | "
       "avgEV | 位移均值 | 判定 |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def _fmt_row(r: dict) -> str:
    is_s, oos_s = r["is"], r["oos"]
    is_txt = (f"{is_s.get('wr', 0):.1%}(n={is_s.get('n', 0)},"
              f"Δ{is_s.get('dev_pp', '?')})" if is_s.get("n") else "—")
    oos_txt = (f"{oos_s.get('wr', 0):.1%}(n={oos_s.get('n', 0)},"
               f"Δ{oos_s.get('dev_pp', '?')})" if oos_s.get("n") else "—")
    bp = f"{r['baseline_p']:.1%}" if r["baseline_p"] is not None else "—"
    dev = f"{r['dev_pp']:+.1f}" if r["dev_pp"] is not None else "—"
    ev_txt = f"{r['avg_ev']:+.3f}" if r["avg_ev"] is not None else "—"
    disp_txt = f"{r['avg_disp_pct']:+.3f}%" if r["avg_disp_pct"] is not None else "—"
    return (f"| {r['signal']} | {r['dim']} | {r['bin']} | {r['n']} | "
            f"{(r['wr'] or 0):.1%} | [{(r['ci_lo'] or 0):.1%},{(r['ci_hi'] or 0):.1%}] | "
            f"{bp} | {dev} | {is_txt} | {oos_txt} | {ev_txt} | {disp_txt} | {r['verdict']} |")


def write_report(path: str, json_path: str, rows720: list[dict], rows_quote: list[dict],
                 drift: dict, baseline: dict, global_p: float, meta_all: dict,
                 req_pp: float) -> None:
    L: list[str] = []
    L.append("# 720 天信号周期分段深度分析\n")
    L.append(f"生成：{meta_all['stamp']}（UTC 数据窗 {_iso(meta_all['ts0'])} → "
             f"{_iso(meta_all['ts1'])}，IS/OOS 分界 {_iso(meta_all['is_end'])}）\n")
    L.append(f"\n**统计纪律**：n<{MIN_N} 标 INSUFFICIENT_POWER 不解读；采信 = IS/OOS 两段"
             f"偏离同向 ∧ OOS n≥{MIN_N} ∧ OOS Wilson 越过格基线 ∧ |池化偏离|≥{req_pp}pp"
             f"（多重检验分母 = 全表 prior 格数 {N_PRIOR_CELLS}）。基线 = 该维该格全市场"
             f"次周期方向率（全局 P(DOWN)={global_p:.1%}）；KREV 押 UP 取 1−p_down。"
             f"K 线族无预测市场报价 → avgEV=—，另给 @0.50 假设与方向化位移均值双口径。\n")
    L.append(f"\n**冻结阈值（IS 段分位，断言通过）**：ER q50="
             f"{meta_all['quantiles']['er50']:.4f} / q75={meta_all['quantiles']['er75']:.4f}；"
             f"RV q33={meta_all['quantiles']['rv33']:.6f} / "
             f"q67={meta_all['quantiles']['rv67']:.6f}；ret24 ±{TREND_TH:.0%}。\n")

    L.append("\n## 对账闸结果（全过）\n")
    g = meta_all["gates"]
    L.append(f"- KREV 时间切片：{g['krev_slice']}（缺口带 [0,{KREV_GAP_MAX}]）")
    L.append(f"- 场景族胜率：{g['scene_wr']}")
    L.append(f"- regime 占比：{g['regime_share']}")
    L.append(f"- 事件量：{json.dumps(g['event_counts'], ensure_ascii=False)}\n")

    L.append("\n## Section A：720d 可重建族（场景 + KREV）× 4 周期维 + IS/OOS 双向\n")
    L.append(HDR)
    for r in sorted(rows720, key=lambda x: (x["signal"], x["dim"], str(x["bin"]))):
        L.append(_fmt_row(r))

    L.append("\n## 月份漂移（explore 只报告，不采信）\n")
    for key, d in drift.items():
        L.append(f"\n### {key}（前半−后半斜率 {d['slope_first_minus_last_pp']}pp）\n")
        mk = sorted(d["monthly"])
        cells = " ".join(f"{m[2:]}:{d['monthly'][m]['wr']:.0%}({d['monthly'][m]['n']})"
                         for m in mk)
        L.append(f"- 逐月：{cells}")
        rg_str = " ".join(
            f"{m[2:]}[牛{d['regime_share'][m].get('趋势牛', 0):.0%}/"
            f"熊{d['regime_share'][m].get('趋势熊', 0):.0%}/"
            f"震{d['regime_share'][m].get('震荡', 0):.0%}]" for m in mk)
        L.append(f"- regime 结构：{rg_str}")

    L.append("\n## Section B：短历史报价族（~38 天，仅 session+ret24，物理分区）\n")
    L.append("> IS/OOS = 数据前后两个半段；同向仅记弱证据；MDE>5pp 的格子降级"
             "INSUFFICIENT_POWER（38d 样本功效不足）。\n")
    L.append(HDR)
    for r in sorted(rows_quote, key=lambda x: (x["signal"], x["dim"], str(x["bin"]))):
        L.append(_fmt_row(r))

    # ---- 三层结论 ----
    adopted = [r for r in rows720 + rows_quote if r["verdict"] == "ADOPTED"]
    cand = [r for r in rows720 + rows_quote if r["verdict"] == "CANDIDATE"]
    L.append("\n## 三层结论（采信要求 IS/OOS 双向同向）\n")
    L.append("**采信（%d）**：%s" % (len(adopted), "；".join(
        f"{r['signal']}×{r['dim']}={r['bin']}（n={r['n']}，偏离 {r['dev_pp']:+}pp）"
        for r in adopted) or "无"))
    L.append("\n**候选（%d）**：%s" % (len(cand), "；".join(
        f"{r['signal']}×{r['dim']}={r['bin']}（n={r['n']}，偏离 {r['dev_pp']:+}pp）"
        for r in cand) or "无"))
    L.append("\n**探索**：其余全部格子（含 INSUFFICIENT_POWER），只作行情结构参考。\n")

    L.append("\n## 交叉引用\n")
    L.append("- 日历 11 维结论引用 `output/timeslice_winrate_report_20260829_1152.md`"
             "（本脚本仅补 session_utc 交互）")
    L.append("- S5 regime 先例：`output/s5_regime_analysis.log`（本脚本将单信号结论"
             "推广到全信号族并加 IS/OOS 双向闸）")
    L.append("- KREV 口径保真：`tests/test_kline_shadow_detector.py`（时间切片对账见对账闸）")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta_all, "baseline": baseline, "global_p_down": global_p,
                   "rows_720d": rows720, "rows_quote": rows_quote,
                   "monthly_drift": drift},
                  f, ensure_ascii=False, indent=1, default=str)


# ======================================================================
# main
# ======================================================================

def main() -> int:
    sys.stdout = Tee()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M")
    t_start = _time.monotonic()
    print("=" * 72)
    print("720 天信号周期分段深度分析")
    print("=" * 72)

    print("[1/6] 加载 720d 5m K 线（JSON 快路径）...")
    c5 = load_c5_json()
    print(f"      {len(c5)} 根 | {_iso(c5[0][0])} → {_iso(c5[-1][0])} UTC")

    print("[2/6] B 区标签器（ER_7d/RV_7d/ret24/session/month，IS 段冻结阈值断言）...")
    labels, lmeta = build_labels(c5)
    label_at = lmeta["label_at"]
    is_end = lmeta["is_end"]
    print(f"      15m 周期 {lmeta['n_cycles']} | 暖机丢弃 {lmeta['n_warm']} 根"
          f" | IS/OOS 分界 {_iso(is_end)}")

    print("[3/6] 基线层（各维格全市场次周期方向率）...")
    baseline, global_p = compute_baseline(label_at, c5)
    print(f"      全局 P(DOWN)={global_p:.1%}")
    for d in DIMS_720:
        cells = " ".join(f"{b}:{v['p_down']:.1%}(n={v['n']})"
                         for b, v in sorted(baseline[d].items()))
        print(f"      {d}: {cells}")

    print("[4/6] A 区事件表 ...")
    scene_recs, _scene_meta = replay_scene_recs(c5)
    print(f"      场景族={len(scene_recs)}（S1/S2/S4/S5）")

    # ---- 对账闸 ②：场景族胜率对齐 timeslice 基准（>1pp 中止）----
    wr_msgs = []
    for key, (bn, bwr) in SCENE_BENCH.items():
        rs = [r for r in scene_recs if r.key == key]
        n = len(rs)
        wr = sum(r.win for r in rs) / n if n else 0.0
        dpp = (wr - bwr) * 100
        wr_msgs.append(f"{key} n={n}(基准{bn}) wr={wr:.1%}(基准{bwr:.1%}，Δ{dpp:+.1f}pp)")
        if abs(dpp) > WR_TOL_PP:
            raise SystemExit(f"[对账闸] {key} 胜率偏差 {dpp:+.2f}pp > {WR_TOL_PP}pp")
    print("      对账②：" + "；".join(wr_msgs))

    # ---- 对账闸 ③：regime 占比对齐 s5_regime_analysis.log ----
    share = {rg: 0 for rg in REGIME_SHARE_FROZEN}
    n_lab = 0
    for j_lab in labels["er_band"]:
        if j_lab in share:
            share[j_lab] += 1
            n_lab += 1
    share = {k: v / n_lab for k, v in share.items()}
    share_msg = " ".join(f"{k}:{share[k]:.1%}/冻结{v:.1%}"
                         for k, v in REGIME_SHARE_FROZEN.items())
    worst = max(abs(share[k] - v) for k, v in REGIME_SHARE_FROZEN.items())
    if worst > SHARE_TOL:
        raise SystemExit(f"[对账闸] regime 占比漂移 {worst:.1%} > {SHARE_TOL:.0%}：{share_msg}")
    print(f"      对账③：{share_msg}（最大漂移 {worst:.1%} ≤ {SHARE_TOL:.0%} OK）")

    krev_recs, krev_meta = replay_krev_recs()
    # ---- 对账闸 ①：KREV 时间切片（窗口滚动 3 天 → 缺口带 [0,8]）----
    slice_msgs = []
    for v, total in KREV_TOTAL.items():
        sl = krev_meta["slice"][v]
        gap = total - sl
        slice_msgs.append(f"{v} 全窗={krev_meta['total'][v]} 切片={sl} "
                          f"缺口={gap}(=注册表{total}−切片)")
        if not (0 <= gap <= KREV_GAP_MAX):
            raise SystemExit(f"[对账闸] KREV 时间切片缺口异常：{slice_msgs[-1]}")
    print("      对账①：" + "；".join(slice_msgs))
    print(f"      KREV 族={len(krev_recs)}（A={krev_meta['total']['krev_a_v1']}，"
          f"B={krev_meta['total']['krev_b_v1']}）")

    print("      报价族（~38 天情绪窗重放：x4_v1/v2 + quote_edge 8 版）...")
    with open(SENTIMENT_JSON, encoding="utf-8") as f:
        raw_wins = json.load(f)
    wins = {int(w["start_time"]): w for w in raw_wins}
    ordered = [wins[k] for k in sorted(wins)]
    x4_recs, x4_meta = replay_x4_recs(wins, ordered)
    quote_raw, _qmeta = tsw.replay_shadow_signals()
    quote_recs = _rec_from_shadow(quote_raw)
    print(f"      x4={len(x4_recs)}（v1={x4_meta['n_v1']} v2={x4_meta['n_v2']}，"
          f"无入场价丢弃 {x4_meta['n_no_entry']}）| quote_edge={len(quote_recs)}")

    recs720 = scene_recs + krev_recs
    recs_quote = x4_recs + quote_recs
    dir_map = {**{r.key: r.direction for r in recs720},
               **{r.key: r.direction for r in recs_quote}}

    print("[5/6] C/D 区统计（多重检验预算 + IS/OOS 双向闸门）...")
    mt = multiple_testing_threshold(BONF_BASE_PP, N_PRIOR_CELLS)
    req_pp = mt["required_pp"]
    print(f"      {mt['note']}")
    rows720, drop720 = group_table(recs720, DIMS_720, baseline, label_at,
                                   is_end, req_pp, dir_map)
    rows_quote, dropq = group_table(recs_quote, DIMS_QUOTE, baseline, label_at,
                                    None, req_pp, dir_map)   # is_end=None → 前后半段切分
    # 报价族功效裁剪 + 弱证据封顶（计划 Step3）：MDE>5pp 降级；
    # 前后半段同向仅记弱证据 → ADOPTED 封顶为 CANDIDATE，38d 样本不作采信。
    for r in rows_quote:
        if r["verdict"] == "ADOPTED":
            r["verdict"] = "CANDIDATE"
            r["note"] = "半段同向仅弱证据（~38d）"
        if r["verdict"] != "INSUFFICIENT_POWER" and (r["mde_pp"] or 0) > 5.0:
            r["verdict"] = "INSUFFICIENT_POWER"
            r["note"] = f"MDE={r['mde_pp']}pp>5pp，38d 样本功效不足"
    print(f"      720d 族格子={len(rows720)}（暖机/缺标签丢弃 {drop720}）| "
          f"报价族格子={len(rows_quote)}（丢弃 {dropq}）")
    n_ad = sum(1 for r in rows720 + rows_quote if r["verdict"] == "ADOPTED")
    n_cd = sum(1 for r in rows720 + rows_quote if r["verdict"] == "CANDIDATE")
    n_ip = sum(1 for r in rows720 + rows_quote if r["verdict"] == "INSUFFICIENT_POWER")
    print(f"      采信={n_ad} 候选={n_cd} 功效不足={n_ip}")

    drift = monthly_drift(recs720, label_at)

    print("[6/6] E 区报告 ...")
    event_counts: dict[str, int] = {}
    for r in recs720 + recs_quote:
        event_counts[r.key] = event_counts.get(r.key, 0) + 1
    meta_all = {
        "stamp": stamp, "ts0": c5[0][0], "ts1": c5[-1][0], "is_end": is_end,
        "quantiles": lmeta["quantiles"], "n_warm_dropped": lmeta["n_warm"],
        "gates": {
            "krev_slice": "；".join(slice_msgs),
            "scene_wr": "；".join(wr_msgs),
            "regime_share": share_msg,
            "event_counts": event_counts,
        },
    }
    md_path = os.path.join(OUT_DIR, f"cycle_regime_report_{stamp}.md")
    json_path = os.path.join(OUT_DIR, f"cycle_regime_report_{stamp}.json")
    write_report(md_path, json_path, rows720, rows_quote, drift, baseline,
                 global_p, meta_all, req_pp)
    print(f"\n报告 → {md_path}")
    print(f"JSON → {json_path}")
    print(f"耗时 {_time.monotonic() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
