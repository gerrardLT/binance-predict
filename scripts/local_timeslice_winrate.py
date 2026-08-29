#!/usr/bin/env python3
"""信号时段胜率全方位回测（2026-08-29）：所有场景 + 影子信号按时间维度切分胜率。

核心假设：BTC 不同时段流动性差异会改变各信号胜率。维度 = 小时(UTC/北京) /
时段(亚欧美深夜) / 周几 / 周末 / 月内段 / 月份 / 季节，全部表驱动（DIMS）。

数据源分层（线上仅 ~17 天，月/年维度必须重放）：
  replay  场景 S1/S2/S4/S5：720d K 线重放（build_events + F25 pos4h 补齐）
          影子 10 版本：本地全曲线情绪窗口重放（冻结口径常量自 detector import）
  online  Actions 全量导出 CSV（优先）→ API 快照 JSON（降级，200 条截断标注）

统计纪律（复用 backtest/stats.py，不重造）：
  n<30 标 INSUFFICIENT 不解读；Wilson 95% CI + MDE；Bonferroni 按先验检验数；
  三层采信：探索（全格子）→ 候选（n≥30 且 CI 下界越过基线）→ 采信（候选 ∧
  Bonferroni ∧ 双源复现）。信号全押 DOWN → 必须对照 720d 各时段 DOWN 基线，
  输出「偏离」而非裸胜率。late_night 族本身时段限定（HOUR_GUARDS），标
  time_gated 不参与时段发现解读（循环论证防护）。

口径冻结引用（勿复制改写）：
  quote_edge_detector.py:92-101 QUOTE_EDGE_RULES / :109-112 HOUR_GUARDS /
  :117-119 LN_DD_GUARDS / :128-132 V2_PRICE_GUARDS / :136-140 V3_*
  fake_breakout_detector.py:90-102 S1/S2/S4 定义（F25 pos4h≥0.9 / 连阳≥3）/
  :228-249 S5 确认（⟺ events.py z5<0）
  misalignment_detector.py X4_END_MAX=40 / DECISION_T=150s / past1h±10min

用法：.venv\\Scripts\\python.exe -X utf8 scripts/local_timeslice_winrate.py
输出：output/timeslice_winrate_report_<stamp>.md / .json（+ 可选热力图 PNG）
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import sys
import time as _time
from dataclasses import dataclass
from datetime import timezone as _tz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from binance_predict.backtest.stats import (  # noqa: E402
    min_detectable_effect,
    multiple_testing_threshold,
    wilson,
)

# 服务层常量延迟到函数内 import（模块顶层保持轻依赖，测试可安全 import）

KLINES_CSV = os.path.join(ROOT, "output", "klines_5m_720d.csv")
SENTIMENT_JSON = os.path.join(ROOT, "output", "sentiment_windows_online_fixed.json")
EXPORT_GLOB = os.path.join(ROOT, "output", "timeslice_export_*")
SNAPSHOT_GLOB = os.path.join(ROOT, "output", "timeslice_snapshot_*.json")

# 线上场景信号结算参考胜率（真 OOS，fake_breakout_detector RESEARCH_WIN_RATES）
SCENE_BENCH = {
    "S1": 0.644, "S2": 0.536, "S4": 0.554, "S5": 0.785,
}
# 影子族既有回测基准（local_shadow_full_analysis BENCH，对账用）
SHADOW_BENCH = {
    "x4_v1": (0.635, 0.254), "quote_momentum_v1": (0.799, 0.097),
    "quote_contrarian_v1": (0.240, 0.155),
}


@dataclass(frozen=True)
class Record:
    """统一信号记录：双源（replay/online）统一口径的唯一载体。"""
    source: str            # "replay" | "online"
    signal_key: str        # SIGNALS 注册表键
    ts_ms: int             # 信号时点锚（影子=窗口开 / 场景=信号K开盘）
    direction: str         # 押注方向（本系统几乎全 DOWN；S2 为 UP）
    win: bool
    q: float | None        # 入场报价（场景重放无报价 → None）
    ev: float | None


# 信号注册表：key → family / market_period / breakeven_mode / settle_dir / time_gated
# breakeven_mode: "x4" = 0.98/(q+0.01)−1（费2%+溢0.01）/ "quote" = 0.98/q−1（费2%无溢价）
# time_gated=True 的版本其 hour/session 维度不参与时段发现（循环论证防护）。
# krev_v1/v2 为注册位：2026-08-28 上线无数据，攒 2-3 周后零改动接入。
SIGNALS: dict[str, dict] = {
    "x4_v1":                     {"family": "shadow", "market_period": "5m-next",
                                  "breakeven_mode": "x4",    "settle_dir": "DOWN", "time_gated": False},
    "x4_v2":                     {"family": "shadow", "market_period": "5m-next",
                                  "breakeven_mode": "x4",    "settle_dir": "DOWN", "time_gated": False},
    "quote_momentum_v1":         {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "quote_momentum_v2":         {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "quote_contrarian_v1":       {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "quote_contrarian_v2":       {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "quote_contrarian_v3a":      {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "quote_contrarian_v3b":      {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": False},
    "late_night_contrarian_v1":  {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": True},
    "late_night_contrarian_v2":  {"family": "shadow", "market_period": "same",
                                  "breakeven_mode": "quote", "settle_dir": "DOWN", "time_gated": True},
    "S1": {"family": "scene", "market_period": "next15m",
           "breakeven_mode": "x4", "settle_dir": "DOWN", "time_gated": False},
    "S2": {"family": "scene", "market_period": "next15m",
           "breakeven_mode": "x4", "settle_dir": "UP",   "time_gated": False},
    "S4": {"family": "scene", "market_period": "next15m",
           "breakeven_mode": "x4", "settle_dir": "DOWN", "time_gated": False},
    "S5": {"family": "scene", "market_period": "next15m",
           "breakeven_mode": "x4", "settle_dir": "DOWN", "time_gated": False},
    "krev_v1": {"family": "shadow", "market_period": "-",
                "breakeven_mode": "-", "settle_dir": "-", "time_gated": False, "placeholder": True},
    "krev_v2": {"family": "shadow", "market_period": "-",
                "breakeven_mode": "-", "settle_dir": "-", "time_gated": False, "placeholder": True},
}


# ----------------------------------------------------------------------
# 3b. 时间标签器（纯函数，全 UTC ms 锚定；北京时间 = UTC+8 固定偏移，
#     沿用 signal_notify TZ_BJT 先例。加维度 = 加一行 DIMS 条目）
# ----------------------------------------------------------------------

def _gm(ts_ms: int) -> _time.struct_time:
    return _time.gmtime(ts_ms / 1000.0)


def _gm_bjt(ts_ms: int) -> _time.struct_time:
    return _time.gmtime(ts_ms / 1000.0 + 8 * 3600)


def hour_utc(ts_ms: int) -> str:
    return f"{_gm(ts_ms).tm_hour:02d}"


def hour_bjt(ts_ms: int) -> str:
    return f"{_gm_bjt(ts_ms).tm_hour:02d}"


_DOW_CN = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def dow_utc(ts_ms: int) -> str:
    return _DOW_CN[_gm(ts_ms).tm_wday]


def dow_bjt(ts_ms: int) -> str:
    return _DOW_CN[_gm_bjt(ts_ms).tm_wday]


def weekend_utc(ts_ms: int) -> str:
    return "weekend" if _gm(ts_ms).tm_wday >= 5 else "weekday"


def weekend_bjt(ts_ms: int) -> str:
    return "weekend" if _gm_bjt(ts_ms).tm_wday >= 5 else "weekday"


# 非重叠 4 段（表驱动可调）：亚洲 00-07 / 欧洲 07-13 / 美洲 13-21 / 深夜 21-24
# 锚点依据：欧盘伦敦开 ~07-08 UTC、美盘纽约开 ~13:30-14:30 UTC
SESSION_EDGES = (("asia", 0, 7), ("europe", 7, 13), ("america", 13, 21), ("late", 21, 24))


def session_utc(ts_ms: int) -> str:
    h = _gm(ts_ms).tm_hour
    for name, lo, hi in SESSION_EDGES:
        if lo <= h < hi:
            return name
    return "late"


def session_bjt(ts_ms: int) -> str:
    h = _gm_bjt(ts_ms).tm_hour
    for name, lo, hi in SESSION_EDGES:
        if lo <= h < hi:
            return name
    return "late"


def dom_phase(ts_ms: int) -> str:
    d = _gm(ts_ms).tm_mday
    return "early(01-10)" if d <= 10 else ("mid(11-20)" if d <= 20 else "late(21-31)")


def month(ts_ms: int) -> str:
    return f"{_gm(ts_ms).tm_mon:02d}"


def season(ts_ms: int) -> str:
    m = _gm(ts_ms).tm_mon
    return "spring" if m in (3, 4, 5) else ("summer" if m in (6, 7, 8)
                                             else ("autumn" if m in (9, 10, 11) else "winter"))


# 维度表：name → (bin_fn, kind)；kind=prior 先验假设（主判据）/ explore 探索性
DIMS: dict[str, tuple] = {
    "hour_utc":    (hour_utc,    "explore"),
    "hour_bjt":    (hour_bjt,    "explore"),
    "session_utc": (session_utc, "prior"),
    "session_bjt": (session_bjt, "prior"),
    "dow_utc":     (dow_utc,     "explore"),
    "dow_bjt":     (dow_bjt,     "explore"),
    "weekend_utc": (weekend_utc, "prior"),
    "weekend_bjt": (weekend_bjt, "prior"),
    "dom_phase":   (dom_phase,   "explore"),
    "month":       (month,       "explore"),
    "season":      (season,      "explore"),
}


# ----------------------------------------------------------------------
# 口径纯函数（测试直接覆盖）
# ----------------------------------------------------------------------

def breakeven_of(signal_key: str, q: float) -> float:
    """逐笔盈亏平衡胜率（与各版本落库 EV 口径一致，勿混用）：
    x4 与场景族含溢 0.01 → (q+0.01)/0.98；quote 族无溢价 → q/0.98。
    （照抄 local_shadow_full_analysis.py:58-60 模式，按注册表判定。）
    """
    mode = SIGNALS[signal_key]["breakeven_mode"]
    return (q + 0.01) / 0.98 if mode == "x4" else q / 0.98


def ev_of(signal_key: str, win: bool, q: float) -> float:
    """逐笔已实现 EV：赢 0.98/(q[+0.01])−1 / 输 −1。"""
    mode = SIGNALS[signal_key]["breakeven_mode"]
    if not win:
        return -1.0
    return (0.98 / min(max(q + 0.01, 0.01), 0.99) - 1.0) if mode == "x4" else (0.98 / q - 1.0)


# ----------------------------------------------------------------------
# 影子重放 ex-ante 工具（复制自 local_shadow_v2v3_real_backtest.py 母本，
# 冻结口径：只用 ≤触发时点采样点）
# ----------------------------------------------------------------------

def btc_at_or_before(curve: list | None, ts: int) -> float | None:
    best = None
    for p in sorted(curve or [], key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if int(t) <= ts:
            best = float(v)
    return best


def window_open_btc(w: dict) -> float | None:
    p = w.get("entry_price")
    if p is not None and float(p) > 0:
        return float(p)
    for pt in sorted(w.get("curve_btc_price") or [], key=lambda x: x.get("t") or 0):
        v = pt.get("v")
        if v is not None and float(v) > 0:
            return float(v)
    return None


def find_first_hit(curve: list | None, start_ms: int,
                   t_lo: float, t_hi: float, q_lo: float, q_hi: float):
    if not curve:
        return None
    for p in sorted(curve, key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        t_rel = (int(t) - start_ms) / 1000.0
        if t_rel < 0:
            continue
        if t_rel >= t_hi:
            break
        if t_rel < t_lo:
            continue
        v = float(v)
        if q_lo <= v < q_hi:
            return v, int(t)
    return None


def curve_end_pct(curve: list | None) -> float | None:
    pts = sorted(curve or [], key=lambda p: p.get("t", 0))
    for p in reversed(pts):
        v = p.get("v")
        if v is not None:
            return float(v)
    return None


def price_within(curve: list | None, start_ms: int, max_sec: float):
    """≤max_sec 内最晚采样点（x4 决策点同口径）。"""
    best = None
    for p in sorted(curve or [], key=lambda x: x.get("t") or 0):
        t, v = p.get("t"), p.get("v")
        if t is None or v is None:
            continue
        if (int(t) - start_ms) / 1000.0 <= max_sec:
            best = float(v)
    return best


def build_day_high(wins: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """UTC 日 -> 按 t 升序的 (t, running_max) 前缀表（brainstorm 同口径）。"""
    by_day: dict[int, list[tuple[int, float]]] = {}
    for w in wins:
        for p in w.get("curve_btc_price") or []:
            t, v = p.get("t"), p.get("v")
            if t is not None and v is not None:
                by_day.setdefault(int(t) // 86_400_000, []).append((int(t), float(v)))
    out = {}
    for d, pts in by_day.items():
        pts.sort()
        run = -1.0
        out[d] = [(t, run := max(run, v)) for t, v in pts]
    return out


def day_high_before(prefix: dict, ts: int) -> float | None:
    pts = prefix.get(ts // 86_400_000)
    if not pts:
        return None
    best = None
    for t, m in pts:
        if t <= ts:
            best = m
        else:
            break
    return best


def _x4_base_local(wins: dict, ws: int) -> float | None:
    """start−1h（±10min 容差，300s 格点）内最晚有 entry_price 的窗口价。"""
    target = ws - 3_600_000
    best_t, best_p = None, None
    for s in range(target - 600_000, target + 600_000 + 1, 300_000):
        w = wins.get(s)
        if w and w.get("entry_price"):
            if best_t is None or s > best_t:
                best_t, best_p = s, float(w["entry_price"])
    return best_p


# ----------------------------------------------------------------------
# 3a-1. 场景重放（720d K 线 → S1/S2/S4/S5）
# ----------------------------------------------------------------------

def load_klines_5m(path: str = KLINES_CSV) -> list[tuple]:
    """720d 5m K 线：[(open_time_ms, o, h, l, c, v)]。
    CSV 时间列为 ISO 格式（2024-09-06T21:25:00+00:00，非 epoch ms）。"""
    rows: list[tuple] = []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            t = int(dt.datetime.fromisoformat(r["timestamp"]).timestamp() * 1000)
            rows.append((t, float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"]), float(r["volume"])))
    return rows


def replay_scene_events(c5: list[tuple]) -> tuple[list[Record], dict]:
    """720d K 线重放场景信号（口径对齐线上 detector，冻结注释见文件头）。

    S1 = events.scene1 ∧ F25 pos4h≥0.9（events.py 的 scene1 不含 pos4h，
         重放侧用 16 根 15m 收盘（含当前根）补齐——对齐
         fake_breakout_detector.py:625-632）→ win = next_down
    S2 = events.scene2 → win = not next_down
    S4 = 连阳≥3（含信号K）∧ close_pos≥0.85，无破位（周期粒度，15m 闭区间）
         → win = next_down（fake_breakout_detector.py:94-101, :195-225）
    S5 = S1 ∧ 次周期第1根5m收盘<次周期开盘（⟺ z5<0，
         fake_breakout_detector.py:228-249 confirm_bull_exhaust_5m）
    场景重放无预测市场报价 → q/ev 为 None，EV 段不参与（报告标注）。
    """
    from binance_predict.backtest import build_events
    from binance_predict.services.scene_params import DEFAULT_SCENE_PARAMS

    now_ms = c5[-1][0] + 300_000 if c5 else int(_time.time() * 1000)
    res = build_events(c5, DEFAULT_SCENE_PARAMS, now_ms)
    events = res["events"]
    agg = res["agg"]
    o15, c15 = agg["o15"], agg["c15"]
    cycs = agg["cycs"]
    n15 = len(cycs)

    # 周期级公共量：close_pos / next_down / pos4h（16 根收盘滚动区间含当前根）
    h15, l15 = agg["h15"], agg["l15"]
    close_pos = []
    for j in range(n15):
        rng = h15[j] - l15[j]
        close_pos.append((c15[j] - l15[j]) / rng if rng > 0 and o15[j] > 0 else None)
    next_down = [None] * n15
    for j in range(n15 - 1):
        if cycs[j + 1] == cycs[j] + 1 and o15[j + 1] > 0 and c15[j + 1] != o15[j + 1]:
            next_down[j] = c15[j + 1] < o15[j + 1]
    pos4h = [None] * n15
    for j in range(15, n15):  # 需 16 根（含当前）
        win = c15[j - 15:j + 1]
        hi_, lo_ = max(win), min(win)
        if hi_ > lo_:
            pos4h[j] = (c15[j] - lo_) / (hi_ - lo_)

    # j → 该周期是否有 high 侧破位事件（S4 排除：破位周期 S1 优先，避免重复计数）
    broke_high = {e["cyc"] for e in events if e["side"] == "high"}
    cyc_to_j = {c: j for j, c in enumerate(cycs)}

    # S1/S2/S5：以破位事件为单元（次周期唯一 → 同周期多级别事件只取首条）
    recs: list[Record] = []
    seen_s1 = set()
    for e in events:
        j = cyc_to_j[e["cyc"]]
        if not e["has_next"] or next_down[j] is None:
            continue
        ts_ms = e["cyc"] * 900_000
        if e["scene1"] and pos4h[j] is not None and pos4h[j] >= 0.9:
            if e["cyc"] in seen_s1:
                continue
            seen_s1.add(e["cyc"])
            recs.append(Record("replay", "S1", ts_ms, "DOWN", bool(e["next_down"]), None, None))
            # S5 = S1 ∧ z5<0（确认组；z5 None = 次周期首根数据不完整，保守跳过）
            if e["z5"] is not None and e["z5"] < 0:
                recs.append(Record("replay", "S5", ts_ms, "DOWN", bool(e["next_down"]), None, None))

    # S2 单独遍历（同周期多级别破位仅计一次，与 S1 同思路）
    seen_s2 = set()
    for e in events:
        j = cyc_to_j[e["cyc"]]
        if not e["scene2"] or not e["has_next"] or next_down[j] is None:
            continue
        if e["cyc"] in seen_s2:
            continue
        seen_s2.add(e["cyc"])
        recs.append(Record("replay", "S2", e["cyc"] * 900_000, "UP",
                           not bool(e["next_down"]), None, None))

    # S4：周期粒度（连阳≥3 含信号K ∧ 光头 ∧ 无 high 侧破位）
    for j in range(2, n15):
        if o15[j] <= 0 or c15[j] <= o15[j]:
            continue
        if close_pos[j] is None or close_pos[j] < 0.85:
            continue
        if cycs[j] in broke_high:
            continue
        if not (c15[j - 1] > o15[j - 1] and c15[j - 2] > o15[j - 2]):
            continue
        if next_down[j] is None:
            continue
        recs.append(Record("replay", "S4", cycs[j] * 900_000, "DOWN",
                           bool(next_down[j]), None, None))

    meta = {"n_events": len(events), "n_cycles": n15,
            "n_s1": len(seen_s1), "n_s2": len(seen_s2),
            "ts0": c5[0][0], "ts1": c5[-1][0]}
    return recs, meta


# ----------------------------------------------------------------------
# 3a-2. 影子重放（本地全曲线情绪窗口 → 10 版本，门禁链逐字段对齐
#       quote_edge_detector.py:443-475 主循环；冻结常量 import 不复制）
# ----------------------------------------------------------------------

def replay_shadow_signals(path: str = SENTIMENT_JSON) -> tuple[list[Record], dict]:
    """在 sentiment_windows_online_fixed.json（07-13~08-19）上重放 10 个影子版本。

    门禁链（与线上检测器逐字段一致）：
      base 4 版 = QUOTE_EDGE_RULES 区间首命中（含 late_night 两版）
      v2（momentum/contrarian）= base ∩ chg 门禁（V2_PRICE_GUARDS）
      v3a = contrarian_v2 ∩ 前窗 outcome==DOWN；v3b = v3a ∩ 距日高回落≤V3_DD_THRESHOLD
      late_night_v1 = base ∩ HOUR_GUARDS；v2 = v1 ∩ 距日高回落≤LN_DD 阈值
    缺 curve_down_price 的窗口保守跳过并统计覆盖率（与母本同处理）。
    """
    from binance_predict.services.quote_edge_detector import (
        HOUR_GUARDS, LN_DD_GUARDS, QUOTE_EDGE_RULES,
        V2_PRICE_GUARDS, V3_DD_THRESHOLD, V3_ENV_GUARDS,
    )

    with open(path, encoding="utf-8") as f:
        wins = json.load(f)
    wins = {int(w["start_time"]): w for w in wins}
    ordered = [wins[k] for k in sorted(wins)]
    prefix = build_day_high(ordered)

    # 全版本触发区间表（对齐主循环 rules 构建：v2/v3 用 base 区间）
    rules: dict[str, tuple[float, float, float, float]] = dict(QUOTE_EDGE_RULES)
    for v2, (base, _m, _p) in V2_PRICE_GUARDS.items():
        rules[v2] = QUOTE_EDGE_RULES[base]
    for v3 in V3_ENV_GUARDS:
        rules[v3] = QUOTE_EDGE_RULES["quote_contrarian_v1"]

    recs: list[Record] = []
    n_no_curve = 0
    n_skip_guard = {v: 0 for v in rules}

    def _hour_ok(version: str, ws: int) -> bool:
        g = HOUR_GUARDS.get(version)
        if g is None:
            return True
        return g[0] <= (ws // 3_600_000 + 8) % 24 < g[1]

    def _dd_pct(w: dict, ts: int) -> float | None:
        """触发时点距当日（UTC）高点回落%（≤0 为回落）；数据缺失 → None。"""
        cur = btc_at_or_before(w.get("curve_btc_price"), ts)
        dh = day_high_before(prefix, ts)
        if cur is None or not dh:
            return None
        return (cur - dh) / dh * 100.0

    for w in ordered:
        ws = int(w["start_time"])
        outcome = w.get("outcome")
        if outcome not in ("UP", "DOWN"):
            continue
        curve = w.get("curve_down_price")
        if not curve:
            n_no_curve += 1
            continue

        for version, (t_lo, t_hi, q_lo, q_hi) in rules.items():
            hit = find_first_hit(curve, ws, t_lo, t_hi, q_lo, q_hi)
            if hit is None:
                continue
            q, ts = hit
            if not _hour_ok(version, ws):
                n_skip_guard[version] += 1
                continue
            # v2 chg 门禁（数据缺失保守跳过，与线上一致）
            if version in V2_PRICE_GUARDS:
                _b, mode, thr = V2_PRICE_GUARDS[version]
                base_p = window_open_btc(w)
                cur = btc_at_or_before(w.get("curve_btc_price"), ts)
                if base_p is None or cur is None:
                    n_skip_guard[version] += 1
                    continue
                chg = (cur - base_p) / base_p * 100.0
                ok = chg <= thr if mode == "min_drop" else chg < thr
                if not ok:
                    continue
            # v3 = contrarian_v2 chg 门禁 ∩ 前窗 DOWN（∩ v3b 距日高回落）
            if version in V3_ENV_GUARDS:
                _b, mode, thr = V2_PRICE_GUARDS["quote_contrarian_v2"]
                base_p = window_open_btc(w)
                cur = btc_at_or_before(w.get("curve_btc_price"), ts)
                if base_p is None or cur is None:
                    n_skip_guard[version] += 1
                    continue
                chg = (cur - base_p) / base_p * 100.0
                if not chg < thr:  # max_rise 模式
                    continue
                prev = wins.get(ws - 300_000)
                if (prev or {}).get("outcome") != "DOWN":
                    continue
                if V3_ENV_GUARDS[version]:
                    dd = _dd_pct(w, ts)
                    if dd is None or dd > V3_DD_THRESHOLD:
                        n_skip_guard[version] += 1
                        continue
            # 深夜门禁 v2：距日高回落 ≤ 阈值（含边界）
            if version in LN_DD_GUARDS:
                _b, thr = LN_DD_GUARDS[version]
                dd = _dd_pct(w, ts)
                if dd is None or dd > thr:
                    n_skip_guard[version] += 1
                    continue
            win = outcome == "DOWN"
            recs.append(Record("replay", version, ws, "DOWN", win, q,
                               ev_of(version, win, q)))

    n_win = len(ordered)
    meta = {"n_windows": n_win, "n_no_curve": n_no_curve,
            "curve_coverage": 1.0 - n_no_curve / n_win if n_win else 0.0,
            "ts0": ordered[0]["start_time"] if ordered else None,
            "ts1": ordered[-1]["start_time"] if ordered else None,
            "n_skip_guard": n_skip_guard}
    return recs, meta


# ----------------------------------------------------------------------
# 3a-3. 线上数据适配（Actions CSV 优先 → API 快照 JSON 降级）
# ----------------------------------------------------------------------

FB_PATTERN_MAP = {"bull_exhaust": "S1", "bear_exhaust": "S2",
                  "momentum_fade": "S4", "bull_exhaust_confirm": "S5"}


def _parse_bool(s) -> bool | None:
    if isinstance(s, bool):
        return s
    if s is None or s == "":
        return None
    return str(s).strip().lower() in ("true", "t", "1")


def _parse_iso_ms(s) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _fb_records(rows: list[dict]) -> list[Record]:
    """fake_breakout 行 → Record（win = 结算方向 == 该场景押注方向）。"""
    out = []
    for r in rows:
        if r.get("status") != "SETTLED":
            continue
        key = FB_PATTERN_MAP.get(r.get("pattern_type"))
        if key is None:
            continue
        settle = r.get("settle_outcome")
        if settle not in ("UP", "DOWN"):
            continue
        ts = r.get("signal_time")
        ts_ms = _parse_iso_ms(ts) if isinstance(ts, str) else (int(ts) if ts else None)
        if ts_ms is None:
            continue
        settle_dir = SIGNALS[key]["settle_dir"]
        qv = r.get("entry_down_price_15m") if settle_dir == "DOWN" else None
        q = float(qv) if qv not in (None, "") else None
        evv = r.get("ev_at_entry")
        out.append(Record("online", key, ts_ms, settle_dir,
                          settle == settle_dir, q,
                          float(evv) if evv not in (None, "") else None))
    return out


def load_online_records() -> tuple[list[Record], dict]:
    """线上真实信号 → Record。优先 Actions CSV（output/timeslice_export_*/），
    回退 API 快照 JSON（output/timeslice_snapshot_*.json，200 条截断可见）。"""
    recs: list[Record] = []
    meta: dict = {"source": None, "truncated": {}}

    # ---- 优先：Actions 全量导出 CSV ----
    csv_dir = None
    for d in sorted(glob.glob(EXPORT_GLOB)):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "misalignment_signals.csv")):
            csv_dir = d
    if csv_dir:
        meta["source"] = csv_dir
        with open(os.path.join(csv_dir, "misalignment_signals.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                v = r.get("version")
                if v not in SIGNALS or r.get("status") != "SETTLED":
                    continue
                win = _parse_bool(r.get("win"))
                if win is None or not r.get("window_start"):
                    continue
                q_raw = (r.get("entry_down_price") if r.get("direction") == "DOWN"
                         else r.get("entry_up_price"))
                q = float(q_raw) if q_raw not in (None, "") else None
                evv = r.get("ev_at_entry")
                recs.append(Record("online", v, int(float(r["window_start"])),
                                   r.get("direction") or "DOWN", win, q,
                                   float(evv) if evv not in (None, "") else None))
        fb_path = os.path.join(csv_dir, "fake_breakout_signals.csv")
        if os.path.exists(fb_path):
            with open(fb_path, encoding="utf-8") as f:
                recs.extend(_fb_records(list(csv.DictReader(f))))
        return recs, meta

    # ---- 回退：API 快照 JSON（_pull_timeslice_data.py 产物）----
    snaps = sorted(glob.glob(SNAPSHOT_GLOB))
    if not snaps:
        meta["note"] = "无 Actions 导出与 API 快照 → 仅重放源"
        return recs, meta
    snap_path = snaps[-1]
    meta["source"] = snap_path
    with open(snap_path, encoding="utf-8") as f:
        snap = json.load(f)
    for v, d in (snap.get("by_version") or {}).items():
        if v not in SIGNALS:
            continue
        sigs = d.get("signals") or []
        settled_n = (d.get("stats") or {}).get("settled")
        if settled_n and len(sigs) < settled_n:
            meta["truncated"][v] = (settled_n, len(sigs))
        for r in sigs:
            if r.get("status") != "SETTLED" or r.get("win") is None:
                continue
            q_raw = (r.get("entry_down_price") if r.get("direction") == "DOWN"
                     else r.get("entry_up_price"))
            q = float(q_raw) if q_raw not in (None, "") else None
            recs.append(Record("online", v, int(r["window_start"]),
                               r.get("direction") or "DOWN", bool(r["win"]), q,
                               float(r["ev_at_entry"]) if r.get("ev_at_entry") is not None else None))
    recs.extend(_fb_records((snap.get("fake_breakout") or {}).get("signals") or []))
    meta["analytics"] = snap.get("analytics")
    return recs, meta


# ----------------------------------------------------------------------
# 3c. DOWN 基线层（720d 15m 周期：各维度格子次周期 DOWN 概率 + 平均振幅）
# ----------------------------------------------------------------------

def compute_baseline(c5: list[tuple]) -> tuple[dict, dict, float]:
    """各维度 DOWN 基线。信号几乎全押 DOWN，裸胜率无意义，须看对基线的偏离。

    Returns: (baseline[dim][bin]={"n","p_down"}, amp[dim][bin]=平均振幅%, 全局 p_down)
    结算口径与场景信号一致：次周期 15m 末价 vs 次周期开盘（锚=信号周期开盘）。
    """
    from binance_predict.backtest.data import aggregate_15m

    agg = aggregate_15m(c5)
    cycs, o15, h15, l15, c15 = agg["cycs"], agg["o15"], agg["h15"], agg["l15"], agg["c15"]
    n15 = len(cycs)
    bl: dict[str, dict[str, dict]] = {d: {} for d in DIMS}
    amp_acc: dict[str, dict[str, list]] = {d: {} for d in DIMS}
    n_total = n_down_total = 0
    for j in range(n15 - 1):
        if cycs[j + 1] != cycs[j] + 1 or o15[j + 1] <= 0:
            continue
        if c15[j + 1] == o15[j + 1] or o15[j] <= 0:
            continue
        ts_ms = cycs[j] * 900_000
        down = c15[j + 1] < o15[j + 1]
        amp = (h15[j] - l15[j]) / o15[j] * 100.0
        n_total += 1
        n_down_total += down
        for dim, (fn, _kind) in DIMS.items():
            b = fn(ts_ms)
            cell = bl[dim].setdefault(b, {"n": 0, "down": 0})
            cell["n"] += 1
            cell["down"] += down
            amp_acc[dim].setdefault(b, []).append(amp)
    baseline = {}
    amp_mean = {}
    for dim in DIMS:
        baseline[dim] = {b: {"n": v["n"], "p_down": v["down"] / v["n"]}
                         for b, v in bl[dim].items()}
        amp_mean[dim] = {b: sum(vs) / len(vs) for b, vs in amp_acc[dim].items()}
    global_p = n_down_total / n_total if n_total else 0.5
    return baseline, amp_mean, global_p


# ----------------------------------------------------------------------
# 3d. 分组统计层（三层采信：探索 → 候选 → 采信）
# ----------------------------------------------------------------------

MIN_N = 30          # 解读下限（Wilson CI 在更小样本宽达 ±15pp）
BONF_BASE_PP = 2.0  # Bonferroni 基准门槛（与 backtest_engine 一致）


def _n_prior_tests() -> int:
    """先验检验数（Bonferroni 分母）：prior 类维度格子数合计。"""
    sizes = {"session_utc": 4, "session_bjt": 4, "weekend_utc": 2, "weekend_bjt": 2}
    return max(1, sum(sizes[d] for d, (_f, k) in DIMS.items() if k == "prior"))


def group_stats(records: list[Record], baseline: dict,
                global_p: float) -> tuple[dict, dict, dict]:
    """(source × signal_key × dim × bin) 分组 → 行统计 + 三层采信裁决。

    Returns: (rows 全格子, candidates 候选, adopted 采信)
    """
    groups: dict[tuple, list[Record]] = {}
    for r in records:
        for dim, (fn, _kind) in DIMS.items():
            groups.setdefault((r.source, r.signal_key, dim, fn(r.ts_ms)), []).append(r)

    # 每信号全量基线（候选判据之一：CI 下界越过自身全量胜率）
    sig_all: dict[tuple, list] = {}
    for r in records:
        sig_all.setdefault((r.source, r.signal_key), []).append(r)
    overall = {sk: {"n": len(rs), "wr": sum(r.win for r in rs) / len(rs)}
               for sk, rs in sig_all.items()}

    mt = multiple_testing_threshold(BONF_BASE_PP, _n_prior_tests())
    required_pp = mt["required_pp"]

    rows: dict[tuple, dict] = {}
    for (src, key, dim, b), rs in groups.items():
        n = len(rs)
        k = sum(r.win for r in rs)
        wr = k / n
        lo, hi = wilson(wr, n)
        evs = [r.ev for r in rs if r.ev is not None]
        qs = [r.q for r in rs if r.q is not None]
        base_p = (baseline.get(dim) or {}).get(b, {}).get("p_down")
        # S2 押 UP：基线应取 1 − p_down（UP 基础概率）
        if SIGNALS[key]["settle_dir"] == "UP" and base_p is not None:
            base_p = 1.0 - base_p
        be = sum(breakeven_of(key, q) for q in qs) / len(qs) if qs else None
        row = {
            "source": src, "signal": key, "dim": dim, "bin": b,
            "n": n, "wins": k, "wr": round(wr, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "mde_pp": round(min_detectable_effect(n, wr) * 100, 2),
            "avg_ev": round(sum(evs) / len(evs), 4) if evs else None,
            "avg_breakeven": round(be, 4) if be else None,
            "baseline_p": round(base_p, 4) if base_p is not None else None,
            "dev_pp": round((wr - base_p) * 100, 2) if base_p is not None else None,
        }
        rows[(src, key, dim, b)] = row

    # ---- 候选：n≥30 ∧ Wilson CI 下界越过（自身全量胜率 / 该格基线 / 盈亏平衡线）----
    # 注：候选只确认「该格胜率真实存在」（越过基线/盈亏线）；时段**分化**
    # 在采信层用 vs 信号自身全量胜率的偏离（dev_own_pp）判定——否则强信号
    # （如 momentum_v1 全时段 75-80%）会因「全时段都胜过 DOWN 基线」被误报。
    candidates = {}
    for tk, row in rows.items():
        src, key, dim, _b = tk
        if row["n"] < MIN_N:
            row["verdict"] = "INSUFFICIENT"
            continue
        if SIGNALS[key].get("time_gated") and dim in ("hour_bjt", "session_bjt",
                                                      "hour_utc", "session_utc"):
            row["verdict"] = "TIME_GATED"   # 循环论证防护：不参与时段发现
            continue
        own_wr = overall[(src, key)]["wr"]
        row["dev_own_pp"] = round((row["wr"] - own_wr) * 100, 2)
        lo = row["ci_lo"]
        base_p = row["baseline_p"]
        crossed = (lo > own_wr) or (base_p is not None and lo > base_p) \
            or (row["avg_breakeven"] is not None and lo > row["avg_breakeven"])
        row["verdict"] = "CANDIDATE" if crossed else "EXPLORE"
        if crossed:
            candidates[tk] = row

    # ---- 采信：候选 ∧ 相对信号自身全量胜率的偏离 ≥Bonferroni 门槛 ∧
    #      双源复现（另一源同格同向偏离且 n≥30）----
    adopted = {}
    for tk, row in candidates.items():
        src, key, dim, b = tk
        if abs(row["dev_own_pp"]) < required_pp:
            continue
        other = "online" if src == "replay" else "replay"
        twin = rows.get((other, key, dim, b))
        dup_ok = (twin is not None and twin["n"] >= MIN_N
                  and twin.get("dev_own_pp") is not None
                  and twin["dev_own_pp"] * row["dev_own_pp"] > 0)
        row["dual_source"] = dup_ok
        if dup_ok:
            row["verdict"] = "ADOPTED"
            adopted[tk] = row
    return rows, candidates, adopted


# ----------------------------------------------------------------------
# 交叉验证：重叠期（08-13~08-19）重放 vs 线上逐笔对账（win 一致率 ≥99%
# 才采信重放；重放 07-13~08-19 与线上 08-13~now 两段部分独立）
# ----------------------------------------------------------------------

def cross_validate(replay_recs: list[Record], online_recs: list[Record]) -> dict:
    rp = {(r.signal_key, r.ts_ms): r.win for r in replay_recs}
    out: dict[str, dict] = {}
    for r in online_recs:
        w = rp.get((r.signal_key, r.ts_ms))
        if w is None:
            continue
        s = out.setdefault(r.signal_key, {"n": 0, "agree": 0})
        s["n"] += 1
        s["agree"] += int(w == r.win)
    for s in out.values():
        s["rate"] = round(s["agree"] / s["n"], 4) if s["n"] else None
    return out


# ----------------------------------------------------------------------
# 3e. 报告生成层（Tee 双写 md + JSON + 可选热力图 PNG）
# ----------------------------------------------------------------------

DIM_TITLES = {
    "hour_utc": "逐小时（UTC，探索性）", "hour_bjt": "逐小时（北京时间，探索性）",
    "session_utc": "时段 4 段（UTC，先验主判据）",
    "session_bjt": "时段 4 段（北京时间，先验主判据）",
    "dow_utc": "周内（UTC，探索性）", "dow_bjt": "周内（北京时间，探索性）",
    "weekend_utc": "周末 vs 工作日（UTC，先验）",
    "weekend_bjt": "周末 vs 工作日（北京时间，先验）",
    "dom_phase": "月内三段（探索性，仅重放源样本充足）",
    "month": "月度（探索性；每格仅 2 个年度样本）",
    "season": "季节（探索性；720d 仅 2 个年度周期，只能看不能验）",
}


def _fmt_row(row: dict) -> str:
    be = f"{row['avg_breakeven']:.1%}" if row["avg_breakeven"] is not None else "—"
    base = f"{row['baseline_p']:.1%}" if row["baseline_p"] is not None else "—"
    dev = f"{row['dev_pp']:+.1f}" if row["dev_pp"] is not None else "—"
    dev_own = f"{row['dev_own_pp']:+.1f}" if row.get("dev_own_pp") is not None else "—"
    ev = f"{row['avg_ev']:+.3f}" if row["avg_ev"] is not None else "—"
    return (f"| {row['signal']} | {row['bin']} | {row['n']} | {row['wr']:.1%} "
            f"| [{row['ci_lo']:.1%},{row['ci_hi']:.1%}] | {row['mde_pp']:.1f} "
            f"| {be} | {base} | {dev} | {dev_own} | {ev} | {row['verdict']} |")


_ROW_HDR = ("| 信号 | 格 | n | 胜率 | Wilson95% | MDEpp | 盈亏平衡 | 基线 | 偏离基线pp "
            "| vs自身全量pp | avgEV | 判定 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def _dim_table(out, rows: dict, dim: str, min_show: int = 0) -> None:
    sel = [r for r in rows.values() if r["dim"] == dim and r["n"] > min_show]
    if not sel:
        out("（无数据）\n")
        return
    sel.sort(key=lambda r: (r["signal"], r["source"], r["bin"]))
    out(_ROW_HDR)
    for r in sel:
        out(_fmt_row(r))
    out()


def _feasibility(records: list[Record], shadow_meta: dict) -> str:
    """数据可行性矩阵：信号源 × 维度 → OK / 稀疏 / N/A。"""
    cnt: dict[tuple, int] = {}
    for r in records:
        cnt[(r.signal_key, r.source)] = cnt.get((r.signal_key, r.source), 0) + 1
    lines = ["| 信号 | 源 | n | 天/周维度 | 月内/月/季维度 |",
             "|---|---|---|---|---|"]
    for key in SIGNALS:
        if SIGNALS[key].get("placeholder"):
            continue
        for src in ("replay", "online"):
            n = cnt.get((key, src), 0)
            if n == 0:
                continue
            daily = "OK" if n >= MIN_N else "稀疏(n<30/格均薄)"
            if SIGNALS[key]["family"] == "scene":
                monthly = "OK（720d 重放）" if src == "replay" else "稀疏(线上~17天)"
            else:
                span_d = (shadow_meta.get("ts1", 0) - shadow_meta.get("ts0", 0)) / 86_400_000 \
                    if src == "replay" else 17
                monthly = ("稀疏(重放~38天)" if src == "replay" else "N/A(线上~17天)") \
                    if span_d < 90 else "OK"
            lines.append(f"| {key} | {src} | {n} | {daily} | {monthly} |")
    return "\n".join(lines)


def write_report(rows, candidates, adopted, records, online_meta, shadow_meta,
                 scene_meta, baseline, amp_mean, global_p, xv, path_md: str,
                 analytics=None, required_pp: float = 0.0) -> None:
    lines: list[str] = []
    required_pp_txt = f"{required_pp:.2f}pp"

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    n_replay = sum(1 for r in records if r.source == "replay")
    n_online = len(records) - n_replay
    out("# 信号时段胜率全方位回测报告")
    out()
    out(f"生成时间：{dt.datetime.now(_tz.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
        f"记录总数：{len(records)}（重放 {n_replay} + 线上 {n_online}）  "
        f"720d DOWN 全局基线：{global_p:.1%}")
    out()
    out("**统计纪律**：n<30 标 INSUFFICIENT 不解读；候选 = n≥30 ∧ Wilson CI 下界"
        "越过（自身全量胜率 / 该格基线 / 盈亏平衡线）；采信 = 候选 ∧ 相对自身全量"
        f"胜率的偏离≥Bonferroni 门槛（{required_pp_txt}）∧ 双源复现。"
        "「偏离基线pp」对照 DOWN 基础概率（排除基线漂移），「vs自身全量pp」才是"
        "时段分化判据。late_night 族标 TIME_GATED（时段限定信号，其时段维度不参与"
        "发现，防循环论证）。场景重放无预测市场报价 → q/EV 为—。")
    out()

    # 1. 执行摘要 + 采信结论 + 可行性矩阵
    out("## 1. 执行摘要")
    out()
    if adopted:
        out(f"**采信层结论（{len(adopted)} 条）**：")
        out(_ROW_HDR)
        for tk in sorted(adopted):
            out(_fmt_row(adopted[tk]))
    else:
        out("**采信层结论：无**——没有任何格子同时满足 偏离≥Bonferroni 门槛 ∧ 双源复现。"
            "（多重比较校正后，多数「显著」时段属随机噪声；见下方候选层参考。）")
    out()
    if candidates:
        out(f"候选层（供后续积累样本，共 {len(candidates)} 条，前 20）：")
        out(_ROW_HDR)
        for tk in sorted(candidates, key=lambda t: -candidates[t]["n"])[:20]:
            out(_fmt_row(candidates[tk]))
        out()
    out("### 数据可行性矩阵")
    out()
    out(_feasibility(records, shadow_meta))
    out()

    # 2. 数据源总览与对账
    out("## 2. 数据源总览与对账")
    out()
    out("| 信号 | 源 | n | 胜率 | 参考基准 | 偏离 |")
    out("|---|---|---|---|---|---|")
    sig_all: dict[tuple, list] = {}
    for r in records:
        sig_all.setdefault((r.signal_key, r.source), []).append(r)
    for (key, src), rs in sorted(sig_all.items()):
        n = len(rs)
        wr = sum(r.win for r in rs) / n
        if SIGNALS[key]["family"] == "scene":
            b = SCENE_BENCH.get(key)
        else:
            b = SHADOW_BENCH.get(key)
        b_s = f"{b[0]:.1%}" if isinstance(b, tuple) else (f"{b:.1%}" if b else "—")
        dev = f"{(wr - (b[0] if isinstance(b, tuple) else b)) * 100:+.1f}pp" if b else "—"
        out(f"| {key} | {src} | {n} | {wr:.1%} | {b_s} | {dev} |")
    out()
    out(f"重放覆盖：场景 {scene_meta.get('n_cycles')} 个 15m 周期 / "
        f"破位事件 {scene_meta.get('n_events')}；影子 {shadow_meta.get('n_windows')} 窗，"
        f"curve_down_price 覆盖 {shadow_meta.get('curve_coverage', 0):.0%}（缺者保守跳过）。")
    out()
    out(f"线上源：{online_meta.get('source') or '无'}"
        + (f" | 截断：{online_meta['truncated']}" if online_meta.get("truncated") else ""))
    if analytics and isinstance(analytics, dict) and "error" not in analytics:
        out(f"analytics 全量聚合（对账基准）：已随快照载入，见 JSON 产物 analytics 字段。")
    out()

    # 3-8. 各维度明细（先验在前）
    for i, dim in enumerate(("session_utc", "session_bjt", "weekend_utc", "weekend_bjt",
                             "hour_utc", "hour_bjt", "dow_utc", "dow_bjt",
                             "dom_phase", "month", "season"), start=3):
        out(f"## {i}. {DIM_TITLES[dim]}")
        out()
        _dim_table(out, rows, dim)

    # DOWN 基线对照段（并入小时维度展示）
    out("## 14. DOWN 基线对照（720d 各时段次周期 DOWN 概率 + 平均振幅）")
    out()
    out("| 维度 | 格 | n | P(DOWN) | 平均振幅% |")
    out("|---|---|---|---|---|")
    for dim in ("hour_utc", "session_utc", "weekend_utc"):
        for b, v in sorted((baseline.get(dim) or {}).items()):
            a = (amp_mean.get(dim) or {}).get(b)
            a_s = f"{a:.3f}" if a is not None else "—"
            out(f"| {dim} | {b} | {v['n']} | {v['p_down']:.1%} | {a_s} |")
    out()
    out("解读：振幅 = 流动性代理。若某时段基线 P(DOWN) 显著偏离全局 "
        f"{global_p:.1%}，全押 DOWN 信号在该时段的裸胜率变化可能只是基线漂移；"
        "只有第 3-13 节中「偏离pp」列仍显著的格子才值得采信。")
    out()

    # 9. 交叉验证段
    out("## 15. 交叉验证：重叠期（08-13~08-19）重放 vs 线上逐笔对账")
    out()
    if xv:
        out("| 信号 | 重叠笔数 | win 一致 | 一致率 | 判定 |")
        out("|---|---|---|---|---|")
        for k, s in sorted(xv.items()):
            ok = "PASS" if (s["rate"] or 0) >= 0.99 else "FAIL"
            out(f"| {k} | {s['n']} | {s['agree']} | {s['rate']:.1%} | {ok} |")
        out()
        out("一致率 ≥99% 才采信重放口径；FAIL 的版本其重放结果降级为探索性参考。")
    else:
        out("无重叠样本（线上源缺失或时段不交叠）——重放结果整体降级为探索性参考。")
    out()

    # 10. 行动建议
    out("## 16. 行动建议（仅采信层结论；无采信则不行动）")
    out()
    if adopted:
        out("以下格子可考虑落地（仅说明改法，**不直接改**，留给用户决策）：")
        for tk in sorted(adopted):
            r = adopted[tk]
            out(f"- {r['source']} / {r['signal']} / {r['dim']}={r['bin']}："
                f"胜率 {r['wr']:.1%}（vs 自身全量 {r['dev_own_pp']:+.1f}pp，"
                f"vs DOWN 基线 {r['dev_pp']:+.1f}pp）。"
                f"若为正向时段 → 可考虑在 HOUR_GUARDS / LIVE_CHANNELS_JSON / 日限配置"
                f"中对该时段加权；负向 → 相反。")
    else:
        out("无满足采信标准的时段。建议：维持现有配置不动；待线上信号积累 ≥ 90 天"
            "（Actions 全量导出就绪后）重跑本脚本复核。KREV 族（08-28 上线）攒 2-3 周"
            "样本后零改动接入（注册位已留）。")
    out()
    out("---")
    out("口径冻结引用：quote_edge_detector.py:92-101/:109-112/:117-119/:128-140；"
        "fake_breakout_detector.py:90-102/:195-225/:228-249。统计内核："
        "backtest/stats.py（wilson / exact_binomial_p / min_detectable_effect / "
        "multiple_testing_threshold）。")

    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_json_out(path: str, **payload) -> None:
    def _key(t):
        return "|".join(str(x) for x in t)
    clean = {}
    for k, v in payload.items():
        if isinstance(v, dict) and any(isinstance(kk, tuple) for kk in v):
            clean[k] = {_key(kk): vv for kk, vv in v.items()}
        else:
            clean[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=1, default=str)


def try_heatmap(rows: dict, path_png: str) -> str | None:
    """可选：信号 × 小时(UTC) 胜率热力图（matplotlib 缺失时优雅跳过）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    keys = [k for k in ("x4_v1", "quote_momentum_v1", "quote_contrarian_v1",
                        "S1", "S2", "S4", "S5")
            if any(r["signal"] == k for r in rows.values())]
    hours = [f"{h:02d}" for h in range(24)]
    grid = []
    for k in keys:
        line = []
        for h in hours:
            cand = [r for r in rows.values()
                    if r["signal"] == k and r["dim"] == "hour_utc" and r["bin"] == h]
            cand.sort(key=lambda r: r["n"], reverse=True)
            line.append(cand[0]["wr"] * 100 if cand and cand[0]["n"] >= 5 else float("nan"))
        grid.append(line)
    fig, ax = plt.subplots(figsize=(14, max(3, 0.6 * len(keys))))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(24), hours)
    ax.set_yticks(range(len(keys)), keys)
    ax.set_title("Winrate by signal x hour(UTC)  (blank: n<5)")
    fig.colorbar(im, label="winrate %")
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)
    return path_png


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    stamp = _time.strftime("%Y%m%d_%H%M")
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

    print("[1/7] 载入 720d 5m K 线...")
    c5 = load_klines_5m()
    print(f"      bars={len(c5)}  "
          f"{dt.datetime.fromtimestamp(c5[0][0] / 1000, _tz.utc):%Y-%m-%d} → "
          f"{dt.datetime.fromtimestamp(c5[-1][0] / 1000, _tz.utc):%Y-%m-%d}")

    print("[2/7] 场景重放（S1/S2/S4/S5）...")
    scene_recs, scene_meta = replay_scene_events(c5)
    print(f"      场景记录={len(scene_recs)} | meta={scene_meta}")

    print("[3/7] DOWN 基线（720d × 11 维度）...")
    baseline, amp_mean, global_p = compute_baseline(c5)
    print(f"      全局 P(DOWN)={global_p:.1%}")

    print("[4/7] 影子重放（10 版本）...")
    shadow_recs, shadow_meta = replay_shadow_signals()
    print(f"      影子记录={len(shadow_recs)} | 窗={shadow_meta['n_windows']} "
          f"曲线覆盖={shadow_meta['curve_coverage']:.0%}")

    print("[5/7] 线上数据（CSV 优先 → 快照降级）...")
    online_recs, online_meta = load_online_records()
    print(f"      线上记录={len(online_recs)} | 源={online_meta.get('source')}")

    records = scene_recs + shadow_recs + online_recs
    print("[6/7] 分组统计（三层采信）...")
    rows, candidates, adopted = group_stats(records, baseline, global_p)
    xv = cross_validate(scene_recs + shadow_recs, online_recs)
    print(f"      格子={len(rows)} 候选={len(candidates)} 采信={len(adopted)}")

    print("[7/7] 生成报告...")
    md_path = os.path.join(ROOT, "output", f"timeslice_winrate_report_{stamp}.md")
    json_path = os.path.join(ROOT, "output", f"timeslice_winrate_report_{stamp}.json")
    write_report(rows, candidates, adopted, records, online_meta, shadow_meta,
                 scene_meta, baseline, amp_mean, global_p, xv, md_path,
                 analytics=online_meta.get("analytics"),
                 required_pp=multiple_testing_threshold(BONF_BASE_PP, _n_prior_tests())["required_pp"])
    write_json_out(json_path, rows=rows, candidates=candidates, adopted=adopted,
                   baseline=baseline, amp_mean=amp_mean, global_p=global_p,
                   cross_validation=xv, scene_meta=scene_meta,
                   shadow_meta=shadow_meta, online_meta=online_meta)
    png = try_heatmap(rows, os.path.join(ROOT, "output",
                                         f"timeslice_heatmap_{stamp}.png"))
    print(f"\n报告 → {md_path}")
    print(f"JSON → {json_path}")
    print("热力图 → " + (png or "跳过（matplotlib 未安装，不影响结论）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
