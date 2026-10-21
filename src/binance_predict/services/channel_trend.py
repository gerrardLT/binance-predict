"""通道 EV 趋势在线报警器（纯函数，无 IO）。

口径（2026-10-22 业务评估拍板）：
- 主触发不看「滚动胜率环比」（n=20 时两窗口胜率差 SE≈16pp，误报极高），
  而是对单位本金 EV 序列做在线单侧 CUSUM（μ_ref=0）：恶化 = 已不再赚钱
  并被统计确认；相对衰减（+0.10→+0.03 仍为正）只作面板展示、不进报警。
- σ 用 1.4826×MAD（抗单笔极端赢单撑大方差）；CUSUM 当前段归零即视为
  恢复（segment 清空），避免一次旧衰减长期挂灯。
- 三级：L1 观察（面板）/ L2 预警（微信 + 自动 LLM）/ L3 确认
  （p_edge_negative≥0.90 且 n≥30，走既有红灯链路，本模块只标注不推送）。
- 触发按「结算单数」计，不按小时；保本线覆盖率不足时不出结论
  （缺入场价会让 edge 静默退化成纯胜率，回到最差口径）。

⚠️ EV 单笔噪声极大（预测市场输单接近 −100%），本报警器实际捕捉的是
「灾难级失效」（连续大亏），小幅衰减由 p_edge_negative 后验缓慢确认。
"""
from __future__ import annotations

import statistics
from typing import Any, Sequence

from binance_predict.services import signal_health as sh

TREND_MIN_N = 40          # 趋势结论最低已结算样本
CUSUM_K = 0.5             # CUSUM 参考值偏移（σ 倍数）
CUSUM_H = 5.0             # CUSUM 报警阈值（σ 倍数）
CUSUM_SIGMA_FLOOR = 0.01  # σ 下限（全相同 EV 的退化序列防除零）
L1_CONFIRM_N = 10         # L1 需变化点后最少单数
L2_CONFIRM_N = 15         # L2 需变化点后最少单数
COVERAGE_FLOOR = 0.70     # 保本线覆盖率下限（低于则趋势结论禁用）

LEVEL_RANK = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

Event = dict[str, Any]


def mad_sigma(values: Sequence[float]) -> float | None:
    """中位数绝对偏差 → 稳健标准差估计（1.4826×MAD）；少于 2 个值 → None。"""
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs])
    return 1.4826 * mad


def ev_cusum_alarm(events: Sequence[Event], k: float = CUSUM_K,
                   h: float = CUSUM_H, min_n: int = TREND_MIN_N) -> dict:
    """单位 EV 的单侧（下行）CUSUM 报警。

    S_i = max(0, S_{i-1} − ev_i − kσ)，μ_ref=0。S 越过 hσ 视为本轮恶化
    起点（segment_start 只记第一次越界）；S 回落到 0 视为恢复、清空段。
    返回最近一轮的越界信息与「变化点后已观察单数」since_alarm_n。
    """
    vals = [float(e["ev"]) for e in events if e.get("ev") is not None]
    n = len(vals)
    out: dict[str, Any] = {"alarm": False, "n": n, "cusum": 0.0, "sigma": None,
                           "alarm_at_ts": None, "since_alarm_n": None}
    if n < min_n:
        return out
    sigma = max(mad_sigma(vals) or 0.0, CUSUM_SIGMA_FLOOR)
    limit = h * sigma
    s = 0.0
    segment_start: int | None = None
    for i, v in enumerate(vals):
        s = max(0.0, s - v - k * sigma)
        if s == 0.0:
            segment_start = None
        elif s > limit and segment_start is None:
            segment_start = i
    out["sigma"] = round(sigma, 6)
    out["cusum"] = round(s, 4)
    if segment_start is not None and s > limit:
        ts_seq = [e.get("ts") for e in events if e.get("ev") is not None]
        out["alarm"] = True
        out["alarm_at_ts"] = ts_seq[segment_start]
        out["since_alarm_n"] = n - 1 - segment_start
    return out


def classify_trend(events: Sequence[Event], coverage_ratio: float = 1.0,
                   metrics: dict | None = None) -> tuple[str, list[str]]:
    """趋势分级（L0=无结论 | L1 观察 | L2 预警 | L3 确认）。

    L3 与既有体检红灯同口径（p_edge_negative≥0.90 且 n≥30），只做标注，
    推送仍由 signal_health_loop 的红灯链路负责，避免重复告警。
    """
    if coverage_ratio < COVERAGE_FLOOR:
        return "L0", [f"保本线覆盖率 {coverage_ratio:.0%} < {COVERAGE_FLOOR:.0%}，趋势结论禁用"]
    m = metrics if metrics is not None else sh.compute_metrics(events)
    n = int(m.get("n") or 0)
    if n < TREND_MIN_N:
        return "L0", []
    p_neg = m.get("p_edge_negative")
    if p_neg is not None and p_neg >= sh.RED_P_NEG:
        return "L3", [f"优势已变负的概率 {p_neg:.0%}（胜率 {m['win_rate']:.1%} vs 保本 "
                      f"{(m.get('mean_breakeven') or 0):.1%}，n={n}）"]
    alarm = ev_cusum_alarm(events)
    if not (alarm["alarm"] and (alarm["since_alarm_n"] or 0) >= L1_CONFIRM_N):
        return "L0", []
    reasons = [f"EV CUSUM 越界（{alarm['cusum']:.2f}σ），变化点后已观察 {alarm['since_alarm_n']} 单"]
    if (alarm["since_alarm_n"] or 0) < L2_CONFIRM_N:
        return "L1", reasons
    # L2 确认门：后验已偏负，或变化后窗口的胜率-保本边际 ≤0
    recent_after = events[-(alarm["since_alarm_n"] + 1):]
    after = sh.calibration_metrics(recent_after)
    after_edge = after.get("edge")
    if (p_neg is not None and p_neg >= sh.YELLOW_P_NEG) or (
            after_edge is not None and after_edge <= 0):
        if p_neg is not None and p_neg >= sh.YELLOW_P_NEG:
            reasons.append(f"优势疑似变负（后验概率 {p_neg:.0%}）")
        if after_edge is not None and after_edge <= 0:
            reasons.append(f"变化后窗口边际 {after_edge:+.3f}≤0（胜率 {after['win_rate']:.1%} vs "
                           f"保本 {(after.get('mean_breakeven') or 0):.1%}）")
        return "L2", reasons
    reasons.append("后验与变化后边际均未确认，维持观察")
    return "L1", reasons


def cusum_series(events: Sequence[Event], k: float = CUSUM_K,
                 h: float = CUSUM_H) -> list[dict]:
    """逐单 CUSUM 曲线（下钻图表用；与 ev_cusum_alarm 同口径）。"""
    rows = [(e.get("ts"), float(e["ev"])) for e in events if e.get("ev") is not None]
    n = len(rows)
    if n < 2:
        return []
    sigma = max(mad_sigma([v for _, v in rows]) or 0.0, CUSUM_SIGMA_FLOOR)
    limit = round(h * sigma, 4)
    s = 0.0
    out: list[dict] = []
    for i, (ts, v) in enumerate(rows):
        s = max(0.0, s - v - k * sigma)
        out.append({"ts": ts, "i": i + 1, "cusum": round(s, 4), "threshold": limit})
    return out


def trend_payload(channel: str, events: Sequence[Event],
                  coverage_ratio: float = 1.0,
                  metrics: dict | None = None) -> dict:
    """单通道轻量趋势载荷（体检报告 trend 段 / advisor 端点共用）。"""
    m = metrics if metrics is not None else sh.compute_metrics(events)
    level, reasons = classify_trend(events, coverage_ratio, m)
    alarm = ev_cusum_alarm(events)
    recent20 = m.get("recent20") or {}
    return {
        "channel": channel,
        "level": level,
        "reasons": reasons,
        "n": m.get("n", 0),
        "win_rate": m.get("win_rate"),
        "p_edge_negative": m.get("p_edge_negative"),
        "mean_breakeven": m.get("mean_breakeven"),
        "edge": m.get("edge"),
        "rolling20_win_rate": recent20.get("win_rate"),
        "cusum": alarm.get("cusum"),
        "cusum_sigma": alarm.get("sigma"),
        "cusum_alarm": alarm.get("alarm"),
        "since_alarm_n": alarm.get("since_alarm_n"),
        "alarm_at_ts": alarm.get("alarm_at_ts"),
        "coverage_ratio": round(float(coverage_ratio), 3),
        "last_ts": m.get("last_ts"),
        "level_rank": LEVEL_RANK.get(level, 0),
    }


def format_trend_alert(payload: dict, display_name: str,
                       advice_hint: str | None = None) -> str:
    """L2 趋势预警微信文案（与既有红灯文案风格一致）。"""
    wr = payload.get("win_rate")
    be = payload.get("mean_breakeven")
    roll = payload.get("rolling20_win_rate")
    lines = [
        "### ⚠️ 信号趋势预警（L2）",
        f"**{display_name}**（{payload.get('channel')}）",
        f"- {'；'.join(payload.get('reasons') or ['EV 趋势恶化'])}",
        f"- 样本 {payload.get('n', 0)} 笔 · 胜率 "
        f"{'--' if wr is None else f'{wr:.1%}'} · 保本 "
        f"{'--' if be is None else f'{be:.1%}'} · 近20单胜率 "
        f"{'--' if roll is None else f'{roll:.1%}'}",
    ]
    if advice_hint:
        lines.append(f"- AI 策略建议：{advice_hint}")
    lines.append("- 仅为建议：请到「实盘交易 → 信号实盘通道」查看趋势与 AI 策略建议，"
                 "是否调整由你决定。")
    return "\n".join(lines)
