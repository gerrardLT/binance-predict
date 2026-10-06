"""信号体检：通道级健康指标与红黄绿灯（纯函数，无 I/O）。

影子与实盘共用同一套指标口径。输入是「事件」列表（升序）：
    {"ts": 毫秒, "win": bool, "ev": 单位本金收益|None, "be": 盈亏平衡概率|None, "q": 入场价|None}

核心判据：
- p_edge_negative：以 Beta(1+赢, 1+输) 为真实胜率的后验，P(真实胜率 < 平均盈亏平衡)。
  贝叶斯后验可随每笔新样本持续更新而不受「反复偷看」影响，适合持续监测。
- p_below_bench：若真实胜率等于冻结基准，观测到不多于当前赢单数的概率（单侧二项）。
- ev_ex_top5：剔除最赚 5 单后的平均收益，用于识别「右尾彩票」。
红灯阈值按用户确认：p_edge_negative ≥ 0.90 且 n ≥ 30。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from scipy.stats import beta as _beta
from scipy.stats import binom as _binom

from binance_predict.backtest.stats import wilson

MIN_N = 30            # 低于此样本不下结论（灰灯）
RED_P_NEG = 0.90      # 红灯：优势已变负的后验概率
YELLOW_P_NEG = 0.75   # 黄灯：优势疑似变负
YELLOW_P_BENCH = 0.10 # 黄灯：前向显著低于冻结基准
TAIL_SHARE = 0.80     # 黄灯：前 5 单贡献 ≥80% 的正收益
MIN_BUCKET_N = 10     # 分桶样本不足标记线

Event = dict[str, Any]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(events: Sequence[Event], bench_win: float | None = None) -> dict[str, Any]:
    """通道级体检指标。events 须按时间升序。"""
    n = len(events)
    wins = sum(1 for e in events if e["win"])
    win_rate = wins / n if n else None
    evs = [float(e["ev"]) for e in events if e.get("ev") is not None]
    bes = [float(e["be"]) for e in events if e.get("be") is not None]
    mean_be = _mean(bes)

    p_neg = None
    if n and mean_be is not None:
        p_neg = float(_beta.cdf(mean_be, 1 + wins, 1 + n - wins))

    p_bench = None
    if n and bench_win is not None:
        p_bench = float(_binom.cdf(wins, n, bench_win))

    ranked = sorted(evs, reverse=True)
    ex_top = {k: (_mean(ranked[k:]) if len(ranked) > k else None) for k in (1, 3, 5)}
    pos_sum = sum(v for v in ranked if v > 0)
    top5_share = (sum(v for v in ranked[:5] if v > 0) / pos_sum) if pos_sum > 0 else None

    def _window(seq: Sequence[Event]) -> dict[str, Any]:
        w = sum(1 for e in seq if e["win"])
        e_vals = [float(e["ev"]) for e in seq if e.get("ev") is not None]
        return {"n": len(seq), "win_rate": w / len(seq) if seq else None, "avg_ev": _mean(e_vals)}

    lo, hi = wilson(win_rate, n) if n else (None, None)
    return {
        "n": n, "wins": wins, "win_rate": win_rate,
        "win_rate_ci95": [lo, hi] if n else None,
        "ev_n": len(evs), "avg_ev": _mean(evs),
        "mean_breakeven": mean_be, "be_n": len(bes),
        "edge": (win_rate - mean_be) if (win_rate is not None and mean_be is not None) else None,
        "p_edge_negative": p_neg,
        "bench_win_rate": bench_win, "p_below_bench": p_bench,
        "ev_ex_top1": ex_top[1], "ev_ex_top3": ex_top[3], "ev_ex_top5": ex_top[5],
        "top5_share": top5_share,
        "recent20": _window(events[-20:]) if n >= 20 else None,
        "prev20": _window(events[-40:-20]) if n >= 40 else None,
        "last_ts": events[-1]["ts"] if n else None,
    }


def classify(m: dict[str, Any]) -> tuple[str, list[str]]:
    """红黄绿灯。返回 (light, 原因列表)；light ∈ GRAY | GREEN | YELLOW | RED。"""
    if m["n"] < MIN_N:
        return "GRAY", [f"样本 {m['n']}/{MIN_N}，暂不下结论"]
    p_neg = m["p_edge_negative"]
    if p_neg is not None and p_neg >= RED_P_NEG:
        return "RED", [f"优势已变负的概率 {p_neg:.0%}（胜率 {m['win_rate']:.1%} vs 保本 {m['mean_breakeven']:.1%}）"]
    reasons: list[str] = []
    if p_neg is not None and p_neg >= YELLOW_P_NEG:
        reasons.append(f"优势疑似变负（概率 {p_neg:.0%}）")
    pb = m["p_below_bench"]
    if pb is not None and pb < YELLOW_P_BENCH:
        reasons.append(f"前向胜率 {m['win_rate']:.1%} 显著低于冻结基准 {m['bench_win_rate']:.1%}（p={pb:.3f}）")
    if m["avg_ev"] is not None and m["avg_ev"] > 0 and m["ev_ex_top5"] is not None and m["ev_ex_top5"] <= 0:
        reasons.append(f"剔除最赚 5 单后 EV {m['ev_ex_top5']:+.3f}≤0：收益靠右尾")
    if m["top5_share"] is not None and m["top5_share"] >= TAIL_SHARE:
        reasons.append(f"前 5 单贡献 {m['top5_share']:.0%} 的正收益")
    return ("YELLOW" if reasons else "GREEN"), reasons


def _bucket(events: Sequence[Event], key_fn: Callable[[Event], str], order: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[Event]] = {}
    for e in events:
        groups.setdefault(key_fn(e), []).append(e)
    rows = []
    for label in sorted(groups, key=lambda s: order.index(s) if s in order else len(order)):
        g = groups[label]
        n, w = len(g), sum(1 for e in g if e["win"])
        evs = [float(e["ev"]) for e in g if e.get("ev") is not None]
        lo, hi = wilson(w / n, n)
        rows.append({"segment": label, "n": n, "win_rate": w / n, "win_rate_ci95": [lo, hi],
                     "avg_ev": _mean(evs), "small_sample": n < MIN_BUCKET_N})
    return rows


_PRICE_LABELS = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)] + ["未知"]
_HOUR_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
_TREND_4H_LABELS = ["急跌(<-1%)", "偏弱(-1%~0)", "偏强(0~+1%)", "大涨(>+1%)", "未知"]
_TREND_24H_LABELS = ["暴跌(<-2%)", "下行(-2%~0)", "上行(0~+2%)", "过热(>+2%)", "未知"]
_VOL_LABELS = ["低波动", "中波动", "高波动", "未知"]


def price_buckets(events: Sequence[Event]) -> list[dict[str, Any]]:
    """按入场价分桶（0.1 一档）。探索性：不可据此直接改护栏。"""
    def key(e: Event) -> str:
        q = e.get("q")
        return "未知" if q is None or q <= 0 else _PRICE_LABELS[min(9, int(q * 10))]
    return _bucket(events, key, _PRICE_LABELS)


def hour_buckets(events: Sequence[Event]) -> list[dict[str, Any]]:
    """按 UTC 时段分桶（4 小时一档，以窗口起点计）。探索性。"""
    return _bucket(events, lambda e: _HOUR_LABELS[int(e["ts"] // 3_600_000 % 24) // 4], _HOUR_LABELS)


def trend_4h_buckets(events: Sequence[Event]) -> list[dict[str, Any]]:
    """按 4h BTC 趋势分桶。"""
    def key(e: Event) -> str:
        t = e.get("ret_4h")
        if t is None: return "未知"
        pct = t * 100.0
        if pct < -1.0: return "急跌(<-1%)"
        if pct <= 0.0: return "偏弱(-1%~0)"
        if pct <= 1.0: return "偏强(0~+1%)"
        return "大涨(>+1%)"
    return _bucket(events, key, _TREND_4H_LABELS)


def trend_24h_buckets(events: Sequence[Event]) -> list[dict[str, Any]]:
    """按 24h BTC 趋势分桶。"""
    def key(e: Event) -> str:
        t = e.get("ret_24h")
        if t is None: return "未知"
        pct = t * 100.0
        if pct < -2.0: return "暴跌(<-2%)"
        if pct <= 0.0: return "下行(-2%~0)"
        if pct <= 2.0: return "上行(0~+2%)"
        return "过热(>+2%)"
    return _bucket(events, key, _TREND_24H_LABELS)


def volatility_buckets(events: Sequence[Event]) -> list[dict[str, Any]]:
    """按 4h 波动率分桶（基于 5m 收益标准差）。"""
    def key(e: Event) -> str:
        v = e.get("vol_4h")
        if v is None: return "未知"
        pct = v * 100.0
        if pct < 0.15: return "低波动"
        if pct <= 0.30: return "中波动"
        return "高波动"
    return _bucket(events, key, _VOL_LABELS)


# ---------------- P3：死因诊断、变化点检测、相关性分析 ----------------

def diagnose_death_status(
    events: Sequence[Event],
    bench_win: float | None = None,
    min_hits: int = 30,
) -> tuple[str, str]:
    """模式死因判定（P3 对接 verification.diagnose_death 思想）。
    
    返回 (status, reason)：
    - ALIVE: 存活良好或样本不足
    - EXPIRED: 曾显著盈利但已衰退（regime 变迁）
    - SPURIOUS: 从未显著盈利过（过拟合假规律）
    """
    n = len(events)
    if n < min_hits:
        return "ALIVE", f"样本不足 {min_hits} 笔，无法判定死因"
    
    # 历史最佳滚动 20 胜率
    peak_wr = 0.0
    for i in range(20, n + 1):
        w = sum(1 for e in events[i - 20:i] if e["win"]) / 20.0
        if w > peak_wr:
            peak_wr = w

    # 最近 20 笔表现
    recent_events = events[-20:]
    recent_wins = sum(1 for e in recent_events if e["win"])
    recent_wr = recent_wins / 20.0
    bes = [float(e["be"]) for e in events if e.get("be") is not None]
    mean_be = _mean(bes) or 0.51
    recent_lo, recent_hi = wilson(recent_wr, 20)

    # 1. 过期规律 (EXPIRED): 曾经很高（胜率≥保本+8% 或 ≥基准+5%），但近期跌破保本
    had_glory = peak_wr >= (mean_be + 0.08) or (bench_win is not None and peak_wr >= bench_win + 0.05)
    now_failing = recent_wr < mean_be and recent_hi < (mean_be + 0.05)
    if had_glory and now_failing:
        return "EXPIRED", f"曾峰值胜率 {peak_wr:.1%} 显著盈利，近期近20胜率跌至 {recent_wr:.1%} 出现衰减"

    # 2. 伪规律 (SPURIOUS): 峰值从未突破（甚至没超过保本+2%），且当前 CI 覆盖或低于保本
    never_worked = peak_wr < (mean_be + 0.02)
    overall_wins = sum(1 for e in events if e["win"])
    all_lo, _ = wilson(overall_wins / n, n)
    if never_worked and all_lo < mean_be:
        return "SPURIOUS", f"全生命周期峰值胜率仅 {peak_wr:.1%}，从未建立超额优势，疑似过拟合"

    return "ALIVE", "未触发死因衰退标准"


def detect_change_point(events: Sequence[Event]) -> dict[str, Any] | None:
    """变化点检测（P3 CUSUM 极值拐折点）：找出累计 EV 发生转折的时刻。"""
    n = len(events)
    if n < 15:
        return None

    cum = 0.0
    peak_ev = -9999.0
    peak_idx = -1
    peak_ts = None

    cum_series = []
    for i, e in enumerate(events):
        ev = float(e["ev"]) if e.get("ev") is not None else 0.0
        cum += ev
        cum_series.append(cum)
        if cum > peak_ev:
            peak_ev = cum
            peak_idx = i
            peak_ts = e["ts"]

    # 从峰值到终点的回撤幅度
    final_ev = cum_series[-1]
    drawdown = peak_ev - final_ev

    # 如果峰值后至少有 10 单，且回撤幅度较明显（> 1.0 单位 EV）
    is_declining = (n - 1 - peak_idx) >= 10 and drawdown >= 1.0

    return {
        "change_ts": peak_ts if is_declining else None,
        "peak_cum_ev": round(peak_ev, 3),
        "current_cum_ev": round(final_ev, 3),
        "drawdown_ev": round(drawdown, 3),
        "post_peak_n": n - 1 - peak_idx,
        "is_declining": is_declining,
    }


def compute_channel_correlations(
    events_by_key: Mapping[str, Sequence[Event]],
    min_shared: int = 5,
) -> list[dict[str, Any]]:
    """计算各通道间同窗共振与重合度（P3）。
    
    返回按重叠度从大到小排序的通道对关系。
    """
    keys = sorted(events_by_key.keys())
    # 提取每个通道的 windows 集合
    ts_map: dict[str, dict[int, bool]] = {}
    for k in keys:
        ts_map[k] = {e["ts"]: bool(e["win"]) for e in events_by_key[k]}

    results = []
    for i in range(len(keys)):
        k1 = keys[i]
        set1 = ts_map[k1]
        n1 = len(set1)
        if n1 == 0: continue
        for j in range(i + 1, len(keys)):
            k2 = keys[j]
            set2 = ts_map[k2]
            n2 = len(set2)
            if n2 == 0: continue

            shared_ts = set(set1.keys()) & set(set2.keys())
            shared_n = len(shared_ts)
            if shared_n < min_shared:
                continue

            # Jaccard 相似度
            union_n = len(set(set1.keys()) | set(set2.keys()))
            jaccard = shared_n / union_n if union_n else 0.0

            # 同向一致率（同窗时胜负一致性）
            agree = sum(1 for t in shared_ts if set1[t] == set2[t])
            agree_rate = agree / shared_n if shared_n else 0.0

            results.append({
                "channel_a": k1,
                "channel_b": k2,
                "shared_windows": shared_n,
                "jaccard": round(jaccard, 3),
                "agreement_rate": round(agree_rate, 3),
            })

    results.sort(key=lambda r: (r["jaccard"], r["shared_windows"]), reverse=True)
    return results


def series(events: Sequence[Event], max_points: int = 200) -> list[dict[str, Any]]:
    """累计 EV / 滚动 20 胜率 / 优势变负概率的时间序列（等间隔抽样至 max_points）。"""
    n = len(events)
    if n == 0:
        return []
    stride = max(1, n // max_points)
    idx = sorted(set(list(range(stride - 1, n, stride)) + [n - 1]))
    out, cum, wins, be_sum, be_n, j = [], 0.0, 0, 0.0, 0, 0
    for i, e in enumerate(events):
        cum += float(e["ev"]) if e.get("ev") is not None else 0.0
        wins += int(bool(e["win"]))
        if e.get("be") is not None:
            be_sum += float(e["be"])
            be_n += 1
        if j < len(idx) and i == idx[j]:
            j += 1
            tail = events[max(0, i - 19): i + 1]
            out.append({
                "ts": e["ts"], "i": i + 1, "cum_ev": round(cum, 4),
                "roll20_win": (sum(1 for x in tail if x["win"]) / len(tail)) if len(tail) >= 20 else None,
                "p_neg": float(_beta.cdf(be_sum / be_n, 1 + wins, 1 + i + 1 - wins)) if be_n else None,
            })
    return out


def build_diagnosis_payload(channel: str, source: str, light: str, reasons: list[str],
                            metrics: dict[str, Any], price: list, hour: list,
                            death_status: str | None = None,
                            change_point: dict[str, Any] | None = None,
                            trend_4h: list | None = None,
                            trend_24h: list | None = None,
                            volatility: list | None = None) -> dict[str, Any]:
    """LLM 诊断预留口：把体检结论打包为结构化输入（未接入模型，仅定义契约）。"""
    return {
        "schema": "signal_health_diagnosis_v1",
        "channel": channel, "source": source, "light": light, "reasons": reasons,
        "death_status": death_status or "ALIVE",
        "change_point": change_point,
        "metrics": metrics, "price_buckets": price, "hour_buckets": hour,
        "trend_4h_buckets": trend_4h or [],
        "trend_24h_buckets": trend_24h or [],
        "volatility_buckets": volatility or [],
        "guardrails": "只提出待验证假设，不得建议直接改实盘护栏；任何改动须走预注册与前向验证。",
    }


_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}


def new_red_transitions(prev: dict[tuple[str, str], str], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """本轮新变红的通道（prev 中已有记录且此前非红）。首轮/新通道/灰灯不报，避免冷启动刷屏。"""
    out = []
    for r in rows:
        old = prev.get((r["scope"], r["key"]))
        if r["light"] == "RED" and old in _RANK and _RANK[old] < _RANK["RED"]:
            out.append(r)
    return out


def format_red_alert(row: dict[str, Any], display_name: str) -> str:
    m = row["metrics"]
    src = "实盘" if row["scope"] == "live" else "影子"
    return (
        f"### ⚠️ 信号体检转红（{src}）\n"
        f"**{display_name}**（{row['key']}）\n"
        f"- {'；'.join(row['reasons'])}\n"
        f"- 样本 {m['n']} 笔 · 胜率 {m['win_rate']:.1%} · 平均 EV "
        f"{'--' if m['avg_ev'] is None else format(m['avg_ev'], '+.3f')}\n"
        f"- 仅为建议：请到「运行监控」查看下钻，是否关闭由你决定。"
    )
