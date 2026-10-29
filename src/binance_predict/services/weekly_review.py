"""LLM 每周复盘：确定性事实卡 + 10 条注意点规则 + 报告编排。

设计（docs/WEEKLY_REVIEW_PLAN.md）：

- **事实卡先行**：所有数字由确定性代码算（复用 attribution_engine 的
  compute_channel_facts / attribution_stats / signal_health 口径），LLM 只是
  解释者，永远是「建议」不是「执行」。
- **注意点是规则**：10 条 watchout 由代码判（阈值写在下方常量），LLM 可以
  补充但必须过 claim 闸，且不能降级规则产出的级别。
- **只读分析层**：本模块不写 trade_orders，不进下单/结算路径；唯一写库是
  weekly_reviews（报告归档）与经 apply 端点的通道参数（人工触发）。

失败语义：任何 LLM / 数据异常都不影响交易；报告降级为「纯事实卡 + 注意点」。
"""
from __future__ import annotations

import datetime as dt
import math
import time
from typing import Any, Awaitable, Callable, Sequence

from loguru import logger
from sqlalchemy import and_ as sa_and
from sqlalchemy import or_ as sa_or
from sqlalchemy import select as sa_select

from binance_predict.config.settings import settings
from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import TradeOrderModel, WeeklyReview

from . import attribution_stats as astats
from . import signal_health as sh
from .attribution_engine import _orders_to_events, compute_channel_facts
from .live_channel_benchmarks import get_channel_benchmark
from .live_performance import is_price_guard_rejection

SCHEMA_VERSION = "weekly_review_facts_v1"
ENGINE_VERSION = "weekly_review_v1"

# ---- 注意点阈值（改这里 = 改口径，必须同步 docs/WEEKLY_REVIEW_PLAN.md §3.2）----
CONCENTRATION_TOP1 = 0.30
CONCENTRATION_TOP3 = 0.65
SMALL_SAMPLE_N = 30
BENCHMARK_GAP = 0.10
GUARD_NEAR_BREAKEVEN = 0.98
REJECT_RATE_HIGH = 0.50
UNSETTLED_BACKLOG_MIN = 30
EQUITY_DRIFT_RATIO = 0.20
EDGE_NEGATIVE_P = sh.RED_P_NEG
NEW_CHANNEL_DAYS = 60

SEVERITY_RANK = {"P0": 3, "P1": 2, "P2": 1}

EquityProvider = Callable[[], Awaitable[float | None]]


# ======================================================================
# 时间窗口
# ======================================================================

def week_window(now_ms: int | None = None, offset_weeks: int = -1) -> dict:
    """ISO 周窗口（周一 00:00 UTC 为界，7 天整）。

    offset_weeks=-1 → 上一个完整周（周一定时任务的默认目标）；0 → 本周至今
    （人工触发「看看现在」用，但 end 仍取周界，数据会偏少）。
    """
    now = dt.datetime.fromtimestamp((now_ms or int(time.time() * 1000)) / 1000,
                                    dt.timezone.utc)
    monday = (now - dt.timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    start = monday + dt.timedelta(weeks=offset_weeks)
    end = start + dt.timedelta(days=7)
    prev_start = start - dt.timedelta(days=7)
    iso = start.isocalendar()
    return {
        "key": f"{iso.year}-W{iso.week:02d}",
        "start_ms": int(start.timestamp() * 1000),
        "end_ms": int(end.timestamp() * 1000),
        "prev_start_ms": int(prev_start.timestamp() * 1000),
        "prev_end_ms": int(start.timestamp() * 1000),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }


# ======================================================================
# 数据加载（单条 SQL，窗口内全部状态）
# ======================================================================

async def _load_orders(db, start_ms: int, end_ms: int,
                       channels: Sequence[str]) -> list[dict]:
    """目标窗口落在 [start,end) 内的全部订单行（FILLED / FAILED / PENDING）。

    时间口径 = `window_start`（订单对应的市场窗口），与 `/api/live/performance-summary`
    和体检面板同口径；`window_start` 缺失的历史行回落 `created_at`。
    用 `created_at` 分桶会把「周日深夜下单、窗口跨到周一」的单归错周，
    并会让周报成本与性能面板无法对账（2026-10-30 审核 P1）。

    显式取列构造 dict（SQLAlchemy 2.0 ORM 对象不可 dict() 展开，口径同
    attribution_engine._load_channel_orders）。
    """
    start = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc)
    end = dt.datetime.fromtimestamp(end_ms / 1000, dt.timezone.utc)
    rows = (await db.execute(sa_select(
        TradeOrderModel.id, TradeOrderModel.signal_version, TradeOrderModel.direction,
        TradeOrderModel.win, TradeOrderModel.pnl, TradeOrderModel.amount_in,
        TradeOrderModel.created_at, TradeOrderModel.quote_json,
        TradeOrderModel.status, TradeOrderModel.window_start,
        TradeOrderModel.market_period, TradeOrderModel.settled_at,
        TradeOrderModel.error_message,
    ).where(
        TradeOrderModel.signal_version.in_(list(channels)),
        sa_or(
            sa_and(TradeOrderModel.window_start >= start_ms,
                   TradeOrderModel.window_start < end_ms),
            sa_and(TradeOrderModel.window_start.is_(None),
                   TradeOrderModel.created_at >= start,
                   TradeOrderModel.created_at < end),
        ),
    ).order_by(TradeOrderModel.window_start.asc(),
               TradeOrderModel.id.asc()))).all()

    def _pf(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    out: list[dict] = []
    for r in rows:
        q = r.quote_json or {}
        try:
            amount = float(r.amount_in or 0) / 1e18
        except (TypeError, ValueError):
            amount = 0.0
        out.append({
            "id": r.id, "channel": r.signal_version, "direction": r.direction,
            "win": r.win, "pnl": r.pnl, "amount_usdt": amount,
            "created_ts": int(r.created_at.timestamp() * 1000) if r.created_at else 0,
            "avg_price": _pf(q.get("averagePrice")),
            "quoted_avg_price": _pf(q.get("quotedAvgPrice")),
            "status": r.status, "window_start": r.window_start,
            "market_period": r.market_period,
            "settled": r.settled_at is not None,
            "settled_ts": int(r.settled_at.timestamp() * 1000) if r.settled_at else None,
            "error_message": r.error_message,
        })
    return out


def _reject_stats(rows: Sequence[dict]) -> dict:
    """弃单分类：护栏/白名单（is_price_guard_rejection）vs 其它失败。"""
    failed = [r for r in rows if r.get("status") == "FAILED"]
    guard = sum(1 for r in failed if is_price_guard_rejection(r.get("error_message")))
    return {"total": len(failed), "guard": guard, "other": len(failed) - guard}


# ======================================================================
# 事实卡
# ======================================================================

def _portfolio_metrics(rows: Sequence[dict]) -> dict:
    """组合级指标（口径与 /api/live/performance-summary 对齐）。"""
    filled = [r for r in rows if r.get("status") == "FILLED"]
    settled = [r for r in filled if r.get("win") is not None]
    events = _orders_to_events(settled)
    m = sh.compute_metrics(events) if events else {"n": 0}
    evs = [e["ev"] for e in events if e.get("ev") is not None]
    ts = [e["ts"] for e in events if e.get("ev") is not None]
    boot = astats.block_bootstrap_mean_p(evs, ts) if evs else {"n": 0, "p_le_zero": None}
    cost = sum(float(r.get("amount_usdt") or 0) for r in filled)
    pnl = sum(float(r["pnl"]) for r in filled if r.get("pnl") is not None)
    return {
        "n": len(filled),
        "settled_n": len(settled),
        "cost": round(cost, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / cost, 4) if cost > 0 else None,
        "win_rate": _r(m.get("win_rate")),
        "mean_breakeven": _r(m.get("mean_breakeven")),
        "edge": _r(m.get("edge")),
        "p_edge_negative": _r(m.get("p_edge_negative")),
        "ev_bootstrap": {
            "n": boot.get("n"),
            "p_le_zero": _r(boot.get("p_le_zero")),
            "significant_positive": bool(boot.get("significant_positive")),
        },
        "rejected": _reject_stats(rows),
    }


def _r(value: Any, digits: int = 4) -> float | None:
    """数值统一 round（事实卡里所有数字都会被 LLM 原样引用）。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def _daily_series(rows: Sequence[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for r in rows:
        if r.get("status") != "FILLED":
            continue
        day = dt.datetime.fromtimestamp(
            (r.get("created_ts") or 0) / 1000, dt.timezone.utc).strftime("%m-%d")
        b = buckets.setdefault(day, {"day": day, "n": 0, "cost": 0.0, "pnl": 0.0})
        b["n"] += 1
        b["cost"] += float(r.get("amount_usdt") or 0)
        b["pnl"] += float(r["pnl"]) if r.get("pnl") is not None else 0.0
    for b in buckets.values():
        b["cost"], b["pnl"] = round(b["cost"], 2), round(b["pnl"], 2)
    return [buckets[k] for k in sorted(buckets)]


def _channel_numbers(ch: dict) -> dict:
    """通道数值子集（扁平证据路径 ch.<channel>.<field> 的来源）。"""
    return {
        "n": ch["n"], "settled_n": ch["settled_n"], "cost": ch["cost"],
        "pnl": ch["pnl"], "roi": ch["roi"], "win_rate": ch["win_rate"],
        "win_rate_low": ch["win_rate_low"], "mean_breakeven": ch["mean_breakeven"],
        "edge": ch["edge"], "benchmark": ch["benchmark"],
        "benchmark_gap": ch["benchmark_gap"],
        "p_edge_negative": ch["p_edge_negative"],
        "trend_rank": ch["trend_rank"],
        "avg_stake": ch["avg_stake"], "rule_stake": ch["rule_stake"],
        "amount_usdt": ch["amount_usdt"], "amount_pct": ch["amount_pct"],
        "amount_cap_usdt": ch["amount_cap_usdt"],
        "max_exec_price": ch["max_exec_price"],
        "max_daily_orders": ch["max_daily_orders"],
        "reject_rate": ch["reject_rate"],
        "first_order_age_days": ch["first_order_age_days"],
        "enabled_rank": 1 if ch["enabled"] else 0,
    }


def _channel_facts(rows: Sequence[dict], channel: str, meta: dict | None,
                   now_ms: int) -> dict:
    filled = [r for r in rows if r.get("status") == "FILLED"]
    events = _orders_to_events(filled)
    bench = get_channel_benchmark(channel)
    bench_rate = bench.win_rate if bench is not None else None
    m = sh.compute_metrics(events, bench_win=bench_rate) if events else {"n": 0}
    ci = m.get("win_rate_ci95") or [None, None]
    meta = meta or {}
    # 单通道事实卡：复用归因引擎口径（streak / edge_decomp / 分桶），不另写一套
    extra, _ = compute_channel_facts(filled) if filled else ({}, set())
    trend = extra.get("trend") or {}
    cost = sum(float(r.get("amount_usdt") or 0) for r in filled)
    pnl = sum(float(r["pnl"]) for r in filled if r.get("pnl") is not None)
    stakes = [float(r.get("amount_usdt") or 0) for r in filled]
    rejects = _reject_stats(rows)
    first_ts = min((r.get("created_ts") or 0) for r in filled) if filled else None
    win_rate = _r(m.get("win_rate"))
    return {
        "channel": channel,
        "display_name": meta.get("display_name") or channel,
        "family": meta.get("family"),
        "market_period": meta.get("market_period"),
        "enabled": bool(meta.get("enabled")),
        "params": {
            "amount_usdt": _r(meta.get("amount_usdt"), 2),
            "amount_pct": _r(meta.get("amount_pct"), 6),
            "amount_cap_usdt": _r(meta.get("amount_cap_usdt"), 2),
            "max_daily_orders": meta.get("max_daily_orders"),
            "max_exec_price": _r(meta.get("max_exec_price"), 4),
        },
        "n": len(filled),
        "settled_n": int(m.get("n") or 0),
        "cost": round(cost, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / cost, 4) if cost > 0 else None,
        "win_rate": win_rate,
        "win_rate_low": _r(ci[0]),
        "mean_breakeven": _r(m.get("mean_breakeven")),
        "edge": _r(m.get("edge")),
        "benchmark": _r(bench_rate),
        "benchmark_gap": (_r(win_rate - bench_rate)
                          if win_rate is not None and bench_rate is not None else None),
        "p_edge_negative": _r(m.get("p_edge_negative")),
        "trend_level": trend.get("level"),
        "trend_rank": int(trend.get("level_rank") or 0),
        "trend_reasons": list(trend.get("reasons") or [])[:3],
        "avg_stake": round(sum(stakes) / len(stakes), 2) if stakes else None,
        "rule_stake": _r(meta.get("rule_amount"), 2),
        "amount_usdt": _r(meta.get("amount_usdt"), 2),
        "amount_pct": _r(meta.get("amount_pct"), 6),
        "amount_cap_usdt": _r(meta.get("amount_cap_usdt"), 2),
        "max_exec_price": _r(meta.get("max_exec_price"), 4),
        "max_daily_orders": meta.get("max_daily_orders"),
        "rejected": rejects,
        "reject_rate": (round(rejects["total"] / (rejects["total"] + len(filled)), 4)
                        if (rejects["total"] + len(filled)) else None),
        "first_order_age_days": (round((now_ms - first_ts) / 86_400_000, 1)
                                 if first_ts else None),
        "facts": {
            "streak": extra.get("streak"),
            "edge_decomp": extra.get("edge_decomp"),
            # 只带入场价带分桶（直接对应护栏/白名单改进）；时段分桶在体检面板看，
            # 不进周报事实卡——控 prompt 体积（15 通道 × 2 组桶 ≈ 45KB JSON）。
            "entry_band_ev": extra.get("entry_band_ev"),
        },
    }


def _concentration(channels: Sequence[dict]) -> dict:
    total = sum(float(c["cost"] or 0) for c in channels)
    rows = sorted(
        [{"channel": c["channel"], "cost": c["cost"],
          "share": round(float(c["cost"] or 0) / total, 4) if total > 0 else 0.0}
         for c in channels if (c["cost"] or 0) > 0],
        key=lambda r: r["cost"], reverse=True)
    return {
        "total_cost": round(total, 2),
        "top1_channel": rows[0]["channel"] if rows else None,
        "top1_share": rows[0]["share"] if rows else None,
        "top3_share": round(sum(r["share"] for r in rows[:3]), 4) if rows else None,
        "by_channel": rows,
    }


def _equity_block(base_equity: float, available: float | None,
                  open_cost: float) -> dict:
    drift = None
    if available is not None and base_equity > 0:
        drift = round((available - base_equity) / base_equity, 4)
    return {
        "base_equity_usdt": round(base_equity, 2),
        "available_usdt": _r(available, 2),
        "open_cost": round(open_cost, 2),
        "drift_ratio": drift,
    }


async def _open_cost(db, channels: Sequence[str]) -> float:
    """在途本金（已成交未结算的成本口径）。"""
    rows = (await db.execute(sa_select(TradeOrderModel.amount_in).where(
        TradeOrderModel.signal_version.in_(list(channels)),
        TradeOrderModel.status == "FILLED",
        TradeOrderModel.settled_at.is_(None),
    ))).scalars().all()
    total = 0.0
    for raw in rows:
        try:
            total += float(raw or 0) / 1e18
        except (TypeError, ValueError):
            continue
    return round(total, 2)


def _numeric_paths(node: Any, prefix: str = "", out: set[str] | None = None) -> set[str]:
    """事实卡内可引用路径集合（与 llm_panel._flatten_facts 递归规则严格一致）。

    只看 dict 与数值叶子；**不递归 list**——list 项在 _flatten_facts 里不可寻址，
    若这里放行就会产生「闸门永远拒掉」的假证据 ID。通道数值走 `ch.<channel>.*`。
    """
    out = set() if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            _numeric_paths(v, path, out)
    return out


async def collect_facts(
    db,
    *,
    now_ms: int | None = None,
    week_offset: int = -1,
    channel_meta: dict[str, dict] | None = None,
    equity_provider: EquityProvider | None = None,
    base_equity: float | None = None,
) -> dict:
    """组合 + 单信号事实卡（LLM 唯一数字来源）。"""
    now_ms = now_ms or int(time.time() * 1000)
    win = week_window(now_ms, week_offset)
    channels = list(channel_meta.keys()) if channel_meta else []
    rows = await _load_orders(db, win["start_ms"], win["end_ms"], channels) if channels else []
    prev_rows = (await _load_orders(db, win["prev_start_ms"], win["prev_end_ms"], channels)
                 if channels else [])

    ch_rows: dict[str, list[dict]] = {}
    for r in rows:
        ch_rows.setdefault(r["channel"], []).append(r)

    channels_out = [
        _channel_facts(ch_rows.get(ch, []), ch, (channel_meta or {}).get(ch), now_ms)
        for ch in channels
    ]
    channels_out.sort(key=lambda c: (c["pnl"], c["cost"]), reverse=True)
    # 本周零成交的通道不展开事实块（20 个空块≈40k 字符纯噪声）；只留一行名单，
    # 保留“它本周没开火”这个信息，同时把 prompt 从 ~97k 降到 ~55k 字符。
    idle_channels = [c["channel"] for c in channels_out if not c["n"]]
    channels_out = [c for c in channels_out if c["n"]]

    portfolio = _portfolio_metrics(rows)
    prev_portfolio = _portfolio_metrics(prev_rows)
    portfolio["prev"] = {
        k: prev_portfolio[k] for k in
        ("n", "cost", "pnl", "roi", "win_rate", "edge", "mean_breakeven")
    }
    portfolio["delta"] = {
        k: (round(portfolio[k] - prev_portfolio[k], 4)
            if portfolio.get(k) is not None and prev_portfolio.get(k) is not None else None)
        for k in ("pnl", "roi", "win_rate", "edge", "cost")
    }
    portfolio["daily"] = _daily_series(rows)
    portfolio["concentration"] = _concentration(channels_out)
    portfolio["idle_channels"] = {"n": len(idle_channels), "channels": idle_channels[:40]}

    base_eq = float(base_equity if base_equity is not None
                    else settings.live_base_equity_usdt)
    available = None
    if equity_provider is not None:
        try:
            available = await equity_provider()
        except Exception as exc:
            logger.warning("周报：余额快照获取失败（EQUITY_DRIFT 跳过）| {}", exc)
    portfolio["equity"] = _equity_block(base_eq, available,
                                       await _open_cost(db, channels) if channels else 0.0)

    facts: dict = {
        "schema": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "week": win,
        "portfolio": portfolio,
        "channels": channels_out,
        "ch": {c["channel"]: _channel_numbers(c) for c in channels_out},
    }
    facts["watchouts"] = derive_watchouts(facts)
    facts["evidence_ids"] = sorted(_numeric_paths(facts))
    facts["coverage"] = {
        "channels": len(channels),
        "orders": len(rows),
        "prev_orders": len(prev_rows),
        "orders_with_settlement": sum(1 for r in rows if r.get("win") is not None),
        "unsettled_stuck": _unsettled_stuck(rows, now_ms),
    }
    return facts


def _unsettled_stuck(rows: Sequence[dict], now_ms: int) -> int:
    """已成交、未结算、且窗口结束已超 UNSETTLED_BACKLOG_MIN 分钟的单数。"""
    stuck = 0
    for r in rows:
        if r.get("status") != "FILLED" or r.get("win") is not None:
            continue
        ws, period = r.get("window_start"), r.get("market_period")
        if not ws:
            continue
        span = 900_000 if period == "15m" else 300_000
        if now_ms - (int(ws) + span) > UNSETTLED_BACKLOG_MIN * 60_000:
            stuck += 1
    return stuck


# ======================================================================
# 注意点：10 条确定性规则（不经过 LLM）
# ======================================================================

def _watch(code: str, severity: str, channel: str | None, statement: str,
           evidence_ids: Sequence[str], numbers: Sequence[dict]) -> dict:
    return {"code": code, "severity": severity, "channel": channel,
            "statement": statement, "evidence_ids": list(evidence_ids),
            "numbers": list(numbers)}


def derive_watchouts(facts: dict) -> list[dict]:
    """10 条规则 → 注意点列表（按严重度排序）。纯函数，可单测。"""
    out: list[dict] = []
    portfolio = facts.get("portfolio") or {}
    channels = facts.get("channels") or []
    conc = portfolio.get("concentration") or {}
    equity = portfolio.get("equity") or {}

    top1_share = conc.get("top1_share")
    top3_share = conc.get("top3_share")
    if (top1_share is not None and top1_share >= CONCENTRATION_TOP1) or (
            top3_share is not None and top3_share >= CONCENTRATION_TOP3):
        out.append(_watch(
            "CONCENTRATION", "P1", conc.get("top1_channel"),
            f"资金集中度偏高：top1 {top1_share:.0%} / top3 {top3_share:.0%}"
            if top1_share is not None and top3_share is not None else "资金集中度偏高",
            ["portfolio.concentration.top1_share", "portfolio.concentration.top3_share"],
            [{"name": "portfolio.concentration.top1_share", "value": top1_share},
             {"name": "portfolio.concentration.top3_share", "value": top3_share}]))

    for c in channels:
        ch, p = c["channel"], f"ch.{c['channel']}"
        if c["n"] and c["n"] < SMALL_SAMPLE_N and (c["pnl"] or 0) > 0:
            out.append(_watch(
                "SMALL_SAMPLE_RANK", "P2", ch,
                f"{ch} 仅 {c['n']} 单（<{SMALL_SAMPLE_N}）却为正贡献，排名不可信",
                [f"{p}.n", f"{p}.pnl"],
                [{"name": f"{p}.n", "value": c["n"]},
                 {"name": f"{p}.pnl", "value": c["pnl"]}]))

        gap = c.get("benchmark_gap")
        if (c.get("settled_n") or 0) >= SMALL_SAMPLE_N and gap is not None and abs(gap) >= BENCHMARK_GAP:
            out.append(_watch(
                "BENCHMARK_DIVERGENCE", "P1", ch,
                f"{ch} 前向胜率 {c['win_rate']:.1%} vs 冻结基准 {c['benchmark']:.1%}（差 {gap:+.1%}）",
                [f"{p}.win_rate", f"{p}.benchmark", f"{p}.benchmark_gap"],
                [{"name": f"{p}.win_rate", "value": c["win_rate"]},
                 {"name": f"{p}.benchmark", "value": c["benchmark"]},
                 {"name": f"{p}.benchmark_gap", "value": gap}]))

        guard, bench = c.get("max_exec_price"), c.get("benchmark")
        if guard is not None and bench is not None and guard >= GUARD_NEAR_BREAKEVEN * bench:
            out.append(_watch(
                "GUARD_AT_BREAKEVEN", "P1", ch,
                f"{ch} 护栏 {guard} ≥ 基准胜率×0.98（{GUARD_NEAR_BREAKEVEN * bench:.3f}）："
                "护栏卡在保本线附近，被拦单可能仍是正 EV",
                [f"{p}.max_exec_price", f"{p}.benchmark"],
                [{"name": f"{p}.max_exec_price", "value": guard},
                 {"name": f"{p}.benchmark", "value": bench}]))

        rate = c.get("reject_rate")
        if rate is not None and rate >= REJECT_RATE_HIGH:
            out.append(_watch(
                "GUARD_REJECT_RATE", "P1", ch,
                f"{ch} 弃单率 {rate:.0%}（护栏过紧或窗口不可成交）",
                [f"{p}.reject_rate"],
                [{"name": f"{p}.reject_rate", "value": rate}]))

        edge, p_neg = c.get("edge"), c.get("p_edge_negative")
        if (edge is not None and edge < 0 and p_neg is not None
                and p_neg >= EDGE_NEGATIVE_P and (c.get("settled_n") or 0) >= SMALL_SAMPLE_N):
            out.append(_watch(
                "EDGE_NEGATIVE", "P0", ch,
                f"{ch} 边际 {edge:+.3f} 且优势已变负的后验概率 {p_neg:.0%}"
                f"（n={c['settled_n']}）：停火候选",
                [f"{p}.edge", f"{p}.p_edge_negative", f"{p}.settled_n"],
                [{"name": f"{p}.edge", "value": edge},
                 {"name": f"{p}.p_edge_negative", "value": p_neg},
                 {"name": f"{p}.settled_n", "value": c["settled_n"]}]))

        if c.get("enabled") and c.get("first_order_age_days") is not None \
                and c["first_order_age_days"] < NEW_CHANNEL_DAYS:
            out.append(_watch(
                "NEW_CHANNEL_NO_GATE", "P1", ch,
                f"{ch} 已开实盘但首单仅 {c['first_order_age_days']:.0f} 天前"
                f"（<{NEW_CHANNEL_DAYS} 天影子门槛）",
                [f"{p}.first_order_age_days"],
                [{"name": f"{p}.first_order_age_days",
                  "value": c["first_order_age_days"]}]))

        if c.get("trend_rank", 0) >= 2:
            out.append(_watch(
                "TREND_UNRESOLVED", "P1", ch,
                f"{ch} 趋势 {c['trend_level']}：" +
                ("；".join(c.get("trend_reasons") or []) or "EV 趋势恶化待处置"),
                [f"{p}.trend_rank"],
                [{"name": f"{p}.trend_rank", "value": c["trend_rank"]}]))

    stuck = (facts.get("coverage") or {}).get("unsettled_stuck", 0)
    if stuck > 0:
        out.append(_watch(
            "UNSETTLED_BACKLOG", "P0", None,
            f"本周 {stuck} 笔已成交订单超 {UNSETTLED_BACKLOG_MIN} 分钟仍未结算"
            "（排查 settle-scan / 结算钩子）",
            ["coverage.unsettled_stuck"],
            [{"name": "coverage.unsettled_stuck", "value": stuck}]))

    drift = equity.get("drift_ratio")
    if drift is not None and abs(drift) >= EQUITY_DRIFT_RATIO:
        out.append(_watch(
            "EQUITY_DRIFT", "P2", None,
            f"可用余额与基准本金偏离 {drift:+.0%}：请复核 base_equity_usdt"
            "（分母是人工设定的固定值，不自动跟随权益）",
            ["portfolio.equity.drift_ratio"],
            [{"name": "portfolio.equity.drift_ratio", "value": drift}]))

    out.sort(key=lambda w: (-SEVERITY_RANK.get(w["severity"], 0), w["channel"] or ""))
    return out


# ======================================================================
# 编排（读事实 → LLM → 过闸 → 落库）
# ======================================================================

def weekly_panel():
    """周报模型面板：默认单模型（控成本），settings 可覆盖。

    超时/输出预算都用周报专用值：15 个通道的报告比归因一组长，16000 输出会在
    思考模型上被截断成非法 JSON（2026-10-30 线上真实发生）。
    """
    from .llm_panel import LLMPanel, _provider_entry, default_panel

    timeout_s = settings.weekly_review_timeout_s
    max_tokens = settings.weekly_review_max_tokens
    override = settings.weekly_review_models.strip()
    if override:
        out = []
        for item in override.split(","):
            if ":" not in item:
                continue
            provider, model = item.split(":", 1)
            entry = _provider_entry(provider.strip(), model.strip())
            if entry:
                out.append(entry)
        if out:
            return LLMPanel(out, timeout_s=timeout_s, max_tokens=max_tokens)
    full = default_panel()
    return LLMPanel(full[:1] if full else [], timeout_s=timeout_s,
                    max_tokens=max_tokens)


async def _run_llm(facts: dict, channel_meta: dict[str, dict],
                   base_equity: float) -> tuple[dict | None, list[dict], list[dict], str | None]:
    """调用周报模型并过闸。返回 (report, rejected, models, error)。"""
    from .llm_panel import WEEKLY_SYSTEM_PROMPT, build_weekly_prompt, verify_weekly_review

    panel = weekly_panel()
    prompt = build_weekly_prompt(facts)
    # 必须传周报自己的 system prompt：只换 user prompt 会让模型按归因 schema 输出
    outputs = await panel.analyze(prompt, system_prompt=WEEKLY_SYSTEM_PROMPT)
    models: list[dict] = []
    rejected: list[dict] = []
    for mo in outputs:
        models.append({"model": f"{mo.get('provider')}:{mo.get('model')}",
                       "family": mo.get("family"),
                       "prompt_tokens": mo.get("prompt_tokens"),
                       "output_tokens": mo.get("output_tokens"),
                       "error": mo.get("error")})
        if mo.get("error") or not mo.get("parsed"):
            continue
        report, rej = verify_weekly_review(
            mo["parsed"], facts, set(facts.get("evidence_ids") or []),
            channel_meta=channel_meta, base_equity=base_equity)
        rejected.extend(rej)
        if report is None:
            # 过闸失败必须留可诊断证据（否则线上只能重跑一次才知道为什么）
            parsed = mo["parsed"] if isinstance(mo["parsed"], dict) else {}
            rejected.append({
                "reason": "report_rejected",
                "model": f"{mo.get('provider')}:{mo.get('model')}",
                "schema": parsed.get("schema"),
                "top_keys": list(parsed.keys())[:10],
                "raw_text_head": (mo.get("raw_text_head") or "")[:500],
            })
        if report is not None:
            return report, rejected, models, None
    error = next((m["error"] for m in models if m.get("error")), None) or "no_usable_output"
    return None, rejected, models, error


async def run_weekly_review(
    trigger_type: str = "MANUAL",
    *,
    week_offset: int = -1,
    now_ms: int | None = None,
    channel_meta: dict[str, dict] | None = None,
    equity_provider: EquityProvider | None = None,
    base_equity: float | None = None,
    persist: bool = True,
) -> dict | None:
    """生成一份周报（事实卡一定产出；LLM 段可降级），返回可序列化 dict。

    base_equity: 运行时基准本金（`multi_live_trader.status()["base_equity_usdt"]`）。
    **必须传运行时值**——事实卡分母、EQUITY_DRIFT、仓位闸门的「当前金额」
    必须同源，否则离线改过基准后闸门会拿错基数（2026-10-30 审核 P0）。
    缺省回落 env 默认（仅无执行器时）。
    周报失败/异常绝不抛给调用方（定时任务与 HTTP 端点都不应因此 500）。
    """
    if channel_meta is None:
        from .live_channels import LIVE_CHANNELS
        channel_meta = {ch: {"display_name": spec.display_name,
                             "family": spec.family,
                             "market_period": spec.market_period}
                        for ch, spec in LIVE_CHANNELS.items()}

    win = week_window(now_ms, week_offset)
    row: WeeklyReview | None = None
    if persist and trigger_type == "WEEKLY":
        # 幂等：同一周的 WEEKLY 周报只生成一次（进程在 00:00–00:20 窗口内重启时，
        # 内存 last_week 会丢，否则重复插行、前端出现两份同周报告）。
        existing = await find_weekly_review(win["key"], "WEEKLY")
        if existing is not None:
            logger.info("周报已存在，跳过重复生成 | {} | id={}", win["key"], existing["id"])
            return existing
    if persist:
        async with async_session_factory() as db:
            row = WeeklyReview(
                week_key=win["key"], period_start_ms=win["start_ms"],
                period_end_ms=win["end_ms"], trigger_type=trigger_type,
                status="RUNNING", verdict=None, facts={}, report={},
                rejected=[], models=[], application_log=[])
            db.add(row)
            await db.commit()
            await db.refresh(row)
            run_id = row.id

    facts: dict = {}
    try:
        async with async_session_factory() as db:
            facts = await collect_facts(
                db, now_ms=now_ms, week_offset=week_offset,
                channel_meta=channel_meta, equity_provider=equity_provider,
                base_equity=base_equity)
    except Exception as exc:
        logger.warning("周报事实卡构建失败 | {} | {}", win["key"], exc)
        if row is not None:
            await _finish(run_id, status="FAILED", verdict="FACTS_FAILED",
                          facts={}, report={}, rejected=[], models=[], error=str(exc)[:500])
        return None

    report: dict | None = None
    rejected: list[dict] = []
    models: list[dict] = []
    error: str | None = None
    verdict = "OK"
    if (facts.get("portfolio") or {}).get("n", 0) == 0:
        verdict = "INSUFFICIENT_DATA"
    elif settings.weekly_review_llm_enabled:
        # 断言闸的分母必须与事实卡同一个数（不能再用 env 默认，见 docstring）
        runtime_base = float(
            (facts.get("portfolio") or {}).get("equity", {}).get("base_equity_usdt")
            or base_equity or settings.live_base_equity_usdt)
        try:
            report, rejected, models, error = await _run_llm(
                facts, channel_meta, runtime_base)
        except Exception as exc:  # 模型层异常 → 降级仍出事实卡
            logger.warning("周报 LLM 段失败（降级为纯事实卡）| {}", exc)
            error = f"{type(exc).__name__}: {exc}"[:500]
        if report is None:
            verdict = "LLM_ONLY_FAILED"

    final = _assemble_report(facts, report, verdict)
    if row is not None:
        await _finish(run_id, status="DONE", verdict=verdict, facts=facts,
                      report=final, rejected=rejected, models=models, error=error)
        return await get_weekly_review(run_id)
    return weekly_dict(WeeklyReview(
        week_key=win["key"], period_start_ms=win["start_ms"],
        period_end_ms=win["end_ms"], trigger_type=trigger_type,
        status="DONE", verdict=verdict, facts=facts, report=final,
        rejected=rejected, models=models, llm_error=error))


def weekly_dict(row: WeeklyReview) -> dict:
    """ORM 行 → 可序列化 dict（会话关闭后不得再碰 ORM 属性）。"""
    return {
        "id": row.id, "week_key": row.week_key,
        "period_start_ms": row.period_start_ms, "period_end_ms": row.period_end_ms,
        "trigger_type": row.trigger_type, "status": row.status,
        "verdict": row.verdict, "facts": row.facts, "report": row.report,
        "rejected": row.rejected, "models": row.models,
        "llm_error": row.llm_error, "application_log": row.application_log,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
    }


async def get_weekly_review(run_id: int) -> dict | None:
    async with async_session_factory() as db:
        row = await db.get(WeeklyReview, run_id)
        return weekly_dict(row) if row is not None else None


async def find_weekly_review(week_key: str, trigger_type: str) -> dict | None:
    """按周 + 触发类型查已有周报（幂等与去重用）。"""
    async with async_session_factory() as db:
        row = (await db.execute(sa_select(WeeklyReview).where(
            WeeklyReview.week_key == week_key,
            WeeklyReview.trigger_type == trigger_type,
        ).order_by(WeeklyReview.created_at.desc()).limit(1))).scalars().first()
        return weekly_dict(row) if row is not None else None


def _assemble_report(facts: dict, llm_report: dict | None, verdict: str) -> dict:
    """最终报告 = 确定性段（注意点/事实摘要）+ LLM 段（可为空）。"""
    return {
        "schema": "weekly_review_report_v1",
        "verdict": verdict,
        "week": facts.get("week"),
        "deterministic": {
            "watchouts": facts.get("watchouts") or [],
            "portfolio": {k: (facts.get("portfolio") or {}).get(k) for k in
                          ("n", "cost", "pnl", "roi", "win_rate", "edge",
                           "mean_breakeven", "delta", "concentration", "equity")},
        },
        "llm": llm_report,
        "narrative_note": ("LLM 段只解释确定性事实；每句因果断言都必须过证据闸，"
                           "被剔除的断言见 rejected。"),
    }


async def _finish(run_id: int, *, status: str, verdict: str | None, facts: dict,
                  report: dict, rejected: list, models: list,
                  error: str | None) -> None:
    async with async_session_factory() as db:
        row = await db.get(WeeklyReview, run_id)
        if row is None:
            return
        row.status = status
        row.verdict = verdict
        row.facts = facts
        row.report = report
        row.rejected = rejected
        row.models = models
        row.llm_error = error
        row.finished_at = dt.datetime.now(dt.timezone.utc)
        await db.commit()
