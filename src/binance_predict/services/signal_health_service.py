"""信号体检：事实层组装、快照持久化、情绪日指标、LLM 诊断留口。

取数（影子行/实盘订单）由 main.py 编排后传入，本模块不 import main，避免循环依赖。
"""
from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable, Mapping

from binance_predict.services import signal_health as sh

# LLM 诊断留口：接入时把 async (payload) -> dict 赋给它；未赋值 = 未实现（P1 不接模型）。
llm_diagnose_hook: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None


async def run_llm_diagnosis(payload: dict[str, Any]) -> dict[str, Any]:
    if llm_diagnose_hook is None:
        return {"status": "not_implemented", "message": "AI 诊断尚未接入；下方为将提交给模型的结构化输入。",
                "payload": payload}
    return {"status": "ok", **(await llm_diagnose_hook(payload))}


async def load_market_regime_lookup(db, min_ts: int | None = None, max_ts: int | None = None) -> Callable[[int], dict[str, float | None]]:
    """从 sentiment_windows 预计算 5m 窗口历史行情状态（4h趋势、24h趋势、4h波动率）。

    返回一个查找函数 lookup(ts) -> {"ret_4h": ..., "ret_24h": ..., "vol_4h": ...}
    纯内存计算，不增加网络开销。
    """
    from sqlalchemy import select
    from binance_predict.db.models import SentimentWindow

    # 往前扩 288+ 根 5m 窗口（约 26h）以确保能计算最早事件的 24h 趋势
    query = select(SentimentWindow.start_time, SentimentWindow.entry_price)
    if min_ts is not None:
        lookback_ts = min_ts - (300 * 300_000)
        query = query.where(SentimentWindow.start_time >= lookback_ts)
    if max_ts is not None:
        query = query.where(SentimentWindow.start_time <= max_ts)

    res = (await db.execute(query.order_by(SentimentWindow.start_time.asc()))).all()
    if not res:
        return lambda ts: {"ret_4h": None, "ret_24h": None, "vol_4h": None}

    # 构造按 start_time 升序的有序列表
    times = [int(r[0]) for r in res]
    prices = [float(r[1]) if r[1] is not None and float(r[1]) > 0 else None for r in res]
    n = len(times)

    # 预计算每个索引的 ret_4h(前48窗), ret_24h(前288窗), vol_4h(前48窗收益标准差)
    regime_cache: dict[int, dict[str, float | None]] = {}

    import math
    import bisect

    for i in range(n):
        t = times[i]
        p = prices[i]
        if p is None:
            continue

        r4 = None
        if i >= 48 and prices[i - 48] is not None and prices[i - 48] > 0:
            r4 = p / prices[i - 48] - 1.0

        r24 = None
        if i >= 288 and prices[i - 288] is not None and prices[i - 288] > 0:
            r24 = p / prices[i - 288] - 1.0

        v4 = None
        if i >= 48:
            # 计算过去 48 根 5m 价格回报标准差
            rets = []
            for j in range(i - 47, i + 1):
                p_prev, p_curr = prices[j - 1], prices[j]
                if p_prev and p_curr and p_prev > 0 and p_curr > 0:
                    rets.append(p_curr / p_prev - 1.0)
            if len(rets) >= 20:
                mean_r = sum(rets) / len(rets)
                v4 = math.sqrt(sum((x - mean_r) ** 2 for x in rets) / (len(rets) - 1))

        regime_cache[t] = {"ret_4h": r4, "ret_24h": r24, "vol_4h": v4}

    def lookup(ts: int) -> dict[str, float | None]:
        # 寻找 <= ts 且最接近的 5m 窗口整点
        idx = bisect.bisect_right(times, ts) - 1
        if 0 <= idx < n:
            matched_t = times[idx]
            # 必须在 15 分钟误差范围内
            if abs(ts - matched_t) <= 900_000:
                return regime_cache.get(matched_t, {"ret_4h": None, "ret_24h": None, "vol_4h": None})
        return {"ret_4h": None, "ret_24h": None, "vol_4h": None}

    return lookup


def shadow_events(
    rows: Iterable[Any],
    breakeven_fn: Callable[[str, float], float],
    ev_fn: Callable[[str, bool, float | None], float | None],
    regime_lookup: Callable[[int], dict[str, float | None]] | None = None,
) -> dict[str, list[sh.Event]]:
    """影子 SETTLED 行 → 按版本分组的事件（升序）。EV 优先落库值，缺失按报价兜底现算。"""
    out: dict[str, list[sh.Event]] = defaultdict(list)
    for r in rows:
        if r.win is None or r.window_start is None:
            continue
        ts = int(r.window_start)
        q = r.entry_down_price if r.direction == "DOWN" else r.entry_up_price
        q = float(q) if q is not None and float(q) > 0 else None
        ev = None
        if q is not None:
            ev = float(r.ev_at_entry) if r.ev_at_entry is not None else ev_fn(r.version, bool(r.win), q)
        reg = regime_lookup(ts) if regime_lookup else {}
        out[r.version].append({
            "ts": ts, "win": bool(r.win), "ev": ev,
            "be": breakeven_fn(r.version, q) if q is not None else None, "q": q,
            "ret_4h": reg.get("ret_4h"), "ret_24h": reg.get("ret_24h"), "vol_4h": reg.get("vol_4h"),
        })
    for evs in out.values():
        evs.sort(key=lambda e: e["ts"])
    return out


def live_events(
    orders: Iterable[Any],
    regime_lookup: Callable[[int], dict[str, float | None]] | None = None,
) -> dict[str, list[sh.Event]]:
    """实盘已结算订单 → 按通道分组的事件（升序）。EV 为单位本金收益（与下单金额无关）。"""
    from binance_predict.services.live_performance import (
        _order_win, break_even, confirmed_execution_price, unit_return,
    )

    out: dict[str, list[sh.Event]] = defaultdict(list)
    for o in orders:
        win = _order_win(o)
        if win is None or o.window_start is None or not o.channel:
            continue
        ts = int(o.window_start)
        reg = regime_lookup(ts) if regime_lookup else {}
        out[o.channel].append({
            "ts": ts, "win": win, "ev": unit_return(o),
            "be": break_even(o)["break_even_probability"], "q": confirmed_execution_price(o),
            "ret_4h": reg.get("ret_4h"), "ret_24h": reg.get("ret_24h"), "vol_4h": reg.get("vol_4h"),
        })
    for evs in out.values():
        evs.sort(key=lambda e: e["ts"])
    return out


def assemble_report(
    shadow_ev: Mapping[str, list[sh.Event]],
    live_ev: Mapping[str, list[sh.Event]],
    shadow_meta: Mapping[str, Mapping[str, Any]],
    live_meta: Mapping[str, Mapping[str, Any]],
    bench_fn: Callable[[str, str], float | None],
    now_ms: int | None = None,
) -> dict[str, Any]:
    """装配整份体检报告 + 各 key 事件（供下钻复用，不序列化给前端）。"""
    rows: list[dict[str, Any]] = []
    events: dict[tuple[str, str], list[sh.Event]] = {}
    active_events_by_key: dict[str, list[sh.Event]] = {}

    for scope, evmap, meta in (("live", live_ev, live_meta), ("shadow", shadow_ev, shadow_meta)):
        for key in sorted(set(evmap) | set(meta)):
            evs = evmap.get(key, [])
            b_win = bench_fn(scope, key)
            m = sh.compute_metrics(evs, b_win)
            light, reasons = sh.classify(m)
            info = meta.get(key, {})
            is_active = bool(info.get("active"))

            # P3: 死因诊断与变化点检测
            death_status, death_reason = sh.diagnose_death_status(evs, b_win)
            change_point = sh.detect_change_point(evs)

            rows.append({
                "scope": scope, "key": key, "light": light, "reasons": reasons, "metrics": m,
                "family": info.get("family"), "market_period": info.get("market_period"),
                "enabled": bool(info.get("enabled")), "retired": bool(info.get("retired")),
                "active": is_active,
                "death_status": death_status,
                "death_reason": death_reason,
                "change_point": change_point,
            })
            events[(scope, key)] = evs
            if is_active and evs:
                active_events_by_key[f"{scope}:{key}"] = evs

    # P3: 通道间相关性与同窗共振（针对所有在线通道）
    correlations = sh.compute_channel_correlations(active_events_by_key, min_shared=5)

    tally = {c: sum(1 for r in rows if r["light"] == c and r["active"]) for c in ("RED", "YELLOW", "GREEN", "GRAY")}
    return {
        "generated_at": now_ms or int(time.time() * 1000), "tally_active": tally,
        "thresholds": {"min_n": sh.MIN_N, "red_p_neg": sh.RED_P_NEG, "yellow_p_neg": sh.YELLOW_P_NEG,
                       "yellow_p_bench": sh.YELLOW_P_BENCH, "tail_share": sh.TAIL_SHARE},
        "rows": rows, "correlations": correlations[:20], "_events": events,
    }


def detail_for(scope: str, key: str, row: Mapping[str, Any], evs: list[sh.Event],
               history: list[dict[str, Any]]) -> dict[str, Any]:
    price, hour = sh.price_buckets(evs), sh.hour_buckets(evs)
    t4 = sh.trend_4h_buckets(evs)
    t24 = sh.trend_24h_buckets(evs)
    vol = sh.volatility_buckets(evs)

    death_status = row.get("death_status") or "ALIVE"
    change_point = row.get("change_point")

    diag_payload = sh.build_diagnosis_payload(
        key, scope, row["light"], row["reasons"], row["metrics"],
        price, hour, death_status=death_status, change_point=change_point,
        trend_4h=t4, trend_24h=t24, volatility=vol,
    )

    return {
        "scope": scope, "key": key, "light": row["light"], "reasons": row["reasons"], "metrics": row["metrics"],
        "series": sh.series(evs),
        "price_buckets": price, "hour_buckets": hour,
        "trend_4h_buckets": t4, "trend_24h_buckets": t24, "volatility_buckets": vol,
        "death_status": death_status, "death_reason": row.get("death_reason"),
        "change_point": change_point,
        "history": history,
        "diagnosis_payload": diag_payload,
    }


# ---------------- 快照持久化（每 scope/key/UTC 日一行，小时级覆盖更新） ----------------

def utc_date(now_ms: int) -> str:
    return datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


async def persist_snapshots(db, rows: list[dict[str, Any]], now_ms: int) -> None:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from binance_predict.db.models import SignalHealthSnapshot as S

    day = utc_date(now_ms)
    for r in rows:
        vals = {"scope": r["scope"], "key": r["key"], "snap_date": day, "light": r["light"],
                "n": int(r["metrics"].get("n", 0)), "reasons": r["reasons"], "metrics": r["metrics"]}
        stmt = pg_insert(S).values(**vals)
        await db.execute(stmt.on_conflict_do_update(
            constraint="uq_signal_health_scope_key_date",
            set_={k: stmt.excluded[k] for k in ("light", "n", "reasons", "metrics")},
        ))
    await db.commit()


async def load_prev_lights(db, now_ms: int) -> dict[tuple[str, str], str]:
    """重启后恢复灯色基线：取近 2 日各 (scope,key) 最新一行。"""
    from sqlalchemy import select

    from binance_predict.db.models import SignalHealthSnapshot as S

    since = utc_date(now_ms - 2 * 86_400_000)
    res = (await db.execute(
        select(S.scope, S.key, S.light).where(S.snap_date >= since).order_by(S.updated_at.asc())
    )).all()
    return {(s, k): light for s, k, light in res}


async def load_history(db, scope: str, key: str, days: int = 60) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from binance_predict.db.models import SignalHealthSnapshot as S

    res = (await db.execute(
        select(S.snap_date, S.light, S.n, S.metrics).where(S.scope == scope, S.key == key)
        .order_by(S.snap_date.desc()).limit(days)
    )).all()
    return [{"date": d, "light": li, "n": n, "win_rate": m.get("win_rate"), "avg_ev": m.get("avg_ev"),
             "p_edge_negative": m.get("p_edge_negative")} for d, li, n, m in reversed(res)]


# ---------------- 情绪数据日留存（情绪原始采样会被回收，日指标是长期上下文） ----------------

def sentiment_summary(windows: Iterable[tuple[str | None, float | None, float | None, float | None]]) -> dict[str, Any]:
    """近 24h 情绪窗口汇总。windows 元素 = (outcome, avg_participants, avg_trade_volume, actual_return)。"""
    rows = list(windows)
    n = len(rows)

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    rets = [float(r[3]) for r in rows if r[3] is not None]
    mu = mean(rets)
    sd = (sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5 if mu is not None and len(rets) > 1 else None
    return {
        "n": n, "coverage_288": n / 288,
        "up_share": sum(1 for r in rows if r[0] == "UP") / n if n else None,
        "down_share": sum(1 for r in rows if r[0] == "DOWN") / n if n else None,
        "noise_share": sum(1 for r in rows if r[0] == "NOISE") / n if n else None,
        "avg_participants": mean([float(r[1]) for r in rows if r[1] is not None]),
        "avg_trade_volume": mean([float(r[2]) for r in rows if r[2] is not None]),
        "mean_abs_return": mean([abs(x) for x in rets]), "return_std": sd,
    }


async def persist_sentiment_daily(db, now_ms: int) -> dict[str, Any]:
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from binance_predict.db.models import SentimentWindow as W, SignalHealthSnapshot as S

    res = (await db.execute(
        select(W.outcome, W.avg_participants, W.avg_trade_volume, W.actual_return)
        .where(W.start_time >= now_ms - 86_400_000)
    )).all()
    m = sentiment_summary([tuple(r) for r in res])
    stmt = pg_insert(S).values(scope="sentiment", key="daily", snap_date=utc_date(now_ms),
                               light="GRAY", n=m["n"], reasons=[], metrics=m)
    await db.execute(stmt.on_conflict_do_update(
        constraint="uq_signal_health_scope_key_date",
        set_={k: stmt.excluded[k] for k in ("n", "metrics")}))
    await db.commit()
    return m
