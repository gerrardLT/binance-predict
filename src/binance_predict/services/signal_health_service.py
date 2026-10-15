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


def _empty_regime() -> dict[str, float | None]:
    return {"ret_4h": None, "ret_24h": None, "vol_4h": None, "up_streak_5m": None}


def up_streak_series(prices: list[float | None]) -> list[int | None]:
    """逐窗 5m 连续收阳根数：价格严格高于前窗 +1 连涨累计，否则清零；缺价 → None。

    与「5 分钟 K 一根接一根收阳」的肉眼口径一致（窗口起点价≈前窗收盘价）。
    """
    out: list[int | None] = []
    streak: int | None = None
    for i, p in enumerate(prices):
        if p is None:
            streak = None
        elif i == 0 or prices[i - 1] is None:
            streak = 0
        elif p > prices[i - 1]:
            streak = (streak or 0) + 1
        else:
            streak = 0
        out.append(streak)
    return out


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
        return lambda ts: _empty_regime()

    # 构造按 start_time 升序的有序列表
    times = [int(r[0]) for r in res]
    prices = [float(r[1]) if r[1] is not None and float(r[1]) > 0 else None for r in res]
    n = len(times)
    streaks = up_streak_series(prices)

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

        regime_cache[t] = {"ret_4h": r4, "ret_24h": r24, "vol_4h": v4,
                           "up_streak_5m": streaks[i]}

    def lookup(ts: int) -> dict[str, float | None]:
        # 寻找 <= ts 且最接近的 5m 窗口整点
        idx = bisect.bisect_right(times, ts) - 1
        if 0 <= idx < n:
            matched_t = times[idx]
            # 必须在 15 分钟误差范围内
            if abs(ts - matched_t) <= 900_000:
                return regime_cache.get(matched_t, _empty_regime())
        return _empty_regime()

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
        streak = getattr(o, "up_streak_5m", None)
        if streak is None:
            streak = reg.get("up_streak_5m")
        out[o.channel].append({
            "ts": ts, "win": win, "ev": unit_return(o),
            "be": break_even(o)["break_even_probability"], "q": confirmed_execution_price(o),
            "direction": o.direction,
            "ret_4h": reg.get("ret_4h"), "ret_24h": reg.get("ret_24h"), "vol_4h": reg.get("vol_4h"),
            "up_streak_5m": streak,
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


# ---------------- 买贵校准监控（2026-10-16）：实盘订单 通道×行情档 校准矩阵 ----------------
# 2026-10-15 实测背景：DOWN 单在 5m 连涨 ≥4 根段 ROI 转负（+8%→-22% 单调恶化）、
# 深跌(ret24≤-2%)追单 ROI -45%。本监控把「该行情档实际胜率是否低于入场价保本线」
# 做成常驻仪表：全系统合并(按方向) + 在线通道两层，桶级转红走既有微信告警链路。

CALIBRATION_LENSES: tuple[tuple[str, str, Callable[[sh.Event], str], list[str]], ...] = (
    ("ret_24h", "24小时行情",
     lambda e: sh.ret_24h_label(e.get("ret_24h")), list(sh._TREND_24H_LABELS)),
    ("up_streak_5m", "5m连续收阳",
     lambda e: sh.up_streak_label(e.get("up_streak_5m")), list(sh.UP_STREAK_LABELS)),
)


def build_calibration_report(
    live_ev: Mapping[str, list[sh.Event]],
    live_meta: Mapping[str, Mapping[str, Any]],
    channel_direction: Mapping[str, str] | None = None,
    channel_display: Mapping[str, str] | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """实盘买贵校准报告：lenses[透镜] = [pooled(按方向) + 通道] × 行情档 的桶级校准行。

    通道收录口径：enabled 在线，或注册表内且历史单 ≥30（下线通道也保留体检视野）；
    未注册的退役通道名不单列（其单仍计入 pooled 按方向合并）。
    """
    channel_direction = channel_direction or {}
    channel_display = channel_display or {}
    rows_by_lens: dict[str, list[dict[str, Any]]] = {lens: [] for lens, _, _, _ in CALIBRATION_LENSES}

    def _mode_direction(evs: list[sh.Event]) -> str | None:
        counts: dict[str, int] = {}
        for e in evs:
            d = e.get("direction")
            if d:
                counts[d] = counts.get(d, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def _append(evs: list[sh.Event], entity: str, kind: str, name: str,
                dirn: str | None, enabled: bool = True) -> None:
        for lens, lens_label, fn, labels in CALIBRATION_LENSES:
            for r in sh.calibration_rows(evs, lens, fn, labels):
                r.update(entity=entity, entity_kind=kind, display_name=name,
                         direction=dirn, lens_label=lens_label, enabled=enabled)
                rows_by_lens[lens].append(r)

    # 合并层：全系统按方向（跨通道，回答「什么行情下整体在买贵」）
    for dirn in ("DOWN", "UP"):
        pooled = [e for evs in live_ev.values() for e in evs if e.get("direction") == dirn]
        if pooled:
            _append(pooled, f"ALL_{dirn}", "pooled",
                    "全部DOWN单" if dirn == "DOWN" else "全部UP单", dirn)

    # 通道层
    for ch in sorted(live_ev):
        evs = live_ev[ch]
        if not evs:
            continue
        info = live_meta.get(ch)
        if info is None:
            continue  # 未注册的退役通道不单列
        if not (info.get("enabled") or len(evs) >= 30):
            continue
        dirn = channel_direction.get(ch) or _mode_direction(evs)
        _append(evs, ch, "channel",
                channel_display.get(ch) or ch, dirn, enabled=bool(info.get("enabled")))

    for lens_rows in rows_by_lens.values():
        lens_rows.sort(key=lambda r: (0 if r["entity_kind"] == "pooled" else 1,
                                      -(r["n"] or 0)))

    return {
        "generated_at": now_ms or int(time.time() * 1000),
        "lenses": rows_by_lens,
        "thresholds": {
            "bucket_min_n": sh.BUCKET_MIN_N, "bucket_red_min_n": sh.BUCKET_RED_MIN_N,
            "bucket_red_p_neg": sh.BUCKET_RED_P_NEG, "bucket_yellow_p_neg": sh.BUCKET_YELLOW_P_NEG,
        },
        "note": "体检仪表非门禁：红灯只告警不改护栏；桶级样本天然小，靠后验概率自保护。",
    }


def calibration_snapshot_rows(cal: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """校准报告 → SignalHealthSnapshot 持久化行（scope=calibration，每小时覆盖当日）。"""
    if not cal:
        return []
    rows = []
    for lens_rows in (cal.get("lenses") or {}).values():
        for r in lens_rows:
            rows.append({
                "scope": "calibration",
                "key": f"{r['lens']}|{r['entity']}|{r['bucket']}",
                "light": r["light"],
                "reasons": r["reasons"],
                "metrics": {k: r.get(k) for k in (
                    "n", "wins", "win_rate", "mean_breakeven", "edge",
                    "p_edge_negative", "avg_ev", "avg_price", "direction",
                    "lens", "lens_label", "bucket", "entity", "display_name")},
            })
    return rows


def detail_for(scope: str, key: str, row: Mapping[str, Any], evs: list[sh.Event],
               history: list[dict[str, Any]]) -> dict[str, Any]:
    price, hour = sh.price_buckets(evs), sh.hour_buckets(evs)
    t4 = sh.trend_4h_buckets(evs)
    t24 = sh.trend_24h_buckets(evs)
    vol = sh.volatility_buckets(evs)
    ustreak = sh.up_streak_buckets(evs)

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
        "up_streak_buckets": ustreak,
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
