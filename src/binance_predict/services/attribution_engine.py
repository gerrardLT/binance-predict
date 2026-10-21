"""归因引擎：触发 → 确定性事实卡 → 随机性前置裁决 → 多 LLM 假设 → 统计检验 → 假设库。

闭环定义（2026-10-19 方案）：
  记录（order_context_packs）→ 事实卡（attribution_stats 四问）
  → [统计裁决：RANDOM_CONSISTENT 即止，禁止 LLM 编故事]
  → 多 LLM 独立假设（claim 过闸）→ 假设绑定检验确定性执行
  → SUPPORTED/REJECTED 入库（REJECTED 进错误记忆）
  → 行动建议队列（人工经 /api/live/toggle 执行，LLM 永无执行权）。

触发器：连败事件（结算钩子检测）/ 红灯（signal_health 翻红）/ 周度复盘 / 手动。
每通道每日自动归因次数受 attribution_max_daily_runs 限制（手动不受限）。
"""
from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import math
from typing import Any, Awaitable, Callable

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.config.settings import settings
from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import (
    AttributionAnalysis,
    AttributionRun,
    Hypothesis,
    StrategyAdvice,
    TradeOrderModel,
)
from binance_predict.services import attribution_stats as astats
from binance_predict.services import channel_trend as ctrend
from binance_predict.services.llm_panel import (
    LLMPanel,
    STRATEGY_PROMPT_VERSION,
    _provider_entry,
    build_strategy_prompt,
    build_user_prompt,
    default_panel,
    verify_analysis,
    verify_strategy_advice,
)

STREAK_LOOKBACK = 50          # 触发检测回看的已结算订单数上限
MIN_BASE_N = 30               # 通道基准胜率最少历史单数
DEDUPE_RATIO = 0.72           # 假设文本相似度去重阈值
ENGINE_VERSION = "engine_v1"


# ======================================================================
# 纯函数层（测试直接覆盖）
# ======================================================================

def _ev_per_order(o: dict) -> float | None:
    """单笔费后 EV（pnl / 本金）；缺字段 → None。"""
    try:
        amount = float(o.get("amount_usdt") or 0)
        pnl = o.get("pnl")
        if amount <= 0 or pnl is None:
            return None
        return float(pnl) / amount
    except (TypeError, ValueError):
        return None


def _orders_to_events(orders: list[dict]) -> list[dict]:
    """归因订单 dict → signal_health/channel_trend 事件（升序）。

    EV=pnl/本金；保本线=均价/0.98（与 live_performance.break_even 估算口径同源）。
    """
    events: list[dict] = []
    for o in orders:
        ev = _ev_per_order(o)
        if ev is None:
            continue
        avg = o.get("avg_price")
        events.append({
            "ts": o.get("created_ts") or 0, "win": o.get("win"), "ev": ev,
            "be": (avg / 0.98) if avg else None, "q": avg,
        })
    return events


def _iso_to_ms(value: str | None) -> int:
    """ISO 时间串 → 毫秒；解析失败返回 0（追踪时按全量窗口回退）。"""
    if not value:
        return 0
    try:
        return int(dt.datetime.fromisoformat(value).timestamp() * 1000)
    except (TypeError, ValueError):
        return 0


def _entry_band(p: float | None) -> str:
    if p is None:
        return "unknown"
    lo = math.floor(p * 10) / 10
    return f"[{lo:.1f},{lo + 0.1:.1f})"


def compute_channel_facts(orders: list[dict], window_n: int | None = None,
                          current_params: dict | None = None) -> tuple[dict, set[str]]:
    """确定性事实卡（LLM 的唯一数字来源）。

    orders: 按时间升序的通道订单 dict 列表，字段：
      win, pnl, amount_usdt, direction, created_ts(ms), avg_price, quoted_avg_price
    current_params: 运行时通道参数快照（金额/日限/护栏/开关）——策略建议的
      action.from_value 必须与之精确匹配；归因主流程可不传（空字典）。
    base_rate 取窗口前全部历史胜率（无前视）；window_n = 检验窗口大小（默认 settings）。
    """
    n_win = settings.attribution_streak_window_n if window_n is None else window_n
    evs = [_ev_per_order(o) for o in orders]
    ev_seq = [e for e in evs if e is not None]
    hist = orders[:-n_win] if len(orders) > n_win else []
    base_rate = None
    hw = [o["win"] for o in hist if o.get("win") is not None]
    if len(hw) >= MIN_BASE_N:
        base_rate = sum(hw) / len(hw)
    recent = orders[-n_win:]
    recent_wins = [o.get("win") for o in recent]
    if base_rate is None:
        # 历史不足时以全序列为基准（并在卡内标注 base_n 不足）
        allw = [o.get("win") for o in orders if o.get("win") is not None]
        if len(allw) >= MIN_BASE_N:
            base_rate = sum(allw) / len(allw)

    sa = astats.streak_assessment(recent_wins, base_rate or 0.5,
                                  alpha=settings.attribution_streak_p) if base_rate else {
        "verdict": "INSUFFICIENT_DATA", "n": len([w for w in recent_wins if w is not None])}

    # Q1 同向敞口聚集：亏损单的日集中度 + 方向分布
    losses = [o for o in orders if o.get("win") is False]
    loss_day: dict[str, int] = {}
    for o in losses:
        d = dt.datetime.fromtimestamp((o.get("created_ts") or 0) / 1000, dt.timezone.utc).strftime("%m-%d")
        loss_day[d] = loss_day.get(d, 0) + 1
    top_days = sorted(loss_day.items(), key=lambda kv: -kv[1])[:3]
    dir_count: dict[str, int] = {}
    for o in losses:
        d = o.get("direction") or "?"
        dir_count[d] = dir_count.get(d, 0) + 1

    # Q2 边衰减：CUSUM + 前后半胜率
    wins_seq = [1.0 if o.get("win") else 0.0 for o in orders if o.get("win") is not None]
    half = len(wins_seq) // 2
    cusum = astats.cusum_changepoint(ev_seq)
    ev_boot = astats.block_bootstrap_mean_p(
        ev_seq, [o["created_ts"] for o, e in zip(orders, evs) if e is not None]
    ) if ev_seq else {"n": 0}

    # Q4 执行环境：滑点分布 + 入场价带分桶
    slips = []
    for o in orders:
        a, q = o.get("avg_price"), o.get("quoted_avg_price")
        if a and q and q > 0:
            slips.append((a / q - 1) * 10_000)
    bands: dict[str, dict] = {}
    for o, e in zip(orders, evs):
        if e is None:
            continue
        b = _entry_band(o.get("avg_price"))
        bands.setdefault(b, {"values": [], "timestamps_ms": []})
        bands[b]["values"].append(e)
        bands[b]["timestamps_ms"].append(o.get("created_ts") or 0)
    hour_groups: dict[str, dict] = {}
    for o, e in zip(orders, evs):
        if e is None:
            continue
        h = dt.datetime.fromtimestamp((o.get("created_ts") or 0) / 1000, dt.timezone.utc).strftime("%H")
        g = f"{int(h) // 4 * 4:02d}-{int(h) // 4 * 4 + 4:02d}"
        hour_groups.setdefault(g, {"values": [], "timestamps_ms": []})
        hour_groups[g]["values"].append(e)
        hour_groups[g]["timestamps_ms"].append(o.get("created_ts") or 0)

    # Q5 买贵 vs 信号失效分解（前后半窗口）：Δedge = Δ胜率 − Δ保本线。
    # 同一现象两种处置完全相反（调护栏 vs 停火），LLM 不准靠猜。
    def _wr_be(seq: list[dict]) -> tuple[float | None, float | None]:
        w = [o["win"] for o in seq if o.get("win") is not None]
        b = [o["avg_price"] / 0.98 for o in seq if o.get("avg_price")]
        return (round(sum(w) / len(w), 4) if w else None,
                round(sum(b) / len(b), 4) if b else None)

    _half = len(orders) // 2
    wr1, be1 = _wr_be(orders[:_half])
    wr2, be2 = _wr_be(orders[_half:])

    def _delta(a: float | None, b: float | None) -> float | None:
        return round(b - a, 4) if (a is not None and b is not None) else None

    _edge1 = (wr1 - be1) if (wr1 is not None and be1 is not None) else None
    _edge2 = (wr2 - be2) if (wr2 is not None and be2 is not None) else None
    edge_decomp = {
        "win_rate_delta": _delta(wr1, wr2),
        "mean_breakeven_delta": _delta(be1, be2),
        "edge_delta": _delta(_edge1, _edge2),
    }

    # 趋势分级（与体检/预警同源口径；覆盖不足时 L0 + 原因）
    _events = _orders_to_events(orders)
    if _events:
        _be_n = sum(1 for e in _events if e["be"] is not None)
        trend_payload = ctrend.trend_payload(
            orders[-1].get("channel") if orders else None,
            _events, _be_n / len(_events))
    else:
        trend_payload = {"level": "L0", "reasons": ["无已结算订单"], "coverage_ratio": 0.0}

    facts: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "channel": orders[-1].get("channel") if orders else None,
        "n_total": len(orders),
        "base_rate": round(base_rate, 4) if base_rate else None,
        "base_n": len(hw) if base_rate else 0,
        "streak": sa,
        "randomness_first": {
            "verdict": sa.get("verdict"),
            "note": "RANDOM_CONSISTENT=与随机波动一致，无归因证据；此项为一等假设",
        },
        "current_params": dict(current_params or {}),
        "edge_decomp": edge_decomp,
        "trend": trend_payload,
        "q1_exposure": {
            "n_losses": len(losses),
            "top_loss_days": [{"day": d, "losses": c} for d, c in top_days],
            "loss_directions": dir_count,
        },
        "q2_edge_decay": {
            "cusum": cusum,
            "win_rate_first_half": round(sum(wins_seq[:half]) / half, 4) if half else None,
            "win_rate_second_half": round(sum(wins_seq[half:]) / (len(wins_seq) - half), 4)
            if len(wins_seq) > half else None,
            "ev_block_bootstrap_all": ev_boot,
        },
        "q4_execution": {
            "slippage_bp_mean": round(sum(slips) / len(slips), 2) if slips else None,
            "slippage_bp_max": round(max(slips), 2) if slips else None,
            "n_with_slippage": len(slips),
        },
        "entry_band_ev": astats.bucket_compare(bands),
        "hour_bucket_ev": astats.bucket_compare(hour_groups),
    }
    evidence_ids = {
        "channel", "n_total", "base_rate", "base_n", "streak", "streak.verdict",
        "streak.binom_p_le", "streak.observed_rate", "streak.longest_losing_run",
        "randomness_first.verdict", "q1_exposure.n_losses", "q1_exposure.top_loss_days",
        "q1_exposure.loss_directions", "q2_edge_decay.cusum", "q2_edge_decay.cusum.delta",
        "q2_edge_decay.win_rate_first_half", "q2_edge_decay.win_rate_second_half",
        "q2_edge_decay.ev_block_bootstrap_all.p_le_zero", "q4_execution.slippage_bp_mean",
        "q4_execution.slippage_bp_max", "entry_band_ev", "hour_bucket_ev",
        # 策略建议（2026-10-22）：当前参数原值 / 买贵-失效分解 / 趋势分级
        "current_params.amount_usdt", "current_params.max_daily_orders",
        "current_params.max_exec_price", "current_params.enabled",
        "edge_decomp.win_rate_delta", "edge_decomp.mean_breakeven_delta",
        "edge_decomp.edge_delta",
        "trend.level", "trend.cusum", "trend.reasons", "trend.coverage_ratio",
        "trend.n", "trend.win_rate", "trend.p_edge_negative", "trend.mean_breakeven",
    }
    return facts, evidence_ids


def execute_test(name: str, params: dict, orders: list[dict]) -> dict:
    """确定性执行检验菜单中的一项（LLM 只能点名，不能自造）。

    返回 {"name", "result", "significant", "verdict_note"}；不可执行 → verdict UNTESTABLE。
    """
    out: dict[str, Any] = {"name": name, "params": params or {}}
    if name == "streak_binomial":
        n = int(params.get("n") or settings.attribution_streak_window_n)
        recent = [o.get("win") for o in orders[-n:]]
        hw = [o["win"] for o in orders[:-n] if o.get("win") is not None] if len(orders) > n else \
             [o["win"] for o in orders if o.get("win") is not None]
        base = sum(hw) / len(hw) if len(hw) >= MIN_BASE_N else None
        if base is None:
            out.update({"result": None, "significant": False, "verdict": "UNTESTABLE",
                        "note": "基准胜率样本不足"})
            return out
        res = astats.streak_assessment(recent, base, alpha=settings.attribution_streak_p)
        out.update({"result": res, "significant": res.get("verdict") != "RANDOM_CONSISTENT",
                    "verdict": res.get("verdict")})
        return out
    if name == "ev_block_bootstrap":
        subset = params.get("filter")
        direction = str(params.get("direction") or "positive")
        seq = list(orders)
        if subset == "recent_n":
            seq = orders[-int(params.get("n") or settings.attribution_streak_window_n):]
        rows = [(o, _ev_per_order(o)) for o in seq]
        vals = [e for _, e in rows if e is not None]
        tss = [o.get("created_ts") or 0 for o, e in rows if e is not None]
        res = astats.block_bootstrap_mean_p(vals, tss)
        p_le = res.get("p_le_zero")
        if direction == "negative":
            # p_le_zero = P(mean≤0)；显著为负 ⇔ P(mean≥0) 很小 ⇔ p_le_zero 接近 1
            significant = p_le is not None and p_le > 0.95
            out.update({"result": res, "direction": "negative",
                        "p_ge_zero": round(1.0 - p_le, 4) if p_le is not None else None,
                        "significant": significant,
                        "verdict": "SIGNIFICANT_NEGATIVE" if significant else "NOT_SIGNIFICANT"})
            return out
        out.update({"result": res, "direction": "positive",
                    "significant": bool(res.get("significant_positive")),
                    "verdict": "SIGNIFICANT_POSITIVE" if res.get("significant_positive") else "NOT_SIGNIFICANT"})
        return out
    if name == "bucket_bonferroni":
        dim = params.get("dim", "entry_band")
        groups: dict[str, dict] = {}
        for o in orders:
            e = _ev_per_order(o)
            if e is None:
                continue
            if dim == "hour":
                h = dt.datetime.fromtimestamp((o.get("created_ts") or 0) / 1000, dt.timezone.utc).hour
                key = f"{h // 4 * 4:02d}-{h // 4 * 4 + 4:02d}"
            else:
                key = _entry_band(o.get("avg_price"))
            groups.setdefault(key, {"values": [], "timestamps_ms": []})
            groups[key]["values"].append(e)
            groups[key]["timestamps_ms"].append(o.get("created_ts") or 0)
        res = astats.bucket_compare(groups)
        sig = [k for k, b in (res.get("buckets") or {}).items()
               if b.get("p_bonferroni") is not None and b["p_bonferroni"] < 0.05]
        out.update({"result": res, "significant_buckets": sig, "significant": bool(sig),
                    "verdict": "DIFFERENTIATED" if sig else "NOT_SIGNIFICANT"})
        return out
    if name == "direction_clustering":
        losses = [o for o in orders if o.get("win") is False]
        if not losses:
            out.update({"result": {"n_losses": 0}, "significant": False, "verdict": "UNTESTABLE"})
            return out
        day: dict[str, int] = {}
        for o in losses:
            d = dt.datetime.fromtimestamp((o.get("created_ts") or 0) / 1000, dt.timezone.utc).strftime("%m-%d")
            day[d] = day.get(d, 0) + 1
        top_day, top_c = max(day.items(), key=lambda kv: kv[1])
        share = top_c / len(losses)
        res = {"n_losses": len(losses), "top_day": top_day, "top_day_share": round(share, 3),
               "n_days": len(day)}
        out.update({"result": res, "significant": share >= 0.4,
                    "verdict": "CLUSTERED" if share >= 0.4 else "DISPERSED"})
        return out
    if name == "cusum_changepoint":
        ev_seq = [e for e in (_ev_per_order(o) for o in orders) if e is not None]
        res = astats.cusum_changepoint(ev_seq)
        delta = res.get("delta")
        out.update({"result": res, "significant": abs(delta or 0) >= 0.1,
                    "verdict": "BREAK_DETECTED" if abs(delta or 0) >= 0.1 else "NO_BREAK"})
        return out
    if name == "composition_shift":
        out.update({"result": None, "significant": False, "verdict": "UNTESTABLE",
                    "note": "触发特征分布对比需影子表 join，v1 未实现"})
        return out
    out.update({"result": None, "significant": False, "verdict": "UNKNOWN_TEST"})
    return out


def similar(a: str, b: str) -> bool:
    """假设文本相似度（包含或 SequenceMatcher）。"""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= DEDUPE_RATIO


def consensus_map(per_model: list[list[dict]]) -> dict:
    """跨模型共识/分歧（共识≠正确，分歧是路由信号）。"""
    stmt_models: dict[str, set[int]] = {}
    for mi, hyps in enumerate(per_model):
        for h in hyps:
            key = None
            for k in stmt_models:
                if similar(k, h["statement"]):
                    key = k
                    break
            if key is None:
                stmt_models[h["statement"]] = {mi}
            else:
                stmt_models[key].add(mi)
    consensus = [{"statement": s, "models": sorted(m)} for s, m in stmt_models.items() if len(m) >= 2]
    divergent = [{"statement": s, "models": sorted(m)} for s, m in stmt_models.items() if len(m) == 1]
    return {"n_models": len(per_model), "consensus": consensus, "divergent": divergent}


# ======================================================================
# 引擎（DB 编排层）
# ======================================================================

class AttributionEngine:
    def __init__(self, panel: LLMPanel | None = None) -> None:
        self.panel = panel or LLMPanel()
        self._running: set[str] = set()   # 防同通道并发 run
        # L2 趋势预警回调（main.py 注入微信推送 + 策略建议触发）。
        # 引擎不直接 import notifier，保持分析层与通知层解耦；异常只记日志。
        self.on_trend_alert: Callable[[str, dict], Awaitable[None]] | None = None

    # ---------------- 触发入口 ----------------

    async def on_order_settled(self, channel: str | None) -> None:
        """结算钩子：连败检测（fire-and-forget 安全壳由调用方保证）。"""
        if not channel or not settings.attribution_enabled:
            return
        try:
            await self._maybe_trigger_streak(channel)
        except Exception as exc:
            logger.warning("归因引擎：连败触发检测失败 | {} | {}", channel, exc)

    async def on_red_light(self, channel: str) -> None:
        """红灯钩子（signal_health 翻红时调用）。"""
        if not settings.attribution_enabled:
            return
        await self._run_guarded("RED_LIGHT", channel)

    async def _maybe_trigger_streak(self, channel: str) -> None:
        orders = await self._load_channel_orders(channel, limit=STREAK_LOOKBACK)
        # 趋势检测（2026-10-22）：与连败检测共用同一订单加载，零额外查询；
        # 放在连败窗口门之前，避免 streak_window_n 调大时趋势检测被跳过。
        await self._maybe_emit_trend(channel, orders)
        if len(orders) < settings.attribution_streak_window_n:
            return
        facts, _ = compute_channel_facts(orders)
        verdict = (facts.get("streak") or {}).get("verdict")
        if verdict not in ("ANOMALOUS_LOSING", "ANOMALOUS_WINNING"):
            return
        if not await self._budget_ok(channel):
            return
        await self._run_guarded("STREAK", channel)

    # ---------------- 主流程 ----------------

    async def _maybe_emit_trend(self, channel: str, orders: list[dict]) -> None:
        """L2 趋势预警回调（L1 只进体检面板，L3 走既有红灯链路）。

        orders 口径与 compute_channel_facts 一致（升序）；EV=pnl/本金，
        保本线用 avg_price/0.98（与 live_performance.break_even 估算口径同源）。
        """
        if self.on_trend_alert is None or not settings.strategy_advice_enabled:
            return
        events: list[dict] = []
        for o in orders:
            ev = _ev_per_order(o)
            if ev is None:
                continue
            avg = o.get("avg_price")
            events.append({
                "ts": o.get("created_ts") or 0, "win": o.get("win"), "ev": ev,
                "be": (avg / 0.98) if avg else None, "q": avg,
            })
        if len(events) < ctrend.TREND_MIN_N:
            return
        be_n = sum(1 for e in events if e["be"] is not None)
        payload = ctrend.trend_payload(channel, events, be_n / len(events))
        if payload.get("level") != "L2":
            return
        try:
            await self.on_trend_alert(channel, payload)
        except Exception as exc:
            logger.warning("归因引擎：趋势预警回调失败 | {} | {}", channel, exc)

    async def _run_guarded(self, trigger: str, channel: str) -> None:
        key = f"{trigger}:{channel}"
        if key in self._running:
            return
        self._running.add(key)
        try:
            await self.run_attribution(trigger, channel)
        finally:
            self._running.discard(key)

    async def run_attribution(self, trigger: str, channel: str) -> AttributionRun | None:
        """完整归因流程。手动/周报触发也走这里。"""
        async with async_session_factory() as db:
            run = AttributionRun(trigger_type=trigger, channel=channel,
                                 status="RUNNING", scope={})
            db.add(run)
            await db.commit()
            await db.refresh(run)
            run_id = run.id
        try:
            result = await self._execute(run_id, trigger, channel)
            return result
        except Exception as exc:
            logger.warning("归因运行失败 | run={} | {} | {}", run_id, channel, exc)
            async with async_session_factory() as db:
                r = await db.get(AttributionRun, run_id)
                if r:
                    r.status, r.error = "FAILED", str(exc)[:500]
                    r.finished_at = dt.datetime.now(dt.timezone.utc)
                    await db.commit()
            return None

    async def _execute(self, run_id: int, trigger: str, channel: str) -> AttributionRun | None:
        orders = await self._load_channel_orders(channel, limit=200)
        if not orders:
            async with async_session_factory() as db:
                r = await db.get(AttributionRun, run_id)
                if r:
                    r.status, r.verdict, r.summary = "DONE", "INSUFFICIENT_DATA", "无订单数据"
                    r.finished_at = dt.datetime.now(dt.timezone.utc)
                    await db.commit()
                return r
        facts, evidence_ids = compute_channel_facts(orders)
        streak_verdict = (facts.get("streak") or {}).get("verdict")

        async with async_session_factory() as db:
            r = await db.get(AttributionRun, run_id)
            if r is None:
                return None
            r.scope = {"n_orders": len(orders), "recent_n": facts["streak"].get("n"),
                       "order_ids": [o.get("id") for o in orders[-settings.attribution_streak_window_n:]]}
            r.facts = facts

            # ---- 随机性前置裁决：一致即止，禁止 LLM 编故事 ----
            if streak_verdict in ("RANDOM_CONSISTENT", "INSUFFICIENT_DATA") and trigger != "MANUAL":
                r.status, r.verdict = "DONE", streak_verdict
                r.summary = ("随机性检验：与通道基准胜率下的抽样波动一致（或样本不足），"
                             "无归因证据——深归因被前置门拦截。")
                r.finished_at = dt.datetime.now(dt.timezone.utc)
                await db.commit()
                return r

            r.verdict = "ANOMALOUS" if streak_verdict in (
                "ANOMALOUS_LOSING", "ANOMALOUS_WINNING") else streak_verdict
            await db.commit()

        # ---- 多 LLM 假设（claim 过闸）----
        analyses_rows: list[dict] = []
        per_model: list[list[dict]] = []
        if settings.attribution_llm_enabled and self.panel.panel:
            ledger = await self._load_ledger(channel)
            error_memory = [h["statement"] for h in ledger if h.get("in_error_memory")]
            prompt = build_user_prompt(facts, channel, ledger, error_memory)
            model_outputs = await self.panel.analyze(prompt)
            for mo in model_outputs:
                if mo.get("error") or not mo.get("parsed"):
                    analyses_rows.append({"model": f"{mo['provider']}:{mo['model']}",
                                          "family": mo.get("family", "?"),
                                          "raw_output": mo, "hypotheses": [],
                                          "rejected_claims": [{"reason": mo.get("error")}],
                                          "error": mo.get("error")})
                    continue
                verified, rejected = verify_analysis(mo["parsed"], facts, evidence_ids)
                per_model.append(verified)
                analyses_rows.append({"model": f"{mo['provider']}:{mo['model']}",
                                      "family": mo.get("family", "?"),
                                      "raw_output": {"narrative": mo["parsed"].get("narrative")},
                                      "hypotheses": verified,
                                      "rejected_claims": rejected,
                                      "error": None})
        else:
            analyses_rows.append({"model": "none", "family": "none", "raw_output": {},
                                  "hypotheses": [], "rejected_claims": [],
                                  "error": "llm_disabled"})

        # ---- 假设入库 + 检验执行 ----
        test_results: list[dict] = []
        async with async_session_factory() as db:
            for a in analyses_rows:
                db.add(AttributionAnalysis(
                    run_id=run_id, model=a["model"], family=a["family"],
                    raw_output=a["raw_output"], hypotheses=a["hypotheses"],
                    rejected_claims=a["rejected_claims"], error=a["error"]))
            await db.flush()

            # 显式取列构造 dict（SQLAlchemy 2.0 ORM 对象不可 dict()/迭代展开）
            _rows = (await db.execute(sa_select(Hypothesis))).scalars().all()
            existing = [{"id": h.id, "channel": h.channel, "statement": h.statement,
                         "status": h.status, "support_count": h.support_count,
                         "reject_count": h.reject_count} for h in _rows]
            for mi, hyps in enumerate(per_model):
                for h in hyps:
                    hyp_row = None
                    for ex in existing:
                        same_channel = (ex.get("channel") in (None, channel)
                                        or channel is None)
                        if same_channel and similar(ex["statement"], h["statement"]):
                            hyp_row = ex
                            break
                    t = execute_test(h["test"]["name"], h["test"].get("params") or {}, orders)
                    test_results.append({"statement": h["statement"],
                                         "category": h["category"], "test": t})
                    if hyp_row is None:
                        hyp_row = {"id": None, "support_count": 0, "reject_count": 0}
                        row = Hypothesis(
                            channel=channel, category=h["category"], statement=h["statement"],
                            status="CANDIDATE", test_spec=h["test"], last_test=t,
                            first_run_id=run_id,
                            action_suggestion=h.get("action_suggestion"),
                            action_status="PENDING" if h.get("action_suggestion") else None,
                        )
                        db.add(row)
                        await db.flush()
                        hyp_row.update({"id": row.id, "support_count": 0, "reject_count": 0,
                                        "statement": h["statement"], "channel": channel})
                        existing.append(hyp_row)
                        row_id = row.id
                    else:
                        row = await db.get(Hypothesis, hyp_row["id"])
                        row_id = hyp_row["id"]
                    if row is None:
                        continue
                    row.last_test = t
                    row.test_spec = h["test"]
                    if t.get("significant"):
                        row.support_count += 1
                        row.status = "SUPPORTED"
                        row.in_error_memory = False
                    else:
                        row.reject_count += 1
                        if t.get("verdict") == "UNTESTABLE":
                            row.status = row.status if row.status == "SUPPORTED" else "CANDIDATE"
                        else:
                            row.status = "REJECTED"
                            row.in_error_memory = True

            cons = consensus_map(per_model)
            summary = self._build_summary(channel, facts, streak_verdict, cons, test_results)
            r = await db.get(AttributionRun, run_id)
            r.status = "DONE"
            r.summary = summary
            r.facts = dict(facts, consensus=cons, test_results=test_results)
            r.finished_at = dt.datetime.now(dt.timezone.utc)
            await db.commit()
            return r

    # ---------------- 辅助 ----------------

    def _build_summary(self, channel: str, facts: dict, streak_verdict: str,
                       cons: dict, test_results: list[dict]) -> str:
        lines = [f"{channel}：随机性检验={streak_verdict}；"
                 f"近段胜率={facts['streak'].get('observed_rate')} vs 基准={facts.get('base_rate')}"]
        for tr in test_results[:6]:
            lines.append(f"检验 {tr['test']['name']} → {tr['test'].get('verdict')}"
                         f"（{'显著' if tr['test'].get('significant') else '不显著'}）：{tr['statement'][:60]}")
        if cons.get("consensus"):
            lines.append(f"双模型共识假设 {len(cons['consensus'])} 条；分歧 {len(cons['divergent'])} 条"
                         f"（分歧=统计复核信号，非投票）")
        return "\n".join(lines)

    async def _load_channel_orders(self, channel: str, limit: int) -> list[dict]:
        async with async_session_factory() as db:
            rows = (await db.execute(sa_select(
                TradeOrderModel.id, TradeOrderModel.signal_version,
                TradeOrderModel.direction, TradeOrderModel.win, TradeOrderModel.pnl,
                TradeOrderModel.amount_in, TradeOrderModel.created_at,
                TradeOrderModel.quote_json,
            ).where(
                TradeOrderModel.signal_version == channel,
                TradeOrderModel.status == "FILLED",
                TradeOrderModel.win.isnot(None),
            ).order_by(TradeOrderModel.created_at.desc()).limit(limit))).all()
            rows = list(reversed(rows))
            out = []
            for r in rows:
                q = r.quote_json or {}
                try:
                    amount = float(r.amount_in or 0) / 1e18
                except (TypeError, ValueError):
                    amount = 0.0

                def _pf(v):
                    try:
                        f = float(v)
                        return f if math.isfinite(f) else None
                    except (TypeError, ValueError):
                        return None
                out.append({
                    "id": r.id, "channel": r.signal_version, "direction": r.direction,
                    "win": r.win, "pnl": r.pnl, "amount_usdt": amount,
                    "created_ts": int(r.created_at.timestamp() * 1000) if r.created_at else 0,
                    "avg_price": _pf(q.get("averagePrice")),
                    "quoted_avg_price": _pf(q.get("quotedAvgPrice")),
                })
            return out

    async def _load_ledger(self, channel: str) -> list[dict]:
        async with async_session_factory() as db:
            rows = (await db.execute(sa_select(
                Hypothesis.statement, Hypothesis.status, Hypothesis.in_error_memory,
            ).where(Hypothesis.channel.in_([channel])).limit(50))).all()
            return [{"statement": s, "status": st, "in_error_memory": em}
                    for s, st, em in rows]

    async def _budget_ok(self, channel: str) -> bool:
        """每通道每日自动 run 次数限制（手动触发不查此函数）。"""
        today = dt.datetime.now(dt.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        async with async_session_factory() as db:
            n = (await db.execute(sa_select(AttributionRun.id).where(
                AttributionRun.channel == channel,
                AttributionRun.created_at >= today,
                AttributionRun.trigger_type != "MANUAL",
            ))).scalars()
            return len(list(n)) < settings.attribution_max_daily_runs

    async def weekly_review(self, channels: list[str]) -> int:
        """周度复盘：对每个通道跑一次（随机性门同样生效）。"""
        n = 0
        for ch in channels:
            try:
                r = await self.run_attribution("WEEKLY", ch)
                n += 1 if r else 0
            except Exception as exc:
                logger.warning("周报归因失败 | {} | {}", ch, exc)
        return n

    # ---------------- 策略顾问（2026-10-22） ----------------

    def _strategy_panel(self) -> LLMPanel:
        """策略建议模型：默认单模型控成本（settings 可覆盖为多模型）。"""
        override = settings.strategy_advice_models.strip()
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
                return LLMPanel(out)
        full = default_panel()
        return LLMPanel(full[:1] if full else [])

    async def _advice_budget_ok(self, channel: str) -> bool:
        """每通道每日自动建议次数限制（手动触发不查此函数）。"""
        today = dt.datetime.now(dt.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        async with async_session_factory() as db:
            n = (await db.execute(sa_select(StrategyAdvice.id).where(
                StrategyAdvice.channel == channel,
                StrategyAdvice.created_at >= today,
                StrategyAdvice.trigger_type != "MANUAL",
            ))).scalars()
            return len(list(n)) < settings.strategy_advice_max_daily

    async def run_strategy_advice(self, trigger: str, channel: str, *,
                                  manual: bool = False,
                                  current_params: dict | None = None) -> dict | None:
        """完整策略建议流程（L2 趋势 / 红灯 / 手动共用；与 run_attribution 并列）。

        随机性前置门对 MANUAL 同样生效：与随机一致时不调模型、直接落 HOLD，
        防止用户在纯噪声上反复点出「看起来有道理」的建议（也省 token）。
        LLM 永无执行权：advice_spec.applyable 只表示允许前端展示一键应用按钮。
        """
        if not settings.strategy_advice_enabled:
            return None
        key = f"advice:{channel}"
        if key in self._running:
            return None
        self._running.add(key)
        try:
            if not manual and not await self._advice_budget_ok(channel):
                return None
            orders = await self._load_channel_orders(channel, limit=200)
            if not orders:
                return None
            facts, evidence_ids = compute_channel_facts(
                orders, current_params=current_params)
            # 预置检验：给 LLM 可引用的显著证据，也作为 verify 核对 supporting_test 的依据
            pre_tests: list[dict] = []
            for tname, tparams in (("streak_binomial", {}),
                                   ("ev_block_bootstrap", {"direction": "negative"}),
                                   ("cusum_changepoint", {})):
                pre_tests.append(execute_test(tname, tparams, orders))
            facts["pre_tests"] = pre_tests
            streak_verdict = (facts.get("streak") or {}).get("verdict")

            rejected: list[dict] = []
            if streak_verdict in ("RANDOM_CONSISTENT", "INSUFFICIENT_DATA"):
                # 随机性前置门：与随机一致 → 不调模型，直接 HOLD（一等输出）
                advice_spec = {
                    "schema": "strategy_advice_v1",
                    "channel": channel,
                    "verdict": "HOLD",
                    "diagnosis": {
                        "primary_cause": "RANDOM",
                        "evidence": [],
                        "supporting_test": {"name": "streak_binomial",
                                            "verdict": streak_verdict,
                                            "significant": False},
                        "randomness_check": {"verdict": streak_verdict},
                    },
                    "action": None,
                    "confidence": None,
                    "counter_hypothesis": "观察到的波动可能只是通道基准胜率下的正常抽样噪声",
                    "kill_criteria": "若后续检验出现显著负 EV（direction=negative）"
                                     "或 CUSUM 越界则重新评估",
                    "expected_effect": {},
                    "requires_human_review": True,
                    "clamped": False,
                    "applyable": False,
                    "system_skipped_llm": True,
                }
                rejected.append({"reason": "random_gate_skipped_llm",
                                 "verdict": streak_verdict})
            else:
                ledger = await self._load_ledger(channel)
                error_memory = [h["statement"] for h in ledger if h.get("in_error_memory")]
                prompt = build_strategy_prompt(
                    facts, channel, ledger, error_memory, current_params=current_params)
                outputs = await self._strategy_panel().analyze(prompt)
                advice_spec = None
                llm_errors: list[dict] = []
                for mo in outputs:
                    if mo.get("error") or not mo.get("parsed"):
                        llm_errors.append({
                            "model": f"{mo.get('provider')}:{mo.get('model')}",
                            "error": mo.get("error")})
                        continue
                    advice_spec, rejected = verify_strategy_advice(
                        mo["parsed"], facts, evidence_ids)
                    if advice_spec is not None:
                        break
                if advice_spec is None:
                    rejected = llm_errors + rejected

            trend = facts.get("trend") or {}
            pre_metrics = {
                "n": trend.get("n"), "win_rate": trend.get("win_rate"),
                "mean_breakeven": trend.get("mean_breakeven"),
                "edge": trend.get("edge"),
                "p_edge_negative": trend.get("p_edge_negative"),
                "rolling20_win_rate": trend.get("rolling20_win_rate"),
                "cusum": trend.get("cusum"), "level": trend.get("level"),
            }
            async with async_session_factory() as db:
                row = StrategyAdvice(
                    channel=channel, trigger_type=trigger,
                    prompt_version=STRATEGY_PROMPT_VERSION,
                    advice_spec=advice_spec or {},
                    rejected=rejected,
                    pre_metrics=pre_metrics,
                    status="PENDING",
                    application_log=[],
                    tracking={"target_n": settings.strategy_advice_tracking_n,
                              "observed_n": 0, "verdict": None},
                )
                db.add(row)
                await db.commit()
                await db.refresh(row)
                return self._advice_dict(row)
        except Exception as exc:
            logger.warning("策略建议运行失败 | {} | {}", channel, exc)
            return None
        finally:
            self._running.discard(key)

    async def update_advice_tracking(self, channel: str) -> None:
        """结算钩子：已应用建议的 30 单效果追踪（fire-and-forget 安全壳）。

        判据：p_edge_negative 从 ≥0.90 降至 <0.75 → IMPROVED；满窗仍 ≥0.90 →
        WORSE；其余 NO_CHANGE。⚠️ 30 单胜率 95% CI 约 ±18pp——只能排除「更糟」，
        不能证明「变好」；严禁把均值回归当成「调参有效」。
        """
        if not settings.strategy_advice_enabled:
            return
        try:
            async with async_session_factory() as db:
                row = (await db.execute(sa_select(StrategyAdvice).where(
                    StrategyAdvice.channel == channel,
                    StrategyAdvice.status == "APPLIED",
                ).order_by(StrategyAdvice.updated_at.desc()).limit(1))).scalar_one_or_none()
                if row is None:
                    return
                tr = dict(row.tracking or {})
                target = int(tr.get("target_n") or settings.strategy_advice_tracking_n)
                if int(tr.get("observed_n") or 0) >= target:
                    return
                applied_ms = 0
                for entry in reversed(row.application_log or []):
                    applied_ms = _iso_to_ms(entry.get("applied_at"))
                    if applied_ms:
                        break
                orders = await self._load_channel_orders(channel, limit=200)
                events = _orders_to_events(orders)
                if applied_ms:
                    events = [e for e in events if int(e.get("ts") or 0) >= applied_ms]
                tr["observed_n"] = len(events)
                if len(events) >= target:
                    window = events[-target:]
                    be_n = sum(1 for e in window if e["be"] is not None)
                    tp = ctrend.trend_payload(channel, window, be_n / target)
                    p_neg = tp.get("p_edge_negative")
                    baseline = (row.pre_metrics or {}).get("p_edge_negative")
                    if (p_neg is not None and p_neg < 0.75
                            and baseline is not None and baseline >= 0.90):
                        tr["verdict"] = "IMPROVED"
                    elif p_neg is not None and p_neg >= 0.90:
                        tr["verdict"] = "WORSE"
                    else:
                        tr["verdict"] = "NO_CHANGE"
                    tr["eval"] = {k: tp.get(k) for k in (
                        "n", "win_rate", "edge", "p_edge_negative", "rolling20_win_rate")}
                    tr["note"] = "30 单胜率 CI≈±18pp：只能排除更糟，不能证明变好。"
                row.tracking = tr
                await db.commit()
        except Exception as exc:
            logger.warning("策略建议效果追踪失败 | {} | {}", channel, exc)

    @staticmethod
    def _advice_dict(row: StrategyAdvice) -> dict:
        return {
            "id": row.id, "channel": row.channel,
            "trigger_type": row.trigger_type, "prompt_version": row.prompt_version,
            "advice_spec": row.advice_spec, "rejected": row.rejected,
            "pre_metrics": row.pre_metrics, "status": row.status,
            "application_log": row.application_log, "tracking": row.tracking,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


engine = AttributionEngine()
