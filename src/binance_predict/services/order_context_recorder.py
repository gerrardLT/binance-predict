"""实盘订单全场景证据包组装器（LLM 归因系统的记录层）。

职责：订单结算后异步组装「前/中/后」三维上下文证据包，落 order_context_packs。
安全边界（铁律）：
- 全部事后组装，绝不进入下单/结算决策路径；结算钩子只 enqueue fire-and-forget；
- 任一维度缺失降级 PARTIAL（missing 列出），组装异常不抛回调用方；
- 证据包只增不改：同一订单重复组装覆盖更新（幂等，回填/重跑安全）。

数据源（全部只读）：
- trade_orders / shadow_execution_assessments：订单与执行快照
- prediction_market_samples：5s/15s 报价轨迹（永久保留）
- K 线：注入的 kline_fetcher（生产=collector.fetch_klines_ending_at，测试=替身），
  组装时冻结进 pack——避免依赖交易所 REST 历史可用性
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import OrderContextPack, TradeOrderModel
from binance_predict.services import attribution_stats as astats

PACK_SCHEMA_VERSION = "v2"
PRE_WINDOW_MS = 30 * 60_000          # 前段报价轨迹长度：窗口前 30 分钟
POST_WINDOW_MS = 5 * 60_000          # 后段：结算（窗口结束）后 5 分钟
KLINE_1M_BEFORE = 60                 # 决策时刻前的 1m K线根数
KLINE_5M_BEFORE = 50                 # 5m 根数（够算 ret_4h 与 ATR）
# v2：按「决策时刻」再切一组 K 线，供前端场景卡（24×5m + 15×1m）使用。
# 窗内下单的通道（吸收跟随 +310s、情绪错位 +150s、首触窗内触发）必须用这组，
# 否则画面会比真实决策时刻早 2~5 分钟。
KLINE_1M_DECISION = 30
KLINE_5M_DECISION = 50
MAX_QUOTE_POINTS = 480               # 单段报价轨迹最大点数（超出等距抽稀）

KlineFetcher = Callable[[str, int, int], Awaitable[list[dict] | None]]
# quote_prober(token_id, side, amounts) -> dict（只读报价梯度；默认 None=不采）
QuoteProber = Callable[[str, str, list[float]], Awaitable[dict]]


def _downsample(points: list[dict], cap: int = MAX_QUOTE_POINTS) -> list[dict]:
    if len(points) <= cap:
        return points
    step = math.ceil(len(points) / cap)
    out = points[::step]
    if out and out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def _kline_brief(bars: list[dict] | None) -> list[dict]:
    """K线冻结为紧凑数组 [[ts,o,h,l,c,v], ...]（数值截断控制包体）。"""
    out = []
    for b in bars or []:
        try:
            out.append([
                int(b["open_time"]),
                float(b["open"]), float(b["high"]), float(b["low"]), float(b["close"]),
                round(float(b.get("volume") or 0.0), 4),
            ])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _compute_regime(klines_1m: list[dict], klines_5m: list[dict]) -> dict:
    """决策时刻的市场体制（纯计算，无前视：只用窗口开始前已收盘的 K 线）。"""
    regime: dict[str, float | None] = {
        "ret_1h": None, "ret_4h": None, "atr14_1m_bp": None, "rv_1h_bp": None,
    }
    try:
        if klines_1m and len(klines_1m) >= 15:
            closes = [float(b["close"]) for b in klines_1m]
            regime["ret_1h"] = round((closes[-1] / closes[-min(60, len(closes))] - 1) * 100, 4)
            trs = []
            for i in range(1, len(klines_1m)):
                h, l, c_prev = (float(klines_1m[i]["high"]), float(klines_1m[i]["low"]),
                                float(klines_1m[i - 1]["close"]))
                trs.append(max(h - l, abs(h - c_prev), abs(l - c_prev)))
            if trs:
                regime["atr14_1m_bp"] = round(sum(trs[-14:]) / min(14, len(trs)) / closes[-1] * 10_000, 2)
            rets = [(closes[i] / closes[i - 1] - 1) for i in range(max(1, len(closes) - 60), len(closes))]
            if rets:
                regime["rv_1h_bp"] = round(math.sqrt(sum(r * r for r in rets) / len(rets)) * 10_000, 2)
        if klines_5m and len(klines_5m) >= 10:
            closes5 = [float(b["close"]) for b in klines_5m]
            regime["ret_4h"] = round((closes5[-1] / closes5[-min(48, len(closes5))] - 1) * 100, 4)
    except Exception:
        pass
    return regime


def _resolve_decision_ts(
    order: dict, assessment: dict | None, window_start: int,
) -> tuple[int, str]:
    """解析决策时刻（ms）与来源。

    优先级：报价响应时间 > assessment 报价/触发时刻 > 订单落库时间 > 窗口开始。
    窗内下单的通道（吸收跟随/情绪错位/首触）必须用决策时刻而不是 window_start，
    否则场景卡会整体左移 2~5 分钟。
    """
    quote = order.get("quote_json") or {}
    candidates = (
        (quote.get("timestamp"), "quote"),
        ((assessment or {}).get("quote_ts"), "assessment_quote"),
        ((assessment or {}).get("trigger_ts"), "assessment_trigger"),
    )
    for value, source in candidates:
        try:
            ts = int(value)
        except (TypeError, ValueError):
            continue
        if ts > 0:
            return ts, source
    created = order.get("created_at")
    if created:
        try:
            return int(datetime.fromisoformat(created).timestamp() * 1000), "created_at"
        except (TypeError, ValueError):
            pass
    return int(window_start or 0), "window_start"


def assemble_pack(
    order: dict,
    assessment: dict | None,
    quotes_pre: list[dict],
    quotes_post: list[dict],
    klines_1m_before: list[dict] | None,
    klines_5m_before: list[dict] | None,
    klines_1m_window: list[dict] | None,
    siblings: list[dict],
    channel_prior: dict,
    *,
    klines_1m_decision: list[dict] | None = None,
    klines_5m_decision: list[dict] | None = None,
    decision_ts: int | None = None,
    decision_source: str | None = None,
) -> tuple[dict, list[str]]:
    """纯函数组装证据包（无 IO，测试直接喂 dict）。返回 (pack, missing)。

    v2 新增：`pre.klines_1m_decision / klines_5m_decision`——按真实决策时刻切的
    K 线（已收盘根），供前端场景卡使用；缺失不影响 status（不写入 missing）。
    """
    missing: list[str] = []
    pack: dict[str, Any] = {"schema_version": PACK_SCHEMA_VERSION}

    # ---- 前：决策时刻市场状态 ----
    k1d = _kline_brief(klines_1m_decision)
    k5d = _kline_brief(klines_5m_decision)
    k1b = _kline_brief(klines_1m_before)
    k5b = _kline_brief(klines_5m_before)
    ref_price = k1d[-1][4] if k1d else (k1b[-1][4] if k1b else None)
    pack["pre"] = {
        "klines_1m": k1b,
        "klines_5m": k5b,
        "klines_1m_decision": k1d,
        "klines_5m_decision": k5d,
        "decision_ts": decision_ts,
        "decision_source": decision_source,
        "decision_slice": "ok" if (k1d and k5d) else "missing",
        "ref_price": ref_price,
        "quotes": quotes_pre,
        "regime": _compute_regime(klines_1m_before or [], klines_5m_before or []),
        "channel_state": channel_prior,
    }
    if not k1b:
        missing.append("klines_1m_before")
    if not k5b:
        missing.append("klines_5m_before")
    if not quotes_pre:
        missing.append("quotes_pre")

    # ---- 中：执行过程 ----
    quote_json = order.get("quote_json") or {}
    pack["exec"] = {
        "order": {
            k: order.get(k) for k in (
                "id", "signal_version", "direction", "side", "status", "market_period",
                "window_start", "amount_in", "settle_outcome", "win", "pnl",
                "settle_price", "error_message", "created_at", "settled_at",
            )
        },
        "assessment": assessment,
        "quote": quote_json,
        "timeline": {
            "order_created_at": order.get("created_at"),
            "quote_ts": (assessment or {}).get("quote_ts"),
            "trigger_ts": (assessment or {}).get("trigger_ts"),
            "window_start": order.get("window_start"),
        },
    }
    if not assessment:
        missing.append("assessment")

    # ---- 后：结果与反事实 ----
    pack["post"] = {
        "window_klines_1m": _kline_brief(klines_1m_window),
        "quotes_around_settle": quotes_post,
        "settle": {
            "outcome": order.get("settle_outcome"),
            "win": order.get("win"),
            "pnl": order.get("pnl"),
            "settle_price": order.get("settle_price"),
            "settled_at": order.get("settled_at"),
        },
        "siblings": siblings,
    }
    if not klines_1m_window:
        missing.append("klines_1m_window")
    return pack, missing


def compute_order_facts(order: dict) -> dict:
    """单笔订单确定性事实卡（TCA 分解 + 报价执行要点）。"""
    q = order.get("quote_json") or {}
    try:
        amount = float(order.get("amount_in") or 0) / 1e18
    except (TypeError, ValueError):
        amount = 0.0

    def _f(v) -> float | None:
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    tca = astats.tca_decompose({
        "win": order.get("win"),
        "avg_price": _f(q.get("averagePrice")),
        "quoted_avg_price": _f(q.get("quotedAvgPrice")),
        "amount_usdt": amount,
        "pnl": order.get("pnl"),
        "settle_price": order.get("settle_price"),
    })
    return {
        "order_id": order.get("id"),
        "channel": order.get("signal_version"),
        "direction": order.get("direction"),
        "amount_usdt": round(amount, 4),
        "entry_avg_price": _f(q.get("averagePrice")),
        "quoted_avg_price": _f(q.get("quotedAvgPrice")),
        "win": order.get("win"),
        "pnl": order.get("pnl"),
        "tca": tca,
    }


class OrderContextRecorder:
    """证据包组装器。kline_fetcher 由 main.py 装配注入（测试替身同理）。"""

    def __init__(self) -> None:
        self.kline_fetcher: KlineFetcher | None = None
        self.quote_prober: QuoteProber | None = None
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # 结算钩子入口（fire-and-forget；任何异常只记日志）
    # ------------------------------------------------------------------
    def enqueue(self, order_id: int) -> None:
        from binance_predict.config.settings import settings
        if not settings.attribution_enabled:
            return
        task = asyncio.create_task(self._safe_build(order_id), name=f"ctx_pack_{order_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _safe_build(self, order_id: int) -> None:
        try:
            status = await self.build_and_store(order_id)
            logger.info("证据包组装完成 | order={} | {}", order_id, status)
        except Exception as exc:
            logger.warning("证据包组装失败 | order={} | {} {}", order_id, type(exc).__name__, exc)

    # ------------------------------------------------------------------
    # 组装并落库（幂等：同订单覆盖更新）
    # ------------------------------------------------------------------
    async def build_and_store(self, order_id: int) -> str:
        from binance_predict.config.settings import settings
        from binance_predict.db.models import PredictionMarketSample, ShadowExecutionAssessment

        period_ms = {"5m": 300_000, "15m": 900_000}
        async with async_session_factory() as db:
            order = (await db.execute(
                sa_select(TradeOrderModel).where(TradeOrderModel.id == order_id)
            )).scalar_one_or_none()
            if order is None:
                return "SKIPPED_NO_ORDER"
            o = {
                "id": order.id, "signal_version": order.signal_version,
                "direction": order.direction, "side": order.side, "status": order.status,
                "market_period": order.market_period, "window_start": order.window_start,
                "amount_in": order.amount_in, "quote_json": order.quote_json or {},
                "settle_outcome": order.settle_outcome, "win": order.win,
                "pnl": order.pnl, "settle_price": order.settle_price,
                "error_message": order.error_message,
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "settled_at": order.settled_at.isoformat() if order.settled_at else None,
            }
            win_ms = period_ms.get(order.market_period or "5m", 300_000)
            ws = int(order.window_start or 0)
            we = ws + win_ms

            assessment_dict = None
            if order.assessment_id:
                a = (await db.execute(sa_select(ShadowExecutionAssessment).where(
                    ShadowExecutionAssessment.id == order.assessment_id))).scalar_one_or_none()
                if a is not None:
                    assessment_dict = {
                        "trigger_ts": a.trigger_ts, "quote_ts": a.quote_ts,
                        "effective_max_exec_price": a.effective_max_exec_price,
                        "guard_applied": a.guard_applied,
                        "amount_usdt": a.amount_usdt,
                        "terminal_stage": a.terminal_stage, "reason_code": a.reason_code,
                        "balance_available": a.balance_available,
                        "config_snapshot": a.config_snapshot or {},
                        "input_snapshot": a.input_snapshot or {},
                        "decision_snapshot": a.decision_snapshot or {},
                        "quote_snapshot": a.quote_snapshot or {},
                    }

            quote_cols = (
                PredictionMarketSample.timestamp, PredictionMarketSample.up_price,
                PredictionMarketSample.down_price, PredictionMarketSample.btc_price,
                PredictionMarketSample.trade_volume, PredictionMarketSample.participants,
            )
            pre_rows = (await db.execute(sa_select(*quote_cols).where(
                PredictionMarketSample.market_period == order.market_period,
                PredictionMarketSample.timestamp >= ws - PRE_WINDOW_MS,
                PredictionMarketSample.timestamp < ws,
            ).order_by(PredictionMarketSample.timestamp))).all()
            post_rows = (await db.execute(sa_select(*quote_cols).where(
                PredictionMarketSample.market_period == order.market_period,
                PredictionMarketSample.timestamp >= we - PRE_WINDOW_MS,
                PredictionMarketSample.timestamp <= we + POST_WINDOW_MS,
            ).order_by(PredictionMarketSample.timestamp))).all()

            def _q(rows) -> list[dict]:
                return _downsample([{
                    "ts": r.timestamp, "up": r.up_price, "down": r.down_price,
                    "btc": r.btc_price, "vol": r.trade_volume, "parts": r.participants,
                } for r in rows])

            # v2：按真实决策时刻再切一组 K 线（场景卡专用）
            decision_ts, decision_source = _resolve_decision_ts(o, assessment_dict, ws)
            klines_1m_decision = klines_5m_decision = None
            if self.kline_fetcher is not None and decision_ts > 0:
                try:
                    klines_1m_decision = await self.kline_fetcher(
                        "1m", KLINE_1M_DECISION, decision_ts)
                except Exception as exc:
                    logger.debug("证据包 1m 决策 K 线获取失败 | order={} | {}", order_id, exc)
                try:
                    klines_5m_decision = await self.kline_fetcher(
                        "5m", KLINE_5M_DECISION, decision_ts)
                except Exception as exc:
                    logger.debug("证据包 5m 决策 K 线获取失败 | order={} | {}", order_id, exc)

            klines_1m_before = klines_5m_before = klines_1m_window = None
            if self.kline_fetcher is not None:
                try:
                    klines_1m_before = await self.kline_fetcher("1m", KLINE_1M_BEFORE, ws)
                except Exception as exc:
                    logger.debug("证据包 1m 前置 K 线获取失败 | order={} | {}", order_id, exc)
                try:
                    klines_5m_before = await self.kline_fetcher("5m", KLINE_5M_BEFORE, ws)
                except Exception as exc:
                    logger.debug("证据包 5m 前置 K 线获取失败 | order={} | {}", order_id, exc)
                try:
                    klines_1m_window = await self.kline_fetcher(
                        "1m", win_ms // 60_000 + 5, we + 60_000)
                except Exception as exc:
                    logger.debug("证据包窗口 1m K 线获取失败 | order={} | {}", order_id, exc)

            sib_rows = (await db.execute(sa_select(
                TradeOrderModel.signal_version, TradeOrderModel.direction,
                TradeOrderModel.win, TradeOrderModel.pnl, TradeOrderModel.status,
            ).where(
                TradeOrderModel.window_start == order.window_start,
                TradeOrderModel.id != order.id,
                TradeOrderModel.status == "FILLED",
            ))).all()
            siblings = [{"channel": r.signal_version, "direction": r.direction,
                         "win": r.win, "pnl": r.pnl} for r in sib_rows]

            prior_rows = (await db.execute(sa_select(
                TradeOrderModel.win, TradeOrderModel.pnl,
            ).where(
                TradeOrderModel.signal_version == order.signal_version,
                TradeOrderModel.status == "FILLED",
                TradeOrderModel.win.isnot(None),
                TradeOrderModel.created_at < order.created_at,
            ).order_by(TradeOrderModel.created_at.desc()).limit(20))).all()
            prior_wins = [r.win for r in prior_rows]
            channel_prior = {
                "n_prior": len(prior_rows),
                "rolling20_win_rate": round(sum(prior_wins) / len(prior_wins), 4) if prior_wins else None,
                "rolling20_pnl": round(sum(r.pnl or 0.0 for r in prior_rows), 2),
            }

            ladder = None
            if (settings.attribution_ladder_probe and self.quote_prober is not None
                    and order.token_id):
                try:
                    ladder = await self.quote_prober(order.token_id, "BUY", [1.0, 5.0, 10.0])
                except Exception as exc:
                    ladder = {"error": str(exc)}

            pack, missing = assemble_pack(
                o, assessment_dict, _q(pre_rows), _q(post_rows),
                klines_1m_before, klines_5m_before, klines_1m_window,
                siblings, channel_prior,
                klines_1m_decision=klines_1m_decision,
                klines_5m_decision=klines_5m_decision,
                decision_ts=decision_ts,
                decision_source=decision_source,
            )
            if ladder is not None:
                pack["post"]["ladder"] = ladder
            facts = compute_order_facts(o)
            status = "BUILT" if not missing else "PARTIAL"

            existing = (await db.execute(sa_select(OrderContextPack).where(
                OrderContextPack.order_id == order_id))).scalar_one_or_none()
            if existing is None:
                existing = OrderContextPack(order_id=order_id)
                db.add(existing)
            existing.channel = order.signal_version or ""
            existing.market_period = order.market_period or "5m"
            existing.window_start = ws
            existing.direction = order.direction
            existing.schema_version = PACK_SCHEMA_VERSION
            existing.status = status
            existing.missing = missing
            existing.pack = pack
            existing.facts = facts
            await db.commit()
            return status


recorder = OrderContextRecorder()
