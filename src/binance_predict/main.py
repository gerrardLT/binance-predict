"""
BTC 5min LLM 预测系统 V3 - FastAPI 主应用

系统入口文件，负责：
1. 初始化服务（数据采集、情绪Agent Loop、交易执行）
2. 管理应用生命周期（lifespan）：启动AgentScheduler驱动的四阶段闭环
3. 注册 API 路由

核心引擎：情绪曲线自进化 Agent Loop（SentimentAgent + AgentScheduler），
由预测市场采样(_prediction_market_tracker)/归档(_sentiment_window_archiver)
事件驱动，全自动运转。概率动量分析（MomentumService）作为独立备选方案，
仅支持手动触发，不参与自动决策。
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config.settings import settings
from .db.engine import async_session_factory, get_db
from .db.models import Base, PredictionMarketSample, SentimentWindow
from .services.agent_scheduler import AgentScheduler
from .services.data_collector import BinanceDataCollector
from .services.llm_service import LLMService
from .services.prediction_trading import BinancePredictionTrader
from .services.prediction_market_data import PredictionMarketDataService, MarketQuoteData
from .services.sentiment_agent import SentimentAgent

# ============================================================
# 全局服务实例
# ============================================================

collector = BinanceDataCollector()
llm_service = LLMService()
prediction_trader = BinancePredictionTrader()
market_data_service = PredictionMarketDataService()  # 只读市场数据（与交易模块读写分离）

# 预测市场情绪追踪：每 15s 轮询 UP/DOWN token 报价
from collections import deque as _deque
_pm_history: _deque = _deque(maxlen=2000)  # 约 8 小时（15s × 2000）

# 模块级窗口状态变量（tracker/archiver/predict 共享）
_current_window_end: int | None = None
_window_entry_price: float | None = None
_pm_market_info: dict = {}  # 最新预测市场元数据（供图表 API 只读访问）

# AgentScheduler 全局实例（lifespan 中初始化，tracker/archiver 引用发布事件）
agent_scheduler: AgentScheduler | None = None

# PREDICT 事件触发标志：同一窗口仅触发一次（Req 3.1），窗口切换时重置
_predict_triggered_for_window: bool = False


# ============================================================
# 定时任务
# ============================================================

async def _prediction_market_tracker() -> None:
    """
    预测市场情绪追踪：每 15s 轮询 UP/DOWN token 报价

    通过 PredictionMarketDataService（只读）获取报价数据，
    记录到 _pm_history（内存）+ prediction_market_samples（DB）。
    启动时从 DB 加载最近 2000 条历史记录。

    读写分离设计：本函数不再调用 prediction_trader.list_markets()，
    避免修改交易模块状态（Bug 1.1 修复）。
    """
    global _current_window_end, _window_entry_price, _pm_market_info, _predict_triggered_for_window

    POLL_INTERVAL = 15  # 秒
    _restored_current_window = False  # 标记是否已从 DB 恢复当前窗口数据

    # 启动时校准 Binance 服务器时钟
    await market_data_service.sync_server_time()

    # 预测市场追踪启动：清空内存缓存，启动后从 DB 恢复当前窗口数据
    logger.info("预测市场追踪启动（读写分离模式）")
    _pm_history.clear()

    while True:
        try:
            # 对齐到本地时间 15 秒整数边界（:00, :15, :30, :45）
            now = time.time()
            local = time.localtime(now)
            local_sec_in_min = local.tm_sec
            sleep_sec = (POLL_INTERVAL - (local_sec_in_min % POLL_INTERVAL)) % POLL_INTERVAL
            if sleep_sec < 0.1:
                sleep_sec = POLL_INTERVAL
            target_epoch = now + sleep_sec
            t = time.localtime(target_epoch)
            aligned_sec = (t.tm_sec // POLL_INTERVAL) * POLL_INTERVAL
            aligned_ts = int(time.mktime((
                t.tm_year, t.tm_mon, t.tm_mday,
                t.tm_hour, t.tm_min, aligned_sec,
                t.tm_wday, t.tm_yday, t.tm_isdst
            ))) * 1000
            await asyncio.sleep(sleep_sec)

            # 通过只读服务获取市场报价（不修改交易模块状态）
            try:
                quote = await market_data_service.fetch_market_data()
            except Exception:
                continue

            # Bug 1.2 修复：end_date=None 防御
            if quote.end_date is None:
                logger.warning("end_date 为 None，跳过本轮采样")
                continue

            # 更新市场元数据（供图表 API 只读访问）
            _pm_market_info = {
                "participant_count": quote.participants,
                "trade_volume": quote.trade_volume,
                "start_date": quote.start_date,
                "end_date": quote.end_date,
                "up_price": quote.up_price,
                "down_price": quote.down_price,
                "up_chance": quote.up_chance,
                "down_chance": quote.down_chance,
            }

            # 检测 5 分钟窗口切换：end_date 变化说明进入了新市场
            new_window_end = quote.end_date
            if new_window_end != _current_window_end:
                if _current_window_end is not None:
                    logger.info("5分钟市场窗口切换 | 清空图表缓存 | {} → {}", _current_window_end, new_window_end)
                    _pm_history.clear()
                _current_window_end = new_window_end
                # Bug 1.3 修复：窗口切换时重置 _restored_current_window
                _restored_current_window = False
                # Bug 1.5 修复：窗口开始时快照 entry_price
                _window_entry_price = collector.store.mid_price
                # 窗口切换时重置 PREDICT 触发标志（Req 3.1，同一窗口仅触发一次）
                _predict_triggered_for_window = False

                # 首次进入窗口（含启动/重载）：从 DB 恢复当前窗口的采样数据
                if not _restored_current_window:
                    _restored_current_window = True
                    try:
                        window_end_ms = int(new_window_end)
                        window_start_ms = window_end_ms - 5 * 60 * 1000
                        async with async_session_factory() as db:
                            from sqlalchemy import select as sa_select
                            stmt = (
                                sa_select(PredictionMarketSample)
                                .where(PredictionMarketSample.timestamp >= window_start_ms)
                                .where(PredictionMarketSample.timestamp < window_end_ms)
                                .order_by(PredictionMarketSample.timestamp.asc())
                            )
                            result = await db.execute(stmt)
                            rows = result.scalars().all()
                            for row in rows:
                                _pm_history.append({
                                    "timestamp": row.timestamp,
                                    "up_price": row.up_price,
                                    "down_price": row.down_price,
                                    "up_pct": row.up_pct,
                                    "down_pct": row.down_pct,
                                    "participants": row.participants,
                                    "trade_volume": row.trade_volume,
                                })
                            if rows:
                                logger.info("从 DB 恢复当前窗口数据 | {} 条采样 | 窗口 {}~{}", len(rows), window_start_ms, window_end_ms)
                    except Exception as e:
                        logger.warning("从 DB 恢复当前窗口数据失败: {}", e)

            up_chance = quote.up_chance
            down_chance = quote.down_chance
            up_price = quote.up_price
            down_price = quote.down_price

            if up_chance is not None or down_chance is not None:
                point = {
                    "timestamp": aligned_ts,
                    "up_price": up_price,
                    "down_price": down_price,
                    "up_pct": round(up_chance * 100, 1) if up_chance is not None else None,
                    "down_pct": round(down_chance * 100, 1) if down_chance is not None else None,
                    "participants": quote.participants,
                    "trade_volume": float(quote.trade_volume) if quote.trade_volume is not None else None,
                }
                _pm_history.append(point)

                # 持久化到 DB
                try:
                    async with async_session_factory() as db:
                        db.add(PredictionMarketSample(
                            timestamp=point["timestamp"],
                            up_price=point["up_price"],
                            down_price=point["down_price"],
                            up_pct=point["up_pct"],
                            down_pct=point["down_pct"],
                            participants=point["participants"],
                            trade_volume=point["trade_volume"],
                        ))
                        await db.commit()
                except Exception as e:
                    logger.warning("预测市场采样入库失败: {}", e)

                # 检查是否达到 PREDICT 触发条件（Req 3.1）：
                # 当前窗口累计有效采样点达到 agent_predict_trigger_samples 时，
                # 向 AgentScheduler 发布 PREDICT 事件（同一窗口仅触发一次）。
                # 注意：len(_pm_history) 即为当前窗口内采样数，因为窗口切换时
                # 执行了 _pm_history.clear()，所以 deque 长度等于窗口内累积计数。
                if (
                    not _predict_triggered_for_window
                    and agent_scheduler is not None
                    and len(_pm_history) >= settings.agent_predict_trigger_samples
                ):
                    _predict_triggered_for_window = True
                    # 构建 current_curve：当前窗口的 UP/DOWN% 时序切片
                    current_curve = [
                        {"t": p["timestamp"], "up_pct": p["up_pct"], "down_pct": p["down_pct"]}
                        for p in _pm_history
                    ]
                    agent_scheduler.publish("PREDICT", {
                        "window_end_ms": _current_window_end,
                        "current_curve": current_curve,
                    })
                    logger.info(
                        "PREDICT 事件发布 | 窗口={} | 采样点={} | 阈值={}",
                        _current_window_end,
                        len(_pm_history),
                        settings.agent_predict_trigger_samples,
                    )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("预测市场追踪异常: {}", e)


async def _sentiment_window_archiver() -> None:
    """
    情绪窗口归档器：每 5 分钟将采样点聚合为一条 SentimentWindow 记录

    修复版：
    - Bug 1.6: 归档门槛从 3 提高到 8
    - Bug 1.7: entry_price 使用模块级 _window_entry_price 快照，exit_price 使用实时 mid_price
    - [DEPRECATED] Bug 1.8/1.9: 回测缓存失效 + 自动触发回测已退役，由 AgentScheduler 发布事件取代
    """
    from sqlalchemy import select as sa_select, delete as sa_delete, func as sa_func
    from sqlalchemy.exc import IntegrityError

    # 等待第一个 5 分钟边界
    await asyncio.sleep(10)

    while True:
        try:
            # 等待到下一个 5 分钟整点（本地时间）
            now = time.time()
            local = time.localtime(now)
            sec_in_5m = (local.tm_min % 5) * 60 + local.tm_sec
            sleep_to_boundary = (5 * 60 - sec_in_5m) % (5 * 60)
            if sleep_to_boundary < 5:
                sleep_to_boundary = 5 * 60
            await asyncio.sleep(sleep_to_boundary + 15)  # 多等 15 秒确保 tracker 的边界采样已写入

            # 使用模块级 _current_window_end 计算窗口范围（Bug 1.10 修复）
            if _current_window_end is None:
                logger.debug("情绪窗口跳过 | tracker 尚未就绪（_current_window_end 为 None）")
                continue

            end_ms = int(_current_window_end)
            start_ms = end_ms - 5 * 60 * 1000

            async with async_session_factory() as db:
                # 查询窗口内的采样点
                stmt = (
                    sa_select(PredictionMarketSample)
                    .where(PredictionMarketSample.timestamp >= start_ms)
                    .where(PredictionMarketSample.timestamp < end_ms)
                    .order_by(PredictionMarketSample.timestamp.asc())
                )
                result = await db.execute(stmt)
                samples = result.scalars().all()

                # Bug 1.6 修复：归档门槛从 3 提高到 8
                if len(samples) < 8:
                    logger.debug("情绪窗口跳过 | {}~{} | 采样点不足({}<8)", start_ms, end_ms, len(samples))
                else:
                    # 构建曲线数据
                    curve_up = [{"t": s.timestamp, "v": s.up_pct} for s in samples if s.up_pct is not None]
                    curve_down = [{"t": s.timestamp, "v": s.down_pct} for s in samples if s.down_pct is not None]

                    # Bug 1.7 修复：使用快照价格替代 kline 匹配
                    entry_price = _window_entry_price  # 窗口开始时的 mid_price 快照
                    exit_price = collector.store.mid_price  # 归档时实时 mid_price

                    # 计算实际结果
                    actual_return = None
                    outcome = None
                    if entry_price and exit_price and entry_price > 0:
                        actual_return = exit_price / entry_price - 1
                        noise_threshold = settings.noise_threshold
                        if actual_return > noise_threshold:
                            outcome = "UP"
                        elif actual_return < -noise_threshold:
                            outcome = "DOWN"
                        else:
                            outcome = "NOISE"

                    # 聚合参与者/交易量均值
                    participant_vals = [s.participants for s in samples if s.participants is not None]
                    volume_vals = [s.trade_volume for s in samples if s.trade_volume is not None]
                    avg_part = sum(participant_vals) / len(participant_vals) if participant_vals else None
                    avg_vol = sum(volume_vals) / len(volume_vals) if volume_vals else None

                    # 存入 SentimentWindow（唯一约束防重复）
                    window = SentimentWindow(
                        start_time=start_ms,
                        end_time=end_ms,
                        curve_up_pct=curve_up,
                        curve_down_pct=curve_down,
                        sample_count=len(samples),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        actual_return=actual_return,
                        outcome=outcome,
                        avg_participants=avg_part,
                        avg_trade_volume=avg_vol,
                    )
                    try:
                        db.add(window)
                        await db.commit()
                        await db.refresh(window)  # 确保 window.id 可用于事件发布
                        logger.info(
                            "情绪窗口归档 | {}~{} | {}个点 | {} | return={:.4f}",
                            start_ms, end_ms, len(samples), outcome, actual_return or 0,
                        )

                        # 向 AgentScheduler 发布 WINDOW_ARCHIVED 事件（Req 6.1/6.2）：
                        # 驱动 Validate→Learn 闭环，替换原 _auto_run_backtest 直调（已退役）
                        if agent_scheduler is not None:
                            agent_scheduler.publish("WINDOW_ARCHIVED", {"window_id": window.id})
                            logger.debug(
                                "WINDOW_ARCHIVED 事件发布 | window_id={} | {}~{}",
                                window.id, start_ms, end_ms,
                            )

                    except IntegrityError:
                        await db.rollback()
                        logger.debug("情绪窗口已存在（跳过重复归档）| {}~{}", start_ms, end_ms)

                # 清理 1 小时前的旧采样记录（防止 DB 无限增长）
                cleanup_threshold_ms = end_ms - 3600 * 1000
                del_result = await db.execute(
                    sa_delete(PredictionMarketSample)
                    .where(PredictionMarketSample.timestamp < cleanup_threshold_ms)
                )
                if del_result.rowcount > 0:
                    await db.commit()
                    logger.debug("清理旧采样记录 | 删除 {} 条（早于 {}）", del_result.rowcount, cleanup_threshold_ms)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("情绪窗口归档异常: {}", e)
            await asyncio.sleep(30)


# ============================================================
# 应用生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("BTC 5min LLM 预测系统 V2 启动中...")

    # 1. 数据库表初始化
    # 优先使用 Alembic 迁移（alembic upgrade head），若无迁移则 fallback 到 create_all
    from .db.engine import engine
    try:
        # 尝试 Alembic 迁移
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command
        import os

        alembic_ini = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "alembic.ini")
        if os.path.exists(alembic_ini):
            alembic_cfg = AlembicConfig(alembic_ini)
            alembic_cfg.set_main_option("sqlalchemy.url", settings.database_url)
            # Alembic 异步迁移需通过 CLI 执行，此处仅检查
            logger.info("Alembic 配置已就绪 | 生产环境请运行: alembic upgrade head")

        # fallback: create_all 确保表存在（开发环境）
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表创建/检查完成")
        # 增量列迁移（create_all 不会给已有表加列）
        async with engine.begin() as conn:
            for col_sql in [
                "ALTER TABLE prediction_market_samples ADD COLUMN IF NOT EXISTS participants INTEGER",
                "ALTER TABLE prediction_market_samples ADD COLUMN IF NOT EXISTS trade_volume FLOAT",
            ]:
                try:
                    await conn.execute(text(col_sql))
                except Exception:
                    pass
    except Exception as e:
        logger.warning("数据库连接失败（开发模式可忽略）: {}", e)
        logger.warning("系统将以降级模式运行，决策/验证任务将跳过数据库操作")

    # 3. 启动高频 asyncio 任务（现货 WS 连接 + SSE 心跳 + 预测市场追踪）
    await collector.start()

    # 4. 实例化并启动 AgentScheduler（Req 2.1/6.1/11.2）
    # 在 tracker/archiver 的 while 循环开始前完成，保证时序安全
    global agent_scheduler
    sentiment_agent = SentimentAgent(llm=llm_service, trader=prediction_trader)
    agent_scheduler = AgentScheduler(agent=sentiment_agent, trader=prediction_trader)
    await agent_scheduler.start()  # 含冷启动检查（Req 11.2）
    logger.info("SentimentAgent + AgentScheduler 已就绪（冷启动检查完成）")

    tasks = [
        asyncio.create_task(collector.connect_spot_ws(), name="spot_ws"),
        asyncio.create_task(_prediction_market_tracker(), name="pm_tracker"),
        asyncio.create_task(_sentiment_window_archiver(), name="sw_archiver"),
    ]
    logger.info("现货 WS + 预测市场追踪 + 情绪窗口归档已启动")

    yield  # 应用运行中

    # 4. 清理
    logger.info("系统关闭中...")
    # 停止 AgentScheduler（优雅关闭，等待当前阶段执行完毕）
    if agent_scheduler is not None:
        await agent_scheduler.stop()
    await collector.stop()
    for t in tasks:
        t.cancel()
    logger.info("系统已关闭")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="BTC 5min LLM 预测系统 V2",
    description="LLM 驱动的 BTC 5 分钟方向预测，支持用户自定义规则注入",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境，生产应限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API 路由（V2 PRD §17）
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "symbol": settings.symbol,
        "mid_price": collector.store.mid_price,
        "ws_spot_connected": collector.store.ws_spot_connected,
    }


# --- 交易订单 API ---

@app.get("/api/trades/latest")
async def get_latest_trade(db: AsyncSession = Depends(get_db)):
    """获取最近一次交易订单"""
    from sqlalchemy import select
    from .db.models import TradeOrderModel

    stmt = (
        select(TradeOrderModel)
        .order_by(TradeOrderModel.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    order = result.scalar_one_or_none()

    if not order:
        return {"error": "暂无交易记录"}

    return {
        "id": order.id,
        "prediction_id": order.prediction_id,
        "side": order.side,
        "amount_in": order.amount_in,
        "amount_out": order.amount_out,
        "order_id": order.order_id,
        "status": order.status,
        "error_message": order.error_message,
        "created_at": order.created_at.isoformat() if order.created_at else None,
    }


@app.get("/api/prediction-markets")
async def list_prediction_markets():
    """查询当前活跃的 BTC 预测市场"""
    markets = await prediction_trader.list_markets()
    return {
        "count": len(markets),
        "markets": markets[:5],  # 只返回前 5 个
        "up_token_id": prediction_trader._up_token_id,
        "down_token_id": prediction_trader._down_token_id,
    }


@app.get("/api/prediction-wallet")
async def get_prediction_wallet():
    """获取预测钱包信息（walletAddress + walletId，自动从 Binance API 获取）"""
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}

    wallet = await prediction_trader.fetch_wallet_info()
    if not wallet:
        return {"error": "未找到预测钱包，请先在 Binance App 中开通预测市场"}

    return {
        "wallet_address": wallet.get("walletAddress"),
        "wallet_id": wallet.get("walletId"),
        "registered_time": wallet.get("registeredTime"),
    }


# ============================================================
# [DEPRECATED] 情绪曲线回测自动触发（已退役，由 SentimentAgent.learn() 取代）
# ============================================================

# _auto_run_backtest 已退役：文本回测被 Learn 阶段的结构化模式发现取代
# 保留函数签名供历史引用，不再实际调用


# ============================================================
# 图表数据 API
# ============================================================


@app.get("/api/chart/prediction-market")
async def get_prediction_market_chart():
    """
    获取 Binance BTC 5 分钟涨跌预测市场实时数据

    返回每 15s 采样的 UP/DOWN chance 百分比时序数据，
    以及当前 5 分钟市场的元数据（参与者、交易量、截止时间等）。
    """
    return {
        "symbol": settings.symbol,
        "poll_interval_sec": 15,
        "points": list(_pm_history),
        "market": _pm_market_info,
    }


# ============================================================
# 情绪曲线分析 API
# ============================================================

# [DEPRECATED] 回测缓存与锁已退役，由 SentimentAgent Learn 阶段取代
# _last_backtest_result / _last_backtest_time / _backtest_lock 不再使用


@app.get("/api/sentiment/windows")
async def get_sentiment_windows(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """查询历史情绪窗口列表"""
    from sqlalchemy import select as sa_select

    stmt = (
        sa_select(SentimentWindow)
        .order_by(SentimentWindow.start_time.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    windows = result.scalars().all()
    return [
        {
            "id": w.id,
            "start_time": w.start_time,
            "end_time": w.end_time,
            "sample_count": w.sample_count,
            "entry_price": w.entry_price,
            "exit_price": w.exit_price,
            "actual_return": w.actual_return,
            "outcome": w.outcome,
            "curve_up_pct": w.curve_up_pct,
            "curve_down_pct": w.curve_down_pct,
        }
        for w in windows
    ]


@app.post("/api/sentiment/backtest")
async def run_sentiment_backtest(
    window_count: int = 24,
    force: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    [DEPRECATED] 情绪曲线回测分析 — 已退役

    该端点依赖的 llm_service.sentiment_backtest 已被 SentimentAgent Learn 阶段取代。
    保留端点返回退役提示，避免前端调用报 404。
    """
    return {
        "status": "deprecated",
        "message": "回测功能已退役，已被 SentimentAgent 的 Learn 阶段取代。请使用 /api/sentiment/agent/patterns 查看模式库。",
    }


@app.post("/api/sentiment/predict")
async def run_sentiment_prediction(
    db: AsyncSession = Depends(get_db),
):
    """
    [DEPRECATED] 基于当前情绪曲线的实时预测 — 已退役

    该端点依赖的 llm_service.sentiment_predict 已被 SentimentAgent Predict 阶段取代。
    保留端点返回退役提示，避免前端调用报 404。
    """
    return {
        "status": "deprecated",
        "message": "情绪预测功能已退役，已被 SentimentAgent 的 Predict 阶段取代。请使用 /api/sentiment/agent/predictions 查看预测历史。",
    }


# ============================================================
# 概率动量预测 API（方案 C：纯算法，不依赖 LLM）
# ============================================================

@app.post("/api/sentiment/momentum-predict")
async def run_momentum_predict():
    """
    概率动量预测（独立方案，纯算法）

    基于预测市场 UP% 时序数据计算多维度动量信号：
    - 概率动量（15s/30s/60s 变化率）
    - 概率波动率（标准差）
    - 参与者增长率
    - 交易量加速度
    - 趋势一致性

    不依赖 K 线，不依赖 LLM，与 K线+LLM 方法互补。
    """
    from .services.momentum_service import MomentumService

    points = list(_pm_history)
    if not points:
        return {"status": "error", "message": "当前无采样数据，请等待数据采集"}

    service = MomentumService()
    result = service.analyze(points)

    return {
        "status": "ok",
        "direction": result.direction,
        "confidence": result.confidence,
        "composite_score": result.composite_score,
        "elapsed_seconds": result.elapsed_seconds,
        "remaining_seconds": result.remaining_seconds,
        "sample_count": result.sample_count,
        "signals": [
            {
                "name": s.name,
                "value": s.value,
                "score": s.score,
                "description": s.description,
            }
            for s in result.signals
        ],
        "reasoning": result.reasoning,
    }


# ============================================================
# Sentiment Agent 查询 API（Req 6.5, 8.3, 8.4, 8.5）
# 端点内导入以避免循环依赖与全局命名空间冲突（sa_select 别名区分于模块级 select）
# ============================================================

@app.get("/api/sentiment/agent/predictions")
async def get_agent_predictions(
    start: datetime | None = None,
    end: datetime | None = None,
    direction: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    查询 Agent 预测历史（Req 8.3）

    支持按时间范围和方向筛选，按 prediction_time 降序返回，最多 100 条。
    """
    from sqlalchemy import select as sa_select
    from .db.models import AgentPrediction
    from .models.schemas import AgentPredictionRecord

    stmt = sa_select(AgentPrediction).order_by(AgentPrediction.prediction_time.desc())

    # 时间范围筛选
    if start is not None:
        stmt = stmt.where(AgentPrediction.prediction_time >= start)
    if end is not None:
        stmt = stmt.where(AgentPrediction.prediction_time <= end)
    # 方向筛选
    if direction is not None:
        stmt = stmt.where(AgentPrediction.predicted_direction == direction)

    stmt = stmt.limit(100)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        AgentPredictionRecord(
            id=r.id,
            prediction_time=r.prediction_time,
            sentiment_window_id=r.sentiment_window_id,
            predicted_direction=r.predicted_direction,
            matched_pattern_id=r.matched_pattern_id,
            matched_pattern_name=r.matched_pattern_name,
            confidence=r.confidence,
            entry_timing=r.entry_timing,
            reasoning=r.reasoning,
            is_correct=r.is_correct,
            actual_outcome=r.actual_outcome,
            actual_return=r.actual_return,
            validated_at=r.validated_at,
            trade_order_id=r.trade_order_id,
            skip_trade_reason=r.skip_trade_reason,
            created_at=r.created_at,
        ).model_dump()
        for r in rows
    ]


@app.get("/api/sentiment/agent/patterns")
async def get_agent_patterns(
    db: AsyncSession = Depends(get_db),
):
    """
    查询 Pattern_Memory 当前状态（Req 8.4）

    返回所有模式（ACTIVE + RETIRED + 统计），按 status(ACTIVE 优先) 和 win_rate 降序排序。
    """
    from sqlalchemy import select as sa_select, case
    from .db.models import PatternMemory
    from .models.schemas import PatternMemoryRecord

    # 按 status 排序：ACTIVE=0, EVOLVING=1, RETIRED=2；同 status 内按 win_rate 降序
    status_order = case(
        (PatternMemory.status == "ACTIVE", 0),
        (PatternMemory.status == "EVOLVING", 1),
        else_=2,
    )
    stmt = (
        sa_select(PatternMemory)
        .order_by(status_order.asc(), PatternMemory.win_rate.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        PatternMemoryRecord(
            id=r.id,
            pattern_name=r.pattern_name,
            description=r.description,
            curve_features=r.curve_features,
            conditions=r.conditions,
            predicted_direction=r.predicted_direction,
            win_rate=r.win_rate,
            sample_count=r.sample_count,
            correct_count=r.correct_count,
            confidence_score=r.confidence_score,
            status=r.status,
            created_at=r.created_at,
            updated_at=r.updated_at,
        ).model_dump()
        for r in rows
    ]


@app.get("/api/sentiment/agent/patterns/{pattern_id}/history")
async def get_pattern_history(
    pattern_id: int,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    """
    查询某模式的进化轨迹（Req 8.5）

    返回该模式的所有 Pattern_Change_Log 记录，按 created_at 正序排列。
    支持 limit 分页参数（默认 200 条）。
    """
    from sqlalchemy import select as sa_select
    from .db.models import PatternChangeLog
    from .models.schemas import PatternChangeLogRecord

    stmt = (
        sa_select(PatternChangeLog)
        .where(PatternChangeLog.pattern_id == pattern_id)
        .order_by(PatternChangeLog.created_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        PatternChangeLogRecord(
            id=r.id,
            pattern_id=r.pattern_id,
            change_type=r.change_type,
            phase=r.phase,
            before_snapshot=r.before_snapshot,
            after_snapshot=r.after_snapshot,
            change_reason=r.change_reason,
            evolve_phase_id=r.evolve_phase_id,
            created_at=r.created_at,
        ).model_dump()
        for r in rows
    ]


@app.get("/api/sentiment/agent/status")
async def get_agent_status(
    db: AsyncSession = Depends(get_db),
):
    """
    查询 Agent 运行状态（Req 6.5）

    返回当前验证计数、ACTIVE 模式数量、调度器运行状态。
    """
    from sqlalchemy import select as sa_select, func as sa_func
    from .db.models import PatternMemory

    # 查询 ACTIVE 模式数量
    count_stmt = (
        sa_select(sa_func.count())
        .select_from(PatternMemory)
        .where(PatternMemory.status == "ACTIVE")
    )
    count_result = await db.execute(count_stmt)
    active_pattern_count = count_result.scalar() or 0

    # 从 AgentScheduler 获取验证计数和运行状态
    validate_counter = 0
    scheduler_running = False
    if agent_scheduler is not None:
        validate_counter = agent_scheduler.validate_counter
        scheduler_running = True

    return {
        "validate_counter": validate_counter,
        "active_pattern_count": active_pattern_count,
        "scheduler_running": scheduler_running,
    }
