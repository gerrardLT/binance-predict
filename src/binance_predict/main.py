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
import calendar
import json
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .config.settings import settings
from .db.engine import async_session_factory, get_db
from .db.models import Base, PredictionMarketSample, SentimentWindow
from .models.schemas import CommitDeepLearnRequest
from .services.agent_scheduler import AgentScheduler
from .services.data_collector import BinanceDataCollector
from .services.llm_service import LLMService
from .services.prediction_trading import BinancePredictionTrader
from .services.prediction_market_data import PredictionMarketDataService, MarketQuoteData
from .services.sentiment_agent import SentimentAgent
from .services.metrics import metrics_collector

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

# Fix #5/#6: 全局状态写入锁，保护 tracker/archiver 共享变量的读写一致性
_state_lock = asyncio.Lock()

# 模块级窗口状态变量（tracker/archiver/predict 共享）
_current_window_end: int | None = None
_window_entry_price: float | None = None
_pm_market_info: dict = {}  # 最新预测市场元数据（供图表 API 只读访问）

# 刚关闭窗口的快照（tracker 在窗口切换时写入，archiver 读取归档）：
# 修复归档器读取"正在填充的当前窗口"导致采样点不足、sentiment_windows 长期不增长的竞态问题。
_last_closed_window_end: int | None = None
_last_closed_window_entry_price: float | None = None
_last_closed_window_exit_price: float | None = None
_last_archived_window_end: int | None = None  # 去重：避免同一已关闭窗口重复归档

# AgentScheduler 全局实例（lifespan 中初始化，tracker/archiver 引用发布事件）
agent_scheduler: AgentScheduler | None = None

# SentimentAgent 全局实例（lifespan 中初始化，供 deep-learn API 调用）
sentiment_agent: SentimentAgent | None = None

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
    global _last_closed_window_end, _last_closed_window_entry_price, _last_closed_window_exit_price

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
            await asyncio.sleep(sleep_sec)

            # Fix #13: 使用 UTC 时间戳计算对齐后的毫秒时间戳，
            # 避免 time.mktime 依赖本地时区导致非 UTC 环境下的时间戳偏差
            aligned_epoch = now + sleep_sec
            aligned_ts = int(round(aligned_epoch)) * 1000

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
                # Fix #5: 使用锁保护全局状态写入，防止 archiver 读到半写状态
                async with _state_lock:
                    if _current_window_end is not None:
                        # 修复：记录刚关闭窗口的快照供 archiver 归档。
                        # 入场价 = 旧窗口起点快照；出场价 = 本次切换时刻的 mid_price（即旧窗口终点）。
                        _last_closed_window_end = _current_window_end
                        _last_closed_window_entry_price = _window_entry_price
                        _last_closed_window_exit_price = collector.store.mid_price
                        logger.info("5分钟市场窗口切换 | 清空图表缓存 | {} → {}", _current_window_end, new_window_end)
                        _pm_history.clear()
                    _current_window_end = new_window_end
                    # Bug 1.3 修复：窗口切换时重置 _restored_current_window
                    _restored_current_window = False
                    # Bug 1.5 修复：窗口开始时快照 entry_price。优先用内存最新
                    # mid_price 快照（非阻塞），避免在 _state_lock 内做阻塞 REST 调用。
                    _window_entry_price = collector.store.mid_price
                    # Fix #12: 内存快照无效时用 REST 后备补偿（罕见路径），仍无效则告警
                    if not _window_entry_price or _window_entry_price <= 0:
                        _window_entry_price = await collector.fetch_mid_price()
                        if not _window_entry_price or _window_entry_price <= 0:
                            logger.warning(
                                "窗口切换时 entry_price 异常({})，将在归档时重新获取",
                                _window_entry_price,
                            )
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

    global _last_archived_window_end

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

            # 修复：归档"刚关闭"的窗口，而非正在填充的当前窗口。
            # 原实现读 _current_window_end（当前活跃窗口），但 archiver 在边界+15s 唤醒时
            # tracker 通常已把 _current_window_end 推进到新窗口，导致查询到刚开始、
            # 采样点不足(<8)的新窗口 → sentiment_windows 长期不增长。改用切换时的快照。
            async with _state_lock:
                closed_end = _last_closed_window_end
                closed_entry = _last_closed_window_entry_price
                closed_exit = _last_closed_window_exit_price
            if closed_end is None:
                logger.debug("情绪窗口跳过 | 尚无已关闭窗口（_last_closed_window_end 为 None）")
                continue
            if closed_end == _last_archived_window_end:
                logger.debug("情绪窗口跳过 | 窗口 {} 已归档，等待下一次窗口切换", closed_end)
                continue

            end_ms = int(closed_end)
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

                # P1-2：采样点数仅作质量标注，不再作为归档闸门。
                # 原 <8 跳过会导致低采样窗口永不归档 → 其内 AgentPrediction 的
                # is_correct 永久为 None（孤儿预测）。改为：低采样仅告警，只要下方
                # 能取到有效首尾价格即照常归档并发布 WINDOW_ARCHIVED 驱动 Validate。
                if len(samples) < 8:
                    logger.warning(
                        "情绪窗口低采样 | {}~{} | 采样点={}(<8)，曲线质量偏低但仍归档",
                        start_ms, end_ms, len(samples),
                    )
                # 归档主体：对任意采样量执行（无有效首尾价格时下方会 continue 跳过）
                if len(samples) >= 0:
                    # 构建曲线数据
                    curve_up = [{"t": s.timestamp, "v": s.up_pct} for s in samples if s.up_pct is not None]
                    curve_down = [{"t": s.timestamp, "v": s.down_pct} for s in samples if s.down_pct is not None]

                    # 修复：使用窗口切换时快照的价格。
                    # entry_price = 已关闭窗口起点快照；exit_price = 切换时刻价（窗口终点）。
                    entry_price = closed_entry
                    exit_price = closed_exit if (closed_exit and closed_exit > 0) else collector.store.mid_price

                    # Fix #12: entry_price 异常时重试获取，避免生成无效归档记录
                    if not entry_price or entry_price <= 0:
                        logger.warning(
                            "情绪窗口归档 | {}~{} | entry_price 异常({})，重试获取",
                            start_ms, end_ms, entry_price,
                        )
                        entry_price = await collector.fetch_mid_price()
                        if not entry_price or entry_price <= 0:
                            logger.error(
                                "情绪窗口跳过 | {}~{} | entry_price 始终无效，跳过本次归档",
                                start_ms, end_ms,
                            )
                            continue

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

                        _last_archived_window_end = end_ms  # 标记已归档，避免重复

                    except IntegrityError:
                        await db.rollback()
                        _last_archived_window_end = end_ms  # 已存在也标记，停止重复尝试
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


async def _health_monitor_loop() -> None:
    """Agent 运行健康后台监控循环。

    按 settings.agent_health_monitor_interval 周期构建健康报告：
    - 非 OK 状态（WARN/CRITICAL）写日志，并经 alert_notifier 按 code 去重后
      主动推送（邮件 + 可选 webhook；同 code 在抑制窗口内不重发）
    - 每 settings.agent_health_snapshot_interval 落一条 HealthSnapshot 供趋势回看，
      并清理早于保留窗口的旧快照

    只读聚合 + 可选落库，异常仅告警不影响主决策流程。
    """
    from .db.models import HealthSnapshot
    from .services.health import health_service
    from .services.alerting import alert_notifier

    if not settings.agent_health_monitor_enabled:
        logger.info("Agent 健康监控已禁用（agent_health_monitor_enabled=False）")
        return

    await asyncio.sleep(30)  # 等待调度器/采集器预热，避免冷启动误报
    last_snapshot_at = 0.0
    logger.info(
        "Agent 健康监控已启动 | 轮询={}s | 落库={}s | 抑制窗口={}s",
        settings.agent_health_monitor_interval,
        settings.agent_health_snapshot_interval,
        settings.agent_alert_suppress_seconds,
    )

    while True:
        try:
            snapshot, consecutive_failures, queue_depth = _collect_memory_state()
            async with async_session_factory() as db:
                report = await health_service.build_report(
                    db,
                    metrics_snapshot=snapshot,
                    consecutive_failures=consecutive_failures,
                    queue_depth=queue_depth,
                )

                # 非 OK 状态写日志；新告警经邮件/webhook 主动推送（同 code 抑制窗口内不重发）
                if report.overall_status == "CRITICAL":
                    logger.warning("[HEALTH] CRITICAL | {}", report.summary)
                elif report.overall_status == "WARN":
                    logger.info("[HEALTH] WARN | {}", report.summary)
                await alert_notifier.notify(report)

                # 周期性落库 + 清理旧快照
                now = time.time()
                if now - last_snapshot_at >= settings.agent_health_snapshot_interval:
                    db.add(HealthSnapshot(
                        overall_status=report.overall_status,
                        alert_count=len(report.alerts),
                        report=report.model_dump(mode="json"),
                    ))
                    await db.commit()
                    last_snapshot_at = now
                    await _cleanup_old_health_snapshots(db)

            await asyncio.sleep(settings.agent_health_monitor_interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Agent 健康监控异常: {}", e)
            await asyncio.sleep(settings.agent_health_monitor_interval)


async def _cleanup_old_health_snapshots(db: AsyncSession) -> None:
    """删除早于保留窗口的 health_snapshots 记录，防止表无限增长。失败仅告警。"""
    from sqlalchemy import delete as sa_delete
    from .db.models import HealthSnapshot

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=settings.agent_health_snapshot_retention_days
    )
    try:
        result = await db.execute(
            sa_delete(HealthSnapshot).where(HealthSnapshot.created_at < cutoff)
        )
        if result.rowcount and result.rowcount > 0:
            await db.commit()
            logger.debug("清理旧健康快照 | 删除 {} 条（早于 {}）", result.rowcount, cutoff.isoformat())
    except Exception as e:
        await db.rollback()
        logger.warning("清理旧健康快照失败: {}", e)


# ============================================================
# 应用生命周期
# ============================================================

def setup_logging() -> None:
    """
    配置 loguru 日志输出。

    默认 loguru 仅输出到 stderr（会随容器重建丢失、无 rotation）。
    此函数在 stderr 之外追加持久化文件输出：
    - 按天切割（log_rotation），保留 log_retention
    - enqueue=True：多协程/线程安全写入
    - 文件级别始终 >= 配置的 log_level

    settings.log_dir 为空字符串时跳过文件日志（仅保留默认 stderr）。
    """
    import os
    import sys

    level = settings.log_level.upper()

    # 重置默认 handler，统一 stderr 格式与级别
    logger.remove()
    logger.add(sys.stderr, level=level, backtrace=False, diagnose=False)

    if not settings.log_dir:
        logger.warning("log_dir 为空，跳过文件日志（仅 stderr 输出）")
        return

    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        log_path = os.path.join(settings.log_dir, "app.log")
        logger.add(
            log_path,
            level=level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="zip",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            encoding="utf-8",
        )
        logger.info("文件日志已启用 | {} | 切割={} | 保留={}", log_path, settings.log_rotation, settings.log_retention)
    except Exception as e:
        # 文件日志失败不应阻断启动，退回 stderr
        logger.warning("文件日志初始化失败，仅使用 stderr 输出: {}", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    setup_logging()
    logger.info("BTC 5min LLM 预测系统 V3 启动中...")

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
                # Deep Learn 双轨：pattern_memory 发现方法与 holdout 统计（与 alembic 迁移等价，存量 dev 库安全网）
                "ALTER TABLE pattern_memory ADD COLUMN IF NOT EXISTS discovery_method VARCHAR(20) NOT NULL DEFAULT 'LEGACY'",
                "ALTER TABLE pattern_memory ADD COLUMN IF NOT EXISTS holdout_win_rate FLOAT",
                "ALTER TABLE pattern_memory ADD COLUMN IF NOT EXISTS holdout_sample_count INTEGER",
                "ALTER TABLE pattern_memory ADD COLUMN IF NOT EXISTS holdout_ci_lower FLOAT",
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
    global agent_scheduler, sentiment_agent
    _sentiment_agent = SentimentAgent(llm=llm_service, trader=prediction_trader)
    sentiment_agent = _sentiment_agent
    agent_scheduler = AgentScheduler(agent=_sentiment_agent, trader=prediction_trader)
    await agent_scheduler.start()  # 含冷启动检查（Req 11.2）
    logger.info("SentimentAgent + AgentScheduler 已就绪（冷启动检查完成）")

    # P1-1：启动对账——回填进程重启后遗漏的未验证预测（孤儿预测）。
    # 调度队列非持久，重启后 is_correct IS NULL 的预测无人回填；
    # 在 scheduler 启动后、正常事件流开始前一次性扫描并回填。
    try:
        reconciled = await _sentiment_agent.reconcile_pending_predictions()
        logger.info("启动对账完成 | 回填未验证预测 {} 条", reconciled)
    except Exception as exc:
        logger.error("启动对账失败（不阻断启动）| {}", exc)

    tasks = [
        asyncio.create_task(collector.connect_spot_ws(), name="spot_ws"),
        asyncio.create_task(_prediction_market_tracker(), name="pm_tracker"),
        asyncio.create_task(_sentiment_window_archiver(), name="sw_archiver"),
        asyncio.create_task(_health_monitor_loop(), name="health_monitor"),
    ]
    logger.info("现货 WS + 预测市场追踪 + 情绪窗口归档 + 健康监控已启动")

    yield  # 应用运行中

    # 4. 清理
    logger.info("系统关闭中...")
    # 停止 AgentScheduler（优雅关闭，等待当前阶段执行完毕）
    if agent_scheduler is not None:
        await agent_scheduler.stop()
    await collector.stop()
    # Fix #15: 关闭复用的 httpx 客户端，避免连接泄漏
    await market_data_service.aclose()
    await prediction_trader.aclose()
    for t in tasks:
        t.cancel()
    logger.info("系统已关闭")


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="BTC 5min LLM 预测系统 V3",
    description="LLM 驱动的 BTC 5 分钟方向预测，支持用户自定义规则注入",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS 中间件（安全修复 #1：禁止 allow_origins=["*"] + credentials=True）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ============================================================
# API 认证依赖（安全修复 #2：Bearer Token 保护敏感端点）
# ============================================================

_bearer_scheme = HTTPBearer(auto_error=False)


async def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Bearer Token 认证依赖。

    仅当 settings.api_auth_token 非空时生效。空值表示开发环境，放行所有请求。
    生产环境必须配置 API_AUTH_TOKEN，否则端点对外完全开放。
    """
    if not settings.api_auth_token:
        return  # 开发模式：未配置 token 则跳过认证
    if credentials is None or credentials.credentials != settings.api_auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
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
async def get_latest_trade(
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
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
async def list_prediction_markets(
    _: None = Depends(_require_auth),
):
    """查询当前活跃的 BTC 预测市场"""
    markets = await prediction_trader.list_markets()
    return {
        "count": len(markets),
        "markets": markets[:5],  # 只返回前 5 个
        "has_up_token": bool(prediction_trader._up_token_id),
        "has_down_token": bool(prediction_trader._down_token_id),
    }


@app.get("/api/prediction-wallet")
async def get_prediction_wallet(
    _: None = Depends(_require_auth),
):
    """获取预测钱包信息（walletAddress + walletId，自动从 Binance API 获取）"""
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}

    wallet = await prediction_trader.fetch_wallet_info()
    if not wallet:
        return {"error": "未找到预测钱包，请先在 Binance App 中开通预测市场"}

    def _mask_addr(addr: str | None) -> str | None:
        """地址脱敏：仅展示前6后4位"""
        if not addr:
            return None
        return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 12 else "***"

    return {
        "wallet_address": _mask_addr(wallet.get("walletAddress")),
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
    # Fix #5/#6: 在锁下快照读取，确保 history 和 market_info 一致性
    # Fix #16: 限制返回最近 400 个点（约 1.5 小时），避免响应体过大
    async with _state_lock:
        history_snapshot = list(_pm_history)[-400:]
        market_snapshot = dict(_pm_market_info)

    return {
        "symbol": settings.symbol,
        "poll_interval_sec": 15,
        "points": history_snapshot,
        "market": market_snapshot,
    }


# ============================================================
# 情绪曲线分析 API
# ============================================================

# [DEPRECATED] 回测缓存与锁已退役，由 SentimentAgent Learn 阶段取代
# _last_backtest_result / _last_backtest_time / _backtest_lock 不再使用


@app.get("/api/sentiment/windows")
async def get_sentiment_windows(
    limit: int = 50,
    _: None = Depends(_require_auth),
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
async def run_momentum_predict(
    _: None = Depends(_require_auth),
):
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
    _: None = Depends(_require_auth),
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
    _: None = Depends(_require_auth),
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
            discovery_method=r.discovery_method,
            holdout_win_rate=r.holdout_win_rate,
            holdout_sample_count=r.holdout_sample_count,
            holdout_ci_lower=r.holdout_ci_lower,
            created_at=r.created_at,
            updated_at=r.updated_at,
        ).model_dump()
        for r in rows
    ]


@app.get("/api/sentiment/agent/patterns/{pattern_id}/history")
async def get_pattern_history(
    pattern_id: int,
    limit: int = 200,
    _: None = Depends(_require_auth),
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
    _: None = Depends(_require_auth),
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
        "queue_depth": agent_scheduler.queue_depth if agent_scheduler is not None else -1,
        "evolve_trigger_mode": settings.agent_evolve_trigger_mode,
        "new_validated_since_evolve": (
            agent_scheduler.new_validated_since_evolve if agent_scheduler is not None else 0
        ),
        "evolve_min_new_samples": settings.agent_evolve_min_new_samples,
        "metrics_summary": {
            "llm_total_cost": metrics_collector.get_snapshot().get("llm", {}).get("total_cost", 0.0),
            "llm_call_count": metrics_collector.get_snapshot().get("llm", {}).get("call_count", 0),
        },
    }


@app.get("/api/sentiment/agent/metrics")
async def get_agent_metrics(
    _: None = Depends(_require_auth),
):
    """
    查询 Agent 运行时详细指标（Req 17 可观测性）

    返回各阶段执行统计、LLM token 用量与估算成本、交易决策统计、队列深度。
    """
    return metrics_collector.get_snapshot()


@app.get("/api/sentiment/agent/evolution")
async def get_agent_evolution(
    days: int = 30,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """进化有效性看板（Item 1）。

    把「Agent 是否真的在进化」量化为可证伪的数字：总体决策胜率与是否
    跑赢随机基线（Wilson 95% 下界>0.5）、按天的样本外胜率趋势、前半程 vs
    近半程代际对比（change≠improvement 的判据）、以及按发现方法拆分的胜率
    （LLM_DEEP / PY_CLUSTER / LEGACY / UNMATCHED）。days 夹在 [1, 90]。
    """
    from .services.evolution_metrics import build_evolution_report

    days = max(1, min(days, 90))
    return await build_evolution_report(db, days=days)


def _collect_memory_state() -> tuple[dict, dict, int | None]:
    """采集进程内内存态指标，供 HealthService 融合。

    Returns:
        (metrics_snapshot, consecutive_failures, queue_depth)
        - metrics_snapshot: metrics_collector.get_snapshot()
        - consecutive_failures: {phase: 连续失败数}
        - queue_depth: 调度器当前队列深度；scheduler 未就绪时为 None
    """
    snapshot = metrics_collector.get_snapshot()
    consecutive_failures = {
        phase: metrics_collector.get_consecutive_failures(phase)
        for phase in ("PREDICT", "VALIDATE", "LEARN", "EVOLVE")
    }
    queue_depth = agent_scheduler.queue_depth if agent_scheduler is not None else None
    return snapshot, consecutive_failures, queue_depth


@app.get("/api/agent/health")
async def get_agent_health(
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    Agent 运行健康报告（监控系统主端点）

    聚合 5 类关键指标（窗口连续性 / predict 匹配率 / 置信度校准 / 调度器心跳 /
    LLM 错误率），派生告警与总体状态，并附自然语言诊断 summary，供人与 LLM
    直接读取做运行诊断。返回结构见 models.schemas.HealthReport。
    """
    from .services.health import health_service

    snapshot, consecutive_failures, queue_depth = _collect_memory_state()
    report = await health_service.build_report(
        db,
        metrics_snapshot=snapshot,
        consecutive_failures=consecutive_failures,
        queue_depth=queue_depth,
    )
    return report.model_dump()


# ============================================================
# 深度模式发现 API（双模式架构）
# ============================================================


@app.post("/api/sentiment/agent/deep-learn")
async def trigger_deep_learn(
    max_windows: int = 100,
    _: None = Depends(_require_auth),
):
    """
    触发手动深度模式发现（预览模式）。

    分析全量历史窗口，返回发现结果供用户审核。
    不写入 DB，需通过 /commit 端点确认写入。
    """
    if sentiment_agent is None:
        return {"status": "error", "message": "Agent 尚未初始化，请等待系统启动完成"}

    # P1-2: 端点入参 clamp 上限，防止外部传入超大 max_windows 拖垮采样
    max_windows = max(1, min(max_windows, settings.agent_deep_learn_max_windows_cap))
    try:
        result = await sentiment_agent.deep_learn(max_windows=max_windows)
        return {
            "status": "ok",
            "reasoning": result.get("reasoning", ""),
            "discoveries": result.get("discoveries", []),
            "count": len(result.get("discoveries", [])),
            "message": "预览模式，数据尚未写入 DB。确认后请调用 POST /api/sentiment/agent/deep-learn/commit",
        }
    except RuntimeError as e:
        # 并发冲突
        return {"status": "busy", "message": str(e)}
    except Exception as e:
        logger.error("深度分析失败: {}", e)
        return {"status": "error", "message": "深度分析失败，请查看服务端日志"}


@app.post("/api/sentiment/agent/deep-learn/stream")
async def stream_deep_learn(
    max_windows: int = 100,
    _: None = Depends(_require_auth),
):
    """流式深度模式发现（SSE）：逐 token 推送 LLM 输出，供前端实时打字机展示。

    与 POST /deep-learn 的一次性返回不同，本端点以 text/event-stream 逐帧推送：
    每帧一行 `data: <json>\\n\\n`，json.type ∈ {step, reasoning, progress, done, error}。
    done 帧携带最终 reasoning 与 discoveries（供前端勾选后走 /commit 写入）。
    不写 DB；成功后会落一条 DEEP_LEARN 轨迹。
    """
    # P1-2: 端点入参 clamp 上限
    max_windows = max(1, min(max_windows, settings.agent_deep_learn_max_windows_cap))

    async def event_gen():
        if sentiment_agent is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent 尚未初始化，请等待系统启动完成'}, ensure_ascii=False)}\n\n"
            return
        try:
            async for ev in sentiment_agent.deep_learn_stream(max_windows=max_windows):
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            logger.error("流式深度分析异常: {} | {}", type(e).__name__, e)
            payload = {"type": "error", "message": f"{type(e).__name__}: {e}"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 显式关闭代理缓冲，确保逐帧下发
        },
    )


@app.post("/api/sentiment/agent/deep-learn/commit")
async def commit_deep_learn(
    request: CommitDeepLearnRequest,
    _: None = Depends(_require_auth),
):
    """
    将用户确认的模式发现写入 pattern_memory。

    discoveries 来自 POST /api/sentiment/agent/deep-learn 的返回值。
    请求体使用 Pydantic Schema 校验，确保数据完整性。
    """
    if sentiment_agent is None:
        return {"status": "error", "message": "Agent 尚未初始化，请等待系统启动完成"}

    if not request.discoveries:
        return {"status": "error", "message": "discoveries 为空，无内容可写入"}

    # 将 Pydantic 模型转换为 dict 列表
    discoveries_dicts = [d.model_dump() for d in request.discoveries]

    # P2-2: 记录预览时的 snapshot_token 供审计；commit 侧不信任预览声明，
    # 而是用每条 discovery 携带的当次 holdout 统计独立重跑准入闸门（兜底一致性）。
    logger.info(
        "Commit Deep Learn: snapshot_token={} | discoveries={}",
        request.snapshot_token, len(discoveries_dicts),
    )
    try:
        # commit_deep_learn 返回 {status, written, rejected, failed}（P0-3 准入 / P1-4 失败收集）
        result = await sentiment_agent.commit_deep_learn(discoveries_dicts)
        return {
            "status": "ok",
            "written": result.get("written", 0),
            "rejected": result.get("rejected", []),
            "failed": result.get("failed", []),
        }
    except RuntimeError as e:
        # 并发冲突
        return {"status": "busy", "message": str(e)}
    except Exception as e:
        logger.error("深度分析写入失败: {}", e)
        return {"status": "error", "message": "写入失败，请查看服务端日志"}


def _summarize_discovery_group(result: dict) -> dict:
    """把一次 deep-learn 结果压成多维对比摘要（发现即时 holdout 维度）。

    汇总发现数、平均 holdout 胜率 / Wilson 下界、holdout 样本量、平均 confidence、
    通过 P0-3 准入闸门的比例、方向分布（UP/DOWN），供 /compare 面板对齐两套方案。
    """
    discoveries = result.get("discoveries", [])
    n = len(discoveries)
    win_rates = [d["holdout_win_rate"] for d in discoveries if d.get("holdout_win_rate") is not None]
    ci_lowers = [d["holdout_ci_lower"] for d in discoveries if d.get("holdout_ci_lower") is not None]
    confidences = [d["confidence_score"] for d in discoveries if d.get("confidence_score") is not None]
    total_samples = sum(int(d.get("holdout_sample_count") or 0) for d in discoveries)
    min_samples = settings.agent_deep_learn_min_holdout_samples
    passed = [
        d for d in discoveries
        if (d.get("holdout_ci_lower") or 0.0) > 0.5
        and (d.get("holdout_sample_count") or 0) >= min_samples
    ]
    up = sum(1 for d in discoveries if (d.get("predicted_direction") or "").upper() == "UP")
    down = sum(1 for d in discoveries if (d.get("predicted_direction") or "").upper() == "DOWN")

    def _avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "method": result.get("method"),
        "discovery_count": n,
        "avg_holdout_win_rate": _avg(win_rates),
        "avg_holdout_ci_lower": _avg(ci_lowers),
        "total_holdout_samples": total_samples,
        "avg_confidence": _avg(confidences),
        "passed_gate_count": len(passed),
        "passed_gate_ratio": round(len(passed) / n, 4) if n else 0.0,
        "direction_up": up,
        "direction_down": down,
        "snapshot_token": result.get("snapshot_token"),
        "train_count": result.get("train_count", 0),
        "holdout_count": result.get("holdout_count", 0),
    }


@app.post("/api/sentiment/agent/deep-learn/pycluster")
async def trigger_deep_learn_pycluster(
    max_windows: int = 100,
    _: None = Depends(_require_auth),
):
    """触发 Python 聚类版深度发现（全程无 LLM，确定性对照组，预览不写库）。

    与纯 LLM 版 /deep-learn 对称：返回 discoveries（每条含 discovery_method=PY_CLUSTER
    与 holdout 统计）+ snapshot_token + train/holdout 计数，供前端预览后走 /commit 写入。
    """
    if sentiment_agent is None:
        return {"status": "error", "message": "Agent 尚未初始化，请等待系统启动完成"}

    # P1-2: 端点入参 clamp 上限
    max_windows = max(1, min(max_windows, settings.agent_deep_learn_max_windows_cap))
    try:
        result = await sentiment_agent.deep_learn_pycluster(max_windows=max_windows)
        return {
            "status": "ok",
            "reasoning": result.get("reasoning", ""),
            "discoveries": result.get("discoveries", []),
            "count": len(result.get("discoveries", [])),
            "method": result.get("method", "PY_CLUSTER"),
            "snapshot_token": result.get("snapshot_token"),
            "train_count": result.get("train_count", 0),
            "holdout_count": result.get("holdout_count", 0),
            "message": "预览模式，数据尚未写入 DB。确认后请调用 POST /api/sentiment/agent/deep-learn/commit",
        }
    except RuntimeError as e:
        return {"status": "busy", "message": str(e)}
    except Exception as e:
        logger.error("聚类深度分析失败: {}", e)
        return {"status": "error", "message": "聚类深度分析失败，请查看服务端日志"}


@app.post("/api/sentiment/agent/deep-learn/compare")
async def compare_deep_learn(
    max_windows: int = 100,
    _: None = Depends(_require_auth),
):
    """同一采样窗口上依次跑 LLM 版与 Python 聚类版，返回对齐的多维对比（预览不写库）。

    两版均调用同一确定性采样，snapshot_token 应一致；不一致时 snapshot_consistent=False，
    前端需提示两次采样窗口不同（对比失真）。comparison 为 [LLM 摘要, PY 摘要]。
    """
    if sentiment_agent is None:
        return {"status": "error", "message": "Agent 尚未初始化，请等待系统启动完成"}

    # P1-2: 端点入参 clamp 上限
    max_windows = max(1, min(max_windows, settings.agent_deep_learn_max_windows_cap))
    try:
        llm_result = await sentiment_agent.deep_learn(max_windows=max_windows)
        py_result = await sentiment_agent.deep_learn_pycluster(max_windows=max_windows)
    except RuntimeError as e:
        return {"status": "busy", "message": str(e)}
    except Exception as e:
        logger.error("对比深度分析失败: {}", e)
        return {"status": "error", "message": "对比深度分析失败，请查看服务端日志"}

    return {
        "status": "ok",
        "snapshot_consistent": (
            llm_result.get("snapshot_token") == py_result.get("snapshot_token")
        ),
        "comparison": [
            _summarize_discovery_group(llm_result),
            _summarize_discovery_group(py_result),
        ],
        "llm": {
            "reasoning": llm_result.get("reasoning", ""),
            "discoveries": llm_result.get("discoveries", []),
        },
        "pycluster": {
            "reasoning": py_result.get("reasoning", ""),
            "discoveries": py_result.get("discoveries", []),
        },
    }


@app.get("/api/sentiment/agent/deep-learn/compare/live")
async def compare_deep_learn_live(
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """按 discovery_method 聚合 pattern_memory 的上线真实指标（LLM_DEEP vs PY_CLUSTER vs LEGACY）。

    live 维度用 Harness 维护的 win_rate/sample_count/correct_count（与发现时 holdout 分开存），
    反映模式上线后的真实表现。仅统计 ACTIVE 模式。
    """
    from sqlalchemy import func as sa_func, select as sa_select
    from .db.models import PatternMemory

    stmt = (
        sa_select(
            PatternMemory.discovery_method,
            sa_func.count(PatternMemory.id),
            sa_func.sum(PatternMemory.sample_count),
            sa_func.sum(PatternMemory.correct_count),
            sa_func.avg(PatternMemory.confidence_score),
            sa_func.avg(PatternMemory.holdout_ci_lower),
        )
        .where(PatternMemory.status == "ACTIVE")
        .group_by(PatternMemory.discovery_method)
    )
    rows = (await db.execute(stmt)).all()
    groups = []
    for method, cnt, samples, correct, avg_conf, avg_ci in rows:
        samples = int(samples or 0)
        correct = int(correct or 0)
        groups.append({
            "method": method,
            "pattern_count": int(cnt or 0),
            "live_sample_count": samples,
            "live_correct_count": correct,
            "live_win_rate": round(correct / samples, 4) if samples else 0.0,
            "avg_confidence": round(float(avg_conf), 4) if avg_conf is not None else 0.0,
            "avg_holdout_ci_lower": round(float(avg_ci), 4) if avg_ci is not None else 0.0,
        })
    return {"status": "ok", "groups": groups}


# ============================================================
# LLM 轨迹审计 API（前端「LLM 轨迹」面板 / 流程审查）
# ============================================================


@app.get("/api/llm/traces")
async def get_llm_traces(
    phase: str | None = None,
    limit: int = 50,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    查询 LLM 调用轨迹列表（倒序，最新在前）。

    列表仅返回轻量摘要字段（不含完整 prompt/输出），供前端悬浮面板快速浏览。
    支持按 phase（LEARN|DEEP_LEARN|PREDICT|EVOLVE）筛选，limit 限制条数（默认 50）。
    """
    from sqlalchemy import select as sa_select
    from .db.models import LLMTrace
    from .models.schemas import LLMTraceSummary

    limit = max(1, min(limit, 200))
    stmt = sa_select(LLMTrace)
    if phase:
        stmt = stmt.where(LLMTrace.phase == phase)
    stmt = stmt.order_by(LLMTrace.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        LLMTraceSummary(
            id=r.id,
            phase=r.phase,
            model=r.model,
            reasoning=r.reasoning,
            result_summary=r.result_summary,
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            estimated_cost_yuan=r.estimated_cost_yuan,
            latency_s=r.latency_s,
            created_at=r.created_at,
        ).model_dump()
        for r in rows
    ]


@app.get("/api/llm/traces/{trace_id}")
async def get_llm_trace_detail(
    trace_id: int,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    查询单条 LLM 调用轨迹完整详情（含 system_prompt / user_message / 结构化输出）。
    """
    from sqlalchemy import select as sa_select
    from .db.models import LLMTrace
    from .models.schemas import LLMTraceRecord

    stmt = sa_select(LLMTrace).where(LLMTrace.id == trace_id)
    result = await db.execute(stmt)
    r = result.scalar_one_or_none()
    if r is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="轨迹不存在")

    return LLMTraceRecord(
        id=r.id,
        phase=r.phase,
        model=r.model,
        reasoning=r.reasoning,
        result_summary=r.result_summary,
        prompt_tokens=r.prompt_tokens,
        completion_tokens=r.completion_tokens,
        estimated_cost_yuan=r.estimated_cost_yuan,
        latency_s=r.latency_s,
        created_at=r.created_at,
        system_prompt=r.system_prompt,
        user_message=r.user_message,
        assistant_output=r.assistant_output,
    ).model_dump()
