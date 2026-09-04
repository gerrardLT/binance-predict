"""
BTC 预测市场多通道实盘 + 影子信号系统 - FastAPI 主应用

系统入口文件，负责：
1. 初始化服务（数据采集、多通道实盘执行器、影子检测器、交易结算）
2. 管理应用生命周期（lifespan）：装配并启动 MultiLiveTrader（15 通道实盘）
   + 多个影子检测器（KREV/HM/反转/nextbar/combo/X4/报价 edge，只记录不下注）
   + 场景检测器（FakeBreakoutDetector）；AgentScheduler 四阶段闭环仅在
   agent_loop_enabled=True 时启动（2026-08-16 起默认退役）
3. 注册 API 路由

核心引擎：MultiLiveTrader 多通道实盘执行器（真单），三族触发——quote_edge
（5m 报价区间命中）/ x4（错位影子 PENDING 轮询）/ scene（假突破钩子）。
情绪窗口归档器持续运行，作为场景信号的 4h 位势数据源。概率动量分析
（MomentumService）为手动接口，不参与自动决策。
"""

from __future__ import annotations

import asyncio
import calendar
import hmac
import json
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config.settings import settings
from .db.engine import async_session_factory, get_db
from .db.models import (
    Base,
    FakeBreakoutSignal,
    PatternBacktestRun,
    PredictionMarketSample,
    SceneParamVersion,
    SentimentWindow,
)
from .models.schemas import (
    CommitDeepLearnRequest,
    LoginRequest,
    ManualTradeTestRequest,
    RedeemRequest,
    TransferInboundRequest,
    TransferOutboundRequest,
    ToggleLiveRequest,
    ToggleShadowRequest,
)
from .services.agent_scheduler import AgentScheduler
from .services.combo_shadow_detector import ComboShadowDetector
from .services.data_collector import BinanceDataCollector
from .services.fake_breakout_detector import FakeBreakoutDetector
from .services.hm_shadow_detector import HmShadowDetector
from .services.kline_shadow_detector import KlineShadowDetector
from .services.misalignment_detector import MisalignmentDetector
from .services.multi_live_trader import MultiLiveTrader
from .services.nextbar_shadow_detector import NextbarShadowDetector
from .services.quote_edge_detector import QuoteEdgeDetector
from .services.reversal_shadow_detector import ReversalShadowDetector
from .services.shadow_version_gate import shadow_gate
from .services.trade_settler import TradeSettler
from .services.llm_service import LLMService
from .services.pattern_reevaluator import pattern_reevaluator
from .services.prediction_trading import BinancePredictionTrader
from .services.prediction_market_data import PredictionMarketDataService, MarketQuoteData
from .services.sentiment_agent import SentimentAgent
from .services.signal_notify import set_live_enabled_resolver
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

# 预测钱包视图 TTL 缓存（P0-1）：静态信息 300s / 余额 20s + 单飞锁。
# 前端 15s 轮询 × N 客户端收敛为全服务每 20s 最多 1 组签名调用，
# 单飞锁防缓存击穿 stampede；资金变动端点（划转）成功后主动置零时间戳作废。
_wallet_view: dict = {}
_wallet_view_ts: dict[str, float] = {"static": 0.0, "balance": 0.0}
_wallet_view_lock = asyncio.Lock()
_WALLET_STATIC_TTL = 300.0
_WALLET_BALANCE_TTL = 20.0

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

# 15m 市场最新报价缓存（tracker 15s 对齐采样 + 边界加速协程（边界后 40s 内 2s 粒度）
# 刷新，假突破检测器读取信号时刻的 15m DOWN 报价与到期时刻；只读共享，无需加锁——
# 单写者事件循环 + 读者可容忍毫秒级旧值，逐 key 赋值段无 await 不交错）
_pm_15m_latest: dict = {
    "down_price": None,
    "up_price": None,
    "start_date": None,
    "end_date": None,
    "updated_ts": None,
    # 周期开盘价（假突破周期锚点结算的判定基准 P(S)）：周期切换时快照，
    # 冷启动时从 klines 精确回读；cycle_open_end 记录其所属周期供配对校验
    "cycle_open_price": None,
    "cycle_open_end": None,
}

# 5m 周期开盘价快照（tracker 窗口切换时更新；_pm_market_info 每轮 clear+update，
# 故用模块级变量暂存，供其复制；假突破检测器周期锚点结算的 5m 判定基准）
_pm_5m_cycle_open_price: float | None = None
_pm_5m_cycle_open_end: int | None = None

# PREDICT 事件触发标志：同一窗口仅触发一次（Req 3.1），窗口切换时重置
_predict_triggered_for_window: bool = False

# 假突破检测器全局实例（lifespan 中初始化；秒级检测日线阻力破位，暂不下注）
fake_breakout_detector: FakeBreakoutDetector | None = None

# X4 情绪错位影子检测器全局实例（M4 影子并行：收阳&end≤40→次窗DOWN，只记录不下注）
misalignment_detector: MisalignmentDetector | None = None

# 报价 edge 影子检测器全局实例（A 顺势 q∈[0.69,0.75) / B 逆势 q∈[0.15,0.25)，只记录不下注）
quote_edge_detector: QuoteEdgeDetector | None = None

# K 线科学发现影子检测器全局实例（KREV 族：720d 冻结注册表条件实时重放，只记录不下注）
kline_shadow_detector: KlineShadowDetector | None = None

# HM 上吊线反弹入场影子检测器全局实例（hm_touch_down_v1：窗内等价位触发，只记录不下注）
hm_shadow_detector: HmShadowDetector | None = None

# 反转形态影子检测器全局实例（P1/P2：15m 连跌/连涨后弱收盘几何反转，只记录不下注）
reversal_shadow_detector: ReversalShadowDetector | None = None

# 下一根 K 线方向影子检测器全局实例（nextbar 族：15m 冠军 + 5m sma_slope 误定价候选，只记录不下注）
nextbar_shadow_detector: NextbarShadowDetector | None = None

# 组合条件影子检测器全局实例（combo 族：45 维大搜索存活 5 组合，只记录不下注）
combo_shadow_detector: ComboShadowDetector | None = None

# 交易结算器全局实例（P0-2：FILLED 订单结算回填输赢/盈亏，常开）
trade_settler: TradeSettler | None = None

# 多通道实盘执行器全局实例（MultiLiveTrader：15 通道三族触发，
# 通道开关/金额/日限各自独立，启动回落 LIVE_CHANNELS_JSON，默认全关）
multi_live_trader: MultiLiveTrader | None = None

# 场景研究调度器全局实例（M2：LLM 研究员触发与编排，lifespan 中初始化）
research_scheduler: "ResearchScheduler | None" = None


# ============================================================
# 定时任务
# ============================================================

# 边界加速采样参数：15m 周期边界后短窗口内加密刷新（只刷内存缓存，不落库）。
# 目的：入场报价快照观测点从开盘后 ~20s 推到 ~8s 内；cycle_open_price（周期
# 锚点 P(S)）滞后从 ≤15s 压到 ≤2s。DB 采样粒度不变（15s 对齐，归档/曲面口径不变）。
EDGE_ACCEL_INTERVAL_S = 2      # 加密刷新间隔
EDGE_ACCEL_WINDOW_MS = 40_000  # 15m 边界后的加速窗口


async def _handle_15m_quote(quote_15m, ts_ms: int, *, persist: bool) -> None:
    """15m 市场通道：切换检测（cycle_open 快照）+ 缓存刷新 + 可选落库。

    两条路径共用：tracker 15s 对齐采样（persist=True）与边界加速协程
    （persist=False）。切换检测基于 end_date 变化，天然幂等——两路径
    并发调用安全（单线程事件循环，逐 key 赋值段无 await 不交错）。
    """
    if quote_15m is None or quote_15m.down_price is None or quote_15m.end_date is None:
        return
    # 周期切换检测：记录新周期开盘价（周期锚点结算的判定基准 P(S)）
    prev_end_15m = _pm_15m_latest.get("end_date")
    if quote_15m.end_date != prev_end_15m:
        if prev_end_15m is None:
            # 冷启动：当前周期已开始，从 klines 精确回读周期开盘价
            start_15m = quote_15m.start_date or (int(quote_15m.end_date) - 900_000)
            open_15m = await collector.fetch_kline_open("15m", int(start_15m))
        else:
            # 正常切换：切换时刻现价即新周期开盘价（加速路径 ≤2s / 常规 ≤15s 滞后）
            open_15m = collector.store.mid_price
            if not open_15m or open_15m <= 0:
                open_15m = await collector.fetch_mid_price()
        _pm_15m_latest["cycle_open_price"] = open_15m if open_15m and open_15m > 0 else None
        _pm_15m_latest["cycle_open_end"] = quote_15m.end_date
        logger.info(
            "15m 市场周期切换 | 开盘价 {} | {} → {}",
            _pm_15m_latest["cycle_open_price"], prev_end_15m, quote_15m.end_date,
        )
    _pm_15m_latest["down_price"] = quote_15m.down_price
    _pm_15m_latest["up_price"] = quote_15m.up_price
    _pm_15m_latest["start_date"] = quote_15m.start_date
    _pm_15m_latest["end_date"] = quote_15m.end_date
    _pm_15m_latest["updated_ts"] = ts_ms
    if not persist:
        return
    try:
        async with async_session_factory() as db:
            db.add(PredictionMarketSample(
                timestamp=ts_ms,
                market_period="15m",
                up_price=quote_15m.up_price,
                down_price=quote_15m.down_price,
                up_pct=round(quote_15m.up_chance * 100, 1) if quote_15m.up_chance is not None else None,
                down_pct=round(quote_15m.down_chance * 100, 1) if quote_15m.down_chance is not None else None,
                participants=quote_15m.participants,
                trade_volume=float(quote_15m.trade_volume) if quote_15m.trade_volume is not None else None,
                btc_price=collector.store.mid_price or None,
            ))
            await db.commit()
    except Exception as e:
        logger.warning("15m 市场采样入库失败: {}", e)


async def _pm_15m_edge_accelerator() -> None:
    """15m 周期边界加速采样：边界后 40s 内每 2s 刷新缓存（不落库）。

    入场报价快照与 cycle_open_price 精度的关键路径——其余时间不产生
    任何额外 API 请求。API 增量：每边界 ~20 次（每 15 分钟），全天 ~1900 次。
    """
    last_poll = 0.0
    while True:
        try:
            await asyncio.sleep(EDGE_ACCEL_INTERVAL_S)
            now_ms = int(time.time() * 1000)
            if now_ms % 900_000 >= EDGE_ACCEL_WINDOW_MS:
                continue
            if time.time() - last_poll < EDGE_ACCEL_INTERVAL_S:
                continue
            last_poll = time.time()
            quotes = await market_data_service.fetch_market_data_multi()
            await _handle_15m_quote(quotes.get("15m"), int(time.time() * 1000), persist=False)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug("边界加速采样异常（下一 tick 重试）: {}", e)


async def _prediction_market_tracker() -> None:
    """
    预测市场情绪追踪：每 15s 轮询 UP/DOWN token 报价

    通过 PredictionMarketDataService（只读）获取报价数据，
    记录到 _pm_history（内存）+ prediction_market_samples（DB）。
    启动时从 DB 加载最近 2000 条历史记录。

    读写分离设计：本函数不再调用 prediction_trader.list_markets()，
    避免修改交易模块状态（Bug 1.1 修复）。
    """
    global _current_window_end, _window_entry_price, _predict_triggered_for_window
    global _last_closed_window_end, _last_closed_window_entry_price, _last_closed_window_exit_price
    global _pm_5m_cycle_open_price, _pm_5m_cycle_open_end

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

            # 通过只读服务获取市场报价（5m + 15m 双周期，不修改交易模块状态）
            try:
                quotes = await market_data_service.fetch_market_data_multi()
            except Exception:
                continue

            # --- 15m 市场通道：共用处理（切换检测 + 缓存 + 落库；边界加速协程共用同一函数） ---
            await _handle_15m_quote(quotes.get("15m"), aligned_ts, persist=True)

            quote = quotes.get("5m")
            if quote is None:
                continue

            # Bug 1.2 修复：end_date=None 防御
            if quote.end_date is None:
                logger.warning("end_date 为 None，跳过本轮采样")
                continue

            # 更新市场元数据（供图表 API 只读访问；场景检测器已不共享此 dict）。
            _pm_market_info.clear()
            _pm_market_info.update({
                "participant_count": quote.participants,
                "trade_volume": quote.trade_volume,
                "start_date": quote.start_date,
                "end_date": quote.end_date,
                "up_price": quote.up_price,
                "down_price": quote.down_price,
                "up_chance": quote.up_chance,
                "down_chance": quote.down_chance,
                # 报价刷新时刻（ms）：供 K 线族影子入场报价快照的新鲜度/近开盘守卫
                "updated_ts": aligned_ts,
                # 周期开盘价（周期锚点结算 P(S5)）：窗口切换时更新的模块级快照
                "cycle_open_price": _pm_5m_cycle_open_price,
                "cycle_open_end": _pm_5m_cycle_open_end,
            })

            # 检测 5 分钟窗口切换：end_date 变化说明进入了新市场
            new_window_end = quote.end_date
            if new_window_end != _current_window_end:
                # Fix #5: 使用锁保护全局状态写入，防止 archiver 读到半写状态
                async with _state_lock:
                    prev_window_end = _current_window_end
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
                    # 冷启动（含部署重启）时当前窗口已开始：klines 精确回读窗口
                    # 开盘价作双用锚点（入场快照 + 周期开盘价）。否则用重启时刻的
                    # mid_price 当 entry → 归档 outcome 翻转 → 订单误结算
                    # （2026-08-27 实锤：20:14 重启污染 20:10~20:15 窗，DOWN 单
                    # 误判赢虚记 +30.33，币安侧实为 UP）。正常切换（进程存活跨越
                    # 边界）内存 mid_price 即边界价，无需 REST。
                    kline_open_5m = 0.0
                    if prev_window_end is None:
                        start_5m = int(quote.start_date) if quote.start_date else int(new_window_end) - 300_000
                        kline_open_5m = await collector.fetch_kline_open("5m", start_5m)
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
                    # 冷启动 kline 回读成功时以其为准（重启落在窗中时 mid_price 是
                    # 窗中价，不是开盘价）；归档器另有断链自愈兜底（双保险）
                    if kline_open_5m > 0:
                        _window_entry_price = kline_open_5m
                    # 周期开盘价快照（假突破 5m 周期锚点结算判定基准）：
                    # 正常切换 = 新窗口起点快照；冷启动时当前周期已开始，klines 精确回读
                    if prev_window_end is None:
                        _pm_5m_cycle_open_price = kline_open_5m if kline_open_5m > 0 else None
                    else:
                        _pm_5m_cycle_open_price = (
                            _window_entry_price if _window_entry_price and _window_entry_price > 0 else None
                        )
                    _pm_5m_cycle_open_end = int(new_window_end)
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
                            # market_period=5m 过滤：与归档同口径，避免把 15m 样本混入
                            # 当前窗 _pm_history（供 PREDICT 事件的 current_curve）。
                            stmt = (
                                sa_select(PredictionMarketSample)
                                .where(PredictionMarketSample.timestamp >= window_start_ms)
                                .where(PredictionMarketSample.timestamp < window_end_ms)
                                .where(PredictionMarketSample.market_period == "5m")
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
                                    "btc_price": row.btc_price,
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
                # BTC 现货中间价快照（与情绪采样同时刻）：验证情绪领先/滞后
                # 价格的关键证据。内存快照无效时存 None，不阻塞、不伪造。
                _btc_mid = collector.store.mid_price
                point = {
                    "timestamp": aligned_ts,
                    "up_price": up_price,
                    "down_price": down_price,
                    "up_pct": round(up_chance * 100, 1) if up_chance is not None else None,
                    "down_pct": round(down_chance * 100, 1) if down_chance is not None else None,
                    "participants": quote.participants,
                    "trade_volume": float(quote.trade_volume) if quote.trade_volume is not None else None,
                    "btc_price": _btc_mid if _btc_mid and _btc_mid > 0 else None,
                }
                _pm_history.append(point)

                # 持久化到 DB
                try:
                    async with async_session_factory() as db:
                        db.add(PredictionMarketSample(
                            timestamp=point["timestamp"],
                            market_period="5m",
                            up_price=point["up_price"],
                            down_price=point["down_price"],
                            up_pct=point["up_pct"],
                            down_pct=point["down_pct"],
                            participants=point["participants"],
                            trade_volume=point["trade_volume"],
                            btc_price=point["btc_price"],
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
                    # 构建 current_curve：当前窗口三通道时序切片（宪法第八条规则 3：
                    # sentiment/price/volume 实时采样齐备，L2 跨通道谓词在线可执行；
                    # btc_price/trade_volume 为 None 的点被符号化层跳过，不阻塞）
                    current_curve = [
                        {
                            "t": p["timestamp"],
                            "up_pct": p["up_pct"],
                            "down_pct": p["down_pct"],
                            "btc_price": p.get("btc_price"),
                            "trade_volume": p.get("trade_volume"),
                        }
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

                # 多通道实盘（quote_edge 族）：DOWN 报价首次进所绑版本规则区间 → 真单。
                # None 守卫：未装配时跳过；check 内纯内存比较，不阻塞采样循环。
                # v2 门禁喂价：_btc_mid（本次采样 BTC 中间价，L397）+ _window_entry_price
                # （窗口开盘快照）——均为现有变量，零新采样成本。
                # v3 日高喂价：window_btc_curve 从 _pm_history 提取本窗 BTC 时序
                # （含本点），供 v3b 日高口径与影子对齐（影子含本窗 ≤quote_ts 点）。
                if multi_live_trader is not None and _current_window_end is not None:
                    multi_live_trader.check(
                        int(_current_window_end) - 300_000,
                        int(_current_window_end),
                        aligned_ts,
                        down_price,
                        btc_price=_btc_mid,
                        window_entry_price=_window_entry_price,
                        window_btc_curve=[
                            {"t": p["timestamp"], "v": p["btc_price"]}
                            for p in _pm_history if p.get("btc_price")
                        ],
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
                # 查询窗口内的采样点。
                # market_period=5m 过滤（关键）：主循环每轮同时写 5m + 15m 两条样本
                # （同一时间戳）。本窗口是 5m 情绪窗，若不过滤会把 15m 市场报价混入
                # 曲线 → 下游影子检测器扫出 15m 幻影触发（实盘只看 5m，口径不一致）。
                # 与回测脚本（local_quote_bin_winrate / local_edge_cell_constraints
                # 均 market_period=="5m"）及实盘执行器对齐。走 ix_pm_samples_period_ts 索引。
                stmt = (
                    sa_select(PredictionMarketSample)
                    .where(PredictionMarketSample.timestamp >= start_ms)
                    .where(PredictionMarketSample.timestamp < end_ms)
                    .where(PredictionMarketSample.market_period == "5m")
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
                    # 价格曲线永久化：采样表仅保留 1 小时，归档时快照否则永久丢失（经济账依赖）
                    curve_up_price = [{"t": s.timestamp, "v": s.up_price} for s in samples if s.up_price is not None]
                    curve_down_price = [{"t": s.timestamp, "v": s.down_price} for s in samples if s.down_price is not None]
                    # 参与者/交易量时序永久化：momentum 类假设的原始证据（此前仅存均值）
                    curve_participants = [{"t": s.timestamp, "v": s.participants} for s in samples if s.participants is not None]
                    curve_trade_volume = [{"t": s.timestamp, "v": s.trade_volume} for s in samples if s.trade_volume is not None]
                    # BTC 局内价格曲线：与情绪曲线同步的现货中间价序列，
                    # 情绪 vs 价格领先/滞后分析的原始证据（归档永久化）
                    curve_btc_price = [{"t": s.timestamp, "v": s.btc_price} for s in samples if s.btc_price is not None]

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

                    # 断链自愈（2026-08-27 误结算事故）：冷启动落在窗中时快照的
                    # entry 是窗中价（与前窗 exit 断链），klines 精确回读开盘价替代。
                    # 归档先于结算（+7min），此处修正即阻断后续订单误结算。
                    from .services.archive_contamination_repair import correct_entry_break
                    repaired_entry = await correct_entry_break(
                        db, start_ms, entry_price, collector)
                    if repaired_entry is not None:
                        entry_price = repaired_entry

                    # 计算实际结果（结算口径对齐）：预测市场只按涨跌方向赔付，
                    # 与幅度无关，故 outcome 按 actual_return 正负号标注；恰好为 0
                    # （极罕见，历史数据 16/3522）无法结算方向，标 NOISE。
                    # noise_threshold 不再参与结果标注，仅保留作策略层横盘过滤备用。
                    actual_return = None
                    outcome = None
                    if entry_price and exit_price and entry_price > 0:
                        actual_return = exit_price / entry_price - 1
                        if actual_return > 0:
                            outcome = "UP"
                        elif actual_return < 0:
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
                        curve_up_price=curve_up_price,
                        curve_down_price=curve_down_price,
                        curve_participants=curve_participants,
                        curve_trade_volume=curve_trade_volume,
                        curve_btc_price=curve_btc_price,
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

                # 旧采样清理：仅当 sample_retention_hours > 0 时执行。
                # 默认 <=0 永不删除——原始采样是唯一的"全字段事实源"，归档表只含
                # 归档时想到要存的字段（历史价格曲线因旧 1 小时清理永久丢失即为教训）。
                if settings.sample_retention_hours > 0:
                    cleanup_threshold_ms = end_ms - int(
                        settings.sample_retention_hours * 3600 * 1000
                    )
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


async def _binance_reconcile_loop() -> None:
    """币安订单定时对账循环（2026-08-28）：启动 5min 后首跑，此后每 30min 一次。

    与手动 /api/trades/sync-binance 共用 _sync_binance_orders_impl：
    PENDING 订单按 window_start 对账回填终态 + 幽灵成交订正，替代此前
    "仅手动触发"的对账节奏（本地结算价源与币安存在差异，靠此自动订正）。
    API Key 未配置 → impl 返回 error，本轮跳过下轮再查；异常只日志不退出
    （下轮重试）。结果仅日志，无推送。
    """
    await asyncio.sleep(300)  # 启动缓冲：等 WS 喂价/实盘下单链路稳定，避开启动风暴
    while True:
        try:
            result = await _sync_binance_orders_impl()
            if result.get("error"):
                logger.info("币安定时对账本轮跳过 | {}", result["error"])
            else:
                logger.info(
                    "币安定时对账完成 | 订正 {} 条 | 币安侧订单 {} 条",
                    result.get("synced", 0), result.get("binance_orders", 0),
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("币安定时对账异常（下轮重试）| {}", exc)
        await asyncio.sleep(1800)


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
                # 参与者/交易量时序曲线（与 alembic 迁移 a7b8c9d0e1f2 等价，存量 dev 库安全网）
                "ALTER TABLE sentiment_windows ADD COLUMN IF NOT EXISTS curve_participants JSONB",
                "ALTER TABLE sentiment_windows ADD COLUMN IF NOT EXISTS curve_trade_volume JSONB",
                # 假突破信号系统 + 模式池分级（与 alembic 迁移 h8b9c0d1e2f3 等价，存量 dev 库安全网）
                "ALTER TABLE prediction_market_samples ADD COLUMN IF NOT EXISTS market_period VARCHAR(5) NOT NULL DEFAULT '5m'",
                "ALTER TABLE pattern_memory ADD COLUMN IF NOT EXISTS tier VARCHAR(2) NOT NULL DEFAULT 'C'",
                # 假突破 5m 兑现口径（与 alembic 迁移 i9b0c1d2e3f4 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS settle_btc_price_5m FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS settle_outcome_5m VARCHAR(10)",
                # 假突破三级别双向（与 alembic 迁移 j0c1d2e3f4g5 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS level VARCHAR(8) NOT NULL DEFAULT 'daily'",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS side VARCHAR(4) NOT NULL DEFAULT 'high'",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS up_price_5m FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS up_price_15m FLOAT",
                # 假突破周期锚点结算口径（与 alembic 迁移 k1d2e3f4g5h6 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS market_start_15m BIGINT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS cycle_open_price_15m FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS market_start_5m BIGINT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS market_end_5m BIGINT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS cycle_open_price_5m FLOAT",
                # 假突破行动过滤器指标（与 alembic 迁移 l2d3e4f5g6h7 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS cycle_offset_sec_15m INTEGER",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS break_pct FLOAT",
                # 场景收盘确认（与 alembic 迁移 n4e5f6g7h8i9 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS pattern VARCHAR(16)",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS close_pos FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS vol_ratio FLOAT",
                # M4 影子并行（与 alembic 迁移 p4g7h8i9j0k1 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS version VARCHAR(40)",
                # 场景统计维度（与 alembic 迁移 r8i9j0k1l2m3 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS pattern_type VARCHAR(32)",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS ev_at_entry FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS cumulative_winrate FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS cumulative_ev FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS n_events_last_7d INTEGER",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS max_drawdown_curves JSONB",
                # +5min 报价快照 / S5 确认入场（与 alembic 迁移 t9j0k1l2m3n4 等价，存量 dev 库安全网）
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS quote5m_down_15m FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS quote5m_up_15m FLOAT",
                "ALTER TABLE fake_breakout_signals ADD COLUMN IF NOT EXISTS quote5m_ts_15m BIGINT",
                # 信号关联字段 + 唯一约束/索引（与 alembic 迁移 u6f7g8h9i0j1 等价，存量 dev 库安全网）
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS signal_version VARCHAR(40)",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS window_start BIGINT",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS signal_id INTEGER",
                "ALTER TABLE trade_orders ADD CONSTRAINT uq_trade_orders_version_window UNIQUE (signal_version, window_start)",
                "CREATE INDEX IF NOT EXISTS ix_trade_orders_signal_version ON trade_orders (signal_version)",
                # token_id 扩宽（与 alembic 迁移 w7a8b9c0d1e2 等价，存量 dev 库安全网；重复执行幂等）
                "ALTER TABLE trade_orders ALTER COLUMN token_id TYPE VARCHAR(128)",
                # 结算闭环字段（与 alembic 迁移 x2y3z4a5b6c7 等价，存量 dev 库安全网；重复执行幂等）
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS direction VARCHAR(8)",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS settle_outcome VARCHAR(10)",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS win BOOLEAN",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS settle_price FLOAT",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS pnl FLOAT",
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ",
                # 奖金领取标记（与 alembic 迁移 l2e3f4g5h6i7 等价，存量 dev 库安全网）
                "ALTER TABLE trade_orders ADD COLUMN IF NOT EXISTS redeemed_at TIMESTAMPTZ",
                ("COMMENT ON COLUMN trade_orders.status IS "
                 "'PENDING | FILLED | FAILED（订单生命周期）；结算结果见 settle_outcome/win/pnl'"),
                ("CREATE INDEX IF NOT EXISTS ix_trade_orders_settle_pending ON trade_orders (window_start) "
                 "WHERE status = 'FILLED' AND settled_at IS NULL AND window_start IS NOT NULL"),
                # 执行价护栏自定义覆盖（与 alembic 迁移 y3z4a5b6c7d8 等价，存量 dev 库安全网）
                ("ALTER TABLE live_channel_overrides ADD COLUMN IF NOT EXISTS "
                 "max_exec_price FLOAT"),
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
    # 2026-08-16 系统B退役：agent_loop_enabled=False 时不实例化（预测循环停用），
    # 全局 agent_scheduler/sentiment_agent 保持 None——tracker/archiver 的
    # publish 调用已有 None 守卫，事件静默跳过；情绪窗口归档器不受影响。
    global agent_scheduler, sentiment_agent
    if settings.agent_loop_enabled:
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
    else:
        logger.info("系统B预测循环已退役（agent_loop_enabled=False）| 情绪窗口归档继续运行（场景信号位势源）")

    tasks = [
        asyncio.create_task(collector.connect_spot_ws(), name="spot_ws"),
        asyncio.create_task(_prediction_market_tracker(), name="pm_tracker"),
        asyncio.create_task(_pm_15m_edge_accelerator(), name="pm_15m_edge_accel"),
        asyncio.create_task(_sentiment_window_archiver(), name="sw_archiver"),
        asyncio.create_task(_health_monitor_loop(), name="health_monitor"),
        asyncio.create_task(_binance_reconcile_loop(), name="binance_reconcile"),
    ]
    logger.info("现货 WS + 预测市场追踪 + 15m边界加速 + 情绪窗口归档 + 健康监控已启动")

    # 多通道实盘执行器（MultiLiveTrader，2026-08-24 取代单版本 QuoteEdgeLiveTrader）：
    # 15 通道各自独立金额/日限/护栏/开关，三族触发（quote_edge 喂价/x4 轮询/场景钩子）；
    # 无条件装配 + 通道标志位控制开火；配置分层：代码默认 → LIVE_CHANNELS_JSON
    # → DB 覆盖层（toggle 持久化，重启不丢设定）。
    # 构造与邮件闸 resolver 注入必须早于三个检测器 start：is_enabled 只读
    # __init__ 就绪的 _configs 不依赖 start()；若注入滞后，短暂重启后新鲜度窗
    # （10 分钟）内的信号会命中首轮轮询——resolver 未注入 → 落表但永久丢推
    #（邮件 fire-once 无补发，2026-08-25 评审发现的时序缝隙）。
    global multi_live_trader
    try:
        multi_live_trader = MultiLiveTrader(prediction_trader)
    except ValueError as exc:
        # 通道配置非法（未知通道名/金额超硬上限/护栏非法）：拒实盘不拖垮其他服务
        logger.error("多通道实盘执行器装配失败：{}", exc)
    else:
        # DB 覆盖层（用户最后一次 toggle 的设定，优先级高于 env）；
        # fail-safe：DB 异常回落 env 层配置，不阻塞服务启动
        try:
            await multi_live_trader.apply_db_overrides()
        except Exception as exc:
            logger.warning("多通道实盘：DB 覆盖层加载失败（回落 env 配置）| {}", exc)
        # 信号邮件推送闸：只推已开实盘开火通道的信号（运行时读 configs，
        # 含 toggle 热调；装配失败时 resolver 保持 None → 一律不推 fail-safe）
        set_live_enabled_resolver(multi_live_trader.is_enabled)
        # v3 非连涨门禁 K 线源注入（与影子检测器同 collector；未注入时
        # v3 通道保守不开火，与影子「collector 缺失不落 v3」同口径）
        multi_live_trader.kline_fetcher = collector.fetch_recent_klines

    # 影子版本开关 gate（前端手动下线能力）：先于所有影子检测器启动，保证首轮
    # 落库判定可用；DB 故障保守全在线（fail-safe，不停采集）。
    await shadow_gate.start()

    # 场景信号系统：4h 破位记 pending → 15m 周期收盘确认 → 次周期信号（不下注）
    global fake_breakout_detector
    if settings.fake_breakout_enabled:
        fake_breakout_detector = FakeBreakoutDetector(
            collector=collector,
            pm_15m_latest=_pm_15m_latest,
        )
        await fake_breakout_detector.start()
        logger.info("场景检测器已启动（S1多头耗尽/S2空头耗尽/S4动量衰竭，真 OOS 修正版，信号模式不下注）")

    # 归档污染一次性自愈（幂等）：历史 5m 归档窗曾被 15m 样本污染（写入路径
    # 已由 commit 5fe6356 修复），此处从原始 5m 采样重建污染窗曲线并重落影子
    # 信号；失败不阻塞启动。必须早于两个影子检测器冷启动回扫。
    try:
        from binance_predict.services.archive_contamination_repair import (
            repair_contaminated_archives,
        )
        await repair_contaminated_archives()
    except Exception as exc:
        logger.error("归档污染自愈失败（不阻塞启动）: {}", exc)

    # 冷启动入场价断链自愈（幂等）：部署重启落在窗中会污染该窗 entry_price →
    # outcome 翻转 → 订单误结算（2026-08-27 事故）。修复近 24h 断链窗并重结算
    # 已结算订单；失败不阻塞启动。
    try:
        from binance_predict.services.archive_contamination_repair import (
            heal_entry_break_windows,
        )
        await heal_entry_break_windows(collector)
    except Exception as exc:
        logger.error("断链自愈失败（不阻塞启动）: {}", exc)

    # X4 情绪错位影子信号（M4）：收阳&end≤40→押次窗DOWN，只记录不下注，
    # 次窗归档后回读真实报价与结算，攒 2~3 周定案经济账后人工 promote
    global misalignment_detector
    if settings.misalignment_enabled:
        misalignment_detector = MisalignmentDetector()
        await misalignment_detector.start()
        logger.info("X4 影子检测器已启动（错位假设工厂产物，回测 63.5%/EV+0.254，影子模式不下注）")

    # 报价 edge 影子信号（A 顺势/B 逆势）：报价分箱校准发现的两格错价，
    # 归档后处理首个命中报价直接落 SETTLED，攒 2 周线上样本复核回测
    global quote_edge_detector
    if settings.quote_edge_enabled:
        quote_edge_detector = QuoteEdgeDetector(collector=collector)
        await quote_edge_detector.start()
        logger.info("报价 edge 影子检测器已启动（A 79.9%/EV+0.097，B 24%/EV+0.155，影子模式不下注）")

    # K 线科学发现影子信号（KREV 族）：720d 冻结注册表条件实时重放——
    # 15m bar 收盘后复用离线特征管道求值注册表条件原文，次根收盘按回测
    # 口径结算；只记录不下注，物理隔离于下单路径（新表不进 X4_VERSIONS/
    # LIVE_CHANNELS）。默认开关关闭，口径保真测试全绿后 .env 显式开启。
    global kline_shadow_detector
    if settings.kline_shadow_enabled:
        kline_shadow_detector = KlineShadowDetector(collector=collector, pm_15m_latest=_pm_15m_latest)
        await kline_shadow_detector.start()
        logger.info("KREV K线影子检测器已启动（720d v2 Top3/Top4 反转做多，holdout 64.2%/63.4%，影子模式不下注）")

    # HM 上吊线反弹入场影子信号（2026-09-01）：弱收盘上吊线 → 次 15m 周期内
    # 等反弹触及 +0.25×ATR（2s 轮询 mid 裁决）→ 记录押 DOWN 的虚拟入场。
    # 只记录不下注，物理隔离于下单路径（新表不进 X4_VERSIONS/LIVE_CHANNELS）。
    global hm_shadow_detector
    if settings.hm_shadow_enabled:
        hm_shadow_detector = HmShadowDetector(
            collector=collector,
            pm_15m_latest=_pm_15m_latest,
        )
        await hm_shadow_detector.start()
        logger.info("HM 影子检测器已启动（720d n=46 触价收跌 58.7% vs 隐含 47.1%，探索性发现，影子模式不下注）")

    # 反转形态影子信号（P1/P2，2026-09-03）：15m 连跌4+弱阴收+量正常→押 UP /
    # 连涨5+弱阳收贴最高→押 DOWN，rev_common 几何口径实时重放，次根收盘按 direction
    # 结算。只记录不下注，物理隔离于下单路径（与 KREV 共表 kline_shadow_signals，
    # version 隔离）。默认开关开启，与其他影子一致。
    global reversal_shadow_detector
    if settings.reversal_shadow_enabled:
        reversal_shadow_detector = ReversalShadowDetector(collector=collector, pm_15m_latest=_pm_15m_latest)
        await reversal_shadow_detector.start()
        logger.info("反转 K线影子检测器已启动（P1 连跌弱阴→UP 62.0% / P2 连涨弱阳→DOWN 62.4%，影子模式不下注）")

    # 下一根 K 线方向影子信号（nextbar 族，2026-09-03）：H=1 方向研究冻结条件实时重放——
    # 15m 冠军（zscore_10+zscore_5+ret_3 深超卖反转，holdout P(up_1)=61.96%）押次根 15m
    # 收阳 UP；5m sma_slope≥1.66 动量误定价候选押次根 5m 收阳 UP。build_feature_matrix
    # （k5=None）+ condition_mask 执行冻结条件原文，次根收盘按 direction 结算；只记录
    # 不下注，物理隔离于下单路径（与 KREV/反转共表 kline_shadow_signals，version+timeframe
    # 双隔离，各用自己 tf 的 K 线结算）。默认开关开启，与其他影子一致。
    global nextbar_shadow_detector
    if settings.nextbar_shadow_enabled:
        nextbar_shadow_detector = NextbarShadowDetector(
            collector=collector,
            pm_15m_latest=_pm_15m_latest,
            pm_5m_info=_pm_market_info,
        )
        await nextbar_shadow_detector.start()
        logger.info("nextbar K线影子检测器已启动（15m 冠军次根收阳 58.92% / 5m sma_slope 47.43% regime 依赖，影子模式不下注）")

    # 组合条件影子信号（combo 族，2026-09-04）：45 维条件大搜索（grand_search_v2）
    # + 1443 天样本外考试 + 50 次置换检验存活 5 组合实时重放——P1 连阳3+∧周末∧乖离
    # ≥+0.3%、P2 大实体(bp冻结)∧连阳3+∧周末、P3 贴1天高∧美盘∧7d涨≥4%（押次根
    # 15m 收阴 DOWN）；P4 收低位∧周末∧RSI14≤25、P5 近光脚∧周末∧RSI14≤25（押次根
    # 15m 收阳 UP）。自包含特征与研究口径逐字一致（硬闸门测试重放 720d 触发计数），
    # 轻探测轮询仅新收盘根拉 700 根；次根收盘按 direction 结算。只记录不下注，与
    # KREV/反转/nextbar 共表 kline_shadow_signals（version 隔离）。默认开关开启。
    global combo_shadow_detector
    if settings.combo_shadow_enabled:
        combo_shadow_detector = ComboShadowDetector(
            collector=collector,
            pm_15m_latest=_pm_15m_latest,
        )
        await combo_shadow_detector.start()
        logger.info("combo 组合影子检测器已启动（5 组合 3DOWN/2UP，720d 62.7~68.7% / oos 59.4~64.8%，影子模式不下注）")

    # 交易结算器（P0-2）：回读 SentimentWindow 结算 FILLED 订单输赢/盈亏。
    # 无开关常开：行为只读窗口 + 回填结算字段，零资金风险。
    global trade_settler
    trade_settler = TradeSettler()
    await trade_settler.start()
    logger.info("交易结算器已启动（60s 扫描 FILLED 未结算订单，settled_at 锚点）")

    # 多通道实盘执行器启动（构造/DB覆盖/邮件闸注入已在检测器 start 前完成）
    if multi_live_trader is not None:
        await multi_live_trader.start()
        # 余额缓存作废钩子：信号单成交后作废 prediction-wallet TTL 缓存
        # （下单扣预测钱包余额；与划转端点同款失效逻辑，前端下次轮询即取新值）
        multi_live_trader._on_balance_change = (
            lambda: _wallet_view_ts.__setitem__("balance", 0.0)
        )
        # 场景钩子注入（检测器构造晚于执行器装配，但钩子赋值无时序依赖；
        # 场景检测关闭时 detector 为 None，场景通道即使开启也无触发源——fail-safe）
        if fake_breakout_detector is not None:
            fake_breakout_detector._on_signal_fired = multi_live_trader.on_scene_signal
            fake_breakout_detector._on_s5_deep_fired = (
                multi_live_trader.on_s5_deep_signal)
        _live_status = multi_live_trader.status()
        _enabled = [c["channel"] for c in _live_status["channels"] if c["enabled"]]
        logger.info(
            "多通道实盘执行器已加载（真单！）| 通道 {}/{} 启用 {} | 默认 {} USDT/单",
            len(_enabled), len(_live_status["channels"]),
            _enabled or "（无，toggle 可开）",
            _live_status["defaults"]["amount_usdt"],
        )

    # 场景研究（M2）：LLM 研究员定期/累积/异常触发评估，假设只落库不生效
    # （M3 裁决 + 人工 promote 后才以 SHADOW 影子身份参与判定）
    global research_scheduler
    if settings.scene_research_enabled:
        from .services.research_scheduler import ResearchScheduler
        from .services.scene_researcher import SceneResearcher
        research_scheduler = ResearchScheduler(
            researcher=SceneResearcher(llm_service._decision_client),
        )
        await research_scheduler.start()

    # 模式池分级与定期重回测（无限进化引擎：新数据累积阈值触发，只发现不下注）
    if settings.pattern_reeval_enabled:
        await pattern_reevaluator.start()
        logger.info("模式重回测调度器已启动（模式池 S/A/B/C 分级）")

    yield  # 应用运行中

    # 4. 清理
    logger.info("系统关闭中...")
    # 停止模式重回测调度器
    await pattern_reevaluator.stop()
    # 停止场景研究调度器
    if research_scheduler is not None:
        await research_scheduler.stop()
    # 停止假突破检测器
    if fake_breakout_detector is not None:
        await fake_breakout_detector.stop()
    # 停止 X4 影子检测器
    if misalignment_detector is not None:
        await misalignment_detector.stop()
    # 停止报价 edge 影子检测器
    if quote_edge_detector is not None:
        await quote_edge_detector.stop()
    # 停止 K 线科学发现影子检测器
    if kline_shadow_detector is not None:
        await kline_shadow_detector.stop()
    # 停止 HM 上吊线反弹入场影子检测器
    if hm_shadow_detector is not None:
        await hm_shadow_detector.stop()
    # 停止反转形态影子检测器
    if reversal_shadow_detector is not None:
        await reversal_shadow_detector.stop()
    # 停止下一根 K 线方向影子检测器
    if nextbar_shadow_detector is not None:
        await nextbar_shadow_detector.stop()
    # 停止组合条件影子检测器
    if combo_shadow_detector is not None:
        await combo_shadow_detector.stop()
    # 停止影子版本开关 gate（所有影子检测器已停，最后收后台刷新任务）
    await shadow_gate.stop()
    # 停止交易结算器
    if trade_settler is not None:
        await trade_settler.stop()
    # 停止多通道实盘执行器（拒新单；在途下单任务等待完成，钱已出去的临界区不打断）
    if multi_live_trader is not None:
        await multi_live_trader.stop()
    # 复位邮件闸 resolver（防同进程二次 lifespan 残留指向旧 trader 的闭包）
    set_live_enabled_resolver(None)
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
    title="BTC 预测市场多通道实盘 + 影子信号系统",
    description="多通道实盘执行（MultiLiveTrader 15 通道）+ 影子信号检测 + 场景信号；情绪 Agent Loop 已退役",
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
# Web 登录全局认证中间件（单一访问密码，密码即 Bearer Token）
# ============================================================

class LoginAuthMiddleware(BaseHTTPMiddleware):
    """全局认证：所有 /api/* 请求须携带 Authorization: Bearer <LOGIN_PASSWORD>。

    放行：非 /api 路径、OPTIONS（CORS 预检）、/api/auth/login 本身、
    /api/health（部署流水线/容器探针需免认证探活）。
    未配置 login_password 时一律 401（不做"空值即放行"的开发旁路）。
    注册于 CORSMiddleware 之后（即最外层），预检先经本层直接放行。
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if (
            request.method == "OPTIONS"
            or not path.startswith("/api")
            or path in ("/api/auth/login", "/api/health")
        ):
            return await call_next(request)
        expected = settings.login_password
        auth_header = request.headers.get("Authorization", "")
        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
        if expected and token and hmac.compare_digest(token, expected):
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing authentication token"},
            headers={"WWW-Authenticate": "Bearer"},
        )


app.add_middleware(LoginAuthMiddleware)


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
# Web 登录 API（单一访问密码，登录态由前端存 localStorage 不过期）
# ============================================================

@app.post("/api/auth/login")
async def auth_login(body: LoginRequest):
    """校验访问密码，成功返回 token（即密码本身，前端存 localStorage 后作 Bearer 携带）。"""
    if not settings.login_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="服务端未配置登录密码（LOGIN_PASSWORD）",
        )
    if not hmac.compare_digest(body.password, settings.login_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="密码错误",
        )
    return {"ok": True, "token": settings.login_password}


# ============================================================
# API 路由（V2 PRD §17）
# ============================================================

# ============================================================
# 假突破信号系统 API
# ============================================================

@app.get("/api/fake-breakout/status")
async def get_fake_breakout_status():
    """假突破检测器状态：当前阻力位、日内信号数、最新报价快照。"""
    snapshot = (
        fake_breakout_detector.status_snapshot
        if fake_breakout_detector is not None
        else {"running": False}
    )
    return {
        **snapshot,
        "enabled": settings.fake_breakout_enabled,
        "btc_mid": collector.store.mid_price,
        "pm_15m": _pm_15m_latest,
        "pm_5m_down_price": _pm_market_info.get("down_price"),
    }


@app.get("/api/fake-breakout/signals")
async def list_fake_breakout_signals(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """假突破信号列表（倒序）。含到期回填的结算方向。"""
    from sqlalchemy import desc as sa_desc, select as sa_select

    limit = max(1, min(limit, 200))
    stmt = (
        sa_select(FakeBreakoutSignal)
        .order_by(sa_desc(FakeBreakoutSignal.signal_time))
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    signals = []
    for s in rows:
        signals.append({
            "id": s.id,
            "level": s.level,
            "side": s.side,
            "signal_time": s.signal_time,
            "resistance": s.resistance,
            "btc_price": s.btc_price,
            "down_price_5m": s.down_price_5m,
            "down_price_15m": s.down_price_15m,
            "up_price_5m": s.up_price_5m,
            "up_price_15m": s.up_price_15m,
            "market_end_15m": s.market_end_15m,
            "market_start_15m": s.market_start_15m,
            "cycle_open_price_15m": s.cycle_open_price_15m,
            "market_start_5m": s.market_start_5m,
            "market_end_5m": s.market_end_5m,
            "cycle_open_price_5m": s.cycle_open_price_5m,
            "cycle_offset_sec_15m": s.cycle_offset_sec_15m,
            "break_pct": s.break_pct,
            "pattern": s.pattern,
            "pattern_type": s.pattern_type,
            "close_pos": s.close_pos,
            "vol_ratio": s.vol_ratio,
            "ev_at_entry": s.ev_at_entry,
            "cumulative_winrate": s.cumulative_winrate,
            "cumulative_ev": s.cumulative_ev,
            "n_events_last_7d": s.n_events_last_7d,
            "entry_down_price_15m": s.entry_down_price_15m,
            "entry_up_price_15m": s.entry_up_price_15m,
            "entry_quote_ts_15m": s.entry_quote_ts_15m,
            "add_down_price_15m": s.add_down_price_15m,
            "add_up_price_15m": s.add_up_price_15m,
            "add_trigger_ts_15m": s.add_trigger_ts_15m,
            "quote5m_down_15m": s.quote5m_down_15m,
            "quote5m_up_15m": s.quote5m_up_15m,
            "quote5m_ts_15m": s.quote5m_ts_15m,
            "settle_btc_price": s.settle_btc_price,
            "settle_outcome": s.settle_outcome,
            "settle_btc_price_5m": s.settle_btc_price_5m,
            "settle_outcome_5m": s.settle_outcome_5m,
            "status": s.status,
            "email_sent": s.email_sent,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })
    return {"signals": signals, "total": len(signals)}


@app.get("/api/misalignment/signals")
async def list_misalignment_signals(
    limit: int = 50,
    version: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """影子信号列表 + 累计统计（promote 判据：WR/EV/实价覆盖率）。

    version 过滤：x4_v1 / quote_momentum_v1 / quote_contrarian_v1。
    列表缺省全部（向后兼容）；但累计统计缺省限定 x4_v1，保持 X4 promote
    判据语义——避免 quote_* 高频样本稀释 X4 的 WR/EV 数字（CodeReview Medium#3）。
    end_pct 字段：x4_v1 = 触发窗末 UP%，quote_* = 触发时刻 DOWN 报价。
    """
    from sqlalchemy import case as sa_case, desc as sa_desc, func as sa_func, select as sa_select
    from .db.models import MisalignmentSignal

    limit = max(1, min(limit, 200))
    stmt = sa_select(MisalignmentSignal)
    agg_stmt = sa_select(
        sa_func.count(MisalignmentSignal.id),
        sa_func.sum(sa_case((MisalignmentSignal.win.isnot(None), 1), else_=0)),
        sa_func.sum(sa_case((MisalignmentSignal.win.is_(True), 1), else_=0)),
        sa_func.sum(sa_case((MisalignmentSignal.ev_at_entry.isnot(None), 1), else_=0)),
        sa_func.avg(MisalignmentSignal.ev_at_entry),
        sa_func.sum(sa_case((MisalignmentSignal.entry_quote_kind == "real", 1), else_=0)),
    ).where(MisalignmentSignal.status == "SETTLED")
    # 列表缺省全部；stats 缺省限定 x4_v1（保持 X4 promote 判据语义）。
    stats_version = version or "x4_v1"
    if version:
        stmt = stmt.where(MisalignmentSignal.version == version)
    agg_stmt = agg_stmt.where(MisalignmentSignal.version == stats_version)
    stmt = stmt.order_by(sa_desc(MisalignmentSignal.window_start)).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    signals = [{
        "id": s.id,
        "version": s.version,
        "window_start": s.window_start,
        "window_end": s.window_end,
        "end_pct": s.end_pct,
        "outcome_base": s.outcome_base,
        "direction": s.direction,
        "target_window_start": s.target_window_start,
        "entry_down_price": s.entry_down_price,
        "entry_up_price": s.entry_up_price,
        "entry_quote_ts": s.entry_quote_ts,
        "entry_quote_kind": s.entry_quote_kind,
        "settle_outcome": s.settle_outcome,
        "win": s.win,
        "ev_at_entry": s.ev_at_entry,
        "status": s.status,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    } for s in rows]

    # 累计统计（全量，不随 limit 截断）：promote 判据四件套
    # 注意：case 必须从 sqlalchemy 顶层导入；func.case 是通用函数生成器，
    # 不接受 else_，会在语句构建时抛 TypeError（曾致生产 500）
    n, n_win_valid, n_wins, n_ev, avg_ev, n_real = (
        await db.execute(agg_stmt)
    ).one()
    n_ev = int(n_ev or 0)
    stats = {
        "settled": int(n or 0),
        "win_rate": (float(n_wins) / float(n_win_valid)) if n_win_valid else None,
        "avg_ev": float(avg_ev) if n_ev else None,
        "real_quote_coverage": (n_real / n_ev) if n_ev else None,
    }
    detector_status = misalignment_detector.status() if misalignment_detector else None
    quote_edge_status = quote_edge_detector.status() if quote_edge_detector else None
    live_status = (
        await multi_live_trader.status_async() if multi_live_trader else None)
    return {"signals": signals, "total": len(signals), "stats": stats,
            "detector": detector_status, "quote_edge_detector": quote_edge_status,
            "live_channels": live_status}


@app.get("/api/scene/versions")
async def list_scene_versions(
    limit: int = 50,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """场景参数版本列表（M3）：状态机 PENDING_REVIEW→SHADOW/REJECTED→ACTIVE→RETIRED。"""
    from sqlalchemy import desc as sa_desc, select as sa_select

    limit = max(1, min(limit, 200))
    stmt = (
        sa_select(SceneParamVersion)
        .order_by(sa_desc(SceneParamVersion.created_at))
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return {
        "versions": [{
            "id": r.id, "version": r.version, "params": r.params, "status": r.status,
            "proposed_by": r.proposed_by, "review_note": r.review_note,
            "has_backtest_report": r.backtest_report is not None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "activated_at": r.activated_at.isoformat() if r.activated_at else None,
        } for r in rows],
        "total": len(rows),
    }


@app.post("/api/scene/versions/{version_id}/adjudicate")
async def adjudicate_scene_version(
    version_id: int,
    _: None = Depends(_require_auth),
):
    """手动触发科学裁决（M3）：同窗 A/B 回测 + 四层硬门禁 → SHADOW/REJECTED。

    耗时约 2~4 分钟（拉 180 天官方 K）；正常链路由 research_scheduler 自动调，
    本端点供人工补跑。
    """
    from .services.hypothesis_arbiter import HypothesisArbiter

    verdict = await HypothesisArbiter().adjudicate(version_id)
    if verdict is None:
        return {"status": "error", "message": f"版本 #{version_id} 不存在或非 PENDING_REVIEW"}
    return {
        "status": "ok",
        "passed": verdict.passed,
        "reasons": verdict.reasons,
        "gates": verdict.gates,
        "report": verdict.report,
    }


@app.post("/api/scene/versions/{version_id}/promote")
async def promote_scene_version(
    version_id: int,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """人工放行（终审，M3/M4）：SHADOW → ACTIVE，原 ACTIVE 转 RETIRED。

    建议条件：影子实盘已结算 ≥ 30 条且不劣于现行 ACTIVE（research_scheduler 会
    邮件提示）；放行后 detector 在下一 15m 周期边界加载新参数。
    """
    from datetime import datetime, timezone as tz
    from sqlalchemy import select as sa_select

    row = await db.get(SceneParamVersion, version_id)
    if row is None:
        return {"status": "error", "message": "版本不存在"}
    if row.status != "SHADOW":
        return {"status": "error", "message": f"仅 SHADOW 可放行（当前 {row.status}）；先经裁决"}

    # 原 ACTIVE 退役
    stmt = (
        sa_select(SceneParamVersion)
        .where(SceneParamVersion.status == "ACTIVE")
    )
    for old in (await db.execute(stmt)).scalars().all():
        old.status = "RETIRED"
        old.retired_at = datetime.now(tz=tz.utc)

    row.status = "ACTIVE"
    row.activated_at = datetime.now(tz=tz.utc)
    row.reviewed_by = "manual-api"
    await db.commit()
    logger.info("场景参数版本放行 #{} {} → ACTIVE", version_id, row.version)
    return {"status": "ok", "version": row.version, "params": row.params}


@app.get("/api/fake-breakout/signals/{signal_id}/path")
async def get_fake_breakout_signal_path(
    signal_id: int,
    db: AsyncSession = Depends(get_db),
):
    """信号次周期路径：15s 采样还原的 BTC 价格 + 15m 市场 DOWN/UP 报价双轨。

    数据源 prediction_market_samples（market_period='15m'，2026-08-13 起积累），
    区间 = 信号周期锚点 [market_start_15m, market_end_15m)。用于 S5/S6/S7
    入场时点研究：早期窗口（t≤4）涨/跌态报价、+5m 确认点、真实结算对照。
    更早的信号无采样，has_data=false。
    """
    from sqlalchemy import asc as sa_asc, select as sa_select

    s = await db.get(FakeBreakoutSignal, signal_id)
    if s is None or not s.market_start_15m or not s.market_end_15m:
        return {"signal_id": signal_id, "has_data": False, "points": []}

    start, end = int(s.market_start_15m), int(s.market_end_15m)
    stmt = (
        sa_select(PredictionMarketSample)
        .where(
            PredictionMarketSample.market_period == "15m",
            PredictionMarketSample.timestamp >= start,
            PredictionMarketSample.timestamp < end,
        )
        .order_by(sa_asc(PredictionMarketSample.timestamp))
    )
    rows = (await db.execute(stmt)).scalars().all()
    points = [
        {
            "off": round((r.timestamp - start) / 1000.0, 1),
            "btc": r.btc_price,
            "down": r.down_price,
            "up": r.up_price,
        }
        for r in rows
        if r.btc_price is not None and r.down_price is not None
    ]
    return {
        "signal_id": signal_id,
        "cycle_start": start,
        "cycle_end": end,
        "open": s.cycle_open_price_15m,
        "side": s.side,
        "settle": s.settle_outcome,
        "quote5m_off": (
            round((s.quote5m_ts_15m - start) / 1000.0, 1)
            if s.quote5m_ts_15m and start <= s.quote5m_ts_15m < end
            else None
        ),
        "quote5m_down": s.quote5m_down_15m if s.side == "high" else s.quote5m_up_15m,
        "has_data": len(points) > 0,
        "points": points,
    }


@app.get("/api/fake-breakout/stats")
async def get_fake_breakout_stats(db: AsyncSession = Depends(get_db)):
    """假突破信号汇总统计：按 级别×方向 分组的胜率（5m/15m 双口径）+ 按场景类型的 EV 统计。

    周期锚点口径：settle_outcome = 信号所在市场周期的涨跌方向
    （周期末价 vs 周期开盘价，与币安预测市场真实结算规则一致）。
    胜负语义：side=high 买 DOWN（周期跌赢）；side=low 买 UP（周期涨赢）。
    """
    from sqlalchemy import func as sa_func, select as sa_select

    from .services.fake_breakout_detector import (
        BREAKEVEN, FEE, ODDS, PREMIUM, RESEARCH_WIN_RATES, compute_pattern_stats,
    )

    total = (await db.execute(
        sa_select(sa_func.count(FakeBreakoutSignal.id))
    )).scalar() or 0

    # 分组统计：(level, side, pattern) × {15m口径, 5m口径}
    rows = (await db.execute(
        sa_select(
            FakeBreakoutSignal.level,
            FakeBreakoutSignal.side,
            FakeBreakoutSignal.pattern,
            FakeBreakoutSignal.settle_outcome,
            FakeBreakoutSignal.settle_outcome_5m,
        ).where(FakeBreakoutSignal.status == "SETTLED")
    )).all()

    groups: dict[str, dict] = {}
    for level, side, pattern, oc15, oc5 in rows:
        key = f"{level}|{side}|{pattern or 'legacy'}"
        g = groups.setdefault(key, {
            "level": level, "side": side, "pattern": pattern,
            "settled_15m": 0, "wins_15m": 0,
            "settled_5m": 0, "wins_5m": 0,
        })
        win_dir = "DOWN" if side == "high" else "UP"
        if oc15 is not None:
            g["settled_15m"] += 1
            if oc15 == win_dir:
                g["wins_15m"] += 1
        if oc5 is not None:
            g["settled_5m"] += 1
            if oc5 == win_dir:
                g["wins_5m"] += 1

    by_group = []
    for g in groups.values():
        by_group.append({
            **g,
            "win_rate_15m": (g["wins_15m"] / g["settled_15m"]) if g["settled_15m"] else None,
            "win_rate_5m": (g["wins_5m"] / g["settled_5m"]) if g["settled_5m"] else None,
        })
    by_group.sort(key=lambda x: (x["level"], x["side"], x["pattern"] or ""))

    # 按场景类型（pattern_type）的实盘统计：胜率 / 累计EV / 入场EV / 收益曲线 / 回撤
    # （仅正式信号：排除 SHADOW 版本名，ACTIVE 版本演进兼容；
    #   口径与 detector._update_pattern_stats 一致）
    pt_rows = (await db.execute(
        sa_select(FakeBreakoutSignal)
        .where(FakeBreakoutSignal.status == "SETTLED")
        .where(await _official_scene_version_filter(db))
        .order_by(FakeBreakoutSignal.signal_time)
    )).scalars().all()
    by_pt: dict[str, list] = {}
    for row in pt_rows:
        by_pt.setdefault(row.pattern_type or row.pattern or "legacy", []).append(row)
    now_ms = int(time.time() * 1000)
    by_pattern_type = []
    for pt, rows_pt in sorted(by_pt.items()):
        stats = compute_pattern_stats(rows_pt)
        stats["pattern_type"] = pt
        stats["n_last_7d"] = sum(
            1 for r in rows_pt if r.signal_time >= now_ms - 7 * 86_400_000
        )
        by_pattern_type.append(stats)

    # 全量汇总（兼容前端旧字段）
    settled = sum(g["settled_15m"] for g in groups.values())
    wins = sum(g["wins_15m"] for g in groups.values())
    settled_5m = sum(g["settled_5m"] for g in groups.values())
    wins_5m = sum(g["wins_5m"] for g in groups.values())
    avg_15m = (await db.execute(
        sa_select(sa_func.avg(FakeBreakoutSignal.down_price_15m)).where(
            FakeBreakoutSignal.down_price_15m.isnot(None)
        )
    )).scalar()
    avg_5m = (await db.execute(
        sa_select(sa_func.avg(FakeBreakoutSignal.down_price_5m)).where(
            FakeBreakoutSignal.down_price_5m.isnot(None)
        )
    )).scalar()
    return {
        "total_signals": total,
        "settled": settled,
        "down_win_rate": (wins / settled) if settled else None,
        "avg_down_price_15m": float(avg_15m) if avg_15m is not None else None,
        "avg_down_price_5m": float(avg_5m) if avg_5m is not None else None,
        "settled_5m": settled_5m,
        "down_win_rate_5m": (wins_5m / settled_5m) if settled_5m else None,
        "by_group": by_group,
        "by_pattern_type": by_pattern_type,
        "research_win_rates": dict(RESEARCH_WIN_RATES),
        "pricing": {"fee": FEE, "premium": PREMIUM, "odds": round(ODDS, 4), "breakeven": round(BREAKEVEN, 4)},
    }


# ============================================================
# 模式池分级与回测快照 API（无限进化引擎，只发现不下注）
# ============================================================

@app.post("/api/agent/patterns/reevaluate")
async def trigger_pattern_reevaluate(_: None = Depends(_require_auth)):
    """手动触发一轮全量模式重回测（与后台调度同一入口）。"""
    summary = await pattern_reevaluator.run_all(trigger="MANUAL")
    return {"ok": True, "summary": summary}


@app.get("/api/agent/patterns/backtest-runs")
async def list_pattern_backtest_runs(
    pattern_id: int | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """模式回测快照列表（纵向对比数据源：同一模式随时间的回测漂移）。"""
    from sqlalchemy import desc as sa_desc, select as sa_select

    limit = max(1, min(limit, 200))
    stmt = sa_select(PatternBacktestRun).order_by(sa_desc(PatternBacktestRun.created_at))
    if pattern_id is not None:
        stmt = stmt.where(PatternBacktestRun.pattern_id == pattern_id)
    stmt = stmt.limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return {
        "runs": [
            {
                "id": r.id,
                "pattern_id": r.pattern_id,
                "data_start": r.data_start,
                "data_end": r.data_end,
                "sample_count": r.sample_count,
                "correct_count": r.correct_count,
                "win_rate": r.win_rate,
                "wilson_lower": r.wilson_lower,
                "wilson_upper": r.wilson_upper,
                "ev_after_fee": r.ev_after_fee,
                "segment_stats": r.segment_stats,
                "delta_vs_prev": r.delta_vs_prev,
                "trigger_reason": r.trigger_reason,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@app.get("/api/agent/patterns/compare")
async def compare_patterns_latest(db: AsyncSession = Depends(get_db)):
    """横向对比：每个模式最新一次回测快照 + 当前 tier（模式间对比）。"""
    from sqlalchemy import desc as sa_desc, select as sa_select
    from .db.models import PatternMemory

    patterns = (await db.execute(
        sa_select(PatternMemory).order_by(sa_desc(PatternMemory.updated_at))
    )).scalars().all()

    # 每个模式的最新一次回测
    latest_runs: dict[int, PatternBacktestRun] = {}
    run_rows = (await db.execute(
        sa_select(PatternBacktestRun).order_by(sa_desc(PatternBacktestRun.created_at))
    )).scalars().all()
    for r in run_rows:
        if r.pattern_id not in latest_runs:
            latest_runs[r.pattern_id] = r

    items = []
    for p in patterns:
        run = latest_runs.get(p.id)
        items.append({
            "pattern_id": p.id,
            "pattern_name": p.pattern_name,
            "status": p.status,
            "tier": p.tier,
            "predicted_direction": p.predicted_direction,
            "discovery_method": p.discovery_method,
            "live_win_rate": p.win_rate,
            "live_sample_count": p.sample_count,
            "latest_run": (
                {
                    "id": run.id,
                    "data_end": run.data_end,
                    "sample_count": run.sample_count,
                    "win_rate": run.win_rate,
                    "wilson_lower": run.wilson_lower,
                    "wilson_upper": run.wilson_upper,
                    "ev_after_fee": run.ev_after_fee,
                    "delta_vs_prev": run.delta_vs_prev,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                }
                if run else None
            ),
        })
    return {"patterns": items, "total": len(items)}


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


@app.get("/api/trades/recent")
async def get_recent_trades(
    limit: int = 20,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """最近交易订单列表（实盘面板：下单历史展示，含人工测试单与信号实盘单）。"""
    from sqlalchemy import select
    from .db.models import TradeOrderModel

    limit = max(1, min(int(limit), 100))
    stmt = (
        select(TradeOrderModel)
        .order_by(TradeOrderModel.created_at.desc())
        .limit(limit)
    )
    orders = (await db.execute(stmt)).scalars().all()
    return {
        "orders": [
            {
                "id": o.id,
                "signal_version": o.signal_version,
                "window_start": o.window_start,
                "market_period": o.market_period,
                "status": o.status,
                "order_id": o.order_id,
                "token_id": o.token_id,
                "amount_in": o.amount_in,
                "average_price": (o.quote_json or {}).get("averagePrice"),
                "direction": o.direction,
                "settle_outcome": o.settle_outcome,
                "win": o.win,
                "pnl": o.pnl,
                "settled_at": o.settled_at.isoformat() if o.settled_at else None,
                "redeemed_at": o.redeemed_at.isoformat() if o.redeemed_at else None,
                "error_message": o.error_message,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in orders
        ]
    }


@app.get("/api/trades/fund-flow")
async def get_trades_fund_flow(
    since_ms: int = 0,
    limit: int = 5000,
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """全量资金流水（资金变化面板）：不受 /api/trades/recent 的 100 条硬上限约束。

    只返回资金派生所需精简字段（无 order_id/token_id/quote_json），
    按 created_at 升序。since_ms 可选：只拉该毫秒时间戳之后的订单。
    """
    from sqlalchemy import select
    from .db.models import TradeOrderModel

    limit = max(1, min(int(limit), 5000))
    stmt = select(TradeOrderModel).order_by(TradeOrderModel.created_at.asc()).limit(limit)
    if since_ms > 0:
        since_dt = datetime.fromtimestamp(int(since_ms) / 1000, tz=timezone.utc)
        stmt = stmt.where(TradeOrderModel.created_at >= since_dt)
    orders = (await db.execute(stmt)).scalars().all()
    return {
        "total": len(orders),
        "orders": [
            {
                "signal_version": o.signal_version,
                "status": o.status,
                "amount_in": o.amount_in,
                "direction": o.direction,
                "win": o.win,
                "pnl": o.pnl,
                "settled_at": o.settled_at.isoformat() if o.settled_at else None,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in orders
        ],
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


@app.get("/api/prediction-markets/all")
async def list_prediction_markets_all(
    _: None = Depends(_require_auth),
):
    """全量 BTC 预测市场（含分页拉取后的所有周期，诊断用）。

    返回精简字段：title/slug/周期分类/起止时间/参与者/交易量，
    用于确认 15m 等新周期市场是否在列表内。
    """
    markets = await prediction_trader.list_markets()
    return {
        "count": len(markets),
        "periods": {
            "5m": bool(prediction_trader._up_token_id),
            "15m_down_price": prediction_trader._15m_down_price,
            "15m_end_date": prediction_trader._15m_end_date,
        },
        "markets": [
            {
                "title": m.get("title"),
                "slug": m.get("slug"),
                "period": BinancePredictionTrader._classify_period(m),
                "startDate": m.get("startDate"),
                "endDate": m.get("endDate"),
                "participantCount": m.get("participantCount"),
                "tradeVolume": m.get("tradeVolume"),
            }
            for m in markets
        ],
    }


@app.get("/api/prediction-wallet")
async def get_prediction_wallet(
    _: None = Depends(_require_auth),
):
    """获取预测钱包信息（地址/ID/注册时间 + 现货与预测钱包余额）。

    服务端 TTL 缓存：静态信息 300s / 余额 20s，单飞锁防 stampede。
    预测钱包余额为探索型查询（wallet 内嵌字段 + asset/list 候选端点），
    失败降级 None 不阻塞其余字段（沿用 spot_usdt_free 的 None 降级模式）。
    """
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}

    def _mask_addr(addr: str | None) -> str | None:
        """地址脱敏：仅展示前6后4位"""
        if not addr:
            return None
        return f"{addr[:6]}...{addr[-4:]}" if len(addr) > 12 else "***"

    now = time.time()
    async with _wallet_view_lock:
        # 拿到锁后重算新鲜度（前一个等待者可能刚刷新完）
        static_fresh = (now - _wallet_view_ts["static"]) < _WALLET_STATIC_TTL
        balance_fresh = (now - _wallet_view_ts["balance"]) < _WALLET_BALANCE_TTL
        wallet_snapshot: dict | None = None
        if not static_fresh:
            wallet = await prediction_trader.fetch_wallet_info()
            if wallet:
                wallet_snapshot = wallet
                _wallet_view.update({
                    "wallet_address": _mask_addr(wallet.get("walletAddress")),
                    "wallet_id": wallet.get("walletId"),
                    "registered_time": wallet.get("registeredTime"),
                })
                _wallet_view_ts["static"] = now
            else:
                # 静态失败退避：60s 内不重试（防止每次请求都打签名 API）
                _wallet_view_ts["static"] = now - _WALLET_STATIC_TTL + 60
        if not balance_fresh:
            spot = await prediction_trader.fetch_spot_usdt_balance()
            pred = await prediction_trader.fetch_prediction_wallet_balance(
                wallet=wallet_snapshot)
            _wallet_view.update({
                "spot_usdt_free": spot,
                "prediction_usdt_free": pred.get("usdt_free"),
                "wallet_assets": pred.get("assets"),
                "prediction_balance_available": (
                    pred.get("usdt_free") is not None or pred.get("assets") is not None
                ),
                # 诊断透传：余额不可查时直接看币安返回的真实错误/响应原文（无 SSH 也能定位）
                "assets_api_error": getattr(prediction_trader, "last_assets_error", None),
                "assets_raw_preview": getattr(prediction_trader, "last_assets_raw", None),
            })
            _wallet_view_ts["balance"] = now
        if "wallet_address" not in _wallet_view:
            return {"error": "未找到预测钱包，请先在 Binance App 中开通预测市场"}
        return dict(_wallet_view)


@app.get("/api/prediction/quote-preview")
async def get_quote_preview(_: None = Depends(_require_auth)):
    """当前 5m 预测市场报价预览（前端倒计时/指示价用）。

    纯内存读：数据源为 15s 采样器 _prediction_market_tracker 直写的模块级
    _pm_market_info，零币安 API 成本。报价最多 15s 陈旧（指示价）；
    实际成交价以下单响应 average_price 为准。
    """
    async with _state_lock:
        snap = dict(_pm_market_info)
    if not snap:
        return {
            "window_start": None,
            "window_end": None,
            "up_price": None,
            "down_price": None,
            "server_now_ms": int(time.time() * 1000),
            "stale": True,
        }
    start, end = snap.get("start_date"), snap.get("end_date")
    return {
        "window_start": int(start) if start is not None else None,
        "window_end": int(end) if end is not None else None,
        "up_price": snap.get("up_price"),
        "down_price": snap.get("down_price"),
        "server_now_ms": int(time.time() * 1000),
        "stale": False,
    }


@app.post("/api/trade/test")
async def manual_trade_test(
    req: ManualTradeTestRequest,
    _: None = Depends(_require_auth),
):
    """实盘链路人工测试单：钱包→市场→报价→下单→落库全链路验证。

    与信号实盘共用 execute_signal_trade（先占位后下单），signal_version="manual_test"：
    同一 5m 窗口至多一单（唯一键防重），订单落 trade_orders 表可追溯。
    金额硬限 0.1~50 USDT（与实盘单笔硬上限 MAX_ORDER_AMOUNT_USDT 对齐）；
    不设执行价护栏。"""
    if not (0.1 <= req.amount_usdt <= 50.0):
        return {"error": "amount_usdt 仅允许 0.1~50（与实盘单笔硬上限一致）"}
    if req.prediction not in ("UP", "DOWN"):
        return {"error": "prediction 仅允许 UP/DOWN"}

    window_start = int(time.time() * 1000) // 300_000 * 300_000  # 当前 5m 窗口起点
    order = await prediction_trader.execute_signal_trade(
        prediction=req.prediction,
        amount_usdt=req.amount_usdt,
        signal_version="manual_test",
        window_start=window_start,
    )
    if order is None:
        return {
            "error": "下单未执行（API Key 未配置 / 钱包获取失败 / 本窗口已有测试单）",
            "window_start": window_start,
        }
    # 产生订单即作废余额缓存（下单扣预测钱包余额；与划转端点同款失效逻辑）
    _wallet_view_ts["balance"] = 0.0
    # order 为 dict 快照（execute_signal_trade 不再返回 ORM 对象，
    # 避免会话关闭后访问属性报 DetachedInstanceError）
    return {
        "status": order.get("status"),
        "order_id": order.get("order_id"),
        "direction": order.get("direction"),
        "signal_version": order.get("signal_version"),
        "window_start": order.get("window_start"),
        "token_id": order.get("token_id") or None,
        "average_price": order.get("average_price"),
        "amount_in": order.get("amount_in"),
        "error_message": order.get("error_message"),
    }


@app.post("/api/live/toggle")
async def live_toggle(
    req: ToggleLiveRequest,
    _: None = Depends(_require_auth),
):
    """实时开关/热调单个实盘通道（MultiLiveTrader，多通道时代重写）。

    通道白名单 LIVE_CHANNELS；可选 amount_usdt/max_daily_orders/max_exec_price
    同步热调（校验同启动配置，非法值拒改不落库），立即生效并持久化到
    live_channel_overrides（重启不丢设定；DB 写失败仅告警不影响运行时生效）。
    护栏传 reset_max_exec=True 清空自定义回落通道预设值。
    ⚠️ 关闭不取消在途任务（High#1），只阻止该通道新单派生。
    """
    if multi_live_trader is None:
        return {"error": "执行器未装配（装配阶段异常？详见后端日志）"}
    try:
        multi_live_trader.set_channel(
            req.channel, enabled=req.enabled,
            amount_usdt=req.amount_usdt,
            max_daily_orders=req.max_daily_orders,
            max_exec_price=req.max_exec_price,
            reset_max_exec=req.reset_max_exec)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        await multi_live_trader.persist_channel(req.channel)
    except Exception as exc:
        # 运行时已生效，仅持久化失败：告警不报错，下次 toggle 自愈
        logger.warning("实盘通道配置持久化失败（重启将回落 env 配置）| {} | {}",
                       req.channel, exc)
    _status = await multi_live_trader.status_async()
    _ch = next((c for c in _status["channels"] if c["channel"] == req.channel), None)
    if _ch is None:
        return {"error": f"通道 {req.channel} 不在状态列表（内部错误）"}
    return {
        "message": (
            f"通道 {_ch['display_name']} 已{'开启' if _ch['enabled'] else '关闭'}"
            f"（金额 {_ch['amount_usdt']} USDT/单 | 日限 {_ch['max_daily_orders']} |"
            f" 护栏 {_ch['max_exec_price']}）"
        ),
        "status": _status,
    }


@app.post("/api/shadow/toggle")
async def shadow_toggle(
    req: ToggleShadowRequest,
    _: None = Depends(_require_auth),
):
    """手动下线/上线影子信号版本（2026-09-04：前端影子管理能力）。

    下线=各检测器停止采集该版本新信号（落库前 gate 拦截）+面板置灰；
    历史已落库信号不受影响（下线≠删数据，曲线照常显示已有样本）。
    白名单=SHADOW_BENCH；持久化 shadow_version_overrides（重启不丢），
    gate 缓存本进程立即生效，多 worker 靠 60s TTL 收敛。
    """
    if req.version not in SHADOW_BENCH:
        raise HTTPException(status_code=422, detail=f"未知影子版本: {req.version}")
    await shadow_gate.set_enabled(req.version, req.enabled)
    return {
        "message": (
            f"影子版本 {req.version} 已{'上线' if req.enabled else '下线'}"
            f"（{'恢复采集新信号' if req.enabled else '停止采集新信号，历史数据保留'}）"
        ),
        "version": req.version,
        "enabled": req.enabled,
    }


@app.get("/api/live/pnl-curve")
async def live_pnl_curve(
    _: None = Depends(_require_auth),
    db: AsyncSession = Depends(get_db),
):
    """每实盘通道已结算订单的累计盈亏趋势（右侧抽屉「盈利趋势」tab 数据源）。

    口径：trade_orders 中 signal_version 属于通道注册表（LIVE_CHANNELS）、
    FILLED 且已结算（win/pnl 非空）的行，按 window_start 时序逐单累计
    本地估算 pnl（USDT）。未结算/FAILED/manual_test 单不计入。
    返回全部有已结算数据的通道（含当前已关闭的），enabled 字段供前端过滤。
    """
    from sqlalchemy import select as sa_select
    from .db.models import TradeOrderModel
    from .services.live_channels import LIVE_CHANNELS

    rows = (await db.execute(
        sa_select(
            TradeOrderModel.signal_version,
            TradeOrderModel.window_start,
            TradeOrderModel.settled_at,
            TradeOrderModel.win,
            TradeOrderModel.pnl,
        ).where(
            TradeOrderModel.signal_version.in_(list(LIVE_CHANNELS)),
            TradeOrderModel.status == "FILLED",
            TradeOrderModel.win.isnot(None),
            TradeOrderModel.pnl.isnot(None),
        ).order_by(TradeOrderModel.window_start.asc())
    )).all()

    # 通道元数据（中文名/开关态）优先取运行时状态，执行器未装配时回落注册表
    meta: dict[str, dict] = {}
    if multi_live_trader is not None:
        try:
            meta = {c["channel"]: c for c in
                    (await multi_live_trader.status_async())["channels"]}
        except Exception as exc:
            logger.warning("pnl-curve 通道状态查询失败，回落注册表元数据 | {}", exc)
    for ch, spec in LIVE_CHANNELS.items():
        meta.setdefault(ch, {
            "channel": ch, "display_name": spec.display_name,
            "family": spec.family, "market_period": spec.market_period,
            "enabled": False,
        })

    # 按通道分组累计（window_start 已升序；settle 时序与窗口时序一致）
    by_ch: dict[str, list] = {}
    for ver, ws, settled_at, win, pnl in rows:
        by_ch.setdefault(ver, []).append(
            {"window_start": int(ws), "win": bool(win), "pnl": float(pnl),
             "t": int(settled_at.timestamp() * 1000) if settled_at else int(ws)})

    channels_out = []
    for ch, pts in by_ch.items():
        cum = 0.0
        points = []
        for n, p in enumerate(pts, start=1):
            cum += p["pnl"]
            points.append({"n": n, "t": p["t"], "pnl": round(p["pnl"], 4),
                           "cum": round(cum, 4), "win": p["win"]})
        wins = sum(1 for p in pts if p["win"])
        m = meta.get(ch, {})
        channels_out.append({
            "channel": ch,
            "display_name": m.get("display_name", ch),
            "family": m.get("family"),
            "market_period": m.get("market_period"),
            "enabled": bool(m.get("enabled")),
            "settled_count": len(pts),
            "win_count": wins,
            "win_rate": round(wins / len(pts), 4) if pts else None,
            "total_pnl": round(cum, 4),
            "points": points,
        })
    channels_out.sort(key=lambda c: c["total_pnl"], reverse=True)

    total_pnl = round(sum(c["total_pnl"] for c in channels_out), 4)
    total_cnt = sum(c["settled_count"] for c in channels_out)
    total_win = sum(c["win_count"] for c in channels_out)
    return {
        "channels": channels_out,
        "total": {
            "settled_count": total_cnt,
            "total_pnl": total_pnl,
            "win_rate": round(total_win / total_cnt, 4) if total_cnt else None,
        },
    }


@app.post("/api/prediction/transfer-in")
async def prediction_transfer_in(
    req: TransferInboundRequest,
    _: None = Depends(_require_auth),
):
    """现货账户 → 预测钱包划转入金。

    预测市场下单扣的是预测钱包内余额：现货余额充足但下单报 -9000
    （ensure enough USDT）时，需先把 USDT 划转入预测钱包。
    金额硬限 0.1~20 USDT（小额运维通道）。"""
    if not (0.1 <= req.amount_usdt <= 20.0):
        return {"error": "amount_usdt 仅允许 0.1~20"}
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}

    # 确保钱包信息已加载（自动获取）
    if not prediction_trader._wallet_address or not prediction_trader._wallet_id:
        wallet = await prediction_trader.fetch_wallet_info()
        if not wallet:
            return {"error": "未找到预测钱包，请先在 Binance App 中开通"}

    resp = await prediction_trader.transfer_in(req.amount_usdt)
    if resp is None:
        return {
            "status": "FAILED",
            "error": prediction_trader.last_api_error or "划转失败（无详情）",
        }
    # 划转成功立即作废余额缓存（Commit 2 的 TTL 缓存），下次轮询取新值
    _wallet_view_ts["balance"] = 0.0
    # 划转后刷新现货余额供前端即时确认
    return {
        "status": "SUCCESS",
        "transfer": resp,
        "spot_usdt_free": await prediction_trader.fetch_spot_usdt_balance(),
    }


@app.post("/api/prediction/transfer-out")
async def prediction_transfer_out(
    req: TransferOutboundRequest,
    _: None = Depends(_require_auth),
):
    """预测钱包 → 现货账户划出提走（P1-1）。

    ⚠️ 官方端点命名反转：inbound = 预测钱包→CEX 现货（提走），
    outbound = CEX→预测钱包（入金）。方向自证：划转前后各查一次现货
    余额，余额增加才证明资金确实回到现货（响应回显 spot_before/
    spot_after/direction_confirmed，防真金反向移动）。
    金额硬限 0.1~20 USDT（小额运维通道；上线后先 0.1 金丝雀验证）。"""
    if not (0.1 <= req.amount_usdt <= 20.0):
        return {"error": "amount_usdt 仅允许 0.1~20"}
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}

    # 确保钱包信息已加载（自动获取）
    if not prediction_trader._wallet_address or not prediction_trader._wallet_id:
        wallet = await prediction_trader.fetch_wallet_info()
        if not wallet:
            return {"error": "未找到预测钱包，请先在 Binance App 中开通"}

    spot_before = await prediction_trader.fetch_spot_usdt_balance()
    resp = await prediction_trader.transfer_out(req.amount_usdt)
    if resp is None:
        return {
            "status": "FAILED",
            "error": prediction_trader.last_api_error or "划转失败（无详情）",
        }
    # 划出成功立即作废余额缓存（预测钱包余额已减少）
    _wallet_view_ts["balance"] = 0.0
    # 方向自证：划转后刷新对比（余额增加才证明方向正确）
    spot_after = await prediction_trader.fetch_spot_usdt_balance()
    confirmed = (
        spot_before is not None and spot_after is not None
        and spot_after > spot_before
    )
    out = {
        "status": "SUCCESS",
        "transfer": resp,
        "spot_before": spot_before,
        "spot_after": spot_after,
        "direction_confirmed": confirmed,
    }
    if not confirmed:
        out["warning"] = (
            "划转已提交但现货余额未见增加——请立即人工核对币安 App 划转记录"
            "（防方向反转，真金反向移动风险）"
        )
    return out


@app.get("/api/prediction/redeemable")
async def prediction_redeemable(_: None = Depends(_require_auth)):
    """可领取奖金查询：钱包 PENDING_CLAIM（链上事实）+ DB win 未领取（对账）。

    两源合并去重：官方 position/list?tab=PENDING_CLAIM 是链上事实（权威），
    本地 DB win=true 且 redeemed_at IS NULL 兑底（官方端点探索期降级时仍有参考）。
    诊断透传：api_error / raw_preview（无 SSH 收敛用）。
    """
    positions = await prediction_trader.fetch_pending_claim_positions()
    wallet_tokens = prediction_trader._extract_token_ids(positions or [])

    from sqlalchemy import select
    from .db.models import TradeOrderModel

    async with async_session_factory() as db:
        rows = (await db.execute(
            select(TradeOrderModel.id, TradeOrderModel.token_id)
            .where(TradeOrderModel.win.is_(True))
            .where(TradeOrderModel.redeemed_at.is_(None))
            .where(TradeOrderModel.token_id.isnot(None))
            .where(TradeOrderModel.token_id != "")
        )).all()
    db_tokens = [r.token_id for r in rows]

    if positions is None:
        # 钱包查询降级：DB win 未领取兑底（官方端点探索期降级时仍有参考）
        merged = list(dict.fromkeys(wallet_tokens + db_tokens))  # 去重保序
    else:
        # 钱包查询成功：链上事实为权威。DB 中链上已赎回/不可赎的 token
        # 不得计入可领（否则送去 batch-redeem 会触发币安 400 -9000 SYSTEM_ERROR）
        merged = wallet_tokens
    return {
        "claimable_count": len(merged),
        "claimable_tokens": merged,
        "wallet_source": "ok" if positions is not None else "degraded",
        "db_win_unclaimed_ids": [r.id for r in rows],
        "positions_preview": (positions or [])[:10],
        # 诊断透传（探索型端点收敛用）
        "api_error": prediction_trader.last_pending_error,
        "raw_preview": prediction_trader.last_pending_raw,
    }


@app.post("/api/prediction/redeem")
async def prediction_redeem(
    req: RedeemRequest,
    _: None = Depends(_require_auth),
):
    """领取获胜 token 奖金（POST batch-redeem：链上 token → USDT）。

    token_ids 缺省 = 自动收集（钱包 PENDING_CLAIM 优先，DB win 未领取兑底）；
    成功后按 token_id 标记 DB redeemed_at + 作废余额缓存（领取入 CeDeFi）。
    """
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}
    if not prediction_trader._wallet_address or not prediction_trader._wallet_id:
        wallet = await prediction_trader.fetch_wallet_info()
        if not wallet:
            return {"error": "未找到预测钱包，请先在 Binance App 中开通"}

    token_ids = req.token_ids
    positions = None
    if not token_ids:
        positions = await prediction_trader.fetch_pending_claim_positions()
        token_ids = prediction_trader._extract_token_ids(positions or [])
    if not token_ids and positions is None:
        # DB 兜底仅限钱包查询降级时：链上返回成功且 PENDING_CLAIM 为空即代表无可领，
        # 把 DB 未标记的 token 送去 batch-redeem 会触发币安 400 -9000 SYSTEM_ERROR
        from sqlalchemy import select
        from .db.models import TradeOrderModel

        async with async_session_factory() as db:
            rows = (await db.execute(
                select(TradeOrderModel.token_id)
                .where(TradeOrderModel.win.is_(True))
                .where(TradeOrderModel.redeemed_at.is_(None))
                .where(TradeOrderModel.token_id.isnot(None))
                .where(TradeOrderModel.token_id != "")
            )).all()
        token_ids = [r.token_id for r in rows]
    if not token_ids:
        return {"status": "NOOP", "message": "无可领取持仓（没有赢单 token 待领取）"}

    resp = await prediction_trader.redeem_tokens(token_ids)
    if resp is None:
        return {"status": "FAILED", "error": prediction_trader.last_api_error}

    # 成功：按 token_id 标记 DB 已领取（win 单限定，幂等：redeemed_at IS NULL 守卫）
    from sqlalchemy import update
    from .db.models import TradeOrderModel

    async with async_session_factory() as db:
        await db.execute(
            update(TradeOrderModel)
            .where(TradeOrderModel.token_id.in_(token_ids))
            .where(TradeOrderModel.win.is_(True))
            .where(TradeOrderModel.redeemed_at.is_(None))
            .values(redeemed_at=datetime.now(timezone.utc))
        )
        await db.commit()
    # 领取入 CeDeFi：作废余额缓存，前端下次轮询即取新值
    _wallet_view_ts["balance"] = 0.0
    return {
        "status": "SUCCESS",
        "redeemed": len(token_ids),
        "tokens": token_ids,
        "resp_preview": str(resp)[:300],
    }


@app.get("/api/trades/binance-history")
async def get_binance_order_history(
    limit: int = 20,
    _: None = Depends(_require_auth),
):
    """币安侧预测钱包订单历史（对账用：本地卡 PENDING 时确认是否真实成交）。"""
    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}
    if not prediction_trader._wallet_address or not prediction_trader._wallet_id:
        wallet = await prediction_trader.fetch_wallet_info()
        if not wallet:
            return {"error": "未找到预测钱包"}
    limit = max(1, min(int(limit), 100))
    orders = await prediction_trader.query_order_history(limit=limit)
    if orders is None:
        # 回显币安侧具体错误，便于定位参数/权限问题
        return {"error": prediction_trader.last_api_error or "查询失败（详见后端日志）", "orders": []}
    return {"orders": orders}


async def _sync_binance_orders_impl() -> dict:
    """对账回填实现（/api/trades/sync-binance 端点主体，2026-08-28 抽函数不改行为）。

    手动端点与定时对账循环（_binance_reconcile_loop）共用。
    场景一：端点曾返回 500 但币安侧实际已成交（钱已花出），本地行停留
    在 PENDING。按 window_start 匹配（slug btc-updown-5m-<秒> 编码窗口
    起始秒）；币安侧无对应订单的行保持 PENDING 不动。
    场景二（幽灵成交订正）：本地 FILLED 但币安侧同 orderId 为 FAILED
    （FOK 受理后翻转未成交）——改判 FAILED 并清除 win/pnl 等结算字段，
    避免不存在的盈亏污染统计、win 单误入可领取列表。
    场景三（部分成交订正，2026-08-28）：FOK 只吃到部分深度时实际成交金额 <
    请求金额（生产实锤：请求 3U 只成交 2U，本地按 3U 估算赢向 pnl 虚高 55%）——
    以币安历史行 filledUsdtAmount 为权威订正 amount_in，并重算已结算行 pnl。
    """
    import re
    from decimal import Decimal
    from sqlalchemy import select
    from .db.models import TradeOrderModel

    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}
    if not prediction_trader._wallet_address:
        wallet = await prediction_trader.fetch_wallet_info()
        if not wallet:
            return {"error": "未找到预测钱包"}
    history = await prediction_trader.query_order_history(limit=100)
    if history is None:
        return {"error": prediction_trader.last_api_error or "查询失败", "synced": 0}

    # slug 末尾即窗口起始秒（如 btc-updown-5m-1787418600）
    by_window: dict[int, dict] = {}
    by_orderid: dict[str, dict] = {}
    for o in history:
        if not isinstance(o, dict):
            continue
        if o.get("orderId"):
            by_orderid[str(o["orderId"])] = o
        m = re.search(r"-(\d{10})$", o.get("slug") or "")
        if m:
            by_window[int(m.group(1)) * 1000] = o

    synced: list[dict] = []
    amount_corrected: list[dict] = []
    async with async_session_factory() as db:
        stmt = select(TradeOrderModel).where(
            TradeOrderModel.status == "PENDING",
            TradeOrderModel.window_start.isnot(None),
        )
        rows = (await db.execute(stmt)).scalars().all()
        for row in rows:
            bo = by_window.get(row.window_start)
            if not bo:
                continue
            filled = Decimal(str(bo.get("filledUsdtAmount") or "0"))
            row.status = "FILLED" if bo.get("status") == "FILLED" else "FAILED"
            row.order_id = bo.get("orderId")
            row.amount_in = str(int(filled * (10 ** 18)))
            row.quote_json = {
                "averagePrice": float(bo.get("price") or 0),
                "filledShareQty": bo.get("filledShareQty"),
                "source": "binance_history_sync",
            }
            row.error_message = None
            synced.append({
                "id": row.id, "window_start": row.window_start,
                "status": row.status, "order_id": row.order_id,
            })

        # 幽灵成交订正：本地 FILLED 而币安侧同 orderId 为 FAILED（FOK 受理后
        # 翻转未成交）→ 改判 FAILED 并清结算字段（2026-08 对账发现 2 笔）
        filled_rows = (await db.execute(
            select(TradeOrderModel)
            .where(TradeOrderModel.status == "FILLED")
            .where(TradeOrderModel.order_id.isnot(None))
        )).scalars().all()
        for row in filled_rows:
            bo = by_orderid.get(str(row.order_id))
            if not bo or bo.get("status") != "FAILED":
                continue
            row.status = "FAILED"
            row.win = None
            row.pnl = None
            row.settle_outcome = None
            row.settled_at = None
            row.error_message = "对账改判：币安侧订单终态 FAILED（幽灵成交订正）"
            synced.append({
                "id": row.id, "window_start": row.window_start,
                "status": "FAILED", "order_id": row.order_id,
                "corrected": "ghost_filled",
            })

        # 场景三：部分成交订正——以币安历史行 filledUsdtAmount 为权威，
        # 订正 FILLED 行 amount_in（FOK 只吃到部分深度时小于请求金额），
        # 并按结算同口径重算已结算行 pnl（赢优先股数−成本，回退 amount/均价−amount，
        # 输=−amount）；filledShareQty 同步回填 quote_json（已扣 marketProviderFee，
        # 赢向无费口径比币安实现盈亏高约费用额，生产实锤 10.5 vs 10.27）。
        for row in filled_rows:
            bo = by_orderid.get(str(row.order_id))
            if not bo or bo.get("status") != "FILLED":
                continue
            try:
                filled_usdt = Decimal(str(bo.get("filledUsdtAmount") or "0"))
            except Exception:
                continue
            if filled_usdt <= 0:
                continue
            actual_wei = int(filled_usdt * (10 ** 18))
            try:
                local_wei = int(row.amount_in or 0)
            except (TypeError, ValueError):
                local_wei = 0
            amount_mismatch = abs(actual_wei - local_wei) > 10 ** 15  # ≤0.001 USDT 视为一致（精度噪声）
            shares = TradeSettler._to_float(bo.get("filledShareQty"))
            has_local_shares = TradeSettler._shares(row) is not None
            # 跳过条件：金额一致 且（本地已有股数，或 币安未给股数/行未结算赢向 → 无新信息可回填）
            if not amount_mismatch and (
                has_local_shares
                or not (shares and row.settled_at is not None and row.win)
            ):
                continue
            amt = float(filled_usdt)
            if amount_mismatch:
                row.amount_in = str(actual_wei)
            if shares and not has_local_shares:
                qj = dict(row.quote_json or {})
                qj["filledShareQty"] = shares
                row.quote_json = qj
            avg = TradeSettler._avg_price(row)
            if row.settled_at is not None and row.win is not None:
                if row.win and shares:
                    row.pnl = shares - amt
                elif row.win and avg is not None:
                    row.pnl = amt / avg - amt
                elif not row.win:
                    row.pnl = -amt
            amount_corrected.append({
                "id": row.id, "order_id": row.order_id,
                "old_amount": local_wei / 1e18, "new_amount": amt,
                "shares": shares, "pnl": row.pnl,
            })
            logger.warning(
                "部分成交订正 | 订单 {} | id={} | 金额 {} → {} USDT | 股数 {} | pnl={}",
                row.order_id, row.id, local_wei / 1e18, amt, shares,
                f"{row.pnl:+.4f}" if row.pnl is not None else "N/A")
        await db.commit()
    return {"synced": len(synced), "details": synced,
            "amount_corrected": amount_corrected,
            "binance_orders": len(history)}


@app.post("/api/trades/sync-binance")
async def sync_binance_orders(_: None = Depends(_require_auth)):
    """对账回填：用币安侧订单历史订正本地行终态（主体见 _sync_binance_orders_impl，
    与 30min 定时对账循环共用，2026-08-28 抽函数）。"""
    return await _sync_binance_orders_impl()


@app.post("/api/trades/sync-status")
async def sync_order_status_from_binance(
    _: None = Depends(_require_auth),
):
    """从币安同步真实订单状态到本地数据库（解决前端展示与币安不一致问题）。

    背景：前端展示的盈亏 (pnl) 和"待领取 token"都是基于本地计算，
    可能与币安实际不符（例如"WIN +30.33"对应币安侧 FAILED）。
    
    功能：实时调用币安订单查询接口，更新本地的 status/win/pnl/redeemed_at。
    
    Returns:
        {
            "synced_count": int,
            "updated_orders": [
                {
                    "id": order_id,
                    "order_id": binance_order_id,
                    "old_status": "FILLED",
                    "new_status": "FAILED",
                    "old_pnl": 30.33,
                    "new_pnl": -1.0,
                    "binance_status": "FAILED",
                    "changes": {...}
                }
            ],
            "total_checked": int,
        }
    """
    from decimal import Decimal
    from sqlalchemy import select
    from .db.models import TradeOrderModel

    if not prediction_trader._api_key:
        return {"error": "Binance API Key 未配置"}
    
    synced_count = 0
    updated_orders = []
    
    # 1. 查询所有近期未完成的订单（包括 FILLED/FAILED）
    async with async_session_factory() as db:
        stmt = select(TradeOrderModel).where(
            TradeOrderModel.status.in_("FILLED", "FAILED"),
            TradeOrderModel.order_id.isnot(None),
        )
        orders = (await db.execute(stmt)).scalars().all()
    
    if not orders:
        return {"synced_count": 0, "updated_orders": [], "total_checked": 0, "message": "没有需要同步的订单"}
    
    for order in orders:
        # 2. 查询币安订单详情（通过 verify_order_by_id）
        binance_order = await prediction_trader.verify_order_by_id(order.order_id)
        
        if not binance_order:
            logger.warning(f"无法查询币安订单 {order.order_id} | 跳过")
            continue
        
        # 3. 对比状态和价格
        old_status = order.status
        old_pnl = order.pnl
        binance_status = binance_order.get("status")  # "FILLED" | "FAILED" | "EXPIRED"
        
        # 4. 更新数据库
        changes = {}
        should_update = False
        
        # 检查状态变更
        if binance_status != order.status:
            order.status = binance_status
            changes["status"] = {"old": old_status, "new": binance_status}
            should_update = True
            logger.info(f"订单 {order.order_id} 状态变更：{old_status} → {binance_status}")
        
        # 检查价格变更（如果成交了）
        if binance_status == "FILLED":
            avg_price = binance_order.get("averagePrice")
            filled_amount = binance_order.get("amountIn")  # 实际成交金额
            
            if avg_price and float(avg_price) != float(order.quote_json.get("averagePrice", 0)):
                order.quote_json["averagePrice"] = float(avg_price)
                order.amount_in = str(int(float(filled_amount) * 1e18)) if filled_amount else order.amount_in
                changes["price"] = {
                    "old": order.quote_json.get("averagePrice"),
                    "new": avg_price
                }
                should_update = True
                logger.info(f"订单 {order.order_id} 均价更新：{order.quote_json.get('averagePrice')} → {avg_price}")
        
        # 5. 重新计算 pnl（使用真实成交价）
        new_pnl = _calculate_pnl_from_binance(order, binance_order)
        if abs(new_pnl - (order.pnl or 0)) > 0.001:
            order.pnl = new_pnl
            changes["pnl"] = {"old": order.pnl, "new": new_pnl}
            should_update = True
            logger.info(f"订单 {order.order_id} 重新计算 pnl: {order.pnl:.4f}")
        
        if should_update:
            await db.commit()
            synced_count += 1
            updated_orders.append({
                "id": order.id,
                "order_id": order.order_id,
                "old_status": old_status,
                "new_status": order.status,
                "old_pnl": old_pnl,
                "new_pnl": order.pnl,
                "binance_status": binance_status,
                "changes": changes,
            })
    
    return {
        "synced_count": synced_count,
        "updated_orders": updated_orders,
        "total_checked": len(orders),
    }


def _calculate_pnl_from_binance(order: TradeOrderModel, binance_order: dict) -> float:
    """根据币安真实订单数据计算 pnl"""
    try:
        amount_in = Decimal(binance_order.get("amountIn", "0"))
        average_price = Decimal(str(binance_order.get("averagePrice", 0)))
        
        if average_price <= 0:
            return -float(amount_in)  # 没成交就视为失败
        
        # 赢：收到的 USDT = amount_in / average_price
        usdt_received = amount_in / average_price
        
        # pnl = 收到的 USDT - 投入的 USDT
        pnl = float(usdt_received - amount_in)
        
        return pnl
        
    except Exception as e:
        logger.error(f"计算 pnl 失败：{e} | 返回 -1 作为默认值")
        return -1.0


@app.post("/api/trades/settle-scan")
async def settle_scan(_: None = Depends(_require_auth)):
    """手动触发一次结算扫描（验证/补算用）：返回本次结算条数。

    TradeSettler 常开 60s 自动扫描；本端点用于部署后验证、
    归档器恢复后的手动补算。幂等：settled_at IS NULL 守卫，
    重复调用只给未结算行生效。
    """
    if trade_settler is None:
        return {"error": "结算器未装配（启动异常？详见后端日志）", "settled": 0}
    try:
        settled = await trade_settler.poll_once()
    except Exception as e:
        return {"error": f"扫描失败: {e}", "settled": 0}
    return {"settled": settled}


@app.get("/api/logs/tail")
async def tail_logs(
    lines: int = 200,
    grep: str = "",
    _: None = Depends(_require_auth),
):
    """读 loguru 文件日志尾部（生产诊断：容器 stdout 重建即丢，文件日志持久化在 volume）。

    grep 支持子串过滤（如 CRITICAL / 订单终态落库失败）。
    """
    import os
    from collections import deque

    if not settings.log_dir:
        return {"error": "文件日志未启用（log_dir 为空）"}
    path = os.path.join(settings.log_dir, "app.log")

    # 目录清单（诊断：卷内文件属主/mtime 是否正常，非 root 容器无权写 root 属主文件）
    entries: list[dict] = []
    try:
        for name in sorted(os.listdir(settings.log_dir)):
            st = os.stat(os.path.join(settings.log_dir, name))
            import datetime as _dt
            entries.append({
                "name": name, "size": st.st_size,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(),
                "uid": st.st_uid, "gid": st.st_gid,
            })
    except Exception as e:
        entries = [{"error": f"列目录失败: {e}"}]

    if not os.path.exists(path):
        return {"error": f"日志文件不存在: {path}", "cwd": os.getcwd(), "entries": entries}

    maxlen = max(1, min(int(lines), 1000))
    buf: deque = deque(maxlen=maxlen)
    out: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if grep:
                # grep 模式：全文件流式过滤后取尾部（tail 窗口太浅会错过早期目标行）
                for ln in f:
                    ln = ln.rstrip("\n")
                    if grep in ln:
                        out.append(ln)
                out = out[-maxlen:]
                tail_lines = -1  # 全文件扫描
            else:
                for ln in f:
                    buf.append(ln.rstrip("\n"))
    except Exception as e:
        return {"error": f"读取失败: {e}"}

    if not grep:
        out = list(buf)
    return {
        "path": path,
        "tail_lines": len(buf) if not grep else tail_lines,
        "matched": len(out),
        "lines": out, "cwd": os.getcwd(), "entries": entries,
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


@app.get("/api/chart/prediction-market/15m")
async def get_prediction_market_chart_15m(
    limit: int = 400,
    since: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """
    获取 15 分钟预测市场采样曲线（从 prediction_market_samples 查 market_period='15m'）

    默认返回最近 400 个点（约 100 分钟，覆盖多个 15m 期次）的 UP/DOWN 报价与
    BTC 快照价；market 字段为检测器缓存的最新 15m 市场快照（到期时刻等）。

    历史导出（S6 定价分析用）：since>0 时按时间正序返回 timestamp>=since 的
    前 limit 条（上限 50000，约 12.5 天），配合响应中的 oldest_ts 可翻页遍历。
    """
    from sqlalchemy import asc as sa_asc, desc as sa_desc, select as sa_select

    limit = max(1, min(limit, 50_000))
    if since > 0:
        stmt = (
            sa_select(PredictionMarketSample)
            .where(PredictionMarketSample.market_period == "15m")
            .where(PredictionMarketSample.timestamp >= since)
            .order_by(sa_asc(PredictionMarketSample.timestamp))
            .limit(limit)
        )
        rows = (await db.execute(stmt)).scalars().all()
        points = [
            {
                "timestamp": s.timestamp,
                "up_price": s.up_price,
                "down_price": s.down_price,
                "up_pct": s.up_pct,
                "down_pct": s.down_pct,
                "btc_price": s.btc_price,
            }
            for s in rows
        ]
    else:
        stmt = (
            sa_select(PredictionMarketSample)
            .where(PredictionMarketSample.market_period == "15m")
            .order_by(sa_desc(PredictionMarketSample.timestamp))
            .limit(limit)
        )
        rows = (await db.execute(stmt)).scalars().all()
        points = [
            {
                "timestamp": s.timestamp,
                "up_price": s.up_price,
                "down_price": s.down_price,
                "up_pct": s.up_pct,
                "down_pct": s.down_pct,
                "btc_price": s.btc_price,
            }
            for s in reversed(rows)
        ]
    return {
        "symbol": settings.symbol,
        "poll_interval_sec": 15,
        "points": points,
        "market": dict(_pm_15m_latest),
        "oldest_ts": points[0]["timestamp"] if points else None,
    }


# ============================================================
# 信号分析面板 API（胜率曲线 × BTC K线 × 周期归因）
# 口径与 scripts/local_shadow_full_analysis.py、local_scene_signal_full_analysis.py
# 审计后版本一致：三套 EV 口径、逐版本盈亏平衡、PUMP_TS 周期切分。
# ============================================================

# 影子版本回测冻结基准（胜率, EV, 说明）；v2/v3 门禁版基准为 2026-08-26 真实数据
# 回测（scripts/local_shadow_v2v3_real_backtest.py）：A 段=线上全历史 v1 已结算行
# 过门禁（门禁为 v1 触发集纯子集），B 段=07-13~07-30 全曲线盲验重放；合并钉胜率。
SHADOW_BENCH: dict[str, tuple[float | None, float | None, str]] = {
    "x4_v1": (0.635, 0.254, "错位: 本窗收阳&end≤40 → 次窗 DOWN"),
    "quote_momentum_v1": (0.799, 0.097, "顺势: 深折价方向同窗押注"),
    "quote_contrarian_v1": (0.240, 0.155, "逆势: 赔率型，胜率低赔率高"),
    "x4_v2": (0.553, 0.211, "错位v2: v1+|past1h|<0.5%平静市（真实回测 n=94 胜率55.3%，盲验段30笔66.7%）"),
    "quote_momentum_v2": (0.750, 0.019, "顺势v2: v1+触发时已跌≥0.10%（真实回测 n=24 胜率75.0%，小样本CI宽）"),
    "quote_contrarian_v2": (0.258, 0.235, "逆势v2: v1+触发时未涨≥0.10%（真实回测 n=155 胜率25.8%）"),
    "quote_contrarian_v3a": (0.318, 0.528, "逆势v3a: v2+前窗DOWN交替环境（真实回测 n=85 胜率31.8%）"),
    "quote_contrarian_v3b": (0.338, 0.646, "逆势v3b: v3a+距日高回落≥0.30%含边界（真实回测 n=65 胜率33.8%）"),
    # v4 regime 门禁版：基准为 Predexon 真实订单簿 62 天回测（down 段），
    # EV 基准钉 +0.372（与回测实现口径同源：赢 0.98/(q+0.01)−1 / 输 −1）
    "quote_contrarian_v4": (0.303, 0.372, "逆势v4: v1+ret24≤−1.0%下跌周期门禁（62天真实订单簿回测 down段 n=413 胜率30.3%，up/range段 EV≈0）"),
    # 深夜时段变体：基准为 720 天 K 线代理回测（非真实报价同源），只钉胜率；
    # EV 基准留 None（代理回测含溢 0.01 与影子无溢价口径不可直比）
    "late_night_contrarian_v1": (0.347, None, "深夜逆势v1: 北京22~24时×t45~90s q∈[0.25,0.30)（720天K线代理回测胜率≈34.7%/赔率型边际，43天小样本51~56%系噪声）"),
    # 深夜门禁 v2：基准为线上情绪窗重放 OOS（F1 优化发现，r6 扩样池；非 K 线代理），
    # 只钉胜率；EV 基准留 None（重放均值与逐笔影子 EV 口径不直比）
    "late_night_contrarian_v2": (0.440, None, "深夜逆势v2: v1+触发时点距日高回落≥0.30%（线上情绪窗重放OOS n=50 胜率44.0% CI[31.2%,57.7%]，盈亏平衡≈27%，剔除段IS wr 0.140/EV负）"),
    # KREV K 线反转族（2026-08-28；2026-09-03 起落库目标窗真实入场报价）：基准为 720d
    # 发现流水线冻结 holdout（样本外），只钉胜率；bench EV 留 None（无冻结 EV 回测基准；
    # 面板 EV/累计 EV 由落库真实报价前向现算，holdout 参考 EV@0.50：A +0.234 / B +0.219）
    "krev_a_v1": (0.642, None, "K线反转A: 距前低≤-0.09ATR+5根高效率阴跌+3子阴齐跌 → 押次根收阳（720d holdout n=137 胜率64.2%，月一致性0.957；EV按目标窗真实报价前向现算）"),
    "krev_b_v1": (0.634, None, "K线反转B: 区间贴底+5根高效率阴跌+3子阴齐跌 → 押次根收阳（720d holdout n=134 胜率63.4%；EV按目标窗真实报价前向现算）"),
    # HM 上吊线反弹入场族（2026-09-01）：基准为 720d 探索性回测（x=0.25 触及格），
    # 只钉胜率；EV 基准留 None（回测基准=市场隐含代理，与逐触价影子报价口径不直比）
    "hm_touch_down_v1": (0.587, None, "HM上吊线反弹入场: 弱收盘上吊线→次15m周期反弹触及+0.25×ATR→押收跌（720d n=46 触价口径收跌58.7% vs 隐含47.1%，覆盖率~36%，探索性发现待前向验证）"),
    # HM v2（2026-09-01 切片分析后验假设）：v1+非下跌段∧非低波门禁；基准为后验切片数，
    # 非预注册，影子期前向验证决定是否保留（审计锚点：scripts/hm_slice_followup_720d.py，
    # 冻结数复核脚本 scripts/hm_v2_freeze_counts_720d.py；硬闸门见 tests/test_hm_shadow_detector.py）
    "hm_touch_down_v2": (0.690, None, "HM上吊线反弹v2: v1+非下跌段∧非低波门禁（720d后验切片 门禁后触发78/触价29 收跌69.0%；下跌段与低波样本负边际被排除，待前向验证）"),
    # 反转形态 P1/P2（2026-09-03；同日补目标窗真实入场报价）：基准为 720d/oos 几何口径
    # 回测点估计，只钉胜率；bench EV 留 None（无冻结 EV 基准；面板 EV 由真实报价前向现算）
    "rev_p1_v1": (0.620, None, "反转P1: 15m连跌4+弱阴收(贴最低)+量正常[1.0,1.5) → 押次根收阳UP（720d 62.0%/oos 63.9%；EV按目标窗真实报价前向现算）"),
    "rev_p2_v1": (0.624, None, "反转P2: 15m连涨5+弱阳收(贴最高) → 押次根收阴DOWN（720d 62.4%/oos 61.3%；EV按目标窗真实报价前向现算）"),
    # S5 深档（2026-09-03）：S1+5min确认且 z5≤−20bp 深回落子集，落 pattern_shadow_signals，
    # 记 +5min 真实 DOWN 报价；基准为深档回测点估计（EV 偏乐观含机械成分），影子前向验证
    "s5_deep_z20_v1": (0.913, None, "S5深档: S1+5min回落确认且z5≤−20bp → 押次周期15m DOWN（回测~91.3%，深档EV偏乐观含机械成分，影子前向验证）"),
    # 报价动量 v3（2026-09-03）：v1 区间 ∩ 非连涨门禁，落 misalignment_signals，真实报价 EV；
    # 回测修正未来函数后门禁效应≈0（80.2% vs 连涨76.4%，CI重叠），影子前向验证门禁是否真实有效
    "quote_momentum_v3": (0.802, None, "报价动量v3: v1(t90~120s q∈[0.69,0.75))∩末收15m非连涨 → 押DOWN（修正未来函数后回测80.2% vs 连涨76.4%，+3.8pp CI重叠，影子前向验证门禁是否真实有效）"),
    # nextbar 族（2026-09-03）：H=1 方向研究冻结条件，落 kline_shadow_signals，次根收阳结算；
    # 同日补目标窗真实入场报价 → 面板 EV 前向现算；bench EV 留 None（无冻结基准），
    # 基准胜率=720d 全样本次根收阳点估计
    "nb_zschamp_15m_v1": (0.5892, None, "nextbar 15m冠军: zscore_10≤-1.651∧zscore_5≤-1.538∧ret_3≤-0.00395 深超卖反转 → 押次根15m收阳UP（converge_registry L3 ROBUST，holdout P(up_1)=61.96% n=368；720d次根收阳58.92%/2006触发，月一致性0.958/wf1.00；EV按目标窗真实报价前向现算）"),
    "nb_smaslope_5m_v1": (0.4743, None, "nextbar 5m误定价: sma_slope_atr_5≥1.661 短期动量 → 押次根5m收阳UP（阶段E市场报价钝在q̄0.500而Jul-Aug真实P(UP)=0.534，B⁺ t=1.73未达t>3；720d次根收阳仅47.43%/19597触发=长样本反指，edge依赖regime，影子前向验证是否持续；EV按目标窗真实报价前向现算）"),
    # combo 组合族（2026-09-04）：45 维条件大搜索（grand_search_v2，720d 三三组合扫描）
    # + 1443 天样本外考试 + 50 次置换检验三重过滤存活 5 组合，落 kline_shadow_signals，
    # 次根收盘按 direction 结算（DOWN收阴赢/UP收阳赢）；基准胜率=720d 冻结口径点估计、
    # oos=2020-10~2024-09 样本外（bench EV 留 None 无冻结基准，面板 EV 前向现算）
    "combo_p1_v1": (0.639, None, "combo组合P1: 连阳3+∧周末∧EMA20乖离≥+0.3% → 押次根15m收阴DOWN（720d n=490 63.9%/oos n=1588 60.6%；时间×动量维度组合非K线形态，45维搜索存活真层；EV按目标窗真实报价前向现算）"),
    "combo_p2_v1": (0.665, None, "combo组合P2: 大实体(body_bp≥23.46)∧连阳3+∧周末 → 押次根15m收阴DOWN（720d n=206 66.5%/oos n=957 60.5%；大实体为研究美元P80的bp冻结口径；EV按目标窗真实报价前向现算）"),
    "combo_p3_v1": (0.682, None, "combo组合P3: 贴1天高∧美盘(UTC h≥16)∧7d涨≥4% → 押次根15m收阴DOWN（720d n=176 68.2%/oos n=212 59.4%；高位+美盘时段+中期动量衰减；EV按目标窗真实报价前向现算）"),
    "combo_p4_v1": (0.627, None, "combo组合P4: 收低位(pos≤0.25)∧周末∧RSI14≤25 → 押次根15m收阳UP（720d n=322 62.7%/oos n=459 63.6%；周末超卖反弹；EV按目标窗真实报价前向现算）"),
    "combo_p5_v1": (0.687, None, "combo组合P5: 近光脚∧周末∧RSI14≤25 → 押次根15m收阳UP（720d n=131 68.7%/oos n=142 64.8%；P4同簇严格版（下影≤5%），影子期实数据对比两口径；EV按目标窗真实报价前向现算）"),
}
# 周期切分点：08-19 00:00 UTC（三根大阳起点）；< 为震荡期（大涨前），≥ 为大涨期
PUMP_TS_MS = int(datetime(2026, 8, 19, tzinfo=timezone.utc).timestamp() * 1000)

# BTC K 线图表缓存：interval:档位 -> (缓存时刻, klines)，避免前端轮询打爆 Binance。
# limit 就近向上归档到固定档位，防止任意 limit 枚举缓存键绕过保护。
_BTC_KLINE_LIMIT_TIERS = (30, 60, 120, 168, 200)
_btc_kline_cache: dict[str, tuple[float, list[dict]]] = {}
_BTC_KLINE_CACHE_TTL = 60.0
# 短负缓存：interval -> 上次上游失败时刻（10s 内直接返回空，避免轮询连环打上游）
_btc_kline_fail: dict[str, float] = {}
_BTC_KLINE_FAIL_TTL = 10.0


@app.get("/api/chart/btc-klines")
async def get_btc_klines(interval: str = "1d", limit: int = 30):
    """BTC K 线代理（信号分析面板背景图）：仅返回已收盘 K，升序。"""
    if interval not in ("5m", "15m", "1h", "4h", "1d"):
        raise HTTPException(status_code=422, detail=f"interval 仅支持 5m/15m/1h/4h/1d: {interval}")
    limit = max(10, min(limit, 200))
    tier = next((t for t in _BTC_KLINE_LIMIT_TIERS if t >= limit), _BTC_KLINE_LIMIT_TIERS[-1])
    now = time.time()
    if now - _btc_kline_fail.get(interval, 0.0) < _BTC_KLINE_FAIL_TTL:
        return {"interval": interval, "klines": []}
    key = f"{interval}:{tier}"
    cached = _btc_kline_cache.get(key)
    if cached and now - cached[0] < _BTC_KLINE_CACHE_TTL:
        return {"interval": interval, "klines": cached[1][-limit:]}
    klines = await collector.fetch_recent_klines(interval, tier)
    if klines:
        _btc_kline_cache[key] = (now, klines)
    else:
        _btc_kline_fail[interval] = now
    return {"interval": interval, "klines": klines[-limit:] if klines else []}


def _shadow_breakeven(version: str, q: float) -> float:
    """逐笔盈亏平衡胜率（与各版本 EV 口径一致）：x4 系含溢 0.01，其余无溢价。"""
    return (q + 0.01) / 0.98 if version.startswith("x4") else q / 0.98


def _shadow_realized_ev(version: str, win: bool, q: float | None) -> float | None:
    """逐笔实现 EV（与各检测器落库 ev_at_entry 同口径）：赢 0.98/entry−1 / 输 −1。

    entry 与 _shadow_breakeven 一致：x4 系含溢 0.01，其余无溢价；无报价（q 空）→ None。
    供未落库逐笔 ev_at_entry 的表聚合时兜底现算：pattern_shadow_signals（HM/S5）按落库
    报价，kline_shadow_signals（KREV/P1/P2/nextbar/combo）按落库目标窗入场报价（entry_up/down_price）；
    报价缺失行（如冷启动回补、窗口未对齐）q 空 → None。
    """
    if not q:
        return None
    entry = q + 0.01 if version.startswith("x4") else q
    return (0.98 / entry - 1.0) if win else -1.0


async def _official_scene_version_filter(db: AsyncSession):
    """正式场景信号的 version 过滤条件：排除 SHADOW 版本名，其余均为正式信号。

    FakeBreakoutSignal.version 语义：NULL/v1/历史 ACTIVE 名（如 v1-20260816）均为
    正式信号；仅 SceneParamVersion 中 status=SHADOW 的版本名是影子对照行。
    旧口径「version NULL/v1」在 ACTIVE 版本演进后会把正式信号误排除
    （2026-08-16 起 ACTIVE=v1-20260816，导致 stats/面板只剩少量历史样本）。
    NULL 行须显式 OR 保留（SQL 中 NOT (NULL IN ...) 不为 TRUE）。
    """
    from sqlalchemy import not_ as sa_not, or_ as sa_or, select as sa_select, true as sa_true

    from .db.models import SceneParamVersion

    shadow_names = [r[0] for r in (await db.execute(
        sa_select(SceneParamVersion.version).where(SceneParamVersion.status == "SHADOW")
    )).all()]
    if not shadow_names:
        return sa_true()
    return sa_or(
        FakeBreakoutSignal.version.is_(None),
        sa_not(FakeBreakoutSignal.version.in_(shadow_names)),
    )


_CURVE_MAX_POINTS = 500  # 单版本曲线点数上限（防响应体随信号量无界膨胀）


@app.get("/api/signals/analytics")
async def get_signals_analytics(db: AsyncSession = Depends(get_db)):
    """信号分析面板聚合端点：口径常量固化在后端（单一事实源）。

    - shadow: 影子各版本累计胜率/EV 曲线 + 汇总（含逐笔平均盈亏平衡、回测基准）
      —— EV 优先直读落库 ev_at_entry（与审计「逐笔交叉验证零误差」口径同源）；
      未落库的表（pattern 系 HM/S5）按落库报价兜底现算实现 EV（赢 0.98/q−1 / 输 −1）
    - scene: 各场景（pattern_type）累计胜率曲线 + 汇总（基准=research_win_rates）；
      EV 按审计口径现算（费2%+溢0.01 逐笔实现 EV：赢 0.98/(q+0.01)−1 截断 / 输 −1，
      q 按 side 取 entry_up/down_15m，缺失不计入）——落库 ev_at_entry 是期望 EV 口径，
      与审计实现口径不可混用
    - regime: 大涨前(<08-19 UTC) vs 大涨期胜率对比（合并 + 逐影子版本）+ 按 UTC 日胜率
    胜负判定（场景/legacy）按 side 映射：side=high 押 DOWN、side=low 押 UP，
    与 /api/fake-breakout/stats 的 compute_pattern_stats 同语义。
    """
    from sqlalchemy import literal as sa_literal, select as sa_select

    from .db.models import KlineShadowSignal, MisalignmentSignal, PatternShadowSignal
    from .services.fake_breakout_detector import RESEARCH_WIN_RATES

    # ---- 影子信号：全量 SETTLED 升序（仅取所需列，避免整行 ORM 实体化）----
    sh_rows = (await db.execute(
        sa_select(
            MisalignmentSignal.version, MisalignmentSignal.window_start,
            MisalignmentSignal.win, MisalignmentSignal.ev_at_entry,
            MisalignmentSignal.entry_down_price, MisalignmentSignal.entry_up_price,
            MisalignmentSignal.direction,
        )
        .where(MisalignmentSignal.status == "SETTLED")
        .order_by(MisalignmentSignal.window_start)
    )).all()
    # KREV/反转/nextbar/combo（kline_shadow_signals 表）：K 线族影子，结算口径为次根 K 线涨跌；
    # 2026-09-03 起落库目标窗真实入场报价（entry_up/down_price），聚合层按 direction 取
    # 对应侧 q 现算真实 EV（赢 0.98/q−1 / 输 −1）；报价缺失/冷启动回补行 entry 为 NULL →
    # 该笔 EV 不计（与旧「纯 K 线无报价」口径兼容）。ev_at_entry 仍未落库（恒 None，兜底现算）。
    krev_rows = (await db.execute(
        sa_select(
            KlineShadowSignal.version,
            KlineShadowSignal.target_bar_start.label("window_start"),
            KlineShadowSignal.win,
            sa_literal(None).label("ev_at_entry"),
            KlineShadowSignal.entry_down_price.label("entry_down_price"),
            KlineShadowSignal.entry_up_price.label("entry_up_price"),
            KlineShadowSignal.direction.label("direction"),
        )
        .where(KlineShadowSignal.status == "SETTLED")
        .order_by(KlineShadowSignal.target_bar_start)
    )).all()
    # HM 上吊线反弹入场（pattern_shadow_signals 表，2026-09-01 并入）：窗内触价入场影子，
    # 结算口径=目标根收阴（押 DOWN），入场报价=触及时刻 DOWN 真实报价；仅 TOUCHED 行
    # 进 SETTLED（其余入场态无结算不进面板）。逐笔 EV 未落库（ev 恒 None），
    # 聚合时按 entry_down_quote 兜底现算实现 EV（_shadow_realized_ev）。
    pattern_rows = (await db.execute(
        sa_select(
            PatternShadowSignal.version,
            PatternShadowSignal.target_bar_start.label("window_start"),
            PatternShadowSignal.win,
            sa_literal(None).label("ev_at_entry"),
            PatternShadowSignal.entry_down_quote.label("entry_down_price"),
            sa_literal(None).label("entry_up_price"),
            sa_literal("DOWN").label("direction"),
        )
        .where(PatternShadowSignal.status == "SETTLED")
        .order_by(PatternShadowSignal.target_bar_start)
    )).all()
    sh_rows = list(sh_rows) + list(krev_rows) + list(pattern_rows)
    # 版本 = 冻结基准已知版本 ∪ 数据中出现的版本（新版本缺基准不崩，bench 为 None）
    versions = [
        "x4_v1", "quote_momentum_v1", "quote_contrarian_v1",
        "x4_v2", "quote_momentum_v2", "quote_contrarian_v2",  # v2 门禁版（部署即入面板）
        "quote_contrarian_v3a", "quote_contrarian_v3b",  # v3 环境门禁版（可选实盘通道，默认 OFF）
        "quote_contrarian_v4",  # v4 regime 门禁版（下跌周期，默认 OFF）
        "late_night_contrarian_v1",  # 深夜时段变体（纯影子，2026-08-26）
        "late_night_contrarian_v2",  # 深夜门禁 v2（纯影子，2026-08-27：v1+距日高回落≥0.30%）
        "krev_a_v1", "krev_b_v1",  # K 线反转族（纯影子，2026-08-28：新表 kline_shadow_signals）
        "hm_touch_down_v1",  # HM 上吊线反弹入场（纯影子，2026-09-01：新表 pattern_shadow_signals）
        "hm_touch_down_v2",  # HM v2：v1+非下跌段∧非低波门禁（纯影子，2026-09-01 后验切片）
        "rev_p1_v1", "rev_p2_v1",  # 反转形态 P1/P2（纯影子，2026-09-03：共表 kline_shadow_signals）
        "nb_zschamp_15m_v1", "nb_smaslope_5m_v1",  # nextbar 族（纯影子，2026-09-03：共表 kline_shadow_signals，15m+5m 双 tf）
        "combo_p1_v1", "combo_p2_v1", "combo_p3_v1", "combo_p4_v1", "combo_p5_v1",  # combo 组合族（纯影子，2026-09-04：共表 kline_shadow_signals，5 组合 3DOWN/2UP）
        "s5_deep_z20_v1",  # S5 深档 z5≤−20bp（纯影子，2026-09-03：共表 pattern_shadow_signals）
        "quote_momentum_v3",  # 报价动量 v3 非连涨门禁（纯影子，2026-09-03：misalignment_signals）
    ]
    versions += sorted({s.version for s in sh_rows} - set(versions))
    shadow = {}
    for v in versions:
        g = [s for s in sh_rows if s.version == v and s.win is not None]
        curve, wins, evs, bes = [], 0, [], []
        cum_ev = 0.0
        for i, s in enumerate(g, 1):
            wins += int(bool(s.win))
            q = s.entry_down_price if s.direction == "DOWN" else s.entry_up_price
            # EV 优先读落库 ev_at_entry；未落库的表（pattern 系 HM/S5、kline 系
            # KREV/P1/P2/nextbar/combo）按落库报价兜底现算实现 EV；报价缺失行 q 空 → None
            ev = (float(s.ev_at_entry) if s.ev_at_entry is not None
                  else _shadow_realized_ev(v, bool(s.win), float(q) if q else None))
            if ev is not None:
                evs.append(ev)
                cum_ev += ev
            if q:
                bes.append(_shadow_breakeven(v, float(q)))
            curve.append({
                "i": i, "ts": s.window_start,
                "cum_wr": round(wins / i, 4), "cum_ev": round(cum_ev, 4),
            })
        n = len(g)
        bwr, bev, desc = SHADOW_BENCH.get(v, (None, None, ""))
        shadow[v] = {
            "summary": {
                "n": n,
                "win_rate": wins / n if n else None,
                "avg_ev": sum(evs) / len(evs) if evs else None,
                "cum_ev": round(cum_ev, 4) if evs else None,
                "avg_breakeven": sum(bes) / len(bes) if bes else None,
                "bench_winrate": bwr, "bench_ev": bev, "desc": desc,
                # 影子开关状态（前端手动下线能力）：下线版本置灰+可重新上线
                "enabled": shadow_gate.is_enabled(v),
            },
            "curve": curve[-_CURVE_MAX_POINTS:],
        }

    # ---- 场景信号：正式信号（排除 SHADOW 版本名；ACTIVE 版本演进兼容）按 pattern_type 分组 ----
    official_ver = await _official_scene_version_filter(db)
    sc_rows = (await db.execute(
        sa_select(
            FakeBreakoutSignal.signal_time, FakeBreakoutSignal.side,
            FakeBreakoutSignal.settle_outcome, FakeBreakoutSignal.pattern_type,
            FakeBreakoutSignal.pattern, FakeBreakoutSignal.cumulative_winrate,
            FakeBreakoutSignal.entry_down_price_15m, FakeBreakoutSignal.entry_up_price_15m,
        )
        .where(FakeBreakoutSignal.status == "SETTLED")
        .where(official_ver)
        .order_by(FakeBreakoutSignal.signal_time)
    )).all()
    by_pt: dict[str, list] = {}
    for r in sc_rows:
        by_pt.setdefault(r.pattern_type or r.pattern or "legacy", []).append(r)
    scene = {}
    for pt, rows_pt in sorted(by_pt.items()):
        curve, wins, evs = [], 0, []
        cum_ev = 0.0
        for i, r in enumerate(rows_pt, 1):
            won = r.settle_outcome == ("DOWN" if r.side == "high" else "UP")
            wins += int(won)
            # 审计口径逐笔实现 EV：赢 0.98/(q+0.01)−1（截断[0.01,0.99]）/ 输 −1；
            # q 按 side 取入场报价，缺失不计入（与 local_scene_signal_full_analysis.py 一致）
            q = r.entry_up_price_15m if r.side == "low" else r.entry_down_price_15m
            ev = None
            if q and float(q) > 0:
                ev = (0.98 / min(max(float(q) + 0.01, 0.01), 0.99) - 1.0) if won else -1.0
                evs.append(ev)
                cum_ev += ev
            # 累计胜率优先 DB 落库字段（detector 同口径），缺失回退自算
            cw = r.cumulative_winrate if r.cumulative_winrate is not None else wins / i
            curve.append({
                "i": i, "ts": r.signal_time,
                "cum_wr": round(float(cw), 4), "cum_ev": round(cum_ev, 4),
            })
        n = len(rows_pt)
        scene[pt] = {
            "summary": {
                "n": n,
                "winrate": wins / n if n else None,
                "avg_ev": sum(evs) / len(evs) if evs else None,
                "cum_ev": round(cum_ev, 4) if evs else None,
                "bench_winrate": RESEARCH_WIN_RATES.get(pt),
            },
            "curve": curve[-_CURVE_MAX_POINTS:],
        }

    # ---- 周期归因：场景 + 影子合并的 pre/pump 切分、逐影子版本拆分与按天胜率 ----
    pairs: list[tuple[int, bool]] = [
        (s.window_start, bool(s.win))
        for s in sh_rows if s.win is not None
    ]
    pairs += [
        (r.signal_time, r.settle_outcome == ("DOWN" if r.side == "high" else "UP"))
        for rows_pt in by_pt.values() for r in rows_pt
    ]
    phases: dict[str, list[int]] = {}
    daily: dict[str, list[int]] = {}
    # 逐影子版本 × 阶段（对齐审计报告「表二」的归因维度）
    by_version: dict[str, dict[str, dict]] = {}
    for s in sh_rows:
        if s.win is None:
            continue
        ph = "pump" if s.window_start >= PUMP_TS_MS else "pre"
        g = by_version.setdefault(s.version, {}).setdefault(ph, {"n": 0, "wins": 0})
        g["n"] += 1
        g["wins"] += int(bool(s.win))
    for v in by_version:
        for ph, g in by_version[v].items():
            g["winrate"] = g["wins"] / g["n"] if g["n"] else None
    for ts, won in pairs:
        ph = phases.setdefault("pump" if ts >= PUMP_TS_MS else "pre", [0, 0])
        ph[0] += 1
        ph[1] += int(won)
        day = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        d = daily.setdefault(day, [0, 0])
        d[0] += 1
        d[1] += int(won)
    return {
        "pump_ts": PUMP_TS_MS,
        "shadow": shadow,
        "scene": scene,
        "regime": {
            "phases": {
                ph: {"n": g[0], "wins": g[1], "winrate": g[1] / g[0] if g[0] else None}
                for ph, g in phases.items()
            },
            "by_version": by_version,
            "daily": [
                {"date": day, "n": g[0], "wins": g[1], "winrate": g[1] / g[0] if g[0] else None}
                for day, g in sorted(daily.items())
            ],
        },
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
            "curve_btc_price": w.curve_btc_price,
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
        return {"status": "deprecated", "message": "系统B预测循环已退役（2026-08-16 拍板），deep-learn 系列端点随之退役"}

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
            yield f"data: {json.dumps({'type': 'error', 'message': '系统B预测循环已退役（2026-08-16 拍板），deep-learn 系列端点随之退役'}, ensure_ascii=False)}\n\n"
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
        return {"status": "deprecated", "message": "系统B预测循环已退役（2026-08-16 拍板），deep-learn 系列端点随之退役"}

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
    """把一次 deep-learn 结果压成多维对比摘要（双轨：holdout 维度 + Q6 初筛维度）。

    旧轨（PY_CLUSTER/LEGACY）维度保持不变：发现数、平均 holdout 胜率 / Wilson
    下界、holdout 样本量、平均 confidence、通过 P0-3 准入闸门的比例、方向分布。
    谓词轨（LLM_DEEP 新轨，predicate 非空）追加：假设条数、screen 三档裁决计数、
    过闸（ACTIVE+OBSERVE 可写库）比例、平均 screen lift、分箱版本。
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

    # --- 谓词轨（Q6 初筛）维度 ---
    pred_track = [d for d in discoveries if d.get("predicate")]
    pn = len(pred_track)
    screen_lifts = [
        d["screen_lift"] for d in pred_track if d.get("screen_lift") is not None
    ]
    active_n = sum(1 for d in pred_track if d.get("screen_verdict") == "ACTIVE")
    observe_n = sum(1 for d in pred_track if d.get("screen_verdict") == "OBSERVE")
    reject_n = sum(1 for d in pred_track if d.get("screen_verdict") == "REJECT")

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
        # 谓词轨扩展（旧轨结果这些字段为 0/None，不影响旧面板）
        "binning_version": result.get("binning_version"),
        "predicate_count": pn,
        "screen_active_count": active_n,
        "screen_observe_count": observe_n,
        "screen_reject_count": reject_n,
        "screen_admitted_count": active_n + observe_n,
        "screen_admitted_ratio": round((active_n + observe_n) / pn, 4) if pn else 0.0,
        "avg_screen_lift": _avg(screen_lifts),
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
        return {"status": "deprecated", "message": "系统B预测循环已退役（2026-08-16 拍板），deep-learn 系列端点随之退役"}

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
        return {"status": "deprecated", "message": "系统B预测循环已退役（2026-08-16 拍板），deep-learn 系列端点随之退役"}

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
