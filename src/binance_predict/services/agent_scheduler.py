"""
情绪曲线自进化 Agent Loop —— AgentScheduler 调度器

本模块实现 Agent 各阶段的事件驱动调度：
- 优先级队列（asyncio.PriorityQueue）按优先级出队事件
- 双 Worker 架构（Plan 步骤 11）：PREDICT 快速通道 + HEAVY 重型通道
  - PREDICT worker：只读 ACTIVE 模式，不持锁，独立队列
  - HEAVY worker：串行执行 Validate → Learn → Evolve，写入时持 _write_lock
- 回退单 Worker：feature flag agent_dual_worker_enabled=False 时使用原单 worker
- 分阶段超时：每个阶段独立超时
- PREDICT 排队超时保护（Plan 步骤 12）：排队过久的 PREDICT 事件被跳过

设计约束：
- publish() 为非阻塞 put_nowait，队列满时记录告警而非静默丢弃
- 分阶段超时（外层界）> 对应 LLM 超时（内层界），使 LLM 先以干净异常返回
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import PatternMemory, SentimentWindow
from .alerting import AlertService
from .metrics import metrics_collector

if TYPE_CHECKING:
    from .prediction_trading import BinancePredictionTrader
    from .sentiment_agent import SentimentAgent


class PhasePriority(IntEnum):
    """
    事件优先级：数值越小越先出队（asyncio.PriorityQueue 最小堆）

    - PREDICT(0)：时间敏感，最高优先级（Req 3.1）
    - WINDOW_ARCHIVED(1)：触发 Validate→Learn（Req 6.2）
    - EVOLVE(3)：重度反思，最低优先级（Req 5.1）
    """

    PREDICT = 0
    WINDOW_ARCHIVED = 1
    EVOLVE = 3


@dataclass(order=True)
class PhaseEvent:
    """
    阶段事件数据类（order=True 使其可被 PriorityQueue 比较排序）

    排序规则：先比 priority（IntEnum 值），同优先级再比 seq（单调递增保证 FIFO）。
    kind 与 payload 不参与排序（compare=False）。
    """

    priority: int
    seq: int  # 单调递增序号，保证同优先级 FIFO
    kind: str = field(compare=False)  # "PREDICT" | "WINDOW_ARCHIVED" | "EVOLVE"
    payload: dict = field(compare=False, default_factory=dict)


class AgentScheduler:
    """
    Agent 调度器：事件驱动、双 Worker 架构（可回退单 Worker）。

    双 Worker 架构（Plan 步骤 11）：
    - _predict_worker：快速通道，消费 PREDICT 事件，只读 ACTIVE 模式
    - _heavy_worker：重型通道，消费 WINDOW_ARCHIVED + EVOLVE 事件

    单 Worker 回退：agent_dual_worker_enabled=False 时使用原单 worker 串行消费。

    生命周期由 main.py lifespan 管理：start() 启动 worker，stop() 优雅停止。
    """

    def __init__(self, agent: SentimentAgent, trader: BinancePredictionTrader) -> None:
        self._agent = agent
        self._trader = trader
        self._seq_counter: int = 0
        self._running: bool = False
        self._validate_counter: int = 0
        # Item 5：累计「自上次 Evolve 以来新验证的预测样本数」，用于样本量驱动的 Evolve 触发。
        self._new_validated_since_evolve: int = 0
        self._alert_service = AlertService(metrics_collector)
        # Fix #19: 将告警服务注入 agent，使 predict 阶段交易门控能查询
        # trading_blocked 熔断标志（成本超限/阶段连续失败时阻断新交易）。
        self._agent._alert_service = self._alert_service

        # 双 Worker 架构（Plan 步骤 11）
        self._dual_worker = settings.agent_dual_worker_enabled
        self._write_lock: asyncio.Lock = asyncio.Lock()

        if self._dual_worker:
            # 双队列：PREDICT 快速通道 + HEAVY 重型通道
            self._predict_queue: asyncio.Queue[PhaseEvent] = asyncio.Queue(maxsize=128)
            self._heavy_queue: asyncio.PriorityQueue[PhaseEvent] = asyncio.PriorityQueue(
                maxsize=256
            )
            self._predict_worker_task: asyncio.Task | None = None
            self._heavy_worker_task: asyncio.Task | None = None
            # 兼容旧的 _queue 引用（单 worker 模式下使用）
            self._queue: asyncio.PriorityQueue[PhaseEvent] = asyncio.PriorityQueue(
                maxsize=256
            )
            self._worker_task: asyncio.Task | None = None
        else:
            # 单 Worker 回退
            self._queue = asyncio.PriorityQueue(maxsize=256)
            self._worker_task = None
            self._predict_queue = asyncio.Queue(maxsize=128)
            self._heavy_queue = asyncio.PriorityQueue(maxsize=256)
            self._predict_worker_task = None
            self._heavy_worker_task = None

    # ======================================================================
    # 只读属性
    # ======================================================================

    @property
    def validate_counter(self) -> int:
        """当前累计完成的 Validate 次数（只读，供 API/监控查询）。"""
        return self._validate_counter

    @property
    def new_validated_since_evolve(self) -> int:
        """自上次 Evolve 以来累计新验证的预测样本数（只读，供 API/监控查询）。"""
        return self._new_validated_since_evolve

    @property
    def queue_depth(self) -> int:
        """当前队列深度（双 Worker 时聚合两个队列，单 Worker 时返回主队列深度）。"""
        if self._dual_worker:
            return self._predict_queue.qsize() + self._heavy_queue.qsize()
        return self._queue.qsize()

    # ======================================================================
    # 生命周期
    # ======================================================================

    async def start(self) -> None:
        """启动 worker 消费循环 + 冷启动检查。"""
        if self._running:
            logger.warning("AgentScheduler: start() 被重复调用，忽略")
            return

        self._running = True

        if self._dual_worker:
            self._predict_worker_task = asyncio.create_task(
                self._predict_worker_loop(), name="agent_predict_worker"
            )
            self._heavy_worker_task = asyncio.create_task(
                self._heavy_worker_loop(), name="agent_heavy_worker"
            )
            logger.info("AgentScheduler: 双 Worker 架构启动（PREDICT + HEAVY）")
        else:
            self._worker_task = asyncio.create_task(
                self._run_loop(), name="agent_scheduler_worker"
            )
            logger.info("AgentScheduler: 单 Worker 消费循环已启动")

        await self._cold_start_check()

    async def _cold_start_check(self) -> None:
        """
        冷启动检查：报告模式库状态。

        双模式架构下：
        - manual 模式：仅记录日志，提示用户通过 API 触发深度分析
        - auto 模式：模式库为空时自动调用 agent.learn()
        """
        try:
            active_count = await self._count_active_patterns()
            logger.info(
                "AgentScheduler: 冷启动检查 | ACTIVE 模式数={}",
                active_count,
            )

            if active_count == 0:
                if settings.agent_learn_mode == "manual":
                    logger.warning(
                        "AgentScheduler: 模式库为空 | 请通过 API "
                        "POST /api/sentiment/agent/deep-learn 触发手动模式发现"
                    )
                else:
                    # auto 模式：保留旧逻辑，自动触发 Learn
                    logger.info(
                        "AgentScheduler: 模式库为空，触发冷启动学习（auto 模式）| "
                        "直调 agent.learn() 进行初始模式发现"
                    )
                    await self._agent.learn()
                    logger.info("AgentScheduler: 冷启动学习完成")
        except Exception as exc:
            # 冷启动失败不阻塞系统启动，记录错误后继续
            logger.error(
                "AgentScheduler: 冷启动检查/学习失败 | error_type={} | error={} | "
                "系统将正常运行，等待后续窗口归档或手动触发",
                type(exc).__name__,
                str(exc),
            )

    async def stop(self) -> None:
        """优雅停止 worker：取消所有 worker task。"""
        if not self._running:
            logger.debug("AgentScheduler: stop() 调用时未在运行，跳过")
            return

        self._running = False

        tasks_to_cancel = []
        if self._dual_worker:
            tasks_to_cancel = [
                ("predict", self._predict_worker_task),
                ("heavy", self._heavy_worker_task),
            ]
        else:
            tasks_to_cancel = [("worker", self._worker_task)]

        for name, task in tasks_to_cancel:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._worker_task = None
        self._predict_worker_task = None
        self._heavy_worker_task = None
        logger.info("AgentScheduler: 所有 worker 已停止")

    # ======================================================================
    # 事件发布（由 tracker / archiver 调用）
    # ======================================================================

    def publish(self, kind: str, payload: dict | None = None) -> None:
        """非阻塞入队事件。双 Worker 时按 kind 路由到对应队列。"""
        priority_map = {
            "PREDICT": PhasePriority.PREDICT,
            "WINDOW_ARCHIVED": PhasePriority.WINDOW_ARCHIVED,
            "EVOLVE": PhasePriority.EVOLVE,
        }

        priority = priority_map.get(kind)
        if priority is None:
            logger.error("AgentScheduler.publish: 未知事件类型 kind='{}' | 已丢弃", kind)
            return

        self._seq_counter += 1
        event = PhaseEvent(
            priority=int(priority),
            seq=self._seq_counter,
            kind=kind,
            payload=payload or {},
        )

        # 双 Worker 路由
        if self._dual_worker:
            if kind == "PREDICT":
                # PREDICT 事件附加入队时间戳（Plan 步骤 12）
                event.payload["_enqueued_at"] = time.monotonic()
                try:
                    self._predict_queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "AgentScheduler.publish: PREDICT 队列已满 | kind={} seq={}",
                        kind, event.seq,
                    )
            else:
                try:
                    self._heavy_queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning(
                        "AgentScheduler.publish: HEAVY 队列已满 | kind={} seq={}",
                        kind, event.seq,
                    )
            logger.debug(
                "AgentScheduler.publish: 入队成功 | kind={} seq={} queue={}",
                kind, event.seq,
                "predict" if kind == "PREDICT" else "heavy",
            )
        else:
            # 单 Worker 回退
            try:
                self._queue.put_nowait(event)
                logger.debug(
                    "AgentScheduler.publish: 入队成功 | kind={} seq={} queue_size={}",
                    kind, event.seq, self._queue.qsize(),
                )
            except asyncio.QueueFull:
                logger.warning(
                    "AgentScheduler.publish: 队列已满（maxsize=256）| "
                    "事件被丢弃 kind={} seq={}",
                    kind, event.seq,
                )

    # ======================================================================
    # 单 worker 消费循环（Req 6.3：同一时刻至多一个阶段执行）
    # ======================================================================

    async def _run_loop(self) -> None:
        """
        单 worker 消费循环：从 PriorityQueue 取事件 → 分派到对应阶段。

        循环直到 _running=False 或 task 被 cancel。
        使用短超时的 asyncio.wait_for(queue.get()) 以便及时响应 stop() 信号。
        """
        logger.debug("AgentScheduler._run_loop: worker 循环开始")

        _loop_iteration = 0
        while self._running:
            try:
                # 使用 1 秒超时轮询，以便及时响应 _running=False
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # 队列为空且超时 → 继续循环等待
                continue
            except asyncio.CancelledError:
                logger.debug("AgentScheduler._run_loop: worker 被取消")
                break

            # 定期采样队列深度（每 10 次迭代）
            _loop_iteration += 1
            if _loop_iteration % 10 == 0:
                metrics_collector.record_queue_depth(self._queue.qsize())

            # 分派事件到对应阶段
            try:
                await self._dispatch(event)
            except asyncio.CancelledError:
                logger.debug("AgentScheduler._run_loop: _dispatch 被取消")
                break
            except Exception as exc:
                # 阶段执行异常不应杀死 worker（否则后续事件无人消费）
                logger.error(
                    "AgentScheduler._run_loop: _dispatch 异常 | kind={} | "
                    "error_type={} | error={}",
                    event.kind,
                    type(exc).__name__,
                    str(exc),
                )
                metrics_collector.record_phase(
                    event.kind, 0.0, False, type(exc).__name__
                )
            finally:
                self._queue.task_done()

            # 每次 dispatch 后执行告警检查
            try:
                self._alert_service.check_and_alert()
            except Exception:
                pass  # 告警失败不影响主循环

        logger.debug("AgentScheduler._run_loop: worker 循环结束")

    # ======================================================================
    # 双 Worker 架构（Plan 步骤 11）
    # ======================================================================

    async def _predict_worker_loop(self) -> None:
        """PREDICT 快速通道 worker：只消费 PREDICT 事件，不持锁。"""
        logger.debug("AgentScheduler._predict_worker_loop: PREDICT worker 启动")

        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._predict_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Plan 步骤 12：排队超时检查
            enqueued_at = event.payload.pop("_enqueued_at", None)
            if enqueued_at is not None:
                wait_time = time.monotonic() - enqueued_at
                if wait_time > settings.agent_predict_max_queue_wait:
                    logger.warning(
                        "AgentScheduler: PREDICT 排队超时 | wait={:.1f}s > "
                        "max={:.1f}s | 跳过本次预测",
                        wait_time,
                        settings.agent_predict_max_queue_wait,
                    )
                    metrics_collector.record_phase(
                        "PREDICT", wait_time, False, "QUEUE_TIMEOUT"
                    )
                    self._predict_queue.task_done()
                    continue

            try:
                await self._dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "AgentScheduler._predict_worker_loop: 异常 | kind={} | {}",
                    event.kind, exc,
                )
                metrics_collector.record_phase(
                    event.kind, 0.0, False, type(exc).__name__
                )
            finally:
                self._predict_queue.task_done()

            # 告警检查
            try:
                self._alert_service.check_and_alert()
            except Exception:
                pass

        logger.debug("AgentScheduler._predict_worker_loop: PREDICT worker 结束")

    async def _heavy_worker_loop(self) -> None:
        """HEAVY 重型通道 worker：消费 WINDOW_ARCHIVED + EVOLVE 事件。"""
        logger.debug("AgentScheduler._heavy_worker_loop: HEAVY worker 启动")

        _loop_iteration = 0
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._heavy_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            _loop_iteration += 1
            if _loop_iteration % 10 == 0:
                metrics_collector.record_queue_depth(self._heavy_queue.qsize())

            try:
                # HEAVY 事件写入时持 _write_lock（保护 pattern_memory）
                async with self._write_lock:
                    await self._dispatch(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "AgentScheduler._heavy_worker_loop: 异常 | kind={} | {}",
                    event.kind, exc,
                )
                metrics_collector.record_phase(
                    event.kind, 0.0, False, type(exc).__name__
                )
            finally:
                self._heavy_queue.task_done()

            try:
                self._alert_service.check_and_alert()
            except Exception:
                pass

        logger.debug("AgentScheduler._heavy_worker_loop: HEAVY worker 结束")

    # ======================================================================
    # 事件分派（含分阶段超时 Req 6.4 + 阶段计数器 Req 5.1/6.5）
    # ======================================================================

    async def _dispatch(self, event: PhaseEvent) -> None:
        """
        根据事件类型分派到对应阶段执行，每个阶段调用包裹 asyncio.wait_for 超时保护。

        子方法返回 bool：True=成功，False=失败（超时/异常）。
        _dispatch 据此记录 metrics 和日志。
        """
        logger.info(
            "AgentScheduler._dispatch: 开始处理 | kind={} seq={} payload_keys={}",
            event.kind,
            event.seq,
            list(event.payload.keys()),
        )

        t0 = time.monotonic()
        dispatch_success = True
        dispatch_error: str | None = None

        try:
            if event.kind == "PREDICT":
                ok = await self._dispatch_predict(event)
            elif event.kind == "WINDOW_ARCHIVED":
                ok = await self._dispatch_window_archived(event)
            elif event.kind == "EVOLVE":
                ok = await self._dispatch_evolve(event)
            else:
                logger.warning(
                    "AgentScheduler._dispatch: 未知事件类型 kind='{}'，跳过",
                    event.kind,
                )
                return

            if not ok:
                dispatch_success = False
                dispatch_error = f"{event.kind}_FAILED"
        except Exception as exc:
            dispatch_success = False
            dispatch_error = type(exc).__name__

        # 记录阶段耗时到 MetricsCollector
        duration = time.monotonic() - t0
        metrics_collector.record_phase(
            event.kind, duration, dispatch_success, dispatch_error
        )

        logger.info(
            "AgentScheduler._dispatch: 完成 | kind={} seq={} duration={:.2f}s success={} error={}",
            event.kind,
            event.seq,
            duration,
            dispatch_success,
            dispatch_error,
        )

    async def _dispatch_predict(self, event: PhaseEvent) -> bool:
        """PREDICT 事件分派。返回 True=成功，False=失败。"""
        window_end_ms = event.payload.get("window_end_ms", 0)
        current_curve = event.payload.get("current_curve", [])
        timeout = settings.agent_phase_timeouts["PREDICT"]
    
        try:
            await asyncio.wait_for(
                self._agent.predict(
                    window_end_ms=window_end_ms,
                    current_curve=current_curve,
                ),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            logger.error(
                "AgentScheduler: PREDICT 阶段超时 | "
                "超时上限={}s | window_end_ms={} | "
                "该阶段视为失败（Req 6.4），Predict 内部已降级为 NO_TRADE",
                timeout,
                window_end_ms,
            )
            return False

    async def _dispatch_window_archived(self, event: PhaseEvent) -> bool:
        """
        WINDOW_ARCHIVED 事件分派：先 Validate 再 Learn。

        返回 True=两个阶段均成功，False=任一阶段失败。
        """
        window_id = event.payload.get("window_id")
        if window_id is None:
            logger.error(
                "AgentScheduler._dispatch: WINDOW_ARCHIVED 事件缺少 window_id，跳过"
            )
            return False

        window = await self._fetch_window(window_id)
        if window is None:
            logger.error(
                "AgentScheduler._dispatch: 未找到 SentimentWindow id={}，跳过",
                window_id,
            )
            return False

        all_succeeded = True

        # --- Validate 阶段 ---
        validate_timeout = settings.agent_phase_timeouts["VALIDATE"]
        try:
            logger.debug(
                "AgentScheduler: WINDOW_ARCHIVED → 执行 Validate | "
                "window_id={} | timeout={}s",
                window_id,
                validate_timeout,
            )
            validated_ids = await asyncio.wait_for(
                self._agent.validate(window),
                timeout=validate_timeout,
            )

            self._validate_counter += 1
            # Item 5：累计本轮 Evolve 以来新验证的预测样本数（validated_ids 长度）。
            self._new_validated_since_evolve += len(validated_ids)
            logger.debug(
                "AgentScheduler: Validate 完成 | window_id={} | "
                "validated_ids={} | validate_counter={} | new_validated_since_evolve={}",
                window_id,
                validated_ids,
                self._validate_counter,
                self._new_validated_since_evolve,
            )

            # Evolve 触发判定（Item 5）：默认按「新验证样本量」驱动，确保进化建立在
            # 足够新证据上；windows 模式回退旧的「每 N 次窗口归档」行为。
            if settings.agent_evolve_trigger_mode == "samples":
                should_evolve = (
                    self._new_validated_since_evolve
                    >= settings.agent_evolve_min_new_samples
                )
                evolve_reason = (
                    f"new_validated_since_evolve={self._new_validated_since_evolve} "
                    f">= min_new_samples={settings.agent_evolve_min_new_samples}"
                )
            else:  # "windows" 回退
                should_evolve = (
                    self._validate_counter > 0
                    and self._validate_counter % settings.agent_evolve_interval == 0
                )
                evolve_reason = (
                    f"validate_counter={self._validate_counter} 达到 "
                    f"evolve_interval={settings.agent_evolve_interval} 倍数"
                )

            if should_evolve:
                logger.info(
                    "AgentScheduler: 触发 EVOLVE 事件（Req 5.1/6.5）| mode={} | {}",
                    settings.agent_evolve_trigger_mode,
                    evolve_reason,
                )
                self.publish("EVOLVE")
                # 重置样本累计器（windows 模式下该字段不参与触发，重置无副作用）。
                self._new_validated_since_evolve = 0

        except asyncio.TimeoutError:
            all_succeeded = False
            logger.error(
                "AgentScheduler: VALIDATE 阶段超时 | "
                "超时上限={}s | window_id={} | "
                "该阶段视为失败（Req 6.4），跳过计数器递增",
                validate_timeout,
                window_id,
            )

        # --- Learn 阶段：双模式架构下自动模式跳过 LLM ---
        if settings.agent_learn_mode == "manual":
            logger.debug(
                "AgentScheduler: 手动模式跳过自动 Learn | "
                "请使用 POST /api/sentiment/agent/deep-learn 触发深度分析"
            )
        else:
            # auto 模式：保留旧逻辑
            learn_timeout = settings.agent_phase_timeouts["LEARN"]
            try:
                logger.debug(
                    "AgentScheduler: WINDOW_ARCHIVED → 执行 Learn (auto 模式) | "
                    "window_id={} | timeout={}s",
                    window_id,
                    learn_timeout,
                )
                await asyncio.wait_for(
                    self._agent.learn(),
                    timeout=learn_timeout,
                )
            except asyncio.TimeoutError:
                all_succeeded = False
                logger.error(
                    "AgentScheduler: LEARN 阶段超时 | "
                    "超时上限={}s | window_id={} | "
                    "该阶段视为失败（Req 6.4），本次 Learn 跳过，模式库保持不变",
                    learn_timeout,
                    window_id,
                )

        return all_succeeded

    async def _dispatch_evolve(self, event: PhaseEvent) -> bool:
        """EVOLVE 事件分派。返回 True=成功，False=失败。"""
        timeout = settings.agent_phase_timeouts["EVOLVE"]
    
        try:
            await asyncio.wait_for(
                self._agent.evolve(),
                timeout=timeout,
            )
            return True
        except asyncio.TimeoutError:
            logger.error(
                "AgentScheduler: EVOLVE 阶段超时 | "
                "超时上限={}s | "
                "该阶段视为失败（Req 6.4），模式库保持不变",
                timeout,
            )
            return False

    # ======================================================================
    # 内部辅助
    # ======================================================================

    @staticmethod
    async def _fetch_window(window_id: int) -> SentimentWindow | None:
        """
        从数据库查询 SentimentWindow 实例。

        Args:
            window_id: 情绪窗口 ID

        Returns:
            SentimentWindow 实例，不存在则返回 None
        """
        async with async_session_factory() as session:
            stmt = select(SentimentWindow).where(SentimentWindow.id == window_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    @staticmethod
    async def _count_active_patterns() -> int:
        """
        查询 pattern_memory 中 ACTIVE 状态的模式数量（用于冷启动检查）。

        Returns:
            ACTIVE 状态模式的数量
        """
        async with async_session_factory() as session:
            stmt = select(func.count(PatternMemory.id)).where(
                PatternMemory.status == "ACTIVE"
            )
            result = await session.execute(stmt)
            return result.scalar_one() or 0
