"""
情绪曲线自进化 Agent Loop —— SentimentAgent 四阶段业务编排

本模块为 Sentiment_Agent 的核心实现：Learn / Predict / Validate / Evolve 四阶段
闭环。纯业务逻辑与 LLM/DB I/O 分离——可测的纯函数（验证判定、门控、淘汰选择、
输入组装）已独立至 `services/agent_logic.py`。

设计约束：
- 无静默降级（规则 3）：所有异常记录日志，不吞错误、不伪造数据。
- 独立 async_session_factory() 会话（沿用项目现有模式），LLM 调用在事务外完成。
- Validate 阶段：本阶段不做任何 LLM 调用。

对应 spec：.kiro/specs/sentiment-agent-loop/design.md「Components and Interfaces §2. SentimentAgent」
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import (
    AgentPrediction,
    PatternChangeLog,
    PatternMemory,
    SentimentWindow,
)
from .agent_logic import (
    PatternRow,
    PatternStat,
    TradeGateContext,
    WindowRow,
    compress_windows_for_deep_learn,
    compute_is_correct,
    compute_pattern_fingerprint,
    detect_duplicate_pattern,
    evaluate_trade_gate,
    is_prediction_stale,
    plan_active_patterns,
    plan_learn_windows,
    recompute_win_rate,
    select_retire_candidates,
    should_trade,
)
from .alerting import AlertService
from .llm_service import LLMService
from .llm_validator import (
    validate_evolve_output,
    validate_learn_output,
    validate_predict_output,
)
from .prediction_trading import BinancePredictionTrader
from .risk_control import RiskController

if TYPE_CHECKING:
    from ..models.schemas import ChangeType


class SentimentAgent:
    """
    情绪曲线自进化 Agent 四阶段编排器

    持有 LLMService（结构化 LLM 调用通道）与 BinancePredictionTrader（交易执行通道），
    由 AgentScheduler 按事件驱动依次调用各阶段方法。

    当前实现：
    - validate()：完整实现（任务 6.1）
    - learn()：完整实现（任务 6.7）
    - apply_pattern_change()：通用模式变更持久化辅助（任务 6.7）
    - predict()：完整实现（任务 6.3）
    - evolve()：完整实现（任务 6.9）
    """

    def __init__(
        self,
        llm: LLMService,
        trader: BinancePredictionTrader,
        risk_controller: RiskController | None = None,
        alert_service: AlertService | None = None,
    ) -> None:
        """
        初始化 SentimentAgent。

        Args:
            llm: 结构化 LLM 调用服务（Instructor Tool Calling 通道）
            trader: Binance 预测市场交易执行服务
            risk_controller: 风控控制器（可选，None 时自动创建默认实例）
            alert_service: 告警服务（可选，Fix #19：交易门控查询其
                trading_blocked 熔断标志。由 AgentScheduler 在创建后注入）
        """
        self._llm = llm
        self._trader = trader
        self._risk_controller = risk_controller or RiskController()
        self._alert_service = alert_service
        # 深度分析并发锁：防止多个 deep_learn/commit 同时执行
        self._deep_learn_lock = asyncio.Lock()

    # ======================================================================
    # Validate 阶段（Req 4.1 / 4.2 / 4.3 / 4.4）
    # ======================================================================

    async def validate(self, window: SentimentWindow) -> list[int]:
        """
        验证阶段：对该窗口关联的未验证 AgentPrediction 回填验证结果。

        流程：
        1. 查询该窗口对应的所有未验证 AgentPrediction（is_correct is None 且
           sentiment_window_id == window.id）
        2. 对每条预测调用 compute_is_correct(predicted_direction, window.outcome)
           回填 is_correct / actual_outcome(=window.outcome) / actual_return(=window.actual_return)
           / validated_at
        3. 若 matched_pattern_id 非空：查对应 PatternMemory，sample_count += 1，
           若 is_correct 则 correct_count += 1，用 recompute_win_rate 重算 win_rate
        4. 若 matched_pattern_id 为空（冷启动/无匹配）：仅回填 is_correct，不更新任何
           模式统计（决策 6：验证边界）

        Args:
            window: 已归档的 SentimentWindow 实例（含 id / outcome / actual_return）

        Returns:
            本次验证到的 AgentPrediction.id 列表（供调度器计数器累加）

        Raises:
            异常不吞——记录日志后向上抛出，由调度器按「无静默降级」策略处置。
        """
        validated_ids: list[int] = []

        async with async_session_factory() as session:
            # 1. 查询该窗口时间范围内的未验证预测
            # Bug fix: Predict 时窗口可能尚未归档，sentiment_window_id 可能为 None，
            # 改为时间范围匹配以确保 Validate 能找到所有关联预测
            window_start_dt = datetime.fromtimestamp(window.start_time / 1000, tz=timezone.utc)
            window_end_dt = datetime.fromtimestamp(window.end_time / 1000, tz=timezone.utc)

            stmt = (
                select(AgentPrediction)
                .where(
                    AgentPrediction.is_correct.is_(None),
                    # Fix #9: 增加 sentiment_window_id IS NULL 条件，
                    # 防止已被其他窗口验证的预测被重复匹配（双重验证保护）
                    AgentPrediction.sentiment_window_id.is_(None),
                    AgentPrediction.prediction_time >= window_start_dt,
                    AgentPrediction.prediction_time <= window_end_dt,
                )
            )
            result = await session.execute(stmt)
            predictions = result.scalars().all()

            if not predictions:
                logger.debug(
                    "Validate: 窗口 id={} 无待验证预测，跳过", window.id
                )
                return validated_ids

            logger.info(
                "Validate: 窗口 id={} outcome={} | 待验证预测 {} 条",
                window.id,
                window.outcome,
                len(predictions),
            )

            now = datetime.now(tz=timezone.utc)

            for pred in predictions:
                # 2. 调用纯函数计算 is_correct
                is_correct = compute_is_correct(
                    pred.predicted_direction, window.outcome
                )

                # 回填验证结果
                pred.is_correct = is_correct
                pred.actual_outcome = window.outcome
                pred.actual_return = window.actual_return
                pred.validated_at = now

                # 回填 sentiment_window_id（Predict 时窗口可能未归档导致为 None）
                if pred.sentiment_window_id is None:
                    pred.sentiment_window_id = window.id

                # 3. 若匹配了模式 → 更新模式统计
                if pred.matched_pattern_id is not None:
                    pattern_stmt = select(PatternMemory).where(
                        PatternMemory.id == pred.matched_pattern_id
                    )
                    pattern_result = await session.execute(pattern_stmt)
                    pattern = pattern_result.scalar_one_or_none()

                    if pattern is not None:
                        pattern.sample_count += 1
                        if is_correct:
                            pattern.correct_count += 1
                        pattern.win_rate = recompute_win_rate(
                            pattern.correct_count, pattern.sample_count
                        )
                        logger.debug(
                            "Validate: 模式 id={} '{}' 统计更新 | sample={} correct={} win_rate={:.4f}",
                            pattern.id,
                            pattern.pattern_name,
                            pattern.sample_count,
                            pattern.correct_count,
                            pattern.win_rate,
                        )
                    else:
                        logger.warning(
                            "Validate: 预测 id={} 引用的模式 id={} 不存在，跳过模式统计更新",
                            pred.id,
                            pred.matched_pattern_id,
                        )
                # 4. 无匹配模式（冷启动/无匹配）：仅回填 is_correct，不更新统计（决策 6）

                validated_ids.append(pred.id)
                logger.debug(
                    "Validate: 预测 id={} direction={} | is_correct={} | matched_pattern_id={}",
                    pred.id,
                    pred.predicted_direction,
                    is_correct,
                    pred.matched_pattern_id,
                )

            # 提交所有变更（预测回填 + 模式统计）
            await session.commit()

        logger.info(
            "Validate: 窗口 id={} 完成 | 验证 {} 条预测",
            window.id,
            len(validated_ids),
        )
        return validated_ids

    # ======================================================================
    # 共享辅助：模式变更持久化（Req 1.4 / 2.5 / 2.6 / 5.5 / 5.6 / 5.7 / 8.2）
    # ======================================================================

    @staticmethod
    def _pattern_to_snapshot(pattern: PatternMemory) -> dict:
        """
        将 PatternMemory ORM 实例序列化为快照 dict（用于 change_log 的 before/after）。

        仅含业务字段，不含 SQLAlchemy 内部属性。
        """
        return {
            "id": pattern.id,
            "pattern_name": pattern.pattern_name,
            "description": pattern.description,
            "curve_features": pattern.curve_features,
            "conditions": pattern.conditions,
            "predicted_direction": pattern.predicted_direction,
            "win_rate": pattern.win_rate,
            "sample_count": pattern.sample_count,
            "correct_count": pattern.correct_count,
            "confidence_score": pattern.confidence_score,
            "status": pattern.status,
        }

    async def apply_pattern_change(
        self,
        session: AsyncSession,
        operation: ChangeType,
        pattern_data: dict,
        phase: str,
        evolve_phase_id: str | None = None,
    ) -> PatternMemory:
        """
        在同一事务内执行模式变更并生成恰好一条 pattern_change_log。

        由 learn() / evolve() 负责 session 生命周期（事务的 begin/commit/rollback），
        本方法只做写操作不管事务控制。

        Args:
            session: 由调用方提供的异步会话（事务已由调用方管理）
            operation: 变更类型 "CREATE" | "UPDATE" | "RETIRE"
            pattern_data: 变更数据，含义随 operation 变化：
                - CREATE: 必须包含 pattern_name/description/curve_features/conditions/
                          predicted_direction/confidence_score/change_reason
                - UPDATE: 必须包含 target_pattern_id 及待更新字段 + change_reason
                - RETIRE: 必须包含 target_pattern_id + change_reason
            phase: 触发阶段 "LEARN" | "EVOLVE"
            evolve_phase_id: Evolve 执行 ID（LEARN 触发时为 None）

        Returns:
            被操作的 PatternMemory 实例

        Raises:
            ValueError: operation 非法或必要字段缺失
        """
        change_reason = pattern_data.get("change_reason", "")

        if operation == "CREATE":
            # 新建模式：状态 ACTIVE，统计初始 0
            new_pattern = PatternMemory(
                pattern_name=pattern_data["pattern_name"],
                description=pattern_data["description"],
                curve_features=pattern_data.get("curve_features", {}),
                conditions=pattern_data.get("conditions", {}),
                predicted_direction=pattern_data["predicted_direction"],
                confidence_score=pattern_data.get("confidence_score", 0.5),
                status="ACTIVE",
                win_rate=0.0,
                sample_count=0,
                correct_count=0,
            )
            session.add(new_pattern)
            # flush 以获取自增 id（同一事务内）
            await session.flush()

            after_snapshot = self._pattern_to_snapshot(new_pattern)

            # 写变更日志
            log_entry = PatternChangeLog(
                pattern_id=new_pattern.id,
                change_type="CREATE",
                phase=phase,
                before_snapshot=None,
                after_snapshot=after_snapshot,
                change_reason=change_reason,
                evolve_phase_id=evolve_phase_id,
            )
            session.add(log_entry)

            logger.info(
                "apply_pattern_change: CREATE 模式 id={} name='{}' | phase={}",
                new_pattern.id,
                new_pattern.pattern_name,
                phase,
            )
            return new_pattern

        elif operation == "UPDATE":
            target_id = pattern_data.get("target_pattern_id")
            if target_id is None:
                raise ValueError("UPDATE 操作必须提供 target_pattern_id")

            # 查找已有模式
            stmt = select(PatternMemory).where(PatternMemory.id == target_id)
            result = await session.execute(stmt)
            pattern = result.scalar_one_or_none()
            if pattern is None:
                raise ValueError(f"UPDATE 目标模式 id={target_id} 不存在")

            # 记录变更前快照
            before_snapshot = self._pattern_to_snapshot(pattern)

            # 更新指定字段（仅更新 LLM 可写的业务字段）
            updatable_fields = (
                "pattern_name",
                "description",
                "curve_features",
                "conditions",
                "predicted_direction",
                "confidence_score",
            )
            for field in updatable_fields:
                if field in pattern_data:
                    setattr(pattern, field, pattern_data[field])

            # 记录变更后快照
            after_snapshot = self._pattern_to_snapshot(pattern)

            # 写变更日志
            log_entry = PatternChangeLog(
                pattern_id=pattern.id,
                change_type="UPDATE",
                phase=phase,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                change_reason=change_reason,
                evolve_phase_id=evolve_phase_id,
            )
            session.add(log_entry)

            logger.info(
                "apply_pattern_change: UPDATE 模式 id={} name='{}' | phase={}",
                pattern.id,
                pattern.pattern_name,
                phase,
            )
            return pattern

        elif operation == "RETIRE":
            target_id = pattern_data.get("target_pattern_id")
            if target_id is None:
                raise ValueError("RETIRE 操作必须提供 target_pattern_id")

            # 查找已有模式
            stmt = select(PatternMemory).where(PatternMemory.id == target_id)
            result = await session.execute(stmt)
            pattern = result.scalar_one_or_none()
            if pattern is None:
                raise ValueError(f"RETIRE 目标模式 id={target_id} 不存在")

            # 记录变更前快照
            before_snapshot = self._pattern_to_snapshot(pattern)

            # 置为 RETIRED
            pattern.status = "RETIRED"

            # 记录变更后快照（status 已为 RETIRED）
            after_snapshot = self._pattern_to_snapshot(pattern)

            # 写变更日志
            log_entry = PatternChangeLog(
                pattern_id=pattern.id,
                change_type="RETIRE",
                phase=phase,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                change_reason=change_reason,
                evolve_phase_id=evolve_phase_id,
            )
            session.add(log_entry)

            logger.info(
                "apply_pattern_change: RETIRE 模式 id={} name='{}' | phase={}",
                pattern.id,
                pattern.pattern_name,
                phase,
            )
            return pattern

        else:
            raise ValueError(f"不支持的操作类型: {operation}")

    # ======================================================================
    # Learn 阶段（Req 2.1 / 2.2 / 2.3 / 2.4 / 2.5 / 2.6 / 7.4 / 8.2）
    # ======================================================================

    async def learn(self) -> None:
        """
        学习阶段：分析最近情绪窗口，发现或更新情绪曲线模式。

        流程（design.md §2 Learn 段）：
        1. plan_learn_windows：读最近 N 个 outcome 非空的 SentimentWindow → 纯函数筛选 → 序列化
        2. plan_active_patterns：读全部 ACTIVE 模式 → 纯函数筛选 → 序列化
        3. 调用 self._llm.agent_learn(windows_dicts, patterns_dicts, timeout) → LearnOutput
        4. 对 LearnOutput.discoveries 逐条执行 apply_pattern_change（CREATE 或 UPDATE）
        5. LLM 失败/超时/重试耗尽 → 记录完整错误日志 + 跳过本次 Learn（不写入任何部分模式）

        设计约束：
        - LLM 调用在事务开启前完成（避免长事务占用连接池）
        - 使用独立 async_session_factory() 会话
        """
        # [DEPRECATED] 双模式架构下 Learn 仅在 auto 模式由 scheduler 调用
        # 推荐使用 POST /api/sentiment/agent/deep-learn 触发手动深度分析
        logger.debug(
            "SentimentAgent.learn() 调用 [deprecated] | "
            "双模式架构下建议使用 deep_learn() API"
        )
        # ========== Step 1 & 2：读取数据（只读会话）==========
        windows_dicts: list[dict] = []
        patterns_dicts: list[dict] = []

        async with async_session_factory() as session:
            # 查询最近的 SentimentWindow（outcome 非空，按 start_time 降序）
            stmt = (
                select(SentimentWindow)
                .where(SentimentWindow.outcome.isnot(None))
                .order_by(SentimentWindow.start_time.desc())
                .limit(settings.agent_learn_window_count * 2)  # 多取一些，纯函数再精确筛选
            )
            result = await session.execute(stmt)
            raw_windows = result.scalars().all()

            # 构建 WindowRow 列表 → 纯函数筛选
            window_rows = [
                WindowRow(id=w.id, start_time=w.start_time, outcome=w.outcome)
                for w in raw_windows
            ]
            selected_window_rows = plan_learn_windows(
                window_rows, limit=settings.agent_learn_window_count
            )
            # 用筛选后的 id 集合对应回原始 ORM 行，取完整字段序列化
            selected_ids = {wr.id for wr in selected_window_rows}
            for w in raw_windows:
                if w.id in selected_ids:
                    windows_dicts.append({
                        "id": w.id,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "curve_up_pct": w.curve_up_pct,
                        "curve_down_pct": w.curve_down_pct,
                        "outcome": w.outcome,
                        "actual_return": w.actual_return,
                        "sample_count": w.sample_count,
                    })

            # 查询全部模式（含各种状态，纯函数筛选 ACTIVE）
            pattern_stmt = select(PatternMemory)
            pattern_result = await session.execute(pattern_stmt)
            raw_patterns = pattern_result.scalars().all()

            # 构建 PatternRow 列表 → 纯函数筛选 ACTIVE
            pattern_rows = [
                PatternRow(
                    id=p.id,
                    status=p.status,
                    pattern_name=p.pattern_name,
                    predicted_direction=p.predicted_direction,
                )
                for p in raw_patterns
            ]
            active_rows = plan_active_patterns(pattern_rows)
            active_ids = {pr.id for pr in active_rows}
            for p in raw_patterns:
                if p.id in active_ids:
                    patterns_dicts.append({
                        "id": p.id,
                        "pattern_name": p.pattern_name,
                        "description": p.description,
                        "curve_features": p.curve_features,
                        "conditions": p.conditions,
                        "predicted_direction": p.predicted_direction,
                        "win_rate": p.win_rate,
                        "sample_count": p.sample_count,
                        "confidence_score": p.confidence_score,
                    })

        logger.info(
            "Learn: 数据准备完成 | 合格窗口={} | ACTIVE 模式={}",
            len(windows_dicts),
            len(patterns_dicts),
        )

        # ========== Step 3：LLM 调用（在事务外完成，避免长事务）==========
        try:
            learn_output = await self._llm.agent_learn(
                windows=windows_dicts,
                active_patterns=patterns_dicts,
                timeout=settings.agent_llm_timeouts["LEARN"],
            )
        except Exception as exc:
            # Req 7.4：LLM 失败/超时/重试耗尽 → 记录完整错误日志 + 跳过本次 Learn
            logger.error(
                "Learn: LLM 调用失败，跳过本次 Learn（不写入任何模式）| "
                "error_type={} | error={}",
                type(exc).__name__,
                str(exc),
            )
            return

        if not learn_output.discoveries:
            logger.info("Learn: LLM 未返回任何模式发现，本次 Learn 结束")
            return

        # ========== Step 3.5：LLM 输出语义验证（Plan 步骤 5）==========
        if settings.agent_llm_validation_enabled:
            active_ids = {p["id"] for p in patterns_dicts}
            hard_failures, soft_warnings = validate_learn_output(
                learn_output, active_ids
            )
            if soft_warnings:
                for w in soft_warnings:
                    logger.warning("Learn 语义验证 SOFT: {}", w)
            if hard_failures:
                for f in hard_failures:
                    logger.error("Learn 语义验证 HARD: {}", f)
                # 过滤掉 HARD_FAIL 的 discoveries
                original_count = len(learn_output.discoveries)
                learn_output.discoveries = [
                    d for i, d in enumerate(learn_output.discoveries)
                    if not any(
                        f"discoveries[{i}]" in f for f in hard_failures
                    )
                ]
                logger.info(
                    "Learn: 语义验证过滤 {}/{} 条发现",
                    original_count - len(learn_output.discoveries),
                    original_count,
                )
                if not learn_output.discoveries:
                    logger.info("Learn: 所有发现被过滤，本次 Learn 结束")
                    return

        logger.info(
            "Learn: LLM 返回 {} 条模式发现 | reasoning={}...",
            len(learn_output.discoveries),
            learn_output.reasoning[:100],
        )

        # ========== Step 4：写入模式变更（独立事务，每条操作用 savepoint 隔离）==========
        # 记录本批次已创建的模式指纹，用于批内去重
        batch_created_patterns: list[dict] = []

        async with async_session_factory() as session:
            for idx, discovery in enumerate(learn_output.discoveries, 1):
                try:
                    # Fix #10: 在去重检查前初始化标志，避免后续重置覆盖
                    is_dedup_downgrade = False
                    # Plan 步骤 15：CREATE 前去重检查
                    if (
                        discovery.operation == "CREATE"
                        and settings.agent_dedup_enabled
                    ):
                        # 同时检查已有模式和本批次已创建的模式
                        all_patterns_for_dedup = patterns_dicts + batch_created_patterns
                        existing_id = detect_duplicate_pattern(
                            discovery.curve_features,
                            discovery.predicted_direction,
                            all_patterns_for_dedup,
                        )
                        if existing_id is not None:
                            logger.warning(
                                "Learn: 疑似重复模式 '{}'，与 id={} 指纹一致 | "
                                "auto_downgrade={}",
                                discovery.pattern_name,
                                existing_id,
                                settings.agent_dedup_auto_downgrade,
                            )
                            if settings.agent_dedup_auto_downgrade:
                                # 将 CREATE 降级为 UPDATE，仅更新 curve_features 和 conditions
                                discovery.operation = "UPDATE"
                                discovery.target_pattern_id = existing_id
                                discovery.change_reason = (
                                    f"去重降级: 与 id={existing_id} 指纹一致，"
                                    f"原 CREATE 转为 UPDATE"
                                )
                                is_dedup_downgrade = True
                            else:
                                # 未启用 auto_downgrade，仅记录警告，跳过该条
                                continue

                    async with session.begin_nested():  # savepoint：失败仅回滚此子事务
                        if is_dedup_downgrade:
                            # 去重降级：仅更新 curve_features 和 conditions，保留原 name/description
                            pattern_data = {
                                "curve_features": discovery.curve_features,
                                "conditions": discovery.conditions,
                                "change_reason": discovery.change_reason,
                                "target_pattern_id": discovery.target_pattern_id,
                            }
                        else:
                            # 正常操作：全字段
                            pattern_data = {
                                "pattern_name": discovery.pattern_name,
                                "description": discovery.description,
                                "curve_features": discovery.curve_features,
                                "conditions": discovery.conditions,
                                "predicted_direction": discovery.predicted_direction,
                                "confidence_score": discovery.confidence_score,
                                "change_reason": discovery.change_reason,
                            }
                            if discovery.operation == "UPDATE":
                                pattern_data["target_pattern_id"] = discovery.target_pattern_id

                        new_pattern = await self.apply_pattern_change(
                            session=session,
                            operation=discovery.operation,
                            pattern_data=pattern_data,
                            phase="LEARN",
                            evolve_phase_id=None,  # Learn 触发，无 evolve_phase_id
                        )

                        # 记录新创建的模式用于批内去重
                        if discovery.operation == "CREATE" and new_pattern is not None:
                            batch_created_patterns.append({
                                "id": new_pattern.id,
                                "curve_features": new_pattern.curve_features,
                                "predicted_direction": new_pattern.predicted_direction,
                            })
                except Exception as exc:
                    # savepoint 已自动回滚，session 仍可继续
                    logger.error(
                        "Learn: 第 {}/{} 条发现写入失败 | operation={} pattern_name='{}' | "
                        "error_type={} | error={}",
                        idx,
                        len(learn_output.discoveries),
                        discovery.operation,
                        discovery.pattern_name,
                        type(exc).__name__,
                        str(exc),
                    )
                    continue

            # 提交所有成功的变更
            await session.commit()

        logger.info("Learn: 阶段完成 | 成功处理模式变更")

    # ======================================================================
    # 深度分析（手动触发，双模式架构）
    # ======================================================================

    async def deep_learn(
        self,
        max_windows: int | None = None,
    ) -> dict:
        """
        深度模式发现（手动触发）：分析全量历史窗口，返回发现结果供用户预览。

        与 learn() 的区别：
        - 读取全量窗口（不限于最近 50）
        - 全量原始曲线直接输入 LLM（不压缩，保留完整曲线形态）
        - LLM max_tokens 更大（16384）
        - **不直接写入 DB**，返回 discoveries 列表

        Args:
            max_windows: 最大读取窗口数，默认使用 settings.agent_deep_learn_max_windows

        Returns:
            dict 包含：
            - reasoning: LLM 分析推理过程
            - discoveries: 序列化的发现列表
            用户可通过 commit_deep_learn() 确认后写入 DB
        """
        # 并发保护：同一时刻只允许一个 deep_learn 执行
        if self._deep_learn_lock.locked():
            raise RuntimeError("已有深度分析任务正在执行，请稍后重试")

        async with self._deep_learn_lock:
            if max_windows is None:
                max_windows = settings.agent_deep_learn_max_windows

            logger.info(
                "Deep Learn: 开始全量历史分析 | max_windows={}",
                max_windows,
            )

            # Step 1: 从 DB 读取全量窗口（全量原始数据，不压缩）
            async with async_session_factory() as session:
                stmt = (
                    select(SentimentWindow)
                    .where(SentimentWindow.outcome.isnot(None))
                    .order_by(SentimentWindow.start_time.desc())
                    .limit(max_windows)
                )
                result = await session.execute(stmt)
                raw_windows = result.scalars().all()

                # 序列化为 dict（保留完整曲线数据）
                windows_dicts = [
                    {
                        "id": w.id,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "curve_up_pct": w.curve_up_pct,
                        "curve_down_pct": w.curve_down_pct,
                        "outcome": w.outcome,
                        "actual_return": w.actual_return,
                        "sample_count": w.sample_count,
                    }
                    for w in raw_windows
                ]

                # 查询 ACTIVE 模式
                pattern_stmt = select(PatternMemory).where(PatternMemory.status == "ACTIVE")
                pattern_result = await session.execute(pattern_stmt)
                active_patterns = [
                    {
                        "id": p.id,
                        "pattern_name": p.pattern_name,
                        "description": p.description,
                        "curve_features": p.curve_features,
                        "conditions": p.conditions,
                        "predicted_direction": p.predicted_direction,
                        "win_rate": p.win_rate,
                        "sample_count": p.sample_count,
                        "confidence_score": p.confidence_score,
                    }
                    for p in pattern_result.scalars().all()
                ]

            logger.info(
                "Deep Learn: 数据准备完成 | 窗口数={} | ACTIVE 模式={}",
                len(windows_dicts),
                len(active_patterns),
            )

            if not windows_dicts:
                logger.warning("Deep Learn: 无可用历史窗口，跳过分析")
                return {"reasoning": "", "discoveries": []}

            # Step 2: LLM 深度分析（全量原始窗口直接输入，不压缩）
            timeout = settings.agent_llm_timeouts.get("LEARN", 100.0)
            try:
                learn_output = await self._llm.agent_deep_learn(
                    windows=windows_dicts,
                    active_patterns=active_patterns,
                    timeout=timeout,
                )
            except Exception as exc:
                logger.error(
                    "Deep Learn: LLM 调用失败 | error_type={} | error={}",
                    type(exc).__name__,
                    str(exc),
                )
                raise

            logger.info(
                "Deep Learn: LLM 返回 {} 条发现 | reasoning={}...",
                len(learn_output.discoveries),
                learn_output.reasoning[:200] if learn_output.reasoning else "",
            )

            # Step 3: 序列化返回（不写入 DB）
            discoveries_serialized = []
            for d in learn_output.discoveries:
                discoveries_serialized.append({
                    "operation": d.operation,
                    "target_pattern_id": d.target_pattern_id,
                    "pattern_name": d.pattern_name,
                    "description": d.description,
                    "curve_features": d.curve_features,
                    "conditions": d.conditions,
                    "predicted_direction": d.predicted_direction,
                    "confidence_score": d.confidence_score,
                    "change_reason": d.change_reason,
                })

            return {
                "reasoning": learn_output.reasoning or "",
                "discoveries": discoveries_serialized,
            }

    async def deep_learn_stream(
        self,
        max_windows: int | None = None,
    ):
        """深度模式发现（流式版）：逐步产出事件供前端 SSE 实时展示（打字机效果）。

        流程与 deep_learn 一致（读全量窗口 → LLM 分析 → 序列化 discoveries，不写 DB），
        区别在于全程以事件流形式产出，且 LLM 调用走 create_partial 流式 + 空闲超时。

        产出事件（dict，交由 main 转成 SSE data 帧）：
        - {"type": "step",      "message": str}        阶段性进度（读取窗口/开始调用等）
        - {"type": "reasoning", "delta": str}          reasoning 增量（打字机）
        - {"type": "progress",  "discoveries": int}    已解析发现条数
        - {"type": "done",      "reasoning": str, "discoveries": list}  最终结果（供勾选提交）
        - {"type": "error",     "message": str}        并发冲突/空闲超时/流式异常

        并发保护与 deep_learn 共用 self._deep_learn_lock（二者互斥）。
        """
        if self._deep_learn_lock.locked():
            yield {"type": "error", "message": "已有深度分析任务正在执行，请稍后重试"}
            return

        async with self._deep_learn_lock:
            if max_windows is None:
                max_windows = settings.agent_deep_learn_max_windows

            logger.info("Deep Learn(stream): 开始全量历史分析 | max_windows={}", max_windows)
            yield {"type": "step", "message": f"开始全量历史分析（max_windows={max_windows}）"}

            # Step 1: 读取全量窗口 + ACTIVE 模式（与 deep_learn 相同）
            async with async_session_factory() as session:
                stmt = (
                    select(SentimentWindow)
                    .where(SentimentWindow.outcome.isnot(None))
                    .order_by(SentimentWindow.start_time.desc())
                    .limit(max_windows)
                )
                result = await session.execute(stmt)
                raw_windows = result.scalars().all()
                windows_dicts = [
                    {
                        "id": w.id,
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "curve_up_pct": w.curve_up_pct,
                        "curve_down_pct": w.curve_down_pct,
                        "outcome": w.outcome,
                        "actual_return": w.actual_return,
                        "sample_count": w.sample_count,
                    }
                    for w in raw_windows
                ]

                pattern_stmt = select(PatternMemory).where(PatternMemory.status == "ACTIVE")
                pattern_result = await session.execute(pattern_stmt)
                active_patterns = [
                    {
                        "id": p.id,
                        "pattern_name": p.pattern_name,
                        "description": p.description,
                        "curve_features": p.curve_features,
                        "conditions": p.conditions,
                        "predicted_direction": p.predicted_direction,
                        "win_rate": p.win_rate,
                        "sample_count": p.sample_count,
                        "confidence_score": p.confidence_score,
                    }
                    for p in pattern_result.scalars().all()
                ]

            logger.info(
                "Deep Learn(stream): 数据准备完成 | 窗口数={} | ACTIVE 模式={}",
                len(windows_dicts),
                len(active_patterns),
            )
            yield {
                "type": "step",
                "message": f"数据准备完成：窗口 {len(windows_dicts)} 个 · ACTIVE 模式 {len(active_patterns)} 条",
            }

            if not windows_dicts:
                logger.warning("Deep Learn(stream): 无可用历史窗口，跳过分析")
                yield {"type": "done", "reasoning": "", "discoveries": []}
                return

            # Step 2: 流式 LLM 深度分析（空闲超时，全程转发事件）
            yield {"type": "step", "message": f"调用 LLM（model={settings.decision_model}）分析中…"}
            final = None
            async for ev in self._llm.agent_deep_learn_stream(
                windows=windows_dicts,
                active_patterns=active_patterns,
                idle_timeout=settings.agent_deep_learn_idle_timeout,
            ):
                if ev.get("type") == "done":
                    final = ev.get("result")
                elif ev.get("type") == "error":
                    yield ev
                    return
                else:
                    # reasoning / progress 直接透传给前端
                    yield ev

            if final is None:
                yield {"type": "error", "message": "LLM 未返回任何内容"}
                return

            # Step 3: 序列化 discoveries（防御式：跳过分片未填满的项）
            discoveries_serialized = []
            for d in getattr(final, "discoveries", None) or []:
                op = getattr(d, "operation", None)
                name = getattr(d, "pattern_name", None)
                direction = getattr(d, "predicted_direction", None)
                if not op or not name or not direction:
                    continue
                discoveries_serialized.append({
                    "operation": op,
                    "target_pattern_id": getattr(d, "target_pattern_id", None),
                    "pattern_name": name,
                    "description": getattr(d, "description", "") or "",
                    "curve_features": getattr(d, "curve_features", None) or {},
                    "conditions": getattr(d, "conditions", None) or {},
                    "predicted_direction": direction,
                    "confidence_score": getattr(d, "confidence_score", None) or 0.0,
                    "change_reason": getattr(d, "change_reason", "") or "",
                })

            logger.info(
                "Deep Learn(stream): 完成 | 有效发现={}",
                len(discoveries_serialized),
            )
            yield {
                "type": "done",
                "reasoning": getattr(final, "reasoning", None) or "",
                "discoveries": discoveries_serialized,
            }

    async def commit_deep_learn(
        self,
        discoveries: list[dict],
    ) -> int:
        """
        将用户确认的 discoveries 写入 pattern_memory。

        复用 apply_pattern_change + savepoint 隔离逻辑，
        与 learn() Step 4 相同的写入机制，并包含去重检查。

        Args:
            discoveries: 用户确认后的发现列表（来自 deep_learn 返回值）

        Returns:
            成功写入的模式数量
        """
        # 并发保护
        if self._deep_learn_lock.locked():
            raise RuntimeError("已有深度分析任务正在执行，请稍后重试")

        async with self._deep_learn_lock:
            if not discoveries:
                logger.info("Commit Deep Learn: 无 discoveries 待写入")
                return 0

            logger.info(
                "Commit Deep Learn: 开始写入 {} 条发现",
                len(discoveries),
            )

            # 读取现有 ACTIVE 模式用于去重检查
            existing_patterns: list[dict] = []
            async with async_session_factory() as read_session:
                pattern_stmt = select(PatternMemory).where(PatternMemory.status == "ACTIVE")
                pattern_result = await read_session.execute(pattern_stmt)
                existing_patterns = [
                    {
                        "id": p.id,
                        "curve_features": p.curve_features,
                        "predicted_direction": p.predicted_direction,
                    }
                    for p in pattern_result.scalars().all()
                ]

            # 记录本批次已创建的模式指纹，用于批内去重
            batch_created_patterns: list[dict] = []
            written_count = 0

            async with async_session_factory() as session:
                for idx, d in enumerate(discoveries, 1):
                    try:
                        operation = d.get("operation", "CREATE")

                        # CREATE 前去重检查（与 learn() 相同逻辑）
                        if operation == "CREATE" and settings.agent_dedup_enabled:
                            all_patterns_for_dedup = existing_patterns + batch_created_patterns
                            existing_id = detect_duplicate_pattern(
                                d.get("curve_features", {}),
                                d.get("predicted_direction", ""),
                                all_patterns_for_dedup,
                            )
                            if existing_id is not None:
                                logger.warning(
                                    "Commit Deep Learn: 疑似重复模式 '{}'，与 id={} 指纹一致 | "
                                    "auto_downgrade={}",
                                    d.get("pattern_name"),
                                    existing_id,
                                    settings.agent_dedup_auto_downgrade,
                                )
                                if settings.agent_dedup_auto_downgrade:
                                    # 将 CREATE 降级为 UPDATE
                                    operation = "UPDATE"
                                    d["operation"] = "UPDATE"
                                    d["target_pattern_id"] = existing_id
                                    d["change_reason"] = (
                                        f"去重降级: 与 id={existing_id} 指纹一致，"
                                        f"原 CREATE 转为 UPDATE"
                                    )
                                else:
                                    # 未启用 auto_downgrade，跳过该条
                                    logger.info(
                                        "Commit Deep Learn: 跳过重复模式 '{}' ",
                                        d.get("pattern_name"),
                                    )
                                    continue

                        async with session.begin_nested():
                            if operation == "UPDATE":
                                # UPDATE 操作：使用指定字段
                                pattern_data = {
                                    "pattern_name": d.get("pattern_name"),
                                    "description": d.get("description"),
                                    "curve_features": d.get("curve_features"),
                                    "conditions": d.get("conditions"),
                                    "predicted_direction": d.get("predicted_direction"),
                                    "confidence_score": d.get("confidence_score", 0.5),
                                    "change_reason": d.get("change_reason", ""),
                                    "target_pattern_id": d.get("target_pattern_id"),
                                }
                            else:
                                # CREATE 操作
                                pattern_data = {
                                    "pattern_name": d.get("pattern_name"),
                                    "description": d.get("description"),
                                    "curve_features": d.get("curve_features"),
                                    "conditions": d.get("conditions"),
                                    "predicted_direction": d.get("predicted_direction"),
                                    "confidence_score": d.get("confidence_score", 0.5),
                                    "change_reason": d.get("change_reason", ""),
                                }

                            new_pattern = await self.apply_pattern_change(
                                session=session,
                                operation=operation,
                                pattern_data=pattern_data,
                                phase="DEEP_LEARN",  # 使用专用阶段标识
                                evolve_phase_id=None,
                            )

                            # 记录新创建的模式用于批内去重
                            if operation == "CREATE" and new_pattern is not None:
                                batch_created_patterns.append({
                                    "id": new_pattern.id,
                                    "curve_features": new_pattern.curve_features,
                                    "predicted_direction": new_pattern.predicted_direction,
                                })

                            written_count += 1
                            logger.info(
                                "Commit Deep Learn: {}/{} 写入成功 | {} '{}'",
                                idx,
                                len(discoveries),
                                operation,
                                d.get("pattern_name"),
                            )
                    except Exception as exc:
                        logger.error(
                            "Commit Deep Learn: {}/{} 写入失败 | {} '{}' | error={}",
                            idx,
                            len(discoveries),
                            d.get("operation"),
                            d.get("pattern_name"),
                            str(exc),
                        )
                        continue

                await session.commit()

            logger.info(
                "Commit Deep Learn: 完成 | 成功写入 {}/{} 条",
                written_count,
                len(discoveries),
            )
            return written_count

    # ======================================================================
    # Predict 阶段（Req 3.2 / 3.3 / 3.4 / 3.5 / 3.6 / 10.1 / 10.2 / 10.3 / 11.1）
    # ======================================================================

    async def predict(
        self, window_end_ms: int, current_curve: list[dict]
    ) -> AgentPrediction | None:
        """
        预测阶段：基于当前窗口实时曲线与 ACTIVE 模式匹配，给出方向预测并执行交易门控。

        流程（design.md §2 Predict 段）：
        1. 读 ACTIVE 模式列表（plan_active_patterns）
        2. 冷启动检查：若 ACTIVE 模式数为 0 → 直接构造 NO_TRADE 预测记录，不调用 LLM（Req 11.1）
        3. 否则：计算 remaining_seconds，调用 llm.agent_predict → PredictOutput
        4. 写 AgentPrediction 记录（prediction_time=now, 映射 PredictOutput 字段）
        5. 交易门控：should_trade(direction, confidence, threshold)
           - 通过：调 trader.execute_trade → 回填 trade_order_id
           - 跳过：写 skip_trade_reason
        6. LLM 失败/超时/重试耗尽 → 落库 direction=NO_TRADE, confidence=0,
           reasoning=f"LLM 调用失败: {error}", skip_trade_reason="LLM 调用失败"（Req 3.6）

        Args:
            window_end_ms: 当前窗口结束时间戳（毫秒），用于计算剩余时间与匹配窗口 ID
            current_curve: 当前窗口已采集的实时曲线数据（[{t, v}, ...]），
                由 Scheduler dispatch 时从 _pm_history 切片传入

        Returns:
            写入的 AgentPrediction 实例；极端异常下返回 None

        设计约束：
        - LLM 调用在事务开启前完成（避免长事务占用连接池）
        - 使用独立 async_session_factory() 会话
        - commit 后才拿 pred.id 去调交易（保证 id 已分配）
        - 无静默降级：所有失败路径明确记录原因（规则 3）
        """
        # ========== Step 1：读取 ACTIVE 模式（只读会话）==========
        patterns_dicts: list[dict] = []

        async with async_session_factory() as session:
            pattern_stmt = select(PatternMemory)
            pattern_result = await session.execute(pattern_stmt)
            raw_patterns = pattern_result.scalars().all()

            # 构建 PatternRow 列表 → 纯函数筛选 ACTIVE
            pattern_rows = [
                PatternRow(
                    id=p.id,
                    status=p.status,
                    pattern_name=p.pattern_name,
                    predicted_direction=p.predicted_direction,
                )
                for p in raw_patterns
            ]
            active_rows = plan_active_patterns(pattern_rows)
            active_ids = {pr.id for pr in active_rows}
            for p in raw_patterns:
                if p.id in active_ids:
                    patterns_dicts.append({
                        "id": p.id,
                        "pattern_name": p.pattern_name,
                        "description": p.description,
                        "curve_features": p.curve_features,
                        "conditions": p.conditions,
                        "predicted_direction": p.predicted_direction,
                        "win_rate": p.win_rate,
                        "sample_count": p.sample_count,
                        "confidence_score": p.confidence_score,
                    })

            # 尝试匹配 sentiment_window_id（窗口可能尚未归档，允许为 None）
            sw_stmt = select(SentimentWindow.id).where(
                SentimentWindow.end_time == window_end_ms
            )
            sw_result = await session.execute(sw_stmt)
            sentiment_window_id = sw_result.scalar_one_or_none()

        active_count = len(patterns_dicts)
        logger.info(
            "Predict: 数据准备完成 | ACTIVE 模式={} | 曲线点数={} | window_end_ms={}",
            active_count,
            len(current_curve),
            window_end_ms,
        )

        # ========== Step 2：冷启动检查（Req 11.1）==========
        if active_count == 0:
            logger.info("Predict: 冷启动——ACTIVE 模式数为 0，直接输出 NO_TRADE（不调用 LLM）")
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning="模式库为空，等待学习积累",
                sentiment_window_id=sentiment_window_id,
            )

        # ========== Step 3：计算剩余时间 + LLM 调用（在事务外完成）==========
        remaining_seconds = max(0, (window_end_ms - int(time.time() * 1000)) // 1000)

        try:
            predict_output = await self._llm.agent_predict(
                current_curve=current_curve,
                active_patterns=patterns_dicts,
                remaining_seconds=remaining_seconds,
                timeout=settings.agent_llm_timeouts["PREDICT"],
            )
        except Exception as exc:
            # Req 3.6 / 7.4：LLM 失败/超时/重试耗尽 → 落库 NO_TRADE + 错误原因
            error_msg = f"LLM 调用失败: {type(exc).__name__}: {exc}"
            logger.error(
                "Predict: LLM 调用失败，落库 NO_TRADE（无静默降级）| error={}",
                error_msg,
            )
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning=error_msg,
                sentiment_window_id=sentiment_window_id,
                skip_trade_reason="LLM 调用失败",
            )

        logger.info(
            "Predict: LLM 返回 | direction={} | confidence={:.4f} | matched_pattern={}",
            predict_output.direction,
            predict_output.confidence,
            predict_output.matched_pattern_name,
        )

        # ========== Step 3.5：LLM 输出语义验证（Plan 步骤 5）==========
        if settings.agent_llm_validation_enabled:
            active_ids = {p["id"] for p in patterns_dicts}
            hard_failures, soft_warnings = validate_predict_output(
                predict_output, active_ids
            )
            if soft_warnings:
                for w in soft_warnings:
                    logger.warning("Predict 语义验证 SOFT: {}", w)
            if hard_failures:
                for f in hard_failures:
                    logger.error("Predict 语义验证 HARD: {}", f)
                # HARD_FAIL 时降级为 NO_TRADE
                logger.warning(
                    "Predict: 语义验证 HARD_FAIL，降级为 NO_TRADE | failures={}",
                    hard_failures,
                )
                return await self._write_prediction_and_trade(
                    predicted_direction="NO_TRADE",
                    matched_pattern_id=None,
                    matched_pattern_name=None,
                    confidence=0.0,
                    entry_timing="SKIP",
                    reasoning=f"语义验证失败: {hard_failures}",
                    sentiment_window_id=sentiment_window_id,
                    skip_trade_reason="LLM 输出语义验证失败",
                )

        # ========== Step 3.6：时间窗口保护（Plan 步骤 10）==========
        remaining_after_llm = max(0, (window_end_ms - int(time.time() * 1000)) // 1000)
        if is_prediction_stale(
            remaining_after_llm,
            min_remaining=settings.agent_prediction_min_remaining_seconds,
        ):
            logger.warning(
                "Predict: 预测已过时 | 剩余 {}s < 阈值 {}s | 降级为 NO_TRADE",
                remaining_after_llm,
                settings.agent_prediction_min_remaining_seconds,
            )
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning=f"预测已过时（剩余 {remaining_after_llm}s）",
                sentiment_window_id=sentiment_window_id,
                skip_trade_reason=f"预测已过时(剩余 {remaining_after_llm}s)",
            )

        # ========== Step 4 & 5 & 6：写入预测 + 交易门控 ==========
        return await self._write_prediction_and_trade(
            predicted_direction=predict_output.direction,
            matched_pattern_id=predict_output.matched_pattern_id,
            matched_pattern_name=predict_output.matched_pattern_name,
            confidence=predict_output.confidence,
            entry_timing=predict_output.entry_timing,
            reasoning=predict_output.reasoning,
            sentiment_window_id=sentiment_window_id,
        )

    async def _write_prediction_and_trade(
        self,
        *,
        predicted_direction: str,
        matched_pattern_id: int | None,
        matched_pattern_name: str | None,
        confidence: float,
        entry_timing: str,
        reasoning: str,
        sentiment_window_id: int | None,
        skip_trade_reason: str | None = None,
    ) -> AgentPrediction | None:
        """
        Predict 阶段共享辅助：写入 AgentPrediction 记录并执行交易门控。

        现在使用 evaluate_trade_gate 进行扩展门控（Plan 步骤 9）。

        单次 session 内完成：flush 获取 id → 交易执行 → 回填 trade_order_id → commit，
        保证预测记录与 trade_order_id 在同一事务内原子提交。

        Args:
            predicted_direction: 预测方向 UP | DOWN | NO_TRADE
            matched_pattern_id: 匹配模式 ID（可空）
            matched_pattern_name: 匹配模式名称（可空）
            confidence: 置信度 0~1
            entry_timing: 入场时机 NOW | WAIT | SKIP
            reasoning: LLM 推理过程 / 冷启动说明 / 错误原因
            sentiment_window_id: 关联的情绪窗口 ID（可空，Validate 时回填）
            skip_trade_reason: 预设的跳过原因（如 LLM 失败场景已知跳过原因）

        Returns:
            写入的 AgentPrediction 实例；极端异常返回 None
        """
        now = datetime.now(tz=timezone.utc)

        # 交易门控（扩展版，Plan 步骤 8/9）
        if settings.agent_risk_control_enabled:
            # 刷新日内风控统计
            await self._risk_controller.refresh_daily_stats()

        # 查询匹配模式的胜率/样本数/状态（如有）
        matched_win_rate: float | None = None
        matched_sample_count: int | None = None
        matched_pattern_status: str | None = None
        if matched_pattern_id is not None:
            async with async_session_factory() as session:
                pat_stmt = select(PatternMemory).where(PatternMemory.id == matched_pattern_id)
                pat_result = await session.execute(pat_stmt)
                pat = pat_result.scalar_one_or_none()
                if pat is not None:
                    matched_win_rate = pat.win_rate
                    matched_sample_count = pat.sample_count
                    matched_pattern_status = pat.status

        # 风控开关决定门控深度：
        # - 开启：扩展 8 级规则链（方向 + 置信度 + 模式证据 + 风控维度）
        # - 关闭：仅基础门控（方向 + 置信度），跳过风控维度
        if settings.agent_risk_control_enabled:
            gate_ctx = TradeGateContext(
                direction=predicted_direction,
                confidence=confidence,
                auto_trade_enabled=settings.agent_auto_trade,
                threshold=settings.agent_trade_confidence_threshold,
                matched_pattern_win_rate=matched_win_rate,
                matched_pattern_sample_count=matched_sample_count,
                recent_loss_streak=self._risk_controller.recent_loss_streak,
                daily_trade_count=self._risk_controller.daily_trade_count,
                daily_pnl=self._risk_controller.daily_pnl,
                alert_blocked=(
                    self._alert_service.trading_blocked
                    if self._alert_service is not None
                    else False
                ),
            )
            do_trade, trade_reason = evaluate_trade_gate(
                gate_ctx,
                min_pattern_samples=settings.agent_min_pattern_samples,
                min_pattern_win_rate=settings.agent_min_pattern_win_rate,
                max_consecutive_losses=settings.agent_max_consecutive_losses,
                max_daily_trades=settings.agent_max_daily_trades,
                max_daily_loss_usdt=settings.agent_max_daily_loss_usdt,
            )
        else:
            # 风控关闭：仅基础门控（方向 + 置信度）
            do_trade, trade_reason = should_trade(
                direction=predicted_direction,
                confidence=confidence,
                threshold=settings.agent_trade_confidence_threshold,
                auto_trade_enabled=settings.agent_auto_trade,
            )
        # 若调用方已指定 skip_trade_reason（如 LLM 失败），优先使用
        final_skip_reason = skip_trade_reason if skip_trade_reason else (
            None if do_trade else trade_reason
        )

        # ---- 单次 session 内原子完成：pred + trade_order_id ----
        async with async_session_factory() as session:
            pred = AgentPrediction(
                prediction_time=now,
                sentiment_window_id=sentiment_window_id,
                predicted_direction=predicted_direction,
                matched_pattern_id=matched_pattern_id,
                matched_pattern_name=matched_pattern_name,
                confidence=confidence,
                entry_timing=entry_timing,
                reasoning=reasoning,
                skip_trade_reason=final_skip_reason,
            )
            session.add(pred)
            await session.flush()  # 获取自增 id（事务内，不 commit）

            logger.info(
                "Predict: 预测记录已 flush | pred.id={} | direction={} | confidence={:.4f} | "
                "matched_pattern_id={} | skip_trade_reason={}",
                pred.id,
                predicted_direction,
                confidence,
                matched_pattern_id,
                final_skip_reason,
            )

            # ---- 交易执行 / 跳过 ----
            if do_trade and not skip_trade_reason:
                # 交易前二次确认：匹配模式仍为 ACTIVE（防止双 Worker 时序窗口）
                if matched_pattern_id is not None and matched_pattern_status != "ACTIVE":
                    logger.warning(
                        "Predict: 匹配模式已非 ACTIVE | pattern_id={} status={} | 跳过交易",
                        matched_pattern_id,
                        matched_pattern_status,
                    )
                    pred.skip_trade_reason = f"模式已退役/暂停（status={matched_pattern_status}）"
                    final_skip_reason = pred.skip_trade_reason
                    do_trade = False

            if do_trade and not skip_trade_reason:
                # 交易门控通过且非预设跳过场景 → 执行交易
                logger.info(
                    "Predict: 交易门控通过 | reason='{}' | 调用 execute_trade",
                    trade_reason,
                )
                try:
                    order = await self._trader.execute_trade(
                        prediction=predicted_direction,
                        confidence=confidence,
                        agent_prediction_id=pred.id,
                    )
                    # 回填 trade_order_id（Req 10.3 双向关联），同一事务内
                    if order is not None:
                        pred.trade_order_id = order.id
                        logger.info(
                            "Predict: 交易完成 | order.id={} | 已回填 trade_order_id",
                            order.id,
                        )
                        from .metrics import metrics_collector
                        metrics_collector.record_trade("EXECUTED", trade_reason)
                    else:
                        logger.warning(
                            "Predict: execute_trade 返回 None（交易未成功），不回填 trade_order_id"
                        )
                        from .metrics import metrics_collector
                        metrics_collector.record_trade("FAILED", "execute_trade returned None")
                except Exception as trade_exc:
                    # 交易失败不影响预测记录——记录错误但不回退预测
                    logger.error(
                        "Predict: 交易执行异常 | pred.id={} | error={}",
                        pred.id,
                        trade_exc,
                    )
            else:
                logger.info(
                    "Predict: 跳过交易 | reason='{}'",
                    final_skip_reason,
                )
                from .metrics import metrics_collector
                metrics_collector.record_trade("SKIPPED", final_skip_reason or "")

            # 原子提交：pred 记录 + trade_order_id（如有）在同一事务内
            await session.commit()

        return pred

    # ======================================================================
    # Evolve 阶段（Req 5.2 / 5.3 / 5.4 / 5.5 / 5.6 / 5.7 / 5.8 / 7.4 / 11.3）
    # ======================================================================

    async def evolve(self) -> None:
        """
        进化阶段：基于全部模式与最近预测验证结果进行自我反思与模式进化。

        流程（design.md §2 Evolve 段）：
        1. 读全部模式（ACTIVE + 近期 RETIRED，最近 7 天退役的）+
           最近 agent_evolve_interval 次 AgentPrediction 的预测记录和验证结果 → 序列化为 dict 列表
        2. LLM 调用在事务外：agent_evolve(all_patterns_dicts, recent_predictions_dicts, timeout)
        3. 生成唯一 evolve_phase_id（UUID4 前 8 位 + 时间戳）
        4. 对 EvolveOutput.operations 逐条应用：
           - RETAIN：跳过（不做 DB 操作）
           - MODIFY：apply_pattern_change(session, "UPDATE", {...}, "EVOLVE", evolve_phase_id)
           - RETIRE：apply_pattern_change(session, "RETIRE", {...}, "EVOLVE", evolve_phase_id)
           - CREATE：apply_pattern_change(session, "CREATE", {...}, "EVOLVE", evolve_phase_id)
        5. 上限约束（Req 5.8）：LLM 操作应用完后，检查 ACTIVE 数是否 > active_cap
           → 若是，用 select_retire_candidates 选出超额 → 逐个 apply_pattern_change RETIRE
        6. 冷启动保护（Req 11.3）：ACTIVE < 3 时忽略所有 RETIRE 操作
           （含 LLM 返回的与上限触发的），侧重发现
        7. 单条操作失败 → rollback 该操作、继续其余；
           LLM 失败 → 跳过且模式库不变（Req 7.4）

        设计约束：
        - LLM 调用在事务开启前完成（避免长事务占用连接池）
        - 使用独立 async_session_factory() 会话
        - loguru 日志，无静默降级
        """
        # ========== Step 1：读取数据（只读会话）==========
        all_patterns_dicts: list[dict] = []
        recent_predictions_dicts: list[dict] = []

        async with async_session_factory() as session:
            # 读全部 ACTIVE 模式
            active_stmt = select(PatternMemory).where(
                PatternMemory.status == "ACTIVE"
            )
            active_result = await session.execute(active_stmt)
            active_patterns = active_result.scalars().all()

            # 读近期 RETIRED 模式（最近 7 天退役的，通过 updated_at 判断）
            seven_days_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)
            retired_stmt = select(PatternMemory).where(
                PatternMemory.status == "RETIRED",
                PatternMemory.updated_at >= seven_days_ago,
            )
            retired_result = await session.execute(retired_stmt)
            retired_patterns = retired_result.scalars().all()

            # 合并 ACTIVE + 近期 RETIRED → 序列化为 dict
            all_patterns_orm = list(active_patterns) + list(retired_patterns)
            for p in all_patterns_orm:
                all_patterns_dicts.append({
                    "id": p.id,
                    "pattern_name": p.pattern_name,
                    "description": p.description,
                    "curve_features": p.curve_features,
                    "conditions": p.conditions,
                    "predicted_direction": p.predicted_direction,
                    "win_rate": p.win_rate,
                    "sample_count": p.sample_count,
                    "correct_count": p.correct_count,
                    "confidence_score": p.confidence_score,
                    "status": p.status,
                })

            # 读最近 agent_evolve_interval 次 AgentPrediction（含验证结果）
            pred_stmt = (
                select(AgentPrediction)
                .order_by(AgentPrediction.prediction_time.desc())
                .limit(settings.agent_evolve_interval)
            )
            pred_result = await session.execute(pred_stmt)
            recent_preds = pred_result.scalars().all()

            for pred in recent_preds:
                recent_predictions_dicts.append({
                    "id": pred.id,
                    "prediction_time": pred.prediction_time.isoformat()
                    if pred.prediction_time else None,
                    "predicted_direction": pred.predicted_direction,
                    "matched_pattern_id": pred.matched_pattern_id,
                    "matched_pattern_name": pred.matched_pattern_name,
                    "confidence": pred.confidence,
                    "reasoning": pred.reasoning,
                    "is_correct": pred.is_correct,
                    "actual_outcome": pred.actual_outcome,
                    "actual_return": pred.actual_return,
                })

        active_count = len(active_patterns)
        logger.info(
            "Evolve: 数据准备完成 | ACTIVE 模式={} | 近期 RETIRED={} | 最近预测={}",
            active_count,
            len(retired_patterns),
            len(recent_predictions_dicts),
        )

        # ========== 空库短路：无任何可进化模式时跳过 LLM 调用 ==========
        # 冷启动/manual 模式下模式库长期为空，此时 Evolve 无对象可反思，
        # 若仍调用 LLM 只会稳定返回「无操作」并浪费 token 与 heavy 队列时间。
        if not all_patterns_dicts:
            logger.info(
                "Evolve: 模式库为空（无 ACTIVE 且无近期 RETIRED），"
                "跳过本次 Evolve（不调用 LLM）"
            )
            return

        # ========== Step 2：LLM 调用（在事务外完成，避免长事务）==========
        try:
            evolve_output = await self._llm.agent_evolve(
                all_patterns=all_patterns_dicts,
                recent_predictions=recent_predictions_dicts,
                timeout=settings.agent_llm_timeouts["EVOLVE"],
            )
        except Exception as exc:
            # Req 7.4：LLM 失败/超时/重试耗尽 → 跳过本次 Evolve，模式库保持不变
            logger.error(
                "Evolve: LLM 调用失败，跳过本次 Evolve（模式库不变）| "
                "error_type={} | error={}",
                type(exc).__name__,
                str(exc),
            )
            return

        if not evolve_output.operations:
            logger.info("Evolve: LLM 未返回任何进化操作，本次 Evolve 结束")
            return

        # ========== Step 2.5：LLM 输出语义验证（Plan 步骤 5）==========
        if settings.agent_llm_validation_enabled:
            all_ids = {p["id"] for p in all_patterns_dicts}
            hard_failures, soft_warnings = validate_evolve_output(
                evolve_output, all_ids
            )
            if soft_warnings:
                for w in soft_warnings:
                    logger.warning("Evolve 语义验证 SOFT: {}", w)
            if hard_failures:
                for f in hard_failures:
                    logger.error("Evolve 语义验证 HARD: {}", f)
                # 过滤掉 HARD_FAIL 的 operations
                original_count = len(evolve_output.operations)
                evolve_output.operations = [
                    op for i, op in enumerate(evolve_output.operations)
                    if not any(
                        f"operations[{i}]" in f for f in hard_failures
                    )
                ]
                logger.info(
                    "Evolve: 语义验证过滤 {}/{} 条操作",
                    original_count - len(evolve_output.operations),
                    original_count,
                )
                if not evolve_output.operations:
                    logger.info("Evolve: 所有操作被过滤，本次 Evolve 结束")
                    return

        logger.info(
            "Evolve: LLM 返回 {} 条进化操作 | reasoning={}...",
            len(evolve_output.operations),
            evolve_output.reasoning[:100],
        )

        # ========== Step 3：生成唯一 evolve_phase_id ==========
        # Fix #22: 使用截断的 UUID（前8位）+ 时间戳，保持可读性且符合 DB 字段长度
        evolve_phase_id = f"{uuid.uuid4().hex[:8]}-{int(time.time())}"

        # ========== Step 4：应用进化操作（独立事务，含冷启动保护）==========
        async with async_session_factory() as session:
            # 获取当前 ACTIVE 模式数用于冷启动保护判断
            current_active_stmt = select(PatternMemory).where(
                PatternMemory.status == "ACTIVE"
            )
            current_active_result = await session.execute(current_active_stmt)
            current_active_count = len(current_active_result.scalars().all())

            # 冷启动保护标志：ACTIVE < 3 时忽略所有 RETIRE 操作（Req 11.3）
            cold_start_protection = current_active_count < 3
            if cold_start_protection:
                logger.info(
                    "Evolve: 冷启动保护激活（ACTIVE={} < 3），忽略所有 RETIRE 操作",
                    current_active_count,
                )

            applied_count = 0
            skipped_retain = 0
            skipped_cold_start = 0
            failed_count = 0

            for idx, op in enumerate(evolve_output.operations, 1):
                # RETAIN：跳过（不做 DB 操作）
                if op.action == "RETAIN":
                    skipped_retain += 1
                    continue

                # 冷启动保护：ACTIVE < 3 时忽略 RETIRE 操作
                if op.action == "RETIRE" and cold_start_protection:
                    skipped_cold_start += 1
                    logger.debug(
                        "Evolve: 跳过 RETIRE 操作（冷启动保护）| target_pattern_id={}",
                        op.target_pattern_id,
                    )
                    continue

                try:
                    async with session.begin_nested():  # savepoint：失败仅回滚此子事务
                        if op.action == "MODIFY":
                            # 构建 UPDATE 的 pattern_data
                            pattern_data: dict = {
                                "target_pattern_id": op.target_pattern_id,
                                "change_reason": op.reason,
                            }
                            # 将 modifications 中的可更新字段合并到 pattern_data
                            if op.modifications:
                                updatable_fields = (
                                    "pattern_name",
                                    "description",
                                    "curve_features",
                                    "conditions",
                                    "predicted_direction",
                                    "confidence_score",
                                )
                                for field in updatable_fields:
                                    if field in op.modifications:
                                        pattern_data[field] = op.modifications[field]

                            await self.apply_pattern_change(
                                session=session,
                                operation="UPDATE",
                                pattern_data=pattern_data,
                                phase="EVOLVE",
                                evolve_phase_id=evolve_phase_id,
                            )
                            applied_count += 1

                        elif op.action == "RETIRE":
                            pattern_data = {
                                "target_pattern_id": op.target_pattern_id,
                                "change_reason": op.reason,
                            }
                            await self.apply_pattern_change(
                                session=session,
                                operation="RETIRE",
                                pattern_data=pattern_data,
                                phase="EVOLVE",
                                evolve_phase_id=evolve_phase_id,
                            )
                            applied_count += 1

                        elif op.action == "CREATE":
                            if op.new_pattern is None:
                                logger.warning(
                                    "Evolve: 第 {}/{} 条 CREATE 操作缺少 new_pattern，跳过",
                                    idx,
                                    len(evolve_output.operations),
                                )
                                failed_count += 1
                                continue

                            pattern_data = {
                                "pattern_name": op.new_pattern.pattern_name,
                                "description": op.new_pattern.description,
                                "curve_features": op.new_pattern.curve_features,
                                "conditions": op.new_pattern.conditions,
                                "predicted_direction": op.new_pattern.predicted_direction,
                                "confidence_score": op.new_pattern.confidence_score,
                                "change_reason": op.reason,
                            }
                            await self.apply_pattern_change(
                                session=session,
                                operation="CREATE",
                                pattern_data=pattern_data,
                                phase="EVOLVE",
                                evolve_phase_id=evolve_phase_id,
                            )
                            applied_count += 1

                        else:
                            logger.warning(
                                "Evolve: 第 {}/{} 条操作的 action='{}' 不识别，跳过",
                                idx,
                                len(evolve_output.operations),
                                op.action,
                            )

                except Exception as exc:
                    # savepoint 已自动回滚，session 仍可继续
                    failed_count += 1
                    logger.error(
                        "Evolve: 第 {}/{} 条操作失败 | action={} target_id={} | "
                        "error_type={} | error={}",
                        idx,
                        len(evolve_output.operations),
                        op.action,
                        op.target_pattern_id,
                        type(exc).__name__,
                        str(exc),
                    )
                    continue

            # ========== Step 5：上限约束（Req 5.8）==========
            # 重新查询当前 ACTIVE 数（因为前面的操作可能改变了数量）
            cap_stmt = select(PatternMemory).where(
                PatternMemory.status == "ACTIVE"
            )
            cap_result = await session.execute(cap_stmt)
            cap_active_patterns = cap_result.scalars().all()
            cap_active_count = len(cap_active_patterns)

            if cap_active_count > settings.agent_active_pattern_cap:
                # 冷启动保护也适用于上限触发的淘汰（Req 11.3）
                if cap_active_count < 3:
                    logger.info(
                        "Evolve: 虽超上限但 ACTIVE={} < 3，冷启动保护生效，不执行上限淘汰",
                        cap_active_count,
                    )
                else:
                    # 构建 PatternStat 列表用于 select_retire_candidates
                    # Plan 步骤 13：批量查询所有 ACTIVE 模式的最近预测，计算 recent_win_rate
                    pattern_ids = [p.id for p in cap_active_patterns]
                    recent_pred_stmt = (
                        select(
                            AgentPrediction.matched_pattern_id,
                            AgentPrediction.is_correct,
                        )
                        .where(
                            AgentPrediction.matched_pattern_id.in_(pattern_ids),
                            AgentPrediction.is_correct.isnot(None),
                        )
                        .order_by(
                            AgentPrediction.matched_pattern_id,
                            AgentPrediction.prediction_time.desc(),
                        )
                    )
                    recent_pred_result = await session.execute(recent_pred_stmt)
                    all_recent_preds = recent_pred_result.all()

                    # 按 pattern_id 分组，每组取前 10 条
                    preds_by_pattern: dict[int, list] = defaultdict(list)
                    for row in all_recent_preds:
                        pid = row[0]
                        if len(preds_by_pattern[pid]) < 10:
                            preds_by_pattern[pid].append(row[1])

                    pattern_stats = []
                    for p in cap_active_patterns:
                        recent_wr: float | None = None
                        recent_preds = preds_by_pattern.get(p.id, [])
                        if recent_preds:
                            correct = sum(1 for v in recent_preds if v is True)
                            recent_wr = correct / len(recent_preds)

                        pattern_stats.append(
                            PatternStat(
                                id=p.id,
                                status=p.status,
                                win_rate=p.win_rate,
                                sample_count=p.sample_count,
                                recent_win_rate=recent_wr,
                            )
                        )
                    # select_retire_candidates 内部已含冷启动保护（active < 3 → []）
                    retire_ids = select_retire_candidates(
                        patterns=pattern_stats,
                        active_cap=settings.agent_active_pattern_cap,
                        min_sample=settings.agent_min_sample,
                    )

                    if retire_ids:
                        logger.info(
                            "Evolve: ACTIVE={} > 上限={}，触发上限淘汰 {} 个模式 | ids={}",
                            cap_active_count,
                            settings.agent_active_pattern_cap,
                            len(retire_ids),
                            retire_ids,
                        )
                        for retire_id in retire_ids:
                            try:
                                async with session.begin_nested():  # savepoint：失败仅回滚此子事务
                                    await self.apply_pattern_change(
                                        session=session,
                                        operation="RETIRE",
                                        pattern_data={
                                            "target_pattern_id": retire_id,
                                            "change_reason": (
                                                f"上限淘汰：ACTIVE 模式数 {cap_active_count} "
                                                f"超过上限 {settings.agent_active_pattern_cap}，"
                                                f"按 win_rate 升序淘汰"
                                            ),
                                        },
                                        phase="EVOLVE",
                                        evolve_phase_id=evolve_phase_id,
                                    )
                                    applied_count += 1
                            except Exception as exc:
                                # savepoint 已自动回滚，session 仍可继续
                                failed_count += 1
                                logger.error(
                                    "Evolve: 上限淘汰模式 id={} 失败 | error={}",
                                    retire_id,
                                    exc,
                                )
                                continue

            # 提交所有成功的变更
            await session.commit()

        logger.info(
            "Evolve: 阶段完成 | evolve_phase_id={} | 已应用={} | RETAIN 跳过={} | "
            "冷启动跳过={} | 失败={}",
            evolve_phase_id,
            applied_count,
            skipped_retain,
            skipped_cold_start,
            failed_count,
        )
