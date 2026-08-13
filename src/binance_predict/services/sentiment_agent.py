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
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import (
    AgentPrediction,
    BinningSnapshotModel,
    PatternChangeLog,
    PatternMemory,
    SentimentWindow,
)
from .agent_logic import (
    HIT_CONCORDANT,
    HIT_CONFLICT,
    HIT_NONE,
    HIT_SINGLE,
    PatternRow,
    PatternStat,
    PredicateHit,
    TradeGateContext,
    WindowRow,
    compute_is_correct,
    compute_pattern_fingerprint,
    detect_duplicate_pattern,
    evaluate_trade_gate,
    is_prediction_stale,
    pattern_confidence,
    plan_active_patterns,
    plan_learn_windows,
    recompute_win_rate,
    resolve_predicate_hits,
    select_retire_candidates,
    should_trade,
)
from .alerting import AlertService
from .backtest import (
    evaluate_on_holdout,
    snapshot_token,
    time_split,
    wilson_lower_bound,
)
from .curve_features import (
    cluster_windows,
    extract_features,
)
from .discovery import (
    VERDICT_ACTIVE,
    VERDICT_OBSERVE,
    screen_hypotheses,
)
from .ev_gate import hypothesis_ev, truncate_to_decision_point
from .hypothesis_miner import mine_hints
from .llm_service import LLMService
from .llm_validator import (
    validate_arbitrate_output,
    validate_evolve_output,
    validate_learn_output,
)
from .predicates import evaluate_predicate
from .prediction_trading import BinancePredictionTrader
from .risk_control import RiskController
from .symbolizer import (
    CHANNEL_FIELDS,
    BinningSnapshot,
    WindowView,
    build_window_view,
    compute_channel_snapshots,
    should_freeze,
)
from .verification import (
    DEATH_ALIVE,
    DEATH_EXPIRED,
    DEATH_SPURIOUS,
    MIN_DEATH_HITS,
    diagnose_death,
    live_lift_summary,
    pooled_local_baseline,
)

if TYPE_CHECKING:
    from ..models.schemas import ChangeType


# Q7-2 反馈循环：注入 Deep Learn prompt 的负样本（SPURIOUS 死亡模式）条数上限，
# 与发现预算（每轮 ≤20 条假设）协调，防 token 膨胀
_FEEDBACK_NEGATIVE_LIMIT: int = 20


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
        # 深度分析并发锁：仅防止多个 deep_learn 发现任务（LLM 长调用）同时执行，
        # 用于成本/资源控制，不覆盖 pattern_memory 写入。
        self._deep_learn_lock = asyncio.Lock()
        # P0-2：pattern_memory 唯一写锁——Validate / Evolve / commit_deep_learn 的
        # DB 写事务统一持此锁，取代原「调度器 _write_lock 与 _deep_learn_lock 两把锁
        # 守护同一张表」的竞态设计。锁粒度仅覆盖 DB 写，不含 LLM 调用。
        self._pattern_write_lock = asyncio.Lock()

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

        # P0-2：Validate 回填预测 + 更新模式统计属 pattern_memory 写操作，
        # 持唯一写锁与 Evolve / commit_deep_learn 串行，避免并发写竞态。
        async with self._pattern_write_lock, async_session_factory() as session:
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

                # 3. 若匹配了模式且为方向性预测 → 更新模式统计。
                # 结算口径对齐：NO_TRADE 是弃权（未下注、无盈亏），不应计入
                # 模式胜率——outcome 改按涨跌正负号标注后 NOISE 几乎绝迹，
                # 若仍计入会把弃权全部记为失败，无辜拖垮模式胜率。
                if (
                    pred.matched_pattern_id is not None
                    and pred.predicted_direction in ("UP", "DOWN")
                ):
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
            "predicate": pattern.predicate,
            "binning_version": pattern.binning_version,
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
                # P0-1：允许调用方指定初始状态（Evolve CREATE 传 EVOLVING 观察态），
                # 默认 ACTIVE（deep-learn 经准入闸门后写库沿用旧行为）。
                status=pattern_data.get("status", "ACTIVE"),
                win_rate=0.0,
                sample_count=0,
                correct_count=0,
                discovery_method=pattern_data.get("discovery_method", "LEGACY"),
                holdout_win_rate=pattern_data.get("holdout_win_rate"),
                holdout_sample_count=pattern_data.get("holdout_sample_count"),
                holdout_ci_lower=pattern_data.get("holdout_ci_lower"),
                # 科学发现轨（Q4/Q5）：谓词 DSL 与发现时的分箱快照版本
                predicate=pattern_data.get("predicate"),
                binning_version=pattern_data.get("binning_version"),
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

            # 更新指定字段（仅更新 LLM 可写的业务字段 + Deep Learn 样本外统计 + 谓词轨字段）
            updatable_fields = (
                "pattern_name",
                "description",
                "curve_features",
                "conditions",
                "predicted_direction",
                "confidence_score",
                "discovery_method",
                "holdout_win_rate",
                "holdout_sample_count",
                "holdout_ci_lower",
                "predicate",
                "binning_version",
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

            # 置为 RETIRED；Q7-1：死因判定时回填 death_cause / lifespan_days
            # （调用方未提供时保持 None——上限淘汰等非死因路径不写这两列）
            pattern.status = "RETIRED"
            if pattern_data.get("death_cause") is not None:
                pattern.death_cause = pattern_data["death_cause"]
            if pattern_data.get("lifespan_days") is not None:
                pattern.lifespan_days = pattern_data["lifespan_days"]

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

    @staticmethod
    def _uniform_sample(items: list, k: int) -> list:
        """从有序 items 中按位置均匀抽取至多 k 个（保持原顺序，去重索引）。

        k>=len 或 k<=0 时原样返回。用于分层后层内按时间均匀抽样，
        保证抽样覆盖整个时间跨度而非仅集中在两端。
        """
        n = len(items)
        if k <= 0 or k >= n:
            return list(items)
        if k == 1:
            return [items[n // 2]]
        seen: set[int] = set()
        picked: list = []
        for i in range(k):
            idx = round(i * (n - 1) / (k - 1))
            if idx not in seen:
                seen.add(idx)
                picked.append(items[idx])
        return picked

    async def _fetch_deep_learn_windows(
        self,
        days_back: int | None = None,
        max_windows: int | None = None,
    ) -> list[dict]:
        """P0-1 时间分层采样：圈定近 days_back 天 outcome 非空窗口，按 outcome
        分层、层内按时间均匀抽样至多 max_windows 个，返回 start_time 升序的序列化 dict。

        取代原 `order_by(start_time.desc()).limit(max_windows)`——后者只取最近 N 个
        却当作"全量历史"，既丢失早期样本又误导 LLM 的时间跨度认知。
        """
        if days_back is None:
            days_back = settings.agent_deep_learn_days_back
        if max_windows is None:
            max_windows = settings.agent_deep_learn_max_windows

        cutoff_ms = int(time.time() * 1000) - int(days_back) * 86_400_000
        async with async_session_factory() as session:
            stmt = (
                select(SentimentWindow)
                .where(
                    SentimentWindow.outcome.isnot(None),
                    SentimentWindow.start_time >= cutoff_ms,
                )
                .order_by(SentimentWindow.start_time.asc())
            )
            result = await session.execute(stmt)
            raw_windows = result.scalars().all()
            all_dicts = [
                {
                    "id": w.id,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "curve_up_pct": w.curve_up_pct,
                    "curve_down_pct": w.curve_down_pct,
                    # 科学发现轨三通道（Q2）：价格与交易量曲线供 price/volume 通道符号化
                    "curve_btc_price": w.curve_btc_price,
                    "curve_trade_volume": w.curve_trade_volume,
                    # 经济闸入场价来源（V1.1，Q6 第 5 步）：真实 token 价曲线
                    "curve_up_price": w.curve_up_price,
                    "curve_down_price": w.curve_down_price,
                    "outcome": w.outcome,
                    "actual_return": w.actual_return,
                    "sample_count": w.sample_count,
                }
                for w in raw_windows
            ]

        total = len(all_dicts)
        if total <= max_windows:
            return all_dicts

        # 按 outcome 分层（各层已随全局按 start_time 升序）
        layers: dict[str, list[dict]] = defaultdict(list)
        for w in all_dicts:
            layers[(w.get("outcome") or "NOISE").upper()].append(w)

        # 按层大小比例分配配额，层内均匀抽样
        sampled: list[dict] = []
        for key, group in layers.items():
            quota = max(1, round(max_windows * len(group) / total))
            sampled.extend(self._uniform_sample(group, quota))

        # 配额合计可能略超 max_windows，再做一次全局均匀收敛
        sampled.sort(key=lambda w: w.get("start_time", 0))
        if len(sampled) > max_windows:
            sampled = self._uniform_sample(sampled, max_windows)
        return sampled

    async def _load_latest_binning_snapshots(self) -> dict[str, BinningSnapshot]:
        """读取 binning_snapshots 表最新一版快照（created_at_epoch 降序首见即最新）。"""
        async with async_session_factory() as session:
            stmt = select(BinningSnapshotModel).order_by(
                BinningSnapshotModel.created_at_epoch.desc(),
                BinningSnapshotModel.id.desc(),
            )
            rows = (await session.execute(stmt)).scalars().all()
        latest: dict[str, BinningSnapshot] = {}
        for row in rows:
            if row.channel in latest:
                continue
            latest[row.channel] = BinningSnapshot(
                version=row.version,
                edges=tuple(float(x) for x in row.edges),  # type: ignore[arg-type]
                created_at_epoch=float(row.created_at_epoch),
                sample_count=int(row.sample_count),
            )
        return latest

    async def _load_binning_snapshots_by_versions(
        self, versions: set[str]
    ) -> dict[str, dict[str, BinningSnapshot]]:
        """按版本集合加载分箱快照（Predict 谓词符号化用，宪法第八条规则 2）。

        仪器精度对齐：模式须以其出生 binning_version 对应的边界符号化。
        返回 {version: {channel: BinningSnapshot}}；表中不存在的版本缺席，
        由调用方跳过对应模式并记 warning（不静默降级）。
        """
        if not versions:
            return {}
        async with async_session_factory() as session:
            stmt = select(BinningSnapshotModel).where(
                BinningSnapshotModel.version.in_(sorted(versions))
            )
            rows = (await session.execute(stmt)).scalars().all()
        result: dict[str, dict[str, BinningSnapshot]] = {}
        for row in rows:
            result.setdefault(row.version, {})[row.channel] = BinningSnapshot(
                version=row.version,
                edges=tuple(float(x) for x in row.edges),  # type: ignore[arg-type]
                created_at_epoch=float(row.created_at_epoch),
                sample_count=int(row.sample_count),
            )
        return result

    async def _ensure_binning_snapshots(
        self, windows: list[dict]
    ) -> dict[str, BinningSnapshot]:
        """分箱快照管理（Q4）：返回当前生效的每通道独立分箱快照。

        读 binning_snapshots 最新版 → should_freeze 判定（从未冻结或距上次 >=30 天，
        或通道覆盖不全）→ 需冻结时用本次采样窗口计算每通道独立分位边界并以新
        版本号落库（同一 version 三通道各一行）。冻结失败（全部通道样本不足）
        时退回旧版快照并记 warning，不静默吞错。
        """
        latest = await self._load_latest_binning_snapshots()
        probe = next(iter(latest.values()), None)
        if (
            len(latest) >= len(CHANNEL_FIELDS)
            and not should_freeze(probe)
        ):
            return latest

        now = time.time()
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        snapshots = compute_channel_snapshots(windows, version, created_at_epoch=now)
        if not snapshots:
            logger.warning(
                "Deep Learn: 分箱冻结失败（全部通道差值样本不足），退回旧版快照 | 旧版通道={}",
                sorted(latest.keys()),
            )
            return latest

        rows = [
            BinningSnapshotModel(
                version=snap.version,
                channel=channel,
                edges=list(snap.edges),
                sample_count=snap.sample_count,
                created_at_epoch=snap.created_at_epoch,
            )
            for channel, snap in snapshots.items()
        ]
        async with async_session_factory() as session:
            session.add_all(rows)
            await session.commit()
        logger.info(
            "Deep Learn: 分箱快照已冻结 | version={} | 通道={}",
            version,
            sorted(snapshots.keys()),
        )
        return snapshots

    @staticmethod
    def _snapshots_version(snapshots: dict[str, BinningSnapshot]) -> str | None:
        """从快照 dict 提取版本号（同一 version 覆盖三通道；空 dict 返回 None）。"""
        probe = next(iter(snapshots.values()), None)
        return probe.version if probe is not None else None

    @staticmethod
    def _view_to_payload(view: WindowView) -> dict:
        """WindowView → LLM 消费的序列化 payload（Q1：符号串为主，几何摘要为辅）。

        geometry 剔除 extrema 明细（逐极值点列表 token 开销大且对 LLM 无增益，
        其统计结论已含于 peak_count/extremum_spacing）。
        """
        return {
            "start_time": view.start_time,
            "outcome": view.outcome,
            "channels": {
                channel: {
                    "symbols": cv.symbols,
                    "geometry": {
                        k: v for k, v in cv.geometry.items() if k != "extrema"
                    },
                }
                for channel, cv in view.channels.items()
            },
        }

    @staticmethod
    def _current_window_dict(window_end_ms: int, current_curve: list[dict]) -> dict:
        """PREDICT 事件 current_curve → build_window_view 输入形态（宪法第八条规则 3）。

        current_curve 每点形态 {t, up_pct, down_pct, btc_price, trade_volume}
        （main.py 从 _pm_history 切片）；本方法拆为三通道 [{t, v}] 曲线。
        v=None 的点被 symbolizer._series_values 防御跳过，对应通道有效点 <2
        时不产生符号，谓词对缺失通道按防御求值 False。
        """
        def _points(key: str) -> list[dict]:
            return [{"t": p.get("t"), "v": p.get(key)} for p in current_curve]

        return {
            # start_time 仅供 WindowView 记录（谓词执行不读），取首个采样时刻，
            # 空曲线时回推 5min 窗口起点（Q3）
            "start_time": (
                int(current_curve[0].get("t") or 0)
                if current_curve
                else window_end_ms - 300_000
            ),
            "curve_up_pct": _points("up_pct"),
            "curve_btc_price": _points("btc_price"),
            "curve_trade_volume": _points("trade_volume"),
        }

    def _screen_and_serialize(
        self,
        hypotheses: list,
        holdout_views: list[WindowView],
        binning_version: str | None,
        holdout_windows: list[dict] | None = None,
    ) -> list[dict]:
        """Q6 初筛流水线 + 序列化为 DeepLearnDiscovery 兼容 dict（预览与 commit 用）。

        LLM 只产出假设（predicate + target_outcome + 命名/描述），统计审判全部
        由 screen_hypotheses 在 holdout 符号化视图上完成（宪法第〇条角色分离）。
        每条假设附加 screen_* 审判证据；holdout_* 三列不再由新轨填充（旧轨
        pycluster 保留该路径）。REJECT 原因非静默落 screen_reject_reason。

        V1.1（Q6 第 5 步经济闸）：双轨 ACTIVE 的假设另算入场价经济账
        （ev_gate.hypothesis_ev，需 holdout_windows 提供价格曲线），
        费后 EV CI 下界 ≤0 或注数不足 → 降级 OBSERVE（经济功效不足而非
        模式无效，与 FDR 降级同语义）；screen_ev_* 证据随裁决一并返回。
        """
        # 防御式过滤半成品（流式 partial 可能字段残缺；非流式 Pydantic 已保证完整）
        valid = [
            h
            for h in hypotheses
            if getattr(h, "pattern_name", None)
            and getattr(h, "predicate", None)
            and getattr(h, "target_outcome", None) in ("UP", "DOWN")
        ]
        hypotheses_dicts = [
            {"predicate": h.predicate, "target_outcome": h.target_outcome}
            for h in valid
        ]
        screened = screen_hypotheses(hypotheses_dicts, holdout_views)

        results: list[dict] = []
        for h, s in zip(valid, screened):
            lr = s.lift_result
            verdict = s.verdict
            ev_result = None
            # Q6 第 5 步经济闸（V1.1）：仅双轨 ACTIVE 需要经济账；不通过降 OBSERVE
            if (
                settings.agent_ev_gate_enabled
                and verdict == VERDICT_ACTIVE
                and holdout_windows is not None
            ):
                ev_result = hypothesis_ev(
                    h.predicate,
                    h.target_outcome,
                    holdout_views,
                    holdout_windows,
                    t_sec=settings.agent_decision_point_sec,
                )
                if not ev_result.passed:
                    verdict = VERDICT_OBSERVE
            results.append({
                "operation": "CREATE",
                "target_pattern_id": None,
                "pattern_name": h.pattern_name,
                "description": getattr(h, "description", "") or "",
                "curve_features": {},
                "conditions": {},
                # 谓词命中时的预期偏向映射到旧方向字段（保持前端/Schema 兼容）
                "predicted_direction": h.target_outcome,
                "confidence_score": getattr(h, "confidence_score", None) or 0.0,
                "change_reason": getattr(h, "rationale", "") or "",
                "discovery_method": "LLM_DEEP",
                "predicate": h.predicate,
                "binning_version": binning_version,
                "screen_verdict": verdict,
                "screen_lift": lr.lift if lr else None,
                "screen_ci_lower": lr.ci_lower if lr else None,
                "screen_ci_upper": lr.ci_upper if lr else None,
                "screen_hit_count": len(s.hit_start_times),
                "screen_reject_reason": s.reject_reason,
                # 经济闸证据（V1.1；非 ACTIVE 候选为 None）
                "screen_ev": (
                    round(ev_result.ev, 4)
                    if ev_result is not None and ev_result.ev is not None
                    else None
                ),
                "screen_ev_ci_lower": (
                    round(ev_result.ev_ci_lower, 4)
                    if ev_result is not None and ev_result.ev_ci_lower is not None
                    else None
                ),
                "screen_ev_ci_upper": (
                    round(ev_result.ev_ci_upper, 4)
                    if ev_result is not None and ev_result.ev_ci_upper is not None
                    else None
                ),
                "screen_ev_fires": ev_result.n_fires if ev_result is not None else None,
                "screen_ev_passed": ev_result.passed if ev_result is not None else None,
            })
        return results

    async def _load_discovery_feedback(self) -> dict:
        """组装 Deep Learn 反馈包（宪法 Q7-2 反馈策略）。

        - negatives（负样本）：SPURIOUS 死亡的谓词模式全量细节（名称/谓词/方向/
          描述/live 统计），波普尔排除法——被证伪的结构禁止重提
        - positive_summary（正样本）：存活 ACTIVE 谓词模式的统计摘要（数量/平均
          胜率/方向分布），**不含谓词结构**——防近亲繁殖导致全库同质化
        - lifespan_stats（元信息）：EXPIRED 死亡的 lifespan_days 分布——
          规律的预期寿命量级，引导 LLM 提稳健、跨 regime 的结构
        """
        async with async_session_factory() as session:
            # 负样本：最近处决的 SPURIOUS 谓词模式（上限防 token 膨胀）
            neg_stmt = (
                select(PatternMemory)
                .where(
                    PatternMemory.death_cause == DEATH_SPURIOUS,
                    PatternMemory.predicate.isnot(None),
                )
                .order_by(PatternMemory.updated_at.desc())
                .limit(_FEEDBACK_NEGATIVE_LIMIT)
            )
            negatives = (await session.execute(neg_stmt)).scalars().all()

            # 正样本摘要：只取统计字段（不取谓词结构）
            pos_stmt = select(
                PatternMemory.predicted_direction,
                PatternMemory.win_rate,
            ).where(
                PatternMemory.status == "ACTIVE",
                PatternMemory.predicate.isnot(None),
            )
            pos_rows = (await session.execute(pos_stmt)).all()

            # 存活期：EXPIRED 死亡的 lifespan_days
            life_stmt = select(PatternMemory.lifespan_days).where(
                PatternMemory.death_cause == DEATH_EXPIRED,
                PatternMemory.lifespan_days.isnot(None),
            )
            lifespans = sorted(
                float(x) for x in (await session.execute(life_stmt)).scalars().all()
            )

        positive_count = len(pos_rows)
        direction_dist = Counter((r[0] or "UNKNOWN") for r in pos_rows)
        n = len(lifespans)
        lifespan_stats = {
            "count": n,
            "mean": round(sum(lifespans) / n, 2) if n else None,
            "median": (
                round(
                    lifespans[n // 2]
                    if n % 2
                    else (lifespans[n // 2 - 1] + lifespans[n // 2]) / 2,
                    2,
                )
                if n
                else None
            ),
            "max": lifespans[-1] if n else None,
        }

        return {
            "negatives": [
                {
                    "pattern_name": p.pattern_name,
                    "predicate": p.predicate,
                    "predicted_direction": p.predicted_direction,
                    "description": p.description,
                    "win_rate": p.win_rate,
                    "sample_count": p.sample_count,
                }
                for p in negatives
            ],
            "positive_summary": {
                "count": positive_count,
                "avg_win_rate": (
                    round(sum(float(r[1] or 0.0) for r in pos_rows) / positive_count, 4)
                    if positive_count
                    else 0.0
                ),
                "up_count": direction_dist.get("UP", 0),
                "down_count": direction_dist.get("DOWN", 0),
            },
            "lifespan_stats": lifespan_stats,
        }

    async def deep_learn(
        self,
        max_windows: int | None = None,
    ) -> dict:
        """
        深度模式发现（科学发现轨，Phase 2）：LLM 作为假设生成器，程序做统计审判。

        流程（宪法第〇条角色分离 + Q1/Q4/Q6）：
        1. 时间分层采样取数（P0-1）
        2. 分箱快照管理（Q4：读 binning_snapshots，到期/缺失则冻结新版）
        3. time_split（P0-2：LLM 仅看 train 符号串，holdout 真正留出）
        4. LLM 消费 train 符号化视图，产出谓词假设（不做任何自我验证）
        5. 程序在 holdout 符号化视图上跑 Q6 初筛流水线（screen_hypotheses）
        6. 返回假设 + screen_* 审判证据供预览；不直接写 DB（commit 走 Q6 闸门）

        Args:
            max_windows: 最大读取窗口数，默认使用 settings.agent_deep_learn_max_windows

        Returns:
            dict 包含：
            - reasoning: LLM 分析推理过程
            - discoveries: 序列化假设列表（每条含 predicate / screen_* 审判证据）
            - method / snapshot_token / train_count / holdout_count / binning_version
            用户可通过 commit_deep_learn() 确认后写入 DB
        """
        # 并发保护：同一时刻只允许一个 deep_learn 执行
        if self._deep_learn_lock.locked():
            raise RuntimeError("已有深度分析任务正在执行，请稍后重试")

        async with self._deep_learn_lock:
            days_back = settings.agent_deep_learn_days_back
            if max_windows is None:
                max_windows = settings.agent_deep_learn_max_windows

            logger.info(
                "Deep Learn: 开始科学发现 | days_back={} | max_windows={}",
                days_back,
                max_windows,
            )

            # Step 1: 时间分层采样取数（P0-1）
            windows_dicts = await self._fetch_deep_learn_windows(days_back, max_windows)
            if not windows_dicts:
                logger.warning("Deep Learn: 无可用历史窗口，跳过分析")
                return {
                    "reasoning": "",
                    "discoveries": [],
                    "method": "LLM_DEEP",
                    "snapshot_token": snapshot_token([]),
                    "train_count": 0,
                    "holdout_count": 0,
                    "binning_version": None,
                }

            # Step 1.5: 决策点截断对齐（V1.1 规则 8）：发现/初筛与在线 predict
            # 统一用开窗后前 150s 视图，消除全窗初筛的截断错位；快照同从截断
            # 视图差值冻结（同一测量仪器，Q4 延伸）
            windows_dicts = truncate_to_decision_point(
                windows_dicts, settings.agent_decision_point_sec
            )

            # Step 2: 分箱快照（Q4：到期/缺失则冻结新版）
            snapshots = await self._ensure_binning_snapshots(windows_dicts)
            binning_version = self._snapshots_version(snapshots)

            # Step 3: 按时间切 train/holdout（P0-2：LLM 仅看 train，holdout 真正留出）
            train_windows, holdout_windows = time_split(
                windows_dicts, settings.agent_deep_learn_holdout_ratio
            )

            # Step 4: train 符号化 → LLM 假设生成；holdout 符号化供初筛
            train_views = [build_window_view(w, snapshots) for w in train_windows]
            train_payload = [self._view_to_payload(v) for v in train_views]
            holdout_views = [
                build_window_view(w, snapshots) for w in holdout_windows
            ]

            logger.info(
                "Deep Learn: 数据准备完成 | 采样窗口={} | train={} | holdout={} | "
                "分箱版本={} | 通道={}",
                len(windows_dicts),
                len(train_windows),
                len(holdout_windows),
                binning_version,
                sorted(snapshots.keys()),
            )

            # Step 4.5: 反馈包（Q7-2：负样本全量 + 正样本摘要 + 存活期统计）
            feedback = await self._load_discovery_feedback()
            logger.info(
                "Deep Learn: 反馈包就绪 | 负样本={} | 存活 ACTIVE={} | EXPIRED 存活期样本={}",
                len(feedback["negatives"]),
                feedback["positive_summary"]["count"],
                feedback["lifespan_stats"]["count"],
            )

            # Step 4.6: 假设矿机（程序预筛轨）：train 集穷举谓词 → 线索榜单。
            # 纯 CPU 计算放入线程，避免阻塞事件循环；只跑 train，holdout 绝不泄漏。
            hints = await asyncio.to_thread(mine_hints, train_views)
            logger.info("Deep Learn: 预筛线索榜单就绪 | 线索={}", len(hints))

            timeout = settings.agent_deep_learn_timeout
            try:
                discovery_output = await self._llm.agent_deep_learn(
                    symbolized_windows=train_payload,
                    feedback=feedback,
                    timeout=timeout,
                    hints=hints,
                )
            except Exception as exc:
                logger.error(
                    "Deep Learn: LLM 调用失败 | error_type={} | error={}",
                    type(exc).__name__,
                    str(exc),
                )
                raise

            logger.info(
                "Deep Learn: LLM 返回 {} 条假设 | reasoning={}...",
                len(discovery_output.hypotheses),
                discovery_output.reasoning[:200] if discovery_output.reasoning else "",
            )

            # Step 5: holdout 初筛（Q6：谓词执行 + lift 检验 + BH-FDR + 裁决）
            discoveries_serialized = self._screen_and_serialize(
                list(discovery_output.hypotheses),
                holdout_views,
                binning_version,
                holdout_windows=holdout_windows,
            )

            return {
                "reasoning": discovery_output.reasoning or "",
                "discoveries": discoveries_serialized,
                "method": "LLM_DEEP",
                "snapshot_token": snapshot_token([w["id"] for w in windows_dicts]),
                "train_count": len(train_windows),
                "holdout_count": len(holdout_windows),
                "binning_version": binning_version,
            }

    async def deep_learn_stream(
        self,
        max_windows: int | None = None,
    ):
        """深度模式发现（流式版，科学发现轨）：逐步产出事件供前端 SSE 实时展示。

        流程与 deep_learn 一致（采样 → 分箱快照 → time_split → LLM 假设生成 →
        Q6 初筛 → 预览，不写 DB），区别在于全程以事件流形式产出，且 LLM 调用
        走 create_partial 流式 + 空闲超时。

        产出事件（dict，交由 main 转成 SSE data 帧）：
        - {"type": "step",      "message": str}        阶段性进度（读取窗口/开始调用等）
        - {"type": "reasoning", "delta": str}          reasoning 增量（打字机）
        - {"type": "progress",  "hypotheses": int}     已解析假设条数
        - {"type": "done",      "reasoning": str, "discoveries": list}  最终结果（供勾选提交）
        - {"type": "error",     "message": str}        并发冲突/空闲超时/流式异常

        并发保护与 deep_learn 共用 self._deep_learn_lock（二者互斥）。
        """
        if self._deep_learn_lock.locked():
            yield {"type": "error", "message": "已有深度分析任务正在执行，请稍后重试"}
            return

        async with self._deep_learn_lock:
            days_back = settings.agent_deep_learn_days_back
            if max_windows is None:
                max_windows = settings.agent_deep_learn_max_windows

            logger.info(
                "Deep Learn(stream): 开始科学发现 | days_back={} | max_windows={}",
                days_back,
                max_windows,
            )
            yield {
                "type": "step",
                "message": f"开始历史分析（days_back={days_back} · max_windows={max_windows}）",
            }

            # Step 1: 时间分层采样取数（P0-1）
            windows_dicts = await self._fetch_deep_learn_windows(days_back, max_windows)
            if not windows_dicts:
                logger.warning("Deep Learn(stream): 无可用历史窗口，跳过分析")
                yield {
                    "type": "done",
                    "reasoning": "",
                    "discoveries": [],
                    "method": "LLM_DEEP",
                    "snapshot_token": snapshot_token([]),
                    "train_count": 0,
                    "holdout_count": 0,
                    "binning_version": None,
                }
                return

            # Step 1.5: 决策点截断对齐（V1.1 规则 8），与 deep_learn 同口径
            windows_dicts = truncate_to_decision_point(
                windows_dicts, settings.agent_decision_point_sec
            )

            # Step 2: 分箱快照（Q4）+ time_split（P0-2：LLM 仅看 train）
            snapshots = await self._ensure_binning_snapshots(windows_dicts)
            binning_version = self._snapshots_version(snapshots)
            train_windows, holdout_windows = time_split(
                windows_dicts, settings.agent_deep_learn_holdout_ratio
            )
            train_views = [build_window_view(w, snapshots) for w in train_windows]
            train_payload = [self._view_to_payload(v) for v in train_views]
            holdout_views = [
                build_window_view(w, snapshots) for w in holdout_windows
            ]

            logger.info(
                "Deep Learn(stream): 数据准备完成 | 采样窗口={} | train={} | holdout={} | 分箱版本={}",
                len(windows_dicts),
                len(train_windows),
                len(holdout_windows),
                binning_version,
            )
            yield {
                "type": "step",
                "message": (
                    f"数据准备完成：采样 {len(windows_dicts)} 个"
                    f"（train {len(train_windows)} / holdout {len(holdout_windows)}）"
                    f" · 分箱版本 {binning_version or '无'}"
                ),
            }

            # Step 3: 反馈包（Q7-2）+ 假设矿机预筛（仅 train，holdout 绝不泄漏）
            feedback = await self._load_discovery_feedback()
            hints = await asyncio.to_thread(mine_hints, train_views)
            yield {
                "type": "step",
                "message": f"程序预筛完成：穷举谓词命中线索 {len(hints)} 条",
            }
            yield {"type": "step", "message": f"调用 LLM（model={settings.decision_model}）生成谓词假设…"}
            final = None
            async for ev in self._llm.agent_deep_learn_stream(
                symbolized_windows=train_payload,
                feedback=feedback,
                idle_timeout=settings.agent_deep_learn_idle_timeout,
                hints=hints,
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

            # Step 4: Q6 初筛审判（谓词执行 + lift 检验 + FDR 控制）+ 序列化
            yield {"type": "step", "message": "初筛审判中（谓词执行 + lift 检验 + FDR 控制）…"}
            discoveries_serialized = self._screen_and_serialize(
                list(getattr(final, "hypotheses", None) or []),
                holdout_views,
                binning_version,
                holdout_windows=holdout_windows,
            )

            logger.info(
                "Deep Learn(stream): 完成 | 有效假设={}",
                len(discoveries_serialized),
            )
            yield {
                "type": "done",
                "reasoning": getattr(final, "reasoning", None) or "",
                "discoveries": discoveries_serialized,
                "method": "LLM_DEEP",
                "snapshot_token": snapshot_token([w["id"] for w in windows_dicts]),
                "train_count": len(train_windows),
                "holdout_count": len(holdout_windows),
                "binning_version": binning_version,
            }

    async def deep_learn_pycluster(
        self,
        days_back: int | None = None,
        max_windows: int | None = None,
    ) -> dict:
        """P0-2 Python 聚类版深度发现（全程无 LLM，作为纯 LLM 版的确定性对照组）。

        流程：时间分层采样 → time_split → train 提特征 → KMeans 聚类 →
        每簇 outcome 多数投票定方向（NOISE 主导簇丢弃）→ 质心存 curve_features →
        evaluate_on_holdout 得样本外统计。同一份数据必得同一结果（random_state 固定）。

        Returns:
            {reasoning, discoveries, method='PY_CLUSTER', snapshot_token, train_count, holdout_count}
            discoveries 每条含 discovery_method/holdout_* 字段，供 commit 准入闸门复用。
        """
        if self._deep_learn_lock.locked():
            raise RuntimeError("已有深度分析任务正在执行，请稍后重试")

        async with self._deep_learn_lock:
            if days_back is None:
                days_back = settings.agent_deep_learn_days_back
            if max_windows is None:
                max_windows = settings.agent_deep_learn_max_windows

            logger.info(
                "Deep Learn(pycluster): 开始确定性聚类 | days_back={} | max_windows={}",
                days_back,
                max_windows,
            )

            windows_dicts = await self._fetch_deep_learn_windows(days_back, max_windows)
            if not windows_dicts:
                logger.warning("Deep Learn(pycluster): 无可用历史窗口，跳过分析")
                return {
                    "reasoning": "无可用历史窗口",
                    "discoveries": [],
                    "method": "PY_CLUSTER",
                    "snapshot_token": snapshot_token([]),
                    "train_count": 0,
                    "holdout_count": 0,
                }

            train_windows, holdout_windows = time_split(
                windows_dicts, settings.agent_deep_learn_holdout_ratio
            )

            # train 窗口提特征 → KMeans 聚类
            train_feats = [
                extract_features(
                    w.get("curve_up_pct") or [], w.get("curve_down_pct") or []
                )
                for w in train_windows
            ]
            if not train_feats:
                return {
                    "reasoning": "train 窗口为空",
                    "discoveries": [],
                    "method": "PY_CLUSTER",
                    "snapshot_token": snapshot_token([w["id"] for w in windows_dicts]),
                    "train_count": 0,
                    "holdout_count": len(holdout_windows),
                }
            matrix = np.vstack(train_feats)
            labels = cluster_windows(matrix, settings.agent_deep_learn_target_clusters)
            label_list = [int(x) for x in labels.tolist()]

            discoveries: list[dict] = []
            for cid in sorted(set(label_list)):
                member_idx = [i for i, lb in enumerate(label_list) if lb == cid]
                if not member_idx:
                    continue
                votes = Counter(
                    (train_windows[i].get("outcome") or "NOISE").upper()
                    for i in member_idx
                )
                direction, _ = votes.most_common(1)[0]
                if direction == "NOISE":
                    # NOISE 主导簇丢弃（无可交易方向）
                    continue
                centroid = np.mean(
                    np.vstack([train_feats[i] for i in member_idx]), axis=0
                )
                stats = evaluate_on_holdout(centroid, direction, holdout_windows)
                up_c = int(votes.get("UP", 0))
                down_c = int(votes.get("DOWN", 0))
                noise_c = int(votes.get("NOISE", 0))
                n_members = len(member_idx)
                curve_features = {
                    "_feature_vector": centroid.tolist(),
                    "cluster_id": cid,
                    "member_count": n_members,
                    "up_count": up_c,
                    "down_count": down_c,
                    "noise_count": noise_c,
                }
                discoveries.append({
                    "operation": "CREATE",
                    "target_pattern_id": None,
                    "pattern_name": f"PY-{direction}-C{cid}-n{n_members}",
                    "description": (
                        f"Python 聚类簇 {cid}：{n_members} 个 train 窗口，"
                        f"outcome 分布 UP={up_c}/DOWN={down_c}/NOISE={noise_c}，"
                        f"多数方向 {direction}（确定性特征 + KMeans，无 LLM）"
                    ),
                    "curve_features": curve_features,
                    "conditions": {},
                    "predicted_direction": direction,
                    "confidence_score": stats["ci_lower"],
                    "change_reason": (
                        f"PY_CLUSTER 自动发现（train {len(train_windows)} / "
                        f"holdout {len(holdout_windows)}）"
                    ),
                    "discovery_method": "PY_CLUSTER",
                    "holdout_win_rate": stats["win_rate"],
                    "holdout_sample_count": stats["sample_count"],
                    "holdout_ci_lower": stats["ci_lower"],
                })

            logger.info(
                "Deep Learn(pycluster): 完成 | train={} | holdout={} | 候选模式={}",
                len(train_windows),
                len(holdout_windows),
                len(discoveries),
            )
            return {
                "reasoning": (
                    f"确定性聚类：采样 {len(windows_dicts)} 窗口"
                    f"（train {len(train_windows)} / holdout {len(holdout_windows)}），"
                    f"KMeans 产出 {len(discoveries)} 个非 NOISE 候选模式"
                ),
                "discoveries": discoveries,
                "method": "PY_CLUSTER",
                "snapshot_token": snapshot_token([w["id"] for w in windows_dicts]),
                "train_count": len(train_windows),
                "holdout_count": len(holdout_windows),
            }

    async def commit_deep_learn(
        self,
        discoveries: list[dict],
    ) -> dict:
        """将用户确认的 discoveries 写入 pattern_memory（双轨准入闸门）。

        P2-1 字段校验：pattern_name/predicted_direction 必填非空（拦截流式半成品；
        旧轨另需 curve_features）。
        准入按轨分流（科学发现 Phase 2，宪法 Q6）：
        - 谓词轨（predicate 非空）：screen_verdict ACTIVE → status=ACTIVE 直上线；
          OBSERVE → status=EVOLVING 观察仓攒样本；其余拒绝。仅支持 CREATE。
          confidence_score 保留 LLM 主观先验，holdout_* 三列不填充，
          predicate/binning_version 落库（Q4/Q5）。
        - 旧轨（pycluster/LEGACY）：P0-3 样本外闸门，holdout_ci_lower > 0.5 且
          holdout_sample_count >= settings.agent_deep_learn_min_holdout_samples
          才写库；confidence_score 用 holdout_ci_lower 覆盖 LLM 主观值。
        P1-4：UPDATE 目标必须存在且 ACTIVE；写库失败不再静默 continue，收集入 failed。

        Args:
            discoveries: 用户确认后的发现列表（每条携带 discovery_method 与 holdout 统计）

        Returns:
            {"status": "ok", "written": int, "rejected": list[dict], "failed": list[dict]}
        """
        # 并发保护
        if self._deep_learn_lock.locked():
            raise RuntimeError("已有深度分析任务正在执行，请稍后重试")

        async with self._deep_learn_lock:
            rejected: list[dict] = []
            failed: list[dict] = []
            if not discoveries:
                logger.info("Commit Deep Learn: 无 discoveries 待写入")
                return {"status": "ok", "written": 0, "rejected": rejected, "failed": failed}

            logger.info(
                "Commit Deep Learn: 开始写入 {} 条发现",
                len(discoveries),
            )

            min_samples = settings.agent_deep_learn_min_holdout_samples

            # 记录本批次已创建的模式指纹，用于批内去重
            batch_created_patterns: list[dict] = []
            written_count = 0

            # P0-2：读取现有 ACTIVE + 写入 pattern_memory 全程持唯一写锁，
            # 与 Validate / Evolve 串行，保证去重判定 / active_cap 计数 / 写入的原子性。
            # LLM 调用已在本方法外完成，锁粒度不含 LLM。
            async with self._pattern_write_lock, async_session_factory() as session:
                # 读取现有 ACTIVE 模式（去重 + UPDATE 目标校验），与写入同一会话/同一锁
                existing_patterns: list[dict] = []
                active_ids: set[int] = set()
                _existing_result = await session.execute(
                    select(PatternMemory).where(PatternMemory.status == "ACTIVE")
                )
                for p in _existing_result.scalars().all():
                    existing_patterns.append({
                        "id": p.id,
                        "curve_features": p.curve_features,
                        "predicted_direction": p.predicted_direction,
                    })
                    active_ids.add(p.id)

                for idx, d in enumerate(discoveries, 1):
                    name = d.get("pattern_name")
                    operation = d.get("operation", "CREATE")
                    direction = d.get("predicted_direction")
                    curve_features = d.get("curve_features")
                    predicate = d.get("predicate")

                    # P2-1: 必填字段校验（拦截流式半成品）
                    # 谓词轨（科学发现 Phase 2）无特征向量，不要求 curve_features
                    if not name or not direction or (not predicate and not curve_features):
                        rejected.append({
                            "name": name,
                            "reason": "字段残缺（pattern_name/predicted_direction 必填非空；旧轨另需 curve_features）",
                        })
                        logger.warning(
                            "Commit Deep Learn: {}/{} 字段残缺被拒 | '{}'",
                            idx, len(discoveries), name,
                        )
                        continue

                    # 准入闸门按轨分流：谓词轨走 Q6 双轨裁决；旧轨走 P0-3 样本外闸门
                    initial_status: str | None = None
                    ci_lower = d.get("holdout_ci_lower")
                    sample_count = d.get("holdout_sample_count")
                    if predicate:
                        # 谓词轨（Q6）：screen_verdict ACTIVE → ACTIVE 直上线；
                        # OBSERVE → EVOLVING 观察仓攒样本；其余（REJECT/缺失）拒绝
                        verdict = d.get("screen_verdict")
                        if verdict == VERDICT_ACTIVE:
                            initial_status = "ACTIVE"
                        elif verdict == VERDICT_OBSERVE:
                            initial_status = "EVOLVING"
                        else:
                            rejected.append({
                                "name": name,
                                "reason": (
                                    f"初筛裁决 {verdict or '缺失'} 不予写库"
                                    f"（原因：{d.get('screen_reject_reason') or '无'}）"
                                ),
                            })
                            logger.info(
                                "Commit Deep Learn: {}/{} 初筛裁决未过被拒 | '{}' verdict={}",
                                idx, len(discoveries), name, verdict,
                            )
                            continue
                        # 谓词假设无 UPDATE 语义（Phase 2 仅 CREATE）
                        if operation != "CREATE":
                            rejected.append({
                                "name": name,
                                "reason": "谓词轨仅支持 CREATE（假设无 UPDATE 语义）",
                            })
                            continue
                    else:
                        # 旧轨（P0-3 样本外准入闸门）：pycluster/LEGACY 保留
                        if ci_lower is None or sample_count is None:
                            rejected.append({
                                "name": name,
                                "reason": "缺少 holdout 统计，无法通过准入闸门",
                            })
                            continue
                        if not (ci_lower > 0.5 and sample_count >= min_samples):
                            rejected.append({
                                "name": name,
                                "reason": (
                                    f"未过准入闸门（holdout_ci_lower={ci_lower:.3f} 需>0.5，"
                                    f"holdout_sample_count={sample_count} 需>={min_samples}）"
                                ),
                            })
                            logger.info(
                                "Commit Deep Learn: {}/{} 未过准入闸门被拒 | '{}'",
                                idx, len(discoveries), name,
                            )
                            continue

                    try:
                        # CREATE 前去重检查（谓词轨跳过：指纹体系基于特征向量/结构化
                        # key，对谓词模式不适用；谓词精确重复由用户勾选时把关）
                        if operation == "CREATE" and not predicate and settings.agent_dedup_enabled:
                            all_patterns_for_dedup = existing_patterns + batch_created_patterns
                            existing_id = detect_duplicate_pattern(
                                curve_features, direction, all_patterns_for_dedup,
                            )
                            if existing_id is not None:
                                logger.warning(
                                    "Commit Deep Learn: 疑似重复模式 '{}'，与 id={} 指纹一致 | "
                                    "auto_downgrade={}",
                                    name, existing_id, settings.agent_dedup_auto_downgrade,
                                )
                                if settings.agent_dedup_auto_downgrade:
                                    operation = "UPDATE"
                                    d["target_pattern_id"] = existing_id
                                    d["change_reason"] = (
                                        f"去重降级: 与 id={existing_id} 指纹一致，原 CREATE 转为 UPDATE"
                                    )
                                else:
                                    rejected.append({
                                        "name": name,
                                        "reason": f"疑似重复（与 id={existing_id} 指纹一致）",
                                    })
                                    continue

                        # P1-4: UPDATE 目标必须存在且 ACTIVE
                        if operation == "UPDATE":
                            target_id = d.get("target_pattern_id")
                            if target_id is None or target_id not in active_ids:
                                failed.append({
                                    "name": name,
                                    "reason": f"UPDATE 目标 id={target_id} 不存在或非 ACTIVE",
                                })
                                logger.error(
                                    "Commit Deep Learn: {}/{} UPDATE 目标非法 | '{}' target={}",
                                    idx, len(discoveries), name, target_id,
                                )
                                continue

                        async with session.begin_nested():
                            if predicate:
                                # 谓词轨：confidence 保留 LLM 主观先验（程序审判证据在
                                # predicate/screen_*），holdout_* 三列不填充；
                                # status 由 Q6 闸门决定（ACTIVE 直上线 / EVOLVING 观察仓）
                                pattern_data = {
                                    "pattern_name": name,
                                    "description": d.get("description"),
                                    "curve_features": {},
                                    "conditions": {},
                                    "predicted_direction": direction,
                                    "confidence_score": d.get("confidence_score") or 0.5,
                                    "change_reason": d.get("change_reason", ""),
                                    "discovery_method": d.get("discovery_method", "LLM_DEEP"),
                                    "status": initial_status,
                                    "predicate": predicate,
                                    "binning_version": d.get("binning_version"),
                                }
                            else:
                                pattern_data = {
                                    "pattern_name": name,
                                    "description": d.get("description"),
                                    "curve_features": curve_features,
                                    "conditions": d.get("conditions"),
                                    "predicted_direction": direction,
                                    # P0-3: confidence 用 holdout_ci_lower 覆盖 LLM 主观值
                                    "confidence_score": ci_lower,
                                    "change_reason": d.get("change_reason", ""),
                                    "discovery_method": d.get("discovery_method", "LEGACY"),
                                    "holdout_win_rate": d.get("holdout_win_rate"),
                                    "holdout_sample_count": sample_count,
                                    "holdout_ci_lower": ci_lower,
                                }
                            if operation == "UPDATE":
                                pattern_data["target_pattern_id"] = d.get("target_pattern_id")

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
                                idx, len(discoveries), operation, name,
                            )
                    except Exception as exc:
                        # P1-4: 不再静默 continue，收集 failed
                        failed.append({"name": name, "reason": str(exc)})
                        logger.error(
                            "Commit Deep Learn: {}/{} 写入失败 | {} '{}' | error={}",
                            idx, len(discoveries), operation, name, str(exc),
                        )
                        continue

                await session.commit()

            logger.info(
                "Commit Deep Learn: 完成 | 写入 {} · 拒绝 {} · 失败 {}",
                written_count, len(rejected), len(failed),
            )
            return {
                "status": "ok",
                "written": written_count,
                "rejected": rejected,
                "failed": failed,
            }

    # ======================================================================
    # Predict 阶段（Req 3.2 / 3.3 / 3.4 / 3.5 / 3.6 / 10.1 / 10.2 / 10.3 / 11.1）
    # ======================================================================

    async def predict(
        self, window_end_ms: int, current_curve: list[dict]
    ) -> AgentPrediction | None:
        """
        预测阶段（科学发现宪法第八条，Phase 3）：程序执行谓词匹配（确定性），
        LLM 降级为仲裁者——仅在多模式命中冲突时消歧，零命中输出 NO_TRADE。

        流程（design.md 第八条操作化规则 1~7）：
        1. 候选集：仅 predicate 非空的 ACTIVE ∪ EVOLVING 模式（predicate 为空的
           旧模式不删库、live 统计冻结，由 Evolve/手动逐步 RETIRE）；
           候选为空 → NO_TRADE 冷启动（不调 LLM）
        2. 仪器精度对齐：按 binning_version 分组候选，加载对应版本快照，每组
           用其出生版本符号化当前窗口；版本缺失或快照查不到的组跳过并记 warning
        3. 程序谓词执行：evaluate_predicate 逐模式判定 → 命中集 PredicateHit
        4. 命中解析四分支（resolve_predicate_hits，顺序不可换）：
           零命中 → NO_TRADE；单命中 → 直接采用；多命中同向 → 取证据最强者；
           多命中异向 → 调 LLM 仲裁（agent_arbitrate）
        5. confidence：程序路径由 pattern_confidence 合成（live win_rate 优先，
           样本不足回退 LLM 先验）；仲裁路径取 LLM 对选定模式的把握
        6. entry_timing：程序命中路径恒 NOW（谓词命中即形态已确认）；仲裁路径
           由 LLM 评估；剩余时间保护（is_prediction_stale）不变
        7. 写 AgentPrediction + 交易门控（_write_prediction_and_trade 不变；
           EVOLVING 模式命中仅落库攒 live 样本，二次确认拦截实盘下单）

        Args:
            window_end_ms: 当前窗口结束时间戳（毫秒），用于计算剩余时间与匹配窗口 ID
            current_curve: 当前窗口实时采样点（[{t, up_pct, down_pct, btc_price,
                trade_volume}, ...]），由 Scheduler dispatch 时从 _pm_history 切片传入；
                btc_price/trade_volume 缺失的点被符号化层跳过，谓词按防御求值 False

        Returns:
            写入的 AgentPrediction 实例；极端异常下返回 None

        设计约束：
        - LLM 调用（仅冲突仲裁分支）在事务开启前完成（避免长事务占用连接池）
        - 无静默降级：所有失败路径明确记录原因（规则 3）
        """
        # ========== Step 1：读取谓词模式候选（只读会话）==========
        candidates: list[dict] = []

        async with async_session_factory() as session:
            pattern_result = await session.execute(select(PatternMemory))
            raw_patterns = pattern_result.scalars().all()

            for p in raw_patterns:
                # 规则 1：仅谓词模式参与 Predict；EVOLVING 观察态参与匹配以积累
                # live 样本，交易门控二次确认 status==ACTIVE 才会真正下单
                if not p.predicate or p.status not in ("ACTIVE", "EVOLVING"):
                    continue
                candidates.append({
                    "id": p.id,
                    "pattern_name": p.pattern_name,
                    "description": p.description,
                    "predicted_direction": p.predicted_direction,
                    "predicate": p.predicate,
                    "binning_version": p.binning_version,
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

        logger.info(
            "Predict: 数据准备完成 | 谓词候选={} | 曲线点数={} | window_end_ms={}",
            len(candidates),
            len(current_curve),
            window_end_ms,
        )

        # ========== Step 2：冷启动检查（规则 1：候选为空 → NO_TRADE）==========
        if not candidates:
            logger.info("Predict: 冷启动——谓词模式候选为 0，直接输出 NO_TRADE（不调用 LLM）")
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning="谓词模式库为空，等待发现积累",
                sentiment_window_id=sentiment_window_id,
            )

        # ========== Step 3：仪器精度对齐 + 程序谓词执行（规则 2/3）==========
        by_version: dict[str, list[dict]] = defaultdict(list)
        for c in candidates:
            if c["binning_version"]:
                by_version[c["binning_version"]].append(c)
            else:
                logger.warning(
                    "Predict: 模式 id={} name='{}' 缺 binning_version，跳过",
                    c["id"],
                    c["pattern_name"],
                )

        snapshots_by_version = await self._load_binning_snapshots_by_versions(
            set(by_version)
        )
        window_dict = self._current_window_dict(window_end_ms, current_curve)

        hits: list[PredicateHit] = []
        views_by_version: dict[str, WindowView] = {}
        for version, group in by_version.items():
            snaps = snapshots_by_version.get(version)
            if not snaps:
                logger.warning(
                    "Predict: binning_snapshots 查不到版本 {}，跳过 {} 个模式",
                    version,
                    len(group),
                )
                continue
            view = build_window_view(window_dict, snaps)
            views_by_version[version] = view
            for c in group:
                try:
                    if evaluate_predicate(c["predicate"], view):
                        hits.append(PredicateHit(
                            pattern_id=c["id"],
                            pattern_name=c["pattern_name"],
                            direction=c["predicted_direction"],
                            win_rate=c["win_rate"],
                            sample_count=c["sample_count"],
                            prior_confidence=c["confidence_score"],
                        ))
                except Exception as exc:
                    # 库内谓词写入时已过 DSL 校验，此处异常属数据腐败/程序缺陷；
                    # 单模式失败不阻塞整体判定，显式记录（无静默降级）
                    logger.warning(
                        "Predict: 谓词执行异常 | id={} name='{}' | error={}",
                        c["id"],
                        c["pattern_name"],
                        exc,
                    )

        branch, chosen = resolve_predicate_hits(hits)
        logger.info(
            "Predict: 谓词匹配完成 | 候选={} | 命中={} | 分支={}",
            len(candidates),
            len(hits),
            branch,
        )

        # ========== Step 4 分支一：零命中 → NO_TRADE（不调 LLM）==========
        if branch == HIT_NONE:
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning=f"谓词零命中（{len(candidates)} 个候选模式无一命中当前窗口）",
                sentiment_window_id=sentiment_window_id,
            )

        # ========== Step 4 分支二/三：单命中 / 多命中同向 → 程序直采 ==========
        if branch in (HIT_SINGLE, HIT_CONCORDANT) and chosen is not None:
            confidence = pattern_confidence(
                chosen.win_rate,
                chosen.sample_count,
                chosen.prior_confidence,
                min_samples=settings.agent_min_pattern_samples,
            )
            branch_label = (
                "单命中" if branch == HIT_SINGLE else f"多命中同向({len(hits)} 个取最强)"
            )
            reasoning = (
                f"谓词{branch_label}：{chosen.pattern_name}(id={chosen.pattern_id}) "
                f"方向 {chosen.direction} | live 样本 {chosen.sample_count} "
                f"胜率 {chosen.win_rate:.4f} | confidence={confidence:.4f}"
            )
            return await self._write_prediction_and_trade(
                predicted_direction=chosen.direction,
                matched_pattern_id=chosen.pattern_id,
                matched_pattern_name=chosen.pattern_name,
                confidence=confidence,
                # 规则 7：程序命中路径恒 NOW（谓词命中即形态已确认）
                entry_timing="NOW",
                reasoning=reasoning,
                sentiment_window_id=sentiment_window_id,
            )

        # ========== Step 4 分支四：多命中异向 → LLM 仲裁（规则 6）==========
        conflict_ids = {h.pattern_id for h in hits}
        conflict_candidates = [c for c in candidates if c["id"] in conflict_ids]
        # 仲裁展示视图：取已符号化版本中快照最新者（最新仪器精度）
        latest_version = max(
            views_by_version.keys(),
            key=lambda v: next(iter(snapshots_by_version[v].values())).created_at_epoch,
        )
        window_payload = self._view_to_payload(views_by_version[latest_version])

        remaining_seconds = max(0, (window_end_ms - int(time.time() * 1000)) // 1000)
        arbitrate_candidates = [
            {
                "id": c["id"],
                "pattern_name": c["pattern_name"],
                "description": c["description"],
                "predicted_direction": c["predicted_direction"],
                "predicate": c["predicate"],
                "win_rate": c["win_rate"],
                "sample_count": c["sample_count"],
            }
            for c in conflict_candidates
        ]

        try:
            arb_output = await self._llm.agent_arbitrate(
                window_payload=window_payload,
                candidates=arbitrate_candidates,
                remaining_seconds=remaining_seconds,
                timeout=settings.agent_llm_timeouts["PREDICT"],
            )
        except Exception as exc:
            # 规则 6：LLM 失败/超时/重试耗尽 → NO_TRADE（无静默降级）
            error_msg = f"LLM 仲裁调用失败: {type(exc).__name__}: {exc}"
            logger.error("Predict: {}，落库 NO_TRADE", error_msg)
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
            "Predict: 仲裁返回 | selected_pattern_id={} | confidence={:.4f} | 冲突数={}",
            arb_output.selected_pattern_id,
            arb_output.confidence,
            len(conflict_candidates),
        )

        # 规则 6：程序 HARD 校验——选定 id 必须在冲突候选集合内（不受验证
        # 开关影响，LLM 无权发明候选外模式；越界即仲裁无效降级 NO_TRADE）
        selected = next(
            (c for c in conflict_candidates if c["id"] == arb_output.selected_pattern_id),
            None,
        )
        if arb_output.selected_pattern_id is not None and selected is None:
            logger.error(
                "Predict: 仲裁选定 id={} 不在冲突候选集合，降级 NO_TRADE",
                arb_output.selected_pattern_id,
            )
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning=f"仲裁选定 id={arb_output.selected_pattern_id} 越界（非冲突候选）",
                sentiment_window_id=sentiment_window_id,
                skip_trade_reason="LLM 输出语义验证失败",
            )

        # 语义验证（soft 告警落日志；hard 与上方程序兜底冗余双保险）
        if settings.agent_llm_validation_enabled:
            hard_failures, soft_warnings = validate_arbitrate_output(
                arb_output, conflict_ids
            )
            for w in soft_warnings:
                logger.warning("Predict 仲裁验证 SOFT: {}", w)
            if hard_failures:
                for f in hard_failures:
                    logger.error("Predict 仲裁验证 HARD: {}", f)
                return await self._write_prediction_and_trade(
                    predicted_direction="NO_TRADE",
                    matched_pattern_id=None,
                    matched_pattern_name=None,
                    confidence=0.0,
                    entry_timing="SKIP",
                    reasoning=f"仲裁输出语义验证失败: {hard_failures}",
                    sentiment_window_id=sentiment_window_id,
                    skip_trade_reason="LLM 输出语义验证失败",
                )

        # 仲裁放弃：冲突不可调和 → NO_TRADE（放弃的决定无时效问题，不做 stale 检查）
        if selected is None:
            return await self._write_prediction_and_trade(
                predicted_direction="NO_TRADE",
                matched_pattern_id=None,
                matched_pattern_name=None,
                confidence=0.0,
                entry_timing="SKIP",
                reasoning=(
                    f"仲裁放弃（冲突 {len(conflict_candidates)} 模式不可调和）："
                    f"{arb_output.reasoning}"
                ),
                sentiment_window_id=sentiment_window_id,
                skip_trade_reason="仲裁放弃交易",
            )

        # 规则 7：剩余时间保护（仲裁路径有 LLM 延迟，保持不变）
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

        # 仲裁选定：direction 由程序从模式 predicted_direction 推导（规则 6）
        return await self._write_prediction_and_trade(
            predicted_direction=selected["predicted_direction"],
            matched_pattern_id=selected["id"],
            matched_pattern_name=selected["pattern_name"],
            confidence=arb_output.confidence,
            entry_timing=arb_output.entry_timing,
            reasoning=(
                f"仲裁选定（冲突 {len(conflict_candidates)} 模式）：{arb_output.reasoning}"
            ),
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
                    pred.skip_trade_reason = f"模式非 ACTIVE 不交易（status={matched_pattern_status}）"
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

    async def _diagnose_pattern_deaths(
        self, session: AsyncSession, evolve_phase_id: str
    ) -> dict:
        """双轨死因巡检（宪法 Q7-1）：对谓词轨存活模式做 live lift 诊断，判死即 RETIRE。

        在 evolve() 的 LLM 操作应用前执行（程序确定性死因优先于 LLM 创造性进化）。
        候选：status ∈ {ACTIVE, EVOLVING} 且 predicate 非空的模式（谓词轨；
        旧轨 predicate 为空的模式统计已冻结，由既有上限淘汰路径管理）。

        每模式流水线：全部已结算命中（is_correct 非空，时间升序）→ 命中窗口
        ±3 天局部基准池（pooled_local_baseline，Q6-b）→ live_lift_summary
        （recent 20 次 lift/CI + 前缀峰值）→ diagnose_death 双轨裁决：
        - SPURIOUS（假规律：从未显著）→ RETIRE + 全量细节进负样本反馈池
        - EXPIRED（过期规律：曾显著后衰减）→ RETIRE + 记录存活期 + 建议再发现
        - ALIVE（或命中 < MIN_DEATH_HITS 样本不足）→ 不动

        单模式失败仅回滚该 savepoint，不中断其余巡检（无静默降级：记 error）。
        """
        stats: dict = {
            "checked": 0,
            "skipped": 0,
            "spurious": [],
            "expired": [],
        }

        # Step 1: 候选模式（谓词轨存活者）
        stmt = select(PatternMemory).where(
            PatternMemory.status.in_(["ACTIVE", "EVOLVING"]),
            PatternMemory.predicate.isnot(None),
        )
        candidates = (await session.execute(stmt)).scalars().all()
        if not candidates:
            return stats
        pattern_ids = [p.id for p in candidates]

        # Step 2: 批量取全部已结算命中（join 窗口拿 start_time；孤儿预测自动排除）
        hit_stmt = (
            select(
                AgentPrediction.matched_pattern_id,
                AgentPrediction.is_correct,
                AgentPrediction.prediction_time,
                SentimentWindow.start_time,
            )
            .join(
                SentimentWindow,
                AgentPrediction.sentiment_window_id == SentimentWindow.id,
            )
            .where(
                AgentPrediction.matched_pattern_id.in_(pattern_ids),
                AgentPrediction.is_correct.isnot(None),
            )
            .order_by(
                AgentPrediction.matched_pattern_id,
                AgentPrediction.prediction_time.asc(),
            )
        )
        hit_rows = (await session.execute(hit_stmt)).all()
        hits_by_pattern: dict[int, list] = defaultdict(list)
        for pid, is_correct, _pred_time, window_start in hit_rows:
            hits_by_pattern[pid].append((bool(is_correct), int(window_start)))

        # Step 3: 局部基准原料——覆盖全部命中窗口 ±3 天的 outcome 非空窗口
        all_hit_times = [st for rows in hits_by_pattern.values() for _, st in rows]
        all_windows: list[dict] = []
        if all_hit_times:
            span_ms = 3 * 86_400_000
            win_stmt = select(
                SentimentWindow.start_time, SentimentWindow.outcome
            ).where(
                SentimentWindow.outcome.isnot(None),
                SentimentWindow.start_time >= min(all_hit_times) - span_ms,
                SentimentWindow.start_time <= max(all_hit_times) + span_ms,
            )
            all_windows = [
                {"start_time": st, "outcome": oc}
                for st, oc in (await session.execute(win_stmt)).all()
            ]

        # Step 4: 逐模式诊断（命中 < MIN_DEATH_HITS 直接跳过，样本不足不判死）
        now = datetime.now(tz=timezone.utc)
        for pattern in candidates:
            hits = hits_by_pattern.get(pattern.id, [])
            if len(hits) < MIN_DEATH_HITS:
                stats["skipped"] += 1
                continue
            stats["checked"] += 1

            hits_chrono = [ok for ok, _ in hits]
            hit_start_times = [st for _, st in hits]
            base_events, base_total = pooled_local_baseline(
                hit_start_times, all_windows, pattern.predicted_direction
            )
            summary = live_lift_summary(hits_chrono, base_events, base_total)
            verdict = diagnose_death(
                summary.recent.lift,
                summary.recent.ci_lower,
                summary.recent.ci_upper,
                summary.peak_lift,
                summary.hit_count,
            )
            if verdict == DEATH_ALIVE:
                continue

            created_at = pattern.created_at
            if created_at is not None and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            lifespan_days = (
                round((now - created_at).total_seconds() / 86_400.0, 2)
                if created_at is not None
                else None
            )
            lr = summary.recent
            if verdict == DEATH_SPURIOUS:
                reason = (
                    f"死因判定 SPURIOUS（假规律）：{summary.hit_count} 次 live 命中 "
                    f"recent lift {lr.lift:.2f} CI[{lr.ci_lower:.2f},{lr.ci_upper:.2f}] "
                    f"覆盖 1，历史峰值 {summary.peak_lift:.2f} 从未显著；"
                    f"全量细节反馈发现器（波普尔排除）"
                )
            else:
                reason = (
                    f"死因判定 EXPIRED（过期规律）：历史峰值 lift "
                    f"{summary.peak_lift:.2f} 曾显著，最近 {MIN_DEATH_HITS} 次命中 "
                    f"lift {lr.lift:.2f} < 1.1（regime 变迁）；建议触发新一轮发现"
                )

            try:
                async with session.begin_nested():  # savepoint：失败仅回滚此模式
                    await self.apply_pattern_change(
                        session=session,
                        operation="RETIRE",
                        pattern_data={
                            "target_pattern_id": pattern.id,
                            "change_reason": reason,
                            "death_cause": verdict,
                            "lifespan_days": lifespan_days,
                        },
                        phase="EVOLVE",
                        evolve_phase_id=evolve_phase_id,
                    )
                stats["spurious" if verdict == DEATH_SPURIOUS else "expired"].append(
                    {"id": pattern.id, "name": pattern.pattern_name}
                )
                logger.info(
                    "Evolve: 死因巡检 RETIRE 模式 id={} '{}' | verdict={} | "
                    "hits={} recent_lift={:.2f} peak_lift={:.2f} lifespan={}天",
                    pattern.id,
                    pattern.pattern_name,
                    verdict,
                    summary.hit_count,
                    lr.lift,
                    summary.peak_lift,
                    lifespan_days,
                )
            except Exception as exc:
                logger.error(
                    "Evolve: 死因巡检 RETIRE 失败 | id={} verdict={} | "
                    "error_type={} | error={}",
                    pattern.id,
                    verdict,
                    type(exc).__name__,
                    str(exc),
                )
                continue

        if stats["expired"]:
            logger.warning(
                "Evolve: {} 个模式死于 EXPIRED（regime 变迁），建议触发新一轮 Deep Learn "
                "发现（遵守发现预算：轮次间隔 ≥1 天）| ids={}",
                len(stats["expired"]),
                [x["id"] for x in stats["expired"]],
            )
        return stats

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

            # P1-3：反思查询对齐触发阈值 + 仅取已验证预测。
            # limit 由 agent_evolve_interval 提升至 agent_evolve_min_new_samples
            # （与 Evolve 触发阈值一致）；并过滤 is_correct IS NULL 的未验证预测，
            # 避免未验证证据污染反思。
            reflect_limit = settings.agent_evolve_min_new_samples
            pred_stmt = (
                select(AgentPrediction)
                .where(AgentPrediction.is_correct.isnot(None))
                .order_by(AgentPrediction.prediction_time.desc())
                .limit(reflect_limit)
            )
            pred_result = await session.execute(pred_stmt)
            recent_preds = pred_result.scalars().all()
            if len(recent_preds) < reflect_limit:
                logger.warning(
                    "Evolve: 已验证预测仅 {} 条 < 反思目标 {} 条，"
                    "证据不足但继续（不阻断）",
                    len(recent_preds), reflect_limit,
                )

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
        # P0-2：Step 4 写 pattern_memory，持唯一写锁与 Validate / commit_deep_learn 串行。
        async with self._pattern_write_lock, async_session_factory() as session:
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

            # ========== Q7-1：双轨死因巡检（程序确定性，先于一切 LLM 操作）==========
            death_stats = await self._diagnose_pattern_deaths(session, evolve_phase_id)
            if death_stats["checked"] or death_stats["skipped"]:
                logger.info(
                    "Evolve: 死因巡检完成 | 诊断={} 样本不足跳过={} | "
                    "SPURIOUS={} EXPIRED={}",
                    death_stats["checked"],
                    death_stats["skipped"],
                    len(death_stats["spurious"]),
                    len(death_stats["expired"]),
                )

            # ========== P0-1：EVOLVING 观察态晋升/淘汰 ==========
            # 达到最小 holdout 样本量后，用 live 样本的 Wilson 置信下界裁决：
            # 下界 > 0.5 → 晋升 ACTIVE（confidence 用下界覆盖）；否则 RETIRE。
            # 未达样本量则保持 EVOLVING 继续观察。
            promoted_count = 0
            evolving_retired_count = 0
            evolving_result = await session.execute(
                select(PatternMemory).where(PatternMemory.status == "EVOLVING")
            )
            evolving_patterns = evolving_result.scalars().all()
            for ep in evolving_patterns:
                if ep.sample_count < settings.agent_deep_learn_min_holdout_samples:
                    continue
                lb = wilson_lower_bound(ep.correct_count, ep.sample_count)
                try:
                    async with session.begin_nested():  # savepoint：失败仅回滚此子事务
                        if lb > 0.5:
                            before_snapshot = self._pattern_to_snapshot(ep)
                            ep.status = "ACTIVE"
                            ep.confidence_score = lb
                            after_snapshot = self._pattern_to_snapshot(ep)
                            session.add(PatternChangeLog(
                                pattern_id=ep.id,
                                change_type="UPDATE",
                                phase="EVOLVE",
                                before_snapshot=before_snapshot,
                                after_snapshot=after_snapshot,
                                change_reason=(
                                    f"EVOLVING 晋升 ACTIVE：live 样本 {ep.sample_count} "
                                    f"（>= {settings.agent_deep_learn_min_holdout_samples}），"
                                    f"Wilson 下界 {lb:.3f} > 0.5"
                                ),
                                evolve_phase_id=evolve_phase_id,
                            ))
                            promoted_count += 1
                            logger.info(
                                "Evolve: EVOLVING 模式 id={} '{}' 晋升 ACTIVE | "
                                "sample={} wilson_lb={:.3f}",
                                ep.id, ep.pattern_name, ep.sample_count, lb,
                            )
                        else:
                            # Q7-1：攒够样本但 Wilson 下界未过 0.5 = 从未显著，
                            # 按假规律（SPURIOUS）归档，全量细节进负样本反馈池
                            ep_created = ep.created_at
                            if ep_created is not None and ep_created.tzinfo is None:
                                ep_created = ep_created.replace(tzinfo=timezone.utc)
                            ep_lifespan = (
                                round(
                                    (datetime.now(tz=timezone.utc) - ep_created)
                                    .total_seconds() / 86_400.0,
                                    2,
                                )
                                if ep_created is not None
                                else None
                            )
                            await self.apply_pattern_change(
                                session=session,
                                operation="RETIRE",
                                pattern_data={
                                    "target_pattern_id": ep.id,
                                    "change_reason": (
                                        f"EVOLVING 淘汰：live 样本 {ep.sample_count} "
                                        f"（>= {settings.agent_deep_learn_min_holdout_samples}），"
                                        f"Wilson 下界 {lb:.3f} 未过 0.5"
                                    ),
                                    "death_cause": DEATH_SPURIOUS,
                                    "lifespan_days": ep_lifespan,
                                },
                                phase="EVOLVE",
                                evolve_phase_id=evolve_phase_id,
                            )
                            evolving_retired_count += 1
                            logger.info(
                                "Evolve: EVOLVING 模式 id={} '{}' 淘汰 | "
                                "sample={} wilson_lb={:.3f}",
                                ep.id, ep.pattern_name, ep.sample_count, lb,
                            )
                except Exception as exc:
                    failed_count += 1
                    logger.error(
                        "Evolve: EVOLVING 晋升/淘汰 id={} 失败 | error={}",
                        ep.id, exc,
                    )
                    continue

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
                                # P0-1：Evolve 新建模式先入 EVOLVING 观察态，不直接参与交易，
                                # 积累 live 样本达标后由后续 Evolve 晋升 ACTIVE。
                                "status": "EVOLVING",
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
            "冷启动跳过={} | 失败={} | EVOLVING晋升={} | EVOLVING淘汰={}",
            evolve_phase_id,
            applied_count,
            skipped_retain,
            skipped_cold_start,
            failed_count,
            promoted_count,
            evolving_retired_count,
        )

    # ======================================================================
    # 降级 / 对账辅助（P2-3 / P1-1）
    # ======================================================================

    async def write_no_trade(
        self, window_end_ms: int, reason: str
    ) -> AgentPrediction | None:
        """将降级场景落库为 NO_TRADE 预测（无静默降级，规则 3）。

        P2-3：当 PREDICT 事件在队列中等待过久、current_curve 快照陈旧时，
        不喂陈旧曲线给 LLM，而是直接落库一条 NO_TRADE 预测并标注原因。

        Args:
            window_end_ms: 当前窗口结束时间戳（毫秒）
            reason: 降级原因（写入 reasoning 与 skip_trade_reason）

        Returns:
            写入的 AgentPrediction 实例；极端异常返回 None
        """
        sentiment_window_id: int | None = None
        async with async_session_factory() as session:
            sw_stmt = select(SentimentWindow.id).where(
                SentimentWindow.end_time == window_end_ms
            )
            sentiment_window_id = (
                await session.execute(sw_stmt)
            ).scalar_one_or_none()

        logger.warning(
            "Predict: 降级为 NO_TRADE | window_end_ms={} | reason={}",
            window_end_ms,
            reason,
        )
        return await self._write_prediction_and_trade(
            predicted_direction="NO_TRADE",
            matched_pattern_id=None,
            matched_pattern_name=None,
            confidence=0.0,
            entry_timing="SKIP",
            reasoning=reason,
            sentiment_window_id=sentiment_window_id,
            skip_trade_reason=reason,
        )

    async def reconcile_pending_predictions(self) -> int:
        """启动对账：回填进程重启后遗漏的未验证预测（P1-1）。

        调度队列非持久，进程重启后待验证预测（is_correct IS NULL）无人回填；
        archiver 又不会重发已归档窗口的 WINDOW_ARCHIVED，导致预测永久孤儿。
        本方法在 lifespan 启动阶段一次性扫描：对每条超过一个窗口时长的
        未验证预测，找到覆盖它的已归档（outcome 非空）SentimentWindow，调 validate 回填。
        找不到已归档窗口的（窗口从未归档，见 P1-2）本轮不处理。

        Returns:
            本次回填的预测总数
        """
        # 仅处理已超过一个窗口时长的未验证预测，避免误碰仍在进行中的窗口
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(minutes=5)
        validated_total = 0

        async with async_session_factory() as session:
            pend_stmt = (
                select(AgentPrediction.prediction_time)
                .where(
                    AgentPrediction.is_correct.is_(None),
                    AgentPrediction.prediction_time <= cutoff,
                )
                .order_by(AgentPrediction.prediction_time.asc())
            )
            pend_times = (await session.execute(pend_stmt)).scalars().all()
            if not pend_times:
                logger.info("Reconcile: 无待回填的未验证预测")
                return 0

            # 为每条待验证预测找到覆盖它的已归档窗口（outcome 非空），去重
            window_ids: set[int] = set()
            windows_to_validate: list[SentimentWindow] = []
            for pt in pend_times:
                pt_ms = int(pt.timestamp() * 1000)
                win_stmt = (
                    select(SentimentWindow)
                    .where(
                        SentimentWindow.start_time <= pt_ms,
                        SentimentWindow.end_time >= pt_ms,
                        SentimentWindow.outcome.isnot(None),
                    )
                    .order_by(SentimentWindow.end_time.desc())
                    .limit(1)
                )
                win = (await session.execute(win_stmt)).scalar_one_or_none()
                if win is not None and win.id not in window_ids:
                    window_ids.add(win.id)
                    windows_to_validate.append(win)

        logger.info(
            "Reconcile: 发现 {} 条未验证预测，匹配到 {} 个已归档窗口待回填",
            len(pend_times),
            len(windows_to_validate),
        )

        for win in windows_to_validate:
            try:
                ids = await self.validate(win)
                validated_total += len(ids)
            except Exception as exc:
                # 无静默降级：单窗口回填失败不阻断其余，但记录错误
                logger.error(
                    "Reconcile: 窗口 id={} 回填失败 | error={}", win.id, exc
                )
                continue

        logger.info("Reconcile: 完成，共回填 {} 条预测", validated_total)
        return validated_total
