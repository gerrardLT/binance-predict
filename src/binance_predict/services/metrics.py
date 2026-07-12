"""
Agent 运行时指标收集器 —— 内存有界、O(1) 非阻塞

为系统提供轻量级可观测性底座（Plan 组 A），收集：
- 阶段（PREDICT / VALIDATE / LEARN / EVOLVE）执行耗时与成功率
- LLM 调用 token 用量与估算成本
- 交易决策统计
- 调度队列深度

所有数据存储在 collections.deque(maxlen=10000) 中，内存有界。
对外暴露 get_snapshot() 返回聚合统计字典，由 API 端点按需返回。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from loguru import logger

# ============================================================
# 内部记录结构
# ============================================================


@dataclass(frozen=True, slots=True)
class _PhaseRecord:
    """单次阶段执行记录"""

    phase: str  # "PREDICT" | "VALIDATE" | "LEARN" | "EVOLVE"
    duration_s: float
    success: bool
    error_type: str | None
    timestamp: float  # time.time()


@dataclass(frozen=True, slots=True)
class _LLMRecord:
    """单次 LLM 调用记录"""

    phase: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost: float  # 估算成本（元）
    latency_s: float
    timestamp: float


@dataclass(frozen=True, slots=True)
class _TradeRecord:
    """单次交易决策记录"""

    decision: str  # "EXECUTED" | "SKIPPED" | "FAILED"
    reason: str
    timestamp: float


# ============================================================
# 聚合统计结构
# ============================================================


@dataclass(slots=True)
class _PhaseStats:
    """某阶段的聚合统计"""

    total: int = 0
    success: int = 0
    failure: int = 0
    total_duration_s: float = 0.0
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error_type: str | None = None

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def avg_duration_s(self) -> float:
        return self.total_duration_s / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "success": self.success,
            "failure": self.failure,
            "success_rate": round(self.success_rate, 4),
            "avg_duration_s": round(self.avg_duration_s, 3),
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_error_type": self.last_error_type,
        }


# ============================================================
# MetricsCollector
# ============================================================

_MAX_RECORDS = 10_000  # deque 容量上限


class MetricsCollector:
    """
    Agent 运行时指标收集器。

    线程安全性：设计为单进程 asyncio 使用，无需加锁。
    所有 record_* 方法为 O(1) 非阻塞。
    """

    def __init__(self, maxlen: int = _MAX_RECORDS) -> None:
        self._phase_records: deque[_PhaseRecord] = deque(maxlen=maxlen)
        self._llm_records: deque[_LLMRecord] = deque(maxlen=maxlen)
        self._trade_records: deque[_TradeRecord] = deque(maxlen=maxlen)
        self._queue_depths: deque[tuple[float, int]] = deque(
            maxlen=1000
        )  # (timestamp, depth)

        # 增量聚合（避免每次 get_snapshot 遍历全量 deque）
        self._phase_stats: dict[str, _PhaseStats] = {}

        # LLM 增量聚合
        self._llm_total_prompt_tokens: int = 0
        self._llm_total_completion_tokens: int = 0
        self._llm_total_cost: float = 0.0
        self._llm_call_count: int = 0

        # 交易增量聚合
        self._trade_executed: int = 0
        self._trade_skipped: int = 0
        self._trade_failed: int = 0

        # 阶段连续失败计数（供告警使用）
        self._consecutive_failures: dict[str, int] = {}

        self._started_at: float = time.time()

    # ==================================================================
    # 记录方法（O(1) 非阻塞）
    # ==================================================================

    def record_phase(
        self,
        phase: str,
        duration_s: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """记录一次阶段执行结果。"""
        record = _PhaseRecord(
            phase=phase,
            duration_s=duration_s,
            success=success,
            error_type=error_type,
            timestamp=time.time(),
        )
        self._phase_records.append(record)

        # 增量聚合
        stats = self._phase_stats.setdefault(phase, _PhaseStats())
        stats.total += 1
        stats.total_duration_s += duration_s
        if success:
            stats.success += 1
            stats.last_success_at = record.timestamp
            self._consecutive_failures[phase] = 0
        else:
            stats.failure += 1
            stats.last_failure_at = record.timestamp
            stats.last_error_type = error_type
            self._consecutive_failures[phase] = (
                self._consecutive_failures.get(phase, 0) + 1
            )

    def record_llm_call(
        self,
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost: float,
        latency_s: float = 0.0,
    ) -> None:
        """记录一次 LLM 调用的 token 用量与估算成本。"""
        record = _LLMRecord(
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost=estimated_cost,
            latency_s=latency_s,
            timestamp=time.time(),
        )
        self._llm_records.append(record)

        # 增量聚合
        self._llm_total_prompt_tokens += prompt_tokens
        self._llm_total_completion_tokens += completion_tokens
        self._llm_total_cost += estimated_cost
        self._llm_call_count += 1

    def record_trade(self, decision: str, reason: str = "") -> None:
        """记录一次交易决策。

        Args:
            decision: "EXECUTED" | "SKIPPED" | "FAILED"
            reason: 决策原因
        """
        record = _TradeRecord(
            decision=decision, reason=reason, timestamp=time.time()
        )
        self._trade_records.append(record)

        if decision == "EXECUTED":
            self._trade_executed += 1
        elif decision == "SKIPPED":
            self._trade_skipped += 1
        elif decision == "FAILED":
            self._trade_failed += 1

    def record_queue_depth(self, depth: int) -> None:
        """采样当前调度队列深度。"""
        self._queue_depths.append((time.time(), depth))

    # ==================================================================
    # 查询方法
    # ==================================================================

    def get_consecutive_failures(self, phase: str) -> int:
        """获取某阶段的当前连续失败次数。"""
        return self._consecutive_failures.get(phase, 0)

    def get_snapshot(self) -> dict:
        """
        返回所有指标的聚合快照。

        返回结构：
        {
            "uptime_seconds": float,
            "phases": { phase_name: { total, success, failure, success_rate, ... } },
            "llm": { call_count, total_prompt_tokens, total_completion_tokens, total_cost },
            "trades": { executed, skipped, failed },
            "queue": { latest_depth, max_depth, avg_depth },
        }
        """
        uptime = time.time() - self._started_at

        # 阶段统计
        phases = {
            name: stats.to_dict() for name, stats in self._phase_stats.items()
        }

        # LLM 统计
        llm = {
            "call_count": self._llm_call_count,
            "total_prompt_tokens": self._llm_total_prompt_tokens,
            "total_completion_tokens": self._llm_total_completion_tokens,
            "total_tokens": self._llm_total_prompt_tokens
            + self._llm_total_completion_tokens,
            "total_cost": round(self._llm_total_cost, 4),
        }

        # 交易统计
        trades = {
            "executed": self._trade_executed,
            "skipped": self._trade_skipped,
            "failed": self._trade_failed,
            "total": self._trade_executed + self._trade_skipped + self._trade_failed,
        }

        # 队列深度统计
        if self._queue_depths:
            depths = [d for _, d in self._queue_depths]
            queue = {
                "latest_depth": depths[-1],
                "max_depth": max(depths),
                "avg_depth": round(sum(depths) / len(depths), 1),
                "sample_count": len(depths),
            }
        else:
            queue = {
                "latest_depth": 0,
                "max_depth": 0,
                "avg_depth": 0.0,
                "sample_count": 0,
            }

        return {
            "uptime_seconds": round(uptime, 1),
            "phases": phases,
            "llm": llm,
            "trades": trades,
            "queue": queue,
        }


# ============================================================
# 全局单例（由 main.py lifespan 注入到各服务）
# ============================================================

metrics_collector = MetricsCollector()
