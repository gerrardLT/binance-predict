"""
Agent 阈值告警服务

基于 MetricsCollector 的聚合数据执行阈值检查，触发条件时通过 loguru 记录告警日志。
设计为轻量级、无外部依赖，后续可扩展 webhook/邮件等通知渠道。

告警条件：
- 阶段连续失败 >= N 次
- LLM 累计成本超过日限额
- 调度队列积压超阈值
"""

from __future__ import annotations

from loguru import logger

from ..config.settings import settings
from .metrics import MetricsCollector


class AlertService:
    """
    阈值告警服务。

    调用 check_and_alert() 检查各项指标是否超过预设阈值，
    超阈值时通过 loguru.warning 输出 [ALERT] 前缀的告警日志。
    """

    def __init__(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics
        # 避免重复告警：记录已触发告警的阶段/类型
        self._alerted_phase_failures: dict[str, int] = {}
        # Fix #19: 熟断器阻断标志。任一严重告警（成本超限/阶段连续失败）
        # 触发时置 True，供 evaluate_trade_gate 查询以拒绝新交易。
        self._trading_blocked: bool = False

    @property
    def trading_blocked(self) -> bool:
        """当前是否因告警而需阻断交易（Fix #19）。

        仅在 settings.agent_alert_block_trades 开启时生效。
        """
        return settings.agent_alert_block_trades and self._trading_blocked

    def check_and_alert(self) -> None:
        """执行全部告警检查。"""
        if not settings.agent_alert_enabled:
            return

        # 每轮重新评估阻断状态：先复位，再由各 _check_* 根据当前指标置位，
        # 使指标恢复正常后能自动解除阻断（Fix #19）。
        self._trading_blocked = False
        self._check_phase_failures()
        self._check_cost_limit()
        self._check_queue_depth()

    def _check_phase_failures(self) -> None:
        """检查各阶段连续失败次数是否超过阈值。"""
        threshold = settings.agent_alert_consecutive_failures
        for phase in ("PREDICT", "VALIDATE", "LEARN", "EVOLVE"):
            count = self._metrics.get_consecutive_failures(phase)
            prev_alerted = self._alerted_phase_failures.get(phase, 0)
            if count >= threshold and count > prev_alerted:
                logger.warning(
                    "[ALERT] 阶段 {} 连续失败 {} 次（阈值 {}）| "
                    "请检查 LLM 服务/数据库连接/配置参数",
                    phase,
                    count,
                    threshold,
                )
                self._alerted_phase_failures[phase] = count
                # Fix #19: 阶段连续失败超阈，置位阻断标志
                self._trading_blocked = True
            elif count == 0 and prev_alerted > 0:
                # 恢复正常后重置告警状态
                self._alerted_phase_failures[phase] = 0

    def _check_cost_limit(self) -> None:
        """检查 LLM 累计成本是否超过日限额。"""
        snapshot = self._metrics.get_snapshot()
        total_cost = snapshot.get("llm", {}).get("total_cost", 0.0)
        limit = settings.agent_alert_daily_cost_limit_usd
        if total_cost >= limit:
            logger.warning(
                "[ALERT] LLM 累计成本 {:.4f} 元已达日限额 {:.4f} 元 | "
                "建议检查调用频率或调整限额配置",
                total_cost,
                limit,
            )
            # Fix #19: 成本超限，置位阻断标志（避免失控持续花费/下单）
            self._trading_blocked = True

    def _check_queue_depth(self) -> None:
        """检查调度队列积压是否超过阈值。"""
        snapshot = self._metrics.get_snapshot()
        latest_depth = snapshot.get("queue", {}).get("latest_depth", 0)
        threshold = settings.agent_alert_queue_depth_threshold
        if latest_depth >= threshold:
            logger.warning(
                "[ALERT] 调度队列深度 {} 超过阈值 {} | "
                "可能存在阶段执行阻塞，请检查 worker 消费速度",
                latest_depth,
                threshold,
            )
