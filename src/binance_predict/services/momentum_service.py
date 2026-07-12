"""
概率动量策略服务（方案 C）

基于预测市场 UP/DOWN 概率时序数据，计算多维度动量信号并产出独立预测。
不依赖 K 线数据，与 K 线+LLM 决策方法天然互补。

核心信号：
1. 概率动量 (momentum) — UP% 在 15s/30s/1min 内的变化速度
2. 概率波动率 (volatility) — 窗口内 UP% 的标准差
3. 参与者增长 (participant_growth) — 窗口内参与人数变化率
4. 交易量加速度 (volume_acceleration) — 交易量的变化速率
5. 概率趋势一致性 (trend_consistency) — 多个时间尺度的方向一致性
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class MomentumSignal:
    """单个维度的信号"""
    name: str
    value: float          # 原始值
    score: float          # 归一化到 [-1, +1]
    description: str = ""


@dataclass
class MomentumResult:
    """概率动量预测结果"""
    direction: str = "NO_TRADE"       # UP / DOWN / NO_TRADE
    confidence: float = 0.0           # 0~1
    composite_score: float = 0.0      # 综合评分 [-1, +1]
    signals: list[MomentumSignal] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    sample_count: int = 0             # 使用的采样点数
    elapsed_seconds: int = 0          # 窗口已过秒数
    remaining_seconds: int = 0        # 窗口剩余秒数


class MomentumService:
    """
    概率动量分析引擎

    输入：_pm_history 中的时序数据点列表
    输出：MomentumResult（方向 + 置信度 + 各信号明细）
    """

    # ── 评分阈值 ──────────────────────────────────────
    ENTRY_THRESHOLD = 0.45        # 综合评分超过此值才给方向
    HIGH_CONFIDENCE = 0.70        # 高置信度阈值
    MIN_SAMPLES = 4               # 最少需要 4 个采样点（1 分钟数据）
    MIN_ELAPSED_SEC = 60          # 至少经过 60 秒才做预测（前 1 分钟观察期）

    def analyze(self, points: list[dict], window_duration_sec: int = 300) -> MomentumResult:
        """
        对当前窗口的时序数据做概率动量分析

        Parameters
        ----------
        points : list[dict]
            _pm_history 中的数据点，每个包含：
            timestamp, up_pct, down_pct, participants, trade_volume
        window_duration_sec : int
            窗口总时长（默认 5 分钟 = 300 秒）

        Returns
        -------
        MomentumResult
        """
        result = MomentumResult()

        # ── 基本校验 ──────────────────────────────────
        if not points or len(points) < self.MIN_SAMPLES:
            result.reasoning.append(f"采样点不足（{len(points)} 个，需要 ≥{self.MIN_SAMPLES}）")
            return result

        # 过滤有效 UP% 数据
        up_series = [
            (p["timestamp"], p["up_pct"])
            for p in points
            if p.get("up_pct") is not None
        ]
        if len(up_series) < self.MIN_SAMPLES:
            result.reasoning.append(f"有效 UP% 数据不足（{len(up_series)} 个）")
            return result

        result.sample_count = len(up_series)

        # 计算窗口经过时间
        first_ts = up_series[0][0]
        last_ts = up_series[-1][0]
        elapsed_ms = last_ts - first_ts
        result.elapsed_seconds = max(int(elapsed_ms / 1000), 1)
        result.remaining_seconds = max(window_duration_sec - result.elapsed_seconds, 0)

        if result.elapsed_seconds < self.MIN_ELAPSED_SEC:
            result.reasoning.append(f"窗口刚开始（已过 {result.elapsed_seconds}s），建议先观察")
            return result

        # ── 1. 概率动量 (Probability Momentum) ───────
        momentum_signals = self._compute_momentum(up_series)
        result.signals.extend(momentum_signals)

        # ── 2. 概率波动率 (Probability Volatility) ──
        vol_signal = self._compute_volatility(up_series)
        result.signals.append(vol_signal)

        # ── 3. 参与者增长 (Participant Growth) ──────
        participant_signal = self._compute_participant_growth(points)
        if participant_signal:
            result.signals.append(participant_signal)

        # ── 4. 交易量加速度 (Volume Acceleration) ───
        volume_signal = self._compute_volume_acceleration(points)
        if volume_signal:
            result.signals.append(volume_signal)

        # ── 5. 趋势一致性 (Trend Consistency) ───────
        trend_signal = self._compute_trend_consistency(up_series)
        result.signals.append(trend_signal)

        # ── 综合评分 ────────────────────────────────
        return self._synthesize(result)

    # ================================================================
    # 信号计算
    # ================================================================

    def _compute_momentum(self, up_series: list[tuple[int, float]]) -> list[MomentumSignal]:
        """计算多时间尺度的概率动量"""
        signals = []
        last_ts, last_val = up_series[-1]

        # 在 up_series 中查找距今约 15s/30s/60s 的数据点
        for label, target_delta_ms in [("15s", 15_000), ("30s", 30_000), ("60s", 60_000)]:
            target_ts = last_ts - target_delta_ms
            # 找最接近目标时间的点
            best = None
            best_diff = float("inf")
            for ts, val in up_series:
                diff = abs(ts - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best = (ts, val)

            if best and best_diff < target_delta_ms * 0.8:  # 容差 80%
                delta_pct = last_val - best[1]
                actual_sec = max((last_ts - best[0]) / 1000, 1)
                rate = delta_pct / actual_sec  # %点/秒

                # 归一化：0.5 %/s 映射到 ±1
                score = max(min(rate / 0.5, 1.0), -1.0)
                signals.append(MomentumSignal(
                    name=f"动量({label})",
                    value=round(rate, 4),
                    score=round(score, 3),
                    description=f"UP% 变化 {delta_pct:+.1f}pp / {actual_sec:.0f}s = {rate:+.4f} pp/s",
                ))

        return signals

    def _compute_volatility(self, up_series: list[tuple[int, float]]) -> MomentumSignal:
        """计算概率波动率（标准差）"""
        values = [v for _, v in up_series]
        std = statistics.stdev(values) if len(values) >= 2 else 0

        # 波动率含义：
        # 高波动 (>15) = 市场分歧大，信号不可靠 → score 趋近 0
        # 低波动 (<5) = 市场共识强 → 保留动量方向（不额外调整）
        if std > 15:
            score = 0.0  # 高波动 → 中性
            desc = f"σ={std:.1f}（高分歧，信号不可靠）"
        elif std > 8:
            score = 0.0
            desc = f"σ={std:.1f}（中等分歧）"
        else:
            score = 0.0  # 波动率本身不提供方向，只是信心调节器
            desc = f"σ={std:.1f}（共识较强）"

        return MomentumSignal(
            name="概率波动率",
            value=round(std, 2),
            score=score,
            description=desc,
        )

    def _compute_participant_growth(self, points: list[dict]) -> MomentumSignal | None:
        """计算参与者增长率"""
        part_series = []
        for p in points:
            val = p.get("participants")
            if val is not None:
                try:
                    part_series.append((p["timestamp"], float(val)))
                except (ValueError, TypeError):
                    pass
        if len(part_series) < 2:
            return None

        first_p = part_series[0][1]
        last_p = part_series[-1][1]
        elapsed_sec = max((part_series[-1][0] - part_series[0][0]) / 1000, 1)

        growth_rate = (last_p - first_p) / elapsed_sec  # 人/秒

        # 参与者增长 → 新信息驱动，信号更可靠
        # 归一化：2 人/秒 映射到 ±1
        score = max(min(growth_rate / 2.0, 1.0), -1.0)

        return MomentumSignal(
            name="参与者增长",
            value=round(growth_rate, 3),
            score=round(score, 3),
            description=f"{first_p}→{last_p}人，增长 {growth_rate:+.2f} 人/秒",
        )

    def _compute_volume_acceleration(self, points: list[dict]) -> MomentumSignal | None:
        """计算交易量加速度"""
        vol_series = []
        for p in points:
            val = p.get("trade_volume")
            if val is not None:
                try:
                    vol_series.append((p["timestamp"], float(val)))
                except (ValueError, TypeError):
                    pass
        if len(vol_series) < 3:
            return None

        # 将序列分为前半和后半，比较增量变化
        mid = len(vol_series) // 2
        first_half = vol_series[:mid]
        second_half = vol_series[mid:]

        # 前半段交易量增量
        delta_first = first_half[-1][1] - first_half[0][1]
        span_first = max((first_half[-1][0] - first_half[0][0]) / 1000, 1)
        rate_first = delta_first / span_first

        # 后半段交易量增量
        delta_second = second_half[-1][1] - second_half[0][1]
        span_second = max((second_half[-1][0] - second_half[0][0]) / 1000, 1)
        rate_second = delta_second / span_second

        # 加速度 = 后半速率 - 前半速率
        acceleration = rate_second - rate_first

        # 交易量加速 → 市场活跃度上升，信号增强
        # 归一化：50 $/s² 映射到 ±1
        score = max(min(acceleration / 50.0, 1.0), -1.0)

        return MomentumSignal(
            name="交易量加速度",
            value=round(acceleration, 2),
            score=round(score, 3),
            description=f"前段 {rate_first:.1f} $/s → 后段 {rate_second:.1f} $/s，加速度 {acceleration:+.1f}",
        )

    def _compute_trend_consistency(self, up_series: list[tuple[int, float]]) -> MomentumSignal:
        """计算趋势一致性：多个时间尺度方向是否一致"""
        if len(up_series) < 4:
            return MomentumSignal(
                name="趋势一致性", value=0.0, score=0.0,
                description="数据不足，无法评估",
            )

        values = [v for _, v in up_series]
        first_val = values[0]
        last_val = values[-1]
        mid_val = values[len(values) // 2]

        # 短期方向：后半段
        short_dir = 1 if last_val > mid_val else (-1 if last_val < mid_val else 0)
        # 中期方向：全程
        long_dir = 1 if last_val > first_val else (-1 if last_val < first_val else 0)
        # 中间点方向
        mid_dir = 1 if mid_val > first_val else (-1 if mid_val < first_val else 0)

        # 一致性评分：3 个方向一致 = 1.0，2 个一致 = 0.33，全不同 = -0.33
        dirs = [short_dir, mid_dir, long_dir]
        agreement = sum(dirs)
        if abs(agreement) == 3:
            consistency = 1.0 if agreement > 0 else -1.0
        elif abs(agreement) >= 1:
            consistency = 0.33 if agreement > 0 else -0.33
        else:
            consistency = 0.0

        return MomentumSignal(
            name="趋势一致性",
            value=round(consistency, 2),
            score=round(consistency, 3),
            description=f"短/中/长期方向: {short_dir:+d}/{mid_dir:+d}/{long_dir:+d}（一致度 {consistency:+.2f}）",
        )

    # ================================================================
    # 综合评分
    # ================================================================

    def _synthesize(self, result: MomentumResult) -> MomentumResult:
        """将各信号综合为最终预测"""
        # 分离方向信号和调节信号
        direction_signals = [s for s in result.signals if s.name.startswith("动量")]
        modifier_signals = [s for s in result.signals if not s.name.startswith("动量") and s.name != "概率波动率"]

        # 动量信号加权平均（15s/30s/60s 各有权重：近期权重更高）
        weights = {"15s": 0.25, "30s": 0.35, "60s": 0.40}
        if direction_signals:
            weighted_sum = 0
            total_weight = 0
            for s in direction_signals:
                w = 1.0
                for key, val in weights.items():
                    if key in s.name:
                        w = val
                        break
                weighted_sum += s.score * w
                total_weight += w
            momentum_avg = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            momentum_avg = 0

        # 调节因子：参与者增长和交易量加速度作为确认信号
        modifier_avg = 0
        if modifier_signals:
            modifier_avg = sum(s.score for s in modifier_signals) / len(modifier_signals)

        # 趋势一致性加成
        trend = next((s for s in result.signals if s.name == "趋势一致性"), None)
        trend_bonus = trend.score * 0.15 if trend else 0

        # 波动率惩罚
        vol = next((s for s in result.signals if s.name == "概率波动率"), None)
        vol_penalty = 0.5 if vol and vol.value > 15 else (0.8 if vol and vol.value > 8 else 1.0)

        # 综合评分 = 动量均值 × (1 + 调节因子 × 0.2 + 趋势加成) × 波动率惩罚
        composite = momentum_avg * (1 + modifier_avg * 0.2 + trend_bonus) * vol_penalty
        composite = max(min(composite, 1.0), -1.0)
        result.composite_score = round(composite, 3)

        # 决策
        if abs(composite) < self.ENTRY_THRESHOLD:
            result.direction = "NO_TRADE"
            result.confidence = round(1 - abs(composite) / self.ENTRY_THRESHOLD, 2)
            result.reasoning.append(
                f"综合评分 {composite:+.3f} 未达入场阈值 ±{self.ENTRY_THRESHOLD}"
            )
        else:
            result.direction = "UP" if composite > 0 else "DOWN"
            # 置信度 = 评分超出阈值的程度，映射到 [0.5, 1.0]
            excess = (abs(composite) - self.ENTRY_THRESHOLD) / (1.0 - self.ENTRY_THRESHOLD)
            result.confidence = round(0.5 + excess * 0.5, 2)
            result.reasoning.append(
                f"综合评分 {composite:+.3f} → {'看涨' if composite > 0 else '看跌'}，"
                f"置信度 {result.confidence:.0%}"
            )

        # 组装推理明细
        for s in result.signals:
            if s.name != "概率波动率":
                result.reasoning.append(f"  {s.name}: {s.description}")

        return result
