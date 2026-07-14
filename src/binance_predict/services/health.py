"""
Agent 运行健康监控服务 —— 融合「内存实时指标」与「数据库业务真值」

MetricsCollector 提供进程内易失的性能指标（耗时/成本/队列/连续失败），
但看不到业务真值（窗口是否漏跑、predict 匹配率、置信度是否准）。本模块做融合层：

- 从数据库聚合：窗口连续性、predict 匹配率与方向分布、置信度校准表
- 从传入的内存快照读取：调度器心跳（各阶段最近成功距今）、LLM 调用与错误率
- 派生：告警列表 alerts、总体状态 overall_status、自然语言诊断 summary

设计要点：
- 聚合/派生逻辑为顶层纯函数（无 I/O），便于单测（对齐 agent_logic.py 传统）
- HealthService.build_report 只做只读 DB 查询 + 组装，不写库
- 内存态数据（metrics_snapshot/consecutive_failures/queue_depth）为可选入参：
  Web 进程内可传入完整值；CLI --db-only 时为 None，报告仅含 DB 层指标
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config.settings import settings
from ..db.models import AgentPrediction, PatternMemory, SentimentWindow
from ..models.schemas import CalibrationBucket, HealthAlert, HealthReport

# 情绪窗口固定 5 分钟；用于连续性 gap 检测与 PREDICT 心跳停摆判定
_WINDOW_INTERVAL_MS = 300_000
_WINDOW_INTERVAL_S = 300.0

# 置信度分桶边界（左闭右开，最后一桶含 1.0）；低置信度归入 [0.00~0.50)
_CALIBRATION_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)

# 参与心跳/成功率评估的 LLM 阶段
_PHASES = ("PREDICT", "VALIDATE", "LEARN", "EVOLVE")


# ============================================================
# 纯函数：指标计算（无 I/O，可单测）
# ============================================================

def compute_window_continuity(
    start_times_ms: list[int], now_ms: int
) -> dict:
    """由最近若干窗口的 start_time（ms）计算连续性指标。

    Args:
        start_times_ms: 窗口开始时间戳列表（ms），顺序不限
        now_ms: 当前时间戳（ms）

    Returns:
        {last_window_age_s, gap_count, recent_count, expected_interval_s}
        无窗口时 last_window_age_s=None、gap_count=0。
    """
    if not start_times_ms:
        return {
            "last_window_age_s": None,
            "gap_count": 0,
            "recent_count": 0,
            "expected_interval_s": _WINDOW_INTERVAL_S,
        }

    ordered = sorted(start_times_ms)
    last = ordered[-1]
    age_s = max(0.0, (now_ms - last) / 1000.0)

    # gap 检测：相邻窗口间隔应约等于一个窗口周期，缺口按缺失窗口数累加
    gap_count = 0
    for prev, cur in zip(ordered, ordered[1:]):
        diff = cur - prev
        if diff > _WINDOW_INTERVAL_MS * 1.5:
            gap_count += round(diff / _WINDOW_INTERVAL_MS) - 1

    return {
        "last_window_age_s": round(age_s, 1),
        "gap_count": gap_count,
        "recent_count": len(ordered),
        "expected_interval_s": _WINDOW_INTERVAL_S,
    }


def compute_predict_stats(
    directions: list[str], matched_flags: list[bool], active_pattern_count: int
) -> dict:
    """计算 predict 匹配率与方向分布。

    Args:
        directions: 最近若干条预测的方向（UP/DOWN/NO_TRADE）
        matched_flags: 与 directions 等长，元素为「是否匹配到模式」
        active_pattern_count: 当前 ACTIVE 模式数（用于判断是否已脱离冷启动）
    """
    total = len(directions)
    matched = sum(1 for m in matched_flags if m)
    dist = {"UP": 0, "DOWN": 0, "NO_TRADE": 0}
    for d in directions:
        if d in dist:
            dist[d] += 1
    return {
        "total": total,
        "matched": matched,
        "match_rate": round(matched / total, 4) if total > 0 else 0.0,
        "direction_distribution": dist,
        "active_pattern_count": active_pattern_count,
    }


def compute_calibration(pairs: list[tuple[float, bool]]) -> list[CalibrationBucket]:
    """按置信度分桶，计算每桶平均置信度与实际命中率。

    Args:
        pairs: (confidence, is_correct) 列表，仅含已验证样本

    Returns:
        非空分桶的 CalibrationBucket 列表（按区间升序）。
    """
    n_buckets = len(_CALIBRATION_EDGES) - 1
    counts = [0] * n_buckets
    conf_sums = [0.0] * n_buckets
    correct = [0] * n_buckets

    for conf, is_correct in pairs:
        for i in range(n_buckets):
            lo, hi = _CALIBRATION_EDGES[i], _CALIBRATION_EDGES[i + 1]
            if lo <= conf < hi:
                counts[i] += 1
                conf_sums[i] += conf
                if is_correct:
                    correct[i] += 1
                break

    buckets: list[CalibrationBucket] = []
    for i in range(n_buckets):
        if counts[i] == 0:
            continue
        lo = _CALIBRATION_EDGES[i]
        hi = min(_CALIBRATION_EDGES[i + 1], 1.0)
        avg_conf = conf_sums[i] / counts[i]
        hit_rate = correct[i] / counts[i]
        buckets.append(
            CalibrationBucket(
                range=f"{lo:.2f}~{hi:.2f}",
                count=counts[i],
                avg_confidence=round(avg_conf, 4),
                hit_rate=round(hit_rate, 4),
                gap=round(avg_conf - hit_rate, 4),
            )
        )
    return buckets


def compute_phase_ages(metrics_snapshot: dict | None, now_epoch: float) -> dict:
    """由内存快照计算各阶段「最近成功距今」秒数。

    Returns:
        {phase: age_s | None}；快照缺失或阶段从未成功时值为 None。
    """
    ages: dict[str, float | None] = {}
    phases = (metrics_snapshot or {}).get("phases", {})
    for phase in _PHASES:
        last = phases.get(phase, {}).get("last_success_at")
        ages[phase] = round(now_epoch - last, 1) if last else None
    return ages


def derive_alerts(
    *,
    window_continuity: dict,
    predict_stats: dict,
    phase_ages: dict,
    queue_depth: int | None,
    llm: dict | None,
    consecutive_failures: dict | None,
    has_memory: bool,
) -> list[HealthAlert]:
    """依据各指标派生告警列表。

    has_memory=False（CLI --db-only）时跳过所有依赖内存态的告警。
    """
    alerts: list[HealthAlert] = []

    # 1. 窗口停摆（DB 层，始终评估）
    age = window_continuity.get("last_window_age_s")
    if age is not None and age > settings.agent_health_window_stale_seconds:
        alerts.append(HealthAlert(
            level="CRITICAL", code="WINDOW_STALE",
            message=f"最近情绪窗口距今 {age:.0f}s，超过停摆阈值 "
                    f"{settings.agent_health_window_stale_seconds:.0f}s，归档可能已停止",
        ))
    # 2. 窗口缺口（DB 层）
    gap = window_continuity.get("gap_count", 0)
    if gap > 0:
        alerts.append(HealthAlert(
            level="WARN", code="WINDOW_GAP",
            message=f"最近窗口序列存在 {gap} 个缺口，采样/归档可能间歇性中断",
        ))
    # 3. 脱离冷启动后匹配率为 0（DB 层）
    if (
        predict_stats.get("active_pattern_count", 0) > 0
        and predict_stats.get("total", 0) > 0
        and predict_stats.get("matched", 0) == 0
    ):
        alerts.append(HealthAlert(
            level="WARN", code="NO_MATCH",
            message=f"已有 {predict_stats['active_pattern_count']} 个 ACTIVE 模式，但最近 "
                    f"{predict_stats['total']} 条预测匹配率为 0，模式形态可能与实时曲线口径不符",
        ))

    if not has_memory:
        return alerts

    # 4. PREDICT 心跳停摆（内存态）
    predict_age = phase_ages.get("PREDICT")
    stale_limit = settings.agent_health_predict_stale_multiplier * _WINDOW_INTERVAL_S
    if predict_age is not None and predict_age > stale_limit:
        alerts.append(HealthAlert(
            level="CRITICAL", code="PREDICT_STALE",
            message=f"PREDICT 最近成功距今 {predict_age:.0f}s，超过 {stale_limit:.0f}s，"
                    f"调度可能未触发预测",
        ))
    # 5. 队列积压（内存态）
    if queue_depth is not None and queue_depth >= settings.agent_alert_queue_depth_threshold:
        alerts.append(HealthAlert(
            level="WARN", code="QUEUE_DEEP",
            message=f"调度队列深度 {queue_depth} 达阈值 "
                    f"{settings.agent_alert_queue_depth_threshold}，worker 可能消费不及",
        ))
    # 6. LLM 连续失败（内存态）
    cf = consecutive_failures or {}
    for phase, cnt in cf.items():
        if cnt >= settings.agent_alert_consecutive_failures:
            alerts.append(HealthAlert(
                level="CRITICAL", code="LLM_FAILURES",
                message=f"阶段 {phase} 连续失败 {cnt} 次（阈值 "
                        f"{settings.agent_alert_consecutive_failures}），请检查 LLM/DB/配置",
            ))
    # 7. LLM 成功率偏低（内存态）
    rates = (llm or {}).get("phase_success_rates", {})
    for phase, info in rates.items():
        rate = info.get("success_rate")
        total = info.get("total", 0)
        if total > 0 and rate is not None and rate < settings.agent_health_llm_success_rate_floor:
            alerts.append(HealthAlert(
                level="WARN", code="LLM_ERROR_RATE",
                message=f"阶段 {phase} 成功率 {rate:.0%} 低于下限 "
                        f"{settings.agent_health_llm_success_rate_floor:.0%}（{total} 次调用）",
            ))
    return alerts


def derive_overall_status(alerts: list[HealthAlert]) -> str:
    """由告警列表派生总体状态：任一 CRITICAL→CRITICAL，任一 WARN→WARN，否则 OK。"""
    if any(a.level == "CRITICAL" for a in alerts):
        return "CRITICAL"
    if any(a.level == "WARN" for a in alerts):
        return "WARN"
    return "OK"


def build_summary(
    *,
    window_continuity: dict,
    predict_stats: dict,
    calibration: list[CalibrationBucket],
    alerts: list[HealthAlert],
    has_memory: bool,
) -> str:
    """生成一段自然语言诊断，供人或 LLM 一眼读懂当前健康态。"""
    parts: list[str] = []

    # 归档
    age = window_continuity.get("last_window_age_s")
    gap = window_continuity.get("gap_count", 0)
    if age is None:
        parts.append("尚无情绪窗口记录")
    elif age > settings.agent_health_window_stale_seconds:
        parts.append(f"⚠️ 归档疑似停摆（最近窗口距今 {age:.0f}s）")
    else:
        parts.append(f"归档正常（最近窗口距今 {age:.0f}s，缺口 {gap}）")

    # 匹配
    total = predict_stats.get("total", 0)
    active = predict_stats.get("active_pattern_count", 0)
    if active == 0:
        parts.append("仍处冷启动（无 ACTIVE 模式，predict 恒为 NO_TRADE）")
    elif total == 0:
        parts.append(f"已有 {active} 个模式但暂无预测记录")
    else:
        mr = predict_stats.get("match_rate", 0.0)
        parts.append(f"已脱离冷启动，最近 {total} 条预测匹配率 {mr:.0%}")

    # 校准
    n_samples = sum(b.count for b in calibration)
    min_s = settings.agent_health_min_calibration_samples
    if n_samples < min_s:
        parts.append(f"校准样本不足（{n_samples}<{min_s}），暂不可评置信度准度")
    else:
        # 找最大 |gap| 的桶点出偏差方向
        worst = max(
            (b for b in calibration if b.gap is not None),
            key=lambda b: abs(b.gap),
            default=None,
        )
        if worst is not None and abs(worst.gap) >= 0.15:
            tone = "过度自信" if worst.gap > 0 else "过度保守"
            parts.append(f"校准偏差明显：{worst.range} 区间{tone}（gap={worst.gap:+.2f}）")
        else:
            parts.append(f"置信度校准尚可（{n_samples} 样本）")

    if not has_memory:
        parts.append("（仅 DB 层指标；调度器/LLM 内存态未采集）")

    # 告警摘要
    if alerts:
        crit = [a.code for a in alerts if a.level == "CRITICAL"]
        warn = [a.code for a in alerts if a.level == "WARN"]
        seg = []
        if crit:
            seg.append("CRITICAL: " + ", ".join(crit))
        if warn:
            seg.append("WARN: " + ", ".join(warn))
        parts.append("；".join(seg))
    else:
        parts.append("无告警")

    return "；".join(parts) + "。"


# ============================================================
# HealthService：只读 DB 聚合 + 组装报告
# ============================================================

class HealthService:
    """构建 Agent 运行健康报告。只读，不写库。"""

    async def build_report(
        self,
        db: AsyncSession,
        *,
        metrics_snapshot: dict | None = None,
        consecutive_failures: dict | None = None,
        queue_depth: int | None = None,
    ) -> HealthReport:
        """聚合 DB 真值与内存快照，产出 HealthReport。

        Args:
            db: 只读异步会话
            metrics_snapshot: metrics_collector.get_snapshot()；CLI --db-only 时为 None
            consecutive_failures: {phase: 连续失败数}；同上
            queue_depth: 调度器当前队列深度；同上
        """
        now = datetime.now(timezone.utc)
        now_ms = int(time.time() * 1000)
        now_epoch = time.time()
        has_memory = metrics_snapshot is not None

        # --- DB 聚合 ---
        win_res = await db.execute(
            select(SentimentWindow.start_time)
            .order_by(SentimentWindow.start_time.desc())
            .limit(settings.agent_health_recent_windows)
        )
        start_times = [row[0] for row in win_res.all()]

        pred_res = await db.execute(
            select(AgentPrediction.predicted_direction, AgentPrediction.matched_pattern_id)
            .order_by(AgentPrediction.prediction_time.desc())
            .limit(settings.agent_health_recent_predictions)
        )
        pred_rows = pred_res.all()
        directions = [r[0] for r in pred_rows]
        matched_flags = [r[1] is not None for r in pred_rows]

        calib_res = await db.execute(
            select(AgentPrediction.confidence, AgentPrediction.is_correct)
            .where(AgentPrediction.is_correct.is_not(None))
            .order_by(AgentPrediction.prediction_time.desc())
            .limit(settings.agent_health_calibration_sample_limit)
        )
        pairs = [(float(r[0]), bool(r[1])) for r in calib_res.all()]

        active_res = await db.execute(
            select(func.count(PatternMemory.id)).where(PatternMemory.status == "ACTIVE")
        )
        active_count = active_res.scalar_one() or 0

        # --- 指标计算 ---
        window_continuity = compute_window_continuity(start_times, now_ms)
        predict_stats = compute_predict_stats(directions, matched_flags, active_count)
        calibration = compute_calibration(pairs)
        phase_ages = compute_phase_ages(metrics_snapshot, now_epoch)

        # --- 组装内存态 dict ---
        scheduler = {
            "queue_depth": queue_depth,
            "phase_ages_s": phase_ages,
            "uptime_seconds": (metrics_snapshot or {}).get("uptime_seconds"),
        }
        phase_success_rates = {
            phase: {
                "success_rate": info.get("success_rate"),
                "total": info.get("total", 0),
            }
            for phase, info in (metrics_snapshot or {}).get("phases", {}).items()
        }
        llm = {
            **(metrics_snapshot or {}).get("llm", {}),
            "phase_success_rates": phase_success_rates,
            "consecutive_failures": consecutive_failures or {},
        }

        # --- 派生告警 / 状态 / 诊断 ---
        alerts = derive_alerts(
            window_continuity=window_continuity,
            predict_stats=predict_stats,
            phase_ages=phase_ages,
            queue_depth=queue_depth,
            llm=llm,
            consecutive_failures=consecutive_failures,
            has_memory=has_memory,
        )
        overall_status = derive_overall_status(alerts)
        summary = build_summary(
            window_continuity=window_continuity,
            predict_stats=predict_stats,
            calibration=calibration,
            alerts=alerts,
            has_memory=has_memory,
        )

        return HealthReport(
            generated_at=now,
            overall_status=overall_status,
            alerts=alerts,
            window_continuity=window_continuity,
            predict_stats=predict_stats,
            calibration=calibration,
            scheduler=scheduler,
            llm=llm,
            summary=summary,
        )


# 全局单例（无状态，可安全复用）
health_service = HealthService()
