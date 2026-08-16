"""研究调度器（M2）：场景评估的触发与编排。

触发器（满足其一，且距上次评估 ≥ 冷却时长）：
  T1 定期：距上次评估 ≥ scene_research_interval_days（默认 7 天）
  T2 累积：已结算信号自上次评估新增 ≥ scene_research_new_signals（默认 30）
  T3 异常：实盘某场景胜率跌破样本外基准 CI 下界（样本 ≥ min_live_sample）

上次评估时刻持久化在 llm_traces(phase=SCENE_RESEARCH) 最新一条（零新表）。
触发后：构建评估上下文（DB 实况 + 固定基线数字）→ SceneResearcher.evaluate →
假设逐条写 scene_param_versions(PENDING_REVIEW)，等 M3 裁决。
回测基线数字不在此处现算（评估上下文用 M1 引擎固化的基准快照），
实况统计查 fake_breakout_signals。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy import select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import FakeBreakoutSignal, LLMTrace, SceneParamVersion
from .scene_researcher import SceneResearcher

# 真实样本外基准（M1 引擎 2026-08-16 窗口实测，后 60 天盲验口径——
# 注意：原「验证集 63.6%/57.8%」标签实为 180 天全样本，此处用真正的样本外数字）
SCENE_BASELINES = {
    "bull_exhaust": {"p": 0.620, "ci_lower": 0.540, "n": 150},
    "bear_exhaust": {"p": 0.564, "ci_lower": 0.484, "n": 149},
}
# 全样本基准（供上下文展示；口径澄清见 scripts/backtest_engine.py 输出）
BASELINE_FULL = {
    "bull_exhaust": {"p": 0.637, "n": 457},
    "bear_exhaust": {"p": 0.579, "n": 508},
}
# 验证集可检测下限（n≈150 → 5.8pp@80% 功效量级；M3 功效预检动态计算，此处供 LLM 参考）
MIN_DETECTABLE_PP = 10.0

_CHECK_INTERVAL_S = 60.0


class ResearchScheduler:
    """场景评估后台调度（lifespan 挂载，模式照抄 FakeBreakoutDetector start/stop）。"""

    def __init__(self, researcher: SceneResearcher) -> None:
        self._researcher = researcher
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_run: float | None = None  # 内存态；冷启动从 llm_traces 回读
        self._last_settled_count: int | None = None  # T2 增量基线（评估后刷新）

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._last_run = await self._load_last_run_from_traces()
        self._task = asyncio.create_task(self._loop(), name="scene_research_scheduler")
        logger.info(
            "场景研究调度器启动 | T1={}天 T2=+{}条 T3=CI下界 冷却={}h | 上次评估: {}",
            settings.scene_research_interval_days,
            settings.scene_research_new_signals,
            settings.scene_research_cooldown_hours,
            datetime.fromtimestamp(self._last_run, tz=timezone.utc).strftime("%m-%d %H:%M")
            if self._last_run else "从未",
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("场景研究调度器已停止")

    async def _loop(self) -> None:
        while self._running:
            try:
                trigger = await self._check_triggers()
                if trigger:
                    await self._run_evaluation(trigger)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("场景研究调度循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(_CHECK_INTERVAL_S)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # 触发判定
    # ------------------------------------------------------------------

    async def _load_last_run_from_traces(self) -> float | None:
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(sa_func.max(LLMTrace.created_at))
                    .where(LLMTrace.phase == "SCENE_RESEARCH")
                )
                latest = (await session.execute(stmt)).scalar()
                if latest is None:
                    return None
                return latest.timestamp()
        except Exception as exc:
            logger.warning("回读上次评估时刻失败（按从未评估处理）| {}", exc)
            return None

    async def _check_triggers(self) -> str | None:
        """Returns: 'T1' | 'T2' | 'T3' | None。"""
        if self._last_run is not None:
            if time.time() - self._last_run < settings.scene_research_cooldown_hours * 3600:
                return None  # 冷却期内
            if time.time() - self._last_run >= settings.scene_research_interval_days * 86400:
                return "T1"
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(sa_func.count(FakeBreakoutSignal.id))
                    .where(FakeBreakoutSignal.status == "SETTLED")
                    .where(FakeBreakoutSignal.pattern.isnot(None))
                )
                settled_total = (await session.execute(stmt)).scalar() or 0
            # T2：自上次评估新增 ≥ 阈值（冷启动无基线时用总量首次触发）
            baseline_count = getattr(self, "_last_settled_count", None)
            if baseline_count is None:
                if settled_total >= settings.scene_research_new_signals:
                    return "T2"
            elif settled_total - baseline_count >= settings.scene_research_new_signals:
                return "T2"
        except Exception as exc:
            logger.debug("T2 检查失败（跳过）| {}", exc)

        # T3：实盘胜率跌破样本外基准 CI 下界
        try:
            live = await self._live_stats()
            for pattern, base in SCENE_BASELINES.items():
                s = live.get(pattern)
                if s and s["n"] >= settings.scene_research_min_live_sample:
                    if s["win_rate"] < base["ci_lower"]:
                        return "T3"
        except Exception as exc:
            logger.debug("T3 检查失败（跳过）| {}", exc)
        return None

    # ------------------------------------------------------------------
    # 评估执行
    # ------------------------------------------------------------------

    async def _live_stats(self) -> dict:
        """线上已结算场景信号统计（按 pattern 分组；胜负：high买DOWN赢⟺DOWN / low买UP赢⟺UP）。"""
        async with async_session_factory() as session:
            rows = (
                select(FakeBreakoutSignal.pattern, FakeBreakoutSignal.side, FakeBreakoutSignal.settle_outcome)
                .where(FakeBreakoutSignal.status == "SETTLED")
                .where(FakeBreakoutSignal.pattern.isnot(None))
            )
            detail = (await session.execute(rows)).all()
        stats: dict[str, dict] = {}
        for pattern, side, outcome in detail:
            s = stats.setdefault(pattern, {"n": 0, "wins": 0})
            win = (side == "high" and outcome == "DOWN") or (side == "low" and outcome == "UP")
            s["n"] += 1
            s["wins"] += int(bool(win))
        for pattern, s in stats.items():
            s["win_rate"] = s["wins"] / s["n"] if s["n"] else None
        return stats

    async def _failure_profile(self) -> list[str]:
        """输的信号特征摘要（脱敏聚合，≤20 行）。"""
        async with async_session_factory() as session:
            stmt = (
                select(FakeBreakoutSignal)
                .where(FakeBreakoutSignal.status == "SETTLED")
                .where(FakeBreakoutSignal.pattern.isnot(None))
                .order_by(FakeBreakoutSignal.signal_time.desc())
                .limit(60)
            )
            rows = (await session.execute(stmt)).scalars().all()
        lines = []
        for s in rows:
            won = (s.side == "high" and s.settle_outcome == "DOWN") or (
                s.side == "low" and s.settle_outcome == "UP"
            )
            if won:
                continue
            lines.append(
                f"{s.pattern} | close_pos={s.close_pos} vol_ratio={s.vol_ratio} | "
                f"{datetime.fromtimestamp(s.signal_time / 1000, tz=timezone.utc).strftime('%m-%d %H:%M')}"
            )
        return lines

    async def _active_version(self) -> tuple[str | None, dict | None]:
        async with async_session_factory() as session:
            stmt = (
                select(SceneParamVersion)
                .where(SceneParamVersion.status == "ACTIVE")
                .order_by(SceneParamVersion.activated_at.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalars().first()
        if row is None:
            return None, None
        return row.version, dict(row.params)

    async def _run_evaluation(self, trigger: str) -> None:
        logger.info("场景研究评估触发 [{}]：构建上下文 → LLM 评估 → 假设落库", trigger)
        live = await self._live_stats()
        active_version, active_params = await self._active_version()
        live_stats_str = "; ".join(
            f"{p}: n={s['n']} 胜率 {s['win_rate']:.1%}" if s["win_rate"] is not None else f"{p}: n=0"
            for p, s in live.items()
        ) or "尚无已结算场景信号"
        context = {
            "active_version": active_version or "v1-20260816（无版本行，按默认参数）",
            "active_params": active_params or {"close_pos_min": 0.85, "vol_ratio_min": 2.0},
            "baseline_full": "; ".join(
                f"场景{'①' if p == 'bull_exhaust' else '②'} {v['p']:.1%} (n={v['n']})"
                for p, v in BASELINE_FULL.items()
            ),
            "baseline_validation": "; ".join(
                f"场景{'①' if p == 'bull_exhaust' else '②'} {v['p']:.1%} [CI下界 {v['ci_lower']:.1%}] (n={v['n']})"
                for p, v in SCENE_BASELINES.items()
            ),
            "min_detectable": MIN_DETECTABLE_PP,
            "live_stats": live_stats_str,
            "failure_profile": await self._failure_profile(),
        }
        try:
            assessment = await self._researcher.evaluate(context)
        except Exception as exc:
            logger.error("场景研究评估失败（本轮放弃，等待下次触发）| {}", exc)
            return

        # 假设落库（PENDING_REVIEW）→ 逐个交 M3 科学裁决（同窗 A/B + 硬门禁 →
        # SHADOW/REJECTED + 邮件）；裁决是纯代码，与 LLM 无关
        saved_ids: list[int] = []
        try:
            async with async_session_factory() as session:
                for h in assessment.hypotheses[:3]:
                    version = f"h{int(time.time())}-{len(saved_ids)}"
                    row = SceneParamVersion(
                        version=version,
                        params=(active_params or {}) | dict(h.param_overrides),
                        status="PENDING_REVIEW",
                        proposed_by="llm-researcher",
                        review_note=f"[{trigger}] {h.change_suggestion} | 机制: {h.mechanism_reason[:200]} | 声称改善: {h.expected_impact_pp}pp",
                    )
                    session.add(row)
                    await session.flush()  # 拿 id
                    saved_ids.append(row.id)
                await session.commit()
        except Exception as exc:
            logger.error("假设落库失败（评估结果已在 llm_traces 留档）| {}", exc)
        self._last_run = time.time()
        # 刷新 T2 增量基线
        try:
            async with async_session_factory() as session:
                stmt = (
                    select(sa_func.count(FakeBreakoutSignal.id))
                    .where(FakeBreakoutSignal.status == "SETTLED")
                    .where(FakeBreakoutSignal.pattern.isnot(None))
                )
                self._last_settled_count = (await session.execute(stmt)).scalar() or 0
        except Exception:
            pass

        # M3 自动裁决链（每个假设同窗 A/B 回测 ~2-3 分钟，串行执行）
        if saved_ids:
            from .hypothesis_arbiter import HypothesisArbiter
            arbiter = HypothesisArbiter()
            for vid in saved_ids:
                try:
                    verdict = await arbiter.adjudicate(vid)
                    logger.info("假设 #{} 裁决: {}", vid,
                                "SHADOW（待人工审核）" if verdict and verdict.passed else "REJECTED")
                except Exception as exc:
                    logger.error("假设 #{} 裁决异常（可手动补跑 /adjudicate）| {}", vid, exc)
        logger.info(
            "场景研究评估完成 | maintain_status_quo={} | 假设落库 {} 条（已裁决）",
            assessment.maintain_status_quo, len(saved_ids),
        )
