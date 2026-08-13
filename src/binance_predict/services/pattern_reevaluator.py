"""
模式定期重回测与分级服务（模式池无限进化引擎）。

用户规则：Agent 闭环只发现模式、不下注；模式池分 S/A/B/C 四级；
新数据累积到阈值后，对每个模式用全量历史窗口重新回测，与前一次
回测快照做细节对比，驱动模式无限进化（晋级/保级/降级/衰减预警）。

核心机制：
1. 后台循环每 reeval_check_interval 检查一次：自上次回测数据终点以来，
   新归档且已标注 outcome 的窗口数 >= reeval_min_new_windows 时触发全量重回测
2. 重回测 = 程序谓词重放（确定性，不调 LLM）：
   对每个有 predicate 的 ACTIVE/EVOLVING 模式，按其 binning_version 的
   分箱快照符号化全部历史窗口（同 version 共享符号化结果，避免重复计算），
   逐窗口 evaluate_predicate 判定命中，命中窗口 outcome 与模式方向比对
3. 每次回测落表 pattern_backtest_runs（胜率/Wilson CI/费后 EV/按月分段），
   并与上一次快照对比生成 delta_vs_prev（漂移/新增样本表现/定级建议）
4. tier 定级规则（简单透明）：
   - S: 样本>=50 且 Wilson 下界>=0.55
   - A: 样本>=30 且 Wilson 下界>=0.52
   - B: 样本>=10 且 胜率>=0.50
   - C: 其余（新发现模式的默认池）
   - 衰减预警：最近分段（最新 25% 数据）样本>=10 且胜率<0.50 → 降一级

费后 EV 口径与回测脚本一致：0.5 定价，赢 +0.9216 / 输 -1。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import asc, desc, func, select

from ..config.settings import settings
from ..db.engine import async_session_factory
from ..db.models import (
    BinningSnapshotModel,
    PatternBacktestRun,
    PatternMemory,
    SentimentWindow,
)
from .predicates import evaluate_predicate
from .symbolizer import BinningSnapshot, build_window_view

# 费后盈亏（0.5 定价口径，与回测脚本 _bet_pnl(win, 0.5) 一致）
_WIN_PNL = (1.0 - 0.02) / (0.5 + 0.01) - 1.0  # ≈ +0.9216
_LOSS_PNL = -1.0

_TIER_ORDER = ("C", "B", "A", "S")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Wilson score 95% 置信区间。"""
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def _tier_of(sample: int, win_rate: float, wilson_lower: float | None) -> str:
    """分级规则（简单透明，只升保降由衰减预警另行处理）。"""
    wl = wilson_lower if wilson_lower is not None else 0.0
    if sample >= 50 and wl >= 0.55:
        return "S"
    if sample >= 30 and wl >= 0.52:
        return "A"
    if sample >= 10 and win_rate >= 0.50:
        return "B"
    return "C"


def _demote(tier: str) -> str:
    """降一级（衰减预警用）。"""
    i = _TIER_ORDER.index(tier)
    return _TIER_ORDER[max(0, i - 1)]


class PatternReevaluator:
    """模式定期重回测调度器。生命周期由 main.py lifespan 管理。"""

    def __init__(self) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        # 上次回测覆盖的数据终点（ms），启动时从最近一条 run 恢复
        self._last_data_end: int = 0
        self._last_run_at: float = 0.0
        self._last_summary: dict | None = None
        # 并发互斥：后台调度（DATA_THRESHOLD）与手动 API（MANUAL）共用 run_all，
        # 同一事件循环内可能交错触发，加锁避免重复快照/对比基准互相污染/tier 双写。
        self._run_lock = asyncio.Lock()

    # ==================================================================
    # 生命周期
    # ==================================================================

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self._restore_last_data_end()
        self._task = asyncio.create_task(self._loop(), name="pattern_reevaluator")
        logger.info(
            "模式重回测调度器启动 | 检查间隔={}s | 新窗口阈值={}",
            settings.pattern_reeval_check_interval,
            settings.pattern_reeval_min_new_windows,
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

    async def _restore_last_data_end(self) -> None:
        try:
            async with async_session_factory() as session:
                stmt = select(func.max(PatternBacktestRun.data_end))
                self._last_data_end = (await session.execute(stmt)).scalar() or 0
        except Exception as exc:
            logger.warning("模式重回测：恢复上次数据终点失败 | {}", exc)
            self._last_data_end = 0

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(settings.pattern_reeval_check_interval)
                if not self._running:
                    break
                new_windows = await self._count_new_windows()
                if new_windows >= settings.pattern_reeval_min_new_windows:
                    logger.info(
                        "模式重回测触发 | 新增已标注窗口 {} >= 阈值 {}",
                        new_windows, settings.pattern_reeval_min_new_windows,
                    )
                    await self.run_all(trigger="DATA_THRESHOLD")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "模式重回测循环异常 | error_type={} | error={}",
                    type(exc).__name__, exc,
                )

    async def _count_new_windows(self) -> int:
        try:
            async with async_session_factory() as session:
                stmt = select(func.count(SentimentWindow.id)).where(
                    SentimentWindow.outcome.isnot(None),
                    SentimentWindow.end_time > self._last_data_end,
                )
                return (await session.execute(stmt)).scalar() or 0
        except Exception:
            return 0

    # ==================================================================
    # 主入口：全量模式重回测（循环触发 + API 手动触发共用）
    # ==================================================================

    async def run_all(self, trigger: str = "MANUAL") -> dict:
        """对全部谓词模式执行重回测。返回摘要（供 API 与日志）。

        并发互斥：后台调度与手动触发共用本方法，加锁避免重复快照。
        正在运行时手动触发直接返回 busy，不排队（重测幂等，稍后再触发即可）。
        """
        if self._run_lock.locked():
            logger.info("模式重回测：已有回测运行中，本次触发跳过（trigger={}）", trigger)
            return {"busy": True, "message": "回测正在运行，请稍后重试"}

        t0 = time.monotonic()

        async with self._run_lock:
            async with async_session_factory() as session:
                # 1. 加载候选模式（有谓词 + ACTIVE/EVOLVING）
                patterns = (await session.execute(
                    select(PatternMemory).where(
                        PatternMemory.predicate.isnot(None),
                        PatternMemory.status.in_(("ACTIVE", "EVOLVING")),
                    )
                )).scalars().all()

                if not patterns:
                    logger.info("模式重回测：无谓词模式，跳过")
                    return {"patterns": 0, "runs": 0, "duration_s": 0.0}

                # 2. 加载已标注窗口（取最新 N 条，NOISE 除外）。desc+limit 保留最新窗口，
                #    再 reverse 回时间升序（CodeReview Major-2：原 asc+limit 会丢弃最新数据）
                rows = (await session.execute(
                    select(SentimentWindow)
                    .where(
                        SentimentWindow.outcome.isnot(None),
                        SentimentWindow.outcome != "NOISE",
                    )
                    .order_by(desc(SentimentWindow.start_time))
                    .limit(settings.pattern_reeval_max_windows)
                )).scalars().all()
                windows = list(reversed(rows))

                if not windows:
                    logger.info("模式重回测：无已标注窗口，跳过")
                    return {"patterns": len(patterns), "runs": 0, "duration_s": 0.0}

                # 3. 按 binning_version 分组模式；每组加载对应快照
                versions = {p.binning_version for p in patterns if p.binning_version}
                snapshots_by_version = await self._load_snapshots(session, versions)

                # 4. 每个 version 预符号化全部窗口一次（同 version 模式共享）
                views_by_version: dict[str, list] = {}
                for version in versions:
                    snaps = snapshots_by_version.get(version)
                    if not snaps:
                        continue
                    views = []
                    for w in windows:
                        try:
                            views.append(build_window_view(self._window_dict(w), snaps))
                        except Exception:
                            continue
                    views_by_version[version] = views

                # 5. 逐模式重放 + 落表 + 定级
                data_start = int(windows[0].start_time)
                data_end = int(windows[-1].end_time)
                runs = 0
                tier_changes: list[str] = []
                for p in patterns:
                    views = views_by_version.get(p.binning_version or "")
                    if views is None:
                        logger.warning(
                            "模式重回测：模式 id={} 的 binning_version={} 无快照，跳过",
                            p.id, p.binning_version,
                        )
                        continue
                    try:
                        changed = await self._reevaluate_one(
                            session, p, views, windows, data_start, data_end, trigger
                        )
                        runs += 1
                        if changed:
                            tier_changes.append(changed)
                    except Exception as exc:
                        logger.error(
                            "模式重回测：模式 id={} 重放失败 | error_type={} | error={}",
                            p.id, type(exc).__name__, exc,
                        )

                await session.commit()

        # CodeReview Minor-4：0 产出时不推进数据终点，避免这批新窗口的阈值触发被永久消费
        if runs > 0:
            self._last_data_end = data_end
        self._last_run_at = time.time()
        duration = time.monotonic() - t0
        summary = {
            "patterns": len(patterns),
            "runs": runs,
            "windows": len(windows),
            "data_start": data_start,
            "data_end": data_end,
            "tier_changes": tier_changes,
            "duration_s": round(duration, 2),
        }
        self._last_summary = summary
        logger.info(
            "模式重回测完成 | 模式 {} 个 | 窗口 {} 个 | 定级变更 {} 起 | 耗时 {:.1f}s",
            len(patterns), len(windows), len(tier_changes), duration,
        )
        return summary

    # ==================================================================
    # 单模式重放
    # ==================================================================

    async def _reevaluate_one(
        self,
        session,
        pattern: PatternMemory,
        views: list,
        windows: list[SentimentWindow],
        data_start: int,
        data_end: int,
        trigger: str,
    ) -> str | None:
        """单模式谓词重放 + 落表 + 定级。返回 tier 变更描述（无变更返回 None）。"""
        assert len(views) == len(windows) or True  # views 失败窗口被跳过，用 zip 对齐

        # 逐窗口判定（views 与 windows 同序；build 失败的窗口已被跳过，
        # 用 window_id 对齐防御）
        wid_to_outcome = {w.id: w.outcome for w in windows}
        hits: list[tuple[int, str]] = []  # (start_time, outcome)
        for v in views:
            if v.window_id is None:
                continue
            outcome = wid_to_outcome.get(v.window_id)
            if outcome is None:
                continue
            try:
                if evaluate_predicate(pattern.predicate, v):
                    hits.append((v.start_time, outcome))
            except Exception:
                continue

        n = len(hits)
        k = sum(1 for _, oc in hits if oc == pattern.predicted_direction)
        win_rate = k / n if n else 0.0
        wl, wu = _wilson(k, n)
        ev = win_rate * _WIN_PNL + (1 - win_rate) * _LOSS_PNL if n else None

        # 按月分段统计（纵向对比细节）
        segments: dict[str, dict] = {}
        for ts, oc in hits:
            month = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")
            seg = segments.setdefault(month, {"n": 0, "k": 0})
            seg["n"] += 1
            if oc == pattern.predicted_direction:
                seg["k"] += 1
        segment_stats = {
            m: {"n": s["n"], "k": s["k"], "win_rate": s["k"] / s["n"] if s["n"] else 0.0}
            for m, s in sorted(segments.items())
        }

        # 最近分段（最新 25% 数据）衰减检测
        latest_seg_wr: float | None = None
        latest_seg_n = 0
        if hits:
            cutoff_idx = int(len(hits) * 0.75)
            recent = hits[cutoff_idx:]
            latest_seg_n = len(recent)
            if latest_seg_n:
                rk = sum(1 for _, oc in recent if oc == pattern.predicted_direction)
                latest_seg_wr = rk / latest_seg_n

        # 上一次回测快照（纵向对比基准）
        prev = (await session.execute(
            select(PatternBacktestRun)
            .where(PatternBacktestRun.pattern_id == pattern.id)
            .order_by(desc(PatternBacktestRun.created_at))
            .limit(1)
        )).scalar_one_or_none()

        # --- tier 定级 ---
        old_tier = pattern.tier or "C"
        new_tier = _tier_of(n, win_rate, wl)
        decay_warning = False
        if latest_seg_n >= 10 and latest_seg_wr is not None and latest_seg_wr < 0.50:
            demoted = _demote(new_tier)
            if demoted != new_tier:
                decay_warning = True
                new_tier = demoted

        # --- delta_vs_prev ---
        delta = self._build_delta(
            prev=prev,
            cur={
                "sample_count": n,
                "correct_count": k,
                "win_rate": win_rate,
                "wilson_lower": wl,
                "ev_after_fee": ev,
                "segment_stats": segment_stats,
                "latest_seg_win_rate": latest_seg_wr,
                "latest_seg_n": latest_seg_n,
            },
            old_tier=old_tier,
            new_tier=new_tier,
            decay_warning=decay_warning,
        )

        run = PatternBacktestRun(
            pattern_id=pattern.id,
            data_start=data_start,
            data_end=data_end,
            sample_count=n,
            correct_count=k,
            win_rate=win_rate,
            wilson_lower=wl,
            wilson_upper=wu,
            ev_after_fee=ev,
            segment_stats=segment_stats,
            delta_vs_prev=delta,
            trigger_reason=trigger,
        )
        session.add(run)

        # --- tier 变更 ---
        if new_tier != old_tier:
            pattern.tier = new_tier
            logger.info(
                "模式定级变更 | id={} name='{}' | {} → {} | 样本={} 胜率={:.1%} "
                "Wilson下界={} 最近段胜率={}{}",
                pattern.id, pattern.pattern_name, old_tier, new_tier,
                n, win_rate,
                f"{wl:.3f}" if wl is not None else "N/A",
                f"{latest_seg_wr:.1%}" if latest_seg_wr is not None else "N/A",
                "（衰减降级）" if decay_warning else "",
            )
            return f"{pattern.pattern_name}: {old_tier}→{new_tier}"
        return None

    def _build_delta(
        self,
        prev: PatternBacktestRun | None,
        cur: dict,
        old_tier: str,
        new_tier: str,
        decay_warning: bool,
    ) -> dict:
        """与上一次回测的细节对比。"""
        delta: dict = {
            "tier_change": (
                {"from": old_tier, "to": new_tier} if new_tier != old_tier else None
            ),
            "decay_warning": decay_warning,
            "latest_seg_win_rate": cur["latest_seg_win_rate"],
            "latest_seg_n": cur["latest_seg_n"],
        }
        if prev is None:
            delta["note"] = "首次回测，无对比基准"
            return delta

        new_samples = cur["sample_count"] - prev.sample_count
        new_correct = cur["correct_count"] - prev.correct_count
        delta["prev_run_id"] = prev.id
        delta["prev_win_rate"] = prev.win_rate
        delta["win_rate_drift"] = cur["win_rate"] - prev.win_rate
        delta["new_samples"] = new_samples
        delta["new_samples_win_rate"] = (
            new_correct / new_samples if new_samples > 0 else None
        )

        # 文本建议（供前端直接展示）
        suggestions: list[str] = []
        if delta["win_rate_drift"] is not None and abs(delta["win_rate_drift"]) >= 0.05:
            direction = "上升" if delta["win_rate_drift"] > 0 else "下滑"
            suggestions.append(
                f"胜率{direction} {abs(delta['win_rate_drift']):.1%}（{prev.win_rate:.1%} → {cur['win_rate']:.1%}）"
            )
        if new_samples > 0 and delta["new_samples_win_rate"] is not None:
            nwr = delta["new_samples_win_rate"]
            if nwr < 0.45:
                suggestions.append(f"新增 {new_samples} 样本胜率仅 {nwr:.1%}，信号可能正在失效")
            elif nwr > 0.65:
                suggestions.append(f"新增 {new_samples} 样本胜率 {nwr:.1%}，信号持续有效")
        if decay_warning:
            suggestions.append("最近 25% 数据胜率 <50%，已触发衰减降级")
        if new_tier != old_tier:
            suggestions.append(f"模式池 {old_tier} → {new_tier}")
        delta["suggestions"] = suggestions or ["表现平稳，无显著变化"]
        return delta

    # ==================================================================
    # 内部辅助
    # ==================================================================

    @staticmethod
    def _window_dict(w: SentimentWindow) -> dict:
        return {
            "id": w.id,
            "start_time": w.start_time,
            "outcome": w.outcome,
            "curve_up_pct": w.curve_up_pct,
            "curve_btc_price": w.curve_btc_price,
            "curve_trade_volume": w.curve_trade_volume,
        }

    @staticmethod
    async def _load_snapshots(
        session, versions: set[str]
    ) -> dict[str, dict[str, BinningSnapshot]]:
        """按 version 加载三通道分箱快照（与 predict 仪器精度对齐）。"""
        if not versions:
            return {}
        rows = (await session.execute(
            select(BinningSnapshotModel).where(
                BinningSnapshotModel.version.in_(versions)
            )
        )).scalars().all()
        grouped: dict[str, dict[str, BinningSnapshot]] = defaultdict(dict)
        for r in rows:
            grouped[r.version][r.channel] = BinningSnapshot(
                version=r.version,
                edges=tuple(float(x) for x in r.edges),  # type: ignore[arg-type]
                created_at_epoch=r.created_at_epoch,
                sample_count=r.sample_count,
            )
        return dict(grouped)

    @property
    def status_snapshot(self) -> dict:
        return {
            "running": self._running,
            "last_data_end": self._last_data_end,
            "last_run_at": self._last_run_at,
            "last_summary": self._last_summary,
        }


# 进程内单例
pattern_reevaluator = PatternReevaluator()
