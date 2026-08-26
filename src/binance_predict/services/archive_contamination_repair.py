"""归档污染一次性自愈：重建被 15m 样本污染的 5m 归档窗曲线并重落影子信号。

背景（2026-08）：主循环每轮同写 5m + 15m 两条样本（同一时间戳），归档查询
未过滤 market_period → 5m 情绪窗曲线混入 15m 报价（指纹：曲线内重复时间戳，
sample_count≈40=20 轮询×2 市场）。下游报价 edge 影子系统性扫出"更便宜一侧"
的 15m 幻影触发，WR/EV 记账虚高（拆对重算：记录价 cum +80.9 vs 真价 +17.5）。
写入路径已由 commit 5fe6356 修复（归档查询加 market_period=='5m'），
本模块修复历史数据：

    1. 重建：污染窗曲线从 prediction_market_samples 原始 5m 样本重建
       （原始采样默认永久保留，sample_retention_hours<=0）；
       entry_price/exit_price/outcome 是 BTC 现货口径，不受污染，原样保留。
    2. 重落：删除受影响影子信号（quote_edge 四版本按 window_start；
       x4 两版本按 window_start 或 target_window_start），按干净曲线重扫——
       quote_edge 重扫污染窗自身；x4 还需重扫污染窗的前一窗
       （触发可能来自干净前窗、结算价被污染目标窗污染）。
       v3 跨窗依赖：v3b 日高读取当日全部早窗 BTC 曲线，重建窗同一 UTC 日
       的全部后续窗 v3 行一并删除重落（v1/v2 无跨窗依赖，不受影响）。
    3. 实盘对账兼容：删除前先把引用旧信号 id 的订单 signal_id 置 NULL，
       heal 扫描会按 (version, window_start) 重新关联新信号。

幂等：污染指纹 = curve 内重复时间戳；重建后曲线干净，再次启动 0 命中即 no-op。
场景信号（fake_breakout）不受影响：本就交易 15m 市场、入场价取实时 15m 报价；
实盘执行器喂的是实时 5m 采样，同样不受影响。
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from sqlalchemy import delete as sa_delete
from sqlalchemy import select as sa_select
from sqlalchemy import update as sa_update

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import (
    MisalignmentSignal,
    PredictionMarketSample,
    SentimentWindow,
    TradeOrderModel,
)

# 15m 采样开始积累的时间（samples 表注释口径）：之前不存在 15m 样本，
# 不可能有污染 → 扫描下界，避免全表 JSON 解析。
CONTAMINATION_FLOOR_MS = int(
    datetime(2026, 8, 13, tzinfo=timezone.utc).timestamp() * 1000
)
WINDOW_MS = 5 * 60 * 1000
MIN_REBUILD_SAMPLES = 5          # 原始样本过少无法可靠重建 → 跳过
QUOTE_EDGE_VERSIONS = (
    "quote_momentum_v1", "quote_contrarian_v1",
    "quote_momentum_v2", "quote_contrarian_v2",
    "quote_contrarian_v3a", "quote_contrarian_v3b",
    "late_night_contrarian_v1",  # 深夜时段变体（无跨窗依赖，与 v1/v2 同等待遇）
    "late_night_contrarian_v2",  # 深夜门禁 v2（有日高跨窗依赖，归入 QE_DAY_HIGH_VERSIONS）
)
# 仅作版本族语义分组保留；后续窗删除口径已迁移至 QE_DAY_HIGH_VERSIONS（2026-08-27：
# v3a 无日高跨窗依赖，重建后存量行仍有效，不再参与删除重落）
QE_V3_VERSIONS = ("quote_contrarian_v3a", "quote_contrarian_v3b")
# 日高跨窗依赖版本（重建窗同日后续窗需删除重落）：v3b + 深夜门禁 v2（2026-08-27 只加不改）
QE_DAY_HIGH_VERSIONS = ("quote_contrarian_v3b", "late_night_contrarian_v2")
X4_VERSIONS = ("x4_v1", "x4_v2")


def has_duplicate_ts(curve: list | None) -> bool:
    """污染指纹：同一时间戳多个点（主循环每轮同写 5m+15m 两条样本）。"""
    ts_list = [p.get("t") for p in (curve or []) if p.get("t") is not None]
    return len(ts_list) != len(set(ts_list))


def window_is_contaminated(w: SentimentWindow) -> bool:
    return (
        has_duplicate_ts(w.curve_down_price)
        or has_duplicate_ts(w.curve_up_price)
        or has_duplicate_ts(w.curve_up_pct)
        or has_duplicate_ts(w.curve_down_pct)
    )


async def find_contaminated_windows(floor_ms: int) -> list[SentimentWindow]:
    """按 end_time 升序返回全部污染窗（曲线含重复时间戳）。"""
    async with async_session_factory() as session:
        rows = (await session.execute(
            sa_select(SentimentWindow)
            .where(SentimentWindow.start_time >= floor_ms)
            .order_by(SentimentWindow.end_time.asc())
        )).scalars().all()
    return [w for w in rows if window_is_contaminated(w)]


async def rebuild_window_from_raw_samples(w: SentimentWindow) -> SentimentWindow | None:
    """用原始 5m 采样重建曲线（与 main.py 归档器构建逻辑同口径）。

    注意：入参 w 是旧 session 的 detached 对象，必须在新 session 内重查后
    再改，否则变更不被跟踪、commit 静默空转。返回重建后的窗口对象
    （供后续重扫直接消费）；样本缺失无法重建时返回 None。
    entry_price/exit_price/actual_return/outcome 为 BTC 现货快照口径，
    与采样表污染无关，原样保留。
    """
    start_ms, end_ms = int(w.start_time), int(w.end_time)
    async with async_session_factory() as session:
        w = (await session.execute(
            sa_select(SentimentWindow).where(
                SentimentWindow.start_time == start_ms,
                SentimentWindow.end_time == end_ms,  # 与 uq_sw_start_end 对齐
            )
        )).scalar_one_or_none()
        if w is None:
            return None
        samples = (await session.execute(
            sa_select(PredictionMarketSample)
            .where(PredictionMarketSample.timestamp >= start_ms)
            .where(PredictionMarketSample.timestamp < end_ms)
            .where(PredictionMarketSample.market_period == "5m")
            .order_by(PredictionMarketSample.timestamp.asc())
        )).scalars().all()
        if len(samples) < MIN_REBUILD_SAMPLES:
            logger.warning(
                "污染自愈：窗口 {}~{} 原始 5m 样本仅 {} 条（<{}），跳过重建",
                start_ms, end_ms, len(samples), MIN_REBUILD_SAMPLES,
            )
            return None
        part_vals = [s.participants for s in samples if s.participants is not None]
        vol_vals = [s.trade_volume for s in samples if s.trade_volume is not None]
        w.curve_up_pct = [
            {"t": s.timestamp, "v": s.up_pct} for s in samples if s.up_pct is not None]
        w.curve_down_pct = [
            {"t": s.timestamp, "v": s.down_pct} for s in samples if s.down_pct is not None]
        w.curve_up_price = [
            {"t": s.timestamp, "v": s.up_price} for s in samples if s.up_price is not None]
        w.curve_down_price = [
            {"t": s.timestamp, "v": s.down_price} for s in samples if s.down_price is not None]
        w.curve_participants = [
            {"t": s.timestamp, "v": s.participants}
            for s in samples if s.participants is not None]
        w.curve_trade_volume = [
            {"t": s.timestamp, "v": s.trade_volume}
            for s in samples if s.trade_volume is not None]
        w.curve_btc_price = [
            {"t": s.timestamp, "v": s.btc_price} for s in samples if s.btc_price is not None]
        w.sample_count = len(samples)
        w.avg_participants = sum(part_vals) / len(part_vals) if part_vals else None
        w.avg_trade_volume = sum(vol_vals) / len(vol_vals) if vol_vals else None
        await session.commit()
        await session.refresh(w)  # 重载全部属性，detached 后仍可安全读取
        return w


def quote_edge_affected_starts(contaminated: set[int]) -> set[int]:
    """quote_edge 信号：本窗即目标窗 → window_start ∈ 污染集。"""
    return set(contaminated)


def qe_v3_successor_starts(rebuilt_starts: set[int]) -> set[int]:
    """日高跨窗依赖重扫集：重建窗同一 UTC 日的全部后续窗（不含重建窗自身）。

    日高依赖版本（v3b / 深夜门禁 v2）读取当日 ≤本窗的全部已归档窗 BTC 曲线：
    污染窗重建后，同日后续窗的判定基准全部变化，其存量行基于重建前的脏曲线，
    必须删除后按干净曲线重落。仅对成功重建的窗展开（跳过窗曲线仍脏，
    重扫无意义）；无跨窗依赖版本不受影响。升序重扫天然保证日高口径一致。
    """
    out: set[int] = set()
    for c in rebuilt_starts:
        day_end = ((c // 86_400_000) + 1) * 86_400_000
        out.update(range(c + WINDOW_MS, day_end, WINDOW_MS))
    return out


def x4_affected_condition_values(contaminated: set[int]) -> set[int]:
    """x4 信号：触发窗（end_pct 判据）或结算目标窗（入场报价）任一被污染。"""
    return set(contaminated)


def x4_reprocess_starts(contaminated: set[int]) -> set[int]:
    """x4 重扫窗集合 = 污染窗 ∪ 前一窗 ∪ 后一窗。

    前一窗：可能是干净触发源（其信号的结算价被污染目标窗污染，已被删除），
    重扫后按干净曲线重新触发 PENDING，轮到目标窗时正常结算。
    后一窗：污染窗自身触发的重建信号（target=后一窗）需要后一窗来结算；
    若后一窗干净且不在集合内，PENDING 无人结算会被检测器误标 EXPIRED。
    升序迭代天然保证先触发后结算。
    """
    prevs = {c - WINDOW_MS for c in contaminated}
    nexts = {c + WINDOW_MS for c in contaminated}
    return contaminated | prevs | nexts


async def _delete_affected_signals(contaminated: set[int],
                                  v3_successors: set[int]) -> dict:
    """删除受影响信号；先置空订单 signal_id 防悬挂（heal 扫描会重新关联）。

    v3_successors：重建窗同日后续窗，只删日高依赖版本行（无跨窗依赖版本不删）。
    """
    qe_starts = quote_edge_affected_starts(contaminated)
    x4_starts = x4_affected_condition_values(contaminated)
    deleted = {"quote_edge": 0, "quote_edge_v3_succ": 0, "x4": 0, "orders_unlinked": 0}
    if not qe_starts and not x4_starts and not v3_successors:
        return deleted
    async with async_session_factory() as session:
        qe_ids = (await session.execute(
            sa_select(MisalignmentSignal.id).where(
                MisalignmentSignal.version.in_(QUOTE_EDGE_VERSIONS),
                MisalignmentSignal.window_start.in_(qe_starts),
            )
        )).scalars().all()
        v3_succ_ids = (await session.execute(
            sa_select(MisalignmentSignal.id).where(
                MisalignmentSignal.version.in_(QE_DAY_HIGH_VERSIONS),
                MisalignmentSignal.window_start.in_(v3_successors),
            )
        )).scalars().all() if v3_successors else []
        x4_ids = (await session.execute(
            sa_select(MisalignmentSignal.id).where(
                MisalignmentSignal.version.in_(X4_VERSIONS),
                (MisalignmentSignal.window_start.in_(x4_starts)
                 | MisalignmentSignal.target_window_start.in_(x4_starts)),
            )
        )).scalars().all()
        if qe_ids:
            res = await session.execute(
                sa_update(TradeOrderModel)
                .where(TradeOrderModel.signal_id.in_(qe_ids))
                .values(signal_id=None)
            )
            deleted["orders_unlinked"] = int(res.rowcount or 0)
            res = await session.execute(
                sa_delete(MisalignmentSignal).where(MisalignmentSignal.id.in_(qe_ids))
            )
            deleted["quote_edge"] = int(res.rowcount or 0)
        if v3_succ_ids:
            res = await session.execute(
                sa_delete(MisalignmentSignal).where(MisalignmentSignal.id.in_(v3_succ_ids))
            )
            deleted["quote_edge_v3_succ"] = int(res.rowcount or 0)
        if x4_ids:
            res = await session.execute(
                sa_delete(MisalignmentSignal).where(MisalignmentSignal.id.in_(x4_ids))
            )
            deleted["x4"] = int(res.rowcount or 0)
        await session.commit()
    return deleted


async def repair_contaminated_archives() -> dict:
    """启动时一次性自愈入口（幂等）。返回统计 dict 供日志/诊断。"""
    stats = {
        "contaminated": 0, "rebuilt": 0, "skipped_no_samples": 0,
        "deleted_quote_edge": 0, "deleted_x4": 0, "orders_unlinked": 0,
        "rescanned": 0,
    }
    contaminated_wins = await find_contaminated_windows(CONTAMINATION_FLOOR_MS)
    stats["contaminated"] = len(contaminated_wins)
    if not contaminated_wins:
        return stats
    logger.warning(
        "污染自愈：发现 {} 个被 15m 样本污染的归档窗（{}~{}），开始重建",
        len(contaminated_wins), contaminated_wins[0].start_time,
        contaminated_wins[-1].start_time,
    )

    # --- 1) 重建曲线 ---
    rebuilt_wins: list[SentimentWindow] = []
    for w in contaminated_wins:
        try:
            rw = await rebuild_window_from_raw_samples(w)
            if rw is not None:
                rebuilt_wins.append(rw)
            else:
                stats["skipped_no_samples"] += 1
        except Exception as exc:
            stats["skipped_no_samples"] += 1
            logger.warning("污染自愈：重建失败（跳过）| window {} | {}",
                           w.start_time, exc)
    stats["rebuilt"] = len(rebuilt_wins)

    # --- 2) 删除受影响信号（全部污染窗：重建窗重落，跳过窗曲线不可信、
    # 其信号同样不可信，一并删除避免虚高 EV 继续计入统计）---
    contaminated_starts = {int(w.start_time) for w in contaminated_wins}
    rebuilt_starts = {int(w.start_time) for w in rebuilt_wins}
    # v3b/深夜门禁 v2 跨窗依赖：重建窗同日后续窗的日高依赖行基于重建前脏曲线，一并删除重落
    v3_successors = qe_v3_successor_starts(rebuilt_starts) - contaminated_starts
    deleted = await _delete_affected_signals(contaminated_starts, v3_successors)
    stats["deleted_quote_edge"] = deleted["quote_edge"]
    stats["deleted_x4"] = deleted["x4"]
    stats["orders_unlinked"] = deleted["orders_unlinked"]
    if not rebuilt_wins:
        return stats

    # --- 3) 按干净曲线重落信号（复用检测器单窗逻辑，幂等约束防重）---
    from .misalignment_detector import MisalignmentDetector
    from .quote_edge_detector import QuoteEdgeDetector

    by_start = {int(w.start_time): w for w in rebuilt_wins}
    # quote_edge 重扫集 = 重建窗 ∪ 日高依赖同日后续窗（后者从 DB 加载，
    # 可能不存在/缺口）；升序处理保证日高口径与实时路径一致，
    # 后续窗已有的无跨窗依赖行由 dup 查重跳过，只补日高依赖行。
    qe_detector = QuoteEdgeDetector()
    missing_succ = sorted(s for s in v3_successors if s not in by_start)
    if missing_succ:
        async with async_session_factory() as session:
            succ_wins = (await session.execute(
                sa_select(SentimentWindow)
                .where(SentimentWindow.start_time.in_(missing_succ))
            )).scalars().all()
        for w in succ_wins:
            by_start[int(w.start_time)] = w
    for s in sorted(set(by_start) & (rebuilt_starts | v3_successors)):
        try:
            await qe_detector._process_window(by_start[s])
            stats["rescanned"] += 1
        except Exception as exc:
            logger.warning("污染自愈：quote_edge 重扫失败 | window {} | {}", s, exc)
    # x4：干净前窗（触发源重建）与干净后窗（结算重建信号）同样需参与重扫，
    # 缺失的从 DB 加载；升序保证先触发后结算。
    neighbor_starts = sorted(
        s for s in x4_reprocess_starts(rebuilt_starts) if s not in by_start
    )
    if neighbor_starts:
        async with async_session_factory() as session:
            neighbor_wins = (await session.execute(
                sa_select(SentimentWindow)
                .where(SentimentWindow.start_time.in_(neighbor_starts))
            )).scalars().all()
        for w in neighbor_wins:
            by_start[int(w.start_time)] = w
    x4_detector = MisalignmentDetector()
    for s in sorted(set(by_start) & x4_reprocess_starts(rebuilt_starts)):
        try:
            await x4_detector._process_window(by_start[s])
        except Exception as exc:
            logger.warning("污染自愈：x4 重扫失败 | window {} | {}", s, exc)

    logger.info(
        "污染自愈完成 | 污染窗 {} 重建 {} 跳过 {} | 删除信号 qe={} x4={} "
        "| 订单解链 {} | quote_edge 重扫 {}",
        stats["contaminated"], stats["rebuilt"], stats["skipped_no_samples"],
        stats["deleted_quote_edge"], stats["deleted_x4"],
        stats["orders_unlinked"], stats["rescanned"],
    )
    return stats
