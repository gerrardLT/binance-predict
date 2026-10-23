"""交易结算器（P0-2）：回读结算源回填 FILLED 订单的输赢/盈亏。

结算源分流（三口径，_settle_row 入口硬编码，防错配）：
1. 5m 订单（默认）：回读 SentimentWindow——每 5 分钟归档器按 entry/exit
   价判定 outcome（UP/DOWN/NOISE），start_time 与订单 window_start 同为
   5m 窗口起点 ms，可直接对齐。
2. 15m 场景订单（market_period='15m' 且有 scene_signal_id）：回读
   FakeBreakoutSignal.settle_outcome（周期锚点口径：P(E) vs P(S)，与币安
   15m 市场真实结算一致）。**为何必须分流**：15m 周期起点与 5m 窗口起点
   数值重合（900s 网格 ⊂ 300s 网格），若走 SentimentWindow 会被同名 5m 窗
   错口径结算输赢——这是多通道改造最隐蔽的坑。
3. K 线影子族订单（market_period='15m' 且 scene_signal_id IS NULL）：
   回读 KlineShadowSignal（version=signal_version ∧ target_bar_start=
   window_start）的 settle_outcome/settle_close——s2_cond 条件单族
   （t4/t5d）与 nextbar 15m 族（hm_inside_15m_v2/ih_inside_15m_v2）的信号
   行落 kline_shadow_signals，下单路径无 signal_id 关联列。5m Rev2
   （hm_inside_5m_v2）仍走 5m SentimentWindow，因为它的结算源就是 5m 窗口归档。
   **为何必须独立
   成路**：2026-09-10 生产实锤，旧口径把这类行塞进 2（缺 scene_signal_id →
   CRITICAL + EXPIRED/win=NULL/pnl=0），FILLED 真单的真实亏损被抹平
   （id=466 真实 -1.00 记为 +0.00），统计口径整体失真。

扫描锚点 = trade_orders.settled_at IS NULL（部分索引
ix_trade_orders_settle_pending 只覆盖待结算行，空转亚毫秒）。

结算规则：
    outcome ∈ {UP, DOWN}：win = (direction == outcome)；
        赢：优先 shares − amount（shares = quote_json.filledShareQty，币安实际
        到手股数已扣 marketProviderFee，对齐实现盈亏；2026-08-28），
        回退 amount/avg_price − amount；输：−amount。
        amount = amount_in/1e18，avg_price = quote_json.averagePrice。
    outcome == NOISE：win=None, pnl=0.0（return=0 极罕见 16/3522，
        真实赔付规则未知，保守记 0；settled_at 锚点天然终止重扫）。
    无匹配窗口（归档缺失/未归档/outcome NULL）且 created_at 超 24h：
        settle_outcome=EXPIRED, win=None, pnl=0.0（防无限重扫；
        归档器恢复后可手动重算）。
    无匹配窗口且未超 24h：跳过，下轮重试。

pnl 口径注释：本地估算（优先币安实际到手股数，回退无费成交均价口径），
币安侧验证阶段校正——不用 amount_out（落库的是 quote 预估值非成交实际值）。

官方结算优先（2026-10-23）：币安 Up-or-Down 市场官方结算源是 Chainlink
BTC/USDT top-of-book mid price，不是币安现货 K 线。历史核对 1175 笔中 9 笔
（0.77%）在 0.001~1.73bp 微幅窗口方向相反。故 _settle_row 先查官方
winning_outcome（official_settlement），命中则直接采信、settle_price=None；
查不到/查询失败才回退下方各自 K 线口径。官方返回 PENDING（市场未出结果）时
30 分钟内下轮重试。

每日对账：reconcile_official 回看最近 2 天已结算订单，与官方结果不一致时
CRITICAL 日志 + 企微告警，不自动改财务数据（人工确认后用一次性 workflow 修）。

结构克隆 quote_edge_detector 范式（60s 轮询 + per-row 独立
commit/rollback + 异常不中断）；无 backscan——锚点是 settled_at
IS NULL，启动首次 poll_once 即全量回补历史未结算行，无需水位。
"""
from __future__ import annotations

import asyncio
from typing import Callable
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select as sa_select, update as sa_update

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import FakeBreakoutSignal, KlineShadowSignal, SentimentWindow, TradeOrderModel
from binance_predict.services.candlestick_shadow_detector import CANDLESTICK_SIGNAL_IDS, EVENT_VERSION_BY_TF
from binance_predict.services.official_settlement import get_official_settlement_client
from binance_predict.services.shadow_execution_store import patch_assessment
from binance_predict.services.shadow_execution_types import AssessmentPatch, ReasonCode, TerminalStage
from binance_predict.services.wechat_notifier import wechat_notifier

# 2026-10-23：5m nextbar 管道（hm_quality/hm_moderate）也落 kline_shadow_signals，
# 结算优先按影子行（version + target_bar_start）对齐；影子行缺失（gate 关闭 /
# 检测器未回补）时回退既有 5m SentimentWindow 路径，避免真单被记 EXPIRED/pnl=0。
_KLINE_SHADOW_5M_VERSIONS = frozenset({"hm_quality_5m_v1", "hm_moderate_5m_v1"})

logger = logging.getLogger(__name__)

POLL_INTERVAL = 60.0                    # 轮询间隔（秒）
SETTLE_DELAY = timedelta(minutes=7)     # 等 5m 窗口归档（boundary+15s）+ 冗余
EXPIRE_AFTER = timedelta(hours=24)      # 无归档超时 → EXPIRED
SCAN_BATCH = 50                         # 单轮最多结算行数
PENDING_RETRY = timedelta(minutes=30)   # 官方结果 PENDING 时的重试窗口
RECONCILE_HOUR_UTC = 3                  # 每日对账时刻（UTC；北京 11:00）
RECONCILE_DAYS = 2                      # 对账回看天数

_AUTO = object()  # official_client 未注入哨兵（测试可显式传 None 禁网）


class TradeSettler:
    """FILLED 未结算订单结算器：回读 SentimentWindow 回填输赢/盈亏。

    只读窗口 + 回填结算字段，零资金风险（lifespan 无开关常开）。
    """

    def __init__(self, on_settled: "Callable[[int, str | None], None] | None" = None,
                 official_client: object = _AUTO) -> None:
        self._running = False
        self._task: asyncio.Task | None = None
        self._reconcile_task: asyncio.Task | None = None
        self._settled_count = 0
        # 官方结算客户端：默认首次使用时自动取进程单例（未配置 key → None，
        # 回退 K 线）；测试可显式传 None 彻底禁网。
        self._official = None if official_client is _AUTO else official_client
        self._official_auto = official_client is _AUTO
        # 对账互斥：每日任务与手动端点不并发打官方源
        self._reconcile_lock = asyncio.Lock()
        # 结算成功回调（order_id, channel）：LLM 归因钩子（2026-10-19）。
        # 调用方负责 fire-and-forget；此处 try/except 双保险——回调故障
        # 绝不影响结算事务本身。
        self._on_settled = on_settled

    def _fire_on_settled(self, row: TradeOrderModel) -> None:
        if self._on_settled is None:
            return
        try:
            self._on_settled(int(row.id), row.signal_version)
        except Exception as exc:
            logger.warning("结算回调异常（不影响结算）: %s", exc)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="trade_settler")
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(), name="settlement_reconciler")
        logger.info(
            "交易结算器启动 | %ds 轮询 | 锚点 settled_at IS NULL | 延迟 %s 等归档 | "
            "官方结算源=%s | 每日 %02d:00 UTC 对账",
            int(POLL_INTERVAL), SETTLE_DELAY,
            "ON" if self._resolve_official_client() is not None else "OFF（回退 K 线）",
            RECONCILE_HOUR_UTC,
        )

    async def stop(self) -> None:
        self._running = False
        for task in (self._task, self._reconcile_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._reconcile_task = None
        logger.info("交易结算器已停止 | 累计结算 %d 单", self._settled_count)

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("交易结算器：循环异常 | %s | %s", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _reconcile_loop(self) -> None:
        """每日 UTC 03:00 对账；启动 15 分钟后先补跑一次（防部署恰好跨过 03:00）。"""
        first = True
        while self._running:
            if first:
                delay = 15 * 60
                first = False
            else:
                now = datetime.now(timezone.utc)
                next_run = now.replace(
                    hour=RECONCILE_HOUR_UTC, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                delay = (next_run - now).total_seconds()
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            if not self._running:
                break
            try:
                result = await self.reconcile_official(days=RECONCILE_DAYS)
                logger.info(
                    "每日结算对账完成 | checked=%s | mismatches=%s",
                    result.get("checked"),
                    len(result.get("mismatches") or []),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("每日结算对账异常 | %s: %s", type(exc).__name__, exc)

    async def reconcile_official(self, days: int = RECONCILE_DAYS) -> dict:
        """对账最近 N 天已结算订单与官方 winning_outcome。

        不一致时 CRITICAL 日志 + 企微告警，**不自动改财务数据**：
        人工确认后用一次性修复 workflow 更正（避免自动改写历史盈亏）。
        """
        client = self._resolve_official_client()
        if client is None:
            return {"error": "官方结算源未配置（PREDEXON_API_KEY）",
                    "checked": 0, "mismatches": []}
        days = max(1, min(int(days), 7))
        async with self._reconcile_lock:
            since = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = (
                sa_select(TradeOrderModel)
                .where(
                    TradeOrderModel.status == "FILLED",
                    TradeOrderModel.settled_at.isnot(None),
                    TradeOrderModel.settled_at >= since,
                    TradeOrderModel.window_start.isnot(None),
                    TradeOrderModel.side == "BUY",
                    # EXPIRED 也纳入：归档缺失导致 pnl=0 的 FILLED 真单是最需核对的类
                    TradeOrderModel.settle_outcome.in_(
                        ("UP", "DOWN", "NOISE", "EXPIRED")),
                )
                .order_by(TradeOrderModel.window_start.asc())
            )
            async with async_session_factory() as session:
                rows = (await session.execute(stmt)).scalars().all()

            checked = 0
            mismatches: list[dict] = []
            # 同一窗口多通道共用一次官方查询（1 rps 下避免重复请求）
            official_cache: dict[tuple[str, int], str | None] = {}
            for row in rows:
                if row.direction is None or row.window_start is None:
                    continue
                key = (row.market_period or "5m", int(row.window_start))
                if key not in official_cache:
                    official_cache[key] = await self._official_outcome(row)
                official = official_cache[key]
                if official not in ("UP", "DOWN"):
                    continue  # 未结算/查询失败：跳过，不计入不一致
                checked += 1
                official_win = row.direction == official
                if row.win is None or bool(official_win) != bool(row.win):
                    mismatches.append({
                        "id": int(row.id),
                        "channel": row.signal_version,
                        "period": row.market_period,
                        "window_start": int(row.window_start),
                        "direction": row.direction,
                        "local_outcome": row.settle_outcome,
                        "local_win": row.win,
                        "local_pnl": row.pnl,
                        "official_outcome": official,
                        "official_win": official_win,
                    })

            if mismatches:
                logger.critical(
                    "结算对账不一致 | 最近 %s 天 | checked=%s | mismatches=%s | ids=%s",
                    days, checked, len(mismatches),
                    ",".join(str(m["id"]) for m in mismatches[:20]),
                )
                try:
                    wechat_notifier.notify_settlement_mismatch(mismatches)
                except Exception as exc:
                    logger.warning("结算对账告警发送失败: %s", exc)
            return {"checked": checked, "mismatches": mismatches}

    async def poll_once(self) -> int:
        """扫描并结算一批未结算订单，返回本轮结算条数（settle-scan 端点复用）。"""
        cutoff = datetime.now(timezone.utc) - SETTLE_DELAY
        stmt = (
            sa_select(TradeOrderModel)
            .where(
                TradeOrderModel.status == "FILLED",
                TradeOrderModel.settled_at.is_(None),
                TradeOrderModel.window_start.isnot(None),
                TradeOrderModel.created_at < cutoff,
                # 手动平仓 SELL 记录行不参与窗口结算（盈亏在 close_position
                # 已实现落库；重算会按 BUY 口径错记，CodeReview R2）
                TradeOrderModel.side == "BUY",
            )
            .order_by(TradeOrderModel.window_start.asc())
            .limit(SCAN_BATCH)
        )
        async with async_session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        settled = 0
        for row in rows:
            try:
                if await self._settle_row(row):
                    settled += 1
            except Exception as exc:
                logger.warning(
                    "交易结算器：单行结算失败（下轮重试）| id=%s | window=%s | %s",
                    row.id, row.window_start, exc)
        return settled

    # ------------------------------------------------------------------
    # 单行结算
    # ------------------------------------------------------------------

    async def _settle_row(self, row: TradeOrderModel) -> bool:
        # 口径分流（防错配，见模块 docstring）：
        #   ① 15m + 有 scene_signal_id → scene 族，回读 FakeBreakoutSignal
        #   ② 15m + 无 scene_signal_id → K 线影子族（s2_cond 条件单 / nextbar hm·ih），
        #      回读 KlineShadowSignal（按 version + target_bar_start 对齐）
        #   ③ 5m → SentimentWindow
        # 为何必须分 ②：旧口径把 ② 塞进 _settle_scene_row，缺 scene_signal_id 即
        # CRITICAL + EXPIRED/pnl=0 —— FILLED 真单的真实盈亏被抹平（2026-09-10 生产
        # 实锤 id=466 真实 -1.00 记为 +0.00，且 EXPIRED 语义让「已成交」显示为过期）。
        # 按「有无 scene_signal_id」分流而非硬编码通道名：新增 15m 影子族通道无需改此处。
        # 官方结算优先（2026-10-23）：先查 Chainlink 口径的官方结果，
        # 命中则跳过下方 K 线归档/影子行回退链路。PENDING（市场未出结果）
        # 30 分钟内下轮重试，超时回退 K 线。direction=NULL 旧数据不查。
        official: str | None = None
        if row.direction is not None:
            official = await self._official_outcome(row)
            if official == "PENDING":
                return False

        if getattr(row, "signal_version", None) in CANDLESTICK_SIGNAL_IDS:
            return await self._settle_kline_shadow_row(row, official)
        # 5m nextbar 管道：影子行与订单同表对齐，优先按 K 线影子行结算；
        # 影子行缺失时回退 5m SentimentWindow（见 _has_kline_shadow_row）。
        if getattr(row, "signal_version", None) in _KLINE_SHADOW_5M_VERSIONS:
            if await self._has_kline_shadow_row(row):
                return await self._settle_kline_shadow_row(row, official)
        # 手动 15m 单（manual_test_*）：无影子信号行，回读 KlineShadowSignal 必落空
        # → 24h 后被错记 EXPIRED/pnl=0（CodeReview R2 Critical#1，与 id=466 同型）。
        # 分流到独立路径：用 15m 窗内首尾两个 5m SentimentWindow 拼出周期开盘/收盘价。
        if (row.market_period == "15m" and row.scene_signal_id is None
                and (row.signal_version or "").startswith("manual_test")):
            return await self._settle_manual_15m_row(row, official)
        if row.market_period == "15m" and row.scene_signal_id is None:
            return await self._settle_kline_shadow_row(row, official)
        if row.market_period == "15m" or row.scene_signal_id is not None:
            return await self._settle_scene_row(row, official)

        if row.direction is None:
            # 旧数据（direction 落库前的订单）：无法判赢，且字段无回填机制
            #（direction 与下单同事务写入）——等待毫无意义，被扫出（超 7min
            # 延迟）即 EXPIRED 出清（win=None/pnl=0 不计统计）。防永久卡
            # 「在途持仓」+ 每 60s 空扫（2026-08-23 生产实锤：4 笔 8/22 旧单
            # direction=NULL 卡在途 3.85 USDT，每轮空扫告警刷屏；首版等 24h
            # 反而让旧单多挂一天，收敛为立即出清）。
            now_dt = datetime.now(timezone.utc)
            async with async_session_factory() as session:
                stmt = (
                    sa_update(TradeOrderModel)
                    .where(
                        TradeOrderModel.id == row.id,
                        TradeOrderModel.settled_at.is_(None),
                    )
                    .values(
                        settle_outcome="EXPIRED",
                        win=None,
                        settle_price=None,
                        pnl=0.0,
                        settled_at=now_dt,
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
            if result.rowcount == 0:
                return False
            await self._patch_settlement_assessment(row, "EXPIRED")
            self._settled_count += 1
            logger.info(
                "订单结算 | id=%s | direction=NULL（旧数据）→ EXPIRED 出清 | window=%s",
                row.id, row.window_start)
            return True

        now_dt = datetime.now(timezone.utc)
        window = await self._find_window(row.window_start)
        if official in ("UP", "DOWN"):
            # 官方结果无对应价格（Chainlink mid 不公开结算值）：settle_price 留空
            outcome = official
            settle_price = None
        else:
            outcome = window.outcome if window is not None else None
            settle_price = self._to_float(window.exit_price) if window is not None else None

        if outcome in ("UP", "DOWN"):
            win = row.direction == outcome
            amount = self._amount_usdt(row)
            avg_price = self._avg_price(row)
            shares = self._shares(row)
            if win and amount is not None and shares is not None:
                pnl = shares - amount  # 币安实际到手股数（已扣费）：对齐实现盈亏
            elif win and amount is not None and avg_price is not None:
                pnl = amount / avg_price - amount
            elif not win and amount is not None:
                pnl = -amount
            else:
                pnl = None  # 均价/金额缺失：输向无需均价，赢向无法估算
        elif outcome == "NOISE":
            win = None
            pnl = 0.0
        else:
            # 窗口未归档 / outcome NULL（归档缺结算价）：超 24h 兜底 EXPIRED，否则重试
            created = self._aware(row.created_at)
            if created is not None and created > now_dt - EXPIRE_AFTER:
                return False  # 未超 24h：归档可能迟到，下轮重试
            win = None
            settle_price = None
            pnl = 0.0
            outcome = "EXPIRED"

        # 幂等守卫：WHERE settled_at IS NULL（并发/手动补算竞争时只生效一次）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False  # 已被并发结算（幂等守卫生效）
        await self._patch_settlement_assessment(row, str(outcome))
        self._settled_count += 1
        self._fire_on_settled(row)
        logger.info(
            "订单结算 | id=%s | window=%s | direction=%s → %s | win=%s | pnl=%s | settle_price=%s | 来源=%s",
            row.id, row.window_start, row.direction, outcome, win,
            f"{pnl:+.4f}" if pnl is not None else "N/A", settle_price,
            "official" if official in ("UP", "DOWN") else "kline",
        )
        try:
            wechat_notifier.notify_order_settled(
                channel=row.signal_version or "unknown",
                window_start=int(row.window_start),
                direction=str(row.direction),
                outcome=str(outcome),
                win=win,
                pnl=pnl,
                amount_usdt=self._amount_usdt(row),
                settle_price=settle_price,
            )
        except Exception as exc:
            logger.warning("企微结算通知异常: %s", exc)
        return True

    async def _settle_scene_row(self, row: TradeOrderModel, official: str | None = None) -> bool:
        """15m 场景订单结算：官方优先，回退 FakeBreakoutSignal.settle_outcome。

        结算判定与 pnl 公式与 5m 路径同口径（win = direction == outcome；
        赢优先 股数−成本 回退 amount/avg_price−amount / 输 −amount）；官方命中时
        settle_price 留空；回退时取信号行 settle_btc_price（15m 周期末 BTC 中间价
        P(E)）。信号未结算（PENDING）时 15m 检测器结算可能迟到：settle_deadline+24h
        内下轮重试，超期 EXPIRED 出清（与 5m 兑底一致）。
        """
        now_dt = datetime.now(timezone.utc)
        official_hit = official in ("UP", "DOWN")
        sig = None
        if not official_hit:
            if row.scene_signal_id is None:
                # 理论不可达（scene_signal_id 与占位同事务落库）：出清防每 60s
                # 空扫；CRITICAL 留痕供日志健康检查排查数据链异常
                logger.critical(
                    "订单结算 | id=%s | 15m 订单缺 scene_signal_id（数据异常）"
                    "→ EXPIRED 出清 | window=%s", row.id, row.window_start)
                return await self._expire_row(row, now_dt)

            async with async_session_factory() as session:
                sig = await session.get(FakeBreakoutSignal, int(row.scene_signal_id))
            if sig is None:
                # 信号行被运维 TRUNCATE/删除：无法判赢，出清不计统计
                logger.critical(
                    "订单结算 | id=%s | 场景信号 %s 缺失（被清理？）→ EXPIRED 出清",
                    row.id, row.scene_signal_id)
                return await self._expire_row(row, now_dt)

        if official_hit:
            outcome = official
            settle_price = None
        else:
            outcome = sig.settle_outcome
            settle_price = self._to_float(sig.settle_btc_price)

        if outcome in ("UP", "DOWN"):
            win = row.direction == outcome
            amount = self._amount_usdt(row)
            avg_price = self._avg_price(row)
            shares = self._shares(row)
            if win and amount is not None and shares is not None:
                pnl = shares - amount  # 币安实际到手股数（已扣费）：对齐实现盈亏
            elif win and amount is not None and avg_price is not None:
                pnl = amount / avg_price - amount
            elif not win and amount is not None:
                pnl = -amount
            else:
                pnl = None  # 均价/金额缺失：输向无需均价，赢向无法估算
        elif outcome == "NOISE":
            # 与 5m 路径同口径：NOISE 即时结算（win=None/pnl=0），避免空扫重扫
            # 直到 settle_deadline+24h 才被误记为 EXPIRED
            win = None
            pnl = 0.0
        elif sig.settle_deadline is not None and datetime.fromtimestamp(
                sig.settle_deadline / 1000.0, tz=timezone.utc
        ) > now_dt - EXPIRE_AFTER:
            return False  # 检测器结算迟到（PENDING）：deadline+24h 内重试
        else:
            win = None
            settle_price = None
            pnl = 0.0
            outcome = "EXPIRED"

        # 幂等守卫：WHERE settled_at IS NULL（与 _settle_row 同模式）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False  # 已被并发结算（幂等守卫生效）
        await self._patch_settlement_assessment(row, str(outcome))
        self._settled_count += 1
        self._fire_on_settled(row)
        logger.info(
            "订单结算 | id=%s | scene_signal=%s | window=%s | direction=%s → %s"
            " | win=%s | pnl=%s | settle_price=%s | 来源=%s",
            row.id, row.scene_signal_id, row.window_start, row.direction, outcome,
            win, f"{pnl:+.4f}" if pnl is not None else "N/A", settle_price,
            "official" if official_hit else "scene",
        )
        try:
            wechat_notifier.notify_order_settled(
                channel=row.signal_version or "scene",
                window_start=int(row.window_start),
                direction=str(row.direction),
                outcome=str(outcome),
                win=win,
                pnl=pnl,
                amount_usdt=self._amount_usdt(row),
                settle_price=settle_price,
            )
        except Exception as exc:
            logger.warning("企微场景结算通知异常: %s", exc)
        return True

    async def _has_kline_shadow_row(self, row: TradeOrderModel) -> bool:
        """5m nextbar 通道：影子行是否存在（决定走 K 线影子还是 SentimentWindow）。"""
        async with async_session_factory() as session:
            found = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == row.signal_version,
                    KlineShadowSignal.target_bar_start == row.window_start,
                ).limit(1)
            )).scalar_one_or_none()
        return found is not None

    async def _settle_kline_shadow_row(self, row: TradeOrderModel,
                                       official: str | None = None) -> bool:
        """K 线影子族订单结算：官方优先，回退 KlineShadowSignal。

        适用范围：15m 且无 scene_signal_id 的通道——s2_cond 条件单族（t4/t5d）与
        nextbar 15m 族（hm_inside_15m_v2/ih_inside_15m_v2）；以及 2026-10-23 注册的
        5m nextbar 管道（hm_quality_5m_v1/hm_moderate_5m_v1，显式路由）。这些通道的信号行落
        kline_shadow_signals（检测器在次根收盘时用 15m K 线 open/close 判向），
        下单路径无 signal_id 关联列，故按 (version=signal_version,
        target_bar_start=window_start) 回查。

        判向与 pnl 口径与其余两路径完全一致：
            outcome ∈ {UP,DOWN}：win = (direction == outcome)；
                赢：优先 shares − amount，回退 amount/avg_price − amount；输：−amount
            outcome == NOISE：win=None、pnl=0（检测器把 NOISE 行 status 记为
                EXPIRED 但 settle_outcome 仍为 NOISE，故本路径只认 settle_outcome
                不认 status，避免把平盘误判为「无结算源」）
            影子行缺失 / settle_outcome 为空：created_at 超 EXPIRE_AFTER 才出清
                EXPIRED，否则下轮重试（检测器结算可能迟到）
        settle_price 取次根收盘价 settle_close（15m 周期末 BTC 价，与币安 15m
        市场结算价同源）；官方命中时 settle_price 留空。
        """
        now_dt = datetime.now(timezone.utc)
        official_hit = official in ("UP", "DOWN")
        sig = None
        if not official_hit:
            logical_candle = row.signal_version in CANDLESTICK_SIGNAL_IDS
            lookup_version = (
                EVENT_VERSION_BY_TF[row.market_period] if logical_candle else row.signal_version
            )
            async with async_session_factory() as session:
                stmt = sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == lookup_version,
                    KlineShadowSignal.target_bar_start == row.window_start,
                )
                sig = (await session.execute(stmt)).scalar_one_or_none()
            if logical_candle and sig is not None:
                snapshot = sig.feature_snapshot or {}
                labels = snapshot.get("all_matched_signal_ids", snapshot.get("matched_signal_ids", []))
                if row.signal_version not in labels:
                    sig = None

            if sig is None:
                # 影子行缺失：影子 gate 被手动下线 / 检测器重启未回补 / 通道名与
                # version 不一致。无法判赢 → 超 24h 出清，未超期下轮重试（gate 恢复
                # 或检测器补录后仍可结算）。
                created = self._aware(row.created_at)
                if created is not None and created > now_dt - EXPIRE_AFTER:
                    logger.warning(
                        "订单结算 | id=%s | K 线影子行缺失（version=%s target_bar=%s），下轮重试",
                        row.id, row.signal_version, row.window_start)
                    return False
                logger.critical(
                    "订单结算 | id=%s | K 线影子行缺失且超 %s → EXPIRED 出清 | signal=%s | window=%s",
                    row.id, EXPIRE_AFTER, row.signal_version, row.window_start)
                return await self._expire_row(row, now_dt)

            outcome = (sig.settle_outcome or "").upper() or None
            settle_price = self._to_float(sig.settle_close)
            if outcome not in ("UP", "DOWN", "NOISE"):
                # 影子行仍 PENDING（次根未收盘 / 检测器结算迟到）：24h 内重试
                created = self._aware(row.created_at)
                if created is not None and created > now_dt - EXPIRE_AFTER:
                    return False
                logger.warning(
                    "订单结算 | id=%s | 影子行无 outcome（status=%s）且超 %s → EXPIRED 出清",
                    row.id, sig.status, EXPIRE_AFTER)
                return await self._expire_row(row, now_dt)
        else:
            outcome = official
            settle_price = None

        if outcome in ("UP", "DOWN"):
            win = row.direction == outcome
        elif outcome == "NOISE":
            win = None

        amount = self._amount_usdt(row)
        avg_price = self._avg_price(row)
        shares = self._shares(row)
        if outcome == "NOISE":
            pnl = 0.0
        elif win and amount is not None and shares is not None:
            pnl = shares - amount  # 币安实际到手股数（已扣费）：对齐实现盈亏
        elif win and amount is not None and avg_price is not None:
            pnl = amount / avg_price - amount
        elif not win and amount is not None:
            pnl = -amount
        else:
            pnl = None  # 均价/金额缺失：输向无需均价，赢向无法估算

        # 幂等守卫：WHERE settled_at IS NULL（与其余两路径同模式）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False  # 已被并发结算（幂等守卫生效）
        await self._patch_settlement_assessment(row, str(outcome))
        self._settled_count += 1
        self._fire_on_settled(row)
        logger.info(
            "订单结算 | id=%s | kline_shadow | signal=%s | window=%s | direction=%s → %s"
            " | win=%s | pnl=%s | settle_price=%s | 来源=%s",
            row.id, row.signal_version, row.window_start, row.direction, outcome,
            win, f"{pnl:+.4f}" if pnl is not None else "N/A", settle_price,
            "official" if official_hit else "kline",
        )
        try:
            # notify_order_settled 是同步方法（与 5m/scene 两路径同口径调用）
            wechat_notifier.notify_order_settled(
                channel=row.signal_version or "kline_shadow",
                window_start=int(row.window_start),
                direction=str(row.direction),
                outcome=str(outcome),
                win=win,
                pnl=pnl,
                amount_usdt=amount,
                settle_price=settle_price,
            )
        except Exception as exc:
            logger.warning("企微 K 线影子结算通知异常: %s", exc)
        return True

    async def _expire_row(self, row: TradeOrderModel, now_dt: datetime) -> bool:
        """异常行出清（EXPIRED/win=None/pnl=0）：防永久卡「在途持仓」+ 空扫。"""
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome="EXPIRED",
                    win=None,
                    settle_price=None,
                    pnl=0.0,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False
        await self._patch_settlement_assessment(row, "EXPIRED")
        self._settled_count += 1
        return True

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 官方结算源（winning_outcome）
    # ------------------------------------------------------------------

    def _resolve_official_client(self):
        """首次使用时取进程单例；测试可显式传 None / 替身。"""
        if self._official_auto and self._official is None:
            self._official = get_official_settlement_client()
        return self._official

    async def _official_outcome(self, row: TradeOrderModel) -> str | None:
        """官方结果："UP"/"DOWN"；"PENDING"（30min 内下轮重试）；
        None（未配置/查询失败/超时 → 调用方回退 K 线）。"""
        client = self._resolve_official_client()
        if client is None or row.window_start is None:
            return None
        try:
            outcome = await client.fetch_outcome(
                getattr(row, "market_period", None) or "5m",
                int(row.window_start),
            )
        except Exception as exc:  # noqa: BLE001 - 任何异常均回退 K 线
            logger.warning(
                "官方结算查询异常（回退 K 线）| id=%s | %s: %s",
                row.id, type(exc).__name__, exc,
            )
            return None
        if outcome == "PENDING":
            created = self._aware(row.created_at)
            if created is not None and created > datetime.now(timezone.utc) - PENDING_RETRY:
                return "PENDING"
            return None
        return outcome if outcome in ("UP", "DOWN") else None

    @staticmethod
    async def _patch_settlement_assessment(row: TradeOrderModel, outcome: str) -> None:
        """Advance observation only after the authoritative order settlement commits."""
        assessment_id = getattr(row, "assessment_id", None)
        if assessment_id is None:
            return
        reason = (
            ReasonCode.SETTLED
            if outcome in ("UP", "DOWN")
            else ReasonCode.SETTLEMENT_EXPIRED
        )
        try:
            async with async_session_factory() as session:
                await patch_assessment(
                    session,
                    AssessmentPatch(
                        assessment_id=int(assessment_id),
                        stage=TerminalStage.SETTLEMENT,
                        reason=reason,
                    ),
                )
                await session.commit()
        except Exception as exc:
            logger.error(
                "交易结算器：执行评估推进失败（不影响订单）| assessment=%s | reason=%s | %s",
                assessment_id, reason.value, exc,
            )

    async def _find_window(self, window_start: int) -> SentimentWindow | None:
        stmt = sa_select(SentimentWindow).where(
            SentimentWindow.start_time == window_start)
        async with async_session_factory() as session:
            return (await session.execute(stmt)).scalar_one_or_none()

    async def _settle_manual_15m_row(self, row: TradeOrderModel,
                                     official: str | None = None) -> bool:
        """手动 15m 单结算：官方优先，回退首尾 5m 窗口拼接口径。

        回退结算源：15m 窗 [t0, t0+900s) 内首尾两个 5m SentimentWindow——
        w1(t0) 的 entry_price 即 15m 开盘价；w3(t0+600s) 的 exit_price 即
        15m 收盘价（5m 窗边界价快照，与 15m K 线 open/close 同口径）。
        pnl 公式与 5m 路径同口径（赢优先 股数−成本，回退 amount/均价−amount，
        输=−amount）；首尾窗归档迟到（未超 24h）则重试，超时 EXPIRED 兕底。
        官方命中时 settle_price 留空。
        """
        from datetime import datetime, timezone

        now_dt = datetime.now(timezone.utc)
        official_hit = official in ("UP", "DOWN")
        exit_ = None
        if official_hit:
            outcome = official
            settle_price = None
        else:
            t0 = int(row.window_start)
            w_first = await self._find_window(t0)
            w_last = await self._find_window(t0 + 600_000)
            entry = self._to_float(w_first.entry_price) if w_first else None
            exit_ = self._to_float(w_last.exit_price) if w_last else None

            if entry is not None and exit_ is not None and entry > 0:
                if exit_ > entry:
                    outcome = "UP"
                elif exit_ < entry:
                    outcome = "DOWN"
                else:
                    outcome = "NOISE"
            else:
                # 首尾窗未齐（归档迟到）：未超 24h 重试，超时 EXPIRED 兕底（同 5m 路径）
                created = self._aware(row.created_at)
                if created is not None and created > now_dt - EXPIRE_AFTER:
                    return False
                outcome = "EXPIRED"
            settle_price = exit_

        if outcome == "NOISE":
            win, pnl = None, 0.0
        elif outcome == "EXPIRED":
            win, settle_price, pnl = None, None, 0.0
        else:
            win = row.direction == outcome
            amount = self._amount_usdt(row)
            avg_price = self._avg_price(row)
            shares = self._shares(row)
            if win and amount is not None and shares is not None:
                pnl = shares - amount
            elif win and amount is not None and avg_price is not None:
                pnl = amount / avg_price - amount
            elif not win and amount is not None:
                pnl = -amount
            else:
                pnl = None

        # 幂等守卫：WHERE settled_at IS NULL（并发竞争只生效一次）
        async with async_session_factory() as session:
            stmt = (
                sa_update(TradeOrderModel)
                .where(
                    TradeOrderModel.id == row.id,
                    TradeOrderModel.settled_at.is_(None),
                )
                .values(
                    settle_outcome=outcome,
                    win=win,
                    settle_price=settle_price,
                    pnl=pnl,
                    settled_at=now_dt,
                )
            )
            result = await session.execute(stmt)
            await session.commit()
        if result.rowcount == 0:
            return False
        await self._patch_settlement_assessment(row, str(outcome))
        self._settled_count += 1
        self._fire_on_settled(row)
        logger.info(
            "订单结算 | id=%s | 手动 15m | window=%s | direction=%s → %s | win=%s | pnl=%s | 来源=%s",
            row.id, row.window_start, row.direction, outcome, win,
            f"{pnl:+.4f}" if pnl is not None else "N/A",
            "official" if official_hit else "window-stitch")
        return True

    @staticmethod
    def _amount_usdt(row: TradeOrderModel) -> float | None:
        """amount_in（wei 字符串）→ USDT；无效/零返回 None。"""
        try:
            v = int(row.amount_in) / 1e18
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _avg_price(row: TradeOrderModel) -> float | None:
        """quote_json.averagePrice → 成交均价；无效/零返回 None。"""
        try:
            v = float((row.quote_json or {}).get("averagePrice") or 0)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _shares(row: TradeOrderModel) -> float | None:
        """quote_json.filledShareQty → 币安实际到手股数（已扣 marketProviderFee，
        成交确认/对账回填）；缺失/无效/零返回 None。2026-08-28：费用以少给股数
        体现，无费估算（amount/均价）比币安实现盈亏高约费用额。"""
        try:
            v = float((row.quote_json or {}).get("filledShareQty") or 0)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(v) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _aware(dt: datetime | None) -> datetime | None:
        """naive datetime（测试桩/驱动差异）按 UTC 补 tzinfo。"""
        if dt is None:
            return None
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
