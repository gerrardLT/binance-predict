"""多通道实盘执行器（MultiLiveTrader，2026-08-24 取代单版本 QuoteEdgeLiveTrader）。

10 通道（通道注册表见 live_channels.py）可同时开启，每通道独立
金额/日限/护栏/开关；通道 ID 与影子信号版本名对齐（订单 signal_version
直接用通道名，实盘 vs 影子对账天然一致）。三族触发机制并存：

1. quote_edge 族（5m 市场）：5m 采样循环喂价 → 窗内 DOWN 报价首次进入
   所绑版本规则区间 → 真单押 DOWN（区间复用 QUOTE_EDGE_RULES 冻结口径）。
   v2 门禁版通过实时 BTC 喂价解锁：chg = (btc − 窗口开盘) / 开盘，
   阈值复用 V2_PRICE_GUARDS（min_drop ≤ −0.10% / max_rise < +0.10%）；
   门禁数据缺失不触发（与影子保守口径一致）。
2. x4 族（5m 市场）：轮询 misalignment_signals PENDING x4 信号 → 睡到
   次窗 +150s 决策点（DECISION_T_SEC 同源）→ 真单押 DOWN；
   错过决策点（容差 30s）不追单（与影子 PENDING→EXPIRED 语义一致）。
3. scene 族（15m 市场）：fake_breakout_detector fire 钩子回调 →
   次周期开盘真单（S1/S4/S5 押 DOWN，S2 押 UP；S5 为 +5min 确认后回调），
   订单落 scene_signal_id 直接关联场景信号行（结算走 FakeBreakoutSignal）。

安全护栏（继承旧执行器全部教训）：
1. 每通道每窗至多一单：内存 fired 集合 + 先占位后下单（place_order 前先插
   PENDING 行占住 (signal_version, window_start) 唯一键，重复窗口在花钱前被拒）。
2. 执行价护栏：报价 averagePrice > 通道护栏 → 弃单且滑点按护栏动态收紧。
3. 日单量护栏：每通道当日 FILLED 达上限停火（DB 口径，重启不清零）。
4. 不回灌：quote_edge 只盯活跃窗口，重启不补已过去的窗口；x4 PENDING
   决策点已过（容差外）不追。
5. 下单任务 create_task 脱离采样循环；stop() 只取消辅助任务（回填/自愈/
   轮询），在途下单任务等待完成（钱已出去的临界区绝不打断）。
6. signal_id 回填：quote_edge 延迟回填 + 周期自愈；x4 下单成交即回填
   （信号已存在）；scene 走 scene_signal_id 无需回填。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy import select as sa_select
from sqlalchemy import update as sa_update

from binance_predict.config.settings import settings
from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import MisalignmentSignal, TradeOrderModel

from .live_channels import (
    LIVE_CHANNELS,
    MAX_DAILY_ORDERS_CAP,
    MAX_ORDER_AMOUNT_USDT,
    parse_channel_config,
    resolve_max_exec,
    scene_pattern_to_channel,
)
from .misalignment_detector import DECISION_T_SEC
from .quote_edge_detector import QUOTE_EDGE_RULES, V2_PRICE_GUARDS

X4_VERSIONS = ("x4_v1", "x4_v2")
SIGNAL_BACKFILL_DELAY_MS = 180_000  # 窗口结束后 180s 回读影子信号（归档+结算已就绪）
HEAL_INTERVAL_S = 300.0             # signal_id 自愈扫描间隔（重启/延迟不丢对账）
X4_POLL_INTERVAL_S = 30.0           # x4 PENDING 信号轮询间隔
X4_DECISION_TOLERANCE_MS = 30_000   # 决策点错过容差（超时不追单）
X4_RECENT_WINDOW_MS = 10 * 60 * 1000  # 只看近期 PENDING（更早的已无意义）


class MultiLiveTrader:
    """多通道实盘执行器：喂价/轮询/钩子三族触发 → 统一护栏 → 真单。"""

    def __init__(self, trader) -> None:
        self._trader = trader  # BinancePredictionTrader（复用签名/报价/下单链路）
        self._specs = LIVE_CHANNELS
        # 解析启动配置（LIVE_CHANNELS_JSON 非法 → ValueError 拒绝装配，fail fast）
        self._configs = parse_channel_config()
        for ch, cfg in self._configs.items():
            # 默认金额/日限也可能被 settings 误写超上限（parse 只校验 JSON 覆盖段）
            if not (0.1 <= cfg.amount_usdt <= MAX_ORDER_AMOUNT_USDT):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 单笔金额 {cfg.amount_usdt} 超硬上限"
                    f" {MAX_ORDER_AMOUNT_USDT} USDT（配置误写拒绝启动）")
            if not (1 <= cfg.max_daily_orders <= MAX_DAILY_ORDERS_CAP):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 日限 {cfg.max_daily_orders} 超硬上限"
                    f" {MAX_DAILY_ORDERS_CAP}（配置误写拒绝启动）")
        self._enabled_at_startup = {ch: cfg.enabled for ch, cfg in self._configs.items()}
        self._tasks: set[asyncio.Task] = set()   # 在途任务（下单/回填/自愈/轮询）
        # 余额缓存作废钩子（main 装配区注入：services 层不反向 import main，用回调解耦）
        self._on_balance_change = None
        self._x4_seen: set[int] = set()          # 已调度过的 x4 信号 id（幂等）
        self._healed_total = 0
        self._stopped = False                    # stop 后拒绝派生新下单任务
        self._running = False

    # ------------------------------------------------------------------
    # quote_edge 族：采样循环喂价 → 区间命中 → 开火
    # ------------------------------------------------------------------

    def check(self, window_start_ms: int, window_end_ms: int,
              ts_ms: int, down_price: float | None,
              btc_price: float | None = None,
              window_entry_price: float | None = None) -> list[str]:
        """每次 5m 采样调用一次；返回本轮开火的通道名列表。

        纯内存快速路径（不阻塞采样循环）；命中通道派生下单任务。
        v2 门禁需要 btc_price（当前采样 BTC 中间价）与 window_entry_price
        （窗口开盘 BTC 快照），缺失 → v2 通道不触发（fail-safe）。
        """
        if self._stopped or down_price is None:
            return []
        t_rel = (ts_ms - window_start_ms) / 1000.0
        fired: list[str] = []
        for ch, spec in self._specs.items():
            if spec.family != "quote_edge":
                continue
            cfg = self._configs[ch]
            if not cfg.enabled:
                continue
            if window_start_ms in cfg.fired:
                continue
            # v2 用 base(v1) 冻结区间（V2_PRICE_GUARDS 映射，勿复制数值）
            rule_key = V2_PRICE_GUARDS[ch][0] if ch in V2_PRICE_GUARDS else ch
            t_lo, t_hi, q_lo, q_hi = QUOTE_EDGE_RULES[rule_key]
            if not (t_lo <= t_rel < t_hi):
                continue
            if not (q_lo <= float(down_price) < q_hi):
                continue
            if spec.v2_guard is not None and not self._pass_live_v2_guard(
                    ch, btc_price, window_entry_price):
                continue
            # 命中：立即占位（内存），防同通道同窗后续采样重复派生
            cfg.fired.add(window_start_ms)
            task = asyncio.create_task(
                self._fire_quote_edge(ch, window_start_ms, window_end_ms,
                                      t_rel, float(down_price)),
                name=f"live_qe_{ch}_{window_start_ms}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            fired.append(ch)
        return fired

    @staticmethod
    def _pass_live_v2_guard(version: str, btc_price: float | None,
                            window_entry_price: float | None) -> bool:
        """v2 门禁实时版：chg% = (当前 BTC − 窗口开盘 BTC) / 开盘 × 100。

        阈值复用 V2_PRICE_GUARDS（min_drop: chg≤阈值 / max_rise: chg<阈值）。
        数据缺失 → False 不触发（与影子「门禁数据缺失不落表」同保守口径）。
        """
        mode, threshold = V2_PRICE_GUARDS[version][1], V2_PRICE_GUARDS[version][2]
        if not btc_price or not window_entry_price or window_entry_price <= 0:
            return False
        chg_pct = (btc_price - window_entry_price) / window_entry_price * 100.0
        return chg_pct <= threshold if mode == "min_drop" else chg_pct < threshold

    # ------------------------------------------------------------------
    # x4 族：PENDING 信号轮询 → 决策点下单
    # ------------------------------------------------------------------

    async def _x4_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(X4_POLL_INTERVAL_S)
                await self._x4_poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("多通道实盘：x4 轮询异常 | {}", exc)

    async def _x4_poll_once(self) -> None:
        """捞 PENDING x4 信号调度决策点任务（seen 集合幂等，重启自然恢复）。"""
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(
                    MisalignmentSignal.id,
                    MisalignmentSignal.version,
                    MisalignmentSignal.target_window_start,
                ).where(
                    MisalignmentSignal.version.in_(X4_VERSIONS),
                    MisalignmentSignal.status == "PENDING",
                    MisalignmentSignal.target_window_start >= now_ms - X4_RECENT_WINDOW_MS,
                ).order_by(MisalignmentSignal.target_window_start.asc()).limit(20)
            )).all()
        for sig_id, version, target_start in rows:
            if sig_id in self._x4_seen:
                continue
            self._x4_seen.add(sig_id)
            decision_ms = int(target_start) + int(DECISION_T_SEC * 1000)
            if now_ms > decision_ms + X4_DECISION_TOLERANCE_MS:
                continue  # 错过决策点：不追单（迟到价破坏回测口径）
            task = asyncio.create_task(
                self._fire_x4(int(sig_id), str(version), int(target_start)),
                name=f"live_x4_{version}_{target_start}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _fire_x4(self, sig_id: int, version: str, target_start: int) -> None:
        """睡到次窗 +150s 决策点 → 护栏/日限/防重 → 真单押 DOWN → 立即回填。

        睡眠分片（5s 粒度查停止标志）：stop() 后最多 5s 内退出，无需 cancel
        （避免打断下单临界区的取消竞态）。
        """
        decision_s = time.monotonic() + (target_start + DECISION_T_SEC * 1000
                                         - time.time() * 1000) / 1000.0
        while not self._stopped and time.monotonic() < decision_s:
            await asyncio.sleep(min(decision_s - time.monotonic(), 5.0))
        now_ms = int(time.time() * 1000)
        decision_ms = target_start + int(DECISION_T_SEC * 1000)
        if self._stopped or now_ms > decision_ms + X4_DECISION_TOLERANCE_MS:
            return  # 停机/睡过头（GC 卡顿）：不追
        cfg = self._configs[version]
        if not cfg.enabled:
            return
        spec = self._specs[version]
        try:
            filled_today = await self._count_filled_today(version)
            if filled_today >= cfg.max_daily_orders:
                logger.warning(
                    "多通道实盘：{} 日单量护栏停火 | 目标窗 {} | 今日已成交 {} ≥ {}",
                    version, _fmt_win(target_start), filled_today, cfg.max_daily_orders)
                return
            if await self._has_attempt(version, target_start):
                return
            logger.info(
                "多通道实盘开火 | {} | x4 决策点押 DOWN | 目标窗 {} | 金额 {} | 护栏 {}",
                version, _fmt_win(target_start), cfg.amount_usdt,
                resolve_max_exec(spec, cfg))
            order = await self._trader.execute_signal_trade(
                prediction="DOWN",
                amount_usdt=cfg.amount_usdt,
                signal_version=version,
                window_start=target_start,
                max_exec_price=resolve_max_exec(spec, cfg),
                market_period="5m",
            )
            await self._after_fill(order, version, cfg)
            if order is not None and order.get("status") == "FILLED":
                # x4 信号已存在：成交即回填 signal_id（无需等结算）
                await self._link_signal_id(version, target_start, sig_id)
        except Exception as exc:
            logger.warning("多通道实盘：x4 下单任务异常 | {} | {} | {}",
                           version, _fmt_win(target_start), exc)

    # ------------------------------------------------------------------
    # scene 族：检测器 fire 钩子 → 次周期开盘（15m 市场）
    # ------------------------------------------------------------------

    def on_scene_signal(self, sig: dict) -> None:
        """fake_breakout 检测器钩子（同步接口，fire-and-forget）。

        payload：{id, pattern_type, side, market_start_15m, market_end_15m}。
        通道未启用直接返回（影子检测继续落表，互不干扰）；任何异常只告警
        不抛（绝不阻塞检测循环——邮件 SMTP 卡死循环 16 分钟的事故教训）。
        """
        try:
            if self._stopped:
                return  # stop 后拒绝派生新下单任务（与 check/x4 同契约）
            channel = scene_pattern_to_channel(str(sig.get("pattern_type") or ""))
            if channel is None:
                return
            cfg = self._configs[channel]
            if not cfg.enabled:
                return
            market_start = int(sig["market_start_15m"])
            if market_start in cfg.fired:
                return
            cfg.fired.add(market_start)
            task = asyncio.create_task(
                self._fire_scene(channel, sig),
                name=f"live_scene_{channel}_{market_start}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.warning("多通道实盘：场景钩子异常（不影响检测循环）| {}", exc)

    async def _fire_scene(self, channel: str, sig: dict) -> None:
        spec = self._specs[channel]
        cfg = self._configs[channel]
        market_start = int(sig["market_start_15m"])
        # side=high（多头耗尽/动量衰竭）押 DOWN；side=low（空头耗尽）押 UP
        prediction = "DOWN" if sig.get("side") == "high" else "UP"
        try:
            filled_today = await self._count_filled_today(channel)
            if filled_today >= cfg.max_daily_orders:
                logger.warning(
                    "多通道实盘：{} 日单量护栏停火 | 周期 {} | 今日已成交 {} ≥ {}",
                    channel, _fmt_win(market_start), filled_today, cfg.max_daily_orders)
                return
            if await self._has_attempt(channel, market_start):
                return
            logger.info(
                "多通道实盘开火 | {} | 15m 周期开盘押 {} | 周期 {} | 金额 {} | 护栏 {}",
                channel, prediction, _fmt_win(market_start), cfg.amount_usdt,
                resolve_max_exec(spec, cfg))
            order = await self._trader.execute_signal_trade(
                prediction=prediction,
                amount_usdt=cfg.amount_usdt,
                signal_version=channel,
                window_start=market_start,
                max_exec_price=resolve_max_exec(spec, cfg),
                market_period="15m",
                scene_signal_id=int(sig["id"]),
            )
            await self._after_fill(order, channel, cfg)
            # scene_signal_id 下单即落库（无需 signal_id 回填）
        except Exception as exc:
            logger.warning("多通道实盘：场景下单任务异常 | {} | {} | {}",
                           channel, _fmt_win(market_start), exc)

    # ------------------------------------------------------------------
    # 下单主流程（quote_edge 族）
    # ------------------------------------------------------------------

    async def _fire_quote_edge(self, channel: str, window_start: int,
                               window_end: int, t_rel: float,
                               down_price: float) -> None:
        spec = self._specs[channel]
        cfg = self._configs[channel]
        win_label = _fmt_win(window_start)
        try:
            # 日单量护栏（DB 口径，重启不清零）
            filled_today = await self._count_filled_today(channel)
            if filled_today >= cfg.max_daily_orders:
                logger.warning(
                    "多通道实盘：{} 日单量护栏停火 | 窗口 {} | 今日已成交 {} ≥ {}",
                    channel, win_label, filled_today, cfg.max_daily_orders)
                return
            # 重启防重：DB 已有本通道本窗尝试记录（FILLED/FAILED 皆算）则跳过
            if await self._has_attempt(channel, window_start):
                return
            logger.info(
                "多通道实盘开火 | {} | LIVE 下单押 DOWN | 窗口 {} | t=+{:.0f}s"
                " q={:.3f} | 金额 {} USDT | 执行价上限 {}",
                channel, win_label, t_rel, down_price, cfg.amount_usdt,
                resolve_max_exec(spec, cfg))
            order = await self._trader.execute_signal_trade(
                prediction="DOWN",
                amount_usdt=cfg.amount_usdt,
                signal_version=channel,
                window_start=window_start,
                max_exec_price=resolve_max_exec(spec, cfg),
                market_period="5m",
            )
            if order is None:
                # 同窗已有占位（重启/并发重复）或前置配置缺失，未花钱，正常路径
                return
            await self._after_fill(order, channel, cfg)
            if order.get("status") == "FILLED":
                self._schedule(
                    self._backfill_signal_link(window_start, channel), window_end)
        except Exception as exc:
            logger.warning("多通道实盘：下单任务异常 | {} | 窗口 {} | {}",
                           channel, win_label, exc)

    async def _after_fill(self, order: dict | None, channel: str,
                          cfg) -> None:
        """成交后公共动作：计数 + 余额缓存作废。"""
        if order is None:
            return
        cfg.fire_total += 1
        # 成交后作废余额缓存（下单扣预测钱包余额，前端最多 35s 才能看到旧值）
        if self._on_balance_change is not None:
            try:
                self._on_balance_change()
            except Exception:
                logger.warning("多通道实盘：余额缓存作废回调异常（不影响下单）")

    # ------------------------------------------------------------------
    # signal_id 回填（quote_edge 延迟回填 + 自愈；x4 即时回填）
    # ------------------------------------------------------------------

    def _schedule(self, coro, window_end_ms: int) -> None:
        """延迟执行回填（窗口结束后等待归档+影子结算就绪）。"""
        delay_s = max(0.0, (window_end_ms + SIGNAL_BACKFILL_DELAY_MS
                            - time.time() * 1000) / 1000.0)

        async def _delayed():
            await asyncio.sleep(delay_s)
            await coro

        task = asyncio.create_task(_delayed(), name="multi_live_backfill")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _backfill_signal_link(self, window_start: int,
                                    version: str | None = None) -> None:
        """窗口结算后把订单关联回影子信号（实盘 vs 影子一一对账）。

        统一按 target_window_start 匹配：quote_edge 信号 target==本窗、
        x4 信号 target==次窗，均等于订单 window_start（版本名已隔离族）。
        """
        try:
            async with async_session_factory() as session:
                sig = (await session.execute(
                    sa_select(MisalignmentSignal.id).where(
                        MisalignmentSignal.version == version,
                        MisalignmentSignal.target_window_start == window_start,
                    )
                )).first()
                if sig is None:
                    logger.debug("多通道实盘：影子信号未就绪，跳过回填 | 窗口 {}",
                                 window_start)
                    return
                await session.execute(
                    sa_update(TradeOrderModel).where(
                        TradeOrderModel.signal_version == version,
                        TradeOrderModel.window_start == window_start,
                        TradeOrderModel.signal_id.is_(None),
                    ).values(signal_id=sig[0])
                )
                await session.commit()
                logger.info("多通道实盘：订单已关联影子信号 | {} | 窗口 {} | signal_id={}",
                            version, window_start, sig[0])
        except Exception as exc:
            logger.warning("多通道实盘：signal_id 回填失败 | {} | 窗口 {} | {}",
                           version, window_start, exc)

    async def _link_signal_id(self, version: str, window_start: int,
                              sig_id: int) -> None:
        """x4 成交即时回填（信号已存在，无需等归档）。"""
        try:
            async with async_session_factory() as session:
                await session.execute(
                    sa_update(TradeOrderModel).where(
                        TradeOrderModel.signal_version == version,
                        TradeOrderModel.window_start == window_start,
                        TradeOrderModel.signal_id.is_(None),
                    ).values(signal_id=sig_id)
                )
                await session.commit()
        except Exception as exc:
            logger.warning("多通道实盘：x4 signal_id 即时回填失败 | {} | {}", version, exc)

    # ------------------------------------------------------------------
    # DB 查询（每通道独立口径）
    # ------------------------------------------------------------------

    async def _count_filled_today(self, version: str) -> int:
        # date_trunc 取 PG 会话时区（容器内 UTC）即 UTC 自然日；护栏语义自洽。
        # 多通道时代：每通道独立计数（旧跨版本合计口径随单版本互斥一起退役）。
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(sa_func.count(TradeOrderModel.id)).where(
                    TradeOrderModel.signal_version == version,
                    TradeOrderModel.status == "FILLED",
                    TradeOrderModel.created_at >= sa_func.date_trunc("day", sa_func.now()),
                )
            )).scalar()
        return int(row or 0)

    async def _has_attempt(self, version: str, window_start: int) -> bool:
        # 每通道预检：本通道本窗已开过/尝试过（FILLED/FAILED 皆算）则不再开。
        # DB 唯一键按 (signal_version, window_start) 隔离；同窗跨通道并行下单
        # 是多通道时代的预期行为（每通道每窗至多一单）。
        async with async_session_factory() as session:
            row = (await session.execute(
                sa_select(TradeOrderModel.id).where(
                    TradeOrderModel.signal_version == version,
                    TradeOrderModel.window_start == window_start,
                ).limit(1)
            )).first()
        return row is not None

    # ------------------------------------------------------------------
    # signal_id 自愈扫描（重启/影子延迟不丢对账）
    # ------------------------------------------------------------------

    async def _heal_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(HEAL_INTERVAL_S)
                await self._heal_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("多通道实盘：自愈扫描异常 | {}", exc)

    async def _heal_once(self) -> None:
        """为超过 10 分钟仍缺 signal_id 的 5m 订单重试回填（幂等）。

        只扫 5m 订单：scene 订单走 scene_signal_id（下单即落库），
        signal_id 语义上恒为 NULL，纳入扫描会永久空转。
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
            async with async_session_factory() as session:
                orders = (await session.execute(
                    sa_select(TradeOrderModel.window_start,
                              TradeOrderModel.signal_version).where(
                        TradeOrderModel.signal_version.in_(list(self._specs)),
                        TradeOrderModel.signal_id.is_(None),
                        TradeOrderModel.market_period == "5m",
                        TradeOrderModel.created_at < cutoff,
                    ).order_by(TradeOrderModel.window_start.asc()).limit(20)
                )).all()
            for ws, ver in orders:
                await self._backfill_signal_link(int(ws), ver)
            if orders:
                self._healed_total += len(orders)
        except Exception as exc:
            logger.warning("多通道实盘：自愈扫描查询失败 | {}", exc)

    # ------------------------------------------------------------------
    # 运行时控制（toggle 端点）与状态
    # ------------------------------------------------------------------

    def set_channel(self, channel: str, enabled: bool | None = None,
                    amount_usdt: float | None = None,
                    max_daily_orders: int | None = None) -> None:
        """运行时热调单通道（toggle 端点调用；重启回落 live_channels_json）。

        白名单/数值校验同启动配置（不靠自律靠拒改）；在途任务不受影响。
        原子性：先校验全部参数再统一生效——任一非法则整体拒改，
        不会出现「enabled 已置位但金额校验失败」的半生效状态。
        """
        if channel not in self._specs:
            raise ValueError(
                f"多通道实盘：未知通道 {channel!r}（白名单 {list(self._specs)}）")
        cfg = self._configs[channel]
        if amount_usdt is not None and not (
                0.1 <= amount_usdt <= MAX_ORDER_AMOUNT_USDT):
            raise ValueError(
                f"多通道实盘：通道 {channel} 单笔金额 {amount_usdt} 超界"
                f" [0.1, {MAX_ORDER_AMOUNT_USDT}]")
        if max_daily_orders is not None and not (
                1 <= max_daily_orders <= MAX_DAILY_ORDERS_CAP):
            raise ValueError(
                f"多通道实盘：通道 {channel} 日限 {max_daily_orders} 超界"
                f" [1, {MAX_DAILY_ORDERS_CAP}]")
        if enabled is not None:
            cfg.enabled = bool(enabled)
        if amount_usdt is not None:
            cfg.amount_usdt = float(amount_usdt)
        if max_daily_orders is not None:
            cfg.max_daily_orders = int(max_daily_orders)

    def status(self) -> dict:
        """同步状态（不含 DB 查询；今日成交数见 status_async）。"""
        return {
            "enabled_any": any(cfg.enabled for cfg in self._configs.values()),
            "amount_cap": MAX_ORDER_AMOUNT_USDT,
            "defaults": {
                "amount_usdt": settings.live_default_amount_usdt,
                "max_daily_orders": settings.live_default_max_daily_orders,
            },
            "healed_total": self._healed_total,
            "pending_tasks": len(self._tasks),
            "channels": [self._channel_status(ch) for ch in self._specs],
        }

    def _channel_status(self, ch: str) -> dict:
        spec = self._specs[ch]
        cfg = self._configs[ch]
        return {
            "channel": ch,
            "display_name": spec.display_name,
            "family": spec.family,
            "market_period": spec.market_period,
            "direction": spec.direction,
            "enabled": cfg.enabled,
            "enabled_at_startup": self._enabled_at_startup[ch],
            "amount_usdt": cfg.amount_usdt,
            "max_daily_orders": cfg.max_daily_orders,
            "max_exec_price": resolve_max_exec(spec, cfg),
            "auto_max_exec": spec.auto_max_exec,
            "fire_total": cfg.fire_total,
            "fired_windows": sorted(cfg.fired)[-10:],
        }

    async def status_async(self) -> dict:
        """聚合状态 + 每通道今日 FILLED 数（单次 GROUP BY，60s 轮询无压力）。"""
        out = self.status()
        try:
            async with async_session_factory() as session:
                rows = (await session.execute(
                    sa_select(TradeOrderModel.signal_version,
                              sa_func.count(TradeOrderModel.id)).where(
                        TradeOrderModel.signal_version.in_(list(self._specs)),
                        TradeOrderModel.status == "FILLED",
                        TradeOrderModel.created_at >= sa_func.date_trunc(
                            "day", sa_func.now()),
                    ).group_by(TradeOrderModel.signal_version)
                )).all()
            filled = {v: int(c) for v, c in rows}
            for ch_st in out["channels"]:
                ch_st["filled_today"] = filled.get(ch_st["channel"], 0)
        except Exception as exc:
            logger.warning("多通道实盘：今日成交聚合查询失败 | {}", exc)
        return out

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动辅助循环（heal + x4 轮询；quote_edge 触发由采样循环喂价）。"""
        if self._running:
            return
        self._running = True
        enabled = [ch for ch, cfg in self._configs.items() if cfg.enabled]
        logger.info(
            "多通道实盘执行器启动 | 通道 {}/{} | 启用 {}",
            len(enabled), len(self._specs), enabled or "（无，toggle 可开）")
        for coro_factory, name in (
            (self._heal_loop, "multi_live_heal_loop"),
            (self._x4_loop, "multi_live_x4_loop"),
        ):
            task = asyncio.create_task(coro_factory(), name=name)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        """停止：先拒新单，只取消辅助任务（回填/自愈/轮询）；在途下单任务
        不 cancel、等待完成——钱已出去的临界区被打断会造成「交易所成交、
        DB 无终态」。x4 决策点任务自带 5s 分片轮询停止标志，无需 cancel。"""
        self._stopped = True
        self._running = False
        for task in list(self._tasks):
            if task.get_name() in ("multi_live_heal_loop", "multi_live_backfill",
                                   "multi_live_x4_loop"):
                task.cancel()
        for task in list(self._tasks):
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=45)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._tasks.clear()
        totals = {ch: cfg.fire_total for ch, cfg in self._configs.items()
                  if cfg.fire_total}
        logger.info("多通道实盘执行器已停止 | 各通道开火 {}", totals)


def _fmt_win(window_start_ms: int) -> str:
    return datetime.fromtimestamp(
        window_start_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
