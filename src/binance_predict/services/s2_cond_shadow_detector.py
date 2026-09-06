"""S2 条件单影子检测器（s2_cond_t4_v1 / s2_cond_t5d_v1）：实盘 S2 派生的窗内条件确认入场 → 押 UP。

研究结论落地（预注册冻结口径，审计锚点 scripts/s2_cond_freeze_counts_720d.py）：
S2（bear_exhaust：破 4h 支撑 + 收阴 + 放量）开盘即买 UP 的 EV≈−0.042 不赚钱；改为等
次周期窗内 t=4/t=5 判价的条件单入场更优——价跌时 UP token 变便宜，低买 UP 的正 EV 来自
入场价而非胜率。两版均押 UP、均由**实盘 S2 信号派生**（复用 fake_breakout 检测口径，
单一事实源，杜绝第二套 S2 实现漂移）：

    s2_cond_t4_v1  : 次周期 t=4(+240s) 1m 收盘 px4 < 周期开盘 open（全深度回落）→ 押 UP
                     720d 触发 1069/2176(49.1%,1.48/天)，价-only 胜率 38.9%
    s2_cond_t5d_v1 : 次周期 t=5(+300s) 0 < ln(open/px5) < 15bp（中度回落剔深）→ 押 UP
                     720d 触发 643/2176(29.5%,0.89/天)，价-only 胜率 44.8%

判赢 = 次周期 15m 收阳（close>open）。研究 EV（+0.237/+0.283）属报价表口径**乐观上界**，
真实 EV 由生产报价前向现算（面板聚合按 entry_up_price），故 SHADOW_BENCH.bench_ev 不钉死。

落 kline_shadow_signals（与 KREV/反转/nextbar/combo「一家族一检测器共表、version 隔离」
的既有约定一致）：该表已方向感知（direction/entry_up_price/entry_quote_ts），结算
win=(up ∧ dir==UP) ∨ (¬up ∧ dir==DOWN)，UP→收阳赢，天然适配押 UP；免 alembic 迁移、
零风险触碰在跑的 HM/S5-deep（pattern_shadow_signals）结算。本检测器只认自己的 2 个
version 结算/过期（S2_COND_VERSIONS 严格隔离，不碰 nb_/combo_/krev_/hm 行）。

入场报价直接快照 _pm_15m（窗口对齐 start_date==next_start + 龄守卫 ≤20s + up∈(0,1)），
不用 snapshot_entry_quote（后者「近开盘 ≤120s」守卫会拒 t=4/t=5 的 240~300s 报价）；
镜像 hm_shadow_detector._finalize_entry 同款守卫。S5 已在 +5min(300s) 验证 _pm_15m 该
时点窗口对齐可靠。

影子纪律（2026-09-06 promote 后更新）：影子落表与实盘下单解耦——影子仍只记录不下注
（不占风控配额），实盘通道 s2_cond_t4_v1/s2_cond_t5d_v1 已注册 LIVE_CHANNELS，由
_on_live_fire 钩子驱动 MultiLiveTrader 真单（通道 enabled 由 toggle 管，与影子 gate
互不影响：影子下线只停采集，实盘照常；实盘关停影子照常落表）。重启**不回补入场**
（同 S5-deep）；已落 PENDING 行由轮询 _settle_pending 结算，重启安全。
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from binance_predict.services import clock_sync
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（预注册，勿改；审计锚点 scripts/s2_cond_freeze_counts_720d.py）----
VAR_T4 = "s2_cond_t4_v1"
VAR_T5D = "s2_cond_t5d_v1"
S2_COND_VERSIONS = (VAR_T4, VAR_T5D)

DIP_MAX = 0.0015                   # 15bp：t=5 剔深上界（0<回落<15bp）
T4_DELAY_MS = 240_000              # t=4：次周期第 4 分钟末（+240s）；px4=1m bar(+180s).close
T5_DELAY_MS = 300_000              # t=5：次周期第 5 分钟末（+300s）；px5=1m bar(+240s).close
PX4_OFFSET_MS = 180_000            # px4 所在 1m bar 相对 next_start 的偏移（该 bar close=t=4 价）
PX5_OFFSET_MS = 240_000            # px5 所在 1m bar 相对偏移（该 bar close=t=5 价）
GRACE_MS = 8_000                   # 收盘后缓冲（等 1m klines 落库）
MAX_WAIT_MS = 90_000               # 1m K 拉不到的放弃上限
QUOTE_MAX_AGE_MS = 20_000          # 入场报价快照可接受最大龄（tracker ~15s 采样）
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED（兜底）
POLL_INTERVAL = 60.0               # 结算轮询间隔（秒）
RETRY_INTERVAL_S = 5               # 1m K 重试间隔（秒）
BAR_MS_15M = 900_000
SETTLE_BARS = 40                   # 结算/回补拉取的 15m 根数（10h 窗，覆盖停机期 PENDING）
PARENT_OFFSET_MS = 900_000         # signal_bar_start = next_start − 900_000（父 S2 根，保唯一约束）
KLINE_FETCH_1M = 7                 # 入场判价拉取的 1m 根数（覆盖 next_start..+240s + 余量）

DISCOVERY_ID_T4 = "s2c_t4"         # 占位标识（≤16 字符，镜像 nextbar "nb_15m_zs"），审计可辨
DISCOVERY_ID_T5D = "s2c_t5d"

RULE_TEXT_T4 = (
    "S2 条件单 t=4：实盘 S2(bear_exhaust，破 4h 支撑+收阴+放量) 派生；次周期 t=4(+240s) "
    "1m 收盘 px4 < 周期开盘 open（全深度回落，不设上界）→ 押次周期 15m UP（收阳赢）；入场"
    "快照 +240s 真实 15m UP 报价。720d 触发 1069/2176(49.1%,1.48/天)，价-only 胜率 38.9%"
    "（低买 UP 的正 EV 来自入场价，真实 EV 前向现算）。"
)
RULE_TEXT_T5D = (
    "S2 条件单 t=5 剔深：实盘 S2(bear_exhaust) 派生；次周期 t=5(+300s) 0 < ln(open/px5) < "
    "15bp（中度回落剔深，px5=1m bar(next_start+240s).close）→ 押次周期 15m UP（收阳赢）；"
    "入场快照 +300s 真实 15m UP 报价。720d 触发 643/2176(29.5%,0.89/天)，价-only 胜率 44.8%。"
)


# ---------------------------------------------------------------------------
# 纯函数（冻结判定，供实时/测试共用；与审计锚点逐字同源）
# ---------------------------------------------------------------------------

def dip(open_px: float, px: float) -> float:
    """回落深度 = ln(open/px)；>0 表示 px<open（价低于开盘）。"""
    return math.log(open_px / px)


def judge_t4(open_px: float, px4: float) -> bool:
    """变体 A：t=4 价<开盘（全深度）→ 命中押 UP。px4<open ⟺ dip(open,px4)>0。"""
    return px4 < open_px


def judge_t5d(open_px: float, px5: float) -> bool:
    """变体 B：t=5 剔深（0<回落<15bp）→ 命中押 UP。"""
    d = dip(open_px, px5)
    return 0 < d < DIP_MAX


class S2CondShadowDetector:
    """S2 条件单影子检测器：实盘 S2 钩子派生窗内 t=4/t=5 条件确认入场 → 押 UP，只落表不下注。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        # 入场报价源（只读共享缓存，main 装配传入）：窗口对齐+龄守卫时快照 UP/DOWN 真实报价
        self._pm_15m_latest = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()   # 在飞的入场确认任务（stop 时取消）
        self._watched: set[int] = set()          # 已派生确认任务的 next_start（去重）
        self._trigger_count = 0
        self._settle_count = 0
        # 实盘开火钩子（main 装配注入 multi_live_trader.on_s2_cond_signal；
        # 未注入 = 纯影子模式，只落表不下注）
        self._on_live_fire = None

    # ------------------------------------------------------------------
    # 实盘 S2 钩子（同步接口，fire-and-forget；main 装配 fake_breakout._on_s2_cond）
    # ------------------------------------------------------------------

    def on_s2_signal(self, sig: dict) -> None:
        """fake_breakout 命中实盘 S2(bear_exhaust) 时回调（复用实盘检测口径）。

        payload：{id, pattern_type, market_start_15m, market_end_15m}。按 next_start
        去重后 spawn 异步入场确认任务（睡到 t=4/t=5 判价）；任何异常只告警不抛
        （绝不阻塞实盘检测循环——同 on_scene_signal/on_s5_deep_signal 契约）。
        """
        try:
            if not self._running:
                return  # stop 后拒绝派生新确认任务（与 check/scene 同契约）
            next_start = int(sig["market_start_15m"])
            next_end = int(sig["market_end_15m"])
            if next_start in self._watched:
                return
            self._watched.add(next_start)
            task = asyncio.create_task(
                self._confirm_entry(sig.get("id"), next_start, next_end),
                name=f"s2cond_{next_start}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.warning("S2 条件单影子：钩子异常（不影响检测循环）| {}", exc)

    # ------------------------------------------------------------------
    # 入场确认（延迟任务：t=4 判 A、t=5 判 B，命中快照报价并落 PENDING）
    # ------------------------------------------------------------------

    def _dispatch_live(self, version: str, next_start: int, next_end: int,
                       parent_id, up: float | None) -> None:
        """实盘开火分派（2026-09-06 promote）：判定命中即回调 MultiLiveTrader。

        与影子 gate 解耦（影子下线只停落表，实盘由 trader 侧通道 enabled 管）；
        钩子未装配 = 纯影子模式直接跳过；异常只告警不抛（不阻断影子落表）。
        """
        hook = self._on_live_fire
        if hook is None:
            return
        try:
            hook({
                "version": version,
                "market_start_15m": next_start,
                "market_end_15m": next_end,
                "parent_id": parent_id,
                "up_quote": up,
            })
        except Exception as exc:
            logger.warning("S2 条件单：实盘开火分派异常（不影响影子采集）| {}", exc)

    async def _confirm_entry(self, parent_id, next_start: int, next_end: int) -> None:
        """睡到窗内 t=4/t=5，回读 1m K 判价，命中则快照 UP 报价并落影子行。

        时序：t=4+GRACE → 重试拉 1m K 取 open(=1m bar next_start.open)/px4(=1m bar
        next_start+180s.close) → judge_t4 命中落 VAR_T4；再 t=5+GRACE → 取 px5(=1m bar
        next_start+240s.close) → judge_t5d 命中落 VAR_T5D。1m K 拉不到/价非法 → 放弃该版
        （不影响另一版、不影响实盘父信号）。全程异常只告警不抛。
        """
        try:
            # ---- 变体 A：t=4 价<开盘（全深度）----
            await self._sleep_until(next_start + T4_DELAY_MS + GRACE_MS)
            deadline4 = min(next_start + T4_DELAY_MS + MAX_WAIT_MS, next_end)
            bars4 = await self._fetch_1m_bars(next_start, deadline4, (0, PX4_OFFSET_MS))
            open_px = self._bar_open(bars4.get(0))
            px4 = self._bar_close(bars4.get(PX4_OFFSET_MS))
            if open_px is not None and px4 is not None and judge_t4(open_px, px4):
                up, down, ts = self._snapshot_up_quote(next_start)
                # 先分派实盘（独立于影子 gate），再落影子行（gate 管采集）
                self._dispatch_live(VAR_T4, next_start, next_end, parent_id, up)
                await self._record(VAR_T4, DISCOVERY_ID_T4, RULE_TEXT_T4,
                                   next_start, open_px, px4, up, down, ts, parent_id)

            # ---- 变体 B：t=5 剔深（0<回落<15bp）----
            await self._sleep_until(next_start + T5_DELAY_MS + GRACE_MS)
            deadline5 = min(next_start + T5_DELAY_MS + MAX_WAIT_MS, next_end)
            bars5 = await self._fetch_1m_bars(next_start, deadline5, (0, PX5_OFFSET_MS))
            if open_px is None:
                open_px = self._bar_open(bars5.get(0))
            px5 = self._bar_close(bars5.get(PX5_OFFSET_MS))
            if open_px is not None and px5 is not None and judge_t5d(open_px, px5):
                up, down, ts = self._snapshot_up_quote(next_start)
                # 先分派实盘（独立于影子 gate），再落影子行（gate 管采集）
                self._dispatch_live(VAR_T5D, next_start, next_end, parent_id, up)
                await self._record(VAR_T5D, DISCOVERY_ID_T5D, RULE_TEXT_T5D,
                                   next_start, open_px, px5, up, down, ts, parent_id)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("S2 条件单影子：入场确认任务异常 #{} | {}", parent_id, exc)

    async def _fetch_1m_bars(self, next_start: int, deadline_ms: int,
                             offsets: tuple[int, ...]) -> dict[int, dict]:
        """重试拉 1m K，直到拿到 offsets（相对 next_start 的 ms 偏移）对应 bar 或超时/停机。

        返回 {offset: bar_dict}（未拿到的 offset 不在其中）。fetch_recent_klines 按币安
        服务器时间只返回已收盘 K，天然规避边界抢跑。
        """
        got: dict[int, dict] = {}
        want = set(offsets)
        while self._running and clock_sync.now_ms() < deadline_ms and want:
            try:
                klines = await self._collector.fetch_recent_klines("1m", KLINE_FETCH_1M)
            except Exception:
                klines = []
            by_off = {int(k["open_time"]) - next_start: k for k in klines}
            for off in list(want):
                bar = by_off.get(off)
                if bar is not None:
                    got[off] = bar
                    want.discard(off)
            if want:
                await asyncio.sleep(RETRY_INTERVAL_S)
        return got

    @staticmethod
    def _bar_open(bar: dict | None) -> float | None:
        """1m bar 的 open（>0 有效，否则 None）。"""
        if bar is None:
            return None
        try:
            v = float(bar["open"])
        except (KeyError, TypeError, ValueError):
            return None
        return v if v > 0 else None

    @staticmethod
    def _bar_close(bar: dict | None) -> float | None:
        """1m bar 的 close（>0 有效，否则 None）。"""
        if bar is None:
            return None
        try:
            v = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return None
        return v if v > 0 else None

    def _snapshot_up_quote(self, next_start: int) -> tuple[float | None, float | None, int | None]:
        """快照次周期 UP 真实报价（镜像 hm _finalize_entry 守卫，取 up_price）。

        守卫：窗口对齐 start_date==next_start ∧ 龄 now−updated_ts≤20s ∧ up∈(0,1)。
        返回 (up, down, ts)；不满足 → (None, None, None)（该笔 EV 不计，胜率仍结算）。
        """
        q = dict(self._pm_15m_latest)
        upd = q.get("updated_ts")
        up = q.get("up_price")
        down = q.get("down_price")
        if (q.get("start_date") == next_start and up is not None and upd is not None
                and clock_sync.now_ms() - int(upd) <= QUOTE_MAX_AGE_MS):
            try:
                up_f = float(up)
            except (TypeError, ValueError):
                return None, None, None
            if 0.0 < up_f < 1.0:
                try:
                    down_f = float(down) if down is not None else None
                except (TypeError, ValueError):
                    down_f = None
                return up_f, down_f, int(upd)
        return None, None, None

    async def _record(self, version: str, discovery_id: str, rule_text: str,
                      next_start: int, open_px: float, px: float,
                      up: float | None, down: float | None, q_ts: int | None,
                      parent_id) -> None:
        """幂等落 PENDING（shadow_gate 闸 + 唯一约束 (version, signal_bar_start) 先查后插）。

        signal_bar_start = next_start − 900_000（父 S2 根，保唯一）；signal_bar_end =
        target_bar_start = next_start（入场&结算的次周期）；direction=UP、timeframe=15m。
        feature_snapshot 存审计 {open, px, dip, parent_id}。报价缺失 → entry_up_price=None
        （该笔 EV 不计，胜率仍结算，同 kline 约定）。自持 session（t=4/t=5 相隔 60s，不
        跨睡持有连接；镜像 _fire_s5_deep_shadow）。
        """
        if not shadow_gate.is_enabled(version):
            return  # 手动下线/退役：停止采集新信号（历史保留；实盘钩子不受影响）
        signal_bar_start = next_start - PARENT_OFFSET_MS
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == version,
                    KlineShadowSignal.signal_bar_start == signal_bar_start,
                )
            )).scalar_one_or_none()
            if exists is not None:
                return
            session.add(KlineShadowSignal(
                version=version,
                discovery_id=discovery_id,
                condition_text=rule_text,
                timeframe="15m",
                signal_bar_start=signal_bar_start,
                signal_bar_end=next_start,
                direction="UP",
                target_bar_start=next_start,
                feature_snapshot={
                    "open": open_px, "px": px,
                    "dip": round(dip(open_px, px), 6), "parent_id": parent_id,
                },
                entry_up_price=up,
                entry_down_price=down,
                entry_quote_ts=q_ts,
                status="PENDING",
            ))
            await session.commit()
        self._trigger_count += 1
        logger.info(
            "S2 条件单影子触发 | {} | 父信号 #{} | 次周期 {} | open={:.2f} px={:.2f} "
            "dip={:+.4%} | UP 报价 {}",
            version, parent_id, next_start, open_px, px, dip(open_px, px),
            up if up is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # 结算（按 direction 判 win：UP→次周期收阳赢；仅认 S2_COND_VERSIONS + 15m K 线）
    # ------------------------------------------------------------------

    async def _settle_pending(self, closed_15m: list[dict]) -> None:
        """次周期 15m 收盘结算：用 15m K 按 direction 判 win，仅认 S2_COND_VERSIONS。

        version 严格隔离是生命线：只结自己的 2 个 version，绝不触碰共表的 nb_/combo_/
        krev_/反转行（复制 nextbar._settle_pending 结构，filtered to S2_COND_VERSIONS）。
        """
        by_start = {int(r["open_time"]): r for r in closed_15m}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(S2_COND_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(starts),
                )
            )).scalars().all()
            if not pendings:
                return
            for sig in pendings:
                bar = by_start[int(sig.target_bar_start)]
                o, c = float(bar["open"]), float(bar["close"])
                if c == o:
                    sig.settle_outcome, sig.win, sig.status = "NOISE", None, "EXPIRED"
                else:
                    up = c > o
                    sig.settle_outcome = "UP" if up else "DOWN"
                    # direction=UP → 次周期收阳赢；direction=DOWN → 收阴赢（本族恒 UP）
                    sig.win = (up and sig.direction == "UP") or (not up and sig.direction == "DOWN")
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "S2 条件单影子结算 | {} | 次周期 {} → {} | win={}",
                    sig.version, int(sig.target_bar_start), sig.settle_outcome,
                    sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self) -> None:
        """目标根起点后 4h 仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED（仅本族 version）。"""
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(S2_COND_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("S2 条件单影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("S2 条件单影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        """轮询 15m 收盘 → 结算到期 PENDING + 过期兜底（超时清理不依赖 closed）。"""
        closed = await self._collector.fetch_recent_klines("15m", SETTLE_BARS)
        if closed:
            await self._settle_pending(closed)
        await self._expire_stale_pending()

    async def _sleep_until(self, target_ms: int) -> None:
        """睡到目标时刻（币安服务器时钟；分段睡防长阻塞 stop()）。"""
        while self._running:
            wait_s = (target_ms - clock_sync.now_ms()) / 1000
            if wait_s <= 0:
                return
            await asyncio.sleep(min(wait_s, 30))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._backscan()
        except Exception as exc:
            logger.warning("S2 条件单影子：冷启动结算失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="s2_cond_shadow_detector")
        logger.info(
            "S2 条件单检测器启动 | {} | 实盘 S2(bear_exhaust) 派生窗内 t=4/t=5 条件确认 "
            "→ 押 UP（720d 触发 1069/643，价-only 胜率 38.9%/44.8%；低买 UP 正 EV 来自入场价，"
            "真实 EV 前向现算）（影子落表 + 实盘通道已注册，开火由 trader 侧 enabled 管）",
            "/".join(S2_COND_VERSIONS),
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
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("S2 条件单影子检测器已停止 | 触发 {} 结算 {}",
                    self._trigger_count, self._settle_count)

    async def _backscan(self) -> None:
        """启动即结算/过期停机期间已到期的 PENDING 行（**不回补入场**，同 S5-deep）。"""
        closed = await self._collector.fetch_recent_klines("15m", SETTLE_BARS)
        if not closed:
            logger.warning("S2 条件单影子：冷启动 15m 数据为空，跳过结算回补")
            return
        await self._settle_pending(closed)
        await self._expire_stale_pending()

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "active_confirm_tasks": len(self._tasks),
            "watched": len(self._watched),
            "versions": list(S2_COND_VERSIONS),
        }
