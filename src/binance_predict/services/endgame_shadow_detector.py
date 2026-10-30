"""尾盘买热门影子检测器（endgame 族）：15m 窗口最后 45 秒买当前热门侧的实时重放。

信号定义（研究冻结口径见 docs/research/SIGNAL_SCORECARD_20261030.md）：
    触发（窗口内 5s 对齐采样喂价）：t_rel = (ts − window_start)/1000 ∈ [855, 900)
        且**热门一侧**展示价 ∈ [0.65, 0.85)
        —— 热门 = up_price ∈ [0.65,0.85) 时押 UP，否则 down_price ∈ [0.65,0.85) 时押 DOWN
    结算 win = close(本 15m 窗) > open（押 UP；押 DOWN 取反；平局 NOISE）

研究依据（PREDEXON 2026-07→10，9,749 个 15m 市场、1,450 次触发）：
    胜率 81.0%、净 +3.9%（固定 1.9%/赔付；按真实价格相关费率 +5.2%）、
    按天聚类 t=2.97、4/4 月为正、13.1 笔/天。
    **幅度对定义敏感**：放宽到 0.60–0.80 / t≥840 只剩 +1.5%（t=0.75）——
    故本通道只做影子前向验证，实盘默认 OFF。
    ⚠ 本质是速度套利：延迟 1s 剩 68%、2s 剩 48%（生产实测中位 1.8s），
    且限价挂单被逆向选择吃死（挂低 1 分钱胜率从 81% 掉到 71%）。

影子纪律：每个触发窗落一行（PENDING），触发即回填入场价；窗口收盘结算。
只落表不下注；实盘开火由 on_live_fire 钩子派发到
MultiLiveTrader.on_endgame_signal（通道默认 OFF）。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal

ENDGAME_VERSION = "endgame_fav_15m_v1"
BAR_15M_MS = 900_000
# 触发窗口：窗口内 [855s, 900s)。起点取 855 与研究的 t≥855 一致；
# 终点即窗口结束（此后市场已结算，报价无意义）。
TRIGGER_FROM_S = 855
# 热门侧价位带（研究冻结）：0.65 起为"热门"，0.85 以上边际低于成本（费率+点差吃掉）。
FAV_LO = 0.65
FAV_HI = 0.85
# 结算轮询间隔：15s（与 excur 同口径，收盘后 15s 内结算）
SETTLE_POLL_S = 15


def pick_favorite(up_price: float | None, down_price: float | None) -> tuple[str, float] | None:
    """返回 (方向, 热门价)；两侧都不在带内返回 None。纯函数，便于单测。

    ⚠口径说明：入参是币安**展示价**（接近可成交 ask，而非中间价），
    而回测触发用的是中间价 ∈[0.65,0.85)。两者差半个点差，
    因此影子触发的价位分布会比回测略窄（上边界更严），属预期偏差，不做修正。
    """
    if up_price is not None and FAV_LO <= float(up_price) < FAV_HI:
        return "UP", float(up_price)
    if down_price is not None and FAV_LO <= float(down_price) < FAV_HI:
        return "DOWN", float(down_price)
    return None


class EndgameFavShadowDetector:
    """尾盘买热门：影子落表 + 实盘钩子（默认不下注）。"""

    def __init__(self, collector=None) -> None:
        self._collector = collector
        self._running = False
        self._task: asyncio.Task | None = None
        self._fired: set[int] = set()      # 已触发 window_start（内存去重）
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._collector is None:
            # 结算靠 collector.fetch_recent_klines；缺失会让行永远卡 PENDING，
            # 必须启动时大声报错（不能静默退化）。
            logger.error("endgame 影子：未注入 collector，结算不会运行（行将积压）")
        self._task = asyncio.create_task(self._settle_loop(), name="endgame_shadow_detector")
        logger.info("endgame 尾盘买热门检测器启动 | {} | 影子落表 + 实盘钩子（默认关闭）",
                    ENDGAME_VERSION)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def observe_sample(self, window_start: int, window_end: int,
                             ts_ms: int, up_price: float | None,
                             down_price: float | None) -> None:
        """每次 15m 对齐采样调用（main._handle_15m_quote persist 路径）。"""
        if window_start in self._fired:
            return
        t_rel = (ts_ms - window_start) / 1000.0
        if t_rel < TRIGGER_FROM_S or ts_ms >= window_end:
            return
        pick = pick_favorite(up_price, down_price)
        if pick is None:
            return
        direction, fav_price = pick
        row_id = await self._ensure_row(window_start, window_end, ts_ms, direction,
                                        fav_price, up_price, down_price, t_rel)
        if row_id is None:
            return
        # 去重标记放在调用侧：不依赖 _ensure_row 内部实现（重放/重启路径也要幂等）
        self._fired.add(window_start)
        self._trigger_count += 1
        logger.info(
            "endgame 影子触发 | {} | 窗口 {} | t=+{:.1f}s 热门={} 价={} | up={} down={}",
            ENDGAME_VERSION, window_start, t_rel, direction, fav_price, up_price, down_price)
        if self._on_live_fire is not None:
            try:
                self._on_live_fire({
                    "version": ENDGAME_VERSION,
                    "market_start": window_start,
                    "market_end": window_end,
                    "direction": direction,
                    "signal_bar_start": window_start,
                    "favored_price": fav_price,
                    "t_rel": round(t_rel, 1),
                })
            except Exception as exc:
                logger.warning("endgame 实盘钩子异常（不影响影子）| {}", exc)

    async def _ensure_row(self, window_start: int, window_end: int, ts_ms: int,
                          direction: str, fav_price: float,
                          up_price: float | None, down_price: float | None,
                          t_rel: float) -> int | None:
        """触发行只落一次（DB 存在性核验 + 内存缓存，防重启/重放重复落表）。"""
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == ENDGAME_VERSION,
                    KlineShadowSignal.signal_bar_start == window_start,
                ).limit(1))).first()
            if exists:
                self._fired.add(window_start)
                return None
            row = KlineShadowSignal(
                version=ENDGAME_VERSION,
                discovery_id="endgame_fav_15m",
                condition_text=(
                    f"t_rel >= {TRIGGER_FROM_S}s AND fav_price in "
                    f"[{FAV_LO}, {FAV_HI}) AND hold_to_window_close"),
                timeframe="15m",
                signal_bar_start=window_start,
                signal_bar_end=window_start + BAR_15M_MS,
                direction=direction,
                # 目标根 = 本窗（尾盘入场，结算看本窗收盘）
                target_bar_start=window_start,
                entry_up_price=(float(up_price) if up_price is not None else None),
                entry_down_price=(float(down_price) if down_price is not None else None),
                entry_quote_ts=int(ts_ms),
                feature_snapshot={
                    "t_rel_s": round(t_rel, 1),
                    "fav_side": direction,
                    "fav_price": round(float(fav_price), 4),
                    "up_price": (round(float(up_price), 4) if up_price is not None else None),
                    "down_price": (round(float(down_price), 4) if down_price is not None else None),
                    "cross_cost": (round(float(up_price) + float(down_price) - 1, 4)
                                   if up_price is not None and down_price is not None else None),
                    "entry_quote_kind": "market_snapshot",
                    "trigger_from_s": TRIGGER_FROM_S,
                },
                status="PENDING",
            )
            session.add(row)
            await session.commit()
            self._fired.add(window_start)
            return int(row.id)

    async def _settle_loop(self) -> None:
        """后台结算：本 15m 根收盘后按 close vs open 结算触发行。"""
        while self._running:
            try:
                await self._settle_pending()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("endgame 影子：结算循环异常 | {} | {}",
                               type(exc).__name__, exc)
            try:
                await asyncio.sleep(SETTLE_POLL_S)
            except asyncio.CancelledError:
                break

    async def _settle_pending(self) -> None:
        if self._collector is None:
            return
        now_ms = int(time.time() * 1000)
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version == ENDGAME_VERSION,
                    KlineShadowSignal.status == "PENDING",
                ))).scalars().all()
            if not rows:
                return
            oldest = min(int(r.target_bar_start) for r in rows)
            need = max(8, min((now_ms - oldest) // BAR_15M_MS + 2, 288))
            closed15 = await self._collector.fetch_recent_klines("15m", need)
            # bar 无 close_time 字段（同 wbid 2026-10-25 生产修复）：收盘时刻用 open_time 推导。
            by_start = {int(b["open_time"]): b for b in closed15} if closed15 else {}
            for r in rows:
                bar = by_start.get(int(r.target_bar_start))
                if bar is None or int(bar["open_time"]) + BAR_15M_MS > now_ms:
                    if now_ms - int(r.target_bar_start) > 10 * BAR_15M_MS:
                        r.status = "EXPIRED"
                    continue
                o, c = float(bar["open"]), float(bar["close"])
                r.settle_open, r.settle_close = o, c
                outcome = "UP" if c > o else ("DOWN" if c < o else "NOISE")
                r.settle_outcome = outcome
                if outcome == "NOISE":
                    r.win = None
                else:
                    r.win = (outcome == r.direction)
                r.status = "SETTLED"
                r.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
            await session.commit()
