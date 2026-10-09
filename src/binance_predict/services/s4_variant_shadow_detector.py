"""S4 动量衰竭派生影子：延迟入场（影子+实盘通道）/ 高信心档（record-only）；
另承载 S1 入场变体（早确认 / 低吸：影子 + 实盘通道共用同一触发，实盘默认关闭）。

四个版本均由 FakeBreakoutDetector 在 15m 周期收盘确认时派生，落 kline_shadow_signals，
押目标 15m 周期 DOWN，用完整 15m K 线结算。s4_hiconf 仅记录不接实盘钩子；
其余三版命中时另经 _on_live_fire 通知 MultiLiveTrader（通道未启用即无动作）。

- s4_delay60_v1（2026-10-09 注册实盘）：正式 S4（已跳过续发）命中后，等目标窗
  第 1 根 1m 收盘（+60s），若该 1m 收盘 < 窗口开盘（BTC 已回落）才入场，
  入场价快照 +60s 后的真实 DOWN 投价。
  依据：predict.fun 真实盘口 2026-05→10-03，P1 均EV +0.119（n56）/ P2 +0.107（n96），
  720d K 线重放 n=837 胜率 66.2%。入场价更高（≈0.62），未扣 20U 深度滑点。
  实盘通道同名（默认 OFF），订单 scene_signal_id = 父 S4 信号 id；与
  scene_momentum_fade 同窗同向（S4 开盘单 +60s 确认单叠加敞口），叠加敞口
  提示见 live_channels 注释。
- s4_hiconf_v1：连阳≥5（含信号K）∧ close_pos≥0.9，允许破位周期（与 S1 重叠约 52%）。
  依据：720d 重放 n=399 胜率 63.2%；真实盘口 P1/P2 均EV +0.119/+0.147（n26/n52）。
- s1_early2_v1（S1m2）：正式 S1 命中后，目标窗第 2 根 1m 收盘 < 窗口开盘（BTC 已回落）
  才在 +2min 入场押 DOWN。替代 S5（+5min）的早确认版：买得更便宜（≈0.67 对 0.71）。
  依据：scene_opt 研究，真实盘口 P1/P2 均EV +0.118/+0.099（S5 为 +0.015/+0.080），
  720d K 线均EV +0.090 对 +0.058；胜率比 S5 低约 3pp。
- s1_dip45_v1（S1 低吸）：正式 S1 命中后，在 +62/122/182/302/482s 五个时点首次见
  DOWN 报价 ≤ 阈值即入场。依据：真实盘口 P1/P2 均EV +0.097/+0.199（n57/n123），
  同规则用在随机窗口上为负/≈0；与 S1 开盘单同向叠加敞口。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from binance_predict.services import clock_sync
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.signal_notify import is_live_enabled
from binance_predict.services.shadow_version_gate import shadow_gate

S4_DELAY_VERSION = "s4_delay60_v1"
S4_HICONF_VERSION = "s4_hiconf_v1"
S1_EARLY_VERSION = "s1_early2_v1"
S1_DIP_VERSION = "s1_dip45_v1"
# 另有同名实盘通道的版本（live_channels 同名注册，默认 OFF）：
# 实盘开关开启时即使影子下线也照常派生（两开关独立，同 s1_dyn_sq_v1）。
LIVE_VARIANT_VERSIONS = (S4_DELAY_VERSION, S1_EARLY_VERSION, S1_DIP_VERSION)
S4_VARIANT_VERSIONS = (S4_DELAY_VERSION, S4_HICONF_VERSION, S1_EARLY_VERSION, S1_DIP_VERSION)
BAR_MS_15M = 900_000
DELAY_MS = 60_000            # 目标窗第 1 根 1m 收盘时刻
GRACE_MS = 8_000             # 收盘后缓冲（等 1m K 落库 / 报价缓存刷新）
MAX_WAIT_MS = 90_000         # 1m K 拉不到的放弃上限
RETRY_INTERVAL_S = 5
EARLY_MS = 120_000           # S1m2：第 2 根 1m 收盘时刻
EARLY_MAX_WAIT_MS = 30_000   # S1m2：1m K 拉不到的放弃上限（入场时点已过，不追）
EARLY_QUOTE_MIN_OFFSET_MS = 105_000   # S1m2 入场报价须取自 +105s 之后（排除开盘陈旧值）
EARLY_QUOTE_MAX_OFFSET_MS = 180_000
# 低吸检查时点 = 研究的 1/2/3/5/8 分钟快照（+2s 对齐报价缓存刷新）；不改成连续轮询，
# 保持与已验证规则同口径。
DIP_OFFSETS_MS = (62_000, 122_000, 182_000, 302_000, 482_000)
# 触发阈值：报价缓存 DOWN 价 ≤ 0.42。研究用的是真实盘口卖一价 ≤ 0.45，而缓存价比真实成交
# 价低约 0.03（S4 校准：pm_samples 口径系统性偏低），故折算为 0.42；实盘护栏 0.45 管真实成交价。
DIP_TRIGGER = 0.42
DIP_QUOTE_FRESH_MS = 20_000
POLL_INTERVAL = 60.0
SETTLE_BARS = 40
PENDING_EXPIRE_MS = 4 * 3_600_000
RULE_TEXT = {
    S4_DELAY_VERSION: (
        "S4延迟入场：正式S4(连阳≥3+光头阳,无破位,已跳过续发)命中后，目标窗第1根1m收盘<窗口开盘"
        "(BTC已回落)才入场，入场价取+60s真实DOWN报价；押目标15m DOWN。影子+同名实盘通道(默认关闭)。"
    ),
    S1_EARLY_VERSION: (
        "S1早确认(S1m2)：正式S1命中后，目标窗第2根1m收盘<窗口开盘(BTC已回落)才在+2min入场，"
        "入场价取+2min真实DOWN报价；押目标15m DOWN。影子+同名实盘通道(默认关闭)。"
    ),
    S1_DIP_VERSION: (
        "S1低吸：正式S1命中后，在+62/122/182/302/482s首次见DOWN报价≤0.42(≈真实成交0.45)即入场；"
        "押目标15m DOWN。与S1开盘单同向叠加敞口。影子+同名实盘通道(默认关闭)。"
    ),
    S4_HICONF_VERSION: (
        "S4高信心档：连阳≥5(含信号K)且收盘位置≥0.9，允许破位周期(与S1重叠)；押次周期15m DOWN，"
        "入场价取开盘后真实DOWN报价。仅记录不下单。"
    ),
}
DISCOVERY_ID = {S4_DELAY_VERSION: "s4_delay60", S4_HICONF_VERSION: "s4_hiconf",
                S1_EARLY_VERSION: "s1_early2", S1_DIP_VERSION: "s1_dip45"}


def first_minute_dropped(bar: dict | None, window_start: int) -> bool:
    """目标窗第 1 根 1m：收盘严格低于开盘（= 窗口开盘价）才算已回落；平盘/缺失不算。"""
    if bar is None or int(bar.get("open_time", -1)) != window_start:
        return False
    try:
        o, c = float(bar["open"]), float(bar["close"])
    except (KeyError, TypeError, ValueError):
        return False
    return o > 0 and c > 0 and c < o


def early_confirmed(bar1: dict | None, bar2: dict | None, window_start: int) -> bool:
    """S1m2 判定：目标窗第 2 根 1m 收盘严格低于第 1 根 1m 开盘（= 窗口开盘）；缺根/错位/平盘不算。"""
    if bar1 is None or bar2 is None:
        return False
    if int(bar1.get("open_time", -1)) != window_start or int(bar2.get("open_time", -1)) != window_start + 60_000:
        return False
    try:
        o, c = float(bar1["open"]), float(bar2["close"])
    except (KeyError, TypeError, ValueError):
        return False
    return o > 0 and c > 0 and c < o

class S4VariantShadowDetector:
    """S4 派生影子：钩子同步接收、异步确认，落 PENDING，轮询 15m K 线结算。"""

    def __init__(self, collector, pm_15m_latest: dict) -> None:
        self._collector = collector
        self._pm_15m_latest = pm_15m_latest
        self._running = False
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._watched: set[tuple[str, int]] = set()
        self._trigger_count = 0
        self._settle_count = 0
        # S1/S4 派生实盘钩子（main 注入 MultiLiveTrader.on_s1_variant_signal）；与影子 gate 独立
        self._on_live_fire = None

    # ---------------- 钩子（FakeBreakoutDetector 同步调用，异常不外抛）----------------
    def on_delay(self, sig: dict) -> None:
        self._spawn(S4_DELAY_VERSION, sig)

    def on_hiconf(self, sig: dict) -> None:
        self._spawn(S4_HICONF_VERSION, sig)

    def on_s1_entry(self, sig: dict) -> None:
        self._spawn(S1_EARLY_VERSION, sig)
        self._spawn(S1_DIP_VERSION, sig)

    def _spawn(self, version: str, sig: dict) -> None:
        try:
            # 同名实盘通道版本：影子下线不阻止已启用的实盘（两个开关独立，同 s1_dyn_sq_v1）
            if not self._running or not (
                shadow_gate.is_enabled(version)
                or (version in LIVE_VARIANT_VERSIONS and is_live_enabled(version))
            ):
                return
            key = (version, int(sig["market_start_15m"]))
            if key in self._watched:
                return
            self._watched.add(key)
            task = asyncio.create_task(self._confirm(version, sig), name=f"s4var_{version}_{key[1]}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as exc:
            logger.warning("S4派生影子：钩子异常（不影响检测循环）| {} | {}", version, exc)

    async def _confirm(self, version: str, sig: dict) -> None:
        try:
            ws = int(sig["market_start_15m"])
            snap = {"close_pos": sig.get("close_pos"), "hi_break": sig.get("hi_break")}
            if version in (S1_EARLY_VERSION, S1_DIP_VERSION):
                await self._confirm_s1_variant(version, sig, ws)
                return
            if version == S4_DELAY_VERSION:
                await self._sleep_until(ws + DELAY_MS + GRACE_MS)
                bar = await self._first_minute(ws)
                if not first_minute_dropped(bar, ws):
                    return
                snap.update(m1_open=float(bar["open"]), m1_close=float(bar["close"]))
                up, down, ts = snapshot_entry_quote(self._pm_15m_latest, ws)
                # 与 S1 变体同序：先实盘后落表（时效优先）；id 缺失（父 S4 落库
                # 失败）时放弃实盘派单——无 scene_signal_id 溯源的订单不可结算。
                self._fire_live(version, sig, ws)
                await self._record(version, int(sig["signal_bar_start"]), ws, snap, up, down, ts)
            else:
                await self._sleep_until(ws + GRACE_MS)
                up, down, ts = snapshot_entry_quote(self._pm_15m_latest, ws)
                await self._record(version, int(sig["signal_bar_start"]), ws, snap, up, down, ts)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("S4派生影子：确认任务异常 | {} | {}", version, exc)

    async def _confirm_s1_variant(self, version: str, sig: dict, ws: int) -> None:
        """S1 早确认 / 低吸：命中即先通知实盘（时效优先），再落影子。"""
        snap: dict = {}
        if version == S1_EARLY_VERSION:
            await self._sleep_until(ws + EARLY_MS + GRACE_MS)
            bar1, bar2 = await self._first_two_minutes(ws)
            if not early_confirmed(bar1, bar2, ws):
                return
            snap.update(m1_open=float(bar1["open"]), m2_close=float(bar2["close"]))
            up, down, ts = snapshot_entry_quote(self._pm_15m_latest, ws, EARLY_QUOTE_MAX_OFFSET_MS)
            if ts is not None and ts - ws < EARLY_QUOTE_MIN_OFFSET_MS:
                up = down = ts = None   # 开盘陈旧值不冒充 +2min 入场价
        else:
            for off in DIP_OFFSETS_MS:
                await self._sleep_until(ws + off)
                if not self._running:
                    return
                up, down, ts = snapshot_entry_quote(self._pm_15m_latest, ws, off + 30_000)
                if down is None or ts < clock_sync.now_ms() - DIP_QUOTE_FRESH_MS:
                    continue
                if down <= DIP_TRIGGER:   # 贴线含
                    snap.update(dip_offset_s=off // 1000, dip_trigger=DIP_TRIGGER)
                    break
            else:
                return
        self._fire_live(version, sig, ws)
        await self._record(version, int(sig["signal_bar_start"]), ws, snap, up, down, ts)

    def _fire_live(self, version: str, sig: dict, ws: int) -> None:
        """同步 fire-and-forget；通道未启用时执行器直接返回，异常不影响影子落表。

        sig["id"] = 父场景信号行 id（订单 scene_signal_id 溯源锚点，结算回读
        FakeBreakoutSignal）；缺失（父信号落库失败/旧 payload）→ 放弃实盘派单，
        无溯源锚点的订单不可结算（宁少勿多）。
        """
        from .shadow_forward_evidence import observe_kline
        observe_kline({"version": version, "market_start": ws,
                       "market_end": ws + BAR_MS_15M, "direction": "DOWN"})
        if self._on_live_fire is None:
            return
        try:
            sig_id = sig.get("id")
            if sig_id is None:
                logger.warning("S4派生实盘派单放弃：父信号 id 缺失 | {} | 窗口 {}", version, ws)
                return
            self._on_live_fire({
                "version": version, "id": int(sig_id), "pattern_type": version, "side": "high",
                "market_start_15m": ws, "market_end_15m": ws + BAR_MS_15M,
            })
        except Exception as exc:
            logger.warning("S4派生实盘钩子异常（不影响影子）| {} | {}", version, exc)

    async def _first_two_minutes(self, ws: int) -> tuple[dict | None, dict | None]:
        deadline = ws + EARLY_MS + EARLY_MAX_WAIT_MS
        while self._running and clock_sync.now_ms() < deadline:
            try:
                bars = await self._collector.fetch_recent_klines("1m", 4)
            except Exception:
                bars = []
            by = {int(b["open_time"]): b for b in bars}
            if ws in by and ws + 60_000 in by:
                return by[ws], by[ws + 60_000]
            await asyncio.sleep(RETRY_INTERVAL_S)
        return None, None

    async def _first_minute(self, ws: int) -> dict | None:
        deadline = ws + DELAY_MS + MAX_WAIT_MS
        while self._running and clock_sync.now_ms() < deadline:
            try:
                bars = await self._collector.fetch_recent_klines("1m", 3)
            except Exception:
                bars = []
            bar = next((b for b in bars if int(b["open_time"]) == ws), None)
            if bar is not None:
                return bar
            await asyncio.sleep(RETRY_INTERVAL_S)
        return None

    async def _sleep_until(self, target_ms: int) -> None:
        while self._running:
            wait_s = (target_ms - clock_sync.now_ms()) / 1000
            if wait_s <= 0:
                return
            await asyncio.sleep(min(wait_s, 30))

    async def _record(self, version: str, signal_start: int, target_start: int, snap: dict,
                      up: float | None, down: float | None, quote_ts: int | None) -> None:
        if not shadow_gate.is_enabled(version):
            return
        async with async_session_factory() as session:
            exists = (await session.execute(
                sa_select(KlineShadowSignal.id).where(
                    KlineShadowSignal.version == version,
                    KlineShadowSignal.signal_bar_start == signal_start,
                )
            )).scalar_one_or_none()
            if exists is not None:
                return
            session.add(KlineShadowSignal(
                version=version,
                discovery_id=DISCOVERY_ID[version],
                condition_text=RULE_TEXT[version],
                timeframe="15m",
                signal_bar_start=signal_start,
                signal_bar_end=target_start,
                direction="DOWN",
                target_bar_start=target_start,
                feature_snapshot=snap,
                entry_up_price=up,
                entry_down_price=down,
                entry_quote_ts=quote_ts,
                status="PENDING",
            ))
            await session.commit()
        self._trigger_count += 1
        logger.info("S4派生影子触发 | {} | 目标窗 {} | DOWN报价 {}", version, target_start,
                    down if down is not None else "N/A")

    # ---------------- 结算 ----------------
    async def settle(self, closed_15m: list[dict]) -> None:
        by_start = {int(bar["open_time"]): bar for bar in closed_15m}
        if not by_start:
            return
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(S4_VARIANT_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start.in_(list(by_start)),
                )
            )).scalars().all()
            for row in rows:
                bar = by_start[int(row.target_bar_start)]
                o, c = float(bar["open"]), float(bar["close"])
                row.settle_open, row.settle_close = o, c
                row.settled_at = datetime.now(timezone.utc)
                if c == o:
                    row.settle_outcome, row.win, row.status = "NOISE", None, "EXPIRED"
                else:
                    row.settle_outcome = "UP" if c > o else "DOWN"
                    row.win = row.settle_outcome == row.direction
                    row.status = "SETTLED"
            if rows:
                self._settle_count += len(rows)
                await session.commit()

    async def _expire_stale(self) -> None:
        cutoff = int(time.time() * 1000) - BAR_MS_15M - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            rows = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(S4_VARIANT_VERSIONS),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            for row in rows:
                row.status = "EXPIRED"
            if rows:
                await session.commit()

    async def _poll_once(self) -> None:
        closed = await self._collector.fetch_recent_klines("15m", SETTLE_BARS)
        if closed:
            await self.settle(closed)
        await self._expire_stale()

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)
                await self._poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("S4派生影子：结算轮询异常 | {}", exc)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._poll_once()
        except Exception as exc:
            logger.warning("S4派生影子：冷启动结算失败（循环内重试）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="s4_variant_shadow_detector")

    async def stop(self) -> None:
        self._running = False
        for t in list(self._tasks):
            t.cancel()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def status(self) -> dict:
        return {"versions": list(S4_VARIANT_VERSIONS), "running": self._running,
                "trigger_count": self._trigger_count, "settle_count": self._settle_count,
                "in_flight": len(self._tasks)}
