"""复利下单链（手动）：A→B→C… 依次下注 5m 窗口，上一腿赢则把到手股数（本金+利润）滚入下一腿。

设计取舍（真金，宁停勿错）：
- 全链同一方向、同一执行价护栏；输即终止。提前下单使最大亏损可能扩大到「当前腿 + 已提前下的下一腿」。
- 提前下单（用户拍板 2026-10-10）：腿 N 成交后盯本窗实时报价与 BTC 领先幅度，命中任一条件
  即把腿 N+1 提前押上，锚定开盘附近的 0.5 报价（未来窗开盘报价中位 0.51，89% 落在 0.45~0.55）：
    ① 距收盘 60~150s 且领先一侧报价 ≥0.95
    ② 距收盘 ≤15s 且 |现价−开盘| ≥3bp 且方向与下注一致
  两条件都未命中则退回「收盘后按 K 线定输赢再下」的保守路径。
- 判赢输用币安 5m K 线 open/close（与 sentiment 归档同源），不等 7 分钟的结算器。
- 下一腿本金 = 上一腿 filledShareQty（已扣费到手股数，1 股=1U），不封顶；提前下单时上一腿尚未
  结算，按「本金 + 按报价算得的利润」＝预估到手股数下（已下订单无法事后改额）。
- 赢单 token 后台 batch-redeem（10 分钟内重试），不阻塞下一腿；下一腿由钱包可用余额垫付。
- 状态全落 compound_chains；重启后按 RUNNING 续跑，下单前先认领已存在订单（防重复下单）。
"""
from __future__ import annotations

import asyncio
import json
import math
import time

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.sql import func

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import CompoundChain, TradeOrderModel
from binance_predict.services.wechat_notifier import wechat_notifier

STEP_MS = 300_000
MAX_BASE = 10.0           # 首腿本金上限（录入防手滑）；后续腿滚动不封顶
MIN_BASE = 0.1
MAX_LEGS = 8
MIN_LEAD_MS = 5_000       # 未来窗距开盘 <5s 拒（同 /api/trade/test）
MIN_REMAIN_MS = 15_000    # 下单时目标窗剩余 <15s 放弃（成交确认 + 币安收盘前拒单）
RESOLVE_DELAY_MS = 3_000  # 窗口收盘后等 K 线定稿
REDEEM_RETRY_S = 5
VERSION_PREFIX = "manual_chain_"
POLL_S = 2                # 提前下单条件的盯盘间隔
EARLY_QUOTE = 0.95        # 条件① 领先一侧报价阈值
EARLY_QUOTE_MIN_REMAIN_MS = 60_000
EARLY_QUOTE_MAX_REMAIN_MS = 150_000   # 条件① 上限：开盘前段冲高易回落，只在末段触发
LEAD_MS = 15_000          # 条件② 距收盘时间
LEAD_MIN_BP = 3.0         # 条件② 领先幅度（bp）


def chain_version(chain_id: int) -> str:
    return f"{VERSION_PREFIX}{chain_id}"


def validate_plan(base: float, direction: str, windows: list[int], guard: float,
                  now_ms: int) -> str | None:
    """返回错误文案；None=通过。"""
    if direction not in ("UP", "DOWN"):
        return "direction 仅允许 UP/DOWN"
    if not (MIN_BASE <= base <= MAX_BASE):
        return f"base_amount 仅允许 {MIN_BASE}~{MAX_BASE}"
    if not (0.0 < guard < 1.0):
        return "max_exec_price 仅允许 0~1 开区间"
    if not (2 <= len(windows) <= MAX_LEGS):
        return f"窗口数需 2~{MAX_LEGS}"
    if windows != sorted(set(windows)):
        return "windows 必须升序且不重复"
    if any(w % STEP_MS for w in windows):
        return "window_start 必须对齐 5m 网格"
    cur = now_ms // STEP_MS * STEP_MS
    first = windows[0]
    if first < cur:
        return "首窗不能早于当前窗口"
    if first == cur and first + STEP_MS - now_ms < 20_000:
        return "当前窗距收盘不足 20 秒，请从下一窗起选"
    if first > cur and first - now_ms < MIN_LEAD_MS:
        return "首窗距开盘不足 5 秒"
    if windows[-1] > now_ms + 80 * 60_000:
        return "窗口超出扫描 horizon（now+80min）"
    return None


def decide(direction: str, open_price: float, close_price: float) -> str:
    """WIN | LOSS | NOISE（收盘=开盘，方向无从判定，按终止处理）。"""
    if close_price == open_price:
        return "NOISE"
    outcome = "UP" if close_price > open_price else "DOWN"
    return "WIN" if outcome == direction else "LOSS"


def next_stake(shares: float) -> float:
    """下一腿本金：到手股数向下取 4 位（不封顶；余额不足由下单预检弃单终止链）。"""
    return math.floor(shares * 10_000) / 10_000


def early_trigger(direction: str, up: float | None, down: float | None,
                  mid: float | None, open_price: float | None,
                  remain_ms: int) -> str | None:
    """提前下单条件（返回命中的条件名；None=未命中，退回收盘后路径）。"""
    if up is None or down is None or up <= 0 or down <= 0:
        return None
    if ("UP" if up >= down else "DOWN") != direction:
        return None  # 市场领先方向已与下注方向相反：不下
    if EARLY_QUOTE_MIN_REMAIN_MS < remain_ms <= EARLY_QUOTE_MAX_REMAIN_MS and max(up, down) >= EARLY_QUOTE:
        return "quote"
    if remain_ms <= LEAD_MS and mid and open_price and open_price > 0:
        bp = (mid - open_price) / open_price * 1e4
        if (bp >= LEAD_MIN_BP and direction == "UP") or (bp <= -LEAD_MIN_BP and direction == "DOWN"):
            return "lead"
    return None


class CompoundChainRunner:
    def __init__(self, trader, collector, on_balance_change=None, quote_provider=None) -> None:
        self._trader = trader
        self._collector = collector
        self._on_balance_change = on_balance_change
        self._quote_provider = quote_provider   # () -> (up, down)：实时 5m 报价（每 15s 刷新）
        self._tasks: dict[int, asyncio.Task] = {}
        self._bg: set[asyncio.Task] = set()
        self._open_cache: dict[int, float] = {}

    # ---------------- 生命周期 ----------------

    async def start(self) -> None:
        async with async_session_factory() as s:
            ids = (await s.execute(
                select(CompoundChain.id).where(CompoundChain.status == "RUNNING"))).scalars().all()
        for cid in ids:
            self.spawn(cid)
        if ids:
            logger.info("复利链续跑 | ids={}", ids)

    def spawn(self, chain_id: int) -> None:
        t = asyncio.create_task(self._run(chain_id), name=f"compound_chain_{chain_id}")
        self._tasks[chain_id] = t
        t.add_done_callback(lambda _t, c=chain_id: self._tasks.pop(c, None))

    async def stop(self) -> None:
        pending = [*self._tasks.values(), *self._bg]
        for t in pending:
            t.cancel()  # 链状态仍 RUNNING，下次启动续跑；未赎回的奖金手动领取
        await asyncio.gather(*pending, return_exceptions=True)

    # ---------------- 可覆写依赖（测试替身） ----------------

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    async def _sleep(self, sec: float) -> None:
        await asyncio.sleep(sec)

    def _live_quote(self) -> tuple[float | None, float | None]:
        if self._quote_provider is not None:
            try:
                u, d = self._quote_provider()
                return (u if u and u > 0 else None, d if d and d > 0 else None)
            except Exception:
                return None, None
        return (getattr(self._trader, "_5m_up_price", None),
                getattr(self._trader, "_5m_down_price", None))

    def _mid_price(self) -> float | None:
        try:
            p = self._collector.mid_price()
            return p if p and p > 0 else None
        except Exception:
            return None

    async def _load(self, cid: int) -> dict | None:
        async with async_session_factory() as s:
            c = await s.get(CompoundChain, cid)
            if c is None:
                return None
            return {"id": c.id, "status": c.status, "direction": c.direction,
                    "base_amount": c.base_amount, "max_exec_price": c.max_exec_price,
                    "windows": list(c.windows), "legs": list(c.legs or []),
                    "stop_reason": c.stop_reason}

    async def _save(self, cid: int, **vals) -> None:
        async with async_session_factory() as s:
            await s.execute(update(CompoundChain).where(CompoundChain.id == cid).values(**vals))
            await s.commit()

    async def _existing_order(self, version: str, ws: int) -> dict | None:
        async with async_session_factory() as s:
            r = (await s.execute(select(TradeOrderModel).where(
                TradeOrderModel.signal_version == version,
                TradeOrderModel.window_start == ws))).scalar_one_or_none()
            if r is None:
                return None
            return {"id": r.id, "status": r.status, "token_id": r.token_id,
                    "amount_in": r.amount_in, "quote_json": r.quote_json,
                    "error_message": r.error_message}

    async def _mark_redeemed(self, order_id: int) -> None:
        async with async_session_factory() as s:
            await s.execute(update(TradeOrderModel).where(
                TradeOrderModel.id == order_id, TradeOrderModel.redeemed_at.is_(None)
            ).values(redeemed_at=func.now()))
            await s.commit()

    def _notify(self, text: str) -> None:
        try:
            wechat_notifier.push_markdown(text)
        except Exception as exc:  # 通知失败不影响资金流程
            logger.warning("复利链通知异常: {}", exc)

    # ---------------- 状态机 ----------------

    async def _run(self, cid: int) -> None:
        try:
            while True:
                ch = await self._load(cid)
                if ch is None or ch["status"] != "RUNNING":
                    return
                if await self._step(ch):
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("复利链异常终止 | id={}", cid)
            await self._finish(cid, "STOPPED", f"内部异常: {type(exc).__name__}: {exc}"[:200])

    async def _finish(self, cid: int, status: str, reason: str) -> bool:
        await self._save(cid, status=status, stop_reason=reason[:200])
        logger.warning("复利链结束 | id={} | {} | {}", cid, status, reason)
        self._notify(f"**复利链 #{cid} {status}**\n{reason}")
        return True

    async def _step(self, ch: dict) -> bool:
        """推进一步；True=链已终结。提前下单会使 legs 出现「未结算在前、已下单在后」。"""
        legs, n = ch["legs"], len(ch["legs"])
        if n == 0:
            return await self._place(ch, ch["windows"][0], ch["base_amount"])
        idx = next((i for i, l in enumerate(legs) if l["result"] is None), None)
        if idx is not None:
            return await self._resolve(ch, idx)   # 结算最靠前的未结算腿
        last = legs[-1]
        if n == len(ch["windows"]):
            return await self._finish(ch["id"], "DONE", f"全部 {n} 腿获胜")
        if last["result"] != "WIN":
            return await self._finish(ch["id"], "STOPPED", f"第{n}腿 {last['result']}，链终止")
        return await self._place(ch, ch["windows"][n], last["next_stake"])

    async def _place(self, ch: dict, ws: int, stake: float) -> bool:
        cid, legs = ch["id"], ch["legs"]
        if ws + STEP_MS - self._now_ms() < MIN_REMAIN_MS:
            return await self._finish(cid, "STOPPED", f"第{len(legs)+1}腿窗口剩余不足 15s，错过")
        ver = chain_version(cid)
        order = await self._existing_order(ver, ws)  # 重启认领，防重复下单
        if order is None:
            order = await asyncio.shield(self._trader.execute_signal_trade(
                prediction=ch["direction"], amount_usdt=stake, signal_version=ver,
                window_start=ws, max_exec_price=ch["max_exec_price"],
                market_period="5m", scan_budget=160))
            if self._on_balance_change:
                self._on_balance_change()
        if order is None:
            return await self._finish(cid, "STOPPED", f"第{len(legs)+1}腿下单未执行（Key/钱包/槽位/护栏）")
        qj = order.get("quote_json")
        if isinstance(qj, str):
            qj = json.loads(qj)
        qj = qj if isinstance(qj, dict) else {}
        try:
            filled = int(order.get("amount_in") or 0) / 1e18
            shares = float(qj.get("filledShareQty") or 0)
        except (TypeError, ValueError):
            filled, shares = 0.0, 0.0
        status = order.get("status")
        leg = {"ws": ws, "stake": stake, "order_id": order.get("id"), "status": status,
               "token_id": order.get("token_id") or "", "filled": filled,
               "avg_price": qj.get("averagePrice"), "shares": shares or None,
               "result": None, "pnl": None, "next_stake": None}
        if status != "FILLED" or not shares:
            leg["result"] = "FAILED" if status == "FAILED" else "UNCONFIRMED"
            await self._save(cid, legs=legs + [leg])
            return await self._finish(
                cid, "STOPPED",
                f"第{len(legs)+1}腿 {status}: {order.get('error_message') or '成交未确认/缺股数'}")
        await self._save(cid, legs=legs + [leg])
        self._notify(f"**复利链 #{cid} 第{len(legs)+1}腿已成交**\n"
                     f"{ch['direction']} {filled:.4f}U @{leg['avg_price']} → 到手 {shares:.4f} 股")
        return False

    async def _resolve(self, ch: dict, idx: int) -> bool:
        """结算 legs[idx]：先盯提前下单条件，再在收盘后按 K 线判输赢。"""
        cid, legs = ch["id"], list(ch["legs"])
        leg = dict(legs[idx])
        ws = leg["ws"]
        end = ws + STEP_MS
        if idx + 1 < len(ch["windows"]) and not leg.get("next_placed"):
            while self._now_ms() < end:
                remain = end - self._now_ms()
                up, dn = self._live_quote()
                hit = early_trigger(ch["direction"], up, dn, self._mid_price(),
                                    await self._window_open(ws), remain)
                if hit:
                    leg["next_placed"] = True
                    leg["early"] = hit
                    legs[idx] = leg
                    ch["legs"] = legs          # _place 读 ch["legs"]，须带上本次标记
                    await self._save(cid, legs=legs)
                    logger.info("复利链提前下单 | id={} | 第{}腿 | 条件={}", cid, idx + 2, hit)
                    # 上一腿尚未结算：按「本金+报价算得的利润」＝预估到手股数下（已下订单无法事后改额）
                    return await self._place(ch, ch["windows"][idx + 1],
                                             next_stake(leg["shares"]) if leg.get("shares")
                                             else (leg["next_stake"] or leg["stake"]))
                await self._sleep(POLL_S)
        await self._sleep_until(end + RESOLVE_DELAY_MS)
        o = c = 0.0
        for _ in range(15):
            o = await self._collector.fetch_kline_open("5m", ws)
            c = await self._collector.fetch_kline_close("5m", ws)
            if o > 0 and c > 0:
                break
            await self._sleep(2)
        else:
            return await self._finish(cid, "STOPPED", f"第{idx+1}腿 K 线结算源不可用，待结算器处理")
        res = decide(ch["direction"], o, c)
        leg["result"] = res
        if res == "WIN":
            leg["pnl"], leg["next_stake"] = leg["shares"] - leg["filled"], next_stake(leg["shares"])
            legs[idx] = leg
            await self._save(cid, legs=legs)
            self._redeem_bg(leg)  # 不阻塞：确认获胜即进入下一腿，奖金后台赎回
            return False
        leg["pnl"] = -leg["filled"] if res == "LOSS" else 0.0
        legs[idx] = leg
        await self._save(cid, legs=legs)
        if res == "LOSS":
            tail = "（已提前押上的下一腿仍会结算）" if leg.get("next_placed") else ""
            return await self._finish(cid, "DONE", f"第{idx+1}腿输，亏 {leg['filled']:.4f}U{tail}")
        return await self._finish(cid, "STOPPED", f"第{idx+1}腿平局(NOISE)，方向无法判定")

    async def _window_open(self, ws: int) -> float | None:
        """本窗开盘价（K 线，缓存；与结算同源）。"""
        if ws in self._open_cache:
            return self._open_cache[ws]
        try:
            v = await self._collector.fetch_kline_open("5m", ws)
        except Exception:
            return None
        if v and v > 0:
            self._open_cache[ws] = v
            return v
        return None

    async def _sleep_until(self, ms: int) -> None:
        while True:
            d = ms - self._now_ms()
            if d <= 0:
                return
            await self._sleep(min(d / 1000, POLL_S))

    async def _redeem_bg(self, leg: dict) -> None:
        t = asyncio.create_task(self._redeem(leg, self._now_ms() + 600_000))
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)

    async def _redeem(self, leg: dict, deadline_ms: int) -> bool:
        tok = leg.get("token_id")
        if not tok:
            return False
        while True:
            if await self._trader.redeem_tokens([tok]) is not None:
                await self._mark_redeemed(leg["order_id"])
                if self._on_balance_change:
                    self._on_balance_change()
                return True
            if self._now_ms() >= deadline_ms:
                logger.warning("复利链赎回超时 | token={} | {}", tok, self._trader.last_api_error)
                return False
            await self._sleep(REDEEM_RETRY_S)
