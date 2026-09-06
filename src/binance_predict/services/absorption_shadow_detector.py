"""吸收/欠反应跟随影子检测器（absorption_follow_v1 族，2026-09-04）。

信号定义（冻结自 .pytest_tmp/absorption_realprice.py 真实价复核，规则冻结勿动）：
    核心假设：正常时 UP token 报价随 BTC 以弹性 k 联动（"报价随 K 线变化的正常速率"）。
    当 5m 窗内 BTC 出现明显位移、而报价「欠跟随」（粘滞），是知情者顶住人群吸筹的脚印
    → 结算倾向朝 BTC 位移方向继续（follow 补涨：btc 涨押 UP / btc 跌押 DOWN）。

    窗开 TD 秒后（严格 ex-ante，只读 ≤TD 采样点）：
        btc_move = (btc@TD − open) / open × 1e4                        (bp)
        up_move  = (up_price@TD − up_price_open) × 100                 (pp，real 基）
    滚动 trailing 14 天缓冲（只用当前窗之前的窗，严格 ex-ante）标定：
        k, b     = np.polyfit(btc_move, up_move, 1)                    （弹性/截距）
        位移门    = pct_sorted(|btc_move|, 50)                           （p50，nearest-rank）
        欠反应门  = pct_sorted(under[过位移门子集], 80)                   （p80）
        under    = −sign(btc_move)·(up_move − (k·btc_move + b))         （>0=欠反应/粘滞）
    signal = |btc_move| ≥ 位移门 ∩ under ≥ 欠反应门 → follow 双向押注。

    双 variant（各维护独立缓冲 + 独立标定）：
        absorption_follow_td120_v1  TD=120s
        absorption_follow_td150_v1  TD=150s
    Δvol / Δpar 为 soft 记录维度（不作门，供前向分层分析）。

命门已关（真实价复核 2026-09-04）：裸 pct − 真实价乐观偏差 ±0.001（旧法 pct+溢价悲观
    −0.014~−0.019）；RECENT TD150 real EV +0.105 CI[+0.050,+0.165]✓、TD120 real +0.065
    CI[+0.006,+0.123]✓；边缘温和（非小样本高估的 +0.16~0.37）。k 强 regime 依赖
    （FULL 2.53~2.59 / RECENT 2.00~2.04 / 旧 dump 4.1）→ 滚动标定为必须（非可选）。

口径保真（影子阶段的生命线）：
    - pct_sorted 逐字复刻研究 nearest-rank 口径 round(p/100·(n−1)) 索引（非 np.percentile
      线性插值）——两者插值不同会让位移门/欠反应门漂移。
    - _at 逐字复刻 absorption_backtest._at_or_before（≤TD 最后一点，越界即 break，严格
      ex-ante）；曲线先 _ser 清洗排序（防御 JSONB 乱序，同 quote_edge 口径）。
    - 特征有效性过滤逐字复刻 realprice.features：btc_move≠0 ∩ up_pct/down_pct/up_price/
      down_price 齐全 ∩ q_pct>0 ∩ 0<q_real<1，样本集定义与已验证 EV 完全一致。
    - EV = 赢 0.98/q−1 / 输 −1（费 2% 无溢价，与 quote_edge._ev_at_entry 同源）。

数据流（归档后处理，同 quote_edge_detector）：
    1. 启动预热：分块载入 trailing 14 天已归档 SentimentWindow，填充两 variant 标定缓冲
       （只填缓冲不落表——预热窗自身无完整 14 天前置缓冲，落表会违背冻结口径）；
    2. 每 60s 轮询新归档窗（end_time > 水位，升序 limit 12）；
    3. 逐窗逐 variant：算特征 → 快照 calib（严格 s<本窗 start）→ 本窗入缓冲 →
       标定 k/b/门 → 评估 signal → 命中且结算可判 → 直接落 SETTLED（无 PENDING 阶段）；
    4. 幂等：(version, window_start) 唯一约束防重；shadow_gate.is_enabled 落表前拦截。

影子纪律：只记录不下注、不占风控配额、不注册 LIVE_CHANNELS、不进 X4_VERSIONS，本表
    （absorption_shadow_signals）不被任何下单代码引用（物理隔离）。无实盘通道 → 不推邮件。
    攒 2~3 周真实前向样本、核对触发分布与 SHADOW_BENCH 基准一致后，人工 promote 才谈上线。
    重启：预热重填缓冲 + 水位推进到最新，从部署点前向攒样本（宕机期窗只入缓冲不落表，
    前向样本轻微缺失属可接受口径，非正确性问题）。
"""
from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import AbsorptionShadowSignal, SentimentWindow
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（realprice.py 复核同源，勿动）----
# version -> 判定时点 TD（窗开后秒）；双 variant 各维护独立缓冲 + 独立标定。
ABSORPTION_SPECS: dict[str, int] = {
    "absorption_follow_td120_v1": 120,
    "absorption_follow_td150_v1": 150,
}
FEE_RET = 0.98               # EV = 赢 0.98/q−1 / 输 −1（费 2%，无溢价，回测口径）
DISP_PCT = 50                # 位移门 = p50(|btc_move|)
UNDER_PCT = 80               # 欠反应门 = p80(under)（过位移门子集）
POLL_INTERVAL = 60.0         # 轮询间隔（秒）
BACKSCAN_WINDOWS = 12        # 主循环单次拉取的新归档窗数（1 小时）
CALIB_BUFFER_DAYS = 14       # 滚动标定缓冲跨度（trailing ~14 天，用户冻结）
CALIB_BUFFER_MS = CALIB_BUFFER_DAYS * 86_400_000
MIN_CALIB = 500              # 触发前最小标定样本窗数（冷启动保护，稳态远高于此）
WARMUP_BATCH = 500           # 预热分块载入批大小（限制 JSONB 曲线峰值内存）


# ----------------------------------------------------------------------
# 冻结统计算子（逐字复刻研究口径，勿用 np.percentile 替换）
# ----------------------------------------------------------------------

def _pct_sorted(vals: list[float], p: float) -> float:
    """nearest-rank 分位（复刻 absorption_backtest.pct_sorted）：
    i = round(p/100·(n−1)) 取排序后第 i 个（非线性插值）。空集 → nan。"""
    if not vals:
        return float("nan")
    v = sorted(vals)
    i = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[i]


def _ev_at_entry(win: bool, price: float) -> float:
    """单注 EV（回测口径，无溢价）：赢 0.98/q−1 / 输 −1。"""
    return (FEE_RET / price - 1.0) if win else -1.0


def _outcome_of(w: SentimentWindow) -> str | None:
    """可判定的结算方向：actual_return None/0 → None（与 quote_edge 结算同口径）。"""
    ret = w.actual_return
    if ret is None or float(ret) == 0.0:
        return None
    return w.outcome if (w.outcome or "") in ("UP", "DOWN") else None


# ----------------------------------------------------------------------
# 曲线算子（ORM [{t,v}] 口径；先清洗排序防御 JSONB 乱序）
# ----------------------------------------------------------------------

def _ser(curve: list | None) -> list[dict]:
    """清洗 + 按 t 升序：只保留 t/v 均非空的采样点。"""
    return sorted(
        [p for p in (curve or []) if p.get("t") is not None and p.get("v") is not None],
        key=lambda p: int(p["t"]),
    )


def _first(pts: list[dict]) -> float | None:
    """窗口开盘值 = 首个有效采样点（pts 已升序清洗）。"""
    return float(pts[0]["v"]) if pts else None


def _at(pts: list[dict], start_ms: int, td_s: float) -> float | None:
    """决策点 TD(秒) 处的值 = ≤TD 的最后一点（严格 ex-ante，越界即 break）。
    逐字复刻 absorption_backtest._at_or_before。"""
    best = None
    for p in pts:
        if (int(p["t"]) - start_ms) / 1000.0 <= td_s:
            best = float(p["v"])
        else:
            break
    return best


def _at_pt(pts: list[dict], start_ms: int, td_s: float) -> tuple[float, int] | None:
    """同 _at，但一并返回采样时刻 (value, ts_ms)（入场报价快照审计用）。"""
    best = None
    for p in pts:
        if (int(p["t"]) - start_ms) / 1000.0 <= td_s:
            best = (float(p["v"]), int(p["t"]))
        else:
            break
    return best


def _extract(w: SentimentWindow, td_s: float) -> dict | None:
    """抽取单窗 TD 时刻特征（real 基）；数据不全/退化 → None（逐字复刻 realprice.features
    的有效性过滤，样本集定义与已验证 EV 一致）。"""
    start = int(w.start_time)
    btc = _ser(getattr(w, "curve_btc_price", None))
    up_p = _ser(getattr(w, "curve_up_price", None))
    dn_p = _ser(getattr(w, "curve_down_price", None))
    up_pct = _ser(getattr(w, "curve_up_pct", None))
    dn_pct = _ser(getattr(w, "curve_down_pct", None))

    # 开盘 BTC 基准：entry_price 优先，回退 curve_btc_price 首点（同 quote_edge 口径）
    ep = getattr(w, "entry_price", None)
    bo = float(ep) if (ep is not None and float(ep) > 0) else _first(btc)
    bd = _at(btc, start, td_s)
    if not bo or bd is None or bo <= 0:
        return None
    btc_move = (bd - bo) / bo * 1e4                    # bp
    if btc_move == 0:
        return None

    up_pct_o, up_pct_d = _first(up_pct), _at(up_pct, start, td_s)
    dn_pct_d = _at(dn_pct, start, td_s)
    up_pr_o = _first(up_p)
    up_pt, dn_pt = _at_pt(up_p, start, td_s), _at_pt(dn_p, start, td_s)
    if up_pt is None or dn_pt is None or up_pr_o is None:
        return None
    up_pr_d, up_ts = up_pt
    dn_pr_d, dn_ts = dn_pt
    if None in (up_pct_o, up_pct_d, dn_pct_d):
        return None

    bet = "UP" if btc_move > 0 else "DOWN"             # follow：顺 btc 补涨
    q_pct = (dn_pct_d if bet == "DOWN" else up_pct_d) / 100.0
    q_real = dn_pr_d if bet == "DOWN" else up_pr_d
    if q_pct <= 0 or q_real <= 0 or q_real >= 1:
        return None
    up_move = (up_pr_d - up_pr_o) * 100.0              # pp（real 基）

    # Δvol / Δpar（累积语义 → 反应窗增量；soft 记录维度，不作门）
    vol = _ser(getattr(w, "curve_trade_volume", None))
    par = _ser(getattr(w, "curve_participants", None))
    vol_o, vol_d = _first(vol), _at(vol, start, td_s)
    par_o, par_d = _first(par), _at(par, start, td_s)
    dvol = (vol_d - vol_o) if (vol_o is not None and vol_d is not None) else None
    dpar = (par_d - par_o) if (par_o is not None and par_d is not None) else None

    return dict(
        start=start, btc_move=btc_move, up_move=up_move, bet=bet, q_real=q_real,
        up_pr_d=up_pr_d, dn_pr_d=dn_pr_d, entry_ts=(dn_ts if bet == "DOWN" else up_ts),
        dvol=dvol, dpar=dpar,
    )


def _calibrate(calib: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """从标定缓冲 [(btc_move, up_move)] 拟合 k/b + 位移门/欠反应门（复刻 realprice.analyze
    的发现集标定片段）。样本不足/退化/非有限 → None（保守不落表）。"""
    if len(calib) < MIN_CALIB:
        return None
    X = np.array([bm for bm, _um in calib], dtype=float)
    Y = np.array([um for _bm, um in calib], dtype=float)
    if float(X.max()) == float(X.min()):
        return None                                    # btc_move 零方差 → polyfit 退化
    try:
        k, b = np.polyfit(X, Y, 1)
    except Exception:
        return None
    k, b = float(k), float(b)
    if not (math.isfinite(k) and math.isfinite(b)):
        return None
    disp_gate = _pct_sorted([abs(float(x)) for x in X], DISP_PCT)
    unders: list[float] = []
    for bm, um in calib:
        if abs(bm) >= disp_gate:
            resid = um - (k * bm + b)
            unders.append(-math.copysign(1.0, bm) * resid)
    if not unders:
        return None
    under_gate = _pct_sorted(unders, UNDER_PCT)
    if not (math.isfinite(disp_gate) and math.isfinite(under_gate)):
        return None
    return k, b, float(disp_gate), float(under_gate)


class AbsorptionShadowDetector:
    """吸收/欠反应跟随影子检测器：预热 14 天标定缓冲 → 轮询新归档窗 → 滚动标定评估
    → 命中直接落 SETTLED（归档后处理，无 PENDING 阶段）。

    双 variant（TD120/TD150）各维护独立 deque 缓冲 [(start_ms, btc_move, up_move)]。
    """

    def __init__(self, collector=None) -> None:
        self._collector = collector  # 兼容启动装配签名；本检测器全部数据取自 ORM 曲线，未使用
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_window_end: int | None = None       # 已处理过的最大窗口 end_time（水位）
        self._trigger_count = 0
        self._buffers: dict[str, deque] = {v: deque() for v in ABSORPTION_SPECS}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            await self._warmup()
        except Exception as exc:
            logger.warning("吸收影子：预热标定缓冲失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="absorption_shadow_detector")
        logger.info(
            "吸收/欠反应跟随影子检测器启动 | variant {} | 滚动标定 trailing {}d（min {}）"
            " | 位移门 p{} 欠反应门 p{} | EV=0.98/q−1（费 2%，影子只记录不下注）",
            {v: f"TD={td}s" for v, td in ABSORPTION_SPECS.items()},
            CALIB_BUFFER_DAYS, MIN_CALIB, DISP_PCT, UNDER_PCT,
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
        logger.info("吸收影子检测器已停止 | 触发（含前向）{}", self._trigger_count)

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
                logger.warning("吸收影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        stmt = (
            sa_select(SentimentWindow)
            .where(SentimentWindow.end_time > (self._last_window_end or 0))
            .order_by(SentimentWindow.end_time.asc())
            .limit(BACKSCAN_WINDOWS)
        )
        async with async_session_factory() as session:
            wins = (await session.execute(stmt)).scalars().all()
        for w in wins:
            try:
                await self._process_window(w)
            except Exception as exc:
                logger.warning("吸收影子：窗口处理失败 | window {} | {}",
                               getattr(w, "start_time", None), exc)
            self._last_window_end = max(self._last_window_end or 0, int(w.end_time))

    # ------------------------------------------------------------------
    # 预热：分块载入 trailing 14 天已归档窗，只填缓冲不落表
    # ------------------------------------------------------------------

    async def _warmup(self) -> None:
        async with async_session_factory() as session:
            latest_end = (await session.execute(
                sa_select(sa_func.max(SentimentWindow.end_time))
            )).scalar_one_or_none()
        if latest_end is None:
            logger.warning("吸收影子：无归档窗口，跳过预热（缓冲空 → MIN_CALIB 保护，循环内自愈）")
            return
        floor = int(latest_end) - CALIB_BUFFER_MS
        last_end, total = floor, 0
        while True:
            async with async_session_factory() as session:
                batch = (await session.execute(
                    sa_select(SentimentWindow)
                    .where(SentimentWindow.end_time > last_end)
                    .order_by(SentimentWindow.end_time.asc())
                    .limit(WARMUP_BATCH)
                )).scalars().all()
            if not batch:
                break
            for w in batch:
                self._absorb_into_buffers(w)
            total += len(batch)
            last_end = max(int(w.end_time) for w in batch)
            if len(batch) < WARMUP_BATCH:
                break
        # 水位推进到最新：主循环从此只处理新归档窗（预热窗只入缓冲不落表）
        self._last_window_end = last_end
        logger.info(
            "吸收影子：预热标定缓冲 {} 窗（trailing {}d，end_time>{}）| 缓冲 {}",
            total, CALIB_BUFFER_DAYS, floor,
            {v: len(buf) for v, buf in self._buffers.items()},
        )

    def _absorb_into_buffers(self, w: SentimentWindow) -> None:
        """把一个已归档窗的特征并入两 variant 标定缓冲（预热/不落表路径）。"""
        if _outcome_of(w) is None:
            return
        start = int(w.start_time)
        for version, td in ABSORPTION_SPECS.items():
            ext = _extract(w, float(td))
            if ext is None:
                continue
            self._buffers[version].append((start, ext["btc_move"], ext["up_move"]))

    # ------------------------------------------------------------------
    # 核心：单窗口处理（滚动标定 → 评估 signal → 落 SETTLED）
    # ------------------------------------------------------------------

    async def _process_window(self, w: SentimentWindow) -> None:
        outcome = _outcome_of(w)
        if outcome is None:
            return                                     # NOISE/缺结算：胜负不可判，不产生信号
        start_ms, end_ms = int(w.start_time), int(w.end_time)
        async with async_session_factory() as session:
            # per-variant 独立 commit：单 variant 落表失败不回滚、不影响另一 variant。
            for version, td in ABSORPTION_SPECS.items():
                try:
                    ext = _extract(w, float(td))
                    buf = self._buffers[version]
                    cutoff = start_ms - CALIB_BUFFER_MS
                    while buf and buf[0][0] < cutoff:   # 按龄逐出（升序处理 → popleft 单调）
                        buf.popleft()
                    # calib 严格 ex-ante：只取本窗之前、且在 trailing 14d 内的缓冲窗
                    calib = [(bm, um) for (s, bm, um) in buf if cutoff <= s < start_ms]
                    if ext is None:
                        continue                        # 特征不可算：不入缓冲、不评估
                    buf.append((start_ms, ext["btc_move"], ext["up_move"]))  # 本窗入缓冲供后续窗
                    fitted = _calibrate(calib)
                    if fitted is None:
                        continue                        # 缓冲不足/退化：保守不落表
                    k, b, disp_gate, under_gate = fitted
                    btc_move, up_move = ext["btc_move"], ext["up_move"]
                    if abs(btc_move) < disp_gate:
                        continue                        # 位移门未过
                    resid = up_move - (k * btc_move + b)
                    under = -math.copysign(1.0, btc_move) * resid
                    if under < under_gate:
                        continue                        # 欠反应门未过 → 非吸收信号
                    if not shadow_gate.is_enabled(version):
                        continue                        # 手动下线：停止采集该版本（历史保留）
                    dup = await session.execute(
                        sa_select(AbsorptionShadowSignal.id).where(
                            AbsorptionShadowSignal.version == version,
                            AbsorptionShadowSignal.window_start == start_ms,
                        )
                    )
                    if dup.first() is not None:
                        continue                        # 幂等：(version, window_start) 已存在
                    bet, q_real = ext["bet"], ext["q_real"]
                    win = outcome == bet
                    ev = _ev_at_entry(win, q_real)
                    session.add(AbsorptionShadowSignal(
                        version=version,
                        window_start=start_ms,
                        window_end=end_ms,
                        td=int(td),
                        direction=bet,
                        k=k, b=b, disp_gate=disp_gate, under_gate=under_gate,
                        calib_n=len(calib),
                        btc_move_bp=btc_move, up_move_pp=up_move,
                        resid_pp=resid, under_pp=under,
                        dvol=ext["dvol"], dpar=ext["dpar"],
                        entry_up_price=ext["up_pr_d"], entry_down_price=ext["dn_pr_d"],
                        entry_quote_ts=ext["entry_ts"], entry_quote_kind="real",
                        settle_outcome=outcome, win=win, ev_at_entry=ev,
                        status="SETTLED",
                    ))
                    await session.commit()
                    self._trigger_count += 1
                    logger.info(
                        "吸收影子触发+结算 | {} | 窗口 {} | btc_move={:+.1f}bp under={:+.2f}pp"
                        " (门{:+.2f}) → 押{} | win={} ev={:+.3f} | calib_n={}",
                        version,
                        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                        .strftime("%m-%d %H:%M"),
                        btc_move, under, under_gate, bet, win, ev, len(calib),
                    )
                except Exception as exc:
                    await session.rollback()
                    logger.warning("吸收影子：variant {} 落表失败 | window {} | {}",
                                   version, start_ms, exc)

    # ------------------------------------------------------------------
    # 状态（status API 用）
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """状态（status API 用）：trigger_count 计两 variant 落表行数合计。"""
        return {
            "running": self._running,
            "last_window_end": self._last_window_end,
            "trigger_count": self._trigger_count,
            "calib_buffer": {v: len(buf) for v, buf in self._buffers.items()},
        }
