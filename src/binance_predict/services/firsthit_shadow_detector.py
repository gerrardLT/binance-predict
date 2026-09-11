"""5m DOWN 首触反转影子检测器（firsthit_down 族，2026-09-07）。

信号定义（冻结自 scripts/local_shape_scan_v2.py 方法论修正版，规则冻结勿动）：
    触发：5m 窗内 DOWN token 报价**首次**进入 (0.005, 0.1] 的采样点
          → 以该报价买 DOWN（押注 DOWN 最终结算获胜 =「反转」）。
    特征（触发时刻可见，严格 ex-ante，只读 ≤触发时刻的采样）：
        chg_bps = (btc@触 − 开盘)/开盘×1e4     开盘 = entry_price 优先，回退 btc 曲线首点
        body_r  = |btc@触 − 开盘| / (路径 max−min)   路径 = 开盘 + t≤触发 的全部 btc 采样
        wick01  = 路径高点 > max(btc@触, 开盘) ? 1 : 0（soft，供 G6/G7 事后重构）
        rng_bps = (路径 max−min)/开盘×1e4      （soft）
        npts    = t≤触发 的 btc 采样点数；npts < 8 → 整窗不落表（v2 主分析口径，
                  触发前路径太稀疏的特征不可信）
    十 version 同表隔离（G1/G3/G4/G7 族均 ⊂ G0；全特征落库，交叉门可事后重构）：
        firsthit_down_v1       G0 基底（全部首触，对照组）
        firsthit_down_body_v1  G1：body_r ≤ 0.35（FDR q=0.007，logit β=+1.28 p=0.000）
        firsthit_down_chg_v1   G3：chg_bps ≤ +2.82（logit β=+0.07 p=0.000）
        firsthit_down_g4_v1    G4：G1 ∩ G3（FDR q=0.063，confirm STRICT_PASS）
                               ⚠ G4 ⊂ G1 且 G4 ⊂ G3 —— 三者非独立，同窗全中即对同一
                                 DOWN 事件重复下注；实盘暴露须按「事件」而非「通道」核算。
                               ⚠ 前向功效偏紧：calib EV(+1.03) 与 4 周前向 CI 半宽(≈1.08)
                                 接近 → 即使真实效应等于 calib 估计，通过概率也仅 ≈50%；
                                 FAIL 时应延长观察期而非直接否决（见证据文档 D3）。
        firsthit_down_g7_v1    G7：G1 ∧ wick01=1（FDR q=0.000，全表最强门）
        g7_* 五个变体          G7 + streak/upper_wick/q/td_sec 收紧（见 FIRSTHIT_SPECS）

    EV（本检测器的生命线，逐事件真实触发价，禁用任何均值/pct 代理）：
        win  → 0.98/q − 1     （费 2% 无溢价，与 absorption/quote_edge._ev_at_entry 同源）
        lose → −1.0

数据流（归档后处理，同 absorption_shadow_detector / quote_edge_detector）：
    1. 启动预热：分块载入 trailing 14 天已归档 SentimentWindow，只推进水位不落表
       （历史样本由离线研究覆盖，影子只从部署点前向攒样本——与吸收影子同取舍）；
    2. 每 60s 轮询新归档窗（end_time > 水位，升序 limit 12）；
    3. 逐窗：算首触特征 → 结算可判（outcome UP/DOWN）→ 逐 version 评估门 →
       shadow_gate 拦截 + (version, window_start) 幂等 → 落 SETTLED（无 PENDING）。

影子纪律：只记录不下注，物理隔离于下单路径（本表不被任何下单代码引用，
不进 X4_VERSIONS/LIVE_CHANNELS）。前向裁决标准（预注册，4 周）：
    G1 通过 = 前向 P ≥ 12% 且 EV 日聚类 CI 下界 > 0；
    G3 通过 = 前向 P ≥ 10% 且 EV 日聚类 CI 下界 > 0；
    整体否决 = G0 前向 EV 日聚类 CI 上界 < 0（DOWN 侧 edge 消失，全部重来）。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy import func as sa_func
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import FirstHitShadowSignal, SentimentWindow
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（local_shape_scan_v2.py / G7 系列扫描）----
FIRSTHIT_SPECS: list[tuple[str, str]] = [
    ("firsthit_down_v1", "G0 基底（全部首触）"),
    ("firsthit_down_body_v1", "G1 小实体 body_r≤0.35"),
    ("firsthit_down_chg_v1", "G3 价格偏离 chg≤+2.82bp"),
    # G4 interaction（shape_scan_v2 已 FDR q=0.063 + confirm STRICT_PASS）：
    # chg_bps ≤ 2.82 ∧ body_r ≤ 0.35，样本量偏小但 EV 极高；注册影子版本用于前向验证
    ("firsthit_down_g4_v1", "G4 交互门 chg≤2.82∧body≤0.35"),
    ("firsthit_down_g7_v1", "G7 纯组合基底 body_r≤0.35∧wick=1"),
    ("g7_streak_v1", "G7+非强连阳 streak_up≤1"),
    ("g7_wick20_v1", "G7+长上影 upper_wick≥2.0bp"),
    ("g7_strict_v1", "G7严格版 streak_up≤1∧upper_wick≥1.5bp"),
    ("g7_q05_v1", "G7+深折价 q≤0.05"),
    ("g7_t270_v1", "G7+非极晚 t≤270s"),
]
Q_LO, Q_HI = 0.005, 0.1      # 首触报价区间 (0.005, 0.1]
BODY_R_GATE = 0.35           # G1 门：路径归一实体 ≤ 0.35
CHG_BPS_GATE = 2.82          # G3 门：BTC 相对开盘涨幅 ≤ +2.82 bp
MIN_PTS = 8                  # 路径质量门：触发前 btc 采样点数 ≥ 8（v2 主分析口径）

FEE_RET = 0.98               # EV = 赢 0.98/q−1 / 输 −1（费 2%，无溢价，回测口径）
POLL_INTERVAL = 60.0         # 轮询间隔（秒）
BACKSCAN_WINDOWS = 12        # 主循环单次拉取的新归档窗数（1 小时）
WARMUP_DAYS = 14             # 预热水位回看跨度（只推进水位不落表）
WARMUP_MS = WARMUP_DAYS * 86_400_000
WARMUP_BATCH = 500           # 预热分块载入批大小（限制 JSONB 曲线峰值内存）


def _ev_at_entry(win: bool, price: float) -> float:
    """单注 EV（回测口径，无溢价）：赢 0.98/q−1 / 输 −1。逐事件真实触发价。"""
    return (FEE_RET / price - 1.0) if win else -1.0


def _outcome_of(w: SentimentWindow) -> str | None:
    """可判定的结算方向：actual_return None/0 → None（与 absorption/quote_edge 同口径）。"""
    ret = w.actual_return
    if ret is None or float(ret) == 0.0:
        return None
    return w.outcome if (w.outcome or "") in ("UP", "DOWN") else None


def _ser(curve: list | None) -> list[dict]:
    """清洗 + 按 t 升序：只保留 t/v 均非空的采样点（同 absorption._ser 口径）。"""
    return sorted(
        [p for p in (curve or []) if p.get("t") is not None and p.get("v") is not None],
        key=lambda p: int(p["t"]),
    )


def extract_firsthit_features(
    window_start: int,
    btc_open: float | None,
    down_curve: list | None,
    btc_curve: list | None,
    *,
    max_trigger_ts: int | None = None,
) -> dict | None:
    """首触特征纯函数：归档影子与实时实盘共用，防止两套公式漂移。

    返回 None 的情形：无首触 / 开盘基准缺失 / 触发时刻无 btc / npts < MIN_PTS。
    max_trigger_ts 用于实时重放：先看 ≤max_trigger_ts 的历史找首触，再基于该最大时间戳计算 npts（允许窗口内后期成熟）。
    
    Key fix for early-touch pinning bug:
    - First touch detection is pinned to the earliest qualifying sample (correct semantics).
    - But npts counts ALL btc samples ≤ max_trigger_ts, NOT ≤ trigger_ts. This allows windows 
      with early first touches (e.g., at t=30s) to mature and fire when path has ≥8 BTC samples 
      by t=150s+ instead of being permanently suppressed.
    """
    start = int(window_start)
    dn_p = _ser(down_curve)
    btc = _ser(btc_curve)

    # 首触：升序采样中第一个 down_price ∈ (Q_LO, Q_HI] 的点
    trig = None
    for p in dn_p:
        ts = int(p["t"])
        if max_trigger_ts is not None and ts > int(max_trigger_ts):
            break
        v = float(p["v"])
        if Q_LO < v <= Q_HI:
            trig = p
            break
    if trig is None:
        return None
    q = float(trig["v"])
    trigger_ts = int(trig["t"])

    # 开盘 BTC 基准：调用方优先提供 entry_price，缺失时回退 btc 曲线首点
    bo = float(btc_open) if (btc_open is not None and float(btc_open) > 0) else (
        float(btc[0]["v"]) if btc else None)
    if not bo or bo <= 0:
        return None

    # KEY FIX: 触发前路径使用 max_trigger_ts（当前采样时刻的最新历史），而非仅≤trigger_ts
    # 这允许早期首触窗口在 t≥105s 后 npts 达标时正常开火，而不是永久被扼杀。
    pre = [float(p["v"]) for p in btc if int(p["t"]) <= int(max_trigger_ts)] if max_trigger_ts else [float(p["v"]) for p in btc if int(p["t"]) <= trigger_ts]
    npts = len(pre)
    if npts < MIN_PTS:
        return None                                    # 路径质量门（v2 主分析口径）
    if pre[-1] <= 0:
        return None
    pts = [bo] + pre
    btc_trig = pre[-1]                                 # 触发时刻 btc = ≤max_trigger_ts 的最后一点

    # 特征（与 local_shape_scan_v2.py / comprehensive_firsthit_backtest.py 同式）
    chg_bps = (btc_trig / bo - 1.0) * 1e4
    cum_hi, cum_lo = max(pts), min(pts)
    rng = cum_hi - cum_lo
    if rng > 1e-9:
        body_r = abs(btc_trig - bo) / rng
        wick01 = 1.0 if cum_hi > max(btc_trig, bo) + 1e-9 else 0.0
        upper_wick_bps = (cum_hi - max(btc_trig, bo)) / bo * 1e4
        rng_bps = rng / bo * 1e4
    else:
        body_r, wick01, upper_wick_bps, rng_bps = 0.0, 0.0, 0.0, 0.0

    return dict(
        q=q, trigger_ts=trigger_ts, td_sec=int((trigger_ts - start) / 1000),
        chg_bps=chg_bps, body_r=body_r, wick01=wick01, upper_wick_bps=upper_wick_bps,
        rng_bps=rng_bps, npts=npts, dvol=None, dpar=None,
    )


def _extract(w: SentimentWindow) -> dict | None:
    """抽取单窗首触特征（严格 ex-ante，只读 ≤触发时刻采样）。

    返回 None 的情形：无首触 / 开盘基准缺失 / 触发时刻无 btc / npts < MIN_PTS。
    样本集定义与 local_shape_scan_v2.py 主分析（npts≥8 子集）完全一致。
    """
    btc = _ser(getattr(w, "curve_btc_price", None))
    ep = getattr(w, "entry_price", None)
    btc_open = float(ep) if (ep is not None and float(ep) > 0) else (
        float(btc[0]["v"]) if btc else None)
    ext = extract_firsthit_features(
        int(w.start_time), btc_open,
        getattr(w, "curve_down_price", None), btc)
    if ext is None:
        return None

    # Δvol / Δpar（soft 记录维度，不作门）
    vol = _ser(getattr(w, "curve_trade_volume", None))
    par = _ser(getattr(w, "curve_participants", None))
    trigger_ts = int(ext["trigger_ts"])

    def _at_le(pts_: list[dict], ts: int) -> float | None:
        best = None
        for p in pts_:
            if int(p["t"]) <= ts:
                best = float(p["v"])
            else:
                break
        return best

    vol_o = float(vol[0]["v"]) if vol else None
    vol_d = _at_le(vol, trigger_ts)
    par_o = float(par[0]["v"]) if par else None
    par_d = _at_le(par, trigger_ts)
    ext["dvol"] = (vol_d - vol_o) if (vol_o is not None and vol_d is not None) else None
    ext["dpar"] = (par_d - par_o) if (par_o is not None and par_d is not None) else None
    return ext


def _gate_of(version: str, ext: dict, streak_up: int | None = None) -> bool:
    """version → 落表门（G0 恒真；G1 body；G3 chg；G4=G1∩G3；G7 及各变体）。

    特征缺失（None）一律不落表——宁可少样本，不用不可信特征污染前向裁决。
    """
    if version == "firsthit_down_v1":
        return True
    if version == "firsthit_down_body_v1":
        return ext["body_r"] is not None and ext["body_r"] <= BODY_R_GATE
    if version == "firsthit_down_chg_v1":
        return ext["chg_bps"] is not None and ext["chg_bps"] <= CHG_BPS_GATE
    if version == "firsthit_down_g4_v1":
        # G4 = G1 ∩ G3（chg_bps ≤ 2.82 ∧ body_r ≤ 0.35）。
        # 注意：G4 ⊂ G1 且 G4 ⊂ G3，与两者非独立——同窗三门全中即同一注的 3 倍暴露。
        return (
            ext["chg_bps"] is not None and ext["chg_bps"] <= CHG_BPS_GATE
            and ext["body_r"] is not None and ext["body_r"] <= BODY_R_GATE
        )

    # G7 基准条件：小实体 + 上影拒绝
    is_g7 = (
        ext["body_r"] is not None and ext["body_r"] <= BODY_R_GATE
        and ext.get("wick01") == 1.0
    )
    if not is_g7:
        return False

    if version == "firsthit_down_g7_v1":
        return True
    if version == "g7_streak_v1":
        # 非强连阳：streak_up <= 1（streak_up 为 None 时若调用方未提供则不放行）
        return streak_up is not None and streak_up <= 1
    if version == "g7_wick20_v1":
        return ext.get("upper_wick_bps") is not None and ext["upper_wick_bps"] >= 2.0
    if version == "g7_strict_v1":
        return (
            streak_up is not None and streak_up <= 1
            and ext.get("upper_wick_bps") is not None and ext["upper_wick_bps"] >= 1.5
        )
    if version == "g7_q05_v1":
        return ext.get("q") is not None and ext["q"] <= 0.05
    if version == "g7_t270_v1":
        return ext.get("td_sec") is not None and ext["td_sec"] <= 270

    return False


class FirstHitShadowDetector:
    """5m DOWN 首触反转影子检测器：轮询新归档窗 → 首触特征 → 逐 version 门 → 落 SETTLED。

    归档后处理（同 absorption/quote_edge）：窗已结算才处理，插行即 SETTLED，无 PENDING。
    """

    def __init__(self, collector=None) -> None:
        self._collector = collector  # 兼容启动装配签名；本检测器全部数据取自 ORM 曲线，未使用
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_window_end: int | None = None       # 已处理过的最大窗口 end_time（水位）
        self._trigger_counts: dict[str, int] = {v: 0 for v, _ in FIRSTHIT_SPECS}

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
            logger.warning("首触影子：预热失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="firsthit_shadow_detector")
        logger.info(
            "首触反转影子检测器启动 | {} | 门: G1 body_r≤{} G3 chg≤{:+.2f}bp | 路径质量 npts≥{}"
            " | EV=0.98/q−1（费 2%，逐事件真实触发价，影子只记录不下注）",
            {v: n for v, n in FIRSTHIT_SPECS}, BODY_R_GATE, CHG_BPS_GATE, MIN_PTS,
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
        logger.info("首触影子检测器已停止 | 触发 {}", self._trigger_counts)

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
                logger.warning("首触影子：循环异常 | {} | {}", type(exc).__name__, exc)
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
                logger.warning("首触影子：窗口处理失败 | window {} | {}",
                               getattr(w, "start_time", None), exc)
            self._last_window_end = max(self._last_window_end or 0, int(w.end_time))

    # ------------------------------------------------------------------
    # 预热：分块载入 trailing 14 天已归档窗，只推进水位不落表
    # ------------------------------------------------------------------

    async def _warmup(self) -> None:
        async with async_session_factory() as session:
            latest_end = (await session.execute(
                sa_select(sa_func.max(SentimentWindow.end_time))
            )).scalar_one_or_none()
        if latest_end is None:
            logger.warning("首触影子：无归档窗口，跳过预热（水位 0，循环内自愈）")
            return
        floor = int(latest_end) - WARMUP_MS
        last_end = floor
        total = 0
        while True:
            async with async_session_factory() as session:
                batch = (await session.execute(
                    sa_select(SentimentWindow.end_time)
                    .where(SentimentWindow.end_time > last_end)
                    .order_by(SentimentWindow.end_time.asc())
                    .limit(WARMUP_BATCH)
                )).scalars().all()
            if not batch:
                break
            total += len(batch)
            last_end = max(int(b) for b in batch)
            if len(batch) < WARMUP_BATCH:
                break
        # 水位推进到最新：主循环只处理新归档窗（历史样本由离线研究覆盖，
        # 影子从部署点前向攒样本——与吸收影子同取舍）
        self._last_window_end = last_end
        logger.info(
            "首触影子：预热完成，扫描 {} 窗（trailing {}d）推进水位到 {}（预热窗不落表，"
            "历史样本由 scripts/local_shape_scan_v2.py 离线研究覆盖）",
            total, WARMUP_DAYS, datetime.fromtimestamp(last_end / 1000, tz=timezone.utc)
            .strftime("%m-%d %H:%M"),
        )

    # ------------------------------------------------------------------
    # 核心：单窗口处理（首触特征 → 结算 → 三 version 门 → 落 SETTLED）
    # ------------------------------------------------------------------

    async def _process_window(self, w: SentimentWindow) -> None:
        outcome = _outcome_of(w)
        if outcome is None:
            return                                     # NOISE/缺结算：胜负不可判，不产生信号
        start_ms, end_ms = int(w.start_time), int(w.end_time)
        ext = _extract(w)
        if ext is None:
            return                                     # 无首触/路径质量不足：整窗不落表

        win = outcome == "DOWN"                        # 买 DOWN：窗结算 DOWN 即赢
        q = ext["q"]
        ev = _ev_at_entry(win, q)                      # 逐事件真实触发价，禁均值/pct 代理

        # 计算前驱 5m 连续阳线数（用于 g7_streak_v1 / g7_strict_v1）
        streak_up = 0
        try:
            L_ms = 300_000
            async with async_session_factory() as session:
                prev_wins = (await session.execute(
                    sa_select(SentimentWindow.actual_return)
                    .where(
                        SentimentWindow.start_time >= start_ms - 4 * L_ms,
                        SentimentWindow.start_time < start_ms,
                    )
                    .order_by(SentimentWindow.start_time.desc())
                )).scalars().all()
                for ret_val in prev_wins:
                    if ret_val is not None and float(ret_val) > 0:
                        streak_up += 1
                    else:
                        break
        except Exception as exc:
            logger.warning("首触影子：查询前驱连阳失败 | window {} | {}", start_ms, exc)
            streak_up = None

        async with async_session_factory() as session:
            # per-version 独立 commit：单 version 落表失败不回滚、不影响其他 version。
            for version, _name in FIRSTHIT_SPECS:
                try:
                    if not _gate_of(version, ext, streak_up=streak_up):
                        continue                        # 门未命中：不落表
                    if not shadow_gate.is_enabled(version):
                        continue                        # 手动下线：停止采集该版本（历史保留）
                    dup = await session.execute(
                        sa_select(FirstHitShadowSignal.id).where(
                            FirstHitShadowSignal.version == version,
                            FirstHitShadowSignal.window_start == start_ms,
                        )
                    )
                    if dup.first() is not None:
                        continue                        # 幂等：(version, window_start) 已存在
                    session.add(FirstHitShadowSignal(
                        version=version,
                        window_start=start_ms,
                        window_end=end_ms,
                        trigger_ts=ext["trigger_ts"],
                        td_sec=ext["td_sec"],
                        entry_down_price=q,
                        chg_bps=ext["chg_bps"],
                        body_r=ext["body_r"],
                        wick01=ext["wick01"],
                        rng_bps=ext["rng_bps"],
                        npts=ext["npts"],
                        dvol=ext["dvol"],
                        dpar=ext["dpar"],
                        settle_outcome=outcome,
                        win=win,
                        ev_at_entry=ev,
                        status="SETTLED",
                    ))
                    await session.commit()
                    self._trigger_counts[version] = self._trigger_counts.get(version, 0) + 1
                    logger.info(
                        "首触影子触发+结算 | {} | 窗口 {} | t={}s q={:.3f} chg={:+.2f}bp"
                        " body_r={:.3f} → 买DOWN | win={} ev={:+.3f} | npts={}",
                        version,
                        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
                        .strftime("%m-%d %H:%M"),
                        ext["td_sec"], q, ext["chg_bps"], ext["body_r"], win, ev, ext["npts"],
                    )
                except Exception as exc:
                    await session.rollback()
                    logger.warning("首触影子：version {} 落表失败 | window {} | {}",
                                   version, start_ms, exc)

    # ------------------------------------------------------------------
    # 状态（status API 用）
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "versions": {v: shadow_gate.is_enabled(v) for v, _ in FIRSTHIT_SPECS},
            "triggers": dict(self._trigger_counts),
            "last_window_end": self._last_window_end,
            "gates": {"q_range": [Q_LO, Q_HI], "body_r": BODY_R_GATE,
                      "chg_bps": CHG_BPS_GATE, "min_pts": MIN_PTS},
        }
