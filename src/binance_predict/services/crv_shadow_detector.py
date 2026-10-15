"""CRV 情绪反转影子检测器（2026-10-04 研究 run 20261004T083106Z-candle-reversal-certainty）。

信号定义（与预注册研究 src/features.py 的冻结网格逐位一致，禁止手抄第二套口径）：
    crv_hi_brk20_15m_v1（15m 高位阳线∧突破20高 → 押次根收阴 DOWN）：
        阳线 ∧ close > prior20高 ∧ pos100 > 0.90
        holdout n=212 胜率 63.7%（Wilson95%下界 57.0%），费后 EV(0.51买入+2%费)
        +0.117/单位；13/13 月为正；同族最强（唯一 30/90/180D 全 >61%）。
    crv_brk20_15m_v1（15m 突破20高阳线 → DOWN）：
        阳线 ∧ close > prior20高
        holdout n=458 胜率 58.5%（LB 54.0%），EV +0.065；360D 累计EV +120.3 单位注金。
    crv_brk50_hi_15m_v1（15m 突破50高∧pos100>0.75 → DOWN）：
        阳线 ∧ close > prior50高 ∧ pos100 > 0.75
        holdout n=226 胜率 60.6%（LB 54.1%），EV +0.086。
    crv_brk20_5m_v1（5m 突破20高阳线 → DOWN，用户点名部署）：
        阳线 ∧ close > prior20高
        holdout n=1552 胜率 54.5%（LB 52.0%，5m 唯一强档），EV +0.025（薄，护栏只放行
        便宜 DOWN）。
    crv_dnex_15m_v1（15m 三因子探索级，仅影子不注册实盘）：跌破20低∧贴20低∧实体>ATR
        disc n=132 61.4% / val n=108 60.2%；未过盲测（探索性，晋级需前向影子 ≥8 周）。
    crv_os_st4_5m_v1（5m 三因子探索级，仅影子）：RSI14<30∧连4阴∧跌破20低
        disc n=330 60.6% / val n=200 52.5%；未过盲测。
    crv_hi_z3_15m_v1（15m 三因子观察级，2026-10-04 AMENDMENT-2 全枚举新发现，仅影子）：
        阳线∧pos100>0.90∧z20>+2∧恰连3阳 → 押次根收阴 DOWN
        disc n=190 62.1%（Wilson95%下界 55.0%，全场深度3最高）/ val n=74 67.6%（n<100
        未达冻结晋级门槛）；180/90/30D 58.7%/56.8%/56.3% 不衰减。晋级由前向影子裁决。
    crv_dnbrk24_15m_v1 / crv_dnbrk24_5m_v1（2026-10-12 复核研究 run
        20261012T084631Z-kline-regime-transition 的「A 信号下破半」，确认级）：
        阴线吞没（当前实体 > 1.5×前实体 ∧ 方向反转）∧ 收盘跌破前 24 根低 → 押次根收阳 UP。
        15m 样本外(验证+盲测+前向 288d) n=249 胜率 61.9%（发现 59.2%/验证 60.6%/
        盲测 63.4%/前向 62.5%），相对「仅破24低无吞没」+6.3pp；护栏 0.606（61.9%×0.98）。
        5m 样本外 n=888 胜率 55.7%（盲测段 52.2% 偏弱），相对对照仅 +1.1pp；
        护栏 0.546 刻意只放行便宜 UP。两版本均默认实盘 OFF。
    crv_dnvol20_15m_v1 / crv_bodybrk20_5m_v1（2026-10-15 反转研究 run
        20261015T061535Z-reversal-study 的修正后回顾性验证，两版本均默认实盘 OFF）：
        crv_dnvol20_15m_v1：5m 阴线 ∧ 收盘跌破前 20 根低 ∧ 成交量 ≥1.5×前 20 根均量，
        且该 5m 根正好是 15m 窗口最后一根（open_time%15m==10min）→ 押下一根 **15m**
        收阳 UP。信号在 5m 求值，市场/结算周期为 15m（spec.target_tf）；非边界命中
        的 5m 事件不触发（它们的目标是跨窗的滚动 15 分钟，无对应合约）。
        对齐边界 n=2904 胜率 59.1%（训练 59.8%/验证 57.6%/末段 59.2%）。
        crv_bodybrk20_5m_v1：5m 阳线 ∧ 实体/振幅 ≥0.75 ∧ 收盘突破前 20 根高 → 押次根 5m
        收阴 DOWN。n=8649 胜率 55.2%（训练 54.5%/验证 54.8%/末段 56.5%）；事件集是
        crv_brk20_5m_v1 的严格子集（同方向同目标窗），已并入其 5m 互斥组。
        ⚠ 末段 360D 在筛选时已被看过，所有数字是回顾性，不是独立盲测，影子前向样本
        才是裁决依据。
    kdc_s1_96hi / kdc_s2_ext / kdc_s3_3up / kdc_s4_3dn（2026-10-16 注册，研究 run
        20261015T153500Z-kline-directional-certainty——720d 预注册三段隔离 + 盲测一次
        + PREDEXON 报价层 EV 验证 + 衰退归因，四版本均默认实盘 OFF，含执行层优化
        1-4：点差护栏 / 高波减仓 / 入场价带记录 / 双时点报价，见 live_channels 与
        multi_live_trader 注释）：
        kdc_s1_96hi_15m_v1：收在最高20%（(c−l)/range≥0.8）∧ 收盘破前96根高 → 押次根
        收阴 DOWN。盲测 W=0.657（n=134）；报价段(07-01→10-15, 10U五档VWAP) W=0.640、
        EV +0.096/份——但分月看全部来自7-8月，9月归零、10月转负（归因见研究
        ATTRIBUTION_REPORT）；护栏 0.64=0.657×0.98。
        kdc_s2_ext_15m_v1：收盘超24根高≥1×ATR20 ∧ 6根累计涨幅≥2×ATR20 → 押DOWN。
        盲测 W=0.630；报价段 EV +0.034；护栏 0.61。
        kdc_s3_3up_15m_v1：收在最高20% ∧ 3连阳 ∧ 破前20根高 → 押DOWN。盲测 W=0.628；
        报价段 EV +0.069；护栏 0.61。
        kdc_s4_3dn_15m_v1：收盘破前20根低 ∧ 3连阴 ∧ 大实体阴(实体/振幅≥0.75) → 押
        次根收阳 UP。盲测 W=0.603；报价段 EV +0.083（CI 跨 0）；护栏 0.59。
        ⚠ ATR20 口径=TR（含前收）近20根均值（含当前根）；6根动量=Σ(c[j]−c[j−1]) j=i−5..i
        （含当前差分）；连阳/连阴在 K 线断点重置（线上保守适配，研究数据零缺口）。
        ⚠ 影子/实盘一致性：cross_cost（up+down−1，两侧展示价）与 vol_tier 在信号落表
        时刻由同一缓存快照计算，影子 feature_snapshot 与实盘 payload 数值同源；t1 报价
        为目标窗开盘后首次轮询（0~60s，非研究回测的 +3.5s 对齐口径），q10_* 为开盘+10s
        定时二次快照（±15s 缓存粒度）——两档口径差异在研究报告与通道注释中显式声明。

口径保真（影子阶段的生命线）：
    特征公式逐位移植研究 features.py（trailing 滚动窗、prior 高低点显式 shift、
    Wilder RSI、实体方向 streak），NaN 一律保守不触发；测试
    tests/test_crv_shadow_detector.py 用 pandas 参照逐位对照尾部掩码。
    实时适配（研究数据零缺口、线上有缺口）：streak 在 K 线断点重置（保守方向）。

采集与执行解耦：影子信号照常落表 kline_shadow_signals（version+timeframe 双隔离，
与 KREV/反转/nextbar 共表）；四个确认级版本注册同名实盘通道（默认 OFF），
仅正常轮询中目标根开盘后 90 秒内的新鲜命中经钩子交给 MultiLiveTrader
（on_kline_reversal_signal → kline_reversal 族）；冷启动回补不追单。
结算口径与回测 rev1 一致：押 DOWN → 次根收阴赢；押 UP → 次根收阳赢；平盘 NOISE/EXPIRED。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import numpy as np
from loguru import logger
from sqlalchemy import select as sa_select

from binance_predict.db.engine import async_session_factory
from binance_predict.db.models import KlineShadowSignal
from binance_predict.discovery.data import Klines
from binance_predict.services.shadow_entry_quote import snapshot_entry_quote
from binance_predict.services.shadow_version_gate import shadow_gate

# ---- 冻结口径（研究 run 20261004T083106Z FROZEN_CANDIDATES.json，勿动）----
# 可选键 target_tf：市场/结算周期（缺省 = tf）。tf 是求值用的 K 线周期，target_tf 是
# 押注的预测市场与结算 K 线周期（crv_dnvol20_15m_v1：5m 求值、押下一根 15m）。
CRV_SHADOW_SPECS: list[dict] = [
    {
        "version": "crv_hi_brk20_15m_v1",
        "tf": "15m",
        "discovery_id": "crv20261004_hi20",
        "kind": "hi_brk20",
        "direction": "DOWN",           # 高位突破阳线 → 次根收阴
        "condition_text": "up & break20_up & pos100>0.90 → 次根收阴DOWN",
    },
    {
        "version": "crv_brk20_15m_v1",
        "tf": "15m",
        "discovery_id": "crv20261004_b20",
        "kind": "brk20",
        "direction": "DOWN",
        "condition_text": "up & break20_up → 次根收阴DOWN",
    },
    {
        "version": "crv_brk50_hi_15m_v1",
        "tf": "15m",
        "discovery_id": "crv20261004_b50hi",
        "kind": "brk50_hi",
        "direction": "DOWN",
        "condition_text": "up & break50_up & pos100>0.75 → 次根收阴DOWN",
    },
    {
        "version": "crv_dnex_15m_v1",
        "tf": "15m",
        "discovery_id": "crv20261004_dnex3",
        "kind": "dnex3",
        "direction": "UP",             # 三因子探索级：跌破20低∧贴20低∧实体>ATR → 次根收阳
        "condition_text": "down & break20_dn & near_lo20 & body>ATR → 次根收阳UP（探索级，仅影子）",
    },
    {
        "version": "crv_brk20_5m_v1",
        "tf": "5m",
        "discovery_id": "crv20261004_b20_5m",
        "kind": "brk20",
        "direction": "DOWN",
        "condition_text": "up & break20_up(5m) → 次根收阴DOWN",
    },
    {
        "version": "crv_os_st4_5m_v1",
        "tf": "5m",
        "discovery_id": "crv20261004_osst4",
        "kind": "os_st4",
        "direction": "UP",             # 三因子探索级：RSI<30∧连4阴∧跌破20低 → 次根收阳
        "condition_text": "down & rsi14<30 & streak==4 & break20_dn(5m) → 次根收阳UP（探索级，仅影子）",
    },
    {
        "version": "crv_hi_z3_15m_v1",
        "tf": "15m",
        "discovery_id": "crv20261004_hiz3",
        "kind": "hi_z3",
        "direction": "DOWN",           # 三因子观察级：高位∧z20>+2∧恰连3阳 → 次根收阴
        "condition_text": "up & pos100>0.90 & z20>+2 & streak==3 → 次根收阴DOWN（观察级，仅影子）",
    },
    {
        "version": "crv_dnbrk24_15m_v1",
        "tf": "15m",
        "discovery_id": "dnbrk20261012_15m",
        "kind": "dnbrk24",
        "direction": "UP",             # 阴线吞没∧跌破24低 → 次根收阳 UP（2026-10-12 复核）
        "condition_text": "dn & engulf(实体>1.5×前实体) & close<prior24低 → 次根收阳UP",
    },
    {
        "version": "crv_dnbrk24_5m_v1",
        "tf": "5m",
        "discovery_id": "dnbrk20261012_5m",
        "kind": "dnbrk24",
        "direction": "UP",
        "condition_text": "dn & engulf(实体>1.5×前实体) & close<prior24低(5m) → 次根收阳UP",
    },
    {
        "version": "crv_dnvol20_15m_v1",
        "tf": "5m",
        "target_tf": "15m",            # 5m 求值、押下一根 15m（仅 15m 边界收盘的 5m 根）
        "discovery_id": "dnvol20261015_15m",
        "kind": "dnvol20",
        "direction": "UP",
        "condition_text": "5m dn & close<prior20低 & 量>=1.5×前20均量 & 15m边界收盘 → 次根15m收阳UP",
    },
    {
        "version": "crv_bodybrk20_5m_v1",
        "tf": "5m",
        "discovery_id": "bodybrk20261015_5m",
        "kind": "bodybrk20",
        "direction": "DOWN",
        "condition_text": "5m up & 实体/振幅>=0.75 & close>prior20高 → 次根5m收阴DOWN",
    },
    # KDC 方向确定性四版本（2026-10-16，研究 run 20261015T153500Z；kdc 标记启用
    # 执行层优化快照：cross_cost / vol_tier / entry_band / q10 双时点报价）
    {
        "version": "kdc_s1_96hi_15m_v1",
        "tf": "15m",
        "discovery_id": "kdc20261015_s1",
        "kind": "kdc_96hi",
        "direction": "DOWN",           # 收在最高20%∧破96根高 → 次根收阴
        "condition_text": "(c-l)/range>=0.8 & close>prior96高 → 次根收阴DOWN",
        "kdc": True,
    },
    {
        "version": "kdc_s2_ext_15m_v1",
        "tf": "15m",
        "discovery_id": "kdc20261015_s2",
        "kind": "kdc_ext",
        "direction": "DOWN",           # 超24高≥1ATR ∧ 6根冲顶≥2ATR → 次根收阴
        "condition_text": "(close-prior24高)/ATR20>=1.0 & Σ6Δc>=2*ATR20 → 次根收阴DOWN",
        "kdc": True,
    },
    {
        "version": "kdc_s3_3up_15m_v1",
        "tf": "15m",
        "discovery_id": "kdc20261015_s3",
        "kind": "kdc_3up",
        "direction": "DOWN",           # 收最高20% ∧ 3连阳 ∧ 破20高 → 次根收阴
        "condition_text": "(c-l)/range>=0.8 & 3连阳 & close>prior20高 → 次根收阴DOWN",
        "kdc": True,
    },
    {
        "version": "kdc_s4_3dn_15m_v1",
        "tf": "15m",
        "discovery_id": "kdc20261015_s4",
        "kind": "kdc_3dn",
        "direction": "UP",             # 破20低 ∧ 3连阴 ∧ 大实体阴 → 次根收阳
        "condition_text": "close<prior20低 & 3连阴 & 实体/振幅>=0.75 → 次根收阳UP",
        "kdc": True,
    },
]
CRV_VERSIONS = [s["version"] for s in CRV_SHADOW_SPECS]
# KDC 四版本（方向确定性研究；落表时附带执行层优化快照）
KDC_VERSIONS = [s["version"] for s in CRV_SHADOW_SPECS if s.get("kdc")]
# 波动分层（优化项2）：ATR20 在近 500 根（含当前）的分位 ≥0.66 判高波（实盘金额×
# HIGHVOL_AMOUNT_FACTOR）；<0.33 低波。分界与 EV 验证三分位一致（研究 656 窗盘口：
# 低/中/高 EV +0.028/+0.103/+0.042，高波减半仓的证据口径）。
KDC_ATR_PCT_HIGH = 0.66
KDC_ATR_PCT_LOW = 0.33
# 入场价带（优化项3，只记录不拦截）：与 EV 验证的分带一致（<0.52 带 EV +0.143、
# ≥0.58 带 −0.033 的观察口径），供影子前向按带复核。
KDC_ENTRY_BANDS = ((0.52, "<0.52"), (0.55, "[0.52,0.55)"), (0.58, "[0.55,0.58)"))
# 按「结算周期」分组（settle/expire 用目标周期的 K 线）；求值按 spec["tf"]
CRV_VERSIONS_BY_TF: dict[str, list[str]] = {
    tf: [s["version"] for s in CRV_SHADOW_SPECS if s.get("target_tf", s["tf"]) == tf]
    for tf in ("15m", "5m")
}

BAR_MS = {"15m": 900_000, "5m": 300_000}
POLL_INTERVAL = 60.0            # 轮询间隔（秒）
# KDC 波动分层需 ATR20 近 500 根分位（含当前）→ 500 + 96 高点窗 + 12 回补余量。
# （旧值 130 仅覆盖 pos100；扩窗不改变既有版本特征值——滚动窗参数固定，数组加长
# 只影响可供特征求值的深度。）
WARMUP_BARS = 560
BACKSCAN_BARS = 12              # 冷启动/追赶回补根数
PENDING_EXPIRE_MS = 4 * 3_600_000  # 目标根起点后 4h 仍未结算 → EXPIRED
CRV_LIVE_MAX_LAG_MS = 90_000    # 目标根开盘后 90s 内的新鲜命中才派实盘


# ---------------------------------------------------------------- 滚动特征（研究口径移植）
def roll_max(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 最大（含当前根；前 w-1 根 NaN；窗内含 NaN → NaN，对齐 pandas 默认）。"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        out[i] = np.nan if np.isnan(win).any() else float(win.max())
    return out


def roll_min(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        out[i] = np.nan if np.isnan(win).any() else float(win.min())
    return out


def roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        out[i] = np.nan if np.isnan(win).any() else float(win.mean())
    return out


def roll_std_ddof1(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 样本标准差（ddof=1；与 pandas rolling(w).std(ddof=1) 同口径）。"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        if np.isnan(win).any():
            continue
        d = win - win.mean()
        out[i] = float(np.sqrt((d * d).sum() / (w - 1)))
    return out


def roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    """trailing 求和（含当前根；窗内含 NaN → NaN，对齐 pandas 默认）。"""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(w - 1, n):
        win = x[i - w + 1:i + 1]
        out[i] = np.nan if np.isnan(win).any() else float(win.sum())
    return out


def _shift1(x: np.ndarray) -> np.ndarray:
    """shift(1)：x[i-1]，首根 NaN（对齐 pandas shift 语义）。"""
    out = np.full(len(x), np.nan)
    if len(x) > 1:
        out[1:] = x[:-1]
    return out


def compute_crv_features(kl: Klines) -> dict:
    """CRV 判定所需特征（公式逐位移植研究 features.py，审计锚点见模块 docstring）。"""
    o, h, l, c = kl.o, kl.h, kl.l, kl.c
    up = c > o
    dn = c < o
    body = np.abs(c - o)
    # prior 高低点：rolling(w).max/min().shift(1) —— 不含当前根
    prior_hi20 = _shift1(roll_max(h, 20))
    prior_lo20 = _shift1(roll_min(l, 20))
    prior_hi50 = _shift1(roll_max(h, 50))
    # KDC：24/96 根高（kdc_s2 超幅与 kdc_s1 破 24h 高用）
    prior_hi24 = _shift1(roll_max(h, 24))
    prior_hi96 = _shift1(roll_max(h, 96))
    # prior_lo24：下破吞没族（crv_dnbrk24_*）用，收盘跌破前 24 根低
    prior_lo24 = _shift1(roll_min(l, 24))
    # engulf：阴线吞没（2026-10-12 复核口径）——当前根与前一实体方向相反，且当前
    # 实体 > 1.5×前一实体。前一根缺失（首根/断点）→ 不触发。
    prev_body = _shift1(body)
    prev_sign = _shift1(np.sign(c - o))
    with np.errstate(invalid="ignore"):
        engulf = ((np.sign(c - o) != prev_sign) & (np.sign(c - o) != 0)
                  & (body > 1.5 * prev_body))
    engulf[np.isnan(prev_body)] = False
    # pos100：(close − roll_min(l,100)) / (roll_max(h,100) − roll_min(l,100) + 1e-12)
    lo100, hi100 = roll_min(l, 100), roll_max(h, 100)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos100 = (c - lo100) / (hi100 - lo100 + 1e-12)
    # z20：(close − roll_mean(c,20)) / roll_std(c,20,ddof=1).clip(1e-12)
    sma20 = roll_mean(c, 20)
    std20 = roll_std_ddof1(c, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        z20 = (c - sma20) / np.maximum(std20, 1e-12)
    # ATR14：TR 的 14 根 SMA（TR[0] 因前收缺失为 NaN，与 pandas 口径一致）
    pc = _shift1(c)
    tr = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - pc[i]), abs(l[i] - pc[i]))
    atr = roll_mean(tr, 14)
    # KDC：ATR20（TR 近 20 根均值，含当前根——研究 engine.atr20 逐位口径）
    atr20 = roll_mean(tr, 20)
    # KDC：6 根收盘动量 Σ(c[j]−c[j−1]) j=i−5..i（含当前差分；前收缺失 → NaN 保守）
    dc = np.full(len(c), np.nan)
    dc[1:] = c[1:] - c[:-1]
    mom6 = roll_sum(dc, 6)
    # KDC：收盘位置 (c−l)/(h−l)（零振幅 → NaN 保守不触发）
    with np.errstate(invalid="ignore", divide="ignore"):
        close_pos = np.where((h - l) > 0, (c - l) / np.where((h - l) > 0, h - l, 1.0), np.nan)
    # RSI14（Wilder：ewm alpha=1/14 adjust=False，首根 delta NaN 跳过）
    delta = np.full(len(c), np.nan)
    delta[1:] = c[1:] - c[:-1]
    gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0.0, None))
    loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0.0, None))
    ag = al = np.nan
    rsi = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if np.isnan(ag):
            ag, al = gain[i], loss[i]
        else:
            ag += (gain[i] - ag) / 14.0
            al += (loss[i] - al) / 14.0
        rsi[i] = 100.0 - 100.0 / (1.0 + ag / max(al, 1e-12))
    # 实体方向 streak（连续同向实体根数，含当前根；与研究 features.py 逐位一致：
    # 首根恒 0（延续未知），后续同向 +1 / 换向重起 1 / 零实体归 0；断点重置为线上
    # 保守适配——研究数据零缺口。130 根窗口对「窗口前起始的超长同向段」存在截断
    # 误差，但 streak==4 判定只受 >118 根同向段影响（实际不可能出现）。
    sign = np.sign(c - o)
    streak = np.zeros(len(c), dtype=np.int64)
    run_sign, run_len = 0, 0
    for i in range(len(c)):
        if i > 0 and not kl.cont[i]:
            run_sign, run_len = 0, 0
        s = int(sign[i])
        if s == 0:
            streak[i] = 0
            run_sign, run_len = 0, 0
        elif run_sign == s:
            run_len += 1
            streak[i] = run_len
        else:
            run_sign = s
            run_len = 1 if i > 0 else 0
            streak[i] = run_len
    with np.errstate(invalid="ignore"):
        near_lo20 = np.abs(c - prior_lo20) < 0.25 * atr
    # vol_ratio：当前根成交量 / 前 20 根均量（不含当前根；基础币量，与研究 base_volume_high 同口径）
    # body_ratio：实体 / 振幅（零振幅 → NaN，保守不触发）
    prior_vol_mean20 = _shift1(roll_mean(kl.v, 20))
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio = kl.v / prior_vol_mean20
        span = h - l
        body_ratio = np.where(span > 0, body / np.where(span > 0, span, 1.0), np.nan)
    # b15_end：该根是否为 15m 窗口的最后一根 5m（open_time 距 15m 边界 10 分钟）。
    # 仅对 5m K 线有意义；15m/其它周期恒 False。
    b15_end = (kl.t % 900_000) == 600_000
    return {
        "close": c, "up": up, "dn": dn, "body": body,
        "prior_hi20": prior_hi20, "prior_lo20": prior_lo20, "prior_hi50": prior_hi50,
        "prior_lo24": prior_lo24, "engulf": engulf,
        "pos100": pos100, "z20": z20, "atr": atr, "rsi": rsi, "streak": streak,
        "near_lo20": near_lo20,
        "vol_ratio": vol_ratio, "body_ratio": body_ratio, "b15_end": b15_end,
        # KDC 特征（2026-10-16）
        "prior_hi24": prior_hi24, "prior_hi96": prior_hi96,
        "atr20": atr20, "mom6": mom6, "close_pos": close_pos,
    }


def _gt(a, b) -> np.ndarray:
    """NaN 安全比较：任一侧 NaN → False（保守不触发）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    out = a > b
    out[np.isnan(a) | np.isnan(b)] = False
    return out


def _ge(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    out = a >= b
    out[np.isnan(a) | np.isnan(b)] = False
    return out


def _lt(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    out = a < b
    out[np.isnan(a) | np.isnan(b)] = False
    return out


def _le(a, b) -> np.ndarray:
    """NaN 安全 ≤（kdc_s4 大实体阴 body_ratio≥0.75 含贴线）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    out = a <= b
    out[np.isnan(a) | np.isnan(b)] = False
    return out


def atr_pct_tail(atr20: np.ndarray, idx: int, w: int = 500) -> float | None:
    """ATR20[idx] 在近 w 根（含当前）中的分位；数据不足/NaN → None。

    与研究归因口径一致（inclusive rank）；仅对触发根求值（每轮 ≤12 根，O(w) 可忽略）。
    """
    if idx < w - 1:
        return None
    win = atr20[idx - w + 1: idx + 1]
    cur = atr20[idx]
    if np.isnan(cur) or np.isnan(win).any():
        return None
    return float((win <= cur).mean())


def spec_mask(spec: dict, feat: dict) -> np.ndarray:
    """按 spec.kind 构建逐根布尔命中掩码（冻结条件，NaN 保守不触发）。"""
    kind = spec["kind"]
    c = feat["close"]
    if kind == "brk20":
        return feat["up"] & _gt(c, feat["prior_hi20"])
    if kind == "hi_brk20":
        return feat["up"] & _gt(c, feat["prior_hi20"]) & _gt(feat["pos100"], 0.90)
    if kind == "brk50_hi":
        return feat["up"] & _gt(c, feat["prior_hi50"]) & _gt(feat["pos100"], 0.75)
    if kind == "dnex3":
        return (feat["dn"] & _lt(c, feat["prior_lo20"]) & feat["near_lo20"]
                & _gt(feat["body"], feat["atr"]))
    if kind == "os_st4":
        return (feat["dn"] & _lt(feat["rsi"], 30.0) & (feat["streak"] == 4)
                & _lt(c, feat["prior_lo20"]))
    if kind == "hi_z3":
        return (feat["up"] & _gt(feat["pos100"], 0.90) & _gt(feat["z20"], 2.0)
                & (feat["streak"] == 3))
    if kind == "dnbrk24":
        # 阴线吞没 ∧ 收盘跌破前 24 根低 → 押次根收阳 UP（2026-10-12 复核研究
        # run 20261012T084631Z 的「A 信号下破半」；15m 样本外 61.9% / 5m 55.7%）
        return feat["dn"] & feat["engulf"] & _lt(c, feat["prior_lo24"])
    if kind == "dnvol20":
        # 5m 阴线 ∧ 跌破前 20 低 ∧ 放量(≥1.5×前20均量) ∧ 15m 边界收盘 → 押下一根 15m 收阳 UP
        return (feat["dn"] & _lt(c, feat["prior_lo20"]) & _ge(feat["vol_ratio"], 1.5)
                & feat["b15_end"])
    if kind == "bodybrk20":
        # 5m 阳线 ∧ 实体/振幅≥0.75 ∧ 突破前 20 高 → 押次根 5m 收阴 DOWN
        return (feat["up"] & _ge(feat["body_ratio"], 0.75) & _gt(c, feat["prior_hi20"]))
    # ---- KDC 方向确定性四版本（2026-10-16；条件逐位移植研究 engine.py 冻结口径）----
    if kind == "kdc_96hi":
        # 收在最高20%（(c−l)/range≥0.8，含贴线；无方向要求）∧ 收盘破前 96 根高
        return _ge(feat["close_pos"], 0.8) & _gt(c, feat["prior_hi96"])
    if kind == "kdc_ext":
        # 收盘超前 24 根高 ≥1×ATR20 ∧ 6 根收盘动量 ≥2×ATR20（含贴线）
        with np.errstate(invalid="ignore", divide="ignore"):
            ext24 = (c - feat["prior_hi24"]) / feat["atr20"]
        return _ge(ext24, 1.0) & _ge(feat["mom6"] / feat["atr20"], 2.0)
    if kind == "kdc_3up":
        # 收在最高20% ∧ 至少 3 连阳（streak 断点重置为线上保守适配）∧ 破前 20 根高
        return (_ge(feat["close_pos"], 0.8) & feat["up"] & (feat["streak"] >= 3)
                & _gt(c, feat["prior_hi20"]))
    if kind == "kdc_3dn":
        # 收盘破前 20 根低 ∧ 至少 3 连阴 ∧ 大实体阴（实体/振幅≥0.75，含贴线）
        return (_lt(c, feat["prior_lo20"]) & feat["dn"] & (feat["streak"] >= 3)
                & _ge(feat["body_ratio"], 0.75))
    raise ValueError(f"未知 CRV spec.kind: {kind}")


def evaluate_crv(feat: dict, specs: list[dict], n_tail: int) -> list[dict]:
    """对末 n_tail 根逐条求值 CRV 条件（纯函数，供实时/回补/测试共用）。"""
    hits: list[dict] = []
    n = len(feat["close"])
    for spec in specs:
        mask = spec_mask(spec, feat)
        tail = mask[max(0, n - n_tail):]
        for off, hit in enumerate(tail):
            if bool(hit):
                hits.append({"spec": spec, "idx": n - len(tail) + off, "bar_offset": off})
    return hits


def _to_klines(rows: list[dict], bar_ms: int) -> Klines:
    """data_collector 的 K 线 dict 列表 → discovery.data.Klines（升序、已收盘）。"""
    t = np.asarray([r["open_time"] for r in rows], dtype=np.int64)
    kl = Klines(
        t=t,
        o=np.asarray([r["open"] for r in rows], dtype=np.float64),
        h=np.asarray([r["high"] for r in rows], dtype=np.float64),
        l=np.asarray([r["low"] for r in rows], dtype=np.float64),
        c=np.asarray([r["close"] for r in rows], dtype=np.float64),
        v=np.asarray([r["volume"] for r in rows], dtype=np.float64),
        cont=np.ones(len(rows), dtype=bool),
    )
    if len(t) > 1:
        kl.cont[1:] = (t[1:] - t[:-1]) == bar_ms
        kl.cont[0] = False
    return kl


class CrvShadowDetector:
    """CRV 情绪反转影子检测器：轮询 5m/15m 收盘 → 冻结条件求值/结算，影子落表；
    四个确认级版本同名实盘通道默认 OFF，新鲜命中经钩子派单。"""

    def __init__(self, collector, pm_15m_latest: dict, pm_5m_info: dict) -> None:
        self._collector = collector
        # 目标窗入场报价源（只读共享缓存）：15m 用 _pm_15m_latest、5m 用 _pm_market_info
        self._pm_by_tf: dict[str, dict] = {"15m": pm_15m_latest, "5m": pm_5m_info}
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_evaluated_bar: dict[str, int | None] = {"15m": None, "5m": None}
        self._trigger_count = 0
        self._settle_count = 0
        self._on_live_fire = None
        self._q10_tasks: set[asyncio.Task] = set()

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
            logger.warning("CRV 影子：冷启动回补失败（忽略，循环内自愈）| {}", exc)
        self._task = asyncio.create_task(self._loop(), name="crv_shadow_detector")
        logger.info(
            "CRV 情绪反转影子检测器启动 | 15m {} | 5m {} | 冻结条件实时求值"
            "（影子模式：只记录不下注；实盘通道默认 OFF）",
            "/".join(CRV_VERSIONS_BY_TF["15m"]), "/".join(CRV_VERSIONS_BY_TF["5m"]),
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
        for t in list(self._q10_tasks):  # 未到点的 +10s 报价任务一并取消
            t.cancel()
        logger.info("CRV 影子检测器已停止 | 触发 {} 结算 {}", self._trigger_count, self._settle_count)

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
                logger.warning("CRV 影子：循环异常 | {} | {}", type(exc).__name__, exc)
            try:
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                break

    async def _poll_once(self) -> None:
        for tf in ("15m", "5m"):
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                continue  # 拉取失败/不足，下轮重试
            last_start = int(closed[-1]["open_time"])
            if (self._last_evaluated_bar[tf] is None
                    or last_start > self._last_evaluated_bar[tf]):
                await self._evaluate_new_bars(tf, closed)
                self._last_evaluated_bar[tf] = last_start
            await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    # ------------------------------------------------------------------
    # 触发
    # ------------------------------------------------------------------

    async def _evaluate_new_bars(self, tf: str, closed: list[dict]) -> None:
        """评估 _last_evaluated_bar 之后的新收盘根（含冷启动回补的末 12 根）。"""
        bar_ms = BAR_MS[tf]
        kl = _to_klines(closed, bar_ms)
        feat = compute_crv_features(kl)
        specs = [s for s in CRV_SHADOW_SPECS if s["tf"] == tf]
        collect_live = self._last_evaluated_bar[tf] is not None
        if self._last_evaluated_bar[tf] is None:
            n_tail = BACKSCAN_BARS  # 冷启动：回补最近 12 根，不追真钱单
        else:
            starts = [int(r["open_time"]) for r in closed]
            try:
                first_new = next(
                    i for i, s in enumerate(starts) if s > self._last_evaluated_bar[tf])
            except StopIteration:
                return
            n_tail = len(starts) - first_new
        n_tail = min(n_tail, BACKSCAN_BARS)  # 长停机不追全史，最多回补 12 根
        hits = evaluate_crv(feat, specs, n_tail)
        if not hits:
            return
        live_payloads: list[dict] = []
        q10_pending: list[tuple[int, int, str]] = []  # (sig_id, target_bar_start, target_tf)
        async with async_session_factory() as session:
            added = 0
            last_bar = None
            for hit in hits:
                bar = closed[hit["idx"]]
                last_bar = bar
                payloads = live_payloads if collect_live else None
                sig_row = await self._record_signal(
                    tf, session, hit["spec"], bar, feat, hit["idx"], payloads)
                if sig_row is not None:
                    added += 1
                    if hit["spec"].get("kdc"):
                        q10_pending.append((int(sig_row.id), int(sig_row.target_bar_start),
                                            hit["spec"].get("target_tf", tf)))
            if added:
                await session.commit()
                self._trigger_count += added
                logger.info("CRV 影子触发 +{} | {} | 信号根 {}", added, tf, int(last_bar["open_time"]))
        # 优化项4：+10s 双时点报价（影子行 feature_snapshot.q10_*；实盘不下单于此档）
        for sig_id, target_start, target_tf in q10_pending:
            self._schedule_q10(sig_id, target_start, target_tf)
        self._dispatch_live(live_payloads)

    def _dispatch_live(self, payloads: list[dict]) -> None:
        from .shadow_forward_evidence import observe_kline
        for payload in payloads:
            observe_kline(payload)
        hook = self._on_live_fire
        if hook is None:
            return
        for payload in payloads:
            try:
                hook(payload)
            except Exception as exc:
                logger.warning("CRV：实盘开火分派异常（不影响影子采集）| {}", exc)

    async def _record_signal(self, tf: str, session, spec: dict, bar: dict, feat: dict,
                             idx: int, live_payloads: list[dict] | None = None):
        """幂等落 PENDING，并为正常轮询中的新鲜命中收集实盘 payload。返回落表行。"""
        sig_ms = BAR_MS[tf]
        target_tf = spec.get("target_tf", tf)  # 市场/结算周期（缺省同信号周期）
        target_ms = BAR_MS[target_tf]
        start_ms = int(bar["open_time"])
        exists = (await session.execute(
            sa_select(KlineShadowSignal.id).where(
                KlineShadowSignal.version == spec["version"],
                KlineShadowSignal.signal_bar_start == start_ms,
            )
        )).scalar_one_or_none()
        if exists is not None:
            return None
        target_bar_start = start_ms + sig_ms
        # 目标窗入场报价快照（窗口对齐+近开盘守卫；缺失/回补 → None，该笔 EV 不计）。
        # 在 gate 之前取一次，payload 与影子行共用同一份数值（影子/实盘同源的生命线）。
        up_q, down_q, q_ts = snapshot_entry_quote(self._pm_by_tf[target_tf], target_bar_start)
        # KDC 执行层优化快照（与实盘 payload 同源计算）
        kdc_snap: dict = {}
        if spec.get("kdc"):
            atr_pct = atr_pct_tail(feat["atr20"], idx)
            if atr_pct is not None:
                tier = ("high" if atr_pct >= KDC_ATR_PCT_HIGH
                        else "low" if atr_pct < KDC_ATR_PCT_LOW else "mid")
                kdc_snap["atr_pct"] = round(atr_pct, 4)
                kdc_snap["vol_tier"] = tier
            else:
                kdc_snap["vol_tier"] = "unknown"
            if up_q is not None and down_q is not None:
                kdc_snap["quote_cross_cost"] = round(up_q + down_q - 1.0, 6)
            bet_q = down_q if spec["direction"] == "DOWN" else up_q
            if bet_q is not None:
                band = ">=0.58"
                for edge, name in KDC_ENTRY_BANDS:
                    if bet_q < edge:
                        band = name
                        break
                kdc_snap["entry_band"] = band
        if live_payloads is not None and (
                0 <= int(time.time() * 1000) - target_bar_start <= CRV_LIVE_MAX_LAG_MS):
            payload = {
                "version": spec["version"],
                "market_start": target_bar_start,
                "market_end": target_bar_start + target_ms,
                "direction": spec["direction"],
                "signal_bar_start": start_ms,
            }
            if spec.get("kdc"):
                # 点差护栏/高波减仓用：与影子 feature_snapshot 同一份缓存快照导出
                payload["cross_cost"] = kdc_snap.get("quote_cross_cost")
                payload["vol_tier"] = kdc_snap.get("vol_tier")
            live_payloads.append(payload)
        if not shadow_gate.is_enabled(spec["version"]):
            return None  # 手动下线只停影子采集；实盘由 trader enabled 独立控制
        # 审计快照：条件涉及的实际特征值（供实时值与研究口径对照）
        snapshot: dict = {}
        streak_v = int(feat["streak"][idx])
        snapshot["streak"] = streak_v
        for name in ("pos100", "rsi", "z20", "vol_ratio", "body_ratio"):
            v = float(feat[name][idx])
            if not np.isnan(v):
                snapshot[name] = round(v, 6)
        if not np.isnan(feat["atr"][idx]) and feat["atr"][idx] > 0:
            snapshot["body_atr_ratio"] = round(float(feat["body"][idx] / feat["atr"][idx]), 6)
        if not np.isnan(feat["prior_lo20"][idx]):
            snapshot["dist_prior_lo20"] = round(
                float(feat["close"][idx] - feat["prior_lo20"][idx]), 6)
        # dnbrk24 审计字段：吞没实体比 与 距 24 低（供实时值与研究口径对照）
        prev_body_v = feat["body"][idx - 1] if idx > 0 else np.nan
        if not np.isnan(prev_body_v) and prev_body_v > 0:
            snapshot["body_prev_ratio"] = round(float(feat["body"][idx] / prev_body_v), 6)
        if not np.isnan(feat["prior_lo24"][idx]):
            snapshot["dist_prior_lo24"] = round(
                float(feat["close"][idx] - feat["prior_lo24"][idx]), 6)
        # KDC 快照字段（优化项1-3 的影子侧记录）
        if kdc_snap:
            snapshot.update(kdc_snap)
        row = KlineShadowSignal(
            version=spec["version"],
            discovery_id=spec["discovery_id"],
            condition_text=spec["condition_text"],
            timeframe=target_tf,  # 存市场/结算周期（与执行账本 market_period 一致）
            signal_bar_start=start_ms,
            signal_bar_end=target_bar_start,
            direction=spec["direction"],
            target_bar_start=target_bar_start,
            feature_snapshot=snapshot,
            entry_up_price=up_q,
            entry_down_price=down_q,
            entry_quote_ts=q_ts,
            status="PENDING",
        )
        session.add(row)
        await session.flush()  # 取 id 供 +10s 报价任务（避免 commit 过期后异步访问）
        return row

    # ------------------------------------------------------------------
    # KDC 优化项4：目标窗开盘 +10s 二次报价（影子行 feature_snapshot.q10_*）
    # ------------------------------------------------------------------

    def _schedule_q10(self, sig_id: int, target_bar_start: int, target_tf: str) -> None:
        now_ms = int(time.time() * 1000)
        fire_at = target_bar_start + 10_000
        if now_ms - fire_at > 60_000:
            return  # 轮询滞后超 1 分钟：缓存已进窗深处，放弃（影子行缺 q10_* 可审计）
        delay_s = max(0.0, (fire_at - now_ms) / 1000)
        task = asyncio.create_task(
            self._q10_snapshot(sig_id, delay_s, target_bar_start, target_tf),
            name=f"crv_q10_{sig_id}")
        self._q10_tasks.add(task)
        task.add_done_callback(self._q10_tasks.discard)

    async def _q10_snapshot(self, sig_id: int, delay_s: float,
                            target_bar_start: int, target_tf: str) -> None:
        """开盘 +10s 定时再取一次报价缓存（±15s 缓存粒度，offset 见 q10_ts 审计）。"""
        try:
            await asyncio.sleep(delay_s)
            up, down, ts = snapshot_entry_quote(
                self._pm_by_tf[target_tf], target_bar_start, max_offset_ms=30_000)
            if up is None:
                return  # 缓存缺失/错窗：不写字段（可审计）
            async with async_session_factory() as session:
                row = (await session.execute(
                    sa_select(KlineShadowSignal).where(KlineShadowSignal.id == sig_id)
                )).scalar_one_or_none()
                if row is None:
                    return
                snap = dict(row.feature_snapshot) if isinstance(row.feature_snapshot, dict) else {}
                snap["q10_up"] = up
                snap["q10_down"] = down
                snap["q10_ts"] = ts
                row.feature_snapshot = snap
                await session.commit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("CRV 影子：q10 二次报价失败（不影响主流程）| sig#{} | {}", sig_id, exc)

    # ------------------------------------------------------------------
    # 结算（按 direction 判 win：DOWN→次根收阴赢 / UP→次根收阳赢；只认本检测器 version）
    # ------------------------------------------------------------------

    async def _settle_pending(self, tf: str, closed: list[dict]) -> None:
        by_start = {int(r["open_time"]): r for r in closed}
        starts = sorted(by_start)
        if not starts:
            return
        async with async_session_factory() as session:
            pendings = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(CRV_VERSIONS_BY_TF[tf]),
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
                    sig.win = (up and sig.direction == "UP") or (not up and sig.direction == "DOWN")
                    sig.status = "SETTLED"
                sig.settle_open, sig.settle_close = o, c
                sig.settled_at = datetime.now(timezone.utc)
                self._settle_count += 1
                logger.info(
                    "CRV 影子结算 | {} | 信号根 {} | 次根 {} → {} | win={}",
                    sig.version, int(sig.signal_bar_start), int(sig.target_bar_start),
                    sig.settle_outcome, sig.win if sig.status == "SETTLED" else "N/A",
                )
            await session.commit()

    async def _expire_stale_pending(self, tf: str) -> None:
        """目标根起点后 4h 仍未结算（币安缺 K / 长时间拉取失败）→ EXPIRED。"""
        cutoff = int(time.time() * 1000) - BAR_MS[tf] - PENDING_EXPIRE_MS
        async with async_session_factory() as session:
            stale = (await session.execute(
                sa_select(KlineShadowSignal).where(
                    KlineShadowSignal.version.in_(CRV_VERSIONS_BY_TF[tf]),
                    KlineShadowSignal.status == "PENDING",
                    KlineShadowSignal.target_bar_start < cutoff,
                )
            )).scalars().all()
            if not stale:
                return
            for sig in stale:
                sig.status = "EXPIRED"
                logger.warning("CRV 影子：PENDING 超时转 EXPIRED | {} | 目标根 {}",
                               sig.version, int(sig.target_bar_start))
            await session.commit()

    # ------------------------------------------------------------------
    # 冷启动回补（幂等，唯一约束防重）
    # ------------------------------------------------------------------

    async def _backscan(self) -> None:
        for tf in ("15m", "5m"):
            closed = await self._collector.fetch_recent_klines(tf, WARMUP_BARS)
            if len(closed) < WARMUP_BARS:
                logger.warning("CRV 影子：冷启动回补 {} 数据不足（{} 根），跳过", tf, len(closed))
                continue
            await self._evaluate_new_bars(tf, closed)
            self._last_evaluated_bar[tf] = int(closed[-1]["open_time"])
            # 顺带结算停机期间已到期信号
            await self._settle_pending(tf, closed)
            await self._expire_stale_pending(tf)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_evaluated_bar": dict(self._last_evaluated_bar),
            "trigger_count": self._trigger_count,
            "settle_count": self._settle_count,
            "versions": list(CRV_VERSIONS),
            "live_channels_registered": True,
        }
