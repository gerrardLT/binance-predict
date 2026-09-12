"""多通道实盘注册表（MultiLiveTrader 的静态描述层 + 配置解析，纯函数无 DB）。

通道 ID 与影子信号版本名对齐（订单 signal_version 直接用通道名，对账/统计天然一致）。
七族触发机制（由 MultiLiveTrader 分别驱动，见 multi_live_trader.py）：
- quote_edge：5m 采样循环喂价 → 窗内报价区间命中（v2 附加 BTC 门禁，实时喂价解锁；
  v3 环境门禁版再叠加前窗 DOWN/距日高回落，异步 DB 核验后下单）；
- x4：轮询 misalignment_signals PENDING → 次窗 +150s 决策点下单；
- scene：fake_breakout_detector fire 钩子 → 次周期开盘下单（15m 市场）；
- s2_cond：S2CondShadowDetector 实时判价钩子（实盘 S2 派生窗内 t=4/t=5 1m 收盘<周期开盘）
  → 当刻真单押 UP（15m 市场，次周期中段入场）；
- nextbar：NextbarShadowDetector 新根收盘钩子 → 次根 5m 市场开盘后 90s 内下单押 UP；
- absorption：5m 采样循环喂价内联判定（窗开快照 + TD 秒双快照，标定来自
  AbsorptionShadowDetector 滚动缓冲）→ 跟随 BTC 位移方向押注（动态 UP/DOWN，5m 市场）。
- firsthit：5m 采样循环按本窗完整历史重放 DOWN 首次进入 (0.005,0.1] 的触发点，
  复用 firsthit_shadow_detector 的特征/门纯函数 → 命中后买 DOWN（5m 市场）。

护栏数值依据：盈亏平衡入场价 entry* = wr×(1−FEE)（干净口径历史胜率）：
S1 wr64.4%→0.63 / S5 78.5%→0.77 / S2 53.6%→0.525 / S4 55.4%→0.54 /
x4_v1 41.2%→0.40 / x4_v2 45.3%→0.44，护栏设在平衡价附近或略下方。
S2 真实 UP 报价常在 0.79+（跌态无折扣），护栏 0.55 会保护性弃单——正确行为（EV 保护）。
2026-09-06 影子 promote 三族（护栏同口径）：s2_cond_t4 38.9%→0.38 / s2_cond_t5d
44.8%→0.44 / nb_smaslope_5m 47.43%→0.46 / absorption_td120 79.8%→0.78 /
absorption_td150 88.5%→0.86。
2026-09-06 x4_v3 趋势过滤版：下单层入场价白名单 [0,0.2)∪[0.3,0.4) 为主护栏
（超带弃单，研究冻结口径不进运行时覆盖项），0.50 兜底护栏与 x4_v2 相同——
保持 v2/v3 唯一差异只有拦截规则。
所有护栏可被 LIVE_CHANNELS_JSON 按通道覆盖。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from binance_predict.config.settings import settings

from .quote_edge_detector import QUOTE_EDGE_RULES

MAX_ORDER_AMOUNT_USDT = 50.0    # 单笔金额硬上限（配置误写拒绝启动，不靠自律）
MAX_DAILY_ORDERS_CAP = 500      # 日限硬上限（防配置误写 ×100 敞口失控）
_QE_GUARD_MARGIN = 0.03         # quote_edge 族自动护栏 = 入场区间上界 + 此余量（滑点容忍）


@dataclass(frozen=True)
class ChannelSpec:
    """通道静态描述（冻结，运行时不可变）。"""

    channel: str
    family: str                   # quote_edge | x4 | scene | s2_cond | nextbar | absorption | firsthit
    market_period: str            # 5m | 15m
    direction: str                # 典型下单方向（scene/s2_cond 族由信号 side 决定；
                                  # absorption 族由 BTC 位移符号动态决定，此处为基准方向）
    auto_max_exec: float          # 默认执行价护栏（可被配置覆盖）
    display_name: str
    order_type: str = "MARKET"    # 下单类型：'MARKET'（市价 FOK） | 'LIMIT'（限价挂单 GTC）
    v2_guard: str | None = None   # quote_edge v2 门禁模式：min_drop | max_rise | None
    v3_env: bool = False          # quote_edge v3 环境门禁（前窗DOWN [+距日高回落]，异步核验）
    regime_gate: bool = False     # quote_edge v4 regime 门禁（ret24≤阈值，K 线异步核验）
    streak_gate: bool = False     # quote_edge v3 非连涨门禁（末收15m，K 线异步核验）
    entry_band_whitelist: tuple[tuple[float, float], ...] | None = None
    hour_guard: tuple[int, int] | None = None   # quote_edge 时段门禁（北京时间 window_start hour∈[lo,hi)）
    ln_dd_guard: bool = False     # quote_edge 深夜距日高回落门（触发时点距当日高点回落≥0.30%，与 v3b 同口径；仅深夜版用）
    # 下单层入场价白名单（x4_v3）：决策点真实成交均价须落在任一 [lo,hi) 区间，
    # 否则 FAILED 弃单（prediction_trading 报价后检查，含 FOK 重试轮复检）。
    # None = 不启用（既有通道零影响）；研究冻结口径，不进 parse_channel_config 可覆盖项。


def _qe_guard(version: str) -> float:
    """quote_edge 族自动护栏：区间上界 + 0.03（与旧 QuoteEdgeLiveTrader 同推导）。"""
    return round(QUOTE_EDGE_RULES[version][3] + _QE_GUARD_MARGIN, 4)


LIVE_CHANNELS: dict[str, ChannelSpec] = {
    # --- quote_edge 族（5m 窗内报价触发，区间引用 QUOTE_EDGE_RULES 冻结口径）---
    # v2 门禁版：contrarian v1 冻结区间 + BTC 门禁（chg<+0.10%，阈值引用 V2_PRICE_GUARDS）
    # ⚠ 区间/护栏仍派生自已退役的 quote_contrarian_v1 冻结条目（QUOTE_EDGE_RULES
    # 是回测口径事实源，整体保留）；退役只停「v1 自身作为通道/影子版本」。
    "quote_contrarian_v2": ChannelSpec(
        "quote_contrarian_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·门禁版", order_type="LIMIT", v2_guard="max_rise",
    ),
    # late_night_contrarian_v2（2026-09-08 实盘接入）：v1 ∩ 时段门(北京 22-24) ∩ 距日高回落≥0.30%
    # 依据（F1 优化发现）：OOS n=50 wr 44.0% CI[31.2%,57.7%] vs 盈亏平衡≈27%；门禁数据缺失→不落表。
    # ⚠️ 纪律推翻：原 docstring "纯影子前向攒样本" 被用户拍板改为实盘注册（线上已开启下单）。
    # ⚠️ 不要给本通道加 v2_guard：门禁组合只有「时段门 + 距日高回落」，与影子
    #   late_night_contrarian_v2 落表口径一致（quote_edge_detector 只查 HOUR_GUARDS
    #   + LN_DD_GUARDS，不查 V2_PRICE_GUARDS）。曾误加 v2_guard="max_rise" 而
    #   V2_PRICE_GUARDS 未登记该版本 → 实盘查表 KeyError 被采样循环静默吞掉，
    #   影子 36 单 / 实盘 0 单（2026-09-10 归因修复）。
    "late_night_contrarian_v2": ChannelSpec(
        "late_night_contrarian_v2", "quote_edge", "5m", "DOWN", _qe_guard("late_night_contrarian_v2"),
        "深夜逆势·日高回落门禁版", order_type="LIMIT", hour_guard=(22, 24), ln_dd_guard=True,
    ),
    # --- x4 族（影子 PENDING → 次窗 +150s 决策点，入场价历史偏低）---
    "x4_v2": ChannelSpec(
        "x4_v2", "x4", "5m", "DOWN", 0.50, "情绪错位·平静市门禁版",
        # 下单层入场价白名单（2026-09-11 死区止血）：挖掉 [0.10,0.30) —— 实盘 n=32 pnl=-45.99 USDT
        # WR=9.4%（保本需 20.0%），其中 v2 占 28 笔 -42.22；保住 [0.30,0.40) —— 实盘 n=26 pnl=+26.60 WR=61.5%。
        # ⚠️ 生效前提：max_exec_price 必须 > 0.40（DB 覆盖层），否则 [0.35,0.40) 仍被护栏先弃——
        # 白名单与护栏相互独立（prediction_trading 两道都要过）。
        entry_band_whitelist=((0.0, 0.1), (0.3, 0.4)),
    ),
    # x4_v3：v2 门禁 + 双趋势门禁（检测层）+ 入场价白名单（下单层，主护栏上界 0.4）；
    # 0.50 为第二道防线（与 v2 相同，保持 v2/v3 唯一差异只有拦截规则）。实盘默认 OFF。
    "x4_v3": ChannelSpec(
        "x4_v3", "x4", "5m", "DOWN", 0.50, "情绪错位·趋势过滤版",
        entry_band_whitelist=((0.0, 0.1), (0.3, 0.4)),
    ),
    # --- 场景族（15m 市场次周期开盘入场；S5 为 +5min 确认入场）---
    # S1 护栏 0.70：2026-08-30 盘口数据校准（原 0.60 拍脑袋值）——
    # 41 个已结算信号中入场价 >=0.60 的 8 个胜率 75%（> 放行区间 61%），
    # 开盘 DOWN 贵=回落已在发生（确认而非警告），0.60 误拦正收益单。
    "scene_bull_exhaust": ChannelSpec(
        "scene_bull_exhaust", "scene", "15m", "DOWN", 0.70, "场景S1 多头耗尽（押DOWN）",
    ),
    "scene_bull_exhaust_confirm": ChannelSpec(
        "scene_bull_exhaust_confirm", "scene", "15m", "DOWN", 0.75, "场景S5 确认入场（押DOWN）",
    ),
    # S2 护栏 0.65：2026-08-30 盘口数据校准（与 S1 同病）——47% 信号被旧 0.55 拦截，
    # 被拦段 8 单胜率 88%、均值EV +0.39（与放行段同样赚钱）；开盘 UP 贵=反弹已启动。
    "scene_bear_exhaust": ChannelSpec(
        "scene_bear_exhaust", "scene", "15m", "UP", 0.65, "场景S2 空头耗尽（押UP）",
    ),
    "scene_momentum_fade": ChannelSpec(
        "scene_momentum_fade", "scene", "15m", "DOWN", 0.55, "场景S4 动量衰竭（押DOWN）",
    ),
    # S5 深档（2026-09-03 影子 → 实盘小金额前向验证）：S5 确认钩子 z5≤−20bp 子集，
    # 通道名与 fake_breakout_detector.S5_DEEP_VERSION 同名（订单 signal_version 与
    # 影子版本对账对齐）；+5min 确认即下单（15m 市场）。护栏 0.88：盈亏平衡入场价
    # = 回测 91.3%×0.98≈0.895 略下方；深档 EV 偏乐观含机械成分，小金额前向验证。
    # 与 S5 确认通道同窗互斥（SAME_WINDOW_EXCLUSIVE，至多一单成交）。
    "s5_deep_z20_v1": ChannelSpec(
        "s5_deep_z20_v1", "scene", "15m", "DOWN", 0.88, "S5深档·深回落门禁版",
    ),
    # --- s2_cond 族（2026-09-06 影子 promote）：S2CondShadowDetector 实时判价钩子，
    # 实盘 S2(bear_exhaust) 派生窗内 t=4/t=5 判 1m 收盘<周期开盘 → 当刻真单押 UP
    # （15m 市场次周期中段入场）。正 EV 来自低买入场价而非胜率（720d 冻结：t4 触发
    # 1069/2176=49.1% 价-only 胜率 38.9%；t5 剔深 643/2176=29.5% 胜率 44.8%；
    # 报价表研究 EV +0.237/+0.283 属乐观上界，真实 EV 前向现算）。
    # 护栏 = 价-only 胜率平衡价（wr×0.98）略下方；真实报价均值 ~0.30，护栏只拦贵单。
    # 两版同源同目标周期 → 同窗互斥（至多一单成交，弃单不占槽）。
    "s2_cond_t4_v1": ChannelSpec(
        "s2_cond_t4_v1", "s2_cond", "15m", "UP", 0.38, "S2条件t=4价<开（押UP）",
        order_type="LIMIT",
    ),
    "s2_cond_t5d_v1": ChannelSpec(
        "s2_cond_t5d_v1", "s2_cond", "15m", "UP", 0.44, "S2条件t=5剔深（押UP）",
    ),
    # --- nextbar 族（2026-09-06 影子 promote）：NextbarShadowDetector 新根收盘钩子，
    # 冻结条件 sma_slope_atr_5≥1.6606 → 次根 5m 市场押 UP（目标窗开盘后 90s 内）。
    # ⚠ 研究原文自评：720d 次根收阳仅 47.43%（长样本反指），edge 依赖 Jul-Aug regime，
    # record-only 科学仪器非背书——用户拍板小额前向验证。护栏 0.46 = 平衡价略下方，
    # 只在 UP 便宜（市场看衰而条件看涨）时成交，成交率低是保命设计非 bug。
    "nb_smaslope_5m_v1": ChannelSpec(
        "nb_smaslope_5m_v1", "nextbar", "5m", "UP", 0.46, "nextbar 5m动量误定价（押UP）",
    ),
    # --- absorption 族（2026-09-06 影子 promote）：采样循环喂价内联判定（窗开快照 +
    # TD 秒实时双快照），标定 k/b/位移门/欠反应门复用 AbsorptionShadowDetector 滚动
    # 缓冲（同一采样 feed 同基）。跟随 BTC 位移方向（动态 UP/DOWN，当前 5m 市场中段
    # 入场）。真实价复核：TD150 real EV +0.105 CI[+0.050,+0.165]、TD120 +0.065
    # CI[+0.006,+0.123]；价-only 胜率 79.8%/88.5% → 平衡价 0.782/0.867 略下方。
    # 两版同窗同假设 → 同窗互斥。
    "absorption_follow_td120_v1": ChannelSpec(
        "absorption_follow_td120_v1", "absorption", "5m", "UP", 0.78,
        "吸收跟随TD120欠反应→顺势",
    ),
    "absorption_follow_td150_v1": ChannelSpec(
        "absorption_follow_td150_v1", "absorption", "5m", "UP", 0.86,
        "吸收跟随TD150欠反应→顺势",
    ),
    # --- firsthit 族（2026-09-07 影子 promote）：5m 采样循环按本窗完整历史重放
    # DOWN 首次进入 (0.005,0.1] 的触发点，命中后买 DOWN。所有版本共用
    # firsthit_shadow_detector.extract_firsthit_features / _gate_of，保证实盘与
    # 影子口径同源。嵌套版本共用同一首触事件，统一互斥组限制同窗最多一笔 FILLED。
    # 护栏为保守成交上限，不代表前向胜率背书：G0 用研究价-only 胜率 8.9%×0.98≈0.087
    # 下方 0.08；G1/G3 的 confirm 段已 burned，分别按冻结研究口径保守取 0.12/0.09。
    # 触发后实际成交均价高于护栏弃单，不追价。
    # 2026-09-10 优化（实盘逆向选择归因后）：
    # - G1 护栏 0.12→0.15：影子端 15/22 赢单落在 q∈(0.08,0.10]，0.12 仍切除部分
    #   反弹窗；G1 是唯一前向正 EV 版本（n=30 wr13.3% cum_ev+28.8），放宽成交空间。
    # - 下单层叠加盘前过滤器（连阳>1 / |chg|>50bp / 上影<0.5bp 跳过）与动态护栏
    #   （只收紧不放松，clamp [0.05, base]），实现见 multi_live_trader.py 顶部
    #   FIRSTHIT_PRE_MARKET_* 常量区。过滤参数为保守先验未经离线回测，前向观测
    #   裁决后可调；影子端口径不变（仍记录全部触发），护栏选择效应由离线分析覆盖。
    "firsthit_down_v1": ChannelSpec(
        "firsthit_down_v1", "firsthit", "5m", "DOWN", 0.08,
        "首触G0基底 q∈(0.005,0.1]（押DOWN）", order_type="LIMIT",
    ),
    "firsthit_down_body_v1": ChannelSpec(
        "firsthit_down_body_v1", "firsthit", "5m", "DOWN", 0.15,
        "首触G1小实体 body_r≤0.35（押DOWN）", order_type="LIMIT",
    ),
    "firsthit_down_chg_v1": ChannelSpec(
        "firsthit_down_chg_v1", "firsthit", "5m", "DOWN", 0.09,
        "首触G3偏离 chg≤+2.82bp（押DOWN）", order_type="LIMIT",
    ),
    # G4 交互门（2026-09-08 影子+实盘接入）：G4 = G1 ∩ G3（chg≤2.82 ∧ body≤0.35）
    # 依据 shape_scan_v2 冻结扫描：calib n=214 P=12.6% EV+1.03 CI[+0.08,+2.22]；
    # confirm n=86 P=23.3% EV+1.85 CI[+0.35,+3.44]；binom p=0.025 → BH-FDR q=0.063 ✅ STRICT_PASS。
    # 护栏推导：盈亏平衡 q* = P×0.98；calib P=12.6% → q*=0.1235，取 0.12（边际 2.8%）。
    # ⚠️ 非独立暴露：G4 ⊂ G1 且 G4 ⊂ G3，同窗三门全中即同一 DOWN 事件 4 倍下注（含 G0）。
    # ⚠️ 功效警告（见 .pytest_tmp/firsthit_optimization_evidence_v2.md D3）：G4 calib EV(+1.03)
    #   与前向 CI 半宽(≈1.08)几乎相等 → 即使真实效应等于 calib 估计，前向通过概率仅 ≈50%。
    #   预期它大概率 FAIL 时应延长观察期而非直接否决。
    # ⚠️ 研究账本 Phase 0（2026-09-08）政策替换 ΔV：G4 vs G1 = −0.062（显著负）——
    #   用户知情后仍拍板维持实盘（线上已开启下单），前向影子+实盘双重验证裁决。
    "firsthit_down_g4_v1": ChannelSpec(
        "firsthit_down_g4_v1", "firsthit", "5m", "DOWN", 0.12,
        "首触G4交互门 chg≤2.82∧body≤0.35（押DOWN）", order_type="LIMIT",
    ),
    # G7 族（2026-09-08 影子+实盘接入）：G7 纯组合基底及 5 个科学变体
    # 护栏设置依据：回测中均值触发价约 0.068~0.075，盈亏平衡平衡价 wr*0.98（胜率 18%~24% 对应 0.17~0.23）；
    # 取值保持极度保守（只吃深折价优质单）：基底 0.10；streak 0.10；wick20 0.12；strict 0.12；q05 0.05；t270 0.10。
    # ⚠️ 研究账本 Phase 0（2026-09-08）政策替换 ΔV：G7 vs G1 ≈0、g7strict vs g7streak −0.055
    #   （显著负）——用户知情后仍拍板维持实盘（线上已开启下单），前向验证裁决。
    "firsthit_down_g7_v1": ChannelSpec(
        "firsthit_down_g7_v1", "firsthit", "5m", "DOWN", 0.10,
        "首触G7基底 body_r≤0.35∧wick=1（押DOWN）", order_type="LIMIT",
    ),
    "g7_streak_v1": ChannelSpec(
        "g7_streak_v1", "firsthit", "5m", "DOWN", 0.10,
        "首触G7+非强连阳 streak_up≤1（押DOWN）", order_type="LIMIT",
    ),
    "g7_wick20_v1": ChannelSpec(
        "g7_wick20_v1", "firsthit", "5m", "DOWN", 0.12,
        "首触G7+长上影 upper_wick≥2bp（押DOWN）", order_type="LIMIT",
    ),
    "g7_strict_v1": ChannelSpec(
        "g7_strict_v1", "firsthit", "5m", "DOWN", 0.12,
        "首触G7严格版 streak≤1∧wick≥1.5bp（押DOWN）", order_type="LIMIT",
    ),
    "g7_q05_v1": ChannelSpec(
        "g7_q05_v1", "firsthit", "5m", "DOWN", 0.05,
        "首触G7+深折价 q≤0.05（押DOWN）", order_type="LIMIT",
    ),
    "g7_t270_v1": ChannelSpec(
        "g7_t270_v1", "firsthit", "5m", "DOWN", 0.10,
        "首触G7+非极晚 t≤270s（押DOWN）", order_type="LIMIT",
    ),
    # --- rev2 族：孕线反转 + 短影精选，默认 0.30 GTC LIMIT（可热调覆盖） ---
    "hm_inside_5m_v2": ChannelSpec(
        "hm_inside_5m_v2", "nextbar", "5m", "DOWN", 0.30,
        "5m孕线上吊线精选反转（押DOWN）", order_type="LIMIT",
    ),
    "hm_inside_15m_v2": ChannelSpec(
        "hm_inside_15m_v2", "nextbar", "15m", "DOWN", 0.30,
        "15m孕线上吊线反转（押DOWN）", order_type="LIMIT",
    ),
    "ih_inside_15m_v2": ChannelSpec(
        "ih_inside_15m_v2", "nextbar", "15m", "UP", 0.30,
        "15m孕线倒垂线反转（押UP）", order_type="LIMIT",
    ),
}


# ---------------------------------------------------------------------------
# 已退役通道（2026-09-04 用户拍板「信号直接不要了」）：不在 LIVE_CHANNELS
# → 执行器不装配、不可 toggle 上线、status 不列出。spec 本体保留在此处，
# 两个用途：① parse_channel_config 对 LIVE_CHANNELS_JSON / live_channel_overrides
# 里残留的退役通道名做「告警跳过」而非 raise（防生产配置引用退役名导致
# 整个实盘执行器装配失败静默全停）；② 机制类单测的 fixture 源（v3_env /
# regime_gate / streak_gate / 同窗互斥 等门禁机制代码仍在，用退役 spec 驱动）。
# 退役依据同 shadow_version_gate.RETIRED_VERSIONS（两处名单必须一致）。
# ---------------------------------------------------------------------------
RETIRED_CHANNEL_SPECS: dict[str, ChannelSpec] = {
    # A1 momentum 族：「深折价顺势」假设被前向数据证伪
    "quote_momentum_v1": ChannelSpec(
        "quote_momentum_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量（A格顺势）",
    ),
    "quote_momentum_v2": ChannelSpec(
        "quote_momentum_v2", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量·门禁版", v2_guard="min_drop",
    ),
    "quote_momentum_v3": ChannelSpec(
        "quote_momentum_v3", "quote_edge", "5m", "DOWN", _qe_guard("quote_momentum_v1"),
        "报价动量·非连涨门禁版", streak_gate=True,
    ),
    # A2 被同名 v2 严格支配（v2 = v1 触发集纯子集 + 门禁）
    "quote_contrarian_v1": ChannelSpec(
        "quote_contrarian_v1", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向（B格逆势）",
    ),
    "x4_v1": ChannelSpec("x4_v1", "x4", "5m", "DOWN", 0.45, "情绪错位（收阳押次窗DOWN）"),
    # B 档：门禁未兑现，被 contrarian_v2 支配
    "quote_contrarian_v3a": ChannelSpec(
        "quote_contrarian_v3a", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·交替环境版", v2_guard="max_rise", v3_env=True,
    ),
    "quote_contrarian_v3b": ChannelSpec(
        "quote_contrarian_v3b", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·日高回落版", v2_guard="max_rise", v3_env=True,
    ),
    "quote_contrarian_v4": ChannelSpec(
        "quote_contrarian_v4", "quote_edge", "5m", "DOWN", _qe_guard("quote_contrarian_v1"),
        "报价反向·下跌周期版", regime_gate=True,
    ),
}

RETIRED_CHANNELS: frozenset[str] = frozenset(RETIRED_CHANNEL_SPECS)


# 同窗互斥组：组内每市场窗口至多一个通道成交（防同源通道同窗同向叠加敞口；
# 未成交（护栏弃单/失败/占位）不占坑，组内互补通道仍可下单）。
# - S5 深档与 S5 确认通道同源（+5min 确认钩子）、价格带互补（确认护栏 0.75
#   自拦深档高报价窗），互斥防未来护栏调高后叠加。
# momentum 族 v1/v2/v3 互斥组已随三成员全退役而移除（见
# RETIRED_SAME_WINDOW_EXCLUSIVE，仅留作机制测试 fixture）。
SAME_WINDOW_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset({"scene_bull_exhaust_confirm", "s5_deep_z20_v1"}),
    # S2 条件单双变体同源（同一实盘 S2 事件、同一目标周期；t=4/t=5 判价高度重叠），
    # 防同事件双成交叠加敞口（2026-09-06 promote）。
    frozenset({"s2_cond_t4_v1", "s2_cond_t5d_v1"}),
    # 吸收跟随双变体同窗同假设（TD120/TD150 只是判定时点不同，欠反应状态连续），
    # 防同窗双成交（2026-09-06 promote）。
    frozenset({"absorption_follow_td120_v1", "absorption_follow_td150_v1"}),
    # firsthit 全族共享同一首触母事件，防嵌套版本同窗重复下注。
    frozenset({
        "firsthit_down_v1", "firsthit_down_body_v1", "firsthit_down_chg_v1",
        "firsthit_down_g4_v1", "firsthit_down_g7_v1", "g7_streak_v1",
        "g7_wick20_v1", "g7_strict_v1", "g7_q05_v1", "g7_t270_v1",
    }),
)

# 退役的同窗互斥组（不参与生产判定：exclusive_group 只读 SAME_WINDOW_EXCLUSIVE）
RETIRED_SAME_WINDOW_EXCLUSIVE: tuple[frozenset[str], ...] = (
    frozenset({"quote_momentum_v1", "quote_momentum_v2", "quote_momentum_v3"}),
)


def exclusive_group(channel: str) -> frozenset[str] | None:
    """通道所属同窗互斥组（None = 不与他通道同窗成交限制）。"""
    for grp in SAME_WINDOW_EXCLUSIVE:
        if channel in grp:
            return grp
    return None


@dataclass
class ChannelConfig:
    """通道运行时配置（toggle/热调可变；重启回落 live_channels_json 解析结果）。"""

    enabled: bool = False
    amount_usdt: float = 2.0
    max_daily_orders: int = 100
    max_exec_price: float | None = None   # None → ChannelSpec.auto_max_exec
    fired: set[int] = field(default_factory=set, repr=False)  # 本进程已开火 window_start（防同窗重单）
    fire_total: int = 0


def resolve_max_exec(spec: ChannelSpec, cfg: ChannelConfig) -> float:
    """生效护栏：显式配置优先，缺省回落 spec.auto_max_exec。"""
    return cfg.max_exec_price if cfg.max_exec_price is not None else spec.auto_max_exec


def parse_channel_config() -> dict[str, ChannelConfig]:
    """默认配置 + LIVE_CHANNELS_JSON 覆盖 + 校验（非法值 ValueError 拒启）。

    默认全部 OFF（enabled=False），金额/日限取全局默认；JSON 里逐通道覆盖。
    """
    configs: dict[str, ChannelConfig] = {
        ch: ChannelConfig(
            enabled=False,
            amount_usdt=settings.live_default_amount_usdt,
            max_daily_orders=settings.live_default_max_daily_orders,
        )
        for ch in LIVE_CHANNELS
    }

    raw = (settings.live_channels_json or "").strip()
    if not raw:
        return configs

    try:
        overrides: dict = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"多通道实盘：LIVE_CHANNELS_JSON 解析失败：{exc}") from exc
    if not isinstance(overrides, dict):
        raise ValueError("多通道实盘：LIVE_CHANNELS_JSON 顶层必须是对象 {channel: {...}}")

    for ch, ov in overrides.items():
        if ch not in LIVE_CHANNELS:
            if ch in RETIRED_CHANNELS:
                # 退役通道在配置里残留（env JSON / DB 覆盖行）：告警跳过而非 raise。
                # 若这里 raise，main 装配只捕 ValueError 后会把 multi_live_trader
                # 置 None → 整个实盘（含保留通道）静默全停。拼写错误仍走下方 raise。
                logger.warning("多通道实盘：配置中的退役通道已忽略 | {}", ch)
                continue
            raise ValueError(
                f"多通道实盘：未知通道 {ch!r}（白名单 {list(LIVE_CHANNELS)}）")
        if ov is None:
            continue
        if not isinstance(ov, dict):
            raise ValueError(f"多通道实盘：通道 {ch} 的配置必须是对象，got {type(ov).__name__}")
        cfg = configs[ch]
        if "enabled" in ov:
            cfg.enabled = bool(ov["enabled"])
        if "amount_usdt" in ov:
            try:
                amount = float(ov["amount_usdt"])
            except (TypeError, ValueError) as exc:
                # null/非数值必须归一为 ValueError（main 装配只捕 ValueError）；
                # 若放 TypeError 穿透 lifespan 会拖垮整个服务启动
                raise ValueError(
                    f"多通道实盘：通道 {ch} amount_usdt 非法：{ov['amount_usdt']!r}"
                    "（需为数值）") from exc
            if not (0.1 <= amount <= MAX_ORDER_AMOUNT_USDT):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 单笔金额 {amount} 超界"
                    f" [0.1, {MAX_ORDER_AMOUNT_USDT}]（配置误写拒绝启动）")
            cfg.amount_usdt = amount
        if "max_daily_orders" in ov:
            raw_daily = ov["max_daily_orders"]
            # 拒绝小数/布尔/null：日限必须是干净整数，防 int(2.7)→2 静默截断
            if isinstance(raw_daily, bool) or not isinstance(raw_daily, int):
                raise ValueError(
                    f"多通道实盘：通道 {ch} max_daily_orders 非法：{raw_daily!r}"
                    "（需为整数）")
            daily = raw_daily
            if not (1 <= daily <= MAX_DAILY_ORDERS_CAP):
                raise ValueError(
                    f"多通道实盘：通道 {ch} 日限 {daily} 超界 [1, {MAX_DAILY_ORDERS_CAP}]")
            cfg.max_daily_orders = daily
        if "max_exec_price" in ov and ov["max_exec_price"] is not None:
            try:
                exec_price = float(ov["max_exec_price"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"多通道实盘：通道 {ch} max_exec_price 非法：{ov['max_exec_price']!r}"
                    "（需为数值）") from exc
            if not (0.01 <= exec_price <= 0.99):
                raise ValueError(f"多通道实盘：通道 {ch} 护栏 {exec_price} 超界 [0.01, 0.99]")
            cfg.max_exec_price = exec_price
    return configs


def scene_pattern_to_channel(pattern_type: str) -> str | None:
    """场景 pattern_type → 通道名（fake_breakout 钩子 payload 映射）。"""
    ch = f"scene_{pattern_type}"
    return ch if ch in LIVE_CHANNELS else None


def format_channel_title(channel: str | None) -> str:
    """将策略通道英文标识转换为中文+英文展示格式。

    例如: 'hm_inside_15m_v2' -> '**15m孕线上吊线反转（押DOWN）** (`hm_inside_15m_v2`)'
    若未匹配到中文名或为空，则安全回退为 '`channel`'。
    """
    if not channel:
        return f"`{channel or ''}`"
    spec = LIVE_CHANNELS.get(channel) or RETIRED_CHANNEL_SPECS.get(channel)
    if spec and spec.display_name:
        return f"**{spec.display_name}** (`{channel}`)"
    return f"`{channel}`"

