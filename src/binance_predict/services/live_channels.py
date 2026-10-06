"""多通道实盘注册表（MultiLiveTrader 的静态描述层 + 配置解析，纯函数无 DB）。

通道 ID 与影子信号版本名对齐（订单 signal_version 直接用通道名，对账/统计天然一致）。
七族触发机制（由 MultiLiveTrader 分别驱动，见 multi_live_trader.py）：
- quote_edge：5m 采样循环喂价 → 窗内报价区间命中（v2 附加 BTC 门禁，实时喂价解锁；
  v3 环境门禁版再叠加前窗 DOWN/距日高回落，异步 DB 核验后下单）；
- x4：轮询 misalignment_signals PENDING → 次窗 +150s 决策点下单；
- scene：fake_breakout_detector fire 钩子 → 次周期开盘下单（15m 市场）；
- s2_cond：S2CondShadowDetector 实时判价钩子（实盘 S2 派生窗内 t=4/t=5 1m 收盘<周期开盘）
  → 当刻真单押 UP（15m 市场，次周期中段入场）；
- nextbar：NextbarShadowDetector / KREV / P1-P2 检测器的新根收盘钩子 → 次根市场开盘后
  90s 内按冻结方向下单；
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

from .candlestick_shadow_detector import (
    CANDLESTICK_BACKTEST,
    CANDLESTICK_DIRECTIONS,
    CANDLESTICK_DISPLAY_NAMES,
    CANDLESTICK_LOGICAL_SPECS,
    CANDLESTICK_SIGNAL_IDS,
)
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
    # S1 动态轧空路由：复用正式 S1 父信号。NORMAL 在目标窗开盘入场；
    # SQUEEZE 仅首根 5m 收阴后入场；720d 严格时序回放 n=1782/wr=64.93%。
    # 无独立预测市场报价回测，护栏按 wr×0.98≈0.6363 保守取 0.63；默认 OFF。
    "s1_dyn_sq_v1": ChannelSpec(
        "s1_dyn_sq_v1", "scene", "15m", "DOWN", 0.63,
        "S1动态轧空路由（押DOWN）",
    ),
    "scene_bull_exhaust_confirm": ChannelSpec(
        "scene_bull_exhaust_confirm", "scene", "15m", "DOWN", 0.75, "场景S5 确认入场（押DOWN）",
    ),
    # S2 护栏 0.65：2026-08-30 盘口数据校准（与 S1 同病）——47% 信号被旧 0.55 拦截，
    # 被拦段 8 单胜率 88%、均值EV +0.39（与放行段同样赚钱）；开盘 UP 贵=反弹已启动。
    "scene_bear_exhaust": ChannelSpec(
        "scene_bear_exhaust", "scene", "15m", "UP", 0.65, "场景S2 空头耗尽（押UP）",
    ),
    # S2 优化版：复用原 S2，只保留 2≤量比<4 且最近14日区间位置≥0.33。
    # 720d K线回测 n=975/wr=58.97%；无真实预测市场报价，默认护栏按
    # wr×0.98≈0.578 保守取 0.57。parse_channel_config 默认 OFF，先影子前向验证。
    "scene_bear_exhaust_opt_v1": ChannelSpec(
        "scene_bear_exhaust_opt_v1", "scene", "15m", "UP", 0.57,
        "场景S2 空头耗尽·温和放量非低位（押UP）",
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
    # S1 入场变体（2026-10-04，scene_opt 研究；影子+实盘通道同名，实盘默认 OFF）：
    # - s1_early2_v1（S1m2）：S1 命中 + 第 2 根 1m 收盘<窗口开盘 → +2min 入场押 DOWN，
    #   S5 的早确认替代版（买得更便宜）。真实盘口 P1/P2 均EV +0.118/+0.099（S5 +0.015/+0.080）。
    #   护栏 0.75：P1+P2 入场价网格（q2<0.72/0.75/0.78/0.85）里 0.75 累计EV最高（+20.9，n153）。
    #   与 S5 同源同向，同时开启会同窗叠加敞口；替换 S5 时应先关 S5（未并入互斥组，
    #   避免改变现行 S5 生产语义）。
    # - s1_dip45_v1（S1 低吸）：S1 命中后 +62/122/182/302/482s 首次 DOWN≤0.45 入场。
    #   真实盘口 P1/P2 均EV +0.097/+0.199（n57/n123），随机窗口对照 −0.074/+0.011。
    #   护栏 0.47 = 研究阈值 0.45 + 报价噪声余量（成交价对盘口一价 MAE≈0.04）；
    #   与 S1 开盘单同向叠加敞口，金额应小于 S1。
    "s1_early2_v1": ChannelSpec(
        "s1_early2_v1", "scene", "15m", "DOWN", 0.75, "S1早确认·+2min回落（押DOWN）",
    ),
    "s1_dip45_v1": ChannelSpec(
        "s1_dip45_v1", "scene", "15m", "DOWN", 0.47, "S1低吸·DOWN≤0.45（押DOWN）",
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
    # --- nextbar 族：NextbarShadowDetector 新根收盘钩子，目标窗开盘后 90s 内开火。---
    # 15m 冠军（2026-09-21 promote）：深超卖+急跌+卖盘衰竭 → 次根 UP；720d n≈2006
    # 胜率 58.92%，holdout n=368 胜率 61.96%，生产影子 n=28 胜率 67.9%、报价样本
    # n=25 平均 EV +0.130（CI 尚跨 0）。护栏 0.57 < 长期费后保本价 0.5774；默认 OFF。
    "nb_zschamp_15m_v1": ChannelSpec(
        "nb_zschamp_15m_v1", "nextbar", "15m", "UP", 0.57,
        "nextbar 15m冠军·深超卖反转（押UP）",
    ),
    # 5m 误定价候选：冻结条件 sma_slope_atr_5≥1.6606 → 次根 UP。720d 次根收阳
    # 仅 47.43%（长样本反指），edge 依赖 Jul-Aug regime；护栏 0.46 只放行便宜 UP。
    "nb_smaslope_5m_v1": ChannelSpec(
        "nb_smaslope_5m_v1", "nextbar", "5m", "UP", 0.46, "nextbar 5m动量误定价（押UP）",
    ),
    # --- K线反转族（2026-09-18 影子 promote）：KREV-A/B 与 P1/P2 均在 15m 信号根
    # 收盘后预测次根方向；仅目标根开盘后 90s 内的新鲜命中经检测器钩子派单，冷启动
    # 回补不追单。默认全部 OFF；护栏 = 冻结回测胜率 × 0.98（2% 费用保本价）。
    "krev_a_v1": ChannelSpec(
        "krev_a_v1", "kline_reversal", "15m", "UP", 0.629,
        "KREV-A 反转（押UP）",
    ),
    "krev_b_v1": ChannelSpec(
        "krev_b_v1", "kline_reversal", "15m", "UP", 0.621,
        "KREV-B 反转（押UP）",
    ),
    "rev_p1_v1": ChannelSpec(
        "rev_p1_v1", "kline_reversal", "15m", "UP", 0.608,
        "P1 连跌弱阴反转（押UP）",
    ),
    "rev_p2_v1": ChannelSpec(
        "rev_p2_v1", "kline_reversal", "15m", "DOWN", 0.612,
        "P2 连涨弱阳反转（押DOWN）",
    ),
    # --- absorption 族（2026-09-06 影子 promote；2026-10-06 实盘归因优化）：
    # 采样循环喂价内联判定（窗开快照 + TD 秒实时双快照），标定 k/b/位移门/欠反应门
    # 复用 AbsorptionShadowDetector 滚动缓冲。2026-10-06 优化：实盘仅放行 DOWN 顺势
    # （阻断 UP 假冲刺出血点）；宽限期收窄至 20s（禁绝 180~240s 残值期下注）；
    # 配置 entry_band_whitelist=((0.55, 0.99),) 阻断 <0.55 假突破毒单。两版同窗互斥。
    "absorption_follow_td120_v1": ChannelSpec(
        "absorption_follow_td120_v1", "absorption", "5m", "DOWN", 0.78,
        "吸收跟随TD120欠反应→顺势",
        entry_band_whitelist=((0.55, 0.99),),
    ),
    "absorption_follow_td150_v1": ChannelSpec(
        "absorption_follow_td150_v1", "absorption", "5m", "DOWN", 0.86,
        "吸收跟随TD150欠反应→顺势",
        entry_band_whitelist=((0.55, 0.99),),
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
    "firsthit_down_chg_early180_v1": ChannelSpec(
        "firsthit_down_chg_early180_v1", "firsthit_early", "5m", "DOWN", 0.10,
        "首触G3前3分钟研究版（押DOWN）", order_type="LIMIT",
    ),
    "firsthit_down_k10_early180_v1": ChannelSpec(
        "firsthit_down_k10_early180_v1", "firsthit_early", "5m", "DOWN", 0.10,
        "首触K10前3分钟研究版（押DOWN）", order_type="LIMIT",
    ),
    # 2026-10-05：process_recovery_down_v1 移入 RETIRED_CHANNEL_SPECS（前向深析
    # 剔 Top5 赢单后 EV −0.38，无修复价值）。
    "firsthit_down_v1": ChannelSpec(
        "firsthit_down_v1", "firsthit", "5m", "DOWN", 0.08,
        "首触G0基底 q∈(0.005,0.1]（押DOWN）", order_type="LIMIT",
    ),
    "firsthit_down_body_v1": ChannelSpec(
        "firsthit_down_body_v1", "firsthit", "5m", "DOWN", 0.15,
        "首触G1小实体 body_r≤0.35（押DOWN）", order_type="LIMIT",
    ),
    # G3（2026-10-06 双波峰利润带改造）：实盘执行限定在 15S 细粒度分桶（4549 笔
    # 母事件）定位的仅有两个正 EV 波峰——早峰 105~120s（13.8%/EV+0.607）与
    # 主峰 210~240s（10.1%/EV+0.265）× 浅价 q≥0.07；坚决避开 240s+ 衰竭区
    # （240-300s 占 41% 样本暴亏 -124U）与 285s+ 归零深渊（4.97%/-121U）。
    # 叠加下单时刻剩余 ≥60s 硬门（80s→60s，打通 220-240s 主峰死区）与
    # LIMIT→MARKET（限价逆向选择修复：曾把 32% 信号层胜率打成 5% 实盘）。
    # 带门在 multi_live_trader 执行层实施，影子版本仍全样本采集（研究资产）。
    # 护栏 0.09 语义随 MARKET 变为弃单线，动态护栏（q×1.03）继续生效。
    "firsthit_down_chg_v1": ChannelSpec(
        "firsthit_down_chg_v1", "firsthit", "5m", "DOWN", 0.09,
        "首触G3偏离·双波峰版 触发105-120s/210-240s∧q≥0.07（押DOWN）",
    ),
    # 2026-10-05：K10 两版移入 RETIRED_CHANNEL_SPECS——影子本身负 EV
    # （−0.44/−0.42，剔 Top5 后 −0.95），信号已死非执行问题；盈利版的末 60s
    # veto 在实盘路径曾整体失效（17 单全部距窗末<60s 成交），一并退役。
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
    # 2026-10-05：G7 全系六通道移入 RETIRED_CHANNEL_SPECS——前向深析剔 Top5
    # 赢单后 EV 全负（基底 −0.34 ~ strict −0.91），正 EV 是右尾彩票驱动；收紧门
    # 在利润带内部反向选择（浅价带胜率 G4 25.6% → strict 15.0% 逐级稀释）。
    # G4 保留观察（前向 CI 半宽大，延长观察期）。
    # --- rev2 族：孕线反转 + 短影精选（2026-09-20 LIMIT→MARKET）：原 GTC 挂 0.30
    # 等回落被动成交——成交价恒等于挂单价（10/10 笔），只吃到市场跌穿 0.30 的逆向
    # 选择单；护栏被热调至 0.20 后 hm_inside_15m_v2 直接 8 天零成交。改为 MARKET：
    # K 线收盘钩子即下单，均价 >= 0.30（含贴线）弃单，0.30 从挂单价变为弃单线。 ---
    "hm_inside_5m_v2": ChannelSpec(
        "hm_inside_5m_v2", "nextbar", "5m", "DOWN", 0.30,
        "5m孕线上吊线精选反转（押DOWN）",
    ),
    "hm_inside_15m_v2": ChannelSpec(
        "hm_inside_15m_v2", "nextbar", "15m", "DOWN", 0.30,
        "15m孕线上吊线反转（押DOWN）",
    ),
    "ih_inside_15m_v2": ChannelSpec(
        "ih_inside_15m_v2", "nextbar", "15m", "UP", 0.30,
        "15m孕线倒垂线反转（押UP）",
    ),
    # --- CRV 情绪反转族（2026-10-04 研究 run 20261004T083106Z-candle-reversal-certainty
    # 影子 promote + 用户拍板注册实盘，默认全部 OFF）：
    # 「高位/突破阳线 → 次根收阴」的反转族。护栏 = 盲测胜率 × 0.98（费后保本入场价，
    # 与 KREV/P1/P2 同口径）；EV 基准为研究代理模型（0.51 买入+2% 费），前向真实 EV
    # 由影子目标窗报价现算。检测器 crv_shadow_detector 落 kline_shadow_signals，
    # 新鲜命中经 on_kline_reversal_signal 钩子派单（目标根开盘 ≤90s）。三个三因子
    # 探索/观察级版本（crv_dnex_15m_v1 / crv_os_st4_5m_v1 / crv_hi_z3_15m_v1）
    # 未过盲测，仅影子不注册实盘。---
    "crv_hi_brk20_15m_v1": ChannelSpec(
        "crv_hi_brk20_15m_v1", "kline_reversal", "15m", "DOWN", 0.62,
        "CRV高位突破20高反转（押DOWN）",
    ),
    "crv_brk20_15m_v1": ChannelSpec(
        "crv_brk20_15m_v1", "kline_reversal", "15m", "DOWN", 0.57,
        "CRV突破20高反转（押DOWN）",
    ),
    "crv_brk50_hi_15m_v1": ChannelSpec(
        "crv_brk50_hi_15m_v1", "kline_reversal", "15m", "DOWN", 0.59,
        "CRV突破50高∧高位反转（押DOWN）",
    ),
    "crv_brk20_5m_v1": ChannelSpec(
        "crv_brk20_5m_v1", "kline_reversal", "5m", "DOWN", 0.53,
        "CRV 5m突破20高反转（押DOWN）",
    ),
    # --- BRK 假突破回归族（2026-10-05 研究 run 20261005T051500Z-sentiment-kline-reversal
    # 影子 promote + 用户拍板注册实盘，默认全部 OFF）：
    # 「收盘突破前 32/96 根（8 小时）高点 → 次根收阴」的假突破回归——本轮研究三段
    # （训练→验证→盲测）全部一致的最强结构。护栏 = 盲测胜率 × 0.98（费后保本
    # 入场价，与 CRV 同口径）：15m 55.7%×0.98≈0.5459→0.54；5m 取市场结算口径
    # （NOISE 计输）盲测 56.9%×0.98≈0.5580→0.55。⚠ 5m 证据强度弱于 15m：56.9%
    # 是 20 天 n=144 的盲测点估计（Wilson 下界 48.8% 低于盈亏平衡线），且 rev1
    # 口径未过研究验证门槛（验证段 53.5%）——开启前需 ≥4 周影子前向样本 + 近期
    # NOISE 率确认（横盘 regime 如 2026-07 NOISE 率 50% 会摧毁经济 EV）。EV 基准为
    # 研究代理模型，前向真实 EV 由影子目标窗报价现算。检测器 brk_reversion_detector
    # 落 kline_shadow_signals，新鲜命中经 on_kline_reversal_signal 钩子派单（目标根
    # 开盘 ≤90s）。事件集 ⊂ crv_brk20 同周期通道 → 各周期与 CRV 突破族同窗互斥
    # （防同一根突破阳线双倍押 DOWN）。---
    "brkrv_brk8h_15m_v1": ChannelSpec(
        "brkrv_brk8h_15m_v1", "kline_reversal", "15m", "DOWN", 0.54,
        "BRK假突破8h高回归·15m（押DOWN）",
    ),
    "brkrv_brk8h_5m_v1": ChannelSpec(
        "brkrv_brk8h_5m_v1", "kline_reversal", "5m", "DOWN", 0.55,
        "BRK假突破8h高回归·5m（押DOWN）",
    ),
}

# 蜡烛逻辑版本共享 5m/15m 物理事件，但各自可独立注册/热调。默认配置仍全 OFF；
# 护栏取冻结报价期胜率 × 0.98（费后理论最大保本入场价），不因注册自动下单。
for _candle_spec in CANDLESTICK_LOGICAL_SPECS:
    _version = _candle_spec["signal_id"]
    LIVE_CHANNELS[_version] = ChannelSpec(
        _version,
        "candlestick_reversal",
        _candle_spec["timeframe"],
        CANDLESTICK_DIRECTIONS[_version],
        round(CANDLESTICK_BACKTEST[_version][0] * 0.98, 4),
        CANDLESTICK_DISPLAY_NAMES[_version],
    )


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
    # ---- 2026-10-05 退役（firsthit 前向深析，与 shadow_version_gate.RETIRED_VERSIONS 同步）：
    # G7 全系六通道（剔 Top5 赢单后 EV −0.34~−0.91，右尾彩票驱动）、K10 两版
    # （影子本身负 EV）、过程恢复（剔 Top5 后 −0.38）。历史订单与影子数据保留。----
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
    "firsthit_down_k10_v1": ChannelSpec(
        "firsthit_down_k10_v1", "firsthit", "5m", "DOWN", 0.10,
        "首触K10标准化结算距离（押DOWN）", order_type="LIMIT",
    ),
    "firsthit_down_k10_profit_v1": ChannelSpec(
        "firsthit_down_k10_profit_v1", "firsthit", "5m", "DOWN", 0.10,
        "首触K10盈利过滤版（押DOWN）", order_type="LIMIT",
    ),
    "process_recovery_down_v1": ChannelSpec(
        "process_recovery_down_v1", "process_recovery", "5m", "DOWN", 0.35,
        "过程恢复双确认（押DOWN）",
    ),
}

RETIRED_CHANNELS: frozenset[str] = frozenset(RETIRED_CHANNEL_SPECS)


# 同窗互斥组：组内每市场窗口至多一个通道成交（防同源通道同窗同向叠加敞口；
# 未成交（护栏弃单/失败/占位）不占坑，组内互补通道仍可下单）。
# - S5 深档与 S5 确认通道同源（+5min 确认钩子）、价格带互补（确认护栏 0.75
#   自拦深档高报价窗），互斥防未来护栏调高后叠加。
# momentum 族 v1/v2/v3 互斥组已随三成员全退役而移除（见
# RETIRED_SAME_WINDOW_EXCLUSIVE，仅留作机制测试 fixture）。
_CANDLE_5M_CHANNELS = frozenset(v for v in CANDLESTICK_SIGNAL_IDS if "_5m_" in v)
_CANDLE_15M_CHANNELS = frozenset(v for v in CANDLESTICK_SIGNAL_IDS if "_15m_" in v)

SAME_WINDOW_EXCLUSIVE: tuple[frozenset[str], ...] = (
    # 动态 S1 是原 S1 的路由替代版；两者同源、同方向、同目标窗，不能双成交。
    # 不扩大到既有 S5/深档组，避免仅注册新通道就改变当前 S1/S5 生产语义。
    frozenset({"scene_bull_exhaust", "s1_dyn_sq_v1"}),
    frozenset({"scene_bull_exhaust_confirm", "s5_deep_z20_v1"}),
    # 原 S2 与优化版是同一父事件、同方向、同目标窗；若同时启用只允许一单成交。
    frozenset({"scene_bear_exhaust", "scene_bear_exhaust_opt_v1"}),
    # S2 条件单双变体同源（同一实盘 S2 事件、同一目标周期；t=4/t=5 判价高度重叠），
    # 防同事件双成交叠加敞口（2026-09-06 promote）。
    frozenset({"s2_cond_t4_v1", "s2_cond_t5d_v1"}),
    # 吸收跟随双变体同窗同假设（TD120/TD150 只是判定时点不同，欠反应状态连续），
    # 防同窗双成交（2026-09-06 promote）。
    frozenset({"absorption_follow_td120_v1", "absorption_follow_td150_v1"}),
    # CRV 15m 高位/突破族三通道同源（同一根突破阳线物理事件、同押 DOWN、同目标窗：
    # hi_brk20 ⊇ brk20 的事件子集、brk50_hi 高度重叠）→ 同窗至多一单成交。
    # BRK 8h 突破回归事件集是 crv_brk20 的子集（前窗更长 ⇒ 更难突破），同押 DOWN
    # 同目标窗 → 并入同组防同一根突破阳线双倍押 DOWN（仅两通道都开启才生效）。
    frozenset({"crv_hi_brk20_15m_v1", "crv_brk20_15m_v1", "crv_brk50_hi_15m_v1",
               "brkrv_brk8h_15m_v1"}),
    # 同理 5m 侧：brk8h 事件集 ⊂ brk20 事件集（都开启时同窗至多一单 DOWN）。
    frozenset({"crv_brk20_5m_v1", "brkrv_brk8h_5m_v1"}),
    # 同周期蜡烛逻辑版本共享物理事件，避免主版/对照/组件/潜力标签重复暴露。
    _CANDLE_5M_CHANNELS,
    _CANDLE_15M_CHANNELS,
    # 2026-09-12：解除 firsthit 全族同窗互斥，允许各 G 系列及 G7 变体按门禁独立下单对比实盘效果
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

