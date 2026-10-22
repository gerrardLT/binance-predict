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
- gap_crowd：采样循环喂价内联判定（5m 与 15m 市场各自调用）——窗后半段预测市场报价
  比人群投票倾向便宜 ≥5pp → 买便宜侧 UP；判定函数与影子同源
  （gap_crowd_shadow_detector.evaluate_gap_crowd 纯函数）。

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
# 2026-10-22 通道治理：归档名单的单一事实源在 shadow_version_gate.ARCHIVED_VERSIONS
# （实盘/影子双端共用一个名单，避免两处硬编码不同步）。candlestick 已先导入
# shadow_version_gate，此处为缓存命中，无循环导入。
from .shadow_version_gate import ARCHIVED_VERSIONS as _ARCHIVED_VERSIONS

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
    # 低波门禁（蜡烛 5m 家族）：1h 实现波动率 7 日分位 ≤ 该值不开仓（贴线拒单，
    # 含边界）；决策时信息（窗口开始前 1m 收盘，无前视），数据缺失保守弃单。
    # None = 不启用；研究冻结口径，不进 parse_channel_config 可覆盖项。
    min_rv_pct: float | None = None
    # 点差护栏（KDC 族 2026-10-16，优化项1）：信号落表时刻两侧展示价之和−1
    # （up_price+down_price−1，与影子 feature_snapshot.quote_cross_cost 同一缓存快照）
    # 严格大于该值 → 实盘弃单。None = 不启用。依据：研究 656 窗 PREDEXON +3.5s 盘口，
    # 点差 >0.04 的 126 事件 EV −0.018（≤0.02 的 772 事件 +0.077）；2026-09 整月
    # 点差 0.048 恰被拦截。cross_cost 缺失（缓存无快照）→ 不拦（护栏非安全闸，
    # 价格保护仍由 max_exec_price 承担）。⚠ 阈值按 PREDEXON best_ask−best_bid 校准、
    # 以币安展示价 cross_cost 代理执行——代理分布由影子 q10/t1 前向数据校验后再
    # 考虑启用实盘。
    spread_guard: float | None = None
    # 高波减仓（KDC 族，优化项2）：信号携带 vol_tier=="high"（ATR20 近500根分位≥0.66）
    # 时实盘金额乘以该因子（0.5=半仓）。依据：研究 656 窗 EV 低/中/高波 +0.028/
    # +0.103/+0.042（高波砍半）。tier unknown（数据不足）→ 不减仓（保守=维持基准
    # 行为）。None = 不启用。影子行记录 vol_tier/atr_pct，前向可按层复核。
    highvol_amount_factor: float | None = None
    # 新通道默认启用标记（2026-10-23）：True 时 parse_channel_config 初始 enabled=True，
    # 仍可被 LIVE_CHANNELS_JSON / DB live_channel_overrides 覆盖关闭。仅用于用户明确
    # 要求"注册即开启"的通道；既有通道保持 False，行为零变化。
    default_enabled: bool = False


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
    # 入场价白名单 [0.50,0.65)（2026-10-16 用户拍板恢复 0.50 下限；史：10-12 加
    # ≥0.50 → 10-15 收紧 [0.55,0.65) → 10-16 放宽回 [0.50,0.65)）：全量深审
    # （PREDEXON 五档费后，115 生产触发窗 08-16→10-15）——[0.50,0.55) 带 n=59
    # 胜率 54.2%/平均 EV +0.75%≈0（非负，恢复放行攒前向样本）；[0.55,0.65) 带
    # n=33 胜率 75.8%/EV +28.8%。<0.50 仍拦（10-12 实盘归因 8 单 1 胜、费后
    # EV −0.73/单）。⚠ 样本内择带：独立回放事件集 7 月切片 −5.6%/单、10 月
    # +2.9%/单（负/平），与 10-12 距低门同族风险（该门已样本外证伪撤回，勿复加）。
    # 前向裁决线（冻结）：[0.50,0.65) 带内 ≥30 笔成交平均费后 EV ≤ 0 即回退复盘。
    # 2026-10-13 撤回同批上线的「距 24h 低点 <0.45%」门：2025-09→2026-08 K 线复现
    # 样本外 831 事件差 +1.6pp [−5.6,+8.8] 不显著，实盘 74 单结论属样本内拟合。
    "scene_bear_exhaust": ChannelSpec(
        "scene_bear_exhaust", "scene", "15m", "UP", 0.65, "场景S2 空头耗尽（押UP）",
        entry_band_whitelist=((0.50, 0.65),),
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
    # S4 延迟入场（2026-10-09 用户拍板注册实盘，默认 OFF）：正式 S4 命中后等目标窗
    # 第 1 根 1m 收盘 < 窗口开盘（BTC 已回落）才在 +60s 入场押 DOWN（S1m2 同款
    # 「先确认后买」机制，检测器 s4_variant_shadow_detector 确认后经
    # on_s1_variant_signal 钩子派单）。护栏 0.65 = 720d 重放胜率 66.2%×0.98
    # （费后保本入场价）略下方；入场价≈0.62 高于开盘 S4 单，EV 基准为研究代理
    # 模型（P1/P2 均EV +0.119/+0.107，未扣 20U 深度滑点），前向真实 EV 由影子
    # 目标窗报价现算。订单 scene_signal_id = 父 S4 信号 id（结算走 FakeBreakoutSignal）。
    # ⚠️ 与 scene_momentum_fade 同父事件、同窗、同押 DOWN：S4 开盘单与延迟确认单
    # 是同向叠加敞口（未并入互斥组——两者入场时点/价位互补，先并行前向观测，
    # 与 s1_early2_v1 对 S5 的处理一致）；同时开启时金额应保守。
    "s4_delay60_v1": ChannelSpec(
        "s4_delay60_v1", "scene", "15m", "DOWN", 0.65, "S4延迟入场·首分钟回落（押DOWN）",
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
    # 2026-10-15 优化（TD150 挖除 [0.78,0.81) 假摔失血带）：80 天全量回放
    # （2026-07-26→10-15，生产 15s 报价链 + 币安 5m 结算，DOWN 侧 984 笔）——
    # [0.78,0.81) 是该侧唯一确定性失血带：n=47 胜率 68.1% / 费后 EV −0.156
    # （q≈0.79 保本线 80.6% 不可达）；挖除后放行段 [0.55,0.82) 全量 EV
    # +0.0078→+0.0285、近月 −0.0340→−0.0237。[0.81,0.82) 保留（16/16 全胜，
    # 与 [0.82,0.90) 高确信段同族）；TD120 未动（实盘默认 OFF，无同等证据）。
    "absorption_follow_td120_v1": ChannelSpec(
        "absorption_follow_td120_v1", "absorption", "5m", "DOWN", 0.78,
        "吸收跟随TD120欠反应→顺势",
        entry_band_whitelist=((0.55, 0.99),),
    ),
    "absorption_follow_td150_v1": ChannelSpec(
        "absorption_follow_td150_v1", "absorption", "5m", "DOWN", 0.86,
        "吸收跟随TD150欠反应→顺势",
        entry_band_whitelist=((0.55, 0.78), (0.81, 0.99)),
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
    # G3v2（2026-10-06 双波峰利润带改造）：升级为独立 v2 版本，实盘与影子从 0 开始。
    # 彻底告别 v1 历史逆向选择与时点漂移；实盘与影子完全同源：触发 105~120s / 210~240s
    # 且浅价 q≥0.07（4549 笔前向事件 15S 分桶验证：早峰 13.8%/EV+0.607，主峰 10.1%/EV+0.265）。
    # 锁定首触 BTC 特征、下单剩余 ≥60s 硬门、MARKET 市价单、动态护栏（q×1.03）。
    # 2026-10-13：通道护栏 auto_max_exec 0.09→0.101（用户拍板）。首触价天然贴 Q_HI=0.1
    # 上限（实测入场价只有 0.09/0.10 两档），0.09 护栏会把全部 v2 信号判为超价弃单，
    # 面板「带护栏实际」长期 0 笔；0.101 放行 q∈[0.07,0.1] 全区间。
    # 2026-10-19：实盘 0 单根因修复——该族豁免旧 firsthit 盘前否决（上影<0.5bp/连阳>1），
    # 且实盘动态护栏改为 q×1.3 封顶 0.12（见 multi_live_trader.G3_LIVE_GUARD_*）。
    # ⚠ 本值只作用于影子「带护栏」口径与通道预设，实盘 firsthit 下单走动态护栏，不用本值。
    "firsthit_down_chg_v2": ChannelSpec(
        "firsthit_down_chg_v2", "firsthit", "5m", "DOWN", 0.101,
        "首触G3偏离v2·双波峰版 触发105-120s/210-240s∧q≥0.07（押DOWN）",
    ),
    # 2026-10-13 分桶组合 D/E（q≥0.07 ∩ chg≤+3.5bp ∩ 首触 15s 分桶；75D 线上采样回测，
    # 样本内择优，默认 OFF，影子+小额实盘前向裁决）。门由 firsthit_shadow_detector._gate_of
    # 单点定义，影子与实盘同源；首触 <240s 保证下单剩余 ≥60s。
    # 护栏 0.101 仅作用于面板「带护栏」口径；实盘下单走动态护栏 q×1.3 封顶 0.12
    # （2026-10-19 与 G3v2 同批修复，豁免旧盘前否决）。
    # ⚠ E ⊂ D 且与 G3v2 事件重叠：firsthit 全族不设同窗互斥，同开会对同一事件重复下注。
    "firsthit_down_chg35_d_v1": ChannelSpec(
        "firsthit_down_chg35_d_v1", "firsthit", "5m", "DOWN", 0.101,
        "首触分桶D q≥0.07∧chg≤3.5bp 首触[105,195)∪[210,240)（押DOWN）",
    ),
    "firsthit_down_chg35_e_v1": ChannelSpec(
        "firsthit_down_chg35_e_v1", "firsthit", "5m", "DOWN", 0.101,
        "首触分桶E q≥0.07∧chg≤3.5bp 首触[105,150)∪[180,195)∪[210,240)（押DOWN）",
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
    # 2026-10-23 用户拍板注册（PREDEXON 5m 真实盘口验 EV 后）：
    # hm_quality_5m_v1 720d n=28 胜率 78.6%；盘口 n=4 3胜 EV +43.3%（样本小）。
    # hm_moderate_5m_v1 720d n=184 胜率 64.1%（三段 62.8/65.0/67.7）；
    # 盘口 n=23 17胜 EV +43.2%。两者 T 收盘后在次根 5m 开盘押 DOWN。
    # 护栏 0.55：盘口证据 ≤0.55 EV +50.8%（moderate）/ +36.6%（quality）。
    "hm_quality_5m_v1": ChannelSpec(
        "hm_quality_5m_v1", "nextbar", "5m", "DOWN", 0.55,
        "5m收顶阳线+长下影阴线·质量版（押DOWN）", default_enabled=True,
    ),
    "hm_moderate_5m_v1": ChannelSpec(
        "hm_moderate_5m_v1", "nextbar", "5m", "DOWN", 0.55,
        "5m收顶阳线+长下影阴线·中等冲刺版（押DOWN）", default_enabled=True,
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
    # --- crv_dnbrk24 下破吞没族（2026-10-12 复核研究 run 20261012T084631Z-
    # kline-regime-transition「A 信号下破半」，用户拍板注册影子+实盘，默认 OFF）：
    # 「阴线吞没 ∧ 收盘跌破前 24 根低 → 押次根收阳 UP」——与 CRV 突破族方向相反
    # （那条押 DOWN），事件不重叠。15m 样本外(288d) n=249 胜率 61.9%（发现 59.2%/
    # 验证 60.6%/盲测 63.4%/前向 62.5%），护栏 0.606=61.9%×0.98；5m 样本外 n=888
    # 胜率 55.7%（盲测段 52.2% 偏弱，相对「仅破24低」对照仅 +1.1pp），护栏 0.546
    # 刻意只放行便宜 UP。检测器 crv_shadow_detector 落 kline_shadow_signals，新鲜
    # 命中经 on_kline_reversal_signal 钩子派单（目标根开盘 ≤90s）。---
    "crv_dnbrk24_15m_v1": ChannelSpec(
        "crv_dnbrk24_15m_v1", "kline_reversal", "15m", "UP", 0.606,
        "CRV下破吞没24低反转（押UP）",
    ),
    "crv_dnbrk24_5m_v1": ChannelSpec(
        "crv_dnbrk24_5m_v1", "kline_reversal", "5m", "UP", 0.546,
        "CRV 5m下破吞没24低反转（押UP）",
    ),
    # --- 2026-10-15 反转研究 run 20261015T061535Z-reversal-study「修正后回顾性验证」
    # 两个新信号（用户拍板注册影子+实盘，默认 OFF）。⚠ 末段 360D 筛选时已被看过，
    # 研究数字均为回顾性、非独立盲测，影子前向样本才是裁决依据。
    # crv_dnvol20_15m_v1：5m 阴线∧跌破前20低∧放量(≥1.5×均量)，且该 5m 根为 15m 窗口
    # 最后一根 → 押下一根 **15m** 收阳 UP（信号 5m 求值、市场 15m）。对齐边界样本
    # n=2904 胜率 59.1%（训练 59.8%/验证 57.6%/末段 59.2%）。护栏 0.57 = 费后保本价
    # (59.1%−0.01)/1.02。2026-07→10-14 的 15m 真实盘口回填（Predict.fun，开盘+3.5s，
    # 五档 VWAP，10U）：185/205 事件可用，VWAP 均值 0.549，不设护栏每份费后 EV +0.095；
    # 护栏 0.57 仍成交 137 笔、ROI +18.2%。该段合约胜率 66.5% 高于历史 59.1%（9 月
    # 单月 76% 拉高），属小样本，上线判断以影子前向为准。
    # 与 crv_dnbrk24_15m_v1 同方向同目标窗（同窗重叠约 12.8%）→ 同组互斥。
    "crv_dnvol20_15m_v1": ChannelSpec(
        "crv_dnvol20_15m_v1", "kline_reversal", "15m", "UP", 0.57,
        "CRV 5m放量下破20低→15m反弹（押UP）",
    ),
    # --- KDC 方向确定性族（2026-10-16 研究 run 20261015T153500Z-kline-directional-
    # certainty：720d 预注册三段隔离+盲测一次+PREDEXON 报价层 EV 验证+衰退归因。
    # 四通道实盘默认 OFF，影子常开；带执行层优化 1-4：
    #   1) spread_guard 点差护栏（信号时刻两侧价和−1 > 0.04 弃单，与影子同源快照）；
    #   2) highvol_amount_factor=0.5 高波减仓（ATR20 近500根分位≥0.66 时半仓）；
    #   3) 入场价带只记录不拦截（影子 feature_snapshot.entry_band：<0.52/[0.52,0.55)/
    #      [0.55,0.58)/≥0.58，研究带 EV +0.143/+0.060/+0.041/−0.033）；
    #   4) 影子双时点报价（entry_up/down_price≈开盘后首轮询 0~60s + q10_*≈+10s±15s；
    #      研究回测为 +3.5s 对齐口径，t1 与之不同——显式声明，影子前向统一以 t1/q10
    #      为准）。护栏=盲测胜率×0.98；报价段(07-01→10-15,10U五档VWAP) EV 见注释。
    # ⚠ 衰退警示（归因报告 20261015T162800Z）：四信号聚合 EV 全部来自 2026-07/08，
    #   09 月市场重定价归零、10 月真实胜率塌陷转负——启用实盘前必须先看影子前向
    #   分月 EV 与错价（W−mid），复活判据=滚动60d K线Δ≤−8pp 且错价回正。---
    "kdc_s1_96hi_15m_v1": ChannelSpec(
        "kdc_s1_96hi_15m_v1", "kline_reversal", "15m", "DOWN", 0.64,
        "KDC破96高收顶反转（押DOWN）",
        spread_guard=0.04, highvol_amount_factor=0.5,
    ),
    "kdc_s2_ext_15m_v1": ChannelSpec(
        "kdc_s2_ext_15m_v1", "kline_reversal", "15m", "DOWN", 0.61,
        "KDC超24高冲顶反转（押DOWN）",
        spread_guard=0.04, highvol_amount_factor=0.5,
    ),
    "kdc_s3_3up_15m_v1": ChannelSpec(
        "kdc_s3_3up_15m_v1", "kline_reversal", "15m", "DOWN", 0.61,
        "KDC收顶3连阳破20高（押DOWN）",
        spread_guard=0.04, highvol_amount_factor=0.5,
    ),
    "kdc_s4_3dn_15m_v1": ChannelSpec(
        "kdc_s4_3dn_15m_v1", "kline_reversal", "15m", "UP", 0.59,
        "KDC破20低3连阴大阴（押UP）",
        spread_guard=0.04, highvol_amount_factor=0.5,
    ),
    # crv_bodybrk20_5m_v1：5m 阳线∧实体/振幅≥0.75∧突破前20高 → 押次根 5m 收阴 DOWN。
    # n=8649 胜率 55.2%（训练 54.5%/验证 54.8%/末段 56.5%）；护栏 0.53 = 费后保本价
    # (55.2%−0.01)/1.02。事件集 ⊂ crv_brk20_5m_v1（同向同目标窗）→ 并入其 5m 互斥组。
    # 月度稳定性弱于信号 2（2026-03 胜率 47%），开启前先看影子前向。
    "crv_bodybrk20_5m_v1": ChannelSpec(
        "crv_bodybrk20_5m_v1", "kline_reversal", "5m", "DOWN", 0.53,
        "CRV 5m大实体突破20高反转（押DOWN）",
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
    # --- gap_crowd 族已于 2026-10-10 同日证伪退役（见 RETIRED_CHANNEL_SPECS 内说明）---
}

# 蜡烛逻辑版本共享 5m/15m 物理事件，但各自可独立注册/热调。默认配置仍全 OFF；
# 护栏取冻结报价期胜率 × 0.98（费后理论最大保本入场价），不因注册自动下单。
# 2026-10-09 方案A（用户拍板）：5m 长下影家族低波门禁——1h 实现波动率 7 日分位
# ≤0.30（含贴线）不开仓。归因四口径（c2_live 39 / fam_live 115 / c2_shadow 153 /
# fam_shadow 199）避输:丢赢 1.4~1.9:1、fam_live 费后 +63→+129U、拆半稳定；
# 被弃赢单均为 +5U 级小赢，0.2~0.3 价位大赢单全在高波段保留。口径实现见
# multi_live_trader.compute_rv_1h_pct7d（与审计 prep.feats 同源）；15m 版本与
# candidate_* 假说版不在研究范围，不加门。仅 candle_hm_bull_5m_*（长下影家族，
# 含 asia 时段子集）生效。
CANDLE_5M_MIN_RV_PCT = 0.30
for _candle_spec in CANDLESTICK_LOGICAL_SPECS:
    _version = _candle_spec["signal_id"]
    _is_5m_wick = _candle_spec["timeframe"] == "5m" and _version.startswith("candle_hm_bull_5m_")
    LIVE_CHANNELS[_version] = ChannelSpec(
        _version,
        "candlestick_reversal",
        _candle_spec["timeframe"],
        CANDLESTICK_DIRECTIONS[_version],
        round(CANDLESTICK_BACKTEST[_version][0] * 0.98, 4),
        CANDLESTICK_DISPLAY_NAMES[_version],
        min_rv_pct=CANDLE_5M_MIN_RV_PCT if _is_5m_wick else None,
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
    # --- gap_crowd 族：2026-10-10 注册 → 同日证伪退役（勿重新启用）。
    # 触发条件只在 up_price/down_price 快照不一致的瞬间成立（盲测 948 事件 100%
    # 报价和 <0.95、中位 0.91；全样本仅 1.03% 不一致）：记录入场价不可成交，
    # 一致化入场价重算 EV +0.036 → −0.072、只留一致样本则 0 事件。
    # 详见研究报告 ERRATUM 与 gap_crowd_shadow_detector 模块头。
    "gap_crowd_5m_v1": ChannelSpec(
        "gap_crowd_5m_v1", "gap_crowd", "5m", "UP", 0.50,
        "报价落后人群·5m后半窗（已证伪退役）",
    ),
    "gap_crowd_15m_v1": ChannelSpec(
        "gap_crowd_15m_v1", "gap_crowd", "15m", "UP", 0.56,
        "报价落后人群·15m后半窗（已证伪退役）",
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
    # 2026-10-06 退役：旧 G3v1 升级为 v2 双波峰版，旧数据归档保留供历史审计。
    "firsthit_down_chg_v1": ChannelSpec(
        "firsthit_down_chg_v1", "firsthit", "5m", "DOWN", 0.09,
        "首触G3价格偏离（已退役：已升级为v2双波峰版）", order_type="MARKET",
    ),
}

# ---------------------------------------------------------------------------
# 2026-10-22 通道治理（用户拍板）：除保留清单外全部归档。
# 保留 = 当时实盘 ON 的 15 个 + kdc 四版 + crv 全族 + 复核观察 5 个
# （x4_v2 / S5 确认 / 吸收 TD120+TD150 / candle 15m union）。
# 归档名单单一事实源：shadow_version_gate.ARCHIVED_VERSIONS（与影子隐藏共用）；
# 显式名单而非「不在白名单就归档」——防未来新注册通道被默归归档而静默不交易。
# 归档 spec 本体保留在 RETIRED_CHANNEL_SPECS（审计/回滚），注释见上。
# ---------------------------------------------------------------------------
KEPT_CHANNELS_20261022: frozenset[str] = frozenset({
    # scene 族（实盘主力）
    "scene_bull_exhaust", "scene_bear_exhaust", "scene_momentum_fade",
    # nextbar / kline_reversal（实盘 ON）
    "krev_b_v1", "hm_inside_15m_v2",
    "hm_quality_5m_v1", "hm_moderate_5m_v1",
    # firsthit D/E（实盘 ON，观察期）
    "firsthit_down_chg35_d_v1", "firsthit_down_chg35_e_v1",
    # CRV 全族
    "crv_hi_brk20_15m_v1", "crv_brk20_15m_v1", "crv_brk50_hi_15m_v1",
    "crv_brk20_5m_v1", "crv_dnbrk24_15m_v1", "crv_dnbrk24_5m_v1",
    "crv_dnvol20_15m_v1", "crv_bodybrk20_5m_v1",
    # KDC 四版
    "kdc_s1_96hi_15m_v1", "kdc_s2_ext_15m_v1", "kdc_s3_3up_15m_v1",
    "kdc_s4_3dn_15m_v1",
    # BRK 假突破回归 5m（实盘 ON）
    "brkrv_brk8h_5m_v1",
    # 蜡烛族：仅保留实盘 ON 版与 15m union
    "candle_hm_bull_5m_trend4h_up_shadow_hyp_v1",
    "candle_hm_bull_15m_union_shadow_v1",
    # 复核观察（不归档）
    "x4_v2", "scene_bull_exhaust_confirm",
    "absorption_follow_td120_v1", "absorption_follow_td150_v1",
})
ARCHIVED_CHANNELS_20261022: frozenset[str] = frozenset(_ARCHIVED_VERSIONS)
_unknown_archived = ARCHIVED_CHANNELS_20261022 - set(LIVE_CHANNELS)
if _unknown_archived:
    # 拼写错误/已移除版本不应静默：告警供启动日志审计（不阻断装配）
    logger.warning("通道治理：归档名单中的未知通道已忽略 | {}", sorted(_unknown_archived))
for _archived_ch in ARCHIVED_CHANNELS_20261022 & set(LIVE_CHANNELS):
    RETIRED_CHANNEL_SPECS[_archived_ch] = LIVE_CHANNELS.pop(_archived_ch)

RETIRED_CHANNELS: frozenset[str] = frozenset(RETIRED_CHANNEL_SPECS)


# 同窗互斥组：组内每市场窗口至多一个通道成交（防同源通道同窗同向叠加敞口；
# 未成交（护栏弃单/失败/占位）不占坑，组内互补通道仍可下单）。
# - S5 深档与 S5 确认通道同源（+5min 确认钩子）、价格带互补（确认护栏 0.75
#   自拦深档高报价窗），互斥防未来护栏调高后叠加。
# momentum 族 v1/v2/v3 互斥组已随三成员全退役而移除（见
# RETIRED_SAME_WINDOW_EXCLUSIVE，仅留作机制测试 fixture）。
# 2026-10-22 通道治理：组常量保留完整语义（含已归档成员，兼作测试 fixture 源），
# 归档后是否生效由 exclusive_group() 在运行时与 LIVE_CHANNELS 求交决定。
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
    # 2026-10-23 新增两版同源同向（同一根 A/T 可同时满足 quality 与 moderate 分支），
    # 默认均开启实盘 → 必须互斥，防同窗双倍 DOWN 敞口。
    frozenset({"hm_quality_5m_v1", "hm_moderate_5m_v1"}),
    # CRV 15m 高位/突破族三通道同源（同一根突破阳线物理事件、同押 DOWN、同目标窗：
    # hi_brk20 ⊇ brk20 的事件子集、brk50_hi 高度重叠）→ 同窗至多一单成交。
    # BRK 8h 突破回归事件集是 crv_brk20 的子集（前窗更长 ⇒ 更难突破），同押 DOWN
    # 同目标窗 → 并入同组防同一根突破阳线双倍押 DOWN（仅两通道都开启才生效）。
    # KDC 三条 DOWN 版本（2026-10-16）同为「破 N 根高→押 DOWN」物理事件（96/24+冲量/
    # 20+3连阳 均落在突破阳线上，事件集与 brk20 族高度重叠）→ 并入同组。
    frozenset({"crv_hi_brk20_15m_v1", "crv_brk20_15m_v1", "crv_brk50_hi_15m_v1",
               "brkrv_brk8h_15m_v1", "kdc_s1_96hi_15m_v1", "kdc_s2_ext_15m_v1",
               "kdc_s3_3up_15m_v1"}),
    # 同理 5m 侧：brk8h 事件集 ⊂ brk20 事件集（都开启时同窗至多一单 DOWN）。
    # crv_bodybrk20_5m 事件集 ⊂ crv_brk20_5m（大实体阳线是阳线突破的子集）→ 同组。
    frozenset({"crv_brk20_5m_v1", "brkrv_brk8h_5m_v1", "crv_bodybrk20_5m_v1"}),
    # 15m 押 UP：放量下破(5m 求值)与下破吞没同向同目标窗、同窗重叠约 12.8% → 同窗至多一单。
    # KDC 破20低3连阴大阴（kdc_s4_3dn）同为「跌破 20 低→押 UP」物理事件 → 并入同组。
    frozenset({"crv_dnbrk24_15m_v1", "crv_dnvol20_15m_v1", "kdc_s4_3dn_15m_v1"}),
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
    """通道所属同窗互斥组（None = 不与他通道同窗成交限制）。

    2026-10-22 通道治理：SAME_WINDOW_EXCLUSIVE 常量保留完整语义（含已归档
    成员，兼作机制测试 fixture），运行时只与当前 LIVE_CHANNELS 求交——
    归档通道不参与生产判定，交集不足 2 个成员时互斥语义消失，返回 None。
    """
    if channel not in LIVE_CHANNELS:
        return None  # 归档/未注册通道不参与生产互斥判定
    for grp in SAME_WINDOW_EXCLUSIVE:
        if channel in grp:
            active = frozenset(grp & set(LIVE_CHANNELS))
            return active if len(active) >= 2 else None
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

    默认按 ChannelSpec.default_enabled（既有通道 False，2026-10-23 新增两通道 True）；
    金额/日限取全局默认；JSON / DB 覆盖优先。
    """
    configs: dict[str, ChannelConfig] = {
        ch: ChannelConfig(
            enabled=spec.default_enabled,
            amount_usdt=settings.live_default_amount_usdt,
            max_daily_orders=settings.live_default_max_daily_orders,
        )
        for ch, spec in LIVE_CHANNELS.items()
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

