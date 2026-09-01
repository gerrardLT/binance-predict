# 裸K 调研：关键词图谱、来源分级与可机检化判定

> 任务：为「裸K形态 / 高级裸K形态 / K线结构」建立可追溯到来源的形态库，并判定每个术语
> 能否写成**逐根 OHLCV 布尔判据**。判据不成立的术语一律进 §6 不可验证清单，不编造。
> 本文档只记录调研过程与采信边界；**任何外部胜率数字都不作为本项目的结论**，
> 实证结果一律见同目录 `REPORT.md`（本项目数据 + 本项目费用结构算出来的）。

调研方法论（用户指定）：**先联想关键词 → 再用关键词搜索**。共 4 轮检索 + 2 轮补漏。

---

## 1. 关键词联想链（A-1，先联想后搜索）

按三层结构各自展开，中英双语对照。左列是联想源，右列是实际投入检索的词条。

### T1 经典裸K（单根 / 双根 / 三根）

| 联想锚点 | 英文检索词 | 中文检索词 |
|---|---|---|
| 实体占比极端 | doji, gravestone doji, dragonfly doji, four-price doji, spinning top, long-legged doji, high-wave | 十字星 墓碑线 蜻蜓线 长腿十字 纺锤线 上下影十字 |
| 影线单侧极端 | hammer, hanging man, inverted hammer, shooting star, pin bar, pin bar rejection | 锤子线 上吊线 倒锤子 射击之星 流星线 针形线 |
| 无影线 | marubozu, closing marubozu, opening marubozu, bald candle | 光头光脚 光头阳线 光脚阴线 引擎线 |
| 两根包裹关系 | bullish/bearish engulfing, outside bar, harami, harami cross, inside bar, tweezer top/bottom, identical three crows | 吞没 阳包阴 阴包阳 孕线 十字孕 内包线 外包线 平头顶 平头底 |
| 两根同向并列 | piercing line, dark cloud cover, separating lines, confluence/advance block, matched low | 曙光初现 刺透形态 乌云盖顶 分手线 并列阳线 |
| 三根结构 | morning star, evening star, doji star, three white soldiers, three black crows, three inside up/down, three-line strike, rising/falling three methods, abandoned baby, concealing baby swallow, tri-star | 启明星 黄昏星 十字启明星 红三兵 黑三鸦 三颗内部 三线反击 上升三法 下降三法 弃婴形态 藏婴吞没 三星 |
| 缺口族 | breakaway gap, runaway gap, exhaustive gap, island reversal, upside/two gap side-by-side white | 突破缺口 中继缺口 衰竭缺口 岛形反转 跳空并列白线 |

### T2 高级裸K / 聪明钱概念（SMC）

| 联想锚点 | 英文检索词 | 中文检索词 |
|---|---|---|
| Wyckoff 事件 | spring (test 1/2/3), upthrust (UT), upthrust after distribution (UTAD), sign of strength, sign of weakness, no demand, no supply, back-up, economical low, Phase A–E of accumulation/distribution | 威科夫 弹簧 二次弹簧 上冲回落 派发后上冲 强势信号 弱势信号 无量反弹 吸筹 派发 阶段 |
| 流动性 | liquidity sweep, stop hunt, purge, ranse, judas swing, external range liquidity (ERL), internal range liquidity (IRL), equal highs/lows (EQH/EQL), liquidity pool, buy-side/sell-side liquidity | 流动性猎杀 扫止损 扫盘 假突破 内外流动性 等高 等低 流动性池 |
| 市场结构 | break of structure (BOS), change of character (CHoCH), market structure shift (MSS), higher high / lower low sequence, structure break, HH/HL/LH/LL | 结构破坏 特性改变 趋势转换 结构突破 高低点序列 |
| 订单块 | order block (bullish/bearish OB), mitigation, breaker block, referral zone, demand/supply zone, base/candle prior to displacement | 订单块 机构订单 缓解 回补 破坏块 需求区 供给区 |
| 不平衡 | fair value gap (FVG), imbalance, value area gap (VAG), inefficient price (IP), gap fill, 3-candle imbalance | 公允价值缺口 不平衡区 缺口回补 无效价格 |
| VSA | effort vs result, no demand bar, no supply bar, upthrust, test bar, spread vs volume divergence, stopping volume | 量价背离 试盘 无量反弹 停止量 窄幅放量 |
| 形态法则 | 2B rule, quasimodo (QMS/QML), high-low-double-top, wick failure model, liquidity sweep + reclaim | 2B 法则 Quasimodo 头肩变形 影线失效模型 |
| 假突破 | bull/bear trap, failed breakout, breakout-retest, sweep-and-reclaim | 假突破 多头陷阱 突破回踩 扫高收回 |

### T3 K线结构与位置语境

| 联想锚点 | 英文检索词 | 中文检索词 |
|---|---|---|
| 摆动点 | swing high/low, Williams fractal, pivot high/low, zigzag, local extremum, N-bar confirmation | 分形 摆动高低点 之字转向 局部极值 |
| 位势 | support/resistance, S/R flip, role reversal, retest, range boundary, range position, prior day/week high-low | 支撑 阻力 支撑阻力互换 前高前低 区间位置 区间边界 |
| 压缩-扩张 | NR4, NR7, inside bar squeeze, BB width percentile, volatility contraction/expansion, coiled spring, breakout from compression | 窄幅整理 四内七内 波动收缩 扩张 布林收口 蓄力 |
| 多周期 | multi-timeframe alignment, HTF bias, top-down, higher timeframe confluence, aggregate/resample | 多周期共振 大周期定向 自上而下 |
| 时段 | session killzone (Asia/London/NY), hour of day, funding rate settlement (00/08/16 UTC), Asian range breakout | 交易时段 亚盘 伦敦盘 纽约盘 资金费结算 时段效应 |
| 状态 | volatility regime, trend/range regime, ATR percentile, efficiency ratio (Kaufman ER), Hurst exponent | 波动率状态 趋势震荡 regime 效率系数 |

---

## 2. 检索轮次与结果（A-2）

| 轮次 | 目的 | 主要产出 | 采信结果 |
|---|---|---|---|
| R1 定义/判据 | 找每个词的精确 OHLCV 判据 | TA-Lib 形态识别函数族、StockCharts 形态词典、Wikipedia、TraderMade、IG | TA-Lib 作为**经典形态判据的事实标准**（61 个 CDL* 函数，C 实现可查）；其余作交叉 |
| R2 统计基准 | 找"形态成功率/平均增益/失败率"的成规模统计 | Bulkowski《Encyclopedia of Candlestick Charts》、ChartScout 转述、Pipster | **降级为 Tier B**——见 §3 与 §5 说明；未取得可核实的逐形态数值表，故本文不引用任何具体胜率 |
| R3 反例/证伪 | 主动找"形态无效"的证据 | SAGE Open《Profitability of Candlestick Charting Patterns…》、Finance Research Letters《A reality check on trading rule performance in the cryptocurrency market》、ResearchGate 日本股市研究、Eugster (EFM 2018) | 采信为核心先验：**学术侧共识偏向"多数形态无预测力"**；见 §4 |
| R4 中文术语落地 | 中文 SMC 生态的定义 | 知乎 BOS/OB/FVG、币安广场 SMC 结构交易、EBM SMC 策略、腾讯新闻订单区块、B站订单流六概念 | **全部 Tier C**——只取"能否机检"，不取任何效果声称 |
| R5 补漏：结构类判据 | 摆动点/压缩的量化口径 | LuxAlgo（Williams Fractal、NR4/NR7）、LinnSoft、Forex Factory、pyquantlab VSA | 采信为 Tier B+（判据清晰、社区/教材一致）；分形 k=2 标准、NR7=近 7 根最窄幅 |
| R6 补漏：时段效应 | 资金费率与小时效应的学术证据 | MDPI《Temporal Dynamics of Market Microstructure in Cryptocurrency》、arXiv《The Quarter-Hour Effect》、Wiley《Arbitrage, contract design, and market structure in Bitcoin futures》 | **Tier A**：证实"hour-of-day 存在与资金费结算时点相关的系统模式"、"资金费一般每 8 小时结算" |

未能取得全文的文献（抓取被 403/404 阻断）：MDPI 2079-3197/12/7/132、SAGE 2158244017736799、CRAN candlewick vignette。
这几篇**只采信搜索摘要中出现的表述**，并在 §4 逐条标注"摘要级采信"。

---

## 3. 来源分级与采信边界

**Tier A —— 可复现研究，披露样本区间与检验方法**
用途：作为"该不该期待形态有效"的**先验**，以及验证方法论的正误。
- `journals.sagepub.com/doi/full/10.1177/2158244017736799` — 加密货币所在市场形态盈利性检验。摘要级采信其结论表述：*"binomial tests also confirm that most candlestick patterns, even the ones with significant mean returns, cannot reliably predict market directions"*。
  → **直接对应本任务的裁决设计**：均值收益显著 ≠ 方向可预测；本任务必须同时报 `win_rate` 与 `expectancy/avg_win/avg_loss`，不得只看其一。
- `sciencedirect.com/science/article/abs/pii/S1544612320304414` — 加密市场交易规则的 reality check（数据窥探校正）。
  → 支撑本任务的多重检验纪律（BH-FDR + 冻结 holdout 只触碰一次）。
- `mdpi.com/2227-7072/14/5/103` — 加密微观结构的日内动态：*"hour-of-day analysis reveals a systematic pattern linked to funding rate settlement times"*。
  → 支撑 T3 时段类形态纳入验证（`funding_slot`、`slot_in_1h/4h`）。
- `arxiv.org/html/2607.09426v2` — 周期性算法交易与开盘可预测性（15 分钟效应）。
  → 支撑"分钟格点对齐效应"作为对照组而非魔法。
- `onlinelibrary.wiley.com/doi/10.1002/fut.22305` — 资金费一般每 8 小时结算，各所时点不同。
  → 支撑 00/08/16 UTC 槽位定义（与本项目 `funding_slot` 口径一致）。
- `github.com/mrjbq7/ta-lib/issues/87` — TA-Lib 部分形态识别函数实现有误的公开 issue。
  → **重要告警**：即便以 TA-Lib 为判据蓝本，也不可声称"与 TA-Lib 一致"即为正确；本任务自持实现 + 自证因果性。

**Tier B —— 成体系的定义/统计，但口径与本任务不同**
用途：**采信定义，不采信胜率数字**。
- Bulkowski《Encyclopedia of Candlestick Charts》— 形态统计的原典。其数字基于**美股、日频、无衍生品费用结构、无 ±1 tick 平盘判定**，且样本年代早于加密市场。
  → 本任务**不引用其任何胜率数字**（未取得可核实的原始数值表，转述链过长）。ChartScout 的转述本身也承认：*"Every Bulkowski success rate in this cheat sheet comes from stock market data… crypto is not"*。
- TA-Lib `CDL*` 函数族（`ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html`）— 经典形态判据的事实标准；本任务 T1 判据的主要蓝本。
- StockCharts Candlestick Pattern Dictionary、Wikipedia `Candlestick pattern`、IG、TraderMade、QuestDB — 交叉核对用。
- LuxAlgo（Williams Fractal / NR4-NR7）、LinnSoft、Forex Factory、pyquantlab（VSA rolling 实现）— T3 结构类判据来源。

**Tier C —— 教学/社区/自媒体**
用途：只在**能翻译成布尔判据**时纳入，且判据一律标注为"本文操作化定义"。
- 知乎/币安广场/腾讯新闻/EBM/ACY/ThinkMarkets/TradeForGood/Pipcy/FBS/Alchemy/Ultima/B站 等 SMC 内容。
- wyckoffanalytics、CMC Markets（Wyckoff 阶段与 UTAD 定义相对最严谨，仍属 Tier C 教学）。

**关键采信纪律**：SMC 生态对同一概念（尤其 Order Block、CHoCH）**没有统一的数值定义**，不同来源给的确认根数、影线容忍度、时间回溯窗互不相同。因此本任务输出的 T2 形态结果，**衡量的是"该概念的一种可机检化实现"的表现，不是该概念本身的表现**。这一条必须出现在最终报告结论里。

---

## 4. 与本项目既有事实的对齐点

调研结论必须让位于本仓库已有证据，以下几条优先级高于任何外部资料：

1. **本项目实测入场报价推翻"0.50 打平"假设**。`src/binance_predict/db/models.py:761` 原文：
   *「盈亏平衡胜率 p\* = (entry+0.01)/0.98 随报价漂移，无真实报价则胜率优势无法折算 EV（实测记录期开盘价中位 DOWN=0.615 而非 0.5）」*。
   → 打平胜率不是 `stats.py` 常数派生的 52.04%，按实测报价应为 **63.8%**（q̄=0.615）乃至 **75.5%**（S5 延迟入场实测 q̄=0.73，见 `scripts/local_s5_real_quote_ev.py:6-8`）。
2. **本项目已有同构台架的负结果**。`.kiro/specs/scientific-discovery/design.md:66-68`：
   *「lift 初筛存活者费后 EV CI 下界>0 的为 0 个，全窗初筛 lift 与 EV 相关性仅 r=0.16」*。
   → 与 Tier A 的 SAGE 结论方向一致。本任务的**默认预期是"绝大多数形态不可交易"**；若报告呈现大量"高胜率形态"，应先怀疑口径而非先庆祝。
3. **形态定义敏感性已知会翻转结论**。`scripts/local_candle_def_scan.py:84-91` 已用 6 套定义（D0~D5，影线倍率与振幅下限不同）扫描并记录结论翻转；`scripts/local_candle_pattern_check.py:31-34` 有既用阈值 `MIN_RANGE_PCT=0.04 / WICK_BODY_RATIO=2.0 / WICK_RANGE_MIN=0.35 / BODY_RANGE_MIN=0.15`。
   → 本任务的判据风格沿用该参数化（实体/振幅占比、影线/实体倍率），并做 ±1 档敏感性。
4. **部分"新"形态在本项目已是实盘逻辑，不得计作独立新证据**。`services/fake_breakout_detector.py:142-192`（`classify_close_pattern`，S1/S5 在跑的"光头阳/长上影拒绝"）；`config/discovery_rounds/r3_regime_conditioned.json` 的 R3-001~004 已注册 `breakout_high/low_50`、`sweep_hi/lo_fail` × `regime_vol_high`。
   → 注册表用 `overlap_with` 字段逐条标注。
5. **本项目已冻结的裸K几何原子是本任务的最佳交叉基准**：`discovery/features.py:203/204/207/210/211/213/214/216-217/218-221/276-277`
   （`doji: body_r<=0.1`、`hammer_geometry: lo_r>=0.6 & body_r<=0.3 & up_r<=0.15`、`inside_bar/outside_bar`、`bullish/bearish_engulfing`、`marubozu: body_r>=0.9`、`tweezer_*`、`sweep_hi/lo_fail`）。
   → 同义形态的掩码应与之一致；不一致即为待查项（但**不修改 `features.py`**，它是生产依赖）。

---

## 5. 未纳入与部分纳入清单（诚实披露）

### 5.1 完全未纳入（写不出逐根 OHLCV 布尔判据）

| 术语 | 未纳入理由 |
|---|---|
| Wyckoff 阶段 A–E / 完整吸筹派发图 | 需要跨数十至数百根人工划定"交易区间 + 事件序列 + 原因"，事件次序的定义依赖分析者对"背景"的主观判断；无社区统一数值口径，无法唯一机检 |
| Judas swing | 依附于"真实开盘价"这一日级概念，且要求先验知道当日区间与假突破方向 |
| "背景 / context"、"叙事"、"结构质量" | 本质是分析者的综合判断，未定义 |
| Support/Resistance 的"强弱/测试次数"打分 | 各来源给的是定性权重，无数值化规范 |
| Volume profile / POC / VAH / VAL | 需要分价成交分布，本项目数据只有单根总量 `volume`，无逐笔/分价数据 |
| Footprint / delta / CVD / 订单流不平衡 | 同上，缺逐笔买卖方向数据 |
| Demand/supply zone 的"基-趋势-基"结构 | 需人工识别 zone 的起止与新鲜度，各来源阈值互斥 |
| "机构意图""聪明钱在做什么" | 不可证伪 |

### 5.2 已机检化但与社区定义有偏差（本任务定义为准，报告中不得声称等价）

| 术语 | 社区定义的歧义点 | 本文操作化选择 |
|---|---|---|
| Order Block | 有的用"最后一根反向 K"，有的用"位移前整段 base"，有的只取实体，mitigation 判定有 50% 与 100% 两派 | 取"突破前最近一根反向 K"的 `l..h` 全区间；mitigation = 后续首次价格回到区间内（需右移确认） |
| CHoCH vs BOS | 是否要求先有确认摆动点、是否允许用影线破位，两派 | 统一要求**已确认摆动点**（分形 k=2 → 右移 2 根）+ **收盘价破位**（影线不算） |
| FVG | 三根式（`l[t]>h[t-2]`）与两根式缺口并存；"填充"有 50%/100% 两口径 | 取三根式（社区主流，且可纯逐根判定）；填充口径单独标注 |
| Spring / UTAD | 是否要求先有 TR、"收回区间内"的边界容忍度 | 简化为"破前 N 根区间低/高后收回，且量能萎缩"，**明确声明这是 sweep 的区间版变体，不代表 Wyckoff 原义** |
| Effort vs Result | "narrow spread" 与"high volume" 的量化门各源不同 | 用振幅分位 × 量分位的联合分位判定，并做敏感性 |
| 2B 法则 | 原始出处（Connors）口径与流传版本不一致 | 取流传版：创新低后于下一根收回前低上方 → 押反转 |
| Quasimodo | 各源对第三段回抽目标位（前低 vs 结构低点）不一致 | 取"HH → 破结构低 → 回抽至结构低附近"的三段摆动实现 |
| 多周期共振 | 用什么周期、什么算"同向"（收盘 vs EMA vs ADX） | 用聚合后父周期的当根方向作为 bias，且父根**必须已收盘**（右移对齐） |

### 5.3 检索未取得可核实数值的项

Bulkowski 的逐形态「成功率 / 平均增益 / 失败率 / 回调率」表格：检索到的多为 PDF 镜像与二手转述，未取得可信的一手数值表。
→ **本任务不引用这些数字，也不与外部胜率做对照**。形态有效性一律以本项目数据实测为准。

---

## 6. 调研产出的采信出口

- `pattern_catalog.csv`：形态 → 判据 → 方向 → 来源 URL → tier → status(VERIFIABLE / UNSPECIFIED) → overlap_with
- `../../../config/naked_k_patterns.json`：注册表（唯一口径源，冻结后不可改）
- `REPORT.md`：实证结果（本项目数据 + 本项目费用结构），与本文档的采信边界互相引用
