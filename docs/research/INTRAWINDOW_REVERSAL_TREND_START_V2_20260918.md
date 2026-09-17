# BTC 预测市场：盘中情绪反转与小趋势启动研究 V2

> 分支：`research/reversal-continuation-paths`  
> 研究问题：极低报价后的可交易反转；未来 3/5/8 根多数同向的小趋势启动  
> 原则：先发散、后收敛；所有盘中特征严格截断到决策点；长影只作事后归因  
> 状态：研究完成；没有新策略达到上线或前向影子晋级门槛

## 1. 这次如何纠正研究对象

上一版把“最终形成长影的 K 线”当作信号，在 K 线收盘后预测下一根。这不符合真实机会：真正的赔率优势发生在**当前预测窗口尚未结束、行情已把某一侧 token 打到极低价**的时候。最终长影只是这一过程的事后投影。

V2 将每个反转事件重建为严格时间带：

1. 只读取窗口开始前的 K 线上下文；
2. 逐个读取当前窗口截至事件点的真实 UP/DOWN 报价和 BTC 价格；
3. 按可执行规则定义入场：首次触达 q 阈值、触达后反弹、反弹后二次试探；
4. 入场价就是事件点真实 token 报价；
5. 窗口结束后才读取结算方向与最终长影标签。

因此，最终 long wick 字段在特征字典中明确标为 `retrospective_label_only`，禁止进入实时规则。

趋势方向也改为真正的“小趋势”目标：启动 K 收盘后，未来 H=3/5/8 根中：

- 简单多数同向；
- 强多数同向：3/3、4/5、6/8；
- 同向比例；
- 最长连续 run；
- 沿启动方向的净位移；
- 假设每根等额下注时的 episode 总收益。

## 2. 数据、时钟与功效边界

### 盘中反转数据

- 5m：14,660 个真实报价窗口，约每 15 秒一个采样点，2026-07-26 至 2026-09-15。
- 15m：3,175 个窗口，约每窗 60 点，2026-08-13 至 2026-09-15。
- 每个窗口两侧分别评估 q≤0.02/0.05/0.08/0.10/0.15/0.20。
- BTC 盘中价格是否可用单独标记；BTC recovery/背离机制只在事件时已有至少两个 BTC 点时判断。
- 按自然时间 50% discovery / 25% validation / 25% sealed holdout。由于全部历史只有约 7 周，sealed holdout 仅 2–3 周、1 个月，天然无法满足预注册的 ≥8 周晋级门槛。

### 趋势启动数据

- 5m：208,280 根；15m：69,426 根；重复 0、缺口 0。
- 同样使用 50/25/25 自然时间切分。
- 价格 K 线功效充足；情绪确认维度只在有真实报价窗口的尾部历史可用，不能等同 720 天覆盖。

## 3. 发散假设空间

### 3.1 盘中极低价反转

研究没有只扫“长影比例”，而是覆盖十类机制：

1. q 深度与费用后打平概率；
2. 极端形成速度、加速度、jerk；
3. 触极端后的回收、V/W、二次试探；
4. 极端停留、穿越与 turning count；
5. BTC—token 背离；
6. BTC 极值后的回吐比例，即长影形成过程；
7. 前置 1/3/5 根方向、streak、ATR、压缩、1h/4h 状态；
8. 报价和、参与者与成交额变化；
9. UP/DOWN 镜像；
10. 信息正确、过度反应、噪声、流动性异常四个竞争解释。

单因子预算：918 个预注册测试，按 discovery/validation 分开并做 BH-FDR。只有存活单因子才允许进入二因子组合；本次没有任何规则同时满足预设的 D2→D3 入场门槛，所以组合搜索诚实地停在 0，而不是降低标准硬凑组合。

### 3.2 小趋势启动

覆盖：几何、突破、压缩释放、动量效率、HH/HL 或 LH/LL、波动、量能、多周期共振、高周期 slot、已完成窗口情绪确认、streak 对照和启动/确认层级。单因子 120 个；没有实际启动机制同时在 discovery/validation 形成可靠正增量，因此未进入二因子组合。

## 4. 盘中反转结果

### 4.1 描述性规律：低价并不自动等于便宜

极低 token 的真实胜率本来就很低，必须比较 `P(win)` 与费用后打平率 `q/0.98`，不能看裸胜率。

5m q≤0.10 首次触达的总体图显示：

- DOWN 侧早/中段在部分 discovery/validation 桶中有正校准残差；
- UP 侧在窗口后 80% 以后明显低于隐含概率，越接近结算越像“信息正确地归零”，不是错杀；
- 同一个 q、同一个时点，UP/DOWN 明显不对称，强行镜像会掩盖方向结构。

这支持用户的核心洞察：**机会不在长影本身，而在低价形成的条件和轨迹。**但也说明“报价低”必须配路径条件。

### 4.2 最强 pre-holdout 机制：5m DOWN 首触 0.10 + BTC recovery

定义：

- 当前 5m 窗口 DOWN token 首次 q≤0.10；
- 截至触发时，BTC 已先向 DOWN 方向走到不利极值，随后从该极值回收至少 25%；
- 当时立即按真实 DOWN 报价买入。

Discovery：

- n=318；胜率 12.89%；均价约 0.0736；费用后打平率约 7.51%；
- EV +1.102R/笔；day bootstrap CI [+0.330,+2.195]；FDR q=0.0378。

Validation：

- n=166；胜率 16.87%；均价约 0.0760；打平率约 7.75%；
- EV +1.428R/笔；day CI [+0.755,+2.187]；FDR q=0.00445。

这是一条非常有吸引力、机制清晰的历史规律：**BTC 已经回收，但 DOWN token 仍被压在 ≤0.10，市场情绪可能反应过度。**

### 4.3 Sealed holdout：主候选失效

Holdout 在冻结规则后只打开一次：

- n=198；胜率 6.06%；均价 0.07495；费用后打平率 7.65%；
- 校准残差 -1.59pp；EV **-0.289R/笔**；
- day CI [-0.645,+0.113]；FDR q=0.800；
- 仅 2 周、1 个月。

结论：该条件表现出明显 regime drift，不能晋级。它是值得继续前向收集的**科学候选**，但不是可交易结论。

### 4.4 非对称性与确认成本

- BOTH 合并版 holdout EV +0.020R，CI 跨 0；说明 DOWN 历史优势不能泛化成对称规律。
- 镜像 UP + BTC recovery 在 holdout 点估计 EV +0.410R，但 pre-holdout discovery/validation 不稳定、CI 跨 0；不能倒过来事后选 UP。
- BTC—情绪背离子集 holdout EV -0.276R，失败。
- 等待报价从低点反弹 0.03 后再买（rebound）holdout EV -0.073R；确认损失的赔率没有换来稳定胜率增益。
- 事后形成长影的比例在候选中约 25%–43%，证明长影只是部分事件的表现，不是机制全体。

### 4.5 15m

15m DOWN q≤0.05 晚期首触在 discovery/validation 有正点估计，但只有约 2 周有效分段；holdout 同样无法达到 ≥8 周/≥60 独立跨周事件的科研门槛。UP 镜像为负。15m 只能列为待采集假设。

## 5. 小趋势启动结果

### 5.1 科学假设被大范围证伪

实际机制维度——强实体、收盘极端、range expansion、volume expansion、20/50 根突破、压缩释放、路径效率、HH/HL、动量加速、1h/4h 共振、情绪确认——没有在 discovery/validation 中形成稳定、显著的未来 majority 增量。

最典型的冻结反例：

#### 5m 突破前 20 根极值

- H=3 简单 majority：44.98%，较基线 -4.07pp；
- H=5：44.20%，-4.44pp；
- H=8：30.40%，-4.21pp；
- 每个 episode 等额逐根下注的 CI 全部显著为负。

#### 15m 突破前 20 根极值

- H=3：41.95%，-6.62pp；
- H=5：41.53%，-7.01pp；
- H=8：28.44%，-5.67pp；
- 强 majority 也显著低于基线。

这不是“找不到足够精确的前置条件”，而是在当前维度预算下，**突破本身更像短周期耗竭/吸收，而不是小趋势启动。**

### 5.2 压缩释放、range expansion、情绪确认

- 5m 压缩→扩张在 H=3/5/8 的简单 majority 均低于基线；只有 H=3 的 3/3 强 majority 点估计 +1.14pp，但 H=5/8 失效，且 episode 收益不稳定。
- 15m UP range expansion 在 H=3/5/8 分别低于基线约 1.21/3.95/2.42pp；强 majority 同样没有优势。
- 已完成窗口情绪方向≥0.65 只有约 0.2–1.8pp 的小幅点估计，逐根 episode 收益仍为负或 CI 跨 0，不足以构成策略。

### 5.3 streak1 的小增量为什么不能交易

“第一根同色 K”控制组在大样本中出现约 0.6–1.2pp majority lift，部分 p 值很小；但：

- 同向比例仍约 49.5%–49.9%，小于 50%；
- 未来每根等额下注的 episode return 为负；
- 沿启动方向的净位移接近 0 或为负；
- 统计显著来自巨大样本量，经济效应很小。

因此它不是用户期待的“后面大部分 K 大概率同向”的实用启动条件。

## 6. 科学归因

### 6.1 极低价反转为什么会漂移

最合理的竞争解释排序：

1. **regime drift**：pre-holdout 的 DOWN recovery 可能集中在特定两周；holdout 只有 9 月一个月，效应翻转。
2. **方向性市场偏差**：UP/DOWN 不是镜像；BTC 在特定市场阶段的下跌/回收结构不同。
3. **报价不是纯概率**：盘口薄、买卖方向、延迟和 UP+DOWN spread 会影响 token 价格。
4. **回收阈值仍过粗**：25% recovery 把真正 V 型拒绝与缓慢回抽混在一起；但不能在 holdout 后继续调阈值。
5. **样本相关与短覆盖**：历史只有约 7 周，任何漂亮的两段结果都可能是阶段性。

### 6.2 趋势启动为什么难找

1. BTC 短周期存在明显的突破后均值回归；
2. 波动聚类不等于方向自相关；
3. “未来多数同向”是高噪声标签，启动 K 的几何信息可能不足；
4. 真正趋势可能由外部信息流/订单簿失衡驱动，OHLCV 与 15 秒预测报价不能完整观测；
5. 即使 majority 有 1pp 增量，逐根下注仍会被错向 episode 的相关损失抵消。

## 7. 下一步最有价值的研究方向

### 7.1 反转方向：保留一个前向科研标签

建议只前向记录，不下真钱：

> 5m，任一侧首次 q≤0.10；BTC 已向该侧不利方向走出至少 0.5 ATR，随后回收 ≥25%；记录 side、首次触达时间、实时 q、BTC 极值与回收、报价和、15/30/60 秒后 q，以及最终结算。

为什么仍值得记录：

- 机制合理；
- discovery/validation 很强；
- sealed holdout 明确失败，正需要独立新数据判断是永久失效还是 regime 条件缺失。

建议至少新增：

- **8–12 周**连续前向数据；
- 每侧 ≥200 个独立 first-hit 事件；
- 至少 2–3 个不同波动 regime；
- 禁止期间调阈值。

基于 pre-holdout 观察效应的粗略 80% 功效估计约需 196 个事件/段；若真实效应更接近 holdout UP 侧的小幅残差，则需约 1,145 个事件，说明短期不应仓促裁决。

下一轮可在新数据开始前预注册两个竞争模型，而不是继续网格搜索：

- M1 过度反应：深极端 + 快速形成 + BTC V 型回收 + token 滞后；
- M2 信息正确：深极端 + BTC 不回收或继续创新极值 + token 锁定。

### 7.2 趋势方向：先暂停 K 线/情绪维度扩搜

当前广泛维度都没有提供实用的小趋势启动信号。继续组合这些弱原子只会过拟合。若要再研究，必须引入新的信息源，而不是更多阈值：

- 盘口 order-book imbalance / aggressive buy-sell flow；
- 永续合约资金费率、open interest、清算流；
- 跨交易所同步突破；
- 新闻/宏观事件时钟。

在没有这些新信息前，不建议新增趋势启动影子通道。

## 8. 产物与复现

### 脚本

- `scripts/research_intrawindow_reversal_v2.py`
- `scripts/research_trend_start_v2.py`

### 反转产物

`output/intrawindow_reversal_v2_20260918/`

- `intrawindow_event_band_preholdout.csv.gz`
- `feature_dictionary.json`
- `descriptive_map_preholdout.csv`
- `single_factor_preholdout.csv`
- `combo_preholdout.csv`
- `hypothesis_registry.json`
- `intrawindow_holdout_results.csv`
- `intrawindow_holdout_events.csv.gz`
- `intrawindow_holdout_regimes.csv`
- `data_audit.json`

### 趋势产物

`output/trend_start_v2_20260918/`

- `trend_event_band_preholdout.csv.gz`
- `feature_dictionary.json`
- `trend_single_factor_preholdout.csv`
- `trend_combo_preholdout.csv`
- `hypothesis_registry.json`
- `trend_holdout_results.csv`
- `trend_holdout_events.csv.gz`
- `trend_holdout_regimes.csv`
- `data_audit.json`

### 复现

```powershell
.venv\Scripts\python.exe scripts/research_intrawindow_reversal_v2.py --stage self-check
.venv\Scripts\python.exe scripts/research_intrawindow_reversal_v2.py --stage baseline
.venv\Scripts\python.exe scripts/research_intrawindow_reversal_v2.py --stage final --registry output/intrawindow_reversal_v2_20260918/hypothesis_registry.json

.venv\Scripts\python.exe scripts/research_trend_start_v2.py --stage self-check
.venv\Scripts\python.exe scripts/research_trend_start_v2.py --stage baseline
.venv\Scripts\python.exe scripts/research_trend_start_v2.py --stage final --registry output/trend_start_v2_20260918/hypothesis_registry.json
```

两个 registry 都已记录 holdout 打开时间；对同一 registry 再运行 final 会被拒绝。

## 9. 最终裁决

1. **你的反转研究方向是成立的研究问题**：极低报价 + 可实时观测的回收/背离路径，比“最终长影形态”更接近可交易机制。
2. **目前没有可上线答案**：最强历史规律在 sealed holdout 中翻转为负，必须前向收集。
3. **长影可保留为解释标签**：它能帮助复盘哪些极端最终形成拒绝，但不能进入实时模型。
4. **趋势启动方向当前证据不支持**：广泛 K 线、结构、量价、多周期和情绪维度均未产生稳定、经济上有意义的 future-majority 优势。
5. **下一步应是少数冻结假设的前向采集，而不是继续扩大阈值和组合搜索。**
