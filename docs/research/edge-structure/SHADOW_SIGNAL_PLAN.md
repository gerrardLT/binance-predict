# 边缘结构影子信号（EDG 族）技术方案

> 状态：**方案评审稿**（未写生产代码）。数据依据：`output/edge_structure_{720,2160}d/`。
> 目标：把「高胜率形态 / 确定性结构」研究发现落地为**只记录不下注**的前向影子信号，攒真实样本复核后再议 promote。

---

## 0. 决策依据：2160d 牛熊样本外复核

| 信号类 | 代表条件 | 720d holdout | 2160d 牛熊 holdout | 稳健性判定 |
|---|---|---|---|---|
| **触及确定性** | `range_atr≥1.74` / `volatility_ratio_5_20≥1.45` | touchup_2 rel **+134%** | touchup_2 rel **+139%** | **样本无关铁律**（波动率聚集普适，跨牛熊几乎不变） |
| **方向性（短窗）** | `zscore_10≤-1.65` / `rsi_7≤21.8 ∧ zscore_10≤-1.63` | up_2 P=**0.563** 月1.00 | up_2 P=**0.572** 月0.93 | 跨样本稳健（>52% 打平线） |
| **方向性（长窗）** | `dist_prior_low_atr_10≤0.125 ∧ sma_slope_atr_50` | up_12 P=**0.670** | 未进 top（<57%） | **行情依赖**（下跌市超跌反弹红利，牛市减弱）→ 不作 v1 |

**结论**：
1. **触及路（EDG-T）是最可靠的基础**——rel 提升跨牛熊几乎不变（+134%/+139%），因为波动率聚集是市场普适特性。
2. **方向路（EDG-D）v1 只选短窗 up_2/up_3**（跨样本稳健 57%），长窗 up_12 的 67% 是 720d 下跌市红利，降级为 v2 探索。
3. 两路都建（用户要求），但**方向路必须标注行情依赖**，promote 前需跨行情复核。

---

## 1. 两路影子信号总览

| 路 | version 族 | 规律 | 结算口径 | 数据表 | 可否直接下注 |
|---|---|---|---|---|---|
| **A 方向性** | `edgd_*` | 深度超跌 → 押 UP | H 根后**收盘净方向** c[t+H]>c[t] | `KlineShadowSignal`（version 隔离） | 是（预测市场方向盘） |
| **B 触及确定性** | `edgt_*` | 波动爆发 → 触及 ±1ATR | 未来 H 根 **high/low 触及** c±1×ATR | `PatternShadowSignal`（复用 HM 触及设施） | 否（波动率维度 / regime 过滤器） |

两路均：只记录不下注、物理隔离下单路径、settings 开关默认关、攒 2-3 周人工 promote。

---

## 2. 路 A：方向性影子（EDG-D）

### 2.1 冻结信号（v1，L1 单条件，参数 STABLE、跨样本一致）

```
version:  edgd_up_z10_v1
condition: zscore_10 <= -1.65079327        # 深度超卖（价格远低于 10 根均线 1.65σ）
direction: UP                               # 押涨（均值回归反弹）
horizon:   H = 2                            # 2 根 15m（30min）后结算
```
- 720d holdout：up_2 P=0.563（基准 0.499）rel +13%，月 1.00 / 折 1.00，参数 STABLE
- 2160d 牛熊：同机制 up_2 P=0.572 月 0.93

v2 候选（L3 组合，胜率略高但需防过拟合）：`rsi_7 <= 21.8369614 AND zscore_10 <= -1.63274664 AND dist_prior_low_atr_10 <= …`

### 2.2 结算口径（与回测 `build_edge_targetsets` 的 up_H 逐位一致）
- `target_bar_start = signal_bar_start + H × 900_000`（H=2 → 次次根起点）
- `win = close[target] > close[signal]`（**净涨**，非"次根收阳"）
- 平盘（c==0 差）→ NOISE / EXPIRED；数据断点 → EXPIRED
- 无报价 → `ev_at_entry = NULL`（同 KREV，纯 K 线中性价）

### 2.3 与 KREV 的关键差异（决定实现方式）
| 维度 | KREV（现有） | EDG-D（新） |
|---|---|---|
| 目标 | reversal_1（次根 close>open） | up_H（c[t+H]>c[t]，H=2） |
| 结算窗 | 次根 | 第 H 根 |
| 方向语义 | 硬编码押 UP | 按 spec.direction |
⇒ **KREV 的 `_settle_pending` 硬编码"次根收阳赢"，不可直接复用**，需新 detector 或参数化结算窗（见 §4）。

---

## 3. 路 B：触及确定性影子（EDG-T）

### 3.1 冻结信号（v1，L1 单条件，全 horizon ROBUST）

```
version:  edgt_touch_ratr_v1
condition: range_atr >= 1.74272353          # 当前 K 线波幅 ≥ 1.74×ATR（巨幅 K 线 = 波动爆发）
监测:      未来 2 根内 high ≥ c+1×ATR（touchup）或 low ≤ c-1×ATR（touchdn）
```
- 720d holdout：touchup_2 P=0.441（基准 0.256）rel **+72%**；touchdn_2 P=0.431 rel **+65%**；月/折 1.00
- 2160d 牛熊：`range_atr≥1.73` 同机制 ROBUST，组合版 rel 达 +126%~+139%
- 剂量-反应：沿 range_atr/volume_z 轴 P(触及) 单调 ↑（mono=1.00），阈值无关铁证

### 3.2 结算口径（新，"触及"非"方向"）
- 监测窗 = signal 后 2 根 15m（30min），逐根回读 high/low
- `touched_up = max(high[t+1..t+2]) >= c[t] + 1×ATR[t]`
- `touched_dn = min(low[t+1..t+2]) <= c[t] - 1×ATR[t]`
- ATR 口径：`discovery.features.atr_series(kl)`（前 20 根 range% 均值 × open，**单一事实源**，禁手抄）
- 记录：`touched_up/touched_dn`（可同真=双向都触及）、首个触及时刻/价格

### 3.3 数据模型（推荐复用 PatternShadowSignal）
- HM 检测器已有触及基础设施：`entry_state`(WAITING/TOUCHED)、`touch_ts`、`touch_price`、`_spawn_watcher`
- 复用 `PatternShadowSignal`，`version=edgt_*` 隔离；`win` 语义改为"波动兑现"（touched_up OR touched_dn），或新增 `touched_up/touched_dn` 布尔列（需 alembic 迁移）
- **备选**：新表 `EdgeTouchShadowSignal`（字段 touchup_hit/touchdn_hit/touch_up_price/touch_dn_price/atr_at_signal）——更干净但需建表迁移

### 3.4 用途（不可直接下注）
1. 前向验证"波动率可预测"的真实性（回测 rel+134% 是否线上复现）
2. 作**方向信号的 regime 过滤器**：假设"波动爆发时，方向信号（EDG-D）的 EV 更高"——影子期用 EDG-T∩EDG-D 交集验证

---

## 4. 检测器实现方案

**推荐：新建 `src/binance_predict/services/edge_shadow_detector.py`（EDG 族统一）**
- 复用 KREV/HM 的骨架：`__init__(collector)` / `start`(backscan+loop) / `_poll_once`(60s) / `_evaluate_new_bars` / `_record_signal`(幂等) / `_settle_pending` / `_expire_stale_pending` / `status`
- 求值：`build_feature_matrix`（单一事实源）+ `condition_mask(parse_condition(cond))`（与 KREV 同）
- 两路结算分支：EDG-D 走"H 根后净方向"，EDG-T 走"2 根内触及 ±1ATR"
- 落库：EDG-D → `KlineShadowSignal`（version=edgd_*）；EDG-T → `PatternShadowSignal`（version=edgt_*）
- **不改动** KREV/HM/P1P2 现有检测器（避免污染已上线影子逻辑）

备选：EDG-D 扩展 `kline_shadow_detector`（参数化结算窗）、EDG-T 扩展 `hm_shadow_detector`——改动分散、回归面大，不推荐。

---

## 5. 装配与开关

- `config/settings`：新增 `edge_shadow_enabled: bool = False`（默认关，口径保真测试全绿后 .env 显式开）
- `main.py` lifespan（参照 kline/hm 装配块）：
  ```python
  global edge_shadow_detector
  if settings.edge_shadow_enabled:
      edge_shadow_detector = EdgeShadowDetector(collector=collector)
      await edge_shadow_detector.start()
      logger.info("EDG 边缘结构影子检测器启动（edgd 超跌押UP / edgt 波动爆发触及，影子模式不下注）")
  ```
- 物理隔离：不注册 `LIVE_CHANNELS`、不进 `X4_VERSIONS`，两表不被任何下单代码引用
- 前端 `App.tsx` 影子面板 + `/api/signals/analytics`：新增 edgd_*/edgt_* 版本行（沿用现有聚合，动态发现新版本）

---

## 6. 测试计划（AGENTS.md：交易链路改动必须配测试）

`tests/test_edge_shadow_detector.py`（参照 `test_hm_shadow_detector.py` 替身模式）：
1. **口径保真**（生命线）：同一批 K 线，实时 `condition_mask` 命中 == 回测 `build_edge_targetsets` 的 win 掩码（逐位一致）
2. **EDG-D 触发/结算**：命中 → 幂等落 PENDING（唯一约束 version+signal_bar_start）；H 根后 c[t+H]>c[t] → SETTLED win=True；反之 win=False；平盘 NOISE/EXPIRED
3. **EDG-T 触及结算**：未来 2 根 high≥c+1ATR → touched_up；low≤c-1ATR → touched_dn；双向都触及 → 两者皆真
4. **超时**：target 起点后 4h 未结算 → EXPIRED
5. **缺特征/NaN 保守不触发**
6. **version 隔离**：edgd_*/edgt_* 结算互不污染，且不触碰 KREV/HM/P1P2 的行

---

## 7. 前向验证与 promote 门槛

- 攒 **2-3 周**真实线上样本（EDG-D 理论频率：zscore_10≤-1.65 约 10% 分位，~数条/天；EDG-T：range_atr≥1.74 约数条/天）
- 复核：影子胜率 vs 回测 holdout（EDG-D 偏离 ≤ 若干 pp；EDG-T touched 率 vs 回测 rel）
- EV：EDG-D 接真实报价后算 EV（打平线 52%）；EDG-T∩EDG-D 交集 EV（过滤器增益）
- **人工 promote 门槛**（满足才议上线）：n≥阈值、跨行情胜率稳定、费后 EV>0、口径保真测试全绿

---

## 8. 风险与边界

1. **方向胜率行情依赖**：up_12 在 720d 下跌市 67%，2160d 牛熊仅 57%。v1 只用短窗 up_2/up_3（稳健 57%），promote 前必须跨行情复核。
2. **触及路不可直接下注**：EDG-T 方向不定，仅用于波动率验证 + regime 过滤器，不单独构成预测市场信号。
3. **阈值冻结自发现段**：实时求值必须走 `build_feature_matrix` 单一事实源，禁手抄阈值（口径保真测试兜底）。
4. **描述性 ≠ 可交易**：所有 rel 提升为条件概率，未经报价/成本/滑点/执行延迟检验；EV 需接真实报价前向验证。
5. **不主动 commit/push**：方案确认 + 测试全绿后，等用户明确指令再走 CI 部署。

---

## 附：待用户确认的实现细节
- [ ] EDG-T 数据模型：复用 `PatternShadowSignal`（加 touched_up/dn 列，需 alembic 迁移）vs 新表 `EdgeTouchShadowSignal`
- [ ] EDG-D 持有窗：H=2（30min）v1，是否同时挂 H=3 对照
- [ ] 检测器：新建统一 `edge_shadow_detector.py`（推荐）vs 扩展 KREV/HM
- [ ] 前端影子面板 + analytics 是否本轮一并接入
