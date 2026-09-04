# Binance Prediction 量化交易系统

BTC 预测市场（Binance Prediction Markets，5m / 15m 二元方向市场）的**策略研究 + 实盘执行系统**。
**真金实盘在跑**：`push main` 即自动部署到 VPS，通道开启后会用真实 USDT 下单。

核心方法论是三段式流水线，任何策略都必须逐级通过才能碰到钱：

```text
① 离线科学发现          ② 影子前向验证            ③ 小金额实盘
discovery/ + scripts/    6 个 shadow detector      MultiLiveTrader
720d K线 / 报价分箱      只记录不下注               15 通道，默认全 OFF
BH-FDR 多重检验校正  ->  攒 2~3 周线上样本     ->   人工 promote 才开火
holdout 只触碰一次       与下单路径物理隔离         每通道独立金额/日限/护栏
```

> **命名遗留说明**：仓库名 `binance-predict`、`pyproject.toml` 描述「LLM 全自动预测系统 V2」、
> `main.py` 文档串「V3」、以及下文提到的「情绪 Agent / 系统 B」，指的都是**已退役的旧架构**。
> 详见文末[已退役组件](#已退役组件)。当前唯一的自动决策引擎是 `MultiLiveTrader`。

---

## 交易安全（改动前必读）

### 三道默认关闭的闸门

| 闸门 | 默认值 | 说明 |
| --- | --- | --- |
| `AGENT_AUTO_TRADE` | `false` | 旧 Agent Loop 的自动交易开关；该 Loop 本身已退役，此开关当前不生效 |
| `LIVE_CHANNELS_JSON` | 空 | **实盘真正的总闸**。空 = 15 通道全部 `enabled=False` |
| `live_channel_overrides` 表 | 空 | 前端 toggle 的持久化层，优先级**高于** env；删表行才回落 env |

**开启实盘的前提**：`BINANCE_API_KEY` / `BINANCE_API_SECRET`（需币安后台开通 Prediction Trading 权限）
\+ `PREDICTION_WALLET_ADDRESS` / `PREDICTION_WALLET_ID` 就绪，且预测钱包已有余额（划转端点或币安 App）。

### 硬护栏：不靠自律靠拒启

配置非法时 `MultiLiveTrader` 构造抛 `ValueError`，**拒绝装配实盘但不拖垮其他服务**：

- 单笔金额硬上限 `MAX_ORDER_AMOUNT_USDT = 50.0`（`live_channels.py`）
- 日单量硬上限 `MAX_DAILY_ORDERS_CAP = 500`
- 未知通道名、`max_daily_orders` 传小数/布尔、护栏越界 `[0.01, 0.99]` → 一律拒启

### 这些机制**不是**仓位或亏损保护

执行价护栏、日单量上限、同窗互斥组只约束单笔入口，**没有**总敞口上限、没有止损、没有回撤熔断。
实盘前请自行确认 API 权限范围、钱包余额、单笔金额与风险承受范围。

### 交易语义陷阱（改下单/结算代码前必读）

- `amount_in` 单位是 **wei**：展示层一律 `/1e18`
- 执行价护栏判定**含贴线**（`avg_price >= max_exec_price` 即弃单）：贴线时滑点空间为 0，
  币安拒收 `slippageBps=0`（错误码 `-1102`）
- 失败单 `amount_in="0"`、`token_id=""`（未提交到币安）；`error_message` 可能是泛化文案，
  真实原因要看 `place_order` 层日志
- 划转（transfer-in / out）**不落库**；资金类统计口径从订单派生
- 每通道每窗口至多一单；15m 市场必须先登记进 `trader._15m_markets`，否则被锚定守卫拒单
- **结算口径分流**：5m 订单回读 `sentiment_windows`，15m 订单回读 `fake_breakout_signals`。
  15m 周期起点与 5m 窗口起点数值重合（900s 网格 ⊂ 300s 网格），走错表会被同名 5m 窗错口径结算
- 日单量护栏按 **UTC 自然日**（`date_trunc('day', now())`）计数，北京时间早上 8 点重置；
  而 `late_night_*` 的时段门禁 `HOUR_GUARDS` 用**北京时间**。两套口径并存，改动时注意

---

## 运行架构

### 数据源

```text
Binance Spot WS bookTicker  -> best_bid/best_ask -> mid_price
                               -> 情绪窗口 entry_price / exit_price（结算依据）
                               -> 各影子检测器的实时裁决价

Binance Prediction REST     -> 15 秒采样 -> prediction_market_samples
                             -> 5 分钟归档 -> sentiment_windows（含报价/参与者/成交量曲线）
                             -> 市场列表 / token_id / 报价 / 下单 / 对账
```

### 常驻后台任务（`main.py` lifespan）

| 任务 | 职责 |
| --- | --- |
| `spot_ws` | 现货 bookTicker 长连接，维护 `collector.store.mid_price` |
| `pm_tracker` | 预测市场 15s 采样落库 |
| `pm_15m_edge_accel` | 15m 周期边界加速采样 |
| `sw_archiver` | 5m 情绪窗口归档 + outcome 标注（**场景信号的位势数据源，独立于 Agent 开关**） |
| `health_monitor` | 60s 轮询健康报告，300s 落库快照，CRITICAL/WARN 分级告警 |
| `binance_reconcile` | 币安侧订单对账 |

### 启动期自愈（幂等，失败不阻塞启动）

| 自愈 | 对应事故 |
| --- | --- |
| `repair_contaminated_archives` | 历史 5m 归档窗曾被 15m 样本污染，从原始采样重建曲线 |
| `heal_entry_break_windows` | 部署重启落在窗中会污染 `entry_price` → outcome 翻转 → 订单误结算（2026-08-27） |
| `reconcile_pending_predictions` | 调度队列非持久，重启后孤儿预测回填（仅 Agent Loop 开启时） |
| `_heal_loop` | `signal_id` 关联自愈扫描（重启/影子延迟不丢对账） |

---

## 实盘通道注册表（15 通道）

**口径源是 `services/live_channels.py`**（前端 `SIGNAL_INFO` 常量须与之同步）。
三族触发机制：

- **quote_edge 族**（5m）：采样循环喂价 → 窗内报价区间命中
- **x4 族**（5m）：轮询 `misalignment_signals` PENDING → 次窗 +150s 决策点下单
- **scene 族**（15m）：`fake_breakout_detector` fire 钩子 → 次周期开盘下单（S5 为 +5min 确认入场）

| 通道 | 族 | 周期 | 方向 | 护栏 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `quote_momentum_v1` | quote_edge | 5m | DOWN | 0.78 | 报价动量（A 格顺势），t∈[90,120)s × q∈[0.69,0.75) |
| `quote_momentum_v2` | quote_edge | 5m | DOWN | 0.78 | v1 + BTC 门禁 `chg ≤ −0.10%` |
| `quote_momentum_v3` | quote_edge | 5m | DOWN | 0.78 | v1 + 末收 15m 非连涨门禁（K 线异步核验） |
| `quote_contrarian_v1` | quote_edge | 5m | DOWN | 0.28 | 报价反向（B 格逆势），t∈[45,60)s × q∈[0.15,0.25) |
| `quote_contrarian_v2` | quote_edge | 5m | DOWN | 0.28 | v1 + BTC 门禁 `chg < +0.10%` |
| `quote_contrarian_v3a` | quote_edge | 5m | DOWN | 0.28 | v1 + v2 门禁 + 前窗 DOWN（异步 DB 核验） |
| `quote_contrarian_v3b` | quote_edge | 5m | DOWN | 0.28 | v1 + v2 门禁 + 距日高回落 ≥0.30% |
| `quote_contrarian_v4` | quote_edge | 5m | DOWN | 0.28 | v1 + `ret24 ≤ −1.0%` regime 门禁 |
| `x4_v1` | x4 | 5m | DOWN | 0.45 | 情绪错位（收阳押次窗 DOWN） |
| `x4_v2` | x4 | 5m | DOWN | 0.50 | 情绪错位 · 平静市门禁版 |
| `scene_bull_exhaust` | scene | 15m | DOWN | 0.70 | 场景 S1 多头耗尽 |
| `scene_bull_exhaust_confirm` | scene | 15m | DOWN | 0.75 | 场景 S5 +5min 确认入场 |
| `s5_deep_z20_v1` | scene | 15m | DOWN | 0.88 | S5 深档（`z5 ≤ −20bp` 子集） |
| `scene_bear_exhaust` | scene | 15m | UP | 0.65 | 场景 S2 空头耗尽 |
| `scene_momentum_fade` | scene | 15m | DOWN | 0.55 | 场景 S4 动量衰竭 |

**同窗互斥组**（组内每市场窗口至多一个通道成交，防同源通道叠加敞口）：

- `{quote_momentum_v1, quote_momentum_v2, quote_momentum_v3}`
- `{scene_bull_exhaust_confirm, s5_deep_z20_v1}`

护栏数值依据是盈亏平衡入场价 `entry* = wr × (1 − FEE)`（干净口径历史胜率），设在平衡价附近或略下方。
**部分护栏的样本量偏小**（如 S1 的 0.70 依据 41 个信号中 8 个的胜率），属探索性设定，
配合 2 USDT 小金额前向验证。全部护栏可被 `LIVE_CHANNELS_JSON` 或 DB 覆盖层按通道改写。

### 配置分层

```text
代码默认（全 OFF）
  -> LIVE_CHANNELS_JSON（env，启动基线，重启保持）
    -> live_channel_overrides 表（前端 toggle 持久化，优先级最高，重启不丢）
```

---

## 信号检测器（7 个：6 影子 + 1 场景）

lifespan 共启动 7 个检测器。其中 **6 个是影子**（只记录不下注），与下单路径**物理隔离**：
其信号表不进 `X4_VERSIONS` / `LIVE_CHANNELS` 白名单，从代码结构上保证影子不可能触发实盘。
新研究线要上线必须显式改注册表。

第 7 个是 `FakeBreakoutDetector`（场景检测器），**它是实盘 scene 族的触发源**，详见表后说明。

| 检测器 | 落表 | version | 开关 |
| --- | --- | --- | --- |
| `QuoteEdgeDetector` | `pattern_shadow_signals` | `quote_momentum_v1/v2/v3`、`quote_contrarian_v1/v2/v3a/v3b/v4`、`late_night_contrarian_v1/v2` | `QUOTE_EDGE_ENABLED` |
| `MisalignmentDetector` | `misalignment_signals` | `x4_v1`、`x4_v2` | `MISALIGNMENT_ENABLED` |
| `KlineShadowDetector` | `kline_shadow_signals` | `krev_a_v1`、`krev_b_v1` | `KLINE_SHADOW_ENABLED` |
| `ReversalShadowDetector` | `kline_shadow_signals` | `rev_p1_v1`、`rev_p2_v1` | `REVERSAL_SHADOW_ENABLED` |
| `NextbarShadowDetector` | `kline_shadow_signals` | `nb_zschamp_15m_v1`、`nb_smaslope_5m_v1` | `NEXTBAR_SHADOW_ENABLED` |
| `HmShadowDetector` | `pattern_shadow_signals` | `hm_touch_down_v1`、`hm_touch_down_v2` | `HM_SHADOW_ENABLED` |
| ⚠️ `FakeBreakoutDetector`（场景，**非影子**） | `fake_breakout_signals` | S1/S2/S4/S5 + `s5_deep_z20_v1` | `FAKE_BREAKOUT_ENABLED` |

`kline_shadow_signals` 被 KREV / 反转 / nextbar 三族共用，靠 `version`（+ nextbar 另加 `timeframe`）
隔离，各检测器只认自己的 version 结算——防止 5m 次根错结 15m 信号。

上表开关全部**代码默认 `true`**（部署即生效，仅作紧急停用制动力），可经同名 env 覆盖为 `false`。
注意这 7 个开关**未收录进 `.env.example`**——线上 `.env` 缺行时走代码默认值，排查行为时以
`config/settings.py` 为准。唯一例外是 `KLINE_SHADOW_EMAIL_ENABLED`（默认 `false`，影子期静默防轰炸）。

**注意**：`FakeBreakoutDetector` 是例外。它本身「信号模式不下注」，但其 fire 钩子
（`_on_signal_fired` / `_on_s5_deep_fired`）接的是 `MultiLiveTrader.on_scene_signal`，
**scene 族通道开启后它会触发真实下单**。场景检测器关闭时钩子为 `None`，场景通道即使开启也无触发源（fail-safe）。

### 研究纪律

`discovery/` 离线流水线（`data → features → hypotheses → l1_tester → combo_search → oos_validator → report`）：

- 时序切分 0.6 / 0.2 / 0.2（发现 / 验证 / 冻结 holdout），**holdout 只触碰一次**
- 漏斗：L1 单因子（**BH-FDR q=0.1**）→ L2 双因子 → L3 三因子 → 冻结 → holdout 终验
- 防泄漏铁律：所有滚动特征只用「不含当前根」的前置窗口（`prev_*` 系列先右移一根）
- 经济口径：`FEE=2%` + `PREMIUM=0.01`，打平胜率 ≈52.04%
- **经济闸**（`ev_gate.py`）：EV bootstrap CI 下界 > 0 才允许直上线，否则降级 OBSERVE。
  二元预测市场里入场价 ≈ 群众隐含概率，是盈利的唯一诚实基准；lift 只是必要非充分条件
- **只加不改**：v2/v3/v4 永远是 v1 ∩ 新门禁，绝不修改已验证的冻结区间

---

## 快速开始

环境：Python 3.11 + [uv](https://docs.astral.sh/uv/)；数据库 PostgreSQL 16 + TimescaleDB。

```powershell
uv sync --all-extras
Copy-Item .env.example .env
# 编辑 .env：填数据库、Binance Key、LOGIN_PASSWORD
# 保持 LIVE_CHANNELS_JSON 为空（= 全通道 OFF），先跑影子观察
uv run alembic upgrade head
uv run python main.py
```

Docker 本地全栈（db + backend + frontend）：

```powershell
docker compose up -d
# backend :8002 -> 容器 :8000，frontend :5173 -> 容器 :80
```

前端开发：

```powershell
Set-Location frontend
npm install
npm run dev          # Vite :5173
npm run lint         # oxlint
```

### 访问认证

所有 `/api/*` 请求须携带 `Authorization: Bearer <LOGIN_PASSWORD>`。
放行例外：非 `/api` 路径、`OPTIONS` 预检、`/api/auth/login`、`/api/health`（容器探针）。
`LOGIN_PASSWORD` 未配置时**一律 401**（不做「空值即放行」的开发旁路）。

前端登录成功后把密码本身作为 token 存 localStorage（不过期）。
`API_AUTH_TOKEN` 是敏感端点的第二层 Bearer 校验，**为空时该层直接放行**——生产环境建议配置。

---

## 数据库迁移

**应用启动不会执行 `alembic upgrade head`。** lifespan 里会跑 `Base.metadata.create_all`
\+ 一批幂等 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`（存量 dev 库安全网，与迁移等价，
异常静默跳过），因此开发环境可以裸启动；**生产部署仍须人工迁移**。

```powershell
uv run alembic current
uv run alembic upgrade head      # 部署前先备份数据库
```

> ⚠️ `b2c3d4e5f6a7_drop_path_a_tables.py` 会**永久删除**路径 A 的 7 张遗留表：
> `predictions`、`prediction_results`、`feature_snapshots`、`custom_rules`、
> `rule_versions`、`prompt_versions`、`review_memories`。
> 没有确认备份前不要执行 `upgrade head`。

### 16 张表

| 分类 | 表 |
| --- | --- |
| 行情/窗口 | `prediction_market_samples`（15s 原始采样，默认永不删除）、`sentiment_windows`（5m 归档 + 曲线） |
| 交易 | `trade_orders`（订单 + 结算字段）、`live_channel_overrides`（通道配置 DB 覆盖层） |
| 信号 | `fake_breakout_signals`（场景）、`misalignment_signals`（X4）、`pattern_shadow_signals`（报价 edge / HM）、`kline_shadow_signals`（KREV / 反转 / nextbar） |
| Agent Loop（退役，只读存档） | `pattern_memory`、`agent_predictions`、`pattern_change_log`、`pattern_backtest_runs`、`binning_snapshots` |
| 研究/运维 | `scene_param_versions`、`llm_traces`、`health_snapshots` |

---

## API

55 个端点。`[auth]` = 额外受 `_require_auth`（`API_AUTH_TOKEN`）保护；全部端点都受登录中间件保护。

### 认证与运维

| 端点 | 说明 |
| --- | --- |
| `POST /api/auth/login` | 密码换 token |
| `GET /api/health` | 服务与中间价状态（**免认证**，供探针） |
| `GET /api/agent/health` | Agent 运行健康报告 `[auth]` |
| `GET /api/logs/tail` | 日志尾部 `[auth]` |

### 实盘交易（**会动真钱**）

| 端点 | 说明 |
| --- | --- |
| `POST /api/live/toggle` | **通道开关 / 金额 / 日限 / 护栏热调**，持久化到 DB `[auth]` |
| `POST /api/trade/test` | 手动测试下单 `[auth]` |
| `GET /api/live/pnl-curve` | 实盘盈亏曲线 `[auth]` |
| `GET /api/trades/latest` · `/recent` · `/fund-flow` | 订单查询（`recent` limit ≤ 100）`[auth]` |
| `GET /api/trades/binance-history` | 币安侧订单历史 `[auth]` |
| `POST /api/trades/sync-binance` · `/sync-status` · `/settle-scan` | 对账 / 状态同步 / 结算扫描 `[auth]` |
| `GET /api/prediction-wallet` | 预测钱包信息 `[auth]` |
| `POST /api/prediction/transfer-in` · `/transfer-out` | **资金划转**（不落库）`[auth]` |
| `GET /api/prediction/redeemable` · `POST /api/prediction/redeem` | 奖金领取 `[auth]` |
| `GET /api/prediction/quote-preview` | 报价预览 `[auth]` |
| `GET /api/prediction-markets` · `/all` | 活跃 / 全部预测市场 `[auth]` |

### 信号与研究

| 端点 | 说明 |
| --- | --- |
| `GET /api/signals/analytics` | 全信号族统计聚合 |
| `GET /api/fake-breakout/status` · `/signals` · `/stats` | 场景检测器状态 / 信号 / 统计 |
| `GET /api/fake-breakout/signals/{id}/path` | 单信号价格路径 |
| `GET /api/misalignment/signals` | X4 影子信号 |
| `GET /api/scene/versions` | 场景参数版本 `[auth]` |
| `POST /api/scene/versions/{id}/adjudicate` · `/promote` | 裁决 / **人工晋升上线** `[auth]` |

### 图表

| 端点 | 说明 |
| --- | --- |
| `GET /api/chart/prediction-market` | 当前 5m 市场报价曲线 |
| `GET /api/chart/prediction-market/15m` | 当前 15m 市场报价曲线 |
| `GET /api/chart/btc-klines` | BTC K 线（`interval` / `limit`） |
| `GET /api/sentiment/windows` | 已归档情绪窗口 `[auth]` |

### Agent Loop（退役，只读存档）

`GET /api/sentiment/agent/predictions` · `/patterns` · `/patterns/{id}/history` · `/status` ·
`/metrics` · `/evolution`；`POST /api/sentiment/agent/deep-learn`（+ `/stream` `/commit`
`/pycluster` `/compare`）；`GET /api/sentiment/agent/deep-learn/compare/live`；
`GET /api/llm/traces` · `/traces/{id}`；`POST /api/agent/patterns/reevaluate`；
`GET /api/agent/patterns/backtest-runs` · `/compare`。均 `[auth]`。

`agent_loop_enabled=False` 时这些端点仍返回历史数据（表保留只读），但不再产生新记录。

### 已废弃端点

`POST /api/sentiment/backtest` 与 `POST /api/sentiment/predict` 已退役，
仅返回 `{"status": "deprecated", ...}` 提示以保持前端兼容，**不做任何计算**。

`POST /api/sentiment/momentum-predict` 是纯算法手动分析接口（不落库、不下单）`[auth]`。

---

## 项目结构

```text
src/binance_predict/
├── main.py                 # FastAPI 装配 + lifespan + 55 个端点（~4100 行单文件）
├── config/settings.py      # Pydantic Settings，全部开关与阈值（~450 行，注释含决策依据）
├── db/
│   ├── engine.py           # async engine + session factory
│   └── models.py           # 16 张表 ORM（~1250 行）
├── models/schemas.py       # Pydantic 契约
├── services/               # 业务核心（42 个模块）
│   ├── live_channels.py    # ★ 15 通道注册表 + 配置解析（口径源）
│   ├── multi_live_trader.py# ★ 实盘调度：三族触发 / 日限 / 互斥组 / 门禁核验
│   ├── prediction_trading.py# ★ 下单链路：占位→报价→护栏→币安下单→落库（~1750 行）
│   ├── trade_settler.py    # 结算：5m/15m 口径分流
│   ├── quote_edge_detector.py   # 报价 edge 影子 + 冻结区间/门禁注册表
│   ├── fake_breakout_detector.py# 场景检测器（S1/S2/S4/S5，~1650 行）
│   ├── misalignment_detector.py # X4 影子
│   ├── kline_shadow_detector.py # KREV 影子
│   ├── reversal_shadow_detector.py / nextbar_shadow_detector.py / hm_shadow_detector.py
│   ├── archive_contamination_repair.py  # 归档污染 + 断链自愈
│   ├── data_collector.py   # 现货 WS bookTicker -> mid_price
│   ├── prediction_market_data.py # 预测市场 REST
│   ├── ev_gate.py          # 经济闸（EV bootstrap CI）
│   ├── predicates.py / symbolizer.py / curve_features.py  # 谓词求值内核
│   ├── risk_control.py / alerting.py / signal_notify.py / health.py / metrics.py
│   ├── research_scheduler.py / scene_researcher.py / hypothesis_*.py
│   └── sentiment_agent.py  # 退役的 Agent Loop（~3200 行，只读存档）
├── discovery/              # K 线科学发现流水线（纯函数无 I/O，分层单向依赖）
│   ├── data.py features.py hypotheses.py targets.py
│   ├── l1_tester.py combo_search.py oos_validator.py report.py
├── backtest/               # 回测内核（data / events / stats / surface）
└── prompts/                # 退役 Agent Loop 的提示词模板

frontend/src/App.tsx        # 前端唯一源文件（~4700 行，React 19 + Tailwind 4 + recharts）
alembic/versions/           # 32 个迁移
scripts/                    # 99 个一次性研究/评估脚本（.py，另有 .sh/.ps1；非运行时依赖）
tests/                      # pytest + pytest-asyncio + hypothesis（50 文件 / 793 用例）
docs/prd/ docs/research/    # PRD v1~v3 与研究线文档
output/ logs/ .pytest_tmp/  # 运行产物与工作草稿区（gitignored，不清理不提交）
```

### 前端约定

- UI 全中文；通道名/指标配 `HelpHint`（`?` 悬浮解释）
- `SIGNAL_INFO` 常量是通道中文名与说明的**唯一事实源**，须与后端 `live_channels.py` 对齐——新通道两边同步加
- recharts 曲线 `dataKey` 用 label 本身：改 `SHADOW_META` / `SCENE_META` 的 label 会同时改曲线与图例
- 新组件进 `App.tsx`，沿用「分块注释 + 就近放置」格局
- 后端新字段：先同步进 interface（过渡可用 `Record<string, unknown>`，热路径要收敛成类型）

---

## 验证

```powershell
# 后端：必须用 venv 的 python（系统 Python 没装本包，会 ModuleNotFoundError）
.venv\Scripts\python.exe -m pytest tests/ -q          # 793 用例，约 75s
.venv\Scripts\python.exe -m pytest tests/test_multi_live_trader.py -q   # 实盘链路

# 前端：build 含 tsc 类型检查
Set-Location frontend; npm run build; npm run lint
```

> Windows + PowerShell 环境**无 `&&`**，用 `;` 分隔命令。

**测试要求**：交易链路（`prediction_trading` / `multi_live_trader`）行为改动，
必须在 `tests/test_multi_live_trader.py` 加或改用例；护栏/边界类改动加贴线、越界正反例。
实盘链路测试模式参照该文件的 `_make_real_trader(monkeypatch)` 替身（不触网络/真实 DB）。

---

## 部署

`push main` = 自动部署。GitHub Actions（`.github/workflows/deploy.yml`）：

```text
Test Gate（uv sync --frozen + pytest tests/ 全量）
  -> Build & Push（backend / frontend 镜像 -> GHCR，tag: latest + git sha）
    -> Deploy to VPS（SSH：拉镜像 -> docker compose up -d -> 健康探活重试 30s）
```

- `concurrency.cancel-in-progress: false`：部署不可取消，避免镜像不一致
- 生产 `.env` 由 `ENV_FILE_CONTENT` secret（base64）写入：**先解码到临时文件并校验非空，
  成功后才原子替换**，避免解码失败清空线上 `.env`
- 部署脚本会强制改写若干 env 行（如 `AGENT_ALERT_NOTIFY_ENABLED=false`、删除历史残留的
  `KLINE_SHADOW_ENABLED=false`）。**排查线上行为时须把这一层计入**，它绕过配置分层设计
- `docker/docker-compose.prod.yml`、`docker/*.Dockerfile`、`nginx.conf` 非必要不改动

生产运维只走两条通道：HTTP API（地址与 Bearer token 在本地 `.env`）或 GitHub Actions。
**本地 SSH 到 VPS 不通**，不要尝试直连生产。

---

## 已退役组件

### 路径 A（LLM 直接预测）— 已物理删除

迁移 `b2c3d4e5f6a7_drop_path_a_tables.py` 删除了 7 张表：`predictions`、`prediction_results`、
`feature_snapshots`、`custom_rules`、`rule_versions`、`prompt_versions`、`review_memories`。

### 系统 B / 情绪曲线 Agent Loop — 2026-08-16 退役

`settings.agent_loop_enabled = False`。lifespan **不实例化** `SentimentAgent` / `AgentScheduler`，
全局引用保持 `None`（`tracker` / `archiver` 的 publish 调用已有 None 守卫，事件静默跳过）。

原设计是 Learn → Predict → Validate → Evolve 的 LLM 自进化闭环，配 `PatternMemory` 模式池
（S/A/B/C 分级 + 定期重回测）。退役原因：LLM 决策链路的 token 成本与延迟不可控，
且统计功效不足以支撑实盘。

**代码与表全部保留**（`sentiment_agent.py`、`agent_scheduler.py`、`agent_logic.py`、
`llm_service.py`、`prompts/`、`pattern_memory` 等 5 张表），翻回 `True` 即恢复。
相关端点仍返回历史数据但不再产生新记录。

**情绪窗口归档器（`sw_archiver`）不受此开关影响，继续运行**——它是场景信号系统的 4h 位势数据源。

### 路径 C — 保留为手动接口

`POST /api/sentiment/momentum-predict`：概率动量纯算法分析，手动触发、不落库、不下单。

### 其他默认关闭

`pattern_reeval_enabled = False`（模式池重回测，随系统 B 一起退役）。
`agent_learn_mode = "manual"`（深度分析改手动触发，控制 token 消耗）。

---

## 相关文档

- `AGENTS.md` — AI 协作约定：命令、架构、交易语义、边界（禁止事项）、Git 交付规范
- `docs/prd/v1.md` `v2.md` `v3.md` — 三代 PRD
- `docs/research/` — 各研究线文档（按研究族分子目录，随研究线增删；`agents-md-spec.md` 为协作规范）
- `docs/runbook-decision-bench.md` — 决策台架 runbook
- `.env.example` — 全部配置项模板（含逐项注释）

<!-- 2026-08-15 deploy note: VPS .env 补 AGENT_ALERT_* 邮件段，触发镜像变更以重建容器加载新 env -->
