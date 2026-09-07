# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

单人操盘者（项目所有者本人），在自己的 BTC 预测市场自动化交易系统上做实盘决策。无多租户、无角色分层——鉴权是单一 `LOGIN_PASSWORD`（`main.py:1394-1420`）。

使用情境：一天中多次、每次几分钟的高频回看，以及出现异常时的长时间诊断。屏幕上同时是真实资金的状态，因此**读数错误会直接导致亏钱**。

在这个界面上做的工作（按频率排序）：

1. **盯实盘**——当前余额、今日成交、各通道开关/金额/护栏、未结算敞口
2. **调参数**——通道开关、单笔金额、`max_exec_price` 护栏、日单量上限（经 `POST /api/live/toggle` 热调，落 `live_channel_overrides` 表）
3. **复盘信号**——胜率/EV/盈亏平衡/与冻结基准的偏离、逐单 PnL 曲线、资金流水
4. **诊断异常**——弃单原因、对账状态、日志尾部、结算是否卡住

## Product Purpose

把一个后台自动运行的多通道实盘交易系统变成**可读、可信、可操作**的界面。系统本身在 VPS 上自主下单；这个界面是操盘者唯一的观察与干预窗口（本地 SSH 到 VPS 不通，生产运维只有 HTTP API 和 GitHub Actions 两条路）。

成功意味着：操盘者能在几秒内判断「现在系统在做什么、赚还是亏、有没有异常」，并且能安全地改参数而不必登录服务器。

## Positioning

这不是一个通用交易面板，而是**这一个特定系统的自省界面**：它展示的是「13 个注册通道 / 12 个当前启用 / 8 个已退役」这套自有信号体系的真实前向表现，包括对自己不利的证据（样本量不足、置信区间、被护栏拒收的单、已退役通道的亏损）。

界面里的每个数字都能追溯到一次 API 调用或一张 DB 表。**它不美化自己的策略。**

## Operating Context

- 后端 FastAPI（`src/binance_predict/main.py`，~4150 行），56 个 `/api/*` 端点
- 前端 React 19 + Vite 8 + TypeScript 6 + Tailwind CSS v4 + recharts 3，纯 CSR，无路由库
- 5 个 tab：`market` 市场情绪 / `agent` Agent自进化 / `monitor` 运行监控 / `analysis` 信号分析 / `live` 实盘交易（`TAB_IDS`，`App.tsx:2138`）
- 轮询驱动：分析面板 60s，行情采样 15s
- 部署：`push main` → CI（lint + test）→ 构建镜像 → SSH 部署 VPS。**前端改动一旦推 main 就上线**
- 市场节奏：5m 与 15m 两种预测市场，资金日内快速循环（今日 `todayTotalCost` 可超过钱包余额）

## Capabilities and Constraints

**必须保留的硬约束**（来自 AGENTS.md 与代码事实）：

- **UI 全中文**。通道名、指标名配 `HelpHint`（`?` 悬浮解释，`App.tsx:594-601`）
- **`SIGNAL_INFO`（`App.tsx:609-695`）是通道中文名与说明的唯一事实源**，必须与后端 `live_channels.py` 注册表对齐；新增通道两边同步加
- **已退役通道的条目必须保留**——历史订单行仍带这些 `signal_version`，删了会让订单表退回成裸版本串（`App.tsx:606-608`）
- **recharts 曲线的 dataKey 就是 label 本身**：改 `SHADOW_META`/`SCENE_META` 的 label 会同时改曲线与图例。改 color 安全，改 label 不安全
- **后端新字段先进 interface**（过渡可用 `Record<string, unknown>`，热路径要收敛成类型）
- **`frontend/src/App.tsx` 是前端唯一源文件**，新组件进此文件，沿用「分块注释 + 就近放置」格局 —— 用户已确认本次改造**继续遵守**，不拆目录
- 构建验证：`cd frontend; npm run build`（含 tsc 类型检查）；lint：`npm run lint`（oxlint）

**语义事实，不是样式，改造中不得翻转**：

- **UP = 绿色，DOWN = 红色**（`DirectionBadge`，`App.tsx:551-563`）。这是西方约定而非中国「红涨绿跌」，但它是系统既有的、用户已习惯的语义。翻转会导致方向误读
- `StatusDot`：ok=绿 / 非 ok=红；`StatusBadge`：ACTIVE 绿 / EVOLVING 黄 / RETIRED 灰
- `SIGNAL_KIND_BADGE`：实盘 绿 / 影子 紫 / 场景 蓝
- 金额单位是 **wei**，展示层一律 `/1e18`

**明确未决 / 本次不解决**：

- 暗色模式此前是坏的（`App.tsx` 里 0 个 `dark:` 变体 + 533 处写死浅色 Tailwind 灰阶），本次改造通过「全部走 CSS 变量 token」修复
- 基础字号 `:root` 18px（≤1024px 时 16px）**与 atoms.dev 的 body 14px/21px 不一致**；用户已决定本次不动字号，以限制回归面。这是一处**已知且被保留的偏差**

## Brand Commitments

- **视觉语言：atoms.dev**（源站 `https://atoms.dev/zh`）。2026-08-22 由用户确认，记录在 `docs/research/atoms-dev-c15a3ca0/zh-d83abd2e/HOST_AUDIT.md`
- **绑定范围**：只迁移设计语言（IBM Plex Sans/Mono、`#4267FF` 品牌蓝、`#F6F6F6` 底色、16/24px 圆角、pill 按钮）。**不新增落地页、不迁移任何营销组件**（pricing / FAQ / CTA / trust-logos / footer 等 13 个组件 spec 一律不移植）
- **字体已装**：`@fontsource/ibm-plex-sans`、`@fontsource/ibm-plex-mono`（`package.json`），且已在 `index.css` 的 `@theme` 中启用
- **中文回退栈必须显式指定**：IBM Plex 不覆盖中文 glyph，须回退到 `"PingFang SC", "Microsoft YaHei"`（`DESIGN_TOKENS.md:20` 明确要求）
- 无 logo、无吉祥物、无品牌插画资产。`src/assets/hero.png` 是脚手架遗留

## Evidence on Hand

**可用的真实数据（界面必须展示这些，不得编造替代品）：**

- 生产 API：`http://165.154.147.155:8082/api`，Bearer = `LOGIN_PASSWORD`（`.env` 中 `API_AUTH_TOKEN` 为空，故 `_require_auth` 走开发模式放行，实际只有一层鉴权）
- 全量订单：`GET /api/trades/fund-flow?limit=5000`（唯一无 100 条上限的出口）
- 通道级真金 PnL：`GET /api/live/pnl-curve`
- 信号聚合（胜率/EV/盈亏平衡/基准/偏离/regime）：`GET /api/signals/analytics`
- 冻结基准字典：`SHADOW_BENCH`（`main.py:3292-3357`），26 个影子版本的 (胜率, EV, 说明+样本量)
- 分析产物：`output/signal_optimization_2026-09/`（2026-09-06 的十维度证据扫描）

**不得编造的内容：**

- 任何收益数字、胜率、EV、回撤、样本量——必须来自 API 或 DB
- 任何用户评价、背书、客户名、benchmark 对比
- 任何「已验证」「稳定盈利」式的定性宣称。系统的真实状态是：样本量小（3 个通道 n=9~11）、利润高度集中（13 天里 2 天贡献 78%）、当前处于活跃回撤中

## Product Principles

1. **数字是主角，装饰不与它争。** 这是一个读数的界面，任何视觉元素若不与「更快更准地读数」相关，就是噪音。
2. **不确定要可见。** 样本量、置信区间、数据口径（`price_kind` 是 `fill` 还是 `quote`）与被隐藏的数字同等重要。界面不制造虚假确定性。
3. **语义色不可协商。** UP/DOWN、盈/亏、启用/退役、实盘/影子的颜色映射是事实层，改样式时保持映射不变。
4. **改动不能破坏可追溯性。** 通道名、版本名、指标口径与后端注册表一一对应；退役条目保留供历史订单对账。
5. **克制的界面配得上克制的策略。** 系统本身的风险来自过度自信；界面不该在视觉上放大确定性。

## Accessibility & Inclusion

未建立产品特定的无障碍标准（无 WCAG 等级要求、无辅助技术用户记录）。

已知的相关事实：

- `HelpHint` 已支持 `tabIndex={0}` + `group-focus-within`，键盘可触发（`App.tsx:596-598`）——这一行为须保留
- 语义色承载方向信息（UP 绿 / DOWN 红），因此**不能只靠颜色**：`DirectionBadge` 已同时带 `↑`/`↓` 箭头与「看涨/看跌」文字，须保留这种冗余
- 全中文界面，中文字形回退栈必须显式（见 Brand Commitments）
