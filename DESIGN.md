---
name: binance-predict 实盘监控台
description: BTC 预测市场多通道实盘系统的自省界面——静默、平坦、数字优先
colors:
  brand: "#4267FF"
  brand-hover: "#2F54EB"
  brand-soft: "#E4E9FF"
  ink-95: "rgba(12,12,12,0.95)"
  ink-80: "rgba(12,12,12,0.80)"
  ink-55: "rgba(12,12,12,0.55)"
  ink-40: "rgba(12,12,12,0.40)"
  ink-black: "#0C0C0C"
  bg-base: "#F6F6F6"
  bg-surface: "#FFFFFF"
  bg-sunken: "#F0F0F1"
  border: "#E5E7EB"
  border-soft: "#F0F0F1"
  positive: "#34A853"
  positive-soft: "rgba(52,168,83,0.10)"
  negative: "#D64545"
  negative-soft: "rgba(214,69,69,0.10)"
  warning: "#E8A33D"
  warning-soft: "rgba(232,163,61,0.12)"
  violet: "#B57EDC"
  violet-soft: "rgba(181,126,220,0.12)"
  indigo: "#7C86D8"
  indigo-soft: "rgba(124,134,216,0.12)"
  chart-1: "#4267FF"
  chart-2: "#34A853"
  chart-3: "#E8A33D"
  chart-4: "#B57EDC"
  chart-5: "#7C86D8"
  chart-6: "#D96C8E"
  chart-7: "#4FA8A0"
  chart-8: "#A8763E"
  chart-9: "#8A9A5B"
  chart-10: "#6E7A8A"
typography:
  headline:
    fontFamily: '"IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "24px"
    fontWeight: 600
    lineHeight: 1.18
    letterSpacing: "-0.24px"
  title:
    fontFamily: '"IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.4
  body:
    fontFamily: '"IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "14px"
    fontWeight: 400
    lineHeight: "21px"
  label:
    fontFamily: '"IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif'
    fontSize: "11px"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.1px"
  numeric:
    fontFamily: '"IBM Plex Mono", ui-monospace, Consolas, monospace'
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
    fontFeature: '"tnum" 1'
rounded:
  pill: "9999px"
  panel: "24px"
  card: "16px"
  sm: "8px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.brand}"
    textColor: "#FFFFFF"
    typography: "{typography.body}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  button-primary-hover:
    backgroundColor: "{colors.brand-hover}"
    textColor: "#FFFFFF"
  button-dark:
    backgroundColor: "{colors.ink-black}"
    textColor: "rgba(255,255,255,0.95)"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink-95}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  card:
    backgroundColor: "{colors.bg-surface}"
    rounded: "{rounded.card}"
    padding: "0"
  badge-up:
    backgroundColor: "{colors.positive-soft}"
    textColor: "{colors.positive}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  badge-down:
    backgroundColor: "{colors.negative-soft}"
    textColor: "{colors.negative}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  badge-neutral:
    backgroundColor: "{colors.bg-sunken}"
    textColor: "{colors.ink-55}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  tab-item:
    backgroundColor: "transparent"
    textColor: "{colors.ink-55}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  tab-item-active:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink-95}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  input:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink-95}"
    rounded: "{rounded.sm}"
    padding: "6px 12px"
  tooltip:
    backgroundColor: "{colors.ink-black}"
    textColor: "rgba(255,255,255,0.95)"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
---

# Design System: binance-predict 实盘监控台

## Overview

**Creative North Star: "静默仪表盘 The Quiet Instrument"**

这是一台读数仪器，不是一块营销版面。它的价值全部在于让操盘者在几秒内准确读到「系统现在在做什么、赚还是亏、有没有异常」，因此视觉系统的任务是**消除一切与读数无关的东西**，而不是增添吸引力。

设计语言来自 atoms.dev（2026-08-22 由用户确认绑定，见 `docs/research/atoms-dev-c15a3ca0/zh-d83abd2e/HOST_AUDIT.md`）。atoms 的语言本质是克制的：浅灰底 `#F6F6F6`、纯白面、三级墨色（95/80/55 不透明度）、唯一一个品牌蓝 `#4267FF`、pill 形按钮、16/24px 圆角，以及——实测确认的——**卡片无阴影、无 hover 态**（`BEHAVIORS.md` 的 hover sweep 显示 pricing card 与 FAQ item 在 hover 前后计算样式完全一致）。这套语言天然适合 Operate 模式：它不制造视觉噪音，把注意力让给内容。

本系统在这套语言上叠加一条自己的纪律：**所有数值走等宽字体**。一个显示金额、胜率、价格、盈亏的界面，数字必须纵向对齐才可比读；比例字体的 `1,111.11` 与 `8,888.88` 宽度不同，会破坏扫读。IBM Plex Mono 本来就是 atoms 语言的一部分（用于代码/等宽场景），这里把它提升为数值的主字体。

**Key Characteristics:**

- 完全平坦：无阴影、无渐变、无玻璃拟态、无发光边缘。深度只靠白面/灰底的色调分层传达
- 三级墨色承担全部文字层级，不引入第四级灰
- 品牌蓝是唯一的强调色，只用于「可操作」与「当前选中」
- 语义色（涨跌/盈亏/状态）是事实层，映射固定不可翻转
- 数值一律等宽；文字一律 IBM Plex Sans + 显式中文回退栈
- 卡片静止：hover 不改变卡片本身，只有可点击元素有过渡

## Colors

浅灰底 + 纯白面 + 三级墨色，唯一强调色是品牌蓝；语义色从 atoms 的 agent 卡片横幅色族派生，饱和度统一压到同一档。

### Primary

- **Atoms 品牌蓝** (`#4267FF`)：唯一强调色。用于主操作按钮、当前选中的 tab、关键链接、需要被注意的单个数字。hover 加深到 `#2F54EB`，浅底用 `#E4E9FF`。
- **墨黑** (`#0C0C0C`)：次级按钮与 tooltip 背景。atoms 用它做黑底按钮；在本系统里它同时承担「反白浮层」的角色——tooltip 靠它而非阴影来脱离页面层级。

### Neutral

- **墨 95** (`rgba(12,12,12,0.95)`)：标题、当前选中项、关键数值
- **墨 80** (`rgba(12,12,12,0.80)`)：正文
- **墨 55** (`rgba(12,12,12,0.55)`)：次要文本、未选中 tab、表头、单位后缀
- **墨 40** (`rgba(12,12,12,0.40)`)：占位符、禁用态、空数据的 `—`
- **底灰** (`#F6F6F6`)：页面背景
- **面白** (`#FFFFFF`)：卡片、表格容器、输入框
- **凹陷灰** (`#F0F0F1`)：代码块、内嵌小块、徽章中性底
- **边框** (`#E5E7EB`)：卡片与输入框描边
- **弱边框** (`#F0F0F1`)：卡片内部的分隔线（表头下、行间）

### Semantic

语义色映射的是**事实**，不是装饰。这些映射在本系统里已被用户长期使用，改造中不得翻转：

- **涨/盈利/健康/实盘** (`#34A853`，浅底 `rgba(52,168,83,0.10)`)：UP 方向、正 pnl、`StatusDot` ok、`StatusBadge` ACTIVE、`SIGNAL_KIND_BADGE` 实盘
- **跌/亏损/异常** (`#D64545`，浅底 `rgba(214,69,69,0.10)`)：DOWN 方向、负 pnl、`StatusDot` 非 ok、错误态
- **待定/演进中** (`#E8A33D`，浅底 `rgba(232,163,61,0.12)`)：`StatusBadge` EVOLVING、`ChangeTypeBadge` UPDATE、需要人工确认的状态
- **影子信号** (`#B57EDC`，浅底 `rgba(181,126,220,0.12)`)：`SIGNAL_KIND_BADGE` 影子、`DiscoveryMethodBadge` LLM_DEEP
- **场景信号** (`#7C86D8`，浅底 `rgba(124,134,216,0.12)`)：`SIGNAL_KIND_BADGE` 场景、`ChangeTypeBadge` CREATE
- **退役/中性** (`{colors.bg-sunken}` 底 + `{colors.ink-55}` 字)：RETIRED、LEGACY、NO_TRADE、已退役通道标注

四个语义色相（绿 `#34A853` / 琥珀 `#E8A33D` / 紫 `#B57EDC` / 靛蓝 `#7C86D8`）直接取自 atoms 的 agent 卡片横幅色族（`DESIGN_TOKENS.md:53`），因此与品牌蓝同处一个饱和度与明度档位。红色 `#D64545` 是派生值——atoms 源站没有红色语义，此处按「与 `#34A853` 同明度同饱和度」推导，是本系统唯一的自增色。

**每色两档：canonical + hover（2026-09-07 实施时补充）.** 实施中发现，token 系统每个语义色若只有一档，机械迁移会把 `bg-green-600 hover:bg-green-700` 变成 `bg-positive hover:bg-positive`——**24 个按钮的悬停反馈静默塌缩**，而这在 Operate 模式的仪表盘上是实打实的可点击性损失。因此每个语义色补 `*-hover` 档，色值取自 `.impeccable/design.json` 的 tonal ramp：亮色模式取深一档、暗色模式取浅一档，与既有 `brand-hover` 完全同一推导逻辑，不是新设计。中性按钮另补 `--sunken-hover`（亮 `#DCDCDF` / 暗 `#1C1D24`）。

**teal `#4FA8A0` = PY 聚类方法色（2026-09-07 用户确认保留）.** 它与 `DiscoveryMethodBadge` 的 PY_CLUSTER 徽章同色，3 个 teal 实色按钮全部是 Python 聚类相关动作，颜色是有意的语义呼应而非随手取值。因此 teal 是 Buttons 规格里 Primary / Dark / Ghost 之外**唯一被批准的第四种实色按钮**，且只能用于 PY 聚类语义，不得挪作他用。

### Chart

22 条影子曲线 + 5 条场景曲线需要可区分的分类色。原实现用的是 matplotlib tab10 / ColorBrewer 的高饱和色（`#1f77b4` `#d62728` `#e377c2` `#8c564b` 等），与 atoms 的克制语言冲突。重调后的十色环：

`chart-1 #4267FF` · `chart-2 #34A853` · `chart-3 #E8A33D` · `chart-4 #B57EDC` · `chart-5 #7C86D8` · `chart-6 #D96C8E` · `chart-7 #4FA8A0` · `chart-8 #A8763E` · `chart-9 #8A9A5B` · `chart-10 #6E7A8A`

**The Eleven-Series Rule（2026-09-07 修订）.** 第 11 条及以后的曲线**不新增颜色**，而是复用 `chart-1..10`，并以「降透明度 + 细线」区分：`strokeOpacity 0.55`、`strokeWidth 1.5`（前十条为 `1` / `2`）。

原规则要求叠加 `strokeDasharray="5 3"`，实施时发现不可行：影子图与场景图里 **dash 已经是语义通道**——实线=线上累计胜率、虚线 `5 4`=回测冻结基准、点线 `1 3`=平均盈亏平衡（图例文字明示）。若主曲线也用 dash，第 11 条主曲线会与基准线不可区分，而「对照冻结基准看偏离」正是 PRODUCT.md 的工作 #3。改用透明度/线宽后，dash 语义完整保留，曲线↔基准的归属关系也保住了。

配套要求：**表格侧必须同步淡化**。影子表的版本名（`style={{ color, opacity }}`）与盈亏表的图例圆点若不同步降透明度，第 11 条起在表格里仍会与前三条撞色，图表与表格就对不上号。

同一规则适用于 `PNL_COLORS`（逐通道盈亏曲线，13 通道 > 10 色）。

### Named Rules

**The One Blue Rule.** 品牌蓝 `#4267FF` 只出现在两类地方：可操作元素（按钮、链接、可点击行）与当前选中态（active tab、选中筛选项）。一屏之内蓝色块不超过 3 处。蓝色一旦用于装饰或强调普通文本，它就失去了「这里能点」的含义。

**The Semantic Freeze Rule.** UP=绿、DOWN=红、盈=绿、亏=红、实盘=绿、影子=紫、场景=蓝——这些映射是事实层，任何视觉改造都不得翻转或重新分配。改的是色值（饱和度/明度对齐 atoms），不是含义。

**The No-Fourth-Gray Rule.** 文字层级只有墨 95 / 80 / 55 / 40 四级，不再有第五级。原实现里 `text-gray-300/400/500/600/700/800/900` 七级灰阶是随手取值的结果，收敛到四级后层级关系反而更清楚。

## Typography

**Body Font:** IBM Plex Sans（回退 `"PingFang SC", "Microsoft YaHei", sans-serif`）
**Numeric Font:** IBM Plex Mono（回退 `ui-monospace, Consolas, monospace`）

**Character:** IBM Plex 的工程感与中性度正是这台仪器需要的——它不为文本增添个性，把个性留给数字本身。Sans 承担所有叙述性文字，Mono 承担所有需要纵向比读的数值。两者的 x-height 接近，混排时不会跳动。

中文 glyph 不在 IBM Plex 覆盖范围内，**回退栈必须显式写出 `PingFang SC` / `Microsoft YaHei`**（`DESIGN_TOKENS.md:20` 明确要求），否则 headless/Windows 环境会落到宋体，观感与参照物完全不符。

### Hierarchy

- **Headline**（600, 24px, 1.18, -0.24px）：tab 内的一级区块标题。≤1024px 降为 20px
- **Title**（600, 14px, 1.4）：卡片标题（`Card` 组件的 `h2`）。这是界面里最常见的标题层级
- **Body**（400, 14px, 21px）：正文、表格单元格、说明文字
- **Label**（500, 11px, 1.4, +0.1px）：徽章文字、表头、字段名、单位
- **Numeric**（Mono 500, 13px, 1.4, `tnum`）：金额、价格、胜率、盈亏、EV、时间戳

`:root` 基础字号为 **18px/145%**（≤1024px 时 16px），**与 atoms 的 body 14px/21px 不一致**。这是用户明确决定保留的偏差（本次改造只修明显错误，不动字号，以限制回归面）。实际上绝大多数组件文本用显式 Tailwind 尺寸（`text-xs`/`text-sm`/`text-[11px]`）覆盖了基础值，因此该偏差的影响面主要是未加类的散落文本。

### Named Rules

**The Tabular Numerals Rule.** 任何会被纵向比读的数字——金额、价格、胜率、pnl、EV、样本量、时间戳——必须用 IBM Plex Mono 并开启 `font-variant-numeric: tabular-nums`。这条规则的理由不是美学：比例字体下 `+7.11` 与 `+160.04` 的字符宽度不同，同一列的数字无法对齐，操盘者必须逐行重读小数点位置。

**The Unit Recede Rule.** 单位（`U`、`%`、`USDT`、`单`）用墨 55、比数值小一档，且与数值之间留 2px。数值是内容，单位是注释。

## Layout

单列容器，居中，最大宽度自适应。原实现的 `#root { width: 1126px }` 是落地页遗留——固定宽度在窄屏溢出、在宽屏浪费，改为 `max-width: 1126px; width: 100%`。

**本次修的一处明显错误**：`#root` 上的 `text-align: center` 移除。全局居中对仪表盘是错的（表格、KV 行、日志都需要左对齐），代码里已有 69 处 `text-center`/`text-left` 在逐个对抗它。移除后那些 `text-left` 变成冗余但无害，可在后续清理。

间距节奏沿用 Tailwind 尺度，实际使用集中在 `4 / 8 / 12 / 16 / 24`：

- 卡片内边距 16px（`p-4`），卡片头 `px-4 py-2`
- 卡片之间 16px
- 表格行内 8px 纵向、12px 横向
- 徽章内 2px 纵向、8px 横向
- 区块标题上方 24px、下方 8px

**The Heading Air Rule.** 标题上方的空间必须大于下方。标题属于它引导的那段内容，不属于上一段。

响应式：≤1024px 时基础字号降为 16px、headline 降为 20px、多列卡片降为单列。atoms 源站用 `lt-md:` 前缀断点，本系统沿用 Tailwind 的 `md:` / `lg:`。

## Elevation & Depth

**本系统不使用阴影。** 这是用户明确选择的「完全平坦」，也是 atoms 的实测行为（`BEHAVIORS.md`：pricing card 与 FAQ item 在 hover 前后 `box-shadow` 均为 `none`，计算样式完全一致；`DESIGN_TOKENS.md:59`：卡片「白底，无明显 shadow，border 极浅或无」）。

深度靠**色调分层**传达，三层足够：

1. `bg-base #F6F6F6` —— 页面底
2. `bg-surface #FFFFFF` —— 卡片、表格容器、输入框（比底亮，自然浮起）
3. `bg-sunken #F0F0F1` —— 卡片内的内嵌小块、代码、徽章中性底（比底暗，自然凹陷）

浮层（tooltip、下拉、抽屉 `ChartDrawer`、模态 `EvolutionModal`/`DeepLearnModal`/`CompareModal`）不用阴影脱离页面，而用**反白**：`ink-black #0C0C0C` 底 + `rgba(255,255,255,0.95)` 字。tooltip 已是这个形态（原 `bg-gray-800`），模态与抽屉用 `border` 描边 + `bg-surface` 与页面区分，必要时加 `rgba(12,12,12,0.40)` 的遮罩层——遮罩是层级手段，不是阴影。

`index.css` 现有的 `--shadow` 变量在本次改造中**停止使用**（保留定义以免破坏未知引用，但不再被任何组件消费）。`Card` 组件当前的 `shadow-sm`（`App.tsx:705`）与 `HelpHint` 的 `shadow-xl`（`App.tsx:598`）都要移除。

### Named Rules

**The Flat-By-Default Rule.** 表面静止时一律平坦。没有任何元素因为 hover、focus 或数据更新而获得阴影或位移。

**The Three-Tone Rule.** 深度只有三个明度档：底灰 / 面白 / 凹陷灰。需要第四层时，说明信息结构有问题，应该重新分组而不是再加一层。

## Shapes

圆角只有四个值，全部来自 atoms：

- **pill**（9999px）：按钮、tab、徽章、状态点。atoms 按钮实测 `border-radius: 40px`（全 pill，`DESIGN_TOKENS.md:58`），本系统用 `9999px` 保证任意高度都是完整胶囊
- **panel**（24px）：大容器、模态、抽屉。atoms 终 CTA 面板 24px
- **card**（16px）：卡片、输入框组、图表容器。atoms 卡片实测约 16px，token 名 `--radius-card`
- **sm**（8px）：徽章内的小元素、tooltip、代码块

**退役值已全部清除**（2026-09-07）：`rounded`（4px，73 次）、`rounded-md`（6px，5 次）、`rounded-lg`（8px，31 次）、`rounded-xl`、`rounded-2xl` 均已按元素类型收敛——按钮与徽章 → `pill`，输入框 / 代码块 / 容器 → `sm`，卡片 → `card`，模态 / 抽屉 → `panel`。`rounded-full` 保留（DESIGN.md 明示 `StatusDot` 用它，语义是「正圆」而非「胶囊」）。

**stock 圆角名重映射守卫.** 仅清除代码里的退役写法不够——Tailwind 默认 `xs=2 / sm=4 / md=6 / lg=8 / xl=12 / 2xl=16 / 3xl=24px`，其中 2·4·6·12 都不在四档内。因此 `index.css` 的 `@theme` 里把**全部 stock 圆角名重映射到四档**：`xs/md/lg → sm`、`xl → card`、`2xl/3xl → panel`。这样偏离规格的圆角**无法用 stock 类名表达**，日后谁写一个 `rounded-xl` 也会拿到 16px 而不是默认的 12px。规格在 token 层被守住，不必依赖 code review 去抓。

边框：1px，`border #E5E7EB`（卡片、输入框）或 `border-soft #F0F0F1`（卡片内分隔线）。**不使用 2px 边框**，不使用彩色边框——需要强调时用底色变化而不是描边加粗。

`StatusDot` 保持 `w-2.5 h-2.5 rounded-full`（10px 正圆），这是 atoms 语言里唯一的纯几何元素。

## Components

### Buttons

- **Shape:** 完整胶囊（pill, 9999px）
- **Primary:** 品牌蓝底 `#4267FF` + 白字，`padding: 6px 16px`，字号 14px/500。用于该区域唯一的主动作（登录、同步、提交测试单、领取）
- **Hover / Focus:** 底色加深到 `#2F54EB`；过渡 `color, background-color, border-color 0.2s cubic-bezier(0.4,0,0.2,1)`（atoms 实测的按钮通用过渡）。`focus-visible` 用 2px `brand-soft` 外环，不用浏览器默认蓝框
- **Dark:** 墨黑 `#0C0C0C` 底 + `rgba(255,255,255,0.95)` 字，1px `rgba(12,12,12,0.55)` 边框。用于次级但需要分量的动作
- **Ghost:** 透明底 + 墨 95 字。hover 时文字降到墨 55（atoms nav 链接的实测行为：hover = 降低文本不透明度至 0.55）
- **Disabled:** 墨 40 字 + `bg-sunken` 底，不可点击，无 hover 态

### Tabs

五个 tab（市场情绪 / Agent自进化 / 运行监控 / 信号分析 / 实盘交易）是界面的主导航。

- **未选中:** 透明底 + 墨 55 字，pill 形
- **选中:** `bg-surface` 白底 + 墨 95 字，pill 形。**不用下划线、不用蓝色填充**——白底在灰底上已经足够清楚，且符合 One Blue Rule（蓝色留给可操作按钮）
- **Hover:** 文字从墨 55 降到墨 40
- 过渡同按钮

### Cards / Containers

`Card`（`App.tsx:703-712`）是界面里最主要的容器，出现数十次。

- **Corner Style:** 16px（`rounded-xl`，已映射 atoms radius-card）
- **Background:** `bg-surface #FFFFFF`
- **Shadow Strategy:** **无**。移除现有的 `shadow-sm`
- **Border:** 1px `border #E5E7EB`（当前 `border-gray-200` 的值恰好就是 `#e5e7eb`，语义已对，只需改走 token）
- **Header:** `px-4 py-2`，下边框 1px `border-soft #F0F0F1`，标题用 Title 层级（14px/600/墨 95，当前是 `text-gray-700` → 改为墨 95）
- **Body:** `p-4`，`overflow-auto`
- **Hover:** **无状态变化**。atoms 实测卡片 hover 前后计算样式完全一致；一个数据容器不该因为鼠标划过而反应

### Badges

系统里有六族徽章，全部改为 pill 形 + 语义色浅底 + 语义色文字（去掉边框，浅底已足够区分）：

- `DirectionBadge`：UP `badge-up`（`↑ 看涨`）/ DOWN `badge-down`（`↓ 看跌`）/ NO_TRADE `badge-neutral`（`⊘ 不交易`）。**箭头与文字必须保留**——语义色不能是唯一的信息通道
- `StatusBadge`：ACTIVE 绿 / EVOLVING 琥珀 / RETIRED 中性
- `ChangeTypeBadge`：CREATE 靛蓝 / UPDATE 琥珀 / RETIRE 中性
- `DiscoveryMethodBadge`：LLM_DEEP 紫 / PY_CLUSTER `chart-7` 青 / LEGACY 中性
- `SIGNAL_KIND_BADGE`：实盘 绿 / 影子 紫 / 场景 靛蓝
- `TierBadge`、`PhaseBadge`、`VERDICT_META`：沿用同一套「浅底 + 同色字 + pill」语法，色相从上表里取，不新增

徽章文字用 Label 层级（11px/500）。`DiscoveryMethodBadge` 现在的 `text-[10px]` 归到 11px。

### Inputs / Fields

- **Style:** `bg-surface` 白底，1px `border`，8px 圆角（`rounded-sm`），`padding: 6px 12px`
- **Focus:** 边框转 `brand #4267FF`，外加 2px `brand-soft #E4E9FF` 环。**不用发光、不用阴影**
- **数值输入**（金额、护栏、日限）用 Numeric 字体，右对齐
- **Disabled:** `bg-sunken` 底 + 墨 40 字
- **Error:** 边框转 `negative`，下方一行 11px `negative` 文字说明原因
- **紧凑变体 `.ds-input-sm`（2026-09-07 实施时扩充）:** `padding: 2px 6px`、字号 11px。**只覆写几何**，颜色 / 边框 / focus / disabled 全部继承 `.ds-input`，三类可组合：`ds-input ds-input-sm ds-input-numeric`。

  理由：实盘通道行里的护栏（`w-14`）与单笔金额（`w-16`）输入框处于 13 行的密集列表，套用标准 `6px 12px` 会让每行撑高、面板整体溢出。这是有意的规格扩充，不是偏离。

**实施状态**：7 个文本/数值/密码输入框已全部接线到 `.ds-input`，宽度类（`w-14`/`w-24`/`w-full`…）保留。

**实施中修正的一处机械映射错误**：登录框原为 `focus:border-cyan-500`，被色相表 `cyan → teal` 带偏成 `focus:border-teal`；本规格要求 focus 边框转 `brand`。同一映射还把登录提交按钮（原 `bg-cyan-600`）带成了 teal——登录是该屏唯一主操作，必须是品牌蓝。**教训：色相映射表能正确处理语义色，但会把「恰好用了某个色相的非语义元素」一起带偏，迁移后必须回看主操作与 focus 态。**

### Tooltip（HelpHint）

`HelpHint`（`App.tsx:594-601`）承载通道说明与指标解释，是 AGENTS.md 明确要求保留的组件。

- **触发器:** 14px 正圆，1px `border`，`?` 用墨 55、11px/700
- **气泡:** `ink-black #0C0C0C` 底 + `rgba(255,255,255,0.95)` 字，8px 圆角，`padding: 8px 12px`，宽度 288px（当前 `w-72`），**左对齐**，13px/1.6
- **无阴影**——反白本身就是最强的层级信号，`shadow-xl` 移除
- **可访问性:** 保留 `tabIndex={0}` 与 `group-focus-within:block`，键盘必须能触发。这是既有行为，不得回退

### Charts

recharts 3。

- 曲线色走 `chart-1..10`；第 11 条起复用色相并以 `strokeOpacity 0.55` + `strokeWidth 1.5` 区分（The Eleven-Series Rule，2026-09-07 修订——**不用 dash**，dash 在本系统另有语义）
- **改 color 安全，改 label 不安全**：dataKey 就是 label 本身，改 label 会同时改曲线与图例（AGENTS.md 明示）。实施时以「解析原条目 → 只重赋 color」的方式重写 `SHADOW_META`，并逐字比对 HEAD 确认 22 个 key 与 label 未变
- 表格侧同步：影子表版本名与盈亏表图例圆点必须同序淡化，否则图表与表格对不上号
- 网格线 `border-soft #F0F0F1`，1px，**无虚线**（原为 `strokeDasharray="3 3"`，已全部去除）
- 坐标轴文字用 Label 层级（11px / 墨 55）；**刻度数值用 Numeric**——已通过 `tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }}` 统一（20 处）
- tooltip 沿用反白形态（`ink-black` 底 + `on-ink` 字，无阴影、无边框）。已抽成共享常量 `TOOLTIP_STYLE` / `TOOLTIP_LABEL_STYLE` / `TOOLTIP_ITEM_STYLE`，替换掉原先 11 处各自散写的白底 `contentStyle`
- **参考线保持与所属曲线同色**（本条为 2026-09-07 修订，推翻原「用墨 55 虚线，不用彩色」）：dash 在这两张图里是语义通道——虚线 `5 4`=回测冻结基准、点线 `1 3`=平均盈亏平衡。若参考线一律改墨 55，22 条曲线 × 2 条参考线 = 44 条灰虚线，将无法判断哪条基准属于哪条曲线，而「对照冻结基准看偏离」正是 PRODUCT.md 的工作 #3。**同色是这里的信息，不是装饰。**
  - 例外：与任何曲线无关的**通用参考线**仍用墨 55——50% 胜率中线（`dasharray "4 4"`）、零轴
- **零轴必须可见**：PnL 类图表的 `y=0` 用 1px `border #E5E7EB` **实线**标出，正负区域的意义完全依赖它（原有一处误用 `dasharray "3 3"`，已改实线）

## Do's and Don'ts

### Do:

- **Do** 所有颜色走 CSS 变量 token（`--color-*` / `--ink-*` / `--bg-*` / `--border-*`）。这是暗色模式的**唯一途径**：在 `:root` 里定义亮色档，在 `@media (prefers-color-scheme: dark)` 里定义暗色档，`@theme inline` 桥接成 Tailwind 工具类；因此代码里**零个 `dark:` 变体**也能实现暗色。实施前曾靠 532 处 stock gray 硬抗浅底、0 个暗色覆写，结果是暗色下全乱；现已归零，暗色自动生效。
- **Do** 用四级墨色（95/80/55/40）表达文字层级，不用 Tailwind 的 `gray-300..900`。（已完成：原 532 处 stock 灰阶归零）
- **Do** 所有纵向比读的数字用 IBM Plex Mono + `tabular-nums`。全局 CSS 已给 `.font-mono` 加上 `font-variant-numeric: tabular-nums`，因此既存等宽用法自动合规；sans 里的数字加 `.num` 类即可
- **Do** 保留徽章里的箭头/文字冗余（`↑ 看涨`），让语义色不是唯一信息通道
- **Do** 保留 `HelpHint` 的键盘可触发性。实施后气泡由 `.ds-hint:hover` / `.ds-hint:focus-within` 驱动，`tabIndex={0}` 在 `.ds-hint` 上——**不再依赖 Tailwind 的 `group` / `group-focus-within`**，改写时别把 `tabIndex` 弄丢
- **Do** 卡片保持静止：无 hover 阴影、无 hover 位移、无 hover 边框变色
- **Do** 中文回退栈显式写全 `"PingFang SC", "Microsoft YaHei"`
- **Do** 新组件继续写进 `frontend/src/App.tsx`，沿用「分块注释 + 就近放置」格局（用户已确认本次不拆目录）
- **Do** 语义色成对使用 canonical 与 `*-hover` 档（`bg-positive hover:bg-positive-hover`）。**禁止用 `hover:opacity-90` 代替色值变化**——降透明度会让按钮在浅底上发灰，且与 `disabled` 态混淆
- **Do** token 名与 Tailwind 工具类名的对应以 `index.css` 顶部注释为准（`bg-base→canvas`、`bg-surface→card`、`border→line` 等）
- **Do** 密集列表里的输入框用 `ds-input ds-input-sm ds-input-numeric` 组合，不要各自散写 `px-1 py-0.5 border border-line rounded-sm`

### Don't:

- **Don't** 使用 `shadow-sm` / `shadow-xl` / 任何 `box-shadow`。本系统完全平坦
- **Don't** 翻转语义色映射（UP 绿 / DOWN 红 / 实盘 绿 / 影子 紫 / 场景 蓝）。这是事实层
- **Don't** 用品牌蓝做装饰、强调普通文本、或作图表的主色序列。蓝色只表示「可操作」与「当前选中」，一屏不超过 3 处
- **Don't** 引入第四级灰或第五级墨色
- **Don't** 使用 `rounded`（4px）与 `rounded-md`（6px）。圆角只有 pill / 24 / 16 / 8 四档
- **Don't** 迁移 atoms.dev 的任何营销组件（pricing / FAQ / final-CTA / trust-logos / footer / hero-chat / ai-team-carousel）。用户 2026-08-22 明确排除，`HOST_AUDIT.md:29-31` 有记录
- **Don't** 改 `SHADOW_META` / `SCENE_META` 的 `label` 字段。dataKey 就是 label，改了会同时改曲线与图例
- **Don't** 删除 `SIGNAL_INFO` 里已退役通道的条目。历史订单行仍带这些 `signal_version`
- **Don't** 添加无限循环动画、环境动效、marquee、呼吸效果。这是轮询驱动的数据面板，持续动效会与数据刷新竞争注意力
- **Don't** 用渐变、玻璃拟态（`backdrop-filter`）、发光边缘。atoms 语言里没有这些，本系统也没有它们的位置
- **Don't** 改动基础字号（`:root` 18px）。这是用户明确保留的偏差，动它会把回归面扩大到每一个面板

## Implementation Status（2026-09-07）

### 量化边界对照

| 项 | 原值 | 现值 |
|---|---|---|
| stock 灰阶（`gray-300..900`） | 532 处 / 7 级 | **0** |
| stock 彩色类（`blue/green/red/…-NNN`） | 322 处 | **0** |
| `bg-white` | 33 处 | **0** |
| `box-shadow` 类（代码，非注释） | 18 处 | **0** |
| `backdrop-blur` | 1 处 | **0** |
| 硬编码 hex | 131 处 | **0** |
| 退役圆角名 | `rounded` 73 / `md` 5 / `lg` 31 / `xl` 3 / `2xl` 4 | **0**（按元素类型收敛到四档） |
| hover 塌缩 | 24 处 | **0** |
| `font-mono` 的 tabular-nums | 1 / 119 | **119 / 119**（全局 CSS 规则） |
| `#root` | `width:1126px` + `text-align:center` | `max-width:1126px; width:100%`，无居中 |
| 基础字号 | 18px | 18px（保留，已知偏差） |

CSS 产物从 56.6 kB 降到 46.1 kB——迁离 stock 灰阶/彩色后 Tailwind tree-shake 生效。

### 运行时验证（真实生产数据，2026-09-07）

静态计数只能证明类名被替换，不能证明 token 解析正确。以下是本地预览下的活 DOM 实测。

**预览接线**：本地**不跑后端**——`lifespan` 里 `multi_live_trader.start()` 无开关常开（`main.py:1257`），而本地 `.env` 持生产 DB 与币安密钥，起后端等于与 VPS 实例并发下真单。改为 `vite.config.ts` 的 dev proxy 指向生产 API（AGENTS.md 认可的两条生产通道之一），并加只读守卫：proxy 目标非本地时，放行 GET 与 `POST /api/auth/login`，其余写方法一律 403，防止视觉走查时误点 `/api/live/toggle` 改生产实盘配置。实测：GET 透传 200/147 kB，`POST /api/live/toggle` → 403。

**实测指标**：

| 检查项 | 市场情绪 | 信号分析 | 实盘交易 |
|---|---|---|---|
| tabular-nums 覆盖 | 275 / 275 | 341 / 341 | 557 / 557 |
| `box-shadow` 元素 | 0 | 0 | 0 |
| 遗留 `*-gray-*` 类 | 0 | 0 | 0 |

- Tab 容器实测 `rgb(246,246,246)` = `#F6F6F6`；选中态实测白底 `rgb(255,255,255)` + 墨 95 `rgba(12,12,12,0.95)`，与 design.json 的 `.ds-tab-active` 一致
- 字体栈实测含显式中文回退：`"IBM Plex Sans", …, "PingFang SC", "Microsoft YaHei", sans-serif`
- 语义冻结实测：影子徽章 `rgb(181,126,220)` 紫、场景徽章 `rgb(124,134,216)` 靛蓝
- HelpHint 键盘可达性实测 **35 / 35** 带 `tabindex="0"`（属性在外层 `.ds-hint`，不在 `.ds-hint-trigger`）

**图表编码实测**（信号分析 tab，4 张图）：

- 影子曲线图 31 条曲线，`stroke-width` 仅两种值 `[2, 1.5]`、`stroke-opacity` 仅两种 `[1, 0.55]`——**精确命中改版后的 Eleven-Series 编码**（前 10 条实线满宽，第 11 条起降宽降透明）
- 曲线 `stroke` 全部为 `var(--chart-1..10)` / `var(--ink-80)`，暗色可整体重映射
- 4 张图的网格线 `stroke-dasharray` 全为 `none`（符合 Charts 约定）
- X 轴刻度实测 `IBM Plex Mono` + `font-variant-numeric: normal` + 墨 55 + `11px`

**One Blue Rule 在 live tab 上的真实张力**（未解决，需目视定夺）：

live tab 有 9 个填充蓝块，纵向分布 `-108 / -16 / 0 / 0 / 0 / 10 / 356 / 431 / 2082`（文档高 2137px）。前 8 个落在首屏内，超过「一屏 ≤3 处」的字面约束。但逐个看都落在规则允许的两类里：`⚡ 下单`、`📈 面板`、`📈 K 线对照` 是 fixed 浮动控件，`现货 → 预测钱包` 是主动作；`7天`、`累计盈亏`、`5m` 是**当前选中态**筛选片；`1` 是分页当前页。

即：每一处蓝色都语义正当，但密度超了字面上限。收紧意味着把交易屏上的真实动作按钮降级为次级形态——这是 UX 决策，不在视觉迁移范围内自行处理。另有 26 个「存」按钮是**描边蓝**（`border-brand bg-card text-brand`，且全部 `disabled`），不计入「蓝色块」。

**验证方法的限制**：内嵌浏览器视口为 `0×0`，截图不可用。recharts `ResponsiveContainer` 在宽度 ≤0 时不生成 `<svg>`——首轮探测得到「0 条曲线」是视口产物，不是代码缺陷；注入宽度并强制重挂载后 4 个 wrapper / 37 条曲线正常渲染。**因此布局、密度、构图与暗色模式的实际观感仍未经目视确认。**

### 已声明未消费的契约组件（9 个）

`.impeccable/design.json` 里的组件契约已全部落到 index.css，其中 24 个 `ds-*` 类已被消费，以下 9 个**已声明但未消费**。保留它们是有意为之（契约完整性），不是遗漏：

- **`.ds-btn-ghost`** — 透明底 + 墨 95 字，hover 降到墨 55。现有次级按钮多用 `bg-sunken` 填充形态（视觉上比 ghost 更明确可点），暂无纯 ghost 场景。
- **`.ds-field` / `.ds-field-label`** — label + input 的垂直分组。现有 label 作为兄弟元素散写并已走 token（`text-ink-55` + 11px），接线需重构 JSX 结构，收益不抵回归风险。
- **`.ds-readout` / `-label` / `-value` / `-unit`、`.ds-pos` / `.ds-neg`** — NumericReadout 组件（20px 数值 + 退后的单位）。现有 statBox 用 16px 密度，是多卡横排下的有意选择；且它们已通过 `font-mono` 全局规则获得 tabular-nums，Tabular Numerals Rule 已满足。**若将来要做单个突出的大读数，直接用这套。**

已删除：`.ds-overlay`（与 `bg-overlay` 工具类完全重复，且不在 design.json 契约内）。

### 决策日志

| 决策 | 理由 |
|---|---|
| **Eleven-Series Rule 改用透明度+线宽** | dash 在影子图/场景图里已是「虚线=冻结基准、点线=盈亏平衡」的语义通道，叠加会让第 11 条主曲线与基准线不可区分。表格侧版本名与图例圆点同步淡化以保持对应。用户 2026-09-07 决策。 |
| **参考线保持与曲线同色** | 推翻初稿的「参考线一律墨 55」：22 曲线 × 2 参考线 = 44 条灰虚线会丢失曲线↔基准归属，而「对照冻结基准看偏离」是 PRODUCT.md 工作 #3。通用参考线（50% 中线、零轴）仍用墨 55。 |
| **teal 保留为第四种实色按钮** | 3 个 teal 按钮全是 Python 聚类动作，与 PY_CLUSTER 徽章同色，是有意的语义呼应。限定只用于 PY 聚类语义。用户 2026-09-07 决策。 |
| **stock 圆角名重映射守卫** | 仅清代码不够——Tailwind 默认 md=6px/xl=12px 会回归。@theme 里把 xs/md/lg→sm、xl→card、2xl/3xl→panel，使 off-spec 圆角无法用 stock 类名表达。 |
| **每个语义色补 `*-hover` 档** | 单档 token 会让 `bg-positive hover:bg-positive` 塌缩掉 24 个按钮的悬停反馈。色值取自 design.json tonal ramp（亮色深一档 / 暗色浅一档），与既有 `brand-hover` 同一推导。 |
| **`.ds-input-sm` 紧凑变体** | 通道行的护栏/金额输入框处于 13 行密集列表，标准 `6px 12px` 会撑高溢出。只覆写几何，其余继承 `.ds-input`。 |

### 迁移方法上的教训

1. **色相映射表会带偏非语义元素**：`cyan → teal` 对语义色正确，但把登录框 `focus:border-cyan-500` 和登录提交按钮 `bg-cyan-600` 一起带偏了——登录是该屏唯一主操作，必须是品牌蓝。**迁移后必须回看主操作与 focus 态。**
2. **Tailwind 对未知类名静默忽略**：`leading-140`、`visibility-hidden` 这类拼错的类不会让 build 失败，只是不产生样式。**build 通过 ≠ 类名有效**，必须查产物 CSS 或用计算样式验证。
3. **token 收敛会塌缩状态差异**：把 7 级灰收敛到 4 级、把多档色相收敛到单档 token 时，原本靠「相邻两档」表达的 base/hover 差异会消失。收敛后要专门跑一次 base∩hover 检测。
4. **子串污染会骗过计数**：`text-gray-50` 的计数会包含 `text-gray-500`，`rounded-l` 会包含 `rounded-lg`。所有计数必须用带词边界的正则。
