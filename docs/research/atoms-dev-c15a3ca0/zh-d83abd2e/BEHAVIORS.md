# BEHAVIORS — atoms.dev/zh

- Source: https://atoms.dev/zh
- Captured: 2026-08-21, headless Chromium 1440×900, playwright-core
- 扫描方法：scroll sweep（滚动前后 header 对比 + reveal 元素检测）、click sweep（定价切换 / FAQ 展开 / 模板 pill）、hover sweep（nav / pricing card / faq item / footer link）、responsive sweep（1440 / 768 / 390）、timed（animation 枚举）

## Scroll 行为

### Sticky Header（无滚动状态变化）
- `position: sticky`，高度恒定 **56px**，class `commonHeader flex items-center justify-between`
- 滚动前（atTop）与滚动后（scrolled）计算样式完全一致：
  - background: `rgba(0, 0, 0, 0)`（透明）
  - box-shadow: `none`
  - backdrop-filter: `none`
  - border-bottom: `0px`
- 结论：header 始终透明叠在内容之上，无"滚动后加背景/阴影"状态。克隆时保持透明即可。

### 滚动显现（scroll reveal）
- `.transitionnode.fade` 元素初始 `opacity: 0`，`transform: none`，`animation: none 0s`
- 进入视口后由 IntersectionObserver 触发淡入（Vue transition，无 keyframes 残留）
- 至少 5 处使用该模式（各 section 标题/内容块）
- 克隆建议：`opacity 0 → 1`，duration ≈ 300–500ms ease，进入视口触发，不回放

### 懒加载
- 中后段 section（08-inspire 起）包裹于 `.lazy-load-wrapper`，滚动接近时才挂载内容
- mobile/tablet 视口下 section 03–11 均在 wrapper 内（responsive sweep 时大量 null 的根因）

## Click 行为

### 定价切换（click-driven，已验证）
- 初始（按年付）：`$0 / 月` · `$15.8 / 月`（划线 `$20`）· `$79 / 月`（划线 `$100`）
- 点击"按月"后：`$0 / 月` · `$20 / 月` · `$100 / 月`（无划线价）
- 点击"按年"后：恢复初始值
- 切换仅改价格文本，卡片布局不变

### FAQ 手风琴（click-driven，已验证）
- 点击问题行：section 高度 1149px → 1193px（**+44px** 展开）
- 展开项答案逐字：
  > 一键即可。Atoms AI 为你处理托管和服务器配置。准备好后即可立即发布在线 URL。无需复杂设置。
- 共 9 个问题（完整列表见 components/10-faq.spec.md）

### Hero 模板 pill（缺口）
- `SaaS 应用 / 电子商务 / 内部工具 / 个人项目` 4 个 pill 在 headless 下 `locator.click` 超时（4s）
- 推断为点击后填充输入框示例文案，未捕获逐状态内容 → 记为行为缺口

## Hover 行为

| 目标 | before | after | 结论 |
|---|---|---|---|
| nav 链接（定价） | color `rgba(12,12,12,0.95)`，bg 透明 | color `rgba(12,12,12,0.55)` | hover = 降低文本不透明度至 0.55；transition all |
| pricing card | bg `rgb(255,255,255)`，无 shadow/transform | 完全一致 | 卡片无 hover 态 |
| FAQ item | bg 透明，color `rgb(0,0,0)` | 完全一致 | 无 hover 态 |
| footer link | color `rgba(255,255,255,0.95)` | 完全一致 | headless 未捕获变化（可能仅 cursor/下划线） |
| hero CTA button | — | 超时 | 缺口：未捕获 hover 态 |
| template card | — | 超时 | 缺口：未捕获 hover 态 |

按钮 hover 通用过渡（来自计算样式 transition 属性）：`color/background-color/border-color/text-decoration-color/fill/stroke 0.2s cubic-bezier(0.4,0,0.2,1)`。

## Time-driven 动画

| 目标 | animation | 说明 |
|---|---|---|
| `.home-v13-trust__logo-track` | `home-v13-logo-loop-82758f74 28s linear infinite` | 信任区 logo 墙无缝 marquee，28s 一循环 |

keyframes 全集见 DESIGN_TOKENS.md（marquee-* / breathing / shimmer / logo-fade-in 等）。

## Responsive 行为

| 视口 | 布局要点 |
|---|---|
| 1440 | 三列卡片、双列 value 布局、logo 墙单行 marquee |
| 768 | 卡片降为 2 列/单列，hero h1 缩小（`lt-md:` 断点前缀生效），padding 收缩 |
| 390 | 全部单列（grid 卡宽 ≈358px），nav 收起，section 3–11 由 `.lazy-load-wrapper` 按需挂载 |

## 已知缺口

1. Hero 模板 pill 点击后的逐状态内容未捕获（headless click 超时）
2. hero CTA 与 template card 的 hover 态未捕获（headless hover 超时）
3. mobile/tablet 下部分 section 因懒加载未展开，计算样式为 null（已用全页截图 + phase4 DOM 探测补齐结构信息）
4. AI Team 轮播的自动滚动周期未计时捕获（仅确认双份卡片循环结构与 marquee 类 keyframes）
