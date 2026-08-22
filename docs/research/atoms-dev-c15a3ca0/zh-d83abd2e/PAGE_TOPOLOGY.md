# PAGE TOPOLOGY — atoms.dev/zh

- Source: https://atoms.dev/zh
- DOM 链：`#__nuxt > .app-container > #page-default-layout > #layoutScrollContainer > .home-page (top 56, h 14786) > .home`（11 个直接子区块）+ 同级 `.customfooter`
- Header（sticky 56px 透明）不在 `.home` 内，为布局级组件
- 滚动容器：window（文档流滚动，无 Lenis/scroll-snap）

## Section 顺序与几何（1440px 全页，top 为绝对坐标）

| # | Selector | top | height | 交互模型 |
|---|---|---|---|---|
| 00 | `.commonHeader`（sticky） | 0 | 56 | static（透明常驻） |
| 01 | `.home-chat--homepage-v13` | 96 | 518 | click-driven（输入框 + pill） |
| 02 | `.what-can-atoms-build` | 614 | 834 | click-driven（模板 pill + 卡片预览） |
| 03 | `.home-v13-trust` | 1448 | 1549 | time-driven（logo marquee 28s）+ scroll-reveal |
| 04 | `.home-ai-team-build-faster--v13` | 2997 | 868 | time-driven（agent-card 轮播）+ scroll-reveal |
| 05 | `.home-value-section-1-wrapper--v13` | 3865 | 1018 | static + scroll-reveal |
| 06 | `.home-value-section-2-wrapper--v13` | 4883 | 4624 | static + scroll-reveal（最高区块，features 网格） |
| 07 | `.home-world-section--v13` | 9507 | 910 | static + scroll-reveal |
| 08 | `.home-inspire-section--v13`（`.lazy-load-wrapper` 内） | 10417 | 1326 | static + 懒加载挂载 |
| 09 | `.home-pricing-section--v13` | 11743 | 1230 | click-driven（按年/按月切换） |
| 10 | `.home-faq-section--v13` | 12973 | 1149 | click-driven（手风琴 +44px/项） |
| 11 | `.home-tell-us-section--v13` | 14122 | 720 | static（渐变 CTA 面板） |
| 12 | `.customfooter` | 14842 | 524 | static |

## 逐区块交互模型

### 01 Hero Chat — click-driven
聊天式输入卡为核心：占位文案轮播 + "开始" CTA + 模板 pill（SaaS 应用/电子商务/内部工具/个人项目）。pill 点击填充示例 prompt（headless 未捕获，见 BEHAVIORS 缺口）。

### 02 Templates — click-driven
h2 "Atoms 能构建什么？" + 6 个模板 pill（SaaS 落地页/独立品牌与批发/电影工作室网站/热门推荐平台/创意视频/无尽跑酷游戏），点击切换下方预览卡。

### 03 Trust — time-driven
白底大区块（padding-top 128px）：h2 56px + 数据卡（170+ 国家、1M+ 构建者、16.8万 GitHub Star、#1 Product Hunt）+ 论文链接列表 + `.home-v13-trust__logo-track` marquee（28s linear infinite）滚动 OpenAI/NVIDIA/Stanford 等 logo。

### 04 AI Team Carousel — time-driven
h2 56px 左对齐 + 副文案 + "立即尝试"链接（→/zh/login）+ `.agent-card` 双份循环轮播（8 个 agent：Iris/Bob/Adrian/Emma/Mike/Sarah/Alex/David，卡片顶部彩色胶囊横幅）。

### 05 Value 1 — static
h2 56px + 3 张 `.value-card`（h3 18px w600 + p 16px 0.8），底部"免费试用"按钮 ×3（卡片内）。

### 06 Value 2 / Features — static
页面最高区块（4624px）：features 网格，逐 feature 卡图文交替，scroll-reveal 逐块淡入。

### 07 World Stories — static
h2 + 全球用户故事行（`.home-world-section__content`，px-48px，gap 64px，overflow hidden）。

### 08 Inspire — static + lazy
`.lazy-load-wrapper` 内，滚动接近才挂载；灵感展示区（1326px）。

### 09 Pricing — click-driven
max-width 1620px 居中；按年/按月 toggle 切换三卡价格（$0 / $15.8←$20 / $79←$100 ↔ $0 / $20 / $100）。

### 10 FAQ — click-driven
h2 64px/72px "常见问题" + 9 个问题手风琴，单击展开 +44px，单项展开。

### 11 Final CTA — static
渐变噪点面板（粉紫→薰衣草→矢车菊蓝，≈24px 圆角，max-width 1346px），h2 48px/56px w600 + 注册 CTA。

### 12 Footer — static
`.customfooter`，背景 `rgb(35,35,36)`，文本 `rgba(255,255,255,0.95)`，flex 布局链接列 + 版权行。

## 全局节奏

- 背景交替：浅灰 #F6F6F6（01/02/04/05）↔ 白 #FFF（03/06/09/10）；footer 为唯一深色区
- section 垂直 padding：80px（常规）/ 128px（trust、AI team）
- 所有非首屏区块标题走 `.transitionnode.fade` 显现
