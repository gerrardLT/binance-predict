# Value2Features Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/06-value-2-features.png`
- Interaction model: static + scroll-reveal（页面最高区块）
- Selector: `.home-value-section-2-wrapper--v13`（rect: top 4883, h 4624, w 1440）

## DOM Structure
- `.container-wrapper home-value-section-2-wrapper--v13`
  - `.home-value-section-2 home-value-section-2--v13`（top 5011, h 4368）
    - h2 总标题
    - feature 卡序列：`.value-card cursor-pointer`（含 `.card-title` + `.card-desc`），每卡带"立即尝试"链接

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(255, 255, 255)`；padding-top/bottom `128px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`
- 立即尝试链接 ×4（捕获样本）：color `rgba(12,12,12,0.8)`，href `/zh/login`
- feature 卡：白底/浅底，cursor `pointer`

## States & Behaviors
- 无 click 状态切换；逐卡 scroll-reveal 淡入
- 卡片整体 cursor-pointer，点击应跳转（href 未逐卡捕获）

## Per-State Content (if applicable)
不适用

## Assets
- feature 卡配图：`64-5.CvYp2tZM.png`、`65-Number_26.DJ2JRRvr.png` 及 light 系列插图（以 ARTIFACT_MANIFEST 为准）

## Text Content (verbatim)
> 你所需的一切
> 构建、发布与变现
>
> 收入集中展示
>
> 可视化编辑器
> 精准实现你的设计。一个可视化编辑器，可快速调整布局和组件。
> 立即尝试
>
> Atoms云
> Atoms Cloud 为你的应用提供全栈后端，包括用户登录、数据库、集成和可扩展的托管。
> 立即尝试
>
> 竞赛模式
> 在多个模型上运行你的 prompt，即刻获得最佳版本。
> 立即尝试
>
> 即时 AI 集成
> 使用 Gemini 和 GPT 等模型为你的产品添加强大的 AI — 无需 API 密钥，无需设置。
> 立即尝试
>
> SEO 智能体
> 自动使你的网站搜索引擎可见，让 Google 能爬取、索引并对带来真实客户的页面进行排名。
> 立即尝试
>
> 广告专家
> 以更少的人工工作实现增长扩展。Ads specialist 负责管理广告系列创建、跟踪和优化。
> 立即尝试
>
> 更多功能
> 深度研究和主题
> 立即尝试

## Responsive Behavior
- 768/390px：feature 卡单列堆叠，图片全宽
