# Value1 Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/05-value-1.png`
- Interaction model: static + scroll-reveal
- Selector: `.home-value-section-1-wrapper--v13`（rect: top 3865, h 1018, w 1440）

## DOM Structure
- `.container-wrapper home-value-section-1-wrapper--v13`
  - `.home-value-section-1 home-value-section-1--v13`（top 3945, h 858）
    - h2 + 3 张 `.value-card`（含 `.card-title`）+ "免费试用" 按钮 ×3

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；padding-top/bottom `80px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`
- h3: `18px` / `26px`，weight `600`，color `rgba(12,12,12,0.95)`
- p: `16px` / `24px`，weight 400，color `rgba(12,12,12,0.8)`
- 免费试用按钮 ×3：卡片内 CTA

## States & Behaviors
- 无 click 状态；卡片随 `.transitionnode.fade` 进入视口淡入

## Per-State Content (if applicable)
不适用

## Assets
- 卡内插图：`59-light-1.D-QZgQBZ.png` … `63-light-5.BoVltOAp.png`（light 系列插图，以 manifest 为准）

## Text Content (verbatim)
> 调研、设计、编程、
> 增长，尽在一处
> 几分钟即可上线，不用等几周
>
> 告诉 Atoms 你的想法，几分钟内就能看到它构建出可用的应用。通过与 AI 聊天，获取完整的功能页面、流程和特性。
>
> 真实应用，不只是演示
> 构建一个可发布、可增长并可扩展的真实产品。所有刚需功能内置，包括用户登录、数据存储和 Stripe 支付。
>
> 商业工具包，尽在一处
> 在一个地方运行整个工作流程。进行市场调研、构建全栈应用、部署、优化 SEO、添加集成并跟踪结果。Atoms 可自动处理繁琐工作，让你专注于最重要的事情。
>
> 获得付费客户和收入
> 把想法变成用户愿意付费的产品。借助全栈业务能力，Atoms 负责发布、托管和日常运营，让你更快实现收入。
>
> 始终拥有完整所有权
> 随时导出代码并同步到 GitHub。在你的业务增长时，保持对项目的完全控制。
>
> 免费试用

（注：原始抓取中各标题文本重复出现两次，为 reveal 过渡双节点所致，渲染内容为单份。）

## Responsive Behavior
- 768px：卡片 2 列；390px：单列全宽
