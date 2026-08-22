# AiTeamCarousel Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/04-ai-team-carousel.png`
- Interaction model: time-driven（agent-card 双份循环轮播）+ scroll-reveal
- Selector: `.home-ai-team-build-faster--v13`（rect: top 2997, h 868, w 1440）

## DOM Structure
- `.container-wrapper home-ai-team-build-faster--v13`
  - `.ai-team-content w-full flex flex-col items-center justify-center gap-40px ai-team-content--v13`（top 3125, h 612, 2 子块）
    - 文案组：h2（左对齐）+ 副文案 + `立即尝试` 链接（→ `/zh/login`，color `rgba(12,12,12,0.95)`）
    - 轮播轨道：`.agent-card` × 16（8 agent × 双份循环），内部 `.agent-card-inner`

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；padding-top/bottom `128px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`，text-align `left`
- agent-card：白底，约 16px 圆角；顶部为彩色胶囊横幅（绿 #34A853 系 / 紫 #B57EDC 系 / 靛蓝 #7C86D8 系 / 琥珀 #E8A33D 系 / 天蓝 / 粉）

## States & Behaviors
- 双份卡片无缝水平循环；自动滚动周期未计时捕获（缺口）
- 卡片内容：agent 名 + 职位 + 头像 + 一段职责描述

## Per-State Content (if applicable)
不适用（连续滚动，无离散状态）

## Assets
- 头像（横幅小头像）：`0-Mike-TeamLeader-Avatar_origin.DmBYWaXT.webp`、`1-Adrian-AdsAgent-Avatar.D1HVIhCr.png`、`2-Sarah-SEOSpecialist-Avatar_origin.DYHquUJp.webp`、`3-Emma-ProductManager-Avatar_origin.BBeqkRr7.webp`、`4-Bob-Architect-Avatar_origin.Cdi-oMPW.webp`、`5-Alex-Engineer-Avatar_origin.zH2J8gqX.webp`、`6-David-DataAnalyst-Avatar_origin.CahzHabe.webp`、`7-Iris-DeepResearcher-Avatar_origin.uohFf0-y.webp`
- 大头像/插图：`48-iris.cWsNABAt.png`、`49-bob.CK05J5j-.png`、`50-adrian.B6ZaN_wb.png`、`51-david.BH0CUhJj.png`、`8-card-background.webp`、`9-lower-right-wash.svg`

## Text Content (verbatim)
> 你的 AI 团队，帮助你更快构建并赢得客户
> 一个完整的 AI 团队，帮助你以更低成本更快发布。你做决策，智能体负责研究、规划、构建、测试和增长。
> 立即尝试
>
> Iris — Deep Researcher：通过 Deep Research 发现真实需求和细分市场，然后将信号转化为聚焦的机会。
> Bob — Architect：设计系统蓝图，选择合适的结构，使你的应用可扩展且可靠。
> Adrian — Ads Specialist：自动运行 Google Ads。Ads Agent 负责管理广告系列创建、跟踪和优化，让你以更少的投入实现增长扩展。
> Emma — Product Manager：将你的想法转化为明确的规格和范围，以便构建保持简单且可用。
> Mike — Team Leader：端到端运行计划，协调 agents，并请求你的批准，这样你在保持知情的同时也能快速行动。
> Sarah — SEO Specialist：快速推出 SEO 页面并自动化优化，以更低的成本快速带来自然流量。
> Alex — Engineer：通过连接前端、后端、集成和部署，构建一个可投入生产的全栈应用。
> David — Data Analyst：通过分析海量数据发现增长机会。并呈现清晰洞察，帮助你做出更明智、数据驱动的决策。
> （以上 8 卡双份重复用于轮播）

## Responsive Behavior
- 768/390px：卡片变窄，轮播保持，文案居中
