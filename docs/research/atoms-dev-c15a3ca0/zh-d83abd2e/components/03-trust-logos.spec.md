# TrustLogos Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/03-trust-logos.png`
- Interaction model: time-driven（logo marquee）+ scroll-reveal + 外链
- Selector: `.home-v13-trust`（rect: top 1448, h 1549, w 1440）

## DOM Structure
- `section.home-v13-trust`（2 子块）
  - `.home-v13-trust__main`（top 1576, h 1089）：h2 + 数据卡网格（`.home-v13-trust__cards`）
  - `.home-v13-trust__logo-wall`（top 2705, h 252）：`.home-v13-trust__logo-track` marquee

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(255, 255, 255)`；padding-top `128px`，padding-bottom `40px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`
- h3（深色卡内）: `20px` / `28px`，weight `400`，color `rgba(255,255,255,0.95)`
- p（强调链接蓝）: `16px` / `24px`，weight 400，color `rgb(66,103,255)`
- logo-track animation: `home-v13-logo-loop-82758f74 28s linear infinite`

## States & Behaviors
- marquee 无缝循环（双份 logo 列表），28s/循环，linear，不暂停（hover 暂停未验证）
- 数据卡与论文链接为 scroll-reveal（`.transitionnode.fade`）

## Per-State Content (if applicable)
不适用（无 click 状态）

## Assets
- 课程 poster：`11-creating-courses-poster.webp`
- 论文徽章：`17-ICLR.svg`、`18-arXiv.svg`、`19-NeurIPS.svg`、`20-ICML.svg`；上下渐隐：`21-papers-fade-top.webp`、`22-papers-fade-bottom.webp`
- GitHub 图表：`23-github-stars-chart.webp`；卡背景：`24-card-background.webp`
- Product Hunt：`72-producthunt.D6_ay3DX.svg`；地球：`10-dotted-globe.webp`、`25-flag_6.svg`
- logo 墙（双份）：`26/27-openai.svg`、`28/29-nvidia.svg`、`30/31-stanford.svg`、`32/33-google.svg`、`34/35-amazon.svg`、`36/37-mit.svg`、`38/39-microsoft.svg`、`40/41-salesforce.svg`、`42/43-uc-berkeley.svg`、`44/45-tesla.svg`、`46/47-samsung.svg`

## Text Content (verbatim)
> 为什么大家信任 Atoms？
> 探索Atoms
>
> 全球触达
>
> 170+
>
> 使用 Atoms 的国家和地区
>
> 探索Atoms
>
> 正在创建课程
>
> 我如何用 Atoms AI 打造可变现的 app
>
> 查看详情
>
> 1M+
>
> 我们社区中的构建者
>
> 用 Atoms 让创意落地。
>
> 约 30 篇论文
> MetaGPT，ICLR 2024 口头报告
> AFlow，ICLR 2025 口头报告
> 基础智能体的进展与挑战，arXiv 2025
> Atom of Thoughts，NeurIPS 2025
> AutoWebWorld，ICML 2026
> AOrchestra，arXiv 2026
> （论文列表双份重复用于 marquee）
>
> GitHub
>
> 16.8万
>
> GitHub Star
>
> 我们团队开源项目累计获得的 GitHub Star。
>
> Product Hunt
>
> #1
>
> 每周精选产品
>
> 受到这些公司的创作者信赖：
>
> OpenAI / NVIDIA / Stanford University / Google / Amazon / MIT / Microsoft / Salesforce / UC Berkeley / Tesla / Samsung

外链：YouTube 课程视频（es7cLQBc7ns）+ 3 条 Google Scholar 论文引用链接（均为 ink-95 色）。

## Responsive Behavior
- 768/390px：数据卡降为单列，marquee 保持，logo 墙高度收缩
