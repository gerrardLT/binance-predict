# FAQ Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/10-faq.png`
- Interaction model: click-driven（手风琴，已验证）
- Selector: `.home-faq-section--v13`（rect: top 12973, h 1149→1193 展开后, w 1440）

## DOM Structure
- `.container-wrapper home-faq-section--v13`
  - `.w-full flex flex-col items-center justify-center gap-16px box-border home-faq-section__content--v13`（top 13053, h 989, 2 子块）
    - h2 "常见问题"
    - 9 个手风琴项（问题行 + 答案容器）

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(255, 255, 255)`；padding-top/bottom `80px`
- h2: `64px` / `72px`，weight `400`，color `rgba(12,12,12,0.95)`，text-align `center`（全页最大标题）
- h3（问题行）: `20px` / `28px`，weight `500`，color `rgba(12,12,12,0.95)`
- p（答案）: `16px` / `28px`，weight 400，color `rgba(12,12,12,0.8)`
- FAQ 项 hover：无样式变化（hover sweep 验证）

## States & Behaviors（click sweep 已验证）
- 默认：全部收起
- 点击问题行 → 单项展开，section 高度 +44px（1149 → 1193）
- 已捕获展开态答案（"Atoms 应用或网站如何部署？"）：
  > 一键即可。Atoms AI 为你处理托管和服务器配置。准备好后即可立即发布在线 URL。无需复杂设置。
- 其余 8 项答案未逐项展开捕获（缺口，模式同上）

## Per-State Content (if applicable)
- 收起态：仅问题行；展开态：+答案段落（高度 +44px）

## Assets
- 无

## Text Content (verbatim)
> 常见问题
> 什么是 Atoms？
> 我可以用 Atoms 构建什么？
> 使用 Atoms 是否需要编程或技术技能？
> Atoms 应用或网站如何部署？
> 一键即可。Atoms AI 为你处理托管和服务器配置。准备好后即可立即发布在线 URL。无需复杂设置。
> 积分体系如何运作，定价是否透明？
> Atoms 如何帮助我获得付费客户？
> 我是否拥有使用 Atoms 构建的内容？
> Atoms 可以与我已经使用的工具集成吗？
> Atoms 是否支持我的母语？

## Responsive Behavior
- 768/390px：问题行全宽，h2 缩放，展开高度自适应
