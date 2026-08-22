# FinalCta Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/11-final-cta.png`
- Interaction model: static（渐变 CTA 面板）+ scroll-reveal
- Selector: `.home-tell-us-section--v13`（rect: top 14122, h 720, w 1440）

## DOM Structure
- `.container-wrapper home-tell-us-section--v13`
  - `.w-full flex flex-col items-center justify-center max-w-1346px mx-auto home-tell-us-section__shell--v13`（top 14202, h 480）
    - 渐变噪点面板：h2 + "开始" CTA

## Computed Styles (exact values from getComputedStyle)
- 外层背景: `rgb(255, 255, 255)`；padding-top `80px`，padding-bottom `160px`
- 面板：max-width `1346px`，高约 480px，圆角约 24px；背景为粉紫（#E9C7F2 系）→ 薰衣草 → 矢车菊蓝（#7FA6F2 系）渐变，叠加噪点纹理（`83-starts.X91XbzmB.png` 星星背景图）
- h2: `48px` / `56px`，weight `600`，color `rgba(12,12,12,0.95)`，text-align `center`
- 开始按钮：蓝色 pill（`#fff` on `rgb(66,103,255)`，radius 40px）

## States & Behaviors
- 无 click 状态切换；面板随 fade reveal 进入

## Per-State Content (if applicable)
不适用

## Assets
- 星星/噪点背景：`83-starts.X91XbzmB.png`（亦作为页面级 background-image 使用）

## Text Content (verbatim)
> 把想法变成可销售的产品
> 开始

## Responsive Behavior
- 390px：面板全宽减边距，h2 缩放，padding-bottom 收缩
