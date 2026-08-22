# WorldStories Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/07-world-stories.png`
- Interaction model: static + scroll-reveal
- Selector: `.home-world-section--v13`（rect: top 9507, h 910, w 1440）

## DOM Structure
- `.container-wrapper home-world-section--v13`
  - `.home-world-section__content px-48px lt-md:px-0 flex flex-col items-center justify-center gap-64px of-hidden`（top 9587, h 750, 2 子块）
    - h2 + 地球视觉（dotted globe）+ `.globe-info-card` 故事卡
    - CTA 按钮

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；padding-top/bottom `80px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`，text-align `center`
- CTA 按钮："开启你的成功故事"（黑底白字 pill 变体：`rgba(255,255,255,0.95)` on `rgb(12,12,12)`，radius 40px）

## States & Behaviors
- 无 click 状态；地球上的点位脉冲（keyframes `globe-point-pulse` 存在）；info-card 定位在地球上（美国点位）

## Per-State Content (if applicable)
不适用（info-card 可能随点位切换，headless 未验证多状态）

## Assets
- 地球：`10-dotted-globe.webp`；国旗：`66-United_States.tH2Ut8qB.png`；用户头像：`73-kausik-lal.BLZi5wsx.png`、`74-mia.DZ7PRk0k.png` 等（以 manifest 为准）

## Text Content (verbatim)
> 遍及 世界 的成功故事
> Sal
> United States
> 我经营一家窗户清洁业务，过去依赖多个应用来完成工作。我构建了一个将预约估价、日程安排和客户文件集中在一个地方的应用。
> 开启你的成功故事

## Responsive Behavior
- 390px：`lt-md:px-0` 取消水平 padding，地球与卡片垂直堆叠
