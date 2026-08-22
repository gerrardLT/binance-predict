# Templates (WhatCanAtomsBuild) Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/02-templates.png`
- Interaction model: click-driven（pill 切换下方预览卡）
- Selector: `.what-can-atoms-build`（rect: top 614, h 834, w 1440）

## DOM Structure
- `section.what-can-atoms-build`
  - `.what-can-atoms-build__inner`（top 694, h 674, 2 子块）
    - h2 标题 + pill 行（6 个模板）
    - 预览卡区（随 pill 切换）

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；padding-top/bottom `80px`；display `block`
- h2: `"IBM Plex Sans"` + 系统栈，`24px`，weight `500`，line-height `32px`，color `rgba(12,12,12,0.95)`（本区标题为小号变体，非 56px）
- pill 按钮：捕获到 4 个带文本按钮（见下）

## States & Behaviors
- 点击 pill → 下方预览卡切换为该模板的演示（截图/视频卡）
- headless 下 pill hover/click 部分超时 → 逐状态视觉未完整捕获（缺口）

## Per-State Content (if applicable)
- 每个 pill 对应一个预览卡状态；逐状态截图未捕获（缺口）

## Assets
- 预览卡媒体：`52-2.cu10e4p4.png`、`53-3.xG7SjSLn.png`、`54-4.BPQEa1AB.png`、`55-5.ImuIZqcW.png`、`56-main.DGt_P6Ei.png`、`57-1.D886DBu4.png`、`58-2.DeoBqn1x.png`、`84-poster.webp` … `89-poster.webp`（部分为视频 poster）

## Text Content (verbatim)
> Atoms 能构建什么？
> SaaS 落地页
> 独立品牌与批发
> 电影工作室网站
> 热门推荐平台
> 创意视频
> 无尽跑酷游戏

## Responsive Behavior
- 768/390px：pill 换行，预览卡单列全宽
