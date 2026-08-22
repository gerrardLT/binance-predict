# HeroChat Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/01-hero-chat.png`
- Interaction model: click-driven（模板 pill + 输入框 + 开始 CTA）
- Selector: `.home-chat--homepage-v13`（rect: top 96, h 518, w 1440）

## DOM Structure
- `.container-wrapper home-chat--homepage-v13`
  - `.home-chat__inner w-full flex flex-col items-center gap-24px h-full relative`
    - Notice 公告条（"New models are live in Atoms"）
    - h1 主标题
    - p 副标题
    - 聊天输入卡（占位轮播 + 开始按钮）+ 头像行
    - 模板 pill 行

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；display: `block`
- h1: `"IBM Plex Sans"`（首 h1 Latin 部分为 IBM Plex Serif），`48px`，weight `500`，line-height `56px`，color `rgba(12,12,12,0.95)`，text-align `center`
- p: `13px` / `22px`，weight 400，color `rgba(12,12,12,0.55)`，max-width `100%`
- 输入卡：白底，圆角约 16–20px，柔和 shadow
- 开始按钮：`16px` / `24px`，weight 500，`#fff` on `rgb(66,103,255)`，radius `40px`

## States & Behaviors
- 占位文案轮播（示例："上线一个支持 Stripe 支付的电子商务网站..."）
- 模板 pill（4 个）：点击填充示例 prompt —— **headless 下 click 超时，逐状态内容未捕获（缺口）**
- pill hover 态未捕获（缺口）

## Per-State Content (if applicable)
- pill 状态内容缺失，见缺口

## Assets
- 头像行：`12-avatar-1.webp` … `16-avatar-5.webp`（资产根 `public/sites/atoms-dev-c15a3ca0/zh-d83abd2e/`）

## Text Content (verbatim)
> Notice
>
> New models are live in Atoms
>
> 把想法变成可销售的 产品
>
> AI 员工用于验证想法、构建产品并获取客户。几分钟内完成。无需编码。
>
> 上线一个支持 Stripe 支付的电子商务网站...
> 开始

## Responsive Behavior
- 390px：标题缩放（`lt-md:` 断点），pill 换行，输入卡全宽（据 mobile-full.png）
