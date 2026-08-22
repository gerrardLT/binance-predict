# Pricing Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/09-pricing.png`
- Interaction model: click-driven（按年/按月 toggle，已验证）
- Selector: `.home-pricing-section--v13`（rect: top 11743, h 1230, w 1440）

## DOM Structure
- `.container-wrapper bg-bgBaseDefaultLow home-pricing-section--v13`
  - `.w-full max-w-1620px mx-auto`（top 11823, h 1070）
    - h2 "定价" + toggle（按年/按月）
    - `.billingCardWrap billingCardWrap--homepage-v13 billingCardWrap--3cols`：3 卡（Free / Pro / Max）
      - `.pricingCardSlot pricingCardSlot--offset`、`.billingCardWrapper`

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(255, 255, 255)`；padding-top/bottom `80px`；内容 max-width `1620px` 居中
- h2: `56px` / `56px`，weight `500`（本区为 500），color `rgba(12,12,12,0.95)`，text-align `center`
- p（卡内说明）: `14px` / `21px`，weight 400，color `rgba(12,12,12,0.8)`
- 卡片：白底，无 hover 变化（hover sweep 验证 before=after）

## States & Behaviors（click sweep 已验证）
| 状态 | Free | Pro | Max |
|---|---|---|---|
| 按年（默认） | $0 / 月 | $15.8 / 月（划线 $20） | $79 / 月（划线 $100） |
| 按月 | $0 / 月 | $20 / 月 | $100 / 月 |
- toggle 按钮文本：`按年 · 最多可节省 21%` / `按月`；切换仅改价格文本，布局不变
- Pro/Max 卡含"21% 折扣"徽标；Max 卡带"推荐"徽标

## Per-State Content (if applicable)
见上表；特性清单不随切换变化。

## Assets
- 无明显图片资产（纯排版卡）

## Text Content (verbatim)
> 定价
> 按年 · 最多可节省 21%
> 按月
>
> Free — $0 / 月
> 入门使用指南
> 15 积分 / 天
> 开始
> 15 每日积分 (最多 25 / 月)
> 2GB 磁盘空间
> 无限项目共享
> 2 个 Atoms 后端项目
>
> Pro — 21% 折扣 — $15.8 / 月（$20）
> 解锁更多功能
> 100 积分 / 月
> 开始
> 15 每日积分 (最多 25 / 月)
> 100 每月积分
> 10GB 磁盘空间
> 私人项目 / 下载项目 / 移除 Atoms™ 徽章 / 编辑项目 / 积分结转
> 无限制的 Atoms 后端项目 / Atoms 生产云 / 自定义域名
> 无限团队成员 / 集中计费 / 集中访问管理
>
> Max — 推荐 — 21% 折扣 — $79 / 月（$100）
> 全功能访问Atoms最佳体验
> 500 积分 / 月
> 开始
> 15 每日积分 (最多 25 / 月)
> 500 每月积分
> 100GB 磁盘空间
> （Pro 全部特性）
> 2倍 计算资源 (相比 Pro)
> 竞赛模式 了解更多
> *

## Responsive Behavior
- 768/390px：三卡降为单列堆叠，toggle 保持
