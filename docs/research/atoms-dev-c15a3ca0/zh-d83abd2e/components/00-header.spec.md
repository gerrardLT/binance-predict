# Header (commonHeader) Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/desktop-full.png`（顶部 56px）
- Interaction model: static（sticky 常驻，无滚动状态变化）
- Selector: `.commonHeader`（布局级，不在 `.home` 内）

## DOM Structure
- `.commonHeader flex items-center justify-between` — `position: sticky`，高 56px
- 左：logo（图片链接，href `/zh`）
- 中/右：nav 链接 `定价 / 关于 / 社区 / 资源` + `登录`（文字链）+ `注册`（按钮）

## Computed Styles (exact values from getComputedStyle)
- position: `sticky`；display: `flex`；height: `56px`
- background: `rgba(0, 0, 0, 0)`；box-shadow: `none`；backdrop-filter: `none`；border-bottom: `0px`
- nav 链接：IBM Plex Sans，16px / 24px，weight 400，color `rgba(12,12,12,0.95)`；hover → `rgba(12,12,12,0.55)`（transition all）
- 注册按钮：16px / 24px，weight 500，color `#fff` on `rgb(66,103,255)`，border-radius `40px`，padding `6px 16px`

## States & Behaviors
- 仅一种状态：滚动前后样式完全一致（已验证，见 BEHAVIORS.md）

## Assets
- logo 图片（CDN，见 ARTIFACT_MANIFEST）

## Text Content (verbatim)
> 定价
> 关于
> 社区
> 资源
> 登录
> 注册

## Responsive Behavior
- mobile（390px）：nav 链接收起，保留 logo + 登录/注册（据 mobile-full.png）
