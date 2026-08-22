# Footer Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/12-footer.png`
- Interaction model: static
- Selector: `.customfooter`（rect: top 14842, h 524, w 1440；与 `.home` 同级）

## DOM Structure
- `.customfooter` — display `flex`，position `relative`
- 链接列组：产品 / 帮助中心（资源）/ 发现 / 公司 / 社区
- 底部行：slogan + 语言切换（中文）+ 版权/法律链接

## Computed Styles (exact values from getComputedStyle)
- 背景: `rgb(35, 35, 36)`（全页唯一深色区）
- 链接文本: `rgba(255, 255, 255, 0.95)`
- hover：headless 未捕获颜色变化（可能仅 cursor，见 BEHAVIORS 缺口）

## States & Behaviors
- 纯静态链接导航

## Per-State Content (if applicable)
不适用

## Assets
- logo（白色变体）+ 社交图标（X / LinkedIn / Discord，见全页截图，具体文件以 manifest 为准）

## Text Content (verbatim)
> 产品
> 定价
> AI Agents
> 模型
> 工具
>
> 帮助中心
> 资源
> 博客
> 用例
> 比较
> 视频
>
> 发现
> GitHub
>
> 公司
> 关于我们
> MetaGPT
> OpenManus
> Foundation Agents
> 隐私政策
> 服务条款
>
> 社区
> 合作伙伴
> 探索者计划
> X / Twitter
> LinkedIn
> Discord
>
> 把想法变成可销售的产品
> 中文

链接样本（已捕获 href）：定价 → `/zh/pricing`；AI Agents → `/zh/ai-agents`；模型 → `/zh/models`；工具 → `/zh/tools`。

## Responsive Behavior
- 768px：链接列 2–3 列网格；390px：列堆叠，slogan 与语言切换置底
