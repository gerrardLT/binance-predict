# Inspire (Testimonials) Specification

## Overview
- Screenshot: `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/sections/08-inspire.png`
- Interaction model: static + 懒加载挂载（`.lazy-load-wrapper`）+ scroll-reveal
- Selector: `.home-inspire-section--v13`（rect: top 10417, h 1326, w 1440）

## DOM Structure
- `.lazy-load-wrapper`
  - `.container-wrapper home-inspire-section--v13`
    - h2 + `.testimonial-card` ×3（多列瀑布/网格，实际抓取截断于第 5 卡）

## Computed Styles (exact values from getComputedStyle)
- 容器背景: `rgb(246, 246, 246)`；padding-top/bottom `80px`
- h2: `56px` / `56px`，weight `400`，color `rgba(12,12,12,0.95)`，text-align `center`
- 卡内用户名 p: `20px` / `28px`，weight `600`，color `rgba(12,12,12,0.95)`
- testimonial-card：白底，约 16px 圆角

## States & Behaviors
- 无 click 状态；滚动接近时由 `.lazy-load-wrapper` 挂载，随后 fade reveal

## Per-State Content (if applicable)
不适用

## Assets
- 用户头像：`67-anusha-k.DMch3NTp.png`、`68-mike-judkins.D4wyzdUc.png`、`69-michel-harvey.Cn-GKO2f.png`、`70-kkangaces.Clqpf7bz.png`、`71-hasan.DPTrSrRy.png`、`75-beau-carnes.Gvii-MEQ.png`、`76-stellar.BabQ1cMh.png`

## Text Content (verbatim)
> 全球用户喜爱
>
> I have to say nothing is like Atoms
> right , I have to say nothing is like Atoms , I have tried all other agents, but nothing comes close tbh. just add the database part and publishing to Play Store, App Store option...
> — Anusha K
>
> Atoms is doing an amazing job
> Hi Atoms team, thanks for building such an awesome tool! Y'all are doing an amazing job. Keep up the great work!
> — Mike Judkins
>
> I love my Atoms experience
> I love my atoms experience. I learned a lot during my first month! Functional website + a web3.0 gaming ecosystem built on my mobile with Atoms. impossible to imagine a few years ago, even a few months ago. I only have positive things to say about Atoms.
> — Michel Harvey
>
> Atoms
> Honestly, go peep Atoms. You just yak what you want, and it cranks out a full web or mobile app front and back end, ready to roll. If you ever wanna go deeper, it has a deep research mode that auto-chops your idea into features or flows. It is a sweet middle ground between no-code and learning by doing, and it does not feel as locked in as most low-code toys.
> — kkangaces210103101
>
> This could be a total workflow transformation
> Finally! An AI that understands research isn't just about finding …（抓取截断，后续卡片文本未完整捕获）

## Responsive Behavior
- 768px：2 列；390px：单列全宽
