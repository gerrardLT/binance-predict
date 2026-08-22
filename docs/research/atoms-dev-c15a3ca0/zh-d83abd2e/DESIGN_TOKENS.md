# DESIGN TOKENS — atoms.dev/zh

- Source: https://atoms.dev/zh
- Site key: `atoms-dev-c15a3ca0` · Page key: `zh-d83abd2e`
- Captured: 2026-08-21 (headless Chromium 1440×900, `html.light` theme)
- Stack: Nuxt (Vue) SPA, UnoCSS-style atomic classes (`lt-md:` 前缀断点), CDN `public-frontend-cos.metadl.com`

## Fonts

页面通过自托管 CSS（metadl CDN）加载以下字体族，`document.fonts` 确认实际加载：

| Family | Weights | Styles | 用途 |
|---|---|---|---|
| **IBM Plex Sans** | 100–700 | normal | 全局 UI 正文字体（body、h2/h3、按钮、链接） |
| **IBM Plex Serif** | 100–700 | normal + italic | 首个 h1（hero 标题 Latin 字符） |
| **IBM Plex Mono** | 100–700 | normal | 代码/等宽场景 |
| **Inter** | 100–700 | normal | 备用/组件库 |
| **Yatra One** | 400 | normal | 装饰性（logo 墙等） |

> 注意：中文 glyph 不在 IBM Plex 覆盖范围内，实际渲染回退到系统 CJK 字体（headless Windows 下呈宋体/衬线观感）。克隆时建议为中文显式指定回退栈（如 `"IBM Plex Sans", "PingFang SC", "Microsoft YaHei", sans-serif`）。

### 计算排版值（getComputedStyle 精确值）

| 角色 | family | size | weight | line-height | color | 其他 |
|---|---|---|---|---|---|---|
| body | IBM Plex Sans + 系统栈 | 14px | 400 | 21px | rgb(0,0,0) | bg rgb(246,246,246) |
| h1 (hero) | IBM Plex Sans（首 h1 为 IBM Plex Serif） | 48px | 500 | 56px | rgba(12,12,12,0.95) | text-align center, max-width 980px, margin-bottom 8px |
| h2 (section 大标题) | IBM Plex Sans | 56px | 400 | 56px | rgba(12,12,12,0.95) | 多数 section；FAQ 为 64px/72px；模板区为 24px/32px w500；定价为 w500；终 CTA 48px/56px w600 |
| h3 | IBM Plex Sans | 18px | 600 | 26px | rgba(12,12,12,0.95) | 卡片标题；20px/28px 变体存在 |
| p (次要) | IBM Plex Sans | 13px | 400 | 22px | rgba(12,12,12,0.55) | max-width 100% |
| p (正文) | IBM Plex Sans | 16px | 400 | 24px | rgba(12,12,12,0.8) | |
| p (强调/链接蓝) | IBM Plex Sans | 16px | 400 | 24px | rgb(66,103,255) | 品牌蓝文本 |
| button (主 CTA 蓝) | IBM Plex Sans | 16px | 500 | 24px | #fff on rgb(66,103,255) | pill radius 40px, padding 6px 16px |
| button (黑) | IBM Plex Sans | 16px | 500 | 24px | rgba(255,255,255,0.95) on rgb(12,12,12) | radius 40px, border 1px rgba(12,12,12,0.55) |
| a (nav) | IBM Plex Sans | 16px | 400 | 24px | rgba(12,12,12,0.95) | hover → rgba(12,12,12,0.55), transition all |

按钮通用 transition: `color/background-color/border-color/text-decoration-color/fill/stroke 0.2s cubic-bezier(0.4,0,0.2,1)`。

## Colors

| Token | 值 | 用途 |
|---|---|---|
| `bg-base` | rgb(246,246,246) / #F6F6F6 | 页面/浅灰 section 背景 |
| `bg-surface` | rgb(255,255,255) | 白底 section（trust、features、pricing、faq）、卡片 |
| `ink-95` | rgba(12,12,12,0.95) | 标题、nav 文本 |
| `ink-80` | rgba(12,12,12,0.8) | 正文 |
| `ink-55` | rgba(12,12,12,0.55) | 次要文本、nav hover |
| `brand-blue` | rgb(66,103,255) / #4267FF | 主 CTA（开始/注册）、强调词、链接 |
| `ink-black` | rgb(12,12,12) | 黑底按钮、圆形 + 按钮 |
| `border` | rgb(229,231,235) / #E5E7EB | 默认边框 |
| `footer-bg` | rgb(35,35,36) / #232324 | 页脚背景 |
| `footer-ink` | rgba(255,255,255,0.95) | 页脚文本 |
| Agent 卡片横幅色 | 绿 #34A853 系 / 紫 #B57EDC 系 / 靛蓝 #7C86D8 系 / 琥珀 #E8A33D 系 / 天蓝 / 粉 | AI Team 轮播卡片顶部胶囊横幅 |
| 终 CTA 渐变 | 粉紫 (#E9C7F2 系) → 薰衣草 → 矢车菊蓝 (#7FA6F2 系)，带噪点纹理 | `home-tell-us-section` 圆角面板 |

## Radii / Shadows / Borders

- 按钮：`border-radius: 40px`（全 pill）
- 卡片（agent/value/testimonial）：约 16px 圆角，白底，无明显 shadow（border 极浅或无）
- hero 输入卡：约 16–20px 圆角，白底，柔和 shadow
- 终 CTA 面板：约 24px 圆角
- 默认 border-color：rgb(229,231,235)

## Global UI Patterns

- `html.light` + `body.scrollbody`；滚动发生在文档流（`#layoutScrollContainer` 为布局容器，window 可滚动）
- 无 Lenis / Locomotive；`scroll-behavior: auto`；无 scroll-snap
- 自定义滚动条：`::-webkit-scrollbar` / `-track` / `-thumb` / `-corner`（含 `.prompt-input-text` 局部变体）
- 滚动显现：`.transitionnode.fade` 元素初始 `opacity: 0`，进入视口后淡入（IntersectionObserver 驱动）
- 懒加载：中后段 section 包裹于 `.lazy-load-wrapper`，滚动到附近才渲染内容
- Header：`position: sticky`，高 56px，背景透明（滚动前后无变化）

## Keyframes（document.styleSheets 枚举）

`blink, shimmer, skeleton-pulse, pulse, spin, iconSpin, moveAround-*, vaul-panel-mask-enter/exit, logo-fade-in, payment-method-modal-rim-flow, breathing, breathing_tiny, blink-*, marquee-*, pulse-*, fadeIn-*, name-fadeIn, shimmer-*, globe-point-pulse, home-v13-paper-loop, home-v13-logo-loop (28s linear infinite), p-overlay-mask-enter/leave, ripple`

## Meta / Favicons

- favicon: `https://atoms.dev/favicon.ico`（已下载）
- og:image: `https://atoms-cos.metadl.com/cms_medias/logo_only_blue_d7ad9c419a.png`（已下载）
- title: `Atoms：用 AI 构建网站与应用，无需编码`
- description: `Atoms 是您团队的网站与应用构建器。借助 Atoms 的 AI 团队成员，在几分钟内验证创意、构建产品并获取客户，完全无需编码。`
