# ARTIFACT MANIFEST — atoms.dev/zh

- Source: https://atoms.dev/zh
- Site key: `atoms-dev-c15a3ca0` · Page key: `zh-d83abd2e`
- Captured: 2026-08-21
- 下载方式: `download-assets.mjs`（本工件根内，node 原生 fetch，4 个一批）；逐条结果见资产根 `_download-results.json`

## 路径约定

| 工件 | 路径 |
|---|---|
| 工件根 | `docs/research/atoms-dev-c15a3ca0/zh-d83abd2e/` |
| 截图根 | `docs/design-references/atoms-dev-c15a3ca0/zh-d83abd2e/` |
| 资产根 | `public/sites/atoms-dev-c15a3ca0/zh-d83abd2e/` |

## 工件清单（工件根）

| 文件 | 内容 |
|---|---|
| `DESIGN_TOKENS.md` | 字体 / 排版 / 颜色 / 圆角 / 全局模式 / keyframes / meta |
| `BEHAVIORS.md` | scroll / click / hover / timed / responsive 扫描结果与缺口 |
| `PAGE_TOPOLOGY.md` | DOM 链、12 区块几何与交互模型 |
| `components/00-header.spec.md` … `12-footer.spec.md` | 13 个组件规格 |
| `download-assets.mjs` | 资产下载 helper（内嵌 92 条 URL） |
| `ARTIFACT_MANIFEST.md` | 本文件 |

## 截图清单（截图根）

- `desktop-full.png`（1440 宽全页，≈1729KB）
- `tablet-full.png`（768 宽全页，≈1958KB）
- `mobile-full.png`（390 宽全页，≈1012KB）
- `sections/01-hero-chat.png` … `sections/12-footer.png`（12 张区块截图）

## 资产下载汇总

- 枚举 92 条；**88 条成功（bytes > 0）**；4 条为 Bing 跟踪像素（bat.bing.com，0 字节，已剔除，非设计资产）
- 未下载：视频流（见下）与字体二进制（见下）

### 按类型分布

| kind | 数量 | 说明 |
|---|---|---|
| img | 81 | agent 头像、logo 墙、插图、testimonial 头像、模板预览图等 |
| bg | 1 | `83-starts.X91XbzmB.png`（星星/噪点，终 CTA 面板） |
| poster | 6 | `84`–`89-poster.webp`（模板区视频封面） |
| favicon | 1 | `90-favicon.ico` |
| og | 1 | `91-logo_only_blue_d7ad9c419a.png` |

### 按区块归属（对应 components/*.spec.md）

| 区块 | 文件 |
|---|---|
| 01 hero | `12`–`16-avatar-*.webp` |
| 02 templates | `52`–`58`（预览图）、`84`–`89-poster.webp` |
| 03 trust | `10-dotted-globe.webp`、`11-creating-courses-poster.webp`、`17`–`22`（论文徽章/渐隐）、`23-github-stars-chart.webp`、`24-card-background.webp`、`25-flag_6.svg`、`26`–`47`（logo 墙双份）、`72-producthunt.svg` |
| 04 AI team | `0`–`7`（agent 头像）、`8-card-background.webp`、`9-lower-right-wash.svg`、`48`–`51`（大头像） |
| 05/06 value & features | `59`–`63`（light 插图）、`64-5.CvYp2tZM.png`、`65-Number_26.png`、`77-consistent-design72.png`、`78-Union.svg` |
| 07 world | `66-United_States.png`、`73-kausik-lal.png`、`74-mia.png` |
| 08 inspire | `67`–`71`、`75-beau-carnes.png`、`76-stellar.png` |
| 11 final CTA | `83-starts.X91XbzmB.png` |
| meta | `90-favicon.ico`、`91-logo_only_blue.png` |

（文件名为"序号-原始名.hash.ext"，逐条源 URL 与字节数见 `_download-results.json`。）

## 未下载资产（记录为缺口）

1. **视频流（7 条）**：6 条为 HLS `.m3u8`（模板区 founder-builds 演示视频，poster 已下载）+ 1 条 mp4。逐条清单见 `.pytest_tmp/pw-extract/out/assets.json` 的 `videos` 字段。克隆时可用 poster 静帧替代，或按 m3u8 URL 另行抓取分片。
2. **字体二进制（199 族×weight 变体）**：经自托管 CSS（metadl CDN，26 个 font CSS 链接，清单见 `.pytest_tmp/pw-extract/out/extract.json` 的 `fontLinks`）加载。族：IBM Plex Sans / Serif / Mono、Inter、Yatra One。克隆时建议直接引用官方开源字体（IBM Plex 为 OFL 授权），无需逐一下载 CDN woff2。

## 采集环境

- playwright-core 1.62.1 + chromium_headless_shell-1228（本地缓存），viewport 1440×900（桌面）/768/390
- 原始采集数据：`.pytest_tmp/pw-extract/out/`（extract.json / tree.json / sections.json / sections-slim.json / behaviors.json / responsive.json / resp-mobile.json / resp-tablet.json / assets.json）
