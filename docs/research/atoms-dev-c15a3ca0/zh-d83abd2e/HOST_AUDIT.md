# HOST_AUDIT — binance-predict frontend

审计时间：2026-08-22 · 迁移目标：atoms.dev 视觉语言 → dashboard 换肤（T4）

## 1. 框架
React 19 + Vite 8 + TypeScript 6（frontend/package.json）。单页应用，无路由库。

## 2. 渲染模型
纯 CSR（Vite dev / build）。无 SSR/RSC 边界问题。

## 3. CSS 方案
Tailwind CSS **v4**（@tailwindcss/vite 插件，`@import "tailwindcss"` 入口）。
App.tsx 内联使用 Tailwind 工具类（text-gray-800 等）+ 少量自定义组件类。

## 4. 现有设计系统
- `src/index.css` 定义 CSS 变量：--text / --text-h / --bg / --border / --accent(#aa3bff 紫) 等，
  含暗色模式（prefers-color-scheme: dark）。
- 字体：system-ui 栈 + @fontsource/ibm-plex-mono / ibm-plex-sans（已安装但当前未启用）。
- #root 固定宽 1126px 居中布局，带左右边框。
- 组件：Card / StatusDot / DirectionBadge / StatusBadge / ChangeTypeBadge /
  DiscoveryMethodBadge / EvoStat / MetricKV / PhaseBadge 等，均为 App.tsx 内组件。
- 图表：recharts 3。

## 5. 构建与验证命令（基线已通过 2026-08-22）
- `npm run build` = tsc -b && vite build ✅ 通过（仅 chunk>500kB 警告）
- `npm run lint` = oxlint

## 6. 集成目标（用户已确认）
不新增落地页、不迁移营销区块；将 atoms.dev 设计语言
（IBM Plex Sans 字体、#4267FF 品牌蓝、#F6F6F6 底色、16/24px 圆角、pill 按钮）
应用到现有 dashboard 全局视觉 —— T4 换肤，需全站回归检查。

## 迁移策略
宿主已是 React + Tailwind → 策略 A 变体：token 替换 + 组件类重样式，
不移植任何营销组件。暗色模式保留（atoms.dev 源站无暗色，暗色沿用宿主原值仅换字体/圆角）。

## Git 安全
迁移分支：style/atoms-dev-restyle（自 main 3a19510 切出）。
