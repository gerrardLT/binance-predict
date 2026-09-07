import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import type { IncomingMessage, ServerResponse } from 'node:http'

// dev proxy 目标。默认本地后端；指向生产 API 时用 VITE_DEV_API_TARGET 覆盖。
// 不把生产地址硬编码进仓库——它属于本地 .env 的范畴（见 AGENTS.md）。
const apiTarget = process.env.VITE_DEV_API_TARGET ?? 'http://127.0.0.1:8000'
// 锚定 host 段，避免把 "http://evil.com/?x=127.0.0.1" 之类误判成本地
const isLocalTarget = /\/\/(127\.0\.0\.1|localhost)(:|\/|$)/.test(apiTarget)

/**
 * 只读守卫：仅当 proxy 指向非本地后端时生效。
 *
 * 前端有真实写操作端点（如 POST /api/live/toggle 改通道开关/单笔金额/护栏，
 * 落 live_channel_overrides 表）。本地起 dev server 做视觉走查时，误点一次
 * 就是改生产实盘配置——真金在跑，不可逆。所以预览态默认放行 GET（看真实数据）
 * 与 POST /api/auth/login（登录门禁），其余写方法一律 403。
 *
 * 指向本地后端时不启用，保证 toggle 等交互仍可正常联调。
 */
const readOnlyBypass = (req: IncomingMessage, res: ServerResponse | undefined) => {
  const method = (req.method ?? 'GET').toUpperCase()
  const url = req.url ?? ''
  if (method === 'GET' || method === 'HEAD' || url.startsWith('/api/auth/login')) {
    return undefined // 放行，走代理
  }
  if (res && !res.headersSent) {
    res.statusCode = 403
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    res.end(
      JSON.stringify({
        detail: '[dev-proxy] 已拦截写操作：本地预览为只读，防止误改生产实盘配置',
      }),
    )
  }
  return false
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
        ...(isLocalTarget ? {} : { bypass: readOnlyBypass }),
      },
    },
  },
})
