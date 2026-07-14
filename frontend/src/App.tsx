import { useState, useEffect, useCallback, useRef, Fragment } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine,
} from 'recharts'

// ============================================================
// Types（与后端字段严格对齐）
// ============================================================

interface HealthData {
  status: string
  symbol: string
  mid_price: number
  ws_spot_connected: boolean
  rest_api_ok: boolean
}

interface PMPoint {
  timestamp: number
  up_price: number | null
  down_price: number | null
  up_pct: number | null
  down_pct: number | null
}

interface MomentumSignal {
  name: string
  value: number
  score: number
  description: string
}

interface MomentumResult {
  status: string
  direction: string
  confidence: number
  composite_score: number
  elapsed_seconds: number
  remaining_seconds: number
  sample_count: number
  signals: MomentumSignal[]
  reasoning: string[]
  message?: string
}

interface AgentStatus {
  validate_counter: number
  active_pattern_count: number
  scheduler_running: boolean
}

interface PatternMemory {
  id: number
  pattern_name: string
  description: string
  curve_features: Record<string, unknown>
  conditions: Record<string, unknown>
  predicted_direction: 'UP' | 'DOWN'
  win_rate: number
  sample_count: number
  correct_count: number
  confidence_score: number
  status: 'ACTIVE' | 'RETIRED' | 'EVOLVING'
  created_at: string | null
  updated_at: string | null
}

interface AgentPrediction {
  id: number
  prediction_time: string
  sentiment_window_id: number | null
  predicted_direction: 'UP' | 'DOWN' | 'NO_TRADE'
  matched_pattern_id: number | null
  matched_pattern_name: string | null
  confidence: number
  entry_timing: 'NOW' | 'WAIT' | 'SKIP'
  reasoning: string
  is_correct: boolean | null
  actual_outcome: string | null
  actual_return: number | null
  validated_at: string | null
  trade_order_id: number | null
  skip_trade_reason: string | null
  created_at: string | null
}

interface PatternChangeLog {
  id: number
  pattern_id: number
  change_type: 'CREATE' | 'UPDATE' | 'RETIRE'
  phase: 'LEARN' | 'EVOLVE'
  before_snapshot: Record<string, unknown> | null
  after_snapshot: Record<string, unknown> | null
  change_reason: string
  evolve_phase_id: string | null
  created_at: string | null
}

interface LLMTraceSummary {
  id: number
  phase: string
  model: string
  reasoning: string | null
  result_summary: string | null
  prompt_tokens: number | null
  completion_tokens: number | null
  estimated_cost_yuan: number | null
  latency_s: number | null
  created_at: string | null
}

interface LLMTraceDetail extends LLMTraceSummary {
  system_prompt: string
  user_message: string
  assistant_output: Record<string, unknown> | null
}

interface DeepLearnDiscovery {
  operation: 'CREATE' | 'UPDATE'
  target_pattern_id: number | null
  pattern_name: string
  description: string
  curve_features: Record<string, unknown>
  conditions: Record<string, unknown>
  predicted_direction: 'UP' | 'DOWN'
  confidence_score: number
  change_reason: string
}

// 深度学习流式（SSE）事件：与后端 deep_learn_stream 产出的 dict 严格对齐
interface DeepLearnStreamEvent {
  type: 'step' | 'reasoning' | 'progress' | 'done' | 'error'
  message?: string
  delta?: string
  discoveries?: number | DeepLearnDiscovery[]
  reasoning?: string
}

// ============================================================
// API helpers（仅保留路径B/C相关端点）
// ============================================================

const api = {
  health: () => fetch('/api/health').then(r => r.json()),
  getPredictionMarket: () => fetch('/api/chart/prediction-market').then(r => r.json()),
  runMomentumPredict: () => fetch('/api/sentiment/momentum-predict', { method: 'POST' }).then(r => r.json()),
  getAgentStatus: () => fetch('/api/sentiment/agent/status').then(r => r.json()),
  getAgentPatterns: () => fetch('/api/sentiment/agent/patterns').then(r => r.json()),
  getAgentPredictions: (direction?: string) =>
    fetch('/api/sentiment/agent/predictions' + (direction ? `?direction=${direction}` : '')).then(r => r.json()),
  getPatternHistory: (id: number) =>
    fetch(`/api/sentiment/agent/patterns/${id}/history`).then(r => r.json()),
  getLLMTraces: (phase?: string) =>
    fetch('/api/llm/traces' + (phase ? `?phase=${phase}` : '')).then(r => r.json()),
  getLLMTraceDetail: (id: number) =>
    fetch(`/api/llm/traces/${id}`).then(r => r.json()),
  triggerDeepLearn: (maxWindows = 100) =>
    fetch(`/api/sentiment/agent/deep-learn?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  commitDeepLearn: (discoveries: DeepLearnDiscovery[]) =>
    fetch('/api/sentiment/agent/deep-learn/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ discoveries }),
    }).then(r => r.json()),
}

// ============================================================
// 公共组件
// ============================================================

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${ok ? 'bg-green-500' : 'bg-red-500'}`} />
}

function DirectionBadge({ direction }: { direction: string }) {
  const colors: Record<string, string> = {
    UP: 'bg-green-100 text-green-800 border-green-300',
    DOWN: 'bg-red-100 text-red-800 border-red-300',
    NO_TRADE: 'bg-gray-100 text-gray-600 border-gray-300',
  }
  const label: Record<string, string> = { UP: '↑ 看涨', DOWN: '↓ 看跌', NO_TRADE: '⊘ 不交易' }
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-bold rounded-full border ${colors[direction] || 'bg-gray-100'}`}>
      {label[direction] || direction}
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    ACTIVE: 'bg-green-100 text-green-700',
    EVOLVING: 'bg-yellow-100 text-yellow-700',
    RETIRED: 'bg-gray-100 text-gray-500',
  }
  return <span className={`px-2 py-0.5 text-xs font-medium rounded ${colors[status] || 'bg-gray-100'}`}>{status}</span>
}

function ChangeTypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    CREATE: 'bg-blue-100 text-blue-700',
    UPDATE: 'bg-amber-100 text-amber-700',
    RETIRE: 'bg-gray-100 text-gray-500',
  }
  return <span className={`px-2 py-0.5 text-xs font-bold rounded ${colors[type] || 'bg-gray-100'}`}>{type}</span>
}

function Card({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col ${className}`}>
      <div className="px-4 py-2 border-b border-gray-100 shrink-0">
        <h2 className="text-sm font-semibold text-gray-700">{title}</h2>
      </div>
      <div className="p-4 flex-1 min-h-0 overflow-auto">{children}</div>
    </div>
  )
}

// ============================================================
// Main App
// ============================================================

export default function App() {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [tab, setTab] = useState<'market' | 'agent'>('market')

  // 市场情绪
  const [pmPoints, setPmPoints] = useState<PMPoint[]>([])
  const [pmMarket, setPmMarket] = useState<Record<string, unknown> | null>(null)
  const [momentumLoading, setMomentumLoading] = useState(false)
  const [momentumResult, setMomentumResult] = useState<MomentumResult | null>(null)

  const refreshHealth = useCallback(() => api.health().then(setHealth).catch(() => {}), [])
  const refreshPm = useCallback(() => {
    api.getPredictionMarket().then(d => {
      setPmPoints(d.points || [])
      setPmMarket(d.market || null)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    refreshHealth()
    const timer = setInterval(refreshHealth, 30000)
    return () => clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    if (tab !== 'market') return
    refreshPm()
    const timer = setInterval(refreshPm, 15000)
    return () => clearInterval(timer)
  }, [tab, refreshPm])

  const handleMomentum = async () => {
    setMomentumLoading(true)
    try {
      const res = await api.runMomentumPredict()
      if (res.status === 'ok') setMomentumResult(res)
      else alert(res.message || '分析失败')
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setMomentumLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-white/95 backdrop-blur border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-2 flex items-center justify-between gap-4 flex-wrap">
          {/* 左上角：紧凑标题 */}
          <div className="flex items-center gap-2 shrink-0">
            <span className="inline-block w-1.5 h-4 bg-blue-500 rounded-full" />
            <h1 className="text-sm font-bold text-gray-800 whitespace-nowrap">BTC 情绪 Agent V3</h1>
          </div>

          {/* 中部：标签切换（高对比度） */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setTab('market')}
              className={`px-4 py-1.5 text-sm font-semibold rounded-md transition ${
                tab === 'market'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              市场情绪
            </button>
            <button
              onClick={() => setTab('agent')}
              className={`px-4 py-1.5 text-sm font-semibold rounded-md transition ${
                tab === 'agent'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              Agent 自进化
            </button>
          </div>

          {/* 右上角：行情 + 价格 */}
          <div className="flex items-center gap-3 text-xs text-gray-500 shrink-0">
            {health && (
              <>
                <div className="flex items-center gap-1.5">
                  <StatusDot ok={health.ws_spot_connected} />
                  <span className="text-[10px]">现货</span>
                  <StatusDot ok={health.rest_api_ok} />
                  <span className="text-[10px]">REST</span>
                </div>
                <div className="font-mono text-gray-700 font-semibold">
                  ${health.mid_price > 0 ? health.mid_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'}
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-6xl w-full mx-auto px-4 py-4 flex-1">
        {tab === 'market' && (
          <div className="space-y-6">
            <Card title="BTC 5分钟内涨或跌（Binance Prediction Markets）">
              <div className="text-xs text-gray-400 mb-3">
                Binance 预测市场上所有交易者用真金白银投票的看多看空共识。每 15 秒自动刷新。
              </div>
              {pmMarket && (
                <div className="flex flex-wrap gap-4 mb-3 text-xs">
                  <span className="px-2 py-1 bg-yellow-50 text-yellow-700 rounded font-medium">🟡 Live</span>
                  <span className="text-gray-500">👥 {String(pmMarket.participant_count ?? '--')} 人参与</span>
                  <span className="text-gray-500">💰 ${String(pmMarket.trade_volume ?? '--')} 交易量</span>
                  {pmMarket.end_date ? (() => {
                    const remaining = Math.max(0, Math.floor(((pmMarket.end_date as number) - Date.now()) / 1000))
                    const min = Math.floor(remaining / 60)
                    const sec = remaining % 60
                    return <span className="text-orange-500 font-mono">⏱ {min}:{String(sec).padStart(2, '0')}</span>
                  })() : null}
                </div>
              )}
              {pmPoints.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={520}>
                    <AreaChart data={pmPoints.map(d => ({
                      time: new Date(d.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                      up_pct: d.up_pct,
                      down_pct: d.down_pct,
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="#9ca3af" interval="preserveStartEnd" />
                      <YAxis tick={{ fontSize: 11 }} stroke="#9ca3af" domain={[0, 100]} tickFormatter={(v: number) => v + '%'} />
                      <Tooltip
                        contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                        formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n === 'up_pct' ? '看涨 (UP)' : '看跌 (DOWN)']}
                      />
                      <Area type="monotone" dataKey="up_pct" stroke="#22c55e" fill="#22c55e20" strokeWidth={2} name="看涨" connectNulls />
                      <Area type="monotone" dataKey="down_pct" stroke="#ef4444" fill="#ef444420" strokeWidth={2} name="看跌" connectNulls />
                      <ReferenceLine y={50} stroke="#9ca3af" strokeDasharray="4 4" />
                    </AreaChart>
                  </ResponsiveContainer>
                  <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
                    <span>当前: <span className="text-green-600 font-medium">{pmPoints[pmPoints.length - 1]?.up_pct?.toFixed(1)}% 看涨</span> / <span className="text-red-500 font-medium">{pmPoints[pmPoints.length - 1]?.down_pct?.toFixed(1)}% 看跌</span></span>
                    <span>共 {pmPoints.length} 个采样点</span>
                  </div>
                </>
              ) : (
                <div className="text-center text-gray-400 py-10 text-sm">正在采集数据...每 15 秒采样一次。</div>
              )}
            </Card>

            <Card title="概率动量分析（纯算法 · 独立备选方案）">
              <div className="text-xs text-gray-400 mb-3">
                基于预测市场 UP% 时序的多维度动量信号，纯算法不依赖 LLM/K线。手动触发，不参与自动决策。
              </div>
              <button
                onClick={handleMomentum}
                disabled={momentumLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-cyan-600 rounded-lg hover:bg-cyan-700 disabled:opacity-50 transition mb-4"
              >
                {momentumLoading ? '📊 计算中...' : '📊 运行概率动量分析'}
              </button>
              {momentumResult && (
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <DirectionBadge direction={momentumResult.direction} />
                    <span className="text-sm text-gray-600">置信度: <strong>{(momentumResult.confidence * 100).toFixed(0)}%</strong></span>
                    <span className="text-sm text-gray-600">综合评分: <strong className="font-mono">{momentumResult.composite_score.toFixed(3)}</strong></span>
                  </div>
                  <div className="text-xs text-gray-500">
                    已过 {momentumResult.elapsed_seconds}s / 剩余 {momentumResult.remaining_seconds}s | {momentumResult.sample_count} 个采样点
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-gray-200 text-gray-500">
                          <th className="py-1 px-2 text-left">信号</th>
                          <th className="py-1 px-2 text-right">评分</th>
                          <th className="py-1 px-2 text-left">说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {momentumResult.signals.map((s, i) => (
                          <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-1 px-2 font-medium text-gray-700">{s.name}</td>
                            <td className={`py-1 px-2 text-right font-mono font-bold ${s.score > 0.1 ? 'text-green-600' : s.score < -0.1 ? 'text-red-600' : 'text-gray-400'}`}>
                              {s.score > 0 ? '+' : ''}{s.score.toFixed(3)}
                            </td>
                            <td className="py-1 px-2 text-gray-500">{s.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {momentumResult.reasoning.length > 0 && (
                    <div className="p-3 bg-cyan-50 rounded-lg border border-cyan-200">
                      <div className="text-xs font-bold text-cyan-800 mb-1">🧮 分析推理</div>
                      <ul className="text-xs text-gray-700 space-y-1">
                        {momentumResult.reasoning.map((r, i) => <li key={i}>{r}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </Card>
          </div>
        )}

        {tab === 'agent' && <AgentTab />}
      </main>

      {/* 右侧悬浮：LLM 轨迹面板（全局可见，5 秒轮询） */}
      <LLMTracePanel />
    </div>
  )
}

// ============================================================
// Agent 自进化 Tab
// ============================================================

function AgentTab() {
  const [status, setStatus] = useState<AgentStatus | null>(null)
  const [patterns, setPatterns] = useState<PatternMemory[]>([])
  const [predictions, setPredictions] = useState<AgentPrediction[]>([])
  const [dirFilter, setDirFilter] = useState<string>('')
  const [expandedPattern, setExpandedPattern] = useState<number | null>(null)
  const [history, setHistory] = useState<PatternChangeLog[]>([])
  const [historyFor, setHistoryFor] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [dlOpen, setDlOpen] = useState(false)

  const refreshStatus = useCallback(() => {
    api.getAgentStatus().then(setStatus).catch(() => {})
  }, [])
  const refreshPatterns = useCallback(() => {
    api.getAgentPatterns().then(d => setPatterns(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])
  const refreshPredictions = useCallback((direction?: string) => {
    api.getAgentPredictions(direction).then(d => setPredictions(Array.isArray(d) ? d : [])).catch(() => {})
  }, [])

  useEffect(() => {
    refreshStatus()
    refreshPatterns()
    refreshPredictions()
    const timer = setInterval(refreshStatus, 15000)
    return () => clearInterval(timer)
  }, [refreshStatus, refreshPatterns, refreshPredictions])

  const toggleHistory = async (id: number) => {
    if (historyFor === id) {
      setHistoryFor(null)
      setHistory([])
      return
    }
    setLoading(true)
    try {
      const d = await api.getPatternHistory(id)
      setHistory(Array.isArray(d) ? d : [])
      setHistoryFor(id)
    } catch {
      setHistory([])
    } finally {
      setLoading(false)
    }
  }

  const fmtTime = (s: string | null) => s ? new Date(s).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '--'

  return (
    <div className="flex flex-col gap-3 h-[calc(100vh-60px)]">
      {/* (a) Agent 状态（紧凑三指标横排） */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-5 py-2.5 shrink-0 flex items-center gap-4">
        {status ? (
          <div className="flex items-center justify-around gap-4 flex-1">
            <div className="flex items-center gap-2">
              <StatusDot ok={status.scheduler_running} />
              <span className="text-xs text-gray-500">调度器</span>
              <span className={`text-sm font-bold ${status.scheduler_running ? 'text-green-600' : 'text-red-600'}`}>
                {status.scheduler_running ? '运行中' : '已停止'}
              </span>
            </div>
            <div className="h-4 w-px bg-gray-200" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">ACTIVE 模式</span>
              <span className="text-sm font-bold text-gray-900 font-mono">{status.active_pattern_count}</span>
            </div>
            <div className="h-4 w-px bg-gray-200" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">累计验证</span>
              <span className="text-sm font-bold text-gray-900 font-mono">{status.validate_counter}</span>
            </div>
          </div>
        ) : <div className="text-gray-400 text-center text-sm flex-1">加载中...</div>}
        <button
          onClick={() => setDlOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition"
          title="全量历史深度分析：预览发现结果，审核后写入模式库"
        >
          🔬 深度学习
        </button>
      </div>

      {/* (b) 模式库 + 预测历史：左右并列 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 min-h-0 flex-1">
        {/* 模式库（Pattern Memory） */}
        <Card title="模式库（Pattern Memory）">
          <div className="flex justify-between items-center mb-2">
            <div className="text-[11px] text-gray-400">LLM 自主发现的情绪曲线模式，点击行查看详情</div>
            <button onClick={refreshPatterns} className="px-2 py-0.5 text-[11px] rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition">刷新</button>
          </div>
          {patterns.length === 0 ? (
            <div className="text-center text-gray-400 py-10 text-sm">暂无模式（需积累情绪窗口后自动发现）</div>
          ) : (
            <div className="overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500">
                    <th className="py-1.5 px-1.5 text-left">模式名称</th>
                    <th className="py-1.5 px-1.5 text-left">方向</th>
                    <th className="py-1.5 px-1.5 text-left">状态</th>
                    <th className="py-1.5 px-1.5 text-right">胜率</th>
                    <th className="py-1.5 px-1.5 text-right">样本</th>
                    <th className="py-1.5 px-1.5 text-right">置信度</th>
                  </tr>
                </thead>
                <tbody>
                  {patterns.map(p => (
                    <Fragment key={p.id}>
                      <tr className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer" onClick={() => setExpandedPattern(expandedPattern === p.id ? null : p.id)}>
                        <td className="py-1.5 px-1.5 font-medium text-gray-800">{p.pattern_name}</td>
                        <td className="py-1.5 px-1.5"><DirectionBadge direction={p.predicted_direction} /></td>
                        <td className="py-1.5 px-1.5"><StatusBadge status={p.status} /></td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.win_rate * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{p.sample_count}</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.confidence_score * 100).toFixed(0)}%</td>
                      </tr>
                      {expandedPattern === p.id && (
                        <tr className="bg-gray-50">
                          <td colSpan={6} className="py-2 px-3">
                            <div className="text-xs text-gray-700 mb-2"><b>描述：</b>{p.description}</div>
                            <div className="grid grid-cols-2 gap-2 mb-2">
                              <div>
                                <div className="text-[10px] font-bold text-gray-500 mb-1">曲线特征</div>
                                <pre className="text-[10px] bg-white p-1.5 rounded border border-gray-200 overflow-x-auto max-h-32">{JSON.stringify(p.curve_features, null, 2)}</pre>
                              </div>
                              <div>
                                <div className="text-[10px] font-bold text-gray-500 mb-1">适用条件</div>
                                <pre className="text-[10px] bg-white p-1.5 rounded border border-gray-200 overflow-x-auto max-h-32">{JSON.stringify(p.conditions, null, 2)}</pre>
                              </div>
                            </div>
                            <button onClick={(e) => { e.stopPropagation(); toggleHistory(p.id) }} className="px-2 py-0.5 text-[10px] rounded bg-blue-100 text-blue-700 hover:bg-blue-200 transition">
                              {historyFor === p.id ? '收起进化轨迹' : '查看进化轨迹'}
                            </button>
                            {historyFor === p.id && (
                              <div className="mt-2">
                                {loading ? <div className="text-[10px] text-gray-400">加载中...</div> : history.length === 0 ? (
                                  <div className="text-[10px] text-gray-400">暂无变更记录</div>
                                ) : (
                                  <ul className="space-y-1.5">
                                    {history.map(h => (
                                      <li key={h.id} className="flex items-start gap-1.5 text-[10px]">
                                        <span className="text-gray-400 shrink-0 font-mono">{fmtTime(h.created_at)}</span>
                                        <ChangeTypeBadge type={h.change_type} />
                                        <span className="px-1 py-0.5 rounded bg-gray-100 text-gray-500 shrink-0">{h.phase}</span>
                                        <span className="text-gray-700">{h.change_reason}</span>
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* Agent 预测历史 */}
        <Card title="Agent 预测历史">
          <div className="flex justify-between items-center mb-2">
            <select
              value={dirFilter}
              onChange={e => { setDirFilter(e.target.value); refreshPredictions(e.target.value || undefined) }}
              className="px-2 py-0.5 border border-gray-200 rounded-lg text-[11px] focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">全部方向</option>
              <option value="UP">UP</option>
              <option value="DOWN">DOWN</option>
              <option value="NO_TRADE">NO_TRADE</option>
            </select>
            <button onClick={() => refreshPredictions(dirFilter || undefined)} className="px-2 py-0.5 text-[11px] rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition">刷新</button>
          </div>
          {predictions.length === 0 ? (
            <div className="text-center text-gray-400 py-10 text-sm">暂无预测记录</div>
          ) : (
            <div className="overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500">
                    <th className="py-1.5 px-1.5 text-left">时间</th>
                    <th className="py-1.5 px-1.5 text-left">方向</th>
                    <th className="py-1.5 px-1.5 text-right">置信度</th>
                    <th className="py-1.5 px-1.5 text-left">匹配模式</th>
                    <th className="py-1.5 px-1.5 text-left">验证</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map(p => (
                    <tr key={p.id} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-1.5 px-1.5 text-gray-600 font-mono">{fmtTime(p.prediction_time)}</td>
                      <td className="py-1.5 px-1.5"><DirectionBadge direction={p.predicted_direction} /></td>
                      <td className="py-1.5 px-1.5 text-right font-mono">{(p.confidence * 100).toFixed(0)}%</td>
                      <td className="py-1.5 px-1.5 text-gray-700 truncate max-w-[120px]" title={p.matched_pattern_name || ''}>{p.matched_pattern_name || '—'}</td>
                      <td className="py-1.5 px-1.5">
                        {p.is_correct === null ? <span className="text-gray-400">待验证</span> :
                          p.is_correct ? <span className="text-green-600 font-bold">✓ {p.actual_outcome}</span> :
                            <span className="text-red-600 font-bold">✗ {p.actual_outcome}</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      {/* 深度学习：预览 + 审核 + 提交（模态） */}
      {dlOpen && <DeepLearnModal onClose={() => setDlOpen(false)} onCommitted={refreshPatterns} />}
    </div>
  )
}

// ============================================================
// LLM 轨迹面板（右侧悬浮抽屉，5 秒轮询）
// ============================================================

const PHASE_META: Record<string, { label: string; cls: string }> = {
  LEARN: { label: 'LEARN', cls: 'bg-blue-100 text-blue-700' },
  DEEP_LEARN: { label: 'DEEP', cls: 'bg-purple-100 text-purple-700' },
  PREDICT: { label: 'PREDICT', cls: 'bg-green-100 text-green-700' },
  EVOLVE: { label: 'EVOLVE', cls: 'bg-amber-100 text-amber-700' },
}

function PhaseBadge({ phase }: { phase: string }) {
  const m = PHASE_META[phase] || { label: phase, cls: 'bg-gray-100 text-gray-600' }
  return <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${m.cls}`}>{m.label}</span>
}

function LLMTracePanel() {
  const [open, setOpen] = useState(false)
  const [traces, setTraces] = useState<LLMTraceSummary[]>([])
  const [phaseFilter, setPhaseFilter] = useState<string>('')
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [detail, setDetail] = useState<LLMTraceDetail | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)

  const refresh = useCallback(() => {
    api.getLLMTraces(phaseFilter || undefined)
      .then(d => setTraces(Array.isArray(d) ? d : []))
      .catch(() => {})
  }, [phaseFilter])

  // 仅在面板打开时轮询（5 秒）
  useEffect(() => {
    if (!open) return
    refresh()
    const timer = setInterval(refresh, 5000)
    return () => clearInterval(timer)
  }, [open, refresh])

  const toggleDetail = async (id: number) => {
    if (expandedId === id) {
      setExpandedId(null)
      setDetail(null)
      return
    }
    setExpandedId(id)
    setDetail(null)
    setLoadingDetail(true)
    try {
      const d = await api.getLLMTraceDetail(id)
      setDetail(d && typeof d === 'object' && 'id' in d ? d : null)
    } catch {
      setDetail(null)
    } finally {
      setLoadingDetail(false)
    }
  }

  const fmtTime = (s: string | null) =>
    s ? new Date(s).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '--'

  return (
    <>
      {/* 悬浮触发按钮（右侧边缘，竖排文字） */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed right-0 top-1/2 -translate-y-1/2 z-40 bg-indigo-600 text-white text-xs font-bold px-2 py-3 rounded-l-lg shadow-lg hover:bg-indigo-700 transition"
          style={{ writingMode: 'vertical-rl' }}
          title="查看 LLM 调用轨迹"
        >
          🧠 LLM 轨迹
        </button>
      )}

      {/* 右侧抽屉 */}
      <div
        className={`fixed top-0 right-0 h-screen w-[440px] max-w-[92vw] bg-white shadow-2xl border-l border-gray-200 z-50 flex flex-col transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        {/* 头部 */}
        <div className="px-4 py-2.5 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-gray-800">🧠 LLM 调用轨迹</span>
            <span className="text-[10px] text-gray-400">每 5 秒刷新</span>
          </div>
          <button onClick={() => setOpen(false)} className="text-gray-400 hover:text-gray-700 text-lg leading-none px-1">✕</button>
        </div>

        {/* 阶段筛选 */}
        <div className="px-4 py-2 border-b border-gray-100 flex items-center gap-1 flex-wrap shrink-0">
          {['', 'LEARN', 'DEEP_LEARN', 'PREDICT', 'EVOLVE'].map(p => (
            <button
              key={p || 'ALL'}
              onClick={() => { setPhaseFilter(p); setExpandedId(null); setDetail(null) }}
              className={`px-2 py-0.5 text-[10px] font-medium rounded transition ${
                phaseFilter === p ? 'bg-indigo-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {p === '' ? '全部' : (PHASE_META[p]?.label ?? p)}
            </button>
          ))}
        </div>

        {/* 轨迹列表 */}
        <div className="flex-1 min-h-0 overflow-auto p-3 space-y-2">
          {traces.length === 0 ? (
            <div className="text-center text-gray-400 py-10 text-sm">暂无 LLM 调用记录</div>
          ) : (
            traces.map(t => (
              <div key={t.id} className="border border-gray-200 rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleDetail(t.id)}
                  className="w-full text-left px-3 py-2 hover:bg-gray-50 transition"
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="flex items-center gap-1.5">
                      <PhaseBadge phase={t.phase} />
                      <span className="text-[10px] text-gray-400 font-mono">{fmtTime(t.created_at)}</span>
                    </div>
                    <span className="text-[10px] text-gray-400 font-mono">
                      {t.latency_s != null ? `${t.latency_s.toFixed(1)}s` : ''}
                    </span>
                  </div>
                  {t.result_summary && (
                    <div className="text-[11px] font-mono text-indigo-700 mb-0.5">{t.result_summary}</div>
                  )}
                  {t.reasoning && (
                    <div className="text-[11px] text-gray-600 line-clamp-2">{t.reasoning}</div>
                  )}
                  <div className="flex items-center gap-3 mt-1 text-[10px] text-gray-400 font-mono">
                    <span>tok {t.prompt_tokens ?? '?'}/{t.completion_tokens ?? '?'}</span>
                    {t.estimated_cost_yuan != null && <span>¥{t.estimated_cost_yuan.toFixed(4)}</span>}
                    <span className="truncate">{t.model}</span>
                  </div>
                </button>

                {/* 展开详情 */}
                {expandedId === t.id && (
                  <div className="border-t border-gray-100 bg-gray-50 px-3 py-2 space-y-2">
                    {loadingDetail ? (
                      <div className="text-[10px] text-gray-400">加载详情中...</div>
                    ) : !detail ? (
                      <div className="text-[10px] text-red-400">详情加载失败</div>
                    ) : (
                      <>
                        <TraceSection title="Reasoning（推理）" text={detail.reasoning || '（无）'} />
                        <TraceSection title="System Prompt（系统提示词）" text={detail.system_prompt} collapsedHeight />
                        <TraceSection title="User Message（输入）" text={detail.user_message} collapsedHeight />
                        <div>
                          <div className="text-[10px] font-bold text-gray-500 mb-1">Assistant Output（结构化输出）</div>
                          <pre className="text-[10px] bg-white p-1.5 rounded border border-gray-200 overflow-auto max-h-64 whitespace-pre-wrap break-words">
                            {detail.assistant_output ? JSON.stringify(detail.assistant_output, null, 2) : '（无）'}
                          </pre>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}

function TraceSection({ title, text, collapsedHeight = false }: { title: string; text: string; collapsedHeight?: boolean }) {
  return (
    <div>
      <div className="text-[10px] font-bold text-gray-500 mb-1">{title}</div>
      <pre className={`text-[10px] bg-white p-1.5 rounded border border-gray-200 overflow-auto whitespace-pre-wrap break-words ${collapsedHeight ? 'max-h-40' : 'max-h-64'}`}>
        {text}
      </pre>
    </div>
  )
}

// ============================================================
// 深度学习模态：全量分析预览 → 勾选审核 → 写入模式库
// ============================================================

function DeepLearnModal({ onClose, onCommitted }: { onClose: () => void; onCommitted: () => void }) {
  const [maxWindows, setMaxWindows] = useState(100)
  const [phase, setPhase] = useState<'idle' | 'analyzing' | 'review' | 'committing'>('idle')
  const [reasoning, setReasoning] = useState('')
  const [discoveries, setDiscoveries] = useState<DeepLearnDiscovery[]>([])
  const [checked, setChecked] = useState<Set<number>>(new Set())
  const [msg, setMsg] = useState('')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [liveLog, setLiveLog] = useState<string[]>([])
  const [progressCount, setProgressCount] = useState(0)
  const reasoningRef = useRef<HTMLPreElement>(null)

  // reasoning 增量到达时自动滚到底部（打字机跟随）
  useEffect(() => {
    const el = reasoningRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [reasoning])

  const runAnalyze = async () => {
    setPhase('analyzing'); setMsg('')
    setReasoning(''); setDiscoveries([]); setChecked(new Set())
    setLiveLog([]); setProgressCount(0)
    try {
      const resp = await fetch(
        `/api/sentiment/agent/deep-learn/stream?max_windows=${maxWindows}`,
        { method: 'POST' },
      )
      if (!resp.ok || !resp.body) {
        setMsg(`请求失败: HTTP ${resp.status}`); setPhase('idle'); return
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      let reasoningAcc = ''
      let doneReceived = false

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        // SSE 帧以空行分隔：data: <json>\n\n
        const frames = buf.split('\n\n')
        buf = frames.pop() ?? ''
        for (const frame of frames) {
          const line = frame.replace(/^data:\s?/, '').trim()
          if (!line) continue
          let ev: DeepLearnStreamEvent
          try { ev = JSON.parse(line) } catch { continue }
          if (ev.type === 'step') {
            setLiveLog(prev => [...prev, ev.message ?? ''])
          } else if (ev.type === 'reasoning') {
            reasoningAcc += ev.delta ?? ''
            setReasoning(reasoningAcc)
          } else if (ev.type === 'progress') {
            setProgressCount(typeof ev.discoveries === 'number' ? ev.discoveries : 0)
          } else if (ev.type === 'error') {
            setMsg(`分析失败: ${ev.message ?? '未知错误'}`); setPhase('idle'); return
          } else if (ev.type === 'done') {
            doneReceived = true
            const ds: DeepLearnDiscovery[] = Array.isArray(ev.discoveries) ? ev.discoveries : []
            setReasoning(ev.reasoning || reasoningAcc)
            setDiscoveries(ds)
            setChecked(new Set(ds.map((_, i) => i)))  // 默认全选
            setPhase('review')
            if (ds.length === 0) setMsg('LLM 未发现任何新模式（本次已产生一条 DEEP_LEARN 轨迹）')
          }
        }
      }
      if (!doneReceived) { setMsg('分析连接中断（未收到完成信号）'); setPhase('idle') }
    } catch (e) {
      setMsg(`请求失败: ${(e as Error).message}`); setPhase('idle')
    }
  }

  const toggle = (i: number) => {
    setChecked(prev => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i); else next.add(i)
      return next
    })
  }

  const runCommit = async () => {
    const selected = discoveries.filter((_, i) => checked.has(i))
    if (selected.length === 0) { setMsg('请至少勾选一条模式'); return }
    setPhase('committing'); setMsg('')
    try {
      const res = await api.commitDeepLearn(selected)
      if (res.status === 'ok') {
        setMsg(`✅ 已写入 ${res.written} 条模式到模式库`)
        onCommitted()
        setDiscoveries([]); setChecked(new Set())
        setPhase('review')
      } else if (res.status === 'busy') {
        setMsg(res.message || '写入冲突，请重试'); setPhase('review')
      } else {
        setMsg(res.message || '写入失败'); setPhase('review')
      }
    } catch (e) {
      setMsg(`请求失败: ${(e as Error).message}`); setPhase('review')
    }
  }

  const selectedCount = checked.size

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-sm font-bold text-gray-800">🔬 深度模式学习</h2>
            <p className="text-[11px] text-gray-400 mt-0.5">全量历史窗口深度分析，预览发现结果，勾选后写入模式库</p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 text-xl leading-none px-1">✕</button>
        </div>

        {/* 主体 */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-4">
          {/* 分析参数 */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-600">分析窗口数上限</label>
            <input
              type="number"
              min={1}
              value={maxWindows}
              onChange={e => setMaxWindows(Math.max(1, Number(e.target.value) || 1))}
              disabled={phase === 'analyzing' || phase === 'committing'}
              className="w-24 px-2 py-1 border border-gray-200 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100"
            />
            <button
              onClick={runAnalyze}
              disabled={phase === 'analyzing' || phase === 'committing'}
              className="px-3 py-1.5 text-xs font-semibold text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 transition"
            >
              {phase === 'analyzing' ? '🔄 LLM 分析中...' : '开始分析'}
            </button>
          </div>

          {/* 实时日志（阶段性进度） */}
          {liveLog.length > 0 && (
            <div>
              <div className="text-[11px] font-bold text-gray-500 mb-1">📡 实时日志</div>
              <div className="text-[11px] font-mono text-gray-600 bg-gray-50 p-2 rounded border border-gray-200 space-y-0.5">
                {liveLog.map((l, i) => (
                  <div key={i} className="flex gap-1.5">
                    <span className="text-gray-300 shrink-0">›</span>
                    <span className="break-words">{l}</span>
                  </div>
                ))}
                {phase === 'analyzing' && (
                  <div className="flex gap-1.5 text-purple-500">
                    <span className="shrink-0">›</span>
                    <span>LLM 流式生成中{progressCount > 0 ? ` · 已解析 ${progressCount} 条模式` : '…'}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 推理过程（流式打字机） */}
          {(reasoning || phase === 'analyzing') && (
            <div>
              <div className="text-[11px] font-bold text-gray-500 mb-1">
                🧠 分析推理{phase === 'analyzing' && <span className="ml-1 text-purple-500 font-normal">（实时）</span>}
              </div>
              <pre
                ref={reasoningRef}
                className="text-[11px] text-gray-700 bg-gray-50 p-2 rounded border border-gray-200 overflow-auto max-h-52 whitespace-pre-wrap break-words"
              >
                {reasoning}
                {phase === 'analyzing' && <span className="inline-block w-1.5 h-3 ml-0.5 align-middle bg-purple-500 animate-pulse" />}
              </pre>
            </div>
          )}

          {/* 发现列表 */}
          {discoveries.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-[11px] font-bold text-gray-500">
                  发现 {discoveries.length} 条 · 已选 {selectedCount} 条
                </div>
                <div className="flex gap-2">
                  <button onClick={() => setChecked(new Set(discoveries.map((_, i) => i)))} className="text-[10px] px-2 py-0.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200">全选</button>
                  <button onClick={() => setChecked(new Set())} className="text-[10px] px-2 py-0.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200">清空</button>
                </div>
              </div>
              <div className="space-y-2">
                {discoveries.map((d, i) => (
                  <div key={i} className="border border-gray-200 rounded-lg">
                    <label className="flex gap-2 items-start p-2.5 cursor-pointer hover:bg-gray-50">
                      <input type="checkbox" checked={checked.has(i)} onChange={() => toggle(i)} className="mt-1 accent-purple-600" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <ChangeTypeBadge type={d.operation} />
                          <span className="font-semibold text-sm text-gray-800">{d.pattern_name}</span>
                          <DirectionBadge direction={d.predicted_direction} />
                          <span className="text-[10px] text-gray-500 font-mono">conf {(d.confidence_score * 100).toFixed(0)}%</span>
                          {d.operation === 'UPDATE' && d.target_pattern_id != null && (
                            <span className="text-[10px] text-amber-600 font-mono">→ 更新 #{d.target_pattern_id}</span>
                          )}
                        </div>
                        <div className="text-xs text-gray-600 mb-1">{d.description}</div>
                        <div className="text-[11px] text-gray-500"><b>理由：</b>{d.change_reason}</div>
                        <button
                          type="button"
                          onClick={e => { e.preventDefault(); setExpanded(expanded === i ? null : i) }}
                          className="mt-1 text-[10px] text-blue-600 hover:underline"
                        >
                          {expanded === i ? '收起特征/条件' : '查看特征/条件'}
                        </button>
                        {expanded === i && (
                          <div className="grid grid-cols-2 gap-2 mt-1.5">
                            <div>
                              <div className="text-[10px] font-bold text-gray-400 mb-0.5">曲线特征</div>
                              <pre className="text-[10px] bg-gray-50 p-1.5 rounded border border-gray-200 overflow-auto max-h-32">{JSON.stringify(d.curve_features, null, 2)}</pre>
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-gray-400 mb-0.5">适用条件</div>
                              <pre className="text-[10px] bg-gray-50 p-1.5 rounded border border-gray-200 overflow-auto max-h-32">{JSON.stringify(d.conditions, null, 2)}</pre>
                            </div>
                          </div>
                        )}
                      </div>
                    </label>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 底部 */}
        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-between gap-3 shrink-0">
          <span className={`text-xs ${msg.startsWith('✅') ? 'text-green-600' : 'text-gray-500'}`}>{msg}</span>
          <div className="flex gap-2 shrink-0">
            <button onClick={onClose} className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 transition">关闭</button>
            {discoveries.length > 0 && (
              <button
                onClick={runCommit}
                disabled={phase === 'committing' || selectedCount === 0}
                className="px-3 py-1.5 text-xs font-semibold text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
              >
                {phase === 'committing' ? '写入中...' : `写入选中的 ${selectedCount} 条模式`}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
