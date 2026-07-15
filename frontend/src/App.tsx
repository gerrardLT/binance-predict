import { useState, useEffect, useCallback, useRef, Fragment } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine,
  BarChart, Bar, Legend, LineChart, Line,
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
  // Item 5：进化时钟与证据量挂钩（后端 status 端点新增字段，旧后端可能缺省）
  evolve_trigger_mode?: string
  new_validated_since_evolve?: number
  evolve_min_new_samples?: number
}

// 进化有效性看板（Item 1）：与后端 evolution_metrics 输出严格对齐
interface EvoBucket {
  sample_count: number
  correct: number
  win_rate: number
  ci_lower: number
  excess_over_random: number
  beats_random: boolean
}
interface EvoOverall extends EvoBucket {
  verdict: 'INSUFFICIENT_SAMPLES' | 'BEATS_RANDOM' | 'INCONCLUSIVE'
}
interface EvoTrendPoint extends EvoBucket {
  date: string
}
interface EvoGenerations {
  comparable: boolean
  older_half: EvoBucket
  newer_half: EvoBucket
  win_rate_delta: number
  significant_improvement: boolean
}
interface EvolutionReport {
  window_days: number
  total_validated: number
  decisive_count: number
  no_trade_count: number
  random_baseline: number
  overall: EvoOverall
  trend_daily: EvoTrendPoint[]
  generations: EvoGenerations
  by_discovery_method: Record<string, EvoBucket>
  summary: string
  generated_at: string
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
  discovery_method: 'LLM_DEEP' | 'PY_CLUSTER' | 'LEGACY'
  holdout_win_rate: number | null
  holdout_sample_count: number | null
  holdout_ci_lower: number | null
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
  discovery_method?: 'LLM_DEEP' | 'PY_CLUSTER'
  holdout_win_rate?: number | null
  holdout_sample_count?: number | null
  holdout_ci_lower?: number | null
}

// 运行监控：与后端 HealthReport（schemas.HealthReport）严格对齐
interface CalibrationBucket {
  range: string
  count: number
  avg_confidence: number
  hit_rate: number | null
  gap: number | null
}
interface HealthAlert {
  level: 'WARN' | 'CRITICAL'
  code: string
  message: string
}
interface HealthReport {
  generated_at: string
  overall_status: 'OK' | 'WARN' | 'CRITICAL'
  alerts: HealthAlert[]
  window_continuity: Record<string, number | null>
  predict_stats: Record<string, unknown>
  calibration: CalibrationBucket[]
  scheduler: Record<string, unknown>
  llm: Record<string, unknown>
  summary: string
}

// 方案对比：与后端 /deep-learn/compare 的每方法摘要对齐
interface CompareSummary {
  method: string | null
  discovery_count: number
  avg_holdout_win_rate: number
  avg_holdout_ci_lower: number
  total_holdout_samples: number
  avg_confidence: number
  passed_gate_count: number
  passed_gate_ratio: number
  direction_up: number
  direction_down: number
  snapshot_token: string | null
  train_count: number
  holdout_count: number
}
interface CompareResult {
  status: string
  snapshot_consistent: boolean
  comparison: CompareSummary[]
  llm: { reasoning: string; discoveries: DeepLearnDiscovery[] }
  pycluster: { reasoning: string; discoveries: DeepLearnDiscovery[] }
  message?: string
}
interface CompareLiveGroup {
  method: string
  pattern_count: number
  live_sample_count: number
  live_correct_count: number
  live_win_rate: number
  avg_confidence: number
  avg_holdout_ci_lower: number
}

// 深度学习流式（SSE）事件：与后端 deep_learn_stream 产出的 dict 严格对齐
interface DeepLearnStreamEvent {
  type: 'step' | 'reasoning' | 'progress' | 'done' | 'error'
  message?: string
  delta?: string
  discoveries?: number | DeepLearnDiscovery[]
  reasoning?: string
  method?: string
  snapshot_token?: string
  train_count?: number
  holdout_count?: number
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
  runPyClusterDeepLearn: (maxWindows = 100) =>
    fetch(`/api/sentiment/agent/deep-learn/pycluster?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  runCompare: (maxWindows = 100) =>
    fetch(`/api/sentiment/agent/deep-learn/compare?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  getCompareLive: () =>
    fetch('/api/sentiment/agent/deep-learn/compare/live').then(r => r.json()),
  getAgentHealth: () => fetch('/api/agent/health').then(r => r.json()),
  getAgentEvolution: (days = 30) =>
    fetch(`/api/sentiment/agent/evolution?days=${days}`).then(r => r.json()),
  commitDeepLearn: (discoveries: DeepLearnDiscovery[], snapshotToken?: string | null) =>
    fetch('/api/sentiment/agent/deep-learn/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ discoveries, snapshot_token: snapshotToken ?? null }),
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

function DiscoveryMethodBadge({ method }: { method?: string }) {
  const meta: Record<string, { label: string; cls: string }> = {
    LLM_DEEP: { label: 'LLM', cls: 'bg-purple-100 text-purple-700' },
    PY_CLUSTER: { label: 'PY聚类', cls: 'bg-teal-100 text-teal-700' },
    LEGACY: { label: '存量', cls: 'bg-gray-100 text-gray-500' },
  }
  const m = meta[method || 'LEGACY'] || meta.LEGACY
  return <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded ${m.cls}`}>{m.label}</span>
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
  const [tab, setTab] = useState<'market' | 'agent' | 'monitor'>('market')

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
            <button
              onClick={() => setTab('monitor')}
              className={`px-4 py-1.5 text-sm font-semibold rounded-md transition ${
                tab === 'monitor'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              运行监控
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

        {tab === 'monitor' && <MonitorTab />}
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
  const [cmpOpen, setCmpOpen] = useState(false)
  const [evoOpen, setEvoOpen] = useState(false)

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
            <div className="h-4 w-px bg-gray-200" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">距下次进化</span>
              <span className="text-sm font-bold text-gray-900 font-mono">
                {status.evolve_trigger_mode === 'samples' && status.evolve_min_new_samples != null
                  ? `${status.new_validated_since_evolve ?? 0}/${status.evolve_min_new_samples}`
                  : '—'}
              </span>
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
        <button
          onClick={() => setCmpOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-teal-600 rounded-lg hover:bg-teal-700 transition"
          title="同一数据上对比纯 LLM 版与 Python 聚类版的多维准确率"
        >
          ⚖️ 方案对比
        </button>
        <button
          onClick={() => setEvoOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition"
          title="用样本外胜率趋势/代际对比/分轨证明系统在变好而非只在变化"
        >
          📈 进化看板
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
                    <th className="py-1.5 px-1.5 text-left">方法</th>
                    <th className="py-1.5 px-1.5 text-left">方向</th>
                    <th className="py-1.5 px-1.5 text-left">状态</th>
                    <th className="py-1.5 px-1.5 text-right">Live 胜率</th>
                    <th className="py-1.5 px-1.5 text-right">Holdout</th>
                    <th className="py-1.5 px-1.5 text-right">样本</th>
                    <th className="py-1.5 px-1.5 text-right">置信度</th>
                  </tr>
                </thead>
                <tbody>
                  {patterns.map(p => (
                    <Fragment key={p.id}>
                      <tr className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer" onClick={() => setExpandedPattern(expandedPattern === p.id ? null : p.id)}>
                        <td className="py-1.5 px-1.5 font-medium text-gray-800">{p.pattern_name}</td>
                        <td className="py-1.5 px-1.5"><DiscoveryMethodBadge method={p.discovery_method} /></td>
                        <td className="py-1.5 px-1.5"><DirectionBadge direction={p.predicted_direction} /></td>
                        <td className="py-1.5 px-1.5"><StatusBadge status={p.status} /></td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.win_rate * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-1.5 text-right font-mono text-gray-500">{p.holdout_win_rate != null ? `${(p.holdout_win_rate * 100).toFixed(0)}%` : '—'}</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{p.sample_count}</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.confidence_score * 100).toFixed(0)}%</td>
                      </tr>
                      {expandedPattern === p.id && (
                        <tr className="bg-gray-50">
                          <td colSpan={8} className="py-2 px-3">
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

      {/* 方案对比：LLM vs Python 聚类多维对比（模态） */}
      {cmpOpen && <CompareModal onClose={() => setCmpOpen(false)} onCommitted={refreshPatterns} />}

      {/* 进化有效性看板（Item 1，模态） */}
      {evoOpen && <EvolutionModal onClose={() => setEvoOpen(false)} />}
    </div>
  )
}

// ============================================================
// 进化有效性看板（Item 1）：GET /api/sentiment/agent/evolution
// ============================================================

const VERDICT_META: Record<string, { label: string; cls: string }> = {
  BEATS_RANDOM: { label: '已显著跑赢随机', cls: 'bg-green-100 text-green-700 border-green-200' },
  INCONCLUSIVE: { label: '尚未显著', cls: 'bg-amber-100 text-amber-700 border-amber-200' },
  INSUFFICIENT_SAMPLES: { label: '样本不足', cls: 'bg-gray-100 text-gray-500 border-gray-200' },
}

function evoPct(v: number | null | undefined): string {
  return typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '--'
}

function EvoStat({ label, value, tone = 'text-gray-800' }: { label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className="bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">
      <div className="text-[10px] text-gray-400">{label}</div>
      <div className={`text-sm font-mono font-bold ${tone}`}>{value}</div>
    </div>
  )
}

function EvolutionModal({ onClose }: { onClose: () => void }) {
  const [days, setDays] = useState(30)
  const [report, setReport] = useState<EvolutionReport | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback((d: number) => {
    setLoading(true)
    api.getAgentEvolution(d)
      .then((r: EvolutionReport) => setReport(r))
      .catch(() => setReport(null))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { load(days) }, [days, load])

  const verdict = report?.overall.verdict ?? 'INSUFFICIENT_SAMPLES'
  const vm = VERDICT_META[verdict] ?? VERDICT_META.INSUFFICIENT_SAMPLES

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[88vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-sm font-bold text-gray-800">📈 进化有效性看板</h2>
            <p className="text-[11px] text-gray-400 mt-0.5">用样本外胜率证明「在变好」而非「只在变化」。仅统计已验证的决策预测（UP/DOWN），随机基线 50%。</p>
          </div>
          <div className="flex items-center gap-2">
            <select value={days} onChange={e => setDays(Number(e.target.value))}
              className="px-2 py-1 border border-gray-200 rounded-lg text-[11px] focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value={7}>近 7 天</option>
              <option value={30}>近 30 天</option>
              <option value={90}>近 90 天</option>
            </select>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-700 text-xl leading-none px-1">✕</button>
          </div>
        </div>

        {/* 主体 */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-4">
          {loading && <div className="text-center text-gray-400 py-10 text-sm">加载中...</div>}
          {!loading && !report && <div className="text-center text-gray-400 py-10 text-sm">暂无数据</div>}
          {!loading && report && (
            <>
              {/* 结论横幅 */}
              <div className={`rounded-lg border px-4 py-3 ${vm.cls}`}>
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-xs font-bold">{vm.label}</span>
                  <span className="text-[10px] opacity-70">窗口 {report.window_days} 天 · 已验证 {report.total_validated} 条 · 决策 {report.decisive_count} · 弃权(NO_TRADE) {report.no_trade_count}</span>
                </div>
                <p className="text-xs leading-relaxed">{report.summary}</p>
              </div>

              {/* 总体指标 */}
              <div>
                <div className="text-[11px] font-semibold text-gray-500 mb-1.5">总体（决策样本 = UP/DOWN）</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <EvoStat label="决策胜率" value={evoPct(report.overall.win_rate)} tone={report.overall.win_rate >= 0.5 ? 'text-green-600' : 'text-red-600'} />
                  <EvoStat label="Wilson 95% 下界" value={evoPct(report.overall.ci_lower)} tone={report.overall.beats_random ? 'text-green-600' : 'text-gray-800'} />
                  <EvoStat label="超额（vs 50%）" value={(report.overall.excess_over_random >= 0 ? '+' : '') + (report.overall.excess_over_random * 100).toFixed(1) + '%'} tone={report.overall.excess_over_random >= 0 ? 'text-green-600' : 'text-red-600'} />
                  <EvoStat label="跑赢随机？" value={report.overall.beats_random ? '是 ✓' : '否'} tone={report.overall.beats_random ? 'text-green-600' : 'text-gray-500'} />
                </div>
              </div>

              {/* 样本外胜率趋势 */}
              <div>
                <div className="text-[11px] font-semibold text-gray-500 mb-1.5">样本外胜率趋势（按天）</div>
                {report.trend_daily.length === 0 ? (
                  <div className="text-center text-gray-400 py-6 text-xs">暂无按天数据</div>
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={report.trend_daily.map(d => ({
                      date: d.date.slice(5),
                      win: +(d.win_rate * 100).toFixed(1),
                      ci: +(d.ci_lower * 100).toFixed(1),
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#9ca3af" />
                      <YAxis tick={{ fontSize: 10 }} stroke="#9ca3af" domain={[0, 100]} tickFormatter={(v: number) => v + '%'} />
                      <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                        formatter={(v, n) => [typeof v === 'number' ? v + '%' : '--', n === 'win' ? '胜率' : 'Wilson 下界']} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <ReferenceLine y={50} stroke="#9ca3af" strokeDasharray="4 4" />
                      <Line type="monotone" dataKey="win" stroke="#2563eb" strokeWidth={2} name="胜率" dot={{ r: 2 }} />
                      <Line type="monotone" dataKey="ci" stroke="#a855f7" strokeWidth={1.5} strokeDasharray="4 3" name="Wilson 下界" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>

              {/* 代际对比 */}
              <div>
                <div className="text-[11px] font-semibold text-gray-500 mb-1.5">代际对比（前半程 vs 近半程）</div>
                {!report.generations.comparable ? (
                  <div className="text-xs text-gray-400 bg-gray-50 rounded-lg px-3 py-2 border border-gray-100">两半程样本不足（各需 ≥15 决策样本），暂不下改善结论。</div>
                ) : (
                  <>
                    <div className="grid grid-cols-3 gap-2">
                      <EvoStat label={`前半程（n=${report.generations.older_half.sample_count}）`} value={evoPct(report.generations.older_half.win_rate)} />
                      <EvoStat label={`近半程（n=${report.generations.newer_half.sample_count}）`} value={evoPct(report.generations.newer_half.win_rate)} tone={report.generations.win_rate_delta >= 0 ? 'text-green-600' : 'text-red-600'} />
                      <EvoStat label="Δ 胜率" value={(report.generations.win_rate_delta >= 0 ? '+' : '') + (report.generations.win_rate_delta * 100).toFixed(1) + '%'} tone={report.generations.significant_improvement ? 'text-green-600' : report.generations.win_rate_delta > 0 ? 'text-amber-600' : 'text-red-600'} />
                    </div>
                    <div className="text-[11px] text-gray-500 mt-1">
                      {report.generations.significant_improvement
                        ? '✓ 近半程保守下界已超前半程点估计，是可信的改善信号。'
                        : report.generations.win_rate_delta > 0
                          ? '有改善迹象但未达显著，可能仍是波动。'
                          : '未见改善——警惕「只在变化、并未变好」。'}
                    </div>
                  </>
                )}
              </div>

              {/* 分发现方法 */}
              <div>
                <div className="text-[11px] font-semibold text-gray-500 mb-1.5">按发现方法分轨（哪条轨道真的产出 alpha）</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-200 text-gray-500">
                      <th className="py-1 px-2 text-left">方法</th>
                      <th className="py-1 px-2 text-right">样本</th>
                      <th className="py-1 px-2 text-right">胜率</th>
                      <th className="py-1 px-2 text-right">Wilson 下界</th>
                      <th className="py-1 px-2 text-center">跑赢随机</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.by_discovery_method).map(([m, s]) => (
                      <tr key={m} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-1 px-2 font-medium text-gray-700">{m}</td>
                        <td className="py-1 px-2 text-right font-mono">{s.sample_count}</td>
                        <td className="py-1 px-2 text-right font-mono">{evoPct(s.win_rate)}</td>
                        <td className="py-1 px-2 text-right font-mono">{evoPct(s.ci_lower)}</td>
                        <td className="py-1 px-2 text-center">{s.beats_random ? <span className="text-green-600 font-bold">✓</span> : <span className="text-gray-300">—</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="text-[10px] text-gray-400 text-right">生成于 {report.generated_at ? new Date(report.generated_at).toLocaleString('zh-CN') : '--'}</div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ============================================================
// 运行监控 Tab（GET /api/agent/health，30s 轮询）
// ============================================================

function healthTone(status: string): { dot: string; text: string; label: string } {
  if (status === 'OK') return { dot: 'bg-green-500', text: 'text-green-600', label: '正常' }
  if (status === 'WARN') return { dot: 'bg-yellow-500', text: 'text-yellow-600', label: '警告' }
  return { dot: 'bg-red-500', text: 'text-red-600', label: '严重' }
}

function MetricKV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-gray-50 last:border-0">
      <span className="text-[11px] text-gray-500">{label}</span>
      <span className="text-xs font-mono font-medium text-gray-800">{value}</span>
    </div>
  )
}

function fmtNum(v: unknown, digits = 2): string {
  if (v == null || typeof v !== 'number' || Number.isNaN(v)) return '--'
  return v.toFixed(digits)
}

function MonitorTab() {
  const [report, setReport] = useState<HealthReport | null>(null)
  const [err, setErr] = useState('')

  const refresh = useCallback(() => {
    api.getAgentHealth()
      .then(d => { if (d && d.overall_status) { setReport(d); setErr('') } else setErr('健康报告返回异常') })
      .catch(() => setErr('健康报告获取失败'))
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 30000)
    return () => clearInterval(timer)
  }, [refresh])

  if (!report) {
    return <div className="text-center text-gray-400 py-16 text-sm">{err || '加载健康报告中...'}</div>
  }

  const tone = healthTone(report.overall_status)
  const wc = report.window_continuity || {}
  const ps = report.predict_stats || {}
  const matchRate = typeof ps.match_rate === 'number' ? ps.match_rate : null
  const dirDist = (ps.direction_distribution as Record<string, number> | undefined) || {}

  return (
    <div className="space-y-3">
      {/* 总体状态红黄绿灯 + 诊断文本 */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm px-5 py-3">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <span className={`inline-block w-3.5 h-3.5 rounded-full ${tone.dot} animate-pulse`} />
            <span className={`text-base font-bold ${tone.text}`}>总体状态：{tone.label}</span>
            <span className="text-[11px] text-gray-400 font-mono">
              {report.generated_at ? new Date(report.generated_at).toLocaleString('zh-CN') : ''}
            </span>
          </div>
          <button onClick={refresh} className="px-2 py-0.5 text-[11px] rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition">立即刷新</button>
        </div>
        {report.summary && (
          <div className="mt-2 text-xs text-gray-700 bg-gray-50 rounded-lg p-3 whitespace-pre-wrap break-words">{report.summary}</div>
        )}
      </div>

      {/* 告警列表 */}
      <Card title={`告警（${report.alerts.length}）`}>
        {report.alerts.length === 0 ? (
          <div className="text-center text-gray-400 py-4 text-sm">无告警</div>
        ) : (
          <ul className="space-y-1.5">
            {report.alerts.map((a, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span className={`px-1.5 py-0.5 rounded font-bold shrink-0 ${a.level === 'CRITICAL' ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>{a.level}</span>
                <span className="font-mono text-gray-400 shrink-0">{a.code}</span>
                <span className="text-gray-700">{a.message}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* 窗口连续性 */}
        <Card title="窗口连续性">
          <MetricKV label="最新窗口龄期 (s)" value={fmtNum(wc.last_window_age_s, 0)} />
          <MetricKV label="缺口数 gap_count" value={wc.gap_count ?? '--'} />
          <MetricKV label="近期窗口数" value={wc.recent_count ?? '--'} />
          <MetricKV label="预期间隔 (s)" value={wc.expected_interval_s ?? '--'} />
        </Card>

        {/* predict 统计 */}
        <Card title="Predict 统计">
          <MetricKV label="总预测数" value={String(ps.total ?? '--')} />
          <MetricKV label="已匹配数" value={String(ps.matched ?? '--')} />
          <MetricKV label="匹配率" value={matchRate != null ? `${(matchRate * 100).toFixed(1)}%` : '--'} />
          <MetricKV label="ACTIVE 模式数" value={String(ps.active_pattern_count ?? '--')} />
          <MetricKV label="方向分布" value={`UP ${dirDist.UP ?? 0} / DOWN ${dirDist.DOWN ?? 0} / NO_TRADE ${dirDist.NO_TRADE ?? 0}`} />
        </Card>

        {/* 调度器 */}
        <Card title="调度器">
          {Object.keys(report.scheduler || {}).length === 0 ? (
            <div className="text-center text-gray-400 py-3 text-xs">无内存态数据</div>
          ) : (
            Object.entries(report.scheduler).map(([k, v]) => (
              <MetricKV key={k} label={k} value={typeof v === 'object' ? JSON.stringify(v) : String(v)} />
            ))
          )}
        </Card>

        {/* LLM 指标 */}
        <Card title="LLM 指标">
          {Object.keys(report.llm || {}).length === 0 ? (
            <div className="text-center text-gray-400 py-3 text-xs">无内存态数据</div>
          ) : (
            Object.entries(report.llm).map(([k, v]) => (
              <MetricKV key={k} label={k} value={typeof v === 'object' ? JSON.stringify(v) : String(v)} />
            ))
          )}
        </Card>
      </div>

      {/* 置信度校准分桶 */}
      <Card title="置信度校准（分桶）">
        {report.calibration.length === 0 ? (
          <div className="text-center text-gray-400 py-4 text-sm">样本不足，暂无校准数据</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 text-gray-500">
                <th className="py-1.5 px-1.5 text-left">区间</th>
                <th className="py-1.5 px-1.5 text-right">样本数</th>
                <th className="py-1.5 px-1.5 text-right">平均置信度</th>
                <th className="py-1.5 px-1.5 text-right">实际命中率</th>
                <th className="py-1.5 px-1.5 text-right">偏差 (gap)</th>
              </tr>
            </thead>
            <tbody>
              {report.calibration.map((b, i) => (
                <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                  <td className="py-1.5 px-1.5 font-mono text-gray-700">{b.range}</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{b.count}</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{(b.avg_confidence * 100).toFixed(0)}%</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{b.hit_rate != null ? `${(b.hit_rate * 100).toFixed(0)}%` : '--'}</td>
                  <td className={`py-1.5 px-1.5 text-right font-mono ${b.gap == null ? 'text-gray-400' : b.gap > 0.1 ? 'text-red-600' : b.gap < -0.1 ? 'text-blue-600' : 'text-gray-600'}`}>
                    {b.gap != null ? `${b.gap > 0 ? '+' : ''}${(b.gap * 100).toFixed(0)}%` : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
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
  const [snapshotToken, setSnapshotToken] = useState<string | null>(null)
  const [trainCount, setTrainCount] = useState(0)
  const [holdoutCount, setHoldoutCount] = useState(0)
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
    setSnapshotToken(null); setTrainCount(0); setHoldoutCount(0)
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
            setSnapshotToken(ev.snapshot_token ?? null)
            setTrainCount(ev.train_count ?? 0)
            setHoldoutCount(ev.holdout_count ?? 0)
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
      const res = await api.commitDeepLearn(selected, snapshotToken)
      if (res.status === 'ok') {
        const rejected = Array.isArray(res.rejected) ? res.rejected.length : 0
        const failed = Array.isArray(res.failed) ? res.failed.length : 0
        const extra = [rejected ? `未过闸门 ${rejected}` : '', failed ? `失败 ${failed}` : ''].filter(Boolean).join(' · ')
        setMsg(`✅ 已写入 ${res.written} 条模式到模式库${extra ? `（${extra}）` : ''}`)
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
                  {(trainCount > 0 || holdoutCount > 0) && (
                    <span className="ml-2 font-normal text-gray-400">train {trainCount} / holdout {holdoutCount}</span>
                  )}
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
                          {d.holdout_win_rate != null && (
                            <span className="text-[10px] text-teal-600 font-mono" title={`holdout 胜率 · 样本 ${d.holdout_sample_count ?? 0} · Wilson下界 ${d.holdout_ci_lower != null ? (d.holdout_ci_lower * 100).toFixed(0) + '%' : '—'}`}>
                              holdout {(d.holdout_win_rate * 100).toFixed(0)}%
                            </span>
                          )}
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

// ============================================================
// 方案对比模态：LLM vs Python 聚类 —— 同一 train/holdout 多维对比
// ============================================================

interface PyClusterResult {
  status: string
  reasoning: string
  discoveries: DeepLearnDiscovery[]
  count: number
  method: string
  snapshot_token: string | null
  train_count: number
  holdout_count: number
  message?: string
}

function pct(v: number | null | undefined): string {
  return v == null ? '—' : `${(v * 100).toFixed(0)}%`
}

function CompareModal({ onClose, onCommitted }: { onClose: () => void; onCommitted: () => void }) {
  const [maxWindows, setMaxWindows] = useState(100)
  const [busy, setBusy] = useState<'' | 'py' | 'cmp' | 'live' | 'commit'>('')
  const [msg, setMsg] = useState('')
  const [py, setPy] = useState<PyClusterResult | null>(null)
  const [pyChecked, setPyChecked] = useState<Set<number>>(new Set())
  const [cmp, setCmp] = useState<CompareResult | null>(null)
  const [live, setLive] = useState<CompareLiveGroup[]>([])

  useEffect(() => { loadLive() }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const loadLive = async () => {
    setBusy('live')
    try {
      const res = await api.getCompareLive()
      setLive(Array.isArray(res.groups) ? res.groups : [])
    } catch (e) { setMsg(`上线指标加载失败: ${(e as Error).message}`) }
    finally { setBusy('') }
  }

  const runPy = async () => {
    setBusy('py'); setMsg(''); setPy(null); setPyChecked(new Set())
    try {
      const res: PyClusterResult = await api.runPyClusterDeepLearn(maxWindows)
      if (res.status === 'ok') {
        setPy(res)
        setPyChecked(new Set(res.discoveries.map((_, i) => i)))
        if (res.discoveries.length === 0) setMsg('Python 聚类未产出任何模式（样本不足或全部未过闸门）')
      } else {
        setMsg(res.message || 'Python 聚类失败')
      }
    } catch (e) { setMsg(`请求失败: ${(e as Error).message}`) }
    finally { setBusy('') }
  }

  const runCmp = async () => {
    setBusy('cmp'); setMsg(''); setCmp(null)
    try {
      const res: CompareResult = await api.runCompare(maxWindows)
      if (res.status === 'ok') { setCmp(res) }
      else { setMsg(res.message || '对比失败') }
    } catch (e) { setMsg(`请求失败: ${(e as Error).message}`) }
    finally { setBusy('') }
  }

  const togglePy = (i: number) => {
    setPyChecked(prev => { const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n })
  }

  const commitPy = async () => {
    if (!py) return
    const selected = py.discoveries.filter((_, i) => pyChecked.has(i))
    if (selected.length === 0) { setMsg('请至少勾选一条 PY 聚类模式'); return }
    setBusy('commit'); setMsg('')
    try {
      const res = await api.commitDeepLearn(selected, py.snapshot_token)
      if (res.status === 'ok') {
        const rejected = Array.isArray(res.rejected) ? res.rejected.length : 0
        const failed = Array.isArray(res.failed) ? res.failed.length : 0
        const extra = [rejected ? `未过闸门 ${rejected}` : '', failed ? `失败 ${failed}` : ''].filter(Boolean).join(' · ')
        setMsg(`✅ 已写入 ${res.written} 条 PY 聚类模式${extra ? `（${extra}）` : ''}`)
        onCommitted(); loadLive()
        setPy(null); setPyChecked(new Set())
      } else { setMsg(res.message || '写入失败') }
    } catch (e) { setMsg(`请求失败: ${(e as Error).message}`) }
    finally { setBusy('') }
  }

  const byMethod = (m: string): CompareSummary | undefined =>
    cmp?.comparison.find(c => c.method === m)
  const llmSum = byMethod('LLM_DEEP')
  const pySum = byMethod('PY_CLUSTER')

  // 归一化到 0-1 的可比维度，画在同一张柱状图
  const chartData = cmp ? [
    { metric: 'Holdout胜率', LLM: llmSum?.avg_holdout_win_rate ?? 0, PY: pySum?.avg_holdout_win_rate ?? 0 },
    { metric: 'Wilson下界', LLM: llmSum?.avg_holdout_ci_lower ?? 0, PY: pySum?.avg_holdout_ci_lower ?? 0 },
    { metric: '平均置信', LLM: llmSum?.avg_confidence ?? 0, PY: pySum?.avg_confidence ?? 0 },
    { metric: '过闸门比', LLM: llmSum?.passed_gate_ratio ?? 0, PY: pySum?.passed_gate_ratio ?? 0 },
  ] : []

  const rows: { label: string; get: (s?: CompareSummary) => string }[] = [
    { label: '发现数', get: s => String(s?.discovery_count ?? 0) },
    { label: '平均 holdout 胜率', get: s => pct(s?.avg_holdout_win_rate) },
    { label: '平均 Wilson 下界', get: s => pct(s?.avg_holdout_ci_lower) },
    { label: 'holdout 样本量', get: s => String(s?.total_holdout_samples ?? 0) },
    { label: '平均 confidence', get: s => pct(s?.avg_confidence) },
    { label: '通过准入', get: s => `${s?.passed_gate_count ?? 0} / ${s?.discovery_count ?? 0}（${pct(s?.passed_gate_ratio)}）` },
    { label: '方向 UP / DOWN', get: s => `${s?.direction_up ?? 0} / ${s?.direction_down ?? 0}` },
  ]

  const pySelected = py ? py.discoveries.filter((_, i) => pyChecked.has(i)).length : 0

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <h2 className="text-base font-bold text-gray-800">⚖️ 方案对比 · LLM vs Python 聚类</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
        </div>

        {/* 控制栏 */}
        <div className="px-5 py-3 border-b border-gray-100 flex items-center gap-3 flex-wrap shrink-0">
          <label className="text-xs text-gray-600 flex items-center gap-1.5">
            窗口数
            <input
              type="number" min={1} max={500} value={maxWindows}
              onChange={e => setMaxWindows(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
              className="w-20 px-2 py-1 text-xs border border-gray-300 rounded"
            />
          </label>
          <button onClick={runPy} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-semibold text-white bg-teal-600 rounded-lg hover:bg-teal-700 disabled:opacity-50">
            {busy === 'py' ? '聚类中...' : '运行 Python 聚类版'}
          </button>
          <button onClick={runCmp} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50">
            {busy === 'cmp' ? '对比中（含 LLM）...' : '对比两套方案'}
          </button>
          <button onClick={loadLive} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50">
            刷新上线指标
          </button>
          {msg && <span className={`text-xs ${msg.startsWith('✅') ? 'text-green-600' : 'text-gray-500'}`}>{msg}</span>}
        </div>

        <div className="p-5 overflow-auto space-y-5">
          {/* 对比图表 + 表格 */}
          {cmp && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-gray-700">发现即时对比（同一 train/holdout）</span>
                {!cmp.snapshot_consistent && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-700" title="两套方案的数据快照不一致">⚠ 快照不一致</span>
                )}
              </div>
              <div className="h-56 bg-gray-50 rounded-lg border border-gray-200 p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="LLM" fill="#a855f7" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="PY" fill="#14b8a6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-200 text-gray-500">
                      <th className="py-1.5 px-2 text-left">维度</th>
                      <th className="py-1.5 px-2 text-right"><DiscoveryMethodBadge method="LLM_DEEP" /></th>
                      <th className="py-1.5 px-2 text-right"><DiscoveryMethodBadge method="PY_CLUSTER" /></th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(r => (
                      <tr key={r.label} className="border-b border-gray-100">
                        <td className="py-1.5 px-2 text-gray-600">{r.label}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-purple-700">{r.get(llmSum)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-teal-700">{r.get(pySum)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 上线真实指标（按 discovery_method 聚合） */}
          <div className="space-y-2">
            <span className="text-sm font-semibold text-gray-700">上线真实指标（模式库 ACTIVE，按来源聚合）</span>
            {live.length === 0 ? (
              <div className="text-xs text-gray-400">暂无上线数据</div>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-200 text-gray-500">
                      <th className="py-1.5 px-2 text-left">来源</th>
                      <th className="py-1.5 px-2 text-right">模式数</th>
                      <th className="py-1.5 px-2 text-right">样本量</th>
                      <th className="py-1.5 px-2 text-right">正确数</th>
                      <th className="py-1.5 px-2 text-right">Live 胜率</th>
                      <th className="py-1.5 px-2 text-right">平均置信</th>
                      <th className="py-1.5 px-2 text-right">平均Wilson下界</th>
                    </tr>
                  </thead>
                  <tbody>
                    {live.map(g => (
                      <tr key={g.method} className="border-b border-gray-100">
                        <td className="py-1.5 px-2"><DiscoveryMethodBadge method={g.method} /></td>
                        <td className="py-1.5 px-2 text-right font-mono">{g.pattern_count}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{g.live_sample_count}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{g.live_correct_count}</td>
                        <td className="py-1.5 px-2 text-right font-mono font-bold">{pct(g.live_win_rate)}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{pct(g.avg_confidence)}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{pct(g.avg_holdout_ci_lower)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* PY 聚类发现列表 + 勾选提交 */}
          {py && py.discoveries.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-700">
                  Python 聚类发现 {py.discoveries.length} 条 · 已选 {pySelected}
                  <span className="ml-2 font-normal text-gray-400">train {py.train_count} / holdout {py.holdout_count}</span>
                </span>
                <div className="flex gap-2">
                  <button onClick={() => setPyChecked(new Set(py.discoveries.map((_, i) => i)))} className="text-[10px] px-2 py-0.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200">全选</button>
                  <button onClick={() => setPyChecked(new Set())} className="text-[10px] px-2 py-0.5 rounded bg-gray-100 text-gray-600 hover:bg-gray-200">清空</button>
                </div>
              </div>
              <div className="space-y-2">
                {py.discoveries.map((d, i) => (
                  <label key={i} className="flex gap-2 items-start p-2.5 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50">
                    <input type="checkbox" checked={pyChecked.has(i)} onChange={() => togglePy(i)} className="mt-1 accent-teal-600" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <ChangeTypeBadge type={d.operation} />
                        <span className="font-semibold text-sm text-gray-800">{d.pattern_name}</span>
                        <DirectionBadge direction={d.predicted_direction} />
                        <span className="text-[10px] text-gray-500 font-mono">conf {(d.confidence_score * 100).toFixed(0)}%</span>
                        {d.holdout_win_rate != null && (
                          <span className="text-[10px] text-teal-600 font-mono" title={`holdout 样本 ${d.holdout_sample_count ?? 0} · Wilson下界 ${pct(d.holdout_ci_lower)}`}>
                            holdout {pct(d.holdout_win_rate)}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-gray-600 mb-1">{d.description}</div>
                      <div className="text-[11px] text-gray-500"><b>理由：</b>{d.change_reason}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 底部 */}
        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-end gap-2 shrink-0">
          <button onClick={onClose} className="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200">关闭</button>
          {py && py.discoveries.length > 0 && (
            <button onClick={commitPy} disabled={busy !== '' || pySelected === 0} className="px-3 py-1.5 text-xs font-semibold text-white bg-green-600 rounded-lg hover:bg-green-700 disabled:opacity-50">
              {busy === 'commit' ? '写入中...' : `写入选中的 ${pySelected} 条 PY 模式`}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
