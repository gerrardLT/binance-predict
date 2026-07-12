import { useState, useEffect, useCallback, Fragment } from 'react'
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
    <div className={`bg-white rounded-xl border border-gray-200 shadow-sm ${className}`}>
      <div className="px-5 py-3 border-b border-gray-100">
        <h2 className="text-sm font-semibold text-gray-700">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
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
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold text-gray-900">BTC 情绪曲线自进化 Agent 系统 V3</h1>
            {health && (
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <StatusDot ok={health.ws_spot_connected} />
                <span>现货</span>
                <StatusDot ok={health.rest_api_ok} />
                <span>REST</span>
              </div>
            )}
          </div>
          {health && (
            <div className="text-xs text-gray-400 font-mono">
              {health.symbol} ${health.mid_price > 0 ? health.mid_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '--'}
            </div>
          )}
        </div>
        <div className="max-w-6xl mx-auto px-4 flex gap-4">
          <button
            onClick={() => setTab('market')}
            className={`pb-2 text-sm font-medium border-b-2 transition ${tab === 'market' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            市场情绪
          </button>
          <button
            onClick={() => setTab('agent')}
            className={`pb-2 text-sm font-medium border-b-2 transition ${tab === 'agent' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            Agent 自进化
          </button>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
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
                  <ResponsiveContainer width="100%" height={400}>
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
    <div className="space-y-6">
      {/* (a) Agent 状态 */}
      <Card title="Agent 运行状态">
        {status ? (
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <StatusDot ok={status.scheduler_running} />
                <span className="text-sm text-gray-600">调度器</span>
              </div>
              <div className={`text-lg font-bold ${status.scheduler_running ? 'text-green-600' : 'text-red-600'}`}>
                {status.scheduler_running ? '运行中' : '已停止'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">ACTIVE 模式数</div>
              <div className="text-lg font-bold text-gray-900 font-mono">{status.active_pattern_count}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-1">累计验证次数</div>
              <div className="text-lg font-bold text-gray-900 font-mono">{status.validate_counter}</div>
            </div>
          </div>
        ) : <div className="text-gray-400 text-center text-sm">加载中...</div>}
      </Card>

      {/* (b) 模式库 */}
      <Card title="模式库（Pattern Memory）">
        <div className="flex justify-between items-center mb-3">
          <div className="text-xs text-gray-400">LLM 自主发现/命名/进化的情绪曲线模式。点击行查看详情与进化轨迹。</div>
          <button onClick={refreshPatterns} className="px-3 py-1 text-xs rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition">刷新</button>
        </div>
        {patterns.length === 0 ? (
          <div className="text-center text-gray-400 py-8 text-sm">暂无模式（Agent 冷启动学习中，需积累情绪窗口后自动发现）</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500">
                  <th className="py-2 px-2 text-left">模式名称</th>
                  <th className="py-2 px-2 text-left">方向</th>
                  <th className="py-2 px-2 text-left">状态</th>
                  <th className="py-2 px-2 text-right">胜率</th>
                  <th className="py-2 px-2 text-right">样本数</th>
                  <th className="py-2 px-2 text-right">置信度</th>
                </tr>
              </thead>
              <tbody>
                {patterns.map(p => (
                  <Fragment key={p.id}>
                    <tr className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer" onClick={() => setExpandedPattern(expandedPattern === p.id ? null : p.id)}>
                      <td className="py-2 px-2 font-medium text-gray-800">{p.pattern_name}</td>
                      <td className="py-2 px-2"><DirectionBadge direction={p.predicted_direction} /></td>
                      <td className="py-2 px-2"><StatusBadge status={p.status} /></td>
                      <td className="py-2 px-2 text-right font-mono">{(p.win_rate * 100).toFixed(1)}%</td>
                      <td className="py-2 px-2 text-right font-mono">{p.sample_count}</td>
                      <td className="py-2 px-2 text-right font-mono">{(p.confidence_score * 100).toFixed(0)}%</td>
                    </tr>
                    {expandedPattern === p.id && (
                      <tr className="bg-gray-50">
                        <td colSpan={6} className="py-3 px-4">
                          <div className="text-xs text-gray-700 mb-2"><b>描述：</b>{p.description}</div>
                          <div className="grid grid-cols-2 gap-3 mb-2">
                            <div>
                              <div className="text-xs font-bold text-gray-500 mb-1">曲线特征 (curve_features)</div>
                              <pre className="text-xs bg-white p-2 rounded border border-gray-200 overflow-x-auto max-h-40">{JSON.stringify(p.curve_features, null, 2)}</pre>
                            </div>
                            <div>
                              <div className="text-xs font-bold text-gray-500 mb-1">适用条件 (conditions)</div>
                              <pre className="text-xs bg-white p-2 rounded border border-gray-200 overflow-x-auto max-h-40">{JSON.stringify(p.conditions, null, 2)}</pre>
                            </div>
                          </div>
                          <button onClick={(e) => { e.stopPropagation(); toggleHistory(p.id) }} className="px-3 py-1 text-xs rounded bg-blue-100 text-blue-700 hover:bg-blue-200 transition">
                            {historyFor === p.id ? '收起进化轨迹' : '查看进化轨迹'}
                          </button>
                          {historyFor === p.id && (
                            <div className="mt-3">
                              {loading ? <div className="text-xs text-gray-400">加载中...</div> : history.length === 0 ? (
                                <div className="text-xs text-gray-400">暂无变更记录</div>
                              ) : (
                                <ul className="space-y-2">
                                  {history.map(h => (
                                    <li key={h.id} className="flex items-start gap-2 text-xs">
                                      <span className="text-gray-400 shrink-0 font-mono">{fmtTime(h.created_at)}</span>
                                      <ChangeTypeBadge type={h.change_type} />
                                      <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 shrink-0">{h.phase}</span>
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

      {/* (c) Agent 预测历史 */}
      <Card title="Agent 预测历史">
        <div className="flex justify-between items-center mb-3">
          <select
            value={dirFilter}
            onChange={e => { setDirFilter(e.target.value); refreshPredictions(e.target.value || undefined) }}
            className="px-3 py-1 border border-gray-200 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">全部方向</option>
            <option value="UP">UP</option>
            <option value="DOWN">DOWN</option>
            <option value="NO_TRADE">NO_TRADE</option>
          </select>
          <button onClick={() => refreshPredictions(dirFilter || undefined)} className="px-3 py-1 text-xs rounded bg-gray-100 text-gray-600 hover:bg-gray-200 transition">刷新</button>
        </div>
        {predictions.length === 0 ? (
          <div className="text-center text-gray-400 py-8 text-sm">暂无预测记录</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-gray-200 text-gray-500">
                  <th className="py-2 px-2 text-left">时间</th>
                  <th className="py-2 px-2 text-left">方向</th>
                  <th className="py-2 px-2 text-right">置信度</th>
                  <th className="py-2 px-2 text-left">匹配模式</th>
                  <th className="py-2 px-2 text-left">入场时机</th>
                  <th className="py-2 px-2 text-left">验证</th>
                  <th className="py-2 px-2 text-left">交易</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map(p => (
                  <tr key={p.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-2 px-2 text-gray-600 font-mono">{fmtTime(p.prediction_time)}</td>
                    <td className="py-2 px-2"><DirectionBadge direction={p.predicted_direction} /></td>
                    <td className="py-2 px-2 text-right font-mono">{(p.confidence * 100).toFixed(0)}%</td>
                    <td className="py-2 px-2 text-gray-700">{p.matched_pattern_name || '—'}</td>
                    <td className="py-2 px-2 text-gray-500">{p.entry_timing}</td>
                    <td className="py-2 px-2">
                      {p.is_correct === null ? <span className="text-gray-400">待验证</span> :
                        p.is_correct ? <span className="text-green-600 font-bold">✓ {p.actual_outcome}</span> :
                          <span className="text-red-600 font-bold">✗ {p.actual_outcome}</span>}
                    </td>
                    <td className="py-2 px-2">
                      {p.trade_order_id != null ? <span className="text-orange-600">已下单#{p.trade_order_id}</span> :
                        <span className="text-gray-400" title={p.skip_trade_reason || ''}>{p.skip_trade_reason ? '跳过' : '—'}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
