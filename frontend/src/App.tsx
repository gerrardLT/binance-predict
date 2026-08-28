import { useState, useEffect, useCallback, useRef, Fragment } from 'react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine,
  BarChart, Bar, Legend, LineChart, Line,
  ReferenceArea, ReferenceDot,
} from 'recharts'

// ============================================================
// Types（与后端字段严格对齐）
// ============================================================

interface PMPoint {
  timestamp: number
  up_price: number | null
  down_price: number | null
  up_pct: number | null
  down_pct: number | null
  btc_price?: number | null
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

// 假突破信号：与后端 /api/fake-breakout/* 输出对齐
interface FakeBreakoutSignal {
  id: number
  level: '1h' | '4h' | 'daily' | 'momentum'
  side: 'high' | 'low'
  signal_time: number
  resistance: number
  btc_price: number
  down_price_5m: number | null
  down_price_15m: number | null
  up_price_5m: number | null
  up_price_15m: number | null
  market_end_15m: number | null
  market_start_15m: number | null
  cycle_open_price_15m: number | null
  market_start_5m: number | null
  market_end_5m: number | null
  cycle_open_price_5m: number | null
  cycle_offset_sec_15m: number | null
  break_pct: number | null
  pattern: string | null
  pattern_type: string | null
  close_pos: number | null
  vol_ratio: number | null
  ev_at_entry: number | null
  cumulative_winrate: number | null
  cumulative_ev: number | null
  n_events_last_7d: number | null
  entry_down_price_15m: number | null
  entry_up_price_15m: number | null
  entry_quote_ts_15m: number | null
  add_down_price_15m: number | null
  add_up_price_15m: number | null
  add_trigger_ts_15m: number | null
  quote5m_down_15m: number | null
  quote5m_up_15m: number | null
  quote5m_ts_15m: number | null
  settle_btc_price: number | null
  settle_outcome: 'UP' | 'DOWN' | 'NOISE' | null
  settle_btc_price_5m: number | null
  settle_outcome_5m: 'UP' | 'DOWN' | 'NOISE' | null
  status: 'PENDING' | 'SETTLED' | 'EXPIRED'
  email_sent: boolean
  created_at: string | null
}
interface FakeBreakoutStatus {
  running: boolean
  enabled: boolean
  levels: Record<string, { resistance: number; support: number }>
  daily_count: number
  eps: number
  btc_mid: number
  pm_15m: { down_price: number | null; up_price: number | null; end_date: number | null; updated_ts: number | null }
  pm_5m_down_price: number | null
}
interface FakeBreakoutGroup {
  level: string
  side: string
  settled_15m: number
  wins_15m: number
  settled_5m: number
  wins_5m: number
  win_rate_15m: number | null
  win_rate_5m: number | null
}
interface FakeBreakoutStats {
  total_signals: number
  settled: number
  down_win_rate: number | null
  avg_down_price_15m: number | null
  avg_down_price_5m: number | null
  settled_5m: number
  down_win_rate_5m: number | null
  by_group: FakeBreakoutGroup[]
  by_pattern_type: FakeBreakoutPatternStats[]
  research_win_rates: Record<string, number>
}
// 按场景类型（pattern_type）的实盘统计：后端 compute_pattern_stats 输出
interface FakeBreakoutPatternStats {
  pattern_type: string
  n: number
  wins: number
  winrate: number | null
  cumulative_ev: number | null
  avg_ev_at_entry: number | null
  equity_curve: number[]
  peak_equity: number
  max_drawdown: number
  n_last_7d: number
}

// 信号分析面板：与后端 /api/signals/analytics + /api/chart/btc-klines 对齐
interface BtcKline {
  open_time: number; open: number; high: number; low: number; close: number; volume: number
}
interface AnalyticsCurvePoint { i: number; ts: number; cum_wr: number; cum_ev: number }
interface ShadowVersionBlock {
  summary: {
    n: number; win_rate: number | null; avg_ev: number | null; cum_ev: number | null
    avg_breakeven: number | null; bench_winrate: number | null; bench_ev: number | null
    desc: string
  }
  curve: AnalyticsCurvePoint[]
}
interface SceneTypeBlock {
  summary: {
    n: number; winrate: number | null; avg_ev: number | null; cum_ev: number | null
    bench_winrate: number | null
  }
  curve: AnalyticsCurvePoint[]
}
interface SignalsAnalytics {
  pump_ts: number
  shadow: Record<string, ShadowVersionBlock>
  scene: Record<string, SceneTypeBlock>
  regime: {
    phases: Record<string, { n: number; wins: number; winrate: number | null }>
    by_version: Record<string, Record<string, { n: number; wins: number; winrate: number | null }>>
    daily: { date: string; n: number; wins: number; winrate: number | null }[]
  }
}

// 模式池分级与回测快照：与后端 /api/agent/patterns/compare + /backtest-runs 对齐
interface PatternBacktestRun {
  id: number
  pattern_id: number
  data_start: number
  data_end: number
  sample_count: number
  correct_count: number
  win_rate: number
  wilson_lower: number | null
  wilson_upper: number | null
  ev_after_fee: number | null
  segment_stats: Record<string, { n: number; k: number; win_rate: number }> | null
  delta_vs_prev: {
    tier_change?: { from: string; to: string } | null
    decay_warning?: boolean
    latest_seg_win_rate?: number | null
    latest_seg_n?: number
    prev_run_id?: number
    prev_win_rate?: number
    win_rate_drift?: number
    new_samples?: number
    new_samples_win_rate?: number | null
    suggestions?: string[]
    note?: string
  } | null
  trigger_reason: string
  created_at: string | null
}
interface PatternCompareItem {
  pattern_id: number
  pattern_name: string
  status: string
  tier: 'S' | 'A' | 'B' | 'C'
  predicted_direction: string
  discovery_method: string
  live_win_rate: number
  live_sample_count: number
  latest_run: {
    id: number
    data_end: number
    sample_count: number
    win_rate: number
    wilson_lower: number | null
    wilson_upper: number | null
    ev_after_fee: number | null
    delta_vs_prev: PatternBacktestRun['delta_vs_prev']
    created_at: string | null
  } | null
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

// ============================================================
// 登录态与请求封装（单一访问密码，存 localStorage 不过期）
// ============================================================

const AUTH_KEY = 'bp_auth_token'

const getAuthToken = (): string | null => localStorage.getItem(AUTH_KEY)

const setAuthToken = (token: string) => localStorage.setItem(AUTH_KEY, token)

const clearAuthToken = () => localStorage.removeItem(AUTH_KEY)

// 任一请求收到 401：清除本地登录态并派发事件，App 监听后回到登录页
type ApiInit = RequestInit & { headers?: Record<string, string> }

function authFetch(url: string, init: ApiInit = {}): Promise<Response> {
  const token = getAuthToken()
  const headers: Record<string, string> = { ...(init.headers || {}) }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return fetch(url, { ...init, headers }).then(resp => {
    if (resp.status === 401) {
      clearAuthToken()
      window.dispatchEvent(new Event('auth-logout'))
    }
    return resp
  })
}

const api = {
  health: () => authFetch('/api/health').then(r => r.json()),
  getPredictionMarket: () => authFetch('/api/chart/prediction-market').then(r => r.json()),
  getPredictionMarket15m: () => authFetch('/api/chart/prediction-market/15m').then(r => r.json()),
  runMomentumPredict: () => authFetch('/api/sentiment/momentum-predict', { method: 'POST' }).then(r => r.json()),
  getAgentStatus: () => authFetch('/api/sentiment/agent/status').then(r => r.json()),
  getAgentPatterns: () => authFetch('/api/sentiment/agent/patterns').then(r => r.json()),
  getAgentPredictions: (direction?: string) =>
    authFetch('/api/sentiment/agent/predictions' + (direction ? `?direction=${direction}` : '')).then(r => r.json()),
  getPatternHistory: (id: number) =>
    authFetch(`/api/sentiment/agent/patterns/${id}/history`).then(r => r.json()),
  getLLMTraces: (phase?: string) =>
    authFetch('/api/llm/traces' + (phase ? `?phase=${phase}` : '')).then(r => r.json()),
  getLLMTraceDetail: (id: number) =>
    authFetch(`/api/llm/traces/${id}`).then(r => r.json()),
  triggerDeepLearn: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  runPyClusterDeepLearn: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn/pycluster?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  runCompare: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn/compare?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  getCompareLive: () =>
    authFetch('/api/sentiment/agent/deep-learn/compare/live').then(r => r.json()),
  getAgentHealth: () => authFetch('/api/agent/health').then(r => r.json()),
  getAgentEvolution: (days = 30) =>
    authFetch(`/api/sentiment/agent/evolution?days=${days}`).then(r => r.json()),
  getFakeBreakoutStatus: () => authFetch('/api/fake-breakout/status').then(r => r.json()),
  getFakeBreakoutSignals: (limit = 50) =>
    authFetch(`/api/fake-breakout/signals?limit=${limit}`).then(r => r.json()),
  getFakeBreakoutSignalPath: (signalId: number) =>
    authFetch(`/api/fake-breakout/signals/${signalId}/path`).then(r => r.json()),
  getFakeBreakoutStats: () => authFetch('/api/fake-breakout/stats').then(r => r.json()),
  getBtcKlines: (interval: string, limit: number) =>
    authFetch(`/api/chart/btc-klines?interval=${interval}&limit=${limit}`).then(r => r.json()),
  getSignalsAnalytics: () => authFetch('/api/signals/analytics').then(r => r.json()),
  getPatternCompare: () => authFetch('/api/agent/patterns/compare').then(r => r.json()),
  getPatternBacktestRuns: (patternId: number, limit = 30) =>
    authFetch(`/api/agent/patterns/backtest-runs?pattern_id=${patternId}&limit=${limit}`).then(r => r.json()),
  triggerReevaluate: () =>
    authFetch('/api/agent/patterns/reevaluate', { method: 'POST' }).then(r => r.json()),
  commitDeepLearn: (discoveries: DeepLearnDiscovery[], snapshotToken?: string | null) =>
    authFetch('/api/sentiment/agent/deep-learn/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ discoveries, snapshot_token: snapshotToken ?? null }),
    }).then(r => r.json()),
  // 实盘面板（2026-08-22）：钱包/实盘状态/下单/订单历史
  getPredictionWallet: () => authFetch('/api/prediction-wallet').then(r => r.json()),
  getLiveStatus: () => authFetch('/api/misalignment/signals').then(r => r.json()),
  getRecentTrades: (limit = 20) => authFetch(`/api/trades/recent?limit=${limit}`).then(r => r.json()),
  postTradeTest: (amount_usdt: number, prediction: string) =>
    authFetch('/api/trade/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount_usdt, prediction }),
    }).then(r => r.json()),
  postTransferIn: (amount_usdt: number) =>
    authFetch('/api/prediction/transfer-in', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount_usdt }),
    }).then(r => r.json()),
  postTransferOut: (amount_usdt: number) =>
    authFetch('/api/prediction/transfer-out', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount_usdt }),
    }).then(r => r.json()),
  postLiveChannel: (channel: string, enabled: boolean, amountUsdt?: number, maxDailyOrders?: number) =>
    authFetch('/api/live/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel, enabled,
        ...(amountUsdt != null ? { amount_usdt: amountUsdt } : {}),
        ...(maxDailyOrders != null ? { max_daily_orders: maxDailyOrders } : {}),
      }),
    }).then(r => r.json()),
  getQuotePreview: () => authFetch('/api/prediction/quote-preview').then(r => r.json()),
  postSyncBinance: () => authFetch('/api/trades/sync-binance', { method: 'POST' }).then(r => r.json()),
  // 奖金领取（2026-08-23）：可领查询 + batch-redeem
  getRedeemable: () => authFetch('/api/prediction/redeemable').then(r => r.json()),
  postRedeem: () =>
    authFetch('/api/prediction/redeem', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    }).then(r => r.json()),
  // 线上信号概览：各影子版本累计统计（stats 按 version 服务端算）
  getMisalignmentSignals: (version: string) =>
    authFetch(`/api/misalignment/signals?limit=1&version=${version}`).then(r => r.json()),
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

// 问号 hover 提示（信号说明等）：纯 CSS group-hover，无依赖
function HelpHint({ text }: { text: string }) {
  return (
    <span className="relative group inline-flex items-center cursor-help align-middle" tabIndex={0}>
      <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-gray-400 text-gray-500 text-[9px] font-bold leading-none select-none">?</span>
      <span className="absolute right-0 bottom-full mb-1.5 hidden group-hover:block group-focus-within:block z-30 w-72 rounded-lg bg-gray-800 text-gray-100 text-[11px] leading-relaxed px-3 py-2 text-left whitespace-normal shadow-xl">{text}</span>
    </span>
  )
}

// 线上信号通道说明（口径源：services/live_channels.py 注册表 + quote_edge_detector 冻结规则）
// 2026-08-24 多通道实盘改造：全部通道支持实盘下单（liveOk），各自独立金额/日限/护栏，
// 通道运行状态（开关/金额/日限/护栏/今日成交）来自后端 multi_live_trader.status_async()。
const SIGNAL_INFO: Record<string, { name: string; kind: '实盘' | '影子' | '场景'; desc: string; liveOk?: boolean }> = {
  quote_contrarian_v1: {
    name: '报价反向（B格逆势）', kind: '实盘', liveOk: true,
    desc: '5 分钟窗口开始后 45~60 秒内，DOWN token 报价首次跌入 [0.15, 0.25)（明显便宜）时买入 DOWN。低胜率高赔付：回测胜率 24%、EV +0.155（赢一次约赚 4 倍）。通道护栏 0.28（区间上界+0.03），每窗至多一单。',
  },
  quote_momentum_v1: {
    name: '报价动量（A格顺势）', kind: '影子', liveOk: true,
    desc: '5 分钟窗口 90~120 秒内，DOWN token 报价首次进入 [0.69, 0.75)（强势确认）时押 DOWN。回测胜率 79.9%、EV +0.097。通道护栏 0.78，每窗至多一单。',
  },
  quote_contrarian_v2: {
    name: '报价反向·门禁版', kind: '影子', liveOk: true,
    desc: 'v1 区间 + BTC 价格门禁：触发时点 BTC 未高于窗口开盘 ≥0.10%（只接「假冲高」，归因显示平盘窗贡献 86% 利润）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.28。',
  },
  quote_momentum_v2: {
    name: '报价动量·门禁版', kind: '影子', liveOk: true,
    desc: 'v1 区间 + BTC 价格门禁：触发时点 BTC 已低于窗口开盘 ≥0.10%（剔「假恐慌」，真跌段胜率 85% vs 假恐慌段 40%）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.78。',
  },
  quote_contrarian_v3a: {
    name: '报价反向·交替环境版', kind: '影子', liveOk: true,
    desc: 'contrarian v1 区间 + v2 价格门禁 + 环境门禁：前窗结算 DOWN（交替环境：前窗跌+本窗涨=V 反弹假冲高）。真实回测 n=85 胜率 31.8%、EV +0.528。实盘已解锁（前窗 outcome 异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  quote_contrarian_v3b: {
    name: '报价反向·日高回落版', kind: '影子', liveOk: true,
    desc: 'v3a + 触发时点 BTC 距当日高点回落 ≥0.30%（含边界，震荡日冲高更易衰竭）。真实回测 n=65 胜率 33.8%、EV +0.646（单笔 EV 最优）。实盘已解锁（日高异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  x4_v1: {
    name: '情绪错位（收阳押次窗DOWN）', kind: '影子', liveOk: true,
    desc: '本窗收阳但 15m 市场收尾情绪 ≤40 的错位 → 次窗 +150s 决策点押 DOWN（回测合并胜率 63.5%、EV +0.254）。实盘已解锁：PENDING 信号轮询→决策点下单，护栏 0.45，错过决策点不追单。',
  },
  x4_v2: {
    name: '情绪错位·平静市门禁版', kind: '影子', liveOk: true,
    desc: 'x4_v1 + 平静市门禁（回测胜率 45.3%，仅平静市况触发）。实盘已解锁：同 x4_v1 决策点机制，护栏 0.50，错过决策点不追单。',
  },
  scene_bull_exhaust: {
    name: '场景S1 多头耗尽（押DOWN）', kind: '场景', liveOk: true,
    desc: '15m 周期刺破 4h 阻力 + 光头阳收盘确认 → 次周期开盘押 DOWN（真 OOS 胜率 64.4%，盈亏平衡 0.63）。实盘已解锁：15m 市场次周期开盘下单，护栏 0.60。',
  },
  scene_bull_exhaust_confirm: {
    name: '场景S5 确认入场（押DOWN）', kind: '场景', liveOk: true,
    desc: 'S1 信号 +5min 确认（次周期第 1 根 5m K 收盘 < 开盘）才买 DOWN（确认组胜率 78.5%，盈亏平衡 0.77）。实盘已解锁：确认时刻 15m 市场下单，护栏 0.75。',
  },
  scene_bear_exhaust: {
    name: '场景S2 空头耗尽（押UP）', kind: '场景', liveOk: true,
    desc: '15m 周期跌破 4h 支撑 + 收阴 + 放量 → 次周期开盘押 UP（胜率 53.6%，盈亏平衡 0.525）。护栏 0.55：跌态 UP 报价常在 0.79+，超护栏保护性弃单（负 EV 保护，属正确行为）。',
  },
  scene_momentum_fade: {
    name: '场景S4 动量衰竭（押DOWN）', kind: '场景', liveOk: true,
    desc: '连阳 ≥3 根 + 光头阳的动量衰竭 → 次周期开盘押 DOWN（胜率 55.4%，盈亏平衡 0.54）。实盘已解锁：15m 市场次周期开盘下单，护栏 0.55。',
  },
}

const SIGNAL_KIND_BADGE: Record<string, string> = {
  '实盘': 'bg-green-100 text-green-700 border-green-300',
  '影子': 'bg-purple-100 text-purple-700 border-purple-300',
  '场景': 'bg-blue-100 text-blue-700 border-blue-300',
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
// 实盘 Tab（账户状态 + 人工测试单 + 订单历史，2026-08-22）
// ============================================================

// 实盘对照图：BTC K 线（5m/15m）× 预测市场情绪曲线（UP/DOWN 报价，15s 采样）
function LiveChartCard() {
  const [p, setP] = useState<'5m' | '15m'>('5m')
  const [klines, setKlines] = useState<BtcKline[]>([])
  const [points, setPoints] = useState<PMPoint[]>([])

  const load = useCallback(() => {
    api.getBtcKlines(p, 96).then(k => {
      if (k && Array.isArray(k.klines)) setKlines(k.klines as BtcKline[])
    }).catch(() => {})
    ;(p === '5m' ? api.getPredictionMarket() : api.getPredictionMarket15m())
      .then(d => {
        const pts = (d as { points?: PMPoint[] })?.points
        setPoints(Array.isArray(pts) ? pts : [])
      }).catch(() => {})
  }, [p])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    const t = setInterval(load, 30_000)
    return () => clearInterval(t)
  }, [load])

  const hhmm = (t: number) =>
    new Date(t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

  return (
    <div className="lg:col-span-2">
      <Card title={`BTC K 线 × 市场情绪对照（${p}，30s 刷新）`}>
        <div className="flex items-center gap-3 mb-2 text-xs flex-wrap">
          {(['5m', '15m'] as const).map(iv => (
            <button key={iv} onClick={() => setP(iv)}
              className={`px-2.5 py-0.5 rounded-md border transition ${
                p === iv ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
              }`}>{iv}</button>
          ))}
          <span className="text-gray-400">
            <span className="text-green-600">— UP 报价</span> · <span className="text-red-500">— DOWN 报价</span> · <span className="text-gray-600">— BTC 收盘</span>（情绪 = 预测市场报价，15s 采样）
          </span>
        </div>
        <div className="space-y-3">
          {klines.length > 0 ? (
            <ResponsiveContainer width="100%" height={170}>
              <LineChart data={klines.map(k => ({ t: k.open_time, close: k.close }))}
                margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                  tickFormatter={hhmm} tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} stroke="#9ca3af" width={64}
                  tickFormatter={(v: number) => v.toLocaleString()} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  labelFormatter={t => new Date(t as number).toLocaleString('zh-CN')}
                  formatter={v => [typeof v === 'number' ? v.toLocaleString() : '--', 'BTC 收盘']} />
                <Line dataKey="close" stroke="#374151" dot={false} strokeWidth={1.6} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-xs text-gray-400 h-10 flex items-center">K 线加载中…（{p}）</div>
          )}
          {points.length > 1 ? (
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={points.map(pt => ({ t: pt.timestamp, up: pt.up_price, down: pt.down_price }))}
                margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                  tickFormatter={hhmm} tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} stroke="#9ca3af" width={64}
                  tickFormatter={(v: number) => v.toFixed(2)} />
                <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  labelFormatter={t => new Date(t as number).toLocaleString('zh-CN')}
                  formatter={(v, name) => [typeof v === 'number' ? v.toFixed(3) : '--', name === 'up' ? 'UP' : 'DOWN']} />
                <Line dataKey="up" stroke="#16a34a" dot={false} strokeWidth={1.6} connectNulls isAnimationActive={false} />
                <Line dataKey="down" stroke="#ef4444" dot={false} strokeWidth={1.6} connectNulls isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-xs text-gray-400 h-10 flex items-center">情绪曲线加载中…（{p}）</div>
          )}
        </div>
      </Card>
    </div>
  )
}

// 多通道实盘状态类型（与后端 multi_live_trader.status_async() 严格对齐）
interface LiveChannelStatus {
  channel: string
  display_name: string
  family: string
  market_period: string
  direction: string
  enabled: boolean
  enabled_at_startup: boolean
  amount_usdt: number
  max_daily_orders: number
  max_exec_price: number
  auto_max_exec: number
  fire_total: number
  fired_windows: number[]
  filled_today?: number
}

// 线上信号概览卡：12 通道一屏总览（60s 轮询，统计为全量累计不随 limit 截断）
// onToggleChannel：行内通道开关回调（由 LiveTradeTab 注入，confirm 统一在那里，
// 避免两处开关状态不一致互咬）
function SignalsOverviewCard({ live, onToggleChannel, busy = false }: {
  live: Record<string, unknown> | null
  onToggleChannel?: (ch: LiveChannelStatus) => void
  busy?: boolean
}) {
  const [stats, setStats] = useState<Record<string, Record<string, unknown> | null>>({})
  const [fb, setFb] = useState<Record<string, unknown> | null>(null)

  const refresh = useCallback(() => {
    // quote_edge/x4 八通道：misalignment 影子统计（版本名与通道名一致）
    const versions = ['quote_contrarian_v1', 'quote_momentum_v1', 'quote_contrarian_v2', 'quote_momentum_v2', 'quote_contrarian_v3a', 'quote_contrarian_v3b', 'x4_v1', 'x4_v2']
    versions.forEach(v => {
      api.getMisalignmentSignals(v)
        .then(d => setStats(prev => ({ ...prev, [v]: d?.stats ?? null })))
        .catch(() => {})
    })
    // scene 四通道：fake_breakout 全局统计（无按 pattern_type 细分的端点，作行内参考）
    api.getFakeBreakoutStats().then(setFb).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 60000)
    return () => clearInterval(t)
  }, [refresh])

  const channels = Array.isArray(live?.channels) ? live.channels as LiveChannelStatus[] : []

  const fbTotal = fb?.total_signals
  const fbWr = fb?.down_win_rate
  const fbStatText = fbTotal != null
    ? `${String(fbTotal)} 信号 · DOWN 胜率 ${typeof fbWr === 'number' ? (fbWr * 100).toFixed(0) : '?'}%`
    : '--'

  return (
    <Card title="线上信号概览">
      <div className="text-xs">
        {channels.map(ch => {
          const info = SIGNAL_INFO[ch.channel]
          const s = stats[ch.channel]
          let statText: string
          if (ch.family === 'scene') {
            statText = `今日 ${String(ch.filled_today ?? 0)} 单 · 开火 ${String(ch.fire_total)}（场景全局：${fbStatText}）`
          } else if (s != null) {
            const n = s.settled as number | undefined
            const wr = s.win_rate as number | null | undefined
            const ev = s.avg_ev as number | null | undefined
            statText = `${String(n ?? 0)} 注 · 胜率 ${wr != null ? (wr * 100).toFixed(0) : '?'}% · EV ${ev != null ? `${ev >= 0 ? '+' : ''}${ev.toFixed(3)}` : '?'}（影子口径）`
          } else {
            statText = '--'
          }
          return (
            <div key={ch.channel} className="flex items-center justify-between gap-2 py-1.5 border-b border-gray-50 last:border-0">
              <span className="flex items-center gap-1.5 min-w-0">
                <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded border shrink-0 ${SIGNAL_KIND_BADGE[info?.kind ?? '影子']}`}>{info?.kind ?? '影子'}</span>
                <span className="text-gray-700 font-medium truncate">{info?.name ?? ch.display_name}</span>
                <span className="text-[10px] text-gray-400 font-mono shrink-0 hidden sm:inline">{ch.channel}</span>
                <HelpHint text={info?.desc ?? ''} />
                {ch.enabled && (
                  <span className="text-[10px] font-semibold shrink-0 text-green-700">
                    · 开火中 {String(ch.amount_usdt)}U/单
                  </span>
                )}
              </span>
              <span className="flex items-center gap-1.5 shrink-0">
                <span className="font-mono text-[11px] text-gray-600 text-right">{statText}</span>
                {onToggleChannel && (
                  <button
                    onClick={() => onToggleChannel(ch)}
                    disabled={busy}
                    className={`px-2 py-0.5 text-[10px] font-semibold rounded text-white disabled:opacity-50 shrink-0 ${ch.enabled ? 'bg-red-500 hover:bg-red-600' : 'bg-green-600 hover:bg-green-700'}`}
                    title={ch.enabled ? '关闭该通道（在途任务不受影响）' : '开启该通道（confirm 后生效，独立金额/护栏/日限）'}
                  >{ch.enabled ? '停火' : '开火'}</button>
                )}
              </span>
            </div>
          )
        })}
        {channels.length === 0 && (
          <div className="text-gray-400 py-2">实盘通道状态不可用（执行器未装配？详见后端日志）</div>
        )}
        <p className="text-[10px] text-gray-400 mt-1.5">
          12 通道全部支持实盘：每通道独立金额/日限/执行价护栏（通道管理与金额热调见「实盘交易」页）。影子统计 60s 刷新。
        </p>
      </div>
    </Card>
  )
}

function LiveTradeTab() {
  const [wallet, setWallet] = useState<Record<string, unknown> | null>(null)
  const [live, setLive] = useState<Record<string, unknown> | null>(null)
  const [orders, setOrders] = useState<Record<string, unknown>[]>([])
  const [amount, setAmount] = useState('1')
  const [side, setSide] = useState<'DOWN' | 'UP'>('DOWN')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [transferAmt, setTransferAmt] = useState('5')
  const [transferring, setTransferring] = useState(false)
  const [transferResult, setTransferResult] = useState<Record<string, unknown> | null>(null)
  // 报价预览（15s 轮询 + 本地 1s 倒计时）：服务端时钟偏移修正本地计时
  const [quote, setQuote] = useState<Record<string, unknown> | null>(null)
  const [clockOffset, setClockOffset] = useState(0)
  const [nowMs, setNowMs] = useState(() => Date.now())
  const [syncing, setSyncing] = useState(false)
  const [syncResult, setSyncResult] = useState<Record<string, unknown> | null>(null)
  const [togglingLive, setTogglingLive] = useState(false)
  // 通道金额热调草稿（key=channel；仅在用户输入时存在，提交后清除）
  const [amountDrafts, setAmountDrafts] = useState<Record<string, string>>({})
  // 可领取奖金（赢单 token → batch-redeem → USDT）
  const [redeemable, setRedeemable] = useState<Record<string, unknown> | null>(null)
  const [redeeming, setRedeeming] = useState(false)
  const [redeemResult, setRedeemResult] = useState<Record<string, unknown> | null>(null)

  const refresh = useCallback(() => {
    api.getPredictionWallet().then(setWallet).catch(() => {})
    api.getLiveStatus().then(d => setLive(d?.live_channels ?? null)).catch(() => {})
    api.getRecentTrades().then(d => setOrders(d?.orders ?? [])).catch(() => {})
    api.getRedeemable().then(setRedeemable).catch(() => {})
    api.getQuotePreview().then((q: Record<string, unknown>) => {
      setQuote(q)
      if (typeof q?.server_now_ms === 'number') {
        setClockOffset((q.server_now_ms as number) - Date.now())
      }
    }).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 15000)
    return () => clearInterval(timer)
  }, [refresh])

  // 本地 1s tick：用 window_end - server_now_ms 偏移算剩余秒（不新增网络轮询）
  useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const handleTestTrade = async () => {
    const amt = parseFloat(amount)
    if (!Number.isFinite(amt) || amt < 0.1 || amt > 50) {
      alert('金额仅允许 0.1~50 USDT（与实盘单笔硬上限一致）')
      return
    }
    if (!window.confirm(
      `确认下真实订单测试单？\n方向: ${side === 'DOWN' ? '↓ 看跌' : '↑ 看涨'} | 金额: ${amt} USDT\n\n这是真实订单，将从现货账户扣款。`)) return
    setBusy(true)
    setResult(null)
    try {
      const res = await api.postTradeTest(amt, side)
      setResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const walletErr = (wallet as { error?: string } | null)?.error
  const regTs = wallet?.registered_time as number | undefined
  const resultFilled = result?.status === 'FILLED'
  // 多通道实盘：live = live_channels status（channels[] 见 LiveChannelStatus）
  const liveChannels = Array.isArray(live?.channels) ? live.channels as LiveChannelStatus[] : []
  const enabledCount = liveChannels.filter(c => c.enabled).length
  const liveDefaults = (live?.defaults ?? {}) as Record<string, unknown>

  // 倒计时：服务端时钟修正后的剩余毫秒；<60s 红色警示
  const windowEnd = quote?.window_end as number | null | undefined
  const remainSec = (typeof windowEnd === 'number' && quote && !quote.stale)
    ? Math.max(0, Math.floor((windowEnd - (nowMs + clockOffset)) / 1000))
    : null
  const urgent = remainSec != null && remainSec < 60

  // 结算统计（订单表尾累计行）+ 在途持仓（FILLED 未结算）
  const settledOrders = orders.filter(o => o.settled_at != null)
  const settledCount = settledOrders.length
  const totalPnl = settledOrders.reduce(
    (s, o) => s + (typeof o.pnl === 'number' ? (o.pnl as number) : 0), 0)
  const openPositions = orders.filter(o => o.status === 'FILLED' && !o.settled_at)
  const openAmount = openPositions.reduce(
    (s, o) => s + (o.amount_in != null ? Number(o.amount_in) / 1e18 : 0), 0)
  // 钱包支付账户/持仓（官方 payment-options：items[].accountType，2026-08-23 生产实测收敛；
  // null=不可查；旧探索形态 asset/symbol 兑底保留）
  const walletAssets = Array.isArray(wallet?.wallet_assets)
    ? (wallet.wallet_assets as Array<Record<string, unknown>>)
    : null
  const payAccounts = (walletAssets ?? []).filter(a =>
    typeof a?.accountType === 'string' && (a?.accountType as string).length > 0)
  const heldTokens = (walletAssets ?? []).filter(a => {
    if (typeof a?.accountType === 'string' && (a?.accountType as string).length > 0) return false  // 账户形态走 payAccounts
    const sym = String(a?.asset ?? a?.symbol ?? '')
    if (sym.toUpperCase() === 'USDT') return false  // USDT 已在上方余额行展示
    const amt = Number(a?.free ?? a?.balance ?? 0)
    return Number.isFinite(amt) && amt > 0
  })
  const PAY_ACCOUNT_LABEL: Record<string, string> = {
    CeDeFi: '预测钱包', SPOT: '现货', FUNDING: '资金账户',
  }

  const handleSyncBinance = async () => {
    if (!window.confirm('确认用币安侧订单历史对账本地 PENDING 订单？\n（本地卡 PENDING 但币安已成交的行会被订正为终态）')) return
    setSyncing(true)
    setSyncResult(null)
    try {
      const res = await api.postSyncBinance()
      setSyncResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setSyncing(false)
    }
  }

  const handleTransferIn = async () => {
    const amt = parseFloat(transferAmt)
    if (!Number.isFinite(amt) || amt < 0.1 || amt > 20) {
      alert('划转金额仅允许 0.1~20 USDT')
      return
    }
    if (!window.confirm(`确认从现货账户划转 ${amt} USDT 到预测钱包？\n（下单扣的是预测钱包内余额）`)) return
    setTransferring(true)
    setTransferResult(null)
    try {
      const res = await api.postTransferIn(amt)
      setTransferResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setTransferring(false)
    }
  }

  const handleTransferOut = async () => {
    const amt = parseFloat(transferAmt)
    if (!Number.isFinite(amt) || amt < 0.1 || amt > 20) {
      alert('划出金额仅允许 0.1~20 USDT')
      return
    }
    if (!window.confirm(`确认从预测钱包划出 ${amt} USDT 回现货账户？\n（首次使用建议 0.1 金丝雀验证：成功后现货余额应增加）`)) return
    setTransferring(true)
    setTransferResult(null)
    try {
      const res = await api.postTransferOut(amt)
      setTransferResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setTransferring(false)
    }
  }

  const handleChannelToggle = async (ch: LiveChannelStatus) => {
    // 通道级开关：confirm 文案带该通道金额/护栏/日限（多通道时代重写，取代版本热切）
    const info = SIGNAL_INFO[ch.channel]
    const name = info?.name ?? ch.display_name
    const next = !ch.enabled
    if (next) {
      if (!window.confirm(
        `确认开启通道实盘？\n通道: ${name}（${ch.channel}）\n每单 ${String(ch.amount_usdt)} USDT | 执行价护栏 ${String(ch.max_exec_price)} | 日限 ${String(ch.max_daily_orders)} 单\n\n命中信号将下真实订单（真金白银）；重启后回落 LIVE_CHANNELS_JSON 配置。`)) return
    } else {
      if (!window.confirm(
        `确认关闭通道实盘？\n通道: ${name}（${ch.channel}）\n（不取消在途任务，只阻止该通道新单派生）`)) return
    }
    setTogglingLive(true)
    try {
      const res = await api.postLiveChannel(ch.channel, next)
      if (res?.error) alert(`切换失败: ${String(res.error)}`)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setTogglingLive(false)
    }
  }

  const handleChannelAmount = async (ch: LiveChannelStatus) => {
    // 金额热调：保存草稿值（校验同后端便硬限 0.1~50），不动开关状态
    const amt = parseFloat(amountDrafts[ch.channel] ?? '')
    if (!Number.isFinite(amt) || amt < 0.1 || amt > 50) {
      alert('金额仅允许 0.1~50 USDT（与实盘单笔硬上限一致）')
      return
    }
    if (amt === ch.amount_usdt) return
    setTogglingLive(true)
    try {
      const res = await api.postLiveChannel(ch.channel, ch.enabled, amt)
      if (res?.error) {
        alert(`金额保存失败: ${String(res.error)}`)
      } else {
        setAmountDrafts(prev => {
          const rest = { ...prev }
          delete rest[ch.channel]
          return rest
        })
      }
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setTogglingLive(false)
    }
  }

  const handleRedeem = async () => {
    const n = Number(redeemable?.claimable_count ?? 0)
    if (n <= 0) return
    if (!window.confirm(`确认领取 ${n} 个获胜 token 的奖金？\n（batch-redeem 赎回后入预测钱包 USDT 余额）`)) return
    setRedeeming(true)
    setRedeemResult(null)
    try {
      const res = await api.postRedeem()
      setRedeemResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setRedeeming(false)
    }
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <LiveChartCard />
      <Card title="人工测试单（真实下单）">
        <div className="space-y-3 text-sm">
          {quote == null || quote.stale ? (
            <div className="text-xs text-gray-400 rounded bg-gray-50 px-2 py-1.5">报价不可用（等待 15s 采样器…）</div>
          ) : (
            <div className={`flex items-center justify-between rounded px-2 py-1.5 text-xs ${urgent ? 'bg-red-50 text-red-700' : 'bg-blue-50 text-blue-700'}`}>
              <span className="font-mono">
                UP {typeof quote.up_price === 'number' ? quote.up_price.toFixed(3) : '--'} / DOWN {typeof quote.down_price === 'number' ? quote.down_price.toFixed(3) : '--'}
                <span className="opacity-60">（指示价）</span>
              </span>
              <span className="font-mono font-bold tabular-nums">
                {remainSec != null ? `剩余 ${Math.floor(remainSec / 60)}:${String(remainSec % 60).padStart(2, '0')}` : '--:--'}
              </span>
            </div>
          )}
          <div className="flex items-center gap-3">
            <label className="text-gray-500 shrink-0">金额 (USDT)</label>
            <input
              type="number" min={0.1} max={50} step={0.5} value={amount}
              onChange={e => setAmount(e.target.value)}
              className="w-24 px-2 py-1 border border-gray-300 rounded text-gray-800"
            />
            <span className="text-xs text-gray-400">0.1~50</span>
          </div>
          {/* 金额预设：百分比按预测钱包余额计算（clamp 0.1~50），固定额直填（100U 超硬上限禁用） */}
          <div className="flex items-center gap-1.5 flex-wrap text-xs">
            <span className="text-gray-400 shrink-0">按余额</span>
            {[2, 5, 10, 20].map(pct => {
              const bal = typeof wallet?.prediction_usdt_free === 'number'
                ? wallet.prediction_usdt_free as number : null
              const disabled = bal == null
              return (
                <button key={pct} disabled={disabled}
                  title={disabled ? '预测钱包余额不可查（等余额端点收敛）' : `${pct}% × ${bal!.toFixed(2)}U`}
                  onClick={() => setAmount(String(Math.min(50, Math.max(0.1, +(bal! * pct / 100).toFixed(2)))))}
                  className="px-2 py-0.5 rounded border border-gray-300 bg-white text-gray-700 hover:border-blue-400 disabled:opacity-40"
                >{pct}%</button>
              )
            })}
            <span className="text-gray-400 shrink-0 ml-2">固定</span>
            {[1, 2, 5, 10, 20, 50, 100].map(u => (
              <button key={u} disabled={u > 50} title={u > 50 ? '超单笔硬上限 50 USDT' : `${u} USDT`}
                onClick={() => setAmount(String(u))}
                className="px-2 py-0.5 rounded border border-gray-300 bg-white text-gray-700 hover:border-blue-400 disabled:opacity-40"
              >{u}U</button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-gray-500 shrink-0">方向</span>
            <button
              onClick={() => setSide('DOWN')}
              className={`px-3 py-1 text-sm font-semibold rounded-full border ${side === 'DOWN' ? 'bg-red-100 text-red-700 border-red-300' : 'bg-white text-gray-500 border-gray-200'}`}
            >↓ 看跌</button>
            <button
              onClick={() => setSide('UP')}
              className={`px-3 py-1 text-sm font-semibold rounded-full border ${side === 'UP' ? 'bg-green-100 text-green-700 border-green-300' : 'bg-white text-gray-500 border-gray-200'}`}
            >↑ 看涨</button>
          </div>
          <button
            onClick={handleTestTrade}
            disabled={busy}
            className={`w-full py-2 rounded-lg font-bold text-white transition ${busy ? 'bg-gray-300 cursor-wait' : 'bg-brand hover:opacity-90'}`}
          >
            {busy ? '下单中…' : '下单（真实订单）'}
          </button>
          {result && (
            <div className={`p-2 rounded text-xs ${resultFilled ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-700'}`}>
              <div className="font-bold">{resultFilled ? '✓ 已成交 FILLED' : `✗ ${String(result.status ?? '未执行')}`}</div>
              {result.order_id != null && <div>订单号: {String(result.order_id)}</div>}
              {result.direction != null && <div>方向: {String(result.direction)}</div>}
              {result.average_price != null && <div>成交均价: {String(result.average_price)}</div>}
              {result.error_message != null && <div>{String(result.error_message)}</div>}
              {result.error != null && <div>{String(result.error)}</div>}
            </div>
          )}
          <p className="text-xs text-gray-400">
            与信号实盘同链路（占位→报价→下单→落库）；同一 5m 窗口至多一单。
          </p>
        </div>
      </Card>

      <SignalsOverviewCard live={live} onToggleChannel={handleChannelToggle} busy={togglingLive} />

      <div className="lg:col-span-2">
      <Card title="账户状态">
        <div className="space-y-2 text-sm">
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">预测钱包</span>
            {walletErr
              ? <span className="text-red-600 text-right">{walletErr}</span>
              : <span className="font-mono text-gray-800">{String(wallet?.wallet_address ?? '--')}</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">钱包 ID</span>
            <span className="font-mono text-gray-600 text-xs truncate">{String(wallet?.wallet_id ?? '--')}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">钱包注册时间</span>
            <span className="text-gray-700">{regTs ? new Date(regTs).toLocaleString() : '--'}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">预测钱包 USDT</span>
            {typeof wallet?.prediction_usdt_free === 'number'
              ? <span className={`font-mono font-bold ${(wallet.prediction_usdt_free as number) >= 1 ? 'text-green-700' : 'text-red-600'}`}>{(wallet.prediction_usdt_free as number).toFixed(4)}</span>
              : <span className="text-gray-400">暂不可查（API 字段待确认）</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">现货 USDT 可用</span>
            {typeof wallet?.spot_usdt_free === 'number'
              ? <span className={`font-mono font-bold ${(wallet.spot_usdt_free as number) >= 1 ? 'text-green-700' : 'text-red-600'}`}>{(wallet.spot_usdt_free as number).toFixed(4)}</span>
              : <span className="text-gray-400">查询失败</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">在途持仓（未结算）</span>
            {openPositions.length > 0
              ? <span className="font-mono text-amber-700">{openPositions.length} 单 · {openAmount.toFixed(2)} USDT</span>
              : <span className="text-gray-400">--</span>}
          </div>
          {/* 可领取奖金：赢单 token 需手动 batch-redeem 才变 USDT（官方链路） */}
          <div className="flex justify-between gap-2 items-center">
            <span className="text-gray-500 shrink-0 flex items-center">
              可领取奖金
              <HelpHint text="赢单的奖金以获胜 token 形式留在链上钱包，不会自动变成 USDT；需要调官方 batch-redeem 赎回后才入预测钱包余额。赢单后记得来这里领取。" />
            </span>
            {Number(redeemable?.claimable_count ?? 0) > 0
              ? <span className="flex items-center gap-2">
                  <span className="font-mono font-semibold text-amber-600">
                    {String(redeemable?.claimable_count)} 个 token 待领取
                    {redeemable?.wallet_source === 'degraded' && <span className="text-gray-400 font-normal">（钱包查询降级，含本地兑底）</span>}
                  </span>
                  <button
                    onClick={handleRedeem} disabled={redeeming}
                    className="px-3 py-1 text-xs font-semibold rounded bg-amber-500 text-white disabled:opacity-50"
                  >{redeeming ? '领取中…' : '领取奖金'}</button>
                </span>
              : <span className="text-gray-400">--</span>}
          </div>
          {redeemResult && (
            <div className={`text-xs px-2 py-1 rounded break-all ${redeemResult.status === 'SUCCESS' ? 'bg-green-50 text-green-700' : redeemResult.status === 'NOOP' ? 'bg-gray-50 text-gray-500' : 'bg-red-50 text-red-600'}`}>
              {redeemResult.status === 'SUCCESS'
                ? `✓ 已领取 ${String(redeemResult.redeemed)} 个 token，奖金入预测钱包余额（前端刷新后可见）`
                : redeemResult.status === 'NOOP'
                  ? String(redeemResult.message ?? '无可领取持仓')
                  : `领取失败: ${String(redeemResult.error ?? '未知错误')}`}
            </div>
          )}
          {payAccounts.length > 0 && (
            <div className="flex justify-between gap-2">
              <span className="text-gray-500 shrink-0">支付账户余额</span>
              <span className="font-mono text-xs text-right break-all text-gray-800">
                {payAccounts.map(a => {
                  const t = String(a.accountType)
                  const label = PAY_ACCOUNT_LABEL[t] ?? t
                  const amt = Number(a.availableBalanceDisplay ?? 0)
                  const disabled = a.enabled === false
                  return `${label} ${Number.isFinite(amt) ? amt.toFixed(2) : '--'}${disabled ? '（禁用）' : ''}`
                }).join(' · ')}
              </span>
            </div>
          )}
          {payAccounts.length === 0 && (
          <div className="flex justify-between gap-2">
            <span className="text-gray-500 shrink-0">钱包持仓 Token</span>
            {walletAssets == null
              ? <span className="text-gray-400">暂不可查（探索型端点）</span>
              : heldTokens.length === 0
                ? <span className="text-gray-400">--（无 outcome token）</span>
                : <span className="font-mono text-xs text-right break-all text-gray-800">
                    {heldTokens.map(a => {
                      const sym = String(a.asset ?? a.symbol ?? '?')
                      const amt = Number(a.free ?? a.balance ?? 0)
                      return `${sym.length > 10 ? sym.slice(0, 10) + '…' : sym} × ${amt > 1e12 ? (amt / 1e18).toFixed(4) : amt}`
                    }).join(' | ')}
                  </span>}
          </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-gray-500 shrink-0">划转入金</span>
            <input
              type="number" min={0.1} max={20} step={0.5} value={transferAmt}
              onChange={e => setTransferAmt(e.target.value)}
              className="w-20 px-2 py-1 border border-gray-300 rounded text-gray-800"
            />
            <button
              onClick={handleTransferIn} disabled={transferring}
              className="px-3 py-1 text-xs font-semibold rounded bg-blue-600 text-white disabled:opacity-50"
            >{transferring ? '划转中…' : '现货 → 预测钱包'}</button>
            <button
              onClick={handleTransferOut} disabled={transferring}
              className="px-3 py-1 text-xs font-semibold rounded bg-emerald-600 text-white disabled:opacity-50"
            >{transferring ? '划转中…' : '预测钱包 → 现货'}</button>
          </div>
          {transferResult && (
            <div className={`text-xs px-2 py-1 rounded ${transferResult.status === 'SUCCESS' && transferResult.direction_confirmed !== false ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600 break-all'}`}>
              {transferResult.status === 'SUCCESS'
                ? transferResult.direction_confirmed === true
                  ? `划出成功，现货 ${typeof transferResult.spot_before === 'number' ? (transferResult.spot_before as number).toFixed(4) : '?'} → ${typeof transferResult.spot_after === 'number' ? (transferResult.spot_after as number).toFixed(4) : '?'}（方向已自证）`
                  : transferResult.direction_confirmed === false
                    ? `⚠ ${String(transferResult.warning ?? '划出已提交但现货余额未见增加，请立即人工核对划转记录')}`
                    : `划转成功，现货余额 → ${typeof transferResult.spot_usdt_free === 'number' ? (transferResult.spot_usdt_free as number).toFixed(4) : '?'}`
                : `划转失败: ${String(transferResult.error ?? '未知错误')}`}
            </div>
          )}
          <div className="border-t border-gray-100 my-2" />
          <div className="flex justify-between gap-2 items-center">
            <span className="text-gray-500 shrink-0 flex items-center">
              信号实盘通道
              <HelpHint text="10 个信号通道独立实盘：每通道独立开关/单笔金额/日限/执行价护栏。命中信号即下真实订单（FOK），重启回落 LIVE_CHANNELS_JSON 配置。" />
            </span>
            {liveChannels.length > 0
              ? <span className="flex items-baseline gap-1.5 text-right">
                  <span className={enabledCount > 0 ? 'text-green-700 font-semibold' : 'text-amber-700 font-semibold'}>
                    {enabledCount > 0 ? `${enabledCount}/${String(liveChannels.length)} 通道开启` : '全部关闭（不开火）'}
                  </span>
                  <span className="text-[10px] text-gray-400">
                    默认 {String(liveDefaults.amount_usdt ?? '--')}U/单 · 在途任务 {String(live?.pending_tasks ?? 0)}
                  </span>
                </span>
              : <span className="text-gray-400">未装配（启动异常，详见后端日志）</span>}
          </div>
          {/* 通道管理面板：每通道一行「信号名/护栏/今日成交/累计开火/金额输入+保存/开关」，独立 toggle + 金额热调 */}
          {liveChannels.length > 0 && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-2 space-y-1">
              {liveChannels.map(ch => {
                const info = SIGNAL_INFO[ch.channel]
                const draft = amountDrafts[ch.channel]
                const dirty = draft != null && draft !== String(ch.amount_usdt)
                return (
                  <div
                    key={ch.channel}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded border text-xs ${ch.enabled ? 'border-green-300 bg-green-50/60' : 'border-gray-200 bg-white'}`}
                  >
                    <span className="flex items-center gap-1.5 min-w-0 flex-1">
                      <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded border shrink-0 ${SIGNAL_KIND_BADGE[info?.kind ?? '影子']}`}>{info?.kind ?? '影子'}</span>
                      <span className="text-gray-800 font-medium truncate">{info?.name ?? ch.display_name}</span>
                      <span className="text-[10px] text-gray-400 font-mono shrink-0 hidden md:inline">{ch.channel}</span>
                      <HelpHint text={info?.desc ?? ch.display_name} />
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-gray-500 hidden sm:inline" title="执行价护栏 / 今日成交/日限 / 累计开火">
                      护栏{String(ch.max_exec_price)} · 今日{String(ch.filled_today ?? 0)}/{String(ch.max_daily_orders)} · 开火{String(ch.fire_total)}
                    </span>
                    <span className="shrink-0 flex items-center gap-1">
                      <input
                        type="number" min={0.1} max={50} step={0.5}
                        value={draft ?? String(ch.amount_usdt)}
                        onChange={e => setAmountDrafts(prev => ({ ...prev, [ch.channel]: e.target.value }))}
                        disabled={togglingLive}
                        title={`单笔金额（0.1~50 USDT，硬上限 ${String(live?.amount_cap ?? 50)}）`}
                        className="w-16 px-1.5 py-0.5 border border-gray-300 rounded text-gray-800 font-mono disabled:opacity-50"
                      />
                      <button
                        onClick={() => handleChannelAmount(ch)}
                        disabled={togglingLive || !dirty}
                        className="px-1.5 py-0.5 rounded border border-blue-400 bg-white text-blue-700 font-semibold disabled:opacity-40"
                        title="保存金额热调（立即生效，不影响在途任务）"
                      >存</button>
                    </span>
                    <button
                      onClick={() => handleChannelToggle(ch)}
                      disabled={togglingLive}
                      className={`shrink-0 px-2 py-0.5 rounded font-bold text-white disabled:opacity-50 ${ch.enabled ? 'bg-red-600' : 'bg-green-600'}`}
                    >{ch.enabled ? '停火' : '开火'}</button>
                  </div>
                )
              })}
              <p className="text-[10px] text-gray-400 pt-0.5">
                金额输入后点「存」热调（立即生效）；开关各通道独立互不影响；重启回落 LIVE_CHANNELS_JSON 配置。5m 通道随窗触发，15m 场景通道次周期开盘入场。
              </p>
            </div>
          )}
        </div>
      </Card>
      </div>

      <div className="lg:col-span-2">
        <Card title="最近订单">
          <div className="flex items-center justify-between mb-2 gap-2">
            <span className="text-xs text-gray-400">{orders.length} 条记录（每 15s 自动刷新）</span>
            <div className="flex items-center gap-2">
              {syncResult && (
                <span className={`text-xs px-2 py-0.5 rounded ${syncResult.error ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-700'}`}>
                  {syncResult.error
                    ? String(syncResult.error)
                    : `币安侧 ${String(syncResult.binance_orders ?? '?')} 单，已同步 ${String(syncResult.synced ?? 0)} 单`}
                </span>
              )}
              <button
                onClick={handleSyncBinance} disabled={syncing}
                className="px-3 py-1 text-xs font-semibold rounded bg-slate-600 text-white disabled:opacity-50"
              >{syncing ? '对账中…' : '对账（同步币安）'}</button>
            </div>
          </div>
          {orders.length === 0 ? (
            <p className="text-sm text-gray-400">暂无订单记录</p>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-gray-500 border-b border-gray-100">
                  <th className="py-1 pr-2">时间</th>
                  <th className="py-1 pr-2">版本</th>
                  <th className="py-1 pr-2">方向</th>
                  <th className="py-1 pr-2">状态</th>
                  <th className="py-1 pr-2">结果</th>
                  <th className="py-1 pr-2">均价</th>
                  <th className="py-1 pr-2">金额 (USDT)</th>
                  <th className="py-1 pr-2">盈亏</th>
                  <th className="py-1">说明</th>
                </tr>
              </thead>
              <tbody>
                {orders.map(o => (
                  <tr key={String(o.id)} className="border-b border-gray-50">
                    <td className="py-1.5 pr-2 text-gray-600 whitespace-nowrap">
                      {o.created_at ? new Date(String(o.created_at)).toLocaleString() : '--'}
                    </td>
                    <td className="py-1.5 pr-2 font-mono text-gray-700">{String(o.signal_version ?? '--')}</td>
                    <td className="py-1.5 pr-2">
                      {o.direction === 'UP'
                        ? <span className="px-1.5 py-0.5 rounded font-bold bg-green-100 text-green-700">UP</span>
                        : o.direction === 'DOWN'
                          ? <span className="px-1.5 py-0.5 rounded font-bold bg-red-100 text-red-700">DOWN</span>
                          : <span className="text-gray-400">--</span>}
                    </td>
                    <td className="py-1.5 pr-2">
                      <span className={`px-1.5 py-0.5 rounded font-bold ${o.status === 'FILLED' ? 'bg-green-100 text-green-700' : o.status === 'FAILED' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
                        {String(o.status)}
                      </span>
                    </td>
                    <td className="py-1.5 pr-2">
                      {o.win === true
                        ? <span className="px-1.5 py-0.5 rounded font-bold bg-green-100 text-green-700">WIN</span>
                        : o.win === false
                          ? <span className="px-1.5 py-0.5 rounded font-bold bg-red-100 text-red-700">LOSE</span>
                          : o.settle_outcome != null
                            ? <span className="px-1.5 py-0.5 rounded font-bold bg-gray-100 text-gray-600">{String(o.settle_outcome)}</span>
                            : <span className="text-gray-400">--</span>}
                    </td>
                    <td className="py-1.5 pr-2 font-mono">{o.average_price != null ? String(o.average_price) : '--'}</td>
                    <td className="py-1.5 pr-2 font-mono">
                      {o.amount_in != null ? (Number(o.amount_in) / 1e18).toFixed(2) : '--'}
                    </td>
                    <td className={`py-1.5 pr-2 font-mono ${typeof o.pnl === 'number' ? ((o.pnl as number) >= 0 ? 'text-green-700' : 'text-red-600') : 'text-gray-400'}`}>
                      {typeof o.pnl === 'number'
                        ? `${(o.pnl as number) >= 0 ? '+' : ''}${(o.pnl as number).toFixed(2)}`
                        : '--'}
                    </td>
                    <td className="py-1.5 text-gray-500">{String(o.error_message ?? '')}</td>
                  </tr>
                ))}
              </tbody>
              {settledCount > 0 && (
                <tfoot>
                  <tr className="border-t border-gray-100 text-gray-600">
                    <td colSpan={7} className="py-1.5">已结算 {settledCount} 单（本地估算口径）</td>
                    <td className={`py-1.5 font-mono font-bold ${totalPnl >= 0 ? 'text-green-700' : 'text-red-600'}`}>
                      {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-gray-400">USDT</td>
                  </tr>
                </tfoot>
              )}
            </table>
          )}
        </Card>
      </div>
    </div>
  )
}

// ============================================================
// Main App
// ============================================================

const TAB_IDS = ['market', 'agent', 'monitor', 'analysis', 'live'] as const
type TabId = (typeof TAB_IDS)[number]

// URL hash 记忆当前 tab：刷新 / 分享链接时回到原页，而不是一律落回首页
const tabFromHash = (): TabId => {
  const h = window.location.hash.slice(1)
  return (TAB_IDS as readonly string[]).includes(h) ? (h as TabId) : 'market'
}

// ============================================================
// 登录页（单一访问密码，登录态存 localStorage 不过期）
// ============================================================

function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    if (!password || loading) return
    setLoading(true); setError('')
    try {
      const resp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      const data = await resp.json().catch(() => ({}))
      if (resp.ok && data.token) {
        setAuthToken(data.token)
        onLogin()
      } else {
        setError(typeof data.detail === 'string' ? data.detail : '登录失败')
      }
    } catch {
      setError('网络错误，请重试')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[var(--bg)] flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-white rounded-2xl border border-gray-200 shadow-sm p-8">
        <h1 className="text-lg font-bold text-gray-900 mb-1 text-center">BTC 5min 预测系统</h1>
        <p className="text-xs text-gray-400 mb-6 text-center">请输入访问密码</p>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') submit() }}
          placeholder="访问密码"
          autoFocus
          className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500/40 focus:border-cyan-500"
        />
        {error && <div className="mt-2 text-xs text-red-500">{error}</div>}
        <button
          onClick={submit}
          disabled={loading || !password}
          className="mt-4 w-full px-4 py-2 text-sm font-semibold text-white bg-cyan-600 rounded-lg hover:bg-cyan-700 disabled:opacity-50 transition"
        >
          {loading ? '登录中...' : '登 录'}
        </button>
      </div>
    </div>
  )
}

export default function App() {
  // 登录门禁：无 token 时只渲染登录页；401/跨标签清除时回到登录页（刷新后凭 localStorage 直接进入）
  const [authed, setAuthed] = useState(() => !!getAuthToken())

  useEffect(() => {
    const onLogout = () => setAuthed(false)
    const onStorage = (e: StorageEvent) => { if (e.key === AUTH_KEY && !e.newValue) setAuthed(false) }
    window.addEventListener('auth-logout', onLogout)
    window.addEventListener('storage', onStorage)
    return () => {
      window.removeEventListener('auth-logout', onLogout)
      window.removeEventListener('storage', onStorage)
    }
  }, [])

  const handleLogout = () => { clearAuthToken(); setAuthed(false) }

  const [tab, setTab] = useState<TabId>(tabFromHash)

  // replaceState 不产生历史条目，避免污染浏览器后退键
  useEffect(() => {
    window.history.replaceState(null, '', '#' + tab)
  }, [tab])

  // 市场情绪
  const [pmPoints, setPmPoints] = useState<PMPoint[]>([])
  const [pmMarket, setPmMarket] = useState<Record<string, unknown> | null>(null)
  const [momentumLoading, setMomentumLoading] = useState(false)
  const [momentumResult, setMomentumResult] = useState<MomentumResult | null>(null)

  const refreshPm = useCallback(() => {
    api.getPredictionMarket().then(d => {
      setPmPoints(d.points || [])
      setPmMarket(d.market || null)
    }).catch(() => {})
  }, [])

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

  if (!authed) return <LoginPage onLogin={() => setAuthed(true)} />

  return (
    <div className="min-h-screen bg-[var(--bg)] flex flex-col">
      <header className="bg-white/95 backdrop-blur border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-1.5 flex justify-center">
          {/* 头部仅保留标签切换，最小化占用空间 */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-full p-1">
            <button
              onClick={() => setTab('market')}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition ${
                tab === 'market'
                  ? 'bg-white text-brand shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              市场情绪
            </button>
            <button
              onClick={() => setTab('agent')}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition ${
                tab === 'agent'
                  ? 'bg-white text-brand shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              Agent 自进化
            </button>
            <button
              onClick={() => setTab('monitor')}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition ${
                tab === 'monitor'
                  ? 'bg-white text-brand shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              运行监控
            </button>
            <button
              onClick={() => setTab('analysis')}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition ${
                tab === 'analysis'
                  ? 'bg-white text-brand shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              信号分析
            </button>
            <button
              onClick={() => setTab('live')}
              className={`px-3 py-1 text-xs font-semibold rounded-full transition ${
                tab === 'live'
                  ? 'bg-white text-brand shadow-sm'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-white/60'
              }`}
            >
              实盘交易
            </button>
          </div>
          <button
            onClick={handleLogout}
            title="退出登录"
            className="absolute right-4 top-1/2 -translate-y-1/2 text-[11px] text-gray-400 hover:text-red-500 transition"
          >
            退出
          </button>
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

            <FakeBreakoutPanel />

            <Market15mPanel />

            <Card title="概率动量分析（纯算法 · 独立备选方案）">
              <div className="text-xs text-gray-400 mb-3">
                基于预测市场 UP% 时序的多维度动量信号，纯算法不依赖 LLM/K线。手动触发，不参与自动决策。
              </div>
              <button
                onClick={handleMomentum}
                disabled={momentumLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-cyan-600 rounded-full hover:bg-cyan-700 disabled:opacity-50 transition mb-4"
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

        {tab === 'analysis' && <SignalAnalyticsTab />}
        {tab === 'live' && <LiveTradeTab />}
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
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-purple-600 rounded-full hover:bg-purple-700 transition"
          title="全量历史深度分析：预览发现结果，审核后写入模式库"
        >
          🔬 深度学习
        </button>
        <button
          onClick={() => setCmpOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-teal-600 rounded-full hover:bg-teal-700 transition"
          title="同一数据上对比纯 LLM 版与 Python 聚类版的多维准确率"
        >
          ⚖️ 方案对比
        </button>
        <button
          onClick={() => setEvoOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-brand rounded-full hover:bg-brand-hover transition"
          title="用样本外胜率趋势/代际对比/分轨证明系统在变好而非只在变化"
        >
          📈 进化看板
        </button>
      </div>

      {/* (a2) 模式池对比面板（横向 + 纵向回测对比） */}
      <div className="shrink-0 max-h-[45vh] overflow-auto">
        <PatternPoolPanel />
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
              className="px-2 py-0.5 border border-gray-200 rounded-lg text-[11px] focus:outline-none focus:ring-2 focus:ring-brand"
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
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[88vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-sm font-bold text-gray-800">📈 进化有效性看板</h2>
            <p className="text-[11px] text-gray-400 mt-0.5">用样本外胜率证明「在变好」而非「只在变化」。仅统计已验证的决策预测（UP/DOWN），随机基线 50%。</p>
          </div>
          <div className="flex items-center gap-2">
            <select value={days} onChange={e => setDays(Number(e.target.value))}
              className="px-2 py-1 border border-gray-200 rounded-lg text-[11px] focus:outline-none focus:ring-2 focus:ring-brand">
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
      const resp = await authFetch(
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
        className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col"
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
              className="px-3 py-1.5 text-xs font-semibold text-white bg-purple-600 rounded-full hover:bg-purple-700 disabled:opacity-50 transition"
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
                          className="mt-1 text-[10px] text-brand hover:underline"
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

// ============================================================
// 假突破信号面板（market tab）：日线阻力破位检测，暂不下注
// ============================================================

function fmtMs(ms: number | null): string {
  if (!ms) return '--'
  return new Date(ms).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function FakeBreakoutPanel() {
  const [status, setStatus] = useState<FakeBreakoutStatus | null>(null)
  const [signals, setSignals] = useState<FakeBreakoutSignal[]>([])
  const [stats, setStats] = useState<FakeBreakoutStats | null>(null)
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set())
  const togglePath = (id: number) => setExpandedIds(prev => {
    const next = new Set(prev)
    if (next.has(id)) next.delete(id); else next.add(id)
    return next
  })

  const refresh = useCallback(() => {
    api.getFakeBreakoutStatus().then(setStatus).catch(() => {})
    api.getFakeBreakoutSignals(50).then(d => setSignals(d.signals || [])).catch(() => {})
    api.getFakeBreakoutStats().then(setStats).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 15000)
    return () => clearInterval(timer)
  }, [refresh])

  const levels = status?.levels || {}
  const levelOrder = ['1h', '4h', 'daily']
  // 胜负判定：side=high 买 DOWN（结算 DOWN 赢）；side=low 买 UP（结算 UP 赢）
  const isWin = (s: FakeBreakoutSignal, oc: string | null) =>
    oc !== null && oc === (s.side === 'high' ? 'DOWN' : 'UP')
  const winBadge = (oc: string | null) =>
    oc === 'DOWN' ? '↓ DOWN' : oc === 'UP' ? '↑ UP' : '— NOISE'

  // 场景类型 → 徽章样式与中文名（S1/S2/S4/S5，2026-08-18 新增 S5 确认入场）
  const sceneMeta: Record<string, { label: string; cls: string; desc: string }> = {
    bull_exhaust: {
      label: 'S1 多头耗尽', cls: 'bg-red-50 text-red-700 border-red-200',
      desc: '破4h高·光头阳·4h上沿 → 次周期 DOWN',
    },
    bear_exhaust: {
      label: 'S2 空头耗尽', cls: 'bg-green-50 text-green-700 border-green-200',
      desc: '破4h低·收阴·放量≥2.0 → 次周期 UP',
    },
    momentum_fade: {
      label: 'S4 动量衰竭', cls: 'bg-blue-50 text-blue-700 border-blue-200',
      desc: '连阳≥3·光头阳，无破位要求 → 次周期 DOWN',
    },
    bull_exhaust_confirm: {
      label: 'S5 确认入场', cls: 'bg-purple-50 text-purple-700 border-purple-200',
      desc: 'S1 信号后 +5min 回落确认（5m 收盘 < 周期开盘）→ 买 DOWN 持有到期；盈亏平衡入场价 0.77',
    },
  }
  const sceneOf = (s: FakeBreakoutSignal) => sceneMeta[s.pattern_type || ''] || null

  return (
    <Card title="假突破信号（1h/4h/日线 × 阻力/支撑 · 暂不下注）">
      <div className="text-xs text-gray-400 mb-3">
        秒级检测 BTC 破位：冲过阻力→看跌（买 DOWN）/ 跌破支撑→看涨（买 UP）。
        回测：日线破阻力 80%、4h 73.5%、1h 65.1%（支撑方向对称成立）。当前只记录信号 + 邮件提醒。
      </div>

      {/* 状态行 */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-3 text-xs">
        <span className="flex items-center gap-1.5">
          <StatusDot ok={!!status?.running} />
          <span className="text-gray-500">检测器</span>
          <span className={`font-bold ${status?.running ? 'text-green-600' : 'text-red-500'}`}>
            {status?.running ? '运行中' : '未运行'}
          </span>
        </span>
        <span className="text-gray-500">
          当前价 <strong className="text-gray-800 font-mono">{status?.btc_mid ? status.btc_mid.toFixed(0) : '--'}</strong>
        </span>
        <span className="text-gray-500">今日信号 <strong className="font-mono text-gray-800">{status?.daily_count ?? 0}</strong> 条</span>
      </div>

      {/* 三级别位势 */}
      {Object.keys(levels).length > 0 && (
        <div className="grid grid-cols-3 gap-2 mb-3 text-xs">
          {levelOrder.filter(l => levels[l]).map(l => {
            const lv = levels[l]
            const mid = status?.btc_mid || 0
            const distRes = mid > 0 ? (mid / lv.resistance - 1) * 100 : null
            const distSup = mid > 0 ? (mid / lv.support - 1) * 100 : null
            const brokeRes = distRes !== null && distRes > 0
            const brokeSup = distSup !== null && distSup < 0
            return (
              <div key={l} className="px-2.5 py-2 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="font-bold text-gray-700 mb-1">{l === 'daily' ? '日线' : l} <span className="text-gray-400 font-normal">（{l === 'daily' ? 288 : l === '4h' ? 48 : 12} 窗）</span></div>
                <div className={`font-mono ${brokeRes ? 'text-red-600 font-bold' : 'text-gray-500'}`}>
                  阻力 {lv.resistance.toFixed(0)}{distRes !== null && <span className="ml-1">({distRes >= 0 ? '+' : ''}{distRes.toFixed(2)}%){brokeRes && ' ⚠️'}</span>}
                </div>
                <div className={`font-mono ${brokeSup ? 'text-green-600 font-bold' : 'text-gray-500'}`}>
                  支撑 {lv.support.toFixed(0)}{distSup !== null && <span className="ml-1">({distSup >= 0 ? '+' : ''}{distSup.toFixed(2)}%){brokeSup && ' ⚠️'}</span>}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* 分组统计（级别×方向） */}
      {stats && stats.total_signals > 0 && (
        <div className="mb-3 px-3 py-2 bg-amber-50 border border-amber-200 rounded-lg text-xs">
          <div className="flex flex-wrap gap-x-5 gap-y-1 mb-1.5">
            <span className="text-gray-600">累计信号 <strong>{stats.total_signals}</strong></span>
            <span className="text-gray-600">已结算（15m） <strong>{stats.settled}</strong></span>
            <span className="text-gray-600">已结算（5m） <strong>{stats.settled_5m}</strong></span>
          </div>
          {stats.by_group.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {stats.by_group.map(g => (
                <span key={`${g.level}-${g.side}`} className="px-2 py-1 bg-white border border-amber-200 rounded font-mono text-[11px]">
                  <strong>{g.level}</strong>{g.side === 'high' ? '↓' : '↑'}：
                  15m {g.win_rate_15m !== null ? `${(g.win_rate_15m * 100).toFixed(0)}%` : '--'}({g.settled_15m})
                  {' '}5m {g.win_rate_5m !== null ? `${(g.win_rate_5m * 100).toFixed(0)}%` : '--'}({g.settled_5m})
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 按场景类型统计（6 统计维度：胜率/累计EV/入场EV/回撤/近7日频率/收益曲线） */}
      {stats && stats.by_pattern_type && stats.by_pattern_type.length > 0 && (
        <div className="mb-3 grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
          {stats.by_pattern_type.map(ps => {
            const meta = sceneMeta[ps.pattern_type]
            const research = stats.research_win_rates?.[ps.pattern_type]
            return (
              <div key={ps.pattern_type} className="px-2.5 py-2 bg-white border border-gray-200 rounded-lg">
                <div className="flex items-center gap-1 mb-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${meta?.cls || 'bg-gray-100 text-gray-500 border-gray-200'}`}>
                    {meta?.label || ps.pattern_type}
                  </span>
                  {research !== undefined && (
                    <span className="text-[10px] text-gray-400" title="回测胜率点估计（入场 EV 用的 p）">回测 {(research * 100).toFixed(1)}%</span>
                  )}
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-gray-600">
                  <span title={`已结算正式信号 ${ps.n} 条（胜 ${ps.wins}）`}>
                    实盘 <strong className={ps.winrate !== null && research !== undefined && ps.winrate >= research ? 'text-green-600' : 'text-gray-800'}>
                      {ps.winrate !== null ? `${(ps.winrate * 100).toFixed(0)}%` : '--'}
                    </strong>({ps.n})
                  </span>
                  <span title="累计实现 EV/事件（1 USDT 本金：赢 0.98/entry−1，输 −1）">
                    EV <strong className={(ps.cumulative_ev ?? 0) >= 0 ? 'text-green-600' : 'text-red-500'}>
                      {ps.cumulative_ev !== null ? `${ps.cumulative_ev >= 0 ? '+' : ''}${ps.cumulative_ev.toFixed(3)}` : '--'}
                    </strong>
                  </span>
                  <span title="入场时刻预期 EV 均值 = p×(1−费)/entry−1（p=回测胜率）">
                    入场EV {ps.avg_ev_at_entry !== null ? `${ps.avg_ev_at_entry >= 0 ? '+' : ''}${ps.avg_ev_at_entry.toFixed(3)}` : '--'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-gray-400 mt-0.5">
                  <span title="累计收益曲线峰值→最大回撤">峰值 {ps.peak_equity.toFixed(2)} / 回撤 {ps.max_drawdown.toFixed(2)}</span>
                  <span title="近 7 日事件数（频率监控）">近7日 {ps.n_last_7d}</span>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* 信号列表 */}
      {signals.length === 0 ? (
        <div className="text-center text-gray-400 py-6 text-sm">暂无信号——BTC 盘中破位任一级别阻力/支撑时自动记录并邮件提醒</div>
      ) : (
        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white">
              <tr className="border-b border-gray-200 text-gray-500">
                <th className="w-6" title="展开次周期价格+报价路径"></th>
                <th className="py-1.5 px-2 text-left">信号时间</th>
                <th className="py-1.5 px-2 text-center">场景</th>
                <th className="py-1.5 px-2 text-center">级别/方向</th>
                <th className="py-1.5 px-2 text-right">破位价 / 位价</th>
                <th className="py-1.5 px-2 text-right" title="入场报价快照：信号后次周期开盘 ~8s 的 15m 市场目标方向 token 价（S5 为 +5min 确认时刻价）；缺快照时回退信号瞬间报价">开盘入场价</th>
                <th className="py-1.5 px-2 text-right" title="信号后 5 分钟（次周期 1/3 处）15m 市场目标方向 token 报价：S5 确认入场真实可得价 / 提前离场定价对照">+5m 报价</th>
                <th className="py-1.5 px-2 text-right" title="信号所在 15m 周期：开盘价 → 周期末价（市场按这两者定涨跌）">15m 开→末</th>
                <th className="py-1.5 px-2 text-center" title="信号所在 5m 周期的涨跌方向：周期末价 vs 周期开盘价，与币安市场真实结算规则一致">5m 周期</th>
                <th className="py-1.5 px-2 text-center" title="信号所在 15m 周期的涨跌方向：周期末价 vs 周期开盘价，即市场真实结算结果">15m 周期</th>
                <th className="py-1.5 px-2 text-center">状态</th>
              </tr>
            </thead>
            <tbody>
              {signals.map(s => {
                const entrySnap = s.side === 'high' ? s.entry_down_price_15m : s.entry_up_price_15m
                const entryFire = s.side === 'high' ? s.down_price_15m : s.up_price_15m
                const entry15 = entrySnap ?? entryFire
                const q5 = s.side === 'high' ? s.quote5m_down_15m : s.quote5m_up_15m
                const scene = sceneOf(s)
                const sceneTip = [
                  scene?.desc,
                  s.close_pos !== null ? `收盘位置 ${(s.close_pos ?? 0).toFixed(3)}` : null,
                  s.vol_ratio !== null ? `量比 ${(s.vol_ratio ?? 0).toFixed(2)}` : null,
                  s.ev_at_entry !== null ? `入场EV ${s.ev_at_entry >= 0 ? '+' : ''}${s.ev_at_entry.toFixed(3)}` : null,
                  s.cumulative_winrate !== null ? `累计胜率 ${(s.cumulative_winrate * 100).toFixed(1)}%` : null,
                  s.cumulative_ev !== null ? `累计EV ${s.cumulative_ev >= 0 ? '+' : ''}${s.cumulative_ev.toFixed(3)}` : null,
                ].filter(Boolean).join('\n')
                return (
                  <Fragment key={s.id}>
                  <tr className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-1.5 px-1 text-center">
                      <button onClick={() => togglePath(s.id)}
                        className="text-gray-400 hover:text-indigo-600 leading-none text-xs"
                        title="展开次周期价格+报价路径（15s 采样，8/13 起有数据）">
                        {expandedIds.has(s.id) ? '▾' : '▸'}
                      </button>
                    </td>
                    <td className="py-1.5 px-2 text-gray-600 font-mono">{fmtMs(s.signal_time)}</td>
                    <td className="py-1.5 px-2 text-center" title={sceneTip}>
                      {scene ? (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${scene.cls}`}>
                          {scene.label}
                        </span>
                      ) : (
                        <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-gray-100 text-gray-400">旧信号</span>
                      )}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200 mr-1">{s.level}</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                        s.side === 'high' ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'
                      }`}>{s.side === 'high' ? '看跌' : '看涨'}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-800">
                      {s.btc_price.toFixed(0)} <span className="text-gray-400">/ {s.resistance.toFixed(0)}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-red-500" title={entrySnap !== null ? '开盘后 ~8s 入场快照' : '信号瞬间报价（无入场快照）'}>
                      {entry15?.toFixed(3) ?? '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-500">{q5?.toFixed(3) ?? '--'}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-600">
                      {s.cycle_open_price_15m ? `${s.cycle_open_price_15m.toFixed(0)}→` : ''}{s.settle_btc_price?.toFixed(0) ?? '--'}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {s.settle_outcome_5m ? (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          isWin(s, s.settle_outcome_5m) ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>{isWin(s, s.settle_outcome_5m) ? `✓ ${winBadge(s.settle_outcome_5m)}` : `✗ ${winBadge(s.settle_outcome_5m)}`}</span>
                      ) : <span className="text-gray-300" title="未到 +5min 结算时点，或该时点因部署重启错过（超宽限不回填防失真）">待结算</span>}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {s.settle_outcome ? (
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          isWin(s, s.settle_outcome) ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>{isWin(s, s.settle_outcome) ? `✓ ${winBadge(s.settle_outcome)} 赢` : `✗ ${winBadge(s.settle_outcome)}`}</span>
                      ) : <span className="text-gray-300" title="15m 市场尚未到期">待结算</span>}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`text-[10px] ${
                        s.status === 'SETTLED' ? 'text-gray-400' : s.status === 'PENDING' ? 'text-orange-500' : 'text-gray-400'
                      }`}>
                        {s.status === 'SETTLED' ? '已结算' : s.status === 'PENDING' ? '跟踪中' : s.status}
                      </span>
                      {s.email_sent && <span className="text-[10px] text-blue-400 ml-1" title="邮件已推送">📧</span>}
                    </td>
                  </tr>
                  {expandedIds.has(s.id) && (
                    <tr className="border-b border-gray-100">
                      <td colSpan={11} className="p-0">
                        <SignalPathPanel sig={s} />
                      </td>
                    </tr>
                  )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

// ============================================================
// 信号次周期路径面板（行内展开）：价格 ±bp 上板 / DOWN 报价下板
// 数据源 prediction_market_samples 15s 采样（8/13 起积累），懒加载
// ============================================================

interface SignalPathPoint { off: number; btc: number; down: number; up: number }
interface SignalPathData {
  signal_id: number
  cycle_start: number
  cycle_end: number
  open: number | null
  side: 'high' | 'low'
  settle: string | null
  quote5m_off: number | null
  quote5m_down: number | null
  has_data: boolean
  points: SignalPathPoint[]
}

function SignalPathPanel({ sig }: { sig: FakeBreakoutSignal }) {
  const [data, setData] = useState<SignalPathData | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let alive = true
    api.getFakeBreakoutSignalPath(sig.id)
      .then(d => { if (alive) setData(d) })
      .catch(() => { if (alive) setFailed(true) })
    return () => { alive = false }
  }, [sig.id])

  if (failed) return <div className="py-4 text-center text-xs text-gray-400">路径加载失败</div>
  if (!data) return <div className="py-4 text-center text-xs text-gray-400">路径加载中…</div>
  if (!data.has_data || !data.open) {
    return (
      <div className="py-4 text-center text-xs text-gray-400" title="15m 市场采样自 2026-08-13 开始积累，更早的信号无路径数据">
        暂无采样数据（8/13 前的信号或采样缺失）
      </div>
    )
  }

  const open = data.open
  const pts = data.points.map(p => ({ off: p.off, dev: (p.btc / open - 1) * 10000, down: p.down }))
  const maxDev = Math.max(5, Math.ceil(Math.max(...pts.map(p => Math.abs(p.dev))) / 5) * 5)
  const x5 = data.quote5m_off ?? 300
  const mmss = (s: number) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`
  const last = pts[pts.length - 1]
  const minDev = Math.min(...pts.map(p => p.dev))
  const maxUp = Math.max(...pts.map(p => p.dev))
  const won = data.settle === (sig.side === 'high' ? 'DOWN' : 'UP')
  const xTicks = [0, 180, 300, 600, 900]
  const tipFmt = (v: unknown, n: unknown) =>
    n === 'dev' ? [`${(v as number).toFixed(1)}bp`, '价格'] : [(v as number).toFixed(3), 'DOWN报价']

  return (
    <div className="px-2 py-2.5 bg-gray-50/70">
      <div className="flex justify-between items-center mb-1 text-[11px] text-gray-500">
        <span>
          次周期路径 · 开盘 {open.toLocaleString()} ·
          <span className="text-blue-600"> ■ 价格 ±bp</span> ·
          <span className="text-red-500"> ■ DOWN 报价</span>
          <span className="ml-2 px-1 bg-amber-50 text-amber-600 border border-amber-200 rounded"
            title="S7 研究结论：早期窗口（t≤4 分钟）是涨态/跌态双分支最优入场区">
            黄带 = t≤4 早期入场研究窗
          </span>
        </span>
        {data.settle && (
          <span className={won ? 'text-green-600 font-bold' : 'text-red-500 font-bold'}>
            结算 {data.settle} · {won ? '信号赢' : '信号输'}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={128}>
        <LineChart data={pts} margin={{ top: 6, right: 44, left: 0, bottom: 0 }}>
          <ReferenceArea x1={0} x2={240} fill="#fef3c7" fillOpacity={0.55} />
          <ReferenceLine y={0} stroke="#9ca3af" strokeDasharray="3 3" />
          <ReferenceLine x={x5} stroke="#a855f7" strokeDasharray="4 3"
            label={{ value: '+5m', position: 'insideTopRight', fill: '#a855f7', fontSize: 10 }} />
          <XAxis dataKey="off" type="number" domain={[0, 900]} ticks={xTicks} hide />
          <YAxis domain={[-maxDev, maxDev]} width={46}
            ticks={[-maxDev, 0, maxDev]} tickFormatter={v => `${v}bp`}
            tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <Tooltip formatter={tipFmt} labelFormatter={o => `+${mmss(o as number)}`} />
          <Line dataKey="dev" stroke="#2563eb" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={74}>
        <LineChart data={pts} margin={{ top: 2, right: 44, left: 0, bottom: 0 }}>
          <ReferenceArea x1={0} x2={240} fill="#fef3c7" fillOpacity={0.55} />
          <ReferenceLine x={x5} stroke="#a855f7" strokeDasharray="4 3" />
          {data.quote5m_down != null && data.quote5m_off != null && (
            <ReferenceDot x={data.quote5m_off} y={data.quote5m_down} r={3.5} fill="#a855f7" stroke="white" />
          )}
          <XAxis dataKey="off" type="number" domain={[0, 900]} ticks={xTicks}
            tickFormatter={v => `${Math.round(v / 60)}分`} tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <YAxis domain={[0, 1]} width={46} ticks={[0, 0.5, 1]}
            tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 10, fill: '#9ca3af' }} />
          <Tooltip formatter={tipFmt} labelFormatter={o => `+${mmss(o as number)}`} />
          <Line dataKey="down" stroke="#dc2626" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] text-gray-500 font-mono">
        <span>极值 {minDev.toFixed(1)} ~ +{maxUp.toFixed(1)}bp</span>
        <span>末点 {last.dev.toFixed(1)}bp / q(DOWN) {last.down.toFixed(3)}</span>
        {data.quote5m_down != null && (
          <span className="text-purple-600">+5m 确认价 q={data.quote5m_down.toFixed(3)}</span>
        )}
      </div>
    </div>
  )
}

// ============================================================
// 模式池对比面板（agent tab）：横向模式对比 + 纵向回测历史
// ============================================================

function TierBadge({ tier }: { tier: string }) {
  const meta: Record<string, string> = {
    S: 'bg-yellow-100 text-yellow-800 border-yellow-400',
    A: 'bg-purple-100 text-purple-800 border-purple-300',
    B: 'bg-blue-100 text-blue-800 border-blue-300',
    C: 'bg-gray-100 text-gray-600 border-gray-300',
  }
  return (
    <span className={`inline-block w-6 text-center px-1 py-0.5 rounded border text-[11px] font-black ${meta[tier] || meta.C}`}>
      {tier}
    </span>
  )
}

function PatternPoolPanel() {
  const [items, setItems] = useState<PatternCompareItem[]>([])
  const [loading, setLoading] = useState(false)
  const [reevalMsg, setReevalMsg] = useState('')
  const [expanded, setExpanded] = useState<number | null>(null)
  const [runs, setRuns] = useState<PatternBacktestRun[]>([])

  const refresh = useCallback(() => {
    api.getPatternCompare().then(d => setItems(d.patterns || [])).catch(() => {})
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const handleReevaluate = async () => {
    setLoading(true)
    setReevalMsg('')
    try {
      const res = await api.triggerReevaluate()
      const s = res.summary || {}
      setReevalMsg(`✅ 重回测完成：${s.patterns ?? 0} 模式 / ${s.windows ?? 0} 窗口 / 定级变更 ${(s.tier_changes || []).length} 起`)
      refresh()
    } catch (e) {
      setReevalMsg(`❌ 失败: ${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  const toggleRuns = async (pid: number) => {
    if (expanded === pid) {
      setExpanded(null)
      setRuns([])
      return
    }
    setExpanded(pid)
    try {
      const d = await api.getPatternBacktestRuns(pid)
      setRuns(d.runs || [])
    } catch {
      setRuns([])
    }
  }

  const pct = (v: number | null | undefined) => v === null || v === undefined ? '--' : `${(v * 100).toFixed(1)}%`

  return (
    <Card title="模式池对比（S/A/B/C 分级 · 定期重回测 · 只发现不下注）">
      <div className="flex justify-between items-center mb-2">
        <div className="text-[11px] text-gray-400">
          新数据累积到阈值后自动全量重回测；点击行展开该模式的回测历史（纵向对比）
        </div>
        <div className="flex items-center gap-2">
          {reevalMsg && <span className="text-[11px] text-gray-500">{reevalMsg}</span>}
          <button
            onClick={handleReevaluate}
            disabled={loading}
            className="px-2.5 py-1 text-[11px] font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
            title="立即对全部谓词模式执行一轮全量历史回测"
          >
            {loading ? '回测中...' : '🔄 立即重回测'}
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="text-center text-gray-400 py-6 text-sm">暂无极式（深度学习发现后自动入库并参与重回测）</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 text-gray-500">
                <th className="py-1.5 px-2 text-center">池</th>
                <th className="py-1.5 px-2 text-left">模式</th>
                <th className="py-1.5 px-2 text-center">方向</th>
                <th className="py-1.5 px-2 text-right">最新回测胜率</th>
                <th className="py-1.5 px-2 text-right">Wilson 下界</th>
                <th className="py-1.5 px-2 text-right">费后 EV</th>
                <th className="py-1.5 px-2 text-right">样本</th>
                <th className="py-1.5 px-2 text-left">最近变化</th>
              </tr>
            </thead>
            <tbody>
              {items.map(p => (
                <Fragment key={p.pattern_id}>
                  <tr
                    className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                    onClick={() => toggleRuns(p.pattern_id)}
                  >
                    <td className="py-1.5 px-2 text-center"><TierBadge tier={p.tier} /></td>
                    <td className="py-1.5 px-2 font-medium text-gray-700">
                      {p.pattern_name}
                      <span className="ml-1 text-[10px] text-gray-400">{p.status}</span>
                    </td>
                    <td className="py-1.5 px-2 text-center"><DirectionBadge direction={p.predicted_direction} /></td>
                    <td className="py-1.5 px-2 text-right font-mono font-bold text-gray-800">
                      {p.latest_run ? pct(p.latest_run.win_rate) : '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-600">
                      {p.latest_run?.wilson_lower !== null && p.latest_run?.wilson_lower !== undefined
                        ? p.latest_run.wilson_lower.toFixed(3) : '--'}
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono font-bold ${
                      p.latest_run?.ev_after_fee !== null && p.latest_run?.ev_after_fee !== undefined && p.latest_run.ev_after_fee > 0
                        ? 'text-green-600' : 'text-red-500'
                    }`}>
                      {p.latest_run?.ev_after_fee !== null && p.latest_run?.ev_after_fee !== undefined
                        ? `${p.latest_run.ev_after_fee > 0 ? '+' : ''}${p.latest_run.ev_after_fee.toFixed(3)}` : '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-600">
                      {p.latest_run?.sample_count ?? 0}
                    </td>
                    <td className="py-1.5 px-2 text-gray-500 text-[11px] max-w-56 truncate" title={(p.latest_run?.delta_vs_prev?.suggestions || []).join('\n')}>
                      {p.latest_run?.delta_vs_prev?.suggestions?.[0] ?? '—'}
                    </td>
                  </tr>
                  {expanded === p.pattern_id && (
                    <tr>
                      <td colSpan={8} className="bg-gray-50 px-4 py-3">
                        <div className="text-[11px] font-bold text-gray-600 mb-2">回测历史（纵向对比：同一模式随数据累积的表现漂移）</div>
                        {runs.length === 0 ? (
                          <div className="text-gray-400 text-xs py-2">该模式尚无回测记录</div>
                        ) : (
                          <div className="space-y-2">
                            {runs.map(r => (
                              <div key={r.id} className="bg-white rounded-lg border border-gray-200 px-3 py-2">
                                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-gray-600">
                                  <span className="font-mono">{r.created_at ? new Date(r.created_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '--'}</span>
                                  <span>胜率 <strong className="font-mono">{pct(r.win_rate)}</strong>（{r.correct_count}/{r.sample_count}）</span>
                                  <span>CI [{r.wilson_lower?.toFixed(3) ?? '--'}, {r.wilson_upper?.toFixed(3) ?? '--'}]</span>
                                  <span>EV <strong className="font-mono">{r.ev_after_fee !== null ? `${(r.ev_after_fee ?? 0) > 0 ? '+' : ''}${r.ev_after_fee?.toFixed(3)}` : '--'}</strong></span>
                                  <span className="text-gray-400">{r.trigger_reason}</span>
                                  {r.delta_vs_prev?.tier_change && (
                                    <span className="text-indigo-600 font-bold">
                                      {r.delta_vs_prev.tier_change.from} → {r.delta_vs_prev.tier_change.to}
                                    </span>
                                  )}
                                  {r.delta_vs_prev?.decay_warning && <span className="text-red-500 font-bold">⚠ 衰减降级</span>}
                                </div>
                                {(r.delta_vs_prev?.suggestions || []).length > 0 && (
                                  <ul className="mt-1 text-[11px] text-gray-500 space-y-0.5">
                                    {(r.delta_vs_prev?.suggestions || []).map((s, i) => <li key={i}>· {s}</li>)}
                                  </ul>
                                )}
                                {r.segment_stats && Object.keys(r.segment_stats).length > 0 && (
                                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-gray-400 font-mono">
                                    {Object.entries(r.segment_stats).map(([month, seg]) => (
                                      <span key={month}>{month}: {pct(seg.win_rate)}({seg.n})</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            ))}
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
  )
}

// ============================================================
// 15 分钟预测市场面板（market tab）：假突破策略的兑现载体报价
// ============================================================

function Market15mPanel() {
  const [points, setPoints] = useState<PMPoint[]>([])
  const [market, setMarket] = useState<Record<string, unknown> | null>(null)
  const [, setTick] = useState(0)

  const refresh = useCallback(() => {
    api.getPredictionMarket15m().then(d => {
      setPoints(d.points || [])
      setMarket(d.market || null)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    refresh()
    const timer = setInterval(refresh, 15000)
    const tickTimer = setInterval(() => setTick(t => t + 1), 1000)  // 倒计时每秒刷新
    return () => { clearInterval(timer); clearInterval(tickTimer) }
  }, [refresh])

  const endMs = (market?.end_date as number) || 0
  const remaining = Math.max(0, Math.floor((endMs - Date.now()) / 1000))
  const min = Math.floor(remaining / 60)
  const sec = remaining % 60
  const last = points[points.length - 1]

  return (
    <Card title="BTC 15 分钟涨跌市场（假突破策略兑现载体）">
      <div className="text-xs text-gray-400 mb-3">
        币安预测市场 15m 期的 UP/DOWN 报价。冲高破位瞬间 DOWN token 被砸出的低价，就是假突破策略的下注赔率。
      </div>
      <div className="flex flex-wrap gap-4 mb-3 text-xs">
        <span className="px-2 py-1 bg-orange-50 text-orange-700 rounded font-medium">🟠 15m 期</span>
        {endMs > 0 && (
          <span className="text-orange-500 font-mono">⏱ 距到期 {min}:{String(sec).padStart(2, '0')}</span>
        )}
        {last && (
          <>
            <span className="text-gray-500">
              DOWN 现价 <strong className="text-red-500 font-mono">{last.down_price !== null ? last.down_price.toFixed(3) : '--'}</strong>
              <span className="text-gray-400">（越低赔率越肥）</span>
            </span>
            <span className="text-gray-500">
              UP 现价 <strong className="text-green-600 font-mono">{last.up_price !== null ? last.up_price.toFixed(3) : '--'}</strong>
            </span>
            {last.btc_price && (
              <span className="text-gray-500">BTC <strong className="font-mono text-gray-800">{last.btc_price.toFixed(0)}</strong></span>
            )}
          </>
        )}
      </div>
      {points.length > 0 ? (
        <>
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={points.map(d => ({
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
            <span>共 {points.length} 个采样点（约 {Math.round(points.length / 4)} 分钟）</span>
          </div>
        </>
      ) : (
        <div className="text-center text-gray-400 py-10 text-sm">正在采集 15m 市场数据...每 15 秒采样一次。</div>
      )}
    </Card>
  )
}

// ============================================================
// 信号分析 Tab：胜率曲线 × BTC K线 × 周期归因（60s 轮询自动刷新）
// 口径全部来自后端 /api/signals/analytics（单一事实源），前端只做渲染
// ============================================================

const SHADOW_META: Record<string, { label: string; color: string }> = {
  x4_v1: { label: 'X4 misalign→DOWN', color: '#1f77b4' },
  quote_momentum_v1: { label: 'A momentum', color: '#d62728' },
  quote_contrarian_v1: { label: 'B contrarian', color: '#2ca02c' },
  krev_a_v1: { label: 'KREV-A 反转→UP', color: '#9467bd' },
  krev_b_v1: { label: 'KREV-B 反转→UP', color: '#e377c2' },
}
const SCENE_META: Record<string, { label: string; color: string }> = {
  bull_exhaust: { label: 'S1 bull_exhaust→DOWN', color: '#1f77b4' },
  bear_exhaust: { label: 'S2 bear_exhaust→UP', color: '#d62728' },
  momentum_fade: { label: 'S4 momentum_fade→DOWN', color: '#2ca02c' },
  bull_exhaust_confirm: { label: 'S5 confirm→DOWN', color: '#9467bd' },
  // legacy = pattern_type 为空的历史信号，胜负按 side 映射（与 /api/fake-breakout/stats
  // 同语义，审计脚本对其「一律 DOWN」的简化口径在 side=low 时不同）
  legacy: { label: 'legacy 历史', color: '#7f7f7f' },
}
// 后端动态发现的新版本/新场景的备用色（超出已知名单时按序取用）
const EXTRA_COLORS = ['#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']

const pct1 = (v: number | null | undefined, digits = 1) =>
  v == null ? '—' : `${(v * 100).toFixed(digits)}%`
const evFmt = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}`
const utcMD = (ts: number) =>
  new Date(ts).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', timeZone: 'UTC' })

// 把各版本曲线按时间戳合并成宽表（recharts 单 data 多 Line 需要）
function mergeCurves(
  blocks: Record<string, { curve: AnalyticsCurvePoint[] }>,
  meta: Record<string, { label: string; color: string }>,
): Record<string, number>[] {
  const rowMap = new Map<number, Record<string, number>>()
  for (const [key, blk] of Object.entries(blocks)) {
    const label = meta[key]?.label ?? key
    for (const p of blk.curve) {
      const row = rowMap.get(p.ts) ?? { ts: p.ts }
      row[label] = +(p.cum_wr * 100).toFixed(1)
      rowMap.set(p.ts, row)
    }
  }
  return Array.from(rowMap.values()).sort((a, b) => (a.ts as number) - (b.ts as number))
}

function SignalAnalyticsTab() {
  const [analytics, setAnalytics] = useState<SignalsAnalytics | null>(null)
  const [klines, setKlines] = useState<BtcKline[]>([])
  const [kinterval, setKinterval] = useState<'1d' | '1h'>('1d')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [err, setErr] = useState('')
  // 请求序号守卫：切换周期时丢弃旧请求的慢返回，防止过期数据覆盖新数据
  const reqIdRef = useRef(0)

  const load = useCallback(() => {
    const id = ++reqIdRef.current
    Promise.all([
      api.getSignalsAnalytics(),
      api.getBtcKlines(kinterval, kinterval === '1d' ? 30 : 168),
    ]).then(([a, k]) => {
      if (id !== reqIdRef.current) return
      if (a && a.shadow) { setAnalytics(a as SignalsAnalytics); setErr('') } else setErr('分析数据返回异常')
      if (k && Array.isArray(k.klines)) setKlines(k.klines as BtcKline[])
      setLastUpdate(new Date())
    }).catch(e => {
      if (id === reqIdRef.current) setErr(`请求失败: ${(e as Error).message}`)
    })
  }, [kinterval])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (!autoRefresh) return
    const t = setInterval(load, 60_000)
    return () => clearInterval(t)
  }, [autoRefresh, load])

  const pumpTs = analytics?.pump_ts ?? null

  // ---- BTC K线图上信号落点（按 bar 聚合：场景=橙 / 影子=蓝）----
  const barW = klines.length > 1 ? klines[1].open_time - klines[0].open_time : 86_400_000
  const barIdxOf = (ts: number) => {
    let lo = 0, hi = klines.length - 1, ans = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (klines[mid].open_time <= ts) { ans = mid; lo = mid + 1 } else hi = mid - 1
    }
    return ans >= 0 && ts - klines[ans].open_time < barW ? ans : -1
  }
  const dotAgg = (curves: AnalyticsCurvePoint[][]) => {
    const m = new Map<number, number>()
    for (const c of curves) for (const p of c) {
      const i = barIdxOf(p.ts)
      if (i >= 0) m.set(i, (m.get(i) ?? 0) + 1)
    }
    return m
  }
  const sceneDots = analytics ? dotAgg(Object.values(analytics.scene).map(b => b.curve)) : new Map<number, number>()
  const shadowDots = analytics ? dotAgg(Object.values(analytics.shadow).map(b => b.curve)) : new Map<number, number>()

  const sceneRows = analytics ? mergeCurves(analytics.scene, SCENE_META) : []
  const shadowRows = analytics ? mergeCurves(analytics.shadow, SHADOW_META) : []
  // 遍历后端返回的键（而非固定 META 名单）：后端动态发现的新版本/新场景也能展示
  const metaFor =
    (meta: Record<string, { label: string; color: string }>) =>
    (key: string, idx: number): { label: string; color: string } =>
      meta[key] ?? { label: key, color: EXTRA_COLORS[idx % EXTRA_COLORS.length] }
  const sceneEntries: [string, { label: string; color: string }][] = analytics
    ? Object.keys(analytics.scene).map((k, i) => [k, metaFor(SCENE_META)(k, i)])
    : []
  const shadowEntries: [string, { label: string; color: string }][] = analytics
    ? Object.keys(analytics.shadow).map((k, i) => [k, metaFor(SHADOW_META)(k, i)])
    : []

  return (
    <div className="space-y-6">
      {/* 控制栏 */}
      <div className="flex items-center gap-4 flex-wrap text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer text-gray-600">
          <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
          自动刷新（60s）
        </label>
        <button
          onClick={load}
          className="px-3 py-1 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition"
        >
          手动刷新
        </button>
        {lastUpdate && <span className="text-gray-400">最后更新 {lastUpdate.toLocaleTimeString('zh-CN')}</span>}
        {err && <span className="text-red-500">{err}</span>}
      </div>

      {/* BTC K线背景 */}
      <Card title={`BTC K线背景（${kinterval === '1d' ? '日线 × 30' : '1小时 × 168'}，UTC 已收盘）`}>
        <div className="flex items-center gap-3 mb-2 text-xs">
          {(['1d', '1h'] as const).map(iv => (
            <button
              key={iv}
              onClick={() => setKinterval(iv)}
              className={`px-2.5 py-0.5 rounded-md border transition ${
                kinterval === iv
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-600 border-gray-300 hover:border-blue-400'
              }`}
            >
              {iv === '1d' ? '日线' : '1小时'}
            </button>
          ))}
          <span className="text-gray-400">
            <span className="text-orange-500">● 场景信号</span> · <span className="text-blue-500">● 影子信号</span>（点大小=当根信号数）
            {pumpTs != null && <span className="ml-2 text-orange-400">| 竖线 = {utcMD(pumpTs)} 大涨分界</span>}
          </span>
        </div>
        {klines.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={klines.map(k => ({ t: k.open_time, close: k.close }))}
              margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                tickFormatter={utcMD} tick={{ fontSize: 10 }} stroke="#9ca3af" />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} stroke="#9ca3af" width={64}
                tickFormatter={(v: number) => v.toLocaleString()} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                formatter={v => [typeof v === 'number' ? v.toLocaleString() : '--', '收盘价']}
              />
              {pumpTs != null && (
                <ReferenceLine x={pumpTs} stroke="#f97316" strokeDasharray="4 3" strokeWidth={1.5}
                  label={{ value: `${utcMD(pumpTs)} 大涨分界`, position: 'insideTopRight', fill: '#f97316', fontSize: 10 }} />
              )}
              <Line dataKey="close" stroke="#374151" dot={false} strokeWidth={1.8} isAnimationActive={false} />
              {Array.from(sceneDots.entries()).map(([i, n]) => (
                <ReferenceDot key={`sc${i}`} x={klines[i].open_time} y={klines[i].close}
                  r={Math.min(3 + n, 6)} fill="#f97316" fillOpacity={0.75} stroke="white" />
              ))}
              {Array.from(shadowDots.entries()).map(([i, n]) => (
                <ReferenceDot key={`sh${i}`} x={klines[i].open_time} y={klines[i].close}
                  r={Math.min(3 + n, 6)} fill="#3b82f6" fillOpacity={0.75} stroke="white" />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center text-gray-400 py-10 text-sm">K 线加载中...</div>
        )}
      </Card>

      {/* 场景信号 */}
      <Card title="场景信号（FakeBreakout 正式信号）：累计胜率 vs 回测冻结基准">
        {analytics && (
          <>
            <div className="overflow-x-auto mb-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500">
                    <th className="py-1 px-2 text-left">场景</th>
                    <th className="py-1 px-2 text-right">n</th>
                    <th className="py-1 px-2 text-right">线上胜率</th>
                    <th className="py-1 px-2 text-right">回测</th>
                    <th className="py-1 px-2 text-right">偏离</th>
                    <th className="py-1 px-2 text-right">平均EV</th>
                    <th className="py-1 px-2 text-right">累计EV</th>
                  </tr>
                </thead>
                <tbody>
                  {sceneEntries.map(([k, m]) => {
                    const s = analytics.scene[k].summary
                    const dev = s.winrate != null && s.bench_winrate != null ? s.winrate - s.bench_winrate : null
                    return (
                      <tr key={k} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-1 px-2 font-medium" style={{ color: m.color }}>{m.label}</td>
                        <td className="py-1 px-2 text-right font-mono">{s.n}</td>
                        <td className="py-1 px-2 text-right font-mono font-bold">{pct1(s.winrate)}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-500">{pct1(s.bench_winrate)}</td>
                        <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-gray-400' : dev >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                        </td>
                        <td className={`py-1 px-2 text-right font-mono ${s.avg_ev == null ? 'text-gray-400' : s.avg_ev >= 0 ? 'text-green-600' : 'text-red-600'}`}>{evFmt(s.avg_ev)}</td>
                        <td className={`py-1 px-2 text-right font-mono ${s.cum_ev == null ? 'text-gray-400' : s.cum_ev >= 0 ? 'text-green-600' : 'text-red-600'}`}>{evFmt(s.cum_ev)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={sceneRows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="ts" type="number" domain={['dataMin', 'dataMax']} scale="time"
                  tickFormatter={utcMD} tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="#9ca3af" width={40}
                  tickFormatter={(v: number) => v + '%'} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                  formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n]}
                />
                {pumpTs != null && <ReferenceLine x={pumpTs} stroke="#f97316" strokeDasharray="4 3" />}
                {sceneEntries.map(([k, m]) => (
                  <Fragment key={k}>
                    <Line dataKey={m.label} stroke={m.color} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                    {analytics.scene[k].summary.bench_winrate != null && (
                      <ReferenceLine y={analytics.scene[k].summary.bench_winrate! * 100}
                        stroke={m.color} strokeDasharray="5 4" strokeOpacity={0.5} />
                    )}
                  </Fragment>
                ))}
              </LineChart>
            </ResponsiveContainer>
            <div className="text-[10px] text-gray-400 mt-1">实线=线上累计胜率，同色虚线=回测冻结基准；x 轴为信号时间（UTC）。</div>
          </>
        )}
      </Card>

      {/* 影子三版本 */}
      <Card title="影子信号（x4 / momentum / contrarian / KREV K线反转）：累计胜率 vs 回测基准 vs 盈亏平衡">
        {analytics && (
          <>
            <div className="overflow-x-auto mb-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-200 text-gray-500">
                    <th className="py-1 px-2 text-left">版本</th>
                    <th className="py-1 px-2 text-right">n</th>
                    <th className="py-1 px-2 text-right">胜率</th>
                    <th className="py-1 px-2 text-right">盈亏平衡</th>
                    <th className="py-1 px-2 text-right">回测</th>
                    <th className="py-1 px-2 text-right">偏离</th>
                    <th className="py-1 px-2 text-right">平均EV</th>
                    <th className="py-1 px-2 text-right">累计EV</th>
                  </tr>
                </thead>
                <tbody>
                  {shadowEntries.map(([k, m]) => {
                    const s = analytics.shadow[k].summary
                    // bench 可空（后端动态发现的新版本无冻结基准），双非空才计算偏离
                    const dev = s.win_rate != null && s.bench_winrate != null ? s.win_rate - s.bench_winrate : null
                    return (
                      <tr key={k} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-1 px-2 font-medium" style={{ color: m.color }} title={s.desc}>{m.label}</td>
                        <td className="py-1 px-2 text-right font-mono">{s.n}</td>
                        <td className="py-1 px-2 text-right font-mono font-bold">{pct1(s.win_rate)}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-500">{pct1(s.avg_breakeven)}</td>
                        <td className="py-1 px-2 text-right font-mono text-gray-500">{pct1(s.bench_winrate)}</td>
                        <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-gray-400' : dev >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                        </td>
                        <td className={`py-1 px-2 text-right font-mono ${s.avg_ev == null ? 'text-gray-400' : s.avg_ev >= 0 ? 'text-green-600' : 'text-red-600'}`}>{evFmt(s.avg_ev)}</td>
                        <td className={`py-1 px-2 text-right font-mono ${s.cum_ev == null ? 'text-gray-400' : s.cum_ev >= 0 ? 'text-green-600' : 'text-red-600'}`}>{evFmt(s.cum_ev)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={shadowRows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="ts" type="number" domain={['dataMin', 'dataMax']} scale="time"
                  tickFormatter={utcMD} tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="#9ca3af" width={40}
                  tickFormatter={(v: number) => v + '%'} />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                  labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                  formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n]}
                />
                {pumpTs != null && <ReferenceLine x={pumpTs} stroke="#f97316" strokeDasharray="4 3" />}
                {shadowEntries.map(([k, m]) => {
                  const s = analytics.shadow[k].summary
                  return (
                    <Fragment key={k}>
                      <Line dataKey={m.label} stroke={m.color} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                      {s.bench_winrate != null && (
                        <ReferenceLine y={s.bench_winrate * 100} stroke={m.color} strokeDasharray="5 4" strokeOpacity={0.5} />
                      )}
                      {s.avg_breakeven != null && (
                        <ReferenceLine y={s.avg_breakeven! * 100} stroke={m.color} strokeDasharray="1 3" strokeOpacity={0.6} />
                      )}
                    </Fragment>
                  )
                })}
              </LineChart>
            </ResponsiveContainer>
            <div className="text-[10px] text-gray-400 mt-1">
              实线=线上累计胜率；同色虚线=回测冻结基准；同色点线=逐版本平均盈亏平衡（x4 含溢价 0.01 口径，其余无溢价）。
            </div>
          </>
        )}
      </Card>

      {/* 周期归因 */}
      <Card title={`周期归因：大涨前 vs 大涨期（${pumpTs != null ? `${utcMD(pumpTs)} 00:00` : '—'} UTC 分界，场景+影子全部信号）`}>
        {analytics && (
          <>
            <div className="flex flex-wrap gap-4 mb-4">
              {[['pre', '大涨前（震荡期）'], ['pump', '大涨期（三根大阳）']].map(([ph, name]) => {
                const g = analytics.regime.phases[ph]
                return (
                  <div key={ph} className="flex-1 min-w-[180px] rounded-lg border border-gray-200 bg-gray-50/60 p-3">
                    <div className="text-xs text-gray-500 mb-1">{name}</div>
                    {g ? (
                      <div className="flex items-baseline gap-3">
                        <span className="text-xl font-bold font-mono text-gray-800">{pct1(g.winrate)}</span>
                        <span className="text-xs text-gray-400">n={g.n} · 赢{g.wins}</span>
                      </div>
                    ) : (
                      <div className="text-sm text-gray-400">无样本</div>
                    )}
                  </div>
                )
              })}
            </div>
            {/* 逐影子版本 × 阶段（对齐审计报告表二归因维度） */}
            {Object.keys(analytics.regime.by_version).length > 0 && (
              <div className="overflow-x-auto mb-4">
                <div className="text-xs text-gray-500 mb-1">逐影子版本拆分</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-200 text-gray-500">
                      <th className="py-1 px-2 text-left">版本 × 阶段</th>
                      <th className="py-1 px-2 text-right">n</th>
                      <th className="py-1 px-2 text-right">胜率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shadowEntries.map(([k, m]) => {
                      const phases = analytics.regime.by_version[k]
                      if (!phases) return null
                      return (
                        <Fragment key={k}>
                          {(['pre', 'pump'] as const).map(ph => {
                            const g = phases[ph]
                            return (
                              <tr key={ph} className="border-b border-gray-100 hover:bg-gray-50">
                                <td className="py-1 px-2" style={{ color: m.color }}>
                                  {m.label}
                                  <span className="text-gray-400 ml-1.5">{ph === 'pre' ? '大涨前' : '大涨期'}</span>
                                </td>
                                <td className="py-1 px-2 text-right font-mono">{g ? g.n : 0}</td>
                                <td className="py-1 px-2 text-right font-mono font-bold">{g ? pct1(g.winrate) : '—'}</td>
                              </tr>
                            )
                          })}
                        </Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {analytics.regime.daily.length > 0 && (
              <>
                <div className="text-xs text-gray-500 mb-1">按天胜率（UTC 日）</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={analytics.regime.daily.map(d => ({
                    date: d.date.slice(5), wr: d.winrate != null ? +(d.winrate * 100).toFixed(1) : 0, n: d.n,
                  }))} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} stroke="#9ca3af" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="#9ca3af" width={40}
                      tickFormatter={(v: number) => v + '%'} />
                    <Tooltip
                      contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e5e7eb' }}
                      formatter={(v, n, item) => n === 'wr'
                        ? [`${v}%（n=${(item?.payload as { n?: number })?.n ?? '-'}）`, '胜率']
                        : [String(v), String(n)]}
                    />
                    <ReferenceLine y={50} stroke="#9ca3af" strokeDasharray="4 4" />
                    <Bar dataKey="wr" fill="#3b82f6" fillOpacity={0.75} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
