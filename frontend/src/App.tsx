import { useState, useEffect, useCallback, useRef, useMemo, useLayoutEffect, useId, Fragment } from 'react'
import { createPortal } from 'react-dom'
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Area, AreaChart, ReferenceLine,
  BarChart, Bar, Legend, LineChart, Line,
  ReferenceArea, ReferenceDot, Cell, ComposedChart,
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
  // 影子检测器运行态（2026-10-10）：采样循环型检测器（gap_crowd/absorption）的
  // 触发/结算计数与 fired 窗口数；null = 未装配（开关关闭或装配失败）
  detectors?: Record<string, Record<string, unknown> | null>
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
    bench_max_entry_price: number | null
    desc: string
    // 带护栏实际数据（过滤掉贴线/超价弃单及白名单外单）：
    guard_price?: number | null
    guarded_n?: number
    guarded_rejected_n?: number
    guarded_fill_rate?: number | null
    guarded_win_rate?: number | null
    guarded_quote_n?: number
    guarded_avg_ev?: number | null
    guarded_cum_ev?: number | null
    guarded_avg_breakeven?: number | null
    // 影子开关（2026-09-04 前端手动下线能力）：下线=停采集+置灰，历史保留
    enabled?: boolean
    // 永久退役（2026-09-04）：后端 shadow_version_gate.RETIRED_VERSIONS 硬闸，
    // 不可逆——开关置灰不可点（toggle API 也会 422 拒），但 bench 基准与历史曲线保留
    retired?: boolean
    // 实盘通道状态（2026-09-06 promote）：version==通道名时后端下发护栏三件套，
    // 「实盘」列按钮据此渲染；null=该影子族未注册实盘通道（置灰不可点）
    live_channel?: {
      enabled: boolean; amount_usdt: number; max_daily_orders: number; max_exec_price: number
    } | null
    collection_mode?: 'RECORD_ONLY'
    execution_mode?: 'SHADOW_ONLY' | 'LIVE_AVAILABLE' | 'LIVE_ACTIVE' | 'LIVE_RETIRED'
    family?: string
    role?: 'PRIMARY' | 'CONTROL' | 'COMPONENT' | 'HYPOTHESIS' | 'STRATEGY'
  }
  curve: AnalyticsCurvePoint[]
  guarded_curve?: AnalyticsCurvePoint[]
}
interface SceneTypeBlock {
  summary: {
    n: number; winrate: number | null; avg_ev: number | null; cum_ev: number | null
    bench_winrate: number | null
    collection_mode?: 'SIGNAL_RECORD'
    execution_mode?: 'RESEARCH_ONLY' | 'LIVE_AVAILABLE' | 'LIVE_ACTIVE'
    family?: 'scene'
    role?: 'STRATEGY'
    live_channel?: string | null
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

// 影子信号详情与实盘对比数据结构
interface ShadowDetailLiveOrder {
  id: number
  order_id: string | null
  status: string
  amount_usdt: number
  average_price: number | null
  price_kind: 'fill' | 'quote' | null
  win: boolean | null
  pnl: number | null
  settle_outcome: string | null
  error_message: string | null
  created_at: string | null
  settled_at: string | null
}

interface ShadowDetailOrderItem {
  window_start: number
  direction: string
  entry_price: number | null
  win: boolean | null
  ev: number | null
  breakeven: number | null
  guarded_pass: boolean
  reject_reason: string | null
  live_order: ShadowDetailLiveOrder | null
}

interface ShadowDetailCurvePoint {
  i: number
  ts: number
  cum_ev_theoretical: number
  cum_ev_guarded: number
  cum_pnl_live: number | null
  win_rate_theoretical: number
  win_rate_guarded: number | null
}

interface ShadowDetailBucket {
  range?: string
  hour_range?: string
  signals: number
  wins: number
  win_rate: number | null
  avg_ev: number | null
  live_orders: number
  live_wins: number
  live_win_rate: number | null
  live_pnl: number | null
}

interface ShadowSignalDetailResponse {
  version: string
  desc: string
  family: string
  role: string
  guard_price: number | null
  bench_winrate: number | null
  summary: {
    total_signals: number
    theoretical_wins: number
    theoretical_win_rate: number | null
    theoretical_avg_ev: number | null
    theoretical_cum_ev: number
    guarded_signals: number
    guarded_wins: number
    guarded_win_rate: number | null
    guarded_pass_rate: number | null
    guarded_avg_ev: number | null
    guarded_cum_ev: number
    rejected_signals: number
    rejected_wins: number
    rejected_losses: number
    rejected_win_rate: number | null
    avoided_loss_ev: number
    live_orders_count: number
    live_filled_count: number
    live_wins: number
    live_win_rate: number | null
    live_total_pnl: number
    live_avg_pnl: number | null
    avg_quote_price: number | null
    avg_live_fill_price: number | null
    slippage_bps: number | null
  }
  curves: ShadowDetailCurvePoint[]
  quote_buckets: ShadowDetailBucket[]
  hour_buckets: ShadowDetailBucket[]
  orders: ShadowDetailOrderItem[]
}

interface ExecutionFunnel {
  theoretical: number
  strategy_eligible: number
  operational_eligible: number
  execution_eligible: number
  submitted: number
  filled: number
  settled: number
  unknown: number
  no_live_mapping: number
}
interface FixedUnitReturn { n: number; sum: number; avg: number | null }
interface ActualExecutionReturn { settled_n: number; stake: number; pnl: number; roi: number | null }
interface ExecutionReturns {
  theoretical_fixed_1u: FixedUnitReturn
  executable_fixed_1u: FixedUnitReturn
  actual: ActualExecutionReturn
}
interface ExecutionGap { n: number; rate: number | null }
interface ExecutionGaps {
  selection: ExecutionGap
  operational: ExecutionGap
  execution: ExecutionGap
  fill: ExecutionGap
  settlement: ExecutionGap
  coverage: ExecutionGap
}
interface ExecutionReasonCount { reason: string; count: number }
interface ExecutionComparisonVersion {
  signal_version: string
  policy_version: string | null
  market_period: string | null
  channel_mapped: boolean
  retired: boolean
  funnel: ExecutionFunnel
  returns: ExecutionReturns
  gaps: ExecutionGaps
  reason_counts: ExecutionReasonCount[]
}
interface ExecutionComparisonCurvePoint {
  timestamp: number
  signal_version: string
  policy_version: string | null
  theoretical_fixed_1u: number | null
  executable_fixed_1u: number | null
  actual_stake: number
  actual_pnl: number | null
  actual_roi: number | null
}
interface ExecutionComparison {
  coverage_matrix?: { signal_version: string; live_frozen_n: number; archive_or_legacy_n: number; capture_capability: string; blocking_reasons: string[] }[]
  forward_evidence?: {
    cohort: string; live_n: number; archive_n: number; independent_windows: number; independent_days: number
    paired_same_window_direction: number; pair_missing: number; blocking_reasons: string[]
    series?: { signal_version: string; policy_version: string; amount: string; n: number; pnl: number; max_drawdown: number; without_top_win_pnl: number; top_win_positive_share: number | null; missing: Record<string, number> }[]
    ladder: Record<string, { n: number; stake: number; pnl: number; stress_1pct: number; stress_3pct: number; missing: number }>
  }
  scope: {
    policy_version: string | null
    from: number | null
    to: number | null
    market_period: string | null
    include_retired: boolean
  }
  funnel: ExecutionFunnel
  returns: ExecutionReturns
  gaps: ExecutionGaps
  reason_counts: ExecutionReasonCount[]
  versions: ExecutionComparisonVersion[]
  curves: ExecutionComparisonCurvePoint[]
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

interface NotifyFieldMeta {
  id: string
  label: string
  hint: string
}
interface NotifyEventMeta {
  label: string
  hint: string
  transports: string[]
  fields: NotifyFieldMeta[]
}
interface NotifyEventConfig {
  wechat?: boolean
  email?: boolean
  fields: Record<string, boolean>
}
interface NotifyGlobalConfig {
  wechat_enabled: boolean
  email_enabled: boolean
  low_balance: { wechat: boolean; fields: Record<string, boolean> }
}
interface NotifyChannelPatch {
  channel: string
  enabled?: boolean
  config?: { events: Record<string, NotifyEventConfig> }
}
interface NotifyChannelRow {
  channel: string
  display_name: string
  family: string
  family_label: string
  market_period: string
  enabled: boolean
  config: { events: Record<string, NotifyEventConfig> }
  overridden: boolean
}
interface NotifyConfigSnapshot {
  catalog: {
    events: Record<string, NotifyEventMeta>
    low_balance_fields: NotifyFieldMeta[]
    families: Record<string, string>
  }
  physical: {
    wechat: { env_enabled: boolean; wxpusher_configured: boolean; wechat_work_configured: boolean; radar_env_enabled: boolean }
    email: { env_enabled: boolean; smtp_configured: boolean; scene_email_env_enabled: boolean }
  }
  global: { config: NotifyGlobalConfig; overridden: boolean }
  channels: NotifyChannelRow[]
  message?: string
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
  triggerDeepLearn: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  runPyClusterDeepLearn: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn/pycluster?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  runCompare: (maxWindows = 100) =>
    authFetch(`/api/sentiment/agent/deep-learn/compare?max_windows=${maxWindows}`, { method: 'POST' }).then(r => r.json()),
  getCompareLive: () =>
    authFetch('/api/sentiment/agent/deep-learn/compare/live').then(r => r.json()),
  getAgentHealth: () => authFetch('/api/agent/health').then(r => r.json()),
  // 信号体检（影子+实盘统一红黄绿灯；下钻；AI 诊断留口）
  getSignalsHealth: (refresh = false) =>
    authFetch(`/api/signals/health${refresh ? '?refresh=true' : ''}`).then(r => r.json()),
  getSignalsHealthDetail: (scope: string, key: string) =>
    authFetch(`/api/signals/health/detail?scope=${scope}&key=${encodeURIComponent(key)}`).then(r => r.json()),
  postSignalsHealthDiagnose: (scope: string, key: string) =>
    authFetch(`/api/signals/health/diagnose?scope=${scope}&key=${encodeURIComponent(key)}`, { method: 'POST' }).then(r => r.json()),
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
  // 实时口径 K 线代理（含当前未收盘 K，不走缓存）：一键下单实时图轮询基线
  getBtcKlinesLive: (interval: string, limit: number) =>
    authFetch(`/api/chart/btc-klines?interval=${interval}&limit=${limit}&include_open=true`).then(r => r.json()),
  getSignalsAnalytics: () => authFetch('/api/signals/analytics?active_only=true').then(r => r.json()),
  getSignalsExecutionComparison: (): Promise<ExecutionComparison> =>
    authFetch('/api/signals/execution-comparison?active_only=true').then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json() as Promise<ExecutionComparison>
    }),
  getShadowSignalDetail: (version: string, limit = 500): Promise<ShadowSignalDetailResponse> =>
    authFetch(`/api/signals/shadow/detail?version=${encodeURIComponent(version)}&limit=${limit}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json()
    }),
  // 影子版本手动下线/上线（2026-09-04）：下线=停采集新信号+面板置灰，历史数据保留
  toggleShadow: (version: string, enabled: boolean) =>
    authFetch('/api/shadow/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version, enabled }),
    }).then(r => r.json()),
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
  getRecentTrades: (limit = 5000) => authFetch(`/api/trades/recent?limit=${limit}`).then(r => r.json()),
  getFundFlow: () => authFetch('/api/trades/fund-flow').then(r => r.json()),
  postTradeTest: (
    amount_usdt: number, prediction: string,
    opts?: {
      market_period?: string
      window_start?: number | null
      order_type?: string
      price_limit?: number | null
      max_exec_price?: number | null
    },
  ) =>
    authFetch('/api/trade/test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        amount_usdt, prediction,
        ...(opts?.market_period ? { market_period: opts.market_period } : {}),
        ...(opts?.window_start != null ? { window_start: opts.window_start } : {}),
        ...(opts?.order_type ? { order_type: opts.order_type } : {}),
        ...(opts?.price_limit != null ? { price_limit: opts.price_limit } : {}),
        ...(opts?.max_exec_price != null ? { max_exec_price: opts.max_exec_price } : {}),
      }),
    }).then(r => r.json()),
  // 未来周期市场（手动下单模态框周期选择条；TTL 30s 服务端缓存）
  getFutureMarkets: (period: string, count = 8, force = false) =>
    authFetch(`/api/prediction/future-markets?period=${period}&count=${count}${force ? '&force=true' : ''}`).then(r => r.json()),
  // 复利下单链：赢则本金+利润滚入下一腿，输即终止
  postCompound: (base_amount: number, direction: string, windows: number[], max_exec_price: number) =>
    authFetch('/api/trade/compound', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base_amount, direction, windows, max_exec_price }),
    }).then(r => r.json()),
  getCompound: () => authFetch('/api/trade/compound').then(r => r.json()),
  stopCompound: (id: number) =>
    authFetch(`/api/trade/compound/${id}/stop`, { method: 'POST' }).then(r => r.json()),
  // 平仓（SELL）：平掉未结算 BUY 持仓
  postClosePosition: (order_id: number) =>
    authFetch('/api/trade/close', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ order_id }),
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
  postLiveChannel: (
    channel: string, enabled: boolean, amountUsdt?: number,
    maxDailyOrders?: number, maxExecPrice?: number, resetMaxExec?: boolean,
  ) =>
    authFetch('/api/live/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        channel, enabled,
        ...(amountUsdt != null ? { amount_usdt: amountUsdt } : {}),
        ...(maxDailyOrders != null ? { max_daily_orders: maxDailyOrders } : {}),
        ...(maxExecPrice != null ? { max_exec_price: maxExecPrice } : {}),
        ...(resetMaxExec ? { reset_max_exec: true } : {}),
      }),
    }).then(r => r.json()),
  getQuotePreview: () => authFetch('/api/prediction/quote-preview').then(r => r.json()),
  // 盈利趋势（2026-08-31）：每实盘通道已结算订单累计盈亏曲线
  getLivePnlCurve: () => authFetch('/api/live/pnl-curve').then(r => r.json()),
  getLivePerformanceSummary: (fromMs = 0, toMs = 0, onlyEnabled = false): Promise<LivePerformanceSummary> =>
    authFetch(`/api/live/performance-summary?from_ms=${fromMs}&to_ms=${toMs}&only_enabled=${onlyEnabled}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json() as Promise<LivePerformanceSummary>
    }),
  getLiveChannelDiagnostics: (channel: string, segmentBy: DiagnosticSegmentBy): Promise<LiveChannelDiagnostics> =>
    authFetch(`/api/live/channel-diagnostics?channel=${encodeURIComponent(channel)}&segment_by=${segmentBy}`).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return r.json() as Promise<LiveChannelDiagnostics>
    }),
  postSyncBinance: () => authFetch('/api/trades/sync-binance', { method: 'POST' }).then(r => r.json()),
  // 奖金领取（2026-08-23）：可领查询 + batch-redeem
  getRedeemable: () => authFetch('/api/prediction/redeemable').then(r => r.json()),
  postRedeem: (tokenIds?: string[]) =>
    authFetch('/api/prediction/redeem', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(tokenIds && tokenIds.length > 0 ? { token_ids: tokenIds } : {}),
    }).then(r => r.json()),
  // 微信通知连通性测试 (2026-09-10)
  postTestWechatNotify: () => authFetch('/api/notify/test-wechat', { method: 'POST' }).then(r => r.json()),
  postTestEmailNotify: () => authFetch('/api/notify/test-email', { method: 'POST' }).then(r => r.json()),
  getNotifyConfig: () => authFetch('/api/notify/config').then(r => r.json()),
  putNotifyConfig: (payload: { global_config?: NotifyGlobalConfig; channels?: NotifyChannelPatch[] }) =>
    authFetch('/api/notify/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json()),
  resetNotifyConfig: (channel?: string | null) =>
    authFetch('/api/notify/config/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(channel ? { channel } : {}),
    }).then(r => r.json()),
}

// ============================================================
// 公共组件
// ============================================================

/* ==========================================================================
   StatusDot — DESIGN.md 规格
   - 10px 正圆（w-2.5 h-2.5 rounded-full），原子几何元素
   - ok=绿 / non-ok=红，映射冻结（Semantic Freeze Rule）
   ========================================================================== */
function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${ok ? 'bg-positive' : 'bg-negative'}`} />
}

/* ==========================================================================
   DirectionBadge — DESIGN.md 规格
   - pill + 语义色浅底 + 同色字，无边框
   - 箭头与文字必须保留（Semantic Freeze Rule：语义色不能是唯一信息通道）
   ========================================================================== */
function DirectionBadge({ direction }: { direction: string }) {
  return (
    <span className={`ds-badge ${
      direction === 'UP'
        ? 'ds-badge-up'
        : direction === 'DOWN'
        ? 'ds-badge-down'
        : 'ds-badge-neutral'
    }`}>
      {direction === 'UP' ? '↑ 看涨' : direction === 'DOWN' ? '↓ 看跌' : '⊘ 不交易'}
    </span>
  )
}

/* ==========================================================================
   StatusBadge — DESIGN.md 规格
   - pill + 语义色浅底 + 同色字，无边框
   - ACTIVE=绿 / EVOLVING=琥珀 / RETIRED=中性
   ========================================================================== */
function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`ds-badge ${
      status === 'ACTIVE'
        ? 'ds-badge-up'
        : status === 'EVOLVING'
        ? 'ds-badge-warn'
        : 'ds-badge-neutral'
    }`}>
      {status}
    </span>
  )
}

/* ==========================================================================
   ChangeTypeBadge — DESIGN.md 规格
   - pill + 语义色浅底 + 同色字，无边框
   - CREATE=靛蓝 / UPDATE=琥珀 / RETIRE=中性
   ========================================================================== */
function ChangeTypeBadge({ type }: { type: string }) {
  return (
    <span className={`ds-badge ${
      type === 'CREATE'
        ? 'ds-badge-indigo'
        : type === 'UPDATE'
        ? 'ds-badge-warn'
        : 'ds-badge-neutral'
    }`}>
      {type}
    </span>
  )
}

function DiscoveryMethodBadge({ method }: { method?: string }) {
  const meta: Record<string, { label: string; cls: string }> = {
    LLM_DEEP: { label: 'LLM', cls: 'ds-badge-violet' },
    PY_CLUSTER: { label: 'PY聚类', cls: 'ds-badge-teal' },
    LEGACY: { label: '存量', cls: 'ds-badge-neutral' },
  }
  const m = meta[method || 'LEGACY'] || meta.LEGACY
  return <span className={`ds-badge ${m.cls}`}>{m.label}</span>
}

// 问号 hover 提示（信号说明等）
// 气泡经 portal 挂到 body、position:fixed 定位：不再被 overflow 容器（表格滚动区/卡片）裁切或压层。
// 定位规则：默认在问号上方右对齐；上方放不下翻到下方；水平方向夹在视口内（留 8px 边距）。
// tabIndex + focus 保留：键盘必须能触发（AGENTS.md 既有行为，不得回退）。
const HINT_GAP = 6
const HINT_MARGIN = 8
function HelpHint({ text }: { text: string }) {
  const triggerRef = useRef<HTMLSpanElement>(null)
  const bubbleRef = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const id = useId()

  const place = useCallback(() => {
    const trigger = triggerRef.current, bubble = bubbleRef.current
    if (!trigger || !bubble) return
    const r = trigger.getBoundingClientRect()
    const w = bubble.offsetWidth, h = bubble.offsetHeight
    const vw = window.innerWidth, vh = window.innerHeight
    // 右对齐问号，再夹进视口
    const left = Math.min(Math.max(r.right - w, HINT_MARGIN), vw - w - HINT_MARGIN)
    const above = r.top - HINT_GAP - h
    const below = r.bottom + HINT_GAP
    const top = above >= HINT_MARGIN || below + h > vh - HINT_MARGIN
      ? Math.max(above, HINT_MARGIN)
      : below
    setPos({ top, left: Math.max(left, HINT_MARGIN) })
  }, [])

  // 打开后先隐形渲染测尺寸，再定位（避免首帧闪到错误位置）
  useLayoutEffect(() => {
    if (!open) { setPos(null); return }
    place()
    // 滚动（含任意祖先滚动容器，capture）/缩放时跟随重算
    window.addEventListener('scroll', place, true)
    window.addEventListener('resize', place)
    return () => {
      window.removeEventListener('scroll', place, true)
      window.removeEventListener('resize', place)
    }
  }, [open, place, text])

  return (
    <span
      ref={triggerRef}
      className="ds-hint align-middle"
      tabIndex={0}
      aria-describedby={open ? id : undefined}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onKeyDown={e => { if (e.key === 'Escape') setOpen(false) }}
    >
      <span className="ds-hint-trigger" aria-hidden="true">?</span>
      {open && createPortal(
        <span
          ref={bubbleRef}
          id={id}
          role="tooltip"
          className="ds-hint-bubble ds-hint-bubble-floating"
          style={pos ? { top: pos.top, left: pos.left } : { top: 0, left: 0, visibility: 'hidden' }}
        >
          {text}
        </span>,
        document.body,
      )}
    </span>
  )
}

// 线上信号通道说明（口径源：services/live_channels.py 注册表 + quote_edge_detector 冻结规则）
// 实盘资格不再放在本地 kind 字段；运行态唯一依据是后端 live_channel / execution_mode。
// 通道运行状态（开关/金额/日限/护栏/今日成交）来自后端 multi_live_trader.status_async()。
// 2026-09-04 退役 8 通道（retired）：已从 live_channels.LIVE_CHANNELS 与
// shadow_version_gate.RETIRED_VERSIONS 移除，不再开火/不再采集。条目必须保留：
// 历史订单行仍带这些 signal_version，删了会让订单表退回成裸版本串。
const SIGNAL_INFO: Record<string, { name: string; desc: string; retired?: boolean }> = {
  quote_contrarian_v1: {
    name: '报价反向（B格逆势）', retired: true,
    desc: '【2026-09-04 已退役：被 quote_contrarian_v2 严格支配（v2 = v1 触发集纯子集 + BTC 门禁）】5 分钟窗口开始后 45~60 秒内，DOWN token 报价首次跌入 [0.15, 0.25)（明显便宜）时买入 DOWN。低胜率高赔付：回测胜率 24%、EV +0.155（赢一次约赚 4 倍）。通道护栏 0.28（区间上界+0.03），每窗至多一单。',
  },
  quote_momentum_v1: {
    name: '报价动量（A格顺势）', retired: true,
    desc: '【2026-09-04 已退役：「深折价顺势」假设被前向数据证伪】5 分钟窗口 90~120 秒内，DOWN token 报价首次进入 [0.69, 0.75)（强势确认）时押 DOWN。回测胜率 79.9%、EV +0.097。通道护栏 0.78，每窗至多一单。',
  },
  quote_contrarian_v2: {
    name: '报价反向·门禁版',
    desc: 'v1 区间 + BTC 价格门禁：触发时点 BTC 未高于窗口开盘 ≥0.10%（只接「假冲高」，归因显示平盘窗贡献 86% 利润）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.28。注：触发区间仍引用已退役 v1 的冻结回测口径（QUOTE_EDGE_RULES 是口径事实源，整体保留）。',
  },
  quote_contrarian_z_high_v1: {
    name: '报价反向·标准化早涨高纯度',
    desc: '【仅记录，不实盘】quote_contrarian_v2 的冻结子集：开窗后第一根完整 1m 收盘的标准化涨幅 zret1_60 ≤ 0.8618732766463684，且开窗前 60 根完整 1m 波动 rv60 ≥ 0.0008177729241675949。依赖 T+60s 收盘后判定，不是 T+45s 可交易信号；缺 K 线、断棒或零波动均不记录。720 天研究 n=1,753、胜率 36.74%、平均 EV +0.482，待前向验证。',
  },
  quote_contrarian_z_mid_v1: {
    name: '报价反向·标准化早涨中频',
    desc: '【仅记录，不实盘】quote_contrarian_v2 的冻结子集：开窗后第一根完整 1m 收盘的标准化涨幅 zret1_60 ≤ 1.4696111971489016。依赖 T+60s 收盘后判定，不是 T+45s 可交易信号；缺 K 线、断棒或零波动均不记录。720 天研究 n=3,908、胜率 29.15%、平均 EV +0.201，待前向验证。',
  },
  quote_momentum_v2: {
    name: '报价动量·门禁版', retired: true,
    desc: '【2026-09-04 已退役：随 momentum 族整体下线】v1 区间 + BTC 价格门禁：触发时点 BTC 已低于窗口开盘 ≥0.10%（剔「假恐慌」，真跌段胜率 85% vs 假恐慌段 40%）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.78。',
  },
  quote_momentum_v3: {
    name: '报价动量·非连涨门禁版', retired: true,
    desc: '【2026-09-04 已退役：修正未来函数后门禁效应≈0（+3.8pp，CI 重叠），不值得占实盘额度】v1 区间（t90~120s q∈[0.69,0.75)）∩ 触发时点末收 15m 非连涨（close≤前根，严格 ex-ante）。回测 80.2% vs 连涨 76.4%。与 v1/v2 同窗互斥（至多一单成交），通道护栏 0.78。',
  },
  quote_contrarian_v3a: {
    name: '报价反向·交替环境版', retired: true,
    desc: '【2026-09-04 已退役：环境门禁未兑现，被 contrarian_v2 支配】contrarian v1 区间 + v2 价格门禁 + 环境门禁：前窗结算 DOWN（交替环境：前窗跌+本窗涨=V 反弹假冲高）。真实回测 n=85 胜率 31.8%、EV +0.528。实盘已解锁（前窗 outcome 异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  quote_contrarian_v3b: {
    name: '报价反向·日高回落版', retired: true,
    desc: '【2026-09-04 已退役：同 v3a，门禁未兑现】v3a + 触发时点 BTC 距当日高点回落 ≥0.30%（含边界，震荡日冲高更易衰竭）。真实回测 n=65 胜率 33.8%、EV +0.646（单笔 EV 最优）。实盘已解锁（日高异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  quote_contrarian_v4: {
    name: '报价反向·下跌周期版', retired: true,
    desc: '【2026-09-04 已退役：regime 门禁未兑现，被 contrarian_v2 支配】contrarian v1 区间 + regime 门禁：触发时点过去 24h BTC 收益 ≤ −1.0%（含边界，只在下跌周期开火）。62 天真实订单簿回测：down 段 n=413 胜率 30.3%、EV +0.372（CI 下界过盈亏平衡线），up/range 段 EV≈0——正边际集中在下跌周期。5m K 线严格 ex-ante 口径（影子/实盘同源，缺失保守弃单），每窗至多一单，通道护栏 0.28。',
  },
  x4_v1: {
    name: '情绪错位（收阳押次窗DOWN）', retired: true,
    desc: '【2026-09-04 已退役：被 x4_v2 严格支配（v2 = v1 触发集纯子集 + 平静市门禁）】本窗收阳但 15m 市场收尾情绪 ≤40 的错位 → 次窗 +150s 决策点押 DOWN（回测合并胜率 63.5%、EV +0.254）。实盘已解锁：PENDING 信号轮询→决策点下单，护栏 0.45，错过决策点不追单。',
  },
  x4_v2: {
    name: '情绪错位·平静市门禁版',
    desc: 'x4_v1 + 平静市门禁（回测胜率 45.3%，仅平静市况触发）。实盘已解锁：同 x4_v1 决策点机制，护栏 0.50，错过决策点不追单。',
  },
  x4_v3: {
    name: '情绪错位·趋势过滤版',
    desc: 'x4_v2 门禁 + 双趋势门禁：触发时点 BTC 过去 4h 跌超 1%（急跌不接刀）或过去 24h 涨超 2%（过热不追空）均不触发，K 线数据缺失同不触发；另有下单层入场价白名单 [0,0.2)∪[0.3,0.4)（决策点成交均价超带弃单，护栏 0.50 兜底）。与 x4_v2 同窗并行对比（各自独立下单），实盘默认关闭需手动开启。回测 C3 保留段 40 笔胜率 42.5%、EV +1.629。',
  },
  scene_bull_exhaust: {
    name: '场景S1 多头耗尽（押DOWN）',
    desc: '15m 周期刺破 4h 阻力 + 光头阳收盘确认 → 次周期开盘押 DOWN（真 OOS 胜率 64.4%，盈亏平衡 0.63）。实盘已解锁：15m 市场次周期开盘下单，护栏 0.60。',
  },
  scene_bull_exhaust_confirm: {
    name: '场景S5 确认入场（押DOWN）',
    desc: 'S1 信号 +5min 确认（次周期第 1 根 5m K 收盘 < 开盘）才买 DOWN（确认组胜率 78.5%，盈亏平衡 0.77）。实盘已解锁：确认时刻 15m 市场下单，护栏 0.75。',
  },
  scene_bear_exhaust: {
    name: '场景S2 空头耗尽（押UP）',
    desc: '15m 周期跌破 4h 支撑 + 收阴 + 放量 → 次周期开盘押 UP（胜率 53.6%，盈亏平衡 0.525）。护栏 0.65。',
  },
  scene_bear_exhaust_opt_v1: {
    name: '场景S2 空头耗尽·温和放量非低位（押UP）',
    desc: '复用原 S2，再要求 2≤量比<4 且信号收盘不在最近14日区间下方33%（位置≥0.33）→ 次周期开盘押 UP。720d n=975、胜率58.97%；影子持续采集，实盘默认关闭，护栏0.57；与原S2同窗互斥。',
  },
  scene_momentum_fade: {
    name: '场景S4 动量衰竭（押DOWN）',
    desc: '连阳 ≥3 根 + 光头阳的动量衰竭 → 次周期开盘押 DOWN（胜率 55.4%，盈亏平衡 0.54）。实盘已解锁：15m 市场次周期开盘下单，护栏 0.55。2026-10-04 起跳过续发：上一根 15m 本身已触发 S4（上一窗押 DOWN 已输、连阳延长）则本根不下单也不记录——720 天续发 n=180 胜率 50.6%，真实盘口近 3 个月续发胜率 32%。',
  },
  s4_delay60_v1: {
    name: 'S4延迟入场·首分钟回落（押DOWN）',
    desc: '正式S4命中后，等目标窗第1根1m收盘<窗口开盘（BTC已回落）才按+60s真实DOWN报价入场，押15m DOWN。720d重放 n=837、胜率66.2%；predict.fun真实盘口两段均EV +0.119/+0.107。同名实盘通道已注册（默认关闭），护栏0.65；与S4开盘单同窗同向叠加敞口。',
  },
  s4_hiconf_v1: {
    name: 'S4高信心档·连阳≥5（押DOWN）',
    desc: '【仅记录，不实盘】连阳 ≥5 根（含信号K）且收盘位置 ≥0.9，允许 4h 破位周期 → 次周期押 DOWN。720 天重放 n=399、胜率 63.2%；真实盘口两段均EV +0.119 / +0.147。约一半与 S1 同窗重叠，上实盘前须并入同窗互斥组。',
  },
  s1_early2_v1: {
    name: 'S1早确认·+2min回落（押DOWN）',
    desc: '正式 S1 命中后，等目标窗第 2 根 1m 收盘：BTC 低于窗口开盘才按 +2min 真实 DOWN 报价入场，否则放弃。是 S5（+5min 确认）的早确认版，买得更便宜（约 0.67 对 0.71）、胜率低约 3 个百分点。真实盘口两段均EV +0.118 / +0.099（S5 为 +0.015 / +0.080），720 天 K 线胜率 71.7%。与 S5 约 76% 重叠。影子持续采集；同名实盘通道已注册，默认关闭，护栏 0.75；替换 S5 时请先关 S5，避免同窗叠加。',
  },
  s1_dip45_v1: {
    name: 'S1低吸·DOWN≤0.45（押DOWN）',
    desc: '正式 S1 命中后，在 +1/+2/+3/+5/+8 分钟首次看到 DOWN 报价 ≤0.45（影子按缓存价 ≤0.42 触发，缓存价比真实成交低约 0.03）即入场。真实盘口两段均EV +0.097 / +0.199（n57 / n123），同规则用在随机窗口上为 −0.074 / +0.011，说明收益来自 S1 信号本身。与 S1 开盘单同向叠加敞口，金额宜小于 S1。影子持续采集；同名实盘通道已注册，默认关闭，护栏 0.47。',
  },
  s1_dyn_sq_v1: {
    name: 'S1 动态轧空路由影子（押DOWN）',
    desc: '父S1命中时，用目标窗之前30天已完成5m K构造滚动1小时收益分布。当前ret1h低于q90按开盘近端报价押DOWN；达到或超过q90视为轧空，只在首根5m收阴确认后按+5min报价押DOWN，否则整窗跳过。阈值随行情波动自适应；720d严格时序重放 n=1782、胜率64.9%，原S1为58.3%。影子持续采集；同名小额实盘通道已注册，默认关闭，护栏0.63，与原S1同窗互斥。',
  },
  s5_deep_z20_v1: {
    name: 'S5深档·深回落门禁版',
    desc: 'S1 + 5min 回落确认且 z5≤−20bp 的深回落子集 → +5min 确认即押次周期 15m DOWN（回测 ~91.3%，EV 偏乐观含机械成分；盈亏平衡 ~86.7%）。小金额实盘前向验证，护栏 0.88；与 S5 确认通道同窗互斥（至多一单成交）。',
  },
  // ---- 2026-09-06 影子 promote 5 通道（口径源：services/live_channels.py + 各检测器冻结研究口径）----
  s2_cond_t4_v1: {
    name: 'S2条件单·t4价<开（押UP）',
    desc: '实盘 S2（bear_exhaust，破 4h 支撑+收阴+放量）派生：次周期 t=4(+240s) 1m 收盘 < 周期开盘（全深度回落）→ 当刻买 UP 押次周期 15m 收阳（价跌时 UP 变便宜，低买 UP 的正 EV 来自入场价而非胜率，真实 EV 由生产报价前向现算）。720d 触发 1069/2176（49.1%，1.48/天），价-only 胜率 38.9%；通道护栏 0.38（38.9%×0.98 平衡价略下方）。与 t5 剔深同窗互斥（至多一单成交）。',
  },
  s2_cond_t5d_v1: {
    name: 'S2条件单·t5剔深（押UP）',
    desc: '实盘 S2（bear_exhaust）派生：次周期 t=5(+300s) 0<回落<15bp（中度回落剔深，排除已深跌的接刀窗）→ 当刻买 UP 押次周期 15m 收阳。720d 触发 643/2176（29.5%，0.89/天），价-only 胜率 44.8%；通道护栏 0.44（44.8%×0.98）。与 t4 价<开同窗互斥（至多一单成交）。',
  },
  nb_zschamp_15m_v1: {
    name: 'nextbar 15m冠军·深超卖反转（押UP）',
    desc: 'zscore_10≤−1.651 ∧ zscore_5≤−1.538 ∧ 3根跌幅≤−0.395% 的深超卖组合 → 押次根15m收阳UP。720d n≈2006、胜率58.92%，holdout n=368、胜率61.96%；生产影子 n=28、胜率67.9%，报价样本平均EV +0.130但CI仍跨0。默认实盘关闭，护栏0.57，仅目标根开盘后90秒内的新鲜命中派单。',
  },
  nb_smaslope_5m_v1: {
    name: 'nextbar 5m动量误定价（押UP）',
    desc: '5m sma_slope_atr_5 ≥ 1.66（动量急升）→ 押次根 5m 收阳 UP。研究自评：720d 次根收阳仅 47.43%（长样本反指），edge 依赖 7-8 月 regime——record-only 科学仪器非背书，影子期前向验证「动量误定价」是否持续。护栏 0.46（47.43%×0.98 下方）刻意只放行 UP 便宜窗，成交少是保命设计。仅目标根开盘 ≤90s 内的新鲜命中下单（冷启动回补天然排除）。',
  },
  krev_a_v1: {
    name: 'KREV-A 反转（押UP）',
    desc: '15m 距前低≤−0.09 ATR + 5 根高效率阴跌 + 周期内 3 根 5m 子阴齐跌 → 押次根 15m 收阳 UP。冻结 holdout n=137、胜率 64.2%；默认实盘关闭，护栏 0.629（64.2%×0.98）。仅目标根开盘后 90 秒内的新鲜命中派单，冷启动回补不追单。',
  },
  krev_b_v1: {
    name: 'KREV-B 反转（押UP）',
    desc: '15m 区间贴底 + 5 根高效率阴跌 + 周期内 3 根 5m 子阴齐跌 → 押次根 15m 收阳 UP。冻结 holdout n=134、胜率 63.4%；默认实盘关闭，护栏 0.621（63.4%×0.98）。仅目标根开盘后 90 秒内的新鲜命中派单，冷启动回补不追单。',
  },
  rev_p1_v1: {
    name: 'P1 连跌弱阴反转（押UP）',
    desc: '15m 连跌至少 4 根 + 弱阴贴最低 + 量比 [1.0,1.5) → 押次根 15m 收阳 UP。冻结回测胜率 62.0%；默认实盘关闭，护栏 0.608（62.0%×0.98）。仅目标根开盘后 90 秒内的新鲜命中派单，冷启动回补不追单。',
  },
  rev_p2_v1: {
    name: 'P2 连涨弱阳反转（押DOWN）',
    desc: '15m 连涨至少 5 根 + 弱阳贴最高 → 押次根 15m 收阴 DOWN。冻结回测胜率 62.4%；默认实盘关闭，护栏 0.612（62.4%×0.98）。仅目标根开盘后 90 秒内的新鲜命中派单，冷启动回补不追单。',
  },
  crv_hi_brk20_15m_v1: {
    name: 'CRV高位突破20高反转（押DOWN）',
    desc: '15m 阳线收盘突破 20 根高且处于 100 根区间顶部（pos100>0.90）→ 押次根 15m 收阴 DOWN（末段突破情绪耗竭）。2026-10-04 研究 720D 预注册穷举 + 90D 盲测：n=212 胜率 63.7%（Wilson95% 下界 57.0%），13/13 月为正，同族最强。默认实盘关闭，护栏 0.62（63.7%×0.98）；仅目标根开盘后 90 秒内的新鲜命中派单。',
  },
  crv_brk20_15m_v1: {
    name: 'CRV突破20高反转（押DOWN）',
    desc: '15m 阳线收盘突破 20 根高 → 押次根 15m 收阴 DOWN（假突破回归）。盲测 n=458 胜率 58.5%（下界 54.0%），360D 累计EV +120 单位注金、13/13 月为正。默认实盘关闭，护栏 0.57；与 CRV 另两通道同窗互斥（至多一单成交）。',
  },
  crv_brk50_hi_15m_v1: {
    name: 'CRV突破50高∧高位反转（押DOWN）',
    desc: '15m 阳线收盘突破 50 根高且 pos100>0.75 → 押次根 15m 收阴 DOWN。盲测 n=226 胜率 60.6%（下界 54.1%）。默认实盘关闭，护栏 0.59；与 CRV 另两通道同窗互斥。',
  },
  crv_brk20_5m_v1: {
    name: 'CRV 5m突破20高反转（押DOWN）',
    desc: '5m 阳线收盘突破 20 根高 → 押次根 5m 收阴 DOWN。盲测 n=1552 胜率 54.5%（下界 52.0%，5m 唯一强档），但单笔 EV 薄（+0.025）、频率高（约 16 信号/天）。默认实盘关闭，护栏 0.53 刻意只放行便宜 DOWN。',
  },
  crv_dnex_15m_v1: {
    name: 'CRV三因子·跌破出清（押UP·探索级）',
    desc: '15m 跌破 20 根低 ∧ 贴 20 低（距前低<0.25×ATR）∧ 实体>ATR → 押次根 15m 收阳 UP（弱手出清反弹）。发现段 61.4%/验证段 60.2%，未过盲测（探索级）：仅影子前向验证 ≥8 周后再评估，不注册实盘。',
  },
  crv_os_st4_5m_v1: {
    name: 'CRV三因子·超卖连阴（押UP·探索级）',
    desc: '5m RSI14<30 ∧ 恰好连续 4 根阴线 ∧ 跌破 20 根低 → 押次根 5m 收阳 UP。发现段 60.6%/验证段 52.5%，未过盲测（探索级）：仅影子，不注册实盘。',
  },
  crv_hi_z3_15m_v1: {
    name: 'CRV三因子·高位z离群连3阳（押DOWN·观察级）',
    desc: '15m 阳线 ∧ pos100>0.90 ∧ z20>+2（收盘高出 20 根均值 2 个标准差）∧ 恰好连续 3 根阳线 → 押次根 15m 收阴 DOWN（动量离群耗竭）。AMENDMENT-2 全枚举新发现：发现段 n=190 胜率 62.1%（Wilson95% 下界 55.0%，全场三因子最高）/ 验证段 n=74 67.6%，近 180/90/30 天 58.7%/56.8%/56.3% 不衰减。未过盲测（观察级）：仅影子前向验证 ≥8 周，不注册实盘。',
  },
  brkrv_brk8h_15m_v1: {
    name: 'BRK假突破8h高回归·15m（押DOWN）',
    desc: '15m 收盘突破前 32 根（8 小时）全部高点 → 押次根 15m 收阴 DOWN（假突破回归，2026-10-05 研究三段训练→验证→盲测全部一致的最强形态家族）。训练 55.2% → 验证 58.8%（Wilson95% 下界 56.0%）→ 盲测 55.7%（n=79）；720D n=2768 胜率 56.8%，90D/30D 59.5%/55.6% 全正。默认实盘关闭，护栏 0.54（盲测 55.7%×0.98）；仅目标根开盘后 90 秒内的新鲜命中派单，与 CRV 15m 突破族同窗互斥。',
  },
  brkrv_brk8h_5m_v1: {
    name: 'BRK假突破8h高回归·5m（押DOWN）',
    desc: '5m 收盘突破前 96 根（8 小时）全部高点 → 押次根 5m 收阴 DOWN。K 线三段：训练 56.4% → 验证 58.1% → 盲测 57.6%（n=144）；盲测市场结算口径（NOISE 计输）56.9%——5m 族唯一经济口径为正的强档。⚠ 横盘 regime（如 2026-07 市场 NOISE 率 50%）会被 NOISE 结算摧毁经济 EV，开启前须确认近期 NOISE 率 <5%。默认实盘关闭，护栏 0.55（56.9%×0.98）；仅目标根开盘后 90 秒内的新鲜命中派单，与 CRV 5m 突破通道同窗互斥。',
  },
  absorption_follow_td120_v1: {
    name: '吸收跟随·TD120（随BTC方向）',
    desc: '5m 窗内 BTC 明显位移而 UP 报价欠跟随（粘滞=知情者顶住人群吸筹的脚印）→ 跟随 BTC 位移方向下注（btc 涨押 UP/跌押 DOWN）。窗开 120s 双快照判定；滚动 14 天标定（弹性 k/位移门 p50/欠反应门 p80，严格 ex-ante，与影子同源）。真实价复核 RECENT EV +0.065 CI[+0.006,+0.123]（胜率 79.8%）；护栏 0.78（79.8%×0.98 下方）。与 TD150 同窗互斥（至多一单成交）。',
  },
  absorption_follow_td150_v1: {
    name: '吸收跟随·TD150（随BTC方向）',
    desc: '同 TD120 机制，TD=窗开 150s（多给 30s 让报价反应，触发更少、欠反应更纯）。真实价复核 RECENT EV +0.105 CI[+0.050,+0.165]（胜率 88.5%）；护栏 0.86（88.5%×0.98 下方）。与 TD120 同窗互斥（至多一单成交）。',
  },
  gap_crowd_5m_v1: {
    name: '报价落后人群·5m后半窗（已证伪退役）',
    desc: '⛔ 2026-10-10 同日证伪退役：原假设「窗后半段预测市场报价比人群倾向便宜 ≥5pp → 买 UP」的触发条件只在 up_price/down_price 快照不一致的瞬间成立（盲测 948 事件 100% 报价和 <0.95、中位 0.91；全样本仅 1.03% 不一致、中位和 1.000），触发价是滞后/偏低值不可成交（生产首单记录价 0.40 vs 实际报价 0.62，被护栏弃单）。用一致化入场价（1−down_price，均值 0.564）重算 EV：+0.036 → −0.072；只留一致样本则 0 事件。原盲测 n=947 胜率 51.4% 系伪迹。通道与影子均已退役停用，历史 1 条信号保留供审计。',
  },
  gap_crowd_15m_v1: {
    name: '报价落后人群·15m后半窗（已证伪退役）',
    desc: '⛔ 同上（15m 版）：盲测 250 事件同样 100% 报价和 <0.95，一致化入场价重算 EV +0.077 → −0.023。原 n=250 胜率 57.6% 系同一伪迹，勿启用。',
  },
  firsthit_down_v1: {
    name: '首触G0基底（押DOWN）',
    desc: '5m 窗内 DOWN 报价首次进入 (0.005,0.1] → 买 DOWN；实时重放本窗完整历史，只认真实第一触。默认关闭；开启后每通道每窗至多一单。G0/G1/G3 为用户确认的独立下单通道，同窗三门全中且全开启时最多 3 单。执行价护栏 0.08，实际成交均价高于护栏弃单，不追价。',
  },
  firsthit_down_body_v1: {
    name: '首触G1小实体（押DOWN）',
    desc: 'G0 + body_r≤0.35（首触前 BTC 路径归一实体小，震荡而非单边跑出去）。默认关闭；与 G0/G3 独立下单，不做跨版本互斥，每通道每窗至多一单。执行价护栏 0.12，实际成交均价高于护栏弃单。confirm 段已 burned，护栏非前向胜率背书。',
  },
  firsthit_down_chg_v1: {
    name: '首触G3偏离v1（押DOWN）',
    desc: '【已退役：已升级为v2双波峰版】历史v1口径：G0+chg≤+2.82bp，因全窗无差别采集包含末段失血单且历史实盘存在LIMIT逆向选择，已归档封存供对账。',
    retired: true,
  },
  firsthit_down_chg_v2: {
    name: '首触G3偏离v2·双波峰版（押DOWN）',
    desc: '【2026-10-06 升级从0起跑】信号门 G0+chg≤+2.82bp；实盘与影子完全同源：限定两个正 EV 波峰 触发 105~120s（早峰 13.8%/EV+0.607）与 210~240s（主峰 10.1%/EV+0.265）× 浅价 q≥0.07（4549 笔前向事件 15S 分桶验证）；坚决避开 240s+ 衰竭区与 285s+ 归零深渊。MARKET 市价单，下单时刻距窗末剩余 ≥60s 弃单，锁定首触时刻 BTC 特征杜绝时点漂移。默认关闭；护栏=触发价×1.03 动态护栏。',
  },
  firsthit_down_chg_early180_v1: {
    name: '首触G3前3分钟研究版（押DOWN）',
    desc: '实际决策时间<180秒且chg≤+2.82bp；使用决策点真实DOWN报价。前向影子研究，实盘默认关闭；与K10 early高度重叠，不应视作独立暴露。',
  },
  firsthit_down_k10_early180_v1: {
    name: '首触K10前3分钟研究版（押DOWN）',
    desc: '真实首触前缀<180秒，满足G3与标准化结算距离z≤1.85567；前序完整5m K线现算。前向影子研究，实盘默认关闭；与G3 early高度重叠。',
  },
  process_recovery_down_v1: {
    name: '过程恢复双确认（押DOWN）',
    desc: '30–180秒过程段首次DOWN报价≤0.20；BTC从不利极值回收≥35%且DOWN报价回升≥0.02时确认，确认须早于180秒、价格在(0.15,0.35]。30秒前低价不排除；不是全窗首触。后验候选，影子默认采集，实盘默认关闭；执行价达到0.35即弃单。',
  },
  firsthit_down_k10_v1: {
    name: '首触K10标准化距离（押DOWN）',
    desc: 'G3基础上，用首触时BTC相对开盘的带符号距离除以前12根完整5m波动按剩余时间缩放的尺度，只保留z≤1.85567。严格锁定真实首触，不用首触后的BTC路径。影子默认采集，实盘默认关闭。',
  },
  firsthit_down_k10_profit_v1: {
    name: '首触K10盈利过滤版（押DOWN）',
    desc: 'K10基础上，首触距窗口结束≤60秒时无条件剔除；更早的首触中，再剔除“前序平方收益扩张且隔离后的前序K线上影≥50%”。末段门含60秒边界；影子默认采集，实盘默认关闭。',
  },
  firsthit_down_g4_v1: {
    name: '首触G4交互门（押DOWN）',
    desc: 'G0 + 小实体(body_r≤0.35) ∧ 价格偏离(chg≤+2.82bp)，即 G1∩G3。冻结扫描 calib n=214 P=12.6% EV+1.03 CI[+0.08,+2.22]；confirm n=86 P=23.3% EV+1.85 CI[+0.35,+3.44]；BH-FDR q=0.063 STRICT_PASS。执行价护栏 0.12。⚠️非独立暴露：G4⊂G1 且 G4⊂G3，同窗三门全中即同一注重复下注。⚠️功效偏紧：calib EV(+1.03) 与前向 CI 半宽(≈1.08) 接近，前向通过概率约 50%，FAIL 时应延长观察期而非直接否决。默认关闭，每通道每窗至多一单。',
  },
  firsthit_down_g7_v1: {
    name: '首触G7基底（押DOWN）',
    desc: 'G0 + 小实体(body_r≤0.35) ∧ 上影拒绝(wick01=1)。40d 全量回测胜率 19%~20%，EV +1.98~+2.34（FDR q=0.0002）。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  g7_streak_v1: {
    name: '首触G7+非强连阳（押DOWN）',
    desc: 'G7 组合 + 前驱 5m 连阳≤1 根（排除多头单边大势逆势送命）。回测胜率 18.2%~20.0%，日均 5.1 单，FDR q=0.0004。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  g7_wick20_v1: {
    name: '首触G7+长上影（押DOWN）',
    desc: 'G7 组合 + 上影线长度≥2.0bp（密集高位拒绝）。回测胜率 17.8%~24.1%，EV +1.84~+3.06。默认关闭，执行价护栏 0.12，每通道每窗至多一单。',
  },
  g7_strict_v1: {
    name: '首触G7严格版（押DOWN）',
    desc: 'G7 组合 + streak≤1 ∧ upper_wick≥1.5bp。双重强化，回测胜率 20.8%~23.0%，EV +2.68~+3.27。默认关闭，执行价护栏 0.12，每通道每窗至多一单。',
  },
  g7_q05_v1: {
    name: '首触G7+深折价（押DOWN）',
    desc: 'G7 组合 + 进场报价 q≤0.05（极端赔率凸性档）。回测单注 EV +4.10~+5.77。默认关闭，执行价护栏 0.05，每通道每窗至多一单。',
  },
  g7_t270_v1: {
    name: '首触G7+非极晚（押DOWN）',
    desc: 'G7 组合 + 触发时刻 t≤270s（保留均值回归扩散时间窗口）。回测胜率 27.9%~30.4%，EV +3.08~+4.41。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  firsthit_down_btcsoft_v1: {
    name: '首触报价错杀（押DOWN）',
    desc: '【仅记录，不实盘】G3 门（chg≤+2.82bp）∩ 浅价 q≥0.07 ∩ 触发前 60 采样点 BTC 未上行（btc_speed_60_bps≥0，DOWN 侧有向口径）——DOWN 报价被砸浅但 BTC 侧无涨势，判定为报价错杀而非信息驱动。来源：2026-10-05 forward 母事件 ex-ante 扫描（09-17~10-04 G3 门内 2144 笔）87 笔胜率 18.4%/EV+1.02、剔 Top5 赢单后 +0.33，为约 30 个扫描格子中唯一稳健候选；多重比较未校正，基准当先验不当结论，影子前向 ≥4 周裁决。',
  },
  late_night_contrarian_v2: {
    name: '深夜逆势·日高回落门禁版（押 DOWN）',
    desc: 'quote_edge 族深夜变体 v2：t∈[45,90)s 且报价 q∈[0.25,0.30) ∩ 北京时间窗 (22~24 时) ∩ 触发时点距当日高点回落≥0.30%。OOS n=50 wr 44.0% CI[31.2%,57.7%] vs 盈亏平衡≈27%；门禁数据缺失→不落表（保守跳过，v1 不受影响）。护栏 0.33（区间上界 0.30+0.03 容忍），开启后每通道每窗至多一单。⚠️纪律推翻：原 docstring"纯影子前向攒样本"被用户拍板改为实盘注册（线上已开启下单）。',
  },
  hm_inside_5m_v2: {
    name: '5m孕线上吊线精选反转（押DOWN）',
    desc: '5m Rev2 精选反转：前置 4 根高点最长实体阳线 + 信号柱完全被包裹（Inside Bar）+ 下影占比 75%~90%、上影占比≤10% → 押次根 5m 收阴 DOWN。MARKET 市价单，护栏 0.30（成交均价 ≥0.30 弃单），可通过实盘通道热配置调整。',
  },
  hm_inside_15m_v2: {
    name: '15m孕线上吊线反转（押DOWN）',
    desc: '15m Ver2 经典反转：前置 4 根高点最长实体阳线 + 信号柱完全被包裹（Inside Bar）+ 顶部长下影上吊线、短上影≤10% → 押次根 15m 收阴 DOWN（720d 胜率 56.6%，30d 胜率 65.4%）。MARKET 市价单，护栏 0.30（成交均价 ≥0.30 弃单），可通过实盘通道热配置调整。',
  },
  ih_inside_15m_v2: {
    name: '15m孕线倒垂线反转（押UP）',
    desc: '15m Ver2 经典反转：前置 4 根低点最长实体阴线 + 信号柱完全被包裹（Inside Bar）+ 底部长上影倒垂线、短下影≤10% → 押次根 15m 收阳 UP（720d 胜率 52.5%，30d 胜率 62.5%）。MARKET 市价单，护栏 0.30（成交均价 ≥0.30 弃单），可通过实盘通道热配置调整。',
  },
  // --- 蜡烛反转族 18 逻辑版本（中文名与后端 candlestick_shadow_detector.CANDLESTICK_DISPLAY_NAMES 对齐）---
  // 共享 5m/15m 物理事件，各逻辑版本独立注册实盘；不收录会让订单表/通道面板退回裸英文 ID。
  candle_hm_bull_5m_consensus2_shadow_v1: {
    name: '5m长下影·至少2趋势',
    desc: '蜡烛反转族 5m 主影子：阳线、下影≥50%、实体≤50%、上影≤10%，且 ret3/ret5_majority/ma10 至少两种上涨 → 押次根 5m DOWN。实盘叠加入场层低波门禁（1h 实现波动率 7 日分位 ≤0.30 弃单）。与对照/归因版本共享物理事件，实盘按逻辑版本独立派单。',
  },
  candle_hm_bull_5m_consensus3_shadow_v1: {
    name: '5m长下影·三趋势对照',
    desc: '蜡烛反转族 5m 严格对照：主影子的三趋势全同意子集；与主版本共用一个物理事件，不代表第二笔独立暴露。',
  },
  candle_hm_bull_15m_union_shadow_v1: {
    name: '15m长下影·任一趋势',
    desc: '蜡烛反转族 15m 主影子：同一长下影阳线形态，三趋势任一种上涨即触发 → 押次根 15m DOWN。',
  },
  candle_hm_bull_15m_consensus3_shadow_v1: {
    name: '15m长下影·三趋势对照',
    desc: '蜡烛反转族 15m 严格对照：主版本的三趋势全同意子集；共用物理事件。',
  },
  candle_hm_bull_5m_ret5_majority_component_shadow_v1: {
    name: '5m ret5归因组件',
    desc: '蜡烛反转族 5m ret5_majority 单趋势归因标签；仅解释边际贡献，与主影子共享物理事件。',
  },
  candle_hm_bull_5m_ret3_component_shadow_v1: {
    name: '5m ret3归因组件',
    desc: '蜡烛反转族 5m ret3 单趋势归因标签；仅解释边际贡献，与主影子共享物理事件。',
  },
  candle_hm_bull_15m_ret3_component_shadow_v1: {
    name: '15m ret3归因组件',
    desc: '蜡烛反转族 15m ret3 单趋势归因标签；仅解释边际贡献，与主影子共享物理事件。',
  },
  candle_hm_bull_15m_ma10_component_shadow_v1: {
    name: '15m ma10归因组件',
    desc: '蜡烛反转族 15m ma10 单趋势归因标签；仅解释边际贡献，与主影子共享物理事件。',
  },
  candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1: {
    name: '5m长下影·中高量能',
    desc: '蜡烛反转族预注册潜力：5m 主形态 + 0.75≤volume/median20<2（中高量能）；后验 regime 假设，只能用新数据前向验证。',
  },
  candle_hm_bull_5m_trend4h_up_shadow_hyp_v1: {
    name: '5m长下影·4h上涨',
    desc: '蜡烛反转族预注册潜力：5m 主形态 + 前 4h 上涨；只作标签，不替代主影子。',
  },
  candle_hm_bull_5m_asia_shadow_hyp_v1: {
    name: '5m长下影·亚洲时段',
    desc: '蜡烛反转族预注册潜力：5m 主形态 + UTC 00~08 亚洲时段；只作标签。',
  },
  candle_hm_bull_15m_q45_55_shadow_hyp_v1: {
    name: '15m长下影·报价0.45~0.55',
    desc: '蜡烛反转族预注册潜力：15m 主形态 + 首个完整 DOWN 报价落在 [0.45,0.55)；只作标签。',
  },
  candle_hm_bull_15m_trend1h_up_shadow_hyp_v1: {
    name: '15m长下影·1h上涨',
    desc: '蜡烛反转族预注册潜力：15m 主形态 + 前 1h 上涨；只作标签。',
  },
  candle_hm_bull_15m_asia_shadow_hyp_v1: {
    name: '15m长下影·亚洲时段',
    desc: '蜡烛反转族预注册潜力：15m 主形态 + UTC 00~08 亚洲时段；低样本，只作标签。',
  },
  candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1: {
    name: '5m上涨阳线长上影',
    desc: '独立潜力：5m ret5 上涨、阳线长上影≥70%、实体≤30%、下影≤10% → 押 DOWN。',
  },
  candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1: {
    name: '5m严格零上影阴线长下影',
    desc: '独立潜力：5m ret3 上涨、阴线长下影≥50%、实体≤50%、原始 OHLC 严格零上影 → 押 DOWN。',
  },
  candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1: {
    name: '15m ret5阳线长下影',
    desc: '独立潜力：15m ret5 上涨、阳线长下影≥50%、实体≤40%、上影≤10% → 押 DOWN。',
  },
  candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1: {
    name: '15m下跌阳线长上影',
    desc: '独立潜力：15m ret3 下跌、阳线长上影≥50%、实体≤40%、下影≤10% → 押 UP。',
  },
}

/* 信号分类拆成独立维度：采集层始终可记录；执行层才决定是否真金下单。 */
const EXECUTION_BADGE: Record<string, { label: string; cls: string }> = {
  SHADOW_ONLY: { label: '仅影子', cls: 'ds-badge-violet' },
  LIVE_AVAILABLE: { label: '可实盘·未开', cls: 'ds-badge-warn' },
  LIVE_ACTIVE: { label: '实盘中', cls: 'ds-badge-up' },
  LIVE_RETIRED: { label: '已退役', cls: 'ds-badge-neutral' },
  RESEARCH_ONLY: { label: '仅研究', cls: 'ds-badge-indigo' },
}
const FAMILY_LABELS: Record<string, string> = {
  candlestick_reversal: '蜡烛反转', kline_reversal: 'K线反转', combo: '组合条件',
  kline: 'K线/组合', misalignment: '情绪/报价', pattern: '形态触价',
  absorption: '吸收跟随', firsthit: '首触反转', gap_crowd: '报价-人群错位',
  scene: '场景突破', quote_edge: '报价边缘', x4: '情绪错位',
  s2_cond: 'S2条件', nextbar: '次根方向', legacy: '其他',
}
const ROLE_LABELS: Record<string, string> = {
  PRIMARY: '主版本', CONTROL: '严格对照', COMPONENT: '归因组件',
  HYPOTHESIS: '预注册潜力', STRATEGY: '策略',
}
const FAMILY_ORDER = ['candlestick_reversal', 'scene', 'quote_edge', 'x4', 'gap_crowd', 's2_cond', 'nextbar', 'absorption', 'firsthit', 'combo', 'kline_reversal', 'pattern', 'misalignment', 'kline', 'legacy']
const familyRank = (family: string) => {
  const rank = FAMILY_ORDER.indexOf(family)
  return rank < 0 ? FAMILY_ORDER.length : rank
}
const signalFamilyLabel = (family: string | undefined) => FAMILY_LABELS[family ?? 'legacy'] ?? family ?? '其他'

/* ==========================================================================
   Card — DESIGN.md 规格
   - 平坦：无阴影（移除 shadow-sm）
   - token：border-line + bg-card + rounded-card
   - 标题 ink-95（之前是 text-ink-80 → ink-95）
   ========================================================================== */
function Card({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`ds-card ${className}`}>
      <div className="ds-card-head">
        <h2 className="ds-card-title">{title}</h2>
      </div>
      <div className="ds-card-body">{children}</div>
    </div>
  )
}

// ============================================================
// 实盘 Tab（账户状态 + 人工测试单 + 订单历史，2026-08-22）
// ============================================================


// ============================================================
// 手动下单模态框：一键下单 5m/15m 多个周期窗口。
// 聚焦实时 K 线、周期窗口、方向、金额和提交；平仓/赎回留在实盘页面主区。
// ============================================================

interface FutureWindow {
  window_start: number
  window_end: number | null
  ahead_sec: number
  trading_status: string
  up_price: number | null
  down_price: number | null
  topic_id: number
  available: boolean
}

const fmtHHMM = (ms: number) =>
  new Date(ms).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

const fmtMMSS = (sec: number) =>
  `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`

// ============================================================
// 实时 K 线小组件（2026-09-18）：下单弹框内嵌，币安 BTC 现货 K 线。
// 数据源：后端代理 /api/chart/btc-klines 5s 轮询为基线（浏览器无需可达币安，
// 任何网络都能出图）；币安 WS 直连可用时自动升级为实时推送、断开回落轮询。
// 自绘 SVG 蜡烛图（不依赖 recharts 蜡烛支持）。
// 周期可切 1m/5m/15m/1h/1d，含量能副图（MA20 均量），并显示所选下单周期当前窗收盘倒计时。
// ============================================================

interface KlineBar { t: number; o: number; h: number; l: number; c: number; v: number }

const BINANCE_WS = 'wss://stream.binance.com:9443/ws'

// 计算单值序列 EMA
function computeEmaSeries(bars: KlineBar[], span: number): (number | null)[] {
  const k = 2 / (span + 1)
  let prevEma: number | null = null
  return bars.map((b, idx) => {
    if (idx === 0) {
      prevEma = b.c
      return prevEma
    }
    if (prevEma == null) return null
    prevEma = b.c * k + prevEma * (1 - k)
    return prevEma
  })
}

// 计算单值序列简单移动平均（量能 MA 用；窗口不足返回 null）
function computeSmaSeries(values: number[], span: number): (number | null)[] {
  return values.map((_, idx) => {
    if (idx < span - 1) return null
    let s = 0
    for (let j = idx - span + 1; j <= idx; j++) s += values[j]
    return s / span
  })
}

function KlineMini({ period: orderPeriod, nowMs }: { period: '5m' | '15m'; nowMs: number }) {
  // 独立选看 K 线周期，可看 1m/5m/15m/1h/1d，初始跟随下单周期
  const [chartPeriod, setChartPeriod] = useState<'1m' | '5m' | '15m' | '1h' | '1d'>(orderPeriod)
  const [bars, setBars] = useState<KlineBar[]>([])
  const [live, setLive] = useState<'ws' | 'proxy' | 'init'>('init')
  const [showIndicators, setShowIndicators] = useState(true)
  const [view, setView] = useState({ i0: 0, count: 60 })
  const previousBarCountRef = useRef(0)

  // 当外部下单周期切换时同步更新默认看盘周期
  useEffect(() => {
    setChartPeriod(orderPeriod)
  }, [orderPeriod])

  // 后端代理轮询（基线）+ 币安 WS 直连（增强）；周期切换重建连接
  useEffect(() => {
    let alive = true
    let ws: WebSocket | null = null
    let pollTimer: number | null = null
    setLive('init')
    setBars([])
    setView({ i0: 0, count: 60 })
    previousBarCountRef.current = 0

    // 代理返回对象数组（open_time/open/...）：与已有 bars 按时间合并补历史，
    // 已在手的（WS 推送）同刻数据优先保留，避免旧快照顶掉新柱
    const applyProxy = (raw: unknown) => {
      const next: KlineBar[] = (Array.isArray(raw) ? raw : []).map(r => {
        const k = r as Record<string, unknown>
        return { t: Number(k.open_time), o: Number(k.open), h: Number(k.high), l: Number(k.low), c: Number(k.close), v: Number(k.volume) }
      })
      if (!alive || !next.length) return
      setBars(prev => {
        if (!prev.length) return next.slice(-120)
        const byT = new Map(next.map(b => [b.t, b]))
        for (const b of prev) byT.set(b.t, b)   // WS 新柱优先
        return [...byT.values()].sort((a, b) => a.t - b.t).slice(-120)
      })
    }
    const mergeKline = (k: KlineBar) => {
      if (!alive) return
      setBars(prev => {
        if (!prev.length) return [k]
        const last = prev[prev.length - 1]
        if (k.t === last.t) return [...prev.slice(0, -1), k]
        if (k.t > last.t) return [...prev, k].slice(-120)
        return prev
      })
    }

    // 基线轮询走后端代理（服务器出网拉币安），浏览器网络不直连币安也能出图
    const startPoll = () => {
      if (pollTimer != null) return
      setLive('proxy')
      const tick = async () => {
        try {
          const d = await api.getBtcKlinesLive(chartPeriod, 120)
          // 无论 WS 是否已接管都合并：WS 先握手时若丢弃代理历史，bars 只剩 1 根会卡在「加载中」
          if (alive) applyProxy(d?.klines)
        } catch { /* 轮询失败静默等下一轮 */ }
      }
      tick()
      pollTimer = window.setInterval(tick, 5_000)
    }
    const stopPoll = () => {
      if (pollTimer != null) { clearInterval(pollTimer); pollTimer = null }
    }

    startPoll()
    try {
      ws = new WebSocket(`${BINANCE_WS}/btcusdt@kline_${chartPeriod}`)
      ws.onopen = () => { if (alive) setLive('ws') }
      ws.onmessage = ev => {
        try {
          const msg = JSON.parse(ev.data)
          const k = msg?.k
          if (!k) return
          mergeKline({ t: k.t, o: +k.o, h: +k.h, l: +k.l, c: +k.c, v: +k.v })
        } catch { /* 非 kline 消息忽略 */ }
      }
      ws.onerror = () => { if (alive) startPoll() }
      ws.onclose = () => { if (alive) startPoll() }
    } catch {
      startPoll()
    }

    return () => {
      alive = false
      if (ws) { ws.onclose = null; ws.close() }
      stopPoll()
    }
  }, [chartPeriod])

  // ---- 交互状态：滚轮/按钮缩放 + 拖拽平移 + 悬停十字线数据 ----
  const W = 840
  const H = 320
  const RIGHT_PAD = 60
  const BOTTOM_PAD = 20
  const VOL_H = 56
  const CHART_W = W - RIGHT_PAD
  const CHART_H = H - BOTTOM_PAD - VOL_H
  const n = bars.length
  const [hover, setHover] = useState<{ i: number; vx: number; vy: number } | null>(null)
  const dragRef = useRef<{ x: number; i0: number } | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  const count = Math.max(15, Math.min(view.count, Math.max(n, 15)))
  const i0 = Math.max(0, Math.min(view.i0, Math.max(0, n - count)))
  const slice = bars.slice(i0, i0 + count)

  // 全量计算 EMA 再切片，保证连续性
  const ema7Full = useMemo(() => computeEmaSeries(bars, 7), [bars])
  const ema25Full = useMemo(() => computeEmaSeries(bars, 25), [bars])
  const ema7Slice = ema7Full.slice(i0, i0 + count)
  const ema25Slice = ema25Full.slice(i0, i0 + count)
  // 量能 MA20（全量算再切片，与 EMA 同套路）
  const volMaFull = useMemo(() => computeSmaSeries(bars.map(b => b.v), 20), [bars])
  const volMaSlice = volMaFull.slice(i0, i0 + count)

  // 首批历史数据定位最新区间；后续仅在原视图贴右时跟随新柱。
  useEffect(() => {
    const previousCount = previousBarCountRef.current
    setView(v => {
      const wasAtRightEdge = previousCount === 0 || v.i0 + v.count >= previousCount - 1
      return wasAtRightEdge ? { ...v, i0: Math.max(0, n - v.count) } : v
    })
    previousBarCountRef.current = n
  }, [n])

  // 滚轮缩放：非 passive 监听 + preventDefault，避免滚动模态框
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const f = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      setView(v => {
        const c = Math.max(15, Math.min(n, Math.round(v.count * (e.deltaY < 0 ? 0.8 : 1.25))))
        const anchor = v.i0 + f * v.count
        return { i0: Math.max(0, Math.min(n - c, Math.round(anchor - f * c))), count: c }
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [n])

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    if (dragRef.current) {
      const per = rect.width / count
      const di = Math.round((e.clientX - dragRef.current.x) / per)
      const base = dragRef.current.i0
      setView(v => ({ ...v, i0: Math.max(0, Math.min(n - v.count, base - di)) }))
      return
    }
    const vx = ((e.clientX - rect.left) / rect.width) * W
    const vy = ((e.clientY - rect.top) / rect.height) * H
    const i = Math.max(0, Math.min(slice.length - 1, Math.floor((vx / CHART_W) * count)))
    setHover({ i, vx, vy })
  }

  // 所选下单周期当前窗收盘倒计时
  const orderStep = orderPeriod === '5m' ? 300_000 : 900_000
  const remain = Math.max(0, Math.floor(((Math.floor(nowMs / orderStep) + 1) * orderStep - nowMs) / 1000))

  // 坐标（基于当前显示 slice 计算真实高低极值；绘图边界另加留白）
  let dataLow = Infinity
  let dataHigh = -Infinity
  let maxIdx = 0
  let minIdx = 0
  slice.forEach((b, i) => {
    if (b.l < dataLow) { dataLow = b.l; minIdx = i }
    if (b.h > dataHigh) { dataHigh = b.h; maxIdx = i }
  })
  if (!Number.isFinite(dataLow)) { dataLow = 0; dataHigh = 1 }
  const pad = (dataHigh - dataLow) * 0.08 || 1
  const chartLow = dataLow - pad
  const chartHigh = dataHigh + pad
  const y = (v: number) => CHART_H - ((v - chartLow) / (chartHigh - chartLow)) * CHART_H
  const bw = slice.length > 0 ? CHART_W / slice.length : CHART_W
  const last = n > 0 ? bars[n - 1] : null
  const hoverBar = hover != null ? slice[hover.i] : null

  // 4 档水平参考价格线
  const yTicks = [0.15, 0.40, 0.65, 0.90].map(ratio => chartLow + ratio * (chartHigh - chartLow))

  // 生成折线 path
  const buildLinePath = (values: (number | null)[]) => {
    let d = ''
    values.forEach((v, idx) => {
      if (v == null) return
      const cx = idx * bw + bw / 2
      const cy = y(v)
      d += d === '' ? `M ${cx} ${cy}` : ` L ${cx} ${cy}`
    })
    return d
  }

  const ema7Path = buildLinePath(ema7Slice)
  const ema25Path = buildLinePath(ema25Slice)

  // 量能副图：比例尺基于可见 slice，紧凑数字格式
  const volMax = Math.max(1e-9, ...slice.map(b => b.v)) * 1.15
  const vy = (v: number) => CHART_H + VOL_H - (v / volMax) * VOL_H
  const fmtVol = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(2)}k` : v.toFixed(1))
  const buildVolLinePath = (values: (number | null)[]) => {
    let d = ''
    values.forEach((v, idx) => {
      if (v == null) return
      d += d === '' ? `M ${idx * bw + bw / 2} ${vy(v)}` : ` L ${idx * bw + bw / 2} ${vy(v)}`
    })
    return d
  }
  const volMaPath = buildVolLinePath(volMaSlice)

  return (
    <div className="space-y-1.5 rounded-card border border-line bg-card p-3">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-bold text-xs text-ink-95">BTC 实时 K 线</span>
        {/* 独立看盘周期切换 */}
        <div className="flex items-center gap-0.5 bg-sunken rounded-pill p-0.5 border border-line-soft">
          {(['1m', '5m', '15m', '1h', '1d'] as const).map(p => (
            <button
              key={p}
              onClick={() => setChartPeriod(p)}
              className={`px-2 py-0.5 rounded-pill text-[11px] font-semibold transition ${chartPeriod === p ? 'bg-brand text-white font-bold' : 'text-ink-55 hover:text-ink-80'}`}
            >
              {p}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowIndicators(v => !v)}
          className={`px-2 py-0.5 rounded border text-[10px] transition ${showIndicators ? 'border-brand text-brand bg-brand-soft/40' : 'border-line text-ink-55'}`}
          title="切换 EMA 7 与 EMA 25 均线显示"
        >
          EMA 7/25
        </button>
        <span className="flex items-center gap-0.5 ml-1" title="滚轮缩放 · 拖拽平移 · 悬停看数据">
          <button onClick={() => setView(v => {
            const c = Math.max(15, Math.round(v.count * 0.8))
            return { i0: Math.max(0, Math.min(n - c, v.i0 + Math.round((v.count - c) / 2))), count: c }
          })} className="px-1.5 py-0.5 rounded border border-line text-[10px] text-ink-55 hover:border-brand">＋</button>
          <button onClick={() => setView(v => {
            const c = Math.min(Math.max(n, 15), Math.round(v.count * 1.25))
            return { i0: Math.max(0, Math.min(n - c, v.i0)), count: c }
          })} className="px-1.5 py-0.5 rounded border border-line text-[10px] text-ink-55 hover:border-brand">－</button>
          <button onClick={() => setView({ i0: Math.max(0, n - 60), count: Math.min(60, Math.max(n, 15)) })}
            className="px-1.5 py-0.5 rounded border border-line text-[10px] text-ink-55 hover:border-brand">重置</button>
        </span>

        {showIndicators && (
          <div className="flex items-center gap-2 text-[10px] font-mono ml-1">
            <span className="text-[#f59e0b]">■ EMA7</span>
            <span className="text-[#06b6d4]">■ EMA25</span>
          </div>
        )}
        <span className="text-[10px] font-mono text-[#a78bfa]">■ 量MA20</span>

        <div className="ml-auto flex items-center gap-2 text-xs">
          <span className="font-mono text-[11px] font-bold tabular-nums" title={`当前下单周期 (${orderPeriod}) 窗口收盘倒计时`}>
            {orderPeriod}窗收盘 {fmtMMSS(remain)}
          </span>
          <span className={`text-[11px] font-medium ${live === 'ws' ? 'text-positive' : 'text-ink-55'}`} title={live === 'ws' ? 'Binance 现货 WebSocket 实时推送' : live === 'proxy' ? '经服务器代理轮询（约 5s 延迟），浏览器直连币安不可达' : '初始化中'}>
            {live === 'ws' ? '● 实时' : live === 'proxy' ? '○ 代理' : '…'}
          </span>
        </div>
      </div>

      {n > 1 ? (
        <div ref={wrapRef} className="relative select-none">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-sm bg-sunken cursor-crosshair" style={{ height: 320 }}
            onMouseMove={onMove}
            onMouseLeave={() => { setHover(null); dragRef.current = null }}
            onMouseDown={e => { dragRef.current = { x: e.clientX, i0 }; setHover(null) }}
            onMouseUp={() => { dragRef.current = null }}
          >
            {/* 网格水平线 + 价格标尺 */}
            {yTicks.map((price, idx) => {
              const yPos = y(price)
              return (
                <g key={idx}>
                  <line x1={0} x2={CHART_W} y1={yPos} y2={yPos} stroke="var(--line-soft)" strokeDasharray="3 3" strokeWidth={0.8} />
                  <text x={CHART_W + 6} y={yPos + 3} fontSize={10} fill="var(--ink-55)" fontFamily="var(--font-stack-mono)">
                    {price.toFixed(1)}
                  </text>
                </g>
              )
            })}

            {/* 蜡烛体 */}
            {slice.map((b, i) => {
              const cx = i * bw + bw / 2
              const up = b.c >= b.o
              const color = up ? 'var(--positive)' : 'var(--negative)'
              const bodyTop = y(Math.max(b.o, b.c))
              const bodyH = Math.max(1.5, Math.abs(y(b.o) - y(b.c)))
              return (
                <g key={b.t}>
                  <line x1={cx} x2={cx} y1={y(b.h)} y2={y(b.l)} stroke={color} strokeWidth={1.2} />
                  <rect x={cx - bw * 0.35} y={bodyTop} width={bw * 0.7} height={bodyH} fill={color} rx={1} />
                </g>
              )
            })}

            {/* EMA 均线 */}
            {showIndicators && (
              <>
                <path d={ema7Path} fill="none" stroke="#f59e0b" strokeWidth={1.5} opacity={0.85} />
                <path d={ema25Path} fill="none" stroke="#06b6d4" strokeWidth={1.5} opacity={0.85} />
              </>
            )}

            {/* 视口高点与低点标注 */}
            {slice.length > 0 && (
              <>
                <g>
                  <text
                    x={maxIdx * bw + bw / 2}
                    y={Math.max(14, y(dataHigh) - 6)}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--positive)"
                    fontFamily="var(--font-stack-mono)"
                    fontWeight="bold"
                  >
                    H {dataHigh.toFixed(1)}
                  </text>
                </g>
                <g>
                  <text
                    x={minIdx * bw + bw / 2}
                    y={Math.min(CHART_H - 4, y(dataLow) + 12)}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--negative)"
                    fontFamily="var(--font-stack-mono)"
                    fontWeight="bold"
                  >
                    L {dataLow.toFixed(1)}
                  </text>
                </g>
              </>
            )}

            {/* 量能副图：成交量柱（涨绿/跌红）+ MA20 均量线，与主图共享时间轴 */}
            <line x1={0} x2={CHART_W} y1={CHART_H} y2={CHART_H} stroke="var(--line-soft)" strokeWidth={1} />
            <text x={4} y={CHART_H + 11} fontSize={9} fill="#a78bfa" fontFamily="var(--font-stack-mono)">量 MA20</text>
            {slice.map((b, i) => {
              const hgt = b.v > 0 ? Math.max(1, (b.v / volMax) * VOL_H) : 0
              return (
                <rect key={b.t} x={i * bw + bw * 0.15} y={CHART_H + VOL_H - hgt} width={bw * 0.7} height={hgt}
                  fill={b.c >= b.o ? 'var(--positive)' : 'var(--negative)'} opacity={0.4} />
              )
            })}
            <path d={volMaPath} fill="none" stroke="#a78bfa" strokeWidth={1.2} />

            {/* 最新价格水平参考线 */}
            {last != null && (
              <g>
                <line x1={0} x2={CHART_W} y1={y(last.c)} y2={y(last.c)} stroke={last.c >= last.o ? 'var(--positive)' : 'var(--negative)'} strokeDasharray="4 2" strokeWidth={1} />
                <rect x={CHART_W + 2} y={y(last.c) - 9} width={RIGHT_PAD - 4} height={18} rx={2} fill={last.c >= last.o ? 'var(--positive)' : 'var(--negative)'} />
                <text x={CHART_W + 5} y={y(last.c) + 3.5} fontSize={10} fill="#FFFFFF" fontFamily="var(--font-stack-mono)" fontWeight="bold">
                  {last.c.toFixed(1)}
                </text>
              </g>
            )}

            {/* 十字线 */}
            {hover != null && (
              <g>
                <line x1={hover.vx} x2={hover.vx} y1={0} y2={CHART_H + VOL_H} stroke="var(--ink-55)" strokeWidth={0.8} strokeDasharray="3 3" />
                <line x1={0} x2={CHART_W} y1={hover.vy} y2={hover.vy} stroke="var(--ink-55)" strokeWidth={0.8} strokeDasharray="3 3" />
              </g>
            )}

            {/* 底部时间坐标刻度 */}
            {slice.map((b, i) => {
              const stepInterval = Math.max(1, Math.floor(slice.length / 6))
              if (i % stepInterval !== 0 && i !== slice.length - 1) return null
              const cx = i * bw + bw / 2
              return (
                <text key={b.t} x={cx} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--ink-55)" fontFamily="var(--font-stack-mono)">
                  {new Date(b.t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}
                </text>
              )
            })}
          </svg>

          {/* 悬浮数据卡片 */}
          {hoverBar != null && hover != null && (
            <div className="absolute z-10 pointer-events-none rounded-card border border-line bg-card/95 backdrop-blur px-3 py-1.5 text-[11px] font-mono text-ink-95 shadow-lg space-y-0.5"
              style={{ left: `${Math.min(70, Math.max(0, (hover.vx / W) * 100))}%`, top: 8 }}>
              <div className="font-bold border-b border-line-soft pb-0.5 mb-0.5">
                {new Date(hoverBar.t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })} · {chartPeriod} 现货
              </div>
              <div className="flex justify-between gap-3 text-ink-80">
                <span>开: {hoverBar.o.toFixed(1)}</span>
                <span>高: {hoverBar.h.toFixed(1)}</span>
              </div>
              <div className="flex justify-between gap-3 text-ink-80">
                <span>低: {hoverBar.l.toFixed(1)}</span>
                <span className="font-bold">收: {hoverBar.c.toFixed(1)}</span>
              </div>
              <div className="flex justify-between gap-3 text-ink-80">
                <span>量: {fmtVol(hoverBar.v)}</span>
                <span>均量: {volMaSlice[hover.i] != null ? fmtVol(volMaSlice[hover.i] as number) : '--'}</span>
              </div>
              <div className={hoverBar.c >= hoverBar.o ? 'text-positive font-bold' : 'text-negative font-bold'}>
                涨跌: {hoverBar.c >= hoverBar.o ? '+' : ''}{(((hoverBar.c - hoverBar.o) / hoverBar.o) * 100).toFixed(2)}%
                <span className="text-ink-55 font-normal ml-2">振幅: {(((hoverBar.h - hoverBar.l) / hoverBar.l) * 100).toFixed(2)}%</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="h-[280px] flex items-center justify-center rounded-sm bg-sunken text-xs text-ink-55">
          BTC 实时 K 线加载中…（支持 1m/5m/15m/1h/1d · 含量能，经服务器代理获取）
        </div>
      )}
    </div>
  )
}

// 悬浮快速下单（右下角 FAB）：一键下单 5m/15m 多个周期窗口（真实订单）。
function TestTradeFab({ quote, remainSec, wallet, refresh, orders, clockOffset }: {
  quote: Record<string, unknown> | null
  remainSec: number | null
  wallet: Record<string, unknown> | null
  refresh: () => void
  orders: Record<string, unknown>[]
  clockOffset: number
}) {
  const [open, setOpen] = useState(false)
  // 唯一切换器：下单周期（5m/15m）；K 线跟随同一周期
  const [period, setPeriod] = useState<'5m' | '15m'>('5m')
  // 窗口多选：'current' 或未来窗起点 ms；默认当前窗
  const [picked, setPicked] = useState<(number | 'current')[]>(['current'])
  const [future, setFuture] = useState<FutureWindow[]>([])
  const [futureErr, setFutureErr] = useState(false)
  const [loading, setLoading] = useState(false)
  // 手动刷新计数：bump 触发未来窗立即重扫
  const [reloadTick, setReloadTick] = useState(0)
  // 上次已消费的刷新计数：reloadTick 变化即强制绕过后端缓存重扫
  const lastReloadRef = useRef(0)
  const [amount, setAmount] = useState('1')
  const [side, setSide] = useState<'UP' | 'DOWN'>('UP')
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<Record<string, unknown>[]>([])
  const [currentWin, setCurrentWin] = useState<Record<string, unknown> | null>(null)
  // 大额（>10U/窗）勾选确认；小单一键直接下
  const [confirmed, setConfirmed] = useState(false)
  // 复利链：每腿执行价护栏 + 最近链进度（弹窗打开时 10s 轮询）
  const [chainGuard, setChainGuard] = useState('0.65')
  const [chains, setChains] = useState<Record<string, unknown>[]>([])
  const [chainMsg, setChainMsg] = useState('')

  const q = (v: unknown) => (typeof v === 'number' ? v : null)
  const nowMs = Date.now() + clockOffset
  const step = period === '5m' ? 300_000 : 900_000
  const baseWindow = Math.floor(nowMs / step) * step

  // 当前窗参考价（5m 用 quote-preview；15m 用 future-markets 的 current）
  const up = period === '5m' ? q(quote?.up_price) : q(currentWin?.up_price)
  const down = period === '5m' ? q(quote?.down_price) : q(currentWin?.down_price)
  const curEnd = period === '5m' ? q(quote?.window_end) : q(currentWin?.window_end)
  const balance = typeof wallet?.prediction_usdt_free === 'number'
    ? wallet.prediction_usdt_free as number : null

  // 当前窗剩余秒（15m 数据未加载时置 null，不误用 5m 口径）
  const remainLocal = curEnd != null
    ? Math.max(0, Math.floor((curEnd - nowMs) / 1000))
    : (period === '5m' ? remainSec : null)
  // 当前窗踩线（<10s）时自动移出选择
  const currentTooLate = remainLocal != null && remainLocal < 10

  const amtNum = parseFloat(amount)
  const amtOk = Number.isFinite(amtNum) && amtNum >= 0.1 && amtNum <= 50
  const bigAmount = amtNum > 10
  // 提交目标：'current' → 当前窗；过滤踩线当前窗与距开盘 <5s 的未来窗
  const targets = picked.map(w => (w === 'current' ? baseWindow : w))
    .filter(ws => {
      if (ws === baseWindow && currentTooLate) return false
      return ws - nowMs >= 5_000 || ws === baseWindow
    })
  const alreadyOrderedWins = new Set(orders
    .filter(o => String(o.signal_version ?? '').startsWith('manual_test')
      && o.market_period === period && o.status !== 'FAILED')
    .map(o => Number(o.window_start)))
  // 可连选的未来窗（可用且距开盘 ≥5s）；连选钮数量随实际窗口数自适应，不再设上限
  const nextAvailWins = future.filter(w =>
    w.available && w.window_start - nowMs >= 5_000)
  const canSubmit = amtOk && !busy && targets.length > 0
    && (!bigAmount || confirmed)

  // 踩线自动移出 + 大额时要求确认
  useEffect(() => {
    if (currentTooLate && picked.includes('current')) {
      setPicked(p => p.filter(w => w !== 'current'))
    }
  }, [currentTooLate, picked])
  useEffect(() => { if (!bigAmount) setConfirmed(false) }, [bigAmount])

  // 打开/切周期/手动刷新拉取未来窗（含当前窗参考价）
  useEffect(() => {
    if (!open) return
    let alive = true
    // ↻ 刷新：reloadTick 变化时 force=true 强制后端重扫（否则 45s 内命中缓存看不出变化）
    const force = reloadTick !== lastReloadRef.current
    lastReloadRef.current = reloadTick
    // 周期切换/手动刷新时先清掉旧列表：上一周期的窗口立即消失，视觉即时切换
    setFuture([])
    const load = (spinner: boolean, forceScan = false) => {
      if (spinner) setLoading(true)
      api.getFutureMarkets(period, 8, forceScan).then((d: Record<string, unknown>) => {
        if (!alive) return
        const wins = Array.isArray(d.windows) ? d.windows as FutureWindow[] : []
        setCurrentWin((d.current ?? null) as Record<string, unknown> | null)
        setFutureErr(false)
        // 静默轮询拿到空结果不覆盖已有列表：多为限流瞬态，避免把窗口清空而误报「未创建」
        if (spinner || wins.length > 0) setFuture(wins)
      }).catch(() => { if (alive) setFutureErr(true) })
        .finally(() => alive && setLoading(false))
    }
    load(true, force)
    // 后台静默刷新（30s）：不闪加载态
    const t = setInterval(() => load(false), 30_000)
    return () => { alive = false; clearInterval(t) }
  }, [open, period, reloadTick])

  // Esc 关闭
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const togglePick = (w: number | 'current') => {
    setPicked(p => p.includes(w) ? p.filter(x => x !== w) : [...p, w])
  }

  const handleSubmit = async () => {
    if (!canSubmit) return
    setBusy(true)
    setResults([])
    const out: Record<string, unknown>[] = []
    try {
      for (const ws of targets) {
        const res = await api.postTradeTest(amtNum, side, {
          market_period: period,
          window_start: ws,
        })
        out.push({ ...res, window_start: ws,
          window_end: ws + step })
        // 逐窗独立尝试（每窗一单占位互不影响）；余额不足等硬错误会逐窗复现于结果
      }
      setResults(out)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      setConfirmed(false)
    }
  }

  const guardNum = parseFloat(chainGuard)
  const chainWins = [...targets].sort((x, y) => x - y)
  const canChain = period === '5m' && amtOk && amtNum <= 10 && !busy && chainWins.length >= 2
    && guardNum > 0 && guardNum < 1 && (!bigAmount || confirmed)

  const loadChains = () => api.getCompound()
    .then((d: Record<string, unknown>) => setChains(Array.isArray(d.chains) ? d.chains as Record<string, unknown>[] : []))
    .catch(() => {})
  useEffect(() => {
    if (!open) return
    loadChains()
    const t = setInterval(loadChains, 10_000)
    return () => clearInterval(t)
  }, [open])

  const handleChain = async () => {
    if (!canChain) return
    const names = chainWins.map(ws => fmtHHMM(ws)).join(' → ')
    if (!window.confirm(
      `复利下单（${side === 'UP' ? '涨' : '跌'}）
${names}
首腿 ${amtNum}U（限 0.1~10U），赢则本金+利润滚入下一腿，输即终止。
`
      + `提前下单：本窗报价 ≥0.95（距收盘 60~150s）或 距收盘 ≤15s 且领先 ≥3bp 时，即提前押下一腿，
`
      + `押注额＝本金+按报价算得的利润。
`
      + `代价：提前押上后本窗若反转，当前腿与下一腿一起亏。后续腿本金不封顶，余额不足即终止。
`
      + `每腿执行价 ≥ ${guardNum} 弃单并终止。确认真实下单？`)) return
    setBusy(true)
    setChainMsg('')
    try {
      const res = await api.postCompound(amtNum, side, chainWins, guardNum)
      setChainMsg(res.error ? `✗ ${String(res.error)}` : `✓ 复利链 #${res.id} 已启动`)
      loadChains()
    } catch (e) {
      setChainMsg(`✗ 请求失败: ${(e as Error).message}`)
    } finally {
      setBusy(false)
      setConfirmed(false)
    }
  }

  const chipCls = (on: boolean) =>
    `px-2.5 py-1 text-xs font-semibold rounded-pill border transition ${on
      ? 'bg-brand text-white border-brand'
      : 'bg-card text-ink-80 border-line hover:border-brand'} disabled:opacity-40`

  return (
    <>
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-40 px-4 py-3 rounded-pill font-bold text-white text-sm transition bg-brand hover:bg-brand-hover"
        title="一键下单（5m/15m 多周期窗口，真实订单）"
      >
        {busy ? '下单中…' : '⚡ 下单'}
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40 bg-ink-black/20 backdrop-blur-xs" onClick={() => setOpen(false)} />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="一键下单与BTC实时行情"
            className="fixed bottom-20 right-4 sm:right-6 z-50 w-[880px] max-w-[96vw] max-h-[90vh] overflow-y-auto bg-card rounded-card border border-line p-5 space-y-3 text-sm shadow-2xl"
          >
            {/* 标题 + 周期切换 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-bold text-base text-ink-95">⚡ 一键下单</span>
                <span className="text-xs text-ink-55">下单目标周期:</span>
                <div className="flex items-center gap-1">
                  {(['5m', '15m'] as const).map(p => (
                    <button key={p} onClick={() => { setPeriod(p); setPicked(['current']) }}
                      className={`px-3 py-1 text-xs font-bold rounded-pill border transition ${period === p ? 'bg-brand text-white border-brand' : 'bg-card text-ink-55 border-line hover:border-brand'}`}
                    >{p} 市场</button>
                  ))}
                </div>
              </div>
              <button
                onClick={() => setOpen(false)}
                className="text-ink-55 hover:text-ink-95 text-xl leading-none px-2 py-1 rounded hover:bg-card-hover transition"
                title="关闭弹窗 (Esc)"
              >
                ✕
              </button>
            </div>

            {/* 参考价 + 当前窗倒计时 */}
            <div className={`flex items-center justify-between rounded-sm px-3.5 py-2 text-xs ${remainLocal != null && remainLocal < 60 ? 'bg-negative-soft text-negative' : 'bg-sunken text-ink-80'}`}>
              <span className="font-mono text-[13px]">
                当前窗参考价：
                <span className="text-positive font-bold">涨 {up != null ? up.toFixed(3) : '--'}</span>
                <span className="opacity-40 mx-1.5">/</span>
                <span className="text-negative font-bold">跌 {down != null ? down.toFixed(3) : '--'}</span>
              </span>
              <span className="font-mono font-bold tabular-nums text-xs">
                {curEnd != null && `到期 ${fmtHHMM(curEnd)} · 剩余 ${fmtMMSS(remainLocal ?? 0)}`}
              </span>
            </div>

            {/* 实时 K 线：币安 BTC 现货，支持 1m/5m/15m 查看 + EMA 均线 + 标尺刻度 */}
            <KlineMini period={period} nowMs={nowMs} />

            {/* 窗口多选（核心）：当前 + 未来窗，点选高亮；
                另提供「下期起连选 N 窗」快捷钮——自动选中从下一周期起往后 N 个可用窗 */}
            <div className="space-y-1">
              <div className="flex items-center text-[11px] text-ink-55">
                <span>点选要下单的窗口（可多选）</span>
                <button onClick={() => setReloadTick(t => t + 1)} disabled={loading}
                  title="立即重新扫描未来窗口"
                  className="ml-2 px-1.5 py-0.5 rounded border border-line text-ink-55 hover:border-brand hover:text-ink-80 transition disabled:opacity-40"
                >↻ 刷新</button>
                <span className="ml-auto">{loading ? '扫描未来周期…' : `已选 ${targets.length} 窗`}</span>
              </div>
              {/* 未来窗加载失败（后端接口异常）与币安空窗期区分提示，均自动重试 */}
              {!loading && future.length === 0 && (futureErr ? (
                <div className="text-[11px] text-negative rounded-sm bg-negative-soft px-2 py-1">
                  未来窗加载失败（后端接口异常），自动重试中…
                </div>
              ) : (
                <div className="text-[11px] text-ink-55 rounded-sm bg-sunken px-2 py-1">
                  币安尚未创建下一批未来市场（周期边界常见），约 30 秒后自动重试；当前窗仍可下单
                </div>
              ))}
              {/* 快捷连选：从下一周期（不含当前窗）往后数 N 个可用窗自动选中 */}
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[11px] text-ink-55">下期起连选</span>
                {Array.from({ length: nextAvailWins.length }, (_, i) => i + 1).map(n => {
                  const autoActive = !picked.includes('current')
                    && picked.length === Math.min(n, nextAvailWins.length)
                    && nextAvailWins.slice(0, n).every(w => picked.includes(w.window_start))
                  return (
                    <button key={n}
                      disabled={loading || nextAvailWins.length === 0}
                      title={`自动选中未来第 1~${n} 个窗口（不含当前窗）`}
                      onClick={() => {
                        const wins = nextAvailWins.slice(0, n)
                        setPicked(wins.map(w => w.window_start))
                      }}
                      className={`px-2 py-0.5 rounded-pill border text-xs transition ${autoActive
                        ? 'bg-brand text-white border-brand font-semibold'
                        : 'border-line bg-card text-ink-80 hover:border-brand'} disabled:opacity-40`}
                    >{n}窗</button>
                  )
                })}
                <span className="text-[11px] text-ink-55">（或直接点选下方窗口）</span>
              </div>
              <div className="flex items-stretch gap-1.5 flex-wrap">
                {/* 窗卡：上行时间、下行该窗真实报价（涨绿/跌红）；选中态白字 */}
                {/* 当前窗卡 */}
                <button onClick={() => togglePick('current')}
                  disabled={currentTooLate}
                  title={currentTooLate ? '当前窗距收盘不足 10 秒' : `剩余 ${fmtMMSS(remainLocal ?? 0)}`}
                  className={`${chipCls(picked.includes('current') && !currentTooLate)} flex-col items-center gap-0.5 px-2.5 py-1 leading-tight`}
                >
                  <span>当前 {fmtHHMM(baseWindow)}</span>
                  <span className="font-mono text-[10px]">
                    <span className={picked.includes('current') ? 'opacity-90' : 'text-positive'}>{up != null ? up.toFixed(2) : '--'}</span>
                    <span className="opacity-40">/</span>
                    <span className={picked.includes('current') ? 'opacity-90' : 'text-negative'}>{down != null ? down.toFixed(2) : '--'}</span>
                  </span>
                </button>
                {future.map(w => {
                  const ahead = Math.max(0, Math.floor((w.window_start - nowMs) / 1000))
                  const tooSoon = ahead < 5
                  const hasOrder = alreadyOrderedWins.has(w.window_start)
                  const on = picked.includes(w.window_start)
                  return (
                    <button key={w.window_start}
                      onClick={() => togglePick(w.window_start)}
                      disabled={!w.available || tooSoon}
                      title={tooSoon ? '距开盘不足 5 秒'
                        : hasOrder ? '该窗口已有手动单（下单会被拒）'
                        : `距开盘 ${ahead}s | 涨 ${w.up_price ?? '--'} / 跌 ${w.down_price ?? '--'}（真实报价）`}
                      className={`${chipCls(on)} ${hasOrder ? 'border-warning/60' : ''} flex-col items-center gap-0.5 px-2.5 py-1 leading-tight`}
                    >
                      <span>{fmtHHMM(w.window_start)}
                        <span className="opacity-60">·{Math.round(ahead / 60)}分</span>
                        {hasOrder && <span className="ml-1 text-warning">●</span>}
                      </span>
                      <span className="font-mono text-[10px]">
                        <span className={on ? 'opacity-90' : 'text-positive'}>{w.up_price != null ? w.up_price.toFixed(2) : '--'}</span>
                        <span className="opacity-40">/</span>
                        <span className={on ? 'opacity-90' : 'text-negative'}>{w.down_price != null ? w.down_price.toFixed(2) : '--'}</span>
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* 方向大卡 */}
            <div className="flex gap-3">
              <button onClick={() => setSide('UP')}
                className={`flex-1 py-3 rounded-card border text-center transition ${side === 'UP' ? 'bg-positive-soft text-positive border-positive' : 'bg-card text-ink-55 border-line hover:border-positive/50'}`}
              >
                <div className="text-base font-bold">↗ 涨</div>
                <div className="text-xs font-mono">{up != null ? up.toFixed(3) : '--'}</div>
              </button>
              <button onClick={() => setSide('DOWN')}
                className={`flex-1 py-3 rounded-card border text-center transition ${side === 'DOWN' ? 'bg-negative-soft text-negative border-negative' : 'bg-card text-ink-55 border-line hover:border-negative/50'}`}
              >
                <div className="text-base font-bold">↘ 跌</div>
                <div className="text-xs font-mono">{down != null ? down.toFixed(3) : '--'}</div>
              </button>
            </div>

            {/* 金额 + 预设 */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <label className="text-ink-55 text-xs shrink-0">金额</label>
                <input type="number" min={0.1} max={50} step={0.5} value={amount}
                  onChange={e => setAmount(e.target.value)}
                  className="ds-input ds-input-numeric w-28" />
                <span className="text-[11px] text-ink-55">0.1~50 U / 每窗</span>
                <span className="ml-auto text-[11px] text-ink-55">余额 {balance != null ? balance.toFixed(2) : '--'} U</span>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap text-xs">
                {[1, 2, 5, 10, 20, 50].map(u => (
                  <button key={u} onClick={() => setAmount(String(u))}
                    className={`px-2.5 py-0.5 rounded-pill border ${amtNum === u ? 'bg-brand text-white border-brand' : 'border-line bg-card text-ink-80 hover:border-brand'}`}
                  >{u}U</button>
                ))}
              </div>
              {up != null && Number.isFinite(amtNum) && (
                <div className="text-[11px] text-positive">
                  每窗潜在赢得 ≈ {(amtNum / (side === 'UP' ? up : down ?? 0.5)).toFixed(2)} U
                  {targets.length > 1 && ` × ${targets.length} 窗 ≈ ${((amtNum / (side === 'UP' ? up : down ?? 0.5)) * targets.length).toFixed(2)} U`}
                </div>
              )}
            </div>

            {/* 大额确认（仅 >10U 出现） */}
            {bigAmount && (
              <label className="flex items-center gap-2 p-2 rounded-sm border border-warning bg-warning/10 text-xs text-ink-80">
                <input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />
                大额单（&gt;10 U/窗）确认真实下单
              </label>
            )}

            {/* 一键下单 */}
            <button onClick={handleSubmit} disabled={!canSubmit}
              className={`ds-btn-primary w-full py-3 text-base font-bold disabled:opacity-40 ${busy ? 'cursor-wait' : ''}`}
            >
              {busy ? '下单中…'
                : `一键下单（${side === 'UP' ? '涨' : '跌'} ${amount}U${targets.length > 1 ? ` × ${targets.length} 窗` : ''}）`}
            </button>

            {/* 结果 */}
            {results.length > 0 && (
              <div className="space-y-1">
                {results.map((r, i) => {
                  const ok = r.status === 'FILLED'
                  return (
                    <div key={i} className={`p-2 rounded-sm text-xs ${ok ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'}`}>
                      <span className="font-bold">{ok ? '✓ 成交' : '✗ 失败'}</span>
                      {r.window_start != null && (
                        <span className="ml-1.5 font-mono">{fmtHHMM(Number(r.window_start))}–{r.window_end != null ? fmtHHMM(Number(r.window_end)) : '--'}</span>
                      )}
                      {r.average_price != null && <span className="ml-1.5">@{String(r.average_price)}</span>}
                      {r.filled_shares != null && <span className="ml-1.5">{String(r.filled_shares)}股</span>}
                      {(r.error || r.error_message) != null && <span className="ml-1.5">{String(r.error ?? r.error_message)}</span>}
                    </div>
                  )
                })}
              </div>
            )}

            {/* 复利下单（仅 5m）：选 ≥2 窗，首腿本金，赢则本金+利润滚入下一腿 */}
            {period === '5m' && (
              <div className="space-y-1.5 border-t border-line pt-3">
                <div className="flex items-center gap-2 text-xs">
                  <label className="text-ink-55 shrink-0">每腿执行价上限</label>
                  <input type="number" min={0.5} max={0.99} step={0.01} value={chainGuard}
                    onChange={e => setChainGuard(e.target.value)}
                    className="ds-input ds-input-numeric w-20" />
                  <span className="text-[11px] text-ink-55">≥ 即弃单并终止链</span>
                  <span className="ml-auto text-[11px] text-ink-55">首腿 0.1~10U</span>
                </div>
                <button onClick={handleChain} disabled={!canChain}
                  className="w-full py-2.5 text-sm font-bold rounded-card border border-brand text-brand hover:bg-brand hover:text-white transition disabled:opacity-40"
                >
                  {chainWins.length >= 2
                    ? `复利下单（${side === 'UP' ? '涨' : '跌'} ${amount}U，${chainWins.length} 腿连环）`
                    : '复利下单（先选 ≥2 个窗口）'}
                </button>
                {chainMsg && <div className="text-xs text-ink-80">{chainMsg}</div>}
                {chains.slice(0, 3).map(c => {
                  const legs = (Array.isArray(c.legs) ? c.legs : []) as Record<string, unknown>[]
                  const wins = (Array.isArray(c.windows) ? c.windows : []) as number[]
                  const running = c.status === 'RUNNING'
                  return (
                    <div key={String(c.id)} className="p-2 rounded-sm border border-line text-[11px] space-y-0.5">
                      <div className="flex items-center gap-2">
                        <span className="font-bold">#{String(c.id)}</span>
                        <span>{c.direction === 'UP' ? '涨' : '跌'} {String(c.base_amount)}U</span>
                        <span className={running ? 'text-brand' : 'text-ink-55'}>
                          {running ? '进行中' : c.status === 'DONE' ? '已完成' : '已终止'}
                        </span>
                        {running && (
                          <button className="ml-auto text-negative underline"
                            onClick={() => { if (window.confirm('终止后续腿？已下的腿照常结算（赢单奖金需手动领取）')) api.stopCompound(Number(c.id)).then(loadChains) }}
                          >终止</button>
                        )}
                      </div>
                      {wins.map((ws, i) => {
                        const l = legs[i]
                        const r = l ? String(l.result ?? '') : ''
                        const label = !l ? '待下单' : r === 'WIN' ? '赢' : r === 'LOSS' ? '输' : r === '' ? '等待结算' : r
                        const cls = r === 'WIN' ? 'text-positive' : r === 'LOSS' || r === 'FAILED' ? 'text-negative' : 'text-ink-55'
                        return (
                          <div key={ws} className="font-mono">
                            {fmtHHMM(ws)} {l ? `${Number(l.filled).toFixed(2)}U` : ''}
                            {l && l.shares != null ? ` → ${Number(l.shares).toFixed(2)}股` : ''}
                            <span className={`ml-1.5 ${cls}`}>{label}</span>
                          </div>
                        )
                      })}
                      {c.stop_reason != null && <div className="text-ink-55">{String(c.stop_reason)}</div>}
                    </div>
                  )
                })}
              </div>
            )}

            <p className="text-[11px] text-ink-55">
              真实订单：每窗至多一单，窗口收盘 BTC 涨跌定输赢；平仓/赎回去实盘面板。
            </p>
          </div>
        </>
      )}
    </>
  )
}



// 订单记录卡片（页面主区展示，2026-08-28 用户要求回归页面；
// 2026-08-29：加状态/通道/方向筛选 + 通道中文化与解释 + 分页；
// 2026-08-29：加「目标周期」列——window_start + market_period 展示真正下注的窗口时段；
// 2026-09-08：失败单从主列表剥离，改双 Tab（最近订单 / 失败订单）——
// 失败单 amount_in=0、token_id=""、均价是报价不是成交价，混在主列表里会污染盈亏阅读）
const DEFAULT_ORDERS_PAGE_SIZE = 20

function OrdersCard({ orders, syncing, syncResult, onSyncBinance }: {
  orders: Record<string, unknown>[]
  syncing: boolean
  syncResult: Record<string, unknown> | null
  onSyncBinance: () => void
}) {
  const [orderTab, setOrderTab] = useState<'active' | 'failed'>('active')
  const [fStatus, setFStatus] = useState('ALL')
  const [fChannel, setFChannel] = useState('ALL')
  const [fDirection, setFDirection] = useState('ALL')
  const [fKeyword, setFKeyword] = useState('')
  const [pageSize, setPageSize] = useState<number>(DEFAULT_ORDERS_PAGE_SIZE)
  const [page, setPage] = useState(1)
  const [sortField, setSortField] = useState<'time' | 'amount' | 'pnl'>('time')
  const [sortAsc, setSortAsc] = useState(false)

  // 失败单严格分流：主 Tab 只留非 FAILED（FILLED / PENDING / 未知状态都在），失败 Tab 只留 FAILED
  const activeOrders = orders.filter(o => String(o.status ?? '') !== 'FAILED')
  const failedOrders = orders.filter(o => String(o.status ?? '') === 'FAILED')
  const base = orderTab === 'active' ? activeOrders : failedOrders

  const channels = Array.from(new Set(
    base.map(o => String(o.signal_version ?? '')).filter(v => v && v !== '--'))).sort()
  const kw = fKeyword.trim().toLowerCase()
  const filtered = base.filter(o => {
    if (orderTab === 'active' && fStatus !== 'ALL' && String(o.status ?? '') !== fStatus) return false
    if (fChannel !== 'ALL' && String(o.signal_version ?? '') !== fChannel) return false
    if (fDirection !== 'ALL' && String(o.direction ?? '') !== fDirection) return false
    if (kw) {
      const idStr = String(o.id ?? '').toLowerCase()
      const ver = String(o.signal_version ?? '').toLowerCase()
      const verName = (SIGNAL_INFO[o.signal_version as string]?.name ?? '').toLowerCase()
      const msg = String(o.error_message ?? '').toLowerCase()
      const dir = String(o.direction ?? '').toLowerCase()
      if (!idStr.includes(kw) && !ver.includes(kw) && !verName.includes(kw) && !msg.includes(kw) && !dir.includes(kw)) {
        return false
      }
    }
    return true
  }).sort((a, b) => {
    let diff = 0
    if (sortField === 'time') {
      const ta = a.created_at ? new Date(String(a.created_at)).getTime() : 0
      const tb = b.created_at ? new Date(String(b.created_at)).getTime() : 0
      diff = ta - tb
    } else if (sortField === 'amount') {
      const aa = a.amount_in != null ? Number(a.amount_in) : 0
      const ab = b.amount_in != null ? Number(b.amount_in) : 0
      diff = aa - ab
    } else if (sortField === 'pnl') {
      const pa = typeof a.pnl === 'number' ? (a.pnl as number) : -999999
      const pb = typeof b.pnl === 'number' ? (b.pnl as number) : -999999
      diff = pa - pb
    }
    return sortAsc ? diff : -diff
  })
  const settledOrders = filtered.filter(o => o.settled_at != null)
  const settledCount = settledOrders.length
  const totalPnl = settledOrders.reduce(
    (s, o) => s + (typeof o.pnl === 'number' ? (o.pnl as number) : 0), 0)
  const filterActive = fStatus !== 'ALL' || fChannel !== 'ALL' || fDirection !== 'ALL' || fKeyword !== ''
  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'
  // 分页：筛选变化时回到第 1 页；超出范围时钳位（避免删数据后空白页）
  useEffect(() => { setPage(1) }, [fStatus, fChannel, fDirection, fKeyword, pageSize])
  // 切 Tab 必须清筛选：否则主 Tab 选的 FILLED 会让失败 Tab 直接空白
  useEffect(() => {
    setFStatus('ALL'); setFChannel('ALL'); setFDirection('ALL'); setFKeyword(''); setPage(1)
  }, [orderTab])

  const actualPageSize = pageSize > 0 ? pageSize : Math.max(1, filtered.length)
  const totalPages = Math.max(1, Math.ceil(filtered.length / actualPageSize))
  const safePage = Math.min(page, totalPages)
  const pageRows = filtered.slice((safePage - 1) * actualPageSize, safePage * actualPageSize)

  const toggleSort = (field: 'time' | 'amount' | 'pnl') => {
    if (sortField === field) setSortAsc(!sortAsc)
    else { setSortField(field); setSortAsc(false) }
  }
  const sortIcon = (field: 'time' | 'amount' | 'pnl') => {
    if (sortField !== field) return <span className="text-ink-40 ml-0.5 text-[10px]">⇅</span>
    return <span className="text-brand ml-0.5 text-[10px]">{sortAsc ? '▲' : '▼'}</span>
  }

  const tabBtn = (t: 'active' | 'failed', label: string, count: number) => (
    <button
      onClick={() => setOrderTab(t)}
      className={`px-3 py-1 text-xs font-semibold rounded-sm border transition ${orderTab === t ? 'bg-brand text-white border-brand' : 'bg-card text-ink-80 border-line hover:border-brand'}`}
    >{label} ({count})</button>
  )

  // 两张表共用的 4 个单元格（时间/目标周期/通道/方向）
  const cellTime = (o: Record<string, unknown>) => (
    <td className="py-1.5 pr-2 text-ink-80 whitespace-nowrap num">
      {o.created_at ? new Date(String(o.created_at)).toLocaleString() : '--'}
    </td>
  )
  const cellPeriod = (o: Record<string, unknown>) => (
    <td className="py-1.5 pr-2 whitespace-nowrap">
      {typeof o.window_start === 'number' ? (() => {
        const ws = o.window_start as number
        const durMin = o.market_period === '15m' ? 15 : 5
        const hhmm = (ms: number) => new Date(ms).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
        return (
          <span className="font-mono text-ink-80" title={`下注窗口：${new Date(ws).toLocaleString()} ~ ${new Date(ws + durMin * 60_000).toLocaleString()}`}>
            {hhmm(ws)}–{hhmm(ws + durMin * 60_000)}
            <span className="ml-1 px-1 rounded-pill bg-sunken text-ink-55">{durMin}m</span>
          </span>
        )
      })() : <span className="text-ink-55">--</span>}
    </td>
  )
  const cellChannel = (o: Record<string, unknown>) => {
    const ver = String(o.signal_version ?? '')
    const info = SIGNAL_INFO[ver]
    return (
      <td className="py-1.5 pr-2">
        {info ? (
          <span className="inline-flex items-center gap-1">
            <span className="text-ink-95">{info.name}</span>
            {info.retired && (
              <span
                className="px-1 py-0.5 rounded-pill text-[9px] font-bold bg-sunken text-ink-55 border border-line shrink-0"
                title="该通道已于 2026-09-04 永久退役（不再开火）；本行为历史订单，保留可追溯"
              >已退役</span>
            )}
            <HelpHint text={`${ver}（实盘订单${info.retired ? '·通道已退役' : ''}）：${info.desc}`} />
          </span>
        ) : (
          <span className="font-mono text-ink-80">{ver || '--'}</span>
        )}
      </td>
    )
  }
  const cellDirection = (o: Record<string, unknown>) => (
    <td className="py-1.5 pr-2">
      {o.direction === 'UP'
        ? <span className="px-1.5 py-0.5 rounded-pill font-bold bg-positive-soft text-positive">UP</span>
        : o.direction === 'DOWN'
          ? <span className="px-1.5 py-0.5 rounded-pill font-bold bg-negative-soft text-negative">DOWN</span>
          : <span className="text-ink-55">--</span>}
    </td>
  )
  // 报价单元格：price_kind==='quote' 时必须挂黄色「报价」徽章，
  // 否则失败单的委托价会被误读成「按这个价买到了」（P2c 口径）
  const cellQuotePrice = (o: Record<string, unknown>) => (
    <td className="py-1.5 pr-2 font-mono">
      {o.average_price != null ? (
        <span className="inline-flex items-center gap-1">
          {String(o.average_price)}
          {o.price_kind === 'quote' && (
            <span
              className="px-1 py-0.5 rounded-pill text-[9px] font-bold bg-warning-soft text-warning border border-warning"
              title="报价/委托价，非成交均价：本单未成交（FOK 未吃满/护栏弃单/待对账），不能用来核对盈亏"
            >报价</span>
          )}
        </span>
      ) : '--'}
    </td>
  )

  const renderPaginationButtons = () => {
    if (totalPages <= 1) return null
    const pages: (number | string)[] = []
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i)
    } else {
      pages.push(1)
      if (safePage > 3) pages.push('...')
      const start = Math.max(2, safePage - 1)
      const end = Math.min(totalPages - 1, safePage + 1)
      for (let i = start; i <= end; i++) pages.push(i)
      if (safePage < totalPages - 2) pages.push('...')
      pages.push(totalPages)
    }
    return pages.map((p, idx) => {
      if (typeof p === 'string') {
        return <span key={`ellipsis-${idx}`} className="px-1 text-ink-40">…</span>
      }
      return (
        <button
          key={p} onClick={() => setPage(p)}
          className={`px-2 py-0.5 rounded-pill border text-xs ${p === safePage ? 'bg-brand text-white border-brand font-semibold' : 'bg-card text-ink-80 border-line hover:border-brand'}`}
        >{p}</button>
      )
    })
  }

  const pager = (
    <div className="flex items-center justify-between gap-2 mt-2 text-xs flex-wrap">
      <div className="flex items-center gap-1 text-ink-55">
        <span>每页</span>
        <select
          value={pageSize}
          onChange={e => setPageSize(Number(e.target.value))}
          className={selectCls}
        >
          <option value={20}>20 条</option>
          <option value={50}>50 条</option>
          <option value={100}>100 条</option>
          <option value={0}>全部显示 ({filtered.length} 条)</option>
        </select>
        <span>共 {filtered.length} 单 / {totalPages} 页</span>
      </div>
      {pageSize > 0 && totalPages > 1 && (
        <div className="flex items-center gap-1">
          <button
            disabled={safePage <= 1} onClick={() => setPage(safePage - 1)}
            className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 disabled:opacity-40 hover:border-brand"
          >上一页</button>
          {renderPaginationButtons()}
          <button
            disabled={safePage >= totalPages} onClick={() => setPage(safePage + 1)}
            className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 disabled:opacity-40 hover:border-brand"
          >下一页</button>
        </div>
      )}
    </div>
  )

  return (
    <Card title="订单记录">
      <div className="flex items-center gap-2 mb-2">
        {tabBtn('active', '最近订单', activeOrders.length)}
        {tabBtn('failed', '失败订单', failedOrders.length)}
        <HelpHint text="失败单（FAILED）已从主列表剥离：这类单未提交成功或未成交，amount_in=0、均价只是委托报价，混在一起会污染盈亏阅读" />
      </div>
      <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
        <div className="flex items-center gap-1.5 flex-wrap">
          {orderTab === 'active' && (
            <select value={fStatus} onChange={e => setFStatus(e.target.value)} className={selectCls} title="按订单状态筛选">
              <option value="ALL">全部状态</option>
              <option value="FILLED">已成交 FILLED</option>
              <option value="PENDING">待定 PENDING</option>
            </select>
          )}
          <select value={fChannel} onChange={e => setFChannel(e.target.value)} className={selectCls} title="按信号通道筛选">
            <option value="ALL">全部通道</option>
            {channels.map(v => (
              <option key={v} value={v}>{SIGNAL_INFO[v]?.name ?? v}</option>
            ))}
          </select>
          <select value={fDirection} onChange={e => setFDirection(e.target.value)} className={selectCls} title="按押注方向筛选">
            <option value="ALL">全部方向</option>
            <option value="DOWN">DOWN 看跌</option>
            <option value="UP">UP 看涨</option>
          </select>
          <input
            type="text"
            placeholder="搜索通道/单号/错误信息..."
            value={fKeyword}
            onChange={e => setFKeyword(e.target.value)}
            className={`${selectCls} w-36`}
          />
          {filterActive && (
            <button
              onClick={() => { setFStatus('ALL'); setFChannel('ALL'); setFDirection('ALL'); setFKeyword('') }}
              className="px-1.5 py-0.5 text-xs text-brand hover:underline"
            >清除筛选</button>
          )}
          <span className="text-xs text-ink-55">{filtered.length}/{base.length} 条（每 15s 自动刷新）</span>
        </div>
        <div className="flex items-center gap-2">
          {syncResult && (
            <span className={`text-xs px-2 py-0.5 rounded-pill ${syncResult.error ? 'bg-negative-soft text-negative' : 'bg-positive-soft text-positive'}`}>
              {syncResult.error
                ? String(syncResult.error)
                : `币安侧 ${String(syncResult.binance_orders ?? '?')} 单，已同步 ${String(syncResult.synced ?? 0)} 单`}
            </span>
          )}
          <button
            onClick={onSyncBinance} disabled={syncing}
            className="ds-btn-dark px-3 py-1 text-xs font-semibold disabled:opacity-50"
          >{syncing ? '对账中…' : '对账（同步币安）'}</button>
          <button
            onClick={() => {
              api.postTestWechatNotify()
                .then((r: Record<string, unknown>) => {
                  alert(r?.message ? String(r.message) : '测试推送已派发，请查收微信！')
                })
                .catch((e: unknown) => alert('测试推送请求失败: ' + String(e)))
            }}
            className="px-2.5 py-1 text-xs font-semibold border border-line rounded-pill bg-card text-ink-80 hover:border-brand hover:text-brand transition-colors"
            title="点击向已配置的 WxPusher / 企业微信发送一条测试消息"
          >🔔 测试微信通知</button>
        </div>
      </div>
      {filtered.length === 0 ? (
        <p className="text-sm text-ink-55">
          {base.length === 0
            ? (orderTab === 'active' ? '暂无订单记录' : '暂无失败订单')
            : (orderTab === 'active' ? '当前筛选条件下无订单' : '当前筛选条件下无失败订单')}
        </p>
      ) : orderTab === 'active' ? (
        <>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-ink-55 border-b border-line-soft select-none">
              <th className="py-1 pr-2 cursor-pointer hover:text-brand" onClick={() => toggleSort('time')}>
                时间{sortIcon('time')}
              </th>
              <th className="py-1 pr-2" title="本单真正下注的市场周期时段（window_start 起，5m/15m 窗口）">目标周期</th>
              <th className="py-1 pr-2">通道</th>
              <th className="py-1 pr-2">方向</th>
              <th className="py-1 pr-2">状态</th>
              <th className="py-1 pr-2">结果</th>
              <th className="py-1 pr-2" title="币安实际成交均价；标「报价」的是下单时的报价/委托价（PENDING 未成交/护栏弃单时 filledUsdtAmount=0 但币安仍回 price），不是成交价，不能用来核对盈亏">均价</th>
              <th className="py-1 pr-2 cursor-pointer hover:text-brand" onClick={() => toggleSort('amount')}>
                金额 (USDT){sortIcon('amount')}
              </th>
              <th className="py-1 pr-2 cursor-pointer hover:text-brand" onClick={() => toggleSort('pnl')}>
                盈亏{sortIcon('pnl')}
              </th>
              <th className="py-1">说明</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map(o => {
              return (
              <tr key={String(o.id)} className="border-b border-line-soft">
                {cellTime(o)}
                {cellPeriod(o)}
                {cellChannel(o)}
                {cellDirection(o)}
                <td className="py-1.5 pr-2">
                  <span className={`px-1.5 py-0.5 rounded-pill font-bold ${o.status === 'FILLED' ? 'bg-positive-soft text-positive' : o.status === 'FAILED' ? 'bg-negative-soft text-negative' : 'bg-sunken text-ink-80'}`}>
                    {o.status === 'FILLED' ? '已成交' : o.status === 'FAILED' ? '失败' : o.status === 'PENDING' ? '待定' : String(o.status)}
                  </span>
                </td>
                <td className="py-1.5 pr-2">
                  {o.win === true
                    ? <span className="px-1.5 py-0.5 rounded-pill font-bold bg-positive-soft text-positive">赢</span>
                    : o.win === false
                      ? <span className="px-1.5 py-0.5 rounded-pill font-bold bg-negative-soft text-negative">输</span>
                      : o.settle_outcome != null
                        ? <span className="px-1.5 py-0.5 rounded-pill font-bold bg-sunken text-ink-80">{String(o.settle_outcome)}</span>
                        : <span className="text-ink-55">--</span>}
                </td>
                {cellQuotePrice(o)}
                <td className="py-1.5 pr-2 font-mono">
                  {o.amount_in != null && Number(o.amount_in) > 0 ? (Number(o.amount_in) / 1e18).toFixed(2) : '--'}
                </td>
                <td className={`py-1.5 pr-2 font-mono ${typeof o.pnl === 'number' ? ((o.pnl as number) >= 0 ? 'text-positive' : 'text-negative') : 'text-ink-55'}`}>
                  {typeof o.pnl === 'number'
                    ? `${(o.pnl as number) >= 0 ? '+' : ''}${(o.pnl as number).toFixed(2)}`
                    : '--'}
                </td>
                <td className="py-1.5 text-ink-55">{String(o.error_message ?? '')}</td>
              </tr>
              )
            })}
          </tbody>
          {settledCount > 0 && (
            <tfoot>
              <tr className="border-t border-line-soft text-ink-80">
                <td colSpan={8} className="py-1.5">已结算 {settledCount} 单（本地估算口径）</td>
                <td className={`py-1.5 font-mono font-bold ${totalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                  {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
                </td>
                <td className="py-1.5 text-ink-55">USDT</td>
              </tr>
            </tfoot>
          )}
        </table>
        {pager}
        </>
      ) : (
        <>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-ink-55 border-b border-line-soft select-none">
              <th className="py-1 pr-2 cursor-pointer hover:text-brand" onClick={() => toggleSort('time')}>
                时间{sortIcon('time')}
              </th>
              <th className="py-1 pr-2" title="本单真正下注的市场周期时段（window_start 起，5m/15m 窗口）">目标周期</th>
              <th className="py-1 pr-2">通道</th>
              <th className="py-1 pr-2">方向</th>
              <th className="py-1 pr-2" title="下单时的报价/委托价（失败单 FOK 未成交，filledUsdtAmount=0 但币安仍回 price），不是成交价">触发报价</th>
              <th className="py-1 pr-2 cursor-pointer hover:text-brand" onClick={() => toggleSort('amount')}>
                计划金额{sortIcon('amount')}
              </th>
              <th className="py-1">
                <span className="inline-flex items-center gap-1">失败原因
                  <HelpHint text="error_message 可能是泛化文案（如「下单失败」），真实拒单原因（护栏弃单 / 币安错误码 -1102 等）要看后端 place_order 层日志" />
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map(o => {
              const msg = String(o.error_message ?? '')
              return (
                <tr key={String(o.id)} className="border-b border-line-soft">
                  {cellTime(o)}
                  {cellPeriod(o)}
                  {cellChannel(o)}
                  {cellDirection(o)}
                  {cellQuotePrice(o)}
                  <td className="py-1.5 pr-2 font-mono">
                    {o.amount_in != null && Number(o.amount_in) > 0 ? (Number(o.amount_in) / 1e18).toFixed(2) : '--'}
                  </td>
                  <td className="py-1.5 text-ink-55 max-w-[320px] truncate" title={msg || undefined}>
                    {msg || '--'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {pager}
        </>
      )}
    </Card>
  )
}
// 资金变化面板（2026-08-29 用户要求）：纯前端从订单派生流水——
// 后端划转（transfer-in/out）不入库，故只含下注流出与结算回流。
// 口径：流出=成交单 amount_in（按下单时间）；回流=已结算赢单 amount_in+pnl（按结算时间）。
interface FlowEvent {
  ts: number
  dir: '流出' | '回流'
  amount: number
  detail: string
}
const FUND_PERIODS = [
  { v: '24h', label: '24小时', ms: 24 * 3600_000 },
  { v: '7d', label: '7天', ms: 7 * 86_400_000 },
  { v: '30d', label: '30天', ms: 30 * 86_400_000 },
  { v: 'all', label: '全部', ms: 0 },
] as const

function FundsFlowCard({ orders }: { orders: Record<string, unknown>[] }) {
  const [period, setPeriod] = useState<string>('7d')
  // 全量流水（新接口不受 recent 的 100 条上限）；拉取成功后替代 orders prop，
  // 失败降级用 prop（最近 100 条）保证面板可用。随抽屉常驻挂载 30s 轮询。
  const [fundOrders, setFundOrders] = useState<Record<string, unknown>[]>([])
  useEffect(() => {
    let alive = true
    const pull = () => api.getFundFlow()
      .then((d: Record<string, unknown>) => {
        if (alive && Array.isArray(d?.orders) && (d.orders as unknown[]).length > 0) {
          setFundOrders(d.orders as Record<string, unknown>[])
        }
      })
      .catch(() => {})
    pull()
    const timer = setInterval(pull, 30_000)
    return () => { alive = false; clearInterval(timer) }
  }, [])
  // 降级路径与后端同口径排除 SELL 行（S#FundsFlow：recent 含平仓 SELL 记录行，
  // 不过滤会把卖出所得记成下注流出，资金面板 phantom 流出）
  const src = fundOrders.length > 0 ? fundOrders : orders.filter(o => o.side !== 'SELL')

  const events = useMemo(() => {
    const evs: FlowEvent[] = []
    for (const o of src) {
      const amtWei = Number(o.amount_in ?? 0)
      if (!(amtWei > 0)) continue  // 失败/未成交单无资金占用（含待定单）
      const amt = amtWei / 1e18
      const ver = String(o.signal_version ?? '--')
      const name = SIGNAL_INFO[ver]?.name ?? ver
      const dirStr = String(o.direction ?? '')
      const created = Date.parse(String(o.created_at ?? ''))
      if (o.status === 'FILLED' && !Number.isNaN(created)) {
        evs.push({ ts: created, dir: '流出', amount: amt, detail: `${name} 下注 ${dirStr}` })
      }
      const settled = Date.parse(String(o.settled_at ?? ''))
      if (o.win === true && !Number.isNaN(settled)) {
        const pnl = typeof o.pnl === 'number' ? (o.pnl as number) : 0
        evs.push({ ts: settled, dir: '回流', amount: amt + pnl, detail: `${name} 结算赢（含盈亏 ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}）` })
      }
    }
    return evs.sort((a, b) => a.ts - b.ts)
  }, [src])

  const p = FUND_PERIODS.find(x => x.v === period) ?? FUND_PERIODS[1]
  const cutoff = p.ms > 0 ? Date.now() - p.ms : 0
  const inRange = events.filter(e => e.ts >= cutoff)
  const outSum = inRange.filter(e => e.dir === '流出').reduce((s, e) => s + e.amount, 0)
  const inSum = inRange.filter(e => e.dir === '回流').reduce((s, e) => s + e.amount, 0)
  const net = inSum - outSum
  const tradeCount = inRange.filter(e => e.dir === '流出').length
  // 在途资金（未结算成交单，不受时间筛选影响）
  const pendingOpen = src.filter(o => o.status === 'FILLED' && o.settled_at == null && Number(o.amount_in ?? 0) > 0)
  const pendingAmt = pendingOpen.reduce((s, o) => s + Number(o.amount_in) / 1e18, 0)

  // 按本地日聚合净流入（柱）+ 累计净盈亏（折线）
  const byDay = new Map<string, number>()
  for (const e of inRange) {
    const key = new Date(e.ts).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
    byDay.set(key, (byDay.get(key) ?? 0) + (e.dir === '回流' ? e.amount : -e.amount))
  }
  let acc = 0
  const daily = Array.from(byDay.entries()).map(([date, netAmt]) => {
    acc += netAmt
    return { date, '当日净流入': +netAmt.toFixed(3), '累计净盈亏': +acc.toFixed(3) }
  })

  // 结算表现统计（口径：本期内已结算成交单；pnl 输=-投入/赢=盈利/NOISE=0）
  const settledInPeriod = src.filter(o => {
    if (o.status !== 'FILLED' || !(Number(o.amount_in ?? 0) > 0)) return false
    const ts = Date.parse(String(o.settled_at ?? ''))
    return !Number.isNaN(ts) && ts >= cutoff
  })
  const winCount = settledInPeriod.filter(o => o.win === true).length
  const loseCount = settledInPeriod.filter(o => o.win === false).length
  const winRate = (winCount + loseCount) > 0 ? winCount / (winCount + loseCount) : null
  const totalPnl = settledInPeriod.reduce(
    (s, o) => s + (typeof o.pnl === 'number' ? (o.pnl as number) : 0), 0)
  const avgPnl = settledInPeriod.length > 0 ? totalPnl / settledInPeriod.length : 0
  // 累计曲线最大回撤（按事件时点展开；起点为 0）
  let run = 0, peak = 0, maxDD = 0
  for (const e of inRange) {
    run += e.dir === '回流' ? e.amount : -e.amount
    if (run > peak) peak = run
    if (peak - run > maxDD) maxDD = peak - run
  }
  const bestDay = daily.length ? daily.reduce((a, b) => (b['当日净流入'] > a['当日净流入'] ? b : a)) : null
  const worstDay = daily.length ? daily.reduce((a, b) => (b['当日净流入'] < a['当日净流入'] ? b : a)) : null

  const rows = [...inRange].reverse()
  const statBox = 'flex-1 min-w-[110px] rounded-card border border-line bg-card px-3 py-2'

  return (
    <Card title="资金变化（订单流水）">
      <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
        <div className="flex items-center gap-1">
          {FUND_PERIODS.map(x => (
            <button
              key={x.v}
              onClick={() => setPeriod(x.v)}
              className={`px-2.5 py-0.5 text-xs rounded-sm border transition ${period === x.v ? 'bg-brand text-white border-brand' : 'bg-card text-ink-80 border-line hover:border-brand'}`}
            >{x.label}</button>
          ))}
        </div>
        <span className="text-[10px] text-ink-55">划转记录不入库，此处仅含下注/结算流水（回流=本金+盈亏）{fundOrders.length > 0 ? ` · 全量 ${fundOrders.length} 单` : ''} · 口径=全量订单（含退役通道历史与 NOISE·EXPIRED），与「盈利趋势」面板不同，勿直接对比</span>
      </div>
      <div className="flex flex-wrap gap-2 mb-3 text-xs">
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">下注流出</div>
          <div className="text-base font-bold font-mono text-negative">-{outSum.toFixed(2)}</div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">结算回流</div>
          <div className="text-base font-bold font-mono text-positive">+{inSum.toFixed(2)}</div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">净盈亏</div>
          <div className={`text-base font-bold font-mono ${net >= 0 ? 'text-positive' : 'text-negative'}`}>
            {net >= 0 ? '+' : ''}{net.toFixed(2)}
          </div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">下注笔数</div>
          <div className="text-base font-bold font-mono text-ink-95">{tradeCount}</div>
        </div>
        {pendingOpen.length > 0 && (
          <div className={statBox}>
            <div className="text-ink-55 mb-0.5">在途未结算</div>
            <div className="text-base font-bold font-mono text-warning">{pendingAmt.toFixed(2)}（{pendingOpen.length} 单）</div>
          </div>
        )}
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">胜率（已结算）</div>
          <div className="text-base font-bold font-mono text-ink-95">
            {winRate === null ? '--' : `${(winRate * 100).toFixed(0)}%`}
            <span className="text-[10px] font-normal text-ink-55 ml-1">{winCount}胜/{loseCount}负</span>
          </div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">已实现盈亏</div>
          <div className={`text-base font-bold font-mono ${totalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
            {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
          </div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">单笔均盈亏</div>
          <div className={`text-base font-bold font-mono ${avgPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
            {settledInPeriod.length === 0 ? '--' : `${avgPnl >= 0 ? '+' : ''}${avgPnl.toFixed(3)}`}
          </div>
        </div>
        <div className={statBox}>
          <div className="text-ink-55 mb-0.5">最大回撤</div>
          <div className="text-base font-bold font-mono text-negative">
            {inRange.length === 0 ? '--' : `-${maxDD.toFixed(2)}`}
          </div>
        </div>
      </div>
      {daily.length > 0 && (
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={daily} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--line-soft)" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
            <YAxis yAxisId="net" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={48} />
            <YAxis yAxisId="cum" orientation="right" tick={{ fontSize: 10 }} stroke="var(--chart-1)" width={48} />
            <Tooltip
              contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
              formatter={(v, name) => [`${typeof v === 'number' && v >= 0 ? '+' : ''}${Number(v).toFixed(2)} USDT`, String(name)]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <ReferenceLine yAxisId="net" y={0} stroke="var(--line)" />
            <Bar yAxisId="net" dataKey="当日净流入" radius={[3, 3, 0, 0]} isAnimationActive={false}>
              {daily.map((d, i) => <Cell key={i} fill={d['当日净流入'] >= 0 ? 'var(--positive)' : 'var(--negative)'} fillOpacity={0.55} />)}
            </Bar>
            <Line yAxisId="cum" type="monotone" dataKey="累计净盈亏" stroke="var(--chart-1)" strokeWidth={2}
              dot={{ r: 2 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      )}
      {bestDay && worstDay && daily.length > 1 && (
        <div className="flex flex-wrap gap-3 text-[11px] text-ink-55 mt-1 mb-1">
          <span>最佳单日：<span className="font-mono text-positive">{bestDay.date} +{bestDay['当日净流入'].toFixed(2)}</span></span>
          <span>最差单日：<span className="font-mono text-negative">{worstDay.date} {worstDay['当日净流入'].toFixed(2)}</span></span>
          <span>已结算 {settledInPeriod.length} 单</span>
        </div>
      )}
      {rows.length === 0 ? (
        <p className="text-sm text-ink-55 mt-2">当前时间范围内无资金流水</p>
      ) : (
        <div className="overflow-x-auto max-h-60 overflow-y-auto mt-2">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-card">
              <tr className="text-left text-ink-55 border-b border-line-soft">
                <th className="py-1 pr-2">时间</th>
                <th className="py-1 pr-2">类型</th>
                <th className="py-1 pr-2">说明</th>
                <th className="py-1 pr-2 text-right">金额 (USDT)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => (
                <tr key={i} className="border-b border-line-soft">
                  <td className="py-1 pr-2 text-ink-80 whitespace-nowrap num">{new Date(e.ts).toLocaleString()}</td>
                  <td className="py-1 pr-2">
                    <span className={`px-1.5 py-0.5 rounded-pill font-bold ${e.dir === '流出' ? 'bg-negative-soft text-negative' : 'bg-positive-soft text-positive'}`}>{e.dir}</span>
                  </td>
                  <td className="py-1 pr-2 text-ink-55">{e.detail}</td>
                  <td className={`py-1 pr-2 text-right font-mono font-bold ${e.dir === '流出' ? 'text-negative' : 'text-positive'}`}>
                    {e.dir === '流出' ? '-' : '+'}{e.amount.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

// ============================================================
// Phase 0 实盘表现诊断：组合决策 / 通道诊断 / 风险归因
// ============================================================

interface PnlPoint { n: number; t: number; pnl: number; cum: number; cost?: number; cum_cost?: number; cum_roi?: number; win: boolean }
interface PnlChannel { channel: string; display_name: string; family: string | null; market_period: string | null; enabled: boolean; settled_count: number; win_count: number; win_rate: number | null; total_pnl: number; total_cost: number; roi: number | null; points: PnlPoint[] }
interface PnlCurveData { channels: PnlChannel[]; total: { settled_count: number; total_pnl: number; total_cost: number; win_rate: number | null; roi: number | null } }
interface PerformanceStat { window_size: number; n: number; wins: number; full_window: boolean; win_rate: number | null; wilson_95: [number, number] | null; jeffreys_95: [number, number] | null }
interface PerformanceEvStat { window_size: number; n: number; full_window: boolean; realized_ev: number | null; pnl: number | null; actual_cost: number | null }
interface ChannelBenchmark { channel: string; win_rate: number; source_ref: string; method_version: string; quality: string; sample_size: number | null }
// 信号真相：剔除动态下单金额后的等权信号质量 vs 金额加权的钱包结果（与后端 signal_truth 同口径）
interface SignalTruth {
  signal_n: number; signal_ev: number | null; signal_ev_ci95: [number, number] | null; capital_roi: number | null; sizing_distortion: number | null
  mean_break_even: number | null; edge: number | null; edge_ci95: [number, number] | null; edge_n: number
  stake_mean: number | null; stake_median: number | null; stake_max: number | null; top3_stake_share: number | null; equal_stake_pnl: number | null
}
interface SignalEvStat { window_size: number; n: number; full_window: boolean; signal_ev: number | null }
interface PerformanceChannel {
  channel: string; display_name: string; family: string | null; market_period: string | null; enabled: boolean; order_count: number; win_count: number; total_cost: number; total_pnl: number; roi: number | null; benchmark: ChannelBenchmark | null
  latest: { rolling20: PerformanceStat; rolling50: PerformanceStat; ewma: number | null; rolling_realized_ev_20: PerformanceEvStat; rolling_realized_ev_50: PerformanceEvStat; rolling_signal_ev_20?: SignalEvStat; rolling_signal_ev_50?: SignalEvStat; mean_break_even_20: number | null; mean_break_even_50: number | null; benchmark_gap_20: number | null; benchmark_gap_50: number | null }
  break_even_coverage: { available_count: number; order_count: number; ratio: number }
  signal_truth?: SignalTruth; max_exec_price?: number | null
}
interface PortfolioSeriesPoint { t: number; order_count: number; net_pnl: number | null; gross_cost: number | null; cumulative_pnl: number; rolling_realized_ev_20: number | null; rolling_realized_ev_50: number | null; settlement_conflict: boolean }
interface LivePerformanceSummary {
  schema_version: string; metric_definitions_version: string; as_of: number
  coverage: { order_count: number; unique_window_count: number; window_key_unavailable_count: number; settlement_conflict_count: number; break_even_available_count: number; assessment_linked_count: number; is_truncated: boolean }
  portfolio: { total_cost: number; total_pnl: number; roi: number | null; order_win_rate: { point_estimate: number | null; rolling: Record<'20' | '50', PerformanceStat | null> }; market_window_win_rate: { point_estimate: number | null; rolling: Record<'20' | '50', PerformanceStat | null> }; rolling_realized_ev: Record<'20' | '50', PerformanceEvStat | null>; signal_truth?: SignalTruth; quote_edge: { display_name: string; metric_semantics: string; is_calibration_metric: boolean; mean: number | null } }
  duplicate_exposure: { window_count: number; duplicate_window_count: number; duplicate_window_ratio: number | null; duplicate_overlap_cost: number; duplicate_investment_ratio: number | null; max_single_window_cost: number; max_channels_per_window: number; same_direction_overlap_count: number; opposite_direction_overlap_count: number; explicit_exclusive_conflict_count: number }
  channels: PerformanceChannel[]; portfolio_series: PortfolioSeriesPoint[]; warnings: string[]
}
type DiagnosticSegmentBy = 'quote' | 'exec_price' | 'trigger_offset' | 'policy_version' | 'stake' | 'trend_4h' | 'trend_24h' | 'volatility' | 'deployment'
interface DiagnosticPoint { id: number | string; t: number; pnl: number | null; cost: number | null; win: boolean | null; direction: string | null; settle_outcome: string | null; policy_version: string | null; trigger_offset_seconds: number | null; cumulative_pnl: number | null; cumulative_cost: number | null; rolling: Record<'20' | '50', PerformanceStat>; ewma_win_rate: number | null; rolling_realized_ev: Record<'20' | '50', PerformanceEvStat>; rolling_signal_ev?: Record<'20' | '50', SignalEvStat>; unit_return?: number | null; break_even_probability: number | null; rolling_mean_break_even_probability: Record<'20' | '50', number | null>; benchmark_win_rate: number | null; benchmark_gap: Record<'20' | '50', number | null> }
interface DiagnosticSegment {
  segment: string; n: number; wins: number; win_rate: number | null; win_rate_wilson_95?: [number, number] | null; total_cost: number; total_pnl: number; realized_ev: number | null
  signal_ev?: number | null; signal_ev_ci95?: [number, number] | null; sizing_distortion?: number | null; mean_break_even?: number | null; edge?: number | null; edge_ci95?: [number, number] | null
  stake_mean?: number | null; stake_max?: number | null; low_sample?: boolean; unique_window_count: number; data_quality: string | null
}
interface LiveChannelDiagnostics { channel: { channel: string; display_name: string; family: string | null; market_period: string | null; enabled: boolean; max_exec_price?: number | null }; benchmark: ChannelBenchmark | null; signal_truth?: SignalTruth; coverage: Record<string, number | boolean | null>; series: DiagnosticPoint[]; segments: DiagnosticSegment[]; warnings: string[] }

type PerformanceView = 'portfolio' | 'channel' | 'risk'
type PortfolioMetric = 'ev' | 'pnl' | 'edge'
type ChannelSortField = 'channel' | 'order_count' | 'ev20' | 'signal_ev' | 'distortion' | 'win20' | 'benchmark' | 'break_even' | 'total_pnl'
type IntervalKind = 'wilson' | 'beta'
const SEGMENT_OPTIONS: { key: DiagnosticSegmentBy; label: string }[] = [{ key: 'quote', label: '保本率档(0.05)' }, { key: 'exec_price', label: '确认成交价档(0.05)' }, { key: 'trigger_offset', label: '触发时点' }, { key: 'policy_version', label: '策略版本' }, { key: 'stake', label: '下单金额档' }, { key: 'trend_4h', label: '4h趋势' }, { key: 'trend_24h', label: '24h趋势' }, { key: 'volatility', label: '波动率' }, { key: 'deployment', label: '部署版本' }]
const performancePct = (v: number | null | undefined, digits = 1) => v == null ? '--' : `${(v * 100).toFixed(digits)}%`
const performanceSigned = (v: number | null | undefined, suffix = '', digits = 2) => v == null ? '--' : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}${suffix}`
const performanceTime = (t: number) => new Date(t).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
// 金额扭曲 Δ = 资金ROI − 信号EV：|Δ|≥10% 红、≥5% 黄（负=大注输得多，正=大注赢得多）
const distortionClass = (v: number | null | undefined) => v == null ? 'text-ink-55' : Math.abs(v) >= 0.10 ? 'text-negative font-bold' : Math.abs(v) >= 0.05 ? 'text-warning' : 'text-ink-55'
const signClass = (v: number | null | undefined) => v == null ? '' : v >= 0 ? 'text-positive' : 'text-negative'
const ciText = (ci: [number, number] | null | undefined) => ci ? `[${performancePct(ci[0], 0)}, ${performancePct(ci[1], 0)}]` : '--'
const SIGNAL_TRUTH_HINTS = {
  signal: '信号EV（等权）= 每笔 PnL/投入 的简单平均，等价于每单固定押 1U 的期望，剔除动态下单金额的影响。括号内为均值 95% 正态区间。',
  capital: '资金ROI（加权）= 总 PnL / 总投入，即钱包实际结果；大注单对它的影响远大于小注单。',
  distortion: '金额扭曲 Δ = 资金ROI − 信号EV。负值：大注输得多（加注毁掉了信号价值）；正值：大注赢得多。|Δ|≥10% 标红。',
  edge: 'Edge = 平均(实际胜负 − 逐笔保本率)。保本率按成交成本/份额推导；正值=命中率超过赔率要求。',
  equal: '等额反事实 = 信号EV × 笔数 × 中位注码：若每笔都下中位金额，理论上的累计 PnL。',
}

// 信号真相条：通道诊断/组合视图共用
function SignalTruthStrip({ truth, realPnl }: { truth: SignalTruth; realPnl?: number | null }) {
  const box = 'rounded-card border border-line bg-card px-3 py-2 min-w-0'
  return <div className="grid grid-cols-2 md:grid-cols-5 gap-2 mb-3">
    <div className={box}><div className="text-xs text-ink-55">信号EV（等权） <HelpHint text={SIGNAL_TRUTH_HINTS.signal} /></div><div className={`font-mono font-bold text-lg ${signClass(truth.signal_ev)}`}>{performancePct(truth.signal_ev)}</div><div className="text-[10px] text-ink-55">95% {ciText(truth.signal_ev_ci95)} · n={truth.signal_n}</div></div>
    <div className={box}><div className="text-xs text-ink-55">资金ROI（加权） <HelpHint text={SIGNAL_TRUTH_HINTS.capital} /></div><div className={`font-mono font-bold text-lg ${signClass(truth.capital_roi)}`}>{performancePct(truth.capital_roi)}</div><div className="text-[10px] text-ink-55">真实 PnL {performanceSigned(realPnl, ' U')}</div></div>
    <div className={box}><div className="text-xs text-ink-55">金额扭曲 Δ <HelpHint text={SIGNAL_TRUTH_HINTS.distortion} /></div><div className={`font-mono text-lg ${distortionClass(truth.sizing_distortion)}`}>{performanceSigned(truth.sizing_distortion == null ? null : truth.sizing_distortion * 100, '%', 1)}</div><div className="text-[10px] text-ink-55">前3大单占投入 {performancePct(truth.top3_stake_share)}</div></div>
    <div className={box}><div className="text-xs text-ink-55">Edge（胜率−保本率） <HelpHint text={SIGNAL_TRUTH_HINTS.edge} /></div><div className={`font-mono font-bold text-lg ${signClass(truth.edge)}`}>{performanceSigned(truth.edge == null ? null : truth.edge * 100, '%', 1)}</div><div className="text-[10px] text-ink-55">95% {ciText(truth.edge_ci95)} · 保本 {performancePct(truth.mean_break_even)}</div></div>
    <div className={box}><div className="text-xs text-ink-55">等额下注反事实 <HelpHint text={SIGNAL_TRUTH_HINTS.equal} /></div><div className={`font-mono font-bold text-lg ${signClass(truth.equal_stake_pnl)}`}>{performanceSigned(truth.equal_stake_pnl, ' U')}</div><div className="text-[10px] text-ink-55">注码 中位 {performanceSigned(truth.stake_median, 'U', 1).replace('+', '')} / 最大 {performanceSigned(truth.stake_max, 'U', 1).replace('+', '')}</div></div>
  </div>
}

function PnlCurveCard({ active }: { active: boolean }) {
  const [summary, setSummary] = useState<LivePerformanceSummary | null>(null)
  const [diagnostics, setDiagnostics] = useState<LiveChannelDiagnostics | null>(null)
  const [view, setView] = useState<PerformanceView>('portfolio')
  const [metric, setMetric] = useState<PortfolioMetric>('ev')
  const [selectedChannel, setSelectedChannel] = useState('')
  const [onlyEnabled, setOnlyEnabled] = useState(true)
  const [segmentBy, setSegmentBy] = useState<DiagnosticSegmentBy>('quote')
  const [intervalKind, setIntervalKind] = useState<IntervalKind>('wilson')
  const [sortField, setSortField] = useState<ChannelSortField>('total_pnl')
  const [sortAsc, setSortAsc] = useState(false)
  const [summaryError, setSummaryError] = useState('')
  const [diagnosticError, setDiagnosticError] = useState('')
  const loadSummary = useCallback(() => {
    if (!active) return
    api.getLivePerformanceSummary(0, 0, onlyEnabled).then(data => {
      setSummary(data)
      setSummaryError('')
      setSelectedChannel(current => data.channels.some(channel => channel.channel === current)
        ? current
        : data.channels[0]?.channel || '')
    }).catch(e => setSummaryError(`汇总加载失败：${(e as Error).message}`))
  }, [active, onlyEnabled])
  useEffect(() => { if (!active) return; loadSummary(); const timer = setInterval(loadSummary, 30_000); return () => clearInterval(timer) }, [active, loadSummary])
  useEffect(() => {
    if (!active || !selectedChannel || view === 'portfolio') return
    let cancelled = false
    api.getLiveChannelDiagnostics(selectedChannel, segmentBy).then(data => { if (!cancelled) { setDiagnostics(data); setDiagnosticError('') } }).catch(e => { if (!cancelled) setDiagnosticError(`诊断加载失败：${(e as Error).message}`) })
    return () => { cancelled = true }
  }, [active, selectedChannel, segmentBy, view])
  const sortedChannels = useMemo(() => [...(summary?.channels ?? [])].sort((a, b) => {
    const value = (c: PerformanceChannel): string | number | null | undefined => sortField === 'channel' ? c.display_name : sortField === 'ev20' ? c.latest.rolling_realized_ev_20.realized_ev : sortField === 'signal_ev' ? c.signal_truth?.signal_ev : sortField === 'distortion' ? c.signal_truth?.sizing_distortion : sortField === 'win20' ? c.latest.rolling20.win_rate : sortField === 'benchmark' ? c.benchmark?.win_rate : sortField === 'break_even' ? c.latest.mean_break_even_20 : c[sortField]
    const av = value(a), bv = value(b); const cmp = typeof av === 'string' && typeof bv === 'string' ? av.localeCompare(bv) : Number(av ?? -Infinity) - Number(bv ?? -Infinity)
    return sortAsc ? cmp : -cmp
  }), [summary, sortField, sortAsc])
  const chartRows = useMemo(() => (diagnostics?.series ?? []).map(point => {
    const stat = point.rolling['20']; const interval = intervalKind === 'wilson' ? stat?.wilson_95 : stat?.jeffreys_95
    return { ...point, rolling20: stat?.win_rate ?? null, rolling50: point.rolling['50']?.win_rate ?? null, intervalLow: interval?.[0] ?? null, intervalBand: interval ? interval[1] - interval[0] : null, breakEven20: point.rolling_mean_break_even_probability['20'] }
  }), [diagnostics, intervalKind])
  // 信号 EV（等权）vs 资金 ROI（加权）双线：两线分叉处即动态下单金额在起作用
  const evRows = useMemo(() => (diagnostics?.series ?? []).map(point => ({
    t: point.t,
    capital20: point.rolling_realized_ev['20']?.realized_ev ?? null,
    signal20: point.rolling_signal_ev?.['20']?.signal_ev ?? null,
  })), [diagnostics])
  const buttonClass = (on: boolean) => `px-3 py-1 text-xs font-bold rounded-sm border transition-colors ${on ? 'bg-brand text-white border-brand' : 'bg-card text-ink-55 border-line hover:border-brand'}`
  const statBox = 'rounded-card border border-line bg-card px-3 py-2 min-w-0'
  const portfolio = summary?.portfolio, coverage = summary?.coverage, duplicate = summary?.duplicate_exposure
  const chooseSort = (field: ChannelSortField) => { if (field === sortField) setSortAsc(v => !v); else { setSortField(field); setSortAsc(false) } }

  return <Card title="实盘表现诊断">
    <div className="flex gap-1.5 flex-wrap items-center mb-3">
      <button className={buttonClass(view === 'portfolio')} onClick={() => setView('portfolio')}>组合决策</button>
      <button className={buttonClass(view === 'channel')} onClick={() => setView('channel')}>通道诊断</button>
      <button className={buttonClass(view === 'risk')} onClick={() => setView('risk')}>风险归因</button>
      <div className="flex items-center gap-1 ml-3 border-l border-line pl-3">
        <button
          className={`px-2 py-0.5 text-xs rounded-pill border ${onlyEnabled ? 'bg-positive-soft text-positive border-positive font-bold' : 'bg-card text-ink-55 border-line hover:border-brand'}`}
          onClick={() => setOnlyEnabled(true)}
          title="默认仅展示当前启用的实盘通道"
        >
          ● 仅看在线
        </button>
        <button
          className={`px-2 py-0.5 text-xs rounded-pill border ${!onlyEnabled ? 'bg-brand text-white border-brand font-bold' : 'bg-card text-ink-55 border-line hover:border-brand'}`}
          onClick={() => setOnlyEnabled(false)}
          title="展示包含已停用通道在内的全部注册通道"
        >
          全量通道
        </button>
      </div>
      <span className="ml-auto text-[10px] text-ink-55 self-center">仅打开时加载 · 汇总每 30s 刷新</span>
    </div>
    {summaryError && <div className="mb-2 text-xs text-negative bg-negative-soft rounded-sm px-2 py-1">{summaryError}</div>}
    {!summary && !summaryError && <div className="text-sm text-ink-55 py-8 text-center">正在加载实盘表现…</div>}
    {summary && view === 'portfolio' && <>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">
        <div className={statBox}><div className="text-xs text-ink-55">实现 EV20 <HelpHint text="实现 EV=真实净 PnL / 实际投入，不是预测概率；EV50 是更慢的次级观察窗。" /></div><div className="font-mono font-bold text-lg">{performancePct(portfolio?.rolling_realized_ev['20']?.realized_ev)}</div><div className="text-[10px] text-ink-55">EV50 {performancePct(portfolio?.rolling_realized_ev['50']?.realized_ev)}</div></div>
        <div className={statBox}><div className="text-xs text-ink-55">真实 PnL / ROI</div><div className={`font-mono font-bold text-lg ${(portfolio?.total_pnl ?? 0) >= 0 ? 'text-positive' : 'text-negative'}`}>{performanceSigned(portfolio?.total_pnl, ' U')}</div><div className="text-[10px] text-ink-55">ROI {performancePct(portfolio?.roi)}</div></div>
        <div className={statBox}><div className="text-xs text-ink-55">唯一窗口净盈利率 <HelpHint text="同一市场窗口的多笔订单先合并净 PnL，再判断该窗口是否盈利。" /></div><div className="font-mono font-bold text-lg">{performancePct(portfolio?.market_window_win_rate.point_estimate)}</div><div className="text-[10px] text-ink-55">{coverage?.unique_window_count ?? '--'} 个唯一窗口</div></div>
        <div className={statBox}><div className="text-xs text-ink-55">重复投入占比 <HelpHint text="重复窗口中的重叠投入 / 总投入，用于识别通道同时下注造成的组合暴露。" /></div><div className="font-mono font-bold text-lg">{performancePct(duplicate?.duplicate_investment_ratio)}</div><div className="text-[10px] text-ink-55">{duplicate?.duplicate_window_count ?? '--'} 个重复窗口</div></div>
      </div>
      {portfolio?.signal_truth && <><div className="text-xs font-bold text-ink-55 mb-1">信号真相（剔除动态下单金额）</div><SignalTruthStrip truth={portfolio.signal_truth} realPnl={portfolio.total_pnl} /></>}
      <div className="flex flex-wrap gap-2 mb-2 text-xs"><span>订单数 <b className="font-mono">{coverage?.order_count ?? '--'}</b> / 唯一窗口数 <b className="font-mono">{coverage?.unique_window_count ?? '--'}</b></span><span className="text-ink-55">当前注册通道内真实 PnL 按订单全计；胜率按唯一窗口净 PnL 去重。</span></div>
      <div className="flex gap-1 mb-2">{([['ev', '实现EV'], ['pnl', '累计PnL'], ['edge', '保本缺口']] as const).map(([key, label]) => <button key={key} className={buttonClass(metric === key)} onClick={() => setMetric(key)}>{label}</button>)}</div>
      {metric === 'edge' ? <div className="h-[260px] rounded-card border border-line bg-sunken flex flex-col items-center justify-center text-center px-6"><div className="text-xs text-ink-55">实际胜负（0/1）减逐笔保本概率的汇总均值</div><div className="font-mono text-2xl font-bold mt-1">{performancePct(portfolio?.quote_edge.mean)}</div><div className="text-xs text-ink-55 mt-2">正值表示样本命中超过成交成本要求；P0 暂无该指标的时间序列，且它不是预测概率校准。</div><div className="w-full mt-5 border-t border-dashed border-line"><span className="relative -top-2 bg-sunken px-2 text-[10px] text-ink-55">0 = 样本命中刚好覆盖保本要求</span></div></div> : <ResponsiveContainer width="100%" height={260}><LineChart data={summary.portfolio_series}><CartesianGrid stroke="var(--line-soft)"/><XAxis dataKey="t" tickFormatter={performanceTime} minTickGap={40} tick={{ fontSize: 10, fill: 'var(--ink-55)' }}/><YAxis tick={{ fontSize: 10, fill: 'var(--ink-55)' }} tickFormatter={(v: number) => metric === 'ev' ? `${(v * 100).toFixed(0)}%` : v.toFixed(1)}/><Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={v => performanceTime(Number(v))} formatter={(v, n) => [metric === 'ev' ? performancePct(Number(v)) : performanceSigned(Number(v), ' U'), n]}/><Legend/><ReferenceLine y={0} stroke="var(--line)" strokeDasharray="3 3"/>{metric === 'ev' ? <><Line name="实现EV20" dataKey="rolling_realized_ev_20" stroke="var(--chart-1)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false}/><Line name="实现EV50" dataKey="rolling_realized_ev_50" stroke="var(--chart-3)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false}/></> : <Line name="累计PnL" dataKey="cumulative_pnl" stroke="var(--positive)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false}/>}</LineChart></ResponsiveContainer>}
    </>}
    {summary && view !== 'portfolio' && <>
      <div className="overflow-x-auto max-h-56 overflow-y-auto mb-3"><table className="w-full min-w-[680px] text-xs"><thead className="sticky top-0 bg-card"><tr className="text-ink-55 border-b border-line-soft">{([['channel','通道'],['order_count','n'],['ev20','EV20'],['signal_ev','信号EV'],['distortion','扭曲Δ'],['win20','滚动20胜率'],['benchmark','冻结基准'],['break_even','保本20'],['total_pnl','PnL']] as [ChannelSortField,string][]).map(([key,label]) => <th key={key} className={`py-1.5 px-2 cursor-pointer ${key === 'channel' ? 'text-left' : 'text-right'}`} onClick={() => chooseSort(key)}>{label}{sortField === key ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>)}</tr></thead><tbody>{sortedChannels.map(c => <tr key={c.channel} onClick={() => setSelectedChannel(c.channel)} className={`border-b border-line-soft cursor-pointer ${selectedChannel === c.channel ? 'bg-brand-soft' : 'hover:bg-card-hover'}`}><td className="py-1.5 px-2 whitespace-nowrap"><span className="font-medium">{SIGNAL_INFO[c.channel]?.name ?? c.display_name}</span>{!c.enabled && <span className="ml-1 rounded-pill bg-sunken px-1.5 py-0.5 text-[9px] text-ink-55">已停火</span>} <HelpHint text={`${c.family ?? '--'} · ${c.market_period ?? '--'}`} /></td><td className="py-1.5 px-2 text-right font-mono">{c.order_count}<div className="text-[9px] text-warning">{c.order_count < 20 ? `${c.order_count}/20 积累中` : ''}</div></td><td className="px-2 text-right font-mono">{performancePct(c.latest.rolling_realized_ev_20?.realized_ev)}</td><td className={`px-2 text-right font-mono ${signClass(c.signal_truth?.signal_ev)}`} title="全样本等权信号EV">{performancePct(c.signal_truth?.signal_ev)}</td><td className={`px-2 text-right font-mono ${distortionClass(c.signal_truth?.sizing_distortion)}`} title="资金ROI − 信号EV">{performanceSigned(c.signal_truth?.sizing_distortion == null ? null : c.signal_truth.sizing_distortion * 100, '%', 1)}</td><td className="px-2 text-right font-mono">{performancePct(c.latest.rolling20?.win_rate)}</td><td className="px-2 text-right font-mono">{performancePct(c.benchmark?.win_rate)}</td><td className="px-2 text-right font-mono">{performancePct(c.latest.mean_break_even_20)}</td><td className={`px-2 text-right font-mono font-bold ${c.total_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{performanceSigned(c.total_pnl)}</td></tr>)}</tbody></table></div>
      {diagnosticError && <div className="mb-2 text-xs text-negative bg-negative-soft rounded-sm px-2 py-1">{diagnosticError}</div>}
      {view === 'channel' && diagnostics?.channel.channel === selectedChannel && <><div className="flex flex-wrap items-center gap-2 mb-2 text-xs"><b>{SIGNAL_INFO[selectedChannel]?.name ?? diagnostics.channel.display_name}</b><span className="text-ink-55">置信区间</span><button className={buttonClass(intervalKind === 'wilson')} onClick={() => setIntervalKind('wilson')}>Wilson</button><button className={buttonClass(intervalKind === 'beta')} onClick={() => setIntervalKind('beta')}>Beta</button><HelpHint text="Wilson 是频率学派二项比例区间；Jeffreys Beta 是 Beta(1/2,1/2) 先验的贝叶斯可信区间，均适合小样本。" /><HelpHint text="实现 EV 是真实净 PnL / 实际投入；保本率是按成交成本推导的最低胜率。" />{diagnostics.channel.max_exec_price != null && <span className="ml-auto text-ink-55">当前护栏 max_exec <b className="font-mono">{diagnostics.channel.max_exec_price}</b></span>}</div>{diagnostics.signal_truth && <SignalTruthStrip truth={diagnostics.signal_truth} realPnl={diagnostics.series.reduce((s, p) => s + (p.pnl ?? 0), 0)} />}<ResponsiveContainer width="100%" height={300}><ComposedChart data={chartRows}><CartesianGrid stroke="var(--line-soft)"/><XAxis dataKey="t" tickFormatter={performanceTime} minTickGap={40} tick={{ fontSize: 10, fill: 'var(--ink-55)' }}/><YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 10, fill: 'var(--ink-55)' }}/><Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={v => performanceTime(Number(v))} formatter={(v, n) => [performancePct(Number(v)), n]}/><Legend/><Area name={`${intervalKind === 'wilson' ? 'Wilson' : 'Jeffreys Beta'} 下界`} dataKey="intervalLow" stackId="ci" stroke="none" fill="transparent"/><Area name="滚动20 95%区间" dataKey="intervalBand" stackId="ci" stroke="none" fill="var(--brand-soft)" fillOpacity={0.7}/>{diagnostics.benchmark != null && <ReferenceLine y={diagnostics.benchmark.win_rate} stroke="var(--warning)" strokeDasharray="6 4" label="冻结基准"/>}<Line name="滚动20" dataKey="rolling20" stroke="var(--chart-1)" strokeWidth={2} dot={false}/><Line name="滚动50" dataKey="rolling50" stroke="var(--chart-3)" strokeWidth={2} dot={false}/><Line name="EWMA" dataKey="ewma_win_rate" stroke="var(--chart-5)" strokeWidth={2} dot={false}/><Line name="动态保本率" dataKey="breakEven20" stroke="var(--negative)" strokeDasharray="2 4" strokeWidth={2} dot={false}/></ComposedChart></ResponsiveContainer>
        <div className="flex items-center gap-2 mt-3 mb-1 text-xs font-bold text-ink-55">信号EV20（等权）vs 资金ROI20（加权） <HelpHint text="两条线分叉处就是动态下单金额在放大或吞噬信号：资金线低于信号线=大注输得多。" /></div>
        <ResponsiveContainer width="100%" height={200}><LineChart data={evRows}><CartesianGrid stroke="var(--line-soft)"/><XAxis dataKey="t" tickFormatter={performanceTime} minTickGap={40} tick={{ fontSize: 10, fill: 'var(--ink-55)' }}/><YAxis tickFormatter={(v: number) => `${Math.round(v * 100)}%`} tick={{ fontSize: 10, fill: 'var(--ink-55)' }}/><Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={v => performanceTime(Number(v))} formatter={(v, n) => [performancePct(Number(v)), n]}/><Legend/><ReferenceLine y={0} stroke="var(--line)" strokeDasharray="3 3"/><Line name="信号EV20（等权）" dataKey="signal20" stroke="var(--chart-1)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false}/><Line name="资金ROI20（加权）" dataKey="capital20" stroke="var(--chart-3)" strokeWidth={2} strokeDasharray="5 3" dot={false} connectNulls isAnimationActive={false}/></LineChart></ResponsiveContainer></>}
      {view === 'risk' && <><div className="flex flex-wrap items-center gap-2 mb-2"><label className="text-xs text-ink-55" htmlFor="risk-segment">归因维度</label><select id="risk-segment" value={segmentBy} onChange={e => setSegmentBy(e.target.value as DiagnosticSegmentBy)} className="bg-card border border-line rounded-sm px-2 py-1 text-xs">{SEGMENT_OPTIONS.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}</select><span className="text-[10px] text-ink-55">趋势、波动率与部署版本只展示实际捕获值。n&lt;10 的分桶灰显，仅供参考。</span>{segmentBy === 'exec_price' && diagnostics?.channel.max_exec_price != null && <span className="text-[10px] text-ink-55 ml-auto">当前护栏 max_exec <b className="font-mono">{diagnostics.channel.max_exec_price}</b>（贴线即弃单）</span>}</div>{diagnostics?.channel.channel === selectedChannel && <div className="overflow-x-auto"><table className="w-full min-w-[980px] text-xs"><thead><tr className="text-ink-55 border-b border-line-soft"><th className="text-left py-1.5 px-2">分桶</th><th className="text-right px-2">n</th><th className="text-right px-2">唯一窗口</th><th className="text-right px-2">胜率 <HelpHint text="括号内为 Wilson 95% 区间。" /></th><th className="text-right px-2">保本率</th><th className="text-right px-2">Edge <HelpHint text={SIGNAL_TRUTH_HINTS.edge} /></th><th className="text-right px-2">信号EV <HelpHint text={SIGNAL_TRUTH_HINTS.signal} /></th><th className="text-right px-2">资金ROI <HelpHint text={SIGNAL_TRUTH_HINTS.capital} /></th><th className="text-right px-2">扭曲Δ <HelpHint text={SIGNAL_TRUTH_HINTS.distortion} /></th><th className="text-right px-2">均/最大注码</th><th className="text-right px-2">PnL</th><th className="text-left px-2">质量</th></tr></thead><tbody>{diagnostics.segments.map(row => {
        const guard = segmentBy === 'exec_price' ? diagnostics.channel.max_exec_price : null
        const low = Number(/^\[([\d.]+),/.exec(row.segment)?.[1] ?? (row.segment === '>=0.70' ? 0.70 : NaN))
        const high = Number(/,([\d.]+)\)$/.exec(row.segment)?.[1] ?? (row.segment === '>=0.70' ? 1 : NaN))
        const guardTag = guard == null || Number.isNaN(low) ? null : low >= guard ? '护栏外' : high > guard ? '跨护栏' : null
        return <tr key={row.segment} className={`border-b border-line-soft ${row.low_sample ? 'opacity-55' : ''}`}><td className="py-1.5 px-2 font-medium whitespace-nowrap">{row.segment === 'LEGACY_UNKNOWN' ? <span className="text-warning">历史未捕获</span> : row.segment}{guardTag && <span className={`ml-1 rounded-pill px-1.5 py-0.5 text-[9px] ${guardTag === '护栏外' ? 'bg-negative-soft text-negative' : 'bg-warning-soft text-warning'}`}>{guardTag}</span>}{row.low_sample && <span className="ml-1 text-[9px] text-ink-55">样本不足</span>}</td><td className="text-right px-2 font-mono">{row.n}</td><td className="text-right px-2 font-mono">{row.unique_window_count}</td><td className="text-right px-2 font-mono whitespace-nowrap">{performancePct(row.win_rate)}<div className="text-[9px] text-ink-55">{ciText(row.win_rate_wilson_95)}</div></td><td className="text-right px-2 font-mono">{performancePct(row.mean_break_even)}</td><td className={`text-right px-2 font-mono whitespace-nowrap ${signClass(row.edge)}`}>{performanceSigned(row.edge == null ? null : row.edge * 100, '%', 1)}<div className="text-[9px] text-ink-55">{ciText(row.edge_ci95)}</div></td><td className={`text-right px-2 font-mono font-bold ${signClass(row.signal_ev)}`}>{performancePct(row.signal_ev)}</td><td className={`text-right px-2 font-mono ${signClass(row.realized_ev)}`}>{performancePct(row.realized_ev)}</td><td className={`text-right px-2 font-mono ${distortionClass(row.sizing_distortion)}`}>{performanceSigned(row.sizing_distortion == null ? null : row.sizing_distortion * 100, '%', 1)}</td><td className="text-right px-2 font-mono whitespace-nowrap">{row.stake_mean == null ? '--' : row.stake_mean.toFixed(1)} / {row.stake_max == null ? '--' : row.stake_max.toFixed(1)}</td><td className={`text-right px-2 font-mono ${row.total_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{performanceSigned(row.total_pnl)}</td><td className="px-2 text-ink-55">{row.data_quality ?? '--'}</td></tr>
      })}</tbody></table></div>}</>}
    </>}
    {(summary?.warnings.length ?? 0) > 0 && <div className="mt-2 text-[10px] text-warning">{summary?.warnings.join('；')}</div>}
  </Card>
}


// 右侧抽屉（从主布局移入）：
// 双 tab：盈利表现 / 资金变化。默认聚焦在跑通道盈利趋势，减轻抽屉负荷。
function ChartDrawer({ orders }: { orders: Record<string, unknown>[] }) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'pnl' | 'funds'>('pnl')
  const tabBtn = (t: 'pnl' | 'funds', label: string) => (
    <button
      onClick={() => setTab(t)}
      className={`px-3 py-1 text-xs font-semibold rounded-sm border transition ${tab === t ? 'bg-brand text-white border-brand' : 'bg-card text-ink-80 border-line hover:border-brand'}`}
    >{label}</button>
  )
  return (
    <>
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed right-0 top-[62%] -translate-y-1/2 z-40 bg-brand text-white text-xs font-bold px-2 py-3 rounded-l-sm hover:bg-brand-hover transition"
          style={{ writingMode: 'vertical-rl' }}
          title="查看盈利表现 / 资金变化"
        >
          📊 表现
        </button>
      )}

      <div
        className={`fixed top-0 right-0 h-screen w-[760px] max-w-[94vw] bg-card border-l border-line z-50 flex flex-col transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="px-4 py-2.5 border-b border-line bg-card flex items-center justify-between shrink-0 gap-2">
          <div className="flex items-center gap-2">
            {tabBtn('pnl', '📊 盈利表现')}
            {tabBtn('funds', '💰 资金变化')}
          </div>
          <button onClick={() => setOpen(false)} className="text-ink-55 hover:text-ink-80 text-lg leading-none px-1">✕</button>
        </div>
        <div className="flex-1 min-h-0 overflow-auto p-3">
          <div className={tab === 'pnl' ? '' : 'hidden'}><PnlCurveCard active={open && tab === 'pnl'} /></div>
          <div className={tab === 'funds' ? '' : 'hidden'}><FundsFlowCard orders={orders} /></div>
        </div>
      </div>
    </>
  )
}

// 多通道实盘状态类型（与后端 multi_live_trader.status_async() 严格对齐）
interface LiveChannelStatus {
  channel: string
  display_name: string
  family: string
  market_period: string
  direction: string
  order_type?: string
  enabled: boolean
  enabled_at_startup: boolean
  amount_usdt: number
  max_daily_orders: number
  max_exec_price: number
  auto_max_exec: number
  custom_max_exec: boolean
  fire_total: number
  fired_windows: number[]
  filled_today?: number
}

function LiveTradeTab() {
  const [wallet, setWallet] = useState<Record<string, unknown> | null>(null)
  const [live, setLive] = useState<Record<string, unknown> | null>(null)
  const [orders, setOrders] = useState<Record<string, unknown>[]>([])
  // 下单表单状态（amount/side/busy/result）已移入悬浮下单 TestTradeFab
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
  // 通道护栏热调草稿（key=channel；同金额模式，保存后清除）
  const [guardDrafts, setGuardDrafts] = useState<Record<string, string>>({})
  // 可领取奖金（赢单 token → batch-redeem → USDT）
  const [redeemable, setRedeemable] = useState<Record<string, unknown> | null>(null)
  const [redeeming, setRedeeming] = useState(false)
  const [redeemResult, setRedeemResult] = useState<Record<string, unknown> | null>(null)
  const [showRedeemDetails, setShowRedeemDetails] = useState(false)
  const [showStoppedChannels, setShowStoppedChannels] = useState(false)
  const [expandedLiveChannel, setExpandedLiveChannel] = useState<string | null>(null)
  // 通道盈亏与 ROI 数据（2026-09-09：在实盘通道卡片中直观映射实盘战绩与盈利率）
  const [pnlData, setPnlData] = useState<PnlCurveData | null>(null)

  const refresh = useCallback(() => {
    api.getPredictionWallet().then(setWallet).catch(() => {})
    api.getLiveStatus().then(d => setLive(d?.live_channels ?? null)).catch(() => {})
    api.getRecentTrades().then(d => setOrders(d?.orders ?? [])).catch(() => {})
    api.getRedeemable().then(setRedeemable).catch(() => {})
    api.getLivePnlCurve().then(setPnlData).catch(() => {})
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

  const walletErr = (wallet as { error?: string } | null)?.error
  const regTs = wallet?.registered_time as number | undefined
  // 多通道实盘：live = live_channels status（channels[] 见 LiveChannelStatus）
  const liveChannels = Array.isArray(live?.channels) ? live.channels as LiveChannelStatus[] : []
  const liveDefaults = (live?.defaults ?? {}) as Record<string, unknown>

  // 通道 pnl 快速索引；列表默认按信号 EV、真实胜率依次降序，无已结算单者排末尾。
  const { pnlByChan, liveSortMetrics } = useMemo(() => {
    const byChannel: Record<string, PnlChannel> = {}
    const metrics: Record<string, { winRate: number | null; signalEv: number | null }> = {}
    for (const channel of pnlData?.channels ?? []) {
      byChannel[channel.channel] = channel
      const returns = channel.points.filter(point => point.cost != null && point.cost > 0).map(point => point.pnl / point.cost!)
      metrics[channel.channel] = {
        winRate: channel.win_rate,
        signalEv: returns.length ? returns.reduce((sum, value) => sum + value, 0) / returns.length : null,
      }
    }
    return { pnlByChan: byChannel, liveSortMetrics: metrics }
  }, [pnlData])
  const byLivePerformance = (a: LiveChannelStatus, b: LiveChannelStatus) => {
    const am = liveSortMetrics[a.channel]
    const bm = liveSortMetrics[b.channel]
    const evDiff = (bm?.signalEv ?? Number.NEGATIVE_INFINITY) - (am?.signalEv ?? Number.NEGATIVE_INFINITY)
    if (evDiff) return evDiff
    const winDiff = (bm?.winRate ?? Number.NEGATIVE_INFINITY) - (am?.winRate ?? Number.NEGATIVE_INFINITY)
    return winDiff || a.channel.localeCompare(b.channel)
  }
  const activeLiveChannels = liveChannels.filter(c => c.enabled).sort(byLivePerformance)
  const stoppedLiveChannels = liveChannels.filter(c => !c.enabled).sort(byLivePerformance)
  const enabledCount = activeLiveChannels.length
  const toggleLiveDetails = (channel: string) => setExpandedLiveChannel(current => current === channel ? null : channel)
  const liveDetails = (ch: LiveChannelStatus) => {
    const p = pnlByChan[ch.channel]
    const metrics = liveSortMetrics[ch.channel]
    const detail = (label: string, value: string, tone = '') => <div><span className="text-ink-55">{label}</span><b className={`ml-1 font-mono ${tone}`}>{value}</b></div>
    return <div className="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-1 px-3 py-2 text-[11px] border-t border-line-soft bg-card" onClick={event => event.stopPropagation()}>
      {detail('真实胜率', metrics?.winRate == null ? '--' : `${(metrics.winRate * 100).toFixed(1)}%`)}
      {detail('信号EV', metrics?.signalEv == null ? '--' : `${metrics.signalEv >= 0 ? '+' : ''}${(metrics.signalEv * 100).toFixed(1)}%`, metrics?.signalEv == null ? '' : metrics.signalEv >= 0 ? 'text-positive' : 'text-negative')}
      {detail('已结算', `${p?.settled_count ?? 0} 单`)}
      {detail('胜 / 负', `${p?.win_count ?? 0} / ${Math.max(0, (p?.settled_count ?? 0) - (p?.win_count ?? 0))}`)}
      {detail('累计投入', p ? `${p.total_cost.toFixed(2)} U` : '--')}
      {detail('真实PnL', p ? `${p.total_pnl >= 0 ? '+' : ''}${p.total_pnl.toFixed(2)} U` : '--', p ? p.total_pnl >= 0 ? 'text-positive' : 'text-negative' : '')}
      {detail('资金ROI', p?.roi == null ? '--' : `${p.roi >= 0 ? '+' : ''}${(p.roi * 100).toFixed(1)}%`, p?.roi == null ? '' : p.roi >= 0 ? 'text-positive' : 'text-negative')}
      {detail('今日 / 日限', `${ch.filled_today ?? 0} / ${ch.max_daily_orders}`)}
      {detail('累计开火', `${ch.fire_total} 次`)}
      {detail('周期', ch.market_period || '--')}
      {detail('方向', ch.direction || '--')}
      {detail('执行价护栏', ch.max_exec_price.toFixed(2))}
    </div>
  }

  // 倒计时：服务端时钟修正后的剩余毫秒；<60s 红色警示
  const windowEnd = quote?.window_end as number | null | undefined
  const remainSec = (typeof windowEnd === 'number' && quote && !quote.stale)
    ? Math.max(0, Math.floor((windowEnd - (nowMs + clockOffset)) / 1000))
    : null

  // 在途持仓（FILLED 未结算；订单明细/累计盈亏见账户状态下方「最近订单」）
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
        `确认开启通道实盘？\n通道: ${name}（${ch.channel}）\n每单 ${String(ch.amount_usdt)} USDT | 执行价护栏 ${String(ch.max_exec_price)} | 日限 ${String(ch.max_daily_orders)} 单\n\n命中信号将下真实订单（真金白银）；设定写入 DB 重启后保持（仅持久化失败时回落 LIVE_CHANNELS_JSON）。`)) return
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

  const handleChannelGuard = async (ch: LiveChannelStatus, reset = false) => {
    // 护栏热调：执行价上限（贴线弃单），校验同后端 0.01~0.99；
    // reset=True 清空自定义回落通道预设值（预设=ChannelSpec.auto_max_exec）
    let val = 0
    if (!reset) {
      val = parseFloat(guardDrafts[ch.channel] ?? '')
      if (!Number.isFinite(val) || val < 0.01 || val > 0.99) {
        alert('护栏仅允许 0.01~0.99（报价均价 ≥ 护栏即弃单，含贴线）')
        return
      }
      if (val === ch.max_exec_price) return
    } else if (!ch.custom_max_exec) return
    setTogglingLive(true)
    try {
      const res = await api.postLiveChannel(
        ch.channel, ch.enabled, undefined, undefined, reset ? undefined : val, reset)
      if (res?.error) {
        alert(`护栏保存失败: ${String(res.error)}`)
      } else {
        setGuardDrafts(prev => {
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

  const handleRedeem = async (tokenIds?: string[]) => {
    const n = tokenIds ? tokenIds.length : Number(redeemable?.claimable_count ?? 0)
    if (n <= 0) return
    const msg = tokenIds
      ? `确认领取选中的 ${n} 笔获胜 token 奖金？\n（batch-redeem 赎回后入预测钱包 USDT 余额）`
      : `确认领取全部 ${n} 个获胜 token 的奖金？\n（batch-redeem 赎回后入预测钱包 USDT 余额）`
    if (!window.confirm(msg)) return
    setRedeeming(true)
    setRedeemResult(null)
    try {
      const res = await api.postRedeem(tokenIds)
      setRedeemResult(res)
      refresh()
    } catch (e) {
      alert(`请求失败: ${(e as Error).message}`)
    } finally {
      setRedeeming(false)
    }
  }

  const claimableCount = Number(redeemable?.claimable_count ?? 0)
  const totalEstPayout = typeof redeemable?.total_est_payout === 'number' ? (redeemable.total_est_payout as number) : 0
  const redeemDetails = Array.isArray(redeemable?.details) ? (redeemable.details as Array<{
    order_id: number
    token_id: string
    signal_version: string | null
    market_period: string
    direction: string | null
    amount_usdt: number
    pnl: number
    est_payout: number
    settle_price: number | null
    settled_at: string | null
    created_at: string | null
    is_in_wallet: boolean
    can_claim: boolean
  }>) : []
  const positionsPreview = Array.isArray(redeemable?.positions_preview) ? (redeemable.positions_preview as Array<Record<string, unknown>>) : []

  return (
    <>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* 下单使用右下角悬浮 FAB；表现诊断与资金变化进入右侧抽屉。 */}
      <div className="lg:col-span-2">
      <Card title="账户状态">
        <div className="space-y-2 text-sm">
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">预测钱包</span>
            {walletErr
              ? <span className="text-negative text-right">{walletErr}</span>
              : <span className="font-mono text-ink-95">{String(wallet?.wallet_address ?? '--')}</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">钱包 ID</span>
            <span className="font-mono text-ink-80 text-xs truncate">{String(wallet?.wallet_id ?? '--')}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">钱包注册时间</span>
            <span className="text-ink-80 num">{regTs ? new Date(regTs).toLocaleString() : '--'}</span>
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">预测钱包 USDT</span>
            {typeof wallet?.prediction_usdt_free === 'number'
              ? <span className={`font-mono font-bold ${(wallet.prediction_usdt_free as number) >= 1 ? 'text-positive' : 'text-negative'}`}>{(wallet.prediction_usdt_free as number).toFixed(4)}</span>
              : <span className="text-ink-55">暂不可查（API 字段待确认）</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">现货 USDT 可用</span>
            {typeof wallet?.spot_usdt_free === 'number'
              ? <span className={`font-mono font-bold ${(wallet.spot_usdt_free as number) >= 1 ? 'text-positive' : 'text-negative'}`}>{(wallet.spot_usdt_free as number).toFixed(4)}</span>
              : <span className="text-ink-55">查询失败</span>}
          </div>
          <div className="flex justify-between gap-2">
            <span className="text-ink-55 shrink-0">在途持仓（未结算）</span>
            {openPositions.length > 0
              ? <span className="font-mono text-warning">{openPositions.length} 单 · {openAmount.toFixed(2)} USDT</span>
              : <span className="text-ink-55">--</span>}
          </div>
          {/* 可领取奖金：赢单 token 需手动 batch-redeem 才变 USDT（官方链路） */}
          <div className="flex justify-between gap-2 items-center flex-wrap">
            <span className="text-ink-55 shrink-0 flex items-center">
              可领取奖金
              <HelpHint text="赢单的奖金以获胜 token 形式留在链上钱包，不会自动变成 USDT；需要调官方 batch-redeem 赎回后才入预测钱包余额。赢单后记得来这里领取。" />
            </span>
            {claimableCount > 0 ? (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono font-semibold text-warning flex items-center gap-1.5">
                  <span>{claimableCount} 笔待领</span>
                  {totalEstPayout > 0 && (
                    <span className="text-positive font-bold">（预计回款 ~{totalEstPayout.toFixed(2)} USDT）</span>
                  )}
                  {redeemable?.wallet_source === 'degraded' && (
                    <span className="text-[10px] text-ink-55 font-normal">（钱包查询降级，本地兑底）</span>
                  )}
                </span>
                <button
                  type="button"
                  onClick={() => setShowRedeemDetails(!showRedeemDetails)}
                  className="px-2 py-0.5 text-xs font-semibold rounded-pill border border-line bg-card text-ink-80 hover:border-brand hover:text-brand transition-colors"
                >
                  明细 {showRedeemDetails ? '▲' : '▼'}
                </button>
                <button
                  onClick={() => handleRedeem()} disabled={redeeming}
                  className="px-3 py-1 text-xs font-semibold rounded-pill bg-warning text-white hover:bg-warning-hover disabled:opacity-50 transition-colors"
                >
                  {redeeming ? '领取中…' : '领取全部奖金'}
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span className="text-ink-55">--</span>
                {redeemDetails.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setShowRedeemDetails(!showRedeemDetails)}
                    className="text-[11px] text-ink-55 hover:text-brand underline cursor-pointer"
                  >
                    查看对账明细 ({redeemDetails.length}) {showRedeemDetails ? '▲' : '▼'}
                  </button>
                )}
              </div>
            )}
          </div>

          {/* 可领取奖金细节展开面板 */}
          {showRedeemDetails && (
            <div className="bg-sunken border border-warning/30 rounded-sm p-2.5 my-2 space-y-2 text-xs">
              <div className="flex items-center justify-between gap-2 border-b border-line-soft pb-1.5 flex-wrap">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-ink-95">获胜待领 Token 细节对账清单</span>
                  <span className="text-[10px] text-ink-55">
                    共 {redeemDetails.length} 笔记录（{claimableCount} 笔链上可赎）
                  </span>
                </div>
                {claimableCount > 0 && (
                  <div className="text-[11px] font-mono text-ink-80">
                    总回款预估：<span className="text-positive font-bold font-mono">+{totalEstPayout.toFixed(2)} USDT</span>
                  </div>
                )}
              </div>

              {redeemDetails.length === 0 ? (
                <p className="text-ink-55 py-2 text-center">暂无待领取的获胜订单</p>
              ) : (
                <div className="overflow-x-auto max-h-56 overflow-y-auto">
                  <table className="w-full text-left">
                    <thead className="sticky top-0 bg-sunken text-[11px] text-ink-55 border-b border-line-soft">
                      <tr>
                        <th className="py-1 pr-2">订单 / 周期</th>
                        <th className="py-1 pr-2">信号通道</th>
                        <th className="py-1 pr-2">方向</th>
                        <th className="py-1 pr-2 text-right">投入</th>
                        <th className="py-1 pr-2 text-right">净收益</th>
                        <th className="py-1 pr-2 text-right">预计兑付</th>
                        <th className="py-1 pr-2">Token ID</th>
                        <th className="py-1 pr-2">状态</th>
                        <th className="py-1 text-right">操作</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line-soft font-mono">
                      {redeemDetails.map(d => {
                        const info = d.signal_version ? SIGNAL_INFO[d.signal_version] : null
                        const name = info?.name ?? d.signal_version ?? '人工/未知'
                        const isClaimable = d.can_claim
                        return (
                          <tr key={d.order_id} className="hover:bg-card/50 transition-colors">
                            <td className="py-1.5 pr-2">
                              <span className="font-bold text-ink-95">#{d.order_id}</span>
                              <span className="ml-1 px-1 rounded-pill bg-card text-[9px] text-ink-55 border border-line">
                                {d.market_period}
                              </span>
                            </td>
                            <td className="py-1.5 pr-2 font-sans text-ink-95 truncate max-w-[140px]" title={d.signal_version ?? ''}>
                              {name}
                            </td>
                            <td className="py-1.5 pr-2">
                              <span className={`px-1 py-0.5 rounded-sm text-[10px] font-bold ${d.direction === 'UP' ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'}`}>
                                {d.direction ?? '--'}
                              </span>
                            </td>
                            <td className="py-1.5 pr-2 text-right text-ink-80">
                              {d.amount_usdt.toFixed(2)}U
                            </td>
                            <td className="py-1.5 pr-2 text-right text-positive font-bold">
                              +{d.pnl.toFixed(2)}U
                            </td>
                            <td className="py-1.5 pr-2 text-right text-warning font-bold">
                              {d.est_payout.toFixed(2)}U
                            </td>
                            <td className="py-1.5 pr-2 text-[10px] text-ink-55">
                              <span title={d.token_id} className="cursor-help underline decoration-dotted">
                                {d.token_id ? `${d.token_id.slice(0, 6)}…${d.token_id.slice(-4)}` : '--'}
                              </span>
                            </td>
                            <td className="py-1.5 pr-2 font-sans">
                              {isClaimable ? (
                                <span className="px-1.5 py-0.5 text-[9px] font-bold rounded-pill bg-positive-soft text-positive border border-positive/30">
                                  可赎回
                                </span>
                              ) : (
                                <span className="px-1.5 py-0.5 text-[9px] font-normal rounded-pill bg-card text-ink-55 border border-line" title="链上已无持仓或已完成兑换">
                                  链上已结
                                </span>
                              )}
                            </td>
                            <td className="py-1.5 text-right">
                              {isClaimable ? (
                                <button
                                  type="button"
                                  disabled={redeeming}
                                  onClick={() => handleRedeem([d.token_id])}
                                  className="px-2 py-0.5 text-[10px] font-bold rounded-pill bg-warning text-white hover:bg-warning-hover disabled:opacity-40"
                                >
                                  单笔领
                                </button>
                              ) : (
                                <span className="text-ink-55 text-[10px] font-sans">已完结</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="pt-1 border-t border-line-soft text-[10px] text-ink-55 leading-relaxed flex items-center justify-between flex-wrap gap-1">
                <span>
                  💡 官方机制：每个获胜的预测合约结算后兑现为 1 USDT。链上作为获胜 Token 存留，点击领取时调用官方 batch-redeem 接口销毁 Token 并划入预测钱包余额。
                </span>
                {positionsPreview.length > 0 && (
                  <span className="text-ink-80 font-mono">
                    链上原始持仓: {positionsPreview.length} 项
                  </span>
                )}
              </div>
            </div>
          )}
          {redeemResult && (
            <div className={`text-xs px-2 py-1 rounded-sm break-all ${redeemResult.status === 'SUCCESS' ? 'bg-positive-soft text-positive' : redeemResult.status === 'NOOP' ? 'bg-sunken text-ink-55' : 'bg-negative-soft text-negative'}`}>
              {redeemResult.status === 'SUCCESS'
                ? `✓ 已领取 ${String(redeemResult.redeemed)} 个 token，奖金入预测钱包余额（前端刷新后可见）`
                : redeemResult.status === 'NOOP'
                  ? String(redeemResult.message ?? '无可领取持仓')
                  : `领取失败: ${String(redeemResult.error ?? '未知错误')}`}
            </div>
          )}
          {payAccounts.length > 0 && (
            <div className="flex justify-between gap-2">
              <span className="text-ink-55 shrink-0">支付账户余额</span>
              <span className="font-mono text-xs text-right break-all text-ink-95">
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
            <span className="text-ink-55 shrink-0">钱包持仓 Token</span>
            {walletAssets == null
              ? <span className="text-ink-55">暂不可查（探索型端点）</span>
              : heldTokens.length === 0
                ? <span className="text-ink-55">--（无 outcome token）</span>
                : <span className="font-mono text-xs text-right break-all text-ink-95">
                    {heldTokens.map(a => {
                      const sym = String(a.asset ?? a.symbol ?? '?')
                      const amt = Number(a.free ?? a.balance ?? 0)
                      return `${sym.length > 10 ? sym.slice(0, 10) + '…' : sym} × ${amt > 1e12 ? (amt / 1e18).toFixed(4) : amt}`
                    }).join(' | ')}
                  </span>}
          </div>
          )}
          <div className="flex items-center gap-2">
            <span className="text-ink-55 shrink-0">划转入金</span>
            <input
              type="number" min={0.1} max={20} step={0.5} value={transferAmt}
              onChange={e => setTransferAmt(e.target.value)}
              className="ds-input ds-input-numeric w-20"
            />
            <button
              onClick={handleTransferIn} disabled={transferring}
              className="px-3 py-1 text-xs font-semibold rounded-pill bg-brand text-white disabled:opacity-50"
            >{transferring ? '划转中…' : '现货 → 预测钱包'}</button>
            <button
              onClick={handleTransferOut} disabled={transferring}
              className="px-3 py-1 text-xs font-semibold rounded-pill bg-positive text-white disabled:opacity-50"
            >{transferring ? '划转中…' : '预测钱包 → 现货'}</button>
          </div>
          {transferResult && (
            <div className={`text-xs px-2 py-1 rounded-sm ${transferResult.status === 'SUCCESS' && transferResult.direction_confirmed !== false ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative break-all'}`}>
              {transferResult.status === 'SUCCESS'
                ? transferResult.direction_confirmed === true
                  ? `划出成功，现货 ${typeof transferResult.spot_before === 'number' ? (transferResult.spot_before as number).toFixed(4) : '?'} → ${typeof transferResult.spot_after === 'number' ? (transferResult.spot_after as number).toFixed(4) : '?'}（方向已自证）`
                  : transferResult.direction_confirmed === false
                    ? `⚠ ${String(transferResult.warning ?? '划出已提交但现货余额未见增加，请立即人工核对划转记录')}`
                    : `划转成功，现货余额 → ${typeof transferResult.spot_usdt_free === 'number' ? (transferResult.spot_usdt_free as number).toFixed(4) : '?'}`
                : `划转失败: ${String(transferResult.error ?? '未知错误')}`}
            </div>
          )}
          <div className="border-t border-line-soft my-2" />
          <div className="flex justify-between gap-2 items-center">
            <span className="text-ink-55 shrink-0 flex items-center">
              信号实盘通道
              <HelpHint text="后端注册表中的实盘通道：每通道独立开关、金额、日限和执行价护栏。默认按信号EV降序，同EV按真实胜率降序；无已结算单排末尾。是否实盘只以后端通道状态为准；影子采集开关不等于真钱开关。" />
            </span>
            {liveChannels.length > 0
              ? <span className="flex items-baseline gap-1.5 text-right">
                  <span className={enabledCount > 0 ? 'text-positive font-semibold' : 'text-warning font-semibold'}>
                    {enabledCount > 0 ? `${enabledCount} 个通道实盘中` : '全部关闭（不开火）'}
                  </span>
                  <span className="text-[10px] text-ink-55">
                    默认 {String(liveDefaults.amount_usdt ?? '--')}U/单 · 在途任务 {String(live?.pending_tasks ?? 0)}
                  </span>
                </span>
              : <span className="text-ink-55">未装配（启动异常，详见后端日志）</span>}
          </div>
          {/* 主区只展示实际 enabled 通道；关闭项折叠到管理区，避免停火通道被误读为正在实盘。 */}
          {liveChannels.length > 0 && (
            <div className="bg-sunken border border-line rounded-sm p-2 space-y-1">
              {activeLiveChannels.length === 0 && (
                <div className="py-2 text-center text-xs text-ink-55">当前没有正在实盘的信号通道</div>
              )}
              {activeLiveChannels.map(ch => {
                const info = SIGNAL_INFO[ch.channel]
                const draft = amountDrafts[ch.channel]
                const dirty = draft != null && draft !== String(ch.amount_usdt)
                const metrics = liveSortMetrics[ch.channel]
                return (
                  <div key={ch.channel} className={`rounded-sm border overflow-hidden ${ch.enabled ? 'border-positive bg-positive-soft' : 'border-line bg-card'}`}>
                  <div
                    role="button" tabIndex={0} aria-expanded={expandedLiveChannel === ch.channel}
                    onClick={() => toggleLiveDetails(ch.channel)}
                    onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggleLiveDetails(ch.channel) } }}
                    className="flex items-center gap-2 px-2 py-1.5 text-xs cursor-pointer"
                  >
                    <span className="flex items-center gap-1.5 min-w-0 flex-1">
                      <span className="ds-badge ds-badge-up shrink-0">实盘通道</span>
                      <span className="text-ink-95 font-medium truncate">{info?.name ?? ch.display_name}</span>
                      <HelpHint text={info?.desc ?? ch.display_name} />
                    </span>
                    <span className="shrink-0 flex items-center gap-2 font-mono text-[11px]" title="按已结算实盘订单统计；信号EV为每笔 PnL/投入的等权平均">
                      <span>胜率 <b>{metrics?.winRate == null ? '--' : `${(metrics.winRate * 100).toFixed(1)}%`}</b></span>
                      <span className={metrics?.signalEv == null ? 'text-ink-55' : metrics.signalEv >= 0 ? 'text-positive' : 'text-negative'}>信号EV <b>{metrics?.signalEv == null ? '--' : `${metrics.signalEv >= 0 ? '+' : ''}${(metrics.signalEv * 100).toFixed(1)}%`}</b></span>
                    </span>
                    <span className="shrink-0 flex items-center gap-1" onClick={event => event.stopPropagation()}>
                      <span className="text-[10px] text-ink-55">护栏</span>
                      <input
                        type="number" min={0.01} max={0.99} step={0.01}
                        value={guardDrafts[ch.channel] ?? String(ch.max_exec_price)}
                        onChange={e => setGuardDrafts(prev => ({ ...prev, [ch.channel]: e.target.value }))}
                        disabled={togglingLive}
                        title={`执行价护栏（0.01~0.99，报价均价 ≥ 护栏即弃单含贴线；预设 ${String(ch.auto_max_exec)}）`}
                        className="ds-input ds-input-sm ds-input-numeric w-14"
                      />
                      <button
                        onClick={() => handleChannelGuard(ch)}
                        disabled={togglingLive || guardDrafts[ch.channel] == null || guardDrafts[ch.channel] === String(ch.max_exec_price)}
                        className="px-1.5 py-0.5 rounded-pill border border-brand bg-card text-brand font-semibold text-[10px] disabled:opacity-40"
                        title="保存护栏热调（立即生效，持久化重启不丢）"
                      >存</button>
                      {ch.custom_max_exec && (
                        <button
                          onClick={() => handleChannelGuard(ch, true)}
                          disabled={togglingLive}
                          className="px-1 py-0.5 rounded-pill border border-warning bg-card text-warning text-[10px] disabled:opacity-40"
                          title={`已自定义覆盖，点击回落通道预设 ${String(ch.auto_max_exec)}`}
                        >↺</button>
                      )}
                    </span>
                    <span className="shrink-0 font-mono text-[10px] text-ink-55 hidden sm:inline" title="今日成交/日限 / 累计开火">
                      今日{String(ch.filled_today ?? 0)}/{String(ch.max_daily_orders)} · 开火{String(ch.fire_total)}
                    </span>
                    {(() => {
                      const p = pnlByChan[ch.channel]
                      const cnt = p?.settled_count ?? 0
                      const pnl = p?.total_pnl ?? 0
                      const roi = p?.roi != null ? p.roi * 100 : (p?.total_cost && p.total_cost > 0 ? (pnl / p.total_cost) * 100 : null)
                      return (
                        <span
                          className="shrink-0 font-mono text-[11px] px-1.5 py-0.5 rounded-sm border select-none hidden md:inline-flex items-center gap-1"
                          title={cnt > 0 ? `已结算 ${cnt} 单 · 总投入 ${(p?.total_cost ?? 0).toFixed(2)} U · 胜率 ${p?.win_rate != null ? (p.win_rate * 100).toFixed(0) : '--'}%` : '暂无已结算单'}
                          style={{
                            backgroundColor: cnt > 0 ? (pnl >= 0 ? 'var(--positive-soft)' : 'var(--negative-soft)') : 'var(--sunken)',
                            borderColor: cnt > 0 ? (pnl >= 0 ? 'var(--positive)' : 'var(--negative)') : 'var(--line)',
                          }}
                        >
                          {cnt > 0 ? (
                            <>
                              <span className={pnl >= 0 ? 'text-positive font-bold' : 'text-negative font-bold'}>
                                {pnl >= 0 ? '+' : ''}{pnl.toFixed(1)}U
                              </span>
                              <span className={`text-[10px] font-semibold ${roi != null && roi >= 0 ? 'text-positive' : 'text-negative'}`}>
                                ({roi != null ? `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%` : '--'})
                              </span>
                            </>
                          ) : (
                            <span className="text-ink-55 text-[10px]">无单</span>
                          )}
                        </span>
                      )
                    })()}
                    <span className="shrink-0 flex items-center gap-1" onClick={event => event.stopPropagation()}>
                      <input
                        type="number" min={0.1} max={50} step={0.5}
                        value={draft ?? String(ch.amount_usdt)}
                        onChange={e => setAmountDrafts(prev => ({ ...prev, [ch.channel]: e.target.value }))}
                        disabled={togglingLive}
                        title={`单笔金额（0.1~50 USDT，硬上限 ${String(live?.amount_cap ?? 50)}）`}
                        className="ds-input ds-input-sm ds-input-numeric w-16"
                      />
                      <button
                        onClick={() => handleChannelAmount(ch)}
                        disabled={togglingLive || !dirty}
                        className="px-1.5 py-0.5 rounded-pill border border-brand bg-card text-brand font-semibold disabled:opacity-40"
                        title="保存金额热调（立即生效，不影响在途任务）"
                      >存</button>
                    </span>
                    <button
                      onClick={event => { event.stopPropagation(); handleChannelToggle(ch) }}
                      disabled={togglingLive}
                      className={`shrink-0 px-2 py-0.5 rounded-pill font-bold text-white disabled:opacity-50 ${ch.enabled ? 'bg-negative' : 'bg-positive'}`}
                    >{ch.enabled ? '停火' : '开火'}</button>
                    <span className="shrink-0 text-ink-55" aria-hidden="true">{expandedLiveChannel === ch.channel ? '▲' : '▼'}</span>
                  </div>
                  {expandedLiveChannel === ch.channel && liveDetails(ch)}
                  </div>
                )
              })}
              {stoppedLiveChannels.length > 0 && (
                <div className="border-t border-line-soft mt-2 pt-2">
                  <button
                    onClick={() => setShowStoppedChannels(v => !v)}
                    className="w-full flex items-center justify-between text-xs text-ink-70 hover:text-brand"
                  >
                    <span>已停火 / 待启用通道（{stoppedLiveChannels.length}）</span>
                    <span>{showStoppedChannels ? '收起 ▲' : '展开 ▼'}</span>
                  </button>
                  {showStoppedChannels && (
                    <div className="mt-2 space-y-1">
                      {stoppedLiveChannels.map(ch => {
                        const info = SIGNAL_INFO[ch.channel]
                        const amountDraft = amountDrafts[ch.channel]
                        const amountDirty = amountDraft != null && amountDraft !== String(ch.amount_usdt)
                        const metrics = liveSortMetrics[ch.channel]
                        return (
                          <div key={ch.channel} className="rounded-sm border border-line bg-card overflow-hidden">
                          <div
                            role="button" tabIndex={0} aria-expanded={expandedLiveChannel === ch.channel}
                            onClick={() => toggleLiveDetails(ch.channel)}
                            onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggleLiveDetails(ch.channel) } }}
                            className="flex items-center gap-2 px-2 py-1.5 text-xs cursor-pointer"
                          >
                            <span className="flex items-center gap-1.5 min-w-0 flex-1">
                              <span className="ds-badge ds-badge-neutral shrink-0">已停火</span>
                              <span className="text-ink-95 font-medium truncate">{info?.name ?? ch.display_name}</span>
                              <HelpHint text={info?.desc ?? ch.display_name} />
                            </span>
                            <span className="shrink-0 flex items-center gap-2 font-mono text-[11px]" title="按已结算实盘订单统计；信号EV为每笔 PnL/投入的等权平均">
                              <span>胜率 <b>{metrics?.winRate == null ? '--' : `${(metrics.winRate * 100).toFixed(1)}%`}</b></span>
                              <span className={metrics?.signalEv == null ? 'text-ink-55' : metrics.signalEv >= 0 ? 'text-positive' : 'text-negative'}>信号EV <b>{metrics?.signalEv == null ? '--' : `${metrics.signalEv >= 0 ? '+' : ''}${(metrics.signalEv * 100).toFixed(1)}%`}</b></span>
                            </span>
                            <span className="shrink-0 flex items-center gap-1" onClick={event => event.stopPropagation()}>
                              <span className="text-[10px] text-ink-55">护栏</span>
                              <input
                                type="number" min={0.01} max={0.99} step={0.01}
                                value={guardDrafts[ch.channel] ?? String(ch.max_exec_price)}
                                onChange={e => setGuardDrafts(prev => ({ ...prev, [ch.channel]: e.target.value }))}
                                disabled={togglingLive}
                                className="ds-input ds-input-sm ds-input-numeric w-14"
                              />
                              <button
                                onClick={() => handleChannelGuard(ch)}
                                disabled={togglingLive || guardDrafts[ch.channel] == null || guardDrafts[ch.channel] === String(ch.max_exec_price)}
                                className="px-1.5 py-0.5 rounded-pill border border-brand bg-card text-brand font-semibold text-[10px] disabled:opacity-40"
                              >存</button>
                            </span>
                            <span className="shrink-0 flex items-center gap-1" onClick={event => event.stopPropagation()}>
                              <input
                                type="number" min={0.1} max={50} step={0.5}
                                value={amountDraft ?? String(ch.amount_usdt)}
                                onChange={e => setAmountDrafts(prev => ({ ...prev, [ch.channel]: e.target.value }))}
                                disabled={togglingLive}
                                className="ds-input ds-input-sm ds-input-numeric w-16"
                              />
                              <button
                                onClick={() => handleChannelAmount(ch)}
                                disabled={togglingLive || !amountDirty}
                                className="px-1.5 py-0.5 rounded-pill border border-brand bg-card text-brand font-semibold disabled:opacity-40"
                              >存</button>
                            </span>
                            <button
                              onClick={event => { event.stopPropagation(); handleChannelToggle(ch) }}
                              disabled={togglingLive}
                              className="shrink-0 px-2 py-0.5 rounded-pill font-bold text-white bg-positive disabled:opacity-50"
                            >开火</button>
                            <span className="shrink-0 text-ink-55" aria-hidden="true">{expandedLiveChannel === ch.channel ? '▲' : '▼'}</span>
                          </div>
                          {expandedLiveChannel === ch.channel && liveDetails(ch)}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )}
              <p className="text-[10px] text-ink-55 pt-0.5">
                主区仅显示实际开启通道；停火通道收进管理折叠区。金额/护栏热调仅对已开启通道显示，设置重启不丢。
              </p>
            </div>
          )}
        </div>
      </Card>
      </div>

      {/* 订单记录保留在主区；表现诊断与资金变化通过右侧抽屉查看。 */}
      {/* 最近订单：主区账户状态下方 */}
      <div className="lg:col-span-2">
        <OrdersCard orders={orders} syncing={syncing} syncResult={syncResult} onSyncBinance={handleSyncBinance} />
      </div>
    </div>

    {/* 悬浮与抽屉：下单 FAB + 右侧盈利表现 / 资金变化双 tab。 */}
    <TestTradeFab quote={quote} remainSec={remainSec} wallet={wallet} refresh={refresh}
      orders={orders} clockOffset={clockOffset} />
    <ChartDrawer orders={orders} />
    </>
  )
}

// ============================================================
// Main App
// ============================================================

const TAB_IDS = ['market', 'agent', 'monitor', 'analysis', 'live', 'notify'] as const
type TabId = (typeof TAB_IDS)[number]

/* tab 中文名。此前内联在 5 个按钮里重复 5 次，收敛为单一事实源。 */
const TAB_LABELS: Record<TabId, string> = {
  market: '市场情绪',
  agent: 'Agent 自进化',
  monitor: '运行监控',
  analysis: '信号分析',
  live: '实盘交易',
  notify: '通知配置',
}

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
    <div className="min-h-screen bg-canvas flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-card rounded-panel border border-line p-8">
        <h1 className="text-lg font-bold text-ink-95 mb-1 text-center">BTC 5min 预测系统</h1>
        <p className="text-xs text-ink-55 mb-6 text-center">请输入访问密码</p>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') submit() }}
          placeholder="访问密码"
          autoFocus
          className={`ds-input w-full ${error ? 'ds-input-error' : ''}`}
        />
        {error && <div className="mt-2 text-xs text-negative">{error}</div>}
        <button
          onClick={submit}
          disabled={loading || !password}
          className="ds-btn-primary mt-4 w-full py-2 font-semibold"
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
    <div className="min-h-screen bg-canvas flex flex-col">
      <header className="bg-card/95 border-b border-line sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-1.5 flex justify-center">
          {/* TabBar — 选中态白底墨 95，不用蓝色填充
              （One Blue Rule：蓝色只表示「可操作」与「当前选中」） */}
          <div className="ds-tabs">
            {TAB_IDS.map(id => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`ds-tab ${tab === id ? 'ds-tab-active' : ''}`}
              >
                {TAB_LABELS[id]}
              </button>
            ))}
          </div>
          <button
            onClick={handleLogout}
            title="退出登录"
            className="absolute right-4 top-1/2 -translate-y-1/2 text-[11px] text-ink-55 hover:text-negative transition"
          >
            退出
          </button>
        </div>
      </header>

      <main className="max-w-6xl w-full mx-auto px-4 py-4 flex-1">
        {tab === 'market' && (
          <div className="space-y-6">
            <Card title="BTC 5分钟内涨或跌（Binance Prediction Markets）">
              <div className="text-xs text-ink-55 mb-3">
                Binance 预测市场上所有交易者用真金白银投票的看多看空共识。每 15 秒自动刷新。
              </div>
              {pmMarket && (
                <div className="flex flex-wrap gap-4 mb-3 text-xs">
                  <span className="px-2 py-1 bg-warning-soft text-warning rounded-sm font-medium">🟡 Live</span>
                  <span className="text-ink-55">👥 {String(pmMarket.participant_count ?? '--')} 人参与</span>
                  <span className="text-ink-55">💰 ${String(pmMarket.trade_volume ?? '--')} 交易量</span>
                  {pmMarket.end_date ? (() => {
                    const remaining = Math.max(0, Math.floor(((pmMarket.end_date as number) - Date.now()) / 1000))
                    const min = Math.floor(remaining / 60)
                    const sec = remaining % 60
                    return <span className="text-warning font-mono">⏱ {min}:{String(sec).padStart(2, '0')}</span>
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
                      <CartesianGrid stroke="var(--line-soft)" />
                      <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" interval="preserveStartEnd" />
                      <YAxis tick={{ fontSize: 11 }} stroke="var(--ink-55)" domain={[0, 100]} tickFormatter={(v: number) => v + '%'} />
                      <Tooltip
                        contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                        formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n === 'up_pct' ? '看涨 (UP)' : '看跌 (DOWN)']}
                      />
                      <Area type="monotone" dataKey="up_pct" stroke="var(--positive)" fill="var(--positive-soft)" strokeWidth={2} name="看涨" connectNulls />
                      <Area type="monotone" dataKey="down_pct" stroke="var(--negative)" fill="#ef444420" strokeWidth={2} name="看跌" connectNulls />
                      <ReferenceLine y={50} stroke="var(--ink-55)" strokeDasharray="4 4" />
                    </AreaChart>
                  </ResponsiveContainer>
                  <div className="flex justify-center gap-6 mt-2 text-xs text-ink-55">
                    <span>当前: <span className="text-positive font-medium num">{pmPoints[pmPoints.length - 1]?.up_pct?.toFixed(1)}% 看涨</span> / <span className="text-negative font-medium num">{pmPoints[pmPoints.length - 1]?.down_pct?.toFixed(1)}% 看跌</span></span>
                    <span>共 {pmPoints.length} 个采样点</span>
                  </div>
                </>
              ) : (
                <div className="text-center text-ink-55 py-10 text-sm">正在采集数据...每 15 秒采样一次。</div>
              )}
            </Card>

            <FakeBreakoutPanel />

            <Market15mPanel />

            <Card title="概率动量分析（纯算法 · 独立备选方案）">
              <div className="text-xs text-ink-55 mb-3">
                基于预测市场 UP% 时序的多维度动量信号，纯算法不依赖 LLM/K线。手动触发，不参与自动决策。
              </div>
              <button
                onClick={handleMomentum}
                disabled={momentumLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-teal rounded-full hover:bg-teal-hover disabled:opacity-50 transition mb-4"
              >
                {momentumLoading ? '📊 计算中...' : '📊 运行概率动量分析'}
              </button>
              {momentumResult && (
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <DirectionBadge direction={momentumResult.direction} />
                    <span className="text-sm text-ink-80">置信度: <strong className="num">{(momentumResult.confidence * 100).toFixed(0)}%</strong></span>
                    <span className="text-sm text-ink-80">综合评分: <strong className="font-mono">{momentumResult.composite_score.toFixed(3)}</strong></span>
                  </div>
                  <div className="text-xs text-ink-55">
                    已过 {momentumResult.elapsed_seconds}s / 剩余 {momentumResult.remaining_seconds}s | {momentumResult.sample_count} 个采样点
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-line text-ink-55">
                          <th className="py-1 px-2 text-left">信号</th>
                          <th className="py-1 px-2 text-right">评分</th>
                          <th className="py-1 px-2 text-left">说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {momentumResult.signals.map((s, i) => (
                          <tr key={i} className="border-b border-line-soft hover:bg-sunken">
                            <td className="py-1 px-2 font-medium text-ink-80">{s.name}</td>
                            <td className={`py-1 px-2 text-right font-mono font-bold ${s.score > 0.1 ? 'text-positive' : s.score < -0.1 ? 'text-negative' : 'text-ink-55'}`}>
                              {s.score > 0 ? '+' : ''}{s.score.toFixed(3)}
                            </td>
                            <td className="py-1 px-2 text-ink-55">{s.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {momentumResult.reasoning.length > 0 && (
                    <div className="p-3 bg-teal-soft rounded-sm border border-teal">
                      <div className="text-xs font-bold text-teal mb-1">🧮 分析推理</div>
                      <ul className="text-xs text-ink-80 space-y-1">
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
        {tab === 'notify' && <NotifyConfigTab />}
      </main>
    </div>
  )
}

// ============================================================
// 通知配置 Tab（逐信号事件/字段）
// ============================================================

function NotifyToggle({
  checked, onChange, disabled, label,
}: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string }) {
  return (
    <label className="inline-flex items-center gap-1.5 text-xs text-ink-80 cursor-pointer select-none">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={e => onChange(e.target.checked)}
        className="accent-brand"
      />
      {label}
    </label>
  )
}

function NotifyConfigTab() {
  const [data, setData] = useState<NotifyConfigSnapshot | null>(null)
  const [q, setQ] = useState('')
  const [family, setFamily] = useState('ALL')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState(false)
  const [dirty, setDirty] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(() => {
    api.getNotifyConfig().then((d: NotifyConfigSnapshot) => {
      setData(d)
      setDirty(false)
      setError('')
    }).catch((e: unknown) => setError('加载失败: ' + String(e)))
  }, [])

  useEffect(() => { load() }, [load])

  const save = async (payload?: { global_config?: NotifyGlobalConfig; channels?: NotifyChannelPatch[] }) => {
    if (!data) return
    setSaving(true); setError('')
    try {
      const body = payload ?? {
        global_config: data.global.config,
        channels: data.channels.map(ch => ({ channel: ch.channel, enabled: ch.enabled, config: ch.config })),
      }
      const r = await api.putNotifyConfig(body) as NotifyConfigSnapshot
      if (r.channels) {
        setData(r)
        setDirty(false)
        setStatus(r.message || '已保存')
      } else {
        setError(String((r as unknown as { detail?: string }).detail || '保存失败'))
      }
    } catch (e) {
      setError('保存失败: ' + String(e))
    } finally {
      setSaving(false)
    }
  }

  const patchGlobal = (next: NotifyGlobalConfig) => {
    if (!data) return
    setData({ ...data, global: { ...data.global, config: next } })
    setDirty(true)
  }
  const patchChannel = (channel: string, next: Partial<NotifyChannelRow>) => {
    if (!data) return
    setData({
      ...data,
      channels: data.channels.map(ch => ch.channel === channel ? { ...ch, ...next } : ch),
    })
    setDirty(true)
  }

  const setAll = (on: boolean) => {
    if (!data) return
    const events = data.catalog.events
    setData({
      ...data,
      global: {
        ...data.global,
        config: {
          wechat_enabled: on,
          email_enabled: on,
          low_balance: {
            wechat: on,
            fields: Object.fromEntries(data.catalog.low_balance_fields.map(f => [f.id, on])),
          },
        },
      },
      channels: data.channels.map(ch => ({
        ...ch,
        enabled: on,
        config: {
          events: Object.fromEntries(Object.entries(events).map(([eid, meta]) => [
            eid,
            {
              ...Object.fromEntries(meta.transports.map(t => [t, on])),
              fields: Object.fromEntries(meta.fields.map(f => [f.id, on])),
            },
          ])),
        },
      })),
    })
    setDirty(true)
  }

  const reset = async (channel?: string) => {
    setSaving(true); setError('')
    try {
      const r = await api.resetNotifyConfig(channel) as NotifyConfigSnapshot
      setData(r)
      setDirty(false)
      setStatus(r.message || '已恢复默认')
    } catch (e) {
      setError('恢复失败: ' + String(e))
    } finally {
      setSaving(false)
    }
  }

  const testWechat = () => {
    api.postTestWechatNotify()
      .then((r: Record<string, unknown>) => alert(String(r?.message || '测试已派发')))
      .catch((e: unknown) => alert('测试失败: ' + String(e)))
  }
  const testEmail = () => {
    api.postTestEmailNotify()
      .then((r: Record<string, unknown>) => alert(String(r?.message || '测试已派发')))
      .catch((e: unknown) => alert('测试失败: ' + String(e)))
  }

  if (!data) {
    return <Card title="通知配置"><p className="text-sm text-ink-55">{error || '加载中…'}</p></Card>
  }

  const families = Array.from(new Set(data.channels.map(c => c.family)))
  const filtered = data.channels.filter(ch => {
    if (family !== 'ALL' && ch.family !== family) return false
    if (!q.trim()) return true
    const info = SIGNAL_INFO[ch.channel]
    const hay = `${ch.channel} ${ch.display_name} ${info?.name ?? ''} ${ch.family_label}`.toLowerCase()
    return hay.includes(q.trim().toLowerCase())
  })
  const g = data.global.config
  const phys = data.physical

  return (
    <div className="space-y-4">
      <Card title="物理通道状态">
        <p className="text-xs text-ink-55 mb-3">凭据只存在服务器 .env，这里只显示是否已配置。测试按钮直连物理通道，不受单信号路由影响。</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="border border-line rounded-sm p-3 space-y-1">
            <div className="font-medium text-ink-95 flex items-center gap-1">微信 / WxPusher <HelpHint text="WxPusher 个人号直推与企业微信群机器人共用同一业务正文。" /></div>
            <div className="text-xs text-ink-80">WxPusher：{phys.wechat.wxpusher_configured ? '已配置' : '未配置'}</div>
            <div className="text-xs text-ink-80">企业微信：{phys.wechat.wechat_work_configured ? '已配置' : '未配置'}</div>
            <div className="text-xs text-ink-80">雷达环境开关：{phys.wechat.radar_env_enabled ? '开' : '关'}</div>
            <button onClick={testWechat} className="mt-2 ds-btn-dark px-3 py-1 text-xs">测试微信</button>
          </div>
          <div className="border border-line rounded-sm p-3 space-y-1">
            <div className="font-medium text-ink-95 flex items-center gap-1">邮箱 SMTP <HelpHint text="复用 agent_alert SMTP 物理通道，开关独立于 Agent 告警。" /></div>
            <div className="text-xs text-ink-80">SMTP：{phys.email.smtp_configured ? '已配置' : '未配置'}</div>
            <div className="text-xs text-ink-80">环境总开关：{phys.email.env_enabled ? '开' : '关'}</div>
            <div className="text-xs text-ink-80">场景邮件环境开关：{phys.email.scene_email_env_enabled ? '开' : '关'}</div>
            <button onClick={testEmail} className="mt-2 ds-btn-dark px-3 py-1 text-xs">测试邮件</button>
          </div>
        </div>
      </Card>

      <Card title="全局渠道">
        <div className="flex flex-wrap gap-4 mb-3">
          <NotifyToggle checked={g.wechat_enabled} onChange={v => patchGlobal({ ...g, wechat_enabled: v })} label="微信总开关" />
          <NotifyToggle checked={g.email_enabled} onChange={v => patchGlobal({ ...g, email_enabled: v })} label="邮件总开关" />
        </div>
        <div className="text-xs text-ink-95 mb-1 flex items-center gap-1">钱包低余额告警 <HelpHint text="系统级事件，不属于某个信号通道。" /></div>
        <div className="flex flex-wrap gap-3">
          <NotifyToggle checked={g.low_balance.wechat} onChange={v => patchGlobal({ ...g, low_balance: { ...g.low_balance, wechat: v } })} label="微信推送" />
          {data.catalog.low_balance_fields.map(f => (
            <NotifyToggle
              key={f.id}
              checked={g.low_balance.fields[f.id] !== false}
              onChange={v => patchGlobal({ ...g, low_balance: { ...g.low_balance, fields: { ...g.low_balance.fields, [f.id]: v } } })}
              label={f.label}
            />
          ))}
        </div>
      </Card>

      <Card title="逐信号通知">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="搜索信号名/通道 ID" className="ds-input w-44 text-xs" />
          <select value={family} onChange={e => setFamily(e.target.value)} className="ds-input text-xs w-32">
            <option value="ALL">全部信号族</option>
            {families.map(f => <option key={f} value={f}>{data.catalog.families[f] || f}</option>)}
          </select>
          <button onClick={() => setAll(true)} className="px-2 py-1 text-xs rounded-pill border border-line">全部全开</button>
          <button onClick={() => setAll(false)} className="px-2 py-1 text-xs rounded-pill border border-line">全部全关</button>
          <button onClick={() => reset()} disabled={saving} className="px-2 py-1 text-xs rounded-pill border border-warning text-warning">恢复默认</button>
          <button onClick={() => save()} disabled={saving || !dirty} className="ds-btn-primary px-3 py-1 text-xs disabled:opacity-50">
            {saving ? '保存中…' : dirty ? '保存更改' : '已保存'}
          </button>
          {status && !dirty && <span className="text-xs text-positive">{status}</span>}
          {error && <span className="text-xs text-negative">{error}</span>}
          {dirty && <span className="text-xs text-warning">有未保存更改</span>}
        </div>
        <p className="text-[11px] text-ink-55 mb-2">默认全开，兼容现网。标题与策略通道始终保留。邮件目前只覆盖结算复盘（quote / X4 / 场景）；雷达、成交、弃单走微信。</p>
        <div className="space-y-2">
          {filtered.map(ch => {
            const info = SIGNAL_INFO[ch.channel]
            const open = !!expanded[ch.channel]
            return (
              <div key={ch.channel} className={`border rounded-sm p-2 ${ch.enabled ? 'border-line bg-card' : 'border-line bg-sunken opacity-90'}`}>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={() => setExpanded(p => ({ ...p, [ch.channel]: !open }))}
                    className="text-xs text-ink-55 w-5"
                  >{open ? '▾' : '▸'}</button>
                  <span className="ds-badge ds-badge-neutral">{ch.family_label}</span>
                  <span className="text-sm font-medium text-ink-95">{info?.name ?? ch.display_name}</span>
                  <HelpHint text={info?.desc ?? ch.display_name} />
                  <span className="text-[10px] font-mono text-ink-55 hidden md:inline">{ch.channel}</span>
                  <span className="ml-auto" />
                  <NotifyToggle checked={ch.enabled} onChange={v => patchChannel(ch.channel, { enabled: v })} label="本信号通知" />
                </div>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 pl-6">
                  {Object.entries(data.catalog.events).map(([eid, meta]) => {
                    const block = ch.config.events[eid] || { fields: {} }
                    return (
                      <div key={eid} className="text-xs text-ink-80 flex items-center gap-2">
                        <span className="text-ink-55" title={meta.hint}>{meta.label}</span>
                        {meta.transports.includes('wechat') && (
                          <NotifyToggle checked={block.wechat !== false} onChange={v => patchChannel(ch.channel, { config: { events: { ...ch.config.events, [eid]: { ...block, wechat: v } } } })} label="微信" />
                        )}
                        {meta.transports.includes('email') && (
                          <NotifyToggle checked={block.email !== false} onChange={v => patchChannel(ch.channel, { config: { events: { ...ch.config.events, [eid]: { ...block, email: v } } } })} label="邮件" />
                        )}
                      </div>
                    )
                  })}
                </div>
                {open && (
                  <div className="mt-2 pl-6 space-y-2">
                    {Object.entries(data.catalog.events).map(([eid, meta]) => {
                      const block = ch.config.events[eid] || { fields: {} }
                      return (
                        <div key={eid}>
                          <div className="text-[11px] text-ink-55 mb-1">{meta.label}字段</div>
                          <div className="flex flex-wrap gap-x-3 gap-y-1">
                            {meta.fields.map(f => (
                              <span key={f.id} className="inline-flex items-center gap-1">
                                <NotifyToggle
                                  checked={block.fields?.[f.id] !== false}
                                  onChange={v => patchChannel(ch.channel, {
                                    config: {
                                      events: {
                                        ...ch.config.events,
                                        [eid]: { ...block, fields: { ...block.fields, [f.id]: v } },
                                      },
                                    },
                                  })}
                                  label={f.label}
                                />
                                <HelpHint text={f.hint} />
                              </span>
                            ))}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
          {filtered.length === 0 && <p className="text-sm text-ink-55">无匹配信号</p>}
        </div>
      </Card>
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
      <div className="bg-card rounded-card border border-line px-5 py-2.5 shrink-0 flex items-center gap-4">
        {status ? (
          <div className="flex items-center justify-around gap-4 flex-1">
            <div className="flex items-center gap-2">
              <StatusDot ok={status.scheduler_running} />
              <span className="text-xs text-ink-55">调度器</span>
              <span className={`text-sm font-bold ${status.scheduler_running ? 'text-positive' : 'text-negative'}`}>
                {status.scheduler_running ? '运行中' : '已停止'}
              </span>
            </div>
            <div className="h-4 w-px bg-line" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-55">ACTIVE 模式</span>
              <span className="text-sm font-bold text-ink-95 font-mono">{status.active_pattern_count}</span>
            </div>
            <div className="h-4 w-px bg-line" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-55">累计验证</span>
              <span className="text-sm font-bold text-ink-95 font-mono">{status.validate_counter}</span>
            </div>
            <div className="h-4 w-px bg-line" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-ink-55">距下次进化</span>
              <span className="text-sm font-bold text-ink-95 font-mono">
                {status.evolve_trigger_mode === 'samples' && status.evolve_min_new_samples != null
                  ? `${status.new_validated_since_evolve ?? 0}/${status.evolve_min_new_samples}`
                  : '—'}
              </span>
            </div>
          </div>
        ) : <div className="text-ink-55 text-center text-sm flex-1">加载中...</div>}
        <button
          onClick={() => setDlOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-violet rounded-full hover:bg-violet-hover transition"
          title="全量历史深度分析：预览发现结果，审核后写入模式库"
        >
          🔬 深度学习
        </button>
        <button
          onClick={() => setCmpOpen(true)}
          className="shrink-0 px-3 py-1.5 text-xs font-semibold text-white bg-teal rounded-full hover:bg-teal-hover transition"
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
            <div className="text-[11px] text-ink-55">LLM 自主发现的情绪曲线模式，点击行查看详情</div>
            <button onClick={refreshPatterns} className="px-2 py-0.5 text-[11px] rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover transition">刷新</button>
          </div>
          {patterns.length === 0 ? (
            <div className="text-center text-ink-55 py-10 text-sm">暂无模式（需积累情绪窗口后自动发现）</div>
          ) : (
            <div className="overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-line text-ink-55">
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
                      <tr className="border-b border-line-soft hover:bg-sunken cursor-pointer" onClick={() => setExpandedPattern(expandedPattern === p.id ? null : p.id)}>
                        <td className="py-1.5 px-1.5 font-medium text-ink-95">{p.pattern_name}</td>
                        <td className="py-1.5 px-1.5"><DiscoveryMethodBadge method={p.discovery_method} /></td>
                        <td className="py-1.5 px-1.5"><DirectionBadge direction={p.predicted_direction} /></td>
                        <td className="py-1.5 px-1.5"><StatusBadge status={p.status} /></td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.win_rate * 100).toFixed(1)}%</td>
                        <td className="py-1.5 px-1.5 text-right font-mono text-ink-55">{p.holdout_win_rate != null ? `${(p.holdout_win_rate * 100).toFixed(0)}%` : '—'}</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{p.sample_count}</td>
                        <td className="py-1.5 px-1.5 text-right font-mono">{(p.confidence_score * 100).toFixed(0)}%</td>
                      </tr>
                      {expandedPattern === p.id && (
                        <tr className="bg-sunken">
                          <td colSpan={8} className="py-2 px-3">
                            <div className="text-xs text-ink-80 mb-2"><b>描述：</b>{p.description}</div>
                            <div className="grid grid-cols-2 gap-2 mb-2">
                              <div>
                                <div className="text-[10px] font-bold text-ink-55 mb-1">曲线特征</div>
                                <pre className="text-[10px] bg-card p-1.5 rounded-sm border border-line overflow-x-auto max-h-32">{JSON.stringify(p.curve_features, null, 2)}</pre>
                              </div>
                              <div>
                                <div className="text-[10px] font-bold text-ink-55 mb-1">适用条件</div>
                                <pre className="text-[10px] bg-card p-1.5 rounded-sm border border-line overflow-x-auto max-h-32">{JSON.stringify(p.conditions, null, 2)}</pre>
                              </div>
                            </div>
                            <button onClick={(e) => { e.stopPropagation(); toggleHistory(p.id) }} className="px-2 py-0.5 text-[10px] rounded-sm text-ink-55 hover:text-ink-95 transition-colors">
                              {historyFor === p.id ? '收起进化轨迹' : '查看进化轨迹'}
                            </button>
                            {historyFor === p.id && (
                              <div className="mt-2">
                                {loading ? <div className="text-[10px] text-ink-55">加载中...</div> : history.length === 0 ? (
                                  <div className="text-[10px] text-ink-55">暂无变更记录</div>
                                ) : (
                                  <ul className="space-y-1.5">
                                    {history.map(h => (
                                      <li key={h.id} className="flex items-start gap-1.5 text-[10px]">
                                        <span className="text-ink-55 shrink-0 font-mono">{fmtTime(h.created_at)}</span>
                                        <ChangeTypeBadge type={h.change_type} />
                                        <span className="px-1 py-0.5 rounded-pill bg-sunken text-ink-55 shrink-0">{h.phase}</span>
                                        <span className="text-ink-80">{h.change_reason}</span>
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
              className="px-2 py-0.5 border border-line rounded-sm text-[11px] focus:outline-none focus:ring-2 focus:ring-brand"
            >
              <option value="">全部方向</option>
              <option value="UP">UP</option>
              <option value="DOWN">DOWN</option>
              <option value="NO_TRADE">NO_TRADE</option>
            </select>
            <button onClick={() => refreshPredictions(dirFilter || undefined)} className="px-2 py-0.5 text-[11px] rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover transition">刷新</button>
          </div>
          {predictions.length === 0 ? (
            <div className="text-center text-ink-55 py-10 text-sm">暂无预测记录</div>
          ) : (
            <div className="overflow-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-line text-ink-55">
                    <th className="py-1.5 px-1.5 text-left">时间</th>
                    <th className="py-1.5 px-1.5 text-left">方向</th>
                    <th className="py-1.5 px-1.5 text-right">置信度</th>
                    <th className="py-1.5 px-1.5 text-left">匹配模式</th>
                    <th className="py-1.5 px-1.5 text-left">验证</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map(p => (
                    <tr key={p.id} className="border-b border-line-soft hover:bg-sunken">
                      <td className="py-1.5 px-1.5 text-ink-80 font-mono">{fmtTime(p.prediction_time)}</td>
                      <td className="py-1.5 px-1.5"><DirectionBadge direction={p.predicted_direction} /></td>
                      <td className="py-1.5 px-1.5 text-right font-mono">{(p.confidence * 100).toFixed(0)}%</td>
                      <td className="py-1.5 px-1.5 text-ink-80 truncate max-w-[120px]" title={p.matched_pattern_name || ''}>{p.matched_pattern_name || '—'}</td>
                      <td className="py-1.5 px-1.5">
                        {p.is_correct === null ? <span className="text-ink-55">待验证</span> :
                          p.is_correct ? <span className="text-positive font-bold">✓ {p.actual_outcome}</span> :
                            <span className="text-negative font-bold">✗ {p.actual_outcome}</span>}
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
  BEATS_RANDOM: { label: '已显著跑赢随机', cls: 'bg-positive-soft text-positive border-positive' },
  INCONCLUSIVE: { label: '尚未显著', cls: 'bg-warning-soft text-warning border-warning' },
  INSUFFICIENT_SAMPLES: { label: '样本不足', cls: 'bg-sunken text-ink-55 border-line' },
}

function evoPct(v: number | null | undefined): string {
  return typeof v === 'number' ? (v * 100).toFixed(1) + '%' : '--'
}

function EvoStat({ label, value, tone = 'text-ink-95' }: { label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className="bg-sunken rounded-sm px-3 py-2 border border-line-soft">
      <div className="text-[10px] text-ink-55">{label}</div>
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
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-overlay p-4" onClick={onClose}>
      <div className="bg-card rounded-panel border border-line w-full max-w-4xl max-h-[88vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-line flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-sm font-bold text-ink-95">📈 进化有效性看板</h2>
            <p className="text-[11px] text-ink-55 mt-0.5">用样本外胜率证明「在变好」而非「只在变化」。仅统计已验证的决策预测（UP/DOWN），随机基线 50%。</p>
          </div>
          <div className="flex items-center gap-2">
            <select value={days} onChange={e => setDays(Number(e.target.value))}
              className="px-2 py-1 border border-line rounded-sm text-[11px] focus:outline-none focus:ring-2 focus:ring-brand">
              <option value={7}>近 7 天</option>
              <option value={30}>近 30 天</option>
              <option value={90}>近 90 天</option>
            </select>
            <button onClick={onClose} className="text-ink-55 hover:text-ink-80 text-xl leading-none px-1">✕</button>
          </div>
        </div>

        {/* 主体 */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-4">
          {loading && <div className="text-center text-ink-55 py-10 text-sm">加载中...</div>}
          {!loading && !report && <div className="text-center text-ink-55 py-10 text-sm">暂无数据</div>}
          {!loading && report && (
            <>
              {/* 结论横幅 */}
              <div className={`rounded-sm border px-4 py-3 ${vm.cls}`}>
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-xs font-bold">{vm.label}</span>
                  <span className="text-[10px] opacity-70">窗口 {report.window_days} 天 · 已验证 {report.total_validated} 条 · 决策 {report.decisive_count} · 弃权(NO_TRADE) {report.no_trade_count}</span>
                </div>
                <p className="text-xs leading-relaxed">{report.summary}</p>
              </div>

              {/* 总体指标 */}
              <div>
                <div className="text-[11px] font-semibold text-ink-55 mb-1.5">总体（决策样本 = UP/DOWN）</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <EvoStat label="决策胜率" value={evoPct(report.overall.win_rate)} tone={report.overall.win_rate >= 0.5 ? 'text-positive' : 'text-negative'} />
                  <EvoStat label="Wilson 95% 下界" value={evoPct(report.overall.ci_lower)} tone={report.overall.beats_random ? 'text-positive' : 'text-ink-95'} />
                  <EvoStat label="超额（vs 50%）" value={(report.overall.excess_over_random >= 0 ? '+' : '') + (report.overall.excess_over_random * 100).toFixed(1) + '%'} tone={report.overall.excess_over_random >= 0 ? 'text-positive' : 'text-negative'} />
                  <EvoStat label="跑赢随机？" value={report.overall.beats_random ? '是 ✓' : '否'} tone={report.overall.beats_random ? 'text-positive' : 'text-ink-55'} />
                </div>
              </div>

              {/* 样本外胜率趋势 */}
              <div>
                <div className="text-[11px] font-semibold text-ink-55 mb-1.5">样本外胜率趋势（按天）</div>
                {report.trend_daily.length === 0 ? (
                  <div className="text-center text-ink-55 py-6 text-xs">暂无按天数据</div>
                ) : (
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={report.trend_daily.map(d => ({
                      date: d.date.slice(5),
                      win: +(d.win_rate * 100).toFixed(1),
                      ci: +(d.ci_lower * 100).toFixed(1),
                    }))}>
                      <CartesianGrid stroke="var(--line-soft)" />
                      <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
                      <YAxis tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" domain={[0, 100]} tickFormatter={(v: number) => v + '%'} />
                      <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                        formatter={(v, n) => [typeof v === 'number' ? v + '%' : '--', n === 'win' ? '胜率' : 'Wilson 下界']} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <ReferenceLine y={50} stroke="var(--ink-55)" strokeDasharray="4 4" />
                      <Line type="monotone" dataKey="win" stroke="var(--chart-1)" strokeWidth={2} name="胜率" dot={{ r: 2 }} />
                      <Line type="monotone" dataKey="ci" stroke="var(--chart-4)" strokeWidth={1.5} strokeDasharray="4 3" name="Wilson 下界" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>

              {/* 代际对比 */}
              <div>
                <div className="text-[11px] font-semibold text-ink-55 mb-1.5">代际对比（前半程 vs 近半程）</div>
                {!report.generations.comparable ? (
                  <div className="text-xs text-ink-55 bg-sunken rounded-sm px-3 py-2 border border-line-soft">两半程样本不足（各需 ≥15 决策样本），暂不下改善结论。</div>
                ) : (
                  <>
                    <div className="grid grid-cols-3 gap-2">
                      <EvoStat label={`前半程（n=${report.generations.older_half.sample_count}）`} value={evoPct(report.generations.older_half.win_rate)} />
                      <EvoStat label={`近半程（n=${report.generations.newer_half.sample_count}）`} value={evoPct(report.generations.newer_half.win_rate)} tone={report.generations.win_rate_delta >= 0 ? 'text-positive' : 'text-negative'} />
                      <EvoStat label="Δ 胜率" value={(report.generations.win_rate_delta >= 0 ? '+' : '') + (report.generations.win_rate_delta * 100).toFixed(1) + '%'} tone={report.generations.significant_improvement ? 'text-positive' : report.generations.win_rate_delta > 0 ? 'text-warning' : 'text-negative'} />
                    </div>
                    <div className="text-[11px] text-ink-55 mt-1">
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
                <div className="text-[11px] font-semibold text-ink-55 mb-1.5">按发现方法分轨（哪条轨道真的产出 alpha）</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-line text-ink-55">
                      <th className="py-1 px-2 text-left">方法</th>
                      <th className="py-1 px-2 text-right">样本</th>
                      <th className="py-1 px-2 text-right">胜率</th>
                      <th className="py-1 px-2 text-right">Wilson 下界</th>
                      <th className="py-1 px-2 text-center">跑赢随机</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.by_discovery_method).map(([m, s]) => (
                      <tr key={m} className="border-b border-line-soft hover:bg-sunken">
                        <td className="py-1 px-2 font-medium text-ink-80">{m}</td>
                        <td className="py-1 px-2 text-right font-mono">{s.sample_count}</td>
                        <td className="py-1 px-2 text-right font-mono">{evoPct(s.win_rate)}</td>
                        <td className="py-1 px-2 text-right font-mono">{evoPct(s.ci_lower)}</td>
                        <td className="py-1 px-2 text-center">{s.beats_random ? <span className="text-positive font-bold">✓</span> : <span className="text-ink-40">—</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="text-[10px] text-ink-55 text-right">生成于 <span className="num">{report.generated_at ? new Date(report.generated_at).toLocaleString('zh-CN') : '--'}</span></div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ============================================================
// Agent 运行健康子视图（GET /api/agent/health，30s 轮询；原「运行监控」页内容）
// ============================================================

function healthTone(status: string): { dot: string; text: string; label: string } {
  if (status === 'OK') return { dot: 'bg-positive', text: 'text-positive', label: '正常' }
  if (status === 'WARN') return { dot: 'bg-warning', text: 'text-warning', label: '警告' }
  return { dot: 'bg-negative', text: 'text-negative', label: '严重' }
}

function MetricKV({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-1 border-b border-line-soft last:border-0">
      <span className="text-[11px] text-ink-55">{label}</span>
      <span className="text-xs font-mono font-medium text-ink-95">{value}</span>
    </div>
  )
}

function fmtNum(v: unknown, digits = 2): string {
  if (v == null || typeof v !== 'number' || Number.isNaN(v)) return '--'
  return v.toFixed(digits)
}

function AgentHealthView() {
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
    return <div className="text-center text-ink-55 py-16 text-sm">{err || '加载健康报告中...'}</div>
  }

  const tone = healthTone(report.overall_status)
  const wc = report.window_continuity || {}
  const ps = report.predict_stats || {}
  const matchRate = typeof ps.match_rate === 'number' ? ps.match_rate : null
  const dirDist = (ps.direction_distribution as Record<string, number> | undefined) || {}

  return (
    <div className="space-y-3">
      {/* 总体状态红黄绿灯 + 诊断文本 */}
      <div className="bg-card rounded-card border border-line px-5 py-3">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <span className={`inline-block w-3.5 h-3.5 rounded-full ${tone.dot} animate-pulse`} />
            <span className={`text-base font-bold ${tone.text}`}>总体状态：{tone.label}</span>
            <span className="text-[11px] text-ink-55 font-mono">
              {report.generated_at ? new Date(report.generated_at).toLocaleString('zh-CN') : ''}
            </span>
          </div>
          <button onClick={refresh} className="px-2 py-0.5 text-[11px] rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover transition">立即刷新</button>
        </div>
        {report.summary && (
          <div className="mt-2 text-xs text-ink-80 bg-sunken rounded-sm p-3 whitespace-pre-wrap break-words">{report.summary}</div>
        )}
      </div>

      {/* 告警列表 */}
      <Card title={`告警（${report.alerts.length}）`}>
        {report.alerts.length === 0 ? (
          <div className="text-center text-ink-55 py-4 text-sm">无告警</div>
        ) : (
          <ul className="space-y-1.5">
            {report.alerts.map((a, i) => (
              <li key={i} className="flex items-start gap-2 text-xs">
                <span className={`px-1.5 py-0.5 rounded-pill font-bold shrink-0 ${a.level === 'CRITICAL' ? 'bg-negative-soft text-negative' : 'bg-warning-soft text-warning'}`}>{a.level}</span>
                <span className="font-mono text-ink-55 shrink-0">{a.code}</span>
                <span className="text-ink-80">{a.message}</span>
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
            <div className="text-center text-ink-55 py-3 text-xs">无内存态数据</div>
          ) : (
            Object.entries(report.scheduler).map(([k, v]) => (
              <MetricKV key={k} label={k} value={typeof v === 'object' ? JSON.stringify(v) : String(v)} />
            ))
          )}
        </Card>

        {/* LLM 指标 */}
        <Card title="LLM 指标">
          {Object.keys(report.llm || {}).length === 0 ? (
            <div className="text-center text-ink-55 py-3 text-xs">无内存态数据</div>
          ) : (
            Object.entries(report.llm).map(([k, v]) => (
              <MetricKV key={k} label={k} value={typeof v === 'object' ? JSON.stringify(v) : String(v)} />
            ))
          )}
        </Card>

        {/* 影子检测器运行态（2026-10-10）：采样循环型检测器不看「有没有落表」也能判断
            是否在跑——触发稀时（如 gap_crowd 约 19 次/天）此前无法区分「没触发」与「没跑」 */}
        <Card title="影子检测器">
          {Object.keys(report.detectors || {}).length === 0 ? (
            <div className="text-center text-ink-55 py-3 text-xs">无数据</div>
          ) : (
            Object.entries(report.detectors || {}).map(([k, v]) => (
              <MetricKV
                key={k}
                label={k}
                value={v == null
                  ? '未装配'
                  : Object.entries(v).map(([kk, vv]) => `${kk}=${vv}`).join(' · ')}
              />
            ))
          )}
        </Card>
      </div>

      {/* 置信度校准分桶 */}
      <Card title="置信度校准（分桶）">
        {report.calibration.length === 0 ? (
          <div className="text-center text-ink-55 py-4 text-sm">样本不足，暂无校准数据</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line text-ink-55">
                <th className="py-1.5 px-1.5 text-left">区间</th>
                <th className="py-1.5 px-1.5 text-right">样本数</th>
                <th className="py-1.5 px-1.5 text-right">平均置信度</th>
                <th className="py-1.5 px-1.5 text-right">实际命中率</th>
                <th className="py-1.5 px-1.5 text-right">偏差 (gap)</th>
              </tr>
            </thead>
            <tbody>
              {report.calibration.map((b, i) => (
                <tr key={i} className="border-b border-line-soft hover:bg-sunken">
                  <td className="py-1.5 px-1.5 font-mono text-ink-80">{b.range}</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{b.count}</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{(b.avg_confidence * 100).toFixed(0)}%</td>
                  <td className="py-1.5 px-1.5 text-right font-mono">{b.hit_rate != null ? `${(b.hit_rate * 100).toFixed(0)}%` : '--'}</td>
                  <td className={`py-1.5 px-1.5 text-right font-mono ${b.gap == null ? 'text-ink-55' : b.gap > 0.1 ? 'text-negative' : b.gap < -0.1 ? 'text-brand' : 'text-ink-80'}`}>
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
// 运行监控 Tab（2026-10-06 改造 P1/P2/P3 完整版）：信号体检（影子+实盘统一红黄绿灯、Regime分桶、死因诊断、共振矩阵）
// 数据：GET /api/signals/health（60s 轮询，后端 5min 缓存）；下钻 /health/detail；AI 诊断仅预留。
// 口径：红灯 = 优势已变负的后验概率 ≥90% 且 n≥30；只建议，不自动关通道。
// ============================================================

interface SignalHealthMetrics {
  n: number; wins: number; win_rate: number | null; win_rate_ci95: [number, number] | null
  avg_ev: number | null; mean_breakeven: number | null; edge: number | null
  p_edge_negative: number | null; bench_win_rate: number | null; p_below_bench: number | null
  ev_ex_top1: number | null; ev_ex_top3: number | null; ev_ex_top5: number | null
  top5_share: number | null; last_ts: number | null
  recent20: { n: number; win_rate: number | null; avg_ev: number | null } | null
  prev20: { n: number; win_rate: number | null; avg_ev: number | null } | null
}
type HealthLight = 'GRAY' | 'GREEN' | 'YELLOW' | 'RED'
type DeathStatus = 'ALIVE' | 'EXPIRED' | 'SPURIOUS'

interface ChangePointInfo {
  change_ts: number | null
  peak_cum_ev: number
  current_cum_ev: number
  drawdown_ev: number
  post_peak_n: number
  is_declining: boolean
}

interface ChannelCorrelation {
  channel_a: string
  channel_b: string
  shared_windows: number
  jaccard: number
  agreement_rate: number
}

interface SignalHealthRow {
  scope: 'live' | 'shadow'; key: string; light: HealthLight; reasons: string[]
  metrics: SignalHealthMetrics; family: string | null; market_period: string | null
  enabled: boolean; retired: boolean; active: boolean
  death_status?: DeathStatus
  death_reason?: string
  change_point?: ChangePointInfo | null
}

interface SignalHealthReport {
  generated_at: number; tally_active: Record<HealthLight, number>
  thresholds: Record<string, number>; rows: SignalHealthRow[]
  correlations?: ChannelCorrelation[]
}

interface HealthBucket {
  segment: string; n: number; win_rate: number; win_rate_ci95: [number, number]
  avg_ev: number | null; small_sample: boolean
}

interface SignalHealthDetail {
  scope: string; key: string; light: HealthLight; reasons: string[]; metrics: SignalHealthMetrics
  series: { ts: number; i: number; cum_ev: number; roll20_win: number | null; p_neg: number | null }[]
  price_buckets: HealthBucket[]; hour_buckets: HealthBucket[]
  trend_4h_buckets?: HealthBucket[]; trend_24h_buckets?: HealthBucket[]; volatility_buckets?: HealthBucket[]
  death_status?: DeathStatus; death_reason?: string
  change_point?: ChangePointInfo | null
  history: { date: string; light: HealthLight; n: number; win_rate: number | null; avg_ev: number | null; p_edge_negative: number | null }[]
  diagnosis_payload: Record<string, unknown>
}

const HEALTH_LIGHT: Record<HealthLight, { label: string; color: string; order: number }> = {
  RED: { label: '红灯', color: 'var(--negative)', order: 0 },
  YELLOW: { label: '黄灯', color: 'var(--warning)', order: 1 },
  GREEN: { label: '绿灯', color: 'var(--positive)', order: 2 },
  GRAY: { label: '样本不足', color: 'var(--ink-55)', order: 3 },
}

function LightDot({ light }: { light: HealthLight }) {
  return <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: HEALTH_LIGHT[light].color }} title={HEALTH_LIGHT[light].label} />
}

function DeathBadge({ status, reason }: { status?: DeathStatus; reason?: string }) {
  if (!status || status === 'ALIVE') return null
  const isExp = status === 'EXPIRED'
  return (
    <span
      className={`rounded-pill px-1.5 py-0.5 text-[9px] font-bold shrink-0 ${isExp ? 'bg-warning-soft text-warning' : 'bg-negative-soft text-negative'}`}
      title={reason ?? (isExp ? '曾显著盈利后衰减' : '从未建立超额优势')}
    >
      {isExp ? '衰减' : '虚假'}
    </span>
  )
}

function BucketTable({ title, rows, hint }: { title: string; rows: HealthBucket[]; hint: string }) {
  return (
    <div>
      <div className="text-xs font-bold text-ink-95 mb-1">{title} <HelpHint text={hint} /></div>
      <table className="w-full text-[11px]">
        <thead><tr className="text-ink-55 border-b border-line-soft">
          <th className="py-1 text-left">区间</th><th className="py-1 text-right">n</th>
          <th className="py-1 text-right">胜率</th><th className="py-1 text-right">95%区间</th><th className="py-1 text-right">均EV</th>
        </tr></thead>
        <tbody>{rows.map(b => (
          <tr key={b.segment} className={`border-b border-line-soft ${b.small_sample ? 'text-ink-55' : ''}`}>
            <td className="py-1 font-mono">{b.segment}{b.small_sample && <span className="ml-1 text-warning" title="样本不足 10，仅供参考">*</span>}</td>
            <td className="py-1 text-right font-mono">{b.n}</td>
            <td className="py-1 text-right font-mono">{performancePct(b.win_rate)}</td>
            <td className="py-1 text-right font-mono">{performancePct(b.win_rate_ci95[0], 0)}~{performancePct(b.win_rate_ci95[1], 0)}</td>
            <td className={`py-1 text-right font-mono ${signClass(b.avg_ev)}`}>{performanceSigned(b.avg_ev)}</td>
          </tr>))}
        </tbody>
      </table>
    </div>
  )
}

function SignalHealthDrill({ scope, keyName, onClose }: { scope: string; keyName: string; onClose: () => void }) {
  const [d, setD] = useState<SignalHealthDetail | null>(null)
  const [err, setErr] = useState('')
  const [diag, setDiag] = useState<Record<string, unknown> | null>(null)
  const [diagBusy, setDiagBusy] = useState(false)
  const [bucketTab, setBucketTab] = useState<'quote' | 'regime'>('quote')

  useEffect(() => {
    let alive = true
    setD(null); setErr(''); setDiag(null)
    api.getSignalsHealthDetail(scope, keyName)
      .then((r: SignalHealthDetail & { detail?: string }) => {
        if (!alive) return
        if (r.metrics) setD(r)
        else setErr(r.detail ?? '加载失败')
      })
      .catch(() => { if (alive) setErr('下钻数据获取失败') })
    return () => { alive = false }
  }, [scope, keyName])

  const runDiagnose = async () => {
    setDiagBusy(true)
    try { setDiag(await api.postSignalsHealthDiagnose(scope, keyName)) }
    catch { setDiag({ status: 'error', message: '请求失败' }) }
    finally { setDiagBusy(false) }
  }

  const m = d?.metrics
  const cp = d?.change_point
  return (
    <div className="bg-card rounded-card border border-line p-4 space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        {d && <LightDot light={d.light} />}
        <span className="font-bold text-sm text-ink-95">{SIGNAL_INFO[keyName]?.name ?? keyName}</span>
        <span className="text-[11px] text-ink-55 font-mono">{scope === 'live' ? '实盘' : '影子'} · {keyName}</span>
        {d?.death_status && d.death_status !== 'ALIVE' && (
          <DeathBadge status={d.death_status} reason={d.death_reason} />
        )}
        <button onClick={onClose} className="ml-auto px-2 py-0.5 text-[11px] rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover transition">收起</button>
      </div>
      {!d && <div className="text-center text-ink-55 py-6 text-sm">{err || '加载下钻中...'}</div>}
      {d && m && (
        <>
          {d.reasons.length > 0 && (
            <ul className="text-xs text-ink-80 bg-sunken rounded-sm px-3 py-2 space-y-0.5">{d.reasons.map((r, i) => <li key={i}>· {r}</li>)}</ul>
          )}

          {cp?.is_declining && (
            <div className="text-xs rounded-sm bg-warning-soft text-warning px-3 py-2 flex items-center justify-between">
              <span>
                ⚠️ <strong>结构拐折点预警</strong>：该通道在 {cp.change_ts ? performanceTime(cp.change_ts) : '--'} 达到累计收益峰值 ({cp.peak_cum_ev} EV)，此后衰退 {cp.post_peak_n} 笔，累计回撤 <strong>-{cp.drawdown_ev} EV</strong>
              </span>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6">
            <MetricKV label="样本 n" value={m.n} />
            <MetricKV label="胜率" value={`${performancePct(m.win_rate)}（${m.win_rate_ci95 ? `${performancePct(m.win_rate_ci95[0], 0)}~${performancePct(m.win_rate_ci95[1], 0)}` : '--'}）`} />
            <MetricKV label="平均保本" value={performancePct(m.mean_breakeven)} />
            <MetricKV label="优势 = 胜率−保本" value={performanceSigned(m.edge == null ? null : m.edge * 100, '%', 1)} />
            <MetricKV label="平均 EV" value={performanceSigned(m.avg_ev, '', 3)} />
            <MetricKV label="剔除最赚1/3/5单" value={`${performanceSigned(m.ev_ex_top1, '', 2)} / ${performanceSigned(m.ev_ex_top3, '', 2)} / ${performanceSigned(m.ev_ex_top5, '', 2)}`} />
            <MetricKV label="前5单占正收益" value={performancePct(m.top5_share, 0)} />
            <MetricKV label="优势变负概率" value={performancePct(m.p_edge_negative, 0)} />
            <MetricKV label="冻结基准胜率" value={performancePct(m.bench_win_rate)} />
            <MetricKV label="低于基准 p 值" value={m.p_below_bench == null ? '--' : m.p_below_bench.toFixed(3)} />
            <MetricKV label="近20 / 前20 胜率" value={`${performancePct(m.recent20?.win_rate, 0)} / ${performancePct(m.prev20?.win_rate, 0)}`} />
            <MetricKV label="死因判定" value={d.death_status === 'ALIVE' ? '存活中' : d.death_status === 'EXPIRED' ? '过期衰减' : '假规律'} />
          </div>

          {d.series.length > 1 && (
            <div>
              <div className="text-xs font-bold text-ink-95 mb-1">
                累计 EV 与「优势变负概率」走势 <HelpHint text="左轴=累计EV（每笔单位本金）；右轴=优势变负概率与滚动20胜率。红色虚线=红灯线90%：概率升破该线表示优势已很可能为负。" />
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <ComposedChart data={d.series}>
                  <CartesianGrid stroke="var(--line-soft)" />
                  <XAxis dataKey="ts" tickFormatter={performanceTime} minTickGap={50} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                  <YAxis yAxisId="l" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                  <YAxis yAxisId="r" orientation="right" domain={[0, 1]} tickFormatter={(v: number) => (Math.round(v * 100) + '%')} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={v => performanceTime(Number(v))} />
                  <Legend />
                  <ReferenceLine yAxisId="r" y={0.9} stroke="var(--negative)" strokeDasharray="5 4" />
                  <Line yAxisId="l" name="累计EV" dataKey="cum_ev" stroke="var(--chart-1)" strokeWidth={2} dot={false} />
                  <Line yAxisId="r" name="优势变负概率" dataKey="p_neg" stroke="var(--negative)" strokeWidth={1.5} dot={false} connectNulls />
                  <Line yAxisId="r" name="滚动20胜率" dataKey="roll20_win" stroke="var(--chart-3)" strokeWidth={1.5} dot={false} connectNulls />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* 分桶维度切换：报价与时段 / 行情与波动 */}
          <div className="space-y-2 border-t border-line-soft pt-2">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-ink-95">分桶诊断维度</span>
              <div className="flex items-center gap-0.5 bg-sunken rounded-pill p-0.5 border border-line-soft">
                <button
                  className={`px-2 py-0.5 rounded-pill text-[11px] font-semibold transition ${bucketTab === 'quote' ? 'bg-brand text-white' : 'text-ink-55 hover:text-ink-80'}`}
                  onClick={() => setBucketTab('quote')}
                >
                  报价与时段
                </button>
                <button
                  className={`px-2 py-0.5 rounded-pill text-[11px] font-semibold transition ${bucketTab === 'regime' ? 'bg-brand text-white' : 'text-ink-55 hover:text-ink-80'}`}
                  onClick={() => setBucketTab('regime')}
                >
                  趋势与波动 (P2 Regime)
                </button>
              </div>
            </div>

            {bucketTab === 'quote' && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <BucketTable title="按入场价分桶" rows={d.price_buckets} hint="探索性分析：小样本桶（带*）仅供参考；发现的规律必须经预注册与前向验证后才能变成护栏，不可直接改实盘。" />
                <BucketTable title="按 UTC 时段分桶" rows={d.hour_buckets} hint="以窗口起点的 UTC 小时分 6 档（北京时间 +8）。探索性分析，同上纪律。" />
              </div>
            )}

            {bucketTab === 'regime' && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <BucketTable title="4h 趋势分层" rows={d.trend_4h_buckets || []} hint="触发时点前 48 根 5m K 收益（严格事前口径）。" />
                <BucketTable title="24h 趋势分层" rows={d.trend_24h_buckets || []} hint="触发时点前 288 根 5m K 收益（事前口径）。" />
                <BucketTable title="4h 波动率分层" rows={d.volatility_buckets || []} hint="过去 48 根 5m K 收益率波动标准差。" />
              </div>
            )}
          </div>

          {d.history.length > 1 && (
            <div className="text-[11px] text-ink-55">
              日快照（近 {Math.min(14, d.history.length)} 天）：
              {d.history.slice(-14).map(h => (
                <span key={h.date} className="inline-flex items-center gap-1 mr-2"
                  title={h.date + ' n=' + h.n + ' 胜率' + performancePct(h.win_rate) + ' 变负概率' + performancePct(h.p_edge_negative, 0)}>
                  <LightDot light={h.light} /><span className="font-mono">{h.date.slice(5)}</span>
                </span>
              ))}
            </div>
          )}

          <div className="border-t border-line-soft pt-3">
            <div className="flex items-center gap-2">
              <button onClick={runDiagnose} disabled={diagBusy}
                className="px-2.5 py-1 text-xs rounded-pill border border-line text-ink-80 hover:border-brand transition disabled:opacity-40">
                {diagBusy ? '诊断中…' : '🤖 AI 诊断'}
              </button>
              <span className="text-[11px] text-ink-55">预留接口：模型尚未接入，点击可查看将提交给模型的结构化输入</span>
            </div>
            {diag && (
              <div className="mt-2 text-xs bg-sunken rounded-sm p-3 space-y-1.5">
                <div className="text-ink-80">{String(diag.message ?? diag.summary ?? diag.status)}</div>
                {diag.payload != null && (
                  <details>
                    <summary className="cursor-pointer text-ink-55">查看诊断输入（JSON）</summary>
                    <pre className="mt-1 max-h-60 overflow-auto text-[10px] font-mono text-ink-80 whitespace-pre-wrap break-all">{JSON.stringify(diag.payload, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function MonitorTab() {
  const [view, setView] = useState<'signals' | 'agent'>('signals')
  const [report, setReport] = useState<SignalHealthReport | null>(null)
  const [err, setErr] = useState('')
  const [loading, setLoading] = useState(false)
  const [scope, setScope] = useState<'all' | 'live' | 'shadow'>('all')
  const [onlyActive, setOnlyActive] = useState(true)
  const [lightFilter, setLightFilter] = useState<'all' | HealthLight>('all')
  const [sel, setSel] = useState<{ scope: string; key: string } | null>(null)
  const [showCorrelations, setShowCorrelations] = useState(false)

  const load = useCallback((force = false) => {
    setLoading(true)
    api.getSignalsHealth(force)
      .then((d: SignalHealthReport) => {
        if (d && Array.isArray(d.rows)) { setReport(d); setErr('') } else setErr('信号体检返回异常')
      })
      .catch(() => setErr('信号体检获取失败'))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (view !== 'signals') return
    load()
    const t = setInterval(() => load(), 60_000)
    return () => clearInterval(t)
  }, [view, load])

  const rows = useMemo(() => {
    if (!report) return []
    return report.rows
      .filter(r => (scope === 'all' || r.scope === scope) && (!onlyActive || r.active) && (lightFilter === 'all' || r.light === lightFilter))
      .sort((a, b) => HEALTH_LIGHT[a.light].order - HEALTH_LIGHT[b.light].order || b.metrics.n - a.metrics.n)
  }, [report, scope, onlyActive, lightFilter])

  const chip = (on: boolean) => 'px-2 py-0.5 rounded-pill text-[11px] font-semibold transition ' + (on ? 'bg-brand text-white' : 'text-ink-55 hover:text-ink-80')
  const tally = report?.tally_active
  const redPct = Math.round((report?.thresholds.red_p_neg ?? 0.9) * 100)
  const minN = report?.thresholds.min_n ?? 30
  const corrs = report?.correlations || []

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-0.5 bg-sunken rounded-pill p-0.5 border border-line-soft w-fit">
        <button className={chip(view === 'signals')} onClick={() => setView('signals')}>信号体检</button>
        <button className={chip(view === 'agent')} onClick={() => setView('agent')}>Agent 运行健康</button>
      </div>

      {view === 'agent' && <AgentHealthView />}

      {view === 'signals' && (
        <>
          <div className="bg-card rounded-card border border-line px-5 py-3 space-y-2">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-4 flex-wrap text-sm">
                <span className="font-bold text-ink-95">在线通道体检</span>
                {tally && (['RED', 'YELLOW', 'GREEN', 'GRAY'] as HealthLight[]).map(l => (
                  <button key={l} onClick={() => setLightFilter(f => (f === l ? 'all' : l))}
                    className={'flex items-center gap-1.5 text-xs ' + (lightFilter === l ? 'font-bold underline' : '')}>
                    <LightDot light={l} />{HEALTH_LIGHT[l].label} <span className="font-mono">{tally[l]}</span>
                  </button>
                ))}
                <HelpHint text={'红灯=优势已变负的后验概率≥' + redPct + '%且样本≥' + minN + '；黄灯=优势疑似变负/前向显著低于冻结基准/收益靠右尾（剔除最赚5单后≤0）；灰=样本不足不下结论。统计仅计「在线」通道（影子开启或实盘开启）。系统只给建议，不会自动关闭任何通道。'} />
              </div>
              <div className="flex items-center gap-2 text-[11px] text-ink-55">
                {report && <span className="font-mono">{new Date(report.generated_at).toLocaleString('zh-CN')}</span>}
                <button onClick={() => load(true)} disabled={loading}
                  className="px-2 py-0.5 rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover transition disabled:opacity-40">
                  {loading ? '计算中…' : '立即重算'}
                </button>
              </div>
            </div>
            <div className="flex items-center gap-3 flex-wrap text-[11px] text-ink-55">
              <div className="flex items-center gap-0.5 bg-sunken rounded-pill p-0.5 border border-line-soft">
                {([['all', '全部'], ['live', '实盘'], ['shadow', '影子']] as const).map(([k, l]) => (
                  <button key={k} className={chip(scope === k)} onClick={() => setScope(k)}>{l}</button>
                ))}
              </div>
              <label className="flex items-center gap-1 cursor-pointer">
                <input type="checkbox" checked={onlyActive} onChange={e => setOnlyActive(e.target.checked)} />仅在线通道
              </label>
              {corrs.length > 0 && (
                <button
                  onClick={() => setShowCorrelations(s => !s)}
                  className={`px-2 py-0.5 rounded-pill border transition ${showCorrelations ? 'bg-brand text-white border-brand' : 'border-line text-ink-80 hover:border-brand'}`}
                >
                  {showCorrelations ? '收起组合共振' : `查看通道共振 (${corrs.length}对)`}
                </button>
              )}
              <span>共 {rows.length} 行 · 点行下钻</span>
            </div>
          </div>

          {/* 组合相关性与同窗共振卡片 (P3) */}
          {showCorrelations && corrs.length > 0 && (
            <div className="bg-card rounded-card border border-line p-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-bold text-ink-95">
                  通道间同窗共振分析 (Top 20 重叠通道对)
                  <HelpHint text="展示在同一 5m/15m 窗口同时触发的不同通道。Jaccard 越高说明两个通道在同一个赌注上同时押注，组合风险集中；同向率表示两通道同时触发时方向相同比例。" />
                </div>
                <button onClick={() => setShowCorrelations(false)} className="text-xs text-ink-55 hover:text-ink-95">关闭</button>
              </div>
              <div className="overflow-x-auto max-h-56 overflow-y-auto">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="text-ink-55 border-b border-line-soft">
                      <th className="py-1 text-left">通道 A</th>
                      <th className="py-1 text-left">通道 B</th>
                      <th className="py-1 text-right">同窗触发数</th>
                      <th className="py-1 text-right">Jaccard 重叠度</th>
                      <th className="py-1 text-right">同向一致率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {corrs.map((c, idx) => (
                      <tr key={idx} className="border-b border-line-soft hover:bg-card-hover">
                        <td className="py-1 font-medium">{SIGNAL_INFO[c.channel_a.split(':')[1]]?.name ?? c.channel_a}</td>
                        <td className="py-1 font-medium">{SIGNAL_INFO[c.channel_b.split(':')[1]]?.name ?? c.channel_b}</td>
                        <td className="py-1 text-right font-mono">{c.shared_windows}</td>
                        <td className="py-1 text-right font-mono font-bold text-brand">{(c.jaccard * 100).toFixed(1)}%</td>
                        <td className="py-1 text-right font-mono">{performancePct(c.agreement_rate)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!report && <div className="text-center text-ink-55 py-12 text-sm">{err || '加载信号体检中...'}</div>}

          {report && (
            <div className="bg-card rounded-card border border-line overflow-x-auto">
              <table className="w-full min-w-[980px] text-xs">
                <thead><tr className="text-ink-55 border-b border-line">
                  <th className="py-2 px-2 text-left">通道</th><th className="py-2 px-2 text-left">来源</th>
                  <th className="py-2 px-2 text-right">n</th><th className="py-2 px-2 text-right">胜率</th>
                  <th className="py-2 px-2 text-right">保本</th><th className="py-2 px-2 text-right">优势</th>
                  <th className="py-2 px-2 text-right">均EV</th><th className="py-2 px-2 text-right">剔Top5</th>
                  <th className="py-2 px-2 text-right">变负概率</th>
                  <th className="py-2 px-2 text-center">状态</th>
                  <th className="py-2 px-2 text-left">提示</th>
                </tr></thead>
                <tbody>
                  {rows.map(r => {
                    const m = r.metrics
                    const on = sel?.scope === r.scope && sel.key === r.key
                    const cp = r.change_point
                    return (
                      <tr key={r.scope + ':' + r.key} onClick={() => setSel(on ? null : { scope: r.scope, key: r.key })}
                        className={'border-b border-line-soft cursor-pointer ' + (on ? 'bg-brand-soft' : 'hover:bg-card-hover')}>
                        <td className="py-1.5 px-2 whitespace-nowrap">
                          <span className="inline-flex items-center gap-1.5">
                            <LightDot light={r.light} />
                            <span className="font-medium">{SIGNAL_INFO[r.key]?.name ?? r.key}</span>
                            {!r.active && <span className="rounded-pill bg-sunken px-1.5 py-0.5 text-[9px] text-ink-55">{r.retired ? '已退役' : '已下线'}</span>}
                          </span>
                        </td>
                        <td className="py-1.5 px-2 text-ink-55 whitespace-nowrap">{r.scope === 'live' ? '实盘' : '影子'}{r.market_period ? ' · ' + r.market_period : ''}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{m.n}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{performancePct(m.win_rate)}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{performancePct(m.mean_breakeven)}</td>
                        <td className={'py-1.5 px-2 text-right font-mono ' + signClass(m.edge)}>{performanceSigned(m.edge == null ? null : m.edge * 100, '%', 1)}</td>
                        <td className={'py-1.5 px-2 text-right font-mono ' + signClass(m.avg_ev)}>{performanceSigned(m.avg_ev, '', 3)}</td>
                        <td className={'py-1.5 px-2 text-right font-mono ' + signClass(m.ev_ex_top5)}>{performanceSigned(m.ev_ex_top5, '', 3)}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{performancePct(m.p_edge_negative, 0)}</td>
                        <td className="py-1.5 px-2 text-center whitespace-nowrap">
                          {r.death_status && r.death_status !== 'ALIVE' ? (
                            <DeathBadge status={r.death_status} reason={r.death_reason} />
                          ) : cp?.is_declining ? (
                            <span className="rounded-pill px-1.5 py-0.5 text-[9px] font-bold bg-warning-soft text-warning" title={`拐折回撤 -${cp.drawdown_ev} EV`}>拐折</span>
                          ) : (
                            <span className="text-[10px] text-ink-55 font-mono">正常</span>
                          )}
                        </td>
                        <td className="py-1.5 px-2 text-ink-80 max-w-[280px] truncate" title={r.reasons.join('；')}>{r.reasons[0] ?? ''}</td>
                      </tr>
                    )
                  })}
                  {rows.length === 0 && <tr><td colSpan={11} className="py-8 text-center text-ink-55">当前筛选下无通道</td></tr>}
                </tbody>
              </table>
            </div>
          )}

          {sel && <SignalHealthDrill scope={sel.scope} keyName={sel.key} onClose={() => setSel(null)} />}
        </>
      )}
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
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-overlay p-4" onClick={onClose}>
      <div
        className="bg-card rounded-panel border border-line w-full max-w-3xl max-h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-line flex items-center justify-between shrink-0">
          <div>
            <h2 className="text-sm font-bold text-ink-95">🔬 深度模式学习</h2>
            <p className="text-[11px] text-ink-55 mt-0.5">全量历史窗口深度分析，预览发现结果，勾选后写入模式库</p>
          </div>
          <button onClick={onClose} className="text-ink-55 hover:text-ink-80 text-xl leading-none px-1">✕</button>
        </div>

        {/* 主体 */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-4">
          {/* 分析参数 */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-ink-80">分析窗口数上限</label>
            <input
              type="number"
              min={1}
              value={maxWindows}
              onChange={e => setMaxWindows(Math.max(1, Number(e.target.value) || 1))}
              disabled={phase === 'analyzing' || phase === 'committing'}
              className="ds-input ds-input-numeric w-24"
            />
            <button
              onClick={runAnalyze}
              disabled={phase === 'analyzing' || phase === 'committing'}
              className="px-3 py-1.5 text-xs font-semibold text-white bg-violet rounded-full hover:bg-violet-hover disabled:opacity-50 transition"
            >
              {phase === 'analyzing' ? '🔄 LLM 分析中...' : '开始分析'}
            </button>
          </div>

          {/* 实时日志（阶段性进度） */}
          {liveLog.length > 0 && (
            <div>
              <div className="text-[11px] font-bold text-ink-55 mb-1">📡 实时日志</div>
              <div className="text-[11px] font-mono text-ink-80 bg-sunken p-2 rounded-pill border border-line space-y-0.5">
                {liveLog.map((l, i) => (
                  <div key={i} className="flex gap-1.5">
                    <span className="text-ink-40 shrink-0">›</span>
                    <span className="break-words">{l}</span>
                  </div>
                ))}
                {phase === 'analyzing' && (
                  <div className="flex gap-1.5 text-violet">
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
              <div className="text-[11px] font-bold text-ink-55 mb-1">
                🧠 分析推理{phase === 'analyzing' && <span className="ml-1 text-violet font-normal">（实时）</span>}
              </div>
              <pre
                ref={reasoningRef}
                className="text-[11px] text-ink-80 bg-sunken p-2 rounded-sm border border-line overflow-auto max-h-52 whitespace-pre-wrap break-words"
              >
                {reasoning}
                {phase === 'analyzing' && <span className="inline-block w-1.5 h-3 ml-0.5 align-middle bg-violet animate-pulse" />}
              </pre>
            </div>
          )}

          {/* 发现列表 */}
          {discoveries.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="text-[11px] font-bold text-ink-55">
                  发现 {discoveries.length} 条 · 已选 {selectedCount} 条
                  {(trainCount > 0 || holdoutCount > 0) && (
                    <span className="ml-2 font-normal text-ink-55">train {trainCount} / holdout {holdoutCount}</span>
                  )}
                </div>
                <div className="flex gap-2">
                  <button onClick={() => setChecked(new Set(discoveries.map((_, i) => i)))} className="text-[10px] px-2 py-0.5 rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover">全选</button>
                  <button onClick={() => setChecked(new Set())} className="text-[10px] px-2 py-0.5 rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover">清空</button>
                </div>
              </div>
              <div className="space-y-2">
                {discoveries.map((d, i) => (
                  <div key={i} className="border border-line rounded-sm">
                    <label className="flex gap-2 items-start p-2.5 cursor-pointer hover:bg-sunken">
                      <input type="checkbox" checked={checked.has(i)} onChange={() => toggle(i)} className="mt-1 accent-violet" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <ChangeTypeBadge type={d.operation} />
                          <span className="font-semibold text-sm text-ink-95">{d.pattern_name}</span>
                          <DirectionBadge direction={d.predicted_direction} />
                          <span className="text-[10px] text-ink-55 font-mono">conf {(d.confidence_score * 100).toFixed(0)}%</span>
                          {d.holdout_win_rate != null && (
                            <span className="text-[10px] text-teal font-mono" title={`holdout 胜率 · 样本 ${d.holdout_sample_count ?? 0} · Wilson下界 ${d.holdout_ci_lower != null ? (d.holdout_ci_lower * 100).toFixed(0) + '%' : '—'}`}>
                              holdout {(d.holdout_win_rate * 100).toFixed(0)}%
                            </span>
                          )}
                          {d.operation === 'UPDATE' && d.target_pattern_id != null && (
                            <span className="text-[10px] text-warning font-mono">→ 更新 #{d.target_pattern_id}</span>
                          )}
                        </div>
                        <div className="text-xs text-ink-80 mb-1">{d.description}</div>
                        <div className="text-[11px] text-ink-55"><b>理由：</b>{d.change_reason}</div>
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
                              <div className="text-[10px] font-bold text-ink-55 mb-0.5">曲线特征</div>
                              <pre className="text-[10px] bg-sunken p-1.5 rounded-sm border border-line overflow-auto max-h-32">{JSON.stringify(d.curve_features, null, 2)}</pre>
                            </div>
                            <div>
                              <div className="text-[10px] font-bold text-ink-55 mb-0.5">适用条件</div>
                              <pre className="text-[10px] bg-sunken p-1.5 rounded-sm border border-line overflow-auto max-h-32">{JSON.stringify(d.conditions, null, 2)}</pre>
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
        <div className="px-5 py-3 border-t border-line flex items-center justify-between gap-3 shrink-0">
          <span className={`text-xs ${msg.startsWith('✅') ? 'text-positive' : 'text-ink-55'}`}>{msg}</span>
          <div className="flex gap-2 shrink-0">
            <button onClick={onClose} className="px-3 py-1.5 text-xs font-medium text-ink-80 bg-sunken rounded-sm hover:bg-sunken-hover transition">关闭</button>
            {discoveries.length > 0 && (
              <button
                onClick={runCommit}
                disabled={phase === 'committing' || selectedCount === 0}
                className="px-3 py-1.5 text-xs font-semibold text-white bg-positive rounded-sm hover:bg-positive-hover disabled:opacity-50 transition"
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-overlay p-4" onClick={onClose}>
      <div className="bg-card rounded-panel border border-line w-full max-w-4xl max-h-[90vh] flex flex-col" onClick={e => e.stopPropagation()}>
        {/* 头部 */}
        <div className="px-5 py-3 border-b border-line flex items-center justify-between shrink-0">
          <h2 className="text-base font-bold text-ink-95">⚖️ 方案对比 · LLM vs Python 聚类</h2>
          <button onClick={onClose} className="text-ink-55 hover:text-ink-80 text-xl leading-none">×</button>
        </div>

        {/* 控制栏 */}
        <div className="px-5 py-3 border-b border-line-soft flex items-center gap-3 flex-wrap shrink-0">
          <label className="text-xs text-ink-80 flex items-center gap-1.5">
            窗口数
            <input
              type="number" min={1} max={500} value={maxWindows}
              onChange={e => setMaxWindows(Math.max(1, Math.min(500, Number(e.target.value) || 1)))}
              className="ds-input ds-input-numeric w-20"
            />
          </label>
          <button onClick={runPy} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-semibold text-white bg-teal rounded-sm hover:bg-teal-hover disabled:opacity-50">
            {busy === 'py' ? '聚类中...' : '运行 Python 聚类版'}
          </button>
          <button onClick={runCmp} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-semibold text-white bg-indigo rounded-sm hover:bg-indigo-hover disabled:opacity-50">
            {busy === 'cmp' ? '对比中（含 LLM）...' : '对比两套方案'}
          </button>
          <button onClick={loadLive} disabled={busy !== ''} className="px-3 py-1.5 text-xs font-medium text-ink-80 bg-sunken rounded-sm hover:bg-sunken-hover disabled:opacity-50">
            刷新上线指标
          </button>
          {msg && <span className={`text-xs ${msg.startsWith('✅') ? 'text-positive' : 'text-ink-55'}`}>{msg}</span>}
        </div>

        <div className="p-5 overflow-auto space-y-5">
          {/* 对比图表 + 表格 */}
          {cmp && (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-ink-80">发现即时对比（同一 train/holdout）</span>
                {!cmp.snapshot_consistent && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-pill bg-warning-soft text-warning" title="两套方案的数据快照不一致">⚠ 快照不一致</span>
                )}
              </div>
              <div className="h-56 bg-sunken rounded-sm border border-line p-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="var(--line-soft)" />
                    <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="LLM" fill="var(--chart-4)" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="PY" fill="var(--teal)" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-line text-ink-55">
                      <th className="py-1.5 px-2 text-left">维度</th>
                      <th className="py-1.5 px-2 text-right"><DiscoveryMethodBadge method="LLM_DEEP" /></th>
                      <th className="py-1.5 px-2 text-right"><DiscoveryMethodBadge method="PY_CLUSTER" /></th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(r => (
                      <tr key={r.label} className="border-b border-line-soft">
                        <td className="py-1.5 px-2 text-ink-80">{r.label}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-violet">{r.get(llmSum)}</td>
                        <td className="py-1.5 px-2 text-right font-mono text-teal">{r.get(pySum)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 上线真实指标（按 discovery_method 聚合） */}
          <div className="space-y-2">
            <span className="text-sm font-semibold text-ink-80">上线真实指标（模式库 ACTIVE，按来源聚合）</span>
            {live.length === 0 ? (
              <div className="text-xs text-ink-55">暂无上线数据</div>
            ) : (
              <div className="overflow-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-line text-ink-55">
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
                      <tr key={g.method} className="border-b border-line-soft">
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
                <span className="text-sm font-semibold text-ink-80">
                  Python 聚类发现 {py.discoveries.length} 条 · 已选 {pySelected}
                  <span className="ml-2 font-normal text-ink-55">train {py.train_count} / holdout {py.holdout_count}</span>
                </span>
                <div className="flex gap-2">
                  <button onClick={() => setPyChecked(new Set(py.discoveries.map((_, i) => i)))} className="text-[10px] px-2 py-0.5 rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover">全选</button>
                  <button onClick={() => setPyChecked(new Set())} className="text-[10px] px-2 py-0.5 rounded-pill bg-sunken text-ink-80 hover:bg-sunken-hover">清空</button>
                </div>
              </div>
              <div className="space-y-2">
                {py.discoveries.map((d, i) => (
                  <label key={i} className="flex gap-2 items-start p-2.5 border border-line rounded-sm cursor-pointer hover:bg-sunken">
                    <input type="checkbox" checked={pyChecked.has(i)} onChange={() => togglePy(i)} className="mt-1 accent-teal" />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1 flex-wrap">
                        <ChangeTypeBadge type={d.operation} />
                        <span className="font-semibold text-sm text-ink-95">{d.pattern_name}</span>
                        <DirectionBadge direction={d.predicted_direction} />
                        <span className="text-[10px] text-ink-55 font-mono">conf {(d.confidence_score * 100).toFixed(0)}%</span>
                        {d.holdout_win_rate != null && (
                          <span className="text-[10px] text-teal font-mono" title={`holdout 样本 ${d.holdout_sample_count ?? 0} · Wilson下界 ${pct(d.holdout_ci_lower)}`}>
                            holdout {pct(d.holdout_win_rate)}
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-ink-80 mb-1">{d.description}</div>
                      <div className="text-[11px] text-ink-55"><b>理由：</b>{d.change_reason}</div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 底部 */}
        <div className="px-5 py-3 border-t border-line flex items-center justify-end gap-2 shrink-0">
          <button onClick={onClose} className="px-3 py-1.5 text-xs font-medium text-ink-80 bg-sunken rounded-sm hover:bg-sunken-hover">关闭</button>
          {py && py.discoveries.length > 0 && (
            <button onClick={commitPy} disabled={busy !== '' || pySelected === 0} className="px-3 py-1.5 text-xs font-semibold text-white bg-positive rounded-sm hover:bg-positive-hover disabled:opacity-50">
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
      label: 'S1 多头耗尽', cls: 'bg-negative-soft text-negative border-negative',
      desc: '破4h高·光头阳·4h上沿 → 次周期 DOWN',
    },
    bear_exhaust: {
      label: 'S2 空头耗尽', cls: 'bg-positive-soft text-positive border-positive',
      desc: '破4h低·收阴·放量≥2.0 → 次周期 UP',
    },
    momentum_fade: {
      label: 'S4 动量衰竭', cls: 'bg-brand-soft text-brand border-brand',
      desc: '连阳≥3·光头阳，无破位要求 → 次周期 DOWN',
    },
    bull_exhaust_confirm: {
      label: 'S5 确认入场', cls: 'bg-violet-soft text-violet border-violet',
      desc: 'S1 信号后 +5min 回落确认（5m 收盘 < 周期开盘）→ 买 DOWN 持有到期；盈亏平衡入场价 0.77',
    },
  }
  const sceneOf = (s: FakeBreakoutSignal) => sceneMeta[s.pattern_type || ''] || null

  return (
    <Card title="假突破信号（1h/4h/日线 × 阻力/支撑 · 暂不下注）">
      <div className="text-xs text-ink-55 mb-3">
        秒级检测 BTC 破位：冲过阻力→看跌（买 DOWN）/ 跌破支撑→看涨（买 UP）。
        回测：日线破阻力 80%、4h 73.5%、1h 65.1%（支撑方向对称成立）。当前只记录信号 + 邮件提醒。
      </div>

      {/* 状态行 */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-3 text-xs">
        <span className="flex items-center gap-1.5">
          <StatusDot ok={!!status?.running} />
          <span className="text-ink-55">检测器</span>
          <span className={`font-bold ${status?.running ? 'text-positive' : 'text-negative'}`}>
            {status?.running ? '运行中' : '未运行'}
          </span>
        </span>
        <span className="text-ink-55">
          当前价 <strong className="text-ink-95 font-mono">{status?.btc_mid ? status.btc_mid.toFixed(0) : '--'}</strong>
        </span>
        <span className="text-ink-55">今日信号 <strong className="font-mono text-ink-95">{status?.daily_count ?? 0}</strong> 条</span>
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
              <div key={l} className="px-2.5 py-2 bg-sunken border border-line rounded-sm">
                <div className="font-bold text-ink-80 mb-1">{l === 'daily' ? '日线' : l} <span className="text-ink-55 font-normal">（{l === 'daily' ? 288 : l === '4h' ? 48 : 12} 窗）</span></div>
                <div className={`font-mono ${brokeRes ? 'text-negative font-bold' : 'text-ink-55'}`}>
                  阻力 {lv.resistance.toFixed(0)}{distRes !== null && <span className="ml-1">({distRes >= 0 ? '+' : ''}{distRes.toFixed(2)}%){brokeRes && ' ⚠️'}</span>}
                </div>
                <div className={`font-mono ${brokeSup ? 'text-positive font-bold' : 'text-ink-55'}`}>
                  支撑 {lv.support.toFixed(0)}{distSup !== null && <span className="ml-1">({distSup >= 0 ? '+' : ''}{distSup.toFixed(2)}%){brokeSup && ' ⚠️'}</span>}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* 分组统计（级别×方向） */}
      {stats && stats.total_signals > 0 && (
        <div className="mb-3 px-3 py-2 bg-warning-soft border border-warning rounded-sm text-xs">
          <div className="flex flex-wrap gap-x-5 gap-y-1 mb-1.5">
            <span className="text-ink-80">累计信号 <strong>{stats.total_signals}</strong></span>
            <span className="text-ink-80">已结算（15m） <strong>{stats.settled}</strong></span>
            <span className="text-ink-80">已结算（5m） <strong>{stats.settled_5m}</strong></span>
          </div>
          {stats.by_group.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {stats.by_group.map(g => (
                <span key={`${g.level}-${g.side}`} className="px-2 py-1 bg-card border border-warning rounded-sm font-mono text-[11px]">
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
              <div key={ps.pattern_type} className="px-2.5 py-2 bg-card border border-line rounded-sm">
                <div className="flex items-center gap-1 mb-1.5">
                  <span className={`px-1.5 py-0.5 rounded-pill text-[10px] font-bold border ${meta?.cls || 'bg-sunken text-ink-55 border-line'}`}>
                    {meta?.label || ps.pattern_type}
                  </span>
                  {research !== undefined && (
                    <span className="text-[10px] text-ink-55" title="回测胜率点估计（入场 EV 用的 p）">回测 {(research * 100).toFixed(1)}%</span>
                  )}
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-ink-80">
                  <span title={`已结算正式信号 ${ps.n} 条（胜 ${ps.wins}）`}>
                    实盘 <strong className={ps.winrate !== null && research !== undefined && ps.winrate >= research ? 'text-positive' : 'text-ink-95'}>
                      {ps.winrate !== null ? `${(ps.winrate * 100).toFixed(0)}%` : '--'}
                    </strong>({ps.n})
                  </span>
                  <span title="累计实现 EV/事件（1 USDT 本金：赢 0.98/entry−1，输 −1）">
                    EV <strong className={(ps.cumulative_ev ?? 0) >= 0 ? 'text-positive' : 'text-negative'}>
                      {ps.cumulative_ev !== null ? `${ps.cumulative_ev >= 0 ? '+' : ''}${ps.cumulative_ev.toFixed(3)}` : '--'}
                    </strong>
                  </span>
                  <span title="入场时刻预期 EV 均值 = p×(1−费)/entry−1（p=回测胜率）">
                    入场EV {ps.avg_ev_at_entry !== null ? `${ps.avg_ev_at_entry >= 0 ? '+' : ''}${ps.avg_ev_at_entry.toFixed(3)}` : '--'}
                  </span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-ink-55 mt-0.5">
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
        <div className="text-center text-ink-55 py-6 text-sm">暂无信号——BTC 盘中破位任一级别阻力/支撑时自动记录并邮件提醒</div>
      ) : (
        <div className="overflow-x-auto max-h-80 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-card">
              <tr className="border-b border-line text-ink-55">
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
                  <tr className="border-b border-line-soft hover:bg-sunken">
                    <td className="py-1.5 px-1 text-center">
                      <button onClick={() => togglePath(s.id)}
                        className="text-ink-55 hover:text-indigo leading-none text-xs"
                        title="展开次周期价格+报价路径（15s 采样，8/13 起有数据）">
                        {expandedIds.has(s.id) ? '▾' : '▸'}
                      </button>
                    </td>
                    <td className="py-1.5 px-2 text-ink-80 font-mono">{fmtMs(s.signal_time)}</td>
                    <td className="py-1.5 px-2 text-center" title={sceneTip}>
                      {scene ? (
                        <span className={`px-1.5 py-0.5 rounded-sm text-[10px] font-bold border ${scene.cls}`}>
                          {scene.label}
                        </span>
                      ) : (
                        <span className="px-1.5 py-0.5 rounded-pill text-[10px] font-bold bg-sunken text-ink-55">旧信号</span>
                      )}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className="px-1.5 py-0.5 rounded-pill text-[10px] font-bold bg-indigo-soft text-indigo border border-indigo mr-1">{s.level}</span>
                      <span className={`px-1.5 py-0.5 rounded-pill text-[10px] font-bold ${
                        s.side === 'high' ? 'bg-negative-soft text-negative' : 'bg-positive-soft text-positive'
                      }`}>{s.side === 'high' ? '看跌' : '看涨'}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-ink-95">
                      {s.btc_price.toFixed(0)} <span className="text-ink-55">/ {s.resistance.toFixed(0)}</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-negative" title={entrySnap !== null ? '开盘后 ~8s 入场快照' : '信号瞬间报价（无入场快照）'}>
                      {entry15?.toFixed(3) ?? '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-ink-55">{q5?.toFixed(3) ?? '--'}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-ink-80">
                      {s.cycle_open_price_15m ? `${s.cycle_open_price_15m.toFixed(0)}→` : ''}{s.settle_btc_price?.toFixed(0) ?? '--'}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {s.settle_outcome_5m ? (
                        <span className={`px-1.5 py-0.5 rounded-pill text-[10px] font-bold ${
                          isWin(s, s.settle_outcome_5m) ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'
                        }`}>{isWin(s, s.settle_outcome_5m) ? `✓ ${winBadge(s.settle_outcome_5m)}` : `✗ ${winBadge(s.settle_outcome_5m)}`}</span>
                      ) : <span className="text-ink-40" title="未到 +5min 结算时点，或该时点因部署重启错过（超宽限不回填防失真）">待结算</span>}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      {s.settle_outcome ? (
                        <span className={`px-1.5 py-0.5 rounded-pill text-[10px] font-bold ${
                          isWin(s, s.settle_outcome) ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'
                        }`}>{isWin(s, s.settle_outcome) ? `✓ ${winBadge(s.settle_outcome)} 赢` : `✗ ${winBadge(s.settle_outcome)}`}</span>
                      ) : <span className="text-ink-40" title="15m 市场尚未到期">待结算</span>}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <span className={`text-[10px] ${
                        s.status === 'SETTLED' ? 'text-ink-55' : s.status === 'PENDING' ? 'text-warning' : 'text-ink-55'
                      }`}>
                        {s.status === 'SETTLED' ? '已结算' : s.status === 'PENDING' ? '跟踪中' : s.status}
                      </span>
                      {s.email_sent && <span className="text-[10px] text-brand ml-1" title="邮件已推送">📧</span>}
                    </td>
                  </tr>
                  {expandedIds.has(s.id) && (
                    <tr className="border-b border-line-soft">
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

  if (failed) return <div className="py-4 text-center text-xs text-ink-55">路径加载失败</div>
  if (!data) return <div className="py-4 text-center text-xs text-ink-55">路径加载中…</div>
  if (!data.has_data || !data.open) {
    return (
      <div className="py-4 text-center text-xs text-ink-55" title="15m 市场采样自 2026-08-13 开始积累，更早的信号无路径数据">
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
    <div className="px-2 py-2.5 bg-sunken">
      <div className="flex justify-between items-center mb-1 text-[11px] text-ink-55">
        <span>
          次周期路径 · 开盘 <span className="num">{open.toLocaleString()}</span> ·
          <span className="text-brand"> ■ 价格 ±bp</span> ·
          <span className="text-negative"> ■ DOWN 报价</span>
          <span className="ml-2 px-1 bg-warning-soft text-warning border border-warning rounded-pill"
            title="S7 研究结论：早期窗口（t≤4 分钟）是涨态/跌态双分支最优入场区">
            黄带 = t≤4 早期入场研究窗
          </span>
        </span>
        {data.settle && (
          <span className={won ? 'text-positive font-bold' : 'text-negative font-bold'}>
            结算 {data.settle} · {won ? '信号赢' : '信号输'}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={128}>
        <LineChart data={pts} margin={{ top: 6, right: 44, left: 0, bottom: 0 }}>
          <ReferenceArea x1={0} x2={240} fill="var(--sunken)" />
          <ReferenceLine y={0} stroke="var(--line)" />
          <ReferenceLine x={x5} stroke="var(--chart-4)" strokeDasharray="4 3"
            label={{ value: '+5m', position: 'insideTopRight', fill: 'var(--chart-4)', fontSize: 10 }} />
          <XAxis dataKey="off" type="number" domain={[0, 900]} ticks={xTicks} hide />
          <YAxis domain={[-maxDev, maxDev]} width={46}
            ticks={[-maxDev, 0, maxDev]} tickFormatter={v => `${v}bp`}
            tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
          <Tooltip formatter={tipFmt} labelFormatter={o => `+${mmss(o as number)}`} />
          <Line dataKey="dev" stroke="var(--chart-1)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={74}>
        <LineChart data={pts} margin={{ top: 2, right: 44, left: 0, bottom: 0 }}>
          <ReferenceArea x1={0} x2={240} fill="var(--sunken)" />
          <ReferenceLine x={x5} stroke="var(--chart-4)" strokeDasharray="4 3" />
          {data.quote5m_down != null && data.quote5m_off != null && (
            <ReferenceDot x={data.quote5m_off} y={data.quote5m_down} r={3.5} fill="var(--chart-4)" stroke="white" />
          )}
          <XAxis dataKey="off" type="number" domain={[0, 900]} ticks={xTicks}
            tickFormatter={v => `${Math.round(v / 60)}分`} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
          <YAxis domain={[0, 1]} width={46} ticks={[0, 0.5, 1]}
            tickFormatter={v => v.toFixed(1)} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
          <Tooltip formatter={tipFmt} labelFormatter={o => `+${mmss(o as number)}`} />
          <Line dataKey="down" stroke="var(--negative)" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[10px] text-ink-55 font-mono">
        <span>极值 {minDev.toFixed(1)} ~ +{maxUp.toFixed(1)}bp</span>
        <span>末点 {last.dev.toFixed(1)}bp / q(DOWN) {last.down.toFixed(3)}</span>
        {data.quote5m_down != null && (
          <span className="text-violet">+5m 确认价 q={data.quote5m_down.toFixed(3)}</span>
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
    S: 'bg-warning-soft text-warning border-warning',
    A: 'bg-violet-soft text-violet border-violet',
    B: 'bg-brand-soft text-brand border-brand',
    C: 'bg-sunken text-ink-80 border-line',
  }
  return (
    <span className={`inline-block w-6 text-center px-1 py-0.5 rounded-pill border text-[11px] font-black ${meta[tier] || meta.C}`}>
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
        <div className="text-[11px] text-ink-55">
          新数据累积到阈值后自动全量重回测；点击行展开该模式的回测历史（纵向对比）
        </div>
        <div className="flex items-center gap-2">
          {reevalMsg && <span className="text-[11px] text-ink-55">{reevalMsg}</span>}
          <button
            onClick={handleReevaluate}
            disabled={loading}
            className="px-2.5 py-1 text-[11px] font-semibold text-white bg-indigo rounded-sm hover:bg-indigo-hover disabled:opacity-50 transition"
            title="立即对全部谓词模式执行一轮全量历史回测"
          >
            {loading ? '回测中...' : '🔄 立即重回测'}
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="text-center text-ink-55 py-6 text-sm">暂无极式（深度学习发现后自动入库并参与重回测）</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line text-ink-55">
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
                    className="border-b border-line-soft hover:bg-sunken cursor-pointer"
                    onClick={() => toggleRuns(p.pattern_id)}
                  >
                    <td className="py-1.5 px-2 text-center"><TierBadge tier={p.tier} /></td>
                    <td className="py-1.5 px-2 font-medium text-ink-80">
                      {p.pattern_name}
                      <span className="ml-1 text-[10px] text-ink-55">{p.status}</span>
                    </td>
                    <td className="py-1.5 px-2 text-center"><DirectionBadge direction={p.predicted_direction} /></td>
                    <td className="py-1.5 px-2 text-right font-mono font-bold text-ink-95">
                      {p.latest_run ? pct(p.latest_run.win_rate) : '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-ink-80">
                      {p.latest_run?.wilson_lower !== null && p.latest_run?.wilson_lower !== undefined
                        ? p.latest_run.wilson_lower.toFixed(3) : '--'}
                    </td>
                    <td className={`py-1.5 px-2 text-right font-mono font-bold ${
                      p.latest_run?.ev_after_fee !== null && p.latest_run?.ev_after_fee !== undefined && p.latest_run.ev_after_fee > 0
                        ? 'text-positive' : 'text-negative'
                    }`}>
                      {p.latest_run?.ev_after_fee !== null && p.latest_run?.ev_after_fee !== undefined
                        ? `${p.latest_run.ev_after_fee > 0 ? '+' : ''}${p.latest_run.ev_after_fee.toFixed(3)}` : '--'}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-ink-80">
                      {p.latest_run?.sample_count ?? 0}
                    </td>
                    <td className="py-1.5 px-2 text-ink-55 text-[11px] max-w-56 truncate" title={(p.latest_run?.delta_vs_prev?.suggestions || []).join('\n')}>
                      {p.latest_run?.delta_vs_prev?.suggestions?.[0] ?? '—'}
                    </td>
                  </tr>
                  {expanded === p.pattern_id && (
                    <tr>
                      <td colSpan={8} className="bg-sunken px-4 py-3">
                        <div className="text-[11px] font-bold text-ink-80 mb-2">回测历史（纵向对比：同一模式随数据累积的表现漂移）</div>
                        {runs.length === 0 ? (
                          <div className="text-ink-55 text-xs py-2">该模式尚无回测记录</div>
                        ) : (
                          <div className="space-y-2">
                            {runs.map(r => (
                              <div key={r.id} className="bg-card rounded-sm border border-line px-3 py-2">
                                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-80">
                                  <span className="font-mono">{r.created_at ? new Date(r.created_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '--'}</span>
                                  <span>胜率 <strong className="font-mono">{pct(r.win_rate)}</strong>（{r.correct_count}/{r.sample_count}）</span>
                                  <span>CI [{r.wilson_lower?.toFixed(3) ?? '--'}, {r.wilson_upper?.toFixed(3) ?? '--'}]</span>
                                  <span>EV <strong className="font-mono">{r.ev_after_fee !== null ? `${(r.ev_after_fee ?? 0) > 0 ? '+' : ''}${r.ev_after_fee?.toFixed(3)}` : '--'}</strong></span>
                                  <span className="text-ink-55">{r.trigger_reason}</span>
                                  {r.delta_vs_prev?.tier_change && (
                                    <span className="text-indigo font-bold">
                                      {r.delta_vs_prev.tier_change.from} → {r.delta_vs_prev.tier_change.to}
                                    </span>
                                  )}
                                  {r.delta_vs_prev?.decay_warning && <span className="text-negative font-bold">⚠ 衰减降级</span>}
                                </div>
                                {(r.delta_vs_prev?.suggestions || []).length > 0 && (
                                  <ul className="mt-1 text-[11px] text-ink-55 space-y-0.5">
                                    {(r.delta_vs_prev?.suggestions || []).map((s, i) => <li key={i}>· {s}</li>)}
                                  </ul>
                                )}
                                {r.segment_stats && Object.keys(r.segment_stats).length > 0 && (
                                  <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-55 font-mono">
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
      <div className="text-xs text-ink-55 mb-3">
        币安预测市场 15m 期的 UP/DOWN 报价。冲高破位瞬间 DOWN token 被砸出的低价，就是假突破策略的下注赔率。
      </div>
      <div className="flex flex-wrap gap-4 mb-3 text-xs">
        <span className="px-2 py-1 bg-warning-soft text-warning rounded-sm font-medium">🟠 15m 期</span>
        {endMs > 0 && (
          <span className="text-warning font-mono">⏱ 距到期 {min}:{String(sec).padStart(2, '0')}</span>
        )}
        {last && (
          <>
            <span className="text-ink-55">
              DOWN 现价 <strong className="text-negative font-mono">{last.down_price !== null ? last.down_price.toFixed(3) : '--'}</strong>
              <span className="text-ink-55">（越低赔率越肥）</span>
            </span>
            <span className="text-ink-55">
              UP 现价 <strong className="text-positive font-mono">{last.up_price !== null ? last.up_price.toFixed(3) : '--'}</strong>
            </span>
            {last.btc_price && (
              <span className="text-ink-55">BTC <strong className="font-mono text-ink-95">{last.btc_price.toFixed(0)}</strong></span>
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
              <CartesianGrid stroke="var(--line-soft)" />
              <XAxis dataKey="time" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 11 }} stroke="var(--ink-55)" domain={[0, 100]} tickFormatter={(v: number) => v + '%'} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n === 'up_pct' ? '看涨 (UP)' : '看跌 (DOWN)']}
              />
              <Area type="monotone" dataKey="up_pct" stroke="var(--positive)" fill="var(--positive-soft)" strokeWidth={2} name="看涨" connectNulls />
              <Area type="monotone" dataKey="down_pct" stroke="var(--negative)" fill="#ef444420" strokeWidth={2} name="看跌" connectNulls />
              <ReferenceLine y={50} stroke="var(--ink-55)" strokeDasharray="4 4" />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2 text-xs text-ink-55">
            <span>共 {points.length} 个采样点（约 {Math.round(points.length / 4)} 分钟）</span>
          </div>
        </>
      ) : (
        <div className="text-center text-ink-55 py-10 text-sm">正在采集 15m 市场数据...每 15 秒采样一次。</div>
      )}
    </Card>
  )
}

// ============================================================
// 信号分析 Tab：胜率曲线 × BTC K线 × 周期归因（60s 轮询自动刷新）
// 口径全部来自后端 /api/signals/analytics（单一事实源），前端只做渲染
// ============================================================

/* recharts Tooltip 统一形态 —— 反白（ink-black 底 + on-ink 字）。
   DESIGN.md「Elevation & Depth」：浮层靠反白脱离页面层级，不靠阴影。
   抽成常量以免 8+ 处 contentStyle 各自漂移。 */
const TOOLTIP_STYLE = {
  background: 'var(--ink-black)',
  border: 'none',
  borderRadius: 8,
  fontSize: 12,
  color: 'var(--on-ink)',
} as const
const TOOLTIP_LABEL_STYLE = { color: 'var(--on-ink)' } as const
const TOOLTIP_ITEM_STYLE = { color: 'var(--on-ink)' } as const

/* 22 条影子曲线 → chart-1..10 循环取色。
   第 11 条起与第 1..10 条同色，靠渲染层的「淡色 + 细线」区分（见 shadowEntries.map）。
   label 是 recharts 的 dataKey 本身，改 label 会同时改曲线与图例，故一字未动。 */
const SHADOW_META: Record<string, { label: string; color: string }> = {
  // 长影短实体反转：主事件只落一次，下列逻辑版本由标签还原。
  candle_hm_bull_5m_consensus2_shadow_v1: { label: '5m 长下影·至少2趋势→DOWN', color: 'var(--chart-1)' },
  candle_hm_bull_5m_consensus3_shadow_v1: { label: '5m 长下影·三趋势对照→DOWN', color: 'var(--chart-2)' },
  candle_hm_bull_15m_union_shadow_v1: { label: '15m 长下影·任一趋势→DOWN', color: 'var(--chart-3)' },
  candle_hm_bull_15m_consensus3_shadow_v1: { label: '15m 长下影·三趋势对照→DOWN', color: 'var(--chart-4)' },
  candle_hm_bull_5m_ret5_majority_component_shadow_v1: { label: '5m ret5归因组件', color: 'var(--chart-5)' },
  candle_hm_bull_5m_ret3_component_shadow_v1: { label: '5m ret3归因组件', color: 'var(--chart-6)' },
  candle_hm_bull_15m_ret3_component_shadow_v1: { label: '15m ret3归因组件', color: 'var(--chart-7)' },
  candle_hm_bull_15m_ma10_component_shadow_v1: { label: '15m ma10归因组件', color: 'var(--chart-8)' },
  candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1: { label: '5m 长下影+中高量能', color: 'var(--chart-9)' },
  candle_hm_bull_5m_trend4h_up_shadow_hyp_v1: { label: '5m 长下影+4h上涨', color: 'var(--chart-10)' },
  candle_hm_bull_5m_asia_shadow_hyp_v1: { label: '5m 长下影+亚洲时段', color: 'var(--chart-1)' },
  candle_hm_bull_15m_q45_55_shadow_hyp_v1: { label: '15m 长下影+DOWN报价0.45~0.55', color: 'var(--chart-2)' },
  candle_hm_bull_15m_trend1h_up_shadow_hyp_v1: { label: '15m 长下影+1h上涨', color: 'var(--chart-3)' },
  candle_hm_bull_15m_asia_shadow_hyp_v1: { label: '15m 长下影+亚洲时段', color: 'var(--chart-4)' },
  candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1: { label: '5m 上涨阳线长上影潜力', color: 'var(--chart-5)' },
  candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1: { label: '5m 严格0上影阴线长下影', color: 'var(--chart-6)' },
  candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1: { label: '15m ret5阳线长下影潜力', color: 'var(--chart-7)' },
  candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1: { label: '15m 下跌阳线长上影→UP', color: 'var(--chart-8)' },
  x4_v1: { label: 'X4 情绪错位→DOWN', color: 'var(--chart-1)' },
  quote_momentum_v1: { label: 'A 报价动量→DOWN', color: 'var(--chart-2)' },
  quote_contrarian_v1: { label: 'B 报价反向→DOWN', color: 'var(--chart-3)' },
  krev_a_v1: { label: 'KREV-A 反转→UP', color: 'var(--chart-4)' },
  krev_b_v1: { label: 'KREV-B 反转→UP', color: 'var(--chart-5)' },
  hm_touch_down_v1: { label: 'HM 上吊线反弹→DOWN', color: 'var(--chart-6)' },
  hm_touch_down_v2: { label: 'HM 上吊线反弹v2→DOWN', color: 'var(--chart-7)' },
  rev_p1_v1: { label: 'P1 连跌弱阴反转→UP', color: 'var(--chart-8)' },
  rev_p2_v1: { label: 'P2 连涨弱阳反转→DOWN', color: 'var(--chart-9)' },
  crv_hi_brk20_15m_v1: { label: 'CRV高位突破20高→DOWN', color: 'var(--chart-10)' },
  crv_brk20_15m_v1: { label: 'CRV突破20高→DOWN', color: 'var(--chart-1)' },
  crv_brk50_hi_15m_v1: { label: 'CRV突破50高∧高位→DOWN', color: 'var(--chart-5)' },
  crv_dnex_15m_v1: { label: 'CRV跌破出清三因子→UP', color: 'var(--chart-7)' },
  crv_brk20_5m_v1: { label: 'CRV 5m突破20高→DOWN', color: 'var(--chart-3)' },
  crv_os_st4_5m_v1: { label: 'CRV 5m超卖连阴三因子→UP', color: 'var(--chart-6)' },
  crv_hi_z3_15m_v1: { label: 'CRV高位z离群连3阳→DOWN', color: 'var(--chart-2)' },
  brkrv_brk8h_15m_v1: { label: 'BRK 15m突破8h高→DOWN', color: 'var(--chart-9)' },
  brkrv_brk8h_5m_v1: { label: 'BRK 5m突破8h高→DOWN', color: 'var(--chart-10)' },
  s5_deep_z20_v1: { label: 'S5深档 z5≤-20bp→DOWN', color: 'var(--chart-10)' },
  s1_dyn_sq_v1: { label: 'S1动态轧空路由→DOWN', color: 'var(--chart-1)' },
  quote_momentum_v3: { label: 'A 报价动量v3(非连涨)→DOWN', color: 'var(--chart-1)' },
  nb_zschamp_15m_v1: { label: 'nextbar 15m冠军 深超卖→UP', color: 'var(--chart-2)' },
  nb_smaslope_5m_v1: { label: 'nextbar 5m动量误定价→UP', color: 'var(--chart-3)' },
  combo_p1_v1: { label: 'combo P1 周末连阳过热→DOWN', color: 'var(--chart-4)' },
  combo_p2_v1: { label: 'combo P2 周末大阳→DOWN', color: 'var(--chart-5)' },
  combo_p3_v1: { label: 'combo P3 贴高美盘→DOWN', color: 'var(--chart-6)' },
  combo_p4_v1: { label: 'combo P4 周末超卖→UP', color: 'var(--chart-7)' },
  combo_p5_v1: { label: 'combo P5 光脚超卖→UP', color: 'var(--chart-8)' },
  absorption_follow_td120_v1: { label: '吸收跟随 TD120 欠反应→顺势', color: 'var(--chart-9)' },
  absorption_follow_td150_v1: { label: '吸收跟随 TD150 欠反应→顺势', color: 'var(--chart-10)' },
  s2_cond_t4_v1: { label: 'S2条件 t=4价<开→UP', color: 'var(--chart-1)' },
  s2_cond_t5d_v1: { label: 'S2条件 t=5剔深→UP', color: 'var(--chart-2)' },
  scene_bear_exhaust_opt_v1: { label: 'S2优化 温和放量非低位→UP', color: 'var(--chart-3)' },
  s4_delay60_v1: { label: 'S4延迟入场 首分钟回落→DOWN', color: 'var(--chart-4)' },
  s4_hiconf_v1: { label: 'S4高信心 连阳≥5→DOWN', color: 'var(--chart-5)' },
  s1_early2_v1: { label: 'S1早确认 +2min回落→DOWN', color: 'var(--chart-6)' },
  s1_dip45_v1: { label: 'S1低吸 DOWN≤0.45→DOWN', color: 'var(--chart-7)' },
  // 2026-09-07/08 首触反转族（专用表 firsthit_shadow_signals，影子+实盘通道）
  firsthit_down_v1: { label: '首触G0 基底 q∈(0.005,0.1]→DOWN', color: 'var(--chart-3)' },
  firsthit_down_body_v1: { label: '首触G1 body_r≤0.35→DOWN', color: 'var(--chart-4)' },
  firsthit_down_chg_v1: { label: '首触G3 v1 (已退役)', color: 'var(--chart-5)' },
  firsthit_down_chg_v2: { label: '首触G3v2 双波峰版 105-120s/210-240s→DOWN', color: 'var(--chart-5)' },
  firsthit_down_chg_early180_v1: { label: '首触G3前3分钟→DOWN', color: 'var(--chart-3)' },
  firsthit_down_k10_early180_v1: { label: '首触K10前3分钟→DOWN', color: 'var(--chart-5)' },
  process_recovery_down_v1: { label: '过程恢复双确认→DOWN', color: 'var(--chart-4)' },
  firsthit_down_k10_v1: { label: '首触K10标准化距离→DOWN', color: 'var(--chart-2)' },
  firsthit_down_k10_profit_v1: { label: '首触K10盈利过滤→DOWN', color: 'var(--chart-4)' },
  firsthit_down_g4_v1: { label: '首触G4 交互 chg≤2.82∧body≤0.35→DOWN', color: 'var(--chart-2)' },
  firsthit_down_g7_v1: { label: '首触G7 基底 body≤0.35∧wick=1→DOWN', color: 'var(--chart-6)' },
  g7_streak_v1: { label: '首触G7 非强连阳 streak≤1→DOWN', color: 'var(--chart-7)' },
  g7_wick20_v1: { label: '首触G7 长上影 wick≥2bp→DOWN', color: 'var(--chart-8)' },
  g7_strict_v1: { label: '首触G7 严格版 streak≤1∧wick≥1.5bp→DOWN', color: 'var(--chart-9)' },
  g7_q05_v1: { label: '首触G7 深折价 q≤0.05→DOWN', color: 'var(--chart-10)' },
  g7_t270_v1: { label: '首触G7 非极晚 t≤270s→DOWN', color: 'var(--chart-1)' },
  firsthit_down_btcsoft_v1: { label: '首触报价错杀 浅价∧BTC未上行→DOWN', color: 'var(--chart-2)' },
  // 2026-09-09 / 2026-09-13 Rev2 经典孕线反转族（共表 kline_shadow_signals，影子+实盘通道）
  hm_inside_15m_v2: { label: '15m 孕线上吊线 Inside Bar→DOWN', color: 'var(--chart-2)' },
  ih_inside_15m_v2: { label: '15m 孕线倒垂线 Inside Bar→UP', color: 'var(--chart-3)' },
  hm_inside_5m_v2: { label: '5m 孕线上吊线精选 Inside Bar→DOWN', color: 'var(--chart-4)' },
  // 2026-10-10 报价-人群错位族（gap_crowd_shadow_signals 表）——注册后同日证伪退役：
  // 标签保留供审计（曲线不再新增），实盘通道与影子采集均已停用。
  gap_crowd_5m_v1: { label: '报价落后人群 5m后半窗→UP（已证伪退役）', color: 'var(--chart-9)' },
  gap_crowd_15m_v1: { label: '报价落后人群 15m后半窗→UP（已证伪退役）', color: 'var(--chart-10)' },
}
/* 场景曲线色 = chart-1..5。
   注意：label 是 recharts 的 dataKey 本身，改 label 会同时改曲线与图例（AGENTS.md 明示），
   因此这里只改 color，label 一字未动。 */
const SCENE_META: Record<string, { label: string; color: string }> = {
  bull_exhaust: { label: 'S1 多头耗尽→DOWN', color: 'var(--chart-1)' },
  bear_exhaust: { label: 'S2 空头耗尽→UP', color: 'var(--chart-2)' },
  momentum_fade: { label: 'S4 动量衰竭→DOWN', color: 'var(--chart-3)' },
  bull_exhaust_confirm: { label: 'S5 确认入场→DOWN', color: 'var(--chart-4)' },
  // legacy = pattern_type 为空的历史信号，胜负按 side 映射（与 /api/fake-breakout/stats
  // 同语义，审计脚本对其「一律 DOWN」的简化口径在 side=low 时不同）
  legacy: { label: 'legacy 历史信号', color: 'var(--chart-10)' },
}
// 分析面板版本/场景的详细解释：优先取 ANALYTICS_EXTRA_DESC（SIGNAL_INFO 未收录的影子版本），
// 其次 SIGNAL_INFO（场景键加 scene_ 前缀）；后端动态发现的新版本回退展示原始键名。
// 2026-09-04 退役的纯影子版本（hm_touch_down_v1/v2、quote_momentum_v3）在此前置
// 【已退役：理由】，与 SIGNAL_INFO 的口径一致——说明保留供审计，但不得让人误以为还在采集。
const ANALYTICS_EXTRA_DESC: Record<string, string> = {
  gap_crowd_5m_v1: '⛔ 2026-10-10 注册后同日证伪退役：触发条件只在 up_price/down_price 快照不一致的瞬间成立（盲测 948 事件 100% 报价和 <0.95、中位 0.91，而全样本仅 1.03% 不一致、中位和 1.000），触发价不可成交（生产首单记录价 0.40 vs 实际 0.62）。一致化入场价重算 EV +0.036 → −0.072，只留一致样本则 0 事件——原「报价落后人群」边脚系数据伪迹。通道与影子均已退役停用（历史 1 条信号保留审计）。',
  gap_crowd_15m_v1: '⛔ 同上（15m 版）：盲测 250 事件 100% 报价和 <0.95，一致化入场价重算 EV +0.077 → −0.023。原 n=250 胜率 57.6% 系同一伪迹，勿启用。',
  krev_a_v1: 'K 线反转 A：15m 距前低≤−0.09 ATR + 5 根高效率阴跌 + 周期内 3 根 5m 子阴齐跌 → 押次根 15m 收阳 UP。影子持续记录；同名实盘通道默认关闭，护栏 0.629，仅开盘后 90 秒内的新鲜命中派单。',
  krev_b_v1: 'K 线反转 B：15m 区间贴底 + 5 根高效率阴跌 + 周期内 3 根 5m 子阴齐跌 → 押次根 15m 收阳 UP。影子持续记录；同名实盘通道默认关闭，护栏 0.621，仅开盘后 90 秒内的新鲜命中派单。',
  crv_hi_brk20_15m_v1: 'CRV 高位突破：15m 阳线突破 20 根高 ∧ pos100>0.90 → 押次根收阴 DOWN。2026-10-04 720D 预注册穷举+90D 盲测 n=212 胜率 63.7%（下界 57.0%），13/13 月为正。影子持续记录；同名实盘通道默认关闭，护栏 0.62。',
  crv_brk20_15m_v1: 'CRV 突破 20 高：15m 阳线突破 20 根高 → 押次根收阴 DOWN（假突破回归）。盲测 n=458 胜率 58.5%（下界 54.0%）。影子持续记录；实盘默认关闭，护栏 0.57；与 crv_hi_brk20/crv_brk50_hi 同窗互斥。',
  crv_brk50_hi_15m_v1: 'CRV 突破 50 高∧高位：15m 阳线突破 50 根高 ∧ pos100>0.75 → 押次根收阴 DOWN。盲测 n=226 胜率 60.6%（下界 54.1%）。影子持续记录；实盘默认关闭，护栏 0.59；同窗互斥。',
  crv_dnex_15m_v1: 'CRV 三因子·跌破出清（探索级）：15m 跌破 20 低 ∧ 贴 20 低 ∧ 实体>ATR → 押次根收阳 UP。发现 61.4%/验证 60.2%，未过盲测——仅影子前向验证，不注册实盘。',
  crv_brk20_5m_v1: 'CRV 5m 突破 20 高：5m 阳线突破 20 根高 → 押次根收阴 DOWN。盲测 n=1552 胜率 54.5%（下界 52.0%，5m 唯一强档），EV 薄、频率高。影子持续记录；实盘默认关闭，护栏 0.53。',
  crv_os_st4_5m_v1: 'CRV 三因子·超卖连阴（探索级）：5m RSI14<30 ∧ 恰连 4 阴 ∧ 跌破 20 低 → 押次根收阳 UP。发现 60.6%/验证 52.5%，未过盲测——仅影子，不注册实盘。',
  crv_hi_z3_15m_v1: 'CRV 三因子·高位 z 离群连 3 阳（观察级）：15m 阳线 ∧ pos100>0.90 ∧ z20>+2 ∧ 恰连 3 阳 → 押次根收阴 DOWN。发现 n=190 62.1%（下界 55.0%）/验证 67.6%，近期不衰减；未过盲测——仅影子前向验证，不注册实盘。',
  brkrv_brk8h_15m_v1: 'BRK 假突破回归：15m 收盘突破前 32 根（8h）高点 → 押次根收阴 DOWN。2026-10-05 研究三段全一致（训练 55.2% → 验证 58.8% → 盲测 55.7%），720D n=2768 胜率 56.8%。影子持续记录；同名实盘通道默认关闭，护栏 0.54；与 CRV 15m 突破族同窗互斥。',
  brkrv_brk8h_5m_v1: 'BRK 假突破回归：5m 收盘突破前 96 根（8h）高点 → 押次根收阴 DOWN。盲测市场结算口径（NOISE 计输）56.9%（n=144）；⚠ 横盘 regime NOISE 率飘高会摧毁经济 EV。影子持续记录；同名实盘通道默认关闭，护栏 0.55；与 CRV 5m 突破通道同窗互斥。',
  hm_touch_down_v1: '【2026-09-04 已退役：信息速率≈0（720d 触发 0.04~0.06 次/天，凑满 n=100 需 4~7 年），影子期无统计意义】HM 上吊线反弹入场：弱收盘上吊线（15m 小实体+深下影+贴 20 根高位+CLV≤0.75）→ 次 15m 周期内 10 分钟里若先反弹触及开盘价+0.25×ATR，按触及时刻真实报价记录押 DOWN（影子，不下单）。先破 −0.25×ATR、迟到触及或未触及均放弃，仅触及样本进胜率统计。720d 回测触价收跌率 58.7% vs 市场隐含 47.1%（n=46，覆盖率~36%），探索性发现，影子期即前向验证。',
  hm_touch_down_v2: '【2026-09-04 已退役：同 v1，720d 门禁后触发仅 78 次，信息速率不足以前向验证】HM 上吊线反弹 v2：v1 基础上叠加触发时点门禁——排除下跌段（过去 24h 跌≥1%）与低波环境（ATR 低于近 24h 中位数 80%）。720d 切片分析发现这两类样本负边际（下跌段 41.7% / 低波 25%），门禁后触发 78、触价 29、收跌率 69.0%。属后验切片假设（非预注册），与 v1 并行双行采集，影子期前向验证决定是否保留。',
  legacy: 'pattern_type 为空的历史信号：胜负按 side 映射（high→DOWN 赢 / low→UP 赢），用于对齐早期统计口径。',
  rev_p1_v1: '反转 P1：15m 连跌 4 根 + 弱阴收盘（贴最低，close_pos≤0.15）+ 成交量正常（[1.0,1.5)×20 根均量）→ 押次根 15m 收阳 UP。影子持续记录；同名实盘通道默认关闭，护栏 0.608，仅开盘后 90 秒内的新鲜命中派单。720d 回测胜率 62.0% / oos 63.9%。',
  rev_p2_v1: '反转 P2：15m 连涨 5 根 + 弱阳收盘（贴最高，close_pos≥0.85）→ 押次根 15m 收阴 DOWN。影子持续记录；同名实盘通道默认关闭，护栏 0.612，仅开盘后 90 秒内的新鲜命中派单。720d 回测胜率 62.4% / oos 61.3%。',
  s5_deep_z20_v1: 'S5 深档：S1 多头耗尽信号 +5min 回落确认，且回落幅度 z5=c5_close/anchor−1≤−20bp（深档）→ 按 +5min 时刻真实 15m DOWN 报价记录押 DOWN 的影子信号，仅记录不下单，次周期 15m 收阴判赢。落 pattern_shadow_signals（entry_state=TOUCHED，借用 HM 结算器）。720d 回测~91.3%（深档样本 EV 偏乐观、含机械成分），影子期即前向验证。',
  s1_early2_v1: 'S1 早确认：正式 S1 命中后，目标窗第 2 根 1m 收盘低于窗口开盘（BTC 已回落）才在 +2min 按真实 DOWN 报价入场，押次周期 15m DOWN。S5 的早确认替代版：买得更便宜、胜率略低、均EV更高。落 kline_shadow_signals；同名实盘通道默认关闭。',
  s1_dip45_v1: 'S1 低吸：正式 S1 命中后，+1/+2/+3/+5/+8 分钟首次 DOWN 报价 ≤0.45（影子按缓存价 ≤0.42）即入场，押次周期 15m DOWN。与 S1 开盘单同向叠加敞口。落 kline_shadow_signals；同名实盘通道默认关闭。',
  s1_dyn_sq_v1: 'S1 动态轧空路由：复用正式 S1 父信号，不改变原 S1/S5 通道配置。目标窗开盘时，以严格 ex-ante 的此前30天已完成5m K构造滚动ret1h分布；当前ret1h低于nearest-rank q90为NORMAL，按开盘近端真实DOWN报价入场；达到或超过q90为SQUEEZE，只在首根5m收阴后按+5min真实DOWN报价入场，否则跳过。q90只由过去行情分布计算，不看S1胜负，随波动regime自适应。720d重放原S1 n=2288/胜率58.3%，动态路由 n=1782/胜率64.9%；轧空确认组 n=684/78.1%，未确认组 n=506/DOWN胜率35.0%。影子持续采集；同名小额实盘通道默认关闭，护栏0.63，与原S1同窗互斥。',
  quote_momentum_v3: '【2026-09-04 已退役：修正未来函数后门禁效应≈0（+3.8pp，CI 重叠），不值得占实盘额度；随 momentum 族整体下线】报价动量 v3：在 v1（触发后 90~120s DOWN 报价 q∈[0.69,0.75)）基础上叠加“非连涨”门禁——用最后已收 15m（触发时刻所属 15m 的前一根，严格防未来函数）判定 close[j]≤close[j−1] 才落表 → 押 DOWN 的影子信号，落 misalignment_signals，按报价 edge 结算。回测修正未来函数后 80.2% vs 连涨 76.4%（+3.8pp，CI 重叠、门禁效应≈0），影子用于前向验证门禁是否真实有效。',
  nb_zschamp_15m_v1: 'nextbar 15m冠军：zscore_10≤-1.651 ∧ zscore_5≤-1.538 ∧ ret_3≤-0.00395（深超卖+急跌+卖盘衰竭）→ 押次根 15m 收阳 UP。源自 H=1 方向研究 converge_registry L3 ROBUST（holdout P(up_1)=61.96% n=368，月一致性 0.958 / walk-forward 1.00）；build_feature_matrix+condition_mask 实时重放冻结条件原文，与 KREV/反转共表 kline_shadow_signals（version+timeframe 隔离）。720d 次根收阳 58.92%（约2006触发）；生产影子 n=28 胜率67.9%、报价样本平均EV+0.130，但置信区间仍跨0。已注册默认关闭的小额实盘通道，MARKET护栏0.57（成交均价≥0.57弃单），仅目标根开盘后90秒内新鲜命中派单。',
  nb_smaslope_5m_v1: 'nextbar 5m误定价：sma_slope_atr_5≥1.661（5 根 SMA 陡峭上行/短期动量）→ 押次根 5m 收阳 UP 的影子信号，仅记录不下单。源自阶段E误定价扫描——市场报价钝在 q̄0.500 而 Jul-Aug 真实 P(UP)=0.534（B⁺ 逐笔 EV t=1.73 未达 t>3 门槛）。注意 720d 全样本次根收阳仅 47.43%（19597 触发，长样本反指），edge 依赖 Jul-Aug regime；影子期前向验证动量误定价是否持续，非背书。EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  combo_p1_v1: 'combo 组合 P1：连阳 3 根以上 ∧ 周末（UTC）∧ EMA20 乖离≥+0.3%（短线过热）→ 押次根 15m 收阴 DOWN 的影子信号，仅记录不下单。源自 45 维条件大搜索（720d 三三组合全扫）+ 1443 天样本外考试 + 50 次置换检验三重过滤后存活的“真层”——胜率来自时间×动量维度而非 K 线形态。720d n=490 胜率 63.9% / 样本外 n=1588 胜率 60.6%；EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  combo_p2_v1: 'combo 组合 P2：大实体（body_bp≥23.46，研究美元 P80 的冻结口径）∧ 连阳 3 根以上 ∧ 周末 → 押次根 15m 收阴 DOWN 的影子信号，仅记录不下单。P1 的加强版（实体更大）。720d n=206 胜率 66.5% / 样本外 n=957 胜率 60.5%；EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  combo_p3_v1: 'combo 组合 P3：贴 1 天高点（≤0.1%）∧ 美盘时段（UTC 16 点后）∧ 7 天涨≥4%（高位+中期动量衰减）→ 押次根 15m 收阴 DOWN 的影子信号，仅记录不下单。45 维大搜索存活组合（不依赖周末）。720d n=176 胜率 68.2% / 样本外 n=212 胜率 59.4%；EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  combo_p4_v1: 'combo 组合 P4：收低位（收盘在整根 K 线 1/4 以下）∧ 周末 ∧ RSI14≤25（超卖）→ 押次根 15m 收阳 UP 的影子信号，仅记录不下单。周末超卖反弹组合，与 P1 方向相反（对照组）。720d n=322 胜率 62.7% / 样本外 n=459 胜率 63.6%；EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  combo_p5_v1: 'combo 组合 P5：近光脚（下影≤5%）∧ 周末 ∧ RSI14≤25 → 押次根 15m 收阳 UP 的影子信号，仅记录不下单。P4 同簇的严格版（无下影=更干净的超卖），与 P4 并行采集、影子期用实数据对比两口径孰优。720d n=131 胜率 68.7% / 样本外 n=142 胜率 64.8%；EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1。',
  absorption_follow_td120_v1: '吸收跟随 TD120：5m 窗开 120s 决策点（严格 ex-ante 只读 ≤TD 采样），BTC 有明显位移（|btc_move|≥滚动 p50 位移门）而 UP token 真实报价「欠反应」（under=−sign(btc_move)·(up_move−(k·btc_move+b))≥滚动 p80 欠反应门，即报价粘滞/犹豫=知情者顶住人群吸筹的脚印）→ 顺 BTC 方向押（涨押 UP / 跌押 DOWN）的影子信号，仅记录不下单，次周期结算判赢。弹性 k 与两门禁用 trailing ~14 天缓冲严格 ex-ante 滚动标定（k 强 regime 依赖，本窗评估后才入缓冲）。落专用表 absorption_shadow_signals，与 TD150 变体前向 A/B 并行采集；Δvol/Δpar 为 soft 记录维度（不作门）。真实价复核 RECENT real n=397 命中 79.8% / EV +0.052（CI[-0.008,+0.112] 单看不显著）、FULL real n=1050 EV +0.089 CI✓；EV 按 TD 真实 token 报价前向现算：赢 0.98/q−1 / 输 −1。',
  absorption_follow_td150_v1: '吸收跟随 TD150：同 TD120 口径（欠反应跟随、双向顺势、trailing ~14 天滚动标定 k/门禁），但决策点延至窗开 150s（给报价更多时间暴露粘滞）。落专用表 absorption_shadow_signals，与 TD120 变体前向 A/B 并行采集。真实价复核 RECENT real n=399 命中 88.5% / EV +0.105（CI[+0.050,+0.165]✓，双 variant 中最稳健）、FULL real n=1083 EV +0.099 CI✓；EV 按 TD 真实 token 报价前向现算：赢 0.98/q−1 / 输 −1。',
  s2_cond_t4_v1: 'S2 条件单 t=4：由实盘 S2（空头耗尽 bear_exhaust：破 4h 支撑 + 收阴 + 放量）信号派生 → 次周期窗内等到 t=4（开盘后 240s），若该时刻 1m 收盘价 < 次周期开盘价（全深度回落，不设上界），按当时真实 15m UP 报价记录押次周期 15m 收阳 UP 的影子信号，仅记录不下单。研究结论：S2 开盘即买 UP 的 EV≈−0.042 不赚钱，等 t=4 价跌时 UP token 变便宜、低买 UP 的正 EV 来自入场价而非胜率。落 kline_shadow_signals（与 KREV/反转/nextbar/combo 共表、version 严格隔离结算，免迁移）。720d 触发 1069/2176（49.1%，1.48 次/天），价-only 胜率 38.9%；EV 按目标窗真实 15m UP 报价前向现算：赢 0.98/q−1 / 输 −1（报价表研究 EV +0.237 属乐观上界，不作基准）。',
  s2_cond_t5d_v1: 'S2 条件单 t=5 剔深：同由实盘 S2（bear_exhaust）派生 → 次周期窗内等到 t=5（开盘后 300s），若回落深度 0 < ln(开盘/px5) < 15bp（中度回落、剔除过深样本），按当时真实 15m UP 报价记录押次周期 15m 收阳 UP 的影子信号，仅记录不下单。剔深版单均优于 t=4 全深度（过深回落常伴随趋势性下破，剔掉后质量更高）。落 kline_shadow_signals（共表、version 严格隔离结算，免迁移）。720d 触发 643/2176（29.5%，0.89 次/天），价-only 胜率 44.8%；EV 按目标窗真实 15m UP 报价前向现算：赢 0.98/q−1 / 输 −1（报价表研究 EV +0.283 属乐观上界，不作基准）。',
  scene_bear_exhaust_opt_v1: 'S2 空头耗尽优化版：完全复用原 S2（跌破4小时支撑、15分钟收阴、量比≥2），再排除极端放量（量比≥4）和最近14日区间下方33%（pos14d<0.33）；保留后押下一15分钟 UP。14日高低点严格截止信号K收盘，无未来函数；历史不足或区间无波动保守不触发。720d n=975、胜率58.97%，前后半段均改善；影子持续采集，同名实盘通道默认关闭，护栏0.57，与原S2同窗互斥。',
  firsthit_down_v1: '首触反转 G0 基底（对照组）：5m 窗内 DOWN token 报价**首次**进入 (0.005, 0.1]（深折价，市场判定几乎不会跌）的采样点 → 按该时刻真实报价买 DOWN，押注最终结算 DOWN =「反转」的影子信号，仅记录不下单。触发时刻特征严格 ex-ante（只读 ≤触发时刻采样）：chg_bps（BTC 相对开盘涨跌）、body_r（|btc@触−开盘|/路径 max−min 归一实体）、npts（路径采样点数，<8 整窗不落表=路径太稀疏特征不可信）。落专用表 firsthit_shadow_signals，G1/G3 为 G0 的纯子集（同表 version 隔离、全特征落库，交叉门 G4=body∧chg 可事后重构）。44d 回测基底 EV +0.22（日聚类 CI[+0.10,+0.35]），EV 按逐事件真实触发价现算：赢 0.98/q−1 / 输 −1（禁用任何均值/分位代理）。预注册裁决（4 周前向）：若 G0 前向 EV 日聚类 CI **上界** < 0 → DOWN 侧 edge 消失，整族否决重来。',
  firsthit_down_body_v1: '首触反转 G1 小实体：G0 基底 + body_r ≤ 0.35 门禁（触发时刻 BTC 相对开盘的净位移只占窗内路径振幅的 ≤35%，即价格来回震荡而非单边跑出去）→ 按首触时刻真实 DOWN 报价买 DOWN 的影子信号，仅记录不下单。研究口径：45 维条件扫描中 body_r≤0.35 是唯一扛住 FDR 多重校正的形态门（q=0.007，logit 控 t+t² 后 β=+1.28 p=0.000），排除时间分段混淆。严格 ex-ante 时间切分两段考试：calib EV +1.44 → confirm EV +1.59（两段日聚类 CI 下界均 > 0）。落专用表 firsthit_shadow_signals（G0 纯子集，version 隔离），EV 逐事件真实触发价现算：赢 0.98/q−1 / 输 −1。预注册裁决（4 周前向）：通过 = 前向触发率 P ≥ 12% 且 EV 日聚类 CI 下界 > 0。',
  firsthit_down_chg_v1: '首触反转 G3 价格偏离（已退役）：历史v1口径因全窗无差别采集包含末段失血单且历史实盘存在LIMIT逆向选择，已归档封存供对账。',
  firsthit_down_chg_v2: '首触反转 G3v2 双波峰版：G0基底 + chg_bps≤+2.82bp 门禁，实盘与影子完全同源，限定两个正 EV 波峰 105~120s（早峰 13.8%/EV+0.607）与 210~240s（主峰 10.1%/EV+0.265）× 浅价 q≥0.07。4549 笔前向母事件 15S 分桶验证，彻底避开 240s+ 衰竭区与 285s+ 归零深渊。影子与实盘同步从 0 笔干净数据全新起跑。',
  firsthit_down_chg_early180_v1: 'G3前3分钟研究版：实际决策t<180秒、chg≤+2.82bp，按决策点真实DOWN报价记录。前向验证，实盘默认关闭。',
  firsthit_down_k10_early180_v1: 'K10前3分钟研究版：真实首触前缀t<180秒，G3且z≤1.85567，缺前序K线不开火。前向验证，实盘默认关闭。',
  process_recovery_down_v1: '过程段首个DOWN低价点建立观察；BTC回收≥35%与token回升≥0.02同点成立才确认。确认t<180秒、0.15<q≤0.35；使用确认价而非触达价。后验候选，无独立盲测背书，实盘默认关闭。',
  firsthit_down_k10_v1: 'K10：G3 + 带符号标准化结算距离 z≤1.85567；z=首触时chg_bps/(前12根完整5m收益标准差×sqrt(剩余秒数/300)×10000)。缺前序K线或首触BTC过旧不产生信号。发现段已用于选参，历史基准不钉死，前向影子现算裁决。',
  firsthit_down_k10_profit_v1: 'K10盈利版：K10通过后，首触距结算≤60秒时无条件剔除；距结算>60秒时，再剔除“前序最近6根平方收益和大于更早6根且隔离后的前序K线上影≥50%”事件。60秒边界按末段处理，必须以前向影子验证。',
  firsthit_down_g4_v1: '首触反转 G4 交互门：G0 基底 + body_r≤0.35（小实体）∧ chg_bps≤+2.82bp（价格偏离有限），即 G1 ∩ G3 的交集门 → 按首触时刻真实 DOWN 报价买 DOWN。机制：长时间窄幅震荡（chg 小）且实体占路径振幅比例低（body 小）后的破位，续跌概率最高。研究口径（shape_scan_v2 冻结扫描，SPLIT=2026-08-24，npts≥8）：calib n=214 P=12.6% EV +1.03 CI[+0.08,+2.22]；confirm n=86 P=23.3%（全表最高）EV +1.85 CI[+0.35,+3.44]；binom p=0.025 → BH-FDR q=0.063 ✅ STRICT_PASS。落专用表 firsthit_shadow_signals，EV 逐事件真实触发价现算：赢 0.98/q−1 / 输 −1。实盘护栏 0.12（按 calib P=12.6% → q*=0.1235 保守取值，边际 2.8%）。⚠️非独立暴露：G4 ⊂ G1 且 G4 ⊂ G3，同窗三门全中即对同一 DOWN 事件重复下注（含 G0 共 4 倍），非 4 个独立信号。⚠️前向功效偏紧：G4 calib EV(+1.03) 与前向 4 周 CI 半宽(≈1.08) 几乎相等，即使真实效应完全等于 calib 估计，通过「EV 日聚类 CI 下界>0」的概率也仅约 50%；FAIL 时应延长观察期而非直接否决。',
  firsthit_down_g7_v1: '首触反转 G7 基底：G0 基底 + body_r≤0.35（小实体）∧ wick01=1.0（上影拒绝），捕捉冲高受阻回落反转。40d 全量回测 Calib 胜率 18.8% (EV +2.34)，Confirm 盲测 20.0% (EV +1.98，CI[+0.45,+3.46])，FDR q=0.0002。日均 ~6.5 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
  g7_streak_v1: '首触反转 G7 非强连阳：G7 组合 + 排除大级别连续多头单边（前驱 5m 连阳数 streak_up≤1 根）。回测 Calib 胜率 20.0% (EV +2.86)，Confirm 胜率 18.2% (EV +1.92，CI[+0.45,+3.30])，FDR q=0.0004，日均 5.1 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
  g7_wick20_v1: '首触反转 G7 长上影：G7 组合 + 显著上影线拒绝 upper_wick_bps≥2.0bp。多头更深力竭，回测 Confirm 胜率 24.1% (EV +3.06，CI[+0.94,+5.09])，FDR q=0.0304，日均 ~3.2 单。落专用表 firsthit_shadow_signals。实盘护栏 0.12。',
  g7_strict_v1: '首触反转 G7 严格版：G7 组合 + streak_up≤1 ∧ upper_wick_bps≥1.5bp。双重质量门，回测 Calib 胜率 23.0% (EV +3.27)，Confirm 胜率 20.8% (EV +2.68，CI[+0.68,+4.51])，FDR q=0.0030，日均 ~2.9 单。落专用表 firsthit_shadow_signals。实盘护栏 0.12。',
  g7_q05_v1: '首触反转 G7 深折价：G7 组合 + 进场报价 q≤0.05（极端凸性赔率档，单注赢付 >18.6x）。回测单注 EV +4.10~+5.77，日均 ~1.3 单。落专用表 firsthit_shadow_signals。实盘护栏 0.05。',
  g7_t270_v1: '首触反转 G7 非极晚：G7 组合 + 触发时刻 t≤270s（排除最后 30s 缺乏均值回归扩散时间的毒瘤窗）。回测 Confirm 胜率 30.4% (EV +3.08)，FDR q=0.0002，日均 ~2.1 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
  hm_inside_15m_v2: '15m 经典孕线上吊线反转：前置 4 根高点最长实体阳线 + 信号柱完全包裹(Inside Bar) + 顶部长下影上吊线(lower_r≥45%)、短上影 upper_r≤10% → 押次根 15m 收阴 DOWN。落表 kline_shadow_signals。720d 全样本回测 n=237 胜率 55.7%，30d 胜率 60.0%。MARKET 市价单护栏 0.30（成交均价 ≥0.30 弃单）。EV 按目标窗真实报价前向现算。',
  ih_inside_15m_v2: '15m 经典孕线倒垂线反转：前置 4 根低点最长实体阴线 + 信号柱完全包裹(Inside Bar) + 底部长上影倒垂线(upper_r≥45%)、短下影 lower_r≤10% → 押次根 15m 收阳 UP。落表 kline_shadow_signals。720d 全样本回测 n=183 胜率 54.6%，30d 胜率 77.8%。MARKET 市价单护栏 0.30（成交均价 ≥0.30 弃单）。EV 按目标窗真实报价前向现算。',
  hm_inside_5m_v2: '5m 孕线上吊线精选反转：前置 4 根高点最长实体阳线 + 信号柱完全包裹(Inside Bar) + 下影占比 [75%,90%)、短上影 upper_r≤10% → 押次根 5m 收阴 DOWN。落表 kline_shadow_signals。720d 全样本回测 n=371 胜率 58.5%，30d 胜率 71.4%。MARKET 市价单护栏 0.30（成交均价 ≥0.30 弃单）。EV 按目标窗真实报价前向现算。',
  candle_hm_bull_5m_consensus2_shadow_v1: '5m 主影子：阳线、下影≥50%、实体≤50%、上影≤10%，且 ret3/ret5_majority/ma10 至少两种上涨，预测次根 DOWN。只记录不下注。',
  candle_hm_bull_5m_consensus3_shadow_v1: '5m 严格对照：主影子的三趋势全同意子集。与主版本共用一个物理事件，不代表第二笔独立暴露。',
  candle_hm_bull_15m_union_shadow_v1: '15m 主影子：同一长下影阳线形态，三趋势任一种上涨即记录，预测次根 DOWN。只记录不下注。',
  candle_hm_bull_15m_consensus3_shadow_v1: '15m 严格对照：主版本的三趋势全同意子集；共用物理事件。',
  candle_hm_bull_5m_ret5_majority_component_shadow_v1: '5m ret5_majority 单趋势归因标签；仅解释边际贡献，不增加事件或暴露。',
  candle_hm_bull_5m_ret3_component_shadow_v1: '5m ret3 单趋势归因标签；仅解释边际贡献，不增加事件或暴露。',
  candle_hm_bull_15m_ret3_component_shadow_v1: '15m ret3 单趋势归因标签；仅解释边际贡献，不增加事件或暴露。',
  candle_hm_bull_15m_ma10_component_shadow_v1: '15m ma10 单趋势归因标签；仅解释边际贡献，不增加事件或暴露。',
  candle_hm_bull_5m_volume_mid_high_shadow_hyp_v1: '预注册潜力：5m 主形态 + 0.75≤volume/median20<2。后验 regime 假设，只能用新数据前向验证。',
  candle_hm_bull_5m_trend4h_up_shadow_hyp_v1: '预注册潜力：5m 主形态 + 前4h上涨。只作标签，不替代主影子。',
  candle_hm_bull_5m_asia_shadow_hyp_v1: '预注册潜力：5m 主形态 + UTC 00~08 亚洲时段。只作标签。',
  candle_hm_bull_15m_q45_55_shadow_hyp_v1: '预注册潜力：15m 主形态 + 首个完整 DOWN 报价落在 [0.45,0.55)。只作标签。',
  candle_hm_bull_15m_trend1h_up_shadow_hyp_v1: '预注册潜力：15m 主形态 + 前1h上涨。只作标签。',
  candle_hm_bull_15m_asia_shadow_hyp_v1: '预注册潜力：15m 主形态 + UTC 00~08 亚洲时段。低样本，只作标签。',
  candidate_5m_ret5_majority_up_upper_bull_l70b30m10_shadow_hyp_v1: '独立潜力：5m ret5上涨、阳线长上影≥70%、实体≤30%、下影≤10%，预测 DOWN。',
  candidate_5m_ret3_up_lower_bear_l50b50m0_shadow_hyp_v1: '独立潜力：5m ret3上涨、阴线长下影≥50%、实体≤50%、原始 OHLC 严格零上影，预测 DOWN。',
  candidate_15m_ret5_majority_up_lower_bull_l50b40m10_shadow_hyp_v1: '独立潜力：15m ret5上涨、阳线长下影≥50%、实体≤40%、上影≤10%，预测 DOWN。',
  candidate_15m_ret3_down_upper_bull_l50b40m10_shadow_hyp_v1: '独立潜力：15m ret3下跌、阳线长上影≥50%、实体≤40%、下影≤10%，预测 UP。',
}
const signalDescFor = (kind: 'scene' | 'shadow', key: string): string => {
  if (ANALYTICS_EXTRA_DESC[key]) return ANALYTICS_EXTRA_DESC[key]
  const info = kind === 'scene' ? SIGNAL_INFO[`scene_${key}`] : SIGNAL_INFO[key]
  return info?.desc ?? ''
}
// 后端动态发现的新版本/新场景的备用色（超出已知名单时按序取用）
/* 后端动态新增版本的兜底色：接在 chart-5 之后，避开 SCENE_META 已占用的 1..5 */
const EXTRA_COLORS = ['var(--chart-6)', 'var(--chart-7)', 'var(--chart-8)', 'var(--chart-9)', 'var(--chart-10)']

const pct1 = (v: number | null | undefined, digits = 1) =>
  v == null ? '—' : `${(v * 100).toFixed(digits)}%`
const evFmt = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(3)}`
const utcMD = (ts: number) =>
  new Date(ts).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', timeZone: 'UTC' })

// 把各版本曲线按时间戳合并成宽表（recharts 单 data 多 Line 需要）
function mergeCurves(
  blocks: Record<string, { curve: AnalyticsCurvePoint[]; guarded_curve?: AnalyticsCurvePoint[] }>,
  meta: Record<string, { label: string; color: string }>,
  useGuarded = false,
): Record<string, number>[] {
  const rowMap = new Map<number, Record<string, number>>()
  for (const [key, blk] of Object.entries(blocks)) {
    const label = meta[key]?.label ?? key
    const pts = (useGuarded && blk.guarded_curve) ? blk.guarded_curve : blk.curve
    for (const p of pts) {
      const row = rowMap.get(p.ts) ?? { ts: p.ts }
      row[label] = +(p.cum_wr * 100).toFixed(1)
      rowMap.set(p.ts, row)
    }
  }
  return Array.from(rowMap.values()).sort((a, b) => (a.ts as number) - (b.ts as number))
}

function isExecutionComparison(value: unknown): value is ExecutionComparison {
  if (!value || typeof value !== 'object') return false
  const v = value as Partial<ExecutionComparison>
  const isNum = (n: unknown) => typeof n === 'number' && Number.isFinite(n)
  const isNullableNum = (n: unknown) => n === null || isNum(n)
  const validFunnel = (f: unknown): f is ExecutionFunnel => {
    if (!f || typeof f !== 'object') return false
    const row = f as Record<string, unknown>
    return ['theoretical', 'strategy_eligible', 'operational_eligible', 'execution_eligible', 'submitted', 'filled', 'settled', 'unknown', 'no_live_mapping'].every(k => isNum(row[k]))
  }
  const validFixed = (r: unknown): r is FixedUnitReturn => !!r && typeof r === 'object'
    && isNum((r as FixedUnitReturn).n) && isNum((r as FixedUnitReturn).sum) && isNullableNum((r as FixedUnitReturn).avg)
  const validReturns = (r: unknown): r is ExecutionReturns => {
    if (!r || typeof r !== 'object') return false
    const row = r as ExecutionReturns
    return validFixed(row.theoretical_fixed_1u) && validFixed(row.executable_fixed_1u)
      && !!row.actual && isNum(row.actual.settled_n) && isNum(row.actual.stake)
      && isNum(row.actual.pnl) && isNullableNum(row.actual.roi)
  }
  const validGaps = (g: unknown): g is ExecutionGaps => {
    if (!g || typeof g !== 'object') return false
    const row = g as Record<string, unknown>
    return ['selection', 'operational', 'execution', 'fill', 'settlement', 'coverage'].every(k => {
      const gap = row[k] as Partial<ExecutionGap> | undefined
      return !!gap && isNum(gap.n) && isNullableNum(gap.rate)
    })
  }
  const validReasons = (r: unknown) => Array.isArray(r) && r.every(x => !!x && typeof x === 'object'
    && typeof (x as ExecutionReasonCount).reason === 'string' && isNum((x as ExecutionReasonCount).count))
  const scope = v.scope
  if (!scope || typeof scope !== 'object' || typeof scope.include_retired !== 'boolean') return false
  if (!(scope.policy_version === null || typeof scope.policy_version === 'string')) return false
  if (!(scope.market_period === null || typeof scope.market_period === 'string')) return false
  if (!isNullableNum(scope.from) || !isNullableNum(scope.to)) return false
  if (!validFunnel(v.funnel) || !validReturns(v.returns) || !validGaps(v.gaps) || !validReasons(v.reason_counts)) return false
  if (!Array.isArray(v.versions) || !v.versions.every(row => !!row && typeof row.signal_version === 'string'
    && (row.policy_version === null || typeof row.policy_version === 'string')
    && (row.market_period === null || typeof row.market_period === 'string')
    && typeof row.channel_mapped === 'boolean' && typeof row.retired === 'boolean'
    && validFunnel(row.funnel) && validReturns(row.returns) && validGaps(row.gaps) && validReasons(row.reason_counts))) return false
  return Array.isArray(v.curves) && v.curves.every(p => !!p && isNum(p.timestamp)
    && typeof p.signal_version === 'string'
    && (p.policy_version === null || typeof p.policy_version === 'string')
    && isNullableNum(p.theoretical_fixed_1u) && isNullableNum(p.executable_fixed_1u)
    && isNum(p.actual_stake) && isNullableNum(p.actual_pnl) && isNullableNum(p.actual_roi))
}

const executionTs = (ts: number) => ts < 10_000_000_000 ? ts * 1000 : ts
const executionPct = (v: number | null) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
const executionMoney = (v: number | null) => v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`

function ShadowExecutionComparisonCard({
  data,
  unavailable,
}: {
  data: ExecutionComparison | null
  unavailable: string
}) {
  const [versionText, setVersionText] = useState('')
  const [period, setPeriod] = useState('ALL')
  const [mapping, setMapping] = useState<'ALL' | 'MAPPED' | 'UNMAPPED'>('ALL')
  const [retired, setRetired] = useState<'ALL' | 'ACTIVE' | 'RETIRED'>('ALL')
  const [policy, setPolicy] = useState('ALL')
  const [evidenceAmount, setEvidenceAmount] = useState('ALL')

  if (!data) {
    return (
      <Card title="影子信号·理论到实盘执行对比">
        <div className="rounded-sm border border-line bg-sunken px-4 py-6 text-center text-xs text-ink-55">
          {unavailable || '执行对比数据加载中...'}
        </div>
      </Card>
    )
  }

  const periods = Array.from(new Set(data.versions.map(v => v.market_period).filter((v): v is string => !!v))).sort()
  const policies = Array.from(new Set(data.versions.map(v => v.policy_version).filter((v): v is string => !!v))).sort()
  const search = versionText.trim().toLowerCase()
  const versions = data.versions.filter(v => {
    if (search && !v.signal_version.toLowerCase().includes(search)) return false
    if (period !== 'ALL' && v.market_period !== period) return false
    if (policy !== 'ALL' && v.policy_version !== policy) return false
    if (mapping === 'MAPPED' && !v.channel_mapped) return false
    if (mapping === 'UNMAPPED' && v.channel_mapped) return false
    if (retired === 'ACTIVE' && v.retired) return false
    if (retired === 'RETIRED' && !v.retired) return false
    return true
  })
  const versionKeys = new Set(versions.map(v => `${v.signal_version}\u0000${v.policy_version ?? ''}`))
  let previousTheoretical = 0
  let previousExecutable = 0
  let previousStake = 0
  let previousPnl = 0
  let selectedTheoretical = 0
  let selectedExecutable = 0
  let selectedStake = 0
  let selectedPnl = 0
  const curves = data.curves.flatMap(p => {
    const theoretical = p.theoretical_fixed_1u ?? previousTheoretical
    const executable = p.executable_fixed_1u ?? previousExecutable
    const pnl = p.actual_pnl ?? previousPnl
    const theoreticalDelta = theoretical - previousTheoretical
    const executableDelta = executable - previousExecutable
    const stakeDelta = p.actual_stake - previousStake
    const pnlDelta = pnl - previousPnl
    previousTheoretical = theoretical
    previousExecutable = executable
    previousStake = p.actual_stake
    previousPnl = pnl
    if (!versionKeys.has(`${p.signal_version}\u0000${p.policy_version ?? ''}`)) return []
    selectedTheoretical += theoreticalDelta
    selectedExecutable += executableDelta
    selectedStake += stakeDelta
    selectedPnl += pnlDelta
    const actualRoi = selectedStake > 0 ? selectedPnl / selectedStake : null
    return [{
      ...p,
      timestamp: executionTs(p.timestamp),
      theoretical_fixed_1u: selectedTheoretical,
      executable_fixed_1u: selectedExecutable,
      actual_stake: selectedStake,
      actual_pnl: selectedPnl,
      actual_roi: actualRoi,
      actual_roi_pct: actualRoi == null ? null : actualRoi * 100,
    }]
  })
  const reasons = new Map<string, number>()
  for (const row of versions) for (const reason of row.reason_counts) {
    reasons.set(reason.reason, (reasons.get(reason.reason) ?? 0) + reason.count)
  }
  const reasonRows = Array.from(reasons.entries()).sort((a, b) => b[1] - a[1])
  const filterActive = !!search || period !== 'ALL' || mapping !== 'ALL' || retired !== 'ALL' || policy !== 'ALL'
  const filteredFunnel = versions.reduce<ExecutionFunnel>((total, row) => ({
    theoretical: total.theoretical + row.funnel.theoretical,
    strategy_eligible: total.strategy_eligible + row.funnel.strategy_eligible,
    operational_eligible: total.operational_eligible + row.funnel.operational_eligible,
    execution_eligible: total.execution_eligible + row.funnel.execution_eligible,
    submitted: total.submitted + row.funnel.submitted,
    filled: total.filled + row.funnel.filled,
    settled: total.settled + row.funnel.settled,
    unknown: total.unknown + row.funnel.unknown,
    no_live_mapping: total.no_live_mapping + row.funnel.no_live_mapping,
  }), {
    theoretical: 0, strategy_eligible: 0, operational_eligible: 0,
    execution_eligible: 0, submitted: 0, filled: 0, settled: 0,
    unknown: 0, no_live_mapping: 0,
  })
  const fixedReturn = (key: 'theoretical_fixed_1u' | 'executable_fixed_1u'): FixedUnitReturn => {
    const total = versions.reduce((acc, row) => ({
      n: acc.n + row.returns[key].n,
      sum: acc.sum + row.returns[key].sum,
    }), { n: 0, sum: 0 })
    return { ...total, avg: total.n > 0 ? total.sum / total.n : null }
  }
  const actual = versions.reduce<ActualExecutionReturn>((total, row) => ({
    settled_n: total.settled_n + row.returns.actual.settled_n,
    stake: total.stake + row.returns.actual.stake,
    pnl: total.pnl + row.returns.actual.pnl,
    roi: null,
  }), { settled_n: 0, stake: 0, pnl: 0, roi: null })
  actual.roi = actual.stake > 0 ? actual.pnl / actual.stake : null
  const filteredReturns: ExecutionReturns = {
    theoretical_fixed_1u: fixedReturn('theoretical_fixed_1u'),
    executable_fixed_1u: fixedReturn('executable_fixed_1u'),
    actual,
  }
  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'
  const funnel = [
    ['理论影子', filteredFunnel.theoretical], ['策略通过', filteredFunnel.strategy_eligible],
    ['运营通过', filteredFunnel.operational_eligible], ['执行通过', filteredFunnel.execution_eligible],
    ['已下单', filteredFunnel.submitted], ['已成交', filteredFunnel.filled], ['已结算', filteredFunnel.settled],
  ] as const
  const help = '理论与门禁子集均采用参考结算的固定 1U 公式，不是可执行收益，不代表报价或成交；门禁通过不证明实时冻结或前向准入，健康不等于准入。本轮仅持有到结算，未验证提前退出。实际 ROI = sum(pnl) / sum(stake)，为本金加权口径。unknown（未知）不是拒绝；no mapping（无实盘映射）不是执行失败。执行可行不保证 LIMIT 限价单成交。历史缺失的瞬时配置、市场、余额与互斥状态不会用当前状态反推重建。'
  const gapLabel: Record<keyof ExecutionGaps, string> = {
    selection: '策略', operational: '运营', execution: '执行', fill: '成交', settlement: '结算', coverage: '覆盖',
  }

  return (
    <Card title="影子信号·理论到实盘执行对比">
      {data.forward_evidence && <div className="border border-line p-3 mb-3 text-xs">
        <strong>前向证据：阻塞准入</strong><HelpHint text="报价不是成交；费用为假设，仅持有到结算。旧账本与归档不计实时证据。" />
        <div className="font-mono">{data.forward_evidence.cohort}</div>
        <div>实时 {data.forward_evidence.live_n} · 归档 {data.forward_evidence.archive_n} · 独立窗 {data.forward_evidence.independent_windows} · 独立日 {data.forward_evidence.independent_days} · 同窗同向配对 {data.forward_evidence.paired_same_window_direction} · 缺失 {data.forward_evidence.pair_missing}</div>
        {Object.entries(data.forward_evidence.ladder).map(([amount, value]) => <div key={amount}>{amount}U 报价结算估计：样本 {value.n} · 本金 {value.stake.toFixed(2)} · 净收益 {value.pnl.toFixed(2)} · 压力1% {value.stress_1pct.toFixed(2)} · 压力3% {value.stress_3pct.toFixed(2)} · 缺失 {value.missing}</div>)}
        <label>证据金额 <select value={evidenceAmount} onChange={e => setEvidenceAmount(e.target.value)}><option value="ALL">全部（独立展示，不合计）</option>{[...new Set(data.forward_evidence.series?.map(row => row.amount))].map(amount => <option key={amount} value={amount}>{amount}U</option>)}</select></label>
        {data.forward_evidence.series?.filter(row => (!search || row.signal_version.toLowerCase().includes(search)) && (evidenceAmount === 'ALL' || row.amount === evidenceAmount) && (policy === 'ALL' || row.policy_version === policy)).map(row => <div key={`${row.signal_version}-${row.policy_version}-${row.amount}`}>{row.signal_version} · {row.amount}U · 样本 {row.n} · 报价估计收益 {row.pnl.toFixed(2)} · 最大回撤 {row.max_drawdown.toFixed(2)} · 去最大赢单 {row.without_top_win_pnl.toFixed(2)} · 最大赢单占比 {row.top_win_positive_share == null ? '未知' : `${(100*row.top_win_positive_share).toFixed(1)}%`} · 缺失 {Object.values(row.missing).reduce((a,b) => a+b, 0)}</div>)}
        <div>{data.forward_evidence.blocking_reasons.join('；')}</div>
        <details><summary>全家族实时与归档覆盖<HelpHint text="零样本不隐去；实时冻结仍须报价、运营和结算证据，归档不可替代准入。" /></summary>
          {data.coverage_matrix?.map(row => <div key={row.signal_version}>{row.signal_version} · 实时 {row.live_frozen_n} · 归档/旧账本 {row.archive_or_legacy_n} · {row.capture_capability === 'LIVE_ENTRY' ? '已接实时入口' : '仅归档，阻塞'} · {row.blocking_reasons.join('；')}</div>)}
        </details>
      </div>}
      <div className="flex items-start gap-1 mb-3 text-xs text-ink-55">
        <span>统一查看信号从理论样本到真实结算的漏斗、收益与缺口。</span><HelpHint text={help} />
      </div>

      <div className="overflow-x-auto mb-3">
        <div className="flex min-w-[760px] items-stretch gap-1">
          {funnel.map(([label, value], i) => (
            <Fragment key={label}>
              <div className="flex-1 border border-line rounded-sm bg-sunken px-2 py-2 text-center">
                <div className="text-[10px] text-ink-55">{label}</div>
                <div className="font-mono text-lg font-bold text-ink-95">{value.toLocaleString()}</div>
                {i > 0 && <div className="text-[10px] text-ink-40">前步 {funnel[i - 1][1] > 0 ? `${(value / funnel[i - 1][1] * 100).toFixed(1)}%` : '—'}</div>}
              </div>
              {i < funnel.length - 1 && <div className="self-center text-ink-40">→</div>}
            </Fragment>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2 mb-4 text-xs">
        <span className="ds-badge ds-badge-neutral">未知 {filteredFunnel.unknown}</span>
        <span className="ds-badge ds-badge-neutral">无实盘映射 {filteredFunnel.no_live_mapping}</span>
        {filterActive && <span className="ds-badge ds-badge-neutral">当前筛选汇总</span>}
        <span className="text-[10px] text-ink-55 self-center">未知不是拒绝；无映射不是执行失败。</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-4">
        {[
          ['理论固定 1U', filteredReturns.theoretical_fixed_1u, 'var(--chart-1)'],
          ['门禁子集参考 1U', filteredReturns.executable_fixed_1u, 'var(--warning)'],
        ].map(([label, raw, color]) => {
          const r = raw as FixedUnitReturn
          return <div key={label as string} className="border border-line rounded-sm bg-card p-3">
            <div className="text-xs text-ink-55">{label as string}</div>
            <div className="mt-1 font-mono font-bold" style={{ color: color as string }}>{executionMoney(r.sum)} U</div>
            <div className="text-[10px] text-ink-55">n={r.n} · 均值 {executionMoney(r.avg)} U</div>
          </div>
        })}
        <div className="border border-line rounded-sm bg-card p-3">
          <div className="text-xs text-ink-55">实际实盘（本金加权）</div>
          <div className={`mt-1 font-mono font-bold ${filteredReturns.actual.pnl >= 0 ? 'text-positive' : 'text-negative'}`}>
            {executionMoney(filteredReturns.actual.pnl)} USDT · {executionPct(filteredReturns.actual.roi)}
          </div>
          <div className="text-[10px] text-ink-55">结算 {filteredReturns.actual.settled_n} · 本金 {filteredReturns.actual.stake.toFixed(2)} USDT</div>
        </div>
      </div>

      <div className="flex items-center justify-between mb-3 gap-2 flex-wrap text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <input value={versionText} onChange={e => setVersionText(e.target.value)} placeholder="搜索版本..." className={`${selectCls} w-36`} />
          <select value={period} onChange={e => setPeriod(e.target.value)} className={selectCls}><option value="ALL">全部周期</option>{periods.map(v => <option key={v}>{v}</option>)}</select>
          <select value={mapping} onChange={e => setMapping(e.target.value as typeof mapping)} className={selectCls}><option value="ALL">全部映射</option><option value="MAPPED">已有映射</option><option value="UNMAPPED">无映射</option></select>
          <select value={retired} onChange={e => setRetired(e.target.value as typeof retired)} className={selectCls}><option value="ALL">全部退役状态</option><option value="ACTIVE">未退役</option><option value="RETIRED">已退役</option></select>
          <select value={policy} onChange={e => setPolicy(e.target.value)} className={selectCls}><option value="ALL">全部策略版本</option>{policies.map(v => <option key={v}>{v}</option>)}</select>
          {filterActive && <button onClick={() => { setVersionText(''); setPeriod('ALL'); setMapping('ALL'); setRetired('ALL'); setPolicy('ALL') }} className="text-brand hover:underline">清除筛选</button>}
        </div>
        <span className="text-ink-55">显示 {versions.length} / 共 {data.versions.length} 个版本</span>
      </div>

      {versions.length === 0 ? <div className="text-center text-ink-55 py-6 text-xs">当前筛选条件下无匹配版本</div> : (
        <div className="overflow-x-auto mb-4">
          <table className="w-full min-w-[1160px] text-xs">
            <thead><tr className="border-b border-line text-ink-55">
              <th className="py-1 px-2 text-left">版本</th><th className="py-1 px-2 text-left">周期 / 策略</th>
              <th className="py-1 px-2 text-right">理论 n / sum</th><th className="py-1 px-2 text-right">执行通过 / sum</th>
              <th className="py-1 px-2 text-right">实际结算 / 本金</th><th className="py-1 px-2 text-right">ROI / PnL</th>
              {(Object.keys(gapLabel) as (keyof ExecutionGaps)[]).map(k => <th key={k} className="py-1 px-2 text-right">{gapLabel[k]}缺口</th>)}
            </tr></thead>
            <tbody>{versions.map(v => <tr key={`${v.signal_version}-${v.policy_version ?? ''}-${v.market_period ?? ''}`} className={`border-b border-line-soft hover:bg-sunken ${v.retired ? 'opacity-55' : ''}`}>
              <td className="py-1.5 px-2"><span className="font-medium text-ink-95">{v.signal_version}</span><div className="text-[10px] text-ink-55">{v.channel_mapped ? '有映射' : '无映射'}{v.retired ? ' · 已退役' : ''}</div></td>
              <td className="py-1.5 px-2 text-ink-55"><span className="font-mono">{v.market_period ?? '—'}</span><div>{v.policy_version ?? '—'}</div></td>
              <td className="py-1.5 px-2 text-right font-mono">{v.returns.theoretical_fixed_1u.n} / {executionMoney(v.returns.theoretical_fixed_1u.sum)}</td>
              <td className="py-1.5 px-2 text-right font-mono">{v.funnel.execution_eligible} / {executionMoney(v.returns.executable_fixed_1u.sum)}</td>
              <td className="py-1.5 px-2 text-right font-mono">{v.returns.actual.settled_n} / {v.returns.actual.stake.toFixed(2)}</td>
              <td className={`py-1.5 px-2 text-right font-mono ${v.returns.actual.pnl >= 0 ? 'text-positive' : 'text-negative'}`}>{executionPct(v.returns.actual.roi)} / {executionMoney(v.returns.actual.pnl)}</td>
              {(Object.keys(gapLabel) as (keyof ExecutionGaps)[]).map(k => <td key={k} className="py-1.5 px-2 text-right font-mono" title={`n=${v.gaps[k].n}`}>{executionPct(v.gaps[k].rate)}</td>)}
            </tr>)}</tbody>
          </table>
        </div>
      )}

      <div className="mb-4">
        <div className="text-xs font-medium text-ink-80 mb-2">主要缺口原因</div>
        {reasonRows.length > 0 ? <div className="flex flex-wrap gap-1.5">{reasonRows.map(([reason, count]) => <span key={reason} className="px-2 py-1 border border-line rounded-sm bg-sunken text-[11px] text-ink-80"><span className="font-mono font-bold">{count}</span> · {reason}</span>)}</div> : <div className="text-xs text-ink-55">当前范围无缺口原因记录。</div>}
      </div>

      {curves.length > 0 ? <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div><div className="text-xs text-ink-55 mb-1">固定 1U 累计收益（U）</div><ResponsiveContainer width="100%" height={220}><LineChart data={curves} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--line-soft)" /><XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} scale="time" tickFormatter={utcMD} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
          <YAxis tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={48} /><Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)} formatter={(v, n) => [typeof v === 'number' ? `${v.toFixed(2)} U` : '—', n]} /><Legend />
          <Line name="理论固定 1U" dataKey="theoretical_fixed_1u" stroke="var(--chart-1)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} /><Line name="门禁子集参考 1U" dataKey="executable_fixed_1u" stroke="var(--warning)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
        </LineChart></ResponsiveContainer></div>
        <div><div className="text-xs text-ink-55 mb-1">实际累计 PnL（USDT）与本金加权 ROI</div><ResponsiveContainer width="100%" height={220}><LineChart data={curves} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--line-soft)" /><XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} scale="time" tickFormatter={utcMD} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
          <YAxis yAxisId="pnl" tick={{ fontSize: 11, fill: 'var(--ink-55)' }} stroke="var(--positive)" width={68} tickFormatter={(v: number) => `${v} USDT`} /><YAxis yAxisId="roi" orientation="right" tick={{ fontSize: 11, fill: 'var(--ink-55)' }} stroke="var(--warning)" width={48} tickFormatter={(v: number) => `${v}%`} />
          <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)} formatter={(v, n) => [typeof v === 'number' ? `${v.toFixed(2)}${n === '实际 ROI' ? '%' : ' USDT'}` : '—', n]} /><Legend />
          <Line yAxisId="pnl" name="实际 PnL" dataKey="actual_pnl" stroke="var(--positive)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} /><Line yAxisId="roi" name="实际 ROI" dataKey="actual_roi_pct" stroke="var(--warning)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
        </LineChart></ResponsiveContainer></div>
      </div> : <div className="text-center text-ink-55 py-3 text-xs">当前范围暂无执行收益曲线。</div>}
      <div className="text-[10px] text-ink-55 mt-2">可执行仅表示通过记录到的执行前置条件，不保证 LIMIT 限价单成交；历史缺失的瞬时配置、市场、余额与互斥状态不从当前状态重建。</div>
    </Card>
  )
}

function SignalAnalyticsTab() {
  const [analytics, setAnalytics] = useState<SignalsAnalytics | null>(null)
  const [executionComparison, setExecutionComparison] = useState<ExecutionComparison | null>(null)
  const [executionUnavailable, setExecutionUnavailable] = useState('')
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

  const loadExecutionComparison = useCallback(() => {
    api.getSignalsExecutionComparison().then(value => {
      if (isExecutionComparison(value)) {
        setExecutionComparison(value)
        setExecutionUnavailable('')
      } else {
        setExecutionComparison(null)
        setExecutionUnavailable('执行对比暂不可用：后端返回格式不兼容。')
      }
    }).catch(() => {
      setExecutionComparison(null)
      setExecutionUnavailable('执行对比暂不可用：当前后端未提供该数据。')
    })
  }, [])

  useEffect(() => { loadExecutionComparison() }, [loadExecutionComparison])
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
  // 后端 active_only 已排除下线与退役版本；页面只渲染当前在线信号。
  const activeShadowEntries = shadowEntries
  const featuredShadowKeys = ['s1_dyn_sq_v1'].filter(k => activeShadowEntries.some(([key]) => key === k))
  const activeShadowRows = analytics ? mergeCurves(analytics.shadow, SHADOW_META) : []

function ShadowSignalDetailModal({
  version,
  onClose,
}: {
  version: string
  onClose: () => void
}) {
  const [data, setData] = useState<ShadowSignalDetailResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<'chart' | 'funnel' | 'buckets' | 'orders'>('orders')

  // 复合筛选器
  const [orderFilter, setOrderFilter] = useState<'ALL' | 'GUARDED' | 'REJECTED' | 'LIVE' | 'WIN' | 'LOSS'>('ALL')
  const [dirFilter, setDirFilter] = useState<'ALL' | 'UP' | 'DOWN'>('ALL')
  const [liveStatusFilter, setLiveStatusFilter] = useState<'ALL' | 'FILLED' | 'FAILED' | 'NONE'>('ALL')
  const [searchKw, setSearchKw] = useState('')
  const [sortField, setSortField] = useState<'time' | 'price' | 'ev' | 'pnl'>('time')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  // 分页
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(25)

  const loadDetail = useCallback(() => {
    setLoading(true)
    setError('')
    api.getShadowSignalDetail(version, 1000)
      .then(res => {
        setData(res)
        setLoading(false)
      })
      .catch(err => {
        setError(`加载失败: ${(err as Error).message}`)
        setLoading(false)
      })
  }, [version])

  useEffect(() => {
    loadDetail()
  }, [loadDetail])

  const s = data?.summary
  const allOrders = data?.orders ?? []

  // 排序与多重筛选
  const filteredOrders = useMemo(() => {
    const rawOrders = data?.orders ?? []
    const kw = searchKw.trim().toLowerCase()
    return rawOrders.filter(o => {
      // 主状态过滤
      if (orderFilter === 'GUARDED' && !o.guarded_pass) return false
      if (orderFilter === 'REJECTED' && o.guarded_pass) return false
      if (orderFilter === 'LIVE' && !o.live_order) return false
      if (orderFilter === 'WIN' && o.win !== true) return false
      if (orderFilter === 'LOSS' && o.win !== false) return false

      // 方向过滤
      if (dirFilter !== 'ALL' && o.direction !== dirFilter) return false

      // 实盘状态过滤
      if (liveStatusFilter === 'FILLED' && o.live_order?.status !== 'FILLED') return false
      if (liveStatusFilter === 'FAILED' && o.live_order?.status !== 'FAILED') return false
      if (liveStatusFilter === 'NONE' && o.live_order != null) return false

      // 关键词搜索
      if (kw) {
        const timeStr = performanceTime(o.window_start).toLowerCase()
        const dirStr = o.direction.toLowerCase()
        const rejectStr = (o.reject_reason ?? '').toLowerCase()
        const orderIdStr = (o.live_order?.order_id ?? '').toLowerCase()
        const errMsg = (o.live_order?.error_message ?? '').toLowerCase()
        if (!timeStr.includes(kw) && !dirStr.includes(kw) && !rejectStr.includes(kw) && !orderIdStr.includes(kw) && !errMsg.includes(kw)) {
          return false
        }
      }

      return true
    }).sort((a, b) => {
      let diff = 0
      if (sortField === 'time') {
        diff = a.window_start - b.window_start
      } else if (sortField === 'price') {
        diff = (a.entry_price ?? -1) - (b.entry_price ?? -1)
      } else if (sortField === 'ev') {
        diff = (a.ev ?? -999) - (b.ev ?? -999)
      } else if (sortField === 'pnl') {
        diff = (a.live_order?.pnl ?? -9999) - (b.live_order?.pnl ?? -9999)
      }
      return sortAsc ? diff : -diff
    })
  }, [data?.orders, orderFilter, dirFilter, liveStatusFilter, searchKw, sortField, sortAsc])

  // 当前筛选下的聚合统计
  const filteredStats = useMemo(() => {
    const total = filteredOrders.length
    if (total === 0) return null
    const wins = filteredOrders.filter(o => o.win === true).length
    const guarded = filteredOrders.filter(o => o.guarded_pass).length
    const guardedWins = filteredOrders.filter(o => o.guarded_pass && o.win === true).length
    const liveFilled = filteredOrders.filter(o => o.live_order?.status === 'FILLED')
    const liveWins = liveFilled.filter(o => o.live_order?.win === true).length
    const livePnlSum = liveFilled.reduce((acc, o) => acc + (o.live_order?.pnl ?? 0), 0)
    const evValid = filteredOrders.filter(o => o.ev != null)
    const evSum = evValid.reduce((acc, o) => acc + (o.ev ?? 0), 0)

    return {
      total,
      winRate: total > 0 ? wins / total : 0,
      guardedPassRate: total > 0 ? guarded / total : 0,
      guardedWinRate: guarded > 0 ? guardedWins / guarded : null,
      avgEv: evValid.length > 0 ? evSum / evValid.length : 0,
      liveFilledCount: liveFilled.length,
      liveWinRate: liveFilled.length > 0 ? liveWins / liveFilled.length : null,
      liveTotalPnl: livePnlSum,
    }
  }, [filteredOrders])

  // 分页计算
  const totalPages = Math.max(1, Math.ceil(filteredOrders.length / pageSize))
  const safePage = Math.min(page, totalPages)
  const pageOrders = filteredOrders.slice((safePage - 1) * pageSize, safePage * pageSize)

  // 常用功能：导出当前筛选结果为 CSV
  const exportCsv = () => {
    if (filteredOrders.length === 0) return
    const headers = [
      '窗口开始时间', '格式化时间', '押注方向', '理论入场价', '理论胜负', '理论实现EV',
      '护栏判定', '弃单原因', '实盘状态', '实盘订单ID', '实盘成交均价', '均价类型',
      '实盘金额USDT', '实盘盈亏USDT', '实盘胜负'
    ]
    const csvRows = [headers.join(',')]
    for (const o of filteredOrders) {
      const lo = o.live_order
      const row = [
        o.window_start,
        `"${performanceTime(o.window_start)}"`,
        o.direction,
        o.entry_price ?? '',
        o.win == null ? '' : o.win ? 'WIN' : 'LOSS',
        o.ev ?? '',
        o.guarded_pass ? 'PASS' : 'REJECT',
        `"${o.reject_reason ?? ''}"`,
        lo?.status ?? 'NONE',
        lo?.order_id ?? '',
        lo?.average_price ?? '',
        lo?.price_kind ?? '',
        lo?.amount_usdt ?? '',
        lo?.pnl ?? '',
        lo?.win == null ? '' : lo.win ? 'WIN' : 'LOSS',
      ]
      csvRows.push(row.join(','))
    }
    const blob = new Blob(['\ufeff' + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `shadow_detail_${version}_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  // 走势与回撤曲线增强数据（Watermark & Drawdown）
  const chartEnhancedData = useMemo(() => {
    if (!data?.curves) return []
    let peakEv = -Infinity
    return data.curves.map(p => {
      const ev = p.cum_ev_guarded
      if (ev > peakEv) peakEv = ev
      const dd = peakEv > -Infinity ? ev - peakEv : 0
      return {
        ...p,
        peak_ev: peakEv,
        drawdown_ev: +dd.toFixed(4),
      }
    })
  }, [data?.curves])

  // 执行漏斗阶段柱状图数据
  const funnelData = useMemo(() => {
    if (!s) return []
    return [
      { stage: '1.理论信号', count: s.total_signals, rate: '100%', fill: 'var(--chart-1)' },
      { stage: '2.护栏通过', count: s.guarded_signals, rate: pct1(s.guarded_pass_rate), fill: 'var(--chart-2)' },
      { stage: '3.实盘派单', count: s.live_orders_count, rate: s.guarded_signals > 0 ? pct1(s.live_orders_count / s.guarded_signals) : '0%', fill: 'var(--chart-3)' },
      { stage: '4.实际成交', count: s.live_filled_count, rate: s.live_orders_count > 0 ? pct1(s.live_filled_count / s.live_orders_count) : '0%', fill: 'var(--positive)' },
      { stage: '5.实盘胜单', count: s.live_wins, rate: s.live_filled_count > 0 ? pct1(s.live_wins / s.live_filled_count) : '0%', fill: 'var(--brand)' },
    ]
  }, [s])

  const info = SIGNAL_INFO[version]
  const displayName = info?.name ?? version

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-overlay p-3 md:p-6" onClick={onClose}>
      <div
        className="bg-card rounded-panel border border-line w-full max-w-6xl max-h-[92vh] flex flex-col shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        {/* 头部导航 */}
        <div className="px-5 py-3 border-b border-line flex items-center justify-between shrink-0 bg-sunken rounded-t-panel">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-bold text-sm text-ink-95">{displayName}</span>
              <span className="font-mono text-xs text-ink-55">{version}</span>
              {data && (
                <>
                  <span className="ds-badge ds-badge-neutral">{signalFamilyLabel(data.family)}</span>
                  {data.guard_price != null && (
                    <span className="ds-badge ds-badge-indigo">执行护栏 ≤ {data.guard_price}</span>
                  )}
                  {data.bench_winrate != null && (
                    <span className="ds-badge ds-badge-violet">回测基准 {pct1(data.bench_winrate)}</span>
                  )}
                </>
              )}
            </div>
            <p className="text-[11px] text-ink-55 mt-1 line-clamp-1">{data?.desc || info?.desc || '暂无说明'}</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={exportCsv}
              disabled={filteredOrders.length === 0}
              className="px-2.5 py-1 text-xs rounded-sm border border-line bg-card hover:bg-sunken text-ink-80 transition disabled:opacity-40 font-medium"
              title="导出当前筛选的订单列表为 CSV"
            >
              📥 导出 CSV
            </button>
            <button
              onClick={loadDetail}
              disabled={loading}
              className="px-2.5 py-1 text-xs rounded-sm border border-line bg-card hover:bg-sunken text-ink-80 transition disabled:opacity-40"
              title="重新获取最新数据"
            >
              {loading ? '刷新中…' : '🔄 刷新'}
            </button>
            <button
              onClick={onClose}
              className="text-ink-55 hover:text-ink-95 text-xl leading-none px-2 py-0.5 rounded-sm hover:bg-sunken"
            >
              ✕
            </button>
          </div>
        </div>

        {/* 核心 KPI 矩阵 */}
        {s && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 px-5 py-2.5 border-b border-line-soft bg-card shrink-0">
            <div className="p-2 rounded-sm bg-sunken border border-line-soft">
              <div className="text-[10px] text-ink-55">信号与通过率</div>
              <div className="font-mono font-bold text-sm text-ink-95">
                {s.guarded_signals} <span className="text-[11px] font-normal text-ink-40">/ {s.total_signals}</span>
              </div>
              <div className="text-[10px] text-ink-55 mt-0.5">
                护栏通过率 <span className="font-mono font-medium text-brand">{pct1(s.guarded_pass_rate)}</span>
              </div>
            </div>

            <div className="p-2 rounded-sm bg-sunken border border-line-soft">
              <div className="text-[10px] text-ink-55">胜率对比（实 vs 理）</div>
              <div className="font-mono font-bold text-sm text-brand">
                {pct1(s.guarded_win_rate)} <span className="text-[11px] font-normal text-ink-40">({s.guarded_wins}胜)</span>
              </div>
              <div className="text-[10px] text-ink-55 mt-0.5">
                全量理论 <span className="font-mono text-ink-80">{pct1(s.theoretical_win_rate)}</span> · 基准 <span className="font-mono">{pct1(data.bench_winrate)}</span>
              </div>
            </div>

            <div className="p-2 rounded-sm bg-sunken border border-line-soft">
              <div className="text-[10px] text-ink-55">理论收益（累计 / 单均）</div>
              <div className={`font-mono font-bold text-sm ${s.guarded_cum_ev >= 0 ? 'text-positive' : 'text-negative'}`}>
                {evFmt(s.guarded_cum_ev)} EV
              </div>
              <div className="text-[10px] text-ink-55 mt-0.5">
                实际单均 <span className="font-mono">{evFmt(s.guarded_avg_ev)}</span> · 理论 <span className="font-mono">{evFmt(s.theoretical_avg_ev)}</span>
              </div>
            </div>

            <div className="p-2 rounded-sm bg-sunken border border-line-soft">
              <div className="text-[10px] text-ink-55">实盘订单与盈亏</div>
              <div className={`font-mono font-bold text-sm ${s.live_total_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                {s.live_total_pnl >= 0 ? '+' : ''}{s.live_total_pnl.toFixed(2)} USDT
              </div>
              <div className="text-[10px] text-ink-55 mt-0.5">
                成交 <span className="font-mono">{s.live_filled_count}</span> 单 · 实盘胜率 <span className="font-mono">{pct1(s.live_win_rate)}</span>
              </div>
            </div>

            <div className="p-2 rounded-sm bg-sunken border border-line-soft">
              <div className="text-[10px] text-ink-55">护栏避损与执行价</div>
              <div className="font-mono font-bold text-sm text-ink-95">
                拦截 {s.rejected_signals} 笔
                <span className={`text-[11px] font-normal ml-1 ${s.avoided_loss_ev >= 0 ? 'text-positive' : 'text-negative'}`}>
                  ({s.avoided_loss_ev >= 0 ? '+' : ''}{s.avoided_loss_ev.toFixed(2)} EV)
                </span>
              </div>
              <div className="text-[10px] text-ink-55 mt-0.5">
                理论均价 <span className="font-mono">{s.avg_quote_price?.toFixed(3) ?? '--'}</span>
                {s.avg_live_fill_price != null && (
                  <> · 实盘 <span className="font-mono">{s.avg_live_fill_price.toFixed(3)}</span></>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 标签栏 */}
        <div className="flex items-center gap-2 px-5 py-2 border-b border-line-soft bg-sunken text-xs shrink-0 flex-wrap">
          <button
            onClick={() => setActiveTab('orders')}
            className={`px-3 py-1 font-semibold rounded-sm transition ${
              activeTab === 'orders' ? 'bg-brand text-white shadow-xs' : 'text-ink-80 hover:text-ink-95 bg-card border border-line-soft'
            }`}
          >
            📋 逐笔订单与实盘对齐 ({allOrders.length})
          </button>
          <button
            onClick={() => setActiveTab('chart')}
            className={`px-3 py-1 font-semibold rounded-sm transition ${
              activeTab === 'chart' ? 'bg-brand text-white shadow-xs' : 'text-ink-80 hover:text-ink-95 bg-card border border-line-soft'
            }`}
          >
            📈 累计收益 & 胜率演进
          </button>
          <button
            onClick={() => setActiveTab('funnel')}
            className={`px-3 py-1 font-semibold rounded-sm transition ${
              activeTab === 'funnel' ? 'bg-brand text-white shadow-xs' : 'text-ink-80 hover:text-ink-95 bg-card border border-line-soft'
            }`}
          >
            🔽 执行漏斗 & 逐笔回报
          </button>
          <button
            onClick={() => setActiveTab('buckets')}
            className={`px-3 py-1 font-semibold rounded-sm transition ${
              activeTab === 'buckets' ? 'bg-brand text-white shadow-xs' : 'text-ink-80 hover:text-ink-95 bg-card border border-line-soft'
            }`}
          >
            📊 报价分档 & 时段归因
          </button>
        </div>

        {/* 主内容区 */}
        <div className="flex-1 min-h-0 overflow-auto p-4">
          {loading && <div className="text-center text-ink-55 py-16 text-xs">加载订单与分析数据中…</div>}
          {error && <div className="text-center text-negative py-16 text-xs">{error}</div>}

          {!loading && !error && data && (
            <>
              {/* Tab 1: 逐笔订单表格（强化筛选+排序+动态统计） */}
              {activeTab === 'orders' && (
                <div className="space-y-3">
                  {/* 复合筛选栏 */}
                  <div className="p-2.5 bg-sunken rounded-sm border border-line-soft flex items-center justify-between gap-2.5 flex-wrap text-xs">
                    <div className="flex items-center gap-2 flex-wrap">
                      <input
                        type="text"
                        placeholder="搜索时间 / 单号 / 状态 / 拒单原因..."
                        value={searchKw}
                        onChange={e => { setSearchKw(e.target.value); setPage(1) }}
                        className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card w-52"
                      />

                      <select
                        value={orderFilter}
                        onChange={e => { setOrderFilter(e.target.value as typeof orderFilter); setPage(1) }}
                        className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card"
                      >
                        <option value="ALL">全部信号状态</option>
                        <option value="GUARDED">仅护栏通过</option>
                        <option value="REJECTED">仅护栏拦截/弃单</option>
                        <option value="LIVE">仅有实盘记录</option>
                        <option value="WIN">仅理论胜单</option>
                        <option value="LOSS">仅理论负单</option>
                      </select>

                      <select
                        value={dirFilter}
                        onChange={e => { setDirFilter(e.target.value as typeof dirFilter); setPage(1) }}
                        className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card"
                      >
                        <option value="ALL">全部方向</option>
                        <option value="DOWN">仅押 DOWN</option>
                        <option value="UP">仅押 UP</option>
                      </select>

                      <select
                        value={liveStatusFilter}
                        onChange={e => { setLiveStatusFilter(e.target.value as typeof liveStatusFilter); setPage(1) }}
                        className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card"
                      >
                        <option value="ALL">全部实盘状态</option>
                        <option value="FILLED">仅实盘已成交</option>
                        <option value="FAILED">仅实盘失败/被拒</option>
                        <option value="NONE">仅未下实盘单</option>
                      </select>

                      <div className="flex items-center gap-1 border-l border-line-soft pl-2">
                        <span className="text-ink-55">排序:</span>
                        <select
                          value={sortField}
                          onChange={e => setSortField(e.target.value as typeof sortField)}
                          className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card"
                        >
                          <option value="time">按时间</option>
                          <option value="price">按入场价</option>
                          <option value="ev">按理论EV</option>
                          <option value="pnl">按实盘盈亏</option>
                        </select>
                        <button
                          onClick={() => setSortAsc(!sortAsc)}
                          className="px-1.5 py-0.5 rounded-sm border border-line bg-card text-ink-80 hover:text-brand"
                        >
                          {sortAsc ? '▲ 升序' : '▼ 降序'}
                        </button>
                      </div>

                      {(searchKw || orderFilter !== 'ALL' || dirFilter !== 'ALL' || liveStatusFilter !== 'ALL') && (
                        <button
                          onClick={() => {
                            setSearchKw('')
                            setOrderFilter('ALL')
                            setDirFilter('ALL')
                            setLiveStatusFilter('ALL')
                            setPage(1)
                          }}
                          className="text-brand hover:underline px-1"
                        >
                          重置筛选
                        </button>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-ink-55">每页</span>
                      <select
                        value={pageSize}
                        onChange={e => { setPageSize(Number(e.target.value)); setPage(1) }}
                        className="px-2 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card"
                      >
                        <option value={15}>15 笔</option>
                        <option value={25}>25 笔</option>
                        <option value={50}>50 笔</option>
                        <option value={100}>100 笔</option>
                      </select>
                    </div>
                  </div>

                  {/* 动态筛选汇总条 */}
                  {filteredStats && (
                    <div className="px-3 py-1.5 bg-brand-soft/40 border border-brand/20 rounded-sm text-[11px] text-ink-80 flex items-center gap-4 flex-wrap">
                      <span>当前视图: <strong className="font-mono text-ink-95">{filteredStats.total}</strong> 笔</span>
                      <span>理论胜率: <strong className="font-mono text-brand">{pct1(filteredStats.winRate)}</strong></span>
                      <span>单均 EV: <strong className="font-mono">{evFmt(filteredStats.avgEv)}</strong></span>
                      <span>护栏通过率: <strong className="font-mono">{pct1(filteredStats.guardedPassRate)}</strong></span>
                      <span>实盘成交: <strong className="font-mono">{filteredStats.liveFilledCount}</strong> 笔</span>
                      <span>实盘胜率: <strong className="font-mono">{pct1(filteredStats.liveWinRate)}</strong></span>
                      <span>实盘盈亏: <strong className={`font-mono font-bold ${filteredStats.liveTotalPnl >= 0 ? 'text-positive' : 'text-negative'}`}>{filteredStats.liveTotalPnl >= 0 ? '+' : ''}{filteredStats.liveTotalPnl.toFixed(2)}U</strong></span>
                    </div>
                  )}

                  {/* 逐笔明细对齐表 */}
                  <div className="overflow-x-auto border border-line rounded-sm">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-line bg-sunken text-ink-55">
                          <th className="py-1.5 px-2 text-left">时间 (UTC / 本地)</th>
                          <th className="py-1.5 px-2 text-center">方向</th>
                          <th className="py-1.5 px-2 text-right">理论入场价</th>
                          <th className="py-1.5 px-2 text-center">理论胜负</th>
                          <th className="py-1.5 px-2 text-right">理论单笔EV</th>
                          <th className="py-1.5 px-2 text-center">护栏判定</th>
                          <th className="py-1.5 px-2 text-center">实盘订单</th>
                          <th className="py-1.5 px-2 text-right">实盘成交价</th>
                          <th className="py-1.5 px-2 text-right">实盘投入</th>
                          <th className="py-1.5 px-2 text-right">实盘盈亏</th>
                          <th className="py-1.5 px-2 text-center">实盘胜负</th>
                        </tr>
                      </thead>
                      <tbody>
                        {pageOrders.length === 0 ? (
                          <tr>
                            <td colSpan={11} className="py-8 text-center text-ink-55">
                              当前筛选条件下无匹配订单
                            </td>
                          </tr>
                        ) : (
                          pageOrders.map((o, idx) => {
                            const won = o.win === true
                            const lo = o.live_order
                            const liveWon = lo?.win === true
                            const liveLost = lo?.win === false
                            return (
                              <tr key={idx} className="border-b border-line-soft hover:bg-sunken transition-colors">
                                <td className="py-1.5 px-2 font-mono text-ink-80 whitespace-nowrap">
                                  {performanceTime(o.window_start)}
                                </td>
                                <td className="py-1.5 px-2 text-center">
                                  <span className={`px-1.5 py-0.2 rounded-xs font-bold text-[10px] ${
                                    o.direction === 'UP' ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'
                                  }`}>
                                    {o.direction}
                                  </span>
                                </td>
                                <td className="py-1.5 px-2 text-right font-mono text-ink-95">
                                  {o.entry_price != null ? o.entry_price.toFixed(3) : '--'}
                                </td>
                                <td className="py-1.5 px-2 text-center">
                                  {o.win == null ? (
                                    <span className="text-ink-40">未结算</span>
                                  ) : won ? (
                                    <span className="text-positive font-bold">胜</span>
                                  ) : (
                                    <span className="text-negative font-bold">负</span>
                                  )}
                                </td>
                                <td className={`py-1.5 px-2 text-right font-mono font-medium ${
                                  o.ev == null ? 'text-ink-40' : o.ev >= 0 ? 'text-positive' : 'text-negative'
                                }`}>
                                  {evFmt(o.ev)}
                                </td>
                                <td className="py-1.5 px-2 text-center">
                                  {o.guarded_pass ? (
                                    <span className="px-1.5 py-0.2 text-[10px] font-medium text-brand bg-brand-soft rounded-pill" title="通过护栏与进场白名单">
                                      通过
                                    </span>
                                  ) : (
                                    <span className="px-1.5 py-0.2 text-[10px] font-medium text-negative bg-negative-soft rounded-pill" title={o.reject_reason ?? '弃单'}>
                                      {o.reject_reason ?? '弃单'}
                                    </span>
                                  )}
                                </td>
                                <td className="py-1.5 px-2 text-center whitespace-nowrap">
                                  {lo ? (
                                    <span className={`px-1.5 py-0.2 text-[10px] font-medium rounded-pill ${
                                      lo.status === 'FILLED'
                                        ? 'bg-positive-soft text-positive border border-positive/30'
                                        : lo.status === 'FAILED'
                                        ? 'bg-negative-soft text-negative border border-negative/30'
                                        : 'bg-warning-soft text-warning'
                                    }`} title={lo.error_message ? `错误: ${lo.error_message}` : `订单: ${lo.order_id ?? lo.id}`}>
                                      {lo.status === 'FILLED' ? '已成交' : lo.status === 'FAILED' ? '失败/拒单' : lo.status}
                                    </span>
                                  ) : (
                                    <span className="text-ink-40 text-[10px]">未下单</span>
                                  )}
                                </td>
                                <td className="py-1.5 px-2 text-right font-mono">
                                  {lo?.average_price != null ? (
                                    <span className="inline-flex items-center gap-0.5 justify-end">
                                      {lo.average_price.toFixed(3)}
                                      {lo.price_kind === 'quote' && (
                                        <span className="text-[9px] px-0.5 py-0.2 rounded-xs bg-warning-soft text-warning" title="报价/委托价">
                                          报
                                        </span>
                                      )}
                                    </span>
                                  ) : '--'}
                                </td>
                                <td className="py-1.5 px-2 text-right font-mono text-ink-80">
                                  {lo ? `${lo.amount_usdt.toFixed(1)}U` : '--'}
                                </td>
                                <td className={`py-1.5 px-2 text-right font-mono font-bold ${
                                  lo?.pnl == null ? 'text-ink-40' : lo.pnl >= 0 ? 'text-positive' : 'text-negative'
                                }`}>
                                  {lo?.pnl != null ? `${lo.pnl >= 0 ? '+' : ''}${lo.pnl.toFixed(2)}U` : '--'}
                                </td>
                                <td className="py-1.5 px-2 text-center">
                                  {lo == null ? (
                                    <span className="text-ink-40">--</span>
                                  ) : liveWon ? (
                                    <span className="text-positive font-bold">胜</span>
                                  ) : liveLost ? (
                                    <span className="text-negative font-bold">负</span>
                                  ) : (
                                    <span className="text-ink-40">{lo.status === 'FAILED' ? '拒' : '--'}</span>
                                  )}
                                </td>
                              </tr>
                            )
                          })
                        )}
                      </tbody>
                    </table>
                  </div>

                  {/* 分页控制 */}
                  {totalPages > 1 && (
                    <div className="flex items-center justify-between gap-2 mt-2 text-xs">
                      <span className="text-ink-55">
                        共 {filteredOrders.length} 笔 · 第 {safePage} / {totalPages} 页
                      </span>
                      <div className="flex items-center gap-1">
                        <button
                          disabled={safePage <= 1}
                          onClick={() => setPage(1)}
                          className="px-2 py-0.5 border border-line rounded-sm disabled:opacity-40 hover:border-brand"
                        >
                          首页
                        </button>
                        <button
                          disabled={safePage <= 1}
                          onClick={() => setPage(safePage - 1)}
                          className="px-2 py-0.5 border border-line rounded-sm disabled:opacity-40 hover:border-brand"
                        >
                          上一页
                        </button>
                        <span className="px-2 font-mono">
                          {safePage} / {totalPages}
                        </span>
                        <button
                          disabled={safePage >= totalPages}
                          onClick={() => setPage(safePage + 1)}
                          className="px-2 py-0.5 border border-line rounded-sm disabled:opacity-40 hover:border-brand"
                        >
                          下一页
                        </button>
                        <button
                          disabled={safePage >= totalPages}
                          onClick={() => setPage(totalPages)}
                          className="px-2 py-0.5 border border-line rounded-sm disabled:opacity-40 hover:border-brand"
                        >
                          末页
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: 走势曲线与回撤走势（增强版图表） */}
              {activeTab === 'chart' && (
                <div className="space-y-4">
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">累计收益与实盘盈亏走势 (Cumulative EV vs Real PnL)</span>
                      <span className="text-[11px] text-ink-55">左轴=EV（理论/带护栏） · 右轴=USDT（实盘真实净盈亏）</span>
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <ComposedChart data={chartEnhancedData}>
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis
                          dataKey="ts"
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          scale="time"
                          tickFormatter={performanceTime}
                          tick={{ fontSize: 10, fill: 'var(--ink-55)' }}
                        />
                        <YAxis yAxisId="ev" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <YAxis yAxisId="pnl" orientation="right" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} tickFormatter={v => `${v}U`} />
                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          labelStyle={TOOLTIP_LABEL_STYLE}
                          itemStyle={TOOLTIP_ITEM_STYLE}
                          labelFormatter={v => performanceTime(Number(v))}
                        />
                        <Legend />
                        <ReferenceLine yAxisId="ev" y={0} stroke="var(--line)" strokeDasharray="3 3" />
                        <Line
                          yAxisId="ev"
                          name="带护栏累计EV"
                          dataKey="cum_ev_guarded"
                          stroke="var(--brand)"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                        <Line
                          yAxisId="ev"
                          name="全量理论累计EV"
                          dataKey="cum_ev_theoretical"
                          stroke="var(--chart-1)"
                          strokeWidth={1.5}
                          strokeDasharray="4 2"
                          dot={false}
                          connectNulls
                        />
                        {s && s.live_orders_count > 0 && (
                          <Line
                            yAxisId="pnl"
                            name="实盘累计盈亏(USDT)"
                            dataKey="cum_pnl_live"
                            stroke="var(--positive)"
                            strokeWidth={2}
                            dot={false}
                            connectNulls
                          />
                        )}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  {/* 胜率演进折线图 */}
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">前向胜率演进 (Rolling Win Rate)</span>
                      <span className="text-[11px] text-ink-55">橙色虚线=冻结回测基准胜率</span>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={chartEnhancedData}>
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis
                          dataKey="ts"
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          scale="time"
                          tickFormatter={performanceTime}
                          tick={{ fontSize: 10, fill: 'var(--ink-55)' }}
                        />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          labelStyle={TOOLTIP_LABEL_STYLE}
                          itemStyle={TOOLTIP_ITEM_STYLE}
                          labelFormatter={v => performanceTime(Number(v))}
                          formatter={v => [typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '--', '']}
                        />
                        <Legend />
                        {data.bench_winrate != null && (
                          <ReferenceLine y={data.bench_winrate} stroke="var(--warning)" strokeDasharray="5 3" label="基准胜率" />
                        )}
                        <Line
                          name="带护栏实际胜率"
                          dataKey="win_rate_guarded"
                          stroke="var(--brand)"
                          strokeWidth={2}
                          dot={false}
                          connectNulls
                        />
                        <Line
                          name="全量理论胜率"
                          dataKey="win_rate_theoretical"
                          stroke="var(--ink-40)"
                          strokeWidth={1.5}
                          dot={false}
                          connectNulls
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* 回撤深度面积图 */}
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">累计 EV 回撤深度 (Drawdown Under Water)</span>
                      <span className="text-[11px] text-ink-55">展示该信号历史回撤幅度与持续时间</span>
                    </div>
                    <ResponsiveContainer width="100%" height={180}>
                      <AreaChart data={chartEnhancedData}>
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis
                          dataKey="ts"
                          type="number"
                          domain={['dataMin', 'dataMax']}
                          scale="time"
                          tickFormatter={performanceTime}
                          tick={{ fontSize: 10, fill: 'var(--ink-55)' }}
                        />
                        <YAxis tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          labelStyle={TOOLTIP_LABEL_STYLE}
                          itemStyle={TOOLTIP_ITEM_STYLE}
                          labelFormatter={v => performanceTime(Number(v))}
                          formatter={v => [`${v} EV`, '回撤深度']}
                        />
                        <Area
                          type="monotone"
                          dataKey="drawdown_ev"
                          name="EV回撤"
                          stroke="var(--negative)"
                          fill="var(--negative-soft)"
                          fillOpacity={0.6}
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Tab 3: 执行漏斗与逐笔回报分布 */}
              {activeTab === 'funnel' && (
                <div className="space-y-4">
                  {/* 执行转化漏斗 */}
                  <div className="border border-line rounded-sm p-4 bg-card">
                    <div className="text-xs font-bold text-ink-95 mb-1">
                      执行转化漏斗 (Execution Conversion Funnel)
                    </div>
                    <p className="text-[11px] text-ink-55 mb-3">
                      量化从理论信号触发到最终实盘成交与获利的每一层损耗率，定位执行瓶颈。
                    </p>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={funnelData} layout="vertical" margin={{ left: 20, right: 30, top: 10, bottom: 10 }}>
                        <CartesianGrid stroke="var(--line-soft)" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <YAxis dataKey="stage" type="category" tick={{ fontSize: 11, fill: 'var(--ink-95)' }} width={80} />
                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          labelStyle={TOOLTIP_LABEL_STYLE}
                          itemStyle={TOOLTIP_ITEM_STYLE}
                          formatter={(v, _, item) => [`${v} 笔 (转化率: ${(item?.payload as { rate?: string })?.rate ?? '--'})`, '数量']}
                        />
                        <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                          {funnelData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* 逐笔理论回报与实盘盈亏柱图 */}
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">逐笔收益柱状图 (Per-Trade Return Sequence)</span>
                      <span className="text-[11px] text-ink-55">近 60 笔信号理论 EV 与实盘盈亏 (USDT)</span>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart
                        data={allOrders.slice(-60).map((o: ShadowDetailOrderItem, idx: number) => ({
                          idx: idx + 1,
                          time: performanceTime(o.window_start),
                          ev: o.ev ?? 0,
                          pnl: o.live_order?.pnl ?? 0,
                        }))}
                        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis dataKey="idx" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <Tooltip
                          contentStyle={TOOLTIP_STYLE}
                          labelStyle={TOOLTIP_LABEL_STYLE}
                          itemStyle={TOOLTIP_ITEM_STYLE}
                          labelFormatter={(idx, list) => {
                            const row = list[0]?.payload as { time?: string } | undefined
                            return `第 ${idx} 笔 · ${row?.time ?? ''}`
                          }}
                          formatter={(v, n) => [typeof v === 'number' ? `${v.toFixed(3)} ${n === 'ev' ? 'EV' : 'U'}` : '--', n === 'ev' ? '理论EV' : '实盘PnL']}
                        />
                        <ReferenceLine y={0} stroke="var(--line)" />
                        <Bar dataKey="ev" name="理论EV" fill="var(--chart-1)" radius={[2, 2, 0, 0]}>
                          {allOrders.slice(-60).map((entry: ShadowDetailOrderItem, index: number) => (
                            <Cell key={`cell-${index}`} fill={(entry.ev ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Tab 4: 报价分桶与时段归因 */}
              {activeTab === 'buckets' && (
                <div className="space-y-4">
                  {/* 报价区间柱状图对比 */}
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">报价分档信号量与胜率分布 (Quote Buckets)</span>
                      <span className="text-[11px] text-ink-55">柱高=信号数 · 折线=胜率</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <ComposedChart data={data.quote_buckets}>
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis dataKey="range" tick={{ fontSize: 10, fill: 'var(--ink-80)' }} />
                        <YAxis yAxisId="n" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <YAxis yAxisId="wr" orientation="right" domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} />
                        <Legend />
                        <Bar yAxisId="n" name="信号数" dataKey="signals" fill="var(--chart-1)" radius={[3, 3, 0, 0]} />
                        <Bar yAxisId="n" name="实盘单" dataKey="live_orders" fill="var(--positive)" radius={[3, 3, 0, 0]} />
                        <Line yAxisId="wr" name="理论胜率" dataKey="win_rate" stroke="var(--brand)" strokeWidth={2} dot={{ r: 3 }} />
                        <Line yAxisId="wr" name="实盘胜率" dataKey="live_win_rate" stroke="var(--warning)" strokeWidth={2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>

                    <div className="overflow-x-auto mt-3">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-line text-ink-55 bg-sunken">
                            <th className="py-1 px-1.5 text-left">报价档位</th>
                            <th className="py-1 px-1.5 text-right">信号数</th>
                            <th className="py-1 px-1.5 text-right">理论胜率</th>
                            <th className="py-1 px-1.5 text-right">单均EV</th>
                            <th className="py-1 px-1.5 text-right">实盘单数</th>
                            <th className="py-1 px-1.5 text-right">实盘胜率</th>
                            <th className="py-1 px-1.5 text-right">实盘盈亏</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.quote_buckets.map((b, i) => (
                            <tr key={i} className="border-b border-line-soft hover:bg-sunken">
                              <td className="py-1 px-1.5 font-mono text-ink-95 font-medium">{b.range}</td>
                              <td className="py-1 px-1.5 text-right font-mono text-ink-80">{b.signals}</td>
                              <td className="py-1 px-1.5 text-right font-mono font-medium text-brand">
                                {pct1(b.win_rate)}
                              </td>
                              <td className={`py-1 px-1.5 text-right font-mono ${
                                b.avg_ev == null ? 'text-ink-40' : b.avg_ev >= 0 ? 'text-positive' : 'text-negative'
                              }`}>
                                {evFmt(b.avg_ev)}
                              </td>
                              <td className="py-1 px-1.5 text-right font-mono text-ink-80">{b.live_orders || '--'}</td>
                              <td className="py-1 px-1.5 text-right font-mono font-medium text-brand">{pct1(b.live_win_rate)}</td>
                              <td className={`py-1 px-1.5 text-right font-mono font-bold ${
                                b.live_pnl == null ? 'text-ink-40' : b.live_pnl >= 0 ? 'text-positive' : 'text-negative'
                              }`}>
                                {b.live_pnl != null ? `${b.live_pnl >= 0 ? '+' : ''}${b.live_pnl.toFixed(1)}U` : '--'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* 时段分布图表对比 */}
                  <div className="border border-line rounded-sm p-3 bg-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-bold text-ink-95">UTC 时段信号分布与胜率 (Hour Buckets)</span>
                      <span className="text-[11px] text-ink-55">柱高=信号数 · 折线=胜率</span>
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <ComposedChart data={data.hour_buckets}>
                        <CartesianGrid stroke="var(--line-soft)" />
                        <XAxis dataKey="hour_range" tick={{ fontSize: 10, fill: 'var(--ink-80)' }} />
                        <YAxis yAxisId="n" tick={{ fontSize: 10, fill: 'var(--ink-55)' }} />
                        <YAxis yAxisId="wr" orientation="right" domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--ink-55)' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE} />
                        <Legend />
                        <Bar yAxisId="n" name="信号数" dataKey="signals" fill="var(--chart-2)" radius={[3, 3, 0, 0]} />
                        <Bar yAxisId="n" name="实盘单" dataKey="live_orders" fill="var(--positive)" radius={[3, 3, 0, 0]} />
                        <Line yAxisId="wr" name="理论胜率" dataKey="win_rate" stroke="var(--brand)" strokeWidth={2} dot={{ r: 3 }} />
                        <Line yAxisId="wr" name="实盘胜率" dataKey="live_win_rate" stroke="var(--warning)" strokeWidth={2} dot={{ r: 3 }} />
                      </ComposedChart>
                    </ResponsiveContainer>

                    <div className="overflow-x-auto mt-3">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-line text-ink-55 bg-sunken">
                            <th className="py-1 px-1.5 text-left">UTC 时段</th>
                            <th className="py-1 px-1.5 text-right">信号数</th>
                            <th className="py-1 px-1.5 text-right">理论胜率</th>
                            <th className="py-1 px-1.5 text-right">单均EV</th>
                            <th className="py-1 px-1.5 text-right">实盘单数</th>
                            <th className="py-1 px-1.5 text-right">实盘胜率</th>
                            <th className="py-1 px-1.5 text-right">实盘盈亏</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.hour_buckets.map((b, i) => (
                            <tr key={i} className="border-b border-line-soft hover:bg-sunken">
                              <td className="py-1 px-1.5 font-mono text-ink-95 font-medium">{b.hour_range}</td>
                              <td className="py-1 px-1.5 text-right font-mono text-ink-80">{b.signals}</td>
                              <td className="py-1 px-1.5 text-right font-mono font-medium text-brand">
                                {pct1(b.win_rate)}
                              </td>
                              <td className={`py-1 px-1.5 text-right font-mono ${
                                b.avg_ev == null ? 'text-ink-40' : b.avg_ev >= 0 ? 'text-positive' : 'text-negative'
                              }`}>
                                {evFmt(b.avg_ev)}
                              </td>
                              <td className="py-1 px-1.5 text-right font-mono text-ink-80">{b.live_orders || '--'}</td>
                              <td className="py-1 px-1.5 text-right font-mono font-medium text-brand">{pct1(b.live_win_rate)}</td>
                              <td className={`py-1 px-1.5 text-right font-mono font-bold ${
                                b.live_pnl == null ? 'text-ink-40' : b.live_pnl >= 0 ? 'text-positive' : 'text-negative'
                              }`}>
                                {b.live_pnl != null ? `${b.live_pnl >= 0 ? '+' : ''}${b.live_pnl.toFixed(1)}U` : '--'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function ShadowSignalCard({
  title,
  entries,
  rows,
  footnote,
  emptyHint,
  analytics,
  load,
  pumpTs,
  isRetired,
}: {
  title: string
  entries: [string, { label: string; color: string }][]
  rows: Record<string, number>[]
  footnote: React.ReactNode
  emptyHint?: string
  analytics: SignalsAnalytics
  load: () => void
  pumpTs: number | null
  isRetired?: boolean
}) {
  const [kw, setKw] = useState('')
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'ONLINE' | 'OFFLINE' | 'LIVE'>('ALL')
  const [familyFilter, setFamilyFilter] = useState('ALL')
  const [roleFilter, setRoleFilter] = useState('ALL')
  const [guardMode, setGuardMode] = useState<'GUARDED' | 'THEORETICAL' | 'BOTH'>('GUARDED')
  const [sortCol, setSortCol] = useState<string>('default')
  const [sortAsc, setSortAsc] = useState<boolean>(false)
  const [detailVersion, setDetailVersion] = useState<string | null>(null)

  if (entries.length === 0) {
    return (
      <Card title={title}>
        <div className="text-center text-ink-55 py-6 text-xs">{emptyHint ?? '无数据'}</div>
      </Card>
    )
  }

  const toggleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(false)
    }
  }

  const thSort = (field: string, label: string, align: 'left' | 'center' | 'right' = 'right', hint?: string) => (
    <th
      className={`py-1 px-2 text-${align} cursor-pointer select-none hover:text-brand transition-colors`}
      onClick={() => toggleSort(field)}
      title={hint}
    >
      <span className={`inline-flex items-center gap-0.5 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start'}`}>
        {label}
        {sortCol === field ? (
          <span className="text-brand font-mono text-[10px]">{sortAsc ? '▲' : '▼'}</span>
        ) : (
          <span className="text-ink-40 font-mono text-[10px]">⇅</span>
        )}
      </span>
    </th>
  )

  const search = kw.trim().toLowerCase()
  const filtered = entries.filter(([k, m]) => {
    const s = analytics.shadow[k]?.summary
    if (!s) return false
    if (search) {
      const matchKey = k.toLowerCase().includes(search)
      const matchLabel = m.label.toLowerCase().includes(search)
      const matchDesc = String(s.desc ?? '').toLowerCase().includes(search)
      const matchExtra = (signalDescFor('shadow', k) ?? '').toLowerCase().includes(search)
      if (!matchKey && !matchLabel && !matchDesc && !matchExtra) return false
    }
    const online = s.enabled !== false
    const retired = s.retired === true
    const live = retired ? null : (s.live_channel ?? null)
    if (familyFilter !== 'ALL' && (s.family ?? 'legacy') !== familyFilter) return false
    if (roleFilter !== 'ALL' && (s.role ?? 'STRATEGY') !== roleFilter) return false
    if (statusFilter === 'ONLINE') {
      if (retired || !online) return false
    } else if (statusFilter === 'OFFLINE') {
      if (retired || online) return false
    } else if (statusFilter === 'LIVE') {
      if (retired || !live || !live.enabled) return false
    }
    return true
  })

  const sorted = [...filtered].sort(([ka, ma], [kb, mb]) => {
    if (sortCol === 'default') return 0
    const sa = analytics.shadow[ka]?.summary
    const sb = analytics.shadow[kb]?.summary
    if (!sa || !sb) return 0
    let diff = 0
    if (sortCol === 'ver') {
      diff = ma.label.localeCompare(mb.label, 'zh-CN')
    } else if (sortCol === 'status') {
      const valA = sa.retired ? -1 : sa.enabled !== false ? 1 : 0
      const valB = sb.retired ? -1 : sb.enabled !== false ? 1 : 0
      diff = valA - valB
    } else if (sortCol === 'live') {
      const valA = sa.retired ? 0 : sa.live_channel?.enabled ? 2 : sa.live_channel ? 1 : 0
      const valB = sb.retired ? 0 : sb.live_channel?.enabled ? 2 : sb.live_channel ? 1 : 0
      diff = valA - valB
    } else if (sortCol === 'n') {
      const na = guardMode === 'GUARDED' ? (sa.guarded_n ?? 0) : (sa.n ?? 0)
      const nb = guardMode === 'GUARDED' ? (sb.guarded_n ?? 0) : (sb.n ?? 0)
      diff = na - nb
    } else if (sortCol === 'win_rate') {
      const wra = guardMode === 'GUARDED' ? (sa.guarded_win_rate ?? -1) : (sa.win_rate ?? -1)
      const wrb = guardMode === 'GUARDED' ? (sb.guarded_win_rate ?? -1) : (sb.win_rate ?? -1)
      diff = wra - wrb
    } else if (sortCol === 'breakeven') {
      const bea = guardMode === 'GUARDED' ? (sa.guarded_avg_breakeven ?? -1) : (sa.avg_breakeven ?? -1)
      const beb = guardMode === 'GUARDED' ? (sb.guarded_avg_breakeven ?? -1) : (sb.avg_breakeven ?? -1)
      diff = bea - beb
    } else if (sortCol === 'bench') {
      diff = (sa.bench_winrate ?? -1) - (sb.bench_winrate ?? -1)
    } else if (sortCol === 'dev') {
      const activeWra = guardMode === 'GUARDED' ? sa.guarded_win_rate : sa.win_rate
      const activeWrb = guardMode === 'GUARDED' ? sb.guarded_win_rate : sb.win_rate
      const devA = activeWra != null && sa.bench_winrate != null ? activeWra - sa.bench_winrate : -999
      const devB = activeWrb != null && sb.bench_winrate != null ? activeWrb - sb.bench_winrate : -999
      diff = devA - devB
    } else if (sortCol === 'avg_ev') {
      const eva = guardMode === 'GUARDED' ? (sa.guarded_avg_ev ?? -999) : (sa.avg_ev ?? -999)
      const evb = guardMode === 'GUARDED' ? (sb.guarded_avg_ev ?? -999) : (sb.avg_ev ?? -999)
      diff = eva - evb
    } else if (sortCol === 'cum_ev') {
      const ceva = guardMode === 'GUARDED' ? (sa.guarded_cum_ev ?? -999) : (sa.cum_ev ?? -999)
      const cevb = guardMode === 'GUARDED' ? (sb.guarded_cum_ev ?? -999) : (sb.cum_ev ?? -999)
      diff = ceva - cevb
    } else if (sortCol === 'fill_rate') {
      diff = (sa.guarded_fill_rate ?? -1) - (sb.guarded_fill_rate ?? -1)
    }
    return sortAsc ? diff : -diff
  })

  const familyOptions = Array.from(new Set(entries.map(([k]) => analytics.shadow[k]?.summary.family ?? 'legacy')))
    .sort((a, b) => familyRank(a) - familyRank(b))
  const filterActive = search !== '' || statusFilter !== 'ALL' || familyFilter !== 'ALL' || roleFilter !== 'ALL'
  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'

  return (
    <Card title={title}>
      {/* 筛选与搜索栏 */}
      <div className="flex items-center justify-between mb-3 gap-2 flex-wrap text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type="text"
            placeholder="搜索版本/说明/代号..."
            value={kw}
            onChange={e => setKw(e.target.value)}
            className={`${selectCls} w-44`}
          />
          <select value={familyFilter} onChange={e => setFamilyFilter(e.target.value)} className={selectCls}>
            <option value="ALL">全部策略族</option>
            {familyOptions.map(family => <option key={family} value={family}>{signalFamilyLabel(family)}</option>)}
          </select>
          <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)} className={selectCls}>
            <option value="ALL">全部角色</option>
            {Object.entries(ROLE_LABELS).map(([role, label]) => <option key={role} value={role}>{label}</option>)}
          </select>
          {!isRetired && (
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value as 'ALL' | 'ONLINE' | 'OFFLINE' | 'LIVE')}
              className={selectCls}
            >
              <option value="ALL">全部在线信号</option>
              <option value="ONLINE">仅在线采集</option>
              <option value="LIVE">仅实盘中</option>
            </select>
          )}
          <div className="inline-flex items-center rounded-sm border border-line bg-sunken p-0.5 text-xs">
            <button
              onClick={() => setGuardMode('GUARDED')}
              className={`px-2 py-0.5 rounded-xs transition-colors font-medium ${
                guardMode === 'GUARDED' ? 'bg-card text-brand shadow-xs' : 'text-ink-55 hover:text-ink-80'
              }`}
              title="仅统计通过实盘执行价护栏及白名单的实际可成交信号"
            >
              带护栏实际
            </button>
            <button
              onClick={() => setGuardMode('THEORETICAL')}
              className={`px-2 py-0.5 rounded-xs transition-colors font-medium ${
                guardMode === 'THEORETICAL' ? 'bg-card text-brand shadow-xs' : 'text-ink-55 hover:text-ink-80'
              }`}
              title="原始全量影子信号（未做任何护栏拦截）"
            >
              全量理论
            </button>
            <button
              onClick={() => setGuardMode('BOTH')}
              className={`px-2 py-0.5 rounded-xs transition-colors font-medium ${
                guardMode === 'BOTH' ? 'bg-card text-brand shadow-xs' : 'text-ink-55 hover:text-ink-80'
              }`}
              title="同时对比理论与带护栏实际数据"
            >
              双口径对比
            </button>
          </div>
          {filterActive && (
            <button
              onClick={() => { setKw(''); setStatusFilter('ALL'); setFamilyFilter('ALL'); setRoleFilter('ALL') }}
              className="px-1.5 py-0.5 text-xs text-brand hover:underline"
            >清除筛选</button>
          )}
          {sortCol !== 'default' && (
            <button
              onClick={() => { setSortCol('default'); setSortAsc(false) }}
              className="px-1.5 py-0.5 text-xs text-ink-55 hover:underline"
            >重置排序</button>
          )}
        </div>
        <span className="text-xs text-ink-55">
          显示 {sorted.length} / 共 {entries.length} 个版本
        </span>
      </div>

      {sorted.length === 0 ? (
        <div className="text-center text-ink-55 py-6 text-xs">当前筛选条件下无匹配版本</div>
      ) : (
        <>
          <div className="space-y-4 mb-3">
          {Array.from(new Set(sorted.map(([k]) => analytics.shadow[k].summary.family ?? 'legacy')))
            .sort((a, b) => familyRank(a) - familyRank(b))
            .map(family => {
              const familyRows = sorted.filter(([k]) => (analytics.shadow[k].summary.family ?? 'legacy') === family)
              return <div key={family} className="overflow-x-auto border border-line rounded-sm">
                <div className="px-3 py-2 bg-sunken text-xs font-bold text-ink-95 flex items-center justify-between">
                  <span>{signalFamilyLabel(family)}</span><span className="font-mono text-ink-55">{familyRows.length} 版</span>
                </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-line text-ink-55">
                  {thSort('ver', '版本 / 分类', 'left')}
                  {thSort('status', '影子采集', 'center')}
                  {thSort('live', '实盘', 'center')}
                  <th className="py-1 px-2 text-center" title="查看单信号逐笔订单与实盘对齐对比分析">订单/对比</th>
                  <th className="py-1 px-2 text-right" title="实盘执行价护栏（超价或贴线弃单）">护栏/弃单</th>
                  {guardMode === 'BOTH' ? (
                    <>
                      {thSort('n', '实际n / 理论n', 'right')}
                      {thSort('fill_rate', '成交率', 'right', '带护栏实际成交信号占理论信号的比例')}
                      {thSort('win_rate', '带护栏实际胜率', 'right')}
                      <th className="py-1 px-2 text-right text-ink-40" title="未做护栏过滤的全量影子理论胜率">理论胜率</th>
                      {thSort('breakeven', '实际保本率', 'right', '带护栏成交信号前向真实入场报价对应的平均保本胜率')}
                      <th className="py-1 px-2 text-right" title="冻结回测胜率×0.98：费后理论允许的最高入场价">回测保本价</th>
                      {thSort('bench', '回测胜率', 'right')}
                      {thSort('dev', '偏离(实-基)', 'right')}
                      {thSort('avg_ev', '实际单均EV', 'right')}
                      <th className="py-1 px-2 text-right text-ink-40" title="理论单均EV">理论EV</th>
                      {thSort('cum_ev', '实际累计EV', 'right')}
                      <th className="py-1 px-2 text-right text-ink-40" title="理论累计EV">理论累计</th>
                    </>
                  ) : guardMode === 'GUARDED' ? (
                    <>
                      {thSort('n', '实际n', 'right')}
                      {thSort('fill_rate', '成交率', 'right', '带护栏实际成交信号占理论信号的比例')}
                      {thSort('win_rate', '实际胜率', 'right')}
                      {thSort('breakeven', '实际保本率', 'right', '带护栏成交信号前向真实入场报价对应的平均保本胜率')}
                      <th className="py-1 px-2 text-right" title="冻结回测胜率×0.98：费后理论允许的最高入场价">回测保本价</th>
                      {thSort('bench', '回测胜率', 'right')}
                      {thSort('dev', '偏离', 'right')}
                      {thSort('avg_ev', '实际单均EV', 'right')}
                      {thSort('cum_ev', '实际累计EV', 'right')}
                    </>
                  ) : (
                    <>
                      {thSort('n', '理论n', 'right')}
                      {thSort('win_rate', '理论胜率', 'right')}
                      {thSort('breakeven', '理论保本率', 'right', '按上线后真实入场报价计算；暂无报价样本时显示—')}
                      <th className="py-1 px-2 text-right" title="冻结回测胜率×0.98：费后理论允许的最高入场价">回测保本价</th>
                      {thSort('bench', '回测胜率', 'right')}
                      {thSort('dev', '偏离', 'right')}
                      {thSort('avg_ev', '理论单均EV', 'right')}
                      {thSort('cum_ev', '理论累计EV', 'right')}
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {familyRows.map(([k, m]) => {
                  const i = sorted.findIndex(([version]) => version === k)
                  const s = analytics.shadow[k].summary
                  // 与图表同序：本区第 11 条起降透明度，保证表格色块与曲线一一对应
                  const faded = i >= 10
                  const displayWr = guardMode === 'GUARDED' ? s.guarded_win_rate : s.win_rate
                  const dev = displayWr != null && s.bench_winrate != null ? displayWr - s.bench_winrate : null
                  const online = s.enabled !== false
                  const retired = s.retired === true
                  const live = retired ? null : (s.live_channel ?? null)
                  return (
                    <tr key={k} className={`border-b border-line-soft hover:bg-sunken ${online ? '' : 'opacity-45'}`}>
                      <td className="py-1 px-2 font-medium" style={{ color: m.color, opacity: faded ? 0.6 : 1 }} title={s.desc}>
                        <span className="inline-flex items-center gap-1 flex-wrap">
                          {m.label}
                          <span className="ds-badge ds-badge-neutral">{signalFamilyLabel(s.family)}</span>
                          {s.role && s.role !== 'STRATEGY' && <span className="ds-badge ds-badge-indigo">{ROLE_LABELS[s.role] ?? s.role}</span>}
                          <span className={`ds-badge ${EXECUTION_BADGE[s.execution_mode ?? 'SHADOW_ONLY']?.cls ?? 'ds-badge-neutral'}`}>
                            {EXECUTION_BADGE[s.execution_mode ?? 'SHADOW_ONLY']?.label ?? '仅影子'}
                          </span>
                          {signalDescFor('shadow', k) && <HelpHint text={`${k}：${signalDescFor('shadow', k)}`} />}
                        </span>
                      </td>
                      <td className="py-1 px-2 text-center">
                        {retired ? (
                          <span
                            className="px-1.5 py-0.5 rounded-pill text-[10px] font-medium border border-line bg-sunken text-ink-55 cursor-not-allowed"
                            title="已永久退役（2026-09-04）：代码级硬闸停发，不可重新上线；冻结 bench 基准与历史曲线仍保留供审计"
                          >已退役</span>
                        ) : (
                          <button
                            onClick={async () => { await api.toggleShadow(k, !online); load() }}
                            className={`px-1.5 py-0.5 rounded-pill text-[10px] font-medium border transition ${
                              online
                                ? 'text-positive border-positive bg-positive-soft hover:bg-positive-soft-hover'
                                : 'text-ink-55 border-line bg-card hover:bg-sunken'
                            }`}
                            title={online
                              ? '点击下线：停止采集新信号（历史数据保留，曲线照常显示已有样本）'
                              : '点击上线：恢复采集新信号'}
                          >
                            {online ? '在线' : '已下线'}
                          </button>
                        )}
                      </td>
                      <td className="py-1 px-2 text-center">
                        {live ? (
                          <button
                            onClick={async () => {
                              const next = !live.enabled
                              if (next) {
                                if (!window.confirm(
                                  `确认开启通道实盘？\n通道: ${SIGNAL_INFO[k]?.name ?? k}（${k}）\n每单 ${String(live.amount_usdt)} USDT | 执行价护栏 ${String(live.max_exec_price)} | 日限 ${String(live.max_daily_orders)} 单\n\n命中信号将下真实订单（真金白银）；设定写入 DB 重启后保持（仅持久化失败时回落 LIVE_CHANNELS_JSON）。`)) return
                              } else {
                                if (!window.confirm(
                                  `确认关闭通道实盘？\n通道: ${SIGNAL_INFO[k]?.name ?? k}（${k}）\n（不取消在途任务，只阻止该通道新单派生）`)) return
                              }
                              try {
                                const res = await api.postLiveChannel(k, next)
                                if (res?.error) alert(`切换失败: ${String(res.error)}`)
                                load()
                              } catch (e) {
                                alert(`请求失败: ${(e as Error).message}`)
                              }
                            }}
                            className={`px-1.5 py-0.5 rounded-pill text-[10px] font-medium border transition ${
                              live.enabled
                                ? 'text-positive border-positive bg-positive-soft hover:bg-positive-soft-hover'
                                : 'text-ink-55 border-line bg-card hover:bg-sunken'
                            }`}
                            title={live.enabled ? '实盘下单中，点击关闭该通道（影子采集不受影响）' : '点击开启实盘下单（真金白银；影子采集不受影响）'}
                          >
                            {live.enabled ? '实盘中' : '加入实盘'}
                          </button>
                        ) : (
                          <span className="inline-flex items-center gap-0.5 text-ink-40">
                            —
                            <HelpHint text="该影子族未注册实盘通道（不下注）；实盘化需在 services/live_channels.py 注册同名 ChannelSpec 并重启" />
                          </span>
                        )}
                      </td>
                      <td className="py-1 px-2 text-center">
                        <button
                          onClick={() => setDetailVersion(k)}
                          className="px-2 py-0.5 rounded-pill text-[10px] font-semibold border border-brand/40 bg-brand-soft text-brand hover:bg-brand hover:text-white transition shadow-2xs"
                          title="查看该信号逐笔详情、实盘订单对齐与分析图表"
                        >
                          详情 / 对比
                        </button>
                      </td>
                      <td className="py-1 px-2 text-right font-mono text-ink-55" title={s.guard_price != null ? `护栏 ≤ ${s.guard_price}；已拦截弃单 ${s.guarded_rejected_n ?? 0} 笔` : '未配置护栏'}>
                        {s.guard_price != null ? (
                          <span>
                            {s.guard_price}
                            <span className="text-[10px] text-negative ml-1">(-{s.guarded_rejected_n ?? 0})</span>
                          </span>
                        ) : '—'}
                      </td>
                      {guardMode === 'BOTH' ? (
                        <>
                          <td className="py-1 px-2 text-right font-mono">
                            <span className="font-bold text-ink-95">{s.guarded_n ?? 0}</span>
                            <span className="text-ink-40 text-[10px]"> / {s.n}</span>
                          </td>
                          <td className="py-1 px-2 text-right font-mono font-medium">{pct1(s.guarded_fill_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono font-bold text-brand">{pct1(s.guarded_win_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-40">{pct1(s.win_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55" title={s.guarded_avg_breakeven == null ? '暂无前向真实报价样本' : '带护栏成交信号前向真实入场报价对应的平均保本胜率'}>{pct1(s.guarded_avg_breakeven)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_max_entry_price)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_winrate)}</td>
                          <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-ink-55' : dev >= 0 ? 'text-positive' : 'text-negative'}`}>
                            {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                          </td>
                          <td className={`py-1 px-2 text-right font-mono font-bold ${s.guarded_avg_ev == null ? 'text-ink-55' : s.guarded_avg_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.guarded_avg_ev)}</td>
                          <td className={`py-1 px-2 text-right font-mono text-[11px] ${s.avg_ev == null ? 'text-ink-40' : s.avg_ev >= 0 ? 'text-positive/70' : 'text-negative/70'}`}>{evFmt(s.avg_ev)}</td>
                          <td className={`py-1 px-2 text-right font-mono font-bold ${s.guarded_cum_ev == null ? 'text-ink-55' : s.guarded_cum_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.guarded_cum_ev)}</td>
                          <td className={`py-1 px-2 text-right font-mono text-[11px] ${s.cum_ev == null ? 'text-ink-40' : s.cum_ev >= 0 ? 'text-positive/70' : 'text-negative/70'}`}>{evFmt(s.cum_ev)}</td>
                        </>
                      ) : guardMode === 'GUARDED' ? (
                        <>
                          <td className="py-1 px-2 text-right font-mono font-bold">{s.guarded_n ?? 0}</td>
                          <td className="py-1 px-2 text-right font-mono font-medium">{pct1(s.guarded_fill_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono font-bold text-brand">{pct1(s.guarded_win_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55" title={s.guarded_avg_breakeven == null ? '暂无前向真实报价样本' : '带护栏成交信号前向真实入场报价对应的平均保本胜率'}>{pct1(s.guarded_avg_breakeven)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_max_entry_price)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_winrate)}</td>
                          <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-ink-55' : dev >= 0 ? 'text-positive' : 'text-negative'}`}>
                            {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                          </td>
                          <td className={`py-1 px-2 text-right font-mono ${s.guarded_avg_ev == null ? 'text-ink-55' : s.guarded_avg_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.guarded_avg_ev)}</td>
                          <td className={`py-1 px-2 text-right font-mono ${s.guarded_cum_ev == null ? 'text-ink-55' : s.guarded_cum_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.guarded_cum_ev)}</td>
                        </>
                      ) : (
                        <>
                          <td className="py-1 px-2 text-right font-mono">{s.n}</td>
                          <td className="py-1 px-2 text-right font-mono font-bold">{pct1(s.win_rate)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55" title={s.avg_breakeven == null ? '暂无前向真实报价样本' : '前向真实入场报价对应的平均保本胜率'}>{pct1(s.avg_breakeven)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_max_entry_price)}</td>
                          <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_winrate)}</td>
                          <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-ink-55' : dev >= 0 ? 'text-positive' : 'text-negative'}`}>
                            {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                          </td>
                          <td className={`py-1 px-2 text-right font-mono ${s.avg_ev == null ? 'text-ink-55' : s.avg_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.avg_ev)}</td>
                          <td className={`py-1 px-2 text-right font-mono ${s.cum_ev == null ? 'text-ink-55' : s.cum_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.cum_ev)}</td>
                        </>
                      )}
                    </tr>
                  )
                })}
              </tbody>
            </table>
              </div>
            })}
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart
              data={
                guardMode === 'GUARDED'
                  ? mergeCurves(
                      Object.fromEntries(sorted.map(([k]) => [k, analytics.shadow[k]])),
                      SHADOW_META,
                      true,
                    )
                  : guardMode === 'THEORETICAL'
                  ? mergeCurves(
                      Object.fromEntries(sorted.map(([k]) => [k, analytics.shadow[k]])),
                      SHADOW_META,
                      false,
                    )
                  : rows
              }
              margin={{ top: 6, right: 12, left: 0, bottom: 0 }}
            >
              <CartesianGrid stroke="var(--line-soft)" />
              <XAxis dataKey="ts" type="number" domain={['dataMin', 'dataMax']} scale="time"
                tickFormatter={utcMD} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={40}
                tickFormatter={(v: number) => v + '%'} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n]}
              />
              {pumpTs != null && <ReferenceLine x={pumpTs} stroke="var(--warning)" strokeDasharray="4 3" />}
              {sorted.map(([k, m], i) => {
                const s = analytics.shadow[k].summary
                const faded = i >= 10
                const activeBreakeven = guardMode === 'GUARDED' ? s.guarded_avg_breakeven : s.avg_breakeven
                return (
                  <Fragment key={k}>
                    <Line dataKey={m.label} stroke={m.color} strokeWidth={faded ? 1.5 : 2}
                      strokeOpacity={faded ? 0.55 : 1} dot={false} connectNulls isAnimationActive={false} />
                    {s.bench_winrate != null && (
                      <ReferenceLine y={s.bench_winrate * 100} stroke={m.color} strokeDasharray="5 4" strokeOpacity={faded ? 0.3 : 0.5} />
                    )}
                    {activeBreakeven != null && (
                      <ReferenceLine y={activeBreakeven * 100} stroke={m.color} strokeDasharray="1 3" strokeOpacity={faded ? 0.35 : 0.6} />
                    )}
                  </Fragment>
                )
              })}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
      <div className="text-[10px] text-ink-55 mt-1">{footnote}</div>
      {detailVersion && (
        <ShadowSignalDetailModal
          version={detailVersion}
          onClose={() => setDetailVersion(null)}
        />
      )}
    </Card>
  )
}

function SceneSignalsCard({
  sceneEntries,
  sceneRows,
  analytics,
  pumpTs,
}: {
  sceneEntries: [string, { label: string; color: string }][]
  sceneRows: Record<string, number>[]
  analytics: SignalsAnalytics
  pumpTs: number | null
}) {
  const [kw, setKw] = useState('')
  const [sortCol, setSortCol] = useState<string>('default')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  const toggleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(false)
    }
  }

  const thSort = (field: string, label: string, align: 'left' | 'center' | 'right' = 'right', hint?: string) => (
    <th
      className={`py-1 px-2 text-${align} cursor-pointer select-none hover:text-brand transition-colors`}
      onClick={() => toggleSort(field)}
      title={hint}
    >
      <span className={`inline-flex items-center gap-0.5 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start'}`}>
        {label}
        {sortCol === field ? (
          <span className="text-brand font-mono text-[10px]">{sortAsc ? '▲' : '▼'}</span>
        ) : (
          <span className="text-ink-40 font-mono text-[10px]">⇅</span>
        )}
      </span>
    </th>
  )

  const search = kw.trim().toLowerCase()
  const filtered = sceneEntries.filter(([k, m]) => {
    const s = analytics.scene[k]?.summary
    if (!s) return false
    if (search) {
      const matchKey = k.toLowerCase().includes(search)
      const matchLabel = m.label.toLowerCase().includes(search)
      const matchExtra = (signalDescFor('scene', k) ?? '').toLowerCase().includes(search)
      if (!matchKey && !matchLabel && !matchExtra) return false
    }
    return true
  })

  const sorted = [...filtered].sort(([ka, ma], [kb, mb]) => {
    if (sortCol === 'default') return 0
    const sa = analytics.scene[ka]?.summary
    const sb = analytics.scene[kb]?.summary
    if (!sa || !sb) return 0
    let diff = 0
    if (sortCol === 'scene') {
      diff = ma.label.localeCompare(mb.label, 'zh-CN')
    } else if (sortCol === 'n') {
      diff = (sa.n ?? 0) - (sb.n ?? 0)
    } else if (sortCol === 'winrate') {
      diff = (sa.winrate ?? -1) - (sb.winrate ?? -1)
    } else if (sortCol === 'bench') {
      diff = (sa.bench_winrate ?? -1) - (sb.bench_winrate ?? -1)
    } else if (sortCol === 'dev') {
      const devA = sa.winrate != null && sa.bench_winrate != null ? sa.winrate - sa.bench_winrate : -999
      const devB = sb.winrate != null && sb.bench_winrate != null ? sb.winrate - sb.bench_winrate : -999
      diff = devA - devB
    } else if (sortCol === 'avg_ev') {
      diff = (sa.avg_ev ?? -999) - (sb.avg_ev ?? -999)
    } else if (sortCol === 'cum_ev') {
      diff = (sa.cum_ev ?? -999) - (sb.cum_ev ?? -999)
    }
    return sortAsc ? diff : -diff
  })

  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'

  return (
    <Card title="场景突破策略：采集记录与实盘执行状态">
      <div className="flex items-center justify-between mb-3 gap-2 flex-wrap text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type="text"
            placeholder="搜索场景/说明..."
            value={kw}
            onChange={e => setKw(e.target.value)}
            className={`${selectCls} w-44`}
          />
          {kw && (
            <button
              onClick={() => setKw('')}
              className="px-1.5 py-0.5 text-xs text-brand hover:underline"
            >清除筛选</button>
          )}
          {sortCol !== 'default' && (
            <button
              onClick={() => { setSortCol('default'); setSortAsc(false) }}
              className="px-1.5 py-0.5 text-xs text-ink-55 hover:underline"
            >重置排序</button>
          )}
        </div>
        <span className="text-xs text-ink-55">
          显示 {sorted.length} / 共 {sceneEntries.length} 个场景
        </span>
      </div>

      {sorted.length === 0 ? (
        <div className="text-center text-ink-55 py-6 text-xs">当前筛选条件下无匹配场景</div>
      ) : (
        <>
          <div className="overflow-x-auto mb-3">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-line text-ink-55">
                  {thSort('scene', '场景', 'left')}
                  {thSort('n', 'n', 'right')}
                  {thSort('winrate', '线上胜率', 'right')}
                  {thSort('bench', '回测', 'right')}
                  {thSort('dev', '偏离', 'right')}
                  {thSort('avg_ev', '平均EV', 'right')}
                  {thSort('cum_ev', '累计EV', 'right')}
                </tr>
              </thead>
              <tbody>
                {sorted.map(([k, m]) => {
                  const s = analytics.scene[k].summary
                  const dev = s.winrate != null && s.bench_winrate != null ? s.winrate - s.bench_winrate : null
                  return (
                    <tr key={k} className="border-b border-line-soft hover:bg-sunken">
                      <td className="py-1 px-2 font-medium" style={{ color: m.color }}>
                        <span className="inline-flex items-center gap-1 flex-wrap">
                          {m.label}
                          <span className="ds-badge ds-badge-neutral">{signalFamilyLabel(s.family)}</span>
                          <span className={`ds-badge ${EXECUTION_BADGE[s.execution_mode ?? 'RESEARCH_ONLY']?.cls ?? 'ds-badge-neutral'}`}>
                            {EXECUTION_BADGE[s.execution_mode ?? 'RESEARCH_ONLY']?.label ?? '仅研究'}
                          </span>
                          {signalDescFor('scene', k) && <HelpHint text={`${k}：${signalDescFor('scene', k)}`} />}
                        </span>
                      </td>
                      <td className="py-1 px-2 text-right font-mono">{s.n}</td>
                      <td className="py-1 px-2 text-right font-mono font-bold">{pct1(s.winrate)}</td>
                      <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.bench_winrate)}</td>
                      <td className={`py-1 px-2 text-right font-mono ${dev == null ? 'text-ink-55' : dev >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {dev == null ? '—' : `${dev >= 0 ? '+' : ''}${(dev * 100).toFixed(1)}pp`}
                      </td>
                      <td className={`py-1 px-2 text-right font-mono ${s.avg_ev == null ? 'text-ink-55' : s.avg_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.avg_ev)}</td>
                      <td className={`py-1 px-2 text-right font-mono ${s.cum_ev == null ? 'text-ink-55' : s.cum_ev >= 0 ? 'text-positive' : 'text-negative'}`}>{evFmt(s.cum_ev)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={sceneRows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--line-soft)" />
              <XAxis dataKey="ts" type="number" domain={['dataMin', 'dataMax']} scale="time"
                tickFormatter={utcMD} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={40}
                tickFormatter={(v: number) => v + '%'} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                formatter={(v, n) => [typeof v === 'number' ? v.toFixed(1) + '%' : '--', n]}
              />
              {pumpTs != null && <ReferenceLine x={pumpTs} stroke="var(--warning)" strokeDasharray="4 3" />}
              {sorted.map(([k, m]) => (
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
        </>
      )}
      <div className="text-[10px] text-ink-55 mt-1">实线=线上累计胜率，同色虚线=回测冻结基准；x 轴为信号时间（UTC）。</div>
    </Card>
  )
}

function RegimeByVersionTable({
  shadowEntries,
  regimeByVersion,
}: {
  shadowEntries: [string, { label: string; color: string }][]
  regimeByVersion: Record<string, Record<string, { n: number; wins: number; winrate: number | null }>>
}) {
  const [kw, setKw] = useState('')
  const [phaseFilter, setPhaseFilter] = useState<'ALL' | 'pre' | 'pump'>('ALL')
  const [sortCol, setSortCol] = useState<'default' | 'ver' | 'phase' | 'n' | 'winrate'>('default')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  const items: Array<{
    key: string
    label: string
    color: string
    phase: 'pre' | 'pump'
    phaseName: string
    n: number
    winrate: number | null
  }> = []

  for (const [k, m] of shadowEntries) {
    const phases = regimeByVersion[k]
    if (!phases) continue
    for (const ph of ['pre', 'pump'] as const) {
      const g = phases[ph]
      items.push({
        key: k,
        label: m.label,
        color: m.color,
        phase: ph,
        phaseName: ph === 'pre' ? '大涨前' : '大涨期',
        n: g ? g.n : 0,
        winrate: g ? g.winrate : null,
      })
    }
  }

  const search = kw.trim().toLowerCase()
  const filtered = items.filter(item => {
    if (search && !item.label.toLowerCase().includes(search) && !item.key.toLowerCase().includes(search)) return false
    if (phaseFilter !== 'ALL' && item.phase !== phaseFilter) return false
    return true
  })

  const sorted = [...filtered].sort((a, b) => {
    if (sortCol === 'default') return 0
    let diff = 0
    if (sortCol === 'ver') {
      diff = a.label.localeCompare(b.label, 'zh-CN')
    } else if (sortCol === 'phase') {
      diff = a.phase.localeCompare(b.phase)
    } else if (sortCol === 'n') {
      diff = a.n - b.n
    } else if (sortCol === 'winrate') {
      diff = (a.winrate ?? -1) - (b.winrate ?? -1)
    }
    return sortAsc ? diff : -diff
  })

  const toggleSort = (col: 'ver' | 'phase' | 'n' | 'winrate') => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }

  const thSort = (field: 'ver' | 'phase' | 'n' | 'winrate', label: string, align: 'left' | 'center' | 'right' = 'right') => (
    <th
      className={`py-1 px-2 text-${align} cursor-pointer select-none hover:text-brand transition-colors`}
      onClick={() => toggleSort(field)}
    >
      <span className={`inline-flex items-center gap-0.5 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start'}`}>
        {label}
        {sortCol === field ? (
          <span className="text-brand font-mono text-[10px]">{sortAsc ? '▲' : '▼'}</span>
        ) : (
          <span className="text-ink-40 font-mono text-[10px]">⇅</span>
        )}
      </span>
    </th>
  )

  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'
  const filterActive = search !== '' || phaseFilter !== 'ALL'

  return (
    <div className="overflow-x-auto mb-4">
      <div className="flex items-center justify-between mb-2 gap-2 flex-wrap text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-ink-55 font-medium">逐影子版本拆分</span>
          <input
            type="text"
            placeholder="搜索版本..."
            value={kw}
            onChange={e => setKw(e.target.value)}
            className={`${selectCls} w-32`}
          />
          <select
            value={phaseFilter}
            onChange={e => setPhaseFilter(e.target.value as 'ALL' | 'pre' | 'pump')}
            className={selectCls}
          >
            <option value="ALL">全部阶段</option>
            <option value="pre">大涨前</option>
            <option value="pump">大涨期</option>
          </select>
          {filterActive && (
            <button
              onClick={() => { setKw(''); setPhaseFilter('ALL') }}
              className="px-1.5 py-0.5 text-xs text-brand hover:underline"
            >清除</button>
          )}
          {sortCol !== 'default' && (
            <button
              onClick={() => { setSortCol('default'); setSortAsc(false) }}
              className="px-1.5 py-0.5 text-xs text-ink-55 hover:underline"
            >重置排序</button>
          )}
        </div>
        <span className="text-xs text-ink-55">显示 {sorted.length} / 共 {items.length} 项</span>
      </div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-line text-ink-55">
            {thSort('ver', '版本', 'left')}
            {thSort('phase', '阶段', 'center')}
            {thSort('n', 'n', 'right')}
            {thSort('winrate', '胜率', 'right')}
          </tr>
        </thead>
        <tbody>
          {sorted.map(item => (
            <tr key={`${item.key}-${item.phase}`} className="border-b border-line-soft hover:bg-sunken">
              <td className="py-1 px-2" style={{ color: item.color }}>
                {item.label}
              </td>
              <td className="py-1 px-2 text-center text-ink-55">
                {item.phaseName}
              </td>
              <td className="py-1 px-2 text-right font-mono">{item.n}</td>
              <td className="py-1 px-2 text-right font-mono font-bold">{pct1(item.winrate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

  return (
    <div className="space-y-6">
      {/* 控制栏 */}
      <div className="flex items-center gap-4 flex-wrap text-xs">
        <label className="flex items-center gap-1.5 cursor-pointer text-ink-80">
          <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
          自动刷新（60s）
        </label>
        <button
          onClick={load}
          className="px-3 py-1 text-xs font-medium text-white bg-brand rounded-sm hover:bg-brand-hover transition"
        >
          手动刷新
        </button>
        {lastUpdate && <span className="text-ink-55">最后更新 {lastUpdate.toLocaleTimeString('zh-CN')}</span>}
        {err && <span className="text-negative">{err}</span>}
      </div>

      {/* 仅保留一个总览入口：类型统计替代重复的 BTC 背景图；实时 K 线集中在下单弹窗。 */}
      {analytics && <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
        <div className="ds-card p-3"><div className="text-xs text-ink-55">实盘执行中</div><div className="text-xl font-bold font-mono text-positive">{[...Object.values(analytics.shadow), ...Object.values(analytics.scene)].filter(block => block.summary.execution_mode === 'LIVE_ACTIVE').length}</div></div>
        <div className="ds-card p-3"><div className="text-xs text-ink-55">仅影子 / 研究</div><div className="text-xl font-bold font-mono text-violet">{[...Object.values(analytics.shadow), ...Object.values(analytics.scene)].filter(block => ['SHADOW_ONLY', 'RESEARCH_ONLY'].includes(block.summary.execution_mode ?? '')).length}</div></div>
        <div className="ds-card p-3"><div className="text-xs text-ink-55">蜡烛反转逻辑版本</div><div className="text-xl font-bold font-mono text-brand">{Object.values(analytics.shadow).filter(block => block.summary.family === 'candlestick_reversal').length}</div><div className="text-[10px] text-ink-55">同窗按单事件多标签去重</div></div>
      </div>}

      <Card title={`信号位置概览（${kinterval === '1d' ? '日线 × 30' : '1小时 × 168'}，UTC 已收盘）`}>
        <div className="flex items-center gap-3 mb-2 text-xs">
          {(['1d', '1h'] as const).map(iv => (
            <button
              key={iv}
              onClick={() => setKinterval(iv)}
              className={`px-2.5 py-0.5 rounded-sm border transition ${
                kinterval === iv
                  ? 'bg-brand text-white border-brand'
                  : 'bg-card text-ink-80 border-line hover:border-brand'
              }`}
            >
              {iv === '1d' ? '日线' : '1小时'}
            </button>
          ))}
          <span className="text-ink-55">
            <span className="text-warning">● 场景信号</span> · <span className="text-brand">● 影子信号</span>（点大小=当根信号数）
            {pumpTs != null && <span className="ml-2 text-warning">| 竖线 = {utcMD(pumpTs)} 大涨分界</span>}
          </span>
        </div>
        {klines.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={klines.map(k => ({ t: k.open_time, close: k.close }))}
              margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--line-soft)" />
              <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                tickFormatter={utcMD} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
              <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={64}
                tickFormatter={(v: number) => v.toLocaleString()} />
              <Tooltip
                contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                labelFormatter={t => new Date(t as number).toUTCString().slice(0, 16)}
                formatter={v => [typeof v === 'number' ? v.toLocaleString() : '--', '收盘价']}
              />
              {pumpTs != null && (
                <ReferenceLine x={pumpTs} stroke="var(--warning)" strokeDasharray="4 3" strokeWidth={1.5}
                  label={{ value: `${utcMD(pumpTs)} 大涨分界`, position: 'insideTopRight', fill: 'var(--warning)', fontSize: 10 }} />
              )}
              <Line dataKey="close" stroke="var(--ink-80)" dot={false} strokeWidth={1.8} isAnimationActive={false} />
              {Array.from(sceneDots.entries()).map(([i, n]) => (
                <ReferenceDot key={`sc${i}`} x={klines[i].open_time} y={klines[i].close}
                  r={Math.min(3 + n, 6)} fill="var(--warning)" fillOpacity={0.75} stroke="white" />
              ))}
              {Array.from(shadowDots.entries()).map(([i, n]) => (
                <ReferenceDot key={`sh${i}`} x={klines[i].open_time} y={klines[i].close}
                  r={Math.min(3 + n, 6)} fill="var(--chart-1)" fillOpacity={0.75} stroke="white" />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center text-ink-55 py-10 text-sm">K 线加载中...</div>
        )}
      </Card>

      {/* 场景信号 */}
      {analytics && (
        <SceneSignalsCard
          sceneEntries={sceneEntries}
          sceneRows={sceneRows}
          analytics={analytics}
          pumpTs={pumpTs}
        />
      )}

      <details className="ds-card">
        <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-ink-95">旧版影子：在线采集与统计（展开查看）<HelpHint text="保留旧版在线采集、前向胜率、回测基准与保本线；新版前向证据及执行漏斗见页面底部。" /></summary>
        <div className="space-y-6 px-2 pb-2">
      {/* 旧版影子入口与完整在线采集指标一并折叠。 */}
      {analytics && featuredShadowKeys.map(k => {
        const s = analytics.shadow[k].summary
        const meta = SHADOW_META[k]
        return <div key={k} className="ds-card p-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap text-sm font-semibold text-ink-95">
              <span>{meta?.label ?? k}</span>
              <span className="ds-badge ds-badge-violet">新上线影子</span>
              <span className="ds-badge ds-badge-neutral">实盘默认关闭</span>
              <HelpHint text={`${k}：${signalDescFor('shadow', k)}`} />
            </div>
            <div className="mt-1 text-xs text-ink-55">30 天滚动 ret1h q90 路由；普通行情开盘模拟 DOWN，轧空行情等首根 5m 收阴确认。</div>
          </div>
          <div className="flex items-center gap-5 text-right">
            <div><div className="text-[10px] text-ink-55">前向样本</div><div className="font-mono font-semibold text-ink-95">{s.n}</div></div>
            <div><div className="text-[10px] text-ink-55">回测胜率</div><div className="font-mono font-semibold text-violet">{pct1(s.bench_winrate)}</div></div>
            <a href="#s1-dyn-shadow" className="text-xs font-medium text-brand hover:underline">查看完整指标 ↓</a>
          </div>
        </div>
      })}

      {/* 影子在线区（2026-09-07 与退役区分开）：只看仍在采集的版本前向进展 */}
      {analytics && (
        <div id="s1-dyn-shadow" className="scroll-mt-4">
        <ShadowSignalCard
          title={`影子信号·在线采集（${activeShadowEntries.length} 版）：前向胜率 vs 回测基准 vs 保本线`}
          entries={activeShadowEntries}
          rows={activeShadowRows}
          analytics={analytics}
          load={load}
          pumpTs={pumpTs}
          footnote={
            <>
              实线=线上累计胜率；同色虚线=回测冻结胜率；同色点线=前向真实报价对应的平均保本率（没有前向报价样本时不绘制）。
              第 11 条起复用前十色相并降为淡色细线（克制色板做不出 22 个可区分色相），表格中的版本名同步淡化以保持对应。
              本区版本仍在采集，曲线末端=最新前向进展；「回测保本价」=冻结胜率×0.98，与前向保本率不是同一指标。
            </>
          }
        />
        </div>
      )}

        </div>
      </details>

      {/* 固定 2026-08-19 的历史大涨切片已过时且与现有 regime 分析重叠，仅保留折叠审计。 */}
      <details className="ds-card">
        <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-ink-95">历史审计：大涨前后切片</summary>
      <Card title={`大涨前 vs 大涨期（${pumpTs != null ? `${utcMD(pumpTs)} 00:00` : '—'} UTC 分界）`}>
        {analytics && (
          <>
            <div className="flex flex-wrap gap-4 mb-4">
              {[['pre', '大涨前（震荡期）'], ['pump', '大涨期（三根大阳）']].map(([ph, name]) => {
                const g = analytics.regime.phases[ph]
                return (
                  <div key={ph} className="flex-1 min-w-[180px] rounded-card border border-line bg-card p-3">
                    <div className="text-xs text-ink-55 mb-1">{name}</div>
                    {g ? (
                      <div className="flex items-baseline gap-3">
                        <span className="text-xl font-bold font-mono text-ink-95">{pct1(g.winrate)}</span>
                        <span className="text-xs text-ink-55">n={g.n} · 赢{g.wins}</span>
                      </div>
                    ) : (
                      <div className="text-sm text-ink-55">无样本</div>
                    )}
                  </div>
                )
              })}
            </div>
            {/* 逐影子版本 × 阶段（对齐审计报告表二归因维度） */}
            {Object.keys(analytics.regime.by_version).length > 0 && (
              <RegimeByVersionTable
                shadowEntries={shadowEntries}
                regimeByVersion={analytics.regime.by_version}
              />
            )}
            {analytics.regime.daily.length > 0 && (
              <>
                <div className="text-xs text-ink-55 mb-1">按天胜率（UTC 日）</div>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={analytics.regime.daily.map(d => ({
                    date: d.date.slice(5), wr: d.winrate != null ? +(d.winrate * 100).toFixed(1) : 0, n: d.n,
                  }))} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="var(--line-soft)" />
                    <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={40}
                      tickFormatter={(v: number) => v + '%'} />
                    <Tooltip
                      contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                      formatter={(v, n, item) => n === 'wr'
                        ? [`${v}%（n=${(item?.payload as { n?: number })?.n ?? '-'}）`, '胜率']
                        : [String(v), String(n)]}
                    />
                    <ReferenceLine y={50} stroke="var(--ink-55)" strokeDasharray="4 4" />
                    <Bar dataKey="wr" fill="var(--chart-1)" fillOpacity={0.75} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </>
        )}
      </Card>
      </details>

      <ShadowExecutionComparisonCard data={executionComparison} unavailable={executionUnavailable} />
    </div>
  )
}
