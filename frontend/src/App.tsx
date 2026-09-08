import { useState, useEffect, useCallback, useRef, useMemo, Fragment } from 'react'
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
  getRecentTrades: (limit = 100) => authFetch(`/api/trades/recent?limit=${limit}`).then(r => r.json()),
  getFundFlow: () => authFetch('/api/trades/fund-flow').then(r => r.json()),
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
  postSyncBinance: () => authFetch('/api/trades/sync-binance', { method: 'POST' }).then(r => r.json()),
  // 奖金领取（2026-08-23）：可领查询 + batch-redeem
  getRedeemable: () => authFetch('/api/prediction/redeemable').then(r => r.json()),
  postRedeem: () =>
    authFetch('/api/prediction/redeem', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
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

// 问号 hover 提示（信号说明等）：纯 CSS group-hover，无依赖
function HelpHint({ text }: { text: string }) {
  /* 反白气泡（ink-black + on-ink）脱离页面层级，不用阴影。
     tabIndex + :focus-within 保留：键盘必须能触发（AGENTS.md 既有行为，不得回退）。 */
  return (
    <span className="ds-hint align-middle" tabIndex={0}>
      <span className="ds-hint-trigger">?</span>
      <span className="ds-hint-bubble">{text}</span>
    </span>
  )
}

// 线上信号通道说明（口径源：services/live_channels.py 注册表 + quote_edge_detector 冻结规则）
// 2026-08-24 多通道实盘改造：全部通道支持实盘下单（liveOk），各自独立金额/日限/护栏，
// 通道运行状态（开关/金额/日限/护栏/今日成交）来自后端 multi_live_trader.status_async()。
// 2026-09-04 退役 8 通道（retired）：已从 live_channels.LIVE_CHANNELS 与
// shadow_version_gate.RETIRED_VERSIONS 移除，不再开火/不再采集。条目必须保留：
// 历史订单行仍带这些 signal_version，删了会让订单表退回成裸版本串。
const SIGNAL_INFO: Record<string, { name: string; kind: '实盘' | '影子' | '场景'; desc: string; liveOk?: boolean; retired?: boolean }> = {
  quote_contrarian_v1: {
    name: '报价反向（B格逆势）', kind: '实盘', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：被 quote_contrarian_v2 严格支配（v2 = v1 触发集纯子集 + BTC 门禁）】5 分钟窗口开始后 45~60 秒内，DOWN token 报价首次跌入 [0.15, 0.25)（明显便宜）时买入 DOWN。低胜率高赔付：回测胜率 24%、EV +0.155（赢一次约赚 4 倍）。通道护栏 0.28（区间上界+0.03），每窗至多一单。',
  },
  quote_momentum_v1: {
    name: '报价动量（A格顺势）', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：「深折价顺势」假设被前向数据证伪】5 分钟窗口 90~120 秒内，DOWN token 报价首次进入 [0.69, 0.75)（强势确认）时押 DOWN。回测胜率 79.9%、EV +0.097。通道护栏 0.78，每窗至多一单。',
  },
  quote_contrarian_v2: {
    name: '报价反向·门禁版', kind: '影子', liveOk: true,
    desc: 'v1 区间 + BTC 价格门禁：触发时点 BTC 未高于窗口开盘 ≥0.10%（只接「假冲高」，归因显示平盘窗贡献 86% 利润）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.28。注：触发区间仍引用已退役 v1 的冻结回测口径（QUOTE_EDGE_RULES 是口径事实源，整体保留）。',
  },
  quote_momentum_v2: {
    name: '报价动量·门禁版', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：随 momentum 族整体下线】v1 区间 + BTC 价格门禁：触发时点 BTC 已低于窗口开盘 ≥0.10%（剔「假恐慌」，真跌段胜率 85% vs 假恐慌段 40%）。实盘已解锁（实时 BTC 喂价门禁），通道护栏 0.78。',
  },
  quote_momentum_v3: {
    name: '报价动量·非连涨门禁版', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：修正未来函数后门禁效应≈0（+3.8pp，CI 重叠），不值得占实盘额度】v1 区间（t90~120s q∈[0.69,0.75)）∩ 触发时点末收 15m 非连涨（close≤前根，严格 ex-ante）。回测 80.2% vs 连涨 76.4%。与 v1/v2 同窗互斥（至多一单成交），通道护栏 0.78。',
  },
  quote_contrarian_v3a: {
    name: '报价反向·交替环境版', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：环境门禁未兑现，被 contrarian_v2 支配】contrarian v1 区间 + v2 价格门禁 + 环境门禁：前窗结算 DOWN（交替环境：前窗跌+本窗涨=V 反弹假冲高）。真实回测 n=85 胜率 31.8%、EV +0.528。实盘已解锁（前窗 outcome 异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  quote_contrarian_v3b: {
    name: '报价反向·日高回落版', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：同 v3a，门禁未兑现】v3a + 触发时点 BTC 距当日高点回落 ≥0.30%（含边界，震荡日冲高更易衰竭）。真实回测 n=65 胜率 33.8%、EV +0.646（单笔 EV 最优）。实盘已解锁（日高异步 DB 核验，缺失弃单），通道护栏 0.28。',
  },
  quote_contrarian_v4: {
    name: '报价反向·下跌周期版', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：regime 门禁未兑现，被 contrarian_v2 支配】contrarian v1 区间 + regime 门禁：触发时点过去 24h BTC 收益 ≤ −1.0%（含边界，只在下跌周期开火）。62 天真实订单簿回测：down 段 n=413 胜率 30.3%、EV +0.372（CI 下界过盈亏平衡线），up/range 段 EV≈0——正边际集中在下跌周期。5m K 线严格 ex-ante 口径（影子/实盘同源，缺失保守弃单），每窗至多一单，通道护栏 0.28。',
  },
  x4_v1: {
    name: '情绪错位（收阳押次窗DOWN）', kind: '影子', liveOk: true, retired: true,
    desc: '【2026-09-04 已退役：被 x4_v2 严格支配（v2 = v1 触发集纯子集 + 平静市门禁）】本窗收阳但 15m 市场收尾情绪 ≤40 的错位 → 次窗 +150s 决策点押 DOWN（回测合并胜率 63.5%、EV +0.254）。实盘已解锁：PENDING 信号轮询→决策点下单，护栏 0.45，错过决策点不追单。',
  },
  x4_v2: {
    name: '情绪错位·平静市门禁版', kind: '影子', liveOk: true,
    desc: 'x4_v1 + 平静市门禁（回测胜率 45.3%，仅平静市况触发）。实盘已解锁：同 x4_v1 决策点机制，护栏 0.50，错过决策点不追单。',
  },
  x4_v3: {
    name: '情绪错位·趋势过滤版', kind: '影子', liveOk: true,
    desc: 'x4_v2 门禁 + 双趋势门禁：触发时点 BTC 过去 4h 跌超 1%（急跌不接刀）或过去 24h 涨超 2%（过热不追空）均不触发，K 线数据缺失同不触发；另有下单层入场价白名单 [0,0.2)∪[0.3,0.4)（决策点成交均价超带弃单，护栏 0.50 兜底）。与 x4_v2 同窗并行对比（各自独立下单），实盘默认关闭需手动开启。回测 C3 保留段 40 笔胜率 42.5%、EV +1.629。',
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
  s5_deep_z20_v1: {
    name: 'S5深档·深回落门禁版', kind: '场景', liveOk: true,
    desc: 'S1 + 5min 回落确认且 z5≤−20bp 的深回落子集 → +5min 确认即押次周期 15m DOWN（回测 ~91.3%，EV 偏乐观含机械成分；盈亏平衡 ~86.7%）。小金额实盘前向验证，护栏 0.88；与 S5 确认通道同窗互斥（至多一单成交）。',
  },
  // ---- 2026-09-06 影子 promote 5 通道（口径源：services/live_channels.py + 各检测器冻结研究口径）----
  s2_cond_t4_v1: {
    name: 'S2条件单·t4价<开（押UP）', kind: '影子', liveOk: true,
    desc: '实盘 S2（bear_exhaust，破 4h 支撑+收阴+放量）派生：次周期 t=4(+240s) 1m 收盘 < 周期开盘（全深度回落）→ 当刻买 UP 押次周期 15m 收阳（价跌时 UP 变便宜，低买 UP 的正 EV 来自入场价而非胜率，真实 EV 由生产报价前向现算）。720d 触发 1069/2176（49.1%，1.48/天），价-only 胜率 38.9%；通道护栏 0.38（38.9%×0.98 平衡价略下方）。与 t5 剔深同窗互斥（至多一单成交）。',
  },
  s2_cond_t5d_v1: {
    name: 'S2条件单·t5剔深（押UP）', kind: '影子', liveOk: true,
    desc: '实盘 S2（bear_exhaust）派生：次周期 t=5(+300s) 0<回落<15bp（中度回落剔深，排除已深跌的接刀窗）→ 当刻买 UP 押次周期 15m 收阳。720d 触发 643/2176（29.5%，0.89/天），价-only 胜率 44.8%；通道护栏 0.44（44.8%×0.98）。与 t4 价<开同窗互斥（至多一单成交）。',
  },
  nb_smaslope_5m_v1: {
    name: 'nextbar 5m动量误定价（押UP）', kind: '影子', liveOk: true,
    desc: '5m sma_slope_atr_5 ≥ 1.66（动量急升）→ 押次根 5m 收阳 UP。研究自评：720d 次根收阳仅 47.43%（长样本反指），edge 依赖 7-8 月 regime——record-only 科学仪器非背书，影子期前向验证「动量误定价」是否持续。护栏 0.46（47.43%×0.98 下方）刻意只放行 UP 便宜窗，成交少是保命设计。仅目标根开盘 ≤90s 内的新鲜命中下单（冷启动回补天然排除）。',
  },
  absorption_follow_td120_v1: {
    name: '吸收跟随·TD120（随BTC方向）', kind: '影子', liveOk: true,
    desc: '5m 窗内 BTC 明显位移而 UP 报价欠跟随（粘滞=知情者顶住人群吸筹的脚印）→ 跟随 BTC 位移方向下注（btc 涨押 UP/跌押 DOWN）。窗开 120s 双快照判定；滚动 14 天标定（弹性 k/位移门 p50/欠反应门 p80，严格 ex-ante，与影子同源）。真实价复核 RECENT EV +0.065 CI[+0.006,+0.123]（胜率 79.8%）；护栏 0.78（79.8%×0.98 下方）。与 TD150 同窗互斥（至多一单成交）。',
  },
  absorption_follow_td150_v1: {
    name: '吸收跟随·TD150（随BTC方向）', kind: '影子', liveOk: true,
    desc: '同 TD120 机制，TD=窗开 150s（多给 30s 让报价反应，触发更少、欠反应更纯）。真实价复核 RECENT EV +0.105 CI[+0.050,+0.165]（胜率 88.5%）；护栏 0.86（88.5%×0.98 下方）。与 TD120 同窗互斥（至多一单成交）。',
  },
  firsthit_down_v1: {
    name: '首触G0基底（押DOWN）', kind: '影子', liveOk: true,
    desc: '5m 窗内 DOWN 报价首次进入 (0.005,0.1] → 买 DOWN；实时重放本窗完整历史，只认真实第一触。默认关闭；开启后每通道每窗至多一单。G0/G1/G3 为用户确认的独立下单通道，同窗三门全中且全开启时最多 3 单。执行价护栏 0.08，实际成交均价高于护栏弃单，不追价。',
  },
  firsthit_down_body_v1: {
    name: '首触G1小实体（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G0 + body_r≤0.35（首触前 BTC 路径归一实体小，震荡而非单边跑出去）。默认关闭；与 G0/G3 独立下单，不做跨版本互斥，每通道每窗至多一单。执行价护栏 0.12，实际成交均价高于护栏弃单。confirm 段已 burned，护栏非前向胜率背书。',
  },
  firsthit_down_chg_v1: {
    name: '首触G3偏离（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G0 + chg≤+2.82bp（首触时 BTC 相对窗开盘涨幅有限，深折价更像报价错杀而非信息驱动）。默认关闭；与 G0/G1 独立下单，不做跨版本互斥，每通道每窗至多一单。执行价护栏 0.09，实际成交均价高于护栏弃单。confirm 段已 burned，护栏非前向胜率背书。',
  },
  firsthit_down_g4_v1: {
    name: '首触G4交互门（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G0 + 小实体(body_r≤0.35) ∧ 价格偏离(chg≤+2.82bp)，即 G1∩G3。冻结扫描 calib n=214 P=12.6% EV+1.03 CI[+0.08,+2.22]；confirm n=86 P=23.3% EV+1.85 CI[+0.35,+3.44]；BH-FDR q=0.063 STRICT_PASS。执行价护栏 0.12。⚠️非独立暴露：G4⊂G1 且 G4⊂G3，同窗三门全中即同一注重复下注。⚠️功效偏紧：calib EV(+1.03) 与前向 CI 半宽(≈1.08) 接近，前向通过概率约 50%，FAIL 时应延长观察期而非直接否决。默认关闭，每通道每窗至多一单。',
  },
  firsthit_down_g7_v1: {
    name: '首触G7基底（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G0 + 小实体(body_r≤0.35) ∧ 上影拒绝(wick01=1)。40d 全量回测胜率 19%~20%，EV +1.98~+2.34（FDR q=0.0002）。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  g7_streak_v1: {
    name: '首触G7+非强连阳（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G7 组合 + 前驱 5m 连阳≤1 根（排除多头单边大势逆势送命）。回测胜率 18.2%~20.0%，日均 5.1 单，FDR q=0.0004。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  g7_wick20_v1: {
    name: '首触G7+长上影（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G7 组合 + 上影线长度≥2.0bp（密集高位拒绝）。回测胜率 17.8%~24.1%，EV +1.84~+3.06。默认关闭，执行价护栏 0.12，每通道每窗至多一单。',
  },
  g7_strict_v1: {
    name: '首触G7严格版（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G7 组合 + streak≤1 ∧ upper_wick≥1.5bp。双重强化，回测胜率 20.8%~23.0%，EV +2.68~+3.27。默认关闭，执行价护栏 0.12，每通道每窗至多一单。',
  },
  g7_q05_v1: {
    name: '首触G7+深折价（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G7 组合 + 进场报价 q≤0.05（极端赔率凸性档）。回测单注 EV +4.10~+5.77。默认关闭，执行价护栏 0.05，每通道每窗至多一单。',
  },
  g7_t270_v1: {
    name: '首触G7+非极晚（押DOWN）', kind: '影子', liveOk: true,
    desc: 'G7 组合 + 触发时刻 t≤270s（保留均值回归扩散时间窗口）。回测胜率 27.9%~30.4%，EV +3.08~+4.41。默认关闭，执行价护栏 0.10，每通道每窗至多一单。',
  },
  late_night_contrarian_v2: {
    name: '深夜逆势·日高回落门禁版（押 DOWN）', kind: '影子', liveOk: true,
    desc: 'quote_edge 族深夜变体 v2：t∈[45,90)s 且报价 q∈[0.25,0.30) ∩ 北京时间窗 (22~24 时) ∩ 触发时点距当日高点回落≥0.30%。OOS n=50 wr 44.0% CI[31.2%,57.7%] vs 盈亏平衡≈27%；门禁数据缺失→不落表（保守跳过，v1 不受影响）。护栏 0.33（区间上界 0.30+0.03 容忍），开启后每通道每窗至多一单。⚠️纪律推翻：原 docstring"纯影子前向攒样本"被用户拍板改为实盘注册（线上已开启下单）。',
  },
}

/* ==========================================================================
   SIGNAL_KIND_BADGE — DESIGN.md 规格
   - 实盘=绿 / 影子=紫 / 场景=靛蓝（映射冻结）
   - pill + 语义色浅底 + 同色字，无边框
   - text-[10px] → text-[11px]（统一徽章字号），移除 rounded border
   ========================================================================== */
const SIGNAL_KIND_BADGE: Record<string, string> = {
  '实盘': 'ds-badge-up',
  '影子': 'ds-badge-violet',
  '场景': 'ds-badge-indigo',
}

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

// 实盘对照图：BTC K 线（5m/15m）× 预测市场情绪曲线（UP/DOWN 报价）× 实盘订单下单点标记
interface LiveOrderMarker {
  id: string | number
  time: number
  window_start: number
  market_period: string
  channel: string
  channelName: string
  direction: string
  status: string
  averagePrice: number | null
  priceKind: string | null
  amountUsdt: number | null
  win: boolean | null
  pnl: number | null
  errorMessage: string | null
  btcPrice: number | null
}

function LiveOrderChartCard({ orders = [] }: { orders?: Record<string, unknown>[] }) {
  const [p, setP] = useState<'5m' | '15m'>('5m')
  const [klines, setKlines] = useState<BtcKline[]>([])
  const [points, setPoints] = useState<PMPoint[]>([])
  const [hoveredMarker, setHoveredMarker] = useState<LiveOrderMarker | null>(null)

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
    const t = setInterval(load, 20_000)
    return () => clearInterval(t)
  }, [load])

  const hhmm = (t: number) =>
    new Date(t).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

  // 提取图表时间范围
  const klineMinTime = klines.length > 0 ? klines[0].open_time : 0
  const klineMaxTime = klines.length > 0 ? klines[klines.length - 1].open_time + (p === '15m' ? 15 : 5) * 60_000 : 0

  // 将 orders 映射为图表上的点
  const orderMarkers: LiveOrderMarker[] = useMemo(() => {
    if (!orders || orders.length === 0 || klines.length === 0) return []
    const list: LiveOrderMarker[] = []
    for (const o of orders) {
      const per = String(o.market_period ?? '5m')
      if (per !== p) continue

      const ws = typeof o.window_start === 'number' ? o.window_start : 0
      const createdAt = o.created_at ? new Date(String(o.created_at)).getTime() : ws
      const t = createdAt || ws
      if (!t || t < klineMinTime - 15 * 60_000 || t > klineMaxTime + 15 * 60_000) continue

      // 寻找最靠近该时间点的 K 线收盘价
      let nearestBtc: number | null = null
      let minDiff = Infinity
      for (const k of klines) {
        const diff = Math.abs(k.open_time - t)
        if (diff < minDiff) {
          minDiff = diff
          nearestBtc = k.close
        }
      }

      const ver = String(o.signal_version ?? '')
      const info = SIGNAL_INFO[ver]
      const dir = String(o.direction ?? '')
      const stat = String(o.status ?? '')
      const avgP = typeof o.average_price === 'number' ? o.average_price : null
      const amt = o.amount_in != null && Number(o.amount_in) > 0 ? Number(o.amount_in) / 1e18 : null

      list.push({
        id: o.id as (string | number),
        time: t,
        window_start: ws,
        market_period: per,
        channel: ver,
        channelName: info?.name ?? ver,
        direction: dir,
        status: stat,
        averagePrice: avgP,
        priceKind: (o.price_kind as string | null) ?? null,
        amountUsdt: amt,
        win: typeof o.win === 'boolean' ? o.win : null,
        pnl: typeof o.pnl === 'number' ? o.pnl : null,
        errorMessage: (o.error_message as string | null) ?? null,
        btcPrice: nearestBtc,
      })
    }
    return list
  }, [orders, klines, p, klineMinTime, klineMaxTime])

  return (
    <div className="lg:col-span-2">
      <Card title={`实盘行情 × 报价走势 × 下单点对照（${p}，20s 自动刷新）`}>
        <div className="flex items-center justify-between mb-3 text-xs flex-wrap gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            {(['5m', '15m'] as const).map(iv => (
              <button key={iv} onClick={() => setP(iv)}
                className={`px-3 py-1 rounded-sm border font-semibold transition ${
                  p === iv ? 'bg-brand text-white border-brand' : 'bg-card text-ink-80 border-line hover:border-brand'
                }`}>{iv} 周期</button>
            ))}
            <span className="text-ink-55 text-[11px] ml-1">
              <span className="text-positive font-semibold">— UP 报价</span> · <span className="text-negative font-semibold">— DOWN 报价</span> · <span className="text-ink-95 font-semibold">— BTC 价格</span>
            </span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-ink-55 flex-wrap">
            <span className="inline-flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-positive inline-block" /> UP 成交
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full bg-negative inline-block" /> DOWN 成交
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-2.5 h-2.5 rounded-full border border-warning bg-card inline-block" /> 失败单
            </span>
            <span className="text-ink-40">|</span>
            <span>图表范围内共 {orderMarkers.length} 笔订单点（悬浮查看详情）</span>
          </div>
        </div>

        {/* 悬浮下单点详情提示条 */}
        <div className="mb-2 min-h-[30px] px-3 py-1.5 bg-sunken rounded-sm border border-line text-xs flex items-center justify-between flex-wrap gap-2">
          {hoveredMarker ? (
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-ink-55">{new Date(hoveredMarker.time).toLocaleTimeString()}</span>
              <span className="font-semibold text-ink-95">#{String(hoveredMarker.id)} {hoveredMarker.channelName}</span>
              <span className={`px-1.5 py-0.2 rounded-pill font-bold ${hoveredMarker.direction === 'UP' ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'}`}>
                {hoveredMarker.direction}
              </span>
              <span className="font-mono">
                {hoveredMarker.status === 'FILLED' ? (
                  <span className="text-positive font-semibold">已成交</span>
                ) : hoveredMarker.status === 'FAILED' ? (
                  <span className="text-negative font-semibold">失败</span>
                ) : (
                  <span className="text-ink-80">待定</span>
                )}
              </span>
              <span className="text-ink-80 font-mono">
                均价: {hoveredMarker.averagePrice != null ? hoveredMarker.averagePrice.toFixed(2) : '--'}
                {hoveredMarker.priceKind === 'quote' && <span className="text-[10px] text-warning ml-0.5">(报价)</span>}
              </span>
              <span className="text-ink-80 font-mono">
                金额: {hoveredMarker.amountUsdt != null ? `${hoveredMarker.amountUsdt.toFixed(2)}U` : '--'}
              </span>
              <span className="font-mono font-bold">
                {hoveredMarker.win === true ? (
                  <span className="text-positive">赢 (+{hoveredMarker.pnl != null ? hoveredMarker.pnl.toFixed(2) : '0.00'}U)</span>
                ) : hoveredMarker.win === false ? (
                  <span className="text-negative">输 ({hoveredMarker.pnl != null ? hoveredMarker.pnl.toFixed(2) : '-'}U)</span>
                ) : (
                  <span className="text-ink-55">未结算</span>
                )}
              </span>
              {hoveredMarker.errorMessage && (
                <span className="text-ink-55 text-[11px] truncate max-w-[200px]" title={hoveredMarker.errorMessage}>
                  {hoveredMarker.errorMessage}
                </span>
              )}
            </div>
          ) : (
            <span className="text-ink-55 text-[11px]">鼠标移动到图表上的订单圆点可查看下单细节，双图时刻垂直对齐</span>
          )}
        </div>

        <div className="space-y-3">
          {/* 上图：BTC 收盘价曲线 + 下单点在 BTC 上的相对位置 */}
          <div>
            <div className="text-[11px] font-semibold text-ink-80 mb-1 flex justify-between">
              <span>BTC 价格走势与下单入场位置（{p}）</span>
              {klines.length > 0 && (
                <span className="font-mono text-ink-55">最新: ${klines[klines.length - 1].close.toLocaleString()}</span>
              )}
            </div>
            {klines.length > 0 ? (
              <ResponsiveContainer width="100%" height={170}>
                <LineChart
                  data={klines.map(k => ({ t: k.open_time, close: k.close }))}
                  margin={{ top: 6, right: 12, left: 0, bottom: 0 }}
                >
                  <CartesianGrid stroke="var(--line-soft)" />
                  <XAxis
                    dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                    tickFormatter={hhmm}
                    tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }}
                    stroke="var(--line-soft)"
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }}
                    stroke="var(--line-soft)" width={68}
                    tickFormatter={(v: number) => v.toLocaleString()}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                    labelFormatter={t => new Date(t as number).toLocaleString('zh-CN')}
                    formatter={v => [typeof v === 'number' ? v.toLocaleString() : '--', 'BTC 价格']}
                  />
                  <Line dataKey="close" stroke="var(--ink-80)" dot={false} strokeWidth={1.6} isAnimationActive={false} />

                  {/* 悬浮时刻垂直参考线 */}
                  {hoveredMarker && (
                    <ReferenceLine x={hoveredMarker.time} stroke="var(--ink-55)" strokeDasharray="3 3" />
                  )}

                  {/* BTC K线上的订单标记点 */}
                  {orderMarkers.map(m => {
                    if (m.btcPrice == null) return null
                    const isUp = m.direction === 'UP'
                    const isFailed = m.status === 'FAILED'
                    const fill = isFailed ? '#FFFFFF' : (isUp ? 'var(--positive)' : 'var(--negative)')
                    const stroke = isFailed ? 'var(--warning)' : (isUp ? 'var(--positive)' : 'var(--negative)')
                    return (
                      <ReferenceDot
                        key={`btc-dot-${m.id}`}
                        x={m.time}
                        y={m.btcPrice}
                        r={hoveredMarker?.id === m.id ? 6.5 : 4.5}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={2}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHoveredMarker(m)}
                        onMouseLeave={() => setHoveredMarker(null)}
                      />
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-xs text-ink-55 h-10 flex items-center">K 线加载中…（{p}）</div>
            )}
          </div>

          {/* 下图：预测市场情绪曲线（UP/DOWN） + 订单成交均价点 */}
          <div>
            <div className="text-[11px] font-semibold text-ink-80 mb-1 flex justify-between">
              <span>预测市场报价曲线（UP/DOWN）与成交均价点（直观核对入场时机与滑点）</span>
              {points.length > 0 && (
                <span className="font-mono text-ink-55">
                  UP: {points[points.length - 1].up_price?.toFixed(2) ?? '--'} · DOWN: {points[points.length - 1].down_price?.toFixed(2) ?? '--'}
                </span>
              )}
            </div>
            {points.length > 1 ? (
              <ResponsiveContainer width="100%" height={140}>
                <LineChart
                  data={points.map(pt => ({ t: pt.timestamp, up: pt.up_price, down: pt.down_price }))}
                  margin={{ top: 6, right: 12, left: 0, bottom: 0 }}
                >
                  <CartesianGrid stroke="var(--line-soft)" />
                  <XAxis
                    dataKey="t" type="number" domain={['dataMin', 'dataMax']} scale="time"
                    tickFormatter={hhmm}
                    tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }}
                    stroke="var(--line-soft)"
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }}
                    stroke="var(--line-soft)" width={68}
                    tickFormatter={(v: number) => v.toFixed(2)}
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                    labelFormatter={t => new Date(t as number).toLocaleString('zh-CN')}
                    formatter={(v, name) => [typeof v === 'number' ? v.toFixed(3) : '--', name === 'up' ? 'UP 报价' : 'DOWN 报价']}
                  />
                  <Line dataKey="up" stroke="var(--positive)" dot={false} strokeWidth={1.6} connectNulls isAnimationActive={false} />
                  <Line dataKey="down" stroke="var(--negative)" dot={false} strokeWidth={1.6} connectNulls isAnimationActive={false} />

                  {/* 悬浮时刻垂直参考线 */}
                  {hoveredMarker && (
                    <ReferenceLine x={hoveredMarker.time} stroke="var(--ink-55)" strokeDasharray="3 3" />
                  )}

                  {/* 情绪曲线上的成交均价标记点 */}
                  {orderMarkers.map(m => {
                    if (m.averagePrice == null) return null
                    const isUp = m.direction === 'UP'
                    const isFailed = m.status === 'FAILED'
                    const fill = isFailed ? '#FFFFFF' : (isUp ? 'var(--positive)' : 'var(--negative)')
                    const stroke = isFailed ? 'var(--warning)' : (isUp ? 'var(--positive)' : 'var(--negative)')
                    return (
                      <ReferenceDot
                        key={`quote-dot-${m.id}`}
                        x={m.time}
                        y={m.averagePrice}
                        r={hoveredMarker?.id === m.id ? 6.5 : 4.5}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={2}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHoveredMarker(m)}
                        onMouseLeave={() => setHoveredMarker(null)}
                      />
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-xs text-ink-55 h-10 flex items-center">情绪曲线加载中…（{p}）</div>
            )}
          </div>
        </div>
      </Card>
    </div>
  )
}

// 悬浮快速下单（右下角 FAB，2026-08-28）：原「人工测试单」卡改为悬浮。
// 表单状态（amount/side/busy/result）自持有；报价/倒计时/钱包余额由 LiveTradeTab 15s 轮询传入。
function TestTradeFab({ quote, remainSec, urgent, wallet, refresh }: {
  quote: Record<string, unknown> | null
  remainSec: number | null
  urgent: boolean
  wallet: Record<string, unknown> | null
  refresh: () => void
}) {
  const [open, setOpen] = useState(false)
  const [amount, setAmount] = useState('1')
  const [side, setSide] = useState<'DOWN' | 'UP'>('DOWN')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)

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

  const resultFilled = result?.status === 'FILLED'

  return (
    <>
      {/* 悬浮按钮（右下角常驻） */}
      <button
        onClick={() => setOpen(o => !o)}
        className="fixed bottom-6 right-6 z-40 px-4 py-3 rounded-pill font-bold text-white text-sm transition bg-brand hover:bg-brand-hover"
        title="人工测试单（真实下单）"
      >
        {busy ? '下单中…' : '⚡ 下单'}
      </button>

      {/* 展开卡片：点遮罩关闭 */}
      {open && (
        <>
          <div className="fixed inset-0 z-40 bg-ink-black/10" onClick={() => setOpen(false)} />
          <div className="fixed bottom-24 right-6 z-50 w-[340px] max-w-[92vw] bg-card rounded-card border border-line p-4 space-y-3 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-sm font-bold text-ink-95">⚡ 快速下单（真实订单）</span>
              <button onClick={() => setOpen(false)} className="text-ink-55 hover:text-ink-80 text-lg leading-none px-1">✕</button>
            </div>
            {quote == null || quote.stale ? (
              <div className="text-xs text-ink-55 rounded-pill bg-sunken px-2 py-1.5">报价不可用（等待 15s 采样器…）</div>
            ) : (
              <div className={`flex items-center justify-between rounded-sm px-2 py-1.5 text-xs ${urgent ? 'bg-negative-soft text-negative' : 'bg-sunken text-ink-80'}`}>
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
              <label className="text-ink-55 shrink-0">金额 (USDT)</label>
              <input
                type="number" min={0.1} max={50} step={0.5} value={amount}
                onChange={e => setAmount(e.target.value)}
                className="ds-input ds-input-numeric w-24"
              />
              <span className="text-xs text-ink-55">0.1~50</span>
            </div>
            {/* 金额预设：百分比按预测钱包余额计算（clamp 0.1~50），固定额直填（100U 超硬上限禁用） */}
            <div className="flex items-center gap-1.5 flex-wrap text-xs">
              <span className="text-ink-55 shrink-0">按余额</span>
              {[2, 5, 10, 20].map(pct => {
                const bal = typeof wallet?.prediction_usdt_free === 'number'
                  ? wallet.prediction_usdt_free as number : null
                const disabled = bal == null
                return (
                  <button key={pct} disabled={disabled}
                    title={disabled ? '预测钱包余额不可查（等余额端点收敛）' : `${pct}% × ${bal!.toFixed(2)}U`}
                    onClick={() => setAmount(String(Math.min(50, Math.max(0.1, +(bal! * pct / 100).toFixed(2)))))}
                    className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 hover:border-brand disabled:opacity-40"
                  >{pct}%</button>
                )
              })}
              <span className="text-ink-55 shrink-0 ml-2">固定</span>
              {[1, 2, 5, 10, 20, 50, 100].map(u => (
                <button key={u} disabled={u > 50} title={u > 50 ? '超单笔硬上限 50 USDT' : `${u} USDT`}
                  onClick={() => setAmount(String(u))}
                  className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 hover:border-brand disabled:opacity-40"
                >{u}U</button>
              ))}
            </div>
            <div className="flex items-center gap-3">
              <span className="text-ink-55 shrink-0">方向</span>
              <button
                onClick={() => setSide('DOWN')}
                className={`px-3 py-1 text-sm font-semibold rounded-full border ${side === 'DOWN' ? 'bg-negative-soft text-negative border-negative' : 'bg-card text-ink-55 border-line'}`}
              >↓ 看跌</button>
              <button
                onClick={() => setSide('UP')}
                className={`px-3 py-1 text-sm font-semibold rounded-full border ${side === 'UP' ? 'bg-positive-soft text-positive border-positive' : 'bg-card text-ink-55 border-line'}`}
              >↑ 看涨</button>
            </div>
            <button
              onClick={handleTestTrade}
              disabled={busy}
              className={`ds-btn-primary w-full py-2 font-bold ${busy ? 'cursor-wait' : ''}`}
            >
              {busy ? '下单中…' : '下单（真实订单）'}
            </button>
            {result && (
              <div className={`p-2 rounded-sm text-xs ${resultFilled ? 'bg-positive-soft text-positive' : 'bg-negative-soft text-negative'}`}>
                <div className="font-bold">{resultFilled ? '✓ 已成交 FILLED' : `✗ ${String(result.status ?? '未执行')}`}</div>
                {result.order_id != null && <div>订单号: {String(result.order_id)}</div>}
                {result.direction != null && <div>方向: {String(result.direction)}</div>}
                {result.average_price != null && <div>成交均价: <span className="num">{String(result.average_price)}</span></div>}
                {result.error_message != null && <div>{String(result.error_message)}</div>}
                {result.error != null && <div>{String(result.error)}</div>}
              </div>
            )}
            <p className="text-xs text-ink-55">
              与信号实盘同链路（占位→报价→下单→落库）；同一 5m 窗口至多一单。
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
const ORDERS_PAGE_SIZE = 20

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
  const [page, setPage] = useState(1)

  // 失败单严格分流：主 Tab 只留非 FAILED（FILLED / PENDING / 未知状态都在），失败 Tab 只留 FAILED
  const activeOrders = orders.filter(o => String(o.status ?? '') !== 'FAILED')
  const failedOrders = orders.filter(o => String(o.status ?? '') === 'FAILED')
  const base = orderTab === 'active' ? activeOrders : failedOrders

  const channels = Array.from(new Set(
    base.map(o => String(o.signal_version ?? '')).filter(v => v && v !== '--'))).sort()
  const filtered = base.filter(o => {
    if (orderTab === 'active' && fStatus !== 'ALL' && String(o.status ?? '') !== fStatus) return false
    if (fChannel !== 'ALL' && String(o.signal_version ?? '') !== fChannel) return false
    if (fDirection !== 'ALL' && String(o.direction ?? '') !== fDirection) return false
    return true
  })
  const settledOrders = filtered.filter(o => o.settled_at != null)
  const settledCount = settledOrders.length
  const totalPnl = settledOrders.reduce(
    (s, o) => s + (typeof o.pnl === 'number' ? (o.pnl as number) : 0), 0)
  const filterActive = fStatus !== 'ALL' || fChannel !== 'ALL' || fDirection !== 'ALL'
  const selectCls = 'px-1.5 py-0.5 border border-line rounded-sm text-xs text-ink-80 bg-card'
  // 分页：筛选变化时回到第 1 页；超出范围时钳位（避免删数据后空白页）
  useEffect(() => { setPage(1) }, [fStatus, fChannel, fDirection])
  // 切 Tab 必须清筛选：否则主 Tab 选的 FILLED 会让失败 Tab 直接空白
  useEffect(() => {
    setFStatus('ALL'); setFChannel('ALL'); setFDirection('ALL'); setPage(1)
  }, [orderTab])
  const totalPages = Math.max(1, Math.ceil(filtered.length / ORDERS_PAGE_SIZE))
  const safePage = Math.min(page, totalPages)
  const pageRows = filtered.slice((safePage - 1) * ORDERS_PAGE_SIZE, safePage * ORDERS_PAGE_SIZE)

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
            <HelpHint text={`${ver}（${info.kind}${info.retired ? '·已退役' : ''}）：${info.desc}`} />
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

  const pager = filtered.length > ORDERS_PAGE_SIZE && (
    <div className="flex items-center justify-end gap-1 mt-2 text-xs">
      <button
        disabled={safePage <= 1} onClick={() => setPage(safePage - 1)}
        className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 disabled:opacity-40"
      >上一页</button>
      {Array.from({ length: totalPages }, (_, i) => i + 1).map(n => (
        <button
          key={n} onClick={() => setPage(n)}
          className={`px-2 py-0.5 rounded-pill border ${n === safePage ? 'bg-brand text-white border-brand' : 'bg-card text-ink-80 border-line hover:border-brand'}`}
        >{n}</button>
      ))}
      <button
        disabled={safePage >= totalPages} onClick={() => setPage(safePage + 1)}
        className="px-2 py-0.5 rounded-pill border border-line bg-card text-ink-80 disabled:opacity-40"
      >下一页</button>
      <span className="text-ink-55 ml-1">每页 {ORDERS_PAGE_SIZE} 条</span>
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
          {filterActive && (
            <button
              onClick={() => { setFStatus('ALL'); setFChannel('ALL'); setFDirection('ALL') }}
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
            <tr className="text-left text-ink-55 border-b border-line-soft">
              <th className="py-1 pr-2">时间</th>
              <th className="py-1 pr-2" title="本单真正下注的市场周期时段（window_start 起，5m/15m 窗口）">目标周期</th>
              <th className="py-1 pr-2">通道</th>
              <th className="py-1 pr-2">方向</th>
              <th className="py-1 pr-2">状态</th>
              <th className="py-1 pr-2">结果</th>
              <th className="py-1 pr-2" title="币安实际成交均价；标「报价」的是下单时的报价/委托价（PENDING 未成交/护栏弃单时 filledUsdtAmount=0 但币安仍回 price），不是成交价，不能用来核对盈亏">均价</th>
              <th className="py-1 pr-2">金额 (USDT)</th>
              <th className="py-1 pr-2">盈亏</th>
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
            <tr className="text-left text-ink-55 border-b border-line-soft">
              <th className="py-1 pr-2">时间</th>
              <th className="py-1 pr-2" title="本单真正下注的市场周期时段（window_start 起，5m/15m 窗口）">目标周期</th>
              <th className="py-1 pr-2">通道</th>
              <th className="py-1 pr-2">方向</th>
              <th className="py-1 pr-2" title="下单时的报价/委托价（失败单 FOK 未成交，filledUsdtAmount=0 但币安仍回 price），不是成交价">触发报价</th>
              <th className="py-1 pr-2" title="计划投入金额；失败单未提交到币安时落库为 0，显示 --">计划金额</th>
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
  const src = fundOrders.length > 0 ? fundOrders : orders

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
// 盈利趋势（2026-08-31）：每实盘通道已结算订单的累计盈亏曲线
// 数据源 GET /api/live/pnl-curve（trade_orders 按通道分组逐单累计 pnl）
// ============================================================

interface PnlPoint { n: number; t: number; pnl: number; cum: number; win: boolean }
interface PnlChannel {
  channel: string
  display_name: string
  family: string | null
  market_period: string | null
  enabled: boolean
  settled_count: number
  win_count: number
  win_rate: number | null
  total_pnl: number
  points: PnlPoint[]
}
interface PnlCurveData {
  channels: PnlChannel[]
  total: { settled_count: number; total_pnl: number; win_rate: number | null }
}

// 通道曲线色板（按 total_pnl 排序后取色；12 色，通道数 >12 时按 idx % 12 循环复用）
/* 逐通道盈亏曲线色 = chart-1..10。通道数 >10 时循环取色，
   第 11 条起靠渲染层的「淡色 + 细线」区分；表格图例圆点同步降透明度以保持对应。 */
const PNL_COLORS = Array.from({ length: 10 }, (_, i) => `var(--chart-${i + 1})`)

// 曲线指标切换：数据点已含 pnl/win，胜率趋势 = 截至第 n 笔的滚动胜率（前端派生）
type PnlMetric = 'cum' | 'rate' | 'pnl'
const PNL_METRICS: { key: PnlMetric; label: string }[] = [
  { key: 'cum', label: '累计盈亏' },
  { key: 'rate', label: '胜率趋势' },
  { key: 'pnl', label: '单笔盈亏' },
]

function PnlCurveCard() {
  const [data, setData] = useState<PnlCurveData | null>(null)
  const [onlyEnabled, setOnlyEnabled] = useState(false)
  const [metric, setMetric] = useState<PnlMetric>('cum')
  useEffect(() => {
    const load = () => api.getLivePnlCurve().then(setData).catch(() => {})
    load()
    const timer = setInterval(load, 30000)
    return () => clearInterval(timer)
  }, [])

  const allChans = data?.channels ?? []
  const chans = allChans.filter(c => !onlyEnabled || c.enabled)
  // 按笔序对齐：第 n 行 = 各通道第 n 笔已结算单的当前指标值（不足为 null，线段断开）
  const maxLen = chans.reduce((m, c) => Math.max(m, c.points.length), 0)
  const winSoFar: Record<string, number> = {}
  for (const c of chans) winSoFar[c.display_name] = 0
  const rows = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number | null> = { n: i + 1 }
    for (const c of chans) {
      const p = c.points[i]
      if (!p) { row[c.display_name] = null; continue }
      if (p.win) winSoFar[c.display_name] += 1
      row[c.display_name] = metric === 'rate'
        ? Math.round((winSoFar[c.display_name] / (i + 1)) * 1000) / 10
        : metric === 'pnl' ? p.pnl : p.cum
    }
    return row
  })
  const sumPnl = chans.reduce((s, c) => s + c.total_pnl, 0)
  const sumCount = chans.reduce((s, c) => s + c.settled_count, 0)
  const sumWin = chans.reduce((s, c) => s + c.win_count, 0)
  const statBox = 'rounded-card border border-line bg-card px-3 py-2'
  const metricLabel = PNL_METRICS.find(m => m.key === metric)?.label ?? '累计盈亏'

  return (
    <Card title={`盈利趋势（每通道${metricLabel}）`}>
      <div className="flex items-center justify-between mb-3 gap-2 flex-wrap">
        <div className="flex items-center gap-1 flex-wrap">
          {PNL_METRICS.map(mt => (
            <button key={mt.key} onClick={() => setMetric(mt.key)}
              className={`px-2 py-0.5 rounded-full text-[11px] border font-bold transition-colors ${metric === mt.key
                ? 'bg-brand text-white border-brand'
                : 'bg-card text-ink-55 border-line hover:border-brand'}`}>
              {mt.label}
            </button>
          ))}
          <label className="ml-2 flex items-center gap-1.5 text-xs text-ink-80 cursor-pointer select-none">
            <input type="checkbox" checked={onlyEnabled}
              onChange={e => setOnlyEnabled(e.target.checked)}
              className="accent-brand" />
            仅当前开启实盘的通道
          </label>
        </div>
        <span className="text-[10px] text-ink-55">
          口径：当前注册实盘通道的已结算成交单（本地估算 pnl，赢=股数−成本，输=−投入）；
          不含 manual_test / 已退役通道历史 / NOISE·EXPIRED / 失败未成交——与「资金变化」面板的全量订单口径不同，勿直接对比；胜率趋势=截至该笔的滚动胜率；每 30s 刷新
        </span>
      </div>
      {allChans.length === 0 ? (
        <p className="text-sm text-ink-55 py-6 text-center">
          暂无已结算的实盘订单——开启通道实盘并等订单结算后生成曲线
        </p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2 mb-3 text-xs">
            <div className={statBox}>
              <div className="text-ink-55 mb-0.5">累计盈亏（USDT）</div>
              <div className={`text-base font-bold font-mono ${sumPnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                {sumPnl >= 0 ? '+' : ''}{sumPnl.toFixed(2)}
              </div>
            </div>
            <div className={statBox}>
              <div className="text-ink-55 mb-0.5">已结算单数</div>
              <div className="text-base font-bold font-mono text-ink-95">{sumCount}</div>
            </div>
            <div className={statBox}>
              <div className="text-ink-55 mb-0.5">总胜率</div>
              <div className="text-base font-bold font-mono text-ink-95">
                {sumCount > 0 ? `${((sumWin / sumCount) * 100).toFixed(0)}%` : '--'}
                <span className="text-[10px] font-normal text-ink-55 ml-1">{sumWin}胜/{sumCount - sumWin}负</span>
              </div>
            </div>
            <div className={statBox}>
              <div className="text-ink-55 mb-0.5">有单通道</div>
              <div className="text-base font-bold font-mono text-ink-95">
                {chans.length}<span className="text-[10px] font-normal text-ink-55 ml-1">/ {allChans.length}</span>
              </div>
            </div>
          </div>
          {maxLen > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="var(--line-soft)" />
                <XAxis dataKey="n" tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)"
                  label={{ value: '已结算单序号', position: 'insideBottomRight', offset: -2, fontSize: 10, fill: 'var(--ink-55)' }} />
                <YAxis tick={{ fontSize: 11, fill: 'var(--ink-55)', fontFamily: 'var(--font-stack-mono)' }} stroke="var(--ink-55)" width={52}
                  domain={metric === 'rate' ? [0, 100] : undefined}
                  tickFormatter={(v: number) => metric === 'rate' ? `${v}%` : v.toFixed(1)} />
                <Tooltip
                  contentStyle={TOOLTIP_STYLE} labelStyle={TOOLTIP_LABEL_STYLE} itemStyle={TOOLTIP_ITEM_STYLE}
                  formatter={(v) => metric === 'rate'
                    ? [`${Number(v).toFixed(1)}%`, undefined]
                    : [`${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(2)} USDT`, undefined]}
                  labelFormatter={(n) => `第 ${n} 笔已结算单`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine y={metric === 'rate' ? 50 : 0} stroke="var(--line)" />
                {chans.map((c, i) => (
                  <Line key={c.channel} type="monotone" dataKey={c.display_name}
                    stroke={PNL_COLORS[i % PNL_COLORS.length]}
                    strokeWidth={i >= 10 ? 1.5 : 2} strokeOpacity={i >= 10 ? 0.55 : 1}
                    dot={false} connectNulls isAnimationActive={false} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-ink-55 py-4 text-center">所选通道暂无已结算订单</p>
          )}
          <div className="overflow-x-auto max-h-52 overflow-y-auto mt-2">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="text-left text-ink-55 border-b border-line-soft">
                  <th className="py-1 pr-2">通道</th>
                  <th className="py-1 pr-2">状态</th>
                  <th className="py-1 pr-2 text-right">已结算</th>
                  <th className="py-1 pr-2 text-right">胜率</th>
                  <th className="py-1 pr-2 text-right">累计盈亏 (USDT)</th>
                </tr>
              </thead>
              <tbody>
                {chans.map(c => {
                  const info = SIGNAL_INFO[c.channel]
                  const ci = chans.findIndex(x => x.channel === c.channel)
                  const color = PNL_COLORS[ci % PNL_COLORS.length]
                  // 圆点与曲线同序淡化，否则第 11 条起的图例色块会与前三条撞色
                  const faded = ci >= 10
                  return (
                    <tr key={c.channel} className="border-b border-line-soft">
                      <td className="py-1 pr-2">
                        <span className="inline-block w-2 h-2 rounded-full mr-1.5 shrink-0" style={{ background: color, opacity: faded ? 0.55 : 1 }} />
                        <span className="text-ink-95">{info?.name ?? c.display_name}</span>
                      </td>
                      <td className="py-1 pr-2">
                        <span className={`px-1.5 py-0.5 text-[10px] font-bold rounded-pill border ${c.enabled ? 'bg-positive-soft text-positive border-positive' : 'bg-sunken text-ink-55 border-line'}`}>
                          {c.enabled ? '实盘中' : '已停火'}
                        </span>
                      </td>
                      <td className="py-1 pr-2 text-right font-mono text-ink-80">{c.settled_count}</td>
                      <td className="py-1 pr-2 text-right font-mono text-ink-80">
                        {c.win_rate == null ? '--' : `${(c.win_rate * 100).toFixed(0)}%`}
                      </td>
                      <td className={`py-1 pr-2 text-right font-mono font-bold ${c.total_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                        {c.total_pnl >= 0 ? '+' : ''}{c.total_pnl.toFixed(2)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  )
}

// 右侧抽屉（2026-08-28 从主布局移入；2026-08-29 双 tab；2026-08-31 增至三 tab：
// K 线对照 / 资金变化 / 盈利趋势）：
// LiveChartCard 常驻挂载保持 30s 轮询数据连续，抽屉仅控制可视；各面板始终渲染、
// 用 hidden 切换，避免切 tab 时 LiveChartCard 卸载导致采样中断。
function ChartDrawer({ orders }: { orders: Record<string, unknown>[] }) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<'chart' | 'funds' | 'pnl'>('chart')
  const tabBtn = (t: 'chart' | 'funds' | 'pnl', label: string) => (
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
          title="查看 K 线对照 / 资金变化 / 盈利趋势"
        >
          📈 面板
        </button>
      )}

      <div
        className={`fixed top-0 right-0 h-screen w-[760px] max-w-[94vw] bg-card border-l border-line z-50 flex flex-col transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="px-4 py-2.5 border-b border-line bg-card flex items-center justify-between shrink-0 gap-2">
          <div className="flex items-center gap-2">
            {tabBtn('chart', '📈 K 线对照')}
            {tabBtn('funds', '💰 资金变化')}
            {tabBtn('pnl', '📊 盈利趋势')}
          </div>
          <button onClick={() => setOpen(false)} className="text-ink-55 hover:text-ink-80 text-lg leading-none px-1">✕</button>
        </div>
        <div className="flex-1 min-h-0 overflow-auto p-3">
          <div className={tab === 'chart' ? '' : 'hidden'}><LiveOrderChartCard orders={orders} /></div>
          <div className={tab === 'funds' ? '' : 'hidden'}><FundsFlowCard orders={orders} /></div>
          <div className={tab === 'pnl' ? '' : 'hidden'}><PnlCurveCard /></div>
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

  const walletErr = (wallet as { error?: string } | null)?.error
  const regTs = wallet?.registered_time as number | undefined
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
    <>
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* 2026-08-28 布局改造：下单→右下角悬浮 FAB；订单/K 线→右侧抽屉；主区仅留账户状态 + 信号概览 */}
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
          <div className="flex justify-between gap-2 items-center">
            <span className="text-ink-55 shrink-0 flex items-center">
              可领取奖金
              <HelpHint text="赢单的奖金以获胜 token 形式留在链上钱包，不会自动变成 USDT；需要调官方 batch-redeem 赎回后才入预测钱包余额。赢单后记得来这里领取。" />
            </span>
            {Number(redeemable?.claimable_count ?? 0) > 0
              ? <span className="flex items-center gap-2">
                  <span className="font-mono font-semibold text-warning">
                    {String(redeemable?.claimable_count)} 个 token 待领取
                    {redeemable?.wallet_source === 'degraded' && <span className="text-ink-55 font-normal">（钱包查询降级，含本地兑底）</span>}
                  </span>
                  <button
                    onClick={handleRedeem} disabled={redeeming}
                    className="px-3 py-1 text-xs font-semibold rounded-pill bg-warning text-white disabled:opacity-50"
                  >{redeeming ? '领取中…' : '领取奖金'}</button>
                </span>
              : <span className="text-ink-55">--</span>}
          </div>
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
              <HelpHint text="10 个信号通道独立实盘：每通道独立开关/单笔金额/日限/执行价护栏。命中信号即下真实订单（FOK），重启回落 LIVE_CHANNELS_JSON 配置。" />
            </span>
            {liveChannels.length > 0
              ? <span className="flex items-baseline gap-1.5 text-right">
                  <span className={enabledCount > 0 ? 'text-positive font-semibold' : 'text-warning font-semibold'}>
                    {enabledCount > 0 ? `${enabledCount}/${String(liveChannels.length)} 通道开启` : '全部关闭（不开火）'}
                  </span>
                  <span className="text-[10px] text-ink-55">
                    默认 {String(liveDefaults.amount_usdt ?? '--')}U/单 · 在途任务 {String(live?.pending_tasks ?? 0)}
                  </span>
                </span>
              : <span className="text-ink-55">未装配（启动异常，详见后端日志）</span>}
          </div>
          {/* 通道管理面板：每通道一行「信号名/护栏/今日成交/累计开火/金额输入+保存/开关」，独立 toggle + 金额热调 */}
          {liveChannels.length > 0 && (
            <div className="bg-sunken border border-line rounded-sm p-2 space-y-1">
              {liveChannels.map(ch => {
                const info = SIGNAL_INFO[ch.channel]
                const draft = amountDrafts[ch.channel]
                const dirty = draft != null && draft !== String(ch.amount_usdt)
                return (
                  <div
                    key={ch.channel}
                    className={`flex items-center gap-2 px-2 py-1.5 rounded-sm border text-xs ${ch.enabled ? 'border-positive bg-positive-soft' : 'border-line bg-card'}`}
                  >
                    <span className="flex items-center gap-1.5 min-w-0 flex-1">
                      <span className={`ds-badge shrink-0 ${SIGNAL_KIND_BADGE[info?.kind ?? '影子']}`}>{info?.kind ?? '影子'}</span>
                      <span className="text-ink-95 font-medium truncate">{info?.name ?? ch.display_name}</span>
                      <span className="text-[10px] text-ink-55 font-mono shrink-0 hidden md:inline">{ch.channel}</span>
                      <HelpHint text={info?.desc ?? ch.display_name} />
                    </span>
                    <span className="shrink-0 flex items-center gap-1">
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
                    <span className="shrink-0 flex items-center gap-1">
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
                      onClick={() => handleChannelToggle(ch)}
                      disabled={togglingLive}
                      className={`shrink-0 px-2 py-0.5 rounded-pill font-bold text-white disabled:opacity-50 ${ch.enabled ? 'bg-negative' : 'bg-positive'}`}
                    >{ch.enabled ? '停火' : '开火'}</button>
                  </div>
                )
              })}
              <p className="text-[10px] text-ink-55 pt-0.5">
                金额/护栏输入后点「存」热调（立即生效，重启不丢）；护栏=报价均价上限（≥ 即弃单含贴线），「↺」回落通道预设；开关各通道独立互不影响。5m 通道随窗触发，15m 场景通道次周期开盘入场。
              </p>
            </div>
          )}
        </div>
      </Card>
      </div>

      {/* 实盘订单与行情对照图表区（位于信号实盘通道与订单记录之间） */}
      <div className="lg:col-span-2">
        <LiveOrderChartCard orders={orders} />
      </div>

      {/* 最近订单：主区账户状态下方（2026-08-28 用户要求回归页面展示） */}
      <div className="lg:col-span-2">
        <OrdersCard orders={orders} syncing={syncing} syncResult={syncResult} onSyncBinance={handleSyncBinance} />
      </div>
    </div>

    {/* 悬浮与抽屉：下单 FAB + 右侧抽屉（K 线对照 / 资金变化双 tab，2026-08-29） */}
    <TestTradeFab quote={quote} remainSec={remainSec} urgent={urgent} wallet={wallet} refresh={refresh} />
    <ChartDrawer orders={orders} />
    </>
  )
}

// ============================================================
// Main App
// ============================================================

const TAB_IDS = ['market', 'agent', 'monitor', 'analysis', 'live'] as const
type TabId = (typeof TAB_IDS)[number]

/* tab 中文名。此前内联在 5 个按钮里重复 5 次，收敛为单一事实源。 */
const TAB_LABELS: Record<TabId, string> = {
  market: '市场情绪',
  agent: 'Agent 自进化',
  monitor: '运行监控',
  analysis: '信号分析',
  live: '实盘交易',
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
// 运行监控 Tab（GET /api/agent/health，30s 轮询）
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
  x4_v1: { label: 'X4 情绪错位→DOWN', color: 'var(--chart-1)' },
  quote_momentum_v1: { label: 'A 报价动量→DOWN', color: 'var(--chart-2)' },
  quote_contrarian_v1: { label: 'B 报价反向→DOWN', color: 'var(--chart-3)' },
  krev_a_v1: { label: 'KREV-A 反转→UP', color: 'var(--chart-4)' },
  krev_b_v1: { label: 'KREV-B 反转→UP', color: 'var(--chart-5)' },
  hm_touch_down_v1: { label: 'HM 上吊线反弹→DOWN', color: 'var(--chart-6)' },
  hm_touch_down_v2: { label: 'HM 上吊线反弹v2→DOWN', color: 'var(--chart-7)' },
  rev_p1_v1: { label: 'P1 连跌弱阴反转→UP', color: 'var(--chart-8)' },
  rev_p2_v1: { label: 'P2 连涨弱阳反转→DOWN', color: 'var(--chart-9)' },
  s5_deep_z20_v1: { label: 'S5深档 z5≤-20bp→DOWN', color: 'var(--chart-10)' },
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
  // 2026-09-07/08 首触反转族（专用表 firsthit_shadow_signals，影子+实盘通道）
  firsthit_down_v1: { label: '首触G0 基底 q∈(0.005,0.1]→DOWN', color: 'var(--chart-3)' },
  firsthit_down_body_v1: { label: '首触G1 body_r≤0.35→DOWN', color: 'var(--chart-4)' },
  firsthit_down_chg_v1: { label: '首触G3 chg≤+2.82bp→DOWN', color: 'var(--chart-5)' },
  firsthit_down_g4_v1: { label: '首触G4 交互 chg≤2.82∧body≤0.35→DOWN', color: 'var(--chart-2)' },
  firsthit_down_g7_v1: { label: '首触G7 基底 body≤0.35∧wick=1→DOWN', color: 'var(--chart-6)' },
  g7_streak_v1: { label: '首触G7 非强连阳 streak≤1→DOWN', color: 'var(--chart-7)' },
  g7_wick20_v1: { label: '首触G7 长上影 wick≥2bp→DOWN', color: 'var(--chart-8)' },
  g7_strict_v1: { label: '首触G7 严格版 streak≤1∧wick≥1.5bp→DOWN', color: 'var(--chart-9)' },
  g7_q05_v1: { label: '首触G7 深折价 q≤0.05→DOWN', color: 'var(--chart-10)' },
  g7_t270_v1: { label: '首触G7 非极晚 t≤270s→DOWN', color: 'var(--chart-1)' },
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
  krev_a_v1: 'K 线反转 A：5m K 线反转形态（看跌衰竭形态）→ 押 UP 的影子信号，仅记录不下单。',
  krev_b_v1: 'K 线反转 B：5m K 线反转形态变体（下影线反转）→ 押 UP 的影子信号，仅记录不下单。',
  hm_touch_down_v1: '【2026-09-04 已退役：信息速率≈0（720d 触发 0.04~0.06 次/天，凑满 n=100 需 4~7 年），影子期无统计意义】HM 上吊线反弹入场：弱收盘上吊线（15m 小实体+深下影+贴 20 根高位+CLV≤0.75）→ 次 15m 周期内 10 分钟里若先反弹触及开盘价+0.25×ATR，按触及时刻真实报价记录押 DOWN（影子，不下单）。先破 −0.25×ATR、迟到触及或未触及均放弃，仅触及样本进胜率统计。720d 回测触价收跌率 58.7% vs 市场隐含 47.1%（n=46，覆盖率~36%），探索性发现，影子期即前向验证。',
  hm_touch_down_v2: '【2026-09-04 已退役：同 v1，720d 门禁后触发仅 78 次，信息速率不足以前向验证】HM 上吊线反弹 v2：v1 基础上叠加触发时点门禁——排除下跌段（过去 24h 跌≥1%）与低波环境（ATR 低于近 24h 中位数 80%）。720d 切片分析发现这两类样本负边际（下跌段 41.7% / 低波 25%），门禁后触发 78、触价 29、收跌率 69.0%。属后验切片假设（非预注册），与 v1 并行双行采集，影子期前向验证决定是否保留。',
  legacy: 'pattern_type 为空的历史信号：胜负按 side 映射（high→DOWN 赢 / low→UP 赢），用于对齐早期统计口径。',
  rev_p1_v1: '反转 P1：15m 连跌 4 根 + 弱阴收盘（贴最低，close_pos≤0.15）+ 成交量正常（[1.0,1.5)×20 根均量）→ 押次根 15m 收阳 UP 的影子信号，仅记录不下单。rev_common 几何口径实时重放，次根收盘按 direction 结算；与 KREV 共表 kline_shadow_signals（version 隔离）。720d 回测胜率 62.0% / oos 63.9%（EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1）。',
  rev_p2_v1: '反转 P2：15m 连涨 5 根 + 弱阳收盘（贴最高，close_pos≥0.85）→ 押次根 15m 收阴 DOWN 的影子信号，仅记录不下单。rev_common 几何口径实时重放，次根收盘按 direction 结算；与 KREV 共表 kline_shadow_signals（version 隔离）。720d 回测胜率 62.4% / oos 61.3%（EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1）。',
  s5_deep_z20_v1: 'S5 深档：S1 多头耗尽信号 +5min 回落确认，且回落幅度 z5=c5_close/anchor−1≤−20bp（深档）→ 按 +5min 时刻真实 15m DOWN 报价记录押 DOWN 的影子信号，仅记录不下单，次周期 15m 收阴判赢。落 pattern_shadow_signals（entry_state=TOUCHED，借用 HM 结算器）。720d 回测~91.3%（深档样本 EV 偏乐观、含机械成分），影子期即前向验证。',
  quote_momentum_v3: '【2026-09-04 已退役：修正未来函数后门禁效应≈0（+3.8pp，CI 重叠），不值得占实盘额度；随 momentum 族整体下线】报价动量 v3：在 v1（触发后 90~120s DOWN 报价 q∈[0.69,0.75)）基础上叠加“非连涨”门禁——用最后已收 15m（触发时刻所属 15m 的前一根，严格防未来函数）判定 close[j]≤close[j−1] 才落表 → 押 DOWN 的影子信号，落 misalignment_signals，按报价 edge 结算。回测修正未来函数后 80.2% vs 连涨 76.4%（+3.8pp，CI 重叠、门禁效应≈0），影子用于前向验证门禁是否真实有效。',
  nb_zschamp_15m_v1: 'nextbar 15m冠军：zscore_10≤-1.651 ∧ zscore_5≤-1.538 ∧ ret_3≤-0.00395（深超卖+急跌+卖盘衰竭）→ 押次根 15m 收阳 UP 的影子信号，仅记录不下单。源自 H=1 方向研究 converge_registry L3 ROBUST（holdout P(up_1)=61.96% n=368，月一致性 0.958 / walk-forward 1.00）；build_feature_matrix+condition_mask 实时重放冻结条件原文，与 KREV/反转共表 kline_shadow_signals（version+timeframe 隔离）。720d 次根收阳 58.92%（2006 触发，EV 按目标窗开盘后首次轮询的真实报价前向现算：赢 0.98/q−1 / 输 −1）。',
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
  firsthit_down_v1: '首触反转 G0 基底（对照组）：5m 窗内 DOWN token 报价**首次**进入 (0.005, 0.1]（深折价，市场判定几乎不会跌）的采样点 → 按该时刻真实报价买 DOWN，押注最终结算 DOWN =「反转」的影子信号，仅记录不下单。触发时刻特征严格 ex-ante（只读 ≤触发时刻采样）：chg_bps（BTC 相对开盘涨跌）、body_r（|btc@触−开盘|/路径 max−min 归一实体）、npts（路径采样点数，<8 整窗不落表=路径太稀疏特征不可信）。落专用表 firsthit_shadow_signals，G1/G3 为 G0 的纯子集（同表 version 隔离、全特征落库，交叉门 G4=body∧chg 可事后重构）。44d 回测基底 EV +0.22（日聚类 CI[+0.10,+0.35]），EV 按逐事件真实触发价现算：赢 0.98/q−1 / 输 −1（禁用任何均值/分位代理）。预注册裁决（4 周前向）：若 G0 前向 EV 日聚类 CI **上界** < 0 → DOWN 侧 edge 消失，整族否决重来。',
  firsthit_down_body_v1: '首触反转 G1 小实体：G0 基底 + body_r ≤ 0.35 门禁（触发时刻 BTC 相对开盘的净位移只占窗内路径振幅的 ≤35%，即价格来回震荡而非单边跑出去）→ 按首触时刻真实 DOWN 报价买 DOWN 的影子信号，仅记录不下单。研究口径：45 维条件扫描中 body_r≤0.35 是唯一扛住 FDR 多重校正的形态门（q=0.007，logit 控 t+t² 后 β=+1.28 p=0.000），排除时间分段混淆。严格 ex-ante 时间切分两段考试：calib EV +1.44 → confirm EV +1.59（两段日聚类 CI 下界均 > 0）。落专用表 firsthit_shadow_signals（G0 纯子集，version 隔离），EV 逐事件真实触发价现算：赢 0.98/q−1 / 输 −1。预注册裁决（4 周前向）：通过 = 前向触发率 P ≥ 12% 且 EV 日聚类 CI 下界 > 0。',
  firsthit_down_chg_v1: '首触反转 G3 价格偏离：G0 基底 + chg_bps ≤ +2.82bp 门禁（触发时刻 BTC 相对窗开盘涨幅不超过 +2.82 个基点，即深折价并非由 BTC 真涨造成，属报价错杀而非信息驱动）→ 按首触时刻真实 DOWN 报价买 DOWN 的影子信号，仅记录不下单。研究口径：logit 控 t+t² 后 β=+0.07 p=0.000（chg 越低反转概率越高，单调）。严格 ex-ante 时间切分两段考试：calib EV +0.39 → confirm EV +0.97（confirm 段更强但样本少）。落专用表 firsthit_shadow_signals（G0 纯子集，version 隔离），EV 逐事件真实触发价现算：赢 0.98/q−1 / 输 −1。预注册裁决（4 周前向）：通过 = 前向触发率 P ≥ 10% 且 EV 日聚类 CI 下界 > 0。',
  firsthit_down_g4_v1: '首触反转 G4 交互门：G0 基底 + body_r≤0.35（小实体）∧ chg_bps≤+2.82bp（价格偏离有限），即 G1 ∩ G3 的交集门 → 按首触时刻真实 DOWN 报价买 DOWN。机制：长时间窄幅震荡（chg 小）且实体占路径振幅比例低（body 小）后的破位，续跌概率最高。研究口径（shape_scan_v2 冻结扫描，SPLIT=2026-08-24，npts≥8）：calib n=214 P=12.6% EV +1.03 CI[+0.08,+2.22]；confirm n=86 P=23.3%（全表最高）EV +1.85 CI[+0.35,+3.44]；binom p=0.025 → BH-FDR q=0.063 ✅ STRICT_PASS。落专用表 firsthit_shadow_signals，EV 逐事件真实触发价现算：赢 0.98/q−1 / 输 −1。实盘护栏 0.12（按 calib P=12.6% → q*=0.1235 保守取值，边际 2.8%）。⚠️非独立暴露：G4 ⊂ G1 且 G4 ⊂ G3，同窗三门全中即对同一 DOWN 事件重复下注（含 G0 共 4 倍），非 4 个独立信号。⚠️前向功效偏紧：G4 calib EV(+1.03) 与前向 4 周 CI 半宽(≈1.08) 几乎相等，即使真实效应完全等于 calib 估计，通过「EV 日聚类 CI 下界>0」的概率也仅约 50%；FAIL 时应延长观察期而非直接否决。',
  firsthit_down_g7_v1: '首触反转 G7 基底：G0 基底 + body_r≤0.35（小实体）∧ wick01=1.0（上影拒绝），捕捉冲高受阻回落反转。40d 全量回测 Calib 胜率 18.8% (EV +2.34)，Confirm 盲测 20.0% (EV +1.98，CI[+0.45,+3.46])，FDR q=0.0002。日均 ~6.5 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
  g7_streak_v1: '首触反转 G7 非强连阳：G7 组合 + 排除大级别连续多头单边（前驱 5m 连阳数 streak_up≤1 根）。回测 Calib 胜率 20.0% (EV +2.86)，Confirm 胜率 18.2% (EV +1.92，CI[+0.45,+3.30])，FDR q=0.0004，日均 5.1 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
  g7_wick20_v1: '首触反转 G7 长上影：G7 组合 + 显著上影线拒绝 upper_wick_bps≥2.0bp。多头更深力竭，回测 Confirm 胜率 24.1% (EV +3.06，CI[+0.94,+5.09])，FDR q=0.0304，日均 ~3.2 单。落专用表 firsthit_shadow_signals。实盘护栏 0.12。',
  g7_strict_v1: '首触反转 G7 严格版：G7 组合 + streak_up≤1 ∧ upper_wick_bps≥1.5bp。双重质量门，回测 Calib 胜率 23.0% (EV +3.27)，Confirm 胜率 20.8% (EV +2.68，CI[+0.68,+4.51])，FDR q=0.0030，日均 ~2.9 单。落专用表 firsthit_shadow_signals。实盘护栏 0.12。',
  g7_q05_v1: '首触反转 G7 深折价：G7 组合 + 进场报价 q≤0.05（极端凸性赔率档，单注赢付 >18.6x）。回测单注 EV +4.10~+5.77，日均 ~1.3 单。落专用表 firsthit_shadow_signals。实盘护栏 0.05。',
  g7_t270_v1: '首触反转 G7 非极晚：G7 组合 + 触发时刻 t≤270s（排除最后 30s 缺乏均值回归扩散时间的毒瘤窗）。回测 Confirm 胜率 30.4% (EV +3.08)，FDR q=0.0002，日均 ~2.1 单。落专用表 firsthit_shadow_signals。实盘护栏 0.10。',
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
  /* 2026-09-07 退役 / 在线分区（纯前端，不碰后端）：口径源是后端已下发的
     summary.retired（= shadow_gate.is_retired，RETIRED_VERSIONS 硬闸）。
     退役版本代码级停发、不再产生新样本，与在线版本混在同一张表/同一张图里会让
     「曲线末端」语义混淆（在线=前向进展，退役=最后采集点），故拆成两张卡片：
     在线区看前向验证，退役区只作历史审计。颜色仍按全量序号分配（跨区不重排），
     faded 淡化按各区自身序号，保证每区前十条都是实色。 */
  const isRetiredVer = (k: string) => analytics?.shadow[k]?.summary.retired === true
  const mergeSub = (keys: Set<string>) =>
    analytics
      ? mergeCurves(
          Object.fromEntries(Object.entries(analytics.shadow).filter(([k]) => keys.has(k))),
          SHADOW_META,
        )
      : []
  const activeShadowEntries = shadowEntries.filter(([k]) => !isRetiredVer(k))
  const retiredShadowEntries = shadowEntries.filter(([k]) => isRetiredVer(k))
  const activeShadowRows = mergeSub(new Set(activeShadowEntries.map(([k]) => k)))
  const retiredShadowRows = mergeSub(new Set(retiredShadowEntries.map(([k]) => k)))

  /* 影子分区卡片渲染（在线区 / 退役区共用）：表格 + 累计胜率曲线 + 脚注。
     与改造前单一「影子信号」卡片的 DOM、交互逐位一致（状态开关、实盘通道按钮、
     HelpHint、bench 虚线、盈亏平衡点线语义全部保留），仅两处变化：
       1) 数据源换成传入的 entries/rows 分区子集；
       2) faded 淡化的序号基准从全量改为本区序号（每区前十条实色）。
     退役区的「状态」列天然渲染成不可点的「已退役」徽章、「实盘」列渲染 —，
     因为 live 在 retired 时短路为 null——分区后语义不变，只是不再与在线行混排。 */
  const renderShadowCard = ({
    title, entries, rows, footnote, emptyHint,
  }: {
    title: string
    entries: [string, { label: string; color: string }][]
    rows: Record<string, number>[]
    footnote: React.ReactNode
    emptyHint?: string
  }) => {
    if (!analytics) return null
    if (entries.length === 0) {
      return (
        <Card title={title}>
          <div className="text-center text-ink-55 py-6 text-xs">{emptyHint ?? '无数据'}</div>
        </Card>
      )
    }
    return (
      <Card title={title}>
        <div className="overflow-x-auto mb-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line text-ink-55">
                <th className="py-1 px-2 text-left">版本</th>
                <th className="py-1 px-2 text-center">状态</th>
                <th className="py-1 px-2 text-center">实盘</th>
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
              {entries.map(([k, m], i) => {
                const s = analytics.shadow[k].summary
                // 与图表同序：本区第 11 条起降透明度，保证表格色块与曲线一一对应
                const faded = i >= 10
                // bench 可空（后端动态发现的新版本无冻结基准），双非空才计算偏离
                const dev = s.win_rate != null && s.bench_winrate != null ? s.win_rate - s.bench_winrate : null
                // enabled 缺失（旧后端）视为在线，与 gate 默认语义一致
                const online = s.enabled !== false
                // retired 缺失（旧后端）视为未退役；退役是不可逆硬闸，优先级高于 enabled
                const retired = s.retired === true
                // 实盘通道（2026-09-06 promote）：version==通道名时后端下发护栏三件套；
                // 已退役行不渲染按钮（通道即便历史上注册过也已随版本硬闸失效）
                const live = retired ? null : (s.live_channel ?? null)
                return (
                  <tr key={k} className={`border-b border-line-soft hover:bg-sunken ${online ? '' : 'opacity-45'}`}>
                    <td className="py-1 px-2 font-medium" style={{ color: m.color, opacity: faded ? 0.6 : 1 }} title={s.desc}>
                      <span className="inline-flex items-center gap-1">
                        {m.label}
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
                    <td className="py-1 px-2 text-right font-mono">{s.n}</td>
                    <td className="py-1 px-2 text-right font-mono font-bold">{pct1(s.win_rate)}</td>
                    <td className="py-1 px-2 text-right font-mono text-ink-55">{pct1(s.avg_breakeven)}</td>
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
          <LineChart data={rows} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
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
            {entries.map(([k, m], i) => {
              const s = analytics.shadow[k].summary
              // 本区第 11 条起与前十条同色，用「淡 + 细」区分，dash 留给基准/盈亏平衡语义
              const faded = i >= 10
              return (
                <Fragment key={k}>
                  <Line dataKey={m.label} stroke={m.color} strokeWidth={faded ? 1.5 : 2}
                    strokeOpacity={faded ? 0.55 : 1} dot={false} connectNulls isAnimationActive={false} />
                  {s.bench_winrate != null && (
                    <ReferenceLine y={s.bench_winrate * 100} stroke={m.color} strokeDasharray="5 4" strokeOpacity={faded ? 0.3 : 0.5} />
                  )}
                  {s.avg_breakeven != null && (
                    <ReferenceLine y={s.avg_breakeven! * 100} stroke={m.color} strokeDasharray="1 3" strokeOpacity={faded ? 0.35 : 0.6} />
                  )}
                </Fragment>
              )
            })}
          </LineChart>
        </ResponsiveContainer>
        <div className="text-[10px] text-ink-55 mt-1">{footnote}</div>
      </Card>
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

      {/* BTC K线背景 */}
      <Card title={`BTC K线背景（${kinterval === '1d' ? '日线 × 30' : '1小时 × 168'}，UTC 已收盘）`}>
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
      <Card title="场景信号（FakeBreakout 正式信号）：累计胜率 vs 回测冻结基准">
        {analytics && (
          <>
            <div className="overflow-x-auto mb-3">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-line text-ink-55">
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
                      <tr key={k} className="border-b border-line-soft hover:bg-sunken">
                        <td className="py-1 px-2 font-medium" style={{ color: m.color }}>
                          <span className="inline-flex items-center gap-1">
                            {m.label}
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
            <div className="text-[10px] text-ink-55 mt-1">实线=线上累计胜率，同色虚线=回测冻结基准；x 轴为信号时间（UTC）。</div>
          </>
        )}
      </Card>

      {/* 影子在线区（2026-09-07 与退役区分开）：只看仍在采集的版本前向进展 */}
      {renderShadowCard({
        title: `影子信号·在线采集（${activeShadowEntries.length} 版）：累计胜率 vs 回测基准 vs 盈亏平衡`,
        entries: activeShadowEntries,
        rows: activeShadowRows,
        footnote: (
          <>
            实线=线上累计胜率；同色虚线=回测冻结基准；同色点线=逐版本平均盈亏平衡（x4 含溢价 0.01 口径，其余无溢价）。
            第 11 条起复用前十色相并降为淡色细线（克制色板做不出 22 个可区分色相），表格中的版本名同步淡化以保持对应。
            本区版本仍在采集，曲线末端=最新前向进展；「回测」列为冻结基准（无冻结基准的新版本显示 —，由前向数据自行裁决）。
          </>
        ),
      })}

      {/* 影子退役区：momentum 族 / contrarian v1系 / x4_v1 / HM 族 已于 2026-09-04 永久下线，
          只作历史审计（代码级硬闸停发，toggle API 拒绝上线），不与在线版本混排 */}
      {renderShadowCard({
        title: `影子信号·已退役历史（${retiredShadowEntries.length} 版）：冻结基准 vs 历史累计胜率`,
        entries: retiredShadowEntries,
        rows: retiredShadowRows,
        emptyHint: '当前无已退役影子版本',
        footnote: (
          <>
            本区版本均为 2026-09-04 永久下线（代码级硬闸停发，不可重新上线），不再产生新样本，
            曲线末端即最后采集点；冻结 bench 基准与历史曲线保留供审计，退役理由见版本名旁「?」悬浮说明。
          </>
        ),
      })}

      {/* 周期归因 */}
      <Card title={`周期归因：大涨前 vs 大涨期（${pumpTs != null ? `${utcMD(pumpTs)} 00:00` : '—'} UTC 分界，场景+影子全部信号）`}>
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
              <div className="overflow-x-auto mb-4">
                <div className="text-xs text-ink-55 mb-1">逐影子版本拆分</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-line text-ink-55">
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
                              <tr key={ph} className="border-b border-line-soft hover:bg-sunken">
                                <td className="py-1 px-2" style={{ color: m.color }}>
                                  {m.label}
                                  <span className="text-ink-55 ml-1.5">{ph === 'pre' ? '大涨前' : '大涨期'}</span>
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
    </div>
  )
}
