"""
BTC 5min LLM 预测系统 V2 - 全局配置模块

从 .env 文件和环境变量加载配置，使用 pydantic-settings 进行类型安全校验。
所有配置项集中管理，确保前后端数据一致性（对应用户规则 8）。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，从 .env 文件加载"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM API 配置 ---
    # DeepSeek 原生 API Key（用于决策模型 deepseek-v4-flash）
    deepseek_api_key: str = ""
    # DeepSeek 原生 API base_url
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # 百炼 DashScope API Key（用于复盘模型 qwen3.7-max）
    dashscope_api_key: str = ""
    # 百炼 OpenAI 兼容接口 base_url
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 决策 LLM 模型名（每 60s 调用一次，走 DeepSeek 原生 API）。
    # 别名自动指向最新版 DeepSeek-V4-Flash-0731；官方 API 不暴露带版本后缀的固定名，
    # 请勿写成 deepseek-v4-flash-0731（会因模型名不存在而调用失败）。
    decision_model: str = "deepseek-v4-flash"
    # 复盘 LLM 模型名（T+5min 到期后调用，走百炼 DashScope）
    review_model: str = "qwen3.7-max"

    # --- 数据库配置 ---
    # Docker 部署时 compose 读取这三个变量初始化 DB + 构建 DATABASE_URL
    db_user: str = "postgres"
    db_password: str = "changeme"
    db_name: str = "binance_predict"
    # PostgreSQL + TimescaleDB 异步连接字符串
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/binance_predict"
    # --- Binance API 配置 ---
    # 现货 REST 公共行情 base URL（api.binance.com 被网络封锁的环境可指向
    # 官方公共行情镜像 data-api.binance.vision，API 路径完全一致；仅公开行情端点）
    binance_api_base: str = "https://api.binance.com"
    # 现货 WebSocket 地址（公开行情，无需 API Key）
    binance_spot_ws_url: str = "wss://stream.binance.com:9443/ws"
    # --- 预测参数 ---
    # 交易品种
    symbol: str = "BTCUSDT"
    # 预测周期
    horizon: str = "5m"
    # 噪声阈值（已从结果标注中移除）：预测市场按方向结算，outcome 改按
    # actual_return 正负号标注（见 main.py 归档器）。此值保留作策略层
    # “预计横盘不下注”过滤器备用（后续计划改为波动率自适应）。
    noise_threshold: float = 0.0005

    # --- 原始采样保留策略 ---
    # prediction_market_samples（15s 级原始数据）的保留时长（小时）。
    # <=0 表示永不删除（默认）：TimescaleDB 下约 5760 行/天、~300MB/年，成本可忽略；
    # 而归档表只保存"当时想到要存的"字段——历史价格曲线因旧的 1 小时清理策略
    # 永久丢失（3510/3522 窗口无价格数据）即为教训。需要旧行为可设为 1。
    sample_retention_hours: float = 0.0

    # --- 置信度运营门槛 ---
    confidence_strong: float = 0.75
    confidence_normal: float = 0.60
    confidence_weak: float = 0.50

    # --- 服务配置 ---
    api_port: int = 8000
    log_level: str = "INFO"
    # 日志文件目录（容器内路径，通过挂载持久化到宿主机）。空字符串禁用文件日志。
    log_dir: str = "logs"
    log_rotation: str = "00:00"  # 每天零点切割
    log_retention: str = "14 days"  # 保留 14 天

    # --- 安全配置 ---
    # CORS 允许的前端源（逗号分隔，如 "http://localhost:5173,https://example.com"）。
    # 空字符串默认仅允许 localhost 开发源。生产环境必须显式指定。
    cors_allowed_origins: str = ""
    # API Bearer Token 认证密钥。空字符串时禁用认证（仅开发环境）。
    # 生产环境必须设置，否则所有 API 端点对外开放。
    api_auth_token: str = ""
    # Web 登录密码（单一访问密码）。未配置时所有 /api 请求返回 401，登录页提示未配置。
    login_password: str = ""

    # --- Binance Prediction Trading 配置 ---
    # Binance API Key（用于预测市场交易，需在币安后台开启 Prediction Trading 权限）
    binance_api_key: str = ""
    # Binance API Secret
    binance_api_secret: str = ""
    # 预测市场单笔交易金额（USDT）
    prediction_trade_amount_usdt: float = 2.0
    # 预测市场钱包地址
    prediction_wallet_address: str = ""
    # 预测市场钱包 ID
    prediction_wallet_id: str = ""

    # --- Sentiment Agent 分阶段外层超时（秒，AgentScheduler 层）---
    # 由 AgentScheduler 用 asyncio.wait_for 施加的硬兜底超时（design.md 决策 4）。
    # 外层超时 > 对应 LLM 内层超时，使 LLM 先以干净异常返回，外层仅作最终保护。
    # 每个字段均可经 .env 独立覆盖（如 AGENT_TIMEOUT_LEARN=120）。
    # Validate 阶段：纯对比无 LLM 调用，10s 足够
    agent_timeout_validate: float = 10.0
    # Predict 阶段：时间敏感、轻量单次匹配
    agent_timeout_predict: float = 30.0
    # Learn 阶段：重载（分析最近 50 窗口 + 最多 2 次重试）
    agent_timeout_learn: float = 110.0
    # Evolve 阶段：重载（全模式 + 最近 12 次预测 + 最多 2 次重试）
    agent_timeout_evolve: float = 110.0

    # --- Sentiment Agent LLM 内层超时（秒，LLMService 层）---
    # 由各阶段 LLM 方法用 asyncio.wait_for 施加（design.md 决策 4）。
    # Validate 无 LLM 调用，故不设内层超时。每个字段均可经 .env 独立覆盖。
    # Predict 阶段 LLM 调用超时（< agent_timeout_predict）
    agent_llm_timeout_predict: float = 25.0
    # Learn 阶段 LLM 调用超时（< agent_timeout_learn）
    agent_llm_timeout_learn: float = 100.0
    # Evolve 阶段 LLM 调用超时（< agent_timeout_evolve）
    agent_llm_timeout_evolve: float = 100.0

    # --- LLM 成本配置（元/百万 Token，按 deepseek-v4-flash 官方价：输入 1 / 输出 2）---
    llm_input_price_per_1m: float = 1.0    # DeepSeek-V4-Flash 输入价格（缓存未命中）
    llm_output_price_per_1m: float = 2.0   # DeepSeek-V4-Flash 输出价格

    # --- 告警配置 ---
    agent_alert_enabled: bool = True
    agent_alert_consecutive_failures: int = 3
    agent_alert_daily_cost_limit_usd: float = 10.0
    agent_alert_queue_depth_threshold: int = 50
    # Fix #19: 告警达阈时是否自动阻断交易（熔断器）。
    # 为 True 时，LLM 成本超限或阶段连续失败超阈会置位阻断标志，
    # 由 evaluate_trade_gate 拒绝新交易，避免异常状态下持续下单。
    agent_alert_block_trades: bool = True

    # --- Agent 运行健康监控（services/health.py + GET /api/agent/health）---
    # 后台监控 loop 总开关：为 True 时 lifespan 启动 _health_monitor_loop
    agent_health_monitor_enabled: bool = True
    # 后台轮询/告警检查间隔（秒）：每次 build_report 并检查 CRITICAL 告警
    agent_health_monitor_interval: float = 60.0
    # 健康快照落库间隔（秒）：>= monitor_interval，控制 health_snapshots 表增长
    agent_health_snapshot_interval: float = 300.0
    # 窗口停摆告警阈值（秒）：最近窗口距今超过此值判 CRITICAL WINDOW_STALE
    # 2026-08-15 600→900：600（仅 2 个归档周期）下归档慢一轮就边缘抖动误报
    # （实测 613s 触发后下一轮即恢复）；900 = 3 个周期，真停摆仍及时报
    agent_health_window_stale_seconds: float = 900.0
    # 匹配率/方向分布统计取最近 N 条 AgentPrediction
    agent_health_recent_predictions: int = 20
    # 窗口连续性 gap 检测取最近 N 条 SentimentWindow
    agent_health_recent_windows: int = 60
    # 置信度校准最小样本数：低于此值 summary 标注样本不足、不做校准判断
    agent_health_min_calibration_samples: int = 30
    # 置信度校准取最近 N 条已验证预测（限制全表扫描，避免随数据量增长逐渐变慢）
    agent_health_calibration_sample_limit: int = 500
    # LLM 阶段成功率告警下限：低于此值判 WARN LLM_ERROR_RATE
    agent_health_llm_success_rate_floor: float = 0.8
    # PREDICT 心跳停摆倍数：最近成功距今 > 倍数×300s 判 CRITICAL PREDICT_STALE
    agent_health_predict_stale_multiplier: float = 2.0
    # health_snapshots 保留天数：落库后清理早于此天数的旧快照，防止无限增长
    agent_health_snapshot_retention_days: int = 7

    # --- 告警推送去重抑制（分级）---
    # 同一告警 code 在抑制窗口内只推送一次，避免 60s 轮询反复轰炸。
    # 仅作用于主动推送渠道（邮件/webhook），不影响日志与落库。
    # CRITICAL 级：真停摆/真故障要尽快知道，保持 15 分钟重推节奏
    agent_alert_suppress_seconds: float = 900.0
    # WARN 级：多为慢性问题（如 NO_MATCH 匹配率低），高频重推无信息量，
    # 拉长到 4 小时（2026-08-15：修复邮件配置后慢性 WARN 曾一天轰炸 96 封）
    agent_alert_suppress_warn_seconds: float = 14400.0

    # --- 告警邮件推送（SMTP，主渠道；非 OK 状态且有新告警时触发）---
    # 告警推送总闸：False 时邮件+webhook 全部暂停（健康监控仍跑，仅不主动推送）。
    # 与信号推送（signal_push_email_enabled）相互独立——暂停告警不影响信号邮件。
    agent_alert_notify_enabled: bool = True
    # 邮件渠道开关；为 False 时不发告警邮件（即便配置了 SMTP）
    agent_alert_email_enabled: bool = False
    # SMTP 服务器地址与端口（587=STARTTLS，465=SSL 需另配；默认走 STARTTLS）
    agent_alert_smtp_host: str = ""
    agent_alert_smtp_port: int = 587
    # SMTP 登录凭据（多数邮箱用「授权码」而非登录密码）
    agent_alert_smtp_user: str = ""
    agent_alert_smtp_password: str = ""
    # 是否使用 STARTTLS（587 端口置 True；若服务器为 465 SSL 端口请置 False 并自行适配）
    agent_alert_smtp_use_tls: bool = True
    # 发件人地址；留空则回退到 smtp_user
    agent_alert_email_from: str = ""
    # 收件人（逗号分隔，可多个）；空则不发
    agent_alert_email_to: str = ""
    # SMTP 连接/发送超时（秒）
    agent_alert_email_timeout: float = 10.0

    # --- 告警 Webhook 推送（通用 JSON POST，可选备用渠道）---
    # 空字符串禁用 webhook。目标为通用 JSON 接收端；接入钉钉/飞书/Telegram
    # 自定义机器人时 payload 格式各异，如需适配请告知具体平台。
    agent_alert_webhook_url: str = ""
    # webhook POST 超时（秒）
    agent_alert_webhook_timeout: float = 5.0

    # --- 信号邮件推送（所有信号族共用；SMTP 物理通道复用 agent_alert_smtp_*）---
    # 总开关：与告警开关解耦——暂停告警（agent_alert_notify_enabled=False）
    # 不影响信号邮件；场景信号还需各自子开关（fake_breakout_email_enabled）。
    signal_push_email_enabled: bool = True
    # 全局日上限（所有信号族合计）：超限后仍落表但不再发邮件，防轰炸。
    # 2026-08-25 用户调至 800：实盘开火闸已收窄推送面（只推已开火通道，
    # 日常 5~10 封），800 仅作异常洪峰（多通道齐开/极端行情）的兜底闸。
    signal_push_max_daily_emails: int = 800

    # --- 风控统计缓存（Fix #20）---
    # RiskController.refresh_daily_stats 的 TTL（秒），避免短时间内重复全量查询。
    risk_stats_cache_ttl_sec: float = 30.0

    # --- LLM 输出语义验证 ---
    agent_llm_validation_enabled: bool = True
    agent_llm_validation_strict: bool = False  # False=仅记录 SOFT_WARN

    # --- 风控参数（Plan 步骤 8/9/10）---
    agent_risk_control_enabled: bool = True
    agent_min_pattern_win_rate: float = 0.4
    agent_min_pattern_samples: int = 5
    agent_max_consecutive_losses: int = 5
    agent_max_daily_trades: int = 20
    agent_max_daily_loss_usdt: float = 10.0
    agent_prediction_min_remaining_seconds: int = 30

    # --- 双 Worker 架构（Plan 步骤 11/12）---
    agent_dual_worker_enabled: bool = True
    agent_predict_max_queue_wait: float = 15.0

    # --- 模式去重（Plan 步骤 14/15）---
    agent_dedup_enabled: bool = True
    agent_dedup_auto_downgrade: bool = False  # True=自动将重复 CREATE 转为 UPDATE

    # --- 模式发现双模式配置 ---
    # manual: 手动触发深度分析（用户控制 token 消耗）
    # auto: 保留旧逻辑，窗口归档自动 Learn（token 消耗不可控）
    agent_learn_mode: str = "manual"
    # 手动模式：深度分析最大窗口数
    agent_deep_learn_max_windows: int = 100
    # 手动模式：max_windows 上限（端点入参 clamp，防止无上限透传 limit）
    agent_deep_learn_max_windows_cap: int = 500
    # 手动模式：KMeans 目标簇数（Python 聚类版；实际 n_clusters=min(该值, 窗口数)）
    agent_deep_learn_target_clusters: int = 25
    # 采样时间跨度（天）：Deep Learn 先按 start_time >= now-days_back 圈定再分层抽样
    agent_deep_learn_days_back: int = 7
    # holdout 比例：按 start_time 排序切出最新该比例作样本外校验集
    agent_deep_learn_holdout_ratio: float = 0.3
    # 准入最小 holdout 样本数：低于此值的候选模式不予写库（样本外统计不可信）。
    # 提高到 50：10 个样本要让 Wilson 下界>0.5 几乎需 9~10 全对，极易把「连续走运」
    # 误当成模式；50+ 才能把 ~55% 的真实 edge 与噪声区分开。过严导致罕见模式难准入
    # 属预期取舍，可经 .env（AGENT_DEEP_LEARN_MIN_HOLDOUT_SAMPLES）按实际数据量回调。
    agent_deep_learn_min_holdout_samples: int = 50
    # 非流式深度分析专用超时（秒）：替代旧的借用 LEARN 100s 硬超时
    agent_deep_learn_timeout: float = 300.0
    # 手动模式：LLM max_tokens（深度分析输出上限，基于实测：全量窗口输入~30k tokens，reasoning+discoveries 输出~10k tokens）
    agent_deep_learn_max_tokens: int = 16384
    # 手动模式：流式深度分析「空闲超时」（秒）。仅当两次 token 之间的间隔超过该值才判定超时，
    # 不再对整体调用施加硬性总超时——只要模型在持续吐字就允许长时间运行（替代旧的 100s 一次性超时）。
    agent_deep_learn_idle_timeout: float = 60.0

    # --- 科学发现 V1.1：决策点截断对齐 + 经济闸（宪法第八条规则 8 / Q6 第 5 步）---
    # 决策点（开窗后秒数）：predict 在第 10 采样点（~150s）触发，发现/初筛/在线
    # 统一使用该截断视图（消除全窗初筛的截断错位）
    agent_decision_point_sec: float = 150.0
    # 经济闸开关：双轨 ACTIVE 假设须按入场价口径（费 2%+溢价 0.01）EV bootstrap
    # CI 下界>0 才允许直上线，否则降级 OBSERVE（经济功效不足，非模式无效）
    agent_ev_gate_enabled: bool = True

    # --- Sentiment Agent 行为参数 ---
    # 自动交易总开关：默认关闭，必须显式设置 AGENT_AUTO_TRADE=true 才允许自动下单。
    agent_auto_trade: bool = False
    # 交易置信度阈值：仅当总开关开启、direction∈{UP,DOWN} 且 confidence > 此值才执行交易。
    agent_trade_confidence_threshold: float = 0.6
    # Evolve 触发间隔：每累计完成 N 次 Validate 触发一次 Evolve（Req 5.1 / 6.5）。
    # 注：samples 触发模式下该值仅用于 windows 回退模式，及 Evolve 读取的近期预测条数。
    agent_evolve_interval: int = 12
    # Evolve 触发模式（Item 5：进化时钟与证据量挂钩）：
    # - "samples"（默认）：累计「新验证的预测样本数」达 agent_evolve_min_new_samples
    #   才触发 Evolve，确保进化建立在足够新标注证据上，避免信号积累慢于窗口推进时空转。
    # - "windows"：回退旧行为，每 agent_evolve_interval 次窗口归档触发一次。
    agent_evolve_trigger_mode: str = "samples"
    # samples 模式下触发 Evolve 所需的「自上次 Evolve 以来新验证预测样本数」阈值。
    agent_evolve_min_new_samples: int = 24
    # Learn 窗口数：Learn 阶段选取最近 N 个 outcome 非空的情绪窗口（Req 2.2）
    agent_learn_window_count: int = 50
    # Predict 触发采样点：当前窗口累计有效采样点达到 N 个时触发 Predict（Req 3.1）
    agent_predict_trigger_samples: int = 10
    # ACTIVE 模式数上限：超过则 Evolve 强制淘汰超额部分（Req 5.8）
    agent_active_pattern_cap: int = 30
    # 淘汰保护最小样本数：sample_count <= 此值的模式不因上限被淘汰（Req 5.8）
    agent_min_sample: int = 5

    # --- 假突破信号系统（4h 阻力/支撑破位检测 + A+B 过滤，暂不下注）---
    # 总开关：为 True 时 lifespan 启动秒级检测循环
    fake_breakout_enabled: bool = True
    # 破位阈值：BTC mid > 4h 阻力 × (1 + eps) 判定冲高破位
    fake_breakout_eps: float = 0.0005
    # 检测循环间隔（秒）：读 collector.store.mid_price（bookTicker 实时价）
    fake_breakout_check_interval: float = 1.0
    # [已废弃] 级别回看窗口数已内置于 detector.LEVEL_LOOKBACKS（仅 4h=48，2026-08-15 收窄）
    fake_breakout_resistance_lookback: int = 288
    # 阻力位刷新间隔（秒）：从 sentiment_windows 重算
    fake_breakout_resistance_refresh_seconds: float = 60.0
    # 风控：同一 (方向,级别) 信号冷却（秒）——一波冲高/冲低每级别每方向只报一次
    fake_breakout_cooldown_seconds: int = 900
    # 风控：日内信号上限（防邮件轰炸；收窄为 4h+A+B 过滤后约 1 条/天，100 留足余量）
    fake_breakout_max_daily_signals: int = 100
    # 结算回读死线缓冲（秒）：signal_time + 15min + 本缓冲后回读 BTC 价回填方向
    fake_breakout_settle_buffer_seconds: int = 90
    # 信号邮件推送开关（复用 agent_alert_* SMTP 配置；2026-08-28 起在
    # _settle_15m 结算复盘推送路径生效——触发不再发预告邮件）
    fake_breakout_email_enabled: bool = True

    # --- X4 情绪错位影子信号（M4 影子并行，2026-08-19）---
    # 影子模式：只记录不下注不发邮件；次窗归档后回读真实报价与结算定案经济账
    misalignment_enabled: bool = True

    # --- 报价 edge 影子信号（A 顺势 / B 逆势，2026-08-20）---
    # 影子模式：只记录不下注；归档后处理首个命中报价，落表即结算
    # 规则冻结：A t∈[90,120)s×q∈[0.69,0.75) / B t∈[45,60)s×q∈[0.15,0.25)
    quote_edge_enabled: bool = True

    # --- K 线科学发现影子信号（KREV 族，2026-08-28）---
    # 720d 发现流水线冻结注册表条件的实时重放：15m bar 收盘后复用离线特征
    # 管道求值（口径逐位一致），次根收盘按回测口径结算。只记录不下注。
    # 默认开启：与其他影子信号（fake_breakout/misalignment/quote_edge）一致，
    # 部署即生效；口径保真测试已全绿（registry replay 381/132/137 & 378/130/134）。
    # 仅作紧急停用制动力，正常情况下无需触碰。
    kline_shadow_enabled: bool = True
    # 影子期邮件推送开关（默认静默，防轰炸；结算复盘口径预留）
    kline_shadow_email_enabled: bool = False

    # --- 多通道实盘（MultiLiveTrader，2026-08-24，取代旧单版本 quote_edge 实盘字段）---
    # 12 通道（quote_edge v1/v2/v3 × contrarian 系 + momentum × 2 + x4 × 2 + 场景 S1/S5/S2/S4）
    # 可同时开启；每通道独立金额/日限/护栏，通道静态描述见 services/live_channels.py。
    # 每通道每窗至多一单（内存 + DB 唯一约束双保险）。
    # 每通道默认单注（USDT，硬上限 50 拒启，同旧哲学：不靠自律靠拒启）
    live_default_amount_usdt: float = 2.0
    # 每通道默认日单量护栏：当日该通道 FILLED 达上限后停火（用户拍板：各自 100）
    live_default_max_daily_orders: int = 100
    # 启动时通道覆盖配置（JSON 字符串，重启保持开启集——解决旧模式每次部署重置 OFF）：
    # {"quote_contrarian_v1":{"enabled":true,"amount_usdt":2.0,"max_daily_orders":100,"max_exec_price":0.28},...}
    # 未知通道/金额超限/非法值 → MultiLiveTrader 构造抛 ValueError 拒绝装配（fail fast）。
    # 运行时 toggle/金额热调为内存态，重启回落本配置。
    live_channels_json: str = ""

    # --- 场景研究（LLM 研究员，M2 2026-08-16）---
    # 总开关：False 时不启动研究调度循环
    scene_research_enabled: bool = True
    # T1 定期触发：距上次评估天数
    scene_research_interval_days: int = 7
    # T2 累积触发：自上次评估新增已结算信号数
    scene_research_new_signals: int = 30
    # T3 异常触发判定所需最小实盘样本（某场景已结算数）
    scene_research_min_live_sample: int = 20
    # 任意触发后的冷却时长（小时，防抖动重复评估）
    scene_research_cooldown_hours: int = 24

    # --- 系统B（情绪 Agent Loop）退役开关（2026-08-16 拍板）---
    # False = lifespan 不实例化 SentimentAgent/AgentScheduler，预测循环停用；
    # 类与表保留（只读存档），可随时翻回 True 恢复。
    # 情绪窗口归档器（sw_archiver）独立于本开关继续运行——它是场景信号系统的 4h 位势数据源。
    agent_loop_enabled: bool = False

    # --- 模式池分级与定期重回测（无限进化引擎，只发现不下注）---
    # 总开关：为 True 时 lifespan 启动重回测调度循环
    # 2026-08-16 随系统B退役默认关闭（模式池为系统B组件）
    pattern_reeval_enabled: bool = False
    # 检查间隔（秒）：每轮检查自上次回测以来新归档的已标注窗口数
    pattern_reeval_check_interval: float = 1800.0
    # 新窗口触发阈值：累积 >= 此数量的新已标注窗口才触发一轮全量重回测
    # （288 ≈ 1 天数据，保证每次对比有统计意义的新增量）
    pattern_reeval_min_new_windows: int = 288
    # 单次重回测最大窗口数（防止全表随数据增长变慢）
    pattern_reeval_max_windows: int = 20000

    @property
    def allowed_origins_list(self) -> list[str]:
        """解析 CORS 允许源列表。空值回退到 localhost 开发默认值。"""
        if self.cors_allowed_origins.strip():
            return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]
        return [
            "http://localhost:5173",
            "http://localhost:8000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8000",
        ]

    @property
    def agent_phase_timeouts(self) -> dict[str, float]:
        """分阶段外层超时映射（秒），供 AgentScheduler 按阶段取超时（design.md 决策 4）。

        取值来自上方标量字段，覆盖请改对应 .env 项（如 AGENT_TIMEOUT_PREDICT）。
        """
        return {
            "VALIDATE": self.agent_timeout_validate,
            "PREDICT": self.agent_timeout_predict,
            "LEARN": self.agent_timeout_learn,
            "EVOLVE": self.agent_timeout_evolve,
        }

    @property
    def agent_llm_timeouts(self) -> dict[str, float]:
        """分阶段 LLM 内层超时映射（秒），供 LLMService 各阶段方法施加内层超时（design.md 决策 4）。

        取值来自上方标量字段，覆盖请改对应 .env 项（如 AGENT_LLM_TIMEOUT_LEARN）。
        """
        return {
            "PREDICT": self.agent_llm_timeout_predict,
            "LEARN": self.agent_llm_timeout_learn,
            "EVOLVE": self.agent_llm_timeout_evolve,
        }


# 全局单例配置
settings = Settings()
