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
    # DeepSeek 原生 API Key（用于决策模型 deepseek-v4-pro）
    deepseek_api_key: str = ""
    # DeepSeek 原生 API base_url
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    # 百炼 DashScope API Key（用于复盘模型 qwen3.7-max）
    dashscope_api_key: str = ""
    # 百炼 OpenAI 兼容接口 base_url
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 决策 LLM 模型名（每 60s 调用一次，走 DeepSeek 原生 API）
    decision_model: str = "deepseek-v4-pro"
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
    # 现货 WebSocket 地址（公开行情，无需 API Key）
    binance_spot_ws_url: str = "wss://stream.binance.com:9443/ws"
    # --- 预测参数 ---
    # 交易品种
    symbol: str = "BTCUSDT"
    # 预测周期
    horizon: str = "5m"
    # 噪声阈值：future_return 绝对值小于此值视为 NOISE
    noise_threshold: float = 0.0005

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

    # --- LLM 成本配置（元/百万 Token）---
    llm_input_price_per_1m: float = 12.0   # DeepSeek 输入价格
    llm_output_price_per_1m: float = 24.0  # DeepSeek 输出价格

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
    agent_health_window_stale_seconds: float = 600.0
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

    # --- 告警推送去重抑制 ---
    # 同一告警 code 在此窗口（秒）内只推送一次，避免 60s 轮询反复轰炸。
    # 仅作用于主动推送渠道（邮件/webhook），不影响日志与落库。
    agent_alert_suppress_seconds: float = 900.0

    # --- 告警邮件推送（SMTP，主渠道；非 OK 状态且有新告警时触发）---
    # 总开关；为 False 时不发邮件（即便配置了 SMTP）
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
    # 手动模式：聚类压缩后目标窗口数
    agent_deep_learn_target_clusters: int = 25
    # 手动模式：LLM max_tokens（深度分析输出上限，基于实测：全量窗口输入~30k tokens，reasoning+discoveries 输出~10k tokens）
    agent_deep_learn_max_tokens: int = 16384
    # 手动模式：流式深度分析「空闲超时」（秒）。仅当两次 token 之间的间隔超过该值才判定超时，
    # 不再对整体调用施加硬性总超时——只要模型在持续吐字就允许长时间运行（替代旧的 100s 一次性超时）。
    agent_deep_learn_idle_timeout: float = 60.0

    # --- Sentiment Agent 行为参数 ---
    # 自动交易总开关：默认关闭，必须显式设置 AGENT_AUTO_TRADE=true 才允许自动下单。
    agent_auto_trade: bool = False
    # 交易置信度阈值：仅当总开关开启、direction∈{UP,DOWN} 且 confidence > 此值才执行交易。
    agent_trade_confidence_threshold: float = 0.6
    # Evolve 触发间隔：每累计完成 N 次 Validate 触发一次 Evolve（Req 5.1 / 6.5）
    agent_evolve_interval: int = 12
    # Learn 窗口数：Learn 阶段选取最近 N 个 outcome 非空的情绪窗口（Req 2.2）
    agent_learn_window_count: int = 50
    # Predict 触发采样点：当前窗口累计有效采样点达到 N 个时触发 Predict（Req 3.1）
    agent_predict_trigger_samples: int = 10
    # ACTIVE 模式数上限：超过则 Evolve 强制淘汰超额部分（Req 5.8）
    agent_active_pattern_cap: int = 30
    # 淘汰保护最小样本数：sample_count <= 此值的模式不因上限被淘汰（Req 5.8）
    agent_min_sample: int = 5

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
