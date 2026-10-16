# AGENTS.md

BTC 预测市场（Binance Prediction）自动化交易系统。**真金实盘在跑**：交易链路改动必须配测试；部署只走 CI；未获用户明确指令不提交、不推送。

## 命令

环境：Windows + PowerShell（**无 `&&`，用 `;` 分隔**）；Python 包管理 uv（3.11）。

- 装依赖：`uv sync --all-extras`
- 全部测试：`.venv\Scripts\python.exe -m pytest tests/ -q`
- 单文件测试：`.venv\Scripts\python.exe -m pytest tests/test_multi_live_trader.py -q`
- **后端测试必须用 `.venv\Scripts\python.exe`**——系统 Python 没装本包，直接 `python -m pytest` 会 `ModuleNotFoundError: binance_predict`
- 前端构建（含 tsc 类型检查）：`cd frontend; npm run build`
- 前端 lint / 开发：`cd frontend; npm run lint`（oxlint）/ `npm run dev`
- DB 迁移：`alembic upgrade head`（`alembic.ini` 在根目录）

## 架构

- `src/binance_predict/main.py` — FastAPI 路由与启动装配（超大单文件 ~6500 行；新增端点按既有分块注释风格就近插入）
- `src/binance_predict/services/` — 业务核心：
  - `prediction_trading.py` — 下单链路：占位→报价→护栏→币安下单→落库
  - `multi_live_trader.py` — 多通道实盘调度；**通道注册表的口径源是 `live_channels.py`**
  - `quote_edge_detector.py` — 5m 窗口报价边缘检测
- `frontend/src/App.tsx` — 前端唯一源文件（~9600 行，React 19 + Tailwind + recharts）；新组件进此文件，沿用「分块注释 + 就近放置」格局
- `tests/` — pytest + pytest-asyncio；实盘链路测试模式参照 `tests/test_multi_live_trader.py` 的 `_make_real_trader(monkeypatch)` 替身
- `alembic/versions/` — DB 迁移；`scripts/` — 一次性/评估脚本（非运行时依赖）
- `output/`、`logs/`、`.pytest_tmp/` — 运行产物与工作草稿区，不清理、不提交

## 新研究隔离（强制）

- 新研究开始前必须完整读取 `docs/RESEARCH_ISOLATION.md`，默认从 `PREREGISTRATION` 阶段开始
- 完成预注册、候选冻结和最终盲测前，禁止读取/搜索/索引旧研究报告、旧参数、旧研究脚本、旧信号表、生产策略实现、Git 历史和外部冷归档
- 发现阶段禁止全仓库搜索；只能访问隔离规范规定的白名单原始数据、字段定义、本轮目录与中性统计工具
- 每轮研究写入独立的 `output/research_runs/<run_id>/`；成功与失败试验都必须登记，不得只保留赢家
- 最终盲测报告和哈希冻结后，只有用户明确同意进入 `COMPARISON`，才可解封旧研究做对照
- 旧研究冷归档位于项目外 `D:/project-archives/binance-predict/legacy-research-2026-09/`；生产代码、数据库历史与交易安全规则不因研究隔离而删除

## 回测数据源（验证信号/门禁前必读）

不要只拿实盘订单或生产采样表下结论——它们只有几十单 / 几周，样本内挑出的阈值几乎必然虚高。按问题选数据源：

- **只问胜率/方向**（如「某条件下次根收阳率是否更低」）：直接用币安公开 K 线回测多年历史，不需要报价。`https://data-api.binance.vision/api/v3/klines`（无需 key，可拉 2021 年至今）。先用 K 线复现信号触发，与线上信号表对账吻合后再用
- **问入场价/EV/价格门**（如护栏、入场价白名单）：用 PREDEXON 历史订单簿，**不要只依赖** `prediction_market_samples`（生产只回溯到 2026-07-26/08-13）。脚本 `scripts/predexon_backfill.py`，端点 `/v2/predictfun/orderbooks`＝币安预测市场（订单 `vendor=PREDICT_FUN`）的历史盘口；`.env` 的 `PREDEXON_API_KEY` 多 key 逗号分隔、每 key 1 rps、须 `curl.exe`+浏览器 UA（urllib 被 Cloudflare 拦）
  - 找市场：`/v2/predictfun/markets?search=<标题>`，BTC 15m 标题形如 `Bitcoin Up or Down - October 12, 8:45AM-9AM ET`（美东时间、整点省略 `:00`）。**标题不含年份**，须用 `created_time` 在开盘前 3 天内去重，否则会命中别年同日期的市场
  - 取盘口：`/v2/predictfun/orderbooks?market_id=&start_time=&end_time=`（毫秒）；Yes=UP token，UP 入场价取 `best_ask`，DOWN 取 `1 − best_bid`；结算看 `winning_outcome`（0=UP，1=DOWN）。BTC 15m 虽从 2026-04-29 起能查到，但 **2026-04-29→06-30 是不完整段，禁止用于策略裁决、护栏优化、累计 EV 或跨段合并统计**：5 月缺失 31%~39%，6 月 S2/S4 仍缺 17%~27%，且缺失与胜负相关。PREDEXON 主分析段固定从 **2026-07-01** 起；每次必须先按通道×月份报告覆盖率，任一桶缺失 >10% 则该桶只作数据质量备注，不参与结论
  - **快照时点必须对齐实盘报价时刻**（场景开盘单约开盘 +3.5s，S5 约 +310s；以订单 `quote_json.timestamp − window_start` 为准）。开盘 +1s 的盘口很薄、常见 0.97 残单，用它模拟会把大量实盘成交误判为弃单（2026-10-13 踩过：S1 74 笔实盘成交里 46 笔被误判）。对齐后实盘成交价 − 一档价中位 0.000
  - 模拟线上护栏要用「按实盘金额吃 5 档的 VWAP」判，不能用一档价判护栏再按 VWAP 成交
- ⚠️ `.pytest_tmp/predexon_curve_research/` 是 **Polymarket** 数据（manifest `source=PREDEXON Polymarket market-price`），不是币安平台，不能当币安报价用
- 实盘订单的正确用途：样本外检验通过之后，核对真实成交价/滑点是否吃掉边

## 交易语义（改下单/结算代码前必读）

- `amount_in` 单位是 **wei**：展示层一律 `/1e18`
- 执行价护栏判定**含贴线**（`avg_price >= max_exec_price` 即弃单）：贴线时滑点空间为 0，币安拒收 `slippageBps=0`（错误码 -1102）
- 失败单 `amount_in="0"`、`token_id=""`（未提交到币安）；`error_message` 可能是泛化文案，真实原因要看 `place_order` 层日志
- 划转（transfer-in/out）**不落库**；资金类统计口径从订单派生
- API 上限：`/api/trades/recent` limit≤5000（支持全量订单历史）；影子信号接口 ≤200 条
- 每通道每窗口至多一单；15m 市场必须先登记进 `trader._15m_markets` 否则被锚定守卫拒单（**测试构造第二窗口时同理**）

## 前端约定

- UI 全中文；通道名/指标配 `HelpHint`（`?` 悬浮解释）。`SIGNAL_INFO` 常量是通道中文名与说明的唯一事实源，与后端 `live_channels.py` 对齐——新通道两边同步加
- recharts 曲线 dataKey 用 label 本身：改 `SHADOW_META`/`SCENE_META` 的 label 会同时改曲线与图例
- 后端新字段：先同步进 interface（过渡可用 `Record<string, unknown>`，热路径要收敛成类型）

## 测试要求

- 交易链路（`prediction_trading` / `multi_live_trader`）行为改动：必须在 `tests/test_multi_live_trader.py` 加或改用例；护栏/边界类改动加贴线、越界正反例
- 提交前本地跑过相关测试；CI Test Gate 会跑全量 `tests/`

## 边界（禁止）

- **绝不提交 `.env` 或任何密钥/token**。秘密只存于 `.env`（gitignored，模板 `.env.example`）与 GitHub Secrets
- **绝不直接操作生产**（本地 SSH 到 VPS 不通）。生产运维只走两条通道：HTTP API（地址与 Bearer token 在本地 `.env`）或 GitHub Actions
- `push main` = 自动部署（CI：test → build 镜像 → 部署 VPS）。提交前确认改动可上线
- 不改 `.github/workflows/*`、`docker/*`、`nginx.conf`，除非任务明确要求
- 临时脚本放 `.pytest_tmp/`（已在 .gitignore），不要散落根目录
- 研究产物（图表/截图/HTML/导出 JSON）同样放 `.pytest_tmp/`，不要新建根目录 `figures/` 之类
- 编辑源码的备份副本（`temp_edit.txt`、`*.bak`）不落仓库根目录：源码在 git 里，用完即删

## Git 与交付

- Conventional Commits + 中文描述：`fix(trade): …` / `feat(ui): …` / `chore(ops): …`；scope 常用：trade / ui / ops / signal / db
- **不主动 commit/push**：完成改动、本地验证、汇报结果，等用户指令
- 用户说「推送部署」= commit + push main + `gh run watch <run-id>` 盯流水线到绿 + 用 HTTP API 验证线上生效
