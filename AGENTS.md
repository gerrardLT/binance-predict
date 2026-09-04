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

- `src/binance_predict/main.py` — FastAPI 路由与启动装配（超大单文件 ~4150 行；新增端点按既有分块注释风格就近插入）
- `src/binance_predict/services/` — 业务核心：
  - `prediction_trading.py` — 下单链路：占位→报价→护栏→币安下单→落库
  - `multi_live_trader.py` — 多通道实盘调度；**通道注册表的口径源是 `live_channels.py`**
  - `quote_edge_detector.py` — 5m 窗口报价边缘检测
- `frontend/src/App.tsx` — 前端唯一源文件（~4700 行，React 19 + Tailwind + recharts）；新组件进此文件，沿用「分块注释 + 就近放置」格局
- `tests/` — pytest + pytest-asyncio；实盘链路测试模式参照 `tests/test_multi_live_trader.py` 的 `_make_real_trader(monkeypatch)` 替身
- `alembic/versions/` — DB 迁移；`scripts/` — 一次性/评估脚本（非运行时依赖）
- `output/`、`logs/`、`.pytest_tmp/` — 运行产物与工作草稿区，不清理、不提交

## 交易语义（改下单/结算代码前必读）

- `amount_in` 单位是 **wei**：展示层一律 `/1e18`
- 执行价护栏判定**含贴线**（`avg_price >= max_exec_price` 即弃单）：贴线时滑点空间为 0，币安拒收 `slippageBps=0`（错误码 -1102）
- 失败单 `amount_in="0"`、`token_id=""`（未提交到币安）；`error_message` 可能是泛化文案，真实原因要看 `place_order` 层日志
- 划转（transfer-in/out）**不落库**；资金类统计口径从订单派生
- API 硬上限：`/api/trades/recent` limit≤100；影子信号接口 ≤200 条
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

## Git 与交付

- Conventional Commits + 中文描述：`fix(trade): …` / `feat(ui): …` / `chore(ops): …`；scope 常用：trade / ui / ops / signal / db
- **不主动 commit/push**：完成改动、本地验证、汇报结果，等用户指令
- 用户说「推送部署」= commit + push main + `gh run watch <run-id>` 盯流水线到绿 + 用 HTTP API 验证线上生效
