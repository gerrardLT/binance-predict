# AGENTS.md 规范标准调研报告

> 调研日期：2026-08-29
> 目的：为 binance-predict 引入 AGENTS.md 前，先确定"高质量 AGENTS.md"的可验证标准，避免凭感觉写。
> 方法：四轮递进调研——官方规范 → 标杆项目实例 → 社区实证研究 → 安全红线与官方最佳实践。

## 一、调研来源

| 轮次 | 来源 | 性质 |
|------|------|------|
| 1 | https://agents.md/（官方站点） | 官方规范（Linux Foundation 旗下 Agentic AI Foundation 托管） |
| 2 | openai/codex 仓库 AGENTS.md（323 行，Rust） | 标杆实例 |
| 2 | apache/airflow 仓库 AGENTS.md（523 行，Python） | 标杆实例（同为 Python 后端，参考价值最高） |
| 3 | BetterClaw《AGENTS.md Best Practices: Template and Guide》（基于 2,500+ 仓库分析） | 实证研究 |
| 4 | OpenAI ChatGPT Learn 官方 Best Practices 指南 | 官方最佳实践 |
| 4 | GitHub Gist《AGENTS.md best practices》（安全红线清单） | 社区实践 |

## 二、官方规范要点（agents.md）

1. **定位**：README 面向人类，AGENTS.md 面向 coding agent——存放构建步骤、测试命令、工程约定等 agent 必需但"不该塞进 README"的上下文。
2. **格式**：就是标准 Markdown，无 schema、无必填字段；agent 直接解析全文。
3. **位置与优先级**：放仓库根目录；monorepo 可在子目录嵌套。冲突规则：**离被编辑文件最近的 AGENTS.md 胜出；用户聊天中的显式指令覆盖一切**。
4. **行为**：agent 会自动执行文件中列出的测试/检查命令，并在任务完成前尝试修复失败。
5. **生态**：OpenAI Codex、Cursor、Gemini CLI、GitHub Copilot coding agent、Aider、Zed、Warp、Jules、Devin、Junie 等 30+ 工具原生读取；GitHub 上 60,000+ 仓库采用。
6. **与 CLAUDE.md / .cursorrules 的关系**：三者约 90% 内容重合；AGENTS.md 是通用格式，工具专属文件仅在需要其独有特性（@imports、glob 规则）时补充。

## 三、标杆项目实例分析

### openai/codex（323 行）

写法特征：**全是硬规则 + 具体命令 + 正反例**，几乎没有一句空话。

- 环境/命令：`just test` / `just fix`，明确"不要直接跑 cargo test"、"不要用 PID 杀 Rust 命令"
- 工程红线：模块目标 <500 LoC、超过 ~800 LoC 必须开新模块；单个变更 ≤800 行（复杂逻辑 ≤500 行）；"resist adding code to codex-core"
- 测试指导：agent 逻辑改动必须配集成测试；禁止为静态值/已删除逻辑写测试
- 沙箱/环境事实：明确告知 agent 自己运行在什么沙箱里、哪些代码不能碰（`CODEX_SANDBOX_*`）

### apache/airflow（523 行）

写法特征：**命令速查表 + 架构边界 + 安全模型分级**。

- 命令区带 `<!-- START generated-commands -->` 注释允许自动更新
- 明确"禁止在宿主机直接跑 pytest/python/airflow——一律走 breeze"
- 架构边界用编号列表写清数据流向（谁永远不直接访问 DB）
- 安全模型教 agent **区分**：真实漏洞 / 已知限制（不要当新发现上报）/ 部署加固建议
- 编码标准具体到反模式："生产代码禁止 assert"、"注释克制：代码说 what、注释说 why"、"函数名用动词开头"、FastAPI 边界把领域异常转 HTTPException
- 提交规范给出 Good/Bad 对照，并写明"不用 Conventional Commits"（说明各家约定不同，必须写自己的）

**共同点**：两个标杆都远超 150 行（300-500 行），但每条都是该项目特有、agent 无从推断的规则——印证"长度不是硬指标，信噪比才是"。

## 四、社区实证研究（BetterClaw，2,500+ 仓库样本）

1. **篇幅**：超过 150 行边际收益递减，推理成本 +20-23% 且不提升任务成功率；技术上限 32 KiB。建议起步 30-50 行。
2. **LLM 生成是最大错误**：实测 8 项设置中 5 项任务成功率下降、每任务多 2.45-3.92 步——因为 LLM 会生成"模型本来就会做"的泛泛建议（"写干净的代码"、"遵循最佳实践"），纯属浪费上下文。
3. **Show, don't tell**：一个真实代码示例（含 CORRECT/WRONG 对照）胜过三段文字描述。
4. **推荐章节与行数预算**：项目栈（5-10）→ 构建/测试命令（5-10，精确到包管理器和 flag）→ 代码风格（10-20，带示例）→ 架构约束（5-15，目录规则/依赖方向）→ 边界（5-10，绝不动的东西）→ Git 工作流（5）。
5. **维护**："同一错误犯两次 → 复盘并更新 AGENTS.md"；约定变更的同一个 PR 里更新；过期的 AGENTS.md 比没有更糟（主动误导）。把它当代码维护，不当文档。
6. OpenAI 官方补充：AGENTS.md 该覆盖 repo 布局、运行方式、build/test/lint 命令、工程约定与 PR 期望、约束与禁止项、"完成"的定义与验证方式；文件过大时保持主文件精炼、按主题引用外部 md（如 code_review.md）。

## 五、安全红线（多来源一致）

**绝不写入 AGENTS.md**（按"可能公开/被用于训练"对待）：

- API key / token / 密码 / 任何凭据
- 带密码的数据库连接串
- 生产 IP、内网 URL
- 客户数据 / PII / 专有算法细节 / 漏洞细节

**替代写法**：记录秘密"存在哪、怎么取"（如：`.env`（gitignored），模板见 `.env.example`；生产配置由 GitHub Secrets 注入），而非秘密本身。提交前跑 secret 扫描。

## 六、综合结论：高级规格标准（本项目采用）

一份高质量 AGENTS.md = **只写本项目特有、agent 无从推断的规则**，且满足：

1. 每条规则可通过"删掉它 agent 会不会猜错？"测试——会猜错才留，猜不错的删
2. 命令区精确可复制（含包管理器、venv 解释器路径、平台差异如 PowerShell 无 `&&`）
3. 含项目专属的运行时事实（单位是 wei、API 硬上限、守卫语义、沙箱/环境约束）
4. 边界区明确"绝不动什么 + 秘密在哪"
5. "完成的定义"可执行（本地测试命令 → CI 门禁 → 部署方式）
6. Git/提交/部署流程写清（谁触发部署、agent 何时该等指令）
7. 篇幅以信噪比为纲：本项目取 ~100-150 行；标杆项目可以更长是因为条条皆特有规则
8. 无任何秘密；维护规则：约定变更同 PR 更新此文件

## 七、本项目落地决策

- 文件位置：仓库根 `AGENTS.md`（单仓无嵌套需求）
- 语言：中文（项目 commit/文档/用户沟通均为中文；命令与代码标识符保持原文）
- 内容取舍：优先收录本项目实证踩坑（系统 Python 无包、`_15m_markets` 未登记被拒单、贴线护栏语义、生产只能走 HTTP API/CI、push main 即自动部署、前端单文件格局、SIGNAL_INFO 口径源），而非通用工程常识
- 后续维护：新踩的"agent 会猜错"的坑 → 同 PR 追加进 AGENTS.md
