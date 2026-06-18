# AI Document Agent

一个基于 Gemini API、LangChain、FAISS 和 Streamlit 的 AI 文档任务 Agent 项目。它从传统“上传 PDF 后问答”的 RAG 系统升级为可以理解自然语言任务、自动识别 intent、规划执行步骤、调用工具、多步骤执行、展示过程并输出结构化结果的文档智能体产品原型。

> Demo 仅用于学习、作品集展示和产品原型验证，不应用于法律、医疗、金融等高风险决策。

## 产品定位

AI Document Agent 面向课程资料、学术论文、企业制度、岗位 JD、报告和合同等 PDF 文档场景。用户不需要手动选择功能，只需要输入自然语言任务，例如“总结这份 PDF”“生成复习题”“对比两份文档差异”“提取合同风险点”，Agent 会自动规划并调用合适工具完成任务。

## 核心功能

- 多 PDF 上传与知识库管理
- PDF 文本解析、切分、Gemini Embedding 向量化
- FAISS 本地向量检索
- Intent Router 任务识别
- Agent Planner 自动生成执行计划
- Agent Tools 多工具调用
- Agent Executor 多步骤执行与过程展示
- 引用溯源：正文使用上标引用，悬停可查看文件名、页码和片段预览；下方卡片展示完整来源
- 输出详细程度控制：默认简洁版，也可切换标准版、详细版，用于平衡回答质量和响应速度
- 场景化模式：
  - 课程资料复习
  - 学术论文阅读
  - 企业制度问答
  - 岗位 JD 分析
  - 报告/合同审阅
- 历史任务记录与继续追问
- 用户反馈闭环
- Markdown/TXT 导出
- Streamlit Community Cloud 部署适配

## Agent 支持的 Intent

系统会把用户自然语言任务识别为以下 intent：

- `answer_question`
- `summarize_document`
- `extract_key_points`
- `generate_faq`
- `generate_quiz`
- `compare_documents`
- `analyze_jd`
- `extract_risks`
- `create_study_guide`
- `export_result`
- `unknown`

识别结果会返回结构化 JSON：

```json
{
  "intent": "summarize_document",
  "task_goal": "总结这份 PDF",
  "target_documents": [],
  "need_retrieval": true,
  "need_citation": true,
  "steps": ["提取文档核心内容", "总结主题和结构", "输出带引用总结"]
}
```

## Agent 工具

项目实现了以下工具：

- `retrieve_passages(query, k)`：从 FAISS 检索相关文本片段
- `summarize_document(doc_name)`：总结文档
- `extract_key_points(doc_name)`：提取核心知识点
- `generate_faq(doc_name)`：生成 FAQ
- `generate_quiz(doc_name, num_questions)`：生成复习题
- `compare_documents(doc_a, doc_b)`：对比两份文档
- `analyze_jd(doc_name)`：分析岗位 JD
- `extract_risks(doc_name)`：提取风险点、问题点、待确认事项
- `create_study_guide(doc_name)`：生成复习大纲
- `export_to_markdown(content)`：导出 Markdown 文本
- `evaluate_answer(answer, sources)`：评估回答是否有引用、是否基于文档、置信度如何

## 技术架构

```text
用户上传 PDF
  ↓
rag.py：PDF 解析、文本切分、Embedding、FAISS 建库
  ↓
用户输入自然语言任务
  ↓
agent.py：Intent Router 识别任务
  ↓
agent.py：Planner 生成执行计划
  ↓
tools.py：调用 RAG / 总结 / FAQ / Quiz / JD 分析 / 风险提取等工具
  ↓
agent.py：Executor 汇总工具结果并评估可信度
  ↓
app.py：Streamlit 展示执行过程、引用来源、最终结果、反馈和导出
```

## 项目结构

```text
.
├── app.py
├── agent.py
├── tools.py
├── rag.py
├── prompts.py
├── memory.py
├── export_utils.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## 本地运行

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 GOOGLE_API_KEY

复制示例环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_FAST_MODEL=gemini-2.5-flash-lite
GEMINI_STABLE_MODEL=gemini-2.5-flash
GEMINI_QUALITY_MODEL=gemini-2.5-pro
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
GEMINI_TEMPERATURE=1.0
```

也可以使用系统环境变量：

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问：

```text
http://localhost:8501
```

## Streamlit Community Cloud 部署

### 1. 上传到 GitHub

仓库至少需要包含：

```text
app.py
agent.py
tools.py
rag.py
prompts.py
memory.py
export_utils.py
requirements.txt
README.md
.gitignore
.env.example
```

不要提交 `.env`。`.gitignore` 已默认忽略：

```text
.env
.streamlit/secrets.toml
.venv/
__pycache__/
*.pyc
.DS_Store
```

### 2. 创建 Cloud 应用

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 使用 GitHub 登录。
3. 点击 `New app`。
4. 选择仓库、分支和入口文件。
5. 入口文件填写：

```text
app.py
```

### 3. 配置 Secrets

进入：

```text
Settings -> Secrets
```

添加：

```toml
GOOGLE_API_KEY = "your_google_api_key_here"

# 可选：如果默认模型不可用，可以替换为你的账号可用模型
GEMINI_FAST_MODEL = "gemini-2.5-flash-lite"
GEMINI_STABLE_MODEL = "gemini-2.5-flash"
GEMINI_QUALITY_MODEL = "gemini-2.5-pro"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_TEMPERATURE = "1.0"
```

保存后重启应用。部署成功后，Streamlit 会生成公网 HTTPS 地址：

```text
https://<your-app-name>.streamlit.app
```

## API Key 读取逻辑

代码会按以下顺序读取：

1. Streamlit Cloud Secrets：`st.secrets["GOOGLE_API_KEY"]`
2. 系统环境变量：`GOOGLE_API_KEY`
3. 本地 `.env` 文件

项目不会把 API Key 写死在代码中。

## 安全与可信度设计

- 不提交 `.env` 或本地 secrets
- 对无文档依据的问题，要求回答“文档中未找到相关信息”
- 回答尽量展示引用来源
- 删除或清空知识库前需要勾选确认
- 页面提示上传文件大小和 OCR 限制
- `evaluate_answer` 会评估引用来源、文档依据和置信度
- Demo 不用于法律、医疗、金融等高风险决策

## 性能与稳定性优化

首次上传 PDF 会比较慢，因为系统需要完成 PDF 文本解析、文本切分、Gemini Embedding 生成和 FAISS 向量库构建。后续提问会更快，因为同一会话中会复用已建立的向量库；同一批 PDF 再次上传时，也会优先复用 session cache，避免重复 Embedding。

Gemini API 偶尔可能返回 `503 UNAVAILABLE`、`429 rate limit` 或 timeout。这通常来自模型高峰期、免费额度繁忙、临时限流或网络抖动。项目通过以下机制提升演示稳定性：

- 模型模式切换：sidebar 支持快速模式、稳定模式和高质量模式。
- 模型 fallback：如果当前生成模型返回 `404 NOT_FOUND`，系统不会重试，而是自动切换到 `gemini-2.5-flash`，并在执行过程里记录原始模型、实际模型和 fallback 原因。
- 快速路由：常见任务使用规则识别 intent 和固定 plan，跳过 LLM Router / Planner。
- 减少 Gemini 调用：总结、FAQ、知识点、JD 分析等常见任务通常只需要 1 次生成调用；默认 `evaluate_answer` 使用规则评估，不额外调用 LLM。
- Retry 机制：LLM 和 Embedding 调用遇到 503、429、timeout、temporarily unavailable 时，会按 2 秒、4 秒、8 秒指数退避重试，最多重试 3 次。
- 友好失败提示：重试后仍失败时，页面会提示“当前 Gemini 模型繁忙，请稍后重试，或切换到快速模式”，不会让 Streamlit 崩溃。
- 结果缓存：同一个用户任务、同一批文档、同一场景、同一模型模式和同一输出详细程度下，如果已经执行过，会直接读取历史结果。
- 可观测指标：结果下方展示响应时间、Gemini 调用次数、是否命中缓存和 retry 次数。
- 输出长度控制：sidebar 支持简洁版、标准版和详细版；总结、知识点、FAQ、复习题和 JD 分析会把长度约束写入 prompt，减少长回答导致的等待时间。

## 常见 Gemini 错误

- `404 NOT_FOUND`：模型名称不可用，或当前 API 版本不支持该模型的 `generateContent`。这类错误重试没有意义，项目会自动 fallback 到 `gemini-2.5-flash`。
- `503 UNAVAILABLE`：模型高峰期繁忙或服务暂时不可用。可以稍后重试，或切换到快速模式。
- `429 rate limit`：额度或频率限制。可以降低调用频率、稍后重试，或检查 API 额度。

## requirements.txt

核心依赖：

```text
streamlit
python-dotenv
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
faiss-cpu
pypdf
```

## 项目亮点

- 从 RAG 问答升级为文档任务 Agent
- 支持自然语言任务入口，而不是固定按钮菜单
- Intent Router 输出结构化 JSON，便于展示产品逻辑
- Planner 自动生成执行计划，过程可解释
- 工具调用过程可视化，适合 Demo 和面试讲解
- 多场景 system prompt，贴近真实业务
- 输出长度可控，可在回答完整度和响应速度之间切换
- 用户反馈和数据看板形成产品闭环
- 适配 Streamlit Community Cloud，可生成公网 HTTPS Demo
- 针对 Gemini 503/429 等高峰期不稳定问题，加入 retry、缓存、模型切换和快速路由

## 适合写进简历的项目描述

产品经理简历描述：

```text
基于 Gemini API、LangChain、FAISS 和 Streamlit 搭建 AI Document Agent，支持 PDF 上传、RAG 检索、任务意图识别、工具调用、多步骤执行、引用溯源、智能摘要、FAQ/复习题生成、JD 分析和用户反馈闭环。
```

性能稳定性补充：

```text
针对大模型 API 高峰期不稳定问题，设计 retry、fallback、模型切换和缓存机制，提升 AI Agent Demo 的可用性与演示稳定性。
```

技术/数据方向简历描述：

```text
设计并实现 AI Document Agent 原型，将 PDF RAG 问答系统升级为多步骤文档任务执行器。项目使用 Gemini API 作为 LLM 与 Embedding 接口，基于 LangChain 封装检索和工具链，使用 FAISS 构建本地向量库，并通过 Streamlit 展示 intent 识别、任务规划、工具调用、中间结果、引用来源和反馈数据看板。
```

项目链接格式：

```text
在线演示：https://<your-app-name>.streamlit.app
GitHub：https://github.com/<your-username>/<your-repo>
```

## 后续优化方向

- 使用 LangGraph 实现更标准的 Agent 状态机
- 增加 OCR 支持，处理扫描版 PDF
- 增加持久化存储，保存上传文件、向量库和历史记录
- 支持多知识库、多用户和权限控制
- 增加更严格的 JSON Schema 输出校验
- 增加异步任务队列，优化大文档处理体验
- 支持更多导出格式，如 Word、PPT、CSV
- 增加自动评测集，评估回答准确率和引用质量
