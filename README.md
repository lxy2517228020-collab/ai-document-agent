# AI 文档知识库工作台

一个可部署到 Streamlit Community Cloud 的 AI PDF 知识库问答产品原型。项目使用 Gemini API、LangChain、FAISS 和 Streamlit 构建，支持上传 PDF、自动建库、文档分析、引用溯源、对话反馈、历史导出和多场景问答。

## 在线访问

部署到 Streamlit Community Cloud 后，应用会生成公网 HTTPS 链接，格式通常为：

```text
https://<your-app-name>.streamlit.app
```

简历或作品集可使用：

```text
在线演示：https://<your-app-name>.streamlit.app
GitHub：https://github.com/<your-username>/<your-repo>
```

## 功能介绍

- 上传一个或多个 PDF，并自动读取文本、切分文档、生成向量和建立 FAISS 知识库
- 左侧知识库管理区展示 PDF 文件名、页数、上传时间和建库状态
- 支持四种使用场景，并为不同场景切换不同 system prompt：
  - 课程资料复习
  - 学术论文阅读
  - 企业制度问答
  - 岗位 JD 分析
- 支持知识库问答，并展示引用来源：
  - PDF 文件名
  - 页码
  - 相关原文片段
- 支持快捷文档分析：
  - 总结全文
  - 提取关键词
  - 生成知识点大纲
  - 生成 10 个复习问题
  - 生成 FAQ
- 支持回答反馈：有帮助 / 没帮助
- 支持数据看板：
  - 已上传文件数
  - 文档页数
  - 文本块数量
  - 提问次数
  - 有帮助反馈数
  - 没帮助反馈数
- 支持对话历史查看
- 支持将问答记录导出为 Markdown 或 TXT

## 技术栈

- Streamlit：网页界面
- Gemini API：聊天模型和 Embedding 模型
- LangChain：PDF 读取、文本切分、向量检索和问答编排
- FAISS：本地向量数据库
- PyPDF：PDF 文本解析
- python-dotenv：本地 `.env` 配置读取

## 项目结构

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

重要说明：本地 `.env` 文件只用于开发环境，已经被 `.gitignore` 忽略，不要上传到 GitHub。

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
GEMINI_CHAT_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
GEMINI_TEMPERATURE=1.0
```

也可以直接使用系统环境变量：

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY="your_google_api_key_here"
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

确认仓库至少包含：

```text
app.py
requirements.txt
README.md
.gitignore
.env.example
```

不要提交 `.env` 文件。`.gitignore` 已包含：

```text
.env
.streamlit/secrets.toml
.venv/
__pycache__/
*.pyc
.DS_Store
```

### 2. 创建 Streamlit 应用

1. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)。
2. 使用 GitHub 登录。
3. 点击 `New app`。
4. 选择你的 GitHub 仓库、分支和入口文件。
5. 入口文件填写：

```text
app.py
```

### 3. 配置 Secrets

在 Streamlit Cloud 应用页面进入：

```text
Settings -> Secrets
```

添加：

```toml
GOOGLE_API_KEY = "your_google_api_key_here"

# 可选：如果默认模型不可用，可以改成你的账号支持的模型
GEMINI_CHAT_MODEL = "gemini-3.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_TEMPERATURE = "1.0"
```

保存后重新部署或重启应用。部署成功后，Streamlit Cloud 会提供公网 HTTPS 链接：

```text
https://<your-app-name>.streamlit.app
```

## API Key 读取逻辑

代码会按以下顺序读取配置：

1. Streamlit Cloud Secrets：`st.secrets["GOOGLE_API_KEY"]`
2. 系统环境变量：`GOOGLE_API_KEY`
3. 本地 `.env` 文件中的 `GOOGLE_API_KEY`

因此：

- 本地开发推荐使用 `.env`
- Streamlit Cloud 部署必须使用 Cloud Secrets
- 不要把真实 API Key 写死在代码里
- 不要把 `.env` 上传到 GitHub

## requirements.txt

项目依赖包含：

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

## 常见问题

### PDF 没有读取到文本

如果 PDF 是扫描件或图片型 PDF，普通 PDF 文本解析器无法提取文字。请先对 PDF 做 OCR，再上传处理后的文件。

### Streamlit Cloud 提示缺少 GOOGLE_API_KEY

请检查 `Settings -> Secrets` 是否配置了：

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

保存后需要重新运行应用。

### Gemini 模型不可用

不同账号、地区和时间点可用模型可能不同。可以在 `.env` 或 Streamlit Secrets 中调整：

```toml
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_TEMPERATURE = "0.2"
```

## 简历项目介绍

项目名称：AI 文档知识库问答系统

项目链接：

```text
在线演示：https://<your-app-name>.streamlit.app
GitHub：https://github.com/<your-username>/<your-repo>
```

项目介绍：

```text
基于 Gemini API、LangChain、FAISS 和 Streamlit 构建的 AI PDF 文档知识库系统。系统支持多 PDF 上传、自动文本切分、Gemini Embedding 向量化、FAISS 本地检索、基于引用来源的问答生成、文档总结/关键词/大纲/复习题/FAQ 一键生成、回答反馈统计、对话历史和 Markdown/TXT 导出，并支持通过 Streamlit Community Cloud 公网部署。
```
