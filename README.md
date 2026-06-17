# AI 文档知识库工作台

一个可部署到 Streamlit Community Cloud 的 AI PDF 知识库问答产品原型。项目使用 Gemini API、LangChain、FAISS 和 Streamlit 构建，支持上传 PDF、自动建库、文档分析、引用溯源、对话反馈、历史导出和多场景问答。

## 在线访问

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

## API Key 读取逻辑

代码会按以下顺序读取配置：

1. Streamlit Cloud Secrets：`st.secrets["GOOGLE_API_KEY"]`
2. 系统环境变量：`GOOGLE_API_KEY`
3. 本地 `.env` 文件中的 `GOOGLE_API_KEY`

因此：

- 本地开发推荐使用 `.env`
- Streamlit Cloud 部署必须使用 Cloud Secrets
- 不会把真实 API Key 写死在代码里
- 不会把 `.env` 上传到 GitHub

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

