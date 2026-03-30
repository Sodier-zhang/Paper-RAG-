# 模块化、全链路透明化的学术论文 RAG（检索增强生成）系统
本项目是一个专注于 RAG（检索增强生成）全流程细粒度拆解与分步优化的实验性系统，旨在通过对文件转换每一个关键环节的模块化处理，实现对学术论文处理流程的深度掌控与精准测试。系统基于 FAISS 向量数据库与通义千问（Qwen）大模型构建，核心流程涵盖了从原始 PDF 到最终问答的完整生命周期：首先利用解析工具将 PDF 转换为结构化的 Document 原始文档，随后经过数据清洗转化为标准 Markdown 格式以保留论文层级，接着提取标题树并实施“父子块”切分策略，最后通过向量化嵌入构建检索索引。此外，系统还集成混合检索、Query 改写及长短期记忆机制，在确保文件转换流程透明、可追溯的同时，为研究人员提供极高准确度的学术问答体验。

📚 Paper-RAG: 智能学术论文问答系统
这是一个基于 LlamaIndex 思想、FAISS 向量数据库和 通义千问 (Qwen) 大模型的学术论文 RAG 系统。系统支持 PDF 深度解析、父子块检索、Query 改写以及多轮对话记忆，旨在为研究人员提供精准的论文内容问答体验。

✨ 核心特性
深度文档解析：集成 LlamaParse 提取 PDF 内容，自动识别标题层级并清理格式。

父子块检索策略：

父块 (2000字)：保持上下文完整性。

子块 (500字)：用于高精度向量匹配，检索后自动回溯至父块提供给 LLM。

混合检索与重排：结合 BM25 关键词检索与 FAISS 向量检索，并支持 GTE 模型进行语义重排。

智能 Query 处理：

Query 改写：自动补全多轮对话中的指代词（如“它”、“这个”），提升检索精度。

Query 扩展：生成学术相关的关键词和假想答案 (HyDE) 增强召回。

长短期记忆融合：

短期记忆：基于滑动窗口（默认5轮）记录对话上下文。

长期记忆：自动生成对话摘要并存储，支持跨段落的长线追问。

交互式 UI：基于 Gradio 构建，支持文档上传、实时处理进度显示及可视化问答界面。

🏗️ 技术架构
项目模块划分清晰，易于扩展：

parse_pdf/：调用 LlamaParse 进行 PDF 转 Markdown 及结构化处理。

chunk_embedding/：执行父子块切分，调用 DashScope API 生成向量并构建 FAISS 索引。

retrival/：混合检索逻辑，包含 BM25、RRF 融合及重排算法。

LLMgeneration/：基于对话记忆生成的 RAG 核心逻辑。

integrated_system.py：全流程集成的 Gradio Web 应用。

🚀 快速开始
1. 环境准备
确保已安装 Python 3.10+，然后安装依赖：

Bash
pip install -r requirements.txt
2. 配置 API Key
在环境变量中设置以下 Key（或直接在代码中填入）：

DASHSCOPE_API_KEY: 阿里云灵积平台 Key（用于 Qwen 模型和 Embedding）。

LLAMAPARSE_API_KEY: LlamaIndex 解析服务 Key。

3. 运行系统
你可以根据需求选择运行模式：

全功能 Web 模式（推荐）：
支持在线上传 PDF 并实时解析问答。

Bash
python paper_RAG/integrated_system.py
API 模式：
基于 FastAPI 部署，可供前端或第三方调用。

Bash
python paper_RAG/api.py
命令行交互模式：

Bash
python paper_RAG/LLMgeneration/LLMgeneration.py --interactive
📂 项目结构
Plaintext
paper_RAG/
├── chunk_embedding/      # 切块与向量化模块
├── LLMgeneration/        # 大模型生成与记忆管理
├── parse_pdf/            # PDF解析与清洗
├── retrival/             # 混合检索测试与优化
├── source_documents/     # 存放上传的原始PDF
├── api.py                # FastAPI 接口服务
├── integrated_system.py  # Gradio 综合演示系统
└── requirements.txt      # 项目依赖清单
🛠️ 关键技术细节
切块逻辑：使用 DocumentChunker 识别 Markdown 标题层级，根据 H1/H2/H3 结构进行逻辑切分，避免硬切断带来的语义损失。

检索增强：通过 HybridRetriever 实现 RRF (Reciprocal Rank Fusion)，有效平衡了关键词匹配的硬度和向量检索的深度。
