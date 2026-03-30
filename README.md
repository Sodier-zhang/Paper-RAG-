# 模块化、全链路透明化的学术论文 RAG（检索增强生成）系统
本项目是一个专注于 RAG（检索增强生成）全流程细粒度拆解与分步优化的实验性系统，旨在通过对文件转换每一个关键环节的模块化处理，实现对学术论文处理流程的深度掌控与精准测试。系统基于 FAISS 向量数据库与通义千问（Qwen）大模型构建，核心流程涵盖了从原始 PDF 到最终问答的完整生命周期：首先利用解析工具将 PDF 转换为结构化的 Document 原始文档，随后经过数据清洗转化为标准 Markdown 格式以保留论文层级，接着提取标题树并实施“父子块”切分策略，最后通过向量化嵌入构建检索索引。此外，系统还集成混合检索、Query 改写及长短期记忆机制，在确保文件转换流程透明、可追溯的同时，为研究人员提供极高准确度的学术问答体验。

# 🌟 项目亮点
本项目不仅是一个工具，更是一个 RAG 流程的“实验台”：

全流程拆解：清晰展示从 原始PDF -> Document对象 -> 清洗后Markdown -> 结构化JSON -> 向量索引 的每一步转换细节。

分步测试：支持对解析、切块、检索、生成等各个阶段进行独立优化与测试，让 RAG 不再是黑盒。

# ✨ 核心特性
🔍 深度文档解析与转换
结构化提取：集成 LlamaParse 提取 PDF 内容，自动识别标题层级并清理格式。

流程透明化：通过解析、清洗、转换等步骤，确保论文的 H1/H2/H3 逻辑结构在转换过程中不丢失。

# 🎯 精准检索策略
父子块检索 (Parent-Child Retrieval)：

子块 (500字)：用于高精度向量匹配，确保检索的灵敏度。

父块 (2000字)：检索匹配后自动回溯至父块，为 LLM 提供完整的上下文背景。

混合检索与重排：通过 HybridRetriever 结合 BM25 关键词检索与 FAISS 向量检索，并利用 RRF (Reciprocal Rank Fusion) 算法平衡检索权重，最后支持 GTE 模型语义重排。

# 🧠 智能对话管理
Query 处理：支持 Query 改写（自动补全指代词）与 Query 扩展（基于 HyDE 生成假想答案），显著提升召回率。

长短期记忆融合：

短期记忆：基于滑动窗口记录近 5 轮对话上下文。

长期记忆：自动生成对话摘要并存储，支持跨段落的长线追问。

# 💻 交互式 UI
可视化界面：基于 Gradio 构建，支持文档上传、实时处理进度显示及直观的问答交互。

# 🏗️ 技术架构
**项目模块划分清晰，代码结构如下：** 

parse_pdf/：调用 LlamaParse 进行 PDF 转 Markdown 及结构化清洗。

chunk_embedding/：执行父子块切分，调用 DashScope 生成向量并构建 FAISS 索引。

retrival/：混合检索逻辑，包含 BM25、RRF 融合及召回率测试模块。

LLMgeneration/：基于对话记忆与 RAG 核心逻辑的答案生成模块。

integrated_system.py：全流程集成的 Gradio Web 应用。

# 🚀 快速开始
**1. 环境准备**

确保已安装 Python 3.10+，然后安装项目依赖：

pip install -r requirements.txt

**2. 配置 API Key**
在环境变量中设置以下 Key（或在代码对应处填写）：

DASHSCOPE_API_KEY: 阿里云灵积平台 Key（用于 Qwen 模型和 Embedding）。

LLAMAPARSE_API_KEY: LlamaIndex 解析服务 Key。

**3. 运行系统**
你可以根据需求选择不同的运行模式：

全功能 Web 模式（推荐）：支持在线上传 PDF 并实时解析。

python paper_RAG/integrated_system.py

API 模式：基于 FastAPI 部署。

python paper_RAG/api.py

python paper_RAG/LLMgeneration/LLMgeneration.py --interactive

# 📂 项目结构

<img width="623" height="280" alt="image" src="https://github.com/user-attachments/assets/80bf810e-783b-4195-8dfb-b9045bc49bb3" />

# 🛠️ 关键技术细节
**切块逻辑：** DocumentChunker 能够识别 Markdown 标题层级，避免传统长度切分带来的语义截断，最大程度保留学术论文的逻辑严密性。

**检索增强：** 采用混合检索架构，有效平衡了关键词精确匹配与向量语义检索的优缺点，适用于术语众多的学术场景。

希望这个项目能帮助你更好地理解和优化 RAG 流程！欢迎在 Issue 中交流意见。
