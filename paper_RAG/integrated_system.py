import os
import sys
import gradio as gr
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLMgeneration"))

from parse_pdf.parse_clean.parse_paper import DocumentParser
from parse_pdf.parse_clean.clean_to_md import DocumentCleaner
from parse_pdf.parse_clean.md_to_json import MDHeadingParser
from chunk_embedding.chunk import DocumentChunker
from chunk_embedding.embedding import ChunkEmbedder
from LLMgeneration import RAGGenerator


class RAGSystem:
    def __init__(self):
        self.faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
        self.parsed_pdf_dir = r"D:\AI\paper_RAG\parse_pdf\parsed_pdf"
        self.chunked_json_dir = r"D:\AI\paper_RAG\parse_pdf\parsed_pdf"
        self.source_docs_dir = r"D:\AI\paper_RAG\source_documents"

        os.makedirs(self.source_docs_dir, exist_ok=True)
        os.makedirs(self.faiss_dir, exist_ok=True)

        self.generator = None
        self.current_doc_name = None
        self._init_generator()

    def _init_generator(self):
        chunks_json = os.path.join(self.faiss_dir, "chunks.json")
        if os.path.exists(chunks_json):
            self.generator = RAGGenerator(self.faiss_dir, chunks_json, memory_window=5)
            self.current_doc_name = "当前文档"
            print(f"RAG系统初始化完成，文档: {self.current_doc_name}")
        else:
            print("RAG系统初始化完成，暂无文档")

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.stem

        print(f"\n{'='*60}")
        print(f"开始处理文档: {doc_name}")
        print(f"{'='*60}")

        parsed_dir = os.path.join(self.parsed_pdf_dir, f"{doc_name}_documents")
        if os.path.exists(parsed_dir):
            shutil.rmtree(parsed_dir)
        os.makedirs(parsed_dir, exist_ok=True)

        print("\n[1/6] 解析PDF...")
        parser = DocumentParser(output_dir=self.parsed_pdf_dir)
        parser.parse_pdf(str(pdf_path))
        print(f"  -> 解析完成: {parsed_dir}")

        md_file = os.path.join(self.parsed_pdf_dir, f"{doc_name}.md")
        print("\n[2/6] 清理并转换为Markdown...")
        cleaner = DocumentCleaner(parsed_dir)
        cleaner.process()
        print(f"  -> Markdown完成: {md_file}")

        headings_file = os.path.join(self.parsed_pdf_dir, f"{doc_name}_headings.json")
        print("\n[3/6] 提取标题结构...")
        md_parser = MDHeadingParser(md_file)
        content = md_parser.load_md()
        headings = md_parser.find_headings(content)
        tree = md_parser.build_tree_with_content()
        with open(headings_file, 'w', encoding='utf-8') as f:
            json.dump({'headings': headings, 'tree': tree}, f, ensure_ascii=False, indent=2)
        print(f"  -> 标题提取完成: {len(headings)} 个标题")

        chunks_file = os.path.join(self.parsed_pdf_dir, f"{doc_name}_chunks.json")
        print("\n[4/6] 父子块切分...")
        chunker = DocumentChunker(headings_file)
        chunker.process()
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({'chunks': chunker.chunks}, f, ensure_ascii=False, indent=2)
        parent_count = sum(1 for c in chunker.chunks if c.get('is_parent', False))
        child_count = sum(1 for c in chunker.chunks if not c.get('is_parent', False))
        print(f"  -> 切块完成: {len(chunker.chunks)} 块 (父{parent_count}/子{child_count})")

        print("\n[5/6] 向量化嵌入...")
        embedder = ChunkEmbedder(self.chunked_json_dir)
        result = embedder.process(self.faiss_dir)
        print(f"  -> 向量化完成: {result['total_chunks']} chunks, 维度 {result['embedding_dim']}")

        print("\n[6/6] 重新初始化RAG生成器...")
        chunks_json = os.path.join(self.faiss_dir, "chunks.json")
        self.generator = RAGGenerator(self.faiss_dir, chunks_json, memory_window=5)
        self.current_doc_name = doc_name

        dest_path = os.path.join(self.source_docs_dir, f"{doc_name}.pdf")
        shutil.copy2(str(pdf_path), dest_path)

        print(f"\n{'='*60}")
        print(f"文档处理完成: {doc_name}")
        print(f"{'='*60}")

        return {
            "status": "success",
            "doc_name": doc_name,
            "pages": len(list(Path(parsed_dir).glob("page_*.json"))),
            "headings": len(headings),
            "chunks": len(chunker.chunks),
            "parent_chunks": parent_count,
            "child_chunks": child_count,
            "vector_dim": result['embedding_dim']
        }

    def list_documents(self) -> List[str]:
        docs = []
        for f in Path(self.source_docs_dir).glob("*.pdf"):
            docs.append(f.stem)
        return docs

    def qa(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.generator:
            return {
                "query": query,
                "answer": "系统未初始化，请先上传文档",
                "rewritten_query": None,
                "retrieved": [],
                "retrieved_count": 0
            }
        return self.generator.qa(query, top_k=top_k, show_retrieved=True)

    def clear_memory(self):
        if self.generator:
            self.generator.clear_memory()
            self.generator.clear_long_term()
        return "记忆已清空"


rag_system = RAGSystem()


def chat_fn(query, history, top_k):
    if not query.strip():
        return "", history

    result = rag_system.qa(query, top_k=int(top_k))
    answer = result["answer"]

    retrieved_info = ""
    if result.get("retrieved"):
        retrieved_info = "\n\n📚 参考片段：\n"
        for i, chunk in enumerate(result["retrieved"][:3], 1):
            title = chunk.get("h1_title", "") or chunk.get("h2_title", "") or chunk.get("h3_title", "") or "正文"
            retrieved_info += f"[{i}] {title}\n"

    full_response = answer + retrieved_info

    if result.get("rewritten_query"):
        full_response = f"🔄 Query改写: {result['rewritten_query']}\n\n" + full_response

    history.append((query, full_response))
    return "", history


def clear_fn():
    rag_system.clear_memory()
    return []


def upload_fn(file_obj):
    if file_obj is None:
        return {"status": "error", "message": "未选择文件"}, "暂无文档"

    try:
        result = rag_system.process_document(file_obj.name)
        doc_md = f"**{result['doc_name']}**\n\n- 页数: {result['pages']}\n- 标题: {result['headings']}\n- 切块: {result['chunks']} (父{result['parent_chunks']}/子{result['child_chunks']})\n- 向量维度: {result['vector_dim']}"
        return result, doc_md
    except Exception as e:
        return {"status": "error", "message": str(e)}, "处理失败"


def list_docs_fn():
    docs = rag_system.list_documents()
    return [{"name": d} for d in docs]


def build_ui():
    with gr.Blocks(title="论文RAG智能问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 论文RAG智能问答系统\n\n支持文档上传、向量数据库更新、多轮对话")
        gr.Markdown("---")

        with gr.Tab("问答"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=450, show_copy_button=True, type='messages')
                    msg = gr.Textbox(placeholder="输入问题，按回车发送...", label="问题", lines=2)
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空记忆")
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 当前文档")
                    doc_info = gr.Markdown("暂无文档")
                    gr.Markdown("### ⚙️ 设置")
                    top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K")

            submit_btn.click(chat_fn, inputs=[msg, chatbot, top_k_slider], outputs=[msg, chatbot])
            msg.submit(chat_fn, inputs=[msg, chatbot, top_k_slider], outputs=[msg, chatbot])
            clear_btn.click(clear_fn, outputs=[chatbot])

        with gr.Tab("文档管理"):
            gr.Markdown("### 📤 上传新文档")
            gr.Markdown("支持PDF格式，上传后自动完成解析、切块、向量化和索引更新")

            with gr.Row():
                file_input = gr.File(label="选择PDF文件", file_types=[".pdf"])
                upload_btn = gr.Button("上传并处理", variant="primary")

            status_output = gr.JSON(label="处理状态")

            upload_btn.click(
                upload_fn,
                inputs=[file_input],
                outputs=[status_output, doc_info]
            )

            gr.Markdown("### 📚 已上传文档")
            docs_list = gr.JSON(value=list_docs_fn)

        with gr.Tab("使用说明"):
            gr.Markdown("""
            ## 使用指南

            ### 1. 上传文档
            - 点击「文档管理」标签
            - 选择PDF文件并点击「上传并处理」
            - 系统自动完成：解析 → 清理 → 切块 → 向量化 → 更新索引

            ### 2. 问答交互
            - 在「问答」标签输入问题
            - 系统支持多轮对话，可连续追问
            - 使用「清空记忆」重置对话历史

            ### 3. 核心技术
            - **父子块检索**: 父块2000字符，子块500字符
            - **Query改写**: 处理指代词（它、这个等）
            - **长短期记忆**: FAISS长期向量 + 滑动窗口短期
            """)

    return demo


if __name__ == "__main__":
    print("启动RAG智能问答系统...")
    print(f"FAISS目录: {r'D:\AI\paper_RAG\chunk_embedding\faiss_db'}")
    print(f"源文档目录: {r'D:\AI\paper_RAG\source_documents'}")
    print()

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
