import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLMgeneration"))
from LLMgeneration import RAGGenerator

faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
chunks_json = r"D:\AI\paper_RAG\chunk_embedding\faiss_db\chunks.json"

print("初始化RAG生成器...")
print(f"FAISS目录: {faiss_dir}")
print(f"Chunks文件: {chunks_json}")
generator = RAGGenerator(faiss_dir, chunks_json, memory_window=5)

def chat(query, history):
    if not query.strip():
        return "", history

    result = generator.qa(query, top_k=5, show_retrieved=True)

    answer = result["answer"]

    retrieved_info = ""
    if result.get("retrieved"):
        retrieved_info = "\n\n📚 参考片段：\n"
        for i, chunk in enumerate(result["retrieved"][:3], 1):
            title = chunk.get("h1_title", "") or chunk.get("h2_title", "") or chunk.get("h3_title", "") or "正文"
            source = chunk.get("source", "retrieved")
            score = chunk.get("score", 0)
            retrieved_info += f"[{i}][{source}] {title} (相关度: {score:.2f})\n"

    full_response = answer + retrieved_info

    if result.get("rewritten_query"):
        full_response = f"🔄 Query改写: {result['rewritten_query']}\n\n" + full_response

    history.append((query, full_response))
    return "", history

def clear():
    generator.clear_memory()
    generator.clear_long_term()
    return [], "记忆已清空"

def main():
    with gr.Blocks(title="论文RAG问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📚 论文RAG智能问答系统\n\n支持多轮对话、Query改写、长短期记忆融合")
        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, show_copy_button=True)
                msg = gr.Textbox(
                    placeholder="输入问题，按回车发送...",
                    label="问题",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空记忆")
                gr.Markdown("---")
                gr.Markdown("**💡 使用提示**")
                gr.Markdown("- 支持多轮对话，可连续追问")
                gr.Markdown("- 使用'清空记忆'按钮重置对话历史")
                gr.Markdown("- 回答基于论文内容，引用编号来自检索片段")

        submit_btn.click(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(clear, outputs=[chatbot, msg])

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
