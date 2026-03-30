import os
import re
import json
import dashscope
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from dashscope import Generation, TextEmbedding


class ConversationMemory:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def add(self, query: str, answer: str):
        self.history.append({"query": query, "answer": answer})

    def get_history_text(self) -> str:
        if not self.history:
            return ""
        history_text = ""
        for i, turn in enumerate(self.history, 1):
            history_text += f"轮次{i}: 问: {turn['query']}\n答: {turn['answer']}\n"
        return history_text

    def get_recent_queries(self) -> List[str]:
        return [turn["query"] for turn in self.history]


class RAGGenerator:
    def __init__(self, faiss_dir: str, chunks_json: str, api_key: str = None, memory_window: int = 5):
        self.faiss_dir = Path(faiss_dir)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = self.api_key

        self.index = faiss.read_index(str(self.faiss_dir / "faiss.index"))
        with open(self.faiss_dir / chunks_json, "r", encoding="utf-8") as f:
            self.chunks_data = json.load(f)

        self.memory = ConversationMemory(window_size=memory_window)
        self.long_term_memory = []

    def _remove_proxy(self):
        self.saved_http = os.environ.pop("HTTP_PROXY", None)
        self.saved_https = os.environ.pop("HTTPS_PROXY", None)
        self.saved_no_proxy = os.environ.pop("NO_PROXY", None)

    def _restore_proxy(self):
        if self.saved_http:
            os.environ["HTTP_PROXY"] = self.saved_http
        if self.saved_https:
            os.environ["HTTPS_PROXY"] = self.saved_https
        if self.saved_no_proxy:
            os.environ["NO_PROXY"] = self.saved_no_proxy

    def embed_query(self, text: str) -> List[float]:
        self._remove_proxy()
        try:
            response = TextEmbedding.call(
                model=TextEmbedding.Models.text_embedding_v2,
                input=[text]
            )
            if response.status_code == 200:
                return response.output["embeddings"][0]["embedding"]
            else:
                raise Exception(f"Embedding调用失败: {response.message}")
        finally:
            self._restore_proxy()

    def rewrite_query(self, query: str, history_text: str) -> str:
        if not history_text:
            return query

        self._remove_proxy()
        try:
            prompt = f"""你是一个学术论文问答系统的Query改写专家。

任务：将包含指代（他、她、它、这个、那个、上述、该等）的口语化问题，改写为完整、明确的学术搜索查询。

历史对话：
{history_text}

当前问题：{query}

要求：
1. 将指代词替换为具体的学术术语
2. 补全被省略的核心概念
3. 保持学术查询的准确性
4. 只输出改写后的查询语句，不要解释

改写结果："""

            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )

            if response.status_code == 200:
                rewritten = response.output["text"].strip()
                return rewritten if rewritten else query
            else:
                return query
        finally:
            self._restore_proxy()

    def retrieve_from_long_term(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.long_term_memory:
            return []

        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        memories = self.long_term_memory
        if len(memories) > 50:
            memory_texts = [m.get("content", "")[:500] for m in memories]
        else:
            memory_texts = [m.get("content", "") for m in memories]

        if not memory_texts:
            return []

        embeddings = []
        batch_size = 10
        for i in range(0, len(memory_texts), batch_size):
            batch = memory_texts[i:i+batch_size]
            self._remove_proxy()
            try:
                resp = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v2,
                    input=batch
                )
                if resp.status_code == 200:
                    embeddings.extend([e["embedding"] for e in resp.output["embeddings"]])
            finally:
                self._restore_proxy()

        if not embeddings:
            return []

        memory_matrix = np.array(embeddings).astype('float32')
        distances, indices = self.index.search(memory_matrix, min(top_k, len(memories)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(memories):
                results.append({
                    "index": int(idx),
                    "score": float(dist),
                    "content": memories[idx].get("content", ""),
                    "source": "long_term"
                })
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_vector, top_k * 3)

        retrieved = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks_data[idx]
            content = chunk.get("content", "")

            if not chunk.get("is_parent", False) and chunk.get("parent_index") is not None:
                parent_idx = chunk.get("parent_index")
                if parent_idx is not None and parent_idx < len(self.chunks_data):
                    parent_chunk = self.chunks_data[parent_idx]
                    content = parent_chunk.get("content", "")
                elif chunk.get("parent_content"):
                    content = chunk.get("parent_content", "")

            retrieved.append({
                "index": int(idx),
                "score": float(dist),
                "h1_title": chunk.get("h1_title", ""),
                "h2_title": chunk.get("h2_title", ""),
                "h3_title": chunk.get("h3_title", ""),
                "content": content,
                "is_parent": chunk.get("is_parent", False),
                "source": "faiss"
            })

        return retrieved[:top_k]

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                       history_text: str = "", use_rag: bool = True) -> str:
        self._remove_proxy()
        try:
            if use_rag and retrieved_chunks:
                context_parts = []
                for i, chunk in enumerate(retrieved_chunks, 1):
                    title = chunk.get("h1_title", "") or chunk.get("h2_title", "") or chunk.get("h3_title", "")
                    source = chunk.get("source", "retrieved")
                    if title:
                        context_parts.append(f"[{i}][{source}] {title}:\n{chunk['content']}")
                    else:
                        context_parts.append(f"[{i}][{source}] {chunk['content']}")

                context = "\n\n".join(context_parts)

                if history_text:
                    prompt = f"""基于以下检索信息和对话历史，回答问题。注意利用历史信息理解指代词。

对话历史：
{history_text}

检索信息：
{context}

问题：{query}

要求：
1. 根据检索信息回答，如果信息不足则说明
2. 理解对话历史中的指代（如"它"、"这个"指代上文提到的内容）
3. 用中文回答，保持学术风格
4. 引用来源编号

回答："""
                else:
                    prompt = f"""基于以下检索信息，回答问题。

检索信息：
{context}

问题：{query}

要求：
1. 根据检索信息回答，如果信息不足则说明
2. 用中文回答，保持学术风格
3. 引用来源编号

回答："""

            else:
                if history_text:
                    prompt = f"""基于对话历史回答问题。

对话历史：
{history_text}

问题：{query}

要求：用中文回答。"""
                else:
                    prompt = f"""问题：{query}

请用中文回答。"""

            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )

            if response.status_code == 200:
                return response.output["text"].strip()
            else:
                return f"生成失败: {response.message}"

        finally:
            self._restore_proxy()

    def add_to_long_term(self, query: str, answer: str, retrieved: List[Dict]):
        summary_prompt = f"请用一句话概括以下内容的主要信息（不超过50字）：\n\n问：{query}\n答：{answer}"
        self._remove_proxy()
        try:
            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=summary_prompt,
                max_tokens=100,
                temperature=0.3
            )
            if response.status_code == 200:
                summary = response.output["text"].strip()
            else:
                summary = answer[:100]
        finally:
            self._restore_proxy()

        retrieved_content = " ".join([r.get("content", "")[:200] for r in retrieved[:2]])
        self.long_term_memory.append({
            "query": query,
            "answer": answer,
            "summary": summary,
            "content": retrieved_content
        })

        if len(self.long_term_memory) > 100:
            self.long_term_memory = self.long_term_memory[-100:]

    def qa(self, query: str, top_k: int = 5, show_retrieved: bool = True) -> Dict[str, Any]:
        history_text = self.memory.get_history_text()
        rewritten_query = self.rewrite_query(query, history_text)
        original_query = query
        query_to_use = rewritten_query if rewritten_query != query else query

        long_term_results = self.retrieve_from_long_term(query_to_use, top_k=2)

        retrieved = self.retrieve(query_to_use, top_k=top_k)

        if long_term_results:
            for lt in long_term_results:
                retrieved.append(lt)

        answer = self.generate_answer(query, retrieved, history_text, use_rag=True)

        self.memory.add(query, answer)
        self.add_to_long_term(query, answer, retrieved)

        result = {
            "query": query,
            "rewritten_query": rewritten_query if rewritten_query != query else None,
            "retrieved_count": len(retrieved),
            "answer": answer,
            "retrieved": retrieved if show_retrieved else None
        }

        return result

    def clear_memory(self):
        self.memory = ConversationMemory(window_size=self.memory.window_size)

    def clear_long_term(self):
        self.long_term_memory = []


def main():
    import sys

    if len(sys.argv) < 2:
        print("用法: python LLMgeneration.py [问题]")
        print("示例: python LLMgeneration.py 论文的主要研究内容是什么？")
        print("示例: python LLMgeneration.py --interactive  # 交互模式")
        sys.exit(1)

    faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
    chunks_json = "chunks.json"

    print("初始化RAG生成器（多轮对话版）...")
    generator = RAGGenerator(faiss_dir, chunks_json, memory_window=5)

    if sys.argv[1] == "--interactive":
        print("\n=== RAG 多轮对话模式 ===")
        print("输入问题，按回车生成答案，输入 'quit' 退出，'clear' 清空记忆\n")

        while True:
            query = input("\n问题: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query.lower() == 'clear':
                generator.clear_memory()
                generator.clear_long_term()
                print("记忆已清空")
                continue
            if not query:
                continue

            result = generator.qa(query, top_k=5, show_retrieved=True)

            if result.get('rewritten_query'):
                print(f"\n[Query改写] {result['rewritten_query']}")
            print(f"\n答案:\n{result['answer']}")

            if result.get('retrieved'):
                print(f"\n参考片段 (共{result['retrieved_count']}个):")
                for i, chunk in enumerate(result['retrieved'][:3], 1):
                    title = chunk.get('h1_title', '') or chunk.get('h2_title', '') or chunk.get('h3_title', '') or '正文'
                    source = chunk.get('source', 'unknown')
                    print(f"  [{i}][{source}] {title} (score: {chunk['score']:.2f})")

    else:
        query = " ".join(sys.argv[1:])
        result = generator.qa(query, top_k=5, show_retrieved=True)

        print(f"\n问题: {result['query']}")
        if result.get('rewritten_query'):
            print(f"改写后: {result['rewritten_query']}")
        print(f"\n答案:\n{result['answer']}")

        if result.get('retrieved'):
            print(f"\n参考片段 (共{result['retrieved_count']}个):")
            for i, chunk in enumerate(result['retrieved'][:3], 1):
                title = chunk.get('h1_title', '') or chunk.get('h2_title', '') or chunk.get('h3_title', '') or '正文'
                source = chunk.get('source', 'unknown')
                print(f"  [{i}][{source}] {title} (score: {chunk['score']:.2f})")


if __name__ == "__main__":
    main()
