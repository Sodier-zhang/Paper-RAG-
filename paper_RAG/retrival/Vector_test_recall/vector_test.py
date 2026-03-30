import os
import re
import json
import dashscope
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm


class VectorRetriever:
    def __init__(self, faiss_dir: str, chunks_json: str, api_key: str = None):
        self.faiss_dir = Path(faiss_dir)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = self.api_key

        self.index = faiss.read_index(str(self.faiss_dir / "faiss.index"))
        with open(self.faiss_dir / chunks_json, "r", encoding="utf-8") as f:
            self.chunks_data = json.load(f)

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
        from dashscope import TextEmbedding

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

    def extract_keywords(self, text: str) -> List[str]:
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "什么", "怎么", "如何", "为什么", "哪", "哪个", "哪些", "吗", "呢", "吧", "啊"}

        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()

        keywords = []
        for word in words:
            if len(word) >= 2 and word not in stopwords:
                if not word.isdigit():
                    keywords.append(word)

        return keywords

    def keyword_search(self, keywords: List[str], top_k: int = 50) -> List[Tuple[int, float]]:
        scores = []
        for idx, chunk in enumerate(self.chunks_data):
            content_lower = chunk.get("content", "").lower()
            score = 0
            for kw in keywords:
                if kw in content_lower:
                    score += 1
            if score > 0:
                scores.append((idx, score / len(keywords)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))
        return results

    def hybrid_search(self, query: str, top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        keywords = self.extract_keywords(query)

        kw_results = self.keyword_search(keywords, top_k=top_k * 3)
        sem_results = self.semantic_search(query, top_k=top_k * 2)

        kw_dict = {idx: score for idx, score in kw_results}
        sem_dict = {idx: 1 / (1 + dist) for idx, dist in sem_results}

        all_indices = set(kw_dict.keys()) | set(sem_dict.keys())

        combined_scores = {}
        for idx in all_indices:
            kw_score = kw_dict.get(idx, 0)
            sem_score = sem_dict.get(idx, 0)
            combined_scores[idx] = semantic_weight * sem_score + (1 - semantic_weight) * kw_score

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in sorted_results:
            chunk = self.chunks_data[idx]
            results.append({
                "index": idx,
                "score": score,
                "h1_title": chunk.get("h1_title", ""),
                "h2_title": chunk.get("h2_title", ""),
                "h3_title": chunk.get("h3_title", ""),
                "content": chunk.get("content", ""),
                "source_file": chunk.get("source_file", "")
            })

        return results


def load_questions(question_file: str) -> List[Dict[str, str]]:
    questions = []
    current_q = None

    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            q_match = re.match(r'^[\d]+\.?\s*[QqQ][：:\s]\s*(.+)$', line)
            if q_match:
                current_q = {"question": q_match.group(1), "answer": ""}
                continue

            a_match = re.match(r'^[Aa][：:\s]\s*(.+)$', line)
            if a_match and current_q:
                current_q["answer"] = a_match.group(1)
                questions.append(current_q)
                current_q = None

    return questions


def evaluate_recall(retrieved_chunks: List[Dict], ground_truth: str, keywords: List[str]) -> Dict[str, Any]:
    retrieved_text = " ".join([chunk["content"] for chunk in retrieved_chunks]).lower()

    gt_lower = ground_truth.lower()

    keyword_hits = 0
    for kw in keywords:
        if kw in retrieved_text or kw in gt_lower:
            keyword_hits += 0.5
        if kw in retrieved_text and kw in gt_lower:
            keyword_hits += 0.5

    recall = min(keyword_hits / len(keywords), 1.0) if keywords else 0.0

    return {
        "recall": recall,
        "keyword_hits": keyword_hits,
        "total_keywords": len(keywords)
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python vector_test.py <test_question.txt> [top_k]")
        print("示例: python vector_test.py ./test_question.txt 5")
        sys.exit(1)

    question_file = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
    chunks_json = "chunks.json"

    print("初始化检索器...")
    retriever = VectorRetriever(faiss_dir, chunks_json)

    print("加载测试问题...")
    questions = load_questions(question_file)
    print(f"共 {len(questions)} 个问题\n")

    results_all = []
    total_recall = 0.0

    for i, qa in enumerate(tqdm(questions, desc="检索中")):
        query = qa["question"]
        ground_truth = qa["answer"]

        retrieved_indices = retriever.semantic_search(query, top_k=top_k)
        retrieved = []
        for idx, dist in retrieved_indices:
            chunk = retriever.chunks_data[idx]
            content = chunk.get("content", "")

            if not chunk.get("is_parent", False) and chunk.get("parent_index") is not None:
                parent_idx = chunk.get("parent_index")
                if parent_idx is not None and parent_idx < len(retriever.chunks_data):
                    parent_chunk = retriever.chunks_data[parent_idx]
                    content = parent_chunk.get("content", "")
                elif chunk.get("parent_content"):
                    content = chunk.get("parent_content", "")

            retrieved.append({
                "index": idx,
                "score": dist,
                "h1_title": chunk.get("h1_title", ""),
                "h2_title": chunk.get("h2_title", ""),
                "h3_title": chunk.get("h3_title", ""),
                "content": content,
                "is_parent": chunk.get("is_parent", False)
            })

        keywords = retriever.extract_keywords(ground_truth)
        eval_result = evaluate_recall(retrieved, ground_truth, keywords)
        total_recall += eval_result["recall"]

        results_all.append({
            "q_id": i + 1,
            "question": query,
            "ground_truth": ground_truth,
            "retrieved": retrieved,
            "eval": eval_result
        })

    avg_recall = total_recall / len(questions) if questions else 0.0

    print("\n" + "=" * 80)
    print(f"测试完成! Top-{top_k} 召回率: {avg_recall:.2%}")
    print("=" * 80)

    output_file = Path(question_file).parent / f"recall_results_top{top_k}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "avg_recall": avg_recall,
            "top_k": top_k,
            "total_questions": len(questions),
            "results": results_all
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")
