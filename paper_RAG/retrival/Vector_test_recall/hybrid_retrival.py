import os
import re
import json
import dashscope
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents

        try:
            self.en_stopwords = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            self.en_stopwords = set(stopwords.words('english'))

        self.stemmer = PorterStemmer()

        self.special_terms = {
            'vec', 'v2i', 'v2v', 'v2x', 'rsu', 'dnn', 'cnn', 'rnn', 'lstm',
            'gpu', 'cpu', 'cpu', 'ram', 'rom', 'api', 'http', 'https', 'ftp',
            'iot', 'mec', '边缘计算', '车载', '车辆', 'RSU', 'V2I', 'DNN', 'VEC',
            'llm', 'rag', 'nlp', 'cv', 'ai', 'ml', 'dl'
        }
        self.protected_pattern = re.compile(r'\b(' + '|'.join(sorted(self.special_terms, key=len, reverse=True)) + r')\b', re.IGNORECASE)

        self.tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> List[str]:
        text_lower = text.lower()

        protected_matches = {}
        def protect_special(match):
            placeholder = f"__PROTECTED_{len(protected_matches)}__"
            protected_matches[placeholder] = match.group(0).lower()
            return placeholder

        text_protected = self.protected_pattern.sub(protect_special, text_lower)

        text_protected = re.sub(r'[^\w\s]', ' ', text_protected)
        words = text_protected.split()

        tokens = []
        for w in words:
            if w.startswith('__PROTECTED_'):
                tokens.append(protected_matches[w])
                continue
            if len(w) < 2:
                continue
            if w in self.en_stopwords:
                continue
            stemmed = self.stemmer.stem(w)
            tokens.append(stemmed)

        return tokens

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        result = [(i, float(score)) for i, score in enumerate(scores)]
        result.sort(key=lambda x: x[1], reverse=True)
        return result


class HybridRetriever:
    def __init__(self, faiss_dir: str, chunks_json: str, api_key: str = None):
        self.faiss_dir = Path(faiss_dir)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = self.api_key

        self.index = faiss.read_index(str(self.faiss_dir / "faiss.index"))
        with open(self.faiss_dir / chunks_json, "r", encoding="utf-8") as f:
            self.chunks_data = json.load(f)

        self.bm25 = BM25([chunk.get("content", "") for chunk in self.chunks_data])

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

    def rewrite_query(self, query: str, use_hyde: bool = True) -> str:
        from dashscope import Generation

        self._remove_proxy()
        try:
            if use_hyde:
                prompt = f"""You are an academic paper assistant. Generate a hypothetical answer in ENGLISH to the given question as if you were writing a section of an academic paper.

Requirements:
1. Output MUST be in English only
2. Be 2-3 sentences long
3. Use formal academic language
4. Include relevant technical terms and abbreviations (e.g., V2I, RSU, DNN, VEC)
5. Mention key concepts that would appear in the paper

Generate ONLY the English hypothetical answer, nothing else:

Question: {query}
Hypothetical Answer (English):"""
            else:
                prompt = f"""You are an academic paper retrieval expert. For the given query, generate comprehensive English search terms.

Tasks:
1. Translate the query to English
2. Identify core academic concepts and their: abbreviations, full forms, synonyms, related terms
3. Extract keywords for retrieval

Return format (only search terms, nothing else):
English Query: ...
Abbreviations: ...
Keywords: ...
Full Terms: ...

Query: {query}"""

            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt
            )

            if response.status_code == 200:
                result = response.output["text"].strip()
                return result
            else:
                return query
        finally:
            self._restore_proxy()

    def generate_multi_query(self, query: str) -> Dict[str, str]:
        from dashscope import Generation

        self._remove_proxy()
        try:
            prompt = f"""You are an academic paper retrieval expert. For the given query, generate multiple search representations.

Generate the following:
1. English translation of the query
2. Academic keywords (abbreviations, full forms)
3. A hypothetical passage that might appear in an academic paper answering this question

Output format (only the content, nothing else):
TRANSLATION: <english translation>
KEYWORDS: <comma-separated keywords>
HYPOTHETICAL: <2-3 sentence hypothetical passage in English>

Query: {query}"""

            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt
            )

            if response.status_code == 200:
                result = response.output["text"].strip()
                lines = result.split('\n')

                translation = query
                keywords = ""
                hypothetical = ""

                for line in lines:
                    if line.startswith('TRANSLATION:'):
                        translation = line.split(':', 1)[1].strip()
                    elif line.startswith('KEYWORDS:'):
                        keywords = line.split(':', 1)[1].strip()
                    elif line.startswith('HYPOTHETICAL:'):
                        hypothetical = line.split(':', 1)[1].strip()

                return {
                    "original": query,
                    "translation": translation,
                    "keywords": keywords,
                    "hypothetical": hypothetical
                }
            else:
                return {"original": query, "translation": query, "keywords": "", "hypothetical": query}
        finally:
            self._restore_proxy()

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
        try:
            en_stopwords = set(stopwords.words('english'))
        except:
            en_stopwords = set()

        text_lower = text.lower()

        special_terms = {
            'vec', 'v2i', 'v2v', 'v2x', 'rsu', 'dnn', 'cnn', 'rnn', 'lstm',
            'gpu', 'cpu', 'ram', 'rom', 'api', 'http', 'https', 'ftp',
            'iot', 'mec', 'llm', 'rag', 'nlp', 'cv', 'ai', 'ml', 'dl'
        }
        protected_pattern = re.compile(r'\b(' + '|'.join(sorted(special_terms, key=len, reverse=True)) + r')\b', re.IGNORECASE)

        protected_matches = {}
        def protect_special(match):
            placeholder = f"__PROTECTED_{len(protected_matches)}__"
            protected_matches[placeholder] = match.group(0).lower()
            return placeholder

        text_protected = protected_pattern.sub(protect_special, text_lower)

        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_chars = chinese_pattern.findall(text_protected)
        chinese_words = [char for chars in chinese_chars for char in chars]

        text_protected = chinese_pattern.sub(' ', text_protected)
        text_protected = re.sub(r'[^\w\s]', ' ', text_protected)
        words = text_protected.split()

        stemmer = PorterStemmer()
        keywords = []

        for word in words:
            if word.startswith('__PROTECTED_'):
                keywords.append(protected_matches[word])
                continue
            if len(word) >= 2 and word not in en_stopwords:
                if not word.isdigit():
                    stemmed = stemmer.stem(word)
                    keywords.append(stemmed)

        keywords.extend(chinese_words)

        return list(set(keywords))

    def keyword_search(self, keywords: List[str], top_k: int = 50) -> List[Tuple[int, float]]:
        query = " ".join(keywords)
        return self.bm25.score(query)[:top_k]

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_embedding = self.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))
        return results

    def get_parent_chunks(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        h1 = chunk.get("h1_title", "")
        h2 = chunk.get("h2_title", "")
        h3 = chunk.get("h3_title", "")

        same_section_chunks = []
        if h3:
            same_section_chunks = [c for c in self.chunks_data if c.get("h1_title") == h1 and c.get("h2_title") == h2 and c.get("h3_title") == h3]
        elif h2:
            same_section_chunks = [c for c in self.chunks_data if c.get("h1_title") == h1 and c.get("h2_title") == h2]
        elif h1:
            same_section_chunks = [c for c in self.chunks_data if c.get("h1_title") == h1]

        parent_chunks = [c for c in same_section_chunks if c.get("is_parent", False)]
        if parent_chunks:
            return parent_chunks

        return same_section_chunks if same_section_chunks else [chunk]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        from dashscope import TextReRank

        if not candidates:
            return []

        self._remove_proxy()
        try:
            documents = [c.get("parent_content", "") or c.get("sub_chunk", "") for c in candidates]

            response = TextReRank.call(
                model=TextReRank.Models.gte_rerank,
                query=query,
                documents=documents
            )

            if response.status_code == 200:
                reranked = []
                results = response.output["results"]
                for rerank_result in results:
                    score = rerank_result.relevance_score
                    doc_idx = rerank_result.index
                    reranked.append({
                        **candidates[doc_idx],
                        "rerank_score": float(score)
                    })
                return reranked[:top_k]
            else:
                return candidates[:top_k]
        finally:
            self._restore_proxy()

    def _should_rewrite(self, query: str) -> bool:
        return False

    def lightweight_query_expansion(self, query: str) -> Dict[str, str]:
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_terms = chinese_pattern.findall(query)

        en_query = chinese_pattern.sub('', query).strip()

        words = en_query.lower().split()
        expanded_keywords = set()

        for word in words:
            if word not in self.bm25.en_stopwords if hasattr(self, 'bm25') and self.bm25.en_stopwords else True:
                expanded_keywords.add(word)
                stemmed = self.bm25.stemmer.stem(word) if hasattr(self, 'bm25') else word
                expanded_keywords.add(stemmed)

        synonyms = {
            'nn': 'neural network',
            'dl': 'deep learning',
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'cv': 'computer vision',
            'nlp': 'natural language processing',
            'lama': 'LAMA-GAP',
            'ba': 'BA-MA BA-GR baseline',
            'gpu': 'GPU computing',
            'rsu': 'Roadside Unit',
            'v2i': 'vehicle to infrastructure',
            'v2v': 'vehicle to vehicle',
            'vec': 'vehicular edge computing',
            'dnn': 'deep neural network',
            'cnn': 'convolutional neural network',
            'rnn': 'recurrent neural network',
            'lstm': 'long short-term memory',
            'mec': 'multi-access edge computing'
        }

        for word in list(expanded_keywords):
            if word in synonyms:
                expanded_keywords.add(synonyms[word])
            for syn_key, syn_val in synonyms.items():
                if word == syn_key:
                    expanded_keywords.add(syn_val.split()[0])

        return {
            "original": query,
            "translation": en_query,
            "keywords": ", ".join(expanded_keywords),
            "hypothetical": en_query
        }

    def hybrid_search(self, query: str, top_k: int = 5, use_rewrite: bool = True, use_rerank: bool = True, semantic_weight: float = 0.5) -> List[Dict[str, Any]]:
        multi_query = self.lightweight_query_expansion(query)
        print(f"    原文: {query}")
        print(f"    关键词: {multi_query['keywords']}")

        keywords = self.extract_keywords(multi_query['keywords'])
        kw_results = self.keyword_search(keywords, top_k=50)
        vec_results = self.vector_search(query, top_k=50)

        seen_indices = set()
        candidates = []
        for idx, score in kw_results:
            if idx not in seen_indices:
                seen_indices.add(idx)
                chunk = self.chunks_data[idx]
                h1 = chunk.get("h1_title", "")
                h2 = chunk.get("h2_title", "")
                parent_chunks = self.get_parent_chunks(chunk)
                sub_chunk_content = chunk.get("content", "")
                parent_contents = []
                for c in parent_chunks:
                    content = c.get("content", "").strip()
                    if content:
                        parent_contents.append(content)
                parent_content = " ".join(parent_contents)

                candidates.append({
                    "index": idx,
                    "kw_score": score,
                    "vec_score": 0.0,
                    "h1_title": h1,
                    "h2_title": h2,
                    "sub_chunk": sub_chunk_content,
                    "parent_content": parent_content,
                    "source_file": chunk.get("source_file", "")
                })

        for idx, score in vec_results:
            if idx not in seen_indices:
                seen_indices.add(idx)
                chunk = self.chunks_data[idx]
                h1 = chunk.get("h1_title", "")
                h2 = chunk.get("h2_title", "")
                parent_chunks = self.get_parent_chunks(chunk)
                sub_chunk_content = chunk.get("content", "")
                parent_contents = []
                for c in parent_chunks:
                    content = c.get("content", "").strip()
                    if content:
                        parent_contents.append(content)
                parent_content = " ".join(parent_contents)

                candidates.append({
                    "index": idx,
                    "kw_score": 0.0,
                    "vec_score": score,
                    "h1_title": h1,
                    "h2_title": h2,
                    "sub_chunk": sub_chunk_content,
                    "parent_content": parent_content,
                    "source_file": chunk.get("source_file", "")
                })
            else:
                for c in candidates:
                    if c["index"] == idx:
                        c["vec_score"] = max(c["vec_score"], score)
                        break

        print(f"    BM25候选: {len(kw_results)}, 向量候选: {len(vec_results)}")

        rrf_k = 60
        rrf_scores = {}

        for i, (idx, kw_score) in enumerate(kw_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + i)

        for i, (idx, vec_score) in enumerate(vec_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + i)

        all_indices = list(rrf_scores.keys())
        all_indices.sort(key=lambda x: rrf_scores[x], reverse=True)

        final_results = []
        for idx in all_indices[:top_k]:
            chunk = self.chunks_data[idx]
            parent_chunks = self.get_parent_chunks(chunk)
            parent_contents = [c.get("content", "").strip() for c in parent_chunks if c.get("content", "").strip()]
            parent_content = " ".join(parent_contents)

            final_results.append({
                "index": idx,
                "rrf_score": rrf_scores[idx],
                "h1_title": chunk.get("h1_title", ""),
                "h2_title": chunk.get("h2_title", ""),
                "sub_chunk": chunk.get("content", ""),
                "parent_content": parent_content,
                "source_file": chunk.get("source_file", "")
            })

        print(f"    RRF融合后候选: {len(final_results)}")

        return final_results


def load_questions(question_file: str) -> List[Dict[str, str]]:
    questions = []
    current_q = None

    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            q_match = re.match(r'^[\d]+\.?\s*[Qq][：:\s]\s*(.+)$', line)
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
    gt_lower = ground_truth.lower()
    gt_keywords = re.findall(r'\w+', gt_lower)
    gt_keyword_set = set(gt_keywords)

    rr = 0.0

    for i, chunk in enumerate(retrieved_chunks):
        chunk_text = (chunk.get("sub_chunk", "") or chunk.get("parent_content", "")).lower()
        chunk_text += " " + chunk.get("h1_title", "").lower()
        chunk_text += " " + chunk.get("h2_title", "").lower()

        chunk_keywords = set(re.findall(r'\w+', chunk_text))
        overlap = chunk_keywords & gt_keyword_set
        keyword_coverage = len(overlap) / len(gt_keyword_set) if gt_keyword_set else 0

        if keyword_coverage >= 0.3:
            rr = 1.0 / (i + 1)
            break

    return {
        "reciprocal_rank": rr,
        "recall": rr
    }


def evaluate_recall_summary(results: List[Dict[str, Any]], top_k: int) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {"mrr": 0.0, "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0}

    mrr = sum(r.get("eval", {}).get("reciprocal_rank", 0) for r in results) / total
    hit_at_1 = sum(1 for r in results if r.get("eval", {}).get("reciprocal_rank", 0) >= 1.0)
    hit_at_5 = sum(1 for r in results if r.get("eval", {}).get("reciprocal_rank", 0) >= 0.2)
    hit_at_10 = sum(1 for r in results if r.get("eval", {}).get("reciprocal_rank", 0) >= 0.1)

    return {
        "mrr": mrr,
        "hit_at_1": hit_at_1 / total,
        "hit_at_5": hit_at_5 / total,
        "hit_at_10": hit_at_10 / total
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python hybrid_retrival.py <test_question.txt> [top_k] [use_rewrite] [use_rerank] [semantic_weight]")
        print("示例: python hybrid_retrival.py ./test_question.txt 5 1 1 0.5")
        print("参数说明:")
        print("  top_k: 最终返回数量")
        print("  use_rewrite: 1=启用Query改写, 0=禁用")
        print("  use_rerank: 1=启用GTE重排, 0=禁用")
        print("  semantic_weight: 0.0-1.0, 语义权重(BM25权重=1-semantic_weight)")
        sys.exit(1)

    question_file = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    use_rewrite = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    use_rerank = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    semantic_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

    faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
    chunks_json = "chunks.json"

    print("初始化混合检索器...")
    retriever = HybridRetriever(faiss_dir, chunks_json)

    print("加载测试问题...")
    questions = load_questions(question_file)
    print(f"共 {len(questions)} 个问题\n")

    results_all = []
    total_recall = 0.0

    for i, qa in enumerate(tqdm(questions, desc="检索中")):
        query = qa["question"]
        ground_truth = qa["answer"]

        retrieved = retriever.hybrid_search(query, top_k=top_k, use_rewrite=use_rewrite, use_rerank=use_rerank, semantic_weight=semantic_weight)

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

    summary = evaluate_recall_summary(results_all, top_k)

    print("\n" + "=" * 80)
    print(f"混合检索测试完成! Top-{top_k}")
    print(f"MRR: {summary['mrr']:.4f}")
    print(f"Hit@1: {summary['hit_at_1']:.2%}")
    print(f"Hit@5: {summary['hit_at_5']:.2%}")
    print(f"Hit@10: {summary['hit_at_10']:.2%}")
    print("=" * 80)

    output_file = Path(question_file).parent / f"hybrid_recall_results_top{top_k}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "avg_recall": avg_recall,
            "top_k": top_k,
            "total_questions": len(questions),
            "results": results_all
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")
