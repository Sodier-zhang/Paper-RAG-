import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LLMgeneration import RAGGenerator


class RAGEvaluator:
    def __init__(self, faiss_dir: str, chunks_json: str, api_key: str = None):
        self.generator = RAGGenerator(faiss_dir, chunks_json, api_key)

    def evaluate_faithfulness(self, question: str, answer: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        from dashscope import Generation

        context = "\n\n".join([
            f"[{i+1}] {chunk.get('content', '')}" for i, chunk in enumerate(retrieved_chunks[:3])
        ])

        prompt = f"""你是一个RAG评估专家。请评估以下答案是否基于提供的检索内容（忠实度/Groundedness）。

任务：判断答案中的事实性陈述是否可以从检索内容中推断出来。

检索内容：
{context}

问题：{question}

答案：{answer}

评估标准：
- 完全忠实：答案的所有事实都可以从检索内容中找到或推断出来
- 部分忠实：答案的大部分事实来自检索内容，但有少量额外信息
- 不忠实：答案包含大量检索内容中没有的事实

只输出以下格式的JSON，不要其他内容：
{{"score": 0-10, "reason": "简短原因"}}

JSON："""

        self.generator._remove_proxy()
        try:
            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            if response.status_code == 200:
                result_text = response.output["text"].strip()
                return self._parse_json_result(result_text)
            else:
                return {"score": 0, "reason": "API调用失败"}
        finally:
            self.generator._restore_proxy()

    def evaluate_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        from dashscope import Generation

        prompt = f"""你是一个RAG评估专家。请评估以下答案与问题的相关性（Answer Relevance）。

任务：判断答案是否针对问题的核心内容进行回答。

问题：{question}

答案：{answer}

评估标准：
- 10分：答案完全针对问题，核心内容完全匹配
- 7-9分：答案大部分针对问题，有少量偏差
- 4-6分：答案部分针对问题，有明显偏差
- 1-3分：答案基本不相关
- 0分：答案完全跑题

只输出以下格式的JSON，不要其他内容：
{{"score": 0-10, "reason": "简短原因"}}

JSON："""

        self.generator._remove_proxy()
        try:
            response = Generation.call(
                model=Generation.Models.qwen_turbo,
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            if response.status_code == 200:
                result_text = response.output["text"].strip()
                return self._parse_json_result(result_text)
            else:
                return {"score": 0, "reason": "API调用失败"}
        finally:
            self.generator._restore_proxy()

    def _parse_json_result(self, text: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        score_match = re.search(r'"score"\s*:\s*(\d+)', text)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)

        score = int(score_match.group(1)) if score_match else 0
        reason = reason_match.group(1) if reason_match else "解析失败"

        return {"score": score, "reason": reason}

    def evaluate_single(self, question: str, ground_truth: str = None) -> Dict[str, Any]:
        result = self.generator.qa(question, top_k=5, show_retrieved=True)

        faithfulness = self.evaluate_faithfulness(
            question,
            result["answer"],
            result.get("retrieved", [])
        )

        relevance = self.evaluate_answer_relevance(
            question,
            result["answer"]
        )

        return {
            "question": question,
            "ground_truth": ground_truth,
            "answer": result["answer"],
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "retrieved": result.get("retrieved", [])
        }

    def evaluate_dataset(self, questions: List[Dict[str, str]], sample_size: int = None) -> Dict[str, Any]:
        if sample_size:
            questions = questions[:sample_size]

        results = []
        for qa in tqdm(questions, desc="评估中"):
            try:
                result = self.evaluate_single(
                    qa.get("question", ""),
                    qa.get("answer", "")
                )
                results.append(result)
            except Exception as e:
                print(f"评估失败: {qa.get('question', '')[:50]}... - {e}")
                continue

        avg_faithfulness = sum(r["faithfulness"]["score"] for r in results) / len(results) if results else 0
        avg_relevance = sum(r["answer_relevance"]["score"] for r in results) / len(results) if results else 0

        return {
            "total_questions": len(questions),
            "evaluated": len(results),
            "avg_faithfulness": avg_faithfulness,
            "avg_answer_relevance": avg_relevance,
            "avg_overall": (avg_faithfulness + avg_relevance) / 2,
            "results": results
        }


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG生成质量评估")
    parser.add_argument("--question_file", type=str, required=True, help="问题文件路径")
    parser.add_argument("--sample", type=int, default=None, help="采样数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
    chunks_json = "chunks.json"

    print("初始化评估器...")
    evaluator = RAGEvaluator(faiss_dir, chunks_json)

    print(f"加载问题: {args.question_file}")
    questions = load_questions(args.question_file)
    print(f"共 {len(questions)} 个问题")

    if args.sample:
        print(f"采样评估: {args.sample} 个问题")
        questions = questions[:args.sample]

    print("\n开始评估...")
    results = evaluator.evaluate_dataset(questions)

    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)
    print(f"评估问题数: {results['evaluated']}/{results['total_questions']}")
    print(f"忠实度 (Faithfulness): {results['avg_faithfulness']:.2f}/10")
    print(f"答案相关性 (Answer Relevance): {results['avg_answer_relevance']:.2f}/10")
    print(f"综合得分: {results['avg_overall']:.2f}/10")
    print("=" * 60)

    if args.output:
        output_file = args.output
    else:
        question_path = Path(args.question_file)
        output_file = question_path.parent / "evaluation_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
