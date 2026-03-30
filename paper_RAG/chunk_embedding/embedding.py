import os
import json
import dashscope
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


class ChunkEmbedder:
    def __init__(self, chunked_json_dir: str, api_key: str = None):
        self.chunked_json_dir = Path(chunked_json_dir)
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = self.api_key
        self.index = None
        self.chunks_data: List[Dict[str, Any]] = []

    def load_chunks(self, json_file: Path) -> List[Dict[str, Any]]:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("chunks", [])

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        all_chunks = []
        for json_file in self.chunked_json_dir.glob("*_chunks.json"):
            chunks = self.load_chunks(json_file)
            for chunk in chunks:
                chunk["source_file"] = json_file.stem
            all_chunks.extend(chunks)
        return all_chunks

    def create_embeddings(self, texts: List[str], batch_size: int = 25) -> List[List[float]]:
        from dashscope import TextEmbedding

        saved_http = os.environ.pop("HTTP_PROXY", None)
        saved_https = os.environ.pop("HTTPS_PROXY", None)
        saved_no_proxy = os.environ.pop("NO_PROXY", None)

        try:
            all_embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="生成Embedding"):
                batch = texts[i:i + batch_size]
                response = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v2,
                    input=batch
                )
                if response.status_code == 200:
                    batch_embeddings = [item["embedding"] for item in response.output["embeddings"]]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"Embedding调用失败: {response.message}")
        finally:
            if saved_http:
                os.environ["HTTP_PROXY"] = saved_http
            if saved_https:
                os.environ["HTTPS_PROXY"] = saved_https
            if saved_no_proxy:
                os.environ["NO_PROXY"] = saved_no_proxy

        return all_embeddings

    def save_to_faiss(self, embeddings: List[List[float]], chunks: List[Dict[str, Any]], output_dir: str):
        import faiss
        import numpy as np

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        faiss.write_index(self.index, str(output_path / "faiss.index"))

        chunks_file = output_path / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return str(output_path / "faiss.index"), str(chunks_file)

    def process(self, output_dir: str) -> Dict[str, Any]:
        print("加载切块数据...")
        self.chunks_data = self.get_all_chunks()
        print(f"共 {len(self.chunks_data)} 个块")

        texts = [chunk["content"] for chunk in self.chunks_data]

        print("生成Embedding向量 (dashscope text-embedding-v2)...")
        embeddings = self.create_embeddings(texts)

        print("保存到FAISS...")
        index_path, chunks_path = self.save_to_faiss(embeddings, self.chunks_data, output_dir)

        return {
            "total_chunks": len(self.chunks_data),
            "embedding_dim": len(embeddings[0]) if embeddings else 0,
            "faiss_index": index_path,
            "chunks_json": chunks_path
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python embedding.py <chunked_json_dir>")
        print("示例: python embedding.py D:\\AI\\paper_RAG\\chunk_embedding\\chunked_json")
        sys.exit(1)

    chunked_json_dir = sys.argv[1]

    embedder = ChunkEmbedder(chunked_json_dir)
    result = embedder.process(r"D:\AI\paper_RAG\chunk_embedding\faiss_db")

    print(f"\n处理完成!")
    print(f"总块数: {result['total_chunks']}")
    print(f"Embedding维度: {result['embedding_dim']}")
    print(f"FAISS索引: {result['faiss_index']}")
    print(f"块数据: {result['chunks_json']}")
