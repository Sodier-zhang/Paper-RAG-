import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LLMgeneration import RAGGenerator

app = FastAPI(title="论文RAG问答API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

faiss_dir = r"D:\AI\paper_RAG\chunk_embedding\faiss_db"
chunks_json = "chunks.json"

print("初始化RAG生成器...")
generator = RAGGenerator(faiss_dir, chunks_json, memory_window=5)

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    clear_memory: bool = False

class ChatResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    retrieved_count: int
    retrieved: List[Dict[str, Any]]
    timestamp: str

class ClearRequest(BaseModel):
    clear_type: str = "all"

class ClearResponse(BaseModel):
    message: str
    timestamp: str

@app.get("/")
async def root():
    return {"message": "论文RAG问答API", "version": "1.0.0", "endpoints": ["/chat", "/clear", "/health"]}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")

    if request.clear_memory:
        generator.clear_memory()
        generator.clear_long_term()

    result = generator.qa(query=request.query, top_k=request.top_k, show_retrieved=True)

    return ChatResponse(
        query=result["query"],
        rewritten_query=result.get("rewritten_query"),
        answer=result["answer"],
        retrieved_count=result["retrieved_count"],
        retrieved=result.get("retrieved", []),
        timestamp=datetime.now().isoformat()
    )

@app.post("/clear", response_model=ClearResponse)
async def clear_memory(request: ClearRequest):
    if request.clear_type in ["all", "short"]:
        generator.clear_memory()
    if request.clear_type in ["all", "long"]:
        generator.clear_long_term()

    return ClearResponse(
        message=f"记忆已清空 ({request.clear_type})",
        timestamp=datetime.now().isoformat()
    )

@app.get("/history")
async def get_history():
    history = generator.memory.get_history_text()
    long_term_count = len(generator.long_term_memory)
    return {
        "short_term_history": history,
        "short_term_turns": len(generator.memory.history),
        "long_term_count": long_term_count
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
