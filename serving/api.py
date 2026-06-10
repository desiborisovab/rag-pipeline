import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.embed import load_index
from retrieval.retrieve import retrieve
from generation.generate import generate
from tracking.mlflow_logger import log_query_run

app = FastAPI(
    title = "RAG API",
    description="Production RAG pipeline -bg-large + cross encoder + TinyLlama",
    version = "1.0.0"
)

_index = None
_chunks = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index, _chunks
    print("Loading index...")
    _index, _chunks = load_index()
    print("Index loaded")

    yield

    _index = None
    _chunks = None
    print("API shutdown, index released")

class QueryRequest(BaseModel):
    question: str
    top_k: int = None

class ChunkResult(BaseModel):
    text: str
    source: str
    dense_score: float
    rerank_score: float = None

class QueryResponse(BaseModel):
    answers: str
    source: list[str]
    retrieved_chunks: List[ChunkResult]
    latency_ms: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "num_vectors": _index.ntotal if _index else 0
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    strat = time.time()

    hits = retrieve(request.question, _index, _chunks)
    result = generate(request.question, hits)

    latency_ms = (time.time() - strat) * 1000

    try:
        log_query_run(request.question, result, hits, latency_ms)
    except Exception as e:
        print(f"[warn] MLFlow logging failed: {e}")

    return QueryResponse(
        answers=result["answer"],
        source=result["sources"],
        retrieved_chunks=[ChunkResult(text=h["text"], source=h["source"], dense_score=h.get("dense_score", 0.0), rerank_score=h.get("rerank_score")) for h in hits],
        latency_ms=round(latency_ms, 2)
    )
    


