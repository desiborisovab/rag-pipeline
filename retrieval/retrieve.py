import numpy as np
import faiss
from utils.config import get_config
from pipeline.embed import embed
import os
from pathlib import Path

RERANKER_PATH = Path(os.environ.get("WEIGHTS_PATH", "weights")) / "reranker"

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    from sentence_transformers import CrossEncoder
    load_from = str(RERANKER_PATH) if RERANKER_PATH.exists() else get_config()["reranker"]["model_id"]
    _reranker = CrossEncoder(load_from)

    if not RERANKER_PATH.exists():
        RERANKER_PATH.mkdir(parents=True, exist_ok=True)
        _reranker.save(str(RERANKER_PATH))
    return _reranker

def dense_retrieve(query: str, index: faiss.Index, chunks: list[dict], k: int = None ) -> list[dict]:
    cfg = get_config()["retrieval"]
    k = k or cfg["candidate_k"]
    q_vec = embed([query], is_query= True)
    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "dense_score": float(score)})
    return results

def rerank(query: str, candidates: list[dict], top_n: int = None ) -> list[dict]:
    cfg = get_config()["retrieval"]
    top_n = top_n or cfg["top_k"]

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]

def retrieve(query: str, index: faiss.Index, chunks: list[dict] ) -> list[dict]:
    cfg = get_config()
    candidates = dense_retrieve(query, index, chunks, k = cfg["retrieval"]["candidate_k"])
    if cfg["reranker"]["enabled"] and len(candidates) > 0:
        return rerank(query, candidates, top_n = cfg["retrieval"]["top_k"])

    return candidates[: cfg["retrieval"]["top_k"]]
