"""Embeds chunks with bge-large-en-v1.5 and stores them in local FAISS index"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils.config import get_config

_embedder = None

WEIGHTS_BASE = Path(os.environ.get("WEIGHTS_PATH", "weights"))
EMBED_WEIGHTS_PATH = WEIGHTS_BASE / "bge-large"

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder

    cfg = get_config()["embedding"]

    if EMBED_WEIGHTS_PATH.exists():
        print(f"Loading embedding weights from {EMBED_WEIGHTS_PATH}")
        _embedder = SentenceTransformer(str(EMBED_WEIGHTS_PATH), device="cpu")
    else:
        print(f"downloading {cfg['model_id']} (around 1.3 GB, first time only)")
        _embedder = SentenceTransformer(cfg["model_id"], device="cpu")
        print(f"Saving weights to {EMBED_WEIGHTS_PATH}")
        EMBED_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
        _embedder.save(str(EMBED_WEIGHTS_PATH))

    print(f"Embedder ready on device: cpu")
    return _embedder

def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    cfg = get_config()["embedding"]
    if is_query:
        texts = [cfg["query_prefix"] + t for t in texts]

    embedder = get_embedder()
    all_vecs = []
    batch_size = cfg["batch_size"]
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embedder.encode(
          batch,
          normalize_embeddings = True,
          batch_size = cfg["batch_size"],
          show_progress_bar = False
        )
        all_vecs.append(vectors)
        if len(texts) > 50:
           print(f"embedded {min(i + batch_size, len(texts))} / {len(texts)} chunks", end = "\r")

    if len(texts) > 50:
        print()

    return np.vstack(all_vecs).astype("float32")

import faiss
def build_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    import faiss
    cfg_vs = get_config()["vector_store"]
    cfg_emb = get_config()["embedding"]

    print(f" Embedding {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    vectors = embed(texts, is_query = False)
    dim = cfg_emb["dimension"]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, cfg_vs["index_path"])
    with open(cfg_vs["store_path"], "wb") as f:
        pickle.dump(chunks, f)

    print(f" FAISS index saved: {index.ntotal} vectors, dim={dim}")
    return index, chunks

def load_index() -> tuple[faiss.Index, list[dict]]:
    cfg = get_config()["vector_store"]
    index = faiss.read_index(str(cfg["index_path"]))
    with open(cfg["store_path"], "rb") as f:
        chunks = pickle.load(f)
    print(f"Index loaded {index.ntotal} vectors")
    return index, chunks

