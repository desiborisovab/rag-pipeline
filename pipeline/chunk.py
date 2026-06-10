import re
from utils.config import get_config

def fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    """sliding window over characters"""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + size].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 20]

def sentence_chunks(text: str, size: int, overlap_sentence: int = 1) -> list[str]:
    """sliding window over sentences"""
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except ImportError:
        print(f"Could not import nltk. Try installing nltk instead.")
        cfg = get_config()["chunking"]
        return fixed_chunks(text, size, cfg["chunk_overlap"])

    chunks, current, current_len = [], [], 0
    for sent in sentences:
        current.append(sent)
        current_len += len(sent)
        if current_len >= size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentence:]
            current_len = sum(len(s) for s in current)
    if current:
        chunks.append(" ".join(current))
        return [c for c in chunks if len(c) > 20]

def chunk_documents(docs: list[dict]) -> list[dict]:
    """takes documents records from ingest.py and returns chunk records"""
    cfg = get_config()["chunking"]
    strategy = cfg["strategy"]
    size = cfg["chunk_size"]
    overlap = cfg["chunk_overlap"]

    all_chunks = []
    for doc in docs:
        if strategy == "sentences":
            texts = sentence_chunks(doc["text"], size)
        else:
            texts = fixed_chunks(doc["text"], size, overlap)

        for i, text in enumerate(texts):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}__{i:04d}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": text,
                "char_start": i * (size-overlap),
                "strategy": strategy,
            })
    print(f"Chunk records: {len(all_chunks)} chunks from {len(docs)} documents (strategy={strategy})")
    return all_chunks



