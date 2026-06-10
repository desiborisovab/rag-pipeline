import pytest
from pipeline.ingest import load_document
from pipeline.chunk import chunk_documents
from pipeline.embed import build_index
from retrieval.retrieve import retrieve, dense_retrieve

SAMPLE_DOC = "docs/sample.txt"

@pytest.fixture(scope="module")
def index_and_chunks():
    docs = load_document([SAMPLE_DOC])
    chunks = chunk_documents(docs)
    index, chunks = build_index(chunks)
    return index, chunks

def test_index_not_empty(index_and_chunks):
    index, chunks = index_and_chunks
    assert index.ntotal > 0
    assert len(chunks) > 0

def test_dense_retrieve_returns_results(index_and_chunks):
    index, chunks = index_and_chunks
    results = dense_retrieve("living systems", index, chunks, k=5)
    assert len(results) > 0
    assert all("text" in r for r in results)
    assert all("dense_score" in r for r in results)

def test_dense_retrieve_scores_between_0_and_1(index_and_chunks):
    index, chunks = index_and_chunks
    results = dense_retrieve("hydrogen bonds", index, chunks, k=5)
    for r in results:
        assert 0.0 <= r["dense_score"] <= 1.01

def test_retrieve_top_k_respected(index_and_chunks):
    index, chunks = index_and_chunks
    results = dense_retrieve("property of carboxyl", index, chunks, k=3)
    assert len(results) <= 3

def test_known_query_hits_correct_chunk(index_and_chunks):
    index, chunks = index_and_chunks
    results = dense_retrieve("What are enantiomers?", index, chunks, k=5)
    top_texts = " ".join(r["text"].lower() for r in results)
    assert "30 days" in top_texts, (
        "Expected enantiomers chunk in top-5 — retrieval may have regressed"
    )

def test_irrelevant_query_low_score(index_and_chunks):
    index, chunks = index_and_chunks
    results = dense_retrieve("corporate refund policy", index, chunks, k=1)
    assert results[0]["dense_score"] < 0.7, (
        "Irrelevant query returned suspiciously high similarity score"
    )