# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that answers questions based on your own documents. Instead of fine-tuning a model, documents are chunked, embedded into vectors, and stored in a vector index. At query time, the most relevant chunks are retrieved and passed to a language model to generate an answer.

All models run locally from saved weights — no API keys required.

## What it does

You give it documents. You ask it questions. It finds the relevant parts of your documents and generates an answer from them.

## Tools and methods used

**Embedding** — BAAI/bge-large-en-v1.5. Converts text chunks into 1024-dimensional vectors using a pretrained transformer encoder. Documents and queries are embedded differently (asymmetric retrieval) — queries get a prefix to improve search quality.

**Vector search** — FAISS IndexFlatIP. Stores all chunk vectors and finds the closest matches to a query vector using cosine similarity. Runs entirely from a local file, no server needed.

**Reranking** — cross-encoder/ms-marco-MiniLM-L-6-v2. After FAISS retrieves the top 20 candidates, the reranker scores each (query, chunk) pair jointly to find the most relevant 3. More accurate than cosine similarity alone.

**Generation** — TinyLlama-1.1B-Chat. A small local language model that reads the retrieved chunks and generates an answer. Uses Apple MPS for faster inference on Mac.

**Chunking** — documents are split into 500-character overlapping windows before embedding. Overlap prevents answers from being cut off at chunk boundaries.

**Experiment tracking** — MLflow. Every index run and query is logged with parameters and metrics. Compatible with Databricks Community Edition.

**Evaluation** — RAGAS. Measures retrieval and generation quality using faithfulness, answer relevancy, context precision, and context recall.

**Serving** — FastAPI. Exposes a REST endpoint so the pipeline can be queried over HTTP.

**CI/CD** — GitHub Actions. Runs retrieval regression tests and rebuilds the index on every push to main.

## Setup


python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py


## Usage


python main.py index docs/sample.txt    # index your documents
python main.py query "your question"    # ask a question
python main.py chat   # interactive mode
python main.py serve  # start REST API on port 8000
python main.py eval   # run RAGAS evaluation


## Configuration

All parameters are in config.yml. Change models, chunk size, batch size, and retrieval settings there without touching any code.

## Note on Apple Silicon

SentenceTransformers and MPS cause a segfault on Apple Silicon when loading bge-large inside a process that has already imported other heavy libraries. The embedding step runs in an isolated subprocess (embed_worker.py) to work around this. The generator still uses MPS.