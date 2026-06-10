import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import time

def cmd_index(targets: list[str]):
    from pipeline.ingest import load_from_directory, load_document
    from pipeline.chunk import chunk_documents
    from pipeline.embed import build_index
    from tracking.mlflow_logger import log_index_run

    if len(targets) == 1 and __import__("pathlib").Path(targets[0]).is_dir():
        docs = load_from_directory(targets[0])
    else:
        docs = load_document(targets)

    chunks = chunk_documents(docs)
    t0 = time.time()
    build_index(chunks)
    duration = time.time() - t0
    log_index_run(chunks, duration)

def cmd_query(question: str):
    from pipeline.embed import load_index
    from retrieval.retrieve import retrieve
    from generation.generate import generate
    from tracking.mlflow_logger import log_query_run

    index, chunks = load_index()
    t0 = time.time()

    hits = retrieve(question, index, chunks)
    result = generate(question, hits)
    latency_ms = (time.time() - t0) * 1000
    log_query_run(question, result, hits, latency_ms)

    print("Retrieved chunks")
    for i, h in enumerate(hits, 1):
        ds = h.get('dense_score', 0)
        rs = h.get('rerank_score', 0)
        print(f"[{i}] dense={ds:.3f} rerank={rs:.3f} src={h['source']}")
        print(f"{h['text'][:100].strip()}")

    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Latency: {latency_ms:.0f} ms")

def cmd_eval():
    from pipeline.embed import load_index
    from evaluation.evaluate import run_evaluation, load_eval_dataset
    index, chunks = load_index()
    dataset = load_eval_dataset()
    run_evaluation(index, chunks, dataset, log_to_mlflow=True)

def cmd_chat():
    from pipeline.embed import load_index
    from retrieval.retrieve import retrieve
    from generation.generate import generate

    print("Loading index and models")
    index, chunks = load_index()
    print("Index loaded")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question or question.lower() in {"exit", "quit", "q"}:
            break
        hits = retrieve(question, index, chunks)
        result = generate(question, hits)
        print(f"\nAssistant: {result['answer']}\n")

def cmd_serve():
    import uvicorn
    from utils.config import get_config
    cfg = get_config()["serving"]
    uvicorn.run("serving.api:app", host=cfg["host"], port=cfg["port"], reload=False)

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd == "index":
        cmd_index(args or ["docs/"])
    elif cmd == "query":
        cmd_query(" ".join(args))
    elif cmd == "eval":
        cmd_eval()
    elif cmd == "chat":
        cmd_chat()
    elif cmd == "serve":
        cmd_serve()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()

