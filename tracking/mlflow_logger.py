import mlflow
import json
from utils.config import get_config

def setup_mlflow():
    cfg = get_config()["mlflow"]
    uri = cfg.get("tracking_uri")

    if uri:
        mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(cfg["experiment_name"])
    print(f" MLflow experiment name: {cfg['experiment_name']}")

def log_index_run(chunks: list[dict], duration_sec: float):
    cfg = get_config()
    setup_mlflow()

    with mlflow.start_run(run_name="index"):
        mlflow.log_params({
            "embed_model" : cfg["embedding"]["model_id"],
            "chunk_strategy": cfg["chunking"]["strategy"],
            "chunk_size": cfg["chunking"]["chunk_size"],
            "chunk_overlap": cfg["chunking"]["chunk_overlap"],
            "reranker": cfg["reranker"]["model_id"]
        })

        mlflow.log_metrics({
            "num_chunks": len(chunks),
            "num_docs": len({c["doc_id"] for c in chunks}),
            "avg_chunk_chars": sum(len(c["text"]) for c in chunks) / len(chunks),
            "index_duration_s": duration_sec
        })

        stats_path = "/tmp/chunk_stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "total_chunks": len(chunks),
                "by_doc": {
                    doc: len([c for c in chunks if c["doc_id"] == doc])
                    for doc in {c["doc_id"] for c in chunks}
                }
            }, f, indent=2)

        mlflow.log_artifact(stats_path)
    print("MLflow index run logged")

def log_query_run(query: str, result: dict, retrieved_chunks: list[dict], latency_ms: float ):
    cfg = get_config()
    setup_mlflow()

    with mlflow.start_run(run_name="query"):
        mlflow.log_params({
            "embed_model": cfg["embedding"]["model_id"],
            "gen_model": cfg["generation"]["model_id"],
            "top_k": cfg["retrieval"]["top_k"],
            "candidate_k": cfg["retrieval"]["candidate_k"],
            "reranker_on": cfg["reranker"]["enabled"],
            "temperature": cfg["generation"]["temperature"],
        })
        mlflow.log_metrics({
            "latency_ms": latency_ms,
            "num_sources": len(result["sources"]),
            "top_dense_score": retrieved_chunks[0].get("dense_score", 0),
            "top_rerank_score": retrieved_chunks[0].get("rerank_score", 0),
            "answer_len_chars": len(result["answer"]),
        })
        mlflow.log_text(query, "query.txt")
        mlflow.log_text(result["answer"], "answer.txt")
        mlflow.log_text(result["prompt"], "prompt.txt")

    print(f"MLflow query run logged. latency_ms: {latency_ms}")

def log_eval_run(metrics: dict):
    cfg = get_config()
    setup_mlflow()

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_params({
            "embed_model": cfg["embedding"]["model_id"],
            "gen_model": cfg["generation"]["model_id"],
            "chunk_strategy": cfg["chunking"]["strategy"],
            "chunk_size": cfg["chunking"]["chunk_size"],
            "reranker_on": cfg["reranker"]["enabled"],
        })
        mlflow.log_metrics(metrics)

    print(f"MLflow eval run logged: {metrics}")



