import json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
faithfulness,
answer_relevancy,
context_precision,
context_recall,
)
from utils.config import get_config
from tracking.mlflow_logger import log_eval_run

def load_eval_dataset(path: str = None) -> list[dict]:
    path = path or get_config()["evaluation"]["dataset_path"]
    with open(path) as f:
        return json.load(f)

def run_evaluation(index, chunks: list[dict], eval_dataset: list[dict] = None, log_to_mlflow: bool = True) -> dict:
    from retrieval.retrieve import retrieve
    from generation.generate import generate

    eval_dataset = eval_dataset or load_eval_dataset()
    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running pipeline over {len(eval_dataset)} eval questions")

    for item in eval_dataset:
        query = item["question"]
        hits = retrieve(query, index, chunks)
        result = generate(query, hits)

        questions.append(query)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in hits])
        ground_truths.append(item["ground_truth"])

        ragas_data = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "context": contexts,
            "ground_truth": ground_truths,
        })
    print("Scoring with RAGAS")

    result = evaluate(ragas_data, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    metrics = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "context_recall": round(result["context_recall"], 4)
    }

    print("RAGAS results:")
    for k,v in metrics.items():
        print(f"{k:<22} {v:.4f}")

    if log_to_mlflow:
        log_eval_run(metrics)

    return metrics
