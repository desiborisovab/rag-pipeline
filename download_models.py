"""
download_models.py

Downloads bge-large-en-v1.5 and TinyLlama-1.1B-Chat weights
into the ./weights/ directory (or WEIGHTS_PATH env var).

Run once before anything else:
  python download_models.py

After this, the pipeline never hits the internet for model weights.
"""

import os
import sys
from pathlib import Path

#Where to save
WEIGHTS_BASE = Path(os.environ.get("WEIGHTS_PATH", "weights"))
EMBED_PATH   = WEIGHTS_BASE / "bge-large"
GEN_PATH     = WEIGHTS_BASE / "tinyllama"

def download_embedder():
    from sentence_transformers import SentenceTransformer

    if EMBED_PATH.exists():
        print(f"[skip] Embedder already at {EMBED_PATH}")
        return

    print(f"Downloading BAAI/bge-large-en-v1.5 (~1.3 GB) ...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    EMBED_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(EMBED_PATH))
    print(f"Embedder saved to {EMBED_PATH}")


def download_generator():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if GEN_PATH.exists():
        print(f"[skip] Generator already at {GEN_PATH}")
        return

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Downloading {model_id} (~2 GB) ...")

    # Detect device
    if torch.cuda.is_available():    dtype = torch.float16
    elif torch.backends.mps.is_available(): dtype = torch.float16
    else:                            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    GEN_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(GEN_PATH))
    tokenizer.save_pretrained(str(GEN_PATH))
    print(f"Generator saved to {GEN_PATH}")


def download_reranker():
    from sentence_transformers import CrossEncoder
    from pathlib import Path

    reranker_path = WEIGHTS_BASE / "reranker"
    if reranker_path.exists():
        print(f"[skip] Reranker already at {reranker_path}")
        return

    print(f"Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 (90 Mb) ...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_path.mkdir(parents=True, exist_ok=True)
    reranker.save(str(reranker_path))
    print(f"Reranker saved to {reranker_path}")


def verify():
    """Quick sanity check — load from disk and run a tiny forward pass."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("\n Verifying embedder ...")
    emb = SentenceTransformer(str(EMBED_PATH))
    vec = emb.encode(["hello world"], normalize_embeddings=True)
    assert vec.shape == (1, 1024), f"Unexpected shape: {vec.shape}"
    print(f"Embedder OK — output shape {vec.shape}")

    print("  Verifying generator ...")
    tok   = AutoTokenizer.from_pretrained(str(GEN_PATH))
    ids   = tok("Hello", return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(
        str(GEN_PATH),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=5, do_sample=False)
    decoded = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generator OK — sample output: '{decoded}'")


def main():
    print(f"\nSaving weights to: {WEIGHTS_BASE.resolve()}\n")
    WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

    print("Embedding model")
    download_embedder()

    print("\nGeneration model")
    download_generator()

    print("\nReranker model")
    download_reranker()

    print("\nVerification")
    verify()

    print(f"Weights saved to:{EMBED_PATH.resolve()} {GEN_PATH.resolve()}")


if __name__ == "__main__":
    main()