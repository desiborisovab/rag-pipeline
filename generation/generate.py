import os
import torch
from pathlib import Path

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from utils.config import get_config

_generator = None

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context passages."
    "If the answer is not in the context, say 'I dont have enough information to answer that.' Be concise and precise."
)

WEIGHTS_BASE = Path(os.environ.get("WEIGHTS_PATH", "weights"))
GEN_WEIGHTS_PATH = WEIGHTS_BASE / "tinyllama"

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_generator():
    global _generator
    if _generator is not None:
        return _generator

    cfg = get_config()["generation"]
    device = get_device()
    load_from = str(GEN_WEIGHTS_PATH) if GEN_WEIGHTS_PATH.exists() else cfg["model_id"]

    if GEN_WEIGHTS_PATH.exists():
        print("Loading generator from {}".format(GEN_WEIGHTS_PATH))
    else:
        print("Generating generator from {}".format(cfg["model_id"]))

    tokenizer = AutoTokenizer.from_pretrained(load_from)

    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,).to("mps")

    else:
        model = AutoModelForCausalLM.from_pretrained(
            load_from,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    if not GEN_WEIGHTS_PATH.exists():
        print(f"Saving weights to {GEN_WEIGHTS_PATH}")
        GEN_WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
        model.cpu().save_pretrained(str(GEN_WEIGHTS_PATH))
        tokenizer.save_pretrained(str(GEN_WEIGHTS_PATH))
        if device == "mps":
            model.to("mps")

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if device == "mps" else None,
        device_map=None if device == "mps" else "auto"
    )
    print(f"Generator ready on device {device}")

    return _generator

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"<|assistant|>\n"
    )

def generate(query: str, chunks: list[dict]) -> dict:
    cfg = get_config()["generation"]
    gen = get_generator()
    prompt = build_prompt(query, chunks)

    output = gen(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        do_sample=True,
        pad_token_id=gen.tokenizer.eos_token_id
    )

    full_text: str = output[0]["generated_text"]
    answer = full_text[len(prompt):].strip()

    return {
        "answer": answer,
        "prompt": prompt,
        "sources": list({c["source"] for c in chunks})
    }
