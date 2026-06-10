"""
Loads .txt and .md files from a directory into a list of document records
"""

from pathlib import Path
from datetime import datetime

# load text files
def load_document(paths: list[str]) -> list[dict]:
    docs = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"File {path} does not exist")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "doc_id": path.stem,
            "source": str(path),
            "text": text,
            "ingested_at": datetime.utcnow().isoformat(),
        })
        print(f"Loaded {path.name}, ({len(text):,} chars")
    return docs

# load all matching files from dir
def load_from_directory(directory: str, extensions: list[str] = None) -> list[dict]:
    extensions = extensions or [".txt", ".md"]
    paths = [
        str(p) for p in Path(directory).rglob("*")
        if p.suffix in extensions
    ]
    print(f"Loaded {len(paths)} documents from {directory}")
    return load_document(paths)

