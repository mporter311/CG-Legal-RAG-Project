"""
build_index.py
==============
Embeds chunks from mcm_punitive_chunks.jsonl using sentence-transformers
and stores them in a FAISS index with a parallel metadata store.

Why FAISS over Chroma?
  - FAISS is fully offline, no server, tiny footprint.
  - For < 50 k chunks a flat L2/IP index is fast enough on CPU.
  - We store metadata in a companion JSON file; filtering is done
    post-retrieval in Python (fast enough at this scale).
  - Chroma adds a nice server/HTTP interface but introduces more
    moving parts that are unnecessary for a classroom / offline build.

Why all-MiniLM-L6-v2?
  - 80 MB, 384-dim, ~14k tokens/sec on CPU — practical for daily use.
  - Solid general-domain retrieval quality for English legal text.
  - For a future upgrade: 'BAAI/bge-small-en-v1.5' adds ~10% quality
    at ~2× inference cost; 'multi-qa-MiniLM-L6-cos-v1' is tuned for QA pairs.

Usage:
    python src/build_index.py
    python src/build_index.py --chunks data/processed/mcm_punitive_chunks.jsonl \\
                               --model all-MiniLM-L6-v2 \\
                               --index-dir data/index
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Any

SEED = 42


def load_chunks(jsonl_path: Path) -> tuple[list[str], list[dict]]:
    texts, metas = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj.pop("text"))
            metas.append(obj)
    return texts, metas


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Loading embedding model '{model_name}' …")
    model = SentenceTransformer(model_name)

    print(f"[INFO] Embedding {len(texts)} chunks (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot-product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> Any:
    import faiss

    dim = embeddings.shape[1]
    # IndexFlatIP = exact inner-product search on L2-normalised vecs == cosine
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(
    index: Any,
    metas: list[dict],
    texts: list[str],
    index_dir: Path,
) -> None:
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / "mcm_punitive.faiss"
    meta_path  = index_dir / "mcm_punitive_meta.json"

    faiss.write_index(index, str(faiss_path))

    # Store text alongside metadata for retrieval display
    combined = []
    for i, m in enumerate(metas):
        entry = dict(m)
        entry["text"] = texts[i]
        combined.append(entry)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[OK] FAISS index → {faiss_path}")
    print(f"[OK] Metadata   → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS vector index from MCM chunks")
    parser.add_argument("--chunks",    type=Path, default=Path("data/processed/mcm_punitive_chunks.jsonl"))
    parser.add_argument("--model",     type=str,  default="all-MiniLM-L6-v2")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--batch",     type=int,  default=64)
    args = parser.parse_args()

    np.random.seed(SEED)

    print(f"[INFO] Loading chunks from {args.chunks}")
    texts, metas = load_chunks(args.chunks)
    print(f"[INFO] {len(texts)} chunks loaded")

    embeddings = embed_texts(texts, args.model, args.batch)

    index = build_faiss_index(embeddings)

    save_index(index, metas, texts, args.index_dir)
    print("[DONE] Index ready.")


if __name__ == "__main__":
    main()
