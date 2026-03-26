"""
build_index.py
==============
Embeds chunks from all JSONL files in data/processed/ and stores them in a
FAISS index with a parallel metadata store.

By default, loads ALL *.jsonl files in --chunks-dir (data/processed/).
To index only specific files, pass --chunks-files explicitly.

CHANGES (v3):
  - Multi-document support: globs data/processed/*.jsonl by default,
    so MCM + Conduct Manual + Separations Manual are all indexed together.
  - --chunks-files flag for explicit file list when needed.
  - Logs per-source chunk counts for verification.
  - Index name defaults to 'pio_rag' to reflect multi-document scope.

Why FAISS over Chroma?
  Fully offline, no server, tiny footprint. For < 50k chunks a flat
  IndexFlatIP on CPU is fast enough.

Why all-MiniLM-L6-v2 (default)?
  80 MB, 384-dim, ~14k tokens/sec on CPU. Upgrade path: BAAI/bge-small-en-v1.5.
  Changing models REQUIRES rebuilding the index.

Usage:
    # Index everything in data/processed/ (default):
    python src/build_index.py

    # Specific files only:
    python src/build_index.py \\
        --chunks-files data/processed/mcm_punitive_chunks.jsonl

    # Custom index name:
    python src/build_index.py --index-name mcm_only
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Any


SEED = 42


def load_chunks(paths: list[Path]) -> tuple[list[str], list[dict]]:
    """Load and merge chunks from one or more JSONL files."""
    texts, metas = [], []
    source_counts: Counter = Counter()

    for path in paths:
        count_before = len(texts)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(obj.pop("text"))
                metas.append(obj)
        added = len(texts) - count_before
        src = metas[-1].get("source", path.name) if metas else path.name
        source_counts[src] += added
        print(f"[INFO]   {path.name}: {added} chunks")

    print(f"[INFO] Source breakdown:")
    for src, cnt in source_counts.most_common():
        print(f"[INFO]   {src}: {cnt} chunks")

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
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    emb = embeddings.astype(np.float32)

    assert emb.ndim == 2, f"Expected 2D array, got shape {emb.shape}"
    assert emb.shape[0] == len(texts), "Embedding count mismatch"
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        print(f"[WARN] Embeddings not unit-normalized (mean norm={norms.mean():.4f}).")

    return emb


def build_faiss_index(embeddings: np.ndarray) -> Any:
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(
    index: Any,
    metas: list[dict],
    texts: list[str],
    index_dir: Path,
    index_name: str,
) -> None:
    import faiss
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = index_dir / f"{index_name}.faiss"
    meta_path  = index_dir / f"{index_name}_meta.json"

    faiss.write_index(index, str(faiss_path))

    combined = []
    for i, m in enumerate(metas):
        entry = dict(m)
        entry["text"] = texts[i]
        combined.append(entry)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"[OK] FAISS index → {faiss_path}")
    print(f"[OK] Metadata   → {meta_path}")
    print(f"[OK] {index.ntotal} vectors, dim={index.d}")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from all processed chunks"
    )
    parser.add_argument(
        "--chunks-dir", type=Path, default=Path("data/processed"),
        help="Directory to glob for *.jsonl chunk files (default: data/processed/)",
    )
    parser.add_argument(
        "--chunks-files", type=Path, nargs="+", default=None,
        help="Explicit list of JSONL files to index (overrides --chunks-dir glob)",
    )
    parser.add_argument("--model",      type=str,  default="all-MiniLM-L6-v2")
    parser.add_argument("--index-dir",  type=Path, default=Path("data/index"))
    parser.add_argument(
        "--index-name", type=str, default="pio_rag",
        help="Base name for .faiss and _meta.json files (default: pio_rag)",
    )
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(SEED)

    # Resolve chunk files
    if args.chunks_files:
        chunk_paths = args.chunks_files
    else:
        chunk_paths = sorted(args.chunks_dir.glob("*.jsonl"))
        if not chunk_paths:
            print(f"[ERROR] No .jsonl files found in {args.chunks_dir}")
            return

    print(f"[INFO] Loading chunks from {len(chunk_paths)} file(s):")
    texts, metas = load_chunks(chunk_paths)
    print(f"[INFO] {len(texts)} total chunks loaded")

    embeddings = embed_texts(texts, args.model, args.batch)
    index      = build_faiss_index(embeddings)
    save_index(index, metas, texts, args.index_dir, args.index_name)
    print("[DONE] Index ready.")


if __name__ == "__main__":
    main()
