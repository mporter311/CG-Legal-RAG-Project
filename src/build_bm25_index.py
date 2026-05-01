"""
build_bm25_index.py
===================
Builds a BM25 index from the same JSONL chunks used by build_index.py.
The BM25 index is stored as a pickle alongside the FAISS index and loaded
by query.py / query_wo_model.py at retrieval time.

BM25 tokenises each chunk's text using a simple legal-aware tokeniser, then
stores the fitted BM25Okapi object and the original meta list so the same
chunk ordering is maintained.

Usage:
    python src/build_bm25_index.py
    python src/build_bm25_index.py --processed-dir data/processed --index-dir data/index
    python src/build_bm25_index.py --index-name pio_rag
"""

import json
import glob
import pickle
import re
import argparse
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INDEX_DIR     = Path("data/index")
INDEX_NAME    = "pio_rag"


# ---------------------------------------------------------------------------
# Legal-aware tokeniser
# ---------------------------------------------------------------------------

# Preserve hyphenated legal terms and sub-article numbers (e.g. "112a", "87a")
_PRESERVE_RE = re.compile(
    r"\b(\d{1,3}[a-z])\b"          # sub-article numbers: 112a, 87a, 131b
    r"|\b(ucmj|uscg|cgis|lod|ua|"  # acronyms to keep intact
    r"bm25|bm|rir|aob|pcs|tdy|"
    r"oth|bcd|dd|hyt|secb|serb|"
    r"asb|cg-psc|cg-122|caac|"
    r"faiss|gguf|rao|pio)\b",
    re.IGNORECASE,
)

_SPLIT_RE = re.compile(r"[^a-z0-9'-]+")


def tokenise(text: str) -> list[str]:
    """
    Tokenise chunk text for BM25 indexing.

    - Lowercases
    - Preserves hyphenated terms and legal acronyms
    - Removes pure-numeric tokens shorter than 2 digits (page numbers, etc.)
    - Removes very short tokens (< 2 chars)
    """
    text = text.lower()
    tokens = _SPLIT_RE.split(text)
    result = []
    for tok in tokens:
        tok = tok.strip("-'")
        if len(tok) < 2:
            continue
        # Keep tokens that are meaningful: words, legal terms, numbers >= 2 digits
        if tok.isdigit() and len(tok) < 2:
            continue
        result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_all_chunks(processed_dir: Path) -> list[dict]:
    chunks = []
    files = sorted(processed_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {processed_dir}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            file_chunks = [json.loads(line) for line in f]
        print(f"[INFO]   {path.name}: {len(file_chunks)} chunks")
        chunks.extend(file_chunks)
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Build BM25 index for pio-rag")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--index-dir",     type=Path, default=INDEX_DIR)
    parser.add_argument("--index-name",    type=str,  default=INDEX_NAME)
    args = parser.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[ERROR] rank_bm25 not installed.")
        print("        Run: pip install rank_bm25")
        raise SystemExit(1)

    print("[INFO] Loading chunks ...")
    chunks = load_all_chunks(args.processed_dir)
    print(f"[INFO] {len(chunks)} total chunks loaded")

    print("[INFO] Tokenising ...")
    corpus = [tokenise(c.get("text", "")) for c in chunks]

    print("[INFO] Fitting BM25Okapi ...")
    bm25 = BM25Okapi(corpus)

    args.index_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.index_dir / f"{args.index_name}_bm25.pkl"

    payload = {
        "bm25":      bm25,
        "meta":      chunks,          # same order as FAISS meta
        "index_name": args.index_name,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"[OK] BM25 index saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"[INFO] Vocabulary size: {len(bm25.idf)} terms")


if __name__ == "__main__":
    main()
