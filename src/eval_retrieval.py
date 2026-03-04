"""
eval_retrieval.py
=================
Evaluates retrieval quality on a labelled benchmark JSON file.

Metrics computed:
  - Recall@k    — fraction of queries where any gold chunk/article appears in top-k
  - Precision@k — fraction of top-k results that are relevant (gold)
  - MRR         — Mean Reciprocal Rank of the first relevant result

Usage:
    python src/eval_retrieval.py --benchmark eval/benchmark.json
    python src/eval_retrieval.py --benchmark eval/benchmark.json --ks 1 3 5 8
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional


INDEX_DIR  = Path("data/index")
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Retrieval (duplicated lean version to avoid circular imports)
# ---------------------------------------------------------------------------

def load_index(index_dir: Path):
    import faiss
    faiss_path = index_dir / "mcm_punitive.faiss"
    meta_path  = index_dir / "mcm_punitive_meta.json"
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def embed_queries(queries: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def retrieve_batch(query_vecs: np.ndarray, index, top_k: int) -> tuple:
    scores, indices = index.search(query_vecs, top_k)
    return scores, indices


# ---------------------------------------------------------------------------
# Gold matching
# ---------------------------------------------------------------------------

def is_relevant(chunk: dict, gold: dict) -> bool:
    """
    A chunk is relevant if it matches ANY of:
      - gold["chunk_ids"]: exact chunk_id match
      - gold["article_numbers"]: chunk's article_number in the list
    """
    chunk_ids = gold.get("chunk_ids", [])
    article_numbers = gold.get("article_numbers", [])

    if chunk_ids and chunk.get("chunk_id") in chunk_ids:
        return True
    if article_numbers and chunk.get("article_number") in [str(a) for a in article_numbers]:
        return True
    return False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    queries: list[str],
    gold_labels: list[dict],
    index,
    meta: list[dict],
    ks: list[int],
    model_name: str,
) -> dict:

    max_k = max(ks)
    print(f"[INFO] Embedding {len(queries)} queries …")
    query_vecs = embed_queries(queries, model_name)

    print(f"[INFO] Searching index (k={max_k}) …")
    _, indices = retrieve_batch(query_vecs, index, max_k)

    recall_at_k    = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    reciprocal_ranks = []

    for q_idx, (query, gold) in enumerate(zip(queries, gold_labels)):
        ranked_chunks = [meta[i] for i in indices[q_idx] if i >= 0]

        # MRR
        rr = 0.0
        for rank, chunk in enumerate(ranked_chunks[:max_k], 1):
            if is_relevant(chunk, gold):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall & Precision @ k
        for k in ks:
            top_chunks = ranked_chunks[:k]
            n_relevant  = sum(1 for c in top_chunks if is_relevant(c, gold))
            # Recall: was at least one relevant doc found?
            recall_at_k[k].append(1.0 if n_relevant > 0 else 0.0)
            precision_at_k[k].append(n_relevant / k)

    results = {
        "num_queries": len(queries),
        "MRR": float(np.mean(reciprocal_ranks)),
    }
    for k in ks:
        results[f"Recall@{k}"]    = float(np.mean(recall_at_k[k]))
        results[f"Precision@{k}"] = float(np.mean(precision_at_k[k]))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: dict, ks: list[int]) -> None:
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 50)
    print(f"Queries evaluated : {metrics['num_queries']}")
    print(f"MRR               : {metrics['MRR']:.4f}")
    print()
    for k in ks:
        r = metrics.get(f"Recall@{k}", 0)
        p = metrics.get(f"Precision@{k}", 0)
        print(f"  k={k:2d}   Recall={r:.4f}   Precision={p:.4f}")
    print("=" * 50 + "\n")


def log_mlflow(metrics: dict, run_name: str = "mcm_retrieval_eval") -> None:
    try:
        import mlflow
        mlflow.set_experiment("pio-rag-retrieval")
        with mlflow.start_run(run_name=run_name):
            for key, val in metrics.items():
                if key != "num_queries":
                    mlflow.log_metric(key, val)
            mlflow.log_param("num_queries", metrics["num_queries"])
        print("[INFO] MLflow run logged")
    except Exception as exc:
        print(f"[WARN] MLflow logging skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics")
    parser.add_argument("--benchmark",  type=Path, default=Path("eval/benchmark.json"))
    parser.add_argument("--index-dir",  type=Path, default=INDEX_DIR)
    parser.add_argument("--model",      type=str,  default=MODEL_NAME)
    parser.add_argument("--ks",         type=int,  nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--mlflow",     action="store_true", help="Log results to MLflow")
    parser.add_argument("--out",        type=Path, default=None, help="Save results JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading benchmark from {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]
    queries      = [q["query"]  for q in questions]
    gold_labels  = [q["gold"]   for q in questions]

    print(f"[INFO] Loading index from {args.index_dir}")
    index, meta = load_index(args.index_dir)

    metrics = compute_metrics(queries, gold_labels, index, meta, args.ks, args.model)
    print_report(metrics, args.ks)

    if args.mlflow:
        log_mlflow(metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Results saved to {args.out}")


if __name__ == "__main__":
    main()
