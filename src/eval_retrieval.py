"""
eval_retrieval.py
=================
Evaluates retrieval quality on a labelled benchmark JSON file.

CHANGES FROM ORIGINAL (v2):
  - Configurable --index-name (supports both 2019 and 2024 indexes)
  - Per-query detail output (--verbose) to identify which queries are failing
  - Added F1@k metric alongside Recall and Precision
  - Score distribution printed for each evaluation run (diagnostic)
  - Warns if benchmark metadata total_questions doesn't match actual count

Metrics:
  Recall@k    — fraction of queries where any gold chunk/article appears in top-k
  Precision@k — fraction of top-k results that are relevant (gold-labeled)
  F1@k        — harmonic mean of Recall@k and Precision@k
  MRR         — Mean Reciprocal Rank of the first relevant result

Note on current gold label granularity:
  Gold labels currently specify article_numbers only (not specific chunk_ids).
  This means a retrieved chunk is "relevant" if it comes from the correct article,
  regardless of whether it actually contains the specific answer content.
  This inflates both Precision@k and MRR relative to true answer-quality precision.
  TODO: Add chunk-level gold IDs to benchmark.json after index inspection.

Usage:
    python src/eval_retrieval.py --benchmark eval/benchmark.json
    python src/eval_retrieval.py --benchmark eval/benchmark.json --ks 1 3 5 8
    python src/eval_retrieval.py --benchmark eval/benchmark.json --verbose
    python src/eval_retrieval.py --benchmark eval/benchmark.json --index-name mcm2019_punitive
    python src/eval_retrieval.py --benchmark eval/benchmark.json --mlflow
    python src/eval_retrieval.py --benchmark eval/benchmark.json --out eval/results/run1.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional


INDEX_DIR  = Path("data/index")
INDEX_NAME = "pio_rag"
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def load_index(index_dir: Path, index_name: str):
    import faiss
    faiss_path = index_dir / f"{index_name}.faiss"
    meta_path  = index_dir / f"{index_name}_meta.json"
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


def embed_queries(queries: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


def retrieve_batch(query_vecs: np.ndarray, index, top_k: int) -> tuple:
    scores, indices = index.search(query_vecs, top_k)
    return scores, indices


# ---------------------------------------------------------------------------
# Gold matching
# ---------------------------------------------------------------------------

def is_relevant(chunk: dict, gold: dict) -> bool:
    """
    A chunk is relevant if it matches ANY of:
      - gold["chunk_ids"]:       exact chunk_id match  (fine-grained, preferred)
      - gold["article_numbers"]: chunk's article_number (MCM) in the gold list
      - gold["section_numbers"]: chunk's section_number (manuals) in the gold list

    Normalisation: leading zeros stripped, lowercase, for article_number matching.
    section_number matching is exact (e.g. "1.F.3." must match "1.F.3.").
    """
    chunk_ids       = gold.get("chunk_ids", [])
    article_numbers = gold.get("article_numbers", [])
    section_numbers = gold.get("section_numbers", [])   # for Conduct/Sep manual questions

    if chunk_ids and chunk.get("chunk_id") in chunk_ids:
        return True

    # MCM: article_number match (normalised)
    if article_numbers:
        stored = str(chunk.get("article_number", "")).lstrip("0").lower()
        gold_arts = [str(a).lstrip("0").lower() for a in article_numbers]
        if stored and stored in gold_arts:
            return True

    # Manuals: section_number match (exact)
    if section_numbers:
        stored_sec = str(chunk.get("section_number", ""))
        if stored_sec and stored_sec in section_numbers:
            return True

    return False


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    queries: list[str],
    gold_labels: list[dict],
    query_ids: list[str],
    index,
    meta: list[dict],
    ks: list[int],
    model_name: str,
    verbose: bool = False,
) -> dict:

    max_k = max(ks)
    print(f"[INFO] Embedding {len(queries)} queries …")
    query_vecs = embed_queries(queries, model_name)

    print(f"[INFO] Searching index (k={max_k}) …")
    all_scores, indices = retrieve_batch(query_vecs, index, max_k)

    recall_at_k    = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []

    if verbose:
        print("\n" + "─" * 60)
        print("PER-QUERY DETAIL")
        print("─" * 60)

    for q_idx, (qid, query, gold) in enumerate(zip(query_ids, queries, gold_labels)):
        ranked_chunks = [meta[i] for i in indices[q_idx] if i >= 0]
        ranked_scores = [all_scores[q_idx][i] for i, idx in enumerate(indices[q_idx]) if idx >= 0]

        # MRR
        rr = 0.0
        first_hit_rank = None
        for rank, chunk in enumerate(ranked_chunks[:max_k], 1):
            if is_relevant(chunk, gold):
                rr = 1.0 / rank
                first_hit_rank = rank
                break
        reciprocal_ranks.append(rr)

        # Recall & Precision @ k
        for k in ks:
            top_chunks = ranked_chunks[:k]
            n_relevant  = sum(1 for c in top_chunks if is_relevant(c, gold))
            recall_at_k[k].append(1.0 if n_relevant > 0 else 0.0)
            precision_at_k[k].append(n_relevant / k)

        if verbose:
            top_arts = [
                c.get("article_number") or c.get("section_number", "?")
                for c in ranked_chunks[:5]
            ]
            gold_arts = gold.get("article_numbers", [])
            hit_str = f"first hit @ rank {first_hit_rank}" if first_hit_rank else "MISS"
            top_score = ranked_scores[0] if ranked_scores else 0.0
            print(f"\n[{qid}] {query[:60]}")
            print(f"  Gold articles : {gold_arts}")
            print(f"  Top-5 articles: {top_arts}  (top score: {top_score:.3f})")
            print(f"  MRR           : {rr:.3f}  ({hit_str})")
            for k in ks:
                r = 1.0 if sum(1 for c in ranked_chunks[:k] if is_relevant(c, gold)) > 0 else 0.0
                p = sum(1 for c in ranked_chunks[:k] if is_relevant(c, gold)) / k
                print(f"  @{k:<2}  R={r:.2f}  P={p:.2f}")

    results: dict = {
        "num_queries": len(queries),
        "MRR": float(np.mean(reciprocal_ranks)),
    }
    for k in ks:
        r = float(np.mean(recall_at_k[k]))
        p = float(np.mean(precision_at_k[k]))
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        results[f"Recall@{k}"]    = r
        results[f"Precision@{k}"] = p
        results[f"F1@{k}"]        = f1

    # Score distribution for diagnostics
    top1_scores = [float(all_scores[q][0]) for q in range(len(queries)) if len(indices[q]) > 0]
    if top1_scores:
        results["score_distribution"] = {
            "top1_mean":  float(np.mean(top1_scores)),
            "top1_min":   float(np.min(top1_scores)),
            "top1_max":   float(np.max(top1_scores)),
        }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: dict, ks: list[int]) -> None:
    print("\n" + "=" * 56)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 56)
    print(f"Queries evaluated : {metrics['num_queries']}")
    print(f"MRR               : {metrics['MRR']:.4f}")
    print()
    print(f"  {'k':>2}   {'Recall':>8}   {'Precision':>10}   {'F1':>6}")
    print(f"  {'─'*2}   {'─'*8}   {'─'*10}   {'─'*6}")
    for k in ks:
        r  = metrics.get(f"Recall@{k}",    0)
        p  = metrics.get(f"Precision@{k}", 0)
        f1 = metrics.get(f"F1@{k}",        0)
        print(f"  {k:>2}   {r:>8.4f}   {p:>10.4f}   {f1:>6.4f}")
    print()
    if "score_distribution" in metrics:
        sd = metrics["score_distribution"]
        print(f"Top-1 score  mean={sd['top1_mean']:.3f}  "
              f"min={sd['top1_min']:.3f}  max={sd['top1_max']:.3f}")
    print("=" * 56 + "\n")

    # Precision advisory for legal RAG context
    p8 = metrics.get("Precision@8", 0)
    if p8 < 0.50:
        print(
            "[ADVISORY] Precision@8 is below 0.50. For a legal information system\n"
            "           this means more than half of retrieved chunks are from the\n"
            "           wrong article. Consider:\n"
            "             1. Upgrading to BAAI/bge-small-en-v1.5 embedding model\n"
            "             2. Adding BM25 hybrid retrieval\n"
            "             3. Reducing top-k or adding score threshold filtering\n"
        )


def log_mlflow(metrics: dict, run_name: str = "mcm_retrieval_eval") -> None:
    try:
        import mlflow
        mlflow.set_experiment("pio-rag-retrieval")
        with mlflow.start_run(run_name=run_name):
            for key, val in metrics.items():
                if key == "num_queries":
                    mlflow.log_param(key, val)
                elif key == "score_distribution":
                    for k2, v2 in val.items():
                        mlflow.log_metric(k2, v2)
                else:
                    mlflow.log_metric(key, val)
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
    parser.add_argument(
        "--index-name", type=str, default=INDEX_NAME,
        help="Base name of the FAISS index (default: mcm_punitive)",
    )
    parser.add_argument("--model",      type=str,  default=MODEL_NAME)
    parser.add_argument("--ks",         type=int,  nargs="+", default=[1, 3, 5, 8])
    parser.add_argument("--verbose",    action="store_true", help="Print per-query detail")
    parser.add_argument("--mlflow",     action="store_true", help="Log results to MLflow")
    parser.add_argument("--out",        type=Path, default=None, help="Save results JSON")
    args = parser.parse_args()

    print(f"[INFO] Loading benchmark from {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]

    # Validate metadata
    declared = benchmark.get("metadata", {}).get("total_questions")
    actual   = len(questions)
    if declared is not None and int(declared) != actual:
        print(
            f"[WARN] benchmark.json metadata.total_questions={declared} "
            f"but {actual} questions found. Using actual count."
        )

    queries     = [q["query"] for q in questions]
    gold_labels = [q["gold"]  for q in questions]
    query_ids   = [q["id"]    for q in questions]

    print(f"[INFO] {actual} questions loaded")
    print(f"[INFO] Loading index '{args.index_name}' from {args.index_dir}")
    index, meta = load_index(args.index_dir, args.index_name)

    metrics = compute_metrics(
        queries, gold_labels, query_ids,
        index, meta, args.ks, args.model,
        verbose=args.verbose,
    )
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
