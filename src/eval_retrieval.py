"""
eval_retrieval.py
=================
Evaluates retrieval quality on a labelled benchmark JSON file.

CHANGES (v5 — performance fix):
  - Hybrid eval now uses hybrid_retrieve_batch() which accepts pre-computed
    query vectors. The embedding model loads ONCE, not once per query.
    This reduces 200-query hybrid eval from ~20 minutes to ~30 seconds.
  - Dense eval already used batch search (no change needed there).
  - Added --retriever flag: 'dense' (default) or 'hybrid'.
  - PREFIX section_number matching (v3): fixes ~34 false negatives.
  - --compare flag diffs two saved JSON files and shows retriever names.
  - Miss report printed after every run.

Usage:
    # Dense baseline (fast, always worked):
    python src/eval_retrieval.py --benchmark eval/benchmark_200.json --ks 1 3 5 8 --verbose
    python src/eval_retrieval.py --benchmark eval/benchmark_200.json --out eval/results/run_dense.json

    # Hybrid (build BM25 index first, then fast):
    python src/eval_retrieval.py --benchmark eval/benchmark_200.json --retriever hybrid --ks 1 3 5 8
    python src/eval_retrieval.py --benchmark eval/benchmark_200.json --retriever hybrid --out eval/results/run_hybrid.json

    # Compare dense vs hybrid:
    python src/eval_retrieval.py --compare eval/results/run_dense.json eval/results/run_hybrid.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


INDEX_DIR  = Path("data/index")
INDEX_NAME = "pio_rag"
MODEL_NAME = "all-MiniLM-L6-v2"

try:
    from hybrid_retrieval import load_bm25_index, hybrid_retrieve_batch
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False

try:
    from reranker import Reranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Index loading
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
    """Embed all queries at once — one model load total."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Gold matching  — PREFIX section matching (v3 fix)
# ---------------------------------------------------------------------------

def _norm_sec(s: str) -> str:
    return str(s).strip().rstrip(".")


def is_relevant(chunk: dict, gold: dict) -> bool:
    """
    Chunk is relevant if it matches any gold label.

    Matching rules:
      chunk_id:        exact match (highest precision)
      article_number:  bidirectional prefix — '112a' matches gold '112' and vice versa
      section_number:  PREFIX match — '1.D.3.a' matches gold '1.D.'
                       PLUS source_doc check: section_number match only counts if the
                       chunk's source document matches gold["source_doc"].
                       This prevents 'A.' in Separations matching 'A.' in AIM.
                       Cross-document questions set source_doc=None to skip this check.
    """
    chunk_ids       = gold.get("chunk_ids", [])
    article_numbers = gold.get("article_numbers", [])
    section_numbers = gold.get("section_numbers", [])
    gold_src        = gold.get("source_doc")     # None means cross-doc: any source OK

    # 1. Exact chunk_id
    if chunk_ids and chunk.get("chunk_id") in chunk_ids:
        return True

    # 2. MCM article_number (bidirectional prefix, no source check needed — MCM only)
    if article_numbers:
        stored_art = str(chunk.get("article_number", "")).lstrip("0").lower()
        if stored_art:
            for g in article_numbers:
                g_norm = str(g).lstrip("0").lower()
                if stored_art == g_norm:
                    return True
                if stored_art.startswith(g_norm) or g_norm.startswith(stored_art):
                    return True

    # 3. Manual section_number (PREFIX + source_doc guard)
    if section_numbers:
        # Source check: skip cross-doc questions (source_doc=None) or when chunk source matches
        chunk_source = chunk.get("source", "")
        source_ok = (gold_src is None) or (gold_src in chunk_source)
        if source_ok:
            stored_sec = _norm_sec(chunk.get("section_number", ""))
            if stored_sec:
                for g in section_numbers:
                    g_norm = _norm_sec(g)
                    if not g_norm:
                        continue
                    if stored_sec.startswith(g_norm) or g_norm.startswith(stored_sec):
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
    retriever: str = "dense",
    rerank: bool = False,
    verbose: bool = False,
) -> dict:

    max_k = max(ks)
    # When reranking, retrieve a larger candidate pool for the reranker to work from
    retrieve_k = max_k * 3 if rerank else max_k   # top-24 or top-8
    print(f"[INFO] Retriever    : {retriever}")

    # ── Embed all queries ONCE ───────────────────────────────────────────────
    print(f"[INFO] Embedding {len(queries)} queries (model loads once) ...")
    query_vecs = embed_queries(queries, model_name)
    print(f"[INFO] Embedding done. Shape: {query_vecs.shape}")

    # ── Retrieve ALL queries ─────────────────────────────────────────────────
    if retriever == "hybrid" and _HYBRID_AVAILABLE:
        print("[INFO] Loading BM25 index ...")
        bm25_data = load_bm25_index(Path("data/index"), INDEX_NAME)
        if bm25_data is None:
            print("[WARN] BM25 index not found. Run: python src/build_bm25_index.py")
            print("[WARN] Falling back to dense retrieval.")
            retriever = "dense"
        elif len(bm25_data["meta"]) != len(meta):
            print(f"[WARN] BM25 has {len(bm25_data['meta'])} chunks, FAISS has {len(meta)}.")
            print("[WARN] Rebuild BM25: python src/build_bm25_index.py. Falling back to dense.")
            retriever = "dense"
        else:
            print(f"[INFO] Running hybrid_retrieve_batch() on {len(queries)} queries ...")
            all_ranked_chunks = hybrid_retrieve_batch(
                query_vecs, queries, index, meta, bm25_data, top_k=retrieve_k,
            )
            all_ranked_scores = [
                [c.get("score", 0.0) for c in chunks]
                for chunks in all_ranked_chunks
            ]
    elif retriever == "hybrid" and not _HYBRID_AVAILABLE:
        print("[WARN] hybrid_retrieval.py not importable. Using dense.")
        retriever = "dense"

    if retriever == "dense":
        print(f"[INFO] Running dense batch search on {len(queries)} queries ...")
        all_scores_arr, all_indices_arr = index.search(query_vecs, retrieve_k)
        all_ranked_chunks = [
            [meta[idx] for idx in all_indices_arr[q_idx] if idx >= 0]
            for q_idx in range(len(queries))
        ]
        all_ranked_scores = [
            [float(all_scores_arr[q_idx][i])
             for i, idx in enumerate(all_indices_arr[q_idx]) if idx >= 0]
            for q_idx in range(len(queries))
        ]

    print(f"[INFO] Retrieval complete. Computing metrics ...")

    # ── Optional reranking ───────────────────────────────────────────────────
    if rerank:
        if not _RERANKER_AVAILABLE:
            print("[WARN] reranker.py not importable. Skipping reranking.")
        else:
            from reranker import Reranker
            reranker = Reranker()
            reranker._load()   # load model once before the loop
            MINI = 16          # queries per mini-batch (controls memory vs. speed)
            reranked_all = []
            print(f"[INFO] Reranking {len(queries)} queries "
                  f"(pool={retrieve_k}, output={max_k}, mini_batch={MINI}) ...")
            for start in range(0, len(queries), MINI):
                end   = min(start + MINI, len(queries))
                batch_q   = queries[start:end]
                batch_c   = all_ranked_chunks[start:end]
                reranked_all.extend(
                    reranker.rerank_batch(batch_q, batch_c, top_k=max_k)
                )
                if (start // MINI) % 4 == 0:
                    print(f"[INFO]   ... {end}/{len(queries)} queries reranked")
            all_ranked_chunks = reranked_all
            all_ranked_scores = [
                [c.get("ce_score", c.get("score", 0.0)) for c in chunks]
                for chunks in all_ranked_chunks
            ]
            print(f"[INFO] Reranking complete.")

    # ── Metrics loop (no retrieval here — just scoring) ──────────────────────
    recall_at_k    = {k: [] for k in ks}
    precision_at_k = {k: [] for k in ks}
    reciprocal_ranks: list[float] = []
    misses: list[dict] = []

    if verbose:
        print("\n" + "-" * 65)
        print("PER-QUERY DETAIL")
        print("-" * 65)

    for q_idx, (qid, query, gold) in enumerate(zip(query_ids, queries, gold_labels)):
        ranked_chunks = all_ranked_chunks[q_idx]
        ranked_scores = all_ranked_scores[q_idx]

        # MRR
        rr = 0.0
        first_hit_rank = None
        for rank, chunk in enumerate(ranked_chunks[:max_k], 1):
            if is_relevant(chunk, gold):
                rr = 1.0 / rank
                first_hit_rank = rank
                break
        reciprocal_ranks.append(rr)

        if rr == 0.0:
            misses.append({
                "id": qid,
                "query": query,
                "gold": gold,
                "top5": [c.get("article_number") or c.get("section_number", "?")
                         for c in ranked_chunks[:5]],
                "top_score": ranked_scores[0] if ranked_scores else 0.0,
            })

        # Recall / Precision @ k
        for k in ks:
            top_k_chunks = ranked_chunks[:k]
            n_rel = sum(1 for c in top_k_chunks if is_relevant(c, gold))
            recall_at_k[k].append(1.0 if n_rel > 0 else 0.0)
            precision_at_k[k].append(n_rel / k)

        if verbose:
            top_arts = [
                c.get("article_number") or c.get("section_number", "?")
                for c in ranked_chunks[:5]
            ]
            gold_label = (gold.get("article_numbers", []) or
                          gold.get("section_numbers", []))
            hit_str = f"rank {first_hit_rank}" if first_hit_rank else "MISS"
            top_score = ranked_scores[0] if ranked_scores else 0.0
            print(f"\n[{qid}] {query[:65]}")
            print(f"  Gold  : {gold_label}")
            print(f"  Top-5 : {top_arts}  (score: {top_score:.3f})")
            print(f"  MRR   : {rr:.3f}  ({hit_str})")
            for k in ks:
                r = 1.0 if sum(1 for c in ranked_chunks[:k] if is_relevant(c, gold)) > 0 else 0.0
                p = sum(1 for c in ranked_chunks[:k] if is_relevant(c, gold)) / k
                print(f"  @{k:<2} R={r:.2f} P={p:.2f}")

    results: dict = {
        "num_queries": len(queries),
        "retriever":   retriever,
        "MRR":         float(np.mean(reciprocal_ranks)),
        "_misses":     misses,
    }
    for k in ks:
        r  = float(np.mean(recall_at_k[k]))
        p  = float(np.mean(precision_at_k[k]))
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        results[f"Recall@{k}"]    = r
        results[f"Precision@{k}"] = p
        results[f"F1@{k}"]        = f1

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics: dict, ks: list[int]) -> None:
    retriever = metrics.get("retriever", "dense")
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION REPORT  [{retriever.upper()}]")
    print("=" * 60)
    print(f"Queries evaluated : {metrics['num_queries']}")
    print(f"MRR               : {metrics['MRR']:.4f}")
    print()
    print(f"  {'k':>2}   {'Recall':>8}   {'Precision':>10}   {'F1':>6}")
    print(f"  {'--':<2}   {'--------':<8}   {'----------':<10}   {'------':<6}")
    for k in ks:
        r  = metrics.get(f"Recall@{k}",    0)
        p  = metrics.get(f"Precision@{k}", 0)
        f1 = metrics.get(f"F1@{k}",        0)
        print(f"  {k:>2}   {r:>8.4f}   {p:>10.4f}   {f1:>6.4f}")
    print()

    misses  = metrics.get("_misses", [])
    n_miss  = len(misses)
    n_total = metrics["num_queries"]
    hit1    = round(metrics.get("Recall@1", 0) * n_total)
    print(f"Summary: {hit1}/{n_total} HIT@1  |  {n_miss}/{n_total} MISS")
    print("=" * 60)

    if misses:
        print(f"\n-- MISS REPORT ({n_miss} queries) " + "-" * 30)
        for m in misses:
            gold_label = (m["gold"].get("article_numbers") or
                          m["gold"].get("section_numbers") or ["?"])
            print(f"  [{m['id']}] score={m['top_score']:.3f}  gold={gold_label}")
            print(f"         {m['query'][:70]}")
            print(f"         top5: {m['top5']}")

    p8 = metrics.get("Precision@8", 0)
    if p8 < 0.50:
        print(
            "\n[ADVISORY] Precision@8 below 0.50. Try:\n"
            "  1. --retriever hybrid  (if not already using it)\n"
            "  2. Upgrade embedding model to BAAI/bge-small-en-v1.5\n"
            "  3. Review MISS REPORT — some may be benchmark label gaps\n"
        )


def print_comparison(path_a: Path, path_b: Path) -> None:
    with open(path_a) as f: a = json.load(f)
    with open(path_b) as f: b = json.load(f)

    ret_a = a.get("retriever", "unknown")
    ret_b = b.get("retriever", "unknown")
    name_a = f"{path_a.stem} [{ret_a}]"
    name_b = f"{path_b.stem} [{ret_b}]"

    metric_keys = sorted(
        k for k in set(list(a) + list(b))
        if not k.startswith("_")
        and k not in ("num_queries", "retriever", "score_distribution")
    )

    col = 22
    print(f"\n{'Metric':<{col}} {name_a:>{col}} {name_b:>{col}} {'Delta':>10}")
    print("-" * (col * 3 + 12))
    for k in metric_keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            arrow = "^ " if delta > 0.001 else ("v " if delta < -0.001 else "= ")
            print(f"{k:<{col}} {va:>{col}.4f} {vb:>{col}.4f}  {arrow}{abs(delta):>7.4f}")

    if ret_a == ret_b:
        print(f"\n[NOTE] Both runs used '{ret_a}' — delta will be near zero.")
        print("       Run one with --retriever dense and one with --retriever hybrid.")


def log_mlflow(metrics: dict, run_name: str = "pio_rag_eval") -> None:
    try:
        import mlflow
        mlflow.set_experiment("pio-rag-retrieval")
        with mlflow.start_run(run_name=run_name):
            for key, val in metrics.items():
                if key.startswith("_"):
                    continue
                if key in ("num_queries", "retriever"):
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
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval for pio-rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/eval_retrieval.py --benchmark eval/benchmark_200.json --ks 1 3 5 8 --verbose
  python src/eval_retrieval.py --benchmark eval/benchmark_200.json --out eval/results/run_dense.json
  python src/eval_retrieval.py --benchmark eval/benchmark_200.json --retriever hybrid --out eval/results/run_hybrid.json
  python src/eval_retrieval.py --compare eval/results/run_dense.json eval/results/run_hybrid.json
        """,
    )
    parser.add_argument("--benchmark",  type=Path, default=Path("eval/benchmark.json"))
    parser.add_argument("--index-dir",  type=Path, default=INDEX_DIR)
    parser.add_argument("--index-name", type=str,  default=INDEX_NAME)
    parser.add_argument("--model",      type=str,  default=MODEL_NAME)
    parser.add_argument("--ks",         type=int,  nargs="+", default=[1, 3, 5, 8])
    parser.add_argument(
        "--retriever", type=str, default="dense", choices=["dense", "hybrid"],
        help=(
            "'dense' (default) or 'hybrid' (BM25 + dense RRF). "
            "Requires: python src/build_bm25_index.py (run once after build_index.py)."
        ),
    )
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-query detail after the batch run completes")
    parser.add_argument("--mlflow",     action="store_true")
    parser.add_argument("--out",        type=Path, default=None,
                        help="Save metrics JSON (required for --compare)")
    parser.add_argument("--compare",    type=Path, nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Compare two saved result JSON files")
    parser.add_argument(
        "--rerank", action="store_true",
        help=(
            "Apply cross-encoder reranking after first-stage retrieval. "
            "Reranks top-k candidates using cross-encoder/ms-marco-MiniLM-L-6-v2. "
            "Improves Precision@1 at cost of ~2x latency."
        ),
    )
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    print(f"[INFO] Loading benchmark from {args.benchmark}")
    with open(args.benchmark, encoding="utf-8") as f:
        benchmark = json.load(f)

    questions = benchmark["questions"]
    declared  = benchmark.get("metadata", {}).get("total_questions")
    actual    = len(questions)
    if declared is not None and int(declared) != actual:
        print(f"[WARN] metadata.total_questions={declared} but {actual} found.")

    queries     = [q["query"] for q in questions]
    gold_labels = [q["gold"]  for q in questions]
    query_ids   = [q["id"]    for q in questions]

    print(f"[INFO] {actual} questions loaded")
    print(f"[INFO] Loading index '{args.index_name}' from {args.index_dir}")
    index, meta = load_index(args.index_dir, args.index_name)

    metrics = compute_metrics(
        queries, gold_labels, query_ids,
        index, meta, args.ks, args.model,
        retriever=args.retriever,
        rerank=args.rerank,
        verbose=args.verbose,
    )
    print_report(metrics, args.ks)

    if args.mlflow:
        log_mlflow(metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save = {k: v for k, v in metrics.items() if not k.startswith("_")}
        save["reranked"] = args.rerank
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(save, f, indent=2)
        print(f"[OK] Results saved to {args.out}")


if __name__ == "__main__":
    main()
