"""
hybrid_retrieval.py  v4
========================
BM25 + Dense hybrid retrieval using Reciprocal Rank Fusion (RRF).

CHANGES v4:
  - Article injection: when query contains "Article X", all chunks with
    article_number=X are injected into the BM25 candidate list at synthetic
    rank positions 1..N. This guarantees article chunks participate in RRF
    even when the dense model has near-zero scores for them (e.g. q001, q177).
    This is distinct from article_filter (which restricts results) — injection
    only boosts article chunks within an otherwise-unrestricted result set.

CHANGES v3:
  - batch API (hybrid_retrieve_batch) avoids re-embedding per query in eval loop.
  - Dense score preserved separately from RRF score for display.

CHANGES v2:
  - Empty-string guard on article filter (_art_matches).
  - Full fused-list iteration for article filter (no 4× cap).
  - Query expansion table for known semantic gaps.

RRF: score(c) = 1/(60 + dense_rank) + 1/(60 + bm25_rank)
"""

import pickle
import re
import numpy as np
from pathlib import Path
from typing import Optional


RRF_K = 60
SCORE_THRESHOLD_WARN = 0.65

# ---------------------------------------------------------------------------
# Query expansion — known semantic gaps between user phrasing and MCM text
# ---------------------------------------------------------------------------
_QUERY_EXPANSIONS = {
    r"\bhazing\b":                      "cruelty maltreatment Article 93",
    r"\bchild abuse\b":                 "Article 119b child endangerment",
    r"\bdomestic violence\b":           "Article 128b intimate partner",
    r"\bparole violation\b":            "Article 107a",
    r"\b87a\b":                         "resistance flight breach arrest escape",
    r"\b119b\b":                        "child abuse endangerment offenses",
    r"\bdrug use\b":                    "controlled substance wrongful use Article 112a",
    r"\bimpaired driving\b":            "drunken reckless operation Article 113",
    r"\bsanctuary\b":                   "sanctuary provision reserve 18 years 20 years",
    r"\boff.?limits\b":                 "off-limits locations servicemember establishments",
    r"\bunderground newspaper\b":       "underground newspapers printed materials",
    r"\bwrongful use\b":               "controlled substance Article 112a",
    r"\bpersonal property\b":          "personal effects absentee deserter held unit",
    r"\bua\b":                          "unauthorized absence absentee deserter",
    r"\bunauthorized absence\b":        "absentee deserter held unit personal effects",
    # F_PHRASING fixes for remaining misses
    r"\bsecond alcohol incident\b":     "separation conditions alcohol incident H",
    r"\balcohol incident.*conseq\b":    "separation conditions H administrative",
    r"\bdrug abuse.*separat\b":         "misconduct controlled substance Q separation",
    r"\bseparat.*drug abuse\b":         "misconduct controlled substance Q separation",
    r"\barticle 31\b":                  "rights warning DI drug incident legal rights",
    r"\b31\(b\)\b":                   "rights warning suspected UCMJ violation",
    r"\bconvicted.*civilian court\b":   "arrested civil authorities 1.B commanding officer",
    r"\bcivilian court.*convicted\b":   "arrested civil authorities 1.B commanding officer",
    r"\bvoluntary.*separat.*lieu\b":    "SILO separation lieu orders V disposition",
    r"\bseparat.*lieu.*board\b":        "SILO separation lieu orders V disposition",
}


def _expand_query(query: str) -> str:
    for pattern, expansion in _QUERY_EXPANSIONS.items():
        if re.search(pattern, query, re.IGNORECASE):
            return f"{query} {expansion}"
    return query


# ---------------------------------------------------------------------------
# BM25 index loading
# ---------------------------------------------------------------------------

def load_bm25_index(index_dir: Path, index_name: str) -> Optional[dict]:
    bm25_path = index_dir / f"{index_name}_bm25.pkl"
    if not bm25_path.exists():
        return None
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] BM25 index loaded: {len(data['meta'])} chunks, "
          f"vocab={len(data['bm25'].idf)} terms")
    return data


# ---------------------------------------------------------------------------
# Tokeniser — must match build_bm25_index.py exactly
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"[^a-z0-9'-]+")


def _tokenise(text: str) -> list[str]:
    text = text.lower()
    return [t.strip("-'") for t in _SPLIT_RE.split(text)
            if len(t.strip("-'")) >= 2]


# ---------------------------------------------------------------------------
# Embedding — single-query mode only
# ---------------------------------------------------------------------------

def _embed_query(query: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Article helpers
# ---------------------------------------------------------------------------

def detect_article_filter(query: str) -> Optional[str]:
    m = re.search(r"\bArticle\s+(\d{1,3}[a-z]?)\b", query, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\bArt\.?\s*(\d{1,3}[a-z]?)\b", query, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _norm_art(s: str) -> str:
    return str(s).lstrip("0").lower().strip()


def _art_matches(stored: str, target: str) -> bool:
    if not stored:
        return False
    if stored == target:
        return True
    return stored.startswith(target) or target.startswith(stored)


# ---------------------------------------------------------------------------
# Article injection (v4 addition)
# ---------------------------------------------------------------------------

def _inject_article_chunks(
    query_text: str,
    meta: list[dict],
    bm25_ranked: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """
    If the query mentions 'Article X', inject all chunks with article_number=X
    at the front of the BM25 ranked list (synthetic rank 1, 2, ...).

    This guarantees article chunks participate in RRF even when the dense model
    assigns near-zero scores to them (q001: Art.92 dense_score=0.000).

    Injection does NOT filter out other results — it only boosts article chunks.
    The RRF contribution of injected chunks at rank 1 is 1/(60+1) = 0.01639,
    which is strong enough to push them into top-8 even without dense support.
    """
    art_num = detect_article_filter(query_text)
    if not art_num:
        return bm25_ranked

    art_str = _norm_art(art_num)
    injected_indices = [
        i for i, c in enumerate(meta)
        if _art_matches(_norm_art(c.get("article_number", "")), art_str)
    ]

    if not injected_indices:
        return bm25_ranked

    # Build set of already-present indices (avoid duplication)
    existing = {idx for idx, _ in bm25_ranked}
    new_entries = [(i, 9999.0) for i in injected_indices if i not in existing]

    if not new_entries:
        # All article chunks already in list — boost them to front
        article_set = set(injected_indices)
        art_entries  = [(i, s) for i, s in bm25_ranked if i in article_set]
        rest_entries = [(i, s) for i, s in bm25_ranked if i not in article_set]
        return art_entries + rest_entries

    # Prepend injected chunks, then rest of BM25 ranked list
    return new_entries + bm25_ranked


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_ranked: list[tuple[int, float]],
    bm25_ranked:  list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_ranked, 1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(bm25_ranked, 1):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Core: vector-in fusion
# ---------------------------------------------------------------------------

def _fuse_from_vec(
    query_vec: np.ndarray,
    query_text: str,
    dense_index,
    dense_meta: list[dict],
    bm25_data: dict,
    top_k: int,
    article_filter: Optional[str],
    verbose: bool = True,
) -> tuple[list[dict], str]:
    n = len(dense_meta)
    k_pool = min(n, max(top_k * 20, 200))

    # Dense retrieval
    d_scores, d_indices = dense_index.search(query_vec, k_pool)
    dense_ranked = [
        (int(idx), float(score))
        for score, idx in zip(d_scores[0], d_indices[0])
        if idx >= 0
    ]
    dense_score_map = {idx: score for idx, score in dense_ranked}

    # BM25 retrieval with query expansion
    expanded = _expand_query(query_text)
    if verbose and expanded != query_text:
        print(f"[INFO] BM25 query expanded: '{expanded[:80]}'")
    tokens = _tokenise(expanded)
    bm25_scores_arr = (bm25_data["bm25"].get_scores(tokens)
                       if tokens else np.zeros(n))
    bm25_top_idx = np.argsort(bm25_scores_arr)[::-1][:k_pool]
    bm25_ranked = [
        (int(i), float(bm25_scores_arr[i]))
        for i in bm25_top_idx
        if bm25_scores_arr[i] > 0
    ]

    # v4: Article injection — boost article chunks to front of BM25 list
    bm25_ranked = _inject_article_chunks(query_text, dense_meta, bm25_ranked)
    if verbose:
        art = detect_article_filter(query_text)
        if art:
            injected = [i for i, _ in bm25_ranked[:5]
                        if _art_matches(_norm_art(dense_meta[i].get("article_number","")), _norm_art(art))]
            if injected:
                print(f"[INFO] Article '{art}' injection: {len(injected)} chunks boosted in BM25 pool")

    # RRF fusion
    fused = _rrf_fuse(dense_ranked, bm25_ranked)

    # Article post-filter (restricts output to one article — for explicit --article flag)
    if article_filter:
        art_str = _norm_art(article_filter)
        results = []
        for idx, rrf_score in fused:
            chunk = dict(dense_meta[idx])
            stored = _norm_art(chunk.get("article_number", ""))
            if _art_matches(stored, art_str):
                chunk["score"]     = dense_score_map.get(idx, 0.0)
                chunk["rrf_score"] = rrf_score
                results.append(chunk)
            if len(results) >= top_k:
                break
        if results:
            if verbose:
                print(f"[INFO] Hybrid filtered: Article {art_str} -> {len(results)} chunks")
            return results, "hybrid_filtered"
        if verbose:
            print(f"[WARN] Article filter 0 results for '{art_str}'. Unfiltered fallback.")
        results = []
        for idx, rrf_score in fused[:top_k]:
            chunk = dict(dense_meta[idx])
            chunk["score"]     = dense_score_map.get(idx, 0.0)
            chunk["rrf_score"] = rrf_score
            results.append(chunk)
        return results, "hybrid_fallback"

    # Unfiltered
    results = []
    for idx, rrf_score in fused[:top_k]:
        chunk = dict(dense_meta[idx])
        chunk["score"]     = dense_score_map.get(idx, 0.0)
        chunk["rrf_score"] = rrf_score
        results.append(chunk)

    top_dense = dense_ranked[0][1] if dense_ranked else 0.0
    if verbose:
        if top_dense < SCORE_THRESHOLD_WARN:
            print(f"[WARN] Dense top score {top_dense:.3f} < {SCORE_THRESHOLD_WARN}.")
        print(f"[INFO] Hybrid retrieval: {len(results)} chunks "
              f"(dense top: {top_dense:.3f}, BM25 candidates: {len(bm25_ranked)})")
    return results, "hybrid_unfiltered"


# ---------------------------------------------------------------------------
# Public API 1: single-query (query.py / query_wo_model.py)
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    dense_index,
    dense_meta: list[dict],
    bm25_data: Optional[dict],
    top_k: int = 8,
    article_filter: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[dict], str]:
    query_vec = _embed_query(query, model_name)

    if bm25_data is None:
        print("[INFO] BM25 index not found — dense-only.")
        print("[INFO] Run: python src/build_bm25_index.py")
        return _dense_fallback(query_vec, dense_index, dense_meta, top_k, article_filter)

    if len(bm25_data["meta"]) != len(dense_meta):
        print(f"[WARN] BM25/FAISS size mismatch. Rebuild: python src/build_bm25_index.py")
        return _dense_fallback(query_vec, dense_index, dense_meta, top_k, article_filter)

    return _fuse_from_vec(
        query_vec, query, dense_index, dense_meta,
        bm25_data, top_k, article_filter, verbose=True,
    )


# ---------------------------------------------------------------------------
# Public API 2: batch (eval_retrieval.py)
# ---------------------------------------------------------------------------

def hybrid_retrieve_batch(
    query_vecs: np.ndarray,
    queries: list[str],
    dense_index,
    dense_meta: list[dict],
    bm25_data: Optional[dict],
    top_k: int = 8,
) -> list[list[dict]]:
    """
    Pre-computed query vectors; no model reload per query.

    Auto-detects article numbers in each query (mirrors runtime query.py behavior).
    When "Article X" is detected, applies article_filter so Art.X chunks are
    guaranteed in the result set — matching how the system behaves for end users.
    """
    if bm25_data is None or len(bm25_data["meta"]) != len(dense_meta):
        all_scores, all_indices = dense_index.search(query_vecs, top_k)
        return [
            [dict({**dense_meta[idx], "score": float(all_scores[q][i])})
             for i, idx in enumerate(all_indices[q]) if idx >= 0]
            for q in range(len(queries))
        ]

    results = []
    for q_idx, (query_text, query_vec) in enumerate(zip(queries, query_vecs)):
        # Auto-detect article filter — mirrors runtime query.py behavior
        auto_filter = detect_article_filter(query_text)
        chunks, _ = _fuse_from_vec(
            query_vec[np.newaxis, :], query_text,
            dense_index, dense_meta, bm25_data, top_k,
            article_filter=auto_filter, verbose=False,
        )
        results.append(chunks)
    return results


# ---------------------------------------------------------------------------
# Dense-only fallback
# ---------------------------------------------------------------------------

def _dense_fallback(
    query_vec, dense_index, dense_meta, top_k, article_filter
) -> tuple[list[dict], str]:
    k_pool = min(len(dense_meta), max(top_k * 20, 200))
    d_scores, d_indices = dense_index.search(query_vec, k_pool)
    dense_ranked = [(int(idx), float(s))
                    for s, idx in zip(d_scores[0], d_indices[0]) if idx >= 0]

    if article_filter:
        art_str = _norm_art(article_filter)
        results = []
        for idx, score in dense_ranked:
            chunk = dict(dense_meta[idx])
            if _art_matches(_norm_art(chunk.get("article_number", "")), art_str):
                chunk["score"] = score
                results.append(chunk)
            if len(results) >= top_k:
                break
        if results:
            return results, "dense_filtered"
        results = [dict({**dense_meta[idx], "score": s}) for idx, s in dense_ranked[:top_k]]
        return results, "dense_fallback"

    return ([dict({**dense_meta[idx], "score": s}) for idx, s in dense_ranked[:top_k]],
            "dense_unfiltered")
