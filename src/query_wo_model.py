"""
query_wo_model.py  —  Coast Guard Legal RAG query engine (retrieval-only)
=========================================================================

Retrieval-only version of query.py — no LLM/Mistral generation step.
Loads the FAISS index, embeds the query, retrieves top-K chunks, aggregates
and displays structured excerpts with citations. Faster than query.py because
it skips model loading entirely.

TWO MODES
---------
1. Filtered  : --article NNN  (or auto-detected from query text)
               Over-retrieves then post-filters to the specified article.
2. Unfiltered: default; returns top-K by cosine similarity across all chunks.

Usage
-----
  python src/query_wo_model.py "What does Article 92 cover?"
  python src/query_wo_model.py "What does Article 92 cover?" --show-passages
  python src/query_wo_model.py "What does Article 113 cover?" --article 113
  python src/query_wo_model.py "What is wrongful use of controlled substances?" --article 112a
  python src/query_wo_model.py "What does Article 92 cover?" --index-name mcm2019_punitive
"""

import json
import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

INDEX_DIR   = Path("data/index")
INDEX_NAME  = "pio_rag"               # base name: pio_rag.faiss + pio_rag_meta.json
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 8

# Score below which we warn that unfiltered retrieval results may be low-confidence.
SCORE_THRESHOLD_WARN = 0.65

DISCLAIMER = (
    "\n⚠️  I am not a lawyer. This is an informational summary of official "
    "materials only. Consult your chain of command or legal office for advice."
)


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_index(index_dir: Path, index_name: str):
    import faiss
    faiss_path = index_dir / f"{index_name}.faiss"
    meta_path  = index_dir / f"{index_name}_meta.json"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index not found: {faiss_path}\n"
            f"Run: python src/build_index.py --index-name {index_name}"
        )
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[INFO] Index loaded: {index.ntotal} vectors, dim={index.d}")
    return index, meta


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_query(query: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, local_files_only=True)
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Article filter detection
# ---------------------------------------------------------------------------

def detect_article_filter(query: str) -> Optional[str]:
    """
    Return article number as string if query mentions 'Article NNN' or 'Art. NNN'.
    Handles sub-article suffixes like '112a', '87a'.
    """
    m = re.search(r"\bArticle\s+(\d{1,3}[a-z]?)\b", query, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    m = re.search(r"\bArt\.?\s*(\d{1,3}[a-z]?)\b", query, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


# ---------------------------------------------------------------------------
# Retrieval with automatic fallback
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    index,
    meta: list[dict],
    top_k: int = TOP_K,
    article_filter: Optional[str] = None,
    model_name: str = MODEL_NAME,
) -> tuple[list[dict], str]:
    """
    Dense retrieval with two-stage article filtering.

    Returns (results, mode) where mode in {"filtered", "fallback", "unfiltered"}.
    """
    query_vec = embed_query(query, model_name)

    if article_filter:
        art_str = article_filter.lstrip("0").lower()

        k_search = min(len(meta), top_k * 20)
        scores, indices = index.search(query_vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            stored = str(chunk.get("article_number", "")).lstrip("0").lower()
            if stored == art_str:
                results.append(chunk)
            if len(results) >= top_k:
                break

        if results:
            print(f"[INFO] Article filter applied: Article {art_str} → {len(results)} chunks")
            return results, "filtered"

        print(f"[WARN] Article filter returned 0 results for Article {art_str}.")
        print(f"[INFO] Retrying without filter.")
        print(f"[INFO] (If this article exists, rebuild the index: python src/build_index.py)")
        scores, indices = index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        print(f"[INFO] Fallback retrieval → {len(results)} chunks (unfiltered)")
        return results, "fallback"

    else:
        scores, indices = index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        top_score = results[0]["score"] if results else 0.0
        if top_score < SCORE_THRESHOLD_WARN:
            print(
                f"[WARN] Top retrieval score {top_score:.3f} is below threshold "
                f"{SCORE_THRESHOLD_WARN}. Results may be low-confidence for this query.\n"
                f"       Try adding the article number explicitly, e.g. --article 128"
            )
        print(f"[INFO] Unfiltered retrieval → {len(results)} chunks (top score: {top_score:.3f})")
        return results, "unfiltered"


# ---------------------------------------------------------------------------
# Passage cleaning and aggregation
# ---------------------------------------------------------------------------

_ARTIFACT_LINE_RE = re.compile(
    r"^\s*("
    r"IV-\d+|"
    r"V-\d+|"
    r"\[…\]|\[\.{2,}\]|"
    r"¶\S*|"
    r"[Pp]age\s+\d+|"
    r"\d{1,3}\.\s*$"
    r")\s*$"
)

_BROKEN_HYPHEN_RE  = re.compile(r"(\w+)-\n\s*(\w+)")
_OVERLAP_MARKER_RE = re.compile(r"\n\n\[(?:…|\.{2,})\]\n\n")


def clean_text(text: str) -> str:
    """Remove PDF layout artifacts and fix broken hyphenation."""
    text = _BROKEN_HYPHEN_RE.sub(r"\1\2", text)
    lines = [l for l in text.split("\n") if not _ARTIFACT_LINE_RE.match(l)]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def aggregate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge chunks from the same document section into coherent passages.
    """
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        key = (chunk.get("article_number")
               or chunk.get("section_number")
               or "unknown")
        groups.setdefault(key, []).append(chunk)

    passages = []
    for key, grp in groups.items():
        if any("chunk_index" in c for c in grp):
            grp.sort(key=lambda c: c.get("chunk_index", 0))

        combined = ""
        seen_hashes: set[str] = set()
        for c in grp:
            h = c.get("content_hash", "")
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)
            block = clean_text(c["text"])
            if block:
                combined += ("\n\n" if combined else "") + block

        combined = _OVERLAP_MARKER_RE.sub("\n\n", combined)
        combined = re.sub(r"\n{3,}", "\n\n", combined).strip()

        first = grp[0]
        source = first.get("source", "Unknown")

        passages.append({
            "article_number": first.get("article_number", ""),
            "article_title":  (first.get("article_title") or "").strip(),
            "section_number": first.get("section_number", ""),
            "section_title":  (first.get("section_title") or "").strip(),
            "chapter_number": first.get("chapter_number", ""),
            "chapter_title":  (first.get("chapter_title") or "").strip(),
            "heading_path":   first.get("heading_path", []),
            "page_start":     min(c.get("page_start", 0) for c in grp),
            "page_end":       max(c.get("page_end",   0) for c in grp),
            "chunk_ids":      [c["chunk_id"] for c in grp],
            "score":          max(c.get("score", 0.0) for c in grp),
            "source":         source,
            "text":           combined,
        })

    passages.sort(key=lambda p: p["score"], reverse=True)
    print(f"[INFO] Aggregated {len(chunks)} chunks into {len(passages)} passage(s)")
    return passages


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def format_citation(passage: dict, rank: int) -> str:
    source = passage.get("source", "MCM")
    ps     = passage.get("page_start", "?")
    pe     = passage.get("page_end",   "?")
    pages  = f"p.{ps}" if ps == pe else f"pp.{ps}–{pe}"
    cids   = ", ".join(passage.get("chunk_ids", []))
    score  = passage.get("score", 0.0)

    art_num   = passage.get("article_number")
    sec_num   = passage.get("section_number")
    art_title = (passage.get("article_title") or "").strip()
    sec_title = (passage.get("section_title") or "").strip()

    if art_num:
        title_part   = f" — {art_title}" if art_title else ""
        section_part = f"Punitive Articles | Art. {art_num}{title_part}"
    else:
        heading_path = passage.get("heading_path", [])
        if heading_path:
            section_part = " > ".join(heading_path)
        elif sec_num:
            title_part   = f" — {sec_title}" if sec_title else ""
            section_part = f"{sec_num}{title_part}"
        else:
            section_part = passage.get("section", "Unknown Section")

    return (
        f"[{rank}] [{source} | {section_part} | "
        f"{pages} | chunks: {cids} | score: {score:.3f}]"
    )


# ---------------------------------------------------------------------------
# Smart excerpt — prioritises Elements section
# ---------------------------------------------------------------------------

_ELEMENTS_RE = re.compile(r"\bElements?\b", re.IGNORECASE)

_LEGAL_KW = re.compile(
    r"\b(element|explanation|punish|maximum|means|defined|guilty|offense|"
    r"statute|conviction|intent|knowledge|willful|wrongful|duty|order|"
    r"regulation|absence|derelict|assault|larceny|murder|robbery|"
    r"sentence|discharge|confinement|forfeiture|reckless|drunken|impaired|"
    r"operating|vessel|aircraft|vehicle|control)\b",
    re.IGNORECASE,
)


def format_legal_text(text: str) -> str:
    """Re-apply indentation to legal outline text extracted from a two-column PDF."""
    MARKERS = [
        (re.compile(r"^([a-e])\.\s"), 0),
        (re.compile(r"^\((\d{1,2})\)\s"), 4),
        (re.compile(r"^\(([a-z])\)\s"), 8),
        (re.compile(r"^\((i{1,3}|iv|vi{0,3}|ix|xi{0,3})\)\s", re.IGNORECASE), 12),
        (re.compile(r"^\[Note:", re.IGNORECASE), 4),
    ]

    lines = text.split("\n")
    result = []
    current_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        matched = False
        for pattern, indent in MARKERS:
            if pattern.match(stripped):
                current_indent = indent
                result.append(" " * indent + stripped)
                matched = True
                break

        if not matched:
            cont_indent = current_indent + 2 if current_indent > 0 else 0
            result.append(" " * cont_indent + stripped)

    return "\n".join(result)


def smart_excerpt(text: str, max_chars: int = 800) -> str:
    """Extract the most legally informative portion of a cleaned passage."""
    lines = [l for l in text.split("\n") if not _ARTIFACT_LINE_RE.match(l) and l.strip()]
    if not lines:
        return text[:max_chars]

    cleaned = "\n".join(lines)
    if len(cleaned) <= max_chars:
        return cleaned

    elements_start: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*(?:[a-z]\.)?\s*Elements?\.?\s*$", line, re.IGNORECASE):
            elements_start = i
            break
        if re.search(r"\bElements?\.", line, re.IGNORECASE) and len(line) < 80:
            elements_start = i
            break

    best = elements_start if elements_start is not None else None

    if best is None:
        scores = [len(_LEGAL_KW.findall(l)) for l in lines]
        best   = scores.index(max(scores)) if max(scores) > 0 else 0

    result_lines, total = [], 0
    for i in range(best, len(lines)):
        l = lines[i]
        if total + len(l) + 1 > max_chars and result_lines:
            break
        result_lines.append(l)
        total += len(l) + 1

    if best > 0 and total < max_chars // 2:
        for i in range(best - 1, -1, -1):
            l = lines[i]
            if total + len(l) + 1 > max_chars:
                break
            result_lines.insert(0, l)
            total += len(l) + 1

    result = "\n".join(result_lines).strip()
    return (result + " …") if len(cleaned) > total else result


# ---------------------------------------------------------------------------
# Retrieval-only answer
# ---------------------------------------------------------------------------

def build_retrieval_only_answer(
    query: str, chunks: list[dict], retrieval_mode: str
) -> str:
    if not chunks:
        return (
            "No relevant passages found.\n"
            "Try rephrasing or use --article NNN to target a specific article.\n"
            + DISCLAIMER
        )

    passages = aggregate_chunks(chunks)
    fallback_note = (
        "\n⚠️  Article filter returned 0 results; showing top unfiltered matches.\n"
        if retrieval_mode == "fallback" else ""
    )

    lines = [
        f"Query: {query}{fallback_note}\n",
        "=" * 70,
        "## Retrieved passages",
        "(Best-match excerpts from the MCM Punitive Articles)\n",
    ]

    for i, passage in enumerate(passages, 1):
        art_num    = passage.get("article_number")
        art_title  = (passage.get("article_title") or "").strip()
        sec_num    = passage.get("section_number")

        if art_num:
            title_part = f" — {art_title}" if art_title else ""
            header = f"Article {art_num}{title_part}"
        else:
            heading_path = passage.get("heading_path", [])
            header = " > ".join(heading_path) if heading_path else (sec_num or "Unknown")

        excerpt    = format_legal_text(smart_excerpt(passage["text"], max_chars=800))
        lines.append(f"--- [{i}] {header} ---")
        lines.append(excerpt)
        lines.append("")

    lines += ["=" * 70, "## Citations", ""]
    for i, passage in enumerate(passages, 1):
        lines.append(format_citation(passage, i))

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debug: raw passages
# ---------------------------------------------------------------------------

def print_raw_passages(chunks: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("RAW PASSAGES (--show-passages debug mode)")
    print("=" * 70)
    for i, c in enumerate(chunks, 1):
        art = c.get("article_number", "?")
        cid = c.get("chunk_id", "?")
        sc  = c.get("score", 0.0)
        print(f"\n[{i}] Art.{art} | chunk: {cid} | score: {sc:.3f}")
        print("-" * 50)
        print(c["text"][:1200])
        if len(c["text"]) > 1200:
            print("  [... truncated ...]")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query the MCM Punitive Articles RAG system (retrieval-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/query_wo_model.py "What does Article 92 cover?"
  python src/query_wo_model.py "What does Article 92 cover?"  --show-passages
  python src/query_wo_model.py "What does Article 113 cover?" --article 113
  python src/query_wo_model.py "What is wrongful use of controlled substances?" --article 112a
  python src/query_wo_model.py "What does Article 92 cover?"  --index-name mcm2019_punitive
        """,
    )
    parser.add_argument("query",            type=str,  help="Natural language question")
    parser.add_argument("--index-dir",      type=Path, default=INDEX_DIR)
    parser.add_argument(
        "--index-name", type=str, default=INDEX_NAME,
        help="Base name of the FAISS index (default: pio_rag)",
    )
    parser.add_argument("--model",          type=str,  default=MODEL_NAME)
    parser.add_argument("--top-k",          type=int,  default=TOP_K)
    parser.add_argument(
        "--article", type=str, default=None,
        help="Filter to a specific article number, e.g. --article 92  or  --article 112a",
    )
    parser.add_argument(
        "--show-passages", action="store_true",
        help="Print raw retrieved chunks before the formatted answer",
    )
    args = parser.parse_args()

    article_filter = args.article or detect_article_filter(args.query)
    if article_filter:
        print(f"[INFO] Article filter detected: Article {article_filter}")

    print("[INFO] Loading index ...")
    index, meta = load_index(args.index_dir, args.index_name)

    print(f"[INFO] Retrieving top-{args.top_k} chunks ...")
    results, retrieval_mode = retrieve(
        args.query, index, meta,
        top_k=args.top_k,
        article_filter=article_filter,
        model_name=args.model,
    )

    if not results:
        print("[WARN] No chunks returned. Check that the index is built.")
        return

    if args.show_passages:
        print_raw_passages(results)

    answer = build_retrieval_only_answer(args.query, results, retrieval_mode)
    print("\n" + answer)


if __name__ == "__main__":
    main()
