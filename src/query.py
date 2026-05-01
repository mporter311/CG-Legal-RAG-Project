"""
query.py  —  CG Legal RAG query engine
=======================================

TWO MODES
---------
1. Retrieval-only  (default, no --llm)
   Aggregates and cleans retrieved chunks; shows structured excerpts.
   Useful for quick lookups and index verification.

2. LLM synthesis  (--llm /path/to/model.gguf)
   Full RAG: retrieved passages → aggregated context → LLM → grounded summary.
   Automatically wraps the prompt in the correct instruction format for
   Mistral / LLaMA / Phi-3 / ChatML models.

IMPORTANT: rebuild the index whenever you re-run ingest_mcm.py:
    python src/build_index.py

Usage
-----
  python src/query.py "What does Article 92 cover?"              # dense-only
  python src/query.py "What does Article 92 cover?" --retriever hybrid
  python src/query.py "What does Article 92 cover?" --retriever hybrid --rerank
  python src/query.py "What is the hazing policy?"
  python src/query.py "What are the elements of Article 128?" --article 128
  python src/query.py "What does Article 92 cover?" --llm "C:\\models\\mistral.gguf"
  python src/query.py "What does Article 92 cover?" --llm "C:\\models\\mistral.gguf" --rerank
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from hybrid_retrieval import load_bm25_index, hybrid_retrieve
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False

try:
    from reranker import Reranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

INDEX_DIR  = Path("data/index")
INDEX_NAME = "pio_rag"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 8

DISCLAIMER = (
    "\n⚠️  I am not a lawyer. This is an informational summary of official "
    "materials only. Consult your chain of command or legal office for advice."
)

LLM_HOW_TO = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 RETRIEVAL-ONLY MODE — no NLP synthesis.
   For a real grounded summary, add --llm with a local GGUF model:

     python src/query.py "What does Article 92 cover?" ^
         --llm "C:\\Users\\YourName\\models\\Mistral-7B-Instruct.Q4_K_M.gguf"

   Free GGUF models : https://huggingface.co/models?search=gguf+instruct
   Install backend  : pip install llama-cpp-python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_index(index_dir: Path, index_name: str = INDEX_NAME):
    """Load FAISS index and metadata. index_name defaults to 'pio_rag'."""
    import faiss
    faiss_path = index_dir / f"{index_name}.faiss"
    meta_path  = index_dir / f"{index_name}_meta.json"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index '{index_name}' not found in {index_dir}.\n"
            f"Run: python src/build_index.py"
        )
    index = faiss.read_index(str(faiss_path))
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_query(query: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Article filter detection
# ---------------------------------------------------------------------------

def detect_article_filter(query: str) -> Optional[str]:
    """
    Return article number if query mentions 'Article NNN' or 'Art. NNN'.
    Captures sub-articles: 87a, 112a, 119b, 128b, etc.
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

    Returns (results, mode) where mode is one of:
      "filtered"   — article filter applied successfully
      "fallback"   — filter returned 0 results; retried without filter
      "unfiltered" — no filter was requested
    """
    query_vec = embed_query(query, model_name)

    if article_filter:
        art_str = str(article_filter).lstrip("0")  # normalise "092" → "92"

        # Stage 1: over-retrieve, then post-filter by article_number
        k_search = min(len(meta), top_k * 15)
        scores, indices = index.search(query_vec, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            # Normalise stored article_number the same way for comparison
            stored = str(chunk.get("article_number", "")).lstrip("0")
            if stored == art_str:
                results.append(chunk)
            if len(results) >= top_k:
                break

        if results:
            print(f"[INFO] Article filter applied: Article {art_str} → {len(results)} chunks")
            return results, "filtered"

        # Stage 2: fallback — article not found in index under that number
        print(f"[WARN] Article filter returned 0 results for Article {art_str}.")
        print(f"[INFO] Article filter fallback activated — retrying without filter.")
        print(f"[INFO] (If this article should exist, rebuild index: python src/build_index.py)")

        scores, indices = index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        print(f"[INFO] Fallback retrieval → {len(results)} chunks (top matches, unfiltered)")
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
        print(f"[INFO] Unfiltered retrieval → {len(results)} chunks")
        return results, "unfiltered"


# ---------------------------------------------------------------------------
# Passage cleaning and aggregation
# ---------------------------------------------------------------------------

# Lines that are purely PDF layout artifacts
_ARTIFACT_LINE_RE = re.compile(
    r"^\s*("
    r"Article\s+\d+|"             # bare "Article 92" heading lines
    r"IV-\d+|"                    # page markers like "IV-28"
    r"\[…\]|\[\.{2,}\]|"         # truncation markers (fixed: was missing [...])
    r"¶\S*|"                      # paragraph reference markers
    r"[Pp]age\s+\d+|"             # page numbers
    r"\d{1,3}\.\s*$"              # bare numbering lines
    r")\s*$"
)

# Broken hyphenation: "enfor-\nceability" → "enforceability"
_BROKEN_HYPHEN_RE = re.compile(r"(\w+)-\n\s*(\w+)")

# Overlap separators injected during ingestion
_OVERLAP_MARKER_RE = re.compile(r"\n\n\[(?:…|\.{2,})\]\n\n")


def clean_text(text: str) -> str:
    """
    Remove PDF layout artifacts and fix broken hyphenation.
    Preserves legal structure (element lists, subsection labels).
    """
    # Fix split words across lines first
    text = _BROKEN_HYPHEN_RE.sub(r"\1\2", text)
    # Remove artifact-only lines
    lines = [l for l in text.split("\n") if not _ARTIFACT_LINE_RE.match(l)]
    # Collapse 3+ blank lines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def aggregate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge chunks from the same article into coherent passages.

    1. Group by article_number.
    2. Sort each group by chunk_index (document order).
    3. Clean and concatenate, removing overlap markers and duplicate content_hashes.
    4. Return one passage dict per article group, sorted by max score.

    Passage dict keys: article_number, article_title, page_start, page_end,
                       chunk_ids, score, text (cleaned + joined).
    """
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        art = chunk.get("article_number", "unknown")
        groups.setdefault(art, []).append(chunk)

    passages = []
    for art, art_chunks in groups.items():
        art_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        combined = ""
        seen_hashes: set[str] = set()
        for c in art_chunks:
            h = c.get("content_hash", "")
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)
            block = clean_text(c["text"])
            if block:
                combined += ("\n\n" if combined else "") + block

        # Remove overlap markers that may remain after joining
        combined = _OVERLAP_MARKER_RE.sub("\n\n", combined)
        # Final collapse of excess blank lines
        combined = re.sub(r"\n{3,}", "\n\n", combined).strip()

        passages.append({
            "article_number": art,
            "article_title":  art_chunks[0].get("article_title", "").strip(),
            "page_start":     min(c.get("page_start", 0) for c in art_chunks),
            "page_end":       max(c.get("page_end",   0) for c in art_chunks),
            "chunk_ids":      [c["chunk_id"] for c in art_chunks],
            "score":          max(c.get("score", 0.0) for c in art_chunks),
            "text":           combined,
        })

    passages.sort(key=lambda p: p["score"], reverse=True)
    print(f"[INFO] Aggregated {len(chunks)} chunks into {len(passages)} passage(s)")
    return passages


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def format_citation(passage: dict, rank: int) -> str:
    source = passage.get("source", "")
    art    = passage.get("article_number", "")
    sec    = passage.get("section_number", "")
    title  = (passage.get("article_title") or passage.get("section_title") or "").strip()
    hp     = passage.get("heading_path", [])
    hp_list = hp if isinstance(hp, list) else []
    ps     = passage.get("page_start", "")
    pe     = passage.get("page_end",   "")
    score  = passage.get("score", 0.0)

    # Build location: article > section > heading path
    if art and art not in ("unknown", "?", ""):
        loc = f"Art. {art}" + (f" — {title}" if title else "")
    elif sec:
        loc = f"§ {sec}" + (f" {title}" if title
              else (f" {hp_list[2]}" if len(hp_list) >= 3 else ""))
    elif hp_list:
        loc = " › ".join(str(h) for h in hp_list[-2:])
    else:
        loc = title or "General"

    pages    = f"pp.{ps}–{pe}" if ps and pe and ps != pe else (f"p.{ps}" if ps else "")
    src_short = source.split("(")[0].strip() if source else "USCG Source"
    return f"[{rank}]  {src_short}  |  {loc}  |  {pages}"

# ---------------------------------------------------------------------------
# Smart excerpt (retrieval-only mode)
# ---------------------------------------------------------------------------

_LEGAL_KW = re.compile(
    r"\b(element|explanation|punish|maximum|means|defined|guilty|offense|"
    r"statute|conviction|intent|knowledge|willful|wrongful|duty|order|"
    r"regulation|absence|derelict|assault|larceny|murder|robbery|"
    r"sentence|discharge|confinement|forfeiture|reckless|drunken|impaired|"
    r"operating|vessel|aircraft|vehicle|control)\b",
    re.IGNORECASE,
)


def smart_excerpt(text: str, max_chars: int = 700) -> str:
    """Extract the most legally informative portion of cleaned passage text."""
    lines = [l for l in text.split("\n") if not _ARTIFACT_LINE_RE.match(l) and l.strip()]
    if not lines:
        return text[:max_chars]

    cleaned = "\n".join(lines)
    if len(cleaned) <= max_chars:
        return cleaned

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
        LLM_HOW_TO,
        f"Query: {query}{fallback_note}\n",
        "=" * 70,
        "## Retrieved passages  (retrieval-only — no NLP synthesis)",
        "(Best-match excerpts from official USCG legal sources)\n",
    ]

    for i, passage in enumerate(passages, 1):
        art        = passage["article_number"]
        title      = passage["article_title"]
        title_part = f" — {title}" if title else ""
        excerpt    = smart_excerpt(passage["text"], max_chars=700)
        lines.append(f"--- [{i}] Article {art}{title_part} ---")
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
# LLM prompt — system instructions
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """\
You are a Coast Guard legal information assistant. You explain what official \
USCG and military law documents say in plain English. You are NOT a lawyer \
and do NOT give legal advice.

RULES — follow exactly:
1. Use ONLY the source passages provided. Do not add outside facts or inferences.
2. Answer the question DIRECTLY in the first 1-2 sentences of ## Answer.
3. Use these section headers IN ORDER (omit ## Punishment if not relevant):

## Answer
(1-3 sentences directly answering the question in plain English)

## Key Points
(3-6 concise bullet points with the most important legal details)

## Punishment
(state max punishment clearly; omit entire section if not about punishment)

## Sources
(numbered list: [N] Source | Section | Pages)

4. End with exactly this one line:
⚠ Informational summary only — consult your legal office for specific guidance.
"""
# ---------------------------------------------------------------------------
# Instruction-format wrapping (model-aware)
# ---------------------------------------------------------------------------

def detect_prompt_format(model_path: str) -> str:
    """
    Detect the instruction-wrapping format from the model filename.
    Returns one of: 'mistral', 'phi3', 'chatml', 'raw'.
    """
    name = Path(model_path).name.lower()
    if any(k in name for k in ("mistral", "llama", "orca", "vicuna", "solar")):
        return "mistral"
    if any(k in name for k in ("phi-3", "phi3", "phi_3")):
        return "phi3"
    if any(k in name for k in ("hermes", "openhermes", "qwen", "chatml", "nous")):
        return "chatml"
    return "raw"


def wrap_prompt(system: str, user_content: str, fmt: str) -> str:
    """
    Wrap system + user content in the model's expected instruction format.

    mistral : <s>[INST] system\\n\\nuser [/INST]
    phi3    : <|system|>\\nsystem<|end|>\\n<|user|>\\nuser<|end|>\\n<|assistant|>
    chatml  : <|im_start|>system\\nsystem<|im_end|>\\n<|im_start|>user\\nuser<|im_end|>\\n<|im_start|>assistant\\n
    raw     : system\\n\\nuser  (plain concatenation, weakest instruction following)
    """
    if fmt == "mistral":
        return f"<s>[INST] {system}\n\n{user_content} [/INST]"
    if fmt == "phi3":
        return (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user_content}<|end|>\n"
            f"<|assistant|>\n"
        )
    if fmt == "chatml":
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    # raw fallback
    return f"{system}\n\n{user_content}"


# User-facing content block (separate from system instructions)
USER_CONTENT_TEMPLATE = """\
QUESTION: {query}

SOURCE PASSAGES:
{separator}
{passages}
{separator}

Answer using ONLY the passages above. Follow the section headers from your rules.
Be concise. Synthesize — do not copy text verbatim.
"""

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def build_llm_answer(
    query: str,
    chunks: list[dict],
    llm_path: str,
    show_prompt: bool = False,
) -> str:
    """
    Run full RAG generation. Only falls back on genuine import/IO failures.
    """
    # --- Guard: import check ---
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        print(
            f"[ERROR] llama-cpp-python not importable: {exc}\n"
            "        Install with: pip install llama-cpp-python\n"
            "        Falling back to retrieval-only output.\n"
        )
        return build_retrieval_only_answer(query, chunks, "unfiltered")

    # --- Guard: file check ---
    llm_path_obj = Path(llm_path)
    if not llm_path_obj.exists():
        print(
            f"[ERROR] Model file not found: {llm_path}\n"
            "        Double-check the path (use quotes for paths with spaces).\n"
            "        Falling back to retrieval-only output.\n"
        )
        return build_retrieval_only_answer(query, chunks, "unfiltered")

    # Aggregate and clean chunks into coherent passages
    passages = aggregate_chunks(chunks)

    # Build passage text block
    sep = "─" * 40
    passage_blocks = []
    for i, p in enumerate(passages, 1):
        cit   = format_citation(p, i)
        ptext = p["text"][:2800]
        if len(p["text"]) > 2800:
            ptext += "\n[... passage continues ...]"
        passage_blocks.append(f"[Passage {i}]\n{cit}\n\n{ptext}")
    passages_text = f"\n{sep}\n".join(passage_blocks)

    user_content = USER_CONTENT_TEMPLATE.format(
        separator=sep,
        passages=passages_text,
        query=query,
    )

    # Detect model format and wrap prompt
    fmt    = detect_prompt_format(str(llm_path_obj))
    prompt = wrap_prompt(LLM_SYSTEM_PROMPT, user_content, fmt)

    print(f"[INFO] Using LLM synthesis mode: {llm_path_obj.name}")
    print(f"[INFO] Prompt format detected: {fmt}")
    print(f"[INFO] Prompt length: ~{len(prompt)//4} tokens  ({len(passages)} passage(s))")

    if show_prompt:
        print("\n" + "=" * 70)
        print("LLM PROMPT (--show-passages)")
        print("=" * 70)
        print(prompt[:3000])
        print("[... prompt continues ...]" if len(prompt) > 3000 else "")
        print("=" * 70 + "\n")

    # --- Load model ---
    print("[INFO] Loading model (may take 10–60 s on first load) ...")
    try:
        llm = Llama(
            model_path=str(llm_path_obj),
            n_ctx=4096,
            n_threads=4,
            verbose=False,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}\n  Falling back to retrieval-only.")
        return build_retrieval_only_answer(query, chunks, "unfiltered")

    # --- Generate ---
    print("[INFO] Generating answer ...")
    try:
        out = llm(
            prompt,
            max_tokens=1200,
            stop=["</s>", "[INST]", "<|im_end|>", "<|end|>", "USER:"],
            temperature=0.1,   # low temperature → factual, grounded output
            repeat_penalty=1.1,
        )
        answer = out["choices"][0]["text"].strip()
    except Exception as exc:
        print(f"[ERROR] Generation failed: {exc}\n  Falling back to retrieval-only.")
        return build_retrieval_only_answer(query, chunks, "unfiltered")

    # Safety: append disclaimer only if model omitted it entirely
    if "legal office" not in answer.lower():
        answer += (
            "\n\n⚠ Informational summary only — "
            "consult your legal office for specific guidance."
        )

    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query the CG Legal RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/query.py "What does Article 92 cover?"
  python src/query.py "What does Article 92 cover?" --retriever hybrid --rerank
  python src/query.py "What are the elements of Article 128?" --article 128
  python src/query.py "What is the Coast Guard hazing policy?"
  python src/query.py "What does Article 92 cover?" --llm "C:\\models\\mistral.gguf"
  python src/query.py "What is the max punishment for larceny?" --llm "C:\\models\\mistral.gguf" --rerank
  python src/query.py "What does Article 112a cover?" --show-passages
        """,
    )
    parser.add_argument("query",           type=str,  help="Natural language question")
    parser.add_argument("--index-dir",     type=Path, default=INDEX_DIR)
    parser.add_argument("--model",         type=str,  default=MODEL_NAME,
                        help="Sentence-transformers embedding model name")
    parser.add_argument("--top-k",         type=int,  default=TOP_K,
                        help="Number of chunks to retrieve (default: 8)")
    parser.add_argument("--article",       type=str,  default=None,
                        help="Filter results to a specific article number, e.g. --article 92")
    parser.add_argument("--llm",           type=str,  default=None,
                        help="Path to a local .gguf model for grounded RAG generation")
    parser.add_argument("--show-passages", action="store_true",
                        help="Print raw retrieved chunks; with --llm also prints the full prompt")
    parser.add_argument(
        "--index-name", type=str, default=INDEX_NAME,
        help="Base name of the FAISS index (default: pio_rag)",
    )
    parser.add_argument(
        "--retriever", type=str, default="hybrid", choices=["dense", "hybrid"],
        help="Retrieval strategy. 'hybrid' uses BM25+dense RRF (recommended).",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help=(
            "Apply cross-encoder reranking over candidates. "
            "Improves answer quality at cost of ~1s additional latency."
        ),
    )
    args = parser.parse_args()

    # Auto-detect article filter
    article_filter = args.article or detect_article_filter(args.query)
    if article_filter:
        print(f"[INFO] Article filter detected: Article {article_filter}")

    print("[INFO] Loading index ...")
    index, meta = load_index(args.index_dir, args.index_name)

    print(f"[INFO] Retrieving top-{args.top_k} chunks ...")
    if args.retriever == "hybrid" and _HYBRID_AVAILABLE:
        bm25_data = load_bm25_index(args.index_dir, args.index_name)
        results, retrieval_mode = hybrid_retrieve(
            args.query, index, meta, bm25_data,
            top_k=args.top_k,
            article_filter=article_filter,
            model_name=args.model,
        )
    else:
        if args.retriever == "hybrid" and not _HYBRID_AVAILABLE:
            print("[WARN] hybrid_retrieval.py not found — falling back to dense.")
        results, retrieval_mode = retrieve(
            args.query, index, meta,
            top_k=args.top_k,
            article_filter=article_filter,
            model_name=args.model,
        )

    if not results:
        print("[WARN] No chunks returned. Check that the index is built.")
        return

    # Optional cross-encoder reranking
    if args.rerank:
        if not _RERANKER_AVAILABLE:
            print("[WARN] reranker.py not found. Skipping reranking.")
        else:
            rerank_pool = max(args.top_k * 3, 20)
            print(f"[INFO] Reranking top-{rerank_pool} candidates ...")
            if args.retriever == "hybrid" and _HYBRID_AVAILABLE:
                pool_results, _ = hybrid_retrieve(
                    args.query, index, meta, bm25_data,
                    top_k=rerank_pool,
                    article_filter=article_filter,
                    model_name=args.model,
                )
            else:
                pool_results, _ = retrieve(
                    args.query, index, meta,
                    top_k=rerank_pool,
                    article_filter=article_filter,
                    model_name=args.model,
                )
            rr = Reranker()
            results = rr.rerank(args.query, pool_results, top_k=args.top_k)
            print(f"[INFO] Top result CE score: {results[0].get('ce_score', 0):.2f}")

    if args.show_passages:
        print_raw_passages(results)

    if args.llm:
        answer = build_llm_answer(
            args.query, results, args.llm,
            show_prompt=args.show_passages,
        )
    else:
        answer = build_retrieval_only_answer(args.query, results, retrieval_mode)

    print("\n" + answer)


if __name__ == "__main__":
    main()
