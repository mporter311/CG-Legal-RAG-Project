"""
query.py  —  MCM 2019 Punitive Articles RAG query engine
=========================================================

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
  python src/query.py "What does Article 92 cover?"
  python src/query.py "What does Article 92 cover?"  --show-passages
  python src/query.py "What does Article 113 cover?" --article 113
  python src/query.py "What does Article 92 cover?"  --llm "C:\\models\\mistral.gguf"
  python src/query.py "What does Article 92 cover?"  --llm "C:\\models\\mistral.gguf" --show-passages
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

INDEX_DIR  = Path("data/index")
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

def load_index(index_dir: Path):
    import faiss
    faiss_path = index_dir / "mcm_punitive.faiss"
    meta_path  = index_dir / "mcm_punitive_meta.json"
    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run: python src/build_index.py"
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
    """Return article number as string if query mentions 'Article NNN' / 'Art. NNN'."""
    m = re.search(r"\bArticle\s+(\d{1,3})\b", query, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\bArt\.?\s*(\d{1,3})\b", query, re.IGNORECASE)
    if m:
        return m.group(1)
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
    art        = passage.get("article_number", "?")
    title      = passage.get("article_title", "").strip()
    title_part = f" — {title}" if title else ""
    ps         = passage.get("page_start", "?")
    pe         = passage.get("page_end",   "?")
    pages      = f"p.{ps}" if ps == pe else f"pp.{ps}–{pe}"
    cids       = ", ".join(passage.get("chunk_ids", []))
    score      = passage.get("score", 0.0)
    return (
        f"[{rank}] [MCM2019 | Punitive Articles | Art. {art}{title_part} | "
        f"{pages} | chunks: {cids} | score: {score:.3f}]"
    )


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
        "(Best-match excerpts from the 2019 MCM Punitive Articles)\n",
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
You are a Coast Guard legal information assistant helping cadets and junior \
personnel understand U.S. military law. You are NOT a lawyer. You do NOT give \
legal advice. Your role is to explain, in plain language, what the official \
sources say.

STRICT RULES:
1. Use ONLY the provided source passages. Never add outside facts.
2. If passages are insufficient, say so explicitly and recommend the legal office.
3. Answer the QUESTION DIRECTLY in the first paragraph of your summary.
4. Explain what the article criminalizes or regulates.
5. When elements appear in the passages, list them as concise bullets.
6. When maximum punishment appears, state it clearly.
7. Write for a cadet audience: clear, precise, no unexplained jargon.
8. Every factual claim must trace to a citation:
   [MCM2019 | Art. NNN | pp.X-Y | chunk: chunk_id]
9. Use EXACTLY these five section headers, in order:

## Plain-language summary
## Key elements
## Maximum punishment
## Where this comes from (citations)
## Disclaimer
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
RETRIEVED SOURCE PASSAGES
{separator}
{passages}

{separator}
USER QUESTION
{separator}
{query}

Write your response using the five sections specified in your instructions.
Synthesize the passages — explain and summarise, do NOT just copy them.
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
        ptext = p["text"][:2000]
        if len(p["text"]) > 2000:
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

    # Safety: always end with disclaimer even if model omitted it
    if "not a lawyer" not in answer.lower():
        answer += (
            "\n\n## Disclaimer\n"
            "I am not a lawyer; this is an informational summary of official "
            "materials; consult your chain of command or legal office for advice."
        )

    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query the MCM 2019 Punitive Articles RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/query.py "What does Article 92 cover?"
  python src/query.py "What does Article 92 cover?"  --show-passages
  python src/query.py "What does Article 113 cover?" --article 113
  python src/query.py "What does Article 92 cover?"  --llm "C:\\models\\mistral.gguf"
  python src/query.py "What does Article 92 cover?"  --llm "C:\\models\\mistral.gguf" --show-passages
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
    args = parser.parse_args()

    # Auto-detect article filter
    article_filter = args.article or detect_article_filter(args.query)
    if article_filter:
        print(f"[INFO] Article filter detected: Article {article_filter}")

    print("[INFO] Loading index ...")
    index, meta = load_index(args.index_dir)

    print(f"[INFO] Retrieving top-{args.top_k} chunks ...")
    results, retrieval_mode = retrieve(
        args.query, index, meta,
        top_k=args.top_k,
        article_filter=article_filter,
        model_name=args.model,
    )

    if not results:
        print("[WARN] No chunks returned at all. Check that the index is built.")
        return

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
