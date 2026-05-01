"""
chat.py — Ask a Coast Guard Lawyer (Interactive Chat Interface)
===============================================================

Presentation-ready TUI chatbot for the CG Legal RAG system.

Launch:
    python src/chat.py
    python src/chat.py --llm "C:\\path\\to\\mistral-7b-instruct.Q4_K_M.gguf"
    python src/chat.py --retriever hybrid --rerank

Controls:
    Type any question and press Enter.
    Type 'mode' to toggle between retrieval-only and LLM-summary.
    Type 'demo'  to cycle through demo questions.
    Type 'clear' to clear the screen.
    Type 'quit'  to exit.
"""

import json
import sys
import os
import re
import argparse
from pathlib import Path

# ── Rich terminal formatting ──────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    from rich.markdown import Markdown
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

# ── System defaults ───────────────────────────────────────────────────────────
INDEX_DIR   = Path("data/index")
INDEX_NAME  = "pio_rag"
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 8

DISCLAIMER = (
    "\n⚠️  NOT LEGAL ADVICE. This system summarises official USCG materials only. "
    "Consult your chain of command or legal office for guidance on specific cases."
)

DEMO_QUESTIONS = [
    "What does Article 92 cover and what are its elements?",
    "What is the maximum punishment for larceny under Article 121?",
    "What is the Coast Guard policy on hazing and bullying?",
    "What are the Coast Guard rules on fraternization between officers and enlisted?",
    "What types of administrative investigations does the Coast Guard use?",
    "What are the grounds for separating an enlisted member for misconduct?",
    "What constitutes an alcohol incident under Coast Guard policy?",
    "What are the limitations on imposing nonjudicial punishment under Article 15?",
    "Can hazing constitute a UCMJ offense? What article applies?",
    "What is the difference between a punitive discharge and an administrative separation?",
]

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from hybrid_retrieval import (
        load_bm25_index, hybrid_retrieve, detect_article_filter
    )
    _HYBRID_AVAILABLE = True
except ImportError:
    _HYBRID_AVAILABLE = False
    def detect_article_filter(q): return None

try:
    from reranker import Reranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index(index_dir: Path, index_name: str):
    import faiss
    idx   = faiss.read_index(str(index_dir / f"{index_name}.faiss"))
    with open(index_dir / f"{index_name}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta


# ── Dense retrieval ───────────────────────────────────────────────────────────

def dense_retrieve(query, index, meta, model, top_k, article_filter):
    import numpy as np
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    k_pool = min(len(meta), max(top_k * 20, 200))
    scores, indices = index.search(vec, k_pool)
    ranked = [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    if article_filter:
        art = str(article_filter).lstrip("0").lower()
        results = [
            {**meta[i], "score": s}
            for i, s in ranked
            if str(meta[i].get("article_number","")).lstrip("0").lower() == art
        ][:top_k]
        if results:
            return results, "filtered"
        # fallback
        return [{**meta[i], "score": s} for i, s in ranked[:top_k]], "fallback"

    return [{**meta[i], "score": s} for i, s in ranked[:top_k]], "unfiltered"


# ── Passage aggregation ───────────────────────────────────────────────────────

def _norm_text(text: str) -> str:
    import re
    text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
    lines = [l for l in text.split("\n") if l.strip()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def aggregate_chunks(chunks):
    """Group chunks by article or section, deduplicate, return passage dicts."""
    seen = set()
    passages = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        # Group with prior passage if same article/section
        art = c.get("article_number", "")
        sec = c.get("section_number", "")
        key = art if art else sec
        merged = False
        if key and passages:
            last = passages[-1]
            if (last.get("article_number") == art and art) or \
               (last.get("section_number") == sec and sec):
                last["text"] += "\n\n" + c["text"]
                last.get("chunk_ids", [last.get("chunk_id","")]).append(cid)
                merged = True
        if not merged:
            p = dict(c)
            p["chunk_ids"] = [cid]
            passages.append(p)
    return passages


# ── Citation formatting ───────────────────────────────────────────────────────

def format_citation(c: dict, rank: int) -> str:
    src   = c.get("source", "Unknown")
    art   = c.get("article_number", "")
    sec   = c.get("section_number", "")
    title = (c.get("article_title") or c.get("section_title") or "").strip()
    hp    = c.get("heading_path", [])
    p1    = c.get("page_start", "")
    p2    = c.get("page_end", "")
    score = c.get("ce_score", c.get("score", 0.0))
    cids  = c.get("chunk_ids", [c.get("chunk_id","?")])

    # Build location string
    if art:
        loc = f"Art. {art}"
        if title:
            loc += f" — {title}"
    elif sec:
        loc = f"§ {sec}"
        if title:
            loc += f" {title}"
        elif isinstance(hp, list) and len(hp) >= 3:
            loc += f" {hp[2]}"
    else:
        loc = " > ".join(str(h) for h in hp[-2:]) if hp else "?"

    pages = f"pp.{p1}–{p2}" if p1 and p2 else (f"p.{p1}" if p1 else "")
    chunk_str = ", ".join(cids[:2]) + ("…" if len(cids) > 2 else "")

    return f"[{rank}] {src} | {loc} | {pages} | {chunk_str} | score: {score:.3f}"


# ── Display helpers ───────────────────────────────────────────────────────────

console = Console() if _RICH else None

def _print(text, style=None, markup=True):
    if _RICH:
        console.print(text, style=style, markup=markup)
    else:
        print(text)

def print_banner():
    if _RICH:
        banner = Panel(
            "[bold white]Ask a Coast Guard Lawyer[/bold white]\n"
            "[dim]RAG System for USCG Law and Policy[/dim]\n\n"
            "[cyan]Commands:[/cyan] [yellow]demo[/yellow] · [yellow]mode[/yellow] · "
            "[yellow]clear[/yellow] · [yellow]quit[/yellow]",
            title="[bold blue]⚓  CG Legal RAG[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(banner)
    else:
        print("\n" + "=" * 65)
        print("  Ask a Coast Guard Lawyer — CG Legal RAG System")
        print("  Commands: demo, mode, clear, quit")
        print("=" * 65)

def print_retrieval_answer(query, passages, mode, retriever, reranked=False):
    _print(f"\n[dim]Query:[/dim] {query}", markup=True)
    _print(f"[dim]Retriever:[/dim] {retriever}" + (" + reranked" if reranked else ""), markup=True)
    if _RICH:
        console.print(Rule(style="blue"))
    else:
        print("-" * 65)

    if not passages:
        _print("[yellow]No relevant passages found.[/yellow]", markup=True)
        return

    for i, p in enumerate(passages, 1):
        art = p.get("article_number","")
        sec = p.get("section_number","")
        hp  = p.get("heading_path", [])
        title = p.get("article_title") or p.get("section_title") or ""
        if art:
            header = f"Article {art}" + (f" — {title}" if title else "")
        elif sec:
            header = f"§ {sec}"
            if len(hp) >= 3:
                header += f"  {hp[2]}"
        else:
            header = " > ".join(str(h) for h in hp[-2:])

        text_preview = _norm_text(p.get("text",""))[:600]
        if len(_norm_text(p.get("text",""))) > 600:
            text_preview += " …"

        if _RICH:
            console.print(Panel(
                text_preview,
                title=f"[bold cyan][{i}] {header}[/bold cyan]",
                border_style="dim blue",
                padding=(0, 1),
            ))
        else:
            print(f"\n--- [{i}] {header} ---")
            print(text_preview)

    # Citations
    if _RICH:
        console.print(Rule("Citations", style="dim"))
    else:
        print("\nCitations:")

    for i, p in enumerate(passages, 1):
        _print(f"  [dim]{format_citation(p, i)}[/dim]", markup=True)

    _print(f"\n[bold yellow]{DISCLAIMER}[/bold yellow]", markup=True)


def print_llm_answer(query, answer_text):
    if _RICH:
        console.print(Rule(style="blue"))
        md = Markdown(answer_text)
        console.print(md)
    else:
        print("-" * 65)
        print(answer_text)


# ── LLM generation ────────────────────────────────────────────────────────────

def build_context(passages):
    parts = []
    for i, p in enumerate(passages, 1):
        art = p.get("article_number","")
        sec = p.get("section_number","")
        src = p.get("source","")
        loc = f"Article {art}" if art else f"§ {sec}"
        cite = format_citation(p, i)
        parts.append(f"--- Source [{i}]: {cite} ---\n{p.get('text','').strip()}")
    return "\n\n".join(parts)

def run_llm(query, passages, llm_path):
    try:
        from llama_cpp import Llama
    except ImportError:
        return "[LLM not available. Install llama-cpp-python or run without --llm]"

    context = build_context(passages)

    system_prompt = (
        "You are a legal information assistant for U.S. Coast Guard personnel. "
        "Your role is to summarise official USCG legal and policy documents. "
        "You do NOT provide legal advice. You only summarise what official sources say. "
        "Every factual claim must be grounded in the provided source excerpts. "
        "If information is not in the excerpts, say so explicitly."
    )

    user_prompt = (
        f"Using ONLY the following source excerpts, answer this question:\n\n"
        f"QUESTION: {query}\n\n"
        f"SOURCE EXCERPTS:\n{context}\n\n"
        "Provide a clear, structured answer with:\n"
        "1. A plain-language summary (2-4 sentences)\n"
        "2. Key elements or conditions (if applicable)\n"
        "3. Important exceptions or limitations\n"
        "4. Citations for each claim (reference Source [N] numbers above)\n\n"
        "End with: '⚠️ This is a summary of official sources only, not legal advice.'"
    )

    try:
        llm = Llama(
            model_path=llm_path, n_ctx=8192,
            n_threads=4, verbose=False,
        )
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
            repeat_penalty=1.1,
        )
        return out["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ── Main chat loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CG Legal RAG — Interactive Chat")
    parser.add_argument("--index-dir",   type=Path, default=INDEX_DIR)
    parser.add_argument("--index-name",  type=str,  default=INDEX_NAME)
    parser.add_argument("--model",       type=str,  default=MODEL_NAME)
    parser.add_argument("--top-k",       type=int,  default=TOP_K)
    parser.add_argument("--retriever",   type=str,  default="hybrid",
                        choices=["dense","hybrid"])
    parser.add_argument("--rerank",      action="store_true",
                        help="Apply cross-encoder reranking")
    parser.add_argument("--llm",         type=str,  default=None,
                        help="Path to .gguf model for LLM summaries")
    parser.add_argument("--mode",        type=str,  default="auto",
                        choices=["retrieval","llm","auto"],
                        help="Start in retrieval-only or llm mode (auto=retrieval if no --llm)")
    args = parser.parse_args()

    # Resolve mode
    llm_available = args.llm is not None
    if args.mode == "auto":
        current_mode = "llm" if llm_available else "retrieval"
    else:
        current_mode = args.mode
        if current_mode == "llm" and not llm_available:
            print("[WARN] --mode llm requires --llm <path>. Defaulting to retrieval.")
            current_mode = "retrieval"

    # Load models
    print("[INFO] Loading system ...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(args.model)

    print("[INFO] Loading index ...")
    index, meta = load_index(args.index_dir, args.index_name)

    bm25_data = None
    if args.retriever == "hybrid" and _HYBRID_AVAILABLE:
        bm25_data = load_bm25_index(args.index_dir, args.index_name)

    reranker = None
    if args.rerank and _RERANKER_AVAILABLE:
        reranker = Reranker()
        print("[INFO] Reranker loaded.")

    # Determine actual retriever label
    if args.retriever == "hybrid" and bm25_data is not None:
        retriever_label = "Hybrid BM25+Dense"
    else:
        retriever_label = "Dense"
    if reranker:
        retriever_label += " + Cross-Encoder"

    print(f"[INFO] Retriever: {retriever_label}")
    print(f"[INFO] Mode:      {'LLM Summary' if current_mode == 'llm' else 'Retrieval-Only'}")
    if current_mode == "llm":
        print(f"[INFO] LLM:       {Path(args.llm).name}")
    print()

    os.system("cls" if os.name == "nt" else "clear")
    print_banner()

    demo_idx = 0
    history = []

    while True:
        # Prompt
        mode_label = "[LLM]" if current_mode == "llm" else "[Retrieval]"
        if _RICH:
            try:
                query = console.input(f"\n[bold green]{mode_label}[/bold green] [bold]You:[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            try:
                query = input(f"\n{mode_label} You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if not query:
            continue

        # Commands
        if query.lower() == "quit":
            _print("\n[dim]Session ended. Semper Paratus.[/dim]", markup=True)
            break

        if query.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        if query.lower() == "mode":
            if current_mode == "retrieval":
                if not llm_available:
                    _print("[yellow]LLM not loaded. Start with --llm <path> to enable.[/yellow]", markup=True)
                else:
                    current_mode = "llm"
                    _print("[green]Switched to LLM summary mode.[/green]", markup=True)
            else:
                current_mode = "retrieval"
                _print("[green]Switched to retrieval-only mode.[/green]", markup=True)
            continue

        if query.lower() == "demo":
            query = DEMO_QUESTIONS[demo_idx % len(DEMO_QUESTIONS)]
            demo_idx += 1
            _print(f"\n[dim]Demo question:[/dim] [italic]{query}[/italic]", markup=True)

        if query.lower() == "help":
            _print(
                "\n[bold]Commands:[/bold]\n"
                "  [yellow]demo[/yellow]  — cycle through demo questions\n"
                "  [yellow]mode[/yellow]  — toggle retrieval ↔ LLM mode\n"
                "  [yellow]clear[/yellow] — clear screen\n"
                "  [yellow]quit[/yellow]  — exit\n"
                "  Any other text is treated as a legal question.\n",
                markup=True
            )
            continue

        # ── Retrieval ─────────────────────────────────────────────────────────
        article_filter = detect_article_filter(query)
        if article_filter:
            _print(f"[dim]Article filter: {article_filter}[/dim]", markup=True)

        if args.retriever == "hybrid" and bm25_data is not None:
            chunks, ret_mode = hybrid_retrieve(
                query, index, meta, bm25_data,
                top_k=args.top_k if not reranker else args.top_k * 3,
                article_filter=article_filter,
                model_name=args.model,
            )
        else:
            chunks, ret_mode = dense_retrieve(
                query, index, meta, embed_model,
                top_k=args.top_k if not reranker else args.top_k * 3,
                article_filter=article_filter,
            )

        if not chunks:
            _print("[yellow]No relevant content found.[/yellow]", markup=True)
            continue

        # ── Reranking ─────────────────────────────────────────────────────────
        if reranker:
            chunks = reranker.rerank(query, chunks, top_k=args.top_k)
            reranked = True
        else:
            chunks = chunks[:args.top_k]
            reranked = False

        passages = aggregate_chunks(chunks)
        history.append({"query": query, "passages": passages, "mode": current_mode})

        # ── Output ────────────────────────────────────────────────────────────
        if current_mode == "retrieval" or not llm_available:
            print_retrieval_answer(query, passages, ret_mode, retriever_label, reranked)
        else:
            _print(f"\n[dim]Query:[/dim] {query}", markup=True)
            _print("[dim]Generating summary …[/dim]", markup=True)
            answer = run_llm(query, passages, args.llm)
            print_llm_answer(query, answer)
            if _RICH:
                console.print(Rule("Citations", style="dim"))
            else:
                print("\nCitations:")
            for i, p in enumerate(passages, 1):
                _print(f"  [dim]{format_citation(p, i)}[/dim]", markup=True)


if __name__ == "__main__":
    main()
