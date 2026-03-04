"""
ingest_mcm.py
=============
Extracts the Punitive Articles (Part IV) from the 2019 MCM PDF,
chunks by article boundaries, and writes metadata-rich JSONL to
data/processed/mcm_punitive_chunks.jsonl

Usage:
    python src/ingest_mcm.py --pdf /path/to/2019_MCM.pdf

Output:
    data/processed/mcm_punitive_chunks.jsonl
"""

import re
import json
import argparse
import hashlib
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------

def extract_pages_pypdf(pdf_path: Path, start_page: int, end_page: int) -> list[dict]:
    """Return list of {page_num, text} dicts (1-indexed page numbers)."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for i in range(start_page - 1, min(end_page, len(reader.pages))):
        text = reader.pages[i].extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})
    return pages


def extract_pages_unstructured(pdf_path: Path, start_page: int, end_page: int) -> list[dict]:
    """Fallback: use unstructured for richer extraction."""
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(
            filename=str(pdf_path),
            include_page_breaks=True,
            strategy="fast",
        )
        # group by page
        page_texts: dict[int, list[str]] = {}
        current_page = start_page
        for el in elements:
            if hasattr(el, "metadata") and el.metadata.page_number:
                current_page = el.metadata.page_number
            if start_page <= current_page <= end_page:
                page_texts.setdefault(current_page, []).append(str(el))
        return [
            {"page_num": pn, "text": "\n".join(txts)}
            for pn, txts in sorted(page_texts.items())
        ]
    except Exception as exc:
        raise RuntimeError(f"unstructured fallback also failed: {exc}")


def extract_part_iv(pdf_path: Path) -> tuple[list[dict], int, int]:
    """
    Auto-detect Part IV / Punitive Articles page range then extract.
    Returns (pages, start_page, end_page).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    # Locate Part IV start: first page with "PUNITIVE ARTICLES" + "Part IV"
    start_page = None
    for i in range(200, min(total, 500)):
        text = reader.pages[i].extract_text() or ""
        if "PUNITIVE ARTICLES" in text.upper() and re.search(r"Part\s+IV", text, re.IGNORECASE):
            start_page = i + 1  # 1-indexed
            break

    if start_page is None:
        # fallback: known offset for 2019 MCM
        start_page = 308
        print(f"[WARN] Could not auto-detect Part IV start; using default page {start_page}")

    # Locate end: last page with "IV-" prefix before another part begins
    end_page = start_page
    for i in range(start_page - 1, min(total, start_page + 250)):
        text = reader.pages[i].extract_text() or ""
        stripped = text.strip()
        if stripped.startswith("IV-"):
            end_page = i + 1
        elif i > start_page + 10 and re.match(r"^V-\d|PART\s+V", stripped):
            break  # hit Part V

    print(f"[INFO] Punitive Articles: PDF pages {start_page}–{end_page} ({end_page - start_page + 1} pages)")

    try:
        pages = extract_pages_pypdf(pdf_path, start_page, end_page)
    except Exception as exc:
        print(f"[WARN] pypdf failed ({exc}), trying unstructured …")
        pages = extract_pages_unstructured(pdf_path, start_page, end_page)

    return pages, start_page, end_page


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

# Matches headings like:
#   "77. Article 77—Murder"
#   "Article 77."
#   "Art. 92"  (sometimes appears mid-text)
#   "1. Article 77" (paragraph number style used in MCM Part IV)
ARTICLE_HEADING_RE = re.compile(
    r"""
    (?:^|\n)                          # start of line
    (?:
        (\d{1,3})\.\s+                 # paragraph number (e.g. "1. ")
        Article\s+(\d{1,3})           # followed by Article NNN
        [—\-–\.]?\s*                  # separator
        ([^\n]{0,80})                  # title (rest of line)
      |
        Article\s+(\d{1,3})           # OR just "Article NNN"
        [—\-–\.]?\s*
        ([^\n]{0,80})
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_article_number(match: re.Match) -> tuple[Optional[str], Optional[str]]:
    """Extract (article_number_str, article_title) from a regex match."""
    # groups: (para_num, art_num_1, title_1, art_num_2, title_2)
    g = match.groups()
    art_num = g[1] or g[3]
    title = (g[2] or g[4] or "").strip().strip("—-–.")
    return art_num, title


def split_by_articles(pages: list[dict]) -> list[dict]:
    """
    Concatenate all page text, then split on article headings.
    Returns list of article dicts: {article_number, article_title, text, page_start, page_end}
    """
    # Build a combined text with page-break markers so we can recover page numbers
    PAGE_MARKER = "\n<<<PAGE:{page_num}>>>\n"
    combined = ""
    page_positions: list[tuple[int, int]] = []  # (char_offset, page_num)
    for p in pages:
        page_positions.append((len(combined), p["page_num"]))
        combined += PAGE_MARKER.format(page_num=p["page_num"])
        combined += p["text"]

    def char_to_page(char_idx: int) -> int:
        """Return page number for a character offset."""
        page = pages[0]["page_num"]
        for offset, pnum in page_positions:
            if offset <= char_idx:
                page = pnum
            else:
                break
        return page

    # Find all article heading positions
    splits = []
    for m in ARTICLE_HEADING_RE.finditer(combined):
        art_num, title = parse_article_number(m)
        if art_num and int(art_num) >= 77:  # UCMJ punitive articles start at 77
            splits.append((m.start(), art_num, title))

    if not splits:
        # No article boundaries found—treat whole text as one chunk
        return [
            {
                "article_number": None,
                "article_title": "Punitive Articles",
                "text": combined,
                "page_start": pages[0]["page_num"],
                "page_end": pages[-1]["page_num"],
            }
        ]

    articles = []
    for idx, (char_start, art_num, title) in enumerate(splits):
        char_end = splits[idx + 1][0] if idx + 1 < len(splits) else len(combined)
        text = combined[char_start:char_end]
        # Remove page markers from text
        text = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()
        articles.append(
            {
                "article_number": art_num,
                "article_title": title,
                "text": text,
                "page_start": char_to_page(char_start),
                "page_end": char_to_page(char_end - 1),
            }
        )

    return articles


# ---------------------------------------------------------------------------
# Token-based chunking within an article
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    """Estimate token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def chunk_article(
    article: dict,
    chunk_size_tokens: int = 600,
    overlap_tokens: int = 100,
    source: str = "MCM 2019",
    doc_path: str = "",
    global_offset: int = 0,
) -> list[dict]:
    """
    Split a single article's text into overlapping token-sized chunks.
    Tries to split on paragraph/sentence boundaries first.
    """
    text = article["text"]
    if not text.strip():
        return []

    # Split into sentences/paragraphs (keep legal structure together)
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if rough_token_count(current + "\n\n" + para) <= chunk_size_tokens:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph > chunk_size, split by sentences
            if rough_token_count(para) > chunk_size_tokens:
                sentences = re.split(r"(?<=[.;])\s+", para)
                sub = ""
                for sent in sentences:
                    if rough_token_count(sub + " " + sent) <= chunk_size_tokens:
                        sub = (sub + " " + sent).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = sent
                if sub:
                    current = sub
                else:
                    current = ""
            else:
                current = para.strip()

    if current:
        chunks.append(current)

    art_num = article.get("article_number") or "unknown"
    art_title = article.get("article_title") or ""
    # Clean up any residual page markers from article title
    art_title = re.sub(r"<<<PAGE:\d+>>>", "", art_title).strip()

    # Apply overlap: prepend tail of previous chunk
    overlap_chars = overlap_tokens * 4
    result = []
    local_idx = 0  # counts only chunks we actually keep (skipping near-empty ones)
    for i, chunk_text in enumerate(chunks):
        if i > 0 and overlap_chars > 0:
            prev = chunks[i - 1]
            overlap_prefix = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            chunk_text = overlap_prefix + "\n\n[...]\n\n" + chunk_text

        # Remove any residual page markers from text
        chunk_text = re.sub(r"<<<PAGE:\d+>>>", "", chunk_text).strip()

        # Skip near-empty chunks (page-header remnants like "Article 92\nIV-28")
        if rough_token_count(chunk_text) < 20:
            continue

        # global_offset ensures chunk_ids are unique even when the same article
        # number appears in multiple boundary-split segments (e.g., Art. 92 appears
        # across many MCM pages and gets split into many article objects).
        global_pos = global_offset + local_idx
        chunk_id = f"mcm2019_{art_num.zfill(3)}_{global_pos:03d}"
        local_idx += 1

        # deterministic hash for dedup
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

        result.append(
            {
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "source": source,
                "doc_path": doc_path,
                "section": "Punitive Articles",
                "article_number": art_num,
                "article_title": art_title,
                "page_start": article["page_start"],
                "page_end": article["page_end"],
                "heading_path": ["Part IV", "Punitive Articles", f"Article {art_num}"],
                "chunk_index": local_idx - 1,
                "total_chunks_in_article": 0,  # patched below
                "token_estimate": rough_token_count(chunk_text),
                "text": chunk_text,
            }
        )

    # patch total_chunks_in_article now that we know the real count
    for r in result:
        r["total_chunks_in_article"] = len(result)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest MCM 2019 Punitive Articles")
    parser.add_argument("--pdf", type=Path, default=Path("/mnt/user-data/uploads/2019_MCM__Final___20190108_.pdf"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/mcm_punitive_chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=600, help="Target chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap in tokens")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading PDF: {args.pdf}")
    pages, start_pg, end_pg = extract_part_iv(args.pdf)
    print(f"[INFO] Extracted {len(pages)} pages of raw text")

    print("[INFO] Splitting by article boundaries …")
    articles = split_by_articles(pages)
    print(f"[INFO] Found {len(articles)} articles/sections")

    all_chunks = []
    global_offset = 0
    for article in articles:
        chunks = chunk_article(
            article,
            chunk_size_tokens=args.chunk_size,
            overlap_tokens=args.overlap,
            source="MCM 2019",
            doc_path=str(args.pdf),
            global_offset=global_offset,
        )
        global_offset += len(chunks)
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks: {len(all_chunks)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK] Chunks written to {args.out}")

    # Quick stats
    arts = {c["article_number"] for c in all_chunks if c["article_number"]}
    print(f"[STATS] Articles indexed: {sorted(arts, key=lambda x: int(x) if x.isdigit() else 0)[:10]} … ({len(arts)} total)")
    tokens = [c["token_estimate"] for c in all_chunks]
    print(f"[STATS] Token estimates — min:{min(tokens)} avg:{sum(tokens)//len(tokens)} max:{max(tokens)}")


if __name__ == "__main__":
    main()
