"""
ingest_conduct_manual.py — Ingest the CG Discipline & Conduct Manual
                            (COMDTINST M1600.2) into pio-rag chunks.

Document structure
------------------
- 163 pages total; pages 1–14 are change notices + TOC → skipped
- Content pages 15–163 split into 6 chapters
- Section hierarchy:  1.  →  1.A.  →  1.A.1.  →  1.A.1.a.
- Chapter 6 uses single-letter definitions:  A. Abuse.  B. CPO.  etc.
- PDF is single-column; pypdf returns ~62-char lines with only single
  newlines (same layout issue as 2024 MCM → reuse normalize_paragraphs)

Chunking strategy
-----------------
- Split on section headings as boundaries (not article numbers)
- Group small adjacent sections up to CHUNK_SIZE_TOKENS
- Preserve 4-level heading path in metadata for citation quality
- Skip exhibit/table pages (83–86) flagged as low-quality
- Chapter 6 definitions chunked one definition per chunk

Output
------
data/processed/conduct_manual_chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# pypdf import
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader
except ImportError:
    print("[ERROR] pypdf not installed.  Run:  pip install pypdf")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_LABEL   = "CG Conduct Manual"
DOC_CITATION   = "COMDTINST M1600.2"
CHUNK_ID_PREFIX = "cgcm"

# Pages to skip entirely (0-indexed internally; displayed as 1-indexed)
SKIP_PAGES_1INDEXED = set(range(1, 15))     # change notices + TOC
EXHIBIT_PAGES_1INDEXED = set(range(83, 87)) # confinement charts/tables

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 80     # slightly smaller than MCM — sections are shorter

# Chapter definitions: (start_page_1indexed, end_page_1indexed, number, title)
CHAPTERS = [
    (15,  100, "1", "Discipline"),
    (101, 137, "2", "Conduct"),
    (138, 147, "3", "Hazing and Bullying"),
    (148, 152, "4", "Possession of Firearms"),
    (153, 160, "5", "No Contact Orders and Military Protective Orders"),
    (161, 163, "6", "Definitions"),
]

# ---------------------------------------------------------------------------
# Section heading regex
# ---------------------------------------------------------------------------
# Matches headings like:
#   1.A.          1.A.1.          1.A.1.a.
#   2.B.2.        1.F.10.c.
# Also matches Chapter 6 definition headings:  "A. Abuse."  "B. CPO."
# (single uppercase letter followed by period and a word)

SECTION_RE = re.compile(
    r"""
    (?:^|\n)                        # line start
    (
        \d{1,2}                     # chapter number
        \.[A-Z]                     # major section letter
        (?:\.\d{1,2})?              # optional subsection number
        (?:\.[a-z])?                # optional sub-subsection letter
        \.                          # trailing period
    )
    \s*(.{0,120})                   # section title (rest of line)
    """,
    re.VERBOSE,
)

# Chapter 6 definition headings:  "A. Abuse means..."
DEFINITION_RE = re.compile(
    r"""
    (?:^|\n)
    ([A-Z])                         # single capital letter
    \.\s+
    ([A-Z][A-Za-z ]{2,40})          # definition term (starts with capital)
    [\.\:]                          # period or colon after term
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Utility: rough token estimate
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Utility: normalize paragraphs (reused from ingest_mcm approach)
# ---------------------------------------------------------------------------

def normalize_paragraphs(text: str) -> list[str]:
    """
    Split single-newline-only text into logical paragraphs using
    structural cues.  Works for both the MCM and the Conduct Manual
    since both are extracted by pypdf with no double newlines.
    """
    PARA_START = re.compile(
        r"""
        (?:^|\n)
        (?:
            \d{1,2}\.[A-Z](?:\.\d{1,2})?(?:\.[a-z])?\.  |  # section heading
            [A-Z]\.\s+[A-Z]                               |  # Ch6 definition
            \(\d{1,2}\)\s                                 |  # (1) numbered
            \([a-z]\)\s                                   |  # (a) lettered
            \[\d\]\s                                      |  # [1] bracketed
            Note:\s                                       |  # Note: ...
            [A-Z]{4,}                                        # ALL-CAPS header
        )
        """,
        re.VERBOSE,
    )

    blocks = re.split(r"\n{2,}", text)
    paragraphs: list[str] = []

    for block in blocks:
        lines = block.split("\n")
        if len(lines) <= 2:
            p = block.strip()
            if p:
                paragraphs.append(p)
            continue

        current: list[str] = []
        for line in lines:
            if current and PARA_START.match(line):
                p = "\n".join(current).strip()
                if p:
                    paragraphs.append(p)
                current = [line]
            else:
                current.append(line)
        if current:
            p = "\n".join(current).strip()
            if p:
                paragraphs.append(p)

    return [p for p in paragraphs if p.strip()]


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path, start_1: int, end_1: int) -> list[dict]:
    """Extract pages start_1..end_1 (1-indexed, inclusive)."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i in range(start_1 - 1, min(end_1, len(reader.pages))):
        page_num = i + 1
        if page_num in SKIP_PAGES_1INDEXED:
            continue
        text = reader.pages[i].extract_text() or ""
        # Strip the running header "COMDTINST M1600.2" that appears on every page
        text = re.sub(r"COMDTINST\s+M1600\.2\s*", "", text)
        # Strip leading whitespace/blank lines that pypdf adds
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)
        pages.append({
            "page_num": page_num,
            "text":     text,
            "is_exhibit": page_num in EXHIBIT_PAGES_1INDEXED,
        })
    return pages


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def split_into_sections(pages: list[dict], chapter_num: str) -> list[dict]:
    """
    Concatenate page texts (with page-position markers), then split on
    section headings.  Returns list of section dicts with metadata.

    Special handling for Chapter 6: splits on definition letters instead.
    """
    PAGE_MARKER = "\n<<<PAGE:{pnum}>>>\n"
    combined    = ""
    page_positions: list[tuple[int, int]] = []

    for p in pages:
        if p["is_exhibit"]:
            continue
        page_positions.append((len(combined), p["page_num"]))
        combined += PAGE_MARKER.format(pnum=p["page_num"])
        combined += p["text"]

    def char_to_page(idx: int) -> int:
        page = page_positions[0][1] if page_positions else 0
        for offset, pnum in page_positions:
            if offset <= idx:
                page = pnum
            else:
                break
        return page

    # Chapter 6: definition-based splitting
    if chapter_num == "6":
        return _split_definitions(combined, char_to_page, chapter_num)

    # All other chapters: section-heading splitting
    splits: list[tuple[int, str, str]] = []
    for m in SECTION_RE.finditer(combined):
        sec_num   = m.group(1).strip()
        sec_title = m.group(2).strip()
        # Only keep sections belonging to this chapter
        if sec_num.startswith(chapter_num + "."):
            splits.append((m.start(), sec_num, sec_title))

    if not splits:
        # Fallback: return whole chapter as one section
        text = re.sub(r"<<<PAGE:\d+>>>", "", combined).strip()
        return [{
            "section_number": chapter_num,
            "section_title":  "",
            "chapter_num":    chapter_num,
            "text":           text,
            "page_start":     page_positions[0][1] if page_positions else 0,
            "page_end":       page_positions[-1][1] if page_positions else 0,
        }]

    # Deduplicate: keep first occurrence of each section number
    seen: set[str] = set()
    deduped: list[tuple[int, str, str]] = []
    for entry in splits:
        if entry[1] not in seen:
            seen.add(entry[1])
            deduped.append(entry)

    sections = []
    for idx, (char_start, sec_num, sec_title) in enumerate(deduped):
        char_end  = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(combined)
        text      = combined[char_start:char_end]
        text      = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()
        page_start = char_to_page(char_start)
        page_end   = char_to_page(deduped[idx + 1][0]) if idx + 1 < len(deduped) else page_positions[-1][1]

        sections.append({
            "section_number": sec_num,
            "section_title":  sec_title,
            "chapter_num":    chapter_num,
            "text":           text,
            "page_start":     page_start,
            "page_end":       page_end,
        })

    return sections


def _split_definitions(combined: str, char_to_page, chapter_num: str) -> list[dict]:
    """Split Chapter 6 on single-letter definition headings."""
    splits = []
    for m in DEFINITION_RE.finditer(combined):
        letter    = m.group(1)
        term      = m.group(2).strip()
        splits.append((m.start(), letter, term))

    if not splits:
        text = re.sub(r"<<<PAGE:\d+>>>", "", combined).strip()
        return [{
            "section_number": "6",
            "section_title":  "Definitions",
            "chapter_num":    "6",
            "text":           text,
            "page_start":     161,
            "page_end":       163,
        }]

    definitions = []
    for idx, (char_start, letter, term) in enumerate(splits):
        char_end   = splits[idx + 1][0] if idx + 1 < len(splits) else len(combined)
        text       = combined[char_start:char_end]
        text       = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()
        definitions.append({
            "section_number": f"6.{letter}",
            "section_title":  term,
            "chapter_num":    "6",
            "text":           text,
            "page_start":     char_to_page(char_start),
            "page_end":       char_to_page(char_end - 1),
        })

    return definitions


# ---------------------------------------------------------------------------
# Build heading path for metadata
# ---------------------------------------------------------------------------

CHAPTER_TITLES = {c[2]: c[3] for c in CHAPTERS}


def build_heading_path(chapter_num: str, chapter_title: str,
                       section_number: str, section_title: str) -> list[str]:
    """
    Return a 3- or 4-element heading path for display in citations.
    Example:  ["Chapter 1", "Discipline", "1.F.3.", "Pretrial Confinement"]
    """
    path = [f"Chapter {chapter_num}", chapter_title]
    if section_number and section_number != chapter_num:
        path.append(section_number)
    if section_title:
        # Truncate very long auto-extracted titles
        path.append(section_title[:80])
    return path


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_sections(
    sections: list[dict],
    source: str,
    doc_path: str,
) -> list[dict]:
    """
    Group sections into token-sized chunks with overlap.
    Small adjacent sections are merged; large sections are split by paragraph.
    Returns list of chunk dicts ready for JSONL output.
    """
    chunks: list[dict] = []
    global_idx = 0

    # Accumulate sections into groups up to CHUNK_SIZE_TOKENS
    buffer_text    = ""
    buffer_secs:   list[dict] = []

    def flush_buffer():
        nonlocal buffer_text, buffer_secs, global_idx
        if not buffer_text.strip():
            return
        _emit_chunk(buffer_text, buffer_secs, source, doc_path, global_idx, chunks)
        global_idx += 1
        buffer_text = ""
        buffer_secs = []

    for sec in sections:
        sec_text  = sec["text"]
        sec_tokens = rough_token_count(sec_text)

        if sec_tokens > CHUNK_SIZE_TOKENS:
            # Flush anything in the buffer first
            flush_buffer()
            # Split this oversized section by paragraph
            paras = normalize_paragraphs(sec_text)
            sub_buf = ""
            for para in paras:
                candidate = (sub_buf + "\n" + para).strip() if sub_buf else para.strip()
                if rough_token_count(candidate) <= CHUNK_SIZE_TOKENS:
                    sub_buf = candidate
                else:
                    if sub_buf:
                        _emit_chunk(sub_buf, [sec], source, doc_path, global_idx, chunks)
                        global_idx += 1
                    sub_buf = para.strip()
            if sub_buf:
                _emit_chunk(sub_buf, [sec], source, doc_path, global_idx, chunks)
                global_idx += 1
        else:
            # Try to merge into buffer
            candidate = (buffer_text + "\n\n" + sec_text).strip() if buffer_text else sec_text
            if rough_token_count(candidate) <= CHUNK_SIZE_TOKENS:
                buffer_text = candidate
                buffer_secs.append(sec)
            else:
                flush_buffer()
                buffer_text = sec_text
                buffer_secs = [sec]

    flush_buffer()

    # Apply overlap: prepend tail of previous chunk
    overlap_chars = OVERLAP_TOKENS * 4
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        prefix    = prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text
        chunks[i]["text"] = prefix + "\n\n" + chunks[i]["text"]
        chunks[i]["token_estimate"] = rough_token_count(chunks[i]["text"])

    return chunks


def _emit_chunk(
    text: str,
    sections: list[dict],
    source: str,
    doc_path: str,
    idx: int,
    out: list[dict],
) -> None:
    """Build one chunk dict and append to out."""
    if not text.strip() or rough_token_count(text) < 20:
        return

    # Use metadata from the FIRST section in the group
    first      = sections[0]
    last       = sections[-1]
    ch_num     = first["chapter_num"]
    ch_title   = CHAPTER_TITLES.get(ch_num, f"Chapter {ch_num}")
    sec_num    = first["section_number"]
    sec_title  = first["section_title"]
    page_start = first["page_start"]
    page_end   = last["page_end"]

    chunk_id     = f"{CHUNK_ID_PREFIX}_ch{ch_num.zfill(2)}_{idx:04d}"
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    heading_path = build_heading_path(ch_num, ch_title, sec_num, sec_title)

    out.append({
        "chunk_id":        chunk_id,
        "content_hash":    content_hash,
        "source":          source,
        "doc_path":        doc_path,
        "section":         f"Chapter {ch_num} — {ch_title}",
        "section_number":  sec_num,
        "section_title":   sec_title,
        "chapter_number":  ch_num,
        "chapter_title":   ch_title,
        "page_start":      page_start,
        "page_end":        page_end,
        "heading_path":    heading_path,
        "chunk_index":     idx,
        "token_estimate":  rough_token_count(text),
        "text":            text,
    })


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest(pdf_path: Path, output_path: Path) -> None:
    print(f"[INFO] Source  : {pdf_path.name}")
    print(f"[INFO] Citation: {DOC_CITATION}")
    print(f"[INFO] Content pages: 15–163  (pages 1–14 skipped)")

    all_chunks: list[dict] = []
    source_label = f"{SOURCE_LABEL} ({DOC_CITATION})"

    for start_p, end_p, ch_num, ch_title in CHAPTERS:
        print(f"\n[INFO] Chapter {ch_num}: {ch_title}  (pp.{start_p}–{end_p})")

        pages    = extract_pages(pdf_path, start_p, end_p)
        sections = split_into_sections(pages, ch_num)
        print(f"       Sections found : {len(sections)}")

        chunks = chunk_sections(sections, source=source_label, doc_path=str(pdf_path))
        print(f"       Chunks produced: {len(chunks)}")

        # Token stats
        if chunks:
            toks = [c["token_estimate"] for c in chunks]
            print(f"       Tokens  min={min(toks)}  avg={sum(toks)//len(toks)}  max={max(toks)}")

        all_chunks.extend(chunks)

    print(f"\n[INFO] Total chunks: {len(all_chunks)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK]  Chunks written to {output_path}")

    # Spot-check: show sample chunks from each chapter
    print("\n[SAMPLE CHUNKS]")
    by_chapter: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_chapter.setdefault(c["chapter_number"], []).append(c)

    for ch_num in sorted(by_chapter.keys()):
        ch_chunks = by_chapter[ch_num]
        sample    = ch_chunks[len(ch_chunks) // 2]  # middle chunk
        print(f"\n  Ch.{ch_num}  {sample['chunk_id']}  "
              f"pp.{sample['page_start']}-{sample['page_end']}  "
              f"{sample['token_estimate']} tok")
        print(f"  Heading: {' > '.join(sample['heading_path'])}")
        print(f"  Text:    {sample['text'][:120].replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest CG Discipline & Conduct Manual (COMDTINST M1600.2)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/CG_Conduct_Manual.pdf"),
        help="Path to the PDF (default: data/raw/CG_Conduct_Manual.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/conduct_manual_chunks.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf, args.output)


if __name__ == "__main__":
    main()
