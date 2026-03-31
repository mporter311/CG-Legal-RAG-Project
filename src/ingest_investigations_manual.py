"""
ingest_investigations_manual.py  --  Ingest the CG Administrative Investigations Manual
                                      (COMDTINST M5830.1A) into pio-rag chunks.

Document structure
------------------
- 221 pages total
- Pages 1-17:  cover page, instruction header, TOC  -> skipped
- Pages 18-221: 11 chapters of content

Chapter layout (detected via CHAPTER heading + internal page markers):
  Ch. 1  pp.  18-30   General Policies                    (~5,846 words)
  Ch. 2  pp.  31-37   When Investigations Are Required     (~2,379 words)
  Ch. 3  pp.  38-58   Convening Administrative Investigations (~7,143 words)
  Ch. 4  pp.  59-80   Conducting Standard Investigations   (~8,729 words)
  Ch. 5  pp.  81-92   Preparing Investigative Reports      (~4,670 words)
  Ch. 6  pp.  93-104  Forwarding, Review, and Action       (~5,175 words)
  Ch. 7  pp. 105-153  Investigation of Disease/Injury/Death (~17,021 words)
  Ch. 8  pp. 154-166  Conducting a Formal Investigation    (~4,291 words)
  Ch. 9  pp. 167-195  Courts of Inquiry                   (~11,723 words)
  Ch.10  pp. 196-203  Parties and Witnesses                (~3,905 words)
  Ch.11  pp. 204-221  Requirements for Specific Investigations (~5,918 words)

Section heading format
----------------------
  ALL-CAPS letter + period + 2+ spaces + title, e.g.:
    "A.  PURPOSE AND SCOPE"
    "G.  COORDINATION WITH OTHER ORGANIZATIONS AND INFORMATION"
  Some chapters reuse letters (A, B, C...) - deduplication handled per-chapter.

PDF extraction characteristics
-------------------------------
  - CLEAN single-column layout. pypdf extracts with ~75-char lines,
    single newlines, NO mid-word split problem. No special cleaning needed.
  - Running header "COMDTINST M5830.1A" on every page -> stripped.
  - Internal page numbers "N-NN" on their own line -> stripped.
  - Many EXHIBIT pages (sample forms, flowcharts, scripts): ~30 pages
    throughout the document. These are low-retrieval-value procedural
    templates, not policy content.
    Detection rule: pages with "EXHIBIT" keyword AND no section heading
    AND < 400 words are flagged as exhibits and excluded.
    Pages WITH a section heading are ALWAYS kept (avoids false exclusions).

Chunking strategy
-----------------
  - Split on ALL-CAPS section headings (A.  TITLE) as boundaries.
  - Each chapter resets the letter sequence, so sections are keyed as
    "Ch.N - Letter" (e.g. "7-G") to avoid confusion across chapters.
  - Group small adjacent sections up to CHUNK_SIZE_TOKENS.
  - Split oversized sections by paragraph (double-newline breaks,
    which exist in this clean PDF).
  - Heading path: ["Chapter N", "Title", "A.", "Section Title"]

Output
------
  data/processed/investigations_manual_chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("[ERROR] pypdf not installed.  Run:  pip install pypdf")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_LABEL    = "CG Administrative Investigations Manual"
DOC_CITATION    = "COMDTINST M5830.1A"
CHUNK_ID_PREFIX = "cgim"

SKIP_PAGES_1INDEXED = set(range(1, 18))  # cover + header + TOC

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 80

CHAPTERS = {
    "1":  (18,  30,  "General Policies for Administrative Investigations"),
    "2":  (31,  37,  "When Investigations Are Required"),
    "3":  (38,  58,  "Convening Administrative Investigations"),
    "4":  (59,  80,  "Conducting Standard Investigations"),
    "5":  (81,  92,  "Preparing Investigative Reports"),
    "6":  (93,  104, "Forwarding, Review, and Action on Investigative Reports"),
    "7":  (105, 153, "Investigation of Disease, Injury, or Death"),
    "8":  (154, 166, "Conducting a Formal Investigation"),
    "9":  (167, 195, "Courts of Inquiry"),
    "10": (196, 203, "Parties and Witnesses"),
    "11": (204, 221, "Requirements for Specific Investigations"),
}

# ALL-CAPS section heading: "A.  TITLE" or "G.  TITLE"
# Requires 2+ spaces after the period to distinguish from "1.  Item." sub-items
SECTION_RE = re.compile(
    r"(?:^|\n)"             # line start
    r"([A-Z])\."            # single capital letter + period
    r"\s+"                  # 1+ spaces (some headings use only 1 space)
    r"([A-Z][A-Z\s\(\)/,\-\.]{2,80})"  # ALL-CAPS title
    r"(?:\n|$)"
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def is_exhibit_page(text: str, lines: list[str]) -> bool:
    """
    Return True if a page is an exhibit/form/sample template that should
    be excluded from the corpus.

    Rule: page has "EXHIBIT" keyword AND has no section heading AND < 400 words.
    Pages WITH a section heading are always kept even if they mention exhibits.
    """
    has_section = any(
        re.match(r'^[A-Z]\.\s{2,}[A-Z]', l) for l in lines
    )
    if has_section:
        return False

    has_exhibit = bool(re.search(r'\bEXHIBIT\b|Exhibit\s+[\(\d]', text, re.IGNORECASE))
    word_count  = len(text.split())
    return has_exhibit and word_count < 400


def clean_page_text(text: str) -> str:
    """
    Strip boilerplate from a single page.

    This PDF is clean (no split-word problem), so cleaning is minimal:
      1. Strip running header "COMDTINST M5830.1A"
      2. Strip internal page numbers "N-NN" on their own line
      3. Collapse multiple blank lines
    """
    # Strip running header
    text = re.sub(r"COMDTINST\s+M5830\.1A\s*\n?", "", text)
    # Strip internal page numbers (e.g. "1-1", "7-32") on their own line
    text = re.sub(r"(?:^|\n)\d{1,2}-\d{1,3}(?:\n|$)", "\n", text, flags=re.MULTILINE)
    # Collapse multiple blank lines to single
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_paragraphs(text: str) -> list[str]:
    """
    Split text into logical paragraphs.

    This PDF has genuine double newlines between paragraphs (unlike the
    2024 MCM and Conduct Manual), so simple double-newline splitting works
    well. For single-newline blocks, structural markers are used.
    """
    # Double-newline split first (works in this clean PDF)
    blocks = re.split(r"\n{2,}", text)

    PARA_START = re.compile(
        r"""
        (?:^|\n)
        (?:
            [A-Z]\.\s{2,}[A-Z]    |  # section heading: A.  TITLE
            \d{1,2}\.\s+[A-Z]     |  # numbered para:   1.  Text
            [a-z]\.\s+[A-Z]       |  # lettered sub:    a.  Text
            \(\d{1,2}\)\s         |  # paren-numbered:  (1) Text
            \([a-z]\)\s           |  # paren-lettered:  (a) Text
            NOTE:\s               |  # NOTE: ...
            [A-Z]{3,}\s               # ALL-CAPS label
        )
        """,
        re.VERBOSE,
    )

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
    """Extract, clean, and filter pages start_1..end_1 (1-indexed, inclusive)."""
    reader  = PdfReader(str(pdf_path))
    pages   = []
    skipped = 0

    for i in range(start_1 - 1, min(end_1, len(reader.pages))):
        page_num = i + 1
        if page_num in SKIP_PAGES_1INDEXED:
            continue
        raw   = reader.pages[i].extract_text() or ""
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        if is_exhibit_page(raw, lines):
            skipped += 1
            continue

        text = clean_page_text(raw)
        pages.append({"page_num": page_num, "text": text})

    if skipped:
        print(f"       (skipped {skipped} exhibit/form pages)")

    return pages


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def clean_section_title(raw_title: str) -> str:
    """
    Convert an ALL-CAPS pypdf section title to readable Title Case.

    Handles pypdf artifacts found in this document:
      1. Embedded newlines:         "INFORMATION \\nSHARING" -> "INFORMATION SHARING"
      2. Hyphen-split words:        "ONE-O FFICER" -> "ONE-OFFICER"
      3. Isolated single-letter fragments: "AN D AGGRESSIVE" -> "AND AGGRESSIVE"
         (skips A and I which are valid single-letter English words)
      4. Short non-word 2-char fragments:  "Fi Nal" -> "Final"
         (keeps known 2-char English words: As, Or, Of, At, To, In, etc.)
      5. Trailing period from regex over-capture: "INVESTIGATIONS ." -> "INVESTIGATIONS"

    Note: multi-char mid-word splits like "INVEST IGATION" or "GROUNDI NG" appear
    in 2 of 55 section titles and are accepted as minor heading-display imperfections.
    They do not affect the retrieval text content.
    """
    _KEEP_SEPARATE = {
        'As', 'Or', 'Of', 'At', 'To', 'In', 'An', 'By', 'My', 'Is',
        'If', 'On', 'Up', 'Do', 'So', 'No', 'He', 'We', 'Be', 'It',
        'Am', 'Us', 'Me', 'Vs', 'Re', 'Ex',
    }
    title = raw_title.strip()
    # 1. Strip embedded newlines
    title = re.sub(r'\s*\n\s*', ' ', title)
    # 2. Fix hyphen-split in ALL-CAPS: "ONE-O FFICER" -> "ONE-OFFICER"
    title = re.sub(r'-([A-Z]) ([A-Z])', r'-\1\2', title)
    # 3. Fix isolated single-letter fragment (not A/I): "AN D WORD" -> "AND WORD"
    title = re.sub(r'([A-Z]{2,}) ([B-HJ-Z]) ([A-Z]{2,})', r'\1\2 \3', title)
    # 4. Strip trailing period artifact
    title = re.sub(r'\s*\.$', '', title)
    # 5. Convert to Title Case
    title = title.title()
    # 6. Rejoin 2-char non-word Title fragments: "Fi Nal" -> "Final"
    def _rejoin(m: re.Match) -> str:
        frag, rest = m.group(1), m.group(2)
        if frag in _KEEP_SEPARATE:
            return m.group(0)
        return frag + rest[0].lower() + rest[1:]
    title = re.sub(r'\b([A-Z][a-z]) ([A-Z][a-z]+)', _rejoin, title)
    return title.strip()




def split_into_sections(pages: list[dict], chapter_num: str,
                        chapter_title: str) -> list[dict]:
    """
    Concatenate pages with position markers, split on section headings.
    Deduplicates by keeping the first occurrence of each section letter
    (some letters repeat in different chapters but each chapter is processed
    independently so there's no cross-chapter confusion).
    """
    PAGE_MARKER = "\n<<<PAGE:{pnum}>>>\n"
    combined    = ""
    page_positions: list[tuple[int, int]] = []

    for p in pages:
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

    # Find all section heading matches
    raw_splits: list[tuple[int, str, str]] = []
    for m in SECTION_RE.finditer(combined):
        letter = m.group(1).strip()
        title  = clean_section_title(m.group(2))  # Convert ALL-CAPS to Title Case
        raw_splits.append((m.start(), letter, title))

    if not raw_splits:
        text = re.sub(r"<<<PAGE:\d+>>>", "", combined).strip()
        return [{
            "section_letter": "A",
            "section_title":  chapter_title,
            "chapter_num":    chapter_num,
            "chapter_title":  chapter_title,
            "text":           text,
            "page_start":     page_positions[0][1] if page_positions else 0,
            "page_end":       page_positions[-1][1] if page_positions else 0,
        }]

    # Deduplicate: keep first occurrence of each letter within this chapter
    seen: set[str] = set()
    splits: list[tuple[int, str, str]] = []
    for entry in raw_splits:
        if entry[1] not in seen:
            seen.add(entry[1])
            splits.append(entry)

    sections = []
    for idx, (char_start, letter, title) in enumerate(splits):
        char_end   = splits[idx + 1][0] if idx + 1 < len(splits) else len(combined)
        text       = combined[char_start:char_end]
        text       = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()
        page_start = char_to_page(char_start)
        page_end   = char_to_page(splits[idx + 1][0]) if idx + 1 < len(splits) \
                     else page_positions[-1][1]

        sections.append({
            "section_letter": letter,
            "section_title":  title[:100],
            "chapter_num":    chapter_num,
            "chapter_title":  chapter_title,
            "text":           text,
            "page_start":     page_start,
            "page_end":       page_end,
        })

    return sections


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_sections(sections: list[dict], source: str,
                   doc_path: str) -> list[dict]:
    """
    Merge small sections up to CHUNK_SIZE_TOKENS; split large ones by paragraph.
    Apply overlap between consecutive chunks.
    """
    chunks:   list[dict] = []
    global_idx  = 0
    buf_text    = ""
    buf_secs:   list[dict] = []

    def flush() -> None:
        nonlocal buf_text, buf_secs, global_idx
        if not buf_text.strip():
            return
        _emit(buf_text, buf_secs, source, doc_path, global_idx, chunks)
        global_idx += 1
        buf_text = ""
        buf_secs = []

    for sec in sections:
        sec_text   = sec["text"]
        sec_tokens = rough_token_count(sec_text)

        if sec_tokens > CHUNK_SIZE_TOKENS:
            flush()
            paras = normalize_paragraphs(sec_text)
            sub   = ""
            for para in paras:
                cand = (sub + "\n\n" + para).strip() if sub else para.strip()
                if rough_token_count(cand) <= CHUNK_SIZE_TOKENS:
                    sub = cand
                else:
                    if sub:
                        _emit(sub, [sec], source, doc_path, global_idx, chunks)
                        global_idx += 1
                    sub = para.strip()
            if sub:
                _emit(sub, [sec], source, doc_path, global_idx, chunks)
                global_idx += 1
        else:
            cand = (buf_text + "\n\n" + sec_text).strip() if buf_text else sec_text
            if rough_token_count(cand) <= CHUNK_SIZE_TOKENS:
                buf_text = cand
                buf_secs.append(sec)
            else:
                flush()
                buf_text = sec_text
                buf_secs = [sec]

    flush()

    # Apply overlap: prepend tail of previous chunk
    overlap_chars = OVERLAP_TOKENS * 4
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]["text"]
        pfx  = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        chunks[i]["text"]           = pfx + "\n\n" + chunks[i]["text"]
        chunks[i]["token_estimate"] = rough_token_count(chunks[i]["text"])

    return chunks


def _emit(text: str, sections: list[dict], source: str, doc_path: str,
          idx: int, out: list[dict]) -> None:
    if not text.strip() or rough_token_count(text) < 20:
        return

    first      = sections[0]
    last       = sections[-1]
    ch_num     = first["chapter_num"]
    ch_title   = first["chapter_title"]
    sec_letter = first["section_letter"]
    sec_title  = first["section_title"]

    chunk_id     = f"{CHUNK_ID_PREFIX}_ch{ch_num.zfill(2)}_{idx:04d}"
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    heading_path = [f"Chapter {ch_num}", ch_title, sec_letter + ".", sec_title]

    out.append({
        "chunk_id":        chunk_id,
        "content_hash":    content_hash,
        "source":          f"{SOURCE_LABEL} ({DOC_CITATION})",
        "doc_path":        doc_path,
        "section":         f"Chapter {ch_num} -- {ch_title}",
        "section_number":  sec_letter + ".",
        "section_title":   sec_title,
        "chapter_number":  ch_num,
        "chapter_title":   ch_title,
        "page_start":      first["page_start"],
        "page_end":        last["page_end"],
        "heading_path":    heading_path,
        "chunk_index":     idx,
        "token_estimate":  rough_token_count(text),
        "text":            text,
    })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest(pdf_path: Path, output_path: Path) -> None:
    print(f"[INFO] Source  : {pdf_path.name}")
    print(f"[INFO] Citation: {DOC_CITATION}")
    print(f"[INFO] Content pages: 18-221  (pages 1-17 skipped)")
    print(f"[INFO] Exhibit/form pages will be excluded automatically")

    all_chunks: list[dict] = []

    for ch_num, (start_p, end_p, ch_title) in CHAPTERS.items():
        print(f"\n[INFO] Chapter {ch_num:>2}: {ch_title[:50]}  (pp.{start_p}-{end_p})")

        pages    = extract_pages(pdf_path, start_p, end_p)
        sections = split_into_sections(pages, ch_num, ch_title)
        print(f"       Sections found : {len(sections)}")

        chunks = chunk_sections(sections, source=f"{SOURCE_LABEL} ({DOC_CITATION})",
                                doc_path=str(pdf_path))
        print(f"       Chunks produced: {len(chunks)}")

        if chunks:
            toks = [c["token_estimate"] for c in chunks]
            print(f"       Tokens  min={min(toks)}  avg={sum(toks)//len(toks)}"
                  f"  max={max(toks)}")

        all_chunks.extend(chunks)

    print(f"\n[INFO] Total chunks: {len(all_chunks)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK]  Chunks written to {output_path}")

    # Sample output
    print("\n[SAMPLE CHUNKS]")
    by_ch: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_ch.setdefault(c["chapter_number"], []).append(c)

    for ch_num in sorted(by_ch.keys(), key=int):
        clist  = by_ch[ch_num]
        sample = clist[len(clist) // 2]
        print(f"\n  Ch.{ch_num}  {sample['chunk_id']}  "
              f"pp.{sample['page_start']}-{sample['page_end']}  "
              f"{sample['token_estimate']} tok")
        print(f"  Path: {' > '.join(sample['heading_path'])}")
        print(f"  Text: {sample['text'][:110].replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest CG Administrative Investigations Manual (COMDTINST M5830.1A)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/Investigations_Manual.pdf"),
        help="Path to PDF (default: data/raw/Investigations_Manual.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/investigations_manual_chunks.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf, args.output)


if __name__ == "__main__":
    main()
