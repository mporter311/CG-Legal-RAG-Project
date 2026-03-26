"""
ingest_separations_manual.py — Ingest the CG Military Separations Manual
                                (COMDTINST 1000.4C) into pio-rag chunks.

Document structure
------------------
- 209 pages total
- Pages 1–11:  cover page, blank pages, instruction header, TOC  → skipped
- Pages 12–209: 10 chapters of content

Chapter layout (detected via internal page numbers N-M):
  Ch. 1  pp.  12–52   Commissioned Officers         (~15,938 words)
  Ch. 2  pp.  53–126  Enlisted Members              (~29,065 words)  ← largest
  Ch. 3  pp. 127–157  Retirements                   (~12,409 words)
  Ch. 4  pp. 158–160  Dependency or Hardship        (~ 1,037 words)
  Ch. 5  pp. 161–169  Disability Retirement          (~ 3,703 words)
  Ch. 6  pp. 170–177  Enlisted High Year Tenure      (~ 2,632 words)
  Ch. 7  pp. 178–183  Senior Enlisted Continuation   (~ 1,890 words)
  Ch. 8  pp. 184–185  Selective Early Retirement     (~   610 words)
  Ch. 9  pp. 186–189  Reserve Component Managers     (~ 1,305 words)
  Ch.10  pp. 190–209  Reserve Separation/Retirement  (~ 8,340 words)

Section heading format
----------------------
  A.  B.  C. … Z.  AA.  BB. … PP.   (single and double uppercase letters)
  Each followed by a title on the same line.

Sub-item format (body text)
---------------------------
  1. 2. 3.    numbered paragraphs
  a. b. c.    lettered sub-items
  (1) (2)     numbered sub-sub-items   [rare]
  (a) (b)     lettered sub-sub-items   [rare]

Critical PDF extraction issues
-------------------------------
  1. Mid-word newlines are SEVERE (~1 per 16 words). pypdf inserts \n
     inside words when the PDF renderer used column-based text placement.
     Examples: "T\nhe" → "The",  "C\nommanding" → "Commanding"
     Fix: re.sub(r'([A-Z])\n([a-z])', ...) + single-char line merging.
  2. Running header "COMDTINST 1000.4C" on every page → stripped.
  3. Internal page numbers "N-M" on their own line → stripped.
  4. Single-newline-only layout (no double newlines) → same as MCM/Conduct
     Manual; normalize_paragraphs() handles this.

Chunking strategy
-----------------
  - Split on section letter headings (A., B., AA., etc.) as boundaries.
  - Use internal page numbering to detect chapter boundaries reliably
    (more robust than regex-matching chapter headings, which have split-word
    artifacts like "CH\nAPTER").
  - Group small adjacent sections up to CHUNK_SIZE_TOKENS.
  - Large sections split by paragraph using normalize_paragraphs().
  - Heading path format:
      ["Chapter 1", "Commissioned Officers", "A.", "General"]

Output
------
  data/processed/separations_manual_chunks.jsonl
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

SOURCE_LABEL    = "CG Military Separations Manual"
DOC_CITATION    = "COMDTINST 1000.4C"
CHUNK_ID_PREFIX = "cgsep"

SKIP_PAGES_1INDEXED = set(range(1, 12))   # cover + blanks + header + TOC

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 80

# Chapter definitions keyed by chapter number string.
# page ranges determined by internal "N-M" page-number markers.
CHAPTERS = {
    "1":  (12,  52,  "Commissioned Officers"),
    "2":  (53,  126, "Enlisted Members"),
    "3":  (127, 157, "Retirements"),
    "4":  (158, 160, "Dependency or Hardship Discharges"),
    "5":  (161, 169, "Disability Retirement and Severance Procedures"),
    "6":  (170, 177, "Enlisted High Year Tenure"),
    "7":  (178, 183, "Senior Enlisted Continuation Board"),
    "8":  (184, 185, "Selective Early Retirement Board"),
    "9":  (186, 189, "Reserve Component Managers"),
    "10": (190, 209, "Reserve Separation, Retirement, and Transfer"),
}

# ---------------------------------------------------------------------------
# Section heading regex
# ---------------------------------------------------------------------------
# Matches:  "A. General."   "AA. Procedures to Effect Transfer"
# Must be at the start of a line after text cleaning.
SECTION_RE = re.compile(
    r"(?:^|\n)"             # line start
    r"([A-Z]{1,2})"         # section letter(s): A–Z or AA–PP
    r"\.\s+"                # period + whitespace
    r"([A-Z\d][^\n]{2,})",  # title: starts with capital/digit, at least 3 chars
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def clean_page_text(text: str) -> str:
    """
    Fix all PDF extraction artifacts in the Separations Manual.

    Six distinct artifact types, fixed in strict order:

      1. Split-line page numbers "N-\n M\n" → removed
      2. Single-line page numbers "N-M" → removed
      3. Trailing whitespace on lines ("RCM \n" → "RCM\n")
      4. Leading-space continuation lines ("RCM\n captains" → "RCM captains")
      5. Single-letter split at end of line ("members m\n ust" → "members must")
      6. Capital\nlower mid-word splits ("T\nhe" → "The") — safe version only
      7. Isolated lowercase letter on its own line
      8. Space-padded capital splits ("G eneral" → "General")
      9. Space before period ("Harassment ." → "Harassment.")
    """
    # 1. Strip running header
    text = re.sub(r"COMDTINST\s+1000\.4C\s*\n?", "", text)
    # 2. Strip split-line page numbers: "N-\nM\n"
    text = re.sub(r"(\d{1,2})-\n(\d{1,3})\n", "\n", text)
    # 3. Strip single-line page numbers
    text = re.sub(r"(?:^|\n)\d{1,2}-\d{1,3}(?:\n|$)", "\n", text, flags=re.MULTILINE)
    # 4. Strip trailing whitespace from lines (fixes "RCM \n" → "RCM\n")
    text = "\n".join(l.rstrip() for l in text.split("\n"))
    # 5. Fix leading-space continuation lines
    text = re.sub(r"(?<=[A-Za-z])\n ([a-z])", r" \1", text)
    # 6. Fix single trailing letter split: "word m\nust" → "word must"
    #    Excludes 'a' and 'i' which are legitimate English words, not split letters.
    #    "submit a\nnarrative" must NOT become "anarrative".
    text = re.sub(r"(?<=\s)([b-hj-z])\n([a-z])", r"\1\2", text)
    # 7. Fix Capital\nlower (safe: only when capital preceded by space/punct)
    text = re.sub(r"(?<=[\s\n\.\(\[])([A-Z])\n([a-z])", r"\1\2", text)
    # 8. Fix single isolated lowercase letter on its own line
    text = re.sub(r"\n([a-z])\n", r"\1", text)
    # 9. Fix space-padded capital splits: "G eneral" → "General"
    text = re.sub(r"(?<=[\s\n\.\)])([A-Z]) ([a-z]{2,})", r"\1\2", text)
    # 10. Fix space before period: "Harassment ." → "Harassment."
    text = re.sub(r"([a-zA-Z]) \.", r"\1.", text)
    # 11. Collapse multiple blank lines
    text = re.sub(r"\n{2,}", "\n", text)
    # 12. Normalise multiple spaces
    text = re.sub(r"  +", " ", text)
    return text


def clean_section_title(raw_title: str) -> str:
    """
    Extract only the meaningful title from a captured section heading,
    discarding body text that the regex captured beyond the true title.

    Handles two problems unique to this PDF:
      1. Space-padded splits: "G eneral" → "General"
      2. Over-capture:  "Types of Discharge. This Section discusses..." →
                        "Types of Discharge"
    """
    # Fix space-padded splits (lookbehind needs context; handle start separately)
    title = re.sub(r"(?<=[\s\n\.\)])([A-Z]) ([a-z]{2,})", r"\1\2", raw_title)
    title = re.sub(r"^([A-Z]) ([a-z]{2,})", r"\1\2", title)
    # Truncate at first sentence boundary (". Capital") after char 5
    m = re.search(r"\.\s{1,2}[A-Z][a-z]", title)
    if m and m.start() > 5:
        title = title[:m.start() + 1]
    # Strip trailing punctuation noise
    title = title.rstrip(". ").strip()
    return title


def normalize_paragraphs(text: str) -> list[str]:
    """
    Split single-newline text into logical paragraphs using structural cues.
    Handles the Separations Manual's outline format:
      numbered paragraphs: 1. 2. 3.
      lettered sub-items:  a. b. c.
      parenthetical:       (1) (a)
      section headings:    A. AA.
    """
    PARA_START = re.compile(
        r"""
        (?:^|\n)
        (?:
            [A-Z]{1,2}\.\s+[A-Z\d]    |   # section heading:  A. General
            \d{1,2}\.\s+[A-Z]         |   # numbered para:    1. Policy
            [a-z]\.\s+[A-Z\dT]        |   # lettered sub:     a. The...
            \(\d{1,2}\)\s             |   # paren-numbered:   (1) ...
            \([a-z]\)\s               |   # paren-lettered:   (a) ...
            [A-Z]{3,}                     # ALL-CAPS header
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
    """Extract and clean pages start_1..end_1 (1-indexed, inclusive)."""
    reader = PdfReader(str(pdf_path))
    pages  = []
    for i in range(start_1 - 1, min(end_1, len(reader.pages))):
        page_num = i + 1
        if page_num in SKIP_PAGES_1INDEXED:
            continue
        raw  = reader.pages[i].extract_text() or ""
        text = clean_page_text(raw)
        pages.append({"page_num": page_num, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def split_into_sections(pages: list[dict], chapter_num: str,
                        chapter_title: str) -> list[dict]:
    """
    Build combined text with page markers, then split on section headings.
    Deduplicates section letters (keeps first/longest occurrence).
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

    # Collect all section heading matches
    raw_splits: list[tuple[int, str, str]] = []
    for m in SECTION_RE.finditer(combined):
        letter = m.group(1).strip()
        title  = m.group(2).strip()
        raw_splits.append((m.start(), letter, title))

    if not raw_splits:
        # Fallback: treat entire chapter as one section
        text = re.sub(r"<<<PAGE:\d+>>>", "", combined).strip()
        return [{
            "section_letter": chapter_num,
            "section_title":  chapter_title,
            "chapter_num":    chapter_num,
            "chapter_title":  chapter_title,
            "text":           text,
            "page_start":     page_positions[0][1] if page_positions else 0,
            "page_end":       page_positions[-1][1] if page_positions else 0,
        }]

    # Deduplicate: keep first occurrence of each section letter
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
            "section_title":  clean_section_title(title),
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
    Merge small adjacent sections up to CHUNK_SIZE_TOKENS.
    Split oversized sections by paragraph.
    Apply overlap between consecutive chunks.
    """
    chunks: list[dict] = []
    global_idx  = 0
    buf_text    = ""
    buf_secs:   list[dict] = []

    def flush():
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
            paras  = normalize_paragraphs(sec_text)
            sub    = ""
            for para in paras:
                cand = (sub + "\n" + para).strip() if sub else para.strip()
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

    # Apply overlap
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
    page_start = first["page_start"]
    page_end   = last["page_end"]

    chunk_id     = f"{CHUNK_ID_PREFIX}_ch{ch_num.zfill(2)}_{idx:04d}"
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    heading_path = [f"Chapter {ch_num}", ch_title, sec_letter + ".", sec_title]

    out.append({
        "chunk_id":        chunk_id,
        "content_hash":    content_hash,
        "source":          f"{SOURCE_LABEL} ({DOC_CITATION})",
        "doc_path":        doc_path,
        "section":         f"Chapter {ch_num} — {ch_title}",
        "section_number":  sec_letter + ".",
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
# Main pipeline
# ---------------------------------------------------------------------------

def ingest(pdf_path: Path, output_path: Path) -> None:
    print(f"[INFO] Source  : {pdf_path.name}")
    print(f"[INFO] Citation: {DOC_CITATION}")
    print(f"[INFO] Skipping pages 1–11 (cover/TOC)")

    all_chunks: list[dict] = []

    for ch_num, (start_p, end_p, ch_title) in CHAPTERS.items():
        print(f"\n[INFO] Chapter {ch_num:>2}: {ch_title}  (pp.{start_p}–{end_p})")

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
        description="Ingest CG Military Separations Manual (COMDTINST 1000.4C)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/Separations_Manual.pdf"),
        help="Path to PDF (default: data/raw/Separations_Manual.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/separations_manual_chunks.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf, args.output)


if __name__ == "__main__":
    main()
