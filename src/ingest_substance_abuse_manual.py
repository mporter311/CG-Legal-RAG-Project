"""
ingest_substance_abuse_manual.py  --  Ingest the CG Military Substance Abuse and
                                       Behavioral Addiction Program instruction.
                                       (COMDTINST 1000.10B, October 2022)

Document structure
------------------
- 61 pages total
- Pages 1-11:  cover, blank, instruction header (pp3-6), TOC (pp7-11) -> skipped
- Pages 12-61: 9 chapters of content

Chapter layout:
  Ch. 1  pp.  12-12   Overview                                             (~242 words)
  Ch. 2  pp.  13-14   Roles and Responsibilities                           (~593 words)
  Ch. 3  pp.  15-20   Alcohol Use Disorders                               (~1,977 words)
  Ch. 4  pp.  21-25   Driving Under the Influence                         (~1,785 words)
  Ch. 5  pp.  26-33   Drug Incidents                                      (~2,884 words)
  Ch. 6  pp.  34-52   Testing for Controlled and Prohibited Substances    (~6,826 words)
  Ch. 7  pp.  53-57   Prohibition on Hemp and Marijuana Establishments    (~1,632 words)
  Ch. 8  pp.  58-59   Prohibition on Possession of Drug Paraphernalia       (~483 words)
  Ch. 9  pp.  60-61   Behavioral Addictions                                 (~426 words)

Section heading format
----------------------
Two patterns, both start with a capital letter + period:
  1. Standalone:  "C. CO/OIC Responsibility."   (title ends the line)
  2. Inline body: "A. Objective. This chapter states..."  (title + period + body text)

In both cases the title ends at the FIRST period that is followed by whitespace or
end-of-line. The unified regex captures these correctly.

One multi-line heading exception:
  "E. Sample Adulteration, Substitution, Dilution and Refusal to Follow Contracted"
  "Guidelines"   <- wraps to next line
  Fixed by joining the continuation line before running the section regex.

PDF extraction characteristics
-------------------------------
- CLEAN single-column layout for most pages (~70-char lines, single newlines, NO
  split-word problem in the main text).
- Running header "COMDTINST 1000.10B" on every page -> stripped.
- Internal page numbers "N-N" on their own line -> stripped.
- Page 40 is a SCANNED image page with OCR artifacts. Known corrections applied:
    esco1i -> escort, pmpose -> purpose, trnstwo1ihy -> trustworthy,
    comi-maiiial -> court-martial, laborato1y -> laboratory, polial -> portal,
    uscg.Inil -> uscg.mil, adininistrative -> administrative, ai·e -> are,
    heai·ing -> hearing, procedmes -> procedures, Guai·d -> Guard,
    primaiy -> primary, ofDTOs -> of DTOs, maiiial -> martial
- Three pages (pp38-40) contain redaction blocks from EO 14151/14168.
  The redaction boilerplate text is stripped; surrounding policy content is preserved.
- Page 20 is a 30-word continuation page (end of Ch.3 section J); kept as content.

Chunking strategy
-----------------
- Split on lettered section headings as boundaries.
- Each section within a chapter is processed independently.
- Small sections grouped up to CHUNK_SIZE_TOKENS; large sections split by paragraph.
- Heading path: ["Chapter N", "Chapter Title", "A.", "Section Title"]

Output
------
  data/processed/substance_abuse_manual_chunks.jsonl
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

SOURCE_LABEL    = "CG Military Substance Abuse and Behavioral Addiction Program"
DOC_CITATION    = "COMDTINST 1000.10B"
CHUNK_ID_PREFIX = "cgsa"

SKIP_PAGES_1INDEXED = set(range(1, 12))  # cover + header + TOC (pp1-11)

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 80

CHAPTERS = {
    "1":  (12, 12,  "Overview"),
    "2":  (13, 14,  "Roles and Responsibilities"),
    "3":  (15, 20,  "Alcohol Use Disorders"),
    "4":  (21, 25,  "Driving Under the Influence"),
    "5":  (26, 33,  "Drug Incidents"),
    "6":  (34, 52,  "Testing for Controlled and Prohibited Substances"),
    "7":  (53, 57,  "Prohibition on Hemp and Marijuana Establishments"),
    "8":  (58, 59,  "Prohibition on Possession of Drug Paraphernalia"),
    "9":  (60, 61,  "Behavioral Addictions"),
}

# Section headings come in two forms in this document:
#
# Pattern A — period-terminated (most common):
#   "A. Title."                         (standalone, body on next line)
#   "A. Title. Body text continues..."  (title + body on same line)
#   Captured: title up to the FIRST period followed by space/EOL.
#
# Pattern B — no trailing period (only Ch.6 section E):
#   "E. Sample Adulteration, Substitution, Dilution and Refusal to Follow Contracted Guidelines"
#   This heading wraps across two lines in the PDF and has no trailing period.
#   join_wrapped_headings() joins the two lines before this regex runs.
#   Captured: the full title text to end of line.

_HEADING_WITH_PERIOD = re.compile(
    r'^([A-Z])\.\s+'
    r'([A-Z][A-Za-z\s\(\)/,\-]+?)'
    r'\.'
    r'(?:\s|$)'
)
_HEADING_NO_PERIOD = re.compile(
    r'^([A-Z])\.\s+'
    r'([A-Z][A-Za-z ,\-/\(\)]+)\s*$'
)


def _match_section_heading(line: str) -> tuple[str, str] | None:
    """Return (letter, title) if line is a section heading, else None."""
    m = _HEADING_WITH_PERIOD.match(line)
    if m:
        return m.group(1), m.group(2).strip()
    m = _HEADING_NO_PERIOD.match(line)
    if m:
        return m.group(1), m.group(2).strip()
    return None

# OCR correction table for the single scanned page (p40)
OCR_CORRECTIONS = [
    (r'\besco1i\b',         "escort"),
    (r'\bpmpose\b',         "purpose"),
    (r'\btrnstwo1ihy\b',    "trustworthy"),
    (r'\bcomi-maiiial\b',   "court-martial"),
    (r'\bmaiiial\b',        "martial"),
    (r'\blaborato1y\b',     "laboratory"),
    (r'\bpolial\b',         "portal"),
    (r'uscg\.Inil\b',       "uscg.mil"),
    (r'\badininistrative\b',"administrative"),
    (r'\bai·e\b',           "are"),
    (r'\bheai·ing\b',       "hearing"),
    (r'\bprocedmes\b',      "procedures"),
    (r'Guai·d\b',           "Guard"),
    (r'\bprimaiy\b',        "primary"),
    (r'\bofDTOs\b',         "of DTOs"),
    (r'1000\.l0B',          "1000.10B"),
    # Garbled characters from redaction artifacts
    (r'rocmn1,\.\.,\.\.\.,\.+', ""),
]

# Redaction boilerplate patterns to strip cleanly
REDACTION_RE = re.compile(
    r'(Redacted per EO[^\n]*\n?'
    r'|Removed content per Presidential Executive Order[^\n]*\n?'
    r'|No\.\s*14168[^\n]*\n?'
    r'|No\.14151[^\n]*\n?'
    r'|– black ink redaction overrides content[^\n]*\n?)',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Page extraction and cleaning
# ---------------------------------------------------------------------------

def apply_ocr_corrections(text: str) -> str:
    """Apply targeted corrections for the single scanned page (p40)."""
    for pattern, replacement in OCR_CORRECTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def clean_page_text(text: str, page_num: int) -> str:
    """
    Strip boilerplate from a page and fix known artifacts.

      1. Apply OCR corrections for the scanned page (p40).
      2. Strip redaction boilerplate text.
      3. Strip running header "COMDTINST 1000.10B".
      4. Strip internal page numbers "N-N" on their own line.
      5. Collapse excess blank lines.
    """
    if page_num == 40:
        text = apply_ocr_corrections(text)

    # Strip redaction boilerplate (keeps the surrounding policy content)
    text = REDACTION_RE.sub(" ", text)

    # Strip running header variants
    text = re.sub(r"COMDTINST\s+1000\.10B\s*\n?", "", text)

    # Strip internal page numbers e.g. "1-1", "6-12" on their own line
    text = re.sub(r"(?:^|\n)\d{1,2}-\d{1,3}(?:\n|$)", "\n", text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


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
        text = clean_page_text(raw, page_num)
        pages.append({"page_num": page_num, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def join_wrapped_headings(lines: list[str]) -> list[str]:
    """
    Join section heading lines that wrap across two lines.

    Specifically handles:
      "E. Sample Adulteration, Substitution, Dilution and Refusal to Follow Contracted"
      "Guidelines"
    where "Guidelines" is a bare continuation word that belongs to the heading.
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line_s = line.strip()
        # Detect a heading that ends mid-title: starts with "X. Title" but no period at end
        # and doesn't yet match as a complete heading.
        if (_match_section_heading(line_s) is None
                and re.match(r'^[A-Z]\.\s+[A-Z]', line_s)
                and not line_s.endswith('.')):
            # Check if next line is a short continuation (1-5 words, no period/colon)
            if (i + 1 < len(lines)):
                next_s = lines[i+1].strip()
                if (len(next_s.split()) <= 5
                        and not re.match(r'^[A-Z0-9]\.', next_s)
                        and '.' not in next_s):
                    result.append(line_s + " " + next_s)
                    i += 2
                    continue
        result.append(line)
        i += 1
    return result


def split_into_sections(pages: list[dict], chapter_num: str,
                        chapter_title: str) -> list[dict]:
    """
    Concatenate pages with position markers, split on section headings.
    Deduplicates by keeping the first occurrence of each section letter per chapter.
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

    # Find section headings after joining wrapped continuations
    lines      = combined.split("\n")
    lines      = join_wrapped_headings(lines)
    rejoined   = "\n".join(lines)

    raw_splits: list[tuple[int, str, str]] = []
    for line in lines:
        result = _match_section_heading(line.strip())
        if result:
            letter, title = result
            char_start = rejoined.find(line.strip())
            if char_start != -1:
                raw_splits.append((char_start, letter, title))

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
        char_end   = splits[idx + 1][0] if idx + 1 < len(splits) else len(rejoined)
        text       = rejoined[char_start:char_end]
        text       = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()
        page_start = char_to_page(char_start)
        page_end   = (char_to_page(splits[idx + 1][0])
                      if idx + 1 < len(splits)
                      else page_positions[-1][1])

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

def normalize_paragraphs(text: str) -> list[str]:
    """
    Split text into logical paragraphs.

    This document uses single newlines throughout. We split on structural
    markers: numbered items, lettered sub-items, and new sentences that
    start with a capital after a full stop.
    """
    PARA_START = re.compile(
        r"""
        (?:^|\n)
        (?:
            [A-Z]\.\s+[A-Z]        |  # section heading: A. Title
            \d{1,2}\.\s+[A-Z]      |  # numbered para: 1. Text
            [a-z]\.\s+[A-Z]        |  # lettered sub: a. Text
            \(\d{1,2}\)\s          |  # paren-num: (1) Text
            \([a-z]\)\s            |  # paren-letter: (a) Text
            NOTE:\s
        )
        """,
        re.VERBOSE,
    )

    # Try double-newline split first
    blocks = re.split(r"\n{2,}", text)
    paragraphs: list[str] = []
    for block in blocks:
        lines   = block.split("\n")
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


def chunk_sections(sections: list[dict], source: str,
                   doc_path: str) -> list[dict]:
    """
    Merge small sections up to CHUNK_SIZE_TOKENS; split large ones by paragraph.
    Apply overlap between consecutive chunks.
    """
    chunks:    list[dict] = []
    global_idx = 0
    buf_text   = ""
    buf_secs:  list[dict] = []

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

    first     = sections[0]
    last      = sections[-1]
    ch_num    = first["chapter_num"]
    ch_title  = first["chapter_title"]
    sec_let   = first["section_letter"]
    sec_title = first["section_title"]

    chunk_id     = f"{CHUNK_ID_PREFIX}_ch{ch_num.zfill(2)}_{idx:04d}"
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    heading_path = [f"Chapter {ch_num}", ch_title, sec_let + ".", sec_title]

    out.append({
        "chunk_id":       chunk_id,
        "content_hash":   content_hash,
        "source":         f"{SOURCE_LABEL} ({DOC_CITATION})",
        "doc_path":       doc_path,
        "section":        f"Chapter {ch_num} -- {ch_title}",
        "section_number": sec_let + ".",
        "section_title":  sec_title,
        "chapter_number": ch_num,
        "chapter_title":  ch_title,
        "page_start":     first["page_start"],
        "page_end":       last["page_end"],
        "heading_path":   heading_path,
        "chunk_index":    idx,
        "token_estimate": rough_token_count(text),
        "text":           text,
    })


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ingest(pdf_path: Path, output_path: Path) -> None:
    print(f"[INFO] Source  : {pdf_path.name}")
    print(f"[INFO] Citation: {DOC_CITATION}")
    print(f"[INFO] Content pages: 12-61  (pages 1-11 skipped)")
    print(f"[INFO] Note: p40 has OCR artifacts (scanned page) - corrections applied")
    print(f"[INFO] Note: pp38-40 have redaction blocks - boilerplate stripped")

    all_chunks: list[dict] = []

    for ch_num, (start_p, end_p, ch_title) in CHAPTERS.items():
        print(f"\n[INFO] Chapter {ch_num:>2}: {ch_title[:55]}  (pp.{start_p}-{end_p})")

        pages    = extract_pages(pdf_path, start_p, end_p)
        sections = split_into_sections(pages, ch_num, ch_title)
        print(f"       Sections found : {len(sections)}")

        chunks = chunk_sections(
            sections,
            source   = f"{SOURCE_LABEL} ({DOC_CITATION})",
            doc_path = str(pdf_path),
        )
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
        description="Ingest CG Substance Abuse and Behavioral Addiction Program "
                    "(COMDTINST 1000.10B)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/Substance_Abuse_Manual.pdf"),
        help="Path to PDF (default: data/raw/Substance_Abuse_Manual.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/substance_abuse_manual_chunks.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf, args.output)


if __name__ == "__main__":
    main()
