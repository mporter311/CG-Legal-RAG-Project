"""
ingest_mjm.py  --  Ingest the Coast Guard Military Justice Manual
                    (COMDTINST M5810.1H)

Document structure
------------------
- 197 pages total
- Pages  1-12: cover, instruction header (pp3-5), TOC (pp7-11), blanks -> skipped
- Pages 13-197: 28 chapters of content

Chapter layout (pp.start - pp.end):
  Ch. 1  pp. 13- 16   Nonpunitive Measures                          (~1,532 words)
  Ch. 2  pp. 17- 48   Nonjudicial Punishment (NJP)                 (~20,184 words)
  Ch. 3  pp. 49- 58   Summary Courts-Martial (SCM)                  (~4,926 words)
  Ch. 4  pp. 59- 64   Jurisdiction                                  (~2,225 words)
  Ch. 5  pp. 65- 68   Court-Martial Convening Authorities           (~1,419 words)
  Ch. 6  pp. 69- 74   Pretrial Restraint and Confinement            (~2,359 words)
  Ch. 7  pp. 75- 78   Pre-Referral Matters                          (~1,446 words)
  Ch. 8  pp. 79- 80   Evidentiary Matters                            (~707 words)
  Ch. 9  pp. 81- 82   Classified Matters                             (~696 words)
  Ch.10  pp. 83- 84   Preliminary Hearing under Article 32           (~1,024 words)
  Ch.11  pp. 85- 86   Referral of Charges                           (~1,152 words)
  Ch.12  pp. 87- 90   Preparation, Forwarding, Changes to Charges   (~1,950 words)
  Ch.13  pp. 91- 98   Courts-Martial Personnel                       (~3,564 words)
  Ch.14  pp. 99-100   Courtroom                                        (~414 words)
  Ch.15  pp.101-104   Authority to Grant Immunity                    (~1,333 words)
  Ch.16  pp.105-116   Information and Services to Victims/Witnesses  (~5,705 words)
  Ch.17  pp.117-118   Witness Reimbursement                          (~1,083 words)
  Ch.18  pp.119-124   Oaths, Article 39(a), Release of Information  (~2,723 words)
  Ch.19  pp.125-126   Government Appeals from Adverse Rulings        (~1,047 words)
  Ch.20  pp.127-132   Sentencing Matters                             (~2,466 words)
  Ch.21  pp.133-142   Post-Trial Processing                          (~4,524 words)
  Ch.22  pp.143-150   Review of Courts-Martial                       (~3,418 words)
  Ch.23  pp.151-174   Article 140a UCMJ Implementation               (~7,458 words)
  Ch.24  pp.175-176   Military Justice Practitioners                   (~727 words)
  Ch.25  pp.177-182   Article 138, UCMJ - Complaints                 (~3,025 words)
  Ch.26  pp.183-188   Certification and Designation of Judges        (~2,815 words)
  Ch.27  pp.189-194   Search Authorizations                          (~2,993 words)
  Ch.28  pp.195-197   Delivery of Personnel to Civil Authorities     (~1,246 words)

Section heading format
----------------------
  "Section A. Title of the Section"
  Appears once per section, always on its own line.
  Letters run A-Z; Ch.2 NJP uses all 26 letters.

Sub-section format (inline, NOT used as chunk boundaries)
---------------------------------------------------------
  "A.1. Label  Body text continues on the same line..."
  "A.1.a. Sub-label  Text..."
  These are preserved within chunk text for context but do NOT trigger splits.

PDF extraction characteristics
-------------------------------
  - CLEAN single-column layout. ~70-80 char lines, single newlines.
  - Zero mid-word split-word problem (only ~3 isolated occurrences in full document).
  - Running header "COMDTINST M5810.1H" on most content pages -> stripped.
  - Chapter title appears as a large heading on first page of each chapter
    (before or after the running header) -> stripped after section splitting.
  - Internal page numbers "N-NN" on their own line -> stripped.
  - Blank divider pages scattered throughout (pp6,12,16,58,64,68,74,78,98,100,
    104,116,124,132,142,150,188) -> automatically skipped (zero word count).

Chunking strategy
-----------------
  - Split on "Section X. Title" as primary boundaries within each chapter.
  - Each section becomes the base unit for chunking.
  - Sections ≤ CHUNK_SIZE_TOKENS are merged with adjacent sections.
  - Sections > CHUNK_SIZE_TOKENS are split by paragraph/sub-section.
  - Overlap applied between consecutive chunks.
  - Heading path: ["Chapter N", "Chapter Title", "Section X", "Section Title"]

Output
------
  data/processed/mjm_chunks.jsonl
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

SOURCE_LABEL    = "Coast Guard Military Justice Manual"
DOC_CITATION    = "COMDTINST M5810.1H"
CHUNK_ID_PREFIX = "cgmjm"

# Pages 1-12 are front matter (cover, header, TOC, blanks)
SKIP_PAGES_1INDEXED = set(range(1, 13))

# Blank divider pages throughout the document
BLANK_PAGES_1INDEXED = {6, 12, 16, 58, 64, 68, 74, 78, 98, 100,
                        104, 116, 124, 132, 142, 150, 188}

CHUNK_SIZE_TOKENS = 600
OVERLAP_TOKENS    = 80

CHAPTERS: dict[str, tuple[int, int, str]] = {
    "1":  (13,  16,  "Nonpunitive Measures"),
    "2":  (17,  48,  "Nonjudicial Punishment (NJP)"),
    "3":  (49,  58,  "Summary Courts-Martial (SCM)"),
    "4":  (59,  64,  "Jurisdiction"),
    "5":  (65,  68,  "Court-Martial Convening Authorities"),
    "6":  (69,  74,  "Pretrial Restraint and Confinement"),
    "7":  (75,  78,  "Pre-Referral Matters"),
    "8":  (79,  80,  "Evidentiary Matters"),
    "9":  (81,  82,  "Classified Matters"),
    "10": (83,  84,  "Preliminary Hearing under Article 32, UCMJ"),
    "11": (85,  86,  "Referral of Charges to Special or General Courts-Martial"),
    "12": (87,  90,  "Preparation, Forwarding, Changes to, and Withdrawal of Charges"),
    "13": (91,  98,  "Courts-Martial Personnel"),
    "14": (99,  100, "Courtroom"),
    "15": (101, 104, "Authority to Grant Immunity from Prosecution"),
    "16": (105, 116, "Information and Services to be Provided to Victims and Witnesses"),
    "17": (117, 118, "Witness Reimbursement"),
    "18": (119, 124, "Oaths, Article 39(a) Sessions, and Release of Information"),
    "19": (125, 126, "Government Appeals from Adverse Rulings"),
    "20": (127, 132, "Sentencing Matters"),
    "21": (133, 142, "Post-Trial Processing in General and Special Courts-Martial"),
    "22": (143, 150, "Review of Courts-Martial"),
    "23": (151, 174, "Article 140a, UCMJ Implementation"),
    "24": (175, 176, "Military Justice Practitioners"),
    "25": (177, 182, "Article 138, UCMJ - Complaints"),
    "26": (183, 188, "Certification and Designation of Judges"),
    "27": (189, 194, "Search Authorizations"),
    "28": (195, 197, "Delivery of Personnel to Civil Authorities"),
}

# "Section A. Title of the Section" - whole line
SECTION_RE = re.compile(
    r'(?:^|\n)'
    r'Section\s+([A-Z])\.\s+'
    r'([^\n]+)'
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Page cleaning
# ---------------------------------------------------------------------------

def clean_page_text(text: str) -> str:
    """
    Strip boilerplate from a single page.

      1. Strip running header "COMDTINST M5810.1H"
      2. Strip internal page numbers "N-NN" on their own line
      3. Strip chapter title lines that appear as standalone headings
         (chapter title appears at top of first content page, not needed
          since chapter metadata is in the heading_path field)
      4. Collapse excess blank lines
    """
    text = re.sub(r"COMDTINST\s+M5810\.\d+[A-Z]?\s*\n?", "", text)
    text = re.sub(r"(?:^|\n)\d{1,2}-\d{1,3}\s*(?:\n|$)", "\n",
                  text, flags=re.MULTILINE)
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
        if page_num in SKIP_PAGES_1INDEXED | BLANK_PAGES_1INDEXED:
            continue
        raw  = reader.pages[i].extract_text() or ""
        if not raw.strip():
            continue
        text = clean_page_text(raw)
        if text:
            pages.append({"page_num": page_num, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def split_into_sections(pages: list[dict], chapter_num: str,
                        chapter_title: str) -> list[dict]:
    """
    Concatenate all pages with position markers, then split on
    "Section X. Title" boundaries.

    Returns a list of section dicts, each with:
      section_letter, section_title, chapter_num, chapter_title,
      text, page_start, page_end
    """
    PAGE_MARKER = "\n<<<PAGE:{pnum}>>>\n"
    combined    = ""
    page_positions: list[tuple[int, int]] = []

    for p in pages:
        page_positions.append((len(combined), p["page_num"]))
        combined += PAGE_MARKER.format(pnum=p["page_num"])
        combined += p["text"]

    def char_to_page(idx: int) -> int:
        """Map character offset to PDF page number."""
        page = page_positions[0][1] if page_positions else 0
        for offset, pnum in page_positions:
            if offset <= idx:
                page = pnum
            else:
                break
        return page

    # Find section boundaries
    raw_splits: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for m in SECTION_RE.finditer(combined):
        letter = m.group(1)
        title  = m.group(2).strip().rstrip(".")
        if letter not in seen:
            seen.add(letter)
            raw_splits.append((m.start(), letter, title))

    if not raw_splits:
        # No sections found — treat entire chapter as one block
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

    # Any text before the first section heading (chapter title + intro)
    # is prepended to Section A so nothing is discarded.
    preamble = re.sub(r"<<<PAGE:\d+>>>", "", combined[:raw_splits[0][0]]).strip()

    sections = []
    for idx, (char_start, letter, title) in enumerate(raw_splits):
        char_end   = raw_splits[idx + 1][0] if idx + 1 < len(raw_splits) else len(combined)
        text       = combined[char_start:char_end]
        text       = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()

        # Attach the chapter preamble (chapter title + any pre-section intro) to
        # the first section only, so none of the chapter content is dropped.
        if idx == 0 and preamble:
            text = preamble + "\n\n" + text

        page_start = page_positions[0][1] if idx == 0 else char_to_page(char_start)
        page_end   = (char_to_page(raw_splits[idx + 1][0])
                      if idx + 1 < len(raw_splits)
                      else page_positions[-1][1])

        sections.append({
            "section_letter": letter,
            "section_title":  title[:120],
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
    Split a section's text into logical paragraphs for large-section splitting.

    The MJM uses inline sub-section labels "A.1. Label  Text..." which act as
    natural paragraph breaks. We split on these plus explicit blank lines.

    As a last resort, any remaining paragraph block that exceeds CHUNK_SIZE_TOKENS
    is further split at sentence boundaries to prevent runaway chunk sizes.
    """
    SUBSEC = re.compile(
        r"""
        (?:^|\n)
        (?:
            [A-Z]\.\d+(?:\.[a-z](?:\.[ivx]+)?)?\.\s  |  # A.1.  A.1.a.  A.1.a.i.
            [a-z]\.\s                                     |  # a. item
            \([ivx]+\)\s                                 |  # (i) item
            \(\d+\)\s                                   |  # (1) item
            \([a-z]\)\s                                  |  # (a) item
            Section\s[A-Z]\.                                 # Section A.
        )
        """,
        re.VERBOSE,
    )
    SENTENCE_SPLIT = re.compile(r'(?<=[.!?])  ?(?=[A-Z])')

    # Primary: split on double-newlines and sub-section labels
    blocks = re.split(r"\n{2,}", text)
    paragraphs: list[str] = []
    for block in blocks:
        lines   = block.split("\n")
        current: list[str] = []
        for line in lines:
            if current and SUBSEC.match(line):
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

    # Fallback: sentence-split any paragraph that is still oversized
    result: list[str] = []
    for para in paragraphs:
        if rough_token_count(para) > CHUNK_SIZE_TOKENS:
            sentences = SENTENCE_SPLIT.split(para)
            current   = ""
            for sent in sentences:
                cand = (current + "  " + sent).strip() if current else sent
                if rough_token_count(cand) <= CHUNK_SIZE_TOKENS:
                    current = cand
                else:
                    if current:
                        result.append(current)
                    current = sent
            if current:
                result.append(current)
        else:
            result.append(para)

    return [p for p in result if p.strip()]

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

    first    = sections[0]
    last     = sections[-1]
    ch_num   = first["chapter_num"]
    ch_title = first["chapter_title"]
    sec_let  = first["section_letter"]
    sec_title = first["section_title"]

    chunk_id     = f"{CHUNK_ID_PREFIX}_ch{ch_num.zfill(2)}_{idx:04d}"
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    heading_path = [f"Chapter {ch_num}", ch_title,
                    f"Section {sec_let}", sec_title]

    out.append({
        "chunk_id":       chunk_id,
        "content_hash":   content_hash,
        "source":         f"{SOURCE_LABEL} ({DOC_CITATION})",
        "doc_path":       doc_path,
        "section":        f"Chapter {ch_num} -- {ch_title}",
        "section_number": f"Section {sec_let}",
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
    print(f"[INFO] Content pages: 13-197  (pages 1-12 skipped)")
    print(f"[INFO] Blank divider pages skipped automatically")

    all_chunks: list[dict] = []

    for ch_num, (start_p, end_p, ch_title) in CHAPTERS.items():
        pages    = extract_pages(pdf_path, start_p, end_p)
        sections = split_into_sections(pages, ch_num, ch_title)
        chunks   = chunk_sections(
            sections,
            source   = f"{SOURCE_LABEL} ({DOC_CITATION})",
            doc_path = str(pdf_path),
        )

        print(f"\n[INFO] Chapter {ch_num:>2}: {ch_title[:50]}  (pp.{start_p}-{end_p})")
        print(f"       Sections found : {len(sections)}")
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

    # Sample one chunk per chapter
    print("\n[SAMPLE CHUNKS]")
    by_ch: dict[str, list[dict]] = {}
    for c in all_chunks:
        by_ch.setdefault(c["chapter_number"], []).append(c)

    for ch_num in sorted(by_ch.keys(), key=int):
        clist  = by_ch[ch_num]
        sample = clist[len(clist) // 2]
        print(f"\n  Ch.{ch_num}  {sample['chunk_id']}"
              f"  pp.{sample['page_start']}-{sample['page_end']}"
              f"  {sample['token_estimate']} tok")
        print(f"  Path: {' > '.join(sample['heading_path'])}")
        print(f"  Text: {sample['text'][:110].replace(chr(10), ' ')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the CG Military Justice Manual (COMDTINST M5810.1H)"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/MJM.pdf"),
        help="Path to PDF (default: data/raw/MJM.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mjm_chunks.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"[ERROR] PDF not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf, args.output)


if __name__ == "__main__":
    main()
