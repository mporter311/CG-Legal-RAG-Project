"""
ingest_mcm.py
=============
Extracts the Punitive Articles (Part IV) from the MCM PDF (2024 or 2019 edition),
chunks by article boundaries, and writes metadata-rich JSONL to
data/processed/mcm_punitive_chunks.jsonl

CHANGES FROM ORIGINAL (v2):
  - Added UCMJ_TITLES lookup table (fixes broken article_title extraction)
  - Added MCM_EDITIONS registry so 2019 and 2024 coexist cleanly
  - Overlap marker ([...]) removed from chunk text before embedding
  - rough_token_count uses word-count proxy (more accurate for legal text)
  - Chunk IDs now use local article index, not global offset, for readability
  - Part IV auto-detection updated to handle both 2019 and 2024 page layouts
  - Added --edition flag; defaults to "2024"

Usage:
    # 2024 edition (default):
    python src/ingest_mcm.py --pdf /path/to/MCM_2024_ed.pdf

    # 2019 edition (legacy):
    python src/ingest_mcm.py --pdf /path/to/2019_MCM.pdf --edition 2019

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
# UCMJ Article Title Lookup
# ---------------------------------------------------------------------------
# These are the authoritative short titles as they appear in the UCMJ statute.
# Used to override the noisy regex-extracted titles from the PDF body text.
# 2024 MCM added several sub-articles (87a, 93a, etc.) not in the 2019 edition.
# Both sets are included here; only articles present in the extracted text will
# be used.

UCMJ_TITLES: dict[str, str] = {
    # Core articles (both editions)
    "77":   "Principals",
    "78":   "Accessory After the Fact",
    "79":   "Conviction of Lesser Included Offense",
    "80":   "Attempts",
    "81":   "Conspiracy",
    "82":   "Soliciting Commission of Offenses",
    "83":   "Malingering",
    "84":   "Breach of Medical Quarantine",
    "85":   "Desertion",
    "86":   "Absence Without Leave",
    "87":   "Missing Movement; Jumping from Vessel",
    "87a":  "Resistance, Flight, Breach of Arrest, and Escape",
    "87b":  "Offenses Against Correctional Custody and Restriction",
    "88":   "Contempt Toward Officials",
    "89":   "Disrespect Toward Superior Commissioned Officer; Assault of Superior Commissioned Officer",
    "90":   "Willfully Disobeying Superior Commissioned Officer",
    "91":   "Insubordinate Conduct Toward Warrant Officer, Noncommissioned Officer, or Petty Officer",
    "92":   "Failure to Obey Order or Regulation",
    "93":   "Cruelty and Maltreatment",
    "93a":  "Prohibited Activities with Military Recruit or Trainee by Person in Position of Special Trust",
    "94":   "Mutiny or Sedition",
    "95":   "Offenses by Sentinel or Lookout",
    "95a":  "Disrespect Toward Sentinel or Lookout",
    "96":   "Release of Prisoner Without Proper Authority",
    "97":   "Unlawful Detention",
    "98":   "Misconduct as Prisoner",
    "99":   "Misbehavior Before the Enemy",
    "100":  "Subordinate Compelling Surrender",
    "101":  "Improper Use of Countersign",
    "102":  "Forcing a Safeguard",
    "103":  "Spies",
    "103a": "Espionage",
    "103b": "Aiding the Enemy",
    "104":  "Public Records Offenses",
    "104a": "Fraudulent Enlistment, Appointment, or Separation",
    "104b": "Unlawful Enlistment, Appointment, or Separation",
    "105":  "Forgery",
    "105a": "False or Unauthorized Pass Offenses",
    "106":  "Impersonation of Officer, Noncommissioned or Petty Officer, or Agent or Official",
    "106a": "Wearing Unauthorized Insignia, Medal, Decoration, Badge, Ribbon, Device, or Lapel Button",
    "107":  "False Official Statements; False Swearing",
    "107a": "Parole Violation",
    "108":  "Military Property — Loss, Damage, Destruction, Disposition",
    "108a": "Captured or Abandoned Property",
    "109":  "Property Other Than Military — Waste, Spoilage, or Destruction",
    "109a": "Mail Matter: Wrongful Taking, Opening, Etc.",
    "110":  "Improper Hazarding of Vessel or Aircraft",
    "111":  "Leaving Scene of Vehicle Accident",
    "112":  "Drunkenness and Other Incapacitation Offenses",
    "112a": "Wrongful Use, Possession, Etc. of Controlled Substances",
    "113":  "Drunken or Reckless Operation of a Vehicle, Aircraft, or Vessel",
    "114":  "Endangerment Offenses",
    "115":  "Communicating Threats",
    "116":  "Riot or Breach of Peace",
    "117":  "Provoking Speeches or Gestures",
    "117a": "Wrongful Broadcast or Publication",
    "118":  "Murder",
    "119":  "Manslaughter",
    "119a": "Death or Injury of an Unborn Child",
    "119b": "Child Abuse Offenses",
    "120":  "Rape and Sexual Assault Generally",
    "120a": "Mails: Deposit of Obscene Matter",
    "120b": "Rape and Sexual Assault of a Child",
    "120c": "Other Sexual Misconduct",
    "121":  "Larceny and Wrongful Appropriation",
    "121a": "Fraudulent Use of Credit Cards, Debit Cards, or Other Access Devices",
    "121b": "False Pretenses to Obtain Services",
    "122":  "Robbery",
    "122a": "Receiving Stolen Property",
    "123":  "Offenses Concerning Government Computers",
    "123a": "Making, Drawing, or Uttering Check, Draft, or Order Without Sufficient Funds",
    "124":  "Frauds Against the United States",
    "124a": "Bribery",
    "124b": "Graft",
    "125":  "Kidnapping",
    "126":  "Arson; Burning Property with Intent to Defraud",
    "127":  "Extortion",
    "128":  "Assault",
    "128a": "Maiming",
    "128b": "Domestic Violence",
    "129":  "Burglary; Unlawful Entry",
    "130":  "Stalking",
    "131":  "Perjury",
    "131a": "Subornation of Perjury",
    "131b": "Obstructing Justice",
    "131c": "Misprision of Serious Offense",
    "131d": "Wrongful Refusal to Testify",
    "131e": "Prevention of Authorized Seizure of Property",
    "131f": "Noncompliance with Procedural Rules",
    "131g": "Wrongful Interference with Adverse Administrative Proceeding",
    "132":  "Retaliation",
    "133":  "Conduct Unbecoming an Officer and a Gentleman",
    "134":  "General Article",
}

# ---------------------------------------------------------------------------
# Edition registry
# ---------------------------------------------------------------------------
# Stores per-edition constants so the same codebase supports both MCM versions.
# page_start_fallback: used if auto-detection fails.
# part_iv_marker: string expected on the first page of Part IV.

MCM_EDITIONS: dict[str, dict] = {
    "2024": {
        "source_label":       "MCM 2024",
        "chunk_id_prefix":    "mcm2024",
        "page_start_fallback": 311,
        "page_end_fallback":   467,
        "index_name":         "mcm_punitive",   # data/index/mcm_punitive.faiss
    },
    "2019": {
        "source_label":       "MCM 2019",
        "chunk_id_prefix":    "mcm2019",
        "page_start_fallback": 308,
        "page_end_fallback":   458,
        "index_name":         "mcm_punitive",
    },
}


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


def extract_part_iv(pdf_path: Path, edition_cfg: dict) -> tuple[list[dict], int, int]:
    """
    Auto-detect Part IV / Punitive Articles page range then extract.
    Returns (pages, start_page, end_page).

    Detection logic:
      - Scans pages 200–550 for a page whose text contains both
        "PUNITIVE ARTICLES" and "Part IV".
      - End page: last page whose text starts with "IV-NNN" before
        a "PART V" or "APPENDIX" header appears.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    start_page: Optional[int] = None
    for i in range(200, min(total, 550)):
        text = reader.pages[i].extract_text() or ""
        if "PUNITIVE ARTICLES" in text.upper() and re.search(r"Part\s+IV", text, re.IGNORECASE):
            start_page = i + 1
            break

    if start_page is None:
        start_page = edition_cfg["page_start_fallback"]
        print(f"[WARN] Could not auto-detect Part IV start; using default page {start_page}")

    end_page = start_page
    for i in range(start_page - 1, min(total, start_page + 300)):
        text = reader.pages[i].extract_text() or ""
        stripped = text.strip()
        # IV-NNN marker lines confirm we are still inside Part IV
        if re.match(r"^\s*IV-\d+", stripped):
            end_page = i + 1
        elif i > start_page + 10 and re.match(r"^\s*(PART\s+V|V-\d|APPENDIX)", stripped, re.IGNORECASE):
            break

    print(f"[INFO] Punitive Articles: PDF pages {start_page}–{end_page} ({end_page - start_page + 1} pages)")

    try:
        pages = extract_pages_pypdf(pdf_path, start_page, end_page)
    except Exception as exc:
        print(f"[WARN] pypdf failed ({exc}), trying unstructured …")
        pages = extract_pages_unstructured(pdf_path, start_page, end_page)

    return pages, start_page, end_page


# ---------------------------------------------------------------------------
# Article boundary splitting
# ---------------------------------------------------------------------------

# 2024 MCM uses numbered paragraph headings:
#   "18. Article 92 (10 U.S.C. 892) —Failure to obey order or regulation"
# 2019 MCM uses plain headings:
#   "Article 92—Failure to Obey Order or Regulation"
# Both forms are matched by the pattern below.
ARTICLE_HEADING_RE = re.compile(
    r"""
    (?:^|\n)                                    # line start
    (?:
        \d{1,3}[a-z]?\.\s+                     # optional paragraph number "18. "
    )?
    Article\s+(\d{1,3}[a-z]?)                  # "Article 92" or "Article 112a"
    (?:\s+\(10\s+U\.S\.C\.\s+\d+[a-z]?\))?    # optional USC citation
    \s*[—\-–\.]\s*                              # separator
    ([^\n]{0,120})                              # title (rest of line)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def split_by_articles(pages: list[dict]) -> list[dict]:
    """
    Concatenate all page text with page-break markers, split on article headings.
    Returns list of article dicts:
      {article_number, article_title, text, page_start, page_end}

    Deduplication logic:
      The 2024 MCM contains many cross-references like "Article 92." inside
      other articles' bodies (e.g. "See paragraph 18 (Article 92)"). The regex
      matches these as extra splits. We keep only the FIRST (and largest) segment
      per article number, which is always the canonical article definition.
      This collapses Article 134's 20 false splits down to 1, etc.

    page_end tracking:
      page_end is set to the page of the NEXT article's heading character minus 1,
      not the page of char_end of the combined string (which mapped to the last
      page of Part IV for intermediate articles). This gives correct page ranges.
    """
    PAGE_MARKER = "\n<<<PAGE:{page_num}>>>\n"
    combined = ""
    page_positions: list[tuple[int, int]] = []
    for p in pages:
        page_positions.append((len(combined), p["page_num"]))
        combined += PAGE_MARKER.format(page_num=p["page_num"])
        combined += p["text"]

    def char_to_page(char_idx: int) -> int:
        page = pages[0]["page_num"]
        for offset, pnum in page_positions:
            if offset <= char_idx:
                page = pnum
            else:
                break
        return page

    # Collect all regex matches, flagging whether each has a USC citation.
    # Headings WITH "(10 U.S.C. NNN)" are canonical article definitions.
    # Headings WITHOUT it are cross-references embedded in other articles' text.
    USC_RE = re.compile(
        r"""(?:^|\n)(?:\d{1,3}[a-z]?\.\s+)?Article\s+(\d{1,3}[a-z]?)\s+\(10\s+U\.S\.C\.\s+\d+[a-z]?\)\s*[—\-–\.]\s*([^\n]{0,120})""",
        re.IGNORECASE,
    )
    # Build two sets: canonical (USC) matches and fallback (bare) matches
    canonical_by_art: dict[str, tuple[int, str, str]] = {}   # art_num → (char_start, art_num, title)
    fallback_by_art:  dict[str, tuple[int, str, str]] = {}

    for m in ARTICLE_HEADING_RE.finditer(combined):
        art_num = m.group(1).strip()
        raw_title = (m.group(2) or "").strip().strip("—-–.")
        title = UCMJ_TITLES.get(art_num, raw_title)
        try:
            num_int = int(re.sub(r"[^0-9]", "", art_num) or "0")
        except ValueError:
            num_int = 0
        if num_int < 77:
            continue
        entry = (m.start(), art_num, title)
        has_usc = USC_RE.search(m.group()) is not None
        if has_usc and art_num not in canonical_by_art:
            canonical_by_art[art_num] = entry
        elif not has_usc and art_num not in fallback_by_art:
            fallback_by_art[art_num] = entry

    # Merge: canonical wins; fall back to bare match only if no canonical exists
    all_art_nums_seen: set[str] = set()
    all_splits_raw: list[tuple[int, str, str]] = []
    for m in ARTICLE_HEADING_RE.finditer(combined):
        art_num = m.group(1).strip()
        try:
            num_int = int(re.sub(r"[^0-9]", "", art_num) or "0")
        except ValueError:
            num_int = 0
        if num_int < 77:
            continue
        if art_num in canonical_by_art:
            all_splits_raw.append(canonical_by_art[art_num])
        elif art_num in fallback_by_art:
            all_splits_raw.append(fallback_by_art[art_num])
        # Deduplicate by art_num — only add once
    
    # Deduplicate while preserving character-position order
    seen_art: set[str] = set()
    all_splits: list[tuple[int, str, str]] = []
    for entry in sorted(all_splits_raw, key=lambda x: x[0]):
        if entry[1] not in seen_art:
            seen_art.add(entry[1])
            all_splits.append(entry)

    if not all_splits:
        return [{
            "article_number": None,
            "article_title":  "Punitive Articles",
            "text":           combined,
            "page_start":     pages[0]["page_num"],
            "page_end":       pages[-1]["page_num"],
        }]

    canonical_splits = all_splits  # already deduplicated, USC-prioritised, char-ordered
    print(f"[INFO] Resolved {len(canonical_splits)} canonical article segments "
          f"(USC-citation headings preferred over bare cross-references)")

    # Build article segments using only canonical split boundaries
    articles = []
    for idx, (char_start, art_num, title) in enumerate(canonical_splits):
        # End at the next CANONICAL article's heading, not the next raw split
        if idx + 1 < len(canonical_splits):
            char_end = canonical_splits[idx + 1][0]
        else:
            char_end = len(combined)

        text = combined[char_start:char_end]
        text = re.sub(r"<<<PAGE:\d+>>>", "", text).strip()

        page_start = char_to_page(char_start)
        # page_end: use start of next article's heading char to get the correct page,
        # not char_end - 1 (which for the last article gives the last page of Part IV)
        if idx + 1 < len(canonical_splits):
            page_end = char_to_page(canonical_splits[idx + 1][0])
        else:
            page_end = pages[-1]["page_num"]

        articles.append({
            "article_number": art_num,
            "article_title":  title,
            "text":           text,
            "page_start":     page_start,
            "page_end":       page_end,
        })

    return articles


# ---------------------------------------------------------------------------
# Token-based chunking
# ---------------------------------------------------------------------------

def rough_token_count(text: str) -> int:
    """
    Estimate token count using word count (more accurate for legal text than
    character division). Legal prose averages ~1.3 tokens per word.
    """
    words = len(text.split())
    return max(1, int(words * 1.3))


def normalize_paragraphs(text: str) -> list[str]:
    """
    Convert raw pypdf text into a clean list of logical paragraphs.

    The 2024 MCM is a two-column layout. pypdf extracts it as a stream of
    short lines separated ONLY by single newlines — double newlines are nearly
    absent. A naïve re.split(r"\\n{2,}") therefore produces 1-3 giant blocks
    instead of proper paragraphs.

    Strategy (handles both 2019 and 2024 layout):
    1. Collapse the line stream into logical paragraphs using structural cues:
       - Lines starting with a letter+period pattern: (a), (b), b., c. etc.
       - Lines starting with a digit+period: (1), (2), 1., 2.
       - Lines starting with legal section labels: "Text of statute", "Elements",
         "Explanation", "Maximum punishment", "Sample specifications"
       - Double newlines (when they exist, as in 2019)
    2. Merge continuation lines (lines that don't start a new section) into the
       preceding paragraph.
    3. Return the resulting paragraph list.
    """
    # First: honour any existing double-newline paragraph breaks
    # (these work for 2019 and for any section that has them)
    blocks = re.split(r"\n{2,}", text)

    # For each block, further split on legal structure markers
    # These are line-starting patterns that reliably indicate a new logical unit
    SECTION_START = re.compile(
        r"""
        (?:^|\n)                    # start of line
        (?:
            [a-z]\.\s              |  # "b. Elements"
            \([a-z]\)\s            |  # "(a) That the accused..."
            \(\d{1,2}\)\s          |  # "(1) That..."
            \d{1,2}\.\s            |  # "1. ..."  "18. Article..."
            \[Note:                |  # "[Note: ...]"
            [A-Z][a-z]+\s+of\s+statute  |  # "Text of statute"
            Elements?\.?           |  # "Elements." / "Elements"
            Explanation\.?         |  # "Explanation."
            Maximum\s+punishment   |  # "Maximum punishment"
            Sample\s+spec          |  # "Sample specification(s)"
            Discussion             |  # "Discussion"
            [A-Z]{2,}              # ALL-CAPS headers (rare)
        )
        """,
        re.VERBOSE,
    )

    paragraphs: list[str] = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) <= 2:
            # Short block — keep as-is
            p = block.strip()
            if p:
                paragraphs.append(p)
            continue

        # Split block into sub-paragraphs on legal structure markers
        current_lines: list[str] = []
        for line in lines:
            # Check if this line starts a new legal section
            if current_lines and SECTION_START.match(line):
                p = "\n".join(current_lines).strip()
                if p:
                    paragraphs.append(p)
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            p = "\n".join(current_lines).strip()
            if p:
                paragraphs.append(p)

    return [p for p in paragraphs if p.strip()]


def chunk_article(
    article: dict,
    chunk_size_tokens: int = 600,
    overlap_tokens: int = 100,
    source: str = "MCM 2024",
    doc_path: str = "",
    chunk_id_prefix: str = "mcm2024",
) -> list[dict]:
    """
    Split a single article's text into overlapping token-sized chunks.

    Changes from v2:
    - Uses normalize_paragraphs() instead of re.split(r"\\n{2,}") to handle
      the 2024 MCM's two-column single-newline-only PDF extraction format.
      This is the root cause of the "2 chunks for Article 92" bug — the old
      splitter produced only 3 giant paragraphs, causing the sentence splitter
      to produce oversized chunks that were then aggregated incorrectly.
    - Overlap text appended WITHOUT [...]  marker (no embedding noise).
    - Chunk IDs use LOCAL article index for readability.
    - Near-empty chunks (< 20 tokens) are silently dropped.
    """
    text = article["text"]
    if not text.strip():
        return []

    paragraphs = normalize_paragraphs(text)

    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n" + para).strip() if current else para.strip()
        if rough_token_count(candidate) <= chunk_size_tokens:
            current = candidate
        else:
            if current:
                raw_chunks.append(current)
            # Oversized paragraph — split by sentence boundaries
            if rough_token_count(para) > chunk_size_tokens:
                sentences = re.split(r"(?<=[.;])\s+", para)
                sub = ""
                for sent in sentences:
                    candidate_sub = (sub + " " + sent).strip() if sub else sent.strip()
                    if rough_token_count(candidate_sub) <= chunk_size_tokens:
                        sub = candidate_sub
                    else:
                        if sub:
                            raw_chunks.append(sub)
                        sub = sent.strip()
                current = sub if sub else ""
            else:
                current = para.strip()

    if current:
        raw_chunks.append(current)

    art_num   = article.get("article_number") or "unknown"
    art_title = article.get("article_title") or UCMJ_TITLES.get(art_num, "")

    # Apply overlap: prepend tail of previous chunk without any marker
    overlap_chars = overlap_tokens * 4
    result: list[dict] = []
    local_idx = 0

    for i, chunk_text in enumerate(raw_chunks):
        if i > 0 and overlap_chars > 0:
            prev = raw_chunks[i - 1]
            overlap_prefix = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            chunk_text = overlap_prefix + "\n\n" + chunk_text

        chunk_text = chunk_text.strip()

        if rough_token_count(chunk_text) < 20:
            continue

        chunk_id     = f"{chunk_id_prefix}_{art_num.zfill(3)}_{local_idx:03d}"
        content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]

        result.append({
            "chunk_id":                chunk_id,
            "content_hash":            content_hash,
            "source":                  source,
            "doc_path":                doc_path,
            "section":                 "Punitive Articles",
            "article_number":          art_num,
            "article_title":           art_title,
            "page_start":              article["page_start"],
            "page_end":                article["page_end"],
            "heading_path":            ["Part IV", "Punitive Articles", f"Article {art_num}"],
            "chunk_index":             local_idx,
            "total_chunks_in_article": 0,   # patched below
            "token_estimate":          rough_token_count(chunk_text),
            "text":                    chunk_text,
        })
        local_idx += 1

    for r in result:
        r["total_chunks_in_article"] = len(result)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest MCM Punitive Articles (2024 or 2019 edition)")
    parser.add_argument(
        "--pdf", type=Path,
        default=Path("/mnt/user-data/uploads/MCM__2024_ed_.pdf"),
        help="Path to MCM PDF",
    )
    parser.add_argument(
        "--edition", type=str, default="2024", choices=["2024", "2019"],
        help="MCM edition (affects source labels, chunk ID prefix, fallback pages)",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/processed/mcm_punitive_chunks.jsonl"),
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap",    type=int, default=100)
    args = parser.parse_args()

    edition_cfg = MCM_EDITIONS[args.edition]
    source      = edition_cfg["source_label"]
    id_prefix   = edition_cfg["chunk_id_prefix"]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Edition : {source}")
    print(f"[INFO] PDF     : {args.pdf}")
    pages, start_pg, end_pg = extract_part_iv(args.pdf, edition_cfg)
    print(f"[INFO] Extracted {len(pages)} pages")

    print("[INFO] Splitting by article boundaries …")
    articles = split_by_articles(pages)
    print(f"[INFO] Found {len(articles)} article segments")

    all_chunks: list[dict] = []
    for article in articles:
        chunks = chunk_article(
            article,
            chunk_size_tokens=args.chunk_size,
            overlap_tokens=args.overlap,
            source=source,
            doc_path=str(args.pdf),
            chunk_id_prefix=id_prefix,
        )
        all_chunks.extend(chunks)

    print(f"[INFO] Total chunks: {len(all_chunks)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[OK] Chunks written to {args.out}")

    arts = {c["article_number"] for c in all_chunks if c["article_number"]}
    art_list = sorted(arts, key=lambda x: (int(re.sub(r"[^0-9]", "", x) or "0"), x))
    print(f"[STATS] Articles indexed: {art_list[:12]} … ({len(arts)} total)")
    tokens = [c["token_estimate"] for c in all_chunks]
    print(
        f"[STATS] Token estimates — "
        f"min:{min(tokens)}  avg:{sum(tokens)//len(tokens)}  max:{max(tokens)}"
    )


if __name__ == "__main__":
    main()
