# pio-rag -- Coast Guard Legal Information RAG System

A retrieval-augmented generation (RAG) system for answering questions about
U.S. Coast Guard and military law. Retrieves from official source documents,
generates plain-language summaries with citations, and is explicitly **not**
legal advice.

---

## Current Status (Phase 1 -- Corpus Ingestion)

Three source documents are ingested and ready to index:

| Source | Citation | Chunks | Coverage |
|---|---|---|---|
| MCM 2024 Punitive Articles | Part IV, Arts. 77-134 | ~292 | UCMJ offenses, elements, max punishments |
| CG Conduct Manual | COMDTINST M1600.2 | 158 | Discipline, fraternization, hazing, MPOs |
| CG Separations Manual | COMDTINST 1000.4C | 218 | Officer/enlisted separation, retirement |

**Total indexed chunks: ~668** across all three sources.

---

## Repo Structure

```
pio-rag/
|-- data/
|   |-- raw/                          # Place source PDFs here
|   |   |-- 2024_MCM.pdf
|   |   |-- CG_Conduct_Manual.pdf
|   |   \-- Separations_Manual.pdf
|   |-- processed/                    # Generated JSONL chunk files
|   |   |-- mcm_punitive_chunks.jsonl
|   |   |-- conduct_manual_chunks.jsonl
|   |   \-- separations_manual_chunks.jsonl
|   \-- index/                        # Generated FAISS index
|       |-- pio_rag.faiss
|       \-- pio_rag_meta.json
|-- src/
|   |-- ingest_mcm.py                 # MCM 2024 Part IV ingestion
|   |-- ingest_conduct_manual.py      # CG Conduct Manual ingestion
|   |-- ingest_separations_manual.py  # CG Separations Manual ingestion
|   |-- build_index.py                # Embed all chunks + build FAISS index
|   |-- query.py                      # Retrieve + synthesise answers
|   |-- eval_retrieval.py             # Benchmark evaluation metrics
|   \-- prompt_template.py            # Canonical LLM system prompt
|-- eval/
|   \-- benchmark.json                # 10-question MCM gold benchmark
|-- README.md
\-- requirements.txt
```

---

## Pipeline

```
PDFs (MCM, Conduct Manual, Separations Manual)
    |
    v  ingest_*.py (one per document)
Per-document JSONL chunk files in data/processed/
    | Section-aware chunking (600 tok, 80-100 overlap)
    | Metadata: source, section/article, heading path, pages, chunk_id
    |
    v  build_index.py
Globs all *.jsonl in data/processed/ -> single combined index
    | sentence-transformers (all-MiniLM-L6-v2, 384-dim)
    | L2-normalised vectors -> FAISS IndexFlatIP (exact cosine)
    |
    v  data/index/pio_rag.faiss + pio_rag_meta.json
    |
    v  query.py
Dense retrieval (top-8) -> optional article/section filter -> ranked chunks
    |
    |-- Retrieval-only mode: structured excerpts + citations
    \-- LLM mode (--llm): passages -> Mistral/LLaMA GGUF -> grounded summary
    |
    v  eval_retrieval.py
Recall@k | Precision@k | F1@k | MRR
```

---

## Quick Start

### 0. Setup
```bash
conda create -n pio-rag python=3.12 -y
conda activate pio-rag
pip install pypdf sentence-transformers faiss-cpu numpy
# Optional -- for local LLM generation:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 1. Ingest documents

Run each ingestion script for documents you have in `data/raw/`:

```bash
# MCM 2024 Punitive Articles
python src/ingest_mcm.py --pdf data/raw/2024_MCM.pdf --edition 2024

# CG Conduct Manual (COMDTINST M1600.2)
python src/ingest_conduct_manual.py --pdf data/raw/CG_Conduct_Manual.pdf

# CG Separations Manual (COMDTINST 1000.4C)
python src/ingest_separations_manual.py --pdf data/raw/Separations_Manual.pdf
```

Each script writes a JSONL file to `data/processed/`. Expected chunk counts:
- MCM 2024: ~292 chunks
- Conduct Manual: ~158 chunks
- Separations Manual: ~218 chunks

### 2. Build the combined index

```bash
# Index ALL .jsonl files in data/processed/ (default -- recommended):
python src/build_index.py

# Index only specific files:
python src/build_index.py --chunks-files data/processed/mcm_punitive_chunks.jsonl

# Custom index name:
python src/build_index.py --index-name my_index
```

The index is written to `data/index/pio_rag.faiss` and `pio_rag_meta.json`.

### 3. Query

```bash
# Basic query (retrieval-only, no LLM needed):
python src/query.py "What does Article 92 cover?"

# Query with article filter (auto-detected from query text):
python src/query.py "What are the elements of Article 86?"

# Explicit article filter:
python src/query.py "maximum punishment for larceny" --article 121

# Query about manual content:
python src/query.py "What is the Coast Guard policy on fraternization?"
python src/query.py "What constitutes hazing under USCG policy?"
python src/query.py "What are the grounds for misconduct discharge?"

# With local LLM (generates plain-language summary):
python src/query.py "What does Article 92 cover?"

# The LLM_MODEL env variable avoids typing the path every time:
conda env config vars set LLM_MODEL="C:\path\to\mistral-7b-instruct.Q4_K_M.gguf"
# Then restart the environment and run without --llm flag
```

### 4. Evaluate retrieval

```bash
# Basic evaluation:
python src/eval_retrieval.py --benchmark eval/benchmark.json

# With per-query breakdown:
python src/eval_retrieval.py --benchmark eval/benchmark.json --ks 1 3 5 8 --verbose

# Save results:
python src/eval_retrieval.py --benchmark eval/benchmark.json --out eval/results/run1.json
```

---

## Expected Output Format

```
Query: What does Article 92 cover?

--- [1] Article 92 -- Failure to Obey Order or Regulation ---
b. Elements.
    (1) Violation of or failure to obey a lawful general order or regulation.
        (a) That there was in effect a certain lawful general order or regulation;
        (b) That the accused had a duty to obey it; and
        (c) That the accused violated or failed to obey the order or regulation.
    (2) Failure to obey other lawful order.
    (3) Dereliction in the performance of duties.
        ...

## Citations
[1] [MCM 2024 | Punitive Articles | Art. 92 -- Failure to Obey Order or Regulation |
    pp.337-339 | chunks: mcm2024_092_000, mcm2024_092_001 | score: 0.698]

!  I am not a lawyer. This is an informational summary of official materials only.
    Consult your chain of command or legal office for advice.
```

With `--llm`, the output adds five structured sections:
`## Plain-language summary` / `## Key elements` / `## Maximum punishment` /
`## Where this comes from (citations)` / `## Disclaimer`

---

## Baseline Evaluation (MCM-only 10-question benchmark)

Measured on the combined MCM 2024 index (`all-MiniLM-L6-v2`):

```
MRR        : 0.75
Recall@1   : 0.70    Precision@1 : 0.70
Recall@3   : 0.80    Precision@3 : 0.50
Recall@5   : 0.80    Precision@5 : 0.44
Recall@8   : 0.80    Precision@8 : 0.31
```

**Known failure modes:**
- `q001` / `q003`: "What does Article 92 cover" and "larceny punishment" miss
  without an explicit article number -- semantic gap between query terms and
  article text in `all-MiniLM-L6-v2`.
- `q004`: Article 128 (Assault) ranks behind 128b (Domestic Violence) without
  a filter -- sub-articles pollute unfiltered results.

**Note on precision:** Gold labels are currently article-level only. True
chunk-level precision is unknown until `gold.chunk_ids` are populated in
`benchmark.json` (requires one index run + inspection).

---

## Chunk Metadata Schema

All chunk types share these common fields:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | Unique ID: `mcm2024_092_000`, `cgcm_ch01_0004`, `cgsep_ch02_0015` |
| `source` | str | Human-readable source label with citation |
| `heading_path` | list[str] | Breadcrumb: `["Chapter 1", "Discipline", "1.F.3."]` |
| `page_start` | int | PDF page number (1-indexed) |
| `page_end` | int | PDF page number (1-indexed) |
| `token_estimate` | int | Approx token count (word_count x 1.3) |
| `chunk_index` | int | Position within parent section (for ordering) |
| `text` | str | Chunk content |

MCM-specific fields: `article_number`, `article_title`, `total_chunks_in_article`

Manual-specific fields: `section_number`, `section_title`, `chapter_number`, `chapter_title`

---

## Embedding Model Options

| Model | Dims | Size | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` <- **current** | 384 | 80 MB | Good general |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | +~10% retrieval quality |
| `BAAI/bge-base-en-v1.5` | 768 | 430 MB | Best quality, 3x slower |

Swap with `--model BAAI/bge-small-en-v1.5` in `build_index.py` and `query.py`.
**You must rebuild the index after changing models.**

---

## Known Limitations

1. **Embedding model** -- `all-MiniLM-L6-v2` is general-domain. Legal
   terminology mappings ("larceny" -> Article 121) are weak without explicit
   article numbers in the query. Upgrade to `BAAI/bge-small-en-v1.5` is planned.

2. **Article-level gold labels** -- The benchmark evaluates at the article level,
   not the chunk level. Precision@k is inflated relative to true answer quality.

3. **Single-document queries** -- The article filter only works for MCM article
   numbers. Manual section filters (e.g. `--section 1.F.3.`) are not yet
   implemented.

4. **No BM25** -- Purely dense retrieval. Exact legal terms ("dereliction",
   "general court-martial") may be better served by a hybrid BM25 + dense approach.

5. **Separations Manual minor artifacts** -- Three instances of run-together
   words (`CountrySection`, `MattersCommanding`, `anarrative`) remain in body
   text due to PDF extraction edge cases. These affect ~3 of 218 chunks and do
   not impact heading paths or citations.

6. **LLM synthesis** -- Maximum punishment answers are incomplete when the
   punishment table is split across chunks not retrieved in the top-k.

---

## Next Steps (Phase 2)

**Do next (in order):**
1. Run all three ingest scripts and rebuild the combined index
2. Swap embedding model to `BAAI/bge-small-en-v1.5` and re-evaluate
3. Receive and integrate the 200-question CG legal Q&A benchmark
4. Use benchmark to evaluate retrieval quality and identify failure patterns
5. Implement BM25 hybrid retrieval to address exact-term lookup failures

**Do NOT yet:**
- Fine-tune the embedding model (need 10k+ labeled pairs, not 200)
- Add a web UI or API wrapper (retrieval quality not yet sufficient)
- Ingest additional documents until Phase 2 evaluation is complete

**Remaining Phase 1 documents (lower priority):**
- CG Military Justice Manual (COMDTINST M5810.1)
- CG Investigations Manual
- CG Substance Abuse Manual

---

## Requirements

```
pypdf>=4.0
sentence-transformers>=2.7
faiss-cpu>=1.8
numpy>=1.26
llama-cpp-python        # optional, for local LLM generation
mlflow>=2.12            # optional, for experiment tracking
```
